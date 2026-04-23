"""Optimized dense single-volume EM engine (v2) with half-spectrum GEMMs.

Key optimizations over engine_fused.py:
1. Two-pass blockwise posterior normalization -- no full (batch, n_rot, n_trans) tensor
2. JIT-compiled per-block E-step and M-step kernels -- eliminates Python dispatch overhead
3. E-step scores computed twice (pass1: logsumexp stats, pass2: normalize+accumulate M-step)
   This trades 2x E-step compute for eliminating the giant residual tensor and
   enabling much larger rotation block sizes.
4. Half-spectrum GEMMs: operate on N_half = H * (W//2+1) instead of N = H*W,
   giving ~2x speedup on all GEMMs (Phase 1 of RELION-parity plan).
5. Coordinate-preserving Fourier windowing: when current_size < full resolution,
   GEMMs operate on only the low-frequency subset of the half-spectrum (N_windowed
   instead of N_half).  This gives ~15x fewer FLOPs at current_size=32.
   (Phase 3 of RELION-parity plan.)

Translation handling (see docs/math/translation_handling_analysis.md):
   Both E-step and M-step use GEMM with explicit shifted-image copies.
   The n_trans factor inflates the GEMM matrices but enables 200x better
   data reuse vs the FFT alternative (1.5 GB vs 327 GB memory traffic).
   GEMM: 45 ms at 47 TFLOPS.  FFT: 1500 ms at 0.7 TFLOPS.  Same result.
   FFT wins only for single-rotation refinement (2x faster per rotation).

Half-spectrum inner product identity (for real-valued images):
   Re<a, b>_full = Re[sum_half w(k) * conj(a(k)) * b(k)]
   where w(k) = 1 for DC and Nyquist columns, w(k) = 2 for interior columns.
   The weights are absorbed into projections (precomputed once per rotation block)
   to avoid extra elementwise multiplies in the hot GEMM loops.

Fourier windowing:
   At low current_size, only frequencies with radius <= current_size//2 are
   included in the GEMMs.  The window is applied as a gather from the full
   half-spectrum after phase shifting (shift-then-gather), preserving correct
   physical frequency spacing.  For the M-step, the windowed GEMM result is
   scattered back to a full half-spectrum before adjoint_slice_volume.
"""

import logging
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core
from recovar.core.configs import ForwardModelConfig

from .helpers.types import EMProfileStats, NoiseStats, RelionStats

logger = logging.getLogger(__name__)

# TRACKED TODOs: DENSE_ENGINE_BOUNDARY
# TODO(DENSE_ENGINE_BOUNDARY/E001): this file is dense/global-only, local search belongs in local_em_engine.py
# TODO(DENSE_ENGINE_BOUNDARY/E002): extract shared primitives, do not let local logic grow back here
# TODO(DENSE_ENGINE_BOUNDARY/E003): half/full spectrum conversions need a single explicit boundary
# TODO(DENSE_ENGINE_BOUNDARY/E004): dtype-policy cleanup needed, reduce ad hoc casts and flags
# See docs/relion_local_engine_refactor.md


def _block_until_ready(*values):
    """Synchronize one or more JAX values before host-side timing reads."""
    for value in values:
        if value is not None:
            jax.block_until_ready(value)


# -- Half-spectrum utilities -------------------------------------------------


def make_half_image_weights(image_shape):
    """Return (N_half,) Hermitian weights for half-spectrum inner products.

    For a real-valued image of shape (H, W), the rfft-packed half-spectrum
    has shape (H, W//2+1).  The full inner product is recovered from the half:

        Re<a, b>_full = Re[sum_k w(k) * conj(a_half(k)) * b_half(k)]

    where:
        w = 2 for all interior pixels (each represents itself and its conjugate)
        w = 1 for packed column 0 (DC) -- has no conjugate partner
        w = 1 for packed column -1 (Nyquist, even W only) -- self-conjugate

    NOTE: These are the CORRECT Hermitian weights for Parseval-preserving inner
    products.  RELION does NOT use these — it sums with w=1 everywhere, computing
    roughly half the true likelihood.  The ``half_spectrum_scoring=True`` path in
    run_em uses ones() to match RELION.  This function is used by the
    non-RELION-parity path (``half_spectrum_scoring=False``).
    See TODO(RELION-parity-debt) in run_em for details.
    """
    H, W = image_shape
    w = 2.0 * jnp.ones((H, W // 2 + 1), dtype=jnp.float32)
    w = w.at[:, 0].set(1.0)  # packed column 0 = DC
    w = w.at[:, -1].set(1.0)  # packed column -1 = Nyquist
    return w.reshape(-1)  # (N_half,)


def make_shell_indices_half(image_shape):
    """Return (N_half,) int32 mapping each half-spectrum pixel to its radial shell.

    Uses the rfft-packed layout matching ``full_image_to_half_image``.
    Shell indices range from 0 (DC) to ``image_shape[0] // 2`` (Nyquist).
    """
    # get_grid_of_radial_distances_real returns shape (H, W//2+1) with rounded int distances
    radii = fourier_transform_utils.get_grid_of_radial_distances_real(
        image_shape,
        voxel_size=1,
        scaled=False,
        frequency_shift=0,
        rounded=True,
    )
    return radii.reshape(-1).astype(jnp.int32)


# -- JIT-compiled kernels ---------------------------------------------------


@partial(jax.jit, static_argnums=(5, 6, 7))
def _preprocess_batch(
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    n_images,
    n_trans,
    score_with_masked_images=False,
):
    """Preprocess one image batch for E-step scoring.

    When ``score_with_masked_images`` is True, the likelihood path uses the
    dataset's masked-image preprocessing. The returned ``batch_norm`` is the
    per-image constant ``||y||^2 / sigma^2`` term from the Gaussian
    likelihood. The score kernels intentionally omit that constant from the
    relative candidate scores to avoid catastrophic float32 cancellation, and
    add it back only when absolute log-evidence outputs are requested.
    """

    ## TODO: ALL OF THIS SHOULD BE DONE IN HALF_IMAGE NATIVELY. NO FULL -> HALF CONVERSIONS
    CTF = config.compute_ctf(ctf_params)
    processed = config.process_fn(
        batch,
        apply_image_mask=score_with_masked_images,
    )
    ctf_weighted = processed * CTF / noise_variance
    # Phase shifts operate on full spectrum (need all frequencies for correct shift)
    shifted = core.batch_trans_translate_images(
        ctf_weighted,
        jnp.repeat(translations[None], n_images, axis=0),
        config.image_shape,
    )
    shifted_flat = shifted.reshape(n_images * n_trans, -1)
    # Convert to half-spectrum for all subsequent GEMMs
    shifted_half = fourier_transform_utils.full_image_to_half_image(shifted_flat, config.image_shape)

    batch_norm = jnp.linalg.norm(processed / jnp.sqrt(noise_variance), axis=-1, keepdims=True) ** 2
    ctf2_over_nv = CTF**2 / noise_variance
    # Also convert ctf2_over_nv to half for norm-term GEMM
    ctf2_over_nv_half = fourier_transform_utils.full_image_to_half_image(ctf2_over_nv, config.image_shape)
    return shifted_half, batch_norm, ctf2_over_nv_half


@partial(jax.jit, static_argnums=(5, 6))
def _prepare_reconstruction_batch(
    batch,
    ctf_params,
    noise_variance,
    translations,
    config,
    n_images,
    n_trans,
):
    ## TODO: ALL OF THIS SHOULD BE DONE IN HALF_IMAGE NATIVELY. NO FULL -> HALF CONVERSIONS. 

    """Preprocess one image batch for the unmasked M-step path."""
    CTF = config.compute_ctf(ctf_params)
    processed = config.process_fn(batch, apply_image_mask=False)
    ctf_weighted = processed * CTF / noise_variance
    shifted = core.batch_trans_translate_images(
        ctf_weighted,
        jnp.repeat(translations[None], n_images, axis=0),
        config.image_shape,
    )
    shifted_flat = shifted.reshape(n_images * n_trans, -1)
    return fourier_transform_utils.full_image_to_half_image(
        shifted_flat,
        config.image_shape,
    )


@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def _e_step_block_scores(
    shifted_half,
    batch_norm,
    ctf2_over_nv_half,
    proj_half_weighted,
    proj_abs2_half,
    half_weights,
    n_images,
    n_trans,
    image_shape,
    volume_shape,
):
    """E-step for one rotation block using half-spectrum GEMMs.

    The cross-term GEMM uses weighted projections (half_weights absorbed into
    projections, precomputed once per rotation block) to recover the full inner
    product from half-spectrum data:

        cross[i,r] = -2 Re(conj(shifted_half) @ proj_half_weighted.T)

    The norm-term similarly uses half-weighted |proj|^2.

    Args:
        shifted_half: (n_images * n_trans, N_half) complex -- phase-shifted CTF-weighted images.
        batch_norm: (n_images, 1) float -- ignored additive constant
            ``||processed_image / sqrt(noise)||^2`` carried separately.
        ctf2_over_nv_half: (n_images, N_half) float -- CTF^2/noise in half layout.
        proj_half_weighted: (rot_block, N_half) complex -- projections * half_weights.
        proj_abs2_half: (rot_block, N_half) float -- |proj|^2 * half_weights.
        half_weights: (N_half,) float -- Hermitian weights (for reference, already absorbed).
        n_images, n_trans, image_shape, volume_shape: static args.

    Returns:
        scores: (n_images, rot_block, n_trans) -- log-probability scores (higher = more probable).
    """
    rot_block_size = proj_half_weighted.shape[0]
    # Cross-term: -2 Re(conj(Y_half) @ (P_half * w).T)
    # This recovers -2 Re<Y, P>_full via the half-spectrum identity
    cross = -2.0 * (jnp.conj(shifted_half) @ proj_half_weighted.T).real
    # ``batch_norm`` is a per-image additive constant over the entire
    # rotation x translation grid. Omitting it preserves the posterior exactly
    # while keeping the relative scores representable in float32.
    cross = cross.reshape(n_images, n_trans, rot_block_size)
    cross = cross.swapaxes(1, 2)  # (n_images, rot_block_size, n_trans)
    # Norm-term: sum_k w(k) * CTF^2/noise(k) * |proj(k)|^2
    norms = ctf2_over_nv_half @ proj_abs2_half.T  # (n_images, rot_block_size)
    residuals = cross + norms[..., None]
    # Convert to log-prob scores (higher = more probable)
    return -0.5 * residuals


@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10))
def _e_step_block_scores_windowed(
    shifted_windowed,
    batch_norm,
    ctf2_over_nv_windowed,
    proj_windowed_weighted,
    proj_abs2_windowed,
    half_weights_windowed,
    n_images,
    n_trans,
    n_windowed,
    image_shape,
    volume_shape,
):
    """E-step for one rotation block using windowed half-spectrum GEMMs.

    Same as _e_step_block_scores but operates on the windowed subset of the
    half-spectrum.  All inputs have inner dimension n_windowed instead of N_half.

    Args:
        shifted_windowed: (n_images * n_trans, n_windowed) complex
        batch_norm: (n_images, 1) float -- ignored additive constant
            ``||processed_image / sqrt(noise)||^2`` carried separately.
        ctf2_over_nv_windowed: (n_images, n_windowed) float
        proj_windowed_weighted: (rot_block, n_windowed) complex
        proj_abs2_windowed: (rot_block, n_windowed) float
        half_weights_windowed: (n_windowed,) float
        n_images, n_trans, n_windowed, image_shape, volume_shape: static args.

    Returns:
        scores: (n_images, rot_block, n_trans) log-probability scores.
    """
    rot_block_size = proj_windowed_weighted.shape[0]
    # Cross-term on windowed subset
    cross = -2.0 * (jnp.conj(shifted_windowed) @ proj_windowed_weighted.T).real
    cross = cross.reshape(n_images, n_trans, rot_block_size)
    cross = cross.swapaxes(1, 2)  # (n_images, rot_block_size, n_trans)
    # Norm-term on windowed subset
    norms = ctf2_over_nv_windowed @ proj_abs2_windowed.T  # (n_images, rot_block_size)
    residuals = cross + norms[..., None]
    return -0.5 * residuals


@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11))
def _m_step_block_windowed(
    shifted_windowed,
    scores_block,
    log_Z,
    rotations_block,
    ctf2_over_nv_windowed,
    Ft_y,
    Ft_ctf,
    n_images,
    n_trans,
    n_windowed,
    image_shape,
    volume_shape,
):
    """Normalize scores to probs and accumulate M-step for one rotation block (windowed).

    The M-step GEMM operates on the windowed subset (n_windowed dimension).
    After the GEMM, the result is scattered back to a full half-spectrum
    array before calling adjoint_slice_volume.
    """
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    rot_block_size = rotations_block.shape[0]

    # Normalize
    probs = jnp.exp(scores_block - log_Z[:, None, None])

    # M-step GEMM on windowed subset: P @ shifted_windowed -> (rot_block, n_windowed)
    P = probs.swapaxes(0, 1).reshape(rot_block_size, n_images * n_trans)
    summed_windowed = P @ shifted_windowed  # (rot_block, n_windowed)

    # CTF backprojection on windowed subset
    probs_sum_t = jnp.sum(probs, axis=-1)  # (n_images, rot_block)
    ctf_probs_windowed = probs_sum_t.T @ ctf2_over_nv_windowed  # (rot_block, n_windowed)

    # Hard assignment contribution
    block_best = jnp.max(scores_block.reshape(n_images, -1), axis=1)
    block_argmax = jnp.argmax(scores_block.reshape(n_images, -1), axis=1)

    return Ft_y, Ft_ctf, probs, block_best, block_argmax, summed_windowed, ctf_probs_windowed


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def _adjoint_slice_volume_windowed(
    windowed_half,
    window_indices,
    rotations_block,
    volume,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
):
    """Scatter a windowed half-spectrum into a full half-grid and adjoint-slice.

    This keeps the scatter inside a single jitted helper so the grouped local
    path does not bounce back through Python between the windowed GEMM output
    and the adjoint accumulation.
    """
    return core.adjoint_slice_volume_indexed(
        windowed_half,
        window_indices,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        volume=volume,
        half_image=half_image,
        half_volume=half_volume,
    )


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def _adjoint_slice_volume_half(
    half_block,
    rotations_block,
    volume,
    image_shape,
    volume_shape,
    disc_type,
    half_image,
    half_volume=False,
):
    ## UNNECESSARY? 
    """Adjoint-slice a half-spectrum block into the volume accumulator."""
    return core.adjoint_slice_volume(
        half_block,
        rotations_block,
        image_shape,
        volume_shape,
        disc_type,
        volume=volume,
        half_image=half_image,
        half_volume=half_volume,
    )


@partial(jax.jit, static_argnums=())
def _update_logsumexp(max_s, sum_exp, scores_block):
    """Streaming logsumexp update: accumulate normalization stats from one block.

    Accumulates in float64 to avoid underflow when the posterior is sharp
    (e.g. with RELION's narrow translation prior sigma ~ 0.3 px, most
    candidates get exp(score - max) < float32 epsilon).
    """

    ## TODO: IS THIS FLOAT64 NECESSARY? WE SHOULD PROBABLY HAVE ONE WAY TO SWAP ALL COMPUTATOIN FROM FLOAT32 TO FLOAT64 (E.G. A GLOBAL FLAG LIKE _USE_FLOAT32)
    scores_flat = scores_block.reshape(scores_block.shape[0], -1)  # (n_images, block*trans)
    block_max = jnp.max(scores_flat, axis=1)  # (n_images,)
    new_max = jnp.maximum(max_s, block_max)
    # Use float64 for the exponential sums to avoid underflow
    exp_terms = jnp.sum(
        jnp.exp((scores_flat - new_max[:, None]).astype(jnp.float64)),
        axis=1,
    )
    sum_exp = sum_exp * jnp.exp((max_s - new_max).astype(jnp.float64)) + exp_terms
    return new_max, sum_exp


@partial(jax.jit, static_argnums=(7, 8, 9, 10))
def _m_step_block_compute(
    shifted_half,
    scores_block,
    log_Z,
    rotations_block,
    ctf2_over_nv_half,
    Ft_y,
    Ft_ctf,
    n_images,
    n_trans,
    image_shape,
    volume_shape,
):
    """Normalize scores and compute one non-windowed M-step block.

    The M-step GEMM computes P @ shifted_half -> (rot_block, N_half).
    This is a weighted sum (not an inner product), so no Hermitian weights
    are needed.

    Returns intermediates ``summed_half`` and ``ctf_probs_half`` for optional
    downstream adjoint and noise accumulation.
    """
    rot_block_size = rotations_block.shape[0]
    # Normalize
    probs = jnp.exp(scores_block - log_Z[:, None, None])
    # M-step GEMM: P @ shifted_half -> (rot_block, N_half)
    # This sums shifted half-images weighted by probabilities -- already in half layout!
    P = probs.swapaxes(0, 1).reshape(rot_block_size, n_images * n_trans)
    summed_half = P @ shifted_half  # (rot_block, N_half) -- directly in half layout
    # CTF backprojection: probs_sum_t @ ctf2_over_nv_half -> (rot_block, N_half)
    probs_sum_t = jnp.sum(probs, axis=-1)  # (n_images, rot_block)
    ctf_probs_half = probs_sum_t.T @ ctf2_over_nv_half  # (rot_block, N_half)
    # Hard assignment contribution: argmax over this block
    block_best = jnp.max(scores_block.reshape(n_images, -1), axis=1)
    block_argmax = jnp.argmax(scores_block.reshape(n_images, -1), axis=1)
    ## TODO: THIS FUNCTION HAS A LOT OF USELESS ARGUMENTS NOT BEING USED? IS THIS A BUG? OR A CLEAN UP ft_y Ft_ctf / shapes
    return Ft_y, Ft_ctf, probs, block_best, block_argmax, summed_half, ctf_probs_half


@partial(jax.jit, static_argnums=(7, 8, 9, 10))
def _m_step_block(
    shifted_half,
    scores_block,
    log_Z,
    rotations_block,
    ctf2_over_nv_half,
    Ft_y,
    Ft_ctf,
    n_images,
    n_trans,
    image_shape,
    volume_shape,
):
    ## TODO: IS THIS DEAD CODE?
    """Backwards-compatible non-windowed M-step helper with adjoint included."""
    Ft_y, Ft_ctf, probs, block_best, block_argmax, summed_half, ctf_probs_half = _m_step_block_compute(
        shifted_half,
        scores_block,
        log_Z,
        rotations_block,
        ctf2_over_nv_half,
        Ft_y,
        Ft_ctf,
        n_images,
        n_trans,
        image_shape,
        volume_shape,
    )
    Ft_y = _adjoint_slice_volume_half(
        summed_half,
        rotations_block,
        Ft_y,
        image_shape,
        volume_shape,
        "linear_interp",
        True,
    )
    Ft_ctf = _adjoint_slice_volume_half(
        ctf_probs_half,
        rotations_block,
        Ft_ctf,
        image_shape,
        volume_shape,
        "linear_interp",
        True,
    )
    return Ft_y, Ft_ctf, probs, block_best, block_argmax, summed_half, ctf_probs_half


@partial(jax.jit, static_argnums=(6,))
def _compute_noise_block(
    proj_half, proj_abs2_half, summed_masked, ctf_probs, noise_variance_half, shell_indices, n_shells
):
    ## TODO: QUESTION? Projections (unweighted by half_weights). IS THIS RIGHT? ARE DOCS WRONG? I THOUGHT RELION DID NOT USE WEIGHTS AT ALL? 

    ## TODO: SHOULD WE REALLY BE KEEPING AROUDN BOTH PROJ AND |PROJ|^2 THROUGHOUT CODE? SEEMS WASTEFUL IN MEMORY?


    """Accumulate RELION-style posterior-weighted noise for one rotation block.

    Uses the decomposition::

        E_w[|CTF*proj - img|^2] = E_w[|CTF*proj|^2] - 2*Re(E_w[conj(img)*CTF*proj]) + |img|^2
                                 =     A2            -           2*XA                  + P_img

    ``P_img`` is handled by the caller (image-only, no rotation dependence).
    This function computes the ``A2 - 2*XA`` contribution from one rotation
    block, binned to resolution shells.

    The key identity: since CTF is real-valued,
    ``conj(raw_img_shifted) * CTF = conj(shifted_half) * sigma2``,
    so the XA GEMM output ``P @ shifted_masked`` can be reused.

    Parameters
    ----------
    proj_half : (rot_block, N) complex
        Projections (unweighted by half_weights).
    proj_abs2_half : (rot_block, N) float
        ``|proj|^2``.
    summed_masked : (rot_block, N) complex
        ``P @ shifted_masked_half`` -- masked-image M-step GEMM output.
    ctf_probs : (rot_block, N) float
        ``probs_sum_t.T @ (CTF^2 / noise_variance)`` -- already computed
        for Ft_ctf.
    noise_variance_half : (N,) float
        Per-pixel noise variance in half-spectrum layout.
    shell_indices : (N,) int32
        Radial shell index per half-spectrum pixel.
    n_shells : int (static)
        Number of resolution shells.

    Returns
    -------
    noise_shells : (n_shells,) float
        ``sum_{k in shell} (A2(k) - 2*XA(k))`` contribution from this block.
    a2_shells : (n_shells,) float
        ``sum_{k in shell} A2(k)`` (diagnostic split).
    xa_shells : (n_shells,) float
        ``sum_{k in shell} XA(k)`` (diagnostic split).
    """
    # A2 term: sum_r |proj_r|^2 * (ctf_probs * noise_variance)[r, k]
    # ctf_probs has CTF^2/sigma2; multiply by sigma2 to get CTF^2
    ctf_probs_raw = ctf_probs * noise_variance_half  # (rot_block, N)
    A2 = jnp.sum(proj_abs2_half * ctf_probs_raw, axis=0)  # (N,) sum over rotations

    # XA term: noise_variance * Re(sum_r proj_r * conj(summed_masked_r))
    # summed_masked = P @ shifted_masked, where shifted_masked = img*CTF/sigma2*phase
    # conj(raw*CTF) = conj(shifted_half) * sigma2, so:
    # XA = sigma2 * Re(sum_r proj_r * conj(summed_masked_r))
    cross = jnp.sum(proj_half * jnp.conj(summed_masked), axis=0)  # (N,) complex
    XA = noise_variance_half * cross.real  # (N,)

    # Per-pixel block contribution (without P_img)
    block_noise = A2 - 2.0 * XA
    
    ## TODO: IS THIS REALLY WHAT RELION DOES? WHY ARE STORING THE MIDDLE TERMS LIKE A2 AND XA?

    ## TODO, SO HERE N IS N_SHELLS? WE SHOULD MAKE THAT CLEAR.
    # Bin to resolution shells (no Hermitian weights -- matching RELION)
    ## TODO SHOULD THERE BE HERMITIAN WEIGHTS AT ALL, EVEN IF RELION USED THEM? NOT COMPLEETELY SURE, TRIPLE CHECK
    noise_shells = jnp.zeros(n_shells, dtype=jnp.float32)
    noise_shells = noise_shells.at[shell_indices].add(block_noise)
    a2_shells = jnp.zeros(n_shells, dtype=jnp.float32)
    a2_shells = a2_shells.at[shell_indices].add(A2)
    xa_shells = jnp.zeros(n_shells, dtype=jnp.float32)
    xa_shells = xa_shells.at[shell_indices].add(XA)
    return noise_shells, a2_shells, xa_shells


def _compute_projections_block(volume, rotations_block, image_shape, volume_shape, disc_type):
    """Forward-slice one rotation block in half-spectrum layout.

    Returns (proj_half, |proj_half|^2) on device, both in half-spectrum layout.
    """
    proj_half = core.slice_volume(volume, rotations_block, image_shape, volume_shape, disc_type, half_image=True)
    ## TODO: WE SHOULD THINK ABOUT WHETHER STORING SQUARES IS WORTH IT.
    proj_abs2_half = jnp.abs(proj_half) ** 2
    return proj_half, proj_abs2_half


def run_em(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    rotations,
    translations,
    disc_type: str,
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    current_size: int = None,
    rotation_log_prior: np.ndarray = None,
    translation_log_prior: np.ndarray = None,
    image_indices: np.ndarray = None,
    rotation_translation_mask: np.ndarray = None,
    *, ## TODO: WHAT IS THIS FROM? SEEMS AWKWARD. COULD WE MAKE OPTIONS, PARTICULARLY DEBUG OPTIONS LIKE THIS INTO A CONFIG OBJECT?
    score_with_masked_images: bool = False,
    return_stats: bool = False,
    accumulate_noise: bool = False,
    half_spectrum_scoring: bool = False,
    projection_padding_factor: int = 1,
    reconstruction_padding_factor: int = 1,
    image_corrections: np.ndarray = None,
    scale_corrections: np.ndarray = None,
    image_pre_shifts: np.ndarray = None,
    use_float64_scoring: bool = False,
    use_float64_projections: bool = False,
    do_gridding_correction: bool = False,
    square_window: bool = False,
    return_profile: bool = False,
    sparse_pass2: bool = True,
    disable_adjoint_y: bool = False,
    disable_adjoint_ctf: bool = False,
):
    """One EM iteration with JIT-fused two-pass blockwise normalization and half-spectrum GEMMs.

    Key properties:
    - Never materializes full (n_images, n_rot, n_trans) tensor
    - E-step scores computed twice (for normalization stats, then for M-step)
    - All per-block operations are JIT-compiled
    - Projections computed directly in half-spectrum layout via slice_volume(half_image=True)
    - Half-spectrum GEMMs: N_half = H*(W//2+1) instead of N = H*W (~2x speedup)
    - Hermitian weights absorbed into projections (precomputed once per rotation block)
    - Optional Fourier windowing via current_size: restricts GEMMs to low-frequency
      subset of the half-spectrum for further speedup at early iterations.

    Parameters
    ----------
    current_size : int or None
        Diameter in pixels (like RELION's rlnCurrentImageSize).
        When None, use full resolution (same as Phase 1 behavior).
        When set, only frequencies with radius <= current_size // 2 are
        included in the E-step and M-step GEMMs.
    rotation_log_prior : np.ndarray or None
        Log-prior weights added to E-step scores before softmax. Supports
        either a shared vector of shape ``(n_rot,)`` or an image-specific
        matrix of shape ``(n_images, n_rot)`` for exact local-search unions.
        When None (default), a flat prior is used.
    translation_log_prior : np.ndarray or None
        Log-prior weights added to E-step scores before softmax over the
        translation axis. Supports either a shared vector of shape
        ``(n_trans,)`` or an image-specific matrix of shape
        ``(n_images, n_trans)``.
    image_indices : np.ndarray or None
        Optional subset of images to process. When provided, the returned
        hard assignments and per-image stats are ordered according to this
        subset rather than the full dataset.
    rotation_translation_mask : np.ndarray or None, shape (n_rot, n_trans)
        Optional boolean validity mask over the Cartesian rotation x
        translation grid. Entries set to False are excluded by forcing
        their scores to ``-inf`` in both E-step passes.
    score_with_masked_images : bool
        When True, compute E-step scores from masked images but keep the
        M-step reconstruction on unmasked images.
    return_stats : bool
        When True, also return a :class:`RelionStats` container with the
        per-image log normalizer, best score, maximum posterior
        probability, and additive posterior mass per rotation computed
        during the E-step.
    image_corrections : np.ndarray or None, shape (n_images,)
        Per-image multiplicative correction applied to Fourier images
        before scoring and M-step accumulation.  For RELION parity this
        is ``(avg_norm / normcorr[i]) * scale[group_id[i]]`` — RELION
        applies ``img *= avg_norm_correction / normcorr`` before FFT
        (ml_optimiser.cpp:6240) and scale to the reference
        (ml_optimiser.cpp:7298).  See ``scale_corrections`` for the
        companion norm-term / denominator fix.
    scale_corrections : np.ndarray or None, shape (n_images,)
        Per-image scale correction (``rlnGroupScaleCorrection``).
        RELION applies scale to the *reference* (``Frefctf *= myscale``
        at ml_optimiser.cpp:7298 and ``Mctf *= myscale`` at
        ml_optimiser.cpp:8516).  This means the E-step norm-term and
        the M-step CTF denominator must both carry a ``scale**2``
        factor.  When provided, ``ctf2_over_nv`` is multiplied by
        ``scale**2`` per image to match RELION's convention.
    image_pre_shifts : np.ndarray or None, shape (n_images, 2)
        Per-image translation (in pixels) applied to the processed
        Fourier-space images before scoring.  RELION pre-centers each
        image by its ``old_offset`` (rlnOriginXAngst/pixel_size) via
        ``selfTranslate`` before scoring (ml_optimiser.cpp:6225).
        The equivalent Fourier-space operation is multiplication by
        ``exp(-2πi k·shift)``.  The candidate translations from the
        grid are then relative to this centered position.
    return_profile : bool
        When True, append an :class:`EMProfileStats` timing summary to the
        return tuple.  This is diagnostic only.
    sparse_pass2 : bool
        When True, use pass-1 block maxima to skip pass-2 rotation blocks
        whose posterior mass is negligible for every image in the batch.
    disable_adjoint_y : bool
        Experimental ablation flag. When True, skip the weighted-image
        adjoint accumulation into ``Ft_y``.
    disable_adjoint_ctf : bool
        Experimental ablation flag. When True, skip the CTF adjoint
        accumulation into ``Ft_ctf``.
    """
    overall_t0 = time.time()
    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    image_indices = np.arange(experiment_dataset.n_units) if image_indices is None else np.asarray(image_indices)
    n_images = image_indices.size
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    ## TODO: set default params same as RELION (when starting from GUI) pf=2 I think is there.
    # Pad volume in real space for smoother trilinear projection (RELION pf=2).
    if projection_padding_factor > 1:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection
        ## TODO: ALL VOLUMES SHOULD BE IN SOME KND OF HALF VOLUME FORMAT THROUGHOUT WHEN IN FOURIER DOMAIN. SAME FOR IMAGES (OR THINGS SIZE OF IMAGE E..G CTF)
        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            mean,   ## TODO rename mean? Doesn't make a lot of sense here. also mean_var etc
            volume_shape,
            projection_padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=current_size,
        )
    else:
        mean_for_proj = mean
        proj_volume_shape = volume_shape

    # NOTE: float64 projections tested (Slurm 7174969) — identical results to float32.
    # The 0.0074 Pmax gap is NOT float precision; it's boundary handling in trilinear interp.
    if use_float64_projections:
        mean_for_proj = jnp.asarray(mean_for_proj, dtype=jnp.complex128)

    # Backprojection padding: accumulate into a (pf*N)³ grid for finer
    # trilinear interpolation, matching RELION's --pad flag.
    if reconstruction_padding_factor > 1:
        recon_volume_shape = tuple(d * reconstruction_padding_factor for d in volume_shape)
        recon_volume_size = int(np.prod(recon_volume_shape))
    else:
        recon_volume_shape = volume_shape
        recon_volume_size = int(np.prod(volume_shape))

    H, W = image_shape
    n_half = H * (W // 2 + 1)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    # Precompute half-spectrum weights for E-step scoring.
    #
    # TODO(RELION-parity-debt): RELION sums over the rfft half-image with
    # weight=1 for ALL pixels — no Hermitian doubling.  This is mathematically
    # INCORRECT: for a real-valued signal, the half-spectrum inner product
    # should use weight=2 for interior frequencies (which represent both +k
    # and the conjugate -k) and weight=1 for DC and Nyquist (self-conjugate).
    # By using w=1 everywhere, RELION effectively computes HALF the true
    # Gaussian log-likelihood.  Consequences:
    #   - Posterior is softer (Pmax lower) than the true Bayesian posterior
    #   - MAP orientation is UNCHANGED (same ranking, just scaled score)
    #   - Resolution-dependent signal weighting is slightly wrong at
    #     DC/Nyquist boundaries (negligible in practice)
    # We match RELION exactly here for parity.  Once parity is confirmed,
    # switching to correct Hermitian weights (make_half_image_weights) would
    # sharpen posteriors and may improve convergence speed.  This is tracked
    # as a post-parity improvement.
    #
    # TODO(local-engine-debt): If we keep any dense score path after the local
    # engine split, there is still an inner-product/GEMM-shaped optimization
    # opportunity around the translation dimension. RELION appears to fuse
    # project+translate+score in custom kernels instead of BLAS here, so this
    # is not a parity requirement. Still, we should remember to revisit that
    # opportunity once the local path stops forcing per-image neighborhoods
    # through the shared-grid dense engine.
    if half_spectrum_scoring:
        H_w, W_w = image_shape
        half_weights = jnp.ones(H_w * (W_w // 2 + 1), dtype=jnp.float32)
    else:
        half_weights = make_half_image_weights(image_shape)

    # Precompute window indices if current_size is set
    use_window = current_size is not None and current_size < image_shape[0]
    if use_window:
        from .helpers.fourier_window import make_fourier_window_indices_np

        window_indices_np, n_windowed = make_fourier_window_indices_np(image_shape, current_size, square=square_window)
        window_indices = jnp.asarray(window_indices_np)
        half_weights_windowed = half_weights[window_indices]
        window_desc = "square" if square_window else "circular"
        logger.info(
            "Fourier windowing (%s): current_size=%d, n_windowed=%d / n_half=%d (%.1f%% reduction)",
            window_desc,
            current_size,
            n_windowed,
            n_half,
            100.0 * (1.0 - n_windowed / n_half),
        )
    else:
        window_indices = None
        n_windowed = n_half

    # Upcast half_weights to float64 when scoring in double precision
    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        if use_window:
            half_weights_windowed = half_weights[window_indices]

    # Pad rotations to multiple of block size for fixed shapes
    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate([rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))], axis=0)
    else:
        rotations_padded = rotations

    batch_fetch_time = 0.0
    preprocess_time = 0.0
    score_prep_time = 0.0
    pass1_projection_time = 0.0
    pass1_score_time = 0.0
    pass1_postprocess_time = 0.0
    pass1_logsumexp_time = 0.0
    pass2_skipmask_time = 0.0
    pass2_projection_time = 0.0
    pass2_score_time = 0.0
    pass2_postprocess_time = 0.0
    mstep_time = 0.0
    window_scatter_time = 0.0
    adjoint_y_time = 0.0
    adjoint_ctf_time = 0.0
    noise_time = 0.0
    assignment_time = 0.0
    stats_finalize_time = 0.0
    host_stats_time = 0.0
    solve_time = 0.0
    sync_timers = bool(return_profile)
    sparse_pass2_log_threshold = float(np.log(1e-6))
    sparse_pass2_total_blocks = 0
    sparse_pass2_skipped_blocks = 0
    sparse_pass2_omitted_mass_upper_total = 0.0
    sparse_pass2_omitted_mass_upper_max = 0.0
    sparse_pass2_omitted_mass_upper_image_count = 0

    # Prepare per-rotation log-prior (pad to match rotations_padded)
    per_image_log_prior = False
    if rotation_log_prior is not None:
        rotation_log_prior = np.asarray(rotation_log_prior, dtype=np.float32)
        if rotation_log_prior.ndim == 1:
            log_prior_padded = np.full(n_rot_padded, -1e30, dtype=np.float32)
            log_prior_padded[:n_rot] = rotation_log_prior
        elif rotation_log_prior.ndim == 2:
            if rotation_log_prior.shape != (n_images, n_rot):
                raise ValueError(
                    "rotation_log_prior must have shape "
                    f"({n_images}, {n_rot}) when image-specific, got "
                    f"{rotation_log_prior.shape}",
                )
            log_prior_padded = np.full((n_images, n_rot_padded), -1e30, dtype=np.float32)
            log_prior_padded[:, :n_rot] = rotation_log_prior
            per_image_log_prior = True
        else:
            raise ValueError(
                f"rotation_log_prior must be 1D or 2D, got {rotation_log_prior.ndim} dimensions",
            )
        log_prior_padded_jnp = jnp.asarray(log_prior_padded)
        finite_prior = rotation_log_prior[np.isfinite(rotation_log_prior)]
        if finite_prior.size == 0:
            finite_prior = np.array([-1e30], dtype=np.float32)
        logger.info(
            "Using rotation log-prior: %d rotations%s, range [%.2f, %.2f]",
            n_rot,
            " (per-image)" if per_image_log_prior else "",
            float(finite_prior.min()),
            float(finite_prior.max()),
        )
    else:
        log_prior_padded_jnp = None

    per_image_translation_log_prior = False
    if translation_log_prior is not None:
        translation_log_prior = np.asarray(translation_log_prior, dtype=np.float32)
        if translation_log_prior.ndim == 1:
            if translation_log_prior.shape != (n_trans,):
                raise ValueError(
                    f"translation_log_prior must have shape ({n_trans},), got {translation_log_prior.shape}",
                )
            translation_log_prior_jnp = jnp.asarray(translation_log_prior)
        elif translation_log_prior.ndim == 2:
            if translation_log_prior.shape != (n_images, n_trans):
                raise ValueError(
                    "translation_log_prior must have shape "
                    f"({n_images}, {n_trans}) when image-specific, got "
                    f"{translation_log_prior.shape}",
                )
            translation_log_prior_jnp = jnp.asarray(translation_log_prior)
            per_image_translation_log_prior = True
        else:
            raise ValueError(
                f"translation_log_prior must be 1D or 2D, got {translation_log_prior.ndim} dimensions",
            )
        finite_translation_prior = translation_log_prior[np.isfinite(translation_log_prior)]
        if finite_translation_prior.size == 0:
            finite_translation_prior = np.array([-1e30], dtype=np.float32)
        logger.info(
            "Using translation log-prior: %d translations%s, range [%.2f, %.2f]",
            n_trans,
            " (per-image)" if per_image_translation_log_prior else "",
            float(finite_translation_prior.min()),
            float(finite_translation_prior.max()),
        )
    else:
        translation_log_prior_jnp = None

    candidate_mask_padded_jnp = None
    if rotation_translation_mask is not None:
        candidate_mask = np.asarray(rotation_translation_mask, dtype=bool)
        if candidate_mask.shape != (n_rot, n_trans):
            raise ValueError(
                f"rotation_translation_mask must have shape ({n_rot}, {n_trans}), got {candidate_mask.shape}",
            )
        candidate_mask_padded = np.zeros((n_rot_padded, n_trans), dtype=bool)
        candidate_mask_padded[:n_rot] = candidate_mask
        candidate_mask_padded_jnp = jnp.asarray(candidate_mask_padded)
        logger.info(
            "Using rotation-translation mask: %d / %d candidates valid",
            int(candidate_mask.sum()),
            int(candidate_mask.size),
        )

    # Initialize accumulators (at padded resolution for pf>1 backprojection)
    Ft_y = jnp.zeros(recon_volume_size, dtype=experiment_dataset.dtype)
    Ft_ctf = jnp.zeros(recon_volume_size, dtype=experiment_dataset.dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    log_evidence_per_image = None
    best_log_score_per_image = None
    max_posterior_per_image = None
    rotation_posterior_sums = None
    if return_stats:
        log_evidence_per_image = np.empty(n_images, dtype=np.float32)
        best_log_score_per_image = np.empty(n_images, dtype=np.float32)
        max_posterior_per_image = np.empty(n_images, dtype=np.float32)
        rotation_posterior_sums = np.zeros(n_rot, dtype=np.float64)

    # Noise accumulation precomputation (RELION parity)
    noise_wsum = None
    noise_img_power = None
    noise_sumw = 0.0
    noise_a2 = None  # diagnostic
    noise_xa = None  # diagnostic
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        shell_indices_half = make_shell_indices_half(image_shape)
        if use_window:
            shell_indices_noise = shell_indices_half[window_indices]
        else:
            shell_indices_noise = shell_indices_half
        # noise_variance in half-spectrum layout
        noise_variance_half = fourier_transform_utils.full_image_to_half_image(
            noise_variance.reshape(1, -1),
            image_shape,
        ).squeeze()
        if use_window:
            noise_variance_windowed = noise_variance_half[window_indices]
        else:
            noise_variance_windowed = noise_variance_half
        ## TODO: is this just here for DEBUGGING? IF SO WE NEED TO CLEAN IT UP, THE NON-DEBUGGING SHOULDNT HAVE A BUNCH OF EXTRA COMPUTE/MEMORY FOR TERMS THAT ARE USELESS IF WE'RE NOT IN DEBUGGING. ESEPCIALLY IF IT INVOLVES CPU<-> GPU TRANSFERS.
        noise_wsum = np.zeros(n_shells, dtype=np.float64)
        noise_img_power = np.zeros(n_shells, dtype=np.float64)
        noise_a2 = np.zeros(n_shells, dtype=np.float64)
        noise_xa = np.zeros(n_shells, dtype=np.float64)

    start_idx = 0

    batch_iter = experiment_dataset.iter_batches(
        image_batch_size,
        indices=image_indices,
        by_image=False,
    )
    while True:
        batch_fetch_t0 = time.time()
        try:
            batch_data, _, _, ctf_params, _, _, indices = next(batch_iter)
        except StopIteration:
            batch_fetch_time += time.time() - batch_fetch_t0
            break
        batch_fetch_time += time.time() - batch_fetch_t0
        batch_size = len(indices)
        end_idx = start_idx + batch_size
        batch_data = jnp.asarray(batch_data)
        projection_cache = []
        block_max_per_image = [] if sparse_pass2 else None
        block_pose_counts = [] if sparse_pass2 else None

        # -- PREPROCESS (once per image batch) -- returns half-spectrum --
        preprocess_t0 = time.time()
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            batch_data,
            ctf_params,
            noise_variance,
            translations,
            config,
            batch_size,
            n_trans,
            score_with_masked_images,
        )
        shifted_recon_half = (
            _prepare_reconstruction_batch(
                batch_data,
                ctf_params,
                noise_variance,
                translations,
                config,
                batch_size,
                n_trans,
            )
            if score_with_masked_images
            else shifted_half
        )
        if sync_timers:
            _block_until_ready(shifted_half, shifted_recon_half)

        preprocess_time += time.time() - preprocess_t0

        score_prep_t0 = time.time()
        # -- Per-image corrections (RELION parity: avg_norm/normcorr * scale) --
        # RELION: img *= avg_norm_correction / normcorr  (ml_optimiser.cpp:6240)
        # then   Frefctf *= scale                        (ml_optimiser.cpp:7298)
        # The image-side multiplier is (avg_norm/normcorr)*scale.
        # shifted_half has shape (batch_size * n_trans, N_half) — broadcast
        # the per-image correction across n_trans copies.
        if image_corrections is not None:
            batch_corr = jnp.asarray(image_corrections[np.asarray(indices)])
            # Expand to (batch_size * n_trans,) by repeating each correction n_trans times
            corr_expanded = jnp.repeat(batch_corr, n_trans)
            shifted_half = shifted_half * corr_expanded[:, None]
            shifted_recon_half = shifted_recon_half * corr_expanded[:, None]
            # batch_norm also scales: ||img * corr / sqrt(sigma)||^2 = corr^2 * ||img / sqrt(sigma)||^2
            batch_norm = batch_norm * (batch_corr**2)[:, None]

        # -- Per-image scale correction on CTF²/σ² (RELION parity) --
        # RELION applies scale_correction to the REFERENCE: Frefctf *= myscale
        # (ml_optimiser.cpp:7298) and Mctf *= myscale (ml_optimiser.cpp:8516).
        # This means both the E-step norm-term (ctf²/σ² @ |proj|²) and the
        # M-step denominator (Σ γ·ctf²/σ²) carry scale².  Apply it here so
        # all downstream uses of ctf2_over_nv_half see the correct factor.
        if scale_corrections is not None:
            batch_scale = jnp.asarray(scale_corrections[np.asarray(indices)])
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]

        # -- Per-image pre-centering (RELION parity: old_offset phase shift) --
        # RELION pre-centers each image by its stored translation (old_offset)
        # before scoring.  In Fourier space this is multiplication by
        # exp(-2πi k·shift).  Phase-shift the half-spectrum images so that
        # the translation grid searches relative to the centered position.
        # batch_norm is unaffected (|exp(iθ)| = 1).
        if image_pre_shifts is not None:
            batch_shifts = jnp.asarray(image_pre_shifts[np.asarray(indices)])
            # Compute per-pixel phase factors in half-spectrum layout
            lattice_half = fourier_transform_utils.get_k_coordinate_of_each_pixel_half(
                image_shape, voxel_size=1, scaled=True
            )
            # phase_factors: (batch_size, N_half) complex
            phase_factors = jnp.exp(-2j * jnp.pi * (lattice_half @ batch_shifts.T)).T
            # Expand to (batch_size * n_trans, N_half)
            phase_expanded = jnp.repeat(phase_factors, n_trans, axis=0)
            shifted_half = shifted_half * phase_expanded
            shifted_recon_half = shifted_recon_half * phase_expanded
            ## TODO: THE ORDER (WHILE CORRECT IS A BIT CONFUSING)

        # -- Save pre-DC-exclusion arrays for M-step + noise accumulation --
        # RELION excludes DC from likelihood scores (Minvsigma2[0]=0) but
        # INCLUDES DC in reconstruction weights (backprojector CTF^2) and
        # noise estimation.  Save original arrays before DC zeroing.
        shifted_half_with_dc = shifted_half
        ctf2_over_nv_half_with_dc = ctf2_over_nv_half

        # -- DC exclusion (RELION parity: Minvsigma2[0] = 0) --
        # RELION excludes the DC pixel from likelihood scores.
        # In recovar's half-spectrum layout, DC is NOT at flat index 0.
        # Find the DC pixel by locating shell index 0 in the precomputed
        # shell_indices_half array.
        if half_spectrum_scoring:
            ## TODO: THIS SEEMS LIKE A VERY INFECCICIENT WAY TO DO THIS. JUST FIND INDEX AND .SET IT 0? 
            dc_shell_idx = make_shell_indices_half(image_shape)
            dc_mask = dc_shell_idx == 0  # True at DC pixel(s)
            # Zero out DC in SCORING arrays only
            shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
            ctf2_over_nv_half = jnp.where(dc_mask[None, :], 0.0, ctf2_over_nv_half)

        # -- WINDOW gather (if active) --
        # DC-zeroed arrays for scoring, with-DC arrays for M-step accumulation.
        # RELION excludes DC from likelihood (Minvsigma2[0]=0) but includes
        # it in reconstruction weights (backprojector CTF^2 weight at DC).
        ## TODO: IS THIS WINDOWING STUFF ON BY DEFAULT IN RELION? DO WE NEED TO KEEP BOTH AROUND?
        ## IF WE DO, IT SHOULD BE HANDLED MORE GRACEFULLY THAN HAVING IF STATEMENTS THROUGHOUT THE CODE,
        ## PERHAPS IT SHOULD BE AN OPTION IN TEH DOWNSTREAM FUNCTIONS OR SOMETHING (IF NECESSARY)
        if use_window:
            shifted_windowed = shifted_half[:, window_indices]
            shifted_recon_windowed = shifted_recon_half[:, window_indices]
            ctf2_over_nv_windowed = ctf2_over_nv_half[:, window_indices]
            ctf2_over_nv_windowed_mstep = ctf2_over_nv_half_with_dc[:, window_indices]
        else:
            shifted_windowed = shifted_half
            shifted_recon_windowed = shifted_recon_half
            ctf2_over_nv_windowed = ctf2_over_nv_half
            ctf2_over_nv_windowed_mstep = ctf2_over_nv_half_with_dc

        # -- Noise: precompute per-batch image power spectrum --
        if accumulate_noise:
            # P_img = sum_i |masked_img_i(k)|^2 per half-spectrum pixel
            # Use the masked processed images (score path).
            ## TODO ONCE AGAIN THIS SHOULD BE IN HALF IMAGE NATIVELY
            processed_masked = config.process_fn(batch_data, apply_image_mask=score_with_masked_images)
            processed_masked_half = fourier_transform_utils.full_image_to_half_image(
                processed_masked,
                image_shape,
            )
            # Sum |img|^2 over images in this batch, bin to shells (FULL spectrum, not windowed)
            batch_img_power = jnp.sum(jnp.abs(processed_masked_half) ** 2, axis=0)  # (N_half,)
            batch_img_power_shells = jnp.zeros(n_shells, dtype=jnp.float32)
            batch_img_power_shells = batch_img_power_shells.at[shell_indices_half].add(batch_img_power)
            # TODO: IS THIS SENDING TO CPU NECESSARY? SHOULD BE HANDLED BY PUT INSTEAD OF NP.() CALLS, AND SHOULD PROBABLY BE KEPT TO A MINIMIMUM (I THINK TI CAUSES JAX TO SYNCHRONIZE?)
            noise_img_power += np.asarray(batch_img_power_shells, dtype=np.float64)
            noise_sumw += batch_size
            # Masked shifted images for the noise GEMM: use WITH-DC versions
            # (RELION includes DC in noise but excludes from scoring)
            if use_window:
                shifted_masked_for_noise = shifted_half_with_dc[:, window_indices]
            else:
                shifted_masked_for_noise = shifted_half_with_dc

        # -- Float64 scoring upcast (RELION parity: RFLOAT=double) --
        ## IF THIS IS DEFAULT, THEN BE IT, BUT THERE SHOULD BE A NICER WAY TO DO THIS. E.G. DTYPE_SCORING = JNP.FLOAT 64 IF... etc instead of having a bunch of if statements
        # RELION uses double precision for all scoring arithmetic
        # (macros.h:77: RFLOAT=double unless RELION_SINGLE_PRECISION).
        # Upcast the scoring arrays to float64/complex128 before the GEMMs
        # so that accumulation over ~4900 windowed elements matches RELION.
        if use_float64_scoring:
            shifted_half = shifted_half.astype(jnp.complex128)
            ctf2_over_nv_half = ctf2_over_nv_half.astype(jnp.float64)
            if use_window:
                shifted_windowed = shifted_windowed.astype(jnp.complex128)
                ctf2_over_nv_windowed = ctf2_over_nv_windowed.astype(jnp.float64)
            else:
                shifted_windowed = shifted_half
                ctf2_over_nv_windowed = ctf2_over_nv_half
            shifted_recon_half = shifted_recon_half.astype(jnp.complex128)
            if use_window:
                shifted_recon_windowed = shifted_recon_windowed.astype(jnp.complex128)
            else:
                shifted_recon_windowed = shifted_recon_half

        if sync_timers:
            ready_values = [
                shifted_half,
                shifted_recon_half,
                batch_norm,
                ctf2_over_nv_half,
                shifted_windowed,
                shifted_recon_windowed,
                ctf2_over_nv_windowed,
            ]
            if accumulate_noise:
                ready_values.append(shifted_masked_for_noise)
            _block_until_ready(*ready_values)
        score_prep_time += time.time() - score_prep_t0

        # -- PASS 1: streaming logsumexp over rotation blocks --
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size, dtype=jnp.float64)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_t0 = time.time()
            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean_for_proj, rots_b, image_shape, proj_volume_shape, disc_type
            )
            if sync_timers:
                _block_until_ready(proj_half_b, proj_abs2_half_b)
            pass1_projection_time += time.time() - proj_t0
            if projection_cache is not None:
                projection_cache.append((proj_half_b, proj_abs2_half_b))

            score_t0 = time.time()
            if use_window:
                # Gather windowed subset from projections
                proj_windowed_b = proj_half_b[:, window_indices]
                proj_abs2_windowed_b = proj_abs2_half_b[:, window_indices]
                proj_windowed_weighted_b = proj_windowed_b * half_weights_windowed
                proj_abs2_windowed_weighted_b = proj_abs2_windowed_b * half_weights_windowed

                scores = _e_step_block_scores_windowed(
                    shifted_windowed,
                    batch_norm,
                    ctf2_over_nv_windowed,
                    proj_windowed_weighted_b,
                    proj_abs2_windowed_weighted_b,
                    half_weights_windowed,
                    batch_size,
                    n_trans,
                    n_windowed,
                    image_shape,
                    volume_shape,
                )
            else:
                # Full half-spectrum path (Phase 1 behavior)
                proj_half_weighted_b = proj_half_b * half_weights
                proj_abs2_weighted_b = proj_abs2_half_b * half_weights

                scores = _e_step_block_scores(
                    shifted_half,
                    batch_norm,
                    ctf2_over_nv_half,
                    proj_half_weighted_b,
                    proj_abs2_weighted_b,
                    half_weights,
                    batch_size,
                    n_trans,
                    image_shape,
                    volume_shape,
                )

            if sync_timers:
                _block_until_ready(scores)
            pass1_score_time += time.time() - score_t0

            pass1_postprocess_t0 = time.time()
            # Add rotation log-prior (Gaussian angular prior for local search)
            if log_prior_padded_jnp is not None:
                if per_image_log_prior:
                    log_prior_block = log_prior_padded_jnp[start_idx:end_idx, r0:r1]
                    scores = scores + log_prior_block[:, :, None]
                else:
                    log_prior_block = log_prior_padded_jnp[r0:r1]
                    scores = scores + log_prior_block[None, :, None]

            if translation_log_prior_jnp is not None:
                if per_image_translation_log_prior:
                    translation_prior_block = translation_log_prior_jnp[start_idx:end_idx]
                    scores = scores + translation_prior_block[:, None, :]
                else:
                    scores = scores + translation_log_prior_jnp[None, None, :]

            if candidate_mask_padded_jnp is not None:
                candidate_mask_block = candidate_mask_padded_jnp[r0:r1]
                scores = jnp.where(candidate_mask_block[None, :, :], scores, -jnp.inf)

            # Mask padding rotations (set their scores to -inf)
            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)

            if block_max_per_image is not None:
                actual_rot = max(0, min(rotation_block_size, n_rot - r0))
                block_max_per_image.append(jnp.max(scores, axis=(1, 2)))
                block_pose_counts.append(actual_rot * n_trans)

            if sync_timers:
                _block_until_ready(scores)
            pass1_postprocess_time += time.time() - pass1_postprocess_t0

            logsumexp_t0 = time.time()
            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)
            if sync_timers:
                _block_until_ready(max_s, sum_exp)
            pass1_logsumexp_time += time.time() - logsumexp_t0

        log_Z = max_s + jnp.log(sum_exp)  # (batch_size,)
        skip_pass2_block = np.zeros(n_blocks, dtype=bool)
        pass2_skipmask_t0 = time.time()
        if block_max_per_image:
            block_max_matrix = jnp.stack(block_max_per_image, axis=0)
            block_log_pose_counts = jnp.log(jnp.asarray(block_pose_counts, dtype=jnp.float64))[:, None]
            finite_log_z = jnp.isfinite(log_Z)
            log_omitted_mass_upper = jnp.where(
                finite_log_z[None, :],
                block_log_pose_counts + block_max_matrix.astype(jnp.float64) - log_Z[None, :].astype(jnp.float64),
                jnp.inf,
            )
            skip_pass2_block = np.asarray(
                jnp.all(log_omitted_mass_upper < sparse_pass2_log_threshold, axis=1),
                dtype=bool,
            )
            sparse_pass2_total_blocks += int(n_blocks)
            sparse_pass2_skipped_blocks += int(skip_pass2_block.sum())
            if np.any(skip_pass2_block):
                skipped_mass_upper = jnp.sum(
                    jnp.where(
                        jnp.asarray(skip_pass2_block)[:, None],
                        jnp.exp(jnp.minimum(log_omitted_mass_upper, 50.0)),
                        0.0,
                    ),
                    axis=0,
                )
                skipped_mass_upper_np = np.asarray(skipped_mass_upper, dtype=np.float64)
                sparse_pass2_omitted_mass_upper_total += float(np.sum(skipped_mass_upper_np))
                sparse_pass2_omitted_mass_upper_max = max(
                    sparse_pass2_omitted_mass_upper_max,
                    float(np.max(skipped_mass_upper_np)),
                )
                sparse_pass2_omitted_mass_upper_image_count += int(batch_size)
            if sync_timers:
                _block_until_ready(block_max_matrix, log_omitted_mass_upper)
        pass2_skipmask_time += time.time() - pass2_skipmask_t0

        # -- PASS 2: recompute scores, normalize, accumulate M-step --
        best_score = jnp.full(batch_size, -jnp.inf)
        best_argmax = jnp.zeros(batch_size, dtype=jnp.int32)

        for b in range(n_blocks):
            if skip_pass2_block[b]:
                continue

            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            if projection_cache is not None:
                proj_half_b, proj_abs2_half_b = projection_cache[b]
            else:
                proj_t0 = time.time()
                proj_half_b, proj_abs2_half_b = _compute_projections_block(
                    mean_for_proj, rots_b, image_shape, proj_volume_shape, disc_type
                )
                pass2_projection_time += time.time() - proj_t0

            score_t0 = time.time()
            if use_window:
                # Gather windowed subset
                proj_windowed_b = proj_half_b[:, window_indices]
                proj_abs2_windowed_b = proj_abs2_half_b[:, window_indices]
                proj_windowed_weighted_b = proj_windowed_b * half_weights_windowed
                proj_abs2_windowed_weighted_b = proj_abs2_windowed_b * half_weights_windowed

                scores = _e_step_block_scores_windowed(
                    shifted_windowed,
                    batch_norm,
                    ctf2_over_nv_windowed,
                    proj_windowed_weighted_b,
                    proj_abs2_windowed_weighted_b,
                    half_weights_windowed,
                    batch_size,
                    n_trans,
                    n_windowed,
                    image_shape,
                    volume_shape,
                )
            else:
                proj_half_weighted_b = proj_half_b * half_weights
                proj_abs2_weighted_b = proj_abs2_half_b * half_weights

                scores = _e_step_block_scores(
                    shifted_half,
                    batch_norm,
                    ctf2_over_nv_half,
                    proj_half_weighted_b,
                    proj_abs2_weighted_b,
                    half_weights,
                    batch_size,
                    n_trans,
                    image_shape,
                    volume_shape,
                )

            if sync_timers:
                _block_until_ready(scores)
            pass2_score_time += time.time() - score_t0

            pass2_postprocess_t0 = time.time()
            # Add rotation log-prior (must match pass 1 exactly)
            if log_prior_padded_jnp is not None:
                if per_image_log_prior:
                    log_prior_block = log_prior_padded_jnp[start_idx:end_idx, r0:r1]
                    scores = scores + log_prior_block[:, :, None]
                else:
                    log_prior_block = log_prior_padded_jnp[r0:r1]
                    scores = scores + log_prior_block[None, :, None]

            if translation_log_prior_jnp is not None:
                if per_image_translation_log_prior:
                    translation_prior_block = translation_log_prior_jnp[start_idx:end_idx]
                    scores = scores + translation_prior_block[:, None, :]
                else:
                    scores = scores + translation_log_prior_jnp[None, None, :]

            if candidate_mask_padded_jnp is not None:
                candidate_mask_block = candidate_mask_padded_jnp[r0:r1]
                scores = jnp.where(candidate_mask_block[None, :, :], scores, -jnp.inf)

            # Mask padding
            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)
            if sync_timers:
                _block_until_ready(scores)
            pass2_postprocess_time += time.time() - pass2_postprocess_t0

            if use_window:
                # Windowed M-step: GEMM at reduced dimension, then scatter back
                # Use with-DC ctf2 for M-step accumulation (DC is excluded
                # from scoring but must be included in reconstruction weights).
                mstep_t0 = time.time()
                (Ft_y, Ft_ctf, probs, block_best, block_argmax, summed_windowed, ctf_probs_windowed) = (
                    _m_step_block_windowed(
                        shifted_recon_windowed,
                        scores,
                        log_Z,
                        rots_b,
                        ctf2_over_nv_windowed_mstep,
                        Ft_y,
                        Ft_ctf,
                        batch_size,
                        n_trans,
                        n_windowed,
                        image_shape,
                        volume_shape,
                    )
                )
                if sync_timers:
                    _block_until_ready(
                        Ft_y,
                        Ft_ctf,
                        probs,
                        block_best,
                        block_argmax,
                        summed_windowed,
                        ctf_probs_windowed,
                    )
                mstep_time += time.time() - mstep_t0

                if not disable_adjoint_y:
                    adjoint_y_t0 = time.time()
                    Ft_y = _adjoint_slice_volume_windowed(
                        summed_windowed,
                        window_indices,
                        rots_b,
                        Ft_y,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        True,
                    )
                    if sync_timers:
                        _block_until_ready(Ft_y)
                    adjoint_y_time += time.time() - adjoint_y_t0

                if not disable_adjoint_ctf:
                    adjoint_ctf_t0 = time.time()
                    Ft_ctf = _adjoint_slice_volume_windowed(
                        ctf_probs_windowed,
                        window_indices,
                        rots_b,
                        Ft_ctf,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        True,
                    )
                    if sync_timers:
                        _block_until_ready(Ft_ctf)
                    adjoint_ctf_time += time.time() - adjoint_ctf_t0
            else:
                # Non-windowed path: use with-DC ctf2 for M-step accumulation
                mstep_t0 = time.time()
                (Ft_y, Ft_ctf, probs, block_best, block_argmax, summed_half_block, ctf_probs_half_block) = (
                    _m_step_block_compute(
                        shifted_recon_half,
                        scores,
                        log_Z,
                        rots_b,
                        ctf2_over_nv_half_with_dc,
                        Ft_y,
                        Ft_ctf,
                        batch_size,
                        n_trans,
                        image_shape,
                        recon_volume_shape,
                    )
                )
                if sync_timers:
                    _block_until_ready(
                        Ft_y,
                        Ft_ctf,
                        probs,
                        block_best,
                        block_argmax,
                        summed_half_block,
                        ctf_probs_half_block,
                    )
                mstep_time += time.time() - mstep_t0

                if not disable_adjoint_y:
                    adjoint_y_t0 = time.time()
                    Ft_y = _adjoint_slice_volume_half(
                        summed_half_block,
                        rots_b,
                        Ft_y,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        True,
                    )
                    if sync_timers:
                        _block_until_ready(Ft_y)
                    adjoint_y_time += time.time() - adjoint_y_t0

                if not disable_adjoint_ctf:
                    adjoint_ctf_t0 = time.time()
                    Ft_ctf = _adjoint_slice_volume_half(
                        ctf_probs_half_block,
                        rots_b,
                        Ft_ctf,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        True,
                    )
                    if sync_timers:
                        _block_until_ready(Ft_ctf)
                    adjoint_ctf_time += time.time() - adjoint_ctf_t0

            # -- Noise accumulation for this rotation block --
            if accumulate_noise:
                noise_t0 = time.time()
                rot_block_size_actual = rots_b.shape[0]
                # Compute masked GEMM: P @ shifted_masked (with DC intact)
                P_noise = probs.swapaxes(0, 1).reshape(rot_block_size_actual, batch_size * n_trans)
                summed_masked_noise = P_noise @ shifted_masked_for_noise  # (rot_block, N_noise)
                # ctf_probs for noise: recompute WITH DC (M-step used DC-zeroed version)
                probs_sum_t_noise = jnp.sum(probs, axis=-1)  # (n_images, rot_block)
                if use_window:
                    ctf2_nv_noise = ctf2_over_nv_half_with_dc[:, window_indices]
                    ctf_probs_for_noise = probs_sum_t_noise.T @ ctf2_nv_noise
                    nv_for_noise = noise_variance_windowed
                    si_for_noise = shell_indices_noise
                    proj_for_noise = proj_windowed_b
                    proj_abs2_for_noise = proj_abs2_windowed_b
                else:
                    ctf_probs_for_noise = probs_sum_t_noise.T @ ctf2_over_nv_half_with_dc
                    nv_for_noise = noise_variance_half
                    si_for_noise = shell_indices_noise
                    proj_for_noise = proj_half_b
                    proj_abs2_for_noise = proj_abs2_half_b

                block_noise_shells, block_a2_shells, block_xa_shells = _compute_noise_block(
                    proj_for_noise,
                    proj_abs2_for_noise,
                    summed_masked_noise,
                    ctf_probs_for_noise,
                    nv_for_noise,
                    si_for_noise,
                    n_shells,
                )
                if sync_timers:
                    _block_until_ready(block_noise_shells, block_a2_shells, block_xa_shells)
                noise_wsum += np.asarray(block_noise_shells, dtype=np.float64)
                noise_a2 += np.asarray(block_a2_shells, dtype=np.float64)
                noise_xa += np.asarray(block_xa_shells, dtype=np.float64)
                noise_time += time.time() - noise_t0

            # Track global hard assignment
            assignment_t0 = time.time()
            improved = block_best > best_score
            best_score = jnp.where(improved, block_best, best_score)
            best_argmax = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax)
            if sync_timers:
                _block_until_ready(best_score, best_argmax)
            assignment_time += time.time() - assignment_t0

            if return_stats:
                host_stats_t0 = time.time()
                actual_rot = max(0, min(rotation_block_size, n_rot - r0))
                if actual_rot > 0:
                    block_rotation_sums = np.asarray(
                        jnp.sum(probs[:, :actual_rot, :], axis=(0, 2)),
                        dtype=np.float64,
                    )
                    rotation_posterior_sums[r0 : r0 + actual_rot] += block_rotation_sums
                host_stats_time += time.time() - host_stats_t0

        if return_stats:
            stats_finalize_t0 = time.time()
            log_score_offset = -0.5 * jnp.squeeze(batch_norm, axis=1)
            pmax = jnp.exp(best_score - log_Z)
            log_evidence_per_image[start_idx:end_idx] = np.asarray(
                log_Z + log_score_offset,
                dtype=np.float32,
            )
            best_log_score_per_image[start_idx:end_idx] = np.asarray(
                best_score + log_score_offset,
                dtype=np.float32,
            )
            max_posterior_per_image[start_idx:end_idx] = np.asarray(
                pmax,
                dtype=np.float32,
            )
            if sync_timers:
                _block_until_ready(log_Z, best_score, pmax)
            stats_finalize_time += time.time() - stats_finalize_t0

        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax)
        start_idx = end_idx

    # -- SOLVE --
    from recovar.reconstruction import relion_functions

    if reconstruction_padding_factor > 1:
        new_mean = None
    else:
        solve_t0 = time.time()
        new_mean = relion_functions.post_process_from_filter(
            experiment_dataset,
            Ft_ctf,
            Ft_y,
            tau=mean_variance,
            disc_type=disc_type,
        ).reshape(-1)
        if sync_timers:
            _block_until_ready(new_mean)
        solve_time += time.time() - solve_t0

    noise_stats = None
    if accumulate_noise:
        # Diagnostic: log per-shell A2, XA, img_power, wsum for the first 6 shells
        # so we can compare across iterations of refine.
        try:
            n_log_shells = min(6, len(noise_wsum))
            logger.info(
                "[NOISE-DIAG] sumw=%.0f n_rot=%d use_window=%s",
                float(noise_sumw),
                int(n_rot),
                bool(use_window),
            )
            logger.info(
                "[NOISE-DIAG] A2 (first %d shells): %s",
                n_log_shells,
                ", ".join(f"{noise_a2[i]:.3e}" for i in range(n_log_shells)),
            )
            logger.info(
                "[NOISE-DIAG] XA (first %d shells): %s",
                n_log_shells,
                ", ".join(f"{noise_xa[i]:.3e}" for i in range(n_log_shells)),
            )
            logger.info(
                "[NOISE-DIAG] img_power (first %d shells): %s",
                n_log_shells,
                ", ".join(f"{noise_img_power[i]:.3e}" for i in range(n_log_shells)),
            )
            logger.info(
                "[NOISE-DIAG] wsum=A2-2XA (first %d shells): %s",
                n_log_shells,
                ", ".join(f"{noise_wsum[i]:.3e}" for i in range(n_log_shells)),
            )
        except Exception as exc:
            logger.warning("noise diagnostic logging failed: %s", exc)
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.asarray(noise_wsum, dtype=jnp.float32),
            wsum_img_power=jnp.asarray(noise_img_power, dtype=jnp.float32),
            sumw=float(noise_sumw),
        )

    if sparse_pass2 and sparse_pass2_total_blocks:
        omitted_mass_upper_mean = (
            sparse_pass2_omitted_mass_upper_total / sparse_pass2_omitted_mass_upper_image_count
            if sparse_pass2_omitted_mass_upper_image_count
            else 0.0
        )
        logger.info(
            "Sparse pass2 skipped %d / %d pass2 rotation blocks (%.1f%% of pass2 blocks); "
            "omitted posterior mass upper bound mean=%.3e max=%.3e sum=%.3e",
            sparse_pass2_skipped_blocks,
            sparse_pass2_total_blocks,
            100.0 * sparse_pass2_skipped_blocks / sparse_pass2_total_blocks,
            omitted_mass_upper_mean,
            sparse_pass2_omitted_mass_upper_max,
            sparse_pass2_omitted_mass_upper_total,
        )

    if return_stats:
        host_stats_t0 = time.time()
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.asarray(log_evidence_per_image),
            best_log_score_per_image=jnp.asarray(best_log_score_per_image),
            max_posterior_per_image=jnp.asarray(max_posterior_per_image),
            rotation_posterior_sums=jnp.asarray(rotation_posterior_sums, dtype=jnp.float32),
        )
        host_stats_time += time.time() - host_stats_t0
    if return_profile:
        ready_values = [new_mean, Ft_y, Ft_ctf]
        if noise_stats is not None:
            ready_values.extend([noise_stats.wsum_sigma2_noise, noise_stats.wsum_img_power])
        if return_stats:
            ready_values.extend(
                [
                    relion_stats.log_evidence_per_image,
                    relion_stats.best_log_score_per_image,
                    relion_stats.max_posterior_per_image,
                    relion_stats.rotation_posterior_sums,
                ]
            )
        _block_until_ready(*ready_values)
        total_wall_time = time.time() - overall_t0
        omitted_mass_upper_mean = (
            sparse_pass2_omitted_mass_upper_total / sparse_pass2_omitted_mass_upper_image_count
            if sparse_pass2_omitted_mass_upper_image_count
            else 0.0
        )
        attributed_time = (
            batch_fetch_time
            + preprocess_time
            + score_prep_time
            + pass1_projection_time
            + pass1_score_time
            + pass1_postprocess_time
            + pass1_logsumexp_time
            + pass2_skipmask_time
            + pass2_projection_time
            + pass2_score_time
            + pass2_postprocess_time
            + mstep_time
            + window_scatter_time
            + adjoint_y_time
            + adjoint_ctf_time
            + noise_time
            + assignment_time
            + stats_finalize_time
            + host_stats_time
            + solve_time
        )
        em_profile = EMProfileStats(
            batch_fetch_s=float(batch_fetch_time),
            preprocess_s=float(preprocess_time),
            score_prep_s=float(score_prep_time),
            pass1_projection_s=float(pass1_projection_time),
            pass1_score_s=float(pass1_score_time),
            pass1_postprocess_s=float(pass1_postprocess_time),
            pass1_logsumexp_s=float(pass1_logsumexp_time),
            pass2_skipmask_s=float(pass2_skipmask_time),
            pass2_projection_s=float(pass2_projection_time),
            pass2_score_s=float(pass2_score_time),
            pass2_postprocess_s=float(pass2_postprocess_time),
            mstep_s=float(mstep_time),
            window_scatter_s=float(window_scatter_time),
            adjoint_y_s=float(adjoint_y_time),
            adjoint_ctf_s=float(adjoint_ctf_time),
            noise_s=float(noise_time),
            assignment_s=float(assignment_time),
            stats_finalize_s=float(stats_finalize_time),
            host_stats_s=float(host_stats_time),
            solve_s=float(solve_time),
            accounted_s=float(attributed_time),
            total_wall_s=float(total_wall_time),
            unattributed_s=float(max(total_wall_time - attributed_time, 0.0)),
            n_images=int(n_images),
            n_trans=int(n_trans),
            n_rot=int(n_rot),
            n_rot_padded=int(n_rot_padded),
            n_blocks=int(n_blocks),
            n_windowed=int(n_windowed),
            use_window=bool(use_window),
            reused_pass1_projections=True,
            sparse_pass2_total_blocks=int(sparse_pass2_total_blocks),
            sparse_pass2_skipped_blocks=int(sparse_pass2_skipped_blocks),
            sparse_pass2_omitted_mass_upper_mean=float(omitted_mass_upper_mean),
            sparse_pass2_omitted_mass_upper_max=float(sparse_pass2_omitted_mass_upper_max),
            sparse_pass2_omitted_mass_upper_sum=float(sparse_pass2_omitted_mass_upper_total),
        )
    else:
        em_profile = None

    if return_stats:
        if accumulate_noise:
            if return_profile:
                return new_mean, hard_assignment, Ft_y, Ft_ctf, relion_stats, noise_stats, em_profile
            return new_mean, hard_assignment, Ft_y, Ft_ctf, relion_stats, noise_stats
        if return_profile:
            return new_mean, hard_assignment, Ft_y, Ft_ctf, relion_stats, em_profile
        return new_mean, hard_assignment, Ft_y, Ft_ctf, relion_stats

    if accumulate_noise:
        if return_profile:
            return new_mean, hard_assignment, Ft_y, Ft_ctf, noise_stats, em_profile
        return new_mean, hard_assignment, Ft_y, Ft_ctf, noise_stats

    if return_profile:
        return new_mean, hard_assignment, Ft_y, Ft_ctf, em_profile
    return new_mean, hard_assignment, Ft_y, Ft_ctf


def compute_e_step_weights(
    experiment_dataset,
    mean,
    noise_variance,
    rotations,
    translations,
    disc_type: str,
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    current_size: int = None,
    score_with_masked_images: bool = False,
):
    """E-step only: compute posterior weights for all (rotation, translation) pairs.

    This runs pass 1 (logsumexp) and pass 2 (normalize) of the blockwise
    E-step but does NOT accumulate M-step statistics.  Used by the adaptive
    oversampling module to identify significant samples before pass 2.

    Returns the posterior weight matrix (n_images, n_rot * n_trans) which
    sums to ~1.0 per image.  For large grids this can be memory-intensive;
    the caller should use it for significance pruning then discard it.

    Parameters
    ----------
    experiment_dataset : dataset object
    mean : jnp.ndarray, shape (volume_size,)
    noise_variance : jnp.ndarray, shape (image_size,)
    rotations : np.ndarray, shape (n_rot, 3, 3)
    translations : jnp.ndarray, shape (n_trans, 2)
    disc_type : str
    image_batch_size : int
    rotation_block_size : int
    current_size : int or None

    Returns
    -------
    weights : np.ndarray, shape (n_images, n_rot * n_trans), dtype float32
        Posterior weights (probabilities).
    hard_assignments : np.ndarray, shape (n_images,), dtype int32
        Best (rotation_idx * n_trans + trans_idx) per image.
    """
    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    n_images = experiment_dataset.n_units
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape

    H, W = image_shape
    n_half = H * (W // 2 + 1)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    half_weights = make_half_image_weights(image_shape)

    use_window = current_size is not None and current_size < image_shape[0]
    if use_window:
        from .helpers.fourier_window import make_fourier_window_indices_np

        window_indices_np, n_windowed = make_fourier_window_indices_np(image_shape, current_size)
        window_indices = jnp.asarray(window_indices_np)
        half_weights_windowed = half_weights[window_indices]
    else:
        window_indices = None
        n_windowed = n_half

    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate([rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))], axis=0)
    else:
        rotations_padded = rotations

    # Allocate output weights array on host
    all_weights = np.empty((n_images, n_rot * n_trans), dtype=np.float32)
    hard_assignment = np.empty(n_images, dtype=np.int32)

    image_indices = np.arange(n_images)
    start_idx = 0

    for batch_data, _, _, ctf_params, _, _, indices in experiment_dataset.iter_batches(
        image_batch_size,
        indices=image_indices,
        by_image=False,
    ):
        batch_size = len(indices)
        end_idx = start_idx + batch_size
        batch_data = jnp.asarray(batch_data)

        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            batch_data,
            ctf_params,
            noise_variance,
            translations,
            config,
            batch_size,
            n_trans,
            score_with_masked_images,
        )

        if use_window:
            shifted_windowed = shifted_half[:, window_indices]
            ctf2_over_nv_windowed = ctf2_over_nv_half[:, window_indices]
        else:
            shifted_windowed = shifted_half
            ctf2_over_nv_windowed = ctf2_over_nv_half

        # Pass 1: streaming logsumexp
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size, dtype=jnp.float64)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean, rots_b, image_shape, volume_shape, disc_type
            )

            if use_window:
                proj_windowed_b = proj_half_b[:, window_indices]
                proj_abs2_windowed_b = proj_abs2_half_b[:, window_indices]
                proj_windowed_weighted_b = proj_windowed_b * half_weights_windowed
                proj_abs2_windowed_weighted_b = proj_abs2_windowed_b * half_weights_windowed
                scores = _e_step_block_scores_windowed(
                    shifted_windowed,
                    batch_norm,
                    ctf2_over_nv_windowed,
                    proj_windowed_weighted_b,
                    proj_abs2_windowed_weighted_b,
                    half_weights_windowed,
                    batch_size,
                    n_trans,
                    n_windowed,
                    image_shape,
                    volume_shape,
                )
            else:
                proj_half_weighted_b = proj_half_b * half_weights
                proj_abs2_weighted_b = proj_abs2_half_b * half_weights
                scores = _e_step_block_scores(
                    shifted_half,
                    batch_norm,
                    ctf2_over_nv_half,
                    proj_half_weighted_b,
                    proj_abs2_weighted_b,
                    half_weights,
                    batch_size,
                    n_trans,
                    image_shape,
                    volume_shape,
                )

            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)

            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)

        log_Z = max_s + jnp.log(sum_exp)

        # Pass 2: recompute scores and normalize to weights
        best_score = jnp.full(batch_size, -jnp.inf)
        best_argmax = jnp.zeros(batch_size, dtype=jnp.int32)
        batch_weights_blocks = []

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean, rots_b, image_shape, volume_shape, disc_type
            )

            if use_window:
                proj_windowed_b = proj_half_b[:, window_indices]
                proj_abs2_windowed_b = proj_abs2_half_b[:, window_indices]
                proj_windowed_weighted_b = proj_windowed_b * half_weights_windowed
                proj_abs2_windowed_weighted_b = proj_abs2_windowed_b * half_weights_windowed
                scores = _e_step_block_scores_windowed(
                    shifted_windowed,
                    batch_norm,
                    ctf2_over_nv_windowed,
                    proj_windowed_weighted_b,
                    proj_abs2_windowed_weighted_b,
                    half_weights_windowed,
                    batch_size,
                    n_trans,
                    n_windowed,
                    image_shape,
                    volume_shape,
                )
            else:
                proj_half_weighted_b = proj_half_b * half_weights
                proj_abs2_weighted_b = proj_abs2_half_b * half_weights
                scores = _e_step_block_scores(
                    shifted_half,
                    batch_norm,
                    ctf2_over_nv_half,
                    proj_half_weighted_b,
                    proj_abs2_weighted_b,
                    half_weights,
                    batch_size,
                    n_trans,
                    image_shape,
                    volume_shape,
                )

            if r1 > n_rot:
                valid = n_rot - r0
                pmask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(pmask[None, :, None], scores, -jnp.inf)

            # Normalize to probabilities
            probs = jnp.exp(scores - log_Z[:, None, None])

            # Track hard assignment
            block_best = jnp.max(scores.reshape(batch_size, -1), axis=1)
            block_argmax = jnp.argmax(scores.reshape(batch_size, -1), axis=1)
            improved = block_best > best_score
            best_score = jnp.where(improved, block_best, best_score)
            best_argmax = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax)

            # Trim padding rotations and store block weights
            actual_rot = min(rotation_block_size, n_rot - r0)
            block_probs = probs[:, :actual_rot, :]  # (batch, actual_rot, n_trans)
            batch_weights_blocks.append(np.asarray(block_probs.reshape(batch_size, -1)))

        # Concatenate blocks -> (batch_size, n_rot * n_trans)
        batch_weights = np.concatenate(batch_weights_blocks, axis=1)
        all_weights[start_idx:end_idx] = batch_weights
        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax)
        start_idx = end_idx

    return all_weights, hard_assignment
