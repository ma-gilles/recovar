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

from recovar import core, utils
from recovar.core.configs import ForwardModelConfig
import recovar.core.fourier_transform_utils as fourier_transform_utils

from .types import MeanStats

logger = logging.getLogger(__name__)


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
    """
    H, W = image_shape
    w = 2.0 * jnp.ones((H, W // 2 + 1), dtype=jnp.float32)
    w = w.at[:, 0].set(1.0)    # packed column 0 = DC
    w = w.at[:, -1].set(1.0)   # packed column -1 = Nyquist
    return w.reshape(-1)        # (N_half,)


# -- JIT-compiled kernels ---------------------------------------------------

@partial(jax.jit, static_argnums=(5, 6))
def _preprocess_batch(batch, ctf_params, noise_variance, translations, config, n_images, n_trans):
    """Preprocess one image batch: CTF weight + phase shifts -> half-spectrum.

    Returns shifted images in half-spectrum layout (N_half per pixel),
    along with batch norms and CTF^2/noise for the norm-term computation.
    """
    CTF = config.compute_ctf(ctf_params)
    processed = config.process_fn(batch, apply_image_mask=False)
    ctf_weighted = processed * CTF / noise_variance
    # Phase shifts operate on full spectrum (need all frequencies for correct shift)
    shifted = core.batch_trans_translate_images(
        ctf_weighted, jnp.repeat(translations[None], n_images, axis=0), config.image_shape,
    )
    shifted_flat = shifted.reshape(n_images * n_trans, -1)
    # Convert to half-spectrum for all subsequent GEMMs
    shifted_half = fourier_transform_utils.full_image_to_half_image(shifted_flat, config.image_shape)

    batch_norm = jnp.linalg.norm(processed / jnp.sqrt(noise_variance), axis=-1, keepdims=True) ** 2
    ctf2_over_nv = CTF ** 2 / noise_variance
    # Also convert ctf2_over_nv to half for norm-term GEMM
    ctf2_over_nv_half = fourier_transform_utils.full_image_to_half_image(ctf2_over_nv, config.image_shape)
    return shifted_half, batch_norm, ctf2_over_nv_half


@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def _e_step_block_scores(shifted_half, batch_norm, ctf2_over_nv_half,
                         proj_half_weighted, proj_abs2_half,
                         half_weights,
                         n_images, n_trans, image_shape, volume_shape):
    """E-step for one rotation block using half-spectrum GEMMs.

    The cross-term GEMM uses weighted projections (half_weights absorbed into
    projections, precomputed once per rotation block) to recover the full inner
    product from half-spectrum data:

        cross[i,r] = -2 Re(conj(shifted_half) @ proj_half_weighted.T)

    The norm-term similarly uses half-weighted |proj|^2.

    Args:
        shifted_half: (n_images * n_trans, N_half) complex -- phase-shifted CTF-weighted images.
        batch_norm: (n_images, 1) float -- ||processed_image / sqrt(noise)||^2.
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
    cross = cross.reshape(n_images, n_trans, rot_block_size) + batch_norm[:, None, :]
    cross = cross.swapaxes(1, 2)  # (n_images, rot_block_size, n_trans)
    # Norm-term: sum_k w(k) * CTF^2/noise(k) * |proj(k)|^2
    norms = ctf2_over_nv_half @ proj_abs2_half.T  # (n_images, rot_block_size)
    residuals = cross + norms[..., None]
    # Convert to log-prob scores (higher = more probable)
    return -0.5 * residuals


@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10))
def _e_step_block_scores_windowed(shifted_windowed, batch_norm, ctf2_over_nv_windowed,
                                  proj_windowed_weighted, proj_abs2_windowed,
                                  half_weights_windowed,
                                  n_images, n_trans, n_windowed, image_shape, volume_shape):
    """E-step for one rotation block using windowed half-spectrum GEMMs.

    Same as _e_step_block_scores but operates on the windowed subset of the
    half-spectrum.  All inputs have inner dimension n_windowed instead of N_half.

    Args:
        shifted_windowed: (n_images * n_trans, n_windowed) complex
        batch_norm: (n_images, 1) float
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
    cross = cross.reshape(n_images, n_trans, rot_block_size) + batch_norm[:, None, :]
    cross = cross.swapaxes(1, 2)  # (n_images, rot_block_size, n_trans)
    # Norm-term on windowed subset
    norms = ctf2_over_nv_windowed @ proj_abs2_windowed.T  # (n_images, rot_block_size)
    residuals = cross + norms[..., None]
    return -0.5 * residuals


@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11))
def _m_step_block_windowed(shifted_windowed, scores_block, log_Z, rotations_block,
                           ctf2_over_nv_windowed,
                           Ft_y, Ft_ctf,
                           n_images, n_trans, n_windowed, image_shape, volume_shape):
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
    summed_windowed = P @ shifted_windowed   # (rot_block, n_windowed)

    # CTF backprojection on windowed subset
    probs_sum_t = jnp.sum(probs, axis=-1)  # (n_images, rot_block)
    ctf_probs_windowed = probs_sum_t.T @ ctf2_over_nv_windowed  # (rot_block, n_windowed)

    # Hard assignment contribution
    block_best = jnp.max(scores_block.reshape(n_images, -1), axis=1)
    block_argmax = jnp.argmax(scores_block.reshape(n_images, -1), axis=1)

    return Ft_y, Ft_ctf, probs, block_best, block_argmax, summed_windowed, ctf_probs_windowed


@partial(jax.jit, static_argnums=())
def _update_logsumexp(max_s, sum_exp, scores_block):
    """Streaming logsumexp update: accumulate normalization stats from one block."""
    scores_flat = scores_block.reshape(scores_block.shape[0], -1)  # (n_images, block*trans)
    block_max = jnp.max(scores_flat, axis=1)  # (n_images,)
    new_max = jnp.maximum(max_s, block_max)
    sum_exp = (sum_exp * jnp.exp(max_s - new_max)
               + jnp.sum(jnp.exp(scores_flat - new_max[:, None]), axis=1))
    return new_max, sum_exp


@partial(jax.jit, static_argnums=(7, 8, 9, 10))
def _m_step_block(shifted_half, scores_block, log_Z, rotations_block, ctf2_over_nv_half,
                  Ft_y, Ft_ctf,
                  n_images, n_trans, image_shape, volume_shape):
    """Normalize scores to probs and accumulate M-step for one rotation block.

    The M-step GEMM computes P @ shifted_half -> (rot_block, N_half).
    This is a weighted sum (not an inner product), so no Hermitian weights needed.
    The result goes directly to adjoint_slice_volume with half_image=True.
    """
    rot_block_size = rotations_block.shape[0]
    # Normalize
    probs = jnp.exp(scores_block - log_Z[:, None, None])
    # M-step GEMM: P @ shifted_half -> (rot_block, N_half)
    # This sums shifted half-images weighted by probabilities -- already in half layout!
    P = probs.swapaxes(0, 1).reshape(rot_block_size, n_images * n_trans)
    summed_half = P @ shifted_half   # (rot_block, N_half) -- directly in half layout
    # No full_image_to_half_image conversion needed!
    Ft_y = core.adjoint_slice_volume(
        summed_half, rotations_block, image_shape, volume_shape,
        "linear_interp", volume=Ft_y, half_image=True,
    )
    # CTF backprojection: probs_sum_t @ ctf2_over_nv_half -> (rot_block, N_half)
    probs_sum_t = jnp.sum(probs, axis=-1)  # (n_images, rot_block)
    ctf_probs_half = probs_sum_t.T @ ctf2_over_nv_half  # (rot_block, N_half)
    # No full_image_to_half_image conversion needed!
    Ft_ctf = core.adjoint_slice_volume(
        ctf_probs_half, rotations_block, image_shape, volume_shape,
        "linear_interp", volume=Ft_ctf, half_image=True,
    )
    # Hard assignment contribution: argmax over this block
    block_best = jnp.max(scores_block.reshape(n_images, -1), axis=1)
    block_argmax = jnp.argmax(scores_block.reshape(n_images, -1), axis=1)
    return Ft_y, Ft_ctf, probs, block_best, block_argmax


def _compute_projections_block(volume, rotations_block, image_shape, volume_shape, disc_type):
    """Forward-slice one rotation block in half-spectrum layout.

    Returns (proj_half, |proj_half|^2) on device, both in half-spectrum layout.
    """
    proj_half = core.slice_volume(
        volume, rotations_block, image_shape, volume_shape, disc_type, half_image=True
    )
    proj_abs2_half = jnp.abs(proj_half) ** 2
    return proj_half, proj_abs2_half


def run_em_v2(
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
    """
    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    n_images = experiment_dataset.n_units
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape

    H, W = image_shape
    n_half = H * (W // 2 + 1)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type, process_fn=experiment_dataset.process_images,
    )

    # Precompute half-spectrum weights once
    half_weights = make_half_image_weights(image_shape)

    # Precompute window indices if current_size is set
    use_window = current_size is not None and current_size < image_shape[0]
    if use_window:
        from .fourier_window import make_fourier_window_indices_np
        window_indices_np, n_windowed = make_fourier_window_indices_np(image_shape, current_size)
        window_indices = jnp.asarray(window_indices_np)
        half_weights_windowed = half_weights[window_indices]
        logger.info(
            "Fourier windowing: current_size=%d, n_windowed=%d / n_half=%d (%.1f%% reduction)",
            current_size, n_windowed, n_half, 100.0 * (1.0 - n_windowed / n_half),
        )
    else:
        window_indices = None
        n_windowed = n_half

    # Pad rotations to multiple of block size for fixed shapes
    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate([
            rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))
        ], axis=0)
    else:
        rotations_padded = rotations

    # Initialize accumulators
    Ft_y = jnp.zeros(experiment_dataset.volume_size, dtype=experiment_dataset.dtype)
    Ft_ctf = jnp.zeros(experiment_dataset.volume_size, dtype=experiment_dataset.dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)

    image_indices = np.arange(n_images)
    start_idx = 0

    for (batch_data, _, _, ctf_params, _, _, indices) in experiment_dataset.iter_batches(
        image_batch_size, indices=image_indices, by_image=False,
    ):
        batch_size = len(indices)
        end_idx = start_idx + batch_size
        batch_data = jnp.asarray(batch_data)

        # -- PREPROCESS (once per image batch) -- returns half-spectrum --
        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            batch_data, ctf_params, noise_variance, translations, config,
            batch_size, n_trans,
        )

        # -- WINDOW gather (if active) --
        if use_window:
            shifted_windowed = shifted_half[:, window_indices]
            ctf2_over_nv_windowed = ctf2_over_nv_half[:, window_indices]
        else:
            shifted_windowed = shifted_half
            ctf2_over_nv_windowed = ctf2_over_nv_half

        # -- PASS 1: streaming logsumexp over rotation blocks --
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean, rots_b, image_shape, volume_shape, disc_type)

            if use_window:
                # Gather windowed subset from projections
                proj_windowed_b = proj_half_b[:, window_indices]
                proj_abs2_windowed_b = proj_abs2_half_b[:, window_indices]
                proj_windowed_weighted_b = proj_windowed_b * half_weights_windowed
                proj_abs2_windowed_weighted_b = proj_abs2_windowed_b * half_weights_windowed

                scores = _e_step_block_scores_windowed(
                    shifted_windowed, batch_norm, ctf2_over_nv_windowed,
                    proj_windowed_weighted_b, proj_abs2_windowed_weighted_b,
                    half_weights_windowed,
                    batch_size, n_trans, n_windowed, image_shape, volume_shape,
                )
            else:
                # Full half-spectrum path (Phase 1 behavior)
                proj_half_weighted_b = proj_half_b * half_weights
                proj_abs2_weighted_b = proj_abs2_half_b * half_weights

                scores = _e_step_block_scores(
                    shifted_half, batch_norm, ctf2_over_nv_half,
                    proj_half_weighted_b, proj_abs2_weighted_b, half_weights,
                    batch_size, n_trans, image_shape, volume_shape,
                )

            # Mask padding rotations (set their scores to -inf)
            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)

            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)

        log_Z = max_s + jnp.log(sum_exp)  # (batch_size,)

        # -- PASS 2: recompute scores, normalize, accumulate M-step --
        best_score = jnp.full(batch_size, -jnp.inf)
        best_argmax = jnp.zeros(batch_size, dtype=jnp.int32)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean, rots_b, image_shape, volume_shape, disc_type)

            if use_window:
                # Gather windowed subset
                proj_windowed_b = proj_half_b[:, window_indices]
                proj_abs2_windowed_b = proj_abs2_half_b[:, window_indices]
                proj_windowed_weighted_b = proj_windowed_b * half_weights_windowed
                proj_abs2_windowed_weighted_b = proj_abs2_windowed_b * half_weights_windowed

                scores = _e_step_block_scores_windowed(
                    shifted_windowed, batch_norm, ctf2_over_nv_windowed,
                    proj_windowed_weighted_b, proj_abs2_windowed_weighted_b,
                    half_weights_windowed,
                    batch_size, n_trans, n_windowed, image_shape, volume_shape,
                )
            else:
                proj_half_weighted_b = proj_half_b * half_weights
                proj_abs2_weighted_b = proj_abs2_half_b * half_weights

                scores = _e_step_block_scores(
                    shifted_half, batch_norm, ctf2_over_nv_half,
                    proj_half_weighted_b, proj_abs2_weighted_b, half_weights,
                    batch_size, n_trans, image_shape, volume_shape,
                )

            # Mask padding
            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)

            if use_window:
                # Windowed M-step: GEMM at reduced dimension, then scatter back
                (Ft_y, Ft_ctf, probs, block_best, block_argmax,
                 summed_windowed, ctf_probs_windowed) = _m_step_block_windowed(
                    shifted_windowed, scores, log_Z, rots_b, ctf2_over_nv_windowed,
                    Ft_y, Ft_ctf,
                    batch_size, n_trans, n_windowed, image_shape, volume_shape,
                )
                # Scatter windowed GEMM results back to full half-spectrum
                rot_block_size_actual = rots_b.shape[0]
                summed_half = jnp.zeros((rot_block_size_actual, n_half), dtype=summed_windowed.dtype)
                summed_half = summed_half.at[:, window_indices].set(summed_windowed)
                ctf_probs_half = jnp.zeros((rot_block_size_actual, n_half), dtype=ctf_probs_windowed.dtype)
                ctf_probs_half = ctf_probs_half.at[:, window_indices].set(ctf_probs_windowed)

                # Adjoint slice at full resolution
                Ft_y = core.adjoint_slice_volume(
                    summed_half, rots_b, image_shape, volume_shape,
                    "linear_interp", volume=Ft_y, half_image=True,
                )
                Ft_ctf = core.adjoint_slice_volume(
                    ctf_probs_half, rots_b, image_shape, volume_shape,
                    "linear_interp", volume=Ft_ctf, half_image=True,
                )
            else:
                Ft_y, Ft_ctf, probs, block_best, block_argmax = _m_step_block(
                    shifted_half, scores, log_Z, rots_b, ctf2_over_nv_half,
                    Ft_y, Ft_ctf,
                    batch_size, n_trans, image_shape, volume_shape,
                )

            # Track global hard assignment
            improved = block_best > best_score
            best_score = jnp.where(improved, block_best, best_score)
            best_argmax = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax)

        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax)
        start_idx = end_idx

    # -- SOLVE --
    from recovar.reconstruction import relion_functions
    new_mean = relion_functions.post_process_from_filter(
        experiment_dataset, Ft_ctf, Ft_y, tau=mean_variance, disc_type=disc_type,
    ).reshape(-1)

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
        experiment_dataset, disc_type=disc_type, process_fn=experiment_dataset.process_images,
    )

    half_weights = make_half_image_weights(image_shape)

    use_window = current_size is not None and current_size < image_shape[0]
    if use_window:
        from .fourier_window import make_fourier_window_indices_np
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
        rotations_padded = np.concatenate([
            rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))
        ], axis=0)
    else:
        rotations_padded = rotations

    # Allocate output weights array on host
    all_weights = np.empty((n_images, n_rot * n_trans), dtype=np.float32)
    hard_assignment = np.empty(n_images, dtype=np.int32)

    image_indices = np.arange(n_images)
    start_idx = 0

    for (batch_data, _, _, ctf_params, _, _, indices) in experiment_dataset.iter_batches(
        image_batch_size, indices=image_indices, by_image=False,
    ):
        batch_size = len(indices)
        end_idx = start_idx + batch_size
        batch_data = jnp.asarray(batch_data)

        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            batch_data, ctf_params, noise_variance, translations, config,
            batch_size, n_trans,
        )

        if use_window:
            shifted_windowed = shifted_half[:, window_indices]
            ctf2_over_nv_windowed = ctf2_over_nv_half[:, window_indices]
        else:
            shifted_windowed = shifted_half
            ctf2_over_nv_windowed = ctf2_over_nv_half

        # Pass 1: streaming logsumexp
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean, rots_b, image_shape, volume_shape, disc_type)

            if use_window:
                proj_windowed_b = proj_half_b[:, window_indices]
                proj_abs2_windowed_b = proj_abs2_half_b[:, window_indices]
                proj_windowed_weighted_b = proj_windowed_b * half_weights_windowed
                proj_abs2_windowed_weighted_b = proj_abs2_windowed_b * half_weights_windowed
                scores = _e_step_block_scores_windowed(
                    shifted_windowed, batch_norm, ctf2_over_nv_windowed,
                    proj_windowed_weighted_b, proj_abs2_windowed_weighted_b,
                    half_weights_windowed,
                    batch_size, n_trans, n_windowed, image_shape, volume_shape,
                )
            else:
                proj_half_weighted_b = proj_half_b * half_weights
                proj_abs2_weighted_b = proj_abs2_half_b * half_weights
                scores = _e_step_block_scores(
                    shifted_half, batch_norm, ctf2_over_nv_half,
                    proj_half_weighted_b, proj_abs2_weighted_b, half_weights,
                    batch_size, n_trans, image_shape, volume_shape,
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
                mean, rots_b, image_shape, volume_shape, disc_type)

            if use_window:
                proj_windowed_b = proj_half_b[:, window_indices]
                proj_abs2_windowed_b = proj_abs2_half_b[:, window_indices]
                proj_windowed_weighted_b = proj_windowed_b * half_weights_windowed
                proj_abs2_windowed_weighted_b = proj_abs2_windowed_b * half_weights_windowed
                scores = _e_step_block_scores_windowed(
                    shifted_windowed, batch_norm, ctf2_over_nv_windowed,
                    proj_windowed_weighted_b, proj_abs2_windowed_weighted_b,
                    half_weights_windowed,
                    batch_size, n_trans, n_windowed, image_shape, volume_shape,
                )
            else:
                proj_half_weighted_b = proj_half_b * half_weights
                proj_abs2_weighted_b = proj_abs2_half_b * half_weights
                scores = _e_step_block_scores(
                    shifted_half, batch_norm, ctf2_over_nv_half,
                    proj_half_weighted_b, proj_abs2_weighted_b, half_weights,
                    batch_size, n_trans, image_shape, volume_shape,
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
