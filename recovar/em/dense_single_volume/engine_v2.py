"""Optimized dense single-volume EM engine (v2).

Key optimizations over engine_fused.py:
1. Two-pass blockwise posterior normalization — no full (batch, n_rot, n_trans) tensor
2. JIT-compiled per-block E-step and M-step kernels — eliminates Python dispatch overhead
3. E-step scores computed twice (pass1: logsumexp stats, pass2: normalize+accumulate M-step)
   This trades 2x E-step compute for eliminating the giant residual tensor and
   enabling much larger rotation block sizes.

Translation handling (see docs/math/translation_handling_analysis.md):
   Both E-step and M-step use GEMM with explicit shifted-image copies.
   The n_trans factor inflates the GEMM matrices but enables 200× better
   data reuse vs the FFT alternative (1.5 GB vs 327 GB memory traffic).
   GEMM: 45 ms at 47 TFLOPS.  FFT: 1500 ms at 0.7 TFLOPS.  Same result.
   FFT wins only for single-rotation refinement (2× faster per rotation).
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


# ── JIT-compiled kernels ──────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(5, 6))
def _preprocess_batch(batch, ctf_params, noise_variance, translations, config, n_images, n_trans):
    """Preprocess one image batch: CTF weight + phase shifts. Returns flat shifted images."""
    CTF = config.compute_ctf(ctf_params)
    processed = config.process_fn(batch, apply_image_mask=False)
    ctf_weighted = processed * CTF / noise_variance
    shifted = core.batch_trans_translate_images(
        ctf_weighted, jnp.repeat(translations[None], n_images, axis=0), config.image_shape,
    )
    shifted_flat = shifted.reshape(n_images * n_trans, -1)
    batch_norm = jnp.linalg.norm(processed / jnp.sqrt(noise_variance), axis=-1, keepdims=True) ** 2
    ctf2_over_nv = CTF ** 2 / noise_variance
    return shifted_flat, batch_norm, ctf2_over_nv


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def _e_step_block_scores(shifted_flat, batch_norm, ctf2_over_nv, proj_block, proj_abs2_block,
                         n_images, n_trans, image_shape, volume_shape):
    """E-step for one rotation block: compute raw scores (not yet normalized)."""
    rot_block_size = proj_block.shape[0]
    # Cross-term: -2 Re(conj(Y) @ P^T)
    cross = -2.0 * (jnp.conj(shifted_flat) @ proj_block.T).real
    cross = cross.reshape(n_images, n_trans, rot_block_size) + batch_norm[:, None, :]
    cross = cross.swapaxes(1, 2)  # (n_images, rot_block_size, n_trans)
    # Norm-term
    norms = ctf2_over_nv @ proj_abs2_block.T  # (n_images, rot_block_size)
    residuals = cross + norms[..., None]
    # Convert to log-prob scores (higher = more probable)
    return -0.5 * residuals


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
def _m_step_block(shifted_flat, scores_block, log_Z, rotations_block, ctf2_over_nv,
                  Ft_y, Ft_ctf,
                  n_images, n_trans, image_shape, volume_shape):
    """Normalize scores to probs and accumulate M-step for one rotation block."""
    rot_block_size = rotations_block.shape[0]
    # Normalize
    probs = jnp.exp(scores_block - log_Z[:, None, None])
    # M-step GEMM: P @ shifted → (rot_block, image_size)
    P = probs.swapaxes(0, 1).reshape(rot_block_size, n_images * n_trans)
    summed = P @ shifted_flat
    summed_half = fourier_transform_utils.full_image_to_half_image(summed, image_shape)
    Ft_y = core.adjoint_slice_volume(
        summed_half, rotations_block, image_shape, volume_shape,
        "linear_interp", volume=Ft_y, half_image=True,
    )
    # CTF backprojection
    probs_sum_t = jnp.sum(probs, axis=-1)  # (n_images, rot_block)
    ctf_probs = probs_sum_t.T @ ctf2_over_nv
    ctf_half = fourier_transform_utils.full_image_to_half_image(ctf_probs, image_shape)
    Ft_ctf = core.adjoint_slice_volume(
        ctf_half, rotations_block, image_shape, volume_shape,
        "linear_interp", volume=Ft_ctf, half_image=True,
    )
    # Hard assignment contribution: argmax over this block
    block_best = jnp.max(scores_block.reshape(n_images, -1), axis=1)
    block_argmax = jnp.argmax(scores_block.reshape(n_images, -1), axis=1)
    return Ft_y, Ft_ctf, probs, block_best, block_argmax


def _compute_projections_block(volume, rotations_block, image_shape, volume_shape, disc_type):
    """Forward-slice one rotation block. Returns (proj, |proj|^2) on device."""
    proj = core.slice_volume(volume, rotations_block, image_shape, volume_shape, disc_type)
    proj_abs2 = jnp.abs(proj) ** 2
    return proj, proj_abs2


# ── FFT-based translation kernels ────────────────────────────────────────
#
# These compute the same E-step scores and M-step accumulations as the GEMM
# kernels above, but use cross-correlation via iFFT instead of explicit
# shifted-image copies. See docs/math/translation_handling_analysis.md.
#
# Trade-off:
#   - Per single rotation: FFT is ~2x faster (avoids creating shifted copies)
#   - Per batch of rotations: GEMM is ~35x faster (one matmul reuses all data)
#   - FFT can marginalize over ALL pixel translations (dense grid) at no extra cost
#   - FFT uses n_trans less memory (no shifted image copies)
#
# Use FFT path when: n_trans is large, or per-image refinement (few rotations
# per image), or dense translation marginalization is desired.


@partial(jax.jit, static_argnums=(4, 5))
def _e_step_block_scores_fft(images_ctf_weighted, batch_norm, ctf2_over_nv,
                              proj_block, n_images, image_shape):
    """FFT-based E-step for one rotation block.

    Instead of creating n_trans shifted copies + GEMM, computes the cross-
    correlation via element-wise product + batched iFFT for each rotation.
    Uses lax.scan over rotations to keep memory at (n_images, N) per step.

    Args:
        images_ctf_weighted: (n_images, N) complex — CTF*image/noise, NOT shifted.
        batch_norm: (n_images, 1) — ||image||^2 / noise.
        ctf2_over_nv: (n_images, N) — CTF^2 / noise.
        proj_block: (rot_block, N) complex — projections for this block.
        n_images: static int.
        image_shape: static (H, W).

    Returns:
        marginal_scores: (n_images, rot_block) — log Σ_t exp(-½ d_{i,r,t}),
            i.e. the log-marginal over ALL pixel translations per (image, rotation).
    """
    H, W = image_shape
    N = H * W
    conj_images = jnp.conj(images_ctf_weighted)

    # Norm term per rotation: Σ_k CTF²_i/σ² · |p_r|² — shape (n_images, rot_block)
    proj_abs2 = jnp.abs(proj_block) ** 2
    norm_term = ctf2_over_nv @ proj_abs2.T  # (n_images, rot_block)

    def process_one_rot(carry, proj_r):
        # proj_r: (N,) — one projection
        # Cross-correlation via iFFT: gives score at ALL N pixel translations
        product = conj_images * proj_r  # (n_images, N) broadcast
        xcorr = jnp.fft.ifft2(product.reshape(n_images, H, W))  # (n_images, H, W)
        cross_all = -2.0 * xcorr.reshape(n_images, N).real * N  # scale for unnorm FFT
        return carry, cross_all  # (n_images, N)

    # Scan over rotations: memory stays at (n_images, N) per step
    _, cross_all_rots = jax.lax.scan(process_one_rot, None, proj_block)
    # cross_all_rots: (rot_block, n_images, N)

    # Full residual: d_{i,r,t} = batch_norm_i + cross_{i,r,t} + norm_{i,r}
    # scores = -0.5 * d  (higher = more probable)
    scores = -0.5 * (cross_all_rots + batch_norm.T[:, :, None] + norm_term.T[:, :, None])
    # scores: (rot_block, n_images, N)

    # Marginalize over ALL translations: log Σ_t exp(score_{i,r,t})
    scores_max = jnp.max(scores, axis=-1)  # (rot_block, n_images)
    log_marginal = scores_max + jnp.log(
        jnp.sum(jnp.exp(scores - scores_max[..., None]), axis=-1)
    )  # (rot_block, n_images)

    return log_marginal.T  # (n_images, rot_block)


@partial(jax.jit, static_argnums=(5, 6))
def _m_step_block_fft(images_ctf_weighted, scores_block, log_Z,
                      rotations_block, ctf2_over_nv, proj_block,
                      Ft_y, Ft_ctf,
                      n_images, image_shape, volume_shape):
    """FFT-based M-step for one rotation block.

    Recomputes cross-correlation scores, normalizes to per-translation
    probabilities, collapses translations via DFT, then backprojects.

    This avoids the (n_images * n_trans, N) shifted-image matrix entirely.
    The M-step GEMM becomes (rot_block, n_images) @ (n_images, N) — no n_trans.

    Memory: (n_images, N) per scan step + (rot_block, N) for summed images.
    """
    H, W = image_shape
    N = H * W
    rot_block_size = rotations_block.shape[0]
    conj_images = jnp.conj(images_ctf_weighted)

    # Norm term (same as E-step)
    proj_abs2 = jnp.abs(proj_block) ** 2
    norm_term = ctf2_over_nv @ proj_abs2.T  # (n_images, rot_block)

    def process_one_rot(summed_acc, inputs):
        proj_r, rot_r, norm_r = inputs
        # proj_r: (N,), rot_r: (1, 3, 3), norm_r: (n_images,)

        # Recompute cross-correlation at all translations
        product = conj_images * proj_r  # (n_images, N)
        xcorr = jnp.fft.ifft2(product.reshape(n_images, H, W))
        cross_all = -2.0 * xcorr.reshape(n_images, N).real * N

        # Per-translation scores and probabilities for this rotation
        scores_irt = -0.5 * (cross_all + norm_r[:, None])  # drop batch_norm (cancels in softmax)
        # Normalize: probs_{i,t} = exp(score_{i,r,t}) / Z_i  where Z_i uses log_Z from E-step
        probs_irt = jnp.exp(scores_irt - log_Z[:, None])  # (n_images, N)

        # Translation-weighted image: Σ_t γ_{i,r,t} S_t^*(w_i)
        # In Fourier domain: S_t^* = conj(S_t) = phase(-t), so
        # Σ_t γ_{i,r,t} S_t^*(w_i) = w_i · conj(DFT(γ_{i,r,:}))
        # But γ is already in "pixel-position" space (indexed by translation pixel),
        # so DFT(γ) is the Fourier-domain kernel, and we want:
        #   a_{i,r} = FFT2(γ_{i,r,:} viewed as 2D) · w_i   (element-wise in Fourier)
        # Wait — γ lives at all N pixels (FFT gave us all translations), so:
        probs_2d = probs_irt.reshape(n_images, H, W)
        kernel = jnp.fft.fft2(probs_2d).reshape(n_images, N)  # (n_images, N)
        weighted_images = kernel * images_ctf_weighted  # (n_images, N)

        # Sum over images for this rotation: Σ_i a_{i,r}
        summed_r = jnp.sum(weighted_images, axis=0)  # (N,)

        # Probability marginal for Ft_ctf: Σ_t γ_{i,r,t}
        gamma_ir = jnp.sum(probs_irt, axis=-1)  # (n_images,)
        ctf_summed_r = gamma_ir @ ctf2_over_nv  # (N,)

        return summed_acc, (summed_r, ctf_summed_r)

    _, (summed_all, ctf_summed_all) = jax.lax.scan(
        process_one_rot, None,
        (proj_block, rotations_block[:, None, :, :].squeeze(1), norm_term.T),
    )
    # summed_all: (rot_block, N), ctf_summed_all: (rot_block, N)

    # Backproject
    summed_half = fourier_transform_utils.full_image_to_half_image(summed_all, image_shape)
    Ft_y = core.adjoint_slice_volume(
        summed_half, rotations_block, image_shape, volume_shape,
        "linear_interp", volume=Ft_y, half_image=True,
    )
    ctf_half = fourier_transform_utils.full_image_to_half_image(ctf_summed_all, image_shape)
    Ft_ctf = core.adjoint_slice_volume(
        ctf_half, rotations_block, image_shape, volume_shape,
        "linear_interp", volume=Ft_ctf, half_image=True,
    )

    return Ft_y, Ft_ctf


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
):
    """One EM iteration with JIT-fused two-pass blockwise normalization.

    Key properties:
    - Never materializes full (n_images, n_rot, n_trans) tensor
    - E-step scores computed twice (for normalization stats, then for M-step)
    - All per-block operations are JIT-compiled
    - Projections recomputed per block (faster than host→device transfer)
    """
    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    n_images = experiment_dataset.n_units
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type, process_fn=experiment_dataset.process_images,
    )

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

        # ── PREPROCESS (once per image batch) ──
        shifted_flat, batch_norm, ctf2_over_nv = _preprocess_batch(
            batch_data, ctf_params, noise_variance, translations, config,
            batch_size, n_trans,
        )

        # ── PASS 1: streaming logsumexp over rotation blocks ──
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_b, proj_abs2_b = _compute_projections_block(
                mean, rots_b, image_shape, volume_shape, disc_type)

            scores = _e_step_block_scores(
                shifted_flat, batch_norm, ctf2_over_nv, proj_b, proj_abs2_b,
                batch_size, n_trans, image_shape, volume_shape,
            )

            # Mask padding rotations (set their scores to -inf)
            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)

            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)

        log_Z = max_s + jnp.log(sum_exp)  # (batch_size,)

        # ── PASS 2: recompute scores, normalize, accumulate M-step ──
        best_score = jnp.full(batch_size, -jnp.inf)
        best_argmax = jnp.zeros(batch_size, dtype=jnp.int32)
        best_block_offset = jnp.zeros(batch_size, dtype=jnp.int32)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_b, proj_abs2_b = _compute_projections_block(
                mean, rots_b, image_shape, volume_shape, disc_type)

            scores = _e_step_block_scores(
                shifted_flat, batch_norm, ctf2_over_nv, proj_b, proj_abs2_b,
                batch_size, n_trans, image_shape, volume_shape,
            )

            # Mask padding
            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)

            Ft_y, Ft_ctf, probs, block_best, block_argmax = _m_step_block(
                shifted_flat, scores, log_Z, rots_b, ctf2_over_nv,
                Ft_y, Ft_ctf,
                batch_size, n_trans, image_shape, volume_shape,
            )

            # Track global hard assignment
            improved = block_best > best_score
            best_score = jnp.where(improved, block_best, best_score)
            best_argmax = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax)

        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax)
        start_idx = end_idx

    # ── SOLVE ──
    from recovar.reconstruction import relion_functions
    new_mean = relion_functions.post_process_from_filter(
        experiment_dataset, Ft_ctf, Ft_y, tau=mean_variance, disc_type=disc_type,
    ).reshape(-1)

    return new_mean, hard_assignment, Ft_y, Ft_ctf
