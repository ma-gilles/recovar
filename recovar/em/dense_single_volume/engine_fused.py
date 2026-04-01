"""Fused dense single-volume EM engine.

Implements the loop nesting from docs/math/dense_single_volume_cleanup_brief.md
Section 6: E-step and M-step are fused per image batch, preprocessing is shared,
and rotation blocks are streamed.

Key differences from engine.py / E_with_precompute + M_with_precompute:
1. Image preprocessing (CTF weighting, phase shifts) done ONCE per batch
2. E-step processes rotation blocks within image batch (no full prob tensor in host RAM)
3. M-step immediately follows E-step per batch (reuses shifted images)
4. Projection cache can be precomputed OR recomputed per block
"""

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from recovar import core, utils
from recovar.core.configs import ForwardModelConfig
import recovar.core.fourier_transform_utils as fourier_transform_utils

from .types import MeanStats

logger = logging.getLogger(__name__)


def _precompute_all_projections(volume, rotations, image_shape, volume_shape, disc_type, batch_size):
    """Precompute P_r mu for all rotations, stored on host."""
    n_rot = rotations.shape[0]
    image_size = int(np.prod(image_shape))
    projections = np.zeros((n_rot, image_size), dtype=np.complex64)
    for rot_indices in utils.index_batch_iter(n_rot, batch_size):
        projections[rot_indices] = np.asarray(core.slice_volume(
            volume, rotations[rot_indices], image_shape, volume_shape, disc_type
        ))
    return projections


@jax.jit
def _softmax_normalize(residuals):
    """Numerically stable softmax over all axes except first (image axis)."""
    all_but_first = tuple(range(1, residuals.ndim))
    residuals = 0.5 * residuals
    residuals -= jnp.min(residuals, axis=all_but_first, keepdims=True)
    exp_res = jnp.exp(-residuals)
    return exp_res / jnp.sum(exp_res, axis=all_but_first, keepdims=True)


def _e_step_rotation_block(shifted_images_flat, batch_norm, proj_block, proj_abs2_block,
                           ctf2_over_nv, n_images, n_trans):
    """Compute E-step residuals for one rotation block.

    Returns residuals: (n_images, n_rot_block, n_trans) float32.
    """
    n_rot_block = proj_block.shape[0]
    n_shifted = n_images * n_trans

    # Cross-term: -2 Re(conj(shifted_images) @ proj.T)
    cross = -2.0 * (jnp.conj(shifted_images_flat) @ proj_block.T).real
    cross = cross.reshape(n_images, n_trans, n_rot_block) + batch_norm[:, None, :]
    cross = cross.swapaxes(1, 2)  # (n_images, n_rot_block, n_trans)

    # Norm-term: CTF^2/nv @ |proj|^2.T → (n_images, n_rot_block)
    norms = ctf2_over_nv @ proj_abs2_block.T  # (n_images, n_rot_block)

    return cross + norms[..., None]


def _m_step_rotation_block(shifted_images_flat, probs_block, rotations_block,
                           ctf2_over_nv, image_shape, volume_shape, Ft_y, Ft_ctf):
    """Accumulate M-step stats for one rotation block.

    probs_block: (n_images, n_rot_block, n_trans) float32.
    """
    n_rot_block = rotations_block.shape[0]
    n_shifted = shifted_images_flat.shape[0]

    # Probability-weighted image sum: P @ shifted → (n_rot_block, image_size)
    P = probs_block.swapaxes(0, 1).reshape(n_rot_block, n_shifted)
    summed_images = P @ shifted_images_flat

    # Backproject Ft_y
    summed_half = fourier_transform_utils.full_image_to_half_image(summed_images, image_shape)
    Ft_y = core.adjoint_slice_volume(
        summed_half, rotations_block, image_shape, volume_shape,
        "linear_interp", volume=Ft_y, half_image=True,
    )

    # CTF-weighted backprojection for Ft_ctf
    probs_sum_trans = jnp.sum(probs_block, axis=-1)  # (n_images, n_rot_block)
    CTF_probs = probs_sum_trans.T @ ctf2_over_nv  # (n_rot_block, image_size)
    CTF_probs_half = fourier_transform_utils.full_image_to_half_image(CTF_probs, image_shape)
    Ft_ctf = core.adjoint_slice_volume(
        CTF_probs_half, rotations_block, image_shape, volume_shape,
        "linear_interp", volume=Ft_ctf, half_image=True,
    )

    return Ft_y, Ft_ctf


def run_fused_em_iteration(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    rotations,
    translations,
    disc_type: str,
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    precompute_projections: bool = True,
):
    """One EM iteration with fused E+M per image batch and rotation blocking.

    Args:
        experiment_dataset: CryoEM dataset.
        mean: Current volume estimate, (volume_size,) complex.
        mean_variance: Prior variance.
        noise_variance: Noise level, (image_size,) float.
        rotations: (n_rot, 3, 3) rotation matrices.
        translations: (n_trans, 2) in-plane translations.
        disc_type: Discretization type.
        image_batch_size: Images per batch.
        rotation_block_size: Rotations per block for E-step and M-step.
        precompute_projections: If True, precompute all projections once.
            If False, recompute per rotation block per image batch.

    Returns:
        (new_mean, hard_assignment, Ft_y, Ft_ctf)
    """
    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    n_images = experiment_dataset.n_units
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    image_size = experiment_dataset.image_size

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type, process_fn=experiment_dataset.process_images,
    )

    # Precompute projections (host-side cache)
    t0 = time.time()
    if precompute_projections:
        proj_host = _precompute_all_projections(
            mean, rotations, image_shape, volume_shape, disc_type,
            batch_size=min(rotation_block_size, n_rot),
        )
        proj_abs2_host = np.abs(proj_host) ** 2
        logger.info("Projection precompute: %.1fs, %.1f GB",
                     time.time() - t0, proj_host.nbytes / 1e9)

    # Initialize accumulators
    Ft_y = jnp.zeros(experiment_dataset.volume_size, dtype=experiment_dataset.dtype)
    Ft_ctf = jnp.zeros(experiment_dataset.volume_size, dtype=experiment_dataset.dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)

    # Rotation block indices
    rot_blocks = list(utils.index_batch_iter(n_rot, rotation_block_size))

    image_indices = np.arange(n_images)
    start_idx = 0

    for (batch, _, _, ctf_params, _, _, indices) in experiment_dataset.iter_batches(
        image_batch_size, indices=image_indices, by_image=False,
    ):
        batch_size = len(indices)
        end_idx = start_idx + batch_size

        # ── PREPROCESSING (done ONCE, shared between E and M) ──
        batch = jnp.asarray(batch)
        CTF = config.compute_ctf(ctf_params)
        processed = config.process_fn(batch, apply_image_mask=False) * CTF / noise_variance
        shifted_images = core.batch_trans_translate_images(
            processed, jnp.repeat(translations[None], batch_size, axis=0), image_shape,
        )
        shifted_images_flat = shifted_images.reshape(batch_size * n_trans, image_size)

        # Batch norm for E-step: ||y_i||^2 / sigma^2
        batch_norm = jnp.linalg.norm(
            config.process_fn(batch, apply_image_mask=False) / jnp.sqrt(noise_variance),
            axis=-1, keepdims=True,
        ) ** 2

        # CTF^2 / noise_variance (reused in E-step norms and M-step)
        ctf2_over_nv = CTF ** 2 / noise_variance

        # ── E-STEP: accumulate residuals over rotation blocks ──
        residuals = np.empty((batch_size, n_rot, n_trans), dtype=np.float32)

        for block_idx, rot_idx in enumerate(rot_blocks):
            rot_idx = np.asarray(rot_idx)
            r0, r1 = rot_idx[0], rot_idx[-1] + 1

            if precompute_projections:
                proj_block = jnp.asarray(proj_host[r0:r1])
                proj_abs2_block = jnp.asarray(proj_abs2_host[r0:r1])
            else:
                proj_block = core.slice_volume(
                    mean, rotations[r0:r1], image_shape, volume_shape, disc_type)
                proj_abs2_block = jnp.abs(proj_block) ** 2

            residuals[:, r0:r1, :] = _e_step_rotation_block(
                shifted_images_flat, batch_norm, proj_block, proj_abs2_block,
                ctf2_over_nv, batch_size, n_trans,
            )

        # ── NORMALIZE ──
        probs = np.asarray(_softmax_normalize(jnp.asarray(residuals)))
        hard_assignment[start_idx:end_idx] = np.argmax(
            probs.reshape(batch_size, -1), axis=-1)

        # ── M-STEP: accumulate over rotation blocks (reusing shifted_images_flat) ──
        for block_idx, rot_idx in enumerate(rot_blocks):
            rot_idx = np.asarray(rot_idx)
            r0, r1 = rot_idx[0], rot_idx[-1] + 1

            Ft_y, Ft_ctf = _m_step_rotation_block(
                shifted_images_flat,
                jnp.asarray(probs[:, r0:r1, :]),
                rotations[r0:r1],
                ctf2_over_nv,
                image_shape, volume_shape,
                Ft_y, Ft_ctf,
            )

        start_idx = end_idx

    # ── SOLVE ──
    from recovar.reconstruction import relion_functions
    new_mean = relion_functions.post_process_from_filter(
        experiment_dataset, Ft_ctf, Ft_y, tau=mean_variance, disc_type=disc_type,
    ).reshape(-1)

    return new_mean, hard_assignment, Ft_y, Ft_ctf
