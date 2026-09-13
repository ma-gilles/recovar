"""Materialized dense posterior reference for adaptive-oversampling tests.

Preserved from the former runtime ``compute_e_step_weights`` API after caller
review found only test consumers. This retains its own two-sweep orchestration
and complete host posterior; it still shares preprocessing and scoring kernels
with production, so it is not an independent reference for those kernels.
"""

import jax.numpy as jnp
import numpy as np

from recovar.core.configs import ForwardModelConfig
from recovar.em.dense.em_engine import _iter_dense_rotation_blocks
from recovar.em.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.helpers.fourier_window import make_fourier_window_spec
from recovar.em.helpers.half_spectrum import make_half_image_weights
from recovar.em.helpers.preprocessing import preprocess_batch as _preprocess_batch
from recovar.em.helpers.projection import compute_projections_block as _compute_projections_block
from recovar.em.scoring.scoring import _score_rotation_block, _update_logsumexp
from recovar.reconstruction import noise as noise_utils


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
    E-step but does NOT accumulate M-step statistics. Adaptive-oversampling
    tests use this materialized posterior to check significant-sample selection.

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
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()

    half_weights = make_half_image_weights(image_shape)
    precision_policy = DensePrecisionPolicy()

    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        include_recon_window=False,
    )
    projection_kwargs = window_spec.projection_kwargs()

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
            experiment_dataset,
            batch_data,
            ctf_params,
            noise_variance_half,
            translations,
            config,
            score_with_masked_images,
        )

        shifted_windowed = window_spec.score_values(shifted_half)
        ctf2_over_nv_windowed = window_spec.score_values(ctf2_over_nv_half)

        # Pass 1: streaming logsumexp
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size, dtype=precision_policy.normalization_real_dtype)

        for block in _iter_dense_rotation_blocks(rotations_padded, n_rot, n_blocks, rotation_block_size):
            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean,
                block.rotations,
                image_shape,
                volume_shape,
                disc_type,
                **projection_kwargs,
            )

            scores = _score_rotation_block(
                window_spec,
                shifted_score=shifted_windowed,
                batch_norm=batch_norm,
                score_weight=ctf2_over_nv_windowed,
                proj_half=proj_half_b,
                proj_abs2_half=proj_abs2_half_b,
                half_weights=half_weights,
                n_images=batch_size,
                n_trans=n_trans,
                image_shape=image_shape,
                volume_shape=volume_shape,
                score_mode="gaussian",
                precision_policy=precision_policy,
            )

            if block.actual_rot < rotation_block_size:
                mask = jnp.arange(rotation_block_size) < block.actual_rot
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)

            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)

        log_Z = max_s + jnp.log(sum_exp)

        # Pass 2: recompute scores and normalize to weights
        best_score = jnp.full(batch_size, -jnp.inf)
        best_argmax = jnp.zeros(batch_size, dtype=jnp.int32)
        batch_weights_blocks = []

        for block in _iter_dense_rotation_blocks(rotations_padded, n_rot, n_blocks, rotation_block_size):
            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean,
                block.rotations,
                image_shape,
                volume_shape,
                disc_type,
                **projection_kwargs,
            )

            scores = _score_rotation_block(
                window_spec,
                shifted_score=shifted_windowed,
                batch_norm=batch_norm,
                score_weight=ctf2_over_nv_windowed,
                proj_half=proj_half_b,
                proj_abs2_half=proj_abs2_half_b,
                half_weights=half_weights,
                n_images=batch_size,
                n_trans=n_trans,
                image_shape=image_shape,
                volume_shape=volume_shape,
                score_mode="gaussian",
                precision_policy=precision_policy,
            )

            if block.actual_rot < rotation_block_size:
                pmask = jnp.arange(rotation_block_size) < block.actual_rot
                scores = jnp.where(pmask[None, :, None], scores, -jnp.inf)

            # Normalize to probabilities
            probs = jnp.exp(scores - log_Z[:, None, None])

            # Track hard assignment
            block_best = jnp.max(scores.reshape(batch_size, -1), axis=1)
            block_argmax = jnp.argmax(scores.reshape(batch_size, -1), axis=1)
            improved = block_best > best_score
            best_score = jnp.where(improved, block_best, best_score)
            best_argmax = jnp.where(improved, block_argmax + block.r0 * n_trans, best_argmax)

            # Trim padding rotations and store block weights
            block_probs = probs[:, : block.actual_rot, :]
            batch_weights_blocks.append(np.asarray(block_probs.reshape(batch_size, -1)))

        # Concatenate blocks -> (batch_size, n_rot * n_trans)
        batch_weights = np.concatenate(batch_weights_blocks, axis=1)
        all_weights[start_idx:end_idx] = batch_weights
        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax)
        start_idx = end_idx

    return all_weights, hard_assignment

