"""Per-image low-frequency scorer for paper-faithful BnB.

Each image carries its own (axis_rotations_i, shift_cells_i) candidate
set; this module computes recovar's Gaussian score for each image at
low-frequency radius L. The implementation is one image at a time —
calls into the existing ``helpers.scoring._score_rotation_block`` kernel
with batch_size=1 per image, n_trans=n_shift_i, n_rot_block=min(n_axis_i,
rotation_block_size).

The single-image-at-a-time loop trades batching efficiency for
correctness on per-image-ragged candidate sets. JAX's compilation cache
makes the per-image cost dominated by GEMM/projection FLOPs, not Python
overhead, once the per-shape JIT is warm.

A bucketed (pad-to-max) variant is a future optimisation; for the first
working version we pay the per-image kernel-launch cost.
"""

from __future__ import annotations

import logging
import time

import jax.numpy as jnp
import numpy as np

from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.fourier_window import FourierWindowSpec
from recovar.em.dense_single_volume.helpers.half_spectrum import make_half_image_weights
from recovar.em.dense_single_volume.helpers.preprocessing import preprocess_batch
from recovar.em.dense_single_volume.helpers.projection import compute_projections_block
from recovar.em.dense_single_volume.helpers.scoring import _score_rotation_block

from .frequency import make_bnb_low_window_spec
from .per_image_state import PerImageBnBPoseState

logger = logging.getLogger(__name__)


def _to_half_noise(noise_variance, image_shape):
    from .hierarchical_support import _to_half_noise as _impl
    return _impl(noise_variance, image_shape)


def score_per_image_at_low_freq(
    experiment_dataset,
    mean: jnp.ndarray,
    noise_variance: jnp.ndarray,
    state: PerImageBnBPoseState,
    image_indices: np.ndarray,
    *,
    L: int,
    disc_type: str = "linear_interp",
    image_batch_size: int = 187,
) -> list[np.ndarray]:
    """Score each image's per-image-ragged (axis, shift) candidates at radius L.

    Returns
    -------
    scores : list of np.ndarray
        ``scores[i]`` has shape ``(state.axis_cells[i].shape[0],
        state.shift_cells[i].shape[0])`` and contains recovar's Gaussian
        score for image ``i`` at every joint candidate position. Inactive
        candidates (sample_mask == False) get score = -inf.
    """
    image_shape = experiment_dataset.image_shape
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    image_indices = np.asarray(image_indices, dtype=np.int32)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    low_window: FourierWindowSpec = make_bnb_low_window_spec(image_shape, L, n_half)
    half_weights = make_half_image_weights(image_shape)
    precision_policy = DensePrecisionPolicy()
    nv_half = _to_half_noise(noise_variance, image_shape)

    n_images = int(image_indices.shape[0])
    scores: list[np.ndarray | None] = [None] * n_images

    # Iterate the dataset in image batches so we use the dataset's own
    # streaming. Inside each batch we still have per-image preprocessing
    # because each image has its own shift list.
    image_idx_to_local = {int(g): i for i, g in enumerate(image_indices)}

    t0 = time.time()
    for batch in experiment_dataset.iter_batches(
        image_batch_size, indices=image_indices, by_image=False,
    ):
        batch_data = batch[0]
        ctf_params = batch[3]
        batch_global_indices = np.asarray(batch[5], dtype=np.int32)
        batch_size = int(jnp.asarray(batch_data).shape[0])

        for b_local in range(batch_size):
            global_id = int(batch_global_indices[b_local])
            i_local = image_idx_to_local.get(global_id)
            if i_local is None:
                continue

            shifts_i = jnp.asarray(state.shift_cells[i_local])
            axis_rots_i = np.asarray(state.axis_rotations[i_local], dtype=np.float32)
            n_axis_i = int(axis_rots_i.shape[0])
            n_shift_i = int(shifts_i.shape[0])
            if n_axis_i == 0 or n_shift_i == 0:
                scores[i_local] = np.zeros((max(n_axis_i, 0), max(n_shift_i, 0)), dtype=np.float32)
                continue

            # Preprocess this single image with its own shift grid.
            single_batch = jnp.asarray(batch_data[b_local : b_local + 1])
            single_ctf = jnp.asarray(ctf_params[b_local : b_local + 1])
            shifted_half, batch_norm, ctf2_over_nv_half = preprocess_batch(
                experiment_dataset, single_batch, single_ctf,
                nv_half, shifts_i, config, False,
            )
            shifted_windowed = low_window.score_values(shifted_half)
            ctf2_windowed = low_window.score_values(ctf2_over_nv_half)

            # Score axis rotations (rotation block loop in case n_axis_i is
            # larger than the engine's compile-friendly chunk).
            scores_i = np.empty((n_axis_i, n_shift_i), dtype=np.float32)
            chunk = max(1, min(n_axis_i, 1024))
            for r0 in range(0, n_axis_i, chunk):
                r1 = min(r0 + chunk, n_axis_i)
                rot_block = axis_rots_i[r0:r1]
                proj_half, proj_abs2_half = compute_projections_block(
                    mean, rot_block, image_shape,
                    experiment_dataset.volume_shape, disc_type,
                    return_abs2=True,
                )
                block_scores = _score_rotation_block(
                    low_window,
                    shifted_score=shifted_windowed,
                    batch_norm=batch_norm,
                    score_weight=ctf2_windowed,
                    proj_half=proj_half,
                    proj_abs2_half=proj_abs2_half,
                    half_weights=half_weights,
                    n_images=1,
                    n_trans=n_shift_i,
                    image_shape=image_shape,
                    volume_shape=experiment_dataset.volume_shape,
                    score_mode="gaussian",
                    precision_policy=precision_policy,
                )
                # block_scores: (1, chunk, n_shift_i)
                bs = np.asarray(block_scores)[0, : (r1 - r0), :]
                scores_i[r0:r1, :] = bs.astype(np.float32, copy=False)

            # Apply sample_mask: inactive candidates get -inf.
            mask_i = state.sample_mask[i_local]
            scores_i = np.where(mask_i, scores_i, -np.inf)
            scores[i_local] = scores_i

    elapsed = time.time() - t0
    logger.debug(
        "score_per_image_at_low_freq: L=%d, n_images=%d, %.2fs (%.1f images/s)",
        L, n_images, elapsed, n_images / max(elapsed, 1e-6),
    )

    # Sanity: every image should have a score array now.
    missing = [i for i, s in enumerate(scores) if s is None]
    if missing:
        raise RuntimeError(
            f"score_per_image_at_low_freq: {len(missing)} images missing "
            f"from the data iterator (first: image_indices[{missing[0]}]="
            f"{int(image_indices[missing[0]])})",
        )
    return [np.asarray(s) for s in scores]
