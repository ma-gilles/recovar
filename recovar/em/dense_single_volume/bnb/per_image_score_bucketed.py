"""Bucketed pad-to-max per-image low-frequency scorer for paper-faithful BnB.

The per-image loop in ``per_image_score.score_per_image_at_low_freq`` calls
into JAX once per image, paying ~10-100 ms of kernel-launch overhead per
image. At 100k images that's 17-170 minutes per BnB stage of pure overhead
(measured: 26 min for stage 0 at 100k 256², while the FLOP-only floor is
~10 min).

This module amortises by grouping a batch of images into "buckets" with a
shared padded shape ``(bucket_n_images, bucket_n_axis, bucket_n_shift,
n_half)``. Per-image rotations and shifts are padded to the bucket maxima;
sample_mask marks the genuine-vs-padded entries. One JAX kernel call per
bucket replaces hundreds of per-image calls.

Padded values for inactive entries:
- axis_rotations[pad] = identity matrix (won't crash the projector)
- shift_cells[pad] = zero shift (no phase change)
- sample_mask[pad] = False (forces -inf score)

The output is identical to the per-image loop within numerical noise.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers.fourier_window import FourierWindowSpec
from recovar.em.dense_single_volume.helpers.half_spectrum import make_half_image_weights
from recovar.em.dense_single_volume.helpers.preprocessing import preprocess_batch
from recovar.em.dense_single_volume.helpers.projection import compute_projections_block

from .frequency import make_bnb_low_window_spec
from .per_image_score import _to_half_noise
from .per_image_state import PerImageBnBPoseState

logger = logging.getLogger(__name__)


def _bucket_size(n: int, quantum: int) -> int:
    """Round up to the next multiple of ``quantum`` (min 1)."""
    if n <= 0:
        return max(1, int(quantum))
    return int(np.ceil(n / quantum)) * int(quantum)


def _group_into_buckets(
    image_indices_in_batch: np.ndarray,
    n_axis_per_image: np.ndarray,
    n_shift_per_image: np.ndarray,
    *,
    axis_quantum: int = 64,
    shift_quantum: int = 8,
    max_per_bucket: int = 187,
) -> list[tuple[int, int, np.ndarray]]:
    """Group images so each bucket shares a padded ``(n_axis_pad, n_shift_pad)``.

    Returns a list of ``(n_axis_pad, n_shift_pad, image_indices_in_bucket)``.
    """
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k, gid in enumerate(image_indices_in_batch):
        ax = _bucket_size(int(n_axis_per_image[k]), axis_quantum)
        sh = _bucket_size(int(n_shift_per_image[k]), shift_quantum)
        groups[(ax, sh)].append(int(gid))

    buckets: list[tuple[int, int, np.ndarray]] = []
    for (ax, sh), gids in groups.items():
        gids_arr = np.asarray(gids, dtype=np.int32)
        for start in range(0, gids_arr.size, max_per_bucket):
            buckets.append((ax, sh, gids_arr[start : start + max_per_bucket]))
    return buckets


@partial(jax.jit, static_argnames=("n_low",))
def _score_bucket_kernel(
    proj_padded,            # (B, R, L_low) complex
    proj_abs2_padded,       # (B, R, L_low) float
    shifted_windowed,       # (B, T, L_low) complex (image × shift × pixel)
    ctf2_over_nv_windowed,  # (B, L_low) float
    half_weights_low,       # (L_low,) float
    valid_axis,             # (B, R) bool
    valid_shift,            # (B, T) bool
    n_low: int,
):
    """Score (B, T, R) Gaussian residual under per-image-ragged padding.

    cross[b, t, r] = -2 Re(sum_l conj(shifted[b, t, l]) * proj[b, r, l] * h_l)
    norms[b, r]    = sum_l ctf2[b, l] * proj_abs2[b, r, l] * h_l
    score[b, t, r] = -0.5 * (cross + norms[..., None])

    Returns (B, R, T) float32 with -inf at invalid positions.
    """
    proj_weighted = proj_padded * half_weights_low[None, None, :]
    cross = -2.0 * jnp.einsum(
        "btl,brl->btr",
        jnp.conj(shifted_windowed),
        proj_weighted,
        precision=jax.lax.Precision.HIGHEST,
    ).real
    proj_abs2_weighted = proj_abs2_padded * half_weights_low[None, None, :]
    norms = jnp.einsum(
        "bl,brl->br",
        ctf2_over_nv_windowed,
        proj_abs2_weighted,
        precision=jax.lax.Precision.HIGHEST,
    )
    residuals = cross + norms[:, None, :]
    scores = -0.5 * residuals  # (B, T, R)

    valid = valid_axis[:, None, :] & valid_shift[:, :, None]  # (B, T, R)
    scores = jnp.where(valid, scores, -jnp.inf)
    return scores.swapaxes(1, 2)  # (B, R, T)


def score_per_image_at_low_freq_bucketed(
    experiment_dataset,
    mean: jnp.ndarray,
    noise_variance: jnp.ndarray,
    state: PerImageBnBPoseState,
    image_indices: np.ndarray,
    *,
    L: int,
    disc_type: str = "linear_interp",
    image_batch_size: int = 187,
    axis_quantum: int = 64,
    shift_quantum: int = 8,
) -> list[np.ndarray]:
    """Bucketed pad-to-max version of per-image low-frequency scoring.

    Same return semantics as ``score_per_image_at_low_freq``: a list of
    per-image ``(n_axis_i, n_shift_i)`` score arrays with -inf at inactive
    candidates. Faster by amortising JAX kernel launches across buckets of
    images with similar (rounded) candidate counts.
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
    n_low = int(low_window.n_score)
    half_weights = make_half_image_weights(image_shape)
    half_weights_low_np = np.asarray(half_weights)[
        np.asarray(low_window.score_indices_np if low_window.use_window else np.arange(n_half))
    ]
    half_weights_low = jnp.asarray(half_weights_low_np, dtype=jnp.float32)

    nv_half = _to_half_noise(noise_variance, image_shape)
    n_images = int(image_indices.shape[0])
    scores: list[np.ndarray | None] = [None] * n_images
    image_idx_to_local = {int(g): i for i, g in enumerate(image_indices)}

    n_axis_global = np.asarray(
        [state.axis_cells[i].shape[0] for i in range(state.n_images)],
        dtype=np.int32,
    )
    n_shift_global = np.asarray(
        [state.shift_cells[i].shape[0] for i in range(state.n_images)],
        dtype=np.int32,
    )

    t0 = time.time()
    buckets_seen = 0
    for batch in experiment_dataset.iter_batches(
        image_batch_size, indices=image_indices, by_image=False,
    ):
        batch_data = batch[0]
        ctf_params = batch[3]
        batch_global = np.asarray(batch[5], dtype=np.int32)
        batch_size = int(jnp.asarray(batch_data).shape[0])

        # Local indices into image_indices for this batch.
        batch_local_idx = np.asarray(
            [image_idx_to_local[int(g)] for g in batch_global if int(g) in image_idx_to_local],
            dtype=np.int32,
        )
        if batch_local_idx.size == 0:
            continue

        n_axis_batch = n_axis_global[batch_local_idx]
        n_shift_batch = n_shift_global[batch_local_idx]

        bucket_specs = _group_into_buckets(
            batch_local_idx, n_axis_batch, n_shift_batch,
            axis_quantum=axis_quantum, shift_quantum=shift_quantum,
            max_per_bucket=batch_size,
        )

        for n_axis_pad, n_shift_pad, bucket_local_ids in bucket_specs:
            bucket_size = bucket_local_ids.size
            if bucket_size == 0:
                continue

            # --- pad rotations ---
            padded_rotations = np.tile(
                np.eye(3, dtype=np.float32),
                (bucket_size, n_axis_pad, 1, 1),
            )
            valid_axis = np.zeros((bucket_size, n_axis_pad), dtype=bool)
            for b, ilocal in enumerate(bucket_local_ids):
                axis_i = state.axis_rotations[int(ilocal)]
                n_real = int(axis_i.shape[0])
                padded_rotations[b, :n_real] = axis_i
                valid_axis[b, :n_real] = True

            # --- pad shifts ---
            padded_shifts = np.zeros((bucket_size, n_shift_pad, 2), dtype=np.float32)
            valid_shift = np.zeros((bucket_size, n_shift_pad), dtype=bool)
            for b, ilocal in enumerate(bucket_local_ids):
                sh_i = state.shift_cells[int(ilocal)]
                n_real_s = int(sh_i.shape[0])
                padded_shifts[b, :n_real_s] = sh_i
                valid_shift[b, :n_real_s] = True

            # --- preprocess each image with its own padded shifts ---
            # We can't share preprocessing across images (each has its own
            # shift list), but we can do them all in one batched pass via
            # a Python loop (cheaper than the score kernel).
            shifted_per_image = np.empty(
                (bucket_size, n_shift_pad, n_low), dtype=np.complex64,
            )
            ctf2_per_image = np.empty(
                (bucket_size, n_low), dtype=np.float32,
            )
            for b, ilocal in enumerate(bucket_local_ids):
                global_id = int(image_indices[int(ilocal)])
                # Find this image in the batch_data.
                batch_pos = int(np.where(batch_global == global_id)[0][0])
                single_img = jnp.asarray(batch_data[batch_pos : batch_pos + 1])
                single_ctf = jnp.asarray(ctf_params[batch_pos : batch_pos + 1])

                shifts_padded_jnp = jnp.asarray(padded_shifts[b])  # (n_shift_pad, 2)
                shifted_half, _, ctf2_over_nv_half = preprocess_batch(
                    experiment_dataset, single_img, single_ctf,
                    nv_half, shifts_padded_jnp, config, False,
                )
                # shifted_half: (1*n_shift_pad, n_half) — flattened by preprocess
                shifted_half = shifted_half.reshape(n_shift_pad, n_half)
                shifted_windowed = low_window.score_values(shifted_half)
                ctf2_windowed = low_window.score_values(ctf2_over_nv_half)
                shifted_per_image[b] = np.asarray(shifted_windowed, dtype=np.complex64)
                ctf2_per_image[b] = np.asarray(ctf2_windowed[0], dtype=np.float32)

            # --- project: per image, project bucket_n_axis_pad rotations ---
            # Total projections: bucket_size × n_axis_pad. Done by looping
            # over images so each gets its own padded rotation set.
            proj_per_image = np.empty((bucket_size, n_axis_pad, n_low), dtype=np.complex64)
            proj_abs2_per_image = np.empty((bucket_size, n_axis_pad, n_low), dtype=np.float32)
            for b in range(bucket_size):
                proj_half_b, proj_abs2_half_b = compute_projections_block(
                    mean, padded_rotations[b], image_shape,
                    experiment_dataset.volume_shape, disc_type,
                    return_abs2=True,
                )
                proj_per_image[b] = np.asarray(low_window.score_values(proj_half_b), dtype=np.complex64)
                proj_abs2_per_image[b] = np.asarray(low_window.score_values(proj_abs2_half_b), dtype=np.float32)

            # --- one batched kernel call for the whole bucket ---
            scores_bucket = _score_bucket_kernel(
                jnp.asarray(proj_per_image),
                jnp.asarray(proj_abs2_per_image),
                jnp.asarray(shifted_per_image),
                jnp.asarray(ctf2_per_image),
                half_weights_low,
                jnp.asarray(valid_axis),
                jnp.asarray(valid_shift),
                n_low=n_low,
            )
            scores_bucket = np.asarray(scores_bucket, dtype=np.float32)
            buckets_seen += 1

            # --- distribute scores back to per-image ---
            for b, ilocal in enumerate(bucket_local_ids):
                n_real_a = int(state.axis_cells[int(ilocal)].shape[0])
                n_real_s = int(state.shift_cells[int(ilocal)].shape[0])
                # Apply per-image sample_mask (kernel only knows about
                # padded-vs-real, not about state.sample_mask).
                bucket_score = scores_bucket[b, :n_real_a, :n_real_s]
                mask_i = state.sample_mask[int(ilocal)]
                scores[int(ilocal)] = np.where(mask_i, bucket_score, -np.inf).astype(np.float32)

    elapsed = time.time() - t0
    logger.info(
        "score_per_image_at_low_freq_bucketed: L=%d, n_images=%d, %d buckets, %.2fs (%.0f img/s)",
        L, n_images, buckets_seen, elapsed, n_images / max(elapsed, 1e-6),
    )

    missing = [i for i, s in enumerate(scores) if s is None]
    if missing:
        raise RuntimeError(
            f"score_per_image_at_low_freq_bucketed: {len(missing)} images missing "
            f"(first: image_indices[{missing[0]}]={int(image_indices[missing[0]])})",
        )
    return [np.asarray(s) for s in scores]
