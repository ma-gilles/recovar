"""Adjoint (BPref) accumulation of the sparse bucketed pass 2.

The chunked adjoint-block accumulation over hypothesis rows, the RELION
x-half per-particle launch order, the active flat-row adjoint and the
projection-gather budget split of compact-pair buckets.
``sparse_pass2_bucketed`` accumulates every bucket through these owners.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers import bpref_diagnostics
from recovar.em.dense_single_volume.helpers.adjoint import adjoint_slice_volume_half as _adjoint_slice_volume_half
from recovar.em.dense_single_volume.helpers.adjoint import (
    adjoint_slice_volume_windowed as _adjoint_slice_volume_windowed,
)
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_flag
from recovar.em.dense_single_volume.helpers.sparse_bucket_arrays import _compact_bucket_size_for_class
from recovar.em.dense_single_volume.helpers.sparse_pass2_budget import (
    _complex_counterpart_real_dtype,
    _dtype_itemsize,
    _optional_positive_int_env,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_compact_pair_sums import (
    _active_flat_gather_chunk_rows,
    _select_active_flat_rows,
    _select_active_flat_values,
)
from recovar.em.dense_single_volume.local_layout import _exact_bucket_rotation_size

logger = logging.getLogger(__name__)


_SPARSE_KCLASS_GROUP_PAIR_BUCKETS_BY_ROTATION_SIGNATURE_ENV = (
    "RECOVAR_SPARSE_KCLASS_GROUP_PAIR_BUCKETS_BY_ROTATION_SIGNATURE"
)


_RELION_X_HALF_BP_PARTICLE_POOL_SIZE_ENV = (
    "RECOVAR_K1_RELION_X_HALF_BP_PARTICLE_POOL_SIZE"
)


_active_flat_gather_chunk_log_keys: set[tuple[str, int, int, int, int]] = set()


_adjoint_block_chunk_log_keys: set[tuple[str, int, int, int, int]] = set()


def _flat_block_row_bytes(flat_block) -> int:
    if flat_block is None or len(flat_block.shape) == 0:
        return 1
    n_pixels = int(flat_block.shape[1]) if len(flat_block.shape) > 1 else 1
    return max(1, n_pixels * _dtype_itemsize(flat_block.dtype))


def _adjoint_block_chunk_rows(flat_block, *, max_block_bytes: int) -> int:
    if flat_block is None:
        return 1
    row_bytes = _flat_block_row_bytes(flat_block)
    return max(1, int(max_block_bytes) // row_bytes)


def _accumulate_relion_x_half_per_particle_launches(
    values,
    ctf_values,
    rotations,
    actual_counts,
    y_volume,
    ctf_volume,
    *,
    window_indices,
    image_shape,
    volume_shape,
    disc_type,
    half_volume,
    max_r,
    log_label_prefix: str,
    winner_take_all: bool = False,
    strict_particle_order: bool = False,
):
    """Accumulate one particle-owned orientation grid per FFI launch.

    This preserves a RELION-like particle launch boundary and local orientation
    order.  In soft-posterior mode it remains a causal arm rather than strict
    RELION closure because RECOVAR reduces translations before this boundary.
    The optional fused-atomics diagnostic interleaves each neighbor's real,
    imaginary, and weight atomics in one kernel.
    """

    diagnostic_fused_atomics = bpref_diagnostics.relion_x_half_bp_fused_atomics_enabled()
    # Fresh K=1 --firstiter_cc contributes exactly one winning hypothesis per
    # particle. Unlike later soft-posterior iterations, no translation
    # reduction changes the contributor stream. RELION still dispatches each
    # MPI pool concurrently through per-thread CUDA streams, so the optional
    # pool diagnostic below changes only that outer launch grouping.
    production_firstiter_fused_atomics = bool(winner_take_all and strict_particle_order)
    use_fused_atomics = bool(
        diagnostic_fused_atomics or production_firstiter_fused_atomics
    )
    if not winner_take_all and not strict_particle_order:
        logger.warning(
            "RECOVAR soft-posterior per-particle x-half causal arm enabled; "
            "this does not claim RELION hypothesis-arithmetic closure"
        )
    if use_fused_atomics:
        import recovar.cuda_backproject as cuda_backproject

        if (
            diagnostic_fused_atomics
            and not production_firstiter_fused_atomics
            and not bpref_diagnostics.relion_x_half_bp_per_particle_launch_enabled()
        ):
            raise RuntimeError(
                "RELION fused-atomics diagnostic requires "
                "RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH=1"
            )
        if (
            diagnostic_fused_atomics
            and not production_firstiter_fused_atomics
            and not cuda_backproject.relion_x_half_bp_block_topology_enabled()
        ):
            raise RuntimeError(
                "RELION fused-atomics diagnostic requires "
                "RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY=1"
            )
        if production_firstiter_fused_atomics:
            logger.info(
                "STRICT-PARITY: fresh K=1 firstiter-CC uses RELION fused "
                "real/imaginary/weight atomics for particle-owned launches "
                "(label=%s)",
                log_label_prefix,
            )
        else:
            logger.info(
                "RELION x-half diagnostic: fused real/imaginary/weight atomics enabled "
                "for particle-owned launches (label=%s)",
                log_label_prefix,
            )

        # The fused CUDA target is specialized by the BPref accumulator
        # dtype. Scoring and M-step precision are independently gated, so
        # convert reduced rows once at this boundary rather than assuming that
        # scoring precision matches accumulator precision.
        fused_complex_dtype = (
            jnp.complex128
            if y_volume.dtype == jnp.dtype(jnp.complex128)
            else jnp.complex64
        )
        values = jnp.asarray(values, dtype=fused_complex_dtype)
        ctf_values = jnp.asarray(ctf_values, dtype=ctf_volume.dtype)

    actual_counts = np.asarray(actual_counts, dtype=np.int64)
    if values.shape[:2] != rotations.shape[:2] or ctf_values.shape[:2] != values.shape[:2]:
        raise ValueError("per-particle x-half diagnostic requires matching (particle, rotation) axes")
    if actual_counts.shape != (int(values.shape[0]),):
        raise ValueError(
            f"per-particle x-half actual_counts shape mismatch: {actual_counts.shape} vs {(int(values.shape[0]),)}"
        )
    particle_pool_size = (
        _optional_positive_int_env(_RELION_X_HALF_BP_PARTICLE_POOL_SIZE_ENV) or 1
    )
    if particle_pool_size > 1:
        if not (winner_take_all and strict_particle_order and use_fused_atomics):
            raise RuntimeError(
                "RELION particle-pool diagnostic requires fresh K=1 winner-take-all, "
                "strict particle order, and fused atomics"
            )
        logger.info(
            "STRICT-PARITY diagnostic: grouping consecutive fresh K=1 BPref "
            "particles into RELION-sized launch pools (pool_size=%d, particles=%d, "
            "label=%s)",
            particle_pool_size,
            int(values.shape[0]),
            log_label_prefix,
        )
    else:
        logger.info(
            "RELION x-half diagnostic: one adjoint launch per particle "
            "(particles=%d, rotations min/median/max=%d/%d/%d, label=%s)",
            int(values.shape[0]),
            int(actual_counts.min()) if actual_counts.size else 0,
            int(np.median(actual_counts)) if actual_counts.size else 0,
            int(actual_counts.max()) if actual_counts.size else 0,
            log_label_prefix,
        )
    for pool_start in range(0, actual_counts.size, particle_pool_size):
        pool_stop = min(pool_start + particle_pool_size, actual_counts.size)
        value_rows = []
        ctf_rows = []
        rotation_rows = []
        for particle_index in range(pool_start, pool_stop):
            count = int(actual_counts[particle_index])
            if count <= 0:
                continue
            particle_slice = (
                slice(particle_index, particle_index + 1),
                slice(0, count),
            )
            value_rows.append(values[particle_slice].reshape(count, values.shape[-1]))
            ctf_rows.append(
                ctf_values[particle_slice].reshape(count, ctf_values.shape[-1])
            )
            rotation_rows.append(rotations[particle_slice].reshape(count, 3, 3))
        if not value_rows:
            continue
        particle_values = jnp.concatenate(value_rows, axis=0)
        particle_ctf_values = jnp.concatenate(ctf_rows, axis=0)
        particle_rotations = jnp.concatenate(rotation_rows, axis=0)
        if use_fused_atomics:
            if disc_type != "linear_interp" or not half_volume:
                raise RuntimeError(
                    "RELION fused-atomics diagnostic requires linear interpolation and a half-volume accumulator"
                )
            y_volume, ctf_volume = cuda_backproject.relion_fused_x_half_backproject_indexed(
                y_volume,
                ctf_volume,
                particle_values,
                particle_ctf_values,
                window_indices,
                particle_rotations,
                image_shape=image_shape,
                volume_shape=volume_shape,
                max_r=max_r,
            )
        else:
            y_volume = _adjoint_slice_volume_windowed(
                particle_values,
                window_indices,
                particle_rotations,
                y_volume,
                image_shape,
                volume_shape,
                disc_type,
                True,
                half_volume,
                max_r,
                True,
            )
            ctf_volume = _adjoint_slice_volume_windowed(
                particle_ctf_values,
                window_indices,
                particle_rotations,
                ctf_volume,
                image_shape,
                volume_shape,
                disc_type,
                True,
                half_volume,
                max_r,
                True,
            )
    return y_volume, ctf_volume


def _accumulate_adjoint_block_chunked(
    flat_block,
    flat_rotations,
    volume,
    *,
    window_indices=None,
    use_windowed_adjoint: bool,
    image_shape,
    volume_shape,
    disc_type,
    half_image: bool,
    half_volume: bool,
    max_r,
    relion_x_half: bool,
    max_block_bytes: int,
    log_label: str,
):
    """Accumulate adjoint-slice rows in capped chunks for pathological tails."""

    if flat_block is None:
        return volume
    n_rows = int(flat_block.shape[0])
    if n_rows == 0:
        return volume
    max_rows = _adjoint_block_chunk_rows(flat_block, max_block_bytes=max_block_bytes)
    if n_rows <= max_rows:
        if use_windowed_adjoint:
            return _adjoint_slice_volume_windowed(
                flat_block,
                window_indices,
                flat_rotations,
                volume,
                image_shape,
                volume_shape,
                disc_type,
                half_image,
                half_volume,
                max_r,
                relion_x_half,
            )
        return _adjoint_slice_volume_half(
            flat_block,
            flat_rotations,
            volume,
            image_shape,
            volume_shape,
            disc_type,
            half_image,
            half_volume,
        )

    n_chunks = (n_rows + max_rows - 1) // max_rows
    n_pixels = int(flat_block.shape[1]) if len(flat_block.shape) > 1 else 1
    log_key = (str(log_label), n_rows, n_pixels, max_rows, int(max_block_bytes))
    if log_key not in _adjoint_block_chunk_log_keys:
        _adjoint_block_chunk_log_keys.add(log_key)
        logger.info(
            "Sparse pass-2 adjoint block chunking: %s rows=%d pixels=%d max_rows=%d chunks=%d max_block_bytes=%.2f GiB",
            log_label,
            n_rows,
            n_pixels,
            max_rows,
            n_chunks,
            float(max_block_bytes) / float(1024**3),
        )

    for start in range(0, n_rows, max_rows):
        stop = min(start + max_rows, n_rows)
        if use_windowed_adjoint:
            volume = _adjoint_slice_volume_windowed(
                flat_block[start:stop],
                window_indices,
                flat_rotations[start:stop],
                volume,
                image_shape,
                volume_shape,
                disc_type,
                half_image,
                half_volume,
                max_r,
                relion_x_half,
            )
        else:
            volume = _adjoint_slice_volume_half(
                flat_block[start:stop],
                flat_rotations[start:stop],
                volume,
                image_shape,
                volume_shape,
                disc_type,
                half_image,
                half_volume,
            )
    return volume


def _accumulate_active_flat_rows_adjoint_chunked(
    values,
    ctf_values,
    flat_rotations,
    active_indices,
    active_mask,
    y_volume,
    ctf_volume,
    *,
    window_indices=None,
    use_windowed_adjoint: bool,
    image_shape,
    volume_shape,
    disc_type,
    half_image: bool,
    half_volume: bool,
    max_r,
    relion_x_half: bool,
    max_block_bytes: int,
    log_label_prefix: str,
):
    """Gather active M-step rows in bounded chunks before adjoint accumulation."""

    n_rows = int(active_indices.size)
    if n_rows == 0:
        return y_volume, ctf_volume
    max_rows = _active_flat_gather_chunk_rows(
        values,
        ctf_values,
        flat_rotations,
        max_block_bytes=max_block_bytes,
    )
    if n_rows > max_rows:
        n_pixels = int(values.shape[-1]) if len(values.shape) > 1 else 1
        n_chunks = (n_rows + max_rows - 1) // max_rows
        log_key = (str(log_label_prefix), n_rows, n_pixels, max_rows, int(max_block_bytes or 0))
        if log_key not in _active_flat_gather_chunk_log_keys:
            _active_flat_gather_chunk_log_keys.add(log_key)
            logger.info(
                "Sparse pass-2 active flat-row gather chunking: %s rows=%d pixels=%d max_rows=%d "
                "chunks=%d max_block_bytes=%.2f GiB",
                log_label_prefix,
                n_rows,
                n_pixels,
                max_rows,
                n_chunks,
                int(max_block_bytes or 0) / float(1024**3),
            )

    for start in range(0, n_rows, max_rows):
        stop = min(start + max_rows, n_rows)
        chunk_indices = active_indices[start:stop]
        chunk_mask = None if active_mask is None else active_mask[start:stop]
        flat_summed, active_flat_rotations = _select_active_flat_rows(
            values,
            flat_rotations,
            chunk_indices,
            chunk_mask,
        )
        flat_ctf_probs = _select_active_flat_values(
            ctf_values,
            chunk_indices,
            chunk_mask,
        )
        y_volume = _accumulate_adjoint_block_chunked(
            flat_summed,
            active_flat_rotations,
            y_volume,
            window_indices=window_indices,
            use_windowed_adjoint=use_windowed_adjoint,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type=disc_type,
            half_image=half_image,
            half_volume=half_volume,
            max_r=max_r,
            relion_x_half=relion_x_half,
            max_block_bytes=max_block_bytes,
            log_label=f"{log_label_prefix}-y",
        )
        ctf_volume = _accumulate_adjoint_block_chunked(
            flat_ctf_probs,
            active_flat_rotations,
            ctf_volume,
            window_indices=window_indices,
            use_windowed_adjoint=use_windowed_adjoint,
            image_shape=image_shape,
            volume_shape=volume_shape,
            disc_type=disc_type,
            half_image=half_image,
            half_volume=half_volume,
            max_r=max_r,
            relion_x_half=relion_x_half,
            max_block_bytes=max_block_bytes,
            log_label=f"{log_label_prefix}-ctf",
        )
    return y_volume, ctf_volume


def _projection_gather_bytes_per_rotation_row(
    *,
    n_score_pixels: int,
    n_recon_pixels: int,
    projection_complex_dtype,
    include_recon_noise: bool,
) -> int:
    complex_bytes = _dtype_itemsize(projection_complex_dtype)
    real_dtype = _complex_counterpart_real_dtype(projection_complex_dtype)
    row_bytes = int(n_score_pixels) * complex_bytes
    if include_recon_noise:
        row_bytes += int(n_recon_pixels) * (complex_bytes + _dtype_itemsize(real_dtype))
    return max(1, int(row_bytes))


def _projection_rotation_chunk_size(
    *,
    batch_size: int,
    n_score_pixels: int,
    n_recon_pixels: int,
    projection_complex_dtype,
    include_recon_noise: bool,
    max_gather_bytes: int | None,
    max_projected_rotations: int | None,
) -> int | None:
    """Return a per-image rotation chunk cap for bounded projection gathers."""

    if max_gather_bytes is None:
        return max_projected_rotations
    max_gather_bytes = int(max_gather_bytes)
    if max_gather_bytes <= 0:
        return max_projected_rotations
    row_bytes = _projection_gather_bytes_per_rotation_row(
        n_score_pixels=n_score_pixels,
        n_recon_pixels=n_recon_pixels,
        projection_complex_dtype=projection_complex_dtype,
        include_recon_noise=include_recon_noise,
    )
    max_flat_rows = max(1, max_gather_bytes // row_bytes)
    max_rows = max(1, max_flat_rows // max(1, int(batch_size)))
    if max_projected_rotations is not None:
        max_rows = min(max_rows, max(1, int(max_projected_rotations)))
    return max_rows


def _split_compact_pair_buckets_by_projection_gather_budget(
    compact_buckets,
    per_image_inputs_by_class,
    *,
    n_score_pixels: int,
    n_recon_pixels: int,
    projection_complex_dtype,
    max_gather_bytes: int | None,
    max_dense_mstep_bytes: int | None = None,
    n_fine_trans: int | None = None,
    prob_dtype=np.float64,
    max_prepare_images_per_microbatch: int | None = None,
    rotation_block_size_for_quantization: int,
):
    """Split compact-pair execution buckets by gather/prep/dense-M-step memory."""

    group_by_rotation_signature = parse_env_flag(
        _SPARSE_KCLASS_GROUP_PAIR_BUCKETS_BY_ROTATION_SIGNATURE_ENV,
        default=True,
    )
    if (
        not group_by_rotation_signature
        and max_gather_bytes is None
        and max_prepare_images_per_microbatch is None
        and max_dense_mstep_bytes is None
    ):
        return list(compact_buckets)
    ungrouped_bucket_count = len(compact_buckets)
    original_pair_bucket_max_images: dict[int, int] = {}
    for bucket in compact_buckets:
        pair_bucket_size = int(bucket["pair_bucket_size"])
        original_pair_bucket_max_images[pair_bucket_size] = max(
            original_pair_bucket_max_images.get(pair_bucket_size, 0),
            len(bucket["image_indices"]),
        )
    if group_by_rotation_signature:
        image_indices_by_pair_bucket: dict[int, list[int]] = {}
        for bucket in compact_buckets:
            image_indices_by_pair_bucket.setdefault(
                int(bucket["pair_bucket_size"]),
                [],
            ).extend(
                np.asarray(bucket["image_indices"], dtype=np.int64).tolist(),
            )
        signature_floors_by_pair_bucket: dict[int, int] = {}
        for pair_bucket_size, pair_image_indices in image_indices_by_pair_bucket.items():
            percentile_index = max(
                0,
                int(np.ceil(0.95 * len(pair_image_indices))) - 1,
            )
            signature_floors_by_pair_bucket[pair_bucket_size] = sorted(
                max(
                    _exact_bucket_rotation_size(
                        int(per_image_inputs["oversampled_rots"][image_idx].shape[0]),
                        rotation_block_size_for_quantization,
                    )
                    for per_image_inputs in per_image_inputs_by_class
                )
                for image_idx in pair_image_indices
            )[percentile_index]
        grouped_image_indices: dict[tuple[int, tuple[int, ...]], list[int]] = {}
        for bucket in compact_buckets:
            pair_bucket_size = int(bucket["pair_bucket_size"])
            signature_floor = signature_floors_by_pair_bucket[pair_bucket_size]
            for image_idx in np.asarray(bucket["image_indices"], dtype=np.int64).tolist():
                exact_rotation_signature = tuple(
                    _exact_bucket_rotation_size(
                        int(per_image_inputs["oversampled_rots"][image_idx].shape[0]),
                        rotation_block_size_for_quantization,
                    )
                    for per_image_inputs in per_image_inputs_by_class
                )
                outlier_bucket_size = max(exact_rotation_signature)
                rotation_signature = (
                    (outlier_bucket_size,) * len(exact_rotation_signature)
                    if outlier_bucket_size > signature_floor
                    else (signature_floor,) * len(exact_rotation_signature)
                )
                grouped_image_indices.setdefault(
                    (pair_bucket_size, rotation_signature),
                    [],
                ).append(int(image_idx))
        compact_buckets = [
            {
                "pair_bucket_size": pair_bucket_size,
                "image_indices": np.asarray(image_indices, dtype=np.int64),
                "class_bucket_sizes": _rotation_signature,
            }
            for (pair_bucket_size, _rotation_signature), image_indices in grouped_image_indices.items()
        ]
        logger.info(
            "Sparse fused K-class compact-pair rotation-signature grouping: "
            "buckets %d -> %d (default on; set %s=0 to opt out)",
            ungrouped_bucket_count,
            len(compact_buckets),
            _SPARSE_KCLASS_GROUP_PAIR_BUCKETS_BY_ROTATION_SIGNATURE_ENV,
        )
    max_gather_bytes = None if max_gather_bytes is None else int(max_gather_bytes)
    if max_gather_bytes is not None and max_gather_bytes <= 0:
        max_gather_bytes = None
    max_dense_mstep_bytes = None if max_dense_mstep_bytes is None else int(max_dense_mstep_bytes)
    if max_dense_mstep_bytes is not None and max_dense_mstep_bytes <= 0:
        max_dense_mstep_bytes = None
    if max_dense_mstep_bytes is not None and n_fine_trans is None:
        raise ValueError("n_fine_trans is required when max_dense_mstep_bytes is set")
    n_fine_trans_int = 0 if n_fine_trans is None else int(n_fine_trans)
    if max_dense_mstep_bytes is not None and n_fine_trans_int <= 0:
        raise ValueError("n_fine_trans must be positive when max_dense_mstep_bytes is set")
    max_prepare_images = (
        None
        if max_prepare_images_per_microbatch is None
        else max(1, int(max_prepare_images_per_microbatch))
    )
    if max_gather_bytes is None and max_prepare_images is None and max_dense_mstep_bytes is None:
        return list(compact_buckets)
    row_bytes = _projection_gather_bytes_per_rotation_row(
        n_score_pixels=n_score_pixels,
        n_recon_pixels=n_recon_pixels,
        projection_complex_dtype=projection_complex_dtype,
        include_recon_noise=True,
    )
    prob_item_bytes = _dtype_itemsize(prob_dtype)
    split_buckets = []
    split_bucket_count = 0
    original_bucket_count = len(compact_buckets)
    original_max_images = 0
    split_max_images = 0
    max_dense_bytes_per_image = 0
    for bucket in compact_buckets:
        image_indices = np.asarray(bucket["image_indices"], dtype=np.int64)
        original_max_images = max(original_max_images, int(image_indices.size))
        if image_indices.size <= 1:
            split_buckets.append(bucket)
            split_max_images = max(split_max_images, int(image_indices.size))
            continue
        max_images = int(image_indices.size)
        max_images = min(
            max_images,
            original_pair_bucket_max_images[int(bucket["pair_bucket_size"])],
        )
        if max_gather_bytes is not None or max_dense_mstep_bytes is not None:
            if "class_bucket_sizes" in bucket:
                max_class_bucket_size = max(int(value) for value in bucket["class_bucket_sizes"])
            else:
                max_class_bucket_size = max(
                    _compact_bucket_size_for_class(
                        bucket,
                        per_image_inputs,
                        rotation_block_size_for_quantization,
                    )
                    for per_image_inputs in per_image_inputs_by_class
                )
        if max_gather_bytes is not None:
            per_image_bytes = max(1, int(max_class_bucket_size) * row_bytes)
            max_images = min(max_images, max(1, max_gather_bytes // per_image_bytes))
        if max_dense_mstep_bytes is not None:
            dense_bytes_per_image = max(1, int(max_class_bucket_size) * n_fine_trans_int * prob_item_bytes)
            max_dense_bytes_per_image = max(max_dense_bytes_per_image, dense_bytes_per_image)
            max_images = min(max_images, max(1, max_dense_mstep_bytes // dense_bytes_per_image))
        if max_prepare_images is not None:
            max_images = min(max_images, max_prepare_images)
        if image_indices.size <= max_images:
            split_buckets.append(bucket)
            split_max_images = max(split_max_images, int(image_indices.size))
            continue
        split_bucket_count += 1
        for start in range(0, image_indices.size, max_images):
            chunk = image_indices[start : start + max_images]
            chunk_bucket = dict(bucket)
            chunk_bucket["image_indices"] = np.asarray(chunk, dtype=np.int64)
            split_buckets.append(chunk_bucket)
            split_max_images = max(split_max_images, int(chunk.size))
    if split_bucket_count:
        logger.info(
            "Sparse fused K-class compact-pair execution split: buckets %d -> %d; "
            "split_source_buckets=%d, max_images %d -> %d, max_gather_bytes=%s, "
            "row_bytes=%d, max_dense_mstep_bytes=%s, dense_prob_item_bytes=%d, "
            "max_dense_mstep_bytes_per_image=%s, max_prepare_images=%s",
            original_bucket_count,
            len(split_buckets),
            split_bucket_count,
            original_max_images,
            split_max_images,
            "unset" if max_gather_bytes is None else f"{max_gather_bytes / float(1024**3):.2f} GiB",
            row_bytes,
            "unset" if max_dense_mstep_bytes is None else f"{max_dense_mstep_bytes / float(1024**3):.2f} GiB",
            prob_item_bytes,
            "unset" if max_dense_mstep_bytes is None else f"{max_dense_bytes_per_image / float(1024**3):.2f} GiB",
            "unset" if max_prepare_images is None else str(max_prepare_images),
        )
    return split_buckets
