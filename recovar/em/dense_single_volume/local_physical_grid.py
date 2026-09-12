"""RELION physical-particle-grid accumulation of the exact local EM engine.

RELION's physical particle ordering for BPref accumulation, the VDAM variant
with its per-particle launches and the source-faithful particle chunking
that bounds them.
"""

from __future__ import annotations

import functools
import os

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers import vdam_replay

EXACT_LOCAL_SOURCE_BPREF_PARTICLE_CHUNK_SIZE_ENV = (
    "RECOVAR_EXACT_LOCAL_SOURCE_BPREF_PARTICLE_CHUNK_SIZE"
)


def _accumulate_relion_physical_particle_grid(
    summed,
    ctf_probs,
    rotations,
    row_mask,
    Ft_y,
    Ft_ctf,
    *,
    pixel_indices,
    image_shape,
    volume_shape,
    max_r,
):
    """Scatter one physically ordered VDAM bucket with RELION fused atomics."""

    if max_r is None:
        raise ValueError("RELION physical particle-grid accumulation requires max_r")
    summed = jnp.asarray(summed, dtype=jnp.complex64)
    ctf_probs = jnp.asarray(ctf_probs, dtype=jnp.float32)
    rotations = jnp.asarray(rotations, dtype=jnp.float32)
    row_mask = jnp.asarray(row_mask, dtype=bool)
    if summed.shape != ctf_probs.shape:
        raise ValueError(
            "RELION physical particle-grid data/weight shapes differ: "
            f"{summed.shape} vs {ctf_probs.shape}"
        )
    if row_mask.shape != summed.shape[:2]:
        raise ValueError(
            "RELION physical particle-grid row mask must match particle/rotation axes: "
            f"{row_mask.shape} vs {summed.shape[:2]}"
        )
    if rotations.shape != (*summed.shape[:2], 3, 3):
        raise ValueError(
            "RELION physical particle-grid rotations must match particle/rotation axes: "
            f"{rotations.shape} vs {(*summed.shape[:2], 3, 3)}"
        )
    summed = jnp.where(row_mask[..., None], summed, 0.0)
    ctf_probs = jnp.where(row_mask[..., None], ctf_probs, 0.0)

    from recovar import cuda_backproject

    return cuda_backproject.relion_fused_x_half_backproject_particle_grid_indexed(
        Ft_y,
        Ft_ctf,
        summed,
        ctf_probs,
        jnp.asarray(pixel_indices, dtype=jnp.int32),
        rotations,
        tuple(int(value) for value in image_shape),
        tuple(int(value) for value in volume_shape),
        float(max_r),
    )


def _accumulate_relion_vdam_physical_particle_grid(
    images,
    ctf,
    minvsigma2,
    posterior_over_weight_norm,
    translation_angles,
    reference,
    rotations,
    row_mask,
    Ft_y,
    Ft_ctf,
    *,
    projector_full=None,
    scoring_rotations=None,
    projector_r_max=None,
    projection_padding_factor=1,
    pixel_indices,
    image_shape,
    volume_shape,
    max_r,
    stable_dense_positions=None,
    logical_current_size=None,
    reconstruction_group_ids=None,
    worker_lane_ids=None,
    particle_trace_ids=None,
    serial_rotation_replay=False,
    persistent_serial_rotation_replay=False,
    float64_accumulator_replay=False,
    reverse_rotation_replay=False,
    rotation_replay_stride=0,
    rotation_replay_order=None,
    rotation_replay_counts=None,
    particle_start_offsets_ns=None,
    native_trace_shape_replay=False,
    materialized_rotation_replay=False,
    particle_replay_order=None,
    candidate_trace_active=False,
    serial_particle_accumulation=False,
    runtime_projector_radius=None,
    transaction_queue=None,
):
    """Form and scatter VDAM residuals in physical particle order."""

    if max_r is None:
        raise ValueError("RELION VDAM physical particle-grid accumulation requires max_r")
    if transaction_queue is not None and (
        projector_full is None
        or materialized_rotation_replay
        or particle_replay_order is not None
        or serial_particle_accumulation
    ):
        raise ValueError(
            "BPref transactions require inline projection without materialized or serial particle replay"
        )
    if runtime_projector_radius is not None and projector_full is None:
        raise ValueError("BPref projector capacity requires the inline projector")
    images = jnp.asarray(images, dtype=jnp.complex64)
    ctf = jnp.asarray(ctf, dtype=jnp.float32)
    minvsigma2 = jnp.asarray(minvsigma2, dtype=jnp.float32)
    posterior_over_weight_norm = jnp.asarray(
        posterior_over_weight_norm,
        dtype=jnp.float32,
    )
    reference = (
        None
        if reference is None
        else jnp.asarray(reference, dtype=jnp.complex64)
    )
    rotations = jnp.asarray(rotations, dtype=jnp.float32)
    row_mask = jnp.asarray(row_mask, dtype=bool)
    if ctf.shape != images.shape or minvsigma2.shape != images.shape:
        raise ValueError("RELION VDAM image/CTF/noise operands must have matching shapes")
    if posterior_over_weight_norm.ndim != 3:
        raise ValueError("RELION VDAM posterior operands must be particle/rotation/translation")
    particle_count, rotation_count, _ = posterior_over_weight_norm.shape
    if projector_full is None:
        if reference is None:
            raise ValueError(
                "RELION VDAM preprojected accumulation requires reference operands"
            )
        if reference.shape != (particle_count, rotation_count, images.shape[1]):
            raise ValueError(
                "RELION VDAM reference operands must match particle/rotation/pixel axes"
            )
    elif reference is not None and reference.shape != (
        particle_count,
        rotation_count,
        images.shape[1],
    ):
        raise ValueError(
            "RELION VDAM optional reference operands must match particle/rotation/pixel axes"
        )
    if rotations.shape != (particle_count, rotation_count, 3, 3):
        raise ValueError("RELION VDAM rotations must match particle/rotation axes")
    if row_mask.shape != (particle_count, rotation_count):
        raise ValueError("RELION VDAM row mask must match particle/rotation axes")
    if reconstruction_group_ids is not None:
        reconstruction_group_ids = jnp.asarray(
            reconstruction_group_ids,
            dtype=jnp.int32,
        )
        if reconstruction_group_ids.shape != (particle_count,):
            raise ValueError(
                "RELION VDAM reconstruction groups must match the particle axis"
            )
    if particle_replay_order is not None:
        particle_replay_order = np.asarray(particle_replay_order, dtype=np.int32)
        if particle_replay_order.shape != (particle_count,) or not np.array_equal(
            np.sort(particle_replay_order),
            np.arange(particle_count, dtype=np.int32),
        ):
            raise ValueError("VDAM particle replay order must be a particle-axis bijection")
        images = jnp.take(images, particle_replay_order, axis=0)
        ctf = jnp.take(ctf, particle_replay_order, axis=0)
        minvsigma2 = jnp.take(minvsigma2, particle_replay_order, axis=0)
        posterior_over_weight_norm = jnp.take(
            posterior_over_weight_norm, particle_replay_order, axis=0
        )
        if reference is not None:
            reference = jnp.take(reference, particle_replay_order, axis=0)
        rotations = jnp.take(rotations, particle_replay_order, axis=0)
        row_mask = jnp.take(row_mask, particle_replay_order, axis=0)
        if scoring_rotations is not None:
            scoring_rotations = jnp.take(
                jnp.asarray(scoring_rotations, dtype=jnp.float32),
                particle_replay_order,
                axis=0,
            )
        if reconstruction_group_ids is not None:
            reconstruction_group_ids = jnp.take(
                reconstruction_group_ids, particle_replay_order, axis=0
            )
        if worker_lane_ids is not None:
            worker_lane_ids = jnp.take(
                jnp.asarray(worker_lane_ids, dtype=jnp.int32),
                particle_replay_order,
                axis=0,
            )
        if particle_trace_ids is not None:
            particle_trace_ids = jnp.take(
                jnp.asarray(particle_trace_ids, dtype=jnp.int32),
                particle_replay_order,
                axis=0,
            )
        if rotation_replay_order is not None:
            rotation_replay_order = jnp.take(
                jnp.asarray(rotation_replay_order, dtype=jnp.int32),
                particle_replay_order,
                axis=0,
            )
        if rotation_replay_counts is not None:
            rotation_replay_counts = jnp.take(
                jnp.asarray(rotation_replay_counts, dtype=jnp.int32),
                particle_replay_order,
                axis=0,
            )
        if particle_start_offsets_ns is not None:
            particle_start_offsets_ns = jnp.take(
                jnp.asarray(particle_start_offsets_ns, dtype=jnp.int32),
                particle_replay_order,
                axis=0,
            )
    posterior_over_weight_norm = jnp.where(
        row_mask[..., None],
        posterior_over_weight_norm,
        0.0,
    )
    if serial_particle_accumulation:
        # One worker lane gives the fused callback the same inter-particle
        # ordering as separate one-particle callbacks without rebuilding the
        # projector texture and host staging allocations for every particle.
        # Kernels for successive particles are issued to one CUDA stream; the
        # callback's existing per-lane synchronization therefore closes each
        # particle before the next one contributes to the shared BPref.
        worker_lane_ids = jnp.zeros((particle_count,), dtype=jnp.int32)
    if materialized_rotation_replay:
        if rotation_replay_order is None:
            raise ValueError("materialized VDAM replay requires a captured row order")
        replay_order = jnp.asarray(rotation_replay_order, dtype=jnp.int32)
        if replay_order.shape != (particle_count, rotation_count):
            raise ValueError("materialized VDAM row order must match particle/rotation axes")
        posterior_over_weight_norm = vdam_replay._materialize_relion_vdam_rotation_rows(
            posterior_over_weight_norm,
            replay_order,
        )
        if reference is not None:
            reference = vdam_replay._materialize_relion_vdam_rotation_rows(
                reference,
                replay_order,
            )
        rotations = vdam_replay._materialize_relion_vdam_rotation_rows(
            rotations,
            replay_order,
        )
        if scoring_rotations is not None:
            scoring_rotations = jnp.asarray(scoring_rotations, dtype=jnp.float32)
            scoring_rotations = vdam_replay._materialize_relion_vdam_rotation_rows(
                scoring_rotations,
                replay_order,
            )
        rotation_replay_order = None

    from recovar import cuda_backproject

    common_args = (
        Ft_y,
        Ft_ctf,
        images,
        ctf,
        minvsigma2,
        posterior_over_weight_norm,
        jnp.asarray(translation_angles, dtype=jnp.float32),
        jnp.asarray(pixel_indices, dtype=jnp.int32),
    )
    if projector_full is None:
        if stable_dense_positions is not None or logical_current_size is not None:
            raise ValueError(
                "stable Fourier-window BPref requires the inline RELION projector"
            )
        if reconstruction_group_ids is not None:
            raise ValueError(
                "grouped VDAM reconstruction requires the inline projector path"
            )
        Ft_y, Ft_ctf, _ = cuda_backproject.relion_vdam_mstep_fused_x_half(
            *common_args,
            reference,
            rotations,
            tuple(int(value) for value in image_shape),
            tuple(int(value) for value in volume_shape),
            float(max_r),
        )
    else:
        if scoring_rotations is None or projector_r_max is None:
            raise ValueError(
                "inline RELION VDAM projection requires scoring rotations and projector radius"
            )
        callback = cuda_backproject.relion_vdam_mstep_fused_projector_x_half
        if transaction_queue is not None:
            callback = functools.partial(transaction_queue.accumulate, callback)
        Ft_y, Ft_ctf, _ = (
            callback(
                *common_args,
                jnp.asarray(projector_full, dtype=jnp.complex64),
                jnp.asarray(scoring_rotations, dtype=jnp.float32),
                tuple(int(value) for value in image_shape),
                tuple(int(value) for value in volume_shape),
                float(max_r),
                int(projector_r_max),
                int(projection_padding_factor),
                reconstruction_group_ids=reconstruction_group_ids,
                worker_lane_ids=worker_lane_ids,
                particle_trace_ids=particle_trace_ids,
                serial_rotation_replay=serial_rotation_replay,
                persistent_serial_rotation_replay=persistent_serial_rotation_replay,
                float64_accumulator_replay=float64_accumulator_replay,
                reverse_rotation_replay=reverse_rotation_replay,
                rotation_replay_stride=rotation_replay_stride,
                rotation_replay_order=rotation_replay_order,
                rotation_replay_counts=rotation_replay_counts,
                particle_start_offsets_ns=particle_start_offsets_ns,
                native_trace_shape_replay=native_trace_shape_replay,
                parallel_worker_replay=(
                    False
                    if serial_particle_accumulation or particle_replay_order is not None
                    else None
                ),
                candidate_trace_active=candidate_trace_active,
                stable_dense_positions=stable_dense_positions,
                logical_current_size=logical_current_size,
                runtime_projector_radius=runtime_projector_radius,
            )
        )
    return Ft_y, Ft_ctf


def _source_faithful_bpref_particle_chunk_size(
    *,
    image_count: int,
    rotation_count: int,
    n_recon_pixels: int,
    max_gb: float,
    max_particles: int | None = None,
) -> int:
    """Bound deferred BPref operands while preserving particle-major order.

    The source-faithful path keeps a complex64 numerator and float32
    denominator for every particle/rotation/pixel triplet.  Split only the
    leading particle axis: consecutive calls then visit particles and every
    rotation within each particle in the same order as RELION.
    """

    bytes_per_particle = max(1, int(rotation_count)) * max(1, int(n_recon_pixels)) * 12
    cap_bytes = max(0, int(float(max_gb) * 1e9))
    chunk_size = min(max(1, int(image_count)), max(1, cap_bytes // bytes_per_particle))
    if max_particles is not None:
        if int(max_particles) < 1:
            raise ValueError("source-faithful BPref particle chunk cap must be positive")
        chunk_size = min(chunk_size, int(max_particles))
    return chunk_size


def _source_faithful_bpref_particle_chunk_cap() -> int | None:
    raw = os.environ.get(EXACT_LOCAL_SOURCE_BPREF_PARTICLE_CHUNK_SIZE_ENV, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{EXACT_LOCAL_SOURCE_BPREF_PARTICLE_CHUNK_SIZE_ENV} must be a positive integer"
        ) from exc
    if value < 1:
        raise ValueError(
            f"{EXACT_LOCAL_SOURCE_BPREF_PARTICLE_CHUNK_SIZE_ENV} must be a positive integer"
        )
    return value


def _source_faithful_bpref_particle_slices(
    image_count: int,
    max_particles: int | None,
) -> tuple[tuple[int, int], ...]:
    """Partition a physical particle stream without reordering it."""

    image_count = int(image_count)
    if image_count < 1:
        raise ValueError("source-faithful BPref particle stream must be non-empty")
    chunk_size = image_count if max_particles is None else min(image_count, int(max_particles))
    if chunk_size < 1:
        raise ValueError("source-faithful BPref particle chunk cap must be positive")
    return tuple(
        (particle_start, min(image_count, particle_start + chunk_size))
        for particle_start in range(0, image_count, chunk_size)
    )
