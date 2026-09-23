"""Native fresh-K1 BPref operand formation, ordered accumulation and staging.

Native accumulation stays in RELION units until the half has finished; callers
own the final conversion to RECOVAR FFT normalization and CTF sign.
"""
from __future__ import annotations

import logging
import os
from functools import partial
from typing import NamedTuple

logger = logging.getLogger(__name__)
import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.diagnostics import bpref_diagnostics, finite_check
from recovar.em.helpers.env_flags import parse_env_flag as _env_flag_enabled
from recovar.em.sparse_pass2 import sparse_pass2_budget, sparse_pass2_policy
from recovar.em.sparse_pass2.sparse_pass2_budget import _optional_positive_int_env

_RELION_FIRSTITER_FUSED_BPREF_ENV = "RECOVAR_K1_RELION_FIRSTITER_FUSED_BPREF"
_RELION_FIRSTITER_DEFERRED_BPREF_ENV = "RECOVAR_RELION_FIRSTITER_DEFERRED_BPREF"
_RELION_FIRSTITER_DEFERRED_BPREF_MAX_HOST_BYTES_ENV = "RECOVAR_RELION_FIRSTITER_DEFERRED_BPREF_MAX_HOST_BYTES"
_AUTO_DEFERRED_FIRSTITER_BPREF_DEVICE_FRACTION = 0.500
_DEFAULT_DEFERRED_FIRSTITER_BPREF_MAX_HOST_BYTES = 128 * 1024**3


class DeferredFirstiterBPrefBatch(NamedTuple):
    """Host snapshot of one scored fresh-firstiter BPref launch group."""

    raw_images: np.ndarray
    raw_ctf: np.ndarray
    raw_minvsigma2: np.ndarray
    posterior: np.ndarray
    rotations: np.ndarray
    actual_counts: np.ndarray
    particle_half_local_indices: np.ndarray
    particle_original_indices: np.ndarray


def _relion_firstiter_fused_bpref_enabled(
    *,
    fresh_k1_guard: bool,
    winner_take_all: bool,
    preserve_bpref_particle_order: bool,
    relion_exact_bpref_operands: bool,
    use_relion_x_half_mstep: bool,
    score_only: bool,
) -> bool:
    """Route only fresh single-class firstiter BPref through its native kernel."""

    supported = bool(
        fresh_k1_guard
        and winner_take_all
        and preserve_bpref_particle_order
        and relion_exact_bpref_operands
        and use_relion_x_half_mstep
        and not score_only
        and int(bpref_diagnostics._bpref_contribution_context["iteration"]) == 1
        and int(bpref_diagnostics._bpref_contribution_context["half"]) in {1, 2}
    )
    raw = os.environ.get(_RELION_FIRSTITER_FUSED_BPREF_ENV)
    if raw is None or not raw.strip():
        return supported
    requested = _env_flag_enabled(_RELION_FIRSTITER_FUSED_BPREF_ENV)
    if requested and not supported:
        raise ValueError(
            f"{_RELION_FIRSTITER_FUSED_BPREF_ENV}=1 requires the fresh K=1 "
            "iteration-1 winner-take-all x-half path"
        )
    return bool(requested and supported)


def _relion_firstiter_bpref_overlap_bytes(
    *,
    projector_shape,
    projector_dtype,
    recon_volume_size: int,
    recon_y_dtype,
    recon_ctf_dtype,
) -> tuple[int, int, int]:
    """Estimate direct-path persistent bytes and its texture-time peak.

    The projection FFI creates one CUDA texture-array copy of the supplied
    Projector::data slab.  The direct route therefore owns two projector
    slabs while both native x-half accumulators are live.
    """

    projector_bytes = int(np.prod(projector_shape)) * int(
        np.dtype(projector_dtype).itemsize
    )
    accumulator_bytes = int(recon_volume_size) * (
        int(np.dtype(recon_y_dtype).itemsize)
        + int(np.dtype(recon_ctf_dtype).itemsize)
    )
    direct_peak_bytes = 2 * projector_bytes + accumulator_bytes
    return projector_bytes, accumulator_bytes, direct_peak_bytes


def _relion_firstiter_deferred_bpref_enabled(
    *,
    relion_firstiter_fused_bpref: bool,
    use_relion_projector: bool,
    projector_device_owned: bool,
    projector_shape,
    projector_dtype,
    recon_volume_size: int,
    recon_y_dtype,
    recon_ctf_dtype,
    device_memory_bytes: int | None,
    allocator_free_memory_bytes: int | None = None,
    diagnostics_active: bool = False,
) -> bool:
    """Select exact host-staged BPref when the direct GPU overlap is unsafe."""

    supported = bool(
        relion_firstiter_fused_bpref
        and use_relion_projector
        and projector_device_owned
        and not diagnostics_active
    )
    raw = os.environ.get(_RELION_FIRSTITER_DEFERRED_BPREF_ENV)
    if raw is not None and raw.strip():
        requested = _env_flag_enabled(_RELION_FIRSTITER_DEFERRED_BPREF_ENV)
        if requested and not supported:
            raise ValueError(
                f"{_RELION_FIRSTITER_DEFERRED_BPREF_ENV}=1 requires a fresh "
                "firstiter fused-BPref call with a host-owned RELION projector"
            )
        return bool(requested and supported)
    if not supported or device_memory_bytes is None or int(device_memory_bytes) <= 0:
        return False
    _, accumulator_bytes, direct_peak_bytes = _relion_firstiter_bpref_overlap_bytes(
        projector_shape=projector_shape,
        projector_dtype=projector_dtype,
        recon_volume_size=recon_volume_size,
        recon_y_dtype=recon_y_dtype,
        recon_ctf_dtype=recon_ctf_dtype,
    )
    exceeds_physical_peak_budget = direct_peak_bytes > int(
        float(device_memory_bytes)
        * _AUTO_DEFERRED_FIRSTITER_BPREF_DEVICE_FRACTION
    )
    exceeds_allocator_budget = bool(
        allocator_free_memory_bytes is not None
        and int(allocator_free_memory_bytes) > 0
        and accumulator_bytes > int(0.8 * float(allocator_free_memory_bytes))
    )
    return bool(exceeds_physical_peak_budget or exceeds_allocator_budget)


def _deferred_firstiter_bpref_estimated_host_bytes(
    buckets,
    *,
    n_half: int,
    n_fine_trans: int,
) -> int:
    """Estimate the exact retained snapshot bytes for fresh firstiter BPref.

    Each particle's complex64 image and float32 CTF is retained once.  The
    inverse-noise row is retained once per launch group, while posterior and
    rotation arrays use the group's padded support.  The three identity/count
    vectors are staged as int64 by :func:`_stage_deferred_firstiter_bpref_batch`.
    """

    n_half = int(n_half)
    n_fine_trans = int(n_fine_trans)
    if n_half <= 0 or n_fine_trans <= 0:
        raise ValueError("deferred firstiter BPref dimensions must be positive")
    total_bytes = 0
    for bucket in buckets:
        batch = int(np.asarray(bucket["image_indices"]).size)
        bucket_size = int(bucket["bucket_size"])
        if batch <= 0 or bucket_size <= 0:
            raise ValueError("deferred firstiter BPref buckets must be nonempty")
        total_bytes += batch * n_half * (
            np.dtype(np.complex64).itemsize + np.dtype(np.float32).itemsize
        )
        total_bytes += n_half * np.dtype(np.float32).itemsize
        total_bytes += batch * bucket_size * (
            n_fine_trans * np.dtype(np.float32).itemsize
            + 9 * np.dtype(np.float32).itemsize
        )
        total_bytes += batch * 3 * np.dtype(np.int64).itemsize
    return int(total_bytes)


def _deferred_firstiter_bpref_max_host_bytes() -> int:
    """Return the fail-closed cap for retained fresh-firstiter host operands."""

    configured = _optional_positive_int_env(
        _RELION_FIRSTITER_DEFERRED_BPREF_MAX_HOST_BYTES_ENV,
    )
    return int(
        _DEFAULT_DEFERRED_FIRSTITER_BPREF_MAX_HOST_BYTES
        if configured is None
        else configured
    )


def _accumulate_relion_firstiter_bpref_fused(
    raw_images,
    raw_ctf,
    raw_minvsigma2,
    posterior,
    rotations,
    actual_counts,
    particle_half_local_indices,
    particle_original_indices,
    data_volume,
    weight_volume,
    *,
    centered_pixel_indices,
    fftw_pixel_indices,
    translation_angles,
    physical_image_shape,
    volume_shape,
    max_r: float,
    adaptive_fraction: float,
):
    """Accumulate fresh firstiter BPref in RELION native units."""

    return _accumulate_relion_firstiter_bpref_fused_impl(
        raw_images,
        raw_ctf,
        raw_minvsigma2,
        posterior,
        rotations,
        actual_counts,
        particle_half_local_indices,
        particle_original_indices,
        (data_volume, weight_volume),
        split_accumulators=False,
        centered_pixel_indices=centered_pixel_indices,
        fftw_pixel_indices=fftw_pixel_indices,
        translation_angles=translation_angles,
        physical_image_shape=physical_image_shape,
        volume_shape=volume_shape,
        max_r=max_r,
        adaptive_fraction=adaptive_fraction,
    )


def _accumulate_relion_firstiter_bpref_fused_split(
    raw_images,
    raw_ctf,
    raw_minvsigma2,
    posterior,
    rotations,
    actual_counts,
    particle_half_local_indices,
    particle_original_indices,
    data_volume_real,
    data_volume_imag,
    weight_volume,
    *,
    centered_pixel_indices,
    fftw_pixel_indices,
    translation_angles,
    physical_image_shape,
    volume_shape,
    max_r: float,
    adaptive_fraction: float,
):
    """Accumulate fresh firstiter BPref without interleaving complex data."""

    return _accumulate_relion_firstiter_bpref_fused_impl(
        raw_images,
        raw_ctf,
        raw_minvsigma2,
        posterior,
        rotations,
        actual_counts,
        particle_half_local_indices,
        particle_original_indices,
        (data_volume_real, data_volume_imag, weight_volume),
        split_accumulators=True,
        centered_pixel_indices=centered_pixel_indices,
        fftw_pixel_indices=fftw_pixel_indices,
        translation_angles=translation_angles,
        physical_image_shape=physical_image_shape,
        volume_shape=volume_shape,
        max_r=max_r,
        adaptive_fraction=adaptive_fraction,
    )


def _accumulate_relion_firstiter_bpref_fused_impl(
    raw_images,
    raw_ctf,
    raw_minvsigma2,
    posterior,
    rotations,
    actual_counts,
    particle_half_local_indices,
    particle_original_indices,
    accumulators,
    *,
    split_accumulators: bool,
    centered_pixel_indices,
    fftw_pixel_indices,
    translation_angles,
    physical_image_shape,
    volume_shape,
    max_r: float,
    adaptive_fraction: float,
):
    """Shared operand preparation and ordered firstiter particle launches."""

    from recovar import cuda_backproject
    from recovar.em.cuda import kernels as em_cuda_kernels

    if split_accumulators:
        data_volume_real, data_volume_imag, weight_volume = accumulators
        data_volume = None
    else:
        data_volume, weight_volume = accumulators
        data_volume_real = None
        data_volume_imag = None

    actual_counts = np.asarray(actual_counts, dtype=np.int64)
    if actual_counts.shape != (int(posterior.shape[0]),):
        raise ValueError("firstiter fused BPref particle counts do not match posterior rows")
    if raw_images.shape != raw_ctf.shape or raw_images.ndim != 2:
        raise ValueError("firstiter fused BPref image/CTF rows must be aligned")
    if raw_minvsigma2.shape != (int(raw_images.shape[1]),):
        raise ValueError("firstiter fused BPref inverse-noise row is not pixel-aligned")
    particle_half_local_indices = np.asarray(
        particle_half_local_indices, dtype=np.int64
    ).reshape(-1)
    particle_original_indices = np.asarray(
        particle_original_indices, dtype=np.int64
    ).reshape(-1)
    if (
        particle_half_local_indices.shape != actual_counts.shape
        or particle_original_indices.shape != actual_counts.shape
    ):
        raise ValueError(
            "firstiter fused BPref particle identities do not match posterior rows"
        )

    delta_config = bpref_diagnostics._bpref_accumulator_delta_config()
    if delta_config is not None:
        context_iteration = int(bpref_diagnostics._bpref_contribution_context["iteration"])
        context_half = int(bpref_diagnostics._bpref_contribution_context["half"])
        if (
            context_iteration != int(delta_config["iteration"])
            or context_half != int(delta_config["half"])
        ):
            delta_config = None
    if split_accumulators and delta_config is not None:
        raise RuntimeError(
            "split firstiter BPref accumulation is incompatible with accumulator-delta diagnostics"
        )

    centered_pixel_indices = jnp.asarray(centered_pixel_indices, dtype=jnp.int32)
    fftw_pixel_indices = jnp.asarray(fftw_pixel_indices, dtype=jnp.int32)
    source_images = jnp.asarray(raw_images[:, centered_pixel_indices], dtype=jnp.complex64)
    source_ctf = jnp.asarray(raw_ctf[:, centered_pixel_indices], dtype=jnp.float32)
    source_minvsigma2 = jnp.broadcast_to(
        jnp.asarray(raw_minvsigma2[centered_pixel_indices], dtype=jnp.float32)[None, :],
        source_ctf.shape,
    )
    dense_images, dense_indices, current_height, current_half_width = (
        cuda_backproject._prepare_relion_x_half_block_topology_operands(
            source_images,
            fftw_pixel_indices,
            physical_image_shape,
            max_r,
        )
    )
    dense_ctf, ctf_indices, ctf_height, ctf_half_width = (
        cuda_backproject._prepare_relion_x_half_block_topology_operands(
            source_ctf,
            fftw_pixel_indices,
            physical_image_shape,
            max_r,
        )
    )
    dense_minvsigma2, noise_indices, noise_height, noise_half_width = (
        cuda_backproject._prepare_relion_x_half_block_topology_operands(
            source_minvsigma2,
            fftw_pixel_indices,
            physical_image_shape,
            max_r,
        )
    )
    topology = (current_height, current_half_width, dense_indices.shape)
    if topology != (ctf_height, ctf_half_width, ctf_indices.shape) or topology != (
        noise_height,
        noise_half_width,
        noise_indices.shape,
    ):
        raise RuntimeError("firstiter fused BPref dense operand topologies differ")
    if current_half_width != current_height // 2 + 1:
        raise RuntimeError("firstiter fused BPref did not form an FFTW half square")

    # RELION passes these host-known values as kernel parameters. Keeping them
    # as Python scalars avoids a per-particle device-to-host synchronization in
    # the exact-source CUDA FFI target.
    threshold = float(np.float32(adaptive_fraction))
    weight_norm = 1.0
    translation_angles = jnp.asarray(translation_angles, dtype=jnp.float32)
    current_image_shape = (int(current_height), int(current_height))
    for particle_index, count in enumerate(actual_counts.tolist()):
        if count <= 0:
            continue
        original_index = int(particle_original_indices[particle_index])
        capture_particle = bool(
            delta_config is not None
            and original_index in delta_config["original_indices"]
        )
        before_data = np.asarray(data_volume).copy() if capture_particle else None
        before_weight = np.asarray(weight_volume).copy() if capture_particle else None
        particle_posterior = jnp.asarray(
            posterior[particle_index, :count], dtype=jnp.float32
        )
        native_eulers = jnp.asarray(
            rotations[particle_index, :count], dtype=jnp.float32
        ).transpose(0, 2, 1)
        if split_accumulators:
            data_volume_real, data_volume_imag, weight_volume = (
                em_cuda_kernels.relion_firstiter_bpref_fused_x_half_split(
                    data_volume_real,
                    data_volume_imag,
                    weight_volume,
                    dense_images[particle_index],
                    dense_ctf[particle_index],
                    dense_minvsigma2[particle_index],
                    particle_posterior,
                    translation_angles,
                    native_eulers,
                    threshold,
                    weight_norm,
                    current_image_shape,
                    volume_shape,
                    max_r,
                )
            )
        else:
            data_volume, weight_volume = (
                em_cuda_kernels.relion_firstiter_bpref_fused_x_half(
                    data_volume,
                    weight_volume,
                    dense_images[particle_index],
                    dense_ctf[particle_index],
                    dense_minvsigma2[particle_index],
                    particle_posterior,
                    translation_angles,
                    native_eulers,
                    threshold,
                    weight_norm,
                    current_image_shape,
                    volume_shape,
                    max_r,
                )
            )
        if capture_particle:
            isolated_data, isolated_weight = (
                em_cuda_kernels.relion_firstiter_bpref_fused_x_half(
                    jnp.zeros_like(data_volume),
                    jnp.zeros_like(weight_volume),
                    dense_images[particle_index],
                    dense_ctf[particle_index],
                    dense_minvsigma2[particle_index],
                    particle_posterior,
                    translation_angles,
                    native_eulers,
                    threshold,
                    weight_norm,
                    current_image_shape,
                    volume_shape,
                    max_r,
                )
            )
            bpref_diagnostics._write_bpref_accumulator_delta_v1(
                config=delta_config,
                original_index=original_index,
                particle_launch_ordinal=int(
                    particle_half_local_indices[particle_index]
                ),
                particle_rotation_count=count,
                before_data=before_data,
                before_weight=before_weight,
                after_data=np.asarray(data_volume).copy(),
                after_weight=np.asarray(weight_volume).copy(),
                isolated_data=np.asarray(isolated_data).copy(),
                isolated_weight=np.asarray(isolated_weight).copy(),
                isolated_layout=(
                    "RECOVAR firstiter fused interleaved complex64 data plus "
                    "float32 weight"
                ),
                volume_shape=volume_shape,
                max_r=max_r,
                operand_bundle={
                    "operand_source_image": np.asarray(
                        dense_images[particle_index], dtype=np.complex64
                    ).copy(),
                    "operand_ctf": np.asarray(
                        dense_ctf[particle_index], dtype=np.float32
                    ).copy(),
                    "operand_minvsigma2": np.asarray(
                        dense_minvsigma2[particle_index], dtype=np.float32
                    ).copy(),
                    "operand_posterior": np.asarray(
                        particle_posterior, dtype=np.float32
                    ).copy(),
                    "operand_translation_angles": np.asarray(
                        translation_angles, dtype=np.float32
                    ).copy(),
                    "operand_eulers": np.asarray(
                        native_eulers, dtype=np.float32
                    ).copy(),
                    "operand_threshold": np.asarray(threshold, dtype=np.float32).copy(),
                    "operand_weight_norm": np.asarray(weight_norm, dtype=np.float32).copy(),
                    "operand_centered_pixel_indices": np.asarray(
                        centered_pixel_indices, dtype=np.int32
                    ).copy(),
                    "operand_fftw_pixel_indices": np.asarray(
                        fftw_pixel_indices, dtype=np.int32
                    ).copy(),
                },
            )
    if split_accumulators:
        return data_volume_real, data_volume_imag, weight_volume
    return data_volume, weight_volume


def _host_snapshot(value) -> np.ndarray:
    """Make an independent, bit-preserving host copy of a scored operand."""

    return np.array(jax.device_get(value), copy=True)


def _stage_deferred_firstiter_bpref_batch(
    *,
    raw_images,
    raw_ctf,
    raw_minvsigma2,
    posterior,
    rotations,
    actual_counts,
    particle_half_local_indices,
    particle_original_indices,
) -> DeferredFirstiterBPrefBatch:
    """Snapshot one launch group without changing any source dtype or order."""

    return DeferredFirstiterBPrefBatch(
        raw_images=_host_snapshot(raw_images),
        raw_ctf=_host_snapshot(raw_ctf),
        raw_minvsigma2=_host_snapshot(raw_minvsigma2),
        posterior=_host_snapshot(posterior),
        rotations=_host_snapshot(rotations),
        actual_counts=np.array(actual_counts, dtype=np.int64, copy=True),
        particle_half_local_indices=np.array(
            particle_half_local_indices,
            dtype=np.int64,
            copy=True,
        ),
        particle_original_indices=np.array(
            particle_original_indices,
            dtype=np.int64,
            copy=True,
        ),
    )


def _deferred_firstiter_bpref_batch_nbytes(
    batch: DeferredFirstiterBPrefBatch,
) -> int:
    """Return retained host bytes for one deferred launch group."""

    return int(sum(value.nbytes for value in batch))


def _replay_deferred_firstiter_bpref_batches(
    batches: list[DeferredFirstiterBPrefBatch],
    data_volume_real,
    data_volume_imag,
    weight_volume,
    *,
    centered_pixel_indices,
    fftw_pixel_indices,
    translation_angles,
    physical_image_shape,
    volume_shape,
    max_r: float,
    adaptive_fraction: float,
):
    """Replay staged groups while retaining RELION's exact split layout."""

    for batch in batches:
        data_volume_real, data_volume_imag, weight_volume = (
            _accumulate_relion_firstiter_bpref_fused_split(
                batch.raw_images,
                batch.raw_ctf,
                batch.raw_minvsigma2,
                batch.posterior,
                batch.rotations,
                batch.actual_counts,
                batch.particle_half_local_indices,
                batch.particle_original_indices,
                data_volume_real,
                data_volume_imag,
                weight_volume,
                centered_pixel_indices=centered_pixel_indices,
                fftw_pixel_indices=fftw_pixel_indices,
                translation_angles=translation_angles,
                physical_image_shape=physical_image_shape,
                volume_shape=volume_shape,
                max_r=max_r,
                adaptive_fraction=adaptive_fraction,
            )
        )
        # A box-800 launch group owns tens of MiB of uploaded image/CTF/noise
        # operands.  The accumulator dependency preserves launch order, but
        # without this boundary Python can enqueue many later groups before
        # those operands are retired and rebuild the live-set pressure this
        # path avoids.
        jax.block_until_ready(
            (data_volume_real, data_volume_imag, weight_volume),
        )
    return data_volume_real, data_volume_imag, weight_volume


@partial(jax.jit, donate_argnums=(0, 1, 2))
def _normalize_split_relion_firstiter_bpref_accumulators(
    data_volume_real,
    data_volume_imag,
    weight_volume,
    fft_size,
    fft_size_squared,
):
    """Normalize and consume the three full-volume split accumulators."""

    # Scalar division preserves the historical complex result for every
    # nonzero component without the extra multiplies in an expanded complex
    # quotient (which change float32 rounding).  Complex division has special
    # signed-zero results because its zero imaginary divisor still enters the
    # component products; repair exactly those raw-zero components below.
    normalized_real = -data_volume_real / fft_size
    normalized_imag = -data_volume_imag / fft_size
    positive_zero = jnp.asarray(0.0, dtype=data_volume_real.dtype)
    negative_zero = jnp.asarray(-0.0, dtype=data_volume_real.dtype)
    real_zero_is_negative = jnp.logical_and(
        jnp.logical_not(jnp.signbit(data_volume_real)),
        jnp.logical_not(jnp.signbit(data_volume_imag)),
    )
    imag_zero_is_negative = jnp.logical_and(
        jnp.logical_not(jnp.signbit(data_volume_imag)),
        jnp.signbit(data_volume_real),
    )
    normalized_real = jnp.where(
        data_volume_real == 0,
        jnp.where(real_zero_is_negative, negative_zero, positive_zero),
        normalized_real,
    )
    normalized_imag = jnp.where(
        data_volume_imag == 0,
        jnp.where(imag_zero_is_negative, negative_zero, positive_zero),
        normalized_imag,
    )
    return (
        normalized_real,
        normalized_imag,
        weight_volume / fft_size_squared,
    )


def _release_deferred_firstiter_projection_buffers(
    projector_device_buffer,
    projection_cache,
) -> None:
    """Delete only function-owned scoring buffers, once per device object."""

    values = [projector_device_buffer]
    if projection_cache is not None:
        values.extend(projection_cache.values())
    deleted_ids: set[int] = set()
    for value in values:
        if value is None or id(value) in deleted_ids:
            continue
        deleted_ids.add(id(value))
        delete = getattr(value, "delete", None)
        if callable(delete):
            try:
                delete()
            except RuntimeError as exc:
                logger.warning(
                    "Deferred firstiter BPref could not explicitly release %s: %s",
                    type(value).__name__,
                    exc,
                )


def _relion_firstiter_compact_batch_planning_safe(
    *,
    relion_firstiter_fused_bpref: bool,
    projector_host_owned: bool,
    diagnostics_active: bool,
    deferred_firstiter_bpref: bool,
    direct_peak_bytes: int,
    fixed_base_bytes: int,
    projector_dtype,
    score_complex_dtype,
) -> bool:
    """Return whether the c64/f32 K=1 lifetime admits phase-max planning."""

    compact_dtypes = bool(
        np.dtype(projector_dtype) == np.dtype(np.complex64)
        and np.dtype(score_complex_dtype) == np.dtype(np.complex64)
    )
    return bool(
        relion_firstiter_fused_bpref
        and projector_host_owned
        and not diagnostics_active
        and compact_dtypes
        and (
            deferred_firstiter_bpref
            or int(direct_peak_bytes) <= int(fixed_base_bytes)
        )
    )

def _relion_soft_compact_batch_planning_safe(
    *,
    source_faithful_spectrum_norm: bool,
    preserve_bpref_particle_order: bool,
    use_relion_x_half_mstep: bool,
    relion_cuda_images: bool,
    projector_half,
    score_complex_dtype,
    model_current_size: int,
    image_size: int,
    bpref_device_signature_active: bool,
) -> bool:
    """Admit compact planning for the exact windowed K=1 soft-posterior path."""

    if projector_half is None:
        return False
    diagnostics_active = (
        finite_check.finite_check_enabled()
        or bpref_diagnostics._relion_firstiter_bpref_diagnostics_active(
            bpref_device_signature_active=bpref_device_signature_active,
        )
    )
    projector_host_owned = not isinstance(projector_half, jax.Array)
    return bool(
        source_faithful_spectrum_norm
        and preserve_bpref_particle_order
        and use_relion_x_half_mstep
        and relion_cuda_images
        and projector_host_owned
        and np.dtype(projector_half.dtype) == np.dtype(np.complex64)
        and np.dtype(score_complex_dtype) == np.dtype(np.complex64)
        and 0 < int(model_current_size) < int(image_size)
        and not diagnostics_active
    )

class _RelionFirstiterCompactBatchPlanningDecision(NamedTuple):
    enabled: bool
    deferred_firstiter_bpref: bool
    direct_peak_bytes: int

def _relion_firstiter_compact_batch_planning_decision(
    *,
    source_faithful_spectrum_norm: bool,
    winner_take_all: bool,
    preserve_bpref_particle_order: bool,
    use_relion_x_half_mstep: bool,
    projector_half,
    score_complex_dtype,
    recon_volume_size: int,
    bpref_device_signature_active: bool,
    fixed_base_bytes: int,
) -> _RelionFirstiterCompactBatchPlanningDecision:
    """Preflight compact planning through the authoritative BPref gates."""

    if projector_half is None:
        return _RelionFirstiterCompactBatchPlanningDecision(False, False, 0)
    projector_shape = tuple(projector_half.shape)
    projector_dtype = np.dtype(projector_half.dtype)
    projector_host_owned = not isinstance(projector_half, jax.Array)
    diagnostics_active = (
        finite_check.finite_check_enabled()
        or bpref_diagnostics._relion_firstiter_bpref_diagnostics_active(
            bpref_device_signature_active=bpref_device_signature_active,
        )
    )
    fresh_k1_guard = bool(source_faithful_spectrum_norm)
    relion_exact_bpref_operands = sparse_pass2_policy._relion_exact_bpref_operands_enabled(
        fresh_k1_guard=fresh_k1_guard,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
    )
    relion_firstiter_fused_bpref = _relion_firstiter_fused_bpref_enabled(
        fresh_k1_guard=fresh_k1_guard,
        winner_take_all=winner_take_all,
        preserve_bpref_particle_order=preserve_bpref_particle_order,
        relion_exact_bpref_operands=relion_exact_bpref_operands,
        use_relion_x_half_mstep=use_relion_x_half_mstep,
        score_only=False,
    )
    _, _, direct_peak_bytes = _relion_firstiter_bpref_overlap_bytes(
        projector_shape=projector_shape,
        projector_dtype=projector_dtype,
        recon_volume_size=recon_volume_size,
        recon_y_dtype=np.complex64,
        recon_ctf_dtype=np.float32,
    )
    deferred_firstiter_bpref = _relion_firstiter_deferred_bpref_enabled(
        relion_firstiter_fused_bpref=relion_firstiter_fused_bpref,
        use_relion_projector=True,
        projector_device_owned=projector_host_owned,
        projector_shape=projector_shape,
        projector_dtype=projector_dtype,
        recon_volume_size=recon_volume_size,
        recon_y_dtype=np.complex64,
        recon_ctf_dtype=np.float32,
        device_memory_bytes=sparse_pass2_budget._device_memory_limit_bytes(),
        allocator_free_memory_bytes=sparse_pass2_budget._jax_allocator_free_memory_bytes(),
        diagnostics_active=diagnostics_active,
    )
    enabled = _relion_firstiter_compact_batch_planning_safe(
        relion_firstiter_fused_bpref=relion_firstiter_fused_bpref,
        projector_host_owned=projector_host_owned,
        diagnostics_active=diagnostics_active,
        deferred_firstiter_bpref=deferred_firstiter_bpref,
        direct_peak_bytes=direct_peak_bytes,
        fixed_base_bytes=fixed_base_bytes,
        projector_dtype=projector_dtype,
        score_complex_dtype=score_complex_dtype,
    )
    return _RelionFirstiterCompactBatchPlanningDecision(
        enabled,
        deferred_firstiter_bpref,
        direct_peak_bytes,
    )
