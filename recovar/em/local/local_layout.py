"""Per-image local hypothesis layout and bucketization helpers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from recovar import utils
from recovar.em.helpers.batch_planning import (
    _plan_consecutive_padded_batches,
)
from recovar.em.helpers.orientation_priors import make_relion_translation_log_prior
from recovar.em.helpers.shape_buckets import coarse_bucket, power_bucket
from recovar.em.sampling import (
    _normalized_log_weights,
    _wrapped_abs_diff_deg,
    apply_relion_rotation_perturbation_to_eulers,
    build_local_search_grid_metadata,
    get_local_rotation_grid_fast,
    get_oversampled_rotation_grid_from_samples,
    get_oversampled_translation_grid,
    infer_translation_step,
    rotation_grid_n_in_planes,
    rotation_grid_size,
    rotation_indices_to_relion_eulers,
)
from recovar.em.scoring.significant_samples import significant_sample_ids

EXACT_LOCAL_BUCKET_QUANTUM_ENV = "RECOVAR_EXACT_LOCAL_BUCKET_QUANTUM"
EXACT_LOCAL_BUCKET_RADIX_ENV = "RECOVAR_EXACT_LOCAL_BUCKET_RADIX"
EXACT_LOCAL_BUCKET_MIN_QUANTUM = 256

LOCAL_IMAGE_CAPACITY_LADDER_ENV = "RECOVAR_LOCAL_IMAGE_CAPACITY_LADDER"
DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER = (16, 32, 64, 128, 256)


def resolve_local_image_capacity_ladder(explicit=None) -> tuple[int, ...]:
    """Resolve the opt-in image-axis capacity ladder; ``()`` means off.

    The exact local engine's images-per-bucket capacity is
    ``min(image_batch_size, max_hypotheses_per_microbatch // bucket_rotations)``.
    ``image_batch_size`` comes from a memory estimate that moves by a few images
    between iterations, and it is the leading axis of every per-bucket program,
    so a one-image change recompiles the whole bucket program set. Snapping the
    capacity to a fixed ladder makes that axis stable across iterations.

    ``explicit`` beats the environment. Accepted values: ``None`` (consult the
    environment), ``False``/``""``/``"0"``/``"off"`` (ladder off), ``True``/
    ``"1"``/``"on"``/``"auto"`` (the default ladder) or an explicit sequence or
    comma-separated string of positive capacities.

    Only ``run_local_em_exact`` calls this with ``None``. The bucket planners
    below treat ``None`` as off, because they are also called by
    ``recovar/em/ppca_refinement/local_dataset.py``, a pipeline with its own
    validation that an EM-scoped environment variable must not re-bucket.
    """

    source = "local_image_capacity_ladder"
    raw = explicit
    if raw is None:
        source = LOCAL_IMAGE_CAPACITY_LADDER_ENV
        raw = os.environ.get(LOCAL_IMAGE_CAPACITY_LADDER_ENV, "")
    if raw is False:
        return ()
    if raw is True:
        return DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in {"", "0", "off", "false", "no", "none"}:
            return ()
        if token in {"1", "on", "true", "yes", "auto", "default"}:
            return DEFAULT_LOCAL_IMAGE_CAPACITY_LADDER
        raw = [part for part in token.replace(" ", "").split(",") if part]
    try:
        ladder = tuple(sorted({int(value) for value in raw}))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} must be a comma-separated list of positive image capacities"
        ) from exc
    if not ladder:
        return ()
    if ladder[0] < 1:
        raise ValueError(f"{source} capacities must be positive")
    return ladder


def _planner_image_capacity_ladder(explicit) -> tuple[int, ...]:
    """Resolve a planner's ladder argument; ``None`` means off, not "ask the env"."""

    if explicit is None:
        return ()
    return resolve_local_image_capacity_ladder(explicit)


def _ladder_image_capacity(max_images: int, ladder: tuple[int, ...]) -> int:
    """Snap an images-per-bucket capacity DOWN to the ladder.

    Snapping down, never up: ``max_images`` is already the planner's memory
    bound (the ``image_batch_size`` estimate and the hypothesis cap), so a
    larger capacity would plan a bucket the caller's own estimate refused. A
    capacity below the ladder's minimum is left alone for the same reason --
    there is no smaller ladder rung to fall back to that keeps the bucket
    non-empty, and raising it would break the bound.
    """

    max_images = int(max_images)
    if not ladder:
        return max_images
    rungs = [rung for rung in ladder if rung <= max_images]
    if not rungs:
        return max_images
    return int(rungs[-1])



def _resolve_exact_local_bucket_radix(explicit: int | None = None) -> int:
    """Resolve and validate the exact-local small-bucket radix."""

    source = "exact_local_bucket_radix"
    raw_value = explicit
    if raw_value is None:
        source = EXACT_LOCAL_BUCKET_RADIX_ENV
        raw_value = os.environ.get(EXACT_LOCAL_BUCKET_RADIX_ENV, "2")
    try:
        bucket_radix = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be an integer at least 2") from exc
    if bucket_radix < 2:
        raise ValueError(f"{source} must be at least 2")
    return bucket_radix


def _exact_bucket_rotation_size(
    local_rotation_count: int,
    rotation_block_size: int,
    *,
    large_bucket_quantum: int | None = None,
    exact_local_bucket_radix: int | None = None,
) -> int:
    """Return a compile-friendly padded size for one exact local neighborhood.

    The exact local engine cannot safely cap the bucket size below the true
    per-image neighborhood cardinality. Use power-of-two style padding for
    smaller neighborhoods. For larger exact neighborhoods, round up to a
    coarse fixed quantum so nearby local-support
    sizes reuse the same compiled shapes instead of each exact count generating
    its own XLA program.
    """
    local_rotation_count = int(local_rotation_count)
    if local_rotation_count <= 0:
        return 1
    engine_cap = int(_local_search_engine_rotation_block_size(rotation_block_size))
    if local_rotation_count <= engine_cap:
        bucket_radix = _resolve_exact_local_bucket_radix(exact_local_bucket_radix)
        if bucket_radix != 2:
            return int(
                power_bucket(
                    local_rotation_count,
                    base=bucket_radix,
                    minimum=16,
                    maximum=engine_cap,
                ),
            )
        return int(
            coarse_bucket(
                local_rotation_count,
                small_power2_max=engine_cap,
                large_multiple=engine_cap,
                minimum=16,
            ),
        )
    # ``RECOVAR_LOCAL_BUCKET_QUANTUM`` lets callers override the large-bucket
    # quantization. The default is deliberately coarser than the exact-local
    # engine cap: outlier-heavy/local-search tails otherwise generate hundreds
    # of near-duplicate XLA shapes. Hypothesis/tile caps still chunk each
    # bucket, so this changes padding/shape reuse rather than the candidate set.
    env_quantum = os.environ.get("RECOVAR_LOCAL_BUCKET_QUANTUM", "")
    if env_quantum:
        large_bucket_quantum = max(1, int(env_quantum))
    elif large_bucket_quantum is None:
        large_bucket_quantum = max(4096, engine_cap)
    else:
        large_bucket_quantum = max(1, int(large_bucket_quantum))
    return int(
        coarse_bucket(
            local_rotation_count,
            small_power2_max=engine_cap,
            large_multiple=large_bucket_quantum,
            minimum=16,
        ),
    )


def _exact_local_large_bucket_quantum(rotation_block_size: int, explicit: int | None = None) -> int:
    """Return the large-neighborhood bucket quantum for exact local search."""

    if explicit is not None:
        return max(1, int(explicit))
    env_quantum = os.environ.get(EXACT_LOCAL_BUCKET_QUANTUM_ENV, "")
    if env_quantum:
        return max(1, int(env_quantum))
    engine_cap = int(_local_search_engine_rotation_block_size(rotation_block_size))
    return max(EXACT_LOCAL_BUCKET_MIN_QUANTUM, engine_cap)


@dataclass(frozen=True)
class LocalHypothesisLayout:
    """Flat per-image local hypothesis storage."""

    n_global_rotations: int
    n_pixels: int
    n_psi: int
    rotation_offsets: np.ndarray
    rotation_ids_flat: np.ndarray
    rotations_flat: np.ndarray
    rotation_log_priors_flat: np.ndarray
    rotation_counts: np.ndarray
    translation_grid: np.ndarray
    translation_log_priors: np.ndarray
    rotation_posterior_ids_flat: np.ndarray | None = None
    sample_mask_bits: np.ndarray | None = None  # uint8, translation bits packed little-endian
    mstep_rotations_flat: np.ndarray | None = None
    source_eulers_flat: np.ndarray | None = None

    def sample_mask_rows(self, start=0, stop=None) -> np.ndarray | None:
        """Expand only the requested rotation rows to the kernel's boolean mask."""
        if self.sample_mask_bits is None:
            return None
        return np.unpackbits(
            self.sample_mask_bits[start:stop], axis=1,
            count=int(self.translation_grid.shape[0]), bitorder="little",
        ).view(np.bool_)

    @property
    def n_images(self) -> int:
        return int(self.rotation_counts.shape[0])

    @property
    def total_local_rotations(self) -> int:
        return int(self.rotation_ids_flat.shape[0])


@dataclass(frozen=True)
class LocalBucketSpec:
    """Static-shape padded execution batch for the exact local engine."""

    image_indices: np.ndarray
    bucket_image_count: int
    bucket_rotation_count: int
    actual_rotation_counts: np.ndarray
    local_rotation_ids: np.ndarray
    local_rotations: np.ndarray
    local_rotation_log_prior: np.ndarray
    local_rotation_mask: np.ndarray
    translation_log_prior: np.ndarray
    local_rotation_posterior_ids: np.ndarray | None = None
    local_sample_mask: np.ndarray | None = None
    local_mstep_rotations: np.ndarray | None = None
    local_source_eulers: np.ndarray | None = None
    # Class-segmented rows (one engine for K=1 and K>1). Row axis is class-major:
    # class ``k`` owns rows ``[k*seg, (k+1)*seg)`` with ``seg`` the segment width, so
    # ``bucket_rotation_count == n_classes * class_segment_rotation_count``. For K=1
    # the segment is the whole row axis and every array is identical to the
    # single-class bucketer's output. ``class_actual_rotation_counts`` is ``[B, K]``.
    n_classes: int = 1
    class_segment_rotation_count: int | None = None
    class_actual_rotation_counts: np.ndarray | None = None

    @property
    def segment_rotation_count(self) -> int:
        if self.class_segment_rotation_count is None:
            return int(self.bucket_rotation_count)
        return int(self.class_segment_rotation_count)


def _local_mstep_rotations(bucket: LocalBucketSpec) -> np.ndarray:
    """Return the adjoint-only rotations, falling back to scoring rotations.

    Preserves the source array's own dtype rather than forcing float32:
    ``bucket.local_mstep_rotations``/``local_rotations`` are already built by
    ``bucket_local_hypothesis_layout`` at whatever precision the caller's
    ``LocalHypothesisLayout`` chose (float64 under double-precision scoring).
    """

    rotations = bucket.local_mstep_rotations
    if rotations is None:
        rotations = bucket.local_rotations
    return np.asarray(rotations)


def _resolve_prior_rotations(prior_rotations: np.ndarray, healpix_order: int, grid_metadata):
    """Return RELION eulers and rotation matrices for local-support construction."""

    prior_rotations = np.asarray(prior_rotations)
    if prior_rotations.ndim == 0:
        prior_rotations = prior_rotations.reshape(1)

    if prior_rotations.ndim == 1:
        if "eulers_full" in grid_metadata:
            prior_eulers = np.asarray(grid_metadata["eulers_full"], dtype=np.float32)[prior_rotations.astype(np.int64)]
        else:
            prior_eulers = rotation_indices_to_relion_eulers(prior_rotations.astype(np.int64), healpix_order)
        prior_rotation_mats = utils.R_from_relion(prior_eulers, degrees=True)
        return np.asarray(prior_eulers, dtype=np.float32), np.asarray(prior_rotation_mats, dtype=np.float64)
    if prior_rotations.ndim == 2 and prior_rotations.shape[-1] == 3:
        prior_eulers = np.asarray(prior_rotations, dtype=np.float32).reshape(-1, 3)
        prior_rotation_mats = utils.R_from_relion(prior_eulers, degrees=True)
        return prior_eulers, np.asarray(prior_rotation_mats, dtype=np.float64)
    prior_rotation_mats = np.asarray(prior_rotations, dtype=np.float64).reshape(-1, 3, 3)
    prior_eulers = utils.R_to_relion(prior_rotation_mats, degrees=True).astype(np.float32)
    return prior_eulers, prior_rotation_mats


def _local_selector_chunk_size(n_images: int, n_pixels: int, n_psi: int, use_direction: bool, use_psi: bool) -> int:
    explicit = os.environ.get("RECOVAR_LOCAL_SELECTOR_CHUNK_SIZE", "")
    if explicit:
        return max(1, min(int(n_images), int(explicit)))

    max_elements = int(os.environ.get("RECOVAR_LOCAL_SELECTOR_MAX_ELEMENTS", "16000000"))
    per_image_elements = 0
    if use_direction:
        per_image_elements += int(n_pixels)
    if use_psi:
        per_image_elements += int(n_psi)
    if per_image_elements <= 0:
        return max(1, int(n_images))
    return max(1, min(int(n_images), max_elements // per_image_elements))


def _build_factorized_local_entries(
    prior_rotations: np.ndarray,
    healpix_order: int,
    sigma_rot: float,
    sigma_psi: float,
    grid_metadata,
    *,
    dtype: np.dtype = np.float32,
):
    """Build exact per-image local supports for factorized HEALPix x psi grids."""

    prior_eulers, prior_rotation_mats = _resolve_prior_rotations(prior_rotations, healpix_order, grid_metadata)
    dir_vecs = np.asarray(grid_metadata["dir_vecs"], dtype=np.float64)
    psi_deg_grid = np.asarray(grid_metadata["psi_deg"], dtype=np.float64)
    n_pixels = int(grid_metadata["n_pixels"])

    prior_dir_vecs = np.asarray(prior_rotation_mats[:, 2, :], dtype=np.float64)
    prior_dir_norm = np.linalg.norm(prior_dir_vecs, axis=1, keepdims=True)
    prior_dir_norm = np.where(prior_dir_norm > 0.0, prior_dir_norm, 1.0)
    prior_dir_vecs = prior_dir_vecs / prior_dir_norm
    prior_psi_deg = np.mod(np.asarray(prior_eulers[:, 2], dtype=np.float64), 360.0)

    sigma_rot_deg = float(np.rad2deg(sigma_rot))
    sigma_psi_deg = float(np.rad2deg(sigma_psi))
    # RELION widens the direction cone with max(sigma_rot, sigma_tilt).
    # In this SPA path sigma_tilt == sigma_rot; sigma_psi only controls
    # in-plane support and must not widen directions.
    biggest_sigma_deg = sigma_rot_deg
    cutoff_dir_deg = 3.0 * biggest_sigma_deg
    cutoff_psi_deg = 3.0 * sigma_psi_deg

    rotation_ids_parts: list[np.ndarray] = []
    log_prior_parts: list[np.ndarray] = []
    n_images = int(prior_eulers.shape[0])
    counts = np.zeros(n_images, dtype=np.int32)
    offsets = np.zeros(n_images + 1, dtype=np.int64)
    chunk_size = _local_selector_chunk_size(
        n_images,
        n_pixels,
        int(grid_metadata["n_psi"]),
        sigma_rot_deg > 0.0,
        sigma_psi_deg > 0.0,
    )
    running_offset = 0

    for chunk_start in range(0, n_images, chunk_size):
        chunk_stop = min(n_images, chunk_start + chunk_size)
        if sigma_rot_deg > 0.0:
            dots = np.clip(prior_dir_vecs[chunk_start:chunk_stop] @ dir_vecs.T, -1.0, 1.0)
            diffang_chunk = np.rad2deg(np.arccos(dots))
        else:
            diffang_chunk = None

        if sigma_psi_deg > 0.0:
            diffpsi_chunk = _wrapped_abs_diff_deg(
                psi_deg_grid[None, :],
                prior_psi_deg[chunk_start:chunk_stop, None],
            )
        else:
            diffpsi_chunk = None

        for local_idx, image_idx in enumerate(range(chunk_start, chunk_stop)):
            if sigma_rot_deg > 0.0:
                diffang_i = diffang_chunk[local_idx]
                dir_mask = diffang_i < cutoff_dir_deg
                dir_indices = np.flatnonzero(dir_mask).astype(np.int64)
                if dir_indices.size == 0:
                    dir_indices = np.array([int(np.argmin(diffang_i))], dtype=np.int64)
                    dir_log_prior = np.zeros(1, dtype=dtype)
                else:
                    dir_log_prior = _normalized_log_weights(diffang_i[dir_indices], biggest_sigma_deg)
            else:
                dir_indices = np.arange(n_pixels, dtype=np.int64)
                dir_log_prior = np.full(n_pixels, -np.log(max(n_pixels, 1)), dtype=dtype)

            if sigma_psi_deg > 0.0:
                diffpsi_i = diffpsi_chunk[local_idx]
                psi_mask = diffpsi_i < cutoff_psi_deg
                psi_indices = np.flatnonzero(psi_mask).astype(np.int64)
                if psi_indices.size == 0:
                    psi_indices = np.array([int(np.argmin(diffpsi_i))], dtype=np.int64)
                    psi_log_prior = np.zeros(1, dtype=dtype)
                else:
                    psi_log_prior = _normalized_log_weights(diffpsi_i[psi_indices], sigma_psi_deg)
            else:
                psi_indices = np.arange(int(grid_metadata["n_psi"]), dtype=np.int64)
                psi_log_prior = np.full(
                    psi_indices.shape[0],
                    -np.log(max(psi_indices.shape[0], 1)),
                    dtype=dtype,
                )

            local_ids = (psi_indices[:, None] * n_pixels + dir_indices[None, :]).reshape(-1).astype(np.int64)
            local_log_prior = (psi_log_prior[:, None] + dir_log_prior[None, :]).reshape(-1).astype(dtype)
            counts[image_idx] = int(local_ids.shape[0])
            running_offset += int(local_ids.shape[0])
            offsets[image_idx + 1] = running_offset
            rotation_ids_parts.append(local_ids)
            log_prior_parts.append(local_log_prior)

    rotation_ids_flat = _flat_parts(rotation_ids_parts, empty_shape=0, dtype=np.int64, cast=np.int64)
    rotation_log_priors_flat = _flat_parts(log_prior_parts, empty_shape=0, dtype=dtype)
    return offsets, counts, rotation_ids_flat, rotation_log_priors_flat


def _build_parent_expanded_local_entries(
    prior_rotations: np.ndarray,
    fine_healpix_order: int,
    sigma_rot: float,
    sigma_psi: float,
    *,
    oversampling_order: int,
    rotation_log_prior: np.ndarray | None = None,
    random_perturbation: float = 0.0,
    generate_relion_mstep_rotations: bool = False,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build RELION-style local support by expanding selected coarse parents.

    RELION local search first calls
    ``selectOrientationsWithNonZeroPriorProbability`` on the current coarse
    sampling object, then ``getOrientations`` expands those selected coarse
    direction/psi parents into oversampled children. The child orientations
    inherit the parent prior; the prior is not redistributed over children.
    """

    oversampling_order = int(oversampling_order)
    if oversampling_order <= 0:
        raise ValueError("oversampling_order must be positive for parent-expanded local support")
    fine_healpix_order = int(fine_healpix_order)
    parent_order = fine_healpix_order - oversampling_order
    if parent_order < 0:
        raise ValueError(
            "fine_healpix_order must be >= oversampling_order for parent-expanded local support; "
            f"got fine_healpix_order={fine_healpix_order}, oversampling_order={oversampling_order}"
        )

    parent_metadata = build_local_search_grid_metadata(parent_order)
    rotation_log_prior_np = None
    if rotation_log_prior is not None:
        rotation_log_prior_np = np.asarray(rotation_log_prior, dtype=dtype)
        expected_parent_size = rotation_grid_size(parent_order)
        if rotation_log_prior_np.shape[0] != expected_parent_size:
            raise ValueError(
                "rotation_log_prior must have one value per parent-grid rotation "
                f"({expected_parent_size}) for parent-expanded local search; got {rotation_log_prior_np.shape}"
            )
    parent_offsets, parent_counts, parent_ids_flat, parent_log_priors_flat = _build_factorized_local_entries(
        prior_rotations,
        parent_order,
        sigma_rot,
        sigma_psi,
        parent_metadata,
        dtype=dtype,
    )

    n_images = int(parent_counts.shape[0])
    offsets = np.zeros(n_images + 1, dtype=np.int64)
    counts = np.zeros(n_images, dtype=np.int32)
    rotation_ids_parts: list[np.ndarray] = []
    log_prior_parts: list[np.ndarray] = []
    rotations_parts: list[np.ndarray] = []
    mstep_rotations_parts: list[np.ndarray] = []
    source_eulers_parts = []
    running_offset = 0

    for image_idx in range(n_images):
        start = int(parent_offsets[image_idx])
        stop = int(parent_offsets[image_idx + 1])
        parent_ids = np.asarray(parent_ids_flat[start:stop], dtype=np.int64)
        parent_log_prior = np.asarray(parent_log_priors_flat[start:stop], dtype=dtype)
        if rotation_log_prior_np is not None:
            parent_log_prior = parent_log_prior + rotation_log_prior_np[parent_ids]
        oversampled = get_oversampled_rotation_grid_from_samples(
            parent_ids,
            parent_order,
            oversampling_order=oversampling_order,
            random_perturbation=float(random_perturbation),
            return_rotation_indices=True,
            return_source_eulers=True,
            return_mstep_rotations=bool(generate_relion_mstep_rotations),
            rotation_index_order="recovar",
            dtype=dtype,
        )
        source_eulers_parts.append(oversampled[-1])
        child_rotations, parent_map, child_ids = oversampled[:3]
        child_mstep_rotations = oversampled[3] if bool(generate_relion_mstep_rotations) else None
        parent_map = np.asarray(parent_map, dtype=np.int64)
        child_ids = np.asarray(child_ids, dtype=np.int64)
        child_log_prior = parent_log_prior[parent_map].astype(dtype, copy=False)

        counts[image_idx] = int(child_ids.shape[0])
        running_offset += int(child_ids.shape[0])
        offsets[image_idx + 1] = running_offset
        rotation_ids_parts.append(child_ids)
        log_prior_parts.append(child_log_prior)
        rotations_parts.append(np.asarray(child_rotations, dtype=dtype))
        if child_mstep_rotations is not None:
            mstep_rotations_parts.append(np.asarray(child_mstep_rotations, dtype=dtype))

    rotation_ids_flat = _flat_parts(rotation_ids_parts, empty_shape=0, dtype=np.int64, cast=np.int64)
    rotation_log_priors_flat = _flat_parts(log_prior_parts, empty_shape=0, dtype=dtype)
    rotations_flat = _flat_parts(rotations_parts, empty_shape=(0, 3, 3), dtype=dtype)
    mstep_rotations_flat = (
        np.concatenate(mstep_rotations_parts, axis=0)
        if mstep_rotations_parts
        else (np.zeros((0, 3, 3), dtype=dtype) if generate_relion_mstep_rotations else None)
    )
    source_eulers_flat = (
        np.concatenate(source_eulers_parts)
        if source_eulers_parts and all(x is not None for x in source_eulers_parts)
        else None
    )
    return (
        offsets,
        counts,
        rotation_ids_flat,
        rotation_log_priors_flat,
        rotations_flat,
        mstep_rotations_flat,
        source_eulers_flat,
    )


def _rotation_eulers_from_grid_metadata(
    rotation_ids: np.ndarray,
    grid_metadata,
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Return canonical RELION Euler angles for selected rotation ids only."""

    rotation_ids = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    if rotation_ids.size == 0:
        return np.zeros((0, 3), dtype=dtype)
    if "eulers_full" in grid_metadata:
        return np.asarray(grid_metadata["eulers_full"], dtype=dtype)[rotation_ids]
    if str(grid_metadata["mode"]) != "factorized":
        raise ValueError("Selected rotation eulers require factorized metadata or eulers_full")
    n_pixels = int(grid_metadata["n_pixels"])
    pixel_idx = rotation_ids % n_pixels
    psi_idx = rotation_ids // n_pixels
    return np.stack(
        [
            np.asarray(grid_metadata["rot_deg"], dtype=dtype)[pixel_idx],
            np.asarray(grid_metadata["tilt_deg"], dtype=dtype)[pixel_idx],
            np.asarray(grid_metadata["psi_deg"], dtype=dtype)[psi_idx],
        ],
        axis=1,
    ).astype(dtype, copy=False)


def _selected_rotation_matrices(
    rotation_ids: np.ndarray,
    rotation_grid_rotations: np.ndarray | None,
    grid_metadata,
    *,
    random_perturbation: float = 0.0,
    angular_sampling_deg: float | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build matrices for selected local ids without materializing the full grid."""

    rotation_ids = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    if rotation_ids.size == 0:
        return np.zeros((0, 3, 3), dtype=dtype)
    if rotation_grid_rotations is not None:
        return np.asarray(rotation_grid_rotations, dtype=dtype).reshape(-1, 3, 3)[rotation_ids]
    unique_ids, inverse = np.unique(rotation_ids, return_inverse=True)
    # Stage eulers at the requested dtype before apply_relion_rotation_perturbation_to_eulers
    # re-derives float64 internally regardless; truncating to float32 here
    # first (the previous unconditional default) would discard precision
    # that a dtype=float64 caller asked to keep, even though it leaves the
    # existing float32 default path's output bit-for-bit unchanged. Matches
    # the mstep sibling below, which already does this.
    selected_eulers = _rotation_eulers_from_grid_metadata(unique_ids, grid_metadata, dtype=dtype)
    if abs(float(random_perturbation)) > 1e-12:
        if angular_sampling_deg is None:
            raise ValueError("angular_sampling_deg is required when random_perturbation is nonzero")
        rotations, _ = apply_relion_rotation_perturbation_to_eulers(
            selected_eulers,
            float(random_perturbation),
            float(angular_sampling_deg),
            dtype=dtype,
        )
    else:
        # Preserve RELION's accelerated-path handoff: host RFLOAT inverse
        # matrices are cast to XFLOAT before scoring on the device (a no-op
        # cast under ACC_DOUBLE_PRECISION, when dtype=float64).
        rotations, _ = apply_relion_rotation_perturbation_to_eulers(
            selected_eulers,
            0.0,
            0.0,
            dtype=dtype,
        )
    return rotations.astype(dtype, copy=False)[inverse]


def _selected_mstep_rotation_matrices(
    rotation_ids: np.ndarray,
    rotation_grid_mstep_rotations: np.ndarray | None,
    grid_metadata,
    *,
    random_perturbation: float = 0.0,
    angular_sampling_deg: float | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build RELION host-path adjoint matrices for selected local ids."""

    rotation_ids = np.asarray(rotation_ids, dtype=np.int64).reshape(-1)
    if rotation_ids.size == 0:
        return np.zeros((0, 3, 3), dtype=dtype)
    if rotation_grid_mstep_rotations is not None:
        return np.asarray(rotation_grid_mstep_rotations, dtype=dtype).reshape(-1, 3, 3)[rotation_ids]
    unique_ids, inverse = np.unique(rotation_ids, return_inverse=True)
    selected_eulers = _rotation_eulers_from_grid_metadata(unique_ids, grid_metadata, dtype=np.float64)
    if angular_sampling_deg is None:
        if abs(float(random_perturbation)) > 1e-12:
            raise ValueError("angular_sampling_deg is required when random_perturbation is nonzero")
        angular_sampling_deg = 0.0
    mstep_rotations, _ = apply_relion_rotation_perturbation_to_eulers(
        selected_eulers,
        float(random_perturbation),
        float(angular_sampling_deg),
        dtype=dtype,
    )
    return np.asarray(mstep_rotations, dtype=dtype)[inverse]


def _flat_parts(parts, *, empty_shape, dtype, cast=None):
    """Concatenate per-image layout parts along axis 0, or the typed empty array when no image contributed.

    ``cast`` recasts the concatenation (the rotation ids are carried as int64).
    """

    if not parts:
        return np.zeros(empty_shape, dtype=dtype)
    flat = np.concatenate(parts, axis=0)
    return flat if cast is None else flat.astype(cast, copy=False)


def build_local_hypothesis_layout(
    prior_rotations: np.ndarray,
    rotation_grid_rotations: np.ndarray | None,
    sigma_rot: float,
    sigma_psi: float,
    healpix_order: int,
    translations: np.ndarray,
    prior_translations: np.ndarray,
    sigma_offset_angstrom: float,
    offset_range_pixels: float | None,
    voxel_size: float,
    *,
    grid_metadata,
    translation_prior_reference_translations: np.ndarray | None = None,
    rotation_log_prior: np.ndarray | None = None,
    rotation_grid_random_perturbation: float = 0.0,
    rotation_grid_angular_sampling_deg: float | None = None,
    local_parent_oversampling_order: int = 0,
    rotation_grid_mstep_rotations: np.ndarray | None = None,
    generate_relion_mstep_rotations: bool = False,
    dtype: np.dtype = np.float32,
) -> LocalHypothesisLayout:
    """Build exact per-image local neighborhoods and translation priors.

    ``dtype`` controls the precision of every rotation/translation/prior
    array built here (default float32, matching RELION's accelerated-GPU
    single-precision path). Pass ``np.float64`` to keep this local-search
    hypothesis grid genuinely double precision end to end; the caller is
    responsible for deriving this from ``use_float64_scoring`` /
    ``use_float64_projections`` so the default stays unchanged.
    """

    prior_rotations = np.asarray(prior_rotations, dtype=dtype)
    if rotation_grid_rotations is not None:
        rotation_grid_rotations = np.asarray(rotation_grid_rotations, dtype=dtype).reshape(-1, 3, 3)
    if rotation_grid_mstep_rotations is not None:
        rotation_grid_mstep_rotations = np.asarray(rotation_grid_mstep_rotations, dtype=dtype).reshape(-1, 3, 3)
        expected_rotation_count = (
            int(rotation_grid_rotations.shape[0])
            if rotation_grid_rotations is not None
            else int(grid_metadata["n_pixels"]) * int(grid_metadata["n_psi"])
        )
        if int(rotation_grid_mstep_rotations.shape[0]) != expected_rotation_count:
            raise ValueError(
                "rotation_grid_mstep_rotations must match the scoring grid size: "
                f"{rotation_grid_mstep_rotations.shape[0]} vs {expected_rotation_count}",
            )
    generate_relion_mstep_rotations = bool(
        generate_relion_mstep_rotations or rotation_grid_mstep_rotations is not None
    )
    translations = np.asarray(translations, dtype=dtype)
    prior_translations = np.asarray(prior_translations, dtype=dtype).reshape(-1, translations.shape[1])
    rotation_log_prior_np = None if rotation_log_prior is None else np.asarray(rotation_log_prior, dtype=dtype)

    source_eulers_flat = None
    rotations_flat_override = None
    mstep_rotations_flat_override = None
    if int(local_parent_oversampling_order) > 0:
        (
            offsets,
            counts,
            rotation_ids_flat,
            rotation_log_priors_flat,
            rotations_flat_override,
            mstep_rotations_flat_override,
            source_eulers_flat,
        ) = _build_parent_expanded_local_entries(
            prior_rotations,
            healpix_order,
            sigma_rot,
            sigma_psi,
            oversampling_order=int(local_parent_oversampling_order),
            rotation_log_prior=rotation_log_prior_np,
            random_perturbation=float(rotation_grid_random_perturbation),
            generate_relion_mstep_rotations=generate_relion_mstep_rotations,
            dtype=dtype,
        )
    elif str(grid_metadata["mode"]) == "factorized":
        offsets, counts, rotation_ids_flat, rotation_log_priors_flat = _build_factorized_local_entries(
            prior_rotations,
            healpix_order,
            sigma_rot,
            sigma_psi,
            grid_metadata,
            dtype=dtype,
        )
    else:
        n_images = int(prior_rotations.shape[0])
        offsets = np.zeros(n_images + 1, dtype=np.int64)
        counts = np.zeros(n_images, dtype=np.int32)
        rotation_ids_parts: list[np.ndarray] = []
        log_prior_parts: list[np.ndarray] = []

        for image_idx in range(n_images):
            local_ids, local_log_prior = get_local_rotation_grid_fast(
                prior_rotations[image_idx : image_idx + 1],
                sigma_rot,
                sigma_psi,
                healpix_order,
                sigma_cutoff=3.0,
                per_image=True,
                grid_metadata=grid_metadata,
            )
            local_ids = np.asarray(local_ids, dtype=np.int64).reshape(-1)
            local_log_prior = np.asarray(local_log_prior[0], dtype=dtype).reshape(-1)
            counts[image_idx] = int(local_ids.shape[0])
            offsets[image_idx + 1] = offsets[image_idx] + local_ids.shape[0]
            rotation_ids_parts.append(local_ids)
            log_prior_parts.append(local_log_prior)

        rotation_ids_flat = _flat_parts(rotation_ids_parts, empty_shape=0, dtype=np.int64, cast=np.int64)
        rotation_log_priors_flat = _flat_parts(log_prior_parts, empty_shape=0, dtype=dtype)
    if rotation_log_prior_np is not None and int(local_parent_oversampling_order) <= 0:
        if rotation_log_prior_np.shape[0] != int(grid_metadata["n_pixels"]) * int(grid_metadata["n_psi"]):
            raise ValueError(
                "rotation_log_prior must have one value per local-grid rotation "
                f"({int(grid_metadata['n_pixels']) * int(grid_metadata['n_psi'])}); "
                f"got {rotation_log_prior_np.shape}"
            )
        rotation_log_priors_flat = (
            rotation_log_priors_flat + rotation_log_prior_np[np.asarray(rotation_ids_flat, dtype=np.int64)]
        )
    rotations_flat = (
        rotations_flat_override
        if rotations_flat_override is not None
        else _selected_rotation_matrices(
            rotation_ids_flat,
            rotation_grid_rotations,
            grid_metadata,
            random_perturbation=rotation_grid_random_perturbation,
            angular_sampling_deg=rotation_grid_angular_sampling_deg,
            dtype=dtype,
        )
    )
    mstep_rotations_flat = None
    if generate_relion_mstep_rotations:
        mstep_rotations_flat = (
            mstep_rotations_flat_override
            if mstep_rotations_flat_override is not None
            else _selected_mstep_rotation_matrices(
                rotation_ids_flat,
                rotation_grid_mstep_rotations,
                grid_metadata,
                random_perturbation=rotation_grid_random_perturbation,
                angular_sampling_deg=rotation_grid_angular_sampling_deg,
                dtype=dtype,
            )
        )
    translation_grid = translations
    translation_parent = None
    if int(local_parent_oversampling_order) > 0:
        translation_grid, translation_parent = get_oversampled_translation_grid(
            translations,
            infer_translation_step(translations),
            oversampling_order=int(local_parent_oversampling_order),
        )
        translation_grid = np.asarray(translation_grid, dtype=dtype)
        translation_parent = np.asarray(translation_parent, dtype=np.int32)

    reference_translations = (
        np.asarray(translation_prior_reference_translations, dtype=dtype)
        if translation_prior_reference_translations is not None
        else translations
    )

    coarse_translation_log_priors = make_relion_translation_log_prior(
        reference_translations,
        voxel_size,
        sigma_offset_angstrom,
        prior_translations,
        offset_range_pixels=offset_range_pixels,
        dtype=dtype,
    ).astype(dtype, copy=False)
    if translation_parent is None:
        translation_log_priors = coarse_translation_log_priors
    else:
        translation_log_priors = _fine_translation_log_prior(
            coarse_translation_log_priors,
            translation_parent,
            int(prior_translations.shape[0]),
            int(translation_grid.shape[0]),
            dtype=dtype,
        )

    if rotation_grid_rotations is not None:
        n_global_rotations = int(rotation_grid_rotations.shape[0])
    else:
        n_global_rotations = int(grid_metadata["n_pixels"]) * int(grid_metadata["n_psi"])

    if (
        source_eulers_flat is None
        and int(local_parent_oversampling_order) == 0
        and rotation_grid_rotations is None
        and str(grid_metadata["mode"]) == "factorized"
    ):
        source_eulers_flat = get_oversampled_rotation_grid_from_samples(
            rotation_ids_flat,
            healpix_order,
            oversampling_order=0,
            random_perturbation=float(rotation_grid_random_perturbation),
            return_source_eulers=True,
            dtype=dtype,
        )[-1]
    return LocalHypothesisLayout(
        n_global_rotations=n_global_rotations,
        n_pixels=int(grid_metadata["n_pixels"]),
        n_psi=int(grid_metadata["n_psi"]),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=translation_grid,
        translation_log_priors=np.asarray(translation_log_priors, dtype=dtype),
        mstep_rotations_flat=mstep_rotations_flat,
        source_eulers_flat=source_eulers_flat,
    )


def build_local_adaptive_pass2_hypothesis_layout(
    parent_layout: LocalHypothesisLayout,
    significant_sample_indices,
    parent_healpix_order: int,
    *,
    oversampling_order: int,
    random_perturbation: float = 0.0,
    translation_step: float | None = None,
    dtype: np.dtype = np.float32,
) -> LocalHypothesisLayout:
    """Expand local adaptive parent support while preserving significant pairs.

    RELION's adaptive local pass 2 first expands significant coarse orientation
    parents, then only scores fine translation children for coarse
    ``(orientation, translation)`` pairs that survived pass 1. ``parent_layout``
    carries the image-specific local Gaussian priors from that coarse pass.
    """

    oversampling_order = int(oversampling_order)
    if oversampling_order <= 0:
        raise ValueError("oversampling_order must be positive for adaptive local pass 2")
    parent_healpix_order = int(parent_healpix_order)
    fine_healpix_order = parent_healpix_order + oversampling_order
    n_images = int(parent_layout.n_images)
    if len(significant_sample_indices) != n_images:
        raise ValueError(
            "significant_sample_indices must have one entry per image; "
            f"got {len(significant_sample_indices)} for {n_images} images",
        )

    coarse_translations = np.asarray(parent_layout.translation_grid, dtype=dtype)
    n_coarse_trans = int(coarse_translations.shape[0])
    if translation_step is None:
        translation_step = infer_translation_step(coarse_translations)
    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        coarse_translations,
        float(translation_step),
        oversampling_order=oversampling_order,
    )
    fine_translations = np.asarray(fine_translations, dtype=dtype)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    n_fine_trans = int(fine_translations.shape[0])

    offsets = np.zeros(n_images + 1, dtype=np.int64)
    counts = np.zeros(n_images, dtype=np.int32)
    rotations_parts: list[np.ndarray] = []
    source_eulers_parts: list[np.ndarray | None] = []
    mstep_rotations_parts: list[np.ndarray] = []
    rotation_ids_parts: list[np.ndarray] = []
    posterior_ids_parts: list[np.ndarray] = []
    log_prior_parts: list[np.ndarray] = []
    sample_mask_parts: list[np.ndarray | None] = []

    n_parent_global = int(parent_layout.n_global_rotations)
    running_offset = 0
    for image_idx, sig_samples in enumerate(significant_sample_indices):
        parent_start = int(parent_layout.rotation_offsets[image_idx])
        parent_stop = int(parent_layout.rotation_offsets[image_idx + 1])
        local_parent_ids = np.asarray(parent_layout.rotation_ids_flat[parent_start:parent_stop], dtype=np.int64)
        local_parent_log_prior = np.asarray(
            parent_layout.rotation_log_priors_flat[parent_start:parent_stop],
            dtype=dtype,
        )
        if local_parent_ids.size == 0:
            raise ValueError(f"Image {image_idx} has no local parent rotations for adaptive pass 2")

        if sig_samples is None:
            unique_rot = local_parent_ids
            coarse_rot = local_parent_ids
            coarse_trans = np.tile(np.arange(n_coarse_trans, dtype=np.int32), local_parent_ids.size)
            use_full_candidate_mask = True
        else:
            sig_samples = np.asarray(sig_samples, dtype=np.int64).reshape(-1)
            if sig_samples.size == 0:
                unique_rot = local_parent_ids
                coarse_rot = local_parent_ids
                coarse_trans = np.tile(np.arange(n_coarse_trans, dtype=np.int32), local_parent_ids.size)
                use_full_candidate_mask = True
            else:
                coarse_rot = sig_samples // n_coarse_trans
                coarse_trans = sig_samples % n_coarse_trans
                unique_rot = np.unique(coarse_rot).astype(np.int64, copy=False)
                use_full_candidate_mask = False

        if np.any(unique_rot < 0) or np.any(unique_rot >= n_parent_global):
            raise ValueError(f"Image {image_idx} has significant rotation ids outside the parent grid")
        selected_parent_log_prior, matched_parent_ids = _lookup_values_by_id(
            local_parent_ids,
            local_parent_log_prior,
            unique_rot,
        )
        if not np.all(matched_parent_ids):
            missing = unique_rot[~matched_parent_ids]
            raise ValueError(
                f"Image {image_idx} has significant rotations outside its local parent support: {missing[:8].tolist()}"
            )

        oversampled_rots, parent_map, oversampled_rot_indices, oversampled_mstep_rots, source_eulers = (
            get_oversampled_rotation_grid_from_samples(
                unique_rot,
                parent_healpix_order,
                oversampling_order=oversampling_order,
                random_perturbation=float(random_perturbation),
                return_rotation_indices=True,
                return_mstep_rotations=True,
                return_source_eulers=True,
                rotation_index_order="recovar",
                dtype=dtype,
            )
        )
        oversampled_rots = np.asarray(oversampled_rots, dtype=dtype)
        oversampled_mstep_rots = np.asarray(oversampled_mstep_rots, dtype=dtype)
        parent_map = np.asarray(parent_map, dtype=np.int32)
        oversampled_rot_indices = np.asarray(oversampled_rot_indices, dtype=np.int32)
        parent_posterior_ids = unique_rot[parent_map].astype(np.int32, copy=False)

        if use_full_candidate_mask:
            sample_mask = None
        else:
            local_idx_per_sample, matched_coarse_rot = _positions_in_sorted_unique_ids(unique_rot, coarse_rot)
            if not np.all(matched_coarse_rot):
                raise ValueError(f"Image {image_idx} has significant samples outside unique parent rotations")
            significance_mask_coarse = np.zeros((unique_rot.shape[0], n_coarse_trans), dtype=bool)
            significance_mask_coarse[local_idx_per_sample, coarse_trans] = True
            sample_mask = significance_mask_coarse[parent_map][:, fine_translation_parent]

        if sample_mask is not None and not np.any(sample_mask):
            raise ValueError(f"Image {image_idx} has no valid adaptive local pass-2 candidates")

        counts[image_idx] = int(oversampled_rots.shape[0])
        running_offset += int(oversampled_rots.shape[0])
        offsets[image_idx + 1] = running_offset
        rotations_parts.append(oversampled_rots)
        source_eulers_parts.append(source_eulers)
        mstep_rotations_parts.append(oversampled_mstep_rots)
        rotation_ids_parts.append(oversampled_rot_indices)
        posterior_ids_parts.append(parent_posterior_ids)
        log_prior_parts.append(selected_parent_log_prior[parent_map].astype(dtype, copy=False))
        sample_mask_parts.append(None if sample_mask is None else np.packbits(sample_mask, axis=1, bitorder="little"))

    fine_metadata = build_local_search_grid_metadata(fine_healpix_order)
    rotations_flat = _flat_parts(rotations_parts, empty_shape=(0, 3, 3), dtype=dtype)
    mstep_rotations_flat = _flat_parts(mstep_rotations_parts, empty_shape=(0, 3, 3), dtype=dtype)
    rotation_ids_flat = _flat_parts(rotation_ids_parts, empty_shape=0, dtype=np.int64, cast=np.int64)
    posterior_ids_flat = _flat_parts(posterior_ids_parts, empty_shape=0, dtype=np.int32)
    rotation_log_priors_flat = _flat_parts(log_prior_parts, empty_shape=0, dtype=dtype)
    if not sample_mask_parts:
        sample_mask_bits = np.zeros((0, (n_fine_trans + 7) // 8), dtype=np.uint8)
    elif all(sample_mask is None for sample_mask in sample_mask_parts):
        # ``None`` is the exact-local engine's compact representation of full
        # per-rotation/per-translation support. Avoid materializing massive
        # all-ones masks for RELION full-parent local pass 2.
        sample_mask_bits = None
    else:
        sample_mask_bits = np.concatenate(
            [
                np.packbits(np.ones((int(count), n_fine_trans), dtype=bool), axis=1, bitorder="little")
                if sample_mask is None else sample_mask
                for sample_mask, count in zip(sample_mask_parts, counts, strict=True)
            ],
            axis=0,
        )
    return LocalHypothesisLayout(
        n_global_rotations=rotation_grid_size(parent_healpix_order),
        n_pixels=int(fine_metadata["n_pixels"]),
        n_psi=int(fine_metadata["n_psi"]),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        source_eulers_flat=(
            np.concatenate(source_eulers_parts)
            if source_eulers_parts and all(x is not None for x in source_eulers_parts)
            else (np.empty((0, 3), dtype=np.float64) if not source_eulers_parts else None)
        ),
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=fine_translations,
        translation_log_priors=np.asarray(parent_layout.translation_log_priors, dtype=dtype)[
            :, fine_translation_parent
        ],
        rotation_posterior_ids_flat=posterior_ids_flat,
        sample_mask_bits=sample_mask_bits,
        mstep_rotations_flat=mstep_rotations_flat,
    )


def _lookup_values_by_id(ids: np.ndarray, values: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``values`` for integer ids without allocating a global id table."""

    ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
    values_np = np.asarray(values)
    query_np = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    if query_np.size == 0:
        return values_np[:0], np.ones(0, dtype=bool)
    if ids_np.size == 0:
        return values_np[:0], np.zeros(query_np.shape, dtype=bool)

    order = np.argsort(ids_np, kind="stable")
    sorted_ids = ids_np[order]
    # Match the previous dense table behavior for duplicate ids: later writes
    # won, so search to the right and take the last matching entry.
    pos = np.searchsorted(sorted_ids, query_np, side="right") - 1
    valid = pos >= 0
    matched = np.zeros(query_np.shape, dtype=bool)
    if np.any(valid):
        matched[valid] = sorted_ids[pos[valid]] == query_np[valid]
    if not np.all(matched):
        return values_np[:0], matched
    return values_np[order[pos]], matched


def _positions_in_sorted_unique_ids(sorted_ids: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map query ids to row positions in a sorted unique id array."""

    sorted_np = np.asarray(sorted_ids, dtype=np.int64).reshape(-1)
    query_np = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    if query_np.size == 0:
        return np.zeros(0, dtype=np.int64), np.ones(0, dtype=bool)
    pos = np.searchsorted(sorted_np, query_np)
    valid = pos < sorted_np.size
    matched = np.zeros(query_np.shape, dtype=bool)
    if np.any(valid):
        matched[valid] = sorted_np[pos[valid]] == query_np[valid]
    return pos.astype(np.int64, copy=False), matched


def _fine_translation_log_prior(
    translation_log_prior: np.ndarray | None,
    fine_translation_parent: np.ndarray,
    n_images: int,
    n_fine_translations: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if translation_log_prior is None:
        return np.zeros((n_images, n_fine_translations), dtype=dtype)
    translation_log_prior_np = np.asarray(translation_log_prior, dtype=dtype)
    if translation_log_prior_np.ndim == 1:
        fine = translation_log_prior_np[fine_translation_parent]
        return np.broadcast_to(fine[None, :], (n_images, n_fine_translations)).astype(dtype, copy=False)
    if translation_log_prior_np.ndim == 2:
        if translation_log_prior_np.shape[0] != n_images:
            raise ValueError(
                "translation_log_prior must have one row per image when 2D; "
                f"got {translation_log_prior_np.shape[0]} rows for {n_images} images",
            )
        return translation_log_prior_np[:, fine_translation_parent].astype(dtype, copy=False)
    raise ValueError(f"translation_log_prior must be 1D or 2D, got {translation_log_prior_np.ndim} dimensions")


def _pass2_translation_log_prior(
    translation_log_prior: np.ndarray | None,
    fine_translation_log_prior: np.ndarray | None,
    fine_translation_parent: np.ndarray,
    n_images: int,
    n_fine_translations: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if fine_translation_log_prior is None:
        return _fine_translation_log_prior(
            translation_log_prior,
            fine_translation_parent,
            n_images,
            n_fine_translations,
            dtype=dtype,
        )
    if translation_log_prior is not None:
        raise ValueError("translation_log_prior and fine_translation_log_prior are mutually exclusive")

    prior_np = np.asarray(fine_translation_log_prior, dtype=dtype)
    if prior_np.ndim == 1:
        if prior_np.shape[0] != n_fine_translations:
            raise ValueError(
                "fine_translation_log_prior must have one value per fine translation; "
                f"got {prior_np.shape[0]} values for {n_fine_translations} translations",
            )
        return np.broadcast_to(prior_np[None, :], (n_images, n_fine_translations)).astype(dtype, copy=False)
    if prior_np.ndim == 2:
        if prior_np.shape != (n_images, n_fine_translations):
            raise ValueError(
                f"fine_translation_log_prior must have shape ({n_images}, {n_fine_translations}); got {prior_np.shape}",
            )
        return prior_np.astype(dtype, copy=False)
    raise ValueError(f"fine_translation_log_prior must be 1D or 2D, got {prior_np.ndim} dimensions")


def build_pass2_hypothesis_layout(
    significant_sample_indices,
    n_coarse_rotations: int,
    n_coarse_translations: int,
    nside_level: int,
    translations: np.ndarray,
    *,
    oversampling_order: int,
    translation_step: float | None = None,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    fine_translation_log_prior: np.ndarray | None = None,
    random_perturbation: float = 0.0,
    rotation_index_order: str = "recovar",
    allow_empty: bool = False,
    dtype: np.dtype = np.float32,
) -> LocalHypothesisLayout:
    """Build exact-local layout for RELION adaptive pass-2 hypotheses.

    Pass 2 is not a Gaussian local search around one previous best pose. RELION
    oversamples the coarse ``(rotation, translation)`` samples that survived
    pass 1. The exact local engine can score the same structure if each image
    carries its own oversampled rotations plus a sparse ``(R, T)`` mask.
    """

    dtype = np.dtype(dtype)
    translations_np = np.asarray(translations, dtype=dtype)
    if translation_step is None:
        translation_step = infer_translation_step(translations_np)
    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        translations_np,
        float(translation_step),
        oversampling_order=oversampling_order,
    )
    fine_translations = np.asarray(fine_translations, dtype=dtype)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    n_fine_translations = int(fine_translations.shape[0])
    n_images = len(significant_sample_indices)
    rotation_log_prior_np = None if rotation_log_prior is None else np.asarray(rotation_log_prior, dtype=dtype)

    coarse_rows = []
    for image_idx, sig_samples in enumerate(significant_sample_indices):
        if sig_samples is None:
            unique_rot = np.arange(n_coarse_rotations, dtype=np.int32)
            coarse_rot = unique_rot
            coarse_trans = None
            use_full_candidate_mask = True
        else:
            sig_samples = significant_sample_ids(
                sig_samples, int(n_coarse_rotations) * int(n_coarse_translations),
            )
            if sig_samples.size == 0:
                if not allow_empty:
                    raise ValueError(f"Image {image_idx} has no significant coarse samples for sparse pass 2")
                unique_rot = np.zeros(1, dtype=np.int64)
                coarse_rot = unique_rot
                coarse_trans = np.zeros(0, dtype=np.int64)
            else:
                coarse_rot = sig_samples // int(n_coarse_translations)
                coarse_trans = sig_samples % int(n_coarse_translations)
                unique_rot = np.unique(coarse_rot).astype(np.int64, copy=False)
            use_full_candidate_mask = False

        if np.any(unique_rot < 0) or np.any(unique_rot >= int(n_coarse_rotations)):
            raise ValueError(f"Image {image_idx} has significant rotation ids outside the coarse grid")

        coarse_rows.append((unique_rot, coarse_rot, coarse_trans, use_full_candidate_mask))

    # Child orientations depend on the coarse sample and iteration perturbation,
    # not on the image. Generate only the requested union once, then gather each
    # image's rows in the same parent/child order as independent generation.
    shared_parent_ids = (
        np.unique(np.concatenate([row[0] for row in coarse_rows]))
        if coarse_rows else np.zeros(0, dtype=np.int64)
    )
    shared_rotations, shared_parent_map, shared_rotation_ids, shared_eulers = (
        get_oversampled_rotation_grid_from_samples(
            shared_parent_ids,
            int(nside_level),
            oversampling_order=oversampling_order,
            random_perturbation=random_perturbation,
            return_rotation_indices=True,
            return_source_eulers=True,
            rotation_index_order=rotation_index_order,
            dtype=dtype,
        )
    )
    children_per_parent = 8 ** int(oversampling_order)
    if not np.array_equal(
        shared_parent_map,
        np.repeat(np.arange(shared_parent_ids.size), children_per_parent),
    ):
        raise RuntimeError("Pass-2 oversampling must retain contiguous children per parent")
    child_offsets = np.arange(children_per_parent, dtype=np.int64)

    # Allocate the final layout once; retaining every image's arrays until
    # concatenation otherwise doubles the largest host allocations.
    counts = np.asarray([row[0].size * children_per_parent for row in coarse_rows], dtype=np.int32)
    offsets = np.zeros(n_images + 1, dtype=np.int64)
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    n_rows = int(offsets[-1])
    rotations_flat = np.empty_like(shared_rotations, shape=(n_rows, 3, 3), dtype=dtype)
    rotation_ids_flat = np.empty(n_rows, dtype=np.int64)
    posterior_ids_flat = np.empty(n_rows, dtype=np.int32)
    rotation_log_priors_flat = np.empty(n_rows, dtype=dtype)
    sample_mask_bits = np.empty((n_rows, (n_fine_translations + 7) // 8), dtype=np.uint8)
    source_eulers_flat = np.empty_like(shared_eulers, shape=(n_rows, 3)) if shared_eulers is not None else None
    if not n_images:
        source_eulers_flat = np.empty((0, 3), dtype=np.float64)

    for image_idx, (unique_rot, coarse_rot, coarse_trans, use_full_candidate_mask) in enumerate(coarse_rows):
        shared_positions = np.searchsorted(shared_parent_ids, unique_rot)
        rows = (shared_positions[:, None] * children_per_parent + child_offsets).reshape(-1)
        oversampled_rots = np.asarray(shared_rotations[rows], dtype=dtype)
        oversampled_rot_indices = np.asarray(shared_rotation_ids[rows], dtype=np.int64)
        parent_map = np.repeat(np.arange(unique_rot.size, dtype=np.int32), children_per_parent)
        coarse_parent_ids = unique_rot[parent_map].astype(np.int32, copy=False)

        if rotation_log_prior_np is None:
            local_rotation_log_prior = np.zeros(oversampled_rots.shape[0], dtype=dtype)
        else:
            local_rotation_log_prior = rotation_log_prior_np[unique_rot][parent_map].astype(dtype, copy=False)

        if use_full_candidate_mask:
            sample_mask = np.ones((oversampled_rots.shape[0], n_fine_translations), dtype=bool)
        else:
            sample_mask = np.zeros((oversampled_rots.shape[0], n_fine_translations), dtype=bool)
            if coarse_trans.size:
                # Vectorized replacement of the inner-image Python loop over
                # unique_rot. Build a (n_unique_rot, n_coarse_trans) mask of
                # significant (rot, trans) pairs, then expand by parent_map
                # and fine_translation_parent. At 50k/256 K=1 this cuts pass2
                # layout-build time from ~10 s/iter to <1 s.
                significance_mask_coarse = np.zeros(
                    (unique_rot.shape[0], int(n_coarse_translations)),
                    dtype=bool,
                )
                local_idx_per_sample, matched_coarse_rot = _positions_in_sorted_unique_ids(unique_rot, coarse_rot)
                if not np.all(matched_coarse_rot):
                    raise ValueError(f"Image {image_idx} has significant samples outside unique coarse rotations")
                significance_mask_coarse[local_idx_per_sample, coarse_trans] = True
                sample_mask = significance_mask_coarse[parent_map][:, fine_translation_parent]

        if not np.any(sample_mask) and not allow_empty:
            raise ValueError(f"Image {image_idx} has no valid sparse pass-2 candidates after oversampling")

        target = slice(offsets[image_idx], offsets[image_idx + 1])
        rotations_flat[target] = oversampled_rots
        rotation_ids_flat[target] = oversampled_rot_indices
        posterior_ids_flat[target] = coarse_parent_ids
        rotation_log_priors_flat[target] = local_rotation_log_prior
        sample_mask_bits[target] = np.packbits(sample_mask, axis=1, bitorder="little")
        if source_eulers_flat is not None:
            source_eulers_flat[target] = shared_eulers[rows]

    n_pixels = 12 * (2 ** int(nside_level)) ** 2

    return LocalHypothesisLayout(
        n_global_rotations=int(n_coarse_rotations),
        n_pixels=int(n_pixels),
        n_psi=int(rotation_grid_n_in_planes(int(nside_level))),
        rotation_offsets=offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=rotations_flat,
        source_eulers_flat=source_eulers_flat,
        rotation_log_priors_flat=rotation_log_priors_flat,
        rotation_counts=counts,
        translation_grid=fine_translations,
        translation_log_priors=_pass2_translation_log_prior(
            translation_log_prior,
            fine_translation_log_prior,
            fine_translation_parent,
            n_images,
            n_fine_translations,
            dtype=dtype,
        ),
        rotation_posterior_ids_flat=posterior_ids_flat,
        sample_mask_bits=sample_mask_bits,
    )


@dataclass(frozen=True)
class LocalBucketPlan:
    """Image order and padded capacities, without per-candidate array storage."""

    image_indices: np.ndarray
    bucket_image_count: int
    bucket_rotation_count: int
    actual_rotation_counts: np.ndarray


class LocalBucketSequence(Sequence):
    """Indexable bucket plans that retain no materialized candidate arrays."""

    def __init__(self, layout: LocalHypothesisLayout, plans: Sequence[LocalBucketPlan]):
        self.layout = layout
        self.plans = plans

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return LocalBucketSequence(self.layout, self.plans[index])
        return _materialize_local_bucket(self.layout, self.plans[index])


def plan_local_hypothesis_buckets(
    layout: LocalHypothesisLayout,
    image_batch_size: int,
    rotation_block_size: int,
    *,
    max_hypotheses_per_microbatch: int = 32768,
    unify_bucket_sizes: bool | None = None,
    large_bucket_quantum: int | None = None,
    preserve_image_order: bool = False,
    exact_local_bucket_radix: int | None = None,
    consecutive_mixed_bucket_size: int | None = None,
    image_capacity_ladder=None,
) -> list[LocalBucketPlan]:
    """Plan static bucket shapes and image order without allocating candidate arrays."""

    image_batch_size = int(max(1, image_batch_size))
    max_hypotheses_per_microbatch = int(max(1, max_hypotheses_per_microbatch))
    image_capacity_ladder = _planner_image_capacity_ladder(image_capacity_ladder)
    rotations_dtype = np.asarray(layout.rotations_flat).dtype
    mstep_rotations_flat = (
        np.asarray(layout.rotations_flat, dtype=rotations_dtype)
        if layout.mstep_rotations_flat is None
        else np.asarray(layout.mstep_rotations_flat)
    )
    if mstep_rotations_flat.shape != np.asarray(layout.rotations_flat).shape:
        raise ValueError(
            "mstep_rotations_flat must match rotations_flat shape: "
            f"{mstep_rotations_flat.shape} vs {np.asarray(layout.rotations_flat).shape}",
        )
    resolved_large_bucket_quantum = _exact_local_large_bucket_quantum(rotation_block_size, large_bucket_quantum)
    bucket_sizes = np.asarray(
        [
            _exact_bucket_rotation_size(
                int(count),
                rotation_block_size,
                large_bucket_quantum=resolved_large_bucket_quantum,
                exact_local_bucket_radix=exact_local_bucket_radix,
            )
            for count in layout.rotation_counts
        ],
        dtype=np.int32,
    )
    # ``RECOVAR_LOCAL_BUCKET_UNIFY=1`` forces all images to share a single
    # rotation-count bucket class so the JIT only compiles one shape per
    # layout. At 50k/256 K=1 this collapses ~13 unique shapes (per-image
    # significant rotation counts vary widely across iters) into 1, removing
    # per-bucket JIT compilation overhead. Memory cost: smaller-significance
    # images carry extra rotation padding.
    if unify_bucket_sizes is None:
        unify_bucket_sizes = os.environ.get("RECOVAR_LOCAL_BUCKET_UNIFY", "").lower() in {"1", "true", "yes", "on"}
    if consecutive_mixed_bucket_size is not None:
        consecutive_mixed_bucket_size = int(consecutive_mixed_bucket_size)
        if consecutive_mixed_bucket_size <= 0:
            raise ValueError("consecutive_mixed_bucket_size must be positive")
        if not preserve_image_order:
            raise ValueError("consecutive mixed buckets require preserved image order")
        if bool(unify_bucket_sizes):
            raise ValueError("consecutive mixed buckets cannot use run-global bucket unification")
    if bucket_sizes.size and bool(unify_bucket_sizes):
        bucket_sizes = np.full_like(bucket_sizes, int(bucket_sizes.max()))
    processing_order = (
        np.arange(layout.n_images, dtype=np.int32)
        if preserve_image_order
        else np.lexsort((layout.rotation_counts, bucket_sizes)).astype(np.int32)
    )

    if processing_order.size == 0:
        return []

    planned_groups: list[tuple[np.ndarray, int, int]] = []
    if consecutive_mixed_bucket_size is not None:
        for plan in _plan_consecutive_padded_batches(
            bucket_sizes,
            processing_order=processing_order,
            target_items_per_batch=consecutive_mixed_bucket_size,
            max_items_per_batch=image_batch_size,
            max_padded_values_per_batch=max_hypotheses_per_microbatch,
            item_alignment=3,
        ):
            planned_groups.append(
                (
                    np.asarray(plan.item_indices, dtype=np.int32),
                    int(plan.padded_size),
                    int(plan.padded_item_capacity),
                )
            )
    elif preserve_image_order:
        boundaries = np.flatnonzero(
            np.r_[True, bucket_sizes[processing_order][1:] != bucket_sizes[processing_order][:-1], True]
        )
        bucket_groups = [
            processing_order[start:stop] for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
    else:
        bucket_groups = [
            processing_order[bucket_sizes[processing_order] == bucket_size]
            for bucket_size in np.unique(bucket_sizes[processing_order])
        ]
    if consecutive_mixed_bucket_size is None:
        for bucket_images in bucket_groups:
            bucket_size = int(bucket_sizes[int(bucket_images[0])])
            max_images = max(1, min(image_batch_size, max_hypotheses_per_microbatch // int(bucket_size)))
            if preserve_image_order and max_images >= 3:
                # RELION's InitialModel default processes pools of three particles.
                # Keep static-shape FFI boundaries on pool boundaries so a new call
                # never changes which physical particles may update BPref together.
                max_images = max(3, (max_images // 3) * 3)
            else:
                # Only when the pool rule is not in force: ladder rungs are not
                # multiples of three, so snapping here would move an InitialModel
                # FFI boundary off a particle pool.
                max_images = _ladder_image_capacity(max_images, image_capacity_ladder)
            # Every group of this rotation class keeps the SAME capacity, the
            # remainder group included. Giving the remainder its own smaller rung
            # saves a little padding and costs a whole extra compiled program per
            # rotation class, which measured 7.96 s -> 16.26 s of local-engine
            # compile at the 10k/256 order-4 state.
            for start in range(0, bucket_images.shape[0], max_images):
                planned_groups.append(
                    (
                        np.asarray(bucket_images[start : start + max_images], dtype=np.int32),
                        bucket_size,
                        max_images,
                    )
                )

    return [
        LocalBucketPlan(indices, max_images, size, layout.rotation_counts[indices].astype(np.int32, copy=False))
        for indices, size, max_images in planned_groups
    ]


def _materialize_local_bucket(layout: LocalHypothesisLayout, plan: LocalBucketPlan) -> LocalBucketSpec:
    """Construct one bucket with the layout dtypes and planned physical order."""

    image_indices = plan.image_indices
    bucket_size = plan.bucket_rotation_count
    max_images = plan.bucket_image_count
    rotations_dtype = np.asarray(layout.rotations_flat).dtype
    mstep_rotations_flat = (
        np.asarray(layout.rotations_flat, dtype=rotations_dtype)
        if layout.mstep_rotations_flat is None
        else np.asarray(layout.mstep_rotations_flat)
    )
    actual_counts = plan.actual_rotation_counts
    batch_size = int(image_indices.shape[0])
    padded_rotations = np.broadcast_to(
        np.eye(3, dtype=rotations_dtype),
        (batch_size, int(bucket_size), 3, 3),
    ).copy()
    padded_mstep_rotations = (
        None
        if layout.mstep_rotations_flat is None
        else np.broadcast_to(
            np.eye(3, dtype=mstep_rotations_flat.dtype),
            (batch_size, int(bucket_size), 3, 3),
        ).copy()
    )
    padded_rotation_ids = np.full((batch_size, int(bucket_size)), -1, dtype=np.int64)
    padded_log_prior = np.full(
        (batch_size, int(bucket_size)), -1e30, dtype=np.asarray(layout.rotation_log_priors_flat).dtype
    )
    padded_mask = np.zeros((batch_size, int(bucket_size)), dtype=bool)
    padded_posterior_ids = (
        None
        if layout.rotation_posterior_ids_flat is None
        else np.full((batch_size, int(bucket_size)), -1, dtype=np.int32)
    )
    padded_sample_mask = (
        None
        if layout.sample_mask_bits is None
        else np.zeros(
            (batch_size, int(bucket_size), int(layout.translation_grid.shape[0])),
            dtype=bool,
        )
    )

    padded_source_eulers = (
        None if layout.source_eulers_flat is None else np.zeros((batch_size, int(bucket_size), 3), dtype=np.float64)
    )

    for row, image_idx in enumerate(image_indices.tolist()):
        start_off = int(layout.rotation_offsets[image_idx])
        end_off = int(layout.rotation_offsets[image_idx + 1])
        count = end_off - start_off
        padded_rotations[row, :count] = layout.rotations_flat[start_off:end_off]
        if padded_source_eulers is not None:
            padded_source_eulers[row, :count] = layout.source_eulers_flat[start_off:end_off]
        if padded_mstep_rotations is not None:
            padded_mstep_rotations[row, :count] = mstep_rotations_flat[start_off:end_off]
        padded_rotation_ids[row, :count] = layout.rotation_ids_flat[start_off:end_off]
        padded_log_prior[row, :count] = layout.rotation_log_priors_flat[start_off:end_off]
        padded_mask[row, :count] = True
        if padded_posterior_ids is not None:
            padded_posterior_ids[row, :count] = layout.rotation_posterior_ids_flat[start_off:end_off]
        if padded_sample_mask is not None:
            padded_sample_mask[row, :count, :] = layout.sample_mask_rows(start_off, end_off)

    return LocalBucketSpec(
        image_indices=image_indices,
        bucket_image_count=int(max_images),
        bucket_rotation_count=int(bucket_size),
        actual_rotation_counts=actual_counts,
        local_rotation_ids=padded_rotation_ids,
        local_rotations=padded_rotations,
        local_source_eulers=padded_source_eulers,
        local_rotation_log_prior=padded_log_prior,
        local_rotation_mask=padded_mask,
        translation_log_prior=np.asarray(layout.translation_log_priors[image_indices]),
        local_mstep_rotations=padded_mstep_rotations,
        local_rotation_posterior_ids=padded_posterior_ids,
        local_sample_mask=padded_sample_mask,
    )


def bucket_local_hypothesis_layout(
    layout: LocalHypothesisLayout,
    image_batch_size: int,
    rotation_block_size: int,
    *,
    max_hypotheses_per_microbatch: int = 32768,
    unify_bucket_sizes: bool | None = None,
    large_bucket_quantum: int | None = None,
    preserve_image_order: bool = False,
    exact_local_bucket_radix: int | None = None,
    consecutive_mixed_bucket_size: int | None = None,
    image_capacity_ladder=None,
) -> list[LocalBucketSpec]:
    """Materialize all local buckets; use plans for bounded-memory iteration."""

    plans = plan_local_hypothesis_buckets(
        layout,
        image_batch_size,
        rotation_block_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        unify_bucket_sizes=unify_bucket_sizes,
        large_bucket_quantum=large_bucket_quantum,
        preserve_image_order=preserve_image_order,
        exact_local_bucket_radix=exact_local_bucket_radix,
        consecutive_mixed_bucket_size=consecutive_mixed_bucket_size,
        image_capacity_ladder=image_capacity_ladder,
    )
    return [_materialize_local_bucket(layout, plan) for plan in plans]


def _plan_local_bucket_groups(
    rotation_counts: np.ndarray,
    bucket_sizes: np.ndarray,
    *,
    image_batch_size: int,
    max_hypotheses_per_microbatch: int,
    preserve_image_order: bool,
    consecutive_mixed_bucket_size: int | None,
    image_capacity_ladder=None,
) -> list[tuple[np.ndarray, int, int]]:
    """Group images into static-shape buckets: ``(image_indices, padded_rows, image_capacity)``.

    ``bucket_sizes`` is the padded row count per image (all class segments
    together). Shared by the single-class and class-segmented bucketers so both
    plan identical buckets for identical row counts.
    """
    rotation_counts = np.asarray(rotation_counts)
    bucket_sizes = np.asarray(bucket_sizes)
    image_capacity_ladder = _planner_image_capacity_ladder(image_capacity_ladder)
    n_images = int(rotation_counts.shape[0])
    processing_order = (
        np.arange(n_images, dtype=np.int32)
        if preserve_image_order
        else np.lexsort((rotation_counts, bucket_sizes)).astype(np.int32)
    )
    planned_groups: list[tuple[np.ndarray, int, int]] = []
    if processing_order.size == 0:
        return planned_groups

    if consecutive_mixed_bucket_size is not None:
        for plan in _plan_consecutive_padded_batches(
            bucket_sizes,
            processing_order=processing_order,
            target_items_per_batch=consecutive_mixed_bucket_size,
            max_items_per_batch=image_batch_size,
            max_padded_values_per_batch=max_hypotheses_per_microbatch,
            item_alignment=3,
        ):
            planned_groups.append(
                (
                    np.asarray(plan.item_indices, dtype=np.int32),
                    int(plan.padded_size),
                    int(plan.padded_item_capacity),
                )
            )
        return planned_groups

    if preserve_image_order:
        boundaries = np.flatnonzero(
            np.r_[True, bucket_sizes[processing_order][1:] != bucket_sizes[processing_order][:-1], True]
        )
        bucket_groups = [
            processing_order[start:stop] for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
    else:
        bucket_groups = [
            processing_order[bucket_sizes[processing_order] == bucket_size]
            for bucket_size in np.unique(bucket_sizes[processing_order])
        ]
    for bucket_images in bucket_groups:
        bucket_size = int(bucket_sizes[int(bucket_images[0])])
        max_images = max(1, min(image_batch_size, max_hypotheses_per_microbatch // int(bucket_size)))
        if preserve_image_order and max_images >= 3:
            # RELION's InitialModel default processes pools of three particles.
            # Keep static-shape FFI boundaries on pool boundaries so a new call
            # never changes which physical particles may update BPref together.
            max_images = max(3, (max_images // 3) * 3)
        else:
            # Only when the pool rule is not in force: ladder rungs are not
            # multiples of three, so snapping here would move an InitialModel
            # FFI boundary off a particle pool.
            max_images = _ladder_image_capacity(max_images, image_capacity_ladder)
        # One capacity per rotation class, the remainder group included; see
        # plan_local_hypothesis_buckets for why the remainder keeps it.
        for start in range(0, bucket_images.shape[0], max_images):
            planned_groups.append(
                (
                    np.asarray(bucket_images[start : start + max_images], dtype=np.int32),
                    bucket_size,
                    max_images,
                )
            )
    return planned_groups


def bucket_class_local_hypothesis_layouts(
    class_layouts: Sequence[LocalHypothesisLayout],
    class_log_priors,
    image_batch_size: int,
    rotation_block_size: int,
    *,
    max_hypotheses_per_microbatch: int = 32768,
    unify_bucket_sizes: bool | None = None,
    large_bucket_quantum: int | None = None,
    preserve_image_order: bool = False,
    exact_local_bucket_radix: int | None = None,
    consecutive_mixed_bucket_size: int | None = None,
    image_capacity_ladder=None,
) -> list[LocalBucketSpec]:
    """Bucket K per-class hypothesis layouts into class-major segmented rows.

    Each image's rows are ``[class 0 rows | pad | class 1 rows | pad | ...]`` with
    one segment width per bucket, chosen from the image's largest per-class
    count by the same ladder the single-class bucketer uses, so the joint
    class-by-pose posterior is one log-sum-exp over the row axis and every
    class segment keeps a static shape. ``class_log_priors[k]`` is folded into
    the rows' log prior (RELION adds ``log pdf_class`` to every hypothesis of
    class ``k``). With one layout and a zero prior this reproduces
    ``bucket_local_hypothesis_layout`` exactly.
    """
    class_layouts = tuple(class_layouts)
    n_classes = len(class_layouts)
    if n_classes == 0:
        raise ValueError("class-segmented bucketing needs at least one class layout")
    class_log_priors_np = np.asarray(class_log_priors, dtype=np.float64).reshape(-1)
    if class_log_priors_np.shape[0] != n_classes:
        raise ValueError(f"class_log_priors must have {n_classes} entries, got {class_log_priors_np.shape[0]}")
    first = class_layouts[0]
    n_images = int(first.n_images)
    for k, layout in enumerate(class_layouts[1:], start=1):
        if int(layout.n_images) != n_images:
            raise ValueError(f"class {k} layout has {layout.n_images} images, class 0 has {n_images}")
        if not np.array_equal(np.asarray(layout.translation_grid), np.asarray(first.translation_grid)):
            raise ValueError(f"class {k} layout uses a different fine translation grid than class 0")
        if not np.array_equal(np.asarray(layout.translation_log_priors), np.asarray(first.translation_log_priors)):
            raise ValueError(f"class {k} layout uses different translation log priors than class 0")
        if (layout.rotation_posterior_ids_flat is None) != (first.rotation_posterior_ids_flat is None):
            raise ValueError("class layouts must agree on carrying posterior ids")
        if (layout.sample_mask_bits is None) != (first.sample_mask_bits is None):
            raise ValueError("class layouts must agree on carrying sample masks")
        if (layout.source_eulers_flat is None) != (first.source_eulers_flat is None):
            raise ValueError("class layouts must agree on carrying source eulers")
    image_batch_size = int(max(1, image_batch_size))
    max_hypotheses_per_microbatch = int(max(1, max_hypotheses_per_microbatch))
    rotations_dtype = np.asarray(first.rotations_flat).dtype
    log_prior_dtype = np.asarray(first.rotation_log_priors_flat).dtype
    mstep_flat = [
        np.asarray(layout.rotations_flat, dtype=rotations_dtype)
        if layout.mstep_rotations_flat is None
        else np.asarray(layout.mstep_rotations_flat)
        for layout in class_layouts
    ]
    for k, (layout, mstep) in enumerate(zip(class_layouts, mstep_flat, strict=True)):
        if mstep.shape != np.asarray(layout.rotations_flat).shape:
            raise ValueError(f"class {k}: mstep_rotations_flat must match rotations_flat shape")

    class_counts = np.stack(
        [np.asarray(layout.rotation_counts, dtype=np.int64) for layout in class_layouts], axis=1,
    )  # [n_images, K]
    segment_counts = class_counts.max(axis=1) if n_images else np.zeros(0, dtype=np.int64)
    resolved_large_bucket_quantum = _exact_local_large_bucket_quantum(rotation_block_size, large_bucket_quantum)
    segment_sizes = np.asarray(
        [
            _exact_bucket_rotation_size(
                int(count),
                rotation_block_size,
                large_bucket_quantum=resolved_large_bucket_quantum,
                exact_local_bucket_radix=exact_local_bucket_radix,
            )
            for count in segment_counts
        ],
        dtype=np.int32,
    )
    if unify_bucket_sizes is None:
        unify_bucket_sizes = os.environ.get("RECOVAR_LOCAL_BUCKET_UNIFY", "").lower() in {"1", "true", "yes", "on"}
    if consecutive_mixed_bucket_size is not None:
        consecutive_mixed_bucket_size = int(consecutive_mixed_bucket_size)
        if consecutive_mixed_bucket_size <= 0:
            raise ValueError("consecutive_mixed_bucket_size must be positive")
        if not preserve_image_order:
            raise ValueError("consecutive mixed buckets require preserved image order")
        if bool(unify_bucket_sizes):
            raise ValueError("consecutive mixed buckets cannot use run-global bucket unification")
    if segment_sizes.size and bool(unify_bucket_sizes):
        segment_sizes = np.full_like(segment_sizes, int(segment_sizes.max()))
    total_sizes = (segment_sizes.astype(np.int64) * n_classes).astype(np.int32)

    bucket_specs: list[LocalBucketSpec] = []
    planned_groups = _plan_local_bucket_groups(
        segment_counts,
        total_sizes,
        image_batch_size=image_batch_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
        preserve_image_order=preserve_image_order,
        consecutive_mixed_bucket_size=consecutive_mixed_bucket_size,
        image_capacity_ladder=image_capacity_ladder,
    )
    n_trans = int(np.asarray(first.translation_grid).shape[0])
    for image_indices, bucket_size, max_images in planned_groups:
        bucket_size = int(bucket_size)
        if bucket_size % n_classes:
            raise RuntimeError(f"planned bucket of {bucket_size} rows is not divisible by {n_classes} classes")
        seg = bucket_size // n_classes
        batch_size = int(image_indices.shape[0])
        per_class_counts = class_counts[image_indices].astype(np.int32, copy=False)  # [B, K]
        if np.any(per_class_counts > seg):
            raise RuntimeError("a class segment is narrower than an image's rows for that class")
        padded_rotations = np.broadcast_to(np.eye(3, dtype=rotations_dtype), (batch_size, bucket_size, 3, 3)).copy()
        # Mirror the single-class convention: when no class supplies its own M-step
        # rotations, leave the field unmaterialized rather than duplicating the
        # scoring geometry.
        padded_mstep_rotations = None if all(
            layout.mstep_rotations_flat is None for layout in class_layouts
        ) else np.broadcast_to(np.eye(3, dtype=mstep_flat[0].dtype), (batch_size, bucket_size, 3, 3)).copy()
        padded_rotation_ids = np.full((batch_size, bucket_size), -1, dtype=np.int64)
        padded_log_prior = np.full((batch_size, bucket_size), -1e30, dtype=log_prior_dtype)
        padded_mask = np.zeros((batch_size, bucket_size), dtype=bool)
        padded_posterior_ids = None if first.rotation_posterior_ids_flat is None else np.full((batch_size, bucket_size), -1, dtype=np.int32)
        padded_sample_mask = None if first.sample_mask_bits is None else np.zeros((batch_size, bucket_size, n_trans), dtype=bool)
        padded_source_eulers = None if first.source_eulers_flat is None else np.zeros((batch_size, bucket_size, 3), dtype=np.float64)
        for row, image_idx in enumerate(image_indices.tolist()):
            for k, layout in enumerate(class_layouts):
                start_off = int(layout.rotation_offsets[image_idx])
                end_off = int(layout.rotation_offsets[image_idx + 1])
                count = end_off - start_off
                lo = k * seg
                hi = lo + count
                padded_rotations[row, lo:hi] = layout.rotations_flat[start_off:end_off]
                if padded_mstep_rotations is not None:
                    padded_mstep_rotations[row, lo:hi] = mstep_flat[k][start_off:end_off]
                padded_rotation_ids[row, lo:hi] = layout.rotation_ids_flat[start_off:end_off]
                padded_log_prior[row, lo:hi] = (
                    np.asarray(layout.rotation_log_priors_flat[start_off:end_off], dtype=np.float64)
                    + class_log_priors_np[k]
                ).astype(log_prior_dtype, copy=False)
                padded_mask[row, lo:hi] = True
                if padded_posterior_ids is not None:
                    padded_posterior_ids[row, lo:hi] = layout.rotation_posterior_ids_flat[start_off:end_off]
                if padded_sample_mask is not None:
                    padded_sample_mask[row, lo:hi, :] = layout.sample_mask_rows(start_off, end_off)
                if padded_source_eulers is not None:
                    padded_source_eulers[row, lo:hi] = layout.source_eulers_flat[start_off:end_off]
        bucket_specs.append(
            LocalBucketSpec(
                image_indices=image_indices,
                bucket_image_count=int(max_images),
                bucket_rotation_count=bucket_size,
                actual_rotation_counts=per_class_counts.sum(axis=1).astype(np.int32),
                local_rotation_ids=padded_rotation_ids,
                local_rotations=padded_rotations,
                local_source_eulers=padded_source_eulers,
                local_rotation_log_prior=padded_log_prior,
                local_rotation_mask=padded_mask,
                translation_log_prior=np.asarray(first.translation_log_priors[image_indices]),
                local_mstep_rotations=padded_mstep_rotations,
                local_rotation_posterior_ids=padded_posterior_ids,
                local_sample_mask=padded_sample_mask,
                n_classes=n_classes,
                class_segment_rotation_count=seg,
                class_actual_rotation_counts=per_class_counts,
            )
        )
    return bucket_specs


def _local_search_engine_rotation_block_size(rotation_block_size: int) -> int:
    """Cap the exact local-search engine block size.

    Local search already reduces the candidate set per image from the full
    HEALPix grid down to a few thousand rotations. Reusing the dense-search
    5k rotation tile size here creates oversized XLA kernels whose compile
    time dominates the first local-search iteration. A 1k cap keeps the
    candidate set exact while making the compiled score kernels much smaller.
    """
    return int(max(64, min(int(rotation_block_size), 1024)))
