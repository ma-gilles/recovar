"""Half-set splitting logic for cryo-EM reconstruction.

Provides functions for splitting a dataset into two independent half-sets
used for FSC-based resolution estimation.  Supports random splits,
RELION _rlnRandomSubset, explicit halfset files, and tilt-series-aware
particle-level splitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from recovar.data_io._index_utils import (
    TiltSeriesOriginalIndexMap,
    deduplicate_preserve_order,
    filter_preserve_order,
    load_index_like,
    normalize_image_indices,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HalfsetDatasetSpec:
    """Normalized file/loader settings for constructing a halfset dataset."""

    particles_file: str
    poses_file: str | None = None
    ctf_file: str | None = None
    datadir: str | None = None
    uninvert_data: bool = False
    padding: int = 0
    n_images: int | None = None
    tilt_series: bool = False
    tilt_series_ctf: str | None = None
    angle_per_tilt: float | None = None
    dose_per_tilt: float | None = None
    premultiplied_ctf: bool = False
    strip_prefix: str | None = None
    downsample_D: int | None = None

    @classmethod
    def from_args(cls, args):
        uninvert_data_str = getattr(args, "uninvert_data", "false")
        if uninvert_data_str in ("automatic", "false"):
            uninvert_data = False
        elif uninvert_data_str == "true":
            uninvert_data = True
        else:
            raise ValueError(f"uninvert_data must be 'automatic', 'true', or 'false'; got {uninvert_data_str!r}")

        return cls(
            particles_file=args.particles,
            ctf_file=getattr(args, "ctf", None),
            poses_file=getattr(args, "poses", None),
            datadir=getattr(args, "datadir", None),
            n_images=getattr(args, "n_images", -1),
            padding=getattr(args, "padding", 0),
            tilt_series=getattr(args, "tilt_series", False),
            tilt_series_ctf=getattr(args, "tilt_series_ctf", "cryoem"),
            angle_per_tilt=getattr(args, "angle_per_tilt", None),
            dose_per_tilt=getattr(args, "dose_per_tilt", None),
            premultiplied_ctf=getattr(args, "premultiplied_ctf", False),
            strip_prefix=getattr(args, "strip_prefix", None),
            downsample_D=getattr(args, "downsample", None),
            uninvert_data=uninvert_data,
        )


# ---------------------------------------------------------------------------
# Core splitting
# ---------------------------------------------------------------------------


def split_index_list(all_valid_image_indices, split_random_seed=0):
    """Split a list of indices into two balanced halves with reproducible randomization.

    Args:
        all_valid_image_indices: Array of indices to split
        split_random_seed: Random seed for reproducible splits

    Returns:
        List of two numpy arrays containing the split indices
    """
    all_valid_image_indices = np.asarray(all_valid_image_indices)
    if len(all_valid_image_indices) == 0:
        raise ValueError("Cannot split empty index list")

    n_indices = len(all_valid_image_indices)
    half_ind_size = n_indices // 2

    # Keep the legacy global-RNG shuffle to preserve main-branch halfset
    # assignments exactly. The Generator API produces different partitions.
    np.random.seed(split_random_seed)
    shuffled_ind = np.arange(n_indices)
    np.random.shuffle(shuffled_ind)

    ind_split = [
        np.sort(all_valid_image_indices[shuffled_ind[:half_ind_size]]),
        np.sort(all_valid_image_indices[shuffled_ind[half_ind_size:]]),
    ]
    return ind_split


# ---------------------------------------------------------------------------
# SPA halfset splitting
# ---------------------------------------------------------------------------


def get_split_indices(
    particles_file,
    datadir=None,
    strip_prefix=None,
    ind_file=None,
    split_random_seed=0,
    validate_split=True,
    n_images=None,
):
    """Get indices for splitting dataset into halfsets.

    Args:
        particles_file: Path to particles STAR file
        datadir: Data directory (optional)
        strip_prefix: Prefix to strip from file paths (optional)
        ind_file: File containing specific indices to use (optional)
        split_random_seed: Random seed for reproducible splits
        validate_split: Whether to validate the split is balanced
        n_images: Pre-computed image count (avoids re-reading the file)

    Returns:
        List of two numpy arrays containing indices for each halfset
    """
    from recovar.data_io.cryoem_dataset import get_num_images_in_dataset

    if ind_file is None:
        if n_images is None:
            n_images = get_num_images_in_dataset(particles_file, datadir=datadir, strip_prefix=strip_prefix)
        indices = np.arange(n_images, dtype=np.int32)
    else:
        raw_indices = load_index_like(ind_file)
        n_images_total = None
        if np.asarray(raw_indices).dtype == bool:
            n_images_total = get_num_images_in_dataset(particles_file, datadir=datadir, strip_prefix=strip_prefix)
        indices = normalize_image_indices(raw_indices, n_total=n_images_total, name="ind_file")
        indices = deduplicate_preserve_order(indices, name="ind_file").astype(np.int32, copy=False)

    if len(indices) == 0:
        raise ValueError("No valid indices found for dataset splitting")

    split_indices = split_index_list(indices, split_random_seed=split_random_seed)

    if validate_split:
        n1, n2 = len(split_indices[0]), len(split_indices[1])
        total = n1 + n2
        if abs(n1 - n2) > max(1, total * 0.01):
            logger.warning(
                "Split is imbalanced: %s vs %s images (%.1f%% difference)", n1, n2, abs(n1 - n2) / total * 100
            )

        overlap = np.intersect1d(split_indices[0], split_indices[1])
        if len(overlap) > 0:
            raise ValueError(f"Split contains {len(overlap)} overlapping indices")

    logger.info("Split dataset into halfsets: %s and %s images", len(split_indices[0]), len(split_indices[1]))
    return split_indices


# ---------------------------------------------------------------------------
# Tilt-series halfset splitting
# ---------------------------------------------------------------------------


def _select_complete_central_tilts(
    index_map,
    particles_file,
    n_central,
    candidate_particles,
    allowed_images,
):
    """Select identical nominal positions per tilt series and drop incomplete particles."""
    if not isinstance(n_central, (int, np.integer)) or isinstance(n_central, (bool, np.bool_)):
        raise TypeError("central_tilts must be a positive integer")
    n_central = int(n_central)
    if n_central <= 0:
        raise ValueError("central_tilts must be a positive integer")

    from recovar.data_io import starfile

    particles = starfile.StarFile.load(particles_file).df
    if "_rlnTiltName" in particles.columns:
        tilt_identity_column = "_rlnTiltName"
    elif "_rlnMicrographName" in particles.columns:
        tilt_identity_column = "_rlnMicrographName"
    else:
        raise ValueError("--central-tilts requires _rlnTiltName or _rlnMicrographName in the particles STAR file")

    tilt_names = np.asarray(particles[tilt_identity_column], dtype=str)
    if tilt_names.shape != (index_map.n_images,):
        raise ValueError(
            "Particles STAR row count does not match the tilt-series image index map: "
            f"{tilt_names.size} rows versus {index_map.n_images} images"
        )

    # Build tilt-series components. A RELION N@stack name whose stack is
    # shared by all images of a particle supplies an explicit tomogram key.
    # Per-tilt files legitimately have different stack suffixes, so those
    # fall back to joining particles through shared physical-tilt names.
    parents = np.arange(index_map.n_particles, dtype=np.int32)

    def find(particle_idx):
        particle_idx = int(particle_idx)
        while parents[particle_idx] != particle_idx:
            parents[particle_idx] = parents[parents[particle_idx]]
            particle_idx = int(parents[particle_idx])
        return particle_idx

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    tomo_names = None
    if "_rlnTomoName" in particles.columns:
        tomo_names = np.asarray(particles["_rlnTomoName"], dtype=str)
        if tomo_names.shape != (index_map.n_images,):
            raise ValueError("_rlnTomoName row count does not match the particles STAR file")

    component_owner = {}
    for particle_idx, particle_images in enumerate(index_map.particle_to_images):
        names = tilt_names[particle_images]
        if tomo_names is not None:
            particle_tomos = np.unique(tomo_names[particle_images])
            if particle_tomos.size != 1:
                raise ValueError("A particle contains images assigned to multiple _rlnTomoName values")
            merge_keys = [("tomo", str(particle_tomos[0]))]
        else:
            parsed_stacks = []
            for name in names:
                image_number, separator, stack_name = name.partition("@")
                if separator and image_number.isdigit() and stack_name:
                    parsed_stacks.append(stack_name)
            if len(parsed_stacks) == len(names) and len(set(parsed_stacks)) == 1:
                merge_keys = [("stack", parsed_stacks[0])]
            else:
                merge_keys = [("tilt", str(name)) for name in np.unique(names)]
        for merge_key in merge_keys:
            if merge_key in component_owner:
                union(particle_idx, component_owner[merge_key])
            else:
                component_owner[merge_key] = particle_idx

    component_particles = {}
    for particle_idx in range(index_map.n_particles):
        component_particles.setdefault(find(particle_idx), []).append(particle_idx)

    # Include the tilt-series component in the physical identity. This keeps
    # generic names such as tilt_001 distinct across tomograms.
    image_components = np.full(index_map.n_images, -1, dtype=np.int32)
    for particle_idx, particle_images in enumerate(index_map.particle_to_images):
        image_components[particle_images] = find(particle_idx)
    if np.any(image_components < 0):
        raise ValueError("Could not assign every particles STAR row to a tilt series")

    physical_tilt_ids = {}
    image_to_tilt = np.empty(index_map.n_images, dtype=np.int32)
    for image_idx, (component, tilt_name) in enumerate(zip(image_components, tilt_names, strict=True)):
        key = (int(component), str(tilt_name))
        tilt_idx = physical_tilt_ids.setdefault(key, len(physical_tilt_ids))
        image_to_tilt[image_idx] = tilt_idx

    n_physical_tilts = len(physical_tilt_ids)
    tilt_counts = np.bincount(image_to_tilt, minlength=n_physical_tilts)

    def aggregate_metadata(column, *, require_consistent=False):
        if column not in particles.columns:
            return None
        values = np.asarray(particles[column], dtype=np.float64)
        if values.shape != (index_map.n_images,):
            raise ValueError(f"{column} row count does not match the particles STAR file")
        finite = np.isfinite(values)
        finite_counts = np.bincount(image_to_tilt[finite], minlength=n_physical_tilts)
        sums = np.bincount(
            image_to_tilt[finite],
            weights=values[finite],
            minlength=n_physical_tilts,
        )
        result = np.full(n_physical_tilts, np.nan, dtype=np.float64)
        valid = finite_counts == tilt_counts
        result[valid] = sums[valid] / finite_counts[valid]
        if require_consistent:
            minima = np.full(n_physical_tilts, np.inf, dtype=np.float64)
            maxima = np.full(n_physical_tilts, -np.inf, dtype=np.float64)
            np.minimum.at(minima, image_to_tilt[finite], values[finite])
            np.maximum.at(maxima, image_to_tilt[finite], values[finite])
            inconsistent = valid & ~np.isclose(minima, maxima, rtol=1e-6, atol=1e-6)
            if np.any(inconsistent):
                raise ValueError(f"A physical tilt identity has inconsistent {column} values")
        return result

    dose_column = "_rlnMicrographPreExposure"
    tilt_doses = aggregate_metadata(dose_column, require_consistent=True)
    angle_column = "_rlnTomoNominalStageTiltAngle"
    tilt_angles = aggregate_metadata(angle_column)

    def expected_angle_offsets(count):
        offsets = [0]
        radius = 1
        while len(offsets) < count:
            offsets.append(-radius)
            if len(offsets) < count:
                offsets.append(radius)
            radius += 1
        return np.asarray(offsets, dtype=np.float64)

    def match_regular_positions(values, expected_offsets, *, use_closest_to_zero):
        """Match a complete central lattice without replacing a missing position."""
        values = np.asarray(values, dtype=np.float64)
        if values.size < expected_offsets.size or not np.all(np.isfinite(values)):
            return None

        spacings = np.diff(np.sort(values))
        spacings = spacings[spacings > 1e-6]
        if spacings.size == 0:
            if expected_offsets.size == 1:
                candidate = int(np.argmin(np.abs(values)))
                return np.array([candidate], dtype=np.int32)
            return None

        step = float(np.median(spacings))
        tolerance = max(0.35 * step, 1e-3)
        origin_idx = int(np.argmin(np.abs(values))) if use_closest_to_zero else int(np.argmin(values))
        origin = float(values[origin_idx])
        # Nominal angles and RELION pre-exposure both have a zero-origin
        # central position. If it is absent, do not promote the next view.
        if abs(origin) > tolerance:
            return None

        targets = origin + expected_offsets * step
        matched = []
        for target in targets:
            candidate = int(np.argmin(np.abs(values - target)))
            if abs(float(values[candidate]) - float(target)) > tolerance or candidate in matched:
                return None
            matched.append(candidate)
        return np.asarray(matched, dtype=np.int32)

    target_tilts_by_component = {}
    angle_selected_components = 0
    dose_selected_components = 0
    incomplete_components = 0
    for root, particle_indices in component_particles.items():
        component_images = np.concatenate(
            [index_map.particle_to_images[int(particle_idx)] for particle_idx in particle_indices]
        )
        component_tilts = np.unique(image_to_tilt[component_images])
        if component_tilts.size < n_central:
            incomplete_components += 1
            continue

        component_angles = None if tilt_angles is None else tilt_angles[component_tilts]
        angles_are_informative = (
            component_angles is not None and np.all(np.isfinite(component_angles)) and np.ptp(component_angles) > 1e-3
        )
        if angles_are_informative:
            matched = match_regular_positions(
                component_angles,
                expected_angle_offsets(n_central),
                use_closest_to_zero=True,
            )
            method = "angle"
        else:
            component_doses = None if tilt_doses is None else tilt_doses[component_tilts]
            matched = match_regular_positions(
                component_doses if component_doses is not None else np.array([], dtype=np.float64),
                np.arange(n_central, dtype=np.float64),
                use_closest_to_zero=False,
            )
            method = "dose"

        if matched is None:
            incomplete_components += 1
            continue
        if method == "angle":
            angle_selected_components += 1
        else:
            dose_selected_components += 1
        target_tilts_by_component[root] = component_tilts[matched]

    allowed_mask = np.zeros(index_map.n_images, dtype=bool)
    allowed_mask[np.asarray(allowed_images, dtype=np.int64)] = True

    complete_particles = []
    complete_images = []
    for particle_idx in np.asarray(candidate_particles, dtype=np.int32):
        target_tilts = target_tilts_by_component.get(find(particle_idx))
        if target_tilts is None:
            continue
        particle_images = index_map.particle_to_images[int(particle_idx)]
        selected = particle_images[
            np.isin(image_to_tilt[particle_images], target_tilts) & allowed_mask[particle_images]
        ]
        if selected.size != n_central:
            continue
        if np.unique(image_to_tilt[selected]).size != n_central:
            continue
        complete_particles.append(int(particle_idx))
        complete_images.extend(selected.tolist())

    if not complete_particles:
        raise ValueError(f"No particles contain all central_tilts={n_central} nominal positions after applying filters")

    complete_particles = np.asarray(complete_particles, dtype=np.int32)
    complete_images = np.asarray(complete_images, dtype=np.int32)
    logger.info(
        "Central-tilt selection: %d positions per tilt series; %d series selected by angle, %d by dose, "
        "and %d lacked a complete central lattice; retained %d/%d particles and %d images",
        n_central,
        angle_selected_components,
        dose_selected_components,
        incomplete_components,
        complete_particles.size,
        np.asarray(candidate_particles).size,
        complete_images.size,
    )
    return complete_particles, complete_images


def get_split_tilt_indices(
    particles_file,
    ind_file=None,
    tilt_ind_file=None,
    ntilts=None,
    datadir=None,
    particle_halfset_indices_file=None,
    central_tilts=None,
):
    """Split a tilt-series dataset into two halfsets (image indices).

    Supports optional filtering by image/particle indices and precomputed splits.
    ``central_tilts`` keeps only particles containing the same nominally central
    physical tilt identities within each tilt series and never substitutes a later view.
    """
    if central_tilts is not None and ntilts is not None:
        raise ValueError("central_tilts and ntilts are mutually exclusive")

    index_map = TiltSeriesOriginalIndexMap.from_particles_file(
        particles_file,
        datadir=datadir,
        ntilts=ntilts,
    )

    def _sanitize_particle_ids(values, *, name, allowed_particles=None):
        raw = np.asarray(values)
        if raw.dtype != bool:
            raw = np.asarray(raw).reshape(-1)
            dropped = int(np.sum((raw < 0) | (raw >= index_map.n_particles)))
            if dropped > 0:
                logger.warning("Dropping %d out-of-range particle ids from %s.", dropped, name)
        sanitized = index_map.sanitize_particle_indices(
            values,
            name=name,
            allowed_particles=allowed_particles,
        )
        duplicates = int(np.asarray(values).reshape(-1).size - sanitized.size) if np.asarray(values).ndim <= 1 else 0
        if duplicates > 0 and allowed_particles is not None:
            logger.warning("Dropping duplicate particle ids from %s.", name)
        return sanitized

    def _sanitize_image_ids(values, *, name):
        raw = np.asarray(values)
        if raw.dtype != bool:
            raw = np.asarray(raw).reshape(-1)
            dropped = int(np.sum((raw < 0) | (raw >= index_map.n_images)))
            if dropped > 0:
                logger.warning("Dropping %d out-of-range image ids from %s.", dropped, name)
        return index_map.sanitize_image_indices(values, name=name)

    if tilt_ind_file is not None:
        particle_ind = _sanitize_particle_ids(
            load_index_like(tilt_ind_file),
            name="tilt_ind_file",
        )
    else:
        particle_ind = np.arange(index_map.n_particles, dtype=np.int32)

    if particle_ind.size == 0:
        empty = np.array([], dtype=np.int32)
        return [empty, empty]

    allowed_image_indices = index_map.image_indices_from_particles(particle_ind)
    if ind_file is not None:
        image_ind = _sanitize_image_ids(load_index_like(ind_file), name="ind_file")
        allowed_image_indices = filter_preserve_order(allowed_image_indices, image_ind)

    if allowed_image_indices.size == 0:
        empty = np.array([], dtype=np.int32)
        return [empty, empty]

    if central_tilts is not None:
        particle_ind, allowed_image_indices = _select_complete_central_tilts(
            index_map,
            particles_file,
            central_tilts,
            particle_ind,
            allowed_image_indices,
        )

    valid_particles = index_map.particle_indices_from_images(allowed_image_indices)
    if valid_particles.size == 0:
        empty = np.array([], dtype=np.int32)
        return [empty, empty]

    if particle_halfset_indices_file is not None:
        split_particles_raw = load_index_like(particle_halfset_indices_file)
        if len(split_particles_raw) != 2:
            raise ValueError("particle_halfset_indices_file must contain exactly two halfsets")
        split_particles = [
            _sanitize_particle_ids(
                split_particles_raw[0],
                name="particle_halfset_indices_file[0]",
                allowed_particles=valid_particles,
            ),
            _sanitize_particle_ids(
                split_particles_raw[1],
                name="particle_halfset_indices_file[1]",
                allowed_particles=valid_particles,
            ),
        ]
    else:
        split_particles = split_index_list(valid_particles)

    split_image_indices = []
    for halfset_particle_indices in split_particles:
        split_image_indices.append(
            index_map.image_indices_from_particles(
                halfset_particle_indices,
                allowed_images=allowed_image_indices,
                ntilts=ntilts,
            )
        )

    return split_image_indices


# ---------------------------------------------------------------------------
# RELION halfset detection
# ---------------------------------------------------------------------------


def _read_relion_halfsets_from_star(particles_file, ind_file=None, datadir=None, strip_prefix=None):
    """Read halfset assignments from `_rlnRandomSubset` when present.

    Returns ``(halfsets, n_total)`` where *halfsets* is a list of two index
    arrays when the column is present and valid, or ``None`` when the column
    is absent. Non-STAR inputs are ignored by design; malformed STAR inputs
    fail loudly.
    """
    if not str(particles_file).endswith(".star"):
        return None, None

    from recovar.data_io.starfile import read_star

    df, _ = read_star(particles_file)

    n_total = len(df)

    if "_rlnRandomSubset" not in df.columns:
        return None, n_total

    subsets = df["_rlnRandomSubset"].values.astype(int)
    unique_vals = np.unique(subsets)
    if not (set(unique_vals) <= {1, 2}):
        logger.warning(
            "_rlnRandomSubset contains values other than 1/2 (%s); ignoring",
            unique_vals,
        )
        return None, n_total

    all_indices = np.arange(len(subsets), dtype=np.int32)
    halfsets = [
        all_indices[subsets == 1],
        all_indices[subsets == 2],
    ]

    if ind_file is not None:
        raw_indices = load_index_like(ind_file)
        n_images_total = len(subsets)
        ind = normalize_image_indices(raw_indices, n_total=n_images_total, name="ind_file")
        halfsets = [h[np.isin(h, ind)] for h in halfsets]

    if len(halfsets[0]) == 0 or len(halfsets[1]) == 0:
        logger.warning("RELION halfsets are empty after filtering; falling back to random split")
        return None, n_total

    logger.info(
        "Using RELION halfsets from _rlnRandomSubset: %d and %d images",
        len(halfsets[0]),
        len(halfsets[1]),
    )
    return halfsets, n_total


# ---------------------------------------------------------------------------
# High-level dataset splitting
# ---------------------------------------------------------------------------


def load_halfset_dataset(spec: HalfsetDatasetSpec, *, ind_split, lazy=False):
    """Load one dataset view and attach halfset-local indices for iteration."""
    from recovar.data_io.cryoem_dataset import load_dataset

    all_indices = np.unique(np.concatenate(ind_split))

    full = load_dataset(
        spec.particles_file,
        spec.poses_file,
        spec.ctf_file,
        datadir=spec.datadir,
        n_images=spec.n_images,
        ind=all_indices,
        lazy=lazy,
        padding=spec.padding,
        uninvert_data=spec.uninvert_data,
        tilt_series=spec.tilt_series,
        tilt_series_ctf=spec.tilt_series_ctf,
        angle_per_tilt=spec.angle_per_tilt,
        dose_per_tilt=spec.dose_per_tilt,
        premultiplied_ctf=spec.premultiplied_ctf,
        strip_prefix=spec.strip_prefix,
        downsample_D=spec.downsample_D,
    )

    orig_to_local = np.empty(int(all_indices.max()) + 1, dtype=np.int32)
    orig_to_local[all_indices] = np.arange(len(all_indices), dtype=np.int32)

    local_split = [orig_to_local[s] for s in ind_split]

    full.halfset_indices = [np.asarray(split, dtype=np.int32) for split in local_split]
    return full


def resolve_halfset_indices(args):
    """Determine which images belong to each reconstruction half-set.

    Priority order:
      1. Explicit halfsets file (``--halfsets``).
      2. _rlnRandomSubset column in the STAR file (RELION convention).
      3. Random 50/50 split of all valid images.
    """
    from recovar.data_io.cryoem_dataset import get_num_images_in_dataset

    is_tilt = getattr(args, "tilt_series", False) or getattr(args, "tilt_series_ctf", "cryoem") != "cryoem"
    datadir = getattr(args, "datadir", None)
    strip_prefix = getattr(args, "strip_prefix", None)
    ind_file = getattr(args, "ind", None)
    tilt_ind_file = getattr(args, "tilt_ind", None)
    ntilts = getattr(args, "ntilts", None)
    central_tilts = getattr(args, "central_tilts", None)
    n_images = getattr(args, "n_images", None) or -1

    if central_tilts is not None:
        if not getattr(args, "tilt_series", False):
            raise ValueError("--central-tilts requires tilt-series data")
        if ntilts is not None:
            raise ValueError("--central-tilts and --ntilts are mutually exclusive")
        if n_images > 0:
            raise ValueError("--n-images cannot be combined with --central-tilts")

    if args.halfsets is None:
        n_total_from_star = None
        if not is_tilt:
            halfsets, n_total_from_star = _read_relion_halfsets_from_star(
                args.particles,
                ind_file=ind_file,
                datadir=datadir,
                strip_prefix=strip_prefix,
            )
            if halfsets is not None:
                if n_images > 0:
                    halfsets = [halfset[: n_images // 2] for halfset in halfsets]
                    logger.info("using only %s particles", n_images)
                return halfsets

        logger.info("Randomly splitting dataset into halfsets")
        if is_tilt:
            halfsets = get_split_tilt_indices(
                args.particles,
                ind_file=ind_file,
                tilt_ind_file=tilt_ind_file,
                ntilts=ntilts,
                central_tilts=central_tilts,
                datadir=datadir,
            )
        else:
            halfsets = get_split_indices(
                args.particles,
                datadir=datadir,
                strip_prefix=strip_prefix,
                ind_file=ind_file,
                n_images=n_total_from_star,
            )

    else:
        logger.info("Loading halfsets from file")
        if is_tilt:
            halfsets = get_split_tilt_indices(
                args.particles,
                ind_file=ind_file,
                tilt_ind_file=tilt_ind_file,
                ntilts=ntilts,
                central_tilts=central_tilts,
                datadir=datadir,
                particle_halfset_indices_file=args.halfsets,
            )
        else:
            halfsets = load_index_like(args.halfsets)
            logger.info("Loaded halfsets from file")
            if len(halfsets) != 2:
                raise ValueError("halfsets file must contain exactly two halfsets")

            needs_n_images = any(np.asarray(h).dtype == bool for h in halfsets)
            n_images_total = None
            if needs_n_images:
                n_images_total = get_num_images_in_dataset(
                    args.particles,
                    datadir=datadir,
                    strip_prefix=strip_prefix,
                )
            halfsets = [
                normalize_image_indices(halfsets[0], n_total=n_images_total, name="halfsets[0]"),
                normalize_image_indices(halfsets[1], n_total=n_images_total, name="halfsets[1]"),
            ]

            if ind_file is not None:
                ind_raw = load_index_like(ind_file)
                if n_images_total is None and np.asarray(ind_raw).dtype == bool:
                    n_images_total = get_num_images_in_dataset(
                        args.particles,
                        datadir=datadir,
                        strip_prefix=strip_prefix,
                    )
                ind = normalize_image_indices(ind_raw, n_total=n_images_total, name="ind")
                halfsets = [np.asarray(halfset)[np.isin(np.asarray(halfset), ind)] for halfset in halfsets]

    if n_images > 0:
        halfsets = [halfset[: n_images // 2] for halfset in halfsets]
        logger.info("using only %s particles", n_images)
    return halfsets


def load_halfset_dataset_from_args(args, lazy=False, ind_split=None):
    """Resolve halfsets from args and load the shared dataset view."""
    if ind_split is None:
        ind_split = resolve_halfset_indices(args)
    dataset_spec = HalfsetDatasetSpec.from_args(args)
    return load_halfset_dataset(dataset_spec, ind_split=ind_split, lazy=lazy)
