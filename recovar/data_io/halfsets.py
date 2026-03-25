"""Half-set splitting logic for cryo-EM reconstruction.

Provides functions for splitting a dataset into two independent half-sets
used for FSC-based resolution estimation.  Supports random splits,
RELION _rlnRandomSubset, explicit halfset files, and tilt-series-aware
particle-level splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

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


def get_split_tilt_indices(
    particles_file,
    ind_file=None,
    tilt_ind_file=None,
    ntilts=None,
    datadir=None,
    particle_halfset_indices_file=None,
):
    """Split a tilt-series dataset into two halfsets (image indices).

    Supports optional filtering by image/particle indices and precomputed splits.
    """
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
    n_images = getattr(args, "n_images", None) or -1

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
