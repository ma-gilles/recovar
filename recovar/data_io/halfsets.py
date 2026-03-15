"""Half-set splitting logic for cryo-EM reconstruction.

Provides functions for splitting a dataset into two independent half-sets
used for FSC-based resolution estimation.  Supports random splits,
RELION _rlnRandomSubset, explicit halfset files, and tilt-series-aware
particle-level splitting.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np

from recovar.data_io import cryo_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (imported lazily from dataset to avoid circular imports)
# ---------------------------------------------------------------------------

def _load_index_like(value):
    if value is None:
        return None
    if isinstance(value, (np.ndarray, list, tuple)):
        return value
    with open(value, "rb") as f:
        return pickle.load(f)


def _get_normalize_and_dedup():
    """Lazy import to avoid circular dependency with dataset.py."""
    from recovar.data_io.dataset import _normalize_image_indices, _deduplicate_preserve_order
    return _normalize_image_indices, _deduplicate_preserve_order


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

    shuffled_ind = np.arange(n_indices)
    rng = np.random.default_rng(split_random_seed)
    rng.shuffle(shuffled_ind)

    ind_split = [
        np.sort(all_valid_image_indices[shuffled_ind[:half_ind_size]]),
        np.sort(all_valid_image_indices[shuffled_ind[half_ind_size:]]),
    ]
    return ind_split


# ---------------------------------------------------------------------------
# SPA halfset splitting
# ---------------------------------------------------------------------------

def get_split_indices(particles_file, datadir=None, strip_prefix=None,
                      ind_file=None, split_random_seed=0, validate_split=True,
                      n_images=None):
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
    from recovar.data_io.dataset import get_num_images_in_dataset
    _normalize_image_indices, _deduplicate_preserve_order = _get_normalize_and_dedup()

    if ind_file is None:
        if n_images is None:
            n_images = get_num_images_in_dataset(particles_file, datadir=datadir, strip_prefix=strip_prefix)
        indices = np.arange(n_images, dtype=np.int32)
    else:
        raw_indices = _load_index_like(ind_file)
        n_images_total = None
        if np.asarray(raw_indices).dtype == bool:
            n_images_total = get_num_images_in_dataset(particles_file, datadir=datadir, strip_prefix=strip_prefix)
        indices = _normalize_image_indices(raw_indices, n_images_total=n_images_total, name="ind_file")
        indices = _deduplicate_preserve_order(indices, name="ind_file").astype(np.int32, copy=False)

    if len(indices) == 0:
        raise ValueError("No valid indices found for dataset splitting")

    split_indices = split_index_list(indices, split_random_seed=split_random_seed)

    if validate_split:
        n1, n2 = len(split_indices[0]), len(split_indices[1])
        total = n1 + n2
        if abs(n1 - n2) > max(1, total * 0.01):
            logger.warning("Split is imbalanced: %s vs %s images (%.1f%% difference)",
                           n1, n2, abs(n1 - n2) / total * 100)

        overlap = np.intersect1d(split_indices[0], split_indices[1])
        if len(overlap) > 0:
            raise ValueError(f"Split contains {len(overlap)} overlapping indices")

    logger.info("Split dataset into halfsets: %s and %s images",
                len(split_indices[0]), len(split_indices[1]))
    return split_indices


# ---------------------------------------------------------------------------
# Tilt-series halfset splitting
# ---------------------------------------------------------------------------

def get_split_tilt_indices(
    particles_file, ind_file=None, tilt_ind_file=None, ntilts=None,
    datadir=None, particle_halfset_indices_file=None,
):
    """Split a tilt-series dataset into two halfsets (image indices).

    Supports optional filtering by image/particle indices and precomputed splits.
    """
    _normalize_image_indices, _ = _get_normalize_and_dedup()

    def _filter_preserve_order(values, allowed):
        values = np.asarray(values)
        allowed = np.asarray(allowed)
        if values.size == 0:
            return values.astype(np.int32, copy=False)
        return values[np.isin(values, allowed)]

    def _normalize_particle_ids(values, n_particles_total):
        arr = np.asarray(values)
        if arr.dtype == bool:
            if arr.ndim != 1:
                raise ValueError("tilt_ind_file/particle halfset boolean mask must be 1D")
            if arr.size != int(n_particles_total):
                raise ValueError(
                    f"tilt_ind_file/particle halfset boolean mask length {arr.size} "
                    f"must match number of particles {int(n_particles_total)}"
                )
            return np.flatnonzero(arr).astype(np.int64, copy=False)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("tilt_ind_file/particle halfset ids must be 1D")
        if arr.dtype.kind not in ("i", "u"):
            raise TypeError("tilt_ind_file/particle halfset ids must be integer or boolean mask")
        return arr.astype(np.int64, copy=False).reshape(-1)

    def _normalize_image_ids(values, n_images_total):
        return _normalize_image_indices(values, n_images_total=n_images_total, name="ind_file")

    def _sanitize_particle_ids(values, n_particles_total, allowed_particles):
        values = _normalize_particle_ids(values, n_particles_total=n_particles_total)
        if values.size == 0:
            return values.astype(np.int32, copy=False)
        in_bounds = (values >= 0) & (values < int(n_particles_total))
        if not np.all(in_bounds):
            dropped = int(np.sum(~in_bounds))
            logger.warning("Dropping %d out-of-range particle ids from precomputed halfset.", dropped)
        values = values[in_bounds]
        values = _filter_preserve_order(values, allowed_particles)
        if values.size > 0:
            _, first_idx = np.unique(values, return_index=True)
            if first_idx.size != values.size:
                dropped = int(values.size - first_idx.size)
                logger.warning("Dropping %d duplicate particle ids from precomputed halfset.", dropped)
            values = values[np.sort(first_idx)]
        return values.astype(np.int32, copy=False)

    # Step 1: Parse STAR file for mapping
    particles_to_tilts, tilts_to_particles = cryo_dataset.TiltSeriesDataset.parse_particle_tilt(particles_file)

    # Step 2: Optionally get tilt numbers for ntilts filtering
    tilt_numbers = None
    if ntilts is not None and ntilts > 0:
        dataset_tmp = cryo_dataset.TiltSeriesDataset(particles_file, datadir=datadir)
        tilt_numbers = dataset_tmp.tilt_numbers

    n_particles_total = len(particles_to_tilts)

    # Step 3: Determine which particles to use
    if tilt_ind_file is not None:
        particle_ind = _sanitize_particle_ids(
            _load_index_like(tilt_ind_file),
            n_particles_total=n_particles_total,
            allowed_particles=np.arange(n_particles_total, dtype=np.int32),
        )
    else:
        particle_ind = np.arange(n_particles_total, dtype=np.int32)

    if particle_ind.size == 0:
        empty = np.array([], dtype=np.int32)
        return [empty, empty]

    # Map selected particles to image indices
    allowed_image_indices = cryo_dataset.tilt_series_to_images(particle_ind, particles_file)

    # Step 4: Optionally filter by image indices
    if ind_file is not None:
        ind_images = _normalize_image_ids(
            _load_index_like(ind_file),
            n_images_total=len(tilts_to_particles),
        )
        allowed_image_indices = _filter_preserve_order(allowed_image_indices, ind_images)

    if len(allowed_image_indices) == 0:
        empty = np.array([], dtype=np.int32)
        return [empty, empty]

    # Step 5: Keep only particles with at least one allowed image.
    image_to_particle = np.fromiter(
        (tilts_to_particles[int(i)] for i in np.asarray(allowed_image_indices).reshape(-1)),
        dtype=np.int32,
        count=len(allowed_image_indices),
    )
    valid_particles = np.unique(image_to_particle)
    if valid_particles.size == 0:
        empty = np.array([], dtype=np.int32)
        return [empty, empty]

    # Step 6: Determine halfset split (by particles)
    if particle_halfset_indices_file is not None:
        split_particles_raw = _load_index_like(particle_halfset_indices_file)
        if len(split_particles_raw) != 2:
            raise ValueError("particle_halfset_indices_file must contain exactly two halfsets")
        split_particles = [
            _sanitize_particle_ids(split_particles_raw[0], n_particles_total=n_particles_total,
                                   allowed_particles=valid_particles),
            _sanitize_particle_ids(split_particles_raw[1], n_particles_total=n_particles_total,
                                   allowed_particles=valid_particles),
        ]
    else:
        split_particles = split_index_list(valid_particles)

    # Step 7: For each halfset, filter by ntilts and allowed images
    split_image_indices = []
    for half in split_particles:
        if len(half) == 0:
            split_image_indices.append(np.array([], dtype=np.int32))
            continue
        imgs = np.concatenate([particles_to_tilts[ind] for ind in half])
        if ntilts is not None:
            if ntilts <= 0:
                imgs = imgs[:0]
            else:
                imgs = imgs[tilt_numbers[imgs] < ntilts]
        imgs = _filter_preserve_order(imgs, allowed_image_indices)
        split_image_indices.append(imgs)

    return split_image_indices


# ---------------------------------------------------------------------------
# RELION halfset detection
# ---------------------------------------------------------------------------

def _read_relion_halfsets_from_star(particles_file, ind_file=None, datadir=None,
                                    strip_prefix=None):
    """Try to read halfset assignments from _rlnRandomSubset in a STAR file.

    Returns ``(halfsets, n_total)`` where *halfsets* is a list of two index
    arrays when the column is present and valid, or ``None`` when the column
    is absent or the file is not a STAR file.
    """
    _normalize_image_indices, _ = _get_normalize_and_dedup()

    if not str(particles_file).endswith('.star'):
        return None, None

    try:
        from recovar.data_io.starfile import read_star
        df, _ = read_star(particles_file)
    except (ImportError, FileNotFoundError, ValueError):
        return None, None

    n_total = len(df)

    if '_rlnRandomSubset' not in df.columns:
        return None, n_total

    subsets = df['_rlnRandomSubset'].values.astype(int)
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
        raw_indices = _load_index_like(ind_file)
        n_images_total = len(subsets)
        ind = _normalize_image_indices(raw_indices, n_images_total=n_images_total, name="ind_file")
        halfsets = [h[np.isin(h, ind)] for h in halfsets]

    if len(halfsets[0]) == 0 or len(halfsets[1]) == 0:
        logger.warning("RELION halfsets are empty after filtering; falling back to random split")
        return None, n_total

    logger.info(
        "Using RELION halfsets from _rlnRandomSubset: %d and %d images",
        len(halfsets[0]), len(halfsets[1]),
    )
    return halfsets, n_total


# ---------------------------------------------------------------------------
# High-level dataset splitting
# ---------------------------------------------------------------------------

def get_split_datasets(particles_file, poses_file=None, ctf_file=None, datadir=None,
                       uninvert_data=False, ind_file=None,
                       padding=0, n_images=None, tilt_series=False,
                       tilt_series_ctf=None,
                       angle_per_tilt=3, dose_per_tilt=2.9,
                       ind_split=None, lazy=False, premultiplied_ctf=False,
                       strip_prefix=None, downsample_D=None):
    """Load a dataset and split it into two CryoEMHalfsets.

    Loads ONE full dataset and stores ``halfset_indices`` on it so that
    the new ``dataset.iterate(half, ...)`` API can be used.  Also creates
    two subset views for backward-compatible ``cryos[0]`` / ``cryos[1]``
    access.
    """
    from recovar.data_io.dataset import load_dataset, CryoEMHalfsets

    all_indices = np.unique(np.concatenate(ind_split))

    full = load_dataset(
        particles_file, poses_file, ctf_file, datadir=datadir,
        n_images=n_images, ind=all_indices, lazy=lazy, padding=padding,
        uninvert_data=uninvert_data, tilt_series=tilt_series,
        tilt_series_ctf=tilt_series_ctf, angle_per_tilt=angle_per_tilt,
        dose_per_tilt=dose_per_tilt, premultiplied_ctf=premultiplied_ctf,
        strip_prefix=strip_prefix, downsample_D=downsample_D,
    )

    orig_to_local = np.empty(int(all_indices.max()) + 1, dtype=np.int32)
    orig_to_local[all_indices] = np.arange(len(all_indices), dtype=np.int32)

    local_split = [orig_to_local[s] for s in ind_split]

    # Store halfset indices on the full dataset for the new iterate() API
    full.halfset_indices = [np.asarray(s, dtype=np.int32) for s in local_split]

    # Create subset views for backward-compatible half-by-half access
    half1 = full.subset(local_split[0])
    half2 = full.subset(local_split[1])
    return CryoEMHalfsets(half1, half2, dataset=full)


def make_dataset_loader_dict(args):
    """Build the loader-configuration dict consumed by get_split_datasets."""
    uninvert_data_str = getattr(args, 'uninvert_data', 'false')
    if uninvert_data_str in ('automatic', 'false'):
        uninvert_data = False
    elif uninvert_data_str == 'true':
        uninvert_data = True
    else:
        raise ValueError(
            f"uninvert_data must be 'automatic', 'true', or 'false'; got {uninvert_data_str!r}"
        )

    return {
        'particles_file':   args.particles,
        'ctf_file':         getattr(args, 'ctf', None),
        'poses_file':       getattr(args, 'poses', None),
        'datadir':          getattr(args, 'datadir', None),
        'n_images':         getattr(args, 'n_images', -1),
        'ind_file':         getattr(args, 'ind', None),
        'padding':          getattr(args, 'padding', 0),
        'tilt_series':      getattr(args, 'tilt_series', False),
        'tilt_series_ctf':  getattr(args, 'tilt_series_ctf', 'cryoem'),
        'angle_per_tilt':   getattr(args, 'angle_per_tilt', None),
        'dose_per_tilt':    getattr(args, 'dose_per_tilt', None),
        'premultiplied_ctf': getattr(args, 'premultiplied_ctf', False),
        'strip_prefix':     getattr(args, 'strip_prefix', None),
        'downsample_D':     getattr(args, 'downsample', None),
        'uninvert_data':    uninvert_data,
    }


def figure_out_halfsets(args):
    """Determine which images belong to each reconstruction half-set.

    Priority order:
      1. Explicit halfsets file (``--halfsets``).
      2. _rlnRandomSubset column in the STAR file (RELION convention).
      3. Random 50/50 split of all valid images.
    """
    from recovar.data_io.dataset import get_num_images_in_dataset
    _normalize_image_indices, _ = _get_normalize_and_dedup()

    is_tilt = getattr(args, 'tilt_series', False) or getattr(args, 'tilt_series_ctf', 'cryoem') != 'cryoem'
    datadir = getattr(args, 'datadir', None)
    strip_prefix = getattr(args, 'strip_prefix', None)
    ind_file = getattr(args, 'ind', None)
    tilt_ind_file = getattr(args, 'tilt_ind', None)
    ntilts = getattr(args, 'ntilts', None)
    n_images = getattr(args, 'n_images', None) or -1

    if args.halfsets is None:
        n_total_from_star = None
        if not is_tilt:
            halfsets, n_total_from_star = _read_relion_halfsets_from_star(
                args.particles, ind_file=ind_file,
                datadir=datadir, strip_prefix=strip_prefix,
            )
            if halfsets is not None:
                if n_images > 0:
                    halfsets = [halfset[:n_images // 2] for halfset in halfsets]
                    logger.info("using only %s particles", n_images)
                return halfsets

        logger.info("Randomly splitting dataset into halfsets")
        if is_tilt:
            halfsets = get_split_tilt_indices(
                args.particles, ind_file=ind_file, tilt_ind_file=tilt_ind_file,
                ntilts=ntilts, datadir=datadir,
            )
        else:
            halfsets = get_split_indices(
                args.particles, datadir=datadir, strip_prefix=strip_prefix,
                ind_file=ind_file, n_images=n_total_from_star,
            )

    else:
        logger.info("Loading halfsets from file")
        if is_tilt:
            halfsets = get_split_tilt_indices(
                args.particles, ind_file=ind_file, tilt_ind_file=tilt_ind_file,
                ntilts=ntilts, datadir=datadir,
                particle_halfset_indices_file=args.halfsets,
            )
        else:
            with open(args.halfsets, 'rb') as f:
                halfsets = pickle.load(f)
            logger.info("Loaded halfsets from file")
            if len(halfsets) != 2:
                raise ValueError("halfsets file must contain exactly two halfsets")

            needs_n_images = any(np.asarray(h).dtype == bool for h in halfsets)
            n_images_total = None
            if needs_n_images:
                n_images_total = get_num_images_in_dataset(
                    args.particles, datadir=datadir, strip_prefix=strip_prefix,
                )
            halfsets = [
                _normalize_image_indices(halfsets[0], n_images_total=n_images_total, name="halfsets[0]"),
                _normalize_image_indices(halfsets[1], n_images_total=n_images_total, name="halfsets[1]"),
            ]

            if ind_file is not None:
                ind_raw = _load_index_like(ind_file)
                if n_images_total is None and np.asarray(ind_raw).dtype == bool:
                    n_images_total = get_num_images_in_dataset(
                        args.particles, datadir=datadir, strip_prefix=strip_prefix,
                    )
                ind = _normalize_image_indices(ind_raw, n_images_total=n_images_total, name="ind")
                halfsets = [
                    np.asarray(halfset)[np.isin(np.asarray(halfset), ind)]
                    for halfset in halfsets
                ]

    if n_images > 0:
        halfsets = [halfset[:n_images // 2] for halfset in halfsets]
        logger.info("using only %s particles", n_images)
    return halfsets


def load_dataset_from_args(args, lazy=False, ind_split=None):
    """Load a dataset split into halfsets from command-line args."""
    if ind_split is None:
        ind_split = figure_out_halfsets(args)
    dataset_loader_dict = make_dataset_loader_dict(args)
    return get_split_datasets(**dataset_loader_dict, ind_split=ind_split, lazy=lazy)
