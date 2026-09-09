"""Host encodings for coarse significant-sample support.

None means full support; explicit IDs or a sparse complement represent partial
support. Selection policy and scoring remain in significance.py.
"""

from typing import NamedTuple

import numpy as np


class ComplementSignificantSampleIndices(NamedTuple):
    """Exact dense significance mask stored as a sparse complement.

    ``None`` remains the representation for an all-True support mask.  This
    object is used when most, but not all, coarse samples are significant:
    storing the included indices would be O(n_samples) per image, while storing
    the excluded tail preserves the exact RELION adaptive mask with bounded
    host memory.
    """

    excluded_indices: np.ndarray
    total_size: int

    @property
    def size(self) -> int:
        return int(self.total_size) - int(np.asarray(self.excluded_indices).size)


def significant_sample_count(samples, total_size: int) -> int:
    """Return the number of included coarse samples for any support encoding."""

    if samples is None:
        return int(total_size)
    if isinstance(samples, ComplementSignificantSampleIndices):
        return int(samples.size)
    return int(np.asarray(samples).size)


def significant_sample_ids(samples, total_size: int) -> np.ndarray:
    """Materialize included ids for diagnostics or dense fallbacks."""

    if samples is None:
        return np.arange(int(total_size), dtype=np.int64)
    if isinstance(samples, ComplementSignificantSampleIndices):
        excluded = np.asarray(samples.excluded_indices, dtype=np.int64).reshape(-1)
        if excluded.size == 0:
            return np.arange(int(total_size), dtype=np.int64)
        keep = np.ones(int(total_size), dtype=bool)
        keep[excluded] = False
        return np.flatnonzero(keep).astype(np.int64, copy=False)
    return np.asarray(samples, dtype=np.int64).reshape(-1)


def compact_significant_sample_indices_from_mask(mask) -> object:
    """Encode one boolean significance mask without materializing dense keeps."""

    mask_np = np.asarray(mask, dtype=bool).reshape(-1)
    if bool(np.all(mask_np)):
        return None
    included = int(np.count_nonzero(mask_np))
    excluded = int(mask_np.size - included)
    if included > excluded:
        return ComplementSignificantSampleIndices(
            excluded_indices=np.flatnonzero(~mask_np).astype(np.int32),
            total_size=int(mask_np.size),
        )
    return np.flatnonzero(mask_np).astype(np.int32)


# Preserve stored NamedTuple identity; significance imports this same class.
ComplementSignificantSampleIndices.__module__ = "recovar.em.dense_single_volume.helpers.significance"
