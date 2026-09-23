"""Dataset batch fetch helpers shared by dense/local EM paths."""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np


def fetch_indexed_batch(experiment_dataset, image_indices):
    """Fetch one explicitly indexed image batch and return dataset indices."""

    batch_iter = experiment_dataset.iter_batches(
        len(image_indices),
        indices=np.asarray(image_indices),
        by_image=False,
    )
    batch_data, _, _, ctf_params, _, _, indices = next(batch_iter)
    return batch_data, ctf_params, np.asarray(indices)


def original_image_indices(experiment_dataset, local_indices) -> np.ndarray:
    """Map local batch image indices to original image ids for debug dumps."""
    local_indices = np.asarray(local_indices, dtype=np.int64)
    mapper = getattr(experiment_dataset, "original_image_indices_from_local", None)
    if mapper is not None:
        return np.asarray(mapper(local_indices), dtype=np.int64)
    original_indices_all = getattr(experiment_dataset, "dataset_indices", None)
    if original_indices_all is None:
        return local_indices
    return np.asarray(original_indices_all, dtype=np.int64)[local_indices]


PREFETCH_BATCHES_ENV = "RECOVAR_EM_PREFETCH_BATCHES"


def prefetch_depth() -> int:
    """Number of image batches to read ahead on a worker thread (0 = off).

    The pass-2 chunk loop and the coarse significance pass fetch each batch
    synchronously (page-cache read, collation, host->device copy) before the
    device can start on it: ~25 s per 100k/256 K=4 iteration in pass 2 plus the
    coarse pass's share (job 13826984). Reading the next batch while the device
    works on the current one hides that; the batches and their order are
    unchanged, so every result is bit-identical.
    """
    raw = os.environ.get(PREFETCH_BATCHES_ENV, "0").strip()
    if raw == "":
        return 0
    try:
        depth = int(raw)
    except ValueError as exc:
        raise ValueError(f"{PREFETCH_BATCHES_ENV} must be a non-negative integer, got {raw!r}") from exc
    if depth < 0:
        raise ValueError(f"{PREFETCH_BATCHES_ENV} must be a non-negative integer, got {raw!r}")
    return depth


@contextmanager
def prefetched_batches(iterable, *, depth=None):
    """Own an ordered, bounded batch iterator and close it on every exit.

    Reuse the dataset loader's cancellable queue: closing a consumer must not
    leave a producer blocked on a full queue with device batches retained.
    Depth zero preserves synchronous iteration. Source values and exceptions
    retain their order; the consumer owns the lifetime explicitly.
    """
    from recovar.data_io.image_backends import _PrefetchIterator

    depth = prefetch_depth() if depth is None else int(depth)
    if depth < 0:
        raise ValueError("prefetch depth must be non-negative")
    iterator = iter(_PrefetchIterator(iterable, buffer_size=depth)) if depth else iter(iterable)
    try:
        yield iterator
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
