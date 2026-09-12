"""Dataset batch fetch helpers shared by dense/local EM paths."""

from __future__ import annotations

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
