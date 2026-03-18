"""Batch iteration utilities for explicit dataset field access.

The iterator in this module yields plain batch fields:

``(images, rotation_matrices, translations, ctf_params,
noise_variance, particle_indices, image_indices)``

This keeps the data path explicit at call sites and avoids passing a
catch-all batch object through downstream functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


BatchFields = Tuple[object, np.ndarray, np.ndarray, np.ndarray, object, object, np.ndarray]


@dataclass(frozen=True)
class IteratorOptions:
    """Configuration for dataset batch iteration."""

    batch_size: int
    batch_mode: Literal["images", "groups"] = "images"
    index_subset: Optional[np.ndarray] = None
    noise_model: object = None
    noise_half: bool = True
    noise_by_particle: bool = False


class BatchIterator:
    """Yield explicit batch fields from a dataset."""

    def __init__(self, dataset, options: IteratorOptions):
        self.dataset = dataset
        self.options = options

    def _select_generator(self):
        subset = self.options.index_subset
        batch_size = self.options.batch_size
        if self.options.batch_mode == "images":
            if subset is None:
                return self.dataset._get_image_generator(batch_size=batch_size)
            return self.dataset._get_image_subset_generator(
                batch_size=batch_size,
                subset_indices=subset,
            )
        if subset is None:
            return self.dataset._get_dataset_generator(batch_size=batch_size)
        return self.dataset._get_dataset_subset_generator(
            batch_size=batch_size,
            subset_indices=subset,
        )

    def __iter__(self):
        generator = self._select_generator()
        use_particle_noise = (
            self.options.noise_by_particle and self.options.batch_mode == "groups"
        )
        noise_model = self.options.noise_model

        for images, particle_indices, image_indices in generator:
            image_indices = np.asarray(image_indices, dtype=np.int32).reshape(-1)
            particle_indices = np.asarray(particle_indices)
            rotation_matrices, translations, ctf_params = self.dataset.metadata.get_batch(image_indices)

            if noise_model is None:
                noise_variance = None
            else:
                noise_indices = particle_indices if use_particle_noise else image_indices
                getter = noise_model.get_half if self.options.noise_half else noise_model.get
                noise_variance = getter(noise_indices)

            yield (
                images,
                rotation_matrices,
                translations,
                ctf_params,
                noise_variance,
                particle_indices,
                image_indices,
            )


def coerce_batch_fields(
    images_or_batch,
    rotation_matrices=None,
    translations=None,
    ctf_params=None,
    noise_variance=None,
    particle_indices=None,
    image_indices=None,
) -> BatchFields:
    """Accept either explicit fields or a legacy batch object.

    This is only for compatibility while call sites are migrated to explicit
    field passing. New code should pass arrays explicitly.
    """

    if (
        rotation_matrices is None
        and translations is None
        and ctf_params is None
        and hasattr(images_or_batch, "rotation_matrices")
        and hasattr(images_or_batch, "translations")
        and hasattr(images_or_batch, "ctf_params")
    ):
        batch = images_or_batch
        return (
            batch.images,
            np.asarray(batch.rotation_matrices),
            np.asarray(batch.translations),
            np.asarray(batch.ctf_params),
            batch.noise_variance,
            getattr(batch, "particle_indices", None),
            getattr(batch, "image_indices", None),
        )

    if (
        rotation_matrices is None
        and translations is None
        and ctf_params is None
        and isinstance(images_or_batch, tuple)
        and len(images_or_batch) == 7
    ):
        (
            images,
            rotation_matrices,
            translations,
            ctf_params,
            tuple_noise_variance,
            particle_indices,
            image_indices,
        ) = images_or_batch
        if noise_variance is None:
            noise_variance = tuple_noise_variance
        return (
            images,
            np.asarray(rotation_matrices),
            np.asarray(translations),
            np.asarray(ctf_params),
            noise_variance,
            np.asarray(particle_indices),
            np.asarray(image_indices, dtype=np.int32).reshape(-1),
        )

    if rotation_matrices is None or translations is None or ctf_params is None:
        raise ValueError(
            "Explicit batch fields require rotation_matrices, translations, and ctf_params."
        )

    if image_indices is not None:
        image_indices = np.asarray(image_indices, dtype=np.int32).reshape(-1)

    if particle_indices is not None:
        particle_indices = np.asarray(particle_indices)

    return (
        images_or_batch,
        np.asarray(rotation_matrices),
        np.asarray(translations),
        np.asarray(ctf_params),
        noise_variance,
        particle_indices,
        image_indices,
    )


def iter_batch_fields(iterable):
    """Yield normalized explicit batch fields from any iterator source."""
    for batch_item in iterable:
        yield coerce_batch_fields(batch_item)
