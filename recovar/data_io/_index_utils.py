"""Explicit index-domain helpers for cryo-EM / cryo-ET datasets.

This module centralizes the translation between:

- local image indices inside a loaded dataset view
- original image indices in the source file
- local group indices inside a loaded dataset view
- original group indices in the source file

For SPA datasets, image and group domains are identical. For grouped datasets
such as cryo-ET tilt series, a group corresponds to one particle / tilt
series and expands to one or more local images.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


def normalize_indices(
    values,
    n_total: int,
    *,
    name: str = "indices",
    allow_none: bool = False,
):
    """Normalize int/bool indices to an int32 array with bounds checking."""
    if values is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must not be None")

    arr = np.asarray(values)

    if arr.dtype == bool:
        if arr.ndim != 1:
            raise ValueError(f"{name} boolean mask must be 1D")
        if arr.size != int(n_total):
            raise ValueError(
                f"{name} boolean mask length {arr.size} must match total size {int(n_total)}"
            )
        return np.flatnonzero(arr).astype(np.int32, copy=False)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} must be integer indices or boolean mask")

    arr = arr.astype(np.int64, copy=False).reshape(-1)
    if arr.size == 0:
        return arr.astype(np.int32, copy=False)

    if np.any(arr < 0):
        raise IndexError(f"{name} contains negative values")
    if np.any(arr >= int(n_total)):
        raise IndexError(
            f"{name} contains out-of-range values for total size {int(n_total)}"
        )

    return arr.astype(np.int32, copy=False)


def load_index_like(value):
    """Return an in-memory index selection from an array-like or pickle path."""
    if value is None:
        return None
    if isinstance(value, (np.ndarray, list, tuple)):
        return value
    with open(value, "rb") as handle:
        return pickle.load(handle)


def normalize_image_indices(values, *, n_total: Optional[int] = None, name: str = "indices"):
    """Normalize image indices, optionally without a known dataset size.

    When ``n_total`` is known, this is strict bounds-checked normalization.
    When it is unknown, the function still validates rank, dtype, and
    non-negativity, but cannot reject out-of-range values.
    """
    if n_total is not None:
        return normalize_indices(values, int(n_total), name=name)

    if values is None:
        raise ValueError(f"{name} must not be None")

    arr = np.asarray(values)
    if arr.dtype == bool:
        raise ValueError(f"{name} boolean mask requires known dataset size for validation")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} indices must be 1D")
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} indices must be integer or boolean mask")

    arr = arr.astype(np.int64, copy=False).reshape(-1)
    if arr.size > 0 and np.any(arr < 0):
        raise ValueError(f"{name} indices must be non-negative")
    return arr.astype(np.int32, copy=False)


def deduplicate_preserve_order(values, *, name: str = "indices"):
    """Drop duplicate values while keeping the first occurrence order."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if arr.size == 0:
        return arr.astype(np.int32, copy=False)
    _, first_idx = np.unique(arr, return_index=True)
    return arr[np.sort(first_idx)].astype(np.int32, copy=False)


def filter_preserve_order(values, allowed):
    """Return the subset of *values* that appears in *allowed*, keeping order."""
    values = np.asarray(values)
    allowed = np.asarray(allowed)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError("values must be 1D")
    if allowed.ndim == 0:
        allowed = allowed.reshape(1)
    if allowed.ndim != 1:
        raise ValueError("allowed must be 1D")
    if values.size == 0:
        return values.astype(np.int32, copy=False)
    return values[np.isin(values, allowed)].astype(np.int32, copy=False)


def _normalize_index_array(values, *, name: str):
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} must contain integer indices")
    arr = arr.astype(np.int64, copy=False).reshape(-1)
    if arr.size > 0 and np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    return arr.astype(np.int32, copy=False)


def _build_last_write_inverse(original_indices):
    # Reverse lookups in subset views need to cope with duplicate original ids.
    # We intentionally keep the *last* local position for each original id so
    # remaps match the historical subset behavior used elsewhere in recovar.
    original_indices = np.asarray(original_indices, dtype=np.int32).reshape(-1)
    if original_indices.size == 0:
        return np.full(0, -1, dtype=np.int32)
    size = int(np.max(original_indices)) + 1
    inverse = np.full(size, -1, dtype=np.int32)
    inverse[original_indices] = np.arange(original_indices.size, dtype=np.int32)
    return inverse


@dataclass(frozen=True)
class DatasetIndexLayout:
    """Index mapping for one dataset view.

    Parameters
    ----------
    original_image_indices
        For each local image index, the source-file image index.
    grouped
        ``False`` for SPA, where image and group domains are the same.
        ``True`` for grouped datasets such as tilt-series data.
    original_group_indices
        For each local group index, the source-file group index.
    group_local_image_indices
        Only used when ``grouped=True``. Each entry lists the local images
        belonging to one local group.

    Notes
    -----
    Original image/group ids may repeat in SPA subsets created from duplicate
    selections. Reverse lookup therefore uses explicit "last-write-wins"
    semantics, matching the previous subset remap behavior.
    """

    original_image_indices: np.ndarray
    grouped: bool = False
    original_group_indices: Optional[np.ndarray] = None
    group_local_image_indices: Optional[tuple[np.ndarray, ...]] = None

    def __post_init__(self):
        original_image_indices = _normalize_index_array(
            self.original_image_indices,
            name="original_image_indices",
        )
        n_images = int(original_image_indices.size)

        if not bool(self.grouped):
            # SPA: images and groups are the same domain, so every local image is
            # its own local group.
            if self.original_group_indices is None:
                original_group_indices = original_image_indices.copy()
            else:
                original_group_indices = _normalize_index_array(
                    self.original_group_indices,
                    name="original_group_indices",
                )
                if original_group_indices.shape != original_image_indices.shape:
                    raise ValueError(
                        "SPA layouts require original_group_indices to match original_image_indices"
                    )
            image_local_to_group_local = np.arange(n_images, dtype=np.int32)
            group_local_image_indices = None
        else:
            # Grouped datasets (cryo-ET tilt series): validate the explicit
            # image->group partition and build a dense local image -> local
            # group lookup table for fast remaps later on.
            if self.original_group_indices is None:
                raise ValueError("grouped layouts require original_group_indices")
            if self.group_local_image_indices is None:
                raise ValueError("grouped layouts require group_local_image_indices")

            original_group_indices = _normalize_index_array(
                self.original_group_indices,
                name="original_group_indices",
            )
            if original_group_indices.size != len(self.group_local_image_indices):
                raise ValueError(
                    "grouped layouts require one original_group_indices entry per group"
                )

            image_local_to_group_local = np.full(n_images, -1, dtype=np.int32)
            normalized_groups = []
            for group_idx, local_images in enumerate(self.group_local_image_indices):
                local_images = _normalize_index_array(
                    local_images,
                    name=f"group_local_image_indices[{group_idx}]",
                )
                if local_images.size == 0:
                    raise ValueError(
                        f"group_local_image_indices[{group_idx}] must not be empty"
                    )
                if local_images.size > 0 and np.any(local_images >= n_images):
                    raise IndexError(
                        f"group_local_image_indices[{group_idx}] contains out-of-range local images"
                    )
                normalized_groups.append(local_images.astype(np.int32, copy=False))
                image_local_to_group_local[local_images] = np.int32(group_idx)

            if n_images > 0 and np.any(image_local_to_group_local < 0):
                raise ValueError(
                    "group_local_image_indices must cover every local image exactly once"
                )
            group_local_image_indices = tuple(normalized_groups)

        object.__setattr__(
            self,
            "original_image_indices",
            original_image_indices.astype(np.int32, copy=False),
        )
        object.__setattr__(
            self,
            "original_group_indices",
            original_group_indices.astype(np.int32, copy=False),
        )
        object.__setattr__(self, "grouped", bool(self.grouped))
        object.__setattr__(self, "group_local_image_indices", group_local_image_indices)
        object.__setattr__(self, "_image_local_to_group_local", image_local_to_group_local)
        object.__setattr__(
            self,
            "_original_image_to_local",
            _build_last_write_inverse(original_image_indices),
        )
        object.__setattr__(
            self,
            "_original_group_to_local",
            _build_last_write_inverse(original_group_indices),
        )

    @classmethod
    def from_image_indices(cls, original_image_indices):
        return cls(
            original_image_indices=np.asarray(original_image_indices, dtype=np.int32),
            grouped=False,
        )

    @classmethod
    def from_grouped_images(
        cls,
        *,
        original_image_indices,
        original_group_indices,
        group_local_image_indices: Iterable[np.ndarray],
    ):
        return cls(
            original_image_indices=np.asarray(original_image_indices, dtype=np.int32),
            grouped=True,
            original_group_indices=np.asarray(original_group_indices, dtype=np.int32),
            group_local_image_indices=tuple(
                np.asarray(local_images, dtype=np.int32)
                for local_images in group_local_image_indices
            ),
        )

    @property
    def n_images(self) -> int:
        return int(self.original_image_indices.size)

    @property
    def n_groups(self) -> int:
        return int(self.original_group_indices.size)

    @property
    def image_local_to_group_local(self):
        return self._image_local_to_group_local

    def subset(self, local_image_indices):
        # Build a new layout in the child-local coordinate system:
        # 1. keep only the selected local images
        # 2. compact them to 0..N_child-1
        # 3. rebuild grouped structure so parent-local image ids never leak out
        local_image_indices = normalize_indices(
            local_image_indices,
            self.n_images,
            name="local_image_indices",
        )
        if not self.grouped:
            return DatasetIndexLayout.from_image_indices(
                self.original_image_indices[local_image_indices]
            )

        original_image_indices = self.original_image_indices[local_image_indices]
        old_to_new_local_image = np.full(self.n_images, -1, dtype=np.int32)
        old_to_new_local_image[local_image_indices] = np.arange(
            local_image_indices.size,
            dtype=np.int32,
        )

        # Keep only groups touched by the subset, then remap each group's old
        # local-image list into the child's compact local-image numbering.
        selected_old_groups = np.unique(
            self._image_local_to_group_local[local_image_indices]
        ).astype(np.int32, copy=False)
        group_local_image_indices = []
        for old_group_idx in selected_old_groups:
            old_group_images = self.group_local_image_indices[int(old_group_idx)]
            new_group_images = old_to_new_local_image[old_group_images]
            new_group_images = new_group_images[new_group_images >= 0]
            if new_group_images.size == 0:
                continue
            group_local_image_indices.append(new_group_images.astype(np.int32, copy=False))

        return DatasetIndexLayout.from_grouped_images(
            original_image_indices=original_image_indices,
            original_group_indices=self.original_group_indices[selected_old_groups],
            group_local_image_indices=group_local_image_indices,
        )

    def original_image_indices_for_local(self, local_image_indices=None):
        if local_image_indices is None:
            return self.original_image_indices
        local_image_indices = normalize_indices(
            local_image_indices,
            self.n_images,
            name="local_image_indices",
        )
        return self.original_image_indices[local_image_indices].astype(np.int32, copy=False)

    def original_group_indices_for_local(self, local_group_indices=None):
        if local_group_indices is None:
            return self.original_group_indices
        local_group_indices = normalize_indices(
            local_group_indices,
            self.n_groups,
            name="local_group_indices",
        )
        return self.original_group_indices[local_group_indices].astype(np.int32, copy=False)

    def local_image_indices_from_original(self, original_image_indices, *, allow_missing=False):
        # Reverse-map source-file image ids back into this dataset view.
        # This is the main bridge used when a child dataset/view needs to turn
        # "original file ids" back into the current local numbering.
        original_image_indices = _normalize_index_array(
            original_image_indices,
            name="original_image_indices",
        )
        if original_image_indices.size == 0:
            return original_image_indices
        local_image_indices = np.full(original_image_indices.shape, -1, dtype=np.int32)
        in_bounds = original_image_indices < self._original_image_to_local.size
        local_image_indices[in_bounds] = self._original_image_to_local[original_image_indices[in_bounds]]
        if not allow_missing and np.any(~in_bounds):
            raise IndexError("original_image_indices contains ids outside this layout")
        if not allow_missing and np.any(local_image_indices < 0):
            raise IndexError("original_image_indices contains ids not present in this layout")
        return local_image_indices.astype(np.int32, copy=False)

    def local_group_indices_from_original(self, original_group_indices, *, allow_missing=False):
        # Same as local_image_indices_from_original(), but in particle/tilt
        # group space for grouped datasets.
        original_group_indices = _normalize_index_array(
            original_group_indices,
            name="original_group_indices",
        )
        if original_group_indices.size == 0:
            return original_group_indices
        local_group_indices = np.full(original_group_indices.shape, -1, dtype=np.int32)
        in_bounds = original_group_indices < self._original_group_to_local.size
        local_group_indices[in_bounds] = self._original_group_to_local[original_group_indices[in_bounds]]
        if not allow_missing and np.any(~in_bounds):
            raise IndexError("original_group_indices contains ids outside this layout")
        if not allow_missing and np.any(local_group_indices < 0):
            raise IndexError("original_group_indices contains ids not present in this layout")
        return local_group_indices.astype(np.int32, copy=False)

    def local_group_indices_from_local_images(self, local_image_indices):
        # Collapse image selections down to the unique local groups they touch.
        # For SPA this is a no-op because image ids and group ids are identical.
        local_image_indices = normalize_indices(
            local_image_indices,
            self.n_images,
            name="local_image_indices",
        )
        if not self.grouped:
            return local_image_indices.astype(np.int32, copy=False)
        return np.unique(
            self._image_local_to_group_local[local_image_indices]
        ).astype(np.int32, copy=False)

    def original_group_indices_from_local_images(self, local_image_indices):
        local_group_indices = self.local_group_indices_from_local_images(local_image_indices)
        return self.original_group_indices_for_local(local_group_indices)

    def local_image_indices_from_local_groups(self, local_group_indices):
        # Expand particle/tilt-group selections back to the concrete local image
        # ids belonging to those groups.
        local_group_indices = normalize_indices(
            local_group_indices,
            self.n_groups,
            name="local_group_indices",
        )
        if not self.grouped:
            return local_group_indices.astype(np.int32, copy=False)
        if local_group_indices.size == 0:
            return np.array([], dtype=np.int32)
        return np.concatenate(
            [self.group_local_image_indices[int(group_idx)] for group_idx in local_group_indices]
        ).astype(np.int32, copy=False)

    def original_image_indices_from_local_groups(self, local_group_indices):
        local_image_indices = self.local_image_indices_from_local_groups(local_group_indices)
        return self.original_image_indices_for_local(local_image_indices)


class TiltSeriesOriginalIndexMap:
    """Original-file particle/image mapping used by cryo-ET selection logic."""

    def __init__(self, particle_to_images, image_to_particle, tilt_numbers=None):
        self._particle_to_images = tuple(
            _normalize_index_array(image_indices, name=f"particle_to_images[{idx}]")
            for idx, image_indices in enumerate(particle_to_images)
        )
        self._image_to_particle = _normalize_index_array(
            image_to_particle,
            name="image_to_particle",
        )
        self.tilt_numbers = None if tilt_numbers is None else _normalize_index_array(
            tilt_numbers,
            name="tilt_numbers",
        )
        if self.tilt_numbers is not None and self.tilt_numbers.shape[0] != self._image_to_particle.shape[0]:
            raise ValueError("tilt_numbers must have one entry per original image")

    @classmethod
    def from_particles_file(cls, particles_file, *, datadir=None, ntilts=None):
        from recovar.data_io import image_backends

        # Parse the original file once and freeze the canonical
        # particle<->image relationship before any dataset subsetting/view logic.
        particle_to_images, tilt_to_particle = image_backends.TiltSeriesDataset.parse_particle_tilt(
            particles_file
        )
        particle_to_images = tuple(
            _normalize_index_array(image_indices, name=f"particle_to_images[{idx}]")
            for idx, image_indices in enumerate(particle_to_images)
        )
        if isinstance(tilt_to_particle, dict):
            if tilt_to_particle:
                n_images = int(max(tilt_to_particle)) + 1
                image_to_particle = np.full(n_images, -1, dtype=np.int32)
                for image_idx, particle_idx in tilt_to_particle.items():
                    image_to_particle[int(image_idx)] = int(particle_idx)
            else:
                image_to_particle = np.array([], dtype=np.int32)
        else:
            image_to_particle = np.asarray(tilt_to_particle, dtype=np.int32).reshape(-1)

        inferred_n_images = (
            max((int(np.max(imgs)) for imgs in particle_to_images if imgs.size > 0), default=-1) + 1
        )
        if image_to_particle.size < inferred_n_images:
            extended = np.full(inferred_n_images, -1, dtype=np.int32)
            if image_to_particle.size > 0:
                extended[:image_to_particle.size] = image_to_particle
            image_to_particle = extended

        tilt_numbers = None
        if ntilts is not None and ntilts > 0:
            tilt_dataset = image_backends.TiltSeriesDataset(
                particles_file,
                datadir=datadir,
                lazy=True,
            )
            tilt_numbers = np.asarray(tilt_dataset.tilt_numbers, dtype=np.int32)

        return cls(
            particle_to_images=particle_to_images,
            image_to_particle=image_to_particle,
            tilt_numbers=tilt_numbers,
        )

    @property
    def n_particles(self) -> int:
        return len(self._particle_to_images)

    @property
    def n_images(self) -> int:
        return int(self._image_to_particle.shape[0])

    @property
    def particle_to_images(self):
        return self._particle_to_images

    @property
    def image_to_particle(self):
        return self._image_to_particle

    def sanitize_particle_indices(self, values, *, name, allowed_particles=None):
        # CLI/user-provided particle selections may arrive as boolean masks,
        # integer lists, or preloaded arrays. Normalize them once here so the
        # rest of the halfset logic can assume a clean int32 particle-id array.
        arr = np.asarray(values)
        if arr.dtype == bool:
            if arr.ndim != 1:
                raise ValueError(f"{name} boolean mask must be 1D")
            if arr.size != int(self.n_particles):
                raise ValueError(
                    f"{name} boolean mask length {arr.size} must match number of particles {int(self.n_particles)}"
                )
            particle_indices = np.flatnonzero(arr).astype(np.int32, copy=False)
        else:
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.ndim != 1:
                raise ValueError(f"{name} ids must be 1D")
            if arr.dtype.kind not in ("i", "u"):
                raise TypeError(f"{name} ids must be integer or boolean mask")
            arr = arr.astype(np.int64, copy=False).reshape(-1)
            in_bounds = (arr >= 0) & (arr < self.n_particles)
            particle_indices = arr[in_bounds].astype(np.int32, copy=False)
        if allowed_particles is not None:
            particle_indices = filter_preserve_order(particle_indices, allowed_particles)
        # Particle-domain selections should be unique: selecting the same
        # particle twice should never duplicate the whole tilt series.
        return deduplicate_preserve_order(particle_indices, name=name)

    def sanitize_image_indices(self, values, *, name, allowed_images=None):
        # Image selections stay in image space and preserve order. Unlike
        # particles we intentionally do not deduplicate here.
        arr = np.asarray(values)
        if arr.dtype == bool:
            if arr.ndim != 1:
                raise ValueError(f"{name} boolean mask must be 1D")
            if arr.size != int(self.n_images):
                raise ValueError(
                    f"{name} boolean mask length {arr.size} must match total size {int(self.n_images)}"
                )
            image_indices = np.flatnonzero(arr).astype(np.int32, copy=False)
        else:
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1D")
            if arr.dtype.kind not in ("i", "u"):
                raise TypeError(f"{name} must be integer array")
            arr = arr.astype(np.int64, copy=False).reshape(-1)
            in_bounds = (arr >= 0) & (arr < self.n_images)
            image_indices = arr[in_bounds].astype(np.int32, copy=False)
        if allowed_images is not None:
            image_indices = filter_preserve_order(image_indices, allowed_images)
        return image_indices.astype(np.int32, copy=False)

    def particle_indices_from_images(self, image_indices):
        # Many tilt images can belong to the same particle, so image->particle
        # always collapses duplicates to a unique particle-id list.
        image_indices = normalize_indices(image_indices, self.n_images, name="image_indices")
        if image_indices.size == 0:
            return np.array([], dtype=np.int32)
        particle_indices = self._image_to_particle[image_indices]
        particle_indices = particle_indices[particle_indices >= 0]
        if particle_indices.size == 0:
            return np.array([], dtype=np.int32)
        return np.unique(particle_indices).astype(np.int32, copy=False)

    def image_indices_from_particles(
        self,
        particle_indices,
        *,
        allowed_images=None,
        ntilts=None,
    ):
        # Expand a particle selection back to the concrete image ids belonging
        # to those particles, then optionally trim by allowed image set and
        # tilt count.
        particle_indices = normalize_indices(
            particle_indices,
            self.n_particles,
            name="particle_indices",
        )
        if particle_indices.size == 0:
            return np.array([], dtype=np.int32)

        image_indices = np.concatenate(
            [self._particle_to_images[int(particle_idx)] for particle_idx in particle_indices]
        ).astype(np.int32, copy=False)
        if ntilts is not None:
            if ntilts <= 0:
                return np.array([], dtype=np.int32)
            if self.tilt_numbers is None:
                raise ValueError("tilt_numbers are required when ntilts is specified")
            image_indices = image_indices[self.tilt_numbers[image_indices] < int(ntilts)]
        if allowed_images is not None:
            image_indices = filter_preserve_order(image_indices, allowed_images)
        return image_indices.astype(np.int32, copy=False)
