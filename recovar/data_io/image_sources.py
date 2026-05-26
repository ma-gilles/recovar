"""Image-source layer for cryo-EM / cryo-ET datasets.

This module cleanly separates image loading from metadata storage and from the
top-level dataset/view object. It provides:

- backend sources that load images from files, lazily or eagerly
- subset views that remap image/group indices without leaking that logic into
  the dataset class
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

import numpy as np

from recovar.data_io import image_backends
from recovar.data_io._index_utils import DatasetIndexLayout, normalize_indices


BatchMode = Literal["images", "groups"]


def _normalize_indices(values, n_total, *, name):
    return normalize_indices(values, n_total=int(n_total), name=name)


def _infer_backend_original_image_indices(backend):
    source = getattr(backend, "source", None)
    if source is not None and hasattr(source, "selection_indices"):
        return np.asarray(source.selection_indices, dtype=np.int32)

    raw_ind = getattr(backend, "ind", None)
    if raw_ind is None:
        return np.arange(_infer_backend_n_images(backend), dtype=np.int32)

    arr = np.asarray(raw_ind)
    if arr.dtype == bool:
        return np.flatnonzero(arr).astype(np.int32, copy=False)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("backend ind must be 1D")
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError("backend ind must contain integer indices or boolean mask")
    return arr.astype(np.int32, copy=False)


def _infer_backend_n_images(backend):
    for attr in ("n_images", "num_images", "N"):
        if hasattr(backend, attr):
            return int(getattr(backend, attr))
    for attr in ("ctfscalefactor", "tilt_numbers"):
        if hasattr(backend, attr):
            return int(len(getattr(backend, attr)))
    raise AttributeError("backend does not expose image count metadata")


def _build_backend_index_layout(backend, *, tilt_series):
    original_image_indices = _infer_backend_original_image_indices(backend)
    if not tilt_series or not hasattr(backend, "dataset_tilt_indices") or not hasattr(backend, "_particle_tilts"):
        return DatasetIndexLayout.from_image_indices(original_image_indices)

    original_group_indices = np.asarray(
        getattr(backend, "dataset_tilt_indices"),
        dtype=np.int32,
    )
    group_local_image_indices = tuple(
        np.asarray(local_images, dtype=np.int32) for local_images in getattr(backend, "_particle_tilts")
    )
    return DatasetIndexLayout.from_grouped_images(
        original_image_indices=original_image_indices,
        original_group_indices=original_group_indices,
        group_local_image_indices=group_local_image_indices,
    )


@dataclass(frozen=True)
class ImageSourceInfo:
    particles_file: Optional[str] = None
    datadir: Optional[str] = None
    lazy: bool = True
    tilt_series: bool = False
    tilt_series_ctf: Optional[str] = None
    dose_per_tilt: Optional[float] = None
    angle_per_tilt: Optional[float] = None
    sort_with_Bfac: bool = False
    strip_prefix: Optional[str] = None
    downsample_D: Optional[int] = None
    invert_data: bool = False


class ImageSource:
    """Abstract image-source interface used by the dataset layer."""

    info: ImageSourceInfo

    @property
    def already_prefetches(self) -> bool:
        """Whether batch iteration already performs background prefetching."""
        return False

    @property
    def index_layout(self) -> DatasetIndexLayout:
        raise NotImplementedError

    @property
    def tilt_series(self) -> bool:
        return bool(self.index_layout.grouped)

    @property
    def n_images(self) -> int:
        raise NotImplementedError

    @property
    def n_groups(self) -> int:
        return self.index_layout.n_groups

    @property
    def image_shape(self):
        raise NotImplementedError

    @property
    def image_size(self) -> int:
        raise NotImplementedError

    @property
    def grid_size(self) -> int:
        raise NotImplementedError

    @property
    def padding(self) -> int:
        raise NotImplementedError

    @property
    def unpadded_D(self) -> int:
        raise NotImplementedError

    @property
    def mask(self):
        raise NotImplementedError

    @property
    def data_multiplier(self):
        raise NotImplementedError

    @data_multiplier.setter
    def data_multiplier(self, value):
        raise NotImplementedError

    @property
    def mult(self):
        return self.data_multiplier

    @mult.setter
    def mult(self, value):
        self.data_multiplier = value

    @property
    def dataset_tilt_indices(self):
        if not self.tilt_series:
            return None
        return self.index_layout.original_group_indices

    @property
    def tilt_particles(self):
        if not self.tilt_series:
            return None
        return [
            np.asarray(local_images, dtype=np.int32) for local_images in self.index_layout.group_local_image_indices
        ]

    @property
    def particle_tilts(self):
        return self.tilt_particles

    def __getitem__(self, index):
        raise NotImplementedError

    def process_images(self, images, apply_image_mask=False):
        raise NotImplementedError

    def process_images_half(self, images, apply_image_mask=False):
        raise NotImplementedError("ImageSource subclasses must implement native process_images_half")

    def iter_batches(
        self,
        batch_size: int,
        *,
        batch_mode: BatchMode,
        subset_indices=None,
        num_workers: int = 0,
        **kwargs,
    ):
        raise NotImplementedError

    def subset(self, image_indices):
        return SubsetImageSource(self, image_indices)

    def resolve_group_indices_from_images(self, image_indices):
        return self.index_layout.local_group_indices_from_local_images(image_indices)

    def group_image_indices(self, group_indices):
        return self.index_layout.local_image_indices_from_local_groups(group_indices)


class BackendImageSource(ImageSource):
    """Image source backed by the low-level file/image backends."""

    def __init__(self, backend, *, info: ImageSourceInfo):
        self.backend = backend
        self.info = info
        self._index_layout = _build_backend_index_layout(
            backend,
            tilt_series=bool(self.info.tilt_series),
        )

    def __getattr__(self, name):
        return getattr(self.backend, name)

    @property
    def index_layout(self) -> DatasetIndexLayout:
        return self._index_layout

    @property
    def already_prefetches(self) -> bool:
        return True

    @property
    def n_images(self) -> int:
        return int(self.backend.n_images)

    @property
    def n_groups(self) -> int:
        return self._index_layout.n_groups

    @property
    def image_shape(self):
        return tuple(self.backend.image_shape)

    @property
    def image_size(self) -> int:
        return int(np.prod(self.image_shape))

    @property
    def grid_size(self) -> int:
        return int(self.backend.D)

    @property
    def padding(self) -> int:
        return int(self.backend.padding)

    @property
    def unpadded_D(self) -> int:
        return int(self.backend.unpadded_D)

    @property
    def mask(self):
        return self.backend.mask

    @property
    def data_multiplier(self):
        return self.backend.data_multiplier

    @data_multiplier.setter
    def data_multiplier(self, value):
        self.backend.data_multiplier = value
        self.info = replace(self.info, invert_data=bool(value < 0))

    @property
    def mult(self):
        return self.data_multiplier

    def __getitem__(self, index):
        return self.backend[index]

    def process_images(self, images, apply_image_mask=False):
        return self.backend.process_images(images, apply_image_mask=apply_image_mask)

    def process_images_half(self, images, apply_image_mask=False):
        if not hasattr(self.backend, "process_images_half"):
            raise ValueError("Image backend must implement native process_images_half")
        return self.backend.process_images_half(images, apply_image_mask=apply_image_mask)

    def iter_batches(
        self,
        batch_size: int,
        *,
        batch_mode: BatchMode,
        subset_indices=None,
        num_workers: int = 0,
        **kwargs,
    ):
        if batch_mode == "images":
            if subset_indices is None:
                generator = self.backend.get_image_generator(batch_size, num_workers=num_workers)
            else:
                subset_indices = _normalize_indices(subset_indices, self.n_images, name="subset_indices")
                generator = self.backend.get_image_subset_generator(batch_size, subset_indices, num_workers=num_workers)
        else:
            if subset_indices is None:
                generator = self.backend.get_dataset_generator(batch_size, num_workers=num_workers, **kwargs)
            else:
                subset_indices = _normalize_indices(subset_indices, self.n_groups, name="subset_indices")
                generator = self.backend.get_dataset_subset_generator(
                    batch_size, subset_indices, num_workers=num_workers, **kwargs
                )
        yield from generator


class SubsetImageSource(ImageSource):
    """Image-source view over a subset of images."""

    def __init__(self, parent: ImageSource, image_indices):
        self.parent = parent
        self._parent_local_image_indices = _normalize_indices(
            image_indices, parent.n_images, name="image_indices"
        ).astype(np.int32, copy=False)
        self._index_layout = parent.index_layout.subset(self._parent_local_image_indices)
        self._parent_local_group_indices = parent.index_layout.local_group_indices_from_original(
            self._index_layout.original_group_indices,
            allow_missing=False,
        ).astype(np.int32, copy=False)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    @property
    def index_layout(self) -> DatasetIndexLayout:
        return self._index_layout

    @property
    def already_prefetches(self) -> bool:
        return self.parent.already_prefetches

    @property
    def info(self) -> ImageSourceInfo:
        return self.parent.info

    @property
    def n_images(self) -> int:
        return self._index_layout.n_images

    @property
    def n_groups(self) -> int:
        return self._index_layout.n_groups

    @property
    def image_shape(self):
        return self.parent.image_shape

    @property
    def image_size(self) -> int:
        return self.parent.image_size

    @property
    def grid_size(self) -> int:
        return self.parent.grid_size

    @property
    def padding(self) -> int:
        return self.parent.padding

    @property
    def unpadded_D(self) -> int:
        return self.parent.unpadded_D

    @property
    def mask(self):
        return self.parent.mask

    @property
    def data_multiplier(self):
        return self.parent.data_multiplier

    @data_multiplier.setter
    def data_multiplier(self, value):
        self.parent.data_multiplier = value

    @property
    def mult(self):
        return self.data_multiplier

    def __getitem__(self, index):
        if self.tilt_series:
            return self.parent[self._parent_local_group_indices[int(index)]]
        return self.parent[self._parent_local_image_indices[int(index)]]

    def process_images(self, images, apply_image_mask=False):
        return self.parent.process_images(images, apply_image_mask=apply_image_mask)

    def process_images_half(self, images, apply_image_mask=False):
        return self.parent.process_images_half(images, apply_image_mask=apply_image_mask)

    def _remap_parent_images(self, parent_local_image_indices):
        parent_original_image_indices = self.parent.index_layout.original_image_indices_for_local(
            parent_local_image_indices
        )
        return self._index_layout.local_image_indices_from_original(
            parent_original_image_indices,
            allow_missing=True,
        )

    def _remap_parent_groups(self, parent_local_group_indices):
        parent_original_group_indices = self.parent.index_layout.original_group_indices_for_local(
            parent_local_group_indices
        )
        return self._index_layout.local_group_indices_from_original(
            parent_original_group_indices,
            allow_missing=True,
        )

    def iter_batches(
        self,
        batch_size: int,
        *,
        batch_mode: BatchMode,
        subset_indices=None,
        num_workers: int = 0,
        **kwargs,
    ):
        if batch_mode == "images":
            if subset_indices is None:
                parent_subset = self._parent_local_image_indices
            else:
                subset_indices = _normalize_indices(subset_indices, self.n_images, name="subset_indices")
                parent_subset = self._parent_local_image_indices[subset_indices]

            for images, _parent_particle_indices, parent_image_indices in self.parent.iter_batches(
                batch_size,
                batch_mode="images",
                subset_indices=parent_subset,
                num_workers=num_workers,
                **kwargs,
            ):
                local_image_indices = self._remap_parent_images(np.asarray(parent_image_indices, dtype=np.int32))
                yield (
                    images,
                    local_image_indices,
                    local_image_indices,
                )
            return

        if subset_indices is None:
            parent_group_subset = self._parent_local_group_indices
        else:
            subset_indices = _normalize_indices(subset_indices, self.n_groups, name="subset_indices")
            parent_group_subset = self._parent_local_group_indices[subset_indices]

        for images, parent_group_indices, parent_image_indices in self.parent.iter_batches(
            batch_size,
            batch_mode="groups",
            subset_indices=parent_group_subset,
            num_workers=num_workers,
            **kwargs,
        ):
            parent_image_indices = np.asarray(parent_image_indices, dtype=np.int32)
            local_image_indices = self._remap_parent_images(parent_image_indices)
            keep_mask = local_image_indices >= 0
            if not np.all(keep_mask):
                images = np.asarray(images)[keep_mask]
                parent_group_indices = np.asarray(parent_group_indices)[keep_mask]
                local_image_indices = local_image_indices[keep_mask]
                if local_image_indices.size == 0:
                    continue

            local_group_indices = self._remap_parent_groups(parent_group_indices)
            yield (
                images,
                local_group_indices.astype(np.int32, copy=False),
                local_image_indices.astype(np.int32, copy=False),
            )

    def resolve_group_indices_from_images(self, image_indices):
        return self._index_layout.local_group_indices_from_local_images(image_indices)

    def group_image_indices(self, group_indices):
        return self._index_layout.local_image_indices_from_local_groups(group_indices)


def create_image_source(
    particles_file,
    *,
    ind=None,
    lazy=True,
    tilt_series=False,
    tilt_series_ctf=None,
    uninvert_data=False,
    datadir=None,
    padding=0,
    strip_prefix=None,
    downsample_D=None,
    sort_with_Bfac=False,
):
    if tilt_series:
        tilt_file_option = "relion5" if tilt_series_ctf == "relion5" else "warp"
        backend = image_backends.TiltSeriesDataset(
            particles_file,
            ind=ind,
            lazy=lazy,
            datadir=datadir,
            invert_data=uninvert_data,
            tilt_file_option=tilt_file_option,
            strip_prefix=strip_prefix,
            sort_with_Bfac=sort_with_Bfac,
        )
    else:
        backend = image_backends.ParticleImageDataset(
            particles_file,
            ind=ind,
            lazy=lazy,
            datadir=datadir,
            padding=padding,
            invert_data=uninvert_data,
            strip_prefix=strip_prefix,
            downsample_D=downsample_D,
        )

    info = ImageSourceInfo(
        particles_file=particles_file,
        datadir=datadir,
        lazy=lazy,
        tilt_series=tilt_series,
        tilt_series_ctf=tilt_series_ctf,
        dose_per_tilt=getattr(backend, "dose_per_tilt", None),
        angle_per_tilt=getattr(backend, "angle_per_tilt", None),
        sort_with_Bfac=sort_with_Bfac,
        strip_prefix=strip_prefix,
        downsample_D=downsample_D,
        invert_data=bool(uninvert_data),
    )
    return BackendImageSource(backend, info=info)
