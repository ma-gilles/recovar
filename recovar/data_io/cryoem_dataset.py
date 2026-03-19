"""Top-level cryo-EM / cryo-ET dataset assembly and batch iteration.

Architecture:
- ``image_sources.py`` owns raw image loading, lazy/eager access, and subset views
- ``image_metadata.py`` owns poses and CTF metadata only
- ``CryoEMDataset`` coordinates both layers and exposes the single explicit
  batch iterator used by downstream code
"""

from __future__ import annotations

import logging
import pickle
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core
from recovar.core import mask
from recovar.data_io.image_sources import (
    BackendImageSource,
    ImageSource,
    ImageSourceInfo,
    create_image_source,
)
from recovar.data_io.image_metadata import MetadataStore, Metadata
from recovar.output import plot_utils

logger = logging.getLogger(__name__)

from recovar.data_io._index_utils import (
    DatasetIndexLayout,
    deduplicate_preserve_order,
    normalize_indices,
)
from recovar.data_io.image_loader import ImageLoader


_SENTINEL = object()


def _prefetch_iter(iterable):
    """1-lookahead prefetch: loads next batch while caller processes current."""
    from concurrent.futures import ThreadPoolExecutor
    it = iter(iterable)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(next, it, _SENTINEL)
        while True:
            result = future.result()
            if result is _SENTINEL:
                return
            future = pool.submit(next, it, _SENTINEL)
            yield result


def get_num_images_in_dataset(mrc_path, datadir=None, strip_prefix=None):
    return ImageLoader.image_count(mrc_path, datadir=datadir, strip_prefix=strip_prefix)


def _load_index_like(value):
    if value is None:
        return None
    if isinstance(value, (np.ndarray, list, tuple)):
        return value
    with open(value, "rb") as f:
        return pickle.load(f)


def _normalize_image_indices(values, n_images_total=None, name="indices"):
    """Normalize image indices using the canonical implementation."""
    if values is None:
        raise ValueError(f"{name} must not be None")
    if n_images_total is None:
        # Legacy compatibility: when n_total is unknown, skip range check
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
    return normalize_indices(values, n_total=int(n_images_total), name=name)


def _deduplicate_preserve_order(values, name):
    values = np.asarray(values)
    if values.size == 0:
        return values.astype(np.int32, copy=False)
    deduped = deduplicate_preserve_order(values, name=name)
    dropped = int(values.size - deduped.size)
    if dropped > 0:
        logger.warning("Dropping %d duplicate entries from %s.", dropped, name)
    return deduped


class _ImageIndexRemappedNoiseAdapter:
    """Wrap a noise model with an explicit child-local -> parent-local remap."""

    def __init__(self, noise_model, parent_image_indices):
        self._noise = noise_model
        self._parent_image_indices = np.asarray(parent_image_indices, dtype=np.int32)

    def get(self, indices):
        return self._noise.get(self._parent_image_indices[np.asarray(indices)])

    def get_half(self, indices):
        return self._noise.get_half(self._parent_image_indices[np.asarray(indices)])

    @property
    def dose_indices(self):
        dose_indices = getattr(self._noise, "dose_indices", None)
        if dose_indices is None:
            raise AttributeError("wrapped noise model has no dose_indices")
        return np.asarray(dose_indices)[self._parent_image_indices]

    @property
    def __class__(self):
        return type(self._noise)

    def __getattr__(self, name):
        return getattr(self._noise, name)


def _coerce_image_source(image_source, *, tilt_series_flag):
    if image_source is None or isinstance(image_source, ImageSource):
        return image_source
    return BackendImageSource(
        image_source,
        info=ImageSourceInfo(tilt_series=tilt_series_flag),
    )


class CryoEMDataset:
    """Core dataset class for cryo-EM heterogeneity analysis.

    Wraps particle images with per-image metadata (poses, CTF parameters) and
    provides geometry helpers for 3-D reconstruction and embedding.

    For half-set reconstructions, two ``CryoEMDataset`` instances are typically
    managed via ``halfset_indices`` on the dataset.

    Attributes:
        grid_size: Side length of the image (and default 3-D reconstruction grid).
        voxel_size: Pixel / voxel size in Angstroms.
        n_images: Number of particle images in this dataset.
        image_source: Underlying image-loading layer (``None`` for simulation).
        tilt_series_flag: ``True`` when the dataset represents tilt-series data.
    """

    __slots__ = (
        # Grid geometry
        'voxel_size', 'grid_size', 'volume_shape', 'volume_size',
        # Image stack and image geometry
        '_image_source', 'image_shape', 'image_size', 'n_images', 'padding',
        # Processing flags
        'tilt_series_flag', 'premultiplied_ctf', 'n_units',
        '_ctf_evaluator', 'hpad', 'volume_mask_threshold',
        # Data types
        'dtype', 'dtype_real',
        # Per-image metadata (private — access via iter_batches/metadata helpers)
        '_metadata',
        # Explicit local/original image/group index mapping
        '_index_layout',
        # Mutable state
        'dataset_indices', 'noise',
        # Half-set views (index arrays into this dataset's ordering)
        'halfset_indices',
        # Loader paths (for reloading independent half-datasets)
        'particles_file', 'poses_file', 'ctf_file', 'datadir',
    )

    def __init__(
        self,
        image_source,
        voxel_size: float,
        metadata: MetadataStore,
        ctf_evaluator=None,
        dtype: type = np.complex64,
        dataset_indices: Optional[NDArray[np.integer]] = None,
        grid_size: Optional[int] = None,
        tilt_series_flag: bool = False,
        premultiplied_ctf: bool = False,
    ) -> None:
        image_source = _coerce_image_source(
            image_source,
            tilt_series_flag=tilt_series_flag,
        )
        # --- Grid geometry ---
        if image_source is not None:
            grid_size = image_source.grid_size
        elif grid_size is None:
            raise ValueError("Must specify grid_size if image_source is None")
        elif metadata.n_images <= 0:
            raise ValueError("metadata must contain at least one image when image_source is None")

        if image_source is not None and metadata.n_images != image_source.n_images:
            raise ValueError(
                "metadata/image-source size mismatch: "
                f"{metadata.n_images} metadata rows vs {image_source.n_images} images"
            )

        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.volume_shape = (grid_size, grid_size, grid_size)
        self.volume_size = grid_size ** 3

        # --- Image source and image geometry ---
        self._image_source = image_source
        if image_source is None:
            self.image_shape = (grid_size, grid_size)
            self.image_size = grid_size ** 2
            self.n_images = metadata.n_images
            self.padding = 0
        else:
            self.image_shape = tuple(image_source.image_shape)
            self.image_size = int(np.prod(image_source.image_shape))
            self.n_images = image_source.n_images
            self.padding = image_source.padding

        # --- Processing flags ---
        self.tilt_series_flag = bool(tilt_series_flag or (image_source is not None and image_source.tilt_series))
        self.premultiplied_ctf = premultiplied_ctf
        self.n_units = image_source.n_groups if self.tilt_series_flag and image_source is not None else self.n_images
        if ctf_evaluator is None:
            self._ctf_evaluator = core.CTFEvaluator()
        else:
            self._ctf_evaluator = core.as_ctf_evaluator(ctf_evaluator)
        self.hpad = self.padding // 2
        # Heuristic mask threshold scaled by grid size
        self.volume_mask_threshold = 4 * self.grid_size / 128

        # --- Data types ---
        self.dtype = dtype
        self.dtype_real = dtype(0).real.dtype

        # --- Per-image metadata ---
        self._metadata = metadata

        # --- Explicit index mapping ---
        if image_source is not None:
            source_layout = image_source.index_layout
            if dataset_indices is None:
                dataset_indices = source_layout.original_image_indices
                self._index_layout = source_layout
            else:
                dataset_indices = _normalize_image_indices(
                    dataset_indices,
                    n_images_total=None,
                    name="dataset_indices",
                )
                dataset_indices = np.asarray(dataset_indices, dtype=np.int32)
                if source_layout.n_images != dataset_indices.size:
                    raise ValueError(
                        "dataset_indices must have the same length as the image_source"
                    )
                if np.array_equal(
                    dataset_indices,
                    np.asarray(source_layout.original_image_indices, dtype=np.int32),
                ):
                    self._index_layout = source_layout
                elif source_layout.grouped:
                    self._index_layout = DatasetIndexLayout.from_grouped_images(
                        original_image_indices=dataset_indices,
                        original_group_indices=source_layout.original_group_indices,
                        group_local_image_indices=source_layout.group_local_image_indices,
                    )
                else:
                    self._index_layout = DatasetIndexLayout.from_image_indices(dataset_indices)
            if hasattr(image_source, "_index_layout"):
                image_source._index_layout = self._index_layout
        else:
            if dataset_indices is None:
                dataset_indices = np.arange(metadata.n_images, dtype=np.int32)
            else:
                dataset_indices = _normalize_image_indices(
                    dataset_indices,
                    n_images_total=None,
                    name="dataset_indices",
                )
            self._index_layout = DatasetIndexLayout.from_image_indices(dataset_indices)

        # --- Mutable state ---
        self.dataset_indices = np.asarray(dataset_indices, dtype=np.int32)
        self.noise = None
        self.halfset_indices = None
        self.particles_file = None
        self.poses_file = None
        self.ctf_file = None
        self.datadir = None

    # --- Metadata access (public API) ---

    @property
    def metadata(self):
        """The per-image metadata store."""
        return self._metadata

    @property
    def image_source(self):
        """Image-loading layer for this dataset."""
        return self._image_source

    @property
    def index_layout(self):
        """Explicit local/original image/group mapping for this dataset."""
        return self._index_layout

    @property
    def original_image_indices(self):
        """Original source-file image index for each local image."""
        return self._index_layout.original_image_indices

    @property
    def original_group_indices(self):
        """Original source-file group index for each local group."""
        return self._index_layout.original_group_indices

    def original_image_indices_from_local(self, local_image_indices=None):
        return self._index_layout.original_image_indices_for_local(local_image_indices)

    def local_image_indices_from_original(self, original_image_indices, *, allow_missing=False):
        return self._index_layout.local_image_indices_from_original(
            original_image_indices,
            allow_missing=allow_missing,
        )

    def original_group_indices_from_local(self, local_group_indices=None):
        return self._index_layout.original_group_indices_for_local(local_group_indices)

    def local_group_indices_from_original(self, original_group_indices, *, allow_missing=False):
        return self._index_layout.local_group_indices_from_original(
            original_group_indices,
            allow_missing=allow_missing,
        )

    # --- Convenience delegation to Metadata ---

    @property
    def rotation_matrices(self):
        """Per-image rotation matrices (read-only view)."""
        return self._metadata._rotation_matrices

    @rotation_matrices.setter
    def rotation_matrices(self, value):
        self._metadata._rotation_matrices = np.asarray(value, dtype=self._metadata.rotation_dtype)

    @property
    def translations(self):
        """Per-image translations (read-only view)."""
        return self._metadata._translations

    @translations.setter
    def translations(self, value):
        self._metadata._translations = np.asarray(value, dtype=self._metadata.real_dtype)

    @property
    def CTF_params(self):
        """Per-image CTF parameters (read-only view)."""
        return self._metadata._ctf_params

    @CTF_params.setter
    def CTF_params(self, value):
        self._metadata._ctf_params = np.asarray(value, dtype=self._metadata.ctf_dtype)

    def get_ctf_column(self, col):
        """Read a single CTF parameter column for all images."""
        return self._metadata.get_ctf_column(col)

    def get_ctf_params_copy(self):
        """Return a mutable copy of the full CTF parameter array."""
        return self._metadata.get_ctf_params_copy()

    def update_poses(self, rots, trans):
        """Replace all poses."""
        self._metadata.set_poses(rots, trans)

    def update_ctf(self, ctf_params):
        """Replace all CTF parameters."""
        self._metadata.set_ctf(ctf_params)

    def __repr__(self) -> str:
        return (
            f"CryoEMDataset(n_images={self.n_images}, grid_size={self.grid_size}, "
            f"voxel_size={self.voxel_size:.4f}, tilt_series={self.tilt_series_flag})"
        )

    # --- Delegating properties (avoid leaking image-source internals) ---

    def process_images(self, images, apply_image_mask=False):
        """Apply windowing + full DFT preprocessing to raw images."""
        return self.image_source.process_images(images, apply_image_mask=apply_image_mask)

    def process_images_half(self, images, apply_image_mask=False):
        """Apply windowing + rfft2 preprocessing → half-spectrum output."""
        return self.image_source.process_images_half(images, apply_image_mask=apply_image_mask)

    @property
    def image_mask(self):
        """Circular window mask from the image stack."""
        return self.image_source.mask if self.image_source is not None else None

    @property
    def data_multiplier(self):
        """Sign multiplier for data inversion (±1)."""
        return getattr(self.image_source, 'mult', 1)

    @data_multiplier.setter
    def data_multiplier(self, value):
        if self.image_source is not None:
            self.image_source.mult = value

    @property
    def dataset_tilt_indices(self):
        """Per-particle tilt index lists (tilt-series only)."""
        if not self.tilt_series_flag:
            return None
        return self._index_layout.original_group_indices

    @property
    def tilt_particles(self):
        """List of per-particle tilt index arrays (tilt-series only)."""
        if not self.tilt_series_flag:
            return None
        return [np.asarray(local_images, dtype=np.int32) for local_images in self._index_layout.group_local_image_indices]

    def get_noise_variance(self, indices):
        if self.noise is None:
            return None
        return self.noise.get(indices)

    def subset(self, indices):
        """Return a new CryoEMDataset containing only the images at *indices*.

        The returned dataset uses an ``ImageSource`` subset view, so the
        subset/remap logic stays inside the image-loading layer rather than
        being duplicated in the dataset class.
        """
        indices = _normalize_image_indices(indices, n_images_total=self.n_images, name="indices")

        composed_indices = self._index_layout.original_image_indices_for_local(indices)

        sub = CryoEMDataset(
            image_source=None if self.image_source is None else self.image_source.subset(indices),
            voxel_size=self.voxel_size,
            metadata=self._metadata.subset(indices),
            ctf_evaluator=self.ctf_evaluator,
            grid_size=self.grid_size,
            tilt_series_flag=self.tilt_series_flag,
            premultiplied_ctf=self.premultiplied_ctf,
            dataset_indices=composed_indices,
        )
        if self.noise is not None:
            sub.noise = _ImageIndexRemappedNoiseAdapter(self.noise, indices)
        return sub

    def reload_from_original_images(self, original_image_indices, *, lazy=None):
        """Reload a dataset view from original file image indices.

        This is used only when an independent file-backed dataset is required.
        The input indices are always in original file ordering, never this
        dataset's local ordering.
        """
        if self.image_source is None:
            raise ValueError("Cannot reload dataset without an image source")
        if self.particles_file is None:
            raise ValueError("Cannot reload dataset without stored loader paths")

        original_image_indices = _normalize_image_indices(
            original_image_indices,
            n_images_total=None,
            name="original_image_indices",
        )

        info = self.image_source.info
        if lazy is None:
            lazy = info.lazy

        reloaded = load_dataset(
            self.particles_file,
            self.poses_file,
            self.ctf_file,
            datadir=self.datadir,
            ind=original_image_indices,
            lazy=lazy,
            padding=self.padding,
            uninvert_data=info.invert_data,
            tilt_series=self.tilt_series_flag,
            tilt_series_ctf=info.tilt_series_ctf,
            dose_per_tilt=info.dose_per_tilt,
            angle_per_tilt=info.angle_per_tilt,
            premultiplied_ctf=self.premultiplied_ctf,
            strip_prefix=info.strip_prefix,
            sort_with_Bfac=info.sort_with_Bfac,
            downsample_D=info.downsample_D,
        )

        parent_local_image_indices = self.local_image_indices_from_original(
            original_image_indices,
            allow_missing=False,
        )
        reloaded.update_poses(
            self.rotation_matrices[parent_local_image_indices],
            self.translations[parent_local_image_indices],
        )
        reloaded.update_ctf(self.CTF_params[parent_local_image_indices])

        if self.noise is not None:
            reloaded.noise = _ImageIndexRemappedNoiseAdapter(
                self.noise,
                parent_local_image_indices,
            )
        return reloaded

    def get_halfset_dataset(self, halfset_id: int, *, independent=False, lazy=None):
        """Return one halfset as either a lightweight view or independent reload."""
        if independent:
            return self.reload_from_original_images(
                self.halfset_original_image_indices(halfset_id),
                lazy=lazy,
            )
        return self.subset(self.halfset_local_image_indices(halfset_id))

    @property
    def ctf_evaluator(self):
        """The :class:`~recovar.core.ctf.CTFEvaluator` for this dataset."""
        return self._ctf_evaluator

    def get_valid_frequency_indices(self,rad = None):
        rad = self.grid_size//2 -1 if rad is None else rad
        return np.array(self.get_volume_radial_mask(rad))


    #### All functions below are only just for plotting/debugging

    def compute_CTF(self, CTF_params):
        return self.ctf_evaluator(CTF_params, self.image_shape, self.voxel_size)

    def get_CTF(self, indices):
        return self.compute_CTF(self.CTF_params[indices])

    def get_volume_radial_mask(self, radius = None):
        return mask.get_radial_mask(self.volume_shape, radius = radius).reshape(-1)


    def get_image_radial_mask(self, radius = None):        
        return mask.get_radial_mask(self.image_shape, radius = radius).reshape(-1)

    def get_proj(self, X, to_real = np.real, axis = 0, hide_padding = True):
        im = to_real(fourier_transform_utils.get_idft2(jnp.take(X.reshape(self.volume_shape), self.grid_size//2, axis = axis)))
        if hide_padding:
            im = im[self.hpad:self.image_source.unpadded_D + self.hpad,self.hpad:self.image_source.unpadded_D + self.hpad]
        return im


    def get_slice(self, X, to_real_fn = np.abs, axis = 0):
        # zero_th freq
        z_freq = self.grid_size//2 +1
        im = to_real_fn(jnp.take(X.reshape(self.volume_shape), z_freq, axis = axis))
        return im

    def get_slice_real(self, X, to_real_fn = np.real, axis = 0):
        im = to_real_fn(fourier_transform_utils.get_idft3(X.reshape(self.volume_shape)))
        im2 = jnp.take(im, self.grid_size//2, axis = axis)
        return to_real_fn(im2)

    def get_image(self, i , tilt_idx = None):
        if self.tilt_series_flag:
            if tilt_idx is None:
                raise ValueError("Tilt index must be specified for tilt series")

        if tilt_idx is None:
            image = self.image_source.__getitem__(i)[0]
        else:
            image = self.image_source.__getitem__(i)[0][tilt_idx][None]

        processed_image = self.image_source.process_images(image)
        return processed_image.reshape(self.image_shape)

    def get_CTF_image(self, i ):
        return self.get_CTF(np.array([i])).reshape(self.image_shape)

    def get_image_real(self,i, tilt_idx = None, to_real= np.real, hide_padding = True):
        hpad= self.image_source.padding//2
        if hide_padding:
            return to_real(fourier_transform_utils.get_idft2(self.get_image(i,tilt_idx))[hpad:self.image_shape[0]-hpad,hpad:self.image_shape[1]-hpad])
        else:
            return to_real(fourier_transform_utils.get_idft2(self.get_image(i,tilt_idx)))


    def get_denoised_image(self,i, tilt_idx=None, to_real= np.real, hide_padding = True, weiner_param =1):
        batch_image_ind = np.array([i])
        if self.tilt_series_flag:
            if tilt_idx is None:
                raise ValueError("Tilt index must be specified for tilt series")

        if tilt_idx is not None:
            images, _, image_ind = self.image_source.__getitem__(i)
            images = images[tilt_idx][None]
            CTFs = self.ctf_evaluator(self.CTF_params[image_ind[tilt_idx]][None], self.image_shape, self.voxel_size) # Compute CTF
        else:
            images, _, _ = self.image_source.__getitem__(i)
            CTFs = self.ctf_evaluator(self.CTF_params[i][None], self.image_shape, self.voxel_size) # Compute CTF
        images = self.image_source.process_images(images) # Compute DFT, masking
        images = (CTFs / (CTFs**2 + weiner_param)) * images  # CTF correction
        images = images.reshape(self.image_shape)
        return to_real(fourier_transform_utils.get_idft2(images))


    def plot_FSC(self, image1 = None, image2 = None, filename = None, threshold = 0.5, curve = None, ax = None):
        score = plot_utils.plot_fsc_new(image1, image2, self.volume_shape, self.voxel_size,  curve = curve, ax = ax, threshold = threshold, filename = filename)
        return score
    
    def get_image_mask(self, indices, mask, binary = True, soften = 5):
        indices = np.asarray(indices, dtype=int)
        from recovar.heterogeneity import covariance_core # Not sure I want this depency to exist... Could make some circular imports
        mask = covariance_core.get_per_image_tight_mask(mask, self.rotation_matrices[indices], self.image_source.mask, self.volume_mask_threshold, self.image_shape, self.volume_shape, self.grid_size, self.padding, disc_type = 'linear_interp',  binary = binary, soften = soften)
        mask_ft = fourier_transform_utils.get_dft2(mask).reshape(mask.shape[0], -1)
        # Usually images are translated, here we translate back.
        batch = core.translate_images(mask_ft, -self.translations[indices].astype(int) , self.image_shape)
        mask2 = fourier_transform_utils.get_idft2(batch.reshape(-1, *self.image_shape))
        return mask2.real


    def get_predicted_image(self, indices, volume, skip_ctf = False, spatial = True):
        """Get predicted images for given indices using forward model.
        
        Args:
            indices: Array of indices to predict images for
            volume: Volume to use for prediction
            skip_ctf: Whether to skip CTF application
            spatial: Whether to return images in real space (True) or Fourier space (False)
            
        Returns:
            Predicted images in real space if spatial=True, otherwise in Fourier space
        """
        from recovar.core.configs import ForwardModelConfig
        import recovar.core.forward as core_forward
        config = ForwardModelConfig.from_dataset(self, disc_type='linear_interp')
        predicted_images = core_forward.forward_model(
            config, volume, self.CTF_params[indices],
            self.rotation_matrices[indices], skip_ctf=skip_ctf,
        )
        if spatial:
            predicted_images = fourier_transform_utils.get_idft2(predicted_images.reshape(-1, *self.image_shape)).real
        return predicted_images


    def set_radial_noise_model(self, noise_variance):
        from recovar.reconstruction import noise
        self.noise = noise.RadialNoiseModel(noise_variance, image_shape = self.image_shape)

    def set_variable_radial_noise_model(self, noise_variance_radials):
        from recovar.reconstruction import noise
        _, dose_indices = jnp.unique(self.CTF_params[:,core.CTFParamIndex.DOSE], return_inverse=True)
        # If noise_variance_radials is 1D (single radial profile), broadcast
        # to 2D (one row per dose level) so VariableRadialNoiseModel can index
        # by dose_indices.
        if noise_variance_radials is not None and np.ndim(noise_variance_radials) == 1:
            n_doses = int(jnp.max(dose_indices)) + 1
            noise_variance_radials = np.tile(noise_variance_radials, (n_doses, 1))
        self.noise = noise.VariableRadialNoiseModel(noise_variance_radials, dose_indices, image_shape = self.image_shape)

    # --- Half-set iteration (new v4 API) ---

    def n_halfset_images(self, halfset_id: int) -> int:
        """Number of images in a given halfset."""
        if self.halfset_indices is None:
            raise ValueError("halfset_indices not set on this dataset")
        return len(self.halfset_indices[halfset_id])

    def halfset_local_image_indices(self, halfset_id: int):
        if self.halfset_indices is None:
            raise ValueError("halfset_indices not set on this dataset")
        return np.asarray(self.halfset_indices[halfset_id], dtype=np.int32)

    def halfset_original_image_indices(self, halfset_id: int):
        return self._index_layout.original_image_indices_for_local(
            self.halfset_local_image_indices(halfset_id)
        )

    def halfset_local_group_indices(self, halfset_id: int):
        return self._index_layout.local_group_indices_from_local_images(
            self.halfset_local_image_indices(halfset_id)
        )

    def halfset_original_group_indices(self, halfset_id: int):
        return self._index_layout.original_group_indices_from_local_images(
            self.halfset_local_image_indices(halfset_id)
        )

    def get_particle_halfset_indices(self):
        """Per-half canonical particle indices for tilt-series datasets.

        For SPA datasets, this simply returns ``halfset_indices`` (images
        and particles are 1-to-1).  For tilt-series, it maps each half's
        image indices through the image→particle mapping and returns the
        unique canonical (``dataset_tilt_indices``) particle ids per half.
        """
        if self.halfset_indices is None:
            raise ValueError("halfset_indices not set on this dataset")
        return [self.halfset_original_group_indices(halfset_id) for halfset_id in range(2)]

    def split_halfset_array(self, arr, per_particle=False):
        """Split a concatenated halfset-ordered array into [half0, half1].

        Parameters
        ----------
        per_particle : bool
            If True **and** this is a tilt-series dataset, split at the
            particle boundary instead of the image boundary.
        """
        if per_particle and self.tilt_series_flag:
            n0 = len(self.halfset_original_group_indices(0))
        else:
            n0 = self.n_halfset_images(0)
        return [arr[:n0], arr[n0:]]

    def _resolve_iteration_subset(self, *, halfset_id=None, indices=None, by_image=True):
        """Resolve halfset or subset selection for iteration."""
        if halfset_id is not None and indices is not None:
            raise ValueError("Cannot specify both halfset_id and indices")
        if halfset_id is None:
            return indices

        if self.halfset_indices is None:
            raise ValueError("halfset_indices not set on this dataset")

        halfset_image_indices = self.halfset_indices[halfset_id]
        if by_image or not self.tilt_series_flag:
            return halfset_image_indices

        return self.halfset_local_group_indices(halfset_id)

    def _iter_explicit_batches(
        self,
        *,
        batch_size,
        batch_mode,
        index_subset,
        noise_model,
        noise_half,
        noise_by_particle,
    ):
        """Yield explicit batch fields from the image source and metadata store."""
        if self.image_source is None:
            raise ValueError("Cannot iterate batches without an image source")

        use_particle_noise = noise_by_particle and batch_mode == "groups"
        generator = self.image_source.iter_batches(
            batch_size=batch_size,
            batch_mode=batch_mode,
            subset_indices=index_subset,
        )

        for images, particle_indices, image_indices in generator:
            image_indices = np.asarray(image_indices, dtype=np.int32).reshape(-1)
            particle_indices = np.asarray(particle_indices)
            rotation_matrices, translations, ctf_params = self.metadata.get_batch(image_indices)

            if noise_model is None:
                noise_variance = None
            else:
                noise_indices = particle_indices if use_particle_noise else image_indices
                noise_getter = noise_model.get_half if noise_half else noise_model.get
                noise_variance = noise_getter(noise_indices)

            yield (
                images,
                rotation_matrices,
                translations,
                ctf_params,
                noise_variance,
                particle_indices,
                image_indices,
            )

    def iter_batches(self, batch_size, *, halfset_id=None, indices=None,
                     noise_model=None, noise_half=True, noise_by_particle=False,
                     by_image=True, prefetch=True):
        """Iterate over dataset batches, yielding explicit batch fields.

        Parameters
        ----------
        batch_size : int
        halfset_id : int, optional
            Halfset index (0 or 1). Mutually exclusive with *indices*.
        indices : array-like, optional
            Iterate over this subset of image indices.
        noise_model : optional
            Noise model used to populate the yielded ``noise_variance`` field.
        noise_half : bool
            Use half-spectrum noise (default True for mean reconstruction).
        noise_by_particle : bool
            Index noise by particle group (for covariance path).
        by_image : bool
            True = flat per-image iteration; False = particle-grouped (tilt).
        prefetch : bool
            Enable 1-lookahead prefetch buffer (default True).

        Yields
        ------
        tuple
            ``(images, rotation_matrices, translations, ctf_params,
            noise_variance, particle_indices, image_indices)``
        """
        resolved_indices = self._resolve_iteration_subset(
            halfset_id=halfset_id,
            indices=indices,
            by_image=by_image,
        )
        inner = self._iter_explicit_batches(
            batch_size=batch_size,
            batch_mode="images" if by_image else "groups",
            index_subset=resolved_indices,
            noise_model=noise_model,
            noise_half=noise_half,
            noise_by_particle=noise_by_particle,
        )
        if prefetch and not self.image_source.already_prefetches:
            return _prefetch_iter(inner)
        return inner

    def set_contrasts(self, contrasts: NDArray):
        """Multiply per-image CTF contrast column by *contrasts*.

        *contrasts* must be in this dataset's ordering (original ordering
        for a full dataset, or local ordering for a subset).
        For tilt-series with per-particle contrasts (len < n_images),
        each particle's tilt images share a single contrast value.
        """
        if self.tilt_series_flag and contrasts.shape[0] != self.n_images:
            # Per-particle contrast in tilt series
            for i, p in enumerate(self.tilt_particles):
                self.CTF_params[p, core.CTFParamIndex.CONTRAST] *= contrasts[i]
        else:
            self.CTF_params[:, core.CTFParamIndex.CONTRAST] *= contrasts

    def set_noise(self, noise_variance):
        """Set the radial noise model for this dataset.

        If the dataset already has a ``VariableRadialNoiseModel``, updates
        it; otherwise sets a ``RadialNoiseModel``.
        """
        from recovar.reconstruction import noise as noise_mod
        if self.noise is not None and isinstance(self.noise, noise_mod.VariableRadialNoiseModel):
            self.set_variable_radial_noise_model(noise_variance)
        else:
            self.set_radial_noise_model(noise_variance)



def _normalize_dataset_indices(ind_value, n_total):
    """Normalize dataset selection indices (integer array or boolean mask)."""
    if ind_value is None:
        return None
    return _normalize_image_indices(ind_value, n_images_total=n_total, name="ind")


def _create_image_source(particles_file, ind, lazy, tilt_series, tilt_series_ctf,
                         uninvert_data, datadir, padding, strip_prefix, downsample_D,
                         sort_with_Bfac=False):
    """Create the image-loading layer for this dataset."""
    return create_image_source(
        particles_file,
        ind=ind,
        lazy=lazy,
        tilt_series=tilt_series,
        tilt_series_ctf=tilt_series_ctf,
        uninvert_data=uninvert_data,
        datadir=datadir,
        padding=padding,
        strip_prefix=strip_prefix,
        downsample_D=downsample_D,
        sort_with_Bfac=sort_with_Bfac,
    )


def _load_ctf_params(particles_file, ctf_file, D, ind, n_images):
    """Load CTF parameters from pickle or auto-extract from STAR/CS.

    Returns ``(ctf_params, dataset_indices)`` where *ctf_params* includes
    appended bfactor (=0) and contrast (=1) columns.
    """
    from recovar.data_io import load_utils
    if ctf_file is not None and ctf_file.endswith('.pkl'):
        ctf_params_all = np.array(load_utils.load_ctf_params(D, ctf_file))
        dataset_indices = _normalize_dataset_indices(ind, n_total=ctf_params_all.shape[0])
        if dataset_indices is None and ctf_params_all.shape[0] != n_images:
            raise ValueError(
                f"CTF parameter count ({ctf_params_all.shape[0]}) must match loaded image count ({n_images}) "
                "when ind is not provided"
            )
        ctf_params = ctf_params_all if dataset_indices is None else ctf_params_all[dataset_indices]
    else:
        from recovar.data_io import metadata_readers
        source_file = ctf_file if ctf_file is not None else particles_file
        ctf_params = metadata_readers.auto_parse_ctf(source_file, D)
        dataset_indices = _normalize_dataset_indices(ind, n_total=ctf_params.shape[0])
        if dataset_indices is not None:
            ctf_params = ctf_params[dataset_indices]
        elif ctf_params.shape[0] != n_images:
            raise ValueError(
                f"CTF parameter count ({ctf_params.shape[0]}) must match loaded image count ({n_images}) "
                "when ind is not provided"
            )
        logger.info("Auto-extracted CTF parameters from %s", source_file)

    # Append bfactor (=0) and contrast (=1) columns
    ctf_params = np.concatenate([
        ctf_params,
        np.zeros_like(ctf_params[:, 0][..., None]),
        np.ones_like(ctf_params[:, 0][..., None]),
    ], axis=-1)
    return ctf_params, dataset_indices


def _load_poses(particles_file, poses_file, D, n_images, dataset_indices):
    """Load rotation matrices and translations.

    Returns ``(rots, translations)`` as float32 arrays.
    """
    if poses_file is not None and poses_file.endswith('.pkl'):
        from recovar.data_io import load_utils
        rots, trans, _ = load_utils.load_poses(poses_file, n_images, D, ind=dataset_indices)
    else:
        from recovar.data_io import metadata_readers
        source_file = poses_file if poses_file is not None else particles_file
        rots_raw, trans_frac = metadata_readers.auto_parse_poses(source_file, D)
        if dataset_indices is not None:
            rots_raw = rots_raw[dataset_indices]
            trans_frac = trans_frac[dataset_indices]
        elif rots_raw.shape[0] != n_images:
            raise ValueError(
                f"Pose count ({rots_raw.shape[0]}) must match loaded image count ({n_images}) "
                "when ind is not provided"
            )
        rots = rots_raw
        trans = trans_frac * D
        logger.info("Auto-extracted poses from %s", source_file)

    rots = np.asarray(rots, dtype=np.float32)
    if rots.ndim != 3 or rots.shape[1:] != (3, 3):
        raise ValueError(f"Rotation array must have shape (N, 3, 3), got {rots.shape}")

    if trans is None:
        translations = np.zeros((rots.shape[0], 2), dtype=np.float32)
    else:
        translations = np.asarray(trans, dtype=np.float32)
        expected_t_shape = (rots.shape[0], 2)
        if translations.shape != expected_t_shape:
            raise ValueError(
                f"Translation array must have shape {expected_t_shape}, got {translations.shape}"
            )
    return rots, translations


def _apply_tilt_ctf_augmentation(ctf_params, tilt_dataset, tilt_series_ctf,
                                  dose_per_tilt, angle_per_tilt):
    """Apply tilt-series-specific CTF augmentation.

    Returns ``(ctf_params, ctf_evaluator)`` with augmented columns.
    """
    ctf_eval = core.CTFEvaluator()

    if tilt_series_ctf == 'relion5':
        ctf_params[:, core.CTFParamIndex.CONTRAST + 1] = tilt_dataset.ctfscalefactor
        dose = tilt_dataset.dose
        angles = np.zeros_like(dose)
        ctf_params = np.concatenate([ctf_params, dose[..., None], angles[..., None]], axis=-1)
        ctf_eval = core.CTFEvaluator(mode=core.CTFMode.CRYO_ET)

    if "scale_from_star" in tilt_series_ctf:
        angle_per_tilt = 0

    if tilt_series_ctf == "from_star":
        ctf_params[:, core.CTFParamIndex.CONTRAST + 1] = tilt_dataset.ctfscalefactor
        ctf_params[:, core.CTFParamIndex.BFACTOR + 1] = -tilt_dataset.ctfBfactor
        logger.info('CTF from star')

    elif (tilt_series_ctf == "scale_from_star") or (tilt_series_ctf == "from_dose"):
        if "scale_from_star" in tilt_series_ctf:
            ctf_params[:, core.CTFParamIndex.CONTRAST + 1] = tilt_dataset.ctfscalefactor

        tilt_numbers = tilt_dataset.tilt_numbers
        ctf_params = np.concatenate([ctf_params, tilt_numbers[..., None]], axis=-1)

        if not (np.isclose(ctf_params[0, 4], 200) or np.isclose(ctf_params[0, 4], 300)):
            raise ValueError("Critical exposure calculation requires 200kV or 300kV imaging")
        ctf_eval = core.CTFEvaluator(mode=core.CTFMode.TILT_SERIES,
                                      dose_per_tilt=dose_per_tilt,
                                      angle_per_tilt=angle_per_tilt)
        logger.info('CTF from dose weighting')

    elif "v2" in tilt_series_ctf:
        tilt_numbers = tilt_dataset.tilt_numbers
        dose = -(tilt_dataset.ctfBfactor / 4)

        angles = jnp.ceil(tilt_numbers / 2) * angle_per_tilt
        if 'scale_from_star' in tilt_series_ctf:
            ctf_params[:, core.CTFParamIndex.CONTRAST + 1] = tilt_dataset.ctfscalefactor
            logger.warning("Using scale from star")

        if dose_per_tilt is None:
            dose = -(tilt_dataset.ctfBfactor / 4)
            logger.warning("Using dose from star file (- Bfactor/4)")

        ctf_eval = core.CTFEvaluator(mode=core.CTFMode.CRYO_ET)
        ctf_params = np.concatenate([ctf_params, dose[..., None], angles[..., None]], axis=-1)
        if not (np.isclose(ctf_params[0, 4], 200) or np.isclose(ctf_params[0, 4], 300)):
            raise ValueError("Critical exposure calculation requires 200kV or 300kV imaging")
        logger.info('CTF from dose weighting - V2')

    return ctf_params, ctf_eval


def _resolve_tilt_series_ctf_mode(tilt_series, tilt_series_ctf):
    """Normalize the requested CTF mode for SPA and cryo-ET inputs."""
    if tilt_series_ctf is None:
        return "relion5" if tilt_series else "cryoem"
    if tilt_series_ctf == "warp":
        return "v2_scale_from_star"
    return tilt_series_ctf


def _get_tilt_ctf_source(
    image_source,
    *,
    particles_file,
    ind,
    lazy,
    tilt_series,
    tilt_series_ctf,
    uninvert_data,
    datadir,
    padding,
    strip_prefix,
    downsample_D,
    sort_with_Bfac,
):
    """Return the grouped image source needed for tilt-specific CTF metadata."""
    if tilt_series or tilt_series_ctf == "cryoem":
        return image_source
    return _create_image_source(
        particles_file,
        ind=ind,
        lazy=lazy,
        tilt_series=True,
        tilt_series_ctf=tilt_series_ctf,
        uninvert_data=uninvert_data,
        datadir=datadir,
        padding=padding,
        strip_prefix=strip_prefix,
        downsample_D=downsample_D,
        sort_with_Bfac=sort_with_Bfac,
    )


def load_dataset(
    particles_file,
    poses_file=None,
    ctf_file=None,
    datadir=None,
    n_images=None,
    ind=None,
    lazy=True,
    padding=0,
    uninvert_data=False,
    tilt_series=False,
    tilt_series_ctf=None,
    dose_per_tilt=2.9,
    angle_per_tilt=3,
    premultiplied_ctf=False,
    strip_prefix=None,
    sort_with_Bfac=False,
    downsample_D=None,
):
    """Load a cryo-EM / cryo-ET dataset.

    Poses and CTF can come from:
    - Pickle files (legacy cryoDRGN format) via *poses_file* / *ctf_file*
    - Auto-extracted from the particles STAR or CS file when those are None
    """
    # ---- Validate auto-extraction capability ----
    if poses_file is None or ctf_file is None:
        from recovar.data_io import metadata_readers
        if not metadata_readers.can_extract_poses(particles_file):
            raise ValueError(
                f"Cannot auto-extract poses/CTF from '{particles_file}'. "
                "Provide --poses and --ctf, or use a .star or .cs particles file."
            )

    # ---- CTF mode defaults ----
    tilt_series_ctf = _resolve_tilt_series_ctf_mode(tilt_series, tilt_series_ctf)

    # ---- Create image source ----
    image_source = _create_image_source(
        particles_file, ind, lazy, tilt_series, tilt_series_ctf,
        uninvert_data, datadir, padding, strip_prefix, downsample_D,
        sort_with_Bfac=sort_with_Bfac,
    )

    # ---- Load CTF parameters ----
    ctf_params, dataset_indices = _load_ctf_params(
        particles_file, ctf_file, image_source.grid_size, ind, image_source.n_images,
    )

    # ---- Apply tilt-series CTF augmentation ----
    if tilt_series_ctf != 'cryoem':
        tilt_dataset = _get_tilt_ctf_source(
            image_source,
            particles_file=particles_file,
            ind=ind,
            lazy=lazy,
            tilt_series=tilt_series,
            tilt_series_ctf=tilt_series_ctf,
            uninvert_data=uninvert_data,
            datadir=datadir,
            padding=padding,
            strip_prefix=strip_prefix,
            downsample_D=downsample_D,
            sort_with_Bfac=sort_with_Bfac,
        )
        ctf_params, ctf_eval = _apply_tilt_ctf_augmentation(
            ctf_params, tilt_dataset, tilt_series_ctf, dose_per_tilt, angle_per_tilt,
        )
    else:
        ctf_eval = core.CTFEvaluator()

    # ---- Load poses ----
    rots, translations = _load_poses(
        particles_file, poses_file, image_source.unpadded_D,
        image_source.n_images, dataset_indices,
    )

    # ---- Validate voxel sizes ----
    voxel_sizes = ctf_params[:, 0]
    if not np.all(np.isclose(voxel_sizes - voxel_sizes[0], 0)):
        raise ValueError("All voxel sizes must be the same")
    voxel_size = np.float32(voxel_sizes[0])

    ctf_params = ctf_params.astype(np.float32)
    dtype_real = np.complex64(0).real.dtype

    meta = MetadataStore(rots, translations, ctf_params[:, 1:],
                         rotation_dtype=np.float32,
                         ctf_dtype=dtype_real,
                         real_dtype=dtype_real)
    ds = CryoEMDataset(image_source, voxel_size, meta,
                       ctf_evaluator=ctf_eval,
                       dataset_indices=dataset_indices,
                       tilt_series_flag=tilt_series,
                       premultiplied_ctf=premultiplied_ctf)
    # Store loader paths for downstream reload (e.g. independent half-datasets).
    ds.particles_file = particles_file
    ds.poses_file = poses_file
    ds.ctf_file = ctf_file
    ds.datadir = datadir
    return ds



# ---------------------------------------------------------------------------
# Half-set splitting — delegated to recovar.data_io.halfsets
# ---------------------------------------------------------------------------

from recovar.data_io.halfsets import (  # noqa: E402
    split_index_list,
    get_split_indices,
    get_split_tilt_indices,
    _read_relion_halfsets_from_star,
    figure_out_halfsets,
    get_split_datasets,
    make_dataset_loader_dict,
    load_dataset_from_args,
)




def reorder_to_original_indexing_from_halfsets(arr, halfsets, num_images = None ):
    if isinstance(arr, list):
        if len(arr) == 0:
            arr = np.array([])
        else:
            arr = np.concatenate(arr)

    dataset_indices = np.concatenate(halfsets)
    dataset_indices = np.asarray(dataset_indices).reshape(-1)
    if dataset_indices.size == 0:
        if num_images is None:
            num_images = 0
    else:
        if np.any(dataset_indices < 0):
            raise ValueError("dataset indices must be non-negative")
        unique_count = np.unique(dataset_indices).size
        if unique_count != dataset_indices.size:
            raise ValueError("dataset indices contain duplicates across halfsets")
        inferred_num_images = int(np.max(dataset_indices)) + 1
        if num_images is None:
            num_images = inferred_num_images
        elif num_images < inferred_num_images:
            raise ValueError(
                f"num_images={num_images} is smaller than required size {inferred_num_images} from dataset indices"
            )

    if arr.shape[0] != dataset_indices.size:
        raise ValueError(
            f"arr first dimension ({arr.shape[0]}) must match number of dataset indices ({dataset_indices.size})"
        )

    arr_reorder_shape = (num_images, *arr.shape[1:])
    arr_reorder = np.ones(arr_reorder_shape) * np.nan # nan things which are not in halfsets. They have been filtered out.
    arr_reorder[dataset_indices] = arr
    return arr_reorder


def reorder_to_original_indexing(arr, ds, use_tilt_indices=False):
    """Reorder a halfset-concatenated array back to original file ordering.

    For SPA (``use_tilt_indices=False``), uses ``ds.halfset_indices``
    (image-level).  For tilt-series (``use_tilt_indices=True``), uses
    the canonical particle indices derived from each half's images so
    that per-particle data is scattered to its original particle position.
    """
    if use_tilt_indices:
        halfsets = [ds.halfset_original_group_indices(halfset_id) for halfset_id in range(2)]
    else:
        halfsets = [ds.halfset_original_image_indices(halfset_id) for halfset_id in range(2)]
    return reorder_to_original_indexing_from_halfsets(arr, halfsets)


def reorder_to_dataset_indexing(arr, ds, use_tilt_indices=False):
    """Reorder a halfset-concatenated array back to this dataset's local ordering."""
    if use_tilt_indices:
        halfsets = [ds.halfset_local_group_indices(halfset_id) for halfset_id in range(2)]
        n_items = ds.n_units
    else:
        halfsets = [ds.halfset_local_image_indices(halfset_id) for halfset_id in range(2)]
        n_items = ds.n_images
    return reorder_to_original_indexing_from_halfsets(arr, halfsets, num_images=n_items)


def subsample_cryoem_dataset(cryo, good_indices):
    """Return a new CryoEMDataset containing only the images at *good_indices*."""
    return cryo.subset(good_indices)
