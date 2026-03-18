"""Dataset loading, half-set splitting, and image access for cryo-EM/cryo-ET."""

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
from recovar.data_io import cryo_dataset
from recovar.output import plot_utils

logger = logging.getLogger(__name__)

from recovar.data_io._index_utils import normalize_indices
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

def set_standard_mask(D, dtype):
    return mask.window_mask(D, 0.85, 0.99)


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
        return values
    _, first_idx = np.unique(values, return_index=True)
    if first_idx.size != values.size:
        dropped = int(values.size - first_idx.size)
        logger.warning("Dropping %d duplicate entries from %s.", dropped, name)
    return values[np.sort(first_idx)]


class _SubsetNoiseAdapter:
    """Wrap a noise model to translate subset-local indices to original indices.

    When ``CryoEMDataset.subset()`` creates a view, generators remap yielded
    indices to local 0..n-1 space.  The parent's noise model still expects
    *original* image-stack indices.  This adapter intercepts ``get()`` and
    ``get_half()`` and translates local → original before delegating.

    All other attributes (``get_average_radial_noise``, ``dose_indices``, etc.)
    are forwarded transparently, including ``isinstance`` checks.
    """

    def __init__(self, noise_model, subset_indices):
        self._noise = noise_model
        self._subset_indices = np.asarray(subset_indices)

    def get(self, indices):
        return self._noise.get(self._subset_indices[indices])

    def get_half(self, indices):
        return self._noise.get_half(self._subset_indices[indices])

    @property
    def __class__(self):
        return type(self._noise)

    def __getattr__(self, name):
        return getattr(self._noise, name)


class Metadata:
    """Per-image metadata store (poses + CTF parameters).

    Callers access metadata only through ``get_batch()`` or narrow
    column/mutation methods — never by indexing raw arrays.
    """

    __slots__ = ('_rotation_matrices', '_translations', '_ctf_params',
                 'rotation_dtype', 'ctf_dtype', 'real_dtype')

    def __init__(self, rotation_matrices, translations, ctf_params, *,
                 rotation_dtype=np.float32, ctf_dtype=np.float32,
                 real_dtype=np.float32):
        self._rotation_matrices = np.asarray(rotation_matrices, dtype=rotation_dtype)
        self._translations = np.asarray(translations, dtype=real_dtype)
        self._ctf_params = np.asarray(ctf_params, dtype=ctf_dtype)
        self.rotation_dtype = rotation_dtype
        self.ctf_dtype = ctf_dtype
        self.real_dtype = real_dtype

    @property
    def n_images(self):
        return self._rotation_matrices.shape[0]

    # --- Batch access (primary API) ---

    def get_batch(self, indices):
        """Return (rotation_matrices, translations, ctf_params) for indices."""
        return (self._rotation_matrices[indices],
                self._translations[indices],
                self._ctf_params[indices])

    # --- Narrow accessors ---

    def get_ctf_column(self, col):
        """Read a single CTF parameter column for all images."""
        return self._ctf_params[:, col]

    def get_ctf_params_copy(self):
        """Return a mutable copy of the full CTF parameter array."""
        return self._ctf_params.copy()

    def get_rotations_copy(self):
        """Return a mutable copy of rotation matrices."""
        return self._rotation_matrices.copy()

    # --- Mutations ---

    def set_poses(self, rotation_matrices, translations):
        """Replace all poses (used by EM hard assignment)."""
        self._rotation_matrices = np.asarray(rotation_matrices, dtype=self.rotation_dtype)
        self._translations = np.asarray(translations, dtype=self.real_dtype)

    def set_ctf(self, ctf_params):
        """Replace all CTF parameters (used by Ewald preprocessing)."""
        self._ctf_params = np.asarray(ctf_params, dtype=self.ctf_dtype)

    def set_ctf_column(self, col, values):
        """Update a single CTF column in-place."""
        self._ctf_params[:, col] = values

    def scale_ctf_column(self, col, multipliers):
        """Multiply a CTF column by per-image values."""
        self._ctf_params[:, col] *= multipliers

    def scale_ctf_element(self, row_indices, col, multiplier):
        """Multiply a specific CTF element for given rows."""
        self._ctf_params[row_indices, col] *= multiplier

    def subset(self, indices):
        """Return a new Metadata for the given image indices."""
        return Metadata(
            self._rotation_matrices[indices],
            self._translations[indices],
            self._ctf_params[indices],
            rotation_dtype=self.rotation_dtype,
            ctf_dtype=self.ctf_dtype,
            real_dtype=self.real_dtype,
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
        image_stack: Underlying image loader (``None`` for simulation).
        tilt_series_flag: ``True`` when the dataset represents tilt-series data.
    """

    __slots__ = (
        # Grid geometry
        'voxel_size', 'grid_size', 'volume_shape', 'volume_size',
        # Image stack and image geometry
        'image_stack', 'image_shape', 'image_size', 'n_images', 'padding',
        # Processing flags
        'tilt_series_flag', 'premultiplied_ctf', 'n_units',
        '_ctf_evaluator', 'hpad', 'volume_mask_threshold',
        # Data types
        'dtype', 'dtype_real',
        # Per-image metadata (private — access via iterate/make_batch_data)
        '_metadata',
        # Mutable state
        'dataset_indices', 'noise',
        # Half-set views (index arrays into this dataset's ordering)
        'halfset_indices',
        # Subset view (index into original image_stack)
        '_subset_indices',
    )

    def __init__(
        self,
        image_stack,
        voxel_size: float,
        metadata: Metadata,
        ctf_evaluator=None,
        dtype: type = np.complex64,
        dataset_indices: Optional[NDArray[np.integer]] = None,
        grid_size: Optional[int] = None,
        tilt_series_flag: bool = False,
        premultiplied_ctf: bool = False,
    ) -> None:
        # --- Grid geometry ---
        if image_stack is not None:
            grid_size = image_stack.D
        elif grid_size is None:
            raise ValueError("Must specify grid_size if image_stack is None")

        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.volume_shape = (grid_size, grid_size, grid_size)
        self.volume_size = grid_size ** 3

        # --- Image stack and image geometry ---
        if image_stack is None:
            self.image_stack = None
            self.image_shape = (grid_size, grid_size)
            self.image_size = grid_size ** 2
            self.n_images = metadata.n_images
            self.padding = 0
        else:
            self.image_stack = image_stack
            self.image_shape = tuple(image_stack.image_shape)
            self.image_size = int(np.prod(image_stack.image_shape))
            self.n_images = image_stack.n_images
            self.padding = image_stack.padding

        # --- Processing flags ---
        self.tilt_series_flag = tilt_series_flag
        self.premultiplied_ctf = premultiplied_ctf
        # For SPA n_units == n_images; for tilt series n_units == n_particles
        self.n_units = self.image_stack.Np if self.tilt_series_flag else self.n_images
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

        # --- Mutable state ---
        self.dataset_indices = dataset_indices
        self.noise = None
        self.halfset_indices = None
        self._subset_indices = None

    # --- Metadata access (public API) ---

    @property
    def metadata(self):
        """The per-image metadata store."""
        return self._metadata

    def make_batch_data(self, images, indices, *, noise_variance=None,
                        particle_indices=None):
        """Bundle images with per-image metadata for a batch."""
        from recovar.core.configs import BatchData
        rots, trans, ctf = self._metadata.get_batch(indices)
        return BatchData(
            images=images,
            rotation_matrices=rots,
            translations=trans,
            ctf_params=ctf,
            noise_variance=noise_variance,
            particle_indices=particle_indices,
            image_indices=indices,
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

    # --- Delegating properties (avoid leaking image_stack internals) ---

    def process_images(self, images, apply_image_mask=False):
        """Apply windowing + DFT preprocessing to raw images."""
        return self.image_stack.process_images(images, apply_image_mask=apply_image_mask)

    @property
    def image_mask(self):
        """Circular window mask from the image stack."""
        return self.image_stack.mask if self.image_stack is not None else None

    @property
    def data_multiplier(self):
        """Sign multiplier for data inversion (±1)."""
        return getattr(self.image_stack, 'mult', 1)

    @data_multiplier.setter
    def data_multiplier(self, value):
        if self.image_stack is not None:
            self.image_stack.mult = value

    @property
    def dataset_tilt_indices(self):
        """Per-particle tilt index lists (tilt-series only)."""
        if not self.tilt_series_flag:
            return None
        return self.image_stack.dataset_tilt_indices

    @property
    def tilt_particles(self):
        """List of per-particle tilt index arrays (tilt-series only)."""
        if not self.tilt_series_flag:
            return None
        return getattr(self.image_stack, 'particles', None)

    def get_noise_variance(self, indices):
        if self.noise is None:
            return None
        return self.noise.get(indices)

    def subset(self, indices):
        """Return a new CryoEMDataset containing only the images at *indices*.

        The returned dataset keeps a reference to the *original* image_stack
        and stores ``_subset_indices`` for on-the-fly subsetting in generators,
        instead of wrapping the stack in an intermediate object.
        """
        indices = _normalize_image_indices(indices, n_images_total=self.n_images, name="indices")

        # Compose subset indices: if the parent is already a subset view,
        # map through its indices to reach the original image_stack.
        if self._subset_indices is not None:
            composed_subset = self._subset_indices[indices]
        else:
            composed_subset = indices

        # Compose dataset_indices for original file-level indexing.
        if self.dataset_indices is not None:
            composed_indices = self.dataset_indices[indices]
        else:
            composed_indices = indices

        sub = CryoEMDataset(
            image_stack=self.image_stack,
            voxel_size=self.voxel_size,
            metadata=self._metadata.subset(indices),
            ctf_evaluator=self.ctf_evaluator,
            grid_size=self.grid_size,
            tilt_series_flag=self.tilt_series_flag,
            premultiplied_ctf=self.premultiplied_ctf,
            dataset_indices=composed_indices,
        )
        sub._subset_indices = composed_subset
        # Override n_images/n_units to match the subset size, not the full
        # image_stack size (since we share the stack reference).
        sub.n_images = len(indices)
        if self.tilt_series_flag and hasattr(self.image_stack, '_particle_tilts'):
            # Count how many particles have tilts in the subset.
            subset_set = set(np.asarray(composed_subset).ravel().tolist())
            n_particles = sum(
                1 for tilts in self.image_stack._particle_tilts
                if any(t in subset_set for t in tilts)
            )
            sub.n_units = n_particles
        else:
            sub.n_units = len(indices)
        if self.noise is not None:
            sub.noise = _SubsetNoiseAdapter(self.noise, composed_subset)
        return sub

    def _remap_generator(self, gen, *, remap_particles=False):
        """Remap original image-stack indices to local 0..n-1 for subset views.

        When *remap_particles* is True (tilt-series particle-grouped path),
        remap particle_indices as well as image_indices using their respective
        mappings.
        """
        remap = np.empty(int(self._subset_indices.max()) + 1, dtype=np.int32)
        remap[self._subset_indices] = np.arange(len(self._subset_indices), dtype=np.int32)

        if remap_particles and self.tilt_series_flag and hasattr(self.image_stack, '_particle_tilts'):
            # Build global-particle → local-particle remap
            subset_set = set(np.asarray(self._subset_indices).ravel().tolist())
            global_to_local_particle = {}
            local_pid = 0
            for g_pid, tilts in enumerate(self.image_stack._particle_tilts):
                if any(t in subset_set for t in tilts):
                    global_to_local_particle[g_pid] = local_pid
                    local_pid += 1
            for images, particles_ind, image_indices in gen:
                local_img = remap[np.asarray(image_indices)]
                local_part = np.array([global_to_local_particle[int(p)] for p in np.asarray(particles_ind).ravel()],
                                      dtype=np.int32)
                yield images, local_part, local_img
        else:
            for images, _, image_indices in gen:
                local = remap[np.asarray(image_indices)]
                yield images, local, local

    def _get_backing_subset_generator(self, batch_size, indices, num_workers=0, *, particle_grouped=False, **kwargs):
        """Get the right subset generator from the backing image_stack."""
        if self.tilt_series_flag and not particle_grouped and hasattr(self.image_stack, 'get_image_subset_generator'):
            return self.image_stack.get_image_subset_generator(
                batch_size, indices, num_workers=num_workers)
        return self.image_stack.get_dataset_subset_generator(
            batch_size, indices, num_workers=num_workers, **kwargs)

    def _get_dataset_generator(self, batch_size, num_workers=0, **kwargs):
        if self._subset_indices is not None:
            if self.tilt_series_flag and hasattr(self.image_stack, '_particle_tilts'):
                # Convert image-level _subset_indices to particle-level indices
                # for the particle-grouped generator.
                subset_set = set(np.asarray(self._subset_indices).ravel().tolist())
                particle_indices = np.array([
                    p_idx for p_idx, tilts in enumerate(self.image_stack._particle_tilts)
                    if any(t in subset_set for t in tilts)
                ], dtype=np.int32)
                gen = self.image_stack.get_dataset_subset_generator(
                    batch_size, particle_indices, num_workers=num_workers, **kwargs)
                return self._remap_generator(gen, remap_particles=True)
            else:
                gen = self._get_backing_subset_generator(
                    batch_size, self._subset_indices, num_workers=num_workers,
                    particle_grouped=True, **kwargs)
                return self._remap_generator(gen, remap_particles=True)
        return self.image_stack.get_dataset_generator(batch_size, num_workers=num_workers, **kwargs)

    def _get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
        if subset_indices is None:
            return self._get_dataset_generator(batch_size, num_workers=num_workers, **kwargs)
        # Map local subset_indices through _subset_indices to original indices.
        subset_indices = _normalize_image_indices(
            subset_indices, n_images_total=self.n_images, name="subset_indices")
        if self._subset_indices is not None:
            orig_indices = self._subset_indices[subset_indices]
            gen = self._get_backing_subset_generator(
                batch_size, orig_indices, num_workers=num_workers, **kwargs)
            # Remap the yielded original indices back to the local space of this dataset.
            remap = np.empty(int(self._subset_indices.max()) + 1, dtype=np.int32)
            remap[self._subset_indices] = np.arange(len(self._subset_indices), dtype=np.int32)
            def _remap(g):
                for images, _, image_indices in g:
                    local = remap[np.asarray(image_indices)]
                    yield images, local, local
            return _remap(gen)
        return self.image_stack.get_dataset_subset_generator(
            batch_size, subset_indices, num_workers=num_workers, **kwargs)

    # Iterate over individual images rather than tilt groups. For SPA, same as get_dataset_subset_generator.
    def _get_image_subset_generator(self, batch_size, subset_indices, num_workers=0):
        if self.tilt_series_flag:
            if self._subset_indices is not None:
                if subset_indices is not None:
                    subset_indices = _normalize_image_indices(
                        subset_indices, n_images_total=self.n_images, name="subset_indices")
                    orig_indices = self._subset_indices[subset_indices]
                else:
                    orig_indices = self._subset_indices
                gen = self.image_stack.get_image_subset_generator(
                    batch_size, orig_indices, num_workers=num_workers)
                return self._remap_generator(gen)
            return self.image_stack.get_image_subset_generator(
                batch_size, subset_indices, num_workers=num_workers)
        return self._get_dataset_subset_generator(batch_size, subset_indices, num_workers=num_workers)

    # Iterate over individual images rather than tilt groups. For SPA, same as get_dataset_generator.
    def _get_image_generator(self, batch_size, num_workers=0):
        if self.tilt_series_flag:
            if self._subset_indices is not None:
                gen = self.image_stack.get_image_subset_generator(
                    batch_size, self._subset_indices, num_workers=num_workers)
                return self._remap_generator(gen)
            return self.image_stack.get_image_generator(batch_size, num_workers=num_workers)
        return self._get_dataset_generator(batch_size, num_workers=num_workers)


    # --- Backward-compatible public aliases for generator methods ---
    def get_dataset_generator(self, *args, **kwargs):
        return self._get_dataset_generator(*args, **kwargs)

    def get_dataset_subset_generator(self, *args, **kwargs):
        return self._get_dataset_subset_generator(*args, **kwargs)

    def get_image_generator(self, *args, **kwargs):
        return self._get_image_generator(*args, **kwargs)

    def get_image_subset_generator(self, *args, **kwargs):
        return self._get_image_subset_generator(*args, **kwargs)

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
            im = im[self.hpad:self.image_stack.unpadded_D + self.hpad,self.hpad:self.image_stack.unpadded_D + self.hpad]
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
            image = self.image_stack.__getitem__(i)[0]#[None]
        else:
            image = self.image_stack.__getitem__(i)[0][tilt_idx][None]

        processed_image = self.image_stack.process_images(image)
        return processed_image.reshape(self.image_shape)

    def get_CTF_image(self, i ):
        return self.get_CTF(np.array([i])).reshape(self.image_shape)

    def get_image_real(self,i, tilt_idx = None, to_real= np.real, hide_padding = True):
        hpad= self.image_stack.padding//2
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
            images, _, image_ind = self.image_stack.__getitem__(i)
            images = images[tilt_idx][None]
            CTFs = self.ctf_evaluator(self.CTF_params[image_ind[tilt_idx]][None], self.image_shape, self.voxel_size) # Compute CTF
        else:
            images, _, _ = self.image_stack.__getitem__(i)
            CTFs = self.ctf_evaluator(self.CTF_params[i][None], self.image_shape, self.voxel_size) # Compute CTF
        images = self.image_stack.process_images(images) # Compute DFT, masking
        images = (CTFs / (CTFs**2 + weiner_param)) * images  # CTF correction
        images = images.reshape(self.image_shape)
        return to_real(fourier_transform_utils.get_idft2(images))


    def plot_FSC(self, image1 = None, image2 = None, filename = None, threshold = 0.5, curve = None, ax = None):
        score = plot_utils.plot_fsc_new(image1, image2, self.volume_shape, self.voxel_size,  curve = curve, ax = ax, threshold = threshold, filename = filename)
        return score
    
    def get_image_mask(self, indices, mask, binary = True, soften = 5):
        indices = np.asarray(indices, dtype=int)
        from recovar.heterogeneity import covariance_core # Not sure I want this depency to exist... Could make some circular imports
        mask = covariance_core.get_per_image_tight_mask(mask, self.rotation_matrices[indices], self.image_stack.mask, self.volume_mask_threshold, self.image_shape, self.volume_shape, self.grid_size, self.padding, disc_type = 'linear_interp',  binary = binary, soften = soften)
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

    def n_images_half(self, half: int) -> int:
        """Number of images in a given halfset."""
        if self.halfset_indices is None:
            raise ValueError("halfset_indices not set on this dataset")
        return len(self.halfset_indices[half])

    def get_particle_halfset_indices(self):
        """Per-half canonical particle indices for tilt-series datasets.

        For SPA datasets, this simply returns ``halfset_indices`` (images
        and particles are 1-to-1).  For tilt-series, it maps each half's
        image indices through the image→particle mapping and returns the
        unique canonical (``dataset_tilt_indices``) particle ids per half.
        """
        if self.halfset_indices is None:
            raise ValueError("halfset_indices not set on this dataset")
        if not self.tilt_series_flag:
            return self.halfset_indices
        _dti = np.asarray(self.dataset_tilt_indices)
        _img_to_particle = np.full(self.n_images, -1, dtype=np.int32)
        for p_idx, tilts in enumerate(self.image_stack._particle_tilts):
            for t in tilts:
                if t < self.n_images:
                    _img_to_particle[t] = p_idx
        result = []
        for half_imgs in self.halfset_indices:
            half_particles = np.unique(_img_to_particle[np.asarray(half_imgs)])
            result.append(_dti[half_particles])
        return result

    def split_halfset_array(self, arr, per_particle=False):
        """Split a concatenated halfset-ordered array into [half0, half1].

        Parameters
        ----------
        per_particle : bool
            If True **and** this is a tilt-series dataset, split at the
            particle boundary instead of the image boundary.
        """
        if per_particle and self.tilt_series_flag:
            particle_halfs = self.get_particle_halfset_indices()
            n0 = len(particle_halfs[0])
        else:
            n0 = self.n_images_half(0)
        return [arr[:n0], arr[n0:]]

    def iterate(self, batch_size, *, half=None, indices=None,
                noise_model=None, noise_half=True, noise_by_particle=False,
                by_image=True, process_images=False, half_images=False,
                prefetch=True):
        """Iterate over images, yielding BatchData with all metadata bundled.

        Parameters
        ----------
        batch_size : int
        half : int, optional
            Halfset index (0 or 1). Mutually exclusive with *indices*.
        indices : array-like, optional
            Iterate over this subset of image indices.
        noise_model : optional
            Noise model for noise_variance in BatchData.
        noise_half : bool
            Use half-spectrum noise (default True for mean reconstruction).
        noise_by_particle : bool
            Index noise by particle group (for covariance path).
        by_image : bool
            True = flat per-image iteration; False = particle-grouped (tilt).
        process_images : bool
            Apply DFT preprocessing to images before yielding.
        half_images : bool
            Convert images to rfft-packed half-spectrum.
        prefetch : bool
            Enable 1-lookahead prefetch buffer (default True).
        """
        if half is not None and indices is not None:
            raise ValueError("Cannot specify both half and indices")
        if half is not None:
            if self.halfset_indices is None:
                raise ValueError("halfset_indices not set on this dataset")
            indices = self.halfset_indices[half]
            # For tilt-series with particle-grouped iteration, convert
            # image-level halfset indices to particle-level indices.  The
            # particle-grouped generator (TiltSeriesDataset) expects particle
            # indices, not image indices.  All tilts from a single particle
            # belong to the same halfset.
            if not by_image and self.tilt_series_flag and hasattr(self.image_stack, '_particle_tilts'):
                img_set = set(np.asarray(indices).ravel().tolist())
                indices = np.array([
                    p_idx for p_idx, tilts in enumerate(self.image_stack._particle_tilts)
                    if any(t in img_set for t in tilts)
                ], dtype=np.int32)

        from recovar.core.configs import DataIterator
        inner = DataIterator(self, batch_size,
                             noise_model=noise_model, noise_half=noise_half,
                             noise_by_particle=noise_by_particle,
                             index_subset=indices, use_image_generator=by_image,
                             apply_process_images=process_images,
                             half_images=half_images)
        if prefetch:
            return _prefetch_iter(inner)
        return inner

    def iterate_all(self, batch_size: int, **kw):
        """Iterate over all images, yielding complete BatchData.

        Convenience wrapper — equivalent to ``iterate(batch_size, **kw)``.
        """
        return self.iterate(batch_size, **kw)

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


def _create_image_stack(particles_file, ind, lazy, tilt_series, tilt_series_ctf,
                        uninvert_data, datadir, padding, strip_prefix, downsample_D):
    """Create the underlying image-stack dataset."""
    if tilt_series:
        tilt_file_option = 'relion5' if tilt_series_ctf == 'relion5' else 'warp'
        return cryo_dataset.TiltSeriesDataset(
            particles_file, ind=ind, datadir=datadir,
            invert_data=uninvert_data, tilt_file_option=tilt_file_option,
            strip_prefix=strip_prefix,
        )
    return cryo_dataset.ParticleImageDataset(
        particles_file, ind=ind, datadir=datadir, padding=padding,
        invert_data=uninvert_data, lazy=lazy, strip_prefix=strip_prefix,
        downsample_D=downsample_D,
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
        from recovar.data_io import metadata_parsing
        source_file = ctf_file if ctf_file is not None else particles_file
        ctf_params = metadata_parsing.auto_parse_ctf(source_file, D)
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
        from recovar.data_io import metadata_parsing
        source_file = poses_file if poses_file is not None else particles_file
        rots_raw, trans_frac = metadata_parsing.auto_parse_poses(source_file, D)
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
        from recovar.data_io import metadata_parsing
        if not metadata_parsing.can_extract_poses(particles_file):
            raise ValueError(
                f"Cannot auto-extract poses/CTF from '{particles_file}'. "
                "Provide --poses and --ctf, or use a .star or .cs particles file."
            )

    # ---- CTF mode defaults ----
    if tilt_series_ctf is None and tilt_series is False:
        tilt_series_ctf = 'cryoem'
    elif tilt_series_ctf is None and tilt_series is True:
        tilt_series_ctf = 'relion5'
    elif tilt_series_ctf == 'warp':
        tilt_series_ctf = 'v2_scale_from_star'

    # ---- Create image stack ----
    image_stack = _create_image_stack(
        particles_file, ind, lazy, tilt_series, tilt_series_ctf,
        uninvert_data, datadir, padding, strip_prefix, downsample_D,
    )

    # ---- Load CTF parameters ----
    ctf_params, dataset_indices = _load_ctf_params(
        particles_file, ctf_file, image_stack.D, ind, image_stack.n_images,
    )

    # ---- Apply tilt-series CTF augmentation ----
    if tilt_series_ctf != 'cryoem':
        if (tilt_series is False) and (tilt_series_ctf != 'cryoem'):
            tilt_dataset = cryo_dataset.TiltSeriesDataset(
                particles_file, ind=ind, datadir=datadir,
                invert_data=uninvert_data, sort_with_Bfac=sort_with_Bfac,
            )
        else:
            tilt_dataset = image_stack
        ctf_params, ctf_eval = _apply_tilt_ctf_augmentation(
            ctf_params, tilt_dataset, tilt_series_ctf, dose_per_tilt, angle_per_tilt,
        )
    else:
        ctf_eval = core.CTFEvaluator()

    # ---- Load poses ----
    rots, translations = _load_poses(
        particles_file, poses_file, image_stack.unpadded_D,
        image_stack.n_images, dataset_indices,
    )

    # ---- Validate voxel sizes ----
    voxel_sizes = ctf_params[:, 0]
    if not np.all(np.isclose(voxel_sizes - voxel_sizes[0], 0)):
        raise ValueError("All voxel sizes must be the same")
    voxel_size = np.float32(voxel_sizes[0])

    ctf_params = ctf_params.astype(np.float32)
    dtype_real = np.complex64(0).real.dtype

    meta = Metadata(rots, translations, ctf_params[:, 1:],
                    rotation_dtype=np.float32,
                    ctf_dtype=dtype_real,
                    real_dtype=dtype_real)
    ds = CryoEMDataset(image_stack, voxel_size, meta,
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
        halfsets = ds.get_particle_halfset_indices()
    else:
        halfsets = ds.halfset_indices
    return reorder_to_original_indexing_from_halfsets(arr, halfsets)


def subsample_cryoem_dataset(cryo, good_indices):
    """Return a new CryoEMDataset containing only the images at *good_indices*."""
    return cryo.subset(good_indices)
