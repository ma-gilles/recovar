"""Equinox-based configuration modules for clean parameter passing.

These modules bundle the many parameters that are passed to JAX-jitted functions
throughout RECOVAR into structured, typed containers. This eliminates the need for
complex ``static_argnums`` lists and makes function signatures readable.

Usage::

    config = ForwardModelConfig.from_dataset(cryo, disc_type='linear_interp')
    batch = BatchData(images=imgs, rotation_matrices=rots,
                      translations=trans, ctf_params=ctf, noise_variance=nv)
    model = ModelState(mean_estimate=mean, volume_mask=vmask)

    # Instead of 20-arg functions:
    result = some_jitted_fn(config, batch, model)
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import numpy as np

from recovar.core.ctf import CTFEvaluator, as_ctf_evaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ForwardModelConfig — static compile-time constants for the forward model
# ---------------------------------------------------------------------------

class ForwardModelConfig(eqx.Module):
    """Bundles geometry, CTF, and discretization parameters.

    All fields are static (compile-time constants), so changing any value
    triggers JAX recompilation — but these values are fixed for a given
    dataset and reconstruction run.
    """

    image_shape: Tuple[int, int] = eqx.field(static=True)
    volume_shape: Tuple[int, int, int] = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)
    padding: int = eqx.field(static=True)
    disc_type: str = eqx.field(static=True)
    ctf: CTFEvaluator = eqx.field(static=True)
    premultiplied_ctf: bool = eqx.field(static=True, default=False)
    volume_mask_threshold: float = eqx.field(static=True, default=0.0)
    volume_upsampling_factor: int = eqx.field(static=True, default=1)
    data_multiplier: float = eqx.field(static=True, default=1.0)
    process_fn: Optional[Callable] = eqx.field(static=True, default=None)

    def compute_ctf(self, ctf_params: jax.Array, *, half_image: bool = False) -> jax.Array:
        """Compute CTF values for a batch of images.

        Parameters
        ----------
        half_image : if True, evaluate on the rfft-packed half-spectrum grid.
        """
        return self.ctf(ctf_params, self.image_shape, self.voxel_size, half_image=half_image)

    def compute_ctf_half(self, ctf_params: jax.Array) -> jax.Array:
        """Convenience alias for ``compute_ctf(ctf_params, half_image=True)``."""
        return self.compute_ctf(ctf_params, half_image=True)

    def compute_ctf_at_shape(self, ctf_params: jax.Array, image_shape: Tuple[int, int]) -> jax.Array:
        """Compute CTF on a different frequency grid (e.g. upsampled)."""
        return self.ctf(ctf_params, image_shape, self.voxel_size)

    @property
    def base_volume_shape(self) -> Tuple[int, int, int]:
        """Original (non-upsampled) volume shape."""
        gs = self.grid_size // self.volume_upsampling_factor
        return (gs, gs, gs)

    def replace(self, **kwargs) -> ForwardModelConfig:
        """Create a new config with some fields replaced.

        Useful for changing static fields (e.g. ``disc_type``) which cannot
        be modified via ``eqx.tree_at``.
        """
        fields = dict(
            image_shape=self.image_shape, volume_shape=self.volume_shape,
            grid_size=self.grid_size, voxel_size=self.voxel_size,
            padding=self.padding, disc_type=self.disc_type,
            ctf=self.ctf,
            premultiplied_ctf=self.premultiplied_ctf,
            volume_mask_threshold=self.volume_mask_threshold,
            volume_upsampling_factor=self.volume_upsampling_factor,
            data_multiplier=self.data_multiplier,
            process_fn=self.process_fn,
        )
        fields.update(kwargs)
        return ForwardModelConfig(**fields)

    @property
    def volume_size(self) -> int:
        return int(np.prod(self.volume_shape))

    @property
    def image_size(self) -> int:
        return int(np.prod(self.image_shape))

    @classmethod
    def from_dataset(
        cls,
        cryo,
        disc_type: str = "linear_interp",
        process_fn: Optional[Callable] = None,
        upsampling_factor: Optional[int] = None,
    ) -> ForwardModelConfig:
        """Create from a CryoEMDataset instance.

        Parameters
        ----------
        cryo : CryoEMDataset
            Source dataset for geometry and CTF configuration.
        disc_type : str
            Discretization type (e.g. 'linear_interp', 'cubic', '').
        process_fn : callable, optional
            Image preprocessing function applied inside jitted code.
        upsampling_factor : int, optional
            Volume upsampling factor (e.g. 2 for 2× oversampled grid).
            Computes the upsampled volume shape directly without
            mutating the dataset object.
        """
        from recovar.data_io.dataset import CryoEMDataset

        # Extract the CTFEvaluator directly (not the dtype-casting method).
        if isinstance(cryo, CryoEMDataset):
            ctf_eval = cryo.ctf_evaluator
        else:
            # Duck-type fallback
            ctf_eval = as_ctf_evaluator(cryo.ctf_evaluator)

        base_grid_size = int(cryo.grid_size)

        if upsampling_factor is not None:
            volume_upsampling = int(upsampling_factor)
            grid_size = base_grid_size * volume_upsampling
            volume_shape = (grid_size, grid_size, grid_size)
        else:
            volume_shape = (base_grid_size,) * 3
            grid_size = base_grid_size
            volume_upsampling = 1

        config = cls(
            image_shape=tuple(int(x) for x in cryo.image_shape),
            volume_shape=volume_shape,
            grid_size=grid_size,
            voxel_size=float(cryo.voxel_size),
            padding=int(getattr(cryo, 'padding', 0)),
            disc_type=disc_type,
            ctf=ctf_eval,
            premultiplied_ctf=bool(getattr(cryo, 'premultiplied_ctf', False)),
            volume_mask_threshold=float(getattr(cryo, 'volume_mask_threshold', 0.0)),
            volume_upsampling_factor=volume_upsampling,
            data_multiplier=float(getattr(cryo, 'data_multiplier', 1.0)),
            process_fn=process_fn,
        )
        logger.debug("ForwardModelConfig: grid=%d, image=%s, disc=%s, premult_ctf=%s",
                     grid_size, config.image_shape, disc_type, config.premultiplied_ctf)
        return config


# ---------------------------------------------------------------------------
# BatchData — per-batch dynamic arrays (GPU-transferred by eqx.filter_jit)
# ---------------------------------------------------------------------------

class BatchData(eqx.Module):
    """Per-batch image data passed to jitted computation kernels.

    All fields are dynamic JAX array leaves. ``eqx.filter_jit`` handles
    GPU transfer automatically when these are passed to jitted functions.
    """

    images: jax.Array
    rotation_matrices: jax.Array
    translations: jax.Array
    ctf_params: jax.Array
    noise_variance: Optional[jax.Array] = None
    particle_indices: Optional[jax.Array] = None
    image_indices: Optional[jax.Array] = None


# ---------------------------------------------------------------------------
# DataIterator — legacy compatibility wrapper over the dataset iterator
# ---------------------------------------------------------------------------

class DataIterator:
    """Legacy compatibility wrapper yielding :class:`BatchData` objects.

    New code should prefer ``dataset.iter_batches()``, which yields explicit
    batch fields directly. This wrapper exists only for older call sites that
    still expect :class:`BatchData`.

    Iterator selection and noise lookup are delegated to the dataset iterator
    backend so there is a single implementation of batching logic.

    Parameters
    ----------
    dataset : CryoEMDataset
        Source for images and per-image metadata
        (``rotation_matrices``, ``translations``, ``CTF_params``).
    batch_size : int
    noise_model : optional
        Object exposing ``get_half(indices)`` and ``get(indices)``
        (e.g. :class:`~recovar.reconstruction.noise.RadialNoiseModel`).
        When *None*, ``BatchData.noise_variance`` is left as *None*.
    noise_half : bool, default True
        When *True*, call ``noise_model.get_half(indices)`` (half-spectrum
        noise for half-image backprojection). When *False*, call
        ``noise_model.get(indices)`` (full-spectrum noise).
    noise_by_particle : bool, default False
        When *True*, use ``particles_ind`` (particle-group indices) for
        noise lookup instead of flat ``indices``.  Needed by the covariance
        column path where ``noise.get(particles_ind)`` is the correct call.
    index_subset : array-like, optional
        If given, iterate only over these image indices.
    use_image_generator : bool, default True
        When *True* (default), use ``get_image_generator`` /
        ``get_image_subset_generator`` (flat per-image iteration, correct for
        mean reconstruction and tilt-series treated image-by-image).
        When *False*, use ``get_dataset_generator`` /
        ``get_dataset_subset_generator`` (particle-grouped, for tilt-series EM).
    apply_process_images : bool, default False
        When *True*, call ``dataset.process_images(batch,
        apply_image_mask=False)`` on each raw batch before yielding.
        Required for kernels that expect preprocessed real-space images
        (e.g. the heterogeneity and residual kernels).
    half_images : bool, default False
        When *True*, convert images to half-image (rfft-packed) format before
        yielding.  Applied after ``apply_process_images`` (if any), so that
        ``process_images`` always operates on full-spectrum images.

    Example::

        nm = noise.as_noise_model(cov_noise, config.image_shape)
        for batch in DataIterator(dataset, batch_size=512, noise_model=nm):
            Ft_y, Ft_ctf = relion_kernel_batch(config, batch, Ft_y, Ft_ctf)

        # Heterogeneity pipeline — preprocessed images, full-spectrum noise:
        for batch in DataIterator(dataset, batch_size=512, noise_model=nm,
                                  noise_half=False, apply_process_images=True):
            ...
    """

    def __init__(
        self, dataset, batch_size, *,
        noise_model=None, noise_half=True, noise_by_particle=False,
        index_subset=None, use_image_generator=True, apply_process_images=False,
        half_images=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.noise_model = noise_model
        self.noise_half = noise_half
        self.noise_by_particle = noise_by_particle
        self.index_subset = index_subset
        self.use_image_generator = use_image_generator
        self.apply_process_images = apply_process_images
        self.half_images = half_images

    def __iter__(self):
        do_process = self.apply_process_images
        do_half = self.half_images
        for batch, rotation_matrices, translations, ctf_params, noise_variance, particle_indices, image_indices in self.dataset.iter_batches(
            self.batch_size,
            indices=self.index_subset,
            noise_model=self.noise_model,
            noise_half=self.noise_half,
            noise_by_particle=self.noise_by_particle,
            by_image=self.use_image_generator,
            prefetch=False,
        ):
            if do_process and do_half:
                # Fused path: pad + rfft2 directly → half-spectrum output.
                batch = self.dataset.process_images_half(batch, apply_image_mask=False)
            elif do_process:
                batch = self.dataset.process_images(batch, apply_image_mask=False)
            elif do_half:
                import recovar.core.fourier_transform_utils as ftu
                batch = ftu.full_image_to_half_image(batch, tuple(self.dataset.image_shape))
            yield BatchData(
                images=batch,
                rotation_matrices=rotation_matrices,
                translations=translations,
                ctf_params=ctf_params,
                noise_variance=noise_variance,
                particle_indices=particle_indices,
                image_indices=image_indices,
            )


# ---------------------------------------------------------------------------
# ModelState — current reconstruction state (dynamic arrays)
# ---------------------------------------------------------------------------

class ModelState(eqx.Module):
    """Current reconstruction state passed to jitted functions.

    Contains the mean estimate, volume mask, and optionally the PCA basis
    and eigenvalues used for covariance/embedding computations.
    """

    mean_estimate: jax.Array
    volume_mask: jax.Array
    basis: Optional[jax.Array] = None
    eigenvalues: Optional[jax.Array] = None


# ---------------------------------------------------------------------------
# Per-function option modules (all static — controls compilation)
# ---------------------------------------------------------------------------

class CovColumnOpts(eqx.Module):
    """Static options for covariance column (H/B) computation."""

    right_kernel: str = eqx.field(static=True, default="triangular")
    left_kernel: str = eqx.field(static=True, default="triangular")
    right_kernel_width: int = eqx.field(static=True, default=2)
    mask_images: bool = eqx.field(static=True, default=True)
    soften_mask: int = eqx.field(static=True, default=3)


class CovarianceOpts(eqx.Module):
    """Options for covariance estimation inner functions."""

    disc_type_u: str = eqx.field(static=True)
    do_mask_images: bool = eqx.field(static=True, default=True)
    shared_label: bool = eqx.field(static=True, default=False)
    soften: int = eqx.field(static=True, default=5)


class EmbeddingOpts(eqx.Module):
    """Options for embedding / latent coordinate computation."""

    compute_covariances: bool = eqx.field(static=True, default=False)
    compute_bias: bool = eqx.field(static=True, default=False)
    shared_label: bool = eqx.field(static=True, default=False)
    contrast_shared_across_tilt_series: bool = eqx.field(static=True, default=True)
