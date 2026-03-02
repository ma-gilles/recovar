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

logger = logging.getLogger(__name__)


def _resolve_ctf_half(ctf_fun):
    """Resolve the native half-grid CTF function for a given full CTF function.

    Checks for a ``_half_variant`` attribute (set by closure factories like
    ``get_cryo_ET_CTF_fun``), then falls back to a lookup table of known
    module-level CTF functions.  Returns ``None`` if no half variant is found
    (caller should fall back to full → half extraction).
    """
    half = getattr(ctf_fun, '_half_variant', None)
    if half is not None:
        return half

    from recovar.core import ctf as ctf_mod
    _KNOWN_HALF_MAP = {
        ctf_mod.cryodrgn_CTF: ctf_mod.cryodrgn_CTF_half,
        ctf_mod.evaluate_ctf_wrapper: ctf_mod.evaluate_ctf_wrapper_half,
        ctf_mod.evaluate_ctf_wrapper_tilt_series_v2: ctf_mod.evaluate_ctf_wrapper_tilt_series_v2_half,
        ctf_mod.evaluate_ctf_wrapper_tilt_series: ctf_mod.evaluate_ctf_wrapper_tilt_series_half,
    }
    return _KNOWN_HALF_MAP.get(ctf_fun, None)


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
    CTF_fun: Callable = eqx.field(static=True)
    CTF_fun_half: Optional[Callable] = eqx.field(static=True, default=None)
    premultiplied_ctf: bool = eqx.field(static=True, default=False)
    volume_mask_threshold: float = eqx.field(static=True, default=0.0)
    volume_upsampling_factor: int = eqx.field(static=True, default=1)
    data_multiplier: float = eqx.field(static=True, default=1.0)
    process_fn: Optional[Callable] = eqx.field(static=True, default=None)

    def compute_ctf(self, ctf_params: jax.Array) -> jax.Array:
        """Compute CTF values for a batch of images (full spectrum)."""
        return self.CTF_fun(ctf_params, self.image_shape, self.voxel_size)

    def compute_ctf_half(self, ctf_params: jax.Array) -> jax.Array:
        """Compute CTF at half-spectrum (rfft-packed) frequencies.

        Uses the native half-grid CTF function when available, otherwise
        falls back to computing full CTF then extracting the half-spectrum.
        """
        if self.CTF_fun_half is not None:
            return self.CTF_fun_half(ctf_params, self.image_shape, self.voxel_size)
        import recovar.core.fourier_transform_utils as ftu
        full_ctf = self.CTF_fun(ctf_params, self.image_shape, self.voxel_size)
        return ftu.full_image_to_half_image(full_ctf, self.image_shape)

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
            CTF_fun=self.CTF_fun, CTF_fun_half=self.CTF_fun_half,
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
        use_upsampled: bool = False,
        upsampling_factor: Optional[int] = None,
    ) -> ForwardModelConfig:
        """Create from a CryoEMDataset or CryoEMHalfsets instance.

        Parameters
        ----------
        cryo : CryoEMDataset or CryoEMHalfsets
            Source dataset for geometry and CTF configuration.
        disc_type : str
            Discretization type (e.g. 'linear_interp', 'cubic', '').
        process_fn : callable, optional
            Image preprocessing function applied inside jitted code.
        use_upsampled : bool
            If True, use the upsampled volume shape/grid_size from the
            dataset (reads ``cryo.volume_upsampling_factor``).
            Prefer ``upsampling_factor`` for new code.
        upsampling_factor : int, optional
            Volume upsampling factor (e.g. 2 for 2× oversampled grid).
            Computes the upsampled volume shape directly without
            mutating the dataset object.  Cannot be combined with
            ``use_upsampled=True``.
        """
        if use_upsampled and upsampling_factor is not None:
            raise ValueError(
                "Cannot specify both use_upsampled=True and upsampling_factor. "
                "Prefer upsampling_factor for new code."
            )

        # CryoEMHalfsets.CTF_fun is a method; we need the underlying callable
        # that takes (ctf_params, image_shape, voxel_size) directly.
        from recovar.data_io.dataset import CryoEMDataset, CryoEMHalfsets

        if isinstance(cryo, CryoEMHalfsets):
            ctf_fun = cryo[0].CTF_fun
        elif isinstance(cryo, CryoEMDataset):
            ctf_fun = cryo.CTF_fun
        else:
            # Duck-type: assume it has the right attributes
            ctf_fun = cryo.CTF_fun

        base_grid_size = int(cryo.grid_size)

        if upsampling_factor is not None:
            volume_upsampling = int(upsampling_factor)
            grid_size = base_grid_size * volume_upsampling
            volume_shape = (grid_size, grid_size, grid_size)
        elif use_upsampled:
            volume_shape = tuple(int(x) for x in cryo.upsampled_volume_shape)
            grid_size = int(cryo.upsampled_grid_size)
            volume_upsampling = int(getattr(cryo, 'volume_upsampling_factor', 1))
        else:
            volume_shape = tuple(int(x) for x in cryo.volume_shape)
            grid_size = base_grid_size
            volume_upsampling = 1

        # Resolve native half-grid CTF function
        ctf_fun_half = _resolve_ctf_half(ctf_fun)

        config = cls(
            image_shape=tuple(int(x) for x in cryo.image_shape),
            volume_shape=volume_shape,
            grid_size=grid_size,
            voxel_size=float(cryo.voxel_size),
            padding=int(getattr(cryo, 'padding', 0)),
            disc_type=disc_type,
            CTF_fun=ctf_fun,
            CTF_fun_half=ctf_fun_half,
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

class CovarianceOpts(eqx.Module):
    """Options for covariance estimation inner functions."""

    disc_type_u: str = eqx.field(static=True)
    do_mask_images: bool = eqx.field(static=True, default=True)
    shared_label: bool = eqx.field(static=True, default=False)
    parallel_analysis: bool = eqx.field(static=True, default=False)
    soften: int = eqx.field(static=True, default=5)


class EmbeddingOpts(eqx.Module):
    """Options for embedding / latent coordinate computation."""

    compute_covariances: bool = eqx.field(static=True, default=False)
    compute_bias: bool = eqx.field(static=True, default=False)
    shared_label: bool = eqx.field(static=True, default=False)
    contrast_shared_across_tilt_series: bool = eqx.field(static=True, default=True)
