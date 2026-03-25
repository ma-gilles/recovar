"""Equinox-based configuration modules for clean parameter passing.

These modules bundle the many parameters that are passed to JAX-jitted functions
throughout RECOVAR into structured, typed containers. This eliminates the need for
complex ``static_argnums`` lists and keeps function signatures readable.
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
            image_shape=self.image_shape,
            volume_shape=self.volume_shape,
            grid_size=self.grid_size,
            voxel_size=self.voxel_size,
            padding=self.padding,
            disc_type=self.disc_type,
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
        from recovar.data_io.cryoem_dataset import CryoEMDataset

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
            padding=int(getattr(cryo, "padding", 0)),
            disc_type=disc_type,
            ctf=ctf_eval,
            premultiplied_ctf=bool(getattr(cryo, "premultiplied_ctf", False)),
            volume_mask_threshold=float(getattr(cryo, "volume_mask_threshold", 0.0)),
            volume_upsampling_factor=volume_upsampling,
            data_multiplier=float(getattr(cryo, "data_multiplier", 1.0)),
            process_fn=process_fn,
        )
        logger.debug(
            "ForwardModelConfig: grid=%d, image=%s, disc=%s, premult_ctf=%s",
            grid_size,
            config.image_shape,
            disc_type,
            config.premultiplied_ctf,
        )
        return config


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
