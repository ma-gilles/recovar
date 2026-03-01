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

from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import numpy as np


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
    premultiplied_ctf: bool = eqx.field(static=True)
    volume_mask_threshold: float = eqx.field(static=True)
    process_fn: Optional[Callable] = eqx.field(static=True, default=None)

    def compute_ctf(self, ctf_params: jax.Array) -> jax.Array:
        """Compute CTF values for a batch of images.

        Replaces the ubiquitous ``CTF_fun(CTF_params, image_shape, voxel_size)``
        pattern with ``config.compute_ctf(ctf_params)``.
        """
        return self.CTF_fun(ctf_params, self.image_shape, self.voxel_size)

    def replace(self, **kwargs) -> ForwardModelConfig:
        """Create a new config with some fields replaced.

        Useful for changing static fields (e.g. ``disc_type``) which cannot
        be modified via ``eqx.tree_at``.
        """
        fields = dict(
            image_shape=self.image_shape, volume_shape=self.volume_shape,
            grid_size=self.grid_size, voxel_size=self.voxel_size,
            padding=self.padding, disc_type=self.disc_type,
            CTF_fun=self.CTF_fun, premultiplied_ctf=self.premultiplied_ctf,
            volume_mask_threshold=self.volume_mask_threshold,
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
            If True, use the upsampled volume shape/grid_size for
            back-projection (needed by RELION-style reconstruction).
        """
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

        if use_upsampled:
            volume_shape = tuple(cryo.upsampled_volume_shape)
            grid_size = int(cryo.upsampled_grid_size)
        else:
            volume_shape = tuple(cryo.volume_shape)
            grid_size = int(cryo.grid_size)

        return cls(
            image_shape=tuple(cryo.image_shape),
            volume_shape=volume_shape,
            grid_size=grid_size,
            voxel_size=float(cryo.voxel_size),
            padding=int(getattr(cryo, 'padding', 0)),
            disc_type=disc_type,
            CTF_fun=ctf_fun,
            premultiplied_ctf=bool(getattr(cryo, 'premultiplied_ctf', False)),
            volume_mask_threshold=float(getattr(cryo, 'volume_mask_threshold', 0.0)),
            process_fn=process_fn,
        )


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
