"""Mean regularization helpers for PPCA refinement.

PPCA solves an augmented ``[mu, W]`` system, but component 0 is still the
homogeneous mean. Its diagonal regularization should therefore follow the same
RELION/K-class tau convention as the K=1/Class3D mean solve. The PPCA loading
columns deliberately keep their separate variance-like W prior.
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.ppca import AugmentedPPCAStats
from recovar.reconstruction import relion_functions


KCLASS_RELION_MINRES_MAP = 5


def _coerce_tau_for_half_relion(tau, volume_shape):
    """Return full-layout tau accepted by RELION's half-volume filter path."""

    volume_shape = tuple(int(s) for s in volume_shape)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_size = int(np.prod(half_shape))
    full_size = int(np.prod(volume_shape))

    tau_arr = jnp.asarray(tau).real.reshape(-1)
    if tau_arr.size == full_size:
        return tau_arr
    if tau_arr.size == half_size:
        tau_half = tau_arr.reshape(half_shape)
    else:
        # Treat any other 1-D input as a radial shell tau. JAX's indexing
        # clamps beyond the last shell, matching the existing RELION helper.
        tau_half = utils.make_radial_image_half(tau_arr, volume_shape).reshape(half_shape)
    return ftu.half_volume_to_full_volume(tau_half, volume_shape).reshape(-1).real


def relion_style_mean_precision_from_filter(
    mean_filter,
    mean_tau,
    volume_shape,
    *,
    padding_factor: int = 1,
    tau2_fudge: float = 1.0,
    minres_map: int = KCLASS_RELION_MINRES_MAP,
):
    """Compute the K-class/RELION mean precision added to a PPCA mean row.

    Parameters
    ----------
    mean_filter
        Component-0 diagonal of the PPCA backprojection lhs in packed
        half-Fourier layout.
    mean_tau
        Mean signal variance/tau. Accepts full Fourier, packed half-Fourier, or
        radial shell input. This is variance-like, not precision-like.
    volume_shape
        Native real-space volume shape.

    Returns
    -------
    jax.Array
        Packed half-Fourier precision to add to the augmented M-step's mean
        diagonal. It includes RELION's tau2_fudge, minres_map, and denominator
        shell floor behavior. Real-space masking/grid correction remain outside
        this frequency-local augmented solve, just as W regularization remains
        a separate PPCA prior.
    """

    volume_shape = tuple(int(s) for s in volume_shape)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_size = int(np.prod(half_shape))
    mean_filter = jnp.asarray(mean_filter).real.reshape(-1)
    if mean_filter.shape != (half_size,):
        raise ValueError(f"mean_filter shape {mean_filter.shape} != ({half_size},)")

    tau_for_relion = _coerce_tau_for_half_relion(mean_tau, volume_shape)
    regularized_filter = relion_functions.adjust_regularization_relion_style(
        mean_filter,
        volume_shape,
        tau=tau_for_relion,
        padding_factor=int(padding_factor),
        max_res_shell=None,
        half_volume=True,
        tau2_fudge=float(tau2_fudge),
        minres_map=int(minres_map),
    ).reshape(-1)
    # The shell floor can also increase the denominator; the augmented solve
    # needs the full additive precision, not only 1/tau.
    return jnp.maximum(regularized_filter - mean_filter, 0.0).astype(mean_filter.dtype)


def relion_style_mean_precision_from_stats(
    stats: AugmentedPPCAStats,
    mean_tau,
    volume_shape,
    *,
    padding_factor: int = 1,
    tau2_fudge: float = 1.0,
    minres_map: int = KCLASS_RELION_MINRES_MAP,
):
    """Compute RELION/K-class mean precision from PPCA accumulated stats."""

    lhs_tri = jnp.asarray(stats.lhs_tri)
    if lhs_tri.ndim != 2 or lhs_tri.shape[1] < 1:
        raise ValueError(f"stats.lhs_tri must have shape [n_frequency, tri], got {lhs_tri.shape}")
    return relion_style_mean_precision_from_filter(
        lhs_tri[:, 0].real,
        mean_tau,
        volume_shape,
        padding_factor=padding_factor,
        tau2_fudge=tau2_fudge,
        minres_map=minres_map,
    )
