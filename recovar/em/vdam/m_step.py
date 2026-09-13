"""VDAM M-step: gradient moment update + reference reconstruction.

Defaults to the RELION C++ moment/reconstruction transaction. The opt-in JAX
backend retains the same state layout and native diagnostic boundaries.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from recovar.em.vdam.state import InitialModelState, VdamAccumulator
from recovar.em.vdam.mstep_single_class import vdam_m_step_single_class


def relion_solvent_mask(
    *,
    ori_size: int,
    pixel_size: float,
    particle_diameter_ang: float,
    width_mask_edge_px: float,
) -> np.ndarray:
    """Return RELION's centered spherical ``solventFlatten`` mask."""
    if particle_diameter_ang <= 0.0:
        raise ValueError(f"particle_diameter_ang must be positive, got {particle_diameter_ang}")
    if width_mask_edge_px < 0.0:
        raise ValueError(f"width_mask_edge_px must be non-negative, got {width_mask_edge_px}")
    if pixel_size <= 0.0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}")

    n = int(ori_size)
    radius = float(particle_diameter_ang) / (2.0 * float(pixel_size))
    width = float(width_mask_edge_px)
    radius_p = radius + width

    coords = np.arange(-(n // 2), n - (n // 2), dtype=np.float64)
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)

    mask = np.zeros((n, n, n), dtype=np.float64)
    mask[r < radius] = 1.0
    if width > 0.0:
        edge = (r >= radius) & (r <= radius_p)
        mask[edge] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - r[edge]) / width)

    return mask


def relion_solvent_flatten_state(
    state: InitialModelState,
    *,
    particle_diameter_ang: float | None = None,
    width_mask_edge_px: float | None = None,
    mask: np.ndarray | None = None,
    compute_dtype: Literal["float32", "float64"] = "float64",
) -> InitialModelState:
    """Apply RELION's spherical ``solventFlatten`` mask to all references (post-maximization)."""
    if compute_dtype not in {"float32", "float64"}:
        raise ValueError(f"Unknown solvent compute_dtype: {compute_dtype!r}")
    iref = np.asarray(state.Iref)
    if compute_dtype == "float32" and iref.dtype != np.dtype(np.float32):
        raise ValueError("float32 solvent multiplication requires float32 state.Iref")
    if iref.ndim != 4 or iref.shape[1:] != (state.ori_size,) * 3:
        raise ValueError(f"state.Iref must have shape (K, {state.ori_size}, ...), got {iref.shape}")
    if mask is None:
        if particle_diameter_ang is None or width_mask_edge_px is None:
            raise ValueError("particle_diameter_ang and width_mask_edge_px are required when mask is not provided")
        mask = relion_solvent_mask(
            ori_size=int(state.ori_size),
            pixel_size=float(state.pixel_size),
            particle_diameter_ang=float(particle_diameter_ang),
            width_mask_edge_px=float(width_mask_edge_px),
        )
    mask = np.asarray(mask, dtype=np.dtype(compute_dtype))
    if mask.shape != (state.ori_size,) * 3:
        raise ValueError(f"mask must have shape ({state.ori_size},)*3, got {mask.shape}")

    return replace(state, Iref=(iref * mask[None, :, :, :]).astype(iref.dtype, copy=False))

def vdam_m_step(
    state: InitialModelState,
    accumulators: list[VdamAccumulator],
    *,
    grad_current_stepsize: float,
    tau2_fudge_factor: float,
    grad_min_resol_shell: float | None = None,
    padding_factor: int = 1,
    use_native_transaction: bool = True,
    mstep_backend: Literal["native", "jax"] = "native",
    mstep_compute_dtype: Literal["float32", "float64"] = "float64",
) -> InitialModelState:
    """Full VDAM M-step over K classes.

    ``accumulators`` holds ``2K`` entries when ``pseudo_halfsets`` is active
    (halfset 0 of each class first, then halfset 1), else ``K``.
    """
    K = state.K
    expected = 2 * K if state.pseudo_halfsets else K
    if len(accumulators) != expected:
        raise ValueError(f"expected {expected} accumulators, got {len(accumulators)}")

    out = state
    for k in range(K):
        out = vdam_m_step_single_class(
            out,
            k=k,
            accum_h0=accumulators[k],
            accum_h1=accumulators[K + k] if state.pseudo_halfsets else None,
            grad_current_stepsize=grad_current_stepsize,
            tau2_fudge_factor=tau2_fudge_factor,
            grad_min_resol_shell=grad_min_resol_shell,
            padding_factor=padding_factor,
            use_native_transaction=use_native_transaction,
            mstep_backend=mstep_backend,
            mstep_compute_dtype=mstep_compute_dtype,
        )
    return out
