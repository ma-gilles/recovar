"""Joint augmented ``[mu, W]`` M-step solvers for PPCA refinement."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .pose_accumulators import AugmentedPPCAStats
from .triangular import _tri_size, unpack_tri_to_full


def solve_augmented_ppca_mstep(
    stats: AugmentedPPCAStats,
    *,
    mean_prior,
    W_prior,
    reg_floor: float = 1e-16,
    chunk_size: int | None = None,
):
    """Solve per-frequency augmented normal equations.

    This foundation solver is intentionally direct and small: for each
    frequency it solves

    ``(lhs_aug + diag([1/mean_prior, 1/W_prior_1, ...])) theta = rhs``.

    It is suitable for unit-sized stats and pins the required joint algebra.
    The production volume solver should replace this with the existing masked
    PCG path once dense PPCA backprojection is integrated.
    """
    rhs = jnp.asarray(stats.rhs)
    lhs_tri = jnp.asarray(stats.lhs_tri)
    mean_prior = jnp.asarray(mean_prior)
    W_prior = jnp.asarray(W_prior)

    if rhs.ndim != 2:
        raise ValueError(f"stats.rhs must have shape [n_frequency, q+1], got {rhs.shape}")
    n_frequency, p = rhs.shape
    q = p - 1
    if p < 1:
        raise ValueError("augmented component count must be at least 1")
    if lhs_tri.shape != (n_frequency, _tri_size(p)):
        raise ValueError(f"stats.lhs_tri shape {lhs_tri.shape} does not match [n_frequency, tri({p})]")
    if mean_prior.shape != (n_frequency,):
        raise ValueError(f"mean_prior shape {mean_prior.shape} != ({n_frequency},)")
    if W_prior.shape != (n_frequency, q):
        raise ValueError(f"W_prior shape {W_prior.shape} != ({n_frequency}, {q})")

    def _solve_slice(rhs_s, lhs_tri_s, mean_prior_s, W_prior_s):
        lhs = unpack_tri_to_full(lhs_tri_s, p, hermitian=True)
        lhs = 0.5 * (lhs + jnp.swapaxes(jnp.conj(lhs), -1, -2))
        mean_reg = 1.0 / (mean_prior_s + reg_floor)
        W_reg = 1.0 / (W_prior_s + reg_floor)
        reg = jnp.concatenate([mean_reg[:, None], W_reg], axis=1).astype(lhs.real.dtype)
        lhs_reg = lhs + jnp.eye(p, dtype=lhs.dtype)[None, :, :] * reg[:, None, :]
        return jax.vmap(jnp.linalg.solve)(lhs_reg, rhs_s)

    if chunk_size is None:
        theta = _solve_slice(rhs, lhs_tri, mean_prior, W_prior)
    else:
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive or None, got {chunk_size}")
        chunks = []
        for start in range(0, n_frequency, chunk_size):
            end = min(start + chunk_size, n_frequency)
            chunks.append(
                _solve_slice(
                    rhs[start:end],
                    lhs_tri[start:end],
                    mean_prior[start:end],
                    W_prior[start:end],
                )
            )
        theta = jnp.concatenate(chunks, axis=0) if chunks else jnp.zeros_like(rhs)
    return theta[:, 0], theta[:, 1:]
