"""Joint augmented ``[mu, W]`` M-step solvers for PPCA refinement."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .pose_accumulators import AugmentedPPCAStats
from .triangular import _tri_size, unpack_tri_to_full


@dataclass(frozen=True)
class AugmentedPPCAObjective:
    """Components of one fixed-statistics augmented PPCA M-step objective."""

    data_linear: jax.Array
    data_quadratic: jax.Array
    mean_prior: jax.Array
    W_prior: jax.Array

    @property
    def data(self):
        return self.data_linear + self.data_quadratic

    @property
    def prior(self):
        return self.mean_prior + self.W_prior

    @property
    def total(self):
        return self.data + self.prior

    def diagnostics(self, prefix: str, *, n_images: int | None = None) -> dict[str, float]:
        """Return flat JSON-friendly diagnostics with a shared prefix."""

        values = {
            f"{prefix}_total": float(self.total),
            f"{prefix}_data": float(self.data),
            f"{prefix}_data_linear": float(self.data_linear),
            f"{prefix}_data_quadratic": float(self.data_quadratic),
            f"{prefix}_prior": float(self.prior),
            f"{prefix}_mean_prior": float(self.mean_prior),
            f"{prefix}_W_prior": float(self.W_prior),
        }
        if n_images is not None and int(n_images) > 0:
            denom = float(n_images)
            values.update({f"{key}_per_image": value / denom for key, value in list(values.items())})
        return values


def _mean_regularization(
    n_frequency: int,
    *,
    mean_prior,
    mean_precision,
    reg_floor: float,
):
    if mean_precision is None:
        if mean_prior is None:
            raise ValueError("mean_prior is required when mean_precision is not provided")
        mean_prior = jnp.asarray(mean_prior)
        if mean_prior.shape != (n_frequency,):
            raise ValueError(f"mean_prior shape {mean_prior.shape} != ({n_frequency},)")
        return 1.0 / (mean_prior + reg_floor)
    mean_reg = jnp.asarray(mean_precision).real
    if mean_reg.shape != (n_frequency,):
        raise ValueError(f"mean_precision shape {mean_reg.shape} != ({n_frequency},)")
    return mean_reg


def augmented_ppca_mstep_objective(
    stats: AugmentedPPCAStats,
    mu,
    W,
    *,
    mean_prior=None,
    W_prior,
    mean_precision=None,
    reg_floor: float = 1e-16,
    chunk_size: int | None = None,
) -> AugmentedPPCAObjective:
    """Evaluate the fixed-statistics augmented PPCA M-step objective.

    This is the EM lower-bound term optimized by ``solve_augmented_ppca_mstep``
    for one already accumulated set of sufficient statistics:

    ``Re(theta^H rhs) - 0.5 theta^H lhs theta - 0.5 theta^H precision theta``.

    It intentionally excludes pose log normalizers, constants, and
    post-solve masking/grid-correction heuristics. Values are comparable within
    one iteration for the input, raw solved, and final scoring models, but not
    across iterations with different posterior supports/statistics.
    """

    rhs = jnp.asarray(stats.rhs)
    lhs_tri = jnp.asarray(stats.lhs_tri)
    W_prior = jnp.asarray(W_prior)
    mu = jnp.asarray(mu)
    W = jnp.asarray(W)

    if rhs.ndim != 2:
        raise ValueError(f"stats.rhs must have shape [n_frequency, q+1], got {rhs.shape}")
    n_frequency, p = rhs.shape
    q = p - 1
    if lhs_tri.shape != (n_frequency, _tri_size(p)):
        raise ValueError(f"stats.lhs_tri shape {lhs_tri.shape} does not match [n_frequency, tri({p})]")
    if mu.shape != (n_frequency,):
        raise ValueError(f"mu shape {mu.shape} != ({n_frequency},)")
    if W.shape != (n_frequency, q):
        raise ValueError(f"W shape {W.shape} != ({n_frequency}, {q})")
    if W_prior.shape != (n_frequency, q):
        raise ValueError(f"W_prior shape {W_prior.shape} != ({n_frequency}, {q})")
    mean_reg = _mean_regularization(
        n_frequency,
        mean_prior=mean_prior,
        mean_precision=mean_precision,
        reg_floor=reg_floor,
    )
    theta = jnp.concatenate([mu[:, None], W], axis=1)

    def _evaluate_slice(rhs_s, lhs_tri_s, theta_s, mean_reg_s, W_prior_s):
        lhs = unpack_tri_to_full(lhs_tri_s, p, hermitian=True)
        lhs = 0.5 * (lhs + jnp.swapaxes(jnp.conj(lhs), -1, -2))
        lhs_theta = jnp.einsum("fij,fj->fi", lhs, theta_s)
        data_linear = jnp.real(jnp.sum(jnp.conj(theta_s) * rhs_s))
        data_quadratic = -0.5 * jnp.real(jnp.sum(jnp.conj(theta_s) * lhs_theta))
        mean_prior_term = -0.5 * jnp.sum(mean_reg_s * (jnp.abs(theta_s[:, 0]) ** 2))
        if q:
            W_reg = 1.0 / (W_prior_s + reg_floor)
            W_prior_term = -0.5 * jnp.sum(W_reg * (jnp.abs(theta_s[:, 1:]) ** 2))
        else:
            W_prior_term = jnp.asarray(0.0, dtype=mean_prior_term.dtype)
        return data_linear, data_quadratic, mean_prior_term, W_prior_term

    if chunk_size is None:
        parts = _evaluate_slice(rhs, lhs_tri, theta, mean_reg, W_prior)
    else:
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive or None, got {chunk_size}")
        accum = None
        for start in range(0, n_frequency, chunk_size):
            end = min(start + chunk_size, n_frequency)
            chunk = _evaluate_slice(
                rhs[start:end],
                lhs_tri[start:end],
                theta[start:end],
                mean_reg[start:end],
                W_prior[start:end],
            )
            accum = chunk if accum is None else tuple(a + c for a, c in zip(accum, chunk, strict=True))
        if accum is None:
            zero = jnp.asarray(0.0, dtype=rhs.real.dtype)
            accum = (zero, zero, zero, zero)
        parts = accum
    return AugmentedPPCAObjective(*parts)


def solve_augmented_ppca_mstep(
    stats: AugmentedPPCAStats,
    *,
    mean_prior=None,
    W_prior,
    mean_precision=None,
    fixed_mean=None,
    reg_floor: float = 1e-16,
    chunk_size: int | None = None,
):
    """Solve per-frequency augmented normal equations.

    This foundation solver is intentionally direct and small: for each
    frequency it solves

    ``(lhs_aug + diag([mean_precision, 1/W_prior_1, ...])) theta = rhs``.

    If ``fixed_mean`` is provided, component 0 is held fixed and only the
    loading columns are solved from the conditional normal equation

    ``(lhs_WW + W_precision) W = rhs_W - lhs_Wmu * fixed_mean``.

    When ``mean_precision`` is omitted, the legacy variance-like
    ``mean_prior`` is inverted to form the mean precision. Dataset-level PPCA
    refinement can pass a precomputed RELION/K-class mean precision while
    keeping the PPCA loading prior separate.

    It is suitable for unit-sized stats and pins the required joint algebra.
    The production volume solver should replace this with the existing masked
    PCG path once dense PPCA backprojection is integrated.
    """
    rhs = jnp.asarray(stats.rhs)
    lhs_tri = jnp.asarray(stats.lhs_tri)
    W_prior = jnp.asarray(W_prior)

    if rhs.ndim != 2:
        raise ValueError(f"stats.rhs must have shape [n_frequency, q+1], got {rhs.shape}")
    n_frequency, p = rhs.shape
    q = p - 1
    if p < 1:
        raise ValueError("augmented component count must be at least 1")
    if lhs_tri.shape != (n_frequency, _tri_size(p)):
        raise ValueError(f"stats.lhs_tri shape {lhs_tri.shape} does not match [n_frequency, tri({p})]")
    if W_prior.shape != (n_frequency, q):
        raise ValueError(f"W_prior shape {W_prior.shape} != ({n_frequency}, {q})")
    mean_reg = _mean_regularization(
        n_frequency,
        mean_prior=mean_prior,
        mean_precision=mean_precision,
        reg_floor=reg_floor,
    )
    if fixed_mean is not None:
        fixed_mean = jnp.asarray(fixed_mean)
        if fixed_mean.shape != (n_frequency,):
            raise ValueError(f"fixed_mean shape {fixed_mean.shape} != ({n_frequency},)")

    def _solve_slice(rhs_s, lhs_tri_s, mean_reg_s, W_prior_s):
        lhs = unpack_tri_to_full(lhs_tri_s, p, hermitian=True)
        lhs = 0.5 * (lhs + jnp.swapaxes(jnp.conj(lhs), -1, -2))
        W_reg = 1.0 / (W_prior_s + reg_floor)
        reg = jnp.concatenate([mean_reg_s[:, None], W_reg], axis=1).astype(lhs.real.dtype)
        lhs_reg = lhs + jnp.eye(p, dtype=lhs.dtype)[None, :, :] * reg[:, None, :]
        return jax.vmap(jnp.linalg.solve)(lhs_reg, rhs_s)

    def _solve_fixed_mean_slice(rhs_s, lhs_tri_s, fixed_mean_s, W_prior_s):
        if q == 0:
            return fixed_mean_s, jnp.zeros((rhs_s.shape[0], 0), dtype=rhs_s.dtype)
        lhs = unpack_tri_to_full(lhs_tri_s, p, hermitian=True)
        lhs = 0.5 * (lhs + jnp.swapaxes(jnp.conj(lhs), -1, -2))
        lhs_wmu = lhs[:, 1:, 0]
        lhs_ww = lhs[:, 1:, 1:]
        W_reg = 1.0 / (W_prior_s + reg_floor)
        lhs_ww_reg = lhs_ww + jnp.eye(q, dtype=lhs_ww.dtype)[None, :, :] * W_reg[:, None, :]
        rhs_cond = rhs_s[:, 1:] - lhs_wmu * fixed_mean_s[:, None]
        W_theta = jax.vmap(jnp.linalg.solve)(lhs_ww_reg, rhs_cond)
        return fixed_mean_s, W_theta

    if fixed_mean is not None:
        if chunk_size is None:
            return _solve_fixed_mean_slice(rhs, lhs_tri, fixed_mean, W_prior)
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive or None, got {chunk_size}")
        mu_chunks = []
        W_chunks = []
        for start in range(0, n_frequency, chunk_size):
            end = min(start + chunk_size, n_frequency)
            mu_chunk, W_chunk = _solve_fixed_mean_slice(
                rhs[start:end],
                lhs_tri[start:end],
                fixed_mean[start:end],
                W_prior[start:end],
            )
            mu_chunks.append(mu_chunk)
            W_chunks.append(W_chunk)
        mu = jnp.concatenate(mu_chunks, axis=0) if mu_chunks else jnp.zeros((0,), dtype=rhs.dtype)
        W = jnp.concatenate(W_chunks, axis=0) if W_chunks else jnp.zeros((0, q), dtype=rhs.dtype)
        return mu, W

    if chunk_size is None:
        theta = _solve_slice(rhs, lhs_tri, mean_reg, W_prior)
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
                    mean_reg[start:end],
                    W_prior[start:end],
                )
            )
        theta = jnp.concatenate(chunks, axis=0) if chunks else jnp.zeros_like(rhs)
    return theta[:, 0], theta[:, 1:]
