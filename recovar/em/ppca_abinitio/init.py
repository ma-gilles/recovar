"""Initializers for the PPCA-ab-initio v0 stages.

Per spec Section 9.5:

- `init_truth_perturbed(gt, eps_mu, eps_U)` — positive control for
  Stage 1B/1C. Perturbations are added in real space and re-encoded
  to half-volume layout, so the half-volume invariant is preserved
  automatically.
- `init_random_lowpass(volume_shape, q, k_max, seed)` — stress
  control / negative initializer. Generates real-space volumes,
  FT to half-volume, band-limit. Used as a stress test in Stage 1A
  and Phase 2.

External-mean and atlas initializers are deferred to Phase 2/3.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu

from .half_volume import (
    make_half_volume_weights,
    radial_band_limit_half,
    real_volume_orthonormalize_half,
)
from .types import PPCAInit


def _real_volume_to_half_flat(real_vol):
    return ftu.get_dft3_real(jnp.asarray(real_vol)).reshape(-1)


def _half_to_real_volume(v_half_flat, volume_shape):
    return ftu.get_idft3_real(
        v_half_flat.reshape((volume_shape[0], volume_shape[1], volume_shape[2] // 2 + 1)),
        volume_shape=tuple(volume_shape),
    )


def init_truth_perturbed(
    *,
    mu_half_true: jnp.ndarray,
    U_half_true: jnp.ndarray,
    s_true: jnp.ndarray,
    volume_shape,
    eps_mu: float,
    eps_U: float,
    seed: int = 0,
) -> PPCAInit:
    """Perturb (mu, U) in **real space** and re-encode to half-volume.

    Per spec Section 9.5: perturbing in real space and round-tripping
    through `get_dft3_real` preserves the structural Hermitian
    symmetry of the half-volume layout automatically — no projection
    step is needed.

    The perturbation magnitudes `eps_mu` and `eps_U` are expressed
    as **fractions of the L2 norm** of the corresponding ground
    truth in real space.
    """
    rng = np.random.default_rng(seed)
    q = int(U_half_true.shape[0])

    mu_real = np.asarray(_half_to_real_volume(mu_half_true, volume_shape), dtype=np.float64)
    mu_norm = float(np.linalg.norm(mu_real))
    mu_pert_real = mu_real + eps_mu * mu_norm * rng.standard_normal(mu_real.shape) / np.sqrt(mu_real.size)

    U_real = np.stack(
        [np.asarray(_half_to_real_volume(U_half_true[k], volume_shape), dtype=np.float64) for k in range(q)]
    )
    U_pert_real = np.empty_like(U_real)
    for k in range(q):
        u_norm = float(np.linalg.norm(U_real[k]))
        U_pert_real[k] = U_real[k] + eps_U * u_norm * rng.standard_normal(U_real[k].shape) / np.sqrt(U_real[k].size)

    mu_half_pert = jnp.asarray(_real_volume_to_half_flat(mu_pert_real), dtype=jnp.complex128)
    U_half_pert = jnp.stack([_real_volume_to_half_flat(U_pert_real[k]) for k in range(q)]).astype(jnp.complex128)

    return PPCAInit(
        mu=mu_half_pert,
        U=U_half_pert,
        s=jnp.asarray(s_true, dtype=jnp.float64),
        volume_shape=tuple(int(x) for x in volume_shape),
    )


def init_random_lowpass(
    *,
    volume_shape,
    q: int,
    k_max: float,
    s_init: jnp.ndarray | None = None,
    seed: int = 0,
    orthonormalize: bool = True,
) -> PPCAInit:
    """Random low-pass initializer.

    Generates `q` real-space random volumes, FFTs them to half-volume
    layout, applies a radial band-limit at `k_max`, and (optionally)
    orthonormalizes the rows in the real-space sense.

    `mu` is initialized as the zero half-volume; the loop is expected
    to update it from data. `s_init` defaults to `1/(k+1)` for
    `k = 0..q-1`.
    """
    rng = np.random.default_rng(seed)

    # mu = 0
    half_shape = (volume_shape[0], volume_shape[1], volume_shape[2] // 2 + 1)
    half_size = int(np.prod(half_shape))
    mu_half = jnp.zeros(half_size, dtype=jnp.complex128)

    # U: random real-space volumes → FT → band-limit → orthonormalize
    U_real = rng.standard_normal((q,) + tuple(volume_shape)).astype(np.float64)
    U_half = jnp.stack([_real_volume_to_half_flat(U_real[k]) for k in range(q)]).astype(jnp.complex128)
    U_half = radial_band_limit_half(U_half, volume_shape, k_max)

    if orthonormalize:
        weights = make_half_volume_weights(volume_shape)
        N_full = int(np.prod(volume_shape))
        U_half = real_volume_orthonormalize_half(U_half, weights, N_full)

    if s_init is None:
        s_init_arr = (1.0 / (np.arange(q) + 1.0)).astype(np.float64)
    else:
        s_init_arr = np.asarray(s_init, dtype=np.float64)

    return PPCAInit(
        mu=mu_half,
        U=U_half,
        s=jnp.asarray(s_init_arr, dtype=jnp.float64),
        volume_shape=tuple(int(x) for x in volume_shape),
    )


def init_oracle(
    *,
    mu_half_true: jnp.ndarray,
    U_half_true: jnp.ndarray,
    s_true: jnp.ndarray,
    volume_shape,
) -> PPCAInit:
    """Oracle initializer: use the ground-truth (mu, U, s) directly.

    Used for the Stage 0B oracle-score diagnostic. This is not a
    realistic initializer — it exists so that the score-only test
    can answer the question "with the *true* factors, does PPCA
    score better than homogeneous?" without conflating that with
    factor learning.
    """
    return PPCAInit(
        mu=jnp.asarray(mu_half_true, dtype=jnp.complex128),
        U=jnp.asarray(U_half_true, dtype=jnp.complex128),
        s=jnp.asarray(s_true, dtype=jnp.float64),
        volume_shape=tuple(int(x) for x in volume_shape),
    )
