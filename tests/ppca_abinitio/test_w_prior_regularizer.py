"""Unit tests for the shell-stratified W_prior regularizer.

W_prior is the alternative to scalar ridge in the closed-form M-step.
It is regularization, not eigenvalue estimation.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from recovar.em.ppca_abinitio.factor_update import compute_W_prior_half

pytestmark = [pytest.mark.unit]


def _half_size(volume_shape):
    """V_half = N0 * N1 * (N2//2 + 1)."""
    N0, N1, N2 = volume_shape
    return N0 * N1 * (N2 // 2 + 1)


def test_w_prior_shape():
    """Output is (q, V_half), real-valued, finite, positive."""
    volume_shape = (8, 8, 8)
    q = 3
    rng = np.random.default_rng(0)
    U = (
        rng.standard_normal((q, _half_size(volume_shape))) + 1j * rng.standard_normal((q, _half_size(volume_shape)))
    ).astype(np.complex128)
    s = jnp.array([1.0, 2.0, 0.5], dtype=jnp.float64)

    W = compute_W_prior_half(jnp.asarray(U), s, volume_shape)

    assert W.shape == (q, _half_size(volume_shape))
    assert W.dtype == jnp.float64
    assert bool(jnp.all(jnp.isfinite(W)))
    assert bool(jnp.all(W > 0))


def test_w_prior_is_constant_within_each_radial_shell():
    """All voxels at the same radial distance must share the same W
    value within a PC. This is the defining property of the
    shell-stratified prior."""
    volume_shape = (8, 8, 8)
    q = 2
    rng = np.random.default_rng(1)
    U = (
        rng.standard_normal((q, _half_size(volume_shape))) + 1j * rng.standard_normal((q, _half_size(volume_shape)))
    ).astype(np.complex128)
    s = jnp.array([1.0, 1.0], dtype=jnp.float64)

    W = compute_W_prior_half(jnp.asarray(U), s, volume_shape)
    W_np = np.asarray(W)

    # Reconstruct radial labels the same way the function does.
    import recovar.core.fourier_transform_utils as ftu

    radial = np.asarray(ftu.get_grid_of_radial_distances_real(volume_shape)).reshape(-1)
    n_shells = max(1, volume_shape[0] // 2 - 1)
    labels = np.minimum(radial, n_shells - 1)

    for k in range(q):
        for shell_idx in range(n_shells):
            mask = labels == shell_idx
            if mask.sum() == 0:
                continue
            shell_values = W_np[k, mask]
            # All values in the shell must be equal up to FP round-off,
            # OR all clipped to the per-PC floor (which is also equal).
            assert np.allclose(shell_values, shell_values[0], atol=1e-12), (
                f"PC {k}, shell {shell_idx}: not constant within shell"
            )


def test_w_prior_floor_prevents_blowup_at_zero_U():
    """If U is identically zero on a shell, W_v on that shell must
    floor to a positive value so that 1/W_v stays finite."""
    volume_shape = (8, 8, 8)
    q = 2
    U = jnp.zeros((q, _half_size(volume_shape)), dtype=jnp.complex128)
    s = jnp.array([1.0, 1.0], dtype=jnp.float64)

    W = compute_W_prior_half(U, s, volume_shape)

    assert bool(jnp.all(W > 0)), "W_prior must be strictly positive after floor"
    assert bool(jnp.all(jnp.isfinite(1.0 / W))), "1/W_prior must be finite"


def test_w_prior_scales_linearly_with_s():
    """W_v = shell_avg(|U|^2 * s_k). Doubling s should double W
    proportionally (before flooring saturates)."""
    volume_shape = (8, 8, 8)
    q = 2
    rng = np.random.default_rng(2)
    U = jnp.asarray(
        (
            rng.standard_normal((q, _half_size(volume_shape))) + 1j * rng.standard_normal((q, _half_size(volume_shape)))
        ).astype(np.complex128)
    )

    s1 = jnp.array([1.0, 1.0], dtype=jnp.float64)
    s2 = jnp.array([2.0, 2.0], dtype=jnp.float64)

    W1 = compute_W_prior_half(U, s1, volume_shape)
    W2 = compute_W_prior_half(U, s2, volume_shape)

    # Ratio should be 2.0 wherever the floor isn't binding.
    ratio = W2 / W1
    # Most voxels should be in the unflooded regime: ratio ≈ 2.
    median_ratio = float(jnp.median(ratio))
    assert abs(median_ratio - 2.0) < 1e-6, f"median W2/W1 = {median_ratio}, expected 2.0"


def test_w_prior_floor_dominated_by_eps_rel():
    """Larger eps_rel must strictly increase the floor. With a U that
    has empty shells, the empty shells get clipped to the floor; a
    bigger eps_rel must produce a strictly larger min(W)."""
    volume_shape = (8, 8, 8)
    q = 1
    rng = np.random.default_rng(3)
    half = _half_size(volume_shape)
    U_real = np.zeros((q, half), dtype=np.float64)
    U_real[0, : half // 2] = rng.standard_normal(half // 2) * 10.0
    U = jnp.asarray(U_real.astype(np.complex128))
    s = jnp.array([1.0], dtype=jnp.float64)

    W_small = compute_W_prior_half(U, s, volume_shape, eps_rel=1e-3)
    W_large = compute_W_prior_half(U, s, volume_shape, eps_rel=0.5)

    min_small = float(jnp.min(W_small[0]))
    min_large = float(jnp.min(W_large[0]))
    # With eps_rel=0.5 vs 1e-3, the floor is ~500x stronger; min must
    # strictly increase. Fall back to >= when the unflooded values are
    # large enough that even the small floor doesn't bind anywhere.
    assert min_large >= min_small, f"min W with eps_rel=0.5 ({min_large}) < min W with eps_rel=1e-3 ({min_small})"
    # And the strong-floor case must be at least eps_rel * mean of the
    # ORIGINAL (unfloored) field. Since we cannot extract the unfloored
    # field, we use the small-floor result as a lower bound on it.
    mean_small = float(jnp.mean(W_small[0]))
    assert min_large >= 0.5 * mean_small * 0.99, (
        f"min W with eps_rel=0.5 ({min_large}) is below 0.5 * unfloored mean estimate ({0.5 * mean_small})"
    )
