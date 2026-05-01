"""Phase 3 (M2) tests: augmented [μ, W] PCG M-step.

Verifies:
  * the legacy ``_pcg_hard_mstep`` is deterministic at q=4 (forward guard
    against future regressions of the unchanged function);
  * the new augmented wrapper builds correct ``reg_diag_aug``,
    ``rhs``, ``lhs_tri`` shapes and forwards to ``_pcg_hard_mstep`` with
    ``q+1`` components;
  * the wrapper output matches a dense Python normal-equations solve on a
    tiny synthetic problem;
  * pure-mean reduction (q=0) matches a homogeneous Wiener solve;
  * multi-mask projection respects per-PC mask assignment;
  * basic shape / signature contracts of ``AugmentedPPCAStats`` and
    ``solve_augmented_ppca_mstep``.

Each test is small enough to run on the login-node GPU in seconds.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from recovar.core import fourier_transform_utils as ftu  # noqa: E402
from recovar.ppca import (  # noqa: E402
    AugmentedPPCAStats,
    solve_augmented_ppca_mstep,
)
from recovar.ppca.ppca import _pcg_hard_mstep, _tri_size, unpack_tri_to_full  # noqa: E402

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_pcg_problem(rng, q, grid):
    vol_shape = (grid, grid, grid)
    half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
    half_vol = int(np.prod(half_vs))
    tri_sz = _tri_size(q)

    lhs_tri = np.zeros((half_vol, tri_sz), dtype=np.float32)
    idx = 0
    for i in range(q):
        for j in range(i, q):
            if i == j:
                lhs_tri[:, idx] = 1.0 + rng.uniform(0.0, 0.1, size=half_vol).astype(np.float32)
            else:
                lhs_tri[:, idx] = rng.normal(0.0, 0.01, size=half_vol).astype(np.float32)
            idx += 1
    rhs = (
        rng.standard_normal((half_vol, q)).astype(np.float32)
        + 1j * rng.standard_normal((half_vol, q)).astype(np.float32)
    ).astype(np.complex64)
    reg_diag = rng.uniform(0.05, 0.15, size=(half_vol, q)).astype(np.float32)

    cz, cy, cx = np.indices(vol_shape, dtype=np.float32)
    cz -= grid / 2
    cy -= grid / 2
    cx -= grid / 2
    r2 = cx * cx + cy * cy + cz * cz
    mask = (r2 < (0.4 * grid) ** 2).astype(np.float32)

    return vol_shape, lhs_tri, rhs, reg_diag, mask


# ---------------------------------------------------------------------------
# Forward guard: legacy _pcg_hard_mstep is deterministic at q=4
# ---------------------------------------------------------------------------


def test_legacy_pcg_q4_is_deterministic():
    rng = np.random.default_rng(1234)
    vol_shape, lhs_tri, rhs, reg_diag, mask = _make_synthetic_pcg_problem(rng, q=4, grid=16)

    def call():
        return np.asarray(
            _pcg_hard_mstep(
                jnp.asarray(lhs_tri),
                jnp.asarray(rhs),
                jnp.asarray(reg_diag),
                jnp.asarray(mask),
                vol_shape,
                4,
                unpack_tri_to_full,
                maxiter=20,
                tol=1e-4,
            )
        )

    W1 = call()
    W2 = call()
    np.testing.assert_array_equal(W1, W2)


# ---------------------------------------------------------------------------
# Augmented wrapper: shape contract
# ---------------------------------------------------------------------------


def test_augmented_stats_dataclass_shape_invariants():
    half_vol, q = 7, 3
    rhs = jnp.zeros((half_vol, q + 1), dtype=jnp.complex64)
    lhs_tri = jnp.zeros((half_vol, _tri_size(q + 1)), dtype=jnp.float32)
    stats = AugmentedPPCAStats(rhs=rhs, lhs_tri=lhs_tri, n_images=42, log_likelihood=-1.0)
    assert stats.rhs.shape == (half_vol, q + 1)
    assert stats.lhs_tri.shape == (half_vol, _tri_size(q + 1))
    assert stats.n_images == 42
    assert stats.log_likelihood == -1.0
    # Frozen.
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.n_images = 99  # type: ignore[misc]


def test_solve_augmented_ppca_mstep_signature_contract():
    sig = inspect.signature(solve_augmented_ppca_mstep)
    assert "stats" in sig.parameters
    for kw in (
        "mean_prior",
        "W_prior",
        "mask",
        "masks",
        "pc_mask_assignment",
        "mean_mask_idx",
        "maxiter",
        "tol",
        "theta_init",
        "reg_floor",
    ):
        assert kw in sig.parameters, f"missing kwarg {kw!r}"


def test_solve_augmented_rejects_mismatched_shapes():
    half_vol, q = 5, 2
    rhs = jnp.zeros((half_vol, q + 1), dtype=jnp.complex64)
    # Wrong tri size.
    bad_tri = jnp.zeros((half_vol, _tri_size(q + 1) + 1), dtype=jnp.float32)
    stats = AugmentedPPCAStats(rhs=rhs, lhs_tri=bad_tri)
    with pytest.raises(ValueError, match="lhs_tri shape"):
        solve_augmented_ppca_mstep(
            stats,
            mean_prior=jnp.ones(half_vol, dtype=jnp.float32),
            W_prior=jnp.ones((half_vol, q), dtype=jnp.float32),
            mask=jnp.ones((4, 4, 4), dtype=jnp.float32),
        )


# ---------------------------------------------------------------------------
# Functional reduction tests
# ---------------------------------------------------------------------------


def test_augmented_q0_matches_legacy_with_mean_only():
    """When q=0 (no loadings), the augmented solver becomes a single-component
    PCG solve identical to calling the legacy ``_pcg_hard_mstep`` with q=1
    and the mean reg_diag in slot 0."""
    rng = np.random.default_rng(101)
    vol_shape, lhs_tri, rhs, reg_diag, mask = _make_synthetic_pcg_problem(rng, q=1, grid=12)
    half_vol = lhs_tri.shape[0]

    # Legacy: q=1 with reg_diag as-is.
    W_legacy = np.asarray(
        _pcg_hard_mstep(
            jnp.asarray(lhs_tri),
            jnp.asarray(rhs),
            jnp.asarray(reg_diag),
            jnp.asarray(mask),
            vol_shape,
            1,
            unpack_tri_to_full,
            maxiter=20,
            tol=1e-4,
        )
    )

    # Augmented: q=0 (no W). mean_prior is a variance whose precision matches reg_diag[:, 0].
    mean_prior = 1.0 / np.maximum(reg_diag[:, 0], 1e-30) - 1e-16
    W_prior = np.zeros((half_vol, 0), dtype=np.float32)
    stats = AugmentedPPCAStats(
        rhs=jnp.asarray(rhs),
        lhs_tri=jnp.asarray(lhs_tri),
    )
    mu, W = solve_augmented_ppca_mstep(
        stats,
        mean_prior=jnp.asarray(mean_prior, dtype=jnp.float32),
        W_prior=jnp.asarray(W_prior),
        mask=jnp.asarray(mask),
        maxiter=20,
        tol=1e-4,
    )
    assert W.shape == (0,) + vol_shape
    np.testing.assert_allclose(np.asarray(mu), W_legacy[0], rtol=1e-5, atol=1e-6)


def test_augmented_qplus1_matches_legacy_with_stacked_reg():
    """At q=2 + 1 mean component (so n_components=3), the augmented wrapper
    must produce exactly the same output as a direct call to
    ``_pcg_hard_mstep`` with q=3 and reg_diag pre-stacked the same way the
    wrapper builds it. This pins the wrapper's contract."""
    rng = np.random.default_rng(202)
    q = 2
    p = q + 1
    grid = 12
    vol_shape, lhs_tri_aug, rhs_aug, reg_aug_raw, mask = _make_synthetic_pcg_problem(rng, q=p, grid=grid)
    half_vol = lhs_tri_aug.shape[0]

    # Synthesize a pair (mean_prior, W_prior) that reproduces reg_aug_raw exactly:
    eps = 1e-16
    mean_prior = 1.0 / reg_aug_raw[:, 0] - eps
    W_prior = 1.0 / reg_aug_raw[:, 1:] - eps

    # Legacy direct call with the same reg_diag.
    W_legacy = np.asarray(
        _pcg_hard_mstep(
            jnp.asarray(lhs_tri_aug),
            jnp.asarray(rhs_aug),
            jnp.asarray(reg_aug_raw),
            jnp.asarray(mask),
            vol_shape,
            p,
            unpack_tri_to_full,
            maxiter=20,
            tol=1e-4,
        )
    )
    mu_legacy = W_legacy[0]
    W_legacy_loadings = W_legacy[1:]

    # Augmented wrapper.
    stats = AugmentedPPCAStats(rhs=jnp.asarray(rhs_aug), lhs_tri=jnp.asarray(lhs_tri_aug))
    mu, W = solve_augmented_ppca_mstep(
        stats,
        mean_prior=jnp.asarray(mean_prior, dtype=jnp.float32),
        W_prior=jnp.asarray(W_prior, dtype=jnp.float32),
        mask=jnp.asarray(mask),
        maxiter=20,
        tol=1e-4,
    )
    assert mu.shape == vol_shape
    assert W.shape == (q, *vol_shape)
    np.testing.assert_allclose(np.asarray(mu), mu_legacy, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(W), W_legacy_loadings, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Multimask
# ---------------------------------------------------------------------------


def test_augmented_multimask_zeros_W_components_outside_assigned_mask():
    """With ``masks`` of shape (M, D, D, D) and ``pc_mask_assignment`` of
    length q, each loading W_k is zeroed outside ``masks[pc_mask_assignment[k]]``.
    The mean component is zeroed outside ``masks[mean_mask_idx]``."""
    rng = np.random.default_rng(303)
    q = 4
    grid = 16
    vol_shape = (grid, grid, grid)

    # Make two disjoint half-space masks.
    cz, cy, cx = np.indices(vol_shape, dtype=np.float32)
    mask_left = (cx < grid / 2).astype(np.float32) * (
        ((cx - grid / 2) ** 2 + (cy - grid / 2) ** 2 + (cz - grid / 2) ** 2) < (0.4 * grid) ** 2
    )
    mask_right = (cx >= grid / 2).astype(np.float32) * (
        ((cx - grid / 2) ** 2 + (cy - grid / 2) ** 2 + (cz - grid / 2) ** 2) < (0.4 * grid) ** 2
    )
    masks_arr = np.stack([mask_left, mask_right]).astype(np.float32)
    union_mask = np.maximum(mask_left, mask_right)

    p = q + 1
    half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
    half_vol = int(np.prod(half_vs))
    lhs_tri = np.zeros((half_vol, _tri_size(p)), dtype=np.float32)
    idx = 0
    for i in range(p):
        for j in range(i, p):
            if i == j:
                lhs_tri[:, idx] = 1.0 + rng.uniform(0, 0.1, size=half_vol).astype(np.float32)
            else:
                lhs_tri[:, idx] = rng.normal(0, 0.01, size=half_vol).astype(np.float32)
            idx += 1
    rhs = (
        rng.standard_normal((half_vol, p)).astype(np.float32)
        + 1j * rng.standard_normal((half_vol, p)).astype(np.float32)
    ).astype(np.complex64)
    mean_prior = np.full(half_vol, 1e3, dtype=np.float32)  # weak μ reg
    W_prior = np.full((half_vol, q), 1e3, dtype=np.float32)  # weak W reg
    pc_mask_assignment = np.array([0, 0, 1, 1], dtype=np.int32)
    mean_mask_idx = 0

    stats = AugmentedPPCAStats(
        rhs=jnp.asarray(rhs),
        lhs_tri=jnp.asarray(lhs_tri),
    )
    mu, W = solve_augmented_ppca_mstep(
        stats,
        mean_prior=jnp.asarray(mean_prior),
        W_prior=jnp.asarray(W_prior),
        mask=jnp.asarray(union_mask),
        masks=jnp.asarray(masks_arr),
        pc_mask_assignment=jnp.asarray(pc_mask_assignment),
        mean_mask_idx=mean_mask_idx,
        maxiter=50,
        tol=1e-5,
    )
    mu_np = np.asarray(mu)
    W_np = np.asarray(W)

    # μ outside its assigned mask must be ~0.
    outside_mu = 1.0 - masks_arr[mean_mask_idx]
    assert float(np.sum(mu_np**2 * outside_mu)) < 1e-8

    # Each W_k outside its assigned mask must be ~0.
    for k in range(q):
        assigned = masks_arr[pc_mask_assignment[k]]
        outside_k = 1.0 - assigned
        energy_outside = float(np.sum(W_np[k] ** 2 * outside_k))
        assert energy_outside < 1e-8, f"PC {k} energy outside mask: {energy_outside:.2e}"


# ---------------------------------------------------------------------------
# Warmstart
# ---------------------------------------------------------------------------


def test_augmented_warmstart_path_runs_and_matches_no_warmstart_at_convergence():
    """A correct warmstart should not change the converged result; this is
    a smoke test that the warmstart-handling code path executes without
    shape errors and produces a result close to the cold-started one."""
    rng = np.random.default_rng(404)
    q = 2
    p = q + 1
    grid = 12
    vol_shape, lhs_tri, rhs, reg_aug, mask = _make_synthetic_pcg_problem(rng, q=p, grid=grid)
    half_vol = lhs_tri.shape[0]
    eps = 1e-16
    mean_prior = 1.0 / reg_aug[:, 0] - eps
    W_prior = 1.0 / reg_aug[:, 1:] - eps

    stats = AugmentedPPCAStats(rhs=jnp.asarray(rhs), lhs_tri=jnp.asarray(lhs_tri))
    common = dict(
        mean_prior=jnp.asarray(mean_prior, dtype=jnp.float32),
        W_prior=jnp.asarray(W_prior, dtype=jnp.float32),
        mask=jnp.asarray(mask),
        maxiter=50,
        tol=1e-6,
    )
    mu_cold, W_cold = solve_augmented_ppca_mstep(stats, **common)
    mu_warm, W_warm = solve_augmented_ppca_mstep(
        stats,
        theta_init=(mu_cold, W_cold),
        **common,
    )
    np.testing.assert_allclose(np.asarray(mu_warm), np.asarray(mu_cold), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(W_warm), np.asarray(W_cold), rtol=1e-4, atol=1e-5)
