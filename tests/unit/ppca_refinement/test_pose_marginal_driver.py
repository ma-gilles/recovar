"""Phase 6 (M5) tests: pose-marginal EM driver orchestration.

Verifies that ``run_pose_marginal_ppca_refine`` correctly orchestrates the
E-step (dense engine) → backprojection callback → augmented M-step
(``solve_augmented_ppca_mstep``) → halfset combine cycle for a synthetic
in-memory problem. The block_provider and backprojector callbacks are
synthetic — production wiring lands at M10 with the real dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import recovar.core.fourier_transform_utils as ftu  # noqa: E402
from recovar.em.ppca_refinement.iterations import (  # noqa: E402
    IterationOpts,
    PoseBlock,
    run_pose_marginal_ppca_refine,
)
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState  # noqa: E402
from recovar.ppca import AugmentedPPCAStats, PCPriorConfig  # noqa: E402
from recovar.ppca.ppca import _tri_size  # noqa: E402

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic block + backproj fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_state(rng, q, grid):
    vol_shape = (grid, grid, grid)
    half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
    half_vol = int(np.prod(half_vs))

    def zeros_vol():
        return jnp.zeros(vol_shape, dtype=jnp.float32)

    def zeros_W():
        return jnp.zeros((q,) + vol_shape, dtype=jnp.float32)

    state = PoseMarginalPPCAEMState(
        mu_half=(zeros_vol(), zeros_vol()),
        W_half=(zeros_W(), zeros_W()),
        mu_score=zeros_vol(),
        W_score=zeros_W(),
        W_prior=jnp.full((half_vol, q), 1e3, dtype=jnp.float32),
        mean_prior=jnp.full((half_vol,), 1e3, dtype=jnp.float32),
        z_prior_precision_diag=jnp.ones((q,), dtype=jnp.float32),
        noise_variance=jnp.ones((half_vol,), dtype=jnp.float32),
        contrast_params=None,
        masks=None,
        pose_estimates=None,
        pose_priors=None,
        refinement_schedule_state=None,
        hyperparams=None,
    )
    return state, vol_shape, half_vs, half_vol


def _make_block(rng, B, T, R, F, q, halfset_idx):
    P = q + 1
    Y1 = (rng.standard_normal((B, T, F)) + 1j * rng.standard_normal((B, T, F))).astype(np.complex64)
    proj_aug = (rng.standard_normal((R, P, F)) + 1j * rng.standard_normal((R, P, F))).astype(np.complex64)
    ctf2 = rng.uniform(0.1, 1.0, size=(B, F)).astype(np.float32)
    y_norm = rng.uniform(0.5, 2.0, size=(B,)).astype(np.float32)
    return PoseBlock(
        Y1=jnp.asarray(Y1),
        proj_aug=jnp.asarray(proj_aug),
        ctf2_over_noise=jnp.asarray(ctf2),
        y_norm=jnp.asarray(y_norm),
        pose_log_prior=None,
        image_indices=jnp.arange(B, dtype=jnp.int32),
        halfset_idx=halfset_idx,
        rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (R, 3, 3)),
        translations=jnp.zeros((T, 2), dtype=jnp.float32),
    )


def _make_dummy_backprojector(half_vol, q):
    """Returns synthetic AugmentedPPCAStats with diagonally-dominant
    lhs_tri and small random rhs. Independent of the actual moments —
    this only exercises the M-step plumbing."""
    P = q + 1

    def backproject(image_stats, halfset_idx):
        rng = np.random.default_rng(123 + halfset_idx)
        # Diagonally-dominant real lhs_tri so PCG converges.
        tri = _tri_size(P)
        lhs = np.zeros((half_vol, tri), dtype=np.float32)
        idx = 0
        for i in range(P):
            for j in range(i, P):
                if i == j:
                    lhs[:, idx] = 1.0 + rng.uniform(0.0, 0.1, size=half_vol).astype(np.float32)
                else:
                    lhs[:, idx] = rng.normal(0.0, 0.01, size=half_vol).astype(np.float32)
                idx += 1
        rhs = (
            rng.standard_normal((half_vol, P)).astype(np.float32)
            + 1j * rng.standard_normal((half_vol, P)).astype(np.float32)
        ).astype(np.complex64)
        return AugmentedPPCAStats(
            rhs=jnp.asarray(rhs),
            lhs_tri=jnp.asarray(lhs),
            n_images=sum(b.Y1.shape[0] for b, _, _ in image_stats),
        )

    return backproject


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pose_marginal_driver_orchestrates_one_iteration():
    rng = np.random.default_rng(0)
    grid = 12
    q = 2
    state, vol_shape, half_vs, half_vol = _make_synthetic_state(rng, q, grid)
    F = 8
    B, T, R = 3, 2, 4
    blocks = [
        _make_block(rng, B, T, R, F, q, halfset_idx=0),
        _make_block(rng, B, T, R, F, q, halfset_idx=1),
    ]

    def block_provider(theta_score_for_half, iteration):
        return blocks

    backprojector = _make_dummy_backprojector(half_vol, q)

    # Spherical mask of radius ~ grid * 0.4.
    cz, cy, cx = np.indices(vol_shape, dtype=np.float32)
    cz -= grid / 2
    cy -= grid / 2
    cx -= grid / 2
    mask = (cz * cz + cy * cy + cx * cx < (0.4 * grid) ** 2).astype(np.float32)

    final_state, log = run_pose_marginal_ppca_refine(
        state,
        block_provider=block_provider,
        backprojector=backprojector,
        mask=jnp.asarray(mask),
        opts=IterationOpts(EM_iter=1, pcg_maxiter=10),
    )

    assert len(log) == 1
    assert log[0]["iteration"] == 0
    # mu_half should now be non-zero (M-step ran).
    assert float(jnp.linalg.norm(final_state.mu_half[0])) > 0
    assert float(jnp.linalg.norm(final_state.mu_half[1])) > 0
    # W_half shape preserved.
    assert final_state.W_half[0].shape == (q, *vol_shape)
    assert final_state.W_half[1].shape == (q, *vol_shape)
    # mu_score and W_score updated via halfset combiner.
    assert float(jnp.linalg.norm(final_state.mu_score)) > 0
    assert final_state.W_score.shape == (q, *vol_shape)


def test_pose_marginal_driver_iterates_n_times_and_logs():
    rng = np.random.default_rng(0)
    state, vol_shape, half_vs, half_vol = _make_synthetic_state(rng, q=1, grid=10)
    F = 6
    blocks = [
        _make_block(rng, 2, 2, 3, F, 1, halfset_idx=0),
        _make_block(rng, 2, 2, 3, F, 1, halfset_idx=1),
    ]

    def block_provider(theta_score, iteration):
        return blocks

    backprojector = _make_dummy_backprojector(half_vol, 1)
    mask = np.ones(vol_shape, dtype=np.float32)

    final_state, log = run_pose_marginal_ppca_refine(
        state,
        block_provider=block_provider,
        backprojector=backprojector,
        mask=jnp.asarray(mask),
        opts=IterationOpts(EM_iter=3, pcg_maxiter=5),
    )
    assert len(log) == 3
    assert [r["iteration"] for r in log] == [0, 1, 2]
    for r in log:
        assert "log_evidence_total" in r
        assert "pmax_mean" in r
        assert "n_significant_mean" in r


def test_pose_marginal_driver_pc_prior_config_lands_in_iter0_log():
    rng = np.random.default_rng(0)
    state, vol_shape, half_vs, half_vol = _make_synthetic_state(rng, q=1, grid=8)
    blocks = [_make_block(rng, 2, 2, 2, 4, 1, halfset_idx=0), _make_block(rng, 2, 2, 2, 4, 1, halfset_idx=1)]
    backprojector = _make_dummy_backprojector(half_vol, 1)
    cfg = PCPriorConfig(prior_freeze_iters=99)

    def block_provider(theta_score, iteration):
        return blocks

    _, log = run_pose_marginal_ppca_refine(
        state,
        block_provider=block_provider,
        backprojector=backprojector,
        mask=jnp.ones(vol_shape, dtype=jnp.float32),
        opts=IterationOpts(EM_iter=2, pcg_maxiter=5, pc_prior_config=cfg),
    )
    assert "pc_prior_config" in log[0]
    assert log[0]["pc_prior_config"]["prior_freeze_iters"] == 99
    # Subsequent iters do not snapshot the config.
    assert "pc_prior_config" not in log[1]


def test_pose_marginal_driver_prior_recompute_schedule():
    """When ``allow_every_iter_prior_update=False`` and
    ``recompute_once_after_iter=2``, the recompute callback fires only
    once at iter 2."""
    rng = np.random.default_rng(0)
    state, vol_shape, half_vs, half_vol = _make_synthetic_state(rng, q=1, grid=8)
    blocks = [_make_block(rng, 2, 2, 2, 4, 1, halfset_idx=0), _make_block(rng, 2, 2, 2, 4, 1, halfset_idx=1)]
    backprojector = _make_dummy_backprojector(half_vol, 1)
    cfg = PCPriorConfig(prior_freeze_iters=0, recompute_once_after_iter=2)
    recompute_calls = []

    def prior_recompute_fn(s):
        recompute_calls.append(1)
        return s.W_prior  # return same prior — just track the call

    def block_provider(theta_score, iteration):
        return blocks

    run_pose_marginal_ppca_refine(
        state,
        block_provider=block_provider,
        backprojector=backprojector,
        mask=jnp.ones(vol_shape, dtype=jnp.float32),
        prior_recompute_fn=prior_recompute_fn,
        opts=IterationOpts(EM_iter=4, pcg_maxiter=5, pc_prior_config=cfg),
    )
    # Recompute fires exactly once (at iter 2).
    assert sum(recompute_calls) == 1
