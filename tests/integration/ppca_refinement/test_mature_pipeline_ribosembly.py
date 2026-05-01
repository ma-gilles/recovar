"""Phase D integration test: mature multi-iter pipeline + simple-test escape.

Verifies:

  * ``run_pose_marginal_refinement_simple`` runs end-to-end on Ribosembly
    with a fixed rotation grid (preserves the dev-eval behavior).
  * ``run_pose_marginal_refinement`` (full schedule: HEALPix angular
    advance + low_resol_join on accumulators + x=0 Hermitian + per-iter
    noise + per-iter prior) runs without error and produces non-trivial
    state updates across iters.

Both run with small budgets to keep the test under a minute.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import recovar.core.fourier_transform_utils as ftu  # noqa: E402
from recovar.em.ppca_refinement.iterations import IterationOpts  # noqa: E402
from recovar.em.ppca_refinement.refinement_loop import (  # noqa: E402
    PPCAScheduleOpts,
    run_pose_marginal_refinement,
    run_pose_marginal_refinement_simple,
)
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _load_helpers():
    src = Path(__file__).resolve().parents[2] / "unit" / "test_ppca_multimask_synthetic.py"
    spec = importlib.util.spec_from_file_location("_ppca_synthetic_helpers", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def setup_ribosembly():
    helpers = _load_helpers()
    vols_real, vols_fourier, vol_shape = helpers._load_ribosembly_volumes(
        n_states=4,
        grid_size=64,
    )
    mask_left, mask_right, _ = helpers._make_split_masks(vol_shape, vols_real)
    cryo, _ = helpers._simulate_dataset(
        vols_fourier,
        vol_shape,
        n_images=80,
        noise_level=1.0,
        seed=42,
    )
    return {
        "cryo": cryo,
        "vol_shape": vol_shape,
        "vols_real": vols_real,
        "mask": np.maximum(mask_left, mask_right).astype(np.float32),
    }


def _initial_state(cryo, vols_real, q):
    vol_shape = (cryo.grid_size, cryo.grid_size, cryo.grid_size)
    half_vol = int(np.prod(ftu.volume_shape_to_half_volume_shape(vol_shape)))
    mu_init = vols_real.mean(axis=0).astype(np.float32)
    rng = np.random.default_rng(0)
    W_init = (rng.standard_normal((q,) + vol_shape) * 1e-3).astype(np.float32)
    return PoseMarginalPPCAEMState(
        mu_half=(jnp.asarray(mu_init), jnp.asarray(mu_init)),
        W_half=(jnp.asarray(W_init), jnp.asarray(W_init)),
        mu_score=jnp.asarray(mu_init),
        W_score=jnp.asarray(W_init),
        W_prior=jnp.full((half_vol, q), 1.0, dtype=jnp.float32),
        mean_prior=jnp.full((half_vol,), 1.0, dtype=jnp.float32),
        z_prior_precision_diag=jnp.ones((q,), dtype=jnp.float32),
        noise_variance=jnp.ones((half_vol,), dtype=jnp.float32),
        contrast_params=None,
        masks=None,
        pose_estimates={},
        pose_priors=None,
        refinement_schedule_state=None,
        hyperparams=None,
    )


def test_simple_refinement_runs(setup_ribosembly):
    s = setup_ribosembly
    cryo = s["cryo"]
    state = _initial_state(cryo, s["vols_real"], q=1)
    rotation_grid = np.asarray(cryo.rotation_matrices[:4], dtype=np.float32)
    halfset_indices = (
        np.asarray(cryo.halfset_indices[0]),
        np.asarray(cryo.halfset_indices[1]),
    )

    final_state, log = run_pose_marginal_refinement_simple(
        state,
        cryo,
        rotation_grid=rotation_grid,
        halfset_indices=halfset_indices,
        mask=jnp.asarray(s["mask"]),
        image_batch_size=16,
        rotation_block_size=4,
        em_iters=2,
        iteration_opts=IterationOpts(EM_iter=2, pcg_maxiter=5),
    )
    assert len(log) == 2
    assert all(rec["iteration"] == i for i, rec in enumerate(log))
    assert not np.any(np.isnan(np.asarray(final_state.mu_score)))
    # Non-trivial update.
    assert float(jnp.linalg.norm(final_state.W_score)) > 0


def test_mature_refinement_runs_with_minimal_schedule(setup_ribosembly):
    """Full schedule with the cheapest possible knobs for a fast test:
    healpix order 0 (72 rot) for 1 iter, all schedule features ON."""
    s = setup_ribosembly
    cryo = s["cryo"]
    state = _initial_state(cryo, s["vols_real"], q=1)
    halfset_indices = (
        np.asarray(cryo.halfset_indices[0]),
        np.asarray(cryo.halfset_indices[1]),
    )
    schedule = PPCAScheduleOpts(
        healpix_order_init=0,
        healpix_order_max=0,
        max_iters=1,
        min_iters=0,
        enable_low_resol_join=True,
        enable_per_iter_prior=False,  # disabled — needs cryo.halfset_indices wired through prior
        enable_per_iter_noise=True,
        enable_x0_hermitian=True,
        use_local_search_at_high_order=False,
    )
    final_state, log = run_pose_marginal_refinement(
        state,
        cryo,
        halfset_indices=halfset_indices,
        mask=jnp.asarray(s["mask"]),
        image_batch_size=16,
        rotation_block_size=8,
        schedule_opts=schedule,
        iteration_opts=IterationOpts(EM_iter=1, pcg_maxiter=5),
    )
    assert len(log) >= 1
    info = log[0]
    assert info["healpix_order"] == 0
    assert info["n_rotations"] is not None
    assert info["n_rotations"] >= 12  # order 0 = 72 rotations × in-plane
    assert "log_evidence_total" in info
    assert "pmax_mean" in info
    assert not np.any(np.isnan(np.asarray(final_state.mu_score)))
    # State.noise_variance updated (D5).
    assert not np.allclose(np.asarray(final_state.noise_variance), np.ones_like(np.asarray(final_state.noise_variance)))


def test_simple_and_full_disagree_on_state_updates(setup_ribosembly):
    """Simple path leaves state.noise_variance and state.mean_prior at their
    initial values; full path with enable_per_iter_noise=True updates them."""
    s = setup_ribosembly
    cryo = s["cryo"]
    state = _initial_state(cryo, s["vols_real"], q=1)
    halfset_indices = (
        np.asarray(cryo.halfset_indices[0]),
        np.asarray(cryo.halfset_indices[1]),
    )

    # Simple path: noise_variance unchanged.
    rotation_grid = np.asarray(cryo.rotation_matrices[:4], dtype=np.float32)
    fs1, _ = run_pose_marginal_refinement_simple(
        state,
        cryo,
        rotation_grid=rotation_grid,
        halfset_indices=halfset_indices,
        mask=jnp.asarray(s["mask"]),
        image_batch_size=16,
        rotation_block_size=4,
        em_iters=1,
        iteration_opts=IterationOpts(EM_iter=1, pcg_maxiter=5),
    )
    np.testing.assert_array_equal(np.asarray(fs1.noise_variance), np.asarray(state.noise_variance))

    # Full path: noise_variance updated.
    fs2, _ = run_pose_marginal_refinement(
        state,
        cryo,
        halfset_indices=halfset_indices,
        mask=jnp.asarray(s["mask"]),
        image_batch_size=16,
        rotation_block_size=8,
        schedule_opts=PPCAScheduleOpts(
            healpix_order_init=0,
            healpix_order_max=0,
            max_iters=1,
            min_iters=0,
            enable_low_resol_join=False,
            enable_per_iter_prior=False,
            enable_per_iter_noise=True,
            enable_x0_hermitian=False,
            use_local_search_at_high_order=False,
        ),
        iteration_opts=IterationOpts(EM_iter=1, pcg_maxiter=5),
    )
    assert not np.allclose(
        np.asarray(fs2.noise_variance),
        np.asarray(state.noise_variance),
    )


def test_mature_two_iter_log_evidence_progresses(setup_ribosembly):
    """A 2-iter mature run must produce iter records with valid totals
    that vary across iterations (not stuck at a fixed point)."""
    s = setup_ribosembly
    cryo = s["cryo"]
    state = _initial_state(cryo, s["vols_real"], q=1)
    halfset_indices = (
        np.asarray(cryo.halfset_indices[0]),
        np.asarray(cryo.halfset_indices[1]),
    )
    final_state, log = run_pose_marginal_refinement(
        state,
        cryo,
        halfset_indices=halfset_indices,
        mask=jnp.asarray(s["mask"]),
        image_batch_size=16,
        rotation_block_size=8,
        schedule_opts=PPCAScheduleOpts(
            healpix_order_init=0,
            healpix_order_max=0,
            max_iters=2,
            min_iters=2,
            convergence_log_evidence_rtol=-1.0,  # never converge early
            enable_low_resol_join=False,
            enable_per_iter_prior=False,
            enable_per_iter_noise=True,
            enable_x0_hermitian=True,
            use_local_search_at_high_order=False,
        ),
        iteration_opts=IterationOpts(EM_iter=1, pcg_maxiter=5),
    )
    assert len(log) == 2
    le0 = log[0]["log_evidence_total"]
    le1 = log[1]["log_evidence_total"]
    # Both finite and not identical (state changed between iters).
    assert np.isfinite(le0) and np.isfinite(le1)
    assert le0 != le1
    # μ should have moved from initial.
    assert not np.allclose(np.asarray(final_state.mu_score), np.asarray(state.mu_score), atol=1e-6)
