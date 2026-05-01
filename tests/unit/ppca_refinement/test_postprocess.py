"""Phase 10 (M9) tests: post-EM postprocessing + state.pkl save/load.

Verifies:
  * ``finalize_ppca_state`` produces an orthonormal real-space basis from
    a synthetic state, with valid eigenvalues, both single-mask and
    multimask paths.
  * ``save_state`` → ``load_state`` round-trips fields without loss.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import recovar.core.fourier_transform_utils as ftu  # noqa: E402
from recovar.em.ppca_refinement.postprocess import (  # noqa: E402
    finalize_ppca_state,
    load_state,
    save_state,
)
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState  # noqa: E402

pytestmark = pytest.mark.unit


def _synthetic_state(rng, q, grid):
    vol_shape = (grid, grid, grid)
    half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
    half_vol = int(np.prod(half_vs))

    mu = rng.standard_normal(vol_shape).astype(np.float32)
    W_score = rng.standard_normal((q,) + vol_shape).astype(np.float32)
    return PoseMarginalPPCAEMState(
        mu_half=(jnp.asarray(mu), jnp.asarray(mu)),
        W_half=(jnp.asarray(W_score), jnp.asarray(W_score)),
        mu_score=jnp.asarray(mu),
        W_score=jnp.asarray(W_score),
        W_prior=jnp.full((half_vol, q), 1e3, dtype=jnp.float32),
        mean_prior=jnp.full((half_vol,), 1e3, dtype=jnp.float32),
        z_prior_precision_diag=jnp.ones((q,), dtype=jnp.float32),
        noise_variance=jnp.ones((half_vol,), dtype=jnp.float32),
        contrast_params=None,
        masks=None,
        pose_estimates={"foo": "bar"},
        pose_priors=None,
        refinement_schedule_state=None,
        hyperparams=None,
        diagnostics={"final_iter_log": [{"iteration": 0, "log_evidence_total": -1.0}]},
    )


def test_finalize_single_mask_returns_orthonormal_basis():
    rng = np.random.default_rng(0)
    state = _synthetic_state(rng, q=3, grid=10)
    U, S, W_half = finalize_ppca_state(state, volume_shape=(10, 10, 10))
    U_np = np.asarray(U)
    S_np = np.asarray(S)
    assert U_np.shape == (3, 10, 10, 10)
    assert S_np.shape == (3,)
    assert np.all(np.isfinite(U_np))
    assert np.all(np.isfinite(S_np))
    # PPCA Fourier convention: U should be unit-norm in Fourier-space.
    # Real-space norm of each PC volume should be 1/sqrt(vol_size).
    vol_size = 10**3
    norms = np.sqrt(np.sum(U_np.reshape(3, -1) ** 2, axis=1))
    np.testing.assert_allclose(norms, 1.0 / np.sqrt(vol_size), rtol=1e-4)
    # Eigenvalues non-negative.
    assert np.all(S_np >= 0)
    # W_half complex64 with shape (half_vol, q).
    assert W_half.shape[1] == 3
    assert W_half.dtype == np.complex64


def test_finalize_multimask_path_runs():
    rng = np.random.default_rng(1)
    state = _synthetic_state(rng, q=4, grid=10)
    pc_mask_assignment = np.array([0, 0, 1, 1], dtype=np.int32)
    U, S, W_half = finalize_ppca_state(
        state,
        volume_shape=(10, 10, 10),
        pc_mask_assignment=pc_mask_assignment,
    )
    assert U.shape == (4, 10, 10, 10)
    assert S.shape == (4,)


def test_save_state_load_state_roundtrip():
    rng = np.random.default_rng(2)
    state = _synthetic_state(rng, q=2, grid=8)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.pkl"
        save_state(state, path)
        assert path.exists()
        state2 = load_state(path)

    np.testing.assert_array_equal(np.asarray(state.mu_score), np.asarray(state2.mu_score))
    np.testing.assert_array_equal(np.asarray(state.W_score), np.asarray(state2.W_score))
    np.testing.assert_array_equal(np.asarray(state.mu_half[0]), np.asarray(state2.mu_half[0]))
    np.testing.assert_array_equal(np.asarray(state.W_half[1]), np.asarray(state2.W_half[1]))
    np.testing.assert_array_equal(np.asarray(state.W_prior), np.asarray(state2.W_prior))
    assert state2.pose_estimates == {"foo": "bar"}
    assert "final_iter_log" in state2.diagnostics


def test_save_state_creates_parent_dir():
    rng = np.random.default_rng(3)
    state = _synthetic_state(rng, q=1, grid=8)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "nested" / "deep" / "state.pkl"
        save_state(state, path)
        assert path.exists()


def test_load_state_preserves_tuple_halfsets():
    """mu_half and W_half are tuples in the dataclass; pickle of frozen
    dataclass preserves them but list/tuple conversion in load must not
    drop the tuple type."""
    rng = np.random.default_rng(4)
    state = _synthetic_state(rng, q=2, grid=8)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.pkl"
        save_state(state, path)
        state2 = load_state(path)
    assert isinstance(state2.mu_half, tuple)
    assert isinstance(state2.W_half, tuple)
    assert len(state2.mu_half) == 2
    assert len(state2.W_half) == 2
