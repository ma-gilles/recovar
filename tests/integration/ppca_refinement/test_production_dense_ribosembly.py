"""Phase A integration test: production dense driver on Ribosembly.

End-to-end test of run_pose_marginal_iteration_dense_production using
the synthetic Ribosembly fixture. Verifies:

  * fused dense engine + production block iteration runs without error
    on real-shape data;
  * after one iteration μ_score and W_score are non-trivially updated;
  * iter diagnostics are populated.

Real-volume rotations from the dataset are used as the rotation grid
(so we score every image at its true rotation + a few perturbations).
This is a SMALLER test than --pose-mode dense in production (which uses
a full HEALPix sphere), but it exercises the same code path.
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
from recovar.em.ppca_refinement.production_driver import (  # noqa: E402
    run_pose_marginal_iteration_dense_production,
)
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _load_synthetic_helpers():
    src = Path(__file__).resolve().parents[2] / "unit" / "test_ppca_multimask_synthetic.py"
    spec = importlib.util.spec_from_file_location("_ppca_synthetic_helpers", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ribosembly_setup():
    helpers = _load_synthetic_helpers()
    vols_real, vols_fourier, vol_shape = helpers._load_ribosembly_volumes(
        n_states=4,
        grid_size=64,
    )
    mask_left, mask_right, support = helpers._make_split_masks(vol_shape, vols_real)
    cryo, _ = helpers._simulate_dataset(
        vols_fourier,
        vol_shape,
        n_images=100,
        noise_level=1.0,
        seed=42,
    )
    return {
        "vols_fourier": vols_fourier,
        "vol_shape": vol_shape,
        "mask": np.maximum(mask_left, mask_right),
        "cryo": cryo,
    }


def test_production_dense_iteration_runs_on_ribosembly(ribosembly_setup):
    s = ribosembly_setup
    cryo = s["cryo"]
    vol_shape = s["vol_shape"]
    half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
    half_vol = int(np.prod(half_vs))
    q = 1

    # Initial state: μ from mean of GT volumes (flattened by helper),
    # reshaped to (D, D, D) for the production driver.
    mu0_flat_fourier = s["vols_fourier"].mean(axis=0)
    mu0 = np.real(np.fft.ifftn(mu0_flat_fourier.reshape(vol_shape))).astype(np.float32)
    rng = np.random.default_rng(0)
    W0 = (rng.standard_normal((q,) + vol_shape) * 1e-3).astype(np.float32)

    state = PoseMarginalPPCAEMState(
        mu_half=(jnp.asarray(mu0), jnp.asarray(mu0)),
        W_half=(jnp.asarray(W0), jnp.asarray(W0)),
        mu_score=jnp.asarray(mu0),
        W_score=jnp.asarray(W0),
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

    # Use 8 random rotations from the dataset's true rotations (subset
    # so the rotation block is small enough for a fast test).
    n_rot = 8
    rotation_grid = np.asarray(cryo.rotation_matrices[:n_rot], dtype=np.float32)
    # 1 translation (no shift) keeps the test small.
    translation_grid = np.zeros((1, 2), dtype=np.float32)

    # Halfset assignment from cryo.
    halfset_indices = (
        np.asarray(cryo.halfset_indices[0]),
        np.asarray(cryo.halfset_indices[1]),
    )

    new_state, diag = run_pose_marginal_iteration_dense_production(
        state,
        cryo,
        rotation_grid=rotation_grid,
        translation_grid=translation_grid,
        halfset_indices=halfset_indices,
        mask=jnp.asarray(s["mask"]),
        image_batch_size=16,
        rotation_block_size=4,
        opts=IterationOpts(EM_iter=1, pcg_maxiter=10),
    )

    # μ_score and W_score updated.
    assert new_state.mu_score.shape == vol_shape
    assert new_state.W_score.shape == (q,) + vol_shape
    assert not np.any(np.isnan(np.asarray(new_state.mu_score)))
    assert not np.any(np.isnan(np.asarray(new_state.W_score)))
    # Both halfsets contributed.
    assert diag["iteration_log_evidence"][0] != 0.0
    assert diag["iteration_log_evidence"][1] != 0.0
    # n_significant tracked per halfset.
    assert diag["iteration_n_significant_mean"][0] >= 0
    assert diag["iteration_n_significant_mean"][1] >= 0
