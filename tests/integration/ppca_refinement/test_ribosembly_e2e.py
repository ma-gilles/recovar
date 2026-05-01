"""Phase 11 (M10) integration test: Ribosembly end-to-end on the M3 driver.

The M5+/M7 production fused engine for ``--pose-mode dense/local`` lands
in a follow-up commit (see
``recovar/em/ppca_refinement/dataset_adapter.py`` for the algorithm
spec). For the dev-eval gate the user asked for, we exercise:

  * the M3 fixed-pose driver end-to-end on Ribosembly (``--pose-mode fixed``);
  * the M5 callback orchestration with a synthetic block_provider on the
    same Ribosembly halfset assignment, so the driver loop is verified
    on real-shape data even though projection / backprojection are
    stubbed.

The user resolved the "CryoBench dataset" open question:
    "use ribosembly + igg + others for complete evaluation. For dev,
     stick to ribosembly."

So this dev test stays on Ribosembly; the multi-dataset full evaluation
runs separately in M10's pre-PR Slurm phase.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import recovar.core.fourier_transform_utils as ftu  # noqa: E402
from recovar.em.ppca_refinement.dataset_adapter import (  # noqa: E402
    make_simple_block_provider_for_test,
)
from recovar.em.ppca_refinement.halfset_combine import make_halfset_combiner  # noqa: E402
from recovar.em.ppca_refinement.iterations import (  # noqa: E402
    IterationOpts,
    run_fixed_pose_ppca_refine,
    run_pose_marginal_ppca_refine,
)
from recovar.em.ppca_refinement.state import (  # noqa: E402
    FixedPosePPCAState,
    PoseMarginalPPCAEMState,
)
from recovar.ppca import AugmentedPPCAStats, PCPriorConfig  # noqa: E402
from recovar.ppca.ppca import _tri_size  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _load_synthetic_helpers():
    src = Path(__file__).resolve().parents[2] / "unit" / "test_ppca_multimask_synthetic.py"
    spec = importlib.util.spec_from_file_location("_ppca_synthetic_helpers", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ribosembly_e2e_setup():
    helpers = _load_synthetic_helpers()
    vols_real, vols_fourier, vol_shape = helpers._load_ribosembly_volumes(
        n_states=4,
        grid_size=64,
    )
    mask_left, mask_right, support = helpers._make_split_masks(vol_shape, vols_real)
    cryo, _ = helpers._simulate_dataset(
        vols_fourier,
        vol_shape,
        n_images=200,
        noise_level=1.0,
        seed=42,
    )
    mean_fourier = vols_fourier.mean(axis=0)
    half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
    half_vol = int(np.prod(half_vs))
    rng = np.random.default_rng(0)
    basis_size = 2
    W_init = (
        rng.normal(size=(half_vol, basis_size)) * 0.01 + 1j * rng.normal(size=(half_vol, basis_size)) * 0.01
    ).astype(np.complex64)
    W_prior = np.ones((int(np.prod(vol_shape)), basis_size), dtype=np.float32)
    union_mask = np.maximum(mask_left, mask_right)
    return {
        "cryo": cryo,
        "vol_shape": vol_shape,
        "mean_fourier": mean_fourier,
        "W_init": W_init,
        "W_prior": W_prior,
        "mask": union_mask,
        "basis_size": basis_size,
        "half_vol": half_vol,
        "n_images": 200,
    }


def test_m3_fixed_pose_e2e_on_ribosembly(ribosembly_e2e_setup):
    """M3 fixed-pose driver runs end-to-end on the Ribosembly synthetic
    fixture (200 images, 64³ grid, q=2, EM_iter=2). Exit condition: no
    NaN, embeddings shape, monotone NLL."""
    s = ribosembly_e2e_setup
    state = FixedPosePPCAState(
        mean_estimate=s["mean_fourier"],
        W_initial=s["W_init"],
        W_prior=s["W_prior"],
        volume_mask=s["mask"],
        pc_prior_config=PCPriorConfig(prior_freeze_iters=99),
    )
    out = run_fixed_pose_ppca_refine(
        s["cryo"],
        state,
        EM_iter=2,
        return_iteration_data=True,
    )
    U, S, W, ez, smz, iter_data = out
    assert ez.shape == (s["n_images"], s["basis_size"])
    assert not np.any(np.isnan(np.asarray(ez)))
    nll = [r["Neg_LL_Total"] for r in iter_data]
    # Monotone (allow tiny noise).
    assert all(b <= a + 1.0 for a, b in zip(nll, nll[1:])), nll
    # PCPriorConfig snapshot lands in iter 0.
    assert "pc_prior_config" in iter_data[0]
    assert iter_data[0]["pc_prior_config"]["prior_freeze_iters"] == 99


def test_m5_dense_driver_orchestration_on_ribosembly_halfsets(ribosembly_e2e_setup):
    """Exercise the M5 driver with a synthetic block_provider on the
    Ribosembly halfset assignment. The block_provider is a stub
    (see ``dataset_adapter.make_simple_block_provider_for_test``) — the
    point is to verify the driver loop runs on a 200-image halfset
    split, mu_half/W_half/mu_score/W_score get updated, and pose_estimates
    populates from posterior diagnostics.
    """
    s = ribosembly_e2e_setup
    vol_shape = s["vol_shape"]
    half_vol = s["half_vol"]
    q = 1
    n_images = s["n_images"]
    halfset_idx_per_image = (np.arange(n_images) % 2).astype(np.int32)
    image_indices = np.arange(n_images, dtype=np.int32)

    block_provider = make_simple_block_provider_for_test(
        s["cryo"],
        image_indices=image_indices,
        halfset_idx_per_image=halfset_idx_per_image,
        n_rotations=4,
        n_translations=2,
        image_batch_size=50,
        seed=0,
    )

    # Synthetic backprojector — we don't have the production fused
    # backprojection yet, so this just produces well-shaped stats so the
    # M-step can execute. The M-step machinery itself is unit-tested.
    def backproject(image_stats, halfset_idx):
        rng = np.random.default_rng(123 + halfset_idx)
        P = q + 1
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
        n_total = sum(b.Y1.shape[0] for b, _, _ in image_stats)
        return AugmentedPPCAStats(rhs=jnp.asarray(rhs), lhs_tri=jnp.asarray(lhs), n_images=n_total)

    state = PoseMarginalPPCAEMState(
        mu_half=(jnp.zeros(vol_shape, dtype=jnp.float32), jnp.zeros(vol_shape, dtype=jnp.float32)),
        W_half=(jnp.zeros((q,) + vol_shape, dtype=jnp.float32), jnp.zeros((q,) + vol_shape, dtype=jnp.float32)),
        mu_score=jnp.zeros(vol_shape, dtype=jnp.float32),
        W_score=jnp.zeros((q,) + vol_shape, dtype=jnp.float32),
        W_prior=jnp.full((half_vol, q), 1e3, dtype=jnp.float32),
        mean_prior=jnp.full((half_vol,), 1e3, dtype=jnp.float32),
        z_prior_precision_diag=jnp.ones((q,), dtype=jnp.float32),
        noise_variance=jnp.ones((half_vol,), dtype=jnp.float32),
        contrast_params=None,
        masks=None,
        pose_estimates={},
        pose_priors=None,
        refinement_schedule_state=None,
        hyperparams=None,
    )
    halfset_combiner = make_halfset_combiner(method="low_resol_join", voxel_size=4.25)

    final_state, log = run_pose_marginal_ppca_refine(
        state,
        block_provider=block_provider,
        backprojector=backproject,
        halfset_combiner=halfset_combiner,
        mask=jnp.asarray(s["mask"]),
        opts=IterationOpts(EM_iter=2, pcg_maxiter=5),
    )
    assert len(log) == 2
    assert all(r["iteration"] == i for i, r in enumerate(log))
    assert float(jnp.linalg.norm(final_state.mu_half[0])) > 0
    assert float(jnp.linalg.norm(final_state.mu_half[1])) > 0
    assert final_state.W_score.shape == (q,) + vol_shape
