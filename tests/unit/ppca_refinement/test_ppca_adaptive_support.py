import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.ppca_refinement.dense_dataset import DensePPCASignificanceResult
from recovar.em.ppca_refinement.highres_refinement import run_adaptive_ppca_halfset_fused_em_iteration
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState

pytestmark = pytest.mark.unit


class _FakeDataset:
    image_shape = (4, 4)
    volume_shape = (4, 4, 4)
    voxel_size = 1.0
    n_images = 2
    n_units = 2


def _state():
    mean = jnp.zeros((1,), dtype=jnp.complex64)
    W = jnp.zeros((1, 0), dtype=jnp.complex64)
    return PoseMarginalPPCAEMState(
        mu_half=(mean, mean),
        W_half=(W, W),
        mu_score=mean,
        W_score=W,
        W_prior=jnp.zeros((1, 0), dtype=jnp.float32),
        mean_prior=jnp.ones((1,), dtype=jnp.float32),
        noise_variance=jnp.ones((1,), dtype=jnp.float32),
        z_prior_precision_diag=jnp.zeros((0,), dtype=jnp.float32),
        schedule_state=None,
    )


def _sig():
    return DensePPCASignificanceResult(
        significant_sample_indices=[
            np.asarray([0], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
        ],
        n_significant_per_image=np.asarray([1, 1], dtype=np.int32),
        hard_assignment=np.asarray([0, 1], dtype=np.int32),
        best_rotation_idx=np.asarray([0, 0], dtype=np.int32),
        best_translation_idx=np.asarray([0, 1], dtype=np.int32),
        logZ=np.asarray([0.0, 0.0], dtype=np.float64),
        max_posterior_per_image=np.asarray([1.0, 1.0], dtype=np.float32),
        diagnostics={"nsig_mean": 1.0, "pmax_mean": 1.0},
    )


def test_adaptive_ppca_builds_exact_fine_support_from_coarse_significance(monkeypatch):
    observed = {}

    def _compute(*args, **kwargs):
        observed.setdefault("compute_calls", 0)
        observed["compute_calls"] += 1
        observed["adaptive_fraction"] = kwargs["adaptive_fraction"]
        return _sig()

    def _local(state, halfsets, layouts, **kwargs):
        del halfsets, kwargs
        observed["layouts"] = layouts
        return state.replace(
            pose_diagnostics={
                "halfset0": {"pmax_mean": 1.0, "best_rotation_id": np.asarray([0]), "best_translation_idx": np.asarray([0])},
                "halfset1": {"pmax_mean": 1.0, "best_rotation_id": np.asarray([0]), "best_translation_idx": np.asarray([0])},
            }
        )

    monkeypatch.setattr(
        "recovar.em.ppca_refinement.highres_refinement.compute_dense_ppca_adaptive_significance",
        _compute,
    )
    monkeypatch.setattr(
        "recovar.em.ppca_refinement.highres_refinement.run_local_ppca_halfset_fused_em_iteration",
        _local,
    )
    updated = run_adaptive_ppca_halfset_fused_em_iteration(
        _state(),
        (_FakeDataset(), _FakeDataset()),
        coarse_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (12, 3, 3)),
        coarse_translations=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        nside_level=0,
        adaptive_oversampling=0,
        adaptive_fraction=0.75,
    )
    assert observed["compute_calls"] == 2
    assert observed["adaptive_fraction"] == 0.75
    for layout in observed["layouts"]:
        assert layout.sample_mask_flat is not None
        assert layout.sample_mask_flat.shape[1] == 2
        assert np.all(np.sum(layout.sample_mask_flat, axis=(0, 1)) >= 1)
    assert updated.pose_diagnostics["halfset0"]["path"] == "adaptive"
    assert updated.pose_diagnostics["halfset0"]["adaptive_fine_translation_count"] == 2
