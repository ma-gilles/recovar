from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.ppca_refinement.highres_refinement import (
    run_highres_ppca_refinement_with_kclass_pose_hierarchy,
)
from recovar.em.ppca_refinement.dense_dataset import DensePPCASignificanceResult
from recovar.em.ppca_refinement.refinement_loop import HalfsetMeanComparison
from recovar.em.ppca_refinement.schedule import PPCARefinementScheduleState
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState

pytestmark = pytest.mark.unit


class _FakeDataset:
    image_shape = (4, 4)
    volume_shape = (4, 4, 4)
    voxel_size = 1.0

    def get_halfset(self, halfset_id):
        del halfset_id
        return self


class _Bridge:
    def __init__(self, *, do_local_search=True):
        self.state = SimpleNamespace(
            healpix_order=0,
            adaptive_oversampling=0,
            do_local_search=bool(do_local_search),
            translation_step=1.0,
            sigma_rot=0.0,
            sigma_psi=0.0,
            effective_step=10.0,
        )
        self.history = []

    def __call__(self, iteration, state, *, current_size, proposed_current_size, halfset_comparison):
        del state
        self.history.append(
            {
                "iteration": int(iteration),
                "current_size": int(current_size),
                "proposed_current_size": int(proposed_current_size),
                "supported": bool(halfset_comparison.resolution_supports),
            }
        )
        return True


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
        schedule_state=PPCARefinementScheduleState(current_size=4, healpix_order=0, q=0),
        pose_diagnostics={
            "halfset0": {
                "best_rotation_idx": np.asarray([0, 1], dtype=np.int32),
                "best_translation_idx": np.asarray([0, 1], dtype=np.int32),
                "top_rotation_id": np.asarray([[0, 2], [1, 3]], dtype=np.int32),
                "top_translation_idx": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
                "max_posterior_per_image": np.ones(2, dtype=np.float32),
            },
            "halfset1": {
                "best_rotation_idx": np.asarray([2, 3], dtype=np.int32),
                "best_translation_idx": np.asarray([1, 0], dtype=np.int32),
                "top_rotation_id": np.asarray([[2, 0], [3, 1]], dtype=np.int32),
                "top_translation_idx": np.asarray([[1, 0], [0, 1]], dtype=np.int32),
                "max_posterior_per_image": np.ones(2, dtype=np.float32),
            },
        },
    )


def _blocked_comparator(state, proposed_size):
    del state, proposed_size
    return HalfsetMeanComparison(
        means_aligned=True,
        resolution_supports=False,
        no_halfset_drift=True,
        fsc=np.asarray([1.0], dtype=np.float32),
        diagnostics={"blocked": True},
    )


def test_highres_skips_dense_when_full_grid_exceeds_cap(monkeypatch):
    called = {"dense": False}

    def _dense(*args, **kwargs):
        del args, kwargs
        called["dense"] = True
        return _state()

    monkeypatch.setattr(
        "recovar.em.ppca_refinement.highres_refinement.run_dense_ppca_halfset_fused_em_iteration",
        _dense,
    )
    result = run_highres_ppca_refinement_with_kclass_pose_hierarchy(
        _state(),
        _FakeDataset(),
        rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)),
        translations=np.zeros((2, 2), dtype=np.float32),
        n_dense_iterations=1,
        n_adaptive_iterations=0,
        n_local_iterations=0,
        max_dense_pose_candidates=4,
        bridge=_Bridge(do_local_search=False),
    )
    assert called["dense"] is False
    assert result.diagnostics["stage_history"][0]["stage"] == "dense_skipped_grid_cap"


def test_highres_local_support_uses_union_of_top_p_and_gate_can_block(monkeypatch):
    observed = {}

    def _grid(order):
        del order
        return np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy()

    def _layouts(pose_diagnostics, **kwargs):
        observed["half0_rot"] = np.asarray(pose_diagnostics["halfset0"]["top_rotation_id"])
        observed["half1_rot"] = np.asarray(pose_diagnostics["halfset1"]["top_rotation_id"])
        observed["kwargs"] = kwargs
        return ("layout0", "layout1")

    def _local(state, halfsets, layouts, **kwargs):
        del halfsets, kwargs
        observed["layouts"] = layouts
        return state

    monkeypatch.setattr("recovar.em.ppca_refinement.highres_refinement.get_relion_rotation_grid", _grid)
    monkeypatch.setattr(
        "recovar.em.ppca_refinement.highres_refinement.build_top_p_local_hypothesis_layouts_from_diagnostics",
        _layouts,
    )
    monkeypatch.setattr(
        "recovar.em.ppca_refinement.highres_refinement.run_local_ppca_halfset_fused_em_iteration",
        _local,
    )
    result = run_highres_ppca_refinement_with_kclass_pose_hierarchy(
        _state(),
        _FakeDataset(),
        rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)),
        translations=np.zeros((2, 2), dtype=np.float32),
        n_dense_iterations=0,
        n_adaptive_iterations=0,
        n_local_iterations=1,
        top_p_poses=2,
        init_current_size=4,
        max_current_size=8,
        bridge=_Bridge(do_local_search=True),
        halfset_comparator=_blocked_comparator,
    )
    np.testing.assert_array_equal(observed["half0_rot"], np.asarray([[0, 2], [1, 3]], dtype=np.int32))
    np.testing.assert_array_equal(observed["half1_rot"], np.asarray([[2, 0], [3, 1]], dtype=np.int32))
    assert observed["layouts"] == ("layout0", "layout1")
    assert result.iteration_records[0].current_size == 4
    assert "halfset mean comparison below requested resolution" in result.iteration_records[0].resolution_decision.reasons


def test_pose_warmup_scores_dense_then_local_before_first_mstep(monkeypatch):
    observed = {"layout_calls": 0}
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (4, 3, 3)).copy()
    translations = np.zeros((2, 2), dtype=np.float32)

    def _grid(order):
        del order
        return rotations

    def _dense_sig(half_dataset, mu, W, **kwargs):
        del half_dataset, W
        observed.setdefault("dense_mu", []).append(np.asarray(mu).copy())
        half_idx = len(observed["dense_mu"]) - 1
        top_rot = np.asarray([[half_idx, 2], [1, 3]], dtype=np.int32)
        top_trans = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
        diag = {
            "best_rotation_idx": top_rot[:, 0],
            "best_rotation_id": top_rot[:, 0],
            "best_translation_idx": top_trans[:, 0],
            "max_posterior_per_image": np.asarray([0.9, 0.8], dtype=np.float32),
            "pmax_mean": 0.85,
            "logZ_mean": 1.0,
            "nsig_mean": 2.0,
            "top_rotation_idx": top_rot,
            "top_rotation_id": top_rot,
            "top_translation_idx": top_trans,
            "top_log_score": np.asarray([[4.0, 3.0], [2.0, 1.0]], dtype=np.float32),
            "top_posterior": np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32),
        }
        return DensePPCASignificanceResult(
            significant_sample_indices=[None, None],
            n_significant_per_image=np.asarray([2, 2], dtype=np.int32),
            hard_assignment=np.asarray([0, 3], dtype=np.int32),
            best_rotation_idx=diag["best_rotation_idx"],
            best_translation_idx=diag["best_translation_idx"],
            logZ=np.asarray([1.0, 1.0], dtype=np.float64),
            max_posterior_per_image=diag["max_posterior_per_image"],
            diagnostics=diag,
        )

    def _layouts(pose_diagnostics, **kwargs):
        observed["layout_calls"] += 1
        observed.setdefault("layout_top_rot", []).append(np.asarray(pose_diagnostics["halfset0"]["top_rotation_id"]))
        observed.setdefault("layout_kwargs", []).append(kwargs)
        return (f"layout{observed['layout_calls']}_0", f"layout{observed['layout_calls']}_1")

    def _pose_score(state, halfsets, layouts, **kwargs):
        del halfsets, kwargs
        observed["pose_score_layouts"] = layouts
        observed["pose_score_mu"] = np.asarray(state.mu_score).copy()
        diag = {
            "halfset0": {
                "best_rotation_idx": np.asarray([0, 1], dtype=np.int32),
                "best_rotation_id": np.asarray([0, 1], dtype=np.int32),
                "best_translation_idx": np.asarray([0, 1], dtype=np.int32),
                "image_indices": np.asarray([1, 0], dtype=np.int32),
                "max_posterior_per_image": np.ones(2, dtype=np.float32),
                "pmax_mean": 1.0,
                "logZ_mean": 2.0,
                "nsig_mean": 1.0,
                "top_rotation_id": np.asarray([[1, 3], [0, 2]], dtype=np.int32),
                "top_translation_idx": np.asarray([[1, 0], [0, 1]], dtype=np.int32),
            },
            "halfset1": {
                "best_rotation_idx": np.asarray([2, 3], dtype=np.int32),
                "best_rotation_id": np.asarray([2, 3], dtype=np.int32),
                "best_translation_idx": np.asarray([1, 0], dtype=np.int32),
                "image_indices": np.asarray([1, 0], dtype=np.int32),
                "max_posterior_per_image": np.ones(2, dtype=np.float32),
                "pmax_mean": 1.0,
                "logZ_mean": 2.0,
                "nsig_mean": 1.0,
                "top_rotation_id": np.asarray([[3, 1], [2, 0]], dtype=np.int32),
                "top_translation_idx": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
            },
            "delta_rms_mu": 0.0,
            "delta_rms_W": 0.0,
            "pose_score_only": True,
        }
        return state.replace(pose_diagnostics=diag)

    def _local_update(state, halfsets, layouts, **kwargs):
        del halfsets, kwargs
        observed["local_update_layouts"] = layouts
        observed["local_update_start_mu"] = np.asarray(state.mu_score).copy()
        observed["local_update_pose_path"] = state.pose_diagnostics["halfset0"]["path"]
        updated_mu = state.mu_score + jnp.asarray([1.0 + 0.0j], dtype=state.mu_score.dtype)
        return state.replace(mu_half=(updated_mu, updated_mu), mu_score=updated_mu)

    monkeypatch.setattr("recovar.em.ppca_refinement.highres_refinement.get_relion_rotation_grid", _grid)
    monkeypatch.setattr(
        "recovar.em.ppca_refinement.highres_refinement.compute_dense_ppca_adaptive_significance",
        _dense_sig,
    )
    monkeypatch.setattr(
        "recovar.em.ppca_refinement.highres_refinement.build_top_p_local_hypothesis_layouts_from_diagnostics",
        _layouts,
    )
    monkeypatch.setattr(
        "recovar.em.ppca_refinement.highres_refinement.run_local_ppca_halfset_pose_scoring_iteration",
        _pose_score,
    )
    monkeypatch.setattr(
        "recovar.em.ppca_refinement.highres_refinement.run_local_ppca_halfset_fused_em_iteration",
        _local_update,
    )

    result = run_highres_ppca_refinement_with_kclass_pose_hierarchy(
        _state(),
        _FakeDataset(),
        rotations=rotations,
        translations=translations,
        n_pose_warmup_iterations=1,
        n_dense_iterations=0,
        n_adaptive_iterations=0,
        n_local_iterations=1,
        top_p_poses=2,
        init_current_size=4,
        max_current_size=4,
        bridge=_Bridge(do_local_search=True),
        halfset_comparator=_blocked_comparator,
    )

    np.testing.assert_array_equal(observed["dense_mu"][0], np.asarray([0.0 + 0.0j], dtype=np.complex64))
    np.testing.assert_array_equal(observed["pose_score_mu"], np.asarray([0.0 + 0.0j], dtype=np.complex64))
    np.testing.assert_array_equal(observed["local_update_start_mu"], np.asarray([0.0 + 0.0j], dtype=np.complex64))
    assert observed["pose_score_layouts"] == ("layout1_0", "layout1_1")
    assert observed["local_update_layouts"] == ("layout2_0", "layout2_1")
    np.testing.assert_array_equal(
        observed["layout_top_rot"][1],
        np.asarray([[0, 2], [1, 3]], dtype=np.int32),
    )
    assert observed["local_update_pose_path"] == "pose_warmup_local"
    assert np.asarray(result.final_state.mu_score)[0] == np.asarray(1.0 + 0.0j, dtype=np.complex64)
    assert [entry["stage"] for entry in result.diagnostics["stage_history"]] == [
        "pose_warmup_dense",
        "pose_warmup_local",
        "exact_local",
    ]
    assert "pose_probe_dense_vs_input_top1_angle_deg_mean" in result.final_state.pose_diagnostics
