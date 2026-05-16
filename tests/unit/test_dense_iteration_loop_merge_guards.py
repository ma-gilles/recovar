"""Merge guards for dense EM iteration-loop refactors.

These are intentionally small structural tests. They protect the dense
single-volume cleanup from future EM / VDAM / PPCA branch merges without
re-running the expensive parity fixtures.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import inspect

import numpy as np
import pytest

import recovar.em.dense_single_volume.iteration_loop as iteration_loop
from recovar.em.initial_model.iteration_loop import run_vdam_iterations

pytestmark = pytest.mark.unit


def test_per_half_output_shape_stays_bundled_and_trimmed():
    """The refactor depends on one owner for per-half outputs, without dead fields."""

    assert is_dataclass(iteration_loop.HalfScoreResult)
    assert is_dataclass(iteration_loop.PerHalfOutputs)
    assert not hasattr(iteration_loop, "IterationRunSpec")
    assert not hasattr(iteration_loop.PerHalfOutputs, "for_half")

    half_score_fields = {field.name for field in fields(iteration_loop.HalfScoreResult)}
    assert half_score_fields == {
        "ha",
        "Ft_y",
        "Ft_ctf",
        "em_stats",
        "noise_stats",
        "best_pose_rotations",
        "best_pose_rotation_eulers",
        "best_pose_translations",
        "coarse_ha",
        "pose_rotations",
        "pose_rotation_eulers",
    }
    assert half_score_fields.isdisjoint(
        {
            "adaptive_os_local",
            "class_assignments",
            "class_posterior",
            "class_rotation_posterior",
            "noise_stats_per_class",
            "rot_pmap_for_collapse",
        }
    )

    per_half_fields = {field.name for field in fields(iteration_loop.PerHalfOutputs)}
    assert per_half_fields == {
        "hard_assignments",
        "Ft_y",
        "Ft_ctf",
        "coarse_ha",
        "max_posterior",
        "rotation_posterior",
        "class_assignments",
        "class_posterior",
        "class_rotation_posterior",
        "noise_stats",
        "noise_stats_per_class",
        "best_pose_rotations",
        "best_pose_rotation_eulers",
        "best_pose_translations",
        "translation_search_bases",
        "pose_rotations",
        "pose_rotation_eulers",
    }


def test_per_half_update_from_half_score_result_updates_only_score_payload():
    class _Stats:
        max_posterior_per_image = np.array([0.25, 0.75], dtype=np.float64)
        rotation_posterior_sums = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    outs = iteration_loop.PerHalfOutputs.empty()
    hs = iteration_loop.HalfScoreResult(
        ha=np.array([0, 1], dtype=np.int32),
        Ft_y="ft_y",
        Ft_ctf="ft_ctf",
        em_stats=_Stats(),
        noise_stats="noise",
        best_pose_rotations=np.eye(3, dtype=np.float32)[None, :, :],
        best_pose_rotation_eulers=np.zeros((1, 3), dtype=np.float32),
        best_pose_translations=np.zeros((1, 2), dtype=np.float32),
        coarse_ha=np.array([1, 0], dtype=np.int32),
        pose_rotations="pose_rotations",
        pose_rotation_eulers="pose_eulers",
    )

    outs.update_from(1, hs)

    assert outs.hard_assignments == [None, hs.ha]
    assert outs.Ft_y == [None, "ft_y"]
    assert outs.Ft_ctf == [None, "ft_ctf"]
    assert outs.noise_stats == [None, "noise"]
    np.testing.assert_array_equal(outs.max_posterior[1], np.array([0.25, 0.75], dtype=np.float32))
    np.testing.assert_array_equal(outs.rotation_posterior[1], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert outs.class_assignments == [None, None]
    assert outs.class_posterior == [None, None]
    assert outs.class_rotation_posterior == [None, None]
    assert outs.noise_stats_per_class == [None, None]


def test_final_all_data_iteration_stays_on_shared_dense_scoring_path():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    final_marker = "final_outs = PerHalfOutputs.empty()"
    assert final_marker in source
    final_block = source[source.index(final_marker) :]

    assert "final_current_size = int(grid_size)" in source
    assert final_block.count("_score_half_dense(") == 1
    assert "cs_for_engine=final_current_size" in final_block
    assert "relion_firstiter_cc_this_iter=False" in final_block
    assert "return_best_pose_details=False" in final_block
    assert "run_em(" not in final_block
    assert "run_dense_k_class_em(" not in final_block
    assert "run_dense_k_class_em_adaptive(" not in final_block


def test_iteration_loop_monkeypatch_ppca_and_vdam_surfaces_survive_merges():
    required_iteration_loop_symbols = [
        "_align_fourier_volume_sign_to_reference",
        "_combined_noise_stats",
        "_maybe_dump_noise_update_debug",
        "_replay_control_model_iteration",
        "_save_iteration_intermediates",
        "advance_relion_perturbation",
        "apply_relion_rotation_perturbation",
        "apply_relion_rotation_perturbation_to_eulers",
        "apply_relion_translation_perturbation",
        "build_local_hypothesis_layout",
        "compute_data_vs_prior",
        "get_relion_rotation_grid",
        "get_relion_rotation_grid_eulers",
        "get_translation_grid",
        "PPCAKClassScheduleBridge",
        "read_relion_direction_prior",
        "read_relion_direction_priors",
        "read_relion_model_metadata",
        "read_relion_optimiser_metadata",
        "read_relion_sampling_metadata",
        "run_dense_ppca_refinement_with_kclass_schedule",
        "run_local_em_exact",
        "run_local_k_class_em",
        "run_local_ppca_refinement_with_kclass_schedule",
    ]

    missing = [name for name in required_iteration_loop_symbols if not hasattr(iteration_loop, name)]
    assert missing == []
    assert callable(run_vdam_iterations)
