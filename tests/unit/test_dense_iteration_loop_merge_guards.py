"""Merge guards for dense EM iteration-loop refactors.

These are intentionally small structural tests. They protect the dense
single-volume cleanup from future EM / VDAM / PPCA branch merges without
re-running the expensive parity fixtures.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import recovar.em.dense_single_volume.iteration_loop as iteration_loop
import recovar.em.dense_single_volume.local_search_iteration as local_search_iteration
from recovar.em.dense_single_volume import (
    half_scoring,
    mean_helpers,
    ppca_bridge,
    relion_replay,
    relion_worker_scale,
    score_outputs,
    scoring_policy,
)
from recovar.em.dense_single_volume.helpers.convergence import _native_final_perturbation_healpix_order
from recovar.em.dense_single_volume.local_search_iteration import _LocalSearchIterationResult
from recovar.em.initial_model.iteration_loop import run_vdam_iterations

pytestmark = pytest.mark.unit


def _local_score_call_keywords():
    """Resolve the local scorer's literal shared kwargs for source guards."""
    function = ast.parse(inspect.getsource(half_scoring._score_half_local)).body[0]
    dictionaries = {
        node.targets[0].id: dict(zip((key.value for key in node.value.keys), node.value.values))
        for node in function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Dict)
    }
    calls = sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_local_search_iteration"
        ),
        key=lambda node: node.lineno,
    )
    resolved = []
    for call in calls:
        keywords = {}
        for keyword in call.keywords:
            items = dictionaries[keyword.value.id] if keyword.arg is None else {keyword.arg: keyword.value}
            assert not keywords.keys() & items.keys(), "duplicate local keyword"
            keywords.update(items)
        resolved.append({key: ast.unparse(value) for key, value in keywords.items()})
    return resolved


def test_per_half_output_shape_stays_bundled_and_trimmed():
    """The refactor depends on one owner for per-half outputs, without dead fields."""

    assert is_dataclass(score_outputs.HalfScoreResult)
    assert is_dataclass(score_outputs.PerHalfOutputs)
    assert not hasattr(iteration_loop, "IterationRunSpec")
    assert not hasattr(score_outputs.PerHalfOutputs, "for_half")

    half_score_fields = {field.name for field in fields(score_outputs.HalfScoreResult)}
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
        "significant_counts",
        "profile_summary",
        "mstep_full_half_axis",
        "mstep_accumulator_shape",
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

    per_half_fields = {field.name for field in fields(score_outputs.PerHalfOutputs)}
    assert per_half_fields == {
        "hard_assignments",
        "Ft_y",
        "Ft_ctf",
        "coarse_ha",
        "max_posterior",
        "rotation_posterior",
        "class_assignments",
        "class_posterior",
        "class_full_posterior",
        "class_rotation_posterior",
        "noise_stats",
        "noise_stats_per_class",
        "best_pose_rotations",
        "best_pose_rotation_eulers",
        "best_pose_translations",
        "translation_search_bases",
        "pose_rotations",
        "pose_rotation_eulers",
        "mstep_full_half_axis",
        "mstep_accumulator_shape",
    }


def test_per_half_update_from_half_score_result_updates_only_score_payload():
    class _Stats:
        max_posterior_per_image = np.array([0.25, 0.75], dtype=np.float64)
        rotation_posterior_sums = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    outs = score_outputs.PerHalfOutputs.empty()
    hs = score_outputs.HalfScoreResult(
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
        mstep_full_half_axis=0,
        mstep_accumulator_shape=(17, 17, 17),
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
    assert outs.mstep_full_half_axis == [None, 0]
    assert outs.mstep_accumulator_shape == [None, (17, 17, 17)]


def test_per_half_update_preserves_double_posterior_state_in_double_mode(monkeypatch):
    class _Stats:
        max_posterior_per_image = np.array([0.123456789012345], dtype=np.float64)
        rotation_posterior_sums = np.array([0.987654321098765], dtype=np.float64)

    monkeypatch.setitem(scoring_policy._DENSE_EM_STATIC_KWARGS, "use_float64_scoring", True)
    outs = iteration_loop.PerHalfOutputs.empty()
    outs.update_from(
        0,
        iteration_loop.HalfScoreResult(
            ha=np.array([0], dtype=np.int32),
            Ft_y=None,
            Ft_ctf=None,
            em_stats=_Stats(),
            noise_stats=None,
        ),
        dtype=scoring_policy._dense_global_scoring_dtype(),
    )

    assert outs.max_posterior[0].dtype == np.float64
    assert outs.rotation_posterior[0].dtype == np.float64
    assert outs.max_posterior[0][0] == _Stats.max_posterior_per_image[0]
    assert outs.rotation_posterior[0][0] == _Stats.rotation_posterior_sums[0]


def test_mstep_full_half_axis_resolver_keeps_common_axis_or_default():
    assert score_outputs._resolve_mstep_full_half_axis([None, None]) == -1
    assert score_outputs._resolve_mstep_full_half_axis([None, 0]) == 0
    assert score_outputs._resolve_mstep_full_half_axis([0, 0]) == 0

    with pytest.raises(RuntimeError, match="full-half axes disagree"):
        score_outputs._resolve_mstep_full_half_axis([0, -1])


def test_local_search_keeps_relion_x_half_mstep_contract():
    source = inspect.getsource(half_scoring._score_half_local)

    assert "if k_class_enabled" in source
    assert "_k_class_relion_x_half_mstep_enabled()" in source
    assert "else _k1_relion_x_half_mstep_enabled()" in source
    assert "mstep_relion_x_half=local_relion_x_half_mstep" in source
    assert "mstep_full_half_axis=0 if local_relion_x_half_mstep else None" in source
    assert "mstep_accumulator_shape=(" in source
    assert "relion_backprojector_volume_shape(" in source
    assert "reconstruction_current_size_for_engine = (" in source
    assert "if model_current_size_for_engine is None" in source
    assert "else model_current_size_for_engine" in source
    assert "current_size=reconstruction_current_size_for_engine" in source[
        source.index("mstep_accumulator_shape=(") :
    ]


def test_empty_k1_local_or_adaptive_half_keeps_relion_x_half_shape_contract():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    empty_start = source.index("if experiment_datasets[k].n_units == 0:")
    empty_source = source[empty_start : source.index("continue\n            if use_local:", empty_start)]

    assert "empty_k1_x_half_mstep = (" in empty_source
    assert "and (use_local or use_adaptive)" in empty_source
    assert "relion_backprojector_volume_shape(" in empty_source
    assert "current_size=(" in empty_source
    assert "if model_current_size_for_engine is None" in empty_source
    assert "else model_current_size_for_engine" in empty_source
    assert "half_volume_accumulator_shape(empty_mstep_accumulator_shape)" in empty_source
    assert "relion_x_half_accumulators_to_public_layout(" in empty_source
    assert "mstep_full_half_axis=0 if empty_k1_x_half_mstep else None" in empty_source
    assert "mstep_accumulator_shape=empty_mstep_accumulator_shape" in empty_source


def test_relion_norm_scale_updates_are_not_disabled_for_k_class():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    update_start = source.index("can_update_norm_scale = (")
    update_source = source[update_start : source.index("history.record_noise_and_tau2(", update_start)]

    assert "not k_class_enabled" not in update_source
    assert "update_relion_norm_scale_corrections(" in update_source
    assert "experiment_datasets[_half_idx].n_units" in update_source
    assert "np.zeros(int(experiment_datasets[_half_idx].n_units), dtype=np.int64)" in update_source
    assert "_format_relion_correction_range(norm_scale_update.image_corrections_per_half[0])" in update_source
    assert "_format_relion_correction_range(norm_scale_update.image_corrections_per_half[1])" in update_source
    assert "np.min(np.asarray(norm_scale_update.image_corrections_per_half" not in update_source


def test_relion_correction_range_formatter_accepts_empty_halves():
    assert relion_worker_scale._format_relion_correction_range(np.array([], dtype=np.float32)) == "empty"
    assert relion_worker_scale._format_relion_correction_range(np.array([0.5, 2.0], dtype=np.float32)) == "[0.5, 2]"


def test_k1_local_search_significant_reconstruction_uses_actual_local_oversampling():
    source = inspect.getsource(half_scoring._score_half_local)

    assert "local_reconstruct_significant_only = int(local_parent_oversampling_order) > 0" in source
    assert "local_reconstruct_significant_only = state.adaptive_oversampling > 0" not in source


def test_k1_local_search_stats_use_relion_retained_weights():
    source = inspect.getsource(half_scoring._score_half_local)
    wrapper_source = inspect.getsource(local_search_iteration._run_local_search_iteration)

    assert "stats_use_reconstruction_probs=local_reconstruct_significant_only" in source
    assert "stats_use_reconstruction_probs=False" in wrapper_source
    assert "stats_use_reconstruction_probs=stats_use_reconstruction_probs" in wrapper_source


def test_fresh_k1_spectrum_norm_reaches_local_noise_update_only():
    score_source = inspect.getsource(half_scoring._score_half_local)
    wrapper_source = inspect.getsource(local_search_iteration._run_local_search_iteration)
    loop_source = inspect.getsource(iteration_loop._run_relion_iteration_loop)

    assert "if source_faithful_spectrum_norm and k_class_enabled:" in score_source
    calls = _local_score_call_keywords()
    assert len(calls) == 3
    assert all(call["source_faithful_spectrum_norm"] == "source_faithful_spectrum_norm" for call in calls)
    assert "if source_faithful_spectrum_norm:" in wrapper_source
    assert "fresh K=1-only" in wrapper_source
    assert "source_faithful_spectrum_norm=source_faithful_spectrum_norm" in wrapper_source
    local_dispatch = loop_source[
        loop_source.index("if use_local:") : loop_source.index("elif use_adaptive:")
    ]
    assert "source_faithful_spectrum_norm=source_faithful_spectrum_norm" in local_dispatch


def test_k1_local_full_parent_diagnostic_counts_unmasked_parent_layout():
    source = inspect.getsource(half_scoring._score_half_local)

    assert "if parent_layout.sample_mask_flat is not None" in source
    assert "else int(stop - start) * int(current_translations.shape[0])" in source


def test_k1_local_parent_probe_applies_relion_max_significants_cap():
    score_source = inspect.getsource(half_scoring._score_half_local)
    parent_call = score_source[
        score_source.index("parent_outputs = _run_local_search_iteration") : score_source.index(
            "parent_profile = parent_outputs.profile_summary"
        )
    ]
    assert _local_score_call_keywords()[0]["max_significants"] == "max_significants"
    assert "apply_max_significants_to_support=True" in parent_call

    wrapper_source = inspect.getsource(local_search_iteration._run_local_search_iteration)
    assert "apply_max_significants_to_support=False" in wrapper_source
    assert "max_significants=max_significants if apply_max_significants_to_support else -1" in wrapper_source


def test_k1_local_records_coarse_parent_support_not_fine_reconstruction_count():
    source = inspect.getsource(half_scoring._score_half_local)

    parent_count_start = source.index("pruned_parent_significant_sample_indices = significant_sample_indices")
    parent_count_end = source.index("if local_adaptive_pass2_full_parent:", parent_count_start)
    parent_count_source = source[parent_count_start:parent_count_end]
    assert "relion_significant_counts_k = _relion_coarse_significant_counts(" in parent_count_source
    assert "return_significant_counts=False" in source
    assert "significant_counts=relion_significant_counts_k" in source

    counts = half_scoring._relion_coarse_significant_counts(
        [np.array([2, 8], dtype=np.int64), np.array([1, 3, 5, 7], dtype=np.int64)]
    )
    np.testing.assert_array_equal(counts, np.array([2, 4], dtype=np.int32))
    assert half_scoring._relion_coarse_significant_counts([np.array([2]), None]) is None


def test_k1_local_search_does_not_score_learned_global_direction_prior():
    source = inspect.getsource(half_scoring._score_half_local)
    assert "RELION's convertAllSquaredDifferencesToWeights uses mymodel.pdf_direction" in source
    assert "relion_local_rotation_log_prior_k = None" in source
    assert "rotation_log_prior=relion_local_rotation_log_prior_k" in source
    assert "else relion_local_rotation_log_prior_k" in source

    loop_source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    prior_loop = loop_source[
        loop_source.index("for _half_idx in range(2):") : loop_source.index("# --- Run E+M on each half-set ---")
    ]
    assert "if use_local:" in prior_loop
    assert "continue" in prior_loop

    final_prior_start = loop_source.index("final_rotation_log_prior_k = None")
    final_prior_block = loop_source[
        final_prior_start : loop_source.index("if final_use_local:", final_prior_start)
    ]
    assert "if not final_use_local:" in final_prior_block
    assert "use_local=False" in final_prior_block


def test_k1_local_search_passes_relion_x_half_mstep(monkeypatch):
    captured = {}

    class _Stats:
        max_posterior_per_image = np.array([1.0], dtype=np.float32)
        rotation_posterior_sums = np.array([1.0], dtype=np.float32)

    def fake_run_local_search_iteration(*_args, **kwargs):
        best_rotation = np.array(
            [
                [0.93629336, -0.27509585, 0.21835066],
                [0.28962948, 0.95642509, -0.03695701],
                [-0.19866933, 0.09784340, 0.97517033],
            ],
            dtype=np.float32,
        )
        captured.update(kwargs)
        current_size_shape = (19, 19, 19)
        outputs = _LocalSearchIterationResult(
            Ft_y=np.zeros(int(np.prod(current_size_shape)), dtype=np.complex64),
            Ft_ctf=np.zeros(int(np.prod(current_size_shape)), dtype=np.float32),
            hard_assignment=np.array([0], dtype=np.int32),
            best_pose_rotations=best_rotation[None, :, :],
            best_pose_translations=np.zeros((1, 2), dtype=np.float32),
            best_pose_rotation_ids=np.array([0], dtype=np.int32),
            relion_stats=_Stats(),
            noise_stats="noise",
        )
        if kwargs.get("return_significant_counts"):
            outputs.significant_counts = np.array([7], dtype=np.int32)
        return outputs

    monkeypatch.delenv("RECOVAR_K1_RELION_X_HALF_MSTEP", raising=False)
    monkeypatch.setattr(scoring_policy, "_k1_relion_x_half_mstep_default_available", lambda: True)
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", fake_run_local_search_iteration)

    result = half_scoring._score_half_local(
        k=0,
        experiment_dataset=SimpleNamespace(
            voxel_size=1.0,
            image_shape=(16, 16),
            volume_shape=(16, 16, 16),
        ),
        means_k="mean",
        mean_variance="variance",
        noise_variance_k="noise_variance",
        previous_best_rotation_eulers_k=np.zeros((1, 3), dtype=np.float32),
        local_search_rotations=np.eye(3, dtype=np.float32)[None, :, :],
        local_search_rotation_eulers=np.zeros((1, 3), dtype=np.float32),
        local_search_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        current_translations=np.zeros((1, 2), dtype=np.float32),
        base_translations=np.zeros((1, 2), dtype=np.float32),
        trans_prior_center=np.zeros((1, 2), dtype=np.float32),
        trans_prior_center_for_engine=np.zeros((1, 2), dtype=np.float32),
        current_sigma_offset_angstrom=1.0,
        current_translation_range=1.0,
        disc_type="linear_interp",
        cs_for_engine=8,
        model_current_size_for_engine=8,
        local_pass1_current_size=8,
        image_corrections_k=None,
        scale_corrections_k=None,
        translation_search_base=None,
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        max_significants=-1,
        iteration=0,
        save_intermediates_dir=None,
        local_search_random_perturbation=0.0,
        local_search_angular_sampling_deg=None,
        local_parent_oversampling_order=0,
        local_search_translation_prior_mode="coarse",
        replay_prior_translations=None,
        class_log_priors=None,
        k_class_enabled=False,
        collect_local_search_profile=False,
        diagnostic_score_only=False,
        safe_batch_sizes=lambda *_args, **_kwargs: (2, 3),
        outputs=score_outputs.PerHalfOutputs.empty(),
        local_profile_history=[],
    )

    assert captured["mstep_relion_x_half"] is True
    assert captured["return_significant_counts"] is False
    assert result.significant_counts is None
    assert result.mstep_full_half_axis == 0
    assert result.mstep_accumulator_shape == (19, 19, 19)


@pytest.mark.parametrize("denominator_mode", [None, "full_parent", "rotation_only"])
@pytest.mark.parametrize("spectrum_norm", [False, True])
def test_k1_local_search_records_parent_counts_without_changing_fine_mstep(
    monkeypatch, denominator_mode, spectrum_norm
):
    parent_counts = np.array([2, 3], dtype=np.int32)
    best_rotation = np.array(
        [
            [0.93629336, -0.27509585, 0.21835066],
            [0.28962948, 0.95642509, -0.03695701],
            [-0.19866933, 0.09784340, 0.97517033],
        ],
        dtype=np.float32,
    )
    calls = []

    class _Stats:
        log_evidence_per_image = np.array([-2.0, -3.0], dtype=np.float64)
        max_posterior_per_image = np.array([0.75, 0.5], dtype=np.float32)
        rotation_posterior_sums = np.array([1.0, 1.0], dtype=np.float32)

    parent_layout = SimpleNamespace(
        rotation_counts=np.array([2, 2], dtype=np.int32),
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        sample_mask_flat=None,
        translation_grid=np.zeros((1, 2), dtype=np.float32),
    )
    fine_layout = SimpleNamespace(
        rotation_counts=np.array([2, 2], dtype=np.int32),
        rotation_offsets=np.array([0, 2, 4], dtype=np.int64),
        sample_mask_flat=None,
        translation_grid=np.zeros((4, 2), dtype=np.float32),
    )

    def fake_run_local_search_iteration(*_args, **kwargs):
        calls.append(dict(kwargs))
        if kwargs["score_only"]:
            return _LocalSearchIterationResult(
                Ft_y="parent_ft_y",
                Ft_ctf="parent_ft_ctf",
                hard_assignment=np.zeros(2, dtype=np.int32),
                relion_stats=_Stats(),
                profile_summary={
                    "reconstruction_sample_indices_by_image": (
                        np.array([0, 1], dtype=np.int64),
                        np.array([0, 1, 2], dtype=np.int64),
                    ),
                },
            )
        return _LocalSearchIterationResult(
            Ft_y="fine_ft_y",
            Ft_ctf="fine_ft_ctf",
            hard_assignment=np.array([4, 5], dtype=np.int32),
            best_pose_rotations=np.broadcast_to(best_rotation, (2, 3, 3)).copy(),
            best_pose_translations=np.zeros((2, 2), dtype=np.float32),
            best_pose_rotation_ids=np.array([0, 1], dtype=np.int32),
            relion_stats=_Stats(),
            noise_stats="fine_noise",
        )

    monkeypatch.setattr(half_scoring, "build_local_search_grid_metadata", lambda _order: {})
    monkeypatch.setattr(half_scoring, "build_local_hypothesis_layout", lambda *_args, **_kwargs: parent_layout)
    monkeypatch.setattr(
        half_scoring,
        "build_local_adaptive_pass2_hypothesis_layout",
        lambda *_args, **_kwargs: fine_layout,
    )
    monkeypatch.setattr(half_scoring, "_local_adaptive_pass2_full_parent_enabled", lambda: False)
    monkeypatch.setattr(half_scoring, "_local_adaptive_pass2_rotation_only_enabled", lambda: False)
    monkeypatch.setattr(half_scoring, "_local_adaptive_pass2_denominator_support_mode", lambda: denominator_mode)
    monkeypatch.setattr(half_scoring, "_k1_relion_x_half_mstep_enabled", lambda: False)
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", fake_run_local_search_iteration)

    result = half_scoring._score_half_local(
        k=0,
        experiment_dataset=SimpleNamespace(
            voxel_size=1.0,
            image_shape=(16, 16),
            volume_shape=(16, 16, 16),
        ),
        means_k="mean",
        mean_variance="variance",
        noise_variance_k="noise_variance",
        previous_best_rotation_eulers_k=np.zeros((2, 3), dtype=np.float32),
        local_search_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy(),
        local_search_rotation_eulers=np.zeros((2, 3), dtype=np.float32),
        local_search_order=1,
        sigma_rot=0.1,
        sigma_psi=0.1,
        current_translations=np.zeros((1, 2), dtype=np.float32),
        base_translations=np.zeros((1, 2), dtype=np.float32),
        trans_prior_center=np.zeros((2, 2), dtype=np.float32),
        trans_prior_center_for_engine=np.zeros((2, 2), dtype=np.float32),
        current_sigma_offset_angstrom=1.0,
        current_translation_range=1.0,
        disc_type="linear_interp",
        cs_for_engine=8,
        local_pass1_current_size=8,
        image_corrections_k=None,
        scale_corrections_k=None,
        translation_search_base=None,
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        max_significants=23,
        source_faithful_spectrum_norm=spectrum_norm,
        iteration=3,
        save_intermediates_dir=None,
        local_search_random_perturbation=0.0,
        local_search_angular_sampling_deg=None,
        local_parent_oversampling_order=1,
        local_search_translation_prior_mode="coarse",
        replay_prior_translations=None,
        class_log_priors=None,
        k_class_enabled=False,
        collect_local_search_profile=False,
        diagnostic_score_only=False,
        safe_batch_sizes=lambda *_args, **_kwargs: (2, 3),
        outputs=score_outputs.PerHalfOutputs.empty(),
        local_profile_history=[],
    )

    assert len(calls) == (2 if denominator_mode is None else 3)
    parent_call, fine_call = calls[0], calls[-1]
    assert all(call["source_faithful_spectrum_norm"] is spectrum_norm for call in calls)
    assert all(call["max_significants"] == 23 for call in calls)
    if denominator_mode is None:
        assert fine_call["normalization_log_evidence"] is None
    else:
        denominator_call = calls[1]
        assert denominator_call["score_only"] is True
        assert denominator_call["accumulate_noise"] is False
        assert denominator_call["disable_adjoint_y"] is True
        assert denominator_call["disable_adjoint_ctf"] is True
        assert denominator_call["return_best_pose_details"] is False
        np.testing.assert_array_equal(
            fine_call["normalization_log_evidence"], _Stats.log_evidence_per_image
        )
    assert parent_call["score_only"] is True
    assert parent_call.get("return_significant_counts", False) is False
    assert parent_call["apply_max_significants_to_support"] is True
    assert parent_call["max_significants"] == 23
    assert fine_call["score_only"] is False
    assert fine_call["reconstruct_significant_only"] is True
    assert fine_call["stats_use_reconstruction_probs"] is True
    assert fine_call["return_significant_counts"] is False
    assert result.Ft_y == "fine_ft_y"
    assert result.Ft_ctf == "fine_ft_ctf"
    assert result.noise_stats == "fine_noise"
    np.testing.assert_array_equal(result.significant_counts, parent_counts)


def test_kclass_local_search_passes_relion_x_half_mstep(monkeypatch):
    captured = {}

    class _Stats:
        max_posterior_per_image = np.array([1.0], dtype=np.float32)
        rotation_posterior_sums = np.array([1.0], dtype=np.float32)

    def fake_run_local_search_iteration(*_args, **kwargs):
        best_rotation = np.array(
            [
                [0.93629336, -0.27509585, 0.21835066],
                [0.28962948, 0.95642509, -0.03695701],
                [-0.19866933, 0.09784340, 0.97517033],
            ],
            dtype=np.float32,
        )
        captured.update(kwargs)
        current_size_shape = (19, 19, 19)
        return _LocalSearchIterationResult(
            Ft_y=np.zeros((2, int(np.prod(current_size_shape))), dtype=np.complex64),
            Ft_ctf=np.zeros((2, int(np.prod(current_size_shape))), dtype=np.float32),
            hard_assignment=np.array([0], dtype=np.int32),
            best_pose_rotations=best_rotation[None, :, :],
            best_pose_translations=np.zeros((1, 2), dtype=np.float32),
            best_pose_rotation_ids=np.array([0], dtype=np.int32),
            relion_stats=_Stats(),
            noise_stats="noise",
            class_assignments=np.array([1], dtype=np.int32),
            class_posterior_sums=np.array([0.25, 0.75], dtype=np.float64),
            class_full_posterior_sums=np.array([0.2, 0.8], dtype=np.float64),
        )

    monkeypatch.setattr(half_scoring, "_k_class_relion_x_half_mstep_enabled", lambda: True)
    monkeypatch.setattr(half_scoring, "_run_local_search_iteration", fake_run_local_search_iteration)

    result = half_scoring._score_half_local(
        k=0,
        experiment_dataset=SimpleNamespace(
            voxel_size=1.0,
            image_shape=(16, 16),
            volume_shape=(16, 16, 16),
        ),
        means_k="mean",
        mean_variance="variance",
        noise_variance_k="noise_variance",
        previous_best_rotation_eulers_k=np.zeros((1, 3), dtype=np.float32),
        local_search_rotations=np.eye(3, dtype=np.float32)[None, :, :],
        local_search_rotation_eulers=np.zeros((1, 3), dtype=np.float32),
        local_search_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        current_translations=np.zeros((1, 2), dtype=np.float32),
        base_translations=np.zeros((1, 2), dtype=np.float32),
        trans_prior_center=np.zeros((1, 2), dtype=np.float32),
        trans_prior_center_for_engine=np.zeros((1, 2), dtype=np.float32),
        current_sigma_offset_angstrom=1.0,
        current_translation_range=1.0,
        disc_type="linear_interp",
        cs_for_engine=8,
        model_current_size_for_engine=8,
        local_pass1_current_size=8,
        image_corrections_k=None,
        scale_corrections_k=None,
        translation_search_base=None,
        disable_adjoint_y=False,
        disable_adjoint_ctf=False,
        max_significants=-1,
        iteration=0,
        save_intermediates_dir=None,
        local_search_random_perturbation=0.0,
        local_search_angular_sampling_deg=None,
        local_parent_oversampling_order=0,
        local_search_translation_prior_mode="coarse",
        replay_prior_translations=None,
        class_log_priors=np.log(np.array([0.5, 0.5], dtype=np.float64)),
        k_class_enabled=True,
        collect_local_search_profile=False,
        diagnostic_score_only=False,
        safe_batch_sizes=lambda *_args, **_kwargs: (2, 3),
        outputs=score_outputs.PerHalfOutputs.empty(),
        local_profile_history=[],
    )

    assert captured["mstep_relion_x_half"] is True
    assert result.mstep_full_half_axis == 0
    assert result.mstep_accumulator_shape == (19, 19, 19)


def test_final_all_data_iteration_stays_on_shared_dense_scoring_path():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    final_marker = "final_outs = PerHalfOutputs.empty()"
    assert final_marker in source
    final_block = source[source.index(final_marker) :]
    final_reconstruct_marker = "RELION final all-data reconstruction start"
    assert final_reconstruct_marker in final_block
    final_reconstruct_block = final_block[final_block.index(final_reconstruct_marker) :]

    assert "final_current_size = int(grid_size)" in source
    assert final_block.count("_score_half_dense_in_bpref_scope(") == 1
    assert "bpref_device_signature_active=False" in final_block
    assert "cs_for_engine=final_current_size" in final_block
    # Four call sites: K-class, merged K1, regularized half pair and unfiltered
    # half pair. The convergence smoke test checks the five executed K1 calls.
    assert final_reconstruct_block.count("current_size=final_current_size") == 4
    assert final_reconstruct_block.count("tau=None") == 1
    assert final_reconstruct_block.count("use_spherical_mask=True") == 1
    assert "do_map=false only omits the tau2 prior" in final_reconstruct_block
    solvent_fsc_marker = "Computed iter-%d solvent-corrected true FSC"
    solvent_fsc_block = source[: source.index(solvent_fsc_marker)]
    solvent_fsc_block = solvent_fsc_block[solvent_fsc_block.rindex("unfiltered_half_maps = []") :]
    assert solvent_fsc_block.count("use_spherical_mask=True") == 1
    assert '"unfiltered_means": final_unfiltered_means_for_output' in final_reconstruct_block
    prejoin_save = source.index("final_unfiltered_Ft_y_0 = final_Ft_y_0")
    lowres_join = source.index("regularization.join_halves_at_low_resolution(", prejoin_save)
    assert prejoin_save < lowres_join
    unfiltered_start = final_reconstruct_block.index("final_unfiltered_means_for_output = [")
    unfiltered_block = final_reconstruct_block[unfiltered_start:]
    assert "final_unfiltered_Ft_ctf_0" in unfiltered_block
    assert "final_unfiltered_Ft_y_0" in unfiltered_block
    assert "final_unfiltered_Ft_ctf_1" in unfiltered_block
    assert "final_unfiltered_Ft_y_1" in unfiltered_block
    assert "relion_firstiter_cc_this_iter=False" in final_block
    assert "return_best_pose_details=not k_class_enabled" in final_block
    assert "run_em(" not in final_block
    assert "run_dense_k_class_em(" not in final_block
    assert "run_dense_k_class_em_adaptive(" not in final_block


def test_final_all_data_tau2_uses_joined_half_weight_sum():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    marker = "final_mean_variance, _, final_tau2_update_details = regularization.compute_relion_tau2_from_weights("
    assert marker in source
    tau2_call = source[source.index(marker) : source.index("        logger.info(", source.index(marker))]

    assert 'weight_combination="sum"' in tau2_call
    assert '"tau2_weight_combination": np.asarray("class_iref" if k_class_enabled else "sum")' in source
    assert '"tau2_weight_combination_final_all_data": "class_iref" if k_class_enabled else "sum"' in source


def test_kclass_final_all_data_recomputes_tau2_from_iref_and_returns_final_means():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    final_marker = "final_ft_y = final_Ft_y_0 + final_Ft_y_1"
    final_block = source[source.index(final_marker) :]
    kclass_marker = "if k_class_enabled:"
    kclass_tau2_block = final_block[
        final_block.index(kclass_marker) : final_block.index("    else:", final_block.index(kclass_marker))
    ]

    assert "regularization.compute_relion_tau2_from_iref_power_spectrum(" in kclass_tau2_block
    assert "final_join_means[0][class_idx]" in kclass_tau2_block
    assert "current_size=final_current_size" in kclass_tau2_block
    assert "regularization.compute_data_vs_prior(" in kclass_tau2_block
    assert "full_half_axis=final_mstep_full_half_axis" in kclass_tau2_block
    assert "accumulator_volume_shape=final_mstep_accumulator_shape" in kclass_tau2_block
    assert "final_mean_variance = jnp.stack(final_mean_variance_per_class, axis=0)" in kclass_tau2_block

    reconstruct_marker = "if k_class_enabled:"
    reconstruct_block = final_block[final_block.rindex(reconstruct_marker) :]
    assert "final_means_for_output = [final_class_means, final_class_means]" in reconstruct_block
    assert '"tau2_weight_combination_final_all_data": "class_iref" if k_class_enabled else "sum"' in source
    assert 'final_tau2_update_details.get("fsc_shells") is None' in source


def test_final_all_data_grid_correction_defaults_to_gui_quality(monkeypatch):
    monkeypatch.delenv(iteration_loop._FINAL_ALL_DATA_GRID_CORRECT_ENV, raising=False)

    assert iteration_loop._final_all_data_grid_correct_enabled() is False

    monkeypatch.setenv(iteration_loop._FINAL_ALL_DATA_GRID_CORRECT_ENV, "0")
    assert iteration_loop._final_all_data_grid_correct_enabled() is False

    monkeypatch.setenv(iteration_loop._FINAL_ALL_DATA_GRID_CORRECT_ENV, "unexpected")
    assert iteration_loop._final_all_data_grid_correct_enabled() is False

    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    assert "RELION final all-data reconstruction gridding correction enabled" in source
    assert "RELION final all-data reconstruction gridding correction disabled" in source
    assert "grid_correct=final_grid_correct" in source


def test_final_all_data_local_search_uses_replayed_translation_range():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    final_call_idx = source.index("final_result = _score_half_local_in_bpref_scope(")
    final_call = source[final_call_idx : source.index("            )", final_call_idx)]
    assert "current_translation_range=final_translation_range" in final_call
    assert "current_translation_range=float(state.translation_range)" not in final_call
    assert "debug_iteration=final_sampling_relion_iteration" in final_call


def test_final_all_data_sampling_replay_prefers_final_sampling_star_before_last_numbered():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    marker = "final_sampling_candidates = ["
    start = source.index(marker)
    block = source[start : source.index("        for candidate_path", start)]

    final_numbered = (
        'f"{perturb_replay_relion_prefix}_it{final_sampling_relion_iteration:03d}_sampling.star"'
    )
    last_numbered = (
        'f"{perturb_replay_relion_prefix}_it{final_numbered_sampling_relion_iteration:03d}_sampling.star"'
    )
    run_sampling = 'f"{perturb_replay_relion_prefix}_sampling.star"'

    assert final_numbered in block
    assert last_numbered in block
    assert run_sampling in block
    assert block.index(final_numbered) < block.index(run_sampling) < block.index(last_numbered)
    assert '"final-numbered"' in block
    assert '"final"' in block
    assert '"last-numbered"' in block


def test_native_final_perturbation_uses_active_local_order_but_preserves_global_order():
    local_state = SimpleNamespace(do_local_search=True, healpix_order=4)
    global_state = SimpleNamespace(do_local_search=False, healpix_order=4)

    assert _native_final_perturbation_healpix_order(local_state, 3) == 4
    assert _native_final_perturbation_healpix_order(global_state, 3) == 3

    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    assert "final_perturbation_healpix_order = _native_final_perturbation_healpix_order(" in source


def test_iteration_dependencies_and_ppca_vdam_entry_points_are_available():
    required_iteration_loop_symbols = [
        "_maybe_dump_noise_update_debug",
        "_save_iteration_intermediates",
        "advance_relion_perturbation",
        "apply_relion_rotation_perturbation",
        "apply_relion_rotation_perturbation_to_eulers",
        "apply_relion_translation_perturbation",
        "read_relion_optimiser_metadata",
        "read_relion_sampling_metadata",
    ]

    missing = [name for name in required_iteration_loop_symbols if not hasattr(iteration_loop, name)]
    assert missing == []
    assert hasattr(half_scoring, "build_local_hypothesis_layout")
    assert callable(relion_replay.read_relion_model_metadata)
    assert callable(local_search_iteration.run_local_em_exact)
    assert callable(local_search_iteration.run_local_k_class_em)
    assert callable(mean_helpers._align_fourier_volume_sign_to_reference)
    assert callable(mean_helpers._combined_noise_stats)
    assert callable(relion_replay._replay_control_model_iteration)
    assert callable(relion_replay.read_relion_direction_prior)
    assert callable(relion_replay.read_relion_direction_priors)
    assert iteration_loop._translation_grid_for_class_count is relion_replay._translation_grid_for_class_count
    assert callable(ppca_bridge.PPCAKClassScheduleBridge)
    assert callable(ppca_bridge.run_dense_ppca_refinement_with_kclass_schedule)
    assert callable(ppca_bridge.run_local_ppca_refinement_with_kclass_schedule)
    assert callable(run_vdam_iterations)


def test_local_adaptive_pass2_defaults_to_relion_pruned_parent(monkeypatch):
    monkeypatch.delenv(scoring_policy._LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV, raising=False)
    monkeypatch.delenv(scoring_policy._LOCAL_ADAPTIVE_PASS2_DISABLE_FULL_PARENT_ENV, raising=False)

    assert scoring_policy._local_adaptive_pass2_full_parent_enabled() is False

    monkeypatch.setenv(scoring_policy._LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV, "1")
    assert scoring_policy._local_adaptive_pass2_full_parent_enabled() is True

    monkeypatch.setenv(scoring_policy._LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV, "0")
    assert scoring_policy._local_adaptive_pass2_full_parent_enabled() is False

    monkeypatch.setenv(scoring_policy._LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV, "1")
    monkeypatch.setenv(scoring_policy._LOCAL_ADAPTIVE_PASS2_DISABLE_FULL_PARENT_ENV, "1")
    assert scoring_policy._local_adaptive_pass2_full_parent_enabled() is False
