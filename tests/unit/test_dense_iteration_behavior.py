"""Behavioral contracts for dense and local iteration helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import recovar.em.refinement.iteration_loop as iteration_loop
from recovar.em.dense import half_scoring, score_outputs, scoring_policy
from recovar.em.diagnostics import local_debug
from recovar.em.helpers.convergence import _native_final_perturbation_healpix_order
from recovar.em.local.local_search_iteration import _LocalSearchIterationResult
from recovar.em.relion import relion_worker_scale

pytestmark = pytest.mark.unit


def test_per_half_update_from_half_score_result_updates_only_score_payload():
    class _Stats:
        max_posterior_per_image = np.array([0.25, 0.75], dtype=np.float64)
        rotation_posterior_sums = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    outs = score_outputs.PerHalfOutputs()
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
    outs = iteration_loop.PerHalfOutputs()
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


def test_relion_correction_range_formatter_accepts_empty_halves():
    assert relion_worker_scale._format_relion_correction_range(np.array([], dtype=np.float32)) == "empty"
    assert relion_worker_scale._format_relion_correction_range(np.array([0.5, 2.0], dtype=np.float32)) == "[0.5, 2]"


@pytest.mark.parametrize(
    "parent_masked,fine_masked,empty,expected",
    [
        (False, False, False, (4, 6, 3, 6)),
        (True, False, False, (3, 4, 3, 6)),
        (False, True, False, (4, 6, 2, 3)),
        (True, True, False, (3, 4, 2, 3)),
        (False, False, True, (0, 0, 0, 0)),
    ],
)
def test_k1_local_full_parent_diagnostic_counts_unmasked_parent_layout(parent_masked, fine_masked, empty, expected):
    # An empty explicit selection remains empty; repeated explicit IDs still
    # count as entries. None means the full masked or unmasked parent support.
    translations = np.zeros((3, 2), dtype=np.float32)
    counts = np.array([] if empty else [2, 0, 1], dtype=np.int32)
    offsets = np.array([0] if empty else [0, 2, 2, 3], dtype=np.int64)
    mask = np.array([[True, False, True], [False, True, False], [True, True, False]])
    parent_layout = SimpleNamespace(
        rotation_counts=counts,
        rotation_offsets=offsets,
        translation_grid=translations,
        sample_mask_flat=mask if parent_masked else None,
    )
    fine_layout = SimpleNamespace(
        rotation_counts=counts,
        rotation_offsets=offsets,
        translation_grid=translations,
        sample_mask_flat=mask if fine_masked else None,
    )
    selected = [] if empty else [None, np.array([], dtype=np.int64), np.array([0, 0, 1, 2])]
    records = []
    local_debug.log_local_adaptive_support(
        SimpleNamespace(info=lambda *args: records.append(args)),
        parent_layout,
        selected,
        translations,
        fine_layout,
    )
    assert records == [
        (
            "RELION local adaptive pass 2 mask: parent significant samples median=%d max=%d; "
            "fine valid candidates median=%d max=%d",
            *expected,
        )
    ]


@pytest.mark.parametrize(
    "masked,empty,expected",
    [
        (False, False, (3, 6)),
        (True, False, (2, 3)),
        (False, True, (0, 0)),
    ],
)
def test_local_denominator_diagnostic_counts_masked_and_empty_support(masked, empty, expected):
    layout = SimpleNamespace(
        rotation_counts=np.array([] if empty else [2, 0, 1], dtype=np.int32),
        rotation_offsets=np.array([0] if empty else [0, 2, 2, 3], dtype=np.int64),
        translation_grid=np.zeros((3, 2), dtype=np.float32),
        sample_mask_flat=(
            np.array([[True, False, True], [False, True, False], [True, True, False]]) if masked else None
        ),
    )
    records = []
    local_debug.log_local_denominator_support(
        SimpleNamespace(info=lambda *args: records.append(args)), layout, "full_parent", "TEST_ENV"
    )
    assert records == [
        (
            "RELION local adaptive pass 2 diagnostic: denominator support mode=%s "
            "fine valid candidates median=%d max=%d via %s",
            "full_parent",
            *expected,
            "TEST_ENV",
        )
    ]


def test_k1_local_records_coarse_parent_support_not_fine_reconstruction_count():
    counts = half_scoring._relion_coarse_significant_counts(
        [np.array([2, 8], dtype=np.int64), np.array([1, 3, 5, 7], dtype=np.int64)]
    )
    np.testing.assert_array_equal(counts, np.array([2, 4], dtype=np.int32))
    assert half_scoring._relion_coarse_significant_counts([np.array([2]), None]) is None


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
            relion_stats=_Stats(),
            noise_stats="noise",
        )
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
        noise_variance_k="noise_variance",
        previous_best_rotation_eulers_k=np.zeros((1, 3), dtype=np.float32),
        local_search_rotations=np.eye(3, dtype=np.float32)[None, :, :],
        local_search_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        current_translations=np.zeros((1, 2), dtype=np.float32),
        base_translations=np.zeros((1, 2), dtype=np.float32),
        trans_prior_center=np.zeros((1, 2), dtype=np.float32),
        trans_prior_center_for_engine=np.zeros((1, 2), dtype=np.float32),
        current_sigma_offset_angstrom=1.0,
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
        outputs=score_outputs.PerHalfOutputs(),
        local_profile_history=[],
    )

    assert captured["mstep_relion_x_half"] is True
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
        noise_variance_k="noise_variance",
        previous_best_rotation_eulers_k=np.zeros((2, 3), dtype=np.float32),
        local_search_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy(),
        local_search_order=1,
        sigma_rot=0.1,
        sigma_psi=0.1,
        current_translations=np.zeros((1, 2), dtype=np.float32),
        base_translations=np.zeros((1, 2), dtype=np.float32),
        trans_prior_center=np.zeros((2, 2), dtype=np.float32),
        trans_prior_center_for_engine=np.zeros((2, 2), dtype=np.float32),
        current_sigma_offset_angstrom=1.0,
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
        outputs=score_outputs.PerHalfOutputs(),
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
        np.testing.assert_array_equal(fine_call["normalization_log_evidence"], _Stats.log_evidence_per_image)
    assert parent_call["score_only"] is True
    assert "return_significant_counts" not in parent_call
    assert parent_call["apply_max_significants_to_support"] is True
    assert parent_call["max_significants"] == 23
    assert fine_call["score_only"] is False
    assert fine_call["reconstruct_significant_only"] is True
    assert fine_call["stats_use_reconstruction_probs"] is True
    assert "return_significant_counts" not in fine_call
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
        noise_variance_k="noise_variance",
        previous_best_rotation_eulers_k=np.zeros((1, 3), dtype=np.float32),
        local_search_rotations=np.eye(3, dtype=np.float32)[None, :, :],
        local_search_order=0,
        sigma_rot=0.1,
        sigma_psi=0.1,
        current_translations=np.zeros((1, 2), dtype=np.float32),
        base_translations=np.zeros((1, 2), dtype=np.float32),
        trans_prior_center=np.zeros((1, 2), dtype=np.float32),
        trans_prior_center_for_engine=np.zeros((1, 2), dtype=np.float32),
        current_sigma_offset_angstrom=1.0,
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
        outputs=score_outputs.PerHalfOutputs(),
        local_profile_history=[],
    )

    assert captured["mstep_relion_x_half"] is True
    assert result.mstep_full_half_axis == 0
    assert result.mstep_accumulator_shape == (19, 19, 19)


def test_native_final_perturbation_uses_active_local_order_but_preserves_global_order():
    local_state = SimpleNamespace(do_local_search=True, healpix_order=4)
    global_state = SimpleNamespace(do_local_search=False, healpix_order=4)

    assert _native_final_perturbation_healpix_order(local_state, 3) == 4
    assert _native_final_perturbation_healpix_order(global_state, 3) == 3


def test_local_adaptive_pass2_defaults_to_relion_pruned_parent(monkeypatch):
    monkeypatch.delenv(scoring_policy._LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV, raising=False)

    assert scoring_policy._local_adaptive_pass2_full_parent_enabled() is False

    monkeypatch.setenv(scoring_policy._LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV, "1")
    assert scoring_policy._local_adaptive_pass2_full_parent_enabled() is True

    monkeypatch.setenv(scoring_policy._LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV, "0")
    assert scoring_policy._local_adaptive_pass2_full_parent_enabled() is False
