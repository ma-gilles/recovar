import numpy as np
import pytest

from scripts import analyze_relion_k4_panel_numerator_boundary as analyzer

pytestmark = pytest.mark.unit


def test_expf_replay_applies_relion_underflow_predicate():
    shifted = np.asarray([-89.0, -88.0, 0.0], dtype=np.float32)

    replay = analyzer._expf_replay(shifted)

    assert replay.dtype == np.float32
    assert replay[0] == 0
    assert replay[1] > 0
    assert replay[2] == 1


def test_numerator_summary_closes_matching_score_frame():
    shifted = np.asarray([50.0, 49.0], dtype=np.float32)
    raw = analyzer._expf_replay(shifted).astype(np.float64)
    candidate_pmax = 0.5
    probability = raw / (analyzer.EXP50_F32 / candidate_pmax)

    report, arrays = analyzer._numerator_summary(
        relion_raw_weight=raw,
        relion_shifted_log_weight=shifted,
        candidate_probability=probability,
        candidate_pmax=candidate_pmax,
        candidate_shifted_log_weight=shifted,
    )

    assert report["production_raw_numerator_residual"]["residual_energy"] == 0
    assert report["relion_f32_score_replay_roundtrip_residual"]["residual_energy"] == 0
    np.testing.assert_array_equal(arrays["production_raw"], raw)


def test_numerator_summary_localizes_residual_before_exponentiation():
    relion_shifted = np.asarray([2.0, 1.0], dtype=np.float32)
    candidate_shifted = np.asarray([2.2, 1.2], dtype=np.float32)
    relion_raw = analyzer._expf_replay(relion_shifted).astype(np.float64)
    candidate_raw = analyzer._expf_replay(candidate_shifted).astype(np.float64)
    candidate_pmax = 0.25
    candidate_probability = candidate_raw / (
        analyzer.EXP50_F32 / candidate_pmax
    )

    report, _arrays = analyzer._numerator_summary(
        relion_raw_weight=relion_raw,
        relion_shifted_log_weight=relion_shifted,
        candidate_probability=candidate_probability,
        candidate_pmax=candidate_pmax,
        candidate_shifted_log_weight=candidate_shifted,
    )

    assert report["production_raw_numerator_residual"]["residual_energy"] > 0
    assert report["replace_score_with_relion_score_residual_energy_removed_fraction"] == 1
    assert report["replace_score_removes_at_least_99_percent"]
    assert (
        report["candidate_posterior_inferred_vs_f32_score_replay_residual"][
            "residual_energy"
        ]
        <= np.finfo(np.float64).eps**2
    )


def test_numerator_summary_rejects_invalid_operands():
    with pytest.raises(ValueError, match="numerator operands"):
        analyzer._numerator_summary(
            relion_raw_weight=np.asarray([1.0]),
            relion_shifted_log_weight=np.asarray([0.0]),
            candidate_probability=np.asarray([-0.1]),
            candidate_pmax=0.5,
            candidate_shifted_log_weight=np.asarray([0.0]),
        )


def _component_inputs(*, data_delta=0.0, orientation_delta=0.0, translation_delta=0.0):
    relion_data = np.asarray([0.0, 1.0, 2.0])
    relion_orientation = np.asarray([0.0, 0.5, 1.0])
    relion_translation = np.asarray([0.0, -0.25, -0.5])
    data_pattern = np.asarray([-1.0, 0.0, 1.0]) * data_delta
    orientation_pattern = np.asarray([1.0, -1.0, 0.0]) * orientation_delta
    translation_pattern = np.asarray([0.0, 1.0, -1.0]) * translation_delta
    candidate_data = relion_data + data_pattern
    candidate_orientation = relion_orientation + orientation_pattern
    candidate_translation = relion_translation + translation_pattern
    return {
        "relion_data_score": relion_data,
        "relion_orientation_prior": relion_orientation,
        "relion_translation_prior": relion_translation,
        "relion_combined_score": (
            relion_data + relion_orientation + relion_translation
        ),
        "candidate_data_score": candidate_data,
        "candidate_orientation_prior": candidate_orientation,
        "candidate_translation_prior": candidate_translation,
        "candidate_combined_score": (
            candidate_data + candidate_orientation + candidate_translation
        ),
    }


def test_component_summary_identifies_data_score():
    report, arrays = analyzer._component_summary(
        **_component_inputs(data_delta=0.5)
    )

    assert report["strongest_single_component"] == "data_score"
    assert (
        report["component_substitutions"]["data_score"][
            "residual_energy_removed_fraction"
        ]
        == 1
    )
    assert report["component_sum_closure_residual"]["residual_energy"] == pytest.approx(
        0,
        abs=np.finfo(np.float64).eps**2,
    )
    assert analyzer._energy(arrays["data_score"]) > 0


def test_component_summary_can_identify_prior_component():
    report, _arrays = analyzer._component_summary(
        **_component_inputs(orientation_delta=0.5)
    )

    assert report["strongest_single_component"] == "orientation_prior"
    assert (
        report["component_substitutions"]["orientation_prior"][
            "residual_energy_removed_fraction"
        ]
        == 1
    )


def _cohort_row(*, backend, shifted_delta, data_delta):
    relion_shifted = np.asarray([0.0, 1.0])
    candidate_shifted = relion_shifted + shifted_delta
    relion_raw = analyzer._expf_replay(relion_shifted).astype(np.float64)
    candidate_raw = analyzer._expf_replay(candidate_shifted).astype(np.float64)
    component_report, component_arrays = analyzer._component_summary(
        **_component_inputs(data_delta=data_delta)
    )
    del component_report
    arrays = {
        "production_raw": candidate_raw,
        "candidate_score_replay_raw": candidate_raw,
        "relion_raw": relion_raw,
        "relion_score_replay_raw": relion_raw,
        "candidate_shifted": candidate_shifted,
        "relion_shifted": relion_shifted,
        **component_arrays,
    }
    return {
        "active_candidate_count": 2,
        "_arrays": {
            backend: arrays,
            "host_numpy" if backend == "relion_cuda" else "relion_cuda": arrays,
        },
    }


def test_cohort_summary_classifies_upstream_data_score_boundary():
    report = analyzer._cohort_summary(
        [_cohort_row(backend="host_numpy", shifted_delta=0.2, data_delta=0.5)]
    )

    assert report["classification"] == (
        "numerator_residual_localized_upstream_of_exponentiation_to_data_score"
    )
    assert all(
        value["replace_score_removes_at_least_99_percent"]
        and value["strongest_single_component"] == "data_score"
        for value in report["backends"].values()
    )


def test_residual_paths_do_not_use_blas_vdot(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("BLAS-backed vdot must not be used")

    monkeypatch.setattr(np, "vdot", fail)
    report, _arrays = analyzer._component_summary(
        **_component_inputs(data_delta=0.5)
    )

    assert report["centered_combined_score_residual"]["residual_energy"] > 0
