import numpy as np
import pytest

from scripts import analyze_relion_k4_capture_repeatability as repeatability
from scripts import analyze_relion_k4_panel_threeway as threeway


def test_capture_repeatability_centers_common_score_offsets():
    lhs = repeatability._center(np.asarray([10.0, 12.0, 17.0], dtype=np.float32))
    rhs = repeatability._center(np.asarray([110.0, 112.0, 117.0], dtype=np.float32))

    report = repeatability._residual(lhs, rhs)

    assert report["candidate_count"] == 3
    assert report["exact_equal"] is True
    assert report["mismatch_count"] == 0
    assert report["residual_energy"] == 0.0


def test_capture_repeatability_rejects_shape_drift():
    with pytest.raises(ValueError, match="shape changed"):
        repeatability._residual(np.zeros(2), np.zeros(3))


def _aggregate(*, host_data, candidate_data, host_combined, candidate_combined):
    def metric(energy):
        return {"residual_energy": energy}

    return {
        "data": {
            "host_numpy": metric(host_data),
            "relion_cuda": metric(candidate_data),
        },
        "combined": {
            "host_numpy": metric(host_combined),
            "relion_cuda": metric(candidate_combined),
        },
    }


def _floor(data=2.0, combined=3.0):
    return {
        "data": {"residual_energy": data},
        "combined": {"residual_energy": combined},
    }


@pytest.mark.parametrize(
    ("aggregate", "expected"),
    [
        (
            _aggregate(
                host_data=20.0,
                candidate_data=10.0,
                host_combined=30.0,
                candidate_combined=15.0,
            ),
            "relion_cuda_preprocessing_reduces_residual_beyond_capture_repeatability_floor",
        ),
        (
            _aggregate(
                host_data=20.0,
                candidate_data=19.0,
                host_combined=30.0,
                candidate_combined=28.0,
            ),
            "relion_cuda_preprocessing_reduction_is_within_capture_repeatability_floor",
        ),
        (
            _aggregate(
                host_data=20.0,
                candidate_data=19.0,
                host_combined=30.0,
                candidate_combined=31.0,
            ),
            "relion_cuda_preprocessing_does_not_uniformly_reduce_relion_fine_score_residual",
        ),
    ],
)
def test_threeway_classification_is_calibrated_to_capture_floor(aggregate, expected):
    classification, comparison = threeway._classify_improvement(aggregate, _floor())

    assert classification == expected
    assert comparison["data"]["improvement_energy"] == (
        aggregate["data"]["host_numpy"]["residual_energy"]
        - aggregate["data"]["relion_cuda"]["residual_energy"]
    )


def _calibration_inputs():
    exact_fields = {field: True for field in threeway.ALL_PARTICLE_FIELDS}
    exact_fields["rlnMaxValueProbDistribution"] = False
    exact_fields["rlnNrOfSignificantSamples"] = False
    inertness = {
        "schema": threeway.CAPTURE_INERTNESS_SCHEMA,
        "status": "rejected",
        "threshold": 0.999999,
        "dispatch_exact": True,
        "control_perturbation": -0.12306,
        "capture_perturbation": -0.12306,
        "exact_particle_fields": exact_fields,
        "particle_fields": {
            "rlnMaxValueProbDistribution": {"mismatch_count": 12_434, "max_abs": 0.000183},
            "rlnNrOfSignificantSamples": {"mismatch_count": 3, "max_abs": 1},
        },
        "class_map_comparison": [
            {"capture_vs_control_fsc_auc": 0.99999999} for _ in range(4)
        ],
    }
    capture_repeatability = {
        "schema": threeway.CAPTURE_REPEATABILITY_SCHEMA,
        "status": "complete",
        "scope": {
            "target_count": 12,
            "physical_iteration": 10,
            "class_one_based": 2,
            "geometry_exact_all": True,
            "winners_exact_all": True,
        },
        "targets": [
            {
                "factor_rotations_exact": True,
                "factor_translations_exact": True,
                "candidate_topology_exact": {"flags": True, "rotation_id": True},
            }
            for _ in range(12)
        ],
    }
    control_repeatability = {
        "schema": threeway.CONTROL_REPEATABILITY_SCHEMA,
        "status": "rejected",
        "threshold": 0.999999,
        "dispatch_exact": True,
        "perturbation_a": -0.12306,
        "perturbation_b": -0.12306,
        "exact_particle_fields": exact_fields,
        "particle_fields": {
            "rlnMaxValueProbDistribution": {"mismatch_count": 12_485, "max_abs": 0.000174},
            "rlnNrOfSignificantSamples": {"mismatch_count": 5, "max_abs": 1},
        },
        "class_map_comparison": [{"repeat_fsc_auc": 0.99999999} for _ in range(4)],
    }
    screen = {
        "schema": threeway.SCREEN_SCHEMA,
        "status": "complete",
        "topology_exact_all": True,
        "scope": {
            "physical_iteration": 10,
            "current_size": 74,
            "target_count": 12,
            "classes": 4,
        },
    }
    return inertness, screen, capture_repeatability, control_repeatability


def test_threeway_accepts_repeatability_calibrated_non_scorecard_diagnostic():
    inertness, screen, capture_repeatability, control_repeatability = _calibration_inputs()

    report = threeway._validate_calibration_inputs(
        inertness=inertness,
        screen=screen,
        capture_repeatability=capture_repeatability,
        control_repeatability=control_repeatability,
    )

    assert report["classification"] == "repeatability_calibrated_non_scorecard_diagnostic"
    assert report["all_eight_particle_fields_exact"] is False
    assert report["pose_translation_class_fields_exact"] is True
    assert report["capture_pmax_mismatch_count"] == 12_434
    assert report["control_repeat_pmax_mismatch_count"] == 12_485


def test_threeway_rejects_pose_drift_even_when_map_fsc_passes():
    inertness, screen, capture_repeatability, control_repeatability = _calibration_inputs()
    inertness["exact_particle_fields"]["rlnAngleRot"] = False

    with pytest.raises(ValueError, match="pose, translation, or class"):
        threeway._validate_calibration_inputs(
            inertness=inertness,
            screen=screen,
            capture_repeatability=capture_repeatability,
            control_repeatability=control_repeatability,
        )


def test_threeway_rejects_incomplete_preprocess_screen():
    inertness, screen, capture_repeatability, control_repeatability = _calibration_inputs()
    screen["status"] = "running"

    with pytest.raises(ValueError, match="screen is incomplete"):
        threeway._validate_calibration_inputs(
            inertness=inertness,
            screen=screen,
            capture_repeatability=capture_repeatability,
            control_repeatability=control_repeatability,
        )


def test_threeway_rejects_capture_dispatch_drift():
    inertness, screen, capture_repeatability, control_repeatability = _calibration_inputs()
    inertness["dispatch_exact"] = False

    with pytest.raises(ValueError, match="dispatch or perturbation"):
        threeway._validate_calibration_inputs(
            inertness=inertness,
            screen=screen,
            capture_repeatability=capture_repeatability,
            control_repeatability=control_repeatability,
        )


def test_threeway_rejects_changed_fixed_fsc_threshold():
    inertness, screen, capture_repeatability, control_repeatability = _calibration_inputs()
    inertness["threshold"] = 0.99

    with pytest.raises(ValueError, match="threshold changed"):
        threeway._validate_calibration_inputs(
            inertness=inertness,
            screen=screen,
            capture_repeatability=capture_repeatability,
            control_repeatability=control_repeatability,
        )
