import numpy as np
import pytest

from scripts import analyze_relion_k4_panel_posterior_decomposition as analyzer

pytestmark = pytest.mark.unit


def test_counterfactual_summary_closes_exp50_frame():
    relion_raw = np.asarray([8.0, 2.0], dtype=np.float64)
    relion_normalizer = 20.0
    candidate_probability = np.asarray([0.3, 0.2], dtype=np.float64)
    candidate_pmax = 0.5

    report, arrays = analyzer._counterfactual_summary(
        relion_raw,
        relion_normalizer,
        candidate_probability,
        candidate_pmax,
    )

    assert np.array_equal(arrays["production"], candidate_probability)
    assert report["inferred_exp50_frame_reconstruction_max_abs"] == 0
    assert report["class2_probability_mass"] == 0.5
    assert report["production_posterior_residual"]["residual_energy"] > 0


def test_counterfactual_relion_numerator_leaves_only_normalizer_error():
    relion_raw = np.asarray([6.0, 3.0], dtype=np.float64)
    relion_normalizer = 12.0
    candidate_pmax = 0.25
    candidate_normalizer = analyzer.EXP50_F32 / candidate_pmax
    candidate_probability = relion_raw / candidate_normalizer

    report, _arrays = analyzer._counterfactual_summary(
        relion_raw,
        relion_normalizer,
        candidate_probability,
        candidate_pmax,
    )

    assert report["relion_numerator_counterfactual_residual"]["residual_energy"] == report[
        "production_posterior_residual"
    ]["residual_energy"]
    assert report["relion_normalizer_counterfactual_residual"][
        "residual_energy"
    ] <= np.finfo(np.float64).eps**2
    assert report["replace_normalizer_residual_energy_removed_fraction"] == 1


def test_counterfactual_relion_normalizer_leaves_only_numerator_error():
    relion_raw = np.asarray([6.0, 3.0], dtype=np.float64)
    relion_normalizer = analyzer.EXP50_F32 / 0.5
    candidate_probability = np.asarray([0.4, 0.1], dtype=np.float64)

    report, _arrays = analyzer._counterfactual_summary(
        relion_raw,
        relion_normalizer,
        candidate_probability,
        0.5,
    )

    assert report["relion_numerator_counterfactual_residual"]["residual_energy"] == 0
    assert report["relion_normalizer_counterfactual_residual"]["residual_energy"] == report[
        "production_posterior_residual"
    ]["residual_energy"]
    assert report["replace_numerator_residual_energy_removed_fraction"] == 1


def test_counterfactual_rejects_invalid_probability_operands():
    with pytest.raises(ValueError, match="RECOVAR posterior operands"):
        analyzer._counterfactual_summary(
            np.asarray([1.0]),
            2.0,
            np.asarray([-0.1]),
            0.5,
        )


def _synthetic_row(
    *,
    capture_a,
    capture_b,
    host,
    relion_cuda,
    host_numerator=None,
    host_normalizer=None,
    relion_cuda_numerator=None,
    relion_cuda_normalizer=None,
):
    capture_a = np.asarray(capture_a, dtype=np.float64)
    capture_b = np.asarray(capture_b, dtype=np.float64)
    host = np.asarray(host, dtype=np.float64)
    relion_cuda = np.asarray(relion_cuda, dtype=np.float64)
    return {
        "_arrays": {
            "capture_a": capture_a,
            "capture_b": capture_b,
            "host_numpy": {
                "relion_probability": capture_b,
                "production": host,
                "relion_numerator": np.asarray(
                    host if host_numerator is None else host_numerator,
                    dtype=np.float64,
                ),
                "relion_normalizer": np.asarray(
                    capture_b if host_normalizer is None else host_normalizer,
                    dtype=np.float64,
                ),
                "weight_normalizer_relative_error": 0.1,
            },
            "relion_cuda": {
                "relion_probability": capture_b,
                "production": relion_cuda,
                "relion_numerator": np.asarray(
                    relion_cuda if relion_cuda_numerator is None else relion_cuda_numerator,
                    dtype=np.float64,
                ),
                "relion_normalizer": np.asarray(
                    capture_b if relion_cuda_normalizer is None else relion_cuda_normalizer,
                    dtype=np.float64,
                ),
                "weight_normalizer_relative_error": 0.05,
            },
        }
    }


def test_cohort_summary_requires_improvement_beyond_floor():
    rows = [
        _synthetic_row(
            capture_a=[0.0, 0.0],
            capture_b=[0.1, 0.0],
            host=[0.4, 0.0],
            relion_cuda=[0.1, 0.0],
        )
    ]

    report = analyzer._cohort_summary(rows)

    assert report["classification"] == (
        "relion_cuda_reduces_posterior_residual_beyond_capture_repeatability_floor"
    )
    assert report["relion_cuda_improvement"]["improvement_exceeds_capture_repeatability_floor"]


def test_cohort_summary_labels_within_floor_reduction():
    rows = [
        _synthetic_row(
            capture_a=[0.0],
            capture_b=[0.2],
            host=[0.4],
            relion_cuda=[0.3],
        )
    ]

    report = analyzer._cohort_summary(rows)

    assert report["classification"] == (
        "relion_cuda_posterior_reduction_is_within_capture_repeatability_floor"
    )
    assert report["relion_cuda_improvement"]["improvement_positive"]
    assert not report["relion_cuda_improvement"][
        "improvement_exceeds_capture_repeatability_floor"
    ]


def test_cohort_summary_labels_non_reduction_and_counteraction():
    rows = [
        _synthetic_row(
            capture_a=[0.0],
            capture_b=[0.0],
            host=[0.1],
            relion_cuda=[0.2],
            relion_cuda_numerator=[0.3],
            relion_cuda_normalizer=[0.1],
        )
    ]

    report = analyzer._cohort_summary(rows)

    assert report["classification"] == "relion_cuda_does_not_reduce_posterior_residual"
    assert report["backends"]["relion_cuda"]["component_classification"] == (
        "numerator_normalizer_components_counteract"
    )


def test_residual_energy_uses_order_stable_fsum(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("BLAS-backed vdot must not be used")

    monkeypatch.setattr(np, "vdot", fail)
    metric = analyzer._residual(
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1e-8, 1.0, -1.0]),
    )

    assert metric["residual_energy"] == pytest.approx(2.0)
