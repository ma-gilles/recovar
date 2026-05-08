import pytest

from scripts.run_ppca_refinement_regression_guard import (
    GuardThresholds,
    validate_benchmark_summary,
    validate_ppca_summary,
)


pytestmark = pytest.mark.unit


def _summary(
    *,
    slow=False,
    bad_pose=False,
    bad_mstep=False,
    skip_fraction=0.9,
    uses_window=False,
    image_shape=(64, 64),
):
    elapsed = [30.0, 12.0, 11.0]
    if slow:
        elapsed = [50.0, 25.0, 24.0]
    iterations = []
    for idx, elapsed_s in enumerate(elapsed):
        iterations.append(
            {
                "elapsed_s": elapsed_s,
                "diagnostics": {
                    "log_likelihood": float(idx),
                    "input_regularized_objective": float(idx),
                    "mstep_objective_solved_delta_per_image": -1.0 if bad_mstep else 1.0,
                    "pmax_mean": 1.0,
                    "nsig_mean": 1.0,
                    "sparse_pass2_skipped_fraction": skip_fraction,
                    "uses_fourier_window": uses_window,
                    "score_fourier_size": 100,
                    "full_half_fourier_size": 500,
                },
                "gt_pose_diagnostics": {
                    "rotation_exact_fraction": 0.5 if bad_pose else 1.0,
                    "translation_exact_fraction": 1.0,
                },
            }
        )
    return {"passed": True, "image_shape": list(image_shape), "iterations": iterations}


def test_ppca_regression_guard_accepts_saved_good_run_shape():
    failures = validate_ppca_summary(_summary(), GuardThresholds())
    assert failures == []


def test_ppca_regression_guard_fails_slow_or_bad_pose_runs():
    thresholds = GuardThresholds()
    failures = validate_ppca_summary(_summary(slow=True), thresholds)
    assert any("first iteration" in failure for failure in failures)
    assert any("worst steady iteration" in failure for failure in failures)

    failures = validate_ppca_summary(_summary(bad_pose=True), thresholds)
    assert any("final rotation exact fraction" in failure for failure in failures)

    failures = validate_ppca_summary(_summary(skip_fraction=0.1), thresholds)
    assert any("sparse skip fraction" in failure for failure in failures)

    failures = validate_ppca_summary(_summary(bad_mstep=True), thresholds)
    assert any("M-step solved objective delta" in failure for failure in failures)


def test_ppca_window_guard_checks_window_and_shape_without_monotone_objective():
    nonmonotone = _summary(uses_window=True, image_shape=(128, 128))
    nonmonotone["iterations"][1]["diagnostics"]["log_likelihood"] = -10.0
    nonmonotone["iterations"][1]["diagnostics"]["input_regularized_objective"] = -10.0
    assert (
        validate_ppca_summary(
            nonmonotone,
            GuardThresholds(),
            require_monotone_objective=False,
            require_fourier_window=True,
            expected_image_shape=(128, 128),
        )
        == []
    )

    no_window = _summary(uses_window=False, image_shape=(128, 128))
    failures = validate_ppca_summary(
        no_window,
        GuardThresholds(),
        require_monotone_objective=False,
        require_fourier_window=True,
        expected_image_shape=(128, 128),
    )
    assert any("did not use the Fourier window" in failure for failure in failures)

    wrong_shape = _summary(uses_window=True, image_shape=(64, 64))
    failures = validate_ppca_summary(
        wrong_shape,
        GuardThresholds(),
        require_monotone_objective=False,
        require_fourier_window=True,
        expected_image_shape=(128, 128),
    )
    assert any("image_shape" in failure for failure in failures)


def test_benchmark_guard_checks_ppca_time_and_adjusted_ratio():
    good = {
        "passed": True,
        "timing": {
            "ppca": {"median_s": 12.0},
            "ppca_over_kclass_q2_half_adjusted": 0.1,
        },
    }
    assert validate_benchmark_summary(good, GuardThresholds()) == []

    bad = {
        "passed": True,
        "timing": {
            "ppca": {"median_s": 100.0},
            "ppca_over_kclass_q2_half_adjusted": 2.0,
        },
    }
    failures = validate_benchmark_summary(bad, GuardThresholds())
    assert any("PPCA benchmark median" in failure for failure in failures)
    assert any("adjusted PPCA/K-class ratio" in failure for failure in failures)
