import numpy as np
import pytest

from scripts import analyze_k4_preprocess_repeatability as analyzer


def _write_panel(root, *, score_delta=0.0, probability_delta=0.0, flip_class=False):
    root.mkdir()
    for target in (1, 2):
        for class_one_based in (1, 2):
            scores = np.array([[1.0, 0.5]], dtype=np.float64)
            if target == 2 and class_one_based == 2:
                scores[0, 0] += score_delta
            probability_mass = 0.8 if class_one_based == 2 else 0.2
            if flip_class and target == 2:
                probability_mass = 0.1 if class_one_based == 2 else 0.9
            probs = np.array([[probability_mass, 0.0]], dtype=np.float64)
            if target == 2 and class_one_based == 2:
                probs[0, 0] += probability_delta
            np.savez(
                root
                / f"pass2_orig{target:06d}_class{class_one_based:03d}_cs074.npz",
                scores_pre_prior=scores,
                candidate_mask=np.ones_like(scores, dtype=bool),
                probs=probs,
                reconstruction_probs=probs.copy(),
                translations=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            )


def _analyze(reference, repeat):
    return analyzer.analyze(
        reference_directory=reference,
        repeat_directory=repeat,
        expected_target_count=2,
        expected_class_count=2,
        expected_current_size=74,
    )


def test_probability_roundoff_is_measured_with_exact_score_topology_gate(tmp_path):
    reference = tmp_path / "reference"
    repeat = tmp_path / "repeat"
    _write_panel(reference)
    _write_panel(repeat, probability_delta=np.finfo(np.float64).eps)

    report = _analyze(reference, repeat)

    assert report["status"] == "complete"
    assert report["classification"] == "exact_score_topology_posterior_roundoff_only"
    assert report["strict_gates"]["score_and_topology_fields_exact"] is True
    assert report["strict_gates"]["class_predictions_exact"] is True
    assert report["field_stats"]["scores_pre_prior"]["exact"] is True
    assert report["field_stats"]["probs"]["nonzero_finite_delta_count"] == 1
    assert report["scorecard_change_admissible"] is False


def test_all_exact_arrays_are_classified_separately(tmp_path):
    reference = tmp_path / "reference"
    repeat = tmp_path / "repeat"
    _write_panel(reference)
    _write_panel(repeat)

    report = _analyze(reference, repeat)

    assert report["classification"] == "all_arrays_exact"
    assert report["scope"]["array_exact_file_count"] == 4


def test_score_change_is_rejected_without_tolerance(tmp_path):
    reference = tmp_path / "reference"
    repeat = tmp_path / "repeat"
    _write_panel(reference)
    _write_panel(repeat, score_delta=np.finfo(np.float64).eps)

    report = _analyze(reference, repeat)

    assert report["status"] == "rejected"
    assert report["classification"] == "score_topology_or_class_repeatability_failure"
    assert report["strict_gates"]["nonprobability_mismatch_fields"] == [
        "scores_pre_prior"
    ]


def test_class_prediction_change_is_rejected(tmp_path):
    reference = tmp_path / "reference"
    repeat = tmp_path / "repeat"
    _write_panel(reference)
    _write_panel(repeat, flip_class=True)

    report = _analyze(reference, repeat)

    assert report["status"] == "rejected"
    assert report["strict_gates"]["class_predictions_exact"] is False
    assert report["scope"]["class_prediction_exact_count"] == 1


def test_incomplete_panel_is_rejected(tmp_path):
    reference = tmp_path / "reference"
    repeat = tmp_path / "repeat"
    _write_panel(reference)
    _write_panel(repeat)
    next(repeat.glob("*.npz")).unlink()

    with pytest.raises(ValueError, match="exactly 4 NPZ files"):
        _analyze(reference, repeat)


@pytest.mark.parametrize(
    ("targets", "classes", "size", "message"),
    [
        (0, 2, 74, "target count"),
        (2, 0, 74, "class count"),
        (2, 2, 0, "current size"),
    ],
)
def test_expected_scope_must_be_positive(tmp_path, targets, classes, size, message):
    with pytest.raises(ValueError, match=message):
        analyzer.analyze(
            reference_directory=tmp_path / "reference",
            repeat_directory=tmp_path / "repeat",
            expected_target_count=targets,
            expected_class_count=classes,
            expected_current_size=size,
        )
