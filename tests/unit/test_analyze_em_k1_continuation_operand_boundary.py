from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_em_k1_continuation_operand_boundary as analyzer


@pytest.mark.unit
def test_array_metrics_are_scale_sensitive_and_do_not_use_correlation() -> None:
    source = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    target = np.asarray([1.0, 3.0, 2.0], dtype=np.float32)

    metrics = analyzer.array_metrics(source, target)

    assert metrics["bitwise_equal"] is False
    assert metrics["mismatch_elements"] == 2
    assert metrics["evaluated_elements"] == 3
    assert metrics["max_abs"] == 2.0
    assert metrics["relative_l2"] == np.sqrt(5.0) / np.sqrt(21.0)
    assert "correlation" not in metrics


@pytest.mark.unit
def test_array_metrics_treat_matching_nan_sentinels_as_equal() -> None:
    values = np.asarray([1.0, np.nan, 3.0], dtype=np.float32)

    metrics = analyzer.array_metrics(values, values.copy())

    assert metrics["bitwise_equal"] is True
    assert metrics["mismatch_elements"] == 0
    assert metrics["relative_l2"] == 0.0


@pytest.mark.unit
def test_read_star_float_requires_one_finite_value(tmp_path: Path) -> None:
    path = tmp_path / "sampling.star"
    path.write_text("data_sampling_general\n\n_rlnSamplingPerturbInstance  -0.04799\n_rlnSamplingPerturbFactor  0.5\n")

    assert analyzer.read_star_float(path, "_rlnSamplingPerturbInstance") == -0.04799

    path.write_text("_rlnSamplingPerturbInstance  nan\n")
    with pytest.raises(ValueError, match="not finite"):
        analyzer.read_star_float(path, "_rlnSamplingPerturbInstance")


def _summary(exact: int, expected: int = 14) -> dict[str, int]:
    return {
        "bitwise_equal_particles": exact,
        "evaluated_particles": expected,
    }


@pytest.mark.unit
def test_classifies_discarded_sampling_state_before_geometry() -> None:
    preprocess = {name: _summary(0) for name in analyzer.PREPROCESS_FIELDS}
    preprocess["raw_input_real"] = _summary(14)
    operands = {name: _summary(0) for name in analyzer.OPERAND_FIELDS}
    operands["rotation_keys"] = _summary(14)
    operands["local_rotation_indices"] = _summary(14)
    sampling = {
        "source_iteration1": -0.04799,
        "restart_output_iteration1": 0.460047,
        "fresh_iteration2": 0.409490,
        "restart_iteration2": -0.08248,
    }

    assert analyzer.classify_boundary(
        preprocess,
        operands,
        sampling,
        expected_particles=14,
    ) == ("serialized_sampling_perturbation_discarded_before_euler_and_translation_geometry")


@pytest.mark.unit
def test_classification_fails_closed_when_topology_identity_changes() -> None:
    preprocess = {name: _summary(0) for name in analyzer.PREPROCESS_FIELDS}
    preprocess["raw_input_real"] = _summary(14)
    operands = {name: _summary(0) for name in analyzer.OPERAND_FIELDS}
    operands["rotation_keys"] = _summary(13)
    operands["local_rotation_indices"] = _summary(14)
    sampling = {
        "source_iteration1": -0.04799,
        "restart_output_iteration1": 0.460047,
        "fresh_iteration2": 0.409490,
        "restart_iteration2": -0.08248,
    }

    assert (
        analyzer.classify_boundary(
            preprocess,
            operands,
            sampling,
            expected_particles=14,
        )
        == "continuation_operand_boundary_not_uniquely_classified"
    )


@pytest.mark.unit
def test_summarize_field_preserves_fixed_denominator() -> None:
    metrics = [
        analyzer.array_metrics(
            np.asarray([value], dtype=np.float32),
            np.asarray([value + (index == 1)], dtype=np.float32),
        )
        for index, value in enumerate((1.0, 2.0, 3.0))
    ]

    summary = analyzer.summarize_field(metrics)

    assert summary["evaluated_particles"] == 3
    assert summary["bitwise_equal_particles"] == 2
    assert summary["mismatch_elements"] == 1
