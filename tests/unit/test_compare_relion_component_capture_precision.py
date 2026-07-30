from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import compare_relion_component_capture_precision as precision
from scripts.compare_relion_component_capture_precision import (
    build_report,
    compare_artifacts,
)
from scripts.validate_relion_coarse_pass1_components import (
    CoarsePass1Components,
)


def _artifact(tmp_path: Path, name: str, perturbation: float) -> CoarsePass1Components:
    path = tmp_path / name
    path.write_bytes(name.encode())
    reference = np.repeat(
        np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        3,
        axis=1,
    )
    cross = np.arange(12, dtype=np.float32).reshape(4, 3) / 16
    raw = reference + cross + np.float32(9.0)
    raw[0, 1] += np.float32(perturbation)
    header = [0] * 40
    header[6] = 4
    header[7] = 101
    return CoarsePass1Components(
        path=path,
        sha256=name,
        header=tuple(header),
        raw_diff2=raw,
        weights=np.ones_like(raw),
        reference_norms=reference,
        cross_terms=cross,
        significant_mask=np.ones_like(raw, dtype=bool),
        translations=np.zeros((3, 2), dtype=np.float32),
    )


def _validation(replay_passed: int) -> dict[str, object]:
    return {
        "fixed_metric": {
            "replay_p95_passed": replay_passed,
            "reference_translation_invariance_passed": 1,
        },
        "fixed_gates": {
            "centered_replay_p95_abs_max": 5.0e-5,
            "centered_replay_max_abs_max": 5.0e-4,
            "reference_norm_translation_spread_max": 1.0e-6,
        },
    }


def test_compares_identity_matched_capture_pair(tmp_path: Path) -> None:
    report = compare_artifacts(
        _artifact(tmp_path, "baseline.bin", 0.0),
        _artifact(tmp_path, "fp64.bin", 0.0001),
    )
    assert report["active_candidate_count"] == 12
    assert report["raw_score_bitwise_equal_fraction"] < 1.0
    assert report["raw_score_delta"]["centered_max_abs"] > 0.0
    assert report["baseline_replay"]["centered_replay_max_abs"] == 0.0
    assert report["fp64_replay"]["centered_replay_max_abs"] > 0.0


def test_report_preserves_rejection_and_denominator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    baseline = _artifact(tmp_path, "baseline.bin", 0.0)
    fp64 = _artifact(tmp_path, "fp64.bin", 0.01)

    def fake_validate(directory, *, expected_particles):
        assert expected_particles == 1
        if Path(directory).name == "baseline":
            return (baseline,), _validation(1)
        return (fp64,), _validation(0)

    monkeypatch.setattr(precision, "validate_directory", fake_validate)
    report = build_report(
        tmp_path / "baseline",
        tmp_path / "fp64",
        expected_particles=1,
    )
    assert not report["classification_ready"]
    assert report["fixed_metric"]["evaluated_particles"] == 1
    assert report["fixed_metric"]["baseline_replay_p95_passed"] == 1
    assert report["fixed_metric"]["fp64_replay_p95_passed"] == 0
    assert "does_not_replay" in report["classification"]
