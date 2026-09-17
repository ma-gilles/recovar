from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_vdam_map_relative_l2_envelope as analysis_module
from scripts.analyze_vdam_map_relative_l2_envelope import (
    MapRelativeL2EnvelopeError,
    analyze_map_relative_l2_envelope,
    symmetric_relative_l2,
)


def _write_trajectory(root: Path, *, iterations: tuple[int, ...] = (0, 1)) -> None:
    root.mkdir(parents=True)
    (root / "trajectory_audit.json").write_text(
        json.dumps(
            {
                "schema": analysis_module.TRAJECTORY_SCHEMA,
                "suite_id": "suite",
                "case_id": "case",
                "artifact_topology_exact": True,
                "checkpoints": [{"iteration": iteration} for iteration in iterations],
            }
        )
    )


def test_symmetric_relative_l2_is_direct_and_scale_sensitive():
    reference = np.asarray([1.0, 2.0, 3.0])
    scaled = 2.0 * reference

    assert symmetric_relative_l2(reference, reference) == 0.0
    assert symmetric_relative_l2(reference, scaled) == pytest.approx(0.5)
    assert symmetric_relative_l2(scaled, reference) == pytest.approx(0.5)
    assert symmetric_relative_l2(np.zeros(3), np.zeros(3)) == 0.0


def test_symmetric_relative_l2_rejects_invalid_maps():
    with pytest.raises(MapRelativeL2EnvelopeError, match="shapes differ"):
        symmetric_relative_l2(np.zeros(2), np.zeros(3))
    with pytest.raises(MapRelativeL2EnvelopeError, match="finite"):
        symmetric_relative_l2(np.asarray([np.nan]), np.zeros(1))


def test_relative_l2_envelope_reports_first_candidate_escape(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    native_a = tmp_path / "native-a"
    native_b = tmp_path / "native-b"
    for root in (candidate, native_a, native_b):
        _write_trajectory(root)

    def fake_load(path: Path) -> np.ndarray:
        iteration = int(path.name.split("_it", 1)[1].split("_", 1)[0])
        if iteration == 0:
            return np.zeros(4)
        if path.is_relative_to(candidate):
            return np.full(4, 2.0)
        if path.is_relative_to(native_a):
            return np.full(4, 1.0)
        return np.full(4, 1.1)

    monkeypatch.setattr(analysis_module, "_load_relion_volume", fake_load)
    report = analyze_map_relative_l2_envelope(
        candidate_root=candidate,
        native_roots=[native_a, native_b],
    )

    assert report["first_iteration_outside_native_repeat_envelope"] == 1
    assert report["outside_native_repeat_envelope_count"] == 1
    assert report["checkpoints"][0][
        "candidate_within_native_repeat_relative_l2_envelope"
    ] is True
    escaped = report["checkpoints"][1]
    assert escaped["candidate_best_native_relative_l2"] == pytest.approx(0.45)
    assert escaped["native_repeat_max_relative_l2"] == pytest.approx(1.0 / 11.0)
    assert escaped["candidate_best_over_native_repeat_max_relative_l2"] == pytest.approx(
        4.95
    )
    assert escaped["candidate_within_native_repeat_relative_l2_envelope"] is False


def test_relative_l2_envelope_rejects_mixed_checkpoint_topology(tmp_path):
    candidate = tmp_path / "candidate"
    native_a = tmp_path / "native-a"
    native_b = tmp_path / "native-b"
    _write_trajectory(candidate)
    _write_trajectory(native_a)
    _write_trajectory(native_b, iterations=(0, 2))

    with pytest.raises(MapRelativeL2EnvelopeError, match="checkpoint topology differs"):
        analyze_map_relative_l2_envelope(
            candidate_root=candidate,
            native_roots=[native_a, native_b],
        )


def test_relative_l2_envelope_requires_two_native_repeats(tmp_path):
    candidate = tmp_path / "candidate"
    native = tmp_path / "native"
    _write_trajectory(candidate)
    _write_trajectory(native)

    with pytest.raises(MapRelativeL2EnvelopeError, match="at least two"):
        analyze_map_relative_l2_envelope(candidate_root=candidate, native_roots=[native])
