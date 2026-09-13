import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_vdam_mstep_boundary import _stages_for_iteration, analyze
from scripts.analyze_vdam_mstep_repeat_panel import analyze_repeat_panel


def _write_native(path: Path, values: np.ndarray, *, complex_values: bool) -> None:
    values = np.asarray(values, dtype=np.complex128 if complex_values else np.float64)
    with path.open("wb") as stream:
        np.asarray(values.shape, dtype=np.int64).tofile(stream)
        if complex_values:
            values.reshape(-1).view(np.float64).tofile(stream)
        else:
            values.reshape(-1).tofile(stream)


def _make_arm(root: Path, *, native_delta: float, recovar_delta: float) -> None:
    native_dir = root / "native_mstep"
    recovar_dir = root / "recovar_mstep"
    analysis_dir = root / "analysis"
    native_dir.mkdir(parents=True)
    recovar_dir.mkdir()
    analysis_dir.mkdir()
    for _, native_name, recovar_name, complex_values in _stages_for_iteration(1):
        native = np.full((1, 1, 2), 2.0 + native_delta)
        recovar = np.full((1, 1, 2), 2.0 + recovar_delta)
        if complex_values:
            native = native.astype(np.complex128) * (1.0 + 0.5j)
            recovar = recovar.astype(np.complex128) * (1.0 + 0.5j)
        _write_native(native_dir / native_name, native, complex_values=complex_values)
        np.save(recovar_dir / recovar_name, recovar)
    report = analyze(native_dir, recovar_dir, iteration=1)
    (analysis_dir / "mstep_boundary.json").write_text(json.dumps(report))


def test_analyze_vdam_mstep_repeat_panel_reports_native_floor_ratios(tmp_path):
    arm_a = tmp_path / "a"
    arm_b = tmp_path / "b"
    _make_arm(arm_a, native_delta=0.0, recovar_delta=0.2)
    _make_arm(arm_b, native_delta=0.1, recovar_delta=0.4)

    report = analyze_repeat_panel(arm_a, arm_b)

    assert report["status"] == "complete"
    stage = "raw_accumulator_weight_half0"
    assert report["native_repeat"][stage]["relative_l2"] == pytest.approx(0.05)
    assert report["recovar_repeat"][stage]["relative_l2"] == pytest.approx(0.2 / 2.2)
    assert report["native_floor_ratios"][stage][
        "cross_arm_a_over_native_repeat"
    ] == pytest.approx(2.0)
    assert report["native_floor_ratios"][stage][
        "cross_arm_b_over_native_repeat"
    ] == pytest.approx(20.0 / 7.0)


def test_analyze_vdam_mstep_repeat_panel_rejects_unmatched_cross_report(tmp_path):
    arm_a = tmp_path / "a"
    arm_b = tmp_path / "b"
    _make_arm(arm_a, native_delta=0.0, recovar_delta=0.2)
    _make_arm(arm_b, native_delta=0.1, recovar_delta=0.4)
    path = arm_b / "analysis" / "mstep_boundary.json"
    report = json.loads(path.read_text())
    report["iteration"] = 2
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="expected iteration=1"):
        analyze_repeat_panel(arm_a, arm_b)
