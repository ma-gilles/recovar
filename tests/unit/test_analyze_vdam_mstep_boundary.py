import json
from pathlib import Path

import numpy as np

from scripts import analyze_vdam_mstep_boundary as analyzer


def _write_relion(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values)
    with path.open("wb") as stream:
        np.asarray(values.shape, dtype=np.int64).tofile(stream)
        if np.iscomplexobj(values):
            np.asarray(values, dtype=np.complex128).reshape(-1).view(np.float64).tofile(stream)
        else:
            np.asarray(values, dtype=np.float64).tofile(stream)


def test_vdam_mstep_analyzer_localizes_first_nonexact_stage(tmp_path, monkeypatch):
    native = tmp_path / "native"
    recovar = tmp_path / "recovar"
    native.mkdir()
    recovar.mkdir()
    base_real = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    base_complex = base_real.astype(np.complex128) * (1.0 + 0.25j)

    for index, (_name, native_name, recovar_name, complex_values) in enumerate(analyzer._STAGES):
        values = base_complex if complex_values else base_real
        _write_relion(native / native_name, values)
        candidate = values.copy()
        if index == 3:
            candidate.reshape(-1)[0] += 1.0
        np.save(recovar / recovar_name, candidate)

    report = analyzer.analyze(native, recovar)

    assert report["schema"] == analyzer.SCHEMA
    assert report["first_nonexact_stage"] == "raw_accumulator_data_half0"
    assert report["all_stages_bitwise_exact"] is False
    assert report["comparisons"]["input_gradient_half0"]["exact_count"] == 8
    assert report["comparisons"]["raw_accumulator_data_half0"]["exact_count"] == 7


def test_vdam_mstep_analyzer_selects_requested_iteration(tmp_path):
    native = tmp_path / "native"
    recovar = tmp_path / "recovar"
    native.mkdir()
    recovar.mkdir()
    for _name, native_name, recovar_name, complex_values in analyzer._stages_for_iteration(19):
        values = np.ones((1, 1, 1), dtype=np.complex128 if complex_values else np.float64)
        _write_relion(native / native_name, values)
        np.save(recovar / recovar_name, values)

    report = analyzer.analyze(native, recovar, iteration=19)

    assert report["iteration"] == 19
    assert report["all_stages_bitwise_exact"] is True


def test_vdam_mstep_cli_writes_json(tmp_path, monkeypatch):
    native = tmp_path / "native"
    recovar = tmp_path / "recovar"
    native.mkdir()
    recovar.mkdir()
    for _name, native_name, recovar_name, complex_values in analyzer._STAGES:
        values = np.ones((1, 1, 1), dtype=np.complex128 if complex_values else np.float64)
        _write_relion(native / native_name, values)
        np.save(recovar / recovar_name, values)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_vdam_mstep_boundary.py",
            "--native-directory",
            str(native),
            "--recovar-directory",
            str(recovar),
            "--output-json",
            str(output),
        ],
    )

    analyzer.main()

    assert json.loads(output.read_text())["all_stages_bitwise_exact"] is True
