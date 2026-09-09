from pathlib import Path

import numpy as np

from scripts.analyze_vdam_mstep_boundary import _STAGES
from scripts.analyze_vdam_mstep_repeatability import SCHEMA, compare


def _write_native(path: Path, values: np.ndarray, *, complex_values: bool) -> None:
    array = np.asarray(values, dtype=np.complex128 if complex_values else np.float64)
    with path.open("wb") as stream:
        np.asarray(array.shape, dtype=np.int64).tofile(stream)
        array.tofile(stream)


def test_compare_reports_native_and_recovar_repeatability(tmp_path):
    native_a = tmp_path / "native_a"
    native_b = tmp_path / "native_b"
    recovar_a = tmp_path / "recovar_a"
    recovar_b = tmp_path / "recovar_b"
    for directory in (native_a, native_b, recovar_a, recovar_b):
        directory.mkdir()

    for index, (_, native_name, recovar_name, complex_values) in enumerate(_STAGES):
        base = np.full((2, 2, 2), index + 1, dtype=np.float64)
        native_values = base.astype(np.complex128) * (1 + 2j) if complex_values else base
        _write_native(native_a / native_name, native_values, complex_values=complex_values)
        _write_native(native_b / native_name, native_values, complex_values=complex_values)
        np.save(recovar_a / recovar_name, native_values)
        np.save(recovar_b / recovar_name, native_values)

    report = compare(native_a, native_b, recovar_a, recovar_b)

    assert report["schema"] == SCHEMA
    assert report["native_repeat"]["raw_accumulator_data_half0"]["relative_l2"] == 0.0
    assert report["recovar_repeat"]["raw_accumulator_weight_half1"]["relative_l2"] == 0.0
    assert report["native_floor_ratios"]["raw_accumulator_data_half0"] == {
        "cross_arm_a_over_native_repeat": None,
        "cross_arm_b_over_native_repeat": None,
    }
    assert report["cross_arm_a"]["all_stages_bitwise_exact"] is True
    assert report["cross_arm_b"]["all_stages_bitwise_exact"] is True
