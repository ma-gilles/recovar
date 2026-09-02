from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import validate_relion_k4_ppref_capture as validator
from scripts.build_k1_ppref_capture_from_map import write_ppref_capture


def _write_panel(directory: Path, count: int = 4) -> list[Path]:
    paths = []
    for model in range(count):
        values = (
            np.arange(75, dtype=np.float32).reshape(5, 5, 3)
            + np.float32(model + 1)
            + 1j * np.arange(75, 150, dtype=np.float32).reshape(5, 5, 3)
        ).astype(np.complex64)
        path = directory / f"ppref_iter002_rank000_model{model:03d}.bin"
        write_ppref_capture(
            path,
            values,
            iteration=2,
            rank=0,
            model=model,
            current_size=56,
            r_max=28,
            padding_factor=2.0,
        )
        paths.append(path)
    return paths


def test_direct_entrypoint_exposes_fail_closed_arguments() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(validator.__file__).resolve()), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--expected-current-size" in result.stdout


def test_validates_complete_class_panel(tmp_path: Path) -> None:
    _write_panel(tmp_path)
    report = validator.validate_directory(
        tmp_path,
        expected_iteration=2,
        expected_rank=0,
        n_classes=4,
        expected_current_size=56,
    )
    assert report["status"] == "pass"
    assert report["class_count"] == 4
    assert report["shared"] == {
        "current_size": 56,
        "shape_zyx": [5, 5, 3],
        "origin_xyz": [0, -2, -2],
        "r_max": 28,
        "padding_factor": 2.0,
    }
    assert report["bitwise_identical_class_pairs"] == []
    assert [record["model"] for record in report["classes"]] == [0, 1, 2, 3]


def test_rejects_incomplete_class_panel(tmp_path: Path) -> None:
    _write_panel(tmp_path, count=3)
    with pytest.raises(ValueError, match="incomplete or has extras"):
        validator.validate_directory(tmp_path, expected_iteration=2)


def test_rejects_truncated_payload(tmp_path: Path) -> None:
    path = _write_panel(tmp_path)[0]
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        validator.load_ppref(path)


def test_rejects_nonfinite_payload(tmp_path: Path) -> None:
    path = _write_panel(tmp_path)[0]
    payload = bytearray(path.read_bytes())
    payload[-4:] = np.asarray([np.nan], dtype="<f4").tobytes()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="non-finite"):
        validator.load_ppref(path)


def test_rejects_wrong_expected_iteration(tmp_path: Path) -> None:
    _write_panel(tmp_path)
    with pytest.raises(ValueError, match="iteration differs"):
        validator.validate_directory(tmp_path, expected_iteration=3)
