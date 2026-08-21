from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest

from scripts import audit_em_recovar_intermediate_capture_inertness as audit


def _write_map(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=False) as handle:
        handle.set_data(np.asarray(values, dtype=np.float32))


def _write_panel(
    root: Path,
    halves: tuple[np.ndarray, np.ndarray],
    *,
    iteration: int,
) -> None:
    for half, values in enumerate(halves, start=1):
        _write_map(
            root
            / "recovar"
            / "intermediates"
            / f"it{iteration:03d}_half{half}_reg.mrc",
            values,
        )


def test_identical_intermediate_panel_passes(tmp_path: Path) -> None:
    rng = np.random.default_rng(14)
    halves = (
        rng.normal(size=(8, 8, 8)).astype(np.float32),
        rng.normal(size=(8, 8, 8)).astype(np.float32),
    )
    control = tmp_path / "control"
    capture = tmp_path / "capture"
    _write_panel(control, halves, iteration=1)
    _write_panel(capture, halves, iteration=1)

    report = audit.build_report(
        control_root=control,
        capture_root=capture,
        recovar_iteration=1,
        fsc_auc_threshold=0.999999,
    )

    assert report["status"] == "pass"
    assert report["capture_inertness_qualified"]
    assert report["strict_gate"] == {
        "passed": 3,
        "evaluated": 3,
        "expected": 3,
    }
    assert set(report["comparisons"]) == {"half1", "half2", "merged"}
    assert len(report["artifact_sha256"]) == 4


def test_changed_intermediate_map_rejects_panel(tmp_path: Path) -> None:
    rng = np.random.default_rng(28)
    halves = (
        rng.normal(size=(8, 8, 8)).astype(np.float32),
        rng.normal(size=(8, 8, 8)).astype(np.float32),
    )
    control = tmp_path / "control"
    capture = tmp_path / "capture"
    _write_panel(control, halves, iteration=3)
    _write_panel(capture, (-halves[0], halves[1]), iteration=3)

    report = audit.build_report(
        control_root=control,
        capture_root=capture,
        recovar_iteration=3,
        fsc_auc_threshold=0.999999,
    )

    assert report["status"] == "rejected"
    assert not report["capture_inertness_qualified"]
    assert report["strict_gate"]["passed"] == 1
    assert not report["comparisons"]["half1"]["passed"]


def test_rejects_negative_iteration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        audit.build_report(
            control_root=tmp_path / "control",
            capture_root=tmp_path / "capture",
            recovar_iteration=-1,
            fsc_auc_threshold=0.999999,
        )
