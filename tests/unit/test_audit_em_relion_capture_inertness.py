from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np

from scripts import audit_em_relion_capture_inertness as audit


def _write_panel(root: Path, halves: tuple[np.ndarray, np.ndarray]) -> None:
    for half, values in enumerate(halves, start=1):
        path = root / "relion" / f"run_it002_half{half}_class001.mrc"
        path.parent.mkdir(parents=True, exist_ok=True)
        with mrcfile.new(path, overwrite=False) as handle:
            handle.set_data(np.asarray(values, dtype=np.float32))


def test_identical_relion_panel_passes_three_of_three(tmp_path: Path) -> None:
    rng = np.random.default_rng(23)
    halves = (
        rng.normal(size=(8, 8, 8)).astype(np.float32),
        rng.normal(size=(8, 8, 8)).astype(np.float32),
    )
    control = tmp_path / "control"
    capture = tmp_path / "capture"
    _write_panel(control, halves)
    _write_panel(capture, halves)
    report = audit.build_report(
        control_root=control,
        capture_root=capture,
        relion_iteration=2,
        fsc_auc_threshold=0.999999,
    )
    assert report["status"] == "pass"
    assert report["strict_gate"] == {"passed": 3, "evaluated": 3, "expected": 3}


def test_changed_half_rejects_panel(tmp_path: Path) -> None:
    rng = np.random.default_rng(29)
    halves = (
        rng.normal(size=(8, 8, 8)).astype(np.float32),
        rng.normal(size=(8, 8, 8)).astype(np.float32),
    )
    control = tmp_path / "control"
    capture = tmp_path / "capture"
    _write_panel(control, halves)
    _write_panel(capture, (-halves[0], halves[1]))
    report = audit.build_report(
        control_root=control,
        capture_root=capture,
        relion_iteration=2,
        fsc_auc_threshold=0.999999,
    )
    assert report["status"] == "rejected"
    assert report["strict_gate"]["passed"] < 3
