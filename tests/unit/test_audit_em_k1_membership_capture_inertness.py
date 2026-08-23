from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np

from scripts import audit_em_k1_membership_capture_inertness as audit


def _write_map(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=False) as handle:
        handle.set_data(np.asarray(values, dtype=np.float32))


def _write_panel(root: Path, halves: tuple[np.ndarray, np.ndarray]) -> None:
    for half, values in enumerate(halves, start=1):
        _write_map(
            root / "relion" / f"run_it002_half{half}_class001.mrc",
            values,
        )
        _write_map(root / "recovar" / f"final_half{half}.mrc", values)


def test_identical_six_map_panel_passes(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
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
    assert report["capture_inertness_qualified"]
    assert report["strict_gate"]["comparison_count"] == 6
    assert set(report["comparisons"]) == {
        "relion_half1",
        "relion_half2",
        "relion_merged",
        "recovar_half1",
        "recovar_half2",
        "recovar_merged",
    }
    assert len(report["artifact_sha256"]) == 8


def test_changed_map_rejects_panel(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
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
    assert not report["capture_inertness_qualified"]
    assert not report["comparisons"]["relion_half1"]["passed"]
    assert not report["comparisons"]["recovar_half1"]["passed"]


def test_rejects_non_cubic_map(tmp_path: Path) -> None:
    path = tmp_path / "bad.mrc"
    _write_map(path, np.ones((4, 5, 4), dtype=np.float32))
    try:
        audit._load_map(path)
    except ValueError as error:
        assert "cubic" in str(error)
    else:
        raise AssertionError("non-cubic map was accepted")
