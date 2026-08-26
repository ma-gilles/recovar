from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_k1_live_controller_prefix as auditor


def _write_meta(directory: Path, index: int, *, current_size: int, healpix_order: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(
        directory / f"it{index:03d}_meta.npy",
        {
            "iteration": index,
            "current_size": current_size,
            "healpix_order": healpix_order,
            "local_search": False,
            "n_rotations": 768,
            "n_translations": 9,
        },
        allow_pickle=True,
    )


@pytest.mark.unit
def test_live_controller_prefix_passes_exact_sealed_prefix(tmp_path, monkeypatch):
    recovar_dir = tmp_path / "recovar"
    relion_dir = tmp_path / "relion"
    _write_meta(recovar_dir / "intermediates", 0, current_size=56, healpix_order=3)
    _write_meta(recovar_dir / "intermediates", 1, current_size=100, healpix_order=3)
    expected = {1: (56, 3), 2: (100, 3)}
    monkeypatch.setattr(
        auditor,
        "_read_relion_controller",
        lambda directory, iteration: expected[iteration],
    )

    output_json = tmp_path / "audit.json"
    output_markdown = tmp_path / "audit.md"
    status = auditor.main(
        [
            "--recovar-dir",
            str(recovar_dir),
            "--relion-dir",
            str(relion_dir),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ]
    )

    report = json.loads(output_json.read_text())
    assert status == 0
    assert report["status"] == "pass"
    assert report["completion_claim"] is False
    assert report["sealed_iteration_count"] == 2
    assert report["all_controller_topology_exact"] is True
    assert report["failures"] == []
    assert [row["relion_iteration"] for row in report["iterations"]] == [1, 2]
    assert all(row["current_size"]["exact_equal"] for row in report["iterations"])
    assert all(row["healpix_order"]["exact_equal"] for row in report["iterations"])
    assert "does not claim terminal completion" in output_markdown.read_text()


@pytest.mark.unit
def test_live_controller_prefix_fails_on_first_controller_mismatch(tmp_path, monkeypatch):
    recovar_dir = tmp_path / "recovar"
    _write_meta(recovar_dir / "intermediates", 0, current_size=56, healpix_order=3)
    monkeypatch.setattr(auditor, "_read_relion_controller", lambda directory, iteration: (58, 2))

    report = auditor.audit(recovar_dir, tmp_path / "relion")

    assert report["status"] == "fail"
    assert report["all_controller_topology_exact"] is False
    assert report["failures"] == ["it001 current_size mismatch", "it001 healpix_order mismatch"]
    assert report["earliest_failure"] == "it001 current_size mismatch"


@pytest.mark.unit
def test_live_controller_prefix_fails_closed_on_noncontiguous_metadata(tmp_path):
    recovar_dir = tmp_path / "recovar"
    _write_meta(recovar_dir / "intermediates", 1, current_size=56, healpix_order=3)

    with pytest.raises(auditor.AuditError, match="not contiguous"):
        auditor.audit(recovar_dir, tmp_path / "relion")
