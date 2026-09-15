from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_vdam_bpref_replay_repeat_panel import (
    audit_replay_repeat_panel,
)
from scripts.audit_vdam_repeat_panel import RepeatPanelError

SCIENCE_HEAD = "a" * 40
GPU_UUID = "GPU-test"
CUDA_HASH = "b" * 64


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _make_panel(root: Path) -> None:
    _write_json(
        root / "submission_provenance.json",
        {
            "schema": "recovar.vdam_replay_repeat_panel_submission.v2",
            "science_head": SCIENCE_HEAD,
            "physical_gpu_uuid": GPU_UUID,
            "cuda_library_sha256": CUDA_HASH,
        },
    )
    rows = (
        ("arm-01-private", False, 1.0),
        ("arm-02-private", False, 2.0),
        ("arm-03-shared", True, 1.0),
        ("arm-04-shared", True, 4.0),
    )
    for index, (name, shared, first_value) in enumerate(rows, start=1):
        arm = root / name
        _write_json(
            arm / "provenance.json",
            {
                "science_head": SCIENCE_HEAD,
                "gpu_uuid": GPU_UUID,
                "job_id": str(index),
                "shared_accumulators": shared,
                "callbacks_merged_in_native_launch_order": True,
            },
        )
        _write_json(
            arm / "replay" / "worker_private_report.json",
            {
                "schema": "recovar.vdam_worker_private_host_replay.v6",
                "status": "complete",
                "hypothesis": name,
                "worker_count": 8,
                "callback_count": 1,
                "callbacks_merged_in_native_launch_order": True,
                "worker_private_accumulators": not shared,
                "native_physical_grid_replayed": True,
                "native_worker_owners_replayed": True,
                "native_euler_panels_replayed": True,
            },
        )
        value = np.asarray([first_value, 1.0], dtype=np.float32)
        np.savez(
            arm / "replay" / "worker_private_accumulators.npz",
            reduced_real=value,
            reduced_imag=value,
            reduced_weight=value,
        )


@pytest.mark.unit
def test_audits_private_and_shared_repeat_diameters(tmp_path: Path) -> None:
    _make_panel(tmp_path)

    report = audit_replay_repeat_panel(
        tmp_path, repeat_count=2, expected_gpu_uuid=GPU_UUID
    )

    assert report["result"] == "complete"
    assert report["scoring"] is False
    assert report["panels"]["private"]["repeat_count"] == 2
    private = report["panels"]["private"]["metrics"]["reduced_real"]
    shared = report["panels"]["shared"]["metrics"]["reduced_real"]
    np.testing.assert_allclose(
        private["maximum_pairwise_relative_l2"], 1.0 / np.sqrt(2.0)
    )
    np.testing.assert_allclose(
        shared["maximum_pairwise_relative_l2"], 3.0 / np.sqrt(2.0)
    )
    np.testing.assert_allclose(
        report["shared_over_private_maximum_diameter_ratio"]["reduced_real"], 3.0
    )


@pytest.mark.unit
def test_rejects_mixed_physical_gpu(tmp_path: Path) -> None:
    _make_panel(tmp_path)
    provenance = tmp_path / "arm-02-private" / "provenance.json"
    value = json.loads(provenance.read_text())
    value["gpu_uuid"] = "GPU-other"
    provenance.write_text(json.dumps(value))

    with pytest.raises(RepeatPanelError, match="physical GPU differs"):
        audit_replay_repeat_panel(tmp_path, repeat_count=2)


@pytest.mark.unit
def test_rejects_mislabeled_accumulator_topology(tmp_path: Path) -> None:
    _make_panel(tmp_path)
    provenance = tmp_path / "arm-03-shared" / "provenance.json"
    value = json.loads(provenance.read_text())
    value["shared_accumulators"] = False
    provenance.write_text(json.dumps(value))

    with pytest.raises(RepeatPanelError, match="accumulator topology differs"):
        audit_replay_repeat_panel(tmp_path, repeat_count=2)


@pytest.mark.unit
def test_rejects_missing_repeat_arm(tmp_path: Path) -> None:
    _make_panel(tmp_path)
    (tmp_path / "arm-02-private").rename(tmp_path / "arm-02-ignored")

    with pytest.raises(RepeatPanelError, match="expected 2 private replay arms"):
        audit_replay_repeat_panel(tmp_path, repeat_count=2)
