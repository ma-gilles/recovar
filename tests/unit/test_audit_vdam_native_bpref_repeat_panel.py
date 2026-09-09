from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_vdam_native_bpref_repeat_panel import (
    audit_native_bpref_repeat_panel,
)
from scripts.audit_vdam_repeat_panel import RepeatPanelError

SCIENCE_HEAD = "a" * 40
REPLAY_HEAD = "b" * 40
CUDA_HASH = "c" * 64
GPU_UUID = "GPU-test"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_relion(path: Path, value: np.ndarray, *, complex_values: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.asarray(value)
    with path.open("wb") as stream:
        np.asarray((1, 1, value.size), dtype=np.int64).tofile(stream)
        if complex_values:
            np.asarray(value, dtype=np.complex128).ravel().view(np.float64).tofile(stream)
        else:
            np.asarray(value, dtype=np.float64).ravel().tofile(stream)


def _make_replay_panel(root: Path) -> None:
    _write_json(
        root / "submission_provenance.json",
        {
            "schema": "recovar.vdam_replay_repeat_panel_submission.v2",
            "science_head": REPLAY_HEAD,
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
                "science_head": REPLAY_HEAD,
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
        value = np.asarray([[first_value, 1.0], [first_value, 1.0]], dtype=np.float32)
        np.savez(
            arm / "replay" / "worker_private_accumulators.npz",
            reduced_real=value,
            reduced_imag=value,
            reduced_weight=value,
        )


def _make_native_panel(root: Path, *, second_gpu: str = GPU_UUID) -> None:
    filenames = (
        ("pipe_it1_c0_bp_data_pre_reweight.bin", True),
        ("pipe_it1_c0_bp_weight.bin", False),
        ("pipe_it1_c0_bp_data_h_pre_reweight.bin", True),
        ("pipe_it1_c0_bp_weight_h.bin", False),
    )
    for index, first_value in enumerate((1.0, 2.0), start=1):
        arm = root / f"repeat-{index:02d}"
        _write_json(
            arm / "provenance" / "native_only_completion.json",
            {
                "schema": "recovar.vdam_native_bpref_repeat.v1",
                "job_id": str(index),
                "git_head": SCIENCE_HEAD,
                "physical_gpu_uuid": GPU_UUID if index == 1 else second_gpu,
                "iteration": 1,
            },
        )
        (arm / "provenance" / "native_only_evidence.sha256").write_text("seal\n")
        (arm / "NATIVE_ONLY_SUCCESS").touch()
        value = np.asarray([first_value, 1.0])
        for filename, complex_values in filenames:
            _write_relion(
                arm / "native_mstep" / filename,
                value,
                complex_values=complex_values,
            )


@pytest.mark.unit
def test_compares_native_private_and_shared_bpref_width(tmp_path: Path) -> None:
    native = tmp_path / "native"
    replay = tmp_path / "replay"
    _make_native_panel(native)
    _make_replay_panel(replay)

    report = audit_native_bpref_repeat_panel(
        native, replay, repeat_count=2, iteration=1
    )

    assert report["result"] == "complete"
    assert report["physical_gpu_uuid"] == GPU_UUID
    np.testing.assert_allclose(
        report["replay_over_native_maximum_diameter"]["private"]["data_half0"],
        1.0,
    )
    np.testing.assert_allclose(
        report["replay_over_native_maximum_diameter"]["shared"]["data_half0"],
        3.0,
    )


@pytest.mark.unit
def test_rejects_native_repeats_from_mixed_physical_gpus(tmp_path: Path) -> None:
    native = tmp_path / "native"
    replay = tmp_path / "replay"
    _make_native_panel(native, second_gpu="GPU-other")
    _make_replay_panel(replay)

    with pytest.raises(RepeatPanelError, match="spans physical GPUs"):
        audit_native_bpref_repeat_panel(native, replay, repeat_count=2)


@pytest.mark.unit
def test_rejects_missing_native_repeat(tmp_path: Path) -> None:
    native = tmp_path / "native"
    replay = tmp_path / "replay"
    _make_native_panel(native)
    _make_replay_panel(replay)
    (native / "repeat-02").rename(native / "ignored")

    with pytest.raises(RepeatPanelError, match="expected 2 native BPref repeats"):
        audit_native_bpref_repeat_panel(native, replay, repeat_count=2)
