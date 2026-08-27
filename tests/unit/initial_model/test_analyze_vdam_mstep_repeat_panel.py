from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_vdam_mstep_repeat_panel as analysis_module
from scripts.analyze_vdam_mstep_boundary import SCHEMA as BOUNDARY_SCHEMA


def _write_arm(root: Path, *, cross_relative_l2: float, recovar_value: float) -> None:
    native = root / "native_mstep"
    recovar = root / "recovar_mstep"
    analysis = root / "analysis"
    native.mkdir(parents=True)
    recovar.mkdir()
    analysis.mkdir()
    np.save(recovar / "recovar.npy", np.full(4, recovar_value))
    (analysis / "mstep_boundary.json").write_text(
        json.dumps(
            {
                "schema": BOUNDARY_SCHEMA,
                "status": "complete",
                "iteration": 1,
                "native_directory": str(native.resolve()),
                "recovar_directory": str(recovar.resolve()),
                "comparisons": {"stage": {"relative_l2": cross_relative_l2}},
            }
        )
    )


def test_repeat_panel_compares_both_cross_arms_to_native_repeat(tmp_path, monkeypatch):
    arm_a = tmp_path / "a"
    arm_b = tmp_path / "b"
    _write_arm(arm_a, cross_relative_l2=0.2, recovar_value=1.5)
    _write_arm(arm_b, cross_relative_l2=0.3, recovar_value=1.6)
    monkeypatch.setattr(
        analysis_module,
        "_stages_for_iteration",
        lambda iteration: (("stage", "native.bin", "recovar.npy", False),),
    )

    def fake_native(path: Path, *, complex_values: bool) -> np.ndarray:
        assert complex_values is False
        return np.full(4, 1.0 if path.is_relative_to(arm_a) else 1.1)

    monkeypatch.setattr(analysis_module, "_read_relion_array", fake_native)
    report = analysis_module.analyze_repeat_panel(arm_a, arm_b, iteration=1)

    assert report["schema"] == analysis_module.SCHEMA
    assert report["native_repeat"]["stage"]["relative_l2"] == pytest.approx(0.1)
    assert report["recovar_repeat"]["stage"]["relative_l2"] == pytest.approx(1.0 / 15.0)
    assert report["native_floor_ratios"]["stage"] == {
        "cross_arm_a_over_native_repeat": pytest.approx(2.0),
        "cross_arm_b_over_native_repeat": pytest.approx(3.0),
    }


def test_repeat_panel_rejects_mixed_cross_report_provenance(tmp_path):
    arm_a = tmp_path / "a"
    arm_b = tmp_path / "b"
    _write_arm(arm_a, cross_relative_l2=0.2, recovar_value=1.5)
    _write_arm(arm_b, cross_relative_l2=0.3, recovar_value=1.6)
    path = arm_b / "analysis/mstep_boundary.json"
    report = json.loads(path.read_text())
    report["iteration"] = 2
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="expected iteration=1"):
        analysis_module.analyze_repeat_panel(arm_a, arm_b, iteration=1)


def test_repeat_panel_runner_pins_same_gpu_and_nested_capture_contract():
    runner = (
        Path(__file__).parents[3]
        / "scripts/run_vdam_fullschedule_mstep_repeat_panel.sbatch"
    ).read_text()
    required = (
        "#SBATCH --constraint=h100",
        "#SBATCH --gres=gpu:h100:1",
        '${EXPECTED_REPO_HEAD:?pin the tracked source head}',
        '${RECOVAR_CUDA_LIB_SOURCE:?set the immutable qualified CUDA binary}',
        'gpu_uuid_before=$(nvidia-smi --query-gpu=uuid',
        'for arm in a b; do',
        "scripts/run_vdam_fullschedule_mstep_boundary.sbatch",
        'RUN_SUCCESS_${SLURM_JOB_ID}',
        "scripts.analyze_vdam_mstep_repeat_panel",
        'status --porcelain=v1 --untracked-files=no',
        'sha256sum "${REPORT}"',
        "RELION_VDAM_BLOCK_TRACE_REPLAY",
        "RECOVAR_RELION_VDAM_WORKER_REPLAY_TOPOLOGY",
    )
    missing = [token for token in required if token not in runner]
    assert not missing, f"M-step repeat panel lost provenance/same-GPU gates: {missing}"
