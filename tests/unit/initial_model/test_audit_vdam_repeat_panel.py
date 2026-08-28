from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_vdam_repeat_panel import (
    RepeatPanelError,
    _runtime_summary,
    classify_checkpoint,
    classify_native_radius_support,
)

SBATCH_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_vdam_relion_repeat_panel.sbatch"
SUMMARY_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_vdam_repeat_panel_summary.sbatch"


def test_repeat_panel_sbatch_keeps_sibling_repeat_roots_and_failed_evidence():
    text = SBATCH_PATH.read_text()

    assert 'PANEL_ROOT="${OUTPUT_ROOT}"' in text
    assert 'repeat_root="${PANEL_ROOT}/repeat-' in text
    assert '[[ ! -s "${audit_path}" ]]' in text


def test_repeat_panel_summary_can_resume_after_science_job():
    text = SUMMARY_PATH.read_text()

    expected_tokens = [
        'test -s "${repeat_root}/trajectory_audit.json"',
        'if [[ -s "${REPORT}" && -s "${SHELLS}" ]]; then',
        '"${PIXI_PY}" -m scripts.audit_vdam_repeat_panel',
        "#SBATCH --partition=cpu",
        "#SBATCH --time=08:00:00",
    ]
    missing = [token for token in expected_tokens if token not in text]
    assert not missing, f"VDAM repeat summary lost restart-safe wiring: {missing}"


def test_repeat_panel_accepts_bidirectional_native_mode_matches():
    result = classify_checkpoint(
        relion_self_fsc_auc=[0.9971248],
        recovar_self_fsc_auc=[0.9971328],
        cross_engine_fsc_auc=[
            [0.9999934, 0.9971247],
            [0.9971328, 0.9999999],
        ],
        gt_deltas=[-1e-7, -2e-7],
        cross_engine_min=0.999,
        gt_delta_min=-0.002,
    )

    assert result["pass"] is True
    assert all(result["checks"].values())


def test_repeat_panel_rejects_candidate_mode_without_native_match():
    result = classify_checkpoint(
        relion_self_fsc_auc=[0.997],
        recovar_self_fsc_auc=[0.9971],
        cross_engine_fsc_auc=[[0.9995, 0.9994], [0.990, 0.991]],
        gt_deltas=[0.0, 0.0],
        cross_engine_min=0.999,
        gt_delta_min=-0.002,
    )

    assert result["pass"] is False
    assert result["checks"]["every_candidate_run_matches_native_mode_at_frozen_gate"] is False


def test_repeat_panel_rejects_native_mode_missing_from_candidate_panel():
    result = classify_checkpoint(
        relion_self_fsc_auc=[0.997],
        recovar_self_fsc_auc=[0.9999],
        cross_engine_fsc_auc=[[0.9995, 0.990], [0.9994, 0.991]],
        gt_deltas=[0.0, 0.0],
        cross_engine_min=0.999,
        gt_delta_min=-0.002,
    )

    assert result["pass"] is False
    assert result["checks"]["every_native_run_has_candidate_mode_at_frozen_gate"] is False


def test_repeat_panel_keeps_frozen_gt_nondegradation_gate():
    result = classify_checkpoint(
        relion_self_fsc_auc=[0.995],
        recovar_self_fsc_auc=[0.9999],
        cross_engine_fsc_auc=[[0.9995, 0.996], [0.996, 0.9995]],
        gt_deltas=[-0.003],
        cross_engine_min=0.999,
        gt_delta_min=-0.002,
    )

    assert result["pass"] is False
    assert result["checks"]["all_runs_meet_frozen_gt_nondegradation_gate"] is False


def test_native_radius_support_accepts_cross_modes_inside_native_repeat_radius():
    result = classify_native_radius_support(
        relion_self_fsc_auc=[0.54],
        cross_engine_fsc_auc=[[0.74, 0.56], [0.73, 0.57]],
    )

    assert result["pass"] is True
    assert result["candidate_validity_pass"] is True
    assert result["reverse_native_coverage_pass"] is True
    assert result["candidate_matching_native_repeat_indices"] == [[1, 2], [1, 2]]


def test_native_radius_support_reports_candidate_and_reverse_failures_separately():
    candidate_failure = classify_native_radius_support(
        relion_self_fsc_auc=[0.8],
        cross_engine_fsc_auc=[[0.7, 0.6], [0.85, 0.82]],
    )
    reverse_failure = classify_native_radius_support(
        relion_self_fsc_auc=[0.8],
        cross_engine_fsc_auc=[[0.85, 0.79], [0.84, 0.78]],
    )

    assert candidate_failure["candidate_validity_pass"] is False
    assert candidate_failure["reverse_native_coverage_pass"] is True
    assert reverse_failure["candidate_validity_pass"] is True
    assert reverse_failure["reverse_native_coverage_pass"] is False


def test_native_radius_support_uses_each_native_nearest_peer():
    result = classify_native_radius_support(
        relion_self_fsc_auc=[0.9, 0.6, 0.7],
        cross_engine_fsc_auc=[
            [0.91, 0.89, 0.71],
            [0.89, 0.91, 0.72],
            [0.61, 0.71, 0.71],
        ],
    )

    assert result["native_nearest_peer_fsc_auc"] == pytest.approx([0.9, 0.9, 0.7])
    assert result["candidate_matching_native_repeat_indices"] == [[1, 3], [2, 3], [3]]
    assert result["pass"] is True


def test_native_radius_support_rejects_wrong_native_pair_cardinality():
    with pytest.raises(RepeatPanelError, match="expected 3 finite native pairs"):
        classify_native_radius_support(
            relion_self_fsc_auc=[0.9],
            cross_engine_fsc_auc=[[0.9, 0.8, 0.7], [0.8, 0.9, 0.7]],
        )


def _write_timing(root: Path, *, relion_wall: float, recovar_wall: float) -> None:
    for engine, wall in (("relion", relion_wall), ("recovar", recovar_wall)):
        directory = root / engine
        directory.mkdir(parents=True)
        (directory / f"{engine}.timing.json").write_text(
            json.dumps({"exit_status": 0, "external_wall_s": wall})
        )


def test_repeat_panel_reports_runtime_distribution(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_timing(first, relion_wall=10.0, recovar_wall=50.0)
    _write_timing(second, relion_wall=20.0, recovar_wall=80.0)

    result = _runtime_summary(
        [{"index": 1, "root": first}, {"index": 2, "root": second}]
    )

    assert result["scoring"] is False
    assert result["relion_wall_s"]["median"] == pytest.approx(15.0)
    assert result["recovar_wall_s"]["median"] == pytest.approx(65.0)
    assert result["recovar_over_relion"]["median"] == pytest.approx(4.5)


def test_repeat_panel_rejects_invalid_timing(tmp_path):
    root = tmp_path / "repeat"
    _write_timing(root, relion_wall=0.0, recovar_wall=1.0)

    with pytest.raises(RepeatPanelError, match="must be positive"):
        _runtime_summary([{"index": 1, "root": root}])
