from __future__ import annotations

from pathlib import Path

from scripts.audit_vdam_repeat_panel import classify_checkpoint

SBATCH_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_vdam_relion_repeat_panel.sbatch"


def test_repeat_panel_sbatch_keeps_sibling_repeat_roots_and_failed_evidence():
    text = SBATCH_PATH.read_text()

    assert 'PANEL_ROOT="${OUTPUT_ROOT}"' in text
    assert 'repeat_root="${PANEL_ROOT}/repeat-' in text
    assert '[[ ! -s "${audit_path}" ]]' in text


def test_repeat_panel_accepts_cross_engine_floor_inside_native_repeat_envelope():
    result = classify_checkpoint(
        relion_self_fsc_auc=[0.9971248, 0.99999],
        recovar_self_fsc_auc=[0.9999999],
        cross_engine_fsc_auc=[0.9971328, 0.9999934],
        gt_deltas=[-1e-7, -2e-7],
        cross_engine_min=0.999,
        gt_delta_min=-0.002,
    )

    assert result["pass"] is True
    assert all(result["checks"].values())


def test_repeat_panel_rejects_cross_engine_defect_below_native_repeat_floor():
    result = classify_checkpoint(
        relion_self_fsc_auc=[0.997],
        recovar_self_fsc_auc=[0.9999],
        cross_engine_fsc_auc=[0.990, 0.9995],
        gt_deltas=[0.0, 0.0],
        cross_engine_min=0.999,
        gt_delta_min=-0.002,
    )

    assert result["pass"] is False
    assert result["checks"]["worst_cross_engine_meets_frozen_gate_or_native_repeat_floor"] is False


def test_repeat_panel_keeps_frozen_gt_nondegradation_gate():
    result = classify_checkpoint(
        relion_self_fsc_auc=[0.995],
        recovar_self_fsc_auc=[0.9999],
        cross_engine_fsc_auc=[0.996, 0.9995],
        gt_deltas=[-0.003],
        cross_engine_min=0.999,
        gt_delta_min=-0.002,
    )

    assert result["pass"] is False
    assert result["checks"]["all_runs_meet_frozen_gt_nondegradation_gate"] is False
