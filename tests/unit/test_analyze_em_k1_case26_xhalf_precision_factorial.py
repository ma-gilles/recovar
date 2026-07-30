from __future__ import annotations

import pytest

from scripts import analyze_em_k1_case26_xhalf_precision_factorial as analyzer


def test_classifies_numbered_and_final_regression() -> None:
    assert (
        analyzer.classify_precision_effect(
            control_numbered_failures=0,
            double_numbered_failures=3,
            control_final_cross_fsc_auc=0.96,
            double_final_cross_fsc_auc=0.88,
        )
        == analyzer.CLASSIFICATION
    )


def test_classifies_final_only_regression() -> None:
    assert (
        analyzer.classify_precision_effect(
            control_numbered_failures=0,
            double_numbered_failures=0,
            control_final_cross_fsc_auc=0.96,
            double_final_cross_fsc_auc=0.95,
        )
        == "double_xhalf_mstep_worsens_case26_final_parity_only"
    )


def test_classifies_numbered_and_final_improvement() -> None:
    assert analyzer.classify_precision_effect(
        control_numbered_failures=3,
        double_numbered_failures=0,
        control_final_cross_fsc_auc=0.88,
        double_final_cross_fsc_auc=0.96,
    ) == ("double_xhalf_mstep_removes_numbered_failures_and_improves_case26_final_parity")


def test_classifies_exact_neutrality() -> None:
    assert (
        analyzer.classify_precision_effect(
            control_numbered_failures=0,
            double_numbered_failures=0,
            control_final_cross_fsc_auc=0.96,
            double_final_cross_fsc_auc=0.96,
        )
        == "double_xhalf_mstep_is_exactly_neutral_on_fixed_metrics"
    )


def _contract() -> dict[str, str]:
    values = {
        "HEAD": "a" * 40,
        "EM_K1_MATRIX_RUN_RELION": "1",
        "EM_K1_MATRIX_TRAJECTORY_MODE": "autonomous",
        "EM_K1_MATRIX_SCORECARD_MODE": "1",
        "EM_K1_MATRIX_FIXTURE_MANIFEST_SHA256": (analyzer.FIXTURE_MANIFEST_SHA256),
        "EM_K1_MATRIX_MAX_ITER": "999",
        "K1_IMAGE_BATCH_SIZE": "187",
        "K1_ROTATION_BLOCK_SIZE": "8192",
        "STREAMING_CHUNK_SIZE": "1000",
        "RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION": "0.40",
        "RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE": "0",
        "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT": "0",
        "RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT": "0",
        "RECOVAR_SAVE_INTERMEDIATES_DIR": "1",
        "RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED": "1",
    }
    return values


def test_matched_contract_accepts_only_precision_switch() -> None:
    control = {
        **_contract(),
        "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE": "0",
    }
    double = {
        **_contract(),
        "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE": "1",
    }

    result = analyzer.validate_matched_contract(control, double)

    assert result["matched_source_head"] == "a" * 40
    assert result["only_difference"] == {
        "name": "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE",
        "control": "0",
        "double": "1",
    }


def test_matched_contract_rejects_other_difference() -> None:
    control = {
        **_contract(),
        "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE": "0",
    }
    double = {
        **_contract(),
        "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE": "1",
        "K1_IMAGE_BATCH_SIZE": "188",
    }

    with pytest.raises(ValueError, match="outside precision arm"):
        analyzer.validate_matched_contract(control, double)


def test_matched_contract_rejects_unlisted_science_difference() -> None:
    control = {
        **_contract(),
        "RECOVAR_K1_DENSE_PASS2": "",
        "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE": "0",
    }
    double = {
        **control,
        "RECOVAR_K1_DENSE_PASS2": "1",
        "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE": "1",
    }

    with pytest.raises(ValueError, match="outside precision arm"):
        analyzer.validate_matched_contract(control, double)


def test_matched_contract_accepts_bookkeeping_differences() -> None:
    control = {
        **_contract(),
        "SCRATCH_DIR": "/scratch/control",
        "SETUP_JOB_ID": "1",
        "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE": "0",
    }
    double = {
        **_contract(),
        "SCRATCH_DIR": "/scratch/double",
        "SETUP_JOB_ID": "2",
        "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE": "1",
    }

    analyzer.validate_matched_contract(control, double)


def test_matched_contract_rejects_enabled_grid_correction() -> None:
    control = {
        **_contract(),
        "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE": "0",
        "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT": "1",
    }
    double = {
        **control,
        "RECOVAR_RELION_X_HALF_MSTEP_DOUBLE": "1",
    }

    with pytest.raises(ValueError, match="grid correction"):
        analyzer.validate_matched_contract(control, double)
