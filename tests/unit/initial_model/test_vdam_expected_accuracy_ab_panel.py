from __future__ import annotations

from pathlib import Path

SCRIPT = Path("scripts/run_vdam_expected_accuracy_ab_panel.sbatch")


def test_expected_accuracy_ab_panel_is_same_allocation_and_fail_closed() -> None:
    text = SCRIPT.read_text()

    assert "for mode in baseline skip" in text
    assert 'for repeat in $(seq 1 "${REPEATS}")' in text
    assert "RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY=1" in text
    assert "RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY=0" in text
    assert "RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_SUBPROCESS=0" in text
    assert 'TARGET_GPU_UUID="${TARGET_GPU_UUID}"' in text
    assert 'BOUNDARY_DETERMINISTIC_CUDA=1' in text
    assert 'CAPTURE_NATIVE_REPLAY=0' in text
    assert 'bash "${REPO_ROOT}/scripts/run_vdam_first_state_boundary_capture.sbatch"' in text
    assert 'touch "${OUTPUT_ROOT}/COMPLETED"' in text


def test_expected_accuracy_ab_panel_records_provenance_and_disposable_markers() -> None:
    text = SCRIPT.read_text()

    assert 'touch "${OUTPUT_ROOT}/SAFE_TO_DELETE"' in text
    assert 'touch "${run_root}/SAFE_TO_DELETE"' in text
    assert 'git -C "${REPO_ROOT}" rev-parse HEAD' in text
    assert 'nvidia-smi -q > "${OUTPUT_ROOT}/provenance/nvidia_smi.txt"' in text
    assert 'env | LC_ALL=C sort > "${OUTPUT_ROOT}/provenance/submission_environment.txt"' in text
    assert 'printf \'%s\\n\' "${mode}"' in text
