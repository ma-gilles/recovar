from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path("scripts/run_vdam_expected_accuracy_ab_panel.sbatch")
GPU_HELPER = Path("scripts/vdam_gpu_selection.sh")
BOUNDARY_SCRIPT = Path("scripts/run_vdam_first_state_boundary_capture.sbatch")


def test_expected_accuracy_ab_panel_is_same_allocation_and_fail_closed() -> None:
    text = SCRIPT.read_text()

    assert 'for repeat in $(seq 1 "${REPEATS}")' in text
    assert "modes=(baseline skip)" in text
    assert "modes=(skip baseline)" in text
    assert 'for mode in "${modes[@]}"' in text
    assert text.index('for repeat in $(seq 1 "${REPEATS}")') < text.index(
        'for mode in "${modes[@]}"'
    )
    assert "RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY=1" in text
    assert "RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY=0" in text
    assert "RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_SUBPROCESS=0" in text
    assert 'TARGET_GPU_UUID="${TARGET_GPU_UUID}"' in text
    assert 'test "${GPU_MISS_HOLD_SECONDS}" -le 60' in text
    assert 'source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"' in text
    assert 'vdam_select_target_gpu "${TARGET_GPU_UUID}" "${GPU_MISS_HOLD_SECONDS}"' in text
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
    assert '"${OUTPUT_ROOT}/provenance/selected_gpu_uuid.txt"' in text
    assert '"${OUTPUT_ROOT}/provenance/allocated_gpu_uuids.csv"' in text
    assert 'env | LC_ALL=C sort > "${OUTPUT_ROOT}/provenance/submission_environment.txt"' in text
    assert '"${OUTPUT_ROOT}/provenance/execution_order.tsv"' in text
    assert 'printf \'%s\\n\' "${mode}"' in text


def test_shared_gpu_selector_handles_multi_gpu_target_and_missing_target() -> None:
    helper = GPU_HELPER.read_text()
    script = f"""
source {GPU_HELPER}
nvidia-smi() {{ printf '%s\\n' GPU-wrong GPU-target; }}
vdam_select_target_gpu GPU-target 0
printf '%s|%s|%s\\n' "$VDAM_SELECTED_GPU_UUID" "$CUDA_VISIBLE_DEVICES" "$VDAM_VISIBLE_GPU_UUIDS_CSV"
vdam_verify_selected_gpu "$VDAM_SELECTED_GPU_UUID"
"""
    selected = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert selected.stdout.strip() == "GPU-target|GPU-target|GPU-wrong,GPU-target"

    missing = subprocess.run(
        [
            "bash",
            "-c",
            f"source {GPU_HELPER}; nvidia-smi() {{ printf '%s\\n' GPU-wrong; }}; "
            "vdam_select_target_gpu GPU-target 0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 75
    assert "VDAM_TARGET_GPU_MISS expected=GPU-target observed=GPU-wrong" in missing.stderr
    assert "CUDA_VISIBLE_DEVICES=${VDAM_SELECTED_GPU_UUID}" in helper


def test_boundary_runner_reuses_selected_physical_gpu() -> None:
    boundary = BOUNDARY_SCRIPT.read_text()

    assert 'source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"' in boundary
    assert 'vdam_select_target_gpu "${TARGET_GPU_UUID}" 0' in boundary
    assert 'vdam_verify_selected_gpu "${gpu_uuid_before}"' in boundary
    assert "gpu_uuid_after_relion=${VDAM_SELECTED_GPU_UUID}" in boundary
    assert "gpu_uuid_after_recovar=${VDAM_SELECTED_GPU_UUID}" in boundary
