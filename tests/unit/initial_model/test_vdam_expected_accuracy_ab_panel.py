from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path("scripts/run_vdam_expected_accuracy_ab_panel.sbatch")
GPU_HELPER = Path("scripts/vdam_gpu_selection.sh")
BOUNDARY_SCRIPT = Path("scripts/run_vdam_first_state_boundary_capture.sbatch")
BPREF_CHUNK_SCRIPT = Path("scripts/run_vdam_bpref_particle_chunk_panel.sbatch")
WARM_NSYS_SCRIPT = Path("scripts/run_vdam_warm_cache_nsys_pair.sbatch")


def test_expected_accuracy_ab_panel_is_same_allocation_and_fail_closed() -> None:
    text = SCRIPT.read_text()

    assert 'for repeat in $(seq 1 "${REPEATS}")' in text
    assert "VDAM_EXPECTED_ACCURACY_MODES=${VDAM_EXPECTED_ACCURACY_MODES:-baseline,skip}" in text
    assert "requested_modes" in text
    assert "unsupported panel mode" in text
    assert "duplicate panel mode" in text
    assert "repeat % 2 == 0" in text
    assert 'modes=("${requested_modes[1]}" "${requested_modes[0]}")' in text
    assert 'modes=("${requested_modes[@]}")' in text
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
    assert "VDAM_BOUNDARY_DETERMINISTIC_CUDA=${VDAM_BOUNDARY_DETERMINISTIC_CUDA:-1}" in text
    assert "VDAM_BOUNDARY_DETERMINISTIC_CUDA must be 0 or 1" in text
    assert 'BOUNDARY_DETERMINISTIC_CUDA="${VDAM_BOUNDARY_DETERMINISTIC_CUDA}"' in text
    assert "VDAM_CAPTURE_LOCAL_SCORE=${VDAM_CAPTURE_LOCAL_SCORE:-0}" in text
    assert "VDAM_CAPTURE_FUSED_SCORES=${VDAM_CAPTURE_FUSED_SCORES:-0}" in text
    assert "VDAM_CAPTURE_COARSE_SCORE=${VDAM_CAPTURE_COARSE_SCORE:-0}" in text
    assert 'CAPTURE_LOCAL_SCORE="${VDAM_CAPTURE_LOCAL_SCORE}"' in text
    assert 'CAPTURE_FUSED_SCORES="${VDAM_CAPTURE_FUSED_SCORES}"' in text
    assert 'CAPTURE_COARSE_SCORE="${VDAM_CAPTURE_COARSE_SCORE}"' in text
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


def test_boundary_runner_allows_a_pinned_shared_jax_cache() -> None:
    text = BOUNDARY_SCRIPT.read_text()

    assert (
        "export JAX_COMPILATION_CACHE_DIR="
        "${VDAM_JAX_COMPILATION_CACHE_DIR:-${OUTPUT_ROOT}/jax_cache}"
    ) in text
    assert 'touch "${JAX_COMPILATION_CACHE_DIR}/SAFE_TO_DELETE"' in text


def test_boundary_runner_supports_a_bounded_delayed_nsys_trace() -> None:
    text = BOUNDARY_SCRIPT.read_text()

    assert "VDAM_NSYS_OUTPUT=${VDAM_NSYS_OUTPUT:-}" in text
    assert '[[ "${VDAM_NSYS_OUTPUT}" = /* ]]' in text
    assert 'test -x "${VDAM_NSYS_BIN}"' in text
    assert '--trace=cuda,nvtx,osrt' in text
    assert '--sample=none' in text
    assert '--cpuctxsw=none' in text
    assert '--kill=none' in text
    assert '"--delay=${VDAM_NSYS_DELAY_S}"' in text
    assert '"--duration=${VDAM_NSYS_DURATION_S}"' in text
    assert 'test -s "${VDAM_NSYS_OUTPUT}.nsys-rep"' in text
    assert '"${RUN_COMMAND[@]}"' in text


def test_warm_nsys_pair_reuses_one_cache_and_profiles_only_repeat_two() -> None:
    text = WARM_NSYS_SCRIPT.read_text()

    assert 'for repeat in 1 2' in text
    assert 'shared_cache=${OUTPUT_ROOT}_jax_cache' in text
    assert 'touch "${OUTPUT_ROOT}/SAFE_TO_DELETE" "${shared_cache}/SAFE_TO_DELETE"' in text
    assert 'if [[ "${repeat}" == 2 ]]' in text
    assert 'nsys_output=${run_root}/nsight/vdam_warm' in text
    assert 'VDAM_JAX_COMPILATION_CACHE_DIR="${shared_cache}"' in text
    assert 'VDAM_NSYS_OUTPUT="${nsys_output}"' in text
    assert 'TARGET_GPU_UUID="${selected_gpu_uuid}"' in text
    assert 'CAPTURE_NATIVE_REPLAY=0' in text
    assert 'bash "${REPO_ROOT}/scripts/run_vdam_first_state_boundary_capture.sbatch"' in text
    assert 'touch "${OUTPUT_ROOT}/COMPLETED"' in text


def test_bpref_particle_chunk_panel_reuses_and_interleaves_the_boundary_runner() -> None:
    text = BPREF_CHUNK_SCRIPT.read_text()

    assert "VDAM_BPREF_PARTICLE_CHUNK_ARMS=${VDAM_BPREF_PARTICLE_CHUNK_ARMS:-default,3}" in text
    assert 'for repeat in $(seq 1 "${REPEATS}")' in text
    assert "repeat % 2 == 0" in text
    assert "unsupported BPref particle chunk arm" in text
    assert "duplicate BPref particle chunk arm" in text
    assert "unset RECOVAR_EXACT_LOCAL_SOURCE_BPREF_PARTICLE_CHUNK_SIZE" in text
    assert "unset RECOVAR_EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_PARTICLES" in text
    assert "unset RECOVAR_EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_ROTATIONS" in text
    assert "export RECOVAR_EXACT_LOCAL_SOURCE_BPREF_PARTICLE_CHUNK_SIZE=${arm}" in text
    assert '"${arm}" == fused-serial' in text
    assert "export RECOVAR_EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_PARTICLES=1" in text
    assert '"${arm}" == fused-serial-rotations' in text
    assert "export RECOVAR_EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_ROTATIONS=1" in text
    assert "VDAM_EXPECTED_ACCURACY_MODES=baseline" in text
    assert "VDAM_CAPTURE_LOCAL_SCORE=0" in text
    assert "VDAM_CAPTURE_FUSED_SCORES=0" in text
    assert "VDAM_CAPTURE_COARSE_SCORE=0" in text
    assert 'bash "${PANEL_RUNNER}"' in text
    assert 'touch "${OUTPUT_ROOT}/SAFE_TO_DELETE"' in text
    assert '"${OUTPUT_ROOT}/provenance/execution_order.tsv"' in text
    assert 'touch "${OUTPUT_ROOT}/COMPLETED"' in text


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
