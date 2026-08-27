from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_candidate_panel_runner_is_strict_reusable_and_fail_closed():
    source = (REPO_ROOT / "scripts/run_vdam_candidate_against_native_panel.sbatch").read_text()

    assert "#SBATCH --constraint=h100" in source
    assert "EXPECTED_REPO_HEAD" in source
    assert "EXPECTED_CUDA_SHA256" in source
    assert "EXPECTED_RELION_BIND_SHA256" in source
    assert 'test "$(sha256sum "${RELION_BIND_MATCHES[0]}"' in source
    assert "binding_path.is_file()" in source
    assert 'pathlib.Path(_relion_bind_core.__file__).resolve()).startswith' not in source
    assert "TARGET_GPU_UUID" in source
    assert "VDAM_TARGET_GPU_MISS" in source
    assert "exit 75" in source
    assert source.index("VDAM_TARGET_GPU_MISS") < source.index('mkdir -p "${RECOVAR_DIR}"')
    assert 'case "${variable_name}" in RECOVAR_*) unset "${variable_name}"' in source
    assert "VDAM_WORKER_SCHEDULE_NPZ" in source
    assert "EXPECTED_WORKER_SCHEDULE_SHA256" in source
    assert "RECOVAR_RELION_VDAM_WORKER_SCHEDULE_NPZ" in source
    assert "RECOVAR_RELION_VDAM_WORKER_REPLAY_TOPOLOGY" in source
    assert "single_rotation_sm132" in source
    assert "captured_block_start" in source
    assert "captured_block_grid" in source
    assert "captured_native_grid" in source
    assert "captured_native_count" in source
    assert "captured_native_trace_shape" in source
    assert "VDAM_BLOCK_CHRONOLOGY_NPZ" in source
    assert "EXPECTED_BLOCK_CHRONOLOGY_SHA256" in source
    assert "RECOVAR_RELION_VDAM_BLOCK_CHRONOLOGY_NPZ" in source
    assert source.index('case "${variable_name}" in RECOVAR_*) unset') < source.index(
        "export RECOVAR_RELION_VDAM_WORKER_SCHEDULE_NPZ"
    )
    assert "RECOVAR_INITIAL_MODEL_RELION_F32_COARSE_TIE_ULPS" not in source
    assert "build_recovar_command" in source
    assert 'int(definition["nr_classes"]) != 1' in source
    assert 'int(definition["nr_iter"]) != 200' in source
    assert "scripts/run_vdam_candidate_envelope_audits.sbatch" in source
    assert "VDAM_DEFER_AUDIT=${VDAM_DEFER_AUDIT:-1}" in source
    assert '--dependency="afterok:${SLURM_JOB_ID}"' in source
    assert "VDAM_MARK_CANDIDATE_COMPLETE=1" in source
    assert "AUDIT_SUBMITTED" in source
    assert 'sha256sum "${ANALYSIS_DIR}"/*' not in source
    assert "SCIENCE_COMPLETED" in source
    assert "COMPLETED" in source
