from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_relion_worker_trace_runner_is_fail_closed_and_full_schedule():
    runner = (ROOT / "scripts/run_vdam_relion_worker_trace.sbatch").read_text()
    required = [
        "EXPECTED_PHYSICAL_GPU_UUID",
        'export RELION_VDAM_WORKER_LOG=${TRACE_FILE}',
        "export RELION_VDAM_WORKER_TRACE_ITER=1",
        "--iter 200 --grad --denovo_3dref",
        "--n-particles 200",
        "--dataset-particles 3000",
        "--n-threads 8 --pool-size 24",
        "scripts.audit_vdam_kclass_trajectory",
        "scripts.audit_vdam_particle_state_trajectory",
    ]
    for text in required:
        assert text in runner


def test_relion_worker_trace_build_seals_complete_gpu_patch():
    build = (ROOT / "scripts/build_relion_vdam_worker_trace.sbatch").read_text()
    required = [
        "EXPECTED_RELION_BASE",
        'diff "${EXPECTED_RELION_BASE}"..HEAD',
        "src/ml_optimiser.cpp src/ml_optimiser.h",
        "src/acc/cuda/cuda_ml_optimiser.cu",
    ]
    for text in required:
        assert text in build


def test_fullschedule_boundary_can_replay_the_same_native_worker_trace():
    runner = (ROOT / "scripts/run_vdam_fullschedule_mstep_boundary.sbatch").read_text()
    required = [
        "RELION_VDAM_WORKER_TRACE_REPLAY",
        "export RELION_VDAM_WORKER_LOG=${WORKER_TRACE}",
        "export RELION_VDAM_WORKER_TRACE_ITER=${WORKER_TRACE_ITERATION}",
        "scripts.build_vdam_worker_schedule",
        "export RECOVAR_RELION_VDAM_WORKER_SCHEDULE_NPZ=${CANDIDATE_WORKER_SCHEDULE}",
    ]
    for text in required:
        assert text in runner


def test_fullschedule_boundary_seals_and_replays_each_native_block_chronology():
    runner = (ROOT / "scripts/run_vdam_fullschedule_mstep_boundary.sbatch").read_text()
    required = [
        "RELION_VDAM_BLOCK_TRACE_REPLAY",
        "export RELION_VDAM_BLOCK_TRACE=${BLOCK_TRACE}",
        "export RELION_VDAM_BLOCK_TRACE_ITER=1",
        "scripts.build_vdam_block_chronology",
        "captured block topology requires native block tracing in the same arm",
        "export RECOVAR_RELION_VDAM_BLOCK_CHRONOLOGY_NPZ=${REPLAY_BLOCK_CHRONOLOGY}",
        "export RECOVAR_RELION_VDAM_WORKER_REPLAY_TOPOLOGY=${WORKER_REPLAY_TOPOLOGY}",
    ]
    for text in required:
        assert text in runner


def test_fullschedule_boundary_can_seal_passive_candidate_block_chronology():
    runner = (ROOT / "scripts/run_vdam_fullschedule_mstep_boundary.sbatch").read_text()
    required = [
        "RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE_CAPTURE",
        "export RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE=${CANDIDATE_BLOCK_TRACE}",
        "export RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE_ITER=${TARGET_ITERATION}",
        "--capture \"${CANDIDATE_BLOCK_TRACE}\"",
        "--output-npz \"${CANDIDATE_BLOCK_CHRONOLOGY}\"",
        "scripts.analyze_vdam_particle_issue_chronology",
        "--native-chronology \"${REPLAY_BLOCK_CHRONOLOGY}\"",
        '"${WORKER_REPLAY_TOPOLOGY}" = captured_particle_issue',
        '"${WORKER_REPLAY_TOPOLOGY}" = captured_particle_timing',
        'test "${WORKER_REPLAY_TOPOLOGY}" = captured',
    ]
    for text in required:
        assert text in runner


def test_native_block_capture_is_independent_of_serial_replay_topology():
    runner = (ROOT / "scripts/run_vdam_fullschedule_mstep_boundary.sbatch").read_text()
    assert "RELION_VDAM_BLOCK_TRACE_CAPTURE" in runner
    assert 'if [[ "${BLOCK_TRACE_CAPTURE}" = 1 ]]; then' in runner
    assert 'if [[ "${BLOCK_TRACE_REPLAY}" = 1 ]]; then' in runner
    assert 'test "${BLOCK_TRACE_CAPTURE}" = 1' in runner


def test_fullschedule_boundary_seals_candidate_to_native_block_map():
    runner = (ROOT / "scripts/run_vdam_fullschedule_mstep_boundary.sbatch").read_text()
    required = [
        "RECOVAR_VDAM_CANDIDATE_BLOCK_MAP_CAPTURE",
        "export RECOVAR_VDAM_CANDIDATE_BLOCK_MAP=${CANDIDATE_BLOCK_MAP}",
        "scripts.build_vdam_candidate_block_map",
        '--candidate-chronology "${CANDIDATE_BLOCK_CHRONOLOGY}"',
        '--native-chronology "${BLOCK_CHRONOLOGY}"',
        '--worker-schedule "${WORKER_SCHEDULE}"',
    ]
    for text in required:
        assert text in runner
