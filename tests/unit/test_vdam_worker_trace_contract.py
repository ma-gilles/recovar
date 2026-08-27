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
        "export RELION_VDAM_WORKER_TRACE_ITER=1",
        "scripts.build_vdam_worker_schedule",
        "export RECOVAR_RELION_VDAM_WORKER_SCHEDULE_NPZ=${WORKER_SCHEDULE}",
    ]
    for text in required:
        assert text in runner
