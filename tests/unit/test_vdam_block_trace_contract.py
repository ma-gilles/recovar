from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_relion_block_trace_build_seals_the_complete_patch_and_both_schemas():
    build = (ROOT / "scripts/build_relion_vdam_block_trace.sbatch").read_text()
    required = [
        "EXPECTED_RELION_BASE",
        "RELION_LOCAL_BUILD_ROOT",
        'case "${RELION_LOCAL_BUILD_ROOT}" in',
        "/tmp/*",
        "QUALIFIED_BUILD=${RELION_BUILD_ROOT}/build",
        'install -m 755 "${BUILD}/bin/relion_refine" "${BINARY}"',
        'diff "${EXPECTED_RELION_BASE}"..HEAD',
        "src/acc/cuda/cuda_kernels/BP.cuh",
        "src/acc/cuda/vdam_block_trace.h",
        "src/acc/cuda/vdam_block_trace.cu",
        "RELION_VDAM_WORKER_LOG_SCHEMA_V2",
        "RELION_VDAM_BLOCK_TRACE_SCHEMA_V1",
        "sha256sum",
    ]
    for text in required:
        assert text in build


def test_relion_block_trace_runner_is_fail_closed_and_full_trajectory():
    runner = (ROOT / "scripts/run_vdam_relion_block_trace.sbatch").read_text()
    required = [
        "EXPECTED_PHYSICAL_GPU_UUID",
        "EXPECTED_RELION_BINARY_SHA256",
        "EXPECTED_REPO_HEAD",
        "64a2cfb5a0d6827944716b0de7e4a8ea13faccead2206ad0a4b422f98dd44897",
        "3c8280cdf7705028f7423e41dedd53b3f9894dd750d5fba44eea0e243a29ef5a",
        "export RELION_VDAM_WORKER_LOG=${WORKER_TRACE_FILE}",
        "export RELION_VDAM_WORKER_TRACE_ITER=1",
        "export RELION_VDAM_BLOCK_TRACE=${BLOCK_TRACE_FILE}",
        "export RELION_VDAM_BLOCK_TRACE_ITER=1",
        "--iter 200 --grad --denovo_3dref",
        "--random_seed 29 --j 8 --gpu 0",
        "scripts.build_vdam_worker_schedule",
        "scripts.build_vdam_block_chronology",
        "--n-particles 200",
        "--n-threads 8 --n-classes 1 --sm-count 132",
        "scripts.audit_vdam_kclass_trajectory",
        "scripts.audit_vdam_particle_state_trajectory",
    ]
    for text in required:
        assert text in runner


def test_relion_block_trace_checks_gpu_before_creating_native_outputs():
    runner = (ROOT / "scripts/run_vdam_relion_block_trace.sbatch").read_text()
    gpu_check = runner.index('if [[ "${physical_gpu_uuid}" != "${EXPECTED_PHYSICAL_GPU_UUID}" ]]')
    relion_run = runner.index('"${command[@]}"')
    assert gpu_check < relion_run
