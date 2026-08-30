from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_vdam_ordered_scatter_graph_pair_is_same_gpu_cache_balanced_and_pinned():
    runner = (
        ROOT / "scripts/run_vdam_ordered_scatter_graph_pair.sbatch"
    ).read_text()

    required = [
        'test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"',
        ': "${RECOVAR_CUDA_LIB_OVERRIDE:?',
        ': "${TARGET_RELION_PERTURBATION_SOURCE:?',
        'test "$(sha256sum "${RECOVAR_CUDA_LIB_OVERRIDE}"',
        'vdam_select_target_gpu "${TARGET_GPU_UUID}" 0',
        "selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}",
        "shared_cache=${OUTPUT_ROOT}_jax_cache",
        "warmup:control",
        "warmup:candidate",
        "timed:candidate",
        "timed:control",
        'VDAM_JAX_COMPILATION_CACHE_DIR="${shared_cache}"',
        'RECOVAR_VDAM_ORDERED_SCATTER_CUDA_GRAPH="${graph_enabled}"',
        'vdam_verify_selected_gpu "${selected_gpu_uuid}"',
        "CAPTURE_NATIVE_REPLAY=0",
        "RECOVAR_EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_PARTICLES=1",
        "RECOVAR_EXACT_LOCAL_SOURCE_BPREF_LAUNCH_SERIAL_ROTATIONS=1",
        "RECOVAR_VDAM_PRECOMPUTE_ORDERED_RESIDUALS=1",
        "RECOVAR_VDAM_FIXED_WARP_ORDER_SCATTER=1",
        "RECOVAR_K1_RELION_WAVG_SEQUENTIAL_CUDA=1",
        "RECOVAR_EXACT_LOCAL_BUCKET_RADIX=4",
        "RECOVAR_INITIAL_MODEL_PROFILE=1",
        "unset RECOVAR_VDAM_PRECOMPUTE_PERSISTENT_RESIDUALS",
        'runtime_environment.txt',
    ]
    for text in required:
        assert text in runner
