from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_vdam_xhalf_projection_pair_is_same_gpu_cache_balanced_and_pinned():
    runner = (
        ROOT / "scripts/run_vdam_xhalf_projection_microbatch_pair.sbatch"
    ).read_text()

    required = [
        'test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"',
        'vdam_select_target_gpu "${TARGET_GPU_UUID}" 0',
        'selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}',
        'shared_cache=${OUTPUT_ROOT}_jax_cache',
        'warmup:control',
        'warmup:candidate',
        'timed:candidate',
        'timed:control',
        'VDAM_JAX_COMPILATION_CACHE_DIR="${shared_cache}"',
        'RECOVAR_EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS="${row_pixels}"',
        'vdam_verify_selected_gpu "${selected_gpu_uuid}"',
        'CAPTURE_NATIVE_REPLAY=0',
    ]
    for text in required:
        assert text in runner
