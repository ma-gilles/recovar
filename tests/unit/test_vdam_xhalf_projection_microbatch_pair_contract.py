from pathlib import Path

from scripts.summarize_vdam_xhalf_projection_microbatch_pair import derive_it80_topology

ROOT = Path(__file__).resolve().parents[2]


def test_vdam_xhalf_projection_pair_is_same_gpu_warm_balanced_and_pinned():
    runner = (
        ROOT / "scripts/run_vdam_xhalf_projection_microbatch_pair.sbatch"
    ).read_text()

    required = [
        'ACTUAL_REPO_HEAD=$(git -C "${REPO_ROOT}" rev-parse HEAD)',
        ': "${RECOVAR_CUDA_LIB_OVERRIDE:?',
        ': "${LAYOUT_EVIDENCE_META:?',
        "#SBATCH --exclusive",
        'vdam_select_target_gpu "${TARGET_GPU_UUID}" 0',
        "selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}",
        "cache_control=${OUTPUT_ROOT}_jax_cache_control",
        "cache_candidate=${OUTPUT_ROOT}_jax_cache_candidate",
        "warmup-control warmup-candidate",
        "scored-control-1 scored-candidate-1 scored-candidate-2 scored-control-2",
        'VDAM_JAX_COMPILATION_CACHE_DIR="${cache_dir}"',
        "export RECOVAR_EXACT_LOCAL_XHALF_PROJECTION_TARGET_ROW_PIXELS=${row_pixels}",
        "unset RECOVAR_INITIAL_MODEL_PROFILE",
        'vdam_verify_selected_gpu "${selected_gpu_uuid}"',
        "CAPTURE_NATIVE_REPLAY=0",
        'start_ns=$(date +%s%N)',
        'end_ns=$(date +%s%N)',
        "gpu_memory.tsv",
        '--query-gpu=memory.used,memory.total',
        "summarize_vdam_xhalf_projection_microbatch_pair",
    ]
    for text in required:
        assert text in runner

    assert "RECOVAR_CUDA_LIB_OVERRIDE) ;;" in runner
    assert "RECOVAR_*) unset" in runner
    for variable in (
        "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE",
        "RECOVAR_K1_COARSE_FUSED_PROJECTOR",
        "RECOVAR_K1_COARSE_GAUSSIAN_SKIP_PADDED_IMAGES",
        "RECOVAR_K1_COARSE_GAUSSIAN_FFI",
        "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF",
        "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS",
        "RECOVAR_K1_RELION_F32_COARSE_SUPPORT",
        "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION",
    ):
        assert f"unset {variable}" in runner


def test_gf46_it80_layout_derives_2x_3x_and_5x_bucket_counts(monkeypatch):
    monkeypatch.delenv("RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS", raising=False)
    monkeypatch.delenv("RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB", raising=False)
    evidence = {
        "n_translations": 84,
        "halfset_0_profile_summary": {
            "chunk_sizes": [75, 75, 75, 75, 60],
            "chunk_padded_rotations": [4800, 4800, 4800, 4800, 3840],
            "n_projection_windowed": 8320,
        },
    }

    expected = {
        40_000_000: [75, 75, 75, 75, 60],
        80_000_000: [150, 150, 60],
        120_000_000: [225, 135],
        200_000_000: [266, 94],
    }
    for row_pixels, chunk_sizes in expected.items():
        topology = derive_it80_topology(evidence, row_pixels)
        assert topology["predicted_chunk_sizes"] == chunk_sizes
        assert topology["predicted_big_jit_bucket_count"] == len(chunk_sizes)
        assert sum(topology["predicted_chunk_padded_rotations"]) == 23_040
