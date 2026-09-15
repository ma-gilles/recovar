from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_vdam_ordered_scatter_graph_gate_is_h100_focused_and_pinned():
    runner = (
        ROOT / "scripts/run_vdam_ordered_scatter_graph_gate.sbatch"
    ).read_text()

    required = [
        "#SBATCH --constraint=h100",
        ': "${EXPECTED_REPO_HEAD:?',
        ': "${OUTPUT_ROOT:?',
        'test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"',
        "RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1",
        'CUDA_ARCH="${CUDA_ARCH_FLAGS}"',
        "test_relion_vdam_fused_source_uses_native_separate_accumulator_storage",
        "test_relion_vdam_ordered_scatter_cuda_graph_matches_launch_serial",
        'sha256sum "${CUDA_BINARY}"',
        'touch "${OUTPUT_ROOT}/COMPLETED"',
    ]
    for text in required:
        assert text in runner
