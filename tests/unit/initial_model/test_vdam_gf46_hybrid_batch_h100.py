from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "qualify_vdam_gf46_hybrid_batch_h100.py"
RUNNER = ROOT / "scripts" / "run_vdam_gf46_hybrid_batch_h100.sbatch"
SPEC = importlib.util.spec_from_file_location("qualify_vdam_gf46_hybrid_batch_h100", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_hybrid_qualification_uses_unreduced_gf46_geometry() -> None:
    MODULE._assert_exact_geometry()

    assert MODULE.GF46_GEOMETRY._asdict() == {
        "batch_size": 500,
        "translation_count": 29,
        "pixel_count": 5100,
        "rotation_block": 4608,
        "total_rotations": 36864,
        "source_rotation_block": 16,
    }
    assert MODULE.SELECTED_SOURCE_BLOCKS == 2
    assert MODULE.SELECTED_CAPACITY == 8
    assert MODULE.FALLBACK_CAPACITY == 1
    assert MODULE.FALLBACK_BATCH_SIZE == 8


@pytest.mark.unit
def test_harness_exercises_selected_posterior_and_physical_fallback() -> None:
    source = SCRIPT.read_text()

    for required in (
        "significance._compute_coarse_gaussian_gemm_hybrid_batch(",
        "relion_cuda_f32_coarse_posterior(",
        'result.fallback_reason == "block_capacity_overflow"',
        'np.all(block_ids[:, :SELECTED_SOURCE_BLOCKS] == (0, 1))',
        'np.all(selected_values == np.float32(-0.125))',
        'np.all(raw_max == np.float32(-0.125))',
        '"selected_repeat_summary_byte_identical": True',
    ):
        assert required in source
    assert "shifted[:FALLBACK_BATCH_SIZE]" in source
    assert "block_capacity=FALLBACK_CAPACITY" in source
    for option in (
        "--batch-size",
        "--translation-count",
        "--pixel-count",
        "--rotation-block",
        "--total-rotations",
    ):
        assert f'parser.add_argument("{option}"' not in source


@pytest.mark.unit
def test_runner_pins_source_and_builds_exact_h100_cuda_path() -> None:
    source = RUNNER.read_text()

    for required in (
        '"${EXPECTED_REPO_HEAD:?pin the committed hybrid qualification head}"',
        '"${EXPECTED_REPO_TREE:?pin the committed hybrid qualification tree}"',
        '"${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the hybrid source-manifest SHA-256}"',
        "status --porcelain=v1 --untracked-files=all",
        "--constraint=h100",
        "--gres=gpu:h100:1",
        '[[ "${gpu_name}" == *H100* ]]',
        '"CUDA_ARCH=${CUDA_ARCH_FLAGS}"',
        "RelionCoarseDiff2RectangularF32",
        "RelionCoarseDiff2RotationBlocksF32",
        "export JAX_ENABLE_X64=1",
        "export JAX_PLATFORMS=cuda,cpu",
        "--timed-runs 3",
    ):
        assert required in source
    assert "sbatch " not in source


@pytest.mark.unit
def test_source_manifest_covers_hybrid_dispatch_and_cuda_sources() -> None:
    required = {
        "recovar/cuda/cuda_backproject.cu",
        "recovar/cuda_backproject.py",
        "recovar/em/dense_single_volume/helpers/coarse_gemm_hybrid.py",
        "recovar/em/dense_single_volume/helpers/scoring.py",
        "recovar/em/dense_single_volume/helpers/significance.py",
        "scripts/qualify_vdam_gf46_hybrid_batch_h100.py",
        "scripts/run_vdam_gf46_hybrid_batch_h100.sbatch",
        "tests/unit/test_coarse_gemm_hybrid_significance.py",
        "tests/unit/test_coarse_gaussian_gemm_macro.py",
    }

    assert required.issubset(set(MODULE.SOURCE_FILES))
    encoded, entries = MODULE._source_manifest(ROOT)
    assert len(entries) == len(MODULE.SOURCE_FILES)
    assert all(line.count("  ") == 1 for line in encoded.decode().splitlines())
