"""GPU memory stress test: full pipeline on a 256x256, 5k-image synthetic dataset."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.long_test]


def _resolve_output_dir(tmp_path: Path, name: str) -> Path:
    base = os.environ.get("LONG_METRICS_OUTPUT_BASE")
    if base:
        out_dir = Path(base) / "pytest_gpu_stress" / name
    else:
        out_dir = tmp_path / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_pipeline_gpu_memory_stress_256_5k(tmp_path):
    """Run the full recovar pipeline on a 256x256, 5000-image dataset.

    This is a GPU memory stress test: 256-px images with 5k projections
    exercise the covariance estimation and forward-model batching under
    realistic memory pressure.  The test passes if the pipeline completes
    without OOM or other errors.
    """
    from conftest import gpu_subprocess_env

    output_dir = _resolve_output_dir(tmp_path, "gpu_stress_256_5k")
    dataset_dir = output_dir / "dataset"

    # 1. Generate synthetic dataset: 256x256, 5000 images
    create_cmd = [
        sys.executable, "-m", "recovar.commands.make_test_dataset",
        str(dataset_dir),
        "--image-size", "256",
        "--n-images", "5000",
        "--noise-level", "0.5",
        "--seed", "42",
    ]
    subprocess.run(create_cmd, check=True, env=gpu_subprocess_env())

    # Verify dataset was created
    test_dataset = dataset_dir / "test_dataset"
    particles = test_dataset / "particles.256.mrcs"
    assert particles.exists(), f"particles file not created at {particles}"

    # 2. Run full pipeline on GPU
    pipeline_output = test_dataset / "pipeline_stress_output"
    pipeline_cmd = [
        sys.executable, "-m", "recovar.command_line", "pipeline",
        str(particles),
        "--ctf", str(test_dataset / "ctf.pkl"),
        "--poses", str(test_dataset / "poses.pkl"),
        "--mask", "from_halfmaps",
        "-o", str(pipeline_output),
        "--zdim", "10",
        "--lazy",
    ]
    subprocess.run(pipeline_cmd, check=True, env=gpu_subprocess_env())

    # 3. Verify pipeline outputs exist
    assert pipeline_output.exists(), "pipeline output directory not created"
    # Check for key output files (model directory with results)
    model_dir = pipeline_output / "model"
    assert model_dir.exists(), f"model directory not created at {model_dir}"
