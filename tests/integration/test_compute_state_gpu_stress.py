"""GPU memory stress test for compute_state at 256^3.

Exercises the bin-accumulation and kernel-regression path in
adaptive_kernel_discretization.py under realistic memory pressure.
Prior to the OOM fix, this would crash with a 25 GiB scatter allocation.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def test_compute_state_gpu_memory_stress_256(tmp_path):
    """Run compute_state on a 256^3, 5k-image pipeline output.

    Steps:
      1. Generate a synthetic 256x256, 5000-image dataset.
      2. Run the full pipeline (zdim=4) to produce embeddings.
      3. Generate 10 k-means latent points from the zdim=4 embedding.
      4. Run compute_state with n_bins=50 (default).
      5. Pass if no OOM or error.

    The critical path is the (n_bins x half_volume_size) accumulation in
    even_less_naive_heterogeneity_scheme_relion_style.  At 256^3 with
    n_bins=50, this is ~5 GB of arrays that previously caused OOM via
    JAX immutable-array copies on GPU.
    """
    from conftest import gpu_subprocess_env

    output_dir = _resolve_output_dir(tmp_path, "compute_state_stress_256")
    dataset_dir = output_dir / "dataset"

    env = gpu_subprocess_env()

    # 1. Generate synthetic dataset
    create_cmd = [
        sys.executable,
        "-m",
        "recovar.commands.make_test_dataset",
        str(dataset_dir),
        "--image-size",
        "256",
        "--n-images",
        "5000",
        "--noise-level",
        "0.5",
        "--seed",
        "42",
    ]
    subprocess.run(create_cmd, check=True, env=env, timeout=600)

    test_dataset = dataset_dir / "test_dataset"
    particles = test_dataset / "particles.256.mrcs"
    assert particles.exists(), f"particles file not created at {particles}"

    # 2. Run pipeline to get embeddings
    pipeline_output = test_dataset / "pipeline_output"
    pipeline_cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "pipeline",
        str(particles),
        "--ctf",
        str(test_dataset / "ctf.pkl"),
        "--poses",
        str(test_dataset / "poses.pkl"),
        "--mask",
        "from_halfmaps",
        "-o",
        str(pipeline_output),
        "--zdim",
        "4",
        "--lazy",
    ]
    subprocess.run(pipeline_cmd, check=True, env=env, timeout=3600)
    assert pipeline_output.exists(), "pipeline output not created"

    # 3. Generate k-means latent points from zdim=4 embedding
    centers_file = output_dir / "centers.txt"
    kmeans_script = f"""
import numpy as np
from recovar.output import output as out_mod
po = out_mod.PipelineOutput('{pipeline_output}')
zs = np.array(po.get('latent_coords')[4])
from sklearn.cluster import MiniBatchKMeans
km = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=min(5000, len(zs)))
km.fit(zs)
np.savetxt('{centers_file}', km.cluster_centers_)
"""
    subprocess.run(
        [sys.executable, "-c", kmeans_script],
        check=True,
        env=env,
        timeout=120,
    )
    assert centers_file.exists(), "k-means centers file not created"
    centers = np.loadtxt(str(centers_file))
    assert centers.shape == (10, 4), f"unexpected centers shape: {centers.shape}"

    # 4. Run compute_state — this is the OOM stress test
    state_output = output_dir / "state_output"
    state_cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "compute_state",
        str(pipeline_output),
        "-o",
        str(state_output),
        "--latent-points",
        str(centers_file),
        "--lazy",
        "--n-bins",
        "50",
    ]
    result = subprocess.run(
        state_cmd,
        check=True,
        env=env,
        timeout=3600,
        capture_output=True,
        text=True,
    )

    # 5. Verify outputs exist
    assert state_output.exists(), "compute_state output directory not created"
    # Check that volumes were written (one per latent point)
    mrc_files = list(state_output.glob("*.mrc"))
    assert len(mrc_files) > 0, (
        f"No MRC files in {state_output}. stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
