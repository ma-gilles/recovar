import os
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest


pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.io]


def _make_real_volume(idx, n_vols, grid):
    x = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    t = 2.0 * np.pi * idx / max(n_vols, 1)
    vol = (
        np.exp(-((xx - 0.3 * np.cos(t)) ** 2 + (yy - 0.25 * np.sin(t)) ** 2 + (zz - 0.2 * np.cos(2 * t)) ** 2) / (2 * 0.18**2))
        + 0.7
        * np.exp(-((xx + 0.25 * np.sin(1.3 * t)) ** 2 + (yy - 0.2 * np.cos(1.1 * t)) ** 2 + (zz + 0.2 * np.sin(t)) ** 2) / (2 * 0.16**2))
    )
    vol = vol.astype(np.float32)
    vol -= vol.mean()
    denom = np.linalg.norm(vol.ravel())
    if denom > 0:
        vol /= denom
    return vol


def _write_volumes(prefix: Path, n_vols=12, grid=32, voxel_size=4.25):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for i in range(n_vols):
        with mrcfile.new(f"{prefix}{i:04d}.mrc", overwrite=True) as m:
            m.set_data(_make_real_volume(i, n_vols, grid))
            m.voxel_size = voxel_size


def test_run_test_all_metrics_tiny_integration(tmp_path):
    if os.environ.get("RUN_TINY_METRICS_INTEGRATION", "0") != "1":
        pytest.skip("set RUN_TINY_METRICS_INTEGRATION=1 to run")

    vols_prefix = tmp_path / "vol"
    _write_volumes(vols_prefix, n_vols=12, grid=32, voxel_size=4.25)
    out_dir = tmp_path / "run_out"

    cmd = [
        sys.executable,
        "-m",
        "recovar.commands.run_test_all_metrics",
        "--volume-input",
        str(vols_prefix),
        "--output-dir",
        str(out_dir),
        "--grid-size",
        "32",
        "--n-images",
        "800",
        "--noise-level",
        "1.0",
        "--contrast-std",
        "0.1",
    ]
    subprocess.run(cmd, check=True)

    scores_json = out_dir / "test_dataset" / "metrics_plot" / "all_scores.json"
    assert scores_json.exists()
