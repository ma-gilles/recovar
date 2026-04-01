"""Test that contrast estimation works correctly when images need sign inversion.

Regression test for a bug where ``_check_uninvert_data`` set ``backend.mult``
but ``process_images`` read ``backend.data_multiplier`` — a stale copy that
was never updated.  This caused the forward model to have the wrong sign
relative to the images, pegging estimated contrasts at the grid minimum.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.gpu
def test_uninvert_contrast(tmp_path):
    """Flipped images should yield same contrasts as original after auto-uninvert."""
    grid_size = 64
    n_images = 500

    # --- Generate synthetic dataset ---
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "recovar.command_line",
            "make_test_dataset",
            str(tmp_path),
            "--n-images",
            str(n_images),
            "--image-size",
            str(grid_size),
            "--seed",
            "42",
        ],
    )
    ds_dir = tmp_path / "test_dataset"
    star = ds_dir / "particles.star"
    mrcs = ds_dir / f"particles.{grid_size}.mrcs"

    # --- Create sign-flipped copy ---
    flip_dir = tmp_path / "flipped"
    flip_dir.mkdir()
    with mrcfile.open(str(mrcs), mode="r") as f:
        data = -np.array(f.data)
        vs = float(f.voxel_size.x) if f.voxel_size.x > 0 else 1.0
    flip_mrcs = flip_dir / mrcs.name
    with mrcfile.new(str(flip_mrcs), overwrite=True) as f:
        f.set_data(data.astype(np.float32))
        f.voxel_size = vs
    shutil.copy(str(star), str(flip_dir / "particles.star"))
    for fn in ("poses.pkl", "ctf.pkl"):
        src = ds_dir / fn
        if src.exists():
            shutil.copy(str(src), str(flip_dir / fn))

    # --- Run pipeline on both ---
    def run_pipeline(label, star_path, datadir, outdir):
        cmd = [
            sys.executable,
            "-m",
            "recovar.command_line",
            "pipeline",
            str(star_path),
            "-o",
            str(outdir),
            "--mask",
            "sphere",
            "--correct-contrast",
            "--no-do-over-with-contrast",
            "--no-downsample",
            "--zdim",
            "2",
            "--datadir",
            str(datadir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"[{label}] pipeline failed:\n{result.stderr[-500:]}"
        c = np.load(str(outdir / "model" / "zdim_2" / "contrasts.npy"))
        return c[~np.isnan(c)]

    c_orig = run_pipeline("ORIG", star, ds_dir, tmp_path / "out_orig")
    c_flip = run_pipeline("FLIP", flip_dir / "particles.star", flip_dir, tmp_path / "out_flip")

    # Both should have contrasts near 1.0
    assert 0.7 < c_orig.mean() < 1.3, f"Original contrasts wrong: mean={c_orig.mean():.4f}"
    assert 0.7 < c_flip.mean() < 1.3, f"Flipped contrasts wrong: mean={c_flip.mean():.4f}"

    # They should be very close to each other
    assert abs(c_orig.mean() - c_flip.mean()) < 0.05, (
        f"Contrast mismatch: orig={c_orig.mean():.4f} vs flip={c_flip.mean():.4f}"
    )
