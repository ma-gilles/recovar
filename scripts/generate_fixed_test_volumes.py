#!/usr/bin/env python3
"""
Generate deterministic synthetic real-space volumes for run_test_all_metrics baselines.

Output files are written as:
  <output_prefix>0000.mrc, <output_prefix>0001.mrc, ...

Example:
  python scripts/generate_fixed_test_volumes.py \
    --output-prefix /scratch/gpfs/AMITS/mg6942/recovar_fixed_vols/vol \
    --n-volumes 50 \
    --grid-size 128 \
    --voxel-size 4.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mrcfile
import numpy as np


def _gaussian_3d(grid_x, grid_y, grid_z, center, sigma, amp):
    cx, cy, cz = center
    r2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2 + (grid_z - cz) ** 2
    return amp * np.exp(-r2 / (2.0 * sigma**2))


def make_volume(idx: int, n_volumes: int, grid_size: int) -> np.ndarray:
    # Normalized cube coordinates in [-1, 1].
    x = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")

    t = 2.0 * np.pi * (idx / max(n_volumes, 1))
    vol = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    # Slowly moving dominant lobe.
    center1 = (0.35 * np.cos(t), 0.30 * np.sin(t), 0.20 * np.cos(2.0 * t))
    vol += _gaussian_3d(xx, yy, zz, center1, sigma=0.20, amp=1.0)

    # Counter-moving secondary lobe.
    center2 = (-0.30 * np.sin(1.5 * t), 0.25 * np.cos(1.2 * t), -0.25 * np.sin(t))
    vol += _gaussian_3d(xx, yy, zz, center2, sigma=0.16, amp=0.8)

    # Weak static support component.
    vol += _gaussian_3d(xx, yy, zz, center=(0.0, 0.0, 0.0), sigma=0.35, amp=0.25)

    # Normalize to stable scale.
    vol -= np.mean(vol)
    denom = np.linalg.norm(vol.ravel())
    if denom > 0:
        vol /= denom
    return vol.astype(np.float32)


def write_mrc(path: Path, vol: np.ndarray, voxel_size: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(vol)
        m.voxel_size = voxel_size


def main() -> None:
    p = argparse.ArgumentParser(description="Generate fixed deterministic test volumes.")
    p.add_argument("--output-prefix", required=True, help="Prefix path for output MRCs, e.g. /path/to/vol")
    p.add_argument("--n-volumes", type=int, default=50)
    p.add_argument("--grid-size", type=int, default=128)
    p.add_argument("--voxel-size", type=float, default=4.25)
    args = p.parse_args()

    output_prefix = Path(args.output_prefix)
    for i in range(args.n_volumes):
        vol = make_volume(i, args.n_volumes, args.grid_size)
        out = Path(f"{output_prefix}{i:04d}.mrc")
        write_mrc(out, vol, args.voxel_size)

    print(f"Wrote {args.n_volumes} volumes to prefix: {output_prefix}")


if __name__ == "__main__":
    main()
