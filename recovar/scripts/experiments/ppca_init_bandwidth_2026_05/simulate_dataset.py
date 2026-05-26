"""Run recovar's simulator to produce a synthetic dataset (particles.star + ctf.pkl + poses.pkl + simulation_info.pkl)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--vols-prefix", required=True, help="Simulator prefix (e.g. '/path/vol' → vol0000.mrc, vol0001.mrc, ...)"
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-images", type=int, default=20000)
    p.add_argument("--box", type=int, default=128)
    p.add_argument("--voxel", type=float, required=True)
    p.add_argument("--noise", type=float, required=True)
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument(
        "--workdir", type=str, default="/scratch/gpfs/GILLES/mg6942/recovar_wt_ppca_postmerge_followup_20260510_110827"
    )
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, args.workdir)
    import numpy as np

    from recovar.simulation import simulator

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "simulation_info.pkl").exists():
        print(f"SKIP: {args.out_dir}/simulation_info.pkl exists")
        return
    np.random.seed(args.seed)
    simulator.generate_synthetic_dataset(
        str(args.out_dir),
        args.voxel,
        args.vols_prefix,
        args.n_images,
        outlier_file_input=None,
        grid_size=args.box,
        volume_distribution=None,
        dataset_params_option="uniform",
        noise_level=args.noise,
        noise_model="radial1",
        relion_normalize=False,
        put_extra_particles=False,
        percent_outliers=0.0,
        volume_radius=0.7,
        trailing_zero_format_in_vol_name=True,
        noise_scale_std=0.0,
        contrast_std=0.0,
        disc_type="linear_interp",
    )
    print(f"sim done → {args.out_dir}")


if __name__ == "__main__":
    main()
