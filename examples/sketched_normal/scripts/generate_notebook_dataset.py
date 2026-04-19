"""Generate the synthetic 'notebook smoke' dataset used by the headline run.

Produces a trajectory of N_VOLUMES = 10 density maps at the requested grid,
then runs the image simulator with radial-noise model to create a
particles.<grid>.mrcs stack that run_experiment.py / run_ppca_experiment.py
can consume.

Cache-safe: if the volume prefix and particles file already exist, does
nothing.  Edit grid / n / noise-level args to regenerate a different preset.

Usage:
  pixi run python generate_notebook_dataset.py \\
      --grid-size 64 --n-images 5000 --noise-level 1e-5 \\
      --output-dir /scratch/.../sketched_repro
"""

import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from recovar.simulation import simulator
from recovar.simulation.trajectory_generation import generate_trajectory_volumes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-size", type=int, default=64)
    ap.add_argument("--n-images", type=int, default=5000)
    ap.add_argument("--noise-level", type=float, default=1e-5)
    ap.add_argument("--n-volumes", type=int, default=10)
    ap.add_argument("--output-dir", required=True, help="writable dir for vols + dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    voxel_size = 4.25 * 128 / args.grid_size
    vol_root = os.path.join(args.output_dir, f"vols_g{args.grid_size}")
    vol_prefix = os.path.join(vol_root, "vol")
    ds_dir = os.path.join(args.output_dir, f"dataset_g{args.grid_size}_n{args.n_images}_nl{args.noise_level}")

    t0 = time.time()
    if not os.path.isfile(f"{vol_prefix}0000.mrc"):
        os.makedirs(vol_root, exist_ok=True)
        print(
            f"[vols] gen grid={args.grid_size} n_vols={args.n_volumes} -> {vol_prefix}*",
            flush=True,
        )
        generate_trajectory_volumes(
            vol_root,
            grid_size=args.grid_size,
            n_volumes=args.n_volumes,
            voxel_size=voxel_size,
            Bfactor=80,
            max_rotation_degrees=10.0,
            output_prefix=vol_prefix,
        )
    else:
        print(f"[vols] reuse {vol_root}/", flush=True)

    particles = os.path.join(ds_dir, f"particles.{args.grid_size}.mrcs")
    if not os.path.isfile(particles):
        print(
            f"[ds]   gen n={args.n_images} nl={args.noise_level} -> {ds_dir}/",
            flush=True,
        )
        os.makedirs(ds_dir, exist_ok=True)
        np.random.seed(args.seed)
        simulator.generate_synthetic_dataset(
            ds_dir,
            voxel_size,
            vol_prefix,
            args.n_images,
            grid_size=args.grid_size,
            noise_level=args.noise_level,
            noise_model="radial1",
            contrast_std=0.0,
            noise_scale_std=0.0,
            dataset_params_option="uniform",
            disc_type="linear_interp",
            trailing_zero_format_in_vol_name=True,
            put_extra_particles=False,
            percent_outliers=0.0,
        )
    else:
        print(f"[ds]   reuse {ds_dir}", flush=True)

    print(f"[ok] dataset ready: {ds_dir}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
