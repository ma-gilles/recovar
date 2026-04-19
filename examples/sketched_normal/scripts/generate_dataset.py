"""Generate one simulated dataset from a CryoBench volume set.

Usage:
  pixi run python generate_dataset.py \
      --cryobench-name Ribosembly \
      --grid-size 64 --n-images 20000 --noise-level 1.0 \
      --output-dir /scratch/.../datasets/Ribosembly_g64_n20k_nl1

Re-uses recovar.ppca.ppca_scale_sweep.generate_dataset under the hood.
If the output dir already contains simulation_info.pkl and the particles
file, this is a no-op (cache-safe).
"""

import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from recovar.ppca.ppca_scale_sweep import find_cryobench_datasets, generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cryobench-base", default="/home/mg6942/mytigress/cryobench2")
    ap.add_argument("--cryobench-name", required=True)
    ap.add_argument("--grid-size", type=int, default=64)
    ap.add_argument("--n-images", type=int, default=20000)
    ap.add_argument("--noise-level", type=float, default=1.0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    marker = os.path.join(args.output_dir, "simulation_info.pkl")
    particles = os.path.join(args.output_dir, f"particles.{args.grid_size}.mrcs")
    if os.path.exists(marker) and os.path.exists(particles):
        print(f"[skip] already generated: {args.output_dir}", flush=True)
        return

    t0 = time.time()
    datasets = find_cryobench_datasets(args.cryobench_base)
    match = [d for d in datasets if d["name"] == args.cryobench_name]
    if not match:
        raise SystemExit(
            f"no cryobench dataset {args.cryobench_name!r} at {args.cryobench_base}; "
            f"available: {[d['name'] for d in datasets]}"
        )
    info = match[0]
    print(
        f"[gen] {args.cryobench_name}  grid={args.grid_size}  n={args.n_images}  nl={args.noise_level}  -> {args.output_dir}",
        flush=True,
    )
    generate_dataset(info, args.grid_size, args.n_images, args.noise_level, args.output_dir, seed=args.seed)
    print(f"[gen-done] {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
