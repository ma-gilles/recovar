"""F10: GPU speed benchmark for VDAM InitialModel iteration.

Measures wall-clock per-iter time for the run_em pass at pad=1 with
GUI-InitialModel sampling (HEALPix order 1, translation 6/2, current
size 28) on the RELION fixture. Compares against RELION's reported
~2-3 min per iter for 500 particles on CPU (see fixture stdout).

Target: ≤ 60 s per iter on H100. First iteration includes JIT compile
(~30-60 s); subsequent iters are the measurement.

Usage:
  sbatch scripts/run_vdam_f10_benchmark.sh
  # or inline (requires GPU):
  pixi run python scripts/run_vdam_f10_benchmark.py --n-iter 5
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.em.dense_single_volume.em_engine import run_em
from recovar.em.sampling import get_rotation_grid, get_translation_grid
from recovar.reconstruction.noise import make_radial_noise
from recovar.utils.helpers import load_relion_volume

FIXTURE_DIR = Path("/scratch/gpfs/GILLES/mg6942/tmp/relion_initialmodel_64_20260420_121428_8956_run")
PARTICLES_STAR = Path(
    "/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.tmp/"
    "slurm_7178672/pytest-of-mg6942/pytest-0/test_pipeline_spa_gpu0/"
    "gpu_spa/test_dataset/particles.star"
)


def _read_iter0_sigma2(n_shells: int) -> np.ndarray:
    txt = (FIXTURE_DIR / "run_it000_model.star").read_text()
    m = re.search(r"data_model_optics_group_1\n(.*?)(?:\ndata_)", txt, re.DOTALL)
    values = np.zeros(n_shells, dtype=np.float64)
    for line in m.group(1).strip().split("\n"):
        toks = line.split()
        if len(toks) == 3:
            try:
                values[int(toks[0])] = float(toks[2])
            except ValueError:
                continue
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-iter", type=int, default=5, help="Number of iterations to time (first one includes JIT)")
    p.add_argument("--healpix-order", type=int, default=1)
    p.add_argument("--n-psi", type=int, default=12)
    p.add_argument("--offset-range", type=int, default=6)
    p.add_argument("--offset-step", type=int, default=2)
    args = p.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")

    ds = load_dataset(str(PARTICLES_STAR), lazy=False)
    ori = int(ds.grid_size)
    px = float(ds.voxel_size)

    iter0 = np.asarray(load_relion_volume(str(FIXTURE_DIR / "run_it000_class001.mrc")), dtype=np.float64)
    iter0_ft = np.asarray(ftu.get_dft3(jnp.asarray(iter0))).reshape(-1)

    sigma2 = _read_iter0_sigma2(ori // 2 + 1)
    nv = np.asarray(make_radial_noise(sigma2 * ori**4, (ori, ori))).astype(np.float32).reshape(-1)
    rots = get_rotation_grid(args.healpix_order, args.n_psi, matrices=True).astype(np.float32)
    trans = get_translation_grid(args.offset_range, args.offset_step).astype(np.float32)
    print(f"n_rot = {rots.shape[0]}, n_trans = {trans.shape[0]}")

    mv = jnp.asarray((np.abs(iter0_ft) ** 2).astype(np.float32))
    mean_j = jnp.asarray(iter0_ft, dtype=jnp.complex64)

    wall_times = []
    for it in range(args.n_iter):
        jax.block_until_ready(mean_j)
        t0 = time.time()
        result = run_em(
            ds,
            mean=mean_j,
            mean_variance=mv,
            noise_variance=jnp.asarray(nv),
            rotations=rots,
            translations=jnp.asarray(trans),
            disc_type="linear_interp",
            image_batch_size=250,
            rotation_block_size=200,
            current_size=28,
            projection_padding_factor=1,
            reconstruction_padding_factor=1,
            half_spectrum_scoring=True,
            return_stats=True,
        )
        jax.block_until_ready(result[0])
        wall = time.time() - t0
        wall_times.append(wall)
        label = "  [JIT+first]" if it == 0 else ""
        print(f"iter {it}: wall = {wall:.2f}s{label}")

    if len(wall_times) > 1:
        steady = wall_times[1:]
        median = float(np.median(steady))
        print(f"\nSteady-state median = {median:.2f}s / iter")
        relion_reported = 90.0  # RELION reported ~90s/iter on CPU for 500 particles
        print(f"RELION reference:     ~{relion_reported:.0f}s / iter (CPU, --j 4)")
        print(f"Speedup:              {relion_reported / median:.1f}x")
        print(f"F10 target (≤60s):    {'PASS' if median <= 60 else 'FAIL'}")


if __name__ == "__main__":
    main()
