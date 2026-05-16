"""F9: full 200-iter VDAM run vs RELION InitialModel fixture.

Outputs:
  - final volume at --output-mrc
  - CC vs initial_model.mrc (after align_symmetry)
  - FSC vs ground-truth initial_model.mrc per-shell

Usage:
  pixi run python scripts/run_vdam_f9_fixture.py \\
      --nr-iter 200 --blend-step 0.5 \\
      --output-mrc out.mrc
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import jax.numpy as jnp
import mrcfile
import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.em.dense_single_volume.em_engine import run_em
from recovar.em.initial_model.schedules import (
    compute_phase_lengths,
    compute_stepsize,
)
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


def _cc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel() - a.mean()
    bf = b.ravel() - b.mean()
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))


def _radial_fsc(a: np.ndarray, b: np.ndarray, n_shells: int) -> np.ndarray:
    """Per-shell FSC between two real-space volumes."""
    N = a.shape[0]
    Fa = np.fft.rfftn(a)
    Fb = np.fft.rfftn(b)
    kz = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    kx = np.arange(N // 2 + 1, dtype=np.float64)
    ires = np.round(np.sqrt(kz[:, None, None] ** 2 + ky[None, :, None] ** 2 + kx[None, None, :] ** 2)).astype(int)
    num = np.zeros(n_shells, dtype=np.float64)
    da = np.zeros(n_shells, dtype=np.float64)
    db = np.zeros(n_shells, dtype=np.float64)
    for i in range(n_shells):
        mask = ires == i
        if mask.sum() == 0:
            continue
        Ai = Fa[mask]
        Bi = Fb[mask]
        num[i] = np.real(np.sum(np.conj(Ai) * Bi))
        da[i] = np.sum(np.abs(Ai) ** 2)
        db[i] = np.sum(np.abs(Bi) ** 2)
    return num / np.sqrt(np.maximum(da * db, 1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nr-iter", type=int, default=200)
    p.add_argument("--blend-step", type=float, default=0.5)
    p.add_argument("--output-mrc", type=str, required=True)
    args = p.parse_args()

    t0 = time.time()
    ds = load_dataset(str(PARTICLES_STAR), lazy=False)
    ori = int(ds.grid_size)
    px = float(ds.voxel_size)

    iter0 = np.asarray(load_relion_volume(str(FIXTURE_DIR / "run_it000_class001.mrc")), dtype=np.float64)
    final_target = np.asarray(load_relion_volume(str(FIXTURE_DIR / "initial_model.mrc")))
    sigma2 = _read_iter0_sigma2(ori // 2 + 1)
    nv = np.asarray(make_radial_noise(sigma2 * ori**4, (ori, ori))).astype(np.float32).reshape(-1)
    rots = get_rotation_grid(1, 12, matrices=True).astype(np.float32)
    trans = get_translation_grid(6, 2).astype(np.float32)
    phases = compute_phase_lengths(args.nr_iter, 0.3, 0.2)

    current = iter0.copy()
    print(f"Starting F9: nr_iter={args.nr_iter} blend_step={args.blend_step} ori={ori} px={px}")

    for it in range(1, args.nr_iter + 1):
        current_ft = np.asarray(ftu.get_dft3(jnp.asarray(current))).reshape(-1)
        mv = jnp.asarray((np.abs(current_ft) ** 2).astype(np.float32))
        result = run_em(
            ds,
            mean=jnp.asarray(current_ft, dtype=jnp.complex64),
            mean_variance=mv,
            noise_variance=jnp.asarray(nv),
            rotations=rots,
            translations=jnp.asarray(trans),
            disc_type="linear_interp",
            image_batch_size=50,
            rotation_block_size=100,
            current_size=28,
            projection_padding_factor=1,
            reconstruction_padding_factor=1,
            half_spectrum_scoring=True,
            return_stats=True,
        )
        new_mean = np.asarray(result[0]).reshape(ori, ori, ori)
        new_vol = np.asarray(ftu.get_idft3(jnp.asarray(new_mean))).real
        step = compute_stepsize(iter=it, phase_lengths=phases, is_3d_model=True, ref_dim=3)
        # Use VDAM schedule step for amplitude but clamp blend into
        # [blend_step_arg, 0.9] to retain F8's real-space match
        blend = max(args.blend_step, 1.0 - step + 0.3)
        current = (1 - blend) * current + blend * new_vol
        if it % 10 == 0 or it in (1, 2, 5):
            cc_iter = _cc(current, final_target)
            print(
                f"iter {it:3d}: VDAM step={step:.3f} blend={blend:.3f} "
                f"CC(final)={cc_iter:+.4f} std={current.std():.4e} "
                f"elapsed={time.time() - t0:.1f}s"
            )
        assert np.all(np.isfinite(current)), f"iter {it} diverged"

    total = time.time() - t0
    final_cc = _cc(current, final_target)
    fsc = _radial_fsc(current, final_target, ori // 2 + 1)

    print(f"\n=== F9 FINAL ({args.nr_iter} iters, {total:.1f}s wall) ===")
    print(f"  CC(final, initial_model) = {final_cc:+.4f}")
    print(f"  |CC|                     = {abs(final_cc):.4f}")
    print("  Per-shell FSC (head):")
    for i, f in enumerate(fsc[:15]):
        print(f"    shell {i:2d}: {f:+.4f}")
    print(f"  FSC @ 0.5 shell: {np.searchsorted(-fsc, -0.5)}/{ori // 2}")

    Path(args.output_mrc).parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(args.output_mrc, overwrite=True) as m:
        m.set_data(current.astype(np.float32))
        m.voxel_size = (px, px, px)
    print(f"  Wrote {args.output_mrc}")


if __name__ == "__main__":
    main()
