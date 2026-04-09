"""Larger-scale ppca_abinitio validation at vol 16 / 24 / 32.

All v0 unit tests run at toy 8³. This script exercises the full
fixed-grid PPCA loop at non-toy volume sizes to characterize:

  - Whether the iteration is stable (no NaN, no blow-up)
  - How FRE_mu evolves vs the toy 8³ regime
  - Wall time and memory at each size
  - Whether the loop produces sensible posterior summaries

It is **not** a regression test. There are no pinned numerics. The
output JSON is consumed by the v0 PR description as a scaling table.

Usage:
    pixi run python scripts/ppca_abinitio/run_scale_validation.py \\
        --output /scratch/gpfs/GILLES/mg6942/slurmo/ppca_scale.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import make_half_volume_weights
from recovar.em.ppca_abinitio.init import init_oracle, init_truth_perturbed
from recovar.em.ppca_abinitio.loop import run_fixed_grid_ppca
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)
from recovar.em.ppca_abinitio.types import PPCAConfig


def _identity_ctf(CTF_params, image_shape, voxel_size):
    n = CTF_params.shape[0]
    sz = int(np.prod(image_shape))
    return jnp.ones((n, sz), dtype=jnp.float64)


def _identity_process(b, apply_image_mask=False):
    return b


class _SynthCfg(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, p, *, half_image=False):
        full = _identity_ctf(p, self.image_shape, self.voxel_size)
        if half_image:
            return ftu.full_image_to_half_image(full, self.image_shape)
        return full

    def process_fn(self, b, apply_image_mask=False):
        return _identity_process(b, apply_image_mask=apply_image_mask)


@dataclass
class ScaleResult:
    volume_size: int
    n_images: int
    n_rot: int
    init_kind: str
    n_iters: int
    fre_init: float
    fre_final: float
    fre_traj: list
    wall_seconds: float
    peak_jax_mem_gb: float


def _peak_mem_gb():
    """Best-effort peak GPU memory across all visible devices."""
    try:
        out = 0.0
        for d in jax.devices():
            stats = d.memory_stats() if hasattr(d, "memory_stats") else None
            if stats and "peak_bytes_in_use" in stats:
                out = max(out, stats["peak_bytes_in_use"] / 1e9)
        return out
    except Exception:
        return -1.0


def _run_one(volume_size: int, n_images: int, init_kind: str, n_iters: int) -> ScaleResult:
    vs = volume_size
    image_shape = (vs, vs)
    volume_shape = (vs, vs, vs)
    grid = build_fixed_grid(healpix_order=1, max_shift=1)

    print(f"  building dataset vol={vs}, n_images={n_images}, n_rot={int(grid.rotations.shape[0])} ...", flush=True)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=volume_shape,
        image_shape=image_shape,
        grid=grid,
        q=2,
        n_images_train=n_images,
        n_images_val=max(8, n_images // 8),
        sigma_real=0.1,
        seed=0,
    )
    cfg = _SynthCfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)

    if init_kind == "oracle":
        init = init_oracle(
            mu_half_true=ds.mu_half_true,
            U_half_true=ds.U_half_true,
            s_true=ds.s_true,
            volume_shape=volume_shape,
        )
    else:
        init = init_truth_perturbed(
            mu_half_true=ds.mu_half_true,
            U_half_true=ds.U_half_true,
            s_true=ds.s_true,
            volume_shape=volume_shape,
            eps_mu=0.3,
            eps_U=0.0,
            seed=0,
        )

    cfg_run = PPCAConfig(n_iters=n_iters, update_mu=True, update_factor=False, ridge_lambda=0.0)
    weights = make_half_volume_weights(volume_shape)

    # Warm JIT once (do not include in timing)
    print("  warming JIT ...", flush=True)
    _ = run_fixed_grid_ppca(
        cfg,
        ds,
        init,
        PPCAConfig(n_iters=1, update_mu=True, update_factor=False, ridge_lambda=0.0),
        weights_half=weights,
    )

    print("  timed run ...", flush=True)
    t0 = time.perf_counter()
    res = run_fixed_grid_ppca(cfg, ds, init, cfg_run, weights_half=weights)
    jax.block_until_ready(res.final_init.mu)
    wall = time.perf_counter() - t0

    fre_traj = [m.fre_mu_val for m in res.iter_metrics]
    return ScaleResult(
        volume_size=vs,
        n_images=n_images,
        n_rot=int(grid.rotations.shape[0]),
        init_kind=init_kind,
        n_iters=n_iters,
        fre_init=fre_traj[0],
        fre_final=fre_traj[-1],
        fre_traj=fre_traj,
        wall_seconds=wall,
        peak_jax_mem_gb=_peak_mem_gb(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--quick", action="store_true", help="run only the smallest config")
    args = ap.parse_args()

    if args.quick:
        configs = [(12, 256, "oracle", 4)]
    else:
        configs = [
            (12, 512, "oracle", 4),
            (16, 512, "oracle", 4),
            (16, 1024, "perturbed", 4),
            (24, 1024, "oracle", 3),
            (32, 1024, "oracle", 3),
        ]

    print(f"jax devices: {jax.devices()}")
    print(f"running {len(configs)} configurations")
    print()

    results = []
    for vs, ni, kind, n_iters in configs:
        print(f"== vol={vs}, n_img={ni}, init={kind}, n_iters={n_iters} ==", flush=True)
        try:
            r = _run_one(vs, ni, kind, n_iters)
            results.append(asdict(r))
            print(f"  fre_traj={r.fre_traj}")
            print(f"  wall={r.wall_seconds:.2f}s, peak_mem={r.peak_jax_mem_gb:.2f}GB", flush=True)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            results.append({"volume_size": vs, "n_images": ni, "init_kind": kind, "error": f"{type(e).__name__}: {e}"})
        gc.collect()
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
