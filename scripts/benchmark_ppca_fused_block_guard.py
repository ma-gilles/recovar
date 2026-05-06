#!/usr/bin/env python
"""Microbenchmark the dense PPCA fused block used by merge-regression checks.

This is intentionally synthetic and deterministic. It is not a quality test and
does not assert a wall-clock threshold, because login node/GPU load varies. Run
it before and after branch merges and compare the JSON fields:

  - median_elapsed_s / min_elapsed_s for performance drift
  - rhs_abs_checksum / lhs_abs_checksum for accidental numerical drift
  - pmax_mean / logZ_mean for score-path drift
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.em.ppca_refinement.dense_engine import fused_dense_pose_ppca_block
from recovar.em.sampling import get_rotation_grid_at_order


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def _complex_normal(rng, shape, scale):
    return (scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))).astype(np.complex64)


def _make_inputs(args):
    rng = np.random.default_rng(int(args.seed))
    image_shape = (int(args.size), int(args.size))
    volume_shape = (int(args.size), int(args.size), int(args.size))
    n_freq = image_shape[0] * (image_shape[1] // 2 + 1)
    n_half_vol = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    rotations = np.asarray(get_rotation_grid_at_order(0, n_in_planes=max(int(args.rotations), 3)), dtype=np.float32)
    rotations = rotations[: int(args.rotations)]
    Y1 = _complex_normal(rng, (int(args.images), int(args.translations), n_freq), 0.25)
    proj_aug = _complex_normal(rng, (int(args.rotations), int(args.q) + 1, n_freq), 0.2)
    if int(args.q):
        proj_aug[:, 1:, :] *= np.asarray(0.2, dtype=np.float32)
    ctf2_over_noise = rng.uniform(0.5, 2.0, size=(int(args.images), n_freq)).astype(np.float32)
    y_norm = rng.uniform(1.0, 3.0, size=(int(args.images),)).astype(np.float32)
    rhs0 = jnp.zeros((int(args.q) + 1, n_half_vol), dtype=jnp.complex64)
    lhs0 = jnp.zeros(((int(args.q) + 1) * (int(args.q) + 2) // 2, n_half_vol), dtype=jnp.float32)
    return (
        jnp.asarray(Y1),
        jnp.asarray(proj_aug),
        jnp.asarray(ctf2_over_noise),
        jnp.asarray(y_norm),
        jnp.asarray(rotations),
        image_shape,
        volume_shape,
        rhs0,
        lhs0,
    )


def run_benchmark(args):
    inputs = _make_inputs(args)

    def run_once():
        rhs, lhs, diag = fused_dense_pose_ppca_block(*inputs)
        jax.block_until_ready(rhs)
        jax.block_until_ready(lhs)
        jax.block_until_ready(diag.logZ)
        return rhs, lhs, diag

    for _ in range(int(args.warmups)):
        run_once()

    elapsed = []
    last = None
    for _ in range(int(args.repeats)):
        start = time.perf_counter()
        last = run_once()
        elapsed.append(time.perf_counter() - start)

    rhs, lhs, diag = last
    elapsed = np.asarray(elapsed, dtype=np.float64)
    result = {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "jax_devices": [str(device) for device in jax.devices()],
        "params": {
            "seed": int(args.seed),
            "size": int(args.size),
            "images": int(args.images),
            "translations": int(args.translations),
            "rotations": int(args.rotations),
            "q": int(args.q),
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
        },
        "elapsed_s": elapsed.tolist(),
        "median_elapsed_s": float(np.median(elapsed)),
        "min_elapsed_s": float(np.min(elapsed)),
        "mean_elapsed_s": float(np.mean(elapsed)),
        "rhs_abs_checksum": float(jnp.sum(jnp.abs(rhs))),
        "lhs_abs_checksum": float(jnp.sum(jnp.abs(lhs))),
        "pmax_mean": float(jnp.mean(diag.pmax)),
        "logZ_mean": float(jnp.mean(diag.logZ)),
    }
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--translations", type=int, default=5)
    parser.add_argument("--rotations", type=int, default=12)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main():
    args = _parse_args()
    result = run_benchmark(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
