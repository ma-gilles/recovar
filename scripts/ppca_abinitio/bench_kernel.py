"""Microbenchmark for the half-image PPCA posterior kernel.

Measures the JIT-compiled `score_from_half_image_projections` kernel
across a range of (image_size, n_rot, n_img, q) and prints a table.

This is a *characterization* tool, not a regression baseline. We do
NOT pin a numerical timing in the test suite — perf is platform- and
GPU-dependent. Instead, this script feeds rough numbers into the v0
PR description so reviewers can sanity-check the kernel scales as
expected.

Usage:
    pixi run python scripts/ppca_abinitio/bench_kernel.py
    pixi run python scripts/ppca_abinitio/bench_kernel.py --quick
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.posterior import (
    make_half_image_weights,
    score_from_half_image_projections,
)


def _real_to_full(rng, image_shape):
    real = rng.standard_normal(image_shape).astype(np.float64)
    return jnp.asarray(ftu.get_dft2(jnp.asarray(real)).reshape(-1), dtype=jnp.complex128)


def make_inputs(rng, n_rot, q, n_img, image_shape):
    H, W = image_shape
    n_full = H * W
    n_half = H * (W // 2 + 1)
    mean_full = jnp.stack([_real_to_full(rng, image_shape) for _ in range(n_rot)])
    u_full = jnp.stack([jnp.stack([_real_to_full(rng, image_shape) * 0.1 for _ in range(q)]) for _ in range(n_rot)])
    s = jnp.asarray(0.5 + rng.uniform(size=q), dtype=jnp.float64)
    batch_full = jnp.stack([_real_to_full(rng, image_shape) for _ in range(n_img)])

    weights_half = make_half_image_weights(image_shape)
    mean_half = ftu.full_image_to_half_image(mean_full, image_shape)
    u_half = ftu.full_image_to_half_image(u_full.reshape(n_rot * q, n_full), image_shape).reshape(n_rot, q, n_half)
    batch_half = ftu.full_image_to_half_image(batch_full, image_shape)

    ctf2_over_nv = jnp.ones((n_img, n_half), dtype=jnp.float64)
    shifted_half = batch_half[:, None, :].astype(jnp.complex128)  # n_trans=1

    return dict(
        mean_proj_half=mean_half.astype(jnp.complex128),
        u_proj_half=u_half.astype(jnp.complex128),
        s=s,
        shifted_half=shifted_half,
        ctf2_over_nv_half=ctf2_over_nv,
        weights_half=weights_half,
    )


def time_kernel(inputs, n_warmup=2, n_iters=10):
    fn = jax.jit(lambda *args, **kw: score_from_half_image_projections(*args, **kw))
    for _ in range(n_warmup):
        out = fn(**inputs)
        jax.block_until_ready(out.log_scores)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = fn(**inputs)
        jax.block_until_ready(out.log_scores)
    return (time.perf_counter() - t0) / n_iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="run a small subset for fast iteration")
    ap.add_argument("--n-iters", type=int, default=10)
    args = ap.parse_args()

    print(f"jax devices: {jax.devices()}")
    print()

    if args.quick:
        configs = [
            ("16x16,  q=2, n_rot=64,  n_img=64", (16, 16), 64, 64, 2),
            ("32x32,  q=4, n_rot=192, n_img=128", (32, 32), 192, 128, 4),
        ]
    else:
        configs = [
            ("16x16,  q=2,  n_rot=64,  n_img=64", (16, 16), 64, 64, 2),
            ("16x16,  q=4,  n_rot=192, n_img=128", (16, 16), 192, 128, 4),
            ("32x32,  q=2,  n_rot=64,  n_img=64", (32, 32), 64, 64, 2),
            ("32x32,  q=4,  n_rot=192, n_img=128", (32, 32), 192, 128, 4),
            ("32x32,  q=8,  n_rot=192, n_img=128", (32, 32), 192, 128, 8),
            ("64x64,  q=4,  n_rot=192, n_img=128", (64, 64), 192, 128, 4),
            ("64x64,  q=4,  n_rot=768, n_img=128", (64, 64), 768, 128, 4),
        ]

    rng = np.random.default_rng(0)
    print(f"{'config':<48s} {'mean (ms)':>12s} {'GFLOPs/s*':>14s}")
    print("-" * 80)
    for label, image_shape, n_rot, n_img, q in configs:
        inputs = make_inputs(rng, n_rot, q, n_img, image_shape)
        try:
            mean_t = time_kernel(inputs, n_iters=args.n_iters)
        except Exception as e:
            print(f"{label:<48s} FAILED: {type(e).__name__}: {e}")
            continue
        # Rough FLOP estimate (q^2 GEMM dominant): 8 * n_img * n_rot * q^2 * n_half
        # plus 4 * n_img * n_rot * q * n_half (b construction)
        n_half = image_shape[0] * (image_shape[1] // 2 + 1)
        flops = (8 * n_img * n_rot * q * q + 4 * n_img * n_rot * q) * n_half
        gflops_per_sec = flops / mean_t / 1e9
        print(f"{label:<48s} {mean_t * 1000:>12.2f} {gflops_per_sec:>14.1f}")
    print()
    print("* GFLOPs/s estimate counts only the dominant pixel-axis contractions; it ignores")
    print("  the q×q Cholesky and the constant-overhead bookkeeping. Treat as a rough scaling")
    print("  indicator, not a hardware utilization number.")


if __name__ == "__main__":
    main()
