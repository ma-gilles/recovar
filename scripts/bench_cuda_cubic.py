#!/usr/bin/env python
"""Benchmark: CUDA cubic projection vs JAX cubic projection.

Measures wall time and peak GPU memory for forward projection at
various grid sizes (N=64, 128, 256) and image counts.

Usage:
    python scripts/bench_cuda_cubic.py
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

import recovar.core.slicing as slicing
from recovar.cuda_backproject import project as cuda_project


def _make_hermitian_volume(N, rng, dtype=np.complex64):
    real_vol = rng.standard_normal((N, N, N)).astype(np.float64)
    vol_ft = np.fft.fftshift(np.fft.fftn(real_vol))
    return jnp.array(vol_ft, dtype=dtype)


def _random_rotations(n, rng):
    return jnp.array(Rotation.random(n, random_state=rng).as_matrix().astype(np.float32))


def _gpu_memory_bytes():
    """Current GPU memory usage via JAX runtime."""
    try:
        dev = jax.devices("gpu")[0]
        stats = dev.memory_stats()
        return stats.get("peak_bytes_in_use", 0)
    except Exception:
        return 0


def bench_forward(func, *args, warmup=3, repeats=10):
    """Time a GPU function, returning median ms and peak GPU memory."""
    # Warmup
    for _ in range(warmup):
        out = func(*args)
        out.block_until_ready()

    # Reset peak memory tracking
    dev = jax.devices("gpu")[0]
    try:
        dev.clear_memory_stats()
    except Exception:
        pass

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = func(*args)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)

    peak_mb = _gpu_memory_bytes() / 1e6
    median_ms = np.median(times) * 1000
    return median_ms, peak_mb, out


def main():
    rng = np.random.default_rng(42)
    dev = jax.devices("gpu")[0]
    print(f"Device: {dev}")
    print()

    configs = [
        (64,   100),
        (64,   1000),
        (128,  100),
        (128,  1000),
        (256,  100),
        (256,  500),
    ]

    print(f"{'N':>5} {'n_img':>6} │ {'CUDA ms':>10} {'JAX ms':>10} {'Speedup':>8} │ {'CUDA MB':>10} {'JAX MB':>10}")
    print("─" * 75)

    for N, n_images in configs:
        volume_shape = (N, N, N)
        image_shape = (N, N)

        vol = _make_hermitian_volume(N, rng)
        rots = _random_rotations(n_images, rng)
        coeffs = slicing.precompute_cubic_coefficients(vol, volume_shape)

        with jax.default_device(dev):
            coeffs_g = jax.device_put(coeffs)
            rots_g = jax.device_put(rots)

            # CUDA cubic
            def cuda_fn(c, r):
                return cuda_project(c, r, image_shape, volume_shape,
                                    order=3, half_volume=False, half_image=False)

            cuda_ms, cuda_mb, cuda_out = bench_forward(cuda_fn, coeffs_g, rots_g)

            # JAX cubic (force disable CUDA)
            os.environ["RECOVAR_DISABLE_CUDA"] = "1"
            import recovar.cuda_backproject as cb
            cb._cuda_ok = None

            def jax_fn(c, r):
                return slicing.slice_from_cubic_coefficients(
                    c, r, image_shape, volume_shape, half_image=False)

            jax_ms, jax_mb, jax_out = bench_forward(jax_fn, coeffs_g, rots_g)

            # Re-enable CUDA
            os.environ.pop("RECOVAR_DISABLE_CUDA", None)
            cb._cuda_ok = None

            # Verify equivalence
            err = float(jnp.max(jnp.abs(cuda_out - jax_out)))
            rel = float(jnp.max(jnp.abs(cuda_out - jax_out) / (jnp.abs(jax_out) + 1e-10)))

        speedup = jax_ms / cuda_ms if cuda_ms > 0 else float('inf')
        print(f"{N:>5} {n_images:>6} │ {cuda_ms:>10.2f} {jax_ms:>10.2f} {speedup:>7.2f}x │ {cuda_mb:>10.1f} {jax_mb:>10.1f}   max_err={err:.2e} rel={rel:.2e}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
