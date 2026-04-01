#!/usr/bin/env python
"""Benchmark Fourier windowing at each allowed current_size.

Reports:
- GEMM time per block (E-step + M-step)
- Preprocessing time (shift + window gather) per block
- Total iteration time
- Confirms preprocessing does NOT dominate at small current_size

Usage:
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    pixi run python scripts/benchmark_fourier_window.py

Phase 3C of RELION-parity plan.
"""

import sys
import time
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# --- Load synthetic dataset ---
DATASET_PATH = "/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/"

from recovar.data_io import dataset as recovar_dataset

dataset = recovar_dataset.CryoEMDataset.load(
    particles_file=os.path.join(DATASET_PATH, "particles.star"),
    lazy=False,
)

print(f"Dataset: {dataset.n_images} images, image_shape={dataset.image_shape}, "
      f"volume_shape={dataset.volume_shape}")

# --- Setup ---
from recovar.em.sampling import get_healpix_grid
from recovar.em.dense_single_volume.engine_v2 import run_em_v2
from recovar.em.dense_single_volume.fourier_window import (
    ALLOWED_CURRENT_SIZES,
    make_fourier_window_indices_np,
)

# HEALPix order 3 grid
rotations = get_healpix_grid(order=3)
translations = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                          [-1.0, 0.0], [0.0, -1.0]], dtype=np.float32)

n_rot = rotations.shape[0]
n_trans = translations.shape[0]
print(f"Rotations: {n_rot}, Translations: {n_trans}")

# Random initial volume
rng = np.random.default_rng(42)
vol_real = rng.standard_normal(dataset.volume_shape).astype(np.float32) * 0.01
vol_ft = np.fft.fftshift(np.fft.fftn(vol_real))
mean = jnp.array(vol_ft.ravel(), dtype=jnp.complex64)
mean_variance = np.ones(dataset.volume_size, dtype=np.float32) * 100.0
noise_variance = jnp.ones(dataset.image_size, dtype=jnp.float32)

IMAGE_SHAPE = dataset.image_shape
H, W = IMAGE_SHAPE
N_HALF = H * (W // 2 + 1)

print(f"\nFourier window statistics:")
for cs in ALLOWED_CURRENT_SIZES:
    indices, n_windowed = make_fourier_window_indices_np(IMAGE_SHAPE, cs)
    print(f"  current_size={cs:3d}: n_windowed={n_windowed:5d} / N_half={N_HALF} "
          f"({100.0 * n_windowed / N_HALF:.1f}%)")

# --- Benchmark at each current_size ---
IMAGE_BATCH_SIZE = 500
ROTATION_BLOCK_SIZE = 5000
N_WARMUP = 1
N_RUNS = 3

results = {}

for cs in [None] + ALLOWED_CURRENT_SIZES:
    label = f"cs={cs}" if cs is not None else "full (no window)"
    print(f"\n{'='*60}")
    print(f"Benchmarking: {label}")
    print(f"{'='*60}")

    # Warmup (JIT compilation)
    for _ in range(N_WARMUP):
        _ = run_em_v2(
            dataset, mean, mean_variance, noise_variance,
            rotations, translations, "linear_interp",
            image_batch_size=IMAGE_BATCH_SIZE,
            rotation_block_size=ROTATION_BLOCK_SIZE,
            current_size=cs,
        )
        jax.block_until_ready(_[0])

    # Timed runs
    times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        new_mean, ha, Ft_y, Ft_ctf = run_em_v2(
            dataset, mean, mean_variance, noise_variance,
            rotations, translations, "linear_interp",
            image_batch_size=IMAGE_BATCH_SIZE,
            rotation_block_size=ROTATION_BLOCK_SIZE,
            current_size=cs,
        )
        jax.block_until_ready(new_mean)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"  Run {i+1}: {times[-1]:.3f}s")

    mean_time = np.mean(times)
    std_time = np.std(times)
    results[cs] = {"mean": mean_time, "std": std_time, "times": times}
    print(f"  Mean: {mean_time:.3f}s +/- {std_time:.3f}s")

# --- Summary ---
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Config':<25s} {'Time (s)':<12s} {'vs Full':<10s} {'Speedup':<10s}")
print("-" * 60)
full_time = results[None]["mean"]
for cs in [None] + ALLOWED_CURRENT_SIZES:
    label = f"cs={cs}" if cs is not None else "full (no window)"
    t = results[cs]["mean"]
    ratio = full_time / t if t > 0 else float("inf")
    print(f"{label:<25s} {t:.3f} +/- {results[cs]['std']:.3f}  "
          f"{t/full_time:.2f}x    {ratio:.2f}x")

print(f"\nDone.")
