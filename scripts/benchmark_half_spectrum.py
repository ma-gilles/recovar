#!/usr/bin/env python
"""Benchmark half-spectrum vs full-spectrum EM engine.

Usage (on login node GPU):
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    pixi run python scripts/benchmark_half_spectrum.py

Compares:
- Old: full-spectrum GEMM (N = H*W = 16384)
- New: half-spectrum GEMM (N_half = H*(W//2+1) = 8320)

Reports wall-clock time per iteration and speedup factor.
"""

import os
import sys
import time

import numpy as np

# Ensure no user-site pollution
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

print(f"JAX devices: {jax.devices()}")
if not any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in jax.devices()):
    print("WARNING: No GPU found. Benchmark results will not be representative.")

from recovar import core
from recovar.core.configs import ForwardModelConfig
import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume.engine_v2 import (
    make_half_image_weights,
    _preprocess_batch,
    _e_step_block_scores,
    _m_step_block,
    _compute_projections_block,
    _update_logsumexp,
)
from recovar.em import core as em_core

# --- Constants ---
IMAGE_SHAPE = (128, 128)
H, W = IMAGE_SHAPE
N_FULL = H * W              # 16384
N_HALF = H * (W // 2 + 1)   # 8320
VOLUME_SHAPE = (128, 128, 128)
VOLUME_SIZE = 128**3

N_IMAGES = 200      # per batch
N_ROTATIONS = 5000  # one rotation block
N_TRANS = 49        # 7x7
SEED = 42
N_WARMUP = 2
N_TIMED = 5


def make_synthetic_data():
    """Create synthetic data for benchmarking."""
    rng = np.random.default_rng(SEED)

    # Random Hermitian volume
    real_vol = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)
    ft = np.fft.fftshift(np.fft.fftn(real_vol))
    volume = jnp.array(ft.ravel(), dtype=jnp.complex64)

    # Random rotation matrices
    z = rng.standard_normal((N_ROTATIONS, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    rotations = jnp.array(q, dtype=jnp.float32)

    # Translation grid
    t_range = np.arange(-3, 4, dtype=np.float32)
    tx, ty = np.meshgrid(t_range, t_range)
    translations = jnp.array(np.stack([tx.ravel(), ty.ravel()], axis=1))

    # Random images (already in Fourier space, Hermitian)
    images = []
    for i in range(N_IMAGES):
        real_img = rng.standard_normal(IMAGE_SHAPE).astype(np.float32)
        ft_img = np.fft.fftshift(np.fft.fft2(real_img))
        images.append(ft_img.ravel())
    images = jnp.array(np.stack(images), dtype=jnp.complex64)

    noise_variance = jnp.ones(N_FULL, dtype=jnp.float32)
    ctf_params = jnp.zeros((N_IMAGES, 9), dtype=jnp.float32)

    return volume, rotations, translations, images, noise_variance, ctf_params


def benchmark_e_step_full(shifted_flat, batch_norm, ctf2_over_nv, proj_full, proj_abs2_full):
    """E-step using full-spectrum GEMM."""
    # Cross-term
    cross = -2.0 * (jnp.conj(shifted_flat) @ proj_full.T).real
    cross = cross.reshape(N_IMAGES, N_TRANS, N_ROTATIONS) + batch_norm[:, None, :]
    cross = cross.swapaxes(1, 2)
    # Norm-term
    norms = ctf2_over_nv @ proj_abs2_full.T
    residuals = cross + norms[..., None]
    return -0.5 * residuals


def benchmark_e_step_half(shifted_half, batch_norm, ctf2_nv_half, proj_half_w, proj_abs2_half_w):
    """E-step using half-spectrum GEMM."""
    cross = -2.0 * (jnp.conj(shifted_half) @ proj_half_w.T).real
    cross = cross.reshape(N_IMAGES, N_TRANS, N_ROTATIONS) + batch_norm[:, None, :]
    cross = cross.swapaxes(1, 2)
    norms = ctf2_nv_half @ proj_abs2_half_w.T
    residuals = cross + norms[..., None]
    return -0.5 * residuals


def main():
    print(f"\n{'='*60}")
    print(f"Half-Spectrum GEMM Benchmark")
    print(f"{'='*60}")
    print(f"Images: {N_IMAGES}, Rotations: {N_ROTATIONS}, Translations: {N_TRANS}")
    print(f"Image shape: {IMAGE_SHAPE}, N_full={N_FULL}, N_half={N_HALF}")
    print(f"Ratio: {N_HALF/N_FULL:.3f} (theoretical speedup: {N_FULL/N_HALF:.2f}x)")
    print()

    print("Creating synthetic data...")
    volume, rotations, translations, images, noise_variance, ctf_params = make_synthetic_data()

    # Prepare full-spectrum data
    print("Preparing full-spectrum data...")
    # Simulate preprocessing
    shifted_full = core.batch_trans_translate_images(
        images, jnp.repeat(translations[None], N_IMAGES, axis=0), IMAGE_SHAPE,
    )
    shifted_flat = shifted_full.reshape(N_IMAGES * N_TRANS, -1)
    batch_norm = jnp.linalg.norm(images, axis=-1, keepdims=True) ** 2
    ctf2_over_nv = jnp.ones_like(images)  # identity CTF

    # Full projections
    proj_full = core.slice_volume(
        volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=False
    )
    proj_abs2_full = jnp.abs(proj_full) ** 2

    # Prepare half-spectrum data
    print("Preparing half-spectrum data...")
    shifted_half = ftu.full_image_to_half_image(shifted_flat, IMAGE_SHAPE)
    ctf2_nv_half = ftu.full_image_to_half_image(ctf2_over_nv, IMAGE_SHAPE)

    proj_half = core.slice_volume(
        volume, rotations, IMAGE_SHAPE, VOLUME_SHAPE, "linear_interp", half_image=True
    )
    proj_abs2_half = jnp.abs(proj_half) ** 2
    half_weights = make_half_image_weights(IMAGE_SHAPE)
    proj_half_w = proj_half * half_weights
    proj_abs2_half_w = proj_abs2_half * half_weights

    # JIT compile
    print("JIT compiling...")
    jit_full = jax.jit(benchmark_e_step_full)
    jit_half = jax.jit(benchmark_e_step_half)

    # Warmup
    print(f"Warming up ({N_WARMUP} iterations each)...")
    for _ in range(N_WARMUP):
        s = jit_full(shifted_flat, batch_norm, ctf2_over_nv, proj_full, proj_abs2_full)
        s.block_until_ready()
    for _ in range(N_WARMUP):
        s = jit_half(shifted_half, batch_norm, ctf2_nv_half, proj_half_w, proj_abs2_half_w)
        s.block_until_ready()

    # Benchmark full-spectrum
    print(f"\nBenchmarking full-spectrum ({N_TIMED} iterations)...")
    times_full = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        s = jit_full(shifted_flat, batch_norm, ctf2_over_nv, proj_full, proj_abs2_full)
        s.block_until_ready()
        times_full.append(time.perf_counter() - t0)

    # Benchmark half-spectrum
    print(f"Benchmarking half-spectrum ({N_TIMED} iterations)...")
    times_half = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        s = jit_half(shifted_half, batch_norm, ctf2_nv_half, proj_half_w, proj_abs2_half_w)
        s.block_until_ready()
        times_half.append(time.perf_counter() - t0)

    # Report
    avg_full = np.mean(times_full)
    avg_half = np.mean(times_half)
    speedup = avg_full / avg_half

    print(f"\n{'='*60}")
    print(f"Results (E-step GEMM only)")
    print(f"{'='*60}")
    print(f"Full-spectrum ({N_FULL} pixels): {avg_full*1000:.1f} ms  (std: {np.std(times_full)*1000:.1f} ms)")
    print(f"Half-spectrum ({N_HALF} pixels): {avg_half*1000:.1f} ms  (std: {np.std(times_half)*1000:.1f} ms)")
    print(f"Speedup: {speedup:.2f}x")
    print(f"GEMM size reduction: {N_FULL}/{N_HALF} = {N_FULL/N_HALF:.2f}x")
    print()

    # Verify correctness: compare probabilities (which are invariant to additive constants)
    scores_full = jit_full(shifted_flat, batch_norm, ctf2_over_nv, proj_full, proj_abs2_full)
    scores_half = jit_half(shifted_half, batch_norm, ctf2_nv_half, proj_half_w, proj_abs2_half_w)

    # Softmax to get probabilities (scores should be real but force .real for safety)
    sf = scores_full.real.reshape(N_IMAGES, -1)
    sh = scores_half.real.reshape(N_IMAGES, -1)
    log_Z_f = jax.scipy.special.logsumexp(sf, axis=1, keepdims=True)
    log_Z_h = jax.scipy.special.logsumexp(sh, axis=1, keepdims=True)
    probs_full = jnp.exp(sf - log_Z_f)
    probs_half = jnp.exp(sh - log_Z_h)
    max_prob_diff = float(jnp.max(jnp.abs(probs_full - probs_half)))
    mean_prob_diff = float(jnp.mean(jnp.abs(probs_full - probs_half)))
    # Also check hard assignments match
    ha_full = jnp.argmax(sf, axis=1)
    ha_half = jnp.argmax(sh, axis=1)
    ha_match = float(jnp.mean(ha_full == ha_half))

    print(f"Correctness check (probabilities):")
    print(f"  Max |prob_full - prob_half|: {max_prob_diff:.2e}")
    print(f"  Mean |prob_full - prob_half|: {mean_prob_diff:.2e}")
    print(f"  Hard assignment match: {ha_match*100:.1f}%")
    if max_prob_diff < 1e-4:
        print("  PASS: Probabilities match within tolerance")
    else:
        print(f"  WARNING: Probability difference exceeds tolerance")


if __name__ == "__main__":
    main()
