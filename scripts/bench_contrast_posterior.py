#!/usr/bin/env python
"""Benchmark: spectral contrast-marginalized solver vs direct Cholesky.

Usage:
    pixi run python scripts/bench_contrast_posterior.py

Compares wall time and peak memory for representative (B, K, C) sizes.
Not a CI test — meant for manual profiling.
"""

import time
import sys

import jax
import jax.numpy as jnp
import numpy as np

from recovar.heterogeneity.contrast_posterior import (
    make_contrast_quadrature,
    solve_marginalized_contrast,
    _spectral_decomposition,
)


def _random_stats(rng, B, K):
    F = rng.standard_normal((B, K, K + 2)).astype(np.float32)
    H = np.einsum("bij,bkj->bik", F, F)
    H = 0.5 * (H + np.swapaxes(H, -1, -2))
    g = rng.standard_normal((B, K)).astype(np.float32)
    h = rng.standard_normal((B, K)).astype(np.float32)
    t = rng.standard_normal((B,)).astype(np.float32)
    nu = np.abs(rng.standard_normal((B,))).astype(np.float32) + 1.0
    y_norm_sq = np.abs(rng.standard_normal((B,))).astype(np.float32) + 10.0
    lambdas = np.abs(rng.standard_normal((K,))).astype(np.float32) + 0.1
    return (
        jnp.array(H), jnp.array(g), jnp.array(h),
        jnp.array(t), jnp.array(nu), jnp.array(y_norm_sq),
        jnp.array(lambdas),
    )


def direct_cholesky_marginalize(H, g, h, t, nu, y_norm_sq, lambdas, nodes, weights, c_mean, c_var):
    """Reference: per-node Cholesky solve (no eigendecomposition)."""
    B = g.shape[0]
    K = lambdas.shape[0]
    C = nodes.shape[0]

    def _solve_one_node(c):
        c2 = c ** 2
        A = jnp.diag(1.0 / lambdas) + c2 * H  # (B, K, K)
        q = c * (g - c * h)
        mu = jax.scipy.linalg.solve(A, q, assume_a="pos")
        logdet = jnp.sum(jnp.log(jnp.linalg.eigvalsh(A)), axis=-1)
        rho = y_norm_sq - 2.0 * c * t + c2 * nu
        quad = jnp.sum(q * mu, axis=-1)
        return mu, logdet, rho, quad

    # Solve at all nodes
    all_mu = []
    all_logdet = []
    all_rho = []
    all_quad = []
    for j in range(C):
        mu_j, ld_j, rho_j, quad_j = _solve_one_node(nodes[j])
        all_mu.append(mu_j)
        all_logdet.append(ld_j)
        all_rho.append(rho_j)
        all_quad.append(quad_j)

    all_mu = jnp.stack(all_mu, axis=1)  # (B, C, K)
    all_logdet = jnp.stack(all_logdet, axis=1)  # (B, C)
    all_rho = jnp.stack(all_rho, axis=1)
    all_quad = jnp.stack(all_quad, axis=1)

    log_unnorm = jnp.log(jnp.clip(weights, min=1e-30))[None, :] - 0.5 * (all_rho - all_quad + all_logdet)
    omega = jax.nn.softmax(log_unnorm, axis=-1)

    mean_z = jnp.sum(omega[:, :, None] * all_mu, axis=1)
    mean_c = jnp.sum(omega * nodes[None, :], axis=-1)
    return mean_z, mean_c


direct_cholesky_jit = jax.jit(direct_cholesky_marginalize, static_argnums=())


def benchmark(B, K, C, n_warmup=3, n_iter=20):
    rng = np.random.default_rng(42)
    H, g, h, t, nu, y_norm_sq, lambdas = _random_stats(rng, B, K)
    nodes, weights = make_contrast_quadrature(n_nodes=C)
    c_mean = jnp.float32(1.0)
    c_var = jnp.float32(np.inf)

    # Warmup spectral
    for _ in range(n_warmup):
        r = solve_marginalized_contrast(H, g, h, t, nu, y_norm_sq, lambdas, nodes, weights, c_mean, c_var)
        r.mean_z.block_until_ready()

    # Time spectral
    times_spec = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        r = solve_marginalized_contrast(H, g, h, t, nu, y_norm_sq, lambdas, nodes, weights, c_mean, c_var)
        r.mean_z.block_until_ready()
        times_spec.append(time.perf_counter() - t0)

    # Warmup direct Cholesky
    for _ in range(n_warmup):
        mz, mc = direct_cholesky_jit(H, g, h, t, nu, y_norm_sq, lambdas, nodes, weights, c_mean, c_var)
        mz.block_until_ready()

    # Time direct Cholesky
    times_chol = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        mz, mc = direct_cholesky_jit(H, g, h, t, nu, y_norm_sq, lambdas, nodes, weights, c_mean, c_var)
        mz.block_until_ready()
        times_chol.append(time.perf_counter() - t0)

    spec_ms = np.median(times_spec) * 1000
    chol_ms = np.median(times_chol) * 1000
    speedup = chol_ms / spec_ms if spec_ms > 0 else float("inf")

    print(f"  B={B:5d}  K={K:2d}  C={C:2d}  | spectral: {spec_ms:8.2f} ms  cholesky: {chol_ms:8.2f} ms  speedup: {speedup:.2f}x")
    return spec_ms, chol_ms


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()
    print("Benchmark: spectral eigendecomposition vs direct per-node Cholesky")
    print("=" * 85)

    configs = [
        # (B, K, C) — representative RECOVAR sizes
        (100, 4, 16),
        (100, 4, 50),
        (1000, 4, 16),
        (1000, 4, 50),
        (1000, 10, 16),
        (1000, 10, 50),
        (5000, 4, 16),
        (5000, 10, 16),
        (5000, 20, 16),
        (10000, 4, 16),
        (10000, 10, 16),
    ]

    for B, K, C in configs:
        try:
            benchmark(B, K, C)
        except Exception as e:
            print(f"  B={B:5d}  K={K:2d}  C={C:2d}  | ERROR: {e}")


if __name__ == "__main__":
    main()
