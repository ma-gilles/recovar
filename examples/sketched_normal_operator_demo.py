#!/usr/bin/env python
"""Demo: sketched normal-operator products on a synthetic dataset.

Generates a small synthetic cryo-EM dataset, constructs a random low-rank
iterate X = U diag(s) V^T, and computes both sketched products:
  - S_L @ G(X)   (left sketch, no backprojection)
  - G(X) @ Q_R   (right sketch, multi-channel backprojection)

Usage:
    pixi run python examples/sketched_normal_operator_demo.py
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

from recovar import core
from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import CryoEMDataset, ImageMetadata
from recovar.heterogeneity.sketched_normal import compute_normal_residual_sketches
import recovar.simulation.simulator as simulator


# ---------------------------------------------------------------------------
# Synthetic dataset helpers (self-contained, no test imports)
# ---------------------------------------------------------------------------


def _make_dataset(grid_size=16, n_images=64, seed=42):
    """Create a tiny synthetic CryoEMDataset with images."""
    np.random.seed(seed)

    # Two simple Fourier volumes
    vol_size = grid_size ** 3
    vols = np.array([
        np.linspace(0.1, 1.0, vol_size),
        np.linspace(1.0, 0.2, vol_size),
    ], dtype=np.float32).astype(np.complex64)

    # CTF / pose generator
    def param_gen(n, gs):
        ctf = np.zeros((n, 9), dtype=np.float32)
        ctf[:, core.CTFParamIndex.DFU] = 15000.0
        ctf[:, core.CTFParamIndex.DFV] = 15000.0
        ctf[:, core.CTFParamIndex.VOLT] = 300.0
        ctf[:, core.CTFParamIndex.CS] = 2.7
        ctf[:, core.CTFParamIndex.W] = 0.1
        ctf[:, core.CTFParamIndex.BFACTOR] = 50.0
        ctf[:, core.CTFParamIndex.CONTRAST] = 1.0
        rots = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        trans = np.zeros((n, 2), dtype=np.float32)
        return ctf, rots, trans

    images, ctf_params, rots, trans, _, voxel_size, _ = (
        simulator.generate_simulated_dataset(
            volumes=vols, voxel_size=1.5,
            volume_distribution=np.array([0.5, 0.5], dtype=np.float32),
            n_images=n_images,
            noise_variance=np.ones(grid_size // 2 - 1, dtype=np.float32) * 1e-6,
            noise_scale_std=0.0, contrast_std=0.0,
            put_extra_particles=False, percent_outliers=0.0,
            dataset_param_generator=param_gen,
            disc_type="linear_interp",
            image_offset_n_std=0.0,
            per_particle_contrast=True,
            premultiplied_ctf=False,
        )
    )

    # In-memory Fourier image backend
    class FTStack:
        def __init__(self, imgs_real):
            imgs_real = np.asarray(imgs_real)
            self.n_images = imgs_real.shape[0]
            self.D = int(imgs_real.shape[-1])
            self.unpadded_D = self.D
            self.padding = 0
            self.image_shape = (self.D, self.D)
            self.mask = np.ones(self.image_shape, dtype=np.float32)
            self.Np = self.n_images
            self._ft = np.asarray(
                ftu.get_dft2(imgs_real)
            ).reshape(self.n_images, -1).astype(np.complex64)
            self.already_prefetches = False

        def get_dataset_generator(self, batch_size, num_workers=0, **kw):
            for s in range(0, self.n_images, batch_size):
                e = min(s + batch_size, self.n_images)
                idx = np.arange(s, e, dtype=np.int32)
                yield self._ft[idx], idx, idx

        def get_image_generator(self, *a, **kw):
            return self.get_dataset_generator(*a, **kw)

        def get_dataset_subset_generator(self, bs, sub, **kw):
            sub = np.asarray(sub, dtype=np.int32)
            for s in range(0, sub.size, bs):
                idx = sub[s:s + bs]
                yield self._ft[idx], idx, idx

        def get_image_subset_generator(self, *a, **kw):
            return self.get_dataset_subset_generator(*a, **kw)

        def process_images(self, image, apply_image_mask=True):
            return image

    class RadialNoise:
        def __init__(self, size):
            self._n = np.ones((size,), dtype=np.float32)

        def get(self, idx):
            return np.tile(self._n[None], (len(idx), 1))

        def get_half(self, idx):
            return self.get(idx)

    stack = FTStack(images)
    metadata = ImageMetadata(rots, trans, ctf_params)
    cryo = CryoEMDataset(
        image_source=stack,
        voxel_size=voxel_size,
        metadata=metadata,
        ctf_evaluator=core.CTFEvaluator(),
        dataset_indices=np.arange(stack.n_images, dtype=np.int32),
        grid_size=grid_size,
    )
    cryo.noise = RadialNoise(np.prod(cryo.image_shape))
    return cryo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"JAX devices: {jax.devices()}")

    # --- Generate synthetic dataset ---
    grid_size = 16
    n_images = 64
    print(f"\nGenerating synthetic dataset: grid={grid_size}, n_images={n_images}")
    cryo = _make_dataset(grid_size=grid_size, n_images=n_images)
    volume_size = cryo.volume_size
    print(f"  volume_shape={cryo.volume_shape}, image_shape={cryo.image_shape}")

    # --- Random low-rank iterate X = U diag(s) V^T ---
    rank = 5
    sketch_rank = 3
    qrank = 4
    rng = np.random.default_rng(0)

    U_X = (rng.normal(size=(volume_size, rank)) +
           1j * rng.normal(size=(volume_size, rank))).astype(np.complex64)
    sigma_X = np.array([1.0, 0.8, 0.5, 0.3, 0.1], dtype=np.float32)
    V_X = rng.normal(size=(n_images, rank)).astype(np.float32)
    mean = np.zeros(volume_size, dtype=np.complex64)
    noise_variance = np.ones(1, dtype=np.float32)

    S_left = (rng.normal(size=(sketch_rank, volume_size)) +
              1j * rng.normal(size=(sketch_rank, volume_size))).astype(np.complex64)
    Q_right = rng.normal(size=(n_images, qrank)).astype(np.float32)

    # --- Compute ---
    batch_size = 32
    pc_batch_size = 3
    sketch_chunk_size = 2

    print(f"\nComputing sketches: rank={rank}, sketch_rank={sketch_rank}, qrank={qrank}")
    print(f"  batch_size={batch_size}, pc_batch_size={pc_batch_size}")

    t0 = time.time()
    result = compute_normal_residual_sketches(
        cryo, U_X, sigma_X, V_X, mean, noise_variance,
        batch_size=batch_size,
        left_sketch=S_left, right_sketch=Q_right,
        disc_type="nearest",
        pc_batch_size=pc_batch_size,
        sketch_chunk_size=sketch_chunk_size,
    )
    jax.block_until_ready(result["left"])
    jax.block_until_ready(result["right"])
    t_warmup = time.time() - t0

    t0 = time.time()
    result = compute_normal_residual_sketches(
        cryo, U_X, sigma_X, V_X, mean, noise_variance,
        batch_size=batch_size,
        left_sketch=S_left, right_sketch=Q_right,
        disc_type="nearest",
        pc_batch_size=pc_batch_size,
        sketch_chunk_size=sketch_chunk_size,
    )
    jax.block_until_ready(result["left"])
    jax.block_until_ready(result["right"])
    t_run = time.time() - t0

    left, right = result["left"], result["right"]
    print(f"\nResults:")
    print(f"  S_L @ G(X): shape={left.shape}, dtype={left.dtype}, "
          f"norm={np.linalg.norm(np.asarray(left)):.4f}")
    print(f"  G(X) @ Q_R: shape={right.shape}, dtype={right.dtype}, "
          f"norm={np.linalg.norm(np.asarray(right)):.4f}")
    print(f"  all finite: {bool(np.all(np.isfinite(np.asarray(left))) and np.all(np.isfinite(np.asarray(right))))}")
    print(f"\nTiming:")
    print(f"  warmup (JIT compile): {t_warmup:.2f}s")
    print(f"  compiled run:         {t_run:.3f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
