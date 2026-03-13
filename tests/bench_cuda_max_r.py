"""Benchmark CUDA max_r sphere clipping vs external disk masking.

Run on GPU node:
    pixi run python tests/bench_cuda_max_r.py

Measures:
  1. CUDA project/backproject WITHOUT max_r
  2. CUDA project/backproject WITH max_r
  3. External disk mask approach: mask images, then project/backproject without max_r
  4. Relative error between CUDA max_r and relion_interp max_r
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

from recovar.cuda_backproject import project as cuda_project, backproject as cuda_bp
from recovar.core import relion_interp
import recovar.core.fourier_transform_utils as ftu


def random_rotations(n, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return jnp.array(q, dtype=jnp.float32)


def hermitian_volume(N, seed=0):
    rng = np.random.default_rng(seed)
    real_vol = rng.standard_normal((N, N, N)).astype(np.float32)
    ft = np.fft.fftn(real_vol)
    return jnp.array(ft.ravel(), dtype=jnp.complex64)


def disk_mask(image_shape, max_r):
    """Create a 2D frequency-space disk mask (full image, centered)."""
    H, W = image_shape
    k0 = np.arange(H) - H // 2
    k1 = np.arange(W) - W // 2
    K0, K1 = np.meshgrid(k0, k1, indexing='ij')
    r2 = K0**2 + K1**2
    mask = (r2 <= max_r**2).astype(np.float32)
    return jnp.array(mask.ravel())


def half_disk_mask(image_shape, max_r):
    """Create a 2D frequency-space disk mask (half image, rfft layout)."""
    H, W = image_shape
    k0 = np.arange(H) - H // 2
    k1 = np.arange(W // 2 + 1)  # rfft: 0..W/2
    K0, K1 = np.meshgrid(k0, k1, indexing='ij')
    r2 = K0**2 + K1**2
    mask = (r2 <= max_r**2).astype(np.float32)
    return jnp.array(mask.ravel())


def bench(fn, *args, n_warmup=3, n_iter=50, **kwargs):
    """Benchmark a JAX function. Returns ms per call."""
    for _ in range(n_warmup):
        out = fn(*args, **kwargs)
        out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args, **kwargs)
        out.block_until_ready()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000


def main():
    print(f"JAX devices: {jax.devices()}")

    for N in [64, 128, 256]:
        n_images = 1000
        image_shape = (N, N)
        volume_shape = (N, N, N)
        max_r = N // 2 - 1

        print(f"\n{'='*60}")
        print(f"N={N}, n_images={n_images}, max_r={max_r}")
        print(f"{'='*60}")

        rots = random_rotations(n_images)
        vol_full = hermitian_volume(N)
        vol_half = ftu.full_volume_to_half_volume(vol_full, volume_shape)

        for half_vol, half_img in [(False, False), (True, True)]:
            label = f"half_vol={half_vol}, half_img={half_img}"
            vol = vol_half if half_vol else vol_full

            if half_img:
                n_pix = N * (N // 2 + 1)
                mask = half_disk_mask(image_shape, max_r)
            else:
                n_pix = N * N
                mask = disk_mask(image_shape, max_r)

            rng = np.random.default_rng(42)
            slices = jnp.array(
                (rng.standard_normal((n_images, n_pix)) +
                 1j * rng.standard_normal((n_images, n_pix))).astype(np.complex64)
            )

            print(f"\n  --- {label} ---")

            # PROJECT benchmarks
            ms_proj_none = bench(cuda_project, vol, rots, image_shape, volume_shape,
                                 1, half_vol, half_img, None)
            ms_proj_maxr = bench(cuda_project, vol, rots, image_shape, volume_shape,
                                 1, half_vol, half_img, float(max_r))
            # External mask approach: project without max_r, then mask output
            def proj_then_mask(v, r):
                out = cuda_project(v, r, image_shape, volume_shape,
                                   1, half_vol, half_img, None)
                return out * mask[None, :]
            ms_proj_ext = bench(proj_then_mask, vol, rots)

            print(f"  PROJECT:  no_maxr={ms_proj_none:.2f}ms  cuda_maxr={ms_proj_maxr:.2f}ms  "
                  f"ext_mask={ms_proj_ext:.2f}ms  "
                  f"speedup_vs_ext={ms_proj_ext/ms_proj_maxr:.2f}x")

            # BACKPROJECT benchmarks
            vol_shape_flat = ftu.volume_shape_to_half_volume_shape(volume_shape) if half_vol else volume_shape
            vol_size = int(np.prod(vol_shape_flat))
            zero_vol = jnp.zeros(vol_size, dtype=jnp.complex64)

            ms_bp_none = bench(cuda_bp, zero_vol, slices, rots, image_shape, volume_shape,
                               1, half_vol, half_img, None)
            ms_bp_maxr = bench(cuda_bp, zero_vol, slices, rots, image_shape, volume_shape,
                               1, half_vol, half_img, float(max_r))
            # External mask: mask slices first, then backproject
            masked_slices = slices * mask[None, :]
            ms_bp_ext = bench(cuda_bp, zero_vol, masked_slices, rots, image_shape, volume_shape,
                              1, half_vol, half_img, None)

            print(f"  BACKPROJ: no_maxr={ms_bp_none:.2f}ms  cuda_maxr={ms_bp_maxr:.2f}ms  "
                  f"ext_mask={ms_bp_ext:.2f}ms  "
                  f"speedup_vs_ext={ms_bp_ext/ms_bp_maxr:.2f}x")

            # Accuracy: CUDA max_r vs relion_interp max_r (small N only)
            if N <= 64:
                jax_proj = relion_interp.project(
                    vol, rots, image_shape, volume_shape,
                    order=1, half_volume=half_vol, half_image=half_img,
                    max_r=float(max_r),
                )
                cuda_proj = cuda_project(vol, rots, image_shape, volume_shape,
                                         1, half_vol, half_img, float(max_r))
                rel_err = float(jnp.linalg.norm(cuda_proj - jax_proj) / (jnp.linalg.norm(jax_proj) + 1e-30))
                print(f"  ACCURACY (project): CUDA vs relion_interp rel_err = {rel_err:.2e}")

                jax_bp = relion_interp.backproject(
                    slices[:10], rots[:10], image_shape, volume_shape,
                    order=1, half_volume=half_vol, half_image=half_img,
                    max_r=float(max_r),
                )
                cuda_bp_out = cuda_bp(
                    jnp.zeros(vol_size, dtype=jnp.complex64),
                    slices[:10], rots[:10], image_shape, volume_shape,
                    order=1, half_volume=half_vol, half_image=half_img,
                    max_r=float(max_r),
                )
                rel_err_bp = float(jnp.linalg.norm(cuda_bp_out - jax_bp) / (jnp.linalg.norm(jax_bp) + 1e-30))
                print(f"  ACCURACY (backproj): CUDA vs relion_interp rel_err = {rel_err_bp:.2e}")


if __name__ == "__main__":
    main()
