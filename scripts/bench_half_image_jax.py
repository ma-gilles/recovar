"""Benchmark: half-image/half-volume JAX slicing vs full paths.

Measures wall-clock time for forward and adjoint operations across all
(half_image, half_volume) combinations.
Run on a compute node (GPU or CPU) via SLURM.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

# Force JAX CPU to isolate JAX path (no CUDA dispatch)
jax.config.update("jax_platform_name", "cpu")

from scipy.spatial.transform import Rotation

import recovar.core.slicing as core_slicing
import recovar.core.fourier_transform_utils as ftu

# Monkey-patch to ensure JAX path
core_slicing._on_gpu = lambda: False


def random_rotations(rng, n):
    return Rotation.random(n, random_state=rng.integers(2**31)).as_matrix().astype(np.float32)


def random_hermitian_volume(rng, volume_shape):
    """Create a volume with Hermitian symmetry."""
    real = rng.standard_normal(volume_shape).astype(np.float32)
    imag = rng.standard_normal(volume_shape).astype(np.float32)
    vol = real + 1j * imag
    # Enforce Hermitian: V(-k) = conj(V(k))
    vol = (vol + np.conj(vol[::-1, ::-1, ::-1])) / 2
    return vol


def bench(fn, *args, warmup=2, repeats=5):
    """Time a JIT-compiled function, returning median wall-clock in ms."""
    for _ in range(warmup):
        result = fn(*args)
        result.block_until_ready()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        result.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times), result


def disc_type_for_order(order):
    return {0: "nearest", 1: "linear_interp", 3: "cubic"}[order]


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("Benchmark: half-image/half-volume JAX slicing paths")
    print(f"Backend: {jax.default_backend()}")
    print("=" * 70)

    for grid_size in [32, 64, 128]:
        volume_shape = (grid_size, grid_size, grid_size)
        image_shape = (grid_size, grid_size)
        n_images = 50

        vol_full_np = random_hermitian_volume(rng, volume_shape)
        vol_full = jnp.array(vol_full_np.ravel())
        vol_half = jnp.array(np.asarray(ftu.full_volume_to_half_volume(jnp.array(vol_full_np), volume_shape)).ravel())

        rots = jnp.array(random_rotations(rng, n_images))

        # Full images
        full_imgs = jnp.array(
            (
                rng.standard_normal((n_images, grid_size * grid_size))
                + 1j * rng.standard_normal((n_images, grid_size * grid_size))
            ).astype(np.complex64)
        )
        half_imgs = jnp.array(np.asarray(ftu.full_image_to_half_image(full_imgs, image_shape)))

        full_pix = grid_size * grid_size
        half_pix = grid_size * (grid_size // 2 + 1)
        full_vol_size = grid_size**3
        half_vol_size = grid_size * grid_size * (grid_size // 2 + 1)

        print(f"\n{'─' * 70}")
        print(f"Grid size: {grid_size}  |  n_images: {n_images}")
        print(f"Full image pixels: {full_pix}  |  Half image pixels: {half_pix}  ({half_pix / full_pix:.1%})")
        print(
            f"Full volume size: {full_vol_size}  |  Half volume size: {half_vol_size}  ({half_vol_size / full_vol_size:.1%})"
        )
        print(f"{'─' * 70}")

        for order_name, order in [("nearest", 0), ("linear", 1)]:
            dt = disc_type_for_order(order)
            print(f"\n  Order: {order_name} ({order})")

            # ── Forward benchmarks ──
            t_full, _ = bench(core_slicing._jax_slice, vol_full, rots, image_shape, volume_shape, order)
            t_half_img, _ = bench(core_slicing._jax_slice_half_image, vol_full, rots, image_shape, volume_shape, order)

            def fwd_half_vol_full_img(hv):
                fv = ftu.half_volume_to_full_volume(hv, volume_shape)
                return core_slicing._jax_slice(fv, rots, image_shape, volume_shape, order)

            t_hvol_fimg, _ = bench(fwd_half_vol_full_img, vol_half)

            def fwd_half_vol_half_img(hv):
                fv = ftu.half_volume_to_full_volume(hv, volume_shape)
                return core_slicing._jax_slice_half_image(fv, rots, image_shape, volume_shape, order)

            t_hvol_himg, _ = bench(fwd_half_vol_half_img, vol_half)

            print(f"    FORWARD:")
            print(f"      full_vol → full_img:  {t_full:8.2f} ms")
            print(f"      full_vol → half_img:  {t_half_img:8.2f} ms  (speedup: {t_full / t_half_img:.2f}x)")
            print(f"      half_vol → full_img:  {t_hvol_fimg:8.2f} ms  (speedup: {t_full / t_hvol_fimg:.2f}x)")
            print(f"      half_vol → half_img:  {t_hvol_himg:8.2f} ms  (speedup: {t_full / t_hvol_himg:.2f}x)")

            # ── Adjoint benchmarks: all 4 (half_image, half_volume) combos ──
            # 1. full_img → full_vol (baseline)
            def adj_fi_fv(imgs):
                return core_slicing.adjoint_slice_volume(
                    imgs,
                    rots,
                    image_shape,
                    volume_shape,
                    dt,
                    half_image=False,
                    half_volume=False,
                )

            t_adj_fi_fv, _ = bench(adj_fi_fv, full_imgs)

            # 2. half_img → full_vol
            def adj_hi_fv(imgs):
                return core_slicing.adjoint_slice_volume(
                    imgs,
                    rots,
                    image_shape,
                    volume_shape,
                    dt,
                    half_image=True,
                    half_volume=False,
                )

            t_adj_hi_fv, _ = bench(adj_hi_fv, half_imgs)

            # 3. full_img → half_vol
            def adj_fi_hv(imgs):
                return core_slicing.adjoint_slice_volume(
                    imgs,
                    rots,
                    image_shape,
                    volume_shape,
                    dt,
                    half_image=False,
                    half_volume=True,
                )

            t_adj_fi_hv, _ = bench(adj_fi_hv, full_imgs)

            # 4. half_img → half_vol
            def adj_hi_hv(imgs):
                return core_slicing.adjoint_slice_volume(
                    imgs,
                    rots,
                    image_shape,
                    volume_shape,
                    dt,
                    half_image=True,
                    half_volume=True,
                )

            t_adj_hi_hv, _ = bench(adj_hi_hv, half_imgs)

            print(f"    ADJOINT:")
            print(f"      full_img → full_vol:  {t_adj_fi_fv:8.2f} ms  (baseline)")
            print(f"      half_img → full_vol:  {t_adj_hi_fv:8.2f} ms  (speedup: {t_adj_fi_fv / t_adj_hi_fv:.2f}x)")
            print(f"      full_img → half_vol:  {t_adj_fi_hv:8.2f} ms  (speedup: {t_adj_fi_fv / t_adj_fi_hv:.2f}x)")
            print(f"      half_img → half_vol:  {t_adj_hi_hv:8.2f} ms  (speedup: {t_adj_fi_fv / t_adj_hi_hv:.2f}x)")

        # ── Cubic forward with precomputed coefficients ──
        print(f"\n  Order: cubic (3) — precomputed coefficients")
        from recovar.core import cubic_interpolation

        coeffs = cubic_interpolation.calculate_spline_coefficients(jnp.array(vol_full_np))

        t_cubic_full, _ = bench(
            core_slicing.slice_from_cubic_coefficients,
            coeffs,
            rots,
            image_shape,
            volume_shape,
            half_image=False,
        )
        t_cubic_half, _ = bench(
            core_slicing.slice_from_cubic_coefficients,
            coeffs,
            rots,
            image_shape,
            volume_shape,
            half_image=True,
        )
        print(f"    FORWARD (from precomputed coeffs):")
        print(f"      coeffs → full_img:    {t_cubic_full:8.2f} ms")
        print(f"      coeffs → half_img:    {t_cubic_half:8.2f} ms  (speedup: {t_cubic_full / t_cubic_half:.2f}x)")

        # ── Cubic adjoint (all 4 combos) ──
        print(f"    ADJOINT (cubic):")
        dt_cubic = "cubic"
        for hi_label, hi_flag, imgs in [("full_img", False, full_imgs), ("half_img", True, half_imgs)]:
            for hv_label, hv_flag in [("full_vol", False), ("half_vol", True)]:

                def adj_cubic(imgs_arg, _hi=hi_flag, _hv=hv_flag):
                    return core_slicing.adjoint_slice_volume(
                        imgs_arg,
                        rots,
                        image_shape,
                        volume_shape,
                        dt_cubic,
                        half_image=_hi,
                        half_volume=_hv,
                    )

                t, _ = bench(adj_cubic, imgs)
                tag = " (baseline)" if not hi_flag and not hv_flag else ""
                print(f"      {hi_label} → {hv_label}:  {t:8.2f} ms{tag}")

    # ── Memory comparison via output array sizes ──
    print(f"\n{'=' * 70}")
    print("Output array sizes (bytes, complex64):")
    for grid_size in [32, 64, 128, 256]:
        full_img_bytes = 50 * grid_size * grid_size * 8  # complex64 = 8 bytes
        half_img_bytes = 50 * grid_size * (grid_size // 2 + 1) * 8
        full_vol_bytes = grid_size**3 * 8
        half_vol_bytes = grid_size * grid_size * (grid_size // 2 + 1) * 8
        print(
            f"  Grid {grid_size:3d}:  full_img={full_img_bytes / 1e6:7.1f} MB  half_img={half_img_bytes / 1e6:7.1f} MB  "
            f"ratio={half_img_bytes / full_img_bytes:.2f}  |  "
            f"full_vol={full_vol_bytes / 1e6:7.1f} MB  half_vol={half_vol_bytes / 1e6:7.1f} MB  "
            f"ratio={half_vol_bytes / full_vol_bytes:.2f}"
        )

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
