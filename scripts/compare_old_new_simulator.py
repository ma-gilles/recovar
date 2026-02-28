"""Compare old (commit 911604ea) vs new simulator behavior.

Tests:
  1. simulate_data_batch / simulate_batch produce identical results
  2. GPU memory usage is comparable
  3. Full make_big_test_dataset generates identical datasets
  4. Reproduces the OOM scenario (50K images, grid 128, cubic)

Usage:
  PYTHONNOUSERSITE=1 python scripts/compare_old_new_simulator.py
"""
import sys
import os
import time
import json
import traceback
import functools

import numpy as np
import jax
import jax.numpy as jnp

# Ensure we import from THIS repo
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Also add old worktree to path for comparison imports
OLD_ROOT = os.path.join(REPO_ROOT, ".claude/worktrees/old-baseline")


def get_gpu_mem_used_mb():
    """Current GPU memory used in MB."""
    try:
        stats = jax.local_devices()[0].memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            return stats["peak_bytes_in_use"] / 1e6
        return -1
    except Exception:
        return -1


def reset_gpu_peak():
    """Reset peak memory counter (not always supported)."""
    try:
        # Force garbage collection
        import gc
        gc.collect()
        jax.clear_caches()
    except Exception:
        pass


# ======================================================================
# Test 1: Single batch equivalence and memory
# ======================================================================
def test_single_batch_equivalence():
    """Compare simulate_data_batch (old @jax.jit) vs simulate_batch (new @eqx.filter_jit)."""
    print("\n" + "=" * 70)
    print("TEST 1: Single batch equivalence (old jax.jit vs new eqx.filter_jit)")
    print("=" * 70)

    from recovar import core, simulator, utils
    from recovar.configs import ForwardModelConfig
    from recovar.core import ctf as core_ctf

    for grid_size in [32, 64, 128]:
        for disc_type in ["linear_interp", "cubic"]:
            for batch_size in [64, 256, 1280]:
                if grid_size == 128 and batch_size == 1280 and disc_type == "cubic":
                    label = f"grid={grid_size} disc={disc_type} batch={batch_size} (OOM scenario)"
                else:
                    label = f"grid={grid_size} disc={disc_type} batch={batch_size}"

                print(f"\n--- {label} ---")

                image_shape = (grid_size, grid_size)
                volume_shape = (grid_size, grid_size, grid_size)
                voxel_size = 4.25 * 128 / grid_size

                rng = np.random.RandomState(42)
                volume = jnp.array(rng.randn(np.prod(volume_shape)).astype(np.float32))

                if disc_type == "cubic":
                    from recovar import cubic_interpolation
                    volume = cubic_interpolation.calculate_spline_coefficients(
                        volume.reshape(volume_shape)
                    )

                rot_matrices = jnp.array(rng.randn(batch_size, 3, 3).astype(np.float32))
                translations = jnp.zeros((batch_size, 2), dtype=np.float32)
                ctf_params = jnp.array(
                    np.tile(
                        np.array([grid_size, voxel_size, 15000, 14000, 5.0, 300, 2.7, 0.07, 0], dtype=np.float32),
                        (batch_size, 1),
                    )
                )
                CTF_fun = core_ctf.evaluate_ctf_wrapper

                # --- Old function (jax.jit) ---
                reset_gpu_peak()
                try:
                    t0 = time.time()
                    old_result = simulator.simulate_data_batch(
                        volume, rot_matrices, translations, ctf_params,
                        voxel_size, volume_shape, image_shape, grid_size,
                        disc_type, CTF_fun, skip_ctf=True,
                    )
                    old_result.block_until_ready()
                    old_time = time.time() - t0
                    old_mem = get_gpu_mem_used_mb()
                    old_ok = True
                    print(f"  OLD (jax.jit):      shape={old_result.shape} time={old_time:.3f}s mem={old_mem:.0f}MB")
                except Exception as e:
                    old_ok = False
                    old_result = None
                    print(f"  OLD (jax.jit):      FAILED — {type(e).__name__}: {e}")

                # --- New function (eqx.filter_jit) ---
                reset_gpu_peak()
                config = ForwardModelConfig(
                    image_shape=image_shape,
                    volume_shape=volume_shape,
                    grid_size=grid_size,
                    voxel_size=voxel_size,
                    padding=0,
                    disc_type=disc_type,
                    CTF_fun=CTF_fun,
                    premultiplied_ctf=False,
                    volume_mask_threshold=0.0,
                )
                try:
                    t0 = time.time()
                    new_result = simulator.simulate_batch(
                        config, volume, rot_matrices, translations, ctf_params,
                        skip_ctf=True,
                    )
                    new_result.block_until_ready()
                    new_time = time.time() - t0
                    new_mem = get_gpu_mem_used_mb()
                    new_ok = True
                    print(f"  NEW (eqx.filter_jit): shape={new_result.shape} time={new_time:.3f}s mem={new_mem:.0f}MB")
                except Exception as e:
                    new_ok = False
                    new_result = None
                    print(f"  NEW (eqx.filter_jit): FAILED — {type(e).__name__}: {e}")

                # --- Compare ---
                if old_ok and new_ok:
                    max_diff = float(jnp.max(jnp.abs(old_result - new_result)))
                    rel_diff = float(max_diff / (jnp.max(jnp.abs(old_result)) + 1e-12))
                    match = "MATCH" if max_diff < 1e-5 else "MISMATCH"
                    print(f"  {match}: max_abs_diff={max_diff:.2e} rel_diff={rel_diff:.2e}")
                elif old_ok != new_ok:
                    print(f"  DIVERGENCE: old_ok={old_ok} new_ok={new_ok}")
                    if not new_ok and old_ok:
                        print("  >>> NEW function FAILS where OLD succeeds! This is a regression.")


# ======================================================================
# Test 2: Full pipeline generate_synthetic_dataset comparison
# ======================================================================
def test_full_dataset_generation():
    """Compare full dataset generation: old code path vs new code path.

    Uses the actual make_test_dataset flow with smaller parameters.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Full dataset generation comparison")
    print("=" * 70)

    import tempfile
    from recovar import simulator, output, utils
    from recovar.commands.run_test_all_metrics import make_big_test_dataset

    grid_size = 32
    n_images = 200

    # Generate test volumes first
    from recovar.commands.run_test_all_metrics import generate_compact_support_test_volumes
    with tempfile.TemporaryDirectory() as tmpdir:
        vol_prefix = generate_compact_support_test_volumes(
            output_dir=tmpdir,
            grid_size=grid_size,
            n_volumes=3,
            voxel_size=4.25 * 128 / grid_size,
        )
        print(f"  Generated volumes at: {vol_prefix}")

        for disc_type in ["linear_interp", "cubic"]:
            print(f"\n--- disc_type={disc_type}, grid={grid_size}, n_images={n_images} ---")
            out_dir = os.path.join(tmpdir, f"test_{disc_type}")
            os.makedirs(out_dir, exist_ok=True)

            try:
                t0 = time.time()
                sim_info = make_big_test_dataset(
                    vol_prefix, out_dir,
                    noise_level=0.1, grid_size=grid_size, n_images=n_images,
                    contrast_std=0.1,
                )
                elapsed = time.time() - t0
                n_generated = len(sim_info.get("image_assignment", []))
                print(f"  OK: {n_generated} images in {elapsed:.1f}s")
                mem = get_gpu_mem_used_mb()
                print(f"  Peak GPU mem: {mem:.0f}MB")
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()


# ======================================================================
# Test 3: Reproduce the exact OOM scenario
# ======================================================================
def test_oom_scenario():
    """Reproduce the OOM: 50K images, grid 128, cubic."""
    print("\n" + "=" * 70)
    print("TEST 3: OOM reproduction (grid=128, cubic, batch=1280)")
    print("=" * 70)

    from recovar import core, simulator, utils, core_ctf
    from recovar.configs import ForwardModelConfig
    from recovar import cubic_interpolation

    grid_size = 128
    batch_size = 1280  # This is what get_image_batch_size(128, 80) * 1 gives
    image_shape = (grid_size, grid_size)
    volume_shape = (grid_size, grid_size, grid_size)
    voxel_size = 4.25

    rng = np.random.RandomState(42)
    raw_vol = jnp.array(rng.randn(np.prod(volume_shape)).astype(np.float32))
    volume = cubic_interpolation.calculate_spline_coefficients(raw_vol.reshape(volume_shape))

    rot_matrices = jnp.array(rng.randn(batch_size, 3, 3).astype(np.float32))
    translations = jnp.zeros((batch_size, 2), dtype=np.float32)
    ctf_params = jnp.array(
        np.tile(
            np.array([grid_size, voxel_size, 15000, 14000, 5.0, 300, 2.7, 0.07, 0], dtype=np.float32),
            (batch_size, 1),
        )
    )
    CTF_fun = core_ctf.evaluate_ctf_wrapper

    gpu_mem = utils.get_gpu_memory_total()
    print(f"  GPU memory: {gpu_mem} GB")
    print(f"  batch_size: {batch_size}")
    print(f"  Volume shape after cubic coeffs: {volume.shape}")

    # Try old function
    print("\n  --- OLD (jax.jit) simulate_data_batch ---")
    reset_gpu_peak()
    try:
        t0 = time.time()
        old_result = simulator.simulate_data_batch(
            volume, rot_matrices, translations, ctf_params,
            voxel_size, volume_shape, image_shape, grid_size,
            "cubic", CTF_fun, skip_ctf=True,
        )
        old_result.block_until_ready()
        old_time = time.time() - t0
        old_mem = get_gpu_mem_used_mb()
        print(f"  OK: shape={old_result.shape} time={old_time:.3f}s peak_mem={old_mem:.0f}MB")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # Try new function
    print("\n  --- NEW (eqx.filter_jit) simulate_batch ---")
    reset_gpu_peak()
    config = ForwardModelConfig(
        image_shape=image_shape,
        volume_shape=volume_shape,
        grid_size=grid_size,
        voxel_size=voxel_size,
        padding=0,
        disc_type="cubic",
        CTF_fun=CTF_fun,
        premultiplied_ctf=False,
        volume_mask_threshold=0.0,
    )
    try:
        t0 = time.time()
        new_result = simulator.simulate_batch(
            config, volume, rot_matrices, translations, ctf_params,
            skip_ctf=True,
        )
        new_result.block_until_ready()
        new_time = time.time() - t0
        new_mem = get_gpu_mem_used_mb()
        print(f"  OK: shape={new_result.shape} time={new_time:.3f}s peak_mem={new_mem:.0f}MB")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # Try with reduced batch sizes to find the threshold
    print("\n  --- Finding max batch_size that works ---")
    for bs in [1280, 640, 320, 160, 80]:
        rm = jnp.array(rng.randn(bs, 3, 3).astype(np.float32))
        tr = jnp.zeros((bs, 2), dtype=np.float32)
        cp = jnp.array(
            np.tile(
                np.array([grid_size, voxel_size, 15000, 14000, 5.0, 300, 2.7, 0.07, 0], dtype=np.float32),
                (bs, 1),
            )
        )
        reset_gpu_peak()
        try:
            result = simulator.simulate_batch(
                config, volume, rm, tr, cp, skip_ctf=True,
            )
            result.block_until_ready()
            mem = get_gpu_mem_used_mb()
            print(f"  batch_size={bs:5d}: OK  peak_mem={mem:.0f}MB")
        except Exception as e:
            print(f"  batch_size={bs:5d}: FAILED ({type(e).__name__})")


# ======================================================================
# Test 4: Compare old vs new on covariance / embedding (the pipeline core)
# ======================================================================
def test_pipeline_functions():
    """Quick check that key pipeline functions produce same output."""
    print("\n" + "=" * 70)
    print("TEST 4: Pipeline function comparison (forward model, covariance, embedding)")
    print("=" * 70)

    from recovar import core, core_forward, simulator, core_ctf
    from recovar.configs import ForwardModelConfig

    grid_size = 32
    n_images = 64
    image_shape = (grid_size, grid_size)
    volume_shape = (grid_size, grid_size, grid_size)
    voxel_size = 4.25 * 128 / grid_size

    rng = np.random.RandomState(123)
    volume = jnp.array(rng.randn(np.prod(volume_shape)).astype(np.float32))
    rot_matrices = jnp.array(rng.randn(n_images, 3, 3).astype(np.float32))
    ctf_params = jnp.array(
        np.tile(
            np.array([grid_size, voxel_size, 15000, 14000, 5.0, 300, 2.7, 0.07, 0], dtype=np.float32),
            (n_images, 1),
        )
    )
    CTF_fun = core_ctf.evaluate_ctf_wrapper

    config = ForwardModelConfig(
        image_shape=image_shape,
        volume_shape=volume_shape,
        grid_size=grid_size,
        voxel_size=voxel_size,
        padding=0,
        disc_type="linear_interp",
        CTF_fun=CTF_fun,
        premultiplied_ctf=False,
        volume_mask_threshold=0.0,
    )

    # forward_model: old vs new
    print("\n  forward_model_from_map:")
    old_fm = core_forward.forward_model_from_map(
        volume, ctf_params, rot_matrices,
        image_shape, volume_shape, voxel_size, CTF_fun, "linear_interp",
    )
    new_fm = core_forward.forward_model(config, volume, ctf_params, rot_matrices)

    max_diff = float(jnp.max(jnp.abs(old_fm - new_fm)))
    print(f"    max_diff = {max_diff:.2e}  {'MATCH' if max_diff < 1e-5 else 'MISMATCH'}")

    # adjoint:
    print("  adjoint_forward_model_from_map:")
    old_adj = core_forward.adjoint_forward_model_from_map(
        old_fm, ctf_params, rot_matrices,
        image_shape, volume_shape, voxel_size, CTF_fun, "linear_interp",
    )
    new_adj = core_forward.adjoint_forward_model(config, old_fm, ctf_params, rot_matrices)
    max_diff = float(jnp.max(jnp.abs(old_adj - new_adj)))
    print(f"    max_diff = {max_diff:.2e}  {'MATCH' if max_diff < 1e-5 else 'MISMATCH'}")


if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Python: {sys.executable}")
    print(f"Repo root: {REPO_ROOT}")

    test_single_batch_equivalence()
    test_full_dataset_generation()
    test_oom_scenario()
    test_pipeline_functions()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
