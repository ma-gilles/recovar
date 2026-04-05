#!/usr/bin/env python
"""Demo: sketched normal-operator products on a PDB-based dataset.

Generates (or reuses) 5nrl trajectory volumes + simulated dataset,
loads it, builds a random low-rank iterate, runs both sketches.

Usage:
    pixi run python examples/sketched_normal_operator_demo.py
"""

import logging, os, time
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("demo")

# ── Configuration ──────────────────────────────────────────────────────
GRID_SIZE   = 64
N_IMAGES    = 5000
NOISE_LEVEL = 0.1          # lower = higher SNR
N_VOLUMES   = 10
RANK        = 10            # PCs in low-rank iterate
SKETCH_RANK = 5             # left sketch dimension
QRANK       = 5             # right sketch dimension
BATCH_SIZE  = 500
DISC_TYPE   = "linear_interp"
BASE_DIR    = os.path.join(
    os.environ.get("TMPDIR", "/scratch/gpfs/GILLES/mg6942/tmp"),
    "sketched_normal_demo",
)
# ───────────────────────────────────────────────────────────────────────

import jax, jax.numpy as jnp
import recovar.core.fourier_transform_utils as ftu
from recovar import utils
from recovar.simulation import simulator
from recovar.simulation.trajectory_generation import generate_trajectory_volumes
from recovar.ppca.ppca_scale_sweep import _load_simulated_dataset, _with_trailing_separator
from recovar.ppca.sketched_normal import compute_normal_residual_sketches


def _generate_or_load():
    """Generate volumes + dataset if needed, then load."""
    os.makedirs(BASE_DIR, exist_ok=True)
    voxel_size = 4.25 * 128 / GRID_SIZE
    vol_prefix = os.path.join(BASE_DIR, "generated_volumes", "vol")
    ds_dir = os.path.join(BASE_DIR, "test_dataset")

    # Volumes
    if not os.path.isfile(f"{vol_prefix}0000.mrc"):
        logger.info("Generating %d trajectory volumes (grid=%d) ...", N_VOLUMES, GRID_SIZE)
        generate_trajectory_volumes(
            BASE_DIR, grid_size=GRID_SIZE, n_volumes=N_VOLUMES,
            voxel_size=voxel_size, Bfactor=80, max_rotation_degrees=10.0,
        )
    else:
        logger.info("Reusing volumes at %s", vol_prefix)

    # Dataset
    if not os.path.isfile(os.path.join(ds_dir, f"particles.{GRID_SIZE}.mrcs")):
        logger.info("Simulating dataset: n=%d, noise=%.2g ...", N_IMAGES, NOISE_LEVEL)
        np.random.seed(42)
        simulator.generate_synthetic_dataset(
            ds_dir, voxel_size, vol_prefix, N_IMAGES,
            grid_size=GRID_SIZE,
            noise_level=NOISE_LEVEL, noise_model="radial1",
            contrast_std=0.0, noise_scale_std=0.0,
            dataset_params_option="uniform", disc_type=DISC_TYPE,
            trailing_zero_format_in_vol_name=True,
            put_extra_particles=False, percent_outliers=0.0,
        )
    else:
        logger.info("Reusing dataset at %s", ds_dir)

    # Load
    from recovar.ppca.ppca_scale_sweep import _build_halfset_indices
    cryos, sim_info, gt, noise_var = _load_simulated_dataset(
        _with_trailing_separator(ds_dir), GRID_SIZE, N_IMAGES, lazy=False,
    )
    return cryos, sim_info, gt, noise_var


def main():
    logger.info("JAX devices: %s", jax.devices())

    cryos, sim_info, gt, noise_var = _generate_or_load()
    cryo = cryos  # full dataset
    volume_shape = cryo.volume_shape
    half_vol_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol_size = int(np.prod(half_vol_shape))
    n_images = cryo.n_images
    logger.info("Loaded: grid=%d, n_images=%d, half_vol=%d", cryo.grid_size, n_images, half_vol_size)

    # ── Random low-rank iterate (half-volume layout) ──
    rng = np.random.default_rng(0)
    U_X_half = (rng.normal(size=(half_vol_size, RANK))
                + 1j * rng.normal(size=(half_vol_size, RANK))).astype(np.complex64)
    sigma_X = np.logspace(0, -1, RANK).astype(np.float32)
    V_X = rng.normal(size=(n_images, RANK)).astype(np.float32)
    mean_half = jnp.zeros(half_vol_size, dtype=np.complex64)

    # ── Sketch matrices ──
    S_left_half = (rng.normal(size=(SKETCH_RANK, half_vol_size))
                   + 1j * rng.normal(size=(SKETCH_RANK, half_vol_size))).astype(np.complex64)
    Q_right = rng.normal(size=(n_images, QRANK)).astype(np.float32)

    # ── Run ──
    logger.info("Computing sketches: rank=%d, s=%d, t=%d, batch=%d",
                RANK, SKETCH_RANK, QRANK, BATCH_SIZE)

    t0 = time.time()
    result = compute_normal_residual_sketches(
        cryo, U_X_half, sigma_X, V_X, mean_half,
        batch_size=BATCH_SIZE,
        left_sketch_half=S_left_half, right_sketch=Q_right,
        disc_type=DISC_TYPE,
    )
    jax.block_until_ready(result["left"])
    jax.block_until_ready(result["right"])
    t_warmup = time.time() - t0

    t0 = time.time()
    result = compute_normal_residual_sketches(
        cryo, U_X_half, sigma_X, V_X, mean_half,
        batch_size=BATCH_SIZE,
        left_sketch_half=S_left_half, right_sketch=Q_right,
        disc_type=DISC_TYPE,
    )
    jax.block_until_ready(result["left"])
    jax.block_until_ready(result["right"])
    t_run = time.time() - t0

    left, right = result["left"], result["right"]
    logger.info("S_L @ G(X): shape=%s, norm=%.4f", left.shape, np.linalg.norm(np.asarray(left)))
    logger.info("G(X) @ Q_R: shape=%s, norm=%.4f", right.shape, np.linalg.norm(np.asarray(right)))
    logger.info("all finite: %s",
                bool(np.all(np.isfinite(np.asarray(left)))
                     and np.all(np.isfinite(np.asarray(right)))))
    logger.info("warmup: %.2fs, compiled: %.3fs", t_warmup, t_run)


if __name__ == "__main__":
    main()
