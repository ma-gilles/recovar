#!/usr/bin/env python
"""Benchmark a RELION-pose fixed-assignment M-step floor.

This script measures the cost of backprojecting with known RELION poses and
translations for a single RELION iteration, without running the local E-step.
It is intended as a calibration floor for the hp4 local-refinement speed work.

Usage:
  pixi run python scripts/benchmark_relion_pose_mstep.py \
    --relion_dir /path/to/relion_ref_os0 \
    --data_star /path/to/particles.star \
    --iter 6 \
    --output_dir _agent_scratch/relion_pose_mstep_it006
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import starfile

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.core.configs import ForwardModelConfig
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.reconstruction import noise as recon_noise
from recovar.reconstruction import relion_functions


sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def stack_index_from_image_name(name: str) -> int:
    """Return the zero-based stack row encoded in a RELION image name."""
    match = re.match(r"(\d+)@", str(name))
    return int(match.group(1)) - 1 if match else -1


def _map_relion_iteration_to_particle_order(relion_dir: Path, iteration: int, our_names, pixel_size: float):
    iter_prefix = relion_dir / f"run_it{iteration:03d}"
    relion_data = starfile.read(f"{iter_prefix}_data.star")
    relion_df = relion_data["particles"] if isinstance(relion_data, dict) else relion_data
    model_h1 = starfile.read(f"{iter_prefix}_half1_model.star")
    model_h2 = starfile.read(f"{iter_prefix}_half2_model.star")

    relion_names = list(relion_df["rlnImageName"])
    relion_idx_to_pos = {stack_index_from_image_name(name): i for i, name in enumerate(relion_names)}

    subsets = np.zeros(len(our_names), dtype=np.int32)
    rotations = np.zeros((len(our_names), 3, 3), dtype=np.float32)
    translations = np.zeros((len(our_names), 2), dtype=np.float32)

    missing = []
    eulers = np.stack(
        [
            np.asarray(relion_df["rlnAngleRot"], dtype=np.float64),
            np.asarray(relion_df["rlnAngleTilt"], dtype=np.float64),
            np.asarray(relion_df["rlnAnglePsi"], dtype=np.float64),
        ],
        axis=1,
    )
    rotations_relion = utils.R_from_relion(eulers).astype(np.float32)

    if "rlnOriginXAngst" in relion_df.columns:
        translations_relion = np.stack(
            [
                np.asarray(relion_df["rlnOriginXAngst"], dtype=np.float64) / pixel_size,
                np.asarray(relion_df["rlnOriginYAngst"], dtype=np.float64) / pixel_size,
            ],
            axis=1,
        ).astype(np.float32)
    else:
        translations_relion = np.zeros((len(relion_df), 2), dtype=np.float32)

    relion_subsets = np.asarray(relion_df["rlnRandomSubset"], dtype=np.int32)

    for i, name in enumerate(our_names):
        idx = stack_index_from_image_name(name)
        pos = relion_idx_to_pos.get(idx)
        if pos is None:
            missing.append(name)
            continue
        subsets[i] = relion_subsets[pos]
        rotations[i] = rotations_relion[pos]
        translations[i] = translations_relion[pos]

    if missing:
        preview = ", ".join(str(x) for x in missing[:5])
        raise ValueError(f"Failed to map {len(missing)} particle names into RELION iteration data; first few: {preview}")

    sigma2_h1 = np.asarray(model_h1["model_optics_group_1"]["rlnSigma2Noise"], dtype=np.float64)
    sigma2_h2 = np.asarray(model_h2["model_optics_group_1"]["rlnSigma2Noise"], dtype=np.float64)

    return subsets, rotations, translations, sigma2_h1, sigma2_h2


def _subsample_halfsets(subsets: np.ndarray, max_particles: int | None) -> np.ndarray:
    if max_particles is None:
        return subsets
    rng = np.random.RandomState(42)
    subsets = np.array(subsets, copy=True)
    h1_idx = np.where(subsets == 1)[0]
    h2_idx = np.where(subsets == 2)[0]
    n_per_half = max_particles // 2
    if n_per_half < len(h1_idx):
        drop_h1 = rng.choice(h1_idx, size=len(h1_idx) - n_per_half, replace=False)
        subsets[drop_h1] = 0
    if n_per_half < len(h2_idx):
        drop_h2 = rng.choice(h2_idx, size=len(h2_idx) - n_per_half, replace=False)
        subsets[drop_h2] = 0
    return subsets


def _run_fixed_pose_half(ds_half, relion_rotations, relion_translations, cov_noise, batch_size: int):
    config = ForwardModelConfig.from_dataset(ds_half, disc_type="linear_interp", upsampling_factor=1)
    noise_model = recon_noise.as_noise_model(cov_noise, config.image_shape)

    def _first_batch():
        iterator = ds_half.iter_batches(batch_size, noise_model=noise_model, by_image=True)
        return next(iterator, None)

    first = _first_batch()
    if first is None:
        return {
            "warmup_s": 0.0,
            "run_s": 0.0,
            "batch_times_s": np.zeros((0,), dtype=np.float64),
            "n_batches": 0,
            "n_images": 0,
        }

    images, _rot, _trans, ctf_params, noise_variance, _particle_indices, image_indices = first
    relion_rot_b = jnp.asarray(relion_rotations[np.asarray(image_indices)], dtype=jnp.float32)
    relion_trans_b = jnp.asarray(relion_translations[np.asarray(image_indices)], dtype=jnp.float32)

    t0 = time.perf_counter()
    Ft_y_warm, Ft_ctf_warm = relion_functions.relion_kernel_batch(
        config,
        images,
        ctf_params,
        relion_rot_b,
        relion_trans_b,
        noise_variance,
        Ft_y=None,
        Ft_ctf=None,
    )
    jax.block_until_ready(Ft_y_warm)
    jax.block_until_ready(Ft_ctf_warm)
    warmup_s = time.perf_counter() - t0

    Ft_y = None
    Ft_ctf = None
    batch_times = []
    total_images = 0
    t_start = time.perf_counter()
    for (
        images,
        _rotation_matrices,
        _translations,
        ctf_params,
        noise_variance,
        _particle_indices,
        image_indices,
    ) in ds_half.iter_batches(batch_size, noise_model=noise_model, by_image=True):
        relion_rot_b = jnp.asarray(relion_rotations[np.asarray(image_indices)], dtype=jnp.float32)
        relion_trans_b = jnp.asarray(relion_translations[np.asarray(image_indices)], dtype=jnp.float32)
        t_batch = time.perf_counter()
        Ft_y, Ft_ctf = relion_functions.relion_kernel_batch(
            config,
            images,
            ctf_params,
            relion_rot_b,
            relion_trans_b,
            noise_variance,
            Ft_y=Ft_y,
            Ft_ctf=Ft_ctf,
        )
        jax.block_until_ready(Ft_y)
        jax.block_until_ready(Ft_ctf)
        batch_times.append(time.perf_counter() - t_batch)
        total_images += len(np.asarray(image_indices))

    run_s = time.perf_counter() - t_start

    return {
        "warmup_s": float(warmup_s),
        "run_s": float(run_s),
        "batch_times_s": np.asarray(batch_times, dtype=np.float64),
        "n_batches": int(len(batch_times)),
        "n_images": int(total_images),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark RELION-pose fixed-assignment M-step floor")
    parser.add_argument("--relion_dir", required=True)
    parser.add_argument("--data_star", required=True)
    parser.add_argument("--iter", type=int, required=True, help="RELION iteration whose poses to use")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--max_particles", type=int, default=None)
    args = parser.parse_args()

    devices = jax.devices()
    logger.info("JAX devices: %s", devices)
    if not any(d.platform == "gpu" for d in devices):
        raise RuntimeError("No GPU visible to JAX")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.data_star)
    our_particles = starfile.read(args.data_star)
    our_particles = our_particles["particles"] if isinstance(our_particles, dict) else our_particles
    our_names = list(our_particles["rlnImageName"])

    pixel_size = float(ds.voxel_size)
    subsets, rotations, translations, sigma2_h1, sigma2_h2 = _map_relion_iteration_to_particle_order(
        Path(args.relion_dir),
        args.iter,
        our_names,
        pixel_size,
    )
    subsets = _subsample_halfsets(subsets, args.max_particles)

    half1_idx = np.where(subsets == 1)[0]
    half2_idx = np.where(subsets == 2)[0]
    ds_half1 = ds.subset(half1_idx)
    ds_half2 = ds.subset(half2_idx)

    n4 = ds.grid_size**4
    cov_noise_h1 = np.asarray(sigma2_h1 * n4, dtype=np.float32)
    cov_noise_h2 = np.asarray(sigma2_h2 * n4, dtype=np.float32)

    logger.info(
        "Running RELION-pose M-step benchmark: iter=%03d, half sizes=%d + %d, batch_size=%d",
        args.iter,
        len(half1_idx),
        len(half2_idx),
        args.batch_size,
    )

    half1_stats = _run_fixed_pose_half(
        ds_half1,
        rotations[half1_idx],
        translations[half1_idx],
        cov_noise_h1,
        args.batch_size,
    )
    half2_stats = _run_fixed_pose_half(
        ds_half2,
        rotations[half2_idx],
        translations[half2_idx],
        cov_noise_h2,
        args.batch_size,
    )

    total_run_s = half1_stats["run_s"] + half2_stats["run_s"]
    total_warmup_s = half1_stats["warmup_s"] + half2_stats["warmup_s"]

    np.savez(
        output_dir / "relion_pose_mstep_benchmark.npz",
        relion_iteration=np.int32(args.iter),
        batch_size=np.int32(args.batch_size),
        max_particles=np.int32(-1 if args.max_particles is None else args.max_particles),
        half1_batch_times_s=half1_stats["batch_times_s"],
        half2_batch_times_s=half2_stats["batch_times_s"],
        half1_n_batches=np.int32(half1_stats["n_batches"]),
        half2_n_batches=np.int32(half2_stats["n_batches"]),
        half1_n_images=np.int32(half1_stats["n_images"]),
        half2_n_images=np.int32(half2_stats["n_images"]),
        half1_warmup_s=np.float64(half1_stats["warmup_s"]),
        half2_warmup_s=np.float64(half2_stats["warmup_s"]),
        half1_run_s=np.float64(half1_stats["run_s"]),
        half2_run_s=np.float64(half2_stats["run_s"]),
        total_warmup_s=np.float64(total_warmup_s),
        total_run_s=np.float64(total_run_s),
    )

    print("\n=== RELION-pose fixed-assignment M-step benchmark ===")
    print(f"  RELION iter: {args.iter:03d}")
    print(f"  Half sizes : {len(half1_idx)} + {len(half2_idx)}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Warmup     : h1={half1_stats['warmup_s']:.3f}s, h2={half2_stats['warmup_s']:.3f}s, total={total_warmup_s:.3f}s")
    print(f"  Run        : h1={half1_stats['run_s']:.3f}s, h2={half2_stats['run_s']:.3f}s, total={total_run_s:.3f}s")
    if half1_stats["n_batches"]:
        print(
            "  Half1 batch times: "
            f"mean={half1_stats['batch_times_s'].mean():.4f}s, "
            f"median={np.median(half1_stats['batch_times_s']):.4f}s, "
            f"max={half1_stats['batch_times_s'].max():.4f}s"
        )
    if half2_stats["n_batches"]:
        print(
            "  Half2 batch times: "
            f"mean={half2_stats['batch_times_s'].mean():.4f}s, "
            f"median={np.median(half2_stats['batch_times_s']):.4f}s, "
            f"max={half2_stats['batch_times_s'].max():.4f}s"
        )
    print(f"  Saved: {output_dir / 'relion_pose_mstep_benchmark.npz'}")


if __name__ == "__main__":
    main()
