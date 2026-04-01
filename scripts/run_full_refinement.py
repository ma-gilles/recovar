#!/usr/bin/env python
"""Run a full multi-iteration EM refinement and save per-iteration results.

This script loads the synthetic benchmark dataset (5000 images, 128px),
initializes from the low-pass filtered reference volume, and calls
refine_single_volume() with parameters matching the RELION auto-refine run.

Results are saved as a single .npz file with per-iteration arrays for
downstream comparison via compare_vs_relion.py.

Usage:
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        pixi run python scripts/run_full_refinement.py [--output DIR] [--max_iter N]

Environment variables:
    CUDA_VISIBLE_DEVICES: GPU to use
    XLA_PYTHON_CLIENT_PREALLOCATE: set to false for dynamic allocation
"""

import argparse
import logging
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import mrcfile
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run full EM refinement on synthetic data")
    parser.add_argument(
        "--data_dir",
        default="/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data",
        help="Directory containing particles.star, reference_init.mrc, etc.",
    )
    parser.add_argument(
        "--output",
        default="/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/our_results",
        help="Directory to save results",
    )
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum EM iterations")
    parser.add_argument("--healpix_order", type=int, default=3, help="HEALPix order for rotation grid")
    parser.add_argument("--offset_range", type=float, default=3.0, help="Translation search range (pixels)")
    parser.add_argument("--offset_step", type=float, default=1.0, help="Translation step (pixels)")
    parser.add_argument("--adaptive_oversampling", type=int, default=1, help="Oversampling levels (0=off, 1=2x)")
    parser.add_argument("--adaptive_fraction", type=float, default=0.999, help="Significance fraction")
    parser.add_argument("--max_significants", type=int, default=500, help="Max significant samples per image")
    parser.add_argument("--init_resolution", type=float, default=30.0, help="Initial resolution (Angstrom)")
    parser.add_argument("--image_batch_size", type=int, default=500, help="Images per GPU batch")
    parser.add_argument("--rotation_block_size", type=int, default=5000, help="Rotations per block")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for half-set split")
    parser.add_argument(
        "--relion_half_sets",
        default=None,
        help="Path to a RELION data STAR file with rlnRandomSubset column. "
             "If given, use RELION's half-set assignments instead of random seed.",
    )
    parser.add_argument(
        "--relion_current_sizes",
        default=None,
        help="Comma-separated list of per-iteration current_sizes from RELION "
             "(oracle mode). Example: '0,56,30,50,70,98,98,92,88,90'",
    )
    args = parser.parse_args()

    # Verify GPU
    devices = jax.devices()
    logger.info("JAX devices: %s", devices)
    if not any(d.platform == "gpu" for d in devices):
        logger.error("No GPU available. Aborting.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # ---- Load dataset ----
    logger.info("Loading dataset from %s", args.data_dir)
    from recovar.data_io.cryoem_dataset import load_dataset

    ds = load_dataset(
        os.path.join(args.data_dir, "particles.star"),
        lazy=False,
    )
    logger.info("Dataset: %d images, image_shape=%s, voxel_size=%.3f A/px",
                ds.n_units, ds.image_shape, ds.voxel_size)

    # ---- Create half-sets ----
    n_images = ds.n_units

    if args.relion_half_sets is not None:
        # Use RELION's half-set split from rlnRandomSubset
        logger.info("Loading RELION half-set assignments from %s", args.relion_half_sets)
        import re
        import starfile as _starfile

        relion_data = _starfile.read(args.relion_half_sets)
        relion_particles = relion_data['particles']
        relion_subsets = np.array(relion_particles['rlnRandomSubset'])
        relion_names = list(relion_particles['rlnImageName'])

        # Build mapping: particle stack index -> subset
        def _image_name_to_stack_idx(name):
            m = re.match(r'(\d+)@', name)
            return int(m.group(1)) if m else -1

        relion_idx_to_subset = {}
        for i in range(len(relion_names)):
            stack_idx = _image_name_to_stack_idx(relion_names[i])
            relion_idx_to_subset[stack_idx] = relion_subsets[i]

        # Our dataset loads in stack order 1,2,3,...
        # Map to RELION's subset assignments
        our_star = _starfile.read(os.path.join(args.data_dir, "particles.star"))
        our_particles = our_star['particles'] if isinstance(our_star, dict) else our_star
        our_names = list(our_particles['rlnImageName'])
        our_subsets = np.array([
            relion_idx_to_subset[_image_name_to_stack_idx(name)]
            for name in our_names
        ])

        half1_idx = np.where(our_subsets == 1)[0]
        half2_idx = np.where(our_subsets == 2)[0]
        logger.info("Using RELION half-set split: %d (subset=1) + %d (subset=2)",
                    len(half1_idx), len(half2_idx))
    else:
        indices = np.arange(n_images)
        rng = np.random.RandomState(args.seed)
        rng.shuffle(indices)
        half1_idx = np.sort(indices[:n_images // 2])
        half2_idx = np.sort(indices[n_images // 2:])

    ds_half1 = ds.subset(half1_idx)
    ds_half2 = ds.subset(half2_idx)
    logger.info("Half-sets: %d + %d images", ds_half1.n_units, ds_half2.n_units)

    # ---- Load initial volume ----
    init_mrc_path = os.path.join(args.data_dir, "reference_init.mrc")
    with mrcfile.open(init_mrc_path, mode="r") as mrc:
        init_vol_real = np.array(mrc.data, dtype=np.float32)
    assert init_vol_real.shape == ds.volume_shape, (
        f"Volume shape mismatch: {init_vol_real.shape} vs {ds.volume_shape}"
    )
    # Convert to Fourier space (matching recovar convention)
    init_vol_ft = np.fft.fftn(np.fft.ifftshift(init_vol_real)).astype(np.complex64).reshape(-1)
    logger.info("Initial volume loaded: shape=%s", init_vol_real.shape)

    # ---- Set up rotation and translation grids ----
    from recovar.em.sampling import get_rotation_grid, get_translation_grid

    rotations = get_rotation_grid(args.healpix_order, matrices=True).astype(np.float32)
    translations = get_translation_grid(args.offset_range, args.offset_step).astype(np.float32)
    logger.info("Rotation grid: %d rotations (healpix_order=%d)",
                rotations.shape[0], args.healpix_order)
    logger.info("Translation grid: %d translations (range=%.1f, step=%.1f)",
                translations.shape[0], args.offset_range, args.offset_step)

    # ---- Initialize noise and prior ----
    # Start with flat noise and weak prior (will be updated after first iteration)
    image_size = ds.image_size
    volume_size = ds.volume_size

    # Use a reasonable initial noise: 1.0 per pixel (flat)
    noise_variance = jnp.ones(image_size, dtype=jnp.float32)

    # Compute initial signal prior from init volume (weak prior)
    from recovar.reconstruction.regularization import average_over_shells
    init_PS = average_over_shells(jnp.abs(jnp.asarray(init_vol_ft)) ** 2, ds.volume_shape)
    from recovar import utils
    init_prior = utils.make_radial_image(init_PS, ds.volume_shape, extend_last_frequency=True)
    # Scale by a factor to provide regularization without being too strong
    mean_variance = jnp.asarray(init_prior * 0.5 + jnp.max(init_prior) * 1e-4)

    # Compute initial current_size from init_resolution
    init_current_size = max(32, int(2 * ds.voxel_size * ds.grid_size / args.init_resolution))
    logger.info("Initial current_size from resolution %.1f A: %d pixels",
                args.init_resolution, init_current_size)

    # ---- Run refinement ----
    from recovar.em.dense_single_volume.refine import refine_single_volume

    experiment_datasets = [ds_half1, ds_half2]
    translations_jnp = jnp.asarray(translations)

    logger.info("=" * 70)
    logger.info("Starting refinement: max_iter=%d, adaptive_oversampling=%d",
                args.max_iter, args.adaptive_oversampling)
    logger.info("=" * 70)

    # Parse oracle current_sizes if provided
    oracle_current_sizes = None
    if args.relion_current_sizes is not None:
        oracle_current_sizes = [int(x) for x in args.relion_current_sizes.split(",")]
        logger.info("Oracle mode: using RELION current_sizes=%s", oracle_current_sizes)

    t_start = time.time()

    result = refine_single_volume(
        experiment_datasets=experiment_datasets,
        init_volume=jnp.asarray(init_vol_ft),
        init_noise_variance=noise_variance,
        init_mean_variance=mean_variance,
        rotations=rotations,
        translations=translations_jnp,
        disc_type="linear_interp",
        max_iter=args.max_iter,
        image_batch_size=args.image_batch_size,
        rotation_block_size=args.rotation_block_size,
        relion_current_sizes=oracle_current_sizes,
        init_current_size=init_current_size,
        fsc_threshold=1.0 / 7.0,
        adaptive_oversampling=args.adaptive_oversampling,
        adaptive_fraction=args.adaptive_fraction,
        max_significants=args.max_significants,
        nside_level=args.healpix_order if args.adaptive_oversampling > 0 else None,
        translation_pixel_offset=args.offset_step if args.adaptive_oversampling > 0 else None,
    )

    total_time = time.time() - t_start
    logger.info("=" * 70)
    logger.info("Refinement complete in %.1fs (%d iterations)", total_time, args.max_iter)
    logger.info("=" * 70)

    # ---- Save results ----
    save_dict = {
        "current_sizes": np.array(result["current_sizes"]),
        "pixel_resolutions": np.array(result["pixel_resolutions"]),
        "wall_times": np.array(result["wall_times"]),
        "total_time": total_time,
        "n_iterations": args.max_iter,
        "healpix_order": args.healpix_order,
        "n_rotations": rotations.shape[0],
        "n_translations": translations.shape[0],
        "n_images": n_images,
        "image_shape": np.array(ds.image_shape),
        "volume_shape": np.array(ds.volume_shape),
        "voxel_size": ds.voxel_size,
        "adaptive_oversampling": args.adaptive_oversampling,
        "adaptive_fraction": args.adaptive_fraction,
        "max_significants": args.max_significants,
        "half1_indices": half1_idx,
        "half2_indices": half2_idx,
    }

    # Save FSC curves per iteration
    for i, fsc in enumerate(result["fsc_history"]):
        save_dict[f"fsc_iter_{i:03d}"] = np.asarray(fsc)

    # Save significant counts per iteration (if available)
    for i, counts in enumerate(result["significant_counts"]):
        if counts is not None:
            save_dict[f"sig_counts_iter_{i:03d}"] = np.asarray(counts)

    # Save final merged volume (Fourier space)
    save_dict["final_mean_ft"] = np.asarray(result["mean"])

    # Save per-half-set means
    for k in range(2):
        save_dict[f"half{k}_mean_ft"] = np.asarray(result["means"][k])

    # Save hard assignments
    for k in range(2):
        if result["hard_assignments"][k] is not None:
            save_dict[f"hard_assignments_half{k}"] = np.asarray(result["hard_assignments"][k])

    out_path = os.path.join(args.output, "refinement_results.npz")
    np.savez_compressed(out_path, **save_dict)
    logger.info("Results saved to %s", out_path)

    # Also save final merged volume as MRC for visual inspection
    final_mean_ft = np.asarray(result["mean"]).reshape(ds.volume_shape)
    final_mean_real = np.fft.fftshift(np.real(np.fft.ifftn(final_mean_ft))).astype(np.float32)
    mrc_path = os.path.join(args.output, "final_merged.mrc")
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(final_mean_real)
        mrc.voxel_size = ds.voxel_size
    logger.info("Final merged volume saved to %s", mrc_path)

    # Save per-half volumes as MRC
    for k in range(2):
        half_ft = np.asarray(result["means"][k]).reshape(ds.volume_shape)
        half_real = np.fft.fftshift(np.real(np.fft.ifftn(half_ft))).astype(np.float32)
        half_mrc_path = os.path.join(args.output, f"final_half{k+1}.mrc")
        with mrcfile.new(half_mrc_path, overwrite=True) as mrc:
            mrc.set_data(half_real)
            mrc.voxel_size = ds.voxel_size
        logger.info("Half-%d volume saved to %s", k + 1, half_mrc_path)

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("REFINEMENT SUMMARY")
    print("=" * 70)
    print(f"{'Iter':>4s}  {'CurSize':>8s}  {'PixRes':>8s}  {'ResA':>8s}  {'Time(s)':>8s}", end="")
    if any(c is not None for c in result["significant_counts"]):
        print(f"  {'MedSig':>8s}", end="")
    print()
    print("-" * 70)

    for i in range(len(result["current_sizes"])):
        cs = result["current_sizes"][i]
        pr = result["pixel_resolutions"][i]
        res_a = pr / ds.voxel_size if ds.voxel_size > 0 else pr
        wt = result["wall_times"][i]
        line = f"{i+1:4d}  {cs:8d}  {pr:8.1f}  {res_a:8.2f}  {wt:8.1f}"
        if result["significant_counts"][i] is not None:
            med_sig = int(np.median(np.asarray(result["significant_counts"][i])))
            line += f"  {med_sig:8d}"
        print(line)

    print("-" * 70)
    print(f"Total wall time: {total_time:.1f}s")
    print(f"Final current_size: {result['current_sizes'][-1]}")
    print(f"Final pixel resolution: {result['pixel_resolutions'][-1]:.1f}")
    print(f"Final resolution: {result['pixel_resolutions'][-1] / ds.voxel_size:.2f} A")
    print("=" * 70)


if __name__ == "__main__":
    main()
