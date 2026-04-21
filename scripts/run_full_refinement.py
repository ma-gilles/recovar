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
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils as ftu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _shell_index_to_resolution_angstrom(shell_index, grid_size, voxel_size):
    if voxel_size <= 0:
        return float(shell_index)
    shell_index = float(shell_index)
    if shell_index <= 0:
        return float("inf")
    return float(grid_size) * float(voxel_size) / shell_index


def _load_relion_mask_params(optimiser_star_path):
    """Extract RELION image-mask parameters from an optimiser STAR file."""
    text = Path(optimiser_star_path).read_text(errors="ignore")

    particle_match = re.search(r"rlnParticleDiameter\s+([0-9]+(?:\.[0-9]+)?)", text)
    if particle_match is None:
        particle_match = re.search(r"particle_diameter\s+([0-9]+(?:\.[0-9]+)?)", text)

    width_match = re.search(r"rlnWidthMaskEdge\s+([0-9]+(?:\.[0-9]+)?)", text)
    if width_match is None:
        width_match = re.search(r"width_mask_edge\s+([0-9]+(?:\.[0-9]+)?)", text)

    if particle_match is None or width_match is None:
        return None

    return float(particle_match.group(1)), float(width_match.group(1))


def _load_relion_max_significants(optimiser_star_path):
    """Extract RELION's maximum-significant-poses setting from an optimiser STAR."""
    text = Path(optimiser_star_path).read_text(errors="ignore")

    match = re.search(r"rlnMaximumSignificantPoses\s+(-?[0-9]+)", text)
    if match is None:
        match = re.search(r"maximum_significant_poses\s+(-?[0-9]+)", text)
    if match is None:
        return None
    return int(match.group(1))


def _find_relion_optimiser_star(args):
    candidates = []
    if args.relion_half_sets is not None:
        candidates.append(Path(args.relion_half_sets).resolve().parent / "run_optimiser.star")
    candidates.append(Path(args.data_dir) / "relion_ref" / "run_optimiser.star")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _maybe_apply_relion_image_mask(ds, args):
    """Override the dataset scoring mask with RELION's particle-diameter mask."""
    if args.mode != "relion":
        return None

    optimiser_star = _find_relion_optimiser_star(args)
    if optimiser_star is None:
        logger.info("RELION optimiser STAR not found; keeping dataset image mask")
        return None

    params = _load_relion_mask_params(optimiser_star)
    if params is None:
        logger.info("No RELION mask parameters found in %s; keeping dataset image mask", optimiser_star)
        return None

    from recovar.core import mask as core_mask

    particle_diameter_ang, width_mask_edge_px = params
    relion_mask = core_mask.relion_soft_image_mask(
        image_size=ds.image_shape[0],
        pixel_size=ds.voxel_size,
        particle_diameter_ang=particle_diameter_ang,
        width_mask_edge_px=width_mask_edge_px,
    )

    ds.image_source.backend.image_mask = relion_mask
    if hasattr(ds.image_source, "image_mask"):
        ds.image_source.image_mask = relion_mask

    radius_px = particle_diameter_ang / (2.0 * ds.voxel_size)
    logger.info(
        "Applied RELION scoring mask from %s: particle_diameter=%.1f A, width_mask_edge=%.1f px, radius=%.2f px",
        optimiser_star,
        particle_diameter_ang,
        width_mask_edge_px,
        radius_px,
    )
    return params


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
    parser.add_argument(
        "--mode",
        choices=["legacy", "relion"],
        default="relion",
        help="Refinement mode to run. Use 'relion' for the parity path.",
    )
    parser.add_argument(
        "--healpix_order",
        type=int,
        default=3,
        help="HEALPix order for the evaluated orientation grid. In RELION mode "
        "this is the finest order; the coarse pass-1 order is "
        "healpix_order - adaptive_oversampling.",
    )
    parser.add_argument("--offset_range", type=float, default=3.0, help="Translation search range (pixels)")
    parser.add_argument("--offset_step", type=float, default=1.0, help="Translation step (pixels)")
    parser.add_argument(
        "--offset_sigma_angstrom",
        type=float,
        default=10.0,
        help="RELION-style Gaussian translation-prior sigma in Angstrom.",
    )
    parser.add_argument("--adaptive_oversampling", type=int, default=1, help="Oversampling levels (0=off, 1=2x)")
    parser.add_argument("--adaptive_fraction", type=float, default=0.999, help="Significance fraction")
    parser.add_argument(
        "--max_significants",
        type=int,
        default=None,
        help="Max significant samples per image. Use <=0 for RELION-style uncapped mode. "
        "If omitted in RELION mode, read _rlnMaximumSignificantPoses from the optimiser STAR.",
    )
    parser.add_argument(
        "--adaptive_skip_threshold",
        type=float,
        default=0.5,
        help="Skip adaptive pass 2 when the mean significant-sample fraction "
        "is at least this value. Use a negative value to disable the shortcut.",
    )
    parser.add_argument(
        "--tau2_fudge",
        type=float,
        default=4.0,
        help="RELION tau2_fudge regularization strength (default 4.0, "
        "matching RELION's 3D auto-refine default at "
        "ml_optimiser.cpp:~1070 `tau2_fudge_factor = 4`). "
        "Higher values produce smoother volumes (stronger prior).",
    )
    parser.add_argument(
        "--perturb_factor",
        type=float,
        default=0.5,
        help="RELION SamplingPerturbation factor (default 0.5 matching "
        "RELION GUI `--perturb 0.5`). Applies a per-iter random rigid "
        "rotation of the SO(3) trial grid and translation shift, ported "
        "from healpix_sampling.cpp:167-174 / 1909-1934 / 1810-1820. "
        "Set to 0 to disable.",
    )
    parser.add_argument(
        "--perturb_seed",
        type=int,
        default=None,
        help="Optional deterministic seed for the SamplingPerturbation RNG. "
        "If unset, uses np.random.default_rng() (non-reproducible).",
    )
    parser.add_argument("--init_resolution", type=float, default=30.0, help="Initial resolution (Angstrom)")
    parser.add_argument("--image_batch_size", type=int, default=500, help="Images per GPU batch")
    parser.add_argument(
        "--rotation_block_size",
        type=int,
        default=40000,
        help="Rotations per block (larger = faster, less Python overhead)",
    )
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
    relion_mask_params = _maybe_apply_relion_image_mask(ds, args)
    particle_diameter_ang = None if relion_mask_params is None else float(relion_mask_params[0])
    logger.info("Dataset: %d images, image_shape=%s, voxel_size=%.3f A/px", ds.n_units, ds.image_shape, ds.voxel_size)

    # ---- Create half-sets ----
    n_images = ds.n_units

    if args.relion_half_sets is not None:
        # Use RELION's half-set split from rlnRandomSubset
        logger.info("Loading RELION half-set assignments from %s", args.relion_half_sets)
        import re

        import starfile as _starfile

        relion_data = _starfile.read(args.relion_half_sets)
        relion_particles = relion_data["particles"]
        relion_subsets = np.array(relion_particles["rlnRandomSubset"])
        relion_names = list(relion_particles["rlnImageName"])

        # Build mapping: particle stack index -> subset
        def _image_name_to_stack_idx(name):
            m = re.match(r"(\d+)@", name)
            return int(m.group(1)) if m else -1

        relion_idx_to_subset = {}
        for i in range(len(relion_names)):
            stack_idx = _image_name_to_stack_idx(relion_names[i])
            relion_idx_to_subset[stack_idx] = relion_subsets[i]

        # Our dataset loads in stack order 1,2,3,...
        # Map to RELION's subset assignments
        our_star = _starfile.read(os.path.join(args.data_dir, "particles.star"))
        our_particles = our_star["particles"] if isinstance(our_star, dict) else our_star
        our_names = list(our_particles["rlnImageName"])
        our_subsets = np.array([relion_idx_to_subset[_image_name_to_stack_idx(name)] for name in our_names])

        half1_idx = np.where(our_subsets == 1)[0]
        half2_idx = np.where(our_subsets == 2)[0]
        logger.info("Using RELION half-set split: %d (subset=1) + %d (subset=2)", len(half1_idx), len(half2_idx))
    else:
        indices = np.arange(n_images)
        rng = np.random.RandomState(args.seed)
        rng.shuffle(indices)
        half1_idx = np.sort(indices[: n_images // 2])
        half2_idx = np.sort(indices[n_images // 2 :])

    ds_half1 = ds.subset(half1_idx)
    ds_half2 = ds.subset(half2_idx)
    logger.info("Half-sets: %d + %d images", ds_half1.n_units, ds_half2.n_units)

    optimiser_star = _find_relion_optimiser_star(args)
    if args.mode == "relion" and args.max_significants is None and optimiser_star is not None:
        relion_max_significants = _load_relion_max_significants(optimiser_star)
        if relion_max_significants is not None:
            args.max_significants = relion_max_significants
            logger.info(
                "Using RELION max_significants from %s: %d",
                optimiser_star,
                args.max_significants,
            )
    if args.max_significants is None:
        args.max_significants = 500

    # ---- Load initial volume ----
    # CANONICAL recovar idiom for loading a volume: load_mrc + get_dft3.
    # See recovar/output/output.py:980-984 and recovar/simulation/simulator.py:425.
    # NEVER use raw `mrcfile.open` + `np.fft.fftn(np.fft.ifftshift(...))` here:
    # that produces a Fourier volume with the right values but at WRONG array
    # indices (DC at corner instead of center), so `slice_volume` reads
    # Nyquist as if it were DC and projections are off by ~2400x in amplitude
    # at low frequencies.
    from recovar.utils.helpers import load_mrc as _load_mrc

    init_mrc_path = os.path.join(args.data_dir, "reference_init.mrc")
    init_vol_real = _load_mrc(init_mrc_path).astype(np.float32)
    assert init_vol_real.shape == ds.volume_shape, f"Volume shape mismatch: {init_vol_real.shape} vs {ds.volume_shape}"
    # Convert to centered Fourier space using the proper helper.
    init_vol_ft = np.array(ftu.get_dft3(jnp.asarray(init_vol_real))).astype(np.complex64).reshape(-1)
    logger.info("Initial volume loaded: shape=%s", init_vol_real.shape)

    # ---- Set up rotation and translation grids ----
    from recovar.em.sampling import get_rotation_grid, get_translation_grid

    if args.mode == "relion":
        init_healpix_order = max(args.healpix_order - args.adaptive_oversampling, 0)
        rotation_grid_order = init_healpix_order
        logger.info(
            "RELION grid orders: coarse=%d, finest=%d (adaptive_oversampling=%d)",
            init_healpix_order,
            args.healpix_order,
            args.adaptive_oversampling,
        )
    else:
        init_healpix_order = args.healpix_order
        rotation_grid_order = args.healpix_order

    rotations = get_rotation_grid(rotation_grid_order, matrices=True).astype(np.float32)
    translations = get_translation_grid(args.offset_range, args.offset_step).astype(np.float32)
    logger.info("Rotation grid: %d rotations (healpix_order=%d)", rotations.shape[0], rotation_grid_order)
    logger.info(
        "Translation grid: %d translations (range=%.1f, step=%.1f)",
        translations.shape[0],
        args.offset_range,
        args.offset_step,
    )

    # ---- Initialize noise and prior ----
    # Use a RELION-style initial sigma2 estimate from particle power spectra
    # instead of a flat unit spectrum, so iteration 1 starts on a comparable
    # likelihood scale.
    image_size = ds.image_size
    volume_size = ds.volume_size

    from recovar.reconstruction import noise as recon_noise

    initial_noise_subset = np.arange(min(1000, ds.n_units), dtype=np.int32)
    # In RELION mode the E-step scores masked images, so the bootstrap noise
    # MUST come from masked images too — otherwise sigma2 is dominated by the
    # solvent area and the iter-1 chi² is ~3.3-6× too small (verified
    # 2026-04-08 against the tiny parity dataset, see tmp/check_sigma2_mask.py).
    bootstrap_apply_mask = args.mode == "relion"
    initial_noise_radial = recon_noise.estimate_initial_noise_spectrum_from_unaligned_images(
        ds,
        initial_noise_subset,
        batch_size=min(args.image_batch_size, initial_noise_subset.size),
        apply_image_mask=bootstrap_apply_mask,
    )
    noise_variance = recon_noise.make_radial_noise(initial_noise_radial, ds.image_shape)
    logger.info(
        "Initial sigma2_noise estimate from %d images: min=%.3e median=%.3e max=%.3e",
        initial_noise_subset.size,
        float(np.min(np.asarray(initial_noise_radial))),
        float(np.median(np.asarray(initial_noise_radial))),
        float(np.max(np.asarray(initial_noise_radial))),
    )

    # Compute initial signal prior from init volume (weak prior)
    from recovar.reconstruction.regularization import average_over_shells

    init_PS = average_over_shells(jnp.abs(jnp.asarray(init_vol_ft)) ** 2, ds.volume_shape)
    from recovar import utils

    init_prior = utils.make_radial_image(init_PS, ds.volume_shape, extend_last_frequency=True)
    # Scale by a factor to provide regularization without being too strong
    mean_variance = jnp.asarray(init_prior * 0.5 + jnp.max(init_prior) * 1e-4)

    # Compute initial current_size from init_resolution
    init_current_size = max(32, int(2 * ds.voxel_size * ds.grid_size / args.init_resolution))
    logger.info("Initial current_size from resolution %.1f A: %d pixels", args.init_resolution, init_current_size)

    # ---- Run refinement ----
    from recovar.em.dense_single_volume.refine import refine_single_volume

    experiment_datasets = [ds_half1, ds_half2]
    translations_jnp = jnp.asarray(translations)

    logger.info("=" * 70)
    logger.info(
        "Starting refinement: mode=%s, max_iter=%d, adaptive_oversampling=%d",
        args.mode,
        args.max_iter,
        args.adaptive_oversampling,
    )
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
        mode=args.mode,
        max_iter=args.max_iter,
        image_batch_size=args.image_batch_size,
        rotation_block_size=args.rotation_block_size,
        relion_current_sizes=oracle_current_sizes,
        init_current_size=init_current_size,
        fsc_threshold=1.0 / 7.0,
        adaptive_oversampling=args.adaptive_oversampling,
        adaptive_fraction=args.adaptive_fraction,
        max_significants=args.max_significants,
        adaptive_pass2_skip_threshold=args.adaptive_skip_threshold,
        nside_level=rotation_grid_order if args.adaptive_oversampling > 0 else None,
        translation_pixel_offset=args.offset_step if args.adaptive_oversampling > 0 else None,
        init_healpix_order=init_healpix_order,
        init_translation_sigma_angstrom=args.offset_sigma_angstrom,
        particle_diameter_ang=particle_diameter_ang,
        tau2_fudge=args.tau2_fudge,
        perturb_factor=args.perturb_factor,
        perturb_seed=args.perturb_seed,
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
        "coarse_healpix_order": init_healpix_order,
        "n_rotations": rotations.shape[0],
        "n_translations": translations.shape[0],
        "n_images": n_images,
        "image_shape": np.array(ds.image_shape),
        "volume_shape": np.array(ds.volume_shape),
        "voxel_size": ds.voxel_size,
        "adaptive_oversampling": args.adaptive_oversampling,
        "adaptive_fraction": args.adaptive_fraction,
        "max_significants": args.max_significants,
        "adaptive_skip_threshold": args.adaptive_skip_threshold,
        "offset_sigma_angstrom": args.offset_sigma_angstrom,
        "particle_diameter_ang": (np.float64(particle_diameter_ang) if particle_diameter_ang is not None else np.nan),
        "half1_indices": half1_idx,
        "half2_indices": half2_idx,
    }

    if "healpix_order_trajectory" in result:
        save_dict["healpix_order_trajectory"] = np.asarray(
            result["healpix_order_trajectory"],
            dtype=np.int32,
        )
    if "ave_Pmax_trajectory" in result:
        save_dict["ave_Pmax_trajectory"] = np.asarray(
            result["ave_Pmax_trajectory"],
            dtype=np.float64,
        )
    if "sigma_offset_trajectory" in result:
        save_dict["sigma_offset_trajectory"] = np.asarray(
            result["sigma_offset_trajectory"],
            dtype=np.float64,
        )
    if "sigma_offset_used_trajectory" in result:
        save_dict["sigma_offset_used_trajectory"] = np.asarray(
            result["sigma_offset_used_trajectory"],
            dtype=np.float64,
        )
    if "convergence_state" in result:
        state = result["convergence_state"]
        save_dict["convergence_iteration"] = np.int32(state.iteration)
        save_dict["convergence_current_resolution"] = np.float64(state.current_resolution)
        save_dict["convergence_ave_Pmax"] = np.float64(state.ave_Pmax)
        save_dict["convergence_healpix_order"] = np.int32(state.healpix_order)
        save_dict["convergence_has_converged"] = np.bool_(state.has_converged)

    # Save FSC curves per iteration
    for i, fsc in enumerate(result["fsc_history"]):
        save_dict[f"fsc_iter_{i:03d}"] = np.asarray(fsc)

    # Save significant counts per iteration (if available)
    for i, counts in enumerate(result["significant_counts"]):
        if counts is not None:
            save_dict[f"sig_counts_iter_{i:03d}"] = np.asarray(counts)

    if "data_vs_prior_trajectory" in result:
        for i, dvp in enumerate(result["data_vs_prior_trajectory"]):
            save_dict[f"data_vs_prior_iter_{i:03d}"] = np.asarray(dvp)

    # Per-iter per-shell sigma2_noise and tau2 (added 2026-04 for RELION parity diff)
    if "noise_radial_trajectory" in result:
        for i, nr in enumerate(result["noise_radial_trajectory"]):
            if nr is not None:
                save_dict[f"noise_radial_iter_{i:03d}"] = np.asarray(nr, dtype=np.float64)
    if "tau2_radial_trajectory" in result:
        for i, t2 in enumerate(result["tau2_radial_trajectory"]):
            if t2 is not None:
                save_dict[f"tau2_radial_iter_{i:03d}"] = np.asarray(t2, dtype=np.float64)
    for result_key, prefix in [
        ("tau2_sigma2_trajectory", "tau2_sigma2_iter"),
        ("tau2_avg_weight_trajectory", "tau2_avg_weight_iter"),
        ("tau2_shell_sum_trajectory", "tau2_shell_sum_iter"),
        ("tau2_shell_count_trajectory", "tau2_shell_count_iter"),
        ("tau2_fsc_used_trajectory", "tau2_fsc_used_iter"),
        ("tau2_ssnr_trajectory", "tau2_ssnr_iter"),
    ]:
        if result_key in result:
            for i, arr in enumerate(result[result_key]):
                if arr is not None:
                    save_dict[f"{prefix}_{i:03d}"] = np.asarray(arr, dtype=np.float64)

    # Save per-image Pmax per iteration (if available)
    if "pmax_per_image_history" in result:
        for i, pmax in enumerate(result["pmax_per_image_history"]):
            save_dict[f"pmax_per_image_iter_{i:03d}"] = np.asarray(pmax, dtype=np.float32)

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

    # Also save final merged volume as MRC for visual inspection.
    # Use the canonical idiom: get_idft3 + write_mrc (handles axis transpose).
    from recovar.utils.helpers import write_mrc as _write_mrc

    final_mean_ft = np.asarray(result["mean"]).reshape(ds.volume_shape)
    final_mean_real = np.real(np.array(ftu.get_idft3(jnp.asarray(final_mean_ft)))).astype(np.float32)
    mrc_path = os.path.join(args.output, "final_merged.mrc")
    _write_mrc(mrc_path, final_mean_real, voxel_size=ds.voxel_size)
    logger.info("Final merged volume saved to %s", mrc_path)

    # Save per-half volumes as MRC
    for k in range(2):
        half_ft = np.asarray(result["means"][k]).reshape(ds.volume_shape)
        half_real = np.real(np.array(ftu.get_idft3(jnp.asarray(half_ft)))).astype(np.float32)
        half_mrc_path = os.path.join(args.output, f"final_half{k + 1}.mrc")
        _write_mrc(half_mrc_path, half_real, voxel_size=ds.voxel_size)
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
        res_a = _shell_index_to_resolution_angstrom(pr, ds.image_shape[0], ds.voxel_size)
        wt = result["wall_times"][i]
        line = f"{i + 1:4d}  {cs:8d}  {pr:8.1f}  {res_a:8.2f}  {wt:8.1f}"
        if result["significant_counts"][i] is not None:
            med_sig = int(np.median(np.asarray(result["significant_counts"][i])))
            line += f"  {med_sig:8d}"
        print(line)

    print("-" * 70)
    print(f"Total wall time: {total_time:.1f}s")
    print(f"Final current_size: {result['current_sizes'][-1]}")
    print(f"Final pixel resolution: {result['pixel_resolutions'][-1]:.1f}")
    print(
        "Final resolution: "
        f"{_shell_index_to_resolution_angstrom(result['pixel_resolutions'][-1], ds.image_shape[0], ds.voxel_size):.2f} A"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
