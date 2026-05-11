#!/usr/bin/env python
"""Head-to-head comparison of our EM refinement vs RELION.

Runs two comparison cases:
  1. Same half-sets as RELION, own FSC->current_size logic
  2. Same half-sets + oracle current_sizes from RELION (isolates quality)

Then compares per-iteration: current_size, wall time, resolution, and
cross-FSC between matched half-maps.

Usage:
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        pixi run python scripts/run_comparison.py [--max_iter 10]

All output written to OUTPUT_DIR and a markdown summary to stdout.
"""

import argparse
import datetime
import logging
import os
import re
import sys
import time

import jax
import jax.numpy as jnp
import mrcfile
import numpy as np
import starfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (defaults for the benchmark dataset on Della)
# ---------------------------------------------------------------------------
DATA_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data"
RELION_REF = os.path.join(DATA_DIR, "relion_ref")
OUTPUT_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/comparison_results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def image_name_to_stack_idx(name):
    m = re.match(r"(\d+)@", name)
    return int(m.group(1)) if m else -1


def _resolve_relion_ref_dir(data_dir, explicit_dir=None):
    """Return the RELION reference directory for a benchmark dataset."""
    if explicit_dir:
        return explicit_dir

    candidates = [
        os.path.join(data_dir, "relion_ref"),
        os.path.join(data_dir, "relion_ref_benchmark"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "run_it001_data.star")):
            return candidate
    return candidates[0]


def load_relion_half_sets(relion_star_path, our_star_path):
    """Load RELION's rlnRandomSubset and map to our particle ordering.

    Returns (half1_indices, half2_indices) as 0-based numpy arrays in
    our dataset's ordering.
    """
    rd = starfile.read(relion_star_path)
    rp = rd["particles"]
    relion_subsets = np.array(rp["rlnRandomSubset"])
    relion_names = list(rp["rlnImageName"])

    # Build mapping: stack_idx -> subset
    idx_to_subset = {}
    for i in range(len(relion_names)):
        idx_to_subset[image_name_to_stack_idx(relion_names[i])] = relion_subsets[i]

    # Our ordering
    od = starfile.read(our_star_path)
    op = od["particles"] if isinstance(od, dict) else od
    our_names = list(op["rlnImageName"])
    our_subsets = np.array([idx_to_subset[image_name_to_stack_idx(n)] for n in our_names])

    half1 = np.where(our_subsets == 1)[0]
    half2 = np.where(our_subsets == 2)[0]
    return half1, half2


def load_relion_per_iteration():
    """Load RELION's per-iteration metrics from model STAR files."""
    iterations = []
    for it in range(20):  # up to 20 iterations
        path = os.path.join(RELION_REF, f"run_it{it:03d}_half1_model.star")
        if not os.path.exists(path):
            break
        d = starfile.read(path)
        mg = d["model_general"]
        iterations.append(
            {
                "iteration": it,
                "current_size": int(mg["rlnCurrentImageSize"]),
                "resolution_A": float(mg["rlnCurrentResolution"]),
                "mtime": os.path.getmtime(path),
            }
        )
    # Compute elapsed times
    for i in range(len(iterations)):
        if i == 0:
            iterations[i]["elapsed_s"] = 0.0
        else:
            iterations[i]["elapsed_s"] = iterations[i]["mtime"] - iterations[i - 1]["mtime"]
    return iterations


def load_mrc_to_ft(path, volume_shape):
    """Load MRC file and convert to Fourier space."""
    if not os.path.exists(path):
        return None
    with mrcfile.open(path, mode="r") as mrc:
        vol_real = np.array(mrc.data, dtype=np.float32)
    if vol_real.shape != volume_shape:
        logger.warning("Volume shape mismatch: %s vs %s", vol_real.shape, volume_shape)
        return None
    vol_ft = np.fft.fftn(np.fft.ifftshift(vol_real)).astype(np.complex64).reshape(-1)
    return jnp.asarray(vol_ft)


def compute_fsc(vol1_ft, vol2_ft, volume_shape):
    """Compute FSC between two Fourier volumes."""
    from recovar.reconstruction.regularization import get_fsc_gpu

    return np.asarray(get_fsc_gpu(vol1_ft, vol2_ft, volume_shape))


def find_fsc_crossing(fsc, threshold):
    """Find first shell where FSC drops below threshold."""
    for i in range(1, len(fsc)):
        if fsc[i - 1] >= threshold and fsc[i] < threshold:
            return i
    return None


def run_refinement(
    ds,
    half1_idx,
    half2_idx,
    max_iter,
    output_subdir,
    oracle_current_sizes=None,
    adaptive_oversampling=0,
    healpix_order=3,
    offset_range=3.0,
    offset_step=1.0,
    max_significants=-1,
    offset_sigma_angstrom=10.0,
    image_batch_size=500,
    rotation_block_size=5000,
):
    """Run our refinement and return results dict + wall time."""
    from recovar import utils
    from recovar.em.dense_single_volume.iteration_loop import refine_single_volume
    from recovar.em.sampling import get_rotation_grid, get_translation_grid
    from recovar.reconstruction import noise as recon_noise
    from recovar.reconstruction.regularization import average_over_shells

    out_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    ds_half1 = ds.subset(half1_idx)
    ds_half2 = ds.subset(half2_idx)

    # Load init volume
    init_mrc_path = os.path.join(DATA_DIR, "reference_init.mrc")
    with mrcfile.open(init_mrc_path, mode="r") as mrc:
        init_vol_real = np.array(mrc.data, dtype=np.float32)
    init_vol_ft = np.fft.fftn(np.fft.ifftshift(init_vol_real)).astype(np.complex64).reshape(-1)

    # Rotation + translation grids
    rotations = get_rotation_grid(healpix_order, matrices=True).astype(np.float32)
    translations = get_translation_grid(offset_range, offset_step).astype(np.float32)

    # Initial noise and prior
    initial_noise_subset = np.arange(min(1000, ds.n_units), dtype=np.int32)
    initial_noise_radial = recon_noise.estimate_initial_noise_spectrum_from_unaligned_images(
        ds,
        initial_noise_subset,
        batch_size=min(image_batch_size, initial_noise_subset.size),
    )
    noise_variance = recon_noise.make_radial_noise(initial_noise_radial, ds.image_shape)
    init_PS = average_over_shells(jnp.abs(jnp.asarray(init_vol_ft)) ** 2, ds.volume_shape)
    init_prior = utils.make_radial_image(init_PS, ds.volume_shape, extend_last_frequency=True)
    mean_variance = jnp.asarray(init_prior * 0.5 + jnp.max(init_prior) * 1e-4)

    init_current_size = max(32, int(2 * ds.voxel_size * ds.grid_size / 30.0))

    t_start = time.time()

    result = refine_single_volume(
        experiment_datasets=[ds_half1, ds_half2],
        init_volume=jnp.asarray(init_vol_ft),
        init_noise_variance=noise_variance,
        init_mean_variance=mean_variance,
        rotations=rotations,
        translations=jnp.asarray(translations),
        disc_type="linear_interp",
        max_iter=max_iter,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        relion_current_sizes=oracle_current_sizes,
        init_current_size=init_current_size,
        fsc_threshold=1.0 / 7.0,
        adaptive_oversampling=adaptive_oversampling,
        max_significants=max_significants,
        init_translation_sigma_angstrom=offset_sigma_angstrom,
        nside_level=healpix_order if adaptive_oversampling > 0 else None,
        translation_pixel_offset=offset_step if adaptive_oversampling > 0 else None,
    )

    total_time = time.time() - t_start

    # Save per-half volumes as MRC
    for k in range(2):
        half_ft = np.asarray(result["means"][k]).reshape(ds.volume_shape)
        half_real = np.fft.fftshift(np.real(np.fft.ifftn(half_ft))).astype(np.float32)
        path = os.path.join(out_dir, f"half{k + 1}.mrc")
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(half_real)
            mrc.voxel_size = ds.voxel_size

    # Save merged volume
    merged_ft = np.asarray(result["mean"]).reshape(ds.volume_shape)
    merged_real = np.fft.fftshift(np.real(np.fft.ifftn(merged_ft))).astype(np.float32)
    path = os.path.join(out_dir, "merged.mrc")
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(merged_real)
        mrc.voxel_size = ds.voxel_size

    # Save results NPZ
    save_dict = {
        "current_sizes": np.array(result["current_sizes"]),
        "pixel_resolutions": np.array(result["pixel_resolutions"]),
        "wall_times": np.array(result["wall_times"]),
        "total_time": total_time,
    }
    for i, fsc in enumerate(result["fsc_history"]):
        save_dict[f"fsc_iter_{i:03d}"] = np.asarray(fsc)
    for k in range(2):
        save_dict[f"half{k}_mean_ft"] = np.asarray(result["means"][k])
    save_dict["final_mean_ft"] = np.asarray(result["mean"])
    np.savez_compressed(os.path.join(out_dir, "results.npz"), **save_dict)

    return result, total_time


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def main():
    global DATA_DIR, RELION_REF, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Head-to-head comparison vs RELION")
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument(
        "--relion_ref_dir",
        default=None,
        help="Path to the RELION refinement directory. If omitted, prefer "
        "<data_dir>/relion_ref when present, otherwise fall back to "
        "<data_dir>/relion_ref_benchmark.",
    )
    parser.add_argument("--adaptive_oversampling", type=int, default=0)
    parser.add_argument(
        "--max_significants",
        type=int,
        default=-1,
        help="Max significant samples per image. Use <=0 for RELION-style uncapped mode.",
    )
    parser.add_argument("--healpix_order", type=int, default=3)
    parser.add_argument("--offset_range", type=float, default=3.0)
    parser.add_argument("--offset_step", type=float, default=1.0)
    parser.add_argument("--offset_sigma_angstrom", type=float, default=10.0)
    parser.add_argument("--image_batch_size", type=int, default=500)
    parser.add_argument("--rotation_block_size", type=int, default=5000)
    parser.add_argument("--skip_run", action="store_true", help="Skip running refinement; load existing results")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    RELION_REF = _resolve_relion_ref_dir(DATA_DIR, args.relion_ref_dir)
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Resolved RELION reference directory: %s", RELION_REF)

    # Verify GPU
    devices = jax.devices()
    logger.info("JAX devices: %s", devices)
    if not any(d.platform == "gpu" for d in devices):
        logger.error("No GPU available. Aborting.")
        sys.exit(1)

    # ---- Load RELION reference data ----
    relion_iters = load_relion_per_iteration()
    n_relion = len(relion_iters)
    logger.info("Loaded %d RELION iterations", n_relion)

    relion_current_sizes = [ri["current_size"] for ri in relion_iters]
    relion_elapsed = [ri["elapsed_s"] for ri in relion_iters]
    relion_res_A = [ri["resolution_A"] for ri in relion_iters]

    # ---- Load half-set assignments ----
    relion_star = os.path.join(RELION_REF, "run_it001_data.star")
    our_star = os.path.join(DATA_DIR, "particles.star")
    half1_idx, half2_idx = load_relion_half_sets(relion_star, our_star)
    logger.info("RELION half-sets: %d + %d", len(half1_idx), len(half2_idx))

    if not args.skip_run:
        # ---- Load dataset ----
        from run_full_refinement import _maybe_apply_relion_image_mask

        from recovar.data_io.cryoem_dataset import load_dataset

        ds = load_dataset(our_star, lazy=False)

        class _Args:
            data_dir = DATA_DIR
            relion_half_sets = relion_star

        _maybe_apply_relion_image_mask(ds, _Args)
        volume_shape = ds.volume_shape
        voxel_size = ds.voxel_size

        # ---- Run 1: Own FSC->current_size, no adaptive oversampling ----
        logger.info("=" * 70)
        logger.info("RUN 1: Our FSC->current_size, same half-sets, no adaptive")
        logger.info("=" * 70)
        result1, time1 = run_refinement(
            ds,
            half1_idx,
            half2_idx,
            max_iter=args.max_iter,
            output_subdir="run1_own_fsc",
            oracle_current_sizes=None,
            adaptive_oversampling=args.adaptive_oversampling,
            max_significants=args.max_significants,
            healpix_order=args.healpix_order,
            offset_range=args.offset_range,
            offset_step=args.offset_step,
            offset_sigma_angstrom=args.offset_sigma_angstrom,
            image_batch_size=args.image_batch_size,
            rotation_block_size=args.rotation_block_size,
        )

        # ---- Run 2: Oracle current_sizes from RELION ----
        logger.info("=" * 70)
        logger.info("RUN 2: Oracle current_sizes from RELION, same half-sets")
        logger.info("=" * 70)
        result2, time2 = run_refinement(
            ds,
            half1_idx,
            half2_idx,
            max_iter=args.max_iter,
            output_subdir="run2_oracle_cs",
            oracle_current_sizes=relion_current_sizes[: args.max_iter],
            adaptive_oversampling=args.adaptive_oversampling,
            max_significants=args.max_significants,
            healpix_order=args.healpix_order,
            offset_range=args.offset_range,
            offset_step=args.offset_step,
            offset_sigma_angstrom=args.offset_sigma_angstrom,
            image_batch_size=args.image_batch_size,
            rotation_block_size=args.rotation_block_size,
        )
    else:
        # Load existing results
        from run_full_refinement import _maybe_apply_relion_image_mask

        from recovar.data_io.cryoem_dataset import load_dataset

        ds = load_dataset(our_star, lazy=False)

        class _Args:
            data_dir = DATA_DIR
            relion_half_sets = relion_star

        _maybe_apply_relion_image_mask(ds, _Args)
        volume_shape = ds.volume_shape
        voxel_size = ds.voxel_size
        result1 = _load_saved_results(os.path.join(OUTPUT_DIR, "run1_own_fsc"))
        result2 = _load_saved_results(os.path.join(OUTPUT_DIR, "run2_oracle_cs"))
        time1 = result1["total_time"] if result1 else 0
        time2 = result2["total_time"] if result2 else 0

    # ---- Cross-FSC: our half-maps vs RELION's half-maps (same images!) ----
    logger.info("Computing cross-FSC between matched half-maps...")
    cross_fsc_run1 = []
    cross_fsc_run2 = []
    relion_internal_fsc = []

    for it in range(min(args.max_iter, n_relion)):
        # RELION half-maps at this iteration
        r_h1_path = os.path.join(RELION_REF, f"run_it{it:03d}_half1_class001.mrc")
        r_h2_path = os.path.join(RELION_REF, f"run_it{it:03d}_half2_class001.mrc")
        r_h1_ft = load_mrc_to_ft(r_h1_path, volume_shape)
        r_h2_ft = load_mrc_to_ft(r_h2_path, volume_shape)

        if r_h1_ft is not None and r_h2_ft is not None:
            # RELION internal FSC
            fsc_relion = compute_fsc(r_h1_ft, r_h2_ft, volume_shape)
            relion_internal_fsc.append(fsc_relion)

            # Cross-FSC: our half1 vs RELION half1 (same images!)
            if result1 is not None:
                our_h1_ft = jnp.asarray(
                    result1["means"][0]
                    if isinstance(result1, dict) and "means" in result1
                    else _load_half_ft(os.path.join(OUTPUT_DIR, "run1_own_fsc"), 0)
                )
                cross_fsc_run1.append(compute_fsc(our_h1_ft, r_h1_ft, volume_shape))
            else:
                cross_fsc_run1.append(None)

            if result2 is not None:
                our_h1_ft = jnp.asarray(
                    result2["means"][0]
                    if isinstance(result2, dict) and "means" in result2
                    else _load_half_ft(os.path.join(OUTPUT_DIR, "run2_oracle_cs"), 0)
                )
                cross_fsc_run2.append(compute_fsc(our_h1_ft, r_h1_ft, volume_shape))
            else:
                cross_fsc_run2.append(None)
        else:
            relion_internal_fsc.append(None)
            cross_fsc_run1.append(None)
            cross_fsc_run2.append(None)

    # ---- Generate comparison report ----
    _print_report(
        result1,
        result2,
        time1,
        time2,
        relion_iters,
        relion_current_sizes,
        relion_elapsed,
        relion_res_A,
        cross_fsc_run1,
        cross_fsc_run2,
        relion_internal_fsc,
        volume_shape,
        voxel_size,
        args.max_iter,
    )


def _load_saved_results(run_dir):
    """Load previously saved results from NPZ."""
    path = os.path.join(run_dir, "results.npz")
    if not os.path.exists(path):
        logger.warning("Results not found: %s", path)
        return None
    data = dict(np.load(path, allow_pickle=True))
    # Reconstruct partial result dict
    return {
        "current_sizes": list(data["current_sizes"]),
        "pixel_resolutions": list(data["pixel_resolutions"]),
        "wall_times": list(data["wall_times"]),
        "total_time": float(data.get("total_time", 0)),
        "fsc_history": [data[k] for k in sorted(data.keys()) if k.startswith("fsc_iter_")],
        "means": [data.get("half0_mean_ft"), data.get("half1_mean_ft")],
    }


def _load_half_ft(run_dir, half_idx):
    """Load a saved half-map from NPZ."""
    path = os.path.join(run_dir, "results.npz")
    data = np.load(path, allow_pickle=True)
    return jnp.asarray(data[f"half{half_idx}_mean_ft"])


def _print_report(
    result1,
    result2,
    time1,
    time2,
    relion_iters,
    relion_cs,
    relion_elapsed,
    relion_res_A,
    cross_fsc_run1,
    cross_fsc_run2,
    relion_internal_fsc,
    volume_shape,
    voxel_size,
    max_iter,
):
    """Print the full comparison report to stdout."""
    n_relion = len(relion_iters)
    n_iter = max_iter

    # Extract our data
    r1_cs = list(result1["current_sizes"]) if result1 else []
    r1_pr = list(result1["pixel_resolutions"]) if result1 else []
    r1_wt = list(result1["wall_times"]) if result1 else []
    r2_cs = list(result2["current_sizes"]) if result2 else []
    r2_pr = list(result2["pixel_resolutions"]) if result2 else []
    r2_wt = list(result2["wall_times"]) if result2 else []

    print("# Head-to-Head Comparison: recovar EM vs RELION")
    print()
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Dataset: 5000 images, 128px, voxel_size={voxel_size:.3f} A/px")
    print("Same half-set split (RELION's rlnRandomSubset)")
    print()

    # ---- Table 1: Per-iteration comparison ----
    print("## Per-Iteration Comparison")
    print()
    print(
        "| Iter | RELION cs | Our cs (own) | Our cs (oracle) | "
        "RELION res (A) | Our res (A, own) | Our res (A, oracle) | "
        "RELION time (s) | Our time (s, own) | Our time (s, oracle) |"
    )
    print(
        "|------|-----------|--------------|-----------------|"
        "----------------|------------------|---------------------|-"
        "----------------|-------------------|----------------------|"
    )

    for i in range(max(n_iter, n_relion)):
        r_cs = relion_cs[i] if i < n_relion else ""
        r_res = f"{relion_res_A[i]:.2f}" if i < n_relion and relion_res_A[i] != float("inf") else "inf"
        r_time = f"{relion_elapsed[i]:.0f}" if i < n_relion else ""

        o1_cs = str(int(r1_cs[i])) if i < len(r1_cs) else ""
        o1_res = f"{float(r1_pr[i]) / voxel_size:.2f}" if i < len(r1_pr) else ""
        o1_time = f"{float(r1_wt[i]):.1f}" if i < len(r1_wt) else ""

        o2_cs = str(int(r2_cs[i])) if i < len(r2_cs) else ""
        o2_res = f"{float(r2_pr[i]) / voxel_size:.2f}" if i < len(r2_pr) else ""
        o2_time = f"{float(r2_wt[i]):.1f}" if i < len(r2_wt) else ""

        print(
            f"| {i + 1} | {r_cs} | {o1_cs} | {o2_cs} | "
            f"{r_res} | {o1_res} | {o2_res} | "
            f"{r_time} | {o1_time} | {o2_time} |"
        )

    # Totals
    relion_total = sum(relion_elapsed[1:]) if n_relion > 1 else 0  # skip iter 0
    print()
    print("**Total wall time:**")
    print(f"- RELION: {relion_total:.0f}s")
    if time1 > 0:
        print(f"- Ours (own FSC): {time1:.1f}s ({relion_total / time1:.1f}x speedup)")
    if time2 > 0:
        print(f"- Ours (oracle cs): {time2:.1f}s ({relion_total / time2:.1f}x speedup)")
    print()

    # ---- Table 2: Cross-FSC (matched half-maps) ----
    print("## Cross-FSC: Our Half-1 vs RELION Half-1 (Same Images)")
    print()
    print("This is meaningful because both codes reconstruct from the same images.")
    print()
    print("| Iter | Cross-FSC 0.5 (own) | Cross-FSC 0.5 (oracle) | RELION internal 0.143 | Our internal 0.143 (own) |")
    print("|------|---------------------|------------------------|-----------------------|--------------------------|")

    for i in range(min(n_iter, n_relion)):
        cf1 = find_fsc_crossing(cross_fsc_run1[i], 0.5) if cross_fsc_run1[i] is not None else None
        cf2 = find_fsc_crossing(cross_fsc_run2[i], 0.5) if cross_fsc_run2[i] is not None else None
        ri = find_fsc_crossing(relion_internal_fsc[i], 0.143) if relion_internal_fsc[i] is not None else None

        our_fsc = None
        if result1 and i < len(result1.get("fsc_history", [])):
            our_fsc = find_fsc_crossing(np.asarray(result1["fsc_history"][i]), 0.143)

        cf1_s = f"shell {cf1}" if cf1 is not None else "n/a"
        cf2_s = f"shell {cf2}" if cf2 is not None else "n/a"
        ri_s = f"shell {ri}" if ri is not None else "n/a"
        our_s = f"shell {our_fsc}" if our_fsc is not None else "n/a"

        print(f"| {i + 1} | {cf1_s} | {cf2_s} | {ri_s} | {our_s} |")

    print()

    # ---- Table 3: Final volume comparison ----
    print("## Final Volume Quality")
    print()

    # Cross-FSC between final merged volumes
    if result1 and result1.get("means"):
        our_merged = jnp.asarray(np.asarray(result1["means"][0]) + np.asarray(result1["means"][1])) / 2
        relion_merged_ft = load_mrc_to_ft(os.path.join(RELION_REF, "run_class001.mrc"), volume_shape)
        if relion_merged_ft is not None:
            fsc_merged = compute_fsc(our_merged, relion_merged_ft, volume_shape)
            crossing_5 = find_fsc_crossing(fsc_merged, 0.5)
            crossing_143 = find_fsc_crossing(fsc_merged, 0.143)
            print(f"Our merged vs RELION merged: FSC=0.5 at shell {crossing_5}, FSC=0.143 at shell {crossing_143}")

    # GT comparison
    gt_path = os.path.join(DATA_DIR, "reference_gt.mrc")
    gt_ft = load_mrc_to_ft(gt_path, volume_shape)
    if gt_ft is not None:
        if result1 and result1.get("means"):
            our_merged = jnp.asarray(np.asarray(result1["means"][0]) + np.asarray(result1["means"][1])) / 2
            fsc_gt_ours = compute_fsc(our_merged, gt_ft, volume_shape)
            gt_cross_ours = find_fsc_crossing(fsc_gt_ours, 0.5)
            print(f"Our merged vs GT: FSC=0.5 at shell {gt_cross_ours}")

        relion_merged_ft = load_mrc_to_ft(os.path.join(RELION_REF, "run_class001.mrc"), volume_shape)
        if relion_merged_ft is not None:
            fsc_gt_relion = compute_fsc(relion_merged_ft, gt_ft, volume_shape)
            gt_cross_relion = find_fsc_crossing(fsc_gt_relion, 0.5)
            print(f"RELION merged vs GT: FSC=0.5 at shell {gt_cross_relion}")

    print()

    # ---- Detailed cross-FSC at final iteration ----
    final_iter = min(n_iter, n_relion) - 1
    if final_iter >= 0 and cross_fsc_run1[final_iter] is not None:
        fsc = cross_fsc_run1[final_iter]
        print(f"## Detailed Cross-FSC at Iteration {final_iter + 1} (Our half-1 vs RELION half-1)")
        print()
        print("| Shell | Cross-FSC |")
        print("|-------|-----------|")
        for s in range(0, len(fsc), 5):
            print(f"| {s} | {float(fsc[s]):.4f} |")
        if (len(fsc) - 1) % 5 != 0:
            print(f"| {len(fsc) - 1} | {float(fsc[-1]):.4f} |")
        print()

    # ---- Remaining gaps ----
    print("## Remaining Differences After Fixes")
    print()
    print("| Gap | Status | Notes |")
    print("|-----|--------|-------|")
    print("| 1. Expand allowed current_sizes | FIXED | {16,24,32,48,64,96,128} vs RELION's arbitrary even sizes |")
    print("| 2. Per-shell noise in prior | ALREADY CORRECT | noise_variance is per-pixel (radially expanded) |")
    print("| 3. Adaptive oversampling cap | FIXED | max_union_pixels=200; falls back to pass-1-only |")
    print("| 4. Volume padding (--pad 2) | SKIPPED | Requires changing accumulation grid; documented |")
    print("| 5. Same half-set split | FIXED | Reads rlnRandomSubset from RELION STAR file |")
    print("| N/A. Noise estimation (hard vs soft) | NOT FIXED | Would need E-step changes |")
    print("| N/A. Per-image sparse pass 2 | NOT FIXED | Would need gather-scatter kernel |")
    print()


if __name__ == "__main__":
    main()
