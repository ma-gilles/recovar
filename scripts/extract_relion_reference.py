#!/usr/bin/env python
"""Extract RELION auto-refine reference data into per-iteration .npz files.

Usage:
    python scripts/extract_relion_reference.py /path/to/relion_ref/run /path/to/output_dir

Reads RELION STAR files for each iteration and saves iteration_{N:03d}.npz.

Handles two RELION output formats:
  1. Single model: run_it{N}_model.star (no --split_random_halves)
  2. Half-map models: run_it{N}_half1_model.star + run_it{N}_half2_model.star
     (with --split_random_halves, the standard for auto_refine)

For half-map runs, model data is taken from half1 (both halves share the same
FSC, resolution, and image size; noise is estimated per-half but we store
the average). Per-image data comes from run_it{N}_data.star.

Saved fields per iteration:
    - current_resolution: float (Angstrom)
    - current_image_size: int (pixels)
    - sigma2_noise: (n_shells,) array, rlnSigma2Noise
    - sigma2_noise_half1: (n_shells,) array (half-map runs only)
    - sigma2_noise_half2: (n_shells,) array (half-map runs only)
    - reference_sigma2: (n_shells,) array, rlnReferenceSigma2
    - tau2: (n_shells,) array, rlnReferenceTau2
    - fsc: (n_shells,) array, rlnGoldStandardFsc
    - spectral_index, resolution, angstrom_resolution, ssnr: (n_shells,) arrays
    - euler_angles: (n_images, 3) array (Rot, Tilt, Psi) in degrees
    - origins: (n_images, 2) array (OriginXAngst, OriginYAngst)
    - nr_significant_samples: (n_images,) array
    - max_value_prob_distribution, log_likelihood_contribution: (n_images,) arrays
    - image_names: (n_images,) string array
    - pixel_size: float (Angstrom/pixel)
    - original_image_size: int (pixels)
    - n_optics_groups: int (asserted to be 1)
    - split_random_halves: bool (True if half-map format detected)

Requires: starfile, numpy

Assumptions:
    - Single optics group (fails loudly if multiple found)
    - Single class (standard for auto_refine)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import starfile
except ImportError:
    print("ERROR: 'starfile' package required. Install with: pip install starfile", file=sys.stderr)
    sys.exit(1)


def find_iterations(run_prefix: str) -> tuple[list[int], bool]:
    """Find all iteration numbers that have model and data STAR files.

    Returns
    -------
    iterations : list[int]
        Sorted list of iteration numbers.
    split_random_halves : bool
        True if half-map format detected (half1_model.star / half2_model.star).
    """
    run_dir = os.path.dirname(run_prefix)
    run_base = os.path.basename(run_prefix)

    if not run_dir:
        run_dir = "."

    files = set(os.listdir(run_dir))

    # Detect format: check if half1_model.star files exist
    split_random_halves = any(
        f.endswith("_half1_model.star") and f.startswith(run_base + "_it")
        for f in files
    )

    iterations = set()
    for f in files:
        if not f.startswith(run_base + "_it"):
            continue

        if split_random_halves:
            # Look for half1_model.star
            suffix = "_half1_model.star"
            if not f.endswith(suffix):
                continue
            it_str = f[len(run_base) + 3 : f.index(suffix)]
        else:
            # Look for _model.star (but not half*_model.star)
            suffix = "_model.star"
            if not f.endswith(suffix):
                continue
            if "_half" in f:
                continue
            it_str = f[len(run_base) + 3 : f.index(suffix)]

        try:
            it_num = int(it_str)
        except ValueError:
            continue

        # Check that corresponding data.star also exists
        data_file = f"{run_base}_it{it_num:03d}_data.star"
        if data_file in files:
            iterations.add(it_num)

    return sorted(iterations), split_random_halves


def _parse_model_star(model: dict) -> dict:
    """Extract relevant fields from a parsed RELION model STAR file."""
    general = model["model_general"]
    current_resolution = float(general["rlnCurrentResolution"])
    current_image_size = int(general["rlnCurrentImageSize"])
    original_image_size = int(general["rlnOriginalImageSize"])
    pixel_size = float(general["rlnPixelSize"])
    n_optics_groups = int(general["rlnNrOpticsGroups"])
    n_classes = int(general["rlnNrClasses"])

    assert n_optics_groups == 1, (
        f"Expected single optics group, found {n_optics_groups}. "
        "Multi-optics-group STAR files are not supported."
    )
    assert n_classes == 1, (
        f"Expected single class (auto_refine), found {n_classes}."
    )

    class_data = model["model_class_1"]
    spectral_index = class_data["rlnSpectralIndex"].values.astype(np.int32)
    resolution_freq = class_data["rlnResolution"].values.astype(np.float64)
    angstrom_resolution = class_data["rlnAngstromResolution"].values.astype(np.float64)
    ssnr = class_data["rlnSsnrMap"].values.astype(np.float64)
    fsc = class_data["rlnGoldStandardFsc"].values.astype(np.float64)
    reference_sigma2 = class_data["rlnReferenceSigma2"].values.astype(np.float64)
    tau2 = class_data["rlnReferenceTau2"].values.astype(np.float64)

    noise_data = model["model_optics_group_1"]
    sigma2_noise = noise_data["rlnSigma2Noise"].values.astype(np.float64)

    return {
        "current_resolution": current_resolution,
        "current_image_size": current_image_size,
        "original_image_size": original_image_size,
        "pixel_size": pixel_size,
        "n_optics_groups": n_optics_groups,
        "spectral_index": spectral_index,
        "resolution": resolution_freq,
        "angstrom_resolution": angstrom_resolution,
        "ssnr": ssnr,
        "fsc": fsc,
        "reference_sigma2": reference_sigma2,
        "tau2": tau2,
        "sigma2_noise": sigma2_noise,
    }


def extract_iteration(
    run_prefix: str, iteration: int, split_random_halves: bool
) -> dict:
    """Extract all reference data for a single iteration.

    Parameters
    ----------
    run_prefix : str
        Path prefix for RELION output, e.g., '/path/to/relion_ref/run'
    iteration : int
        Iteration number
    split_random_halves : bool
        If True, read half1/half2 model files instead of single model.

    Returns
    -------
    dict
        Dictionary of arrays/scalars to save in .npz
    """
    data_path = f"{run_prefix}_it{iteration:03d}_data.star"

    # --- Parse model STAR(s) ---
    if split_random_halves:
        h1_path = f"{run_prefix}_it{iteration:03d}_half1_model.star"
        h2_path = f"{run_prefix}_it{iteration:03d}_half2_model.star"
        model_h1 = starfile.read(h1_path)
        model_h2 = starfile.read(h2_path)
        parsed_h1 = _parse_model_star(model_h1)
        parsed_h2 = _parse_model_star(model_h2)

        # Use half1 for most fields (FSC, resolution, image_size are identical)
        parsed = parsed_h1.copy()
        # Average noise from both halves
        parsed["sigma2_noise"] = (
            parsed_h1["sigma2_noise"] + parsed_h2["sigma2_noise"]
        ) / 2.0
        # Also store per-half noise
        parsed["sigma2_noise_half1"] = parsed_h1["sigma2_noise"]
        parsed["sigma2_noise_half2"] = parsed_h2["sigma2_noise"]
        # Average reference_sigma2 and tau2 from both halves
        parsed["reference_sigma2"] = (
            parsed_h1["reference_sigma2"] + parsed_h2["reference_sigma2"]
        ) / 2.0
        parsed["tau2"] = (parsed_h1["tau2"] + parsed_h2["tau2"]) / 2.0
    else:
        model_path = f"{run_prefix}_it{iteration:03d}_model.star"
        model = starfile.read(model_path)
        parsed = _parse_model_star(model)

    # --- Parse data STAR ---
    data = starfile.read(data_path)

    optics = data["optics"]
    assert len(optics) == 1, (
        f"Expected single optics group in data STAR, found {len(optics)}."
    )

    particles = data["particles"]

    euler_angles = np.column_stack([
        particles["rlnAngleRot"].values.astype(np.float64),
        particles["rlnAngleTilt"].values.astype(np.float64),
        particles["rlnAnglePsi"].values.astype(np.float64),
    ])

    origins = np.column_stack([
        particles["rlnOriginXAngst"].values.astype(np.float64),
        particles["rlnOriginYAngst"].values.astype(np.float64),
    ])

    image_names = particles["rlnImageName"].values.astype(str)

    def _get_col_or_empty(df, col, dtype, fill=0):
        if col in df.columns:
            return df[col].values.astype(dtype)
        return np.full(len(df), fill, dtype=dtype)

    nr_significant_samples = _get_col_or_empty(
        particles, "rlnNrOfSignificantSamples", np.int32, fill=0
    )
    max_value_prob = _get_col_or_empty(
        particles, "rlnMaxValueProbDistribution", np.float64, fill=0.0
    )
    log_likelihood = _get_col_or_empty(
        particles, "rlnLogLikeliContribution", np.float64, fill=0.0
    )

    result = {
        "current_resolution": np.float64(parsed["current_resolution"]),
        "current_image_size": np.int32(parsed["current_image_size"]),
        "original_image_size": np.int32(parsed["original_image_size"]),
        "pixel_size": np.float64(parsed["pixel_size"]),
        "n_optics_groups": np.int32(parsed["n_optics_groups"]),
        "split_random_halves": np.bool_(split_random_halves),
        "spectral_index": parsed["spectral_index"],
        "resolution": parsed["resolution"],
        "angstrom_resolution": parsed["angstrom_resolution"],
        "sigma2_noise": parsed["sigma2_noise"],
        "reference_sigma2": parsed["reference_sigma2"],
        "tau2": parsed["tau2"],
        "fsc": parsed["fsc"],
        "ssnr": parsed["ssnr"],
        "euler_angles": euler_angles,
        "origins": origins,
        "nr_significant_samples": nr_significant_samples,
        "max_value_prob_distribution": max_value_prob,
        "log_likelihood_contribution": log_likelihood,
        "image_names": image_names,
    }

    # Add per-half noise if available
    if "sigma2_noise_half1" in parsed:
        result["sigma2_noise_half1"] = parsed["sigma2_noise_half1"]
        result["sigma2_noise_half2"] = parsed["sigma2_noise_half2"]

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract RELION auto-refine reference data into per-iteration .npz files."
    )
    parser.add_argument(
        "run_prefix",
        help="Path prefix for RELION output (e.g., /path/to/relion_ref/run)",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write iteration_NNN.npz files",
    )
    parser.add_argument(
        "--iterations",
        type=str,
        default=None,
        help="Comma-separated iteration numbers to extract (default: all found)",
    )
    args = parser.parse_args()

    run_prefix = args.run_prefix
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    all_iterations, split_random_halves = find_iterations(run_prefix)
    if not all_iterations:
        print(f"ERROR: No iterations found for prefix '{run_prefix}'", file=sys.stderr)
        sys.exit(1)

    mode = "split_random_halves" if split_random_halves else "single_model"
    print(f"Detected RELION output format: {mode}")
    print(f"Found {len(all_iterations)} iterations total")

    if args.iterations is not None:
        requested = [int(x.strip()) for x in args.iterations.split(",")]
        iterations = [i for i in requested if i in all_iterations]
        missing = set(requested) - set(all_iterations)
        if missing:
            print(f"WARNING: iterations {sorted(missing)} not found, skipping", file=sys.stderr)
    else:
        iterations = all_iterations

    print(f"Extracting {len(iterations)} iterations: {iterations[0]} to {iterations[-1]}")
    print(f"Output directory: {output_dir}")
    print()

    for it in iterations:
        out_path = os.path.join(output_dir, f"iteration_{it:03d}.npz")
        print(f"  Extracting iteration {it:3d} ... ", end="", flush=True)

        try:
            data = extract_iteration(run_prefix, it, split_random_halves)
            np.savez_compressed(out_path, **data)

            n_images = data["euler_angles"].shape[0]
            n_shells = data["sigma2_noise"].shape[0]
            res = float(data["current_resolution"])
            cs = int(data["current_image_size"])
            median_sig = int(np.median(data["nr_significant_samples"]))

            print(
                f"OK  (res={res:.2f}A, cs={cs}, "
                f"shells={n_shells}, images={n_images}, "
                f"median_sig_samples={median_sig})"
            )
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)
            raise

    # Print summary table
    print()
    print("=" * 80)
    print("RELION Reference Extraction Summary")
    print("=" * 80)
    print(f"{'Iter':>5s}  {'Resolution(A)':>14s}  {'ImageSize':>10s}  "
          f"{'MedianSigSamp':>14s}  {'FSC@Nyq':>8s}")
    print("-" * 80)

    for it in iterations:
        npz_path = os.path.join(output_dir, f"iteration_{it:03d}.npz")
        d = dict(np.load(npz_path, allow_pickle=True))
        res = float(d["current_resolution"])
        cs = int(d["current_image_size"])
        med_sig = int(np.median(d["nr_significant_samples"]))
        fsc_arr = d["fsc"]
        nonzero_mask = fsc_arr != 0.0
        fsc_last = float(fsc_arr[nonzero_mask][-1]) if nonzero_mask.any() else 0.0
        print(f"{it:5d}  {res:14.2f}  {cs:10d}  {med_sig:14d}  {fsc_last:8.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
