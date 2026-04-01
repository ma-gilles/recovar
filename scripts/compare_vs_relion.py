#!/usr/bin/env python
"""Compare our EM refinement results against RELION reference data.

Loads both our results (from run_full_refinement.py) and RELION reference
data (extracted by extract_relion_reference.py), then compares:

a. Resolution trajectory: our current_size vs RELION's rlnCurrentImageSize
b. FSC curves: at selected iterations
c. Significant sample counts: our per-image stats vs RELION's
d. Final volume quality: FSC between our merged volume and RELION's
e. Wall-clock time: per-iteration and total

Usage:
    pixi run python scripts/compare_vs_relion.py [--our_results DIR] [--relion_ref DIR]

Outputs a detailed comparison report to stdout.
"""

import argparse
import logging
import os
import sys

import jax.numpy as jnp
import mrcfile
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def load_our_results(results_dir):
    """Load our refinement results from .npz file."""
    path = os.path.join(results_dir, "refinement_results.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results not found: {path}")
    data = dict(np.load(path, allow_pickle=True))
    return data


def load_relion_iteration(ref_dir, iteration):
    """Load one iteration of RELION reference data."""
    path = os.path.join(ref_dir, f"iteration_{iteration:03d}.npz")
    if not os.path.exists(path):
        return None
    data = dict(np.load(path, allow_pickle=True))
    for key in ("current_resolution", "current_image_size", "pixel_size",
                "original_image_size", "n_optics_groups"):
        if key in data and data[key].ndim == 0:
            data[key] = data[key].item()
    return data


def compare_resolution_trajectory(our_data, relion_dir, n_iter):
    """Compare current_size at each iteration."""
    our_sizes = our_data["current_sizes"]
    relion_sizes = []

    for it in range(n_iter):
        rd = load_relion_iteration(relion_dir, it)
        if rd is not None:
            relion_sizes.append(int(rd["current_image_size"]))
        else:
            relion_sizes.append(0)

    print("\n" + "=" * 70)
    print("A. RESOLUTION TRAJECTORY: current_image_size")
    print("=" * 70)
    print(f"{'Iter':>4s}  {'Ours':>8s}  {'RELION':>8s}  {'Diff':>8s}")
    print("-" * 40)

    n = min(len(our_sizes), len(relion_sizes))
    max_diff = 0
    for i in range(n):
        ours = int(our_sizes[i])
        theirs = relion_sizes[i]
        diff = ours - theirs
        max_diff = max(max_diff, abs(diff))
        print(f"{i+1:4d}  {ours:8d}  {theirs:8d}  {diff:+8d}")

    # Also print pixel resolution comparison
    our_pix_res = our_data["pixel_resolutions"]
    voxel_size = float(our_data.get("voxel_size", 4.25))
    print()
    print(f"{'Iter':>4s}  {'OurPixRes':>10s}  {'OurResA':>10s}  {'RELIONResA':>12s}")
    print("-" * 50)
    for i in range(n):
        pr = float(our_pix_res[i])
        res_a = pr / voxel_size
        rd = load_relion_iteration(relion_dir, i)
        relion_res_a = float(rd["current_resolution"]) if rd is not None else 0.0
        print(f"{i+1:4d}  {pr:10.1f}  {res_a:10.2f}  {relion_res_a:12.2f}")

    print()
    print(f"Maximum current_size difference: {max_diff}")
    print("NOTE: Our allowed sizes are {32, 64, 128} vs RELION's finer granularity.")
    return {"max_diff": max_diff, "our_sizes": our_sizes[:n], "relion_sizes": np.array(relion_sizes[:n])}


def compare_fsc_curves(our_data, relion_dir, iterations_to_compare=None):
    """Compare half-map FSC curves at selected iterations."""
    n_iter = int(our_data["n_iterations"])
    if iterations_to_compare is None:
        # Compare at iterations 1, 3, 5, 9 (0-indexed: 0, 2, 4, 8)
        iterations_to_compare = [i for i in [0, 2, 4, 8] if i < n_iter]

    print("\n" + "=" * 70)
    print("B. FSC CURVE COMPARISON (half-map FSC)")
    print("=" * 70)

    for it in iterations_to_compare:
        our_fsc_key = f"fsc_iter_{it:03d}"
        if our_fsc_key not in our_data:
            print(f"\nIteration {it+1}: No our FSC data available")
            continue

        our_fsc = our_data[our_fsc_key]
        rd = load_relion_iteration(relion_dir, it)
        if rd is None:
            print(f"\nIteration {it+1}: No RELION reference data")
            continue

        relion_fsc = rd["fsc"]
        n_shells = min(len(our_fsc), len(relion_fsc))

        # Find FSC=0.143 crossings
        our_143 = _find_fsc_crossing(our_fsc, 0.143)
        relion_143 = _find_fsc_crossing(relion_fsc, 0.143)

        print(f"\nIteration {it+1}:")
        print(f"  FSC=0.143 crossing: ours=shell {our_143}, RELION=shell {relion_143}")
        print(f"  {'Shell':>6s}  {'OurFSC':>8s}  {'RelFSC':>8s}  {'Diff':>8s}")
        print("  " + "-" * 40)

        # Print every 5th shell plus key shells
        shells_to_show = set(range(0, n_shells, 5))
        if our_143 is not None:
            shells_to_show.add(our_143)
        if relion_143 is not None:
            shells_to_show.add(relion_143)
        shells_to_show.add(n_shells - 1)

        for s in sorted(shells_to_show):
            if s >= n_shells:
                continue
            o = float(our_fsc[s])
            r = float(relion_fsc[s])
            d = o - r
            marker = " <-- 0.143" if s in (our_143, relion_143) else ""
            print(f"  {s:6d}  {o:8.4f}  {r:8.4f}  {d:+8.4f}{marker}")


def compare_significant_samples(our_data, relion_dir, n_iter):
    """Compare per-image significant sample counts."""
    print("\n" + "=" * 70)
    print("C. SIGNIFICANT SAMPLE COUNTS")
    print("=" * 70)
    print(f"{'Iter':>4s}  {'Our_min':>8s}  {'Our_med':>8s}  {'Our_max':>8s}  "
          f"{'Our_mean':>9s}  {'REL_med':>8s}  {'REL_mean':>9s}")
    print("-" * 70)

    for it in range(n_iter):
        our_key = f"sig_counts_iter_{it:03d}"
        rd = load_relion_iteration(relion_dir, it)

        our_str = ""
        if our_key in our_data:
            counts = our_data[our_key]
            our_str = f"{int(np.min(counts)):8d}  {int(np.median(counts)):8d}  {int(np.max(counts)):8d}  {float(np.mean(counts)):9.0f}"
        else:
            our_str = f"{'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>9s}"

        rel_str = ""
        if rd is not None and "nr_significant_samples" in rd:
            rel_counts = rd["nr_significant_samples"]
            if np.any(rel_counts > 0):
                rel_str = f"{int(np.median(rel_counts)):8d}  {float(np.mean(rel_counts)):9.0f}"
            else:
                rel_str = f"{'0':>8s}  {'0.0':>9s}"
        else:
            rel_str = f"{'N/A':>8s}  {'N/A':>9s}"

        print(f"{it+1:4d}  {our_str}  {rel_str}")

    print()
    print("NOTE: RELION counts significant SAMPLES (rot x trans); ours uses the same convention.")
    print("NOTE: Our prior/noise differs from RELION (scalar noise, different estimation).")


def compare_final_volumes(our_data, relion_ref_star_dir, volume_shape):
    """Compute FSC between our final volume and RELION's final volumes."""
    print("\n" + "=" * 70)
    print("D. FINAL VOLUME QUALITY (FSC vs RELION)")
    print("=" * 70)

    from recovar.reconstruction.regularization import get_fsc_gpu

    our_mean_ft = jnp.asarray(our_data["final_mean_ft"])

    # Load RELION's final merged volume
    relion_merged_path = os.path.join(relion_ref_star_dir, "run_class001.mrc")
    relion_half1_path = os.path.join(relion_ref_star_dir, "run_half1_class001_unfil.mrc")
    relion_half2_path = os.path.join(relion_ref_star_dir, "run_half2_class001_unfil.mrc")

    vol_shape_tuple = tuple(volume_shape)

    def load_mrc_to_ft(path):
        if not os.path.exists(path):
            return None
        with mrcfile.open(path, mode="r") as mrc:
            vol_real = np.array(mrc.data, dtype=np.float32)
        vol_ft = np.fft.fftn(np.fft.ifftshift(vol_real)).astype(np.complex64).reshape(-1)
        return jnp.asarray(vol_ft)

    # FSC: our merged vs RELION merged
    relion_merged_ft = load_mrc_to_ft(relion_merged_path)
    if relion_merged_ft is not None:
        fsc_merged = np.asarray(get_fsc_gpu(our_mean_ft, relion_merged_ft, vol_shape_tuple))
        crossing_merged = _find_fsc_crossing(fsc_merged, 0.5)
        crossing_merged_143 = _find_fsc_crossing(fsc_merged, 0.143)
        print(f"\nOur merged vs RELION merged:")
        print(f"  FSC=0.5 at shell {crossing_merged}")
        print(f"  FSC=0.143 at shell {crossing_merged_143}")
        _print_fsc_table(fsc_merged, "Shell", "FSC_merged")
    else:
        print(f"\nRELION merged volume not found at {relion_merged_path}")

    # FSC: our half1 vs RELION half1 (unfiltered)
    if "half0_mean_ft" in our_data:
        our_half1_ft = jnp.asarray(our_data["half0_mean_ft"])
        relion_half1_ft = load_mrc_to_ft(relion_half1_path)
        if relion_half1_ft is not None:
            fsc_h1 = np.asarray(get_fsc_gpu(our_half1_ft, relion_half1_ft, vol_shape_tuple))
            crossing_h1 = _find_fsc_crossing(fsc_h1, 0.5)
            print(f"\nOur half1 vs RELION half1 (unfiltered):")
            print(f"  FSC=0.5 at shell {crossing_h1}")
            _print_fsc_table(fsc_h1, "Shell", "FSC_half1")

    # Also compute our own half-map FSC for self-consistency
    if "half0_mean_ft" in our_data and "half1_mean_ft" in our_data:
        our_h1 = jnp.asarray(our_data["half0_mean_ft"])
        our_h2 = jnp.asarray(our_data["half1_mean_ft"])
        fsc_self = np.asarray(get_fsc_gpu(our_h1, our_h2, vol_shape_tuple))
        crossing_self = _find_fsc_crossing(fsc_self, 0.143)
        print(f"\nOur half1 vs our half2 (internal consistency):")
        print(f"  FSC=0.143 at shell {crossing_self}")

    # Also compare against GT volume
    gt_path = os.path.join(os.path.dirname(relion_ref_star_dir), "reference_gt.mrc")
    gt_ft = load_mrc_to_ft(gt_path)
    if gt_ft is not None:
        fsc_vs_gt = np.asarray(get_fsc_gpu(our_mean_ft, gt_ft, vol_shape_tuple))
        crossing_gt = _find_fsc_crossing(fsc_vs_gt, 0.5)
        crossing_gt_143 = _find_fsc_crossing(fsc_vs_gt, 0.143)
        print(f"\nOur merged vs ground truth:")
        print(f"  FSC=0.5 at shell {crossing_gt}")
        print(f"  FSC=0.143 at shell {crossing_gt_143}")
        _print_fsc_table(fsc_vs_gt, "Shell", "FSC_vs_GT")

        # RELION vs GT
        if relion_merged_ft is not None:
            fsc_relion_gt = np.asarray(get_fsc_gpu(relion_merged_ft, gt_ft, vol_shape_tuple))
            crossing_relion_gt_5 = _find_fsc_crossing(fsc_relion_gt, 0.5)
            crossing_relion_gt_143 = _find_fsc_crossing(fsc_relion_gt, 0.143)
            print(f"\nRELION merged vs ground truth:")
            print(f"  FSC=0.5 at shell {crossing_relion_gt_5}")
            print(f"  FSC=0.143 at shell {crossing_relion_gt_143}")


def compare_wall_times(our_data, n_iter):
    """Print per-iteration and total wall-clock times."""
    print("\n" + "=" * 70)
    print("E. WALL-CLOCK TIME")
    print("=" * 70)

    wall_times = our_data["wall_times"]
    current_sizes = our_data["current_sizes"]

    print(f"{'Iter':>4s}  {'CurSize':>8s}  {'Time(s)':>8s}")
    print("-" * 30)
    for i in range(min(n_iter, len(wall_times))):
        print(f"{i+1:4d}  {int(current_sizes[i]):8d}  {float(wall_times[i]):8.1f}")

    total = float(our_data.get("total_time", np.sum(wall_times)))
    print("-" * 30)
    print(f"Total: {total:.1f}s")
    print()
    print("RELION reference: ~160s per iteration on 1 GPU (varies by resolution).")
    print("NOTE: First iteration includes JIT compilation overhead.")


def summarize_known_differences():
    """Document known algorithmic differences between our code and RELION."""
    print("\n" + "=" * 70)
    print("F. KNOWN DIFFERENCES")
    print("=" * 70)
    print("""
1. NOISE ESTIMATION:
   - Ours: hard-assignment + subset-based (first 1000 images of half-set 0)
   - RELION: posterior-weighted, all images, per-half-set

2. SIGNAL PRIOR (tau^2):
   - Ours: scalar cov_noise in compute_relion_prior
   - RELION: per-shell sigma2_noise

3. ALLOWED CURRENT SIZES:
   - Ours: {32, 64, 128} (quantized to powers of 2)
   - RELION: any even integer up to original_image_size

4. ADAPTIVE OVERSAMPLING (Pass 2):
   - Ours: union-of-significant (evaluates all significant rots for all images)
   - RELION: per-image sparse (evaluates only each image's significant rots)

5. HALF-SET SPLIT:
   - Ours: random shuffle with seed=42
   - RELION: rlnRandomSubset from STAR file (set at import time)

6. PADDING:
   - RELION: --pad 2 (2x zero-padding for volume during reconstruction)
   - Ours: no volume padding (uses image_shape == volume_shape[:2])

7. FFT NORMALIZATION:
   - Different conventions may introduce scale factors

These differences mean we expect approximate (not exact) agreement.
""")


def _find_fsc_crossing(fsc, threshold):
    """Find first shell where FSC drops below threshold."""
    for i in range(1, len(fsc)):
        if fsc[i - 1] >= threshold and fsc[i] < threshold:
            return i
    return None


def _print_fsc_table(fsc, col1_name, col2_name, step=5):
    """Print FSC values at every step-th shell."""
    n = len(fsc)
    print(f"  {col1_name:>6s}  {col2_name:>10s}")
    print("  " + "-" * 20)
    for s in range(0, n, step):
        print(f"  {s:6d}  {float(fsc[s]):10.4f}")
    if (n - 1) % step != 0:
        print(f"  {n-1:6d}  {float(fsc[n-1]):10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare our refinement vs RELION reference")
    parser.add_argument(
        "--our_results",
        default="/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/our_results",
        help="Directory with our refinement_results.npz",
    )
    parser.add_argument(
        "--relion_ref_npz",
        default="/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/relion_ref_npz",
        help="Directory with RELION iteration_NNN.npz files",
    )
    parser.add_argument(
        "--relion_ref_star",
        default="/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/relion_ref",
        help="Directory with RELION STAR + MRC files",
    )
    args = parser.parse_args()

    # Load our results
    our_data = load_our_results(args.our_results)
    n_iter = int(our_data["n_iterations"])
    volume_shape = tuple(our_data["volume_shape"])

    print("=" * 70)
    print("COMPARISON: Our Refinement vs RELION Reference")
    print("=" * 70)
    print(f"Our results: {args.our_results}")
    print(f"RELION reference: {args.relion_ref_npz}")
    print(f"Iterations: {n_iter}")
    print(f"Images: {int(our_data['n_images'])}")
    print(f"Volume shape: {volume_shape}")

    # Run all comparisons
    compare_resolution_trajectory(our_data, args.relion_ref_npz, n_iter)
    compare_fsc_curves(our_data, args.relion_ref_npz)
    compare_significant_samples(our_data, args.relion_ref_npz, n_iter)
    compare_final_volumes(our_data, args.relion_ref_star, volume_shape)
    compare_wall_times(our_data, n_iter)
    summarize_known_differences()

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
