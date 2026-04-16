"""Compare recovar's initial noise estimate vs RELION's iter-0/iter-1 sigma2_noise.

The hypothesis under test: recovar's Pmax at iter 1 (~0.003) is 13x flatter than
RELION's (~0.04) because recovar's initial noise estimate is too high, which
compresses score differences in the E-step.

RELION's FFT is 1/N normalized, so the native conversion factor between
recovar's unnormalized FFT power and RELION's power is N^(-2*D) for a D-D
image, i.e. N^4 for 2D images.

Usage:
    pixi run python scripts/compare_iter0_noise.py
"""

import os
import re
import sys

import numpy as np

DATA_DIR = "/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized"
STAR = os.path.join(DATA_DIR, "particles.star")
RELION_IT000 = os.path.join(DATA_DIR, "relion_ref_os0", "run_it000_half1_model.star")
RELION_IT001 = os.path.join(DATA_DIR, "relion_ref_os0", "run_it001_half1_model.star")

# 2D images in this dataset.
N = 128
FFT_SCALE = N**4  # recovar (unnormalized FFT) power / FFT_SCALE ~= RELION power


def parse_sigma2_from_star(path):
    """Parse _rlnSigma2Noise values from the data_model_optics_group_1 block."""
    with open(path) as f:
        lines = f.readlines()
    # Find the block header
    start = None
    for i, line in enumerate(lines):
        if "data_model_optics_group_1" in line:
            start = i
            break
    if start is None:
        raise RuntimeError(f"data_model_optics_group_1 not found in {path}")
    # Scan forward for the _rlnSigma2Noise column index, then the numeric rows
    col_idx = None
    rows = []
    in_loop = False
    after_labels = False
    for line in lines[start:]:
        s = line.strip()
        if s.startswith("_rlnSigma2Noise"):
            # "#3" -> column 2 (0-based)
            m = re.search(r"#(\d+)", s)
            col_idx = int(m.group(1)) - 1
            in_loop = True
            continue
        if in_loop:
            if not s:
                if after_labels:
                    break
                continue
            if s.startswith("_") or s.startswith("loop_"):
                continue
            parts = s.split()
            if len(parts) <= col_idx:
                continue
            try:
                rows.append(float(parts[col_idx]))
                after_labels = True
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float64)


def recovar_initial_noise():
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.reconstruction.noise import estimate_initial_noise_spectrum_from_unaligned_images

    dataset = load_dataset(STAR, lazy=False)
    n_images = dataset.n_images
    print(f"[recovar] n_images={n_images}, image_shape={dataset.image_shape}")
    subset = np.arange(min(n_images, 5000), dtype=np.int32)
    sigma2 = estimate_initial_noise_spectrum_from_unaligned_images(dataset, image_subset=subset, batch_size=1000)
    return np.asarray(sigma2, dtype=np.float64)


def main():
    print("=" * 80)
    print("Noise initialization comparison: recovar vs RELION")
    print(f"FFT conversion factor: N^4 = {N}^4 = {FFT_SCALE:.4e}")
    print("=" * 80)

    sigma2_recovar_native = recovar_initial_noise()
    sigma2_recovar_relion_units = sigma2_recovar_native / FFT_SCALE

    sigma2_relion_it0 = parse_sigma2_from_star(RELION_IT000)
    sigma2_relion_it1 = parse_sigma2_from_star(RELION_IT001)
    print(f"[relion] it0 n_shells={len(sigma2_relion_it0)}, it1 n_shells={len(sigma2_relion_it1)}")
    print(f"[recovar] n_shells={len(sigma2_recovar_native)}")

    n_compare = min(len(sigma2_recovar_native), len(sigma2_relion_it0), len(sigma2_relion_it1))

    print()
    print(
        f"{'shell':>5}  {'recovar_native':>16}  {'recovar→relion':>16}  "
        f"{'RELION_it0':>14}  {'RELION_it1':>14}  {'rec/rel_it0':>12}  {'rec/rel_it1':>12}"
    )
    print("-" * 120)

    ratios_it0 = []
    ratios_it1 = []
    for s in range(n_compare):
        rc_native = sigma2_recovar_native[s]
        rc_rel = sigma2_recovar_relion_units[s]
        r0 = sigma2_relion_it0[s]
        r1 = sigma2_relion_it1[s]
        ratio0 = rc_rel / r0 if r0 > 0 else float("nan")
        ratio1 = rc_rel / r1 if r1 > 0 else float("nan")
        if np.isfinite(ratio0):
            ratios_it0.append(ratio0)
        if np.isfinite(ratio1):
            ratios_it1.append(ratio1)
        if s < 20 or s % 8 == 0:
            print(
                f"{s:>5d}  {rc_native:>16.6e}  {rc_rel:>16.6e}  "
                f"{r0:>14.6e}  {r1:>14.6e}  {ratio0:>12.3f}  {ratio1:>12.3f}"
            )

    ratios_it0 = np.array(ratios_it0)
    ratios_it1 = np.array(ratios_it1)
    print()
    print("Ratio summary (recovar in RELION units / RELION):")
    print(
        f"  vs RELION it0: median={np.median(ratios_it0):.3f}, "
        f"mean={np.mean(ratios_it0):.3f}, "
        f"min={np.min(ratios_it0):.3f}, max={np.max(ratios_it0):.3f}"
    )
    print(
        f"  vs RELION it1: median={np.median(ratios_it1):.3f}, "
        f"mean={np.mean(ratios_it1):.3f}, "
        f"min={np.min(ratios_it1):.3f}, max={np.max(ratios_it1):.3f}"
    )

    # Low-shell closer look (shells 1..10 — these dominate iter-1 posterior)
    print()
    print("Low-shell (1..10) focus — these dominate iter-1 posterior sharpness:")
    for s in range(1, min(11, n_compare)):
        print(
            f"  shell {s}: recovar(native)={sigma2_recovar_native[s]:.3e}, "
            f"recovar(relion_u)={sigma2_recovar_relion_units[s]:.3e}, "
            f"relion_it0={sigma2_relion_it0[s]:.3e}, "
            f"relion_it1={sigma2_relion_it1[s]:.3e}, "
            f"ratio_it0={sigma2_recovar_relion_units[s] / sigma2_relion_it0[s]:.3f}, "
            f"ratio_it1={sigma2_recovar_relion_units[s] / sigma2_relion_it1[s]:.3f}"
        )

    mean_low_ratio_it0 = np.median(ratios_it0[1:11])
    mean_low_ratio_it1 = np.median(ratios_it1[1:11])
    print()
    print("=" * 80)
    print(
        f"VERDICT: at low shells (1..10), recovar's sigma2 is {mean_low_ratio_it0:.2f}x RELION iter-0, "
        f"{mean_low_ratio_it1:.2f}x RELION iter-1."
    )
    if mean_low_ratio_it0 > 1.5 or mean_low_ratio_it0 < 1 / 1.5:
        print("  -> DIVERGENT: >1.5x difference at shells that drive iter-1 Pmax.")
        print("     This is a plausible source of the Pmax gap.")
    else:
        print("  -> CONSISTENT: ratio within 1.5x, noise init is NOT the primary cause.")
    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main())
