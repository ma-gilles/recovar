#!/usr/bin/env python
"""Phase 2 audit: quantitative comparison of recovar vs RELION prior and noise.

Loads the RELION reference data (iterations 0-9), runs recovar's split_E_M_v2
for the same number of iterations on the same dataset, and compares per-shell
noise variance, signal prior (tau^2), and FSC at each iteration.

Usage:
    CUDA_VISIBLE_DEVICES=1 pixi run python scripts/audit_prior_noise.py

Output:
    Prints comparison tables to stdout and saves detailed per-iteration
    comparison arrays to /scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/audit_results/
"""

import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("audit_prior_noise")

# ── Paths ──────────────────────────────────────────────────────────────────
RELION_REF_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/relion_ref_npz"
DATASET_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data"
OUTPUT_DIR = "/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/audit_results"

# Use Phase 0B comparison helpers (load via importlib to avoid sys.path issues)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "relion_comparison",
    "/scratch/gpfs/GILLES/mg6942/recovar_wt_phase0b/tests/integration/test_relion_comparison.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_relion_reference = _mod.load_relion_reference
compare_noise_spectra = _mod.compare_noise_spectra
compare_prior_spectra = _mod.compare_prior_spectra


def load_relion_iterations(ref_dir, iterations):
    """Load RELION reference data for multiple iterations."""
    data = {}
    for it in iterations:
        try:
            data[it] = load_relion_reference(ref_dir, it)
        except FileNotFoundError:
            logger.warning("RELION reference not found for iteration %d", it)
    return data


def run_recovar_em(dataset_dir, n_iterations, relion_data):
    """Run recovar's EM loop and collect per-iteration diagnostics.

    Returns a dict mapping iteration -> {noise_variance, prior, fsc, mean0, mean1}.
    """
    import jax
    import jax.numpy as jnp

    logger.info("JAX devices: %s", jax.devices())

    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.em import sampling, states
    from recovar.em.iterations import split_E_M_v2
    from recovar.reconstruction import noise, regularization

    # Load dataset
    star_path = os.path.join(dataset_dir, "particles.star")
    logger.info("Loading dataset from %s", star_path)
    dataset = load_dataset(star_path, lazy=False)
    logger.info(
        "Dataset: %d images, image_shape=%s, volume_shape=%s, voxel_size=%.2f",
        dataset.n_images,
        dataset.image_shape,
        dataset.volume_shape,
        dataset.voxel_size,
    )

    # Create half-set datasets with a deterministic random split
    n_images = dataset.n_images
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_images)
    half1_idx = np.sort(perm[: n_images // 2])
    half2_idx = np.sort(perm[n_images // 2 :])
    dataset.halfset_indices = (half1_idx, half2_idx)
    datasets = dataset.materialize_halfset_datasets()
    logger.info(
        "Half-sets: %d and %d images",
        datasets[0].n_images,
        datasets[1].n_images,
    )

    # Set up rotation and translation grids (matching RELION's healpix_order=3)
    from recovar import utils

    healpix_order = 3
    angles = sampling.get_rotation_grid(healpix_order)
    rotations = utils.R_from_relion(angles)
    translations = sampling.get_translation_grid(3, 1)
    logger.info(
        "Grid: %d rotations (order %d), %d translations",
        rotations.shape[0],
        healpix_order,
        translations.shape[0],
    )

    # Load initial reference volume
    import mrcfile

    init_vol_path = os.path.join(dataset_dir, "reference_init.mrc")
    with mrcfile.open(init_vol_path, mode="r") as mrc:
        init_vol_real = np.array(mrc.data, dtype=np.float32)

    # Convert to Fourier space
    init_vol_ft = np.fft.fftn(np.fft.ifftshift(init_vol_real)).reshape(-1).astype(np.complex64)
    logger.info("Initial volume loaded, shape=%s", init_vol_real.shape)

    # Initial noise estimate (from image power spectra)
    cov_noise, noise_profile = noise.estimate_noise_variance(datasets[0], batch_size=100)
    logger.info("Initial scalar noise variance: %.6e", cov_noise)

    noise_variance = noise.make_radial_noise(noise_profile, dataset.image_shape)

    # Initial prior: flat prior scaled from volume power spectrum
    PS = regularization.average_over_shells(
        jnp.abs(jnp.array(init_vol_ft)) ** 2, dataset.volume_shape
    )
    T = 4
    mean_signal_variance = (
        T * 0.5 * utils.make_radial_image(PS, dataset.volume_shape, extend_last_frequency=True)
    )
    mean_signal_variance += np.max(np.array(mean_signal_variance)) * 1e-6

    # Create EM state objects for each half-set
    disc_type = "linear_interp"
    state_objs = [
        states.EMState(
            mean=jnp.array(init_vol_ft),
            mean_variance=mean_signal_variance,
            noise_variance=noise_variance,
        )
        for _ in range(2)
    ]

    # Run EM iterations and collect diagnostics
    results = {}

    for it in range(n_iterations):
        logger.info("=" * 60)
        logger.info("EM ITERATION %d", it + 1)
        logger.info("=" * 60)

        t0 = time.time()
        state_objs, current_pixel_res, hard_assignments = split_E_M_v2(
            datasets, state_objs, rotations, translations, disc_type
        )
        dt = time.time() - t0
        logger.info("Iteration %d completed in %.1f seconds", it + 1, dt)

        # Extract per-shell noise variance
        nv = np.array(state_objs[0].noise_variance)
        # noise_variance is a flattened 2D image; extract radial profile
        noise_radial = np.array(
            regularization.average_over_shells(
                jnp.array(nv), dataset.image_shape
            )
        )

        # Extract per-shell prior (mean_signal_variance)
        mv = np.array(state_objs[0].mean_variance)
        prior_radial = np.array(
            regularization.average_over_shells(
                jnp.array(mv), dataset.volume_shape
            )
        )

        # Compute FSC between half-maps
        fsc = np.array(
            regularization.get_fsc_gpu(
                state_objs[0].mean, state_objs[1].mean, dataset.volume_shape
            )
        )

        results[it + 1] = {
            "noise_radial": noise_radial,
            "prior_radial": prior_radial,
            "fsc": fsc,
            "current_pixel_res": current_pixel_res,
        }

        logger.info(
            "  noise_radial[1:5] = %s",
            noise_radial[1:5],
        )
        logger.info(
            "  prior_radial[1:5] = %s",
            prior_radial[1:5],
        )
        logger.info(
            "  fsc[1:5] = %s",
            fsc[1:5],
        )
        logger.info(
            "  current_pixel_res = %s",
            current_pixel_res,
        )

    return results


def _normalize_spectrum(spectrum):
    """Normalize a spectrum to unit area for shape comparison."""
    total = np.sum(np.abs(spectrum))
    if total > 0:
        return spectrum / total
    return spectrum


def _find_fsc_resolution(fsc, threshold=1.0 / 7.0):
    """Find the shell where FSC drops below threshold."""
    for k in range(1, len(fsc)):
        if fsc[k] < threshold:
            return k
    return len(fsc)


def compare_iterations(recovar_results, relion_data, output_dir):
    """Compare recovar and RELION results at each iteration.

    Because recovar and RELION use different normalization conventions for
    noise_variance and tau^2 (the absolute values differ by a large constant
    factor related to FFT normalization), we focus on:

    1. FSC curves (scale-independent)
    2. Noise spectrum SHAPE (normalized)
    3. SNR = tau^2 / sigma^2 (the ratio, which is convention-invariant)
    4. Resolution trajectory
    """
    os.makedirs(output_dir, exist_ok=True)

    iterations = sorted(set(recovar_results.keys()) & set(relion_data.keys()))
    if not iterations:
        logger.error("No overlapping iterations between recovar and RELION data")
        return

    # Header
    print()
    print("=" * 110)
    print("PHASE 2 AUDIT: recovar vs RELION Prior/Noise Comparison")
    print("=" * 110)
    print()
    print("NOTE: recovar and RELION use different normalization conventions for")
    print("noise and prior (absolute values differ by ~5e7 due to FFT conventions).")
    print("We compare: (1) FSC (scale-free), (2) noise SHAPE, (3) SNR ratios,")
    print("(4) resolution trajectory.")

    # ── 1. FSC comparison ──
    print()
    print("1. FSC COMPARISON (scale-independent)")
    print("-" * 110)
    print(
        f"{'Iter':>5s}  {'OursFSC_res':>11s}  {'RELION_res':>11s}  "
        f"{'MaxFSCdiff':>11s}  {'MedFSCdiff':>11s}  "
        f"{'OursFSC[3:8]':>45s}"
    )
    print("-" * 110)

    for it in iterations:
        ours_fsc = recovar_results[it]["fsc"]
        relion_fsc = relion_data[it]["fsc"]
        n = min(len(ours_fsc), len(relion_fsc))

        fsc_diff = np.abs(ours_fsc[:n] - relion_fsc[:n])
        # Only compare shells 1 to n-1 (skip DC)
        max_fsc_diff = float(np.max(fsc_diff[1:n]))
        med_fsc_diff = float(np.median(fsc_diff[1:n]))

        ours_res = _find_fsc_resolution(ours_fsc)
        relion_res = _find_fsc_resolution(relion_fsc)

        print(
            f"{it:5d}  {ours_res:11d}  {relion_res:11d}  "
            f"{max_fsc_diff:11.4f}  {med_fsc_diff:11.4f}  "
            f"{np.array2string(ours_fsc[3:8], precision=4, separator=', ')}"
        )

    # ── 2. Noise spectrum shape comparison ──
    print()
    print("2. NOISE SPECTRUM SHAPE (both normalized to unit sum)")
    print("-" * 110)
    print(
        f"{'Iter':>5s}  {'MaxShapeDiff':>13s}  {'MedShapeDiff':>13s}  "
        f"{'Corr':>7s}  "
        f"{'ScaleRatio':>12s}  {'Note':>20s}"
    )
    print("-" * 110)

    for it in iterations:
        ours = recovar_results[it]["noise_radial"]
        theirs = relion_data[it]["sigma2_noise"]
        n = min(len(ours), len(theirs))

        ours_norm = _normalize_spectrum(ours[:n])
        theirs_norm = _normalize_spectrum(theirs[:n])
        shape_diff = np.abs(ours_norm - theirs_norm)

        # Pearson correlation of noise spectra (shape similarity)
        corr = float(np.corrcoef(ours[:n], theirs[:n])[0, 1]) if n > 1 else 0.0

        # Scale ratio (the multiplicative constant between conventions)
        valid = theirs[:n] > 1e-30
        if np.any(valid):
            ratios = ours[:n][valid] / theirs[:n][valid]
            scale_ratio = float(np.median(ratios))
        else:
            scale_ratio = float("nan")

        print(
            f"{it:5d}  {float(np.max(shape_diff[1:])):13.6f}  "
            f"{float(np.median(shape_diff[1:])):13.6f}  "
            f"{corr:7.4f}  {scale_ratio:12.2f}  "
            f"{'(diff convention)':>20s}"
        )

    # ── 3. SNR comparison (convention-invariant) ──
    # SNR_recovar = prior / noise (both in recovar units)
    # SNR_relion = tau2 / sigma2_noise (both in RELION units)
    # If the formula is correct, these ratios should match.
    print()
    print("3. SNR = prior/noise COMPARISON (convention-invariant ratio)")
    print("-" * 110)
    print(
        f"{'Iter':>5s}  {'MaxSNRratio':>12s}  {'MedSNRratio':>12s}  "
        f"{'SNR_ours[1:5]':>40s}  {'SNR_relion[1:5]':>40s}"
    )
    print("-" * 110)

    for it in iterations:
        ours_noise = recovar_results[it]["noise_radial"]
        ours_prior = recovar_results[it]["prior_radial"]
        relion_noise = relion_data[it]["sigma2_noise"]
        relion_tau2 = relion_data[it]["tau2"]

        n = min(len(ours_noise), len(relion_noise), len(ours_prior), len(relion_tau2))

        # SNR = prior / noise
        snr_ours = np.where(ours_noise[:n] > 1e-30, ours_prior[:n] / ours_noise[:n], 0.0)
        snr_relion = np.where(relion_noise[:n] > 1e-30, relion_tau2[:n] / relion_noise[:n], 0.0)

        # Ratio of SNRs (should be ~1 if formulas match)
        valid = (snr_relion > 1e-30) & (snr_ours > 1e-30)
        if np.any(valid):
            snr_ratio = snr_ours[valid] / snr_relion[valid]
            max_ratio = float(np.max(snr_ratio))
            med_ratio = float(np.median(snr_ratio))
        else:
            max_ratio = float("nan")
            med_ratio = float("nan")

        print(
            f"{it:5d}  {max_ratio:12.4f}  {med_ratio:12.4f}  "
            f"{np.array2string(snr_ours[1:5], precision=4, separator=', ')}  "
            f"{np.array2string(snr_relion[1:5], precision=4, separator=', ')}"
        )

    # ── 4. Resolution trajectory ──
    print()
    print("4. RESOLUTION TRAJECTORY")
    print("-" * 60)
    print(f"{'Iter':>5s}  {'Ours_px_res':>12s}  {'RELION_res_A':>12s}  {'RELION_cs':>10s}")
    print("-" * 60)
    for it in iterations:
        ours_res = recovar_results[it]["current_pixel_res"]
        relion_res = relion_data[it]["current_resolution"]
        relion_cs = relion_data[it]["current_image_size"]
        print(f"{it:5d}  {ours_res:12.2f}  {relion_res:12.2f}  {relion_cs:10d}")

    # ── 5. Detailed per-shell comparison at a mid iteration ──
    mid_it = min(5, max(iterations))
    if mid_it in iterations:
        print()
        print(f"5. DETAILED PER-SHELL COMPARISON AT ITERATION {mid_it}")
        print("-" * 130)
        print(
            f"{'Shell':>6s}  {'NoiseShape_o':>12s}  {'NoiseShape_r':>12s}  "
            f"{'SNR_ours':>10s}  {'SNR_relion':>10s}  {'SNR_ratio':>10s}  "
            f"{'FSC_ours':>9s}  {'FSC_relion':>9s}  {'FSC_diff':>9s}"
        )
        print("-" * 130)

        ours_noise = recovar_results[mid_it]["noise_radial"]
        relion_noise = relion_data[mid_it]["sigma2_noise"]
        ours_prior = recovar_results[mid_it]["prior_radial"]
        relion_tau2 = relion_data[mid_it]["tau2"]
        ours_fsc = recovar_results[mid_it]["fsc"]
        relion_fsc = relion_data[mid_it]["fsc"]

        n = min(len(ours_noise), len(relion_noise), len(ours_prior), len(relion_tau2))

        ours_noise_norm = _normalize_spectrum(ours_noise[:n])
        relion_noise_norm = _normalize_spectrum(relion_noise[:n])

        snr_ours = np.where(ours_noise[:n] > 1e-30, ours_prior[:n] / ours_noise[:n], 0.0)
        snr_relion = np.where(relion_noise[:n] > 1e-30, relion_tau2[:n] / relion_noise[:n], 0.0)

        for k in range(min(n, 40)):
            snr_r = snr_ours[k] / snr_relion[k] if snr_relion[k] > 1e-30 else float("nan")
            fsc_o = ours_fsc[k] if k < len(ours_fsc) else float("nan")
            fsc_r = relion_fsc[k] if k < len(relion_fsc) else float("nan")
            fsc_d = abs(fsc_o - fsc_r)
            print(
                f"{k:6d}  {ours_noise_norm[k]:12.6e}  {relion_noise_norm[k]:12.6e}  "
                f"{snr_ours[k]:10.4f}  {snr_relion[k]:10.4f}  {snr_r:10.4f}  "
                f"{fsc_o:9.4f}  {fsc_r:9.4f}  {fsc_d:9.4f}"
            )

    # ── Save results ──
    for it in iterations:
        np.savez(
            os.path.join(output_dir, f"comparison_iter_{it:03d}.npz"),
            noise_ours=recovar_results[it]["noise_radial"],
            noise_relion=relion_data[it]["sigma2_noise"],
            prior_ours=recovar_results[it]["prior_radial"],
            tau2_relion=relion_data[it]["tau2"],
            reference_sigma2_relion=relion_data[it]["reference_sigma2"],
            fsc_ours=recovar_results[it]["fsc"],
            fsc_relion=relion_data[it]["fsc"],
        )

    print()
    print(f"Detailed comparison arrays saved to {output_dir}")
    print("=" * 110)


def main():
    logger.info("Starting prior/noise audit")

    # Load RELION reference data for iterations 1-9
    iterations = list(range(1, 10))
    logger.info("Loading RELION reference data for iterations %s", iterations)
    relion_data = load_relion_iterations(RELION_REF_DIR, iterations)
    logger.info("Loaded %d RELION iterations", len(relion_data))

    # Run recovar EM
    n_iterations = 9
    logger.info("Running recovar EM for %d iterations", n_iterations)
    recovar_results = run_recovar_em(DATASET_DIR, n_iterations, relion_data)

    # Compare
    compare_iterations(recovar_results, relion_data, OUTPUT_DIR)


if __name__ == "__main__":
    main()
