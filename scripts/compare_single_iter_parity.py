#!/usr/bin/env python
"""Single-iteration E-step parity: recovar engine_v2 kernels vs RELION bindings.

Strategy: build up from algebraic identity (F4a, all-RELION, FFTW layout)
through layout conversion (F5) to full recovar pipeline (F6), isolating
each potential error source.

Usage:
  pixi run python scripts/compare_single_iter_parity.py \
    --relion_dir data_noise1_5k_normalized/relion_ref_os0 \
    --data_star data_noise1_5k_normalized/particles.star \
    --iter 1 --n_images 5 --n_rotations 48
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def rel_err(a, b):
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom > 0, denom, 1.0)
    return float(np.max(np.abs(a - b) / denom))


def report(stage, name, err, threshold=1e-10):
    color = GREEN if err < threshold else (YELLOW if err < 1e-4 else RED)
    status = "PASS" if err < threshold else ("WARN" if err < 1e-4 else "FAIL")
    print(f"  {color}[{status}]{RESET} {stage}: {name} — rel_err = {err:.3e}", flush=True)
    return err < threshold


def shell_indices_fftw(N):
    ky = np.arange(N)
    ky = np.where(ky <= N // 2, ky, ky - N)
    kx = np.arange(N // 2 + 1)
    ky2d, kx2d = np.meshgrid(ky, kx, indexing="ij")
    return np.round(np.sqrt(ky2d**2 + kx2d**2)).astype(int)


def sigma2_shells_to_full_spectrum(sigma2_shells, image_shape, current_size=-1):
    import recovar.core.fourier_transform_utils as ftu

    radii = ftu.get_grid_of_radial_distances(image_shape, voxel_size=1, scaled=False, frequency_shift=0, rounded=True)
    result = np.full(image_shape, 1e10, dtype=np.float64)
    for s in range(len(sigma2_shells)):
        mask = radii == s
        result[mask] = sigma2_shells[s]
    if current_size > 0:
        result[radii > current_size // 2] = 1e10
    return result.reshape(-1)


def fftw_to_recovar_half(arr_fftw):
    """Convert FFTW half-complex (N, N//2+1) to recovar half layout.

    Both layouts store N*(N//2+1) complex values for the same frequencies.
    FFTW: rows are standard order (DC at row 0).
    recovar half: rows are centered order (DC at row N//2).
    Column order is identical: kx = 0, 1, ..., N//2 (Nyquist aliases kx = ±N/2).
    """
    return np.fft.fftshift(arr_fftw, axes=0)


def recovar_half_to_fftw(arr_rhalf):
    """Inverse of fftw_to_recovar_half."""
    return np.fft.ifftshift(arr_rhalf, axes=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relion_dir", required=True)
    parser.add_argument("--data_star", required=True)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--n_images", type=int, default=5)
    parser.add_argument("--n_rotations", type=int, default=48)
    parser.add_argument("--healpix_order", type=int, default=2)
    parser.add_argument("--offset_range", type=float, default=3.0)
    parser.add_argument("--offset_step", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import mrcfile
    import starfile
    from recovar.relion_bind._relion_bind_core import (
        TRILINEAR,
        convert_squared_differences_to_weights,
        euler_angles_to_matrix,
        get_coarse_orientations,
        get_coarse_translations,
        get_ctf_image,
        project_volume,
        shift_image_in_fourier_transform_2d,
    )

    relion_dir = Path(args.relion_dir)
    prefix = relion_dir / f"run_it{args.iter:03d}"

    print(f"\n{'=' * 70}")
    print("Single-iteration per-function parity test")
    print(f"{'=' * 70}\n")

    # ================================================================
    # LOAD STATE
    # ================================================================
    with mrcfile.open(str(prefix) + "_half1_class001.mrc") as f:
        vol_relion = f.data.copy().astype(np.float64)
    N = vol_relion.shape[0]
    image_shape = (N, N)

    model_h1 = starfile.read(str(prefix) + "_half1_model.star")
    sigma2_h1 = np.array(model_h1["model_optics_group_1"]["rlnSigma2Noise"], dtype=np.float64)
    pixel_size = float(model_h1["model_general"]["rlnPixelSize"])
    current_size = int(model_h1["model_general"]["rlnCurrentImageSize"])

    particles_star = starfile.read(str(args.data_star))
    particles_df = particles_star["particles"] if isinstance(particles_star, dict) else particles_star

    orientations = get_coarse_orientations(args.healpix_order)
    n_rot = min(args.n_rotations, orientations.shape[0])
    translations = get_coarse_translations(args.offset_range, args.offset_step)
    n_trans = translations.shape[0]
    print(f"  Volume: {N}^3, current_size={current_size}, pixel_size={pixel_size}")
    print(f"  Grid: {n_rot} rotations × {n_trans} translations")

    # ================================================================
    # LOAD DATASET
    # ================================================================
    from recovar.data_io.cryoem_dataset import load_dataset

    ds = load_dataset(str(args.data_star))
    n_images = min(args.n_images, ds.n_units)
    image_indices = np.arange(n_images)
    print(f"  Dataset: {ds.n_units} images, testing {n_images}")

    # ================================================================
    # PREPARE NOISE (RELION convention)
    # ================================================================
    sigma2_clamped = np.copy(sigma2_h1)
    sigma2_clamped[sigma2_clamped <= 0] = 1e10
    sidx_fftw = shell_indices_fftw(N)
    Minvsigma2 = np.zeros_like(sidx_fftw, dtype=np.float64)
    for s in range(len(sigma2_clamped)):
        mask = sidx_fftw == s
        if sigma2_clamped[s] < 1e9:
            Minvsigma2[mask] = 1.0 / (2.0 * sigma2_clamped[s])
    Minvsigma2[sidx_fftw == 0] = 0.0
    if current_size > 0:
        Minvsigma2[sidx_fftw > current_size // 2] = 0.0

    # ================================================================
    # PREPARE RELION PROJECTIONS (pf=2)
    # ================================================================
    pf = 2
    print(f"\n  Computing {n_rot} RELION projections (pf={pf})...", flush=True)
    projs_relion_fftw = []
    for ir in range(n_rot):
        rot, tilt, psi = orientations[ir]
        A = euler_angles_to_matrix(float(rot), float(tilt), float(psi))
        p = project_volume(
            vol_relion, A, ori_size=N, padding_factor=pf, interpolator=TRILINEAR, current_size=-1, do_gridding=True
        )
        projs_relion_fftw.append(p)
    projs_relion_fftw = np.array(projs_relion_fftw)  # (n_rot, N, N//2+1)

    # ================================================================
    # PREPARE RECOVAR PROJECTIONS
    # ================================================================
    import recovar.core.fourier_transform_utils as ftu
    from recovar import core
    from recovar.core.configs import ForwardModelConfig
    from recovar.core.geometry import translate_images
    from recovar.reconstruction.relion_functions import pad_volume_for_projection
    from recovar.utils.helpers import R_from_relion, relion_volume_to_recovar

    vol_recovar = relion_volume_to_recovar(vol_relion)
    vol_ft = ftu.get_dft3(vol_recovar).reshape(-1)
    mean_padded, proj_vs = pad_volume_for_projection(vol_ft, (N, N, N), pf)

    euler_arr = orientations[:n_rot]
    rot_matrices = R_from_relion(euler_arr).astype(np.float32)
    projs_recovar_half = np.asarray(
        core.slice_volume(
            jnp.asarray(mean_padded), jnp.asarray(rot_matrices), image_shape, proj_vs, "linear_interp", half_image=True
        )
    )  # (n_rot, N_half)

    # Measure projection scale per-rotation
    scale_per_rot = np.array(
        [
            np.sqrt(np.sum(np.abs(projs_recovar_half[ir]) ** 2))
            / (np.sqrt(np.sum(np.abs(projs_relion_fftw[ir]) ** 2)) + 1e-30)
            for ir in range(n_rot)
        ]
    )
    mean_scale = scale_per_rot.mean()
    print(f"  Projection L2 scale: mean={mean_scale:.1f}, std={scale_per_rot.std():.1f}")
    print(f"  Expected N^3={N**3}, (pf*N)^3={(pf * N) ** 3}")

    config = ForwardModelConfig.from_dataset(ds, disc_type="linear_interp", process_fn=ds.process_images)

    # ================================================================
    # STEP F4a: FFTW-layout algebraic scoring (GOLD STANDARD)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("F4a: FFTW-layout algebraic scoring (all RELION data)")
    print(f"{'=' * 70}")

    f4a_data = {}  # Save per-image data for comparison

    all_pass = True
    for img_i in range(n_images):
        raw = ds.image_source[img_i][0][0]
        p = particles_df.iloc[img_i]
        img_fftw = np.fft.rfft2(raw)

        ctf_fftw = get_ctf_image(
            defU=float(p["rlnDefocusU"]),
            defV=float(p["rlnDefocusV"]),
            defAng=float(p["rlnDefocusAngle"]),
            voltage=300.0,
            Cs=2.7,
            Q0=0.07,
            Bfac=0.0,
            angpix=pixel_size,
            orixdim=N,
            oriydim=N,
            do_ctf_padding=False,
            do_abs=False,
            do_damping=False,
            phase_shift=0.0,
            scale=1.0,
        )

        diff2_relion = np.zeros((n_rot, n_trans))
        cross_4a = np.zeros((n_rot, n_trans))
        norm_4a = np.zeros((n_rot, n_trans))
        score_4a = np.zeros((n_rot, n_trans))
        for ir in range(n_rot):
            for it in range(n_trans):
                tx, ty = translations[it]
                shifted = shift_image_in_fourier_transform_2d(img_fftw, float(N), N, float(tx), float(ty))
                res = shifted - ctf_fftw * projs_relion_fftw[ir]
                diff2_relion[ir, it] = np.sum(Minvsigma2 * np.abs(res) ** 2)
                weighted_shifted = shifted * ctf_fftw * 2.0 * Minvsigma2
                cross_4a[ir, it] = -2.0 * np.sum(np.real(np.conj(weighted_shifted) * projs_relion_fftw[ir]))
                norm_4a[ir, it] = np.sum(2.0 * Minvsigma2 * ctf_fftw**2 * np.abs(projs_relion_fftw[ir]) ** 2)
                score_4a[ir, it] = -0.5 * (cross_4a[ir, it] + norm_4a[ir, it])

        sum_ds = diff2_relion + score_4a
        spread = sum_ds.max() - sum_ds.min()
        mean_val = np.abs(sum_ds).mean()
        const_err = spread / (mean_val + 1e-30) if mean_val > 0 else spread
        ok = report("F4a", f"img[{img_i}] diff2+score=const", const_err, threshold=1e-10)
        all_pass = all_pass and ok

        orient_prior = np.ones(n_rot, dtype=np.float64) / n_rot
        offset_prior = np.ones(n_trans, dtype=np.float64) / n_trans
        w_relion = convert_squared_differences_to_weights(
            diff2_relion, orient_prior, offset_prior, min_diff2=float(diff2_relion.min())
        )

        f4a_data[img_i] = {
            "diff2": diff2_relion,
            "cross": cross_4a,
            "norm": norm_4a,
            "score": score_4a,
            "weights": w_relion,
            "ctf_fftw": ctf_fftw,
            "img_fftw": img_fftw,
        }

        pmax_r = w_relion.max()
        argmax_r = np.unravel_index(w_relion.argmax(), (n_rot, n_trans))
        print(f"  F4a: img[{img_i}] Pmax={pmax_r:.6f} argmax={argmax_r}")
        print(f"    cross[0,0]={cross_4a[0, 0]:.4e}  norm[0,0]={norm_4a[0, 0]:.4e}  score[0,0]={score_4a[0, 0]:.4e}")

    # ================================================================
    # STEP F5: Same scoring in recovar half layout (LAYOUT VALIDATION)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("F5: RELION data in recovar half layout (pure layout test)")
    print(f"{'=' * 70}")

    projs_relion_rhalf = np.array([fftw_to_recovar_half(p) for p in projs_relion_fftw])
    Minvsigma2_rhalf = fftw_to_recovar_half(Minvsigma2)

    for img_i in range(n_images):
        d = f4a_data[img_i]
        ctf_rhalf = fftw_to_recovar_half(d["ctf_fftw"])

        cross_5 = np.zeros((n_rot, n_trans))
        norm_5 = np.zeros((n_rot, n_trans))
        for ir in range(n_rot):
            for it in range(n_trans):
                tx, ty = translations[it]
                shifted_fftw = shift_image_in_fourier_transform_2d(d["img_fftw"], float(N), N, float(tx), float(ty))
                shifted_rhalf = fftw_to_recovar_half(shifted_fftw)
                weighted = shifted_rhalf * ctf_rhalf * 2.0 * Minvsigma2_rhalf
                cross_5[ir, it] = -2.0 * np.sum(np.real(np.conj(weighted) * projs_relion_rhalf[ir]))
                norm_5[ir, it] = np.sum(2.0 * Minvsigma2_rhalf * ctf_rhalf**2 * np.abs(projs_relion_rhalf[ir]) ** 2)

        score_5 = -0.5 * (cross_5 + norm_5)
        err_cross = rel_err(cross_5, d["cross"])
        err_norm = rel_err(norm_5, d["norm"])
        err_score = rel_err(score_5, d["score"])
        report("F5", f"img[{img_i}] cross layout parity", err_cross)
        report("F5", f"img[{img_i}] norm layout parity", err_norm)
        ok = report("F5", f"img[{img_i}] score layout parity", err_score)
        all_pass = all_pass and ok

    # ================================================================
    # STEP F5b: recovar half layout using full_image_to_half_image
    # ================================================================
    print(f"\n{'=' * 70}")
    print("F5b: RELION data via recovar's full_image_to_half_image path")
    print(f"{'=' * 70}")

    noise_full = sigma2_shells_to_full_spectrum(sigma2_clamped, image_shape, current_size=current_size)

    for img_i in range(min(2, n_images)):
        d = f4a_data[img_i]

        # Reconstruct full centered spectrum from FFTW half for the image
        # rfft2(raw) is (N, N//2+1). To get full centered: need Hermitian extension + fftshift.
        raw = ds.image_source[img_i][0][0]
        img_full_centered = np.asarray(ftu.get_dft2(jnp.asarray(raw)))  # (N, N) centered full

        # Extract half using recovar's function
        img_half_recovar = np.asarray(
            ftu.full_image_to_half_image(jnp.asarray(img_full_centered.reshape(1, -1)), image_shape)
        )[0].reshape(N, N // 2 + 1)

        # Compare with fftshift(rfft2)
        img_rhalf_from_fftw = fftw_to_recovar_half(np.fft.rfft2(raw))

        # The centering phase: get_dft2 = (-1)^{ky+j} * fftshift(rfft2) in the half layout
        # (for N=128 where N/2=64 is even)
        ky_arr = np.arange(N)[:, None]
        j_arr = np.arange(N // 2 + 1)[None, :]
        centering_phase = (-1.0) ** (ky_arr + j_arr)
        # For even N where N/2 is even: the phase should relate the two
        # Actually, centered col j in recovar half maps to centered position (ky, j_cent)
        # where j_cent = N/2+j for j<N/2, j_cent = 0 for j=N/2
        # phase = (-1)^{ky + j_cent} = (-1)^{ky + N/2 + j} for j<N/2, (-1)^{ky} for j=N/2
        # For N=128: (-1)^{N/2} = (-1)^64 = 1, so phase = (-1)^{ky+j} uniformly
        j_centered = np.where(j_arr < N // 2, N // 2 + j_arr, 0)
        centering_phase_exact = (-1.0) ** (ky_arr + j_centered)

        ratio = img_half_recovar / (img_rhalf_from_fftw + 1e-30)
        # At non-zero pixels, this should be centering_phase_exact
        nz = np.abs(img_rhalf_from_fftw) > 1e-10 * np.max(np.abs(img_rhalf_from_fftw))
        phase_err = rel_err(ratio[nz].real, centering_phase_exact[nz])
        report("F5b", f"img[{img_i}] centering phase = (-1)^(ky+j_cent)", phase_err, threshold=1e-8)

        # Verify the centering phase cancels in inner product
        # <conj(phase*A), phase*B> = <conj(A), B>  (since |phase|=1)
        proj_relion_r0 = projs_relion_rhalf[0].reshape(N, N // 2 + 1)
        proj_with_phase = proj_relion_r0 * centering_phase_exact
        inner_with = np.sum(np.conj(img_half_recovar) * proj_with_phase).real
        inner_without = np.sum(np.conj(img_rhalf_from_fftw) * proj_relion_r0).real
        inner_err = abs(inner_with - inner_without) / (abs(inner_without) + 1e-30)
        report("F5b", f"img[{img_i}] phase cancellation in inner product", inner_err)

    # ================================================================
    # STEP F5c: Per-pixel noise weight comparison
    # ================================================================
    print(f"\n{'=' * 70}")
    print("F5c: Per-pixel noise weight comparison (RELION vs recovar)")
    print(f"{'=' * 70}")

    # RELION weight per FFTW half pixel: 2*Minvsigma2 = 1/sigma2 (for shell ≤ current_size//2, else 0)
    relion_weight_fftw = 2.0 * Minvsigma2  # (N, N//2+1)
    relion_weight_rhalf = fftw_to_recovar_half(relion_weight_fftw)  # centered rows

    # recovar weight per full-spectrum pixel: 1/noise_full (noise_full = sigma2, 1e10 for high shells)
    recovar_weight_full = 1.0 / noise_full  # (N*N,)
    recovar_weight_half = np.asarray(ftu.full_image_to_half_image(jnp.asarray(recovar_weight_full[None]), image_shape))[
        0
    ].reshape(N, N // 2 + 1)

    # Compare pixel-by-pixel
    nz = relion_weight_rhalf > 1e-20
    if nz.sum() > 0:
        err_nz = rel_err(recovar_weight_half[nz], relion_weight_rhalf[nz])
        report("F5c", f"noise weight parity (nonzero pixels, n={nz.sum()})", err_nz)
        ratio_weights = recovar_weight_half[nz] / (relion_weight_rhalf[nz] + 1e-30)
        print(
            f"    weight ratio: mean={ratio_weights.mean():.6f} std={ratio_weights.std():.6f} "
            f"min={ratio_weights.min():.6f} max={ratio_weights.max():.6f}"
        )

    # Count pixels that differ in zero/nonzero status
    relion_nz = relion_weight_rhalf > 1e-20
    recovar_nz = recovar_weight_half > 1e-20
    agree = (relion_nz == recovar_nz).sum()
    disagree = (relion_nz != recovar_nz).sum()
    print(
        f"    zero/nonzero agreement: {agree}/{agree + disagree} "
        f"(disagree: {disagree}, relion_only={int((relion_nz & ~recovar_nz).sum())}, "
        f"recovar_only={int((~relion_nz & recovar_nz).sum())})"
    )

    # Shell assignment comparison
    # Get recovar shell indices in half layout
    radii_full = np.asarray(
        ftu.get_grid_of_radial_distances(image_shape, voxel_size=1, scaled=False, frequency_shift=0, rounded=True)
    )
    radii_half = (
        np.asarray(
            ftu.full_image_to_half_image(jnp.asarray(radii_full.reshape(1, -1).astype(np.float32)), image_shape)
        )[0]
        .reshape(N, N // 2 + 1)
        .astype(int)
    )
    shell_diff = radii_half != fftw_to_recovar_half(sidx_fftw.astype(float)).astype(int)
    print(
        f"    shell index mismatches: {shell_diff.sum()}/{N * (N // 2 + 1)} "
        f"(at shells {np.unique(radii_half[shell_diff])[:10]}...)"
        if shell_diff.sum() > 0
        else f"    shell indices: all match ({N * (N // 2 + 1)} pixels)"
    )

    # ================================================================
    # STEP F6: Recovar pipeline scoring with DIAGNOSTICS
    # ================================================================
    print(f"\n{'=' * 70}")
    print("F6: Recovar full-pipeline scoring (diagnostics)")
    print(f"{'=' * 70}")

    for img_i in range(n_images):
        d = f4a_data[img_i]
        raw = ds.image_source[img_i][0][0]
        p = particles_df.iloc[img_i]

        ctf_params_i = jnp.asarray(ds.CTF_params[img_i : img_i + 1], dtype=jnp.float32)
        ctf_full = np.asarray(config.compute_ctf(ctf_params_i))[0]  # (N*N,) centered
        processed = np.asarray(ds.process_images(np.array([raw]), apply_image_mask=False))[0]
        weighted_img = processed * ctf_full / noise_full  # (N*N,) full centered

        cross_6 = np.zeros((n_rot, n_trans))
        norm_6 = np.zeros((n_rot, n_trans))

        ctf2_nv_full = (ctf_full**2 / noise_full).reshape(1, -1)
        ctf2_nv_half = np.asarray(ftu.full_image_to_half_image(jnp.asarray(ctf2_nv_full), image_shape))[0]

        trans_arr = np.array(translations[:n_trans], dtype=np.float32)
        for it in range(n_trans):
            tx, ty = trans_arr[it]
            shifted = np.asarray(
                translate_images(jnp.asarray(weighted_img[None]), jnp.asarray(np.array([[tx, ty]])), image_shape)
            )[0]
            shifted_half = np.asarray(ftu.full_image_to_half_image(jnp.asarray(shifted[None]), image_shape))[0]

            for ir in range(n_rot):
                proj_h = projs_recovar_half[ir]
                cross_6[ir, it] = -2.0 * np.sum(np.real(np.conj(shifted_half) * proj_h))
                norm_6[ir, it] = np.sum(ctf2_nv_half * np.abs(proj_h) ** 2)

        score_6 = -0.5 * (cross_6 + norm_6)

        # ---- DIAGNOSTIC: cross/norm ratios ----
        r_cross = np.abs(cross_6[0, 0]) / (np.abs(d["cross"][0, 0]) + 1e-30)
        r_norm = np.abs(norm_6[0, 0]) / (np.abs(d["norm"][0, 0]) + 1e-30)
        print(f"  img[{img_i}] DIAGNOSTIC:")
        print(
            f"    cross: recovar={cross_6[0, 0]:.4e}  relion={d['cross'][0, 0]:.4e}  "
            f"ratio={r_cross:.2f}  (expect ~{mean_scale:.0f})"
        )
        print(
            f"    norm:  recovar={norm_6[0, 0]:.4e}  relion={d['norm'][0, 0]:.4e}  "
            f"ratio={r_norm:.2f}  (expect ~{mean_scale**2:.0f})"
        )
        print(
            f"    |cross/norm| recovar: {np.abs(cross_6[0, 0]) / (norm_6[0, 0] + 1e-30):.4f}  "
            f"relion: {np.abs(d['cross'][0, 0]) / (d['norm'][0, 0] + 1e-30):.4f}"
        )

        # Cross-only ranking (remove norm term)
        cross_only_ranking = np.argsort(-cross_6.ravel())[:5]  # higher cross → better match
        # But cross has -2Re<...> so MORE NEGATIVE = better match for maximizing score
        # Actually score = -0.5*(cross + norm), and we want highest score.
        # cross = -2 Re<weighted_img, proj> → negative when img matches proj
        # norm = Σ CTF²/σ² |proj|² → always positive
        # score = -0.5*(cross + norm) → higher when cross is more negative and norm is small
        # For "cross-only" ranking: sort by -cross (most negative cross = best)
        cross_only_rank = np.argsort(cross_6.ravel())[:5]  # most negative first
        relion_rank = np.argsort(d["diff2"].ravel())[:5]

        # Norm-only ranking
        norm_only_rank = np.argsort(norm_6.ravel())[:5]  # smallest norm = highest score

        overlap_cross = len(set(cross_only_rank) & set(relion_rank))
        overlap_norm = len(set(norm_only_rank) & set(relion_rank))

        print(f"    RELION top-5: {relion_rank}")
        print(f"    cross-only top-5: {cross_only_rank}  overlap={overlap_cross}/5")
        print(f"    norm-only top-5: {norm_only_rank}  overlap={overlap_norm}/5")

        # Full score ranking
        score_rank_6 = np.argsort(-score_6.ravel())[:5]
        overlap_full = len(set(score_rank_6) & set(relion_rank))
        print(f"    full score top-5: {score_rank_6}  overlap={overlap_full}/5")

        # Posteriors
        s_flat = score_6.ravel()
        log_Z = float(jax.scipy.special.logsumexp(jnp.asarray(s_flat)))
        w_6 = np.exp(s_flat - log_Z).reshape(n_rot, n_trans)
        pmax_6 = w_6.max()
        argmax_6 = np.unravel_index(w_6.argmax(), (n_rot, n_trans))
        argmax_r = np.unravel_index(d["weights"].argmax(), (n_rot, n_trans))
        pmax_r = d["weights"].max()

        match = argmax_r == argmax_6
        color = GREEN if match else RED
        print(
            f"    {color}[{'PASS' if match else 'FAIL'}]{RESET} F6: argmax relion={argmax_r} "
            f"recovar={argmax_6}  Pmax {pmax_r:.6f} vs {pmax_6:.6f}"
        )
        all_pass = all_pass and match

    # ================================================================
    # STEP F7: Normalized projections (correct the scale)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("F7: Recovar scoring with per-rotation normalized projections")
    print(f"{'=' * 70}")

    # Normalize each projection individually to match RELION's L2 norm
    projs_normalized = np.zeros_like(projs_recovar_half)
    for ir in range(n_rot):
        projs_normalized[ir] = projs_recovar_half[ir] / scale_per_rot[ir]

    for img_i in range(n_images):
        d = f4a_data[img_i]
        raw = ds.image_source[img_i][0][0]

        ctf_params_i = jnp.asarray(ds.CTF_params[img_i : img_i + 1], dtype=jnp.float32)
        ctf_full = np.asarray(config.compute_ctf(ctf_params_i))[0]
        processed = np.asarray(ds.process_images(np.array([raw]), apply_image_mask=False))[0]
        weighted_img = processed * ctf_full / noise_full

        ctf2_nv_full = (ctf_full**2 / noise_full).reshape(1, -1)
        ctf2_nv_half = np.asarray(ftu.full_image_to_half_image(jnp.asarray(ctf2_nv_full), image_shape))[0]

        cross_7 = np.zeros((n_rot, n_trans))
        norm_7 = np.zeros((n_rot, n_trans))

        trans_arr = np.array(translations[:n_trans], dtype=np.float32)
        for it in range(n_trans):
            tx, ty = trans_arr[it]
            shifted = np.asarray(
                translate_images(jnp.asarray(weighted_img[None]), jnp.asarray(np.array([[tx, ty]])), image_shape)
            )[0]
            shifted_half = np.asarray(ftu.full_image_to_half_image(jnp.asarray(shifted[None]), image_shape))[0]

            for ir in range(n_rot):
                proj_h = projs_normalized[ir]
                cross_7[ir, it] = -2.0 * np.sum(np.real(np.conj(shifted_half) * proj_h))
                norm_7[ir, it] = np.sum(ctf2_nv_half * np.abs(proj_h) ** 2)

        score_7 = -0.5 * (cross_7 + norm_7)

        # Diagnostic: check ratios
        r_cross = np.abs(cross_7[0, 0]) / (np.abs(d["cross"][0, 0]) + 1e-30)
        r_norm = np.abs(norm_7[0, 0]) / (np.abs(d["norm"][0, 0]) + 1e-30)
        print(f"  img[{img_i}] norm'd cross ratio: {r_cross:.4f}, norm ratio: {r_norm:.4f}")

        # Rankings
        score_rank = np.argsort(-score_7.ravel())[:5]
        relion_rank = np.argsort(d["diff2"].ravel())[:5]
        overlap = len(set(score_rank) & set(relion_rank))

        # Posteriors
        s_flat = score_7.ravel()
        log_Z = float(jax.scipy.special.logsumexp(jnp.asarray(s_flat)))
        w_7 = np.exp(s_flat - log_Z).reshape(n_rot, n_trans)
        pmax_7 = w_7.max()
        argmax_7 = np.unravel_index(w_7.argmax(), (n_rot, n_trans))
        argmax_r = np.unravel_index(d["weights"].argmax(), (n_rot, n_trans))
        pmax_r = d["weights"].max()

        match = argmax_r == argmax_7
        color = GREEN if match else RED
        print(
            f"    {color}[{'PASS' if match else 'FAIL'}]{RESET} F7: argmax relion={argmax_r} "
            f"norm_recovar={argmax_7}  Pmax {pmax_r:.6f} vs {pmax_7:.6f}"
        )
        print(f"    top-5 overlap: {overlap}/5  relion={relion_rank}  recovar={score_rank}")

        if not match:
            w_corr = np.corrcoef(d["weights"].ravel(), w_7.ravel())[0, 1]
            print(f"    weight corr: {w_corr:.6f}")

        all_pass = all_pass and match

    # ================================================================
    # STEP F8: RELION projections with centering phase in recovar path
    # ================================================================
    print(f"\n{'=' * 70}")
    print("F8: RELION projections + centering phase → recovar scoring pipeline")
    print(f"{'=' * 70}")

    # Apply centering phase to RELION projections so they're compatible
    # with recovar's process_images output
    ky_arr = np.arange(N)[:, None]
    j_centered = np.where(np.arange(N // 2 + 1)[None, :] < N // 2, N // 2 + np.arange(N // 2 + 1)[None, :], 0)
    centering_phase = (-1.0) ** (ky_arr + j_centered)  # (N, N//2+1)
    centering_phase_flat = centering_phase.reshape(-1)

    # RELION projs in recovar half layout WITH centering phase applied
    projs_relion_phased = projs_relion_rhalf.reshape(n_rot, N, N // 2 + 1) * centering_phase[None]
    projs_relion_phased = projs_relion_phased.reshape(n_rot, -1)

    # DC mask in half layout (matching engine_v2.py:758-768)
    dc_mask_half = (radii_half == 0).reshape(-1)
    print(f"  DC pixels in half layout: {dc_mask_half.sum()} (at indices {np.where(dc_mask_half)[0]})")

    for img_i in range(n_images):
        d = f4a_data[img_i]
        raw = ds.image_source[img_i][0][0]

        ctf_params_i = jnp.asarray(ds.CTF_params[img_i : img_i + 1], dtype=jnp.float32)
        ctf_full = np.asarray(config.compute_ctf(ctf_params_i))[0]
        processed = np.asarray(ds.process_images(np.array([raw]), apply_image_mask=False))[0]
        weighted_img = processed * ctf_full / noise_full

        ctf2_nv_full = (ctf_full**2 / noise_full).reshape(1, -1)
        ctf2_nv_half = np.asarray(ftu.full_image_to_half_image(jnp.asarray(ctf2_nv_full), image_shape))[0]

        # Zero DC in scoring arrays (production code does this)
        ctf2_nv_half_sc = ctf2_nv_half.copy()
        ctf2_nv_half_sc[dc_mask_half] = 0.0

        cross_8 = np.zeros((n_rot, n_trans))
        norm_8 = np.zeros((n_rot, n_trans))

        trans_arr = np.array(translations[:n_trans], dtype=np.float32)
        for it in range(n_trans):
            tx, ty = trans_arr[it]
            shifted = np.asarray(
                translate_images(jnp.asarray(weighted_img[None]), jnp.asarray(np.array([[tx, ty]])), image_shape)
            )[0]
            shifted_half = np.asarray(ftu.full_image_to_half_image(jnp.asarray(shifted[None]), image_shape))[0]
            # Zero DC in shifted image for scoring
            shifted_half_sc = shifted_half.copy()
            shifted_half_sc[dc_mask_half] = 0.0

            for ir in range(n_rot):
                proj_h = projs_relion_phased[ir]
                cross_8[ir, it] = -2.0 * np.sum(np.real(np.conj(shifted_half_sc) * proj_h))
                norm_8[ir, it] = np.sum(ctf2_nv_half_sc * np.abs(proj_h) ** 2)

        score_8 = -0.5 * (cross_8 + norm_8)

        # Diagnostic
        r_cross = cross_8[0, 0] / (d["cross"][0, 0] + 1e-30)
        r_norm = norm_8[0, 0] / (d["norm"][0, 0] + 1e-30)
        print(f"  img[{img_i}] RELION+phase cross ratio: {r_cross:.6f}, norm ratio: {r_norm:.6f}")
        # The CTF sign flip (-1) should be the only remaining difference
        # cross: recovar img has CTF_recovar=-CTF_relion, so cross_8 = -cross_4a (modulo layout)
        # norm: CTF² cancels sign, norm_8 = norm_4a

        # Correct the CTF sign in cross: negate cross_8 to account for CTF_recovar = -CTF_relion
        # Actually: weighted_img = img * CTF_recovar / σ² = img * (-CTF_relion) / σ²
        # cross_8 = -2 Re Σ conj(img * (-CTF) / σ² * trans) * proj
        #         = +2 Re Σ (CTF/σ²) conj(img*trans) * proj  [since CTF real, minus comes out]
        # cross_4a = -2 Re Σ (CTF/σ²) conj(img*trans) * proj
        # So cross_8 = -cross_4a!  The sign flip is real.

        # Rankings with cross negated (to compensate CTF sign)
        score_8_corrected = -0.5 * (-cross_8 + norm_8)

        score_rank = np.argsort(-score_8_corrected.ravel())[:5]
        relion_rank = np.argsort(d["diff2"].ravel())[:5]
        overlap = len(set(score_rank) & set(relion_rank))

        s_flat = score_8_corrected.ravel()
        log_Z = float(jax.scipy.special.logsumexp(jnp.asarray(s_flat)))
        w_8 = np.exp(s_flat - log_Z).reshape(n_rot, n_trans)
        pmax_8 = w_8.max()
        argmax_8 = np.unravel_index(w_8.argmax(), (n_rot, n_trans))
        argmax_r = np.unravel_index(d["weights"].argmax(), (n_rot, n_trans))
        pmax_r = d["weights"].max()

        match = argmax_r == argmax_8
        color = GREEN if match else RED
        print(
            f"    {color}[{'PASS' if match else 'FAIL'}]{RESET} F8 (CTF-corrected): argmax relion={argmax_r} "
            f"recovar={argmax_8}  Pmax {pmax_r:.6f} vs {pmax_8:.6f}"
        )
        print(f"    top-5 overlap: {overlap}/5")

        # Also check: does the uncorrected score give wrong rankings?
        score_rank_raw = np.argsort(-score_8.ravel())[:5]
        overlap_raw = len(set(score_rank_raw) & set(relion_rank))
        print(f"    (uncorrected top-5 overlap: {overlap_raw}/5)")

        all_pass = all_pass and match

    # ================================================================
    # STEP F8b: Isolate translation effect on cross ratio
    # ================================================================
    print(f"\n{'=' * 70}")
    print("F8b: Translation isolation — cross ratio at (0,0) vs other translations")
    print(f"{'=' * 70}")

    print(f"  Translation grid: {n_trans} entries")
    for it in range(min(5, n_trans)):
        print(f"    translations[{it}] = ({translations[it][0]:.1f}, {translations[it][1]:.1f})")

    # Find the (0,0) translation
    trans_dists = np.sqrt(translations[:, 0] ** 2 + translations[:, 1] ** 2)
    it_zero = int(np.argmin(trans_dists))
    print(f"  Closest to (0,0): index {it_zero} = ({translations[it_zero][0]:.2f}, {translations[it_zero][1]:.2f})")

    img_i = 0
    d = f4a_data[img_i]
    raw = ds.image_source[img_i][0][0]

    ctf_params_i = jnp.asarray(ds.CTF_params[img_i : img_i + 1], dtype=jnp.float32)
    ctf_full = np.asarray(config.compute_ctf(ctf_params_i))[0]
    processed = np.asarray(ds.process_images(np.array([raw]), apply_image_mask=False))[0]
    weighted_img = processed * ctf_full / noise_full

    # Compute cross ratio at EACH translation to see if it varies
    print("\n  Cross ratio (cross_8 / cross_4a) at each translation, rot=0:")
    for it in range(n_trans):
        tx, ty = translations[it]
        shifted = np.asarray(
            translate_images(
                jnp.asarray(weighted_img[None]), jnp.asarray(np.array([[float(tx), float(ty)]])), image_shape
            )
        )[0]
        shifted_half = np.asarray(ftu.full_image_to_half_image(jnp.asarray(shifted[None]), image_shape))[0]

        proj_h = projs_relion_phased[0]
        cross_8_val = -2.0 * np.sum(np.real(np.conj(shifted_half) * proj_h))
        cross_4a_val = d["cross"][0, it]
        ratio = cross_8_val / (cross_4a_val + 1e-30)
        marker = " ← (0,0)" if it == it_zero else ""
        print(
            f"    t[{it}]=({tx:+.0f},{ty:+.0f}): cross_8={cross_8_val:.4e} cross_4a={cross_4a_val:.4e} ratio={ratio:.6f}{marker}"
        )

    # Per-pixel diagnostic at the (0,0) translation, rot=0
    print(
        f"\n  Per-pixel cross comparison at (rot=0, trans=({translations[it_zero][0]:.0f},{translations[it_zero][1]:.0f})):"
    )
    tx0, ty0 = translations[it_zero]

    # F4a path: pixel-level contributions
    shifted_fftw = shift_image_in_fourier_transform_2d(d["img_fftw"], float(N), N, float(tx0), float(ty0))
    weighted_fftw = shifted_fftw * d["ctf_fftw"] * 2.0 * Minvsigma2
    cross_per_pixel_f4a = -2.0 * np.real(np.conj(weighted_fftw) * projs_relion_fftw[0])

    # F8 path: pixel-level contributions (in recovar half layout)
    shifted_8 = np.asarray(
        translate_images(
            jnp.asarray(weighted_img[None]), jnp.asarray(np.array([[float(tx0), float(ty0)]])), image_shape
        )
    )[0]
    shifted_8_half = np.asarray(ftu.full_image_to_half_image(jnp.asarray(shifted_8[None]), image_shape))[0].reshape(
        N, N // 2 + 1
    )
    proj_8_h = projs_relion_phased[0].reshape(N, N // 2 + 1)
    cross_per_pixel_f8 = -2.0 * np.real(np.conj(shifted_8_half) * proj_8_h)

    # Convert F4a to recovar half layout for comparison
    cross_per_pixel_f4a_rhalf = fftw_to_recovar_half(cross_per_pixel_f4a)

    # Per-pixel ratio
    nz = np.abs(cross_per_pixel_f4a_rhalf) > 1e-20
    ratio_pp = cross_per_pixel_f8[nz] / (cross_per_pixel_f4a_rhalf[nz] + 1e-30)
    print(f"    nonzero pixels: {nz.sum()}")
    print(
        f"    per-pixel ratio: mean={ratio_pp.mean():.6f} std={ratio_pp.std():.6f} min={ratio_pp.min():.6f} max={ratio_pp.max():.6f}"
    )

    # Check if all ratios are the same (should be -1.0 if CTF sign is the only difference)
    ratio_mode = np.median(ratio_pp)
    outliers = np.abs(ratio_pp - ratio_mode) > 0.01
    print(f"    median ratio: {ratio_mode:.6f}, outlier pixels: {outliers.sum()}")

    # CRITICAL: explicit sums to verify aggregate vs per-pixel
    sum_f8_all = float(cross_per_pixel_f8.sum())
    sum_f4a_all = float(cross_per_pixel_f4a_rhalf.sum())
    sum_f8_nz = float(cross_per_pixel_f8[nz].sum())
    sum_f4a_nz = float(cross_per_pixel_f4a_rhalf[nz].sum())
    sum_f8_extra = sum_f8_all - sum_f8_nz
    sum_abs_f4a = float(np.abs(cross_per_pixel_f4a_rhalf[nz]).sum())
    print("\n    SUM DIAGNOSTIC (at rot=0, trans=(0,0)):")
    print(f"    sum(f4a, all 8320 pixels) = {sum_f4a_all:.4f}")
    print(f"    sum(f4a, 1308 nz pixels)  = {sum_f4a_nz:.4f}")
    print(
        f"    sum(|f4a|, nz pixels)     = {sum_abs_f4a:.4f}  (cancellation ratio: {sum_abs_f4a / (abs(sum_f4a_nz) + 1e-30):.1f}×)"
    )
    print(f"    sum(f8, all 8320 pixels)  = {sum_f8_all:.4f}")
    print(f"    sum(f8, 1308 nz pixels)   = {sum_f8_nz:.4f}")
    print(f"    sum(f8, 7012 extra pixels) = {sum_f8_extra:.6e}")
    print(f"    ratio sum(f8_all)/sum(f4a_all) = {sum_f8_all / (sum_f4a_all + 1e-30):.6f}")
    print(f"    ratio sum(f8_nz)/sum(f4a_nz)  = {sum_f8_nz / (sum_f4a_nz + 1e-30):.6f}")
    print("    F8 main loop cross_8[0,14]     = (check above)")
    print(
        f"    dtypes: weighted_img={weighted_img.dtype}, shifted_8_half={shifted_8_half.dtype}, proj_8_h={proj_8_h.dtype}"
    )

    # Print actual values at a few EXTRA pixels to understand the 2132 contribution
    extra_mask = ~nz  # the 7012 extra pixels
    extra_indices = np.argwhere(extra_mask)
    print(f"\n    EXTRA PIXEL VALUES (first 10 of {extra_mask.sum()}):")
    noise_full_half = np.asarray(ftu.full_image_to_half_image(jnp.asarray(noise_full[None]), image_shape))[0].reshape(
        N, N // 2 + 1
    )
    for idx in range(min(10, len(extra_indices))):
        ky_e, kx_e = extra_indices[idx]
        sh = radii_half[ky_e, kx_e]
        sv = shifted_8_half[ky_e, kx_e]
        pv = proj_8_h[ky_e, kx_e]
        cv = cross_per_pixel_f8[ky_e, kx_e]
        nv = noise_full_half[ky_e, kx_e]
        print(
            f"      [{ky_e:3d},{kx_e:3d}] shell={sh:2d} noise={nv:.2e} "
            f"|shifted|={abs(sv):.4e} |proj|={abs(pv):.4e} cross={cv:+.4e}"
        )
    # Find the pixel with max |cross| among extra pixels
    extra_cross_vals = cross_per_pixel_f8[extra_mask]
    max_idx_in_extra = np.argmax(np.abs(extra_cross_vals))
    max_ky, max_kx = extra_indices[max_idx_in_extra]
    print(
        f"    MAX EXTRA: [{max_ky},{max_kx}] shell={radii_half[max_ky, max_kx]} "
        f"|shifted|={abs(shifted_8_half[max_ky, max_kx]):.4e} |proj|={abs(proj_8_h[max_ky, max_kx]):.4e} "
        f"cross={cross_per_pixel_f8[max_ky, max_kx]:+.4e} noise={noise_full_half[max_ky, max_kx]:.2e}"
    )
    # Also show the F4a side at the same pixel in FFTW coords
    ky_f = (max_ky + N // 2) % N
    kx_f = max_kx  # same column index in half layout
    print(
        f"    F4a at same freq: FFTW[{ky_f},{kx_f}] Minvsigma2={Minvsigma2[ky_f, kx_f]:.6e} "
        f"|weighted_fftw|={abs(weighted_fftw[ky_f, kx_f]):.4e} |proj_fftw|={abs(projs_relion_fftw[0][ky_f, kx_f]):.4e}"
    )

    # Histogram of |cross| values at extra pixels
    extra_cross = np.abs(cross_per_pixel_f8[extra_mask])
    print(f"    extra |cross|: mean={extra_cross.mean():.4e} max={extra_cross.max():.4e} sum={extra_cross.sum():.4e}")
    print(f"    extra cross (signed sum): {cross_per_pixel_f8[extra_mask].sum():.4e}")
    # Also check: are the extra pixels really at high shells?
    extra_shells = radii_half[extra_mask]
    print(
        f"    extra pixel shells: min={extra_shells.min()} max={extra_shells.max()} unique={len(np.unique(extra_shells))}"
    )

    # Decompose into components: image, CTF, noise, projection
    # Check each factor independently at a few pixels
    print("\n  Component comparison at pixel (ky=64, kx=10) [DC row, kx=10]:")
    ky_test, kx_test = 64, 10  # centered coords
    # FFTW equivalent: row = (ky_test + N//2) % N = 0, col = kx_test
    ky_fftw = (ky_test + N // 2) % N
    kx_fftw = kx_test

    # Image values
    img_fftw_val = d["img_fftw"][ky_fftw, kx_fftw]
    img_half_val = np.asarray(ftu.full_image_to_half_image(jnp.asarray(processed[None]), image_shape))[0].reshape(
        N, N // 2 + 1
    )[ky_test, kx_test]
    print(f"    img FFTW[{ky_fftw},{kx_fftw}] = {img_fftw_val:.6e}")
    print(f"    img half[{ky_test},{kx_test}] = {img_half_val:.6e}")
    img_ratio = img_half_val / (img_fftw_val + 1e-30)
    print(
        f"    img ratio (half/fftw) = {img_ratio:.6f}  (expect centering phase = {centering_phase[ky_test, kx_test]:.0f})"
    )

    # CTF values
    ctf_fftw_val = d["ctf_fftw"][ky_fftw, kx_fftw]
    ctf_half_val = np.asarray(ftu.full_image_to_half_image(jnp.asarray(ctf_full[None]), image_shape))[0].reshape(
        N, N // 2 + 1
    )[ky_test, kx_test]
    print(f"    CTF FFTW = {ctf_fftw_val:.6f}")
    print(f"    CTF half = {ctf_half_val:.6f}")
    print(f"    CTF ratio = {ctf_half_val / (ctf_fftw_val + 1e-30):.6f}  (expect -1.0)")

    # Projection values
    proj_fftw_val = projs_relion_fftw[0][ky_fftw, kx_fftw]
    proj_rhalf_val = projs_relion_rhalf[0].reshape(N, N // 2 + 1)[ky_test, kx_test]
    proj_phased_val = proj_8_h[ky_test, kx_test]
    print(f"    proj FFTW = {proj_fftw_val:.6e}")
    print(f"    proj rhalf = {proj_rhalf_val:.6e}  (ratio to FFTW: {proj_rhalf_val / (proj_fftw_val + 1e-30):.6f})")
    print(
        f"    proj phased = {proj_phased_val:.6e}  (ratio to rhalf: {proj_phased_val / (proj_rhalf_val + 1e-30):.6f}, expect phase={centering_phase[ky_test, kx_test]:.0f})"
    )

    # Noise weight
    nv_fftw_val = 2.0 * Minvsigma2[ky_fftw, kx_fftw]
    nv_half_val = 1.0 / noise_full.reshape(N, N)[ky_test, kx_test + N // 2]
    print(f"    noise weight FFTW = {nv_fftw_val:.6e}")
    print(f"    1/noise_full centered = {nv_half_val:.6e}")

    # ================================================================
    # STEP F9: Direct comparison — does CTF sign flip cancel with volume negation?
    # ================================================================
    print(f"\n{'=' * 70}")
    print("F9: Verify sign cancellation (CTF flip × volume negation)")
    print(f"{'=' * 70}")

    # In F6/F7, recovar uses CTF_recovar = -CTF_relion and proj_recovar ≈ -scale*proj_relion.
    # The double negation should cancel in the cross term but NOT in the formula sign:
    # cross = -2 Re Σ conj(img * (-CTF)/σ² * trans) * (-scale * proj)
    #       = -2 Re Σ conj(-CTF/σ²) * (-scale) * conj(img * trans) * proj
    #       = -2 Re Σ (+CTF/σ²) * scale * conj(img*trans) * proj   [CTF real]
    #       = -2 * scale * Re Σ (CTF/σ²) conj(img*trans) * proj
    #       = scale * cross_relion    ← POSITIVE multiple!
    #
    # So cross_recovar = scale * cross_relion (same sign!)
    # And norm_recovar = scale² * norm_relion (always positive)

    for img_i in range(min(2, n_images)):
        d = f4a_data[img_i]
        # From F6 data (which was computed above), check the sign
        raw = ds.image_source[img_i][0][0]
        ctf_params_i = jnp.asarray(ds.CTF_params[img_i : img_i + 1], dtype=jnp.float32)
        ctf_full = np.asarray(config.compute_ctf(ctf_params_i))[0]
        processed = np.asarray(ds.process_images(np.array([raw]), apply_image_mask=False))[0]
        weighted_img = processed * ctf_full / noise_full
        ctf2_nv_full = (ctf_full**2 / noise_full).reshape(1, -1)
        ctf2_nv_half = np.asarray(ftu.full_image_to_half_image(jnp.asarray(ctf2_nv_full), image_shape))[0]

        # Compute for (rot=0, trans=0)
        tx, ty = translations[0]
        shifted = np.asarray(
            translate_images(
                jnp.asarray(weighted_img[None]), jnp.asarray(np.array([[float(tx), float(ty)]])), image_shape
            )
        )[0]
        shifted_half = np.asarray(ftu.full_image_to_half_image(jnp.asarray(shifted[None]), image_shape))[0]

        proj_h = projs_recovar_half[0]
        cross_rc = -2.0 * np.sum(np.real(np.conj(shifted_half) * proj_h))
        norm_rc = np.sum(ctf2_nv_half * np.abs(proj_h) ** 2)

        cross_rl = d["cross"][0, 0]
        norm_rl = d["norm"][0, 0]

        sign_cross = np.sign(cross_rc) * np.sign(cross_rl)
        ratio_cross = cross_rc / (cross_rl + 1e-30)
        ratio_norm = norm_rc / (norm_rl + 1e-30)

        print(f"  img[{img_i}] (rot=0, trans=0):")
        print(f"    cross: recovar={cross_rc:.6e}  relion={cross_rl:.6e}")
        print(f"    cross ratio: {ratio_cross:.4f}  sign: {'SAME' if sign_cross > 0 else 'OPPOSITE'}")
        print(f"    norm:  recovar={norm_rc:.6e}  relion={norm_rl:.6e}")
        print(f"    norm ratio:  {ratio_norm:.4f}")
        print(f"    cross/norm (recovar): {abs(cross_rc) / norm_rc:.6f}")
        print(f"    cross/norm (relion):  {abs(cross_rl) / norm_rl:.6f}")
        print(
            f"    cross/norm ratio of ratios: {abs(cross_rc) / norm_rc / (abs(cross_rl) / norm_rl):.4f}  "
            f"(expect 1/scale ≈ {1 / mean_scale:.2e})"
        )

        # Verify: after dividing by scale, cross should match relion
        cross_norm_check = cross_rc / mean_scale
        norm_norm_check = norm_rc / mean_scale**2
        print(
            f"    After scale correction: cross_err={rel_err(cross_norm_check, cross_rl):.2e}  "
            f"norm_err={rel_err(norm_norm_check, norm_rl):.2e}"
        )

    print(f"\n{'=' * 70}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
