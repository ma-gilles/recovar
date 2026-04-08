#!/usr/bin/env python
"""DEBUG-ONLY: quantitative falsifiable test that RELION's reference volume
on this benchmark is `IFT(CTF * FT(GT))` — i.e. CTF-convolved instead of
CTF-corrected.

Falsifiable prediction: if RELION ran without --ctf, then in Fourier space:
    |vol_relion(k)|^2 ≈ |vol_GT(k)|^2 * <CTF^2(k)>
where <CTF^2(k)> is the radial average of squared CTF over all particles
in the dataset.

So the ratio |vol_relion|^2 / |vol_GT|^2 should match <CTF^2(k)> at every
shell. In particular, the FIRST RING of low power in the relion volume
should sit exactly at the FIRST ZERO of <CTF^2(k)>.

This script:
  1. Loads the dataset's per-particle CTF parameters
  2. Evaluates each particle's |CTF(k)|^2 on a 2D grid and averages them
  3. Radially averages the result -> <CTF^2>(shell)
  4. Computes the radial power spectrum of the GT volume (recovar frame)
  5. Computes the radial power spectrum of the RELION-no-ctf volume
  6. Plots all three on the same axes, plus the ratio

If <CTF^2> and the ratio overlap, the diagnosis is proven.
"""

import argparse
import logging
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.core.ctf import CTFEvaluator, CTFMode, evaluate_ctf
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.utils.helpers import load_mrc, load_relion_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def radial_average_2d(arr2d):
    """Radial average of a 2D array (centered)."""
    h, w = arr2d.shape
    yy, xx = np.indices((h, w))
    cy, cx = h / 2 - 0.5, w / 2 - 0.5
    r = np.round(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(int)
    rmax = min(h, w) // 2
    out = np.zeros(rmax + 1)
    cnt = np.zeros(rmax + 1)
    for ri, val in zip(r.ravel(), arr2d.ravel()):
        if ri <= rmax:
            out[ri] += val
            cnt[ri] += 1
    return out / np.maximum(cnt, 1)


def radial_average_3d(arr3d):
    """Radial average of a 3D array (centered)."""
    n = arr3d.shape[0]
    coords = np.indices(arr3d.shape) - n / 2 + 0.5
    r = np.round(np.sqrt(coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2)).astype(int)
    rmax = n // 2
    out = np.bincount(r.ravel(), weights=arr3d.ravel(), minlength=rmax + 1)
    cnt = np.bincount(r.ravel(), minlength=rmax + 1)
    return out[: rmax + 1] / np.maximum(cnt[: rmax + 1], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--gt_mrc", required=True, help="path to GT volume mrc")
    parser.add_argument("--relion_no_ctf", required=True,
                        help="path to RELION-no-ctf class001.mrc")
    parser.add_argument("--out", required=True, help="output PNG path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ---- 1. Load dataset and per-particle CTF params ----
    logger.info("Loading dataset to extract CTF params...")
    ds = load_dataset(str(data_dir / "particles.star"), lazy=True)
    n_imgs = ds.n_units
    image_shape = ds.image_shape
    voxel_size = ds.voxel_size

    ctf_params = np.asarray(ds.CTF_params)
    logger.info("Loaded %d particles, image_shape=%s, voxel_size=%.3f A",
                n_imgs, image_shape, voxel_size)
    logger.info("CTF params shape: %s", ctf_params.shape)
    logger.info("Defocus stats: U min=%.0f median=%.0f max=%.0f",
                ctf_params[:, 0].min(),
                np.median(ctf_params[:, 0]),
                ctf_params[:, 0].max())

    # ---- 2. Compute <CTF^2(k)> averaged over all particles ----
    logger.info("Computing per-particle CTF on 2D grid...")
    n_subset = min(2000, n_imgs)  # for speed
    subset = ctf_params[:n_subset]

    # Use the CTFEvaluator for SPA mode
    evaluator = CTFEvaluator(mode=CTFMode.SPA)
    ctf_per_image = np.asarray(evaluator(jnp.asarray(subset),
                                          image_shape, voxel_size,
                                          half_image=False))
    # ctf_per_image shape: (n_subset, H*W)
    ctf2_avg_2d = (ctf_per_image ** 2).mean(axis=0).reshape(image_shape)
    logger.info("|CTF^2| avg shape: %s, min/max=%.4e/%.4e",
                ctf2_avg_2d.shape, ctf2_avg_2d.min(), ctf2_avg_2d.max())

    # FFT-shift so the array is centered (CTF helpers use centered grid already? let's check)
    # The CTFEvaluator uses fourier_transform_utils.get_k_coordinate_of_each_pixel
    # which returns a grid with DC at center. So the result is already centered.
    # Radial average of the centered |CTF^2|:
    ctf2_radial = radial_average_2d(ctf2_avg_2d)

    # ---- 3. Load GT volume and compute its radial power ----
    logger.info("Loading GT volume from %s", args.gt_mrc)
    gt_vol = np.asarray(load_mrc(args.gt_mrc), dtype=np.float32)
    gt_ft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(gt_vol)))
    gt_power_radial = radial_average_3d(np.abs(gt_ft) ** 2)

    # ---- 4. Load RELION-no-ctf volume and compute its radial power ----
    logger.info("Loading RELION-no-ctf volume from %s", args.relion_no_ctf)
    relion_vol = np.asarray(load_relion_volume(args.relion_no_ctf), dtype=np.float32)
    relion_ft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(relion_vol)))
    relion_power_radial = radial_average_3d(np.abs(relion_ft) ** 2)

    # ---- 5. Compute the falsifiable ratio ----
    # The prediction: relion_power(k) / gt_power(k) ~ <CTF^2(k)>  (up to a constant scale)
    eps = 1e-30
    ratio_relion = relion_power_radial / (gt_power_radial + eps)
    # Normalize so ratio at DC = ctf2_radial at DC
    n_match = min(len(ctf2_radial), len(ratio_relion))
    if ctf2_radial[0] > 0 and ratio_relion[0] > 0:
        ratio_relion *= ctf2_radial[0] / ratio_relion[0]

    # ---- 6. Find the first zero of <CTF^2> for annotation ----
    smooth = ctf2_radial[:n_match]
    # find first local minimum
    first_dip_shell = -1
    for i in range(2, n_match - 2):
        if smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1] and smooth[i] < 0.1 * smooth[0]:
            first_dip_shell = i
            break
    logger.info("First CTF zero at shell ~%d (resolution = %.1f A)",
                first_dip_shell,
                gt_vol.shape[0] * voxel_size / first_dip_shell if first_dip_shell > 0 else 0)

    # ---- 7. Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    shells = np.arange(n_match)
    ax.plot(shells, ctf2_radial[:n_match], "k-", lw=2.5,
            label=r"$\langle CTF^2 \rangle (k)$  (analytical, from particle defocus)")
    ax.plot(shells, ratio_relion[:n_match], "r--", lw=2,
            label=r"$|F[\mathrm{RELION-noCTF}]|^2 / |F[\mathrm{GT}]|^2$  (measured)")
    if first_dip_shell > 0:
        ax.axvline(first_dip_shell, color="gray", ls=":", alpha=0.7,
                   label=f"first CTF zero (shell {first_dip_shell})")
    ax.set_xlabel("radial shell index")
    ax.set_ylabel("normalized power")
    ax.set_title("Falsifiable test: if RELION ran without CTF correction,\n"
                 r"$|F[\mathrm{vol}]|^2 / |F[\mathrm{GT}]|^2$  should match  $\langle CTF^2 \rangle$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    ax = axes[1]
    ax.plot(shells, ctf2_radial[:n_match], "k-", lw=2.5,
            label=r"$\langle CTF^2 \rangle (k)$")
    # zoom into the first zero region
    ax.plot(shells, ratio_relion[:n_match], "r--", lw=2,
            label="measured ratio")
    if first_dip_shell > 0:
        ax.axvline(first_dip_shell, color="gray", ls=":", alpha=0.7)
    ax.set_xlim(0, min(n_match, 60))
    ax.set_ylim(1e-3, 2.0)
    ax.set_yscale("log")
    ax.set_xlabel("radial shell index")
    ax.set_ylabel("normalized power")
    ax.set_title("Same plot, zoomed to first CTF zero")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "If the two curves overlap, RELION's reference volume IS the "
        "CTF-convolved GT (i.e. ran without --ctf).",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
