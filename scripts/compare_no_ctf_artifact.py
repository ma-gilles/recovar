#!/usr/bin/env python
"""DEBUG: Compare recovar+CTF, recovar-noCTF, and RELION-noCTF volumes
to verify the dark halo / poor-resolution artifact comes from missing
CTF correction.

Loads:
  - GT volume
  - recovar 10k WITH CTF (production)
  - recovar 10k WITHOUT CTF (debug_recovar_no_ctf.py output)
  - RELION 10k WITHOUT --ctf (existing relion_ref_firstiter_cc run)

Computes:
  - radial average of central xy-slice (real space) — dark halo shows as
    a negative dip just outside the bright peak
  - 1D radial power spectrum — CTF oscillations show as wiggles in the
    no-CTF runs
  - central slice + projection plots, all with the same colorbar per row

Outputs PNGs to <out_dir>/.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from recovar.utils.helpers import load_mrc, load_relion_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def radial_average_2d(img):
    """Radial average of a 2D image, centered at the middle pixel."""
    h, w = img.shape
    yy, xx = np.indices((h, w))
    cy, cx = h / 2 - 0.5, w / 2 - 0.5
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rint = r.astype(int)
    rmax = min(h, w) // 2
    radial = np.zeros(rmax + 1)
    counts = np.zeros(rmax + 1)
    flat_r = rint.ravel()
    flat_v = img.ravel()
    for ri, vi in zip(flat_r, flat_v):
        if ri <= rmax:
            radial[ri] += vi
            counts[ri] += 1
    return radial / np.maximum(counts, 1)


def radial_power_3d(vol):
    """1D radial power spectrum of a 3D volume."""
    ft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vol)))
    pwr = np.abs(ft) ** 2
    n = vol.shape[0]
    coords = np.indices(vol.shape) - n / 2 + 0.5
    r = np.sqrt((coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2)).astype(int)
    rmax = n // 2
    bins = np.bincount(r.ravel(), weights=pwr.ravel(), minlength=rmax + 1)
    counts = np.bincount(r.ravel(), minlength=rmax + 1)
    return bins[: rmax + 1] / np.maximum(counts[: rmax + 1], 1)


def safe_load(path, frame):
    p = Path(path)
    if not p.exists():
        logger.warning("Missing %s", path)
        return None
    if frame == "relion":
        return np.asarray(load_relion_volume(str(p)), dtype=np.float32)
    return np.asarray(load_mrc(str(p)), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--recovar_with_ctf",
                        help="path to recovar+CTF final volume mrc")
    parser.add_argument("--recovar_no_ctf",
                        help="path to recovar-noCTF (debug) final volume mrc")
    parser.add_argument("--relion_no_ctf",
                        help="path to RELION-no-ctf class001.mrc")
    parser.add_argument("--relion_with_ctf",
                        help="path to RELION-WITH-ctf class001.mrc (verification)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = safe_load(Path(args.data_dir) / "reference_gt.mrc", "recovar")

    recovar_with = safe_load(args.recovar_with_ctf, "recovar") if args.recovar_with_ctf else None
    recovar_no = safe_load(args.recovar_no_ctf, "recovar") if args.recovar_no_ctf else None
    relion_no = safe_load(args.relion_no_ctf, "relion") if args.relion_no_ctf else None
    relion_with = safe_load(args.relion_with_ctf, "relion") if args.relion_with_ctf else None

    vols = {
        "GT": gt,
        "recovar +CTF": recovar_with,
        "recovar NO-CTF (debug)": recovar_no,
        "RELION +CTF": relion_with,
        "RELION NO-CTF": relion_no,
    }
    vols = {k: v for k, v in vols.items() if v is not None}
    logger.info("Loaded volumes: %s", list(vols.keys()))
    if not vols:
        logger.error("No volumes loaded; aborting")
        return

    # ----- z-score each volume so colorbars are comparable -----
    z = {k: (v - v.mean()) / (v.std() + 1e-12) for k, v in vols.items()}
    n = next(iter(z.values())).shape[0]
    mid = n // 2

    # ============================================================
    # Plot 1: central slice + projection, two rows, same color scale per row
    # ============================================================
    ncols = len(z)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
    if ncols == 1:
        axes = axes[:, None]
    slices = {k: v[mid] for k, v in z.items()}
    projs = {k: v.sum(axis=0) for k, v in z.items()}
    proj_means = np.array([(p - p.mean()) / (p.std() + 1e-12) for p in projs.values()])

    s_vmin = min(s.min() for s in slices.values())
    s_vmax = max(s.max() for s in slices.values())
    p_vmin, p_vmax = float(proj_means.min()), float(proj_means.max())

    for col, (name, vol_z) in enumerate(z.items()):
        sl = slices[name]
        pr = (projs[name] - projs[name].mean()) / (projs[name].std() + 1e-12)
        im0 = axes[0, col].imshow(sl, cmap="gray", vmin=s_vmin, vmax=s_vmax)
        axes[0, col].set_title(f"{name}\nz-score central slice (z={mid})", fontsize=11)
        axes[0, col].axis("off")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)
        im1 = axes[1, col].imshow(pr, cmap="gray", vmin=p_vmin, vmax=p_vmax)
        axes[1, col].set_title(f"{name}\nz-score sum-projection (axis=0)", fontsize=11)
        axes[1, col].axis("off")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)
    fig.suptitle("Central slice (top) and projection (bottom) — same color scale per row", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "vols_slice_proj.png", dpi=130, bbox_inches="tight")
    logger.info("Wrote %s", out_dir / "vols_slice_proj.png")

    # ============================================================
    # Plot 2: radial average of central slice — dark halo as a dip
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, vol in vols.items():
        ra = radial_average_2d(vol[mid])
        # z-score so all curves are comparable
        ra = (ra - ra.mean()) / (ra.std() + 1e-12)
        ax.plot(ra, label=name, lw=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("radius (pixels) from volume center")
    ax.set_ylabel("z-scored radial average of central slice")
    ax.set_title("Radial profile — dark halo from missing CTF correction\n"
                 "shows as a negative dip just outside the bright core")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "radial_profile.png", dpi=130, bbox_inches="tight")
    logger.info("Wrote %s", out_dir / "radial_profile.png")

    # ============================================================
    # Plot 3: log radial power spectrum — CTF oscillations
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, vol in vols.items():
        pwr = radial_power_3d(vol)
        # normalize by total power
        pwr_n = pwr / (pwr.sum() + 1e-12)
        ax.semilogy(pwr_n, label=name, lw=2)
    ax.set_xlabel("shell index (1 / Nyquist)")
    ax.set_ylabel("normalized radial power")
    ax.set_title("1D radial power spectrum — CTF oscillations show as wiggles\n"
                 "in no-CTF runs")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "radial_power.png", dpi=130, bbox_inches="tight")
    logger.info("Wrote %s", out_dir / "radial_power.png")

    print("Comparison plots written to:", out_dir)


if __name__ == "__main__":
    main()
