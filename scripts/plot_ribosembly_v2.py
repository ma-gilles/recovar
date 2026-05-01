#!/usr/bin/env python
"""Ribosembly comparison plot — v2 k-class (20k img, HEALPix order 1, 15 iters)
with v1 PPCA dense/local (5k img, 32 rotations, 8 iters) until v2 PPCA cells finish.

Produces three figure files:

  ribosembly_v2_slices.png   — three orthogonal central slices (xy/xz/yz)
                               for: GT, k-class refined, PPCA dense μ+W,
                               PPCA local μ+W, and PCA-of-k-class.
  ribosembly_v2_fsc.png      — pair-wise FSC curves: each predicted vs its
                               Hungarian-matched GT class.
  ribosembly_v2_subspace.txt — numerical subspace agreement
                               (cos of principal angles between
                                {PPCA dense W, PPCA local W, PCA-of-kclass}
                                and the W of PCA-of-GT).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np


def _load_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(str(path), permissive=True) as mrc:
        return mrc.data.copy().astype(np.float32)


def _normalize(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _three_slices(vol: np.ndarray):
    """Return central xy, xz, yz slices."""
    D = vol.shape[0]
    return [vol[D // 2, :, :], vol[:, D // 2, :], vol[:, :, D // 2]]


def _load_gt_volumes(grid_size: int = 128, n_states: int = 4) -> np.ndarray:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ppca_refine_eval import _CRYOBENCH_VOL_DIRS, _load_volumes_from_dir

    vol_dir = Path("/home/mg6942/mytigress/cryobench2") / _CRYOBENCH_VOL_DIRS["ribosembly"]
    vols_real, _, _ = _load_volumes_from_dir(str(vol_dir), n_states, grid_size)
    return vols_real


def _pca_of_volumes(vols: np.ndarray, q: int):
    K = vols.shape[0]
    vol_shape = vols.shape[1:]
    flat = vols.reshape(K, -1).astype(np.float32)
    mu_flat = flat.mean(axis=0)
    centered = flat - mu_flat[None, :]
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    n = min(q, Vt.shape[0])
    pcs = Vt[:n].astype(np.float32).reshape((n,) + vol_shape)
    if n < q:
        pcs = np.concatenate([pcs, np.zeros((q - n,) + vol_shape, dtype=np.float32)], axis=0)
    mu = mu_flat.reshape(vol_shape).astype(np.float32)
    return mu, pcs


def _fsc_curve(vol1: np.ndarray, vol2: np.ndarray, vol_shape: tuple) -> np.ndarray:
    from recovar.reconstruction.regularization import get_fsc

    return np.asarray(get_fsc(vol1.reshape(-1), vol2.reshape(-1), vol_shape))


def _hungarian_match(pred_vols: np.ndarray, gt_vols: np.ndarray, vol_shape: tuple):
    from scipy.optimize import linear_sum_assignment

    K_pred, K_gt = pred_vols.shape[0], gt_vols.shape[0]
    fsc_curves = np.zeros((K_pred, K_gt), dtype=object)
    fsc_areas = np.zeros((K_pred, K_gt), dtype=np.float32)
    for i in range(K_pred):
        for j in range(K_gt):
            curve = _fsc_curve(pred_vols[i], gt_vols[j], vol_shape)
            fsc_curves[i, j] = curve
            fsc_areas[i, j] = float(np.mean(curve))
    row, col = linear_sum_assignment(-fsc_areas[:K_pred, :K_gt])
    return list(zip(row.tolist(), col.tolist())), fsc_curves


def _ppca_trial_volumes(mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Build (q+1) "extreme" volumes from (μ, W) by sweeping z = +1 along
    each PC direction (signed, scaled to μ RMS). To compare cardinality
    1:1 with K=4 GT classes when q=3, we get 4 trial volumes here.
    For q=4 we get 5 trials."""
    rms_mu = float(np.sqrt(np.mean(mu**2)) + 1e-12)
    trials = [mu]
    for k in range(W.shape[0]):
        rms_W = float(np.sqrt(np.mean(W[k] ** 2)) + 1e-12)
        scale = rms_mu / rms_W if rms_W > 0 else 0.0
        trials.append(mu + scale * W[k])
    return np.stack(trials)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", default="/scratch/gpfs/GILLES/mg6942/ppca_refine_eval_ribosembly_v2")
    parser.add_argument("--v1-root", default="/scratch/gpfs/GILLES/mg6942/ppca_refine_eval_full")
    parser.add_argument("--zdim", type=int, default=4)
    parser.add_argument(
        "--out-prefix", default="/scratch/gpfs/GILLES/mg6942/ppca_refine_eval_ribosembly_v2/ribosembly_v2"
    )
    args = parser.parse_args()

    v2 = Path(args.v2_root)
    v1 = Path(args.v1_root)

    print("loading GT (grid=128)...")
    gt = _load_gt_volumes(grid_size=128, n_states=4)
    K = gt.shape[0]
    vol_shape = gt.shape[1:]

    # v2 k-class.
    kdir = v2 / f"ribosembly__kclass__zdim{args.zdim}__iters15"
    if (kdir / "class_00.mrc").exists():
        kclass_vols = np.stack([_load_mrc(kdir / f"class_{k:02d}.mrc") for k in range(K)])
        kclass_label = "k-class (v2: 20k img, 576 rot, 15 iters)"
    else:
        # Fall back to v1.
        kdir = v1 / f"ribosembly__kclass__zdim{args.zdim}__iters8"
        kclass_vols = np.stack([_load_mrc(kdir / f"class_{k:02d}.mrc") for k in range(K)])
        kclass_label = "k-class (v1: 5k img, 32 rot, 8 iters)"
    print(f"loaded k-class: {kclass_label}")

    # v2 PPCA (if available) else v1.
    def _load_ppca(root: Path, mode: str, iters: int):
        cell = root / f"ribosembly__{mode}__zdim{args.zdim}__iters{iters}"
        # locate latest iter dir
        iters_dirs = sorted(p for p in cell.glob("iter_*") if p.is_dir())
        if not iters_dirs:
            return None
        last = iters_dirs[-1]
        if not (last / "mu_score.mrc").exists():
            return None
        mu = _load_mrc(last / "mu_score.mrc")
        W = np.stack([_load_mrc(last / f"W_{k:02d}_score.mrc") for k in range(args.zdim)])
        return mu, W

    ppca_dense = _load_ppca(v2, "dense", 15)
    ppca_dense_label = "PPCA dense (v2: 20k img, 576 rot, 15 iters)"
    if ppca_dense is None:
        ppca_dense = _load_ppca(v1, "dense", 8)
        ppca_dense_label = "PPCA dense (v1: 5k img, 32 rot, 8 iters)"
    ppca_local = _load_ppca(v2, "local", 15)
    ppca_local_label = "PPCA local (v2: 20k img, local hyp, 15 iters)"
    if ppca_local is None:
        ppca_local = _load_ppca(v1, "local", 8)
        ppca_local_label = "PPCA local (v1: 5k img, local hyp, 8 iters)"
    print(f"loaded ppca dense: {ppca_dense_label}")
    print(f"loaded ppca local: {ppca_local_label}")

    # PCA of k-class output.
    mu_kpca, W_kpca = _pca_of_volumes(kclass_vols, q=args.zdim)
    # PCA of GT (the "oracle" subspace).
    mu_gt, W_gt = _pca_of_volumes(gt, q=args.zdim)

    # ----- Figure 1: three orthogonal central slices -----
    print("plotting slices...")
    rows = [
        ("GT classes", gt),
        (kclass_label, kclass_vols),
        (f"PPCA dense μ + W   [{ppca_dense_label}]", _ppca_trial_volumes(ppca_dense[0], ppca_dense[1])),
        (f"PPCA local μ + W   [{ppca_local_label}]", _ppca_trial_volumes(ppca_local[0], ppca_local[1])),
        ("PCA-of-k-class μ + W", np.concatenate([mu_kpca[None], W_kpca], axis=0)),
        ("PCA-of-GT (oracle) μ + W", np.concatenate([mu_gt[None], W_gt], axis=0)),
    ]
    n_cols = max(K, args.zdim + 1)
    n_rows = len(rows) * 3  # 3 axes per row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 1.8 * n_rows + 1.0))
    fig.suptitle("Ribosembly: GT vs k-class vs PPCA refinement (3 central slices)", fontsize=11)
    axis_names = ["xy", "xz", "yz"]
    for ri, (label, vols) in enumerate(rows):
        for ai in range(3):
            for ci in range(n_cols):
                ax = axes[ri * 3 + ai, ci]
                ax.set_xticks([])
                ax.set_yticks([])
                if ci < vols.shape[0]:
                    slc = _three_slices(vols[ci])[ai]
                    ax.imshow(_normalize(slc), cmap="gray")
                    if ai == 0:
                        ax.set_title(f"{ci}", fontsize=8)
                else:
                    ax.axis("off")
                if ci == 0:
                    if ai == 1:
                        ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=80, ha="right", va="center")
                    elif ai == 0:
                        ax.text(-0.45, 0.5, axis_names[ai], transform=ax.transAxes, fontsize=7, ha="right", va="center")
                    else:
                        ax.text(-0.45, 0.5, axis_names[ai], transform=ax.transAxes, fontsize=7, ha="right", va="center")
    plt.tight_layout(rect=[0.08, 0.0, 1.0, 0.97])
    out_slices = f"{args.out_prefix}_slices.png"
    fig.savefig(out_slices, dpi=130)
    plt.close(fig)
    print(f"wrote {out_slices}")

    # ----- Figure 2: FSC curves vs GT (Hungarian-matched) -----
    print("plotting FSC curves...")

    pred_sets = [
        ("k-class", kclass_vols),
        ("PPCA dense", _ppca_trial_volumes(ppca_dense[0], ppca_dense[1])),
        ("PPCA local", _ppca_trial_volumes(ppca_local[0], ppca_local[1])),
        ("PCA-of-k-class", np.concatenate([mu_kpca[None], W_kpca], axis=0)),
        ("PCA-of-GT (oracle)", np.concatenate([mu_gt[None], W_gt], axis=0)),
    ]

    fig, axes = plt.subplots(1, len(pred_sets), figsize=(3.5 * len(pred_sets), 3.5), sharey=True)
    for ax, (name, preds) in zip(axes, pred_sets):
        matched, fsc_curves = _hungarian_match(preds, gt, vol_shape)
        for i, j in matched:
            curve = fsc_curves[i, j]
            ax.plot(np.arange(len(curve)), curve, lw=1.0, label=f"pred {i} → GT {j}")
        ax.axhline(0.5, color="k", linestyle="--", lw=0.5, alpha=0.5)
        ax.axhline(0.143, color="k", linestyle=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("Fourier shell")
        ax.set_title(name, fontsize=9)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("FSC")
    plt.tight_layout()
    out_fsc = f"{args.out_prefix}_fsc.png"
    fig.savefig(out_fsc, dpi=140)
    plt.close(fig)
    print(f"wrote {out_fsc}")

    # ----- Figure 3: subspace alignment numbers (text) -----
    def _principal_angle_cosines(W1: np.ndarray, W2: np.ndarray):
        Q1, _ = np.linalg.qr(W1.reshape(W1.shape[0], -1).T)
        Q2, _ = np.linalg.qr(W2.reshape(W2.shape[0], -1).T)
        s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
        return s

    print("\nSubspace alignment vs PCA-of-GT (oracle):")
    lines = ["# Subspace alignment vs PCA-of-GT (oracle)\n"]
    lines.append(f"# zdim={args.zdim}, GT vols={K}\n\n")
    for name, W in [
        ("PPCA dense W", ppca_dense[1]),
        ("PPCA local W", ppca_local[1]),
        ("PCA-of-k-class W", W_kpca),
    ]:
        s = _principal_angle_cosines(W, W_gt)
        msg = f"{name:<22s}  cos(principal angles) = {np.round(s, 4)}  mean = {s.mean():.4f}"
        print(msg)
        lines.append(msg + "\n")
    out_txt = f"{args.out_prefix}_subspace.txt"
    Path(out_txt).write_text("".join(lines))
    print(f"wrote {out_txt}")


if __name__ == "__main__":
    main()
