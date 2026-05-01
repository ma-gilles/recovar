#!/usr/bin/env python
"""Ribosembly head-to-head plot: GT vs k-class refinement vs PPCA refinement.

Compares the existing eval cells:
  /scratch/gpfs/GILLES/mg6942/ppca_refine_eval_full/ribosembly__kclass__zdim4__iters8/
  /scratch/gpfs/GILLES/mg6942/ppca_refine_eval_full/ribosembly__dense__zdim4__iters8/

Plots three panels:
  1. GT volumes — central slice of each of the 4 GT classes.
  2. k-class reconstructed volumes — final per-class.
  3. PPCA reconstructed: μ + each of the q PCs (separate row).

Plus a diagnostic:
  4. PCA-of-k-class — take the K reconstructed k-class volumes, mean-center,
     SVD, plot the top q PCs side-by-side with the PPCA-refined PCs.

This tells us: does the PPCA refinement learn the same heterogeneity
subspace as a post-hoc PCA of the k-class output?

Output: <results-root>/ribosembly_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np


def _slice_central(vol: np.ndarray) -> np.ndarray:
    """Return the central z-slice of a (D, D, D) volume."""
    return vol[vol.shape[0] // 2]


def _normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1] for matplotlib imshow."""
    a = arr.astype(np.float32)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _load_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(str(path), permissive=True) as mrc:
        return mrc.data.copy().astype(np.float32)


def _load_gt_volumes(grid_size: int = 128, n_states: int = 4) -> np.ndarray:
    """Reuse the eval's loader for symmetry with the cells."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ppca_refine_eval import _CRYOBENCH_VOL_DIRS, _load_volumes_from_dir

    vol_dir = Path("/home/mg6942/mytigress/cryobench2") / _CRYOBENCH_VOL_DIRS["ribosembly"]
    vols_real, _, _ = _load_volumes_from_dir(str(vol_dir), n_states, grid_size)
    return vols_real  # (n_states, D, D, D)


def _pca_of_volumes(vols: np.ndarray, q: int):
    """Top-q PCA of (K, D, D, D) volumes. Returns (mu, pcs) with
    pcs shape (q, D, D, D) orthonormal in flat-voxel inner product."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="/scratch/gpfs/GILLES/mg6942/ppca_refine_eval_full",
    )
    parser.add_argument("--zdim", type=int, default=4)
    parser.add_argument("--em-iters", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.results_root)
    cell_id_kclass = f"ribosembly__kclass__zdim{args.zdim}__iters{args.em_iters}"
    cell_id_dense = f"ribosembly__dense__zdim{args.zdim}__iters{args.em_iters}"
    kdir = root / cell_id_kclass
    pdir = root / cell_id_dense

    if not kdir.exists() or not pdir.exists():
        raise FileNotFoundError(f"missing cells: {kdir.exists()=} {pdir.exists()=}")

    # --- Load GT ---
    print("loading GT volumes...")
    gt = _load_gt_volumes(grid_size=args.grid_size, n_states=4)  # (4, D, D, D)
    K_gt = gt.shape[0]

    # --- k-class output: per-class final reconstruction ---
    print("loading k-class reconstructed volumes...")
    kclass_vols = np.stack(
        [_load_mrc(kdir / f"class_{k:02d}.mrc") for k in range(K_gt)],
        axis=0,
    )  # (K, D, D, D)

    # --- PPCA dense output: μ + q PCs ---
    print("loading PPCA dense reconstructed (μ, W)...")
    last_iter = max(int(p.name.split("_")[-1]) for p in pdir.glob("iter_*") if p.is_dir())
    pdir_iter = pdir / f"iter_{last_iter:03d}"
    mu_ppca = _load_mrc(pdir_iter / "mu_score.mrc")
    W_ppca = np.stack(
        [_load_mrc(pdir_iter / f"W_{k:02d}_score.mrc") for k in range(args.zdim)],
        axis=0,
    )  # (q, D, D, D)

    # --- Diagnostic: PCA the k-class reconstructed volumes ---
    print("computing PCA of k-class reconstructed volumes...")
    mu_kpca, kpca_pcs = _pca_of_volumes(kclass_vols, q=args.zdim)

    # --- Build figure ---
    print("plotting...")
    n_rows = 4
    n_cols = max(K_gt, args.zdim)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols, 2.5 * n_rows + 0.6),
    )
    fig.suptitle(
        f"Ribosembly — k-class vs PPCA refinement (q={args.zdim}, {args.em_iters} EM iters, grid {args.grid_size})",
        fontsize=12,
    )
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    def _draw_row(row, vols, title_prefix, n_show):
        for c in range(n_cols):
            ax = axes[row, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < n_show:
                ax.imshow(_normalize_for_display(_slice_central(vols[c])), cmap="gray")
                ax.set_title(f"{title_prefix} {c}", fontsize=9)
            else:
                ax.axis("off")

    # Row 0: GT volumes
    _draw_row(0, gt, "GT class", K_gt)
    axes[0, 0].set_ylabel("GT volumes", fontsize=10, rotation=0, labelpad=70, ha="right")

    # Row 1: k-class reconstructed
    _draw_row(1, kclass_vols, "k-class", K_gt)
    axes[1, 0].set_ylabel("k-class refined", fontsize=10, rotation=0, labelpad=70, ha="right")

    # Row 2: PPCA — μ in column 0, then W_0..W_{q-1}
    full_ppca = np.concatenate([mu_ppca[None], W_ppca], axis=0)  # (q+1, D, D, D)
    n_show = min(args.zdim + 1, n_cols)
    for c in range(n_cols):
        ax = axes[2, c]
        ax.set_xticks([])
        ax.set_yticks([])
        if c < n_show:
            label = "μ" if c == 0 else f"W_{c - 1}"
            ax.imshow(_normalize_for_display(_slice_central(full_ppca[c])), cmap="gray")
            ax.set_title(f"PPCA {label}", fontsize=9)
        else:
            ax.axis("off")
    axes[2, 0].set_ylabel("PPCA refined", fontsize=10, rotation=0, labelpad=70, ha="right")

    # Row 3: PCA of k-class reconstructed (μ + q PCs)
    full_kpca = np.concatenate([mu_kpca[None], kpca_pcs], axis=0)
    n_show = min(args.zdim + 1, n_cols)
    for c in range(n_cols):
        ax = axes[3, c]
        ax.set_xticks([])
        ax.set_yticks([])
        if c < n_show:
            label = "μ" if c == 0 else f"PC_{c - 1}"
            ax.imshow(_normalize_for_display(_slice_central(full_kpca[c])), cmap="gray")
            ax.set_title(f"PCA(k-class) {label}", fontsize=9)
        else:
            ax.axis("off")
    axes[3, 0].set_ylabel("PCA of k-class\nrefined vols", fontsize=10, rotation=0, labelpad=70, ha="right")

    plt.tight_layout(rect=[0.06, 0.0, 1.0, 0.96])

    out_path = Path(args.out or root / "ribosembly_comparison.png")
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")

    # --- Numerical comparison: subspace agreement between PPCA W and PCA(k-class) PCs ---
    print("\nSubspace agreement (cos-of-principal-angles between PPCA W and PCA(k-class)):")
    W_flat = W_ppca.reshape(args.zdim, -1)
    kpca_flat = kpca_pcs.reshape(args.zdim, -1)
    # Orthonormalize both bases (W from production driver may not be exactly orthonormal in flat-voxel).
    Q_w, _ = np.linalg.qr(W_flat.T)
    Q_k, _ = np.linalg.qr(kpca_flat.T)
    # Singular values of Q_w^T Q_k are the cosines of the principal angles.
    cos_angles = np.linalg.svd(Q_w.T @ Q_k, compute_uv=False)
    print(f"  cos(principal angles) = {cos_angles}")
    print(f"  mean cos = {cos_angles.mean():.4f} (1.0 = perfect subspace match, 0 = orthogonal)")

    # Distance between PPCA μ and PCA(k-class) μ.
    mu_dist = float(np.linalg.norm(mu_ppca - mu_kpca) / np.linalg.norm(mu_kpca))
    print(f"  ||μ_ppca - μ_kpca|| / ||μ_kpca|| = {mu_dist:.4f}")


if __name__ == "__main__":
    main()
