"""Masked FSC vs GT state 50 (broad mask) for the EXACT volumes in the 5-panel, all full-volume,
same metric as FSC_state50_recovery.png. Adds the theoretical optimal (infinite-image ceiling)."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

WT = "/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_kernel_dev2_rebase"
sys.path.insert(0, WT)
import recovar.core.fourier_transform_utils as ftu
from recovar.output import plot_utils
from recovar.reconstruction import regularization

OUT = "/scratch/gpfs/CRYOEM/gilleslab/tmp/emb2_noise1_movingmask_5panel_20260607"
DL = "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_gt_embeddings_noise1_dataset_20260601/download_emb2_noise1"
BROAD = "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
GRID, VOX = 256, 1.25
VS = (GRID, GRID, GRID)


def load(p):
    with mrcfile.open(p, permissive=True) as m:
        return np.asarray(m.data, np.float32).reshape(-1)


mask = load(BROAD)
gt = load(f"{DL}/GT_state50.mrc")
fg = np.asarray(ftu.get_dft3((gt * mask).reshape(VS))).reshape(-1)
freq = np.arange(GRID // 2) / (GRID * VOX)


def fsc(v):
    fe = np.asarray(ftu.get_dft3((v * mask).reshape(VS))).reshape(-1)
    return np.asarray(regularization.get_fsc(fe, fg, VS))


def res05(c):
    r = plot_utils.fsc_score(c, GRID, VOX, threshold=0.5)
    return 1.0 / float(r) if r > 0 else np.inf


CURVES = [
    ("deconvolved — limit (∞ img)", f"{OUT}/deconv_limit_full.mrc", "#b30000", "-", 2.6),
    ("deconvolved — 1M images", f"{DL}/best_deconvolved_eigenratio_3.99A.mrc", "#ff5b5b", "-", 2.2),
    ("standard — limit (∞ img)", f"{OUT}/standard_limit_full.mrc", "#08306b", "-", 2.6),
    ("standard — 1M images", f"{DL}/best_convolved_standard_4.09A.mrc", "#4d94ff", "-", 2.2),
    ("theoretical optimal (∞ img, best linear)", f"{DL}/theoretical_optimal_infimages_1.25A.mrc", "k", "--", 2.4),
]

fig, ax = plt.subplots(figsize=(9.2, 6.2))
rows = []
for label, path, col, ls, lw in CURVES:
    if not os.path.exists(path):
        print("MISSING", path)
        continue
    c = fsc(load(path))[: len(freq)]
    r = res05(c)
    auc = float(np.mean(c))
    rtxt = f"{r:.2f} Å" if np.isfinite(r) else "≥Nyq"
    ax.plot(freq[: len(c)], c, color=col, ls=ls, lw=lw, label=f"{label} — FSC0.5 {rtxt}, AUC {auc:.3f}")
    rows.append((label, rtxt, auc))
    print(f"{label:42s} FSC0.5={rtxt:8s} AUC={auc:.4f}", flush=True)

ax.axhline(0.5, color="0.5", ls=":", lw=1)
ax.set_xlim(0, 0.4)
ax.set_ylim(0, 1.02)
ax.set_xlabel("spatial frequency (1/Å)", fontsize=11)
ax.set_ylabel("masked FSC vs GT state 50 (broad mask)", fontsize=11)
ax.set_title("Spike state 50, noise=1, emb2 (σ_z≈30): FSC of the 5-panel volumes vs GT", fontsize=12)
tk = [(10, "10Å"), (6, "6Å"), (4, "4Å"), (3, "3Å"), (2.5, "2.5Å")]
ax.set_xticks([1.0 / r for r, _ in tk])
ax.set_xticklabels([f"{1.0 / r:.2f}\n({lab})" for r, lab in tk], fontsize=8)
ax.legend(fontsize=9, loc="lower left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/fsc_5panel_volumes.png", dpi=150)
print("WROTE", f"{OUT}/fsc_5panel_volumes.png", flush=True)
