"""Clean, minimal-text FSC figure + separate legend. Trusted metric: full masked FSC
(broad mask, no mean subtraction) vs GT state 50."""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from matplotlib.lines import Line2D

WT = "/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_kernel_dev2_rebase"
sys.path.insert(0, WT)
import recovar.core.fourier_transform_utils as ftu
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
fg = np.asarray(ftu.get_dft3((load(f"{DL}/GT_state50.mrc") * mask).reshape(VS))).reshape(-1)
freq = np.arange(GRID // 2) / (GRID * VOX)


def fsc(p):
    fe = np.asarray(ftu.get_dft3((load(p) * mask).reshape(VS))).reshape(-1)
    return np.asarray(regularization.get_fsc(fe, fg, VS))[: len(freq)]


# dark = infinite-image limit, light = 1M ; red = deconvolution, blue = standard
SERIES = [
    ("deconvolution, ∞ images", f"{OUT}/deconv_limit_welltuned.mrc", "#c1272d", 3.4),
    ("deconvolution, 1M images", f"{DL}/best_deconvolved_eigenratio_3.99A.mrc", "#f0918a", 2.8),
    ("standard, ∞ images", f"{OUT}/standard_limit_welltuned.mrc", "#1b3f8b", 3.4),
    ("standard, 1M images", f"{DL}/best_convolved_standard_4.09A.mrc", "#84b6e0", 2.8),
]
curves = [(lbl, fsc(p), col, lw) for lbl, p, col, lw in SERIES]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "svg.fonttype": "none"})

# ---------- main plot: minimal text, no legend ----------
fig, ax = plt.subplots(figsize=(6.8, 5.1))
ax.axhline(0.5, color="#c9c9c9", ls=(0, (1, 2.5)), lw=1.3, zorder=0)
for lbl, c, col, lw in curves:  # limits drawn last (on top)
    ax.plot(freq[: len(c)], c, color=col, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=3)
ax.set_xlim(0, 0.4)
ax.set_ylim(0, 1.015)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax.set_yticks([0, 0.5, 1.0])
ax.set_xlabel("spatial frequency (1/Å)", fontsize=12.5, color="#2b2b2b", labelpad=7)
ax.set_ylabel("FSC", fontsize=12.5, color="#2b2b2b", labelpad=7)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#9a9a9a")
    ax.spines[sp].set_linewidth(1.1)
ax.tick_params(colors="#6b6b6b", length=4.5, width=1.0, labelsize=11.5)
fig.subplots_adjust(left=0.13, right=0.96, top=0.96, bottom=0.14)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fsc_main.{ext}", dpi=240)
print("WROTE fsc_main.png/.pdf")

# ---------- separate legend ----------
handles = [Line2D([0], [0], color=col, lw=3.6, solid_capstyle="round") for _, _, col, _ in SERIES]
labels = [lbl for lbl, _, _, _ in SERIES]
figl, axl = plt.subplots(figsize=(3.5, 1.7))
axl.axis("off")
leg = axl.legend(
    handles, labels, loc="center", frameon=False, fontsize=12.5, handlelength=2.1, labelspacing=1.0, borderpad=0
)
for t, (_, _, col, _) in zip(leg.get_texts(), SERIES):
    t.set_color("#2b2b2b")
for ext in ("png", "pdf"):
    figl.savefig(f"{OUT}/fsc_legend.{ext}", dpi=240, bbox_inches="tight")
print("WROTE fsc_legend.png/.pdf")
