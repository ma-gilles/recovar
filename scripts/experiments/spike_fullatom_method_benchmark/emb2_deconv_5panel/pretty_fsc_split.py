"""Two minimal FSC plots (same axes/colors): one for the 1M estimates, one for the ∞ estimates.
deconvolution = red, standard = blue. Shared 2-entry legend. Trusted full masked FSC vs GT state 50."""

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
# match the rendered volume surface colors exactly (dark = ∞ limit, light = 1M)
DECONV_INF, STD_INF = "#b30000", "#08306b"
DECONV_1M, STD_1M = "#ff5b5b", "#4d94ff"


def load(p):
    with mrcfile.open(p, permissive=True) as m:
        return np.asarray(m.data, np.float32).reshape(-1)


mask = load(BROAD)
fg = np.asarray(ftu.get_dft3((load(f"{DL}/GT_state50.mrc") * mask).reshape(VS))).reshape(-1)
freq = np.arange(GRID // 2) / (GRID * VOX)


def fsc(p):
    fe = np.asarray(ftu.get_dft3((load(p) * mask).reshape(VS))).reshape(-1)
    return np.asarray(regularization.get_fsc(fe, fg, VS))[: len(freq)]


plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "svg.fonttype": "none"})


def make_plot(stem, deconv_vol, std_vol, deconv_col, std_col):
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.axhline(0.5, color="#c9c9c9", ls=(0, (1, 2.5)), lw=1.3, zorder=0)
    cs, cd = fsc(std_vol), fsc(deconv_vol)
    ax.plot(freq[: len(cs)], cs, color=std_col, lw=3.4, solid_capstyle="round", zorder=3)
    ax.plot(freq[: len(cd)], cd, color=deconv_col, lw=3.4, solid_capstyle="round", zorder=4)
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
    fig.subplots_adjust(left=0.13, right=0.96, top=0.97, bottom=0.14)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{stem}.{ext}", dpi=240)
    plt.close(fig)
    print(f"WROTE {stem}.png/.pdf")


make_plot("fsc_infty", f"{OUT}/deconv_limit_welltuned.mrc", f"{OUT}/standard_limit_welltuned.mrc", DECONV_INF, STD_INF)
make_plot(
    "fsc_1M",
    f"{DL}/best_deconvolved_eigenratio_3.99A.mrc",
    f"{DL}/best_convolved_standard_4.09A.mrc",
    DECONV_1M,
    STD_1M,
)


def make_legend(stem, deconv_col, std_col):
    figl, axl = plt.subplots(figsize=(2.7, 1.0))
    axl.axis("off")
    h = [
        Line2D([0], [0], color=deconv_col, lw=3.6, solid_capstyle="round"),
        Line2D([0], [0], color=std_col, lw=3.6, solid_capstyle="round"),
    ]
    leg = axl.legend(
        h,
        ["deconvolution", "standard"],
        loc="center",
        frameon=False,
        fontsize=13,
        handlelength=2.1,
        labelspacing=1.0,
        borderpad=0,
    )
    for t in leg.get_texts():
        t.set_color("#2b2b2b")
    for ext in ("png", "pdf"):
        figl.savefig(f"{OUT}/{stem}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(figl)
    print(f"WROTE {stem}.png/.pdf")


make_legend("fsc_legend_infty", DECONV_INF, STD_INF)
make_legend("fsc_legend_1M", DECONV_1M, STD_1M)
