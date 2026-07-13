"""Rebuild the INFINITY-IMAGE (limit) columns with the WELL-TUNED theoretical bounds
(reproduces best_infinite.py at emb2 sigma_z=30): deconv = best Tikhonov (tikh), standard =
best Epanechnikov std kernel, plus polynomial fit. Saves FULL volumes (+mean) and reports
conformational-signal (deviation) FSC AUC + FSC0.5 vs the weak per-image-eigen bound and the 1M recons."""

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
RUN = "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise1_b80_true_oracle_sweep_20260527/n01000000/runs/n01000000_seed0000"
BROAD = "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
GRID, VOX, K, TARGET, SIGZ = 256, 1.25, 100, 50, 30.0
VS = (GRID, GRID, GRID)
kk = np.arange(K)
zs = (kk - TARGET) / (K / 2.0)


def load(p):
    with mrcfile.open(p, permissive=True) as m:
        return np.asarray(m.data, np.float32).reshape(-1)


def blur(s):
    B = np.exp(-((kk[:, None] - kk[None, :]) ** 2) / (2 * s**2))
    return B / B.sum(0, keepdims=True)


def tikh(s, g):
    B = blur(s)
    d = B.mean(1)
    A = B.T
    e = np.zeros(K)
    e[TARGET] = 1.0
    a = A @ np.linalg.solve(A.T @ A + g * np.diag(d), A.T @ e)
    return a / a.sum()


def std_alpha(s, h):
    eps = np.linspace(-8 * s, 8 * s, 4001)
    phi = np.exp(-(eps**2) / (2 * s**2))
    phi /= phi.sum()
    u = ((kk - TARGET)[:, None] + eps[None, :]) / h
    Wm = np.where(np.abs(u) < 1, 0.75 * (1 - u**2), 0.0) / h
    g = (Wm * phi[None, :]).sum(1)
    return g / g.sum()


def poly_fixed(s, p, rcond=1e-9):
    B = blur(s)
    d = B.mean(1)
    P = np.polynomial.legendre.legvander(zs, p)
    ws = np.polynomial.legendre.legvander(np.array([0.0]), p)[0]
    C = P.T @ B.T
    Di = np.diag(1.0 / np.maximum(d, 1e-12))
    return B.T @ (Di @ C.T @ (np.linalg.pinv(C @ Di @ C.T, rcond=rcond) @ ws))


print("loading 100 GT vols...", flush=True)
Vfull = np.empty((K, GRID**3), np.float32)
for k in range(K):
    Vfull[k] = load(os.path.join(RUN, "04_ground_truth", f"gt_vol{k:04d}.mrc"))
Vmean = Vfull.mean(0)
Vdev = Vfull - Vmean[None, :]
mask = load(BROAD)
fg = np.asarray(ftu.get_dft3((Vdev[TARGET] * mask).reshape(VS))).reshape(-1)  # GT conformational signal


def dev_fsc(dev_vec):
    fe = np.asarray(ftu.get_dft3((dev_vec * mask).reshape(VS))).reshape(-1)
    return np.asarray(regularization.get_fsc(fe, fg, VS))


def res05(c):
    r = plot_utils.fsc_score(c, GRID, VOX, threshold=0.5)
    return 1.0 / float(r) if r > 0 else np.inf


def best(fn, knobs):
    bb = (-1, None, None)
    for kn in knobs:
        a = fn(kn)
        if not np.all(np.isfinite(a)) or np.sum(np.abs(a)) > 1e4:
            continue
        dev = a.astype(np.float32) @ Vdev
        c = dev_fsc(dev)
        if np.mean(c) > bb[0]:
            bb = (np.mean(c), c, a)
    return bb


def save_full(name, alpha):
    full = (alpha.astype(np.float32) @ Vdev + Vmean).astype(np.float32)
    with mrcfile.new(os.path.join(OUT, name), overwrite=True) as m:
        m.set_data(full.reshape(GRID, GRID, GRID))
        m.voxel_size = VOX


bd = best(lambda g: tikh(SIGZ, g), np.geomspace(1e2, 1e-6, 40))
bs = best(lambda h: std_alpha(SIGZ, h), np.geomspace(0.5, 60, 30))
bp = best(lambda p: poly_fixed(SIGZ, p), [2, 3, 4, 5, 6, 8, 10, 12, 16, 20])
save_full("deconv_limit_welltuned.mrc", bd[2])
save_full("standard_limit_welltuned.mrc", bs[2])
save_full("poly_limit_welltuned.mrc", bp[2])

# weak per-image-eigen bound (what I wrongly used) + 1M recons, on the SAME deviation metric
weak = dev_fsc(load(f"{OUT}/deconv_limit_full.mrc") - Vmean)
d1m = dev_fsc(load(f"{DL}/best_deconvolved_eigenratio_3.99A.mrc") - Vmean)
s1m = dev_fsc(load(f"{DL}/best_convolved_standard_4.09A.mrc") - Vmean)
freq = np.arange(GRID // 2) / (GRID * VOX)

series = [
    ("deconv — limit, WELL-TUNED (∞)", bd[1], "#b30000", "-", 2.8),
    ("polynomial fit — limit (∞)", bp[1], "#7a0bbf", "-.", 2.0),
    ("deconv — limit, per-image-eigen (∞) [old]", weak, "#b30000", ":", 1.8),
    ("standard — limit (∞)", bs[1], "#08306b", "-", 2.6),
    ("deconv — 1M images", d1m, "#ff5b5b", "-", 1.8),
    ("standard — 1M images", s1m, "#4d94ff", "-", 1.8),
]
print("\n=== conformational-signal (deviation) FSC vs GT state 50 ===", flush=True)
fig, ax = plt.subplots(figsize=(9.4, 6.3))
for label, c, col, ls, lw in series:
    c = c[: len(freq)]
    auc = float(np.mean(c))
    r = res05(c)
    rt = f"{r:.2f} Å" if np.isfinite(r) else "≥Nyq"
    ax.plot(freq[: len(c)], c, color=col, ls=ls, lw=lw, label=f"{label} — AUC {auc:.3f}, FSC0.5 {rt}")
    print(f"  {label:46s} AUC={auc:.3f}  FSC0.5={rt}", flush=True)
ax.axhline(0.5, color="0.5", ls=":", lw=1)
ax.set_xlim(0, 0.4)
ax.set_ylim(0, 1.02)
ax.set_xlabel("spatial frequency (1/Å)")
ax.set_ylabel("FSC of conformational signal vs GT state 50 (broad mask)")
ax.set_title("Spike state 50, noise=1, emb2 (σ_z≈30): well-tuned ∞-image deconvolution vs standard vs 1M")
tk = [(10, "10Å"), (6, "6Å"), (4, "4Å"), (3, "3Å"), (2.5, "2.5Å")]
ax.set_xticks([1.0 / r for r, _ in tk])
ax.set_xticklabels([f"{1.0 / r:.2f}\n({lab})" for r, lab in tk], fontsize=8)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/fsc_welltuned_deviation.png", dpi=150)
print("WROTE fsc_welltuned_deviation.png", flush=True)
print(f"\ntikh best gamma sweep AUC={bd[0]:.3f}; std AUC={bs[0]:.3f}; poly AUC={bp[0]:.3f}", flush=True)
