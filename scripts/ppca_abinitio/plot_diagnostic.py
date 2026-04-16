"""Comprehensive multi-panel diagnostic figure for PPCA CryoBench Ribosembly.

Generates and saves:
  /scratch/gpfs/GILLES/mg6942/tmp/ppca_diagnostic_figure.png

Two experimental conditions:
  1. No annealing (baseline, n_joint=15 for speed)
  2. log1000 annealing (--anneal-schedule log1000 --anneal-iters 30, n_joint=30)

Usage:
    pixi run python scripts/ppca_abinitio/plot_diagnostic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# When run as a script file, sys.path[0] is the script's directory.
# Add the repo root so that `from scripts.ppca_abinitio.run_cryobench import ...`
# and `from recovar...` work correctly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Codebase imports
# ---------------------------------------------------------------------------
print("[0] Importing recovar modules...", flush=True)

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.posterior import score_and_posterior_moments_eqx
from recovar.em.ppca_abinitio.synthetic import SyntheticFamily, make_synthetic_fixed_grid_dataset
from scripts.ppca_abinitio.run_cryobench import (
    _Cfg,
    build_anneal_schedule,
    downsample_volume,
    load_cryobench_gt_volumes,
    resolve_n_burnin,
    run_two_stage,
    summarize_discrete_embedding,
)

print("  imports OK", flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("/home/mg6942/mytigress/cryobench2/Ribosembly")
OUTPUT_PATH = Path("/scratch/gpfs/GILLES/mg6942/tmp/ppca_diagnostic_figure.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

Q = 8
VOL = 32
N_IMAGES = 1024
SIGMA = 0.01
HEALPIX_ORDER = 1
N_STATES = 16  # Ribosembly has 16 discrete states
N_JOINT_BASELINE = 15  # shorter run for baseline (speed)
N_JOINT_ANNEAL = 30  # full run for annealed condition
SEED = 0

# ---------------------------------------------------------------------------
# Step 1: Load GT volumes and build dataset
# ---------------------------------------------------------------------------
print("\n[1] Loading CryoBench GT volumes...", flush=True)
t0 = time.perf_counter()
gt_vols = load_cryobench_gt_volumes(DATASET_ROOT, target_D=VOL)
print(
    f"  Loaded {gt_vols.shape[0]} volumes of shape {gt_vols.shape[1:]}  ({time.perf_counter() - t0:.1f}s)", flush=True
)

print("\n[2] Building synthetic fixed-grid dataset...", flush=True)
t0 = time.perf_counter()
grid = build_fixed_grid(healpix_order=HEALPIX_ORDER, max_shift=1)
image_shape = (VOL, VOL)
volume_shape = (VOL, VOL, VOL)

ds = make_synthetic_fixed_grid_dataset(
    SyntheticFamily.MATCHED_GRID_HET,
    volume_shape=volume_shape,
    image_shape=image_shape,
    grid=grid,
    q=Q,
    n_images_train=N_IMAGES,
    n_images_val=0,
    sigma_real=SIGMA,
    seed=SEED,
    external_volumes_real=gt_vols,
    external_sampling_mode="discrete_volumes",
)
cfg = _Cfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)
print(f"  n_img={ds.n_img}, n_rot={ds.n_rot}, n_trans={ds.n_trans}  ({time.perf_counter() - t0:.1f}s)", flush=True)
print(f"  s_true: {np.asarray(ds.s_true)}", flush=True)


# ---------------------------------------------------------------------------
# Helper: reconstruct a real-space volume from half-vol Fourier representation
# ---------------------------------------------------------------------------
def half_to_real_vol(half_vec, volume_shape):
    """Convert a half-volume complex vector back to real-space 3D array."""
    N0, N1, N2 = volume_shape
    half_shape = (N0, N1, N2 // 2 + 1)
    hv = jnp.asarray(half_vec).reshape(half_shape)
    real_vol = np.asarray(ftu.get_idft3_real(hv, volume_shape=volume_shape))
    return real_vol


def get_alpha_hat(cfg, init, ds):
    """Compute marginal latent embeddings alpha_hat for all images."""
    stats = score_and_posterior_moments_eqx(
        cfg,
        init.mu,
        init.U,
        init.s,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
    )
    pm = np.asarray(stats.post_mean)  # (n_img, n_rot, n_trans, q)
    gamma = np.exp(np.asarray(stats.log_resp))  # (n_img, n_rot, n_trans)
    alpha_hat = np.sum(gamma[..., None] * pm, axis=(1, 2))  # (n_img, q)
    return alpha_hat, stats


def reconstruct_cluster_volumes(init, alpha_hat, state_labels, n_states, volume_shape):
    """For each cluster (by k-means), compute mean volume in real space."""
    km = KMeans(n_clusters=n_states, n_init=10, random_state=0)
    cluster_labels = km.fit_predict(alpha_hat)

    mu_real = half_to_real_vol(np.asarray(init.mu), volume_shape)
    U_np = np.asarray(init.U)  # (q, half_vol_size)

    cluster_vols = []
    for k in range(n_states):
        mask = cluster_labels == k
        if mask.sum() == 0:
            cluster_vols.append(mu_real)
            continue
        mean_alpha = alpha_hat[mask].mean(axis=0)  # (q,)
        # vol = mu + U^H @ alpha (in half-vol space, then IDFT)
        correction = U_np.conj().T @ mean_alpha.astype(np.complex128)  # (half_vol_size,)
        vol_half = np.asarray(init.mu) + correction
        vol_real = half_to_real_vol(vol_half, volume_shape)
        cluster_vols.append(vol_real)
    return cluster_vols, cluster_labels


# ---------------------------------------------------------------------------
# Step 3: Run BASELINE (no annealing, 15 joint iters)
# ---------------------------------------------------------------------------
print("\n[3] Running BASELINE condition (no annealing, n_joint=15)...", flush=True)
t0 = time.perf_counter()
n_burnin = resolve_n_burnin("discrete_volumes", None)  # 0

final_baseline, fre_truth_traj_bl, fre_fp_traj_bl, pe_traj_bl, fre_floor_bl, lm_traj_bl = run_two_stage(
    cfg,
    ds,
    q=Q,
    n_burnin=n_burnin,
    n_joint=N_JOINT_BASELINE,
    mu_init_kind="perturbed",
    mu_perturb_eps=0.5,
    seed=SEED,
    weighted_svd=False,
    anneal_schedule="none",
    anneal_iters=30,
)
print(f"  BASELINE done in {time.perf_counter() - t0:.1f}s", flush=True)

print("\n  Computing baseline embedding...", flush=True)
alpha_hat_bl, stats_bl = get_alpha_hat(cfg, final_baseline, ds)
metrics_bl = summarize_discrete_embedding(cfg, ds, final_baseline)
print(f"  baseline metrics: {metrics_bl}", flush=True)

# ---------------------------------------------------------------------------
# Step 4: Run ANNEALED condition (log1000, 30 joint iters)
# ---------------------------------------------------------------------------
print("\n[4] Running ANNEALED condition (log1000, n_joint=30)...", flush=True)
t0 = time.perf_counter()

final_anneal, fre_truth_traj_an, fre_fp_traj_an, pe_traj_an, fre_floor_an, lm_traj_an = run_two_stage(
    cfg,
    ds,
    q=Q,
    n_burnin=n_burnin,
    n_joint=N_JOINT_ANNEAL,
    mu_init_kind="perturbed",
    mu_perturb_eps=0.5,
    seed=SEED,
    weighted_svd=False,
    anneal_schedule="log1000",
    anneal_iters=30,
)
print(f"  ANNEALED done in {time.perf_counter() - t0:.1f}s", flush=True)

print("\n  Computing annealed embedding...", flush=True)
alpha_hat_an, stats_an = get_alpha_hat(cfg, final_anneal, ds)
metrics_an = summarize_discrete_embedding(cfg, ds, final_anneal)
print(f"  annealed metrics: {metrics_an}", flush=True)

# ---------------------------------------------------------------------------
# Step 5: Reconstruct cluster volumes for annealed result
# ---------------------------------------------------------------------------
print("\n[5] Reconstructing per-cluster volumes from annealed embedding...", flush=True)
state_labels = np.asarray(ds.state_label_true, dtype=np.int64)
cluster_vols_an, cluster_labels_an = reconstruct_cluster_volumes(
    final_anneal, alpha_hat_an, state_labels, N_STATES, volume_shape
)


# Build Hungarian matching: cluster_labels_an -> state_labels
def hungarian_match(labels_true, labels_pred, k):
    C = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(labels_true, labels_pred):
        C[int(t), int(p)] += 1
    row_ind, col_ind = linear_sum_assignment(-C)
    perm = np.zeros(k, dtype=np.int64)
    for r, c in zip(row_ind, col_ind):
        perm[c] = r  # cluster c maps to state perm[c]
    return perm, C


perm_an, contingency_an = hungarian_match(state_labels, cluster_labels_an, N_STATES)
matched_labels_an = np.array([perm_an[c] for c in cluster_labels_an])

# ---------------------------------------------------------------------------
# Step 6: Build the multi-panel figure
# ---------------------------------------------------------------------------
print("\n[6] Building diagnostic figure...", flush=True)

# Color map for 16 states
cmap_states = plt.get_cmap("tab20", N_STATES)
state_colors = [cmap_states(i) for i in range(N_STATES)]

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#1a1a2e")
gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35, left=0.07, right=0.97, top=0.93, bottom=0.05)

PANEL_LABEL_KW = dict(fontsize=13, fontweight="bold", color="white", transform=None, ha="left", va="top")
TITLE_KW = dict(fontsize=9, color="#cccccc", pad=4)
AXIS_KW = dict(facecolor="#0d0d1a")


def style_ax(ax, title=""):
    ax.set_facecolor("#0d0d1a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="#aaaacc", labelsize=7)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    if title:
        ax.set_title(title, **TITLE_KW)


def panel_label(ax, label):
    ax.text(
        -0.08,
        1.08,
        f"({label})",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color="white",
        va="top",
        ha="left",
    )


mid_z = VOL // 2  # central z-slice index

# ---------------------------------------------------------------------------
# Panel A: Grid of example synthetic images (4x4)
# ---------------------------------------------------------------------------
ax_A = fig.add_subplot(gs[0, 0])
style_ax(ax_A)
panel_label(ax_A, "A")
ax_A.set_title("Example synthetic images (16 states)", **TITLE_KW)

n_grid = 4
# Pick one image per state label
img_grid = np.zeros((n_grid * VOL, n_grid * VOL), dtype=np.float64)
for row in range(n_grid):
    for col in range(n_grid):
        state_idx = row * n_grid + col
        # Find first image with this state label
        img_idxs = np.where(state_labels == state_idx)[0]
        if len(img_idxs) == 0:
            continue
        img_idx = img_idxs[0]
        # Convert full Fourier image to real space via ftu.get_idft2
        img_full_ft = jnp.asarray(ds.batch_full[img_idx]).reshape(image_shape)
        img_real = np.asarray(ftu.get_idft2(img_full_ft)).real
        r0, r1 = row * VOL, (row + 1) * VOL
        c0, c1 = col * VOL, (col + 1) * VOL
        img_grid[r0:r1, c0:c1] = img_real

vmin_A, vmax_A = np.percentile(img_grid, [2, 98])
ax_A.imshow(img_grid, cmap="gray", vmin=vmin_A, vmax=vmax_A, interpolation="nearest", origin="upper")
ax_A.axis("off")
# Add colored borders to mark states
for row in range(n_grid):
    for col in range(n_grid):
        state_idx = row * n_grid + col
        color = state_colors[state_idx]
        rect = plt.Rectangle((col * VOL - 0.5, row * VOL - 0.5), VOL, VOL, fill=False, edgecolor=color, linewidth=1.5)
        ax_A.add_patch(rect)

# ---------------------------------------------------------------------------
# Panel B: Central slices of 4 GT volumes
# ---------------------------------------------------------------------------
ax_B = fig.add_subplot(gs[0, 1])
style_ax(ax_B)
panel_label(ax_B, "B")
ax_B.set_title("GT volume central slices (4 of 16 states)", **TITLE_KW)

show_states_B = [0, 4, 8, 12]
n_show = len(show_states_B)
vol_grid_B = np.zeros((VOL, VOL * n_show), dtype=np.float64)
for j, s_idx in enumerate(show_states_B):
    vol_real = gt_vols[s_idx]  # already at target_D if loaded
    if vol_real.shape[0] != VOL:
        vol_real = downsample_volume(vol_real, VOL)
    vol_grid_B[:, j * VOL : (j + 1) * VOL] = vol_real[mid_z]

vmin_B, vmax_B = np.percentile(vol_grid_B[vol_grid_B != 0], [1, 99]) if vol_grid_B.any() else (0, 1)
ax_B.imshow(vol_grid_B, cmap="gray_r", vmin=vmin_B, vmax=vmax_B, interpolation="nearest", origin="upper")
ax_B.axis("off")
for j, s_idx in enumerate(show_states_B):
    ax_B.text(
        j * VOL + VOL / 2,
        -2,
        f"State {s_idx}",
        ha="center",
        va="bottom",
        color=state_colors[s_idx],
        fontsize=7,
        fontweight="bold",
    )

# ---------------------------------------------------------------------------
# Panel C: GT mean volume central slice
# ---------------------------------------------------------------------------
ax_C = fig.add_subplot(gs[0, 2])
style_ax(ax_C)
panel_label(ax_C, "C")
ax_C.set_title("GT mean volume (central z-slice)", **TITLE_KW)

mu_true_real = half_to_real_vol(np.asarray(ds.mu_half_true), volume_shape)
im_C = ax_C.imshow(mu_true_real[mid_z], cmap="gray_r", interpolation="nearest", origin="upper")
ax_C.axis("off")
plt.colorbar(im_C, ax=ax_C, fraction=0.046, pad=0.04, label="Density").ax.tick_params(colors="#aaaacc", labelsize=6)

# ---------------------------------------------------------------------------
# Panel D: 2D PCA of baseline embedding, colored by true state
# ---------------------------------------------------------------------------
ax_D = fig.add_subplot(gs[1, 0])
style_ax(ax_D, "Baseline embedding (2D PCA)")
panel_label(ax_D, "D")

pca = PCA(n_components=2)
emb_bl_2d = pca.fit_transform(alpha_hat_bl)
for s_idx in range(N_STATES):
    mask = state_labels == s_idx
    ax_D.scatter(emb_bl_2d[mask, 0], emb_bl_2d[mask, 1], c=[state_colors[s_idx]], s=8, alpha=0.6, linewidths=0)
ax_D.set_xlabel("PC1", fontsize=8)
ax_D.set_ylabel("PC2", fontsize=8)
ax_D.text(
    0.02,
    0.97,
    f"No anneal | n_joint={N_JOINT_BASELINE}\nhun={metrics_bl['clust_acc_hungarian']:.3f} ARI={metrics_bl['ari']:.3f}",
    transform=ax_D.transAxes,
    fontsize=7,
    color="#cccccc",
    va="top",
    bbox=dict(facecolor="#111133", alpha=0.7, edgecolor="none"),
)

# ---------------------------------------------------------------------------
# Panel E: 2D PCA of annealed embedding, colored by true state
# ---------------------------------------------------------------------------
ax_E = fig.add_subplot(gs[1, 1])
style_ax(ax_E, "Annealed embedding (2D PCA)")
panel_label(ax_E, "E")

pca_an = PCA(n_components=2)
emb_an_2d = pca_an.fit_transform(alpha_hat_an)
for s_idx in range(N_STATES):
    mask = state_labels == s_idx
    ax_E.scatter(emb_an_2d[mask, 0], emb_an_2d[mask, 1], c=[state_colors[s_idx]], s=8, alpha=0.6, linewidths=0)
ax_E.set_xlabel("PC1", fontsize=8)
ax_E.set_ylabel("PC2", fontsize=8)
ax_E.text(
    0.02,
    0.97,
    f"log1000 anneal | n_joint={N_JOINT_ANNEAL}\nhun={metrics_an['clust_acc_hungarian']:.3f} ARI={metrics_an['ari']:.3f}",
    transform=ax_E.transAxes,
    fontsize=7,
    color="#cccccc",
    va="top",
    bbox=dict(facecolor="#111133", alpha=0.7, edgecolor="none"),
)

# Add colorbar legend for states
legend_handles = [Patch(facecolor=state_colors[s], label=f"{s}") for s in range(0, N_STATES, 4)]
ax_E.legend(
    handles=legend_handles,
    loc="lower right",
    ncol=2,
    fontsize=6,
    framealpha=0.5,
    facecolor="#111133",
    edgecolor="#444466",
    labelcolor="white",
    title="State",
    title_fontsize=6,
)

# ---------------------------------------------------------------------------
# Panel F: Confusion matrix for annealed run
# ---------------------------------------------------------------------------
ax_F = fig.add_subplot(gs[1, 2])
style_ax(ax_F, "Confusion matrix (annealed, Hungarian-matched)")
panel_label(ax_F, "F")

cm = confusion_matrix(state_labels, matched_labels_an, labels=np.arange(N_STATES))
im_F = ax_F.imshow(cm, cmap="Blues", interpolation="nearest", vmin=0)
ax_F.set_xlabel("Predicted state", fontsize=7)
ax_F.set_ylabel("True state", fontsize=7)
ax_F.set_xticks(np.arange(N_STATES))
ax_F.set_yticks(np.arange(N_STATES))
ax_F.set_xticklabels(np.arange(N_STATES), fontsize=5)
ax_F.set_yticklabels(np.arange(N_STATES), fontsize=5)
plt.colorbar(im_F, ax=ax_F, fraction=0.046, pad=0.04).ax.tick_params(colors="#aaaacc", labelsize=6)
diag_acc = np.diag(cm).sum() / cm.sum()
ax_F.set_title(f"Confusion matrix (acc={diag_acc:.3f})", **TITLE_KW)

# ---------------------------------------------------------------------------
# Panel G: Per-cluster learned volumes vs GT (4 states)
# ---------------------------------------------------------------------------
ax_G = fig.add_subplot(gs[2, 0])
style_ax(ax_G)
panel_label(ax_G, "G")
ax_G.set_title("Learned vs GT volumes (4 states, central z)", **TITLE_KW)

show_states_G = [0, 4, 8, 12]
# Top row: GT, Bottom row: learned
n_G = len(show_states_G)
panel_G = np.zeros((2 * VOL, n_G * VOL), dtype=np.float64)

# Build GT→cluster mapping from Hungarian matching: for each GT state, find best cluster
# perm_an[c] = gt_state means cluster c → gt_state
gt_to_cluster = {}
for cluster_idx, gt_idx in enumerate(perm_an):
    if int(gt_idx) not in gt_to_cluster:
        gt_to_cluster[int(gt_idx)] = cluster_idx

for j, s_idx in enumerate(show_states_G):
    gt_vol = gt_vols[s_idx]
    if gt_vol.shape[0] != VOL:
        gt_vol = downsample_volume(gt_vol, VOL)
    panel_G[:VOL, j * VOL : (j + 1) * VOL] = gt_vol[mid_z]

    # Get matched cluster volume
    cluster_idx = gt_to_cluster.get(s_idx, 0)
    learned_vol = cluster_vols_an[cluster_idx]
    panel_G[VOL:, j * VOL : (j + 1) * VOL] = learned_vol[mid_z]

vmin_G, vmax_G = np.percentile(panel_G, [2, 98])
ax_G.imshow(panel_G, cmap="gray_r", vmin=vmin_G, vmax=vmax_G, interpolation="nearest", origin="upper")
ax_G.axis("off")
ax_G.text(
    -0.02, 0.75, "GT", transform=ax_G.transAxes, fontsize=8, color="#cccccc", rotation=90, va="center", ha="right"
)
ax_G.text(
    -0.02, 0.25, "Learned", transform=ax_G.transAxes, fontsize=8, color="#cccccc", rotation=90, va="center", ha="right"
)
for j, s_idx in enumerate(show_states_G):
    ax_G.text(
        j * VOL + VOL / 2,
        2 * VOL + 1,
        f"State {s_idx}",
        ha="center",
        va="top",
        color=state_colors[s_idx],
        fontsize=7,
        transform=ax_G.transData,
    )

# ---------------------------------------------------------------------------
# Panel H: Projector error trajectory
# ---------------------------------------------------------------------------
ax_H = fig.add_subplot(gs[2, 1])
style_ax(ax_H, "Projector error (pe) trajectory")
panel_label(ax_H, "H")

iters_bl = np.arange(len(pe_traj_bl))
iters_an = np.arange(len(pe_traj_an))
ax_H.plot(
    iters_bl, pe_traj_bl, "o-", color="#ff6b6b", markersize=4, linewidth=1.5, label=f"No anneal (n={N_JOINT_BASELINE})"
)
ax_H.plot(
    iters_an,
    pe_traj_an,
    "s-",
    color="#4ecdc4",
    markersize=4,
    linewidth=1.5,
    label=f"log1000 anneal (n={N_JOINT_ANNEAL})",
)
ax_H.axvline(0.5, color="#888888", linestyle=":", linewidth=1, alpha=0.5)
ax_H.text(0.6, 0.98, "← SVD warm", transform=ax_H.transAxes, fontsize=6, color="#888888", va="top")
ax_H.set_xlabel("Iteration", fontsize=8)
ax_H.set_ylabel("Projector Frobenius error", fontsize=8)
ax_H.legend(fontsize=7, loc="upper right", facecolor="#111133", edgecolor="#444466", labelcolor="white")
ax_H.grid(True, alpha=0.2, color="#444466")

# ---------------------------------------------------------------------------
# Panel I: Annealing factor schedule + log marginal
# ---------------------------------------------------------------------------
ax_I = fig.add_subplot(gs[2, 2])
style_ax(ax_I, "Anneal schedule & log-marginal")
panel_label(ax_I, "I")

# Annealing schedule
schedule_an = build_anneal_schedule("log1000", 30, N_JOINT_ANNEAL)
ax_I2 = ax_I.twinx()
ax_I2.set_facecolor("#0d0d1a")

iters_sched = np.arange(1, N_JOINT_ANNEAL + 1)
ax_I.plot(iters_sched, schedule_an, "y--", linewidth=1.5, alpha=0.7, label="Anneal factor (log1000)")
ax_I.set_xlabel("Joint iteration", fontsize=8)
ax_I.set_ylabel("Noise scale factor", fontsize=8, color="#cccc44")
ax_I.tick_params(axis="y", colors="#cccc44")
ax_I.set_yscale("log")

# Log-marginal difference trajectory
if len(lm_traj_an) > 1:
    lm_diff_an = np.array([lm_traj_an[i] - lm_traj_an[0] for i in range(len(lm_traj_an))])
    lm_diff_bl = np.array([lm_traj_bl[i] - lm_traj_bl[0] for i in range(len(lm_traj_bl))])
    iters_lm_an = np.arange(len(lm_diff_an))
    iters_lm_bl = np.arange(len(lm_diff_bl))
    ax_I2.plot(iters_lm_an, lm_diff_an, color="#4ecdc4", linewidth=1.5, label="Annealed Δlog-marginal")
    ax_I2.plot(iters_lm_bl, lm_diff_bl, color="#ff6b6b", linewidth=1.5, linestyle="--", label="Baseline Δlog-marginal")
    ax_I2.set_ylabel("Δlog-marginal (from iter 0)", fontsize=7, color="#aaaacc")
    ax_I2.tick_params(axis="y", colors="#aaaacc", labelsize=6)

lines1, labels1 = ax_I.get_legend_handles_labels()
lines2, labels2 = ax_I2.get_legend_handles_labels()
ax_I.legend(
    lines1 + lines2,
    labels1 + labels2,
    fontsize=6,
    loc="lower right",
    facecolor="#111133",
    edgecolor="#444466",
    labelcolor="white",
)
ax_I.grid(True, alpha=0.2, color="#444466")

# ---------------------------------------------------------------------------
# Panel J: Bar chart: summary metrics comparison
# ---------------------------------------------------------------------------
ax_J = fig.add_subplot(gs[3, 0])
style_ax(ax_J, "Summary metrics: baseline vs annealed")
panel_label(ax_J, "J")

metric_names = ["centroid_acc", "hun", "ARI", "NMI"]
metric_keys = ["centroid_acc", "clust_acc_hungarian", "ari", "nmi"]
vals_bl = [metrics_bl.get(k, 0.0) for k in metric_keys]
vals_an = [metrics_an.get(k, 0.0) for k in metric_keys]

x = np.arange(len(metric_names))
w = 0.35
bars1 = ax_J.bar(
    x - w / 2, vals_bl, w, color="#ff6b6b", alpha=0.85, label=f"Baseline (n={N_JOINT_BASELINE})", edgecolor="#cc4444"
)
bars2 = ax_J.bar(
    x + w / 2, vals_an, w, color="#4ecdc4", alpha=0.85, label=f"Annealed (n={N_JOINT_ANNEAL})", edgecolor="#229988"
)
ax_J.set_xticks(x)
ax_J.set_xticklabels(metric_names, fontsize=8, color="#aaaacc")
ax_J.set_ylabel("Score", fontsize=8)
ax_J.set_ylim(0, 1.1)
ax_J.legend(fontsize=7, facecolor="#111133", edgecolor="#444466", labelcolor="white")
ax_J.grid(True, alpha=0.2, color="#444466", axis="y")
for bar in bars1:
    ax_J.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=6,
        color="#ff6b6b",
    )
for bar in bars2:
    ax_J.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=6,
        color="#4ecdc4",
    )

# ---------------------------------------------------------------------------
# Panel K: Latent variance comparison (aligned_std vs true_std)
# ---------------------------------------------------------------------------
ax_K = fig.add_subplot(gs[3, 1])
style_ax(ax_K, "Latent std: aligned vs true (per PC)")
panel_label(ax_K, "K")

# aligned_std from the annealed metrics
if metrics_an and "aligned_std" in metrics_an and "true_std" in metrics_an:
    aligned_std = np.array(metrics_an["aligned_std"])
    true_std = np.array(metrics_an["true_std"])
    pcs = np.arange(len(aligned_std))
    ax_K.bar(pcs - 0.2, true_std, 0.4, color="#ffd700", alpha=0.85, label="GT std", edgecolor="#bbaa00")
    ax_K.bar(pcs + 0.2, aligned_std, 0.4, color="#4ecdc4", alpha=0.85, label="Aligned est std", edgecolor="#229988")
    ax_K.set_xlabel("PC index", fontsize=8)
    ax_K.set_ylabel("Std dev", fontsize=8)
    ax_K.set_xticks(pcs)
    ax_K.legend(fontsize=7, facecolor="#111133", edgecolor="#444466", labelcolor="white")
    ax_K.grid(True, alpha=0.2, color="#444466", axis="y")
else:
    ax_K.text(
        0.5,
        0.5,
        "No aligned_std available",
        transform=ax_K.transAxes,
        ha="center",
        va="center",
        color="#aaaacc",
        fontsize=9,
    )

# Also add baseline aligned_std if available
if metrics_bl and "aligned_std" in metrics_bl:
    aligned_std_bl = np.array(metrics_bl["aligned_std"])
    pcs = np.arange(len(aligned_std_bl))
    ax_K.plot(
        pcs, aligned_std_bl, "o--", color="#ff6b6b", markersize=4, linewidth=1, label="Baseline est std", alpha=0.8
    )
    ax_K.legend(fontsize=7, facecolor="#111133", edgecolor="#444466", labelcolor="white")

# ---------------------------------------------------------------------------
# Panel L: Text summary panel
# ---------------------------------------------------------------------------
ax_L = fig.add_subplot(gs[3, 2])
ax_L.set_facecolor("#0d0d1a")
for spine in ax_L.spines.values():
    spine.set_edgecolor("#444466")
ax_L.axis("off")
panel_label(ax_L, "L")


def fmt_metric(d, k):
    v = d.get(k, float("nan")) if d else float("nan")
    return f"{v:.4f}" if not np.isnan(v) else "N/A"


pe_final_bl = pe_traj_bl[-1] if pe_traj_bl else float("nan")
pe_final_an = pe_traj_an[-1] if pe_traj_an else float("nan")

text_lines = [
    "Key Numbers",
    "─" * 30,
    f"Dataset:  Ribosembly (q={Q}, D={VOL})",
    f"N images: {N_IMAGES}, σ={SIGMA}",
    f"Grid: HEALPix order {HEALPIX_ORDER}",
    f"n_rot={ds.n_rot}, n_trans={ds.n_trans}",
    "",
    "Baseline (no anneal):",
    f"  n_joint={N_JOINT_BASELINE}",
    f"  centroid_acc: {fmt_metric(metrics_bl, 'centroid_acc')}",
    f"  Hungarian:    {fmt_metric(metrics_bl, 'clust_acc_hungarian')}",
    f"  ARI:          {fmt_metric(metrics_bl, 'ari')}",
    f"  NMI:          {fmt_metric(metrics_bl, 'nmi')}",
    f"  pe_final:     {pe_final_bl:.4f}",
    "",
    "Annealed (log1000):",
    f"  n_joint={N_JOINT_ANNEAL}, anneal_iters=30",
    f"  centroid_acc: {fmt_metric(metrics_an, 'centroid_acc')}",
    f"  Hungarian:    {fmt_metric(metrics_an, 'clust_acc_hungarian')}",
    f"  ARI:          {fmt_metric(metrics_an, 'ari')}",
    f"  NMI:          {fmt_metric(metrics_an, 'nmi')}",
    f"  pe_final:     {pe_final_an:.4f}",
]

ax_L.text(
    0.05,
    0.98,
    "\n".join(text_lines),
    transform=ax_L.transAxes,
    fontsize=7.5,
    color="#ddddee",
    va="top",
    ha="left",
    fontfamily="monospace",
    bbox=dict(facecolor="#0d0d1a", alpha=0.0, edgecolor="none"),
)

# ---------------------------------------------------------------------------
# Main title & save
# ---------------------------------------------------------------------------
fig.suptitle(
    "PPCA Ab-Initio on CryoBench Ribosembly (q=8, D=32, N=1024, σ=0.01)",
    fontsize=15,
    fontweight="bold",
    color="white",
    y=0.97,
)

print(f"\n[7] Saving figure to {OUTPUT_PATH} ...", flush=True)
plt.savefig(str(OUTPUT_PATH), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print("  Saved. Done!", flush=True)

# ---------------------------------------------------------------------------
# Print final summary to stdout
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("FINAL SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"Baseline (no anneal, n_joint={N_JOINT_BASELINE}):", flush=True)
for k, v in (metrics_bl or {}).items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}", flush=True)
print(f"  pe_final: {pe_final_bl:.4f}", flush=True)
print(f"\nAnnealed (log1000, n_joint={N_JOINT_ANNEAL}):", flush=True)
for k, v in (metrics_an or {}).items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}", flush=True)
print(f"  pe_final: {pe_final_an:.4f}", flush=True)
print(f"\nFigure saved to: {OUTPUT_PATH}", flush=True)
