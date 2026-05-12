"""Per-cell eval: compute basis-invariant metrics + embedding scatter at GT poses.

Writes:
  eval/metrics.json       — μCC, W subspace ⟨cos⟩, discrim ratios, prior shell ratios
  eval/embedding.npz      — est_z + gt_assignments + mu_out_real + W_out_real
  eval/embedding_grid.png — 2×2 grid of E[z][a]×E[z][b] colored by GT class
  eval/recovered.png      — μ in/out/gt + W_0..W_q-1 in/out/gt + embedding + bars
  eval/final_projection.png — μ + W_0..W_{q-1} init vs final projections
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--gt-vol-glob", required=True)
    p.add_argument("--q", type=int, default=4)
    p.add_argument("--n-score-images", type=int, default=2000)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument(
        "--workdir", type=str, default="/scratch/gpfs/GILLES/mg6942/recovar_wt_ppca_postmerge_followup_20260510_110827"
    )
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, args.workdir)
    import glob as _glob

    import jax.numpy as jnp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from recovar.em.ppca_refinement.initialization import initialize_ppca_from_gt_volumes

    from recovar import core
    from recovar.core import fourier_transform_utils as ftu
    from recovar.core.ctf import CTFEvaluator
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.reconstruction import noise as recon_noise
    from recovar.utils import helpers as _helpers

    sys.path.insert(0, str(Path(args.workdir) / "scripts"))
    from compute_ppca_best_pose_embedding import _best_pose_z_mean

    state = np.load(args.run_dir / "last_state.npz")
    traj = json.loads((args.run_dir / "trajectory.json").read_text())
    BOX = int(traj["dataset_metadata"]["box"])
    VOXEL = float(traj["dataset_metadata"]["voxel"])
    vshape = (BOX, BOX, BOX)
    half_shape = ftu.volume_shape_to_half_volume_shape(vshape)
    mu_init = np.asarray(state["init_mu"], dtype=np.float32)
    W_init = np.asarray(state["init_W"], dtype=np.float32)
    mu_half_out = np.asarray(state["mu_half"], dtype=np.complex64).reshape(half_shape)
    W_half_flat = np.asarray(state["W_half"], dtype=np.complex64)
    mu_out_real = np.asarray(ftu.get_idft3_real(jnp.asarray(mu_half_out), volume_shape=vshape))
    W_out_real = np.stack(
        [
            np.asarray(ftu.get_idft3_real(jnp.asarray(W_half_flat[:, k]).reshape(half_shape), volume_shape=vshape))
            for k in range(args.q)
        ],
        axis=0,
    )

    sim_info = pickle.load(open(args.dataset_dir / "simulation_info.pkl", "rb"))
    gt_paths = sorted(_glob.glob(args.gt_vol_glob))
    gt_vols = np.stack([_helpers.load_mrc(p) for p in gt_paths]).astype(np.float32)
    gt_assign = np.asarray(sim_info["image_assignment"], dtype=np.int64)
    gt_counts = np.bincount(gt_assign, minlength=gt_vols.shape[0]).astype(np.float64)
    gt_w = gt_counts / gt_counts.sum()
    gt_init = initialize_ppca_from_gt_volumes(gt_vols, q=args.q, weights=gt_w, frame="recovar", amplitude_scale=None)
    mu_gt = np.asarray(gt_init.mu, dtype=np.float32)
    W_gt = np.asarray(gt_init.W, dtype=np.float32)

    def _cc(a, b):
        a = a.reshape(-1).astype(np.float64)
        b = b.reshape(-1).astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a @ b) / d) if d > 0 else 0.0

    def _princ_cos(Wa, Wb):
        A = Wa.reshape(Wa.shape[0], -1).astype(np.float64).T
        B = Wb.reshape(Wb.shape[0], -1).astype(np.float64).T
        Qa, _ = np.linalg.qr(A)
        Qb, _ = np.linalg.qr(B)
        return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1).tolist()

    mu_cc = _cc(mu_out_real, mu_gt)
    W_cos = _princ_cos(W_out_real, W_gt)

    # Embedding at GT poses
    dataset = load_dataset(str(args.dataset_dir / "particles.star"))
    ns = min(args.n_score_images, dataset.n_images)
    batch = next(dataset.iter_batches(ns, indices=np.arange(ns), by_image=True))
    imgs_real, _, _, ctf_params_b, _, _, _ = batch
    images_h = (
        np.asarray(dataset.image_source.process_images_half(np.asarray(imgs_real, dtype=np.float32)))
        .reshape(ns, -1)
        .astype(np.complex64)
    )
    ctf_h = np.asarray(CTFEvaluator()(jnp.asarray(ctf_params_b), (BOX, BOX), VOXEL, half_image=True), dtype=np.float32)
    sigma2_h = np.asarray(
        recon_noise.make_radial_noise_half(jnp.asarray(sim_info["noise_variance"]), (BOX, BOX)), dtype=np.float32
    )
    Y1 = (images_h * ctf_h / sigma2_h[None, :]).astype(np.complex64)
    ctf2_n = (ctf_h**2) / sigma2_h[None, :]
    R_gt = np.asarray(sim_info["rots"], dtype=np.float32)[:ns]
    mu_h_flat = mu_half_out.reshape(-1)
    W_h_flat = (
        W_half_flat.reshape(args.q, -1) if W_half_flat.ndim == 2 and W_half_flat.shape[0] == args.q else W_half_flat.T
    )
    aug_h = np.concatenate([mu_h_flat[None, :], W_h_flat], axis=0)
    proj_aug = np.asarray(
        core.batch_slice_volume(
            jnp.asarray(aug_h),
            jnp.asarray(R_gt, dtype=np.float32),
            (BOX, BOX),
            (BOX, BOX, BOX),
            "linear_interp",
            half_volume=True,
            half_image=True,
        )
    )
    proj_aug = np.swapaxes(proj_aug, 0, 1)
    est_z = np.asarray(_best_pose_z_mean(jnp.asarray(Y1), jnp.asarray(ctf2_n), jnp.asarray(proj_aug)))

    # Discrim ratio per dim
    K_gt = gt_vols.shape[0]
    class_means = np.stack(
        [
            np.nanmean(est_z[gt_assign[:ns] == k], axis=0)
            if (gt_assign[:ns] == k).sum() > 0
            else np.full(args.q, np.nan)
            for k in range(K_gt)
        ],
        axis=0,
    )
    total_var = est_z.var(axis=0)
    between = np.nanvar(class_means, axis=0)
    discrim = (between / (total_var + 1e-30)).tolist()

    # Prior shell power vs recovered (last iter)
    W_prior_per_shell = np.asarray(state["W_prior_per_shell"], dtype=np.float64)
    iter_W_shell_power = np.asarray(state["iter_W_shell_power"], dtype=np.float64)
    final_W_shell = iter_W_shell_power[-1]
    prior_shell_ratio = (final_W_shell / (W_prior_per_shell + 1e-30)).tolist()
    print(
        f"prior shell ratios: min={min(prior_shell_ratio):.3f} max={max(prior_shell_ratio):.3f}  "
        f"(ideal=1.0 if prior well-scaled)"
    )

    eval_dir = args.run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "cell": args.run_dir.name,
        "noise_level": traj["dataset_metadata"]["noise_level"],
        "K_gt": int(K_gt),
        "box": BOX,
        "mu_correlation_recovered_vs_gt": mu_cc,
        "W_subspace_cosines_recovered_vs_gt": W_cos,
        "W_subspace_mean_cos_recovered": float(np.mean(W_cos)),
        "discriminative_ratio_per_latent": discrim,
        "max_discriminative_ratio": float(max(discrim)),
        "mu_rms_recovered": float(np.sqrt(np.mean(mu_out_real**2))),
        "mu_rms_gt": float(np.sqrt(np.mean(mu_gt**2))),
        "W_rms_recovered": float(np.sqrt(np.mean(W_out_real**2))),
        "W_rms_gt": float(np.sqrt(np.mean(W_gt**2))),
        "n_score_images": ns,
        "W_prior_per_shell": W_prior_per_shell.tolist(),
        "final_W_shell_power": final_W_shell.tolist(),
        "prior_shell_ratio": prior_shell_ratio,
    }
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        eval_dir / "embedding.npz",
        est_z=est_z,
        gt_assignments=gt_assign[:ns],
        mu_out_real=mu_out_real,
        W_out_real=W_out_real,
    )

    # Plot recovered.png (3×3 panel)
    sz = BOX // 2
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, vol, title in [
        (axes[0, 0], mu_init.sum(0), "μ init (proj)"),
        (axes[0, 1], mu_out_real.sum(0), f"μ out (CC={mu_cc:+.3f})"),
        (axes[0, 2], mu_gt.sum(0), "μ GT"),
    ]:
        ax.imshow(vol, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for k in range(min(args.q, 3)):
        ax = axes[1, k]
        v = W_out_real[k].sum(0)
        vmax = float(np.abs(v).max()) + 1e-30
        ax.imshow(v, cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.set_title(f"W_{k} out  cos[{k}]={W_cos[k]:.2f}", fontsize=10)
        ax.axis("off")
    # Row 3: embedding scatter + discrim bar + prior ratio
    cmap = plt.colormaps.get_cmap("turbo") if K_gt > 10 else plt.colormaps.get_cmap("tab10")
    for k in range(K_gt):
        m = gt_assign[:ns] == k
        if m.sum() > 0:
            axes[2, 0].scatter(est_z[m, 0], est_z[m, 1], s=6, alpha=0.4, color=cmap(k % cmap.N))
    axes[2, 0].set_title(f"E[z][:,0:1]  discrim_max={max(discrim):.3f}", fontsize=10)
    axes[2, 0].set_xlabel("z_0")
    axes[2, 0].set_ylabel("z_1")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axhline(0, color="k", lw=0.5, alpha=0.5)
    axes[2, 0].axvline(0, color="k", lw=0.5, alpha=0.5)
    axes[2, 1].bar(range(args.q), discrim)
    axes[2, 1].set_xticks(range(args.q))
    axes[2, 1].set_xticklabels([f"z_{i}" for i in range(args.q)])
    axes[2, 1].set_title("Discrim ratio", fontsize=10)
    axes[2, 1].set_ylim(0, 1.05)
    axes[2, 2].plot(W_prior_per_shell, "k-", lw=2, label="prior")
    axes[2, 2].plot(final_W_shell, "b-", label="recovered (final)")
    axes[2, 2].set_yscale("log")
    axes[2, 2].set_title("W shell power", fontsize=10)
    axes[2, 2].legend(fontsize=8)
    fig.suptitle(f"{args.run_dir.name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(eval_dir / "recovered.png", dpi=110)
    plt.close(fig)

    # embedding_grid.png (4-panel z-pairs colored by GT class)
    pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    colors = gt_assign[:ns].astype(np.float64) / max(K_gt - 1, 1)
    for ax_idx, (a, b) in enumerate(pairs):
        if a >= args.q or b >= args.q:
            continue
        ax = axes[ax_idx // 2, ax_idx % 2]
        sc = ax.scatter(est_z[:, a], est_z[:, b], c=colors, cmap=cmap, s=6, alpha=0.5)
        ax.set_title(f"E[z][:,{a}] vs E[z][:,{b}]")
        ax.set_xlabel(f"z_{a}")
        ax.set_ylabel(f"z_{b}")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.axvline(0, color="k", lw=0.5, alpha=0.5)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7, label=f"GT class (0..{K_gt - 1})")
    fig.suptitle(f"{args.run_dir.name} | embedding @ GT poses", fontsize=12)
    fig.savefig(eval_dir / "embedding_grid.png", dpi=110)
    plt.close(fig)

    # final_projection.png (2 rows × q+1 cols)
    fig, axes = plt.subplots(2, args.q + 1, figsize=(2.5 * (args.q + 1), 5))
    axes[0, 0].imshow(mu_init.sum(0), cmap="gray")
    axes[0, 0].axis("off")
    axes[0, 0].set_title(f"μ init  RMS={np.sqrt(np.mean(mu_init**2)):.2e}", fontsize=9)
    axes[1, 0].imshow(mu_out_real.sum(0), cmap="gray")
    axes[1, 0].axis("off")
    axes[1, 0].set_title(f"μ final  RMS={np.sqrt(np.mean(mu_out_real**2)):.2e}", fontsize=9)
    for k in range(args.q):
        for ri, (vol, lbl) in enumerate([(W_init[k].sum(0), f"W_{k} init"), (W_out_real[k].sum(0), f"W_{k} final")]):
            ax = axes[ri, k + 1]
            vmax = float(np.abs(vol).max()) + 1e-30
            ax.imshow(vol, cmap="seismic", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{lbl}  RMS={np.sqrt(np.mean(vol**2)):.2e}", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"{args.run_dir.name} | projections")
    fig.tight_layout()
    fig.savefig(eval_dir / "final_projection.png", dpi=110)
    plt.close(fig)

    print(f"[{args.run_dir.name}] μCC={mu_cc:+.3f} W⟨cos⟩={np.mean(W_cos):.3f} discrim={max(discrim):.3f}")


if __name__ == "__main__":
    main()
