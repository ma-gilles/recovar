"""PPCA refinement trajectory runner with prior-vs-recovered W shell diagnostic.

Per iter saves:
  - trajectory.json entry: pose-err, pmax, E[z]/GT corr, μ/W RMS, plus prior-vs-W shell vectors
  - iter_slices.npz: per-iter μ + W projections (sum-along-axis-0), iter 0 = init
  - iter_mu_W_slices.png: per-iter grid
  - last_state.npz: final mu_half + W_half + init mu + init W

Init strategies:
  gt_svd                — μ=GT-mean, W=GT-SVD basis (oracle)
  gt_mu_random_w        — μ=GT, W=random Gaussian matched RMS
  init_npz              — μ + W from anchored RELION K=4 NPZ
  kclass_mean_only      — μ = npz['mu'], W = random; supports --freeze-mean-iters N
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--gt-vol-glob", required=True)
    p.add_argument("--init", required=True, choices=("gt_svd", "gt_mu_random_w", "init_npz", "kclass_mean_only"))
    p.add_argument("--init-npz", type=Path, default=None)
    p.add_argument("--q", type=int, default=4)
    p.add_argument("--n-iters", type=int, default=5)
    p.add_argument("--healpix-order", type=int, default=3)
    p.add_argument("--offset-range", type=float, default=6.0)
    p.add_argument("--offset-step", type=float, default=2.0)
    p.add_argument("--image-batch-size", type=int, default=32)
    p.add_argument("--rotation-block-size", type=int, default=256)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument(
        "--workdir", type=str, default="/scratch/gpfs/GILLES/mg6942/recovar_wt_ppca_postmerge_followup_20260510_110827"
    )
    p.add_argument("--label", default="")
    p.add_argument("--n-score-images", type=int, default=500)
    p.add_argument("--prior-source", choices=("init", "gt"), default="init")
    p.add_argument("--freeze-mean-iters", type=int, default=0)
    p.add_argument(
        "--gt-mu-lowpass-R",
        type=int,
        default=None,
        help="If set, apply Fourier low-pass to GT-derived μ_init by zeroing freqs beyond "
        "radius R voxels (half-vol). Only for gt_svd/gt_mu_random_w inits.",
    )
    p.add_argument(
        "--w-bandlimit-R",
        type=int,
        default=None,
        help="If set, after each M-step low-pass W to freqs ≤ R voxels (half-vol). "
        "Pass a value smaller than --gt-mu-lowpass-R to force W coarser than μ.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    sys.path.insert(0, args.workdir)

    import glob as _glob

    import jax
    import jax.numpy as jnp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from recovar.em.ppca_refinement.config import (
        GeometryConfig,
        PostprocessConfig,
        ScheduleConfig,
        ScoringConfig,
    )
    from recovar.em.ppca_refinement.dense_dataset import run_dense_ppca_fused_em_iteration
    from recovar.em.ppca_refinement.initialization import (
        initialize_ppca_from_gt_volumes,
        loading_row_norm_variance_prior,
        volume_power_variance_prior,
    )

    from recovar import core
    from recovar.core import fourier_transform_utils as ftu
    from recovar.core.ctf import CTFEvaluator
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.em.sampling import get_rotation_grid_at_order, get_translation_grid
    from recovar.reconstruction import noise as recon_noise
    from recovar.utils import helpers as _helpers

    sys.path.insert(0, str(Path(args.workdir) / "scripts"))
    from compute_ppca_best_pose_embedding import _best_pose_z_mean

    # Helper: per-shell average of half-volume values (returns 1D vector, length = nshells)
    def _half_radial_labels(volume_shape):
        N = int(volume_shape[0])
        half = N // 2 + 1
        gz, gy, gx = np.meshgrid(np.fft.fftfreq(N) * N, np.fft.fftfreq(N) * N, np.arange(half), indexing="ij")
        return np.sqrt(gz**2 + gy**2 + gx**2).astype(np.int64).reshape(-1)

    def _shell_mean_1d(values_per_voxel, volume_shape):
        labels = _half_radial_labels(volume_shape)
        nshells = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=nshells).astype(np.float64)
        sums = np.bincount(labels, weights=values_per_voxel.astype(np.float64).reshape(-1), minlength=nshells)
        return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    print(f"jax: {jax.devices()}  args: {vars(args)}")

    sim_info = pickle.load(open(args.dataset_dir / "simulation_info.pkl", "rb"))
    dataset = load_dataset(str(args.dataset_dir / "particles.star"))
    BOX = int(sim_info.get("grid_size", 128))
    VOXEL = float(sim_info.get("voxel_size", sim_info.get("pixel_size", 4.25)))
    NOISE = float(sim_info.get("noise_level", 1.0))
    vshape = (BOX, BOX, BOX)
    half_shape = ftu.volume_shape_to_half_volume_shape(vshape)

    # GT
    gt_paths = sorted(_glob.glob(args.gt_vol_glob))
    gt_vols_real = np.stack([_helpers.load_mrc(p) for p in gt_paths], axis=0).astype(np.float32)
    gt_assignments = np.asarray(sim_info["image_assignment"], dtype=np.int64)
    gt_counts = np.bincount(gt_assignments, minlength=gt_vols_real.shape[0]).astype(np.float64)
    gt_weights = gt_counts / gt_counts.sum()
    gt_init = initialize_ppca_from_gt_volumes(
        gt_vols_real, q=args.q, weights=gt_weights, frame="recovar", amplitude_scale=None
    )
    mu_gt = np.asarray(gt_init.mu, dtype=np.float32)
    W_gt = np.asarray(gt_init.W, dtype=np.float32)
    print(f"GT-SVD σ: {gt_init.diagnostics['singular_values']}")

    # Build Fourier low-pass mask helper (sphere of radius R in half-vol voxel coords).
    def _lowpass_mask_half(R_vox):
        gz, gy, gx = np.meshgrid(
            np.fft.fftfreq(BOX) * BOX,
            np.fft.fftfreq(BOX) * BOX,
            np.arange(half_shape[-1]),
            indexing="ij",
        )
        return ((np.sqrt(gz**2 + gy**2 + gx**2) <= R_vox).astype(np.float32)).reshape(-1)

    def _apply_lowpass_real_vol(vol_real, R_vox):
        f = np.asarray(ftu.get_dft3_real(jnp.asarray(vol_real))).reshape(-1) * _lowpass_mask_half(R_vox)
        return np.asarray(ftu.get_idft3_real(jnp.asarray(f).reshape(half_shape), volume_shape=vshape)).astype(
            np.float32
        )

    # Optional GT-μ low-pass for gt_svd / gt_mu_random_w (simulates a coarse K-class init).
    if args.gt_mu_lowpass_R is not None and args.init in ("gt_svd", "gt_mu_random_w"):
        R = int(args.gt_mu_lowpass_R)
        mu_gt = _apply_lowpass_real_vol(mu_gt, R)
        print(
            f"applied GT-μ low-pass at R={R} voxels (~{BOX * VOXEL / (2 * R):.1f}Å)  new RMS={np.sqrt(np.mean(mu_gt**2)):.3e}"
        )

    rng = np.random.default_rng(args.seed)
    if args.init == "gt_svd":
        mu_init, W_init = mu_gt.copy(), W_gt.copy()
    elif args.init == "gt_mu_random_w":
        mu_init = mu_gt.copy()
        W_init = rng.standard_normal(W_gt.shape).astype(np.float32) * float(np.sqrt(np.mean(W_gt**2)))
    elif args.init == "init_npz":
        z = np.load(args.init_npz)
        mu_init = np.asarray(z["mu"], dtype=np.float32)
        W_init = np.asarray(z["W"], dtype=np.float32)
    elif args.init == "kclass_mean_only":
        z = np.load(args.init_npz)
        mu_init = np.asarray(z["mu"], dtype=np.float32)
        W_init = rng.standard_normal(W_gt.shape).astype(np.float32) * float(np.sqrt(np.mean(W_gt**2)))
    else:
        raise SystemExit(args.init)
    print(f"init μ RMS={np.sqrt(np.mean(mu_init**2)):.3e}  W RMS={np.sqrt(np.mean(W_init**2)):.3e}")

    # Pose grid (static — single HP order, no freq-march for simplicity)
    rotations = np.asarray(get_rotation_grid_at_order(args.healpix_order), dtype=np.float32)
    translations = np.asarray(get_translation_grid(args.offset_range, args.offset_step), dtype=np.float32)
    noise_var_full = np.asarray(
        recon_noise.make_radial_noise(jnp.asarray(sim_info["noise_variance"]), (BOX, BOX)), dtype=np.float32
    ).reshape(-1)
    noise_var_half = np.asarray(
        recon_noise.make_radial_noise_half(jnp.asarray(sim_info["noise_variance"]), (BOX, BOX)), dtype=np.float32
    )

    # Priors (compute once based on prior_source)
    if args.prior_source == "gt":
        mean_prior = volume_power_variance_prior(mu_gt, volume_shape=vshape)
        W_prior = loading_row_norm_variance_prior(W_gt, volume_shape=vshape)
    else:
        mean_prior = volume_power_variance_prior(mu_init, volume_shape=vshape)
        W_prior = loading_row_norm_variance_prior(W_init, volume_shape=vshape)
    # Per-shell W prior (1D) — same value across all q columns of W_prior, so take column 0
    W_prior_per_shell = _shell_mean_1d(W_prior[:, 0], vshape)
    print(
        f"W_prior shells: min={W_prior_per_shell.min():.2e}  max={W_prior_per_shell.max():.2e}  nshells={W_prior_per_shell.size}"
    )

    # GT z per class (latent space reference for E[z] diagnostic)
    W_gt_flat = W_gt.reshape(args.q, -1)
    gt_z_per_class = np.linalg.solve(
        W_gt_flat @ W_gt_flat.T + 1e-8 * np.eye(args.q),
        W_gt_flat
        @ (gt_init.aligned_volumes.reshape(gt_init.aligned_volumes.shape[0], -1) - mu_gt.reshape(-1)[None, :]).T,
    ).T

    cur_mu, cur_W, cur_domain = mu_init, W_init, "real"
    metrics = []
    # Per-iter slice/projection capture (projections!)
    iter_slices_mu = [mu_init.sum(axis=0).astype(np.float32).copy()]
    iter_slices_W = [W_init.sum(axis=1).astype(np.float32).copy()]
    iter_rms_mu = [float(np.sqrt(np.mean(mu_init**2)))]
    iter_rms_W = [float(np.sqrt(np.mean(W_init**2)))]
    # Per-iter W shell power: starting with init
    W_init_half_flat = np.asarray(
        np.stack([np.asarray(ftu.get_dft3_real(jnp.asarray(W_init[k]))).reshape(-1) for k in range(args.q)], axis=1)
    )  # (n_voxels_half, q)
    iter_W_shell_power = [_shell_mean_1d(np.sum(np.abs(W_init_half_flat) ** 2, axis=1), vshape)]

    gt_rots_all = np.asarray(sim_info["rots"], dtype=np.float64)

    for it in range(1, args.n_iters + 1):
        t0 = time.time()
        freeze_this = bool(it <= args.freeze_mean_iters)
        print(f"  iter {it}: HP={args.healpix_order}  freeze_mean={freeze_this}")
        result = run_dense_ppca_fused_em_iteration(
            dataset,
            cur_mu,
            cur_W,
            mean_prior=mean_prior,
            W_prior=W_prior,
            noise_variance=noise_var_full,
            rotations=rotations,
            translations=translations,
            geometry=GeometryConfig(current_size=BOX, q=args.q, volume_domain=cur_domain),
            schedule=ScheduleConfig(
                image_batch_size=args.image_batch_size, rotation_block_size=args.rotation_block_size
            ),
            scoring=ScoringConfig(relion_texture_interp=False),
            postprocess=PostprocessConfig(strategy="none", grid_correct=False),
            disc_type="linear_interp",
            freeze_mean=freeze_this,
        )
        jax.block_until_ready(result.mu_half)
        t_iter = time.time() - t0

        # Pose error
        best_rot_idx = np.asarray(result.diagnostics["best_rotation_idx"])
        est_rots = rotations[best_rot_idx]
        gt_rots = gt_rots_all[: est_rots.shape[0]]
        tr = np.einsum("nij,nij->n", est_rots.astype(np.float64), gt_rots)
        err = np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))

        # E[z] at MAP rotation
        ns = min(args.n_score_images, est_rots.shape[0])
        batch = next(dataset.iter_batches(ns, indices=np.arange(ns), by_image=True))
        imgs_real, _, _, ctf_params_b, _, _, _ = batch
        images_h = (
            np.asarray(dataset.image_source.process_images_half(np.asarray(imgs_real, dtype=np.float32)))
            .reshape(ns, -1)
            .astype(np.complex64)
        )
        ctf_h = np.asarray(
            CTFEvaluator()(jnp.asarray(ctf_params_b), (BOX, BOX), VOXEL, half_image=True), dtype=np.float32
        )
        Y1 = (images_h * ctf_h / noise_var_half[None, :]).astype(np.complex64)
        ctf2_over_noise = (ctf_h**2) / noise_var_half[None, :]
        if cur_domain == "real":
            mu_h = np.asarray(ftu.get_dft3_real(jnp.asarray(cur_mu))).reshape(-1).astype(np.complex64)
            W_h = np.stack(
                [
                    np.asarray(ftu.get_dft3_real(jnp.asarray(cur_W[k]))).reshape(-1).astype(np.complex64)
                    for k in range(args.q)
                ],
                axis=0,
            )
        else:
            mu_h = np.asarray(cur_mu, dtype=np.complex64).reshape(-1)
            W_h = np.asarray(cur_W, dtype=np.complex64).reshape(-1, args.q).T
        aug_h = np.concatenate([mu_h[None, :], W_h], axis=0)
        proj_aug = np.asarray(
            core.batch_slice_volume(
                jnp.asarray(aug_h),
                jnp.asarray(est_rots[:ns], dtype=np.float32),
                (BOX, BOX),
                (BOX, BOX, BOX),
                "linear_interp",
                half_volume=True,
                half_image=True,
            )
        )
        proj_aug = np.swapaxes(proj_aug, 0, 1)
        est_z = np.asarray(_best_pose_z_mean(jnp.asarray(Y1), jnp.asarray(ctf2_over_noise), jnp.asarray(proj_aug)))
        gt_z = gt_z_per_class[gt_assignments[:ns]]
        cc_z = float(np.corrcoef(est_z.flatten(), gt_z.flatten())[0, 1])

        # Per-iter projection capture (post-iter)
        mu_out_real_full = np.asarray(
            ftu.get_idft3_real(jnp.asarray(result.mu_half).reshape(half_shape), volume_shape=vshape)
        )
        W_half_iter = np.asarray(result.W_half)
        W_out_real_full = np.stack(
            [
                np.asarray(ftu.get_idft3_real(jnp.asarray(W_half_iter[:, k]).reshape(half_shape), volume_shape=vshape))
                for k in range(args.q)
            ],
            axis=0,
        )
        iter_slices_mu.append(mu_out_real_full.sum(axis=0).astype(np.float32).copy())
        iter_slices_W.append(W_out_real_full.sum(axis=1).astype(np.float32).copy())
        iter_rms_mu.append(float(np.sqrt(np.mean(mu_out_real_full**2))))
        iter_rms_W.append(float(np.sqrt(np.mean(W_out_real_full**2))))

        # Per-iter W shell power (the prior-diagnostic!)
        W_power = np.sum(np.abs(W_half_iter) ** 2, axis=1)  # (n_voxels_half,)
        W_shell = _shell_mean_1d(W_power, vshape)
        iter_W_shell_power.append(W_shell)

        m = {
            "iter": it,
            "wall_sec": round(t_iter, 2),
            "median_pose_err_deg": float(np.median(err)),
            "p90_pose_err_deg": float(np.percentile(err, 90)),
            "frac_err_lt_5deg": float((err < 5).mean()),
            "pmax_mean": float(result.diagnostics["pmax_mean"]),
            "log_likelihood": float(result.diagnostics["log_likelihood"]),
            "mu_rms_out_real": float(np.sqrt(np.mean(mu_out_real_full**2))),
            "W_rms_out_real": float(np.sqrt(np.mean(W_out_real_full**2))),
            "ez_vs_gt_corr": cc_z,
            "freeze_mean": freeze_this,
            "W_shell_power_recovered": W_shell.tolist(),
        }
        metrics.append(m)
        print(
            f"    med={m['median_pose_err_deg']:5.2f}°  pmax={m['pmax_mean']:.4f}  E[z]/GT={cc_z:+.4f}  "
            f"WshellRatio[lo,hi]={W_shell[2] / W_prior_per_shell[2]:.2f},{W_shell[-3] / W_prior_per_shell[-3]:.2f}"
        )

        cur_mu = np.asarray(result.mu_half)
        cur_W = np.asarray(result.W_half)
        cur_domain = "fourier_half"
        # Optional W bandlimit: low-pass W to freqs ≤ R after each M-step.
        if args.w_bandlimit_R is not None:
            R_W = int(args.w_bandlimit_R)
            mask = _lowpass_mask_half(R_W)  # (n_voxels_half,)
            cur_W = (cur_W * mask[:, None]).astype(np.complex64)
            if it == 1:
                print(f"  applied W bandlimit at R={R_W} vox (~{BOX * VOXEL / (2 * R_W):.1f}Å)")

    # Save outputs
    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset_metadata": {
            "n_images": int(dataset.n_images),
            "box": BOX,
            "voxel": VOXEL,
            "noise_level": NOISE,
            "K_gt": int(gt_vols_real.shape[0]),
        },
        "gt_svd_singular_values": [float(x) for x in gt_init.diagnostics["singular_values"]],
        "W_prior_per_shell": W_prior_per_shell.tolist(),
        "trajectory": metrics,
    }
    (args.out_dir / "trajectory.json").write_text(json.dumps(summary, indent=2))

    np.savez_compressed(
        args.out_dir / "last_state.npz",
        mu_half=np.asarray(cur_mu),
        W_half=np.asarray(cur_W),
        init_mu=mu_init,
        init_W=W_init,
        W_prior_per_shell=W_prior_per_shell,
        iter_W_shell_power=np.stack(iter_W_shell_power, axis=0),
    )

    np.savez_compressed(
        args.out_dir / "iter_slices.npz",
        mu_slices=np.stack(iter_slices_mu, axis=0),
        W_slices=np.stack(iter_slices_W, axis=0),
        mu_rms=np.asarray(iter_rms_mu),
        W_rms=np.asarray(iter_rms_W),
        W_shell_power=np.stack(iter_W_shell_power, axis=0),
        W_prior_per_shell=W_prior_per_shell,
    )

    # Per-iter projection grid
    grows = args.q + 1
    gcols = args.n_iters + 1
    fig, axes = plt.subplots(grows, gcols, figsize=(2.5 * gcols, 2.5 * grows))
    col_labels = ["init"] + [f"iter {i + 1}" for i in range(args.n_iters)]
    row_labels = ["μ"] + [f"W_{k}" for k in range(args.q)]
    for ci in range(gcols):
        ax = axes[0, ci]
        ax.imshow(iter_slices_mu[ci], cmap="gray")
        ax.set_title(f"{col_labels[ci]}\nμ RMS={iter_rms_mu[ci]:.2e}", fontsize=8)
        ax.axis("off")
        for rk in range(args.q):
            ax = axes[rk + 1, ci]
            v = iter_slices_W[ci][rk]
            vmax = float(np.abs(v).max()) + 1e-30
            ax.imshow(v, cmap="seismic", vmin=-vmax, vmax=vmax)
            ax.axis("off")
            if ci == 0:
                ax.set_title(f"{row_labels[rk + 1]}", fontsize=8)
    fig.suptitle(f"{args.label} | projections", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out_dir / "iter_mu_W_slices.png", dpi=110)
    plt.close(fig)

    # NEW: W shell-power diagnostic — radial profile of prior vs recovered W per iter
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    nshells = W_prior_per_shell.size
    x_ang = np.arange(nshells) * (VOXEL * 2)  # crude wavelength axis (just for labeling)
    ax.plot(W_prior_per_shell, "k-", lw=2, label="W prior (target)")
    for ii, W_sh in enumerate(iter_W_shell_power):
        label = "init" if ii == 0 else f"iter {ii}"
        ax.plot(W_sh, lw=1, alpha=0.8, label=label)
    ax.set_xlabel("shell index (Fourier radius, voxels)")
    ax.set_ylabel("Σ_k |W[ξ,k]|² shell-averaged")
    ax.set_yscale("log")
    ax.set_title(f"{args.label} | W shell power vs prior (well-scaled → recovered ≈ prior)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "W_shell_diagnostic.png", dpi=110)
    plt.close(fig)
    print("saved iter_mu_W_slices.png + W_shell_diagnostic.png + trajectory.json + last_state.npz + iter_slices.npz")


if __name__ == "__main__":
    main()
