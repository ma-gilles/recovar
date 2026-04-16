"""Diagnostic: cluster-aware warmstart strategies.

Findings so far:
- Residual-SVD in any form (complex/real/argmax/true-pose) lands in
  a bad basin with hun~0.4-0.55.
- Log-marginal barely distinguishes bad (1.985e9) from good (1.988e9).
- The M-step is too weak to escape this basin.

Hypothesis: per-image residuals have too much pose+shot noise relative
to between-state variance. Averaging backprojections PER STATE before
SVD should recover a much cleaner subspace.

Tests (mu frozen at mu_true):

  A. oracle_U — reference.
  B. argmax_per_state_mean_SVD — use argmax (r,t), then cluster via
     hard k-means on alpha_hat after a homogeneous 1D PPCA init, then
     take per-cluster mean of real-space backprojections and SVD.
  C. true_state_mean_SVD — CHEATING: use state_label_true to average
     per-image backprojections into K=16 cluster means, then SVD.
     This is the upper bound on "cluster-aware real-space PCA".
  D. warmstart_then_cluster — start from argmax SVD, run 12 M-steps,
     cluster alpha_hat, use cluster means to re-warmstart, 12 more.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import recovar.core as core
import recovar.core.fourier_transform_utils as ftu
from recovar.core.slicing import adjoint_slice_volume, slice_volume
from recovar.em.ppca_abinitio.factor_update import update_factor_closed_form
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    half_to_real_volume,
    make_half_volume_weights,
    real_volume_orthonormalize_half,
    real_volume_to_half,
)
from recovar.em.ppca_abinitio.metrics import projector_frobenius_error
from recovar.em.ppca_abinitio.posterior import (
    _preprocess_batch_to_half,
    _slice_mu_half,
    make_half_image_weights,
    score_and_posterior_moments_eqx,
    score_from_half_image_projections,
)
from recovar.em.ppca_abinitio.synthetic import (
    SyntheticFamily,
    make_synthetic_fixed_grid_dataset,
)
from recovar.em.ppca_abinitio.types import PPCAInit
from recovar.utils.helpers import load_mrc

_S_FLOOR = 1e-6


class _Cfg(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, p, *, half_image=False):
        n = p.shape[0]
        sz = int(np.prod(self.image_shape))
        full = jnp.ones((n, sz), dtype=jnp.float64)
        if half_image:
            return ftu.full_image_to_half_image(full, self.image_shape)
        return full

    def process_fn(self, b, apply_image_mask=False):
        return b


def _hungarian(labels_true, labels_pred):
    n = len(labels_true)
    k = max(int(labels_true.max()), int(labels_pred.max())) + 1
    C = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(labels_true, labels_pred):
        C[int(t), int(p)] += 1
    row_ind, col_ind = linear_sum_assignment(-C)
    return float(C[row_ind, col_ind].sum()) / n


def downsample_volume(vol_real, target_D):
    D = vol_real.shape[-1]
    if target_D == D:
        return vol_real.astype(np.float64)
    F = np.asarray(ftu.get_dft3(jnp.asarray(vol_real)), dtype=np.complex128)
    c = D // 2
    h = target_D // 2
    F_crop = F[c - h : c + h, c - h : c + h, c - h : c + h]
    out = np.array(np.asarray(ftu.get_idft3(jnp.asarray(F_crop))).real, dtype=np.float64, copy=True)
    out *= (target_D / D) ** 3
    return out


def load_cryobench_gt_volumes(dataset_root, target_D):
    candidates = [dataset_root / "vols" / "128_org", dataset_root / "vols"]
    vol_dir = next((p for p in candidates if p.exists()), None)
    if vol_dir is None:
        raise FileNotFoundError(f"no vol dir found under {dataset_root}/vols")
    vol_files = sorted(vol_dir.glob("*.mrc"))
    vols = np.stack([np.asarray(load_mrc(str(vf)), dtype=np.float64) for vf in vol_files])
    if vols.shape[-1] != target_D:
        vols = np.stack([downsample_volume(vols[k], target_D) for k in range(vols.shape[0])])
    return vols


def argmax_poses(cfg, ds, mu_true, q):
    image_shape = cfg.image_shape
    volume_shape = cfg.volume_shape
    weights_h = make_half_image_weights(image_shape)
    mean_proj = _slice_mu_half(mu_true, ds.rotations, image_shape, volume_shape).astype(jnp.complex128)
    n_rot = ds.rotations.shape[0]
    u_zero = jnp.zeros((n_rot, q, mean_proj.shape[-1]), dtype=jnp.complex128)
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    shifted_half, ctf2_over_nv_half, _ = _preprocess_batch_to_half(
        cfg, ds.batch_full, ds.translations, ds.ctf_params, ds.noise_variance_full
    )
    stats = score_from_half_image_projections(mean_proj, u_zero, s_kernel, shifted_half, ctf2_over_nv_half, weights_h)
    log_resp = np.asarray(stats.log_resp)
    n_img = log_resp.shape[0]
    n_trans = log_resp.shape[-1]
    arg = log_resp.reshape(n_img, -1).argmax(axis=-1)
    return arg // n_trans, arg % n_trans


def real_space_residual_backprojections(cfg, ds, mu_true, r_idx, t_idx):
    """Return per-image real-space residual backprojections (n_img, N_vox).

    Full-image FT path, matching residual_pca_baseline.
    """
    image_shape = cfg.image_shape
    volume_shape = cfg.volume_shape
    n_img = ds.batch_full.shape[0]

    rot_per = jnp.asarray(np.asarray(ds.rotations)[np.asarray(r_idx)])
    trans_per = jnp.asarray(np.asarray(ds.translations)[np.asarray(t_idx)])

    mu_proj_full = slice_volume(
        mu_true,
        rot_per,
        image_shape,
        volume_shape,
        "nearest",
        half_volume=True,
        half_image=False,
    )

    trans_for_shift = trans_per[:, None, :]
    shifted_full = core.batch_trans_translate_images(ds.batch_full, trans_for_shift, image_shape)[:, 0, :]
    residuals_full = shifted_full - mu_proj_full

    half_volume_size = int(mu_true.shape[0])
    backproj_half = np.zeros((n_img, half_volume_size), dtype=np.complex128)
    for i in range(n_img):
        b = adjoint_slice_volume(
            residuals_full[i : i + 1],
            rot_per[i : i + 1],
            image_shape,
            volume_shape,
            "nearest",
            half_image=False,
            half_volume=True,
        )
        backproj_half[i] = np.asarray(b).reshape(-1)

    real_residuals = np.empty((n_img, int(np.prod(volume_shape))), dtype=np.float64)
    for i in range(n_img):
        rv = half_to_real_volume(jnp.asarray(backproj_half[i]), volume_shape)
        real_residuals[i] = np.asarray(rv).reshape(-1)
    return real_residuals


def svd_to_U_half(real_residuals, q, volume_shape):
    real_residuals = real_residuals - real_residuals.mean(axis=0, keepdims=True)
    _, S_svd, Vh = np.linalg.svd(real_residuals, full_matrices=False)
    print(f"  top-{q} real-space singular values: {S_svd[:q]}", flush=True)
    V = Vh[:q].T  # (n_voxels, q)
    U_half_rows = []
    for k in range(q):
        pc_real = V[:, k].reshape(volume_shape)
        U_half_rows.append(real_volume_to_half(jnp.asarray(pc_real), volume_shape))
    U_half = jnp.stack(U_half_rows).astype(jnp.complex128)
    weights_v = make_half_volume_weights(volume_shape)
    return real_volume_orthonormalize_half(U_half, weights_v, int(np.prod(volume_shape)))


def cluster_mean_SVD_from_labels(real_residuals, labels, q, volume_shape):
    """Given per-image backprojections and cluster labels, compute per-
    cluster mean backprojection and SVD it."""
    n_clusters = int(labels.max()) + 1
    K = n_clusters
    cluster_means = np.zeros((K, real_residuals.shape[1]), dtype=np.float64)
    for k in range(K):
        mask = labels == k
        if mask.any():
            cluster_means[k] = real_residuals[mask].mean(axis=0)
    print(
        f"  cluster sizes: min={int(np.min(np.bincount(labels))):d} "
        f"max={int(np.max(np.bincount(labels))):d} n_clusters={K}",
        flush=True,
    )
    return svd_to_U_half(cluster_means, q, volume_shape)


def summarize_metrics(cfg, ds, init):
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
    pm = np.asarray(stats.post_mean)
    gamma = np.exp(np.asarray(stats.log_resp))
    alpha_hat = np.sum(gamma[..., None] * pm, axis=(1, 2))
    labels_true = np.asarray(ds.state_label_true, dtype=np.int64)
    n_states = int(np.asarray(ds.state_coords_true).shape[0])
    km = KMeans(n_clusters=n_states, n_init=10, random_state=0)
    cluster_labels = km.fit_predict(alpha_hat)
    log_scores = np.asarray(stats.log_scores)
    per_img_log_marg = np.asarray(jax.scipy.special.logsumexp(log_scores.reshape(log_scores.shape[0], -1), axis=-1))
    return {
        "hun": _hungarian(labels_true, cluster_labels),
        "ari": float(adjusted_rand_score(labels_true, cluster_labels)),
        "logm": float(per_img_log_marg.sum()),
        "alpha_hat": alpha_hat,
        "pred_labels": cluster_labels,
    }


def evaluate_from_U(cfg, ds, U_init, mu_true, s_kernel, k_steps, label=""):
    init = PPCAInit(
        mu=mu_true.astype(jnp.complex128),
        U=U_init.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )
    pe = float(projector_frobenius_error(init.U, ds.U_half_true, cfg.volume_shape))
    m = summarize_metrics(cfg, ds, init)
    print(
        f"  [{label:>26s} k= 0] pe={pe:.4f} hun={m['hun']:.4f} ari={m['ari']:.4f} logm={m['logm']:.3e}",
        flush=True,
    )
    rows = [(0, pe, m["hun"], m["ari"], m["logm"])]
    cur = init
    last_m = m
    for k in range(1, max(k_steps) + 1):
        cur = update_factor_closed_form(
            cfg,
            cur,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
        )
        cur = PPCAInit(mu=mu_true.astype(jnp.complex128), U=cur.U, s=cur.s, volume_shape=cur.volume_shape)
        if k in k_steps:
            pe = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
            m = summarize_metrics(cfg, ds, cur)
            rows.append((k, pe, m["hun"], m["ari"], m["logm"]))
            last_m = m
            print(
                f"  [{label:>26s} k={k:>2}] pe={pe:.4f} hun={m['hun']:.4f} ari={m['ari']:.4f} logm={m['logm']:.3e}",
                flush=True,
            )
    return rows, cur, last_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--healpix-order", type=int, default=1)
    ap.add_argument("--n-clusters", type=int, default=16)
    args = ap.parse_args()

    print(
        f"### cluster-warmstart q={args.q} vol={args.vol} n_img={args.n_images} sigma={args.sigma} ###",
        flush=True,
    )

    root = Path("/home/mg6942/mytigress/cryobench2") / "Ribosembly"
    gt_vols = load_cryobench_gt_volumes(root, target_D=args.vol)
    grid = build_fixed_grid(healpix_order=args.healpix_order, max_shift=1)
    image_shape = (args.vol, args.vol)
    volume_shape = (args.vol, args.vol, args.vol)
    ds = make_synthetic_fixed_grid_dataset(
        SyntheticFamily.MATCHED_GRID_HET,
        volume_shape=volume_shape,
        image_shape=image_shape,
        grid=grid,
        q=args.q,
        n_images_train=args.n_images,
        n_images_val=0,
        sigma_real=args.sigma,
        seed=args.seed,
        external_volumes_real=gt_vols,
        external_sampling_mode="discrete_volumes",
    )
    cfg = _Cfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)

    mu_true = ds.mu_half_true.astype(jnp.complex128)
    U_true = ds.U_half_true.astype(jnp.complex128)
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    k_steps = sorted(set(k for k in (0, 1, 3, args.k_max) if k <= args.k_max))

    # Compute argmax poses once — used by variants B and D.
    r_arg, t_arg = argmax_poses(cfg, ds, mu_true, args.q)
    r_true = np.asarray(ds.r_true_idx, dtype=np.int64)

    print("\n=== A. oracle_U ===", flush=True)
    rows_oracle, _, _ = evaluate_from_U(cfg, ds, U_true, mu_true, s_kernel, k_steps, label="oracle_U")

    # Real-space backprojections, used by B/C/D.
    print("\n[building real-space backprojections from argmax poses...]", flush=True)
    t0 = time.perf_counter()
    bp_argmax = real_space_residual_backprojections(cfg, ds, mu_true, r_arg, t_arg)
    print(f"  argmax backproj wall: {time.perf_counter() - t0:.1f}s", flush=True)

    print("\n=== B. argmax_kmeans_cluster_mean_SVD ===", flush=True)
    # Cluster the per-image backprojections directly in voxel space and
    # take cluster means. This is a "noisy" warmstart: the clusters are
    # likely wrong because voxel-space clustering is fragile, but we
    # want to see if it helps at all.
    t0 = time.perf_counter()
    km = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=0)
    labels_km = km.fit_predict(bp_argmax)
    print(f"  kmeans wall: {time.perf_counter() - t0:.1f}s", flush=True)
    # Evaluate initial Hungarian score of the clustering itself
    labels_true = np.asarray(ds.state_label_true, dtype=np.int64)
    init_cluster_hun = _hungarian(labels_true, labels_km)
    print(f"  voxel-space kmeans initial hun: {init_cluster_hun:.4f}", flush=True)
    U_b = cluster_mean_SVD_from_labels(bp_argmax, labels_km, args.q, volume_shape)
    rows_b, _, _ = evaluate_from_U(cfg, ds, U_b, mu_true, s_kernel, k_steps, label="argmax_km_cluster_SVD")

    print("\n=== C. true_state_mean_SVD (cheating, uses state_label_true) ===", flush=True)
    # Use true state labels + true poses for the cleanest possible
    # warmstart. This is an upper bound on what cluster-mean SVD can
    # achieve.
    t0 = time.perf_counter()
    bp_true_pose = real_space_residual_backprojections(cfg, ds, mu_true, r_true, ds.t_true_idx)
    print(f"  true pose backproj wall: {time.perf_counter() - t0:.1f}s", flush=True)
    U_c = cluster_mean_SVD_from_labels(bp_true_pose, labels_true, args.q, volume_shape)
    rows_c, _, _ = evaluate_from_U(cfg, ds, U_c, mu_true, s_kernel, k_steps, label="true_state_mean_SVD")

    print(
        "\n=== D. Bootstrap: SVD warmstart → 12 M-steps → cluster alpha_hat → cluster_mean_SVD → 12 M-steps ===",
        flush=True,
    )
    # Step 1: argmax real-space SVD warmstart
    U_d0 = svd_to_U_half(bp_argmax.copy(), args.q, volume_shape)
    # Step 2: run 12 M-steps
    rows_d0, init_d0, last_m_d0 = evaluate_from_U(cfg, ds, U_d0, mu_true, s_kernel, k_steps, label="bootstrap_phase1")
    # Step 3: cluster alpha_hat (q-D latent)
    alpha_hat_d = last_m_d0["alpha_hat"]
    km2 = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=0)
    labels_alpha = km2.fit_predict(alpha_hat_d)
    alpha_cluster_hun = _hungarian(labels_true, labels_alpha)
    print(f"  phase1 alpha_hat kmeans hun: {alpha_cluster_hun:.4f}", flush=True)
    # Step 4: cluster-mean SVD using alpha_hat labels
    U_d1 = cluster_mean_SVD_from_labels(bp_argmax, labels_alpha, args.q, volume_shape)
    # Step 5: run 12 more M-steps
    rows_d1, _, _ = evaluate_from_U(cfg, ds, U_d1, mu_true, s_kernel, k_steps, label="bootstrap_phase2")

    print("\n=== SUMMARY ===", flush=True)
    rows_all = {
        "oracle_U": rows_oracle,
        "argmax_km_cluster_SVD": rows_b,
        "true_state_mean_SVD": rows_c,
        "bootstrap_phase1": rows_d0,
        "bootstrap_phase2": rows_d1,
    }
    for k_idx, k in enumerate(k_steps):
        print(f"\n-- k = {k} --", flush=True)
        for name, rows in rows_all.items():
            r = rows[k_idx]
            print(
                f"  {name:>26s}: pe={r[1]:.3f} hun={r[2]:.4f} ari={r[3]:.4f} logm={r[4]:.3e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
