"""Diagnostic v3: cluster on residual-PC coordinates instead of PPCA alpha_hat.

v2 findings:
- True-label cluster-mean SVD K=16 reaches hun=0.89 (oracle basin).
- Bootstrap through phase1 PPCA fails: phase1 alpha_hat is biased by
  the phase1 bad-basin U, so kmeans labels are ~0.48 Hungarian.
- K-scan and iteration don't help.

Root cause: phase1 PPCA M-steps land in a bad local minimum, and
alpha_hat is projected onto that U's q=8 basis, which cannot separate
all 16 states.

v3 idea: **bypass PPCA entirely** for clustering.
  1. Compute per-image real-space residual backprojections r_i (what we
     already have for cluster-mean SVD).
  2. SVD all r_i → top-q_clust residual PCs (real voxel space).
  3. Project each r_i onto those PCs → (N, q_clust) coordinate matrix.
  4. KMeans K in {8,16,32} on that coordinate matrix.
  5. Cluster-mean SVD(q=q_ppca) from those labels → U_init.
  6. 12 M-steps.

If q_clust > q_ppca (say q_clust=16,32) the clustering space has enough
capacity to separate all 16 states, and the PPCA model only sees the
final q=8 U init.

Run:
  cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
  CUDA_VISIBLE_DEVICES=0 RECOVAR_DISABLE_CUDA=1 pixi run python \
      scripts/ppca_abinitio/diag_cluster_bootstrap_v3.py --q 8
"""

from __future__ import annotations

import argparse
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
    rr = real_residuals - real_residuals.mean(axis=0, keepdims=True)
    _, S_svd, Vh = np.linalg.svd(rr, full_matrices=False)
    V = Vh[:q].T
    U_half_rows = []
    for k in range(q):
        pc_real = V[:, k].reshape(volume_shape)
        U_half_rows.append(real_volume_to_half(jnp.asarray(pc_real), volume_shape))
    U_half = jnp.stack(U_half_rows).astype(jnp.complex128)
    weights_v = make_half_volume_weights(volume_shape)
    return real_volume_orthonormalize_half(U_half, weights_v, int(np.prod(volume_shape))), S_svd


def cluster_mean_SVD_from_labels(real_residuals, labels, q, volume_shape):
    n_clusters = int(labels.max()) + 1
    cluster_means = np.zeros((n_clusters, real_residuals.shape[1]), dtype=np.float64)
    for k in range(n_clusters):
        mask = labels == k
        if mask.any():
            cluster_means[k] = real_residuals[mask].mean(axis=0)
    return svd_to_U_half(cluster_means, q, volume_shape)[0]


def residual_pc_coords(real_residuals, q_clust):
    """Top-q_clust residual voxel PCs and per-image projections onto them.

    Returns (coords, Vh, S) where coords has shape (N, q_clust).
    """
    rr = real_residuals - real_residuals.mean(axis=0, keepdims=True)
    _, S, Vh = np.linalg.svd(rr, full_matrices=False)
    V = Vh[:q_clust].T
    coords = rr @ V
    return coords, Vh, S


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
    }


def run_m_steps(cfg, ds, init, mu_true, n_steps):
    cur = init
    for _ in range(n_steps):
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
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--healpix-order", type=int, default=1)
    args = ap.parse_args()

    print(
        f"### cluster-bootstrap-v3 q={args.q} vol={args.vol} n_img={args.n_images} sigma={args.sigma} ###",
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
    labels_true = np.asarray(ds.state_label_true, dtype=np.int64)
    n_true_states = int(np.asarray(ds.state_coords_true).shape[0])

    def eval_ppca(U_init, label, n_mstep=12):
        init = PPCAInit(
            mu=mu_true.astype(jnp.complex128),
            U=U_init.astype(jnp.complex128),
            s=s_kernel,
            volume_shape=tuple(int(x) for x in cfg.volume_shape),
        )
        m0 = summarize_metrics(cfg, ds, init)
        cur = run_m_steps(cfg, ds, init, mu_true, n_mstep)
        m_final = summarize_metrics(cfg, ds, cur)
        pe = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
        print(
            f"  [{label:>40s}] k=0 hun={m0['hun']:.4f}  "
            f"k={n_mstep} hun={m_final['hun']:.4f} pe={pe:.3f} logm={m_final['logm']:.3e}",
            flush=True,
        )
        return cur, m_final

    # ---- argmax poses + real-space backprojections (used throughout) ----
    r_arg, t_arg = argmax_poses(cfg, ds, mu_true, args.q)
    r_true = np.asarray(ds.r_true_idx, dtype=np.int64)
    t_true = np.asarray(ds.t_true_idx, dtype=np.int64)
    print(
        f"argmax rot acc: {float(np.mean(r_arg == r_true)):.4f}  trans acc: {float(np.mean(t_arg == t_true)):.4f}",
        flush=True,
    )

    print("\n[building per-image real-space backprojections from argmax poses]", flush=True)
    bp_argmax = real_space_residual_backprojections(cfg, ds, mu_true, r_arg, t_arg)

    print("\n=== A. oracle_U reference ===", flush=True)
    eval_ppca(U_true, "oracle_U")

    print("\n=== B. cluster-mean SVD from TRUE labels (K=16, ceiling) ===", flush=True)
    U_true_lab = cluster_mean_SVD_from_labels(bp_argmax, labels_true, args.q, volume_shape)
    eval_ppca(U_true_lab, "true_labels_K16")

    # ---- q_clust sweep: cluster directly on top-q_clust residual voxel PCs ----
    print("\n=== C. Cluster on top-q_clust residual voxel PCs (no PPCA phase1) ===", flush=True)
    for q_clust in (8, 16, 24, 32):
        coords, _, S_svd = residual_pc_coords(bp_argmax, q_clust)
        print(
            f"\n--- q_clust={q_clust}, top S_svd={S_svd[: min(q_clust, 8)].round(2)} ---",
            flush=True,
        )
        for K in (16, 32):
            km = KMeans(n_clusters=K, n_init=20, random_state=0)
            labels_pred = km.fit_predict(coords)
            hun_lab = _hungarian(labels_true, labels_pred) if K >= n_true_states else -1.0
            U_init = cluster_mean_SVD_from_labels(bp_argmax, labels_pred, args.q, volume_shape)
            eval_ppca(U_init, f"residPC_qc{q_clust}_K{K}_hun{hun_lab:.2f}")

    # ---- whitened variant ----
    print("\n=== D. Whitened coords (divide by singular values) ===", flush=True)
    for q_clust in (16, 32):
        rr = bp_argmax - bp_argmax.mean(axis=0, keepdims=True)
        _, S_svd, Vh = np.linalg.svd(rr, full_matrices=False)
        V = Vh[:q_clust].T
        coords_raw = rr @ V
        coords_white = coords_raw / np.maximum(S_svd[:q_clust], 1e-8)[None, :]
        for K in (16, 32):
            km = KMeans(n_clusters=K, n_init=20, random_state=0)
            labels_pred = km.fit_predict(coords_white)
            hun_lab = _hungarian(labels_true, labels_pred) if K >= n_true_states else -1.0
            U_init = cluster_mean_SVD_from_labels(bp_argmax, labels_pred, args.q, volume_shape)
            eval_ppca(U_init, f"whitePC_qc{q_clust}_K{K}_hun{hun_lab:.2f}")

    # ---- multi-seed consensus: pick the kmeans seed with the highest post-Mstep logm ----
    print("\n=== E. Multi-seed kmeans: best-by-logm on residPC qc=16 K=16 ===", flush=True)
    coords16, _, _ = residual_pc_coords(bp_argmax, 16)
    best_logm = -np.inf
    best_label = None
    best_hun = 0.0
    for seed in range(8):
        km = KMeans(n_clusters=16, n_init=10, random_state=seed)
        labels_pred = km.fit_predict(coords16)
        U_init = cluster_mean_SVD_from_labels(bp_argmax, labels_pred, args.q, volume_shape)
        init = PPCAInit(
            mu=mu_true.astype(jnp.complex128),
            U=U_init.astype(jnp.complex128),
            s=s_kernel,
            volume_shape=tuple(int(x) for x in cfg.volume_shape),
        )
        cur = run_m_steps(cfg, ds, init, mu_true, 12)
        m = summarize_metrics(cfg, ds, cur)
        hun_lab = _hungarian(labels_true, labels_pred)
        print(
            f"  seed={seed}: label_hun={hun_lab:.4f}  ppca_hun={m['hun']:.4f}  logm={m['logm']:.3e}",
            flush=True,
        )
        if m["logm"] > best_logm:
            best_logm = m["logm"]
            best_label = labels_pred
            best_hun = m["hun"]
    print(
        f"  BEST-by-logm: label_hun_final={_hungarian(labels_true, best_label):.4f}  "
        f"ppca_hun={best_hun:.4f}  logm={best_logm:.3e}",
        flush=True,
    )

    # ---- final summary comparison ----
    print("\n=== SUMMARY ===", flush=True)
    print("  oracle:          logm~1.988e9  hun~0.995", flush=True)
    print("  true_labels_K16: logm~1.988e9  hun~0.895", flush=True)
    print("  Looking for: residPC_qc{X}_K{Y} that beats 0.60 hun after 12 M-steps", flush=True)


if __name__ == "__main__":
    main()
