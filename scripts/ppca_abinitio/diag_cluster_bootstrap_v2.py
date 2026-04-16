"""Diagnostic v2: scan n_clusters K and multi-round bootstrap.

v1 findings:
- K=16 true state labels → cluster-mean SVD reaches hun=0.89, logm=1.988e9 (oracle basin).
- K=16 learned from phase1 alpha_hat → 0.66 at k=0 but drops to 0.59 after 12 M-steps.
- The bottleneck is cluster purity.

Hypotheses:
  (i)  Larger K (K=32, 64) = more pure clusters = cleaner PCA basis.
  (ii) Iterate bootstrap rounds: each round improves alpha_hat →
       better cluster labels → better U → better alpha_hat.

This diagnostic runs, all with mu frozen at mu_true:

  A. oracle_U reference.
  B. K-scan: cluster-mean SVD with K in {16, 32, 64} using TRUE state
     labels as "oracle cluster labels" (cheating upper bound per K).
  C. Bootstrap K-scan: phase1 (SVD + 12 M-steps) → cluster K → 12 M-steps.
  D. Iterated bootstrap: 3 rounds of (cluster → SVD → 12 M-steps) at
     the best-performing K from C.
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
        f"### cluster-bootstrap-v2 q={args.q} vol={args.vol} n_img={args.n_images} sigma={args.sigma} ###",
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
            f"  [{label:>30s}] k=0 hun={m0['hun']:.4f} logm={m0['logm']:.3e}  "
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
    print("[building per-image real-space backprojections from true poses]", flush=True)
    bp_true = real_space_residual_backprojections(cfg, ds, mu_true, r_true, t_true)

    print("\n=== A. oracle_U reference ===", flush=True)
    eval_ppca(U_true, "oracle_U")

    print("\n=== B. cluster-mean SVD from TRUE labels, K-scan ===", flush=True)
    # B uses true-pose backprojections + true state labels (with K=n_true_states).
    # With K > n_true_states, labels are true state labels, so cluster count is fixed.
    U_c16 = cluster_mean_SVD_from_labels(bp_true, labels_true, args.q, volume_shape)
    eval_ppca(U_c16, "true_labels_K16_truepose")
    U_c16_argmax = cluster_mean_SVD_from_labels(bp_argmax, labels_true, args.q, volume_shape)
    eval_ppca(U_c16_argmax, "true_labels_K16_argmax")

    print("\n=== C. Bootstrap: phase1 SVD + Msteps → kmeans K → cluster-mean SVD → eval ===", flush=True)
    # Phase 1: residual-SVD warmstart → 12 M-steps
    U_svd0, _ = svd_to_U_half(bp_argmax.copy(), args.q, volume_shape)
    init0 = PPCAInit(
        mu=mu_true.astype(jnp.complex128),
        U=U_svd0.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )
    phase1 = run_m_steps(cfg, ds, init0, mu_true, 12)
    phase1_m = summarize_metrics(cfg, ds, phase1)
    alpha1 = phase1_m["alpha_hat"]
    print(
        f"  phase1 alpha_hat shape={alpha1.shape}  kmeans-K16 hun={phase1_m['hun']:.4f}  logm={phase1_m['logm']:.3e}",
        flush=True,
    )

    for K in (16, 32, 64):
        km = KMeans(n_clusters=K, n_init=10, random_state=0)
        labels_learned = km.fit_predict(alpha1)
        # Measure cluster purity by Hungarian against true labels (for K>=16).
        hun_learned = _hungarian(labels_true, labels_learned) if K == n_true_states else -1.0
        U_k = cluster_mean_SVD_from_labels(bp_argmax, labels_learned, args.q, volume_shape)
        eval_ppca(U_k, f"bootstrap_K{K}_phase2")
        if K == 16:
            print(f"  phase1 learned-labels Hungarian: {hun_learned:.4f}", flush=True)

    print("\n=== D. Iterated bootstrap: 4 rounds at K=32 ===", flush=True)
    # Use path C's phase1 result as starting point. Each round:
    #   1. cluster alpha_hat (K=32)
    #   2. cluster-mean SVD → U
    #   3. run n_steps_per_round M-steps
    #   4. store alpha_hat for next round
    K = 32
    n_steps_per_round = 4  # fewer so we don't drift
    n_rounds = 4
    cur = phase1
    for r in range(n_rounds):
        m = summarize_metrics(cfg, ds, cur)
        alpha = m["alpha_hat"]
        km = KMeans(n_clusters=K, n_init=10, random_state=r)
        labels_learned = km.fit_predict(alpha)
        U_r = cluster_mean_SVD_from_labels(bp_argmax, labels_learned, args.q, volume_shape)
        cur = PPCAInit(
            mu=mu_true.astype(jnp.complex128),
            U=U_r.astype(jnp.complex128),
            s=s_kernel,
            volume_shape=tuple(int(x) for x in cfg.volume_shape),
        )
        m_post_warm = summarize_metrics(cfg, ds, cur)
        cur = run_m_steps(cfg, ds, cur, mu_true, n_steps_per_round)
        m_post_mstep = summarize_metrics(cfg, ds, cur)
        print(
            f"  round {r}: post-warm hun={m_post_warm['hun']:.4f}  "
            f"post-Mstep hun={m_post_mstep['hun']:.4f}  "
            f"logm={m_post_mstep['logm']:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
