"""Diagnostic v7: production-like path — perturbed mu, multi-seed bootstrap,
joint mu+U loop. This tests whether the v6 breakthrough (hun=0.82 with
multi-seed kmeans ensemble) still holds when mu is NOT frozen at truth.

v6 froze mu at mu_true. In run_cryobench.py the default mu_init is
"perturbed" (mu_true + 0.5·noise) and the joint loop updates mu. The
question: does the bootstrap-picked U survive 30 joint mu+U iterations?

Pipeline:
  1. mu = perturbed(mu_true, eps=0.5)  (same as cryobench default)
  2. SVD warmstart for U  (same as cryobench default)
  3. NEW: factor-only phase1 (12 M-steps, mu frozen)
  4. NEW: multi-seed kmeans bootstrap on phase1 alpha_hat →
     best-by-logm cluster-mean SVD → factor-only phase2 (12 M-steps,
     mu frozen) → logm-best candidate U
  5. Joint loop (30 iters) from (mu_init, best U)
  6. Report final hun, ari, logm. Compare to baseline (no-bootstrap
     joint loop).

Run:
  cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
  CUDA_VISIBLE_DEVICES=0 RECOVAR_DISABLE_CUDA=1 pixi run python \
      scripts/ppca_abinitio/diag_cluster_bootstrap_v7.py
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
    project_to_real_volume_subspace,
    real_volume_orthonormalize_half,
    real_volume_to_half,
)
from recovar.em.ppca_abinitio.mean_update import update_mu_residualized
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


def _log_marginal_sum(cfg, init, ds):
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
    log_scores = np.asarray(stats.log_scores)
    flat = log_scores.reshape(log_scores.shape[0], -1)
    m = flat.max(axis=-1, keepdims=True)
    lm_per_img = (m + np.log(np.exp(flat - m).sum(axis=-1, keepdims=True))).reshape(-1)
    return float(lm_per_img.sum())


def argmax_poses_at_mu(cfg, ds, mu_half, q):
    image_shape = cfg.image_shape
    volume_shape = cfg.volume_shape
    weights_h = make_half_image_weights(image_shape)
    mean_proj = _slice_mu_half(mu_half, ds.rotations, image_shape, volume_shape).astype(jnp.complex128)
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


def real_space_residual_backprojections_at_mu(cfg, ds, mu_half, r_idx, t_idx):
    image_shape = cfg.image_shape
    volume_shape = cfg.volume_shape
    n_img = ds.batch_full.shape[0]
    rot_per = jnp.asarray(np.asarray(ds.rotations)[np.asarray(r_idx)])
    trans_per = jnp.asarray(np.asarray(ds.translations)[np.asarray(t_idx)])
    mu_proj_full = slice_volume(
        mu_half,
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
    half_volume_size = int(mu_half.shape[0])
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


def summarize_metrics(cfg, ds, init, n_states):
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


def factor_only_m_steps(cfg, ds, init, n_steps):
    """M-steps with mu FROZEN at init.mu."""
    mu_fixed = init.mu
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
        cur = PPCAInit(mu=mu_fixed, U=cur.U, s=cur.s, volume_shape=cur.volume_shape)
    return cur


def joint_m_steps(cfg, ds, init, n_steps):
    """Joint mu+U M-steps (production-like)."""
    cur = init
    for _ in range(n_steps):
        mres = update_mu_residualized(
            cfg,
            cur,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
            tau=0.0,
        )
        cur = PPCAInit(mu=mres.mu_half, U=cur.U, s=cur.s, volume_shape=cur.volume_shape)
        cur = update_factor_closed_form(
            cfg,
            cur,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
        )
    return cur


def multi_seed_bootstrap(cfg, ds, init, q, n_phase1, n_phase2, n_seeds, n_clusters):
    """Multi-seed cluster-mean SVD bootstrap with FROZEN mu.

    Returns the PPCAInit with the best log-marginal across seeds.
    """
    # Phase 1: factor-only M-steps
    phase1 = factor_only_m_steps(cfg, ds, init, n_phase1)
    # Phase1 alpha_hat + argmax backprojections at init.mu
    stats = score_and_posterior_moments_eqx(
        cfg,
        phase1.mu,
        phase1.U,
        phase1.s,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
    )
    pm = np.asarray(stats.post_mean)
    gamma = np.exp(np.asarray(stats.log_resp))
    alpha_hat = np.sum(gamma[..., None] * pm, axis=(1, 2))

    r_arg, t_arg = argmax_poses_at_mu(cfg, ds, init.mu, q)
    bp = real_space_residual_backprojections_at_mu(cfg, ds, init.mu, r_arg, t_arg)

    best_lm = -np.inf
    best_init = None
    for seed in range(n_seeds):
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        labels_pred = km.fit_predict(alpha_hat)
        U_cluster = cluster_mean_SVD_from_labels(bp, labels_pred, q, cfg.volume_shape)
        cand = PPCAInit(mu=init.mu, U=U_cluster, s=init.s, volume_shape=init.volume_shape)
        cand = factor_only_m_steps(cfg, ds, cand, n_phase2)
        lm = _log_marginal_sum(cfg, cand, ds)
        if lm > best_lm:
            best_lm = lm
            best_init = cand
    return best_init, best_lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--healpix-order", type=int, default=1)
    ap.add_argument("--mu-eps", type=float, default=0.5, help="perturbation of mu_true")
    ap.add_argument("--n-joint", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=16)
    args = ap.parse_args()

    q = args.q
    print(
        f"### cluster-bootstrap-v7 (prod-like) q={q} vol={args.vol} n_img={args.n_images} "
        f"sigma={args.sigma} n_seeds={args.n_seeds} ###",
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
        q=q,
        n_images_train=args.n_images,
        n_images_val=0,
        sigma_real=args.sigma,
        seed=args.seed,
        external_volumes_real=gt_vols,
        external_sampling_mode="discrete_volumes",
    )
    cfg = _Cfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)

    half_vol_size = ds.mu_half_true.shape[0]
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    labels_true = np.asarray(ds.state_label_true, dtype=np.int64)
    n_true_states = int(np.asarray(ds.state_coords_true).shape[0])

    # ---- Perturbed mu init (same as cryobench default) ----
    rng = np.random.default_rng(args.seed + 1)
    weights_v = make_half_volume_weights(volume_shape)
    noise = rng.standard_normal(2 * half_vol_size).view(np.complex128)
    noise_jax = project_to_real_volume_subspace(jnp.asarray(noise, dtype=jnp.complex128), volume_shape)
    mu_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(ds.mu_half_true) * ds.mu_half_true))))
    noise_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(noise_jax) * noise_jax))))
    scale = args.mu_eps * mu_norm / max(noise_norm, 1e-12)
    mu_init = (ds.mu_half_true + scale * noise_jax).astype(jnp.complex128)
    print(f"mu perturbation eps={args.mu_eps}", flush=True)

    # ---- SVD warmstart (from perturbed mu) ----
    print("\n[SVD warmstart at perturbed mu]", flush=True)
    r_arg, t_arg = argmax_poses_at_mu(cfg, ds, mu_init, q)
    bp_init = real_space_residual_backprojections_at_mu(cfg, ds, mu_init, r_arg, t_arg)
    U_svd, _ = svd_to_U_half(bp_init.copy(), q, volume_shape)
    init_svd = PPCAInit(mu=mu_init, U=U_svd, s=s_kernel, volume_shape=tuple(int(x) for x in volume_shape))
    m_svd = summarize_metrics(cfg, ds, init_svd, n_true_states)
    print(f"  k=0 (pre-EM): hun={m_svd['hun']:.4f} logm={m_svd['logm']:.3e}", flush=True)

    # ---- BASELINE: joint loop WITHOUT bootstrap (matches cryobench) ----
    print(f"\n=== BASELINE: joint loop from SVD warmstart ({args.n_joint} iters) ===", flush=True)
    cur = joint_m_steps(cfg, ds, init_svd, args.n_joint)
    m_base = summarize_metrics(cfg, ds, cur, n_true_states)
    pe_base = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
    print(
        f"  baseline final: hun={m_base['hun']:.4f} pe={pe_base:.3f} logm={m_base['logm']:.3e}",
        flush=True,
    )

    # ---- BOOTSTRAP: multi-seed at perturbed mu, then joint loop ----
    print("\n=== BOOTSTRAP: multi-seed bootstrap at perturbed mu + joint loop ===", flush=True)
    best_init, best_lm = multi_seed_bootstrap(
        cfg,
        ds,
        init_svd,
        q,
        n_phase1=12,
        n_phase2=12,
        n_seeds=args.n_seeds,
        n_clusters=n_true_states,
    )
    m_boot = summarize_metrics(cfg, ds, best_init, n_true_states)
    print(
        f"  post-bootstrap (mu perturbed, frozen): hun={m_boot['hun']:.4f} logm={m_boot['logm']:.3e}  "
        f"(best across {args.n_seeds} seeds)",
        flush=True,
    )
    cur_b = joint_m_steps(cfg, ds, best_init, args.n_joint)
    m_final = summarize_metrics(cfg, ds, cur_b, n_true_states)
    pe_final = float(projector_frobenius_error(cur_b.U, ds.U_half_true, cfg.volume_shape))
    print(
        f"  bootstrap+joint final: hun={m_final['hun']:.4f} pe={pe_final:.3f} logm={m_final['logm']:.3e}",
        flush=True,
    )

    # ---- ORACLE reference ----
    print("\n=== ORACLE: joint loop from (mu_true, U_true) ===", flush=True)
    init_or = PPCAInit(
        mu=ds.mu_half_true.astype(jnp.complex128),
        U=ds.U_half_true.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in volume_shape),
    )
    cur_or = joint_m_steps(cfg, ds, init_or, args.n_joint)
    m_or = summarize_metrics(cfg, ds, cur_or, n_true_states)
    print(f"  oracle final: hun={m_or['hun']:.4f} logm={m_or['logm']:.3e}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  baseline (cryobench): hun={m_base['hun']:.4f}", flush=True)
    print(f"  bootstrap+joint:      hun={m_final['hun']:.4f}", flush=True)
    print(f"  joint oracle ceiling: hun={m_or['hun']:.4f}", flush=True)
    print(f"  gain from bootstrap: {m_final['hun'] - m_base['hun']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
