"""Diagnostic v8: understand why bootstrap fails under perturbed mu.

v7 result: baseline joint hun=0.44, bootstrap+joint hun=0.50, oracle 0.955.
v7 bootstrap gives only +0.07 whereas v6 frozen-mu_true gave 0.82. Why?

Hypotheses to test:
  H1. At perturbed mu, NO seed produces a high-hun cluster-mean U
      (alpha_hat is corrupted by wrong mu).
  H2. Some seeds produce high-hun U but logm selection no longer picks
      them (logm/hun correlation breaks in perturbed-mu regime).
  H3. Even high-hun seeds lose their advantage after a 30-iter joint
      loop (the joint loop converges to a common mu-U fixed point
      regardless of U init).
  H4. Homogeneous burn-in first can recover mu enough that bootstrap
      then works (matches v6 intuition).

Cases:
  - A. BASELINE: SVD warmstart → joint loop (no burn-in, no bootstrap)
  - B. BURNIN: 15 burn-in iters → SVD → joint loop
  - C. BOOTSTRAP_LM: SVD → bootstrap (best by logm) → joint loop (v7)
  - D. BOOTSTRAP_HUN: SVD → bootstrap (best by hun, CHEAT) → joint loop
  - E. BURNIN_BOOTSTRAP: burn-in → SVD → bootstrap (best by logm) → joint
  - F. BURNIN_BOOTSTRAP_HUN: burn-in → SVD → bootstrap (best by hun) → joint
  - G. ORACLE: (mu_true, U_true) → joint loop

Per-seed per-case output of hun, logm, pe for the 16 candidate U's so we
can see the landscape directly.

Run:
  cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
  CUDA_VISIBLE_DEVICES=0 RECOVAR_DISABLE_CUDA=1 pixi run python \
      scripts/ppca_abinitio/diag_cluster_bootstrap_v8.py
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
from recovar.em.ppca_abinitio.mean_update import (
    update_mu_homogeneous,
    update_mu_residualized,
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


def homogeneous_burnin(cfg, ds, init, n_steps):
    cur = init
    for _ in range(n_steps):
        mres = update_mu_homogeneous(
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
    return cur


def bootstrap_candidates(cfg, ds, init, q, n_phase1, n_phase2, n_seeds, n_clusters):
    """Produce one candidate init per kmeans seed.

    Returns:
      candidates: list of PPCAInit
      seeds:      list of ints matching candidates
    """
    phase1 = factor_only_m_steps(cfg, ds, init, n_phase1)
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

    candidates = []
    for seed in range(n_seeds):
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        labels_pred = km.fit_predict(alpha_hat)
        U_cluster = cluster_mean_SVD_from_labels(bp, labels_pred, q, cfg.volume_shape)
        cand = PPCAInit(mu=init.mu, U=U_cluster, s=init.s, volume_shape=init.volume_shape)
        cand = factor_only_m_steps(cfg, ds, cand, n_phase2)
        candidates.append(cand)
    return candidates


def evaluate_candidates(cfg, ds, candidates, n_states, label=""):
    """Print per-seed hun + logm. Return (best_lm_idx, best_hun_idx)."""
    huns, logms = [], []
    for i, cand in enumerate(candidates):
        m = summarize_metrics(cfg, ds, cand, n_states)
        huns.append(m["hun"])
        logms.append(m["logm"])
        print(f"    {label} seed={i:2d}: hun={m['hun']:.4f} logm={m['logm']:.3e}", flush=True)
    huns = np.asarray(huns)
    logms = np.asarray(logms)
    print(
        f"    {label} hun: mean={huns.mean():.4f} max={huns.max():.4f} min={huns.min():.4f}  "
        f"logm range={logms.max() - logms.min():.3e} (rel={(logms.max() - logms.min()) / abs(logms.mean()):.2e})",
        flush=True,
    )
    if len(logms) >= 2:
        corr = float(np.corrcoef(logms, huns)[0, 1])
        print(f"    {label} pearson(logm, hun) = {corr:+.4f}", flush=True)
    best_lm_idx = int(np.argmax(logms))
    best_hun_idx = int(np.argmax(huns))
    print(
        f"    {label} best-by-logm: seed={best_lm_idx} hun={huns[best_lm_idx]:.4f} | "
        f"best-by-hun: seed={best_hun_idx} hun={huns[best_hun_idx]:.4f}",
        flush=True,
    )
    return best_lm_idx, best_hun_idx, huns, logms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--healpix-order", type=int, default=1)
    ap.add_argument("--mu-eps", type=float, default=0.5)
    ap.add_argument("--n-burnin", type=int, default=15)
    ap.add_argument("--n-joint", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=16)
    ap.add_argument("--n-phase1", type=int, default=12)
    ap.add_argument("--n-phase2", type=int, default=12)
    args = ap.parse_args()

    q = args.q
    print(
        f"### cluster-bootstrap-v8 (per-seed + burnin) q={q} vol={args.vol} n_img={args.n_images} "
        f"sigma={args.sigma} n_seeds={args.n_seeds} n_burnin={args.n_burnin} ###",
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
    n_true_states = int(np.asarray(ds.state_coords_true).shape[0])

    # perturbed mu
    rng = np.random.default_rng(args.seed + 1)
    weights_v = make_half_volume_weights(volume_shape)
    noise = rng.standard_normal(2 * half_vol_size).view(np.complex128)
    noise_jax = project_to_real_volume_subspace(jnp.asarray(noise, dtype=jnp.complex128), volume_shape)
    mu_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(ds.mu_half_true) * ds.mu_half_true))))
    noise_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(noise_jax) * noise_jax))))
    scale = args.mu_eps * mu_norm / max(noise_norm, 1e-12)
    mu_perturbed = (ds.mu_half_true + scale * noise_jax).astype(jnp.complex128)
    print(f"mu perturbation eps={args.mu_eps}", flush=True)

    results = {}

    def do_svd_warm(mu_half, tag):
        r_arg, t_arg = argmax_poses_at_mu(cfg, ds, mu_half, q)
        bp = real_space_residual_backprojections_at_mu(cfg, ds, mu_half, r_arg, t_arg)
        U_svd, _ = svd_to_U_half(bp.copy(), q, volume_shape)
        init = PPCAInit(mu=mu_half, U=U_svd, s=s_kernel, volume_shape=tuple(int(x) for x in volume_shape))
        m = summarize_metrics(cfg, ds, init, n_true_states)
        print(f"  [{tag}] pre-EM: hun={m['hun']:.4f} logm={m['logm']:.3e}", flush=True)
        return init

    # ===== CASE A: BASELINE =====
    print("\n=== A. BASELINE (SVD warmstart → joint loop) ===", flush=True)
    init_A = do_svd_warm(mu_perturbed, "A")
    cur_A = joint_m_steps(cfg, ds, init_A, args.n_joint)
    m_A = summarize_metrics(cfg, ds, cur_A, n_true_states)
    pe_A = float(projector_frobenius_error(cur_A.U, ds.U_half_true, cfg.volume_shape))
    print(f"  A final: hun={m_A['hun']:.4f} pe={pe_A:.3f} logm={m_A['logm']:.3e}", flush=True)
    results["A_baseline"] = m_A["hun"]

    # ===== CASE B: BURNIN =====
    print("\n=== B. BURNIN (homogeneous burnin → SVD → joint loop) ===", flush=True)
    init_zero_U = PPCAInit(
        mu=mu_perturbed,
        U=jnp.zeros((q, half_vol_size), dtype=jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in volume_shape),
    )
    burnt = homogeneous_burnin(cfg, ds, init_zero_U, args.n_burnin)
    mu_burnt = burnt.mu
    init_B = do_svd_warm(mu_burnt, "B")
    cur_B = joint_m_steps(cfg, ds, init_B, args.n_joint)
    m_B = summarize_metrics(cfg, ds, cur_B, n_true_states)
    pe_B = float(projector_frobenius_error(cur_B.U, ds.U_half_true, cfg.volume_shape))
    print(f"  B final: hun={m_B['hun']:.4f} pe={pe_B:.3f} logm={m_B['logm']:.3e}", flush=True)
    results["B_burnin"] = m_B["hun"]

    # ===== CASE C: BOOTSTRAP_LM (perturbed mu) =====
    print("\n=== C. BOOTSTRAP_LM (perturbed mu: SVD → bootstrap best-by-logm → joint) ===", flush=True)
    cands_C = bootstrap_candidates(cfg, ds, init_A, q, args.n_phase1, args.n_phase2, args.n_seeds, n_true_states)
    print("  C per-seed at perturbed mu (frozen):", flush=True)
    best_lm_C, best_hun_C, huns_C, logms_C = evaluate_candidates(cfg, ds, cands_C, n_true_states, label="C")
    cur_C = joint_m_steps(cfg, ds, cands_C[best_lm_C], args.n_joint)
    m_C = summarize_metrics(cfg, ds, cur_C, n_true_states)
    pe_C = float(projector_frobenius_error(cur_C.U, ds.U_half_true, cfg.volume_shape))
    print(f"  C final (best-by-logm + joint): hun={m_C['hun']:.4f} pe={pe_C:.3f} logm={m_C['logm']:.3e}", flush=True)
    results["C_bootstrap_lm"] = m_C["hun"]

    # ===== CASE D: BOOTSTRAP_HUN (perturbed mu, cheat) =====
    print("\n=== D. BOOTSTRAP_HUN (perturbed mu, CHEAT: best-by-hun seed + joint) ===", flush=True)
    cur_D = joint_m_steps(cfg, ds, cands_C[best_hun_C], args.n_joint)
    m_D = summarize_metrics(cfg, ds, cur_D, n_true_states)
    pe_D = float(projector_frobenius_error(cur_D.U, ds.U_half_true, cfg.volume_shape))
    print(f"  D final (best-by-hun + joint): hun={m_D['hun']:.4f} pe={pe_D:.3f} logm={m_D['logm']:.3e}", flush=True)
    results["D_bootstrap_hun"] = m_D["hun"]

    # ===== CASE E: BURNIN + BOOTSTRAP_LM =====
    print("\n=== E. BURNIN_BOOTSTRAP_LM (burnin → SVD → bootstrap best-by-logm → joint) ===", flush=True)
    cands_E = bootstrap_candidates(cfg, ds, init_B, q, args.n_phase1, args.n_phase2, args.n_seeds, n_true_states)
    print("  E per-seed at burnt-in mu (frozen):", flush=True)
    best_lm_E, best_hun_E, huns_E, logms_E = evaluate_candidates(cfg, ds, cands_E, n_true_states, label="E")
    cur_E = joint_m_steps(cfg, ds, cands_E[best_lm_E], args.n_joint)
    m_E = summarize_metrics(cfg, ds, cur_E, n_true_states)
    pe_E = float(projector_frobenius_error(cur_E.U, ds.U_half_true, cfg.volume_shape))
    print(
        f"  E final (burnin + best-by-logm + joint): hun={m_E['hun']:.4f} pe={pe_E:.3f} logm={m_E['logm']:.3e}",
        flush=True,
    )
    results["E_burnin_bootstrap_lm"] = m_E["hun"]

    # ===== CASE F: BURNIN + BOOTSTRAP_HUN =====
    print("\n=== F. BURNIN_BOOTSTRAP_HUN (burnin → SVD → bootstrap best-by-hun cheat → joint) ===", flush=True)
    cur_F = joint_m_steps(cfg, ds, cands_E[best_hun_E], args.n_joint)
    m_F = summarize_metrics(cfg, ds, cur_F, n_true_states)
    pe_F = float(projector_frobenius_error(cur_F.U, ds.U_half_true, cfg.volume_shape))
    print(
        f"  F final (burnin + best-by-hun + joint): hun={m_F['hun']:.4f} pe={pe_F:.3f} logm={m_F['logm']:.3e}",
        flush=True,
    )
    results["F_burnin_bootstrap_hun"] = m_F["hun"]

    # ===== CASE G: ORACLE =====
    print("\n=== G. ORACLE (mu_true, U_true) → joint loop ===", flush=True)
    init_G = PPCAInit(
        mu=ds.mu_half_true.astype(jnp.complex128),
        U=ds.U_half_true.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in volume_shape),
    )
    cur_G = joint_m_steps(cfg, ds, init_G, args.n_joint)
    m_G = summarize_metrics(cfg, ds, cur_G, n_true_states)
    print(f"  G final: hun={m_G['hun']:.4f} logm={m_G['logm']:.3e}", flush=True)
    results["G_oracle"] = m_G["hun"]

    # ===== SUMMARY =====
    print("\n=== SUMMARY ===", flush=True)
    for k, v in results.items():
        print(f"  {k:30s}: hun={v:.4f}", flush=True)


if __name__ == "__main__":
    main()
