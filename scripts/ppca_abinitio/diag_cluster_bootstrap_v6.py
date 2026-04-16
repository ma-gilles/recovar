"""Diagnostic v6: validate multi-seed best-by-logm selection.

v5 breakthrough: 16-seed kmeans ensemble + best-by-logm → hun=0.823
(seed=14, within 0.07 of the 0.895 ceiling).

Open questions:
  1. Is best-by-logm a reliable selector? (logm gap between good/bad
     basins is only 0.15%, so noise can flip the ordering.)
  2. How does hun correlate with logm across seeds? Should we see
     positive correlation?
  3. Does enlarging to 32 / 64 seeds reliably hit 0.80+?
  4. Is there a better selection criterion than logm?
  5. Does the best kmeans seed change if we re-run with different
     n_init values?

v6 runs 32 seeds, prints (hun_label, hun_ppca_k0, hun_ppca_k12, logm)
for every seed, and reports:
  - best-by-logm, best-by-hun_k12 (cheating), best-by-pe (cheating)
  - the logm distribution
  - median/mean/best/worst hun

Run:
  cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
  CUDA_VISIBLE_DEVICES=0 RECOVAR_DISABLE_CUDA=1 pixi run python \
      scripts/ppca_abinitio/diag_cluster_bootstrap_v6.py
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
    ap.add_argument("--n-seeds", type=int, default=32)
    args = ap.parse_args()

    q = args.q
    print(
        f"### cluster-bootstrap-v6 q={q} vol={args.vol} n_img={args.n_images} sigma={args.sigma} n_seeds={args.n_seeds} ###",
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

    mu_true = ds.mu_half_true.astype(jnp.complex128)
    U_true = ds.U_half_true.astype(jnp.complex128)
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    labels_true = np.asarray(ds.state_label_true, dtype=np.int64)
    n_true_states = int(np.asarray(ds.state_coords_true).shape[0])

    # ---- Argmax poses + real backprojections ----
    r_arg, t_arg = argmax_poses(cfg, ds, mu_true, q)
    print("\n[building per-image real-space backprojections from argmax poses]", flush=True)
    bp_argmax = real_space_residual_backprojections(cfg, ds, mu_true, r_arg, t_arg)

    # ---- Reference: oracle_U ----
    print("\n=== Reference ===", flush=True)
    init_or = PPCAInit(mu=mu_true, U=U_true, s=s_kernel, volume_shape=tuple(int(x) for x in cfg.volume_shape))
    cur_or = run_m_steps(cfg, ds, init_or, mu_true, 12)
    m_or = summarize_metrics(cfg, ds, cur_or, n_true_states)
    pe_or = float(projector_frobenius_error(cur_or.U, ds.U_half_true, cfg.volume_shape))
    print(
        f"  oracle_U k=12 hun={m_or['hun']:.4f} pe={pe_or:.3f} logm={m_or['logm']:.3e}",
        flush=True,
    )

    # ---- Phase1 q=8 ----
    print("\n=== Phase1 q=8 SVD warmstart → 12 M-steps ===", flush=True)
    U_svd0, _ = svd_to_U_half(bp_argmax.copy(), q, volume_shape)
    init_p1 = PPCAInit(mu=mu_true, U=U_svd0, s=s_kernel, volume_shape=tuple(int(x) for x in cfg.volume_shape))
    phase1 = run_m_steps(cfg, ds, init_p1, mu_true, 12)
    m_p1 = summarize_metrics(cfg, ds, phase1, n_true_states)
    alpha_p1 = m_p1["alpha_hat"]
    pe_p1 = float(projector_frobenius_error(phase1.U, ds.U_half_true, cfg.volume_shape))
    print(
        f"  phase1 k=12 hun={m_p1['hun']:.4f} pe={pe_p1:.3f} logm={m_p1['logm']:.3e}",
        flush=True,
    )

    # ---- Multi-seed sweep ----
    print(f"\n=== {args.n_seeds}-seed kmeans sweep (K=16, n_init=20) on phase1 alpha ===", flush=True)
    results = []
    for seed in range(args.n_seeds):
        km = KMeans(n_clusters=16, n_init=20, random_state=seed)
        labels_pred = km.fit_predict(alpha_p1)
        hun_lab = _hungarian(labels_true, labels_pred)
        U_init = cluster_mean_SVD_from_labels(bp_argmax, labels_pred, q, volume_shape)
        init = PPCAInit(mu=mu_true, U=U_init, s=s_kernel, volume_shape=tuple(int(x) for x in cfg.volume_shape))
        cur = run_m_steps(cfg, ds, init, mu_true, 12)
        m = summarize_metrics(cfg, ds, cur, n_true_states)
        pe = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
        results.append(
            {
                "seed": seed,
                "hun_lab": hun_lab,
                "hun_k12": m["hun"],
                "logm": m["logm"],
                "pe": pe,
            }
        )
        print(
            f"  seed={seed:2d}: label_hun={hun_lab:.4f}  ppca_hun={m['hun']:.4f}  pe={pe:.3f}  logm={m['logm']:.6e}",
            flush=True,
        )

    # ---- Analysis ----
    print("\n=== Analysis ===", flush=True)
    hun_arr = np.array([r["hun_k12"] for r in results])
    logm_arr = np.array([r["logm"] for r in results])
    pe_arr = np.array([r["pe"] for r in results])
    label_hun_arr = np.array([r["hun_lab"] for r in results])

    print(
        f"  hun_k12 distribution: min={hun_arr.min():.4f}  max={hun_arr.max():.4f}  "
        f"mean={hun_arr.mean():.4f}  median={np.median(hun_arr):.4f}",
        flush=True,
    )
    print(
        f"  logm distribution:    min={logm_arr.min():.6e}  max={logm_arr.max():.6e}  "
        f"rel_range={(logm_arr.max() - logm_arr.min()) / abs(logm_arr.mean()):.6e}",
        flush=True,
    )
    print(
        f"  pe distribution:      min={pe_arr.min():.3f}   max={pe_arr.max():.3f}",
        flush=True,
    )

    best_by_logm = int(np.argmax(logm_arr))
    best_by_hun = int(np.argmax(hun_arr))
    best_by_pe = int(np.argmin(pe_arr))

    print(
        f"  best-by-logm:  seed={best_by_logm:2d}  hun={hun_arr[best_by_logm]:.4f}  "
        f"logm={logm_arr[best_by_logm]:.6e}  pe={pe_arr[best_by_logm]:.3f}",
        flush=True,
    )
    print(
        f"  best-by-hun:   seed={best_by_hun:2d}  hun={hun_arr[best_by_hun]:.4f}  "
        f"logm={logm_arr[best_by_hun]:.6e}  pe={pe_arr[best_by_hun]:.3f}",
        flush=True,
    )
    print(
        f"  best-by-pe:    seed={best_by_pe:2d}  hun={hun_arr[best_by_pe]:.4f}  "
        f"logm={logm_arr[best_by_pe]:.6e}  pe={pe_arr[best_by_pe]:.3f}",
        flush=True,
    )

    # ---- Correlations ----
    # Sort by logm descending
    order = np.argsort(-logm_arr)
    top5_by_logm = order[:5]
    print(
        f"\n  top-5 seeds by logm: seeds={[int(o) for o in top5_by_logm]}  "
        f"hun_values={hun_arr[top5_by_logm].round(4).tolist()}",
        flush=True,
    )
    logm_hun_corr = float(np.corrcoef(logm_arr, hun_arr)[0, 1])
    print(f"  Pearson corr(logm, hun) = {logm_hun_corr:.4f}", flush=True)
    pe_hun_corr = float(np.corrcoef(pe_arr, hun_arr)[0, 1])
    print(f"  Pearson corr(pe,   hun) = {pe_hun_corr:.4f}", flush=True)
    lab_hun_corr = float(np.corrcoef(label_hun_arr, hun_arr)[0, 1])
    print(f"  Pearson corr(label_hun, hun) = {lab_hun_corr:.4f}", flush=True)


if __name__ == "__main__":
    main()
