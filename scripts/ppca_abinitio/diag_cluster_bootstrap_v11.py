"""Diagnostic v11: deterministic annealing of noise variance.

V9 + v10 finding: U init from residuals/clusters all produce U with
pe ≈ 3.7 (out of pe_max=4.0). The closed-form M-step from any of these
inits converges to a "lazy basin" with hun ≈ 0.4-0.6, far below the
oracle ceiling (0.95) which is reachable from U_true.

Hypothesis: the M-step landscape has many local minima, and the closed-form
solver lands in a lazy one because the initial responsibilities (from a bad U)
are random. Deterministic annealing softens the responsibilities by inflating
the noise variance, then gradually decreases it. Classic recipe to avoid
local minima in EM clustering.

Schedules tested at burnin mu, with U init = U_random, U_svd_per_image,
U_kmeans_K8:
  S0. no annealing (baseline)
  S1. linear inflation 50 → 1 over 30 iters
  S2. log inflation 100 → 1 over 30 iters
  S3. aggressive 1000 → 1 over 30 iters, then 30 more at 1

Success: any annealed schedule reaches hun > 0.75 from any starting U.

Run:
  cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
  CUDA_VISIBLE_DEVICES=1 RECOVAR_DISABLE_CUDA=1 pixi run python \
      scripts/ppca_abinitio/diag_cluster_bootstrap_v11.py
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


def summarize_metrics(cfg, ds, init, n_states):
    """Always evaluate metrics with the TRUE noise variance, not the inflated one."""
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
    }


def joint_m_steps_annealed(cfg, ds, init, n_states, schedule, U_true_half, tag, report_iters):
    """Run joint loop with per-iteration noise variance scaling.

    `schedule[i]` is the noise inflation factor for iteration i (1-indexed).
    """
    cur = init
    m0 = summarize_metrics(cfg, ds, cur, n_states)
    pe0 = float(projector_frobenius_error(cur.U, U_true_half, cfg.volume_shape))
    print(f"    {tag} k= 0:        hun={m0['hun']:.4f} pe={pe0:.3f}", flush=True)
    for step, factor in enumerate(schedule, start=1):
        nv_scaled = ds.noise_variance_full * float(factor)
        mres = update_mu_residualized(
            cfg,
            cur,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            nv_scaled,
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
            nv_scaled,
        )
        if step in report_iters or step == len(schedule):
            m = summarize_metrics(cfg, ds, cur, n_states)
            pe = float(projector_frobenius_error(cur.U, U_true_half, cfg.volume_shape))
            print(
                f"    {tag} k={step:2d} f={factor:7.2f}: hun={m['hun']:.4f} pe={pe:.3f} logm={m['logm']:.3e}",
                flush=True,
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


def svd_to_U_half(real_data, q, volume_shape):
    rr = real_data - real_data.mean(axis=0, keepdims=True)
    _, _, Vh = np.linalg.svd(rr, full_matrices=False)
    V = Vh[:q].T
    U_half_rows = []
    for k in range(q):
        pc_real = V[:, k].reshape(volume_shape)
        U_half_rows.append(real_volume_to_half(jnp.asarray(pc_real), volume_shape))
    U_half = jnp.stack(U_half_rows).astype(jnp.complex128)
    weights_v = make_half_volume_weights(volume_shape)
    return real_volume_orthonormalize_half(U_half, weights_v, int(np.prod(volume_shape)))


def cluster_mean_svd_U(real_residuals, q, K, volume_shape, seed=0):
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels = km.fit_predict(real_residuals)
    n_voxels = real_residuals.shape[1]
    cluster_means = np.zeros((K, n_voxels), dtype=np.float64)
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            cluster_means[k] = real_residuals[mask].mean(axis=0)
    return svd_to_U_half(cluster_means, q, volume_shape)


def random_ortho_U(q, volume_shape, seed):
    rng = np.random.default_rng(seed)
    half_vol_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    noise = rng.standard_normal((q, 2 * half_vol_size)).view(np.complex128)
    U_half_rows = []
    for k in range(q):
        nj = jnp.asarray(noise[k], dtype=jnp.complex128)
        nj = project_to_real_volume_subspace(nj, volume_shape)
        U_half_rows.append(nj)
    U_half = jnp.stack(U_half_rows)
    weights_v = make_half_volume_weights(volume_shape)
    return real_volume_orthonormalize_half(U_half, weights_v, int(np.prod(volume_shape)))


def make_schedules(n_iters_phase=30, n_iters_post=20):
    """Build a few annealing schedules.

    Each schedule is a list of inflation factors. Length is variable.
    A factor f means: noise_variance is scaled by f for that iteration.
    f=1 means use the true noise. f>1 softens responsibilities.
    """
    return {
        "S0_baseline": [1.0] * n_iters_phase,
        # Linear from 50 to 1 over 30 iters
        "S1_linear_50": list(np.linspace(50.0, 1.0, n_iters_phase)) + [1.0] * n_iters_post,
        # Log from 100 to 1 over 30 iters
        "S2_log_100": list(np.logspace(np.log10(100.0), 0.0, n_iters_phase)) + [1.0] * n_iters_post,
        # Aggressive: 1000 to 1
        "S3_log_1000": list(np.logspace(np.log10(1000.0), 0.0, n_iters_phase)) + [1.0] * n_iters_post,
    }


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
    ap.add_argument("--use-mu-true", action="store_true")
    args = ap.parse_args()

    q = args.q
    print(
        f"### v11 deterministic-annealing q={q} vol={args.vol} n_img={args.n_images} sigma={args.sigma} ###", flush=True
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
    n_states = int(np.asarray(ds.state_coords_true).shape[0])
    weights_v = make_half_volume_weights(volume_shape)
    vshape = tuple(int(x) for x in volume_shape)

    if args.use_mu_true:
        print("\n[using mu_true]", flush=True)
        mu_init = ds.mu_half_true
    else:
        rng = np.random.default_rng(args.seed + 1)
        noise = rng.standard_normal(2 * half_vol_size).view(np.complex128)
        noise_jax = project_to_real_volume_subspace(jnp.asarray(noise, dtype=jnp.complex128), volume_shape)
        mu_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(ds.mu_half_true) * ds.mu_half_true))))
        noise_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(noise_jax) * noise_jax))))
        scale = args.mu_eps * mu_norm / max(noise_norm, 1e-12)
        mu_perturbed = (ds.mu_half_true + scale * noise_jax).astype(jnp.complex128)
        print("\n[computing burnin mu]", flush=True)
        init_zero = PPCAInit(
            mu=mu_perturbed,
            U=jnp.zeros((q, half_vol_size), dtype=jnp.complex128),
            s=s_kernel,
            volume_shape=vshape,
        )
        burnt = homogeneous_burnin(cfg, ds, init_zero, args.n_burnin)
        mu_init = burnt.mu

    print("[computing argmax-pose backprojections]", flush=True)
    r_arg, t_arg = argmax_poses_at_mu(cfg, ds, mu_init, q)
    real_residuals = real_space_residual_backprojections_at_mu(cfg, ds, mu_init, r_arg, t_arg)

    print("\n[building U candidates]", flush=True)
    U_inits = {
        "U_random": random_ortho_U(q, volume_shape, seed=args.seed + 100),
        "U_svd": svd_to_U_half(real_residuals, q, volume_shape),
        "U_kmeans_K8": cluster_mean_svd_U(real_residuals, q, 8, volume_shape, seed=args.seed),
    }

    schedules = make_schedules(n_iters_phase=30, n_iters_post=20)

    results = {}
    for u_tag, U0 in U_inits.items():
        for s_tag, schedule in schedules.items():
            tag = f"{u_tag}+{s_tag}"
            print(f"\n--- {tag} ---", flush=True)
            init = PPCAInit(
                mu=jnp.asarray(mu_init, dtype=jnp.complex128),
                U=jnp.asarray(U0, dtype=jnp.complex128),
                s=s_kernel,
                volume_shape=vshape,
            )
            n_iters = len(schedule)
            report_iters = (1, 5, 15, 30, n_iters)
            cur = joint_m_steps_annealed(cfg, ds, init, n_states, schedule, ds.U_half_true, tag, report_iters)
            m = summarize_metrics(cfg, ds, cur, n_states)
            pe = float(projector_frobenius_error(cur.U, ds.U_half_true, volume_shape))
            print(f"  {tag} FINAL: hun={m['hun']:.4f} pe={pe:.3f} logm={m['logm']:.3e}", flush=True)
            results[tag] = (m["hun"], pe)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  {'method':40s}  {'final_hun':>10s}  {'final_pe':>10s}", flush=True)
    for tag, (hun, pe) in results.items():
        print(f"  {tag:40s}  {hun:>10.4f}  {pe:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
