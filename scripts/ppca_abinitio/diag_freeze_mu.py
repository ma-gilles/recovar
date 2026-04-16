"""Diagnostic: at q=8 on Ribosembly, test whether freezing mu after K joint
iters and switching to factor-only updates lands us closer to the
factor-only oracle ceiling (hun~0.995) than the joint-loop ceiling
(hun~0.88).

Motivation
----------
Real q=8 cryobench at iter 100 lands at hun=0.8652. The joint-loop
oracle ceiling is 0.8789. The factor-only oracle ceiling (mu frozen at
truth) is 0.9951. The joint-loop trajectory shows mu drifting AWAY
from truth (fre_truth grows from 0.29 → 0.43) while log_marginal
monotonically grows.

This is textbook EM drift under model misspecification. The fix:
freeze mu after a few joint iters and run factor-only updates.

We test:
  - K=0: no joint iters (mu stays at warmstart, then factor-only)
  - K=1: one joint iter (best mu observed in real run), then factor-only
  - K=2,5,10: more joint iters before freezing
  - K=N: never freeze (current behavior, baseline)

For each K, we report hun at iter K and at iter K+factor_steps.

Run
---
cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
CUDA_VISIBLE_DEVICES=0 RECOVAR_DISABLE_CUDA=1 pixi run python \\
    scripts/ppca_abinitio/diag_freeze_mu.py --q 8 --n-factor 30
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import recovar.core.fourier_transform_utils as ftu
from recovar.core.slicing import adjoint_slice_volume, slice_volume
from recovar.em.ppca_abinitio.factor_update import update_factor_closed_form
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    make_half_volume_weights,
    project_to_real_volume_subspace,
    real_volume_orthonormalize_half,
)
from recovar.em.ppca_abinitio.mean_update import update_mu_residualized
from recovar.em.ppca_abinitio.metrics import projector_frobenius_error
from recovar.em.ppca_abinitio.posterior import (
    _preprocess_batch_to_half,
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


def summarize_clustering(cfg, ds, init):
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
    return {
        "hun": _hungarian(labels_true, cluster_labels),
        "ari": float(adjusted_rand_score(labels_true, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(labels_true, cluster_labels)),
    }


def fre_norm(mu_est, mu_true, weights_v):
    diff = mu_est - mu_true
    num = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(diff) * diff))))
    den = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(mu_true) * mu_true))))
    return num / max(den, 1e-30)


def joint_step(cfg, ds, cur):
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


def factor_step(cfg, ds, cur):
    mu_frozen = cur.mu
    cur = update_factor_closed_form(
        cfg,
        cur,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
    )
    return PPCAInit(mu=mu_frozen, U=cur.U, s=cur.s, volume_shape=cur.volume_shape)


def run_freeze_protocol(cfg, ds, mu_init, U_init, n_joint_before_freeze, n_factor_after_freeze, weights_v):
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    cur = PPCAInit(
        mu=mu_init.astype(jnp.complex128),
        U=U_init.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )
    metrics0 = summarize_clustering(cfg, ds, cur)
    fre0 = fre_norm(cur.mu, ds.mu_half_true, weights_v)
    pe0 = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
    print(f"  init       : hun={metrics0['hun']:.4f} fre={fre0:.4f} pe={pe0:.4f}", flush=True)

    for k in range(1, n_joint_before_freeze + 1):
        t0 = time.perf_counter()
        cur = joint_step(cfg, ds, cur)
        jax.block_until_ready(cur.U)
        if k == n_joint_before_freeze or k <= 3:
            m = summarize_clustering(cfg, ds, cur)
            fre = fre_norm(cur.mu, ds.mu_half_true, weights_v)
            pe = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
            print(
                f"  joint  {k:>2}  : hun={m['hun']:.4f} fre={fre:.4f} pe={pe:.4f}  ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    if n_joint_before_freeze > 0:
        m = summarize_clustering(cfg, ds, cur)
        fre = fre_norm(cur.mu, ds.mu_half_true, weights_v)
        pe = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
        print(f"  ===> freezing mu at fre={fre:.4f} hun={m['hun']:.4f} pe={pe:.4f}", flush=True)

    for k in range(1, n_factor_after_freeze + 1):
        t0 = time.perf_counter()
        cur = factor_step(cfg, ds, cur)
        jax.block_until_ready(cur.U)
        if k <= 3 or k % 5 == 0 or k == n_factor_after_freeze:
            m = summarize_clustering(cfg, ds, cur)
            pe = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
            print(
                f"  factor {k:>2}  : hun={m['hun']:.4f} pe={pe:.4f}  ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    final_metrics = summarize_clustering(cfg, ds, cur)
    return cur, final_metrics


def compute_svd_warmstart(cfg, ds, mu_half, q):
    """Replicates init_U_from_residual_svd unweighted variant."""
    image_shape = cfg.image_shape
    volume_shape = cfg.volume_shape
    weights_h = make_half_image_weights(image_shape)
    rotations = ds.rotations
    n_rot = rotations.shape[0]
    mean_proj = (
        jax.vmap(
            lambda r: slice_volume(
                mu_half, r[None], image_shape, volume_shape, "nearest", half_volume=True, half_image=True
            )
        )(rotations)
        .reshape(n_rot, -1)
        .astype(jnp.complex128)
    )
    u_zero = jnp.zeros((n_rot, q, mean_proj.shape[-1]), dtype=jnp.complex128)
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    shifted_half, ctf2_over_nv_half, _ctf_half = _preprocess_batch_to_half(
        cfg, ds.batch_full, ds.translations, ds.ctf_params, ds.noise_variance_full
    )
    stats = score_from_half_image_projections(mean_proj, u_zero, s_kernel, shifted_half, ctf2_over_nv_half, weights_h)
    log_resp = np.asarray(stats.log_resp)
    n_img = log_resp.shape[0]
    n_trans = log_resp.shape[-1]
    arg = log_resp.reshape(n_img, -1).argmax(axis=-1)
    r_arg = arg // n_trans
    t_arg = arg % n_trans
    rot_per = jnp.asarray(np.asarray(rotations)[r_arg])
    mu_proj_per = jax.vmap(
        lambda r: slice_volume(
            mu_half, r[None], image_shape, volume_shape, "nearest", half_volume=True, half_image=True
        )
    )(rot_per).reshape(n_img, -1)
    shifted_per = jnp.take_along_axis(shifted_half, jnp.asarray(t_arg)[:, None, None], axis=1).squeeze(1)
    residual_per = shifted_per - ctf2_over_nv_half * mu_proj_per
    bp = []
    for i in range(n_img):
        b = adjoint_slice_volume(
            residual_per[i : i + 1],
            rot_per[i : i + 1],
            image_shape,
            volume_shape,
            "nearest",
            half_image=True,
            half_volume=True,
        )
        bp.append(np.asarray(b).reshape(-1))
    residual_volumes = np.stack(bp)
    residual_volumes -= residual_volumes.mean(axis=0, keepdims=True)
    _, S_svd, Vh = np.linalg.svd(residual_volumes, full_matrices=False)
    print(f"  residual SVD top-{q} singular values: {S_svd[:q]}", flush=True)
    U_init = jnp.asarray(Vh[:q], dtype=jnp.complex128)
    weights_v = jnp.asarray(make_half_volume_weights(volume_shape), dtype=jnp.float64)
    return real_volume_orthonormalize_half(U_init, weights_v, int(np.prod(volume_shape)))


def make_perturbed_mu(mu_true, volume_shape, weights_v, seed, eps=0.5):
    half_size = mu_true.shape[0]
    rng = np.random.default_rng(seed + 1)
    noise = rng.standard_normal(2 * half_size).view(np.complex128)
    noise_jax = jnp.asarray(noise, dtype=jnp.complex128)
    noise_jax = project_to_real_volume_subspace(noise_jax, volume_shape)
    mu_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(mu_true) * mu_true))))
    noise_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(noise_jax) * noise_jax))))
    scale = eps * mu_norm / max(noise_norm, 1e-12)
    return (mu_true + scale * noise_jax).astype(jnp.complex128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--n-factor", type=int, default=30)
    ap.add_argument("--mu-perturb-eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--healpix-order", type=int, default=1)
    ap.add_argument(
        "--k-list",
        type=str,
        default="0,1,2,5",
        help="Comma list of K = number of joint iters before freezing mu",
    )
    args = ap.parse_args()

    print(f"### freeze-mu test q={args.q} vol={args.vol} n_img={args.n_images} sigma={args.sigma} ###", flush=True)

    root = Path("/home/mg6942/mytigress/cryobench2") / "Ribosembly"
    print("loading gt volumes...", flush=True)
    gt_vols = load_cryobench_gt_volumes(root, target_D=args.vol)
    print(f"  loaded {gt_vols.shape[0]} gt vols at D={args.vol}", flush=True)

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
    weights_v = jnp.asarray(make_half_volume_weights(volume_shape), dtype=jnp.float64)

    mu_true = ds.mu_half_true.astype(jnp.complex128)
    mu_pert = make_perturbed_mu(mu_true, volume_shape, weights_v, args.init_seed, eps=args.mu_perturb_eps)

    print("computing svd warmstart from mu_perturbed (one-time)...", flush=True)
    U_svd = compute_svd_warmstart(cfg, ds, mu_pert, args.q)

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    summary = []

    for K in k_list:
        n_factor = max(args.n_factor - K, 1)
        print(f"\n=== freeze protocol K={K} (joint {K} iters then factor-only {n_factor} iters) ===", flush=True)
        _cur, m = run_freeze_protocol(
            cfg,
            ds,
            mu_init=mu_pert,
            U_init=U_svd,
            n_joint_before_freeze=K,
            n_factor_after_freeze=n_factor,
            weights_v=weights_v,
        )
        summary.append((K, m))

    print("\n=== SUMMARY (mu_pert + U_svd warmstart) ===", flush=True)
    print(f"{'K':>5}  {'final hun':>12}  {'final ari':>12}  {'final nmi':>12}", flush=True)
    for K, m in summary:
        print(f"{K:>5}  {m['hun']:>12.4f}  {m['ari']:>12.4f}  {m['nmi']:>12.4f}", flush=True)


if __name__ == "__main__":
    main()
