"""Diagnostic: does the SVD warmstart work if we give it the TRUE pose
per image (instead of hard-argmax from the homogeneous E-step)?

Motivation
----------
The SVD warmstart computes per-image residuals at argmax (rotation,
translation). If argmax is wrong for some images, their residuals are
wrong, and the SVD is biased. At sigma=0.01 argmax should be pretty
accurate, but we never verified this.

If true-pose SVD warmstart reaches a GOOD basin (hun close to oracle)
while argmax SVD does not, the warmstart fix is about pose estimation.
If both land in the same BAD basin, the issue is the residual->basin
mapping itself.

Protocol
--------
For each of the 4 variants below, run K=12 closed-form factor updates
with mu frozen at mu_true and report hun/pe/logm:

1. oracle_U: use ds.U_half_true as U_init (reference good basin)
2. argmax_svd: current run_cryobench.py warmstart path
3. true_pose_svd: same SVD path but with r_true_idx / t_true_idx
4. true_pose_true_state_U: cheating baseline — per-state mean residuals

Run
---
cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
CUDA_VISIBLE_DEVICES=1 RECOVAR_DISABLE_CUDA=1 pixi run python \
    scripts/ppca_abinitio/diag_true_pose_svd.py --q 8
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

import recovar.core.fourier_transform_utils as ftu
from recovar.core.slicing import adjoint_slice_volume, slice_volume
from recovar.em.ppca_abinitio.factor_update import update_factor_closed_form
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    make_half_volume_weights,
    real_volume_orthonormalize_half,
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
        f"  [{label:>20s} k= 0] pe={pe:.4f} hun={m['hun']:.4f} ari={m['ari']:.4f} logm={m['logm']:.3e}",
        flush=True,
    )
    rows = [(0, pe, m["hun"], m["ari"], m["logm"])]
    cur = init
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
            print(
                f"  [{label:>20s} k={k:>2}] pe={pe:.4f} hun={m['hun']:.4f} ari={m['ari']:.4f} logm={m['logm']:.3e}",
                flush=True,
            )
    return rows


def svd_warmstart_from_assignments(cfg, ds, mu_true, r_idx, t_idx, q):
    """Build residual SVD warmstart using a provided (r, t) assignment per image.

    Follows the same residual construction as init_U_from_residual_svd in
    run_cryobench.py, but with the (r, t) choice swapped out. r_idx/t_idx
    are per-image indices into ds.rotations / ds.translations.
    """
    image_shape = cfg.image_shape
    volume_shape = cfg.volume_shape
    shifted_half, ctf2_over_nv_half, _ctf_half = _preprocess_batch_to_half(
        cfg, ds.batch_full, ds.translations, ds.ctf_params, ds.noise_variance_full
    )
    # Per-image rotation/translation lookup
    rot_per = jnp.asarray(np.asarray(ds.rotations)[np.asarray(r_idx)])
    # Slice mu through those rotations
    mu_proj_per = jax.vmap(
        lambda r: slice_volume(
            mu_true, r[None], image_shape, volume_shape, "nearest", half_volume=True, half_image=True
        )
    )(rot_per).reshape(rot_per.shape[0], -1)
    shifted_per = jnp.take_along_axis(
        shifted_half, jnp.asarray(np.asarray(t_idx, dtype=np.int64))[:, None, None], axis=1
    ).squeeze(1)
    residual_per = shifted_per - ctf2_over_nv_half * mu_proj_per
    n_img = residual_per.shape[0]
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
    print(f"  top-{q} singular values: {S_svd[:q]}", flush=True)
    U_init = jnp.asarray(Vh[:q], dtype=jnp.complex128)
    weights_v = jnp.asarray(make_half_volume_weights(volume_shape), dtype=jnp.float64)
    return real_volume_orthonormalize_half(U_init, weights_v, int(np.prod(volume_shape)))


def argmax_poses(cfg, ds, mu_true, q):
    """Return (r_argmax, t_argmax) from homogeneous-E E-step with U=0."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--healpix-order", type=int, default=1)
    args = ap.parse_args()

    print(
        f"### true-pose warmstart q={args.q} vol={args.vol} n_img={args.n_images} sigma={args.sigma} ###",
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

    print("\n=== A. oracle_U init ===", flush=True)
    rows_oracle = evaluate_from_U(cfg, ds, U_true, mu_true, s_kernel, k_steps, label="oracle_U")

    print("\n=== B. argmax_svd (current cryobench path) ===", flush=True)
    t0 = time.perf_counter()
    r_arg, t_arg = argmax_poses(cfg, ds, mu_true, args.q)
    # Diagnostic: what fraction of argmax poses are correct?
    r_true = np.asarray(ds.r_true_idx, dtype=np.int64)
    t_true = np.asarray(ds.t_true_idx, dtype=np.int64)
    print(f"  argmax rot accuracy: {float(np.mean(r_arg == r_true)):.4f}", flush=True)
    print(f"  argmax trans accuracy: {float(np.mean(t_arg == t_true)):.4f}", flush=True)
    U_argmax = svd_warmstart_from_assignments(cfg, ds, mu_true, r_arg, t_arg, args.q)
    print(f"  warmstart wall: {time.perf_counter() - t0:.1f}s", flush=True)
    rows_argmax = evaluate_from_U(cfg, ds, U_argmax, mu_true, s_kernel, k_steps, label="argmax_svd")

    print("\n=== C. true_pose_svd (use r_true, t_true instead of argmax) ===", flush=True)
    t0 = time.perf_counter()
    U_true_pose = svd_warmstart_from_assignments(cfg, ds, mu_true, r_true, t_true, args.q)
    print(f"  warmstart wall: {time.perf_counter() - t0:.1f}s", flush=True)
    rows_true_pose = evaluate_from_U(cfg, ds, U_true_pose, mu_true, s_kernel, k_steps, label="true_pose_svd")

    print("\n=== SUMMARY ===", flush=True)
    rows_all = {
        "oracle_U": rows_oracle,
        "argmax_svd": rows_argmax,
        "true_pose_svd": rows_true_pose,
    }
    for k_idx, k in enumerate(k_steps):
        print(f"\n-- k = {k} --", flush=True)
        for name, rows in rows_all.items():
            r = rows[k_idx]
            print(f"  {name:>16s}: pe={r[1]:.3f} hun={r[2]:.4f} ari={r[3]:.4f} logm={r[4]:.3e}", flush=True)


if __name__ == "__main__":
    main()
