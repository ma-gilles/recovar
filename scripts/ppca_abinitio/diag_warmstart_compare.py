"""Diagnostic: compare U warmstart strategies on Ribosembly with mu frozen at truth.

Goal
----
Why is the residual-SVD warmstart in run_cryobench.py weak?

We compare three U-warmstart paths at mu = mu_true on the discrete
Ribosembly setup, then run K closed-form factor updates with mu still
frozen at the truth and report centroid accuracy at K = 0, 1, 3, 12.

Strategies under test
---------------------
1. svd_half        : init_U_from_residual_svd from run_cryobench.py
                     (SVD on half-volume rfft residuals, no metric weighting,
                      no real-volume projection before SVD).
2. svd_half_weighted: same residuals, but multiply by sqrt(half-volume Hermitian
                      weights) before SVD so the Frobenius norm in coefficient
                      space matches the real-space ℓ² norm.
3. baseline        : recovar.em.ppca_abinitio.baselines.residual_pca_baseline
                     — full-image residuals, decode to real space, SVD there,
                     re-encode to half-volume.

Reports per strategy
--------------------
- centroid_acc at K=0/1/3/12 closed-form factor updates with mu frozen at mu_true
- projector Frobenius error pe vs U_true
- |R(U_est, U_true)| matrix angles (informal)

Run
---
CUDA_VISIBLE_DEVICES=2 RECOVAR_DISABLE_CUDA=1 pixi run python \\
    scripts/ppca_abinitio/diag_warmstart_compare.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.core.slicing import adjoint_slice_volume, slice_volume
from recovar.em.ppca_abinitio.baselines import residual_pca_baseline
from recovar.em.ppca_abinitio.factor_update import update_factor_closed_form
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    make_half_volume_weights,
    project_to_real_volume_subspace_batch,
    real_volume_orthonormalize_half,
)
from recovar.em.ppca_abinitio.metrics import _orthogonal_procrustes, projector_frobenius_error
from recovar.em.ppca_abinitio.posterior import (
    _preprocess_batch_to_half,
    _slice_mu_half,
    make_half_image_weights,
    score_and_posterior_moments_eqx,
    score_from_half_image_projections,
)
from recovar.em.ppca_abinitio.synthetic import SyntheticFamily, make_synthetic_fixed_grid_dataset
from recovar.em.ppca_abinitio.types import PPCAInit
from recovar.utils.helpers import load_mrc

_S_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# config used by the kernel and M-step
# ---------------------------------------------------------------------------


def _identity_ctf(p, image_shape, voxel_size):
    n = p.shape[0]
    sz = int(np.prod(image_shape))
    return jnp.ones((n, sz), dtype=jnp.float64)


class _Cfg(eqx.Module):
    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    voxel_size: float = eqx.field(static=True)

    def compute_ctf(self, p, *, half_image=False):
        full = _identity_ctf(p, self.image_shape, self.voxel_size)
        if half_image:
            return ftu.full_image_to_half_image(full, self.image_shape)
        return full

    def process_fn(self, b, apply_image_mask=False):
        return b


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# warmstart strategies
# ---------------------------------------------------------------------------


def init_U_svd_half_unweighted(cfg, ds, mu_half, s_kernel, q):
    """Current run_cryobench.py warmstart: SVD on raw half-volume residuals.

    Half-image residual layout, no metric weights, no real-volume projection
    before the SVD. Real-volume projection + orthonormalization after.
    """
    image_shape = cfg.image_shape
    volume_shape = cfg.volume_shape
    weights_h = make_half_image_weights(image_shape)
    mean_proj = _slice_mu_half(mu_half, ds.rotations, image_shape, volume_shape).astype(jnp.complex128)
    n_rot = ds.rotations.shape[0]
    n_half_image = mean_proj.shape[-1]
    u_zero = jnp.zeros((n_rot, q, n_half_image), dtype=jnp.complex128)
    shifted_half, ctf2_over_nv_half, _ = _preprocess_batch_to_half(
        cfg, ds.batch_full, ds.translations, ds.ctf_params, ds.noise_variance_full
    )
    stats = score_from_half_image_projections(mean_proj, u_zero, s_kernel, shifted_half, ctf2_over_nv_half, weights_h)
    log_resp = np.asarray(stats.log_resp)
    n_img = log_resp.shape[0]
    n_trans = log_resp.shape[-1]
    arg = log_resp.reshape(n_img, -1).argmax(axis=-1)
    r_arg = arg // n_trans
    t_arg = arg % n_trans
    rot_per = jnp.asarray(np.asarray(ds.rotations)[r_arg])
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
    print(f"    [svd_half] top-{q} singular values: {S_svd[:q]}", flush=True)
    U_init = jnp.asarray(Vh[:q], dtype=jnp.complex128)
    weights_v = make_half_volume_weights(volume_shape)
    return real_volume_orthonormalize_half(U_init, weights_v, int(np.prod(volume_shape)))


def init_U_svd_half_weighted(cfg, ds, mu_half, s_kernel, q):
    """Same as svd_half but multiply residuals by sqrt(half-volume weights)
    before SVD so that ℓ² in coefficient space equals real-space ℓ²."""
    image_shape = cfg.image_shape
    volume_shape = cfg.volume_shape
    weights_h = make_half_image_weights(image_shape)
    mean_proj = _slice_mu_half(mu_half, ds.rotations, image_shape, volume_shape).astype(jnp.complex128)
    n_rot = ds.rotations.shape[0]
    n_half_image = mean_proj.shape[-1]
    u_zero = jnp.zeros((n_rot, q, n_half_image), dtype=jnp.complex128)
    shifted_half, ctf2_over_nv_half, _ = _preprocess_batch_to_half(
        cfg, ds.batch_full, ds.translations, ds.ctf_params, ds.noise_variance_full
    )
    stats = score_from_half_image_projections(mean_proj, u_zero, s_kernel, shifted_half, ctf2_over_nv_half, weights_h)
    log_resp = np.asarray(stats.log_resp)
    n_img = log_resp.shape[0]
    n_trans = log_resp.shape[-1]
    arg = log_resp.reshape(n_img, -1).argmax(axis=-1)
    r_arg = arg // n_trans
    t_arg = arg % n_trans
    rot_per = jnp.asarray(np.asarray(ds.rotations)[r_arg])
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

    # Project each residual to the real-volume subspace BEFORE SVD so
    # we are operating on real-space-equivalent vectors.
    residual_volumes_jax = jnp.asarray(residual_volumes, dtype=jnp.complex128)
    residual_volumes_jax = project_to_real_volume_subspace_batch(residual_volumes_jax, volume_shape)

    weights_v = np.asarray(make_half_volume_weights(volume_shape), dtype=np.float64)
    sqrt_w = np.sqrt(weights_v)[None, :]  # (1, V_half)
    rv = np.array(np.asarray(residual_volumes_jax), dtype=np.complex128, copy=True)
    rv -= rv.mean(axis=0, keepdims=True)
    rv_w = rv * sqrt_w  # weighted residual matrix
    _, S_svd, Vh_w = np.linalg.svd(rv_w, full_matrices=False)
    print(f"    [svd_half_weighted] top-{q} singular values: {S_svd[:q]}", flush=True)
    # Vh_w is in the weighted basis. To get the basis in the original
    # half-volume layout, divide by sqrt_w (this undoes the weighting).
    Vh = Vh_w / sqrt_w  # (q, V_half)
    U_init = jnp.asarray(Vh[:q], dtype=jnp.complex128)
    return real_volume_orthonormalize_half(U_init, jnp.asarray(weights_v), int(np.prod(volume_shape)))


def init_U_baseline(cfg, ds, mu_half, s_kernel, q):
    """recovar.em.ppca_abinitio.baselines.residual_pca_baseline.

    Full-image residuals, decode to real space, SVD there, re-encode.
    """
    init = residual_pca_baseline(
        cfg,
        mu_half,
        s_floor=_S_FLOOR,
        batch_full=ds.batch_full,
        rotations=ds.rotations,
        translations=ds.translations,
        ctf_params=ds.ctf_params,
        noise_variance_full=ds.noise_variance_full,
        q=q,
    )
    return init.U


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _hungarian_accuracy(labels_true, labels_pred):
    from scipy.optimize import linear_sum_assignment

    n = len(labels_true)
    k = max(int(labels_true.max()), int(labels_pred.max())) + 1
    C = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(labels_true, labels_pred):
        C[int(t), int(p)] += 1
    row_ind, col_ind = linear_sum_assignment(-C)
    return float(C[row_ind, col_ind].sum()) / n


def metrics_for_init(cfg, ds, init):
    """Return (centroid_acc, hungarian_acc, ari) for a PPCA init."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    if ds.state_label_true is None or ds.state_coords_true is None:
        return float("nan"), float("nan"), float("nan")
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
    alpha_true = np.asarray(ds.alpha_true, dtype=np.float64)
    state_coords_true = np.asarray(ds.state_coords_true, dtype=np.float64)
    state_label_true = np.asarray(ds.state_label_true, dtype=np.int64)
    R = _orthogonal_procrustes(alpha_hat, alpha_true)
    aligned = alpha_hat @ R
    d2 = np.sum((aligned[:, None, :] - state_coords_true[None, :, :]) ** 2, axis=-1)
    pred_label = d2.argmin(axis=-1)
    centroid_acc = float(np.mean(pred_label == state_label_true))

    n_states = int(state_coords_true.shape[0])
    km = KMeans(n_clusters=n_states, n_init=10, random_state=0)
    cluster_labels = km.fit_predict(alpha_hat)
    hungarian = _hungarian_accuracy(state_label_true, cluster_labels)
    ari = float(adjusted_rand_score(state_label_true, cluster_labels))
    return centroid_acc, hungarian, ari


def centroid_acc_for_init(cfg, ds, init):
    return metrics_for_init(cfg, ds, init)[0]


def evaluate_warm(cfg, ds, U_warm, mu_true, s_kernel, q, k_steps=(0, 1, 3, 12), label=""):
    """Run K closed-form factor updates with mu frozen at mu_true and report
    centroid_acc / hungarian / ARI / pe at each k in k_steps."""
    init = PPCAInit(
        mu=mu_true.astype(jnp.complex128),
        U=U_warm.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )
    rows = []
    pe = float(projector_frobenius_error(init.U, ds.U_half_true, cfg.volume_shape))
    cac, hun, ari = metrics_for_init(cfg, ds, init)
    rows.append((0, pe, cac, hun, ari))
    print(f"  [{label} k=0 ] pe={pe:.4f} cac={cac:.4f} hun={hun:.4f} ari={ari:.4f}", flush=True)
    cur = init
    target = max(k_steps)
    for k in range(1, target + 1):
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
            cac, hun, ari = metrics_for_init(cfg, ds, cur)
            rows.append((k, pe, cac, hun, ari))
            print(f"  [{label} k={k:>2}] pe={pe:.4f} cac={cac:.4f} hun={hun:.4f} ari={ari:.4f}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Ribosembly")
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--healpix-order", type=int, default=1)
    ap.add_argument(
        "--skip", type=str, default="", help="comma list of strategies to skip: svd_half,svd_half_weighted,baseline"
    )
    args = ap.parse_args()
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    root = Path("/home/mg6942/mytigress/cryobench2") / args.dataset
    print(
        f"### diag warmstart compare dataset={args.dataset} vol={args.vol} "
        f"n_images={args.n_images} sigma={args.sigma} q={args.q} ###",
        flush=True,
    )

    print("loading cryobench gt volumes...", flush=True)
    gt_vols = load_cryobench_gt_volumes(root, target_D=args.vol)

    print("building synthetic dataset...", flush=True)
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
    print(f"  s_true: {np.asarray(ds.s_true)}, n_rot={ds.n_rot}, n_img={ds.n_img}", flush=True)

    cfg = _Cfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    mu_true = ds.mu_half_true

    # Sanity check: oracle init's metrics using mu_true and U_true
    init_oracle = PPCAInit(
        mu=mu_true.astype(jnp.complex128),
        U=ds.U_half_true.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )
    oracle_cac, oracle_hun, oracle_ari = metrics_for_init(cfg, ds, init_oracle)
    print(
        f"\nORACLE (mu_true, U_true): cac={oracle_cac:.4f} hun={oracle_hun:.4f} ari={oracle_ari:.4f}\n",
        flush=True,
    )
    # Post-EM oracle ceiling: 12 closed-form factor updates from (mu_true, U_true)
    cur_o = init_oracle
    for _ in range(12):
        cur_o = update_factor_closed_form(
            cfg, cur_o, ds.batch_full, ds.rotations, ds.translations, ds.ctf_params, ds.noise_variance_full
        )
        cur_o = PPCAInit(mu=mu_true.astype(jnp.complex128), U=cur_o.U, s=cur_o.s, volume_shape=cur_o.volume_shape)
    o12_cac, o12_hun, o12_ari = metrics_for_init(cfg, ds, cur_o)
    print(
        f"POST-EM ORACLE (U_true + 12 factor steps, mu frozen): cac={o12_cac:.4f} hun={o12_hun:.4f} ari={o12_ari:.4f}\n",
        flush=True,
    )
    oracle_acc = oracle_cac

    # Strategy 1: svd_half (current)
    print("=== strategy: svd_half (current run_cryobench.py warmstart) ===", flush=True)
    t0 = time.perf_counter()
    U_svd = init_U_svd_half_unweighted(cfg, ds, mu_true, s_kernel, args.q)
    print(f"  warmstart wall: {time.perf_counter() - t0:.1f}s", flush=True)
    rows_svd = evaluate_warm(cfg, ds, U_svd, mu_true, s_kernel, args.q, label="svd_half")

    # Strategy 2: svd_half_weighted
    print("\n=== strategy: svd_half_weighted (proper real-space metric on half-vol residuals) ===", flush=True)
    t0 = time.perf_counter()
    U_svd_w = init_U_svd_half_weighted(cfg, ds, mu_true, s_kernel, args.q)
    print(f"  warmstart wall: {time.perf_counter() - t0:.1f}s", flush=True)
    rows_svd_w = evaluate_warm(cfg, ds, U_svd_w, mu_true, s_kernel, args.q, label="svd_half_w")

    # Strategy 3: residual_pca_baseline
    print("\n=== strategy: residual_pca_baseline (decode to real, SVD in real space) ===", flush=True)
    t0 = time.perf_counter()
    U_baseline = init_U_baseline(cfg, ds, mu_true, s_kernel, args.q)
    print(f"  warmstart wall: {time.perf_counter() - t0:.1f}s", flush=True)
    rows_baseline = evaluate_warm(cfg, ds, U_baseline, mu_true, s_kernel, args.q, label="baseline")

    # Summary
    print(
        "\n=== SUMMARY (mu frozen at mu_true): cac=biased centroid_acc, hun=Hungarian k-means, ari=ARI ===", flush=True
    )
    print(f"oracle (U_true) biased centroid_acc ceiling = {oracle_acc:.4f}", flush=True)

    print("\n-- biased centroid_acc --", flush=True)
    print(f"{'k':>3}  {'svd_half':>22}  {'svd_half_weighted':>22}  {'baseline':>22}", flush=True)
    for i, k in enumerate([0, 1, 3, 12]):
        cell = lambda rows: f"cac={rows[i][2]:.4f} pe={rows[i][1]:.4f}"
        print(f"{k:>3}  {cell(rows_svd):>22}  {cell(rows_svd_w):>22}  {cell(rows_baseline):>22}", flush=True)

    print("\n-- Hungarian k-means accuracy (honest) --", flush=True)
    print(f"{'k':>3}  {'svd_half':>22}  {'svd_half_weighted':>22}  {'baseline':>22}", flush=True)
    for i, k in enumerate([0, 1, 3, 12]):
        cell = lambda rows: f"hun={rows[i][3]:.4f} ari={rows[i][4]:.4f}"
        print(f"{k:>3}  {cell(rows_svd):>22}  {cell(rows_svd_w):>22}  {cell(rows_baseline):>22}", flush=True)


if __name__ == "__main__":
    main()
