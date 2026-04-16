"""Diagnostic: can random-U multi-restart escape the bad warmstart basin?

Motivation
----------
At q=8 on Ribosembly, the closed-form factor M-step with mu frozen at
mu_true hits a plateau at hun ~0.55 from the SVD warmstart, while the
factor-only oracle (U_true init) reaches hun=0.9951. The SVD warmstart
subspace has pe ~3.76 out of pe_max=sqrt(2q)=4.0, i.e. nearly orthogonal
to U_true's subspace.

Question: is the factor M-step multi-basin? Can any random U_init find
the oracle basin, or does every random start get stuck in the same
near-orthogonal plateau as the SVD warmstart?

Protocol
--------
For each seed in [0..N-1]:
    1. Sample q random real volumes, FFT to half-volume layout
    2. real_volume_orthonormalize_half -> random orthonormal U_init
    3. Run K closed-form factor updates with mu frozen at mu_true
    4. Record hun / pe / log-marginal at K = 0, 1, 3, 12

Report
------
- per-seed trajectory
- distribution of final hun over seeds
- best-seed vs oracle ceiling vs SVD warmstart baseline

Run
---
cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
CUDA_VISIBLE_DEVICES=2 RECOVAR_DISABLE_CUDA=1 pixi run python \
    scripts/ppca_abinitio/diag_random_U.py --q 8 --n-restarts 8 --k-steps 12
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
    log_scores = np.asarray(stats.log_scores)
    per_img_log_marg = np.asarray(jax.scipy.special.logsumexp(log_scores.reshape(log_scores.shape[0], -1), axis=-1))
    log_marginal = float(per_img_log_marg.sum())
    return {
        "hun": _hungarian(labels_true, cluster_labels),
        "ari": float(adjusted_rand_score(labels_true, cluster_labels)),
        "log_marginal": log_marginal,
    }


def random_orthonormal_U(q, volume_shape, weights_v, volume_size, seed):
    """Sample random U orthonormal in the real-volume half-spectrum metric.

    Procedure: draw q real-valued random 3D volumes from N(0,1), encode
    to half-volume, orthonormalize via real_volume_orthonormalize_half.
    """
    rng = np.random.default_rng(seed)
    N = int(volume_shape[0])
    rows = []
    for _ in range(q):
        vol_real = rng.standard_normal((N, N, N)).astype(np.float64)
        v_half = real_volume_to_half(jnp.asarray(vol_real), tuple(int(x) for x in volume_shape))
        rows.append(v_half)
    U_half = jnp.stack(rows, axis=0).astype(jnp.complex128)
    return real_volume_orthonormalize_half(U_half, weights_v, volume_size)


def compute_svd_warmstart(cfg, ds, mu_half, q):
    """Same as init_U_from_residual_svd unweighted (run_cryobench.py path)."""
    image_shape = cfg.image_shape
    volume_shape = cfg.volume_shape
    weights_h = make_half_image_weights(image_shape)
    rotations = ds.rotations
    n_rot = rotations.shape[0]
    mean_proj = _slice_mu_half(mu_half, rotations, image_shape, volume_shape).astype(jnp.complex128)
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
    U_init = jnp.asarray(Vh[:q], dtype=jnp.complex128)
    weights_v = jnp.asarray(make_half_volume_weights(volume_shape), dtype=jnp.float64)
    return real_volume_orthonormalize_half(U_init, weights_v, int(np.prod(volume_shape)))


def evaluate_from_U(cfg, ds, U_init, mu_true, s_kernel, k_steps, label=""):
    """Run factor-only updates from U_init with mu frozen, report metrics at k_steps."""
    init = PPCAInit(
        mu=mu_true.astype(jnp.complex128),
        U=U_init.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )
    rows = []
    pe = float(projector_frobenius_error(init.U, ds.U_half_true, cfg.volume_shape))
    m = summarize_clustering(cfg, ds, init)
    rows.append((0, pe, m["hun"], m["ari"], m["log_marginal"]))
    print(
        f"  [{label:>12s} k= 0] pe={pe:.4f} hun={m['hun']:.4f} ari={m['ari']:.4f} logm={m['log_marginal']:.3e}",
        flush=True,
    )
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
            m = summarize_clustering(cfg, ds, cur)
            rows.append((k, pe, m["hun"], m["ari"], m["log_marginal"]))
            print(
                f"  [{label:>12s} k={k:>2}] pe={pe:.4f} hun={m['hun']:.4f} ari={m['ari']:.4f} "
                f"logm={m['log_marginal']:.3e}",
                flush=True,
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--n-restarts", type=int, default=8)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--healpix-order", type=int, default=1)
    args = ap.parse_args()

    print(
        f"### random-U multi-restart q={args.q} vol={args.vol} n_img={args.n_images} "
        f"sigma={args.sigma} n_restarts={args.n_restarts} ###",
        flush=True,
    )

    root = Path("/home/mg6942/mytigress/cryobench2") / "Ribosembly"
    print("loading gt volumes...", flush=True)
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
    weights_v = jnp.asarray(make_half_volume_weights(volume_shape), dtype=jnp.float64)
    volume_size = int(np.prod(volume_shape))

    mu_true = ds.mu_half_true.astype(jnp.complex128)
    U_true = ds.U_half_true.astype(jnp.complex128)
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)

    k_steps = sorted(set(k for k in (0, 1, 3, args.k_max) if k <= args.k_max))

    # ---- Reference: oracle ceiling (U_true init) ----
    print("\n=== A. oracle U_true init (factor-only ceiling) ===", flush=True)
    t0 = time.perf_counter()
    rows_oracle = evaluate_from_U(cfg, ds, U_true, mu_true, s_kernel, k_steps, label="oracle")
    print(f"  wall: {time.perf_counter() - t0:.1f}s", flush=True)

    # ---- Reference: SVD warmstart (current cryobench path) ----
    print("\n=== B. svd warmstart from mu_true ===", flush=True)
    t0 = time.perf_counter()
    U_svd = compute_svd_warmstart(cfg, ds, mu_true, args.q)
    print(f"  warmstart wall: {time.perf_counter() - t0:.1f}s", flush=True)
    rows_svd = evaluate_from_U(cfg, ds, U_svd, mu_true, s_kernel, k_steps, label="svd")

    # ---- Random restarts ----
    print(f"\n=== C. random U restarts (n={args.n_restarts}) ===", flush=True)
    all_rows = []
    for r in range(args.n_restarts):
        print(f"\n-- restart {r} (seed={args.seed * 1000 + r}) --", flush=True)
        U_rand = random_orthonormal_U(args.q, volume_shape, weights_v, volume_size, seed=args.seed * 1000 + r)
        pe_init = float(projector_frobenius_error(U_rand, U_true, cfg.volume_shape))
        print(f"  initial pe vs U_true: {pe_init:.4f}", flush=True)
        rows = evaluate_from_U(cfg, ds, U_rand, mu_true, s_kernel, k_steps, label=f"rand{r:02d}")
        all_rows.append(rows)

    # ---- Summary ----
    def cell(rows, idx, fmt):
        return fmt.format(rows[idx])

    print("\n=== SUMMARY (mu frozen at mu_true, k factor M-step iters) ===", flush=True)
    print("oracle and svd are reference rows; rand00..rand{N-1} are random restarts.", flush=True)
    header = f"{'method':>10s} " + " ".join(f"{'k=' + str(k):>20s}" for k in k_steps)
    print(header, flush=True)

    def fmt_row(name, rows):
        def cell(k_idx):
            k, pe, hun, ari, logm = rows[k_idx]
            return f"hun={hun:.3f} pe={pe:.2f}"

        s = f"{name:>10s} " + " ".join(f"{cell(i):>20s}" for i in range(len(k_steps)))
        print(s, flush=True)

    fmt_row("oracle", rows_oracle)
    fmt_row("svd", rows_svd)
    for r in range(args.n_restarts):
        fmt_row(f"rand{r:02d}", all_rows[r])

    # Final hun distribution
    final_huns = [rows[-1][2] for rows in all_rows]
    svd_final_hun = rows_svd[-1][2]
    oracle_final_hun = rows_oracle[-1][2]
    print("\n-- final hun distribution over random restarts --", flush=True)
    print(
        f"  oracle_final={oracle_final_hun:.4f}  svd_final={svd_final_hun:.4f}  "
        f"rand_min={min(final_huns):.4f}  rand_max={max(final_huns):.4f}  "
        f"rand_median={np.median(final_huns):.4f}  rand_mean={np.mean(final_huns):.4f}",
        flush=True,
    )

    final_logms = [rows[-1][4] for rows in all_rows]
    svd_final_logm = rows_svd[-1][4]
    oracle_final_logm = rows_oracle[-1][4]
    print("\n-- final log_marginal distribution over random restarts --", flush=True)
    print(
        f"  oracle_final={oracle_final_logm:.3e}  svd_final={svd_final_logm:.3e}  "
        f"rand_min={min(final_logms):.3e}  rand_max={max(final_logms):.3e}",
        flush=True,
    )
    best_rand_idx = int(np.argmax(final_logms))
    print(
        f"\n  best-by-logm random restart: rand{best_rand_idx:02d} "
        f"(logm={final_logms[best_rand_idx]:.3e}, hun={final_huns[best_rand_idx]:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
