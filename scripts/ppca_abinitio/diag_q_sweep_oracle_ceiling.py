"""Diagnostic: joint-loop oracle ceiling as a function of q on Ribosembly.

Question
--------
Ribosembly has 16 discrete GT volumes. The 16 state means live in an
affine subspace of dimension ≤ 15. A PPCA model with q=4 is
rank-mismatched to the data generating process: no q=4 linear subspace
can represent all 16 states separably. A PPCA model with q=15 matches
the affine dimension of the states exactly — the subspace CAN contain
all 16 states.

The joint-loop oracle ceiling at q tells us the honest reachable
Hungarian when starting from the (μ_true, U_true_topq) under the v0
joint EM loop. If this ceiling rises dramatically between q=4 and
q=15, the "oracle gap" is a rank-matching issue and the fix is
simply to run at higher q. If the ceiling stays low even at q=15,
the issue is somewhere else (metric, warmstart, M-step, etc.).

What this script does
---------------------
For each q in {2, 4, 8, 15}:
  1. Build the Ribosembly synthetic dataset at that q.
  2. Report the top-q PCA clustering accuracy of the 16 state
     centroids (a noiseless, algorithm-free upper bound).
  3. Run `compute_joint_loop_oracle_ceiling(..., n_joint_steps=20)`
     to get the honest reachable Hungarian.

Run
---
cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
CUDA_VISIBLE_DEVICES=3 RECOVAR_DISABLE_CUDA=1 pixi run python \\
    scripts/ppca_abinitio/diag_q_sweep_oracle_ceiling.py
"""

from __future__ import annotations

import time
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.factor_update import update_factor_closed_form
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.mean_update import update_mu_residualized
from recovar.em.ppca_abinitio.posterior import score_and_posterior_moments_eqx
from recovar.em.ppca_abinitio.synthetic import SyntheticFamily, make_synthetic_fixed_grid_dataset
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


def topq_state_separability(gt_vols_real, q):
    """Noiseless upper bound: how well does top-q PCA separate the K state means?

    Returns the Hungarian-matched k-means accuracy of the K state
    centroids projected to the top-q real-space PCA basis.
    """
    K = gt_vols_real.shape[0]
    flat = gt_vols_real.reshape(K, -1)
    mu = flat.mean(axis=0, keepdims=True)
    centered = flat - mu
    _U, _S, Vh = np.linalg.svd(centered, full_matrices=False)
    # State coordinates in the top-q basis
    state_coords = centered @ Vh[:q].T  # (K, q)
    # Clustering on just the K centroids (1 point per class)
    # Hungarian of argmin-nearest-centroid is trivially 1.0 here because
    # each state is its own centroid, so the more interesting metric is:
    # given the K state coordinates as cluster centers, can K-means rediscover
    # the labels on the K points themselves? Yes, always.
    #
    # A meaningful noiseless bound: how much of the total inter-state
    # variance is captured by the top-q PCs?
    total_var = float((centered**2).sum())
    topq_var = float(((centered @ Vh[:q].T) ** 2).sum())
    var_frac = topq_var / max(total_var, 1e-30)
    # And the pairwise-separability: minimum squared distance between
    # pairs of state coords in the top-q basis.
    d2 = ((state_coords[:, None, :] - state_coords[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    min_pair = float(d2.min())
    return var_frac, min_pair, state_coords


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


def joint_loop_from_truth(cfg, ds, q, n_joint=15):
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    cur = PPCAInit(
        mu=ds.mu_half_true.astype(jnp.complex128),
        U=ds.U_half_true.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )
    # Iter 0: (mu_true, U_true)
    out = [("iter=0", summarize_clustering(cfg, ds, cur))]
    for k in range(1, n_joint + 1):
        t0 = time.perf_counter()
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
        dt = time.perf_counter() - t0
        if k in (1, 5, 10, n_joint):
            metrics = summarize_clustering(cfg, ds, cur)
            out.append((f"iter={k} ({dt:.1f}s)", metrics))
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--n-joint", type=int, default=15)
    ap.add_argument("--healpix-order", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--q-list", type=str, default="2,4,8,15")
    args = ap.parse_args()

    q_list = [int(x) for x in args.q_list.split(",")]
    print(f"### diag q sweep oracle ceiling: vol={args.vol} n_img={args.n_images} sigma={args.sigma} ###", flush=True)
    print(f"q values to test: {q_list}", flush=True)

    root = Path("/home/mg6942/mytigress/cryobench2") / "Ribosembly"
    print("loading cryobench gt volumes...", flush=True)
    gt_vols = load_cryobench_gt_volumes(root, target_D=args.vol)
    print(f"  loaded {gt_vols.shape[0]} gt vols at D={args.vol}", flush=True)

    print("\n== noiseless top-q real-space PCA variance captured ==", flush=True)
    for q in q_list:
        var_frac, min_pair, coords = topq_state_separability(gt_vols, q)
        print(f"  q={q:>3}: var_frac={var_frac:.4f}  min_pair_dist²={min_pair:.4g}", flush=True)

    for q in q_list:
        print(f"\n=== q={q} ===", flush=True)
        print("  building synthetic dataset...", flush=True)
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
        print(f"  s_true: {np.asarray(ds.s_true)[: min(q, 6)]}...", flush=True)
        print(f"  running joint loop from (mu_true, U_true_top{q}) for {args.n_joint} iters...", flush=True)
        trajectory = joint_loop_from_truth(cfg, ds, q, n_joint=args.n_joint)
        print(f"  trajectory at q={q}:", flush=True)
        for name, m in trajectory:
            print(f"    {name:<18}: hun={m['hun']:.4f}  ari={m['ari']:.4f}  nmi={m['nmi']:.4f}", flush=True)


if __name__ == "__main__":
    main()
