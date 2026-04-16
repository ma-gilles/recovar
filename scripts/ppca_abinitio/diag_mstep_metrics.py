"""Diagnostic with BETTER METRICS.

Why new metrics
---------------
The existing `pe` and `centroid_acc` metrics compare against `U_half_true`,
which is the top-q real-space PCA of the discrete conformations. For a
**misspecified** PPCA model on 16 discrete ribosome states, `U_half_true`
is *not* the marginal-likelihood maximizer. So:

  - `pe ≠ 0` at the ML optimum is expected and fine.
  - `centroid_acc` uses Procrustes alignment to U_true's basis, which
    biases the metric toward "U is close to top-q PCA" rather than
    "U gives good clustering".

Better metrics
--------------
1. **log_marginal**: sum over images of `logsumexp log_scores[i]`. This
   is the EM objective (up to a U-independent constant) — it should
   monotonically increase under iterated EM. Mode A vs Mode B can be
   directly compared; whichever has higher log_marginal is the EM
   "winner".

2. **ARI** (Adjusted Rand Index): k-means cluster the per-image
   `alpha_hat = Σ_{g,t} γ[i,g,t] m[i,g,t]` with K=n_states, compare to
   true state labels. Invariant to label permutations and basis
   rotations. `1 = perfect`, `0 = random`.

3. **NMI** (Normalized Mutual Information): another clustering metric,
   gives a complementary view.

4. **clust_acc**: Hungarian-matched cluster accuracy. The best
   permutation of cluster labels is used to map to true labels, then
   classification accuracy is reported. Invariant to label permutations.

5. **silhouette**: per-point cluster-quality score using the true
   labels as "ground-truth clusters". Measures how well-separated the
   latent embeddings are by true-state identity.

6. **pe**, **centroid_acc**: kept for comparability with previous runs.

Run
---
CUDA_VISIBLE_DEVICES=2 RECOVAR_DISABLE_CUDA=1 pixi run python \\
    scripts/ppca_abinitio/diag_mstep_metrics.py --qs 2,4,8 --K 12
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.baselines import residual_pca_baseline
from recovar.em.ppca_abinitio.factor_update import update_factor_closed_form
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    make_half_volume_weights,
)
from recovar.em.ppca_abinitio.metrics import _orthogonal_procrustes, projector_frobenius_error
from recovar.em.ppca_abinitio.posterior import score_and_posterior_moments_eqx
from recovar.em.ppca_abinitio.synthetic import SyntheticFamily, make_synthetic_fixed_grid_dataset
from recovar.em.ppca_abinitio.types import PPCAInit
from recovar.utils.helpers import load_mrc

_S_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Forward-model placeholder
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
# Data loading
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
# Metrics
# ---------------------------------------------------------------------------


def clustering_accuracy_hungarian(labels_true, labels_pred):
    """Best-permutation classification accuracy for a clustering.

    Builds a (K_true, K_pred) count matrix, runs the Hungarian
    algorithm to maximize the diagonal, and returns the sum of the
    matched counts divided by n. Invariant to label permutations.
    """
    n = len(labels_true)
    k = max(int(labels_true.max()), int(labels_pred.max())) + 1
    C = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(labels_true, labels_pred):
        C[int(t), int(p)] += 1
    # maximize -> minimize negative
    row_ind, col_ind = linear_sum_assignment(-C)
    return float(C[row_ind, col_ind].sum()) / n


def compute_metrics(cfg, ds, init):
    """Return a dict of metrics on `init`.

    Metrics:
      pe                 - projector Frobenius error vs U_true
      centroid_acc       - original Procrustes-aligned nearest-centroid
      log_marginal       - Σ_i logsumexp log_scores[i] (EM objective)
      ARI                - adjusted Rand index on k-means(alpha_hat)
      NMI                - normalized mutual information
      clust_acc          - Hungarian-matched cluster accuracy
      silhouette_true    - silhouette with true labels (higher = better)
    """
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
    pm = np.asarray(stats.post_mean)
    gamma = np.exp(np.asarray(stats.log_resp))
    alpha_hat = np.sum(gamma[..., None] * pm, axis=(1, 2))  # (n_img, q)

    # Log marginal likelihood (sum over images, up to U-independent constants)
    n_img = log_scores.shape[0]
    log_scores_flat = log_scores.reshape(n_img, -1)
    # Numerical-stable logsumexp
    max_per_img = log_scores_flat.max(axis=-1, keepdims=True)
    log_marginal = float(np.sum(max_per_img.squeeze(-1) + np.log(np.exp(log_scores_flat - max_per_img).sum(axis=-1))))

    # Projector error (vs U_half_true, gauge-invariant)
    pe = float(projector_frobenius_error(init.U, ds.U_half_true, cfg.volume_shape))

    # Original centroid_acc (for comparison)
    alpha_true = np.asarray(ds.alpha_true, dtype=np.float64)
    state_coords_true = np.asarray(ds.state_coords_true, dtype=np.float64)
    state_label_true = np.asarray(ds.state_label_true, dtype=np.int64)
    R = _orthogonal_procrustes(alpha_hat, alpha_true)
    aligned = alpha_hat @ R
    d2 = np.sum((aligned[:, None, :] - state_coords_true[None, :, :]) ** 2, axis=-1)
    centroid_pred = d2.argmin(axis=-1)
    centroid_acc = float(np.mean(centroid_pred == state_label_true))

    # k-means clustering with K = n_states
    n_states = state_coords_true.shape[0]
    km = KMeans(n_clusters=n_states, n_init=10, random_state=0)
    cluster_labels = km.fit_predict(alpha_hat)

    ari = float(adjusted_rand_score(state_label_true, cluster_labels))
    nmi = float(normalized_mutual_info_score(state_label_true, cluster_labels))
    clust_acc = float(clustering_accuracy_hungarian(state_label_true, cluster_labels))

    # Silhouette with true labels (requires at least 2 unique labels and
    # samples per cluster)
    unique_true = np.unique(state_label_true)
    if len(unique_true) > 1 and alpha_hat.shape[0] > len(unique_true):
        try:
            sil = float(silhouette_score(alpha_hat, state_label_true))
        except Exception:
            sil = float("nan")
    else:
        sil = float("nan")

    return {
        "pe": pe,
        "centroid_acc": centroid_acc,
        "log_marginal": log_marginal,
        "ARI": ari,
        "NMI": nmi,
        "clust_acc": clust_acc,
        "sil_true": sil,
    }


def fmt_metrics(m):
    return (
        f"pe={m['pe']:.3f} cac={m['centroid_acc']:.3f} "
        f"ARI={m['ARI']:.3f} NMI={m['NMI']:.3f} "
        f"cacc={m['clust_acc']:.3f} sil={m['sil_true']:.3f} "
        f"LL={m['log_marginal']:.1f}"
    )


# ---------------------------------------------------------------------------
# Warmstarts
# ---------------------------------------------------------------------------


def warmstart_baseline(cfg, ds, mu_true, q):
    return residual_pca_baseline(
        cfg,
        mu_true,
        s_floor=_S_FLOOR,
        batch_full=ds.batch_full,
        rotations=ds.rotations,
        translations=ds.translations,
        ctf_params=ds.ctf_params,
        noise_variance_full=ds.noise_variance_full,
        q=q,
    ).U


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_factor_steps(cfg, ds, init, K, label):
    cur = init
    m0 = compute_metrics(cfg, ds, cur)
    print(f"  [{label} k={0:>2}] {fmt_metrics(m0)}", flush=True)
    rows = [(0, m0)]
    for k in range(1, K + 1):
        cur = update_factor_closed_form(
            cfg,
            cur,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
        )
        cur = PPCAInit(mu=init.mu, U=cur.U, s=cur.s, volume_shape=cur.volume_shape)
        if k in (1, 3, 6, 12):
            m_k = compute_metrics(cfg, ds, cur)
            rows.append((k, m_k))
            print(f"  [{label} k={k:>2}] {fmt_metrics(m_k)}", flush=True)
    return cur, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Ribosembly")
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--healpix-order", type=int, default=1)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--qs", type=str, default="2,4", help="comma list of q values to try")
    args = ap.parse_args()

    qs = [int(x) for x in args.qs.split(",")]
    root = Path("/home/mg6942/mytigress/cryobench2") / args.dataset

    print(f"### diag_mstep_metrics dataset={args.dataset} vol={args.vol} n_images={args.n_images} ###")
    print("loading cryobench gt volumes...", flush=True)
    gt_vols = load_cryobench_gt_volumes(root, target_D=args.vol)

    image_shape = (args.vol, args.vol)
    volume_shape = (args.vol, args.vol, args.vol)
    cfg = _Cfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)
    grid = build_fixed_grid(healpix_order=args.healpix_order, max_shift=1)
    weights_v = make_half_volume_weights(volume_shape)

    summary = {}

    for q in qs:
        print(f"\n========== q = {q} ==========", flush=True)
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
        s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
        mu_true = ds.mu_half_true
        U_true = ds.U_half_true
        n_states = ds.state_coords_true.shape[0]
        print(f"  s_true: {np.asarray(ds.s_true)}", flush=True)
        print(f"  n_states={n_states} n_img={ds.n_img}", flush=True)

        # Init: oracle
        init_oracle = PPCAInit(
            mu=mu_true.astype(jnp.complex128),
            U=U_true.astype(jnp.complex128),
            s=s_kernel,
            volume_shape=tuple(int(x) for x in cfg.volume_shape),
        )

        # Init: baseline warmstart (full-image residual-PCA)
        t0 = time.perf_counter()
        U_baseline = warmstart_baseline(cfg, ds, mu_true, q)
        print(f"  baseline warmstart wall: {time.perf_counter() - t0:.1f}s", flush=True)
        init_baseline = PPCAInit(
            mu=mu_true.astype(jnp.complex128),
            U=U_baseline.astype(jnp.complex128),
            s=s_kernel,
            volume_shape=tuple(int(x) for x in cfg.volume_shape),
        )

        print("\n  --- ORACLE init ---", flush=True)
        _, rows_oracle = run_factor_steps(cfg, ds, init_oracle, args.K, label=f"q{q}_oracle")

        print("\n  --- BASELINE warmstart init ---", flush=True)
        _, rows_bl = run_factor_steps(cfg, ds, init_baseline, args.K, label=f"q{q}_base  ")

        summary[q] = {"oracle": rows_oracle, "baseline": rows_bl}

    # Final summary
    print("\n\n=== FINAL SUMMARY ===", flush=True)
    for q in qs:
        print(f"\n  q={q}", flush=True)
        oracle_rows = summary[q]["oracle"]
        baseline_rows = summary[q]["baseline"]
        print("    init=ORACLE", flush=True)
        for k, m in oracle_rows:
            print(f"      k={k:>2}: {fmt_metrics(m)}", flush=True)
        print("    init=BASELINE_WARMSTART", flush=True)
        for k, m in baseline_rows:
            print(f"      k={k:>2}: {fmt_metrics(m)}", flush=True)


if __name__ == "__main__":
    main()
