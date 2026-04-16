"""Diagnostic v12: characterize the U_true basin of attraction radius.

V9 finding: U_true → joint loop reaches hun=0.95. U_random → joint loop
reaches hun=0.46. There are at least two basins.

Question: how close to U_true does an init need to be to land in the
good basin? Specifically, for U_init = orthonormalize((1-α) U_true + α U_random),
at what α does the basin transition occur?

If even α=0.5 stays in the good basin → the basin radius is large, and
we should focus on getting U init within ~50% rotational distance of U_true.

If even α=0.05 escapes → the basin radius is tiny, and we'd need to
init very close to U_true to land there (essentially impossible).

Run:
  cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
  CUDA_VISIBLE_DEVICES=0 RECOVAR_DISABLE_CUDA=1 pixi run python \
      scripts/ppca_abinitio/diag_basin_radius_v12.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.factor_update import update_factor_closed_form
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    make_half_volume_weights,
    project_to_real_volume_subspace,
    real_volume_orthonormalize_half,
)
from recovar.em.ppca_abinitio.mean_update import (
    update_mu_homogeneous,
    update_mu_residualized,
)
from recovar.em.ppca_abinitio.metrics import projector_frobenius_error
from recovar.em.ppca_abinitio.posterior import (
    score_and_posterior_moments_eqx,
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
    vol_dir = dataset_root / "vols" / "128_org"
    vol_files = sorted(vol_dir.glob("*.mrc"))
    vols = np.stack([np.asarray(load_mrc(str(vf)), dtype=np.float64) for vf in vol_files])
    if vols.shape[-1] != target_D:
        vols = np.stack([downsample_volume(vols[k], target_D) for k in range(vols.shape[0])])
    return vols


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
    return {"hun": _hungarian(labels_true, cluster_labels)}


def joint_m_steps(cfg, ds, init, n_steps, n_states, U_true_half, tag, report_iters):
    cur = init
    m0 = summarize_metrics(cfg, ds, cur, n_states)
    pe0 = float(projector_frobenius_error(cur.U, U_true_half, cfg.volume_shape))
    print(f"    {tag} k= 0: hun={m0['hun']:.4f} pe={pe0:.3f}", flush=True)
    for step in range(1, n_steps + 1):
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
        if step in report_iters or step == n_steps:
            m = summarize_metrics(cfg, ds, cur, n_states)
            pe = float(projector_frobenius_error(cur.U, U_true_half, cfg.volume_shape))
            print(f"    {tag} k={step:2d}: hun={m['hun']:.4f} pe={pe:.3f}", flush=True)
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


def mix_U(U_true_half, U_rand_half, alpha, volume_shape):
    """Build U = (1-α)·U_true + α·U_rand, then real-orthonormalize.

    α=0 → U_true. α=1 → U_rand.
    """
    U_mix = (1.0 - alpha) * U_true_half + alpha * U_rand_half
    weights_v = make_half_volume_weights(volume_shape)
    return real_volume_orthonormalize_half(U_mix, weights_v, int(np.prod(volume_shape)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-joint", type=int, default=30)
    args = ap.parse_args()

    q = args.q
    print(f"### v12 basin-radius q={q} vol={args.vol} n_img={args.n_images} sigma={args.sigma} ###", flush=True)

    root = Path("/home/mg6942/mytigress/cryobench2") / "Ribosembly"
    gt_vols = load_cryobench_gt_volumes(root, target_D=args.vol)
    grid = build_fixed_grid(healpix_order=1, max_shift=1)
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
    vshape = tuple(int(x) for x in volume_shape)

    # Use mu_true to isolate the U-init effect
    print("\n[using mu_true]", flush=True)
    mu_init = ds.mu_half_true

    U_rand = random_ortho_U(q, volume_shape, seed=args.seed + 100)

    alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    print(f"\nTesting α ∈ {alphas}", flush=True)

    results = {}
    for alpha in alphas:
        U_init = mix_U(
            jnp.asarray(ds.U_half_true, dtype=jnp.complex128),
            jnp.asarray(U_rand, dtype=jnp.complex128),
            alpha,
            volume_shape,
        )
        tag = f"α={alpha:.2f}"
        print(f"\n--- {tag} ---", flush=True)
        init = PPCAInit(
            mu=jnp.asarray(mu_init, dtype=jnp.complex128),
            U=U_init,
            s=s_kernel,
            volume_shape=vshape,
        )
        cur = joint_m_steps(cfg, ds, init, args.n_joint, n_states, ds.U_half_true, tag, (1, 5, 15, 30))
        m = summarize_metrics(cfg, ds, cur, n_states)
        pe = float(projector_frobenius_error(cur.U, ds.U_half_true, volume_shape))
        results[alpha] = (m["hun"], pe)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  {'alpha':>6s}  {'final_hun':>10s}  {'final_pe':>10s}", flush=True)
    for alpha, (hun, pe) in results.items():
        print(f"  {alpha:>6.2f}  {hun:>10.4f}  {pe:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
