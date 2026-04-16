"""Diagnostic v9: is the joint loop basin of attraction gated by mu or U?

V8 showed:
  - joint loop from (perturbed_mu, SVD_U) converges to hun ~ 0.44
  - joint loop from (burnin_mu, SVD_U) converges to hun ~ 0.52
  - joint loop from (mu_true, U_true) converges to hun ~ 0.955
  - best-selection bootstrap only gets to hun ~ 0.56 at perturbed mu

Question: what matters for landing in the oracle basin — mu or U?

Experiments:
  1. Joint from (mu_true, U_random_ortho)            -- U doesn't matter if mu ok?
  2. Joint from (mu_true, U_svd_at_mu_true)          -- SVD at true mu
  3. Joint from (mu_true, U_zero)                    -- degenerate init
  4. Joint from (perturbed_mu, U_true)               -- perturbed mu, oracle U
  5. Joint from (burnin_mu, U_true)                  -- burnin mu, oracle U
  6. Joint from (mu_halfway, U_true) where mu_halfway = 0.5*mu_true + 0.5*mu_burnin
  7. Joint from (mu_halfway, U_halfway)              -- gradual interpolation
  8. Joint from (mu_true, U_half) where U_half = mix of U_true and random

If experiments 1-3 all reach oracle → U init doesn't matter, only mu
If experiment 4 reaches oracle → perturbed mu recovers with oracle U
If experiment 4 stays stuck → even oracle U can't save perturbed mu

This tells us whether to focus on mu initialization or U initialization.

Run:
  cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
  CUDA_VISIBLE_DEVICES=0 RECOVAR_DISABLE_CUDA=1 pixi run python \
      scripts/ppca_abinitio/diag_cluster_bootstrap_v9.py
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


def joint_m_steps(cfg, ds, init, n_steps, verbose_every=10):
    cur = init
    traj = []
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
        if step == 1 or step == n_steps or step % verbose_every == 0:
            pass
    return cur


def joint_m_steps_with_traj(cfg, ds, init, n_steps, n_states, tag):
    cur = init
    m0 = summarize_metrics(cfg, ds, cur, n_states)
    print(f"    {tag} k=0:  hun={m0['hun']:.4f} logm={m0['logm']:.3e}", flush=True)
    huns = [m0["hun"]]
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
        if step % 10 == 0 or step == n_steps:
            m = summarize_metrics(cfg, ds, cur, n_states)
            print(f"    {tag} k={step:3d}: hun={m['hun']:.4f} logm={m['logm']:.3e}", flush=True)
            huns.append(m["hun"])
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


def svd_to_U_half(real_residuals, q, volume_shape):
    rr = real_residuals - real_residuals.mean(axis=0, keepdims=True)
    _, _, Vh = np.linalg.svd(rr, full_matrices=False)
    V = Vh[:q].T
    U_half_rows = []
    for k in range(q):
        pc_real = V[:, k].reshape(volume_shape)
        U_half_rows.append(real_volume_to_half(jnp.asarray(pc_real), volume_shape))
    U_half = jnp.stack(U_half_rows).astype(jnp.complex128)
    weights_v = make_half_volume_weights(volume_shape)
    return real_volume_orthonormalize_half(U_half, weights_v, int(np.prod(volume_shape)))


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
    args = ap.parse_args()

    q = args.q
    print(f"### v9 basin-of-attraction q={q} vol={args.vol} n_img={args.n_images} sigma={args.sigma} ###", flush=True)

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

    # Perturbed mu
    rng = np.random.default_rng(args.seed + 1)
    noise = rng.standard_normal(2 * half_vol_size).view(np.complex128)
    noise_jax = project_to_real_volume_subspace(jnp.asarray(noise, dtype=jnp.complex128), volume_shape)
    mu_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(ds.mu_half_true) * ds.mu_half_true))))
    noise_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(noise_jax) * noise_jax))))
    scale = args.mu_eps * mu_norm / max(noise_norm, 1e-12)
    mu_perturbed = (ds.mu_half_true + scale * noise_jax).astype(jnp.complex128)

    # Burn-in mu
    print("\n[computing burnin mu (15 iters homogeneous from zero-U)]", flush=True)
    init_zero = PPCAInit(
        mu=mu_perturbed,
        U=jnp.zeros((q, half_vol_size), dtype=jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in volume_shape),
    )
    burnt = homogeneous_burnin(cfg, ds, init_zero, args.n_burnin)
    mu_burnin = burnt.mu

    # SVD warmstart U at true mu
    print("[computing SVD-U at mu_true]", flush=True)
    r_arg, t_arg = argmax_poses_at_mu(cfg, ds, ds.mu_half_true, q)
    bp = real_space_residual_backprojections_at_mu(cfg, ds, ds.mu_half_true, r_arg, t_arg)
    U_svd_at_true = svd_to_U_half(bp, q, volume_shape)

    # Random ortho U
    print("[computing random orthonormal U]", flush=True)
    U_rand = random_ortho_U(q, volume_shape, seed=args.seed + 100)

    # Zero U
    U_zero = jnp.zeros((q, half_vol_size), dtype=jnp.complex128)

    vshape = tuple(int(x) for x in volume_shape)
    experiments = [
        ("1. (mu_true, U_random)", ds.mu_half_true, U_rand),
        ("2. (mu_true, U_svd_at_true)", ds.mu_half_true, U_svd_at_true),
        ("3. (mu_true, U_true)", ds.mu_half_true, ds.U_half_true),
        ("4. (perturbed, U_true)", mu_perturbed, ds.U_half_true),
        ("5. (burnin,    U_true)", mu_burnin, ds.U_half_true),
        ("6. (perturbed, U_random)", mu_perturbed, U_rand),
        ("7. (burnin,    U_random)", mu_burnin, U_rand),
    ]

    results = {}
    for tag, mu0, U0 in experiments:
        print(f"\n{tag}", flush=True)
        init = PPCAInit(
            mu=jnp.asarray(mu0, dtype=jnp.complex128),
            U=jnp.asarray(U0, dtype=jnp.complex128),
            s=s_kernel,
            volume_shape=vshape,
        )
        cur = joint_m_steps_with_traj(cfg, ds, init, args.n_joint, n_states, tag)
        m = summarize_metrics(cfg, ds, cur, n_states)
        pe = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
        print(f"    {tag} FINAL: hun={m['hun']:.4f} pe={pe:.3f} logm={m['logm']:.3e}", flush=True)
        results[tag] = m["hun"]

    print("\n=== SUMMARY ===", flush=True)
    for tag, hun in results.items():
        print(f"  {tag:40s}: hun={hun:.4f}", flush=True)


if __name__ == "__main__":
    main()
