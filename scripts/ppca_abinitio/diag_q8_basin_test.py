"""Diagnostic: at q=8 on Ribosembly, test which part of initialization
causes the joint EM loop to land in a basin OTHER than the oracle basin.

Motivation
----------
The q-sweep oracle-ceiling diagnostic shows that at q=8, running the
joint mu+U EM loop from literal (mu_true, U_true_top8) gives
Hungarian ≈ 0.995 at iter=15. That's a strong attractor.

However, running the full run_cryobench.py at q=8 with the current
(mu_perturbed, U_svd_warmstart) pair lands in a basin where
  - pe (projector Frobenius error vs U_true_top8) grows from 3.75 -> 3.86
  - fre vs oracle fp grows from 0.15 -> 0.35
  - log marginal rises smoothly but not to the oracle-basin value.

This diagnostic isolates which piece of initialization drops us out
of the oracle basin. For each of four (mu_init, U_init) combinations,
we run 20 joint EM iters and report hun/ari at iter 0, 1, 5, 10, 20.

Combinations:
  A. (mu_true,      U_true_top8)       -- reference, should stay at hun~1
  B. (mu_true,      U_svd_warmstart)   -- mu oracle, U from warmstart
  C. (mu_perturbed, U_true_top8)       -- mu perturbed, U oracle
  D. (mu_perturbed, U_svd_warmstart)   -- same as real cryobench run

Run
---
cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
CUDA_VISIBLE_DEVICES=0 RECOVAR_DISABLE_CUDA=1 pixi run python \\
    scripts/ppca_abinitio/diag_q8_basin_test.py
"""

from __future__ import annotations

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


def joint_loop_from_init(cfg, ds, mu_init, U_init, n_joint, report_iters):
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    cur = PPCAInit(
        mu=mu_init.astype(jnp.complex128),
        U=U_init.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )
    out = []
    if 0 in report_iters:
        out.append(("iter=0", summarize_clustering(cfg, ds, cur), 0.0))
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
        jax.block_until_ready(cur.U)
        dt = time.perf_counter() - t0
        if k in report_iters:
            metrics = summarize_clustering(cfg, ds, cur)
            out.append((f"iter={k} ({dt:.1f}s)", metrics, dt))
    return cur, out


def compute_svd_warmstart(cfg, ds, mu_half, q):
    """Replicates init_U_from_residual_svd unweighted variant, sanity-simplified."""
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


def print_traj(name, out):
    print(f"  {name}:", flush=True)
    for tag, m, dt in out:
        print(f"    {tag:<20}: hun={m['hun']:.4f}  ari={m['ari']:.4f}  nmi={m['nmi']:.4f}", flush=True)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--n-joint", type=int, default=20)
    ap.add_argument("--mu-perturb-eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--healpix-order", type=int, default=1)
    args = ap.parse_args()

    print(f"### q={args.q} basin test vol={args.vol} n_img={args.n_images} sigma={args.sigma} ###", flush=True)

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
    U_true = ds.U_half_true.astype(jnp.complex128)
    mu_pert = make_perturbed_mu(mu_true, volume_shape, weights_v, args.init_seed, eps=args.mu_perturb_eps)

    print("computing svd warmstart from mu_true (one-time)...", flush=True)
    U_svd_from_true = compute_svd_warmstart(cfg, ds, mu_true, args.q)
    print("computing svd warmstart from mu_perturbed (one-time)...", flush=True)
    U_svd_from_pert = compute_svd_warmstart(cfg, ds, mu_pert, args.q)

    report_iters = [0, 1, 5, 10, 20, args.n_joint]
    report_iters = sorted(set([r for r in report_iters if r <= args.n_joint]))

    print("\n=== A. oracle mu, oracle U (reference) ===", flush=True)
    _, outA = joint_loop_from_init(cfg, ds, mu_true, U_true, args.n_joint, report_iters)
    print_traj("A (mu_true, U_true_top8)", outA)

    print("\n=== B. oracle mu, svd warmstart U (from mu_true) ===", flush=True)
    _, outB = joint_loop_from_init(cfg, ds, mu_true, U_svd_from_true, args.n_joint, report_iters)
    print_traj("B (mu_true, U_svd)", outB)

    print("\n=== C. perturbed mu, oracle U ===", flush=True)
    _, outC = joint_loop_from_init(cfg, ds, mu_pert, U_true, args.n_joint, report_iters)
    print_traj("C (mu_pert, U_true_top8)", outC)

    print("\n=== D. perturbed mu, svd warmstart U (from mu_perturbed, same as cryobench) ===", flush=True)
    _, outD = joint_loop_from_init(cfg, ds, mu_pert, U_svd_from_pert, args.n_joint, report_iters)
    print_traj("D (mu_pert, U_svd)", outD)

    print("\n=== SUMMARY ===", flush=True)
    for name, out in [
        ("A (oracle, oracle)        ", outA),
        ("B (mu_true, U_svd)        ", outB),
        ("C (mu_pert, U_true)       ", outC),
        ("D (mu_pert, U_svd) [real] ", outD),
    ]:
        final_hun = out[-1][1]["hun"]
        print(f"  {name} final hun={final_hun:.4f}", flush=True)


if __name__ == "__main__":
    main()
