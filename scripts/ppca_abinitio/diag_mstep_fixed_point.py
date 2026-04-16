"""Diagnostic: is U_true a fixed point of the closed-form factor M-step?

Findings from diag_warmstart_compare.py: ALL three warmstart strategies
(svd_half, svd_half_weighted, baseline) converge to roughly the same
acc ≈ 0.33 after 12 closed-form factor updates with mu frozen at mu_true.
The oracle ceiling is acc ≈ 0.81. Either:

  (A) the M-step has a real bug -> oracle is not a fixed point
  (B) the M-step is fine but EM finds a wrong local minimum -> oracle
      is a fixed point and the warmstarts are in the wrong basin
  (C) q=2 is too small -> raising q should fix it

We test all three:

  1. Oracle stability:    start at (mu_true, U_true), run 12 closed-form
                          factor updates with mu frozen, watch pe and acc.
  2. Oracle perturbation: start at U_true + ε * random_unit, see if it
                          converges back.
  3. Higher q:            run with q=4 and q=8 to test (C).

Run
---
CUDA_VISIBLE_DEVICES=2 RECOVAR_DISABLE_CUDA=1 pixi run python \\
    scripts/ppca_abinitio/diag_mstep_fixed_point.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.em.ppca_abinitio.factor_update import update_factor_closed_form
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    make_half_volume_weights,
    real_volume_orthonormalize_half,
)
from recovar.em.ppca_abinitio.metrics import _orthogonal_procrustes, projector_frobenius_error
from recovar.em.ppca_abinitio.posterior import score_and_posterior_moments_eqx
from recovar.em.ppca_abinitio.synthetic import SyntheticFamily, make_synthetic_fixed_grid_dataset
from recovar.em.ppca_abinitio.types import PPCAInit
from recovar.utils.helpers import load_mrc

_S_FLOOR = 1e-6


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


def centroid_acc_for_init(cfg, ds, init):
    if ds.state_label_true is None or ds.state_coords_true is None:
        return float("nan")
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
    return float(np.mean(pred_label == state_label_true))


def run_factor_steps(cfg, ds, init, K, label):
    """Run K closed-form factor updates with mu frozen and report acc/pe per step."""
    cur = init
    pe = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
    acc = centroid_acc_for_init(cfg, ds, cur)
    print(f"  [{label} k={0:>2}] pe={pe:.4f} acc={acc:.4f}", flush=True)
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
        # Force mu back to mu_true (the closed-form M-step doesn't change it,
        # but be explicit).
        cur = PPCAInit(mu=init.mu, U=cur.U, s=cur.s, volume_shape=cur.volume_shape)
        pe = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
        acc = centroid_acc_for_init(cfg, ds, cur)
        print(f"  [{label} k={k:>2}] pe={pe:.4f} acc={acc:.4f}", flush=True)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Ribosembly")
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--healpix-order", type=int, default=1)
    ap.add_argument("--K", type=int, default=12, help="number of closed-form M-step iterations")
    ap.add_argument("--qs", type=str, default="2,4,8", help="comma list of q values to try")
    args = ap.parse_args()

    qs = [int(x) for x in args.qs.split(",")]
    root = Path("/home/mg6942/mytigress/cryobench2") / args.dataset

    print(f"### diag_mstep_fixed_point dataset={args.dataset} vol={args.vol} n_images={args.n_images} ###")
    print("loading cryobench gt volumes...", flush=True)
    gt_vols = load_cryobench_gt_volumes(root, target_D=args.vol)

    image_shape = (args.vol, args.vol)
    volume_shape = (args.vol, args.vol, args.vol)
    cfg = _Cfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)
    grid = build_fixed_grid(healpix_order=args.healpix_order, max_shift=1)
    weights_v = make_half_volume_weights(volume_shape)

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
        print(f"  s_true: {np.asarray(ds.s_true)}", flush=True)

        init_oracle = PPCAInit(
            mu=mu_true.astype(jnp.complex128),
            U=U_true.astype(jnp.complex128),
            s=s_kernel,
            volume_shape=tuple(int(x) for x in cfg.volume_shape),
        )
        oracle_pe = float(projector_frobenius_error(init_oracle.U, ds.U_half_true, cfg.volume_shape))
        oracle_acc = centroid_acc_for_init(cfg, ds, init_oracle)
        print(f"\n  ORACLE (mu_true, U_true): pe={oracle_pe:.4f} acc={oracle_acc:.4f}", flush=True)

        # === test 1: oracle stability ===
        print("\n  --- test 1: oracle stability (start at U_true) ---", flush=True)
        run_factor_steps(cfg, ds, init_oracle, args.K, label=f"q{q}_oracle")

        # === test 2: oracle perturbation ===
        # Add a small random perturbation to U_true and re-orthonormalize
        # in the real-volume sense, so the perturbed init is still in the
        # right gauge.
        print("\n  --- test 2: small perturbation of U_true ---", flush=True)
        rng = np.random.default_rng(args.seed + 1)
        U_pert = np.array(np.asarray(U_true), dtype=np.complex128, copy=True)
        eps = 0.05
        noise = (rng.standard_normal(U_pert.shape) + 1j * rng.standard_normal(U_pert.shape)) * eps
        U_pert = U_pert + noise.astype(np.complex128)
        U_pert = real_volume_orthonormalize_half(jnp.asarray(U_pert), weights_v, int(np.prod(volume_shape)))
        init_pert = PPCAInit(
            mu=mu_true.astype(jnp.complex128),
            U=U_pert.astype(jnp.complex128),
            s=s_kernel,
            volume_shape=tuple(int(x) for x in cfg.volume_shape),
        )
        run_factor_steps(cfg, ds, init_pert, args.K, label=f"q{q}_pert")


if __name__ == "__main__":
    main()
