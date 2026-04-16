"""Diagnostic v10: better U init methods to escape the lazy basin.

V9 finding: U init is THE bottleneck. With U_true, even perturbed mu reaches
hun=0.95. With random or per-image-SVD U, all paths plateau at hun=0.45-0.61.

Hypothesis: residual SVD picks up noise directions because the per-image
backprojected residuals are dominated by per-image noise. Cluster-averaging
backprojections before SVD should denoise. Low-pass filtering should isolate
state-variation (low-freq) from noise (high-freq).

Methods tested at mu_burnin (15 homog iters):
  A. U_random_ortho (floor)
  B. U_svd_per_image (current production: residual SVD over per-image bp's)
  C. U_kmeans_K16   (kmeans on per-image backprojections, 16 clusters,
                     SVD on cluster-mean differences)
  D. U_kmeans_K32   (same with K=32)
  E. U_kmeans_K8    (same with K=q=8)
  F. U_lowpass      (low-pass filter cutoff r<N/4 on real residuals before SVD)
  G. U_true (oracle ceiling)

For each: report
  - pe (subspace Frobenius distance to U_true)
  - initial hun (E-step only)
  - hun, pe, logm at iter 5, 15, 30 of joint loop

Success criterion: any non-oracle method reaching pe < 2.0 or final hun > 0.75.

Run:
  cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408
  CUDA_VISIBLE_DEVICES=0 RECOVAR_DISABLE_CUDA=1 pixi run python \
      scripts/ppca_abinitio/diag_cluster_bootstrap_v10.py
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


def joint_m_steps_with_traj(cfg, ds, init, n_steps, n_states, tag, U_true_half, report_iters=(5, 15, 30)):
    cur = init
    m0 = summarize_metrics(cfg, ds, cur, n_states)
    pe0 = float(projector_frobenius_error(cur.U, U_true_half, cfg.volume_shape))
    print(f"    {tag} k= 0: hun={m0['hun']:.4f} pe={pe0:.3f} logm={m0['logm']:.3e}", flush=True)
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
            print(f"    {tag} k={step:2d}: hun={m['hun']:.4f} pe={pe:.3f} logm={m['logm']:.3e}", flush=True)
    return cur


def factor_only_steps_with_traj(cfg, ds, init, n_steps, n_states, tag, U_true_half, report_iters=(5, 15, 30)):
    """Run only the factor M-step with mu frozen."""
    cur = init
    m0 = summarize_metrics(cfg, ds, cur, n_states)
    pe0 = float(projector_frobenius_error(cur.U, U_true_half, cfg.volume_shape))
    print(f"    {tag} F k= 0: hun={m0['hun']:.4f} pe={pe0:.3f}", flush=True)
    for step in range(1, n_steps + 1):
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
            print(f"    {tag} F k={step:2d}: hun={m['hun']:.4f} pe={pe:.3f}", flush=True)
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


def svd_to_U_half(real_data, q, volume_shape):
    """Center, SVD, take top-q right singular vectors, encode to half-volume."""
    rr = real_data - real_data.mean(axis=0, keepdims=True)
    _, _, Vh = np.linalg.svd(rr, full_matrices=False)
    V = Vh[:q].T
    U_half_rows = []
    for k in range(q):
        pc_real = V[:, k].reshape(volume_shape)
        U_half_rows.append(real_volume_to_half(jnp.asarray(pc_real), volume_shape))
    U_half = jnp.stack(U_half_rows).astype(jnp.complex128)
    weights_v = make_half_volume_weights(volume_shape)
    return real_volume_orthonormalize_half(U_half, weights_v, int(np.prod(volume_shape)))


def cluster_mean_svd_U(real_residuals, q, K, volume_shape, seed=0):
    """K-means cluster the per-image backprojections, average per cluster,
    then SVD on (cluster_means - global_mean) to get q directions.

    Averaging in each cluster reduces noise by sqrt(images_per_cluster).
    """
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels = km.fit_predict(real_residuals)
    n_voxels = real_residuals.shape[1]
    cluster_means = np.zeros((K, n_voxels), dtype=np.float64)
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            cluster_means[k] = real_residuals[mask].mean(axis=0)
    return svd_to_U_half(cluster_means, q, volume_shape)


def lowpass_filter_real(volume_real, cutoff_voxels):
    """Apply hard low-pass filter to a real-space volume by zeroing
    Fourier coefficients with |k| > cutoff_voxels.

    cutoff_voxels is in voxel index units (e.g., N/4 for half-Nyquist).
    """
    D = volume_real.shape[-1]
    F = np.fft.fftshift(np.fft.fftn(volume_real))
    c = D // 2
    z = np.arange(D) - c
    yy, xx, zz = np.meshgrid(z, z, z, indexing="ij")
    r = np.sqrt(xx * xx + yy * yy + zz * zz)
    mask = (r <= cutoff_voxels).astype(np.float64)
    F_filtered = F * mask
    return np.real(np.fft.ifftn(np.fft.ifftshift(F_filtered))).astype(np.float64)


def lowpass_svd_U(real_residuals, q, volume_shape, cutoff_voxels):
    """Low-pass filter each backprojected residual, then do SVD."""
    n_img = real_residuals.shape[0]
    filtered = np.empty_like(real_residuals)
    for i in range(n_img):
        v = real_residuals[i].reshape(volume_shape)
        filtered[i] = lowpass_filter_real(v, cutoff_voxels).reshape(-1)
    return svd_to_U_half(filtered, q, volume_shape)


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
    ap.add_argument(
        "--use-mu-true", action="store_true", help="Use mu_true instead of burnin mu (cleaner test of U init)."
    )
    args = ap.parse_args()

    q = args.q
    print(f"### v10 better-U-init q={q} vol={args.vol} n_img={args.n_images} sigma={args.sigma} ###", flush=True)

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
    vshape = tuple(int(x) for x in volume_shape)

    # Choose mu source
    if args.use_mu_true:
        print("\n[using mu_true (no burnin)]", flush=True)
        mu_init = ds.mu_half_true
    else:
        rng = np.random.default_rng(args.seed + 1)
        noise = rng.standard_normal(2 * half_vol_size).view(np.complex128)
        noise_jax = project_to_real_volume_subspace(jnp.asarray(noise, dtype=jnp.complex128), volume_shape)
        mu_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(ds.mu_half_true) * ds.mu_half_true))))
        noise_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(noise_jax) * noise_jax))))
        scale = args.mu_eps * mu_norm / max(noise_norm, 1e-12)
        mu_perturbed = (ds.mu_half_true + scale * noise_jax).astype(jnp.complex128)
        print("\n[computing burnin mu (15 iters homogeneous from zero-U)]", flush=True)
        init_zero = PPCAInit(
            mu=mu_perturbed,
            U=jnp.zeros((q, half_vol_size), dtype=jnp.complex128),
            s=s_kernel,
            volume_shape=vshape,
        )
        burnt = homogeneous_burnin(cfg, ds, init_zero, args.n_burnin)
        mu_init = burnt.mu

    # Compute backprojections at the chosen mu, used by all SVD-style inits
    print("[computing argmax-pose residual backprojections at chosen mu]", flush=True)
    r_arg, t_arg = argmax_poses_at_mu(cfg, ds, mu_init, q)
    real_residuals = real_space_residual_backprojections_at_mu(cfg, ds, mu_init, r_arg, t_arg)
    print(f"  real_residuals shape={real_residuals.shape} std={real_residuals.std():.3e}", flush=True)

    # Build U candidates
    candidates = []

    print("\n[building U_random_ortho]", flush=True)
    U_rand = random_ortho_U(q, volume_shape, seed=args.seed + 100)
    candidates.append(("A. U_random_ortho", U_rand))

    print("[building U_svd_per_image]", flush=True)
    U_svd_per = svd_to_U_half(real_residuals, q, volume_shape)
    candidates.append(("B. U_svd_per_image", U_svd_per))

    for K in (16, 32, 8):
        print(f"[building U_kmeans_K{K}]", flush=True)
        U_km = cluster_mean_svd_U(real_residuals, q, K, volume_shape, seed=args.seed)
        candidates.append((f"C. U_kmeans_K{K:02d}", U_km))

    # Low-pass cutoffs (in voxel units; D=32 => Nyquist at 16, half-Nyquist=8)
    for cutoff in (4, 8, 12):
        print(f"[building U_lowpass_r{cutoff}]", flush=True)
        U_lp = lowpass_svd_U(real_residuals, q, volume_shape, cutoff)
        candidates.append((f"F. U_lowpass_r{cutoff:02d}", U_lp))

    candidates.append(("G. U_true (oracle)", ds.U_half_true))

    # Print initial pe for each (subspace error to U_true)
    print("\n=== INITIAL pe (subspace error to U_true) ===", flush=True)
    pe_max = float(np.sqrt(2 * q))
    print(f"  (pe_max = sqrt(2q) = {pe_max:.3f})", flush=True)
    for tag, U0 in candidates:
        pe0 = float(projector_frobenius_error(U0, ds.U_half_true, volume_shape))
        print(f"  {tag:24s}: pe={pe0:.3f}", flush=True)

    # Run joint loop for each
    results = {}
    for tag, U0 in candidates:
        print(f"\n--- {tag} ---", flush=True)
        init = PPCAInit(
            mu=jnp.asarray(mu_init, dtype=jnp.complex128),
            U=jnp.asarray(U0, dtype=jnp.complex128),
            s=s_kernel,
            volume_shape=vshape,
        )
        cur = joint_m_steps_with_traj(
            cfg,
            ds,
            init,
            args.n_joint,
            n_states,
            tag,
            ds.U_half_true,
            report_iters=(1, 5, 15, 30),
        )
        m = summarize_metrics(cfg, ds, cur, n_states)
        pe = float(projector_frobenius_error(cur.U, ds.U_half_true, volume_shape))
        print(f"  {tag} FINAL: hun={m['hun']:.4f} pe={pe:.3f} logm={m['logm']:.3e}", flush=True)
        results[tag] = (m["hun"], pe)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  {'method':30s}  {'final_hun':>10s}  {'final_pe':>10s}", flush=True)
    for tag, (hun, pe) in results.items():
        print(f"  {tag:30s}  {hun:>10.4f}  {pe:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
