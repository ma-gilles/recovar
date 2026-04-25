"""Run v0 ab-initio PPCA on cryobench-derived synthetic data.

We deliberately do NOT use the cryobench particle images. Their
loader/scale/CTF/pose conventions don't match the v0 synthetic
forward model and bridging that is its own engineering task. Instead
we use ONLY the cryobench ground-truth volumes (which describe a
realistic biological conformation manifold), feed them through the
v0 synthetic harness as the conformation ensemble, and let the
harness generate images from them with the v0 forward model. Every
convention then matches by construction.

This script is a **non-gating upper-bound diagnostic**, not a stage
graduation or bootstrap-success claim. Because the conformation
manifold comes directly from CryoBench GT volumes and the synthetic
images are generated on the same inference grid and with the same
forward model used at inference time, this is a best-case aligned
setting for the current PPCA loop. We use it to estimate the ceiling
of the current model/discretization pair once convention mismatch is
removed; we do **not** expect it to represent what ab-initio should
reach from a realistic external initialization.

Interpret results accordingly:

- `mu` FRE and `U` projector error are sanity checks, not the whole
  story.
- For `Ribosembly` especially, the meaningful question is whether the
  latent coordinates induced by the learned `U` separate/cluster the
  underlying conformations in latent space.
- Raw volume/PC distances can be misleading here because two valid
  solutions may be equivalent up to a global 3D rotation, handedness
  flip, or per-PC sign, and we do not yet have a canonical
  rotation-aware comparison for that case.
- `--n-images` means the total number of synthetic images used by the
  diagnostic. This script is not a held-out-generalization benchmark,
  so it does not add an extra validation tail behind your back.
- The Procrustes-aligned `centroid_acc` is biased by alignment to
  the top-q real-space PCA basis of the gt ensemble. Because the
  discrete CryoBench volumes do NOT live exactly on a q-dim linear
  subspace, that basis is *not* the EM-ML optimum; one closed-form
  M-step from `U_true` already moves `U` away and `centroid_acc`
  drops a few points even when actual clustering quality (Hungarian
  / ARI) barely changes. We therefore now also report a
  basis-invariant Hungarian-matched k-means accuracy and ARI/NMI,
  plus a "post-EM oracle ceiling" that runs 12 closed-form M-steps
  from `(mu_true, U_true)` to give the actual reachable target.
- For datasets with many discrete states (e.g. Ribosembly has 16),
  small `q` is severely model-misspecified and clustering quality
  is capped well below the pre-EM oracle. The script emits a
  warning when `q` is much smaller than `n_states`. Empirically on
  Ribosembly at vol=32, q=4 reaches `centroid_acc ≈ 0.83`, more
  than 2x the q=2 result.

Pipeline:
  1. Load cryobench gt volumes (.mrc files)
  2. Downsample to target_D
  3. Pass them as `external_volumes_real` to
     `make_synthetic_fixed_grid_dataset`. In the recommended
     `discrete_volumes` mode, each image is generated from one of the
     actual GT volumes through the recovar simulator, while the
     harness also computes the mean and top-q PCs of the GT ensemble
     for diagnostics.
  4. Run the two-stage burn-in + SVD-warmstart loop and report
     mu FRE and U projector error vs the ground truth.

Usage:
    pixi run python scripts/ppca_abinitio/run_cryobench.py \\
        --dataset IgG-1D --vol 32 --n-images 4096 --sigma 0.1 --q 2
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import recovar.core.fourier_transform_utils as ftu
from recovar.core.slicing import adjoint_slice_volume, slice_volume
from recovar.em.ppca_abinitio.factor_update import (
    compute_W_prior_half,
    update_factor_closed_form,
)
from recovar.em.ppca_abinitio.grid import build_fixed_grid
from recovar.em.ppca_abinitio.half_volume import (
    half_volume_radial_index,
    make_half_volume_weights,
    project_to_real_volume_subspace,
    project_to_real_volume_subspace_batch,
    real_volume_orthonormalize_half,
)
from recovar.em.ppca_abinitio.mean_update import update_mu_homogeneous, update_mu_residualized
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
# Cryobench gt-volume loader (only the volumes, not the particles)
# ---------------------------------------------------------------------------


def downsample_volume(vol_real, target_D):
    """Fourier-crop a real-space volume using recovar's canonical helpers."""
    D = vol_real.shape[-1]
    if target_D == D:
        return vol_real.astype(np.float64)
    assert D % 2 == 0 and target_D % 2 == 0 and target_D <= D
    F = np.asarray(ftu.get_dft3(jnp.asarray(vol_real)), dtype=np.complex128)
    c = D // 2
    h = target_D // 2
    F_crop = F[c - h : c + h, c - h : c + h, c - h : c + h]
    # JAX-backed array views can be read-only once converted to numpy.
    # Make an explicit writable copy before the amplitude rescale.
    out = np.array(np.asarray(ftu.get_idft3(jnp.asarray(F_crop))).real, dtype=np.float64, copy=True)
    out *= (target_D / D) ** 3
    return out


def load_cryobench_gt_volumes(dataset_root: Path, target_D: int):
    """Load all gt .mrc volumes from a cryobench dataset and
    downsample to target_D. Returns (K, target_D, target_D, target_D).
    """
    candidates = [
        dataset_root / "vols" / "128_org",
        dataset_root / "vols",
    ]
    vol_dir = next((p for p in candidates if p.exists()), None)
    if vol_dir is None:
        raise FileNotFoundError(f"no vol dir found under {dataset_root}/vols")

    vol_files = sorted(vol_dir.glob("*.mrc"))
    if not vol_files:
        raise FileNotFoundError(f"no .mrc files in {vol_dir}")
    vols = []
    for vf in vol_files:
        vols.append(np.asarray(load_mrc(str(vf)), dtype=np.float64))
    vols = np.stack(vols)
    print(f"  loaded {vols.shape[0]} gt vols of size {vols.shape[-1]}", flush=True)
    if vols.shape[-1] != target_D:
        print(f"  downsampling to {target_D}^3 ...", flush=True)
        vols = np.stack([downsample_volume(vols[k], target_D) for k in range(vols.shape[0])])
    return vols


# ---------------------------------------------------------------------------
# Config used by the v0 score kernel and M-step
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
# SVD-of-residuals U warmstart at converged mu
# ---------------------------------------------------------------------------


def init_U_from_residual_svd(cfg, ds, mu_half, s_kernel, q, weighted=True):
    """Residual-SVD warmstart on half-volume backprojections.

    Forms per-image residual backprojections at the hard argmax (pose,
    translation), then runs SVD to extract the top-q directions.

    When `weighted=True` (default), we project each residual to the
    real-volume half-spectrum subspace and multiply by
    sqrt(half-volume Hermitian weights) before the SVD so that the
    Frobenius norm in coefficient space matches the real-space ℓ²
    norm. This is required to match the ppca_abinitio gauge
    (`real_volume_orthonormalize_half` uses the same weights). On
    Ribosembly q=4 the weighted variant reaches hun≈0.78 vs
    hun≈0.62 for the raw-half-vol SVD (see `diag_warmstart_compare`).

    When `weighted=False`, the raw half-volume residuals are SVD'd
    directly (the historical behavior, kept for reproducing old runs).
    """
    image_shape = cfg.image_shape
    volume_shape = cfg.volume_shape
    weights_h = make_half_image_weights(image_shape)
    mean_proj = _slice_mu_half(mu_half, ds.rotations, image_shape, volume_shape).astype(jnp.complex128)
    n_rot = ds.rotations.shape[0]
    n_half_image = mean_proj.shape[-1]
    u_zero = jnp.zeros((n_rot, q, n_half_image), dtype=jnp.complex128)
    shifted_half, ctf2_over_nv_half, _ctf_half = _preprocess_batch_to_half(
        cfg, ds.batch_full, ds.translations, ds.ctf_params, ds.noise_variance_full
    )
    stats = score_from_half_image_projections(
        mean_proj,
        u_zero,
        s_kernel,
        shifted_half,
        ctf2_over_nv_half,
        weights_h,
    )
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
    weights_v = np.asarray(make_half_volume_weights(volume_shape), dtype=np.float64)

    if weighted:
        residual_volumes_jax = jnp.asarray(residual_volumes, dtype=jnp.complex128)
        residual_volumes_jax = project_to_real_volume_subspace_batch(residual_volumes_jax, volume_shape)
        sqrt_w = np.sqrt(weights_v)[None, :]
        rv = np.array(np.asarray(residual_volumes_jax), dtype=np.complex128, copy=True)
        rv -= rv.mean(axis=0, keepdims=True)
        rv_w = rv * sqrt_w
        _, S_svd, Vh_w = np.linalg.svd(rv_w, full_matrices=False)
        print(f"    residual SVD top-{q} singular values (weighted): {S_svd[:q]}", flush=True)
        Vh = Vh_w / sqrt_w
    else:
        residual_volumes -= residual_volumes.mean(axis=0, keepdims=True)
        _, S_svd, Vh = np.linalg.svd(residual_volumes, full_matrices=False)
        print(f"    residual SVD top-{q} singular values (unweighted): {S_svd[:q]}", flush=True)
    U_init = jnp.asarray(Vh[:q], dtype=jnp.complex128)
    U_orth = real_volume_orthonormalize_half(U_init, jnp.asarray(weights_v), int(np.prod(volume_shape)))
    return U_orth, S_svd[:q]


# ---------------------------------------------------------------------------
# Sphere-restricted FRE
# ---------------------------------------------------------------------------
#
# The slicer's `max_r = N/2 - 1` clips voxels outside the sphere
# inscribed in the cube. Those corner voxels are unrecoverable by
# the v0 mean update, so the global FRE has a hard floor equal to
# the corner-energy fraction. We compute a sphere-restricted FRE
# that ignores corner voxels — this is the metric the M-step can
# actually optimize.


def fre_sphere(mu_est, mu_true, volume_shape, weights_half, max_r):
    """FRE restricted to voxels at radial index <= max_r."""
    R = np.asarray(half_volume_radial_index(volume_shape))
    mask = jnp.asarray(R <= max_r, dtype=jnp.float64)
    w_masked = weights_half * mask
    diff = mu_est - mu_true
    num = float(jnp.sqrt(jnp.real(jnp.sum(w_masked * jnp.conj(diff) * diff))))
    den = float(jnp.sqrt(jnp.real(jnp.sum(w_masked * jnp.conj(mu_true) * mu_true))))
    return num / max(den, 1e-30)


def _log_marginal_sum(cfg, init, ds):
    """Sum-over-images of image log marginal under the current (mu, U, s).

    Equal to the true EM log marginal up to a theta-independent constant,
    so monotonicity of this value across EM iterations is equivalent to
    EM monotonicity on its own objective. Use this to decide whether a
    rising `pe` diagnostic is drift toward a misspecified-model optimum
    (log marginal still going up) or an actual bug (log marginal going
    down).
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
    flat = log_scores.reshape(log_scores.shape[0], -1)
    m = flat.max(axis=-1, keepdims=True)
    lm_per_img = (m + np.log(np.exp(flat - m).sum(axis=-1, keepdims=True))).reshape(-1)
    return float(lm_per_img.sum())


def _clustering_accuracy_hungarian(labels_true, labels_pred):
    """Best-permutation classification accuracy for a clustering.

    Builds a (K, K) contingency and solves a maximum-weight matching
    via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`
    with a negated objective). Returns `matched_count / n`. Invariant
    to cluster label permutations — unlike raw accuracy, which cares
    which number you called each cluster.
    """
    n = len(labels_true)
    k = max(int(labels_true.max()), int(labels_pred.max())) + 1
    C = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(labels_true, labels_pred):
        C[int(t), int(p)] += 1
    row_ind, col_ind = linear_sum_assignment(-C)
    return float(C[row_ind, col_ind].sum()) / n


def summarize_discrete_embedding(cfg, ds, final_init):
    """Summarize marginal latent recovery for discrete external-volume runs.

    Returns a dict of three clustering-quality metrics, not just one.
    Multiple metrics are needed because the one we had (Procrustes
    centroid accuracy) is biased by basis alignment to the top-q
    real-space PCA of the gt ensemble, which is *not* the EM-ML
    optimum under model misspecification. In practice it makes the
    oracle look artificially good and drifts downward after EM even
    when the embedding is actually still separating states well.

    Metrics
    -------
    centroid_acc : float
        Original metric. Procrustes-align alpha_hat to alpha_true's
        basis, nearest-centroid classify against state_coords_true.
        Easy to interpret but biased; treat as a legacy reference.
    clust_acc_hungarian : float
        Hungarian-matched classification accuracy of a k-means
        clustering of alpha_hat (K = n_states). Invariant to basis
        rotations and label permutations.
    ari : float
        Adjusted Rand Index for the same k-means clustering. Robust
        basis-free clustering quality.
    nmi : float
        Normalized mutual information for the same clustering.
    """
    if ds.state_label_true is None or ds.state_coords_true is None:
        return None

    stats = score_and_posterior_moments_eqx(
        cfg,
        final_init.mu,
        final_init.U,
        final_init.s,
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
    clust_acc_hungarian = _clustering_accuracy_hungarian(state_label_true, cluster_labels)
    ari = float(adjusted_rand_score(state_label_true, cluster_labels))
    nmi = float(normalized_mutual_info_score(state_label_true, cluster_labels))

    return {
        "centroid_acc": centroid_acc,
        "clust_acc_hungarian": clust_acc_hungarian,
        "ari": ari,
        "nmi": nmi,
        "pred_counts": np.bincount(pred_label, minlength=state_coords_true.shape[0]).tolist(),
        "true_counts": np.bincount(state_label_true, minlength=state_coords_true.shape[0]).tolist(),
        "aligned_std": aligned.std(axis=0).tolist(),
        "true_std": alpha_true.std(axis=0).tolist(),
    }


def compute_post_em_oracle_ceiling(cfg, ds, q, n_factor_steps=12):
    """Factor-only upper bound at the chosen q, with mu frozen at truth.

    Runs `n_factor_steps` closed-form M-steps starting at
    (mu_true, U_true) with mu **held at truth**. This is a strict upper
    bound: in the real joint loop mu is free to move, and under
    misspecification it moves away from mu_true toward the EM fixed
    point — see `compute_joint_loop_oracle_ceiling` for the ceiling
    actually achievable by the joint loop.

    Returns None on non-discrete datasets.
    """
    if ds.state_label_true is None or ds.state_coords_true is None:
        return None
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    cur = PPCAInit(
        mu=ds.mu_half_true.astype(jnp.complex128),
        U=ds.U_half_true.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )
    mu_fixed = cur.mu
    for _ in range(n_factor_steps):
        cur = update_factor_closed_form(
            cfg,
            cur,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
        )
        cur = PPCAInit(mu=mu_fixed, U=cur.U, s=cur.s, volume_shape=cur.volume_shape)
    return summarize_discrete_embedding(cfg, ds, cur)


def compute_joint_loop_oracle_ceiling(cfg, ds, q, n_joint_steps=30):
    """Joint-loop EM fixed point starting from (mu_true, U_true).

    This is the **actually reachable** oracle ceiling for the v0 joint
    loop: it runs `n_joint_steps` alternating mu/U updates starting at
    the ground truth and returns the final cluster metrics.

    Why this matters: on Ribosembly q=4 the factor-only ceiling reports
    Hungarian ≈ 0.79 but the joint loop from truth actually converges
    to Hungarian ≈ 0.72 — because the mu update drifts mu away from
    truth toward the PPCA ML fixed point, which is not the same as the
    data-generating process under model misspecification. The honest
    gap the joint loop can close is against this ceiling, not the
    factor-only one.
    """
    if ds.state_label_true is None or ds.state_coords_true is None:
        return None
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    cur = PPCAInit(
        mu=ds.mu_half_true.astype(jnp.complex128),
        U=ds.U_half_true.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )
    for _ in range(n_joint_steps):
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
    return summarize_discrete_embedding(cfg, ds, cur)


def compute_joint_loop_oracle_ceiling_annealed(
    cfg,
    ds,
    q,
    n_joint_steps=30,
    anneal_schedule="log1000",
    anneal_iters=30,
    anneal_factor_only=False,
):
    """Same as compute_joint_loop_oracle_ceiling but with an annealing schedule.

    This is the matched reference for annealed training runs.  Without this,
    "beats the non-annealed ceiling" is ambiguous — the annealed run follows
    a different update map than the non-annealed truth-start reference.
    """
    if ds.state_label_true is None or ds.state_coords_true is None:
        return None
    s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    cur = PPCAInit(
        mu=ds.mu_half_true.astype(jnp.complex128),
        U=ds.U_half_true.astype(jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )
    schedule = build_anneal_schedule(anneal_schedule, anneal_iters, n_joint_steps)
    for it in range(n_joint_steps):
        factor = float(schedule[it])
        nv_iter = ds.noise_variance_full * factor
        nv_mu = ds.noise_variance_full if anneal_factor_only else nv_iter
        mres = update_mu_residualized(
            cfg,
            cur,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            nv_mu,
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
            nv_iter,
        )
    return summarize_discrete_embedding(cfg, ds, cur)


def resolve_n_burnin(external_mode: str, n_burnin: int | None) -> int:
    """Default burn-in by benchmark mode.

    Continuous Gaussian-PC data benefits from a homogeneous mean warmup.
    Discrete GT-volume state mixtures usually do not: on Ribosembly, the
    homogeneous burn-in drifts the mean away from the best reachable
    fixed point and weakens latent-state separation. Keep the override
    explicit so benchmarking scripts can still force a non-default path.
    """
    if n_burnin is None:
        return 0 if external_mode == "discrete_volumes" else 10
    if n_burnin < 0:
        raise ValueError(f"n_burnin must be non-negative, got {n_burnin}")
    return n_burnin


# ---------------------------------------------------------------------------
# Two-stage loop: burn-in mu, SVD warm U, joint mean+factor
# ---------------------------------------------------------------------------


def build_anneal_schedule(kind: str, anneal_iters: int, total_iters: int) -> list:
    """Build a noise-variance multiplicative schedule for the joint loop.

    Returns a list of length `total_iters` where entry i is the factor
    applied to `noise_variance_full` at joint iteration i+1. The first
    `anneal_iters` entries ramp down to 1.0; the remainder stay at 1.0.

    Kinds:
      - 'none'      : constant 1.0 (no annealing)
      - 'linear50'  : linspace(50, 1, anneal_iters)
      - 'log100'    : logspace(2, 0, anneal_iters)
      - 'log1000'   : logspace(3, 0, anneal_iters)  — rescues U_random
                      from the lazy basin on Ribosembly q=8 (v11 finding).
    """
    if kind == "none":
        return [1.0] * total_iters
    n_anneal = min(max(anneal_iters, 1), total_iters)
    if kind == "linear50":
        anneal = np.linspace(50.0, 1.0, n_anneal).tolist()
    elif kind == "log100":
        anneal = np.logspace(2.0, 0.0, n_anneal).tolist()
    elif kind == "log1000":
        anneal = np.logspace(3.0, 0.0, n_anneal).tolist()
    else:
        raise ValueError(f"unknown anneal kind: {kind}")
    return anneal + [1.0] * (total_iters - n_anneal)


def _random_orthonormal_U(q, volume_shape, seed):
    """Generate q random orthonormal vectors in the half-volume subspace."""
    half_vol_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(q):
        noise = rng.standard_normal(2 * half_vol_size).view(np.complex128)
        noise_jax = jnp.asarray(noise, dtype=jnp.complex128)
        noise_jax = project_to_real_volume_subspace(noise_jax, volume_shape)
        rows.append(noise_jax)
    U_half = jnp.stack(rows)
    weights_v = make_half_volume_weights(volume_shape)
    return real_volume_orthonormalize_half(U_half, weights_v, int(np.prod(volume_shape)))


def run_two_stage(
    cfg,
    ds,
    q,
    n_burnin=10,
    n_joint=12,
    mu_init_kind="zero",
    mu_perturb_eps=0.5,
    seed=0,
    weighted_svd=False,
    anneal_schedule="none",
    anneal_iters=30,
    u_init_kind="svd",
    anneal_factor_only=False,
    update_eigenvalues=False,
    post_anneal_s_iters=0,
    s_init_kind="flat",
    ridge_mode="scalar",
    multiclass_K: int = 5,
    multiclass_iters: int = 50,
):
    """Two-stage loop with selectable mu init.

    `mu_init_kind`:
      - 'zero'      : mu = 0 (cold start; only works on easy data)
      - 'oracle'    : mu = mu_half_true (best case)
      - 'perturbed' : mu = mu_half_true + eps · noise (simulates the
                      output of a homogeneous ab-initio reconstruction)

    `anneal_schedule`: optional deterministic annealing of
    `noise_variance_full` over the first `anneal_iters` joint steps.
    See `build_anneal_schedule` for supported kinds. The `log1000`
    schedule rescues a random U init from the lazy basin on
    Ribosembly q=8 (v11 finding).
    """
    weights_v = make_half_volume_weights(cfg.volume_shape)
    half_vol_size = ds.mu_half_true.shape[0]
    if s_init_kind == "truth":
        s_kernel = jnp.maximum(ds.s_true, _S_FLOOR)
    else:
        s_kernel = jnp.ones(q, dtype=jnp.float64)
    # The slicer ignores voxels at r > max_r, so we only score the
    # part of mu the M-step can actually reach.
    slicer_max_r = cfg.volume_shape[0] // 2 - 1

    # Compute the "oracle fixed point": one mean update starting
    # from mu_true. This is what the Wiener filter projects mu_true
    # ONTO under the v0 nearest-disc forward+inverse pair, and is
    # the BEST result the loop can possibly achieve given the
    # discretization of the forward model.
    print("  computing oracle fixed point (one mean update from mu_true)...", flush=True)
    init_oracle = PPCAInit(
        mu=ds.mu_half_true.astype(jnp.complex128),
        U=jnp.zeros((q, half_vol_size), dtype=jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )
    mres_ofp = update_mu_homogeneous(
        cfg,
        init_oracle,
        ds.batch_full,
        ds.rotations,
        ds.translations,
        ds.ctf_params,
        ds.noise_variance_full,
        tau=0.0,
    )
    mu_oracle_fp = mres_ofp.mu_half
    jax.block_until_ready(mu_oracle_fp)
    fre_ofp_vs_truth = fre_sphere(mu_oracle_fp, ds.mu_half_true, cfg.volume_shape, weights_v, slicer_max_r)
    print(
        f"  oracle fixed point: fre vs mu_true = {fre_ofp_vs_truth:.4f} (this is the discretization-induced floor)",
        flush=True,
    )

    def _fre_truth(mu_est):
        return fre_sphere(mu_est, ds.mu_half_true, cfg.volume_shape, weights_v, slicer_max_r)

    def _fre_fp(mu_est):
        return fre_sphere(mu_est, mu_oracle_fp, cfg.volume_shape, weights_v, slicer_max_r)

    def _fre(mu_est):
        # Backward-compatible name for the joint loop's reporting:
        # we report BOTH metrics each iteration.
        return _fre_truth(mu_est)

    if mu_init_kind == "zero":
        mu_init = jnp.zeros(half_vol_size, dtype=jnp.complex128)
    elif mu_init_kind == "oracle":
        mu_init = ds.mu_half_true.astype(jnp.complex128)
    elif mu_init_kind == "perturbed":
        # Add complex Gaussian noise scaled to give relative error eps in
        # the half-volume Hermitian inner product.
        rng = np.random.default_rng(seed + 1)
        noise = rng.standard_normal(2 * half_vol_size).view(np.complex128)
        noise_jax = jnp.asarray(noise, dtype=jnp.complex128)
        noise_jax = project_to_real_volume_subspace(noise_jax, cfg.volume_shape)
        mu_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(ds.mu_half_true) * ds.mu_half_true))))
        noise_norm = float(jnp.sqrt(jnp.real(jnp.sum(weights_v * jnp.conj(noise_jax) * noise_jax))))
        scale = mu_perturb_eps * mu_norm / max(noise_norm, 1e-12)
        mu_init = ds.mu_half_true + scale * noise_jax
        mu_init = mu_init.astype(jnp.complex128)
    elif mu_init_kind == "multiclass":
        # Phase 7c: cold-μ rescue. Run K-class softness-annealed EM
        # from random blob volumes, return mean of class volumes as
        # μ_init. No GT used. See ppca_abinitio_phase6_stress_*.md and
        # cold_init.multiclass_mu_init.
        from recovar.em.ppca_abinitio.cold_init import multiclass_mu_init

        print(
            f"=== STAGE A0: cold-μ multiclass bootstrap (K={multiclass_K}, n_iters={multiclass_iters}) ===", flush=True
        )
        mu_init = multiclass_mu_init(
            cfg,
            ds,
            K=multiclass_K,
            n_iters=multiclass_iters,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown mu_init_kind: {mu_init_kind}")

    cur = PPCAInit(
        mu=mu_init,
        U=jnp.zeros((q, half_vol_size), dtype=jnp.complex128),
        s=s_kernel,
        volume_shape=tuple(int(x) for x in cfg.volume_shape),
    )

    fre_init_t = _fre_truth(cur.mu)
    fre_init_fp = _fre_fp(cur.mu)
    print(
        f"  mu init kind={mu_init_kind}: fre_truth={fre_init_t:.4f}, fre_fp={fre_init_fp:.4f}",
        flush=True,
    )

    print(f"=== STAGE A: homogeneous burn-in for mu ({n_burnin} iters) ===", flush=True)
    for it in range(1, n_burnin + 1):
        t0 = time.perf_counter()
        mres = update_mu_homogeneous(
            cfg, cur, ds.batch_full, ds.rotations, ds.translations, ds.ctf_params, ds.noise_variance_full, tau=0.0
        )
        cur = PPCAInit(mu=mres.mu_half, U=cur.U, s=cur.s, volume_shape=cur.volume_shape)
        jax.block_until_ready(cur.mu)
        ft = _fre_truth(cur.mu)
        ff = _fre_fp(cur.mu)
        print(
            f"  burn {it}: fre_truth={ft:.4f} fre_fp={ff:.4f} ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

    print(f"=== STAGE B: U init (kind={u_init_kind}) ===", flush=True)
    svd_singular_values = None
    if u_init_kind == "svd":
        U_init, svd_singular_values = init_U_from_residual_svd(
            cfg, ds, cur.mu, s_kernel=s_kernel, q=q, weighted=weighted_svd
        )
    elif u_init_kind == "random":
        U_init = _random_orthonormal_U(q, cfg.volume_shape, seed=seed + 42)
    elif u_init_kind == "zero":
        half_vol_size = ds.mu_half_true.shape[0]
        U_init = jnp.zeros((q, half_vol_size), dtype=jnp.complex128)
    else:
        raise ValueError(f"unknown u_init_kind: {u_init_kind}")

    if s_init_kind == "svd":
        if svd_singular_values is None:
            raise ValueError("--s-init=svd requires --u-init=svd")
        n_img = ds.batch_full.shape[0]
        s_kernel = jnp.array(svd_singular_values**2 / n_img, dtype=jnp.float64)
        print(f"  s init (svd): {[f'{v:.4g}' for v in s_kernel]}", flush=True)
    elif s_init_kind == "flat":
        s_kernel = jnp.ones(q, dtype=jnp.float64)
        print(f"  s init (flat): {[f'{v:.4g}' for v in s_kernel]}", flush=True)
    elif s_init_kind == "truth":
        print(f"  s init (truth): {[f'{v:.4g}' for v in s_kernel]}", flush=True)
    cur = PPCAInit(mu=cur.mu, U=U_init, s=s_kernel, volume_shape=cur.volume_shape)
    pe = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
    print(f"  U init: pe = {pe:.4f}", flush=True)

    print(f"=== STAGE C: joint mean+factor loop ({n_joint} iters) ===", flush=True)
    schedule = build_anneal_schedule(anneal_schedule, anneal_iters, n_joint)
    if anneal_schedule != "none":
        print(
            f"  anneal: schedule='{anneal_schedule}' anneal_iters={anneal_iters} "
            f"(factors {schedule[0]:.1f} → {schedule[-1]:.1f}, n_anneal={min(anneal_iters, n_joint)})",
            flush=True,
        )
    pe_traj = [pe]
    fre_truth_traj = [_fre_truth(cur.mu)]
    fre_fp_traj = [_fre_fp(cur.mu)]
    lm_traj = [_log_marginal_sum(cfg, cur, ds)]
    print(f"  warm: log_marginal={lm_traj[0]:.2f}", flush=True)
    for it in range(1, n_joint + 1):
        t0 = time.perf_counter()
        factor = float(schedule[it - 1])
        nv_iter = ds.noise_variance_full * factor
        nv_mu = ds.noise_variance_full if anneal_factor_only else nv_iter
        mres = update_mu_residualized(
            cfg, cur, ds.batch_full, ds.rotations, ds.translations, ds.ctf_params, nv_mu, tau=0.0
        )
        cur = PPCAInit(mu=mres.mu_half, U=cur.U, s=cur.s, volume_shape=cur.volume_shape)
        W_prior = compute_W_prior_half(cur.U, cur.s, cur.volume_shape) if ridge_mode == "w_prior" else None
        cur = update_factor_closed_form(
            cfg,
            cur,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            nv_iter,
            W_prior=W_prior,
            update_s=update_eigenvalues,
        )
        jax.block_until_ready(cur.U)
        ft = _fre_truth(cur.mu)
        ff = _fre_fp(cur.mu)
        pe = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
        lm = _log_marginal_sum(cfg, cur, ds)
        fre_truth_traj.append(ft)
        fre_fp_traj.append(ff)
        pe_traj.append(pe)
        lm_traj.append(lm)
        dlm = lm - lm_traj[-2]
        anneal_tag = f" f={factor:.2f}" if anneal_schedule != "none" else ""
        s_tag = f" s=[{', '.join(f'{v:.3g}' for v in cur.s)}]" if update_eigenvalues else ""
        print(
            f"  joint {it}{anneal_tag}: fre_truth={ft:.4f} fre_fp={ff:.4f} pe={pe:.4f} lm={lm:.2f} dlm={dlm:+.2f}{s_tag} ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

    # -----------------------------------------------------------------
    # Post-annealing eigenvalue refinement (optional)
    # -----------------------------------------------------------------
    if post_anneal_s_iters > 0:
        print(
            f"=== POST-ANNEAL S REFINEMENT ({post_anneal_s_iters} iters at f=1) ===",
            flush=True,
        )
        for it in range(1, post_anneal_s_iters + 1):
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
            W_prior = compute_W_prior_half(cur.U, cur.s, cur.volume_shape) if ridge_mode == "w_prior" else None
            cur = update_factor_closed_form(
                cfg,
                cur,
                ds.batch_full,
                ds.rotations,
                ds.translations,
                ds.ctf_params,
                ds.noise_variance_full,
                W_prior=W_prior,
                update_s=True,
            )
            jax.block_until_ready(cur.U)
            ft = _fre_truth(cur.mu)
            ff = _fre_fp(cur.mu)
            pe = float(projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape))
            lm = _log_marginal_sum(cfg, cur, ds)
            fre_truth_traj.append(ft)
            fre_fp_traj.append(ff)
            pe_traj.append(pe)
            lm_traj.append(lm)
            dlm = lm - lm_traj[-2]
            s_tag = f" s=[{', '.join(f'{v:.3g}' for v in cur.s)}]"
            print(
                f"  refine {it}: fre_truth={ft:.4f} fre_fp={ff:.4f} pe={pe:.4f} lm={lm:.2f} dlm={dlm:+.2f}{s_tag} ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    return cur, fre_truth_traj, fre_fp_traj, pe_traj, fre_ofp_vs_truth, lm_traj


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="IgG-1D")
    ap.add_argument("--vol", type=int, default=32)
    ap.add_argument("--n-images", type=int, default=4096, help="total number of synthetic images")
    ap.add_argument("--sigma", type=float, default=0.1, help="real-space white-noise std for image synthesis")
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--healpix-order", type=int, default=1)
    ap.add_argument(
        "--n-burnin",
        type=int,
        default=None,
        help="homogeneous burn-in iterations; default is mode-dependent (0 for discrete_volumes, 10 otherwise)",
    )
    ap.add_argument(
        "--n-joint",
        type=int,
        default=30,
        help="Joint mu+U EM iterations. On Ribosembly, q=4 plateaus by ~30 "
        "iters; q=8 needs ~100 to reach the joint-loop ceiling. Lower values "
        "underestimate the method.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Dataset construction seed.")
    ap.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Inference init seed (mu perturbation). Defaults to --seed. "
        "Vary this with --seed fixed to do apples-to-apples multi-restart "
        "on the same dataset. Ignored when --n-restarts > 1.",
    )
    ap.add_argument(
        "--n-restarts",
        type=int,
        default=1,
        help="Number of mu-init restarts on the same dataset. With "
        "--n-restarts K, the script runs K inferences with init seeds "
        "{init_seed, init_seed+1, ..., init_seed+K-1} and selects the "
        "restart with the highest best-over-loop log_marginal. On "
        "Ribosembly q=4 the bad 'collapsed-2-cluster' basin sits ~3M "
        "below the good basins in log_marginal — argmax(lm) cleanly "
        "rejects it. Use K>=4 for safety.",
    )
    ap.add_argument(
        "--mu-init",
        choices=["zero", "oracle", "perturbed", "multiclass"],
        default="perturbed",
        help="mu init: 'perturbed' simulates a homogeneous-ab-initio output (recommended). "
        "'multiclass' (Phase 7c, cold-μ rescue) bootstraps μ from K-class softness-annealed EM "
        "starting from random Gaussian blob volumes. Adds a pre-stage of "
        "--multiclass-iters iterations before the main run_two_stage loop. "
        "Does NOT use any GT field. Test against --mu-init zero to see the lift.",
    )
    ap.add_argument("--mu-perturb-eps", type=float, default=0.5)
    ap.add_argument(
        "--multiclass-K",
        type=int,
        default=5,
        help="Number of classes for --mu-init multiclass. Memory says K ≈ n_states or K > q.",
    )
    ap.add_argument(
        "--multiclass-iters",
        type=int,
        default=50,
        help="Multi-class EM iterations for --mu-init multiclass. ~30-50 typical.",
    )
    ap.add_argument(
        "--external-mode",
        choices=["gaussian_pc", "discrete_volumes"],
        default="discrete_volumes",
        help="How to sample images from external GT volumes. "
        "'discrete_volumes' generates each image from an actual GT volume (recommended).",
    )
    ap.add_argument(
        "--u-init",
        choices=["svd", "random", "zero"],
        default="svd",
        help="How to initialize U. 'svd' (default) is the residual-SVD warmstart "
        "(weighted by default — see --svd-warmstart). On Ribosembly q=4 it "
        "reaches the post-EM oracle ceiling (hun≈0.78). 'random' needs log1000 "
        "annealing to avoid the lazy basin. 'zero' starts from U=0.",
    )
    ap.add_argument(
        "--svd-warmstart",
        choices=["weighted", "unweighted"],
        default="weighted",
        help="Only used when --u-init=svd. 'weighted' (default) projects each "
        "residual to the real-volume subspace and applies sqrt(half-vol Hermitian "
        "weights) before SVD so the SVD ℓ² matches the real-space ℓ². On "
        "Ribosembly q=4 the weighted variant gives hun≈0.78 vs hun≈0.62 for "
        "the old unweighted default (diag_warmstart_compare, 20260414). "
        "'unweighted' keeps the legacy raw-half-vol SVD for reproducing old runs.",
    )
    ap.add_argument(
        "--anneal-schedule",
        choices=["none", "linear50", "log100", "log1000"],
        default="none",
        help="Deterministic annealing of noise_variance over the first "
        "--anneal-iters joint steps. 'log1000' (logspace(3,0,anneal_iters)) "
        "rescues a random or SVD U init from the lazy basin on Ribosembly "
        "q=8 (diag_cluster_bootstrap_v11 finding). "
        "PHASE 6.5 GUIDANCE (2026-04-25, see ppca_abinitio_phase6_stress_*.md): "
        "factor-only-log1000 anneal is σ-CONDITIONAL — RESCUES Ribo q=8 σ=0.001 "
        "(+0.19 hun) and Ribo q=8 σ=0.01 with warm μ (+0.17 hun, Phase 1), "
        "but HURTS at σ ≥ 0.3 (-0.04 to -0.06 hun) and is NEUTRAL on cold-μ. "
        "Recommended: enable for σ < 0.005 OR Ribosembly-like discrete q ≥ 8 "
        "with warm μ. Do NOT enable at σ ≥ 0.3 or with cold μ.",
    )
    ap.add_argument(
        "--anneal-iters",
        type=int,
        default=30,
        help="Number of joint iterations over which to anneal noise_variance. "
        "Ignored when --anneal-schedule is 'none'. Typically set equal to --n-joint.",
    )
    ap.add_argument(
        "--anneal-factor-only",
        action="store_true",
        default=True,
        help="(Default) Anneal noise_variance only for the factor M-step; "
        "the mean M-step always uses the real noise_variance.",
    )
    ap.add_argument(
        "--anneal-mu-too",
        action="store_true",
        help="Also anneal the mean update (old behavior). Harmful on "
        "continuous manifolds like IgG-RL where it causes FRE divergence.",
    )
    ap.add_argument(
        "--s-init",
        choices=["truth", "flat", "svd"],
        default="flat",
        help="How to initialize eigenvalues s. 'flat' (default) sets s=1 for all "
        "components — the prior is negligible at cryo-EM SNR so s doesn't affect "
        "the EM trajectory (validated on Ribosembly, IgG-1D, IgG-RL). 'truth' uses "
        "ground-truth eigenvalues (cheating baseline). 'svd' uses sample covariance "
        "eigenvalues from the residual SVD (requires --u-init=svd).",
    )
    ap.add_argument(
        "--update-eigenvalues",
        action="store_true",
        help="Estimate eigenvalues s from the E-step posterior moments "
        "(Tipping-Bishop update) instead of freezing them at the GT values. "
        "Also enables joint orthonormalization of U with s update.",
    )
    ap.add_argument(
        "--post-anneal-s-iters",
        type=int,
        default=0,
        help="After the main joint loop completes, run this many additional "
        "iterations at f=1 with eigenvalue estimation enabled. This avoids "
        "the annealing-induced eigenvalue inflation while still allowing s "
        "to converge to the ML estimate. 0 = disabled (default).",
    )
    ap.add_argument(
        "--ridge-mode",
        choices=["scalar", "w_prior"],
        default="scalar",
        help="Per-voxel M-step regularization. 'scalar' (default) uses a "
        "fixed ridge_lambda * I_q. 'w_prior' uses a Wiener-style "
        "shell-stratified prior R_v = diag(1/W_v), where W_v is the "
        "radial-shell average of |U|^2 * s. Regularization only — does "
        "NOT estimate eigenvalues.",
    )
    ap.add_argument(
        "--post-eigenvalue-refit",
        choices=["none", "projcov"],
        default="none",
        help="One-shot post-EM eigenvalue calibration. 'none' (default) "
        "leaves the EM s as-is. 'projcov' runs the posterior-covariance "
        "refit from recovar.em.ppca_abinitio.eigenvalue_refit (one E-step "
        "at f=1, eigendecompose the sample-averaged posterior covariance, "
        "rotate U accordingly). Does NOT propagate refit s back into EM.",
    )
    ap.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="If set, dump final metrics + trajectories + cell config to "
        "this JSON path. Used by the Phase 1 ablation sweep driver.",
    )
    ap.add_argument(
        "--instrument",
        action="store_true",
        help="Phase 3 instrumentation: print per-iter peak GPU memory and "
        "predicted memory model breakdown. Records to the --save-results "
        "JSON under 'instrumentation' if both flags are set. Adds modest "
        "overhead from jax.live_arrays() introspection.",
    )
    args = ap.parse_args()
    if args.anneal_mu_too:
        args.anneal_factor_only = False
    init_seed = args.init_seed if args.init_seed is not None else args.seed
    n_burnin = resolve_n_burnin(args.external_mode, args.n_burnin)

    root = Path("/home/mg6942/mytigress/cryobench2") / args.dataset
    print(
        f"### dataset={args.dataset} vol={args.vol} n_images={args.n_images} sigma={args.sigma} "
        f"q={args.q} external_mode={args.external_mode} n_burnin={n_burnin} ###",
        flush=True,
    )
    if args.external_mode == "discrete_volumes" and n_burnin > 0:
        print(
            "  warning: homogeneous burn-in is usually harmful in discrete_volumes mode; "
            "continuing because --n-burnin was set explicitly.",
            flush=True,
        )

    print("loading cryobench gt volumes...", flush=True)
    gt_vols = load_cryobench_gt_volumes(root, target_D=args.vol)

    print("building synthetic dataset (cryobench gt vols + v0 forward model)...", flush=True)
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
        external_sampling_mode=args.external_mode,
    )
    if args.s_init == "truth":
        print(f"  s_true (empirical from gt vols): {np.asarray(ds.s_true)}", flush=True)
    else:
        print(f"  s_init={args.s_init} (s_true from GT not used in algorithmic path)", flush=True)
    print(f"  n_rot={ds.n_rot}, n_trans={ds.n_trans}, n_img={ds.n_img}", flush=True)
    if ds.state_coords_true is not None:
        n_states = int(ds.state_coords_true.shape[0])
        print(f"  n_external_states={n_states}", flush=True)
        if args.external_mode == "discrete_volumes" and args.q < max(2, n_states // 4):
            print(
                f"  WARNING: q={args.q} is much smaller than n_states={n_states}. The PPCA model "
                f"is misspecified for this many discrete conformations and clustering quality "
                f"will be capped well below the pre-EM oracle. Consider --q {max(4, n_states // 4)}.",
                flush=True,
            )

    cfg = _Cfg(image_shape=image_shape, volume_shape=volume_shape, voxel_size=1.0)
    print()

    if args.n_restarts <= 1:
        print(f"  data seed={args.seed}, init seed={init_seed}", flush=True)
        final_init, fre_truth_traj, fre_fp_traj, pe_traj, fre_floor, lm_traj = run_two_stage(
            cfg,
            ds,
            q=args.q,
            n_burnin=n_burnin,
            n_joint=args.n_joint,
            mu_init_kind=args.mu_init,
            mu_perturb_eps=args.mu_perturb_eps,
            seed=init_seed,
            weighted_svd=(args.svd_warmstart == "weighted"),
            anneal_schedule=args.anneal_schedule,
            anneal_iters=args.anneal_iters,
            u_init_kind=args.u_init,
            anneal_factor_only=args.anneal_factor_only,
            update_eigenvalues=args.update_eigenvalues,
            post_anneal_s_iters=args.post_anneal_s_iters,
            s_init_kind=args.s_init,
            ridge_mode=args.ridge_mode,
            multiclass_K=args.multiclass_K,
            multiclass_iters=args.multiclass_iters,
        )
    else:
        print(
            f"  data seed={args.seed}, multi-restart over init seeds {init_seed}..{init_seed + args.n_restarts - 1}",
            flush=True,
        )
        restart_results = []
        fre_floor = None
        for k in range(args.n_restarts):
            init_seed_k = init_seed + k
            print()
            print(f"=== RESTART {k + 1}/{args.n_restarts} (init seed {init_seed_k}) ===", flush=True)
            init_k, fre_t_k, fre_f_k, pe_k, fre_floor_k, lm_k = run_two_stage(
                cfg,
                ds,
                q=args.q,
                n_burnin=n_burnin,
                n_joint=args.n_joint,
                mu_init_kind=args.mu_init,
                mu_perturb_eps=args.mu_perturb_eps,
                seed=init_seed_k,
                weighted_svd=(args.svd_warmstart == "weighted"),
                anneal_schedule=args.anneal_schedule,
                anneal_iters=args.anneal_iters,
                u_init_kind=args.u_init,
                anneal_factor_only=args.anneal_factor_only,
                update_eigenvalues=args.update_eigenvalues,
                post_anneal_s_iters=args.post_anneal_s_iters,
                s_init_kind=args.s_init,
                ridge_mode=args.ridge_mode,
                multiclass_K=args.multiclass_K,
                multiclass_iters=args.multiclass_iters,
            )
            if fre_floor is None:
                fre_floor = fre_floor_k
            restart_results.append(
                {
                    "init_seed": init_seed_k,
                    "init": init_k,
                    "fre_truth_traj": fre_t_k,
                    "fre_fp_traj": fre_f_k,
                    "pe_traj": pe_k,
                    "lm_traj": lm_k,
                    "best_lm": max(lm_k),
                }
            )
        print()
        print("=== MULTI-RESTART SELECTION (argmax best log_marginal) ===", flush=True)
        for r in restart_results:
            disc = summarize_discrete_embedding(cfg, ds, r["init"])
            hun = disc["clust_acc_hungarian"] if disc is not None else float("nan")
            print(
                f"  init seed {r['init_seed']}: best_lm={r['best_lm']:.2f} hun={hun:.4f}",
                flush=True,
            )
        best_idx = int(np.argmax([r["best_lm"] for r in restart_results]))
        chosen = restart_results[best_idx]
        print(
            f"  selected: init seed {chosen['init_seed']} with best_lm={chosen['best_lm']:.2f}",
            flush=True,
        )
        final_init = chosen["init"]
        fre_truth_traj = chosen["fre_truth_traj"]
        fre_fp_traj = chosen["fre_fp_traj"]
        pe_traj = chosen["pe_traj"]
        lm_traj = chosen["lm_traj"]

    # -------------------------------------------------------------------
    # Phase 3: optional instrumentation — predicted memory model
    # -------------------------------------------------------------------
    instrumentation = None
    if args.instrument:
        from recovar.em.ppca_abinitio.memory_model import (
            estimate_peak_memory_bytes,
            format_memory_report,
        )

        n_rot_eff = int(ds.rotations.shape[0])
        n_trans_eff = int(ds.translations.shape[0])
        print()
        print("=== PHASE 3 INSTRUMENTATION ===", flush=True)
        print(
            format_memory_report(
                n_img=ds.batch_full.shape[0],
                volume_shape=tuple(int(x) for x in cfg.volume_shape),
                image_shape=tuple(int(x) for x in cfg.image_shape),
                n_rot=n_rot_eff,
                n_trans=n_trans_eff,
                q=args.q,
            ),
            flush=True,
        )
        # Final-iter pose entropy: run one extra E-step at f=1, compute
        # mean per-image entropy of γ over (r, t).
        post_em_stats = score_and_posterior_moments_eqx(
            cfg,
            final_init.mu,
            final_init.U,
            final_init.s,
            ds.batch_full,
            ds.rotations,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
        )
        log_resp = np.asarray(post_em_stats.log_resp)  # (n_img, n_rot, n_trans)
        gamma = np.exp(log_resp)
        # H(γ_i) = -Σ_{r,t} γ_{i,r,t} log γ_{i,r,t}, computed in a way that
        # tolerates numerical zeros.
        flat = gamma.reshape(gamma.shape[0], -1)
        with np.errstate(invalid="ignore", divide="ignore"):
            entropy_per_img = -np.sum(np.where(flat > 0, flat * np.log(flat), 0.0), axis=-1)
        max_entropy = np.log(flat.shape[-1])
        # Effective pose count per image: 1 / max(gamma), clamped at n_pose.
        eff_pose_count = 1.0 / np.maximum(np.max(flat, axis=-1), 1.0 / flat.shape[-1])
        # GPU memory if available.
        try:
            mem_stats = jax.devices()[0].memory_stats()
            peak_mem_gb = mem_stats.get("peak_bytes_in_use", 0) / (1024**3)
        except Exception:
            peak_mem_gb = None

        memory_components = estimate_peak_memory_bytes(
            n_img=ds.batch_full.shape[0],
            volume_shape=tuple(int(x) for x in cfg.volume_shape),
            image_shape=tuple(int(x) for x in cfg.image_shape),
            n_rot=n_rot_eff,
            n_trans=n_trans_eff,
            q=args.q,
        )
        instrumentation = {
            "memory_predicted_gb": memory_components["total"] / (1024**3),
            "memory_components_bytes": {k: int(v) for k, v in memory_components.items()},
            "memory_measured_peak_gb": peak_mem_gb,
            "pose_entropy_mean_nats": float(entropy_per_img.mean()),
            "pose_entropy_max_nats": float(max_entropy),
            "pose_entropy_normalized": float(entropy_per_img.mean() / max(max_entropy, 1e-9)),
            "effective_pose_count_mean": float(eff_pose_count.mean()),
            "effective_pose_count_p10": float(np.percentile(eff_pose_count, 10)),
            "effective_pose_count_p90": float(np.percentile(eff_pose_count, 90)),
            "n_pose_total": int(flat.shape[-1]),
        }
        print(
            f"  pose entropy:           {instrumentation['pose_entropy_mean_nats']:.3f} / "
            f"{max_entropy:.3f} nats "
            f"(normalized: {instrumentation['pose_entropy_normalized']:.3f})",
            flush=True,
        )
        print(
            f"  effective pose count:   p10={instrumentation['effective_pose_count_p10']:.1f}, "
            f"mean={instrumentation['effective_pose_count_mean']:.1f}, "
            f"p90={instrumentation['effective_pose_count_p90']:.1f} "
            f"(of {instrumentation['n_pose_total']} grid poses)",
            flush=True,
        )
        if peak_mem_gb is not None:
            print(f"  GPU peak memory:        {peak_mem_gb:.2f} GB", flush=True)

    # -------------------------------------------------------------------
    # Phase 2: optional post-EM eigenvalue refit (ProjCov-style)
    # -------------------------------------------------------------------
    refit_info = None
    if args.post_eigenvalue_refit == "projcov":
        from recovar.em.ppca_abinitio.eigenvalue_refit import (
            refit_eigenvalues_post_em,
        )

        print()
        print("=== POST-EM EIGENVALUE REFIT (ProjCov, one-shot at f=1) ===", flush=True)
        final_init, refit_info = refit_eigenvalues_post_em(final_init, cfg, ds)
        print(f"  s_em:    {[f'{v:.4g}' for v in refit_info.s_em]}", flush=True)
        print(f"  s_refit: {[f'{v:.4g}' for v in refit_info.s_refit]}", flush=True)

    print()
    print("=== FINAL ===", flush=True)
    print(f"  discretization floor (fre vs truth at oracle fixed point): {fre_floor:.4f}", flush=True)
    print(f"  best fre vs truth over loop:    {min(fre_truth_traj):.4f}", flush=True)
    print(f"  best fre vs oracle FP over loop: {min(fre_fp_traj):.4f}", flush=True)
    print(f"  best pe over loop:               {min(pe_traj):.4f}", flush=True)
    print(f"  best log_marginal over loop:     {max(lm_traj):.2f}", flush=True)
    print(f"  final log_marginal:              {lm_traj[-1]:.2f}", flush=True)
    discrete_summary = summarize_discrete_embedding(cfg, ds, final_init)
    if discrete_summary is not None:
        print(f"  discrete centroid acc:            {discrete_summary['centroid_acc']:.4f}", flush=True)
        print(f"  discrete clust acc (Hungarian):   {discrete_summary['clust_acc_hungarian']:.4f}", flush=True)
        print(f"  discrete ARI:                     {discrete_summary['ari']:.4f}", flush=True)
        print(f"  discrete NMI:                     {discrete_summary['nmi']:.4f}", flush=True)
        print(f"  pred state counts:                {discrete_summary['pred_counts']}", flush=True)
        print(f"  true state counts:                {discrete_summary['true_counts']}", flush=True)
        print(f"  aligned latent std:               {discrete_summary['aligned_std']}", flush=True)
        print(f"  true latent std:                  {discrete_summary['true_std']}", flush=True)

        # Two ceilings:
        #   factor-only: mu frozen at truth, factor-only M-steps (loose upper bound)
        #   joint-loop : actual EM fixed point from (mu_true, U_true), joint mu+U
        # Under PPCA misspecification these differ. The joint-loop ceiling is
        # the honest reachable target for this loop.
        print()
        print("  computing factor-only oracle ceiling (12 factor M-steps, mu frozen at truth)...", flush=True)
        oracle_ceil = compute_post_em_oracle_ceiling(cfg, ds, q=args.q, n_factor_steps=12)
        if oracle_ceil is not None:
            print(f"  factor-only ceiling (q={args.q}, loose):", flush=True)
            print(f"    centroid acc:           {oracle_ceil['centroid_acc']:.4f}", flush=True)
            print(f"    clust acc (Hungarian):  {oracle_ceil['clust_acc_hungarian']:.4f}", flush=True)
            print(f"    ARI:                    {oracle_ceil['ari']:.4f}", flush=True)
            print(f"    NMI:                    {oracle_ceil['nmi']:.4f}", flush=True)

        print()
        print(
            f"  computing joint-loop oracle ceiling ({args.n_joint} joint steps from (mu_true, U_true))...", flush=True
        )
        joint_ceil = compute_joint_loop_oracle_ceiling(cfg, ds, q=args.q, n_joint_steps=args.n_joint)
        if joint_ceil is not None:
            print(f"  joint-loop ceiling (q={args.q}, honest reachable):", flush=True)
            print(f"    centroid acc:           {joint_ceil['centroid_acc']:.4f}", flush=True)
            print(f"    clust acc (Hungarian):  {joint_ceil['clust_acc_hungarian']:.4f}", flush=True)
            print(f"    ARI:                    {joint_ceil['ari']:.4f}", flush=True)
            print(f"    NMI:                    {joint_ceil['nmi']:.4f}", flush=True)
            gap_cac = joint_ceil["centroid_acc"] - discrete_summary["centroid_acc"]
            gap_hun = joint_ceil["clust_acc_hungarian"] - discrete_summary["clust_acc_hungarian"]
            gap_ari = joint_ceil["ari"] - discrete_summary["ari"]
            print(
                f"  remaining gap to joint-loop ceiling: cac={gap_cac:+.4f} hun={gap_hun:+.4f} ari={gap_ari:+.4f}",
                flush=True,
            )

        if args.anneal_schedule != "none":
            print()
            print(
                f"  computing annealed oracle ceiling ({args.n_joint} joint steps from truth, "
                f"schedule={args.anneal_schedule})...",
                flush=True,
            )
            anneal_ceil = compute_joint_loop_oracle_ceiling_annealed(
                cfg,
                ds,
                q=args.q,
                n_joint_steps=args.n_joint,
                anneal_schedule=args.anneal_schedule,
                anneal_iters=args.anneal_iters,
                anneal_factor_only=args.anneal_factor_only,
            )
            if anneal_ceil is not None:
                print(f"  annealed oracle ceiling (q={args.q}, matched reference):", flush=True)
                print(f"    centroid acc:           {anneal_ceil['centroid_acc']:.4f}", flush=True)
                print(f"    clust acc (Hungarian):  {anneal_ceil['clust_acc_hungarian']:.4f}", flush=True)
                print(f"    ARI:                    {anneal_ceil['ari']:.4f}", flush=True)
                print(f"    NMI:                    {anneal_ceil['nmi']:.4f}", flush=True)
                a_gap_hun = anneal_ceil["clust_acc_hungarian"] - discrete_summary["clust_acc_hungarian"]
                a_gap_ari = anneal_ceil["ari"] - discrete_summary["ari"]
                print(
                    f"  gap to annealed ceiling: hun={a_gap_hun:+.4f} ari={a_gap_ari:+.4f}",
                    flush=True,
                )

    # -------------------------------------------------------------------
    # Optional JSON dump for sweep aggregation
    # -------------------------------------------------------------------
    if args.save_results is not None:
        import json

        def _to_pylist(x):
            arr = np.asarray(x)
            return arr.astype(np.float64).tolist()

        results = {
            "config": {
                "dataset": args.dataset,
                "vol": args.vol,
                "n_images": args.n_images,
                "sigma": args.sigma,
                "q": args.q,
                "n_burnin": n_burnin,
                "n_joint": args.n_joint,
                "u_init": args.u_init,
                "svd_warmstart": args.svd_warmstart,
                "anneal_schedule": args.anneal_schedule,
                "anneal_iters": args.anneal_iters,
                "anneal_factor_only": args.anneal_factor_only,
                "s_init": args.s_init,
                "ridge_mode": args.ridge_mode,
                "update_eigenvalues": args.update_eigenvalues,
                "post_anneal_s_iters": args.post_anneal_s_iters,
                "post_eigenvalue_refit": args.post_eigenvalue_refit,
                "external_mode": args.external_mode,
                "n_restarts": args.n_restarts,
                "data_seed": args.seed,
                "init_seed": init_seed,
                "healpix_order": args.healpix_order,
            },
            "trajectories": {
                "fre_truth": _to_pylist(fre_truth_traj),
                "fre_fp": _to_pylist(fre_fp_traj),
                "pe": _to_pylist(pe_traj),
                "lm": _to_pylist(lm_traj),
            },
            "metrics": {
                "fre_floor": float(fre_floor),
                "best_fre_truth": float(min(fre_truth_traj)),
                "best_fre_fp": float(min(fre_fp_traj)),
                "best_pe": float(min(pe_traj)),
                "best_lm": float(max(lm_traj)),
                "final_lm": float(lm_traj[-1]),
            },
        }
        if refit_info is not None:
            results["eigenvalue_refit"] = {
                "s_em": refit_info.s_em.astype(np.float64).tolist(),
                "s_refit": refit_info.s_refit.astype(np.float64).tolist(),
            }
        if instrumentation is not None:
            results["instrumentation"] = instrumentation
        if discrete_summary is not None:
            results["metrics"]["centroid_acc"] = float(discrete_summary["centroid_acc"])
            results["metrics"]["clust_acc_hungarian"] = float(discrete_summary["clust_acc_hungarian"])
            results["metrics"]["ari"] = float(discrete_summary["ari"])
            results["metrics"]["nmi"] = float(discrete_summary["nmi"])
        # Ceilings (may be None on non-discrete datasets)
        try:
            if oracle_ceil is not None:
                results["ceilings"] = {
                    "factor_only_hun": float(oracle_ceil["clust_acc_hungarian"]),
                    "factor_only_ari": float(oracle_ceil["ari"]),
                    "factor_only_nmi": float(oracle_ceil["nmi"]),
                }
        except NameError:
            pass
        try:
            if joint_ceil is not None:
                results.setdefault("ceilings", {})
                results["ceilings"]["joint_loop_hun"] = float(joint_ceil["clust_acc_hungarian"])
                results["ceilings"]["joint_loop_ari"] = float(joint_ceil["ari"])
                results["ceilings"]["joint_loop_nmi"] = float(joint_ceil["nmi"])
        except NameError:
            pass
        try:
            if args.anneal_schedule != "none" and anneal_ceil is not None:
                results.setdefault("ceilings", {})
                results["ceilings"]["annealed_hun"] = float(anneal_ceil["clust_acc_hungarian"])
                results["ceilings"]["annealed_ari"] = float(anneal_ceil["ari"])
                results["ceilings"]["annealed_nmi"] = float(anneal_ceil["nmi"])
        except NameError:
            pass

        out_path = Path(args.save_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"  saved results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
