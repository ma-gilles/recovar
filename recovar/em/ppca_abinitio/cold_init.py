"""Cold-μ rescue via K-class bootstrap (Phase 7c).

The Phase 6 stress sweep showed v0 is mu-init-bound: zero-μ + svd-U
collapses to hun ≈ 0.29 on Ribosembly q=4 (vs 0.79 with perturbed-μ).
σ²-annealing does not rescue cold-μ on any of the three datasets.

This module provides a μ-bootstrapping pre-stage that runs entirely
from random Gaussian-blob initial volumes and produces a non-trivial
μ_init for the standard `run_two_stage` loop. The recipe is the
"multiclass breakthrough" from the prototype scripts:

  1. K random blob volumes from independent seeds
  2. Frequency-marched K-class softness-annealed EM (~50 iters):
       E-step: factored class+pose posterior with class softness
               annealed from 1.0 (uniform) to 0.0 (sharp)
       M-step: per-class Wiener reconstruction, dead-class protection
  3. Return mean-of-class-means as μ_init

The standard `run_two_stage` then runs SVD warmstart against this
μ_init and a normal EM loop on top.

References:
- Original prototype: scripts/ppca_abinitio/run_abinitio_multiclass.py
- Memory: project_ppca_multiclass_abinitio_works.md
  ("hun=0.557 on Ribosembly from zero prior knowledge")
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu

from .half_volume import radial_band_limit_half
from .mean_update import (
    _backproject_to_half_volume,
    _per_rotation_ctf_image,
    _per_rotation_residual_image,
    _solve_wiener,
)
from .posterior import (
    _homogeneous_residual_half,
    _preprocess_batch_to_half,
    _slice_mu_half,
    make_half_image_weights,
)


def init_random_blobs(
    volume_shape: tuple,
    n_blobs: int = 40,
    blob_sigma_frac: float = 0.08,
    seed: int = 0,
    k_max: float | None = None,
) -> jnp.ndarray:
    """Random density map as a sum of Gaussian blobs (RELION-style).

    Returns a half-volume rfft layout array (complex128).
    """
    rng = np.random.default_rng(seed)
    N = volume_shape[0]
    blob_sigma = blob_sigma_frac * N

    vol = np.zeros(volume_shape, dtype=np.float64)
    coords = np.mgrid[0:N, 0:N, 0:N].astype(np.float64)
    for _ in range(n_blobs):
        center = rng.uniform(N * 0.15, N * 0.85, size=3)
        amp = rng.exponential(1.0)
        r2 = sum((coords[d] - center[d]) ** 2 for d in range(3))
        vol += amp * np.exp(-r2 / (2 * blob_sigma**2))
    vol -= vol.mean()

    mu_half = ftu.get_dft3_real(jnp.asarray(vol)).reshape(-1).astype(jnp.complex128)
    if k_max is not None:
        mu_half = radial_band_limit_half(mu_half[None], volume_shape, k_max)[0]
    return mu_half


def _freq_march_schedule(total_iters, k_max_start, k_max_end, interval):
    schedule = []
    for it in range(total_iters):
        k = min(k_max_start + (it // interval), k_max_end)
        schedule.append(k)
    return schedule


def _multiclass_estep(
    cfg,
    mus_half,
    rotations,
    shifted_half,
    ctf2_over_nv_half,
    weights_half,
    class_softness: float = 0.0,
):
    """Factored class+pose posterior with optional class softening.

    Returns
    -------
    log_resps : list of K arrays (n_img, n_rot, n_trans)
    class_probs : (n_img, K)
    """
    K = len(mus_half)
    n_img = shifted_half.shape[0]

    all_log_scores = []
    for k in range(K):
        mean_proj_half_k = _slice_mu_half(mus_half[k], rotations, cfg.image_shape, cfg.volume_shape).astype(
            jnp.complex128
        )
        homog_k = _homogeneous_residual_half(mean_proj_half_k, shifted_half, ctf2_over_nv_half, weights_half)
        all_log_scores.append(-0.5 * homog_k)

    stacked = jnp.stack(all_log_scores, axis=1)  # (n_img, K, n_rot, n_trans)
    within_class = stacked.reshape(n_img, K, -1)
    log_pose_norm = jax.scipy.special.logsumexp(within_class, axis=-1, keepdims=True)
    log_pose_posterior = within_class - log_pose_norm

    class_lm = log_pose_norm.squeeze(-1)
    tempered_lm = (1.0 - class_softness) * class_lm
    class_log_norm = jax.scipy.special.logsumexp(tempered_lm, axis=-1, keepdims=True)
    log_class_posterior = tempered_lm - class_log_norm

    log_resp_all = log_class_posterior[:, :, None] + log_pose_posterior
    log_resp_all = log_resp_all.reshape(stacked.shape)
    log_resps = [log_resp_all[:, k] for k in range(K)]
    class_probs = jnp.exp(log_class_posterior)
    return log_resps, class_probs


def _multiclass_mstep(
    cfg,
    batch_full,
    translations,
    ctf_params,
    noise_variance_full,
    rotations,
    log_resps,
    volume_shape,
    mus_prev,
    class_probs,
    k_max=None,
    tau: float = 0.0,
    min_occ: float = 0.01,
):
    """Per-class Wiener reconstruction with dead-class protection."""
    K = len(log_resps)
    occ = np.asarray(jnp.mean(class_probs, axis=0))
    mus_new = []
    for k in range(K):
        if occ[k] < min_occ:
            mus_new.append(mus_prev[k])
            continue
        per_r = _per_rotation_residual_image(
            cfg,
            batch_full,
            translations,
            ctf_params,
            noise_variance_full,
            log_resps[k],
            residual_subtraction_half=None,
        )
        per_r_ctf = _per_rotation_ctf_image(
            cfg,
            ctf_params,
            noise_variance_full,
            log_resps[k],
        )
        Ft_y = _backproject_to_half_volume(per_r, rotations, cfg.image_shape, volume_shape)
        Ft_ctf = _backproject_to_half_volume(per_r_ctf, rotations, cfg.image_shape, volume_shape)
        mu_k = _solve_wiener(Ft_y, Ft_ctf, tau).astype(jnp.complex128)
        if k_max is not None:
            mu_k = radial_band_limit_half(mu_k[None], volume_shape, k_max)[0]
        mus_new.append(mu_k)
    return mus_new


def multiclass_mu_init(
    cfg,
    ds,
    *,
    K: int = 5,
    n_iters: int = 50,
    n_blobs: int = 40,
    blob_sigma_frac: float = 0.08,
    k_max_start: int = 3,
    freq_march_interval: int = 5,
    softness_anneal_iters: int = 50,
    seed: int = 0,
    verbose: bool = True,
) -> jnp.ndarray:
    """Bootstrap a μ_init from K random blob volumes via multi-class EM.

    Runs Phases 0-1 of the multiclass-abinitio pipeline (init random
    blobs → softness-annealed K-class EM with frequency marching) and
    returns the mean of the K class volumes as μ_init.

    The standard `run_two_stage` is then expected to run the weighted-
    SVD warmstart against this μ_init and a normal EM loop on top.

    Parameters
    ----------
    cfg : ForwardModelConfig
    ds : SyntheticDataset
    K : number of classes (memory: K ≈ n_states or K > q)
    n_iters : multi-class EM iterations
    n_blobs / blob_sigma_frac : RELION-style blob init parameters
    k_max_start / freq_march_interval : frequency marching schedule
    softness_anneal_iters : class-softness anneal length (1.0 → 0.0)
    seed : RNG seed (per-class seeds are seed + k)

    Returns
    -------
    mu_init : (half_vol_size,) complex128
    """
    volume_shape = cfg.volume_shape
    N = volume_shape[0]
    k_max_end = N // 2 - 1
    weights_half = make_half_image_weights(cfg.image_shape)

    if verbose:
        print(f"[cold_init] multiclass μ-bootstrap: K={K}, n_iters={n_iters}", flush=True)

    # Phase 0: K random blob volumes
    mus_half = [
        init_random_blobs(
            volume_shape, n_blobs=n_blobs, blob_sigma_frac=blob_sigma_frac, seed=seed + k, k_max=k_max_start
        )
        for k in range(K)
    ]

    # Preprocess batch (class-independent)
    shifted_half, ctf2_over_nv_half, _ = _preprocess_batch_to_half(
        cfg, ds.batch_full, ds.translations, ds.ctf_params, ds.noise_variance_full
    )

    fm_schedule = _freq_march_schedule(n_iters, k_max_start, k_max_end, freq_march_interval)

    # Softness schedule: 1.0 → 0.0 over softness_anneal_iters, then 0.0
    softness_schedule = np.zeros(n_iters)
    n_anneal = min(softness_anneal_iters, n_iters)
    if n_anneal > 0:
        softness_schedule[:n_anneal] = np.linspace(1.0, 0.0, n_anneal)

    # Phase 1: multi-class EM with frequency marching + softness anneal
    for it in range(1, n_iters + 1):
        t0 = time.perf_counter()
        k_max_it = fm_schedule[it - 1]
        softness_it = float(softness_schedule[it - 1])

        # Band-limit before E-step
        mus_bl = [radial_band_limit_half(m[None], volume_shape, k_max_it)[0] for m in mus_half]

        log_resps, class_probs = _multiclass_estep(
            cfg,
            mus_bl,
            ds.rotations,
            shifted_half,
            ctf2_over_nv_half,
            weights_half,
            class_softness=softness_it,
        )

        mus_half = _multiclass_mstep(
            cfg,
            ds.batch_full,
            ds.translations,
            ds.ctf_params,
            ds.noise_variance_full,
            ds.rotations,
            log_resps,
            volume_shape,
            mus_prev=mus_half,
            class_probs=class_probs,
            k_max=k_max_it,
            tau=0.0,
        )
        for m in mus_half:
            jax.block_until_ready(m)

        if verbose and (it % 10 == 0 or it == 1 or it == n_iters):
            occ = np.asarray(jnp.mean(class_probs, axis=0))
            occ_str = " ".join(f"{o:.2f}" for o in occ)
            soft_tag = f" s={softness_it:.2f}" if softness_it > 0.01 else ""
            print(
                f"[cold_init] mc {it:3d} k={k_max_it:2d}{soft_tag}: occ=[{occ_str}] ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    # Phase 2: aggregate class means → μ_init
    mu_init = jnp.mean(jnp.stack(mus_half), axis=0)
    if verbose:
        print(f"[cold_init] returning μ_init = mean of {K} class volumes", flush=True)
    return mu_init.astype(jnp.complex128)
