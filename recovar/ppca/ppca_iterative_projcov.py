"""Iterative PPCA + Projected Covariance.

Standard PPCA runs EM to convergence, then optionally does one projected
covariance step at the end.  This module interleaves the two: after each
PPCA EM iteration, it orthonormalizes W → U, runs projected covariance in
span(U) to get refined eigenvalues, and reconstructs W = U·diag(√s) for
the next EM iteration.

The projected covariance step recalibrates the spectrum at every iteration,
so the EM E-step sees a better-calibrated B at each step.
"""

from __future__ import annotations

import logging
import os
import time

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core, utils
from recovar.core import linalg
from recovar.heterogeneity import principal_components
from recovar.ppca.ppca import (
    EM_step_half,
    _normalize_experiment_datasets,
    compute_Cz_from_second_moments,
    whiten_W_iterative,
)

logger = logging.getLogger(__name__)


def _orthonormalize_W_to_basis(W_half, volume_shape):
    """Orthonormalize W in real space via SVD.

    Flow: W_half → real space → SVD → returns real-space U and eigenvalues.
    Caller converts to Fourier as needed.

    Returns:
        U_real: (q, *volume_shape) real-space orthonormal eigenvectors
        s: (q,) eigenvalues (squared singular values)
        Vt: (q, q) right singular vectors
    """
    vs = volume_shape
    half_vs = ftu.volume_shape_to_half_volume_shape(vs)
    q = W_half.shape[1]
    vol_size = int(np.prod(vs))

    # half-volume Fourier → real space
    W_real = np.zeros((vol_size, q), dtype=np.float32)
    for j in range(q):
        w_half = np.asarray(W_half[:, j]).reshape(half_vs)
        W_real[:, j] = ftu.get_idft3(
            ftu.half_volume_to_full_volume(w_half, vs)
        ).real.reshape(-1).astype(np.float32)

    # SVD in real space
    U_flat, S_real, Vt = np.linalg.svd(W_real, full_matrices=False)

    # Convert to PPCA convention: eigenvectors have Fourier-unit-norm (= 1/√vol_size in real space)
    # and eigenvalues = Fourier-space singular values squared (= real * vol_size)
    s = (S_real ** 2 * vol_size).astype(np.float32)
    U_flat = U_flat / np.sqrt(vol_size)

    # Reshape to volume layout
    U_real = U_flat.T.reshape(q, *vs).astype(np.float32)  # (q, *vol_shape)

    return U_real, s, Vt


def _reconstruct_W_half(basis_fourier, s, Vt, volume_shape):
    """Reconstruct half-volume W from orthonormal basis and eigenvalues.

    W = U · diag(√s) · Vt  but since we're going back to the whitened frame
    where z ~ N(0, I), we just need W = U · diag(√s).
    The Vt rotation is within the q-dim subspace and absorbed by the z coordinates.

    Returns:
        W_half: (half_vol_size, q) half-volume loading matrix
    """
    vs = volume_shape
    half_vs = ftu.volume_shape_to_half_volume_shape(vs)
    q = basis_fourier.shape[1]

    # W_full = U · diag(√s)
    W_full = basis_fourier * np.sqrt(s)[None, :]

    # full → half volume
    half_vol_size = int(np.prod(half_vs))
    W_half = np.zeros((half_vol_size, q), dtype=np.complex64)
    for j in range(q):
        w_full = W_full[:, j].reshape(vs)
        W_half[:, j] = ftu.full_volume_to_half_volume(w_full, vs).reshape(-1)

    return jnp.array(W_half)


def iterative_ppca_projcov(
    experiment_dataset,
    mean_estimate,
    W_initial,
    W_prior,
    volume_mask,
    dilated_volume_mask,
    EM_iter=20,
    projcov_every=1,
    projcov_start=0,
    batch_size=256,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    disc_type_u="linear_interp",
    gpu_memory_to_use=40,
    use_whitening=False,
    whitening_mode="cz",
    recompute_ll=False,
    mean_estimate_raw=None,
    use_pcg_mean=False,
    contrast_mode="none",
    contrast_grid=None,
    contrast_mean=1.0,
    contrast_variance=np.inf,
    use_gridding_correction=True,
):
    """PPCA EM with projected covariance after each iteration.

    Parameters
    ----------
    experiment_dataset : CryoEMDataset
    mean_estimate : mean volume (Fourier, possibly cubic-precomputed)
    W_initial : (half_vol_size, q) initial loading matrix
    W_prior : per-voxel prior variance for W columns
    volume_mask : real-space volume mask for projected covariance
    dilated_volume_mask : dilated mask
    EM_iter : number of outer iterations
    projcov_every : run projected covariance every N iterations
    projcov_start : start projected covariance after this many iterations
    batch_size : batch size for both EM and projected covariance

    Returns
    -------
    U : (vol_size, q) orthonormal basis in Fourier space
    s : (q,) eigenvalues
    W : (half_vol_size, q) final loading matrix
    expected_zs : (n_images, q) final posterior means
    second_moment_zs : (n_images, q, q) final posterior second moments
    iteration_data : list of per-iteration diagnostics
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_dataset)
    reference_dataset = full_dataset if full_dataset is not None else dataset_list[0]
    vs = reference_dataset.volume_shape

    W = jnp.array(W_initial)
    iteration_data = []
    # Eigenvalues for the E-step prior: None = B=I (standard PPCA), or s_projcov after ProjCov step
    em_eigenvalues = None

    for iter_i in range(EM_iter):
        t0 = time.time()

        # --- Standard PPCA EM step (with calibrated prior if available) ---
        _contrast_mode = contrast_mode if iter_i >= 3 else "none"
        (
            W,
            expected_zs,
            second_moment_zs,
            expected_zs_mean,
            expected_zs_var,
            neg_ll_total,
            neg_ll_data,
            neg_ll_prior,
            mean_c,
        ) = EM_step_half(
            experiment_dataset,
            mean_estimate,
            W,
            batch_size,
            W_prior,
            use_whitening=use_whitening,
            whitening_mode=whitening_mode,
            disc_type_mean=disc_type_mean,
            disc_type=disc_type,
            recompute_ll=recompute_ll,
            mean_estimate_raw=mean_estimate_raw,
            use_pcg_mstep=use_pcg_mean,
            contrast_mode=_contrast_mode,
            contrast_grid=jnp.array(contrast_grid) if contrast_grid is not None else None,
            eigenvalues=em_eigenvalues,
            contrast_mean=contrast_mean,
            contrast_variance=contrast_variance,
            return_mean_c=True,
        )

        # Post-process W to match baseline pipeline: half->real, mask, gridding, real->half
        # This matches what the outer EM loop in ppca.EM does (lines 1764-1798)
        basis_size = W.shape[1]
        half_vs = ftu.volume_shape_to_half_volume_shape(vs)
        # Half-volume Fourier → real space
        W_real = ftu.get_idft3_real(W.T.reshape(basis_size, *half_vs), vs)  # (q, D, D, D)
        # Apply volume mask
        if volume_mask is not None and not np.all(np.asarray(volume_mask) == 1):
            W_real = W_real * jnp.array(volume_mask)[None]
        # Gridding correction (divide by sinc²)
        if use_gridding_correction:
            from recovar.reconstruction.relion_functions import griddingCorrect_square
            for k in range(basis_size):
                W_real = W_real.at[k].set(griddingCorrect_square(W_real[k], vs[0], 1, order=1)[0])
        # Real → half-volume Fourier
        W = ftu.get_dft3_real(W_real).reshape(basis_size, -1).T

        em_time = time.time() - t0

        # --- Projected covariance step: recalibrate eigenvalues for the next E-step ---
        projcov_time = 0.0
        projcov_s = None
        if iter_i >= projcov_start and (iter_i - projcov_start) % projcov_every == 0:
            t1 = time.time()

            # Orthonormalize current W in real space
            U_real, s_em, Vt = _orthonormalize_W_to_basis(W, vs)
            q = U_real.shape[0]

            # Convert U to Fourier (vol_size, q) for projected covariance
            vol_size = int(np.prod(vs))
            basis_fourier = np.zeros((vol_size, q), dtype=np.complex64)
            for j in range(q):
                basis_fourier[:, j] = ftu.get_dft3(U_real[j]).reshape(-1)

            # Projected covariance: get calibrated eigenvalues
            refined_u, projcov_s = principal_components.pca_by_projected_covariance(
                reference_dataset,
                basis_fourier,
                mean_estimate_raw if mean_estimate_raw is not None else mean_estimate,
                dilated_volume_mask,
                disc_type=disc_type,
                disc_type_u=disc_type_u,
                gpu_memory_to_use=gpu_memory_to_use,
                use_mask=True,
                n_pcs_to_compute=q,
            )

            # DON'T reconstruct W — just pass the calibrated eigenvalues to the next E-step.
            # The E-step uses lambdas = eigenvalues as the prior B = diag(lambdas).
            # This replaces the B=I assumption with B=diag(s_projcov).
            # We need eigenvalues in the W-frame: since W = U diag(sqrt(s_em)),
            # and ProjCov gives s in the U-frame, the ratio s_projcov/s_em
            # tells us how much to inflate each component's prior.
            # But the E-step eigenvalues are in z-space where z~N(0,diag(lambdas)),
            # so lambdas = s_projcov / s_em (the ratio, not absolute values).
            em_eigenvalues = jnp.array(projcov_s / (s_em + 1e-10), dtype=jnp.float32)

            projcov_time = time.time() - t1
            logger.info(
                "  Iter %2d: EM %.1fs, ProjCov %.1fs, NLL=%.4e, s_em=[%.2e,%.2e,..], s_pc=[%.2e,%.2e,..]",
                iter_i + 1, em_time, projcov_time,
                float(neg_ll_total),
                float(s_em[0]), float(s_em[1]) if len(s_em) > 1 else 0,
                float(projcov_s[0]), float(projcov_s[1]) if len(projcov_s) > 1 else 0,
            )
        else:
            logger.info(
                "  Iter %2d: EM %.1fs (no ProjCov), NLL=%.4e",
                iter_i + 1, em_time, float(neg_ll_total),
            )

        # Compute current U and eigenvalues at each iter (in PPCA convention)
        U_real_iter, s_iter, _ = _orthonormalize_W_to_basis(W, vs)

        C_z = compute_Cz_from_second_moments(second_moment_zs)
        iter_info = {
            "iteration": iter_i + 1,
            "neg_ll_total": float(neg_ll_total),
            "neg_ll_data": float(neg_ll_data),
            "neg_ll_prior": float(neg_ll_prior),
            "expected_zs_mean": float(np.mean(expected_zs_mean)),
            "expected_zs_var": float(np.mean(expected_zs_var)),
            "em_time": em_time,
            "projcov_time": projcov_time,
            "projcov_s": projcov_s.tolist() if projcov_s is not None else None,
            "C_z_diag": np.diag(np.asarray(C_z)).tolist(),
            "eigenvalues_iter": s_iter.tolist(),
            "U_real_iter": U_real_iter,  # keep in memory; saved later
        }
        iteration_data.append(iter_info)

    # Final orthonormalization in real space
    U_real, s_final, Vt = _orthonormalize_W_to_basis(W, vs)

    return U_real, s_final, W, expected_zs, second_moment_zs, iteration_data


def run_iterative_ppca_projcov(
    pipeline_output,
    output_dir,
    zdim=None,
    batch_size=256,
    n_iters=20,
    projcov_every=1,
    projcov_start=0,
    gpu_memory_to_use=40,
):
    """Run iterative PPCA+ProjCov from a PipelineOutput and save results.

    This is the high-level entry point for the compare harness.
    It loads the dataset and initial W from the PPCA result, runs
    the interleaved EM, and creates a PipelineOutput-compatible directory.
    """
    from recovar.output.output import PipelineOutput
    from recovar.ppca.ppca_refit import create_postprocessed_result_dir

    po = pipeline_output
    volume_shape = tuple(po.params["volume_shape"])

    # Load dataset
    dataset = po.get("dataset")

    # Load initial W from PPCA loadings
    W_loadings = np.load(po.paths.ppca_loadings)  # (vol_size, q) or (half_vol, q)
    if zdim is None:
        zdim = W_loadings.shape[1]
    zdim = min(zdim, W_loadings.shape[1])
    W_loadings = W_loadings[:, :zdim]

    # Convert to half-volume if needed
    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol_size = int(np.prod(half_vs))
    if W_loadings.shape[0] == half_vol_size:
        W_half = jnp.array(W_loadings, dtype=jnp.complex64)
    else:
        # Full volume → half volume
        W_half = np.zeros((half_vol_size, zdim), dtype=np.complex64)
        for j in range(zdim):
            w_full = W_loadings[:, j].reshape(volume_shape)
            W_half[:, j] = ftu.full_volume_to_half_volume(w_full, volume_shape).reshape(-1)
        W_half = jnp.array(W_half)

    # Mean estimate
    mean_fourier = po.get("mean")
    mean_for_slicing = core.precompute_cubic_coefficients(mean_fourier, volume_shape)

    # W prior: use the SAME data-dependent shell prior as the baseline pipeline
    from recovar.ppca import prior_estimation as ppca_prior_estimation
    vol_size = int(np.prod(volume_shape))
    logger.info("Estimating shell prior from data (matching baseline pipeline)...")
    prior_info = ppca_prior_estimation.estimate_hybrid_shell_prior_from_data(
        dataset, mean_fourier, zdim, volume_shape, batch_size,
    )
    W_prior = prior_info["W_prior"]
    logger.info(f"  W_prior shape: {W_prior.shape}, mean: {np.mean(W_prior):.2e}")

    # Masks
    volume_mask = utils.load_mrc(po.paths.mask_volume) if hasattr(po.paths, 'mask_volume') else None
    dilated_volume_mask = utils.load_mrc(po.paths.dilated_mask_volume) if hasattr(po.paths, 'dilated_mask_volume') else None
    if dilated_volume_mask is None:
        dilated_volume_mask = volume_mask

    logger.info(
        "Running iterative PPCA+ProjCov: zdim=%d, n_iters=%d, projcov_every=%d, projcov_start=%d",
        zdim, n_iters, projcov_every, projcov_start,
    )

    u_real, s, W, expected_zs, second_moment_zs, iteration_data = iterative_ppca_projcov(
        dataset,
        mean_for_slicing,
        W_half,
        W_prior,
        volume_mask=volume_mask,
        dilated_volume_mask=dilated_volume_mask,
        EM_iter=n_iters,
        projcov_every=projcov_every,
        projcov_start=projcov_start,
        batch_size=batch_size,
        disc_type_mean="cubic",
        disc_type="linear_interp",
        disc_type_u="linear_interp",
        gpu_memory_to_use=gpu_memory_to_use,
        mean_estimate_raw=mean_fourier,
    )

    q = u_real.shape[0]

    # Save U and s, then recompute embeddings from scratch via compute_per_image_Gi_hi.
    # The whitened posteriors (expected_zs) lose class structure when the basis is
    # rotated by projected covariance between iterations. Recomputing in the final
    # orthonormal basis gives correct embeddings (same path as refit_b).
    from recovar.ppca.ppca_refit import (
        compute_per_image_Gi_hi,
        compute_embeddings_from_UB,
    )

    n_images = expected_zs.shape[0]
    dummy = np.zeros((n_images, zdim), dtype=np.float32)
    dummy_prec = np.tile(np.eye(zdim, dtype=np.float32), (n_images, 1, 1))
    embeddings_dict = {
        "latent_coords": {zdim: dummy},
        "latent_coords_noreg": {zdim: dummy},
        "latent_precision": {zdim: dummy_prec},
        "latent_precision_noreg": {zdim: dummy_prec},
        "contrasts": {zdim: np.ones(n_images, dtype=np.float32)},
        "contrasts_noreg": {zdim: np.ones(n_images, dtype=np.float32)},
    }

    # Save per-iteration U_real to .npy and strip from iteration_data (for JSON-serializable params)
    iters_dir = os.path.join(output_dir, "iterations")
    os.makedirs(iters_dir, exist_ok=True)
    iteration_data_clean = []
    for it in iteration_data:
        u_iter = it.pop("U_real_iter", None)
        if u_iter is not None:
            np.save(os.path.join(iters_dir, f"U_real_iter{it['iteration']:03d}.npy"), u_iter.astype(np.float32))
        iteration_data_clean.append(it)
    iteration_data = iteration_data_clean

    method_info = {
        "method": "iterative_ppca_projcov",
        "n_iters": n_iters,
        "projcov_every": projcov_every,
        "projcov_start": projcov_start,
        "zdim": zdim,
        "eigenvalues": s[:zdim].tolist(),
        "iteration_data": iteration_data,
    }

    create_postprocessed_result_dir(
        po.result_path.rstrip("/"),
        output_dir,
        u_real,
        s[:zdim],
        embeddings_dict,
        method_info=method_info,
    )

    # Recompute embeddings in the final orthonormal basis
    from recovar.output.output import PipelineOutput as PO
    po_new = PO(output_dir)
    G_all, h_all = compute_per_image_Gi_hi(po_new, batch_size=batch_size, zdim=zdim)
    B_diag = np.diag(s[:zdim].astype(np.float64))
    latent_coords, precision_all = compute_embeddings_from_UB(G_all, h_all, B_diag)

    # Overwrite embeddings
    from recovar.output.output_paths import ResultPaths
    paths = ResultPaths(output_dir)
    zdim_dir = paths.embedding_zdim_dir(zdim)
    os.makedirs(zdim_dir, exist_ok=True)
    np.save(os.path.join(zdim_dir, "latent_coords.npy"), latent_coords.astype(np.float32))
    np.save(os.path.join(zdim_dir, "latent_coords_noreg.npy"), latent_coords.astype(np.float32))
    np.save(os.path.join(zdim_dir, "latent_precision.npy"), precision_all.astype(np.float32))
    np.save(os.path.join(zdim_dir, "latent_precision_noreg.npy"), precision_all.astype(np.float32))
    logger.info("Recomputed embeddings via compute_per_image_Gi_hi")

    return {
        "eigenvalues": s[:zdim],
        "U_real": u_real,
        "iteration_data": iteration_data,
        "embeddings": latent_coords,
    }
