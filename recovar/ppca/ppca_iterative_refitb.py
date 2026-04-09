"""Iterative PPCA + RefitB.

Like ``ppca_iterative_projcov`` but the eigenvalue refinement step uses
likelihood-based EM on B (Algorithm 1 / RefitB) instead of the moment-based
projected covariance.

Per outer iteration:

  1. PPCA ``EM_step_half`` updates W using the current calibrated B as the
     E-step prior (``em_eigenvalues``).
  2. Orthonormalize W → U_real (and read off ``s_em`` = squared singular values).
  3. Compute per-image sufficient statistics ``G_i, h_i`` for the new U.
  4. Run RefitB EM (1 or more inner iterations) starting from B = diag(s_em),
     optionally with an inverse-Wishart-style ridge prior:
        B_new = (Σ T_i + κ B_0) / (n + κ)
  5. Pass the refit eigenvalues to the next PPCA EM step as ``em_eigenvalues``
     so the model uses the calibrated prior.

After all outer iterations, embeddings are recomputed from scratch in the
final orthonormal basis (same path as IterPC) so the saved z's are in
α-coordinates relative to the saved U.
"""

from __future__ import annotations

import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core, utils
from recovar.ppca.ppca import (
    EM_step_half,
    _normalize_experiment_datasets,
    _orthonormalize_W_to_basis,
    compute_Cz_from_second_moments,
)

logger = logging.getLogger(__name__)


# Reuse the same vmapped slice helper that ppca_refit.py uses
batch_over_vol_slice_volume = jax.vmap(
    core.slice_volume, in_axes=(1, None, None, None, None), out_axes=1
)


@jax.jit
def _compute_Gi_hi_batch(PU, centered_images):
    """Compute G_i and h_i for one batch.

    PU: (n_batch, q, image_size) noise-whitened projected eigenvectors.
    centered_images: (n_batch, image_size) noise-whitened residual images.
    """
    G_batch = (jnp.conj(PU) @ PU.transpose(0, 2, 1)).real
    h_batch = (jnp.conj(PU) @ centered_images[..., None]).real.squeeze(-1)
    return G_batch, h_batch


def _compute_Gi_hi_from_U_real(
    dataset, mean_fourier, U_real, volume_shape, voxel_size,
    batch_size=128, disc_type="linear_interp", disc_type_mean="cubic",
    apply_image_mask=True,
):
    """Compute per-image G_i, h_i directly from a U_real array (no PipelineOutput).

    Parameters
    ----------
    dataset : CryoEMDataset
    mean_fourier : (vol_size,) flat Fourier mean
    U_real : (q, *volume_shape) real-space eigenvectors. Should already have
        any masks / gridding correction baked in by the caller.
    """
    image_shape = tuple(volume_shape[:2])
    q = int(U_real.shape[0])
    vol_size = int(np.prod(volume_shape))

    # U_real → Fourier (vol_size, q)
    U_fourier = np.zeros((vol_size, q), dtype=np.complex64)
    for j in range(q):
        U_fourier[:, j] = ftu.get_dft3(U_real[j]).reshape(-1)
    U_jax = jnp.array(U_fourier)

    mean_for_slicing = core.precompute_cubic_coefficients(mean_fourier, volume_shape)
    mean_jax = jnp.array(mean_for_slicing)

    n_images = dataset.n_images
    G_all = np.zeros((n_images, q, q), dtype=np.float64)
    h_all = np.zeros((n_images, q), dtype=np.float64)

    for (
        batch,
        rotation_matrices,
        translations,
        ctf_params,
        _noise_var,
        _particle_indices,
        image_indices,
    ) in dataset.iter_batches(
        batch_size,
        by_image=not getattr(dataset, "tilt_series_flag", False),
    ):
        images = dataset.process_images(batch, apply_image_mask=apply_image_mask)
        noise_variance = dataset.noise.get(image_indices)

        images = core.translate_images(images, translations, image_shape) / jnp.sqrt(noise_variance)
        CTF = dataset.ctf_evaluator(ctf_params, image_shape, voxel_size) / jnp.sqrt(noise_variance)

        projected_mean = core.slice_volume(
            mean_jax, rotation_matrices, image_shape, volume_shape, disc_type_mean,
        ) * CTF
        centered_images = images - projected_mean

        PU = batch_over_vol_slice_volume(
            U_jax, rotation_matrices, image_shape, volume_shape, disc_type,
        )
        PU = PU * CTF[:, None, :]

        G_batch, h_batch = _compute_Gi_hi_batch(PU, centered_images)
        indices = np.asarray(image_indices)
        G_all[indices] = np.asarray(G_batch, dtype=np.float64)
        h_all[indices] = np.asarray(h_batch, dtype=np.float64)

    return G_all, h_all


def _refit_B_em_steps(G_all, h_all, B_init, n_inner_iters=3, eps=1e-8,
                     B_prior=None, kappa=0.0):
    """A few EM iterations on B in the fixed span.

    Optionally regularizes with an inverse-Wishart-style ridge:
        B_new = (Σ T_i + κ B_prior) / (n + κ)
    where κ is the prior strength (effective number of pseudo-images) and
    B_prior is the prior mean covariance.
    """
    n, q, _ = G_all.shape
    B = np.array(B_init, dtype=np.float64)
    if B_prior is None:
        B_prior = np.eye(q, dtype=np.float64)
    else:
        B_prior = np.array(B_prior, dtype=np.float64)

    history = []
    for it in range(n_inner_iters):
        B_inv = np.linalg.inv(B)
        P_all = np.linalg.inv(B_inv[None] + G_all)              # (n, q, q)
        m_all = np.einsum("nij,nj->ni", P_all, h_all)           # (n, q)
        T_all = P_all + np.einsum("ni,nj->nij", m_all, m_all)   # (n, q, q)

        sum_T = np.sum(T_all, axis=0)
        if kappa > 0:
            B_new = (sum_T + kappa * B_prior) / (n + kappa)
        else:
            B_new = sum_T / n
        B_new = 0.5 * (B_new + B_new.T) + eps * np.eye(q)

        # NLL diagnostic
        sign_B, logdet_B = np.linalg.slogdet(B)
        _, logdet_BinvG = np.linalg.slogdet(B_inv[None] + G_all)
        quad = np.einsum("ni,ni->n", h_all, m_all)
        nll = float(np.sum(logdet_B + logdet_BinvG - quad))
        history.append(nll)
        B = B_new

    return B, history


def iterative_ppca_refitb(
    experiment_dataset,
    mean_estimate,
    W_initial,
    W_prior,
    volume_mask,
    dilated_volume_mask,
    EM_iter=20,
    refitb_every=1,
    refitb_start=0,
    refitb_inner_iters=3,
    kappa=0.0,
    batch_size=256,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    gpu_memory_to_use=40,
    recompute_ll=False,
    mean_estimate_raw=None,
    contrast_mode="none",
    contrast_grid=None,
    contrast_mean=1.0,
    contrast_variance=np.inf,
    use_gridding_correction=True,
):
    """PPCA EM with RefitB after each (sufficient) iteration.

    See module docstring for the full algorithm. Returns the same tuple shape
    as ``iterative_ppca_projcov`` for parity with the dispatcher.
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_dataset)
    reference_dataset = full_dataset if full_dataset is not None else dataset_list[0]
    vs = reference_dataset.volume_shape
    voxel_size = float(getattr(reference_dataset, "voxel_size", 1.0))

    W = jnp.array(W_initial)
    iteration_data = []
    em_eigenvalues = None  # B for the next E-step (None = identity)

    # Initial B for the inverse-Wishart prior, if used: take the PPCA-init B
    B_prior_for_reg = None

    for iter_i in range(EM_iter):
        t0 = time.time()
        _contrast_mode = contrast_mode if iter_i >= 3 else "none"

        # --- 1. PPCA EM step using current B as the prior ---
        # EM_step_half now uses PCG (masked, half-volume PCG) unconditionally
        # when volume_mask is non-trivial — no solver to plumb through.
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
            disc_type_mean=disc_type_mean,
            disc_type=disc_type,
            recompute_ll=recompute_ll,
            mean_estimate_raw=mean_estimate_raw,
            volume_mask=volume_mask,
            contrast_mode=_contrast_mode,
            contrast_grid=jnp.array(contrast_grid) if contrast_grid is not None else None,
            eigenvalues=em_eigenvalues,
            contrast_mean=contrast_mean,
            contrast_variance=contrast_variance,
            return_mean_c=True,
        )

        # Post-process W: half-Fourier → real → mask → gridding → half-Fourier.
        # pcg_mstep does not internalise K (the gridding kernel), so this
        # post-step is the canonical correction.
        basis_size = W.shape[1]
        half_vs = ftu.volume_shape_to_half_volume_shape(vs)
        W_real = ftu.get_idft3_real(W.T.reshape(basis_size, *half_vs), vs)
        if volume_mask is not None and not np.all(np.asarray(volume_mask) == 1):
            W_real = W_real * jnp.array(volume_mask)[None]
        if use_gridding_correction:
            from recovar.reconstruction.relion_functions import griddingCorrect_square
            for k in range(basis_size):
                W_real = W_real.at[k].set(griddingCorrect_square(W_real[k], vs[0], 1, order=1)[0])
        W = ftu.get_dft3_real(W_real).reshape(basis_size, -1).T

        em_time = time.time() - t0

        # --- 2-5. RefitB step (every refitb_every iterations after refitb_start) ---
        refit_time = 0.0
        refit_s = None
        if iter_i >= refitb_start and (iter_i - refitb_start) % refitb_every == 0:
            t1 = time.time()

            # 2. Orthonormalize W → U_real, get current eigenvalues s_em
            U_real, s_em, _Vt = _orthonormalize_W_to_basis(W, vs)
            q = U_real.shape[0]
            vol_size = int(np.prod(vs))

            # 3. Compute per-image sufficient statistics in the (un-masked,
            # un-gridded) orthonormal basis U_real. We deliberately do NOT
            # pre-apply the mask or gridding correction to U here, because
            # B_refit's eigenvectors / eigenvalues live in whichever basis
            # we used for the fit, and the rebake (step 5) uses the same
            # un-masked U_real. Mixing a masked-basis fit with an un-masked
            # rebake is undefined behavior (the basis-change rule requires
            # an invertible q×q transform; multiplication by a real-space
            # mask is not invertible). This matches iterative_ppca_projcov's
            # convention (which passes the un-masked basis_fourier to projcov
            # and lets projcov handle the mask internally to the data fit).
            G_all, h_all = _compute_Gi_hi_from_U_real(
                reference_dataset,
                mean_estimate_raw if mean_estimate_raw is not None else mean_estimate,
                np.asarray(U_real),
                vs,
                voxel_size,
                batch_size=batch_size,
                disc_type=disc_type,
                disc_type_mean=disc_type_mean,
            )

            # 4. RefitB EM iterations starting from diag(s_em)
            B_init = np.diag(s_em.astype(np.float64))
            if B_prior_for_reg is None:
                B_prior_for_reg = B_init.copy()
            B_refit, refit_history = _refit_B_em_steps(
                G_all, h_all, B_init,
                n_inner_iters=refitb_inner_iters,
                kappa=kappa,
                B_prior=B_prior_for_reg if kappa > 0 else None,
            )

            # Diagonalize the refit B (eigenvalues sorted descending)
            eigvals_refit, R_refit = np.linalg.eigh(B_refit)
            order = np.argsort(eigvals_refit)[::-1]
            eigvals_refit = eigvals_refit[order]
            R_refit = R_refit[:, order]
            refit_s = eigvals_refit.astype(np.float32)

            # 5. Bake the refit calibration directly into W (no eigenvalues
            # plumbing). Set W = U_rotated · diag(sqrt(refit_s)) where
            # U_rotated = U_real @ R_refit aligns with B_refit's eigenbasis.
            # This is the correct way to inject the calibration: PPCA EM's
            # next iteration starts from a W with the right column scales.
            # Same fix as iterative_ppca_projcov uses for projcov_s.
            U_rotated_real = np.einsum("kxyz,kj->jxyz", np.asarray(U_real), R_refit.astype(np.float32))
            # Convert (q, *vs) real → full Fourier (vol_size, q) → half (half_vol, q)
            U_rotated_full_F = np.asarray(ftu.get_dft3(U_rotated_real)).reshape(q, vol_size).T  # (vs, q)
            W_full = (U_rotated_full_F * np.sqrt(refit_s)[None, :]).astype(np.complex64)
            W_full_grid = W_full.T.reshape(q, *vs)
            W_half_grid = ftu.full_volume_to_half_volume(W_full_grid, vs)
            W = jnp.array(np.asarray(W_half_grid).reshape(q, -1).T)

            # No more eigenvalues feedback — calibration is in W now.
            em_eigenvalues = None

            refit_time = time.time() - t1
            logger.info(
                "  Iter %2d: PPCA-EM %.1fs, RefitB %.1fs (κ=%.2f, inner=%d), NLL=%.4e, "
                "s_em=[%.2e,%.2e,..], s_refit=[%.2e,%.2e,..]",
                iter_i + 1, em_time, refit_time, kappa, refitb_inner_iters,
                float(neg_ll_total),
                float(s_em[0]), float(s_em[1]) if len(s_em) > 1 else 0,
                float(refit_s[0]), float(refit_s[1]) if len(refit_s) > 1 else 0,
            )
        else:
            logger.info(
                "  Iter %2d: PPCA-EM %.1fs (no RefitB), NLL=%.4e",
                iter_i + 1, em_time, float(neg_ll_total),
            )

        # Per-iteration diagnostics
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
            "refit_time": refit_time,
            "refit_s": refit_s.tolist() if refit_s is not None else None,
            "C_z_diag": np.diag(np.asarray(C_z)).tolist(),
            "eigenvalues_iter": s_iter.tolist(),
            "U_real_iter": U_real_iter,
        }
        iteration_data.append(iter_info)

    U_real, s_final, _ = _orthonormalize_W_to_basis(W, vs)
    return U_real, s_final, W, expected_zs, second_moment_zs, iteration_data


def run_iterative_ppca_refitb(
    pipeline_output,
    output_dir,
    zdim=None,
    batch_size=256,
    n_iters=20,
    refitb_every=1,
    refitb_start=0,
    refitb_inner_iters=3,
    kappa=0.0,
    gpu_memory_to_use=40,
):
    """High-level entry point for the dispatcher."""
    from recovar.output.output import PipelineOutput
    from recovar.ppca.ppca_refit import (
        create_postprocessed_result_dir,
        compute_per_image_Gi_hi,
        compute_embeddings_from_UB,
    )

    po = pipeline_output
    volume_shape = tuple(po.params["volume_shape"])
    dataset = po.get("dataset")

    # Initial W from PPCA loadings
    W_loadings = np.load(po.paths.ppca_loadings)
    if zdim is None:
        zdim = W_loadings.shape[1]
    zdim = min(zdim, W_loadings.shape[1])
    W_loadings = W_loadings[:, :zdim]

    half_vs = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_vol_size = int(np.prod(half_vs))
    if W_loadings.shape[0] == half_vol_size:
        W_half = jnp.array(W_loadings, dtype=jnp.complex64)
    else:
        W_half = np.zeros((half_vol_size, zdim), dtype=np.complex64)
        for j in range(zdim):
            w_full = W_loadings[:, j].reshape(volume_shape)
            W_half[:, j] = ftu.full_volume_to_half_volume(w_full, volume_shape).reshape(-1)
        W_half = jnp.array(W_half)

    mean_fourier = po.get("mean")
    mean_for_slicing = core.precompute_cubic_coefficients(mean_fourier, volume_shape)

    from recovar.ppca import prior_estimation as ppca_prior_estimation
    logger.info("Estimating shell prior from data (matching baseline pipeline)...")
    prior_info = ppca_prior_estimation.estimate_hybrid_shell_prior_from_data(
        dataset, mean_fourier, zdim, volume_shape, batch_size,
    )
    W_prior = prior_info["W_prior"]

    volume_mask = utils.load_mrc(po.paths.mask_volume) if hasattr(po.paths, "mask_volume") else None
    dilated_volume_mask = utils.load_mrc(po.paths.dilated_mask_volume) if hasattr(po.paths, "dilated_mask_volume") else None
    if dilated_volume_mask is None:
        dilated_volume_mask = volume_mask

    logger.info(
        "Running iterative PPCA + RefitB: zdim=%d n_iters=%d refitb_start=%d refitb_every=%d "
        "inner=%d kappa=%.2f",
        zdim, n_iters, refitb_start, refitb_every, refitb_inner_iters, kappa,
    )

    u_real, s, W, expected_zs, second_moment_zs, iteration_data = iterative_ppca_refitb(
        dataset,
        mean_for_slicing,
        W_half,
        W_prior,
        volume_mask=volume_mask,
        dilated_volume_mask=dilated_volume_mask,
        EM_iter=n_iters,
        refitb_every=refitb_every,
        refitb_start=refitb_start,
        refitb_inner_iters=refitb_inner_iters,
        kappa=kappa,
        batch_size=batch_size,
        disc_type_mean="cubic",
        disc_type="linear_interp",
        gpu_memory_to_use=gpu_memory_to_use,
        mean_estimate_raw=mean_fourier,
    )

    q = u_real.shape[0]
    n_images = expected_zs.shape[0]

    # Placeholder embeddings; recomputed below in the final basis
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

    # Save per-iteration U_real for inspection (and strip from JSON)
    iters_dir = os.path.join(output_dir, "iterations")
    os.makedirs(iters_dir, exist_ok=True)
    iteration_data_clean = []
    for it in iteration_data:
        u_iter = it.pop("U_real_iter", None)
        if u_iter is not None:
            np.save(os.path.join(iters_dir, f"U_real_iter{it['iteration']:03d}.npy"),
                    u_iter.astype(np.float32))
        iteration_data_clean.append(it)
    iteration_data = iteration_data_clean

    method_info = {
        "method": "iterative_ppca_refitb",
        "n_iters": n_iters,
        "refitb_every": refitb_every,
        "refitb_start": refitb_start,
        "refitb_inner_iters": refitb_inner_iters,
        "kappa": kappa,
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

    # Recompute embeddings in the final basis (same as IterPC)
    po_new = PipelineOutput(output_dir)
    G_all, h_all = compute_per_image_Gi_hi(po_new, batch_size=batch_size, zdim=zdim)
    B_diag = np.diag(s[:zdim].astype(np.float64))
    latent_coords, precision_all = compute_embeddings_from_UB(G_all, h_all, B_diag)

    from recovar.output.output_paths import ResultPaths
    paths = ResultPaths(output_dir)
    zdim_dir = paths.embedding_zdim_dir(zdim)
    os.makedirs(zdim_dir, exist_ok=True)
    np.save(os.path.join(zdim_dir, "latent_coords.npy"), latent_coords.astype(np.float32))
    np.save(os.path.join(zdim_dir, "latent_coords_noreg.npy"), latent_coords.astype(np.float32))
    np.save(os.path.join(zdim_dir, "latent_precision.npy"), precision_all.astype(np.float32))
    np.save(os.path.join(zdim_dir, "latent_precision_noreg.npy"), precision_all.astype(np.float32))
    logger.info("Recomputed embeddings via compute_per_image_Gi_hi in final basis")

    return {
        "eigenvalues": s[:zdim],
        "U_real": u_real,
        "iteration_data": iteration_data,
        "embeddings": latent_coords,
    }
