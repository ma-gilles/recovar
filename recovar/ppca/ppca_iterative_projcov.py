"""Iterative PPCA + Projected Covariance — harness wrapper.

The actual algorithm now lives inside :func:`recovar.ppca.ppca.EM` and is
selected with the ``projcov_every`` / ``projcov_start`` / ``bfit_whitening``
parameters. This module only contains the harness glue
(:func:`run_iterative_ppca_projcov`) that wires a ``PipelineOutput`` to
``ppca.EM`` and writes the result back to a PipelineOutput-compatible
directory. Standalone scripts and the comparison driver import this entry
point; nothing else here is part of the public surface.
"""

from __future__ import annotations

import logging
import os

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core, utils
from recovar.ppca import ppca

logger = logging.getLogger(__name__)


def run_iterative_ppca_projcov(
    pipeline_output,
    output_dir,
    zdim=None,
    batch_size=256,
    n_iters=20,
    projcov_every=1,
    projcov_start=0,
    gpu_memory_to_use=40,
    bfit_whitening=True,
):
    """Run iterative PPCA+ProjCov from a ``PipelineOutput`` and save results.

    Loads the dataset and initial W from the PPCA result, calls
    :func:`recovar.ppca.ppca.EM` with the requested ``projcov_every`` /
    ``bfit_whitening`` settings, then re-derives per-image embeddings in the
    final orthonormal basis (matching the refit_b convention) and writes a
    PipelineOutput-compatible directory.
    """
    from recovar.output.output import PipelineOutput  # noqa: F401  (kept for type hint clarity)
    from recovar.ppca import prior_estimation as ppca_prior_estimation
    from recovar.ppca.ppca_refit import (
        compute_embeddings_from_UB,
        compute_per_image_Gi_hi,
        create_postprocessed_result_dir,
    )

    po = pipeline_output
    volume_shape = tuple(po.params["volume_shape"])
    dataset = po.get("dataset")

    # ── Initial W: load PPCA loadings, force half-Fourier shape ─────────
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
        W_half_np = np.zeros((half_vol_size, zdim), dtype=np.complex64)
        for j in range(zdim):
            w_full = W_loadings[:, j].reshape(volume_shape)
            W_half_np[:, j] = ftu.full_volume_to_half_volume(w_full, volume_shape).reshape(-1)
        W_half = jnp.array(W_half_np)

    # ── Prior + masks ───────────────────────────────────────────────────
    mean_fourier = po.get("mean")
    mean_for_slicing = core.precompute_cubic_coefficients(mean_fourier, volume_shape)

    logger.info("Estimating shell prior from data (matches baseline pipeline)")
    prior_info = ppca_prior_estimation.estimate_hybrid_shell_prior_from_data(
        dataset, mean_fourier, zdim, volume_shape, batch_size,
    )
    W_prior = prior_info["W_prior"]

    volume_mask = utils.load_mrc(po.paths.mask_volume) if hasattr(po.paths, "mask_volume") else None
    dilated_volume_mask = (
        utils.load_mrc(po.paths.dilated_mask_volume)
        if hasattr(po.paths, "dilated_mask_volume")
        else None
    )
    if dilated_volume_mask is None:
        dilated_volume_mask = volume_mask

    logger.info(
        "Running ppca.EM(projcov_every=%d, projcov_start=%d, bfit_whitening=%s) zdim=%d n_iters=%d",
        projcov_every, projcov_start, bfit_whitening, zdim, n_iters,
    )

    # ── Run EM with projcov refinement ──────────────────────────────────
    U_em, S_em, W, expected_zs, second_moment_zs = ppca.EM(
        dataset,
        mean_for_slicing,
        W_half,
        W_prior,
        EM_iter=n_iters,
        disc_type_mean="cubic",
        disc_type="linear_interp",
        disc_type_u="linear_interp",
        volume_mask=np.asarray(volume_mask, dtype=np.float32) if volume_mask is not None else None,
        dilated_volume_mask=dilated_volume_mask,
        use_gridding_correction=True,
        projcov_every=projcov_every,
        projcov_start=projcov_start,
        bfit_whitening=bfit_whitening,
        gpu_memory_to_use=gpu_memory_to_use,
    )

    # ── Final orthonormal basis (real space, PPCA convention) ───────────
    u_real, s, _ = ppca._orthonormalize_W_to_basis(W, volume_shape)

    # ── Save shell + recompute embeddings in the final basis ────────────
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
    method_info = {
        "method": "iterative_ppca_projcov",
        "n_iters": n_iters,
        "projcov_every": projcov_every,
        "projcov_start": projcov_start,
        "bfit_whitening": bfit_whitening,
        "zdim": zdim,
        "eigenvalues": s[:zdim].tolist(),
    }

    create_postprocessed_result_dir(
        po.result_path.rstrip("/"),
        output_dir,
        u_real,
        s[:zdim],
        embeddings_dict,
        method_info=method_info,
    )

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

    return {
        "eigenvalues": s[:zdim],
        "U_real": u_real,
        "embeddings": latent_coords,
    }
