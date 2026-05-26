"""Post-processing helpers for PPCA results.

These helpers are used by :mod:`recovar.ppca.ppca_iterative_projcov` and
:mod:`recovar.ppca.ppca_iterative_refitb` to (1) compute per-image
sufficient statistics G_i, h_i in a final orthonormal basis, (2) build
embeddings from a fitted latent covariance B, and (3) write the
postprocessed result to a PipelineOutput-compatible directory. None of
this is part of the EM loop itself; it's the "after EM, save it"
plumbing.
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core, utils
from recovar.output.output_paths import ResultPaths

logger = logging.getLogger(__name__)


# Reuse the vmapped slice routine; same shape recovar.ppca.ppca uses.
batch_over_vol_slice_volume = jax.vmap(
    core.slice_volume, in_axes=(1, None, None, None, None), out_axes=1
)


# ─────────────────────────────────────────────────────────────────────────
# Per-image sufficient statistics (G_i, h_i) in a final orthonormal basis
# ─────────────────────────────────────────────────────────────────────────


def _load_mean_fourier(po):
    mean_real = utils.load_mrc(po.paths.mean_volume)
    return ftu.get_dft3(mean_real).reshape(-1)


def _project_volume_through_ctf(
    volume, ctf_params, rotation_matrices, image_shape, volume_shape,
    voxel_size, ctf_evaluator, disc_type_mean,
):
    slices = core.slice_volume(
        volume, rotation_matrices, image_shape, volume_shape, disc_type_mean,
    )
    return slices * ctf_evaluator(ctf_params, image_shape, voxel_size)


@jax.jit
def _compute_Gi_hi_batch(PU, centered_images):
    """G_i = conj(PU) @ PU^T, h_i = conj(PU) @ y, both real-valued."""
    G_batch = (jnp.conj(PU) @ PU.transpose(0, 2, 1)).real
    h_batch = (jnp.conj(PU) @ centered_images[..., None]).real.squeeze(-1)
    return G_batch, h_batch


def compute_per_image_Gi_hi(
    pipeline_output,
    batch_size=128,
    zdim=None,
    disc_type="linear_interp",
    apply_image_mask=True,
    apply_volume_mask=True,
    apply_gridding_correction=True,
):
    """Compute per-image G_i and h_i in the final basis stored in ``po``.

    For each image i:
        G_i(U) = PU_i^T PU_i / sigma_i^2
        h_i(U) = PU_i^T r_i / sigma_i^2

    where PU_i = CTF_i * A_i U (rotation + CTF projection), r_i is the
    centred (mean-subtracted) image, all in noise-whitened space.

    For parity with the baseline pipeline the eigenvectors are
    multiplied by the dilated volume mask and divided by sinc² (gridding
    correction) before slicing — both can be turned off via the flags.
    """
    po = pipeline_output
    volume_shape = tuple(po.params["volume_shape"])
    image_shape = tuple(volume_shape[:2])
    voxel_size = float(po.params["voxel_size"])

    dataset = po.get("dataset")

    if zdim is None:
        zdim = len(po._list_saved_eigenvector_indices())
    zdim = min(zdim, len(po._list_saved_eigenvector_indices()))

    mean_fourier = _load_mean_fourier(po)

    # Apply mask + gridding correction to U in real space (matches the
    # baseline pipeline's embedding path).
    u_real = po.get_u_real(zdim)  # (q, *vol_shape)
    if apply_volume_mask:
        try:
            dilated_mask = utils.load_mrc(po.paths.dilated_mask_volume)
            for k in range(zdim):
                u_real[k] = u_real[k] * dilated_mask
            logger.info("compute_per_image_Gi_hi: applied dilated volume mask to U")
        except Exception as exc:
            logger.warning("compute_per_image_Gi_hi: could not apply volume mask: %s", exc)
    if apply_gridding_correction:
        try:
            from recovar.reconstruction.relion_functions import griddingCorrect_square
            for k in range(zdim):
                u_real[k] = np.asarray(
                    griddingCorrect_square(jnp.array(u_real[k]), volume_shape[0], 1, order=1)[0]
                )
            logger.info("compute_per_image_Gi_hi: applied gridding correction to U")
        except Exception as exc:
            logger.warning(
                "compute_per_image_Gi_hi: could not apply gridding correction: %s", exc
            )

    vol_size = int(np.prod(volume_shape))
    U_fourier = np.zeros((vol_size, zdim), dtype=np.complex64)
    for j in range(zdim):
        U_fourier[:, j] = ftu.get_dft3(u_real[j]).reshape(-1)

    mean_for_slicing = core.precompute_cubic_coefficients(mean_fourier, volume_shape)
    U_jax = jnp.array(U_fourier)
    mean_jax = jnp.array(mean_for_slicing)

    q = zdim
    n_images = dataset.n_images
    G_all = np.zeros((n_images, q, q), dtype=np.float64)
    h_all = np.zeros((n_images, q), dtype=np.float64)

    logger.info(
        "Computing per-image G_i, h_i: n_images=%d, q=%d, batch_size=%d",
        n_images, q, batch_size,
    )
    t0 = time.time()
    n_processed = 0

    for (
        batch,
        rotation_matrices,
        translations,
        ctf_params,
        _noise_variance,
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

        projected_mean = _project_volume_through_ctf(
            mean_jax, ctf_params, rotation_matrices, image_shape, volume_shape,
            voxel_size, dataset.ctf_evaluator, "cubic",
        ) / jnp.sqrt(noise_variance)
        centered_images = images - projected_mean

        PU = batch_over_vol_slice_volume(
            U_jax, rotation_matrices, image_shape, volume_shape, disc_type,
        )
        PU = PU * CTF[:, None, :]

        G_batch, h_batch = _compute_Gi_hi_batch(PU, centered_images)

        indices = np.asarray(image_indices)
        G_all[indices] = np.asarray(G_batch, dtype=np.float64)
        h_all[indices] = np.asarray(h_batch, dtype=np.float64)

        n_processed += len(indices)
        if n_processed % (batch_size * 10) == 0:
            logger.info("  processed %d / %d images", n_processed, n_images)

    elapsed = time.time() - t0
    logger.info("Computed G_i, h_i for %d images in %.1f s", n_processed, elapsed)
    return G_all, h_all


# ─────────────────────────────────────────────────────────────────────────
# Embeddings in the eigenbasis of B
# ─────────────────────────────────────────────────────────────────────────


def compute_embeddings_from_UB(G_all, h_all, B, scale_to_covariance_convention=True):
    """Build per-image embeddings in the eigenbasis of ``B``.

    Args:
        G_all: (n_images, q, q)
        h_all: (n_images, q)
        B: (q, q) latent covariance.
        scale_to_covariance_convention: If True, scale by sqrt(eigenvalues)
            so the result lives in the same coordinate system the covariance
            pipeline expects.

    Returns:
        latent_coords: (n_images, q)
        latent_precision: (n_images, q, q) — B^{-1} + G_i (precision, not covariance).
    """
    B_inv = np.linalg.inv(B)
    posterior_cov = np.linalg.inv(B_inv[None] + G_all)
    m_all = np.einsum("nij,nj->ni", posterior_cov, h_all)
    precision_all = B_inv[None] + G_all

    eigenvalues, rotation = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    rotation = rotation[:, idx]

    m_rotated = m_all @ rotation
    if scale_to_covariance_convention:
        latent_coords = m_rotated * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]
    else:
        latent_coords = m_rotated

    return latent_coords, precision_all


# ─────────────────────────────────────────────────────────────────────────
# Write a PipelineOutput-compatible result directory from postprocessed PPCA
# ─────────────────────────────────────────────────────────────────────────


_EMBEDDING_FIELDS = [
    "latent_coords", "latent_coords_noreg",
    "latent_precision", "latent_precision_noreg",
    "contrasts", "contrasts_noreg",
]


def _copy_if_exists(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _symlink_safe(src, dst):
    src_abs = os.path.abspath(src)
    if os.path.lexists(dst):
        os.unlink(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src_abs, dst)


def _save_embeddings_per_zdim(paths, embedding_dict):
    all_zdims = set()
    for field in _EMBEDDING_FIELDS:
        if field in embedding_dict and isinstance(embedding_dict[field], dict):
            all_zdims.update(embedding_dict[field].keys())

    for zdim in sorted(all_zdims):
        zdim_dir = paths.embedding_zdim_dir(zdim)
        os.makedirs(zdim_dir, exist_ok=True)
        for field in _EMBEDDING_FIELDS:
            if field in embedding_dict and zdim in embedding_dict[field]:
                arr = np.asarray(embedding_dict[field][zdim])
                np.save(os.path.join(zdim_dir, f"{field}.npy"), arr)
    logger.info("Saved embeddings for zdims %s", sorted(all_zdims))


def create_postprocessed_result_dir(
    source_result_dir,
    output_dir,
    new_u_real,
    new_s,
    new_embeddings_dict,
    method_info=None,
):
    """Write a PipelineOutput-compatible result dir from a postprocessed PPCA fit.

    Copies/symlinks the heavy shared inputs (mean, masks, halfsets) from the
    source result and writes the new eigenvectors, eigenvalues, and embeddings.
    """
    src_paths = ResultPaths(source_result_dir)
    dst_paths = ResultPaths(output_dir)
    dst_paths.ensure_dirs()

    _copy_if_exists(src_paths.halfsets, dst_paths.halfsets)
    _copy_if_exists(src_paths.particles_halfsets, dst_paths.particles_halfsets)
    _copy_if_exists(src_paths.covariance_cols, dst_paths.covariance_cols)

    for attr in ("mean_volume", "mask_volume", "dilated_mask_volume"):
        src = getattr(src_paths, attr)
        dst = getattr(dst_paths, attr)
        if os.path.exists(src):
            _symlink_safe(src, dst)

    src_params = utils.pickle_load(src_paths.params)
    new_params = copy.deepcopy(src_params)

    orig_s = np.asarray(new_params.get("s", np.zeros(0)))
    padded_s = np.zeros(max(len(orig_s), len(new_s)), dtype=np.float32)
    padded_s[: len(new_s)] = new_s
    new_params["s"] = padded_s

    if method_info is not None:
        new_params["ppca_refit_info"] = method_info

    utils.pickle_dump(new_params, dst_paths.params)

    volume_shape = tuple(new_params["volume_shape"])
    voxel_size = float(new_params.get("voxel_size", 1.0))
    for i in range(new_u_real.shape[0]):
        path = dst_paths.eigenvector(i)
        utils.write_mrc(path, new_u_real[i].astype(np.float32), voxel_size=voxel_size)

    n_eigs = min(10, len(new_s))
    variance = np.zeros(volume_shape, dtype=np.float32)
    for i in range(n_eigs):
        variance += float(new_s[i]) * (new_u_real[i].astype(np.float32) ** 2)
    utils.write_mrc(dst_paths.variance(n_eigs), variance, voxel_size=voxel_size)

    if n_eigs != 10:
        variance10 = np.zeros(volume_shape, dtype=np.float32)
        for i in range(min(10, len(new_s), new_u_real.shape[0])):
            variance10 += float(new_s[i]) * (new_u_real[i].astype(np.float32) ** 2)
        utils.write_mrc(dst_paths.variance(10), variance10, voxel_size=voxel_size)

    _save_embeddings_per_zdim(dst_paths, new_embeddings_dict)
    logger.info("Created postprocessed result dir: %s", output_dir)
