"""Probabilistic PCA — fixed-pose closed-form EM solver.

This module implements the canonical Tipping & Bishop (1999) PPCA EM
loop adapted to the cryo-EM forward model with **known per-image
poses**. It is the in-tree reference for the closed-form per-voxel
M-step and is the path that should be cited / extended whenever any
recovar code wants to do PPCA.

Math reference
--------------
The full derivation lives in ``docs/math/ppca_closed_form_mstep.md``.
The short version is:

  Model::

      y_i = CTF_i · S_{t_i} · A_{R_i} · (mu + U alpha_i) + eps_i
      alpha_i ~ N(0, diag(s)),  eps_i ~ N(0, sigma_i^2 I)

  E-step (delegated to ``embedding.get_coords_in_basis_and_contrast_3``)::

      m_i     = posterior mean of alpha_i given y_i
      Hinv_i  = posterior covariance
      C_i     = Hinv_i + m_i m_i^T          # second moment

  M-step for U (per voxel ``v``, **diagonal approximation**
  ``A_g^* (CTF^2 / sigma^2) A_g ≈ diag(Psi_i[v])``)::

      M[v] · U[v, :] = B[v, :]              # q x q linear solve
      M[v]  = sum_i C_i · Psi_i[v]    + lambda I_q
      B[v]  = sum_i m_i · psi_i^B[v]
      Psi_i[v]    = (CTF_i^2 / sigma_i^2) at the pixel where v lands under R_i
      psi_i^B[v]  = (CTF_i / sigma_i^2 · y_i) at the pixel where v lands under R_i

  Gauge fix: SVD orthogonalize ``W = U_svd @ diag(S_svd)`` so the
  columns are orthogonal in the standard L2 inner product.

Why closed form
---------------
PPCA's M-step is famously closed-form (T&B 1999, §3). Solving the
quadratic for U directly avoids the learning-rate / line-search /
inner-loop machinery that gradient-descent variants need, and
side-steps the bistability that gradient PPCA exhibits at
non-trivial init. See ``docs/math/ppca_closed_form_mstep.md`` for
the full discussion.

Pose-marginal extension
-----------------------
For ab-initio (pose-unknown) PPCA, replace the per-image
``sum_i`` with a pose-marginal soft sum ``sum_{i,g} gamma_{i,g}``
over a fixed grid of candidate poses. The same per-voxel q x q
solve applies. The implementation lives at
``recovar/em/ppca_abinitio/factor_update.py::update_factor_closed_form``.

Cross-references
----------------
- ``recovar/em/states.py::HeterogeneousEMState`` uses a closely
  related per-voxel accumulator (``compute_H_B``) for the
  covariance-column path.
- ``recovar/em/ppca_abinitio/factor_update.py::update_factor_closed_form``
  is the pose-marginal v0 ab-initio implementation.
- ``recovar/em/ppca_abinitio/factor_update.py::update_factor_one_outer_step``
  is the **gradient-descent** ab-initio variant. Retained for
  parity testing only; do not use in new code.
"""

import logging

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from recovar import core
from recovar.core import linalg
from recovar.heterogeneity import embedding

logger = logging.getLogger(__name__)


def M_step_batch(
    images,
    lhs_summed,
    rhs_summed,
    mean_batch,
    covariance_batch,
    CTF_params,
    rotation_matrices,
    translations,
    image_shape,
    volume_shape,
    grid_size,
    voxel_size,
    noise_variance,
    ctf,
):
    """Accumulate the per-voxel ``M`` and ``B`` summands for one batch.

    Adds this batch's contribution into the running ``lhs_summed``
    (the per-voxel ``q*q`` matrix ``M[v]``) and ``rhs_summed`` (the
    per-voxel ``q``-vector ``B[v]``). The accumulation uses the
    nearest-gridpoint discretization, so each pixel touches exactly
    one voxel and the ``A_g^* (CTF^2 / sigma^2) A_g`` operator is
    exactly diagonal in the volume basis.

    See module docstring and ``docs/math/ppca_closed_form_mstep.md``
    for the derivation.
    """
    # Precomp piece
    CTF = ctf(CTF_params, image_shape, voxel_size)
    ctf_over_noise_variance = CTF**2 / noise_variance

    grid_point_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape)
    volume_size = np.prod(volume_shape)

    # Second moments: per-image (basis, basis) weighted per-pixel by CTF^2/noise.
    second_moments = covariance_batch + linalg.broadcast_outer(mean_batch, mean_batch)  # (n_images, basis, basis)
    second_moments = second_moments.reshape(second_moments.shape[0], 1, -1)  # (n_images, 1, basis*basis)
    second_moments = second_moments * ctf_over_noise_variance[:, :, None]  # (n_images, pixels, basis*basis)

    lhs_summed = lhs_summed.at[grid_point_indices.reshape(-1)].add(second_moments.reshape(-1, second_moments.shape[-1]))

    images = core.translate_images(images, translations, image_shape)
    images = images * CTF / noise_variance
    images_means_h = linalg.broadcast_outer(images, mean_batch)  # (n_images, pixels, basis)

    rhs_summed = rhs_summed.at[grid_point_indices.reshape(-1)].add(images_means_h.reshape(-1, images_means_h.shape[-1]))

    return lhs_summed, rhs_summed


def M_step(experiment_dataset, latent_means, latent_covariances, noise_variance, batch_size):
    """Closed-form M-step for the PPCA factor ``W``.

    Iterates over batches calling :func:`M_step_batch` to accumulate
    ``lhs_summed`` (the per-voxel ``M[v]`` matrix) and ``rhs_summed``
    (the per-voxel ``B[v]`` vector), then solves the per-voxel
    ``q x q`` linear system

        ``M[v] · W[v, :] = B[v, :]``

    via :func:`linalg.batch_solve`. The result is SVD-orthogonalized
    into the canonical gauge ``W = U @ diag(S)``.

    See module docstring and ``docs/math/ppca_closed_form_mstep.md``
    for the math.
    """
    basis_size = latent_means.shape[-1]
    rhs_summed = jnp.zeros((experiment_dataset.volume_size, basis_size), dtype=experiment_dataset.dtype)
    lhs_summed = jnp.zeros((experiment_dataset.volume_size, basis_size * basis_size), dtype=experiment_dataset.dtype)

    for (
        images,
        rotation_matrices,
        translations,
        ctf_params,
        _noise_variance,
        _particle_indices,
        image_indices,
    ) in experiment_dataset.iter_batches(batch_size):
        lhs_summed, rhs_summed = M_step_batch(
            images,
            lhs_summed,
            rhs_summed,
            latent_means[image_indices],
            latent_covariances[image_indices],
            ctf_params,
            rotation_matrices,
            translations,
            experiment_dataset.image_shape,
            experiment_dataset.volume_shape,
            experiment_dataset.grid_size,
            experiment_dataset.voxel_size,
            noise_variance,
            experiment_dataset.ctf_evaluator,
        )

    # Solve least squares
    lhs_summed = lhs_summed.reshape(experiment_dataset.volume_size, basis_size, basis_size)
    W = linalg.batch_solve(lhs_summed, rhs_summed)
    # Orthogonalize
    U, S, _ = jnp.linalg.svd(W, full_matrices=False)
    W = U @ jnp.diag(S)

    return W


def EM(experiment_dataset, mean_estimate, noise_variance, EM_iter=20, basis_size=10):
    """Standalone PPCA EM driver — fixed-pose, closed-form M-step.

    Initializes ``W`` from random Gaussian noise (FFT'd into volume
    space) and runs ``EM_iter`` outer iterations of E-step
    (:func:`embedding.get_coords_in_basis_and_contrast_3`) followed
    by M-step (:func:`M_step`). Uses ``disc_type="nearest"`` so the
    per-voxel diagonal approximation is exact.

    For pose-marginal ab-initio, see
    ``recovar/em/ppca_abinitio/factor_update.py::update_factor_closed_form``.
    """
    # Initialize
    matrix_key, vector_key = jr.split(jr.PRNGKey(0))
    W = jr.normal(matrix_key, (experiment_dataset.volume_size, basis_size), dtype=experiment_dataset.dtype_real)
    W = linalg.batch_dft3(W, experiment_dataset.volume_shape, basis_size)
    eigenvalue = np.ones(basis_size)
    volume_mask = np.ones(experiment_dataset.volume_shape)
    contrast_grid = np.ones([1])
    batch_size = 1000
    disc_type = "nearest"
    for iter_i in range(EM_iter):
        # E-step
        latent_means, latent_covariances, _ = embedding.get_coords_in_basis_and_contrast_3(
            experiment_dataset,
            mean_estimate,
            W,
            eigenvalue,
            volume_mask,
            noise_variance,
            contrast_grid,
            batch_size,
            disc_type,
            compute_covariances=True,
        )

        # M-step
        W = M_step(experiment_dataset, latent_means, latent_covariances, noise_variance, batch_size)

    return W
