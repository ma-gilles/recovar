"""SOLVAR objectives and fixed-pose optimizer on RECOVAR operators.

This module implements the no-contrast, fixed-pose SOLVAR milestone from
arXiv:2602.17603. Images are whitened with RECOVAR's noise model, projected
with the existing half-spectrum slicing operators, and differentiated through
those operators so the adjoint path remains RECOVAR's backprojection.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core
from recovar.core import linalg
from recovar.ppca import ppca
from recovar.ppca.w_regularization import w_prior_quadratic

logger = logging.getLogger(__name__)

_OBJECTIVES = frozenset({"ls", "mle"})


@dataclass(frozen=True)
class SolvarFitResult:
    """Outputs from a fixed-pose SOLVAR fit."""

    U: jax.Array
    S: jax.Array
    W: jax.Array
    iteration_data: list[dict[str, float]]


def _validate_objective(objective: str) -> str:
    objective = str(objective).lower()
    if objective not in _OBJECTIVES:
        raise ValueError(f"unknown SOLVAR objective {objective!r}; expected one of {sorted(_OBJECTIVES)}")
    return objective


def _half_image_weights(image_shape, dtype):
    w_1d = linalg.half_spectrum_last_axis_weights(image_shape[1], dtype=dtype)
    return jnp.tile(w_1d, image_shape[0]).reshape(-1)


def _weighted_gram(projected_basis, weights):
    weighted_conj = jnp.conj(projected_basis) * weights[None, None, :]
    return jnp.einsum("bkp,bjp->bkj", weighted_conj, projected_basis).real


def _weighted_basis_image_inner(projected_basis, centered_images, weights):
    weighted_conj = jnp.conj(projected_basis) * weights[None, None, :]
    return jnp.einsum("bkp,bp->bk", weighted_conj, centered_images).real


def _weighted_norm_sq(images, weights):
    return jnp.sum(weights[None, :] * jnp.real(jnp.conj(images) * images), axis=-1)


def solvar_image_losses(centered_images, projected_basis, image_shape, *, objective: str):
    """Evaluate per-image SOLVAR LS or MLE losses in whitened image space.

    Parameters
    ----------
    centered_images
        Whitened ``Y_i - P_i mu`` images in rFFT half-spectrum layout,
        shape ``(batch, half_pixels)``.
    projected_basis
        Whitened projected loading volumes ``P_i V`` in rFFT half-spectrum
        layout, shape ``(batch, rank, half_pixels)``.
    image_shape
        Full image shape used to recover full-spectrum inner products from
        packed half-spectrum coefficients.
    objective
        ``"ls"`` for the low-rank least-squares covariance objective or
        ``"mle"`` for the Woodbury maximum-likelihood objective.
    """

    objective = _validate_objective(objective)
    centered_images = jnp.asarray(centered_images)
    projected_basis = jnp.asarray(projected_basis)
    if centered_images.ndim != 2:
        raise ValueError(f"centered_images must have shape (batch, pixels), got {centered_images.shape}")
    if projected_basis.ndim != 3:
        raise ValueError(f"projected_basis must have shape (batch, rank, pixels), got {projected_basis.shape}")
    if projected_basis.shape[0] != centered_images.shape[0] or projected_basis.shape[2] != centered_images.shape[1]:
        raise ValueError(
            "projected_basis and centered_images shape mismatch: "
            f"{projected_basis.shape} vs {centered_images.shape}"
        )

    weights = _half_image_weights(image_shape, centered_images.real.dtype)
    gram = _weighted_gram(projected_basis, weights)
    basis_image = _weighted_basis_image_inner(projected_basis, centered_images, weights)
    image_norm_sq = _weighted_norm_sq(centered_images, weights)

    if objective == "ls":
        gram_sq = jnp.sum(gram * gram, axis=(1, 2))
        image_basis_sq = jnp.sum(basis_image * basis_image, axis=1)
        trace_gram = jnp.trace(gram, axis1=1, axis2=2)
        return image_norm_sq**2 - 2.0 * (image_basis_sq + image_norm_sq) + gram_sq + 2.0 * trace_gram

    rank = projected_basis.shape[1]
    eye = jnp.eye(rank, dtype=gram.dtype)
    M = gram + eye[None, :, :]
    solved = jnp.linalg.solve(M, basis_image[..., None])[..., 0]
    quad = jnp.sum(basis_image * solved, axis=1)
    sign, logabsdet = jnp.linalg.slogdet(M)
    return image_norm_sq - quad + logabsdet


def _prepare_batch(
    W_half,
    images_half,
    mean_for_slicing,
    ctf_params,
    rotation_matrices,
    translations,
    noise_variance_half,
    image_shape,
    volume_shape,
    voxel_size,
    ctf_evaluator,
    disc_type_mean,
    disc_type,
):
    images_half = core.translate_images(images_half, translations, image_shape, half_image=True)
    images_half = images_half / jnp.sqrt(noise_variance_half)
    ctf_half = ctf_evaluator(ctf_params, image_shape, voxel_size, half_image=True)
    ctf_half = ctf_half / jnp.sqrt(noise_variance_half)
    projected_mean = core.slice_volume(
        mean_for_slicing,
        rotation_matrices,
        image_shape,
        volume_shape,
        disc_type_mean,
        half_image=True,
    )
    centered = images_half - projected_mean * ctf_half
    projected_basis = ppca.batch_over_vol_slice_volume_half(
        W_half,
        rotation_matrices,
        image_shape,
        volume_shape,
        disc_type,
    )
    projected_basis = projected_basis * ctf_half[:, None, :]
    return centered, projected_basis


def _batch_total_loss(
    W_half,
    W_prior_half,
    images_half,
    mean_for_slicing,
    ctf_params,
    rotation_matrices,
    translations,
    noise_variance_half,
    image_shape,
    volume_shape,
    voxel_size,
    ctf_evaluator,
    disc_type_mean,
    disc_type,
    objective,
    data_scale,
):
    centered, projected_basis = _prepare_batch(
        W_half,
        images_half,
        mean_for_slicing,
        ctf_params,
        rotation_matrices,
        translations,
        noise_variance_half,
        image_shape,
        volume_shape,
        voxel_size,
        ctf_evaluator,
        disc_type_mean,
        disc_type,
    )
    data_loss = jnp.sum(solvar_image_losses(centered, projected_basis, image_shape, objective=objective))
    prior_loss = w_prior_quadratic(W_half, W_prior_half)
    return data_scale * data_loss + prior_loss


def make_random_loading(volume_shape, basis_size: int, *, seed: int = 0, init_scale: float = 0.01):
    """Build deterministic random real loading volumes in half-Fourier layout."""

    rng = np.random.default_rng(seed)
    volume_shape = tuple(int(s) for s in volume_shape)
    half_volume_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    W_half = np.empty((half_volume_size, int(basis_size)), dtype=np.complex64)
    for j in range(int(basis_size)):
        real_volume = rng.standard_normal(volume_shape).astype(np.float32) * float(init_scale)
        W_half[:, j] = ftu.get_dft3_real(real_volume).reshape(-1)
    return W_half


def make_loading_from_basis(u_rescaled, s_rescaled, basis_size: int, volume_shape):
    """Pack a covariance/PCA basis into SOLVAR's half-Fourier loading matrix.

    RECOVAR's covariance path represents a low-rank covariance as
    ``U diag(s) U*`` in full Fourier volume layout. SOLVAR optimizes the square
    root ``W`` such that ``W W*`` is that covariance, so the warm start is
    ``W = U sqrt(s)`` repacked to the Hermitian half-volume layout used by the
    projection code.
    """

    volume_shape = tuple(int(s) for s in volume_shape)
    volume_size = int(np.prod(volume_shape))
    basis_size = int(basis_size)
    u_rescaled = np.asarray(u_rescaled)
    s_rescaled = np.asarray(s_rescaled)
    if u_rescaled.ndim != 2:
        raise ValueError(f"u_rescaled must be 2D, got shape {u_rescaled.shape}")
    if u_rescaled.shape[0] == volume_size:
        u_full = u_rescaled[:, :basis_size]
    elif u_rescaled.shape[1] == volume_size:
        u_full = u_rescaled[:basis_size, :].T
    else:
        raise ValueError(
            f"u_rescaled shape {u_rescaled.shape} is incompatible with volume size {volume_size}"
        )
    if u_full.shape[1] < basis_size or s_rescaled.shape[0] < basis_size:
        raise ValueError(
            f"requested basis_size={basis_size}, got U shape {u_rescaled.shape} and s shape {s_rescaled.shape}"
        )
    scales = np.sqrt(np.maximum(s_rescaled[:basis_size], 0.0)).astype(np.float32)
    W_full = u_full * scales[None, :]
    return ftu.full_volume_to_half_volume(W_full.T, volume_shape).T.astype(np.complex64)


def project_loading_to_mask(W_half, volume_shape, volume_mask=None):
    """Project half-Fourier loadings to real volumes, apply mask, and repack."""

    W_half = jnp.asarray(W_half)
    volume_shape = tuple(int(s) for s in volume_shape)
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    rank = int(W_half.shape[1])
    W_real = ftu.get_idft3_real(W_half.T.reshape(rank, *half_shape), volume_shape)
    if volume_mask is not None:
        W_real = W_real * jnp.asarray(volume_mask, dtype=W_real.dtype).reshape((1, *volume_shape))
    return ftu.get_dft3_real(W_real).reshape(rank, -1).T.astype(W_half.dtype)


def _as_half_volume_prior(W_prior, W_shape, volume_shape):
    W_prior = jnp.asarray(W_prior)
    if W_prior.shape == W_shape:
        return W_prior
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    if W_prior.shape[0] == half_size:
        return W_prior
    return ftu.full_volume_to_half_volume(W_prior.T, volume_shape).T


def _adam_step(W, grad, m, v, t, *, learning_rate, beta1, beta2, eps):
    # JAX returns conjugated Wirtinger gradients for real-valued functions of
    # complex inputs. Conjugate once here so ``W -= step`` is steepest descent
    # in the real/imaginary coordinates.
    descent_grad = jnp.conj(grad)
    m = beta1 * m + (1.0 - beta1) * descent_grad
    v = beta2 * v + (1.0 - beta2) * (jnp.abs(descent_grad) ** 2)
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    W = W - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
    return W, m, v


def fit(
    experiment_dataset,
    mean_estimate,
    W_initial,
    W_prior,
    *,
    objective: str = "mle",
    n_epochs: int = 40,
    batch_size: int = 200,
    learning_rate: float = 1e-6,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_eps: float = 1e-8,
    gradient_clip_norm: float = 0.0,
    volume_mask=None,
    project_mask: bool = True,
    disc_type_mean: str = "cubic",
    disc_type: str = "linear_interp",
    seed: int | None = None,
    return_iteration_data: bool = False,
):
    """Fit SOLVAR fixed-pose loadings with the LS or MLE objective.

    The optimized objective is the paper's low-rank data term plus the same
    PPCA loading prior used by :mod:`recovar.ppca`: the full-data data loss is
    estimated from each mini-batch and ``sum |W|^2/(W_prior+floor)`` is added
    once per update.
    """

    del seed  # Reserved for future shuffled data iteration.
    objective = _validate_objective(objective)
    if getattr(experiment_dataset, "tilt_series_flag", False):
        raise ValueError("SOLVAR fixed-pose implementation currently supports SPA image datasets only")
    if int(n_epochs) <= 0:
        raise ValueError(f"n_epochs must be positive, got {n_epochs}")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    halfset_datasets = ppca._materialize_halfsets(experiment_dataset)
    volume_shape = tuple(int(s) for s in experiment_dataset.volume_shape)
    W = jnp.asarray(W_initial, dtype=jnp.complex64)
    W = project_loading_to_mask(W, volume_shape, volume_mask if project_mask else None)
    W_prior_half = _as_half_volume_prior(W_prior, W.shape, volume_shape)
    W_prior_half = jnp.asarray(W_prior_half, dtype=W.real.dtype)

    mean_for_slicing = ppca._prepare_mean_estimate_for_slicing(
        mean_estimate,
        mean_estimate,
        volume_shape,
        disc_type_mean,
    )
    mean_for_slicing = jnp.asarray(mean_for_slicing)

    m = jnp.zeros_like(W)
    v = jnp.zeros(W.shape, dtype=W.real.dtype)
    n_total = int(experiment_dataset.n_images)
    iteration_data: list[dict[str, float]] = []
    step = 0

    logger.info(
        "SOLVAR fit: objective=%s epochs=%d rank=%d batch_size=%d learning_rate=%.3e",
        objective,
        int(n_epochs),
        int(W.shape[1]),
        int(batch_size),
        float(learning_rate),
    )

    for epoch in range(int(n_epochs)):
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        epoch_batches = 0
        for ds in halfset_datasets:
            for batch_half, ctf_params, rotation_matrices, translations, batch_image_ind in ppca._iter_processed_batches_half(
                ds, int(batch_size)
            ):
                batch_n = int(batch_half.shape[0])
                data_scale = float(n_total) / float(batch_n)
                noise_variance_half = ds.noise.get_half(batch_image_ind)

                def loss_for_W(W_candidate):
                    return _batch_total_loss(
                        W_candidate,
                        W_prior_half,
                        batch_half,
                        mean_for_slicing,
                        ctf_params,
                        rotation_matrices,
                        translations,
                        noise_variance_half,
                        ds.image_shape,
                        ds.volume_shape,
                        ds.voxel_size,
                        ds.ctf_evaluator,
                        disc_type_mean,
                        disc_type,
                        objective,
                        data_scale,
                    )

                loss, grad = jax.value_and_grad(loss_for_W)(W)
                grad_norm = float(jnp.linalg.norm(grad))
                if gradient_clip_norm and gradient_clip_norm > 0.0 and grad_norm > gradient_clip_norm:
                    grad = grad * (float(gradient_clip_norm) / (grad_norm + 1e-12))
                    grad_norm = float(gradient_clip_norm)
                step += 1
                W, m, v = _adam_step(
                    W,
                    grad,
                    m,
                    v,
                    step,
                    learning_rate=float(learning_rate),
                    beta1=float(beta1),
                    beta2=float(beta2),
                    eps=float(adam_eps),
                )
                if project_mask:
                    W = project_loading_to_mask(W, volume_shape, volume_mask)
                epoch_loss += float(loss)
                epoch_grad_norm += grad_norm
                epoch_batches += 1

        prior_loss = float(w_prior_quadratic(W, W_prior_half))
        row = {
            "epoch": float(epoch),
            "loss_mean_batch_estimate": epoch_loss / max(epoch_batches, 1),
            "prior_loss": prior_loss,
            "grad_norm_mean": epoch_grad_norm / max(epoch_batches, 1),
            "W_norm": float(jnp.linalg.norm(W)),
        }
        iteration_data.append(row)
        logger.info(
            "SOLVAR epoch %d/%d loss=%.6e prior=%.6e grad_norm=%.6e",
            epoch + 1,
            int(n_epochs),
            row["loss_mean_batch_estimate"],
            row["prior_loss"],
            row["grad_norm_mean"],
        )

    U_real, eigenvalues, _ = ppca._orthonormalize_W_to_basis(W, volume_shape)
    rank = U_real.shape[0]
    U_half = ftu.get_dft3_real(jnp.asarray(U_real)).reshape(rank, -1).T
    result = SolvarFitResult(
        U=U_half,
        S=jnp.asarray(np.maximum(eigenvalues, 0.0).astype(np.float32)),
        W=W,
        iteration_data=iteration_data,
    )
    if return_iteration_data:
        return result
    return result.U, result.S, result.W
