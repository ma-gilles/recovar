"""SOLVAR objectives and fixed-pose optimizer on RECOVAR operators.

This module implements the no-contrast, fixed-pose SOLVAR milestone from
arXiv:2602.17603. Images are whitened with RECOVAR's noise model, projected
with the existing half-spectrum slicing operators, and differentiated through
those operators so the adjoint path remains RECOVAR's backprojection.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import recovar.core.fourier_transform_utils as ftu
from recovar import core
from recovar.core import linalg
from recovar.ppca import ppca
from recovar.ppca.w_regularization import w_prior_quadratic
from recovar.simulation.synthetic_dataset import HeterogeneousVolumeDistribution
from recovar.solvar import gt_metrics

logger = logging.getLogger(__name__)

_OBJECTIVES = frozenset({"ls", "mle"})


@dataclass(frozen=True)
class SolvarFitResult:
    """Outputs from a fixed-pose SOLVAR fit."""

    U: jax.Array
    S: jax.Array
    W: jax.Array
    iteration_data: list[dict]


class WHalfParametrization(NamedTuple):
    """
    Parameterizes the half-Fourier loading matrix ``W`` for gradient-based optimization by:
    ``W = U * exp(log_sqrt_eigenvalues)``.
    """

    U: jax.Array
    log_sqrt_eigenvalues: jax.Array

    def apply_masking(self, volume_shape, volume_mask) -> "WHalfParametrization":
        """Apply a real-space volume mask."""
        return WHalfParametrization(U=project_loading_to_mask(self.U, volume_shape, volume_mask), log_sqrt_eigenvalues=self.log_sqrt_eigenvalues)


def loadings_from_state(state: WHalfParametrization) -> jax.Array:
    """Reconstruct the half-Fourier loading matrix ``W = U * exp(log_sqrt_eigenvalues)``.
    """
    scale = jnp.exp(state.log_sqrt_eigenvalues)
    return state.U * scale[None, :]


def _state_from_loadings(W_half, volume_shape) -> WHalfParametrization:
    """Rebase ``W_half`` onto its exact SVD basis: orthonormal ``U`` + log singular values.
    """
    U_real, eigenvalues, _ = ppca._orthonormalize_W_to_basis(W_half, volume_shape)
    rank = U_real.shape[0]
    U_half = ftu.get_dft3_real(jnp.asarray(U_real)).reshape(rank, -1).T.astype(W_half.dtype)
    log_sqrt_eigenvalues = 0.5 * jnp.log(jnp.asarray(eigenvalues, dtype=U_half.real.dtype))
    return WHalfParametrization(U=U_half, log_sqrt_eigenvalues=log_sqrt_eigenvalues)


class TrainConfig(eqx.Module):
    """Structural configuration for one :func:`fit` run — fixed for its whole duration.

    All fields are static (compile-time constants): changing any of them
    triggers a JIT recompilation.
    """

    image_shape: tuple = eqx.field(static=True)
    volume_shape: tuple = eqx.field(static=True)
    ctf_evaluator: core.CTFEvaluator = eqx.field(static=True)
    disc_type_mean: str = eqx.field(static=True)
    disc_type: str = eqx.field(static=True)
    objective: str = eqx.field(static=True)
    project_mask: bool = eqx.field(static=True)
    optimizer: optax.GradientTransformation = eqx.field(static=True)


class TrainArrs(NamedTuple):
    """Arrays fixed for the whole ``fit()`` run but too large to bake in as JIT constants."""

    W_prior_half: jax.Array
    mean_for_slicing: jax.Array
    volume_mask: jax.Array | None


class BatchStruct(NamedTuple):
    """One mini-batch of images and per-image pose/CTF/noise data."""

    images_half: jax.Array
    ctf_params: jax.Array
    rotation_matrices: jax.Array
    translations: jax.Array
    noise_variance_half: jax.Array
    voxel_size: jax.Array
    data_scale: jax.Array



@jax.jit
def _train_step(params: WHalfParametrization, opt_state: optax.OptState, batch: BatchStruct, tensor_data: TrainArrs, config: TrainConfig):
    """
    Batched SOLVAR SGD step.
    """

    def loss_for_params(params):
        masked_params = params.apply_masking(config.volume_shape, tensor_data.volume_mask) if config.project_mask else params
        return _batch_total_loss(
            loadings_from_state(params),
            tensor_data.W_prior_half,
            batch.images_half,
            tensor_data.mean_for_slicing,
            batch.ctf_params,
            batch.rotation_matrices,
            batch.translations,
            batch.noise_variance_half,
            config.image_shape,
            config.volume_shape,
            batch.voxel_size,
            config.ctf_evaluator,
            config.disc_type_mean,
            config.disc_type,
            config.objective,
            batch.data_scale,
        )

    loss, grad = jax.value_and_grad(loss_for_params)(params)
    # JAX's grad of a real-valued function w.r.t. a complex input is the
    # non-conjugated Wirtinger derivative, not the descent direction — conjugate
    # once here so `params + updates` is steepest descent (no-op on the real
    # log_sqrt_eigenvalues leaf).
    grad = jax.tree.map(jnp.conj, grad)
    updates, opt_state = config.optimizer.update(grad, opt_state, params, value=loss)
    params = optax.apply_updates(params, updates)
    if config.project_mask:
        params = params.apply_masking(config.volume_shape, tensor_data.volume_mask)
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.abs(g) ** 2) for g in jax.tree.leaves(grad)))
    return params, opt_state, loss, grad_norm


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
    return jnp.einsum("bkp,bjp->bkj", weighted_conj, projected_basis)


def _weighted_basis_image_inner(projected_basis, centered_images, weights):
    weighted_conj = jnp.conj(projected_basis) * weights[None, None, :]
    return jnp.einsum("bkp,bp->bk", weighted_conj, centered_images)


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
        gram_sq = jnp.sum(gram * jnp.conj(gram), axis=(1, 2)).real
        image_basis_sq = jnp.sum(basis_image * jnp.conj(basis_image), axis=1).real
        trace_gram = jnp.trace(gram, axis1=1, axis2=2).real
        return image_norm_sq**2 - 2.0 * (image_basis_sq + image_norm_sq) + gram_sq + 2.0 * trace_gram

    rank = projected_basis.shape[1]
    eye = jnp.eye(rank, dtype=gram.dtype)
    M = gram + eye[None, :, :]
    solved = jnp.linalg.solve(M, basis_image[..., None])[..., 0]
    quad = jnp.sum(jnp.conj(basis_image) * solved, axis=1).real
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


def _branch_optimizer(branch_learning_rate, gradient_clip_norm: float = 0.0, scheduler_patience: int = 1, scheduler_factor: float = 0.1):
    transformations = [
        optax.adam(branch_learning_rate),
        optax.contrib.reduce_on_plateau(
            factor=scheduler_factor,
            patience = scheduler_patience,
            rtol = 1e-4,
            cooldown = 3,
            #TODO: Scheduler should be fed the per epoch loss
            # which requires seperating it from the optimizer update step.
            # For now a large accumulation size approximates the per epoch loss.
            accumulation_size=100,
        )
    ]
    if gradient_clip_norm and gradient_clip_norm > 0.0:
        transformations.insert(0, optax.clip_by_global_norm(gradient_clip_norm))

    return optax.chain(*transformations)


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
    gradient_clip_norm: float = 0.0,
    log_eigenvalue_lr_factor: float = 100.0,
    volume_mask=None,
    project_mask: bool = True,
    disc_type_mean: str = "cubic",
    disc_type: str = "linear_interp",
    seed: int | None = None,
    gt_data: HeterogeneousVolumeDistribution | None = None,
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

    optimizer = optax.multi_transform(
        {
            "U": _branch_optimizer(learning_rate, gradient_clip_norm),
            "log_sqrt_eigenvalues": _branch_optimizer(learning_rate * log_eigenvalue_lr_factor, gradient_clip_norm),
        },
        param_labels=WHalfParametrization("U", "log_sqrt_eigenvalues"),
    )
    params = _state_from_loadings(W, volume_shape)
    opt_state = optimizer.init(params)
    tensor_data = TrainArrs(W_prior_half=W_prior_half, mean_for_slicing=mean_for_slicing, volume_mask=volume_mask)
    # image_shape/volume_shape/ctf_evaluator are assumed identical across halfsets (only the
    # image subset differs), so one config covers every batch of both.
    config = TrainConfig(
        image_shape=halfset_datasets[0].image_shape,
        volume_shape=halfset_datasets[0].volume_shape,
        ctf_evaluator=halfset_datasets[0].ctf_evaluator,
        disc_type_mean=disc_type_mean,
        disc_type=disc_type,
        objective=objective,
        project_mask=project_mask,
        optimizer=optimizer,
    )

    n_total = int(experiment_dataset.n_images)
    iteration_data: list[dict] = []

    logger.info(
        "SOLVAR fit: objective=%s epochs=%d rank=%d batch_size=%d learning_rate=%.3e log_eigenvalue_lr_factor=%.1f",
        objective,
        int(n_epochs),
        int(params.U.shape[1]),
        int(batch_size),
        float(learning_rate),
        float(log_eigenvalue_lr_factor),
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
                batch = BatchStruct(
                    images_half=batch_half,
                    ctf_params=ctf_params,
                    rotation_matrices=rotation_matrices,
                    translations=translations,
                    noise_variance_half=ds.noise.get_half(batch_image_ind),
                    voxel_size=ds.voxel_size,
                    data_scale=float(n_total) / float(batch_n),
                )

                params, opt_state, loss, grad_norm = _train_step(params, opt_state, batch, tensor_data, config)

                epoch_loss += float(loss)
                epoch_grad_norm += float(grad_norm)
                epoch_batches += 1

        W_current = loadings_from_state(params)
        prior_loss = float(w_prior_quadratic(W_current, W_prior_half))
        row = {
            "epoch": float(epoch),
            "loss_mean_batch_estimate": epoch_loss / max(epoch_batches, 1),
            "prior_loss": prior_loss,
            "grad_norm_mean": epoch_grad_norm / max(epoch_batches, 1),
            "W_norm": float(jnp.linalg.norm(W_current)),
            "step_scaling": float(opt_state[-1]['U'].inner_state[-1].scale.real),
        }
        if gt_data is not None:
            row.update(gt_metrics.compute_eigenvector_metrics(W_current, gt_data, volume_shape))
        iteration_data.append(row)
        logger.info(
            "SOLVAR epoch %d/%d loss=%.6e prior=%.6e grad_norm=%.6e step scaling=%.2e",
            epoch + 1,
            int(n_epochs),
            row["loss_mean_batch_estimate"],
            row["prior_loss"],
            row["grad_norm_mean"],
            row["step_scaling"],
        )
        if gt_data is not None:
            logger.info(
                "SOLVAR epoch %d/%d gt_relative_variance=%.4f gt_cosine_similarity=%.4f gt_fro_relative_error=%.4f",
                epoch + 1,
                int(n_epochs),
                row["gt_relative_variance"],
                row["gt_cosine_similarity"],
                row["gt_fro_relative_error"],
            )


        if row["step_scaling"] < 1e-4:
            logger.info("SOLVAR epoch %d/%d step scaling below threshold, stopping early", epoch + 1, int(n_epochs))
            break

    params = params.apply_masking(config.volume_shape, tensor_data.volume_mask) if config.project_mask else params
    W = loadings_from_state(params)
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
