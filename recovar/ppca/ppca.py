"""Direct PPCA implementation ported onto the current recovar APIs."""

import functools
import logging

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core, utils
from recovar.core import linalg
from recovar.heterogeneity import covariance_estimation
from recovar.output import metrics

logger = logging.getLogger(__name__)


def _normalize_experiment_datasets(experiment_datasets):
    if hasattr(experiment_datasets, "materialize_halfset_datasets"):
        return experiment_datasets, list(experiment_datasets.materialize_halfset_datasets())
    if isinstance(experiment_datasets, (list, tuple)):
        datasets = list(experiment_datasets)
        if not datasets:
            raise ValueError("experiment_datasets cannot be empty")
        full_dataset = None
        if len(datasets) == 1 and hasattr(datasets[0], "materialize_halfset_datasets"):
            full_dataset = datasets[0]
            datasets = list(full_dataset.materialize_halfset_datasets())
        return full_dataset, datasets
    raise TypeError("experiment_datasets must be a CryoEMDataset or a sequence of CryoEMDataset objects")


def _iter_processed_batches(experiment_dataset, batch_size):
    for (
        batch,
        rotation_matrices,
        translations,
        ctf_params,
        _noise_variance,
        _particle_indices,
        image_indices,
    ) in experiment_dataset.iter_batches(
        batch_size,
        by_image=not getattr(experiment_dataset, "tilt_series_flag", False),
    ):
        yield (
            experiment_dataset.process_images(batch, apply_image_mask=False),
            ctf_params,
            rotation_matrices,
            translations,
            image_indices,
        )


def _forward_model_from_map(
    volume,
    ctf_params,
    rotation_matrices,
    image_shape,
    volume_shape,
    voxel_size,
    ctf_evaluator,
    disc_type,
    skip_ctf=False,
):
    slices = core.slice_volume(
        volume,
        rotation_matrices,
        image_shape,
        volume_shape,
        disc_type,
    )
    if not skip_ctf:
        slices = slices * ctf_evaluator(ctf_params, image_shape, voxel_size)
    return slices


# =============================================================================
# WHITENING CONSTRAINT IMPLEMENTATION
# =============================================================================


def sqrtm_psd(C):
    """
    Compute the matrix square root of a positive semi-definite matrix.
    Uses eigendecomposition: C = V @ diag(λ) @ V^T => C^{1/2} = V @ diag(√λ) @ V^T
    """
    if jnp.any(jnp.isnan(C)) or jnp.any(jnp.isinf(C)):
        logger.error("sqrtm_psd: input C has NaN/Inf; min=%s max=%s", float(jnp.nanmin(C)), float(jnp.nanmax(C)))
        raise ValueError("sqrtm_psd received matrix with NaN/Inf")
    eigvals, eigvecs = jnp.linalg.eigh(C)
    if jnp.any(jnp.isnan(eigvals)):
        logger.error("sqrtm_psd: eigh produced NaN eigenvalues (input may be non-PSD or ill-conditioned)")
        raise ValueError("sqrtm_psd: eigh produced NaN eigenvalues")
    eigvals = jnp.clip(eigvals, 1e-12, None)  # Ensure positive
    sqrt_eigvals = jnp.sqrt(eigvals)
    if jnp.any(jnp.isnan(sqrt_eigvals)):
        logger.error("sqrtm_psd: sqrt(eigvals) produced NaN")
        raise ValueError("sqrtm_psd: sqrt(eigvals) produced NaN")
    return (eigvecs * sqrt_eigvals) @ eigvecs.T


def compute_sigma_proj_ls(
    experiment_datasets,
    mean_estimate_raw,
    W,
    volume_mask,
    batch_size,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    do_mask_images=True,
    parallel_analysis=False,
):
    """
    Compute Sigma(W) by projected-covariance least squares using existing
    covariance_estimation.compute_projected_covariance.
    """
    del parallel_analysis
    if mean_estimate_raw is None:
        raise ValueError("mean_estimate_raw is required for proj_ls whitening.")
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_datasets)
    dataset_for_covariance = full_dataset
    if dataset_for_covariance is None:
        if len(dataset_list) == 1 and getattr(dataset_list[0], "halfset_indices", None) is not None:
            dataset_for_covariance = dataset_list[0]
        else:
            raise ValueError("proj_ls whitening requires a CryoEMDataset with halfset_indices set")
    covar = covariance_estimation.compute_projected_covariance(
        dataset_for_covariance,
        mean_estimate_raw,
        W,
        volume_mask,
        batch_size,
        disc_type_mean,
        disc_type,
        do_mask_images=do_mask_images,
    )
    covar = 0.5 * (covar + covar.T)
    return covar


def compute_Cz_from_second_moments(second_moment_zs):
    """
    Compute the empirical posterior covariance Ĉ_z = (1/N) Σ_n E[z_n z_n^T | y_n].

    Args:
        second_moment_zs: Array of shape (N, q, q) containing E[z_n z_n^T | y_n] for each sample
                          where E[z_n z_n^T | y_n] = Σ_n + μ_n μ_n^T

    Returns:
        C_z: The empirical posterior covariance of shape (q, q)
    """
    return jnp.mean(second_moment_zs, axis=0)


def compute_em_ll_and_whitening_grad(
    experiment_datasets, mean_estimate, W, batch_size, disc_type_mean="cubic", disc_type="linear_interp"
):
    """
    Compute EM data log-likelihood and whitening penalty/gradient in two passes.

    This does NOT change any existing EM behavior and only incurs extra compute
    when this function is explicitly called.

    Returns:
        ll_sum: summed data log-likelihood over all images
        F: whitening penalty 0.5 * ||Cz - I||_F^2
        grad_W: gradient of F w.r.t. W (same shape as W)
        C_z: empirical posterior covariance
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_datasets)
    reference_dataset = full_dataset if full_dataset is not None else dataset_list[0]
    q = W.shape[1]
    I = jnp.eye(q, dtype=W.dtype)

    # -------------------------------------------------------------------------
    # Pass 1: compute ll and Cz (via second moments)
    # -------------------------------------------------------------------------
    ll_sum = jnp.array(0.0, dtype=reference_dataset.dtype)
    Cz_sum = jnp.zeros((q, q), dtype=W.dtype)
    n_total = 0

    for experiment_dataset in dataset_list:
        lhs_dummy = jnp.zeros((experiment_dataset.volume_size, q * q), dtype=experiment_dataset.dtype_real)
        rhs_dummy = jnp.zeros((experiment_dataset.volume_size, q), dtype=experiment_dataset.dtype)
        for batch, ctf_params, rotation_matrices, translations, batch_image_ind in _iter_processed_batches(
            experiment_dataset, batch_size
        ):
            noise_variance = experiment_dataset.noise.get(batch_image_ind)
            _, _, _, second_moment_zs_batch, ll_sum_batch, _ = E_M_step_batch(
                batch,
                lhs_dummy,
                rhs_dummy,
                mean_estimate,
                W,
                ctf_params,
                rotation_matrices,
                translations,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                experiment_dataset.grid_size,
                experiment_dataset.voxel_size,
                noise_variance,
                experiment_dataset.ctf_evaluator,
                compute_ll=True,
                disc_type_mean=disc_type_mean,
                disc_type=disc_type,
                compute_stats=False,
            )
            ll_sum += ll_sum_batch
            Cz_sum += jnp.sum(second_moment_zs_batch, axis=0)
            n_total += batch.shape[0]

    if n_total == 0:
        raise ValueError("No images found in datasets.")

    C_z = Cz_sum / float(n_total)
    G = C_z - I
    F = 0.5 * jnp.sum(G * G)

    # -------------------------------------------------------------------------
    # Pass 2: compute whitening gradient using G
    # -------------------------------------------------------------------------
    grad_sum = jnp.zeros_like(W)

    for experiment_dataset in dataset_list:
        for images, ctf_params, rotation_matrices, translations, batch_image_ind in _iter_processed_batches(
            experiment_dataset, batch_size
        ):
            noise_variance = experiment_dataset.noise.get(batch_image_ind)
            images = core.translate_images(
                images,
                translations,
                experiment_dataset.image_shape,
            ) / jnp.sqrt(noise_variance)

            CTF = experiment_dataset.ctf_evaluator(
                ctf_params,
                experiment_dataset.image_shape,
                experiment_dataset.voxel_size,
            ) / jnp.sqrt(noise_variance)
            projected_mean = _forward_model_from_map(
                mean_estimate,
                ctf_params,
                rotation_matrices,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                experiment_dataset.voxel_size,
                experiment_dataset.ctf_evaluator,
                disc_type_mean,
                skip_ctf=False,
            ) / jnp.sqrt(noise_variance)

            centered_images = images - projected_mean
            ctf_squared_over_noise_variance = CTF**2

            PW = batch_over_vol_slice_volume(
                W,
                rotation_matrices,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                disc_type,
            )
            PW *= CTF[..., None, :]

            M_n = (jnp.conj(PW) @ PW.transpose(0, 2, 1)).real + jnp.eye(q)
            btilde = (jnp.conj(PW) @ centered_images[..., None]).real  # (b, q, 1)
            btilde = btilde.squeeze(-1)

            Sigma = jnp.linalg.pinv(M_n, hermitian=True)
            mu = (Sigma @ btilde[..., None]).squeeze(-1)

            u = (G @ mu[..., None]).squeeze(-1)
            suT = btilde[:, :, None] * u[:, None, :]
            H = G[None, :, :] + suT + jnp.swapaxes(suT, -1, -2)
            K = jnp.matmul(Sigma, jnp.matmul(H, Sigma))

            # term1: -2 * sum_n B_n W K_n
            before_backproj_bw = ctf_squared_over_noise_variance[..., None] * PW.transpose(0, 2, 1)
            weighted_bw = before_backproj_bw[..., :, None] * K[:, None, :, :]
            weighted_bw = weighted_bw.reshape(weighted_bw.shape[0], weighted_bw.shape[1], q * q)
            bw_backproj = batch_over_vol_adjoint_slice_volume(
                weighted_bw,
                rotation_matrices,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                disc_type,
            )
            bw_backproj = bw_backproj.reshape(experiment_dataset.volume_size, q, q)
            term1 = -2.0 * jnp.sum(bw_backproj, axis=1)

            # term2:  2 * sum_n b_n (Sigma u)^T
            alpha = (Sigma @ u[..., None]).squeeze(-1)
            before_backproj_bn = CTF[..., None] * centered_images[..., None] * alpha[:, None, :]
            bn_backproj = batch_over_vol_adjoint_slice_volume(
                before_backproj_bn,
                rotation_matrices,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                disc_type,
            )
            term2 = 2.0 * bn_backproj

            grad_sum = grad_sum + term1 + term2

    grad_W = grad_sum / float(n_total)
    return ll_sum, F, grad_W, C_z


def compute_em_ll_and_whitening_grads(
    experiment_datasets, mean_estimate, W, batch_size, W_prior=None, disc_type_mean="cubic", disc_type="linear_interp"
):
    """
    Compute EM data log-likelihood, whitening penalty/gradient, and the M-step
    quadratic objective gradient (a proxy for -LL used in EM).

    This function does extra compute only when called.

    Returns:
        ll_sum: summed data log-likelihood over all images
        grad_neg_ll: gradient of the EM M-step quadratic objective
        F: whitening penalty 0.5 * ||Cz - I||_F^2
        grad_F: gradient of F w.r.t. W
        C_z: empirical posterior covariance
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_datasets)
    reference_dataset = full_dataset if full_dataset is not None else dataset_list[0]
    q = W.shape[1]

    # Pass 1: compute ll, Cz, and M-step sufficient stats
    ll_sum = jnp.array(0.0, dtype=reference_dataset.dtype)
    Cz_sum = jnp.zeros((q, q), dtype=W.dtype)
    n_total = 0

    lhs_summed = jnp.zeros((reference_dataset.volume_size, q * q), dtype=reference_dataset.dtype_real)
    rhs_summed = jnp.zeros((reference_dataset.volume_size, q), dtype=reference_dataset.dtype)

    for experiment_dataset in dataset_list:
        for batch, ctf_params, rotation_matrices, translations, batch_image_ind in _iter_processed_batches(
            experiment_dataset, batch_size
        ):
            noise_variance = experiment_dataset.noise.get(batch_image_ind)
            lhs_summed, rhs_summed, _, second_moment_zs_batch, ll_sum_batch, _ = E_M_step_batch(
                batch,
                lhs_summed,
                rhs_summed,
                mean_estimate,
                W,
                ctf_params,
                rotation_matrices,
                translations,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                experiment_dataset.grid_size,
                experiment_dataset.voxel_size,
                noise_variance,
                experiment_dataset.ctf_evaluator,
                compute_ll=True,
                disc_type_mean=disc_type_mean,
                disc_type=disc_type,
                compute_stats=True,
            )
            ll_sum += ll_sum_batch
            Cz_sum += jnp.sum(second_moment_zs_batch, axis=0)
            n_total += batch.shape[0]

    if n_total == 0:
        raise ValueError("No images found in datasets.")

    C_z = Cz_sum / float(n_total)
    G = C_z - jnp.eye(q, dtype=W.dtype)
    F = 0.5 * jnp.sum(G * G)

    lhs_summed = lhs_summed.reshape(reference_dataset.volume_size, q, q)
    grad_neg_ll = jnp.einsum("vij,vj->vi", lhs_summed, W) - rhs_summed
    if W_prior is not None:
        grad_neg_ll = grad_neg_ll + W / (W_prior + 1e-16)

    # Pass 2: whitening gradient using G
    grad_sum = jnp.zeros_like(W)

    for experiment_dataset in dataset_list:
        for images, ctf_params, rotation_matrices, translations, batch_image_ind in _iter_processed_batches(
            experiment_dataset, batch_size
        ):
            noise_variance = experiment_dataset.noise.get(batch_image_ind)
            images = core.translate_images(
                images,
                translations,
                experiment_dataset.image_shape,
            ) / jnp.sqrt(noise_variance)

            CTF = experiment_dataset.ctf_evaluator(
                ctf_params,
                experiment_dataset.image_shape,
                experiment_dataset.voxel_size,
            ) / jnp.sqrt(noise_variance)
            projected_mean = _forward_model_from_map(
                mean_estimate,
                ctf_params,
                rotation_matrices,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                experiment_dataset.voxel_size,
                experiment_dataset.ctf_evaluator,
                disc_type_mean,
                skip_ctf=False,
            ) / jnp.sqrt(noise_variance)

            centered_images = images - projected_mean
            ctf_squared_over_noise_variance = CTF**2

            PW = batch_over_vol_slice_volume(
                W,
                rotation_matrices,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                disc_type,
            )
            PW *= CTF[..., None, :]

            M_n = (jnp.conj(PW) @ PW.transpose(0, 2, 1)).real + jnp.eye(q)
            btilde = (jnp.conj(PW) @ centered_images[..., None]).real  # (b, q, 1)
            btilde = btilde.squeeze(-1)

            Sigma = jnp.linalg.pinv(M_n, hermitian=True)
            mu = (Sigma @ btilde[..., None]).squeeze(-1)

            u = (G @ mu[..., None]).squeeze(-1)
            suT = btilde[:, :, None] * u[:, None, :]
            H = G[None, :, :] + suT + jnp.swapaxes(suT, -1, -2)
            K = jnp.matmul(Sigma, jnp.matmul(H, Sigma))

            # term1: -2 * sum_n B_n W K_n
            before_backproj_bw = ctf_squared_over_noise_variance[..., None] * PW.transpose(0, 2, 1)
            weighted_bw = before_backproj_bw[..., :, None] * K[:, None, :, :]
            weighted_bw = weighted_bw.reshape(weighted_bw.shape[0], weighted_bw.shape[1], q * q)
            bw_backproj = batch_over_vol_adjoint_slice_volume(
                weighted_bw,
                rotation_matrices,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                disc_type,
            )
            bw_backproj = bw_backproj.reshape(experiment_dataset.volume_size, q, q)
            term1 = -2.0 * jnp.sum(bw_backproj, axis=1)

            # term2:  2 * sum_n b_n (Sigma u)^T
            alpha = (Sigma @ u[..., None]).squeeze(-1)
            before_backproj_bn = CTF[..., None] * centered_images[..., None] * alpha[:, None, :]
            bn_backproj = batch_over_vol_adjoint_slice_volume(
                before_backproj_bn,
                rotation_matrices,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                disc_type,
            )
            term2 = 2.0 * bn_backproj

            grad_sum = grad_sum + term1 + term2

    grad_F = grad_sum / float(n_total)

    return ll_sum, grad_neg_ll, F, grad_F, C_z


def whitening_penalty_and_grad_batched(W, B_n, b_n, batch_size=128):
    """
    Compute F(W) = 0.5 * ||Cz(W) - I||_F^2 and its gradient w.r.t. W using batches.

    This follows constraint_Gradient.tex (Eq. 16 in that file for the gradient template).

    Args:
        W: (d, q) loading matrix.
        B_n: (N, d, d) per-sample symmetric matrices.
        b_n: (N, d) per-sample vectors.
        batch_size: number of samples per batch.

    Returns:
        F: scalar whitening penalty value.
        grad_W: (d, q) gradient of F w.r.t. W.
        C_z: (q, q) empirical posterior covariance.
    """
    N = B_n.shape[0]
    if N == 0:
        raise ValueError("B_n must have at least one sample.")

    q = W.shape[1]
    I = jnp.eye(q, dtype=W.dtype)

    # -------------------------------------------------------------------------
    # First pass: compute Cz(W)
    # -------------------------------------------------------------------------
    C_z_sum = jnp.zeros((q, q), dtype=W.dtype)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = B_n[start:end]
        b = b_n[start:end]

        BW = jnp.einsum("ndd,dq->ndq", B, W)
        A = jnp.einsum("dq,ndq->nqq", W, BW) + I
        Sigma = jnp.linalg.inv(A)
        s = jnp.einsum("dq,nd->nq", W, b)
        mu = jnp.einsum("nqq,nq->nq", Sigma, s)
        C_z_sum = C_z_sum + jnp.sum(Sigma + jnp.einsum("ni,nj->nij", mu, mu), axis=0)

    C_z = C_z_sum / float(N)
    G = C_z - I
    F = 0.5 * jnp.sum(G * G)

    # -------------------------------------------------------------------------
    # Second pass: compute gradient
    # -------------------------------------------------------------------------
    grad_sum = jnp.zeros_like(W)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = B_n[start:end]
        b = b_n[start:end]

        BW = jnp.einsum("ndd,dq->ndq", B, W)
        A = jnp.einsum("dq,ndq->nqq", W, BW) + I
        Sigma = jnp.linalg.inv(A)
        s = jnp.einsum("dq,nd->nq", W, b)
        mu = jnp.einsum("nqq,nq->nq", Sigma, s)

        u = jnp.einsum("qq,nq->nq", G, mu)
        suT = s[:, :, None] * u[:, None, :]
        H = G[None, :, :] + suT + jnp.swapaxes(suT, -1, -2)
        K = jnp.matmul(Sigma, jnp.matmul(H, Sigma))

        term1 = -2.0 * jnp.einsum("ndq,nqk->ndk", BW, K)
        Sigma_u = jnp.einsum("nqq,nq->nq", Sigma, u)
        term2 = 2.0 * b[:, :, None] * Sigma_u[:, None, :]

        grad_sum = grad_sum + jnp.sum(term1 + term2, axis=0)

    grad_W = grad_sum / float(N)
    return F, grad_W, C_z


def apply_whitening_constraint(W, C_z, n_whitening_iters=10, tol=1e-8):
    """
    Apply the whitening constraint Ĉ_z = I to the loading matrix W.

    The key insight is that if we transform W → W @ C_z^{1/2}, then the
    new posterior statistics will have Ĉ_z closer to I.

    This is applied iteratively as a fixed-point iteration:
        W^{(k+1)} = W^{(k)} @ C_z(W^{(k)})^{1/2}

    Args:
        W: Loading matrix of shape (d, q)
        C_z: Current empirical posterior covariance of shape (q, q)
        n_whitening_iters: Maximum number of whitening iterations
        tol: Tolerance for convergence (||C_z - I||_F < tol)

    Returns:
        W_whitened: The whitened loading matrix
        converged: Whether the whitening converged
        final_deviation: The final ||C_z - I||_F
    """
    q = W.shape[1]
    I = jnp.eye(q)

    deviation = jnp.linalg.norm(C_z - I)
    if deviation < tol:
        return W, True, float(deviation)

    # Apply whitening: W → W @ C_z^{1/2}
    C_z_sqrt = sqrtm_psd(C_z)
    W_whitened = W @ C_z_sqrt

    return W_whitened, deviation < tol, float(deviation)


def whiten_W_iterative(W, second_moment_zs, n_iters=20, tol=1e-8, verbose=False):
    """
    Iteratively whiten W using the full posterior second moments.

    Note: This function only adjusts W based on the current C_z.
    For full correctness, the E-step should be re-run after each whitening
    to get updated second_moment_zs. However, in practice, applying the
    whitening once or a few times per EM iteration is sufficient.

    Args:
        W: Loading matrix of shape (d, q)
        second_moment_zs: Array of shape (N, q, q) with E[z z^T | y] for each sample
        n_iters: Maximum whitening iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        W: Whitened loading matrix
        C_z_final: Final C_z after whitening
        converged: Whether converged
    """
    q = W.shape[1]
    I = jnp.eye(q)

    # Compute initial C_z
    C_z = compute_Cz_from_second_moments(second_moment_zs)

    for i in range(n_iters):
        deviation = jnp.linalg.norm(C_z - I)

        if verbose:
            logger.info(f"  Whitening iter {i}: ||C_z - I|| = {deviation:.2e}")

        if deviation < tol:
            if verbose:
                logger.info(f"  Whitening converged at iter {i}")
            return W, C_z, True

        # Apply whitening transform
        C_z_sqrt = sqrtm_psd(C_z)
        W = W @ C_z_sqrt

        # Note: In a single EM step, we don't re-run E-step
        # The C_z would need to be recomputed with updated W
        # For now, we just apply the transformation once per iteration
        # and let subsequent EM iterations handle convergence

        # For more accurate whitening, we could estimate how C_z changes:
        # New C_z ≈ C_z_sqrt^{-1} @ C_z @ C_z_sqrt^{-1} = I (in the limit)
        # But this is approximate since we don't re-run E-step

        # Simple estimate: assume one iteration is enough
        if n_iters == 1:
            break

        # If doing multiple iterations, estimate new C_z
        # This is approximate - true update requires re-running E-step
        C_z_sqrt_inv = jnp.linalg.inv(C_z_sqrt)
        C_z = C_z_sqrt_inv @ C_z @ C_z_sqrt_inv  # Approximate new C_z

    return W, C_z, False


batch_over_vol_slice_volume = jax.vmap(core.slice_volume, in_axes=(1, None, None, None, None), out_axes=1)


def check_imaginary_part(x, image_shape, name, skip_ft=False):
    if not skip_ft:
        if len(image_shape) == 2:
            y = ftu.get_idft2(x.reshape(-1, *image_shape))
        else:
            y = ftu.get_idft3(x.reshape(-1, *image_shape))
    else:
        y = x
    imag_norm = np.linalg.norm(y.imag)
    ratio = np.inf if imag_norm == 0 else np.linalg.norm(y.real) / imag_norm
    print("imaginary part ratio", name, ratio)
    return ratio


batch_over_vol_adjoint_slice_volume = jax.vmap(
    core.adjoint_slice_volume, in_axes=(-1, None, None, None, None), out_axes=-1
)


@functools.partial(jax.jit, static_argnums=[8, 9, 13, 14, 15, 16, 17])
def E_M_step_batch(
    images,
    lhs_summed,
    rhs_summed,
    mean,
    W,
    CTF_params,
    rotation_matrices,
    translations,
    image_shape,
    volume_shape,
    grid_size,
    voxel_size,
    noise_variance,
    ctf_evaluator,
    compute_ll,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    compute_stats=True,
):
    basis_size = W.shape[1]
    volume_size = np.prod(volume_shape)

    # Precomp piece
    images = core.translate_images(images, translations, image_shape) / jnp.sqrt(noise_variance)
    # Just "whiten" the images and the projected mean, and include noise in CTF to simplify
    CTF = ctf_evaluator(CTF_params, image_shape, voxel_size) / jnp.sqrt(noise_variance)
    projected_mean = _forward_model_from_map(
        mean,
        CTF_params,
        rotation_matrices,
        image_shape,
        volume_shape,
        voxel_size,
        ctf_evaluator,
        disc_type_mean,
        skip_ctf=False,
    ) / jnp.sqrt(noise_variance)

    ctf_squared_over_noise_variance = CTF**2
    #
    PW = batch_over_vol_slice_volume(W, rotation_matrices, image_shape, volume_shape, disc_type)
    # n_images x n_basis_functions x image_size
    PW *= CTF[..., None, :]

    # P W .T @ P W
    M_n = (jnp.conj(PW) @ PW.transpose(0, 2, 1)).real + jnp.eye(basis_size)

    centered_images = images - projected_mean
    b_n = (jnp.conj(PW) @ centered_images[..., None]).real
    # check_imaginary_part(b_n, volume_shape, 'bn', skip_ft = True )

    M_n_inv = jax.numpy.linalg.pinv(M_n, hermitian=True)
    expected_zs = (M_n_inv @ b_n).squeeze(-1)
    # check_imaginary_part(expected_zs, volume_shape, '<z>', skip_ft = True )
    # check_imaginary_part(M_n_inv, volume_shape, 'Var(z)', skip_ft = True )
    # print('np.mean(expected_zs, axis=0), np.var(expected_zs, axis=0)', np.mean(expected_zs, axis=0), np.var(expected_zs, axis=0))

    second_moment_zs = M_n_inv + linalg.broadcast_outer(
        expected_zs, jnp.conj(expected_zs)
    )  # expected_zs[...,None] * jnp.conj(expected_zs)[...,None]

    if compute_stats:
        # Should be size n_images x image_size x basis_size x basis_size
        before_backproj_second_moments = (
            ctf_squared_over_noise_variance[..., None, None] * second_moment_zs[:, None, :, :]
        )
        before_backproj_first_moments = CTF[..., None] * centered_images[..., None] * jnp.conj(expected_zs)[:, None, :]

        # grid_point_vec_indices = core.batch_get_nearest_gridpoint_indices(rotation_matrices, image_shape, volume_shape )
        lhs_summed += batch_over_vol_adjoint_slice_volume(
            before_backproj_second_moments.reshape(*before_backproj_second_moments.shape[:-2], -1),
            rotation_matrices,
            image_shape,
            volume_shape,
            disc_type,
        )
        rhs_summed += batch_over_vol_adjoint_slice_volume(
            before_backproj_first_moments, rotation_matrices, image_shape, volume_shape, disc_type
        )
    # lhs_summed = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, before_backproj_second_moments.reshape(*before_backproj_second_moments.shape[:-2], -1), grid_point_vec_indices, lhs_summed)
    # rhs_summed = core.batch_over_vol_summed_adjoint_slice_by_nearest(volume_size, before_backproj_first_moments, grid_point_vec_indices, rhs_summed)

    # return lhs_summed, rhs_summed, expected_zs, second_moment_zs
    # --- Optional log-likelihood (observed-data) ---
    #   ell_n = -0.5 * [ d_n log(2π) + ||r||^2 - u^T M^{-1} u + logdet M ]
    if compute_ll:
        # u = b_n.squeeze(-1)
        u = b_n.squeeze(-1)  # (b, q)

        # quadratic term via M_n_inv (already computed):
        quad = jnp.real(jnp.sum(jnp.conj(u) * (M_n_inv @ u[..., None]).squeeze(-1), axis=-1))  # (b,)

        # ||r||^2
        r2 = jnp.real(jnp.sum(jnp.conj(centered_images) * centered_images, axis=-1))  # (b,)

        # logdet M via Cholesky (more stable than slogdet on Hermitian PD)
        L = jnp.linalg.cholesky(M_n)  # (b, q, q)
        logdetM = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diagonal(L, axis1=1, axis2=2))), axis=-1)  # (b,)

        d_n = images.shape[-1]  # image dimensionality (pixels)
        const = d_n * jnp.log(2.0 * jnp.pi)
        ll_per_image = -0.5 * (const + r2 - quad + logdetM)  # (b,)
        ll_sum = jnp.sum(ll_per_image)
    else:
        ll_sum = jnp.array(0.0, dtype=images.dtype)
        ll_per_image = jnp.zeros((0,), dtype=images.dtype)

    return lhs_summed, rhs_summed, expected_zs, second_moment_zs, ll_sum, ll_per_image


# =============================================================================
# HALF-SPECTRUM E-M STEP (memory-optimized)
# =============================================================================


def _tri_size(basis_size):
    """Number of upper-triangular entries (including diagonal)."""
    return basis_size * (basis_size + 1) // 2


_half_slice_volume = functools.partial(core.slice_volume, half_volume=True, half_image=True)
batch_over_vol_slice_volume_half = jax.vmap(_half_slice_volume, in_axes=(1, None, None, None, None), out_axes=1)

_half_adjoint_slice_volume = functools.partial(core.adjoint_slice_volume, half_image=True, half_volume=True)
batch_over_vol_adjoint_slice_volume_half = jax.vmap(
    _half_adjoint_slice_volume, in_axes=(-1, None, None, None, None), out_axes=-1
)


@functools.partial(jax.jit, static_argnums=[8, 9, 10, 11, 12, 13])
def _e_step_half_inner(
    images_half,
    mean,
    W_half,
    CTF_params,
    rotation_matrices,
    translations,
    voxel_size,
    noise_variance_half,
    image_shape,
    volume_shape,
    ctf_evaluator,
    compute_ll,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    # NOTE: JIT boundary — backprojection is done separately to
    # allow XLA to free memory between chunks.
):
    """JIT'd E-step core: computes posterior moments and log-likelihood.

    Returns everything needed for the backprojection (done outside JIT).
    """
    basis_size = W_half.shape[1]

    w_1d = linalg.half_spectrum_last_axis_weights(image_shape[1])
    rfft_w = jnp.tile(w_1d, (image_shape[0], 1)).reshape(-1)

    images_half = core.translate_images(images_half, translations, image_shape, half_image=True) / jnp.sqrt(
        noise_variance_half
    )
    CTF_half = ctf_evaluator(CTF_params, image_shape, voxel_size, half_image=True) / jnp.sqrt(noise_variance_half)

    projected_mean_half = core.slice_volume(
        mean, rotation_matrices, image_shape, volume_shape, disc_type_mean, half_image=True
    )
    projected_mean_half = (
        projected_mean_half
        * ctf_evaluator(CTF_params, image_shape, voxel_size, half_image=True)
        / jnp.sqrt(noise_variance_half)
    )

    ctf_squared_half = CTF_half**2

    PW_half = batch_over_vol_slice_volume_half(W_half, rotation_matrices, image_shape, volume_shape, disc_type)
    PW_half *= CTF_half[:, None, :]

    PW_w = jnp.conj(PW_half) * rfft_w[None, None, :]
    M_n = (PW_w @ PW_half.transpose(0, 2, 1)).real + jnp.eye(basis_size)

    centered_half = images_half - projected_mean_half
    b_n = (PW_w @ centered_half[..., None]).real

    M_n_inv = jax.numpy.linalg.pinv(M_n, hermitian=True)
    expected_zs = (M_n_inv @ b_n).squeeze(-1)
    second_moment_zs = M_n_inv + linalg.broadcast_outer(expected_zs, jnp.conj(expected_zs))

    ll_sum = jnp.array(0.0, dtype=images_half.dtype)
    if compute_ll:
        u = b_n.squeeze(-1)
        quad = jnp.real(jnp.sum(jnp.conj(u) * (M_n_inv @ u[..., None]).squeeze(-1), axis=-1))
        r2 = jnp.sum(rfft_w * jnp.real(jnp.conj(centered_half) * centered_half), axis=-1)
        L = jnp.linalg.cholesky(M_n)
        logdetM = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diagonal(L, axis1=1, axis2=2))), axis=-1)
        d_n = np.prod(image_shape)
        ll_per_image = -0.5 * (d_n * jnp.log(2.0 * jnp.pi) + r2 - quad + logdetM)
        ll_sum = jnp.sum(ll_per_image)

    return expected_zs, second_moment_zs, ctf_squared_half, centered_half, CTF_half, ll_sum


def E_M_step_batch_half(
    images_half,
    lhs_summed,
    rhs_summed,
    mean,
    W_half,
    CTF_params,
    rotation_matrices,
    translations,
    image_shape,
    volume_shape,
    grid_size,
    voxel_size,
    noise_variance_half,
    ctf_evaluator,
    compute_ll,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    compute_stats=True,
):
    """Half-spectrum, upper-triangular-LHS variant of :func:`E_M_step_batch`.

    E-step is JIT'd. LHS backprojection uses the fused CUDA kernel
    (all tri_sz channels in one call — no chunking needed since the
    fused kernel doesn't materialize the before_chunk tensor).
    """
    basis_size = W_half.shape[1]
    tri_i, tri_j = np.triu_indices(basis_size)
    tri_sz = len(tri_i)

    # --- JIT'd E-step ---
    expected_zs, second_moment_zs, ctf_squared_half, centered_half, CTF_half, ll_sum = _e_step_half_inner(
        images_half,
        mean,
        W_half,
        CTF_params,
        rotation_matrices,
        translations,
        voxel_size,
        noise_variance_half,
        image_shape,
        volume_shape,
        ctf_evaluator,
        compute_ll,
        disc_type_mean,
        disc_type,
    )

    # --- backprojection ---
    if compute_stats:
        from recovar.cuda_backproject import fused_backproject

        half_volume_size = lhs_summed.shape[0]
        second_moment_tri = second_moment_zs[:, tri_i, tri_j]
        ctf_squared_full = ftu.half_image_to_full_image(ctf_squared_half, image_shape)
        _max_r = image_shape[0] // 2 - 1

        # LHS: fused kernel — all tri_sz channels in one call.
        # Reads ctf² (50 MB) and smz_tri (168 KB) separately, no intermediate.
        lhs_bp = fused_backproject(
            jnp.zeros((half_volume_size, tri_sz), dtype=jnp.float32),
            ctf_squared_full.astype(jnp.float32),
            second_moment_tri.astype(jnp.float32),
            jnp.asarray(rotation_matrices),
            image_shape,
            volume_shape,
            max_r=_max_r,
        )
        lhs_summed = lhs_summed + lhs_bp

        # RHS: half-image backprojection (30% fewer scatters for complex).
        before_rhs = (CTF_half[..., None] * centered_half[..., None] * jnp.conj(expected_zs)[:, None, :]).transpose(
            2, 0, 1
        )
        bp_rhs = core.batch_adjoint_slice_volume(
            before_rhs,
            rotation_matrices,
            image_shape,
            volume_shape,
            disc_type,
            half_image=True,
            half_volume=True,
        )
        rhs_summed = rhs_summed + bp_rhs.T

    ll_per_image = jnp.zeros((0,), dtype=images_half.dtype)
    return lhs_summed, rhs_summed, expected_zs, second_moment_zs, ll_sum, ll_per_image


def unpack_tri_to_full(lhs_tri, basis_size):
    """Unpack upper-triangular ``(…, tri_size)`` to symmetric ``(…, q, q)``.

    Useful for converting the output of :func:`E_M_step_batch_half` back to the
    full matrix format expected by downstream solvers.
    """
    tri_i, tri_j = np.triu_indices(basis_size)
    shape = lhs_tri.shape[:-1] + (basis_size, basis_size)
    out = jnp.zeros(shape, dtype=lhs_tri.dtype)
    out = out.at[..., tri_i, tri_j].set(lhs_tri)
    out = out.at[..., tri_j, tri_i].set(lhs_tri)
    return out


batch1_symmetrize_ft_volume = jax.vmap(utils.symmetrize_ft_volume, in_axes=(1, None), out_axes=1)


def _iter_processed_batches_half(experiment_dataset, batch_size):
    """Like _iter_processed_batches but yields half-spectrum images and noise."""
    for (
        batch,
        rotation_matrices,
        translations,
        ctf_params,
        _noise_variance,
        _particle_indices,
        image_indices,
    ) in experiment_dataset.iter_batches(
        batch_size,
        by_image=not getattr(experiment_dataset, "tilt_series_flag", False),
    ):
        yield (
            experiment_dataset.process_images_half(batch, apply_image_mask=False),
            ctf_params,
            rotation_matrices,
            translations,
            image_indices,
        )


def EM_step_half(
    experiment_datasets,
    mean_estimate,
    W_estimate,
    batch_size,
    W_prior,
    use_whitening=False,
    whitening_mode="cz",
    disc_type_mean="cubic",
    disc_type="linear_interp",
    recompute_ll=False,
    mean_estimate_raw=None,
):
    """Half-spectrum EM step for L2-regularized PPCA.

    Same interface as :func:`EM_step` but uses :func:`E_M_step_batch_half`
    internally.  All accumulation happens in half-volume / upper-triangular
    format, and the M-step solve runs on half_volume_size voxels only.

    Memory savings vs ``EM_step``:
    * ``lhs``: ``volume_size × q²`` → ``half_vol × q(q+1)/2``  (~4× less)
    * ``rhs``: ``volume_size × q``  → ``half_vol × q``          (~2× less)
    * ``W``: kept in half-volume during E-step, expanded only for whitening
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_datasets)
    ref = full_dataset if full_dataset is not None else dataset_list[0]
    basis_size = W_estimate.shape[-1]
    volume_shape = ref.volume_shape
    half_volume_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_volume_size = int(np.prod(half_volume_shape))
    tri_sz = _tri_size(basis_size)

    # Convert W to half-volume for the E-step
    # W: (volume_size, basis_size) → transpose, convert, transpose back
    W_half = ftu.full_volume_to_half_volume(W_estimate.T, volume_shape).T

    # Half-volume accumulators
    lhs_summed = jnp.zeros((half_volume_size, tri_sz), dtype=ref.dtype_real)
    rhs_summed = jnp.zeros((half_volume_size, basis_size), dtype=ref.dtype)

    ll_sum = jnp.array(0.0, dtype=ref.dtype)
    expected_zs = []
    second_moment_zs = []

    for experiment_dataset in dataset_list:
        for batch_half, ctf_params, rotation_matrices, translations, batch_image_ind in _iter_processed_batches_half(
            experiment_dataset, batch_size
        ):
            noise_variance_half = experiment_dataset.noise.get_half(batch_image_ind)
            lhs_summed, rhs_summed, ez_batch, smz_batch, ll_batch, _ = E_M_step_batch_half(
                batch_half,
                lhs_summed,
                rhs_summed,
                mean_estimate,
                W_half,
                ctf_params,
                rotation_matrices,
                translations,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                experiment_dataset.grid_size,
                experiment_dataset.voxel_size,
                noise_variance_half,
                experiment_dataset.ctf_evaluator,
                compute_ll=True,
                disc_type_mean=disc_type_mean,
                disc_type=disc_type,
                compute_stats=True,
            )
            expected_zs.append(np.array(ez_batch))
            second_moment_zs.append(np.array(smz_batch))
            ll_sum += ll_batch

    expected_zs = np.concatenate(expected_zs, axis=0)
    second_moment_zs = np.concatenate(second_moment_zs, axis=0)
    expected_zs_mean = np.mean(expected_zs, axis=0)
    expected_zs_var = np.var(expected_zs, axis=0)

    # ------------------------------------------------------------------
    # M-step solve in half-volume space (chunked to avoid OOM)
    # ------------------------------------------------------------------
    W_prior_half = ftu.full_volume_to_half_volume(W_prior.T, volume_shape).T
    reg_diag = 1 / (W_prior_half + 1e-16)  # (half_vol, q)

    # Chunk the solve: unpack tri → full + regularize + solve, one chunk at a time.
    # Full matrix (half_vol, q, q) would be ~13 GB at 256³ — too much.
    # Chunks of 200K voxels use ~300 MB each.
    _SOLVE_CHUNK = 200_000
    W_half_parts = []
    for i0 in range(0, half_volume_size, _SOLVE_CHUNK):
        i1 = min(i0 + _SOLVE_CHUNK, half_volume_size)
        lhs_chunk = unpack_tri_to_full(lhs_summed[i0:i1], basis_size)
        lhs_chunk = lhs_chunk + jax.vmap(jnp.diag)(reg_diag[i0:i1])
        W_half_parts.append(jnp.linalg.solve(lhs_chunk, rhs_summed[i0:i1, :, None])[..., 0])
    W_half = jnp.concatenate(W_half_parts, axis=0)

    # Expand to full volume for whitening and output
    W = ftu.half_volume_to_full_volume(W_half.T, volume_shape).T

    if jnp.any(jnp.isnan(W)):
        logger.error("EM_step_half: NaN in W after M-step")
        raise ValueError("NaN in W after M-step")

    # ------------------------------------------------------------------
    # Whitening
    # ------------------------------------------------------------------
    if use_whitening:
        if whitening_mode == "proj_ls":
            if mean_estimate_raw is None:
                raise ValueError("proj_ls whitening requires mean_estimate_raw")
            volume_mask = getattr(ref, "volume_mask", None)
            if volume_mask is None:
                volume_mask = np.ones(ref.volume_shape)
            Sigma = compute_sigma_proj_ls(
                experiment_datasets,
                mean_estimate_raw,
                W,
                volume_mask,
                batch_size,
                disc_type_mean=disc_type_mean,
                disc_type=disc_type,
                do_mask_images=True,
                parallel_analysis=False,
            )
            if jnp.any(jnp.isnan(Sigma)) or jnp.any(jnp.isinf(Sigma)):
                logger.warning("EM_step_half: ill-conditioned Sigma, skipping whitening")
                Sigma = jnp.eye(Sigma.shape[0], dtype=Sigma.dtype)
            W = W @ sqrtm_psd(Sigma)
        else:
            C_z = compute_Cz_from_second_moments(second_moment_zs)
            q = W.shape[1]
            logger.info(f"  Before whitening: ||Ĉ_z - I|| = {float(jnp.linalg.norm(C_z - jnp.eye(q))):.4f}")
            W, C_z_final, _ = whiten_W_iterative(W, second_moment_zs, n_iters=1, tol=1e-8, verbose=False)
            logger.info(f"  After whitening: ||Ĉ_z - I|| ≈ {float(jnp.linalg.norm(C_z_final - jnp.eye(q))):.4f}")

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------
    ll_prior = jnp.linalg.norm(W / jnp.sqrt(W_prior + 1e-16)) ** 2
    neg_ll_total = float(-ll_sum.real + ll_prior.real)
    neg_ll_data = float(-ll_sum.real)
    neg_ll_prior = float(ll_prior.real)

    if jnp.isnan(W).any():
        logger.error("EM_step_half produced NaN in W")
        raise ValueError("EM_step_half produced NaN in W")

    return W, expected_zs, second_moment_zs, expected_zs_mean, expected_zs_var, neg_ll_total, neg_ll_data, neg_ll_prior


# @functools.partial(jax.jit, static_argnums = [5])
def EM_step(
    experiment_datasets,
    mean_estimate,
    W_estimate,
    batch_size,
    W_prior,
    sparse_PCA=False,
    use_whitening=False,
    whitening_mode="cz",
    l1_sigma=None,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    recompute_ll=False,
    mean_estimate_raw=None,
):
    """
    Perform one EM step for PPCA.

    Args:
        experiment_datasets: List of cryo-EM datasets
        mean_estimate: Mean volume estimate (or precomputed spline coefficients if disc_type_mean='cubic')
        W_estimate: Current loading matrix estimate
        batch_size: Batch size for processing
        W_prior: Prior on W (regularization)
        sparse_PCA: Whether to use sparse PCA with wavelet L1 regularization
        use_whitening: Whether to apply the whitening constraint.
        whitening_mode: "cz" (existing) or "proj_ls" (projected covariance LS).
                       This fixes the scale ambiguity problem in regularized PPCA.
        l1_sigma: Pre-computed L1 regularization sigma (computed once at EM start).
                  If None and sparse_PCA=True, will raise an error.
        disc_type_mean: Interpolation type for mean projection ('cubic' or 'linear_interp')
        disc_type: Interpolation type for W projection ('nearest', 'linear_interp', etc.)
        recompute_ll: If True, recompute data log-likelihood using updated W.

    Returns:
        W: Updated loading matrix
        expected_zs: Posterior means E[z|y]
        second_moment_zs: Posterior second moments E[zz^T|y]
        expected_zs_mean: Mean of posterior means across samples
        expected_zs_var: Variance of posterior means across samples
        neg_ll_total: Negative total log-likelihood (data + prior)
        neg_ll_data: Negative data log-likelihood
        neg_ll_prior: Negative prior log-likelihood
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_datasets)
    reference_dataset = full_dataset if full_dataset is not None else dataset_list[0]
    basis_size = W_estimate.shape[-1]
    rhs_summed = jnp.zeros((reference_dataset.volume_size, basis_size), dtype=reference_dataset.dtype)
    lhs_summed = jnp.zeros((reference_dataset.volume_size, basis_size * basis_size), dtype=reference_dataset.dtype_real)

    ll_sum = jnp.array(0.0, dtype=reference_dataset.dtype)
    expected_zs = []
    second_moment_zs = []
    for experiment_dataset in dataset_list:
        for batch, ctf_params, rotation_matrices, translations, batch_image_ind in _iter_processed_batches(
            experiment_dataset, batch_size
        ):
            noise_variance = experiment_dataset.noise.get(batch_image_ind)
            lhs_summed, rhs_summed, expected_zs_batch, second_moment_zs_batch, ll_sum_batch, _ = E_M_step_batch(
                batch,
                lhs_summed,
                rhs_summed,
                mean_estimate,
                W_estimate,
                ctf_params,
                rotation_matrices,
                translations,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                experiment_dataset.grid_size,
                experiment_dataset.voxel_size,
                noise_variance,
                experiment_dataset.ctf_evaluator,
                compute_ll=True,
                disc_type_mean=disc_type_mean,
                disc_type=disc_type,
                compute_stats=True,
            )
            expected_zs.append(np.array(expected_zs_batch))
            second_moment_zs.append(np.array(second_moment_zs_batch))
            ll_sum += ll_sum_batch

    expected_zs = np.concatenate(expected_zs, axis=0)
    second_moment_zs = np.concatenate(second_moment_zs, axis=0)

    # Calculate statistics for reporting
    expected_zs_mean = np.mean(expected_zs, axis=0)
    expected_zs_var = np.var(expected_zs, axis=0)

    # Solve least squares
    # V = jax.vmap(jnp.diag)(1 / (W_prior + 1e-16 ))

    if sparse_PCA:
        if l1_sigma is None:
            raise ValueError("sparse_PCA=True requires l1_sigma to be pre-computed and passed to EM_step")

        volume_size = reference_dataset.volume_size
        volume_shape = reference_dataset.volume_shape
        normal_size = W_estimate.shape

        lhs_summed = batch1_symmetrize_ft_volume(lhs_summed, volume_shape)
        lhs_summed = lhs_summed.reshape(experiment_dataset.volume_size, basis_size, basis_size)

        rhs_summed = batch1_symmetrize_ft_volume(rhs_summed, volume_shape)
        W_estimate = batch1_symmetrize_ft_volume(W_estimate, volume_shape)

        from recovar.ppca.admm_test import admm_wavelet

        # Use pre-computed sigma for ADMM (fixed across EM iterations)
        W, Z_rec = admm_wavelet(lhs_summed, rhs_summed, l1_sigma, 0.9, 50, volume_shape, normal_size, W_estimate)

    else:
        lhs_summed = lhs_summed.reshape(reference_dataset.volume_size, basis_size, basis_size)

        lhs_summed = lhs_summed + jax.vmap(jnp.diag)(1 / (W_prior + 1e-16))

        # W = linalg.batch_hermitian_linear_solver(lhs_summed, rhs_summed)
        W = linalg.batch_linear_solver(lhs_summed, rhs_summed[..., None])[..., 0]

    # NaN diagnostic: W after M-step (before whitening)
    if jnp.any(jnp.isnan(W)):
        logger.error("EM_step: NaN in W immediately after M-step (batch_linear_solver or ADMM produced NaN)")
        raise ValueError("NaN in W after M-step")

    # =============================================================================
    # WHITENING CONSTRAINT: Apply Ĉ_z = I constraint to fix scale ambiguity
    # =============================================================================
    if use_whitening:
        if whitening_mode == "proj_ls":
            if mean_estimate_raw is None:
                raise ValueError("proj_ls whitening requires mean_estimate_raw (pre-spline).")
            # NaN diagnostic: W before proj_ls
            if jnp.any(jnp.isnan(W)):
                logger.error("EM_step: NaN in W before proj_ls whitening (M-step produced NaN)")
                raise ValueError("NaN in W before proj_ls whitening (M-step produced NaN)")
            volume_mask = getattr(reference_dataset, "volume_mask", None)
            if volume_mask is None:
                volume_mask = np.ones(reference_dataset.volume_shape)
            Sigma = compute_sigma_proj_ls(
                experiment_datasets,
                mean_estimate_raw,
                W,
                volume_mask,
                batch_size,
                disc_type_mean=disc_type_mean,
                disc_type=disc_type,
                do_mask_images=True,
                parallel_analysis=False,
            )
            # NaN/Inf fallback: projected covariance solve can be ill-conditioned (e.g. first iter, small W)
            if jnp.any(jnp.isnan(Sigma)) or jnp.any(jnp.isinf(Sigma)):
                logger.warning(
                    "EM_step: NaN/Inf in Sigma from compute_sigma_proj_ls (ill-conditioned solve); "
                    "skipping proj_ls whitening this step (using Sigma=I)."
                )
                Sigma = jnp.eye(Sigma.shape[0], dtype=Sigma.dtype)
            constraint_violation = float(jnp.linalg.norm(Sigma - jnp.eye(Sigma.shape[0])))
            logger.info(f"  Before whitening (proj_ls): ||Sigma(W)-I|| = {constraint_violation:.4f}")
            sqrt_Sigma = sqrtm_psd(Sigma)
            W = W @ sqrt_Sigma
        else:
            # Compute the empirical posterior covariance
            C_z = compute_Cz_from_second_moments(second_moment_zs)

            # Log the constraint violation before whitening
            q = W.shape[1]
            constraint_violation = float(jnp.linalg.norm(C_z - jnp.eye(q)))
            trace_Cz = float(jnp.trace(C_z))
            logger.info(f"  Before whitening: ||Ĉ_z - I|| = {constraint_violation:.4f}, tr(Ĉ_z) = {trace_Cz:.4f}")

            # Apply whitening: W → W @ C_z^{1/2}
            W, C_z_final, converged = whiten_W_iterative(W, second_moment_zs, n_iters=1, tol=1e-8, verbose=False)

            # Log after whitening (approximate since we didn't re-run E-step)
            constraint_violation_after = float(jnp.linalg.norm(C_z_final - jnp.eye(q)))
            logger.info(f"  After whitening: ||Ĉ_z - I|| ≈ {constraint_violation_after:.4f}")

    # Recompute data log-likelihood for the updated W if requested
    if recompute_ll:
        ll_sum_post = jnp.array(0.0, dtype=reference_dataset.dtype)
        for experiment_dataset in dataset_list:
            for batch, ctf_params, rotation_matrices, translations, batch_image_ind in _iter_processed_batches(
                experiment_dataset, batch_size
            ):
                noise_variance = experiment_dataset.noise.get(batch_image_ind)
                _, _, _, _, ll_sum_batch, _ = E_M_step_batch(
                    batch,
                    lhs_summed,
                    rhs_summed,
                    mean_estimate,
                    W,
                    ctf_params,
                    rotation_matrices,
                    translations,
                    experiment_dataset.image_shape,
                    experiment_dataset.volume_shape,
                    experiment_dataset.grid_size,
                    experiment_dataset.voxel_size,
                    noise_variance,
                    experiment_dataset.ctf_evaluator,
                    compute_ll=True,
                    disc_type_mean=disc_type_mean,
                    disc_type=disc_type,
                    compute_stats=False,
                )
                ll_sum_post += ll_sum_batch
        ll_sum = ll_sum_post

    # Calculate log-likelihood statistics using updated W
    if sparse_PCA:
        from recovar.ppca.admm_test import WaveletL1

        ll_prior = WaveletL1(W.shape, reference_dataset.volume_shape, "db1", sigma=l1_sigma)(W)
    else:
        ll_prior = jnp.linalg.norm(W / jnp.sqrt(W_prior + 1e-16)) ** 2

    neg_ll_total = float(-ll_sum.real + ll_prior.real)
    neg_ll_data = float(-ll_sum.real)
    neg_ll_prior = float(ll_prior.real)

    if jnp.isnan(W).any():
        logger.error("EM_step produced NaN in W")
        raise ValueError("EM_step produced NaN in W")
    return W, expected_zs, second_moment_zs, expected_zs_mean, expected_zs_var, neg_ll_total, neg_ll_data, neg_ll_prior


def batch_vec(x):
    return x.swapaxes(-1, -2).reshape(-1, x.shape[-1] ** 2)


def batch_unvec(x):
    n = np.sqrt(x.shape[-1]).astype(int)
    return x.reshape(-1, n, n).swapaxes(-1, -2)


def EM(
    experiment_dataset,
    mean_estimate,
    W_initial,
    W_prior,
    EM_iter=20,
    sparse_PCA=False,
    U_gt=None,
    S_gt=None,
    make_plots=False,
    use_whitening=False,
    whitening_mode="cz",
    l1_sigma=None,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    return_iteration_data=False,
    recompute_ll=False,
):
    """
    Run EM algorithm for PPCA.

    Args:
        experiment_dataset: List of cryo-EM datasets
        mean_estimate: Mean volume estimate
        W_initial: Initial loading matrix
        W_prior: Prior variance for L2 regularization. Larger values → less regularization.
                 NOTE: Only used when sparse_PCA=False (L2/ridge regression).
                 For L1, this parameter is IGNORED - use l1_sigma instead.
        EM_iter: Number of EM iterations
        sparse_PCA: If False, use L2 (ridge) with W_prior as variance.
                    If True, use L1 (wavelet sparsity) with l1_sigma as threshold.
        U_gt: Ground truth principal components (optional, for evaluation)
        S_gt: Ground truth singular values (optional, for evaluation)
        make_plots: Whether to make plots at each iteration
        use_whitening: Whether to apply the whitening constraint.
        whitening_mode: "cz" (existing) or "proj_ls" (projected covariance LS).
                       RECOMMENDED: Set to True when using regularization.
        l1_sigma: L1 soft-threshold level (REQUIRED when sparse_PCA=True).
                  Larger values → more sparsity. Can be scalar or per-coefficient array.
                  Typical range: 0.01 - 1.0 depending on data scale.
        disc_type_mean: Interpolation type for mean projection ('cubic' or 'linear_interp').
                        'cubic' requires precomputing spline coefficients (done automatically).
        return_iteration_data: If True, return per-iteration diagnostics.
        recompute_ll: If True, recompute data log-likelihood using updated W each iter.

    Regularization summary:
        L2 (sparse_PCA=False): min ||Y - XW||² + ||W||²/W_prior
        L1 (sparse_PCA=True):  min ||Y - XW||² + l1_sigma * ||wavelet(W)||₁

    Returns:
        U: Principal components
        S: Singular values squared
        W: Final loading matrix
        expected_zs: Final posterior means
        second_moment_zs: Final posterior second moments
        iteration_data (optional): List of per-iteration diagnostics
    """
    # Initialize
    # import jax.random as jr
    # matrix_key, vector_key = jr.split(jr.PRNGKey(0))
    # W = jr.normal(matrix_key, (experiment_dataset.volume_size, basis_size), dtype = experiment_dataset.dtype_real)
    # W = linalg.batch_dft3(W, experiment_dataset.volume_shape, basis_size)
    # eigenvalue = np.ones(basis_size)
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_dataset)
    reference_dataset = full_dataset if full_dataset is not None else dataset_list[0]
    volume_mask = np.ones(reference_dataset.volume_shape)
    basis_size = W_initial.shape[-1]
    contrast_grid = np.ones([1])
    # Larger batches amortize per-batch overhead (kernel launches, backprojection).
    # 200 is optimal for 256³/20 PCs on 80 GB GPU (~37 GB peak, 4.6 ms/img).
    # Smaller grids use less memory so the same batch size is safe.
    batch_size = 200
    W = W_initial

    # Precompute spline coefficients for cubic interpolation
    mean_estimate_raw = mean_estimate
    if disc_type_mean == "cubic":
        mean_estimate = core.precompute_cubic_coefficients(
            mean_estimate,
            reference_dataset.volume_shape,
        )

    # =============================================================================
    # L1 REGULARIZATION WEIGHT
    #
    # NOTE: W_prior means different things for L1 vs L2:
    #   - L2 (sparse_PCA=False): W_prior = prior VARIANCE. Larger → less regularization.
    #   - L1 (sparse_PCA=True):  l1_sigma = soft-threshold level. Larger → more sparsity.
    #
    # For L1, pass l1_sigma directly (scalar or array). W_prior is ignored for L1.
    # This allows independent tuning of L1 and L2 hyperparameters.
    # =============================================================================
    if sparse_PCA:
        if l1_sigma is None:
            # Use median of W_prior as scalar l1_sigma (old code passed W_prior but WaveletL1 needs scalar)
            l1_sigma = float(np.median(W_prior[W_prior > 0]))
            print(f"L1 regularization: using median(W_prior)={l1_sigma:.6f} as sigma")
        if np.isscalar(l1_sigma):
            print(f"L1 regularization: sigma={l1_sigma:.6f} (uniform)")
        else:
            print(f"L1 regularization: sigma array, range=[{np.min(l1_sigma):.6f}, {np.max(l1_sigma):.6f}]")

    # Initialize table for collecting iteration data
    iteration_data = []

    # Print table header
    print("\n" + "=" * 130)
    mode_str = "WITH WHITENING" if use_whitening else "WITHOUT WHITENING"
    print(f"EM ALGORITHM CONVERGENCE TABLE ({mode_str})")
    print("=" * 130)
    header = f"{'Iter':>4} | {'Neg_LL_Total':>12} | {'Neg_LL_Data':>12} | {'Neg_LL_Prior':>12} | {'Exp_ZS_Mean':>12} | {'Exp_ZS_Var':>12} | {'Rel_Var_Expl':>12}"
    if U_gt is not None:
        header += f" | {'Top_5_Rel_Var':>20}"
    if use_whitening:
        header += f" | {'||W||_F':>10}"
    print(header)
    print("-" * len(header))

    for iter_i in range(EM_iter):
        if not sparse_PCA:
            # L2: use half-spectrum EM step (4× less memory, ~2× faster M-step)
            (
                W,
                expected_zs,
                second_moment_zs,
                expected_zs_mean,
                expected_zs_var,
                neg_ll_total,
                neg_ll_data,
                neg_ll_prior,
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
            )
        else:
            # L1/sparse: use full-spectrum EM step (ADMM needs full volume)
            (
                W,
                expected_zs,
                second_moment_zs,
                expected_zs_mean,
                expected_zs_var,
                neg_ll_total,
                neg_ll_data,
                neg_ll_prior,
            ) = EM_step(
                experiment_dataset,
                mean_estimate,
                W,
                batch_size,
                W_prior,
                sparse_PCA,
                use_whitening=use_whitening,
                whitening_mode=whitening_mode,
                l1_sigma=l1_sigma,
                disc_type_mean=disc_type_mean,
                disc_type=disc_type,
                recompute_ll=recompute_ll,
                mean_estimate_raw=mean_estimate_raw,
            )

        # Make real
        W = W.T.reshape(basis_size, *reference_dataset.volume_shape)
        W = ftu.get_idft3(W).real
        W = W.reshape(W.shape[0], -1).T

        W = W.T
        W = ftu.get_dft3(W.reshape(W.shape[0], *reference_dataset.volume_shape))
        W = W.reshape(W.shape[0], -1).T

        # plt.figure()
        # plt.imshow(experiment_dataset[0].get_proj(W[:,0].reshape(-1)))

        logger.info(f"Done with EM step {iter_i}")

        # Collect iteration data
        C_z = compute_Cz_from_second_moments(second_moment_zs)
        # E[μ μ^T] = (1/N) Σ_n μ_n μ_n^T from posterior means (expected_zs)
        N_z = expected_zs.shape[0]
        E_mean_outer = (
            (expected_zs.T @ np.conj(expected_zs)).real / N_z
            if expected_zs.dtype in (np.complex64, np.complex128)
            else (expected_zs.T @ expected_zs) / N_z
        )
        E_mean_outer = np.asarray(E_mean_outer)
        q = W.shape[1]
        I_q = jnp.eye(q, dtype=W.dtype)
        trace_Cz = float(jnp.trace(C_z))
        constraint_violation = float(jnp.linalg.norm(C_z - I_q))
        trace_E_mean_outer = float(np.trace(E_mean_outer))
        norm_E_mean_outer_minus_I = float(np.linalg.norm(E_mean_outer - np.eye(q)))
        W_norm = float(jnp.linalg.norm(W))
        iter_info = {
            "Iteration": iter_i,
            "Neg_LL_Total": float(neg_ll_total),
            "Neg_LL_Data": float(neg_ll_data),
            "Neg_LL_Prior": float(neg_ll_prior),
            "Expected_ZS_Mean": float(np.mean(expected_zs_mean)),
            "Expected_ZS_Var": float(np.mean(expected_zs_var)),
            "W_norm": W_norm,
            "trace_Cz": trace_Cz,
            "constraint_violation": constraint_violation,
            "trace_E_mean_outer": trace_E_mean_outer,
            "norm_E_mean_outer_minus_I": norm_E_mean_outer_minus_I,
        }

        if U_gt is not None:
            U, S, _ = jnp.linalg.svd(W, full_matrices=False)
            variance, rel_var, norm_var = metrics.get_all_variance_scores(U, U_gt, S_gt)
            iter_info["Rel_Var_Explained"] = float(rel_var[-1])
            iter_info["Top_5_Rel_Var"] = rel_var[:5]
        else:
            iter_info["Rel_Var_Explained"] = None
            iter_info["Top_5_Rel_Var"] = None

        iteration_data.append(iter_info)

        # Print current iteration row
        row = f"{iter_i:>4} | {neg_ll_total:12.6e} | {neg_ll_data:12.6e} | {neg_ll_prior:12.6e} | {np.mean(expected_zs_mean):12.6e} | {np.mean(expected_zs_var):12.6e}"
        if U_gt is not None:
            row += f" | {rel_var[-1]:12.6e} | {str(rel_var[: min(5, len(rel_var))]):>20}"
        else:
            row += f" | {'N/A':>12}"
        if use_whitening:
            W_norm = float(jnp.linalg.norm(W))
            row += f" | {W_norm:10.4f}"
        print(row)

        if (make_plots or iter_i == EM_iter - 1) and U_gt is not None:
            max_size_this = np.min([20, U.shape[-1]])
            plt.figure()
            plt.plot(rel_var)
            plt.title("relative variance expained at iteration " + str(iter_i))
            plt.show()
            u = {"ppca": U, "gt": U_gt}
            ppca_key = "ppca"
            n_rows = np.max([2, u[ppca_key].shape[-1]])
            n_cols = len(u.keys())
            fig_size = (n_cols * 4, n_rows * 4)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_size))
            for i, u_key in enumerate(u.keys()):
                # Plot PPCA components
                for j in range(u[ppca_key].shape[-1]):
                    axes[j, i].imshow(reference_dataset.get_proj(u[u_key][:, j].reshape(-1)))
                    axes[j, i].set_title(f"{u_key} PC{j + 1}")

            plt.tight_layout()
            plt.show()

    # Print final summary
    print("=" * 130)
    print("EM ALGORITHM COMPLETED")
    print("=" * 130)

    # Report final whitening statistics
    if use_whitening:
        C_z_final = compute_Cz_from_second_moments(second_moment_zs)
        q = W.shape[1]
        final_constraint_violation = float(jnp.linalg.norm(C_z_final - jnp.eye(q)))
        final_trace_Cz = float(jnp.trace(C_z_final))
        final_W_norm = float(jnp.linalg.norm(W))
        print("Final whitening statistics:")
        print(f"  ||Ĉ_z - I||_F = {final_constraint_violation:.6f}")
        print(f"  tr(Ĉ_z) = {final_trace_Cz:.4f} (target: {q})")
        print(f"  ||W||_F = {final_W_norm:.4f}")
        print("=" * 130)

    # Orthogonalize
    U, S, _ = jnp.linalg.svd(W, full_matrices=False)
    if return_iteration_data:
        return U, S**2, W, expected_zs, second_moment_zs, iteration_data
    return U, S**2, W, expected_zs, second_moment_zs


def compute_whitening_diagnostics(W, second_moment_zs):
    """
    Compute diagnostic statistics for the whitening constraint.

    Args:
        W: Loading matrix of shape (d, q)
        second_moment_zs: Posterior second moments E[zz^T|y] of shape (N, q, q)

    Returns:
        dict: Dictionary with diagnostic statistics
    """
    q = W.shape[1]
    I = jnp.eye(q)

    # Compute C_z
    C_z = compute_Cz_from_second_moments(second_moment_zs)

    # Eigenvalues of C_z
    eigvals = jnp.linalg.eigvalsh(C_z)

    return {
        "C_z": np.array(C_z),
        "constraint_violation": float(jnp.linalg.norm(C_z - I)),
        "trace_Cz": float(jnp.trace(C_z)),
        "W_norm": float(jnp.linalg.norm(W)),
        "Cz_eigenvalues": np.array(eigvals),
        "Cz_condition_number": float(eigvals.max() / (eigvals.min() + 1e-10)),
    }
