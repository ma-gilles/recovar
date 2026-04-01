"""PPCA EM with optional contrast marginalization.

E-step: compute per-image posterior moments E[z], E[cz], E[c²zz^T] etc.
M-step: accumulate per-voxel normal equations and solve for W.

See docs/math/contrast_marginalization.md for the derivation.
"""
import logging

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from recovar import core
from recovar.core import linalg
from recovar.core.configs import ForwardModelConfig, ModelState, EmbeddingOpts
from recovar.heterogeneity import contrast_posterior, embedding

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# E-step: compute sufficient statistics → solve latent posterior
# ═══════════════════════════════════════════════════════════════════════

def ppca_e_step(experiment_dataset, mean_estimate, W, eigenvalues,
                volume_mask, noise_variance, batch_size,
                disc_type="linear_interp",
                contrast_mode="none",
                contrast_grid=None, contrast_mean=1.0, contrast_variance=np.inf):
    """E-step: per-image posterior moments with optional contrast marginalization.

    Returns
    -------
    mean_z : (N, K)       E[z|y]
    mean_cz : (N, K)      E[cz|y]
    mean_c2z : (N, K)     E[c²z|y]
    sm_czz : (N, K, K)    E[c²zz^T|y]
    mean_c : (N,)         E[c|y]
    ll_sum : float        Sum of per-image log-likelihoods
    """
    basis_size = W.shape[0] if W.ndim == 2 else W.shape[1]
    # W can be (volume_size, K) or (K, volume_size) — normalize to (K, vol)
    if W.shape[0] > W.shape[1]:
        basis = jnp.asarray(W, dtype=experiment_dataset.dtype)  # (vol, K)
    else:
        basis = jnp.asarray(W.T, dtype=experiment_dataset.dtype)  # (vol, K)

    n_images = experiment_dataset.n_images

    mean_z = np.zeros((n_images, basis_size), dtype=np.float32)
    mean_cz = np.zeros((n_images, basis_size), dtype=np.float32)
    mean_c2z = np.zeros((n_images, basis_size), dtype=np.float32)
    sm_czz = np.zeros((n_images, basis_size, basis_size), dtype=np.float32)
    mean_c = np.zeros(n_images, dtype=np.float32)
    ll_sum = 0.0

    # Setup model and config for _compute_batch_coords_p1
    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    mean_est_gpu = jnp.asarray(mean_estimate, dtype=experiment_dataset.dtype)
    basis_gpu = basis.T  # _prepare expects (K, vol)
    mean_est_gpu, basis_gpu = embedding._prepare_model_half_volumes(config, mean_est_gpu, basis_gpu)
    volume_mask_gpu = jnp.asarray(volume_mask, dtype=experiment_dataset.dtype_real)
    eigenvalues_gpu = jnp.asarray(eigenvalues[:basis_size], dtype=experiment_dataset.dtype_real)

    model = ModelState(
        mean_estimate=mean_est_gpu,
        volume_mask=volume_mask_gpu,
        basis=basis_gpu,
        eigenvalues=eigenvalues_gpu,
    )
    hermitian_weights = embedding._embedding_hermitian_weights(config)
    noise_model = experiment_dataset.noise

    # Contrast quadrature setup
    if contrast_mode == "marginalize" and contrast_grid is not None:
        _, contrast_weights = contrast_posterior.make_contrast_quadrature(
            rule="trapezoid", nodes=np.asarray(contrast_grid))
    else:
        contrast_weights = None

    for (
        batch, rotation_matrices, translations, ctf_params,
        _noise_var, _particle_ind, image_indices,
    ) in experiment_dataset.iter_batches(
        batch_size, by_image=True,
        noise_model=noise_model,
        noise_half=(hermitian_weights is not None),
    ):
        batch = jnp.asarray(batch)
        batch_idx = np.asarray(image_indices).reshape(-1)
        nv = embedding._noise_get_half_or_full(
            noise_model, batch_idx, prefer_half=(hermitian_weights is not None))

        # Phase 1: compute sufficient statistics
        AU_t_images, AU_t_Amean, AU_t_AU, image_norms_sq, image_T_A_mean, A_mean_norm_sq = \
            embedding._compute_batch_coords_p1(
                config, batch, model, hermitian_weights,
                rotation_matrices=rotation_matrices,
                translations=translations,
                ctf_params=ctf_params,
                noise_variance=nv,
            )

        # Phase 2: solve latent posterior
        result = contrast_posterior.solve_latent_posterior(
            H=AU_t_AU,
            g=AU_t_images,
            h=AU_t_Amean,
            t=image_T_A_mean,
            nu=A_mean_norm_sq,
            y_norm_sq=image_norms_sq,
            lambdas=eigenvalues_gpu,
            contrast_mode=contrast_mode,
            contrast_nodes=jnp.asarray(contrast_grid) if contrast_grid is not None else None,
            contrast_weights=contrast_weights,
            contrast_mean=float(contrast_mean),
            contrast_variance=float(contrast_variance),
        )

        mean_z[batch_idx] = np.asarray(result.mean_z)
        mean_cz[batch_idx] = np.asarray(result.mean_cz)
        mean_c2z[batch_idx] = np.asarray(result.mean_c2z)
        sm_czz[batch_idx] = np.asarray(result.second_moment_czz)
        mean_c[batch_idx] = np.asarray(result.mean_c)

    return mean_z, mean_cz, mean_c2z, sm_czz, mean_c, ll_sum


# ═══════════════════════════════════════════════════════════════════════
# M-step: accumulate normal equations, solve per-voxel
# ═══════════════════════════════════════════════════════════════════════

def M_step_batch(
    images, lhs_summed, rhs_summed, mean_corr_summed,
    mean_cz_batch, mean_c2z_batch, sm_czz_batch,
    CTF_params, rotation_matrices, translations,
    image_shape, volume_shape, grid_size, voxel_size,
    noise_variance, ctf,
):
    """Accumulate one batch into LHS, RHS, and mean-correction accumulators.

    LHS(p) += sum_i CTF²/σ² · E[c²zz^T]
    RHS(p) += sum_i CTF/σ² · y_i · E[cz]^T
    mean_corr(p) += sum_i CTF²/σ² · E[c²z]^T
    """
    CTF = ctf(CTF_params, image_shape, voxel_size)
    ctf_over_noise = CTF**2 / noise_variance

    grid_point_indices = core.batch_get_nearest_gridpoint_indices(
        rotation_matrices, image_shape, volume_shape)

    # LHS: CTF²/σ² * E[c²zz^T]
    sm = sm_czz_batch.reshape(sm_czz_batch.shape[0], 1, -1)  # (B, 1, K²)
    sm_weighted = sm * ctf_over_noise[:, :, None]  # (B, pix, K²)
    lhs_summed = lhs_summed.at[grid_point_indices.reshape(-1)].add(
        sm_weighted.reshape(-1, sm_weighted.shape[-1]))

    # RHS term 1: CTF/σ² * y * E[cz]^T
    images = core.translate_images(images, translations, image_shape)
    weighted_images = images * CTF / noise_variance
    rhs_term = linalg.broadcast_outer(weighted_images, mean_cz_batch)  # (B, pix, K)
    rhs_summed = rhs_summed.at[grid_point_indices.reshape(-1)].add(
        rhs_term.reshape(-1, rhs_term.shape[-1]))

    # Mean correction: CTF²/σ² * E[c²z]^T (accumulated, multiplied by μ later)
    mc = mean_c2z_batch.reshape(mean_c2z_batch.shape[0], 1, -1)  # (B, 1, K)
    mc_weighted = mc * ctf_over_noise[:, :, None]  # (B, pix, K)
    mean_corr_summed = mean_corr_summed.at[grid_point_indices.reshape(-1)].add(
        mc_weighted.reshape(-1, mc_weighted.shape[-1]))

    return lhs_summed, rhs_summed, mean_corr_summed


def M_step(experiment_dataset, mean_estimate,
           mean_cz, mean_c2z, sm_czz, noise_variance, batch_size):
    """Solve for W given contrast-weighted posterior moments.

    Per-voxel: (LHS) W = RHS - μ · mean_corr
    """
    basis_size = mean_cz.shape[-1]
    vol_size = experiment_dataset.volume_size
    dtype = experiment_dataset.dtype

    lhs_summed = jnp.zeros((vol_size, basis_size * basis_size), dtype=dtype)
    rhs_summed = jnp.zeros((vol_size, basis_size), dtype=dtype)
    mean_corr_summed = jnp.zeros((vol_size, basis_size), dtype=dtype)

    for (
        images, rotation_matrices, translations, ctf_params,
        _noise_var, _particle_ind, image_indices,
    ) in experiment_dataset.iter_batches(batch_size):
        idx = np.asarray(image_indices).reshape(-1)
        lhs_summed, rhs_summed, mean_corr_summed = M_step_batch(
            images, lhs_summed, rhs_summed, mean_corr_summed,
            jnp.asarray(mean_cz[idx]),
            jnp.asarray(mean_c2z[idx]),
            jnp.asarray(sm_czz[idx]),
            ctf_params, rotation_matrices, translations,
            experiment_dataset.image_shape,
            experiment_dataset.volume_shape,
            experiment_dataset.grid_size,
            experiment_dataset.voxel_size,
            noise_variance,
            experiment_dataset.ctf_evaluator,
        )

    # Apply mean correction: RHS -= μ_p · mean_corr_p
    mean_flat = jnp.asarray(mean_estimate).reshape(-1)
    rhs_final = rhs_summed - mean_flat[:, None] * mean_corr_summed

    # Solve per-voxel: LHS @ W_p = RHS_p
    lhs = lhs_summed.reshape(vol_size, basis_size, basis_size)
    W = linalg.solve_by_SVD(lhs, rhs_final, hermitian=True)

    # Orthogonalize
    U, S, _ = jnp.linalg.svd(W, full_matrices=False)
    W = U @ jnp.diag(S)

    return W


# ═══════════════════════════════════════════════════════════════════════
# EM loop
# ═══════════════════════════════════════════════════════════════════════

def EM(experiment_dataset, mean_estimate, noise_variance,
       EM_iter=20, basis_size=10, batch_size=1000,
       disc_type="linear_interp",
       contrast_mode="none", contrast_grid=None,
       contrast_mean=1.0, contrast_variance=np.inf,
       volume_mask=None, seed=0):
    """PPCA EM with optional contrast marginalization.

    Parameters
    ----------
    contrast_mode : str
        ``"none"`` (c=1), ``"profile"`` (MAP over c), ``"marginalize"`` (quadrature).
    contrast_grid : array or None
        Contrast nodes for profile/marginalize modes.
    """
    if volume_mask is None:
        volume_mask = np.ones(experiment_dataset.volume_shape)
    if contrast_grid is None and contrast_mode != "none":
        contrast_grid = np.linspace(0.0, 3.0, 16)

    # Initialize W randomly
    W = jr.normal(jr.PRNGKey(seed),
                  (experiment_dataset.volume_size, basis_size),
                  dtype=experiment_dataset.dtype_real)
    W = linalg.batch_dft3(W, experiment_dataset.volume_shape, basis_size)
    eigenvalues = np.ones(basis_size, dtype=np.float32)

    for it in range(EM_iter):
        # E-step
        mean_z, mean_cz, mean_c2z, sm_czz, mean_c, ll = ppca_e_step(
            experiment_dataset, mean_estimate, W, eigenvalues,
            volume_mask, noise_variance, batch_size,
            disc_type=disc_type,
            contrast_mode=contrast_mode,
            contrast_grid=contrast_grid,
            contrast_mean=contrast_mean,
            contrast_variance=contrast_variance,
        )

        # M-step
        W = M_step(experiment_dataset, mean_estimate,
                   mean_cz, mean_c2z, sm_czz, noise_variance, batch_size)

        logger.info("EM %d: mean_c=%.3f±%.3f", it, np.mean(mean_c), np.std(mean_c))

    return W, mean_z, mean_c
