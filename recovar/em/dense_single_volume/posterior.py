"""E-step posterior: two GEMMs + normalize.

Computes posterior weights gamma_{i,r,t} for every image, rotation,
and translation on the dense grid.  See docs/math/dense_single_volume_em.md
Section 3 for the derivation.
"""

import logging

import jax.numpy as jnp
import numpy as np

from recovar import utils
from recovar.core.configs import ForwardModelConfig
from recovar.em.core import compute_CTFed_proj_norms_eqx, compute_dot_products_eqx
from recovar.em.e_step import compute_probability_from_residual_normal_squared_one_image

from .projection_cache import precompute_projections
from .types import DenseEMPlan

logger = logging.getLogger(__name__)


def compute_posterior(
    config: ForwardModelConfig,
    experiment_dataset,
    volume: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    noise_variance: np.ndarray,
    disc_type: str,
    plan: DenseEMPlan,
    image_indices=None,
) -> np.ndarray:
    """Full dense E-step: project, dot products, norms, softmax.

    This is the dense homogeneous path of E_with_precompute (u=None),
    extracted verbatim.

    Returns:
        probabilities: (n_images, n_rotations, n_translations) float32.
    """
    n_rotations = rotations.shape[0]
    n_translations = translations.shape[0]
    n_images = experiment_dataset.n_images if image_indices is None else len(image_indices)

    # Step 1: Precompute projections for all rotations
    projections = precompute_projections(
        volume,
        rotations,
        experiment_dataset.image_shape,
        experiment_dataset.volume_shape,
        disc_type,
        batch_size=plan.projection_batch,
    )
    logger.info("done with precomp proj, batch size %s", plan.projection_batch)

    # Step 2: Cross-term GEMM (dot products)
    residuals = np.empty((n_images, n_rotations, n_translations))
    image_indices_arr = np.arange(n_images) if image_indices is None else np.asarray(image_indices)

    start_idx = 0
    for (
        batch,
        _rotation_matrices,
        _translations,
        ctf_params,
        _noise_variance,
        _particle_indices,
        indices,
    ) in experiment_dataset.iter_batches(
        plan.dot_product_batch,
        indices=image_indices_arr,
        by_image=False,
    ):
        end_idx = start_idx + len(indices)
        residuals[start_idx:end_idx] = compute_dot_products_eqx(
            config,
            projections,
            batch,
            translations,
            ctf_params,
            noise_variance,
        )
        start_idx = end_idx

    # Step 3: Compute |P_r mu|^2 for norm term
    projections = (jnp.abs(projections) ** 2).block_until_ready()
    logger.info("done with IP")
    utils.report_memory_device(logger=logger)

    # Step 4: Norm GEMM
    for array_indices, dataset_indices in utils.subset_and_indices_batch_iter(image_indices_arr, plan.norm_batch):
        res = compute_CTFed_proj_norms_eqx(
            config,
            projections,
            experiment_dataset.CTF_params[dataset_indices],
            noise_variance,
        )
        if array_indices[-1] == n_images - 1:
            res = res.block_until_ready()
        residuals[array_indices] += np.array(res[..., None])

    del projections
    logger.info("done with norms. Batch size %s", plan.norm_batch)

    # Step 5: Softmax normalization
    for array_indices, _ in utils.subset_and_indices_batch_iter(image_indices_arr, plan.prob_batch):
        residuals[array_indices] = compute_probability_from_residual_normal_squared_one_image(residuals[array_indices])

    logger.info("done probs. Batch size %s", plan.prob_batch)
    return residuals
