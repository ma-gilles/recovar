"""Top-level orchestration for dense single-volume EM.

This module is the dense homogeneous equivalent of E_M_batches_2 +
finish_up_M_step, readable top-to-bottom by someone who only cares
about dense homogeneous EM.

Execution model:
    prepare iteration
    -> precompute projections once (inside compute_posterior)
    -> for each image batch:
         compute posteriors
         immediately accumulate mean stats
         optionally record hard assignments
    -> finalize mean from stats
"""

import logging

import jax.numpy as jnp
import numpy as np

from recovar import utils
from recovar.core.configs import ForwardModelConfig

from .accumulate import accumulate_sufficient_statistics
from .plan import plan_em_iteration
from .posterior import compute_posterior
from .solver import solve_mean
from .types import MeanStats

logger = logging.getLogger(__name__)


def run_dense_em_iteration(
    experiment_dataset,
    mean: np.ndarray,
    mean_variance,
    noise_variance: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    disc_type: str,
    memory_to_use_gb: float = 128.0,
) -> tuple:
    """Run one complete dense single-volume EM iteration.

    This combines E-step (posterior), M-step (accumulate), and solve
    into a single top-level call.

    Args:
        experiment_dataset: CryoEM dataset with iter_batches, CTF_params, etc.
        mean: Current volume estimate, (volume_size,) complex.
        mean_variance: Prior variance (scalar or array).
        noise_variance: Noise level, (image_size,) float.
        rotations: (n_rot, 3, 3) rotation matrices.
        translations: (n_trans, 2) in-plane translations.
        disc_type: Discretization type (e.g. "linear_interp").
        memory_to_use_gb: Memory budget for outer image batching (GB).

    Returns:
        (new_mean, hard_assignment, Ft_y, Ft_ctf)
    """
    n_rotations = rotations.shape[0]
    n_translations = translations.shape[0]
    if n_rotations <= 0:
        raise ValueError("requires at least one rotation")
    if n_translations <= 0:
        raise ValueError("requires at least one translation")

    total_hidden = n_rotations * n_translations
    logger.info(
        "starting dense EM iteration. Num rotations %s, num translations %s. Total = %s",
        n_rotations,
        n_translations,
        total_hidden,
    )

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    # Outer image batch (same formula as E_M_batches_2)
    n_images_batch = max(1, int(memory_to_use_gb * 1e9 / (total_hidden * 8)))
    logger.info(
        "n_images_batch %s. Number of batches %s",
        n_images_batch,
        int(np.ceil(experiment_dataset.n_units / n_images_batch)),
    )

    # Initialize accumulators
    Ft_y = jnp.zeros(experiment_dataset.volume_size, dtype=experiment_dataset.dtype)
    Ft_ctf = jnp.zeros(experiment_dataset.volume_size, dtype=experiment_dataset.dtype)
    hard_assignment = np.empty(experiment_dataset.n_units, dtype=int)

    for big_image_batch in utils.index_batch_iter(experiment_dataset.n_units, n_images_batch):
        big_image_batch = np.asarray(big_image_batch)

        # Plan for this sub-iteration
        plan = plan_em_iteration(
            grid_size=experiment_dataset.grid_size,
            n_rotations=n_rotations,
            n_translations=n_translations,
            memory_to_use_gb=memory_to_use_gb,
        )

        # E-step: compute posteriors
        probabilities = compute_posterior(
            config,
            experiment_dataset,
            mean,
            rotations,
            translations,
            noise_variance,
            disc_type,
            plan,
            image_indices=big_image_batch,
        )

        # Hard assignment for convergence tracking
        hard_assignment[big_image_batch] = np.argmax(probabilities.reshape(probabilities.shape[0], -1), axis=-1)

        if np.isnan(probabilities).any():
            logger.warning("NaNs detected in probabilities; mean norm=%s", np.linalg.norm(mean))

        # M-step: accumulate sufficient statistics
        stats = accumulate_sufficient_statistics(
            config,
            experiment_dataset,
            probabilities,
            rotations,
            translations,
            noise_variance,
            plan,
            image_indices=big_image_batch,
        )
        Ft_y = Ft_y + stats.Ft_y
        Ft_ctf = Ft_ctf + stats.Ft_ctf

    # Solve: Wiener-filtered reconstruction
    new_mean = solve_mean(
        experiment_dataset,
        MeanStats(Ft_y=Ft_y, Ft_ctf=Ft_ctf),
        mean_variance,
        disc_type,
    )

    return new_mean, hard_assignment, Ft_y, Ft_ctf
