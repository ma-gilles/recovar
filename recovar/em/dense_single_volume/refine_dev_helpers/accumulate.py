"""M-step: accumulate Ft_y and Ft_ctf via batched backprojection.

Consumes posterior weights and accumulates two Fourier-domain sufficient
statistics that are additive over image batches and devices.
See docs/math/dense_single_volume_em.md Section 4 for the derivation.
"""

import logging

import jax.numpy as jnp
import numpy as np

from recovar import utils
from recovar.core.configs import ForwardModelConfig
from recovar.em.m_step import sum_up_images_fixed_rots_eqx

from .types import DenseEMPlan, MeanStats

logger = logging.getLogger(__name__)


def accumulate_sufficient_statistics(
    config: ForwardModelConfig,
    experiment_dataset,
    probabilities: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    noise_variance: np.ndarray,
    plan: DenseEMPlan,
    image_indices=None,
) -> MeanStats:
    """Full dense M-step accumulation.

    Verbatim extraction of M_with_precompute lines 258-298.

    Returns:
        MeanStats(Ft_y, Ft_ctf) accumulated over all image and rotation batches.
    """
    n_rotations = rotations.shape[0]

    Ft_y = jnp.zeros(experiment_dataset.volume_size, dtype=experiment_dataset.dtype)
    Ft_ctf = jnp.zeros(experiment_dataset.volume_size, dtype=experiment_dataset.dtype)

    logger.info(
        "Starting sum up images. Batch size %s, rotation batch %s",
        plan.mstep_image_batch,
        plan.mstep_rotation_batch,
    )

    start_idx = 0
    for (
        batch,
        _rotation_matrices,
        _translations,
        ctf_params,
        _noise_variance,
        _particle_indices,
        batch_image_indices,
    ) in experiment_dataset.iter_batches(
        plan.mstep_image_batch,
        indices=image_indices,
        by_image=False,
    ):
        batch = jnp.asarray(batch)
        end_idx = start_idx + len(batch_image_indices)

        for rot_indices in utils.index_batch_iter(n_rotations, plan.mstep_rotation_batch):
            Ft_y, Ft_ctf = sum_up_images_fixed_rots_eqx(
                config,
                batch,
                probabilities[start_idx:end_idx, rot_indices[0] : rot_indices[-1] + 1],
                translations,
                rotations[rot_indices],
                ctf_params,
                noise_variance,
                Ft_y=Ft_y,
                Ft_ctf=Ft_ctf,
            )

        start_idx = end_idx

    return MeanStats(Ft_y=Ft_y, Ft_ctf=Ft_ctf)
