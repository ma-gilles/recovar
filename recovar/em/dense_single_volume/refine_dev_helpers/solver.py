"""RELION-style filtered solve for dense single-volume EM.

After accumulating Ft_y and Ft_ctf over all images, the mean update
is a regularized Wiener-filtered backprojection solve.
See docs/math/dense_single_volume_em.md Section 4.2.
"""

import numpy as np

from recovar.reconstruction import relion_functions

from .types import MeanStats


def solve_mean(
    experiment_dataset,
    stats: MeanStats,
    mean_variance,
    disc_type: str,
) -> np.ndarray:
    """Wiener-filtered reconstruction from accumulated sufficient statistics.

    Thin wrapper around relion_functions.post_process_from_filter.
    This is the exact call from EMState.finish_up_M_step.

    Returns:
        new_mean: (volume_size,) complex array.
    """
    return relion_functions.post_process_from_filter(
        experiment_dataset,
        stats.Ft_ctf,
        stats.Ft_y,
        tau=mean_variance,
        disc_type=disc_type,
    ).reshape(-1)
