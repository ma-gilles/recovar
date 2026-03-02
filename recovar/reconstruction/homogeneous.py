"""Homogeneous (mean) 3-D reconstruction via direct Fourier inversion."""

import logging
import time

import numpy as np
from recovar.utils.nvtx_shim import nvtx

from recovar.reconstruction import regularization

logger = logging.getLogger(__name__)

NVTX_DOMAIN_HOMO = "homogeneous"


@nvtx.annotate("get_mean_conformation_relion", color="blue", domain=NVTX_DOMAIN_HOMO)
def get_mean_conformation_relion(
    cryos, batch_size, noise_variance=None, use_regularization=False, upsampling_factor=2,
):
    """Compute the mean conformation using RELION-style half-set reconstruction.

    Runs one accumulation pass per half-set, computes the FSC-based prior, then
    optionally re-post-processes with regularization.

    Parameters
    ----------
    cryos : CryoEMHalfsets (iterable of CryoEMDataset)
    batch_size : int
    noise_variance : array, optional
    use_regularization : bool
        If True, ``means["combined"]`` is the average of the per-half
        regularized reconstructions; otherwise it is the average of the
        unregularized reconstructions.
    upsampling_factor : int
        Volume oversampling factor (default 2).

    Returns
    -------
    means : dict with keys ``"corrected0"``, ``"corrected1"``, ``"combined"``,
        ``"corrected0reg"``, ``"corrected1reg"``
    mean_prior : ndarray  —  FSC-derived regularization prior
    fsc : ndarray
    """
    from recovar.reconstruction import relion_functions

    st_time = time.time()

    ft_ctfs = [None, None]
    ft_ys = [None, None]
    corrected = [None, None]

    for idx, cryo in enumerate(cryos):
        ft_ctfs[idx], ft_ys[idx] = relion_functions.relion_style_triangular_kernel(
            cryo, noise_variance.astype(np.float32), batch_size,
            upsampling_factor=upsampling_factor,
        )
        corrected[idx] = relion_functions.post_process_from_filter_v2(
            ft_ctfs[idx], ft_ys[idx], cryo.volume_shape, upsampling_factor,
        )

    mean_prior, fsc, _ = regularization.compute_relion_prior(
        cryos, noise_variance, corrected[0], corrected[1], batch_size,
    )

    corrected_reg = [
        relion_functions.post_process_from_filter_v2(
            ft_ctfs[idx], ft_ys[idx], cryo.volume_shape, upsampling_factor,
            tau=mean_prior,
        )
        for idx, cryo in enumerate(cryos)
    ]

    if use_regularization:
        combined = (corrected_reg[0] + corrected_reg[1]) / 2
    else:
        combined = (corrected[0] + corrected[1]) / 2

    mean_prior = np.array(mean_prior)
    means = {
        "combined":    np.array(combined),
        "corrected0":  np.array(corrected[0]),
        "corrected1":  np.array(corrected[1]),
        "corrected0reg": np.array(corrected_reg[0]),
        "corrected1reg": np.array(corrected_reg[1]),
    }

    logger.info("mean computation completed in %.2fs", time.time() - st_time)
    return means, mean_prior, fsc
