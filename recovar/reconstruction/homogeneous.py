"""Homogeneous (mean) 3-D reconstruction via direct Fourier inversion."""

import logging
import time
from dataclasses import dataclass

import numpy as np
from recovar.utils.nvtx_shim import nvtx

from recovar.reconstruction import regularization

logger = logging.getLogger(__name__)


@dataclass
class MeanEstimate:
    """Result of half-set mean reconstruction.

    Attributes
    ----------
    combined : ndarray — average of the two half-set reconstructions.
    corrected0, corrected1 : ndarray — unregularized half-set means.
    corrected0reg, corrected1reg : ndarray — regularized half-set means.
    lhs : ndarray — average Hermitian (CTF²) filter.
    prior : ndarray — FSC-derived regularization prior.
    """
    combined: np.ndarray
    corrected0: np.ndarray
    corrected1: np.ndarray
    corrected0reg: np.ndarray
    corrected1reg: np.ndarray
    lhs: np.ndarray
    prior: np.ndarray
    cubic_coeffs: np.ndarray = None  # precomputed spline coefficients for combined

    def negate(self):
        """Flip sign of all mean volumes (for uninverting data)."""
        self.combined = -self.combined
        self.corrected0 = -self.corrected0
        self.corrected1 = -self.corrected1
        self.corrected0reg = -self.corrected0reg
        self.corrected1reg = -self.corrected1reg

    def corrected(self, idx):
        """Get unregularized half-map by index (0 or 1)."""
        return self.corrected0 if idx == 0 else self.corrected1

    def corrected_reg(self, idx):
        """Get regularized half-map by index (0 or 1)."""
        return self.corrected0reg if idx == 0 else self.corrected1reg

NVTX_DOMAIN_HOMO = "homogeneous"


@nvtx.annotate("get_mean_conformation_relion", color="blue", domain=NVTX_DOMAIN_HOMO)
def get_mean_conformation_relion(
    dataset, batch_size, noise_variance=None, use_regularization=False, upsampling_factor=2,
):
    """Compute the mean conformation using RELION-style half-set reconstruction.

    Runs one accumulation pass per half-set, computes the FSC-based prior, then
    optionally re-post-processes with regularization.

    Parameters
    ----------
    dataset : CryoEMDataset (with ``halfset_indices`` set) or CryoEMHalfsets
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
    means : MeanEstimate
    mean_prior : ndarray  —  FSC-derived regularization prior
    fsc : ndarray
    """
    from recovar.reconstruction import relion_functions

    # Support both new (single dataset) and legacy (CryoEMHalfsets) API
    from recovar.data_io.dataset import unwrap_dataset
    dataset = unwrap_dataset(dataset)

    st_time = time.time()

    ft_ctfs = [None, None]
    ft_ys = [None, None]
    corrected = [None, None]

    for half in range(2):
        ft_ctfs[half], ft_ys[half] = relion_functions.relion_style_triangular_kernel(
            dataset, noise_variance.astype(np.float32), batch_size,
            index_subset=dataset.halfset_indices[half],
            upsampling_factor=upsampling_factor,
        )
        corrected[half] = relion_functions.post_process_from_filter_v2(
            ft_ctfs[half], ft_ys[half], dataset.volume_shape, upsampling_factor,
        )

    mean_prior, fsc, _ = regularization.compute_relion_prior(
        dataset, noise_variance, corrected[0], corrected[1], batch_size,
    )

    corrected_reg = [
        relion_functions.post_process_from_filter_v2(
            ft_ctfs[half], ft_ys[half], dataset.volume_shape, upsampling_factor,
            tau=mean_prior,
        )
        for half in range(2)
    ]

    if use_regularization:
        combined = (corrected_reg[0] + corrected_reg[1]) / 2
    else:
        combined = (corrected[0] + corrected[1]) / 2

    mean_prior = np.array(mean_prior)
    means = MeanEstimate(
        combined=np.array(combined),
        corrected0=np.array(corrected[0]),
        corrected1=np.array(corrected[1]),
        corrected0reg=np.array(corrected_reg[0]),
        corrected1reg=np.array(corrected_reg[1]),
        lhs=np.array((ft_ctfs[0] + ft_ctfs[1]) / 2),
        prior=mean_prior,
    )

    logger.info("mean computation completed in %.2fs", time.time() - st_time)
    return means, mean_prior, fsc


