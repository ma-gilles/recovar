"""Homogeneous (mean) 3-D reconstruction via direct Fourier inversion."""

import logging
import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from recovar.reconstruction import regularization
from recovar.utils.nvtx_shim import nvtx

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
    dataset,
    batch_size,
    noise_variance=None,
    use_regularization=False,
    upsampling_factor=2,
):
    """Compute the mean conformation using RELION-style half-set reconstruction.

    Runs one accumulation pass per half-set, computes the FSC-based prior, then
    optionally re-post-processes with regularization.

    Parameters
    ----------
    dataset : CryoEMDataset (with ``halfset_indices`` set)
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

    st_time = time.time()
    halfset_datasets = dataset.materialize_halfset_datasets()

    ft_ctfs = [None, None]
    ft_ys = [None, None]
    corrected = [None, None]

    for halfset_id, halfset_dataset in enumerate(halfset_datasets):
        ft_ctfs[halfset_id], ft_ys[halfset_id] = relion_functions.relion_style_triangular_kernel(
            halfset_dataset,
            noise_variance.astype(np.float32),
            batch_size,
            upsampling_factor=upsampling_factor,
        )
        corrected[halfset_id] = relion_functions.post_process_from_filter_v2(
            ft_ctfs[halfset_id],
            ft_ys[halfset_id],
            halfset_dataset.volume_shape,
            upsampling_factor,
        )

    mean_prior, fsc, _ = regularization.compute_relion_prior(
        halfset_datasets,
        noise_variance,
        corrected[0],
        corrected[1],
        batch_size,
    )

    corrected_reg = [
        relion_functions.post_process_from_filter_v2(
            ft_ctfs[halfset_id],
            ft_ys[halfset_id],
            dataset.volume_shape,
            upsampling_factor,
            tau=mean_prior,
        )
        for halfset_id in range(2)
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


def get_mean_pcg(
    dataset,
    batch_size,
    noise_variance=None,
    volume_mask=None,
    upsampling_factor=2,
    lam=0.0,
    pcg_maxiter=20,
    pcg_tol=1e-4,
    x0_real=None,
):
    """Compute masked mean via PCG on support-constrained Wiener system.

    Same accumulation as ``get_mean_conformation_relion``, then solves
    with PCG instead of direct Wiener division.

    Parameters
    ----------
    volume_mask : ndarray (original volume shape), optional
        Real-space support mask.  If None, uses spherical mask.
    lam : float
        Spatial regularization strength (|k|^{-2} weighting).
    pcg_maxiter : int
        Maximum CG iterations (10-20 with warmstart).
    pcg_tol : float
        Relative residual tolerance.
    x0_real : ndarray, optional
        Warmstart initial guess (real-space, original volume shape).
    """
    from recovar.reconstruction import regularization, relion_functions
    from recovar.reconstruction.pcg_mean import pcg_mean

    st_time = time.time()
    halfset_datasets = dataset.materialize_halfset_datasets()

    # Accumulate half-sets
    ft_ctfs = [None, None]
    ft_ys = [None, None]
    corrected = [None, None]

    for halfset_id, halfset_dataset in enumerate(halfset_datasets):
        ft_ctfs[halfset_id], ft_ys[halfset_id] = relion_functions.relion_style_triangular_kernel(
            halfset_dataset,
            noise_variance.astype(np.float32),
            batch_size,
            upsampling_factor=upsampling_factor,
        )
        corrected[halfset_id] = relion_functions.post_process_from_filter_v2(
            ft_ctfs[halfset_id],
            ft_ys[halfset_id],
            halfset_dataset.volume_shape,
            upsampling_factor,
        )

    mean_prior, fsc, _ = regularization.compute_relion_prior(
        halfset_datasets,
        noise_variance,
        corrected[0],
        corrected[1],
        batch_size,
    )

    # Build regularized d and rhs in upsampled space
    og_shape = dataset.volume_shape
    up_shape = tuple(3 * [og_shape[0] * upsampling_factor])

    # Combine both half-sets
    ft_ctf_combined = (ft_ctfs[0] + ft_ctfs[1]) / 2
    ft_y_combined = (ft_ys[0] + ft_ys[1]) / 2

    # Expand half-volume → full if needed
    from recovar.core import fourier_transform_utils as ftu

    packed_shape = ftu.volume_shape_to_half_volume_shape(up_shape)
    if ft_ctf_combined.size == np.prod(packed_shape):
        ft_ctf_full = ftu.half_volume_to_full_volume(
            jnp.array(ft_ctf_combined).reshape(packed_shape), up_shape
        ).real.reshape(up_shape)
        ft_y_full = ftu.half_volume_to_full_volume(jnp.array(ft_y_combined).reshape(packed_shape), up_shape).reshape(
            up_shape
        )
    else:
        ft_ctf_full = jnp.array(ft_ctf_combined).reshape(up_shape).real
        ft_y_full = jnp.array(ft_y_combined).reshape(up_shape)

    # Regularize d (CTF² + prior)
    d_reg = jnp.array(
        relion_functions.adjust_regularization_relion_style(
            ft_ctf_full.reshape(-1).real,
            up_shape,
            tau=jnp.array(mean_prior),
            padding_factor=upsampling_factor,
        )
    ).reshape(up_shape)

    # Build mask in upsampled space
    up_n = up_shape[0]
    N = og_shape[0]
    if volume_mask is not None:
        mask_up = np.zeros(up_shape, dtype=np.float32)
        s = (up_n - N) // 2
        mask_up[s : s + N, s : s + N, s : s + N] = np.array(volume_mask)
    else:
        # Default: spherical mask
        x = np.linspace(-1, 1, up_n)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2)
        mask_up = (r < N / up_n).astype(np.float32)

    mask_jax = jnp.array(mask_up)

    # Optional: spatial weight w = |k|^{-2} on support
    w_support = None
    if lam > 0:
        w_support = mask_jax  # uniform weight for now

    # Warmstart: pad x0 to upsampled grid
    x0_up = None
    if x0_real is not None:
        x0_up = np.zeros(up_shape, dtype=np.float32)
        s = (up_n - N) // 2
        x0_up[s : s + N, s : s + N, s : s + N] = np.array(x0_real)
        x0_up = jnp.array(x0_up)

    # PCG solve
    x_pcg, residuals = pcg_mean(
        d_reg,
        ft_y_full,
        mask_jax,
        lam=lam,
        w_support=w_support,
        x0=x0_up,
        maxiter=pcg_maxiter,
        tol=pcg_tol,
        precondition=True,
    )

    # Post-process: crop to original size + gridding correction
    from recovar.core import padding

    if up_n > N:
        vol = padding.unpad_volume_spatial_domain(x_pcg, up_n - N)
    else:
        vol = x_pcg
    vol_corrected, _ = relion_functions.griddingCorrect_square(vol.reshape(og_shape), N, upsampling_factor, order=1)

    # Return as Fourier volume (flat)
    combined = ftu.get_dft3(vol_corrected.reshape(og_shape)).reshape(-1)

    # Also build the standard corrected maps for FSC comparison
    corrected_reg = [
        relion_functions.post_process_from_filter_v2(
            ft_ctfs[hid], ft_ys[hid], og_shape, upsampling_factor, tau=mean_prior
        )
        for hid in range(2)
    ]

    means = MeanEstimate(
        combined=np.array(combined),
        corrected0=np.array(corrected[0]),
        corrected1=np.array(corrected[1]),
        corrected0reg=np.array(corrected_reg[0]),
        corrected1reg=np.array(corrected_reg[1]),
        lhs=np.array(ft_ctf_combined),
        prior=np.array(mean_prior),
    )

    logger.info(
        "PCG mean: %d iters, rr=%.2e, time=%.2fs",
        len(residuals),
        residuals[-1] if residuals else 0.0,
        time.time() - st_time,
    )
    return means, np.array(mean_prior), fsc
