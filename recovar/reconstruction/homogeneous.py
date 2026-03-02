"""Homogeneous (mean) 3-D reconstruction via direct Fourier inversion."""

import logging
import time

import numpy as np
import nvtx

from recovar.reconstruction import regularization

logger = logging.getLogger(__name__)

# NVTX domain for homogeneous reconstruction
NVTX_DOMAIN_HOMO = "homogeneous"


@nvtx.annotate("get_mean_conformation_relion", color="blue", domain=NVTX_DOMAIN_HOMO)
def get_mean_conformation_relion(cryos, batch_size, noise_variance=None, use_regularization=False, 
                                upsampling_factor=2, disc_type='linear_interp', tau=None, 
                                use_spherical_mask=True, grid_correct=True):
    """
    Compute mean conformation using RELION-style reconstruction.
    
    Args:
        cryos: CryoEMHalfsets or iterable of CryoEMDataset
        batch_size: Batch size for processing
        noise_variance: Noise variance estimate
        use_regularization: Whether to use regularized reconstruction
        upsampling_factor: Volume upsampling factor
        disc_type: Discretization type
        tau: Regularization parameter
        use_spherical_mask: Whether to use spherical mask
        grid_correct: Whether to apply grid correction
    """
    st_time = time.time()
    from recovar.reconstruction import relion_functions

    means = {}
    ft_ctfs = [None, None]
    ft_ys = [None, None]
    kernel = 'triangular' if disc_type == 'linear_interp' else 'square'

    # First pass: compute unregularized reconstructions
    for idx, cryo in enumerate(cryos):
        # Compute triangular kernel filters
        ft_ctfs[idx], ft_ys[idx] = relion_functions.relion_style_triangular_kernel(
            cryo, noise_variance.astype(np.float32), batch_size, disc_type=disc_type,
            upsampling_factor=upsampling_factor,
        )

        # Post-process to get unregularized reconstruction
        means[f"corrected{idx}"] = relion_functions.post_process_from_filter_v2(
            ft_ctfs[idx], ft_ys[idx],
            cryo.volume_shape, upsampling_factor,
            tau=tau, kernel=kernel,
            use_spherical_mask=use_spherical_mask, grid_correct=grid_correct,
            gridding_correct="square", kernel_width=1,
        )

    # Compute prior from unregularized reconstructions
    mean_prior, fsc, _ = regularization.compute_relion_prior(
        cryos, noise_variance, means["corrected0"], means["corrected1"], batch_size
    )

    # Store unregularized combined mean
    means["combined"] = (means["corrected0"] + means["corrected1"]) / 2

    # Second pass: compute regularized reconstructions
    for idx, cryo in enumerate(cryos):
        means[f"corrected{idx}reg"] = relion_functions.post_process_from_filter_v2(
            ft_ctfs[idx], ft_ys[idx],
            cryo.volume_shape, upsampling_factor,
            tau=mean_prior, kernel=kernel,
            use_spherical_mask=use_spherical_mask, grid_correct=grid_correct,
            gridding_correct="square", kernel_width=1,
        )

    # Store regularized combined mean
    means["combined_regularized"] = (means["corrected0reg"] + means["corrected1reg"]) / 2

    # Use regularized version if requested
    if use_regularization:
        means["combined"] = means["combined_regularized"]

    # Compute combined LHS and convert to numpy arrays
    lhs = (ft_ctfs[0] + ft_ctfs[1]) / 2
    mean_prior = np.array(mean_prior)
    
    # Store additional results
    means["prior"] = mean_prior
    means["lhs"] = lhs

    # Convert all means to numpy arrays
    for key in means:
        means[key] = np.array(means[key])

    end_time = time.time()
    logger.info(" mean computation completed in %.2fs", end_time - st_time)

    return means, mean_prior, fsc

