"""Two-pass adaptive oversampling for dense single-volume EM.

Implements Phase 5 of the RELION-parity plan: significance pruning after
a coarse E-step pass, then oversampled evaluation of only significant
orientations x translations.

RELION's approach:
- Pass 1 (coarse): evaluate ALL rotations at base angular sampling using
  a smaller Fourier window.  Compute posterior weights.  Identify significant
  (rotation, translation) pairs per image.
- Pass 2 (fine): for each image, evaluate ONLY its significant coarse
  rotations' children at oversampled angles using a larger Fourier window.

The significance criterion matches RELION's ``adaptive_fraction``: keep
the smallest set of (rotation, translation) samples that together contribute
>= adaptive_fraction of total posterior weight.  Cap at max_significants
to bound memory/compute (RELION's --maxsig semantics, counting SAMPLES
not just orientations -- see C5 in plan_relion_parity.md).

See docs/math/plan_relion_parity.md, Phase 5.
"""

import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Significance pruning
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=())
def find_significant_mask(weights_flat, adaptive_fraction=0.999, max_significants=500):
    """Find significant orientation x translation pairs per image.

    For each image, identifies the smallest set of (rotation, translation)
    samples whose cumulative posterior weight >= adaptive_fraction of total.
    Caps at max_significants per image.

    Parameters
    ----------
    weights_flat : jnp.ndarray, shape (n_images, n_rot * n_trans)
        Posterior weights (probabilities) for each image, flattened over
        the rotation x translation grid.  Must sum to ~1.0 per image.
    adaptive_fraction : float
        Fraction of total weight to keep (default 0.999 = 99.9%).
    max_significants : int
        Maximum number of significant samples per image.

    Returns
    -------
    mask : jnp.ndarray, shape (n_images, n_rot * n_trans), dtype bool
        True for significant samples.
    n_significant : jnp.ndarray, shape (n_images,), dtype int32
        Number of significant samples per image (after capping).
    """
    n_images, n_samples = weights_flat.shape

    # Sort descending per image
    sorted_w = jnp.sort(weights_flat, axis=-1)[:, ::-1]
    cumsum = jnp.cumsum(sorted_w, axis=-1)
    total = weights_flat.sum(axis=-1, keepdims=True)

    # Fraction of total weight accumulated so far
    frac = cumsum / jnp.maximum(total, 1e-30)

    # Find the index where we first exceed adaptive_fraction
    # argmax on a boolean gives the first True index
    threshold_idx = jnp.argmax(frac >= adaptive_fraction, axis=-1)

    # Cap at max_significants - 1 (0-indexed)
    threshold_idx = jnp.minimum(threshold_idx, max_significants - 1)

    # Get the threshold value: the weight at the threshold index
    threshold_val = sorted_w[jnp.arange(n_images), threshold_idx]

    # Mask: keep all samples with weight >= threshold
    mask = weights_flat >= threshold_val[:, None]

    # Count significant samples per image
    n_significant = jnp.sum(mask, axis=-1).astype(jnp.int32)

    return mask, n_significant


def find_significant_rotations(weights_flat, n_rot, n_trans,
                               adaptive_fraction=0.999,
                               max_significants=500):
    """Find significant coarse rotations per image from (rot x trans) weights.

    This extracts the unique rotation indices that have at least one
    significant (rotation, translation) pair, which is what we need
    for generating oversampled children in pass 2.

    Parameters
    ----------
    weights_flat : jnp.ndarray, shape (n_images, n_rot * n_trans)
        Posterior weights.
    n_rot : int
        Number of rotations in the coarse grid.
    n_trans : int
        Number of translations.
    adaptive_fraction : float
        Fraction of total weight to keep.
    max_significants : int
        Maximum significant samples (rot x trans).

    Returns
    -------
    sig_mask : jnp.ndarray, shape (n_images, n_rot * n_trans), dtype bool
        Significance mask over the full rot x trans grid.
    sig_rot_mask : jnp.ndarray, shape (n_images, n_rot), dtype bool
        True for rotations that have at least one significant translation.
    n_significant : jnp.ndarray, shape (n_images,), dtype int32
        Total significant (rot x trans) samples per image.
    """
    sig_mask, n_significant = find_significant_mask(
        weights_flat, adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
    )

    # Reshape to (n_images, n_rot, n_trans) and check if any translation
    # is significant for each rotation
    sig_2d = sig_mask.reshape(-1, n_rot, n_trans)
    sig_rot_mask = jnp.any(sig_2d, axis=-1)  # (n_images, n_rot)

    return sig_mask, sig_rot_mask, n_significant


# ---------------------------------------------------------------------------
# Pass 2: sparse oversampled evaluation
# ---------------------------------------------------------------------------

def compute_pass2_stats(
    experiment_dataset,
    volume,
    mean_variance,
    noise_variance,
    coarse_rotations,
    translations,
    sig_rot_mask,
    nside_level,
    disc_type,
    oversampling_order=1,
    current_size=None,
    image_batch_size=500,
):
    """Pass 2: evaluate oversampled children of significant coarse rotations.

    For each image, generates child rotations (4x per parent at next healpix
    level) for its significant coarse rotations, then evaluates the E-step
    and accumulates M-step statistics.

    This is a simplified implementation that unions all significant rotations
    across the batch and evaluates them densely.  This is correct but may
    evaluate extra rotations for some images.  A per-image sparse approach
    would be more efficient but is deferred to a later optimization pass.

    Parameters
    ----------
    experiment_dataset : dataset object
        One half-set dataset.
    volume : jnp.ndarray, shape (volume_size,)
        Current volume estimate.
    mean_variance : jnp.ndarray
        Signal prior (tau^2).
    noise_variance : jnp.ndarray
        Per-pixel noise variance.
    coarse_rotations : np.ndarray, shape (n_coarse_rot, 3, 3)
        Coarse rotation grid (base healpix level).
    translations : jnp.ndarray, shape (n_trans, 2)
        Translation grid.
    sig_rot_mask : jnp.ndarray, shape (n_images, n_coarse_rot), dtype bool
        Per-image significant rotation mask from pass 1.
    nside_level : int
        HEALPix level of the coarse rotation grid.
    disc_type : str
        Discretization type.
    oversampling_order : int
        Number of healpix subdivision levels (1 = 4x children).
    current_size : int or None
        Fourier window size for pass 2 (can be larger than pass 1).
    image_batch_size : int
        Images per GPU batch.

    Returns
    -------
    Ft_y : jnp.ndarray, shape (volume_size,)
        Accumulated weighted image sums.
    Ft_ctf : jnp.ndarray, shape (volume_size,)
        Accumulated CTF^2 weights.
    hard_assignments : np.ndarray, shape (n_images,)
        Best (rotation_idx * n_trans + trans_idx) indices into the
        OVERSAMPLED grid.
    oversampled_rotations : np.ndarray, shape (n_oversampled, 3, 3)
        The oversampled rotation matrices used.
    """
    from recovar.em.sampling import get_oversampled_rotation_grid
    from .engine_v2 import run_em_v2

    n_images = experiment_dataset.n_units
    n_coarse_rot = coarse_rotations.shape[0]

    # Union of all significant rotation indices across all images
    # This is conservative but ensures we evaluate all needed rotations
    # Support both per-image mask (n_images, n_rot) and global mask (n_rot,)
    sig_mask_np = np.asarray(sig_rot_mask)
    if sig_mask_np.ndim == 2:
        sig_rot_any = np.any(sig_mask_np, axis=0)  # (n_coarse_rot,)
    else:
        sig_rot_any = sig_mask_np.astype(bool)  # already (n_coarse_rot,)
    sig_rot_indices = np.where(sig_rot_any)[0]

    if len(sig_rot_indices) == 0:
        logger.warning("No significant rotations found; skipping pass 2")
        Ft_y = jnp.zeros(experiment_dataset.volume_size, dtype=experiment_dataset.dtype)
        Ft_ctf = jnp.zeros(experiment_dataset.volume_size, dtype=experiment_dataset.dtype)
        ha = np.zeros(n_images, dtype=np.int32)
        return Ft_y, Ft_ctf, ha, coarse_rotations[:0]

    # Map coarse rotation indices to healpix pixel indices
    # The coarse rotation grid has n_pixels * n_in_planes rotations.
    # Each healpix pixel has n_in_planes in-plane angles.
    import healpy as hp
    nside = 2 ** nside_level
    n_pixels = hp.nside2npix(nside)
    angle_res = 360 / (6 * 2 ** nside_level)
    n_in_planes = int(np.round(360 / angle_res))

    # sig_rot_indices are into the full coarse rotation grid (pixel * n_in_planes)
    # Get unique parent healpix pixels
    sig_pixel_indices = sig_rot_indices // n_in_planes
    unique_pixels = np.unique(sig_pixel_indices)

    logger.info(
        "Pass 2: %d significant coarse rotations -> %d unique healpix pixels "
        "-> generating %d oversampled children (order=%d)",
        len(sig_rot_indices), len(unique_pixels),
        len(unique_pixels) * (4 ** oversampling_order), oversampling_order,
    )

    # Generate oversampled rotation matrices for children of significant pixels
    oversampled_rots, parent_map = get_oversampled_rotation_grid(
        unique_pixels, nside_level, oversampling_order=oversampling_order,
    )
    oversampled_rots = np.asarray(oversampled_rots, dtype=np.float32)

    logger.info(
        "Pass 2: %d oversampled rotations (from %d parent pixels)",
        len(oversampled_rots), len(unique_pixels),
    )

    # Run a full dense E+M at the oversampled grid
    # This is correct: we evaluate all oversampled rotations for all images.
    # The significance pruning's benefit is that len(oversampled_rots) <<
    # len(coarse_rotations) * 4^oversampling_order.
    new_mean, ha, Ft_y, Ft_ctf = run_em_v2(
        experiment_dataset,
        volume,
        mean_variance,
        noise_variance,
        oversampled_rots,
        translations,
        disc_type,
        image_batch_size=image_batch_size,
        rotation_block_size=min(5000, len(oversampled_rots)),
        current_size=current_size,
    )

    return Ft_y, Ft_ctf, ha, oversampled_rots


def extract_posterior_weights_from_scores(scores_all_blocks, log_Z, n_images, n_rot, n_trans):
    """Convert accumulated scores from all blocks into posterior weights.

    Parameters
    ----------
    scores_all_blocks : list of jnp.ndarray
        Each element has shape (n_images, block_rot, n_trans).
    log_Z : jnp.ndarray, shape (n_images,)
        Log normalizing constant.
    n_images, n_rot, n_trans : int
        Grid dimensions.

    Returns
    -------
    weights_flat : jnp.ndarray, shape (n_images, n_rot * n_trans)
        Posterior probabilities, summing to ~1 per image.
    """
    # Concatenate all blocks along the rotation axis
    # Each block is (n_images, block_rot, n_trans)
    all_scores = jnp.concatenate(scores_all_blocks, axis=1)  # (n_images, n_rot_padded, n_trans)

    # Trim to actual n_rot (remove padding)
    all_scores = all_scores[:, :n_rot, :]  # (n_images, n_rot, n_trans)

    # Normalize to probabilities
    weights = jnp.exp(all_scores - log_Z[:, None, None])
    weights_flat = weights.reshape(n_images, n_rot * n_trans)

    return weights_flat
