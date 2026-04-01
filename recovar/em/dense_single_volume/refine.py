"""FSC-driven multi-iteration refinement loop for dense single-volume EM.

Wires FSC -> current_size -> Fourier window into the iteration loop,
implementing Phases 4 and 5 of the RELION-parity plan.

The loop:
1. Compute FSC between half-maps -> determine current_size
2. Quantize current_size to allowed values
3. Run engine_v2 E+M on each half-set at that current_size
4. Optionally: two-pass adaptive oversampling (Phase 5):
   - Pass 1 (coarse): dense E-step at coarse resolution, find significant
     (rotation, translation) pairs per image.
   - Pass 2 (fine): oversampled E+M at finer resolution for significant
     rotations only.
5. Wiener-solve each half-map
6. Estimate noise, update prior
7. Log progress

Supports oracle mode: inject RELION's per-iteration current_sizes to
isolate windowing from the statistical model.

See docs/math/plan_relion_parity.md, Phases 4 and 5.
"""

import logging
import time

import jax.numpy as jnp
import numpy as np

from recovar.em.core import hard_assignment_idx_to_pose
from recovar.em.dense_single_volume.engine_v2 import run_em_v2
from recovar.em.dense_single_volume.fourier_window import quantize_current_size
from recovar.em.dense_single_volume.adaptive import (
    find_significant_rotations,
    compute_pass2_stats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batched significance pruning (avoids materializing full weight matrix)
# ---------------------------------------------------------------------------

def _compute_significance_batched(
    experiment_dataset,
    mean,
    noise_variance,
    rotations,
    translations,
    disc_type,
    adaptive_fraction,
    max_significants,
    image_batch_size,
    rotation_block_size,
    current_size,
):
    """Run coarse E-step and find significant rotations in a memory-efficient way.

    Instead of materializing the full (n_images, n_rot * n_trans) weight matrix,
    this processes one image batch at a time: for each batch, it computes the
    posterior weights, finds significance, and accumulates the union of significant
    rotation indices.

    Parameters
    ----------
    Returns
    -------
    sig_rot_any : np.ndarray, shape (n_rot,), dtype bool
        True for rotations that are significant for at least one image.
    n_sig_all : np.ndarray, shape (n_images,), dtype int32
        Per-image count of significant (rot x trans) samples.
    hard_assignments : np.ndarray, shape (n_images,), dtype int32
        Best (rot_idx * n_trans + trans_idx) per image from coarse pass.
    """
    from recovar.em.dense_single_volume.engine_v2 import (
        _preprocess_batch, _compute_projections_block,
        _e_step_block_scores, _e_step_block_scores_windowed,
        _update_logsumexp, make_half_image_weights,
    )
    from recovar.core.configs import ForwardModelConfig
    import recovar.core.fourier_transform_utils as fourier_transform_utils
    from recovar.em.dense_single_volume.adaptive import find_significant_rotations as _find_sig

    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    n_images = experiment_dataset.n_units
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape

    H, W = image_shape
    n_half = H * (W // 2 + 1)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset, disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    half_weights = make_half_image_weights(image_shape)

    use_window = current_size is not None and current_size < image_shape[0]
    if use_window:
        from .fourier_window import make_fourier_window_indices_np
        window_indices_np, n_windowed = make_fourier_window_indices_np(image_shape, current_size)
        window_indices = jnp.asarray(window_indices_np)
        half_weights_windowed = half_weights[window_indices]
    else:
        window_indices = None
        n_windowed = n_half

    # Pad rotations
    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate([
            rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))
        ], axis=0)
    else:
        rotations_padded = rotations

    # Accumulate results
    sig_rot_any = np.zeros(n_rot, dtype=bool)
    n_sig_all = np.empty(n_images, dtype=np.int32)
    hard_assignment = np.empty(n_images, dtype=np.int32)

    image_indices = np.arange(n_images)
    start_idx = 0

    for (batch_data, _, _, ctf_params, _, _, indices) in experiment_dataset.iter_batches(
        image_batch_size, indices=image_indices, by_image=False,
    ):
        batch_size = len(indices)
        end_idx = start_idx + batch_size
        batch_data = jnp.asarray(batch_data)

        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            batch_data, ctf_params, noise_variance, translations, config,
            batch_size, n_trans,
        )

        if use_window:
            shifted_data = shifted_half[:, window_indices]
            ctf2_data = ctf2_over_nv_half[:, window_indices]
        else:
            shifted_data = shifted_half
            ctf2_data = ctf2_over_nv_half

        # Pass 1: streaming logsumexp
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean, rots_b, image_shape, volume_shape, disc_type)

            if use_window:
                proj_w = proj_half_b[:, window_indices]
                proj_abs2_w = proj_abs2_half_b[:, window_indices]
                scores = _e_step_block_scores_windowed(
                    shifted_data, batch_norm, ctf2_data,
                    proj_w * half_weights_windowed,
                    proj_abs2_w * half_weights_windowed,
                    half_weights_windowed,
                    batch_size, n_trans, n_windowed, image_shape, volume_shape,
                )
            else:
                scores = _e_step_block_scores(
                    shifted_data, batch_norm, ctf2_data,
                    proj_half_b * half_weights,
                    proj_abs2_half_b * half_weights,
                    half_weights,
                    batch_size, n_trans, image_shape, volume_shape,
                )

            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)

            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)

        log_Z = max_s + jnp.log(sum_exp)

        # Pass 2: recompute scores, normalize -> batch weights
        best_score = jnp.full(batch_size, -jnp.inf)
        best_argmax = jnp.zeros(batch_size, dtype=jnp.int32)
        batch_weights_blocks = []

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean, rots_b, image_shape, volume_shape, disc_type)

            if use_window:
                proj_w = proj_half_b[:, window_indices]
                proj_abs2_w = proj_abs2_half_b[:, window_indices]
                scores = _e_step_block_scores_windowed(
                    shifted_data, batch_norm, ctf2_data,
                    proj_w * half_weights_windowed,
                    proj_abs2_w * half_weights_windowed,
                    half_weights_windowed,
                    batch_size, n_trans, n_windowed, image_shape, volume_shape,
                )
            else:
                scores = _e_step_block_scores(
                    shifted_data, batch_norm, ctf2_data,
                    proj_half_b * half_weights,
                    proj_abs2_half_b * half_weights,
                    half_weights,
                    batch_size, n_trans, image_shape, volume_shape,
                )

            if r1 > n_rot:
                valid = n_rot - r0
                pmask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(pmask[None, :, None], scores, -jnp.inf)

            probs = jnp.exp(scores - log_Z[:, None, None])

            block_best = jnp.max(scores.reshape(batch_size, -1), axis=1)
            block_argmax = jnp.argmax(scores.reshape(batch_size, -1), axis=1)
            improved = block_best > best_score
            best_score = jnp.where(improved, block_best, best_score)
            best_argmax = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax)

            actual_rot = min(rotation_block_size, n_rot - r0)
            block_probs = probs[:, :actual_rot, :]
            batch_weights_blocks.append(np.asarray(block_probs.reshape(batch_size, -1)))

        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax)

        # Concatenate this batch's weights -> (batch_size, n_rot * n_trans)
        batch_weights = np.concatenate(batch_weights_blocks, axis=1)

        # Find significance for this batch
        _, batch_sig_rot_mask, batch_n_sig = _find_sig(
            jnp.asarray(batch_weights),
            n_rot, n_trans,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
        )

        # Accumulate global union of significant rotations
        batch_sig_rot_any = np.asarray(jnp.any(batch_sig_rot_mask, axis=0))
        sig_rot_any |= batch_sig_rot_any

        n_sig_all[start_idx:end_idx] = np.asarray(batch_n_sig)
        start_idx = end_idx

    return sig_rot_any, n_sig_all, hard_assignment


# ---------------------------------------------------------------------------
# FSC -> current_size conversion
# ---------------------------------------------------------------------------

def fsc_to_current_size(fsc, threshold=1.0 / 7.0, min_size=32):
    """Convert an FSC curve to a current_size (diameter in pixels).

    Parameters
    ----------
    fsc : array-like, shape (n_shells,)
        FSC curve between half-maps.
    threshold : float
        FSC threshold for resolution cutoff.  Default 1/7 ~ 0.143.
    min_size : int
        Minimum returned size (prevents collapse to 0 at first iteration).

    Returns
    -------
    int
        Raw current_size = 2 * shell_index.  Needs quantization before use.
    """
    from recovar.heterogeneity.locres import find_fsc_resol

    fsc_arr = jnp.asarray(fsc)
    pixel_res = float(find_fsc_resol(fsc_arr, threshold=threshold))

    # current_size = 2 * shell_index (Nyquist: need 2 pixels per cycle)
    raw_size = int(2 * pixel_res)
    return max(raw_size, min_size)


# ---------------------------------------------------------------------------
# Main refinement loop
# ---------------------------------------------------------------------------

def refine_single_volume(
    experiment_datasets,
    init_volume,
    init_noise_variance,
    init_mean_variance,
    rotations,
    translations,
    disc_type="linear_interp",
    max_iter=10,
    image_batch_size=500,
    rotation_block_size=5000,
    relion_current_sizes=None,
    init_current_size=32,
    fsc_threshold=1.0 / 7.0,
    adaptive_oversampling=0,
    adaptive_fraction=0.999,
    max_significants=500,
    nside_level=None,
    translation_pixel_offset=None,
):
    """Multi-iteration EM refinement with FSC-driven resolution management.

    Parameters
    ----------
    experiment_datasets : list of 2 dataset objects
        Half-set datasets (same format as split_E_M_v2 expects).
    init_volume : jnp.ndarray, shape (volume_size,)
        Initial volume in Fourier space.
    init_noise_variance : jnp.ndarray, shape (image_size,)
        Initial per-pixel noise variance.
    init_mean_variance : jnp.ndarray, shape (volume_size,)
        Initial signal prior (tau^2).
    rotations : np.ndarray, shape (n_rot, 3, 3)
        Rotation grid.
    translations : jnp.ndarray, shape (n_trans, 2)
        Translation grid.
    disc_type : str
        Discretization type for forward/adjoint slicing.
    max_iter : int
        Maximum number of iterations.
    image_batch_size : int
        Number of images per GPU batch.
    rotation_block_size : int
        Number of rotations per block in engine_v2.
    relion_current_sizes : list of int or None
        Oracle mode: if provided, use these current_sizes instead of
        computing from FSC.  relion_current_sizes[i] is used at iteration i.
    init_current_size : int
        Starting current_size for the first iteration (when no FSC is
        available yet).  Ignored if relion_current_sizes is provided.
    fsc_threshold : float
        FSC threshold for resolution estimation.
    adaptive_oversampling : int
        Number of HEALPix subdivision levels for pass 2 (0=disabled,
        1=2x finer = 4 children, 2=4x finer = 16 children).
    adaptive_fraction : float
        Fraction of posterior weight to keep in significance pruning
        (default 0.999 = 99.9%, matching RELION).
    max_significants : int
        Maximum significant (rotation x translation) samples per image.
        Matches RELION's --maxsig semantics (counts SAMPLES, not just
        orientations; see C5 in plan_relion_parity.md).
    nside_level : int or None
        HEALPix level of the coarse rotation grid.  Required when
        adaptive_oversampling > 0.
    translation_pixel_offset : float or None
        Step size between coarse translation grid points (pixels).
        Required when adaptive_oversampling > 0.

    Returns
    -------
    dict with keys:
        mean : jnp.ndarray -- final merged mean volume
        means : list of 2 jnp.ndarray -- per-half-set means
        fsc : jnp.ndarray -- final FSC curve
        hard_assignments : list of 2 np.ndarray -- per-half-set assignments
        current_sizes : list of int -- current_size at each iteration
        fsc_history : list of jnp.ndarray -- FSC curve at each iteration
        pixel_resolutions : list of float -- pixel resolution at each iter
        wall_times : list of float -- wall time per iteration
        significant_counts : list of (jnp.ndarray or None) -- per-image
            significant sample counts at each iteration (None when
            adaptive_oversampling=0).
    """
    from recovar.reconstruction import regularization, noise, relion_functions

    if adaptive_oversampling > 0 and nside_level is None:
        raise ValueError(
            "nside_level must be provided when adaptive_oversampling > 0"
        )

    cryo = experiment_datasets[0]
    volume_shape = cryo.volume_shape

    # State: two half-set volumes, noise, prior
    means = [jnp.array(init_volume), jnp.array(init_volume)]
    noise_variance = jnp.array(init_noise_variance)
    mean_variance = jnp.array(init_mean_variance)

    # History tracking
    current_sizes = []
    fsc_history = []
    pixel_resolutions = []
    wall_times = []
    hard_assignments = [None, None]
    significant_counts = []

    for iteration in range(max_iter):
        t0 = time.time()

        # --- Determine current_size ---
        if relion_current_sizes is not None:
            if iteration < len(relion_current_sizes):
                cs = int(relion_current_sizes[iteration])
            else:
                cs = int(relion_current_sizes[-1])
            if cs <= 0:
                cs = init_current_size
        elif iteration == 0:
            cs = init_current_size
        else:
            fsc_prev = regularization.get_fsc_gpu(
                means[0], means[1], volume_shape,
            )
            raw_cs = fsc_to_current_size(fsc_prev, threshold=fsc_threshold)
            cs = quantize_current_size(raw_cs)

        cs = quantize_current_size(cs)
        current_sizes.append(cs)

        logger.info(
            "=== Iteration %d/%d: current_size=%d ===",
            iteration + 1, max_iter, cs,
        )

        use_adaptive = adaptive_oversampling > 0
        cs_for_engine = cs if cs < cryo.image_shape[0] else None

        # --- Run E+M on each half-set ---
        iter_sig_counts = None

        for k in range(2):
            if not use_adaptive:
                # Standard single-pass E+M (Phase 4 behavior)
                new_mean_k, ha_k, Ft_y_k, Ft_ctf_k = run_em_v2(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance,
                    rotations,
                    translations,
                    disc_type,
                    image_batch_size=image_batch_size,
                    rotation_block_size=rotation_block_size,
                    current_size=cs_for_engine,
                )
            else:
                # Two-pass adaptive oversampling (Phase 5)
                # Pass 1: batched significance pruning (memory-efficient)
                sig_rot_any, n_sig, ha_coarse = _compute_significance_batched(
                    experiment_datasets[k],
                    means[k],
                    noise_variance,
                    rotations,
                    translations,
                    disc_type,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    image_batch_size=image_batch_size,
                    rotation_block_size=rotation_block_size,
                    current_size=cs_for_engine,
                )

                if k == 0:
                    iter_sig_counts = n_sig

                n_sig_np = np.asarray(n_sig)
                logger.info(
                    "Pass 1 (half %d): significant samples per image: "
                    "min=%d, median=%d, max=%d, mean=%.0f; "
                    "union significant rotations: %d / %d",
                    k, int(n_sig_np.min()), int(np.median(n_sig_np)),
                    int(n_sig_np.max()), float(n_sig_np.mean()),
                    int(np.sum(sig_rot_any)), rotations.shape[0],
                )

                # Pass 2: oversampled E+M on significant rotations
                # sig_rot_any is (n_rot,) bool -- the global union
                Ft_y_k, Ft_ctf_k, ha_k, oversampled_rots = compute_pass2_stats(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance,
                    np.asarray(rotations),
                    translations,
                    sig_rot_any,
                    nside_level,
                    disc_type,
                    oversampling_order=adaptive_oversampling,
                    current_size=cs_for_engine,
                    image_batch_size=image_batch_size,
                )

                # Solve for this half-set using pass 2 statistics
                new_mean_k = relion_functions.post_process_from_filter(
                    experiment_datasets[k], Ft_ctf_k, Ft_y_k,
                    tau=mean_variance, disc_type=disc_type,
                ).reshape(-1)

            means[k] = new_mean_k
            hard_assignments[k] = ha_k

            if k == 0:
                Ft_y_0, Ft_ctf_0 = Ft_y_k, Ft_ctf_k
            else:
                Ft_y_1, Ft_ctf_1 = Ft_y_k, Ft_ctf_k

        significant_counts.append(iter_sig_counts)

        # --- Compute unregularized half-maps for FSC and prior ---
        unreg_means = [
            relion_functions.post_process_from_filter(
                cryo, Ft_ctf_0, Ft_y_0, tau=None, disc_type=disc_type,
            ),
            relion_functions.post_process_from_filter(
                cryo, Ft_ctf_1, Ft_y_1, tau=None, disc_type=disc_type,
            ),
        ]

        # --- Compute FSC between half-maps ---
        fsc = regularization.get_fsc_gpu(
            unreg_means[0], unreg_means[1], volume_shape,
        )
        fsc_history.append(fsc)

        # --- Resolution from FSC ---
        from recovar.heterogeneity.locres import find_fsc_resol
        pixel_res = float(find_fsc_resol(fsc, threshold=fsc_threshold))
        pixel_resolutions.append(pixel_res)

        # --- Update prior (RELION-style tau^2 from FSC) ---
        mean_signal_variance, _, _ = regularization.compute_relion_prior(
            experiment_datasets, noise_variance, unreg_means[0], unreg_means[1], 100,
        )
        mean_variance = mean_signal_variance

        # --- Update noise estimate ---
        # For adaptive oversampling, the hard assignments index into the
        # oversampled grid, not the original.  Use the coarse assignments
        # for pose updates.
        effective_rots = rotations
        effective_trans = translations
        for k in range(2):
            if use_adaptive and oversampled_rots is not None and len(oversampled_rots) > 0:
                best_rots, best_trans = hard_assignment_idx_to_pose(
                    hard_assignments[k], oversampled_rots, translations,
                )
            else:
                best_rots, best_trans = hard_assignment_idx_to_pose(
                    hard_assignments[k], rotations, translations,
                )
            experiment_datasets[k].update_poses(best_rots, best_trans)

        noise_from_res = noise.estimate_noise_level_no_masks(
            experiment_datasets[0],
            np.arange(min(1000, cryo.n_units)),
            means[0],
            100,
            disc_type=disc_type,
        )
        noise_variance = noise.make_radial_noise(noise_from_res, cryo.image_shape)

        # --- Timing ---
        elapsed = time.time() - t0
        wall_times.append(elapsed)

        res_angstrom = pixel_res / cryo.voxel_size if cryo.voxel_size > 0 else pixel_res
        logger.info(
            "Iteration %d: current_size=%d, pixel_res=%.1f, "
            "res=%.2f A, time=%.1fs",
            iteration + 1, cs, pixel_res, res_angstrom, elapsed,
        )

    # --- Final merged mean ---
    merged_mean = (means[0] + means[1]) / 2

    return {
        "mean": merged_mean,
        "means": means,
        "fsc": fsc_history[-1] if fsc_history else None,
        "hard_assignments": hard_assignments,
        "current_sizes": current_sizes,
        "fsc_history": fsc_history,
        "pixel_resolutions": pixel_resolutions,
        "wall_times": wall_times,
        "significant_counts": significant_counts,
    }
