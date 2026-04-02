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

# RELION-parity building blocks (used only by mode="relion")
from recovar.em.dense_single_volume.convergence import (
    RefinementState,
    update_refinement_state,
    check_convergence,
    should_refine_angular_sampling,
    refine_angular_sampling,
    compute_ave_Pmax,
    healpix_angular_step,
)
from recovar.em.sampling import (
    get_rotation_grid_at_order,
    get_local_rotation_grid,
    get_local_rotation_grid_fast,
    get_translation_grid,
)
from recovar.reconstruction.regularization import (
    compute_data_vs_prior,
    resolution_from_data_vs_prior,
    compute_current_size_relion,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coarse image size for adaptive oversampling (RELION parity)
# ---------------------------------------------------------------------------

def compute_coarse_image_size(
    angular_step_deg, pixel_size, ori_size, particle_diameter=None,
):
    """Compute the coarse image size for pass 1 of adaptive oversampling.

    RELION formula (expectation.cpp line 5760):
        rotated_distance = (angular_step / 360) * pi * particle_diameter
        coarse_resolution = rotated_distance / 1.2       (3D)
        image_coarse_size = 2 * ceil(pixel_size * ori_size / coarse_resolution)

    Parameters
    ----------
    angular_step_deg : float
        Effective angular step in degrees (after oversampling).
    pixel_size : float
        Pixel size in Angstrom.
    ori_size : int
        Original image box size in pixels.
    particle_diameter : float or None
        Particle diameter in Angstrom.  If None, use box_size * pixel_size.

    Returns
    -------
    coarse_size : int
        Coarse image size (diameter in pixels), clamped to [8, ori_size].
    """
    if particle_diameter is None:
        particle_diameter = ori_size * pixel_size

    rotated_distance = (angular_step_deg / 360.0) * np.pi * particle_diameter
    coarse_resolution = rotated_distance / 1.2  # keepsafe_factor for 3D

    if coarse_resolution <= 0:
        return ori_size

    coarse_size = int(2 * np.ceil(pixel_size * ori_size / coarse_resolution))
    coarse_size = max(8, min(coarse_size, ori_size))
    return coarse_size


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
    mode="legacy",
    # --- RELION-mode parameters (only used when mode="relion") ---
    init_healpix_order=2,
    max_healpix_order=7,
    init_translation_range=10.0,
    init_translation_step=2.0,
    save_intermediates_dir=None,
):
    """Multi-iteration EM refinement with FSC-driven resolution management.

    Supports two modes:

    - ``mode="legacy"`` (default): Original FSC-driven loop with fixed
      rotation grid.  All existing behavior is preserved exactly.

    - ``mode="relion"``: RELION-parity mode with convergence detection,
      angular step refinement, local angular search, and data_vs_prior
      resolution criterion.  Uses :class:`RefinementState` from
      ``convergence.py`` to drive the iteration.

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
        Rotation grid.  In legacy mode, used directly.  In RELION mode,
        used as the initial grid (overridden when angular step refines).
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
    mode : str
        ``"legacy"`` preserves existing behavior.  ``"relion"`` enables
        RELION-parity convergence-driven refinement.
    init_healpix_order : int
        Starting HEALPix order for RELION mode (default 2, ~14.7 deg).
    max_healpix_order : int
        Maximum HEALPix order (finest angular sampling, default 7).
    init_translation_range : float
        Initial translation search range in pixels (RELION mode).
    init_translation_step : float
        Initial translation step size in pixels (RELION mode).

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

    Additional keys when ``mode="relion"``:
        convergence_state : RefinementState -- final convergence state
        data_vs_prior_trajectory : list of jnp.ndarray -- per-iteration
            data_vs_prior curves
        healpix_order_trajectory : list of int -- HEALPix order per iter
        ave_Pmax_trajectory : list of float -- average Pmax per iter
    """
    if mode not in ("legacy", "relion"):
        raise ValueError(f"Unknown mode={mode!r}; expected 'legacy' or 'relion'")

    if mode == "relion":
        return _refine_relion_mode(
            experiment_datasets=experiment_datasets,
            init_volume=init_volume,
            init_noise_variance=init_noise_variance,
            init_mean_variance=init_mean_variance,
            rotations=rotations,
            translations=translations,
            disc_type=disc_type,
            max_iter=max_iter,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            init_current_size=init_current_size,
            fsc_threshold=fsc_threshold,
            adaptive_oversampling=adaptive_oversampling,
            init_healpix_order=init_healpix_order,
            max_healpix_order=max_healpix_order,
            init_translation_range=init_translation_range,
            init_translation_step=init_translation_step,
            nside_level=nside_level,
            save_intermediates_dir=save_intermediates_dir,
        )

    # ===================================================================
    # mode="legacy" — existing code below is UNTOUCHED
    # ===================================================================
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

                if Ft_y_k is None:
                    # Pass 2 was skipped (union too large); fall back to
                    # pass-1-only mode using the coarse grid.
                    logger.info(
                        "Half %d: pass 2 skipped, running pass-1-only E+M",
                        k,
                    )
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
                    oversampled_rots = None
                else:
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


# ---------------------------------------------------------------------------
# RELION-parity refinement mode
# ---------------------------------------------------------------------------


def _extract_max_posterior_per_image(
    experiment_dataset,
    mean,
    noise_variance,
    rotations,
    translations,
    disc_type,
    image_batch_size,
    rotation_block_size,
    current_size,
):
    """Extract per-image maximum posterior probability from an E-step pass.

    Runs a lightweight E-step (pass 1 only: streaming logsumexp) and
    returns the best log-score per image, converted to a probability
    via exp(best_score - log_Z).

    This is a separate utility because run_em_v2 does not currently expose
    per-image Pmax.  We reuse the same block structure as run_em_v2's
    pass 1 to compute log_Z and best_score in a memory-efficient way.

    Returns
    -------
    max_prob : np.ndarray, shape (n_images,)
        Per-image maximum posterior probability in [0, 1].
    """
    from recovar.em.dense_single_volume.engine_v2 import (
        _preprocess_batch, _compute_projections_block,
        _e_step_block_scores, _e_step_block_scores_windowed,
        _update_logsumexp, make_half_image_weights,
    )
    from recovar.core.configs import ForwardModelConfig
    import recovar.core.fourier_transform_utils as fourier_transform_utils

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

    max_prob_all = np.empty(n_images, dtype=np.float32)

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

        # Streaming logsumexp + best score tracking
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size)
        best_score = jnp.full(batch_size, -jnp.inf)

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
            # Track best score per image across all blocks
            block_best = jnp.max(scores.reshape(batch_size, -1), axis=1)
            best_score = jnp.maximum(best_score, block_best)

        # Pmax = exp(best_score - log_Z)
        log_Z = max_s + jnp.log(sum_exp)
        pmax = jnp.exp(best_score - log_Z)
        max_prob_all[start_idx:end_idx] = np.asarray(pmax)
        start_idx = end_idx

    return max_prob_all


def _refine_relion_mode(
    experiment_datasets,
    init_volume,
    init_noise_variance,
    init_mean_variance,
    rotations,
    translations,
    disc_type,
    max_iter,
    image_batch_size,
    rotation_block_size,
    init_current_size,
    fsc_threshold,
    adaptive_oversampling,
    init_healpix_order,
    max_healpix_order,
    init_translation_range,
    init_translation_step,
    nside_level,
    save_intermediates_dir=None,
):
    """RELION-parity refinement loop with convergence detection.

    This implements the full RELION auto-refine algorithm:
    1. Convergence-driven iteration (not fixed max_iter)
    2. data_vs_prior for resolution instead of FSC < 0.143
    3. Angular step refinement (HEALPix order increments)
    4. Local angular search when HEALPix order >= 4
    5. Per-image best assignment tracking
    6. Average Pmax computation for adaptive current_size growth

    Corresponds to RELION's autoRefine iteration loop.
    See docs/relion5_auto_refine_algorithm.md.
    """
    from recovar.reconstruction import regularization, noise, relion_functions

    cryo = experiment_datasets[0]
    volume_shape = cryo.volume_shape
    grid_size = cryo.image_shape[0]  # ori_size in RELION terms

    # --- Initialize RefinementState ---
    # Corresponds to RELION's initialiseSamplingVectors + initialLowPassFilterReferences
    state = RefinementState(
        iteration=0,
        healpix_order=init_healpix_order,
        adaptive_oversampling=adaptive_oversampling,
        translation_range=init_translation_range,
        translation_step=init_translation_step,
        max_healpix_order=max_healpix_order,
        current_resolution=float("inf"),
    )

    # Use the provided rotation grid as the initial grid.
    # In RELION mode, the grid may be regenerated at higher HEALPix orders
    # when angular step refinement triggers.
    current_rotations = np.asarray(rotations, dtype=np.float32)
    current_translations = jnp.asarray(translations, dtype=jnp.float32)
    current_healpix_order = init_healpix_order

    # Determine nside_level for current rotations.  If nside_level is provided,
    # use it; otherwise infer from init_healpix_order.
    current_nside_level = nside_level if nside_level is not None else init_healpix_order

    # RELION uses padding_factor=2 for reconstruction (8x Fourier grid).
    # Defined once here so all reconstruction calls use the same value.
    PADDING_FACTOR = 2

    def _safe_batch_sizes(n_rot):
        """Reduce batch sizes for large grids to avoid GPU OOM."""
        # Target: ~50M elements in the score tensor (n_img * n_rot * n_trans)
        budget = 50_000_000
        rbs = min(rotation_block_size, max(500, budget // max(n_rot, 1)))
        ibs = min(image_batch_size, max(50, budget // max(n_rot, 1)))
        return ibs, rbs

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
    previous_assignments = [None, None]
    significant_counts = []
    data_vs_prior_trajectory = []
    healpix_order_trajectory = []
    ave_Pmax_trajectory = []

    # RELION uses padding_factor=2 for reconstruction: the 3D Fourier
    # grid is (2*N)^3 to reduce interpolation artifacts at high freq.
    PADDING_FACTOR = 2

    iteration = 0
    while not state.has_converged and iteration < max_iter:
        t0 = time.time()

        # --- Determine current_size using data_vs_prior (RELION C4/C5) ---
        # At iteration 0, no FSC or tau2 available yet; use init_current_size.
        # After iteration 0, compute data_vs_prior from the combined Ft_ctf
        # and the current tau2 (mean_variance).
        if iteration == 0:
            cs = init_current_size
            data_vs_prior_iter = None
        else:
            # Compute FSC between the two half-maps for tau2 estimation
            fsc_prev = regularization.get_fsc_gpu(
                means[0], means[1], volume_shape,
            )
            # Compute data_vs_prior using the combined Fourier weights.
            # Ft_ctf_combined is computed after the previous iteration's M-step.
            # We stored Ft_ctf_0 + Ft_ctf_1 from the previous iteration.
            n_shells = volume_shape[0] // 2 - 1
            # mean_variance is stored as per-voxel; extract radial profile
            # for data_vs_prior (which needs per-shell tau2).
            tau2_radial = regularization.average_over_shells(
                mean_variance.real, volume_shape,
            )
            data_vs_prior_iter = compute_data_vs_prior(
                Ft_ctf_combined, tau2_radial, volume_shape,
                padding_factor=PADDING_FACTOR,
            )
            data_vs_prior_trajectory.append(data_vs_prior_iter)

            # Resolution from data_vs_prior (RELION uses this, not FSC < 0.143)
            res_shell = resolution_from_data_vs_prior(data_vs_prior_iter)

            # Check if FSC is still high at the resolution limit
            # (used by compute_current_size_relion for aggressive growth)
            has_high_fsc = False
            if res_shell < len(np.asarray(fsc_prev)):
                has_high_fsc = float(fsc_prev[res_shell]) > 0.2

            # RELION's current_size growth logic (C5)
            raw_cs = compute_current_size_relion(
                res_shell, grid_size,
                ave_Pmax=state.ave_Pmax,
                has_high_fsc_at_limit=has_high_fsc,
            )
            cs = quantize_current_size(raw_cs)

        cs = quantize_current_size(cs)
        current_sizes.append(cs)
        healpix_order_trajectory.append(state.healpix_order)

        logger.info(
            "=== RELION Iteration %d/%d: current_size=%d, "
            "healpix_order=%d, local_search=%s ===",
            iteration + 1, max_iter, cs,
            state.healpix_order, state.do_local_search,
        )

        # --- Angular step refinement: regenerate rotation grid if needed ---
        # When update_refinement_state incremented healpix_order, we need
        # a new rotation grid at the finer level.
        if state.healpix_order != current_healpix_order:
            logger.info(
                "Regenerating rotation grid: order %d -> %d",
                current_healpix_order, state.healpix_order,
            )
            current_rotations = get_rotation_grid_at_order(
                state.healpix_order, matrices=True,
            ).astype(np.float32)
            current_healpix_order = state.healpix_order
            current_nside_level = state.healpix_order

            # Regenerate translation grid based on updated parameters
            current_translations = jnp.array(
                get_translation_grid(
                    state.translation_range, state.translation_step,
                ).astype(np.float32)
            )
            logger.info(
                "New grid: %d rotations, %d translations "
                "(range=%.1f, step=%.1f)",
                current_rotations.shape[0], current_translations.shape[0],
                state.translation_range, state.translation_step,
            )

        # --- Local angular search with Gaussian prior weighting ---
        # RELION-style local search: restrict the rotation grid to
        # orientations near previous best assignments, AND add Gaussian
        # prior log-weights so orientations far from the previous best
        # are exponentially down-weighted in the E-step.
        #
        # Grid restriction (for speed): use get_local_rotation_grid to
        # select all grid rotations within sigma_cutoff * sigma_rot of
        # any image's previous best orientation.
        #
        # Prior weighting: for each selected rotation r, compute
        #   log_prior[r] = max_i(-d(R_prev[i], R_r)^2 / (2*sigma^2))
        # where the max is over all images (union approach).  This is
        # conservative -- the closest image's prior dominates.
        effective_rotations = current_rotations
        rotation_log_prior = None
        local_rot_indices = None  # mapping from local -> global rotation index
        use_local = (state.do_local_search
                     and previous_assignments[0] is not None
                     and iteration > 0)
        if use_local:
            n_trans_current = current_translations.shape[0]
            sigma_rot = state.sigma_rot
            sigma_psi = state.sigma_psi if state.sigma_psi > 0 else sigma_rot
            if sigma_rot <= 0:
                # Fallback: compute from effective angular step
                step_rad = np.deg2rad(
                    healpix_angular_step(state.healpix_order)
                    / (2 ** state.adaptive_oversampling)
                )
                sigma_rot = np.sqrt(2.0 * 2.0) * step_rad
                sigma_psi = sigma_rot

            # Gather UNIQUE per-image best rotation indices from both half-sets
            unique_rot_idx = set()
            for k in range(2):
                if previous_assignments[k] is not None:
                    rot_idx = previous_assignments[k] // n_trans_current
                    rot_idx = np.clip(rot_idx, 0, current_rotations.shape[0] - 1)
                    unique_rot_idx.update(rot_idx.tolist())
            unique_rot_idx = np.array(sorted(unique_rot_idx))

            # Fast HEALPix-based local search
            t0_local = time.time()
            selected_indices, rotation_log_prior = get_local_rotation_grid_fast(
                unique_rot_idx,
                sigma_rot,
                sigma_psi,
                state.healpix_order,
                sigma_cutoff=3.0,
            )
            dt_local = time.time() - t0_local

            # Cap the local grid to prevent it from growing to the full grid.
            # Keep the highest-prior rotations.  At order 4 with 5K images,
            # the uncapped grid can grow to 200K+ (86% of 295K).
            # A cap of ~20K gives ~20s/iter while maintaining quality.
            MAX_LOCAL_ROTATIONS = 20000
            if len(selected_indices) > MAX_LOCAL_ROTATIONS:
                # Keep the top-N by log_prior
                top_idx = np.argsort(rotation_log_prior)[-MAX_LOCAL_ROTATIONS:]
                selected_indices = selected_indices[top_idx]
                rotation_log_prior = rotation_log_prior[top_idx]
                # Re-sort by index for consistent ordering
                sort_order = np.argsort(selected_indices)
                selected_indices = selected_indices[sort_order]
                rotation_log_prior = rotation_log_prior[sort_order]

            if len(selected_indices) < current_rotations.shape[0]:
                effective_rotations = current_rotations[selected_indices]
                local_rot_indices = selected_indices

                logger.info(
                    "Local search (fast): %d / %d rotations in %.2f s "
                    "(sigma_rot=%.4f rad = %.2f deg, sigma_psi=%.4f rad, "
                    "log_prior range=[%.2f, %.2f])",
                    effective_rotations.shape[0],
                    current_rotations.shape[0],
                    dt_local,
                    sigma_rot, np.rad2deg(sigma_rot),
                    sigma_psi,
                    rotation_log_prior.min(), rotation_log_prior.max(),
                )
            else:
                rotation_log_prior = None
                logger.info(
                    "Local search (fast): all %d rotations selected in %.2f s "
                    "(sigma_rot=%.4f rad); using flat prior",
                    current_rotations.shape[0], dt_local, sigma_rot,
                )

        cs_for_engine = cs if cs < cryo.image_shape[0] else None

        # --- Run E+M on each half-set ---
        # Two modes: single-pass (adaptive_oversampling=0) or two-pass
        # coarse/fine (adaptive_oversampling>=1).
        iter_sig_counts = None
        use_adaptive = (state.adaptive_oversampling > 0
                        and effective_rotations.shape[0] > 16)

        # Track the rotation grids used for pose extraction.
        # When adaptive oversampling is active, ha_k indices refer to the
        # oversampled grid (from pass 2), not effective_rotations.
        pose_rotations = [None, None]  # rotations to use with ha for poses
        # Coarse-grid assignments for local search tracking (always indexed
        # into effective_rotations, even when adaptive oversampling is used).
        coarse_ha = [None, None]

        if use_adaptive:
            # --- TWO-PASS ADAPTIVE OVERSAMPLING (RELION parity) ---
            # Pass 1: coarse E-step at reduced resolution to find
            #         significant orientations.
            # Pass 2: oversampled E+M at full current_size for significant
            #         orientations only.

            # Compute coarse image size from angular step
            effective_step_deg = healpix_angular_step(current_healpix_order)
            pixel_size = cryo.voxel_size if cryo.voxel_size > 0 else 1.0
            coarse_size = compute_coarse_image_size(
                effective_step_deg, pixel_size, grid_size,
            )
            coarse_size = quantize_current_size(coarse_size)
            # Coarse size must be smaller than full current_size
            if cs_for_engine is not None and coarse_size >= cs:
                coarse_size = max(8, cs // 2)
                coarse_size = quantize_current_size(coarse_size)
            coarse_cs = coarse_size if coarse_size < grid_size else None

            logger.info(
                "Adaptive oversampling: pass 1 at coarse_size=%s, "
                "pass 2 at current_size=%s (oversampling=%d)",
                coarse_cs, cs_for_engine, state.adaptive_oversampling,
            )

        for k in range(2):
            if use_adaptive:
                # --- PASS 1: Coarse significance pruning ---
                safe_ibs, safe_rbs = _safe_batch_sizes(effective_rotations.shape[0])

                t_pass1 = time.time()
                sig_rot_any, n_sig_batch, ha_coarse = (
                    _compute_significance_batched(
                        experiment_datasets[k],
                        means[k],
                        noise_variance,
                        effective_rotations,
                        current_translations,
                        disc_type,
                        adaptive_fraction=0.999,
                        max_significants=500,
                        image_batch_size=safe_ibs,
                        rotation_block_size=safe_rbs,
                        current_size=coarse_cs,
                    )
                )
                n_sig_total = int(np.sum(sig_rot_any))
                dt_pass1 = time.time() - t_pass1

                logger.info(
                    "Pass 1 (half %d): %d / %d significant coarse "
                    "rotations in %.1fs (median n_sig/image=%d)",
                    k, n_sig_total, effective_rotations.shape[0],
                    dt_pass1, int(np.median(n_sig_batch)),
                )

                # --- Generate oversampled children ---
                # Use compute_pass2_stats which generates HEALPix children
                # of significant coarse rotations and runs a full E+M at
                # the fine grid.
                t_pass2 = time.time()
                Ft_y_k, Ft_ctf_k, ha_k, oversampled_rots = (
                    compute_pass2_stats(
                        experiment_datasets[k],
                        means[k],
                        mean_variance,
                        noise_variance,
                        effective_rotations,
                        current_translations,
                        sig_rot_any,
                        current_nside_level,
                        disc_type,
                        oversampling_order=state.adaptive_oversampling,
                        current_size=cs_for_engine,
                        image_batch_size=image_batch_size,
                    )
                )
                dt_pass2 = time.time() - t_pass2

                if Ft_y_k is None:
                    # Pass 2 was skipped (too many significant rotations);
                    # fall back to single-pass at full current_size.
                    logger.warning(
                        "Pass 2 skipped for half %d; falling back to "
                        "single-pass E+M.", k,
                    )
                    _, ha_k, Ft_y_k, Ft_ctf_k = run_em_v2(
                        experiment_datasets[k],
                        means[k],
                        mean_variance,
                        noise_variance,
                        effective_rotations,
                        current_translations,
                        disc_type,
                        image_batch_size=image_batch_size,
                        rotation_block_size=rotation_block_size,
                        current_size=cs_for_engine,
                        rotation_log_prior=rotation_log_prior,
                    )
                    pose_rotations[k] = effective_rotations
                    coarse_ha[k] = ha_k  # fallback: same grid
                else:
                    n_oversampled = (oversampled_rots.shape[0]
                                     if oversampled_rots is not None else 0)
                    logger.info(
                        "Pass 2 (half %d): %d oversampled rotations, "
                        "%.1fs",
                        k, n_oversampled, dt_pass2,
                    )
                    # ha_k indices are into oversampled_rots, not
                    # effective_rotations.  Track this for pose extraction.
                    pose_rotations[k] = oversampled_rots

                # Store coarse-grid assignment from pass 1 for local search.
                coarse_ha[k] = ha_coarse

                if iter_sig_counts is None:
                    iter_sig_counts = n_sig_batch
                else:
                    iter_sig_counts = np.concatenate([
                        iter_sig_counts, n_sig_batch
                    ])

            else:
                # --- SINGLE-PASS E+M (no adaptive oversampling) ---
                safe_ibs, safe_rbs = _safe_batch_sizes(effective_rotations.shape[0])
                _, ha_k, Ft_y_k, Ft_ctf_k = run_em_v2(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance,
                    effective_rotations,
                    current_translations,
                    disc_type,
                    image_batch_size=safe_ibs,
                    rotation_block_size=safe_rbs,
                    current_size=cs_for_engine,
                    rotation_log_prior=rotation_log_prior,
                )
                pose_rotations[k] = effective_rotations
                coarse_ha[k] = ha_k  # same grid, no oversampling

            # Reconstruct the regularized mean with padding_factor=2.
            # run_em_v2 uses padding_factor=1 internally; we override here.
            Ft_ctf_k_padded = relion_functions.zero_pad_fourier_volume(
                Ft_ctf_k, volume_shape, PADDING_FACTOR,
            )
            Ft_y_k_padded = relion_functions.zero_pad_fourier_volume(
                Ft_y_k, volume_shape, PADDING_FACTOR,
            )
            means[k] = relion_functions.post_process_from_filter_v2(
                Ft_ctf_k_padded, Ft_y_k_padded,
                volume_shape, PADDING_FACTOR,
                tau=mean_variance,
                kernel="triangular",
                use_spherical_mask=True, grid_correct=True,
                gridding_correct="square",
            ).reshape(-1)
            hard_assignments[k] = ha_k

            if k == 0:
                Ft_y_0, Ft_ctf_0 = Ft_y_k, Ft_ctf_k
            else:
                Ft_y_1, Ft_ctf_1 = Ft_y_k, Ft_ctf_k

        significant_counts.append(iter_sig_counts)

        # --- Combined Fourier weights for data_vs_prior at next iteration ---
        # RELION uses the combined (both half-sets) CTF^2 weight for
        # data_vs_prior.  Store for use at the start of the next iteration.
        Ft_ctf_combined = Ft_ctf_0 + Ft_ctf_1

        # --- Compute unregularized half-maps for FSC and prior ---
        # RELION uses padding_factor=2 by default: the 3D Fourier grid is
        # (2*N)^3 to reduce interpolation artifacts.  We zero-pad the
        # native-size Ft_ctf/Ft_y into the padded grid, then
        # post_process_from_filter_v2 does iDFT on the padded grid and
        # crops back to native size in real space.
        Ft_ctf_0_padded = relion_functions.zero_pad_fourier_volume(
            Ft_ctf_0, volume_shape, PADDING_FACTOR,
        )
        Ft_y_0_padded = relion_functions.zero_pad_fourier_volume(
            Ft_y_0, volume_shape, PADDING_FACTOR,
        )
        Ft_ctf_1_padded = relion_functions.zero_pad_fourier_volume(
            Ft_ctf_1, volume_shape, PADDING_FACTOR,
        )
        Ft_y_1_padded = relion_functions.zero_pad_fourier_volume(
            Ft_y_1, volume_shape, PADDING_FACTOR,
        )
        unreg_means = [
            relion_functions.post_process_from_filter_v2(
                Ft_ctf_0_padded, Ft_y_0_padded,
                volume_shape, PADDING_FACTOR,
                tau=None, kernel="triangular",
                use_spherical_mask=True, grid_correct=True,
                gridding_correct="square",
            ),
            relion_functions.post_process_from_filter_v2(
                Ft_ctf_1_padded, Ft_y_1_padded,
                volume_shape, PADDING_FACTOR,
                tau=None, kernel="triangular",
                use_spherical_mask=True, grid_correct=True,
                gridding_correct="square",
            ),
        ]

        # --- Compute FSC between half-maps ---
        fsc = regularization.get_fsc_gpu(
            unreg_means[0], unreg_means[1], volume_shape,
        )
        fsc_history.append(fsc)

        # --- Save intermediate volumes if requested ---
        if save_intermediates_dir is not None:
            import os
            os.makedirs(save_intermediates_dir, exist_ok=True)
            import mrcfile
            for k_half in range(2):
                vol_real = np.real(
                    np.fft.ifftn(
                        np.fft.ifftshift(
                            np.asarray(means[k_half]).reshape(volume_shape)
                        )
                    )
                ).astype(np.float32)
                mrc_path = os.path.join(
                    save_intermediates_dir,
                    f"it{iteration:03d}_half{k_half+1}_reg.mrc",
                )
                with mrcfile.new(mrc_path, overwrite=True) as mrc:
                    mrc.set_data(vol_real)
                # Also save unregularized half-map
                vol_unreg = np.real(
                    np.fft.ifftn(
                        np.fft.ifftshift(
                            np.asarray(unreg_means[k_half]).reshape(volume_shape)
                        )
                    )
                ).astype(np.float32)
                mrc_unreg_path = os.path.join(
                    save_intermediates_dir,
                    f"it{iteration:03d}_half{k_half+1}_unreg.mrc",
                )
                with mrcfile.new(mrc_unreg_path, overwrite=True) as mrc:
                    mrc.set_data(vol_unreg)
            # Save FSC and noise/tau2 per iteration
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_fsc.npy"),
                np.asarray(fsc),
            )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_noise.npy"),
                np.asarray(noise_variance),
            )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_tau2.npy"),
                np.asarray(mean_variance),
            )
            # Save hard assignments for angular error analysis
            for k_half in range(2):
                if hard_assignments[k_half] is not None:
                    np.save(
                        os.path.join(
                            save_intermediates_dir,
                            f"it{iteration:03d}_ha_half{k_half+1}.npy",
                        ),
                        hard_assignments[k_half],
                    )
            # Save per-iteration metadata
            iter_meta = {
                "iteration": iteration,
                "current_size": int(cs),
                "n_rotations": int(effective_rotations.shape[0]),
                "n_translations": int(current_translations.shape[0]),
                "healpix_order": int(state.healpix_order),
                "local_search": bool(use_local),
                "sigma_rot": float(state.sigma_rot),
            }
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_meta.npy"),
                iter_meta,
            )
            # Save the effective rotation grid for angular error computation
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_rotations.npy"),
                np.asarray(effective_rotations),
            )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_translations.npy"),
                np.asarray(current_translations),
            )
            logger.info(
                "Saved intermediate volumes to %s (iteration %d)",
                save_intermediates_dir, iteration,
            )

        # --- Resolution from data_vs_prior (RELION-style, for convergence) ---
        # Use data_vs_prior criterion, not FSC < 0.143, because with
        # [cryo, cryo] half-sets the FSC is always 1.0.
        # data_vs_prior is computed at the START of the next iteration,
        # but we also compute it here for convergence tracking.
        tau2_radial_iter = regularization.average_over_shells(
            mean_variance.real, volume_shape,
        )
        dvp_iter = compute_data_vs_prior(
            Ft_ctf_combined, tau2_radial_iter, volume_shape,
            padding_factor=PADDING_FACTOR,
        )
        dvp_res_shell = resolution_from_data_vs_prior(dvp_iter)
        pixel_res = float(dvp_res_shell)
        pixel_resolutions.append(pixel_res)

        # --- Compute ave_Pmax ---
        # TODO: extract Pmax from run_em_v2 directly instead of a separate
        # E-step pass (_extract_max_posterior_per_image is too expensive).
        # For now, estimate from hard assignment counts: if assignments are
        # concentrated, Pmax is high.  Use 0.5 as a reasonable default that
        # enables RELION's aggressive current_size growth.
        ave_pmax = 0.5
        ave_Pmax_trajectory.append(ave_pmax)

        # --- Track per-image best assignments for convergence detection ---
        # Combine both half-sets' assignments into a single array for
        # update_refinement_state.  Use coarse_ha (indexed into
        # effective_rotations) for consistent convergence tracking.
        current_combined_ha = coarse_ha[0]
        previous_combined_ha = previous_assignments[0]

        # --- Update prior (RELION-style tau^2 from FSC) ---
        mean_signal_variance, _, _ = regularization.compute_relion_prior(
            experiment_datasets, noise_variance,
            unreg_means[0], unreg_means[1], 100,
        )
        mean_variance = mean_signal_variance

        # --- Update poses and noise ---
        for k in range(2):
            # When adaptive oversampling is used, ha indices refer to the
            # oversampled grid stored in pose_rotations[k], not effective_rotations.
            best_rots, best_trans = hard_assignment_idx_to_pose(
                hard_assignments[k], pose_rotations[k], current_translations,
            )
            experiment_datasets[k].update_poses(best_rots, best_trans)

        # Estimate noise from BOTH half-sets, ALL images (not just 1000).
        # Still hard-assignment based, but less biased than a small subset.
        noise_estimates = []
        for k in range(2):
            n_k = experiment_datasets[k].n_units
            noise_k = noise.estimate_noise_level_no_masks(
                experiment_datasets[k],
                np.arange(n_k),
                means[k],
                100,
                disc_type=disc_type,
            )
            noise_estimates.append(noise_k)
        noise_from_res = (noise_estimates[0] + noise_estimates[1]) / 2
        noise_variance = noise.make_radial_noise(noise_from_res, cryo.image_shape)

        # --- Update convergence state ---
        # This checks assignment changes, resolution stalls, and may trigger
        # angular step refinement or convergence.
        n_rot_current = effective_rotations.shape[0]
        n_trans_current = current_translations.shape[0]

        state = update_refinement_state(
            state,
            current_assignments=current_combined_ha,
            previous_assignments=previous_combined_ha,
            n_rotations=n_rot_current,
            n_translations=n_trans_current,
            translations=np.asarray(current_translations),
            new_resolution=pixel_res,
            max_posterior_per_image=np.full(cryo.n_units, ave_pmax, dtype=np.float32),
        )

        # Track frac_changed for local search fallback
        from recovar.em.dense_single_volume.convergence import compute_assignment_changes
        frac_changed = compute_assignment_changes(
            current_combined_ha, previous_combined_ha,
            n_rot_current, n_trans_current,
            current_healpix_order,
        )
        state._last_frac_changed = frac_changed

        # Save assignments for next iteration's change tracking.
        # Use coarse_ha (indexed into effective_rotations/current_rotations)
        # so that local search and convergence detection work correctly
        # regardless of whether adaptive oversampling was used.
        previous_assignments = [
            ha.copy() if ha is not None else None
            for ha in coarse_ha
        ]

        # --- Timing ---
        elapsed = time.time() - t0
        wall_times.append(elapsed)

        res_angstrom = pixel_res / cryo.voxel_size if cryo.voxel_size > 0 else pixel_res
        logger.info(
            "RELION Iteration %d: current_size=%d, pixel_res=%.1f, "
            "res=%.2f A, ave_Pmax=%.4f, healpix_order=%d, "
            "converged=%s, time=%.1fs",
            iteration + 1, cs, pixel_res, res_angstrom,
            ave_pmax, state.healpix_order,
            state.has_converged, elapsed,
        )

        if state.has_converged:
            # RELION does one final iteration at full resolution with
            # joined half-sets.  For now, we just log and break.
            logger.info(
                "Convergence reached at iteration %d. "
                "Final resolution: %.2f A (pixel_res=%.1f)",
                iteration + 1, res_angstrom, pixel_res,
            )
            break

        iteration += 1

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
        # RELION-mode specific outputs
        "convergence_state": state,
        "data_vs_prior_trajectory": data_vs_prior_trajectory,
        "healpix_order_trajectory": healpix_order_trajectory,
        "ave_Pmax_trajectory": ave_Pmax_trajectory,
    }
