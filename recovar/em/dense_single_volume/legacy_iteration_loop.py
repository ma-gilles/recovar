"""Legacy FSC-driven refinement loop.

This module contains the pre-RELION refinement path extracted from
``iteration_loop.py`` so the active RELION-parity path can remain easier to
read. Behavior should remain identical to the previous inline implementation.
"""

import logging
import time

import jax.numpy as jnp
import numpy as np

from recovar.em.core import hard_assignment_idx_to_pose
from recovar.em.dense_single_volume.em_engine import run_em
from recovar.em.dense_single_volume.helpers.oversampling import compute_pass2_stats
from recovar.em.dense_single_volume.helpers.resolution import (
    fsc_to_current_size,
    shell_index_to_resolution_angstrom,
)
from recovar.em.dense_single_volume.helpers.significance import _compute_significance_batched
from recovar.em.dense_single_volume.helpers.fourier_window import quantize_current_size
from recovar.em.sampling import get_oversampled_translation_grid

logger = logging.getLogger(__name__)

### TODO: I THINK THIS WHOLE FILE CAN GO.

def _run_legacy_iteration_loop(
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
    relion_current_sizes,
    init_current_size,
    fsc_threshold,
    adaptive_oversampling,
    adaptive_fraction,
    max_significants,
    nside_level,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
):
    """Run the original recovar FSC-driven refinement loop."""
    from recovar.reconstruction import noise, regularization, relion_functions

    if adaptive_oversampling > 0 and nside_level is None:
        raise ValueError("nside_level must be provided when adaptive_oversampling > 0")

    cryo = experiment_datasets[0]
    volume_shape = cryo.volume_shape

    means = [jnp.array(init_volume), jnp.array(init_volume)]
    noise_variance = jnp.array(init_noise_variance)
    mean_variance = jnp.array(init_mean_variance)

    current_sizes = []
    fsc_history = []
    pixel_resolutions = []
    wall_times = []
    hard_assignments = [None, None]
    significant_counts = []

    for iteration in range(max_iter):
        t0 = time.time()

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
                means[0],
                means[1],
                volume_shape,
            )
            raw_cs = fsc_to_current_size(fsc_prev, threshold=fsc_threshold)
            cs = quantize_current_size(raw_cs, ori_size=cryo.image_shape[0])

        cs = quantize_current_size(cs, ori_size=cryo.image_shape[0])
        current_sizes.append(cs)

        logger.info(
            "=== Iteration %d/%d: current_size=%d ===",
            iteration + 1,
            max_iter,
            cs,
        )

        use_adaptive = adaptive_oversampling > 0
        cs_for_engine = cs if cs < cryo.image_shape[0] else None

        iter_sig_counts = None
        pose_rotations = [np.asarray(rotations), np.asarray(rotations)]
        pose_translations = [
            np.asarray(translations, dtype=np.float32),
            np.asarray(translations, dtype=np.float32),
        ]

        for k in range(2):
            if not use_adaptive:
                new_mean_k, ha_k, Ft_y_k, Ft_ctf_k = run_em(
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
                    sparse_pass2=False,
                    disable_adjoint_y=disable_adjoint_y,
                    disable_adjoint_ctf=disable_adjoint_ctf,
                )
            else:
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
                    k,
                    int(n_sig_np.min()),
                    int(np.median(n_sig_np)),
                    int(n_sig_np.max()),
                    float(n_sig_np.mean()),
                    int(np.sum(sig_rot_any)),
                    rotations.shape[0],
                )

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
                    logger.info(
                        "Half %d: pass 2 skipped, running pass-1-only E+M",
                        k,
                    )
                    new_mean_k, ha_k, Ft_y_k, Ft_ctf_k = run_em(
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
                        sparse_pass2=False,
                        disable_adjoint_y=disable_adjoint_y,
                        disable_adjoint_ctf=disable_adjoint_ctf,
                    )
                    oversampled_rots = None
                    pose_rotations[k] = np.asarray(rotations)
                    pose_translations[k] = np.asarray(translations, dtype=np.float32)
                else:
                    new_mean_k = relion_functions.post_process_from_filter(
                        experiment_datasets[k],
                        Ft_ctf_k,
                        Ft_y_k,
                        tau=mean_variance,
                        disc_type=disc_type,
                    ).reshape(-1)
                    pose_rotations[k] = np.asarray(oversampled_rots)
                    translation_vals = np.unique(np.asarray(translations, dtype=np.float32))
                    translation_diffs = np.diff(np.sort(translation_vals))
                    translation_diffs = translation_diffs[translation_diffs > 1e-6]
                    translation_step = float(translation_diffs.min()) if translation_diffs.size else 1.0
                    oversampled_translations, _ = get_oversampled_translation_grid(
                        np.asarray(translations, dtype=np.float32),
                        translation_step,
                        oversampling_order=adaptive_oversampling,
                    )
                    pose_translations[k] = np.asarray(
                        oversampled_translations,
                        dtype=np.float32,
                    )

            means[k] = new_mean_k
            hard_assignments[k] = ha_k

            if k == 0:
                Ft_y_0, Ft_ctf_0 = Ft_y_k, Ft_ctf_k
            else:
                Ft_y_1, Ft_ctf_1 = Ft_y_k, Ft_ctf_k

        significant_counts.append(iter_sig_counts)

        unreg_means = [
            relion_functions.post_process_from_filter(
                cryo,
                Ft_ctf_0,
                Ft_y_0,
                tau=None,
                disc_type=disc_type,
            ),
            relion_functions.post_process_from_filter(
                cryo,
                Ft_ctf_1,
                Ft_y_1,
                tau=None,
                disc_type=disc_type,
            ),
        ]

        fsc = regularization.get_fsc_gpu(
            unreg_means[0],
            unreg_means[1],
            volume_shape,
        )
        fsc_history.append(fsc)

        from recovar.heterogeneity.locres import find_fsc_resol

        pixel_res = float(find_fsc_resol(fsc, threshold=fsc_threshold))
        pixel_resolutions.append(pixel_res)

        mean_signal_variance, _, _ = regularization.compute_relion_prior(
            experiment_datasets,
            noise_variance,
            unreg_means[0],
            unreg_means[1],
            100,
        )
        mean_variance = mean_signal_variance

        for k in range(2):
            best_rots, best_trans = hard_assignment_idx_to_pose(
                hard_assignments[k],
                pose_rotations[k],
                pose_translations[k],
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

        elapsed = time.time() - t0
        wall_times.append(elapsed)

        res_angstrom = shell_index_to_resolution_angstrom(
            pixel_res,
            cryo.image_shape[0],
            cryo.voxel_size,
        )
        logger.info(
            "Iteration %d: current_size=%d, pixel_res=%.1f, res=%.2f A, time=%.1fs",
            iteration + 1,
            cs,
            pixel_res,
            res_angstrom,
            elapsed,
        )

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
