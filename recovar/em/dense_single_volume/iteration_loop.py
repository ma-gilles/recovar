"""Core refinement loop for dense single-volume EM.

This file contains the three core algorithm functions:
- ``refine_single_volume`` — public entry point
- ``_run_relion_iteration_loop`` — RELION-parity iteration loop
- ``_run_local_search_iteration`` — exact local angular search

All supporting helpers live in ``helpers/``.
See ``docs/math/relion_refinement_algorithm.md`` for the full algorithm map.
"""

import logging
import os
import time

import jax.numpy as jnp
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils
from recovar.em.core import hard_assignment_idx_to_pose
from recovar.em.dense_single_volume import parity_dump as _parity_dump
from recovar.em.dense_single_volume.em_engine import run_em
from recovar.em.dense_single_volume.helpers.convergence import (
    LOCAL_SEARCH_HEALPIX_ORDER,
    RefinementState,
    calculate_expected_angular_errors,
    healpix_angular_step,
    update_refinement_state,
)
from recovar.em.dense_single_volume.helpers.fourier_window import quantize_current_size
from recovar.em.dense_single_volume.helpers.local_search import (
    _local_search_engine_rotation_block_size,
    _pad_local_search_rotations,
    _partition_local_search_groups,
)
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    collapse_rotation_posterior_to_direction_prior,
    infer_direction_prior_healpix_order,
    make_relion_direction_log_prior,
    make_relion_translation_log_prior,
    relion_translation_search_base,
    remap_direction_prior_to_healpix_order,
)
from recovar.em.dense_single_volume.helpers.oversampling import (
    compute_pass2_stats,
    compute_pass2_stats_sparse,
)
from recovar.em.dense_single_volume.helpers.resolution import (
    ADAPTIVE_PASS2_MAX_SIGNIFICANT_FRACTION,
    _bootstrap_current_size_relion,
    bootstrap_current_size_from_ini_high_relion,
    clamp_relion_coarse_image_size,
    compute_coarse_image_size,
    shell_index_to_resolution_angstrom,
    should_skip_adaptive_pass2,
)
from recovar.em.dense_single_volume.helpers.types import RelionStats
from recovar.em.dense_single_volume.legacy_iteration_loop import _run_legacy_iteration_loop
from recovar.em.dense_single_volume.local_em_engine import run_local_em_exact
from recovar.em.dense_single_volume.local_layout import build_local_hypothesis_layout
from recovar.em.sampling import (
    advance_relion_perturbation,
    apply_relion_rotation_perturbation,
    apply_relion_translation_perturbation,
    build_local_search_grid_metadata,
    get_oversampled_translation_grid,
    get_relion_rotation_grid,
    get_relion_rotation_grid_eulers,
    get_translation_grid,
    read_relion_direction_prior,
    read_relion_model_metadata,
    read_relion_sampling_metadata,
    relion_angular_sampling_deg,
    rotation_grid_size,
)
from recovar.reconstruction.regularization import (
    compute_current_size_relion,
    fsc_to_relion_ssnr,
    resolution_from_data_vs_prior,
    update_relion_growth_state_from_fsc,
)

logger = logging.getLogger(__name__)

# TRACKED TODOs: RELION_LOCAL_ENGINE
# TODO(RELION_LOCAL_ENGINE/T001): grouped-union local search is the wrong active abstraction
# TODO(RELION_LOCAL_ENGINE/T002): active RELION local path must use per-image local hypotheses
# TODO(RELION_LOCAL_ENGINE/T003): local path should not depend on dense shared-grid engine contracts
# TODO(RELION_LOCAL_ENGINE/T004): parity hacks should move inward, out of outer-loop control flow
# See docs/relion_local_engine_refactor.md


def _replay_control_model_iteration(init_relion_iteration: int, loop_iteration: int) -> int:
    """Return the RELION model.star index whose control state governs this replay step."""
    return int(init_relion_iteration) + int(loop_iteration) + 1


# ---- Extracted helpers live in helpers/ ----
# local_search.py: _partition_local_search_groups, _pad_local_search_rotations, etc.
# orientation_priors.py: make_relion_translation_log_prior, make_relion_direction_log_prior, etc.
# resolution.py: shell_index_to_resolution_angstrom, compute_coarse_image_size, fsc_to_current_size, etc.
# convergence.py: RefinementState, check_convergence, update_refinement_state, etc.
# oversampling.py: find_significant_rotations, compute_pass2_stats, etc.


def _reconstruct_volume_eager(
    Ft_ctf,
    Ft_y,
    vol_shape,
    padding_factor,
    tau,
    tau2_fudge,
    projection_padding_factor,
    use_spherical_mask=True,
    grid_correct=True,
):
    """Eager RELION-style reconstruction from full or half Fourier accumulators.

    This keeps the reconstruction boundary out of a single monolithic JIT while
    letting the local exact path keep its accumulators in packed half-volume
    layout until the final iDFT boundary.
    """
    from recovar.reconstruction import relion_functions

    return relion_functions.post_process_from_filter_v2(
        Ft_ctf,
        Ft_y,
        vol_shape,
        padding_factor,
        tau=tau,
        kernel="triangular",
        use_spherical_mask=use_spherical_mask,
        grid_correct=grid_correct,
        gridding_correct="radial",
        kernel_width=1,
        tau2_fudge=tau2_fudge,
        gridding_padding_factor=projection_padding_factor,
    )


def _apply_relion_initial_lowpass_filter(
    volume_ft_flat, volume_shape, voxel_size, ini_high_angstrom, filter_edgewidth=5
):
    """Apply RELION's ``initialLowPassFilterReferences`` to a full Fourier volume."""
    if ini_high_angstrom is None or float(ini_high_angstrom) <= 0.0:
        return volume_ft_flat
    from recovar.heterogeneity import locres

    filtered = locres.low_pass_filter_map(
        jnp.asarray(volume_ft_flat).reshape(volume_shape),
        volume_shape[0],
        float(ini_high_angstrom),
        float(voxel_size),
        int(filter_edgewidth),
        do_highpass_instead=False,
        volume_shape=volume_shape,
    )
    return filtered.reshape(-1)


def _align_fourier_volume_sign_to_reference(volume_ft_flat, reference_ft_flat, volume_shape):
    """Keep reconstructed volumes on the same real-space sign branch as the reference."""
    if reference_ft_flat is None:
        return volume_ft_flat, False
    vol_real = np.asarray(
        fourier_transform_utils.get_idft3(jnp.asarray(volume_ft_flat).reshape(volume_shape)),
        dtype=np.float64,
    ).reshape(-1)
    ref_real = np.asarray(
        fourier_transform_utils.get_idft3(jnp.asarray(reference_ft_flat).reshape(volume_shape)),
        dtype=np.float64,
    ).reshape(-1)
    vol_centered = vol_real - float(np.mean(vol_real))
    ref_centered = ref_real - float(np.mean(ref_real))
    overlap = float(np.dot(ref_centered, vol_centered))
    if overlap < 0.0:
        return -volume_ft_flat, True
    return volume_ft_flat, False


def _run_local_search_iteration(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    prior_rotations,
    rotation_grid_rotations,
    rotation_grid_eulers,
    healpix_order,
    sigma_rot,
    sigma_psi,
    translations,
    prior_translations,
    sigma_offset_angstrom,
    offset_range_pixels,
    disc_type,
    image_batch_size,
    rotation_block_size,
    current_size,
    *,
    accumulate_noise=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    use_float64_scoring=False,
    use_float64_projections=False,
    do_gridding_correction=False,
    square_window=False,
    half_spectrum_scoring=False,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    score_with_masked_images=True,
    return_profile=False,
    sparse_pass2=True,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    adaptive_fraction=0.999,
    max_significants=-1,
    local_engine="grouped_union",
    translation_prior_mode="perturbed",
    translation_prior_reference_translations=None,
):
    """Run exact local search on the fine HEALPix grid.

    Each image carries its own exact prior orientation from the previous
    iteration. ``prior_rotations`` may be either RELION Euler angles
    ``(rot, tilt, psi)`` or rotation matrices. Images are processed in
    chunks; each chunk is evaluated either by the legacy grouped-union path
    or by the new per-image exact local engine.

    TODO(local-engine-debt): This shared-union batching scheme is only a
    stopgap for the active RELION-mode local path. It forces multiple images
    onto one padded union rotation set and then masks invalid image-rotation
    pairs with per-image priors. We should replace it with a dedicated local
    engine that evaluates each image on its own neighborhood. When that
    refactor happens, keep the translation-side inner-product/GEMM opportunity
    in mind as an optimization target, but do not let it keep us trapped in
    the current union-based local abstraction.
    """
    if local_engine not in ("grouped_union", "exact_v1", "exact_v2"):
        raise ValueError(
            f"Unknown local_engine={local_engine!r}; expected 'grouped_union', 'exact_v1', or 'exact_v2'",
        )
    if local_engine == "grouped_union":
        return _run_local_search_iteration_grouped_union(
            experiment_dataset,
            mean,
            mean_variance,
            noise_variance,
            prior_rotations,
            rotation_grid_rotations,
            rotation_grid_eulers,
            healpix_order,
            sigma_rot,
            sigma_psi,
            translations,
            prior_translations,
            sigma_offset_angstrom,
            offset_range_pixels,
            disc_type,
            image_batch_size,
            rotation_block_size,
            current_size,
            accumulate_noise=accumulate_noise,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            use_float64_scoring=use_float64_scoring,
            use_float64_projections=use_float64_projections,
            do_gridding_correction=do_gridding_correction,
            square_window=square_window,
            half_spectrum_scoring=half_spectrum_scoring,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            score_with_masked_images=score_with_masked_images,
            return_profile=return_profile,
            sparse_pass2=sparse_pass2,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
            translation_prior_mode=translation_prior_mode,
            translation_prior_reference_translations=translation_prior_reference_translations,
        )
    if local_engine == "exact_v2":
        logger.warning("local_engine='exact_v2' is not implemented yet; using exact_v1")
    return _run_local_search_iteration_exact_v1(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        prior_rotations,
        rotation_grid_rotations,
        rotation_grid_eulers,
        healpix_order,
        sigma_rot,
        sigma_psi,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        offset_range_pixels,
        disc_type,
        image_batch_size,
        rotation_block_size,
        current_size,
        accumulate_noise=accumulate_noise,
        projection_padding_factor=projection_padding_factor,
        reconstruction_padding_factor=reconstruction_padding_factor,
        use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
        do_gridding_correction=do_gridding_correction,
        square_window=square_window,
        half_spectrum_scoring=half_spectrum_scoring,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        score_with_masked_images=score_with_masked_images,
        return_profile=return_profile,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        translation_prior_mode=translation_prior_mode,
        translation_prior_reference_translations=translation_prior_reference_translations,
    )


def _run_local_search_iteration_grouped_union(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    prior_rotations,
    rotation_grid_rotations,
    rotation_grid_eulers,
    healpix_order,
    sigma_rot,
    sigma_psi,
    translations,
    prior_translations,
    sigma_offset_angstrom,
    offset_range_pixels,
    disc_type,
    image_batch_size,
    rotation_block_size,
    current_size,
    *,
    accumulate_noise=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    use_float64_scoring=False,
    use_float64_projections=False,
    do_gridding_correction=False,
    square_window=False,
    half_spectrum_scoring=False,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    score_with_masked_images=True,
    return_profile=False,
    sparse_pass2=True,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    translation_prior_mode="perturbed",
    translation_prior_reference_translations=None,
):
    """Legacy grouped-union local engine kept for comparison and fallback."""
    rotation_block_size = _local_search_engine_rotation_block_size(rotation_block_size)
    prior_rotations = np.asarray(prior_rotations, dtype=np.float32)
    if prior_rotations.ndim == 3:
        n_prior = prior_rotations.shape[0]
    elif prior_rotations.ndim == 2 and prior_rotations.shape[1] == 3:
        n_prior = prior_rotations.shape[0]
    else:
        raise ValueError(f"prior_rotations must have shape (n,3,3) or (n,3), got {prior_rotations.shape}")
    # The local-search translation prior is evaluated on the relative delta
    # grid after pre-centering by the previous absolute offset, so the prior
    # center must already be in "delta coordinates" (typically
    # -old_offset).
    if prior_translations is None:
        prior_translations = np.zeros(
            (n_prior, np.asarray(translations).shape[1]),
            dtype=np.float32,
        )
    else:
        prior_translations = np.asarray(prior_translations, dtype=np.float32).reshape(
            -1,
            np.asarray(translations).shape[1],
        )
    n_images = experiment_dataset.n_units
    n_trans = int(np.asarray(translations).shape[0])
    rotation_grid_rotations = np.asarray(rotation_grid_rotations, dtype=np.float32).reshape(-1, 3, 3)
    if rotation_grid_rotations.shape[0] != rotation_grid_size(healpix_order):
        raise ValueError(
            f"rotation_grid_rotations must have shape ({rotation_grid_size(healpix_order)}, 3, 3), "
            f"got {rotation_grid_rotations.shape}",
        )
    if rotation_grid_eulers is not None:
        rotation_grid_eulers = np.asarray(rotation_grid_eulers, dtype=np.float32).reshape(-1, 3)
    if rotation_grid_eulers is not None and rotation_grid_eulers.shape[0] != rotation_grid_rotations.shape[0]:
        raise ValueError(
            f"rotation_grid_eulers must match rotation_grid_rotations, got "
            f"{rotation_grid_eulers.shape} vs {rotation_grid_rotations.shape}",
        )
    active_offset_range = (
        float(offset_range_pixels)
        if offset_range_pixels is not None
        else float(np.max(np.linalg.norm(np.asarray(translations, dtype=np.float32), axis=1)))
    )
    volume_size = experiment_dataset.volume_size
    recon_vol_size = volume_size * reconstruction_padding_factor**3

    Ft_y_total = jnp.zeros(recon_vol_size, dtype=experiment_dataset.dtype)
    Ft_ctf_total = jnp.zeros(recon_vol_size, dtype=experiment_dataset.dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    log_evidence = np.empty(n_images, dtype=np.float32)
    best_log_score = np.empty(n_images, dtype=np.float32)
    max_posterior = np.empty(n_images, dtype=np.float32)
    rotation_posterior_sums = np.zeros(
        rotation_grid_size(healpix_order),
        dtype=np.float64,
    )

    # Noise accumulation across chunks (RELION-parity for the noise update).
    n_shells_local = experiment_dataset.image_shape[0] // 2 + 1
    accum_noise_wsum = np.zeros(n_shells_local, dtype=np.float64) if accumulate_noise else None
    accum_img_power = np.zeros(n_shells_local, dtype=np.float64) if accumulate_noise else None
    accum_sumw = 0.0

    total_local_rotations = 0
    max_local_rotations = 0
    chunk_sizes = []
    n_chunks = 0
    metadata_build_time = 0.0
    selector_time = 0.0
    translation_prior_time = 0.0
    em_time = 0.0
    local_setup_time = 0.0
    run_em_wall_time = 0.0
    profile_merge_time = 0.0
    postprocess_time = 0.0
    total_bucket_rotations = 0
    max_bucket_rotations = 0
    chunk_local_rotations = []
    chunk_padded_rotations = []
    chunk_valid_pairs = []
    chunk_union_pairs = []
    chunk_padded_pairs = []
    chunk_nonzero_posterior_rows = []
    sum_union_rows = 0
    sum_padded_rows = 0
    sum_nonzero_posterior_rows = 0
    sum_union_row_pixels = 0
    sum_padded_row_pixels = 0
    total_adjoint_time = 0.0
    seen_global_rotations = np.zeros(rotation_grid_size(healpix_order), dtype=bool)
    seen_nonzero_global_rotations = np.zeros(rotation_grid_size(healpix_order), dtype=bool)
    em_phase_totals = None
    # The local-search selector must operate on the actual trial grid that will
    # be scored for this iteration. When the caller has already applied
    # SamplingPerturbation to `rotation_grid_eulers`, using canonical unperturbed
    # metadata lets the selected support drift away from the true per-image
    # neighborhood on the perturbed grid.
    metadata_t0 = time.time()
    local_grid_metadata = build_local_search_grid_metadata(
        healpix_order,
        grid_eulers=rotation_grid_eulers,
        grid_rotations=rotation_grid_rotations,
    )
    metadata_build_time = time.time() - metadata_t0

    # TODO: IS THIS ALL REALLY A GOOD IDEA? I THINK IT WOULD PROBABLY BE FASTER TO DO A NAIVE THING
    ## DONT GROUP, JUST COMPUTE.
    ## HOW MUCH TIME IS SPENT ON THIS PARTITIONING? HOW MANY EXTRA COMPARISONS ARE WE DOING?
    ## I THINK IN THIS BRANCH, WE MAY HAVE TO JUST ABANDON THE GEMM BASED, WHICH IS OK.
    ## OR IS THIS JUST GROUPING TO BATCH ON GPU ? I DON'T UNDERSTAND. ALSO SHOULDNT ALL IMAGES HAVE SOME BATCH SIZE ANYWAY?
    selector_t0 = time.time()
    grouped_local_search = _partition_local_search_groups(
        prior_rotations,
        sigma_rot,
        sigma_psi,
        healpix_order,
        image_batch_size,
        rotation_block_size,
        local_grid_metadata,
    )
    selector_time += time.time() - selector_t0

    for group_image_indices, local_indices, local_log_prior in grouped_local_search:
        n_chunks += 1
        chunk_sizes.append(len(group_image_indices))
        local_rotations = rotation_grid_rotations[local_indices]
        # C1 (RELION-parity): use the explicit sigma_offset_angstrom from the
        # caller (which is the data-driven value updated each iter) without
        # the legacy `range/3` override (offset_range_pixels=None). The
        # translation grid is still bounded by `active_offset_range` in the
        # engine's score computation.
        translation_prior_t0 = time.time()
        prior_reference_translations = (
            np.asarray(translation_prior_reference_translations, dtype=np.float32)
            if translation_prior_reference_translations is not None
            else np.asarray(translations, dtype=np.float32)
        )
        local_translation_log_prior = make_relion_translation_log_prior(
            prior_reference_translations,
            experiment_dataset.voxel_size,
            sigma_offset_angstrom,
            prior_translations[group_image_indices],
            offset_range_pixels=None,
        )
        translation_prior_time += time.time() - translation_prior_t0

        total_local_rotations += int(local_rotations.shape[0])
        max_local_rotations = max(max_local_rotations, int(local_rotations.shape[0]))

        em_t0 = time.time()
        local_setup_t0 = time.time()
        padded_rotations, padded_log_prior, actual_local_rotation_count, local_rotation_block_size = (
            _pad_local_search_rotations(
                local_rotations,
                local_log_prior,
                int(rotation_block_size),
            )
        )
        padded_total_rotations = int(
            ((padded_rotations.shape[0] + local_rotation_block_size - 1) // local_rotation_block_size)
            * local_rotation_block_size
        )

        ## TODO: THIS IS INCREDIBLY MESSY, WE SHOULDNT HAVE TO DEFINE 12 VARIABLES LIKE THIS.
        ## IF THI SIS TO MAKE MATCHING WITH RELION FOR NOW FINE, BUT KEEP TRACK WE NEED TO TO CLEAN THIS UP AFTER (RELION PARITY HAS NOT BEEN ACHIEVED YET)
        valid_pair_count = int(np.count_nonzero(np.asarray(local_log_prior) > -1e20))
        union_pair_count = int(len(group_image_indices) * actual_local_rotation_count)
        padded_pair_count = int(len(group_image_indices) * padded_total_rotations)
        chunk_local_rotations.append(int(actual_local_rotation_count))
        chunk_padded_rotations.append(int(padded_total_rotations))
        chunk_valid_pairs.append(valid_pair_count)
        chunk_union_pairs.append(union_pair_count)
        chunk_padded_pairs.append(padded_pair_count)
        sum_union_rows += int(actual_local_rotation_count)
        sum_padded_rows += int(padded_total_rotations)
        seen_global_rotations[np.asarray(local_indices[:actual_local_rotation_count], dtype=np.int32)] = True
        total_bucket_rotations += padded_total_rotations
        max_bucket_rotations = max(max_bucket_rotations, int(local_rotation_block_size))
        local_setup_time += time.time() - local_setup_t0

        run_em_t0 = time.time()
        run_em_outputs = run_em(
            experiment_dataset,
            mean,
            mean_variance,
            noise_variance,
            padded_rotations,
            translations,
            disc_type,
            image_batch_size=image_batch_size,
            rotation_block_size=local_rotation_block_size,
            current_size=current_size,
            rotation_log_prior=padded_log_prior,
            translation_log_prior=local_translation_log_prior,
            translation_prior_centers=prior_translations[group_image_indices],
            image_indices=group_image_indices,
            score_with_masked_images=score_with_masked_images,
            return_stats=True,
            accumulate_noise=accumulate_noise,
            half_spectrum_scoring=half_spectrum_scoring,
            projection_padding_factor=projection_padding_factor,
            reconstruction_padding_factor=reconstruction_padding_factor,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            use_float64_scoring=use_float64_scoring,
            use_float64_projections=use_float64_projections,
            do_gridding_correction=do_gridding_correction,
            square_window=square_window,
            return_profile=return_profile,
            sparse_pass2=sparse_pass2,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
        )
        run_em_wall_time += time.time() - run_em_t0
        if accumulate_noise:
            if return_profile:
                _, ha_local, Ft_y_g, Ft_ctf_g, stats_g, noise_stats_g, em_profile_g = run_em_outputs
            else:
                _, ha_local, Ft_y_g, Ft_ctf_g, stats_g, noise_stats_g = run_em_outputs
            accum_noise_wsum += np.asarray(noise_stats_g.wsum_sigma2_noise, dtype=np.float64)
            accum_img_power += np.asarray(noise_stats_g.wsum_img_power, dtype=np.float64)
            accum_sumw += float(noise_stats_g.sumw)
        else:
            if return_profile:
                _, ha_local, Ft_y_g, Ft_ctf_g, stats_g, em_profile_g = run_em_outputs
            else:
                _, ha_local, Ft_y_g, Ft_ctf_g, stats_g = run_em_outputs

        if return_profile:
            profile_merge_t0 = time.time()
            profile_dict = em_profile_g._asdict()
            if em_phase_totals is None:
                em_phase_totals = {key: 0.0 for key in profile_dict}
            for key, value in profile_dict.items():
                em_phase_totals[key] += float(value)
            total_adjoint_time += float(em_profile_g.adjoint_y_s + em_profile_g.adjoint_ctf_s)
            sum_union_row_pixels += int(actual_local_rotation_count * em_profile_g.n_windowed)
            sum_padded_row_pixels += int(padded_total_rotations * em_profile_g.n_windowed)
            profile_merge_time += time.time() - profile_merge_t0

        postprocess_t0 = time.time()
        Ft_y_total = Ft_y_total + Ft_y_g
        Ft_ctf_total = Ft_ctf_total + Ft_ctf_g

        local_rot_idx = ha_local // n_trans
        trans_idx = ha_local % n_trans
        if np.any(local_rot_idx >= actual_local_rotation_count):
            raise RuntimeError(
                "Padded local-search rotation selected despite masked prior; "
                f"got index {int(np.max(local_rot_idx))} with actual_count={actual_local_rotation_count}"
            )
        hard_assignment[group_image_indices] = (local_indices[local_rot_idx] * n_trans + trans_idx).astype(np.int32)
        log_evidence[group_image_indices] = np.asarray(
            stats_g.log_evidence_per_image,
            dtype=np.float32,
        )
        best_log_score[group_image_indices] = np.asarray(
            stats_g.best_log_score_per_image,
            dtype=np.float32,
        )
        max_posterior[group_image_indices] = np.asarray(
            stats_g.max_posterior_per_image,
            dtype=np.float32,
        )
        rotation_posterior_sums[local_indices] += np.asarray(
            stats_g.rotation_posterior_sums[:actual_local_rotation_count],
            dtype=np.float64,
        )
        nonzero_row_mask = np.asarray(stats_g.rotation_posterior_sums[:actual_local_rotation_count]) > 0
        nonzero_row_count = int(np.count_nonzero(nonzero_row_mask))
        chunk_nonzero_posterior_rows.append(nonzero_row_count)
        sum_nonzero_posterior_rows += nonzero_row_count
        if nonzero_row_count:
            seen_nonzero_global_rotations[np.asarray(local_indices[:actual_local_rotation_count])[nonzero_row_mask]] = (
                True
            )
        postprocess_time += time.time() - postprocess_t0
        em_time += time.time() - em_t0

    logger.info(
        "Batched local search: %d chunks, median chunk size=%d, mean local rotations=%.1f, max local rotations=%d, "
        "mean bucket rotations=%.1f, max bucket rotations=%d",
        n_chunks,
        int(np.median(chunk_sizes)) if chunk_sizes else 0,
        float(total_local_rotations / max(n_chunks, 1)),
        max_local_rotations,
        float(total_bucket_rotations / max(n_chunks, 1)),
        max_bucket_rotations,
    )
    logger.info(
        "Batched local search timings: metadata=%.2fs, selector=%.2fs, translation_prior=%.2fs, "
        "em=%.2fs (setup=%.2fs, run_em=%.2fs, profile_merge=%.2fs, postprocess=%.2fs)",
        metadata_build_time,
        selector_time,
        translation_prior_time,
        em_time,
        local_setup_time,
        run_em_wall_time,
        profile_merge_time,
        postprocess_time,
    )

    relion_stats = RelionStats(
        log_evidence_per_image=jnp.asarray(log_evidence),
        best_log_score_per_image=jnp.asarray(best_log_score),
        max_posterior_per_image=jnp.asarray(max_posterior),
        rotation_posterior_sums=jnp.asarray(rotation_posterior_sums, dtype=jnp.float32),
    )
    if accumulate_noise:
        from recovar.em.dense_single_volume.helpers.types import NoiseStats

        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.asarray(accum_noise_wsum, dtype=jnp.float32),
            wsum_img_power=jnp.asarray(accum_img_power, dtype=jnp.float32),
            wsum_sigma2_offset=0.0,
            sumw=float(accum_sumw),
        )
        profile_summary = None
        if return_profile:
            total_valid_pairs = float(np.sum(chunk_valid_pairs))
            total_union_pairs = float(np.sum(chunk_union_pairs))
            total_padded_pairs = float(np.sum(chunk_padded_pairs))
            profile_summary = {
                "metadata_build_time_s": np.float64(metadata_build_time),
                "selector_time_s": np.float64(selector_time),
                "translation_prior_time_s": np.float64(translation_prior_time),
                "em_time_s": np.float64(em_time),
                "local_setup_time_s": np.float64(local_setup_time),
                "run_em_wall_time_s": np.float64(run_em_wall_time),
                "profile_merge_time_s": np.float64(profile_merge_time),
                "postprocess_time_s": np.float64(postprocess_time),
                "accounted_em_time_s": np.float64(
                    local_setup_time + run_em_wall_time + profile_merge_time + postprocess_time
                ),
                "unattributed_em_time_s": np.float64(
                    max(
                        em_time - (local_setup_time + run_em_wall_time + profile_merge_time + postprocess_time),
                        0.0,
                    )
                ),
                "n_chunks": np.int32(n_chunks),
                "chunk_sizes": np.asarray(chunk_sizes, dtype=np.int32),
                "chunk_local_rotations": np.asarray(chunk_local_rotations, dtype=np.int32),
                "chunk_padded_rotations": np.asarray(chunk_padded_rotations, dtype=np.int32),
                "chunk_valid_pairs": np.asarray(chunk_valid_pairs, dtype=np.int64),
                "chunk_union_pairs": np.asarray(chunk_union_pairs, dtype=np.int64),
                "chunk_padded_pairs": np.asarray(chunk_padded_pairs, dtype=np.int64),
                "chunk_nonzero_posterior_rows": np.asarray(chunk_nonzero_posterior_rows, dtype=np.int32),
                "sum_union_rows": np.int64(sum_union_rows),
                "sum_padded_rows": np.int64(sum_padded_rows),
                "sum_nonzero_posterior_rows": np.int64(sum_nonzero_posterior_rows),
                "unique_global_rotations": np.int64(np.count_nonzero(seen_global_rotations)),
                "unique_nonzero_global_rotations": np.int64(np.count_nonzero(seen_nonzero_global_rotations)),
                "duplicate_rotation_factor": np.float64(
                    0.0
                    if not np.any(seen_global_rotations)
                    else sum_union_rows / np.count_nonzero(seen_global_rotations)
                ),
                "sum_union_row_pixels": np.int64(sum_union_row_pixels),
                "sum_padded_row_pixels": np.int64(sum_padded_row_pixels),
                "adjoint_seconds_per_row_pixel": np.float64(
                    0.0 if sum_union_row_pixels == 0 else total_adjoint_time / sum_union_row_pixels
                ),
                "union_waste_fraction": np.float64(
                    0.0 if total_union_pairs == 0 else 1.0 - total_valid_pairs / total_union_pairs
                ),
                "padded_waste_fraction": np.float64(
                    0.0 if total_padded_pairs == 0 else 1.0 - total_valid_pairs / total_padded_pairs
                ),
                "padding_only_waste_fraction": np.float64(
                    0.0 if total_padded_pairs == 0 else (total_padded_pairs - total_union_pairs) / total_padded_pairs
                ),
            }
            if em_phase_totals is not None:
                for key, value in em_phase_totals.items():
                    profile_summary[f"em_{key}"] = np.asarray(value)
        return Ft_y_total, Ft_ctf_total, hard_assignment, relion_stats, noise_stats, profile_summary
    profile_summary = None
    if return_profile:
        total_valid_pairs = float(np.sum(chunk_valid_pairs))
        total_union_pairs = float(np.sum(chunk_union_pairs))
        total_padded_pairs = float(np.sum(chunk_padded_pairs))
        profile_summary = {
            "metadata_build_time_s": np.float64(metadata_build_time),
            "selector_time_s": np.float64(selector_time),
            "translation_prior_time_s": np.float64(translation_prior_time),
            "em_time_s": np.float64(em_time),
            "local_setup_time_s": np.float64(local_setup_time),
            "run_em_wall_time_s": np.float64(run_em_wall_time),
            "profile_merge_time_s": np.float64(profile_merge_time),
            "postprocess_time_s": np.float64(postprocess_time),
            "accounted_em_time_s": np.float64(
                local_setup_time + run_em_wall_time + profile_merge_time + postprocess_time
            ),
            "unattributed_em_time_s": np.float64(
                max(em_time - (local_setup_time + run_em_wall_time + profile_merge_time + postprocess_time), 0.0)
            ),
            "n_chunks": np.int32(n_chunks),
            "chunk_sizes": np.asarray(chunk_sizes, dtype=np.int32),
            "chunk_local_rotations": np.asarray(chunk_local_rotations, dtype=np.int32),
            "chunk_padded_rotations": np.asarray(chunk_padded_rotations, dtype=np.int32),
            "chunk_valid_pairs": np.asarray(chunk_valid_pairs, dtype=np.int64),
            "chunk_union_pairs": np.asarray(chunk_union_pairs, dtype=np.int64),
            "chunk_padded_pairs": np.asarray(chunk_padded_pairs, dtype=np.int64),
            "chunk_nonzero_posterior_rows": np.asarray(chunk_nonzero_posterior_rows, dtype=np.int32),
            "sum_union_rows": np.int64(sum_union_rows),
            "sum_padded_rows": np.int64(sum_padded_rows),
            "sum_nonzero_posterior_rows": np.int64(sum_nonzero_posterior_rows),
            "unique_global_rotations": np.int64(np.count_nonzero(seen_global_rotations)),
            "unique_nonzero_global_rotations": np.int64(np.count_nonzero(seen_nonzero_global_rotations)),
            "duplicate_rotation_factor": np.float64(
                0.0 if not np.any(seen_global_rotations) else sum_union_rows / np.count_nonzero(seen_global_rotations)
            ),
            "sum_union_row_pixels": np.int64(sum_union_row_pixels),
            "sum_padded_row_pixels": np.int64(sum_padded_row_pixels),
            "adjoint_seconds_per_row_pixel": np.float64(
                0.0 if sum_union_row_pixels == 0 else total_adjoint_time / sum_union_row_pixels
            ),
            "union_waste_fraction": np.float64(
                0.0 if total_union_pairs == 0 else 1.0 - total_valid_pairs / total_union_pairs
            ),
            "padded_waste_fraction": np.float64(
                0.0 if total_padded_pairs == 0 else 1.0 - total_valid_pairs / total_padded_pairs
            ),
            "padding_only_waste_fraction": np.float64(
                0.0 if total_padded_pairs == 0 else (total_padded_pairs - total_union_pairs) / total_padded_pairs
            ),
        }
        if em_phase_totals is not None:
            for key, value in em_phase_totals.items():
                profile_summary[f"em_{key}"] = np.asarray(value)
    return Ft_y_total, Ft_ctf_total, hard_assignment, relion_stats, profile_summary


def _run_local_search_iteration_exact_v1(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    prior_rotations,
    rotation_grid_rotations,
    rotation_grid_eulers,
    healpix_order,
    sigma_rot,
    sigma_psi,
    translations,
    prior_translations,
    sigma_offset_angstrom,
    offset_range_pixels,
    disc_type,
    image_batch_size,
    rotation_block_size,
    current_size,
    *,
    accumulate_noise=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    use_float64_scoring=False,
    use_float64_projections=False,
    do_gridding_correction=False,
    square_window=False,
    half_spectrum_scoring=False,
    image_corrections=None,
    scale_corrections=None,
    image_pre_shifts=None,
    score_with_masked_images=True,
    return_profile=False,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    adaptive_fraction=0.999,
    max_significants=-1,
    translation_prior_mode="perturbed",
    translation_prior_reference_translations=None,
):
    """Per-image exact local engine over image-specific rotation neighborhoods."""

    rotation_block_size = _local_search_engine_rotation_block_size(rotation_block_size)
    prior_rotations = np.asarray(prior_rotations, dtype=np.float32)
    if prior_rotations.ndim == 3:
        n_prior = prior_rotations.shape[0]
    elif prior_rotations.ndim == 2 and prior_rotations.shape[1] == 3:
        n_prior = prior_rotations.shape[0]
    else:
        raise ValueError(f"prior_rotations must have shape (n,3,3) or (n,3), got {prior_rotations.shape}")
    if prior_translations is None:
        prior_translations = np.zeros(
            (n_prior, np.asarray(translations).shape[1]),
            dtype=np.float32,
        )
    else:
        prior_translations = np.asarray(prior_translations, dtype=np.float32).reshape(
            -1,
            np.asarray(translations).shape[1],
        )

    metadata_t0 = time.time()
    local_grid_metadata = build_local_search_grid_metadata(
        healpix_order,
        grid_eulers=rotation_grid_eulers,
        grid_rotations=rotation_grid_rotations,
    )
    metadata_build_time = time.time() - metadata_t0

    layout_t0 = time.time()
    local_layout = build_local_hypothesis_layout(
        prior_rotations,
        rotation_grid_rotations,
        sigma_rot,
        sigma_psi,
        healpix_order,
        translations,
        prior_translations,
        sigma_offset_angstrom,
        # Match the grouped RELION-mode path: local translation priors use the
        # learned/model sigma, not the legacy range/3 override.
        None,
        experiment_dataset.voxel_size,
        grid_metadata=local_grid_metadata,
        translation_prior_reference_translations=translation_prior_reference_translations,
    )
    selector_time = time.time() - layout_t0

    engine_outputs = run_local_em_exact(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance,
        local_layout,
        disc_type,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        current_size=current_size,
        accumulate_noise=accumulate_noise,
        projection_padding_factor=projection_padding_factor,
        reconstruction_padding_factor=reconstruction_padding_factor,
        score_with_masked_images=score_with_masked_images,
        half_spectrum_scoring=half_spectrum_scoring,
        use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
        do_gridding_correction=do_gridding_correction,
        square_window=square_window,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        return_profile=return_profile,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        # TODO(local-engine-debt): restore RELION-style significant-sample
        # reconstruction here only after the exact local engine matches the
        # grouped RELION-mode path on the real replay benchmarks. The current
        # exact significance path still drifts on iter-6 local replay, so keep
        # exact_v1 on the full soft posterior for parity.
        reconstruct_significant_only=False,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
    )

    if accumulate_noise:
        if return_profile:
            Ft_y, Ft_ctf, hard_assignment, relion_stats, noise_stats, profile_summary = engine_outputs
        else:
            Ft_y, Ft_ctf, hard_assignment, relion_stats, noise_stats = engine_outputs
            profile_summary = None
    else:
        if return_profile:
            Ft_y, Ft_ctf, hard_assignment, relion_stats, profile_summary = engine_outputs
        else:
            Ft_y, Ft_ctf, hard_assignment, relion_stats = engine_outputs
            profile_summary = None
            noise_stats = None

    if return_profile and profile_summary is not None:
        profile_summary = dict(profile_summary)
        profile_summary["metadata_build_time_s"] = np.float64(metadata_build_time)
        profile_summary["selector_time_s"] = np.float64(selector_time)
        profile_summary["translation_prior_time_s"] = np.float64(0.0)

    if accumulate_noise:
        if return_profile:
            return Ft_y, Ft_ctf, hard_assignment, relion_stats, noise_stats, profile_summary
        return Ft_y, Ft_ctf, hard_assignment, relion_stats, noise_stats
    if return_profile:
        return Ft_y, Ft_ctf, hard_assignment, relion_stats, profile_summary
    return Ft_y, Ft_ctf, hard_assignment, relion_stats


from recovar.em.dense_single_volume.helpers.significance import (
    _compute_significance_batched,
)

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
    adaptive_pass2_skip_threshold=ADAPTIVE_PASS2_MAX_SIGNIFICANT_FRACTION,
    # --- RELION-mode parameters (only used when mode="relion") ---
    init_healpix_order=2,
    max_healpix_order=7,
    init_translation_range=10.0,
    init_translation_step=2.0,
    init_translation_sigma_angstrom=10.0,
    particle_diameter_ang=None,
    save_intermediates_dir=None,
    low_resol_join_halves_angstrom=40.0,
    tau2_fudge=1.0,
    perturb_factor=0.0,
    perturb_seed=None,
    perturb_replay_relion_dir=None,
    init_fsc=None,
    init_ave_Pmax=None,
    init_has_high_fsc_at_limit=None,
    init_relion_iteration=0,
    init_image_corrections=None,
    init_scale_corrections=None,
    init_direction_prior=None,
    init_previous_best_translations=None,
    init_previous_best_rotation_eulers=None,
    replay_iteration_overrides=None,
    skip_final_iteration=False,
    local_search_profile_mode="auto",
    local_search_translation_prior_mode="coarse",
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    local_engine="grouped_union",
    emulate_relion_firstiter_cc=False,
    relion_firstiter_ini_high_angstrom=None,
    first_iteration_score_mode="gaussian",
    first_iteration_reconstruction_mode="soft",
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
        Number of rotations per block in em_engine.
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
    adaptive_pass2_skip_threshold : float
        Skip adaptive pass 2 when the mean significant-sample fraction is at
        least this value. Set to a negative value to disable this shortcut and
        keep the full RELION-style two-pass adaptive search.
    init_healpix_order : int
        Starting HEALPix order for RELION mode (default 2, ~14.7 deg).
    max_healpix_order : int
        Maximum HEALPix order (finest angular sampling, default 7).
    init_translation_range : float
        Initial translation search range in pixels (RELION mode).
    init_translation_step : float
        Initial translation step size in pixels (RELION mode).
    init_translation_sigma_angstrom : float
        Initial RELION-style translation prior width in Angstrom.
    particle_diameter_ang : float or None
        RELION particle diameter in Angstrom for the adaptive coarse-image-size
        formula. When None, fall back to ``ori_size * pixel_size``.

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
    if local_engine not in ("grouped_union", "exact_v1", "exact_v2"):
        raise ValueError(
            f"Unknown local_engine={local_engine!r}; expected 'grouped_union', 'exact_v1', or 'exact_v2'",
        )

    if mode == "relion":
        return _run_relion_iteration_loop(
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
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            init_healpix_order=init_healpix_order,
            max_healpix_order=max_healpix_order,
            init_translation_range=init_translation_range,
            init_translation_step=init_translation_step,
            init_translation_sigma_angstrom=init_translation_sigma_angstrom,
            particle_diameter_ang=particle_diameter_ang,
            nside_level=nside_level,
            adaptive_pass2_skip_threshold=adaptive_pass2_skip_threshold,
            save_intermediates_dir=save_intermediates_dir,
            low_resol_join_halves_angstrom=low_resol_join_halves_angstrom,
            tau2_fudge=tau2_fudge,
            perturb_factor=perturb_factor,
            perturb_seed=perturb_seed,
            perturb_replay_relion_dir=perturb_replay_relion_dir,
            init_fsc=init_fsc,
            init_ave_Pmax=init_ave_Pmax,
            init_has_high_fsc_at_limit=init_has_high_fsc_at_limit,
            init_relion_iteration=init_relion_iteration,
            init_image_corrections=init_image_corrections,
            init_scale_corrections=init_scale_corrections,
            init_direction_prior=init_direction_prior,
            init_previous_best_translations=init_previous_best_translations,
            init_previous_best_rotation_eulers=init_previous_best_rotation_eulers,
            replay_iteration_overrides=replay_iteration_overrides,
            skip_final_iteration=skip_final_iteration,
            local_search_profile_mode=local_search_profile_mode,
            local_search_translation_prior_mode=local_search_translation_prior_mode,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
            local_engine=local_engine,
            emulate_relion_firstiter_cc=emulate_relion_firstiter_cc,
            relion_firstiter_ini_high_angstrom=relion_firstiter_ini_high_angstrom,
            first_iteration_score_mode=first_iteration_score_mode,
            first_iteration_reconstruction_mode=first_iteration_reconstruction_mode,
        )

    return _run_legacy_iteration_loop(
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
        relion_current_sizes=relion_current_sizes,
        init_current_size=init_current_size,
        fsc_threshold=fsc_threshold,
        adaptive_oversampling=adaptive_oversampling,
        adaptive_fraction=adaptive_fraction,
        max_significants=max_significants,
        nside_level=nside_level,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
    )


# ---------------------------------------------------------------------------
# RELION-parity refinement mode
# ---------------------------------------------------------------------------


def _run_relion_iteration_loop(
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
    adaptive_fraction,
    max_significants,
    init_healpix_order,
    max_healpix_order,
    init_translation_range,
    init_translation_step,
    init_translation_sigma_angstrom,
    particle_diameter_ang,
    nside_level,
    adaptive_pass2_skip_threshold,
    save_intermediates_dir=None,
    low_resol_join_halves_angstrom=40.0,
    tau2_fudge=1.0,
    perturb_factor=0.0,
    perturb_seed=None,
    perturb_replay_relion_dir=None,
    init_fsc=None,
    init_ave_Pmax=None,
    init_has_high_fsc_at_limit=None,
    init_relion_iteration=0,
    init_image_corrections=None,
    init_scale_corrections=None,
    init_direction_prior=None,
    init_previous_best_translations=None,
    init_previous_best_rotation_eulers=None,
    replay_iteration_overrides=None,
    skip_final_iteration=False,
    local_search_profile_mode="auto",
    local_search_translation_prior_mode="coarse",
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    local_engine="grouped_union",
    emulate_relion_firstiter_cc=False,
    relion_firstiter_ini_high_angstrom=None,
    first_iteration_score_mode="gaussian",
    first_iteration_reconstruction_mode="soft",
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
    from recovar.reconstruction import noise, regularization

    cryo = experiment_datasets[0]
    volume_shape = cryo.volume_shape
    grid_size = cryo.image_shape[0]  # ori_size in RELION terms

    # --- RELION image mask (softMaskOutsideMap on particles) ---
    # RELION masks images to particle_diameter/(2*pixel_size) with a 5-pixel
    # cosine taper before E-step scoring (ml_optimiser.cpp:6288).  The default
    # edge-taper mask (window_mask(D, 0.85, 0.99)) is too tight — it tapers
    # at 54 px vs RELION's 64 px for a 128-px box.
    RELION_WIDTH_MASK_EDGE = 5
    for ds in experiment_datasets:
        if hasattr(ds.image_source.backend, "image_mask_mode"):
            ds.image_source.backend.image_mask_mode = "multiply"
    if particle_diameter_ang is not None and particle_diameter_ang > 0:
        from recovar.core import mask
        from recovar.core.mask import relion_soft_image_mask

        relion_mask = relion_soft_image_mask(
            image_size=grid_size,
            pixel_size=cryo.voxel_size,
            particle_diameter_ang=particle_diameter_ang,
            width_mask_edge_px=RELION_WIDTH_MASK_EDGE,
        )
        for ds in experiment_datasets:
            ds.image_source.backend.image_mask = relion_mask
            if hasattr(ds.image_source.backend, "image_mask_mode"):
                ds.image_source.backend.image_mask_mode = "relion_background_fill"
        logger.info(
            "RELION mode: image mask radius=%.1f px (particle_diameter=%.1f A, edge=%d px)",
            particle_diameter_ang / (2.0 * cryo.voxel_size),
            particle_diameter_ang,
            RELION_WIDTH_MASK_EDGE,
        )

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
        particle_diameter_angstrom=float(particle_diameter_ang or 0.0),
    )

    # RELION mode owns the coarse HEALPix grid. When coarse-grid metadata is
    # provided, regenerate the matching coarse grid here instead of inheriting
    # any finer caller-supplied rotation table.
    current_healpix_order = int(
        init_healpix_order
    )  ##TODO: SURELY THIS INT IS UNNECESSARY? I WANT TO CLEAN UP THIS KIND OF USELESS CODE
    if nside_level is not None:
        ## TODO: ID LIKE BETTER NAMING THAT NSIDE_LEVEL. WHAT DOES THIS MEAN?
        if int(nside_level) != current_healpix_order:
            logger.info(
                "RELION mode: ignoring caller nside_level=%d and regenerating initial coarse grid at healpix_order=%d",
                int(nside_level),
                current_healpix_order,
            )
        current_rotations = get_relion_rotation_grid(
            current_healpix_order
        ).astype(
            np.float32
        )  # I WANT DTYPE TO BE DECLARED AHEAD. E.G. RIGHT NOW ANYTHING THAT USES NP.FLOAT32 SHOULD HAVE SOMETHING LIKE DEFAULT_DTYPE
        current_rotation_eulers = get_relion_rotation_grid_eulers(
            current_healpix_order
        ).astype(
            np.float32
        )  # I WANT DTYPE TO BE DECLARED AHEAD. E.G. RIGHT NOW ANYTHING THAT USES NP.FLOAT32 SHOULD HAVE SOMETHING LIKE DEFAULT_DTYPE
        ## TODO: I ALSO WOULD LIKE TO NOT HAVE SO MANY DTYPE COMMANDS IF THEY ARENT NECESSARY. GOOD CODING SHOULD MAKE INHERITED TYPE SEEMLESS
        current_nside_level = current_healpix_order
    elif rotations is not None:
        current_rotations = np.asarray(rotations, dtype=np.float32)
        current_rotation_eulers = utils.R_to_relion(np.asarray(current_rotations), degrees=True).astype(np.float32)
        current_nside_level = current_healpix_order
    else:
        current_rotations = get_relion_rotation_grid(current_healpix_order).astype(np.float32)
        current_rotation_eulers = get_relion_rotation_grid_eulers(current_healpix_order).astype(np.float32)
        current_nside_level = current_healpix_order
    if translations is None:
        current_translations = jnp.asarray(
            get_translation_grid(init_translation_range, init_translation_step), dtype=jnp.float32
        )
    else:
        current_translations = jnp.asarray(translations, dtype=jnp.float32)
    # Unperturbed base grid — `current_translations` may be replaced per-iter by
    # a perturbed copy (SamplingPerturbation). Keep the base so each iter
    # perturbs a fresh copy rather than compounding prior perturbations.
    base_translations = current_translations
    if save_intermediates_dir is not None:
        os.makedirs(save_intermediates_dir, exist_ok=True)
    if local_search_profile_mode not in {"auto", "on", "off"}:
        raise ValueError(
            f"local_search_profile_mode must be one of {{'auto', 'on', 'off'}}, got {local_search_profile_mode!r}",
        )
    collect_local_search_profile = (
        save_intermediates_dir is not None if local_search_profile_mode == "auto" else local_search_profile_mode == "on"
    )

    # RELION uses pf=2 for both projection and reconstruction (--pad 2).
    # Projection: real-space zero-pad N³→(2N)³, DFT, then trilinear slice.
    # Reconstruction: backproject into (2N)³ Fourier grid, Wiener solve,
    # iDFT at (2N)³, crop real-space to N³.
    PADDING_FACTOR = 2
    PROJECTION_PADDING_FACTOR = 2
    padded_volume_shape = tuple(d * PADDING_FACTOR for d in volume_shape)

    def _safe_batch_sizes(n_rot, n_trans):
        ## TODO: BUT (NO RUSH ON THIS ONE): THIS SHOULD BE SET BY THE THE GPU CAPACITY SOMEHOW.
        """Reduce batch sizes for large pose grids to avoid GPU OOM.

        2026-04-08: bumped budget from 50M to 200M floats. This is the
        score-tensor size budget; the M-step GEMMs and CTF accumulators
        allocate ~10x this much in working memory, so 200M maps to ~2 GB
        peak. Verified to fit on 80 GB A100s for both 64-px tiny (1k
        particles) and 128-px 5k benchmarks. Larger budgets give faster
        per-iter times on tiny but OOM on 128-px boxes.
        """
        # Target the actual score-tensor size: n_img * n_rot_block * n_trans.
        budget = 200_000_000
        n_trans = max(int(n_trans), 1)
        ibs = min(
            image_batch_size,
            max(1, budget // max(n_rot * n_trans, 1)),
        )
        rbs = min(
            rotation_block_size,
            max(64, budget // max(ibs * n_trans, 1)),
        )
        return ibs, rbs

    ## TODO CHANGE NAMES OF THIGNS THIGSN. MEANS IS NOT A REASONABLE NAME HERE.
    ##

    # State: two half-set volumes, noise, prior
    # init_volume can be a single array (used for both halves) or a list/tuple
    # of 2 arrays (one per half-set, matching RELION auto-refine).
    if isinstance(init_volume, (list, tuple)) and len(init_volume) == 2:
        means = [jnp.array(init_volume[0]), jnp.array(init_volume[1])]
    else:
        means = [jnp.array(init_volume), jnp.array(init_volume)]
    noise_variance = jnp.array(init_noise_variance)
    mean_variance = jnp.array(init_mean_variance)

    ## TODO: WE NEED TO FIND A BETTER WAY TO MANAGE ALL OF THESE VARIABLES.
    ## PERHAPS SOME CONFIGS, SOME MORE OBJECTS/ENUM THING, WE SHOULDNT DEFINE 1000 VARIABLES LIKE THIS IN PYTHON.

    # History tracking
    current_sizes = []
    fsc_history = []
    pixel_resolutions = []
    wall_times = []
    hard_assignments = [None, None]
    previous_assignments = [None, None]
    previous_best_translations = [None, None]
    previous_best_rotations = [None, None]
    previous_best_rotation_eulers = [None, None]
    # TODO: THESE IFS SURELY COULD BE MORE CLEAN UP
    if init_previous_best_translations is not None:
        previous_best_translations = [
            np.asarray(init_previous_best_translations[0], dtype=np.float32)
            if init_previous_best_translations[0] is not None
            else None,
            np.asarray(init_previous_best_translations[1], dtype=np.float32)
            if init_previous_best_translations[1] is not None
            else None,
        ]
    if init_previous_best_rotation_eulers is not None:
        previous_best_rotation_eulers = [
            np.asarray(init_previous_best_rotation_eulers[0], dtype=np.float32)
            if init_previous_best_rotation_eulers[0] is not None
            else None,
            np.asarray(init_previous_best_rotation_eulers[1], dtype=np.float32)
            if init_previous_best_rotation_eulers[1] is not None
            else None,
        ]
    max_posterior_per_half = [None, None]
    rotation_posterior_per_half = [None, None]
    significant_counts = []
    data_vs_prior_trajectory = []
    healpix_order_trajectory = []
    ave_Pmax_trajectory = []
    pmax_per_image_history = []
    # Per-iter per-shell trajectories for RELION parity diff (added for the
    # 2026-04 audit). noise_radial_trajectory[i] = sigma2_noise per shell after
    # iter i's noise update; tau2_radial_trajectory[i] = recovar's tau2 prior
    # per shell after iter i's signal-prior update.
    noise_radial_trajectory = []
    tau2_radial_trajectory = []
    tau2_sigma2_trajectory = []
    tau2_avg_weight_trajectory = []
    tau2_shell_sum_trajectory = []
    tau2_shell_count_trajectory = []
    tau2_fsc_used_trajectory = []
    tau2_ssnr_trajectory = []

    ## TODO: WE NEED MUCH BETTER NAMING THAN THIS, I AHVE NO IDEA WHAT THIS IS SAYING OR WHAT IT DOES
    ## IF ITS FOR RELION PARITY OKAY, BUT MAYBE WE CNA AT LEAST HAVE COMMENTS NEXT TO DEF SO I HAVE SOME IDEA OF WHATS GOING ON

    # C1 (RELION-parity): per-iter sigma2_offset update from data. Initialized
    # from `init_translation_sigma_angstrom`; updated each iter to the
    # MAP-based RMS of per-particle translation deviations from the prior
    # (`state.current_changes_optimal_offsets_angstrom`). Clamped to RELION's
    # min_sigma2_offset = 2 (in pixels² → sqrt(2) pixels = 1.41 * voxel_size Å).

    ## TODO: THIS SOUND EXTREMELY PROBLMATIC. MAKE CHANGE TO THE ENGINE M-STEP, DON'T MAKE RANDOM CHANGES LIKE THIS TO THE LOGIC. THIS IS VERY URGENT AND BAD.
    # Cite: ml_optimiser.cpp:5217-5240. We use the MAP approximation rather
    # than the full posterior-weighted formula because it does not require
    # changes to the engine's M-step accumulation; the difference is small
    # for concentrated posteriors (Pmax > 0.8 typical at iter 6+).
    current_sigma_offset_angstrom = float(init_translation_sigma_angstrom)
    sigma_offset_used_trajectory = []
    sigma_offset_trajectory = []
    frac_changed_trajectory = []
    acc_rot_trajectory = []
    smallest_change_angles_trajectory = []
    smallest_change_offsets_trajectory = []
    best_rotation_eulers_history = []
    best_translations_history = []
    relion_incr_size = 10  # RELION default
    relion_has_high_fsc_at_limit = bool(init_has_high_fsc_at_limit) if init_has_high_fsc_at_limit is not None else False
    global_direction_prior = None
    global_direction_prior_order = None

    ## TODO: DO THE NEXT 2 FOR LOOPS DO THE SAME THING? I'M CONFUSED.
    ## AT LEAST ITS TERRIBLE CODE

    # --- Per-image corrections (RELION parity: avg_norm/normcorr * scale) ---
    # RELION applies img *= avg_norm_correction/normcorr (ml_optimiser.cpp:6240)
    # and scale_correction to the reference (line 7298).  The caller must
    # compute (avg_norm/normcorr)*scale and pass it here.  Passed through to
    # run_em's image_corrections parameter.
    # The arrays are indexed by the HALF-SET dataset order (not global particle
    # order), matching experiment_datasets[k].
    image_corrections_per_half = [None, None]
    if init_image_corrections is not None:
        image_corrections_per_half = init_image_corrections
        for k in range(2):
            if image_corrections_per_half[k] is not None:
                ic = np.asarray(image_corrections_per_half[k], dtype=np.float32)
                logger.info(
                    "RELION mode: image_corrections half-%d: mean=%.4f, std=%.4f, min=%.4f, max=%.4f (%d images)",
                    k + 1,
                    ic.mean(),
                    ic.std(),
                    ic.min(),
                    ic.max(),
                    len(ic),
                )
                image_corrections_per_half[k] = ic

    # --- Per-image scale corrections (RELION parity: reference side) ---
    # RELION applies rlnGroupScaleCorrection to the REFERENCE, not the image
    # (ml_optimiser.cpp:7295-7298).  This means the E-step norm-term and
    # M-step denominator carry scale².  Passed to run_em's
    # scale_corrections parameter to multiply ctf²/σ² by scale².
    scale_corrections_per_half = [None, None]
    if init_scale_corrections is not None:
        scale_corrections_per_half = init_scale_corrections
        for k in range(2):
            if scale_corrections_per_half[k] is not None:
                sc = np.asarray(scale_corrections_per_half[k], dtype=np.float32)
                logger.info(
                    "RELION mode: scale_corrections half-%d: mean=%.4f, std=%.4f, min=%.4f, max=%.4f (%d images)",
                    k + 1,
                    sc.mean(),
                    sc.std(),
                    sc.min(),
                    sc.max(),
                    len(sc),
                )
                scale_corrections_per_half[k] = sc

    # --- Direction prior from snapshot ---
    # When starting from a RELION snapshot, the previous iteration's
    # pdf_orientation is a non-uniform prior over HEALPix directions.
    # RELION applies this in the next E-step.  recovar must do the same.
    if init_direction_prior is not None:
        global_direction_prior = np.asarray(init_direction_prior, dtype=np.float32)
        global_direction_prior_order = infer_direction_prior_healpix_order(global_direction_prior)
        logger.info(
            "RELION mode: loaded init direction prior: %d directions, range=[%.6f, %.6f], %d zero-probability",
            len(global_direction_prior),
            global_direction_prior.min(),
            global_direction_prior.max(),
            int(np.sum(global_direction_prior == 0)),
        )

    # Extract a per-shell radial profile from the input pixel-array
    # noise_variance for diagnostic logging only ("noise update per shell:
    # old=... new=...").
    from recovar.core import fourier_transform_utils

    n_shells_init = cryo.image_shape[0] // 2 + 1
    radial_dist_init = np.clip(
        fourier_transform_utils.get_grid_of_radial_distances(
            cryo.image_shape,
            scaled=False,
            frequency_shift=0,
        )
        .astype(int)
        .reshape(-1),
        0,
        n_shells_init - 1,
    )
    previous_noise_radial = np.zeros(n_shells_init, dtype=np.float64)
    shell_counts = np.zeros(n_shells_init, dtype=np.float64)
    noise_variance_np = np.asarray(noise_variance, dtype=np.float64)
    np.add.at(previous_noise_radial, radial_dist_init[: noise_variance_np.size], noise_variance_np)
    np.add.at(shell_counts, radial_dist_init[: noise_variance_np.size], 1.0)
    shell_counts = np.maximum(shell_counts, 1.0)
    previous_noise_radial = previous_noise_radial / shell_counts
    ##
    previous_noise_radial = jnp.asarray(previous_noise_radial[:n_shells_init], dtype=jnp.float32)

    # --- RELION SamplingPerturbation state (healpix_sampling.cpp:167-174) ---
    # RELION applies a random rigid rotation of the entire SO(3) trial grid at
    # each iteration: A -> A @ R_perturb with R_perturb = R_from_relion([m,m,m])
    # and m = random_perturbation * angular_sampling. The random_perturbation
    # is advanced per iter via realWRAP(prev + rnd_unif(0.5*pf, pf), -pf, +pf).
    # For exact parity replay, read _rlnSamplingPerturbInstance from RELION's
    # per-iter sampling.star.
    random_perturbation = 0.0
    perturb_rng = np.random.default_rng(perturb_seed) if perturb_seed is not None else np.random.default_rng()
    iteration = 0
    while not state.has_converged and iteration < max_iter:
        t0 = time.time()
        _parity_dump.start_iteration(iteration)
        iter_replay_override = None
        if replay_iteration_overrides is not None and iteration < len(replay_iteration_overrides):
            iter_replay_override = replay_iteration_overrides[iteration]
        relion_firstiter_cc_this_iter = bool(
            emulate_relion_firstiter_cc and init_relion_iteration == 0 and iteration == 0
        )
        first_iter_normalized_cc_this_iter = bool(
            first_iteration_score_mode == "normalized_cc" and init_relion_iteration == 0 and iteration == 0
        )
        first_iter_hard_reconstruction_this_iter = bool(
            first_iteration_reconstruction_mode == "hard" and init_relion_iteration == 0 and iteration == 0
        )

        ## TODO: THIS IS REASONABLE, BUT DOES IT BREAK IF WE TEST A SINGLE ITER AS WE HAVE BEEN DOING FOR PARITY?

        # --- Determine current_size using RELION's FSC-derived SSNR (C4/C5) ---
        # At iteration 0, no previous half-map FSC exists yet; use the initial
        # resolution plus RELION's bootstrap image-size growth. After that,
        # mimic RELION's auto-refine update:
        # 1. zero FSC beyond the previous current_size limit
        # 2. convert FSC -> SSNR (= data_vs_prior in split-half auto-refine)
        # 3. grow current_size using ave_Pmax, FSC at the current limit, and
        #    RELION's dynamic incr_size heuristic.
        if iteration == 0:
            if init_relion_iteration == 0:
                seeded_cs = bootstrap_current_size_from_ini_high_relion(
                    grid_size,
                    float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
                    relion_firstiter_ini_high_angstrom,
                    incr_size=relion_incr_size,
                )
            else:
                seeded_cs = None
            if seeded_cs is not None:
                cs = int(seeded_cs)
                data_vs_prior_iter = None
                logger.info(
                    "RELION init bootstrap: seeding iter-1 current_size from ini_high=%.2f A -> %d",
                    float(relion_firstiter_ini_high_angstrom),
                    cs,
                )
            elif init_fsc is not None:
                fsc_prev = np.asarray(init_fsc, dtype=np.float32).copy()
                prev_cs = int(init_current_size)
                if prev_cs < grid_size:
                    fsc_prev[min(len(fsc_prev), prev_cs // 2) :] = 0.0
                data_vs_prior_iter = np.asarray(
                    fsc_to_relion_ssnr(fsc_prev, tau2_fudge=tau2_fudge),
                )
                data_vs_prior_trajectory.append(data_vs_prior_iter)
                res_shell = resolution_from_data_vs_prior(
                    data_vs_prior_iter,
                    allow_high_res_recovery=True,
                )
                relion_incr_size, relion_has_high_fsc_at_limit = update_relion_growth_state_from_fsc(
                    fsc_prev,
                    prev_cs,
                    incr_size=relion_incr_size,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                )
                _init_pmax = float(init_ave_Pmax) if init_ave_Pmax is not None else 0.0
                raw_cs = compute_current_size_relion(
                    res_shell,
                    grid_size,
                    ave_Pmax=_init_pmax,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                    incr_size=relion_incr_size,
                )
                cs = quantize_current_size(raw_cs, ori_size=grid_size)
            else:
                cs = _bootstrap_current_size_relion(init_current_size, grid_size)
                data_vs_prior_iter = None
        else:
            fsc_prev = np.asarray(fsc_history[-1], dtype=np.float32).copy()
            prev_cs = current_sizes[-1]
            if prev_cs < grid_size:
                fsc_prev[min(len(fsc_prev), prev_cs // 2) :] = 0.0

            # data_vs_prior = tau2_fudge * fsc / (1 - fsc), matching
            # RELION's updateSSNRarrays at backprojector.cpp:1117-1123
            # for the gold-standard split-half auto-refine path.
            data_vs_prior_iter = np.asarray(
                fsc_to_relion_ssnr(fsc_prev, tau2_fudge=tau2_fudge),
            )
            data_vs_prior_trajectory.append(data_vs_prior_iter)
            res_shell = resolution_from_data_vs_prior(
                data_vs_prior_iter,
                allow_high_res_recovery=True,
            )
            relion_incr_size, relion_has_high_fsc_at_limit = update_relion_growth_state_from_fsc(
                fsc_prev,
                prev_cs,
                incr_size=relion_incr_size,
                has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
            )

            raw_cs = compute_current_size_relion(
                res_shell,
                grid_size,
                ave_Pmax=state.ave_Pmax,
                has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                incr_size=relion_incr_size,
            )
            cs = quantize_current_size(raw_cs, ori_size=grid_size)

        cs = quantize_current_size(cs, ori_size=grid_size)

        # --- Replay override: force recovar's sampling state to mirror RELION ---
        # When replaying, RELION's per-iter sampling.star dictates the actual
        # hp_order, offset_range, and offset_step used at this iteration.
        # Overriding `state.healpix_order`, `state.translation_range` and
        # `state.translation_step` here makes the downstream grid regen code
        # produce the same grid RELION did, so the perturbation applied later
        # is on the correct base grid.
        _replay_meta = None
        _replay_prior_translations = None
        if perturb_replay_relion_dir is not None:
            _star = os.path.join(
                perturb_replay_relion_dir,
                f"run_it{init_relion_iteration + iteration + 1:03d}_sampling.star",
            )
            _replay_meta = read_relion_sampling_metadata(_star)
            _replay_prior_translations = jnp.array(
                get_translation_grid(
                    float(_replay_meta["offset_range"]),
                    float(_replay_meta["offset_step"]),
                ).astype(np.float32)
            )
            _relion_hp = int(_replay_meta["healpix_order"])
            # RELION stores offset_{range,step} in Angstroms; convert to px.
            _px = float(cryo.voxel_size) if cryo.voxel_size > 0 else 1.0
            _relion_offset_range = float(_replay_meta["offset_range"]) / _px
            _relion_offset_step = float(_replay_meta["offset_step"]) / _px
            _capped_hp = min(_relion_hp, state.max_healpix_order)
            if state.healpix_order != _capped_hp:
                if _capped_hp < _relion_hp:
                    logger.info(
                        "Replay override: healpix_order %d -> %d (RELION %d capped by max_healpix_order=%d, from %s)",
                        state.healpix_order,
                        _capped_hp,
                        _relion_hp,
                        state.max_healpix_order,
                        _star,
                    )
                else:
                    logger.info(
                        "Replay override: healpix_order %d -> %d (from %s)",
                        state.healpix_order,
                        _capped_hp,
                        _star,
                    )
                state.healpix_order = _capped_hp
            _replay_do_local = bool(state.healpix_order >= LOCAL_SEARCH_HEALPIX_ORDER)
            if state.do_local_search != _replay_do_local:
                logger.info(
                    "Replay override: local_search %s -> %s (healpix_order=%d)",
                    state.do_local_search,
                    _replay_do_local,
                    state.healpix_order,
                )
                state.do_local_search = _replay_do_local
                if _replay_do_local:
                    state.sigma_rot = 0.0
                    state.sigma_psi = 0.0
            if (
                abs(float(state.translation_range) - _relion_offset_range) > 1e-6
                or abs(float(state.translation_step) - _relion_offset_step) > 1e-6
            ):
                logger.info(
                    "Replay override: translation_range %.3f -> %.3f px, step %.3f -> %.3f px",
                    float(state.translation_range),
                    _relion_offset_range,
                    float(state.translation_step),
                    _relion_offset_step,
                )
                state.translation_range = _relion_offset_range
                state.translation_step = _relion_offset_step

            # Override current_size from the RELION model star that records the
            # control state for the replayed E-step. Empirically, replaying
            # RELION iter N+1 against the saved benchmark trajectory requires
            # reading run_it{N+1}_model.star, not run_it{N}_model.star:
            # the saved model star already carries the control variables
            # (current_size, sigma_offset) used by that E-step.
            _cs_iter = _replay_control_model_iteration(init_relion_iteration, iteration)
            _model_star = os.path.join(
                perturb_replay_relion_dir,
                f"run_it{_cs_iter:03d}_half1_model.star",
            )
            if os.path.exists(_model_star):
                _model_meta = read_relion_model_metadata(_model_star)
                _relion_cs = int(_model_meta["current_image_size"])
                if _relion_cs <= 0:
                    logger.info(
                        "Replay override: ignoring non-positive current_size=%d from %s",
                        _relion_cs,
                        _model_star,
                    )
                elif cs != _relion_cs:
                    logger.info(
                        "Replay override: current_size %d -> %d (from %s)",
                        cs,
                        _relion_cs,
                        _model_star,
                    )
                    cs = _relion_cs

            if iteration > 0:
                _prior_iter = init_relion_iteration + iteration
                _prior_star = os.path.join(
                    perturb_replay_relion_dir,
                    f"run_it{_prior_iter:03d}_half1_model.star",
                )
                if os.path.exists(_prior_star) and (
                    iter_replay_override is None or iter_replay_override.get("direction_prior") is None
                ):
                    _relion_direction_prior = read_relion_direction_prior(_prior_star)
                    _relion_direction_prior_order = infer_direction_prior_healpix_order(_relion_direction_prior)
                    if _relion_direction_prior_order != state.healpix_order:
                        logger.info(
                            "Replay override: remapping direction prior from healpix_order=%d to %d",
                            _relion_direction_prior_order,
                            state.healpix_order,
                        )
                        _relion_direction_prior = remap_direction_prior_to_healpix_order(
                            _relion_direction_prior,
                            _relion_direction_prior_order,
                            state.healpix_order,
                        )
                        _relion_direction_prior_order = state.healpix_order
                    global_direction_prior = _relion_direction_prior
                    global_direction_prior_order = _relion_direction_prior_order
                    logger.info(
                        "Replay override: direction prior <- %s (%d directions, range=[%.6f, %.6f], zeros=%d)",
                        _prior_star,
                        len(global_direction_prior),
                        float(global_direction_prior.min()),
                        float(global_direction_prior.max()),
                        int(np.sum(global_direction_prior == 0)),
                    )

        if iter_replay_override is not None:
            _replay_sigma = iter_replay_override.get("translation_sigma_angstrom")
            if _replay_sigma is not None:
                current_sigma_offset_angstrom = float(_replay_sigma)
                logger.info(
                    "Replay override: sigma_offset <- %.4f A (iter=%d)",
                    current_sigma_offset_angstrom,
                    iteration + 1,
                )
            _replay_prev_trans = iter_replay_override.get("previous_best_translations")
            if _replay_prev_trans is not None:
                previous_best_translations = [
                    np.asarray(_replay_prev_trans[0], dtype=np.float32) if _replay_prev_trans[0] is not None else None,
                    np.asarray(_replay_prev_trans[1], dtype=np.float32) if _replay_prev_trans[1] is not None else None,
                ]
                logger.info(
                    "Replay override: previous_best_translations <- half1=%s half2=%s",
                    "set" if previous_best_translations[0] is not None else "none",
                    "set" if previous_best_translations[1] is not None else "none",
                )
            _replay_prev_rots = iter_replay_override.get("previous_best_rotations")
            if _replay_prev_rots is not None:
                previous_best_rotations = [
                    np.asarray(_replay_prev_rots[0], dtype=np.float32) if _replay_prev_rots[0] is not None else None,
                    np.asarray(_replay_prev_rots[1], dtype=np.float32) if _replay_prev_rots[1] is not None else None,
                ]
                logger.info(
                    "Replay override: previous_best_rotations <- half1=%s half2=%s",
                    "set" if previous_best_rotations[0] is not None else "none",
                    "set" if previous_best_rotations[1] is not None else "none",
                )
            _replay_prev_eulers = iter_replay_override.get("previous_best_rotation_eulers")
            if _replay_prev_eulers is not None:
                previous_best_rotation_eulers = [
                    np.asarray(_replay_prev_eulers[0], dtype=np.float32)
                    if _replay_prev_eulers[0] is not None
                    else None,
                    np.asarray(_replay_prev_eulers[1], dtype=np.float32)
                    if _replay_prev_eulers[1] is not None
                    else None,
                ]
                logger.info(
                    "Replay override: previous_best_rotation_eulers <- half1=%s half2=%s",
                    "set" if previous_best_rotation_eulers[0] is not None else "none",
                    "set" if previous_best_rotation_eulers[1] is not None else "none",
                )
            _replay_img_corr = iter_replay_override.get("image_corrections")
            if _replay_img_corr is not None:
                image_corrections_per_half = [
                    np.asarray(_replay_img_corr[0], dtype=np.float32) if _replay_img_corr[0] is not None else None,
                    np.asarray(_replay_img_corr[1], dtype=np.float32) if _replay_img_corr[1] is not None else None,
                ]
                logger.info(
                    "Replay override: image_corrections <- half1=%s half2=%s",
                    "set" if image_corrections_per_half[0] is not None else "none",
                    "set" if image_corrections_per_half[1] is not None else "none",
                )
            _replay_scale_corr = iter_replay_override.get("scale_corrections")
            if _replay_scale_corr is not None:
                scale_corrections_per_half = [
                    np.asarray(_replay_scale_corr[0], dtype=np.float32) if _replay_scale_corr[0] is not None else None,
                    np.asarray(_replay_scale_corr[1], dtype=np.float32) if _replay_scale_corr[1] is not None else None,
                ]
                logger.info(
                    "Replay override: scale_corrections <- half1=%s half2=%s",
                    "set" if scale_corrections_per_half[0] is not None else "none",
                    "set" if scale_corrections_per_half[1] is not None else "none",
                )
            _replay_dir_prior = iter_replay_override.get("direction_prior")
            if _replay_dir_prior is not None:
                global_direction_prior = np.asarray(_replay_dir_prior, dtype=np.float32)
                global_direction_prior_order = infer_direction_prior_healpix_order(global_direction_prior)
                if global_direction_prior_order != state.healpix_order:
                    logger.info(
                        "Replay override: remapping provided direction prior from healpix_order=%d to %d",
                        global_direction_prior_order,
                        state.healpix_order,
                    )
                    global_direction_prior = remap_direction_prior_to_healpix_order(
                        global_direction_prior,
                        global_direction_prior_order,
                        state.healpix_order,
                    )
                    global_direction_prior_order = state.healpix_order
                logger.info(
                    "Replay override: direction prior <- provided override (%d directions, range=[%.6f, %.6f], zeros=%d)",
                    len(global_direction_prior),
                    float(global_direction_prior.min()),
                    float(global_direction_prior.max()),
                    int(np.sum(global_direction_prior == 0)),
                )

        sigma_offset_used_trajectory.append(float(current_sigma_offset_angstrom))
        current_sizes.append(cs)
        healpix_order_trajectory.append(state.healpix_order)

        logger.info(
            "=== RELION Iteration %d/%d: current_size=%d, healpix_order=%d, local_search=%s ===",
            iteration + 1,
            max_iter,
            cs,
            state.healpix_order,
            state.do_local_search,
        )

        # --- Angular step refinement: regenerate rotation grid if needed ---
        # When update_refinement_state incremented healpix_order, we need
        # a new rotation grid at the finer level.
        # IMPORTANT: At order >= 5, the full grid has 2.4M+ rotations which
        # OOMs the GPU.  Instead, keep the order-4 grid as the "base" and
        # rely on local search + oversampling to achieve finer angular steps.
        # The order is still tracked for sigma calculation.
        ## TODO: I DON'T LIKE HOW GLOBAL CONSTANTS ARE SET THROUGHOUT THE CODE. WE NEED A BETTER WAY TO DO THIS. PERHAPS SOME CONFIG, SOMETHING
        MAX_FULL_GRID_ORDER = 4
        if state.healpix_order != current_healpix_order:
            new_order = min(state.healpix_order, MAX_FULL_GRID_ORDER)
            if new_order != current_healpix_order:
                logger.info(
                    "Regenerating rotation grid: order %d -> %d",
                    current_healpix_order,
                    new_order,
                )
                current_rotations = get_relion_rotation_grid(new_order).astype(np.float32)
                current_rotation_eulers = get_relion_rotation_grid_eulers(new_order).astype(np.float32)
                current_healpix_order = new_order
                global_direction_prior = None
                global_direction_prior_order = None
            else:
                logger.info(
                    "Angular step refined to order %d (grid stays at order %d — local search handles finer sampling)",
                    state.healpix_order,
                    current_healpix_order,
                )
            current_nside_level = current_healpix_order

            # Regenerate translation grid based on updated parameters
            current_translations = jnp.array(
                get_translation_grid(
                    state.translation_range,
                    state.translation_step,
                ).astype(np.float32)
            )
            base_translations = current_translations
            logger.info(
                "New grid: %d rotations, %d translations (range=%.1f, step=%.1f)",
                current_rotations.shape[0],
                current_translations.shape[0],
                state.translation_range,
                state.translation_step,
            )
        elif _replay_meta is not None:
            # Translation params may have changed under replay without an
            # hp_order bump. Regenerate the translation grid to match RELION.
            _new_t = jnp.array(
                get_translation_grid(
                    state.translation_range,
                    state.translation_step,
                ).astype(np.float32)
            )
            if _new_t.shape != base_translations.shape or not jnp.allclose(_new_t, base_translations):
                current_translations = _new_t
                base_translations = _new_t
                logger.info(
                    "Replay: regenerated translation grid: %d translations (range=%.2f px, step=%.2f px)",
                    current_translations.shape[0],
                    state.translation_range,
                    state.translation_step,
                )

        # --- Local angular search bookkeeping ---
        # Once RELION enters local search, each image should search around its
        # own previous orientation on the true current HEALPix order. Use the
        # exact rotations selected in the previous iteration, not the nearest
        # snapped grid indices.
        effective_rotations = current_rotations
        effective_rotation_eulers = np.asarray(current_rotation_eulers, dtype=np.float32)
        rotation_log_prior = None
        use_local = state.do_local_search and all(eulers is not None for eulers in previous_best_rotation_eulers)
        # --- Apply RELION SamplingPerturbation to the trial grid for this iter ---
        # healpix_sampling.cpp:1909-1934 (rotations) + 1810-1820 (translations)
        # Perturbation is a rigid rotation of SO(3): A := A @ R_perturb applied
        # AFTER oversampling. At adaptive_oversampling=0 (os0 RELION runs),
        # the coarse grid IS the trial grid so we apply directly here.
        if _replay_meta is not None:
            random_perturbation = float(_replay_meta["random_perturbation"])
            logger.info(
                "Perturbation replay: iter=%d rp=%+.5f pf=%.3f relion_hp_order=%d",
                iteration + 1,
                random_perturbation,
                float(_replay_meta["perturbation_factor"]),
                int(_replay_meta["healpix_order"]),
            )
        elif perturb_factor > 0:
            random_perturbation = advance_relion_perturbation(random_perturbation, perturb_factor, perturb_rng)
            logger.info("Perturbation advance: iter=%d rp=%+.5f", iteration + 1, random_perturbation)
        if _replay_meta is not None or perturb_factor > 0:
            # Use RELION's actual hp_order when replaying (recovar's current
            # grid order may be capped at MAX_FULL_GRID_ORDER=4 for memory).
            _angsamp_order = int(_replay_meta["healpix_order"]) if _replay_meta is not None else current_healpix_order
            angsamp_deg = relion_angular_sampling_deg(_angsamp_order, adaptive_oversampling=0)
            effective_rotations = apply_relion_rotation_perturbation(
                np.asarray(effective_rotations),
                random_perturbation,
                angsamp_deg,
            ).astype(np.float32)
            if not use_local:
                effective_rotation_eulers = utils.R_to_relion(np.asarray(effective_rotations), degrees=True).astype(
                    np.float32
                )
            _perturbed_translations = apply_relion_translation_perturbation(
                np.asarray(base_translations),
                random_perturbation,
                float(state.translation_step),
            )
            current_translations = jnp.asarray(_perturbed_translations, dtype=jnp.float32)
        if relion_firstiter_cc_this_iter and current_translations.shape[0] > 1:
            center_idx = int(current_translations.shape[0] // 2)
            current_translations = current_translations[center_idx : center_idx + 1]
            logger.info(
                "RELION iter-1 CC emulation: restricting translation grid to the perturbed center shift %s",
                np.asarray(current_translations[0], dtype=np.float32),
            )
        local_search_order = None
        local_search_rotations = None
        local_search_rotation_eulers = None
        sigma_rot = state.sigma_rot
        sigma_psi = state.sigma_psi if state.sigma_psi > 0 else sigma_rot
        if use_local and sigma_rot <= 0:
            step_rad = np.deg2rad(healpix_angular_step(state.healpix_order) / (2**state.adaptive_oversampling))
            sigma_rot = np.sqrt(2.0 * 2.0) * step_rad
            sigma_psi = sigma_rot

        if use_local:
            local_search_order = state.healpix_order + state.adaptive_oversampling
            if effective_rotations.shape[0] != rotation_grid_size(local_search_order):
                logger.info(
                    "Generating fine local-search grid: order=%d (%d rotations) from capped base order=%d",
                    local_search_order,
                    rotation_grid_size(local_search_order),
                    current_healpix_order,
                )
                local_search_rotations = get_relion_rotation_grid(local_search_order).astype(np.float32)
                if abs(float(random_perturbation)) > 1e-12:
                    local_search_rotations = apply_relion_rotation_perturbation(
                        np.asarray(local_search_rotations, dtype=np.float32),
                        random_perturbation,
                        relion_angular_sampling_deg(local_search_order, adaptive_oversampling=0),
                    ).astype(np.float32)
                    local_search_rotation_eulers = utils.R_to_relion(
                        np.asarray(local_search_rotations, dtype=np.float32),
                        degrees=True,
                    ).astype(np.float32)
                else:
                    local_search_rotation_eulers = get_relion_rotation_grid_eulers(local_search_order).astype(
                        np.float32
                    )
            else:
                local_search_rotations = effective_rotations
                local_search_rotation_eulers = None
            logger.info(
                "Local search (batched exact): fine_order=%d, sigma_rot=%.4f rad (%.2f deg), sigma_psi=%.4f rad",
                local_search_order,
                sigma_rot,
                np.rad2deg(sigma_rot),
                sigma_psi,
            )
        elif global_direction_prior is not None and global_direction_prior_order == current_healpix_order:
            rotation_log_prior = make_relion_direction_log_prior(
                global_direction_prior,
                current_healpix_order,
            )
            logger.info(
                "Using learned global direction prior: %d directions at healpix_order=%d",
                global_direction_prior.shape[0],
                current_healpix_order,
            )

        cs_for_engine = cs if cs < cryo.image_shape[0] else None

        # --- Run E+M on each half-set ---
        # Two modes: single-pass (adaptive_oversampling=0) or two-pass
        # coarse/fine (adaptive_oversampling>=1).
        iter_sig_counts = None
        use_adaptive = state.adaptive_oversampling > 0 and not use_local and effective_rotations.shape[0] > 16
        use_global_significant_support = (
            state.adaptive_oversampling == 0
            and not use_local
            and effective_rotations.shape[0] > 16
            and adaptive_fraction < 1.0
            and not (relion_firstiter_cc_this_iter or first_iter_normalized_cc_this_iter)
            and not first_iter_hard_reconstruction_this_iter
        )

        # Track the rotation grids used for pose extraction.
        # When adaptive oversampling is active, ha_k indices refer to the
        # oversampled grid (from pass 2), not effective_rotations.
        pose_rotations = [None, None]  # rotations to use with ha for poses
        pose_rotation_eulers = [None, None]
        pose_translations = [
            np.asarray(current_translations, dtype=np.float32),
            np.asarray(current_translations, dtype=np.float32),
        ]
        best_pose_rotations = [None, None]
        best_pose_rotation_eulers = [None, None]
        best_pose_translations = [None, None]
        translation_search_bases = [None, None]
        # Coarse-grid assignments for local search tracking (always indexed
        # into effective_rotations, even when adaptive oversampling is used).
        coarse_ha = [None, None]
        adaptive_pass1_diag = [None, None]

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
                effective_step_deg,
                pixel_size,
                grid_size,
                particle_diameter=particle_diameter_ang,
            )
            coarse_size = clamp_relion_coarse_image_size(
                coarse_size,
                cs if cs_for_engine is not None else None,
                grid_size,
            )
            coarse_cs = coarse_size if coarse_size < grid_size else None

            logger.info(
                "Adaptive oversampling: pass 1 at coarse_size=%s, "
                "pass 2 at current_size=%s (oversampling=%d, particle_diameter=%s)",
                coarse_cs,
                cs_for_engine,
                state.adaptive_oversampling,
                (f"{float(particle_diameter_ang):.1f} A" if particle_diameter_ang is not None else "box_size"),
            )

        noise_stats_per_half = [None, None]

        for k in range(2):
            translation_search_base = relion_translation_search_base(previous_best_translations[k])
            translation_search_bases[k] = translation_search_base
            current_translation_range = float(state.translation_range)
            # RELION translation prior sigma (ml_optimiser.cpp:7737-7746):
            # RELION checks `offset_range_x` (rlnOffsetRangeX in optimiser.star),
            # NOT the search-grid `offset_range` (rlnOffsetRange in sampling.star).
            # When offset_range_x > 0: sigma² = range_x²/9 (per-axis override)
            # When offset_range_x <= 0: sigma² = model.sigma2_offset (learned)
            # For this dataset, rlnOffsetRangeX = -1 → model sigma is used.
            # We always use current_sigma_offset_angstrom (from model star).
            #
            # Evaluate the translation prior on the total offset. After
            # pre-centering the image by the previous absolute offset, the
            # search variable is the relative delta. To express
            # |old_offset + delta|^2 as |delta - center|^2, set
            # center = -old_offset.
            trans_prior_center = -translation_search_base if translation_search_base is not None else None
            translation_log_prior = make_relion_translation_log_prior(
                np.asarray(current_translations, dtype=np.float32),
                cryo.voxel_size,
                current_sigma_offset_angstrom,
                trans_prior_center,
                offset_range_pixels=None,
            )
            if use_local:
                ## TODO: Is this STRATEGY REALLY WHAT DOES IN TERMS OF COMPUTE? THIS ALL SEEMS TO HACKY. IF IT IS FINE, BUT WE SHOULD TRIPLE CHECK

                # For local search the per-chunk M-step only sees the
                # cone-restricted rotation set (typically a few thousand
                # rotations per image with high overlap across the chunk)
                # rather than the full ~10⁶-rotation grid at healpix order
                # 5+. Sizing the batch by the full grid produces ibs ≈ 5
                # at order 5 → chunks of 5 images → ~500 chunks per half
                # → ~7 hours per iter on the 5k benchmark.
                #
                # Estimate the per-image cone size from
                #     fraction = (sigma_cutoff * biggest_sigma / pi)^2
                # which is the spherical cap area as a fraction of the
                # full SO(3) volume (good to within ~30% for reasonable
                # cones). Use that to compute an effective rotation count
                # equal to ``chunk_size * cone_size``, with a safety
                # factor of 2x for cone-overlap inefficiency.
                _biggest_sigma = float(max(sigma_rot, sigma_psi))
                _cone_radius = 3.0 * _biggest_sigma  # sigma_cutoff=3.0
                _cone_fraction = max(
                    (_cone_radius / float(np.pi)) ** 2,
                    1.0 / float(rotation_grid_size(local_search_order)),
                )
                _est_cone_rots = int(np.ceil(rotation_grid_size(local_search_order) * _cone_fraction))
                # Per-chunk effective rotations ≈ 2 * cone_size
                # (after dedup of overlapping cones).
                _eff_n_rot = max(64, 2 * _est_cone_rots)
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    _eff_n_rot,
                    current_translations.shape[0],
                )
                if local_engine != "grouped_union":
                    # The new per-image local engine is not yet batch-equivalent
                    # to the grouped-union scorer for all late-iteration cases.
                    # Keep exact local buckets to one image until the
                    # multi-image batching path is proven parity-safe.
                    safe_ibs = 1
                logger.info(
                    "Local search batch sizing: cone_radius=%.3f rad "
                    "(%.2f deg), est_cone_rots=%d, eff_n_rot=%d "
                    "→ image_batch_size=%d, rotation_block_size=%d",
                    _cone_radius,
                    np.rad2deg(_cone_radius),
                    _est_cone_rots,
                    _eff_n_rot,
                    safe_ibs,
                    safe_rbs,
                )
                translation_prior_reference_translations = np.asarray(current_translations, dtype=np.float32)
                if local_search_translation_prior_mode == "coarse":
                    if _replay_prior_translations is not None:
                        translation_prior_reference_translations = np.asarray(
                            _replay_prior_translations, dtype=np.float32
                        )
                    else:
                        translation_prior_reference_translations = np.asarray(base_translations, dtype=np.float32)
                    logger.info(
                        "RELION mode: local translation prior uses coarse base grid (n=%d) while scoring perturbed translations",
                        translation_prior_reference_translations.shape[0],
                    )
                grouped_local_profile_k = None
                grouped_outputs = _run_local_search_iteration(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance,
                    previous_best_rotation_eulers[k],
                    local_search_rotations,
                    local_search_rotation_eulers,
                    local_search_order,
                    sigma_rot,
                    sigma_psi,
                    current_translations,
                    trans_prior_center,
                    current_sigma_offset_angstrom,
                    current_translation_range,
                    disc_type,
                    image_batch_size=safe_ibs,
                    rotation_block_size=safe_rbs,
                    current_size=cs_for_engine,
                    accumulate_noise=True,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    reconstruction_padding_factor=PADDING_FACTOR,
                    use_float64_scoring=True,
                    use_float64_projections=False,
                    do_gridding_correction=True,
                    square_window=False,
                    half_spectrum_scoring=True,
                    image_corrections=image_corrections_per_half[k],
                    scale_corrections=scale_corrections_per_half[k],
                    image_pre_shifts=translation_search_base,
                    score_with_masked_images=True,
                    return_profile=collect_local_search_profile,
                    sparse_pass2=True,
                    disable_adjoint_y=disable_adjoint_y,
                    disable_adjoint_ctf=disable_adjoint_ctf,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    local_engine=local_engine,
                    translation_prior_mode=local_search_translation_prior_mode,
                    translation_prior_reference_translations=translation_prior_reference_translations,
                )
                if len(grouped_outputs) == 6:
                    Ft_y_k, Ft_ctf_k, ha_k, em_stats_k, noise_stats_k, grouped_local_profile_k = grouped_outputs
                else:
                    Ft_y_k, Ft_ctf_k, ha_k, em_stats_k, noise_stats_k = grouped_outputs
                noise_stats_per_half[k] = noise_stats_k
                pose_rotations[k] = None
                coarse_ha[k] = ha_k
                if save_intermediates_dir is not None and grouped_local_profile_k is not None:
                    np.savez_compressed(
                        os.path.join(
                            save_intermediates_dir,
                            f"it{iteration:03d}_half{k + 1}_local_profile.npz",
                        ),
                        **grouped_local_profile_k,
                    )

            elif use_adaptive:
                # --- PASS 1: Coarse significance pruning ---
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                )

                t_pass1 = time.time()
                sig_rot_any, n_sig_batch, ha_coarse, sig_sample_indices = _compute_significance_batched(
                    experiment_datasets[k],
                    means[k],
                    noise_variance,
                    effective_rotations,
                    current_translations,
                    disc_type,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    image_batch_size=safe_ibs,
                    rotation_block_size=safe_rbs,
                    current_size=coarse_cs,
                    score_with_masked_images=True,
                    return_significant_sample_indices=True,
                    rotation_log_prior=rotation_log_prior,
                    translation_log_prior=translation_log_prior,
                    image_corrections=image_corrections_per_half[k],
                    scale_corrections=scale_corrections_per_half[k],
                    image_pre_shifts=translation_search_base,
                    half_spectrum_scoring=True,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    use_float64_scoring=True,
                )
                total_coarse_samples = int(
                    effective_rotations.shape[0] * current_translations.shape[0],
                )
                adaptive_pass1_diag[k] = {
                    "n_significant_per_image": np.asarray(n_sig_batch, dtype=np.int32),
                    "significant_rotation_union_mask": np.asarray(sig_rot_any, dtype=bool),
                    "coarse_hard_assignment": np.asarray(ha_coarse, dtype=np.int32),
                    "coarse_size": -1 if coarse_cs is None else int(coarse_cs),
                    "total_coarse_samples": total_coarse_samples,
                    "significant_rotation_union_count": int(np.sum(sig_rot_any)),
                }
                n_sig_total = int(np.sum(sig_rot_any))
                dt_pass1 = time.time() - t_pass1

                logger.info(
                    "Pass 1 (half %d): %d / %d significant coarse rotations in %.1fs (median n_sig/image=%d)",
                    k,
                    n_sig_total,
                    effective_rotations.shape[0],
                    dt_pass1,
                    int(np.median(n_sig_batch)),
                )

                skip_pass2, sig_fraction = should_skip_adaptive_pass2(
                    n_sig_batch,
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                    threshold=adaptive_pass2_skip_threshold,
                )

                if skip_pass2:
                    logger.info(
                        "Pass 2 skipped (half %d): mean significant fraction=%.3f >= %.3f; "
                        "running single-pass full-resolution E+M",
                        k,
                        sig_fraction,
                        ADAPTIVE_PASS2_MAX_SIGNIFICANT_FRACTION,
                    )
                    _, ha_k, Ft_y_k, Ft_ctf_k, em_stats_k, noise_stats_k = run_em(
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
                        translation_log_prior=translation_log_prior,
                        score_with_masked_images=True,
                        return_stats=True,
                        accumulate_noise=True,
                        half_spectrum_scoring=True,
                        projection_padding_factor=PROJECTION_PADDING_FACTOR,
                        reconstruction_padding_factor=PADDING_FACTOR,
                        image_corrections=image_corrections_per_half[k],
                        scale_corrections=scale_corrections_per_half[k],
                        image_pre_shifts=translation_search_base,
                        translation_prior_centers=trans_prior_center,
                        use_float64_scoring=True,
                        use_float64_projections=False,
                        do_gridding_correction=True,
                        square_window=False,
                        sparse_pass2=False,
                        disable_adjoint_y=disable_adjoint_y,
                        disable_adjoint_ctf=disable_adjoint_ctf,
                        relion_firstiter_score_mode=(
                            "normalized_cc"
                            if (relion_firstiter_cc_this_iter or first_iter_normalized_cc_this_iter)
                            else "gaussian"
                        ),
                        relion_firstiter_winner_take_all=(
                            relion_firstiter_cc_this_iter or first_iter_hard_reconstruction_this_iter
                        ),
                    )
                    noise_stats_per_half[k] = noise_stats_k
                    pose_rotations[k] = effective_rotations
                    pose_rotation_eulers[k] = effective_rotation_eulers
                    pose_translations[k] = np.asarray(current_translations, dtype=np.float32)
                    coarse_ha[k] = ha_k
                elif np.all(np.asarray(n_sig_batch) == total_coarse_samples):
                    # Exact early-iteration fast path: if every coarse sample is
                    # significant for every image, sparse per-image pass 2 is
                    # equivalent to one shared dense oversampled pass.
                    t_pass2 = time.time()
                    pass2_outputs = compute_pass2_stats(
                        experiment_datasets[k],
                        means[k],
                        mean_variance,
                        noise_variance,
                        effective_rotations,
                        current_translations,
                        np.ones(effective_rotations.shape[0], dtype=bool),
                        current_nside_level,
                        disc_type,
                        oversampling_order=state.adaptive_oversampling,
                        current_size=cs_for_engine,
                        translation_step=state.translation_step,
                        rotation_log_prior=rotation_log_prior,
                        translation_log_prior=translation_log_prior,
                        score_with_masked_images=True,
                        return_stats=True,
                        accumulate_noise=True,
                        half_spectrum_scoring=True,
                        projection_padding_factor=PROJECTION_PADDING_FACTOR,
                        reconstruction_padding_factor=PADDING_FACTOR,
                        image_corrections=image_corrections_per_half[k],
                        scale_corrections=scale_corrections_per_half[k],
                        image_pre_shifts=translation_search_base,
                        use_float64_scoring=True,
                        random_perturbation=random_perturbation,
                    )
                    Ft_y_k, Ft_ctf_k, ha_k, oversampled_rots_k, em_stats_k, noise_stats_k = pass2_outputs
                    noise_stats_per_half[k] = noise_stats_k
                    dt_pass2 = time.time() - t_pass2
                    logger.info(
                        "Pass 2 dense exact (half %d): %.1fs using full oversampled grid",
                        k,
                        dt_pass2,
                    )
                    pose_rotations[k] = np.asarray(oversampled_rots_k, dtype=np.float32)
                    pose_rotation_eulers[k] = utils.R_to_relion(
                        np.asarray(oversampled_rots_k),
                        degrees=True,
                    ).astype(np.float32)
                    oversampled_translations, _ = get_oversampled_translation_grid(
                        np.asarray(current_translations, dtype=np.float32),
                        state.translation_step,
                        oversampling_order=state.adaptive_oversampling,
                    )
                    pose_translations[k] = np.asarray(
                        oversampled_translations,
                        dtype=np.float32,
                    )
                    coarse_ha[k] = ha_coarse
                else:
                    # --- Exact sparse pass 2 over significant coarse samples ---
                    t_pass2 = time.time()
                    (
                        Ft_y_k,
                        Ft_ctf_k,
                        ha_k,
                        best_rots_k,
                        best_trans_k,
                        _best_rot_indices_k,
                        em_stats_k,
                        noise_stats_k,
                    ) = compute_pass2_stats_sparse(
                        experiment_datasets[k],
                        means[k],
                        mean_variance,
                        noise_variance,
                        current_translations,
                        sig_sample_indices,
                        current_nside_level,
                        disc_type,
                        oversampling_order=state.adaptive_oversampling,
                        current_size=cs_for_engine,
                        translation_step=state.translation_step,
                        rotation_log_prior=rotation_log_prior,
                        translation_log_prior=translation_log_prior,
                        score_with_masked_images=True,
                        return_stats=True,
                        accumulate_noise=True,
                        half_spectrum_scoring=True,
                        projection_padding_factor=PROJECTION_PADDING_FACTOR,
                        reconstruction_padding_factor=PADDING_FACTOR,
                        image_corrections=image_corrections_per_half[k],
                        scale_corrections=scale_corrections_per_half[k],
                        image_pre_shifts=translation_search_base,
                        use_float64_scoring=True,
                        random_perturbation=random_perturbation,
                    )
                    noise_stats_per_half[k] = noise_stats_k
                    dt_pass2 = time.time() - t_pass2
                    logger.info(
                        "Pass 2 sparse (half %d): %.1fs",
                        k,
                        dt_pass2,
                    )
                    best_pose_rotations[k] = np.asarray(best_rots_k, dtype=np.float32)
                    best_pose_rotation_eulers[k] = utils.R_to_relion(
                        np.asarray(best_rots_k),
                        degrees=True,
                    ).astype(np.float32)
                    best_pose_translations[k] = np.asarray(best_trans_k, dtype=np.float32)
                    oversampled_translations, _ = get_oversampled_translation_grid(
                        np.asarray(current_translations, dtype=np.float32),
                        state.translation_step,
                        oversampling_order=state.adaptive_oversampling,
                    )
                    pose_translations[k] = np.asarray(
                        oversampled_translations,
                        dtype=np.float32,
                    )

                    # Store coarse-grid assignment from pass 1 for local search.
                    coarse_ha[k] = ha_coarse

                if iter_sig_counts is None:
                    iter_sig_counts = n_sig_batch
                else:
                    iter_sig_counts = np.concatenate([iter_sig_counts, n_sig_batch])

            elif use_global_significant_support:
                # --- SINGLE-PASS GLOBAL SIGNIFICANT SUPPORT (RELION os0 parity) ---
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                )
                t_pass1 = time.time()
                sig_rot_any, n_sig_batch, ha_coarse, sig_sample_indices = _compute_significance_batched(
                    experiment_datasets[k],
                    means[k],
                    noise_variance,
                    effective_rotations,
                    current_translations,
                    disc_type,
                    adaptive_fraction=adaptive_fraction,
                    max_significants=max_significants,
                    image_batch_size=safe_ibs,
                    rotation_block_size=safe_rbs,
                    current_size=cs_for_engine,
                    score_with_masked_images=True,
                    return_significant_sample_indices=True,
                    rotation_log_prior=rotation_log_prior,
                    translation_log_prior=translation_log_prior,
                    image_corrections=image_corrections_per_half[k],
                    scale_corrections=scale_corrections_per_half[k],
                    image_pre_shifts=translation_search_base,
                    half_spectrum_scoring=True,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    use_float64_scoring=True,
                )
                total_samples = int(effective_rotations.shape[0] * current_translations.shape[0])
                adaptive_pass1_diag[k] = {
                    "n_significant_per_image": np.asarray(n_sig_batch, dtype=np.int32),
                    "significant_rotation_union_mask": np.asarray(sig_rot_any, dtype=bool),
                    "coarse_hard_assignment": np.asarray(ha_coarse, dtype=np.int32),
                    "coarse_size": -1 if cs_for_engine is None else int(cs_for_engine),
                    "total_coarse_samples": total_samples,
                    "significant_rotation_union_count": int(np.sum(sig_rot_any)),
                }
                dt_pass1 = time.time() - t_pass1
                logger.info(
                    "Global significant support (half %d): median n_sig/image=%d, max=%d / %d in %.1fs",
                    k,
                    int(np.median(n_sig_batch)),
                    int(np.max(n_sig_batch)),
                    total_samples,
                    dt_pass1,
                )

                t_pass2 = time.time()
                (
                    Ft_y_k,
                    Ft_ctf_k,
                    ha_k,
                    best_rots_k,
                    best_trans_k,
                    _best_rot_indices_k,
                    em_stats_k,
                    noise_stats_k,
                ) = compute_pass2_stats_sparse(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance,
                    current_translations,
                    sig_sample_indices,
                    current_nside_level,
                    disc_type,
                    oversampling_order=0,
                    current_size=cs_for_engine,
                    translation_step=state.translation_step,
                    rotation_log_prior=rotation_log_prior,
                    translation_log_prior=translation_log_prior,
                    score_with_masked_images=True,
                    return_stats=True,
                    accumulate_noise=True,
                    half_spectrum_scoring=True,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    reconstruction_padding_factor=PADDING_FACTOR,
                    image_corrections=image_corrections_per_half[k],
                    scale_corrections=scale_corrections_per_half[k],
                    image_pre_shifts=translation_search_base,
                    use_float64_scoring=True,
                    random_perturbation=random_perturbation,
                    translation_prior_centers=trans_prior_center,
                )
                noise_stats_per_half[k] = noise_stats_k
                dt_pass2 = time.time() - t_pass2
                logger.info("Global significant support pass 2 (half %d): %.1fs", k, dt_pass2)

                best_pose_rotations[k] = np.asarray(best_rots_k, dtype=np.float32)
                best_pose_rotation_eulers[k] = utils.R_to_relion(
                    np.asarray(best_rots_k),
                    degrees=True,
                ).astype(np.float32)
                best_pose_translations[k] = np.asarray(best_trans_k, dtype=np.float32)
                pose_rotations[k] = effective_rotations
                pose_rotation_eulers[k] = effective_rotation_eulers
                pose_translations[k] = np.asarray(current_translations, dtype=np.float32)
                coarse_ha[k] = ha_coarse

                if iter_sig_counts is None:
                    iter_sig_counts = np.asarray(n_sig_batch, dtype=np.int32)
                else:
                    iter_sig_counts = np.concatenate([iter_sig_counts, np.asarray(n_sig_batch, dtype=np.int32)])

            else:
                # --- SINGLE-PASS E+M (no adaptive oversampling) ---
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                )
                _, ha_k, Ft_y_k, Ft_ctf_k, em_stats_k, noise_stats_k = run_em(
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
                    translation_log_prior=translation_log_prior,
                    score_with_masked_images=True,
                    return_stats=True,
                    accumulate_noise=True,
                    half_spectrum_scoring=True,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    reconstruction_padding_factor=PADDING_FACTOR,
                    image_corrections=image_corrections_per_half[k],
                    scale_corrections=scale_corrections_per_half[k],
                    image_pre_shifts=translation_search_base,
                    translation_prior_centers=trans_prior_center,
                    use_float64_scoring=True,
                    use_float64_projections=False,
                    do_gridding_correction=True,
                    square_window=False,
                    sparse_pass2=False,
                    disable_adjoint_y=disable_adjoint_y,
                    disable_adjoint_ctf=disable_adjoint_ctf,
                    relion_firstiter_score_mode=(
                        "normalized_cc"
                        if (relion_firstiter_cc_this_iter or first_iter_normalized_cc_this_iter)
                        else "gaussian"
                    ),
                    relion_firstiter_winner_take_all=(
                        relion_firstiter_cc_this_iter or first_iter_hard_reconstruction_this_iter
                    ),
                )
                noise_stats_per_half[k] = noise_stats_k
                pose_rotations[k] = effective_rotations
                pose_rotation_eulers[k] = effective_rotation_eulers
                pose_translations[k] = np.asarray(current_translations, dtype=np.float32)
                coarse_ha[k] = ha_k  # same grid, no oversampling

                # --- Manifest dump for deterministic replay (Phase 0.1) ---
                if save_intermediates_dir is not None:
                    _manifest_path = os.path.join(
                        save_intermediates_dir,
                        f"manifest_iter{iteration}_half{k}.npz",
                    )
                    _manifest = {
                        "effective_rotations": np.asarray(effective_rotations, dtype=np.float32),
                        "current_translations": np.asarray(current_translations, dtype=np.float32),
                        "rotation_log_prior": np.asarray(rotation_log_prior, dtype=np.float64)
                        if rotation_log_prior is not None
                        else np.array([]),
                        "translation_log_prior": np.asarray(translation_log_prior, dtype=np.float64)
                        if translation_log_prior is not None
                        else np.array([]),
                        "image_corrections": np.asarray(image_corrections_per_half[k], dtype=np.float64)
                        if image_corrections_per_half[k] is not None
                        else np.array([]),
                        "scale_corrections": np.asarray(scale_corrections_per_half[k], dtype=np.float64)
                        if scale_corrections_per_half[k] is not None
                        else np.array([]),
                        "image_pre_shifts": np.asarray(translation_search_base, dtype=np.float32)
                        if translation_search_base is not None
                        else np.array([]),
                        "absolute_previous_translations": np.asarray(previous_best_translations[k], dtype=np.float32)
                        if previous_best_translations[k] is not None
                        else np.array([]),
                        "mean_vol_ft": np.asarray(means[k]),
                        "mean_variance": np.asarray(mean_variance),
                        "noise_variance": np.asarray(noise_variance),
                        "current_size": np.int32(cs_for_engine) if cs_for_engine is not None else np.int32(-1),
                        "half_spectrum_scoring": np.bool_(True),
                        "use_float64_scoring": np.bool_(True),
                        "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
                        "reconstruction_padding_factor": np.int32(PADDING_FACTOR),
                        "score_with_masked_images": np.bool_(True),
                        "perturbation_instance": np.float64(random_perturbation),
                        "perturbation_factor": np.float64(perturb_factor),
                        "iteration": np.int32(iteration),
                        "half_index": np.int32(k),
                        "ave_Pmax": np.float64(float(np.mean(em_stats_k.max_posterior_per_image))),
                    }
                    np.savez(_manifest_path, **_manifest)
                    logger.info("Manifest dumped: %s", _manifest_path)

            # NOTE: means[k] reconstruction is DEFERRED until after the
            # low_resol_join_halves step below — we need both halves'
            # Ft_y / Ft_ctf accumulators in hand before we can average
            # the low-frequency shells across the two halves.
            hard_assignments[k] = ha_k
            max_posterior_per_half[k] = np.asarray(
                em_stats_k.max_posterior_per_image,
                dtype=np.float32,
            )
            rotation_posterior_per_half[k] = np.asarray(
                em_stats_k.rotation_posterior_sums,
                dtype=np.float32,
            )

            if k == 0:
                Ft_y_0, Ft_ctf_0 = Ft_y_k, Ft_ctf_k
            else:
                Ft_y_1, Ft_ctf_1 = Ft_y_k, Ft_ctf_k

            _parity_dump.collect_e_step(
                half=k,
                em_stats=em_stats_k,
                hard_assignment=ha_k,
                coarse_hard_assignment=coarse_ha[k],
                noise_stats=noise_stats_per_half[k],
                Ft_y=Ft_y_k,
                Ft_ctf=Ft_ctf_k,
                pose_rotation_eulers=pose_rotation_eulers[k],
                best_pose_rotation_eulers=best_pose_rotation_eulers[k],
                best_pose_translations=best_pose_translations[k],
                translation_search_base=translation_search_bases[k] if "translation_search_bases" in dir() else None,
            )

        # E-step + per-half M-step accumulators are now both populated.
        _parity_dump.mark_stage(iteration, "e_step")

        # --- RELION's --low_resol_join_halves: average the low-resolution
        # shells of the per-half Fourier accumulators between the two halves
        # BEFORE the Wiener solve. This forces the two half-maps to share
        # their low-frequency content, preventing them from diverging in
        # orientation space at SNR-poor low shells. RELION mirrors this in
        # ml_optimiser_mpi.cpp::joinTwoHalvesAtLowResolution; without it
        # recovar's iter-N FSC drops gradually from shell ~2 while RELION's
        # stays at 1.0 through shell 13 (= 40 A for a 128/4.25 dataset),
        # which directly translates to a ~5-shell deficit in
        # ``first_shell_below_0.5`` and a ~10-pixel/iter deficit in
        # ``current_size`` growth (the dominant convergence-speed gap
        # observed in the 2026-04 5k normalized parity benchmark).
        #
        # Use the previous iteration's resolution to cap the join radius
        # (so we never join shells beyond the actual resolution of the
        # map). Mirrors the ``XMIPP_MAX(low_resol_join_halves,
        # 1./mymodel.current_resolution)`` in RELION's source.
        prev_res_angstrom = None
        if pixel_resolutions:
            prev_pixel_res = pixel_resolutions[-1]
            if prev_pixel_res > 0:
                prev_res_angstrom = shell_index_to_resolution_angstrom(
                    prev_pixel_res,
                    grid_size,
                    cryo.voxel_size,
                )
        Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1 = regularization.join_halves_at_low_resolution(
            Ft_y_0,
            Ft_y_1,
            Ft_ctf_0,
            Ft_ctf_1,
            padded_volume_shape,
            cryo.voxel_size,
            grid_size,
            low_resol_join_halves_angstrom,
            current_resolution_angstrom=prev_res_angstrom,
        )

        # --- RELION-exact M-step ordering (auto-refine, split-half) ---
        # RELION (ml_optimiser_mpi.cpp:4031, 4091; backprojector.cpp:1044):
        #   1. compareTwoHalves() -> CURRENT iter's FSC from BPref accumulators
        #   2. maximization() -> updateSSNRarrays(THIS_ITER_FSC) -> tau2
        #   3. reconstruct(tau2) -> regularized half-map
        #
        # Recovar previously called compute_relion_tau2_from_weights with
        # fsc_history[-1] / init_fsc (PREVIOUS iter's FSC). At cold start
        # init_fsc is essentially zeros and at iter 2 prev-iter FSC is
        # poisoned (~0.999) by leakage of the under-regularized iter-1 maps,
        # which gives ssnr ≈ 999 → tau2 amplifies 1e6× → ave_Pmax collapse.
        # Algorithm doc: docs/math/relion_updateSSNR_algorithm_2026_04_25.md
        #
        # Compute unregularized half-maps and CURRENT iter FSC FIRST, then
        # derive tau2 from that fresh FSC, then the regularized Wiener solve.
        _t_unreg_first = time.time()
        _unreg_means_for_fsc = []
        for k_local in range(2):
            Ft_y_k_l = Ft_y_0 if k_local == 0 else Ft_y_1
            Ft_ctf_k_l = Ft_ctf_0 if k_local == 0 else Ft_ctf_1
            _unreg_means_for_fsc.append(
                _reconstruct_volume_eager(
                    Ft_ctf_k_l,
                    Ft_y_k_l,
                    volume_shape,
                    PADDING_FACTOR,
                    tau=None,
                    tau2_fudge=tau2_fudge,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                ).reshape(-1)
            )
        # Sign-align so FSC sees the same orientation downstream maps will use.
        for k_half in range(2):
            _unreg_means_for_fsc[k_half], _ = _align_fourier_volume_sign_to_reference(
                _unreg_means_for_fsc[k_half],
                previous_means[k_half] if "previous_means" in dir() else None,
                volume_shape,
            )
        current_iter_fsc = regularization.get_fsc_gpu(
            _unreg_means_for_fsc[0],
            _unreg_means_for_fsc[1],
            volume_shape,
        )
        logger.info(
            "Computed iter-%d FSC for tau2 (RELION order): %.1fs",
            iteration + 1,
            time.time() - _t_unreg_first,
        )

        mean_signal_variance, _, tau2_update_details = regularization.compute_relion_tau2_from_weights(
            Ft_ctf_0,
            Ft_ctf_1,
            current_iter_fsc,
            volume_shape,
            tau2_fudge=tau2_fudge,
            padding_factor=PADDING_FACTOR,
            return_details=True,
        )
        logger.info(
            "tau2 update from THIS-iter FSC: old_max=%.4e new_max=%.4e",
            float(jnp.max(jnp.abs(mean_variance))),
            float(jnp.max(jnp.abs(mean_signal_variance))),
        )
        mean_variance = mean_signal_variance

        # --- Free previous-iteration means to reclaim GPU memory ---
        previous_means = [np.asarray(mean).copy() if mean is not None else None for mean in means]
        for k in range(2):
            means[k] = None

        # --- Now reconstruct the regularized per-half means from the
        # (post-join) Ft_y / Ft_ctf accumulators.  When PADDING_FACTOR > 1,
        # the engine already backprojected into a (pf*N)³ grid.
        # Use eager (non-JIT) reconstruction to avoid ~30 min XLA compile
        # overhead for the monolithic 256³ graph in post_process_from_filter_v2.
        _t_recon = time.time()
        for k in range(2):
            Ft_y_k_local = Ft_y_0 if k == 0 else Ft_y_1
            Ft_ctf_k_local = Ft_ctf_0 if k == 0 else Ft_ctf_1
            means[k] = _reconstruct_volume_eager(
                Ft_ctf_k_local,
                Ft_y_k_local,
                volume_shape,
                PADDING_FACTOR,
                tau=mean_variance,
                tau2_fudge=tau2_fudge,
                projection_padding_factor=PROJECTION_PADDING_FACTOR,
            ).reshape(-1)

            # RELION's solventFlatten (ml_optimiser.cpp:5469): mask the
            # reconstructed reference outside particle_diameter to remove
            # solvent noise before the next E-step's projections.
            if particle_diameter_ang is not None and particle_diameter_ang > 0:
                flatten_radius = particle_diameter_ang / (2.0 * cryo.voxel_size)
                vol_real = fourier_transform_utils.get_idft3(means[k].reshape(volume_shape))
                vol_real, _ = mask.soft_mask_outside_map(
                    vol_real,
                    radius=flatten_radius,
                    cosine_width=RELION_WIDTH_MASK_EDGE,
                )
                means[k] = fourier_transform_utils.get_dft3(vol_real).reshape(-1)
            if relion_firstiter_cc_this_iter:
                means[k] = _apply_relion_initial_lowpass_filter(
                    means[k],
                    volume_shape,
                    cryo.voxel_size,
                    relion_firstiter_ini_high_angstrom,
                    filter_edgewidth=RELION_WIDTH_MASK_EDGE,
                )
        if relion_firstiter_cc_this_iter and relion_firstiter_ini_high_angstrom is not None:
            logger.info(
                "RELION iter-1 CC emulation: reapplying ini_high low-pass filter at %.2f A",
                float(relion_firstiter_ini_high_angstrom),
            )
        logger.info("Regularized reconstruction (2 halves + flatten): %.1fs", time.time() - _t_recon)
        _parity_dump.mark_stage(iteration, "recon")

        significant_counts.append(iter_sig_counts)

        if (
            not use_local
            and all(rot_sum is not None for rot_sum in rotation_posterior_per_half)
            and effective_rotations.shape[0] == rotation_grid_size(current_healpix_order)
        ):
            global_direction_prior = collapse_rotation_posterior_to_direction_prior(
                np.asarray(rotation_posterior_per_half[0], dtype=np.float64)
                + np.asarray(rotation_posterior_per_half[1], dtype=np.float64),
                current_healpix_order,
            )
            global_direction_prior_order = current_healpix_order

        # --- Combined Fourier weights for data_vs_prior at next iteration ---
        Ft_ctf_combined = Ft_ctf_0 + Ft_ctf_1

        # --- Compute unregularized half-maps for FSC and prior ---
        _t_unreg = time.time()
        unreg_means = [
            _reconstruct_volume_eager(
                Ft_ctf_0,
                Ft_y_0,
                volume_shape,
                PADDING_FACTOR,
                tau=None,
                tau2_fudge=tau2_fudge,
                projection_padding_factor=PROJECTION_PADDING_FACTOR,
            ),
            _reconstruct_volume_eager(
                Ft_ctf_1,
                Ft_y_1,
                volume_shape,
                PADDING_FACTOR,
                tau=None,
                tau2_fudge=tau2_fudge,
                projection_padding_factor=PROJECTION_PADDING_FACTOR,
            ),
        ]
        for k in range(2):
            means[k], sign_flipped = _align_fourier_volume_sign_to_reference(means[k], previous_means[k], volume_shape)
            if sign_flipped:
                unreg_means[k] = -unreg_means[k]
                logger.info("Aligned half-%d volume sign to the previous reference", k + 1)
        logger.info("Unregularized reconstruction (2 halves): %.1fs", time.time() - _t_unreg)

        # FSC was already computed above in the RELION-exact ordering block
        # (current_iter_fsc) and used to derive tau2 BEFORE the Wiener solve.
        # Reuse it here — recomputing would give the same value (same
        # underlying unreg accumulators).
        fsc = current_iter_fsc
        fsc_history.append(fsc)
        _parity_dump.mark_stage(iteration, "fsc")

        # --- Save intermediate volumes if requested ---
        if save_intermediates_dir is not None:
            from recovar.output.output import save_volume

            os.makedirs(save_intermediates_dir, exist_ok=True)
            np.save(os.path.join(save_intermediates_dir, f"it{iteration:03d}_Ft_y_0.npy"), np.asarray(Ft_y_0))
            np.save(os.path.join(save_intermediates_dir, f"it{iteration:03d}_Ft_y_1.npy"), np.asarray(Ft_y_1))
            np.save(os.path.join(save_intermediates_dir, f"it{iteration:03d}_Ft_ctf_0.npy"), np.asarray(Ft_ctf_0))
            np.save(os.path.join(save_intermediates_dir, f"it{iteration:03d}_Ft_ctf_1.npy"), np.asarray(Ft_ctf_1))
            for k_half in range(2):
                save_volume(
                    np.asarray(means[k_half]).reshape(-1),
                    os.path.join(
                        save_intermediates_dir,
                        f"it{iteration:03d}_half{k_half + 1}_reg",
                    ),
                    volume_shape=volume_shape,
                    from_ft=True,
                    voxel_size=cryo.voxel_size,
                )
                save_volume(
                    np.asarray(unreg_means[k_half]).reshape(-1),
                    os.path.join(
                        save_intermediates_dir,
                        f"it{iteration:03d}_half{k_half + 1}_unreg",
                    ),
                    volume_shape=volume_shape,
                    from_ft=True,
                    voxel_size=cryo.voxel_size,
                )
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
                            f"it{iteration:03d}_ha_half{k_half + 1}.npy",
                        ),
                        hard_assignments[k_half],
                    )
            # Save per-iteration metadata
            iter_meta = {
                "iteration": iteration,
                "current_size": int(cs),
                "n_rotations": int(
                    rotation_grid_size(local_search_order) if use_local else effective_rotations.shape[0]
                ),
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
                (np.asarray(effective_rotations) if not use_local else np.empty((0, 3, 3), dtype=np.float32)),
            )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_translations.npy"),
                np.asarray(current_translations),
            )
            for k_half in range(2):
                if coarse_ha[k_half] is not None:
                    np.save(
                        os.path.join(
                            save_intermediates_dir,
                            f"it{iteration:03d}_coarse_ha_half{k_half + 1}.npy",
                        ),
                        np.asarray(coarse_ha[k_half], dtype=np.int32),
                    )
                pass1_diag = adaptive_pass1_diag[k_half]
                if pass1_diag is not None:
                    np.savez_compressed(
                        os.path.join(
                            save_intermediates_dir,
                            f"it{iteration:03d}_half{k_half + 1}_pass1_diag.npz",
                        ),
                        **pass1_diag,
                    )
            logger.info(
                "Saved intermediate volumes to %s (iteration %d)",
                save_intermediates_dir,
                iteration,
            )

        # --- Compute ave_Pmax from the actual E-step maxima ---
        if any(pmax is None for pmax in max_posterior_per_half):
            raise RuntimeError(
                "RELION mode expected per-image posterior maxima from the EM engine",
            )
        combined_max_posterior = np.concatenate(
            [np.asarray(pmax, dtype=np.float32) for pmax in max_posterior_per_half],
            axis=0,
        )
        ave_pmax = float(np.mean(combined_max_posterior))
        ave_Pmax_trajectory.append(ave_pmax)
        pmax_per_image_history.append(combined_max_posterior.copy())

        # --- Track per-image best assignments for convergence detection ---
        # Combine both half-sets' assignments into a single array for
        # update_refinement_state.  Use coarse_ha (indexed into
        # effective_rotations) for consistent convergence tracking.
        current_combined_ha = np.concatenate(
            [np.asarray(ha, dtype=np.int32) for ha in coarse_ha],
            axis=0,
        )
        if all(ha is not None for ha in previous_assignments):
            previous_combined_ha = np.concatenate(
                [np.asarray(ha, dtype=np.int32) for ha in previous_assignments],
                axis=0,
            )
        else:
            previous_combined_ha = None

        # tau2 was already updated BEFORE the Wiener solve (matching RELION's
        # reconstruct() which calls updateSSNRarrays before the filter).

        # --- Resolution from updated FSC-derived SSNR (RELION auto-refine) ---
        # Matches RELION updateSSNRarrays at backprojector.cpp:1117-1123:
        # data_vs_prior[i] = tau2_fudge * fsc / (1 - fsc), with fsc clamped
        # to [0.001, 0.999] inside fsc_to_relion_ssnr.
        dvp_iter = np.asarray(fsc, dtype=np.float32).copy()
        if cs < grid_size:
            dvp_iter[min(len(dvp_iter), cs // 2) :] = 0.0
        dvp_iter = np.asarray(
            fsc_to_relion_ssnr(dvp_iter, tau2_fudge=tau2_fudge),
        )
        dvp_res_shell = resolution_from_data_vs_prior(
            dvp_iter,
            allow_high_res_recovery=True,
        )
        pixel_res = float(dvp_res_shell)
        pixel_resolutions.append(pixel_res)

        # --- Update poses and noise ---
        # Snapshot the iter K-1 best rotations / translations BEFORE the
        # loop overwrites them, so update_refinement_state below can compute
        # the RELION-exact change metrics (B3) between iter K-1 and iter K.
        prior_iter_best_rotations = [
            np.asarray(rot).copy() if rot is not None else None for rot in previous_best_rotations
        ]
        prior_iter_best_translations = [
            np.asarray(trans).copy() if trans is not None else None for trans in previous_best_translations
        ]
        new_iter_best_rotations = [None, None]
        new_iter_best_rotation_eulers = [None, None]
        new_iter_best_translations = [None, None]
        for k in range(2):
            if best_pose_rotations[k] is not None:
                best_rots = np.asarray(best_pose_rotations[k], dtype=np.float32)
                best_eulers = (
                    np.asarray(best_pose_rotation_eulers[k], dtype=np.float32)
                    if best_pose_rotation_eulers[k] is not None
                    else utils.R_to_relion(best_rots, degrees=True).astype(np.float32)
                )
                best_trans = np.asarray(best_pose_translations[k], dtype=np.float32)
            elif use_local:
                rot_idx = hard_assignments[k] // current_translations.shape[0]
                trans_idx = hard_assignments[k] % current_translations.shape[0]
                if local_search_rotations is None:
                    raise ValueError("Local-search hard assignments require the fine local-search grid")
                best_rots = np.asarray(local_search_rotations, dtype=np.float32)[rot_idx]
                if local_search_rotation_eulers is not None:
                    best_eulers = np.asarray(local_search_rotation_eulers, dtype=np.float32)[rot_idx]
                else:
                    best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(np.float32)
                best_trans = np.asarray(current_translations)[trans_idx]
            else:
                # Global search uses the dense grid in pose_rotations[k].
                rot_idx = hard_assignments[k] // current_translations.shape[0]
                best_rots, best_trans = hard_assignment_idx_to_pose(
                    hard_assignments[k],
                    pose_rotations[k],
                    pose_translations[k],
                )
                if pose_rotation_eulers[k] is not None:
                    best_eulers = np.asarray(pose_rotation_eulers[k], dtype=np.float32)[rot_idx]
                else:
                    best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(np.float32)
            new_iter_best_rotations[k] = np.asarray(best_rots, dtype=np.float32)
            new_iter_best_rotation_eulers[k] = np.asarray(best_eulers, dtype=np.float32)
            # When image_pre_shifts is used, best_trans is relative to the
            # previous absolute pre-shift base. Store the total (absolute) translation
            # so the next iteration pre-centers by the updated offset.
            total_trans = np.asarray(best_trans, dtype=np.float32)
            if translation_search_bases[k] is not None:
                total_trans = total_trans + translation_search_bases[k]
            new_iter_best_translations[k] = total_trans
            previous_best_rotations[k] = new_iter_best_rotations[k]
            previous_best_rotation_eulers[k] = new_iter_best_rotation_eulers[k]
            previous_best_translations[k] = new_iter_best_translations[k]
            experiment_datasets[k].update_poses(best_rots, total_trans)

        try:
            best_rotation_eulers_history.append(
                np.concatenate(new_iter_best_rotation_eulers, axis=0).astype(np.float32)
            )
            best_translations_history.append(np.concatenate(new_iter_best_translations, axis=0).astype(np.float32))
        except (ValueError, TypeError):
            best_rotation_eulers_history.append(None)
            best_translations_history.append(None)

        if save_intermediates_dir is not None:
            for k_half in range(2):
                np.save(
                    os.path.join(
                        save_intermediates_dir,
                        f"it{iteration:03d}_best_rotation_eulers_half{k_half + 1}.npy",
                    ),
                    np.asarray(new_iter_best_rotation_eulers[k_half], dtype=np.float32),
                )
                np.save(
                    os.path.join(
                        save_intermediates_dir,
                        f"it{iteration:03d}_best_translations_half{k_half + 1}.npy",
                    ),
                    np.asarray(new_iter_best_translations[k_half], dtype=np.float32),
                )
                if prior_iter_best_rotations[k_half] is not None:
                    np.save(
                        os.path.join(
                            save_intermediates_dir,
                            f"it{iteration:03d}_prev_rotation_matrices_half{k_half + 1}.npy",
                        ),
                        np.asarray(prior_iter_best_rotations[k_half], dtype=np.float32),
                    )
                if prior_iter_best_translations[k_half] is not None:
                    np.save(
                        os.path.join(
                            save_intermediates_dir,
                            f"it{iteration:03d}_prev_translations_half{k_half + 1}.npy",
                        ),
                        np.asarray(prior_iter_best_translations[k_half], dtype=np.float32),
                    )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_effective_rotations.npy"),
                np.asarray(effective_rotations, dtype=np.float32),
            )
            np.save(
                os.path.join(save_intermediates_dir, f"it{iteration:03d}_effective_rotation_eulers.npy"),
                np.asarray(effective_rotation_eulers, dtype=np.float32),
            )

        # --- RELION-exact change tracking inputs (B3 / B4) ---
        # Combine both half-sets in the same image order as
        # current_combined_ha. RELION's monitorHiddenVariableChanges sums
        # over all particles, so the per-half order is irrelevant for the
        # mean -- but we keep the half-0-then-half-1 convention for
        # consistency with the rest of the loop.
        try:
            current_rotation_matrices_combined = np.concatenate(
                new_iter_best_rotations,
                axis=0,
            ).astype(np.float64)
            current_translations_pixel_combined = np.concatenate(
                new_iter_best_translations,
                axis=0,
            ).astype(np.float64)
        except (ValueError, TypeError):
            current_rotation_matrices_combined = None
            current_translations_pixel_combined = None
        if all(rot is not None for rot in prior_iter_best_rotations):
            try:
                previous_rotation_matrices_combined = np.concatenate(
                    prior_iter_best_rotations,
                    axis=0,
                ).astype(np.float64)
                previous_translations_pixel_combined = np.concatenate(
                    prior_iter_best_translations,
                    axis=0,
                ).astype(np.float64)
            except (ValueError, TypeError):
                previous_rotation_matrices_combined = None
                previous_translations_pixel_combined = None
        else:
            previous_rotation_matrices_combined = None
            previous_translations_pixel_combined = None

        # RELION-style posterior-weighted noise update. Sums the wsum/img_power
        # accumulators from both half-sets and normalizes via the M-step formula.
        if noise_stats_per_half[0] is None or noise_stats_per_half[1] is None:
            raise RuntimeError(
                "RELION mode expected per-half NoiseStats from the EM engine; "
                "ensure accumulate_noise=True is plumbed through pass 2.",
            )
        if relion_firstiter_cc_this_iter:
            noise_from_res = np.asarray(previous_noise_radial, dtype=np.float64)
            logger.info(
                "RELION iter-1 CC emulation: keeping previous sigma2_noise (skip first-iter noise update)",
            )
        else:
            wsum_combined = np.asarray(noise_stats_per_half[0].wsum_sigma2_noise, dtype=np.float64) + np.asarray(
                noise_stats_per_half[1].wsum_sigma2_noise, dtype=np.float64
            )
            img_power_combined = np.asarray(noise_stats_per_half[0].wsum_img_power, dtype=np.float64) + np.asarray(
                noise_stats_per_half[1].wsum_img_power, dtype=np.float64
            )
            sumw_combined = noise_stats_per_half[0].sumw + noise_stats_per_half[1].sumw
            noise_from_res = noise.normalize_wsum_to_sigma2_noise(
                wsum_combined,
                img_power_combined,
                sumw_combined,
                cryo.image_shape,
            )

            # Log per-shell noise comparison (first 10 shells) for convergence diagnostics
            old_noise_radial = previous_noise_radial
            n_log = min(10, len(noise_from_res), len(old_noise_radial))
            logger.info(
                "Noise update per shell (first %d): old=[%s] new=[%s]",
                n_log,
                ", ".join(f"{float(x):.3e}" for x in old_noise_radial[:n_log]),
                ", ".join(f"{float(x):.3e}" for x in noise_from_res[:n_log]),
            )

            previous_noise_radial = noise_from_res
            noise_variance = noise.make_radial_noise(noise_from_res, cryo.image_shape)
            _parity_dump.mark_stage(iteration, "noise_update")

        # Save per-iter per-shell sigma2 (after this iter's noise update) and
        # the exact shell-wise tau2 ingredients used in the Wiener update.
        noise_radial_trajectory.append(np.asarray(noise_from_res, dtype=np.float64))
        if tau2_update_details is not None:
            tau2_radial_trajectory.append(np.asarray(tau2_update_details["prior_shells"], dtype=np.float64))
            tau2_sigma2_trajectory.append(np.asarray(tau2_update_details["sigma2_shells"], dtype=np.float64))
            tau2_avg_weight_trajectory.append(np.asarray(tau2_update_details["avg_weight_shells"], dtype=np.float64))
            tau2_shell_sum_trajectory.append(np.asarray(tau2_update_details["shell_sum"], dtype=np.float64))
            tau2_shell_count_trajectory.append(np.asarray(tau2_update_details["shell_count"], dtype=np.float64))
            tau2_fsc_used_trajectory.append(np.asarray(tau2_update_details["fsc_shells"], dtype=np.float64))
            tau2_ssnr_trajectory.append(np.asarray(tau2_update_details["ssnr_shells"], dtype=np.float64))
        else:
            tau2_radial_trajectory.append(None)
            tau2_sigma2_trajectory.append(None)
            tau2_avg_weight_trajectory.append(None)
            tau2_shell_sum_trajectory.append(None)
            tau2_shell_count_trajectory.append(None)
            tau2_fsc_used_trajectory.append(None)
            tau2_ssnr_trajectory.append(None)

        # --- Update convergence state ---
        # This checks assignment changes, resolution stalls, and may trigger
        # angular step refinement or convergence.
        n_rot_current = rotation_grid_size(local_search_order) if use_local else effective_rotations.shape[0]
        n_trans_current = current_translations.shape[0]

        # ``update_refinement_state`` expects ``new_resolution`` in
        # Angstroms (lower = better resolution), matching RELION's
        # ``mymodel.current_resolution``.  Convert from the shell index
        # ``pixel_res`` to Å here so the resol_gain stall detection
        # compares apples to apples (not shell-vs-shell with the wrong
        # sign).
        new_res_angstrom = shell_index_to_resolution_angstrom(
            pixel_res,
            cryo.image_shape[0],
            cryo.voxel_size,
        )

        # RELION's calculateExpectedAngularErrors (ml_optimiser.cpp:9534)
        iter_acc_rot = None
        if iter_sig_counts is not None and len(iter_sig_counts) > 0:
            iter_acc_rot, _ = calculate_expected_angular_errors(
                state.healpix_order,
                iter_sig_counts,
                n_translations=n_trans_current,
            )
            logger.info(
                "acc_rot=%.3f deg (from %d images, mean n_sig=%.1f)",
                iter_acc_rot,
                len(iter_sig_counts),
                float(np.mean(iter_sig_counts)),
            )

        state = update_refinement_state(
            state,
            current_assignments=current_combined_ha,
            previous_assignments=previous_combined_ha,
            n_rotations=n_rot_current,
            n_translations=n_trans_current,
            translations=np.asarray(current_translations),
            new_resolution=new_res_angstrom,
            max_posterior_per_image=combined_max_posterior,
            acc_rot=iter_acc_rot,
            current_rotation_matrices=current_rotation_matrices_combined,
            previous_rotation_matrices=previous_rotation_matrices_combined,
            current_translations_pixel=current_translations_pixel_combined,
            previous_translations_pixel=previous_translations_pixel_combined,
            voxel_size_angstrom=float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
        )

        # Track frac_changed for local search fallback
        from recovar.em.dense_single_volume.helpers.convergence import compute_assignment_changes

        frac_changed = compute_assignment_changes(
            current_combined_ha,
            previous_combined_ha,
            n_rot_current,
            n_trans_current,
            current_healpix_order,
        )
        state._last_frac_changed = frac_changed
        frac_changed_trajectory.append(float(frac_changed))

        # --- C1 (RELION-parity): update sigma2_offset from data ---
        # Prefer RELION's posterior-weighted sufficient statistic:
        #   sigma2_offset_new = wsum_sigma2_offset / (2 * sum_weight)
        # for 2D single-particle data. Fall back to the older hard-assignment
        # proxy only when a path does not propagate the full posterior moment.
        sigma2_offset_wsum = 0.0
        sigma2_offset_sumw = 0.0
        for stats_k in noise_stats_per_half:
            if stats_k is None:
                continue
            sigma2_offset_wsum += float(getattr(stats_k, "wsum_sigma2_offset", 0.0))
            sigma2_offset_sumw += float(getattr(stats_k, "sumw", 0.0))
        if sigma2_offset_wsum > 0.0 and sigma2_offset_sumw > 0.0:
            voxel_size_angstrom = float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0)
            min_sigma2_angstrom2 = 2.0 * voxel_size_angstrom**2
            sigma2_offset_angstrom2 = max(
                sigma2_offset_wsum / (2.0 * sigma2_offset_sumw),
                min_sigma2_angstrom2,
            )
            current_sigma_offset_angstrom = float(np.sqrt(sigma2_offset_angstrom2))
            logger.info(
                "C1: sigma_offset updated %.3f Å from posterior variance (clamp sigma^2 >= %.3f Å^2)",
                current_sigma_offset_angstrom,
                min_sigma2_angstrom2,
            )
        else:
            new_sigma_offset_angstrom = state.current_changes_optimal_offsets_angstrom
            if np.isfinite(new_sigma_offset_angstrom) and new_sigma_offset_angstrom > 0:
                min_sigma_pixels = float(np.sqrt(2.0))  # RELION min_sigma2_offset = 2
                min_sigma_angstrom = min_sigma_pixels * float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0)
                current_sigma_offset_angstrom = max(
                    float(new_sigma_offset_angstrom),
                    min_sigma_angstrom,
                )
                logger.info(
                    "C1 fallback: sigma_offset updated %.3f Å from hard assignments (clamp >= %.3f Å)",
                    current_sigma_offset_angstrom,
                    min_sigma_angstrom,
                )
        sigma_offset_trajectory.append(float(current_sigma_offset_angstrom))
        acc_rot_trajectory.append(float(iter_acc_rot) if iter_acc_rot is not None else np.nan)
        smallest_change_angles_trajectory.append(float(state.current_changes_optimal_orientations))
        smallest_change_offsets_trajectory.append(float(state.current_changes_optimal_offsets_angstrom))

        if _parity_dump.is_active():
            try:
                _parity_dump.dump_iteration(
                    iteration=iteration,
                    init_relion_iteration=int(init_relion_iteration),
                    current_size=int(cs),
                    sigma_offset=float(current_sigma_offset_angstrom),
                    translation_step=float(state.translation_step),
                    translation_range=float(state.translation_range),
                    random_perturbation=float(random_perturbation) if random_perturbation is not None else 0.0,
                    random_perturbation_instance=int(state.perturbation_instance)
                    if hasattr(state, "perturbation_instance")
                    else 0,
                    tau2_fudge=float(tau2_fudge),
                    voxel_size=float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
                    grid_size=int(grid_size),
                    volume_shape=tuple(volume_shape),
                    ave_pmax=float(ave_pmax),
                    fsc=np.asarray(fsc, dtype=np.float64),
                    sigma2_noise=np.asarray(noise_variance, dtype=np.float64)
                    if "noise_variance" in dir()
                    else np.zeros(0),
                    means=means,
                    unreg_means=unreg_means,
                    new_iter_best_rotation_eulers=new_iter_best_rotation_eulers,
                    new_iter_best_translations=new_iter_best_translations,
                )
            except Exception as exc:
                logger.warning("parity_dump.dump_iteration failed at iter %d: %s", iteration, exc)

        # Save assignments for next iteration's change tracking.
        # Use coarse_ha (indexed into effective_rotations/current_rotations)
        # so that local search and convergence detection work correctly
        # regardless of whether adaptive oversampling was used.
        previous_assignments = [ha.copy() if ha is not None else None for ha in coarse_ha]
        _parity_dump.mark_stage(iteration, "convergence")

        # --- Timing ---
        elapsed = time.time() - t0
        wall_times.append(elapsed)

        res_angstrom = shell_index_to_resolution_angstrom(
            pixel_res,
            cryo.image_shape[0],
            cryo.voxel_size,
        )
        logger.info(
            "RELION Iteration %d: current_size=%d, pixel_res=%.1f, "
            "res=%.2f A, ave_Pmax=%.4f, healpix_order=%d, "
            "converged=%s, time=%.1fs",
            iteration + 1,
            cs,
            pixel_res,
            res_angstrom,
            ave_pmax,
            state.healpix_order,
            state.has_converged,
            elapsed,
        )

        if state.has_converged:
            logger.info(
                "Convergence reached at iteration %d. Final resolution: %.2f A (pixel_res=%.1f)",
                iteration + 1,
                res_angstrom,
                pixel_res,
            )
            break

        iteration += 1

    if skip_final_iteration:
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
            "convergence_state": state,
            "data_vs_prior_trajectory": data_vs_prior_trajectory,
            "healpix_order_trajectory": healpix_order_trajectory,
            "ave_Pmax_trajectory": ave_Pmax_trajectory,
            "pmax_per_image_history": pmax_per_image_history,
            "noise_radial_trajectory": noise_radial_trajectory,
            "tau2_radial_trajectory": tau2_radial_trajectory,
            "tau2_sigma2_trajectory": tau2_sigma2_trajectory,
            "tau2_avg_weight_trajectory": tau2_avg_weight_trajectory,
            "tau2_shell_sum_trajectory": tau2_shell_sum_trajectory,
            "tau2_shell_count_trajectory": tau2_shell_count_trajectory,
            "tau2_fsc_used_trajectory": tau2_fsc_used_trajectory,
            "tau2_ssnr_trajectory": tau2_ssnr_trajectory,
            "sigma_offset_used_trajectory": sigma_offset_used_trajectory,
            "sigma_offset_trajectory": sigma_offset_trajectory,
            "frac_changed_trajectory": frac_changed_trajectory,
            "acc_rot_trajectory": acc_rot_trajectory,
            "smallest_change_angles_trajectory": smallest_change_angles_trajectory,
            "smallest_change_offsets_trajectory": smallest_change_offsets_trajectory,
            "best_rotation_eulers_history": best_rotation_eulers_history,
            "best_translations_history": best_translations_history,
        }

    # --- RELION's final iteration: do_join_random_halves + do_use_all_data ---
    # After the EM loop finishes (either by convergence or max_iter), RELION
    # runs ONE more iter with:
    #   - current_size = ori_size (Nyquist, all shells)
    #   - both halves joined (single combined dataset)
    #   - the merged volume as the projection source
    # See ml_optimiser.cpp:10157-10160 (sets do_join_random_halves and
    # do_use_all_data) and ml_optimiser.cpp:5707-5708 (forces current_size to
    # ori_size when do_use_all_data is true).
    #
    # Implementation: average the two half-set volumes, then run one more E+M
    # at full Nyquist on the COMBINED dataset (both halves' particles).
    final_join_means = [(means[0] + means[1]) / 2, (means[0] + means[1]) / 2]
    final_iter_t0 = time.time()
    logger.info("=== RELION final all-data Nyquist iteration (do_join_random_halves=True, do_use_all_data=True) ===")
    final_cs = grid_size  # = ori_size, full Nyquist
    recon_vol_size = int(np.prod([d * PADDING_FACTOR for d in volume_shape]))
    final_ft_y = jnp.zeros(recon_vol_size, dtype=cryo.dtype)
    final_ft_ctf = jnp.zeros(recon_vol_size, dtype=cryo.dtype)
    final_noise_wsum = np.zeros_like(np.asarray(noise_radial_trajectory[-1])) if noise_radial_trajectory else None
    final_img_power = np.zeros_like(np.asarray(noise_radial_trajectory[-1])) if noise_radial_trajectory else None
    final_sumw = 0.0
    for k in range(2):
        # Pass the merged mean as input (both halves get the same projection source).
        # Run on each half-set's particles (avoids loading all particles at once),
        # then accumulate Ft_y/Ft_ctf and noise stats from BOTH halves.
        safe_ibs, safe_rbs = _safe_batch_sizes(
            current_rotations.shape[0],
            current_translations.shape[0],
        )
        _, ha_k_final, Ft_y_k_final, Ft_ctf_k_final, _, noise_stats_k_final = run_em(
            experiment_datasets[k],
            final_join_means[k],
            mean_variance,
            noise_variance,
            current_rotations,
            current_translations,
            disc_type,
            image_batch_size=safe_ibs,
            rotation_block_size=safe_rbs,
            current_size=final_cs,  # full Nyquist
            score_with_masked_images=True,
            return_stats=True,
            accumulate_noise=True,
            half_spectrum_scoring=True,
            projection_padding_factor=PROJECTION_PADDING_FACTOR,
            reconstruction_padding_factor=PADDING_FACTOR,
            image_corrections=image_corrections_per_half[k],
            scale_corrections=scale_corrections_per_half[k],
            image_pre_shifts=relion_translation_search_base(previous_best_translations[k]),
            use_float64_scoring=True,
            use_float64_projections=False,
            do_gridding_correction=True,
            square_window=False,
            sparse_pass2=False,
            disable_adjoint_y=disable_adjoint_y,
            disable_adjoint_ctf=disable_adjoint_ctf,
        )
        # --- Manifest dump for final all-data iteration (Phase 0.1) ---
        if save_intermediates_dir is not None:
            _manifest_path = os.path.join(
                save_intermediates_dir,
                f"manifest_final_half{k}.npz",
            )
            _manifest = {
                "effective_rotations": np.asarray(current_rotations, dtype=np.float32),
                "current_translations": np.asarray(current_translations, dtype=np.float32),
                "rotation_log_prior": np.array([]),
                "translation_log_prior": np.array([]),
                "image_corrections": np.asarray(image_corrections_per_half[k], dtype=np.float64)
                if image_corrections_per_half[k] is not None
                else np.array([]),
                "scale_corrections": np.asarray(scale_corrections_per_half[k], dtype=np.float64)
                if scale_corrections_per_half[k] is not None
                else np.array([]),
                "image_pre_shifts": np.asarray(
                    relion_translation_search_base(previous_best_translations[k]), dtype=np.float32
                )
                if previous_best_translations[k] is not None
                else np.array([]),
                "absolute_previous_translations": np.asarray(previous_best_translations[k], dtype=np.float32)
                if previous_best_translations[k] is not None
                else np.array([]),
                "mean_vol_ft": np.asarray(final_join_means[k]),
                "mean_variance": np.asarray(mean_variance),
                "noise_variance": np.asarray(noise_variance),
                "current_size": np.int32(final_cs),
                "half_spectrum_scoring": np.bool_(True),
                "use_float64_scoring": np.bool_(True),
                "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
                "reconstruction_padding_factor": np.int32(PADDING_FACTOR),
                "score_with_masked_images": np.bool_(True),
                "perturbation_instance": np.float64(random_perturbation),
                "perturbation_factor": np.float64(perturb_factor),
                "iteration": np.int32(-1),
                "half_index": np.int32(k),
            }
            np.savez(_manifest_path, **_manifest)
            logger.info("Final manifest dumped: %s", _manifest_path)

        final_ft_y = final_ft_y + Ft_y_k_final
        final_ft_ctf = final_ft_ctf + Ft_ctf_k_final
        if noise_stats_k_final is not None and final_noise_wsum is not None:
            final_noise_wsum += np.asarray(noise_stats_k_final.wsum_sigma2_noise, dtype=np.float64)
            final_img_power += np.asarray(noise_stats_k_final.wsum_img_power, dtype=np.float64)
            final_sumw += float(noise_stats_k_final.sumw)

    # Reconstruct the final volume from the COMBINED Ft_y/Ft_ctf accumulators
    # at the full Nyquist resolution. Skip the join_halves step (we're already
    # combining the two halves into one dataset for this final iter).
    merged_mean = _reconstruct_volume_eager(
        final_ft_ctf,
        final_ft_y,
        volume_shape,
        PADDING_FACTOR,
        tau=mean_variance,
        tau2_fudge=tau2_fudge,
        projection_padding_factor=PROJECTION_PADDING_FACTOR,
    ).reshape(-1)
    final_iter_elapsed = time.time() - final_iter_t0
    logger.info(
        "Final iter complete: current_size=%d (Nyquist), wall=%.1fs",
        final_cs,
        final_iter_elapsed,
    )
    wall_times.append(final_iter_elapsed)

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
        "pmax_per_image_history": pmax_per_image_history,
        "noise_radial_trajectory": noise_radial_trajectory,
        "tau2_radial_trajectory": tau2_radial_trajectory,
        "tau2_sigma2_trajectory": tau2_sigma2_trajectory,
        "tau2_avg_weight_trajectory": tau2_avg_weight_trajectory,
        "tau2_shell_sum_trajectory": tau2_shell_sum_trajectory,
        "tau2_shell_count_trajectory": tau2_shell_count_trajectory,
        "tau2_fsc_used_trajectory": tau2_fsc_used_trajectory,
        "tau2_ssnr_trajectory": tau2_ssnr_trajectory,
        "sigma_offset_used_trajectory": sigma_offset_used_trajectory,
        "sigma_offset_trajectory": sigma_offset_trajectory,
        "frac_changed_trajectory": frac_changed_trajectory,
        "acc_rot_trajectory": acc_rot_trajectory,
        "smallest_change_angles_trajectory": smallest_change_angles_trajectory,
        "smallest_change_offsets_trajectory": smallest_change_offsets_trajectory,
        "best_rotation_eulers_history": best_rotation_eulers_history,
        "best_translations_history": best_translations_history,
    }
