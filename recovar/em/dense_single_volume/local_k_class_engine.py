"""Native exact-local K-class EM over the joint class x local-pose grid."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from recovar.core.configs import ForwardModelConfig

from .helpers.batch_fetch import fetch_indexed_batch
from .helpers.dtype_policy import DensePrecisionPolicy
from .helpers.fourier_window import make_fourier_window_spec
from .helpers.half_spectrum import (
    make_relion_noise_shell_indices_half,
    make_shell_indices_half,
)
from .helpers.half_volume_mstep import (
    enforce_half_volume_x0,
    half_volume_accumulator_shape,
    half_volume_accumulators_to_full,
)
from .helpers.image_shifts import tiled_half_image_phase_factors
from .helpers.jax_runtime import block_until_ready as _block_until_ready
from .helpers.preprocessing import half_translation_phase_table
from .helpers.projection import (
    compute_noise_block as _compute_noise_block,
    compute_projections_block as _compute_projections_block,
)
from .helpers.translation_prior import (
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from .helpers.types import make_noise_stats, make_relion_stats
from .local_backprojection import (
    compute_local_ctf_sums,
    compute_local_weighted_sums,
    flatten_bucket_rows,
    flatten_bucket_rotations,
)
from .local_debug import (
    maybe_write_debug_score_dump,
    noise_split_diagnostics_requested,
    parse_debug_score_dump_request,
)
from .local_em_engine import (
    _LocalNormalizationState,
    _LocalPostprocessBuffers,
    _LocalTiming,
    _accumulate_local_adjoint_rows,
    _build_local_raw_cache,
    _exact_local_max_hypotheses_per_microbatch,
    _local_raw_cache_enabled,
    _make_local_spectrum_setup,
    _new_local_transfer_timer,
    _postprocess_local_bucket,
    _prepare_local_exact_bucket,
    _reorder_bucket_to_indices,
    _select_local_reconstruction_pack,
)
from .local_layout import LocalHypothesisLayout, bucket_local_hypothesis_layout
from .local_score_pass import (
    compute_k_class_reconstruction_support,
    normalize_local_k_class_scores,
    normalize_local_k_class_scores_with_log_z,
    score_local_k_class_bucket_abs2_on_demand,
    score_local_k_class_bucket_abs2_weighted_on_demand,
)

logger = logging.getLogger(__name__)
_local_k_class_mstep_dump_counter = 0


def _maybe_dump_local_k_class_half_mstep(Ft_y, Ft_ctf, *, current_size, recon_volume_shape, stage: str) -> None:
    dump_dir = os.environ.get("RECOVAR_LOCAL_K_CLASS_MSTEP_DUMP_DIR")
    if not dump_dir:
        return

    global _local_k_class_mstep_dump_counter
    dump_idx = _local_k_class_mstep_dump_counter
    _local_k_class_mstep_dump_counter += 1

    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / f"local_k_class_mstep_{dump_idx:03d}_{stage}_cs{int(current_size or -1):03d}.npz",
        Ft_y=np.asarray(Ft_y),
        Ft_ctf=np.asarray(Ft_ctf),
        current_size=np.int32(-1 if current_size is None else int(current_size)),
        recon_volume_shape=np.asarray(recon_volume_shape, dtype=np.int32),
        stage=np.asarray(stage),
    )


class LocalKClassNativeOutputs(NamedTuple):
    """Raw native local K-class outputs consumed by ``k_class`` assembly."""

    class_log_evidence: np.ndarray
    new_means: list[None]
    Ft_y: list[object]
    Ft_ctf: list[object]
    hard_assignments: np.ndarray
    per_class_stats: tuple[object, ...]
    noise_stats: tuple[object, ...] | None
    per_class_best_pose_rotations: tuple[object, ...] | None
    per_class_best_pose_translations: tuple[object, ...] | None
    per_class_best_pose_rotation_ids: tuple[object, ...] | None
    grouped_Ft_y: object | None = None
    grouped_Ft_ctf: object | None = None


@dataclass(frozen=True)
class _LocalKClassPostprocessState:
    buffers: list[_LocalPostprocessBuffers]
    hard_assignments: np.ndarray
    class_log_evidence: np.ndarray
    class_best_log_score: np.ndarray
    class_max_posterior: np.ndarray
    rotation_posterior_sums: np.ndarray
    best_pose_rotations: np.ndarray | None
    best_pose_translations: np.ndarray | None
    best_pose_rotation_ids: np.ndarray | None

    @classmethod
    def create(
        cls,
        *,
        n_classes: int,
        n_images: int,
        n_global_rotations: int,
        n_translation_dims: int,
        return_best_pose_details: bool,
    ) -> "_LocalKClassPostprocessState":
        hard_assignments = np.empty((n_classes, n_images), dtype=np.int32)
        class_log_evidence = np.empty((n_classes, n_images), dtype=np.float32)
        class_best_log_score = np.empty((n_classes, n_images), dtype=np.float32)
        class_max_posterior = np.empty((n_classes, n_images), dtype=np.float32)
        rotation_posterior_sums = np.zeros((n_classes, n_global_rotations), dtype=np.float64)
        best_pose_rotations = (
            np.empty((n_classes, n_images, 3, 3), dtype=np.float32)
            if return_best_pose_details
            else None
        )
        best_pose_translations = (
            np.empty((n_classes, n_images, n_translation_dims), dtype=np.float32)
            if return_best_pose_details
            else None
        )
        best_pose_rotation_ids = (
            np.empty((n_classes, n_images), dtype=np.int32)
            if return_best_pose_details
            else None
        )
        buffers = [
            _LocalPostprocessBuffers(
                hard_assignment=hard_assignments[class_index],
                log_evidence_per_image=class_log_evidence[class_index],
                best_log_score_per_image=class_best_log_score[class_index],
                max_posterior_per_image=class_max_posterior[class_index],
                rotation_posterior_sums=rotation_posterior_sums[class_index],
                transfer_profile=_new_local_transfer_timer(),
                chunk_nonzero_posterior_rows=[],
                chunk_significant_samples=[],
                chunk_reconstruction_rows=[],
                seen_global_rotations=np.zeros(0, dtype=bool),
                seen_nonzero_global_rotations=np.zeros(0, dtype=bool),
                seen_reconstruction_global_rotations=np.zeros(0, dtype=bool),
                best_pose_rotations=None if best_pose_rotations is None else best_pose_rotations[class_index],
                best_pose_translations=None if best_pose_translations is None else best_pose_translations[class_index],
                best_pose_rotation_ids=None if best_pose_rotation_ids is None else best_pose_rotation_ids[class_index],
            )
            for class_index in range(n_classes)
        ]
        return cls(
            buffers=buffers,
            hard_assignments=hard_assignments,
            class_log_evidence=class_log_evidence,
            class_best_log_score=class_best_log_score,
            class_max_posterior=class_max_posterior,
            rotation_posterior_sums=rotation_posterior_sums,
            best_pose_rotations=best_pose_rotations,
            best_pose_translations=best_pose_translations,
            best_pose_rotation_ids=best_pose_rotation_ids,
        )

    def best_pose_tuple(self, field: str, n_classes: int):
        values = getattr(self, field)
        if values is None:
            return None
        return tuple(values[class_index] for class_index in range(n_classes))


@dataclass
class _LocalKClassNoiseState:
    wsum: list[object]
    img_power: list[object]
    a2: list[object]
    xa: list[object]
    sigma2_offset: list[object]
    sumw: list[object]

    @classmethod
    def zeros(cls, *, n_classes: int, n_shells: int) -> "_LocalKClassNoiseState":
        shell_zeros = [jnp.zeros(n_shells, dtype=jnp.float32) for _ in range(n_classes)]
        return cls(
            wsum=list(shell_zeros),
            img_power=[jnp.zeros(n_shells, dtype=jnp.float32) for _ in range(n_classes)],
            a2=[jnp.zeros(n_shells, dtype=jnp.float32) for _ in range(n_classes)],
            xa=[jnp.zeros(n_shells, dtype=jnp.float32) for _ in range(n_classes)],
            sigma2_offset=[jnp.asarray(0.0, dtype=jnp.float32) for _ in range(n_classes)],
            sumw=[jnp.asarray(0.0, dtype=jnp.float32) for _ in range(n_classes)],
        )

    def stats(self, *, return_noise_split: bool):
        return tuple(
            make_noise_stats(
                wsum_sigma2_noise=self.wsum[class_index],
                wsum_img_power=self.img_power[class_index],
                wsum_sigma2_offset=float(np.asarray(self.sigma2_offset[class_index], dtype=np.float64)),
                sumw=float(np.asarray(self.sumw[class_index], dtype=np.float64)),
                wsum_noise_a2=(self.a2[class_index] if return_noise_split else None),
                wsum_noise_xa=(self.xa[class_index] if return_noise_split else None),
            )
            for class_index in range(len(self.wsum))
        )


def _reject_class_specific_noise(noise_variance, *, n_classes: int, image_shape) -> None:
    noise_np = np.asarray(noise_variance)
    flat_image_size = int(np.prod(image_shape))
    if noise_np.ndim == 2 and noise_np.shape == (n_classes, flat_image_size):
        raise NotImplementedError("Native local K-class currently supports shared noise variance only")
    if noise_np.ndim == 3 and noise_np.shape[0] == n_classes:
        raise NotImplementedError("Native local K-class currently supports shared noise variance only")


def _project_local_k_class_bucket(
    *,
    means_for_projection,
    bucket,
    image_shape,
    proj_volume_shape,
    disc_type: str,
    projection_kwargs: dict,
    window_spec,
    spectrum_setup,
    precision_policy: DensePrecisionPolicy,
):
    projection_rotations = bucket.projection_rotations if bucket.projection_rotations is not None else bucket.local_rotations
    flat_rotations = flatten_bucket_rotations(jnp.asarray(projection_rotations))
    proj_weighted_by_class = []
    proj_for_noise_by_class = []
    for mean_for_proj in means_for_projection:
        proj_half_flat, _ = _compute_projections_block(
            mean_for_proj,
            flat_rotations,
            image_shape,
            proj_volume_shape,
            disc_type,
            return_abs2=False,
            **projection_kwargs,
        )
        if window_spec.use_window:
            proj_half = proj_half_flat[:, window_spec.score_indices].reshape(
                int(bucket.image_indices.shape[0]),
                int(bucket.bucket_rotation_count),
                window_spec.n_score,
            )
            proj_weighted = proj_half * spectrum_setup.half_weights_windowed[None, None, :]
            proj_for_noise = proj_half_flat[:, window_spec.recon_indices].reshape(
                int(bucket.image_indices.shape[0]),
                int(bucket.bucket_rotation_count),
                window_spec.n_recon,
            )
        else:
            n_half = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
            proj_half = proj_half_flat.reshape(
                int(bucket.image_indices.shape[0]),
                int(bucket.bucket_rotation_count),
                n_half,
            )
            proj_weighted = proj_half * spectrum_setup.half_weights[None, None, :]
            proj_for_noise = proj_half

        proj_weighted, proj_for_noise, _, _ = precision_policy.cast_local_projection_scores(
            proj_weighted,
            proj_for_noise,
            None,
            None,
        )
        proj_weighted_by_class.append(proj_weighted)
        proj_for_noise_by_class.append(proj_for_noise)
    return jnp.stack(proj_weighted_by_class, axis=0), jnp.stack(proj_for_noise_by_class, axis=0)


def run_local_k_class_em_native(
    experiment_dataset,
    means,
    mean_variance,
    noise_variance,
    local_layout: LocalHypothesisLayout,
    disc_type: str,
    *,
    class_log_priors,
    image_batch_size: int,
    rotation_block_size: int,
    current_size: int | None,
    accumulate_noise: bool = False,
    projection_padding_factor: int = 1,
    reconstruction_padding_factor: int = 1,
    score_with_masked_images: bool = True,
    reconstruct_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    use_float64_scoring: bool = False,
    use_float64_normalization: bool = True,
    use_float64_projections: bool = False,
    do_gridding_correction: bool = False,
    square_window: bool = False,
    recon_square_window: bool | None = None,
    recon_exact_radius: bool = True,
    half_volume_contract: str = "dense",
    image_corrections: np.ndarray | None = None,
    scale_corrections: np.ndarray | None = None,
    image_pre_shifts: np.ndarray | None = None,
    max_hypotheses_per_microbatch: int | None = None,
    reconstruct_significant_only: bool = False,
    adaptive_fraction: float = 0.999,
    max_significants: int = -1,
    debug_iteration: int | None = None,
    return_best_pose_details: bool = False,
    normalization_log_z: np.ndarray | None = None,
    normalization_log_evidence: np.ndarray | None = None,
    translation_prior_centers: np.ndarray | None = None,
    reconstruction_subtract_projected_reference: bool = False,
    relion_projector_shape: tuple[int, int, int] | None = None,
    reconstruction_group_ids: np.ndarray | None = None,
    reconstruction_group_count: int | None = None,
) -> LocalKClassNativeOutputs:
    """Run exact-local K-class EM with one joint normalizer per image."""

    del mean_variance
    overall_t0 = time.time()
    means = jnp.asarray(means)
    if means.ndim != 2:
        raise ValueError(f"means must have shape (n_classes, volume_size), got {means.shape}")
    n_classes = int(means.shape[0])
    class_log_priors = jnp.asarray(class_log_priors, dtype=jnp.float32)
    if tuple(class_log_priors.shape) != (n_classes,):
        raise ValueError(f"class_log_priors must have shape ({n_classes},), got {class_log_priors.shape}")
    normalization = _LocalNormalizationState.from_inputs(
        normalization_log_z,
        normalization_log_evidence,
        n_images=int(local_layout.n_images),
    )

    image_shape = tuple(experiment_dataset.image_shape)
    volume_shape = tuple(experiment_dataset.volume_shape)
    _reject_class_specific_noise(noise_variance, n_classes=n_classes, image_shape=image_shape)
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    n_trans = int(local_layout.translation_grid.shape[0])
    n_images = int(local_layout.n_images)
    grouped_reconstruction = reconstruction_group_ids is not None
    if grouped_reconstruction:
        reconstruction_group_ids_np = np.asarray(reconstruction_group_ids, dtype=np.int32)
        if reconstruction_group_ids_np.shape != (n_images,):
            raise ValueError(
                "reconstruction_group_ids must have one entry per local-layout image: "
                f"expected ({n_images},), got {reconstruction_group_ids_np.shape}",
            )
        if reconstruction_group_count is None:
            reconstruction_group_count = int(np.max(reconstruction_group_ids_np)) + 1 if n_images else 0
        reconstruction_group_count = int(reconstruction_group_count)
        if reconstruction_group_count <= 0:
            raise ValueError("reconstruction_group_count must be positive when reconstruction_group_ids is set")
        if np.any(reconstruction_group_ids_np < 0) or np.any(reconstruction_group_ids_np >= reconstruction_group_count):
            raise ValueError("reconstruction_group_ids contains entries outside reconstruction_group_count")
    else:
        reconstruction_group_ids_np = None
        reconstruction_group_count = 0
    translation_prior_centers_np = validate_translation_prior_centers(
        translation_prior_centers,
        n_images=n_images,
        n_dims=local_layout.translation_grid.shape[1],
    )

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    if projection_padding_factor > 1:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        means_for_projection = []
        proj_volume_shape = None
        for class_mean in means:
            padded_mean, padded_shape = pad_volume_for_projection(
                class_mean,
                volume_shape,
                projection_padding_factor,
                do_gridding_correction=do_gridding_correction,
                current_size=current_size,
            )
            means_for_projection.append(padded_mean)
            proj_volume_shape = padded_shape
    else:
        means_for_projection = [means[class_index] for class_index in range(n_classes)]
        proj_volume_shape = volume_shape

    precision_policy = DensePrecisionPolicy(
        use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
        use_float64_normalization=use_float64_normalization,
    )
    means_for_projection = [
        precision_policy.cast_projection_volume(mean_for_projection)
        for mean_for_projection in means_for_projection
    ]
    recon_volume_shape = (
        tuple(d * reconstruction_padding_factor for d in volume_shape)
        if reconstruction_padding_factor > 1
        else volume_shape
    )
    recon_accum_shape = half_volume_accumulator_shape(recon_volume_shape)
    recon_volume_size = int(np.prod(recon_accum_shape))

    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=True,
        recon_square=recon_square_window,
        recon_exact_radius=recon_exact_radius,
    )
    spectrum_setup = _make_local_spectrum_setup(
        image_shape,
        n_half,
        noise_variance,
        window_spec,
        half_spectrum_scoring=half_spectrum_scoring,
    )
    noise_variance_half = spectrum_setup.noise_variance_half

    Ft_y = [jnp.zeros(recon_volume_size, dtype=experiment_dataset.dtype) for _ in range(n_classes)]
    Ft_ctf = [jnp.zeros(recon_volume_size, dtype=experiment_dataset.dtype) for _ in range(n_classes)]
    grouped_Ft_y = None
    grouped_Ft_ctf = None
    if grouped_reconstruction:
        grouped_shape = (int(reconstruction_group_count), n_classes, recon_volume_size)
        grouped_Ft_y = jnp.zeros(grouped_shape, dtype=experiment_dataset.dtype)
        grouped_Ft_ctf = jnp.zeros(grouped_shape, dtype=experiment_dataset.dtype)
    postprocess = _LocalKClassPostprocessState.create(
        n_classes=n_classes,
        n_images=n_images,
        n_global_rotations=int(local_layout.n_global_rotations),
        n_translation_dims=int(local_layout.translation_grid.shape[1]),
        return_best_pose_details=return_best_pose_details,
    )

    return_noise_split = noise_split_diagnostics_requested()
    n_shells = image_shape[0] // 2 + 1
    shell_indices_half = make_relion_noise_shell_indices_half(image_shape)
    shell_indices_noise = window_spec.recon_values(shell_indices_half)
    noise_variance_for_noise = window_spec.recon_values(noise_variance_half)
    noise_state = _LocalKClassNoiseState.zeros(n_classes=n_classes, n_shells=n_shells)

    max_hypotheses_per_microbatch = _exact_local_max_hypotheses_per_microbatch(
        max_hypotheses_per_microbatch,
        window_spec.n_score,
    )
    bucket_specs = bucket_local_hypothesis_layout(
        local_layout,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        max_hypotheses_per_microbatch=max_hypotheses_per_microbatch,
    )
    raw_batch_cache = ctf_param_cache = None
    if _local_raw_cache_enabled(n_images, image_shape, getattr(experiment_dataset, "dtype", np.float32)):
        raw_batch_cache, ctf_param_cache = _build_local_raw_cache(experiment_dataset, n_images)

    translation_phases_half = half_translation_phase_table(local_layout.translation_grid, image_shape)
    projection_kwargs = window_spec.projection_kwargs()
    if relion_projector_shape is not None:
        projection_kwargs["relion_projector_shape"] = tuple(int(x) for x in relion_projector_shape)
    transfer_profile = _new_local_transfer_timer()
    timing = _LocalTiming()
    (
        debug_score_dump_dir,
        debug_score_dump_targets,
        debug_score_dump_current_sizes,
        debug_score_dump_iterations,
    ) = parse_debug_score_dump_request()

    for bucket in bucket_specs:
        if raw_batch_cache is None:
            batch_data, ctf_params, fetched_indices = fetch_indexed_batch(experiment_dataset, bucket.image_indices)
        else:
            bucket_image_indices = np.asarray(bucket.image_indices, dtype=np.int32)
            batch_data = raw_batch_cache[bucket_image_indices]
            ctf_params = ctf_param_cache[bucket_image_indices]
            fetched_indices = bucket_image_indices
        bucket = _reorder_bucket_to_indices(bucket, fetched_indices)
        batch_size = int(bucket.image_indices.shape[0])

        translation_sqdist_ang = None
        if translation_prior_centers_np is not None:
            centers = translation_prior_centers_for_images(
                translation_prior_centers_np,
                bucket.image_indices,
                batch_size=batch_size,
            )
            translation_sqdist_ang = translation_sqdist_angstrom(
                local_layout.translation_grid,
                centers,
                experiment_dataset.voxel_size,
            )

        (
            shifted_half,
            shifted_recon_half,
            batch_norm,
            ctf2_over_nv_half,
            processed_score_half,
            real_space_pre_shift_applied,
        ) = _prepare_local_exact_bucket(
            experiment_dataset,
            batch_data,
            ctf_params,
            bucket.image_indices,
            noise_variance_half,
            translation_phases_half,
            config,
            spectrum_setup.norm_half_weights,
            batch_size,
            n_trans,
            score_with_masked_images,
            reconstruct_with_masked_images=reconstruct_with_masked_images,
            image_pre_shifts=image_pre_shifts,
        )
        if scale_corrections is not None:
            batch_scale = jnp.asarray(scale_corrections[np.asarray(bucket.image_indices)])
        else:
            batch_scale = jnp.ones(batch_size, dtype=batch_norm.dtype)

        image_only_corr = None
        if image_corrections is not None:
            batch_corr = jnp.asarray(image_corrections[np.asarray(bucket.image_indices)])
            image_only_corr = batch_corr / batch_scale
            corr_expanded = jnp.repeat(batch_corr, n_trans)
            shifted_half = shifted_half * corr_expanded[:, None]
            shifted_recon_half = shifted_recon_half * corr_expanded[:, None]
            batch_norm = batch_norm * (image_only_corr**2)[:, None]

        if scale_corrections is not None:
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]

        if image_pre_shifts is not None and not real_space_pre_shift_applied:
            batch_shifts = jnp.asarray(image_pre_shifts[np.asarray(bucket.image_indices)])
            phase_expanded = tiled_half_image_phase_factors(image_shape, batch_shifts, n_trans)
            shifted_half = shifted_half * phase_expanded
            shifted_recon_half = shifted_recon_half * phase_expanded

        shifted_half_with_dc = shifted_half
        ctf2_over_nv_half_with_dc = ctf2_over_nv_half
        if half_spectrum_scoring:
            dc_mask = make_shell_indices_half(image_shape) == 0
            shifted_half = jnp.where(dc_mask[None, :], 0.0, shifted_half)
            ctf2_over_nv_half = jnp.where(dc_mask[None, :], 0.0, ctf2_over_nv_half)

        if window_spec.use_window:
            shifted_score = shifted_half[:, window_spec.score_indices]
            shifted_recon = shifted_recon_half[:, window_spec.recon_indices]
            ctf2_over_nv_score = ctf2_over_nv_half[:, window_spec.score_indices]
            ctf2_over_nv_recon = ctf2_over_nv_half_with_dc[:, window_spec.recon_indices]
            shifted_noise = shifted_half_with_dc[:, window_spec.recon_indices]
        else:
            shifted_score = shifted_half
            shifted_recon = shifted_recon_half
            ctf2_over_nv_score = ctf2_over_nv_half
            ctf2_over_nv_recon = ctf2_over_nv_half_with_dc
            shifted_noise = shifted_half_with_dc

        (
            shifted_score,
            shifted_recon,
            shifted_noise,
            ctf2_over_nv_score,
            ctf2_over_nv_recon,
        ) = precision_policy.cast_local_preprocessed_inputs(
            shifted_score,
            shifted_recon,
            shifted_noise,
            ctf2_over_nv_score,
            ctf2_over_nv_recon,
        )
        shifted_score_split = shifted_score.reshape(batch_size, n_trans, -1)
        shifted_recon_split = shifted_recon.reshape(batch_size, n_trans, -1)
        shifted_noise_split = shifted_noise.reshape(batch_size, n_trans, -1)

        proj_weighted, proj_for_noise = _project_local_k_class_bucket(
            means_for_projection=means_for_projection,
            bucket=bucket,
            image_shape=image_shape,
            proj_volume_shape=proj_volume_shape,
            disc_type=disc_type,
            projection_kwargs=projection_kwargs,
            window_spec=window_spec,
            spectrum_setup=spectrum_setup,
            precision_policy=precision_policy,
        )
        local_rotation_log_prior = jnp.asarray(bucket.local_rotation_log_prior)
        translation_log_prior = jnp.asarray(bucket.translation_log_prior)
        local_rotation_mask = jnp.asarray(bucket.local_rotation_mask)
        sample_mask = None if bucket.local_sample_mask is None else jnp.asarray(bucket.local_sample_mask)
        if half_spectrum_scoring:
            scores = score_local_k_class_bucket_abs2_on_demand(
                shifted_score_split,
                ctf2_over_nv_score,
                proj_weighted,
                local_rotation_log_prior,
                class_log_priors,
                translation_log_prior,
                local_rotation_mask,
                sample_mask,
            )
        else:
            scores = score_local_k_class_bucket_abs2_weighted_on_demand(
                shifted_score_split,
                ctf2_over_nv_score,
                proj_weighted,
                spectrum_setup.half_weights_windowed if window_spec.use_window else spectrum_setup.half_weights,
                local_rotation_log_prior,
                class_log_priors,
                translation_log_prior,
                local_rotation_mask,
                sample_mask,
            )
        if normalization.has_log_evidence:
            normalization_dtype = jnp.float64 if use_float64_normalization else batch_norm.dtype
            log_score_offset = (-0.5 * jnp.squeeze(batch_norm, axis=1)).astype(normalization_dtype)
            normalization_log_z_arg = jnp.asarray(
                normalization.log_evidence[np.asarray(bucket.image_indices, dtype=np.int32)],
                dtype=normalization_dtype,
            ) - log_score_offset
        elif normalization.has_log_z:
            normalization_dtype = jnp.float64 if use_float64_normalization else scores.real.dtype
            normalization_log_z_arg = jnp.asarray(
                normalization.log_z[np.asarray(bucket.image_indices, dtype=np.int32)],
                dtype=normalization_dtype,
            )
        else:
            normalization_log_z_arg = None
        (
            _log_Z,
            class_log_Z,
            probs,
            best_log_score_class,
            best_argmax_class,
            _best_argmax,
            max_posterior_class,
        ) = (
            normalize_local_k_class_scores(
                scores,
                use_float64_normalization=use_float64_normalization,
            )
            if normalization_log_z_arg is None
            else normalize_local_k_class_scores_with_log_z(
                scores,
                normalization_log_z_arg,
                use_float64_normalization=use_float64_normalization,
            )
        )
        (
            _reconstruction_sample_mask,
            reconstruction_rotation_mask,
            n_significant_samples,
            reconstruction_probs,
            probs_sum_t,
            _reconstruction_probs_sum_t,
        ) = compute_k_class_reconstruction_support(
            probs,
            local_rotation_mask,
            reconstruct_significant_only=reconstruct_significant_only,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
        )

        for class_index in range(n_classes):
            debug_score_dump_targets = maybe_write_debug_score_dump(
                experiment_dataset=experiment_dataset,
                local_layout=local_layout,
                bucket=bucket,
                image_pre_shifts=image_pre_shifts,
                scores=scores[:, class_index],
                probs=probs[:, class_index],
                log_Z=class_log_Z[:, class_index],
                best_log_score=best_log_score_class[:, class_index],
                max_posterior=max_posterior_class[:, class_index],
                reconstruction_sample_mask=_reconstruction_sample_mask[:, class_index],
                reconstruction_rotation_mask=reconstruction_rotation_mask[:, class_index],
                n_significant_samples=n_significant_samples,
                current_size=current_size,
                debug_iteration=debug_iteration,
                shifted_score_split=shifted_score_split,
                shifted_recon_split=shifted_recon_split,
                ctf2_over_nv_score=ctf2_over_nv_score,
                ctf2_over_nv_recon=ctf2_over_nv_recon,
                proj_weighted=proj_weighted[class_index],
                proj_for_noise=proj_for_noise[class_index],
                proj_abs2_weighted=None,
                dump_dir=debug_score_dump_dir,
                pending_targets=debug_score_dump_targets,
                requested_current_sizes=debug_score_dump_current_sizes,
                requested_iterations=debug_score_dump_iterations,
                dump_suffix=f"class{class_index}",
                remove_after_dump=(class_index == n_classes - 1),
            )

            class_reconstruction_probs = reconstruction_probs[:, class_index]
            class_probs_sum_t = probs_sum_t[:, class_index]
            summed = compute_local_weighted_sums(class_reconstruction_probs, shifted_recon_split)
            ctf_probs = compute_local_ctf_sums(class_reconstruction_probs, ctf2_over_nv_recon)
            if reconstruction_subtract_projected_reference:
                # RELION InitialModel VDAM stores gradients:
                # (Fimg_shift_nomask - Frefctf) * CTF / sigma2.
                summed = summed - proj_for_noise[class_index] * ctf_probs

            pack_selection = _select_local_reconstruction_pack(
                bucket=bucket,
                reconstruction_rotation_mask=reconstruction_rotation_mask[:, class_index],
                probs_sum_t=class_probs_sum_t,
                reconstruct_significant_only=reconstruct_significant_only,
                rotation_block_size=rotation_block_size,
                transfer_profile=transfer_profile,
                pack_start_time=time.time(),
            )
            take_indices_jnp = jnp.asarray(pack_selection.take_indices, dtype=jnp.int32)
            pack_mask_jnp = jnp.asarray(pack_selection.pack_mask)
            packed_rotations_np = np.take_along_axis(
                np.asarray(bucket.local_rotations, dtype=np.float32),
                pack_selection.take_indices[:, :, None, None],
                axis=1,
            )
            packed_summed = jnp.take_along_axis(summed, take_indices_jnp[:, :, None], axis=1)
            packed_summed = jnp.where(pack_mask_jnp[:, :, None], packed_summed, 0.0)
            packed_ctf_probs = jnp.take_along_axis(ctf_probs, take_indices_jnp[:, :, None], axis=1)
            packed_ctf_probs = jnp.where(pack_mask_jnp[:, :, None], packed_ctf_probs, 0.0)
            packed_flat_rotations = flatten_bucket_rotations(jnp.asarray(packed_rotations_np))
            if grouped_reconstruction:
                bucket_group_ids = reconstruction_group_ids_np[np.asarray(bucket.image_indices, dtype=np.int32)]
                for group_index in range(int(reconstruction_group_count)):
                    group_mask = jnp.asarray(bucket_group_ids == int(group_index))
                    grouped_packed_summed = jnp.where(group_mask[:, None, None], packed_summed, 0.0)
                    grouped_packed_ctf = jnp.where(group_mask[:, None, None], packed_ctf_probs, 0.0)
                    group_Ft_y, group_Ft_ctf = _accumulate_local_adjoint_rows(
                        packed_summed_rows=flatten_bucket_rows(grouped_packed_summed),
                        packed_ctf_rows=flatten_bucket_rows(grouped_packed_ctf),
                        packed_flat_rotations=packed_flat_rotations,
                        Ft_y=grouped_Ft_y[group_index, class_index],
                        Ft_ctf=grouped_Ft_ctf[group_index, class_index],
                        recon_window_indices=window_spec.recon_indices,
                        image_shape=image_shape,
                        recon_volume_shape=recon_volume_shape,
                        use_window=window_spec.use_window,
                        current_size=current_size,
                        disable_adjoint_y=False,
                        disable_adjoint_ctf=False,
                        return_profile=False,
                        timing=timing,
                    )
                    grouped_Ft_y = grouped_Ft_y.at[group_index, class_index].set(group_Ft_y)
                    grouped_Ft_ctf = grouped_Ft_ctf.at[group_index, class_index].set(group_Ft_ctf)
            else:
                Ft_y[class_index], Ft_ctf[class_index] = _accumulate_local_adjoint_rows(
                    packed_summed_rows=flatten_bucket_rows(packed_summed),
                    packed_ctf_rows=flatten_bucket_rows(packed_ctf_probs),
                    packed_flat_rotations=packed_flat_rotations,
                    Ft_y=Ft_y[class_index],
                    Ft_ctf=Ft_ctf[class_index],
                    recon_window_indices=window_spec.recon_indices,
                    image_shape=image_shape,
                    recon_volume_shape=recon_volume_shape,
                    use_window=window_spec.use_window,
                    current_size=current_size,
                    disable_adjoint_y=False,
                    disable_adjoint_ctf=False,
                    return_profile=False,
                    timing=timing,
                )

            if accumulate_noise:
                support_mass = jnp.sum(class_reconstruction_probs.reshape(batch_size, -1), axis=1).astype(jnp.float32)
                if translation_sqdist_ang is not None:
                    translation_posterior = jnp.sum(class_reconstruction_probs, axis=1).astype(jnp.float32)
                    noise_sumw_offset = jnp.sum(
                        translation_posterior * jnp.asarray(translation_sqdist_ang, dtype=jnp.float32),
                    )
                else:
                    noise_sumw_offset = jnp.asarray(0.0, dtype=jnp.float32)

                processed_noise_power_half = processed_score_half
                if image_only_corr is not None:
                    processed_noise_power_half = processed_noise_power_half * image_only_corr[:, None]
                batch_img_power = jnp.sum(
                    (jnp.abs(processed_noise_power_half) ** 2) * support_mass[:, None],
                    axis=0,
                ).astype(jnp.float32)
                batch_img_power_shells = jnp.zeros(n_shells, dtype=jnp.float32)
                batch_img_power_shells = batch_img_power_shells.at[shell_indices_half].add(batch_img_power)
                noise_state.img_power[class_index] = noise_state.img_power[class_index] + batch_img_power_shells
                noise_state.sumw[class_index] = noise_state.sumw[class_index] + jnp.sum(support_mass)

                summed_masked_noise = compute_local_weighted_sums(class_reconstruction_probs, shifted_noise_split)
                packed_summed_masked_noise = jnp.take_along_axis(
                    summed_masked_noise,
                    take_indices_jnp[:, :, None],
                    axis=1,
                )
                packed_summed_masked_noise = jnp.where(
                    pack_mask_jnp[:, :, None],
                    packed_summed_masked_noise,
                    0.0,
                )
                class_proj_for_noise = proj_for_noise[class_index]
                packed_proj_for_noise = jnp.take_along_axis(
                    class_proj_for_noise,
                    take_indices_jnp[:, :, None],
                    axis=1,
                )
                packed_proj_for_noise = jnp.where(pack_mask_jnp[:, :, None], packed_proj_for_noise, 0.0)
                flat_proj_for_noise = flatten_bucket_rows(packed_proj_for_noise)
                block_noise_shells, block_a2_shells, block_xa_shells = _compute_noise_block(
                    flat_proj_for_noise,
                    jnp.abs(flat_proj_for_noise) ** 2,
                    flatten_bucket_rows(packed_summed_masked_noise),
                    flatten_bucket_rows(packed_ctf_probs),
                    noise_variance_for_noise,
                    shell_indices_noise,
                    n_shells,
                    return_noise_split,
                )
                noise_state.wsum[class_index] = noise_state.wsum[class_index] + block_noise_shells
                if return_noise_split:
                    noise_state.a2[class_index] = noise_state.a2[class_index] + block_a2_shells
                    noise_state.xa[class_index] = noise_state.xa[class_index] + block_xa_shells
                noise_state.sigma2_offset[class_index] = noise_state.sigma2_offset[class_index] + noise_sumw_offset

            _postprocess_local_bucket(
                image_indices=bucket.image_indices,
                local_rotation_ids=bucket.local_rotation_ids,
                local_rotation_mask=bucket.local_rotation_mask,
                local_rotations=bucket.local_rotations,
                local_rotation_posterior_ids=bucket.local_rotation_posterior_ids,
                translation_grid=local_layout.translation_grid,
                n_trans=n_trans,
                best_argmax=best_argmax_class[:, class_index],
                batch_norm=batch_norm,
                log_Z=class_log_Z[:, class_index],
                best_log_score=best_log_score_class[:, class_index],
                max_posterior=max_posterior_class[:, class_index],
                probs_sum_t=class_probs_sum_t,
                n_significant_samples=n_significant_samples,
                collect_profile_stats=False,
                reconstruction_row_count=pack_selection.row_count,
                reconstruction_take_indices=pack_selection.take_indices,
                reconstruction_pack_mask=pack_selection.pack_mask,
                buffers=postprocess.buffers[class_index],
            )

    if grouped_reconstruction:
        grouped_Ft_y_by_group = []
        grouped_Ft_ctf_by_group = []
        for group_index in range(int(reconstruction_group_count)):
            group_Ft_y = []
            group_Ft_ctf = []
            for class_index in range(n_classes):
                label = f"Exact local K-class group {group_index} class {class_index}"
                group_class_Ft_y, group_class_Ft_ctf = enforce_half_volume_x0(
                    grouped_Ft_y[group_index, class_index],
                    grouped_Ft_ctf[group_index, class_index],
                    recon_volume_shape,
                    logger=logger,
                    label=label,
                )
                group_class_Ft_y, group_class_Ft_ctf = half_volume_accumulators_to_full(
                    group_class_Ft_y,
                    group_class_Ft_ctf,
                    recon_volume_shape,
                    contract=half_volume_contract,
                )
                group_Ft_y.append(group_class_Ft_y)
                group_Ft_ctf.append(group_class_Ft_ctf)
            grouped_Ft_y_by_group.append(jnp.stack(group_Ft_y, axis=0))
            grouped_Ft_ctf_by_group.append(jnp.stack(group_Ft_ctf, axis=0))
        grouped_Ft_y = jnp.stack(grouped_Ft_y_by_group, axis=0)
        grouped_Ft_ctf = jnp.stack(grouped_Ft_ctf_by_group, axis=0)
        Ft_y = [jnp.sum(grouped_Ft_y[:, class_index], axis=0) for class_index in range(n_classes)]
        Ft_ctf = [jnp.sum(grouped_Ft_ctf[:, class_index], axis=0) for class_index in range(n_classes)]
    else:
        for class_index in range(n_classes):
            _maybe_dump_local_k_class_half_mstep(
                Ft_y[class_index],
                Ft_ctf[class_index],
                current_size=current_size,
                recon_volume_shape=recon_volume_shape,
                stage=f"class{class_index}_pre_x0",
            )
            Ft_y[class_index], Ft_ctf[class_index] = enforce_half_volume_x0(
                Ft_y[class_index],
                Ft_ctf[class_index],
                recon_volume_shape,
                logger=logger,
                label=f"Exact local K-class {class_index}",
            )
            _maybe_dump_local_k_class_half_mstep(
                Ft_y[class_index],
                Ft_ctf[class_index],
                current_size=current_size,
                recon_volume_shape=recon_volume_shape,
                stage=f"class{class_index}_post_x0",
            )
            Ft_y[class_index], Ft_ctf[class_index] = half_volume_accumulators_to_full(
                Ft_y[class_index],
                Ft_ctf[class_index],
                recon_volume_shape,
                contract=half_volume_contract,
            )

    noise_stats = noise_state.stats(return_noise_split=return_noise_split) if accumulate_noise else None

    per_class_stats = tuple(
        make_relion_stats(
            log_evidence_per_image=postprocess.class_log_evidence[class_index],
            best_log_score_per_image=postprocess.class_best_log_score[class_index],
            max_posterior_per_image=postprocess.class_max_posterior[class_index],
            rotation_posterior_sums=postprocess.rotation_posterior_sums[class_index],
        )
        for class_index in range(n_classes)
    )
    _block_until_ready(Ft_y, Ft_ctf)
    logger.info(
        "Native local K-class EM completed: K=%d images=%d elapsed=%.1fs",
        n_classes,
        n_images,
        time.time() - overall_t0,
    )
    return LocalKClassNativeOutputs(
        class_log_evidence=postprocess.class_log_evidence,
        new_means=[None for _ in range(n_classes)],
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        hard_assignments=postprocess.hard_assignments,
        per_class_stats=per_class_stats,
        noise_stats=noise_stats,
        per_class_best_pose_rotations=postprocess.best_pose_tuple("best_pose_rotations", n_classes),
        per_class_best_pose_translations=postprocess.best_pose_tuple("best_pose_translations", n_classes),
        per_class_best_pose_rotation_ids=postprocess.best_pose_tuple("best_pose_rotation_ids", n_classes),
        grouped_Ft_y=grouped_Ft_y,
        grouped_Ft_ctf=grouped_Ft_ctf,
    )
