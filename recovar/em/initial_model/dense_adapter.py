"""InitialModel E-step adapter backed by native dense K-class EM.

This is the ab-initio bridge we want long-term: the hidden variable axis is
``class x pose`` inside the dense K-class engine. There is no outer loop over
classes here. Pseudo-halfsets share one dense E-step and ask the dense engine
to split reconstruction accumulators by halfset, so projection/scoring work is
not duplicated while the VDAM M-step still receives independent halfset
BackProjectors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from recovar.em.dense_single_volume.helpers.significance import _compute_k_class_significance_batched
from recovar.em.dense_single_volume.k_class import run_dense_k_class_em, run_local_k_class_em
from recovar.em.dense_single_volume.local_layout import build_pass2_hypothesis_layout
from recovar.em.sampling import (
    apply_relion_rotation_perturbation,
    get_translation_grid,
    relion_angular_sampling_deg,
    rotation_grid_size,
)

from .layout import relion_bpref_frame_scales, run_em_output_to_bpref
from .m_step import VdamAccumulator
from .state import InitialModelState

_ENGINE_DEFAULTS: dict[str, Any] = {
    "current_size": None,
    "projection_padding_factor": None,
    "reconstruction_padding_factor": None,
    "half_spectrum_scoring": True,
    "score_with_masked_images": True,
    "sparse_pass2": False,
}
_INACTIVE_CLASS_LOG_PRIOR = -1.0e30
_SPARSE_PASS2_CONTROL_KEYS = {
    "adaptive_fraction",
    "max_significants",
    "healpix_order",
    "oversampling_order",
    "translation_step",
    "random_perturbation",
    "coarse_translations",
    "coarse_translation_log_prior",
    "return_profile",
}


@dataclass(frozen=True, kw_only=True)
class DenseInitialModelEstepConfig:
    """Configuration for one InitialModel dense K-class E-step."""

    means: Any | None = None
    mean_variance: Any | None = None
    noise_variance: Any
    rotations: Any
    translations: Any
    disc_type: str = "linear_interp"
    image_batch_size: int = 500
    rotation_block_size: int = 5000
    padding_factor: int = 1
    class_log_priors: Any | None = None
    relion_bpref_frame: bool = True
    relion_projector_frame: bool = False
    engine_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DenseInitialModelEstepResult:
    """Output consumed by ``iteration_loop.run_vdam_iterations``."""

    accumulators: list[VdamAccumulator]
    meta: dict[str, Any]
    halfset_results: dict[int, Any]


def split_pseudo_halfset_particle_ids(
    n_images: int,
    micrograph_names: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RELION-style pseudo-halfset image indices.

    RELION alternates halfset ids after sorting particles by micrograph name.
    If names are unavailable, natural-order alternation is the best local
    fallback and is exact when the dataset is already RELION-sorted.
    """
    ids = np.arange(int(n_images), dtype=np.int64)
    if micrograph_names is None:
        return ids[0::2], ids[1::2]
    order = np.argsort(np.asarray(micrograph_names), kind="stable")
    return order[0::2].astype(np.int64, copy=False), order[1::2].astype(np.int64, copy=False)


def class_log_priors_from_state(state: InitialModelState) -> np.ndarray:
    """Return log class priors from ``state.pdf_class``.

    RELION permits class probabilities to collapse to zero and then treats
    those classes as inactive. Encode that as a finite log-probability
    sentinel so the dense log-sum-exp path underflows cleanly.
    """
    weights = np.asarray(state.pdf_class, dtype=np.float64)
    if weights.shape != (state.K,):
        raise ValueError(f"state.pdf_class must have shape ({state.K},), got {weights.shape}")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("state.pdf_class must contain non-negative finite class probabilities")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("state.pdf_class must contain at least one positive class probability")
    out = np.full(state.K, _INACTIVE_CLASS_LOG_PRIOR, dtype=np.float64)
    positive = weights > 0.0
    out[positive] = np.log(weights[positive] / total)
    return out


def _image_groups(
    particle_ids: np.ndarray | None,
    halfset_ids: np.ndarray | None,
    *,
    n_images: int,
    pseudo_halfsets: bool,
) -> list[tuple[int, np.ndarray]]:
    ids = np.arange(n_images, dtype=np.int64) if particle_ids is None else np.asarray(particle_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError(f"particle_ids must be 1D, got {ids.shape}")
    if np.any(ids < 0) or np.any(ids >= n_images):
        raise ValueError("particle_ids contains entries outside the dataset")

    if not pseudo_halfsets:
        return [(0, ids)]

    if halfset_ids is None:
        h0, h1 = ids[0::2], ids[1::2]
    else:
        halves = np.asarray(halfset_ids, dtype=np.int8)
        if halves.shape != ids.shape:
            raise ValueError(f"halfset_ids shape {halves.shape} must match particle_ids shape {ids.shape}")
        if np.any((halves != 0) & (halves != 1)):
            raise ValueError("halfset_ids must contain only 0/1 values")
        h0, h1 = ids[halves == 0], ids[halves == 1]
    return [(0, h0), (1, h1)]


def _pack_image_groups(groups: list[tuple[int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, int]:
    """Pack halfset groups into one image list plus dense reconstruction ids."""

    if not groups:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32), 0
    image_indices = [np.asarray(group_ids, dtype=np.int64) for _, group_ids in groups]
    group_ids = [
        np.full(np.asarray(group_image_ids).shape, int(group_index), dtype=np.int32)
        for group_index, group_image_ids in groups
    ]
    packed_images = np.concatenate(image_indices) if image_indices else np.empty(0, dtype=np.int64)
    packed_groups = np.concatenate(group_ids) if group_ids else np.empty(0, dtype=np.int32)
    return packed_images, packed_groups, max(int(group_index) for group_index, _ in groups) + 1


def _dense_engine_kwargs(state: InitialModelState, config: DenseInitialModelEstepConfig) -> dict[str, Any]:
    engine_kwargs = dict(_ENGINE_DEFAULTS)
    engine_kwargs["current_size"] = None if state.current_size <= 0 else state.current_size
    engine_kwargs["projection_padding_factor"] = config.padding_factor
    engine_kwargs["reconstruction_padding_factor"] = config.padding_factor
    engine_kwargs.update(config.engine_kwargs)

    controlled = ("image_indices", "reconstruction_group_ids", "reconstruction_group_count")
    present = sorted(name for name in controlled if name in config.engine_kwargs)
    if present:
        raise ValueError(f"InitialModel dense E-step controls these dense-engine arguments: {', '.join(present)}")
    if engine_kwargs["projection_padding_factor"] != engine_kwargs["reconstruction_padding_factor"]:
        raise ValueError("InitialModel dense E-step requires matching projection/reconstruction padding factors")
    return engine_kwargs


def _engine_kwargs_for_image_indices(
    engine_kwargs: dict[str, Any],
    image_indices: np.ndarray,
    *,
    n_images: int,
) -> dict[str, Any]:
    """Adapt dense-engine kwargs whose first axis is selected-image order.

    Most dense engine per-image inputs are indexed internally by global image
    ids. ``translation_log_prior`` is different: the dense engine streams it
    by selected-image row. Accepting a full-dataset prior here keeps callers
    from duplicating pseudo-halfset packing logic.
    """

    out = dict(engine_kwargs)
    prior = out.get("translation_log_prior")
    if prior is None:
        return out
    prior_np = np.asarray(prior)
    if prior_np.ndim != 2:
        return out

    image_indices = np.asarray(image_indices, dtype=np.int64)
    if prior_np.shape[0] == int(n_images):
        out["translation_log_prior"] = prior_np[image_indices]
    elif prior_np.shape[0] != int(image_indices.size):
        raise ValueError(
            "translation_log_prior must be shared, selected-image, or full-dataset shaped; "
            f"got first axis {prior_np.shape[0]} for {image_indices.size} selected images and {n_images} total images"
        )
    return out


def _select_image_rows(value, image_indices: np.ndarray, *, n_images: int, name: str):
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        return value
    image_indices = np.asarray(image_indices, dtype=np.int64)
    if array.shape[0] == int(n_images):
        return array[image_indices]
    if array.shape[0] == int(image_indices.size):
        return value
    raise ValueError(
        f"{name} must be shared, selected-image, or full-dataset shaped; "
        f"got first axis {array.shape[0]} for {image_indices.size} selected images and {n_images} total images"
    )


def _group_local_kwargs(
    engine_kwargs: dict[str, Any],
    image_indices: np.ndarray,
    *,
    n_images: int,
) -> dict[str, Any]:
    """Return dense/local kwargs in the compact row space of a dataset subset."""

    out = _engine_kwargs_for_image_indices(engine_kwargs, image_indices, n_images=n_images)
    for name in ("image_pre_shifts", "image_corrections", "scale_corrections", "translation_prior_centers"):
        out[name] = _select_image_rows(out.get(name), image_indices, n_images=n_images, name=name)
    return out


def _pop_sparse_pass2_options(engine_kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split shared dense/local-engine kwargs from InitialModel pass-2 controls."""

    cleaned = dict(engine_kwargs)
    options = {name: cleaned.pop(name) for name in list(cleaned) if name in _SPARSE_PASS2_CONTROL_KEYS}
    cleaned.pop("sparse_pass2", None)
    return cleaned, options


def _translation_step_from_grid(translations: np.ndarray) -> float:
    unique_vals = np.unique(np.asarray(translations, dtype=np.float32))
    diffs = np.diff(np.sort(unique_vals))
    diffs = diffs[diffs > 1.0e-6]
    return float(diffs.min()) if diffs.size else 1.0


def _relion_projector_dense_rotations(rotations: np.ndarray) -> np.ndarray:
    """Map RELION rotation matrices to dense slicing rotations for Projector::data."""

    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"rotations must have shape (R, 3, 3), got {rotations.shape}")
    swap_xz = np.array(
        (
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    flip_x = np.diag((-1.0, 1.0, 1.0)).astype(np.float64)
    inv_t = np.linalg.inv(rotations).transpose(0, 2, 1)
    return np.einsum("rij,jk,kl->ril", inv_t, swap_xz, flip_x).astype(np.float32)


def _dense_rotations_for_config(rotations: Any, config: DenseInitialModelEstepConfig) -> np.ndarray:
    rotations_np = np.asarray(rotations, dtype=np.float32)
    if not config.relion_projector_frame:
        return rotations_np
    return _relion_projector_dense_rotations(rotations_np)


def _coarse_translations_from_config(
    config: DenseInitialModelEstepConfig,
    options: dict[str, Any],
) -> np.ndarray:
    if "coarse_translations" in options:
        return np.asarray(options["coarse_translations"], dtype=np.float32)
    if "translation_step" in options:
        step = float(options["translation_step"])
    else:
        step = _translation_step_from_grid(np.asarray(config.translations, dtype=np.float32))
    max_offset = float(np.max(np.abs(np.asarray(config.translations, dtype=np.float32))))
    return get_translation_grid(max_pixel=max_offset, pixel_offset=step).astype(np.float32)


def _sparse_pass2_result_to_accumulators(
    result,
    state: InitialModelState,
    *,
    halfset_idx: int,
    relion_bpref_frame: bool,
    relion_projector_frame: bool,
    padding_factor: int,
) -> list[VdamAccumulator]:
    return _arrays_to_accumulators(
        result.Ft_y,
        result.Ft_ctf,
        state,
        halfset_idx=halfset_idx,
        relion_bpref_frame=relion_bpref_frame,
        relion_projector_frame=relion_projector_frame,
        padding_factor=padding_factor,
    )


def _sparse_pass2_estep_meta(
    halfset_results: dict[int, Any],
    selected_particle_ids_by_halfset: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Meta merger for separate exact-local K-class pseudo-halfset passes."""

    meta = _estep_meta(halfset_results)
    selected_particle_ids = []
    pose_assignments = []
    class_assignments = []
    max_posterior = []

    for halfset_idx, result in sorted(halfset_results.items()):
        image_ids = np.asarray(selected_particle_ids_by_halfset[int(halfset_idx)], dtype=np.int64)
        selected_particle_ids.append(image_ids)
        if getattr(result, "pose_assignments", None) is not None:
            pose_assignments.append(np.asarray(result.pose_assignments, dtype=np.int32))
        if getattr(result, "class_assignments", None) is not None:
            class_assignments.append(np.asarray(result.class_assignments, dtype=np.int32))
        stats = getattr(result, "stats", None)
        if stats is not None and getattr(stats, "max_posterior_per_image", None) is not None:
            max_posterior.append(np.asarray(stats.max_posterior_per_image, dtype=np.float32))
            meta[f"halfset_{halfset_idx}_pmax_mean"] = (
                float(np.mean(np.asarray(stats.max_posterior_per_image))) if image_ids.size else 0.0
            )

    if selected_particle_ids:
        meta["selected_particle_ids"] = np.concatenate(selected_particle_ids).astype(np.int64, copy=False)
    if pose_assignments:
        meta["pose_assignments"] = np.concatenate(pose_assignments).astype(np.int32, copy=False)
    if class_assignments:
        meta["class_assignments"] = np.concatenate(class_assignments).astype(np.int32, copy=False)
    if max_posterior:
        meta["max_posterior_per_image"] = np.concatenate(max_posterior).astype(np.float32, copy=False)
    meta["sparse_pass2"] = True
    return meta


def _sparse_pass2_profile_summary(
    pass1_time_s: float,
    pass2_time_s: float,
    n_significant_by_image: list[np.ndarray],
) -> dict[str, object]:
    all_counts = (
        np.concatenate([np.asarray(counts, dtype=np.int32).reshape(-1) for counts in n_significant_by_image])
        if n_significant_by_image
        else np.zeros(0, dtype=np.int32)
    )
    if all_counts.size:
        mean_significant = float(np.mean(all_counts))
        max_significant = int(np.max(all_counts))
    else:
        mean_significant = 0.0
        max_significant = 0
    return {
        "pass1_time_s": float(pass1_time_s),
        "pass2_time_s": float(pass2_time_s),
        "mean_significant_samples": mean_significant,
        "max_significant_samples": max_significant,
    }


def _initial_model_pass2_layout(layout, *, n_global_rotations: int):
    """Use fine rotation ids for InitialModel direction posterior updates.

    The shared adaptive refinement path tracks posterior mass on coarse parent
    rotations for direction-prior updates. InitialModel initializes
    ``pdf_direction`` on the dense oversampled grid, so its E-step metadata must
    report mass in the same fine-grid coordinate system.
    """

    n_global_rotations = int(n_global_rotations)
    if n_global_rotations < 0:
        raise ValueError("n_global_rotations must be non-negative")
    if layout.rotation_ids_flat.size and int(np.max(layout.rotation_ids_flat)) >= n_global_rotations:
        raise ValueError(
            "InitialModel pass-2 layout contains a fine rotation id outside "
            f"n_global_rotations={n_global_rotations}"
        )
    if (
        getattr(layout, "rotation_posterior_ids_flat", None) is None
        and int(layout.n_global_rotations) == n_global_rotations
    ):
        return layout
    return replace(layout, n_global_rotations=n_global_rotations, rotation_posterior_ids_flat=None)


def _run_sparse_pass2_initial_model_estep(
    experiment_dataset,
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
    *,
    class_log_priors,
    groups: list[tuple[int, np.ndarray]],
    means,
    mean_variance,
    engine_kwargs: dict[str, Any],
) -> DenseInitialModelEstepResult:
    """Run RELION-style coarse significance plus exact-local K-class pass-2."""

    if state.pseudo_halfsets:
        if any(np.asarray(image_ids).size == 0 for _, image_ids in groups):
            raise NotImplementedError("InitialModel sparse pass-2 currently requires non-empty pseudo-halfsets")
    base_kwargs, options = _pop_sparse_pass2_options(engine_kwargs)
    healpix_order = int(options.get("healpix_order", 1))
    oversampling_order = int(options.get("oversampling_order", 1))
    if oversampling_order < 1:
        raise ValueError("sparse pass-2 requires oversampling_order >= 1")
    adaptive_fraction = float(options.get("adaptive_fraction", 0.999))
    max_significants = int(options.get("max_significants", -1))
    random_perturbation = float(options.get("random_perturbation", 0.0))
    return_profile = bool(options.get("return_profile", False))

    coarse_translations = _coarse_translations_from_config(config, options)
    translation_step = float(options.get("translation_step", _translation_step_from_grid(coarse_translations)))
    coarse_translation_log_prior = options.get("coarse_translation_log_prior")
    n_coarse_rotations = rotation_grid_size(healpix_order)
    if int(np.asarray(config.rotations).shape[0]) == n_coarse_rotations:
        coarse_rotations = np.asarray(config.rotations, dtype=np.float32)
    else:
        from recovar.em import sampling

        coarse_rotations = sampling.get_rotation_grid(
            nside_level=healpix_order,
            n_in_planes=sampling.rotation_grid_n_in_planes(healpix_order),
            matrices=True,
        ).astype(np.float32)
        coarse_rotations = apply_relion_rotation_perturbation(
            coarse_rotations,
            random_perturbation,
            relion_angular_sampling_deg(healpix_order),
        ).astype(np.float32, copy=False)
    coarse_rotations_for_dense = (
        _relion_projector_dense_rotations(coarse_rotations)
        if config.relion_projector_frame
        else np.asarray(coarse_rotations, dtype=np.float32)
    )

    accumulators: list[VdamAccumulator] = []
    halfset_results: dict[int, Any] = {}
    n_significant_by_image: list[np.ndarray] = []
    pass1_time_s = 0.0
    pass2_time_s = 0.0

    for halfset_idx, image_indices in groups:
        image_indices = np.asarray(image_indices, dtype=np.int64)
        if image_indices.size == 0:
            accumulators.extend(_empty_accumulator(state, k, int(halfset_idx)) for k in range(state.K))
            continue
        group_kwargs = _group_local_kwargs(
            base_kwargs,
            image_indices,
            n_images=int(experiment_dataset.n_images),
        )
        group_dataset = experiment_dataset.subset(image_indices)
        if coarse_translation_log_prior is not None:
            coarse_prior_array = np.asarray(coarse_translation_log_prior)
            pass1_translation_log_prior = (
                coarse_translation_log_prior
                if coarse_prior_array.ndim == 1
                else _select_image_rows(
                    coarse_translation_log_prior,
                    image_indices,
                    n_images=int(experiment_dataset.n_images),
                    name="coarse_translation_log_prior",
                )
            )
        else:
            pass1_translation_log_prior = group_kwargs.get("translation_log_prior")

        t0 = time.time()
        sig_result = _compute_k_class_significance_batched(
            group_dataset,
            means,
            config.noise_variance,
            coarse_rotations_for_dense,
            coarse_translations,
            config.disc_type,
            class_log_priors=class_log_priors,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            image_batch_size=config.image_batch_size,
            rotation_block_size=config.rotation_block_size,
            current_size=group_kwargs.get("current_size"),
            score_with_masked_images=bool(group_kwargs.get("score_with_masked_images", False)),
            rotation_log_prior=group_kwargs.get("rotation_log_prior"),
            translation_log_prior=pass1_translation_log_prior,
            image_corrections=group_kwargs.get("image_corrections"),
            scale_corrections=group_kwargs.get("scale_corrections"),
            image_pre_shifts=group_kwargs.get("image_pre_shifts"),
            half_spectrum_scoring=bool(group_kwargs.get("half_spectrum_scoring", False)),
            projection_padding_factor=int(group_kwargs.get("projection_padding_factor", 1)),
            do_gridding_correction=bool(group_kwargs.get("do_gridding_correction", False)),
            square_window=bool(group_kwargs.get("square_window", False)),
            use_float64_scoring=bool(group_kwargs.get("use_float64_scoring", False)),
        )
        (
            _sig_rot_any,
            _n_sig_all,
            _hard_assignment,
            _class_assignment,
            significant_sample_indices,
            _full_stats,
        ) = sig_result
        pass1_time_s += time.time() - t0
        n_significant_by_image.append(np.asarray(_n_sig_all, dtype=np.int32))

        union_significant_samples = []
        for image_idx in range(int(image_indices.size)):
            image_parts = []
            full_support = False
            for class_idx in range(state.K):
                class_samples = significant_sample_indices[class_idx][image_idx]
                if class_samples is None:
                    full_support = True
                    break
                if np.asarray(class_samples).size:
                    image_parts.append(np.asarray(class_samples, dtype=np.int32))
            if full_support:
                union_significant_samples.append(None)
            elif image_parts:
                union_significant_samples.append(np.unique(np.concatenate(image_parts)).astype(np.int32, copy=False))
            else:
                union_significant_samples.append(np.empty(0, dtype=np.int32))

        local_layout = build_pass2_hypothesis_layout(
            union_significant_samples,
            n_coarse_rotations,
            int(coarse_translations.shape[0]),
            healpix_order,
            coarse_translations,
            oversampling_order=oversampling_order,
            translation_step=translation_step,
            rotation_log_prior=group_kwargs.get("rotation_log_prior"),
            fine_translation_log_prior=group_kwargs.get("translation_log_prior"),
            random_perturbation=random_perturbation,
        )
        if config.relion_projector_frame:
            local_layout = replace(
                local_layout,
                rotations_flat=_relion_projector_dense_rotations(local_layout.rotations_flat),
            )
        local_layout = _initial_model_pass2_layout(
            local_layout,
            n_global_rotations=int(np.asarray(config.rotations).shape[0]),
        )

        t0 = time.time()
        result = run_local_k_class_em(
            group_dataset,
            means,
            mean_variance,
            config.noise_variance,
            local_layout,
            config.disc_type,
            class_log_priors=class_log_priors,
            image_batch_size=config.image_batch_size,
            rotation_block_size=config.rotation_block_size,
            current_size=group_kwargs.get("current_size"),
            accumulate_noise=False,
            projection_padding_factor=int(group_kwargs.get("projection_padding_factor", 1)),
            reconstruction_padding_factor=int(group_kwargs.get("reconstruction_padding_factor", 1)),
            score_with_masked_images=bool(group_kwargs.get("score_with_masked_images", False)),
            half_spectrum_scoring=bool(group_kwargs.get("half_spectrum_scoring", False)),
            use_float64_scoring=bool(group_kwargs.get("use_float64_scoring", False)),
            use_float64_normalization=bool(group_kwargs.get("use_float64_scoring", False)),
            use_float64_projections=bool(group_kwargs.get("use_float64_projections", False)),
            do_gridding_correction=bool(group_kwargs.get("do_gridding_correction", False)),
            square_window=bool(group_kwargs.get("square_window", False)),
            image_corrections=group_kwargs.get("image_corrections"),
            scale_corrections=group_kwargs.get("scale_corrections"),
            image_pre_shifts=group_kwargs.get("image_pre_shifts"),
            reconstruct_significant_only=True,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            return_best_pose_details=True,
            translation_prior_centers=group_kwargs.get("translation_prior_centers"),
        )
        pass2_time_s += time.time() - t0
        halfset_results[int(halfset_idx)] = result
        accumulators.extend(
            _sparse_pass2_result_to_accumulators(
                result,
                state,
                halfset_idx=int(halfset_idx),
                relion_bpref_frame=config.relion_bpref_frame,
                relion_projector_frame=config.relion_projector_frame,
                padding_factor=config.padding_factor,
            )
        )

    out = DenseInitialModelEstepResult(
        accumulators=accumulators,
        meta=_sparse_pass2_estep_meta(
            halfset_results,
            {int(group_index): np.asarray(image_ids, dtype=np.int64) for group_index, image_ids in groups},
        ),
        halfset_results=halfset_results,
    )
    if return_profile:
        out.meta["sparse_pass2_profile_summary"] = _sparse_pass2_profile_summary(
            pass1_time_s,
            pass2_time_s,
            n_significant_by_image,
        )
    return out


def reference_to_dense_means(references: np.ndarray) -> np.ndarray:
    """Convert RELION-frame real-space InitialModel references to dense EM means.

    VDAM stores ``Iref`` as RELION real-space volumes. The dense EM engine
    scores unnormalised, centered Fourier volumes. Keep this conversion in
    that scoring frame. The BPref bridge applies RELION's sign/scale convention
    later when moving dense M-step accumulators back into VDAM.
    """
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.reconstruction.relion_functions import griddingCorrect

    refs = np.asarray(references)
    if refs.ndim != 4:
        raise ValueError(f"references must have shape (K, N, N, N), got {refs.shape}")
    n = int(refs.shape[-1])
    means = []
    for ref in refs:
        corrected, _ = griddingCorrect(jnp.asarray(ref), n, padding_factor=1, order=1)
        ft = ftu.get_dft3(corrected).reshape(-1)
        means.append(np.asarray(ft))
    return np.asarray(means, dtype=np.complex64)


def _resolve_class_inputs(
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
) -> tuple[Any, Any]:
    means = config.means if config.means is not None else reference_to_dense_means(state.Iref)
    if config.mean_variance is not None:
        mean_variance = config.mean_variance
    else:
        mean_variance = np.abs(np.asarray(means)) ** 2
    return means, mean_variance


def _empty_accumulator(state: InitialModelState, class_idx: int, halfset_idx: int) -> VdamAccumulator:
    if state.current_size <= 0:
        r_max = state.ori_size // 2
    else:
        r_max = state.current_size // 2
    if r_max >= state.ori_size // 2:
        shape = (state.ori_size, state.ori_size, state.ori_size // 2 + 1)
    else:
        half_ps = r_max + 1
        shape = (2 * half_ps + 1, 2 * half_ps + 1, half_ps + 1)
    return VdamAccumulator(
        data=np.zeros(shape, dtype=np.complex128),
        weight=np.zeros(shape, dtype=np.float64),
        class_idx=class_idx,
        halfset_idx=halfset_idx,
    )


def _result_to_accumulators(
    result,
    state: InitialModelState,
    *,
    halfset_idx: int,
    relion_bpref_frame: bool,
    relion_projector_frame: bool,
    padding_factor: int,
) -> list[VdamAccumulator]:
    return _arrays_to_accumulators(
        result.Ft_y,
        result.Ft_ctf,
        state,
        halfset_idx=halfset_idx,
        relion_bpref_frame=relion_bpref_frame,
        relion_projector_frame=relion_projector_frame,
        padding_factor=padding_factor,
    )


def _arrays_to_accumulators(
    Ft_y_by_class,
    Ft_ctf_by_class,
    state: InitialModelState,
    *,
    halfset_idx: int,
    relion_bpref_frame: bool,
    relion_projector_frame: bool,
    padding_factor: int,
) -> list[VdamAccumulator]:
    r_max = state.ori_size // 2 if state.current_size <= 0 else state.current_size // 2
    data_scale, weight_scale = (1.0, 1.0)
    if relion_bpref_frame:
        data_scale, weight_scale = relion_bpref_frame_scales(state.ori_size)

    accumulators: list[VdamAccumulator] = []
    for k in range(state.K):
        bp_data, bp_weight = run_em_output_to_bpref(
            np.asarray(Ft_y_by_class[k]),
            np.asarray(Ft_ctf_by_class[k]),
            state.ori_size,
            r_max,
            padding_factor=padding_factor,
        )
        if relion_projector_frame:
            bp_data = bp_data[::-1, :, :]
            bp_weight = bp_weight[::-1, :, :]
        accumulators.append(
            VdamAccumulator(
                data=bp_data * data_scale,
                weight=bp_weight * weight_scale,
                class_idx=k,
                halfset_idx=halfset_idx,
            )
        )
    return accumulators


def _grouped_result_to_accumulators(
    result,
    state: InitialModelState,
    groups: list[tuple[int, np.ndarray]],
    *,
    relion_bpref_frame: bool,
    relion_projector_frame: bool,
    padding_factor: int,
) -> list[VdamAccumulator]:
    grouped_Ft_y = getattr(result, "grouped_Ft_y", None)
    grouped_Ft_ctf = getattr(result, "grouped_Ft_ctf", None)
    if grouped_Ft_y is None or grouped_Ft_ctf is None:
        raise RuntimeError("dense K-class result did not return grouped reconstruction accumulators")

    accumulators: list[VdamAccumulator] = []
    grouped_Ft_y = np.asarray(grouped_Ft_y)
    grouped_Ft_ctf = np.asarray(grouped_Ft_ctf)
    for halfset_idx, _ in sorted(groups):
        accumulators.extend(
            _arrays_to_accumulators(
                grouped_Ft_y[int(halfset_idx)],
                grouped_Ft_ctf[int(halfset_idx)],
                state,
                halfset_idx=int(halfset_idx),
                relion_bpref_frame=relion_bpref_frame,
                relion_projector_frame=relion_projector_frame,
                padding_factor=padding_factor,
            )
        )
    return accumulators


def _estep_meta(halfset_results: dict[int, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {"halfset_ids": tuple(sorted(halfset_results))}
    class_posterior_sums = None
    class_direction_posterior_sums = None
    for h, result in halfset_results.items():
        if getattr(result, "class_posterior_sums", None) is not None:
            sums = np.asarray(result.class_posterior_sums, dtype=np.float64)
            meta[f"halfset_{h}_class_posterior_sums"] = sums
            class_posterior_sums = sums if class_posterior_sums is None else class_posterior_sums + sums
        if getattr(result, "class_assignments", None) is not None:
            meta[f"halfset_{h}_class_assignments"] = np.asarray(result.class_assignments, dtype=np.int32)
        stats = getattr(result, "stats", None)
        if stats is not None and getattr(stats, "max_posterior_per_image", None) is not None:
            meta[f"halfset_{h}_pmax_mean"] = float(np.mean(np.asarray(stats.max_posterior_per_image)))
        profile_summary = getattr(result, "profile_summary", None)
        if profile_summary is not None:
            meta[f"halfset_{h}_profile_summary"] = dict(profile_summary)
        per_class_stats = getattr(result, "per_class_stats", None)
        if per_class_stats is not None:
            direction_sums = np.stack(
                [np.asarray(class_stats.rotation_posterior_sums, dtype=np.float64) for class_stats in per_class_stats],
                axis=0,
            )
            class_direction_posterior_sums = (
                direction_sums
                if class_direction_posterior_sums is None
                else class_direction_posterior_sums + direction_sums
            )
    if class_posterior_sums is not None:
        meta["class_posterior_sums"] = class_posterior_sums
    if class_direction_posterior_sums is not None:
        meta["class_direction_posterior_sums"] = class_direction_posterior_sums
    return meta


def _grouped_estep_meta(result, groups: list[tuple[int, np.ndarray]]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "halfset_ids": tuple(int(group_index) for group_index, _ in sorted(groups)),
        "fused_pseudo_halfsets": True,
    }
    class_responsibilities = getattr(result, "class_responsibilities", None)
    class_assignments = getattr(result, "class_assignments", None)
    pose_assignments = getattr(result, "pose_assignments", None)
    stats = getattr(result, "stats", None)
    pmax = None if stats is None else getattr(stats, "max_posterior_per_image", None)
    if getattr(result, "class_posterior_sums", None) is not None:
        meta["class_posterior_sums"] = np.asarray(result.class_posterior_sums, dtype=np.float64)
    per_class_stats = getattr(result, "per_class_stats", None)
    if per_class_stats is not None:
        meta["class_direction_posterior_sums"] = np.stack(
            [np.asarray(class_stats.rotation_posterior_sums, dtype=np.float64) for class_stats in per_class_stats],
            axis=0,
        )

    cursor = 0
    selected_particle_ids = []
    selected_class_assignments = []
    selected_pmax = []
    for h, image_ids in sorted(groups):
        image_ids = np.asarray(image_ids, dtype=np.int64)
        count = int(image_ids.size)
        rows = slice(cursor, cursor + count)
        selected_particle_ids.append(image_ids)
        if class_responsibilities is not None:
            meta[f"halfset_{h}_class_posterior_sums"] = np.sum(
                np.asarray(class_responsibilities, dtype=np.float64)[:, rows],
                axis=1,
            )
        if class_assignments is not None:
            class_rows = np.asarray(class_assignments, dtype=np.int32)[rows]
            meta[f"halfset_{h}_class_assignments"] = class_rows
            selected_class_assignments.append(class_rows)
        if pmax is not None:
            pmax_rows = np.asarray(pmax, dtype=np.float64)[rows]
            meta[f"halfset_{h}_pmax_mean"] = float(np.mean(pmax_rows)) if pmax_rows.size else 0.0
            selected_pmax.append(pmax_rows)
        cursor += count
    if selected_particle_ids:
        meta["selected_particle_ids"] = np.concatenate(selected_particle_ids).astype(np.int64, copy=False)
    if selected_class_assignments:
        meta["class_assignments"] = np.concatenate(selected_class_assignments).astype(np.int32, copy=False)
    if selected_pmax:
        meta["max_posterior_per_image"] = np.concatenate(selected_pmax).astype(np.float32, copy=False)
    if pose_assignments is not None:
        meta["pose_assignments"] = np.asarray(pose_assignments, dtype=np.int32)

    profile_summary = getattr(result, "profile_summary", None)
    if profile_summary is not None:
        meta["fused_profile_summary"] = dict(profile_summary)
    return meta


def run_dense_initial_model_estep(
    experiment_dataset,
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
    *,
    particle_ids: np.ndarray | None = None,
    halfset_ids: np.ndarray | None = None,
) -> DenseInitialModelEstepResult:
    """Run the InitialModel E-step and return VDAM accumulators.

    The dense K-class engine sees all classes at once. For pseudo-halfsets it
    also sees both halfsets at once and returns separate halfset
    backprojection accumulators from the same posterior calculation.
    """
    class_log_priors = (
        class_log_priors_from_state(state) if config.class_log_priors is None else np.asarray(config.class_log_priors)
    )
    groups = _image_groups(
        particle_ids,
        halfset_ids,
        n_images=int(experiment_dataset.n_images),
        pseudo_halfsets=state.pseudo_halfsets,
    )
    engine_kwargs = _dense_engine_kwargs(state, config)
    means, mean_variance = _resolve_class_inputs(state, config)
    dense_rotations = _dense_rotations_for_config(config.rotations, config)

    if bool(engine_kwargs.get("sparse_pass2", False)):
        return _run_sparse_pass2_initial_model_estep(
            experiment_dataset,
            state,
            config,
            class_log_priors=class_log_priors,
            groups=groups,
            means=means,
            mean_variance=mean_variance,
            engine_kwargs=engine_kwargs,
        )

    if state.pseudo_halfsets:
        packed_image_indices, reconstruction_group_ids, reconstruction_group_count = _pack_image_groups(groups)
        if packed_image_indices.size == 0:
            accumulators = [
                _empty_accumulator(state, k, halfset_idx)
                for halfset_idx, _ in sorted(groups)
                for k in range(state.K)
            ]
            return DenseInitialModelEstepResult(
                accumulators=accumulators,
                meta={"halfset_ids": tuple(int(group_index) for group_index, _ in sorted(groups))},
                halfset_results={},
            )

        result = run_dense_k_class_em(
            experiment_dataset,
            means,
            mean_variance,
            config.noise_variance,
            dense_rotations,
            config.translations,
            config.disc_type,
            class_log_priors=class_log_priors,
            image_batch_size=config.image_batch_size,
            rotation_block_size=config.rotation_block_size,
            image_indices=packed_image_indices,
            reconstruction_group_ids=reconstruction_group_ids,
            reconstruction_group_count=reconstruction_group_count,
            **_engine_kwargs_for_image_indices(
                engine_kwargs,
                packed_image_indices,
                n_images=int(experiment_dataset.n_images),
            ),
        )
        return DenseInitialModelEstepResult(
            accumulators=_grouped_result_to_accumulators(
                result,
                state,
                groups,
                relion_bpref_frame=config.relion_bpref_frame,
                relion_projector_frame=config.relion_projector_frame,
                padding_factor=config.padding_factor,
            ),
            meta=_grouped_estep_meta(result, groups),
            halfset_results={int(halfset_idx): result for halfset_idx, _ in groups},
        )

    halfset_results: dict[int, Any] = {}
    by_halfset: dict[int, list[VdamAccumulator]] = {}
    for halfset_idx, image_indices in groups:
        if image_indices.size == 0:
            by_halfset[halfset_idx] = [_empty_accumulator(state, k, halfset_idx) for k in range(state.K)]
            continue
        result = run_dense_k_class_em(
            experiment_dataset,
            means,
            mean_variance,
            config.noise_variance,
            dense_rotations,
            config.translations,
            config.disc_type,
            class_log_priors=class_log_priors,
            image_batch_size=config.image_batch_size,
            rotation_block_size=config.rotation_block_size,
            image_indices=image_indices,
            **_engine_kwargs_for_image_indices(
                engine_kwargs,
                image_indices,
                n_images=int(experiment_dataset.n_images),
            ),
        )
        halfset_results[halfset_idx] = result
        by_halfset[halfset_idx] = _result_to_accumulators(
            result,
            state,
            halfset_idx=halfset_idx,
            relion_bpref_frame=config.relion_bpref_frame,
            relion_projector_frame=config.relion_projector_frame,
            padding_factor=config.padding_factor,
        )

    accumulators: list[VdamAccumulator] = []
    for halfset_idx in sorted(by_halfset):
        accumulators.extend(by_halfset[halfset_idx])
    selected_particle_ids = []
    pose_assignments = []
    class_assignments = []
    max_posterior = []
    for halfset_idx, image_indices in groups:
        result = halfset_results.get(halfset_idx)
        if result is None:
            continue
        result_pose_assignments = getattr(result, "pose_assignments", None)
        result_class_assignments = getattr(result, "class_assignments", None)
        result_stats = getattr(result, "stats", None)
        result_pmax = None if result_stats is None else getattr(result_stats, "max_posterior_per_image", None)
        if result_pose_assignments is None and result_class_assignments is None and result_pmax is None:
            continue
        selected_particle_ids.append(np.asarray(image_indices, dtype=np.int64))
        if result_pose_assignments is not None:
            pose_assignments.append(np.asarray(result_pose_assignments, dtype=np.int32))
        if result_class_assignments is not None:
            class_assignments.append(np.asarray(result_class_assignments, dtype=np.int32))
        if result_pmax is not None:
            max_posterior.append(np.asarray(result_pmax, dtype=np.float32))
    meta = _estep_meta(halfset_results)
    if selected_particle_ids:
        meta["selected_particle_ids"] = np.concatenate(selected_particle_ids).astype(np.int64, copy=False)
    if pose_assignments:
        meta["pose_assignments"] = np.concatenate(pose_assignments).astype(np.int32, copy=False)
    if class_assignments:
        meta["class_assignments"] = np.concatenate(class_assignments).astype(np.int32, copy=False)
    if max_posterior:
        meta["max_posterior_per_image"] = np.concatenate(max_posterior).astype(np.float32, copy=False)
    return DenseInitialModelEstepResult(
        accumulators=accumulators,
        meta=meta,
        halfset_results=halfset_results,
    )


def dense_initial_model_expectation_step(
    experiment_dataset,
    config: DenseInitialModelEstepConfig,
) -> Callable[[InitialModelState, np.ndarray, np.ndarray], tuple[list[VdamAccumulator], dict[str, Any]]]:
    """Build an ``iteration_loop`` expectation-step callback."""

    def _expectation_step(state: InitialModelState, particle_ids: np.ndarray, halfset_ids: np.ndarray):
        result = run_dense_initial_model_estep(
            experiment_dataset,
            state,
            config,
            particle_ids=particle_ids,
            halfset_ids=halfset_ids,
        )
        return result.accumulators, result.meta

    return _expectation_step
