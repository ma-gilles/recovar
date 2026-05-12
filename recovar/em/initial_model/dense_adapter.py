"""InitialModel E-step adapter on the dense K-class engine.

The hidden variable axis is ``class x pose``; pseudo-halfsets share one E-step
(reconstruction accumulators are split per halfset) so projection/scoring isn't
duplicated while the VDAM M-step still gets independent halfset BackProjectors.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from recovar.em.dense_single_volume.helpers.convergence import healpix_angular_step
from recovar.em.dense_single_volume.helpers.resolution import compute_coarse_image_size
from recovar.em.dense_single_volume.helpers.significance import _compute_k_class_significance_batched
from recovar.em.dense_single_volume.k_class import run_dense_k_class_em, run_local_k_class_em
from recovar.em.dense_single_volume.local_layout import build_pass2_hypothesis_layout
from recovar.em.sampling import (
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
    "reconstruct_with_masked_images": True,
    "sparse_pass2": False,
    # RELION InitialModel BPref uses the rounded radial reconstruction support
    # encoded by Minvsigma2, not the full square Fourier crop.
    "recon_square_window": False,
    "recon_exact_radius": False,
    "reconstruction_subtract_projected_reference": True,
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
    "particle_diameter_ang",
    "pass1_current_size",
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

    RELION's InitialModel BPref path routes pseudo-halfsets by global
    ``part_id % 2`` in ``storeWeightedSums``. ``micrograph_names`` is accepted
    for backwards compatibility but does not affect this routing.
    """
    ids = np.arange(int(n_images), dtype=np.int64)
    return ids[0::2], ids[1::2]


def class_log_priors_from_state(state: InitialModelState) -> np.ndarray:
    """Log class priors from ``state.pdf_class`` (collapsed classes get a finite sentinel)."""
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
    """Slice ``translation_log_prior`` (selected-image axis) for the dense engine."""
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


_DENSE_RUN_EM_REJECT = frozenset(
    {
        # InitialModel-only kwargs the dense run_em wrapper doesn't accept.
        "reconstruct_with_masked_images",
        "recon_square_window",
        "recon_exact_radius",
        "reconstruction_subtract_projected_reference",
        "relion_projector_shape",
        # Sparse/local engine kwargs that run_dense_k_class_em rejects.
        "return_profile",
        "return_best_pose_details",
        "return_stats",
        "disable_adjoint_y",
        "disable_adjoint_ctf",
        "normalization_log_evidence",
    }
)


def _dense_run_em_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs unsupported by the dense run_em wrapper (sparse-only escape hatch)."""
    return {k: v for k, v in kwargs.items() if k not in _DENSE_RUN_EM_REJECT}


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


def _resolve_sparse_pass1_current_size(
    state: InitialModelState,
    group_kwargs: dict[str, Any],
    options: dict[str, Any],
) -> int | None:
    """RELION's coarse pass-1 scoring size (``image_coarse_size``) for sparse pass 2."""
    explicit = options.get("pass1_current_size")
    if explicit is not None:
        explicit = int(explicit)
        return None if explicit <= 0 or explicit >= int(state.ori_size) else explicit

    current_size = group_kwargs.get("current_size")
    particle_diameter = options.get("particle_diameter_ang")
    if particle_diameter is None:
        return current_size

    coarse_size = int(
        compute_coarse_image_size(
            healpix_angular_step(int(options.get("healpix_order", 0))),
            float(state.pixel_size),
            int(state.ori_size),
            particle_diameter=float(particle_diameter),
        )
    )
    current_limit = int(current_size) if current_size is not None else int(state.ori_size)
    coarse_size = min(max(2, coarse_size), current_limit, int(state.ori_size))
    if coarse_size % 2:
        coarse_size += 1
    return None if int(coarse_size) >= int(state.ori_size) else int(coarse_size)


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


def _relion_projector_to_dense_volume(projector_data: np.ndarray, ori_size: int) -> np.ndarray:
    """Embed cropped RELION ``Projector::data`` half-complex slab into dense full-centered Fourier cube.

    Out-of-range coordinates are clipped (any-size input accepted; high frequencies truncated).
    """
    import recovar.core.fourier_transform_utils as ftu

    ppref = np.asarray(projector_data, dtype=np.complex128)
    if ppref.ndim != 3:
        raise ValueError(f"projector_data must be 3D, got {ppref.shape}")
    n = int(ori_size)
    center = n // 2
    half = np.zeros((n, n, center + 1), dtype=np.complex128)
    slab = ppref[::-1, :, :]
    z_center = slab.shape[0] // 2
    y_center = slab.shape[1] // 2
    x_max = min(slab.shape[2], center + 1)
    for iz in range(slab.shape[0]):
        z = (iz - z_center) + center
        if z < 0 or z >= n:
            continue
        for iy in range(slab.shape[1]):
            y = (iy - y_center) + center
            if 0 <= y < n:
                half[z, y, :x_max] = slab[iz, iy, :x_max]
    return np.asarray(ftu.half_volume_to_full_volume(half, (n, n, n)), dtype=np.complex128)


def reference_to_relion_projector_dense_means(
    references: np.ndarray,
    *,
    current_size: int,
    padding_factor: int = 1,
    interpolator: int = 1,
) -> np.ndarray:
    """Convert recovar-frame references through RELION's ``Projector`` setup (data_dim=2, scaled by ``-N^2``)."""
    from recovar.relion_bind import _relion_bind_core as bind
    from recovar.utils.helpers import recovar_volume_to_relion

    refs = np.asarray(references)
    if refs.ndim != 4:
        raise ValueError(f"references must have shape (K, N, N, N), got {refs.shape}")
    n = int(refs.shape[-1])
    means = []
    for ref in refs:
        ref_relion = np.asarray(recovar_volume_to_relion(ref), dtype=np.float64)
        projector_data, *_ = bind.compute_fourier_transform_map(
            ref_relion,
            n,
            int(padding_factor),
            int(interpolator),
            int(current_size),
            True,
            2,
        )
        dense = _relion_projector_to_dense_volume(np.asarray(projector_data), n)
        # RECOVAR_DENSE_MEANS_SCALE diag override (see project_k2_c2_cc_root_cause_2026_05_03).
        tok = (os.environ.get("RECOVAR_DENSE_MEANS_SCALE") or "-N2").strip()
        scale = {"-N2": -(n**2), "N2": float(n**2)}.get(tok)
        if scale is None:
            scale = float(tok)
        means.append(dense.reshape(-1) * scale)
    return np.asarray(means, dtype=np.complex64)


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


# (attr_name_on_result, dtype) for fields harvested per halfset and concatenated.
_SPARSE_PASS2_RESULT_FIELDS: tuple[tuple[str, type], ...] = (
    ("pose_assignments", np.int32),
    ("class_assignments", np.int32),
    ("best_pose_rotations", np.float32),
    ("best_pose_translations", np.float32),
    ("best_pose_rotation_ids", np.int32),
)


def _sparse_pass2_estep_meta(
    halfset_results: dict[int, Any],
    selected_particle_ids_by_halfset: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Meta merger for separate exact-local K-class pseudo-halfset passes."""

    meta = _estep_meta(halfset_results)
    selected_particle_ids: list[np.ndarray] = []
    max_posterior: list[np.ndarray] = []
    field_lists: dict[str, list[np.ndarray]] = {attr: [] for attr, _ in _SPARSE_PASS2_RESULT_FIELDS}

    for halfset_idx, result in sorted(halfset_results.items()):
        image_ids = np.asarray(selected_particle_ids_by_halfset[int(halfset_idx)], dtype=np.int64)
        selected_particle_ids.append(image_ids)
        for attr, dtype in _SPARSE_PASS2_RESULT_FIELDS:
            value = getattr(result, attr, None)
            if value is not None:
                field_lists[attr].append(np.asarray(value, dtype=dtype))
        stats = getattr(result, "stats", None)
        if stats is not None and getattr(stats, "max_posterior_per_image", None) is not None:
            max_posterior.append(np.asarray(stats.max_posterior_per_image, dtype=np.float32))
            meta[f"halfset_{halfset_idx}_pmax_mean"] = (
                float(np.mean(np.asarray(stats.max_posterior_per_image))) if image_ids.size else 0.0
            )

    def _merge(arrays: list[np.ndarray], key: str, dtype) -> None:
        if arrays:
            meta[key] = np.concatenate(arrays).astype(dtype, copy=False)

    _merge(selected_particle_ids, "selected_particle_ids", np.int64)
    for attr, dtype in _SPARSE_PASS2_RESULT_FIELDS:
        _merge(field_lists[attr], attr, dtype)
    _merge(max_posterior, "max_posterior_per_image", np.float32)
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
    return {
        "pass1_time_s": float(pass1_time_s),
        "pass2_time_s": float(pass2_time_s),
        "mean_significant_samples": float(np.mean(all_counts)) if all_counts.size else 0.0,
        "max_significant_samples": int(np.max(all_counts)) if all_counts.size else 0,
    }


def _initial_model_pass2_layout(layout):
    """Scatter pass-2 posterior mass into RELION coarse direction bins."""

    parent_ids = getattr(layout, "rotation_posterior_ids_flat", None)
    if parent_ids is None:
        return layout

    n_parent_rotations = int(layout.n_global_rotations)
    n_psi = int(getattr(layout, "n_psi", 0))
    if n_parent_rotations <= 0 or n_psi <= 0 or n_parent_rotations % n_psi != 0:
        raise ValueError(
            "InitialModel pass-2 posterior grid expects parent rotations "
            f"to be direction-major with n_psi={n_psi}, got {n_parent_rotations} rotations",
        )
    parent_ids = np.asarray(parent_ids, dtype=np.int32)
    if np.any(parent_ids < 0) or np.any(parent_ids >= n_parent_rotations):
        raise ValueError("InitialModel pass-2 layout contains an invalid parent posterior id")
    direction_ids = (parent_ids // n_psi).astype(np.int32, copy=False)
    return replace(
        layout,
        n_global_rotations=n_parent_rotations // n_psi,
        rotation_posterior_ids_flat=direction_ids,
    )


def _class_local_rotation_log_prior(class_rotation_log_prior, layout) -> np.ndarray | None:
    if class_rotation_log_prior is None:
        return None
    prior = np.asarray(class_rotation_log_prior, dtype=np.float32)
    parent_ids = np.asarray(getattr(layout, "rotation_posterior_ids_flat", None), dtype=np.int64)
    if parent_ids.size != int(layout.rotation_log_priors_flat.shape[0]):
        raise ValueError("local layout is missing coarse parent rotation ids for class priors")
    if prior.ndim != 2 or prior.shape[1] <= int(np.max(parent_ids, initial=-1)):
        raise ValueError(
            f"class_rotation_log_prior must have shape (n_classes, n_coarse_rotations), got {prior.shape}",
        )
    return prior[:, parent_ids]


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

    base_kwargs, options = _pop_sparse_pass2_options(engine_kwargs)
    healpix_order = int(options.get("healpix_order", 1))
    oversampling_order = int(options.get("oversampling_order", 1))
    if oversampling_order < 1:
        raise ValueError("sparse pass-2 requires oversampling_order >= 1")
    adaptive_fraction = float(options.get("adaptive_fraction", 0.999))
    max_significants = int(options.get("max_significants", -1))
    random_perturbation = float(options.get("random_perturbation", 0.0))
    return_profile = bool(options.get("return_profile", False))
    pass1_time_s = 0.0
    pass2_time_s = 0.0
    n_significant_by_image: list[np.ndarray] = []

    coarse_translations = _coarse_translations_from_config(config, options)
    translation_step = float(options.get("translation_step", _translation_step_from_grid(coarse_translations)))
    coarse_translation_log_prior = options.get("coarse_translation_log_prior")
    n_coarse_rotations = rotation_grid_size(healpix_order)
    if int(np.asarray(config.rotations).shape[0]) == n_coarse_rotations:
        coarse_rotations = np.asarray(config.rotations, dtype=np.float32)
    else:
        from recovar.em import sampling

        coarse_rotations = sampling.get_relion_hidden_rotation_grid(
            healpix_order,
            matrices=True,
        ).astype(np.float32)
        coarse_rotations = sampling.apply_relion_rotation_perturbation(
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
        coarse_rotations_for_pass1 = coarse_rotations_for_dense
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
        pass1_current_size = _resolve_sparse_pass1_current_size(state, group_kwargs, options)

        t0 = time.time()
        sig_result = _compute_k_class_significance_batched(
            group_dataset,
            means,
            config.noise_variance,
            coarse_rotations_for_pass1,
            coarse_translations,
            config.disc_type,
            class_log_priors=class_log_priors,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            image_batch_size=config.image_batch_size,
            rotation_block_size=config.rotation_block_size,
            current_size=pass1_current_size,
            score_with_masked_images=bool(group_kwargs.get("score_with_masked_images", False)),
            rotation_log_prior=group_kwargs.get("class_rotation_log_prior", group_kwargs.get("rotation_log_prior")),
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

        # Per-image union of class-wise significant samples; None if any class is full-support.
        union_significant_samples = []
        for i in range(int(image_indices.size)):
            cls = [significant_sample_indices[k][i] for k in range(state.K)]
            if any(s is None for s in cls):
                union_significant_samples.append(None)
                continue
            arrs = [np.asarray(s, dtype=np.int32) for s in cls if np.asarray(s).size]
            union_significant_samples.append(
                np.unique(np.concatenate(arrs)).astype(np.int32, copy=False) if arrs else np.empty(0, dtype=np.int32)
            )

        # RELION reuses the coarse pass-1 pdf_offset for all oversampled pass-2 children.
        pass2_translation_log_prior = pass1_translation_log_prior
        pass2_fine_translation_log_prior = None
        if pass2_translation_log_prior is None and (fallback := group_kwargs.get("translation_log_prior")) is not None:
            fallback_np = np.asarray(fallback)
            if fallback_np.ndim > 0 and fallback_np.shape[-1] == int(coarse_translations.shape[0]):
                pass2_translation_log_prior = fallback
            else:
                pass2_fine_translation_log_prior = fallback

        local_layout = build_pass2_hypothesis_layout(
            union_significant_samples,
            n_coarse_rotations,
            int(coarse_translations.shape[0]),
            healpix_order,
            coarse_translations,
            oversampling_order=oversampling_order,
            translation_step=translation_step,
            rotation_log_prior=group_kwargs.get("rotation_log_prior"),
            translation_log_prior=pass2_translation_log_prior,
            fine_translation_log_prior=pass2_fine_translation_log_prior,
            random_perturbation=random_perturbation,
            rotation_index_order="relion_hidden",
        )
        class_local_rotation_log_prior = _class_local_rotation_log_prior(
            group_kwargs.get("class_rotation_log_prior"),
            local_layout,
        )
        if config.relion_projector_frame:
            local_layout = replace(
                local_layout,
                rotations_flat=_relion_projector_dense_rotations(local_layout.rotations_flat),
            )
        local_layout = _initial_model_pass2_layout(local_layout)

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
            accumulate_noise=True,
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
            mstep_subtract_ctf_projection=bool(group_kwargs.get("reconstruction_subtract_projected_reference", False)),
            mstep_relion_x_half=bool(config.relion_bpref_frame),
            reconstruct_significant_only=True,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            return_profile=return_profile,
            return_best_pose_details=True,
            translation_prior_centers=group_kwargs.get("translation_prior_centers"),
            class_local_rotation_log_prior=class_local_rotation_log_prior,
        )
        pass2_time_s += time.time() - t0
        halfset_results[int(halfset_idx)] = result
        accumulators.extend(
            _arrays_to_accumulators(
                result.Ft_y,
                result.Ft_ctf,
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
    """Convert recovar-frame InitialModel references to unnormalised centered FFTs for dense scoring."""
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
        means.append(np.asarray(ftu.get_dft3(corrected).reshape(-1)))
    return np.asarray(means, dtype=np.complex64)


def _resolve_class_inputs(
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
) -> tuple[Any, Any]:
    if config.means is not None:
        means = config.means
    elif config.relion_projector_frame:
        means = reference_to_relion_projector_dense_means(
            state.Iref,
            current_size=state.current_size if state.current_size > 0 else state.ori_size,
            padding_factor=config.padding_factor,
        )
    else:
        means = reference_to_dense_means(state.Iref)
    mean_variance = config.mean_variance if config.mean_variance is not None else np.abs(np.asarray(means)) ** 2
    return means, mean_variance


def _empty_accumulator(state: InitialModelState, class_idx: int, halfset_idx: int) -> VdamAccumulator:
    r_max = state.ori_size // 2 if state.current_size <= 0 else state.current_size // 2
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
    dump_dir = os.environ.get("RECOVAR_INITIAL_MODEL_ACCUM_DUMP_DIR")

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
        if dump_dir:
            path = Path(dump_dir)
            path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path / f"accum_h{int(halfset_idx)}_k{int(k)}.npz",
                Ft_y=np.asarray(Ft_y_by_class[k]),
                Ft_ctf=np.asarray(Ft_ctf_by_class[k]),
                bp_data_unscaled=np.asarray(bp_data),
                bp_weight_unscaled=np.asarray(bp_weight),
                bp_data_scaled=np.asarray(bp_data * data_scale),
                bp_weight_scaled=np.asarray(bp_weight * weight_scale),
                data_scale=np.float64(data_scale),
                weight_scale=np.float64(weight_scale),
                relion_projector_frame=np.bool_(relion_projector_frame),
                relion_bpref_frame=np.bool_(relion_bpref_frame),
                padding_factor=np.int32(padding_factor),
                ori_size=np.int32(state.ori_size),
                current_size=np.int32(state.current_size),
            )
        accumulators.append(
            VdamAccumulator(
                data=bp_data * data_scale,
                weight=bp_weight * weight_scale,
                class_idx=k,
                halfset_idx=halfset_idx,
            )
        )
    return accumulators


def _estep_meta(halfset_results: dict[int, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {"halfset_ids": tuple(sorted(halfset_results))}
    class_posterior_sums = None
    class_direction_posterior_sums = None
    noise_totals: dict[str, Any] | None = None
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
        noise_stats = getattr(result, "aggregate_noise_stats", None)
        if noise_stats is not None:
            half = {
                "wsum_sigma2_offset": float(noise_stats.wsum_sigma2_offset),
                "sigma2_offset_sumw": float(noise_stats.sumw),
                "wsum_sigma2_noise": np.asarray(noise_stats.wsum_sigma2_noise, dtype=np.float64),
                "wsum_img_power": np.asarray(noise_stats.wsum_img_power, dtype=np.float64),
                "noise_sumw": float(noise_stats.sumw),
            }
            for k, v in half.items():
                meta[f"halfset_{h}_{k}"] = v
            noise_totals = dict(half) if noise_totals is None else {k: noise_totals[k] + half[k] for k in half}
        per_class_stats = getattr(result, "per_class_stats", None)
        if per_class_stats is not None:
            direction_sums = np.stack(
                [np.asarray(cs.rotation_posterior_sums, dtype=np.float64) for cs in per_class_stats],
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
    if noise_totals is not None:
        meta.update(noise_totals)
    return meta


def run_dense_initial_model_estep(
    experiment_dataset,
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
    *,
    particle_ids: np.ndarray | None = None,
    halfset_ids: np.ndarray | None = None,
) -> DenseInitialModelEstepResult:
    """Run the InitialModel E-step; pseudo-halfsets run as separate E-steps with shared priors."""
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
            accumulate_noise=True,
            **_dense_run_em_kwargs(
                _group_local_kwargs(
                    engine_kwargs,
                    image_indices,
                    n_images=int(experiment_dataset.n_images),
                )
            ),
        )
        halfset_results[halfset_idx] = result
        by_halfset[halfset_idx] = _arrays_to_accumulators(
            result.Ft_y,
            result.Ft_ctf,
            state,
            halfset_idx=halfset_idx,
            relion_bpref_frame=config.relion_bpref_frame,
            relion_projector_frame=config.relion_projector_frame,
            padding_factor=config.padding_factor,
        )

    accumulators: list[VdamAccumulator] = []
    for halfset_idx in sorted(by_halfset):
        accumulators.extend(by_halfset[halfset_idx])

    selected_particle_ids: list[np.ndarray] = []
    field_lists: dict[str, list[np.ndarray]] = {attr: [] for attr, _ in _SPARSE_PASS2_RESULT_FIELDS}
    max_posterior: list[np.ndarray] = []
    for halfset_idx, image_indices in groups:
        result = halfset_results.get(halfset_idx)
        if result is None:
            continue
        attrs = {attr: getattr(result, attr, None) for attr, _ in _SPARSE_PASS2_RESULT_FIELDS}
        stats = getattr(result, "stats", None)
        pmax = None if stats is None else getattr(stats, "max_posterior_per_image", None)
        if pmax is None and all(v is None for v in attrs.values()):
            continue
        selected_particle_ids.append(np.asarray(image_indices, dtype=np.int64))
        for attr, dtype in _SPARSE_PASS2_RESULT_FIELDS:
            if attrs[attr] is not None:
                field_lists[attr].append(np.asarray(attrs[attr], dtype=dtype))
        if pmax is not None:
            max_posterior.append(np.asarray(pmax, dtype=np.float32))

    meta = _estep_meta(halfset_results)
    if selected_particle_ids:
        meta["selected_particle_ids"] = np.concatenate(selected_particle_ids).astype(np.int64, copy=False)
    for attr, dtype in _SPARSE_PASS2_RESULT_FIELDS:
        if field_lists[attr]:
            meta[attr] = np.concatenate(field_lists[attr]).astype(dtype, copy=False)
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
