#!/usr/bin/env python
"""Replay one RELION Class3D iteration with RECOVAR K-class dense EM.

This is the small, direct parity harness for K-class semantics.  It compares
RECOVAR's joint class x pose posterior against RELION Class3D for a fixed
iteration, without going through the single-class auto-refine/half-set replay
script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def stack_index_from_image_name(name: str) -> int:
    match = re.match(r"(\d+)@", str(name))
    return int(match.group(1)) - 1 if match else -1


def _star_particles(star_data):
    return star_data["particles"] if isinstance(star_data, dict) and "particles" in star_data else star_data


@dataclass(frozen=True)
class _ReplayBatchPlan:
    image_batch_size: int
    rotation_block_size: int
    requested_image_batch_size: int
    requested_rotation_block_size: int


def _safe_k_class_replay_batch_plan(
    *,
    requested_image_batch_size: int,
    requested_rotation_block_size: int,
    n_rot: int,
    n_trans: int,
    n_classes: int,
    image_shape,
    volume_shape,
    padding_factor: int,
    current_size: int | None,
) -> _ReplayBatchPlan:
    """Mirror the main RELION replay loop's K-class microbatch planner."""

    from recovar.em.dense_single_volume.batch_planning import _estimate_relion_em_batch_sizes
    from recovar.em.dense_single_volume.firstiter_cc import (
        _safe_dense_k_class_rotation_block_size,
        _safe_firstiter_cc_image_batch_size,
    )

    plan = _estimate_relion_em_batch_sizes(
        requested_image_batch_size=requested_image_batch_size,
        requested_rotation_block_size=requested_rotation_block_size,
        n_rot=n_rot,
        n_trans=n_trans,
        image_shape=image_shape,
        volume_shape=volume_shape,
        padding_factor=padding_factor,
        n_classes=n_classes,
        current_size=current_size,
    )
    image_batch_size = min(
        int(plan.image_batch_size),
        _safe_firstiter_cc_image_batch_size(n_trans, image_shape),
    )
    rotation_block_size = min(
        int(plan.rotation_block_size),
        _safe_dense_k_class_rotation_block_size(n_trans, image_batch_size),
    )
    return _ReplayBatchPlan(
        image_batch_size=max(1, int(image_batch_size)),
        rotation_block_size=max(1, int(rotation_block_size)),
        requested_image_batch_size=max(1, int(requested_image_batch_size)),
        requested_rotation_block_size=max(1, int(requested_rotation_block_size)),
    )


def _batch_plan_note(label: str, plan: _ReplayBatchPlan) -> str:
    return (
        f"{label}: image_batch_size={plan.image_batch_size}"
        f"/{plan.requested_image_batch_size}, rotation_block_size={plan.rotation_block_size}"
        f"/{plan.requested_rotation_block_size}"
    )


def _relion_adaptive_coarse_image_size(
    *,
    healpix_order: int,
    pixel_size: float,
    grid_size: int,
    particle_diameter: float,
    current_size: int,
) -> int:
    """Return RELION's adaptive pass-1 ``image_coarse_size``."""

    from recovar.em.dense_single_volume.helpers.resolution import (
        clamp_relion_coarse_image_size,
        compute_coarse_image_size,
    )
    from recovar.em.sampling import relion_angular_sampling_deg

    coarse_size = compute_coarse_image_size(
        relion_angular_sampling_deg(healpix_order, adaptive_oversampling=0),
        pixel_size,
        grid_size,
        particle_diameter=particle_diameter,
    )
    return int(
        clamp_relion_coarse_image_size(
            coarse_size,
            current_size=current_size,
            ori_size=grid_size,
        )
    )


def _relion_adaptive_fine_translation_grid(
    base_translations: np.ndarray,
    offset_step_px: float,
    adaptive_oversampling: int,
    random_perturbation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RELION pass-2 fine translations and fine-to-coarse parent map."""

    from recovar.em.sampling import (
        apply_relion_translation_perturbation,
        get_oversampled_translation_grid,
    )

    fine_base_translations, trans_parent_map = get_oversampled_translation_grid(
        np.asarray(base_translations, dtype=np.float32),
        float(offset_step_px),
        oversampling_order=int(adaptive_oversampling),
    )
    # RELION applies SamplingPerturbation in units of the original coarse
    # offset step even for oversampled pass-2 children.
    fine_translations = apply_relion_translation_perturbation(
        fine_base_translations,
        float(random_perturbation),
        float(offset_step_px),
    ).astype(np.float32)
    return fine_translations, np.asarray(trans_parent_map, dtype=np.int64)


def _resolve_target_random_perturbation(
    *,
    star_value: float,
    perturbation_factor: float,
    random_seed: int | None,
    target_iteration: int,
    restart_state_iteration: int | None,
    precision_mode: str,
) -> tuple[float, str]:
    """Recover the live RELION perturbation used at a replay boundary."""
    if precision_mode == "star":
        return float(star_value), "star-rounded"
    if precision_mode != "seed_exact":
        raise ValueError(f"Unsupported perturbation precision mode: {precision_mode!r}")
    if random_seed is None:
        raise ValueError(
            "seed-exact perturbation replay requires _rlnRandomSeed in the "
            "previous optimiser STAR"
        )

    from recovar.em.sampling import relion_sampling_perturbation_for_iteration

    exact = relion_sampling_perturbation_for_iteration(
        float(perturbation_factor),
        int(random_seed),
        int(target_iteration),
        restart_state_iteration=restart_state_iteration,
    )
    if not np.isclose(exact, float(star_value), rtol=0.0, atol=5.1e-6):
        restart_note = (
            "none"
            if restart_state_iteration is None
            else str(int(restart_state_iteration))
        )
        raise ValueError(
            "Seed-reconstructed SamplingPerturbation disagrees with the target "
            "sampling STAR; if this target was produced by a RELION continuation, "
            "pass --perturb-restart-state-iteration. "
            f"target_iteration={int(target_iteration)} seed={int(random_seed)} "
            f"restart_state_iteration={restart_note} exact={exact:+.12g} "
            f"star={float(star_value):+.12g}"
        )
    source = "seed-exact"
    if restart_state_iteration is not None:
        source = f"seed-exact-restart@{int(restart_state_iteration)}"
    return float(exact), source


def _scalar(table_or_dict, name: str, default=None):
    if table_or_dict is None:
        if default is None:
            raise KeyError(name)
        return default
    if isinstance(table_or_dict, dict):
        if name not in table_or_dict:
            if default is None:
                raise KeyError(name)
            return default
        return table_or_dict[name]
    if name not in table_or_dict.columns:
        if default is None:
            raise KeyError(name)
        return default
    return table_or_dict[name].iloc[0]


def _class_table(model, class_index: int):
    key = f"model_class_{class_index + 1}"
    if key not in model:
        raise ValueError(f"Missing {key} in RELION model STAR")
    return model[key]


def _tau_spectrum(model, class_index: int) -> np.ndarray:
    """Read RELION's per-class signal power spectrum used in BackProjector reconstruction.

    RELION's ``BackProjector::reconstruct(..., tau2, tau2_fudge, ...)`` is
    invoked at ``ml_optimiser.cpp:6020`` with ``mymodel.tau2_class[iclass]``
    which is the per-class **signal** power spectrum, written to model.star
    under the ``rlnReferenceTau2`` column (EMDL_MLMODEL_TAU2_REF).
    The ``rlnReferenceSigma2`` column is the **noise** power spectrum
    (EMDL_MLMODEL_SIGMA2_REF), which is roughly 60x smaller and is used
    only for diagnostics — using it as the Wiener regulariser would
    over-regularise low-occupancy classes 60x more than RELION does and
    produces the K=4 chained iter≥2 amplitude-deficit pattern.
    """
    table = _class_table(model, class_index)
    if "rlnReferenceTau2" in table.columns:
        column = "rlnReferenceTau2"
    elif "rlnReferenceSigma2" in table.columns:
        column = "rlnReferenceSigma2"
    else:
        raise ValueError(f"Missing reference variance column in model_class_{class_index + 1}")
    return np.asarray(table[column], dtype=np.float64)


def _read_particle_diameter(relion_dir: Path, prev_iter: int) -> float:
    optimiser_path = relion_dir / f"run_it{prev_iter:03d}_optimiser.star"
    text = optimiser_path.read_text()
    match = re.search(r"_rlnParticleDiameter\s+(\S+)", text)
    if not match:
        raise ValueError(f"Missing _rlnParticleDiameter in {optimiser_path}")
    return float(match.group(1))


def _read_relion_optimiser_cli_flags(relion_dir: Path, prev_iter: int) -> dict[str, object]:
    """Extract first-iteration mode flags from RELION's optimiser STAR header."""

    optimiser_path = relion_dir / f"run_it{prev_iter:03d}_optimiser.star"
    text = optimiser_path.read_text()
    cli_line = next(
        (line.lstrip("#").strip() for line in text.splitlines() if line.lstrip().startswith("# --")),
        "",
    )
    ini_high_match = re.search(r"(?:^|\s)--ini_high\s+(\S+)", cli_line)
    return {
        "path": str(optimiser_path),
        "cli_line": cli_line,
        "do_firstiter_cc": bool(re.search(r"(?:^|\s)--firstiter_cc(?:\s|$)", cli_line)),
        "ini_high_angstrom": float(ini_high_match.group(1)) if ini_high_match else None,
    }


def _resolve_firstiter_cc_mode(args, relion_cli_flags: dict[str, object]) -> dict[str, object]:
    relion_requested = (
        int(args.prev_iter) == 0
        and int(args.target_iter) == 1
        and bool(relion_cli_flags.get("do_firstiter_cc", False))
    )
    mode = str(args.firstiter_cc_mode)
    forced_by_legacy_flag = bool(args.winner_take_all_mstep)
    if forced_by_legacy_flag:
        mode = "force"
    if mode == "auto":
        emulate = relion_requested
    elif mode == "force":
        emulate = True
    elif mode == "off":
        emulate = False
    else:  # pragma: no cover - argparse choices should prevent this.
        raise ValueError(f"Unknown firstiter CC mode: {mode}")
    return {
        "requested_mode": str(args.firstiter_cc_mode),
        "effective_mode": mode,
        "forced_by_winner_take_all_mstep": forced_by_legacy_flag,
        "relion_requested": bool(relion_requested),
        "emulate": bool(emulate),
        "score_mode": "normalized_cc" if emulate else "gaussian",
    }


def _resolve_firstiter_lowpass_ini_high_angstrom(args, relion_cli_flags: dict[str, object], firstiter_cc_mode):
    """Return the effective RELION iter-1 low-pass cutoff, or ``None``.

    RELION only reapplies ``initialLowPassFilterReferences`` after the
    firstiter-CC iteration when the original command included a positive
    ``--ini_high``. Class3D GUI runs commonly have ``--firstiter_cc`` without
    ``--ini_high``; defaulting those to 30 A silently over-filters iter-1.
    """

    if not bool(firstiter_cc_mode.get("emulate", False)):
        return None
    override = getattr(args, "firstiter_cc_ini_high_angstrom", None)
    if override is not None:
        value = float(override)
        return value if value > 0.0 else None
    value = relion_cli_flags.get("ini_high_angstrom")
    if value is None:
        return None
    value = float(value)
    return value if value > 0.0 else None


def _class_distributions(model) -> np.ndarray:
    classes = model["model_classes"]
    return np.asarray(classes["rlnClassDistribution"], dtype=np.float64)


def _read_class_direction_priors(model, n_classes: int) -> np.ndarray:
    priors = []
    for class_index in range(n_classes):
        key = f"model_pdf_orient_class_{class_index + 1}"
        if key not in model:
            raise ValueError(f"Missing {key} in RELION model STAR")
        priors.append(np.asarray(model[key]["rlnOrientationDistribution"], dtype=np.float32))
    return np.stack(priors, axis=0)


def _split_class_direction_prior_for_replay(direction_prior, n_classes: int):
    """Mirror production's conditional direction prior plus class-prior split."""
    arr = np.asarray(direction_prior, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != int(n_classes):
        raise ValueError(f"class direction prior must have shape ({int(n_classes)}, n_dirs), got {arr.shape}")
    if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError("class direction prior entries must be finite and non-negative")
    row_sums = arr.sum(axis=1, dtype=np.float64)
    if np.any(row_sums <= 0.0):
        raise ValueError("each class direction prior row must have positive mass")
    total = float(row_sums.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("class direction prior row sums must have positive finite total mass")
    conditional = (arr / row_sums[:, None].astype(np.float32)).astype(np.float32)
    class_weights = (row_sums / total).astype(np.float64)
    class_log_priors = np.log(class_weights).astype(np.float32)
    return conditional, class_log_priors, class_weights.astype(np.float32)


def _image_name_order(data_star: Path, starfile):
    particles = _star_particles(starfile.read(str(data_star)))
    return list(particles["rlnImageName"])


def _dataframe_in_dataset_order(relion_df, dataset_names):
    relion_names = list(relion_df["rlnImageName"])
    relion_by_stack = {stack_index_from_image_name(name): row for row, name in enumerate(relion_names)}
    rows = []
    missing = []
    for name in dataset_names:
        stack_idx = stack_index_from_image_name(name)
        row = relion_by_stack.get(stack_idx)
        if row is None:
            missing.append(name)
        else:
            rows.append(row)
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        raise ValueError(f"RELION data STAR missing {len(missing)} particles from dataset order: {preview}")
    return relion_df.iloc[np.asarray(rows, dtype=np.int64)].reset_index(drop=True)


def _image_and_scale_corrections(model, relion_df_ordered) -> tuple[np.ndarray, np.ndarray]:
    normcorr = np.asarray(relion_df_ordered["rlnNormCorrection"], dtype=np.float64)
    avg_norm = float(_scalar(model["model_general"], "rlnNormCorrectionAverage", 1.0))
    groups = model.get("model_groups", None)
    if groups is not None and "rlnGroupScaleCorrection" in groups.columns:
        group_scales = np.asarray(groups["rlnGroupScaleCorrection"], dtype=np.float64)
    else:
        group_scales = np.asarray([1.0], dtype=np.float64)
    if "rlnGroupNumber" in relion_df_ordered.columns:
        group_numbers = np.asarray(relion_df_ordered["rlnGroupNumber"], dtype=np.int64)
    else:
        group_numbers = np.ones(len(relion_df_ordered), dtype=np.int64)
    scale = group_scales[np.clip(group_numbers - 1, 0, len(group_scales) - 1)]
    return ((avg_norm / normcorr) * scale).astype(np.float32), scale.astype(np.float32)


def _apply_runtime_scale_dump_override(
    image_corrections: np.ndarray,
    scale_corrections: np.ndarray,
    relion_df_ordered,
    dump_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Override one image's scale with the value dumped from RELION scoring."""
    if dump_dir is None:
        return image_corrections, scale_corrections
    stack_path = dump_dir / "pass0_acc_stack_index.bin"
    scale_path = dump_dir / "pass0_img0_scale_correction.bin"
    if not stack_path.exists() or not scale_path.exists():
        raise FileNotFoundError(
            f"--runtime-scale-dump-dir must contain {stack_path.name} and {scale_path.name}: {dump_dir}"
        )
    stack_values = np.fromfile(stack_path, dtype=np.float64)
    scale_values = np.fromfile(scale_path, dtype=np.float64)
    if stack_values.size != 1 or scale_values.size != 1:
        raise ValueError(
            f"Expected one dumped stack/scale value in {dump_dir}, got "
            f"{stack_values.size} and {scale_values.size}"
        )
    relion_stack = int(round(float(stack_values[0])))
    target_stack = relion_stack - 1
    runtime_scale = float(scale_values[0])
    stack_indices = np.asarray(
        [stack_index_from_image_name(name) for name in relion_df_ordered["rlnImageName"]],
        dtype=np.int64,
    )
    matches = np.flatnonzero(stack_indices == target_stack)
    if matches.size != 1:
        raise ValueError(
            f"Expected exactly one dataset image with RELION stack index {target_stack}, got {matches.size}"
        )
    idx = int(matches[0])
    old_scale = float(scale_corrections[idx])
    old_image_corr = float(image_corrections[idx])
    if old_scale == 0.0:
        raise ValueError(f"Cannot rescale image_corrections for zero old scale at dataset row {idx}")
    out_scale = np.array(scale_corrections, copy=True)
    out_image = np.array(image_corrections, copy=True)
    out_scale[idx] = np.float32(runtime_scale)
    out_image[idx] = np.float32(out_image[idx] * (runtime_scale / old_scale))
    print(
        "  runtime scale override: "
        f"relion_stack={relion_stack}, zero_based_stack={target_stack}, dataset_row={idx}, "
        f"scale {old_scale:.9g}->{runtime_scale:.9g}, "
        f"image_corr {old_image_corr:.9g}->{float(out_image[idx]):.9g}"
    )
    return out_image, out_scale


def _previous_translations_pixels(relion_df_ordered, pixel_size: float) -> np.ndarray | None:
    if "rlnOriginXAngst" not in relion_df_ordered.columns or "rlnOriginYAngst" not in relion_df_ordered.columns:
        return None
    return np.stack(
        [
            np.asarray(relion_df_ordered["rlnOriginXAngst"], dtype=np.float64) / pixel_size,
            np.asarray(relion_df_ordered["rlnOriginYAngst"], dtype=np.float64) / pixel_size,
        ],
        axis=1,
    ).astype(np.float32)


def _volume_corr(lhs, rhs) -> float:
    lhs = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    lhs = lhs - lhs.mean()
    rhs = rhs - rhs.mean()
    denom = np.linalg.norm(lhs) * np.linalg.norm(rhs)
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(lhs, rhs) / denom)


def _best_class_permutation(recovar_real, relion_real):
    n_classes = len(recovar_real)
    corr_matrix = np.asarray(
        [
            [_volume_corr(recovar_real[recovar_idx], relion_real[relion_idx]) for relion_idx in range(n_classes)]
            for recovar_idx in range(n_classes)
        ],
        dtype=np.float64,
    )
    finite_corr_matrix = np.nan_to_num(corr_matrix, nan=-2.0, posinf=-2.0, neginf=-2.0)
    try:
        from scipy.optimize import linear_sum_assignment

        # Correlations are bounded by [-1, 1]. Use a finite sentinel so
        # scipy's Hungarian solver never sees infinite assignment costs.
        rows, cols = linear_sum_assignment(-finite_corr_matrix)
        perm = np.empty(n_classes, dtype=np.int64)
        perm[rows] = cols
    except Exception:
        # Fallback for environments without scipy. This is not globally
        # optimal for large K, but keeps diagnostics usable instead of
        # factorially exploding.
        perm = np.full(n_classes, -1, dtype=np.int64)
        unused = set(range(n_classes))
        for recovar_idx in range(n_classes):
            best_relion = max(unused, key=lambda relion_idx: finite_corr_matrix[recovar_idx, relion_idx])
            perm[recovar_idx] = best_relion
            unused.remove(best_relion)
    corrs = [float(corr_matrix[recovar_idx, perm[recovar_idx]]) for recovar_idx in range(n_classes)]
    finite_corrs = [corr for corr in corrs if np.isfinite(corr)]
    return {
        "recovar_to_relion": [int(idx) for idx in perm],
        "map_correlations": corrs,
        "mean_corr": float(np.mean(finite_corrs)) if finite_corrs else float("nan"),
        "nonfinite_corr_count": int(np.size(corr_matrix) - np.count_nonzero(np.isfinite(corr_matrix))),
        "chosen_nonfinite_corr_count": int(sum(not np.isfinite(corr) for corr in corrs)),
    }


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


def _recovar_centered_half_to_relion_current(rows_half, image_shape, current_size):
    """Convert RECOVAR centered-row half spectra to RELION current-size FFTW rows."""

    rows_half = np.asarray(rows_half)
    leading = rows_half.shape[:-1]
    height, width = map(int, image_shape)
    half_width = width // 2 + 1
    current_size = int(current_size)
    current_half = current_size // 2 + 1
    full = rows_half.reshape(*leading, height, half_width)
    natural_rows = np.fft.ifftshift(full, axes=-2)
    out = np.zeros((*leading, current_size, current_half), dtype=rows_half.dtype)
    positive_rows = current_size // 2 + 1
    out[..., :positive_rows, :] = natural_rows[..., :positive_rows, :current_half]
    negative_rows = current_size - positive_rows
    if negative_rows:
        out[..., positive_rows:, :] = natural_rows[..., height - negative_rows :, :current_half]
    return out


def _relion_bpref_maps_from_sparse_support(
    experiment_dataset,
    means,
    noise_variance,
    translations,
    significant_sample_indices_by_class,
    normalization_log_z,
    *,
    nside_level,
    disc_type,
    current_size,
    translation_step,
    class_rotation_log_prior,
    translation_log_prior,
    score_with_masked_images,
    half_spectrum_scoring,
    projection_padding_factor,
    reconstruction_padding_factor,
    image_corrections,
    scale_corrections,
    image_pre_shifts,
    use_float64_scoring,
    do_gridding_correction,
    square_window,
    random_perturbation,
    tau2_spectra,
    tau2_fudge,
    minres_map,
    max_images_per_microbatch,
):
    """Diagnostic: use RELION BackProjector with RECOVAR's joint posterior."""

    import jax
    import jax.numpy as jnp

    from recovar.core.configs import ForwardModelConfig
    from recovar.em.dense_single_volume.helpers.batch_fetch import fetch_indexed_batch
    from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
    from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
    from recovar.em.dense_single_volume.helpers.half_spectrum import make_scoring_half_image_weights
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_projections_block as _compute_projections_block,
    )
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _bucket_pass2_inputs,
        _build_bucket_arrays,
        _normalize_pass2_bucket_with_log_z,
        _prepare_bucket_io,
        _prepare_per_image_pass2_inputs,
        _reorder_to_indices,
        _score_pass2_bucket_relion_gpu_diff2,
    )
    from recovar.em.dense_single_volume.local_backprojection import (
        compute_local_ctf_sums,
        compute_local_weighted_sums,
        flatten_bucket_rotations,
    )
    from recovar.em.sampling import get_oversampled_translation_grid, rotation_grid_size
    from recovar.reconstruction import noise as noise_utils
    from recovar.relion_bind import _relion_bind_core as bind
    from recovar.utils import helpers

    image_shape = tuple(map(int, experiment_dataset.image_shape))
    volume_shape = tuple(map(int, experiment_dataset.volume_shape))
    n_images = int(experiment_dataset.n_units)
    n_classes = int(np.asarray(means).shape[0])
    grid_size = int(image_shape[0])
    n_half = int(image_shape[0] * (image_shape[1] // 2 + 1))
    n4 = float(grid_size**4)
    image_fft_norm = float(grid_size**2)
    n_coarse_trans = int(np.asarray(translations).shape[0])
    n_coarse_rot = int(rotation_grid_size(nside_level))

    if projection_padding_factor > 1:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        projection_volumes = [
            pad_volume_for_projection(
                means[class_index],
                volume_shape,
                projection_padding_factor,
                do_gridding_correction=do_gridding_correction,
                current_size=current_size,
            )
            for class_index in range(n_classes)
        ]
    else:
        projection_volumes = [(means[class_index], volume_shape) for class_index in range(n_classes)]

    fine_translations, fine_translation_parent = get_oversampled_translation_grid(
        np.asarray(translations, dtype=np.float32),
        translation_step,
        oversampling_order=0,
    )
    fine_translations = np.asarray(fine_translations, dtype=np.float32)
    fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    n_fine_trans = int(fine_translations.shape[0])

    if translation_log_prior is None:
        fine_translation_prior_2d = None
    else:
        translation_log_prior_np = np.asarray(translation_log_prior, dtype=np.float32)
        if translation_log_prior_np.ndim == 1:
            fine_tp = translation_log_prior_np[fine_translation_parent]
            fine_translation_prior_2d = np.broadcast_to(fine_tp, (n_images, n_fine_trans)).astype(
                np.float32,
                copy=False,
            )
        elif translation_log_prior_np.ndim == 2:
            fine_translation_prior_2d = translation_log_prior_np[:, fine_translation_parent].astype(
                np.float32,
                copy=False,
            )
        else:
            raise ValueError("translation_log_prior must be 1D or 2D")

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    precision_policy = DensePrecisionPolicy(use_float64_scoring=use_float64_scoring)
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=True,
    )
    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
    )
    half_weights_windowed = window_spec.score_values(half_weights)
    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        half_weights_windowed = window_spec.score_values(half_weights)

    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()
    normalization_log_z_np = np.asarray(normalization_log_z, dtype=np.float64)
    if normalization_log_z_np.shape != (n_images,):
        raise ValueError(f"normalization_log_z must have shape ({n_images},), got {normalization_log_z_np.shape}")

    maps = []
    summaries = []
    for class_index in range(n_classes):
        per_image_inputs = _prepare_per_image_pass2_inputs(
            significant_sample_indices_by_class[class_index],
            n_coarse_rot=n_coarse_rot,
            n_coarse_trans=n_coarse_trans,
            nside_level=nside_level,
            oversampling_order=0,
            n_fine_trans=n_fine_trans,
            fine_translation_parent=fine_translation_parent,
            rotation_log_prior=class_rotation_log_prior[class_index],
            random_perturbation=random_perturbation,
        )
        buckets = _bucket_pass2_inputs(
            per_image_inputs,
            n_fine_trans=n_fine_trans,
            rotation_block_size_for_quantization=5000,
            max_hypotheses_per_microbatch=100_000,
            max_images_per_microbatch=max_images_per_microbatch,
        )
        row_image_chunks = []
        row_weight_chunks = []
        row_rotation_chunks = []
        mean_for_proj, proj_volume_shape = projection_volumes[class_index]
        for bucket_meta in buckets:
            bucket_arrays = _build_bucket_arrays(bucket_meta, per_image_inputs, n_fine_trans)
            image_indices = bucket_arrays["image_indices"]
            batch_data, ctf_params, fetched_indices = fetch_indexed_batch(experiment_dataset, image_indices)
            batch_data = jnp.asarray(batch_data)
            if not np.array_equal(np.asarray(fetched_indices), image_indices):
                (
                    rotations,
                    log_prior,
                    candidate_mask,
                    parent_map_padded,
                    actual_counts,
                ) = _reorder_to_indices(
                    np.asarray(fetched_indices),
                    image_indices,
                    bucket_arrays["rotations"],
                    bucket_arrays["log_prior"],
                    bucket_arrays["candidate_mask"],
                    bucket_arrays["parent_map"],
                    bucket_arrays["actual_counts"],
                )
                image_indices = np.asarray(fetched_indices)
            else:
                rotations = bucket_arrays["rotations"]
                log_prior = bucket_arrays["log_prior"]
                candidate_mask = bucket_arrays["candidate_mask"]
                parent_map_padded = bucket_arrays["parent_map"]
                actual_counts = bucket_arrays["actual_counts"]
            del parent_map_padded

            if fine_translation_prior_2d is None:
                bucket_translation_prior = jnp.zeros((len(image_indices), n_fine_trans), dtype=jnp.float32)
            else:
                bucket_translation_prior = jnp.asarray(fine_translation_prior_2d[image_indices], dtype=jnp.float32)

            (
                shifted_score_half,
                shifted_recon_half,
                _batch_norm,
                ctf2_over_nv_half,
                ctf2_over_nv_half_with_dc,
                _shifted_score_half_with_dc,
                _processed_score_half_for_noise,
                shifted_corrected_score_half,
            ) = _prepare_bucket_io(
                experiment_dataset,
                batch_data,
                ctf_params,
                image_indices,
                noise_variance_half,
                fine_translations,
                config,
                n_fine_trans,
                score_with_masked_images,
                half_spectrum_scoring,
                image_corrections,
                scale_corrections,
                image_pre_shifts,
                use_float64_scoring,
                return_direct_scoring_io=True,
            )
            del shifted_score_half

            if window_spec.use_window:
                shifted_corrected_score = shifted_corrected_score_half[:, window_spec.score_indices]
                ctf2_over_nv_score = ctf2_over_nv_half[:, window_spec.score_indices]
            else:
                shifted_corrected_score = shifted_corrected_score_half
                ctf2_over_nv_score = ctf2_over_nv_half

            flat_rotations = flatten_bucket_rotations(jnp.asarray(rotations))
            projection_kwargs = window_spec.projection_kwargs(return_abs2=False if window_spec.use_window else None)
            proj_half_flat, proj_abs2_half_flat = _compute_projections_block(
                mean_for_proj,
                flat_rotations,
                image_shape,
                proj_volume_shape,
                disc_type,
                **projection_kwargs,
            )
            if window_spec.use_window:
                proj_half = proj_half_flat[:, window_spec.score_indices].reshape(
                    len(image_indices),
                    int(bucket_arrays["bucket_size"]),
                    window_spec.n_score,
                )
            else:
                proj_half = proj_half_flat.reshape(len(image_indices), int(bucket_arrays["bucket_size"]), n_half)
            del proj_abs2_half_flat

            scores = _score_pass2_bucket_relion_gpu_diff2(
                shifted_corrected_score.reshape(len(image_indices), n_fine_trans, -1),
                ctf2_over_nv_score,
                proj_half,
                half_weights_windowed if window_spec.use_window else half_weights,
                jnp.asarray(log_prior),
                bucket_translation_prior,
                jnp.asarray(candidate_mask),
            )
            _, probs, _, _, _ = _normalize_pass2_bucket_with_log_z(
                scores,
                jnp.asarray(normalization_log_z_np[image_indices], dtype=scores.real.dtype),
            )

            shifted_recon_split = shifted_recon_half.reshape(len(image_indices), n_fine_trans, n_half)
            summed = compute_local_weighted_sums(probs, shifted_recon_split) * image_fft_norm
            ctf_probs = compute_local_ctf_sums(probs, ctf2_over_nv_half_with_dc) * n4
            summed_relion = _recovar_centered_half_to_relion_current(
                np.asarray(jax.device_get(summed)),
                image_shape,
                current_size,
            )
            ctf_relion = _recovar_centered_half_to_relion_current(
                np.asarray(jax.device_get(ctf_probs)),
                image_shape,
                current_size,
            ).real

            rotations_np = np.asarray(rotations, dtype=np.float64)
            candidate_mask_np = np.asarray(candidate_mask, dtype=bool)
            for row, count in enumerate(np.asarray(actual_counts, dtype=np.int64)):
                if count <= 0:
                    continue
                keep = np.any(candidate_mask_np[row, :count, :], axis=1)
                if not np.any(keep):
                    continue
                row_image_chunks.append(summed_relion[row, :count][keep])
                row_weight_chunks.append(ctf_relion[row, :count][keep])
                row_rotation_chunks.append(rotations_np[row, :count][keep])

        if not row_image_chunks:
            raise ValueError(f"No RELION BPref rows collected for class {class_index + 1}")
        # RECOVAR's CTF convention is opposite the RELION binding's native
        # Fctf convention; the production path compensates elsewhere, but this
        # diagnostic feeds pre-CTF-weighted rows directly into BackProjector.
        images = -np.concatenate(row_image_chunks, axis=0).astype(np.complex128, copy=False)
        weights = np.concatenate(row_weight_chunks, axis=0).astype(np.float64, copy=False)
        row_rotations = np.concatenate(row_rotation_chunks, axis=0).astype(np.float64, copy=False)
        print(
            f"  RELION BPref class {class_index + 1}: "
            f"{images.shape[0]} image/orientation rows, row_shape={images.shape[1:]}"
        )
        vol_relion = np.asarray(
            bind.backproject_and_reconstruct(
                images,
                row_rotations,
                weights,
                np.asarray(tau2_spectra[class_index], dtype=np.float64),
                int(volume_shape[0]),
                int(reconstruction_padding_factor),
                1,
                True,
                10,
                float(tau2_fudge),
                True,
                int(current_size),
                1.0,
                float(minres_map),
            )
        )
        maps.append(helpers.relion_volume_to_recovar(vol_relion.astype(np.float32, copy=False)))
        summaries.append(
            {
                "n_rows": int(images.shape[0]),
                "row_shape": [int(images.shape[1]), int(images.shape[2])],
                "max_abs_image": float(np.max(np.abs(images))),
                "max_weight": float(np.max(weights)),
            }
        )

    return maps, summaries


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-dir", required=True, type=Path)
    parser.add_argument("--data-star", required=True, type=Path)
    parser.add_argument("--prev-iter", type=int, default=0)
    parser.add_argument("--target-iter", type=int, default=1)
    parser.add_argument(
        "--perturb-replay-precision",
        choices=("seed_exact", "star"),
        default="seed_exact",
        help=(
            "Use the optimiser seed to recover RELION's live perturbation by "
            "default. star is a rounded diagnostic fallback and is not suitable "
            "for strict boundary parity."
        ),
    )
    parser.add_argument(
        "--perturb-restart-state-iteration",
        type=int,
        default=None,
        help=(
            "Saved iteration used to start a RELION continuation that produced "
            "the target iteration. Required for seed-exact replay across an "
            "explicit restart boundary."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-batch-size", type=int, default=250)
    parser.add_argument("--rotation-block-size", type=int, default=5000)
    parser.add_argument("--tau2-fudge", type=float, default=None)
    parser.add_argument("--projection-padding-factor", type=int, default=2)
    parser.add_argument("--reconstruction-padding-factor", type=int, default=2)
    parser.add_argument("--disc-type", default="linear_interp")
    parser.add_argument(
        "--winner-take-all-mstep",
        action="store_true",
        help=(
            "Deprecated diagnostic alias for --firstiter-cc-mode force. "
            "Use RELION first-iteration winner-take-all reconstruction weights "
            "while keeping soft evidence stats."
        ),
    )
    parser.add_argument(
        "--firstiter-cc-mode",
        choices=("auto", "force", "off"),
        default="auto",
        help=(
            "First-iteration CC emulation policy. auto reads RELION's optimiser "
            "CLI and only emulates --firstiter_cc when that command actually "
            "requested it; force is for diagnostics; off disables it."
        ),
    )
    parser.add_argument(
        "--no-firstiter-cc-pass2-only-best-coarse",
        action="store_true",
        help=(
            "Deprecated no-op retained for old diagnostics. The replay harness now keeps "
            "normalized-CC firstiter scoring on the regular adaptive significance support "
            "by default."
        ),
    )
    parser.add_argument(
        "--firstiter-cc-pass2-only-best-coarse",
        action="store_true",
        help=(
            "Diagnostic legacy shortcut: with firstiter_cc emulation, restrict pass-2 to "
            "the single best coarse pose's fine children. Patched RELION storeWavg dumps "
            "show this is not the production parity path."
        ),
    )
    parser.add_argument(
        "--significant-mstep",
        action="store_true",
        help="Reconstruct from the joint class x pose significant support, matching RELION pass-2 support.",
    )
    parser.add_argument(
        "--relion-bpref-mstep",
        action="store_true",
        help="Diagnostic: reconstruct using RELION BackProjector fed by RECOVAR's joint posterior.",
    )
    parser.add_argument(
        "--relion-bpref-max-images-per-microbatch",
        type=int,
        default=32,
        help=(
            "Maximum images per microbatch for the optional RELION BPref diagnostic. "
            "Lower values reduce direct translation-IO memory without changing the "
            "main RECOVAR replay."
        ),
    )
    parser.add_argument(
        "--significance-adaptive-fraction",
        type=float,
        default=0.999,
        help="Posterior mass retained when selecting significant class x pose samples.",
    )
    parser.add_argument(
        "--adaptive-2pass",
        action="store_true",
        help=(
            "Run RELION-style adaptive 2-pass: pass-1 coarse significance pruning + "
            "pass-2 oversampled fine grid evaluation with the pass-1 mask broadcast "
            "to fine children. Mirrors ml_optimiser.cpp::expectationOneParticle line 5022."
        ),
    )
    parser.add_argument(
        "--adaptive-oversampling",
        type=int,
        default=1,
        help="HEALPix subdivision and translation subdivision passes used by --adaptive-2pass.",
    )
    parser.add_argument(
        "--accumulate-noise",
        action="store_true",
        help="Accumulate RELION-style noise statistics during the replay E/M step.",
    )
    parser.add_argument(
        "--sparse-pass2",
        action="store_true",
        help="Use the sparse bucketed adaptive pass-2 path instead of dense pass-2.",
    )
    parser.add_argument(
        "--relion-x-half-mstep",
        action="store_true",
        help=(
            "Use the RELION Class3D x-half BPref accumulation layout in the "
            "replay M-step. Required for device-signature contribution audits."
        ),
    )
    parser.add_argument(
        "--square-window",
        dest="square_window",
        action="store_true",
        default=True,
        help=(
            "Use square Fourier scoring support for the main replay. This "
            "preserves the historical run_k_class_parity.py behavior."
        ),
    )
    parser.add_argument(
        "--radial-window",
        dest="square_window",
        action="store_false",
        help=(
            "Use the radial Fourier scoring support used by the full "
            "RELION/default refinement path."
        ),
    )
    parser.add_argument(
        "--firstiter-cc-ini-high-angstrom",
        type=float,
        default=None,
        help=(
            "Override RELION's iter-1 firstiter_cc low-pass cutoff. By default "
            "the replay reads --ini_high from the RELION optimiser CLI and "
            "applies no low-pass when that command did not include a positive "
            "--ini_high. Set to 0 to disable even when RELION had --ini_high; "
            "set to a positive value only for diagnostics."
        ),
    )
    parser.add_argument(
        "--runtime-scale-dump-dir",
        type=Path,
        default=None,
        help=(
            "Diagnostic parity aid: override the dumped RELION stack index's "
            "scale correction with pass0_img0_scale_correction.bin from this dump directory."
        ),
    )
    parser.add_argument(
        "--image-pre-shift-mode",
        choices=("relion", "flip", "none"),
        default="relion",
        help=(
            "Diagnostic parity aid for old-offset handling. relion uses the "
            "rounded previous offset; flip negates it; none disables image "
            "pre-shifts. Default matches the current production path."
        ),
    )
    parser.add_argument(
        "--stop-after-pass2-dump",
        action="store_true",
        help=(
            "Diagnostic-only: stop successfully as soon as RECOVAR_PASS2_DUMP_DIR "
            "has written the requested sparse pass-2 target dump. This avoids "
            "running the rest of the replay M-step when only score tensors are needed."
        ),
    )
    args = parser.parse_args()
    if args.stop_after_pass2_dump:
        if not os.environ.get("RECOVAR_PASS2_DUMP_DIR"):
            parser.error("--stop-after-pass2-dump requires RECOVAR_PASS2_DUMP_DIR")
        if not (
            os.environ.get("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES")
            or os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
        ):
            parser.error(
                "--stop-after-pass2-dump requires RECOVAR_PASS2_DUMP_ORIGINAL_INDICES "
                "or RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES"
            )
        os.environ["RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET"] = "1"
        os.environ["RECOVAR_K_CLASS_PARITY_STOP_AFTER_PASS2_DUMP"] = "1"

    import jax
    import jax.numpy as jnp
    import starfile

    from recovar import utils
    from recovar.core import fourier_transform_utils as ftu
    from recovar.core import mask
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.em.dense_single_volume.helpers.orientation_priors import (
        make_relion_direction_log_prior,
        make_relion_translation_log_prior,
        relion_translation_prior_center,
        relion_translation_search_base,
    )
    from recovar.em.dense_single_volume.helpers.oversampling import compute_pass2_stats_sparse
    from recovar.em.dense_single_volume.helpers.significance import _compute_k_class_significance_batched
    from recovar.em.dense_single_volume.iteration_loop import RELION_MINRES_MAP, _reconstruct_volume_eager
    from recovar.em.dense_single_volume.k_class import (
        run_dense_k_class_em,
    )
    from recovar.em.initial_model.dense_adapter import reference_to_relion_projector_half_maps
    from recovar.em.sampling import (
        apply_relion_rotation_perturbation_to_eulers,
        apply_relion_translation_perturbation,
        get_relion_rotation_grid_eulers,
        get_translation_grid,
        read_relion_optimiser_metadata,
        read_relion_sampling_metadata,
        relion_angular_sampling_deg,
    )
    from recovar.reconstruction import noise as recon_noise
    from recovar.utils import helpers
    from recovar.utils.helpers import write_relion_mrc

    relion_dir = args.relion_dir
    prev_prefix = relion_dir / f"run_it{args.prev_iter:03d}"
    target_prefix = relion_dir / f"run_it{args.target_iter:03d}"
    output_dir = args.output_dir or relion_dir.parent / "_agent_scratch" / f"k_class_replay_it{args.target_iter:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    prev_model = starfile.read(str(prev_prefix) + "_model.star")
    target_model = starfile.read(str(target_prefix) + "_model.star")
    prev_data = _star_particles(starfile.read(str(prev_prefix) + "_data.star"))
    target_data = _star_particles(starfile.read(str(target_prefix) + "_data.star"))
    dataset_names = _image_name_order(args.data_star, starfile)
    prev_data_ordered = _dataframe_in_dataset_order(prev_data, dataset_names)
    target_data_ordered = _dataframe_in_dataset_order(target_data, dataset_names)

    n_classes = int(_scalar(prev_model["model_general"], "rlnNrClasses"))
    if n_classes < 2:
        raise ValueError(f"This harness is for K-class parity; RELION model has K={n_classes}")
    grid_size = int(_scalar(prev_model["model_general"], "rlnOriginalImageSize"))
    pixel_size = float(_scalar(prev_model["model_general"], "rlnPixelSize"))
    current_size = int(_scalar(target_model["model_general"], "rlnCurrentImageSize"))
    tau2_fudge = float(args.tau2_fudge or _scalar(prev_model["model_general"], "rlnTau2FudgeFactor", 4.0))
    particle_diameter = _read_particle_diameter(relion_dir, args.prev_iter)
    relion_cli_flags = _read_relion_optimiser_cli_flags(relion_dir, args.prev_iter)
    firstiter_cc_mode = _resolve_firstiter_cc_mode(args, relion_cli_flags)
    firstiter_lowpass_ini_high = _resolve_firstiter_lowpass_ini_high_angstrom(
        args,
        relion_cli_flags,
        firstiter_cc_mode,
    )

    print(f"RELION K-class replay: K={n_classes}, N={grid_size}, prev={args.prev_iter}, target={args.target_iter}")
    print(f"  current_size={current_size}, pixel_size={pixel_size}, tau2_fudge={tau2_fudge}")
    print(f"  output_dir={output_dir}")
    print(f"  JAX devices: {jax.devices()}")
    print(
        "  firstiter_cc: "
        f"relion_requested={firstiter_cc_mode['relion_requested']}, "
        f"mode={firstiter_cc_mode['effective_mode']}, "
        f"emulate={firstiter_cc_mode['emulate']}, "
        f"score_mode={firstiter_cc_mode['score_mode']}"
    )
    print(
        "  firstiter_cc lowpass: "
        f"relion_ini_high={relion_cli_flags.get('ini_high_angstrom')}, "
        f"override={args.firstiter_cc_ini_high_angstrom}, "
        f"effective={firstiter_lowpass_ini_high}"
    )
    if firstiter_cc_mode["effective_mode"] == "force" and not firstiter_cc_mode["relion_requested"]:
        print(
            "  WARNING: forcing firstiter_cc emulation even though the RELION optimiser "
            "CLI did not contain --firstiter_cc; use only for diagnostics."
        )

    ds = load_dataset(str(args.data_star))
    backend = getattr(getattr(ds, "image_source", None), "backend", None)
    if backend is None:
        raise ValueError("Dataset backend is required for RELION image-mask parity")
    if hasattr(backend, "set_relion_image_mask"):
        backend.set_relion_image_mask(ds.voxel_size, particle_diameter, width_mask_edge_px=5.0)
    else:
        from recovar.core.mask import relion_soft_image_mask

        backend.image_mask = relion_soft_image_mask(grid_size, ds.voxel_size, particle_diameter, 5)
        backend.image_mask_mode = "relion_background_fill"

    n4 = grid_size**4
    noise_spectrum = np.asarray(prev_model["model_optics_group_1"]["rlnSigma2Noise"], dtype=np.float64)
    noise_variance = jnp.asarray(recon_noise.make_radial_noise(noise_spectrum * n4, (grid_size, grid_size))).reshape(-1)
    mean_variance_prev = jnp.stack(
        [
            jnp.asarray(
                utils.make_radial_image(_tau_spectrum(prev_model, k) * n4, (grid_size, grid_size, grid_size))
            ).reshape(-1)
            for k in range(n_classes)
        ],
        axis=0,
    )
    mean_variance_target = jnp.stack(
        [
            jnp.asarray(
                utils.make_radial_image(_tau_spectrum(target_model, k) * n4, (grid_size, grid_size, grid_size))
            ).reshape(-1)
            for k in range(n_classes)
        ],
        axis=0,
    )
    prev_reference_real = np.stack(
        [
            np.asarray(helpers.load_relion_volume(str(prev_prefix) + f"_class{k + 1:03d}.mrc"), dtype=np.float64)
            for k in range(n_classes)
        ],
        axis=0,
    )
    means = jnp.stack(
        [jnp.asarray(ftu.get_dft3(jnp.asarray(prev_reference_real[k]))).reshape(-1) for k in range(n_classes)],
        axis=0,
    )

    sampling = read_relion_sampling_metadata(str(target_prefix) + "_sampling.star")
    healpix_order = int(sampling["healpix_order"])
    star_random_perturbation = float(sampling["random_perturbation"])
    optimiser = read_relion_optimiser_metadata(str(prev_prefix) + "_optimiser.star")
    random_perturbation, random_perturbation_source = (
        _resolve_target_random_perturbation(
            star_value=star_random_perturbation,
            perturbation_factor=float(sampling["perturbation_factor"]),
            random_seed=optimiser.get("random_seed"),
            target_iteration=args.target_iter,
            restart_state_iteration=args.perturb_restart_state_iteration,
            precision_mode=args.perturb_replay_precision,
        )
    )
    offset_range_px = float(sampling["offset_range"]) / pixel_size
    offset_step_px = float(sampling["offset_step"]) / pixel_size
    rotations, _ = apply_relion_rotation_perturbation_to_eulers(
        get_relion_rotation_grid_eulers(healpix_order),
        random_perturbation,
        relion_angular_sampling_deg(healpix_order, adaptive_oversampling=0),
    )
    base_translations = get_translation_grid(offset_range_px, offset_step_px).astype(np.float32)
    translations = apply_relion_translation_perturbation(
        base_translations,
        random_perturbation,
        offset_step_px,
    ).astype(np.float32)
    print(
        "  sampling: "
        f"healpix={healpix_order}, rotations={rotations.shape[0]}, translations={translations.shape[0]}, "
        f"rp={random_perturbation:+.12g} source={random_perturbation_source}, "
        f"star_rp={star_random_perturbation:+.12g}, "
        f"offset_range_px={offset_range_px:.3f}, offset_step_px={offset_step_px:.3f}"
    )
    coarse_current_size = None
    coarse_engine_current_size = current_size
    if args.adaptive_2pass:
        coarse_current_size = _relion_adaptive_coarse_image_size(
            healpix_order=healpix_order,
            pixel_size=pixel_size,
            grid_size=grid_size,
            particle_diameter=particle_diameter,
            current_size=current_size,
        )
        coarse_engine_current_size = coarse_current_size if coarse_current_size < grid_size else None
        print(
            "  adaptive pass sizes: "
            f"coarse_current_size={coarse_engine_current_size}, fine_current_size={current_size}"
        )
    projector_current_size = current_size
    relion_projector_half_by_class, relion_projector_r_max = reference_to_relion_projector_half_maps(
        prev_reference_real,
        current_size=projector_current_size,
        padding_factor=args.projection_padding_factor,
    )
    print(
        "  exact RELION Projector::data: "
        f"current_size={projector_current_size}, r_max={relion_projector_r_max}, "
        f"shape={tuple(relion_projector_half_by_class.shape)}"
    )

    direction_prior = _read_class_direction_priors(prev_model, n_classes)
    direction_prior_conditional, class_log_priors, class_prior_weights = _split_class_direction_prior_for_replay(
        direction_prior,
        n_classes,
    )
    class_rotation_log_prior = np.stack(
        [make_relion_direction_log_prior(direction_prior_conditional[k], healpix_order) for k in range(n_classes)],
        axis=0,
    )
    image_corrections, scale_corrections = _image_and_scale_corrections(prev_model, prev_data_ordered)
    image_corrections, scale_corrections = _apply_runtime_scale_dump_override(
        image_corrections,
        scale_corrections,
        prev_data_ordered,
        args.runtime_scale_dump_dir,
    )
    previous_translations = _previous_translations_pixels(prev_data_ordered, pixel_size)
    image_pre_shifts = relion_translation_search_base(previous_translations)
    if args.image_pre_shift_mode == "flip" and image_pre_shifts is not None:
        image_pre_shifts = (-np.asarray(image_pre_shifts, dtype=np.float32)).astype(np.float32)
        print("  diagnostic image pre-shift mode: flip")
    elif args.image_pre_shift_mode == "none":
        image_pre_shifts = None
        print("  diagnostic image pre-shift mode: none")
    sigma_offset_angstrom = float(_scalar(prev_model["model_general"], "rlnSigmaOffsetsAngst"))
    translation_prior_centers = relion_translation_prior_center(previous_translations, pixel_size)
    translation_log_prior = make_relion_translation_log_prior(
        base_translations,
        pixel_size,
        sigma_offset_angstrom,
        translation_prior_centers,
        offset_range_pixels=None,
    )
    print(
        "  priors/corrections: "
        f"pdf_row_sums={direction_prior.sum(axis=1).round(6).tolist()}, "
        f"class_priors={class_prior_weights.round(6).tolist()}, "
        f"sigma_offset={sigma_offset_angstrom:.6f}A, "
        f"image_corr_mean={float(image_corrections.mean()):.6f}, scale_mean={float(scale_corrections.mean()):.6f}"
    )
    image_shape = tuple(int(s) for s in getattr(ds, "image_shape", (grid_size, grid_size)))
    volume_shape = tuple(int(s) for s in getattr(ds, "volume_shape", (grid_size, grid_size, grid_size)))
    base_batch_plan = _safe_k_class_replay_batch_plan(
        requested_image_batch_size=args.image_batch_size,
        requested_rotation_block_size=args.rotation_block_size,
        n_rot=int(rotations.shape[0]),
        n_trans=int(translations.shape[0]),
        n_classes=n_classes,
        image_shape=image_shape,
        volume_shape=volume_shape,
        padding_factor=args.reconstruction_padding_factor,
        current_size=coarse_engine_current_size if args.adaptive_2pass else current_size,
    )
    significance_support_batch_plan = base_batch_plan
    print("  batch sizing: " + _batch_plan_note("coarse", base_batch_plan))

    t0 = time.time()
    from recovar.em.dense_single_volume.helpers import (
        sparse_pass2_bucketed as _sparse_pass2_diagnostics,
    )

    # Mirror the numbered-half context supplied by the production iteration
    # loop so opt-in contribution/device-signature captures from this
    # one-boundary replay retain exact target-iteration provenance.
    _sparse_pass2_diagnostics.set_bpref_contribution_dump_context(
        iteration=args.target_iter,
        half=1,
    )
    common_em_kwargs = dict(
        class_log_priors=class_log_priors,
        accumulate_noise=bool(args.accumulate_noise),
        image_batch_size=base_batch_plan.image_batch_size,
        rotation_block_size=base_batch_plan.rotation_block_size,
        class_rotation_log_prior=class_rotation_log_prior,
        translation_log_prior=translation_log_prior,
        translation_prior_centers=translation_prior_centers,
        score_with_masked_images=True,
        half_spectrum_scoring=True,
        projection_padding_factor=args.projection_padding_factor,
        reconstruction_padding_factor=args.reconstruction_padding_factor,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        use_float64_scoring=False,
        use_float64_projections=False,
        do_gridding_correction=True,
        square_window=bool(args.square_window),
        sparse_pass2=bool(args.sparse_pass2),
        relion_firstiter_winner_take_all=bool(firstiter_cc_mode["emulate"]),
        # Match RELION's do_firstiter_cc branch in getAllSquaredDifferences
        # only when the optimiser CLI requested --firstiter_cc, unless the
        # caller explicitly forces the diagnostic mode.
        relion_firstiter_score_mode=str(firstiter_cc_mode["score_mode"]),
        relion_projector_half=relion_projector_half_by_class,
        relion_projector_r_max=relion_projector_r_max,
        mstep_relion_x_half=bool(args.relion_x_half_mstep),
    )
    if args.adaptive_2pass:
        # Build pass-2 fine grid (oversampled) using RELION-parity HEALPix children.
        # Mirrors ml_optimiser.cpp::expectationOneParticle line 5022 onward where
        # nr_sampling_passes=2 and exp_current_oversampling=adaptive_oversampling
        # for pass-2 only. Recovar evaluates the FULL fine grid but masks out
        # fine poses whose coarse parent did not survive pass-1's
        # adaptive_fraction pruning.
        from recovar.em.dense_single_volume.k_class import run_dense_k_class_em_adaptive
        from recovar.em.sampling import (
            get_oversampled_rotation_grid_from_samples,
        )

        adaptive_os = int(args.adaptive_oversampling)
        all_coarse_rot_indices = np.arange(int(rotations.shape[0]), dtype=np.int64)
        fine_rotations, rot_parent_map = get_oversampled_rotation_grid_from_samples(
            all_coarse_rot_indices,
            parent_nside_level=int(healpix_order),
            oversampling_order=adaptive_os,
            random_perturbation=random_perturbation,
        )
        fine_rotations = np.asarray(fine_rotations, dtype=np.float32)
        rot_parent_map = np.asarray(rot_parent_map, dtype=np.int64)
        # Translations: oversample base_translations (pre-perturbation) and
        # apply RELION's per-iteration perturbation to the fine grid the same
        # way as the coarse path.
        fine_translations, trans_parent_map = _relion_adaptive_fine_translation_grid(
            base_translations,
            offset_step_px,
            adaptive_os,
            random_perturbation,
        )
        fine_batch_plan = _safe_k_class_replay_batch_plan(
            requested_image_batch_size=args.image_batch_size,
            requested_rotation_block_size=args.rotation_block_size,
            n_rot=int(fine_rotations.shape[0]),
            n_trans=int(fine_translations.shape[0]),
            n_classes=n_classes,
            image_shape=image_shape,
            volume_shape=volume_shape,
            padding_factor=args.reconstruction_padding_factor,
            current_size=current_size,
        )
        significance_support_batch_plan = base_batch_plan
        print(
            "  adaptive 2-pass: fine grid "
            f"rotations={fine_rotations.shape[0]} (parents {rotations.shape[0]}, "
            f"max children/parent={int(np.bincount(rot_parent_map).max())}), "
            f"translations={fine_translations.shape[0]} (parents {translations.shape[0]}, "
            f"max children/parent={int(np.bincount(trans_parent_map).max())}), "
            f"adaptive_fraction={args.significance_adaptive_fraction:.4f}"
        )
        print(
            "  adaptive batch sizing: "
            + _batch_plan_note("coarse_pass1", base_batch_plan)
            + "; "
            + _batch_plan_note("fine_pass2", fine_batch_plan)
        )
        # RELION firstiter_cc uses normalized-CC scoring, but patched storeWavg
        # dumps retain a small adaptive pass-2 posterior support. The legacy
        # single-best-coarse shortcut is kept only as an explicit diagnostic.
        firstiter_cc = bool(firstiter_cc_mode["emulate"]) and bool(
            args.firstiter_cc_pass2_only_best_coarse
        ) and not bool(args.no_firstiter_cc_pass2_only_best_coarse)
        adaptive_em_kwargs = dict(common_em_kwargs)
        adaptive_em_kwargs["image_batch_size"] = fine_batch_plan.image_batch_size
        adaptive_em_kwargs["rotation_block_size"] = fine_batch_plan.rotation_block_size
        adaptive_em_kwargs["relion_fine_mstep_prune"] = bool(args.sparse_pass2)
        result = run_dense_k_class_em_adaptive(
            ds,
            means,
            mean_variance_prev,
            noise_variance,
            rotations.astype(np.float32),
            translations.astype(np.float32),
            fine_rotations,
            fine_translations,
            rot_parent_map,
            trans_parent_map,
            args.disc_type,
            adaptive_fraction=args.significance_adaptive_fraction,
            coarse_current_size=coarse_engine_current_size,
            fine_current_size=current_size,
            current_size=current_size,
            firstiter_cc_pass2_only_best_coarse=firstiter_cc,
            significance_image_batch_size=base_batch_plan.image_batch_size,
            significance_rotation_block_size=base_batch_plan.rotation_block_size,
            bpref_device_signature_active=bool(
                os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR")
            ),
            **adaptive_em_kwargs,
        )
    else:
        result = run_dense_k_class_em(
            ds,
            means,
            mean_variance_prev,
            noise_variance,
            rotations.astype(np.float32),
            translations.astype(np.float32),
            args.disc_type,
            current_size=current_size,
            **common_em_kwargs,
        )
    _sparse_pass2_diagnostics.clear_bpref_contribution_dump_context()
    elapsed_s = time.time() - t0
    print(f"  RECOVAR K-class E/M step completed in {elapsed_s:.1f}s")

    significant_summary = None
    significant_sample_indices = None
    significant_full_stats = None
    bpref_significant_sample_indices = None
    bpref_full_stats = None
    bpref_significant_summary = None
    need_significant_support = args.significant_mstep or args.relion_bpref_mstep
    if need_significant_support:
        sig_t0 = time.time()
        (
            _sig_rot_any,
            n_sig_all,
            _hard_assignment,
            _class_assignment,
            significant_sample_indices,
            significant_full_stats,
        ) = _compute_k_class_significance_batched(
            ds,
            means,
            noise_variance,
            rotations.astype(np.float32),
            translations.astype(np.float32),
            args.disc_type,
            class_log_priors=class_log_priors,
            adaptive_fraction=args.significance_adaptive_fraction,
            max_significants=-1,
            image_batch_size=significance_support_batch_plan.image_batch_size,
            rotation_block_size=significance_support_batch_plan.rotation_block_size,
            current_size=coarse_engine_current_size if args.adaptive_2pass else current_size,
            score_with_masked_images=True,
            rotation_log_prior=class_rotation_log_prior,
            translation_log_prior=translation_log_prior,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            half_spectrum_scoring=True,
            projection_padding_factor=args.projection_padding_factor,
            do_gridding_correction=True,
            square_window=False,
            use_float64_scoring=False,
            relion_projector_half=relion_projector_half_by_class,
            relion_projector_r_max=relion_projector_r_max,
        )
        normalization_log_z = significant_full_stats["normalization_log_z"]
        significant_mstep_applied = bool(args.significant_mstep and not args.adaptive_2pass)
        if args.significant_mstep and args.adaptive_2pass:
            print(
                "  significant support: adaptive replay keeps pass-2 accumulators; "
                "coarse support is used only for RELION rlnNrOfSignificantSamples diagnostics"
            )
        if significant_mstep_applied:
            sparse_Ft_y = []
            sparse_Ft_ctf = []
            for class_index in range(n_classes):
                class_Ft_y, class_Ft_ctf = compute_pass2_stats_sparse(
                    ds,
                    means[class_index],
                    mean_variance_prev[class_index],
                    noise_variance,
                    translations.astype(np.float32),
                    significant_sample_indices[class_index],
                    healpix_order,
                    args.disc_type,
                    oversampling_order=0,
                    current_size=current_size,
                    translation_step=offset_step_px,
                    rotation_log_prior=class_rotation_log_prior[class_index],
                    score_with_masked_images=True,
                    return_stats=False,
                    translation_log_prior=translation_log_prior,
                    accumulate_noise=False,
                    half_spectrum_scoring=True,
                    projection_padding_factor=args.projection_padding_factor,
                    reconstruction_padding_factor=args.reconstruction_padding_factor,
                    image_corrections=image_corrections,
                    scale_corrections=scale_corrections,
                    image_pre_shifts=image_pre_shifts,
                    use_float64_scoring=False,
                    do_gridding_correction=True,
                    square_window=False,
                    random_perturbation=random_perturbation,
                    normalization_log_z=normalization_log_z,
                    normalization_score_mode="gaussian",
                    relion_projector_half=relion_projector_half_by_class[class_index],
                    relion_projector_r_max=relion_projector_r_max,
                )[:2]
                sparse_Ft_y.append(class_Ft_y)
                sparse_Ft_ctf.append(class_Ft_ctf)

            result = result._replace(
                Ft_y=jnp.stack([jnp.asarray(value) for value in sparse_Ft_y], axis=0),
                Ft_ctf=jnp.stack([jnp.asarray(value) for value in sparse_Ft_ctf], axis=0),
            )
        relion_n_sig = np.asarray(target_data_ordered["rlnNrOfSignificantSamples"], dtype=np.float64)
        n_sig_diff = np.asarray(n_sig_all, dtype=np.float64) - relion_n_sig
        significant_summary = {
            "adaptive_fraction": float(args.significance_adaptive_fraction),
            "elapsed_s": float(time.time() - sig_t0),
            "recovar_mean": float(np.mean(n_sig_all)),
            "relion_mean": float(np.mean(relion_n_sig)),
            "abs_mean": float(np.mean(np.abs(n_sig_diff))),
            "abs_median": float(np.median(np.abs(n_sig_diff))),
            "abs_p95": float(np.percentile(np.abs(n_sig_diff), 95)),
            "abs_max": float(np.max(np.abs(n_sig_diff))),
            "recovar_min": int(np.min(n_sig_all)),
            "recovar_max": int(np.max(n_sig_all)),
            "relion_min": int(np.min(relion_n_sig)),
            "relion_max": int(np.max(relion_n_sig)),
        }
        print(
            "  significant support completed in "
            f"{significant_summary['elapsed_s']:.1f}s; "
            f"n_sig abs mean={significant_summary['abs_mean']:.3g}, "
            f"p95={significant_summary['abs_p95']:.3g}, max={significant_summary['abs_max']:.3g}"
        )
        bpref_significant_sample_indices = significant_sample_indices
        bpref_full_stats = significant_full_stats

    if args.relion_bpref_mstep and args.adaptive_2pass:
        # The adaptive significant-sample diagnostic above intentionally uses
        # RELION's coarse pass-1 current_size.  BPref rows are scored at the
        # reconstruction current_size, so reusing the coarse logZ here makes
        # exp(score - logZ) overflow and produces meaningless diagnostic maps.
        bpref_sig_t0 = time.time()
        print(
            "  RELION BPref diagnostic: recomputing same-window support "
            f"at current_size={current_size} (coarse support used current_size={coarse_engine_current_size})"
        )
        (
            _bpref_sig_rot_any,
            bpref_n_sig_all,
            _bpref_hard_assignment,
            _bpref_class_assignment,
            bpref_significant_sample_indices,
            bpref_full_stats,
        ) = _compute_k_class_significance_batched(
            ds,
            means,
            noise_variance,
            rotations.astype(np.float32),
            translations.astype(np.float32),
            args.disc_type,
            class_log_priors=class_log_priors,
            adaptive_fraction=args.significance_adaptive_fraction,
            max_significants=-1,
            image_batch_size=significance_support_batch_plan.image_batch_size,
            rotation_block_size=significance_support_batch_plan.rotation_block_size,
            current_size=current_size,
            score_with_masked_images=True,
            rotation_log_prior=class_rotation_log_prior,
            translation_log_prior=translation_log_prior,
            image_corrections=image_corrections,
            scale_corrections=scale_corrections,
            image_pre_shifts=image_pre_shifts,
            half_spectrum_scoring=True,
            projection_padding_factor=args.projection_padding_factor,
            do_gridding_correction=True,
            square_window=False,
            use_float64_scoring=False,
            relion_projector_half=relion_projector_half_by_class,
            relion_projector_r_max=relion_projector_r_max,
        )
        bpref_significant_summary = {
            "adaptive_fraction": float(args.significance_adaptive_fraction),
            "elapsed_s": float(time.time() - bpref_sig_t0),
            "current_size": int(current_size),
            "coarse_current_size": int(coarse_engine_current_size),
            "recovar_mean": float(np.mean(bpref_n_sig_all)),
            "recovar_min": int(np.min(bpref_n_sig_all)),
            "recovar_max": int(np.max(bpref_n_sig_all)),
        }
        print(
            "  RELION BPref diagnostic support completed in "
            f"{bpref_significant_summary['elapsed_s']:.1f}s; "
            f"n_sig mean={bpref_significant_summary['recovar_mean']:.3g}, "
            f"max={bpref_significant_summary['recovar_max']}"
        )

    relion_real = [helpers.load_relion_volume(str(target_prefix) + f"_class{k + 1:03d}.mrc") for k in range(n_classes)]
    solvent_mask = mask.raised_cosine_mask(
        ds.volume_shape,
        radius=particle_diameter / (2.0 * ds.voxel_size),
        radius_p=particle_diameter / (2.0 * ds.voxel_size) + 5.0,
        offset=jnp.zeros(3),
    )

    # RELION ml_optimiser.cpp:6389-6418: at iter 1 with --firstiter_cc,
    # RELION reapplies the ini_high low-pass filter only when ini_high > 0.
    # Class3D GUI-default commands often request --firstiter_cc without
    # --ini_high; those must not inherit a synthetic 30 A low-pass.
    apply_firstiter_lowpass = firstiter_lowpass_ini_high is not None

    def reconstruct_variant(
        tau_by_class, *, use_spherical_mask: bool, apply_solvent_mask: bool, grid_correct: bool, minres_map: int
    ):
        real_maps = []
        for class_index in range(n_classes):
            class_ft = _reconstruct_volume_eager(
                result.Ft_ctf[class_index],
                result.Ft_y[class_index],
                ds.volume_shape,
                args.reconstruction_padding_factor,
                tau=tau_by_class[class_index],
                tau2_fudge=tau2_fudge,
                projection_padding_factor=args.projection_padding_factor,
                use_spherical_mask=use_spherical_mask,
                grid_correct=grid_correct,
                minres_map=minres_map,
                # RELION's BackProjector::reconstruct skips voxels with
                # r2 >= max_r2 = ROUND(r_max * padding_factor)^2 (backprojector.cpp:1264).
                # Without current_size, recovar's Wiener filter operates on every
                # padded voxel up to upsampled_volume_shape[0]//2 - 1, producing
                # residual high-shell content from the regularization floor that
                # RELION omits. Passing current_size matches RELION's max_r2 skip.
                current_size=current_size,
            ).reshape(-1)
            if apply_firstiter_lowpass:
                from recovar.em.dense_single_volume.mean_helpers import _apply_relion_initial_lowpass_filter

                class_ft = _apply_relion_initial_lowpass_filter(
                    class_ft,
                    ds.volume_shape,
                    float(ds.voxel_size),
                    float(firstiter_lowpass_ini_high),
                    filter_edgewidth=2.0,
                )
            class_real = ftu.get_idft3(class_ft.reshape(ds.volume_shape)).real
            if apply_solvent_mask:
                class_real = class_real * solvent_mask
            real_maps.append(np.asarray(class_real))
        return real_maps

    variant_specs = [
        ("target_tau_sphere_solvent", mean_variance_target, True, True, True, RELION_MINRES_MAP),
        ("prev_tau_sphere_solvent", mean_variance_prev, True, True, True, RELION_MINRES_MAP),
        ("target_tau_no_sphere_solvent", mean_variance_target, False, True, True, RELION_MINRES_MAP),
        ("target_tau_sphere_no_solvent", mean_variance_target, True, False, True, RELION_MINRES_MAP),
        ("target_tau_sphere_solvent_no_minres", mean_variance_target, True, True, True, 0),
        ("target_tau_sphere_solvent_no_grid", mean_variance_target, True, True, False, RELION_MINRES_MAP),
    ]
    variant_results = {}
    variant_maps = {}
    for name, tau_by_class, use_spherical_mask, apply_solvent_mask, grid_correct, minres_map in variant_specs:
        maps = reconstruct_variant(
            tau_by_class,
            use_spherical_mask=use_spherical_mask,
            apply_solvent_mask=apply_solvent_mask,
            grid_correct=grid_correct,
            minres_map=minres_map,
        )
        variant_maps[name] = maps
        variant_results[name] = _best_class_permutation(maps, relion_real)

    relion_bpref_summary = None
    if args.relion_bpref_mstep:
        bpref_t0 = time.time()
        try:
            if bpref_significant_sample_indices is None or bpref_full_stats is None:
                raise RuntimeError("RELION BPref diagnostic requires significant support")
            bpref_maps, relion_bpref_class_summary = _relion_bpref_maps_from_sparse_support(
                ds,
                means,
                noise_variance,
                translations.astype(np.float32),
                bpref_significant_sample_indices,
                bpref_full_stats["normalization_log_z"],
                nside_level=healpix_order,
                disc_type=args.disc_type,
                current_size=current_size,
                translation_step=offset_step_px,
                class_rotation_log_prior=class_rotation_log_prior,
                translation_log_prior=translation_log_prior,
                score_with_masked_images=True,
                half_spectrum_scoring=True,
                projection_padding_factor=args.projection_padding_factor,
                reconstruction_padding_factor=args.reconstruction_padding_factor,
                image_corrections=image_corrections,
                scale_corrections=scale_corrections,
                image_pre_shifts=image_pre_shifts,
                use_float64_scoring=False,
                do_gridding_correction=True,
                square_window=False,
                random_perturbation=random_perturbation,
                tau2_spectra=[_tau_spectrum(target_model, class_index) for class_index in range(n_classes)],
                tau2_fudge=tau2_fudge,
                minres_map=RELION_MINRES_MAP,
                max_images_per_microbatch=args.relion_bpref_max_images_per_microbatch,
            )
            bpref_maps = [np.asarray(class_map) * np.asarray(solvent_mask) for class_map in bpref_maps]
            variant_maps["relion_bpref_sparse_solvent"] = bpref_maps
            variant_results["relion_bpref_sparse_solvent"] = _best_class_permutation(bpref_maps, relion_real)
            relion_bpref_summary = {
                "elapsed_s": float(time.time() - bpref_t0),
                "classes": relion_bpref_class_summary,
                "support": bpref_significant_summary,
                "max_images_per_microbatch": int(args.relion_bpref_max_images_per_microbatch),
            }
            print(
                "  RELION BPref diagnostic completed in "
                f"{relion_bpref_summary['elapsed_s']:.1f}s; "
                f"mean_corr={variant_results['relion_bpref_sparse_solvent']['mean_corr']:.6f}"
            )
        except Exception as exc:
            relion_bpref_summary = {
                "elapsed_s": float(time.time() - bpref_t0),
                "support": bpref_significant_summary,
                "max_images_per_microbatch": int(args.relion_bpref_max_images_per_microbatch),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  RELION BPref diagnostic failed: {relion_bpref_summary['error']}")

    default_variant = "target_tau_sphere_solvent"
    best_variant = max(variant_results, key=lambda key: variant_results[key]["mean_corr"])
    recovar_real = variant_maps[default_variant]
    for class_index, class_real in enumerate(recovar_real):
        write_relion_mrc(output_dir / f"recovar_class{class_index + 1:03d}.mrc", class_real, voxel_size=ds.voxel_size)
    for class_index, class_real in enumerate(variant_maps[best_variant]):
        write_relion_mrc(
            output_dir / f"recovar_best_variant_class{class_index + 1:03d}.mrc", class_real, voxel_size=ds.voxel_size
        )

    best_perm = variant_results[default_variant]
    perm = np.asarray(best_perm["recovar_to_relion"], dtype=np.int64)

    recovar_full_posterior_weights = np.asarray(result.class_posterior_sums, dtype=np.float64) / float(ds.n_images)
    recovar_mstep_class_mass = getattr(result, "class_mstep_posterior_sums", None)
    if recovar_mstep_class_mass is None:
        recovar_mstep_class_mass = result.class_posterior_sums
    recovar_weights = np.asarray(recovar_mstep_class_mass, dtype=np.float64) / float(ds.n_images)
    relion_weights = _class_distributions(target_model)
    target_class = np.asarray(target_data_ordered["rlnClassNumber"], dtype=np.int64)
    mapped_recovar_class = perm[np.asarray(result.class_assignments, dtype=np.int64)] + 1
    class_accuracy = float(np.mean(mapped_recovar_class == target_class))
    recovar_class_responsibilities = np.asarray(result.class_responsibilities, dtype=np.float64)
    recovar_class_by_responsibility = np.argmax(recovar_class_responsibilities, axis=0).astype(np.int64)
    mapped_recovar_class_by_responsibility = perm[recovar_class_by_responsibility] + 1
    class_accuracy_by_responsibility = float(np.mean(mapped_recovar_class_by_responsibility == target_class))
    assignment_disagreement = np.asarray(result.class_assignments, dtype=np.int64) != recovar_class_by_responsibility
    recovar_pmax = np.asarray(result.stats.max_posterior_per_image, dtype=np.float64)
    relion_pmax = np.asarray(target_data_ordered["rlnMaxValueProbDistribution"], dtype=np.float64)
    pmax_abs = np.abs(recovar_pmax - relion_pmax)

    summary = {
        "relion_dir": str(relion_dir),
        "data_star": str(args.data_star),
        "prev_iter": int(args.prev_iter),
        "target_iter": int(args.target_iter),
        "n_classes": int(n_classes),
        "n_images": int(ds.n_images),
        "current_size": int(current_size),
        "healpix_order": int(healpix_order),
        "n_rotations": int(rotations.shape[0]),
        "n_translations": int(translations.shape[0]),
        "random_perturbation": float(random_perturbation),
        "random_perturbation_star": float(star_random_perturbation),
        "random_perturbation_source": random_perturbation_source,
        "perturb_restart_state_iteration": args.perturb_restart_state_iteration,
        "elapsed_s": float(elapsed_s),
        "relion_optimiser_cli": relion_cli_flags,
        "firstiter_cc_mode": firstiter_cc_mode,
        "firstiter_cc_pass2_only_best_coarse": bool(
            firstiter_cc_mode["emulate"]
            and args.firstiter_cc_pass2_only_best_coarse
            and not args.no_firstiter_cc_pass2_only_best_coarse
        ),
        "firstiter_cc_ini_high_override_angstrom": (
            None
            if args.firstiter_cc_ini_high_angstrom is None
            else float(args.firstiter_cc_ini_high_angstrom)
        ),
        "firstiter_cc_ini_high_angstrom": (
            None if firstiter_lowpass_ini_high is None else float(firstiter_lowpass_ini_high)
        ),
        "firstiter_cc_lowpass_applied": bool(apply_firstiter_lowpass),
        "recovar_class_weights": recovar_weights,
        "recovar_full_posterior_class_weights": recovar_full_posterior_weights,
        "relion_class_weights": relion_weights,
        "relion_class_weights_in_recovar_order": relion_weights[perm],
        "class_weight_abs_diff_in_recovar_order": np.abs(recovar_weights - relion_weights[perm]),
        "class_assignment_accuracy_after_permutation": class_accuracy,
        "class_assignment_by_responsibility_accuracy_after_permutation": class_accuracy_by_responsibility,
        "class_assignment_best_vs_responsibility_disagreement_fraction": float(np.mean(assignment_disagreement)),
        "class_assignment_best_vs_responsibility_disagreement_count": int(np.count_nonzero(assignment_disagreement)),
        "best_permutation": best_perm,
        "reconstruction_variants": variant_results,
        "best_reconstruction_variant": best_variant,
        "significant_mstep": bool(args.significant_mstep),
        "significant_mstep_applied": bool(args.significant_mstep and not args.adaptive_2pass),
        "significant_samples": significant_summary,
        "relion_bpref_significant_samples": bpref_significant_summary,
        "adaptive_coarse_current_size": None if coarse_current_size is None else int(coarse_current_size),
        "relion_bpref_mstep": bool(args.relion_bpref_mstep),
        "relion_bpref_summary": relion_bpref_summary,
        "pmax": {
            "recovar_mean": float(recovar_pmax.mean()),
            "relion_mean": float(relion_pmax.mean()),
            "abs_mean": float(pmax_abs.mean()),
            "abs_median": float(np.median(pmax_abs)),
            "abs_p95": float(np.percentile(pmax_abs, 95)),
            "abs_max": float(pmax_abs.max()),
        },
        "output_maps": [str(output_dir / f"recovar_class{k + 1:03d}.mrc") for k in range(n_classes)],
    }

    parity_arrays = {
        "recovar_class_assignments": np.asarray(result.class_assignments, dtype=np.int32),
        "mapped_recovar_class": mapped_recovar_class,
        "recovar_class_assignments_by_responsibility": recovar_class_by_responsibility.astype(np.int32),
        "mapped_recovar_class_by_responsibility": mapped_recovar_class_by_responsibility,
        "recovar_class_responsibilities": recovar_class_responsibilities,
        "relion_class": target_class,
        "recovar_pmax": recovar_pmax,
        "relion_pmax": relion_pmax,
        "recovar_class_weights": recovar_weights,
        "relion_class_weights": relion_weights,
        "recovar_to_relion": perm,
    }
    if significant_summary is not None:
        parity_arrays["recovar_significant_samples"] = np.asarray(n_sig_all, dtype=np.int32)
        parity_arrays["relion_significant_samples"] = np.asarray(
            target_data_ordered["rlnNrOfSignificantSamples"],
            dtype=np.int32,
        )
    if bpref_significant_summary is not None:
        parity_arrays["recovar_bpref_significant_samples"] = np.asarray(bpref_n_sig_all, dtype=np.int32)
    np.savez(output_dir / "k_class_parity_arrays.npz", **parity_arrays)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(_jsonable(summary), f, indent=2, sort_keys=True)

    print("K-class parity summary:")
    print(f"  recovar weights: {recovar_weights.round(6).tolist()}")
    print(f"  relion weights in recovar order: {relion_weights[perm].round(6).tolist()}")
    print(f"  class assignment accuracy after permutation: {class_accuracy:.4f}")
    print(
        "  class assignment by responsibility accuracy after permutation: "
        f"{class_accuracy_by_responsibility:.4f} "
        f"(best-vs-responsibility disagreements={int(np.count_nonzero(assignment_disagreement))})"
    )
    print(
        "  Pmax abs diff: "
        f"mean={summary['pmax']['abs_mean']:.6g}, p95={summary['pmax']['abs_p95']:.6g}, max={summary['pmax']['abs_max']:.6g}"
    )
    print(
        f"  map correlations: {np.round(best_perm['map_correlations'], 6).tolist()} mean={best_perm['mean_corr']:.6f}"
    )
    print("  reconstruction variants:")
    for name, values in sorted(variant_results.items()):
        print(f"    {name}: mean={values['mean_corr']:.6f}, corrs={np.round(values['map_correlations'], 6).tolist()}")
    print(f"  best reconstruction variant: {best_variant}")
    print(f"  summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if (
            exc.__class__.__name__ == "Pass2DumpComplete"
            and os.environ.get("RECOVAR_K_CLASS_PARITY_STOP_AFTER_PASS2_DUMP") == "1"
        ):
            print(f"RECOVAR pass-2 dump completed; stopping replay before remaining M-step work: {exc}")
            sys.exit(0)
        raise
