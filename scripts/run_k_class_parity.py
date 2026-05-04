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
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def stack_index_from_image_name(name: str) -> int:
    match = re.match(r"(\d+)@", str(name))
    return int(match.group(1)) - 1 if match else -1


def _star_particles(star_data):
    return star_data["particles"] if isinstance(star_data, dict) and "particles" in star_data else star_data


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
    table = _class_table(model, class_index)
    if "rlnReferenceSigma2" in table.columns:
        column = "rlnReferenceSigma2"
    elif "rlnReferenceTau2" in table.columns:
        column = "rlnReferenceTau2"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-dir", required=True, type=Path)
    parser.add_argument("--data-star", required=True, type=Path)
    parser.add_argument("--prev-iter", type=int, default=0)
    parser.add_argument("--target-iter", type=int, default=1)
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
        help="Use RELION first-iteration winner-take-all reconstruction weights while keeping soft evidence stats.",
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
        "--significance-adaptive-fraction",
        type=float,
        default=0.999,
        help="Posterior mass retained when selecting significant class x pose samples.",
    )
    args = parser.parse_args()

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
    from recovar.em.dense_single_volume.k_class import run_dense_k_class_em
    from recovar.em.sampling import (
        apply_relion_rotation_perturbation_to_eulers,
        apply_relion_translation_perturbation,
        get_relion_rotation_grid_eulers,
        get_translation_grid,
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

    print(f"RELION K-class replay: K={n_classes}, N={grid_size}, prev={args.prev_iter}, target={args.target_iter}")
    print(f"  current_size={current_size}, pixel_size={pixel_size}, tau2_fudge={tau2_fudge}")
    print(f"  output_dir={output_dir}")
    print(f"  JAX devices: {jax.devices()}")

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
    means = jnp.stack(
        [
            jnp.asarray(
                ftu.get_dft3(jnp.asarray(helpers.load_relion_volume(str(prev_prefix) + f"_class{k + 1:03d}.mrc")))
            ).reshape(-1)
            for k in range(n_classes)
        ],
        axis=0,
    )

    sampling = read_relion_sampling_metadata(str(target_prefix) + "_sampling.star")
    healpix_order = int(sampling["healpix_order"])
    random_perturbation = float(sampling["random_perturbation"])
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
        f"rp={random_perturbation:+.5f}, offset_range_px={offset_range_px:.3f}, offset_step_px={offset_step_px:.3f}"
    )

    direction_prior = _read_class_direction_priors(prev_model, n_classes)
    class_rotation_log_prior = np.stack(
        [make_relion_direction_log_prior(direction_prior[k], healpix_order) for k in range(n_classes)],
        axis=0,
    )
    image_corrections, scale_corrections = _image_and_scale_corrections(prev_model, prev_data_ordered)
    previous_translations = _previous_translations_pixels(prev_data_ordered, pixel_size)
    image_pre_shifts = relion_translation_search_base(previous_translations)
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
        f"sigma_offset={sigma_offset_angstrom:.6f}A, "
        f"image_corr_mean={float(image_corrections.mean()):.6f}, scale_mean={float(scale_corrections.mean()):.6f}"
    )

    t0 = time.time()
    result = run_dense_k_class_em(
        ds,
        means,
        mean_variance_prev,
        noise_variance,
        rotations.astype(np.float32),
        translations.astype(np.float32),
        args.disc_type,
        class_log_priors=np.zeros(n_classes, dtype=np.float32),
        accumulate_noise=False,
        image_batch_size=args.image_batch_size,
        rotation_block_size=args.rotation_block_size,
        current_size=current_size,
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
        use_float64_projections=False,
        do_gridding_correction=True,
        square_window=True,
        sparse_pass2=False,
        relion_firstiter_winner_take_all=args.winner_take_all_mstep,
        # RELION fixture used --firstiter_cc (run_it000_optimiser.star command);
        # match CC scoring at iter 1 when winner-take-all is on, per
        # ml_optimiser.cpp:8758-8774 (do_firstiter_cc branch in
        # getAllSquaredDifferences).
        relion_firstiter_score_mode=("normalized_cc" if args.winner_take_all_mstep else "gaussian"),
    )
    elapsed_s = time.time() - t0
    print(f"  RECOVAR K-class E/M step completed in {elapsed_s:.1f}s")

    significant_summary = None
    significant_sample_indices = None
    significant_full_stats = None
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
            class_log_priors=np.zeros(n_classes, dtype=np.float32),
            adaptive_fraction=args.significance_adaptive_fraction,
            max_significants=-1,
            image_batch_size=args.image_batch_size,
            rotation_block_size=args.rotation_block_size,
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
        )
        normalization_log_z = significant_full_stats["normalization_log_z"]
        if args.significant_mstep:
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

    relion_real = [helpers.load_relion_volume(str(target_prefix) + f"_class{k + 1:03d}.mrc") for k in range(n_classes)]
    solvent_mask = mask.raised_cosine_mask(
        ds.volume_shape,
        radius=particle_diameter / (2.0 * ds.voxel_size),
        radius_p=particle_diameter / (2.0 * ds.voxel_size) + 5.0,
        offset=jnp.zeros(3),
    )

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
        if significant_sample_indices is None or significant_full_stats is None:
            raise RuntimeError("RELION BPref diagnostic requires significant support")
        bpref_t0 = time.time()
        bpref_maps, relion_bpref_summary = _relion_bpref_maps_from_sparse_support(
            ds,
            means,
            noise_variance,
            translations.astype(np.float32),
            significant_sample_indices,
            significant_full_stats["normalization_log_z"],
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
        )
        bpref_maps = [np.asarray(class_map) * np.asarray(solvent_mask) for class_map in bpref_maps]
        variant_maps["relion_bpref_sparse_solvent"] = bpref_maps
        variant_results["relion_bpref_sparse_solvent"] = _best_class_permutation(bpref_maps, relion_real)
        relion_bpref_summary = {
            "elapsed_s": float(time.time() - bpref_t0),
            "classes": relion_bpref_summary,
        }
        print(
            "  RELION BPref diagnostic completed in "
            f"{relion_bpref_summary['elapsed_s']:.1f}s; "
            f"mean_corr={variant_results['relion_bpref_sparse_solvent']['mean_corr']:.6f}"
        )

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

    recovar_weights = np.asarray(result.class_posterior_sums, dtype=np.float64) / float(ds.n_images)
    relion_weights = _class_distributions(target_model)
    target_class = np.asarray(target_data_ordered["rlnClassNumber"], dtype=np.int64)
    mapped_recovar_class = perm[np.asarray(result.class_assignments, dtype=np.int64)] + 1
    class_accuracy = float(np.mean(mapped_recovar_class == target_class))
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
        "elapsed_s": float(elapsed_s),
        "recovar_class_weights": recovar_weights,
        "relion_class_weights": relion_weights,
        "relion_class_weights_in_recovar_order": relion_weights[perm],
        "class_weight_abs_diff_in_recovar_order": np.abs(recovar_weights - relion_weights[perm]),
        "class_assignment_accuracy_after_permutation": class_accuracy,
        "best_permutation": best_perm,
        "reconstruction_variants": variant_results,
        "best_reconstruction_variant": best_variant,
        "significant_mstep": bool(args.significant_mstep),
        "significant_samples": significant_summary,
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

    np.savez(
        output_dir / "k_class_parity_arrays.npz",
        recovar_class_assignments=np.asarray(result.class_assignments, dtype=np.int32),
        mapped_recovar_class=mapped_recovar_class,
        relion_class=target_class,
        recovar_pmax=recovar_pmax,
        relion_pmax=relion_pmax,
        recovar_class_weights=recovar_weights,
        relion_class_weights=relion_weights,
        recovar_to_relion=perm,
    )
    with (output_dir / "summary.json").open("w") as f:
        json.dump(_jsonable(summary), f, indent=2, sort_keys=True)

    print("K-class parity summary:")
    print(f"  recovar weights: {recovar_weights.round(6).tolist()}")
    print(f"  relion weights in recovar order: {relion_weights[perm].round(6).tolist()}")
    print(f"  class assignment accuracy after permutation: {class_accuracy:.4f}")
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
    main()
