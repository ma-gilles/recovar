#!/usr/bin/env python3
"""Replace one K=1 iteration-1 translation contribution and replay both maps.

This is a narrow causal test.  It backprojects the current and target
translation operands from one captured particle, propagates their difference
through RELION's x-half Hermitian enforcement and low-resolution half join,
adds only that difference to saved post-join accumulators, and repeats the
numbered-map Wiener solve with the stored RECOVAR tau2.

Map acceptance uses signed shellwise FSC/FSC-AUC.  Correlation is not used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# The backprojection intervention requires RECOVAR's CUDA FFI.  Set this
# before importing the CPU-oriented FSC reporter used by map_metrics.
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

from scripts.analyze_em_k1_tau2_substitution import map_metrics


SCHEMA = "recovar.em.k1_single_translation_map_counterfactual.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _infer_cubic_shape(size: int) -> tuple[int, int, int]:
    edge = int(round(int(size) ** (1.0 / 3.0)))
    _require(edge**3 == int(size), f"array size {size} is not cubic")
    return (edge, edge, edge)


def _norm_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "dtype": str(array.dtype),
        "size": int(array.size),
        "nonzero": int(np.count_nonzero(array)),
        "l2": float(np.linalg.norm(array.reshape(-1).astype(np.complex128))),
        "max_abs": float(np.max(np.abs(array))),
    }


def _backproject_row(
    row: np.ndarray,
    rotation: np.ndarray,
    fftw_indices: np.ndarray,
    *,
    image_shape: tuple[int, int],
    accumulator_shape: tuple[int, int, int],
    current_size: int,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject
    from recovar.core import fourier_transform_utils as ftu
    from recovar.em.dense_single_volume.local_backprojection import (
        enforce_relion_half_volume_x0_hermitian_host,
    )

    half_shape = ftu.volume_shape_to_half_volume_shape(accumulator_shape)
    zeros = jnp.zeros((int(np.prod(half_shape)),), dtype=jnp.complex64)
    packed = cuda_backproject.backproject_indexed(
        zeros,
        jnp.asarray(np.asarray(row, dtype=np.complex64)[None]),
        jnp.asarray(fftw_indices, dtype=jnp.int32),
        jnp.asarray(np.asarray(rotation, dtype=np.float32)[None]),
        image_shape=image_shape,
        volume_shape=accumulator_shape,
        order=1,
        half_volume=True,
        half_image=True,
        max_r=float(current_size // 2),
        relion_x_half=True,
    )
    packed = enforce_relion_half_volume_x0_hermitian_host(
        jax.device_get(packed),
        accumulator_shape,
    )
    return np.asarray(packed, dtype=np.complex64)


def _reconstruct_iteration1_maps(
    numerators: list[np.ndarray],
    weights: list[np.ndarray],
    *,
    volume_shape: tuple[int, int, int],
    accumulator_shape: tuple[int, int, int],
    current_size: int,
    voxel_size: float,
    particle_diameter_angstrom: float,
    padding_factor: int,
    projection_padding_factor: int,
    minres_map: int,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Replay the exact fresh firstiter-CC FSC, tau2, solve, filter, and mask."""

    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.em.dense_single_volume.mean_helpers import (
        _reconstruct_and_postprocess_means,
    )
    from recovar.reconstruction import regularization

    fsc = regularization.compute_relion_fsc_from_backprojector(
        numerators[0],
        numerators[1],
        weights[0],
        weights[1],
        volume_shape,
        padding_factor=padding_factor,
        r_max=current_size // 2,
        accumulator_volume_shape=accumulator_shape,
    )
    tau_per_half = []
    for weight in weights:
        tau, _, _ = regularization.compute_relion_tau2_from_weights(
            weight,
            weight,
            fsc,
            volume_shape,
            tau2_fudge=1.0,
            padding_factor=padding_factor,
            r_max=current_size // 2,
            return_details=True,
            full_half_axis=-1,
            accumulator_volume_shape=accumulator_shape,
        )
        tau_per_half.append(tau)

    means: list[Any] = [None, None]
    _reconstruct_and_postprocess_means(
        means,
        Ft_y_0=numerators[0],
        Ft_y_1=numerators[1],
        Ft_ctf_0=weights[0],
        Ft_ctf_1=weights[1],
        Ft_y_combined=None,
        Ft_ctf_combined=None,
        mean_signal_variance=None,
        mean_signal_variance_shells=None,
        mean_signal_variance_per_half=tau_per_half,
        n_classes=1,
        k_class_enabled=False,
        cs=current_size,
        iteration=0,
        grid_size=volume_shape[0],
        cryo=SimpleNamespace(voxel_size=voxel_size),
        volume_shape=volume_shape,
        tau2_fudge=1.0,
        padding_factor=padding_factor,
        projection_padding_factor=projection_padding_factor,
        relion_minres_map=minres_map,
        particle_diameter_ang=particle_diameter_angstrom,
        relion_firstiter_cc_this_iter=True,
        relion_firstiter_ini_high_angstrom=30.0,
        relion_width_mask_edge=5,
        relion_fmask_edge=2,
        accumulator_volume_shape=accumulator_shape,
    )
    real_maps = [
        np.asarray(
            jax.device_get(ftu.get_idft3(jnp.asarray(value).reshape(volume_shape)))
        ).real.astype(np.float64, copy=False)
        for value in means
    ]
    return real_maps, np.asarray(jax.device_get(fsc), dtype=np.float64), [
        np.asarray(jax.device_get(value)) for value in tau_per_half
    ]


def analyze(
    *,
    contribution: Path,
    intermediates: Path,
    native_relion_dir: Path,
    output_dir: Path,
    target_translation_index: int,
    particle_diameter_angstrom: float,
    low_resolution_join_angstrom: float,
    projection_padding_factor: int,
    minres_map: int,
    replay_fsc_auc_gate: float,
    replay_relative_l2_gate: float,
) -> dict[str, Any]:
    import jax

    from recovar import utils
    from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
        relion_backprojector_volume_shape,
        relion_x_half_volume_to_full,
    )
    from recovar.reconstruction import regularization
    from recovar.utils import helpers

    contribution = contribution.resolve()
    intermediates = intermediates.resolve()
    native_relion_dir = native_relion_dir.resolve()
    output_dir = output_dir.resolve()
    _require(contribution.is_file(), f"missing contribution capture: {contribution}")
    _require(intermediates.is_dir(), f"missing intermediates: {intermediates}")
    _require(native_relion_dir.is_dir(), f"missing RELION directory: {native_relion_dir}")

    with np.load(contribution, allow_pickle=False) as dump:
        _require(
            str(np.asarray(dump["schema"]).item())
            == "recovar-bpref-contribution-rows-v3",
            "unsupported contribution schema",
        )
        _require(int(np.asarray(dump["iteration"]).item()) == 1, "capture is not iteration 1")
        _require(int(np.asarray(dump["half"]).item()) == 1, "capture is not half 1")
        current_size = int(np.asarray(dump["current_size"]).item())
        image_shape = tuple(int(value) for value in np.asarray(dump["image_shape"]))
        captured_accumulator_shape = tuple(
            int(value) for value in np.asarray(dump["volume_shape"])
        )
        voxel_size = float(np.asarray(dump["voxel_size"]).item())
        reconstruction_padding_factor = int(
            np.asarray(dump["reconstruction_padding_factor"]).item()
        )
        centered_indices = np.asarray(dump["window_indices"], dtype=np.int32)
        reconstruction_probs = np.asarray(dump["reconstruction_probs"], dtype=np.float32)
        shifted = np.asarray(dump["mstep_shifted_recon"], dtype=np.complex64)
        active_rotation_rows = np.asarray(dump["active_rotation_rows"], dtype=np.int64)
        active_rotations = np.asarray(dump["active_rotations"], dtype=np.float32)
        combined_scores = np.asarray(dump["candidate_combined_scores"], dtype=np.float64)
        original_indices = np.asarray(dump["original_indices"], dtype=np.int64)

    _require(reconstruction_probs.shape[0] == 1, "capture must contain exactly one particle")
    nonzero = np.argwhere(reconstruction_probs[0] != 0.0)
    _require(nonzero.shape == (1, 2), "current reconstruction support must be one class-pose tuple")
    current_rotation_row, current_translation_index = (int(value) for value in nonzero[0])
    _require(
        0 <= target_translation_index < shifted.shape[1],
        "target translation index is outside the captured grid",
    )
    matches = np.flatnonzero(active_rotation_rows == current_rotation_row)
    _require(matches.size == 1, "could not identify the active oversampled rotation")
    rotation = active_rotations[int(matches[0])]
    current_row = shifted[0, current_translation_index]
    target_row = shifted[0, target_translation_index]
    _require(
        np.any(current_row != target_row),
        "current and target translation operands are identical",
    )

    # Schema-v3 contribution capture stores the exact index vector consumed by
    # the x-half CUDA M-step.  It is already in packed FFTW-half coordinates;
    # converting it a second time moves every operand outside the intended
    # current-size support.
    fftw_indices = np.asarray(centered_indices, dtype=np.int32)
    volume_shape = (image_shape[0],) * 3
    accumulator_shape = relion_backprojector_volume_shape(
        volume_shape,
        reconstruction_padding_factor,
        current_size=current_size,
    )
    _require(
        accumulator_shape == captured_accumulator_shape,
        "captured and inferred RELION accumulator shapes differ",
    )
    current_packed = _backproject_row(
        current_row,
        rotation,
        fftw_indices,
        image_shape=image_shape,
        accumulator_shape=accumulator_shape,
        current_size=current_size,
    )
    target_packed = _backproject_row(
        target_row,
        rotation,
        fftw_indices,
        image_shape=image_shape,
        accumulator_shape=accumulator_shape,
        current_size=current_size,
    )
    current_full = np.asarray(
        jax.device_get(relion_x_half_volume_to_full(current_packed, accumulator_shape)),
        dtype=np.complex64,
    )
    target_full = np.asarray(
        jax.device_get(relion_x_half_volume_to_full(target_packed, accumulator_shape)),
        dtype=np.complex64,
    )
    prejoin_delta = np.asarray(target_full - current_full, dtype=np.complex64)
    zeros_y = np.zeros_like(prejoin_delta)
    zeros_w = np.zeros(prejoin_delta.shape, dtype=np.float32)
    delta_half1, delta_half2, _, _ = regularization.join_halves_at_low_resolution(
        prejoin_delta,
        zeros_y,
        zeros_w,
        zeros_w,
        accumulator_shape,
        voxel_size,
        image_shape[0],
        low_resolution_join_angstrom,
        current_resolution_angstrom=None,
        padding_factor=reconstruction_padding_factor,
    )
    delta_halves = [
        np.asarray(jax.device_get(delta_half1), dtype=np.complex64),
        np.asarray(jax.device_get(delta_half2), dtype=np.complex64),
    ]

    meta_path = intermediates / "it000_meta.npy"
    _require(meta_path.is_file(), "missing iteration-1 metadata")
    meta = np.load(meta_path, allow_pickle=True).item()
    _require(int(meta["current_size"]) == current_size, "saved and captured current sizes differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    numerators = []
    weights = []
    saved_map_paths = []
    native_map_paths = []
    for half_index in range(2):
        half = half_index + 1
        numerator_path = intermediates / f"it000_Ft_y_{half_index}.npy"
        weight_path = intermediates / f"it000_Ft_ctf_{half_index}.npy"
        saved_map_path = intermediates / f"it000_half{half}_reg.mrc"
        native_map_path = native_relion_dir / f"run_it001_half{half}_class001.mrc"
        for path in (numerator_path, weight_path, saved_map_path, native_map_path):
            _require(path.is_file(), f"missing required artifact: {path}")
        numerator = np.asarray(np.load(numerator_path))
        weight = np.asarray(np.load(weight_path))
        _require(_infer_cubic_shape(numerator.size) == accumulator_shape, "accumulator shape mismatch")
        _require(numerator.shape == weight.shape, "numerator/weight shape mismatch")
        numerators.append(numerator)
        weights.append(weight)
        saved_map_paths.append(saved_map_path)
        native_map_paths.append(native_map_path)

    counterfactual_numerators = [
        np.asarray(
            numerators[half_index]
            + delta_halves[half_index].astype(numerators[half_index].dtype, copy=False),
            dtype=numerators[half_index].dtype,
        )
        for half_index in range(2)
    ]
    baseline_maps, baseline_fsc, baseline_tau_per_half = _reconstruct_iteration1_maps(
        numerators,
        weights,
        volume_shape=volume_shape,
        accumulator_shape=accumulator_shape,
        current_size=current_size,
        voxel_size=voxel_size,
        particle_diameter_angstrom=particle_diameter_angstrom,
        padding_factor=reconstruction_padding_factor,
        projection_padding_factor=projection_padding_factor,
        minres_map=minres_map,
    )
    counterfactual_maps_values, counterfactual_fsc, counterfactual_tau_per_half = (
        _reconstruct_iteration1_maps(
            counterfactual_numerators,
            weights,
            volume_shape=volume_shape,
            accumulator_shape=accumulator_shape,
            current_size=current_size,
            voxel_size=voxel_size,
            particle_diameter_angstrom=particle_diameter_angstrom,
            padding_factor=reconstruction_padding_factor,
            projection_padding_factor=projection_padding_factor,
            minres_map=minres_map,
        )
    )
    stored_fsc = np.asarray(np.load(intermediates / "it000_fsc.npy"), dtype=np.float64)
    _require(stored_fsc.shape == baseline_fsc.shape, "stored/replayed FSC shapes differ")

    half_reports = []
    counterfactual_map_paths = []
    for half_index in range(2):
        half = half_index + 1
        baseline_map = baseline_maps[half_index]
        counterfactual_map = counterfactual_maps_values[half_index]
        saved_map = np.asarray(helpers.load_mrc(str(saved_map_paths[half_index])), dtype=np.float64)
        native_map = np.asarray(helpers.load_relion_volume(str(native_map_paths[half_index])), dtype=np.float64)
        # The replay carries a small common reconstruction error relative to
        # the map saved by the original run.  Apply only the causal replay
        # delta to that saved map so the counterfactual comparison does not
        # attribute the common replay offset to this one-particle change.
        delta_adjusted_saved_map = saved_map + (counterfactual_map - baseline_map)
        replay = map_metrics(baseline_map, saved_map)
        _require(
            replay["fsc_auc"] >= replay_fsc_auc_gate,
            f"baseline replay FSC-AUC failed for half {half}: {replay['fsc_auc']}",
        )
        _require(
            replay["relative_l2"] <= replay_relative_l2_gate,
            f"baseline replay relative-L2 failed for half {half}: {replay['relative_l2']}",
        )
        output_map_path = output_dir / f"counterfactual_half{half}.mrc"
        utils.write_mrc(
            str(output_map_path),
            np.asarray(counterfactual_map, dtype=np.float32),
            voxel_size=voxel_size,
        )
        delta_adjusted_map_path = output_dir / f"delta_adjusted_saved_half{half}.mrc"
        utils.write_mrc(
            str(delta_adjusted_map_path),
            np.asarray(delta_adjusted_saved_map, dtype=np.float32),
            voxel_size=voxel_size,
        )
        counterfactual_map_paths.append(str(output_map_path))
        half_reports.append(
            {
                "half": half,
                "baseline_replay_vs_saved_recovar": replay,
                "baseline_vs_native_relion": map_metrics(baseline_map, native_map),
                "counterfactual_vs_native_relion": map_metrics(counterfactual_map, native_map),
                "counterfactual_vs_baseline": map_metrics(counterfactual_map, baseline_map),
                "saved_recovar_vs_native_relion": map_metrics(saved_map, native_map),
                "delta_adjusted_saved_vs_native_relion": map_metrics(
                    delta_adjusted_saved_map, native_map
                ),
                "joined_numerator_delta": _norm_summary(delta_halves[half_index]),
                "baseline_tau2": _norm_summary(baseline_tau_per_half[half_index]),
                "counterfactual_tau2": _norm_summary(counterfactual_tau_per_half[half_index]),
                "counterfactual_map": str(output_map_path),
                "delta_adjusted_saved_map": str(delta_adjusted_map_path),
            }
        )

    return {
        "schema": SCHEMA,
        "status": "complete",
        "metric_policy": "signed shellwise FSC/FSC-AUC primary; correlation is not computed",
        "contribution": str(contribution),
        "intermediates": str(intermediates),
        "native_relion_dir": str(native_relion_dir),
        "source_row_zero_based": int(original_indices[0]),
        "current_size": current_size,
        "volume_shape": list(volume_shape),
        "accumulator_shape": list(accumulator_shape),
        "window_index_convention": "packed_fftw_half_exact_mstep_operand",
        "voxel_size": voxel_size,
        "current_rotation_row": current_rotation_row,
        "current_translation_index": current_translation_index,
        "target_translation_index": int(target_translation_index),
        "current_score": float(combined_scores[0, current_rotation_row, current_translation_index]),
        "target_score": float(combined_scores[0, current_rotation_row, target_translation_index]),
        "score_delta_target_minus_current": float(
            combined_scores[0, current_rotation_row, target_translation_index]
            - combined_scores[0, current_rotation_row, current_translation_index]
        ),
        "prejoin_numerator_delta": _norm_summary(prejoin_delta),
        "fsc_replay_max_abs_vs_stored": float(np.max(np.abs(baseline_fsc - stored_fsc))),
        "fsc_counterfactual_max_abs_vs_baseline": float(
            np.max(np.abs(counterfactual_fsc - baseline_fsc))
        ),
        "counterfactual_maps": counterfactual_map_paths,
        "halves": half_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contribution", type=Path, required=True)
    parser.add_argument("--intermediates", type=Path, required=True)
    parser.add_argument("--native-relion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-translation-index", type=int, required=True)
    parser.add_argument("--particle-diameter-angstrom", type=float, default=200.0)
    parser.add_argument("--low-resolution-join-angstrom", type=float, default=40.0)
    parser.add_argument("--projection-padding-factor", type=int, default=2)
    parser.add_argument("--minres-map", type=int, default=5)
    parser.add_argument("--replay-fsc-auc-gate", type=float, default=0.99999)
    parser.add_argument("--replay-relative-l2-gate", type=float, default=0.001)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = analyze(
        contribution=args.contribution,
        intermediates=args.intermediates,
        native_relion_dir=args.native_relion_dir,
        output_dir=args.output_dir,
        target_translation_index=args.target_translation_index,
        particle_diameter_angstrom=args.particle_diameter_angstrom,
        low_resolution_join_angstrom=args.low_resolution_join_angstrom,
        projection_padding_factor=args.projection_padding_factor,
        minres_map=args.minres_map,
        replay_fsc_auc_gate=args.replay_fsc_auc_gate,
        replay_relative_l2_gate=args.replay_relative_l2_gate,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
