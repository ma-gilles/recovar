#!/usr/bin/env python3
"""Replace selected K=1 iteration-1 winners and replay both numbered maps.

The input contribution shards contain exact per-particle BPref operands for a
small fixed panel.  For every selected source row, this script removes the
RECOVAR winner scatter, inserts the matrix/translation-matched RELION winner,
propagates the summed numerator and denominator deltas through the low-
resolution half join, and repeats the iteration-1 FSC/tau2/map solve.

Map acceptance uses signed shellwise FSC/FSC-AUC.  Correlation is not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

from scripts.analyze_em_k1_tau2_substitution import map_metrics
from scripts.analyze_k1_single_translation_map_counterfactual import (
    _infer_cubic_shape,
    _norm_summary,
    _reconstruct_iteration1_maps,
    _require,
)


SCHEMA = "recovar.em.k1_pose_winner_map_counterfactual.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_distances_deg(matrices: np.ndarray, target: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64).reshape(-1, 3, 3)
    target = np.asarray(target, dtype=np.float64).reshape(3, 3)
    relative = np.einsum("nij,jk->nik", np.transpose(matrices, (0, 2, 1)), target)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    skew = np.stack(
        (
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ),
        axis=1,
    )
    sine = 0.5 * np.linalg.norm(skew, axis=1)
    return np.rad2deg(np.arctan2(sine, cosine))


def _match_target_candidate(
    *,
    active_particle_rows: np.ndarray,
    active_rotation_rows: np.ndarray,
    active_rotations: np.ndarray,
    particle_row: int,
    target_rotation: np.ndarray,
    fine_translations: np.ndarray,
    integer_pre_shift: np.ndarray,
    target_translation_pixels: np.ndarray,
) -> dict[str, Any]:
    selected = np.flatnonzero(np.asarray(active_particle_rows) == int(particle_row))
    _require(selected.size > 0, f"particle row {particle_row} has no captured rotations")
    distances = _rotation_distances_deg(
        np.asarray(active_rotations)[selected], target_rotation
    )
    closest_local = int(np.argmin(distances))
    active_index = int(selected[closest_local])
    rotation_row = int(np.asarray(active_rotation_rows)[active_index])

    target_fine_translation = (
        np.asarray(target_translation_pixels, dtype=np.float64)
        - np.asarray(integer_pre_shift, dtype=np.float64)
    )
    translation_distances = np.linalg.norm(
        np.asarray(fine_translations, dtype=np.float64) - target_fine_translation[None, :],
        axis=1,
    )
    translation_index = int(np.argmin(translation_distances))
    return {
        "rotation_row": rotation_row,
        "rotation": np.asarray(active_rotations)[active_index],
        "rotation_error_deg": float(distances[closest_local]),
        "translation_index": translation_index,
        "translation_error_pixels": float(translation_distances[translation_index]),
        "target_fine_translation_pixels": target_fine_translation,
    }


def _backproject_pair(
    numerator_row: np.ndarray,
    denominator_row: np.ndarray,
    rotation: np.ndarray,
    fftw_indices: np.ndarray,
    *,
    image_shape: tuple[int, int],
    accumulator_shape: tuple[int, int, int],
    current_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject
    from recovar.core import fourier_transform_utils as ftu
    from recovar.em.dense_single_volume.local_backprojection import (
        enforce_relion_half_volume_x0_hermitian_host,
    )

    half_shape = ftu.volume_shape_to_half_volume_shape(accumulator_shape)
    numerator0 = jnp.zeros((int(np.prod(half_shape)),), dtype=jnp.complex64)
    denominator0 = jnp.zeros((int(np.prod(half_shape)),), dtype=jnp.float32)
    numerator, denominator = cuda_backproject.relion_fused_x_half_backproject_indexed(
        numerator0,
        denominator0,
        jnp.asarray(np.asarray(numerator_row, dtype=np.complex64)[None]),
        jnp.asarray(np.asarray(denominator_row, dtype=np.float32)[None]),
        jnp.asarray(fftw_indices, dtype=jnp.int32),
        jnp.asarray(np.asarray(rotation, dtype=np.float32)[None]),
        tuple(int(value) for value in image_shape),
        tuple(int(value) for value in accumulator_shape),
        float(current_size // 2),
    )
    numerator = enforce_relion_half_volume_x0_hermitian_host(
        jax.device_get(numerator), accumulator_shape
    )
    denominator = enforce_relion_half_volume_x0_hermitian_host(
        jax.device_get(denominator), accumulator_shape
    )
    return (
        np.asarray(numerator, dtype=np.complex64),
        np.asarray(denominator, dtype=np.float32),
    )


def _load_shards(contribution_dir: Path) -> list[tuple[Path, dict[str, np.ndarray]]]:
    paths = sorted(contribution_dir.glob("bpref_contribution_rows_it001_h*_*.npz"))
    _require(bool(paths), f"no iteration-1 contribution shards in {contribution_dir}")
    loaded = []
    for path in paths:
        with np.load(path, allow_pickle=False) as dump:
            values = {key: np.asarray(dump[key]) for key in dump.files}
        _require(
            str(values["schema"].item()) == "recovar-bpref-contribution-rows-v3",
            f"unsupported contribution schema in {path}",
        )
        _require(bool(values["high_precision_operand_bundle"].item()), f"{path} lacks operands")
        loaded.append((path.resolve(), values))
    return loaded


def analyze(
    *,
    contribution_dir: Path,
    pose_comparison: Path,
    intermediates: Path,
    native_relion_dir: Path,
    output_dir: Path,
    source_rows: tuple[int, ...],
    particle_diameter_angstrom: float,
    low_resolution_join_angstrom: float,
    projection_padding_factor: int,
    minres_map: int,
    rotation_match_gate_deg: float,
    translation_match_gate_pixels: float,
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

    contribution_dir = contribution_dir.resolve()
    pose_comparison = pose_comparison.resolve()
    intermediates = intermediates.resolve()
    native_relion_dir = native_relion_dir.resolve()
    output_dir = output_dir.resolve()
    _require(pose_comparison.is_file(), f"missing pose comparison: {pose_comparison}")
    _require(intermediates.is_dir(), f"missing intermediates: {intermediates}")
    _require(native_relion_dir.is_dir(), f"missing RELION directory: {native_relion_dir}")
    shards = _load_shards(contribution_dir)

    with np.load(pose_comparison, allow_pickle=False) as poses:
        relion_eulers = np.asarray(poses["relion_eulers"], dtype=np.float64)
        relion_translations = np.asarray(poses["relion_translations"], dtype=np.float64)
        recovar_eulers = np.asarray(poses["recovar_eulers"], dtype=np.float64)
        recovar_translations = np.asarray(poses["recovar_translations"], dtype=np.float64)

    target_set = set(int(value) for value in source_rows)
    _require(len(target_set) == len(source_rows), "source rows must be unique")
    _require(target_set, "at least one source row is required")
    _require(max(target_set) < relion_eulers.shape[0], "source row lies outside pose comparison")
    target_rotations = utils.R_from_relion(relion_eulers[list(source_rows)])
    target_rotation_by_source = dict(zip(source_rows, target_rotations, strict=True))

    first = shards[0][1]
    score_current_size = int(first["current_size"].item())
    _require("mstep_current_size" in first, "capture lacks exact M-step current size")
    current_size = int(first["mstep_current_size"].item())
    _require(current_size > 0, "capture has no finite M-step support radius")
    image_shape = tuple(int(value) for value in first["image_shape"])
    captured_accumulator_shape = tuple(int(value) for value in first["volume_shape"])
    voxel_size = float(first["voxel_size"].item())
    reconstruction_padding_factor = int(first["reconstruction_padding_factor"].item())
    fftw_indices = np.asarray(first["window_indices"], dtype=np.int32)
    volume_shape = (image_shape[0],) * 3
    accumulator_shape = relion_backprojector_volume_shape(
        volume_shape, reconstruction_padding_factor, current_size=current_size
    )
    _require(accumulator_shape == captured_accumulator_shape, "accumulator shape mismatch")

    half_prejoin_y = [np.zeros(int(np.prod(accumulator_shape)), np.complex64) for _ in range(2)]
    half_prejoin_w = [np.zeros(int(np.prod(accumulator_shape)), np.float32) for _ in range(2)]
    particle_reports: list[dict[str, Any]] = []
    seen: set[int] = set()

    for shard_path, dump in shards:
        _require(
            int(dump["current_size"].item()) == score_current_size,
            "mixed scoring current sizes",
        )
        _require(
            int(dump["mstep_current_size"].item()) == current_size,
            "mixed M-step current sizes",
        )
        _require(tuple(int(v) for v in dump["image_shape"]) == image_shape, "mixed image shapes")
        _require(np.array_equal(dump["window_indices"], fftw_indices), "mixed window indices")
        half = int(dump["half"].item())
        _require(half in {1, 2}, f"invalid half {half}")
        original_indices = np.asarray(dump["original_indices"], dtype=np.int64)
        selected_rows = np.flatnonzero(np.isin(original_indices, list(target_set)))
        for particle_row_np in selected_rows:
            particle_row = int(particle_row_np)
            source_row = int(original_indices[particle_row])
            _require(source_row not in seen, f"source row {source_row} captured more than once")
            reconstruction_probs = np.asarray(dump["reconstruction_probs"], dtype=np.float32)
            nonzero = np.argwhere(reconstruction_probs[particle_row] != 0.0)
            _require(
                nonzero.shape == (1, 2),
                f"source row {source_row} does not have one winner tuple",
            )
            current_rotation_row, current_translation_index = (
                int(value) for value in nonzero[0]
            )
            weight = float(reconstruction_probs[particle_row, current_rotation_row, current_translation_index])
            _require(weight > 0.0, f"source row {source_row} winner weight is not positive")

            active_particle_rows = np.asarray(dump["active_particle_rows"], dtype=np.int64)
            active_rotation_rows = np.asarray(dump["active_rotation_rows"], dtype=np.int64)
            active_rotations = np.asarray(dump["active_rotations"], dtype=np.float32)
            current_active = np.flatnonzero(
                (active_particle_rows == particle_row)
                & (active_rotation_rows == current_rotation_row)
            )
            _require(current_active.size == 1, f"source row {source_row} current rotation is ambiguous")
            current_rotation = active_rotations[int(current_active[0])]
            target = _match_target_candidate(
                active_particle_rows=active_particle_rows,
                active_rotation_rows=active_rotation_rows,
                active_rotations=active_rotations,
                particle_row=particle_row,
                target_rotation=target_rotation_by_source[source_row],
                fine_translations=np.asarray(dump["fine_translations"], dtype=np.float32),
                integer_pre_shift=np.asarray(dump["integer_pre_shifts"])[particle_row],
                target_translation_pixels=relion_translations[source_row],
            )
            _require(
                target["translation_error_pixels"] <= translation_match_gate_pixels,
                f"source row {source_row} target translation misses grid by "
                f"{target['translation_error_pixels']} px",
            )
            target_rotation_row = int(target["rotation_row"])
            target_translation_index = int(target["translation_index"])
            candidate_mask = np.asarray(dump["candidate_mask"], dtype=bool)
            target_is_grid_rotation = bool(
                target["rotation_error_deg"] <= rotation_match_gate_deg
            )
            target_candidate_active = bool(
                target_is_grid_rotation
                and candidate_mask[
                    particle_row, target_rotation_row, target_translation_index
                ]
            )
            shifted = np.asarray(dump["mstep_shifted_recon"], dtype=np.complex64)
            ctf2 = np.asarray(dump["mstep_ctf2_over_nv"], dtype=np.float32)
            _require(ctf2.ndim == 2, f"unexpected CTF operand shape {ctf2.shape}")
            current_y, current_w = _backproject_pair(
                shifted[particle_row, current_translation_index] * np.float32(weight),
                ctf2[particle_row] * np.float32(weight),
                current_rotation,
                fftw_indices,
                image_shape=image_shape,
                accumulator_shape=accumulator_shape,
                current_size=current_size,
            )
            target_y, target_w = _backproject_pair(
                shifted[particle_row, target_translation_index] * np.float32(weight),
                ctf2[particle_row] * np.float32(weight),
                # Scatter at the actual RELION pose.  The diagnostic also
                # reports whether that pose was present in RECOVAR's fine
                # candidate grid; a different coarse parent must not prevent
                # this map-level causal intervention.
                target_rotation_by_source[source_row],
                fftw_indices,
                image_shape=image_shape,
                accumulator_shape=accumulator_shape,
                current_size=current_size,
            )
            current_y_full = np.asarray(
                jax.device_get(relion_x_half_volume_to_full(current_y, accumulator_shape)),
                dtype=np.complex64,
            )
            target_y_full = np.asarray(
                jax.device_get(relion_x_half_volume_to_full(target_y, accumulator_shape)),
                dtype=np.complex64,
            )
            current_w_full = np.asarray(
                jax.device_get(relion_x_half_volume_to_full(current_w, accumulator_shape)),
                dtype=np.float32,
            )
            target_w_full = np.asarray(
                jax.device_get(relion_x_half_volume_to_full(target_w, accumulator_shape)),
                dtype=np.float32,
            )
            y_delta = np.asarray(target_y_full - current_y_full, dtype=np.complex64)
            w_delta = np.asarray(target_w_full - current_w_full, dtype=np.float32)
            half_prejoin_y[half - 1] += y_delta
            half_prejoin_w[half - 1] += w_delta
            scores = np.asarray(dump["candidate_combined_scores"], dtype=np.float64)
            target_score = (
                float(scores[particle_row, target_rotation_row, target_translation_index])
                if target_candidate_active
                else None
            )
            current_score = float(
                scores[particle_row, current_rotation_row, current_translation_index]
            )
            particle_reports.append(
                {
                    "source_row_zero_based": source_row,
                    "half": half,
                    "shard": str(shard_path),
                    "current_rotation_row": current_rotation_row,
                    "current_translation_index": current_translation_index,
                    "target_rotation_row": target_rotation_row,
                    "target_translation_index": target_translation_index,
                    "target_rotation_error_deg": target["rotation_error_deg"],
                    "target_rotation_in_recovar_grid": target_is_grid_rotation,
                    "target_candidate_active": target_candidate_active,
                    "target_translation_error_pixels": target["translation_error_pixels"],
                    "recovar_euler_deg": recovar_eulers[source_row].tolist(),
                    "relion_euler_deg": relion_eulers[source_row].tolist(),
                    "recovar_translation_pixels": recovar_translations[source_row].tolist(),
                    "relion_translation_pixels": relion_translations[source_row].tolist(),
                    "current_score": current_score,
                    "target_score": target_score,
                    "target_minus_current_score": (
                        None if target_score is None else target_score - current_score
                    ),
                    "prejoin_numerator_delta": _norm_summary(y_delta),
                    "prejoin_denominator_delta": _norm_summary(w_delta),
                }
            )
            seen.add(source_row)

    _require(seen == target_set, f"capture covered {sorted(seen)}, expected {sorted(target_set)}")
    joined_deltas = regularization.join_halves_at_low_resolution(
        half_prejoin_y[0],
        half_prejoin_y[1],
        half_prejoin_w[0],
        half_prejoin_w[1],
        accumulator_shape,
        voxel_size,
        image_shape[0],
        low_resolution_join_angstrom,
        current_resolution_angstrom=None,
        padding_factor=reconstruction_padding_factor,
    )
    joined_y = [np.asarray(jax.device_get(joined_deltas[i]), np.complex64) for i in range(2)]
    joined_w = [np.asarray(jax.device_get(joined_deltas[i + 2]), np.float32) for i in range(2)]

    meta = np.load(intermediates / "it000_meta.npy", allow_pickle=True).item()
    _require(int(meta["current_size"]) == current_size, "saved and captured current sizes differ")
    numerators: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    saved_map_paths: list[Path] = []
    native_map_paths: list[Path] = []
    for half_index in range(2):
        half = half_index + 1
        numerator_path = intermediates / f"it000_Ft_y_{half_index}.npy"
        weight_path = intermediates / f"it000_Ft_ctf_{half_index}.npy"
        saved_map_path = intermediates / f"it000_half{half}_reg.mrc"
        native_map_path = native_relion_dir / f"run_it001_half{half}_class001.mrc"
        for path in (numerator_path, weight_path, saved_map_path, native_map_path):
            _require(path.is_file(), f"missing required artifact: {path}")
        numerator = np.asarray(np.load(numerator_path))
        weight_values = np.asarray(np.load(weight_path))
        _require(_infer_cubic_shape(numerator.size) == accumulator_shape, "saved shape mismatch")
        _require(numerator.shape == weight_values.shape, "numerator/weight shape mismatch")
        numerators.append(numerator)
        weights.append(weight_values)
        saved_map_paths.append(saved_map_path)
        native_map_paths.append(native_map_path)

    counterfactual_numerators = [
        numerators[i] + joined_y[i].astype(numerators[i].dtype, copy=False) for i in range(2)
    ]
    counterfactual_weights = [
        weights[i] + joined_w[i].astype(weights[i].dtype, copy=False) for i in range(2)
    ]
    baseline_maps, baseline_fsc, _ = _reconstruct_iteration1_maps(
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
    counterfactual_maps, counterfactual_fsc, _ = _reconstruct_iteration1_maps(
        counterfactual_numerators,
        counterfactual_weights,
        volume_shape=volume_shape,
        accumulator_shape=accumulator_shape,
        current_size=current_size,
        voxel_size=voxel_size,
        particle_diameter_angstrom=particle_diameter_angstrom,
        padding_factor=reconstruction_padding_factor,
        projection_padding_factor=projection_padding_factor,
        minres_map=minres_map,
    )
    stored_fsc = np.asarray(np.load(intermediates / "it000_fsc.npy"), dtype=np.float64)
    _require(stored_fsc.shape == baseline_fsc.shape, "stored/replayed FSC shapes differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    half_reports = []
    for half_index in range(2):
        half = half_index + 1
        saved_map = np.asarray(helpers.load_mrc(str(saved_map_paths[half_index])), dtype=np.float64)
        native_map = np.asarray(
            helpers.load_relion_volume(str(native_map_paths[half_index])), dtype=np.float64
        )
        replay = map_metrics(baseline_maps[half_index], saved_map)
        _require(replay["fsc_auc"] >= replay_fsc_auc_gate, f"half {half} replay FSC failed")
        _require(
            replay["relative_l2"] <= replay_relative_l2_gate,
            f"half {half} replay relative-L2 failed",
        )
        delta_adjusted = saved_map + (counterfactual_maps[half_index] - baseline_maps[half_index])
        output_map = output_dir / f"counterfactual_half{half}.mrc"
        delta_adjusted_map = output_dir / f"delta_adjusted_saved_half{half}.mrc"
        utils.write_mrc(str(output_map), np.asarray(counterfactual_maps[half_index], np.float32), voxel_size=voxel_size)
        utils.write_mrc(str(delta_adjusted_map), np.asarray(delta_adjusted, np.float32), voxel_size=voxel_size)
        half_reports.append(
            {
                "half": half,
                "baseline_replay_vs_saved_recovar": replay,
                "saved_recovar_vs_native_relion": map_metrics(saved_map, native_map),
                "counterfactual_vs_native_relion": map_metrics(counterfactual_maps[half_index], native_map),
                "delta_adjusted_saved_vs_native_relion": map_metrics(delta_adjusted, native_map),
                "counterfactual_vs_baseline": map_metrics(counterfactual_maps[half_index], baseline_maps[half_index]),
                "joined_numerator_delta": _norm_summary(joined_y[half_index]),
                "joined_denominator_delta": _norm_summary(joined_w[half_index]),
                "counterfactual_map": str(output_map),
                "delta_adjusted_saved_map": str(delta_adjusted_map),
            }
        )

    return {
        "schema": SCHEMA,
        "status": "complete",
        "metric_policy": "signed shellwise FSC/FSC-AUC primary; correlation is not computed",
        "source_rows_zero_based": list(source_rows),
        "score_current_size": score_current_size,
        "current_size": current_size,
        "volume_shape": list(volume_shape),
        "accumulator_shape": list(accumulator_shape),
        "particle_interventions": sorted(particle_reports, key=lambda value: value["source_row_zero_based"]),
        "fsc_replay_max_abs_vs_stored": float(np.max(np.abs(baseline_fsc - stored_fsc))),
        "fsc_counterfactual_max_abs_vs_baseline": float(np.max(np.abs(counterfactual_fsc - baseline_fsc))),
        "halves": half_reports,
        "inputs": {
            "pose_comparison": str(pose_comparison),
            "pose_comparison_sha256": _sha256(pose_comparison),
            "contribution_shards": {str(path): _sha256(path) for path, _ in shards},
            "intermediates": str(intermediates),
            "native_relion_dir": str(native_relion_dir),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contribution-dir", required=True, type=Path)
    parser.add_argument("--pose-comparison", required=True, type=Path)
    parser.add_argument("--intermediates", required=True, type=Path)
    parser.add_argument("--native-relion-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-rows", required=True)
    parser.add_argument("--particle-diameter-angstrom", type=float, default=200.0)
    parser.add_argument("--low-resolution-join-angstrom", type=float, default=40.0)
    parser.add_argument("--projection-padding-factor", type=int, default=2)
    parser.add_argument("--minres-map", type=int, default=5)
    parser.add_argument("--rotation-match-gate-deg", type=float, default=1e-3)
    parser.add_argument("--translation-match-gate-pixels", type=float, default=1e-3)
    parser.add_argument("--replay-fsc-auc-gate", type=float, default=0.99999)
    parser.add_argument("--replay-relative-l2-gate", type=float, default=0.001)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_rows = tuple(int(value) for value in args.source_rows.split(",") if value.strip())
    report = analyze(
        contribution_dir=args.contribution_dir,
        pose_comparison=args.pose_comparison,
        intermediates=args.intermediates,
        native_relion_dir=args.native_relion_dir,
        output_dir=args.output_dir,
        source_rows=source_rows,
        particle_diameter_angstrom=args.particle_diameter_angstrom,
        low_resolution_join_angstrom=args.low_resolution_join_angstrom,
        projection_padding_factor=args.projection_padding_factor,
        minres_map=args.minres_map,
        rotation_match_gate_deg=args.rotation_match_gate_deg,
        translation_match_gate_pixels=args.translation_match_gate_pixels,
        replay_fsc_auc_gate=args.replay_fsc_auc_gate,
        replay_relative_l2_gate=args.replay_relative_l2_gate,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
