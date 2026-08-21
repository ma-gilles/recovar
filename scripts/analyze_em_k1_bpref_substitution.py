#!/usr/bin/env python3
"""Substitute RELION's raw BPref accumulator into a K=1 Wiener replay.

Both replay arms use the same RELION tau2.  The baseline arm reconstructs
from RECOVAR's post-join numerator/weight, while the substitution arm uses
RELION's passively captured post-join raw BPref numerator/weight.  This
isolates the accumulator boundary from the already-rejected tau2 boundary.

The raw-layout conversion must first reproduce RELION's independently dumped
downsampled average to near machine precision.  Map quality uses FSC/FSC-AUC;
normalized L2 and amplitude fits are secondary.  Correlation is not computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.analyze_em_k1_tau2_substitution import (
        _general,
        _model,
        _reconstruct_and_flatten,
        _relion_tau2,
        map_metrics,
    )
    from scripts.compare_iter1_bpref_accum import (
        _apply_recovar_frame,
        downsample_recovar_accumulator,
        load_relion_dump,
    )
else:
    from analyze_em_k1_tau2_substitution import (
        _general,
        _model,
        _reconstruct_and_flatten,
        _relion_tau2,
        map_metrics,
    )
    from compare_iter1_bpref_accum import (
        _apply_recovar_frame,
        downsample_recovar_accumulator,
        load_relion_dump,
    )

OUTPUT_SCHEMA = "recovar.em_k1_bpref_substitution.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_relion_raw(path: Path, *, value_dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Load the passive RELION BPref binary and its eight-int64 header."""

    path = path.resolve()
    _require(path.is_file(), f"missing RELION raw BPref file: {path}")
    dtype = np.dtype(value_dtype)
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype=np.int64, count=8)
        values = np.fromfile(stream, dtype=dtype)
    _require(header.size == 8, f"truncated RELION raw header: {path}")
    rank, call, xsize, ysize, zsize, xinit, yinit, zinit = (
        int(value) for value in header
    )
    _require(rank in {1, 2}, f"unexpected RELION rank {rank}")
    _require(call >= 0, f"unexpected RELION call {call}")
    _require(
        xsize == (zsize + 1) // 2 and ysize == zsize and zsize % 2 == 1,
        f"unsupported RELION raw BPref shape {(zsize, ysize, xsize)}",
    )
    radius = zsize // 2
    _require(
        (xinit, yinit, zinit) == (0, -radius, -radius),
        f"unsupported RELION raw BPref starts {(xinit, yinit, zinit)}",
    )
    expected_count = zsize * ysize * xsize
    _require(
        values.size == expected_count,
        f"RELION raw BPref payload has {values.size} values, expected {expected_count}",
    )
    return header, values.reshape((zsize, ysize, xsize))


def relion_raw_to_recovar_full(
    raw_data: np.ndarray,
    raw_weight: np.ndarray,
    *,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert RELION ``[k,i,j>=0]`` BPref storage to RECOVAR's full cube.

    RECOVAR's centered BPref axes are ``[j,i,k]`` relative to RELION's
    Xmipp coordinates.  The positive ``j`` half is transposed directly and
    the negative half is completed by Hermitian symmetry.  Native numeric
    units differ by ``N^2`` for data and ``N^4`` for weight.
    """

    data = np.asarray(raw_data)
    weight = np.asarray(raw_weight)
    _require(
        data.ndim == weight.ndim == 3 and data.shape == weight.shape,
        "RELION raw data and weight shapes differ",
    )
    zsize, ysize, xsize = data.shape
    _require(
        zsize == ysize and zsize % 2 == 1 and xsize == (zsize + 1) // 2,
        f"unsupported RELION raw shape {data.shape}",
    )
    _require(grid_size > 0, "grid size must be positive")
    center = zsize // 2
    full_data = np.zeros((zsize, zsize, zsize), dtype=np.complex128)
    full_weight = np.zeros((zsize, zsize, zsize), dtype=np.float64)
    n2 = float(grid_size) ** 2
    n4 = float(grid_size) ** 4
    full_data[center:, :, :] = np.transpose(data, (2, 1, 0)) / n2
    full_weight[center:, :, :] = np.transpose(weight, (2, 1, 0)) / n4

    negative_indices = np.arange(center, dtype=np.int64)
    positive_partners = -(negative_indices - center) + center
    centered_partners = zsize - 1 - np.arange(zsize, dtype=np.int64)
    full_data[negative_indices, :, :] = np.conj(
        full_data[
            positive_partners[:, None, None],
            centered_partners[None, :, None],
            centered_partners[None, None, :],
        ]
    )
    full_weight[negative_indices, :, :] = full_weight[
        positive_partners[:, None, None],
        centered_partners[None, :, None],
        centered_partners[None, None, :],
    ]
    return full_data.reshape(-1), full_weight.reshape(-1)


def verify_downsampled_replay(
    numerator: np.ndarray,
    weight: np.ndarray,
    *,
    relion_dump: dict[str, Any],
    grid_size: int,
    volume_shape: tuple[int, int, int],
    accumulator_shape: tuple[int, int, int],
    padding_factor: int,
    max_shell: int,
) -> dict[str, float]:
    """Require the converted raw cube to reproduce RELION's text dump."""

    avg, down_weight, down_radius, _, _, actual_max_shell = (
        downsample_recovar_accumulator(
            numerator,
            weight,
            volume_shape,
            padding_factor,
            max_shell,
            accumulator_shape=accumulator_shape,
        )
    )
    _require(actual_max_shell == max_shell, "converted raw max-shell mismatch")
    avg, down_weight = _apply_recovar_frame(
        avg,
        down_weight,
        grid_size=grid_size,
        recovar_frame="relion",
    )
    k = np.asarray(relion_dump["k"], dtype=np.int64)
    i = np.asarray(relion_dump["i"], dtype=np.int64)
    j = np.asarray(relion_dump["j"], dtype=np.int64)
    replay_avg = avg[i + down_radius, k + down_radius, j]
    replay_weight = down_weight[i + down_radius, k + down_radius, j]
    target_avg = np.asarray(relion_dump["real"]) + 1j * np.asarray(
        relion_dump["imag"]
    )
    target_weight = np.asarray(relion_dump["weight"])
    avg_denominator = float(np.linalg.norm(target_avg))
    weight_denominator = float(np.linalg.norm(target_weight))
    _require(
        avg_denominator > 0.0 and weight_denominator > 0.0,
        "RELION downsampled dump has zero energy",
    )
    return {
        "average_relative_l2": float(
            np.linalg.norm(replay_avg - target_avg) / avg_denominator
        ),
        "weight_relative_l2": float(
            np.linalg.norm(replay_weight - target_weight) / weight_denominator
        ),
        "average_max_absolute_error": float(
            np.max(np.abs(replay_avg - target_avg))
        ),
        "weight_max_absolute_error": float(
            np.max(np.abs(replay_weight - target_weight))
        ),
    }


def align_discrete_map_sign(
    value: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Choose the exact ±1 Fourier-frame sign nearest the target map."""

    value = np.asarray(value)
    target = np.asarray(target)
    _require(value.shape == target.shape, "map-sign alignment shapes differ")
    positive_error = float(np.linalg.norm(value - target))
    negative_error = float(np.linalg.norm(-value - target))
    sign = -1 if negative_error < positive_error else 1
    return sign * value, sign


def classify_accumulator_substitution(
    baseline_relative_l2: float,
    substituted_relative_l2: float,
    *,
    explanation_fraction_gate: float,
) -> dict[str, Any]:
    _require(baseline_relative_l2 > 0.0, "baseline relative L2 must be positive")
    _require(
        substituted_relative_l2 >= 0.0,
        "substituted relative L2 must be nonnegative",
    )
    explained_fraction = float(
        (baseline_relative_l2 - substituted_relative_l2)
        / baseline_relative_l2
    )
    classification = (
        "relion_bpref_accumulator_explains_majority_of_map_residual"
        if explained_fraction >= explanation_fraction_gate
        else "relion_bpref_accumulator_does_not_explain_map_residual"
    )
    return {
        "classification": classification,
        "relative_l2_explained_fraction": explained_fraction,
        "explanation_fraction_gate": float(explanation_fraction_gate),
    }


def analyze(
    *,
    recovar_accumulator: Path,
    recovar_results: Path,
    relion_dir: Path,
    relion_dump_dir: Path,
    relion_call: int,
    reference_iteration: int,
    run_id: str,
    particle_diameter_angstrom: float,
    projection_padding_factor: int,
    minres_map: int,
    raw_replay_relative_l2_gate: float,
    map_replay_fsc_auc_gate: float,
    map_replay_relative_l2_gate: float,
    explanation_fraction_gate: float,
) -> dict[str, Any]:
    from recovar.utils.helpers import load_relion_volume

    recovar_accumulator = recovar_accumulator.resolve()
    recovar_results = recovar_results.resolve()
    relion_dir = relion_dir.resolve()
    relion_dump_dir = relion_dump_dir.resolve()
    _require(recovar_accumulator.is_file(), f"missing {recovar_accumulator}")
    _require(recovar_results.is_file(), f"missing {recovar_results}")
    _require(relion_dir.is_dir(), f"missing {relion_dir}")
    _require(relion_dump_dir.is_dir(), f"missing {relion_dump_dir}")
    _require(reference_iteration >= 1, "reference iteration must be positive")

    with np.load(recovar_accumulator, allow_pickle=False) as archive:
        _require(
            str(np.asarray(archive["schema"]).item()) == "recovar-bpref-accum-v2",
            "unexpected RECOVAR accumulator schema",
        )
        _require(
            str(np.asarray(archive["run_id"]).item()) == run_id,
            "RECOVAR accumulator run_id mismatch",
        )
        _require(
            int(np.asarray(archive["iteration"]).item()) == reference_iteration,
            "RECOVAR accumulator iteration mismatch",
        )
        current_size = int(np.asarray(archive["current_size"]).item())
        grid_size = int(np.asarray(archive["grid_size"]).item())
        padding_factor = int(np.asarray(archive["padding_factor"]).item())
        voxel_size = float(np.asarray(archive["voxel_size"]).item())
        volume_shape = tuple(int(value) for value in archive["volume_shape"])
        accumulator_shape = tuple(
            int(value) for value in archive["mstep_accumulator_shape"]
        )
        recovar_numerator = [
            np.asarray(archive[f"Ft_y_{half}"]) for half in (0, 1)
        ]
        recovar_weight = [
            np.asarray(archive[f"Ft_ctf_{half}"]) for half in (0, 1)
        ]

    with np.load(recovar_results, allow_pickle=False) as results:
        _require(
            tuple(int(value) for value in results["volume_shape"]) == volume_shape,
            "RECOVAR results volume shape differs from accumulator",
        )
    _require(
        accumulator_shape == (accumulator_shape[0],) * 3,
        f"RECOVAR accumulator is not cubic: {accumulator_shape}",
    )
    max_shell = current_size // 2
    n4 = float(grid_size) ** 4
    half_rows = []
    artifact_paths = [recovar_accumulator, recovar_results]
    for half in (1, 2):
        raw_data_path = (
            relion_dump_dir
            / f"raw_bpref_data_rank{half:02d}_call{relion_call:04d}.bin"
        )
        raw_weight_path = (
            relion_dump_dir
            / f"raw_bpref_weight_rank{half:02d}_call{relion_call:04d}.bin"
        )
        text_dump_path = (
            relion_dump_dir
            / f"downsampled_avg_rank{half:02d}_call{relion_call:04d}.txt"
        )
        model_path = (
            relion_dir
            / f"run_it{reference_iteration:03d}_half{half}_model.star"
        )
        map_path = (
            relion_dir
            / f"run_it{reference_iteration:03d}_half{half}_class001.mrc"
        )
        data_header, raw_data = load_relion_raw(
            raw_data_path,
            value_dtype=np.complex128,
        )
        weight_header, raw_weight = load_relion_raw(
            raw_weight_path,
            value_dtype=np.float64,
        )
        _require(
            np.array_equal(data_header, weight_header),
            f"RELION raw headers differ for half {half}",
        )
        _require(
            int(data_header[0]) == half and int(data_header[1]) == relion_call,
            f"RELION raw identity mismatch for half {half}",
        )
        _require(
            tuple(int(value) for value in data_header[[4, 3, 4]])
            == accumulator_shape,
            "RELION and RECOVAR accumulator shapes differ",
        )
        relion_numerator, relion_weight = relion_raw_to_recovar_full(
            raw_data,
            raw_weight,
            grid_size=grid_size,
        )
        relion_dump = load_relion_dump(text_dump_path)
        raw_replay = verify_downsampled_replay(
            relion_numerator,
            relion_weight,
            relion_dump=relion_dump,
            grid_size=grid_size,
            volume_shape=volume_shape,
            accumulator_shape=accumulator_shape,
            padding_factor=padding_factor,
            max_shell=max_shell,
        )
        _require(
            raw_replay["average_relative_l2"] <= raw_replay_relative_l2_gate,
            f"RELION raw average replay failed for half {half}",
        )
        _require(
            raw_replay["weight_relative_l2"] <= raw_replay_relative_l2_gate,
            f"RELION raw weight replay failed for half {half}",
        )

        model = _model(model_path)
        general = _general(model)
        _require(
            int(general["rlnCurrentImageSize"]) == current_size,
            "RELION and RECOVAR current sizes differ",
        )
        _require(
            int(general["rlnPaddingFactor"]) == padding_factor,
            "RELION and RECOVAR padding factors differ",
        )
        _require(
            abs(float(general["rlnPixelSize"]) - voxel_size) <= 1.0e-6,
            "RELION and RECOVAR voxel sizes differ",
        )
        relion_tau = _relion_tau2(model, n4)
        target_map = np.asarray(
            load_relion_volume(str(map_path)),
            dtype=np.float64,
        )
        baseline_map = _reconstruct_and_flatten(
            recovar_weight[half - 1],
            recovar_numerator[half - 1],
            relion_tau,
            volume_shape=volume_shape,
            accumulator_shape=accumulator_shape,
            padding_factor=padding_factor,
            projection_padding_factor=projection_padding_factor,
            current_size=current_size,
            minres_map=minres_map,
            voxel_size=voxel_size,
            particle_diameter_angstrom=particle_diameter_angstrom,
        )
        substituted_map = _reconstruct_and_flatten(
            relion_weight,
            relion_numerator,
            relion_tau,
            volume_shape=volume_shape,
            accumulator_shape=accumulator_shape,
            padding_factor=padding_factor,
            projection_padding_factor=projection_padding_factor,
            current_size=current_size,
            minres_map=minres_map,
            voxel_size=voxel_size,
            particle_diameter_angstrom=particle_diameter_angstrom,
        )
        substituted_map, relion_raw_map_sign = align_discrete_map_sign(
            substituted_map,
            target_map,
        )
        baseline = map_metrics(baseline_map, target_map)
        substituted = map_metrics(substituted_map, target_map)
        _require(
            substituted["fsc_auc"] >= map_replay_fsc_auc_gate,
            f"RELION raw map replay FSC-AUC failed for half {half}",
        )
        _require(
            substituted["relative_l2"] <= map_replay_relative_l2_gate,
            f"RELION raw map replay relative L2 failed for half {half}",
        )
        classification = classify_accumulator_substitution(
            baseline["relative_l2"],
            substituted["relative_l2"],
            explanation_fraction_gate=explanation_fraction_gate,
        )
        half_rows.append(
            {
                "half": half,
                "relion_raw_map_sign_applied": relion_raw_map_sign,
                "raw_downsampled_replay": raw_replay,
                "recovar_accumulator_relion_tau_vs_relion": baseline,
                "relion_accumulator_relion_tau_vs_relion": substituted,
                **classification,
            }
        )
        artifact_paths.extend(
            [
                raw_data_path,
                raw_weight_path,
                text_dump_path,
                model_path,
                map_path,
            ]
        )

    all_explain = all(
        row["classification"]
        == "relion_bpref_accumulator_explains_majority_of_map_residual"
        for row in half_rows
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "parity_score": False,
        "metric_policy": (
            "FSC/FSC-AUC primary; normalized L2 and shellwise amplitude "
            "secondary; correlation is not computed"
        ),
        "isolation": (
            "RECOVAR and RELION accumulator arms use identical RELION tau2; "
            "only post-join numerator/weight are substituted"
        ),
        "reference_iteration": int(reference_iteration),
        "relion_call": int(relion_call),
        "run_id": run_id,
        "current_size": current_size,
        "grid_size": grid_size,
        "volume_shape": list(volume_shape),
        "accumulator_shape": list(accumulator_shape),
        "gates": {
            "maximum_raw_downsampled_replay_relative_l2": float(
                raw_replay_relative_l2_gate
            ),
            "minimum_relion_raw_map_replay_fsc_auc": float(
                map_replay_fsc_auc_gate
            ),
            "maximum_relion_raw_map_replay_relative_l2": float(
                map_replay_relative_l2_gate
            ),
            "minimum_map_residual_explanation_fraction": float(
                explanation_fraction_gate
            ),
        },
        "classification": (
            "relion_bpref_accumulator_explains_majority_of_map_residual"
            if all_explain
            else "relion_bpref_accumulator_not_sufficient_for_both_halves"
        ),
        "halves": half_rows,
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in artifact_paths
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-accumulator", required=True, type=Path)
    parser.add_argument("--recovar-results", required=True, type=Path)
    parser.add_argument("--relion-dir", required=True, type=Path)
    parser.add_argument("--relion-dump-dir", required=True, type=Path)
    parser.add_argument("--relion-call", type=int, default=1)
    parser.add_argument("--reference-iteration", type=int, default=2)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--particle-diameter-angstrom", type=float, required=True)
    parser.add_argument("--projection-padding-factor", type=int, default=2)
    parser.add_argument("--minres-map", type=int, default=5)
    parser.add_argument(
        "--raw-replay-relative-l2-gate",
        type=float,
        default=1.0e-12,
    )
    parser.add_argument("--map-replay-fsc-auc-gate", type=float, default=0.99999)
    parser.add_argument(
        "--map-replay-relative-l2-gate",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--explanation-fraction-gate",
        type=float,
        default=0.5,
    )
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = analyze(
        recovar_accumulator=args.recovar_accumulator,
        recovar_results=args.recovar_results,
        relion_dir=args.relion_dir,
        relion_dump_dir=args.relion_dump_dir,
        relion_call=args.relion_call,
        reference_iteration=args.reference_iteration,
        run_id=args.run_id,
        particle_diameter_angstrom=args.particle_diameter_angstrom,
        projection_padding_factor=args.projection_padding_factor,
        minres_map=args.minres_map,
        raw_replay_relative_l2_gate=args.raw_replay_relative_l2_gate,
        map_replay_fsc_auc_gate=args.map_replay_fsc_auc_gate,
        map_replay_relative_l2_gate=args.map_replay_relative_l2_gate,
        explanation_fraction_gate=args.explanation_fraction_gate,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
