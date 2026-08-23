#!/usr/bin/env python3
"""Audit the K=1 post-join BPref boundary against a RELION repeat.

The audit pairs a versioned RECOVAR ``recovar_bpref_accum_itNNN.npz`` dump
with RELION's passive ``downsampled_avg`` and ``downsampled_fsc`` diagnostics.
It also checks that the diagnostic RELION repeat reproduces the qualified
RELION maps before interpreting any accumulator difference.

FSC/FSC-AUC are the primary quality metrics.  Complex normalized L2,
least-squares amplitude, and weight/support comparisons are secondary
boundary diagnostics.  Correlation is deliberately not computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.analyze_em_k1_tau2_substitution import map_metrics
    from scripts.compare_iter1_bpref_accum import (
        _apply_recovar_frame,
        downsample_recovar_accumulator,
        load_relion_dump,
    )
    from scripts.summarize_em_completion_bench import normalized_fsc_auc
else:
    from analyze_em_k1_tau2_substitution import map_metrics
    from compare_iter1_bpref_accum import (
        _apply_recovar_frame,
        downsample_recovar_accumulator,
        load_relion_dump,
    )
    from summarize_em_completion_bench import normalized_fsc_auc

OUTPUT_SCHEMA = "recovar.em_k1_bpref_boundary.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_l2(value: np.ndarray, target: np.ndarray) -> float:
    value = np.asarray(value)
    target = np.asarray(target)
    _require(value.shape == target.shape, "normalized-L2 shapes differ")
    denominator = float(np.linalg.norm(target))
    _require(denominator > 0.0, "normalized-L2 target has zero energy")
    return float(np.linalg.norm(value - target) / denominator)


def complex_pair_metrics(
    source: np.ndarray,
    target: np.ndarray,
    *,
    shell: np.ndarray,
    max_shell: int,
) -> dict[str, Any]:
    """Return phase-sensitive FSC and amplitude diagnostics for complex data."""

    source = np.asarray(source).reshape(-1)
    target = np.asarray(target).reshape(-1)
    shell = np.asarray(shell, dtype=np.int64).reshape(-1)
    _require(
        source.shape == target.shape == shell.shape,
        "complex pair and shell arrays differ in shape",
    )
    _require(
        np.all(np.isfinite(source)) and np.all(np.isfinite(target)),
        "complex pair contains non-finite values",
    )
    valid = shell <= int(max_shell)
    source = source[valid]
    target = target[valid]
    shell = shell[valid]
    target_norm = float(np.linalg.norm(target))
    source_energy = float(np.vdot(source, source).real)
    _require(target_norm > 0.0 and source_energy > 0.0, "complex pair has zero energy")

    sign = -1 if np.linalg.norm(-source - target) < np.linalg.norm(source - target) else 1
    source = sign * source
    numerator = np.bincount(
        shell,
        weights=np.real(source * np.conj(target)),
        minlength=int(max_shell) + 1,
    )
    source_power = np.bincount(
        shell,
        weights=np.abs(source) ** 2,
        minlength=int(max_shell) + 1,
    )
    target_power = np.bincount(
        shell,
        weights=np.abs(target) ** 2,
        minlength=int(max_shell) + 1,
    )
    denominator = np.sqrt(source_power * target_power)
    fsc = np.full(int(max_shell) + 1, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=fsc, where=denominator > 0.0)

    global_scale = float(np.vdot(source, target).real / source_energy)
    _require(
        np.isfinite(global_scale) and global_scale > 0.0,
        f"complex-pair amplitude scale is invalid: {global_scale}",
    )
    before = _normalized_l2(source, target)
    after = _normalized_l2(source * global_scale, target)
    return {
        "sign_applied_to_recovar": int(sign),
        "fsc_auc": float(normalized_fsc_auc(fsc)),
        "relative_l2": before,
        "global_scale_recovar_to_relion": global_scale,
        "relative_l2_after_global_scale": after,
        "global_scale_explained_fraction": (
            float((before - after) / before) if before > 0.0 else 0.0
        ),
        "fsc": fsc,
    }


def compare_fsc_curves(
    recovar_curve: np.ndarray,
    relion_curve: np.ndarray,
    *,
    max_shell: int,
) -> dict[str, Any]:
    recovar = np.asarray(recovar_curve, dtype=np.float64).reshape(-1)
    relion = np.asarray(relion_curve, dtype=np.float64).reshape(-1)
    count = min(recovar.size, relion.size, int(max_shell) + 1)
    _require(count > 1, "BPref FSC curves have no non-DC shells")
    recovar = recovar[:count]
    relion = relion[:count]
    delta = recovar - relion
    return {
        "shell_range": [0, count - 1],
        "recovar_fsc_auc": float(normalized_fsc_auc(recovar)),
        "relion_fsc_auc": float(normalized_fsc_auc(relion)),
        "fsc_auc_delta_recovar_minus_relion": float(
            normalized_fsc_auc(recovar) - normalized_fsc_auc(relion)
        ),
        "maximum_absolute_shell_delta": float(np.nanmax(np.abs(delta[1:]))),
        "maximum_delta_shell": int(np.nanargmax(np.abs(delta[1:])) + 1),
        "curve_l2": float(np.linalg.norm(delta[np.isfinite(delta)])),
        "recovar_fsc": recovar,
        "relion_fsc": relion,
    }


def classify_boundary(
    *,
    repeat_fsc_aucs: list[float],
    accumulator_fsc_aucs: list[float],
    accumulator_relative_l2: list[float],
    repeat_fsc_auc_gate: float,
    accumulator_fsc_auc_gate: float,
    accumulator_relative_l2_gate: float,
) -> dict[str, Any]:
    _require(len(repeat_fsc_aucs) == 2, "two repeat-map FSC-AUC values are required")
    _require(
        len(accumulator_fsc_aucs) == len(accumulator_relative_l2) == 2,
        "two accumulator comparisons are required",
    )
    repeat_pass = bool(min(repeat_fsc_aucs) >= float(repeat_fsc_auc_gate))
    accumulator_difference = bool(
        min(accumulator_fsc_aucs) < float(accumulator_fsc_auc_gate)
        or max(accumulator_relative_l2) > float(accumulator_relative_l2_gate)
    )
    if not repeat_pass:
        classification = "inconclusive_relion_diagnostic_repeat_gate_failed"
    elif accumulator_difference:
        classification = (
            "physical_iteration2_postjoin_bpref_contains_cross_engine_residual"
        )
    else:
        classification = (
            "physical_iteration2_postjoin_bpref_has_no_material_cross_engine_residual"
        )
    return {
        "classification": classification,
        "relion_repeat_gate_pass": repeat_pass,
        "material_accumulator_difference": accumulator_difference,
        "gates": {
            "minimum_relion_repeat_map_fsc_auc": float(repeat_fsc_auc_gate),
            "minimum_accumulator_cross_engine_fsc_auc": float(
                accumulator_fsc_auc_gate
            ),
            "maximum_accumulator_cross_engine_relative_l2": float(
                accumulator_relative_l2_gate
            ),
        },
    }


def _load_recovar_map(path: Path) -> np.ndarray:
    from recovar.utils.helpers import load_mrc

    _require(path.is_file(), f"missing RECOVAR map: {path}")
    return np.asarray(load_mrc(str(path)), dtype=np.float64)


def _load_relion_map(path: Path) -> np.ndarray:
    from recovar.utils.helpers import load_relion_volume

    _require(path.is_file(), f"missing RELION map: {path}")
    return np.asarray(load_relion_volume(str(path)), dtype=np.float64)


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "fsc"}


def analyze(
    *,
    recovar_accumulator: Path,
    relion_dump_dir: Path,
    relion_call: int,
    recovar_map_dir: Path,
    relion_repeat_dir: Path,
    relion_qualified_dir: Path,
    physical_iteration: int,
    run_id: str,
    repeat_fsc_auc_gate: float,
    accumulator_fsc_auc_gate: float,
    accumulator_relative_l2_gate: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    recovar_accumulator = recovar_accumulator.resolve()
    relion_dump_dir = relion_dump_dir.resolve()
    recovar_map_dir = recovar_map_dir.resolve()
    relion_repeat_dir = relion_repeat_dir.resolve()
    relion_qualified_dir = relion_qualified_dir.resolve()
    _require(recovar_accumulator.is_file(), f"missing {recovar_accumulator}")
    _require(relion_dump_dir.is_dir(), f"missing {relion_dump_dir}")
    _require(physical_iteration >= 1, "physical iteration must be positive")

    with np.load(recovar_accumulator, allow_pickle=False) as archive:
        schema = str(np.asarray(archive["schema"]).item())
        _require(schema == "recovar-bpref-accum-v2", f"unexpected schema {schema!r}")
        archive_iteration = int(np.asarray(archive["iteration"]).item())
        archive_run_id = str(np.asarray(archive["run_id"]).item())
        _require(
            archive_iteration == int(physical_iteration),
            f"RECOVAR dump iteration {archive_iteration} != {physical_iteration}",
        )
        _require(archive_run_id == run_id, f"RECOVAR run_id {archive_run_id!r} != {run_id!r}")
        padding_factor = int(np.asarray(archive["padding_factor"]).item())
        grid_size = int(np.asarray(archive["grid_size"]).item())
        current_size = int(np.asarray(archive["current_size"]).item())
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

    max_shell = current_size // 2
    recovar_avg: list[np.ndarray] = []
    recovar_down_weight: list[np.ndarray] = []
    relion: list[dict[str, Any]] = []
    coordinate_metrics: list[dict[str, Any]] = []
    shell_labels: np.ndarray | None = None
    for half in (0, 1):
        avg, weight, down_radius, _, _, actual_max_shell = (
            downsample_recovar_accumulator(
                recovar_numerator[half],
                recovar_weight[half],
                volume_shape,
                padding_factor,
                max_shell,
                accumulator_shape=accumulator_shape,
            )
        )
        _require(actual_max_shell == max_shell, "RECOVAR max-shell mismatch")
        avg, weight = _apply_recovar_frame(
            avg,
            weight,
            grid_size=grid_size,
            recovar_frame="relion",
        )
        rel_path = (
            relion_dump_dir
            / f"downsampled_avg_rank{half + 1:02d}_call{relion_call:04d}.txt"
        )
        _require(rel_path.is_file(), f"missing {rel_path}")
        rel = load_relion_dump(rel_path)
        _require(int(rel["r_max"]) == max_shell, "RELION max-shell mismatch")
        k = np.asarray(rel["k"], dtype=np.int64)
        i = np.asarray(rel["i"], dtype=np.int64)
        j = np.asarray(rel["j"], dtype=np.int64)
        rec_avg = avg[i + down_radius, k + down_radius, j]
        rec_weight = weight[i + down_radius, k + down_radius, j]
        rel_avg = np.asarray(rel["real"]) + 1j * np.asarray(rel["imag"])
        rel_weight = np.asarray(rel["weight"])
        shell = np.rint(np.sqrt(k * k + i * i + j * j)).astype(np.int64)
        pair = complex_pair_metrics(
            rec_avg,
            rel_avg,
            shell=shell,
            max_shell=max_shell,
        )
        rec_support = rec_weight > 0.0
        rel_support = rel_weight > 0.0
        support_union = int(np.count_nonzero(rec_support | rel_support))
        coordinate_metrics.append(
            {
                **_public_metrics(pair),
                "weight_relative_l2": _normalized_l2(rec_weight, rel_weight),
                "support_jaccard": float(
                    np.count_nonzero(rec_support & rel_support)
                    / max(support_union, 1)
                ),
                "coordinate_count": int(k.size),
            }
        )
        recovar_avg.append(rec_avg)
        recovar_down_weight.append(rec_weight)
        relion.append(rel)
        if shell_labels is None:
            shell_labels = shell
        else:
            _require(np.array_equal(shell_labels, shell), "RELION half coordinates differ")

    assert shell_labels is not None
    recovar_cross_half = complex_pair_metrics(
        recovar_avg[0],
        recovar_avg[1],
        shell=shell_labels,
        max_shell=max_shell,
    )["fsc"]
    relion_fsc_path = (
        relion_dump_dir / f"downsampled_fsc_rank01_call{relion_call:04d}.txt"
    )
    _require(relion_fsc_path.is_file(), f"missing {relion_fsc_path}")
    relion_fsc = np.loadtxt(relion_fsc_path, comments="#", ndmin=2)[:, 4]
    fsc_comparison = compare_fsc_curves(
        recovar_cross_half,
        relion_fsc,
        max_shell=max_shell,
    )

    recovar_reference_iteration = int(physical_iteration) - 1
    repeat_map_metrics = []
    cross_engine_map_metrics = []
    artifact_paths = [recovar_accumulator, relion_fsc_path]
    for half in (1, 2):
        recovar_map_path = (
            recovar_map_dir
            / f"run_it{recovar_reference_iteration:03d}_half{half}_class001.mrc"
        )
        repeat_map_path = (
            relion_repeat_dir
            / f"run_it{physical_iteration:03d}_half{half}_class001.mrc"
        )
        qualified_map_path = (
            relion_qualified_dir
            / f"run_it{physical_iteration:03d}_half{half}_class001.mrc"
        )
        recovar_map = _load_recovar_map(recovar_map_path)
        repeat_map = _load_relion_map(repeat_map_path)
        qualified_map = _load_relion_map(qualified_map_path)
        _require(
            recovar_map.shape == repeat_map.shape == qualified_map.shape,
            "reference-map shapes differ",
        )
        repeat_map_metrics.append(map_metrics(repeat_map, qualified_map))
        cross_engine_map_metrics.append(map_metrics(recovar_map, repeat_map))
        artifact_paths.extend(
            [recovar_map_path, repeat_map_path, qualified_map_path]
        )
    for half in (1, 2):
        artifact_paths.append(
            relion_dump_dir
            / f"downsampled_avg_rank{half:02d}_call{relion_call:04d}.txt"
        )

    classification = classify_boundary(
        repeat_fsc_aucs=[row["fsc_auc"] for row in repeat_map_metrics],
        accumulator_fsc_aucs=[row["fsc_auc"] for row in coordinate_metrics],
        accumulator_relative_l2=[row["relative_l2"] for row in coordinate_metrics],
        repeat_fsc_auc_gate=repeat_fsc_auc_gate,
        accumulator_fsc_auc_gate=accumulator_fsc_auc_gate,
        accumulator_relative_l2_gate=accumulator_relative_l2_gate,
    )
    report = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "parity_score": False,
        "metric_policy": (
            "FSC/FSC-AUC are primary; complex normalized L2, positive "
            "least-squares amplitude, and weight/support are secondary; "
            "correlation is not computed"
        ),
        "boundary": (
            "physical-iteration-2 post-low-resolution-join BPref aggregate "
            "before tau2 update and Wiener reconstruction"
        ),
        "physical_iteration": int(physical_iteration),
        "recovar_reference_iteration": recovar_reference_iteration,
        "relion_call": int(relion_call),
        "run_id": run_id,
        "recovar_dump": {
            "path": str(recovar_accumulator),
            "sha256": _sha256(recovar_accumulator),
            "current_size": current_size,
            "max_shell": max_shell,
            "padding_factor": padding_factor,
            "grid_size": grid_size,
            "volume_shape": list(volume_shape),
            "accumulator_shape": list(accumulator_shape),
        },
        "relion_repeat_vs_qualified_maps": {
            f"half{half}": metrics
            for half, metrics in enumerate(repeat_map_metrics, start=1)
        },
        "recovar_vs_relion_repeat_maps": {
            f"half{half}": metrics
            for half, metrics in enumerate(cross_engine_map_metrics, start=1)
        },
        "postjoin_coordinate_diagnostics": {
            f"half{half}": metrics
            for half, metrics in enumerate(coordinate_metrics, start=1)
        },
        "postjoin_cross_half_fsc_comparison": {
            key: value
            for key, value in fsc_comparison.items()
            if key not in {"recovar_fsc", "relion_fsc"}
        },
        **classification,
        "interpretation_limit": (
            "A positive result localizes an already-present cross-engine "
            "residual to the post-join accumulator boundary; it does not by "
            "itself distinguish candidate membership, scatter arithmetic, "
            "or accumulator-frame conversion."
        ),
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in artifact_paths
        ],
    }
    curves = {
        "recovar_cross_half_fsc": np.asarray(fsc_comparison["recovar_fsc"]),
        "relion_cross_half_fsc": np.asarray(fsc_comparison["relion_fsc"]),
    }
    return report, curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-accumulator", required=True, type=Path)
    parser.add_argument("--relion-dump-dir", required=True, type=Path)
    parser.add_argument("--relion-call", type=int, default=1)
    parser.add_argument("--recovar-map-dir", required=True, type=Path)
    parser.add_argument("--relion-repeat-dir", required=True, type=Path)
    parser.add_argument("--relion-qualified-dir", required=True, type=Path)
    parser.add_argument("--physical-iteration", type=int, default=2)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeat-fsc-auc-gate", type=float, default=0.99999)
    parser.add_argument("--accumulator-fsc-auc-gate", type=float, default=0.99999)
    parser.add_argument(
        "--accumulator-relative-l2-gate",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-curves", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report, curves = analyze(
        recovar_accumulator=args.recovar_accumulator,
        relion_dump_dir=args.relion_dump_dir,
        relion_call=args.relion_call,
        recovar_map_dir=args.recovar_map_dir,
        relion_repeat_dir=args.relion_repeat_dir,
        relion_qualified_dir=args.relion_qualified_dir,
        physical_iteration=args.physical_iteration,
        run_id=args.run_id,
        repeat_fsc_auc_gate=args.repeat_fsc_auc_gate,
        accumulator_fsc_auc_gate=args.accumulator_fsc_auc_gate,
        accumulator_relative_l2_gate=args.accumulator_relative_l2_gate,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_curves.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez(args.output_curves, **curves)
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
