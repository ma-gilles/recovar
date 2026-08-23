#!/usr/bin/env python3
"""Compare native RELION and RECOVAR StoreWavg posteriors across a particle panel."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from scripts.analyze_vdam_storewavg_boundary import (
    _load_native,
    _match_rotations,
    _metric,
    _posterior_metric,
    _production_score_gradient_rows,
    _require,
)

SCHEMA = "recovar.vdam_storewavg_panel.v1"
_NATIVE_NAME = re.compile(
    r"^(?P<prefix>.*_part(?P<part_id>[0-9]+)_storeWavg_)(?P<suffix>[^/]+\.bin)$"
)
_REQUIRED_NATIVE_SUFFIXES = frozenset(
    {
        "orientation_num.bin",
        "translation_num.bin",
        "sorted_weights.bin",
        "sum_weight.bin",
        "significant_weight.bin",
        "eulers.bin",
        "trans_xyz.bin",
        "ctfs.bin",
    }
)


def _original_index_from_image_name(image_name: str) -> int:
    """Convert RELION's one-based ``N@stack`` identity to RECOVAR's index."""

    image_number, separator, _stack = str(image_name).partition("@")
    _require(separator == "@", f"invalid RELION image identity: {image_name!r}")
    value = int(image_number)
    _require(value > 0, f"RELION image number must be positive: {image_name!r}")
    return value - 1


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0, "cannot summarize an empty panel")
    return {
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _pooled_relative_l2(reference_norm2: float, residual_norm2: float) -> float:
    return float(np.sqrt(residual_norm2 / max(reference_norm2, np.finfo(np.float64).tiny)))


def _native_prefixes(native_directory: Path) -> tuple[dict[int, str], dict[int, list[str]]]:
    prefixes: dict[int, str] = {}
    suffixes: dict[int, set[str]] = {}
    for path in native_directory.glob("*.bin"):
        match = _NATIVE_NAME.match(path.name)
        if match is None:
            continue
        part_id = int(match.group("part_id"))
        prefix = match.group("prefix")
        if part_id in prefixes:
            _require(prefixes[part_id] == prefix, f"duplicate native StoreWavg prefix for part_id {part_id}")
        prefixes[part_id] = prefix
        suffixes.setdefault(part_id, set()).add(match.group("suffix"))
    complete = {
        part_id: prefix
        for part_id, prefix in prefixes.items()
        if _REQUIRED_NATIVE_SUFFIXES.issubset(suffixes[part_id])
    }
    incomplete = {
        part_id: sorted(_REQUIRED_NATIVE_SUFFIXES.difference(suffixes[part_id]))
        for part_id in prefixes
        if part_id not in complete
    }
    _require(prefixes, f"no native StoreWavg captures found in {native_directory}")
    return complete, incomplete


def _score_dumps(score_directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in score_directory.glob("local_score_it*_image_*_single_class.npz"):
        with np.load(path, allow_pickle=False) as archive:
            indices = np.asarray(archive["selected_global_image_indices"], dtype=np.int64)
        _require(indices.size == 1, f"score dump is not single-particle: {path}")
        original_index = int(indices[0])
        _require(original_index not in result, f"duplicate score dump for original index {original_index}")
        result[original_index] = path
    _require(result, f"no RECOVAR score dumps found in {score_directory}")
    return result


def _part_to_original_indices(relion_data_star: Path) -> dict[int, int]:
    import starfile

    star = starfile.read(relion_data_star)
    particles = star["particles"] if isinstance(star, dict) else star
    _require("rlnImageName" in particles, "RELION data STAR has no rlnImageName column")
    return {
        int(part_id): _original_index_from_image_name(image_name)
        for part_id, image_name in enumerate(particles["rlnImageName"])
    }


def analyze(
    native_directory: Path,
    score_directory: Path,
    relion_data_star: Path,
    *,
    rotation_tolerance: float = 1.0e-5,
) -> dict[str, object]:
    native_by_part, incomplete_native = _native_prefixes(native_directory)
    score_by_original = _score_dumps(score_directory)
    original_by_part = _part_to_original_indices(relion_data_star)

    candidate_paired_parts = [
        part_id
        for part_id in sorted(native_by_part)
        if part_id in original_by_part and original_by_part[part_id] in score_by_original
    ]
    _require(candidate_paired_parts, "native and RECOVAR captures have no common particles")

    posterior_rel_l2: list[float] = []
    posterior_l1: list[float] = []
    posterior_mass_delta_abs: list[float] = []
    support_mismatch: list[int] = []
    data_rel_l2: list[float] = []
    weight_rel_l2: list[float] = []
    argmax_mismatches = 0
    processed_parts: list[int] = []
    invalid_native_pairs: dict[int, str] = {}
    posterior_reference_norm2 = 0.0
    posterior_residual_norm2 = 0.0
    data_reference_norm2 = 0.0
    data_residual_norm2 = 0.0
    weight_reference_norm2 = 0.0
    weight_residual_norm2 = 0.0
    per_particle: list[dict[str, object]] = []

    for part_id in candidate_paired_parts:
        original_index = original_by_part[part_id]
        score_path = score_by_original[original_index]
        with np.load(score_path, allow_pickle=False) as archive:
            score = {name: archive[name] for name in archive.files}
        try:
            native = _load_native(
                native_directory,
                native_by_part[part_id],
                load_projector=False,
            )
            rotations = np.asarray(score["local_rotation_matrices"], dtype=np.float32)
            rotation_map = _match_rotations(
                np.asarray(native["rotations"], dtype=np.float32),
                rotations,
                rotation_tolerance,
            )
        except (OSError, ValueError) as error:
            invalid_native_pairs[part_id] = f"{type(error).__name__}: {error}"
            continue
        processed_parts.append(part_id)
        current_data, current_weight, current_probs = _production_score_gradient_rows(score)
        current_native_order = current_probs[rotation_map]
        native_probs = np.asarray(native["probabilities"], dtype=np.float32)
        _require(
            native_probs.shape == current_native_order.shape,
            f"posterior topology differs for part_id {part_id}",
        )
        posterior = _posterior_metric(native_probs, current_native_order)

        native_probs_recovar_order = np.zeros_like(current_probs)
        native_probs_recovar_order[rotation_map] = native_probs
        controlled_data, controlled_weight, _ = _production_score_gradient_rows(
            score,
            reconstruction_probs_override=native_probs_recovar_order,
        )
        data_metric = _metric(controlled_data, current_data)
        weight_metric = _metric(controlled_weight, current_weight)

        posterior_residual = current_native_order.astype(np.float64) - native_probs.astype(np.float64)
        data_residual = current_data.astype(np.complex128) - controlled_data.astype(np.complex128)
        weight_residual = current_weight.astype(np.float64) - controlled_weight.astype(np.float64)
        posterior_reference_norm2 += float(np.vdot(native_probs, native_probs).real)
        posterior_residual_norm2 += float(np.vdot(posterior_residual, posterior_residual).real)
        data_reference_norm2 += float(np.vdot(controlled_data, controlled_data).real)
        data_residual_norm2 += float(np.vdot(data_residual, data_residual).real)
        weight_reference_norm2 += float(np.vdot(controlled_weight, controlled_weight).real)
        weight_residual_norm2 += float(np.vdot(weight_residual, weight_residual).real)

        argmax_mismatch = int(np.argmax(native_probs) != np.argmax(current_native_order))
        argmax_mismatches += argmax_mismatch
        posterior_rel_l2.append(float(posterior["relative_l2"]))
        posterior_l1.append(float(posterior["l1"]))
        posterior_mass_delta_abs.append(
            abs(float(posterior["candidate_retained_mass"]) - float(posterior["reference_retained_mass"]))
        )
        support_mismatch.append(int(posterior["support_mismatch_count"]))
        data_rel_l2.append(float(data_metric["relative_l2"]))
        weight_rel_l2.append(float(weight_metric["relative_l2"]))
        per_particle.append(
            {
                "part_id": part_id,
                "original_index": original_index,
                "native_prefix": native_by_part[part_id],
                "score_dump": str(score_path.resolve()),
                "posterior": posterior,
                "argmax_mismatch": bool(argmax_mismatch),
                "same_operands_native_posterior_data": data_metric,
                "same_operands_native_posterior_weight": weight_metric,
            }
        )

    _require(processed_parts, "all candidate native StoreWavg captures were internally inconsistent")
    all_parts = set(original_by_part)
    score_originals = set(score_by_original)
    captured_parts = set(native_by_part)
    missing_native_parts = sorted(
        part_id
        for part_id in all_parts
        if original_by_part[part_id] in score_originals and part_id not in captured_parts
    )
    unpaired_native_parts = sorted(set(native_by_part).difference(candidate_paired_parts))
    return {
        "schema": SCHEMA,
        "coverage": {
            "relion_particle_count": len(original_by_part),
            "native_capture_count": len(native_by_part),
            "incomplete_native_capture_count": len(incomplete_native),
            "incomplete_native_captures": incomplete_native,
            "recovar_score_count": len(score_by_original),
            "candidate_paired_particle_count": len(candidate_paired_parts),
            "paired_particle_count": len(processed_parts),
            "invalid_native_pair_count": len(invalid_native_pairs),
            "invalid_native_pairs": invalid_native_pairs,
            "missing_native_part_ids": missing_native_parts,
            "missing_native_original_indices": [original_by_part[value] for value in missing_native_parts],
            "unpaired_native_part_ids": unpaired_native_parts,
        },
        "posterior": {
            "relative_l2": _quantiles(posterior_rel_l2),
            "l1": _quantiles(posterior_l1),
            "absolute_retained_mass_delta": _quantiles(posterior_mass_delta_abs),
            "support_mismatch_count": _quantiles([float(value) for value in support_mismatch]),
            "argmax_mismatch_count": argmax_mismatches,
            "pooled_relative_l2": _pooled_relative_l2(
                posterior_reference_norm2, posterior_residual_norm2
            ),
        },
        "same_operands_native_posterior_control": {
            "data_relative_l2": _quantiles(data_rel_l2),
            "weight_relative_l2": _quantiles(weight_rel_l2),
            "pooled_data_relative_l2": _pooled_relative_l2(
                data_reference_norm2, data_residual_norm2
            ),
            "pooled_weight_relative_l2": _pooled_relative_l2(
                weight_reference_norm2, weight_residual_norm2
            ),
        },
        "artifacts": {
            "native_directory": str(native_directory.resolve()),
            "score_directory": str(score_directory.resolve()),
            "relion_data_star": str(relion_data_star.resolve()),
        },
        "per_particle": per_particle,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--score-directory", type=Path, required=True)
    parser.add_argument("--relion-data-star", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotation-tolerance", type=float, default=1.0e-5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = analyze(
        args.native_directory,
        args.score_directory,
        args.relion_data_star,
        rotation_tolerance=args.rotation_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("coverage", "posterior", "same_operands_native_posterior_control")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
