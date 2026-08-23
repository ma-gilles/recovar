#!/usr/bin/env python3
"""Compare every shared input field of two K=1 final-pass manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_k1_coarse_capture_ab import _metrics, _sha256
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from analyze_k1_coarse_capture_ab import _metrics, _sha256


ORDERED_FIELDS = (
    "current_size",
    "half_spectrum_scoring",
    "use_float64_scoring",
    "use_float64_projections",
    "projection_padding_factor",
    "reconstruction_padding_factor",
    "score_with_masked_images",
    "perturbation_instance",
    "perturbation_factor",
    "perturbation_applied",
    "perturbation_relion_iteration",
    "local_search",
    "effective_rotations",
    "current_translations",
    "rotation_log_prior",
    "translation_log_prior",
    "translation_prior_centers",
    "image_pre_shifts",
    "absolute_previous_translations",
    "image_corrections",
    "scale_corrections",
    "noise_variance",
    "mean_variance",
    "mean_vol_ft",
)

PARTICLE_FIELDS = frozenset(
    {
        "rotation_log_prior",
        "translation_log_prior",
        "translation_prior_centers",
        "image_corrections",
        "scale_corrections",
        "image_pre_shifts",
        "absolute_previous_translations",
    }
)


def _json_safe(value: object) -> object:
    """Encode undefined numeric diagnostics as JSON null, not NaN/Infinity."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _gather_sha256(gather: np.ndarray) -> str:
    values = np.ascontiguousarray(gather, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _fresh_candidate_row_gathers(
    *,
    candidate_particle_star: Path,
    relion_halfset_star: Path,
    fresh_order_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map source-order continuation rows into fresh RELION physical order."""

    import starfile

    from recovar.em.dense_single_volume.helpers.expected_accuracy import (
        relion_auto_refine_half_orders,
    )

    candidate = starfile.read(candidate_particle_star)
    candidate = candidate["particles"] if isinstance(candidate, dict) else candidate
    relion = starfile.read(relion_halfset_star)
    relion = relion["particles"] if isinstance(relion, dict) else relion
    candidate_names = tuple(str(value) for value in candidate["rlnImageName"])
    relion_names = tuple(str(value) for value in relion["rlnImageName"])
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("candidate particle STAR has duplicate rlnImageName identities")
    if len(set(relion_names)) != len(relion_names):
        raise ValueError("RELION half-set STAR has duplicate rlnImageName identities")
    candidate_row_by_name = {name: row for row, name in enumerate(candidate_names)}
    relion_row_by_name = {name: row for row, name in enumerate(relion_names)}
    if set(candidate_row_by_name) != set(relion_row_by_name):
        raise ValueError("candidate and RELION STAR particle identities differ")

    subsets = np.asarray(relion["rlnRandomSubset"], dtype=np.int64)
    optics = (
        np.asarray(relion["rlnOpticsGroup"], dtype=np.int64)
        if "rlnOpticsGroup" in relion.columns
        else None
    )
    candidate_relion_rows = np.asarray(
        [relion_row_by_name[name] for name in candidate_names],
        dtype=np.int64,
    )
    candidate_subsets = subsets[candidate_relion_rows]
    base_rows = tuple(
        np.flatnonzero(candidate_subsets == half).astype(np.int64)
        for half in (1, 2)
    )
    ordered_relion_rows = relion_auto_refine_half_orders(
        subsets,
        int(fresh_order_seed),
        1,
        optics_group_ids=optics,
    )
    physical_rows = tuple(
        np.asarray(
            [candidate_row_by_name[relion_names[int(row)]] for row in order],
            dtype=np.int64,
        )
        for order in ordered_relion_rows
    )

    gathers = []
    for half, (base, physical) in enumerate(zip(base_rows, physical_rows), start=1):
        base_position = np.full(len(candidate_names), -1, dtype=np.int64)
        base_position[base] = np.arange(base.size, dtype=np.int64)
        gather = base_position[physical]
        if np.any(gather < 0) or not np.array_equal(
            np.sort(gather), np.arange(base.size, dtype=np.int64)
        ):
            raise ValueError(f"half {half} source-to-physical row map is not bijective")
        gathers.append(gather)
    return gathers[0], gathers[1]


def _compare_half(
    control_path: Path,
    candidate_path: Path,
    expected_half: int,
    *,
    candidate_row_gather: np.ndarray | None = None,
) -> dict[str, object]:
    with np.load(control_path, allow_pickle=False) as control, np.load(
        candidate_path, allow_pickle=False
    ) as candidate:
        for archive, label in ((control, "control"), (candidate, "candidate")):
            missing = [field for field in ORDERED_FIELDS if field not in archive.files]
            if missing:
                raise ValueError(f"{label} final manifest is missing fields: {missing}")
            observed_half = int(np.asarray(archive["half_index"]).item())
            if observed_half != expected_half:
                raise ValueError(
                    f"{label} manifest half_index={observed_half}, expected {expected_half}"
                )

        fields: dict[str, dict[str, object]] = {}
        for field in ORDERED_FIELDS:
            candidate_value = np.asarray(candidate[field])
            if candidate_row_gather is not None and field in PARTICLE_FIELDS:
                if candidate_value.ndim > 0 and candidate_value.shape[0] > 0:
                    if candidate_value.shape[0] != candidate_row_gather.size:
                        raise ValueError(
                            f"candidate {field} particle axis has {candidate_value.shape[0]} "
                            f"rows, expected {candidate_row_gather.size}"
                        )
                    candidate_value = candidate_value[candidate_row_gather]
            fields[field] = _metrics(control[field], candidate_value)
        first_unequal = next(
            (
                field
                for field in ORDERED_FIELDS
                if not fields[field].get("shape_equal", False)
                or fields[field].get("bit_equal_fraction") != 1.0
            ),
            None,
        )

    return {
        "half_index": expected_half,
        "first_non_bit_exact_field": first_unequal,
        "fields": fields,
        "artifacts": {
            "control": str(control_path.resolve()),
            "control_sha256": _sha256(control_path),
            "candidate": str(candidate_path.resolve()),
            "candidate_sha256": _sha256(candidate_path),
        },
        "candidate_row_alignment": (
            None
            if candidate_row_gather is None
            else {
                "semantics": "control physical row -> candidate source-half local row",
                "row_count": int(candidate_row_gather.size),
                "sha256": _gather_sha256(candidate_row_gather),
            }
        ),
    }


def analyze(
    *,
    control_half1: Path,
    control_half2: Path,
    candidate_half1: Path,
    candidate_half2: Path,
    candidate_row_gathers: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, object]:
    gathers = (None, None) if candidate_row_gathers is None else candidate_row_gathers
    halves = [
        _compare_half(control_half1, candidate_half1, 0, candidate_row_gather=gathers[0]),
        _compare_half(control_half2, candidate_half2, 1, candidate_row_gather=gathers[1]),
    ]
    first_unequal = next(
        (
            {"half_index": half["half_index"], "field": half["first_non_bit_exact_field"]}
            for half in halves
            if half["first_non_bit_exact_field"] is not None
        ),
        None,
    )
    return {
        "schema": "recovar.em.k1_final_manifest_ab.v1",
        "status": "complete",
        "metric_policy": "exact bytes, maximum absolute error, and relative L2; no correlation",
        "field_order": list(ORDERED_FIELDS),
        "first_non_bit_exact": first_unequal,
        "halves": halves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-half1", required=True, type=Path)
    parser.add_argument("--control-half2", required=True, type=Path)
    parser.add_argument("--candidate-half1", required=True, type=Path)
    parser.add_argument("--candidate-half2", required=True, type=Path)
    parser.add_argument("--candidate-particle-star", type=Path)
    parser.add_argument("--relion-halfset-star", type=Path)
    parser.add_argument("--fresh-order-seed", type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    alignment_args = (
        args.candidate_particle_star,
        args.relion_halfset_star,
        args.fresh_order_seed,
    )
    if any(value is not None for value in alignment_args) and not all(
        value is not None for value in alignment_args
    ):
        raise ValueError(
            "candidate row alignment requires --candidate-particle-star, "
            "--relion-halfset-star, and --fresh-order-seed together"
        )
    candidate_row_gathers = (
        None
        if args.fresh_order_seed is None
        else _fresh_candidate_row_gathers(
            candidate_particle_star=args.candidate_particle_star,
            relion_halfset_star=args.relion_halfset_star,
            fresh_order_seed=args.fresh_order_seed,
        )
    )
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        control_half1=args.control_half1,
        control_half2=args.control_half2,
        candidate_half1=args.candidate_half1,
        candidate_half2=args.candidate_half2,
        candidate_row_gathers=candidate_row_gathers,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
