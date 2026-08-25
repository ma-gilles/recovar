#!/usr/bin/env python3
"""Compare per-particle VDAM cutoff-shell noise terms with native RELION."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_vdam_storewavg_boundary import _require
from scripts.analyze_vdam_storewavg_panel import _part_to_original_indices, _score_dumps

SCHEMA = "recovar.vdam_cutoff_particle_panel.v1"
_COMPONENTS = ("direct_residual", "aa", "xa", "inferred_image_power")


def _native_rows(
    path: Path,
    *,
    iteration: int,
    half: int,
) -> dict[tuple[int, int], dict[str, float | int]]:
    rows: dict[tuple[int, int], dict[str, float | int]] = {}
    prefix = f"acc_components\titer={iteration}\t"
    half_token = f"\thalfset={half}\t"
    with path.open() as stream:
        for line in stream:
            if not line.startswith(prefix) or half_token not in line:
                continue
            fields = {
                key: value
                for key, value in (item.split("=", 1) for item in line.rstrip().split("\t")[1:])
            }
            if int(fields["optics_group"]) != 0:
                continue
            part_id = int(fields["part_id"])
            shell = int(fields["shell"])
            key = (part_id, shell)
            _require(key not in rows, f"duplicate native component row for part {part_id}, shell {shell}")
            rows[key] = {
                **{name: float(fields[name]) for name in _COMPONENTS},
                "sumw_group": float(fields["sumw_group"]),
                "npix": int(fields["Npix_per_shell"]),
            }
    _require(rows, f"no native component rows for iteration {iteration}, half {half}")
    return rows


def _summary(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    _require(candidate.shape == reference.shape and candidate.size > 0, "component topology mismatch")
    residual = candidate - reference
    return {
        "count": int(candidate.size),
        "candidate_sum": float(np.sum(candidate)),
        "reference_sum": float(np.sum(reference)),
        "signed_sum_error": float(np.sum(residual)),
        "signed_mean_error": float(np.mean(residual)),
        "negative_error_count": int(np.count_nonzero(residual < 0.0)),
        "zero_error_count": int(np.count_nonzero(residual == 0.0)),
        "positive_error_count": int(np.count_nonzero(residual > 0.0)),
        "relative_l2": float(
            np.linalg.norm(residual) / max(np.linalg.norm(reference), np.finfo(np.float64).tiny)
        ),
        "median_abs_error": float(np.median(np.abs(residual))),
        "p95_abs_error": float(np.percentile(np.abs(residual), 95)),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def analyze(
    native_components_tsv: Path,
    score_directory: Path,
    relion_data_star: Path,
    *,
    iteration: int,
    half: int,
    image_size: int,
    expected_particle_count: int | None = None,
) -> dict[str, object]:
    native = _native_rows(native_components_tsv, iteration=iteration, half=half)
    score_by_original = _score_dumps(score_directory)
    original_by_part = _part_to_original_indices(relion_data_star)
    native_parts = sorted({part_id for part_id, _shell in native})
    _require(native_parts, "native particle panel is empty")
    _require(max(native_parts) < len(original_by_part), "native part id exceeds RELION data STAR")
    if expected_particle_count is not None:
        _require(
            len(native_parts) == expected_particle_count,
            f"native particle count {len(native_parts)} != expected {expected_particle_count}",
        )

    missing_original_indices = [
        original_by_part[part_id]
        for part_id in native_parts
        if original_by_part[part_id] not in score_by_original
    ]
    _require(not missing_original_indices, f"missing RECOVAR score dumps: {missing_original_indices}")

    n4 = float(image_size**4)
    candidate_values = {name: [] for name in _COMPONENTS}
    reference_values = {name: [] for name in _COMPONENTS}
    candidate_support_mass: list[float] = []
    reference_support_mass: list[float] = []
    per_particle: list[dict[str, object]] = []
    cutoff_shells: set[int] = set()

    for part_id in native_parts:
        original_index = original_by_part[part_id]
        score_path = score_by_original[original_index]
        with np.load(score_path, allow_pickle=False) as payload:
            _require("debug_wavg_cutoff_triplet_xa_aa_diff2" in payload, f"missing cutoff triplet: {score_path}")
            current_size = int(np.asarray(payload["current_size"]).reshape(-1)[0])
            triplet = np.asarray(payload["debug_wavg_cutoff_triplet_xa_aa_diff2"], dtype=np.float64)
            posterior = np.asarray(payload["posterior"], dtype=np.float64)
            reconstruction_probs = np.asarray(
                payload["reconstruction_probs"] if "reconstruction_probs" in payload else posterior,
                dtype=np.float64,
            )
            support = np.asarray(payload["reconstruction_sample_mask"], dtype=bool)
        _require(triplet.shape == (3,), f"invalid cutoff triplet topology: {score_path}")
        _require(posterior.shape == support.shape, f"posterior/support topology mismatch: {score_path}")
        _require(
            reconstruction_probs.shape == support.shape,
            f"reconstruction-probability/support topology mismatch: {score_path}",
        )
        cutoff_shell = current_size // 2
        cutoff_shells.add(cutoff_shell)
        native_key = (part_id, cutoff_shell)
        _require(native_key in native, f"missing native row for part {part_id}, shell {cutoff_shell}")

        xa, aa, direct_residual = triplet / n4
        inferred_image_power = direct_residual - aa + 2.0 * xa
        candidate = {
            "direct_residual": float(direct_residual),
            "aa": float(aa),
            "xa": float(xa),
            "inferred_image_power": float(inferred_image_power),
        }
        reference = {name: float(native[native_key][name]) for name in _COMPONENTS}
        for name in _COMPONENTS:
            candidate_values[name].append(candidate[name])
            reference_values[name].append(reference[name])
        candidate_mass = float(np.sum(reconstruction_probs[support], dtype=np.float64))
        reference_mass = float(native[native_key]["sumw_group"])
        candidate_support_mass.append(candidate_mass)
        reference_support_mass.append(reference_mass)
        per_particle.append(
            {
                "part_id": part_id,
                "original_index": original_index,
                "cutoff_shell": cutoff_shell,
                "score_dump": str(score_path.resolve()),
                "candidate": candidate,
                "reference": reference,
                "error": {name: candidate[name] - reference[name] for name in _COMPONENTS},
                "candidate_support_mass": candidate_mass,
                "reference_support_mass": reference_mass,
                "npix": int(native[native_key]["npix"]),
            }
        )

    _require(len(cutoff_shells) == 1, f"candidate cutoff shells vary across the panel: {cutoff_shells}")
    comparisons = {
        name: _summary(np.asarray(candidate_values[name]), np.asarray(reference_values[name]))
        for name in _COMPONENTS
    }
    comparisons["support_mass"] = _summary(candidate_support_mass, reference_support_mass)
    return {
        "schema": SCHEMA,
        "identity": {
            "iteration": iteration,
            "native_halfset": half,
            "image_size": image_size,
            "cutoff_shell": next(iter(cutoff_shells)),
            "particle_count": len(per_particle),
        },
        "comparisons": comparisons,
        "artifacts": {
            "native_components_tsv": str(native_components_tsv.resolve()),
            "score_directory": str(score_directory.resolve()),
            "relion_data_star": str(relion_data_star.resolve()),
        },
        "per_particle": per_particle,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-components-tsv", type=Path, required=True)
    parser.add_argument("--score-directory", type=Path, required=True)
    parser.add_argument("--relion-data-star", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--half", type=int, default=-1)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--expected-particle-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = analyze(
        args.native_components_tsv,
        args.score_directory,
        args.relion_data_star,
        iteration=args.iteration,
        half=args.half,
        image_size=args.image_size,
        expected_particle_count=args.expected_particle_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("identity", "comparisons")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
