#!/usr/bin/env python3
"""Classify a RELION/RECOVAR fine-pass top-candidate discrepancy.

This is a localization diagnostic, not a parity acceptance gate.  It requires
an exact all-particle RELION capture-inertness report and compares the raw
pre-prior scores of the two engines' distinct top candidates.  Exact ties in
both engines localize the winner difference to candidate enumeration/tie
order; a non-tie in either engine localizes it to fine-score arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

_INERTNESS_SCHEMA = "em_relion_iteration1_particle_state_inertness_v1"
_REQUIRED_INERTNESS_FIELDS = (
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnOriginXAngst",
    "rlnOriginYAngst",
    "rlnClassNumber",
    "rlnMaxValueProbDistribution",
    "rlnNrOfSignificantSamples",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_key(value: Any, *, field: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be a two-element list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field} must contain two integer indices")
    return int(value[0]), int(value[1])


def _finite_score(details: dict[str, Any], *, engine: str, key: tuple[int, int]) -> float:
    engine_details = details.get(engine)
    if not isinstance(engine_details, dict):
        raise ValueError(f"candidate {list(key)} has no {engine} score details")
    value = engine_details.get("score_pre_prior")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{engine} candidate {list(key)} has no finite raw pre-prior score")
    return float(value)


def _validate_inertness(
    inertness: dict[str, Any],
    *,
    expected_original_index: int,
    expected_particle_count: int,
) -> None:
    if inertness.get("schema") != _INERTNESS_SCHEMA:
        raise ValueError("RELION inertness schema is missing or unknown")
    if inertness.get("status") != "pass":
        raise ValueError("RELION capture inertness did not pass")
    if inertness.get("target_original_index") != expected_original_index:
        raise ValueError("RELION inertness target particle does not match")
    if inertness.get("particle_count") != expected_particle_count:
        raise ValueError("RELION inertness particle count does not match")
    fields = inertness.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("RELION inertness field results are missing")
    for field in _REQUIRED_INERTNESS_FIELDS:
        result = fields.get(field)
        if not isinstance(result, dict):
            raise ValueError(f"RELION inertness result is missing field {field}")
        if result.get("exact") is not True or result.get("mismatch_count") != 0:
            raise ValueError(f"RELION inertness field {field} is not exact")
        max_abs = result.get("max_abs")
        if max_abs is not None and float(max_abs) != 0.0:
            raise ValueError(f"RELION inertness field {field} has a nonzero maximum difference")


def classify(
    comparison: dict[str, Any],
    inertness: dict[str, Any],
    *,
    expected_original_index: int,
    expected_current_size: int,
    expected_particle_count: int,
) -> dict[str, Any]:
    """Validate the evidence and classify its exact two-candidate branch."""

    _validate_inertness(
        inertness,
        expected_original_index=expected_original_index,
        expected_particle_count=expected_particle_count,
    )
    if comparison.get("match_mode") != "matrix":
        raise ValueError("fine-pass comparison must use Euler-matrix matching")
    if comparison.get("reconstruction_only") is not False:
        raise ValueError("fine-pass comparison must cover the full candidate support")
    if comparison.get("recovar_original_index") != expected_original_index:
        raise ValueError("comparison target particle does not match")
    if comparison.get("recovar_current_size") != expected_current_size:
        raise ValueError("comparison current size does not match")
    for field in ("relion_candidate_count", "recovar_candidate_count", "common_candidate_count"):
        value = comparison.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"comparison {field} must be a positive integer")

    relion_top = _candidate_key(comparison.get("relion_top_key"), field="relion_top_key")
    recovar_top = _candidate_key(comparison.get("recovar_top_key"), field="recovar_top_key")
    if relion_top == recovar_top:
        raise ValueError("comparison has no top-candidate discrepancy to classify")

    raw_details = comparison.get("cross_top_candidate_details")
    if not isinstance(raw_details, list) or len(raw_details) != 2:
        raise ValueError("comparison must report exactly two cross-top candidate details")
    details_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in raw_details:
        if not isinstance(row, dict):
            raise ValueError("cross-top candidate detail must be an object")
        key = _candidate_key(row.get("key"), field="cross_top_candidate_details.key")
        if key in details_by_key:
            raise ValueError("cross-top candidate details contain a duplicate key")
        details_by_key[key] = row
    expected_keys = {relion_top, recovar_top}
    if set(details_by_key) != expected_keys:
        raise ValueError("cross-top candidate details do not match the two engine winners")

    score_pairs: dict[str, dict[str, float]] = {}
    score_hex: dict[str, dict[str, str]] = {}
    exact_ties: dict[str, bool] = {}
    score_deltas: dict[str, float] = {}
    for engine in ("relion", "recovar"):
        relion_winner_score = _finite_score(details_by_key[relion_top], engine=engine, key=relion_top)
        recovar_winner_score = _finite_score(details_by_key[recovar_top], engine=engine, key=recovar_top)
        engine_top = relion_top if engine == "relion" else recovar_top
        own_score = relion_winner_score if engine == "relion" else recovar_winner_score
        other_score = recovar_winner_score if engine == "relion" else relion_winner_score
        if own_score < other_score:
            raise ValueError(
                f"{engine} reported top candidate {list(engine_top)} has a lower raw score "
                "than the other engine's top candidate"
            )
        score_pairs[engine] = {
            "at_relion_top": relion_winner_score,
            "at_recovar_top": recovar_winner_score,
        }
        score_hex[engine] = {
            "at_relion_top": relion_winner_score.hex(),
            "at_recovar_top": recovar_winner_score.hex(),
        }
        exact_ties[engine] = relion_winner_score == recovar_winner_score
        score_deltas[engine] = relion_winner_score - recovar_winner_score

    if all(exact_ties.values()):
        classification = "compact_candidate_tie_order"
    else:
        classification = "fine_score_arithmetic"

    return {
        "schema": "em_relion_recovar_fine_top_discrepancy_v1",
        "status": "pass",
        "metric_policy": (
            "exact raw pre-prior score equality only; localization diagnostic; "
            "no tolerance, correlation, FSC, or scorecard acceptance"
        ),
        "target_original_index": expected_original_index,
        "target_current_size": expected_current_size,
        "inertness_particle_count": expected_particle_count,
        "relion_top_key": list(relion_top),
        "recovar_top_key": list(recovar_top),
        "raw_pre_prior_scores": score_pairs,
        "raw_pre_prior_score_hex": score_hex,
        "raw_pre_prior_score_delta_at_relion_top_minus_at_recovar_top": score_deltas,
        "exact_raw_pre_prior_tie": exact_ties,
        "classification": classification,
        "classification_rule": (
            "compact_candidate_tie_order iff both engines have exact raw-score ties; "
            "fine_score_arithmetic otherwise"
        ),
        "scorecard_change_admissible": False,
    }


def build_report(
    comparison_path: Path,
    inertness_path: Path,
    *,
    expected_original_index: int,
    expected_current_size: int,
    expected_particle_count: int,
) -> dict[str, Any]:
    comparison = json.loads(comparison_path.read_text())
    inertness = json.loads(inertness_path.read_text())
    report = classify(
        comparison,
        inertness,
        expected_original_index=expected_original_index,
        expected_current_size=expected_current_size,
        expected_particle_count=expected_particle_count,
    )
    report["inputs"] = {
        "comparison_json": str(comparison_path.resolve()),
        "comparison_sha256": _sha256(comparison_path),
        "inertness_json": str(inertness_path.resolve()),
        "inertness_sha256": _sha256(inertness_path),
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-json", required=True, type=Path)
    parser.add_argument("--inertness-json", required=True, type=Path)
    parser.add_argument("--expected-original-index", required=True, type=int)
    parser.add_argument("--expected-current-size", required=True, type=int)
    parser.add_argument("--expected-particle-count", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite classification artifact: {args.output_json}")
    report = build_report(
        args.comparison_json,
        args.inertness_json,
        expected_original_index=args.expected_original_index,
        expected_current_size=args.expected_current_size,
        expected_particle_count=args.expected_particle_count,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
