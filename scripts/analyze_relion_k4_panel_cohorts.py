#!/usr/bin/env python3
"""Stratify the fixed K=4 three-way score panel by predeclared outcome cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THREEWAY_SCHEMA = "recovar.k4_iter10_panel12_threeway_fine_score.v2"
REPEATABILITY_SCHEMA = "relion.k4_iter10_panel12_capture_repeatability.v1"
REPORT_SCHEMA = "recovar.k4_iter10_panel12_cohort_calibration.v1"
EXPECTED_COHORT_COUNTS = {
    "corrected_by_relion_cuda": 4,
    "introduced_by_relion_cuda": 4,
    "persistent_class_mismatch": 4,
}
FAMILY_FIELDS = {
    "data": ("data_score_residual", "centered_raw_diff2_repeatability"),
    "combined": ("combined_score_residual", "centered_combined_repeatability"),
}
NO_UNIFORM_REDUCTION = (
    "relion_cuda_preprocessing_does_not_uniformly_reduce_relion_fine_score_residual"
)
WITHIN_FLOOR = "relion_cuda_preprocessing_reduction_is_within_capture_repeatability_floor"
BEYOND_FLOOR = "relion_cuda_preprocessing_reduces_residual_beyond_capture_repeatability_floor"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_by_identity(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    result = {}
    for row in rows:
        identity = int(row["zero_based_identity_row"])
        _require(identity not in result, f"{label} contains duplicate identity {identity}")
        result[identity] = row
    return result


def _family_summary(
    threeway_rows: list[dict[str, Any]],
    repeatability_rows: list[dict[str, Any]],
    *,
    family: str,
) -> dict[str, float | int | bool]:
    score_field, repeatability_field = FAMILY_FIELDS[family]
    host_energy = math.fsum(
        float(row[score_field]["host_numpy"]["residual_energy"]) for row in threeway_rows
    )
    candidate_energy = math.fsum(
        float(row[score_field]["relion_cuda"]["residual_energy"]) for row in threeway_rows
    )
    floor_energy = math.fsum(
        float(row[repeatability_field]["residual_energy"]) for row in repeatability_rows
    )
    _require(floor_energy > 0, f"{family} cohort repeatability floor must be positive")
    improvement = host_energy - candidate_energy
    return {
        "candidate_count": sum(
            int(row[score_field]["host_numpy"]["candidate_count"]) for row in threeway_rows
        ),
        "host_numpy_residual_energy": host_energy,
        "relion_cuda_residual_energy": candidate_energy,
        "improvement_energy": improvement,
        "capture_repeatability_residual_energy": floor_energy,
        "improvement_to_repeatability_energy_ratio": improvement / floor_energy,
        "improvement_positive": improvement > 0,
        "improvement_exceeds_capture_repeatability_floor": improvement > floor_energy,
    }


def _classify_families(families: dict[str, dict[str, Any]]) -> str:
    improvements = [bool(families[family]["improvement_positive"]) for family in FAMILY_FIELDS]
    beyond = [
        bool(families[family]["improvement_exceeds_capture_repeatability_floor"])
        for family in FAMILY_FIELDS
    ]
    if all(beyond):
        return BEYOND_FLOOR
    if all(improvements):
        return WITHIN_FLOOR
    return NO_UNIFORM_REDUCTION


def analyze(threeway: dict[str, Any], repeatability: dict[str, Any]) -> dict[str, Any]:
    _require(
        threeway.get("schema") == THREEWAY_SCHEMA and threeway.get("status") == "complete",
        "three-way report is incomplete or has the wrong schema",
    )
    _require(
        repeatability.get("schema") == REPEATABILITY_SCHEMA
        and repeatability.get("status") == "complete",
        "capture-repeatability report is incomplete or has the wrong schema",
    )
    _require(
        threeway.get("scorecard_change_admissible") is False
        and repeatability.get("scorecard_change_admissible") is False,
        "input diagnostic unexpectedly permits a scorecard change",
    )
    _require(
        threeway["quality_metric_policy"]["correlation_computed"] is False,
        "three-way diagnostic computed correlation",
    )
    scope = threeway.get("scope", {})
    repeatability_scope = repeatability.get("scope", {})
    _require(
        scope.get("physical_iteration") == 10
        and scope.get("class_one_based") == 2
        and scope.get("current_size") == 74
        and scope.get("target_count") == 12
        and scope.get("winner_evaluable_target_count") == 12
        and scope.get("host_winner_matches_relion_count") == 12
        and scope.get("relion_cuda_winner_matches_relion_count") == 12,
        "three-way panel scope or winner closure changed",
    )
    _require(
        repeatability_scope.get("physical_iteration") == 10
        and repeatability_scope.get("class_one_based") == 2
        and repeatability_scope.get("target_count") == 12
        and repeatability_scope.get("winners_exact_all") is True,
        "capture-repeatability panel scope or winner closure changed",
    )

    threeway_rows = threeway.get("targets")
    repeatability_rows = repeatability.get("targets")
    _require(
        isinstance(threeway_rows, list)
        and isinstance(repeatability_rows, list)
        and len(threeway_rows) == len(repeatability_rows) == 12,
        "panel target rows changed",
    )
    threeway_by_identity = _unique_by_identity(threeway_rows, "three-way report")
    repeatability_by_identity = _unique_by_identity(
        repeatability_rows, "capture-repeatability report"
    )
    _require(
        set(threeway_by_identity) == set(repeatability_by_identity),
        "panel identity sets differ",
    )

    grouped_threeway: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_repeatability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for identity in sorted(threeway_by_identity):
        threeway_row = threeway_by_identity[identity]
        repeatability_row = repeatability_by_identity[identity]
        cohort = threeway_row["cohort"]
        _require(
            cohort == repeatability_row["cohort"]
            and threeway_row["rlnImageName"] == repeatability_row["rlnImageName"],
            f"identity {identity}: cohort or image identity changed",
        )
        candidate_count = int(threeway_row["active_candidate_count"])
        _require(
            candidate_count > 0
            and candidate_count == int(repeatability_row["active_candidate_count"]),
            f"identity {identity}: active candidate count changed",
        )
        winner = threeway_row["winner"]
        _require(
            winner["winner_defined"]
            and winner["host_matches_relion"]
            and winner["recovar_relion_cuda_matches_relion"],
            f"identity {identity}: three-way winner closure changed",
        )
        for family, (score_field, repeatability_field) in FAMILY_FIELDS.items():
            for backend in ("host_numpy", "relion_cuda"):
                _require(
                    int(threeway_row[score_field][backend]["candidate_count"])
                    == candidate_count,
                    f"identity {identity}: {family} {backend} candidate count changed",
                )
            _require(
                int(repeatability_row[repeatability_field]["candidate_count"])
                == candidate_count,
                f"identity {identity}: {family} repeatability candidate count changed",
            )
        grouped_threeway[cohort].append(threeway_row)
        grouped_repeatability[cohort].append(repeatability_row)

    observed_counts = Counter({cohort: len(rows) for cohort, rows in grouped_threeway.items()})
    _require(
        dict(observed_counts) == EXPECTED_COHORT_COUNTS,
        f"predeclared cohort counts changed: {dict(observed_counts)}",
    )

    cohorts = {}
    for cohort in sorted(EXPECTED_COHORT_COUNTS):
        families = {
            family: _family_summary(
                grouped_threeway[cohort],
                grouped_repeatability[cohort],
                family=family,
            )
            for family in FAMILY_FIELDS
        }
        cohorts[cohort] = {
            "target_count": len(grouped_threeway[cohort]),
            "classification": _classify_families(families),
            "families": families,
            "identities": [
                {
                    "zero_based_identity_row": int(row["zero_based_identity_row"]),
                    "rlnImageName": row["rlnImageName"],
                }
                for row in grouped_threeway[cohort]
            ],
        }

    cohort_classifications = [row["classification"] for row in cohorts.values()]
    if BEYOND_FLOOR in cohort_classifications:
        classification = "one_or_more_cohorts_reduce_beyond_capture_repeatability_floor"
    elif all(value == WITHIN_FLOOR for value in cohort_classifications):
        classification = "all_cohorts_reduce_within_capture_repeatability_floor"
    else:
        classification = "heterogeneous_cohort_effect_without_robust_reduction"
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "classification": classification,
        "scorecard_change_admissible": False,
        "scope": {
            "physical_iteration": 10,
            "class_one_based": 2,
            "current_size": 74,
            "target_count": 12,
            "cohort_counts": EXPECTED_COHORT_COUNTS,
        },
        "aggregate_classification": threeway["classification"],
        "cohorts": cohorts,
        "quality_metric_policy": {
            "map_gate": "shellwise FSC/FSC-AUC only",
            "correlation_computed": False,
            "cohort_metrics_are_diagnostic": True,
        },
        "next_step": (
            "Do not change the preprocessing default. The persistent and introduced "
            "cohorts require an upstream score-arithmetic discriminator; a new GPU run "
            "is justified only by a fixed cohort whose improvement exceeds its own floor."
        ),
    }


def _clean_repo_head(repo: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain=v1"], text=True)
    _require(not status, "analyzer repository is dirty")
    return head


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--threeway-json", required=True, type=Path)
    parser.add_argument("--capture-repeatability-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    threeway = json.loads(args.threeway_json.read_text())
    repeatability = json.loads(args.capture_repeatability_json.read_text())
    report = analyze(threeway, repeatability)
    report["inputs"] = {
        "threeway": {
            "path": str(args.threeway_json.resolve()),
            "sha256": _sha256(args.threeway_json),
        },
        "capture_repeatability": {
            "path": str(args.capture_repeatability_json.resolve()),
            "sha256": _sha256(args.capture_repeatability_json),
        },
        "analyzer_repo_head": _clean_repo_head(args.repo),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "aggregate_classification": report["aggregate_classification"],
                "cohort_classifications": {
                    cohort: row["classification"] for cohort, row in report["cohorts"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
