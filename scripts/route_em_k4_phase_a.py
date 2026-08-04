#!/usr/bin/env python3
"""Route an authoritative K=4 Phase-A result to its first unresolved boundary.

The Phase-A analyzer has two nested contracts:

* raw candidate support, common minimum, raw diff2, and centered score;
* the class-1 score path, including orientation/translation priors and the
  replayed combined score.

Passing either contract is intentionally not a joint K-class posterior claim.
This router makes that distinction explicit and prevents a class-1 raw match
from being used to authorize a scorecard change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.analyze_em_k4_raw_diff2_parity import (
    PASS_CLASSIFICATION,
    PASS_SCORE_CLASSIFICATION,
)
from scripts.analyze_em_k4_raw_diff2_parity import (
    SCHEMA as PHASE_A_SCHEMA,
)

SCHEMA = "recovar-k4-phase-a-causal-route-v1"
RAW_MISMATCH_ROUTE = "bounded_raw_operand_freeze"
SCORE_PATH_MISMATCH_ROUTE = "class1_prior_or_combined_score_followup"
JOINT_POSTERIOR_ROUTE = "multiclass_joint_posterior_capture"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_causal_route(phase_a: dict[str, Any]) -> dict[str, Any]:
    """Return the only admissible next boundary for a Phase-A report."""

    _require(phase_a.get("schema") == PHASE_A_SCHEMA, "Phase-A schema changed")
    _require(phase_a.get("status") == "complete", "Phase-A report is incomplete")
    _require(
        phase_a.get("classification_ready") is True,
        "Phase-A classification is not ready",
    )
    _require(
        phase_a.get("scorecard_change_admissible") is False,
        "Phase-A report must not authorize a scorecard change",
    )

    raw_classification = phase_a.get("classification")
    raw_exact = raw_classification == PASS_CLASSIFICATION
    _require(
        phase_a.get("accepted") is raw_exact,
        "raw classification and accepted flag disagree",
    )

    score_path = phase_a.get("score_path")
    _require(isinstance(score_path, dict), "Phase-A score_path is missing")
    score_classification = score_path.get("classification")
    score_exact = score_classification == PASS_SCORE_CLASSIFICATION
    _require(
        score_path.get("accepted") is score_exact,
        "score-path classification and accepted flag disagree",
    )

    support = phase_a.get("support")
    _require(isinstance(support, dict), "Phase-A support record is missing")
    support_exact = support.get("exact") is True

    if not raw_exact:
        route = RAW_MISMATCH_ROUTE
        first_unresolved_boundary = "class1_raw_diff2_or_its_operands"
        authorized_capture = (
            "fixed representative raw operand decomposition before any "
            "posterior, BPref, or map comparison"
        )
    elif not score_exact:
        route = SCORE_PATH_MISMATCH_ROUTE
        first_unresolved_boundary = "class1_prior_or_combined_score"
        authorized_capture = (
            "class1 direct joint orientation prior, translation prior, "
            "float32 operation order, and combined-score replay"
        )
    else:
        route = JOINT_POSTERIOR_ROUTE
        first_unresolved_boundary = (
            "remaining_classes_class_prior_joint_normalization_or_significance"
        )
        authorized_capture = (
            "all-class tuple/raw/prior tables followed by one joint "
            "class-pose logsumexp, posterior, and significance comparison"
        )

    return {
        "schema": SCHEMA,
        "status": "complete",
        "route": route,
        "first_unresolved_boundary": first_unresolved_boundary,
        "authorized_capture": authorized_capture,
        "phase_a": {
            "raw_classification": raw_classification,
            "raw_exact": raw_exact,
            "class1_score_path_classification": score_classification,
            "class1_score_path_exact": score_exact,
            "support_exact": support_exact,
        },
        "claims": {
            "class1_raw_boundary_resolved": raw_exact,
            "class1_score_path_resolved": raw_exact and score_exact,
            "all_class_tuple_and_score_boundary_resolved": False,
            "joint_class_pose_normalization_resolved": False,
            "joint_significance_resolved": False,
            "bpref_operand_boundary_resolved": False,
            "reduction_boundary_resolved": False,
            "map_parity_established": False,
        },
        "phase_b_raw_operand_freeze_required": not raw_exact,
        "scorecard_change_admissible": False,
        "metric_policy": (
            "causal routing only; no map-level acceptance, fitted transform, "
            "correlation, threshold relaxation, or scorecard change"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    phase_a = json.loads(args.phase_a_report.read_text())
    routed = build_causal_route(phase_a)
    routed["input"] = {
        "path": str(args.phase_a_report.resolve()),
        "sha256": _sha256(args.phase_a_report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(routed, indent=2, sort_keys=True) + "\n")
    print(json.dumps(routed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
