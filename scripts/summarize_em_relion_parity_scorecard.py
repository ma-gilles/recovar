#!/usr/bin/env python
"""Validate and render the frozen RECOVAR/RELION parity scorecard.

The scorecard is deliberately checked into the repository.  A suite version
has a fixed denominator and fixed case definitions; adding or changing a case
requires a new suite version rather than silently moving the current score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs" / "math" / "em_relion_parity_scorecard_v1.json"
VALID_RESULTS = {"pass", "fail", "not_run"}
REQUIRED_DEFINITION_FIELDS = {
    "contrast_std",
    "dataset_params_option",
    "grid",
    "image_offset_n_std",
    "n_images",
    "noise_level",
    "noise_model",
    "noise_scale_std",
    "pdb_bfactor",
    "percent_outliers",
    "put_extra_particles",
    "seed",
    "volume_radius",
}


def frozen_case_definitions_sha256(cases: list[dict]) -> str:
    """Hash only the immutable suite identity and fixture definitions."""

    manifest = [
        {
            "id": case.get("id"),
            "name": case.get("name"),
            "definition": case.get("definition"),
        }
        for case in cases
    ]
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_and_validate(path: Path) -> dict:
    scorecard = json.loads(path.read_text())
    if scorecard.get("schema") != "recovar.em_relion_parity_scorecard.v1":
        raise ValueError("unsupported scorecard schema")
    if scorecard.get("suite_version") != 1:
        raise ValueError("v1 scorecard must have suite_version=1")

    cases = scorecard.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    denominator = scorecard.get("frozen_denominator")
    if denominator != len(cases):
        raise ValueError(f"frozen_denominator={denominator} but found {len(cases)} cases")

    expected_ids = [f"k1-{index:02d}" for index in range(1, denominator + 1)]
    actual_ids = [case.get("id") for case in cases]
    if actual_ids != expected_ids:
        raise ValueError("case IDs must be ordered, contiguous, and frozen")
    names = [case.get("name") for case in cases]
    if len(set(names)) != len(names) or not all(isinstance(name, str) and name for name in names):
        raise ValueError("case names must be non-empty and unique")

    calculated_definition_sha256 = frozen_case_definitions_sha256(cases)
    recorded_definition_sha256 = scorecard.get("frozen_case_definitions_sha256")
    if recorded_definition_sha256 != calculated_definition_sha256:
        raise ValueError(
            "frozen case definitions changed without a suite-version change: "
            f"recorded={recorded_definition_sha256!r} calculated={calculated_definition_sha256}"
        )

    for case in cases:
        result = case.get("result")
        if result not in VALID_RESULTS:
            raise ValueError(f"{case['id']}: invalid result {result!r}")
        if case.get("intermediate_result") not in VALID_RESULTS:
            raise ValueError(f"{case['id']}: invalid intermediate_result")
        definition = case.get("definition")
        if not isinstance(definition, dict) or set(definition) != REQUIRED_DEFINITION_FIELDS:
            raise ValueError(f"{case['id']}: incomplete or expanded frozen definition")
        if not case.get("source_head") or not case.get("jobs"):
            raise ValueError(f"{case['id']}: missing immutable source/job evidence")

    calculated = Counter(case["result"] for case in cases)
    recorded = scorecard.get("current_snapshot", {}).get("counts", {})
    expected_counts = {status: calculated.get(status, 0) for status in ("pass", "fail", "not_run")}
    if recorded != expected_counts:
        raise ValueError(f"recorded counts {recorded} do not match cases {expected_counts}")

    history = scorecard.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("history must contain at least the current snapshot")
    history_ids = [snapshot.get("id") for snapshot in history]
    if len(set(history_ids)) != len(history_ids):
        raise ValueError("history snapshot IDs must be unique")
    for snapshot in history:
        counts = snapshot.get("counts")
        if not isinstance(counts, dict) or set(counts) != VALID_RESULTS:
            raise ValueError(f"{snapshot.get('id')}: invalid history counts")
        if any(not isinstance(value, int) or value < 0 for value in counts.values()):
            raise ValueError(f"{snapshot.get('id')}: history counts must be non-negative integers")
        if sum(counts.values()) != denominator:
            raise ValueError(f"{snapshot.get('id')}: history counts do not preserve frozen denominator")
        if not snapshot.get("source_heads") or not snapshot.get("evidence_sha256"):
            raise ValueError(f"{snapshot.get('id')}: missing immutable history evidence")
    if history[-1]["id"] != scorecard["current_snapshot"]["id"]:
        raise ValueError("last history row must be the current snapshot")
    if history[-1]["counts"] != recorded:
        raise ValueError("last history counts must match current snapshot")
    return scorecard


def render_markdown(scorecard: dict) -> str:
    cases = scorecard["cases"]
    counts = scorecard["current_snapshot"]["counts"]
    passed = counts["pass"]
    total = scorecard["frozen_denominator"]
    evaluated = passed + counts["fail"]
    intermediate_passed = sum(case["intermediate_result"] == "pass" for case in cases)
    source = scorecard["current_snapshot"]["source_ledger"]

    lines = [
        "# RECOVAR / RELION EM Parity Scorecard",
        "",
        f"**K=1 fixed-suite score: {passed} / {total} passing "
        f"({evaluated} / {total} evaluated; {intermediate_passed} / {total} intermediate-topology passes).**",
        "",
        f"Suite: `{scorecard['suite_id']}` (version {scorecard['suite_version']}; denominator frozen at {total}).",
        f"Frozen case-definition SHA-256: `{scorecard['frozen_case_definitions_sha256']}`.",
        "",
        "A checked box means the complete autonomous FSC/FSC-AUC trajectory contract passed. "
        "Unchecked cases remain in the denominator. New diagnostics do not enter this suite; changing "
        "the case set or scientific definitions requires a new suite version.",
        "",
        "Acceptance uses shellwise FSC and normalized FSC-AUC, exact schedule/topology, convergence/finalization "
        "semantics, same-physical-GPU RELION/RECOVAR pairs, grid correction unset/off, and no forced K-class-like "
        "finalization. Correlation is not computed or gated.",
        "",
        f"Evidence snapshot: `{source['schema']}`, generated `{source['generated_utc']}`, JSON SHA-256 "
        f"`{source['sha256']}`.",
        "",
        "| Done | Case | Fixture | Trajectory | Topology | Final cross-engine FSC-AUC | Final GT delta | Jobs |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for case in cases:
        checked = "[x]" if case["result"] == "pass" else "[ ]"
        cross = case.get("final_cross_engine_fsc_auc")
        delta = case.get("final_gt_fsc_auc_delta")
        cross_text = "—" if cross is None else f"{cross:.9f}"
        delta_text = "—" if delta is None else f"{delta:+.9f}"
        jobs = case["jobs"]
        job_text = f"science {jobs['science']}; trajectory {jobs['trajectory']}; intermediate {jobs['intermediate']}"
        lines.append(
            f"| {checked} | `{case['id']}` | `{case['name']}` | {case['result']} | "
            f"{case['intermediate_result']} | {cross_text} | {delta_text} | {job_text} |"
        )

    lines += [
        "",
        "## Progress history",
        "",
        "| Snapshot | Date (UTC) | Commit boundary | Passed | Failed | Not evaluated/error |",
        "|---|---|---|---:|---:|---:|",
    ]
    for snapshot in scorecard["history"]:
        snapshot_counts = snapshot["counts"]
        heads = ", ".join(f"`{head[:12]}`" for head in snapshot["source_heads"])
        lines.append(
            f"| `{snapshot['id']}` | {snapshot['recorded_utc']} | {heads} | "
            f"{snapshot_counts['pass']} | {snapshot_counts['fail']} | {snapshot_counts['not_run']} |"
        )
    lines += [
        "",
        "Generate this PR-ready table with:",
        "",
        "```bash",
        "pixi run python scripts/summarize_em_relion_parity_scorecard.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--check", type=Path, help="fail if this generated Markdown file is stale")
    args = parser.parse_args()

    rendered = render_markdown(load_and_validate(args.scorecard))
    if args.check is not None:
        if args.check.read_text() != rendered:
            raise SystemExit(f"stale generated scorecard: {args.check}")
        print(f"scorecard valid and current: {args.check}")
        return 0
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
