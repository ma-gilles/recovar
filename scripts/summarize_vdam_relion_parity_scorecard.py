#!/usr/bin/env python
"""Validate and render the frozen RELION InitialModel/VDAM parity suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs/math/vdam_relion_parity_scorecard_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/math/vdam_relion_parity_scorecard.md"
SCHEMA = "recovar.vdam_relion_parity_scorecard.v1"
SUITE_ID = "vdam-k1-gui-grid0-fixed12"
FROZEN_DENOMINATOR = 12
FROZEN_CASE_DEFINITIONS_SHA256 = "1a37a1b360b022d60eefdd0481eb0784d4a0e98a4d92066199625ceaf6d11dd1"
SOURCE_FIXTURE_MANIFEST = "docs/math/em_relion_parity_fixture_manifest_v2.json"
SOURCE_FIXTURE_MANIFEST_SHA256 = "422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee"
METRIC_POLICY = (
    "Signed shellwise FSC and normalized non-DC FSC-AUC for maps; exact or distributional comparisons for state. "
    "Correlation is not computed or gated."
)
VALID_RESULTS = frozenset({"pass", "fail", "not_run"})
REQUIRED_CHECKPOINTS = (0, 1, 2, 4, 8)
REQUIRED_DEFINITION_FIELDS = frozenset(
    {
        "source_em_case_id",
        "nr_classes",
        "nr_iter",
        "random_seed",
        "tau2_fudge",
        "healpix_order",
        "oversampling",
        "offset_range_px",
        "offset_step_px",
        "padding_factor",
    }
)
ACCEPTANCE_CONTRACT = {
    "cross_engine_fsc_auc_min": 0.999,
    "recovar_minus_relion_gt_fsc_auc_min": -0.002,
    "required_checkpoints": list(REQUIRED_CHECKPOINTS),
    "exact_schedule": True,
    "exact_artifact_topology": True,
    "same_physical_gpu_per_pair": True,
    "correlation_used": False,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_case_definitions_sha256(cases: list[dict[str, Any]]) -> str:
    definitions = [
        {"id": case.get("id"), "name": case.get("name"), "definition": case.get("definition")}
        for case in cases
    ]
    canonical = json.dumps(definitions, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _validate_source_fixture_manifest(scorecard: dict[str, Any]) -> set[str]:
    source = scorecard.get("source_fixture_manifest")
    _require(
        source == {"path": SOURCE_FIXTURE_MANIFEST, "sha256": SOURCE_FIXTURE_MANIFEST_SHA256},
        "source fixture manifest identity changed",
    )
    path = REPO_ROOT / SOURCE_FIXTURE_MANIFEST
    _require(path.is_file(), f"missing source fixture manifest: {path}")
    _require(sha256_file(path) == SOURCE_FIXTURE_MANIFEST_SHA256, "source fixture manifest bytes changed")
    manifest = json.loads(path.read_text())
    _require(manifest.get("frozen_denominator") == 34, "source EM fixture denominator changed")
    cases = manifest.get("cases")
    _require(isinstance(cases, list), "source EM fixture cases must be a list")
    return {str(case.get("id")) for case in cases}


def _validate_evaluated_case(case: dict[str, Any], contract: dict[str, Any]) -> None:
    case_id = case["id"]
    checkpoints = case.get("checkpoint_results")
    expected_keys = {str(value) for value in REQUIRED_CHECKPOINTS}
    _require(isinstance(checkpoints, dict) and set(checkpoints) == expected_keys, f"{case_id}: checkpoint set changed")
    _require(all(value in {"pass", "fail"} for value in checkpoints.values()), f"{case_id}: invalid checkpoint result")

    evidence = case.get("evidence")
    _require(isinstance(evidence, dict), f"{case_id}: evaluated case needs evidence")
    _require(GIT_SHA_RE.fullmatch(str(evidence.get("source_head", ""))) is not None, f"{case_id}: invalid source head")
    _require(SHA256_RE.fullmatch(str(evidence.get("report_sha256", ""))) is not None, f"{case_id}: invalid report digest")
    for name in ("recovar_job", "relion_job", "audit_job"):
        _require(bool(str(evidence.get(name, "")).strip()), f"{case_id}: missing {name}")
    _require(evidence.get("same_physical_gpu") is True, f"{case_id}: comparison was not on one physical GPU")
    _require(evidence.get("correlation_used") is False, f"{case_id}: correlation cannot gate map quality")
    _require(evidence.get("exact_schedule") is True, f"{case_id}: schedule mismatch")
    _require(evidence.get("exact_artifact_topology") is True, f"{case_id}: artifact topology mismatch")

    cross = evidence.get("final_cross_engine_fsc_auc")
    gt_delta = evidence.get("final_gt_fsc_auc_delta")
    _require(isinstance(cross, (int, float)), f"{case_id}: missing final cross-engine FSC-AUC")
    _require(isinstance(gt_delta, (int, float)), f"{case_id}: missing final GT FSC-AUC delta")
    gates_pass = (
        all(value == "pass" for value in checkpoints.values())
        and float(cross) >= float(contract["cross_engine_fsc_auc_min"])
        and float(gt_delta) >= float(contract["recovar_minus_relion_gt_fsc_auc_min"])
    )
    expected_result = "pass" if gates_pass else "fail"
    _require(case.get("result") == expected_result, f"{case_id}: result does not replay its fixed gates")


def load_and_validate(path: Path = DEFAULT_SCORECARD) -> dict[str, Any]:
    scorecard = json.loads(Path(path).read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported VDAM scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "VDAM suite identity changed")
    _require(scorecard.get("suite_version") == 1, "VDAM v1 suite version changed")
    _require(scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR, "VDAM frozen denominator changed")
    _require(scorecard.get("acceptance_contract") == ACCEPTANCE_CONTRACT, "VDAM acceptance contract changed")
    _require(scorecard.get("metric_policy") == METRIC_POLICY, "VDAM metric policy changed")

    source_case_ids = _validate_source_fixture_manifest(scorecard)
    cases = scorecard.get("cases")
    _require(isinstance(cases, list) and len(cases) == FROZEN_DENOMINATOR, "VDAM case denominator mismatch")
    _require(
        [case.get("id") for case in cases] == [f"vdam-{index:02d}" for index in range(1, 13)],
        "VDAM case IDs must remain ordered and contiguous",
    )
    names = [case.get("name") for case in cases]
    _require(all(isinstance(name, str) and name for name in names) and len(set(names)) == len(names), "case names must be unique")
    calculated_digest = frozen_case_definitions_sha256(cases)
    _require(scorecard.get("frozen_case_definitions_sha256") == FROZEN_CASE_DEFINITIONS_SHA256, "recorded definition digest changed")
    _require(calculated_digest == FROZEN_CASE_DEFINITIONS_SHA256, "frozen VDAM case definitions changed")

    source_refs: list[str] = []
    for case in cases:
        case_id = case["id"]
        definition = case.get("definition")
        _require(isinstance(definition, dict) and set(definition) == REQUIRED_DEFINITION_FIELDS, f"{case_id}: definition shape changed")
        source_id = definition["source_em_case_id"]
        _require(source_id in source_case_ids, f"{case_id}: unknown source EM fixture {source_id!r}")
        source_refs.append(source_id)
        _require(definition["nr_classes"] == 1 and definition["nr_iter"] == 8, f"{case_id}: fixed K/iteration contract changed")
        _require(definition["padding_factor"] == 1, f"{case_id}: InitialModel must retain pad=1")
        _require(case.get("result") in VALID_RESULTS, f"{case_id}: invalid result")
        if case["result"] == "not_run":
            _require(case.get("checkpoint_results") == {}, f"{case_id}: not-run case has checkpoint claims")
            _require(case.get("evidence") is None, f"{case_id}: not-run case has evidence")
        else:
            _validate_evaluated_case(case, ACCEPTANCE_CONTRACT)
    _require(len(source_refs) == len(set(source_refs)), "source EM fixtures must not be reused within v1")

    counts = Counter(case["result"] for case in cases)
    expected_counts = {name: counts.get(name, 0) for name in ("pass", "fail", "not_run")}
    snapshot = scorecard.get("current_snapshot")
    _require(isinstance(snapshot, dict) and snapshot.get("counts") == expected_counts, "current counts do not replay cases")
    history = scorecard.get("history")
    _require(isinstance(history, list) and history, "history must not be empty")
    _require(history[-1].get("id") == snapshot.get("id"), "latest history ID differs from current snapshot")
    _require(history[-1].get("counts") == expected_counts, "latest history counts differ from current snapshot")
    history_ids: set[str] = set()
    for index, row in enumerate(history):
        row_id = row.get("id")
        _require(isinstance(row_id, str) and row_id not in history_ids, "history IDs must be unique")
        history_ids.add(row_id)
        row_counts = row.get("counts")
        _require(isinstance(row_counts, dict) and set(row_counts) == VALID_RESULTS, f"{row_id}: invalid history counts")
        _require(sum(row_counts.values()) == FROZEN_DENOMINATOR, f"{row_id}: history denominator changed")
        if index > 0:
            evidence = row.get("evidence")
            _require(isinstance(evidence, dict), f"{row_id}: missing immutable evidence")
            _require(SHA256_RE.fullmatch(str(evidence.get("ledger_sha256", ""))) is not None, f"{row_id}: bad ledger digest")
            _require(GIT_SHA_RE.fullmatch(str(evidence.get("source_head", ""))) is not None, f"{row_id}: bad source head")
    return scorecard


def render_markdown(scorecard: dict[str, Any]) -> str:
    counts = scorecard["current_snapshot"]["counts"]
    evaluated = counts["pass"] + counts["fail"]
    contract = scorecard["acceptance_contract"]
    lines = [
        "# RECOVAR / RELION VDAM InitialModel parity scorecard",
        "",
        f"**Fixed-suite score: {counts['pass']} / {FROZEN_DENOMINATOR} passing "
        f"({evaluated} / {FROZEN_DENOMINATOR} evaluated).**",
        "",
        f"Suite: `{SUITE_ID}` (version 1; denominator frozen at {FROZEN_DENOMINATOR}).",
        f"Frozen case-definition SHA-256: `{FROZEN_CASE_DEFINITIONS_SHA256}`.",
        f"Source fixture manifest SHA-256: `{SOURCE_FIXTURE_MANIFEST_SHA256}`.",
        "",
        "A pass requires every fixed checkpoint to preserve the schedule and artifact topology, "
        f"cross-engine FSC-AUC >= `{contract['cross_engine_fsc_auc_min']}`, and RECOVAR-minus-RELION "
        f"GT FSC-AUC >= `{contract['recovar_minus_relion_gt_fsc_auc_min']}` on the same physical GPU.",
        "Map correlation is not computed or gated. Historical correlation-only runs are non-scoring.",
        "",
        "| Case | Reused fixed EM fixture | Result | Checkpoints | Evidence |",
        "|---|---|---:|---|---|",
    ]
    for case in scorecard["cases"]:
        mark = {"pass": "PASS", "fail": "FAIL", "not_run": "—"}[case["result"]]
        checkpoints = case["checkpoint_results"]
        checkpoint_text = ", ".join(f"{key}:{value}" for key, value in checkpoints.items()) or "not run"
        evidence = case["evidence"]
        evidence_text = "—" if evidence is None else str(evidence["report_sha256"])
        lines.append(
            f"| `{case['id']}` {case['name']} | `{case['definition']['source_em_case_id']}` | "
            f"{mark} | {checkpoint_text} | `{evidence_text}` |"
        )
    lines.extend(
        [
            "",
            "## Fixed checkpoints",
            "",
            "The v1 trajectory checkpoints are iterations `0`, `1`, `2`, `4`, and `8`. "
            "Iteration 0 covers bootstrap/reference initialization; later checkpoints cover the complete "
            "VDAM schedule, E-step state, pseudo-halfset M-step, and written maps.",
            "",
            "Regenerate and validate this page with:",
            "",
            "```bash",
            "pixi run python scripts/summarize_vdam_relion_parity_scorecard.py --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_markdown(load_and_validate(args.scorecard))
    if args.check:
        if not args.output.is_file() or args.output.read_text() != rendered:
            raise SystemExit(f"stale generated scorecard: {args.output}")
    else:
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
