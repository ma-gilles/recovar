#!/usr/bin/env python3
"""Validate and render the fixed K=4 native soft-mask observer scorecard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = (
    REPO_ROOT / "docs" / "math" / "em_k4_native_softmask_repeatability_scorecard_v1.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT / "docs" / "math" / "em_k4_native_softmask_repeatability_scorecard.md"
)
SCHEMA = "recovar.em_k4_native_softmask_repeatability_scorecard.v1"
SUITE_ID = "k4-it2-fixed-a100-native-softmask-observer-repeatability"
CLASSIFICATION = "native_softmask_observer_highres_xi2_residual"
FROZEN_DENOMINATOR = 14
EXPECTED_GATE_RESULTS = {
    "artifact_bytes_exact": "fail",
    "auxiliary_reduction_stream_markers_exact": "pass",
    "capture_headers_mpi2_openmp1_exact": "pass",
    "capture_validators_passed": "pass",
    "class_map_fsc_auc_at_least_threshold": "pass",
    "current_size_and_sampling_topology_exact": "pass",
    "dispatch_bytes_exact": "pass",
    "dispatch_row_count_exact": "pass",
    "no_fatal_runtime_pattern": "pass",
    "particle_count_exact": "pass",
    "powerclass_stream_markers_exact": "pass",
    "softmask_block_partial_markers_exact": "pass",
    "target_state_exact": "pass",
    "thread_replay_markers_exact": "pass",
}
EXPECTED_EVIDENCE_SHA256 = {
    "analysis_result": "2a84d68edfb4067cfbe653f70df8a3ca3373a263e27fbc34281553699c55e724",
    "residual_report": "1ae4303e5ca9713b387a465bccd02a3ea3b64aeeccb5b165348a871a3759fc15",
    "analyzer": "244680137df5bd73f679c669e510881fbcd5a03274cacc090f6adc67c474a7ce",
    "residual_analyzer": "576c4fdf0e66cd9d5e3e0d32ce0322d178c3ff6303b932a5cd6336f81d404ac9",
    "science_completion": "aff22e261093e5226a438c9e9f9baecfd2486efe5a443b2ac222196fb1afcb2e",
    "science_manifest": "b4028eff1e26f063cc5bdbc40546c1c40a7da015db847d44e316a5d779edbe20",
    "static_manifest": "8365d276f13567490ff04a9f43595bd874a911c1fe85939654e079fe397297d5",
    "postterminal_audit": "976606b3d421010200dab9e5c5e4779ad36e68280df87f26cec076b1368f34df",
}
EXPECTED_CLASS_FSC_AUC = (
    0.9999999942598741,
    0.9999999806617705,
    0.9999999895280383,
    0.999999986692646,
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_and_validate(path: Path) -> dict:
    """Load the checked scorecard and enforce its fixed evidence and denominator."""

    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(scorecard.get("classification") == CLASSIFICATION, "classification changed")
    _require(scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR, "denominator changed")
    _require(
        scorecard.get("scorecard_change_admissible") is False
        and scorecard.get("correlation_used") is False
        and scorecard.get("fsc_auc_evaluated") is True,
        "non-scoring metric policy changed",
    )

    contract = scorecard.get("acceptance_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("science_job_id") == 11990914
        and contract.get("science_state") == "COMPLETED"
        and contract.get("science_exit_code") == "0:0"
        and contract.get("postterminal_analysis_completed") is True
        and contract.get("accepted") is False,
        "terminal Slurm acceptance contract changed",
    )

    evidence = scorecard.get("evidence")
    _require(
        isinstance(evidence, dict) and set(evidence) == set(EXPECTED_EVIDENCE_SHA256),
        "fixed evidence identity changed",
    )
    for name, expected_digest in EXPECTED_EVIDENCE_SHA256.items():
        record = evidence[name]
        _require(
            isinstance(record.get("path"), str) and Path(record["path"]).is_absolute(),
            f"{name}: evidence path must be absolute",
        )
        digest = record.get("sha256")
        _require(
            isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"{name}: invalid SHA-256",
        )
        _require(digest == expected_digest, f"{name}: evidence SHA-256 changed")

    cases = scorecard.get("cases")
    _require(
        isinstance(cases, list) and len(cases) == FROZEN_DENOMINATOR,
        "cases do not preserve the frozen denominator",
    )
    _require(
        tuple(case.get("id") for case in cases) == tuple(EXPECTED_GATE_RESULTS),
        "fixed gate identity/order changed",
    )
    for case in cases:
        expected_result = EXPECTED_GATE_RESULTS[case["id"]]
        _require(case.get("result") == expected_result, f"{case['id']}: result changed")
        _require(case.get("checked") is (expected_result == "pass"), f"{case['id']}: checked state changed")

    artifacts = scorecard.get("artifact_exactness")
    _require(
        artifacts
        == {
            "fine_score": True,
            "fine_operand": False,
            "bpref": True,
            "fine_operand_differing_bytes": 3,
        },
        "artifact exactness changed",
    )
    maps = scorecard.get("class_map_fsc_auc")
    _require(
        isinstance(maps, dict)
        and maps.get("threshold") == 0.999999
        and tuple(maps.get("values", ())) == EXPECTED_CLASS_FSC_AUC,
        "signed class-map FSC-AUC telemetry changed",
    )
    boundary = scorecard.get("first_unequal_boundary")
    _require(
        isinstance(boundary, dict)
        and boundary.get("boundary") == "cuda_powerclass_highres_xi2_interblock_atomic"
        and boundary.get("sum_init_expression") == "op.highres_Xi2_img[img_id] / 2"
        and boundary.get("absolute_delta") == 2.2351741790771484e-08
        and boundary.get("ulp_delta") == 3
        and boundary.get("candidate_sum_init_mismatches") == 2
        and boundary.get("pixel_fields_compared") == 1520
        and boundary.get("pixel_field_mismatches") == 0
        and boundary.get("lane_partials_exact") is True
        and boundary.get("production_raw_diff2_exact") is True
        and boundary.get("replay_raw_diff2_exact") is True,
        "first unequal boundary telemetry changed",
    )

    summary = {
        "pass": sum(case["result"] == "pass" for case in cases),
        "evaluated": len(cases),
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(summary == {"pass": 13, "evaluated": 14}, "fixed result changed")
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the checked native soft-mask scorecard as a compact report."""

    summary = scorecard["summary"]
    boundary = scorecard["first_unequal_boundary"]
    maps = scorecard["class_map_fsc_auc"]
    lines = [
        "# K=4 native soft-mask observer repeatability scorecard",
        "",
        "This fixed-denominator same-A100 panel measures native RELION observer",
        "repeatability after deterministic soft-mask block finalization. It is",
        "non-scoring and cannot change the frozen cross-engine FSC-AUC scorecard.",
        "",
        f"Fixed gates: **{summary['pass']} / {scorecard['frozen_denominator']}**.",
        "",
        "| Checked | Fixed gate | Result |",
        "| --- | --- | ---: |",
    ]
    for case in scorecard["cases"]:
        check = "[x]" if case["checked"] else "[ ]"
        lines.append(f"| {check} | `{case['id']}` | {case['result']} |")
    lines.extend(
        [
            "",
            f"Classification: `{scorecard['classification']}`.",
            "",
            (
                "Fine score and BPref were byte-exact. Fine operand differed by "
                f"{scorecard['artifact_exactness']['fine_operand_differing_bytes']} bytes, "
                f"solely at `{boundary['sum_init_expression']}`: "
                f"{boundary['ulp_delta']} float32 ULP "
                f"({boundary['absolute_delta']}). All {boundary['pixel_fields_compared']} "
                "per-pixel fields, lane partials, and selected production/replay raw "
                "`diff2` values were exact."
            ),
            "",
            (
                "Signed normalized non-DC class-map FSC-AUC values were "
                + ", ".join(f"{value:.16f}" for value in maps["values"])
                + f" (threshold {maps['threshold']})."
            ),
            "",
            "Immutable evidence:",
            "",
        ]
    )
    for name, record in scorecard["evidence"].items():
        lines.append(f"- `{name}`: `{record['path']}` (SHA-256 `{record['sha256']}`)")
    lines.extend(
        [
            "",
            "To validate:",
            "",
            "```bash",
            "pixi run python scripts/summarize_em_k4_native_softmask_repeatability_scorecard.py --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    scorecard = load_and_validate(args.scorecard)
    rendered = render_markdown(scorecard)
    if args.check:
        target = DEFAULT_MARKDOWN if args.output is None else args.output
        if target.read_text() != rendered:
            raise SystemExit(f"{target} is stale; regenerate it")
    elif args.output is not None:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
