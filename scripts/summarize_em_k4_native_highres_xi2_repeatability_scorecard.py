#!/usr/bin/env python3
"""Validate and render the fixed K=4 native high-resolution Xi2 scorecard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = (
    REPO_ROOT / "docs" / "math" / "em_k4_native_highres_xi2_repeatability_scorecard_v1.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT / "docs" / "math" / "em_k4_native_highres_xi2_repeatability_scorecard.md"
)
SCHEMA = "recovar.em_k4_native_highres_xi2_repeatability_scorecard.v1"
SUITE_ID = "k4-it2-fixed-a100-native-highres-xi2-observer-repeatability"
CLASSIFICATION = "deterministic_thread_highres_xi2_partial_replay_pair_exact"
FROZEN_DENOMINATOR = 15
EXPECTED_GATE_IDS = (
    "artifact_bytes_exact",
    "auxiliary_reduction_stream_markers_exact",
    "capture_headers_mpi2_openmp1_exact",
    "capture_validators_passed",
    "class_map_fsc_auc_at_least_threshold",
    "current_size_and_sampling_topology_exact",
    "dispatch_bytes_exact",
    "dispatch_row_count_exact",
    "highres_xi2_block_partial_markers_exact",
    "no_fatal_runtime_pattern",
    "particle_count_exact",
    "powerclass_stream_markers_exact",
    "softmask_block_partial_markers_exact",
    "target_state_exact",
    "thread_replay_markers_exact",
)
EXPECTED_EVIDENCE_SHA256 = {
    "analysis_result": "ee98144916d69ac618b8696176a8ec84d97d1c9d7c6dbfa9c3b0632235ecb900",
    "analyzer": "29a9ab99d0ff57619249d03dc11554fbd1fc24d89bff0a70a09164464d9ab003",
    "science_completion": "b78436fb5b88c493cfe077f0e1a9f3b54561bf9a5197652af7d582c276d02f79",
    "analysis_completion": "d1383debda55421ce1348f9facbc68c5c7b550d8af3cc62ac7cf7424da2c5d61",
    "science_manifest": "0af13e1ed0a109842b275e85976907265d89f7834d1a3a7fb158a1b87366f787",
    "static_manifest": "05f53af5ff3bd15b8e12c152d6a6fed222f7a333773c78c4d7e8ea157380adb5",
    "postterminal_audit": "27ac3362cde4ddfb324405817562ad1d567b4795f1424d3851e2cea68e966009",
    "binary": "01e5ee2bd1db2612e374a21060dd7b4b9bd72c3cccea86f9d0225102082849da",
    "build_completion": "0a05f99fd5d8fabfaebace7c88307caeb37ddce36b33a886cbf34fcf824d08f3",
    "build_manifest": "f968e86a12ef7b77442c8c8121840906cae2e8e34334a9ddf4303d0fa1ab629e",
}
EXPECTED_CLASS_FSC_AUC = (
    0.9999999961725112,
    0.999999980710069,
    0.9999999921474709,
    0.9999999852511,
)
EXPECTED_ARTIFACT_SHA256 = {
    "fine_score_sha256": "de5816046f21266c2f675c74cdbed799046bd3654e88d4e40210860ec2ede24b",
    "fine_operand_sha256": "a0ffefab0b30b602c5b18d4c73be70f510b32ee074132046f9bbc6e01d678a54",
    "bpref_sha256": "5d1c9f08eac3e46d9ecb6aa6b1040ec28ffed929cc128637f757309ca82d7f57",
}
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
        and contract.get("science_job_id") == 11993105
        and contract.get("science_state") == "COMPLETED"
        and contract.get("science_exit_code") == "0:0"
        and contract.get("same_gpu_uuid") == "GPU-f3e94635-d095-bea9-dbe3-26e91dd3ea27"
        and contract.get("postterminal_analysis_completed") is True
        and contract.get("accepted") is True
        and contract.get("stable_native_operand_localization_allowed") is True
        and contract.get("joint_posterior_bpref_map_parity_established") is False,
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
        tuple(case.get("id") for case in cases) == EXPECTED_GATE_IDS,
        "fixed gate identity/order changed",
    )
    for case in cases:
        _require(case.get("result") == "pass", f"{case['id']}: result changed")
        _require(case.get("checked") is True, f"{case['id']}: checked state changed")

    artifacts = scorecard.get("artifact_exactness")
    _require(
        isinstance(artifacts, dict)
        and artifacts.get("fine_score") is True
        and artifacts.get("fine_operand") is True
        and artifacts.get("bpref") is True
        and all(artifacts.get(key) == value for key, value in EXPECTED_ARTIFACT_SHA256.items()),
        "artifact exactness changed",
    )
    maps = scorecard.get("class_map_fsc_auc")
    _require(
        isinstance(maps, dict)
        and maps.get("metric") == "signed normalized non-DC FSC-AUC"
        and maps.get("threshold") == 0.999999
        and tuple(maps.get("values", ())) == EXPECTED_CLASS_FSC_AUC,
        "signed class-map FSC-AUC telemetry changed",
    )
    state = scorecard.get("particle_state_telemetry")
    _require(
        state
        == {
            "particle_count": 100000,
            "raw_row_order_exact": True,
            "hard_pose_class_shift_exact": True,
            "pmax_mismatch_count": 13,
            "support_count_mismatch_count": 15,
            "maximum_support_count_delta": 1,
            "full_particle_state_exact": False,
        },
        "particle-state telemetry changed",
    )
    excluded = scorecard.get("excluded_attempt")
    _require(
        isinstance(excluded, dict)
        and excluded.get("science_job_id") == 11992900
        and "not a 15-gate scientific pair" in excluded.get("reason", ""),
        "excluded-attempt provenance changed",
    )

    summary = {
        "pass": sum(case["result"] == "pass" for case in cases),
        "evaluated": len(cases),
    }
    _require(scorecard.get("summary") == summary, "recorded summary changed")
    _require(summary == {"pass": 15, "evaluated": 15}, "fixed result changed")
    return scorecard


def render_markdown(scorecard: dict) -> str:
    """Render the checked native high-resolution Xi2 scorecard."""

    summary = scorecard["summary"]
    maps = scorecard["class_map_fsc_auc"]
    state = scorecard["particle_state_telemetry"]
    lines = [
        "# K=4 native high-resolution Xi2 observer repeatability scorecard",
        "",
        "This fixed-denominator same-A100 panel measures native RELION observer",
        "repeatability after deterministic high-resolution Xi2 block finalization.",
        "It is non-scoring and cannot change the frozen cross-engine FSC-AUC scorecard.",
        "",
        f"Fixed gates: **{summary['pass']} / {scorecard['frozen_denominator']}**.",
        "",
        "| Checked | Fixed gate | Result |",
        "| --- | --- | ---: |",
    ]
    for case in scorecard["cases"]:
        lines.append(f"| [x] | `{case['id']}` | {case['result']} |")
    lines.extend(
        [
            "",
            f"Classification: `{scorecard['classification']}`.",
            "",
            (
                "Fine score, fine operand, and BPref artifacts were byte-exact. "
                "The selected target's hard pose, class, shift, Pmax, and support "
                "state were exact."
            ),
            "",
            (
                "Across all particles, hard pose/class/shift remained exact, but "
                f"{state['pmax_mismatch_count']} Pmax values and "
                f"{state['support_count_mismatch_count']} support counts differed; "
                f"the largest support-count delta was {state['maximum_support_count_delta']}. "
                "Therefore this admits stable native operand localization, not joint "
                "posterior/BPref/map parity."
            ),
            "",
            (
                "Signed normalized non-DC class-map FSC-AUC values were "
                + ", ".join(f"{value:.16f}" for value in maps["values"])
                + f" (threshold {maps['threshold']})."
            ),
            "",
            (
                "Slurm job 11992900 is excluded because its wrapper missed the "
                "predeclared powerClass marker and completed only arm A."
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
            "pixi run python scripts/summarize_em_k4_native_highres_xi2_repeatability_scorecard.py --check",
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
