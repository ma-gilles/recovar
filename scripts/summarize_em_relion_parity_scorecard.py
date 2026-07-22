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
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs" / "math" / "em_relion_parity_scorecard_v1.json"
DEFAULT_FIXTURE_MANIFEST = REPO_ROOT / "docs" / "math" / "em_relion_parity_fixture_manifest_v2.json"
V1_SUITE_ID = "k1-gui-grid0-local-highshell-full34"
V2_FIXTURE_SUITE_ID = f"{V1_SUITE_ID}-artifact-pinned-v2"
V1_FROZEN_DENOMINATOR = 34
V1_FROZEN_CASE_DEFINITIONS_SHA256 = "9e3f2cb7192eb2cbf8a50181cf47de8562adfb98734bab05a736fb7d4d404fc1"
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
REQUIRED_FIXTURE_FILES = {
    "ctf.pkl",
    "generation_config.json",
    "particles.star",
    "poses.pkl",
    "reference_gt.mrc",
    "reference_gt_relion.mrc",
    "reference_init.mrc",
    "reference_init_relion.mrc",
    "simulation_info.pkl",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if scorecard.get("suite_id") != V1_SUITE_ID:
        raise ValueError(f"v1 suite_id must remain {V1_SUITE_ID!r}")

    cases = scorecard.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    denominator = scorecard.get("frozen_denominator")
    if denominator != V1_FROZEN_DENOMINATOR:
        raise ValueError(f"v1 frozen_denominator must remain {V1_FROZEN_DENOMINATOR}")
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
    if recorded_definition_sha256 != V1_FROZEN_CASE_DEFINITIONS_SHA256:
        raise ValueError(
            "v1 frozen case-definition digest changed without a suite-version change: "
            f"expected={V1_FROZEN_CASE_DEFINITIONS_SHA256} recorded={recorded_definition_sha256!r}"
        )
    if calculated_definition_sha256 != V1_FROZEN_CASE_DEFINITIONS_SHA256:
        raise ValueError(
            "frozen case definitions changed without a suite-version change: "
            f"expected={V1_FROZEN_CASE_DEFINITIONS_SHA256} calculated={calculated_definition_sha256}"
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

    case_ids = set(actual_ids)
    replicates = scorecard.get("replicate_diagnostics", [])
    if not isinstance(replicates, list):
        raise ValueError("replicate_diagnostics must be a list")
    for replicate in replicates:
        case_id = replicate.get("case_id")
        if case_id not in case_ids:
            raise ValueError(f"unknown replicate case ID: {case_id!r}")
        if replicate.get("scoring") is not False:
            raise ValueError(f"{case_id}: regenerated replicate must be explicitly non-scoring")
        if replicate.get("trajectory_result") not in {"pass", "fail"}:
            raise ValueError(f"{case_id}: invalid replicate trajectory result")
        if replicate.get("intermediate_result") not in {"pass", "fail"}:
            raise ValueError(f"{case_id}: invalid replicate intermediate result")
        for field in ("particle_stack_sha256", "fixed_fixture_particle_stack_sha256"):
            value = replicate.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{case_id}: invalid replicate {field}")
        if replicate["particle_stack_sha256"] == replicate["fixed_fixture_particle_stack_sha256"]:
            raise ValueError(f"{case_id}: replicate unexpectedly matches the fixed fixture bytes")
        jobs = replicate.get("jobs")
        if not isinstance(jobs, dict) or set(jobs) != {"science", "audit"}:
            raise ValueError(f"{case_id}: replicate jobs must identify science and audit")
    return scorecard


def load_and_validate_fixture_manifest(path: Path, scorecard: dict) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "recovar.em_k1_fixture_manifest.v1":
        raise ValueError("unsupported fixture manifest schema")
    if manifest.get("suite_id") != V2_FIXTURE_SUITE_ID or manifest.get("suite_version") != 2:
        raise ValueError("fixture manifest must identify the artifact-pinned v2 suite")
    if manifest.get("frozen_denominator") != scorecard["frozen_denominator"]:
        raise ValueError("fixture manifest denominator differs from the scorecard")
    if manifest.get("frozen_case_definitions_sha256") != scorecard["frozen_case_definitions_sha256"]:
        raise ValueError("fixture manifest case-definition digest differs from the scorecard")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != scorecard["frozen_denominator"]:
        raise ValueError("fixture manifest must contain the frozen number of cases")
    expected_identity = [(case["id"], case["name"]) for case in scorecard["cases"]]
    actual_identity = [(case.get("id"), case.get("name")) for case in cases]
    if actual_identity != expected_identity:
        raise ValueError("fixture manifest case IDs/names differ from the scorecard")

    source_dirs = []
    for case in cases:
        case_id = case["id"]
        source_data_dir = case.get("source_data_dir")
        if not isinstance(source_data_dir, str) or not source_data_dir:
            raise ValueError(f"{case_id}: missing source_data_dir")
        source_path = Path(source_data_dir)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError(f"{case_id}: source_data_dir must be root-relative and contained")
        source_dirs.append(source_data_dir)

        files = case.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"{case_id}: fixture file list is empty")
        names = [row.get("name") for row in files]
        if len(names) != len(set(names)):
            raise ValueError(f"{case_id}: duplicate fixture filenames")
        if not REQUIRED_FIXTURE_FILES.issubset(names):
            raise ValueError(f"{case_id}: required fixture files are missing")
        stacks = [
            name for name in names if isinstance(name, str) and name.startswith("particles.") and name.endswith(".mrcs")
        ]
        if len(stacks) != 1:
            raise ValueError(f"{case_id}: expected exactly one particle stack")
        for row in files:
            name = row.get("name")
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError(f"{case_id}: fixture filename must be a basename")
            if not isinstance(row.get("size"), int) or row["size"] <= 0:
                raise ValueError(f"{case_id}/{name}: invalid fixture size")
            if not isinstance(row.get("sha256"), str) or SHA256_RE.fullmatch(row["sha256"]) is None:
                raise ValueError(f"{case_id}/{name}: invalid SHA-256")
    if len(source_dirs) != len(set(source_dirs)):
        raise ValueError("fixture manifest source_data_dir values must be unique")
    return manifest


def render_markdown(scorecard: dict, fixture_manifest: dict, fixture_manifest_sha256: str) -> str:
    cases = scorecard["cases"]
    counts = scorecard["current_snapshot"]["counts"]
    passed = counts["pass"]
    total = scorecard["frozen_denominator"]
    evaluated = passed + counts["fail"]
    intermediate_passed = sum(case["intermediate_result"] == "pass" for case in cases)
    source = scorecard["current_snapshot"]["source_ledger"]
    fixture_bytes = sum(row["size"] for case in fixture_manifest["cases"] for row in case["files"])

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
        "The artifact-pinned fixture manifest is checked into the repository and binds all "
        f"{len(fixture_manifest['cases'])} cases ({fixture_bytes:,} bytes) to exact file sizes and SHA-256 "
        f"digests. Manifest SHA-256: `{fixture_manifest_sha256}`. Regenerated inputs are non-scoring "
        "replicates.",
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
    replicates = scorecard.get("replicate_diagnostics", [])
    if replicates:
        lines += [
            "",
            "## Non-scoring regenerated-data diagnostics",
            "",
            "These runs exercise the same parameter definitions with newly generated particle bytes. "
            "They are useful robustness evidence but never change the fixed-suite score.",
            "",
            "| Case | Trajectory | Topology | Final cross-engine FSC-AUC | Final GT delta | Jobs |",
            "|---|---|---|---:|---:|---|",
        ]
        for replicate in replicates:
            jobs = replicate["jobs"]
            lines.append(
                f"| `{replicate['case_id']}` | {replicate['trajectory_result']} | "
                f"{replicate['intermediate_result']} | {replicate['final_cross_engine_fsc_auc']:.9f} | "
                f"{replicate['final_gt_fsc_auc_delta']:+.9f} | "
                f"science {jobs['science']}; audit {jobs['audit']} |"
            )
    lines += [
        "",
        "Generate this PR-ready table with:",
        "",
        "```bash",
        "pixi run python scripts/summarize_em_relion_parity_scorecard.py",
        "```",
        "",
        "Launch a scoring rerun with `--scorecard`. This fail-closed mode requires the",
        "checked-in fixture manifest/root pair and forces autonomous RELION pairing,",
        "per-iteration RECOVAR maps, grid correction off, and valid convergence-only",
        "finalization. For example:",
        "",
        "```bash",
        'EM_K1_MATRIX_FIXTURE_MANIFEST="$PWD/docs/math/em_relion_parity_fixture_manifest_v2.json" \\',
        "EM_K1_MATRIX_FIXTURE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex \\",
        "EM_K1_MATRIX_CASES=2,3 \\",
        "./scripts/run_em_k1_robustness_matrix_slurm.sh --scorecard",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--fixture-manifest", type=Path, default=DEFAULT_FIXTURE_MANIFEST)
    parser.add_argument("--check", type=Path, help="fail if this generated Markdown file is stale")
    args = parser.parse_args()

    scorecard = load_and_validate(args.scorecard)
    fixture_manifest = load_and_validate_fixture_manifest(args.fixture_manifest, scorecard)
    rendered = render_markdown(scorecard, fixture_manifest, sha256_file(args.fixture_manifest))
    if args.check is not None:
        if args.check.read_text() != rendered:
            raise SystemExit(f"stale generated scorecard: {args.check}")
        print(f"scorecard valid and current: {args.check}")
        return 0
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
