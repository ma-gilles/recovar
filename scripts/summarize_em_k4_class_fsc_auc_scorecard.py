#!/usr/bin/env python3
"""Validate and render the fixed 60-cell K=4 FSC-AUC scorecard."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "docs" / "math" / "em_k4_class_fsc_auc_scorecard_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "math" / "em_k4_class_fsc_auc_scorecard.md"
SCHEMA = "recovar.em_k4_class_fsc_auc_scorecard.v1"
SUITE_ID = "k4-relion-cuda-4181d340-20260725-direct-class-fsc-auc"
FROZEN_DENOMINATOR = 60
CLASSES = 4
NUMBERED_ITERATIONS = 15
FSC_AUC_GATE = 0.995
PINNED_SCORECARD_SHA256 = "1d349845d4aba63af9d2308b869d31d8f402f82dc070fe6b089665c0daa34a72"
PINNED_SNAPSHOT_SHA256 = "bc10d0555488b22f0bc8d54afe5afc5288064ddb4708bd1c75f3b55dd4c0060a"
PINNED_TRAJECTORY_SHA256 = "5e030ab63c779b8e3050c8fc63ad4efabcc3e353d3b77ce047da8c20e63076fd"
EXPECTED_GPU_UUID = "GPU-5e619c2e-82b4-ff79-cbcb-ab29514a9f30"
EXPECTED_SOURCE_COMMIT = "4181d340997e548af36c6458cce825e133dba95a"
EXPECTED_PASS_COUNTS = (4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 0, 2, 0, 0, 0)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate(
    path: Path,
    *,
    enforce_pinned_bytes: bool = True,
) -> dict:
    """Load the class-level scorecard and enforce all 60 fixed checks."""

    if enforce_pinned_bytes:
        _require(
            _sha256_file(path) == PINNED_SCORECARD_SHA256,
            "scorecard bytes changed without a suite-version change",
        )
    scorecard = json.loads(path.read_text())
    _require(scorecard.get("schema") == SCHEMA, "unsupported scorecard schema")
    _require(scorecard.get("suite_id") == SUITE_ID, "suite identity changed")
    _require(scorecard.get("suite_version") == 1, "suite version changed")
    _require(
        scorecard.get("frozen_denominator") == FROZEN_DENOMINATOR,
        "frozen denominator changed",
    )
    _require(scorecard.get("classes") == CLASSES, "class denominator changed")
    _require(
        scorecard.get("numbered_iterations") == NUMBERED_ITERATIONS,
        "iteration denominator changed",
    )
    _require(
        scorecard.get("direct_fsc_auc_gate") == FSC_AUC_GATE,
        "direct FSC-AUC gate changed",
    )
    _require(
        scorecard.get("metric_policy")
        == ("Shellwise cross-engine FSC-AUC at the fixed 0.995 gate; correlation is not used."),
        "metric policy changed",
    )
    _require(
        scorecard.get("scorecard_change_admissible") is False,
        "class checklist cannot authorize a scorecard change",
    )
    _require(
        scorecard.get("same_physical_gpu") is True,
        "same-physical-GPU contract changed",
    )
    _require(
        scorecard.get("gpu_uuid") == EXPECTED_GPU_UUID,
        "physical GPU identity changed",
    )
    _require(
        scorecard.get("grid_correction") == "unset",
        "grid-correction policy changed",
    )
    _require(
        scorecard.get("forced_final_all_data_after_nonconvergence") is False,
        "invalid forced final all-data is enabled",
    )
    _require(
        scorecard.get("source_commit") == EXPECTED_SOURCE_COMMIT,
        "source commit changed",
    )

    snapshot = scorecard.get("source_snapshot")
    _require(isinstance(snapshot, dict), "source snapshot is missing")
    _require(
        snapshot.get("path") == "docs/math/em_k4_backend_trajectory_snapshot_v2.json",
        "source snapshot path changed",
    )
    _require(
        snapshot.get("sha256") == PINNED_SNAPSHOT_SHA256,
        "source snapshot SHA-256 changed",
    )
    snapshot_path = REPO_ROOT / snapshot["path"]
    _require(
        _sha256_file(snapshot_path) == PINNED_SNAPSHOT_SHA256,
        "checked source snapshot bytes changed",
    )

    trajectory = scorecard.get("source_trajectory")
    _require(isinstance(trajectory, dict), "source trajectory is missing")
    trajectory_path = trajectory.get("path")
    trajectory_sha256 = trajectory.get("sha256")
    _require(
        isinstance(trajectory_path, str) and Path(trajectory_path).is_absolute(),
        "source trajectory path must be absolute",
    )
    _require(
        isinstance(trajectory_sha256, str) and SHA256_RE.fullmatch(trajectory_sha256) is not None,
        "invalid source trajectory SHA-256",
    )
    _require(
        trajectory_sha256 == PINNED_TRAJECTORY_SHA256,
        "source trajectory SHA-256 changed",
    )

    iterations = scorecard.get("iterations")
    _require(
        isinstance(iterations, list) and len(iterations) == NUMBERED_ITERATIONS,
        "iterations do not preserve the frozen denominator",
    )
    _require(
        [record.get("iteration") for record in iterations] == list(range(1, NUMBERED_ITERATIONS + 1)),
        "fixed iteration identity/order changed",
    )
    pass_counts = []
    for record in iterations:
        values = record.get("cross_engine_fsc_auc")
        _require(
            isinstance(values, list) and len(values) == CLASSES,
            f"iteration {record.get('iteration')}: class checks changed",
        )
        _require(
            all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0
                for value in values
            ),
            f"iteration {record.get('iteration')}: invalid FSC-AUC value",
        )
        pass_counts.append(sum(value >= FSC_AUC_GATE for value in values))
    _require(
        tuple(pass_counts) == EXPECTED_PASS_COUNTS,
        "per-iteration class results changed",
    )

    passed = sum(pass_counts)
    expected_summary = {
        "pass": passed,
        "fail": FROZEN_DENOMINATOR - passed,
        "evaluated": FROZEN_DENOMINATOR,
        "iterations_all_classes_passed": sum(count == CLASSES for count in pass_counts),
    }
    _require(
        scorecard.get("summary") == expected_summary,
        "recorded summary does not match fixed class checks",
    )
    return scorecard


def failed_checks(scorecard: dict) -> list[dict[str, object]]:
    """Return stable identities and values for every failed class cell."""

    return [
        {
            "id": f"k4-it{record['iteration']:02d}-class{class_id}",
            "iteration": record["iteration"],
            "class": class_id,
            "fsc_auc": value,
        }
        for record in scorecard["iterations"]
        for class_id, value in enumerate(
            record["cross_engine_fsc_auc"],
            start=1,
        )
        if value < FSC_AUC_GATE
    ]


def verify_source_trajectory(
    scorecard: dict,
    path: Path | None = None,
    *,
    enforce_pinned_bytes: bool = True,
) -> dict:
    """Verify all checked values directly against the sealed trajectory."""

    source_path = Path(scorecard["source_trajectory"]["path"]) if path is None else path
    if enforce_pinned_bytes:
        _require(
            _sha256_file(source_path) == PINNED_TRAJECTORY_SHA256,
            "source trajectory bytes changed",
        )
    trajectory = json.loads(source_path.read_text())
    _require(
        trajectory.get("schema") == "em_k4_fsc_trajectory_audit_v2",
        "unsupported source trajectory schema",
    )
    _require(
        trajectory.get("n_classes") == CLASSES,
        "source trajectory class denominator changed",
    )
    _require(
        trajectory.get("numbered_iteration_count") == NUMBERED_ITERATIONS,
        "source trajectory iteration denominator changed",
    )
    _require(
        trajectory.get("quality_metric_policy")
        == "shellwise FSC and normalized FSC-AUC only; correlation is not computed",
        "source trajectory metric policy changed",
    )
    _require(
        trajectory.get("thresholds", {}).get("per_class_direct_fsc_auc_min") == FSC_AUC_GATE,
        "source trajectory direct FSC-AUC gate changed",
    )
    source_iterations = trajectory.get("numbered_iterations")
    _require(
        isinstance(source_iterations, list) and len(source_iterations) == NUMBERED_ITERATIONS,
        "source trajectory iterations changed",
    )
    for checked, source in zip(
        scorecard["iterations"],
        source_iterations,
        strict=True,
    ):
        iteration = checked["iteration"]
        _require(
            source.get("relion_iteration") == iteration,
            f"iteration {iteration}: source identity changed",
        )
        classes = source.get("classes")
        _require(
            isinstance(classes, list) and len(classes) == CLASSES,
            f"iteration {iteration}: source class checks changed",
        )
        _require(
            [
                (
                    record.get("recovar_class"),
                    record.get("relion_class"),
                )
                for record in classes
            ]
            == [(class_id, class_id) for class_id in range(1, CLASSES + 1)],
            f"iteration {iteration}: source class mapping changed",
        )
        source_values = [record.get("cross_engine", {}).get("fsc_auc") for record in classes]
        _require(
            source_values == checked["cross_engine_fsc_auc"],
            f"iteration {iteration}: checked FSC-AUC values differ from source",
        )
    return trajectory


def render_markdown(scorecard: dict) -> str:
    """Render all 60 cells with explicit checkmarks and FSC-AUC values."""

    summary = scorecard["summary"]
    lines = [
        "# K=4 direct per-class FSC-AUC scorecard",
        "",
        (
            f"Fixed class score: **{summary['pass']} / "
            f"{scorecard['frozen_denominator']} passing** "
            f"({summary['evaluated']} / "
            f"{scorecard['frozen_denominator']} evaluated; "
            f"{summary['iterations_all_classes_passed']} / "
            f"{scorecard['numbered_iterations']} iterations pass all classes)."
        ),
        "",
        (
            f"Each cell is checked when its shellwise cross-engine FSC-AUC is "
            f"at least `{scorecard['direct_fsc_auc_gate']:.3f}`."
        ),
        "",
        "| Iteration | Class 1 | Class 2 | Class 3 | Class 4 | Passed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in scorecard["iterations"]:
        cells = []
        passed = 0
        for value in record["cross_engine_fsc_auc"]:
            checked = value >= FSC_AUC_GATE
            passed += checked
            cells.append(f"{'[x]' if checked else '[ ]'} {value:.9f}")
        lines.append(f"| {record['iteration']} | {' | '.join(cells)} | {passed} / {CLASSES} |")
    failures = failed_checks(scorecard)
    lines.extend(
        [
            "",
            "Remaining failed cells: " + ", ".join(f"`{record['id']}`" for record in failures) + ".",
            "",
            f"Metric policy: {scorecard['metric_policy']}",
            "",
            (
                f"Source snapshot: `{scorecard['source_snapshot']['path']}` "
                f"(SHA-256 `{scorecard['source_snapshot']['sha256']}`)."
            ),
            (
                f"Source trajectory: `{scorecard['source_trajectory']['path']}` "
                f"(SHA-256 `{scorecard['source_trajectory']['sha256']}`)."
            ),
            "",
            "To validate and regenerate:",
            "",
            "```bash",
            ("pixi run python scripts/summarize_em_k4_class_fsc_auc_scorecard.py --check"),
            "```",
            "",
            "On Della, replay all 60 values against the sealed trajectory:",
            "",
            "```bash",
            ("pixi run python scripts/summarize_em_k4_class_fsc_auc_scorecard.py --check --verify-source-trajectory"),
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
    parser.add_argument("--verify-source-trajectory", action="store_true")
    args = parser.parse_args()
    scorecard = load_and_validate(args.scorecard)
    if args.verify_source_trajectory:
        verify_source_trajectory(scorecard)
    rendered = render_markdown(scorecard)
    if args.check:
        target = DEFAULT_MARKDOWN if args.output is None else args.output
        if target.read_text() != rendered:
            raise SystemExit(f"{target} is stale; regenerate it")
    elif args.output is not None:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
