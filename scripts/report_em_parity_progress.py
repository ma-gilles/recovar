#!/usr/bin/env python3
"""Report the current scientific EM/VDAM acceptance panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_em_k1_realdata_science_equivalence import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_REALDATA_SCORECARD,
)
from scripts.summarize_em_k1_realdata_science_equivalence import (  # noqa: E402
    load_and_validate_scorecard as load_realdata_scorecard,
)
from scripts.summarize_em_k4_class_fsc_auc_scorecard import (  # noqa: E402
    DEFAULT_SCORECARD as DEFAULT_K4_SCORECARD,
)
from scripts.summarize_em_k4_class_fsc_auc_scorecard import failed_checks as k4_failed_checks  # noqa: E402
from scripts.summarize_em_k4_class_fsc_auc_scorecard import (  # noqa: E402
    load_and_validate as load_k4_scorecard,
)
from scripts.summarize_em_relion_parity_scorecard import DEFAULT_SCORECARD as DEFAULT_K1_SCORECARD  # noqa: E402
from scripts.summarize_em_relion_parity_scorecard import load_and_validate as load_k1_scorecard  # noqa: E402
from scripts.summarize_vdam_relion_parity_scorecard import DEFAULT_SCORECARD as DEFAULT_VDAM_SCORECARD  # noqa: E402
from scripts.summarize_vdam_relion_parity_scorecard import load_and_validate as load_vdam_scorecard  # noqa: E402

SCHEMA = "recovar.em_parity_progress.v26"
ARCHIVE_URL = (
    "https://github.com/ma-gilles/recovar-experiments/tree/"
    "58574b101593004d65cea4b14872e795335ad475/"
    "snapshots/em_causal_scorecards_20260921"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _panel(label: str, passed: int, evaluated: int, denominator: int) -> dict[str, Any]:
    if not 0 <= passed <= evaluated <= denominator:
        raise ValueError(f"invalid panel counts for {label}")
    return {
        "label": label,
        "passed": passed,
        "evaluated": evaluated,
        "denominator": denominator,
        "rate_percent": round(100 * passed / denominator, 1),
    }


def build_progress(
    *,
    k1_path: Path = DEFAULT_K1_SCORECARD,
    k4_path: Path = DEFAULT_K4_SCORECARD,
    vdam_path: Path = DEFAULT_VDAM_SCORECARD,
    realdata_path: Path = DEFAULT_REALDATA_SCORECARD,
) -> dict[str, Any]:
    """Validate maintained scorecards and return their compact status."""

    k1 = load_k1_scorecard(k1_path)
    k4 = load_k4_scorecard(k4_path)
    vdam = load_vdam_scorecard(vdam_path)
    realdata = load_realdata_scorecard(realdata_path)

    k1_counts = k1["current_snapshot"]["counts"]
    k4_counts = k4["summary"]
    vdam_counts = vdam["current_snapshot"]["counts"]
    target = next(case for case in realdata["cases"] if case["role"] == "scoring")
    calibrations = [case for case in realdata["cases"] if case["role"] == "calibration"]

    return {
        "schema": SCHEMA,
        "panels": [
            _panel(
                "K=1 strict FSC/FSC-AUC",
                k1_counts["pass"],
                k1_counts["pass"] + k1_counts["fail"],
                k1["frozen_denominator"],
            ),
            _panel(
                "K=4 per-class FSC-AUC",
                k4_counts["pass"],
                k4_counts["evaluated"],
                k4["frozen_denominator"],
            ),
            _panel(
                "K=4 all-class iterations",
                k4_counts["iterations_all_classes_passed"],
                k4["numbered_iterations"],
                k4["numbered_iterations"],
            ),
            _panel(
                "VDAM fixed RELION parity",
                vdam_counts["pass"],
                vdam_counts["pass"] + vdam_counts["fail"],
                vdam["frozen_denominator"],
            ),
        ],
        "realdata": {
            "calibration_cases": len(calibrations),
            "target": target["id"],
            "target_status": target["status"],
        },
        "remaining": {
            "k1": [case["id"] for case in k1["cases"] if case["result"] != "pass"],
            "k4": k4_failed_checks(k4),
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in {
                "k1": k1_path,
                "k4": k4_path,
                "vdam": vdam_path,
                "realdata": realdata_path,
            }.items()
        },
        "historical_diagnostics": ARCHIVE_URL,
    }


def render_markdown(progress: dict[str, Any]) -> str:
    lines = [
        "| Scientific panel | Passed | Evaluated | Denominator | Rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for panel in progress["panels"]:
        lines.append(
            f"| {panel['label']} | **{panel['passed']}** | {panel['evaluated']} | "
            f"{panel['denominator']} | {panel['rate_percent']:.1f}% |"
        )
    realdata = progress["realdata"]
    lines.extend(
        [
            "",
            f"Real-data calibration cases: **{realdata['calibration_cases']}**. "
            f"Target `{realdata['target']}`: **{realdata['target_status']}**.",
            "",
            "Remaining K=1 cases: " + (", ".join(progress["remaining"]["k1"]) or "none") + ".",
            f"Historical causal and repeatability panels: {progress['historical_diagnostics']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()
    progress = build_progress()
    output = render_markdown(progress) if args.format == "markdown" else json.dumps(progress, indent=2) + "\n"
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
