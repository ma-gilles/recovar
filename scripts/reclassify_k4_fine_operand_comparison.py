#!/usr/bin/env python3
"""Correct single-candidate attribution in an immutable K=4 v8 report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_k4_relion_recovar_fine_operands import (  # noqa: E402
    _select_component_classification,
)

SCHEMA = "k4_relion_recovar_fine_operand_classification_v2"
INPUT_SCHEMA = "k4_relion_recovar_fine_operand_comparison_v8"
COMPONENTS = ("reference", "shifted_image", "corr")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_counterfactual(
    value: object,
    *,
    centered: bool,
) -> dict[str, object]:
    _require(isinstance(value, dict), "component counterfactual is missing")
    counterfactual = dict(value)
    _require(
        counterfactual.get("deltas_centered") is centered,
        "component counterfactual centering mode changed",
    )
    records = counterfactual.get("single_component_substitution")
    _require(isinstance(records, dict), "component substitution records are missing")
    _require(tuple(records) == COMPONENTS, "component identity/order changed")
    strongest = counterfactual.get("strongest_single_component")
    _require(strongest in COMPONENTS, "strongest component is invalid")

    target_l2_values = []
    for name in COMPONENTS:
        record = records[name]
        _require(isinstance(record, dict), f"{name}: invalid component record")
        target_l2 = record.get("target_all_recovar_delta_l2")
        removed_fraction = record.get("target_delta_energy_removed_fraction")
        _require(
            isinstance(target_l2, (int, float))
            and math.isfinite(target_l2)
            and target_l2 >= 0,
            f"{name}: invalid target delta L2",
        )
        _require(
            isinstance(removed_fraction, (int, float))
            and math.isfinite(removed_fraction),
            f"{name}: invalid energy-removed fraction",
        )
        target_l2_values.append(float(target_l2))
    _require(
        len(set(target_l2_values)) == 1,
        "component records disagree on target delta L2",
    )
    strongest_fraction = records[strongest][
        "target_delta_energy_removed_fraction"
    ]
    _require(
        float(strongest_fraction)
        == max(
            float(records[name]["target_delta_energy_removed_fraction"])
            for name in COMPONENTS
        ),
        "recorded strongest component is not strongest",
    )

    target_l2 = target_l2_values[0]
    counterfactual["target_all_recovar_delta_l2"] = target_l2
    counterfactual["informative"] = target_l2 > 0
    return counterfactual


def reclassify(comparison_path: Path) -> dict[str, object]:
    """Validate a frozen v8 report and return its corrected attribution."""

    comparison = json.loads(comparison_path.read_text())
    _require(comparison.get("schema") == INPUT_SCHEMA, "input schema changed")
    _require(comparison.get("status") == "complete", "input report is incomplete")
    candidates = comparison.get("candidates")
    _require(isinstance(candidates, list) and candidates, "candidate rows are missing")
    raw = _validated_counterfactual(
        comparison.get("raw_diff2_component_counterfactual"),
        centered=False,
    )
    centered = _validated_counterfactual(
        comparison.get("centered_raw_diff2_component_counterfactual"),
        centered=True,
    )
    classification, basis = _select_component_classification(
        raw,
        centered,
        candidate_count=len(candidates),
    )
    selected = (
        centered
        if basis == "centered_raw_diff2"
        else raw if basis == "raw_diff2" else None
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification": classification,
        "classification_basis": basis,
        "classification_changed": comparison.get("classification") != classification,
        "source_classification": comparison.get("classification"),
        "candidate_count": len(candidates),
        "selected_component": (
            selected["strongest_single_component"] if selected is not None else None
        ),
        "selected_target_delta_energy_removed_fraction": (
            selected["strongest_target_delta_energy_removed_fraction"]
            if selected is not None
            else None
        ),
        "counterfactuals": {
            "raw_diff2": {
                "informative": raw["informative"],
                "target_all_recovar_delta_l2": raw[
                    "target_all_recovar_delta_l2"
                ],
            },
            "centered_raw_diff2": {
                "informative": centered["informative"],
                "target_all_recovar_delta_l2": centered[
                    "target_all_recovar_delta_l2"
                ],
            },
        },
        "metric_policy": (
            "select the largest single-substitution effect from informative "
            "centered deltas for two or more candidates, otherwise from raw "
            "deltas; no fitted scale, sign, threshold, map metric, or correlation"
        ),
        "scorecard_change_admissible": False,
        "input": {
            "comparison": str(comparison_path.resolve()),
            "comparison_sha256": _sha256(comparison_path),
            "comparison_schema": INPUT_SCHEMA,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = reclassify(args.comparison)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
