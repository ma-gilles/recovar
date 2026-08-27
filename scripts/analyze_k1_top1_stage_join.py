#!/usr/bin/env python3
"""Run the preregistered K=1 top-one fine-stage comparison fail-closed."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_k1_fine_score_stages import analyze

MANIFEST_SCHEMA = "recovar.em.k1_top1_stage_join_manifest.v1"
REPORT_SCHEMA = "recovar.em.k1_top1_stage_join.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolve_one(pattern: str, *, label: str) -> Path:
    matches = [Path(value) for value in sorted(glob.glob(pattern))]
    _require(len(matches) == 1, f"expected exactly one {label} for {pattern!r}, got {matches}")
    _require(matches[0].is_file(), f"{label} is not a file: {matches[0]}")
    return matches[0]


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> int:
    _require(name in archive.files, f"RECOVAR capture is missing {name}")
    value = np.asarray(archive[name])
    _require(value.shape == (), f"RECOVAR capture {name} must be scalar, got {value.shape}")
    return int(value.item())


def run_join(
    *,
    manifest_path: Path,
    physical_image_size: int,
    top_count: int,
) -> dict[str, Any]:
    _require(physical_image_size > 0, "physical_image_size must be positive")
    _require(top_count >= 2, "top_count must be at least two")
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "unexpected stage-join manifest schema")
    _require(
        manifest.get("status") == "preregistered_waiting_for_captures",
        "stage-join manifest is not in its sealed preregistered state",
    )
    target = manifest["target"]
    expected = manifest["expected_outputs"]
    recovar_capture = Path(expected["recovar_pass2"])
    _require(recovar_capture.is_file(), f"missing RECOVAR pass-2 capture: {recovar_capture}")
    native_fine_score = _resolve_one(expected["native_fine_score_glob"], label="native fine-score capture")
    native_factor = _resolve_one(expected["native_factor_glob"], label="native factor capture")

    with np.load(recovar_capture, allow_pickle=False) as archive:
        original_index = _scalar(archive, "original_index")
        current_size = _scalar(archive, "current_size")
        iteration = _scalar(archive, "iteration")
    _require(
        original_index == int(target["source_row_zero_based"]),
        f"RECOVAR source row changed: {original_index} != {target['source_row_zero_based']}",
    )
    _require(
        current_size == int(target["current_size"]),
        f"RECOVAR current size changed: {current_size} != {target['current_size']}",
    )
    _require(
        iteration == int(target["relion_iteration"]),
        f"RECOVAR physical iteration changed: {iteration} != {target['relion_iteration']}",
    )

    stage_report = analyze(
        native_factor=native_factor,
        native_fine_score=native_fine_score,
        recovar_capture=recovar_capture,
        physical_image_size=physical_image_size,
        top_count=top_count,
    )
    _require(
        int(stage_report["stack_index_one_based"]) == int(target["stack_index_one_based"]),
        "native stack identity changed",
    )
    # The frozen panel values come from rlnNrOfSignificantSamples in the
    # numbered output STAR, while the stage analyzer counts selected fine
    # candidates in the captured pass-2 table.  These are different
    # quantities (for the sealed case-4 row they are 4/5 versus 20/20), so
    # preserve both as provenance instead of asserting numerical equality.
    metadata_significant_counts = {
        "semantic": "numbered-output STAR rlnNrOfSignificantSamples",
        "native_relion": int(target["relion_significant_count"]),
        "recovar": int(target["recovar_significant_count"]),
    }
    captured_fine_support_counts = {
        "semantic": "selected fine candidates in the captured pass-2 table",
        "native_relion": int(stage_report["native_significant_count"]),
        "recovar": int(stage_report["recovar_significant_count"]),
    }

    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "metric_policy": "exact and relative-L2 intermediates; no correlation",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "target": target,
        "significant_count_semantics": {
            "metadata": metadata_significant_counts,
            "captured_fine_support": captured_fine_support_counts,
            "cross_semantic_equality_asserted": False,
        },
        "stage_analysis": stage_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, default=256)
    parser.add_argument("--top-count", type=int, default=64)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = run_join(
        manifest_path=args.manifest,
        physical_image_size=args.physical_image_size,
        top_count=args.top_count,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
