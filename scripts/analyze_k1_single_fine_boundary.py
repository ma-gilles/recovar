#!/usr/bin/env python3
"""Compare one native RELION fine-score capture with one RECOVAR pass-2 dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_k1_fine_score_boundary import _compare_particle
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import load_fine_score_capture


def analyze(
    *,
    factor_path: Path,
    fine_score_path: Path,
    recovar_path: Path,
    stack_index_one_based: int,
    original_index_zero_based: int,
    physical_image_size: int,
) -> dict[str, object]:
    factor = load_factor_capture(factor_path)
    fine_score = load_fine_score_capture(fine_score_path)
    with np.load(recovar_path, allow_pickle=False) as archive:
        recovar = {name: np.asarray(archive[name]) for name in archive.files}
    recovar["_path"] = np.asarray(str(recovar_path.resolve()))
    current_size = int(np.asarray(recovar["current_size"]).item())
    comparison = _compare_particle(
        target={
            "stack_index_one_based": stack_index_one_based,
            "original_index_zero_based": original_index_zero_based,
            "role": "single_first_divergence_probe",
        },
        factor=factor,
        score=fine_score,
        recovar=recovar,
        native_state_row=None,
        physical_image_size=physical_image_size,
        current_size=current_size,
    )
    return {
        "schema": "recovar.em.k1_single_fine_boundary.v1",
        "status": "complete",
        "metric_policy": "exact staged equality and relative L2; no correlation",
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--fine-score", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--stack-index-one-based", type=int, required=True)
    parser.add_argument("--original-index-zero-based", type=int, required=True)
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        factor_path=args.factor,
        fine_score_path=args.fine_score,
        recovar_path=args.recovar,
        stack_index_one_based=args.stack_index_one_based,
        original_index_zero_based=args.original_index_zero_based,
        physical_image_size=args.physical_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
