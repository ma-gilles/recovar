#!/usr/bin/env python3
"""Compare one RECOVAR VDAM score boundary with repeated native RELION captures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_vdam_native_translation_boundary import analyze  # noqa: E402
from scripts.analyze_em_k1_native_fine_operands import _center, _flat_memmap  # noqa: E402
from scripts.compare_relion_recovar_estep_dump import (  # noqa: E402
    _nearest_rotation_rows_by_matrix,
)


def summarize_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    if len(reports) < 2:
        raise ValueError("score repeat audit requires at least two reports")
    if any(report.get("status") != "complete" for report in reports):
        raise ValueError("every score-boundary report must be complete")
    boundaries = [report["comparisons"]["top_pair_score_boundary"] for report in reports]
    native_keys = [
        (tuple(item["native_best"]["mapped_key"]), tuple(item["native_second"]["mapped_key"]))
        for item in boundaries
    ]
    if any(keys != native_keys[0] for keys in native_keys[1:]):
        raise ValueError("native repeats do not share one top-pair identity")
    native_log_odds = np.asarray(
        [item["native_log_odds_best_over_second"] for item in boundaries], dtype=np.float64
    )
    recovar_log_odds = np.asarray(
        [item["recovar_log_odds_same_order"] for item in boundaries], dtype=np.float64
    )
    if not np.all(recovar_log_odds == recovar_log_odds[0]):
        raise ValueError("RECOVAR replay changed across native repeat reports")
    candidate = float(recovar_log_odds[0])
    return {
        "top_pair_mapped_keys": [list(native_keys[0][0]), list(native_keys[0][1])],
        "native_log_odds": [float(value) for value in native_log_odds],
        "native_log_odds_min": float(native_log_odds.min()),
        "native_log_odds_max": float(native_log_odds.max()),
        "native_log_odds_span": float(native_log_odds.max() - native_log_odds.min()),
        "recovar_log_odds": candidate,
        "recovar_inside_native_range": bool(native_log_odds.min() <= candidate <= native_log_odds.max()),
        "minimum_absolute_distance_to_native": float(np.min(np.abs(native_log_odds - candidate))),
    }


def aligned_score_vectors(native_dir: Path, live: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    native_rotation = np.asarray(
        _flat_memmap(Path(native_dir) / "pass1_acc_rot_idx.bin", np.int32), dtype=np.int32
    )
    native_translation = np.asarray(
        _flat_memmap(Path(native_dir) / "pass1_acc_trans_idx.bin", np.int32), dtype=np.int32
    )
    native_eulers = np.asarray(
        _flat_memmap(Path(native_dir) / "pass1_class0_fine_eulers.bin")
    ).reshape(-1, 3, 3)
    nearest, _, _ = _nearest_rotation_rows_by_matrix(
        native_eulers, live["local_rotation_matrices"]
    )
    rotation = nearest[native_rotation]
    native_raw = np.asarray(
        _flat_memmap(Path(native_dir) / "pass1_exp_Mweight_raw_preprior.bin"),
        dtype=np.float64,
    )
    live_raw = np.asarray(live["pass2_scores_raw"], dtype=np.float64)[
        0, rotation, native_translation
    ]
    return native_raw, -live_raw


def summarize_score_vectors(
    native_vectors: np.ndarray, candidate_vector: np.ndarray
) -> dict[str, object]:
    native = np.asarray(native_vectors, dtype=np.float64)
    candidate = np.asarray(candidate_vector, dtype=np.float64)
    if native.ndim != 2 or native.shape[0] < 2 or candidate.shape != native.shape[1:]:
        raise ValueError("score vectors must have shape (repeat, candidate) and (candidate,)")
    native_centered = np.asarray([_center(row) for row in native], dtype=np.float64)
    candidate_centered = np.asarray(_center(candidate), dtype=np.float64)
    lower = native_centered.min(axis=0)
    upper = native_centered.max(axis=0)
    outside = np.maximum(lower - candidate_centered, candidate_centered - upper)
    outside = np.maximum(outside, 0.0)
    candidate_residuals = native_centered - candidate_centered[None]
    native_pair_rms = [
        float(np.sqrt(np.mean((native_centered[left] - native_centered[right]) ** 2)))
        for left in range(native.shape[0])
        for right in range(left + 1, native.shape[0])
    ]
    return {
        "candidate_count": int(candidate.size),
        "candidate_inside_native_envelope_count": int(np.count_nonzero(outside == 0.0)),
        "candidate_inside_native_envelope_fraction": float(np.mean(outside == 0.0)),
        "candidate_max_distance_outside_native_envelope": float(outside.max()),
        "candidate_rms_distance_outside_native_envelope": float(
            np.sqrt(np.mean(outside * outside))
        ),
        "candidate_nearest_native_centered_rms": float(
            np.min(np.sqrt(np.mean(candidate_residuals * candidate_residuals, axis=1)))
        ),
        "native_pair_centered_rms_min": float(min(native_pair_rms)),
        "native_pair_centered_rms_max": float(max(native_pair_rms)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, default=8)
    parser.add_argument("--live-score", type=Path, required=True)
    parser.add_argument("--full-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat_count < 2:
        parser.error("repeat-count must be at least 2")
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")

    reports = [
        analyze(
            args.panel_root / f"repeat-{index:02d}" / "capture",
            args.live_score,
            full_size=args.full_image_size,
        )
        for index in range(1, args.repeat_count + 1)
    ]
    with np.load(args.live_score, allow_pickle=False) as payload:
        live = {name: np.asarray(payload[name]) for name in payload.files}
    aligned = [
        aligned_score_vectors(
            args.panel_root / f"repeat-{index:02d}" / "capture",
            live,
        )
        for index in range(1, args.repeat_count + 1)
    ]
    candidate_vectors = [item[1] for item in aligned]
    if any(not np.array_equal(candidate_vectors[0], item) for item in candidate_vectors[1:]):
        raise ValueError("native repeat mappings did not preserve one RECOVAR candidate vector")
    payload = {
        "schema": "recovar.vdam_native_score_repeat_panel.v1",
        "status": "complete",
        "panel_root": str(args.panel_root.resolve()),
        "repeat_count": args.repeat_count,
        "summary": summarize_reports(reports),
        "score_vector_envelope": summarize_score_vectors(
            np.asarray([item[0] for item in aligned], dtype=np.float64),
            candidate_vectors[0],
        ),
        "reports": reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
