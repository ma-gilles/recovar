#!/usr/bin/env python3
"""Compare the exact K=1 coarse significance cutoff for one particle."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _map_relion_table,
    _translation_permutation,
)
from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
    load_artifact,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _float32_from_bits(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def _float32_bits(value: float) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def _candidate_key(flat_index: int, n_translations: int) -> dict[str, int]:
    return {
        "flat_index": int(flat_index),
        "rotation_index": int(flat_index // n_translations),
        "translation_index": int(flat_index % n_translations),
    }


def _cutoff_neighborhood(
    weights: np.ndarray,
    threshold: float,
    *,
    n_translations: int,
    radius: int = 3,
) -> list[dict[str, object]]:
    flat = np.asarray(weights, dtype=np.float32).reshape(-1)
    order = np.argsort(flat, kind="stable")
    sorted_weights = flat[order]
    cutoff = int(np.searchsorted(sorted_weights, np.float32(threshold), side="left"))
    rows = []
    for rank in range(max(0, cutoff - radius), min(flat.size, cutoff + radius + 1)):
        index = int(order[rank])
        rows.append(
            {
                "ascending_rank": rank,
                **_candidate_key(index, n_translations),
                "weight": float(flat[index]),
                "weight_bits": _float32_bits(flat[index]),
                "selected_by_threshold": bool(flat[index] >= np.float32(threshold)),
            }
        )
    return rows


def build_report(native_path: Path, recovar_path: Path) -> dict[str, object]:
    native = load_artifact(native_path)
    with np.load(recovar_path, allow_pickle=False) as payload:
        original_index = int(payload["original_index"])
        n_rot = int(payload["n_rot"])
        n_trans = int(payload["n_trans"])
        translations = np.asarray(payload["translations"], dtype=np.float64)
        rec_scores = np.asarray(payload["scores_pre_prior_per_class"][0], dtype=np.float32)
        rec_with_prior = np.asarray(payload["scores_with_prior_per_class"][0], dtype=np.float32)
        rec_weights = np.asarray(payload["weights_per_class"], dtype=np.float32).reshape(n_rot, n_trans)
        rec_mask = np.asarray(payload["significant_mask"], dtype=bool).reshape(n_rot, n_trans)
        adaptive_fraction = float(payload["adaptive_fraction"])

    _require(native.stack_index - 1 == original_index, "particle identity mismatch")
    n_directions, n_psi, native_n_trans = native.header[10:13]
    _require((n_rot, n_trans) == (n_directions * n_psi, native_n_trans), "topology mismatch")
    trans_permutation, trans_mapping = _translation_permutation(native.translations, translations)

    def mapped(values: np.ndarray) -> np.ndarray:
        return _map_relion_table(
            values,
            n_directions=n_directions,
            n_psi=n_psi,
            relion_to_recovar_translation=trans_permutation,
        )

    native_raw = mapped(native.raw_diff2)
    native_weights = mapped(native.weights).astype(np.float32)
    native_mask = mapped(native.significant_mask).astype(bool)
    native_sum = np.float32(_float32_from_bits(native.header[17]))
    native_recorded_threshold = np.float32(_float32_from_bits(native.header[16]))
    native_threshold = (
        np.min(native_weights[native_mask]).astype(np.float32)
        if native_recorded_threshold == 0.0
        else native_recorded_threshold
    )
    _require(native_sum > 0, "native sum weight must be positive")
    native_probabilities = (native_weights / native_sum).astype(np.float32)
    native_threshold_probability = np.float32(native_threshold / native_sum)

    rec_sum = np.float32(np.sum(rec_weights, dtype=np.float32))
    _require(rec_sum > 0, "RECOVAR sum weight must be positive")
    rec_probabilities = (rec_weights / rec_sum).astype(np.float32)
    rec_threshold_probability = np.min(rec_probabilities[rec_mask]).astype(np.float32)

    common = (
        (native_raw != RELION_INVALID_DIFF2)
        & np.isfinite(rec_scores)
        & np.isfinite(rec_with_prior)
    )
    raw_offset = float(
        np.median(
            rec_scores[common].astype(np.float64)
            + native_raw[common].astype(np.float64)
        )
    )
    mismatches = np.flatnonzero(native_mask.reshape(-1) != rec_mask.reshape(-1))
    mismatch_rows = []
    for flat_index in mismatches:
        rotation, translation = divmod(int(flat_index), n_trans)
        native_probability = native_probabilities[rotation, translation]
        rec_probability = rec_probabilities[rotation, translation]
        mismatch_rows.append(
            {
                **_candidate_key(int(flat_index), n_trans),
                "native_selected": bool(native_mask[rotation, translation]),
                "recovar_selected": bool(rec_mask[rotation, translation]),
                "native_raw_diff2": float(native_raw[rotation, translation]),
                "recovar_raw_score": float(rec_scores[rotation, translation]),
                "centered_raw_score_residual": float(
                    rec_scores[rotation, translation]
                    + native_raw[rotation, translation]
                    - raw_offset
                ),
                "recovar_score_with_prior": float(rec_with_prior[rotation, translation]),
                "native_probability": float(native_probability),
                "native_probability_bits": _float32_bits(native_probability),
                "native_margin_to_native_threshold": float(
                    native_probability - native_threshold_probability
                ),
                "recovar_probability": float(rec_probability),
                "recovar_probability_bits": _float32_bits(rec_probability),
                "recovar_margin_to_recovar_threshold": float(
                    rec_probability - rec_threshold_probability
                ),
                "recovar_selected_at_native_threshold": bool(
                    rec_probability >= native_threshold_probability
                ),
                "native_selected_at_recovar_threshold": bool(
                    native_probability >= rec_threshold_probability
                ),
            }
        )

    raw_residual = rec_scores[common].astype(np.float64) + native_raw[common].astype(np.float64)
    raw_residual -= raw_offset
    probability_delta = rec_probabilities.astype(np.float64) - native_probabilities.astype(np.float64)
    return {
        "schema": "recovar.em.k1.significance_cutoff_boundary.v2",
        "physical_iteration": int(native.header[5]),
        "stack_index_one_based": int(native.stack_index),
        "original_index_zero_based": original_index,
        "metric_policy": "exact float32 cutoff bits and absolute intermediate errors; no correlation",
        "counts": {
            "native_significant": int(np.count_nonzero(native_mask)),
            "recovar_significant": int(np.count_nonzero(rec_mask)),
            "mask_mismatches": int(mismatches.size),
        },
        "posterior": {
            "total_variation": float(0.5 * np.sum(np.abs(probability_delta))),
            "max_abs": float(np.max(np.abs(probability_delta))),
        },
        "raw_centered_score": {
            "median_abs": float(np.median(np.abs(raw_residual))),
            "p95_abs": float(np.percentile(np.abs(raw_residual), 95)),
            "max_abs": float(np.max(np.abs(raw_residual))),
        },
        "native_cutoff": {
            "sum_weight": float(native_sum),
            "sum_weight_bits": _float32_bits(native_sum),
            "recorded_raw_threshold": float(native_recorded_threshold),
            "recorded_raw_threshold_bits": _float32_bits(native_recorded_threshold),
            "effective_raw_threshold": float(native_threshold),
            "effective_raw_threshold_bits": _float32_bits(native_threshold),
            "normalized_threshold": float(native_threshold_probability),
            "normalized_threshold_bits": _float32_bits(native_threshold_probability),
            "tail_target": float(np.float32((1.0 - adaptive_fraction) * native_sum)),
        },
        "recovar_cutoff": {
            "sum_probability": float(rec_sum),
            "sum_probability_bits": _float32_bits(rec_sum),
            "normalized_threshold": float(rec_threshold_probability),
            "normalized_threshold_bits": _float32_bits(rec_threshold_probability),
            "tail_target": float(np.float32((1.0 - adaptive_fraction) * rec_sum)),
        },
        "mismatch_candidates": mismatch_rows,
        "native_cutoff_neighborhood": _cutoff_neighborhood(
            native_probabilities,
            native_threshold_probability,
            n_translations=n_trans,
        ),
        "recovar_cutoff_neighborhood": _cutoff_neighborhood(
            rec_probabilities,
            rec_threshold_probability,
            n_translations=n_trans,
        ),
        "translation_mapping": trans_mapping,
        "artifacts": {
            "native": str(native_path.resolve()),
            "recovar": str(recovar_path.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    report = build_report(args.native, args.recovar)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
