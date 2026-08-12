#!/usr/bin/env python3
"""Replay one native RELION fine-significance boundary through RECOVAR.

The fine-score sidecar stores the exact device float32 weights before RELION's
sort/scan.  The geometry-only BPref sidecar stores the production significance
threshold, full weight normalization, and accepted-hypothesis count.  This
script compares those native scalars with both JAX's sort/scan and RECOVAR's
RELION-compatible CUB primitive without rerunning an EM trajectory.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

if __package__:
    from .validate_relion_bpref_factor_capture import load_factor_capture
    from .validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture
else:
    from validate_relion_bpref_factor_capture import load_factor_capture
    from validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture


def _float32_from_header(value: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0])


def _bits(value: np.float32) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def _threshold_summary(
    sorted_weights: np.ndarray,
    cumulative: np.ndarray,
    *,
    adaptive_fraction: float,
) -> dict[str, Any]:
    sorted_weights = np.asarray(sorted_weights, dtype=np.float32)
    cumulative = np.asarray(cumulative, dtype=np.float32)
    sum_weight = np.float32(cumulative[-1])
    double_fraction_target = np.float32(
        np.float64(1.0 - adaptive_fraction) * np.float64(sum_weight)
    )
    parsed_float_fraction_target = np.float32(
        (np.float64(1.0) - np.float64(np.float32(adaptive_fraction)))
        * np.float64(sum_weight)
    )

    def summarize_target(tail_target: np.float32) -> dict[str, Any]:
        threshold_index = min(
            int(np.searchsorted(cumulative, tail_target, side="right")),
            sorted_weights.size - 1,
        )
        threshold = np.float32(sorted_weights[threshold_index])
        return {
            "tail_target": float(tail_target),
            "tail_target_bits": _bits(tail_target),
            "threshold_index": threshold_index,
            "threshold": float(threshold),
            "threshold_bits": _bits(threshold),
            "accepted_count": int(np.count_nonzero(sorted_weights >= threshold)),
        }

    return {
        "sum_weight": float(sum_weight),
        "sum_weight_bits": _bits(sum_weight),
        "double_fraction_target": summarize_target(double_fraction_target),
        "parsed_float_fraction_target": summarize_target(parsed_float_fraction_target),
    }


def analyze(
    fine_score_path: Path,
    factor_path: Path,
    *,
    adaptive_fraction: float,
) -> dict[str, Any]:
    fine = load_fine_score_capture(fine_score_path)
    factor = load_factor_capture(factor_path)
    if fine.stack_index != factor.stack_index:
        raise ValueError("fine-score and BPref captures identify different particles")

    active = (fine.candidates["flags"] & ACTIVE) != 0
    weights = np.asarray(fine.candidates["post_exponent_weight"][active], dtype=np.float32)
    scores = np.asarray(fine.candidates["combined_preexponent"][active], dtype=np.float32)
    if weights.size == 0:
        raise ValueError("native fine-score capture has no active weights")

    jax_weights = jnp.asarray(weights)
    jax_sorted = jnp.sort(jax_weights)
    jax_cumulative = jnp.cumsum(jax_sorted, dtype=jnp.float32)
    jax_summary = _threshold_summary(
        np.asarray(jax_sorted),
        np.asarray(jax_cumulative),
        adaptive_fraction=adaptive_fraction,
    )

    cub_summary: dict[str, Any] | None = None
    if jax.default_backend() == "gpu":
        from recovar.cuda_backproject import relion_cub_sort_scan_f32

        cub_sorted, cub_cumulative = relion_cub_sort_scan_f32(jax_weights)
        cub_summary = _threshold_summary(
            np.asarray(cub_sorted),
            np.asarray(cub_cumulative),
            adaptive_fraction=adaptive_fraction,
        )

    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_f32_fine_reconstruction_probs,
    )

    production = _relion_f32_fine_reconstruction_probs(
        jnp.asarray(scores)[None, :],
        adaptive_fraction=adaptive_fraction,
    )
    production_summary = {
        "accepted_count": int(np.asarray(production[2])[0]),
        "sum_weight": float(np.asarray(production[3])[0]),
        "sum_weight_bits": _bits(np.asarray(production[3])[0]),
        "threshold": float(np.asarray(production[4])[0]),
        "threshold_bits": _bits(np.asarray(production[4])[0]),
    }

    native_threshold = _float32_from_header(factor.header[25])
    native_sum_weight = _float32_from_header(factor.header[26])
    native_count_from_weights = int(np.count_nonzero(weights >= native_threshold))
    return {
        "schema": "recovar.em.k1_native_fine_significance_replay.v2",
        "status": "complete",
        "metric_policy": "exact float32 bits and accepted counts; no correlation",
        "device": str(jax.devices()[0]),
        "stack_index_one_based": fine.stack_index,
        "candidate_count": int(fine.candidates.size),
        "active_candidate_count": int(weights.size),
        "adaptive_fraction": adaptive_fraction,
        "native": {
            "sum_weight": float(native_sum_weight),
            "sum_weight_bits": _bits(native_sum_weight),
            "threshold": float(native_threshold),
            "threshold_bits": _bits(native_threshold),
            "accepted_count_header": int(factor.header[45]),
            "accepted_count_from_captured_weights": native_count_from_weights,
        },
        "jax": jax_summary,
        "cub": cub_summary,
        "production_fine_helper": production_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-score", type=Path, required=True)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--adaptive-fraction", type=float, default=0.999)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.fine_score,
        args.factor,
        adaptive_fraction=args.adaptive_fraction,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
