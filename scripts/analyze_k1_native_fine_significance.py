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


def _exact_metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    mismatch = reference != candidate
    return {
        "exact_equal": bool(not np.any(mismatch)),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "max_abs": float(
            np.max(np.abs(reference - candidate), initial=np.float32(0.0))
        ),
    }


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
    recovar_capture: Path | None = None,
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

    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_f32_fine_reconstruction_probs,
    )

    jax_weights = jnp.asarray(weights)
    jax_sorted = jnp.sort(jax_weights)
    jax_cumulative = jnp.cumsum(jax_sorted, dtype=jnp.float32)
    jax_summary = _threshold_summary(
        np.asarray(jax_sorted),
        np.asarray(jax_cumulative),
        adaptive_fraction=adaptive_fraction,
    )

    cub_summary: dict[str, Any] | None = None
    cuda_expf: dict[str, Any] | None = None
    cuda_dense_rectangular: dict[str, Any] | None = None
    if jax.default_backend() == "gpu":
        from recovar.cuda_backproject import (
            relion_cub_sort_scan_f32,
            relion_divide_f32,
            relion_exponentiate_f32,
        )

        cub_sorted, cub_cumulative = relion_cub_sort_scan_f32(jax_weights)
        cub_summary = _threshold_summary(
            np.asarray(cub_sorted),
            np.asarray(cub_cumulative),
            adaptive_fraction=adaptive_fraction,
        )
        exponent_add = _float32_from_header(fine.header[20])
        cuda_raw_weights = relion_exponentiate_f32(
            jnp.asarray(scores),
            jnp.asarray(exponent_add, dtype=jnp.float32),
        )
        cuda_sorted, cuda_cumulative = relion_cub_sort_scan_f32(cuda_raw_weights)
        cuda_expf = {
            "exponent_add": float(exponent_add),
            "exponent_add_bits": _bits(exponent_add),
            "raw_weight_comparison": _exact_metric(weights, np.asarray(cuda_raw_weights)),
            "sort_scan": _threshold_summary(
                np.asarray(cuda_sorted),
                np.asarray(cuda_cumulative),
                adaptive_fraction=adaptive_fraction,
            ),
        }
        if recovar_capture is not None:
            with np.load(recovar_capture, allow_pickle=False) as archive:
                dense_scores = np.asarray(archive["scores_with_prior"], dtype=np.float32).reshape(-1)
                dense_candidate_mask = np.asarray(archive["candidate_mask"], dtype=bool).reshape(-1)
            dense_finite = dense_candidate_mask & np.isfinite(dense_scores)
            dense_best = np.max(dense_scores[dense_finite])
            dense_add = np.float32(np.float32(50.0) - np.float32(dense_best))
            dense_weights = relion_exponentiate_f32(
                jnp.asarray(dense_scores),
                jnp.asarray(dense_add, dtype=jnp.float32),
            )
            dense_sorted, dense_cumulative = relion_cub_sort_scan_f32(dense_weights)
            dense_production = _relion_f32_fine_reconstruction_probs(
                jnp.asarray(dense_scores)[None, :],
                adaptive_fraction=adaptive_fraction,
            )
            dense_threshold = np.asarray(dense_production[4], dtype=np.float32)[0]
            dense_sum_weight = np.asarray(dense_production[3], dtype=np.float32)[0]
            dense_expected_mask = dense_finite & (
                np.asarray(dense_weights, dtype=np.float32) >= dense_threshold
            )
            dense_cuda_normalized = relion_divide_f32(
                dense_weights,
                jnp.asarray(dense_sum_weight, dtype=jnp.float32),
            )
            dense_expected_probs = np.where(
                dense_expected_mask,
                np.asarray(dense_cuda_normalized, dtype=np.float32),
                np.float32(0.0),
            )
            cuda_dense_rectangular = {
                "candidate_slot_count": int(dense_scores.size),
                "active_candidate_count": int(np.count_nonzero(dense_finite)),
                "exponent_add": float(dense_add),
                "exponent_add_bits": _bits(dense_add),
                "sort_scan": _threshold_summary(
                    np.asarray(dense_sorted),
                    np.asarray(dense_cumulative),
                    adaptive_fraction=adaptive_fraction,
                ),
                "production_helper": {
                    "accepted_count": int(np.asarray(dense_production[2])[0]),
                    "sum_weight": float(dense_sum_weight),
                    "sum_weight_bits": _bits(dense_sum_weight),
                    "threshold": float(dense_threshold),
                    "threshold_bits": _bits(dense_threshold),
                    "probabilities_vs_direct_cuda": _exact_metric(
                        dense_expected_probs,
                        np.asarray(dense_production[0], dtype=np.float32).reshape(-1),
                    ),
                },
            }

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
        "cuda_relion_expf": cuda_expf,
        "cuda_dense_rectangular": cuda_dense_rectangular,
        "production_fine_helper": production_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-score", type=Path, required=True)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--adaptive-fraction", type=float, default=0.999)
    parser.add_argument("--recovar-capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.fine_score,
        args.factor,
        adaptive_fraction=args.adaptive_fraction,
        recovar_capture=args.recovar_capture,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
