#!/usr/bin/env python3
"""Replay one captured RELION Wavg A2 panel with RECOVAR's CUDA atomic reducer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.cuda_backproject import (
    relion_wavg_rotation_atomic_add_f32,
    relion_wavg_rotation_atomic_f32,
)
from scripts.analyze_k1_bpref_contributor_membership import match_rotations
from scripts.analyze_k1_projected_power_boundary import _native_probabilities, _raw_f32
from scripts.analyze_k1_scale_aa_candidates import _metric, _real, _scalar, _sha256
from scripts.analyze_k1_wavg_accumulation_boundary import (
    _flat_int,
    _native_pixel_aa,
    _translation_loop_mass,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def replay(
    native_directory: Path,
    native_pixels: Path,
    *,
    recovar_capture: Path | None,
    native_prefix: str,
    iteration: int,
    half: int,
    part_id: int,
) -> dict[str, object]:
    _require(jax.default_backend() == "gpu", "Wavg atomic replay requires a GPU")
    prefix = native_directory / native_prefix
    native_probabilities, native_rotations, _ = _native_probabilities(prefix)
    orientation_num, translation_num = native_probabilities.shape
    raw = _real(Path(f"{prefix}sorted_weights.bin")).astype(np.float32).reshape(
        orientation_num,
        translation_num,
    )
    sum_weight = np.float32(_scalar(Path(f"{prefix}sum_weight.bin")))
    threshold = np.float32(_scalar(Path(f"{prefix}significant_weight.bin")))
    rotation_mass = _translation_loop_mass(raw, sum_weight, threshold)
    panel_pixels = _flat_int(Path(f"{prefix}project_panel_pixels.bin")).astype(np.int64)
    panel_size = panel_pixels.size
    real = _raw_f32(Path(f"{prefix}project_panel_ref_real.f32")).reshape(
        orientation_num,
        panel_size,
    )
    imag = _raw_f32(Path(f"{prefix}project_panel_ref_imag.f32")).reshape(
        orientation_num,
        panel_size,
    )
    ctf = _real(Path(f"{prefix}ctfs.bin")).astype(np.float32)[panel_pixels]
    scaled_real = (real * ctf[None, :]).astype(np.float32)
    scaled_imag = (imag * ctf[None, :]).astype(np.float32)
    power = (scaled_real * scaled_real + scaled_imag * scaled_imag).astype(np.float32)
    terms = (power * rotation_mass[:, None]).astype(np.float32)

    atomic = np.asarray(
        jax.block_until_ready(relion_wavg_rotation_atomic_f32(terms[None, ...]))[0],
        dtype=np.float32,
    ).astype(np.float64)
    chunked_accumulator = jnp.zeros((1, panel_size), dtype=jnp.float32)
    for start in range(0, orientation_num, 8192):
        chunked_accumulator = relion_wavg_rotation_atomic_add_f32(
            terms[None, start : start + 8192],
            chunked_accumulator,
        )
    chunked_atomic = np.asarray(
        jax.block_until_ready(chunked_accumulator)[0],
        dtype=np.float32,
    ).astype(np.float64)
    native_by_pixel = _native_pixel_aa(
        native_pixels,
        iteration=iteration,
        half=half,
        part_id=part_id,
    )
    native = np.asarray([native_by_pixel[int(pixel)] for pixel in panel_pixels], dtype=np.float64)

    recovar_order_atomic = None
    rotation_order_report = None
    recovar_capture_sha256 = None
    if recovar_capture is not None:
        with np.load(recovar_capture, allow_pickle=False) as payload:
            recovar_probabilities = np.asarray(
                payload["candidate_posterior_probs"],
                dtype=np.float32,
            )
            recovar_rotations = np.asarray(
                payload["candidate_rotation_matrices"],
                dtype=np.float32,
            )
        recovar_rotation_mass = np.sum(recovar_probabilities, axis=1, dtype=np.float64)
        native_active = np.flatnonzero(rotation_mass > 0.0)
        recovar_active = np.flatnonzero(recovar_rotation_mass > 0.0)
        matches = match_rotations(
            native_rotations[native_active],
            recovar_rotations[recovar_active],
            tolerance=0.0,
        )
        _require(matches.relion_unmatched.size == 0, "native active rotation is unmatched")
        _require(matches.recovar_unmatched.size == 0, "RECOVAR active rotation is unmatched")
        matched_native = native_active[matches.pairs[:, 0]]
        matched_recovar = recovar_active[matches.pairs[:, 1]]
        recovar_order_terms = np.zeros(
            (recovar_probabilities.shape[0], panel_size),
            dtype=np.float32,
        )
        recovar_order_terms[matched_recovar] = terms[matched_native]
        recovar_order_accumulator = jnp.zeros((1, panel_size), dtype=jnp.float32)
        for start in range(0, recovar_order_terms.shape[0], 8192):
            recovar_order_accumulator = relion_wavg_rotation_atomic_add_f32(
                recovar_order_terms[None, start : start + 8192],
                recovar_order_accumulator,
            )
        recovar_order_atomic = np.asarray(
            jax.block_until_ready(recovar_order_accumulator)[0],
            dtype=np.float32,
        ).astype(np.float64)

        native_rank_at_recovar_rank = np.empty(recovar_active.size, dtype=np.int64)
        native_rank_at_recovar_rank[matches.pairs[:, 1]] = matches.pairs[:, 0]
        expected_rank = np.arange(native_rank_at_recovar_rank.size, dtype=np.int64)
        rotation_order_report = {
            "native_active_count": int(native_active.size),
            "recovar_active_count": int(recovar_active.size),
            "native_orientation_count": int(orientation_num),
            "recovar_orientation_count": int(recovar_probabilities.shape[0]),
            "fixed_active_positions": int(
                np.count_nonzero(native_rank_at_recovar_rank == expected_rank)
            ),
            "max_active_rank_displacement": int(
                np.max(np.abs(native_rank_at_recovar_rank - expected_rank), initial=0)
            ),
            "active_rank_correlation": float(
                np.corrcoef(expected_rank, native_rank_at_recovar_rank)[0, 1]
            ),
            "recovar_order_native_terms_vs_native": _metric(recovar_order_atomic, native),
        }
        recovar_capture_sha256 = _sha256(recovar_capture)
    float64_sum = np.sum(terms, axis=0, dtype=np.float64)
    forward_float32 = np.cumsum(terms, axis=0, dtype=np.float32)[-1].astype(np.float64)
    baseline_norm = float(np.linalg.norm(float64_sum - native))
    atomic_norm = float(np.linalg.norm(atomic - native))
    chunked_atomic_norm = float(np.linalg.norm(chunked_atomic - native))

    return {
        "schema": "recovar.em.k1_wavg_atomic_replay.v1",
        "identity": {
            "iteration": iteration,
            "half": half,
            "part_id": part_id,
            "orientation_count": orientation_num,
            "translation_count": translation_num,
            "panel_pixel_count": panel_size,
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "replay": {
            "high_accuracy_sum_vs_native": _metric(float64_sum, native),
            "forward_float32_sum_vs_native": _metric(forward_float32, native),
            "cuda_atomic_vs_native": _metric(atomic, native),
            "chunked_cuda_atomic_vs_native": _metric(chunked_atomic, native),
            "cuda_atomic_residual_closure_fraction": (
                1.0 - atomic_norm / baseline_norm if baseline_norm > 0.0 else 0.0
            ),
            "chunked_cuda_atomic_residual_closure_fraction": (
                1.0 - chunked_atomic_norm / baseline_norm if baseline_norm > 0.0 else 0.0
            ),
            "recovar_rotation_order": rotation_order_report,
            "recovar_order_native_terms_residual_closure_fraction": (
                None
                if recovar_order_atomic is None
                else 1.0
                - float(np.linalg.norm(recovar_order_atomic - native)) / baseline_norm
                if baseline_norm > 0.0
                else 0.0
            ),
        },
        "artifacts": {
            "native_directory": str(native_directory.resolve()),
            "native_pixels": str(native_pixels.resolve()),
            "native_pixels_sha256": _sha256(native_pixels),
            "recovar_capture": (
                None if recovar_capture is None else str(recovar_capture.resolve())
            ),
            "recovar_capture_sha256": recovar_capture_sha256,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--native-pixels", type=Path, required=True)
    parser.add_argument("--recovar-capture", type=Path)
    parser.add_argument("--native-prefix", default="img0_part109_storeWavg_")
    parser.add_argument("--iteration", type=int, default=2)
    parser.add_argument("--half", type=int, default=1)
    parser.add_argument("--part-id", type=int, default=109)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = replay(
        args.native_directory,
        args.native_pixels,
        recovar_capture=args.recovar_capture,
        native_prefix=args.native_prefix,
        iteration=args.iteration,
        half=args.half,
        part_id=args.part_id,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
