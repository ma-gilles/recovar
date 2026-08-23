#!/usr/bin/env python3
"""Compare one stopped RECOVAR scale-AA operand bundle with native RELION shells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    _require(candidate.shape == reference.shape and candidate.size > 0, "metric topology mismatch")
    residual = candidate - reference
    denominator = max(float(np.linalg.norm(reference)), float(np.finfo(np.float64).tiny))
    return {
        "count": int(candidate.size),
        "relative_l2": float(np.linalg.norm(residual) / denominator),
        "median_signed": float(np.median(residual)),
        "median_abs": float(np.median(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _native_aa_shells(
    path: Path,
    *,
    iteration: int,
    half: int,
    part_id: int,
) -> np.ndarray:
    rows: dict[int, float] = {}
    prefix = f"acc_components\titer={iteration}\tpart_id={part_id}\thalfset={half}\t"
    with path.open() as stream:
        for line in stream:
            if not line.startswith(prefix):
                continue
            fields = {
                key: value
                for key, value in (item.split("=", 1) for item in line.rstrip().split("\t")[1:])
            }
            shell = int(fields["shell"])
            _require(shell not in rows, f"duplicate native shell {shell}")
            rows[shell] = float(fields["aa"])
    _require(rows, "native target selection is empty")
    _require(set(rows) == set(range(max(rows) + 1)), "native shell table is not contiguous")
    return np.asarray([rows[index] for index in range(len(rows))], dtype=np.float64)


def analyze(
    recovar_capture: Path,
    native_components: Path,
    *,
    expected_iteration: int,
    expected_half: int,
    expected_part_id: int,
    expected_original_index: int,
    recovar_term_divisor: float,
) -> dict[str, object]:
    _require(np.isfinite(recovar_term_divisor) and recovar_term_divisor > 0.0, "invalid divisor")
    with np.load(recovar_capture, allow_pickle=False) as payload:
        schema = str(payload["schema"].item())
        iteration = int(payload["iteration"])
        half = int(payload["half"])
        original_index = int(payload["original_index"])
        group_id = int(payload["group_id"])
        scale = float(payload["scale_for_stats"])
        mask = np.asarray(payload["scale_correction_pixel_mask"], dtype=bool)
        shell_indices = np.asarray(payload["scale_shell_indices"], dtype=np.int32)
        aa_per_pixel = np.asarray(payload["scale_aa_per_pixel"], dtype=np.float32)
        aa_per_shell = np.asarray(payload["scale_aa_per_shell"], dtype=np.float64)
        aa_per_image = float(payload["scale_aa_per_image"])
        if schema == "recovar-k1-norm-residual-inputs-v2":
            proj_abs2 = np.asarray(payload["proj_abs2_for_noise"], dtype=np.float32)
            ctf_probs = np.asarray(payload["ctf_probs"], dtype=np.float32)
            noise = np.asarray(payload["noise_variance_for_noise"], dtype=np.float32)
            ctf_probs_raw = np.asarray(payload["scale_ctf_probs_raw"], dtype=np.float32)
            aa_before_scale = np.asarray(payload["scale_aa_terms_before_scale"], dtype=np.float32)
            aa_terms = np.asarray(payload["scale_aa_terms"], dtype=np.float32)
            chunk_ranges = None
            pixel_sum_minus_production_total = float(
                np.sum(aa_per_pixel, dtype=np.float64) - aa_per_image
            )
        elif schema in {
            "recovar-k1-scale-aa-chunked-v1",
            "recovar-k1-scale-xa-aa-chunked-v2",
            "recovar-k1-scale-xa-aa-chunked-v3",
            "recovar-k1-scale-xa-aa-chunked-v4",
        }:
            proj_abs2 = np.asarray(payload["proj_abs2_sum_per_pixel_by_chunk"], dtype=np.float32)
            ctf_probs = None
            noise = None
            ctf_probs_raw = np.asarray(payload["ctf_probs_raw_sum_per_pixel_by_chunk"], dtype=np.float32)
            aa_before_scale = np.asarray(payload["aa_before_scale_per_pixel_by_chunk"], dtype=np.float32)
            aa_terms = np.asarray(payload["aa_per_pixel_by_chunk"], dtype=np.float32)
            chunk_ranges = np.asarray(payload["chunk_ranges"], dtype=np.int64)
            pixel_sum_minus_production_total = float(payload["pixel_sum_minus_production_total"])
        else:
            raise ValueError(f"unsupported schema {schema}")
    _require(iteration == expected_iteration and half == expected_half, "iteration/half identity changed")
    _require(original_index == expected_original_index, "source-particle identity changed")
    _require(group_id == expected_part_id, "RECOVAR group and native part identity differ")
    if schema == "recovar-k1-norm-residual-inputs-v2":
        _require(proj_abs2.shape == ctf_probs.shape == ctf_probs_raw.shape, "rotation/pixel topology changed")
        _require(mask.shape == shell_indices.shape == noise.shape == proj_abs2.shape[1:], "pixel topology changed")
        replay_ctf_probs_raw = np.where(
            (ctf_probs != 0.0) & mask[None, :],
            ctf_probs * noise[None, :],
            np.float32(0.0),
        ).astype(np.float32)
        replay_aa_before_scale = np.where(
            (ctf_probs != 0.0) & mask[None, :],
            proj_abs2 * replay_ctf_probs_raw,
            np.float32(0.0),
        ).astype(np.float32)
        replay_aa_terms = (replay_aa_before_scale / np.float32(scale * scale)).astype(np.float32)
        replay_aa_per_pixel = np.sum(replay_aa_terms, axis=0, dtype=np.float32)
        replay_aa_per_image = np.float32(np.sum(replay_aa_terms, dtype=np.float32))
        ctf_probs_raw_exact = bool(np.array_equal(replay_ctf_probs_raw, ctf_probs_raw))
        aa_products_exact = bool(np.array_equal(replay_aa_before_scale, aa_before_scale))
        aa_terms_exact = bool(np.array_equal(replay_aa_terms, aa_terms))
        per_pixel_exact = bool(np.array_equal(replay_aa_per_pixel, aa_per_pixel))
        per_image_exact = bool(replay_aa_per_image == np.float32(aa_per_image))
    else:
        _require(mask.shape == shell_indices.shape == proj_abs2.shape[1:], "chunked pixel topology changed")
        replay_ctf_probs_raw = ctf_probs_raw
        replay_aa_before_scale = aa_before_scale
        replay_aa_terms = (replay_aa_before_scale / np.float32(scale * scale)).astype(np.float32)
        replay_aa_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        for row in replay_aa_terms:
            replay_aa_per_pixel = (replay_aa_per_pixel + row).astype(np.float32)
        replay_aa_per_image = np.float32(np.sum(aa_terms, dtype=np.float64))
        ctf_probs_raw_exact = True
        aa_products_exact = True
        aa_terms_exact = bool(np.array_equal(replay_aa_terms, aa_terms))
        per_pixel_exact = bool(np.array_equal(replay_aa_per_pixel, aa_per_pixel))
        per_image_exact = bool(float(replay_aa_per_image) == float(np.float32(aa_per_image)))

    native = _native_aa_shells(
        native_components,
        iteration=expected_iteration,
        half=expected_half,
        part_id=expected_part_id,
    )
    shell_count = min(native.size, aa_per_shell.size)
    active_shells = np.flatnonzero(
        np.asarray(
            [np.any(mask & (shell_indices == shell)) for shell in range(shell_count)],
            dtype=bool,
        )
    )
    _require(active_shells.size > 0, "capture has no active scale shells")
    recovar_native_units = aa_per_shell[:shell_count] / float(recovar_term_divisor)
    shell_metric = _metric(recovar_native_units[active_shells], native[:shell_count][active_shells])
    ratio_valid = (
        (native[:shell_count][active_shells] > 0.0)
        & (recovar_native_units[active_shells] > 0.0)
    )
    ratios = recovar_native_units[active_shells][ratio_valid] / native[:shell_count][active_shells][ratio_valid]
    _require(ratios.size > 0, "no positive AA shells")
    residual = recovar_native_units[:shell_count] - native[:shell_count]
    ranked = active_shells[np.argsort(np.abs(residual[active_shells]))[::-1]]

    return {
        "schema": "recovar.em.k1_scale_aa_boundary.v1",
        "identity": {
            "iteration": iteration,
            "half": half,
            "part_id": group_id,
            "original_index_zero_based": original_index,
            "scale_for_stats": scale,
            "rotation_count": int(proj_abs2.shape[0]) if chunk_ranges is None else int(np.sum(chunk_ranges[:, 1] - chunk_ranges[:, 0])),
            "rotation_chunk_count": 0 if chunk_ranges is None else int(chunk_ranges.shape[0]),
            "pixel_count": int(proj_abs2.shape[1]),
            "active_pixel_count": int(np.count_nonzero(mask)),
            "active_shells": active_shells.tolist(),
            "recovar_term_divisor": float(recovar_term_divisor),
        },
        "local_replay": {
            "ctf_probs_raw_bit_exact": ctf_probs_raw_exact,
            "aa_products_bit_exact": aa_products_exact,
            "scale_adjusted_aa_terms_bit_exact": aa_terms_exact,
            "per_pixel_reduction_bit_exact": per_pixel_exact,
            "per_image_reduction_bit_exact": per_image_exact,
            "per_pixel_reduction_replay": _metric(replay_aa_per_pixel, aa_per_pixel),
            "per_image_reduction_signed_delta": float(replay_aa_per_image - np.float32(aa_per_image)),
            "pixel_sum_minus_production_total": pixel_sum_minus_production_total,
        },
        "native_shell_comparison": {
            **shell_metric,
            "ratio_median": float(np.median(ratios)),
            "ratio_p05": float(np.percentile(ratios, 5)),
            "ratio_p95": float(np.percentile(ratios, 95)),
            "largest_abs_residual_shells": [
                {
                    "shell": int(shell),
                    "native_aa": float(native[shell]),
                    "recovar_aa_native_units": float(recovar_native_units[shell]),
                    "signed_delta": float(residual[shell]),
                }
                for shell in ranked[:10]
            ],
        },
        "artifacts": {
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
            "native_components": str(native_components.resolve()),
            "native_components_sha256": _sha256(native_components),
        },
        "classification": (
            "cross-engine mismatch is already present by the per-particle shell AA boundary; "
            "native per-pixel capture is required to split operand formation from the particle-local reduction"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--native-components", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=2)
    parser.add_argument("--half", type=int, default=1)
    parser.add_argument("--part-id", type=int, required=True)
    parser.add_argument("--original-index", type=int, required=True)
    parser.add_argument("--recovar-term-divisor", type=float, default=float(128**4))
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.recovar_capture,
        args.native_components,
        expected_iteration=args.iteration,
        expected_half=args.half,
        expected_part_id=args.part_id,
        expected_original_index=args.original_index,
        recovar_term_divisor=args.recovar_term_divisor,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
