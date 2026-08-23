#!/usr/bin/env python3
"""Compare a RECOVAR K=1 scale-state dump with a RELION model boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import starfile


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
    absolute = np.abs(residual)
    return {
        "count": int(candidate.size),
        "relative_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(reference), np.finfo(float).tiny)),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "max_abs": float(np.max(absolute)),
        "count_abs_gt_1e-4": int(np.count_nonzero(absolute > 1e-4)),
        "count_abs_gt_1e-3": int(np.count_nonzero(absolute > 1e-3)),
    }


def _model_groups(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    model = starfile.read(path)
    groups = model["model_groups"]
    numbers = np.asarray(groups["rlnGroupNumber"], dtype=np.int64)
    _require(np.array_equal(numbers, np.arange(1, numbers.size + 1)), "RELION group numbers are not contiguous")
    counts = np.asarray(groups["rlnGroupNrParticles"], dtype=np.int64)
    scales = np.asarray(groups["rlnGroupScaleCorrection"], dtype=np.float64)
    names = np.asarray(groups["rlnGroupName"], dtype=str)
    general = model["model_general"]
    if isinstance(general, dict):
        original_size = int(general["rlnOriginalImageSize"])
    else:
        original_size = int(general.iloc[0]["rlnOriginalImageSize"])
    return counts, scales, names, original_size


def _replay_scale_update(xa: np.ndarray, aa: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray | float]:
    raw = np.ones_like(xa, dtype=np.float64)
    np.divide(xa, aa, out=raw, where=aa > 0.0)
    median = float(np.sort(raw)[raw.size // 2])
    _require(np.isfinite(median) and median > 0.0, "invalid scale median")
    clipped = np.clip(raw, median / 5.0, median * 5.0)
    count_sum = float(np.sum(counts, dtype=np.float64))
    _require(count_sum > 0.0, "scale update contains no particles")
    average = float(np.sum(counts.astype(np.float64) * clipped, dtype=np.float64) / count_sum)
    _require(np.isfinite(average) and average > 0.0, "invalid scale normalization average")
    return {
        "raw": raw,
        "clipped": clipped,
        "median": median,
        "average": average,
        "final": clipped / average,
    }


def analyze(
    parity_dump: Path,
    relion_input_models: list[Path],
    relion_output_models: list[Path],
    *,
    top_count: int = 20,
) -> dict[str, object]:
    _require(len(relion_input_models) == len(relion_output_models) == 2, "exactly two half models are required")
    with np.load(parity_dump, allow_pickle=False) as payload:
        dump = {key: np.asarray(payload[key]) for key in payload.files}

    halves = []
    target_ids: set[int] = set()
    for half in (1, 2):
        input_counts, input_scales, input_names, original_size = _model_groups(relion_input_models[half - 1])
        output_counts, output_scales, output_names, output_size = _model_groups(relion_output_models[half - 1])
        _require(original_size == output_size, f"half {half} model size changed")
        _require(np.array_equal(input_names, output_names), f"half {half} group identity changed")
        _require(np.array_equal(input_counts, output_counts), f"half {half} group counts changed")
        prefix = f"half{half}"
        xa = np.asarray(dump[f"{prefix}_wsum_scale_correction_xa"], dtype=np.float64)
        aa = np.asarray(dump[f"{prefix}_wsum_scale_correction_aa"], dtype=np.float64)
        counts = np.asarray(dump[f"{prefix}_group_particle_counts"], dtype=np.int64)
        scales = np.asarray(dump[f"{prefix}_group_scale_corrections"], dtype=np.float64)
        for label, values in (("XA", xa), ("AA", aa), ("counts", counts), ("scales", scales)):
            _require(values.shape == output_scales.shape, f"half {half} {label} topology changed: {values.shape}")
        _require(np.array_equal(counts, output_counts), f"half {half} cross-engine group counts differ")
        active = counts > 0
        replay = _replay_scale_update(xa, aa, counts)
        residual = scales - output_scales
        active_indices = np.flatnonzero(active)
        ranked = active_indices[np.argsort(np.abs(residual[active]))[::-1][: int(top_count)]]
        divisor = float(original_size**4)

        rows = []
        for index in ranked:
            index = int(index)
            target_ids.add(index)
            rows.append(
                {
                    "part_and_group_id_zero_based": index,
                    "group_name": str(output_names[index]),
                    "particle_count": int(counts[index]),
                    "input_scale": float(input_scales[index]),
                    "recovar_xa": float(xa[index]),
                    "recovar_aa": float(aa[index]),
                    "recovar_xa_native_units": float(xa[index] / divisor),
                    "recovar_aa_native_units": float(aa[index] / divisor),
                    "recovar_raw": float(np.asarray(replay["raw"])[index]),
                    "recovar_scale_replay": float(np.asarray(replay["final"])[index]),
                    "recovar_dump_scale": float(scales[index]),
                    "relion_output_scale": float(output_scales[index]),
                    "scale_residual": float(residual[index]),
                }
            )

        halves.append(
            {
                "half": half,
                "particle_count": int(np.sum(counts)),
                "active_group_count": int(np.count_nonzero(active)),
                "original_size": original_size,
                "native_unit_divisor": divisor,
                "recovar_update_median": float(replay["median"]),
                "recovar_update_normalization_average": float(replay["average"]),
                "comparisons": {
                    "recovar_replay_vs_dump": _metric(np.asarray(replay["final"])[active], scales[active]),
                    "recovar_dump_vs_relion_output": _metric(scales[active], output_scales[active]),
                },
                "largest_scale_residuals": rows,
            }
        )

    artifacts = {"parity_dump": str(parity_dump.resolve()), "parity_dump_sha256": _sha256(parity_dump)}
    for half in (1, 2):
        for label, path in (
            ("input", relion_input_models[half - 1]),
            ("output", relion_output_models[half - 1]),
        ):
            artifacts[f"relion_half{half}_{label}_model"] = str(path.resolve())
            artifacts[f"relion_half{half}_{label}_model_sha256"] = _sha256(path)
    return {
        "schema": "recovar.em.k1_scale_state_boundary.v1",
        "halves": halves,
        "native_capture_part_ids_csv": ",".join(str(index) for index in sorted(target_ids)),
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-dump", type=Path, required=True)
    parser.add_argument("--relion-input-half1-model", type=Path, required=True)
    parser.add_argument("--relion-input-half2-model", type=Path, required=True)
    parser.add_argument("--relion-output-half1-model", type=Path, required=True)
    parser.add_argument("--relion-output-half2-model", type=Path, required=True)
    parser.add_argument("--top-count", type=int, default=20)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.parity_dump,
        [args.relion_input_half1_model, args.relion_input_half2_model],
        [args.relion_output_half1_model, args.relion_output_half2_model],
        top_count=args.top_count,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
