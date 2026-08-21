#!/usr/bin/env python3
"""Compare stopped RELION and RECOVAR K=1 group-scale update terms."""

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
    return {
        "count": int(candidate.size),
        "relative_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(reference), np.finfo(float).tiny)),
        "median_abs": float(np.median(np.abs(residual))),
        "p95_abs": float(np.percentile(np.abs(residual), 95)),
        "max_abs": float(np.max(np.abs(residual))),
    }


def _positive_ratio(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    _require(candidate.shape == reference.shape, "ratio topology mismatch")
    valid = np.isfinite(candidate) & np.isfinite(reference) & (candidate > 0.0) & (reference > 0.0)
    _require(np.any(valid), "ratio has no positive finite entries")
    ratio = candidate[valid] / reference[valid]
    return {
        "count": int(ratio.size),
        "median": float(np.median(ratio)),
        "p05": float(np.percentile(ratio, 5)),
        "p95": float(np.percentile(ratio, 95)),
        "min": float(np.min(ratio)),
        "max": float(np.max(ratio)),
    }


def _parse_model_groups(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: dict[int, tuple[str, int, float]] = {}
    in_groups = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "data_model_groups":
            in_groups = True
            continue
        if not in_groups or not stripped or stripped.startswith(("#", "loop_", "_")):
            continue
        fields = stripped.split()
        if len(fields) != 4 or not fields[0].isdigit():
            if rows:
                break
            continue
        group_index = int(fields[0]) - 1
        _require(group_index not in rows, f"duplicate model group {group_index}")
        rows[group_index] = (fields[1], int(fields[2]), float(fields[3]))
    _require(rows and set(rows) == set(range(len(rows))), "model group table is not contiguous")
    names = [rows[index][0] for index in range(len(rows))]
    counts = np.asarray([rows[index][1] for index in range(len(rows))], dtype=np.int64)
    scales = np.asarray([rows[index][2] for index in range(len(rows))], dtype=np.float64)
    return counts, scales, names


def _parse_dvp_shells(path: Path, *, threshold: float) -> np.ndarray:
    selected: list[int] = []
    in_class = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "data_model_class_1":
            in_class = True
            continue
        if not in_class or not stripped or stripped.startswith(("#", "loop_", "_")):
            continue
        fields = stripped.split()
        if len(fields) != 8 or not fields[0].isdigit():
            if selected:
                break
            continue
        if float(fields[3]) > threshold:
            selected.append(int(fields[0]))
    _require(selected, "model contains no scale-correction shells")
    return np.asarray(selected, dtype=np.int64)


def _parse_native_terms(
    path: Path,
    *,
    iteration: int,
    half: int,
    n_groups: int,
    selected_shells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xa = np.zeros(n_groups, dtype=np.float64)
    aa = np.zeros(n_groups, dtype=np.float64)
    seen = np.zeros(n_groups, dtype=bool)
    shell_set = set(int(value) for value in selected_shells)
    prefix = f"acc_components\titer={iteration}\t"
    half_token = f"\thalfset={half}\t"
    with path.open() as stream:
        for line in stream:
            if not line.startswith(prefix) or half_token not in line:
                continue
            fields = {key: value for key, value in (item.split("=", 1) for item in line.rstrip().split("\t")[1:])}
            group_index = int(fields["part_id"])
            _require(0 <= group_index < n_groups, f"native particle/group index outside [0,{n_groups})")
            seen[group_index] = True
            if int(fields["shell"]) in shell_set:
                xa[group_index] += float(fields["xa"])
                aa[group_index] += float(fields["aa"])
    _require(np.any(seen), "native component selection is empty")
    return xa, aa, seen


def _update_from_terms(xa: np.ndarray, aa: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray | float]:
    _require(xa.shape == aa.shape == counts.shape and xa.size > 0, "scale update topology mismatch")
    raw = np.ones_like(xa, dtype=np.float64)
    np.divide(xa, aa, out=raw, where=aa > 0.0)
    median = float(np.sort(raw)[raw.size // 2])
    _require(np.isfinite(median) and median > 0.0, "invalid scale median")
    clipped = np.clip(raw, median / 5.0, 5.0 * median)
    count_sum = float(np.sum(counts, dtype=np.float64))
    _require(count_sum > 0.0, "scale update has no particles")
    average = float(np.sum(counts.astype(np.float64) * clipped) / count_sum)
    _require(np.isfinite(average) and average > 0.0, "invalid scale normalization average")
    return {"raw": raw, "clipped": clipped, "median": median, "average": average, "final": clipped / average}


def analyze(
    native_components: Path,
    native_input_model: Path,
    native_output_model: Path,
    recovar_parity: Path,
    *,
    iteration: int,
    half: int,
    target_group_index: int,
    threshold: float = 3.0,
    top_count: int = 20,
    recovar_term_divisor: float = 1.0,
) -> dict[str, object]:
    native_counts, native_model_scales, native_group_names = _parse_model_groups(native_output_model)
    selected_shells = _parse_dvp_shells(native_input_model, threshold=threshold)
    n_groups = int(native_counts.size)
    native_xa, native_aa, native_seen = _parse_native_terms(
        native_components,
        iteration=iteration,
        half=half,
        n_groups=n_groups,
        selected_shells=selected_shells,
    )
    _require(np.array_equal(native_seen.astype(np.int64), native_counts), "native part/group identity changed")
    native = _update_from_terms(native_xa, native_aa, native_counts)

    with np.load(recovar_parity, allow_pickle=False) as payload:
        prefix = f"half{half}"
        recovar_xa = np.asarray(payload[f"{prefix}_wsum_scale_correction_xa"], dtype=np.float64)
        recovar_aa = np.asarray(payload[f"{prefix}_wsum_scale_correction_aa"], dtype=np.float64)
        recovar_counts = np.asarray(payload[f"{prefix}_group_particle_counts"], dtype=np.int64)
        recovar_group_scales = np.asarray(payload[f"{prefix}_group_scale_corrections"], dtype=np.float64)
        recovar_dvp = np.asarray(payload["scale_correction_data_vs_prior"], dtype=np.float64)
    for label, values in (
        ("RECOVAR XA", recovar_xa),
        ("RECOVAR AA", recovar_aa),
        ("RECOVAR counts", recovar_counts),
        ("RECOVAR scales", recovar_group_scales),
    ):
        _require(values.shape == (n_groups,), f"{label} group topology changed: {values.shape}")
    _require(np.array_equal(recovar_counts, native_counts), "cross-engine group counts differ")
    recovar_shells = np.flatnonzero(recovar_dvp > threshold)
    _require(np.array_equal(recovar_shells, selected_shells), "cross-engine scale shell mask differs")
    recovar = _update_from_terms(recovar_xa, recovar_aa, recovar_counts)
    _require(np.isfinite(recovar_term_divisor) and recovar_term_divisor > 0.0, "invalid RECOVAR term divisor")
    recovar_xa_native_units = recovar_xa / float(recovar_term_divisor)
    recovar_aa_native_units = recovar_aa / float(recovar_term_divisor)

    target = int(target_group_index)
    _require(0 <= target < n_groups, "target group index is out of range")
    active = native_counts > 0
    unclipped = active & (
        (np.asarray(native["raw"]) >= float(native["median"]) / 5.0)
        & (np.asarray(native["raw"]) <= float(native["median"]) * 5.0)
        & (np.asarray(recovar["raw"]) >= float(recovar["median"]) / 5.0)
        & (np.asarray(recovar["raw"]) <= float(recovar["median"]) * 5.0)
    )
    normalization_contribution = (
        recovar_counts.astype(np.float64)
        * (np.asarray(recovar["clipped"]) - np.asarray(native["clipped"]))
        / float(np.sum(recovar_counts))
    )
    ranked = np.argsort(np.abs(normalization_contribution))[::-1][: int(top_count)]

    def group_row(index: int) -> dict[str, float | int | bool]:
        return {
            "group_index_zero_based": int(index),
            "group_name": native_group_names[index],
            "particle_count": int(native_counts[index]),
            "native_xa": float(native_xa[index]),
            "native_aa": float(native_aa[index]),
            "native_raw": float(np.asarray(native["raw"])[index]),
            "native_clipped": float(np.asarray(native["clipped"])[index]),
            "native_final_replay": float(np.asarray(native["final"])[index]),
            "native_model_scale": float(native_model_scales[index]),
            "recovar_xa": float(recovar_xa[index]),
            "recovar_aa": float(recovar_aa[index]),
            "recovar_xa_native_units": float(recovar_xa_native_units[index]),
            "recovar_aa_native_units": float(recovar_aa_native_units[index]),
            "recovar_raw": float(np.asarray(recovar["raw"])[index]),
            "recovar_clipped": float(np.asarray(recovar["clipped"])[index]),
            "recovar_final_replay": float(np.asarray(recovar["final"])[index]),
            "recovar_dump_scale": float(recovar_group_scales[index]),
            "normalization_average_delta_contribution": float(normalization_contribution[index]),
            "both_unclipped": bool(unclipped[index]),
        }

    return {
        "schema": "recovar.em.k1_scale_update_terms.v1",
        "identity": {
            "iteration": iteration,
            "half": half,
            "group_count": n_groups,
            "particle_count": int(np.sum(native_counts)),
            "target_group_index_zero_based": target,
            "data_vs_prior_threshold": threshold,
            "selected_shells": selected_shells.tolist(),
            "recovar_term_divisor": float(recovar_term_divisor),
        },
        "update": {
            "native_median": float(native["median"]),
            "recovar_median": float(recovar["median"]),
            "native_normalization_average": float(native["average"]),
            "recovar_normalization_average": float(recovar["average"]),
            "recovar_over_native_normalization_average": float(recovar["average"] / native["average"]),
        },
        "comparisons": {
            "native_replay_vs_model_scale": _metric(np.asarray(native["final"]), native_model_scales),
            "recovar_replay_vs_dump_scale": _metric(np.asarray(recovar["final"]), recovar_group_scales),
            "active_raw_ratio_recovar_vs_native": _metric(
                np.asarray(recovar["raw"])[active], np.asarray(native["raw"])[active]
            ),
            "both_unclipped_raw_ratio_recovar_vs_native": _metric(
                np.asarray(recovar["raw"])[unclipped], np.asarray(native["raw"])[unclipped]
            ),
            "final_scale_recovar_vs_native": _metric(recovar_group_scales[active], native_model_scales[active]),
            "active_xa_recovar_vs_native": _metric(
                recovar_xa_native_units[active], native_xa[active]
            ),
            "active_aa_recovar_vs_native": _metric(
                recovar_aa_native_units[active], native_aa[active]
            ),
            "active_xa_recovar_over_native": _positive_ratio(
                recovar_xa_native_units[active], native_xa[active]
            ),
            "active_aa_recovar_over_native": _positive_ratio(
                recovar_aa_native_units[active], native_aa[active]
            ),
        },
        "target": group_row(target),
        "largest_normalization_contributors": [group_row(int(index)) for index in ranked if native_counts[index] > 0],
        "artifacts": {
            "native_components": str(native_components.resolve()),
            "native_components_sha256": _sha256(native_components),
            "native_input_model": str(native_input_model.resolve()),
            "native_input_model_sha256": _sha256(native_input_model),
            "native_output_model": str(native_output_model.resolve()),
            "native_output_model_sha256": _sha256(native_output_model),
            "recovar_parity": str(recovar_parity.resolve()),
            "recovar_parity_sha256": _sha256(recovar_parity),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-components", type=Path, required=True)
    parser.add_argument("--native-input-model", type=Path, required=True)
    parser.add_argument("--native-output-model", type=Path, required=True)
    parser.add_argument("--recovar-parity", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--half", type=int, choices=(1, 2), required=True)
    parser.add_argument("--target-group-index", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-count", type=int, default=20)
    parser.add_argument("--recovar-term-divisor", type=float, default=1.0)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_components,
        args.native_input_model,
        args.native_output_model,
        args.recovar_parity,
        iteration=args.iteration,
        half=args.half,
        target_group_index=args.target_group_index,
        top_count=args.top_count,
        recovar_term_divisor=args.recovar_term_divisor,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
