#!/usr/bin/env python3
"""Compare source-ID-aligned production BPref launch deltas in RELION and RECOVAR."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


_NATIVE_PATTERN = re.compile(
    r"rank(?P<rank>-?\d+)_thread(?P<thread>\d+)_part(?P<part>\d+)_"
    r"stack(?P<stack>\d+)_class(?P<class>\d+)_bpref_shadow_metadata\.bin$"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_counted(path: Path, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    _require(path.is_file(), f"missing native accumulator artifact: {path}")
    with path.open("rb") as stream:
        count_values = np.fromfile(stream, dtype=np.uint64, count=1)
        _require(count_values.size == 1, f"truncated native count: {path}")
        count = int(count_values[0])
        values = np.fromfile(stream, dtype=dtype)
    _require(values.size == count, f"native count mismatch in {path}")
    return values


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate).reshape(-1)
    reference = np.asarray(reference).reshape(-1)
    _require(candidate.shape == reference.shape and candidate.size > 0, "metric topology mismatch")
    _require(np.all(np.isfinite(candidate)) and np.all(np.isfinite(reference)), "nonfinite metric")
    metric_dtype = np.complex128 if (
        np.iscomplexobj(candidate) or np.iscomplexobj(reference)
    ) else np.float64
    candidate = candidate.astype(metric_dtype, copy=False)
    reference = reference.astype(metric_dtype, copy=False)
    residual = candidate - reference
    reference_norm = max(float(np.linalg.norm(reference)), np.finfo(np.float64).tiny)
    reference_energy = max(float(np.vdot(reference, reference).real), np.finfo(np.float64).tiny)
    scale = float(np.vdot(reference, candidate).real / reference_energy)
    candidate_support = candidate != 0
    reference_support = reference != 0
    support_union = int(np.count_nonzero(candidate_support | reference_support))
    return {
        "relative_l2": float(np.linalg.norm(residual) / reference_norm),
        "max_absolute": float(np.max(np.abs(residual), initial=0.0)),
        "candidate_to_reference_scale": scale,
        "relative_l2_after_reference_scale": float(
            np.linalg.norm(candidate - scale * reference) / reference_norm
        ),
        "support_mismatch_count": int(np.count_nonzero(candidate_support != reference_support)),
        "support_jaccard": float(
            np.count_nonzero(candidate_support & reference_support) / max(support_union, 1)
        ),
        "candidate_l2": float(np.linalg.norm(candidate)),
        "reference_l2": float(np.linalg.norm(reference)),
    }


def _native_prefix(metadata_path: Path) -> Path:
    suffix = "shadow_metadata.bin"
    _require(metadata_path.name.endswith(suffix), f"unexpected metadata name: {metadata_path}")
    return metadata_path.with_name(metadata_path.name[: -len(suffix)] + "shadow_")


def _load_native_particle(metadata_path: Path, *, grid_size: int) -> dict[str, Any]:
    prefix = _native_prefix(metadata_path)
    metadata = _load_counted(Path(f"{prefix}metadata.bin"), np.uint64)
    _require(metadata.size == 18 and int(metadata[0]) == 1, "native metadata schema changed")
    count = int(metadata[14])
    real = _load_counted(Path(f"{prefix}real.bin"), np.float32)
    imag = _load_counted(Path(f"{prefix}imag.bin"), np.float32)
    weight = _load_counted(Path(f"{prefix}weight.bin"), np.float32)
    _require(real.size == imag.size == weight.size == count, "native accumulator size changed")
    data = real.astype(np.complex64)
    data.imag = imag
    n2 = float(grid_size) ** 2
    n4 = float(grid_size) ** 4
    return {
        "iteration": int(metadata[1]),
        "part_id": int(metadata[2]),
        "stack_index_1based": int(metadata[3]),
        "rank": int(metadata[4]),
        "thread": int(metadata[5]),
        "class_one_based": int(metadata[6]),
        "native_shape": [
            int(metadata[9]),
            int(metadata[8]),
            int(metadata[7]),
        ],
        "native_starts": [int(metadata[11]), int(metadata[10]), 0],
        "max_r": int(metadata[12]),
        "max_r2": int(metadata[13]),
        "count": count,
        # RECOVAR's numerator convention is the negative of native BPref data.
        "particle_data": -data.astype(np.complex128) / n2,
        "particle_weight": weight.astype(np.float64) / n4,
        "metadata_path": metadata_path,
    }


def _load_recovar_particle(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        _require(
            archive["schema"].item() == "recovar-bpref-accumulator-delta-v2",
            f"unexpected RECOVAR schema in {path}",
        )
        return {name: archive[name] for name in archive.files}


def _compare_particle(native: dict[str, Any], recovar: dict[str, Any]) -> dict[str, Any]:
    original_index = int(recovar["original_index"])
    _require(
        native["stack_index_1based"] == original_index + 1,
        "native/RECOVAR immutable stack identity mismatch",
    )
    _require(native["iteration"] == int(recovar["iteration"]) == 1, "iteration mismatch")
    _require(native["count"] == int(recovar["flat_accumulator_size"]), "flat size mismatch")
    _require(
        tuple(native["native_shape"])
        == (
            int(recovar["volume_shape"][0]),
            int(recovar["volume_shape"][1]),
            int(recovar["volume_shape"][2]) // 2 + 1,
        ),
        "native/RECOVAR x-half shape mismatch",
    )

    rec_before_data = np.asarray(recovar["before_data"], dtype=np.complex64)
    rec_after_data = np.asarray(recovar["after_data"], dtype=np.complex64)
    rec_before_weight = np.asarray(recovar["before_weight"], dtype=np.float32)
    rec_after_weight = np.asarray(recovar["after_weight"], dtype=np.float32)
    rec_delta_data = np.asarray(recovar["isolated_data"], dtype=np.complex64)
    rec_delta_weight = np.asarray(recovar["isolated_weight"], dtype=np.float32)
    return {
        "half": int(recovar["half"]),
        "original_index": original_index,
        "stack_index_1based": native["stack_index_1based"],
        "native_part_id": native["part_id"],
        "native_rank": native["rank"],
        "native_thread": native["thread"],
        "recovar_particle_launch_ordinal_within_bucket": int(
            recovar["particle_launch_ordinal"]
        ),
        "particle_rotation_count": int(recovar["particle_rotation_count"]),
        "particle_delta": {
            "numerator": _metric(rec_delta_data, native["particle_data"]),
            "denominator": _metric(rec_delta_weight, native["particle_weight"]),
        },
        "production_increment": {
            "numerator": _metric(
                rec_after_data - rec_before_data, native["particle_data"]
            ),
            "denominator": _metric(
                rec_after_weight - rec_before_weight, native["particle_weight"]
            ),
        },
        "recovar_prefix_data_l2": float(np.linalg.norm(rec_before_data)),
        "recovar_prefix_weight_sum": float(np.sum(rec_before_weight, dtype=np.float64)),
        "native_particle_weight_sum": float(np.sum(native["particle_weight"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory-half1", type=Path, required=True)
    parser.add_argument("--recovar-directory-half2", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=384)
    parser.add_argument("--delta-relative-l2-bound", type=float, default=2e-7)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(args.grid_size > 0, "grid size must be positive")
    _require(args.delta_relative_l2_bound > 0.0, "delta bound must be positive")

    native_by_original: dict[int, dict[str, Any]] = {}
    artifact_paths: list[Path] = []
    for metadata_path in sorted(args.native_directory.glob("*bpref_shadow_metadata.bin")):
        match = _NATIVE_PATTERN.match(metadata_path.name)
        _require(match is not None, f"unexpected native filename: {metadata_path.name}")
        native = _load_native_particle(metadata_path, grid_size=args.grid_size)
        original_index = native["stack_index_1based"] - 1
        _require(original_index not in native_by_original, "duplicate native source identity")
        native_by_original[original_index] = native
        prefix = _native_prefix(metadata_path)
        artifact_paths.extend(
            Path(f"{prefix}{field}.bin")
            for field in ("real", "imag", "weight", "metadata")
        )

    recovar_paths = sorted(args.recovar_directory_half1.glob("*.npz")) + sorted(
        args.recovar_directory_half2.glob("*.npz")
    )
    _require(native_by_original and recovar_paths, "particle-delta inputs are empty")
    comparisons = []
    for path in recovar_paths:
        recovar = _load_recovar_particle(path)
        original_index = int(recovar["original_index"])
        _require(original_index in native_by_original, f"missing native particle {original_index}")
        comparisons.append(_compare_particle(native_by_original[original_index], recovar))
        artifact_paths.append(path)
    _require(
        {item["original_index"] for item in comparisons} == set(native_by_original),
        "native and RECOVAR source-identity sets differ",
    )
    comparisons.sort(key=lambda item: (item["half"], item["recovar_prefix_weight_sum"]))
    bound = float(args.delta_relative_l2_bound)
    numerator_passes = sum(
        item["particle_delta"]["numerator"]["relative_l2"] <= bound
        for item in comparisons
    )
    denominator_passes = sum(
        item["particle_delta"]["denominator"]["relative_l2"] <= bound
        for item in comparisons
    )
    joint_passes = sum(
        item["particle_delta"]["numerator"]["relative_l2"] <= bound
        and item["particle_delta"]["denominator"]["relative_l2"] <= bound
        for item in comparisons
    )
    if joint_passes == len(comparisons):
        classification = "fixed_panel_particle_deltas_close_reduction_or_launch_order_remains"
    elif joint_passes == 0:
        classification = "fixed_panel_particle_deltas_diverge_before_inter_particle_reduction"
    else:
        classification = "fixed_panel_particle_delta_mismatch_is_particle_dependent"
    report = {
        "schema": "recovar.em.k1_bpref_particle_delta_boundary.v1",
        "metric_policy": "scale-sensitive relative-L2 on independent zero-prefix particle replays",
        "grid_size": int(args.grid_size),
        "delta_relative_l2_bound": bound,
        "fixed_panel_score": {
            "numerator_passes": int(numerator_passes),
            "denominator_passes": int(denominator_passes),
            "joint_passes": int(joint_passes),
            "total": len(comparisons),
        },
        "classification": classification,
        "particles": comparisons,
        "artifacts": {str(path.resolve()): _sha256(path) for path in artifact_paths},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["fixed_panel_score"], sort_keys=True))
    print(classification)


if __name__ == "__main__":
    main()
