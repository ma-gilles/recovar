#!/usr/bin/env python3
"""Compare exact native RELION BPref prefixes with RECOVAR particle contributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from scripts.analyze_vdam_storewavg_boundary import _metric, _require

SCHEMA = "recovar.vdam_native_bpref_prefix.v1"
_METADATA_NAME = re.compile(
    r"^half(?P<half>[0-9]+)_part(?P<part>[0-9]+)_stack(?P<stack>[0-9]+)"
    r"_bpref_prefix_metadata\.bin$"
)


def _flat_u64(path: Path, dtype: np.dtype) -> np.ndarray:
    payload = path.read_bytes()
    _require(len(payload) >= 8, f"truncated prefix array: {path}")
    count = int(np.frombuffer(payload, dtype="<u8", count=1)[0])
    values = np.frombuffer(payload, dtype=dtype, offset=8).copy()
    _require(values.size == count, f"prefix-array size mismatch: {path}")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0, "cannot summarize an empty prefix panel")
    return {
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _native_prefixes(directory: Path) -> list[dict[str, object]]:
    prefixes: list[dict[str, object]] = []
    for metadata_path in directory.glob("*_bpref_prefix_metadata.bin"):
        match = _METADATA_NAME.match(metadata_path.name)
        _require(match is not None, f"unrecognized native prefix name: {metadata_path}")
        metadata = _flat_u64(metadata_path, np.dtype("<u8"))
        _require(metadata.size == 15, f"native prefix metadata topology changed: {metadata_path}")
        half = int(match.group("half"))
        part_id = int(match.group("part"))
        stack_index = int(match.group("stack"))
        _require(int(metadata[0]) == 1, f"unsupported native prefix version: {metadata_path}")
        _require(int(metadata[2]) == half, f"native half identity changed: {metadata_path}")
        _require(int(metadata[3]) == part_id, f"native part identity changed: {metadata_path}")
        _require(int(metadata[4]) == stack_index, f"native stack identity changed: {metadata_path}")
        prefix = metadata_path.with_name(metadata_path.name.removesuffix("metadata.bin"))
        mdl_x, mdl_y, mdl_z, mdl_xyz = map(int, (metadata[7], metadata[8], metadata[9], metadata[14]))
        _require(mdl_x * mdl_y * mdl_z == mdl_xyz, f"native BPref dimensions changed: {metadata_path}")
        real = _flat_u64(prefix.with_name(prefix.name + "real.bin"), np.dtype("<f4"))
        imag = _flat_u64(prefix.with_name(prefix.name + "imag.bin"), np.dtype("<f4"))
        weight = _flat_u64(prefix.with_name(prefix.name + "weight.bin"), np.dtype("<f4"))
        _require(real.shape == imag.shape == weight.shape == (mdl_xyz,), f"native BPref size changed: {prefix}")
        prefixes.append(
            {
                "half": half,
                "iteration": int(metadata[1]),
                "part_id": part_id,
                "stack_index": stack_index,
                "original_index": stack_index - 1,
                "shape": (mdl_z, mdl_y, mdl_x),
                "data": (real + np.complex64(1j) * imag).astype(np.complex64),
                "weight": weight,
                "metadata_path": metadata_path.resolve(),
            }
        )
    _require(prefixes, f"no native BPref prefixes found in {directory}")
    prefixes.sort(key=lambda item: int(item["part_id"]))
    _require(
        len({int(item["part_id"]) for item in prefixes}) == len(prefixes),
        "native prefix part IDs are not unique",
    )
    return prefixes


def _recovar_contributions(paths: list[Path]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as capture:
            required = {
                "inline_projector_original_indices",
                "inline_projector_data_volumes",
                "inline_projector_weight_volumes",
                "image_shape",
            }
            missing = sorted(required.difference(capture.files))
            _require(not missing, f"RECOVAR inline capture lacks {missing}: {path}")
            original_indices = np.asarray(capture["inline_projector_original_indices"], dtype=np.int64)
            data = np.asarray(capture["inline_projector_data_volumes"], dtype=np.complex64)
            weight = np.asarray(capture["inline_projector_weight_volumes"], dtype=np.float32)
            _require(data.shape == weight.shape, f"RECOVAR inline data/weight topology differs: {path}")
            _require(data.shape[0] == original_indices.size, f"RECOVAR inline identity topology differs: {path}")
            for slot, original_index in enumerate(original_indices):
                identity = int(original_index)
                _require(identity not in result, f"duplicate RECOVAR inline contribution for {identity}")
                result[identity] = (data[slot].reshape(-1), weight[slot].reshape(-1))
    _require(result, "no RECOVAR inline particle contributions were loaded")
    return result


def analyze(
    native_directory: Path,
    recovar_capture_paths: list[Path],
    *,
    physical_image_size: int = 128,
) -> dict[str, object]:
    native = _native_prefixes(native_directory)
    recovar = _recovar_contributions(recovar_capture_paths)
    data_scale = np.float32(-float(physical_image_size) ** -2)
    weight_scale = np.float32(float(physical_image_size) ** -4)
    shape = tuple(int(value) for value in native[0]["shape"])
    candidate_data = np.zeros(int(np.prod(shape)), dtype=np.complex64)
    candidate_weight = np.zeros(int(np.prod(shape)), dtype=np.float32)
    previous_native_data = np.zeros_like(candidate_data)
    previous_native_weight = np.zeros_like(candidate_weight)
    prefix_data_rel_l2: list[float] = []
    prefix_weight_rel_l2: list[float] = []
    increment_data_rel_l2: list[float] = []
    increment_weight_rel_l2: list[float] = []
    per_particle: list[dict[str, object]] = []
    seen_original_indices: set[int] = set()

    for item in native:
        _require(tuple(item["shape"]) == shape, "native prefix shapes differ")
        original_index = int(item["original_index"])
        _require(original_index in recovar, f"RECOVAR contribution is missing original index {original_index}")
        contribution_data, contribution_weight = recovar[original_index]
        _require(contribution_data.shape == candidate_data.shape, "native and RECOVAR BPref sizes differ")
        native_data = (np.asarray(item["data"], dtype=np.complex64) * data_scale).astype(np.complex64)
        native_weight = (np.asarray(item["weight"], dtype=np.float32) * weight_scale).astype(np.float32)
        native_increment_data = (native_data - previous_native_data).astype(np.complex64)
        native_increment_weight = (native_weight - previous_native_weight).astype(np.float32)
        candidate_data = (candidate_data + contribution_data).astype(np.complex64)
        candidate_weight = (candidate_weight + contribution_weight).astype(np.float32)
        prefix_data = _metric(native_data, candidate_data)
        prefix_weight = _metric(native_weight, candidate_weight)
        increment_data = _metric(native_increment_data, contribution_data)
        increment_weight = _metric(native_increment_weight, contribution_weight)
        prefix_data_rel_l2.append(float(prefix_data["relative_l2"]))
        prefix_weight_rel_l2.append(float(prefix_weight["relative_l2"]))
        increment_data_rel_l2.append(float(increment_data["relative_l2"]))
        increment_weight_rel_l2.append(float(increment_weight["relative_l2"]))
        per_particle.append(
            {
                "native_half": int(item["half"]),
                "native_part_id": int(item["part_id"]),
                "stack_index": int(item["stack_index"]),
                "original_index": original_index,
                "prefix_data": prefix_data,
                "prefix_weight": prefix_weight,
                "increment_data": increment_data,
                "increment_weight": increment_weight,
            }
        )
        previous_native_data = native_data
        previous_native_weight = native_weight
        seen_original_indices.add(original_index)

    return {
        "schema": SCHEMA,
        "identity": {
            "particle_count": len(native),
            "native_half_values": sorted({int(item["half"]) for item in native}),
            "iteration_values": sorted({int(item["iteration"]) for item in native}),
            "first_native_part_id": int(native[0]["part_id"]),
            "last_native_part_id": int(native[-1]["part_id"]),
            "first_original_index": int(native[0]["original_index"]),
            "last_original_index": int(native[-1]["original_index"]),
            "volume_shape": list(shape),
            "unmatched_recovar_original_indices": sorted(set(recovar).difference(seen_original_indices)),
        },
        "frame_scales": {"native_data_to_recovar": float(data_scale), "native_weight_to_recovar": float(weight_scale)},
        "summary": {
            "prefix_data_relative_l2": _quantiles(prefix_data_rel_l2),
            "prefix_weight_relative_l2": _quantiles(prefix_weight_rel_l2),
            "increment_data_relative_l2": _quantiles(increment_data_rel_l2),
            "increment_weight_relative_l2": _quantiles(increment_weight_rel_l2),
            "final_prefix_data": per_particle[-1]["prefix_data"],
            "final_prefix_weight": per_particle[-1]["prefix_weight"],
        },
        "per_particle": per_particle,
        "artifacts": {
            "native_directory": str(native_directory.resolve()),
            "recovar_captures": [
                {"path": str(path.resolve()), "sha256": _sha256(path)} for path in recovar_capture_paths
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-capture", type=Path, action="append", required=True)
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    report = analyze(
        args.native_directory,
        args.recovar_capture,
        physical_image_size=args.physical_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
