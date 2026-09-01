#!/usr/bin/env python3
"""Compare native RELION Wavg posterior inputs with RECOVAR pass-2 dumps.

The join is by immutable one-based stack identity from the native preprocess
capture, never by RELION's shuffled ``part_id`` or a physical row number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

if __package__:
    from scripts.analyze_k1_native_wavg_pixels import (
        _load_counted,
        _normalise_native_weights,
    )
    from scripts.validate_relion_preprocess_capture import load_artifact
else:
    from analyze_k1_native_wavg_pixels import (  # type: ignore[no-redef]
        _load_counted,
        _normalise_native_weights,
    )
    from validate_relion_preprocess_capture import load_artifact  # type: ignore[no-redef]


_WEIGHTS_RE = re.compile(
    r"img(?P<img>\d+)_part(?P<part>\d+)_storeWavg_weights\.bin"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_f32(values: np.ndarray) -> np.ndarray:
    signed = np.asarray(values, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(signed < 0, np.int64(0x80000000) - signed, signed)


def _float_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape:
        return {
            "shape_exact": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    mismatch = reference.view(np.uint32) != candidate.view(np.uint32)
    reference_norm = np.linalg.norm(reference.astype(np.float64).ravel())
    ulp = np.abs(_ordered_f32(reference) - _ordered_f32(candidate))
    return {
        "shape_exact": True,
        "element_count": int(reference.size),
        "bit_exact_count": int(reference.size - np.count_nonzero(mismatch)),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2": (
            float(np.linalg.norm(difference.ravel()) / reference_norm)
            if reference_norm
            else 0.0
        ),
        "max_ulp": int(np.max(ulp, initial=0)),
    }


def _mask_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if reference.shape != candidate.shape:
        return {
            "shape_exact": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    intersection = int(np.count_nonzero(reference & candidate))
    union = int(np.count_nonzero(reference | candidate))
    return {
        "shape_exact": True,
        "reference_count": int(np.count_nonzero(reference)),
        "candidate_count": int(np.count_nonzero(candidate)),
        "mismatch_count": int(np.count_nonzero(reference != candidate)),
        "intersection_count": intersection,
        "union_count": union,
        "jaccard": float(intersection / union) if union else 1.0,
    }


def _is_exact(comparison: dict[str, object]) -> bool:
    return bool(comparison.get("shape_exact")) and int(comparison["mismatch_count"]) == 0


def _native_to_recovar_rows(
    native: np.ndarray,
    recovar: np.ndarray,
    *,
    name: str,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    native = np.asarray(native, dtype=np.float32)
    recovar = np.asarray(recovar, dtype=np.float32)
    if native.shape != recovar.shape or native.ndim < 2:
        raise ValueError(
            f"{name}: native and RECOVAR geometry shapes differ: "
            f"{native.shape} versus {recovar.shape}"
        )
    pair_error = np.max(
        np.abs(native[:, None] - recovar[None, :]),
        axis=tuple(range(2, native.ndim + 1)),
    )
    mapping = np.argmin(pair_error, axis=1).astype(np.int64)
    if np.unique(mapping).size != mapping.size:
        raise ValueError(f"{name}: nearest-row map is not bijective")
    maximum_error = float(np.max(pair_error[np.arange(mapping.size), mapping]))
    if maximum_error > tolerance:
        raise ValueError(
            f"{name}: nearest-row maximum error {maximum_error} exceeds {tolerance}"
        )
    return mapping, maximum_error


def _gather_native_table_in_recovar_order(
    values: np.ndarray,
    rotation_map: np.ndarray,
    translation_map: np.ndarray,
) -> np.ndarray:
    native = np.asarray(values)
    expected_shape = (rotation_map.size, translation_map.size)
    if native.shape != expected_shape:
        raise ValueError(
            f"native posterior table shape {native.shape} differs from {expected_shape}"
        )
    result = np.empty_like(native)
    result[np.ix_(rotation_map, translation_map)] = native
    return result


def _native_prefix(weights_path: Path) -> str:
    return weights_path.name.removesuffix("weights.bin")


def _physical_image_shape(artifact, *, stack_index: int) -> tuple[int, int]:
    image_shape = tuple(int(value) for value in artifact.header[12:14])
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError(
            f"stack {stack_index}: invalid native real-space image shape {image_shape}"
        )
    return image_shape


def _load_native(weights_path: Path) -> dict[str, np.ndarray | int]:
    prefix = _native_prefix(weights_path)
    directory = weights_path.parent

    def counted(name: str, dtype: str) -> np.ndarray:
        return _load_counted(directory / f"{prefix}{name}.bin", dtype)

    orientation_count = int(counted("orientation_num", "<f8")[0])
    translation_count = int(counted("translation_num", "<f8")[0])
    raw_weights = counted("weights", "<f4")
    weights = _normalise_native_weights(
        raw_weights,
        orientation_count,
        translation_count,
    )
    sentinel_mask = raw_weights.reshape(orientation_count, translation_count) != np.finfo(
        np.float32
    ).min
    return {
        "orientation_count": orientation_count,
        "translation_count": translation_count,
        "rotations": counted("eulers", "<f4").reshape(orientation_count, 3, 3),
        "translation_angles": np.stack(
            (counted("trans_x", "<f4"), counted("trans_y", "<f4")), axis=1
        ),
        "weights": weights,
        "sentinel_mask": sentinel_mask,
    }


def analyze(native_directory: Path, recovar_directory: Path) -> dict[str, object]:
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_translation_angles_f32,
    )

    preprocess_by_part = {
        int(artifact.part_id): artifact
        for artifact in map(load_artifact, sorted(native_directory.glob("*.preprocess-v1.bin")))
    }
    if not preprocess_by_part:
        raise ValueError(f"no native preprocess artifacts in {native_directory}")

    records: list[dict[str, object]] = []
    for weights_path in sorted(native_directory.glob("*_storeWavg_weights.bin")):
        match = _WEIGHTS_RE.fullmatch(weights_path.name)
        if match is None:
            continue
        part_id = int(match["part"])
        if part_id not in preprocess_by_part:
            raise ValueError(f"native part {part_id} has no preprocess identity artifact")
        artifact = preprocess_by_part[part_id]
        stack_index = int(artifact.stack_index)
        original_index = stack_index - 1
        image_shape = _physical_image_shape(artifact, stack_index=stack_index)
        native = _load_native(weights_path)
        current_size = int(
            _load_counted(
                native_directory / f"{_native_prefix(weights_path)}current_size.bin",
                "<f8",
            )[0]
        )
        paths = sorted(
            list(
                recovar_directory.glob(
                    f"pass2_orig{original_index:06d}_cs{current_size:03d}.npz"
                )
            )
            + list(
                recovar_directory.glob(
                    f"pass2_orig{original_index:06d}_class001_cs{current_size:03d}.npz"
                )
            )
        )
        if len(paths) != 1:
            raise ValueError(
                f"stack {stack_index}: expected one RECOVAR pass-2 artifact, found {len(paths)}"
            )
        recovar_path = paths[0]
        with np.load(recovar_path, allow_pickle=False) as archive:
            recovar = {name: np.array(archive[name], copy=True) for name in archive.files}

        rotations = np.asarray(recovar["rotations"], dtype=np.float32)
        fine_translations = np.asarray(recovar["fine_translations"], dtype=np.float32)
        translation_angles = np.asarray(
            _relion_translation_angles_f32(fine_translations, image_shape),
            dtype=np.float32,
        )
        native_rotations = np.transpose(
            np.asarray(native["rotations"], dtype=np.float32),
            (0, 2, 1),
        )
        rotation_map, rotation_map_max_abs = _native_to_recovar_rows(
            native_rotations,
            rotations,
            name="fine rotations",
            tolerance=5e-6,
        )
        translation_map, translation_map_max_abs = _native_to_recovar_rows(
            np.asarray(native["translation_angles"], dtype=np.float32),
            translation_angles,
            name="fine translation angles",
            tolerance=5e-6,
        )
        native_weights = _gather_native_table_in_recovar_order(
            np.asarray(native["weights"], dtype=np.float32),
            rotation_map,
            translation_map,
        )
        native_mask = _gather_native_table_in_recovar_order(
            np.asarray(native["sentinel_mask"], dtype=bool),
            rotation_map,
            translation_map,
        )
        comparisons = {
            "rotations_native_vs_recovar_transpose": _float_comparison(
                native_rotations, rotations[rotation_map]
            ),
            "translation_angles": _float_comparison(
                np.asarray(native["translation_angles"]),
                translation_angles[translation_map],
            ),
            "native_sentinel_vs_candidate_mask": _mask_comparison(
                native_mask, recovar["candidate_mask"]
            ),
            "native_weights_vs_probs": _float_comparison(
                native_weights, recovar["probs"]
            ),
        }
        if "reconstruction_mask" in recovar:
            comparisons["native_sentinel_vs_reconstruction_mask"] = _mask_comparison(
                native_mask, recovar["reconstruction_mask"]
            )
        if "reconstruction_probs" in recovar:
            comparisons["native_weights_vs_reconstruction_probs"] = _float_comparison(
                native_weights, recovar["reconstruction_probs"]
            )

        ordered_boundaries = [
            "rotations_native_vs_recovar_transpose",
            "translation_angles",
            "native_sentinel_vs_candidate_mask",
            "native_weights_vs_probs",
            "native_sentinel_vs_reconstruction_mask",
            "native_weights_vs_reconstruction_probs",
        ]
        exact = {
            name: _is_exact(comparisons[name])
            for name in ordered_boundaries
            if name in comparisons
        }
        first_unequal = next((name for name in ordered_boundaries if name in exact and not exact[name]), None)
        records.append(
            {
                "part_id": part_id,
                "stack_index_one_based": stack_index,
                "original_index_zero_based": original_index,
                "native_weights_path": str(weights_path.resolve()),
                "native_weights_sha256": _sha256(weights_path),
                "recovar_path": str(recovar_path.resolve()),
                "recovar_sha256": _sha256(recovar_path),
                "current_size": current_size,
                "physical_image_shape": list(image_shape),
                "rotation_map_native_to_recovar": rotation_map.astype(int).tolist(),
                "rotation_map_max_abs": rotation_map_max_abs,
                "translation_map_native_to_recovar": translation_map.astype(int).tolist(),
                "translation_map_max_abs": translation_map_max_abs,
                "orientation_count": int(native["orientation_count"]),
                "translation_count": int(native["translation_count"]),
                "native_probability_sum_float64": float(
                    np.sum(native_weights, dtype=np.float64)
                ),
                "recovar_probability_sum_float64": float(
                    np.sum(np.asarray(recovar["probs"], dtype=np.float32), dtype=np.float64)
                ),
                "native_pmax": float(np.max(native_weights, initial=0.0)),
                "recovar_pmax": float(
                    np.max(np.asarray(recovar["probs"], dtype=np.float32), initial=0.0)
                ),
                "comparisons": comparisons,
                "stage_exact": exact,
                "first_exact_unequal_boundary": first_unequal,
            }
        )
    if not records:
        raise ValueError(f"no native Wavg weight artifacts in {native_directory}")
    return {
        "schema": "recovar-k1-wavg-posterior-boundary-v1",
        "join_key": "one-based stack identity from native preprocess artifact",
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.native_directory.resolve(), args.recovar_directory.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
