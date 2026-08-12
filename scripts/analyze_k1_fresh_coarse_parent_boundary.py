#!/usr/bin/env python3
"""Compare fresh RECOVAR coarse support with native RELION fine parents."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture
from scripts.analyze_k1_fine_score_boundary import _rotation_map


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_parent_keys(
    factor,
    fine_score,
    *,
    recovar_fine_geometry_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    active = (fine_score.candidates["flags"] & ACTIVE) != 0
    active_rows = fine_score.candidates[active]
    _require(active_rows.size > 0, "native fine support is empty")
    rotation_local = np.asarray(active_rows["rotation_local"], dtype=np.int64)
    translation_local = np.asarray(active_rows["translation_id"], dtype=np.int64)
    _require(
        np.all(rotation_local < factor.rotations.size)
        and np.all(translation_local < factor.translations.size),
        "native fine tuple index exceeds factor geometry",
    )
    if recovar_fine_geometry_path is None:
        coarse_rotation_by_factor_row = np.asarray(
            factor.rotations["orientation_class_key"], dtype=np.int64
        )
    else:
        with np.load(recovar_fine_geometry_path, allow_pickle=False) as geometry:
            rotation_mapping, error = _rotation_map(factor.rotations, geometry["rotations"])
            oversampled_rotation_ids = np.asarray(
                geometry["oversampled_rot_indices"], dtype=np.int64
            )[rotation_mapping]
        _require(error == 0.0, "fine rotation geometry is not exact")
        _require(
            np.array_equal(
                oversampled_rotation_ids % 8,
                np.asarray(factor.rotations["oversampled_rotation"], dtype=np.int64),
            ),
            "fine rotation child numbering differs",
        )
        coarse_rotation_by_factor_row = oversampled_rotation_ids // 8
    coarse_rotation = coarse_rotation_by_factor_row[rotation_local]
    fine_translation = np.asarray(
        factor.translations["translation"][translation_local], dtype=np.int64
    )
    _require(
        np.array_equal(
            np.asarray(factor.translations["translation"], dtype=np.int64),
            np.arange(factor.translations.size, dtype=np.int64),
        )
        and factor.translations.size % 4 == 0,
        "native oversampled-translation indexing changed",
    )
    coarse_translation = fine_translation // 4
    fine_keys = np.column_stack((coarse_rotation, coarse_translation))
    parent_keys, child_counts = np.unique(fine_keys, axis=0, return_counts=True)
    return parent_keys, child_counts


def _recovar_parent_keys(coarse_path: Path) -> tuple[np.ndarray, dict[str, int | float]]:
    with np.load(coarse_path, allow_pickle=False) as archive:
        n_classes = int(np.asarray(archive["n_classes"]).item())
        n_rot = int(np.asarray(archive["n_rot"]).item())
        n_trans = int(np.asarray(archive["n_trans"]).item())
        mask = np.asarray(archive["significant_mask"], dtype=bool)
        n_significant = int(np.asarray(archive["n_significant"]).item())
        adaptive_fraction = float(np.asarray(archive["adaptive_fraction"]).item())
    _require(n_classes == 1, "coarse-parent boundary is K=1 only")
    _require(mask.size == n_rot * n_trans, "coarse significant-mask topology changed")
    _require(int(np.count_nonzero(mask)) == n_significant, "coarse support count changed")
    keys = np.argwhere(mask.reshape(n_rot, n_trans)).astype(np.int64)
    return keys, {
        "n_rot": n_rot,
        "n_trans": n_trans,
        "n_significant": n_significant,
        "adaptive_fraction": adaptive_fraction,
    }


def _records(keys: set[tuple[int, int]], *, limit: int = 64) -> list[list[int]]:
    return [[int(rotation), int(translation)] for rotation, translation in sorted(keys)[:limit]]


def analyze(
    *,
    factor_path: Path,
    fine_score_path: Path,
    coarse_path: Path,
    recovar_fine_geometry_path: Path | None = None,
    physical_image_size: int = 128,
) -> dict[str, object]:
    factor = load_factor_capture(factor_path)
    fine_score = load_fine_score_capture(fine_score_path)
    _require(
        factor.stack_index == fine_score.stack_index,
        "native factor and fine-score captures identify different particles",
    )
    native_keys_array, child_counts = _native_parent_keys(
        factor,
        fine_score,
        recovar_fine_geometry_path=recovar_fine_geometry_path,
    )
    recovar_keys_array, recovar_metadata = _recovar_parent_keys(coarse_path)
    with np.load(coarse_path, allow_pickle=False) as coarse:
        recovar_translations = np.asarray(coarse["translations"], dtype=np.float64)
    native_fine_translations = (
        -np.column_stack((factor.translations["x"], factor.translations["y"]))
        * float(physical_image_size)
        / (2.0 * np.pi)
    )
    native_parent_translations = native_fine_translations.reshape(-1, 4, 2).mean(axis=1)
    translation_distance = np.max(
        np.abs(native_parent_translations[:, None, :] - recovar_translations[None, :, :]),
        axis=2,
    )
    translation_mapping = np.argmin(translation_distance, axis=1)
    translation_error = translation_distance[
        np.arange(native_parent_translations.shape[0]), translation_mapping
    ]
    _require(
        np.all(translation_error <= 1.0e-6)
        and np.unique(translation_mapping).size == translation_mapping.size,
        "native and RECOVAR coarse translations do not map one-to-one",
    )
    native_keys_array[:, 1] = translation_mapping[native_keys_array[:, 1]]
    native_keys = {tuple(map(int, row)) for row in native_keys_array.tolist()}
    recovar_keys = {tuple(map(int, row)) for row in recovar_keys_array.tolist()}
    native_only = native_keys - recovar_keys
    recovar_only = recovar_keys - native_keys
    return {
        "schema": "recovar.em.k1_fresh_coarse_parent_boundary.v1",
        "status": "complete",
        "identity": {
            "stack_index_one_based": factor.stack_index,
        },
        "parent_support": {
            "exact_equal": not native_only and not recovar_only,
            "native_parent_count": len(native_keys),
            "recovar_parent_count": len(recovar_keys),
            "native_only_count": len(native_only),
            "recovar_only_count": len(recovar_only),
            "native_only_first": _records(native_only),
            "recovar_only_first": _records(recovar_only),
        },
        "native_child_multiplicity": {
            "minimum": int(np.min(child_counts)),
            "maximum": int(np.max(child_counts)),
            "complete_32_child_parent_count": int(np.count_nonzero(child_counts == 32)),
            "parent_count": int(child_counts.size),
        },
        "recovar": recovar_metadata,
        "alignment": {
            "rotation": (
                "exact_fine_matrix_bijection_then_oversampled_rotation_id_div_8"
                if recovar_fine_geometry_path is not None
                else "native_orientation_class_key_unmapped"
            ),
            "translation_max_abs": float(np.max(translation_error)),
        },
        "artifacts": {
            "factor": str(factor_path.resolve()),
            "factor_sha256": _sha256(factor_path),
            "fine_score": str(fine_score_path.resolve()),
            "fine_score_sha256": _sha256(fine_score_path),
            "coarse": str(coarse_path.resolve()),
            "coarse_sha256": _sha256(coarse_path),
            "recovar_fine_geometry": (
                None
                if recovar_fine_geometry_path is None
                else str(recovar_fine_geometry_path.resolve())
            ),
            "recovar_fine_geometry_sha256": (
                None
                if recovar_fine_geometry_path is None
                else _sha256(recovar_fine_geometry_path)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--fine-score", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--recovar-fine-geometry", type=Path)
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        factor_path=args.factor,
        fine_score_path=args.fine_score,
        coarse_path=args.coarse,
        recovar_fine_geometry_path=args.recovar_fine_geometry,
        physical_image_size=args.physical_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
