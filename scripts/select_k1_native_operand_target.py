#!/usr/bin/env python3
"""Select the native RELION tuple at the first unequal K=1 fine-stage boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_k1_fine_score_boundary import _rotation_map, _translation_map
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture

REPORT_SCHEMA = "recovar.em.k1_native_operand_target.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _first_unequal_tuple(stage: dict[str, Any]) -> tuple[str, tuple[int, int]]:
    boundary = str(stage["first_exact_unequal_boundary"])
    if boundary in stage.get("first_mismatch", {}):
        mismatch = stage["first_mismatch"][boundary]
        _require(mismatch is not None, f"{boundary} has no mismatch record")
        if "tuple_key" in mismatch:
            key = mismatch["tuple_key"]
        else:
            _require(
                "recovar_rotation_row" in mismatch and "recovar_translation_row" in mismatch,
                f"{boundary} has no tuple coordinates",
            )
            key = [mismatch["recovar_rotation_row"], mismatch["recovar_translation_row"]]
    elif boundary == "significant_support":
        key = stage.get("first_support_native_only_key") or stage.get("first_support_recovar_only_key")
        _require(key is not None, "significant-support mismatch has no differing tuple")
    elif boundary == "hard_winner":
        winner = stage["native_winner"]
        key = [winner["recovar_rotation_row"], winner["recovar_translation_row"]]
    else:
        raise ValueError(f"cannot select an operand tuple from boundary {boundary!r}")
    _require(len(key) == 2, f"unexpected tuple key: {key}")
    return boundary, (int(key[0]), int(key[1]))


def select_target(*, stage_join_json: Path) -> dict[str, Any]:
    join = json.loads(stage_join_json.read_text())
    _require(join.get("schema") == "recovar.em.k1_top1_stage_join.v1", "unexpected stage-join schema")
    _require(join.get("status") == "complete", "stage join is incomplete")
    stage = join["stage_analysis"]
    boundary, (recovar_rotation_row, recovar_translation_row) = _first_unequal_tuple(stage)

    factor_path = Path(stage["native_factor"])
    score_path = Path(stage["native_fine_score"])
    recovar_path = Path(stage["recovar_capture"])
    factor = load_factor_capture(factor_path)
    score = load_fine_score_capture(score_path)
    _require(factor.stack_index == score.stack_index, "native capture identity changed")
    _require(
        int(factor.stack_index) == int(join["target"]["stack_index_one_based"]),
        "native stack differs from the sealed target",
    )
    with np.load(recovar_path, allow_pickle=False) as archive:
        recovar_rotations = np.asarray(archive["rotations"])
        recovar_translations = np.asarray(archive["fine_translations"])
    rotation_map, rotation_error = _rotation_map(factor.rotations, recovar_rotations)
    translation_map, translation_error = _translation_map(
        factor.translations,
        recovar_translations,
        physical_image_size=int(stage["physical_image_size"]),
    )

    active = score.candidates[(score.candidates["flags"] & ACTIVE) != 0]
    mapped_rotation = rotation_map[np.asarray(active["rotation_local"], dtype=np.int64)]
    mapped_translation = translation_map[np.asarray(active["translation_id"], dtype=np.int64)]
    selected = np.flatnonzero(
        (mapped_rotation == recovar_rotation_row) & (mapped_translation == recovar_translation_row)
    )
    _require(selected.size == 1, f"expected one native candidate for the tuple, got {selected.size}")
    row = active[int(selected[0])]
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "stage_join": str(stage_join_json.resolve()),
        "stage_join_sha256": _sha256(stage_join_json),
        "first_exact_unequal_boundary": boundary,
        "stack_index_one_based": int(factor.stack_index),
        "native_rotation_local": int(row["rotation_local"]),
        "native_translation_id": int(row["translation_id"]),
        "native_raw_diff2": float(row["raw_diff2"]),
        "recovar_rotation_row": recovar_rotation_row,
        "recovar_translation_row": recovar_translation_row,
        "rotation_map_max_abs": float(rotation_error),
        "translation_map_max_abs": float(translation_error),
        "artifacts": {
            "native_factor": str(factor_path.resolve()),
            "native_fine_score": str(score_path.resolve()),
            "recovar_capture": str(recovar_path.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-join-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = select_target(stage_join_json=args.stage_join_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
