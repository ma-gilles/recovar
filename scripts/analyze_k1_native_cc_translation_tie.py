#!/usr/bin/env python3
"""Resolve one RELION first-iteration normalized-CC translation tie exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parse_relion_dump_dir import parse_dump_dir  # noqa: E402


SCHEMA = "em_k1_native_cc_translation_tie_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _layout_name(name: str) -> str:
    while True:
        stripped = name
        for prefix in ("pass", "over", "img", "part", "class"):
            stripped = re.sub(rf"^{prefix}\d+_", "", stripped, count=1)
        if stripped == name:
            return name
        name = stripped


def _pass_field(
    payload: dict[str, np.ndarray],
    suffix: str,
    *,
    pass_prefix: str = "pass1",
) -> np.ndarray:
    matches = [
        (name, value)
        for name, value in sorted(payload.items())
        if _layout_name(name) == suffix
        and (name == pass_prefix or name.startswith(pass_prefix + "_"))
    ]
    nonempty = [(name, value) for name, value in matches if np.asarray(value).size]
    if len(nonempty) != 1:
        raise ValueError(
            f"expected one nonempty {pass_prefix} field {suffix!r}, "
            f"found {[name for name, _ in nonempty]}"
        )
    return np.asarray(nonempty[0][1])


def _float_record(value: float) -> dict[str, Any]:
    value64 = np.float64(value)
    value32 = np.float32(value64)
    return {
        "float64": float(value64),
        "float64_hex": float(value64).hex(),
        "float64_bits_hex": f"0x{value64.view(np.uint64).item():016x}",
        "float32": float(value32),
        "float32_hex": float(value32).hex(),
        "float32_bits_hex": f"0x{value32.view(np.uint32).item():08x}",
    }


def _relion_atomic_score(numerator: float, norm: float) -> np.float32:
    numerator32 = np.float32(numerator)
    norm32 = np.float32(norm)
    contribution = np.float32(
        numerator32 / np.float32(np.float32(128.0) * np.sqrt(norm32, dtype=np.float32))
    )
    accumulated = np.float32(0.0)
    for _ in range(128):
        accumulated = np.float32(accumulated + contribution)
    return accumulated


def _unique_row(mask: np.ndarray, *, label: str) -> int:
    rows = np.flatnonzero(mask)
    if rows.size != 1:
        raise ValueError(f"expected one {label} row, found {rows.tolist()}")
    return int(rows[0])


def _recovar_score_panel(
    old_capture: Path,
    current_capture: Path,
    *,
    matched_rotation_row: int,
    translations: tuple[int, int],
) -> dict[str, Any]:
    with np.load(old_capture, allow_pickle=False) as old:
        old_rotation_ids = np.asarray(old["oversampled_rot_indices"], dtype=np.int64)
        if matched_rotation_row < 0 or matched_rotation_row >= old_rotation_ids.size:
            raise ValueError(
                f"matrix-matched RECOVAR rotation row {matched_rotation_row} is outside "
                f"the historical capture's {old_rotation_ids.size} rows"
            )
        old_rotation_row = int(matched_rotation_row)
        global_rotation = int(old_rotation_ids[old_rotation_row])
        old_scores = np.asarray(old["scores_pre_prior"], dtype=np.float64)
        old_flat_winner = int(np.argmax(old_scores.reshape(-1)))
        old_winner = np.unravel_index(old_flat_winner, old_scores.shape)

    with np.load(current_capture, allow_pickle=False) as current:
        if int(np.asarray(current["original_indices"]).reshape(-1)[0]) != 38594:
            raise ValueError("current capture is not source row 38594")
        active_global = np.asarray(
            current["active_global_rotation_indices"], dtype=np.int64
        )
        active_rows = np.asarray(current["active_rotation_rows"], dtype=np.int64)
        active_position = _unique_row(
            active_global == global_rotation,
            label="current RECOVAR active rotation",
        )
        current_rotation_row = int(active_rows[active_position])
        current_scores = np.asarray(
            current["candidate_preprior_scores"], dtype=np.float64
        )[0]
        current_flat_winner = int(np.argmax(current_scores.reshape(-1)))
        current_winner = np.unravel_index(current_flat_winner, current_scores.shape)

    def panel(scores: np.ndarray, rotation_row: int) -> dict[str, Any]:
        left, right = translations
        left_value = scores[rotation_row, left]
        right_value = scores[rotation_row, right]
        return {
            "rotation_row": rotation_row,
            f"translation_{left}": _float_record(left_value),
            f"translation_{right}": _float_record(right_value),
            "right_minus_left": _float_record(right_value - left_value),
            "exact_tie": bool(left_value == right_value),
        }

    return {
        "historical": {
            **panel(old_scores, old_rotation_row),
            "global_rotation": global_rotation,
            "winner_rotation_row": int(old_winner[0]),
            "winner_translation": int(old_winner[1]),
        },
        "current": {
            **panel(current_scores, current_rotation_row),
            "global_rotation": global_rotation,
            "winner_rotation_row": int(current_winner[0]),
            "winner_translation": int(current_winner[1]),
        },
    }


def build_report(
    *,
    relion_dump_dir: Path,
    old_recovar_capture: Path,
    current_recovar_capture: Path,
    comparison_json: Path,
    translations: tuple[int, int],
) -> dict[str, Any]:
    payload = parse_dump_dir(relion_dump_dir)
    raw_cost = _pass_field(payload, "firstiter_cc_exp_Mweight_raw_preonehot").reshape(-1)
    compact_rotation = _pass_field(payload, "firstiter_cc_raw_rot_idx").reshape(-1)
    global_rotation = _pass_field(payload, "firstiter_cc_raw_rot_id").reshape(-1)
    translation = _pass_field(payload, "firstiter_cc_raw_trans_idx").reshape(-1)
    if not (
        raw_cost.size == compact_rotation.size == global_rotation.size == translation.size
    ):
        raise ValueError("RELION candidate arrays have different sizes")

    declared_argmin = int(
        _pass_field(payload, "firstiter_cc_argmin_index").reshape(-1)[0]
    )
    computed_argmin = int(np.argmin(raw_cost))
    if declared_argmin != computed_argmin:
        raise ValueError(
            f"RELION declared argmin {declared_argmin} != first numpy argmin {computed_argmin}"
        )
    winning_compact_rotation = int(compact_rotation[declared_argmin])
    winning_global_rotation = int(global_rotation[declared_argmin])
    winning_translation = int(translation[declared_argmin])

    native_candidates: dict[str, Any] = {}
    native_rows: dict[int, int] = {}
    for translation_id in translations:
        row = _unique_row(
            (compact_rotation == winning_compact_rotation)
            & (translation == translation_id),
            label=f"RELION compact rotation {winning_compact_rotation}, translation {translation_id}",
        )
        native_rows[translation_id] = row
        native_candidates[str(translation_id)] = {
            "candidate_row": row,
            "compact_rotation": int(compact_rotation[row]),
            "raw_global_rotation": int(global_rotation[row]),
            "raw_cost": _float_record(raw_cost[row]),
            "score": _float_record(-raw_cost[row]),
        }

    component_status = "absent"
    try:
        component_weight = _pass_field(payload, "cc_component_weight").reshape(-1)
        component_norm = _pass_field(payload, "cc_component_norm").reshape(-1)
        component_translation_count = int(
            _pass_field(payload, "cc_component_translation_num").reshape(-1)[0]
        )
        component_status = "present"
        for translation_id in translations:
            component_row = winning_compact_rotation * component_translation_count + translation_id
            if component_row >= component_weight.size or component_row >= component_norm.size:
                raise ValueError("RELION CC component row is outside captured arrays")
            reconstructed = _relion_atomic_score(
                component_weight[component_row], component_norm[component_row]
            )
            native_candidates[str(translation_id)]["components"] = {
                "component_row": int(component_row),
                "numerator": _float_record(component_weight[component_row]),
                "norm": _float_record(component_norm[component_row]),
                "reconstructed_score": _float_record(reconstructed),
                "reconstructed_score_matches_raw_cost_bitwise_f32": bool(
                    reconstructed.view(np.uint32)
                    == np.float32(-raw_cost[native_rows[translation_id]]).view(np.uint32)
                ),
            }
    except ValueError as error:
        if "expected one nonempty" not in str(error):
            raise

    comparison = json.loads(comparison_json.read_text())
    if int(comparison["recovar_original_index"]) != 38594:
        raise ValueError("comparison JSON is not source row 38594")
    matched_relion_top = comparison["relion_top_key"]
    if matched_relion_top is None:
        raise ValueError("comparison has no matrix-matched RELION top candidate")
    matched_rotation_row = int(matched_relion_top[0])
    matched_translation = int(matched_relion_top[1])
    if matched_translation != winning_translation:
        raise ValueError("matrix-matched and raw RELION winner translations differ")

    left, right = translations
    native_left = -raw_cost[native_rows[left]]
    native_right = -raw_cost[native_rows[right]]
    recovar = _recovar_score_panel(
        old_recovar_capture,
        current_recovar_capture,
        matched_rotation_row=matched_rotation_row,
        translations=translations,
    )
    if native_left == native_right:
        diagnosis = "native_exact_tie"
    elif native_right > native_left and recovar["current"]["exact_tie"]:
        diagnosis = "current_scorer_lost_native_right_translation_advantage"
    elif native_left > native_right and recovar["current"]["exact_tie"]:
        diagnosis = "current_scorer_lost_native_left_translation_advantage"
    else:
        diagnosis = "native_and_current_are_both_split_or_require_additional_localization"

    source_files = sorted(relion_dump_dir.glob("*.bin"))
    return {
        "schema": SCHEMA,
        "status": "pass",
        "diagnosis": diagnosis,
        "translations": list(translations),
        "relion": {
            "candidate_count": int(raw_cost.size),
            "declared_argmin": declared_argmin,
            "computed_first_argmin": computed_argmin,
            "winner_compact_rotation": winning_compact_rotation,
            "winner_raw_global_rotation": winning_global_rotation,
            "winner_matrix_matched_recovar_rotation_row": matched_rotation_row,
            "winner_translation": winning_translation,
            "component_status": component_status,
            "candidates": native_candidates,
            "right_minus_left_score": _float_record(native_right - native_left),
            "exact_tie": bool(native_left == native_right),
        },
        "recovar": recovar,
        "inputs": {
            str(old_recovar_capture.resolve()): _sha256(old_recovar_capture),
            str(current_recovar_capture.resolve()): _sha256(current_recovar_capture),
            str(comparison_json.resolve()): _sha256(comparison_json),
            "relion_dump_file_count": len(source_files),
            "relion_dump_manifest_sha256": {
                path.name: _sha256(path) for path in source_files
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-dump-dir", required=True, type=Path)
    parser.add_argument("--old-recovar-capture", required=True, type=Path)
    parser.add_argument("--current-recovar-capture", required=True, type=Path)
    parser.add_argument("--comparison-json", required=True, type=Path)
    parser.add_argument("--left-translation", type=int, default=89)
    parser.add_argument("--right-translation", type=int, default=91)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = build_report(
        relion_dump_dir=args.relion_dump_dir,
        old_recovar_capture=args.old_recovar_capture,
        current_recovar_capture=args.current_recovar_capture,
        comparison_json=args.comparison_json,
        translations=(args.left_translation, args.right_translation),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
