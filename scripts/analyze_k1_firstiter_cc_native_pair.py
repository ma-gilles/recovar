#!/usr/bin/env python3
"""Compare one RECOVAR firstiter-CC winner pair with native RELION operands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovar import utils
from scripts.analyze_k1_native_cc_translation_tie import (
    _float_record,
    _pass_field,
    _relion_atomic_score,
    _sha256,
    _unique_row,
)
from scripts.analyze_k1_pose_winner_map_counterfactual import (
    _match_target_candidate,
    _rotation_distances_deg,
)
from scripts.parse_relion_dump_dir import parse_dump_dir


SCHEMA = "recovar.em.k1_firstiter_cc_native_pair.v1"


def analyze(
    *,
    relion_dump_dir: Path,
    recovar_capture: Path,
    pose_comparison: Path,
    source_row: int,
) -> dict[str, object]:
    payload = parse_dump_dir(relion_dump_dir)
    raw_cost = _pass_field(payload, "firstiter_cc_exp_Mweight_raw_preonehot").reshape(-1)
    compact_rotation = _pass_field(payload, "firstiter_cc_raw_rot_idx").reshape(-1)
    global_rotation = _pass_field(payload, "firstiter_cc_raw_rot_id").reshape(-1)
    translation = _pass_field(payload, "firstiter_cc_raw_trans_idx").reshape(-1)
    if not (
        raw_cost.size == compact_rotation.size == global_rotation.size == translation.size
    ):
        raise ValueError("RELION candidate arrays have different sizes")
    declared_argmin = int(_pass_field(payload, "firstiter_cc_argmin_index").reshape(-1)[0])
    computed_argmin = int(np.argmin(raw_cost))
    if declared_argmin != computed_argmin:
        raise ValueError("RELION declared and computed first argmin differ")

    with np.load(recovar_capture, allow_pickle=False) as capture, np.load(
        pose_comparison, allow_pickle=False
    ) as poses:
        original_indices = np.asarray(capture["original_indices"], dtype=np.int64)
        selected = np.flatnonzero(original_indices == int(source_row))
        if selected.size != 1:
            raise ValueError(f"source row {source_row} appears {selected.size} times")
        particle_row = int(selected[0])
        probs = np.asarray(capture["reconstruction_probs"], dtype=np.float32)
        nonzero = np.argwhere(probs[particle_row] != 0.0)
        if nonzero.shape != (1, 2):
            raise ValueError("RECOVAR capture does not have one hard winner")
        current_rotation_row, current_translation = (int(value) for value in nonzero[0])
        target_rotation = utils.R_from_relion(
            np.asarray(poses["relion_eulers"], dtype=np.float64)[source_row]
        )[0]
        target = _match_target_candidate(
            active_particle_rows=np.asarray(capture["active_particle_rows"], dtype=np.int64),
            active_rotation_rows=np.asarray(capture["active_rotation_rows"], dtype=np.int64),
            active_rotations=np.asarray(capture["active_rotations"], dtype=np.float32),
            particle_row=particle_row,
            target_rotation=target_rotation,
            fine_translations=np.asarray(capture["fine_translations"], dtype=np.float32),
            integer_pre_shift=np.asarray(capture["integer_pre_shifts"])[particle_row],
            target_translation_pixels=np.asarray(
                poses["relion_translations"], dtype=np.float64
            )[source_row],
        )
        target_translation = int(target["translation_index"])
        recovar_scores = np.asarray(capture["candidate_preprior_scores"], dtype=np.float64)[
            particle_row
        ]
        current_recovar_score = recovar_scores[current_rotation_row, current_translation]
        target_recovar_score = recovar_scores[current_rotation_row, target_translation]
        current_global_rotation = int(
            np.asarray(capture["oversampled_rotation_indices"], dtype=np.int64)[
                particle_row, current_rotation_row
            ]
        )
        active_rows = np.asarray(capture["active_rotation_rows"], dtype=np.int64)
        active_particles = np.asarray(capture["active_particle_rows"], dtype=np.int64)
        current_active = np.flatnonzero(
            (active_particles == particle_row) & (active_rows == current_rotation_row)
        )
        if current_active.size != 1:
            raise ValueError("RECOVAR current rotation matrix is ambiguous")
        current_rotation_matrix = np.asarray(capture["active_rotations"], dtype=np.float32)[
            int(current_active[0])
        ]

    native_winner_rotation = int(compact_rotation[declared_argmin])
    native_winner_translation = int(translation[declared_argmin])
    native_winner_global_rotation = int(global_rotation[declared_argmin])
    # The CUDA dump stores the device-side inverse/row-vector matrix. RECOVAR
    # stores the forward/column-vector matrix used by its projector.
    native_rotation_matrices = np.transpose(
        _pass_field(payload, "fine_eulers").reshape(-1, 3, 3),
        (0, 2, 1),
    )
    if native_winner_rotation >= native_rotation_matrices.shape[0]:
        raise ValueError("native winner rotation lies outside the matrix panel")
    native_recovar_rotation_error = float(
        _rotation_distances_deg(
            np.asarray([current_rotation_matrix]),
            native_rotation_matrices[native_winner_rotation],
        )[0]
    )
    native_rows = {
        candidate_translation: _unique_row(
            (compact_rotation == native_winner_rotation)
            & (translation == candidate_translation),
            label=(
                f"RELION compact rotation {native_winner_rotation}, "
                f"translation {candidate_translation}"
            ),
        )
        for candidate_translation in (current_translation, target_translation)
    }
    native_scores = {
        candidate_translation: np.float32(-raw_cost[row])
        for candidate_translation, row in native_rows.items()
    }

    components: dict[str, object] = {"status": "absent"}
    try:
        component_weight = _pass_field(payload, "cc_component_weight").reshape(-1)
        component_norm = _pass_field(payload, "cc_component_norm").reshape(-1)
        translation_count = int(
            _pass_field(payload, "cc_component_translation_num").reshape(-1)[0]
        )
        values = {}
        for candidate_translation in (current_translation, target_translation):
            row = native_winner_rotation * translation_count + candidate_translation
            reconstructed = _relion_atomic_score(
                component_weight[row], component_norm[row]
            )
            values[str(candidate_translation)] = {
                "component_row": row,
                "numerator": _float_record(component_weight[row]),
                "norm": _float_record(component_norm[row]),
                "reconstructed_score": _float_record(reconstructed),
                "matches_raw_score_bitwise_f32": bool(
                    reconstructed.view(np.uint32)
                    == native_scores[candidate_translation].view(np.uint32)
                ),
            }
        components = {
            "status": "present",
            "translation_count": translation_count,
            "candidates": values,
        }
    except ValueError as error:
        if "expected one nonempty" not in str(error):
            raise

    recovar_delta = np.float64(target_recovar_score - current_recovar_score)
    native_delta = np.float32(
        native_scores[target_translation] - native_scores[current_translation]
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "source_row_zero_based": int(source_row),
        "current_translation_index": current_translation,
        "target_translation_index": target_translation,
        "relion_winner_translation_index": native_winner_translation,
        "target_rotation_error_deg_to_recovar_grid": target["rotation_error_deg"],
        "target_translation_error_pixels_to_recovar_grid": target[
            "translation_error_pixels"
        ],
        "rotation_identity": {
            "recovar_current_global_rotation": current_global_rotation,
            "relion_winner_global_rotation": native_winner_global_rotation,
            "integer_ids_share_namespace": False,
            "native_to_recovar_matrix_error_deg": native_recovar_rotation_error,
            "matrix_exact_within_1e-3_deg": native_recovar_rotation_error <= 1e-3,
        },
        "recovar": {
            "current_score": _float_record(current_recovar_score),
            "target_score": _float_record(target_recovar_score),
            "target_minus_current": _float_record(recovar_delta),
            "winner": (
                "target" if recovar_delta > 0 else "current" if recovar_delta < 0 else "tie"
            ),
        },
        "relion": {
            "current_score": _float_record(native_scores[current_translation]),
            "target_score": _float_record(native_scores[target_translation]),
            "target_minus_current": _float_record(native_delta),
            "winner": (
                "target" if native_delta > 0 else "current" if native_delta < 0 else "tie"
            ),
            "components": components,
        },
        "diagnosis": (
            "raw_normalized_cc_winner_flip"
            if np.sign(recovar_delta) != np.sign(np.float64(native_delta))
            else "same_pair_ordering"
        ),
        "inputs": {
            "relion_dump_dir": str(relion_dump_dir.resolve()),
            "relion_dump_sha256": {
                path.name: _sha256(path) for path in sorted(relion_dump_dir.glob("*.bin"))
            },
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
            "pose_comparison": str(pose_comparison.resolve()),
            "pose_comparison_sha256": _sha256(pose_comparison),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-dump-dir", required=True, type=Path)
    parser.add_argument("--recovar-capture", required=True, type=Path)
    parser.add_argument("--pose-comparison", required=True, type=Path)
    parser.add_argument("--source-row", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        relion_dump_dir=args.relion_dump_dir,
        recovar_capture=args.recovar_capture,
        pose_comparison=args.pose_comparison,
        source_row=args.source_row,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
