#!/usr/bin/env python3
"""Audit K=1 cutoff-row support changes against native RELION parents."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import starfile


CHANGE_RE = re.compile(
    r"K=1 cutoff-row support change: original_index=(?P<original>\d+) "
    r"half_local_index=(?P<local>\d+) removed=\[(?P<removed>[^]]*)\] "
    r"added=\[(?P<added>[^]]*)\]"
)
SUMMARY_RE = re.compile(
    r"K=1 deterministic cutoff-row rescore complete: margin=(?P<margin>\S+) "
    r"examined=(?P<examined>\d+) ambiguous=(?P<ambiguous>\d+) "
    r"support_changed_images=(?P<images>\d+) "
    r"support_changed_candidates=(?P<candidates>\d+)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_integer_list(text: str) -> list[int]:
    return [] if not text.strip() else [int(token.strip()) for token in text.split(",")]


def parse_change_log(path: Path) -> tuple[list[dict], dict]:
    """Parse immutable changed-row records and the terminal population summary."""

    records = []
    summaries = []
    for line in path.read_text().splitlines():
        match = CHANGE_RE.search(line)
        if match:
            records.append(
                {
                    "original_index": int(match.group("original")),
                    "half_local_index": int(match.group("local")),
                    "removed_flat_pose_ids": _parse_integer_list(match.group("removed")),
                    "added_flat_pose_ids": _parse_integer_list(match.group("added")),
                }
            )
        match = SUMMARY_RE.search(line)
        if match:
            summaries.append(
                {
                    "margin": float(match.group("margin")),
                    "examined_images": int(match.group("examined")),
                    "ambiguous_images": int(match.group("ambiguous")),
                    "support_changed_images": int(match.group("images")),
                    "support_changed_candidates": int(match.group("candidates")),
                }
            )
    if len(summaries) != 1:
        raise ValueError(f"expected one terminal cutoff summary, found {len(summaries)}")
    summary = summaries[0]
    identities = [record["original_index"] for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("changed-row identities are not unique")
    if len(records) != summary["support_changed_images"]:
        raise ValueError(
            "changed-row record count does not match terminal summary: "
            f"{len(records)} != {summary['support_changed_images']}"
        )
    changed_candidate_count = sum(
        len(record["removed_flat_pose_ids"]) + len(record["added_flat_pose_ids"])
        for record in records
    )
    if changed_candidate_count != summary["support_changed_candidates"]:
        raise ValueError(
            "changed candidate count does not match terminal summary: "
            f"{changed_candidate_count} != {summary['support_changed_candidates']}"
        )
    return records, summary


def _relion_mstep_rotation(eulers_deg: np.ndarray) -> np.ndarray:
    alpha, beta, gamma = np.deg2rad(np.asarray(eulers_deg, dtype=np.float64))
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    matrix = np.asarray(
        [
            [cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
            [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
            [sc, ss, cb],
        ],
        dtype=np.float64,
    )
    return np.linalg.inv(matrix).T.astype(np.float32)


def _stack_index(image_name: object) -> int:
    token = str(image_name).split("@", 1)[0]
    if not token.isdigit():
        raise ValueError(f"unexpected rlnImageName: {image_name!r}")
    return int(token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", required=True, type=Path)
    parser.add_argument("--relion-star", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--coarse-order", type=int, default=3)
    parser.add_argument("--oversampling-order", type=int, default=1)
    parser.add_argument("--random-perturbation", type=float, default=0.00928428769112)
    parser.add_argument("--n-translations", type=int, default=29)
    parser.add_argument("--frobenius-tolerance", type=float, default=2.0e-6)
    args = parser.parse_args()

    records, summary = parse_change_log(args.run_log)
    from recovar.em.sampling import get_oversampled_rotation_grid_from_samples

    coarse_ids = np.arange(36864, dtype=np.int64)
    global_rotations, global_parent_map, _ = get_oversampled_rotation_grid_from_samples(
        coarse_ids,
        args.coarse_order,
        oversampling_order=args.oversampling_order,
        random_perturbation=args.random_perturbation,
        return_rotation_indices=True,
    )
    global_rotations = np.asarray(global_rotations, dtype=np.float32)
    global_parent_map = np.asarray(global_parent_map, dtype=np.int64)
    if not np.array_equal(global_parent_map, np.repeat(coarse_ids, 8)):
        raise ValueError("global fine parent ordering is unexpected")

    star = starfile.read(args.relion_star, always_dict=True)["particles"]
    relion_rows = {_stack_index(row.rlnImageName): row for _, row in star.iterrows()}
    if len(relion_rows) != len(star):
        raise ValueError("RELION stack identities are not unique")

    audited = []
    for record in records:
        original_index = record["original_index"]
        relion = relion_rows[original_index + 1]
        target = _relion_mstep_rotation(
            np.asarray(
                [relion.rlnAngleRot, relion.rlnAngleTilt, relion.rlnAnglePsi],
                dtype=np.float64,
            )
        )
        errors = np.linalg.norm(
            global_rotations.astype(np.float64) - target.astype(np.float64)[None],
            axis=(1, 2),
        )
        global_row = int(np.argmin(errors))
        target_parent = int(global_parent_map[global_row])
        removed = np.asarray(record["removed_flat_pose_ids"], dtype=np.int64)
        added = np.asarray(record["added_flat_pose_ids"], dtype=np.int64)
        removed_target = removed[removed // args.n_translations == target_parent]
        added_target = added[added // args.n_translations == target_parent]
        if added_target.size and not removed_target.size:
            direction = "adds_native_output_parent"
        elif removed_target.size and not added_target.size:
            direction = "removes_native_output_parent"
        elif added_target.size and removed_target.size:
            direction = "changes_translation_within_native_output_parent"
        else:
            direction = "does_not_touch_native_output_parent"
        audited.append(
            {
                **record,
                "stack_index_one_based": original_index + 1,
                "relion_pmax": float(relion.rlnMaxValueProbDistribution),
                "native_nearest_global_fine_row": global_row,
                "native_nearest_global_fine_error": float(errors[global_row]),
                "native_fine_rotation_on_grid": bool(
                    errors[global_row] <= args.frobenius_tolerance
                ),
                "native_coarse_parent_rotation": target_parent,
                "removed_rotation_ids": np.unique(
                    removed // args.n_translations
                ).tolist(),
                "added_rotation_ids": np.unique(added // args.n_translations).tolist(),
                "removed_native_parent_pose_ids": removed_target.tolist(),
                "added_native_parent_pose_ids": added_target.tolist(),
                "native_parent_delta_direction": direction,
            }
        )

    direction_counts = {
        direction: sum(
            record["native_parent_delta_direction"] == direction for record in audited
        )
        for direction in (
            "adds_native_output_parent",
            "removes_native_output_parent",
            "changes_translation_within_native_output_parent",
            "does_not_touch_native_output_parent",
        )
    }
    all_native_on_grid = all(record["native_fine_rotation_on_grid"] for record in audited)
    if not all_native_on_grid:
        classification = "native_output_parent_mapping_not_qualified"
    elif direction_counts["removes_native_output_parent"]:
        classification = "cutoff_rescore_has_native_parent_regressions"
    elif direction_counts["adds_native_output_parent"]:
        classification = "cutoff_rescore_is_directionally_positive_at_native_parent_boundary"
    else:
        classification = "cutoff_rescore_parent_direction_is_neutral"

    report = {
        "schema": "recovar.em.k1_case10_cutoff_changed_parent_panel.v1",
        "status": "complete",
        "classification": classification,
        "metric_policy": (
            "immutable source identity, exact support-delta pose IDs, and direct "
            "rotation-matrix Frobenius mapping; no correlation"
        ),
        "population_summary": summary,
        "changed_record_count": len(audited),
        "direction_counts": direction_counts,
        "all_native_fine_rotations_on_grid": all_native_on_grid,
        "run_log": str(args.run_log.resolve()),
        "run_log_sha256": _sha256(args.run_log),
        "relion_star": str(args.relion_star.resolve()),
        "relion_star_sha256": _sha256(args.relion_star),
        "coarse_order": args.coarse_order,
        "oversampling_order": args.oversampling_order,
        "random_perturbation": args.random_perturbation,
        "n_translations": args.n_translations,
        "frobenius_tolerance": args.frobenius_tolerance,
        "records": audited,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json)


if __name__ == "__main__":
    main()
