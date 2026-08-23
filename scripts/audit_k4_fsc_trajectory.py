#!/usr/bin/env python3
"""Strictly audit a complete K=4 RECOVAR/RELION FSC trajectory.

RECOVAR intermediate names are zero-based and use the production debug-dump
spelling ``it000_half1_class1_reg.mrc`` (the class number is not padded).
Their two regularized half maps are averaged before comparison.  RELION
Class3D products are one-based full maps named ``run_it001_class001.mrc``;
Class3D does not emit AutoRefine-style numbered half maps.  Classes are matched
independently at every iteration with a Hungarian assignment that maximizes
normalized FSC-AUC.  Map quality is reported only with shellwise FSC and
normalized FSC-AUC; correlation is intentionally not computed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

if __package__:
    from scripts.summarize_em_completion_bench import (
        _load_recovar_volume,
        _load_relion_volume,
        normalized_fsc_auc,
        shell_fsc,
    )
else:
    from summarize_em_completion_bench import (
        _load_recovar_volume,
        _load_relion_volume,
        normalized_fsc_auc,
        shell_fsc,
    )


SCHEMA = "em_k4_fsc_trajectory_audit_v2"
N_CLASSES = 4
RECOVAR_MAP_RE = re.compile(r"^it(\d{3})_half([12])_class(\d{1,3})_reg\.mrc$")
RELION_MAP_RE = re.compile(r"^run_it(\d{3})_class(\d{3})\.mrc$")


class AuditError(RuntimeError):
    """Raised when required topology or products are incomplete."""


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _metric(curve: np.ndarray, shellwise_key: str) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    auc = float(normalized_fsc_auc(values))
    return {
        "fsc_auc": _finite(auc),
        "n_shells": int(values.size),
        "shellwise_key": shellwise_key,
    }


def _map_metric(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    shellwise_key: str,
    shellwise: dict[str, np.ndarray],
) -> dict[str, Any]:
    curve = np.asarray(shell_fsc(lhs, rhs), dtype=np.float64)
    if curve.size <= 1 or np.count_nonzero(np.isfinite(curve[1:])) < 1:
        raise AuditError(f"{shellwise_key} produced no finite non-DC FSC shells")
    shellwise[shellwise_key] = curve
    return _metric(curve, shellwise_key)


def _fsc_auc(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(normalized_fsc_auc(np.asarray(shell_fsc(lhs, rhs), dtype=np.float64)))


def _hungarian_max(scores: np.ndarray, *, label: str) -> list[int]:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (N_CLASSES, N_CLASSES):
        raise AuditError(f"{label} score matrix has shape {values.shape}, expected {(N_CLASSES, N_CLASSES)}")
    if not np.isfinite(values).all():
        raise AuditError(f"{label} score matrix contains non-finite values")
    rows, cols = linear_sum_assignment(-values)
    if not np.array_equal(rows, np.arange(N_CLASSES)):
        raise AuditError(f"{label} Hungarian assignment did not cover every class")
    return [int(value) for value in cols]


def _discover_recovar_maps(directory: Path) -> dict[int, dict[int, dict[int, Path]]]:
    grouped: dict[int, dict[int, dict[int, Path]]] = {}
    for path in directory.glob("*.mrc"):
        match = RECOVAR_MAP_RE.match(path.name)
        if match is None:
            continue
        iteration, half, class_id = (int(match.group(i)) for i in range(1, 4))
        slot = grouped.setdefault(iteration, {}).setdefault(half, {})
        if class_id in slot:
            raise AuditError(f"duplicate RECOVAR iteration {iteration} half {half} class {class_id}")
        slot[class_id] = path
    return grouped


def _discover_relion_maps(directory: Path) -> dict[int, dict[int, Path]]:
    grouped: dict[int, dict[int, Path]] = {}
    for path in directory.glob("*.mrc"):
        match = RELION_MAP_RE.match(path.name)
        if match is None:
            continue
        iteration, class_id = (int(match.group(i)) for i in range(1, 3))
        if iteration == 0:
            continue
        slot = grouped.setdefault(iteration, {})
        if class_id in slot:
            raise AuditError(f"duplicate RELION iteration {iteration} class {class_id}")
        slot[class_id] = path
    return grouped


def _validate_recovar_topology(maps: dict[int, dict[int, dict[int, Path]]]) -> None:
    if not maps:
        raise AuditError("no RECOVAR numbered K=4 half maps found")
    iterations = sorted(maps)
    expected_iterations = list(range(len(iterations)))
    if iterations != expected_iterations:
        raise AuditError(
            f"RECOVAR iterations are not contiguous zero-based: found {iterations}, expected {expected_iterations}"
        )
    expected_classes = set(range(1, N_CLASSES + 1))
    incomplete: list[str] = []
    for iteration, halves in maps.items():
        if set(halves) != {1, 2}:
            incomplete.append(f"it{iteration:03d} halves={sorted(halves)}")
            continue
        for half in (1, 2):
            classes = set(halves[half])
            if classes != expected_classes:
                incomplete.append(f"it{iteration:03d} half{half} classes={sorted(classes)}")
    if incomplete:
        raise AuditError(f"RECOVAR numbered K=4 topology is incomplete: {incomplete}")


def _validate_relion_topology(maps: dict[int, dict[int, Path]]) -> None:
    if not maps:
        raise AuditError("no RELION numbered Class3D K=4 full maps found")
    iterations = sorted(maps)
    expected_iterations = list(range(1, len(iterations) + 1))
    if iterations != expected_iterations:
        raise AuditError(
            f"RELION iterations are not contiguous one-based: found {iterations}, expected {expected_iterations}"
        )
    expected_classes = set(range(1, N_CLASSES + 1))
    incomplete = [
        f"it{iteration:03d} classes={sorted(classes)}"
        for iteration, classes in maps.items()
        if set(classes) != expected_classes
    ]
    if incomplete:
        raise AuditError(f"RELION numbered Class3D K=4 topology is incomplete: {incomplete}")


def _validate_numbered_topology(
    recovar_maps: dict[int, dict[int, dict[int, Path]]],
    relion_maps: dict[int, dict[int, Path]],
    refinement_results: Path,
) -> list[tuple[int, int]]:
    rec_iterations = sorted(recovar_maps)
    rel_iterations = sorted(relion_maps)
    _validate_recovar_topology(recovar_maps)
    _validate_relion_topology(relion_maps)
    if len(rec_iterations) != len(rel_iterations):
        raise AuditError(
            f"numbered iteration count mismatch: RECOVAR={len(rec_iterations)} RELION={len(rel_iterations)}"
        )
    if not refinement_results.is_file():
        raise AuditError(f"missing RECOVAR refinement results: {refinement_results}")
    with np.load(refinement_results, allow_pickle=False) as payload:
        if "current_sizes" not in payload.files:
            raise AuditError(f"missing current_sizes in {refinement_results}")
        result_count = int(np.asarray(payload["current_sizes"]).size)
    if result_count != len(rec_iterations):
        raise AuditError(
            f"RECOVAR map/result iteration count mismatch: maps={len(rec_iterations)} current_sizes={result_count}"
        )
    return list(zip(rec_iterations, rel_iterations, strict=True))


def _load_recovar_numbered_classes(
    paths: dict[int, dict[int, Path]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    half1 = [_load_recovar_volume(paths[1][class_id]) for class_id in range(1, N_CLASSES + 1)]
    half2 = [_load_recovar_volume(paths[2][class_id]) for class_id in range(1, N_CLASSES + 1)]
    merged = [0.5 * (lhs + rhs) for lhs, rhs in zip(half1, half2, strict=True)]
    return half1, half2, merged


def _load_relion_numbered_classes(paths: dict[int, Path]) -> list[np.ndarray]:
    return [_load_relion_volume(paths[class_id]) for class_id in range(1, N_CLASSES + 1)]


def _assignment_score_matrix(lhs: list[np.ndarray], rhs: list[np.ndarray]) -> np.ndarray:
    return np.asarray([[_fsc_auc(a, b) for b in rhs] for a in lhs], dtype=np.float64)


def _gt_pair_assignment(
    rec_merged: list[np.ndarray],
    rel_merged: list[np.ndarray],
    rel_for_rec: list[int],
    gt: list[np.ndarray],
) -> tuple[list[int], np.ndarray]:
    scores = np.empty((N_CLASSES, N_CLASSES), dtype=np.float64)
    for rec_id in range(N_CLASSES):
        rel_id = rel_for_rec[rec_id]
        for gt_id in range(N_CLASSES):
            scores[rec_id, gt_id] = 0.5 * (
                _fsc_auc(rec_merged[rec_id], gt[gt_id]) + _fsc_auc(rel_merged[rel_id], gt[gt_id])
            )
    return _hungarian_max(scores, label="matched-pair-to-GT"), scores


def _selected_class_metrics(
    *,
    prefix: str,
    rec_merged: list[np.ndarray],
    rel_full: list[np.ndarray],
    gt: list[np.ndarray],
    rel_for_rec: list[int],
    gt_for_rec: list[int],
    shellwise: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec_id in range(N_CLASSES):
        rel_id = rel_for_rec[rec_id]
        gt_id = gt_for_rec[rec_id]
        cross = _map_metric(
            rec_merged[rec_id],
            rel_full[rel_id],
            shellwise_key=f"{prefix}_rec{rec_id + 1:03d}_rel{rel_id + 1:03d}_cross",
            shellwise=shellwise,
        )
        rec_gt = _map_metric(
            rec_merged[rec_id],
            gt[gt_id],
            shellwise_key=f"{prefix}_rec{rec_id + 1:03d}_gt{gt_id + 1:03d}",
            shellwise=shellwise,
        )
        rel_gt = _map_metric(
            rel_full[rel_id],
            gt[gt_id],
            shellwise_key=f"{prefix}_rel{rel_id + 1:03d}_gt{gt_id + 1:03d}",
            shellwise=shellwise,
        )
        rec_auc = rec_gt["fsc_auc"]
        rel_auc = rel_gt["fsc_auc"]
        delta = None if rec_auc is None or rel_auc is None else float(rec_auc - rel_auc)
        out.append(
            {
                "recovar_class": rec_id + 1,
                "relion_class": rel_id + 1,
                "gt_class": gt_id + 1,
                "cross_engine": cross,
                "vs_gt": {"recovar": rec_gt, "relion": rel_gt},
                "gt_fsc_auc_delta": delta,
            }
        )
    return out


def _particle_table(path: Path):
    import starfile

    table = starfile.read(path)
    if isinstance(table, dict):
        for key in ("particles", "data_particles"):
            if key in table:
                return table[key]
        candidates = [value for value in table.values() if hasattr(value, "columns")]
        if len(candidates) == 1:
            return candidates[0]
        raise AuditError(f"cannot identify particle table in {path}")
    return table


def _column(table, name: str):
    for candidate in (name, f"_{name}"):
        if candidate in table.columns:
            return table[candidate]
    raise AuditError(f"missing {name}")


def _class_agreement(
    *,
    refinement_results: Path,
    relion_data_star: Path,
    fixture_particles_star: Path,
    recovar_iteration: int,
    rel_for_rec: list[int],
) -> dict[str, Any]:
    key = f"class_assignments_by_image_iter_{recovar_iteration:03d}"
    if not fixture_particles_star.is_file():
        return {"status": "unavailable", "reason": f"missing fixture particle STAR: {fixture_particles_star}"}
    if not relion_data_star.is_file():
        return {"status": "unavailable", "reason": f"missing RELION iteration data STAR: {relion_data_star}"}
    with np.load(refinement_results, allow_pickle=False) as payload:
        if key not in payload.files:
            return {"status": "unavailable", "reason": f"missing {key} in {refinement_results}"}
        rec = np.asarray(payload[key], dtype=np.int64).reshape(-1)
    try:
        fixture = _particle_table(fixture_particles_star)
        relion = _particle_table(relion_data_star)
        fixture_names = [str(value) for value in _column(fixture, "rlnImageName")]
        relion_names = [str(value) for value in _column(relion, "rlnImageName")]
        relion_classes = np.asarray(_column(relion, "rlnClassNumber"), dtype=np.int64) - 1
    except Exception as exc:
        return {"status": "unavailable", "reason": f"failed to align class assignments by image identity: {exc}"}
    if len(set(fixture_names)) != len(fixture_names) or len(set(relion_names)) != len(relion_names):
        return {"status": "unavailable", "reason": "image identities are not unique"}
    if len(rec) != len(fixture_names):
        return {
            "status": "unavailable",
            "reason": f"RECOVAR assignment length {len(rec)} != fixture particle count {len(fixture_names)}",
        }
    relion_by_name = dict(zip(relion_names, relion_classes, strict=True))
    if set(relion_by_name) != set(fixture_names):
        return {"status": "unavailable", "reason": "RELION and fixture image identity sets differ"}
    rel_ordered = np.asarray([relion_by_name[name] for name in fixture_names], dtype=np.int64)
    valid = (rec >= 0) & (rec < N_CLASSES) & (rel_ordered >= 0) & (rel_ordered < N_CLASSES)
    if not valid.all():
        return {"status": "unavailable", "reason": "class arrays contain missing or out-of-range class ids"}
    mapped = np.asarray(rel_for_rec, dtype=np.int64)[rec]
    return {
        "status": "available",
        "matched_count": int(valid.sum()),
        "agreement": float(np.mean(mapped == rel_ordered)),
        "recovar_to_relion_permutation": [value + 1 for value in rel_for_rec],
        "recovar_assignment_key": key,
        "relion_data_star": str(relion_data_star),
        "fixture_particles_star": str(fixture_particles_star),
    }


def _numbered_row(
    *,
    rec_iteration: int,
    rel_iteration: int,
    rec_paths: dict[int, dict[int, Path]],
    rel_paths: dict[int, Path],
    gt: list[np.ndarray],
    shellwise: dict[str, np.ndarray],
    refinement_results: Path,
    relion_dir: Path,
    fixture_particles_star: Path,
) -> dict[str, Any]:
    rec = _load_recovar_numbered_classes(rec_paths)
    rel = _load_relion_numbered_classes(rel_paths)
    score_matrix = _assignment_score_matrix(rec[2], rel)
    rel_for_rec = _hungarian_max(score_matrix, label=f"it{rel_iteration:03d} RECOVAR-to-RELION")
    gt_for_rec, gt_score_matrix = _gt_pair_assignment(rec[2], rel, rel_for_rec, gt)
    return {
        "recovar_index": rec_iteration,
        "relion_iteration": rel_iteration,
        "recovar_to_relion_assignment": [value + 1 for value in rel_for_rec],
        "matched_pair_to_gt_assignment": [value + 1 for value in gt_for_rec],
        "pairwise_cross_engine_fsc_auc": score_matrix.tolist(),
        "pairwise_matched_pair_gt_mean_fsc_auc": gt_score_matrix.tolist(),
        "classes": _selected_class_metrics(
            prefix=f"it{rel_iteration:03d}",
            rec_merged=rec[2],
            rel_full=rel,
            gt=gt,
            rel_for_rec=rel_for_rec,
            gt_for_rec=gt_for_rec,
            shellwise=shellwise,
        ),
        "class_agreement": _class_agreement(
            refinement_results=refinement_results,
            relion_data_star=relion_dir / f"run_it{rel_iteration:03d}_data.star",
            fixture_particles_star=fixture_particles_star,
            recovar_iteration=rec_iteration,
            rel_for_rec=rel_for_rec,
        ),
    }


def _discover_final(directory: Path, pattern: re.Pattern[str], *, engine: str) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in directory.glob("*.mrc"):
        match = pattern.match(path.name)
        if match is not None:
            found[int(match.group(1))] = path
    expected = set(range(1, N_CLASSES + 1))
    if set(found) != expected:
        raise AuditError(
            f"{engine} final K=4 products are incomplete: classes={sorted(found)}, expected={sorted(expected)}"
        )
    return found


def _finalization_state(refinement_results: Path) -> dict[str, bool]:
    with np.load(refinement_results, allow_pickle=False) as payload:
        required = ("convergence_has_converged", "final_all_data_ran")
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise AuditError(f"missing finalization state {missing} in {refinement_results}")
        state = {key: bool(np.asarray(payload[key]).item()) for key in required}
    if state["final_all_data_ran"]:
        raise AuditError(
            "K=4 final-all-data products require a dedicated RELION comparator; "
            "the last-numbered Class3D comparator is valid only when final_all_data_ran=false"
        )
    return state


def _final_metrics(
    recovar_dir: Path,
    recovar_last_paths: dict[int, dict[int, Path]],
    relion_last_paths: dict[int, Path],
    *,
    recovar_iteration: int,
    relion_iteration: int,
    gt: list[np.ndarray],
    shellwise: dict[str, np.ndarray],
) -> dict[str, Any]:
    rec_paths = _discover_final(recovar_dir, re.compile(r"^final_class(\d{3})\.mrc$"), engine="RECOVAR")
    rec = [_load_recovar_volume(rec_paths[class_id]) for class_id in range(1, N_CLASSES + 1)]
    rec_last = _load_recovar_numbered_classes(recovar_last_paths)[2]
    for class_id, (final_map, numbered_map) in enumerate(zip(rec, rec_last, strict=True), start=1):
        if not np.array_equal(final_map, numbered_map):
            raise AuditError(
                f"RECOVAR final class {class_id} does not exactly match the last numbered "
                f"half-average at iteration {recovar_iteration}"
            )
    rel = _load_relion_numbered_classes(relion_last_paths)
    score_matrix = _assignment_score_matrix(rec, rel)
    rel_for_rec = _hungarian_max(score_matrix, label="final RECOVAR-to-RELION")
    gt_for_rec, gt_score_matrix = _gt_pair_assignment(rec, rel, rel_for_rec, gt)
    classes = []
    for rec_id in range(N_CLASSES):
        rel_id, gt_id = rel_for_rec[rec_id], gt_for_rec[rec_id]
        cross = _map_metric(
            rec[rec_id],
            rel[rel_id],
            shellwise_key=f"final_rec{rec_id + 1:03d}_rel{rel_id + 1:03d}_cross",
            shellwise=shellwise,
        )
        rec_gt = _map_metric(
            rec[rec_id],
            gt[gt_id],
            shellwise_key=f"final_rec{rec_id + 1:03d}_gt{gt_id + 1:03d}",
            shellwise=shellwise,
        )
        rel_gt = _map_metric(
            rel[rel_id],
            gt[gt_id],
            shellwise_key=f"final_rel{rel_id + 1:03d}_gt{gt_id + 1:03d}",
            shellwise=shellwise,
        )
        rec_auc, rel_auc = rec_gt["fsc_auc"], rel_gt["fsc_auc"]
        classes.append(
            {
                "recovar_class": rec_id + 1,
                "relion_class": rel_id + 1,
                "gt_class": gt_id + 1,
                "cross_engine": cross,
                "vs_gt": {"recovar": rec_gt, "relion": rel_gt},
                "gt_fsc_auc_delta": None if rec_auc is None or rel_auc is None else float(rec_auc - rel_auc),
            }
        )
    return {
        "recovar_source_index": recovar_iteration,
        "relion_source_iteration": relion_iteration,
        "recovar_final_matches_last_numbered_half_average": True,
        "recovar_to_relion_assignment": [value + 1 for value in rel_for_rec],
        "matched_pair_to_gt_assignment": [value + 1 for value in gt_for_rec],
        "pairwise_cross_engine_fsc_auc": score_matrix.tolist(),
        "pairwise_matched_pair_gt_mean_fsc_auc": gt_score_matrix.tolist(),
        "classes": classes,
    }


def _apply_gates(
    rows: list[dict[str, Any]],
    final: dict[str, Any],
    *,
    min_cross: float,
    min_gt_delta: float,
    min_class_agreement: float,
) -> list[str]:
    failures: list[str] = []
    for label, classes in [
        *[(f"it{row['relion_iteration']:03d}", row["classes"]) for row in rows],
        ("final", final["classes"]),
    ]:
        for item in classes:
            class_label = f"{label} RECOVAR class {item['recovar_class']} / RELION class {item['relion_class']}"
            cross = item["cross_engine"]["fsc_auc"]
            delta = item["gt_fsc_auc_delta"]
            if cross is None or not math.isfinite(float(cross)) or float(cross) < min_cross:
                failures.append(f"{class_label} direct FSC-AUC {cross!r} < {min_cross:.9f}")
            if delta is None or not math.isfinite(float(delta)) or float(delta) < min_gt_delta:
                failures.append(f"{class_label} GT FSC-AUC delta {delta!r} < {min_gt_delta:+.9f}")
    for row in rows:
        agreement = row["class_agreement"]
        if agreement["status"] == "available" and float(agreement["agreement"]) < min_class_agreement:
            failures.append(
                f"it{row['relion_iteration']:03d} class agreement {float(agreement['agreement']):.9f} "
                f"< {min_class_agreement:.9f}"
            )
    return failures


def audit_case(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case_root = args.case_root.resolve()
    recovar_dir = (args.recovar_dir or case_root / "recovar").resolve()
    relion_dir = (args.relion_dir or case_root / "relion_ref").resolve()
    gt_dir = (args.gt_dir or case_root / "data").resolve()
    fixture_particles_star = (args.fixture_particles_star or gt_dir / "particles.star").resolve()
    intermediates = recovar_dir / "intermediates"
    if not intermediates.is_dir():
        raise AuditError(f"missing RECOVAR intermediates directory: {intermediates}")
    if not relion_dir.is_dir():
        raise AuditError(f"missing RELION directory: {relion_dir}")
    gt_paths = [gt_dir / f"reference_gt_class{class_id:03d}.mrc" for class_id in range(1, N_CLASSES + 1)]
    missing_gt = [str(path) for path in gt_paths if not path.is_file()]
    if missing_gt:
        raise AuditError(f"missing K=4 GT maps: {missing_gt}")
    gt = [_load_recovar_volume(path) for path in gt_paths]

    rec_maps = _discover_recovar_maps(intermediates)
    rel_maps = _discover_relion_maps(relion_dir)
    refinement_results = recovar_dir / "refinement_results.npz"
    pairs = _validate_numbered_topology(rec_maps, rel_maps, refinement_results)
    finalization = _finalization_state(refinement_results)
    shellwise: dict[str, np.ndarray] = {}
    rows = [
        _numbered_row(
            rec_iteration=rec_iteration,
            rel_iteration=rel_iteration,
            rec_paths=rec_maps[rec_iteration],
            rel_paths=rel_maps[rel_iteration],
            gt=gt,
            shellwise=shellwise,
            refinement_results=refinement_results,
            relion_dir=relion_dir,
            fixture_particles_star=fixture_particles_star,
        )
        for rec_iteration, rel_iteration in pairs
    ]
    final_rec_iteration, final_rel_iteration = pairs[-1]
    final = _final_metrics(
        recovar_dir,
        rec_maps[final_rec_iteration],
        rel_maps[final_rel_iteration],
        recovar_iteration=final_rec_iteration,
        relion_iteration=final_rel_iteration,
        gt=gt,
        shellwise=shellwise,
    )
    failures = _apply_gates(
        rows,
        final,
        min_cross=float(args.min_cross_fsc_auc),
        min_gt_delta=float(args.min_gt_delta),
        min_class_agreement=float(args.min_class_agreement),
    )
    unavailable = [
        {"relion_iteration": row["relion_iteration"], "reason": row["class_agreement"]["reason"]}
        for row in rows
        if row["class_agreement"]["status"] == "unavailable"
    ]
    return (
        {
            "schema": SCHEMA,
            "status": "pass" if not failures else "fail",
            "quality_metric_policy": "shellwise FSC and normalized FSC-AUC only; correlation is not computed",
            "paths": {
                "case_root": str(case_root),
                "recovar_dir": str(recovar_dir),
                "relion_dir": str(relion_dir),
                "gt_dir": str(gt_dir),
                "fixture_particles_star": str(fixture_particles_star),
            },
            "n_classes": N_CLASSES,
            "numbered_map_policy": (
                "each RECOVAR per-class map is the arithmetic mean of its two saved regularized half maps; "
                "each RELION Class3D product is the corresponding one-based numbered full map"
            ),
            "final_map_policy": (
                "when final_all_data_ran=false, RECOVAR final_class maps must exactly equal the last numbered "
                "half-average and are compared to RELION's last numbered Class3D full maps"
            ),
            "finalization_state": finalization,
            "thresholds": {
                "per_class_direct_fsc_auc_min": float(args.min_cross_fsc_auc),
                "per_class_recovar_minus_relion_gt_fsc_auc_min": float(args.min_gt_delta),
                "class_assignment_agreement_min_when_available": float(args.min_class_agreement),
            },
            "numbered_iteration_count": len(rows),
            "numbered_iterations": rows,
            "final": final,
            "class_agreement_unavailable": unavailable,
            "failures": failures,
            "earliest_failure": failures[0] if failures else None,
        },
        shellwise,
    )


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.9f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# K=4 FSC trajectory audit",
        "",
        f"Status: **{str(report['status']).upper()}**",
        "",
        "Shellwise FSC and normalized FSC-AUC are the map-quality metrics. Correlation was not computed.",
        "",
        "| State | REC class | RELION class | GT class | Direct FSC-AUC | REC GT FSC-AUC | RELION GT FSC-AUC | GT delta | Class agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("numbered_iterations", []):
        agreement = row["class_agreement"]
        agreement_text = _fmt(agreement.get("agreement")) if agreement["status"] == "available" else "unavailable"
        for item in row["classes"]:
            lines.append(
                f"| it{row['relion_iteration']:03d} | {item['recovar_class']} | {item['relion_class']} | "
                f"{item['gt_class']} | {_fmt(item['cross_engine']['fsc_auc'])} | "
                f"{_fmt(item['vs_gt']['recovar']['fsc_auc'])} | "
                f"{_fmt(item['vs_gt']['relion']['fsc_auc'])} | "
                f"{_fmt(item['gt_fsc_auc_delta'])} | {agreement_text} |"
            )
    final = report.get("final")
    if final:
        for item in final["classes"]:
            lines.append(
                f"| final | {item['recovar_class']} | {item['relion_class']} | {item['gt_class']} | "
                f"{_fmt(item['cross_engine']['fsc_auc'])} | {_fmt(item['vs_gt']['recovar']['fsc_auc'])} | "
                f"{_fmt(item['vs_gt']['relion']['fsc_auc'])} | {_fmt(item['gt_fsc_auc_delta'])} | — |"
            )
    unavailable = report.get("class_agreement_unavailable") or []
    if unavailable:
        lines.extend(["", "## Class-agreement availability", ""])
        lines.extend(f"- it{item['relion_iteration']:03d}: {item['reason']}" for item in unavailable)
    failures = report.get("failures") or []
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--recovar-dir", type=Path)
    parser.add_argument("--relion-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--fixture-particles-star", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--output-shellwise-npz", type=Path)
    parser.add_argument("--min-cross-fsc-auc", type=float, default=0.995)
    parser.add_argument("--min-gt-delta", type=float, default=-0.002)
    parser.add_argument("--min-class-agreement", type=float, default=0.99)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    analysis_dir = args.case_root.resolve() / "trajectory_analysis"
    output_json = (args.output_json or analysis_dir / "k4_fsc_trajectory.json").resolve()
    output_markdown = (args.output_markdown or analysis_dir / "k4_fsc_trajectory.md").resolve()
    output_npz = (args.output_shellwise_npz or analysis_dir / "k4_fsc_trajectory_shellwise.npz").resolve()
    for path in (output_json, output_markdown, output_npz):
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report, shellwise = audit_case(args)
    except AuditError as exc:
        report = {
            "schema": SCHEMA,
            "status": "error",
            "quality_metric_policy": "shellwise FSC and normalized FSC-AUC only; correlation is not computed",
            "failures": [str(exc)],
            "earliest_failure": str(exc),
            "numbered_iterations": [],
            "final": None,
            "class_agreement_unavailable": [],
        }
        shellwise = {}
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    output_markdown.write_text(render_markdown(report))
    np.savez_compressed(output_npz, **shellwise)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
