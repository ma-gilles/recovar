#!/usr/bin/env python3
"""Audit RECOVAR real-data state against two matched RELION trajectories.

The strict point-reference audit remains authoritative and is never rewritten.
This companion report answers a separate question: whether every RECOVAR
particle state agrees with at least one of two RELION repeats, and whether its
Pmax is within the frozen tolerance of the nearer repeat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recovar.data_io.starfile import read_star

if __package__:
    from scripts.audit_em_particle_state_distribution import _angular_error_deg
else:
    from audit_em_particle_state_distribution import _angular_error_deg


SCHEMA = "recovar.vdam_real_repeatability_envelope.v1"


class RepeatabilityEnvelopeError(RuntimeError):
    """Raised when the paired and repeat trajectories cannot be aligned."""


def _column(table: pd.DataFrame, name: str) -> str:
    matches = [column for column in table.columns if str(column).lstrip("_") == name]
    if len(matches) != 1:
        raise RepeatabilityEnvelopeError(f"expected one {name} column, found {matches}")
    return str(matches[0])


def _aligned(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    ref_id = _column(reference, "rlnImageName")
    candidate_id = _column(candidate, "rlnImageName")
    ref_names = reference[ref_id].astype(str).to_numpy()
    candidate_names = candidate[candidate_id].astype(str).to_numpy()
    if len(set(ref_names.tolist())) != ref_names.size or len(set(candidate_names.tolist())) != candidate_names.size:
        raise RepeatabilityEnvelopeError("particle identities must be unique")
    by_name = {name: row for row, name in enumerate(candidate_names.tolist())}
    if set(ref_names.tolist()) != set(candidate_names.tolist()):
        raise RepeatabilityEnvelopeError("particle identity sets differ")
    return candidate.iloc[[by_name[name] for name in ref_names]].reset_index(drop=True)


def _numeric(table: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    return table[[_column(table, name) for name in names]].astype(float).to_numpy(dtype=np.float64)


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise RepeatabilityEnvelopeError("metric values must be finite and nonempty")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compare_particle_tables_to_reference_set(
    recovar: pd.DataFrame,
    canonical: pd.DataFrame,
    repeat: pd.DataFrame,
    *,
    active_image_ids: set[str],
    pose_tolerance_deg: float,
    translation_tolerance_angst: float,
    pmax_absolute_error_p95_max: float,
    pmax_absolute_error_max: float,
) -> dict[str, Any]:
    """Return strict and nearest-repeat state metrics on one active subset."""

    canonical = _aligned(recovar, canonical)
    repeat = _aligned(recovar, repeat)
    identity_column = _column(recovar, "rlnImageName")
    identities = recovar[identity_column].astype(str).to_numpy()
    active = np.asarray([identity in active_image_ids for identity in identities], dtype=bool)
    if int(np.count_nonzero(active)) != len(active_image_ids) or not np.any(active):
        raise RepeatabilityEnvelopeError("active identities are not a nonempty subset of the particle tables")

    euler_names = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
    translation_names = ("rlnOriginXAngst", "rlnOriginYAngst")
    rec_eulers = _numeric(recovar, euler_names)[active]
    canonical_eulers = _numeric(canonical, euler_names)[active]
    repeat_eulers = _numeric(repeat, euler_names)[active]
    rec_translations = _numeric(recovar, translation_names)[active]
    canonical_translations = _numeric(canonical, translation_names)[active]
    repeat_translations = _numeric(repeat, translation_names)[active]

    def state_match(rhs_eulers, rhs_translations):
        pose = _angular_error_deg(rec_eulers, rhs_eulers)
        translation = np.linalg.norm(rec_translations - rhs_translations, axis=1)
        return (pose <= float(pose_tolerance_deg)) & (
            translation <= float(translation_tolerance_angst)
        )

    matches_canonical = state_match(canonical_eulers, canonical_translations)
    matches_repeat = state_match(repeat_eulers, repeat_translations)
    matches_either = matches_canonical | matches_repeat

    pmax_name = "rlnMaxValueProbDistribution"
    rec_pmax = _numeric(recovar, (pmax_name,))[active, 0]
    canonical_pmax = _numeric(canonical, (pmax_name,))[active, 0]
    repeat_pmax = _numeric(repeat, (pmax_name,))[active, 0]
    canonical_error = np.abs(rec_pmax - canonical_pmax)
    repeat_error = np.abs(rec_pmax - repeat_pmax)
    nearest_error = np.minimum(canonical_error, repeat_error)
    control_error = np.abs(canonical_pmax - repeat_pmax)
    nearest_summary = _summary(nearest_error)
    checks = {
        "all_states_match_at_least_one_relion_repeat": bool(np.all(matches_either)),
        "nearest_repeat_pmax_p95_within_tolerance": nearest_summary["p95"]
        <= float(pmax_absolute_error_p95_max),
        "nearest_repeat_pmax_max_within_tolerance": nearest_summary["max"]
        <= float(pmax_absolute_error_max),
    }
    active_identities = identities[active]
    neither_rows = np.flatnonzero(~matches_either)
    return {
        "evaluated_particle_count": int(active_identities.size),
        "recovar_vs_canonical_state_mismatch_count": int(np.count_nonzero(~matches_canonical)),
        "recovar_vs_repeat_state_mismatch_count": int(np.count_nonzero(~matches_repeat)),
        "recovar_vs_either_state_mismatch_count": int(neither_rows.size),
        "canonical_vs_repeat_pmax_absolute_error": _summary(control_error),
        "recovar_vs_nearest_repeat_pmax_absolute_error": nearest_summary,
        "first_particles_matching_neither": active_identities[neither_rows[:12]].tolist(),
        "checks": checks,
        "pass": all(checks.values()),
    }


def audit(*, paired_root: Path, repeat_relion_dir: Path) -> dict[str, Any]:
    strict = json.loads((paired_root / "trajectory_audit.json").read_text())
    provenance = json.loads((paired_root / "run_provenance.json").read_text())
    scorecard = json.loads(Path(provenance["scorecard"]).read_text())
    case = next(row for row in scorecard["cases"] if row["id"] == provenance["case_id"])
    fixture, _ = read_star(case["input_star"])
    fixture_identity_column = _column(fixture, "rlnImageName")
    thresholds = strict["thresholds"]
    checkpoints = []
    for strict_checkpoint in strict["checkpoints"]:
        iteration = int(strict_checkpoint["iteration"])
        if iteration == 0:
            continue
        meta = json.loads((paired_root / "recovar" / f"run_it{iteration:03d}_recovar_meta.json").read_text())
        selected = np.asarray(meta["selected_particle_ids"], dtype=np.int64)
        recovar, _ = read_star(str(paired_root / "recovar" / f"run_it{iteration:03d}_data.star"))
        canonical, _ = read_star(str(paired_root / "relion" / f"run_it{iteration:03d}_data.star"))
        repeat, _ = read_star(str(repeat_relion_dir / f"run_it{iteration:03d}_data.star"))
        if np.any(selected < 0) or np.any(selected >= len(fixture)):
            raise RepeatabilityEnvelopeError(f"iteration {iteration} has invalid selected particle ids")
        active = set(fixture.iloc[selected][fixture_identity_column].astype(str).tolist())
        comparison = compare_particle_tables_to_reference_set(
            recovar,
            canonical,
            repeat,
            active_image_ids=active,
            pose_tolerance_deg=float(thresholds["pose_tolerance_deg"]),
            translation_tolerance_angst=float(thresholds["translation_tolerance_angst"]),
            pmax_absolute_error_p95_max=float(thresholds["pmax_absolute_error_p95_max"]),
            pmax_absolute_error_max=float(thresholds["pmax_absolute_error_max"]),
        )
        checkpoints.append({"iteration": iteration, **comparison})

    map_gate = all(
        float(checkpoint["cross_engine"]["fsc_auc"])
        >= float(thresholds["cross_engine_fsc_auc_min"])
        for checkpoint in strict["checkpoints"]
    )
    calibrated_pass = map_gate and all(checkpoint["pass"] for checkpoint in checkpoints)
    return {
        "schema": SCHEMA,
        "paired_root": str(paired_root.resolve()),
        "repeat_relion_dir": str(repeat_relion_dir.resolve()),
        "strict_point_reference_result": strict["result"],
        "strict_point_reference_is_preserved": True,
        "map_gate_pass": map_gate,
        "repeatability_calibrated_result": "pass" if calibrated_pass else "fail",
        "thresholds": thresholds,
        "checkpoints": checkpoints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--repeat-relion-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        paired_root=args.paired_root.resolve(),
        repeat_relion_dir=args.repeat_relion_dir.resolve(),
    )
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["repeatability_calibrated_result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
