#!/usr/bin/env python3
"""Audit VDAM particle and sampling state against native RELION repeats."""

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
    from scripts.audit_vdam_candidate_native_envelope import (
        SUITE_SCHEMA,
        _candidate_provenance,
        _load_json,
        _native_panel_provenance,
        require_same_physical_gpu,
    )
    from scripts.audit_vdam_sampling_trajectory import audit_sampling_trajectory
else:
    from audit_em_particle_state_distribution import _angular_error_deg
    from audit_vdam_candidate_native_envelope import (
        SUITE_SCHEMA,
        _candidate_provenance,
        _load_json,
        _native_panel_provenance,
        require_same_physical_gpu,
    )
    from audit_vdam_sampling_trajectory import audit_sampling_trajectory


SCHEMA = "recovar.vdam_candidate_state_envelope.v1"
POSE_TOLERANCE_DEG = 1e-3
TRANSLATION_TOLERANCE_ANGST = 1e-4
_ACCURACY_CHECKS = frozenset(("accuracy_rotation", "accuracy_translation"))
_DIAGNOSTIC_ONLY_FIELDS = frozenset(("nr_iter_without_resolution_gain",))


class CandidateStateEnvelopeError(RuntimeError):
    """Raised when state-envelope evidence is incomplete or ambiguous."""


def _column(table: pd.DataFrame, name: str) -> str:
    matches = [str(column) for column in table.columns if str(column).lstrip("_") == name]
    if len(matches) != 1:
        raise CandidateStateEnvelopeError(f"expected one {name} column, found {matches}")
    return matches[0]


def _numeric(table: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    try:
        values = table[[_column(table, name) for name in names]].astype(float).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CandidateStateEnvelopeError(f"non-numeric particle columns: {names}") from exc
    if not np.all(np.isfinite(values)):
        raise CandidateStateEnvelopeError(f"non-finite particle columns: {names}")
    return values


def _aligned(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    reference_name = _column(reference, "rlnImageName")
    candidate_name = _column(candidate, "rlnImageName")
    reference_ids = reference[reference_name].astype(str).to_numpy()
    candidate_ids = candidate[candidate_name].astype(str).to_numpy()
    if len(set(reference_ids.tolist())) != reference_ids.size:
        raise CandidateStateEnvelopeError("candidate particle identities are not unique")
    if len(set(candidate_ids.tolist())) != candidate_ids.size:
        raise CandidateStateEnvelopeError("native particle identities are not unique")
    if set(reference_ids.tolist()) != set(candidate_ids.tolist()):
        raise CandidateStateEnvelopeError("candidate/native particle identity sets differ")
    by_name = {name: row for row, name in enumerate(candidate_ids.tolist())}
    return candidate.iloc[[by_name[name] for name in reference_ids]].reset_index(drop=True)


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise CandidateStateEnvelopeError("state metric values must be finite and nonempty")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compare_particle_tables_to_native_set(
    candidate: pd.DataFrame,
    native_tables: list[pd.DataFrame],
    *,
    active_image_ids: set[str],
    pose_tolerance_deg: float = POSE_TOLERANCE_DEG,
    translation_tolerance_angst: float = TRANSLATION_TOLERANCE_ANGST,
) -> dict[str, Any]:
    """Require every active candidate particle to match one native state."""

    if len(native_tables) < 2:
        raise CandidateStateEnvelopeError("particle envelope requires at least two native repeats")
    aligned = [_aligned(candidate, table) for table in native_tables]
    identity_column = _column(candidate, "rlnImageName")
    identities = candidate[identity_column].astype(str).to_numpy()
    active = np.asarray([identity in active_image_ids for identity in identities], dtype=bool)
    if not active_image_ids or int(np.count_nonzero(active)) != len(active_image_ids):
        raise CandidateStateEnvelopeError("active particle identities are not an exact nonempty subset")

    euler_names = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
    translation_names = ("rlnOriginXAngst", "rlnOriginYAngst")
    candidate_eulers = _numeric(candidate, euler_names)[active]
    candidate_translations = _numeric(candidate, translation_names)[active]
    candidate_pmax = _numeric(candidate, ("rlnMaxValueProbDistribution",))[active, 0]
    pose_errors = []
    translation_errors = []
    pmax_errors = []
    for native in aligned:
        pose_errors.append(_angular_error_deg(candidate_eulers, _numeric(native, euler_names)[active]))
        translation_errors.append(
            np.linalg.norm(candidate_translations - _numeric(native, translation_names)[active], axis=1)
        )
        pmax_errors.append(
            np.abs(candidate_pmax - _numeric(native, ("rlnMaxValueProbDistribution",))[active, 0])
        )
    pose = np.stack(pose_errors, axis=0)
    translation = np.stack(translation_errors, axis=0)
    pmax = np.stack(pmax_errors, axis=0)
    matches = (pose <= float(pose_tolerance_deg)) & (
        translation <= float(translation_tolerance_angst)
    )
    matches_any = np.any(matches, axis=0)
    active_identities = identities[active]
    unmatched = np.flatnonzero(~matches_any)
    return {
        "pass": bool(np.all(matches_any)),
        "evaluated_particle_count": int(active_identities.size),
        "native_repeat_count": len(native_tables),
        "candidate_vs_each_native_mismatch_count": [
            int(np.count_nonzero(~row)) for row in matches
        ],
        "candidate_vs_native_envelope_mismatch_count": int(unmatched.size),
        "candidate_vs_nearest_native_pose_error_deg": _summary(np.min(pose, axis=0)),
        "candidate_vs_nearest_native_translation_error_angst": _summary(
            np.min(translation, axis=0)
        ),
        "candidate_vs_nearest_native_pmax_absolute_error": _summary(np.min(pmax, axis=0)),
        "first_particles_matching_no_native_repeat": active_identities[unmatched[:12]].tolist(),
    }


def classify_schedule_mode_envelope(
    native_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require one complete native schedule mode, not a cross-repeat mixture."""

    if len(native_rows) < 2:
        raise CandidateStateEnvelopeError("schedule envelope requires at least two native repeats")
    iterations = {int(row["iteration"]) for row in native_rows}
    if len(iterations) != 1:
        raise CandidateStateEnvelopeError("native schedule rows do not describe one iteration")
    accuracy_estimated = bool(native_rows[0]["candidate"]["sampling_accuracy_estimated"])
    if any(
        bool(row["candidate"]["sampling_accuracy_estimated"]) != accuracy_estimated
        for row in native_rows[1:]
    ):
        raise CandidateStateEnvelopeError("candidate accuracy-estimation state differs across audits")
    normalized_checks = []
    diagnostic_checks: list[dict[str, bool]] = []
    for row in native_rows:
        checks = dict(row["checks"])
        normalized_checks.append(checks)
        diagnostic_checks.append(
            {
                name: int(row["candidate"][name]) == int(row["native"][name])
                for name in _DIAGNOSTIC_ONLY_FIELDS
                if name in row["candidate"] and name in row["native"]
            }
        )
    check_names = set(normalized_checks[0])
    if any(set(checks) != check_names for checks in normalized_checks[1:]):
        raise CandidateStateEnvelopeError("native schedule audits expose different check sets")
    active_checks = sorted(check_names if accuracy_estimated else check_names - _ACCURACY_CHECKS)
    matching_repeats = [
        index
        for index, checks in enumerate(normalized_checks, start=1)
        if all(bool(checks[name]) for name in active_checks)
    ]
    return {
        "pass": bool(matching_repeats),
        "accuracy_checks_active": accuracy_estimated,
        "active_checks": active_checks,
        "matching_native_repeat_indices": matching_repeats,
        "checks_matching_at_least_one_native": {
            name: any(bool(checks[name]) for checks in normalized_checks)
            for name in active_checks
        },
        "diagnostic_only_fields": sorted(_DIAGNOSTIC_ONLY_FIELDS),
        "diagnostics_matching_at_least_one_native": {
            name: any(bool(checks.get(name, False)) for checks in diagnostic_checks)
            for name in sorted(_DIAGNOSTIC_ONLY_FIELDS)
        },
    }


def audit_candidate_state_envelope(
    *,
    scorecard_path: Path,
    case_id: str,
    candidate_root: Path,
    native_roots: list[Path],
    fixture_dir: Path,
    pixel_size: float,
) -> dict[str, Any]:
    scorecard = _load_json(scorecard_path, label="scorecard")
    if scorecard.get("schema") != SUITE_SCHEMA:
        raise CandidateStateEnvelopeError(f"unsupported scorecard schema: {scorecard.get('schema')!r}")
    matches = [row for row in scorecard.get("cases", ()) if row.get("id") == case_id]
    if len(matches) != 1:
        raise CandidateStateEnvelopeError(f"expected one scorecard row for {case_id}")
    checkpoints = tuple(int(value) for value in scorecard["acceptance_contract"]["required_checkpoints"])
    positive_iterations = tuple(value for value in checkpoints if value > 0)
    candidate_provenance = _candidate_provenance(candidate_root)
    native_provenance = _native_panel_provenance(
        native_roots,
        suite_id=str(scorecard["suite_id"]),
        case_id=case_id,
        checkpoints=checkpoints,
    )
    require_same_physical_gpu(candidate_provenance, native_provenance)

    materialization = _load_json(
        fixture_dir / "fixture_materialization.json", label="fixture materialization"
    )
    if materialization.get("manifest_sha256") != scorecard["source_fixture_manifest"]["sha256"]:
        raise CandidateStateEnvelopeError("fixture identity differs from the frozen scorecard")
    fixture, _ = read_star(str(fixture_dir / "particles.star"))
    fixture_identity_column = _column(fixture, "rlnImageName")

    sampling_reports = [
        audit_sampling_trajectory(
            candidate_root / "recovar",
            root / "relion",
            pixel_size=pixel_size,
            iterations=list(positive_iterations),
        )
        for root in native_roots
    ]
    particle_rows = []
    schedule_rows = []
    for offset, iteration in enumerate(positive_iterations):
        meta = _load_json(
            candidate_root / "recovar" / f"run_it{iteration:03d}_recovar_meta.json",
            label=f"candidate iteration {iteration} metadata",
        )
        selected = np.asarray(meta["selected_particle_ids"], dtype=np.int64)
        if selected.size == 0 or np.any(selected < 0) or np.any(selected >= len(fixture)):
            raise CandidateStateEnvelopeError(f"iteration {iteration} has invalid selected particle ids")
        active_ids = set(fixture.iloc[selected][fixture_identity_column].astype(str).tolist())
        if len(active_ids) != selected.size:
            raise CandidateStateEnvelopeError(f"iteration {iteration} selected identities are not unique")
        candidate_table, _ = read_star(
            str(candidate_root / "recovar" / f"run_it{iteration:03d}_data.star")
        )
        native_tables = [
            read_star(str(root / "relion" / f"run_it{iteration:03d}_data.star"))[0]
            for root in native_roots
        ]
        particle_rows.append(
            {
                "iteration": iteration,
                **compare_particle_tables_to_native_set(
                    candidate_table,
                    native_tables,
                    active_image_ids=active_ids,
                ),
            }
        )
        schedule_rows.append(
            {
                "iteration": iteration,
                **classify_schedule_mode_envelope(
                    [report["iterations"][offset] for report in sampling_reports]
                ),
            }
        )

    particle_pass = all(row["pass"] for row in particle_rows)
    schedule_pass = all(row["pass"] for row in schedule_rows)
    return {
        "schema": SCHEMA,
        "suite_id": scorecard["suite_id"],
        "case_id": case_id,
        "result": "pass" if particle_pass and schedule_pass else "fail",
        "scope": "active-particle native-state coverage and whole native sampling-mode coverage",
        "candidate_provenance": candidate_provenance,
        "native_panel_provenance": native_provenance,
        "thresholds": {
            "pose_tolerance_deg": POSE_TOLERANCE_DEG,
            "translation_tolerance_angst": TRANSLATION_TOLERANCE_ANGST,
        },
        "particle_result": "pass" if particle_pass else "fail",
        "schedule_result": "pass" if schedule_pass else "fail",
        "first_particle_failure_iteration": next(
            (row["iteration"] for row in particle_rows if not row["pass"]), None
        ),
        "first_schedule_failure_iteration": next(
            (row["iteration"] for row in schedule_rows if not row["pass"]), None
        ),
        "particle_checkpoints": particle_rows,
        "schedule_checkpoints": schedule_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, action="append", required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--pixel-size", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_candidate_state_envelope(
        scorecard_path=args.scorecard.resolve(),
        case_id=args.case_id,
        candidate_root=args.candidate_root.resolve(),
        native_roots=[path.resolve() for path in args.native_root],
        fixture_dir=args.fixture_dir.resolve(),
        pixel_size=args.pixel_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
