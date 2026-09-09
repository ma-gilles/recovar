#!/usr/bin/env python3
"""Audit VDAM/RELION particle-state drift by exact image identity.

This diagnostic is intentionally separate from the frozen map-quality gate.
It identifies the first particles whose winning pose or translation changes,
and reports Pmax error growth across a numbered autonomous trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from recovar.data_io.starfile import read_star

if __package__:
    from scripts.audit_em_particle_state_distribution import _angular_error_deg
else:
    from audit_em_particle_state_distribution import _angular_error_deg

SCHEMA = "recovar.vdam_particle_state_trajectory_audit.v1"
DEFAULT_ITERATIONS = (1, 2, 3, 4, 8)
IDENTITY_NAMES = ("_rlnImageName", "rlnImageName")
EULER_NAMES = (
    ("_rlnAngleRot", "rlnAngleRot"),
    ("_rlnAngleTilt", "rlnAngleTilt"),
    ("_rlnAnglePsi", "rlnAnglePsi"),
)
TRANSLATION_NAMES = (
    ("_rlnOriginXAngst", "rlnOriginXAngst"),
    ("_rlnOriginYAngst", "rlnOriginYAngst"),
)
PMAX_NAMES = ("_rlnMaxValueProbDistribution", "rlnMaxValueProbDistribution")


class AuditError(RuntimeError):
    """Raised when particle-state evidence is absent or ambiguous."""


def _column(table, names: tuple[str, ...], *, label: str) -> str:
    matches = [name for name in names if name in table.columns]
    if len(matches) != 1:
        raise AuditError(f"{label} must contain exactly one of {names}, found {matches}")
    return matches[0]


def _numeric_matrix(table, names: tuple[tuple[str, ...], ...], *, label: str) -> np.ndarray:
    columns = [_column(table, alternatives, label=label) for alternatives in names]
    try:
        values = table[columns].astype(float).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} contains non-numeric state columns") from exc
    if not np.isfinite(values).all():
        raise AuditError(f"{label} contains non-finite state values")
    return values


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise AuditError("cannot summarize empty or non-finite state errors")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compare_particle_tables(
    recovar_table,
    relion_table,
    *,
    iteration: int,
    pose_tolerance_deg: float,
    translation_tolerance_angst: float,
    example_limit: int = 12,
) -> dict[str, Any]:
    """Compare two particle tables after exact ``rlnImageName`` alignment."""

    recovar_identity_column = _column(recovar_table, IDENTITY_NAMES, label="RECOVAR table")
    relion_identity_column = _column(relion_table, IDENTITY_NAMES, label="RELION table")
    recovar_identities = recovar_table[recovar_identity_column].astype(str).to_numpy()
    relion_identities = relion_table[relion_identity_column].astype(str).to_numpy()
    if len(set(recovar_identities.tolist())) != recovar_identities.size:
        raise AuditError("RECOVAR table contains duplicate image identities")
    if len(set(relion_identities.tolist())) != relion_identities.size:
        raise AuditError("RELION table contains duplicate image identities")
    relion_by_identity = {identity: idx for idx, identity in enumerate(relion_identities.tolist())}
    missing = [identity for identity in recovar_identities.tolist() if identity not in relion_by_identity]
    extras = sorted(set(relion_identities.tolist()).difference(recovar_identities.tolist()))
    if missing or extras:
        raise AuditError(f"particle identity sets differ: missing={missing[:3]}, extras={extras[:3]}")
    order = np.asarray([relion_by_identity[identity] for identity in recovar_identities], dtype=np.int64)
    aligned_relion = relion_table.iloc[order]

    recovar_eulers = _numeric_matrix(recovar_table, EULER_NAMES, label="RECOVAR table")
    relion_eulers = _numeric_matrix(aligned_relion, EULER_NAMES, label="RELION table")
    recovar_translations = _numeric_matrix(
        recovar_table, TRANSLATION_NAMES, label="RECOVAR table"
    )
    relion_translations = _numeric_matrix(
        aligned_relion, TRANSLATION_NAMES, label="RELION table"
    )
    recovar_pmax = _numeric_matrix(recovar_table, (PMAX_NAMES,), label="RECOVAR table")[:, 0]
    relion_pmax = _numeric_matrix(aligned_relion, (PMAX_NAMES,), label="RELION table")[:, 0]

    pose_error = _angular_error_deg(recovar_eulers, relion_eulers)
    translation_error = np.linalg.norm(recovar_translations - relion_translations, axis=1)
    pmax_error = np.abs(recovar_pmax - relion_pmax)
    pose_mismatch = pose_error > float(pose_tolerance_deg)
    translation_mismatch = translation_error > float(translation_tolerance_angst)
    divergent = pose_mismatch | translation_mismatch
    divergent_rows = np.flatnonzero(divergent)
    examples = []
    for row in divergent_rows[: int(example_limit)]:
        examples.append(
            {
                "recovar_row": int(row),
                "image_name": str(recovar_identities[row]),
                "pose_error_deg": float(pose_error[row]),
                "translation_error_angst": float(translation_error[row]),
                "pmax_absolute_error": float(pmax_error[row]),
                "recovar_pmax": float(recovar_pmax[row]),
                "relion_pmax": float(relion_pmax[row]),
            }
        )
    n_particles = int(recovar_identities.size)
    return {
        "iteration": int(iteration),
        "identity_alignment_exact": True,
        "particle_count": n_particles,
        "pose_match_fraction": float(np.mean(~pose_mismatch)),
        "translation_match_fraction": float(np.mean(~translation_mismatch)),
        "divergent_particle_count": int(divergent_rows.size),
        "pose_error_deg": _summary(pose_error),
        "translation_error_angst": _summary(translation_error),
        "pmax_absolute_error": _summary(pmax_error),
        "first_divergent_particles": examples,
    }


def audit_trajectory(
    recovar_dir: Path,
    relion_dir: Path,
    *,
    iterations: tuple[int, ...] = DEFAULT_ITERATIONS,
    pose_tolerance_deg: float = 1e-3,
    translation_tolerance_angst: float = 1e-4,
) -> dict[str, Any]:
    rows = []
    for iteration in iterations:
        recovar_path = recovar_dir / f"run_it{iteration:03d}_data.star"
        relion_path = relion_dir / f"run_it{iteration:03d}_data.star"
        if not recovar_path.is_file() or not relion_path.is_file():
            raise AuditError(
                f"iteration {iteration} requires both particle STAR files: "
                f"{recovar_path}, {relion_path}"
            )
        recovar_table, _ = read_star(str(recovar_path))
        relion_table, _ = read_star(str(relion_path))
        rows.append(
            compare_particle_tables(
                recovar_table,
                relion_table,
                iteration=iteration,
                pose_tolerance_deg=pose_tolerance_deg,
                translation_tolerance_angst=translation_tolerance_angst,
            )
        )
    first = next((row for row in rows if row["divergent_particle_count"]), None)
    return {
        "schema": SCHEMA,
        "metric_policy": {
            "correlation_used": False,
            "quality_gate": "none; diagnostic localization only",
            "pose_match_tolerance_deg": float(pose_tolerance_deg),
            "translation_match_tolerance_angst": float(translation_tolerance_angst),
        },
        "recovar_dir": str(recovar_dir.resolve()),
        "relion_dir": str(relion_dir.resolve()),
        "first_divergent_iteration": None if first is None else int(first["iteration"]),
        "iterations": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-dir", type=Path, required=True)
    parser.add_argument("--relion-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, nargs="+", default=list(DEFAULT_ITERATIONS))
    args = parser.parse_args()
    report = audit_trajectory(
        args.recovar_dir.resolve(),
        args.relion_dir.resolve(),
        iterations=tuple(args.iterations),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
