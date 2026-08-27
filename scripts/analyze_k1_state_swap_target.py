#!/usr/bin/env python3
"""Classify one K=1 exact-state target capture against autonomous RELION parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_em_particle_state_distribution import (
    AuditError,
    _identity_array,
    _load_relion_state,
    _particle_table,
)
from scripts.audit_k1_live_particle_state import _load_half


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(archive, key: str, *, integer: bool = False) -> float | int:
    if key not in archive.files:
        raise AuditError(f"capture is missing {key}")
    values = np.asarray(archive[key]).reshape(-1)
    if values.size != 1 or not np.isfinite(values).all():
        raise AuditError(f"capture {key} must be one finite scalar")
    return int(values[0]) if integer else float(values[0])


def _autonomous_target(
    intermediates_dir: Path,
    *,
    recovar_iteration: int,
    source_row: int,
) -> dict[str, float | int | str]:
    matches = []
    for half in (1, 2):
        path = intermediates_dir / f"it{recovar_iteration:03d}_particle_state_half{half}.npz"
        state = _load_half(path, expected_iteration=recovar_iteration, expected_half=half)
        positions = np.flatnonzero(state["original_image_indices"] == source_row)
        if positions.size:
            if positions.size != 1:
                raise AuditError(f"source row {source_row} occurs more than once in {path}")
            position = int(positions[0])
            matches.append(
                {
                    "half": half,
                    "pmax": float(state["max_posterior"][position]),
                    "support": int(state["significant_counts"][position]),
                    "source": str(path.resolve()),
                }
            )
    if len(matches) != 1:
        raise AuditError(f"source row {source_row} occurs in {len(matches)} autonomous halves")
    return matches[0]


def analyze_state_swap_target(
    *,
    capture_path: Path,
    significance_capture_path: Path,
    autonomous_intermediates_dir: Path,
    recovar_particles_star: Path,
    relion_star: Path,
    recovar_iteration: int,
) -> dict[str, object]:
    capture_path = capture_path.resolve()
    significance_capture_path = significance_capture_path.resolve()
    autonomous_intermediates_dir = autonomous_intermediates_dir.resolve()
    recovar_particles_star = recovar_particles_star.resolve()
    relion_star = relion_star.resolve()
    if recovar_iteration < 0:
        raise AuditError("RECOVAR iteration must be non-negative")
    if not capture_path.is_file():
        raise AuditError(f"missing target capture: {capture_path}")
    if not significance_capture_path.is_file():
        raise AuditError(f"missing significance capture: {significance_capture_path}")

    with np.load(capture_path, allow_pickle=False) as capture:
        source_row = _scalar(capture, "original_index", integer=True)
        capture_iteration = _scalar(capture, "iteration", integer=True)
        physical_iteration = recovar_iteration + 1
        if capture_iteration != physical_iteration:
            raise AuditError(
                f"capture physical iteration {capture_iteration} does not match "
                f"RECOVAR loop iteration {recovar_iteration} "
                f"(physical iteration {physical_iteration})"
            )
        if "probs" not in capture.files:
            raise AuditError("capture is missing probs")
        probabilities = np.asarray(capture["probs"], dtype=np.float64)
        if probabilities.size == 0 or not np.isfinite(probabilities).all():
            raise AuditError("capture probabilities must be non-empty and finite")
        state_swap_pmax = float(np.max(probabilities))
        state_swap_reconstruction_support = _scalar(
            capture,
            "reconstruction_n_significant",
            integer=True,
        )
        state_swap = {
            "pmax": state_swap_pmax,
            "fine_reconstruction_support": state_swap_reconstruction_support,
            "probability_sum": float(np.sum(probabilities, dtype=np.float64)),
            "candidate_count": int(np.count_nonzero(capture["candidate_mask"])),
            "winner_flat_index": int(np.argmax(probabilities)),
        }

    with np.load(significance_capture_path, allow_pickle=False) as significance:
        significance_source_row = _scalar(significance, "original_index", integer=True)
        if significance_source_row != source_row:
            raise AuditError(
                f"significance source row {significance_source_row} does not match {source_row}"
            )
        significance_iteration = _scalar(significance, "one_based_iteration", integer=True)
        if significance_iteration != physical_iteration:
            raise AuditError(
                f"significance physical iteration {significance_iteration} does not match "
                f"{physical_iteration}"
            )
        state_swap_coarse_support = _scalar(significance, "n_significant", integer=True)
        state_swap["coarse_significant_support"] = state_swap_coarse_support

    source_table = _particle_table(recovar_particles_star)
    identities = _identity_array(source_table, source=recovar_particles_star)
    if source_row < 0 or source_row >= identities.size:
        raise AuditError(f"capture source row {source_row} is outside {identities.size} particles")
    relion_state, _ = _load_relion_state(relion_star, identities)
    relion = {
        "pmax": float(relion_state["pmax"][source_row]),
        "support": int(relion_state["support"][source_row]),
    }
    autonomous = _autonomous_target(
        autonomous_intermediates_dir,
        recovar_iteration=recovar_iteration,
        source_row=source_row,
    )

    autonomous_error = abs(float(autonomous["pmax"]) - relion["pmax"])
    state_swap_error = abs(state_swap_pmax - relion["pmax"])
    error_ratio = None if autonomous_error == 0.0 else state_swap_error / autonomous_error
    support_exact = state_swap_coarse_support == relion["support"]
    if error_ratio is None:
        classification = "autonomous_already_exact"
    elif error_ratio <= 0.1 and support_exact:
        classification = "inherited_state_or_reference"
    elif error_ratio >= 0.8:
        classification = "identical_input_scoring_or_support_residual"
    else:
        classification = "mixed_state_and_identical_input_residual"

    return {
        "schema": "recovar.em.k1_state_swap_target.v2",
        "status": "complete",
        "classification": classification,
        "classification_policy": {
            "inherited_state_or_reference": "Pmax error ratio <= 0.1 and exact support",
            "identical_input_scoring_or_support_residual": "Pmax error ratio >= 0.8",
            "mixed_state_and_identical_input_residual": "0.1 < Pmax error ratio < 0.8",
        },
        "recovar_iteration": recovar_iteration,
        "relion_iteration": physical_iteration,
        "source_row_zero_based": source_row,
        "identity": str(identities[source_row]),
        "autonomous": autonomous,
        "state_swap": state_swap,
        "relion": relion,
        "comparison": {
            "autonomous_abs_pmax_error": autonomous_error,
            "state_swap_abs_pmax_error": state_swap_error,
            "state_swap_to_autonomous_abs_pmax_error_ratio": error_ratio,
            "autonomous_support_delta": int(autonomous["support"]) - relion["support"],
            "state_swap_coarse_support_delta": state_swap_coarse_support - relion["support"],
        },
        "sources": {
            "capture": str(capture_path),
            "significance_capture": str(significance_capture_path),
            "autonomous_intermediates_dir": str(autonomous_intermediates_dir),
            "recovar_particles_star": str(recovar_particles_star),
            "relion_star": str(relion_star),
        },
        "input_sha256": {
            "capture": _sha256(capture_path),
            "significance_capture": _sha256(significance_capture_path),
            "recovar_particles_star": _sha256(recovar_particles_star),
            "relion_star": _sha256(relion_star),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--significance-capture", type=Path, required=True)
    parser.add_argument("--autonomous-intermediates-dir", type=Path, required=True)
    parser.add_argument("--recovar-particles-star", type=Path, required=True)
    parser.add_argument("--relion-star", type=Path, required=True)
    parser.add_argument("--recovar-iteration", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_state_swap_target(
        capture_path=args.capture,
        significance_capture_path=args.significance_capture,
        autonomous_intermediates_dir=args.autonomous_intermediates_dir,
        recovar_particles_star=args.recovar_particles_star,
        relion_star=args.relion_star,
        recovar_iteration=args.recovar_iteration,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
