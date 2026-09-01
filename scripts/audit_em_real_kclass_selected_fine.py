#!/usr/bin/env python3
"""Audit selected real-data K-class fine E-step tables across engines.

The audit joins hypotheses only after their physical rotation matrices and
residual translation coordinates are qualified.  Native hypotheses without a
geometric RECOVAR counterpart remain native-only support; they are never
folded onto a nearest row.  The report combines that fine boundary with the
already-sealed coarse-support audit to identify the first observed mismatch.
"""

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

from scripts.compare_relion_recovar_estep_dump import compare_dumps  # noqa: E402

SCHEMA = "recovar.em.real_kclass_selected_fine.v1"
CLASS_FILE = re.compile(r"local_score_it(?P<iteration>\d+)_image_(?P<row>\d+)_mstep_class(?P<class>\d+)\.npz")
RELION_SHARED_FIELDS = (
    "pass1_acc_rot_id.bin",
    "pass1_acc_rot_idx.bin",
    "pass1_acc_trans_idx.bin",
    "pass1_candidate_class_idx.bin",
    "pass1_candidate_combined_log_prior.bin",
    "pass1_candidate_offset_log_prior.bin",
    "pass1_candidate_orientation_log_prior.bin",
    "pass1_candidate_translation_x.bin",
    "pass1_candidate_translation_y.bin",
    "pass1_candidate_weight_normalized.bin",
    "pass1_exp_Mweight_raw_preprior.bin",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    _require(path.is_file(), f"required artifact is missing: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _class_file_map(score_directory: Path, row: int, iteration: int) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for path in score_directory.glob(
        f"local_score_it{int(iteration):03d}_image_{int(row)}_mstep_class*.npz"
    ):
        match = CLASS_FILE.fullmatch(path.name)
        if match is None:
            continue
        class_index = int(match["class"])
        _require(class_index not in output, f"duplicate RECOVAR class file for row {row}, class {class_index}")
        output[class_index] = path
    return output


def summarize_particle(
    *,
    selection: dict[str, Any],
    coarse_record: dict[str, Any],
    class_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize one selected particle from already-qualified class records."""

    _require(class_records, "particle has no fine class records")
    class_records = sorted(class_records, key=lambda item: int(item["class_index_zero_based"]))
    relion_mass = np.asarray([item["relion_probability_mass"] for item in class_records], dtype=np.float64)
    recovar_mass = np.asarray([item["recovar_probability_mass"] for item in class_records], dtype=np.float64)
    relion_winner = int(class_records[int(np.argmax(relion_mass))]["class_index_zero_based"])
    recovar_winner = int(class_records[int(np.argmax(recovar_mass))]["class_index_zero_based"])
    coarse_support_mismatch = int(coarse_record["comparison"]["significance_support_mismatch_count"])
    fine_support_mismatch = int(
        sum(int(item["relion_only_count"]) + int(item["recovar_only_count"]) for item in class_records)
    )
    raw_max = max(
        (
            float(item["common_score_pre_prior_centered_diff"]["max_abs"])
            for item in class_records
            if item.get("common_score_pre_prior_centered_diff", {}).get("max_abs") is not None
        ),
        default=None,
    )
    translation_prior_max = max(
        (
            float(item["common_translation_log_prior_centered_diff"]["max_abs"])
            for item in class_records
            if item.get("common_translation_log_prior_centered_diff", {}).get("max_abs") is not None
        ),
        default=None,
    )
    rotation_prior_max = max(
        (
            float(item["common_rotation_log_prior_centered_diff"]["max_abs"])
            for item in class_records
            if item.get("common_rotation_log_prior_centered_diff", {}).get("max_abs") is not None
        ),
        default=None,
    )
    first_observed_nonidentical_boundary = (
        "coarse_significance_support" if coarse_support_mismatch else "fine_candidate_support"
    )
    return {
        "selection": selection,
        "coarse_classification": coarse_record["classification"],
        "coarse_joint_winner_exact": bool(coarse_record["comparison"]["joint_winner_exact"]),
        "coarse_significance_support_mismatch_count": coarse_support_mismatch,
        "fine_class_records": class_records,
        "fine_relion_class_probability_mass": relion_mass.tolist(),
        "fine_recovar_class_probability_mass": recovar_mass.tolist(),
        "fine_relion_class_winner_zero_based": relion_winner,
        "fine_recovar_class_winner_zero_based": recovar_winner,
        "fine_class_winner_exact": relion_winner == recovar_winner,
        "fine_support_symmetric_difference_count": fine_support_mismatch,
        "common_centered_raw_score_max_abs": raw_max,
        "common_centered_rotation_log_prior_max_abs": rotation_prior_max,
        "common_centered_translation_log_prior_max_abs": translation_prior_max,
        "first_observed_nonidentical_boundary": first_observed_nonidentical_boundary,
        "classification": (
            "fine_class_winner_exact_with_support_and_prior_differences"
            if relion_winner == recovar_winner
            else "fine_class_winner_mismatch_with_support_and_prior_differences"
        ),
        "strict_claim_boundary": (
            "Every reported common fine candidate passed the physical rotation-matrix and residual "
            "translation-coordinate tolerances. The report localizes observed support, prior, and "
            "posterior differences; it does not by itself prove which implementation generated them."
        ),
    }


def audit_selected_fine(
    *,
    selection_json: Path,
    coarse_report_json: Path,
    relion_root: Path,
    recovar_score_directory: Path,
    iteration: int,
) -> dict[str, Any]:
    selection_payload = json.loads(selection_json.read_text())
    coarse_payload = json.loads(coarse_report_json.read_text())
    selection_by_row = {
        int(item["recovar_source_row_zero_based"]): item for item in selection_payload["particles"]
    }
    coarse_by_row = {
        int(item["selection"]["recovar_source_row_zero_based"]): item for item in coarse_payload["particles"]
    }

    particles: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row, selection in sorted(selection_by_row.items()):
        class_files = _class_file_map(recovar_score_directory, row, iteration)
        if not class_files:
            skipped.append({"recovar_source_row_zero_based": row, "reason": "no_recovar_mstep_class_files"})
            continue
        _require(row in coarse_by_row, f"coarse audit has no selected row {row}")
        stack = int(selection["stack_index_one_based"])
        relion_directory = relion_root / f"stack{stack:06d}" / "capture"
        _require(relion_directory.is_dir(), f"RELION capture directory is missing: {relion_directory}")
        relion_shared_artifacts = {
            name.removesuffix(".bin"): _artifact(relion_directory / name)
            for name in RELION_SHARED_FIELDS
        }

        class_records: list[dict[str, Any]] = []
        for class_index, recovar_path in sorted(class_files.items()):
            relion_eulers = relion_directory / f"pass1_class{class_index}_fine_eulers.bin"
            if not relion_eulers.is_file():
                with np.load(recovar_path, allow_pickle=False) as archive:
                    posterior = np.asarray(archive["posterior"], dtype=np.float64)
                    finite = np.isfinite(np.asarray(archive["pass2_scores_total"], dtype=np.float64))
                class_records.append(
                    {
                        "class_index_zero_based": int(class_index),
                        "status": "no_native_fine_rotation_support",
                        "relion_candidate_count": 0,
                        "recovar_candidate_count": int(np.sum(finite)),
                        "common_candidate_count": 0,
                        "relion_only_count": 0,
                        "recovar_only_count": int(np.sum(finite)),
                        "relion_probability_mass": 0.0,
                        "recovar_probability_mass": float(np.sum(posterior, dtype=np.float64)),
                        "artifacts": {"recovar": _artifact(recovar_path)},
                    }
                )
                continue
            comparison = compare_dumps(
                relion_directory,
                recovar_path,
                recovar_class_index=class_index,
                match_mode="physical",
            )
            _require(
                bool(comparison["match_details"].get("physical_common_geometry_qualified")),
                f"physical fine geometry did not qualify for row {row}, class {class_index}",
            )
            comparison["class_index_zero_based"] = int(class_index)
            comparison["status"] = "physical_common_geometry_qualified"
            comparison["artifacts"] = {
                "recovar": _artifact(recovar_path),
                "relion_fine_eulers": _artifact(relion_eulers),
            }
            class_records.append(comparison)

        particle = summarize_particle(
            selection=selection,
            coarse_record=coarse_by_row[row],
            class_records=class_records,
        )
        particle["relion_shared_artifacts"] = relion_shared_artifacts
        particles.append(particle)

    _require(particles, "no selected particle had a RECOVAR fine capture")
    return {
        "schema": SCHEMA,
        "status": "complete",
        "iteration": int(iteration),
        "inputs": {
            "selection": _artifact(selection_json),
            "coarse_report": _artifact(coarse_report_json),
            "relion_root": str(relion_root.resolve()),
            "recovar_score_directory": str(recovar_score_directory.resolve()),
        },
        "particles": particles,
        "skipped_selection_rows": skipped,
        "classification_counts": {
            classification: sum(item["classification"] == classification for item in particles)
            for classification in sorted({item["classification"] for item in particles})
        },
        "strict_claim_boundary": (
            "This is a two-particle, iteration-1 InitialModel discriminator, not a production K-class "
            "quality claim. It uses only exact-coordinate intersections for arithmetic comparisons and "
            "retains all unmatched hypotheses as engine-specific support."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--coarse-report-json", type=Path, required=True)
    parser.add_argument("--relion-root", type=Path, required=True)
    parser.add_argument("--recovar-score-directory", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = audit_selected_fine(
        selection_json=args.selection_json,
        coarse_report_json=args.coarse_report_json,
        relion_root=args.relion_root,
        recovar_score_directory=args.recovar_score_directory,
        iteration=args.iteration,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
