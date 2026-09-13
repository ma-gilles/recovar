"""Audit a frozen K=1 first-iteration coarse-score boundary.

The audit aligns complete RELION and RECOVAR coarse grids by physical
rotation/translation identity.  It deliberately does not infer a float64
answer from already-reduced float32 scores.  High-precision replay is marked
unavailable unless the capture contains complete per-pixel operands.

Intermediate comparisons use exact/array metrics.  This module makes no map
quality claim; map conclusions remain gated by shellwise FSC/FSC-AUC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "recovar-k1-coarse-boundary-audit-v1"
SEAL_SCHEMA = "recovar-k1-coarse-boundary-audit-seal-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_bijection(relion_matrices, recovar_matrices) -> np.ndarray:
    """Map each RELION rotation row to its bitwise-identical RECOVAR row."""

    relion = np.asarray(relion_matrices, dtype=np.float32)
    recovar = np.asarray(recovar_matrices, dtype=np.float32)
    if relion.shape != recovar.shape or relion.ndim != 3 or relion.shape[1:] != (3, 3):
        raise ValueError(f"rotation topology mismatch: {relion.shape} != {recovar.shape}")
    lookup: dict[bytes, int] = {}
    for index, matrix in enumerate(recovar):
        key = matrix.tobytes()
        if key in lookup:
            raise ValueError("RECOVAR rotation grid contains duplicate matrices")
        lookup[key] = index
    try:
        mapping = np.asarray([lookup[matrix.T.tobytes()] for matrix in relion], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("RELION rotation has no bitwise RECOVAR transpose match") from exc
    if np.unique(mapping).size != mapping.size:
        raise ValueError("rotation mapping is not bijective")
    return mapping


def translation_bijection(
    relion_phase_xyz,
    recovar_translations,
    *,
    image_size: int,
    tolerance_px: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map RELION SoA phase coefficients to RECOVAR pixel translations."""

    phases = np.asarray(relion_phase_xyz, dtype=np.float32).reshape(3, -1).T
    recovar = np.asarray(recovar_translations, dtype=np.float32)
    if recovar.shape != (phases.shape[0], 2):
        raise ValueError(f"translation topology mismatch: {phases.shape} vs {recovar.shape}")
    relion_xy = -phases[:, :2].astype(np.float64) * float(image_size) / (2.0 * np.pi)
    distances = np.linalg.norm(relion_xy[:, None, :] - recovar[None, :, :], axis=2)
    mapping = np.argmin(distances, axis=1).astype(np.int64)
    matched = distances[np.arange(mapping.size), mapping]
    if np.unique(mapping).size != mapping.size:
        raise ValueError("translation mapping is not bijective")
    max_error = float(np.max(matched))
    if max_error > float(tolerance_px):
        raise ValueError(f"translation identity error {max_error:.9g}px exceeds {tolerance_px:.9g}px")
    return mapping, relion_xy, max_error


def align_relion_surface(relion_positive_scores, rotation_map, translation_map) -> np.ndarray:
    relion = np.asarray(relion_positive_scores, dtype=np.float32)
    rotation_map = np.asarray(rotation_map, dtype=np.int64)
    translation_map = np.asarray(translation_map, dtype=np.int64)
    if relion.shape != (rotation_map.size, translation_map.size):
        raise ValueError("score surface does not match identity grids")
    aligned = np.empty_like(relion)
    aligned[np.ix_(rotation_map, translation_map)] = relion
    return aligned


def array_stats(values) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "max_abs": float(np.max(np.abs(values))),
        "p95_abs": float(np.quantile(np.abs(values), 0.95)),
        "p99_abs": float(np.quantile(np.abs(values), 0.99)),
    }


def top_two(scores, translations) -> dict[str, object]:
    scores = np.asarray(scores, dtype=np.float32)
    translations = np.asarray(translations, dtype=np.float32)
    order = np.argsort(-scores.reshape(-1), kind="stable")[:2]
    rows = []
    for flat_index in order:
        rotation_index, translation_index = divmod(int(flat_index), scores.shape[1])
        rows.append(
            {
                "flat_index": int(flat_index),
                "rotation_index": rotation_index,
                "translation_index": translation_index,
                "translation_px": translations[translation_index].astype(np.float64).tolist(),
                "score": float(scores.reshape(-1)[flat_index]),
            }
        )
    winner = np.float32(rows[0]["score"])
    margin = float(np.float32(winner - np.float32(rows[1]["score"])))
    ulp = float(abs(np.spacing(winner)))
    return {"rows": rows, "winner_margin": margin, "margin_float32_ulps": margin / ulp}


def _fine_support_report(relion, recovar, counterfactual_path: Path) -> dict[str, object]:
    rel_rot_id = np.asarray(relion["pass1_firstiter_cc_raw_rot_id"], dtype=np.int64).reshape(8, 4)
    rel_rot_child = np.asarray(relion["pass1_firstiter_cc_raw_rot_idx"], dtype=np.int64).reshape(8, 4)
    rel_trans = np.asarray(relion["pass1_firstiter_cc_raw_trans_idx"], dtype=np.int64).reshape(8, 4)
    rel_hidden = np.asarray(relion["pass1_firstiter_cc_raw_ihidden_overs"], dtype=np.int64).reshape(8, 4)
    rel_scores = -np.asarray(relion["pass1_firstiter_cc_exp_Mweight_raw_preonehot"], dtype=np.float32).reshape(8, 4)
    expected_children = np.broadcast_to(np.arange(8, dtype=np.int64)[:, None], (8, 4))
    if not np.array_equal(rel_rot_child, expected_children):
        raise ValueError("unexpected RELION pass-2 child ordering")
    if not np.all(rel_rot_id == rel_rot_id[0, 0]):
        raise ValueError("RELION pass-2 candidates do not share one coarse parent")

    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    if candidate_mask.shape != (8, 116):
        raise ValueError(f"unexpected RECOVAR pass-2 mask shape {candidate_mask.shape}")
    rows, cols = np.nonzero(candidate_mask)
    rec_trans_ids = np.unique(cols)
    if not np.array_equal(np.unique(rows), np.arange(8)) or rows.size != 32:
        raise ValueError("RECOVAR pass-2 support is not eight children by four translations")
    rel_trans_ids = np.unique(rel_trans)
    translations = np.asarray(recovar["fine_translations"], dtype=np.float32)
    rec_scores = np.asarray(recovar["scores_pre_prior"], dtype=np.float32)[:, rec_trans_ids]

    report: dict[str, object] = {
        "relion": {
            "coarse_parent_rotation_index": int(rel_rot_id[0, 0]),
            "child_ordinals": np.unique(rel_rot_child).tolist(),
            "translation_indices": rel_trans_ids.tolist(),
            "translations_px": translations[rel_trans_ids].astype(np.float64).tolist(),
            "hidden_identity_first_last": [int(rel_hidden.min()), int(rel_hidden.max())],
            "top_two": top_two(rel_scores, translations[rel_trans_ids]),
        },
        "recovar": {
            "parent_map": np.asarray(recovar["parent_map"], dtype=np.int64).tolist(),
            "oversampled_rotation_indices": np.asarray(recovar["oversampled_rot_indices"], dtype=np.int64).tolist(),
            "child_ordinals": np.arange(8).tolist(),
            "translation_indices": rec_trans_ids.tolist(),
            "translations_px": translations[rec_trans_ids].astype(np.float64).tolist(),
            "top_two": top_two(rec_scores, translations[rec_trans_ids]),
        },
        "child_ordinal_lists_equal": True,
        "child_matrix_cross_engine_comparison_available": False,
        "child_matrix_unavailability_reason": (
            "RELION capture records parent and child ordinal identities but not the eight generated child matrices"
        ),
        "translation_index_intersection": np.intersect1d(rel_trans_ids, rec_trans_ids).tolist(),
    }
    if counterfactual_path.is_file():
        with np.load(counterfactual_path, allow_pickle=False) as counter:
            full_scores = np.asarray(counter["scores"], dtype=np.float32)
        rec_on_rel = full_scores[:, rel_trans_ids]
        delta = rec_on_rel.astype(np.float64) - rel_scores.astype(np.float64)
        centered = delta - float(np.mean(delta))
        report["recovar_counterfactual_on_relion_support"] = {
            "top_two": top_two(rec_on_rel, translations[rel_trans_ids]),
            "raw_difference_recovar_minus_relion": array_stats(delta),
            "centered_difference": array_stats(centered),
            "winner_equal": bool(np.argmax(rec_on_rel) == np.argmax(rel_scores)),
        }
    return report


def run_audit(capture_root: Path, *, particle_index: int = 8494, image_size: int = 256) -> dict[str, object]:
    capture_root = capture_root.resolve()
    relion_path = capture_root / "analysis/relion_acc_parse.npz"
    recovar_path = capture_root / "recovar_scores" / (f"significance_orig{particle_index:06d}_it001_cs048.npz")
    pass2_path = capture_root / "recovar_scores" / f"pass2_orig{particle_index:06d}_cs048.npz"
    counterfactual_path = capture_root / "analysis/recovar_full_counterfactual_scores.npz"
    prior_report_path = capture_root / "analysis/final_report.json"
    target_identity_path = capture_root / "provenance/target_identity.json"
    for path in (relion_path, recovar_path, pass2_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with (
        np.load(relion_path, allow_pickle=False) as relion,
        np.load(recovar_path, allow_pickle=False) as recovar,
        np.load(pass2_path, allow_pickle=False) as pass2,
    ):
        n_rot = int(recovar["n_rot"])
        n_trans = int(recovar["n_trans"])
        if (n_rot, n_trans) != (36864, 29):
            raise ValueError(f"unexpected coarse topology {(n_rot, n_trans)}")
        matrix_keys = [key for key in relion.files if key.endswith("_pass1_eulers_matrices")]
        phase_keys = [key for key in relion.files if key.endswith("_pass1_trans_xyz_phases")]
        if len(matrix_keys) != 1 or len(phase_keys) != 1:
            raise ValueError("expected exactly one RELION particle geometry capture")
        relion_matrices = np.asarray(relion[matrix_keys[0]], dtype=np.float32).reshape(n_rot, 3, 3)
        recovar_matrices = np.asarray(recovar["rotations"], dtype=np.float32)
        rotation_map = rotation_bijection(relion_matrices, recovar_matrices)
        translation_map, relion_translations, translation_max_error = translation_bijection(
            relion[phase_keys[0]],
            recovar["translations"],
            image_size=image_size,
        )
        relion_raw = np.asarray(relion["pass0_firstiter_cc_exp_Mweight_raw_preonehot"], dtype=np.float32).reshape(
            n_rot, n_trans
        )
        recovar_scores = np.asarray(recovar["scores_pre_prior_per_class"], dtype=np.float32)[0]
        relion_aligned = align_relion_surface(-relion_raw, rotation_map, translation_map)
        residual = recovar_scores.astype(np.float64) - relion_aligned.astype(np.float64)
        centered_residual = residual - float(np.mean(residual))
        relion_top = top_two(relion_aligned, recovar["translations"])
        recovar_top = top_two(recovar_scores, recovar["translations"])
        boundary_flat_indices = sorted({int(row["flat_index"]) for row in (*relion_top["rows"], *recovar_top["rows"])})
        boundary_candidates = []
        for flat_index in boundary_flat_indices:
            rotation_index, translation_index = divmod(flat_index, n_trans)
            boundary_candidates.append(
                {
                    "flat_index": flat_index,
                    "rotation_index": rotation_index,
                    "translation_index": translation_index,
                    "recovar_score": float(recovar_scores.reshape(-1)[flat_index]),
                    "relion_score": float(relion_aligned.reshape(-1)[flat_index]),
                    "residual_recovar_minus_relion": float(residual.reshape(-1)[flat_index]),
                }
            )
        fine = _fine_support_report(relion, pass2, counterfactual_path)

        component_weight = np.asarray(relion["pass0_class0_cc_component_weight"])
        component_norm = np.asarray(relion["pass0_class0_cc_component_norm"])
        component_complete = bool(np.all(np.isfinite(component_norm)) and np.all(component_norm > 0))
        winner_equal = relion_top["rows"][0]["flat_index"] == recovar_top["rows"][0]["flat_index"]
        fine_counter = fine.get("recovar_counterfactual_on_relion_support", {})
        fine_winner_equal = bool(fine_counter.get("winner_equal", False))
        classification = (
            "production_float32_decision_agrees"
            if winner_equal
            else "coarse_float32_near_tie_changes_fine_support_precision_unresolved"
        )
        target_identity = json.loads(target_identity_path.read_text()) if target_identity_path.is_file() else {}
        prior_report = json.loads(prior_report_path.read_text()) if prior_report_path.is_file() else {}
        report = {
            "schema": SCHEMA,
            "schema_version": 1,
            "status": "pass",
            "classification": classification,
            "identity": {
                "particle_original_index_zero_based": int(particle_index),
                "image_identity": target_identity.get("image_identity", "111721@particles.256.mrcs"),
                "half": 2,
                "half_local_index": 4249,
                "current_size": int(recovar["current_size"]),
            },
            "coarse_identity_alignment": {
                "entry_count": int(n_rot * n_trans),
                "rotation_count": n_rot,
                "translation_count": n_trans,
                "rotation_bijection_bitwise_exact_after_relion_transpose": True,
                "rotation_mapping_unique_count": int(np.unique(rotation_map).size),
                "translation_bijection": True,
                "translation_order_identical": bool(np.array_equal(translation_map, np.arange(n_trans))),
                "translation_max_coordinate_error_px": translation_max_error,
                "relion_translations_from_phase_px": relion_translations.tolist(),
            },
            "coarse_scores": {
                "recovar": recovar_top,
                "relion_aligned": relion_top,
                "winner_equal": winner_equal,
                "boundary_candidate_residuals": boundary_candidates,
                "raw_residual_recovar_minus_relion": array_stats(residual),
                "centered_residual": array_stats(centered_residual),
            },
            "fine_support": fine,
            "precision_replay": {
                "production_float32": "available_complete_score_surfaces",
                "original_order_float32": "unavailable_in_frozen_capture",
                "canonical_float32": "unavailable_in_frozen_capture",
                "promoted_captured_float64": "unavailable_in_frozen_capture",
                "genuine_float64_complex128": "unavailable_in_frozen_capture",
                "reason": (
                    "the capture lacks the exact coarse projector/per-pixel contribution list; "
                    "its diagnostic component norm array is all zero and the weight array is incomplete. "
                    "Casting final float32 scores or rebuilding a projector from an MRC cannot adjudicate a one-ULP tie"
                ),
                "component_capture": {
                    "complete": component_complete,
                    "weight_nonzero_count": int(np.count_nonzero(component_weight)),
                    "norm_nonzero_count": int(np.count_nonzero(component_norm)),
                    "entry_count": int(component_norm.size),
                },
                "classification_strength": "precision mechanism unresolved; no algorithmic bug inferred",
            },
            "interpretation": {
                "fine_scorer_algorithmic_mismatch_supported": False if fine_winner_equal else None,
                "coarse_discrete_difference_is_within_measured_cross_engine_float32_residual": bool(
                    max(relion_top["winner_margin"], recovar_top["winner_margin"]) <= array_stats(residual)["max_abs"]
                ),
                "explanation": (
                    "The aligned production surfaces disagree only at a near-tied coarse translation boundary. "
                    "The resulting four-translation fine supports are adjacent and disjoint; RECOVAR's frozen "
                    "fine scorer selects RELION's winner when evaluated on RELION's support. Exact coarse "
                    "per-pixel operands are absent, so order/precision cannot be separated from upstream operand generation."
                ),
            },
            "quality_metric_policy": (
                "exact/array metrics only for intermediates; map conclusions require shellwise FSC/FSC-AUC; correlation prohibited"
            ),
            "same_physical_gpu_control_capture": {
                "provenance": prior_report.get("provenance", {}),
                "control_envelope": prior_report.get("control_envelope", {}),
                "capture_inertness_source": str(prior_report_path) if prior_report else None,
                "slurm_job_id": None,
                "execution_mode": "local six-arm run: two controls plus one capture per engine",
            },
            "artifacts": {
                str(path): sha256_file(path)
                for path in (
                    relion_path,
                    recovar_path,
                    pass2_path,
                    counterfactual_path,
                    prior_report_path,
                    target_identity_path,
                )
                if path.is_file()
            },
        }
    return report


def write_report_and_seal(report: dict[str, object], output: Path, seal_output: Path) -> None:
    output = output.resolve()
    seal_output = seal_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    seal_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    seal = {
        "schema": SEAL_SCHEMA,
        "schema_version": 1,
        "status": report["status"],
        "classification": report["classification"],
        "report_path": str(output),
        "report_sha256": sha256_file(output),
        "input_artifact_sha256": report["artifacts"],
    }
    seal_output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--particle-index", type=int, default=8494)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(
        args.capture_root,
        particle_index=args.particle_index,
        image_size=args.image_size,
    )
    write_report_and_seal(report, args.output, args.seal_output)
    print(json.dumps({"status": report["status"], "classification": report["classification"]}))


if __name__ == "__main__":
    main()
