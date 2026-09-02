#!/usr/bin/env python3
"""Identify the first material real-data K=4 coarse likelihood operand."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import _relion_parent_to_recovar
from scripts.analyze_em_k1_live_reference_counterfactual import (
    recovar_score_components,
    relion_reference_on_recovar_window,
    relion_values_on_recovar_window,
)
from scripts.analyze_em_real_k4_coarse_score_support import (
    _stable_top_count_mask,
    _support_metric,
)
from scripts.analyze_em_real_k4_native_coarse_components import _aggregate_support
from scripts.analyze_em_real_k4_native_coarse_scores import (
    SUPPORT_SCHEMA,
    _centered_comparison,
    _load_recovar_dump,
    _native_surface,
    _quantiles,
    _sha256,
)
from scripts.audit_relion_k4_coarse_operand_repeatability import (
    SCHEMA as REPEATABILITY_SCHEMA,
)
from scripts.validate_relion_coarse_score_capture import load_coarse_score_capture
from scripts.validate_relion_k4_coarse_operand_capture import (
    K4CoarseOperandCapture,
    load_artifact,
)

SCHEMA = "recovar.em_real_k4_native_coarse_operands.v1"
OPERAND_VALIDATION_SCHEMA = "relion-k4-coarse-operand-validation-v1"
EULER_MAX_ABS_GATE = 1.0e-5
TRANSLATION_PHASE_MAX_ABS_GATE = 1.0e-6
REFERENCE_ENERGY_REMAINING_CEILING = 1.0e-6
NONREFERENCE_ENERGY_REMAINING_FLOOR = 0.99


class AnalysisError(RuntimeError):
    """Raised when a frozen operand panel cannot be joined exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _load_gate(path: Path, schema: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing gate: {path}")
    report = json.loads(path.read_text())
    _require(report.get("schema") == schema, f"gate schema differs: {path}")
    _require(report.get("status") == "pass", f"gate did not pass: {path}")
    return report


def _native_rotation_order(
    artifact: K4CoarseOperandCapture,
    *,
    n_directions: int,
    n_psi: int,
) -> np.ndarray:
    """Return native reference rows in RECOVAR psi-major rotation order."""

    n_rotations = n_directions * n_psi
    class_offset = (artifact.class_one_based - 1) * n_rotations
    local_keys = artifact.rotation_keys.astype(np.int64) - class_offset
    _require(
        np.all((0 <= local_keys) & (local_keys < n_rotations)),
        "native rotation key belongs to another class",
    )
    mapped = np.asarray(
        [
            _relion_parent_to_recovar(
                int(key), n_directions=n_directions, n_psi=n_psi
            )
            for key in local_keys
        ],
        dtype=np.int64,
    )
    _require(
        np.array_equal(np.sort(mapped), np.arange(n_rotations)),
        "native rotation mapping is not a permutation",
    )
    return np.argsort(mapped)


def _load_recovar_operands(path: Path) -> dict[str, Any]:
    record = _load_recovar_dump(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "rotations",
            "translations",
            "translation_phase_source",
            "window_indices",
            "half_weights",
            "shifted_data",
            "ctf2_data",
            "projected_reference_rotation_ids",
            "projected_reference_per_class",
            "projected_reference_norm_score_per_class",
            "projected_cross_score_per_class",
        }
        _require(
            required <= set(payload.files),
            f"RECOVAR dump lacks operand fields: {path}",
        )
        record.update(
            rotations=np.asarray(payload["rotations"], dtype=np.float32),
            translations=np.asarray(payload["translations"], dtype=np.float32),
            translation_phase_source=np.asarray(
                payload["translation_phase_source"], dtype=np.float32
            ),
            window_indices=np.asarray(payload["window_indices"], dtype=np.int64),
            half_weights=np.asarray(payload["half_weights"], dtype=np.float64),
            shifted_data=np.asarray(payload["shifted_data"], dtype=np.complex128),
            ctf2_data=np.asarray(payload["ctf2_data"][0], dtype=np.float64),
            references=np.asarray(
                payload["projected_reference_per_class"], dtype=np.complex128
            ),
            reference_norm=np.asarray(
                payload["projected_reference_norm_score_per_class"],
                dtype=np.float64,
            ),
            cross_term=np.asarray(
                payload["projected_cross_score_per_class"], dtype=np.float64
            ),
            projection_rotation_ids=np.asarray(
                payload["projected_reference_rotation_ids"], dtype=np.int64
            ),
        )
    shape = (record["n_classes"], record["n_rotations"], record["n_translations"])
    n_pixels = record["window_indices"].size
    _require(record["references"].shape == shape[:2] + (n_pixels,), "reference topology differs")
    _require(record["shifted_data"].shape == (shape[2], n_pixels), "shifted-image topology differs")
    _require(record["ctf2_data"].shape == (n_pixels,), "correction topology differs")
    _require(record["half_weights"].shape == (n_pixels,), "half-weight topology differs")
    _require(np.all(record["half_weights"] > 0), "half weights must be positive")
    _require(
        np.array_equal(record["projection_rotation_ids"], np.arange(shape[1])),
        "RECOVAR projection panel is incomplete",
    )
    _require(
        record["reference_norm"].shape == record["cross_term"].shape == shape,
        "RECOVAR component topology differs",
    )
    return record


def _array_comparison(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    left = np.asarray(candidate).reshape(-1).astype(np.complex128)
    right = np.asarray(reference).reshape(-1).astype(np.complex128)
    _require(left.shape == right.shape and left.size > 0, "operand shapes differ")
    _require(np.isfinite(left).all() and np.isfinite(right).all(), "operand is non-finite")
    denominator = float(np.linalg.norm(right))
    _require(denominator > 0, "reference operand has zero norm")
    residual = left - right
    scalar_denominator = float(np.vdot(right, right).real)
    scalar = np.vdot(right, left) / scalar_denominator
    fitted = left - scalar * right
    return {
        "value_count": int(left.size),
        "relative_l2": float(np.linalg.norm(residual) / denominator),
        "p95_abs": float(np.percentile(np.abs(residual), 95)),
        "max_abs": float(np.max(np.abs(residual))),
        "least_squares_scalar": {
            "real": float(scalar.real),
            "imag": float(scalar.imag),
        },
        "after_scalar_fit_relative_l2": float(
            np.linalg.norm(fitted) / denominator
        ),
    }


def _class_components(
    references: np.ndarray,
    shifted_data: np.ndarray,
    correction: np.ndarray,
    half_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    norms = []
    crosses = []
    for reference in np.asarray(references):
        norm, cross = recovar_score_components(
            reference, shifted_data, correction, half_weights
        )
        norms.append(norm)
        crosses.append(cross)
    return np.stack(norms), np.stack(crosses)


def _centered_squared_energy(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    centered = array - float(np.mean(array))
    return float(np.sum(centered * centered))


def _support_factorial(
    *,
    native: dict[str, np.ndarray],
    recovar: dict[str, np.ndarray],
    raw_surfaces: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, float]]:
    reference = native["significant"]
    count = int(np.count_nonzero(reference))
    combined = {
        "native_captured_combined": native["combined_score"],
        "recovar_captured_combined": recovar["combined_score"],
        **{
            name: raw + native["total_prior"]
            for name, raw in raw_surfaces.items()
        },
    }
    support: dict[str, Any] = {}
    for name, scores in combined.items():
        mask, cutoff = _stable_top_count_mask(scores, count)
        support[name] = {**_support_metric(reference, mask), **cutoff}

    baseline_residual = raw_surfaces["recovar_operands"] - native["raw_score"]
    baseline_energy = _centered_squared_energy(baseline_residual)
    _require(baseline_energy > 0, "baseline score residual has zero centered energy")
    energies = {"baseline_centered_squared_energy": baseline_energy}
    for name, raw in raw_surfaces.items():
        energy = _centered_squared_energy(raw - native["raw_score"])
        energies[f"{name}_centered_squared_energy"] = energy
        energies[f"{name}_fraction_baseline_energy_remaining"] = (
            energy / baseline_energy
        )
    return support, energies


def _aggregate_energy(records: list[dict[str, Any]]) -> dict[str, float]:
    fields = tuple(
        key
        for key in records[0]["score_residual_energy"]
        if key.endswith("_centered_squared_energy")
        and key != "baseline_centered_squared_energy"
    )
    baseline = sum(
        float(record["score_residual_energy"]["baseline_centered_squared_energy"])
        for record in records
    )
    return {
        "baseline_centered_squared_energy": baseline,
        **{
            field.replace("_centered_squared_energy", "_fraction_baseline_energy_remaining"): (
                sum(float(record["score_residual_energy"][field]) for record in records)
                / baseline
            )
            for field in fields
        },
    }


def classify(summary: dict[str, Any], record_count: int) -> str:
    """Classify the fixed factorial using support and residual-energy gates."""

    support = summary["support_counterfactuals"]
    energy = summary["score_residual_energy"]
    baseline = support["recovar_operands"]
    if (
        baseline["exact_records"] == record_count
        and baseline["jaccard"] == 1.0
        and support["recovar_captured_combined"]["exact_records"] == record_count
        and support["recovar_captured_combined"]["jaccard"] == 1.0
        and summary["operand_relative_l2"]["projected_reference"]["maximum"] == 0.0
        and summary["mapping"]["euler_transpose_max_abs"] == 0.0
    ):
        return "k4_coarse_projected_reference_and_significant_support_match_native"
    reference = support["native_projected_reference"]
    shifted = support["native_shifted_image"]
    correction = support["native_correction"]
    all_native = support["all_native_operands"]
    if (
        reference["exact_records"] == all_native["exact_records"] == record_count
        and reference["jaccard"] == all_native["jaccard"] == 1.0
        and baseline["exact_records"] < record_count
        and shifted["exact_records"] == correction["exact_records"] == baseline["exact_records"]
        and math.isclose(shifted["jaccard"], baseline["jaccard"], abs_tol=1e-15)
        and math.isclose(correction["jaccard"], baseline["jaccard"], abs_tol=1e-15)
        and energy["native_projected_reference_fraction_baseline_energy_remaining"]
        <= REFERENCE_ENERGY_REMAINING_CEILING
        and energy["all_native_operands_fraction_baseline_energy_remaining"]
        <= REFERENCE_ENERGY_REMAINING_CEILING
        and energy["native_shifted_image_fraction_baseline_energy_remaining"]
        >= NONREFERENCE_ENERGY_REMAINING_FLOOR
        and energy["native_correction_fraction_baseline_energy_remaining"]
        >= NONREFERENCE_ENERGY_REMAINING_FLOOR
    ):
        return "projected_reference_is_first_material_k4_coarse_likelihood_operand_difference"
    return "k4_coarse_likelihood_operand_difference_is_mixed_or_unresolved"


def build_report(
    *,
    capture_dir: Path,
    significance_dir: Path,
    support_report_path: Path,
    operand_validation_path: Path,
    repeatability_path: Path,
    full_image_size: int,
) -> dict[str, Any]:
    _require(full_image_size > 0 and full_image_size % 2 == 0, "full image size is invalid")
    capture_dir = capture_dir.resolve()
    operand_validation = _load_gate(
        operand_validation_path, OPERAND_VALIDATION_SCHEMA
    )
    repeatability = _load_gate(repeatability_path, REPEATABILITY_SCHEMA)
    _require(
        Path(operand_validation["directory"]).resolve() == capture_dir,
        "operand validation names another capture directory",
    )
    admitted_roots = {
        Path(item["run_root"]).resolve() for item in repeatability["repeats"]
    }
    _require(capture_dir.parent in admitted_roots, "selected capture is not repeatability-admitted")

    support_report = json.loads(support_report_path.read_text())
    _require(support_report.get("schema") == SUPPORT_SCHEMA, "support report schema differs")
    _require(support_report.get("status") == "complete", "support report is incomplete")
    expected_particles = {
        int(record["native_particle_id_zero_based"]): record
        for record in support_report["particles"]
    }
    dataset_by_particle = {
        int(key): int(value)
        for key, value in support_report["inputs"][
            "dataset_index_by_native_particle_id"
        ].items()
    }
    _require(set(expected_particles) == set(dataset_by_particle), "support join differs")
    _require(len(expected_particles) == 16, "fixed particle denominator differs")

    operand_paths = sorted(capture_dir.glob("*.coarse-operands-v1.bin"))
    operands = [load_artifact(path) for path in operand_paths]
    operand_by_key = {
        (item.part_id, item.class_one_based): item for item in operands
    }
    _require(len(operand_by_key) == 64, "native operand panel differs")
    score_paths = sorted(capture_dir.glob("*.coarse-score-v1.bin"))
    scores = [load_coarse_score_capture(path) for path in score_paths]
    score_by_part = {item.header[5]: item for item in scores}
    _require(set(score_by_part) == set(expected_particles), "native score panel differs")

    recovar_paths = sorted(significance_dir.glob("significance_*.npz"))
    recovar_records = [_load_recovar_operands(path) for path in recovar_paths]
    recovar_by_dataset = {item["dataset_index"]: item for item in recovar_records}
    _require(
        set(recovar_by_dataset) == set(dataset_by_particle.values()),
        "RECOVAR operand panel differs",
    )

    records: list[dict[str, Any]] = []
    for part_id in sorted(expected_particles):
        recovar = recovar_by_dataset[dataset_by_particle[part_id]]
        score_capture = score_by_part[part_id]
        native = _native_surface(score_capture)
        n_directions, n_psi = score_capture.header[11:13]
        _require(
            n_directions * n_psi == recovar["n_rotations"],
            "native/RECOVAR rotation topology differs",
        )
        particle_operands = [operand_by_key[(part_id, cls)] for cls in range(1, 5)]
        first = particle_operands[0]
        for artifact in particle_operands[1:]:
            for label, values, reference in (
                ("image real", artifact.image_real, first.image_real),
                ("image imaginary", artifact.image_imag, first.image_imag),
                ("correction", artifact.correction, first.correction),
                ("translations", artifact.translations, first.translations),
                ("shifted real", artifact.shifted_real, first.shifted_real),
                ("shifted imaginary", artifact.shifted_imag, first.shifted_imag),
            ):
                _require(np.array_equal(values, reference), f"class-local {label} differs")

        native_references = []
        euler_max_abs = 0.0
        for artifact in particle_operands:
            order = _native_rotation_order(
                artifact, n_directions=n_directions, n_psi=n_psi
            )
            native_references.append(
                relion_reference_on_recovar_window(
                    artifact.reference_real[order].astype(np.float64)
                    + 1j * artifact.reference_imag[order].astype(np.float64),
                    recovar["window_indices"],
                    full_image_size=full_image_size,
                    current_size=recovar["current_size"],
                )
            )
            eulers = artifact.euler_matrices[order].transpose(0, 2, 1)
            euler_max_abs = max(
                euler_max_abs,
                float(np.max(np.abs(eulers - recovar["rotations"]))),
            )
        _require(euler_max_abs <= EULER_MAX_ABS_GATE, "Euler mapping exceeds gate")
        native_references_array = np.stack(native_references)

        native_shifted_raw = relion_values_on_recovar_window(
            first.shifted_real.astype(np.float64)
            + 1j * first.shifted_imag.astype(np.float64),
            recovar["window_indices"],
            full_image_size=full_image_size,
            current_size=recovar["current_size"],
        )
        native_correction_raw = relion_values_on_recovar_window(
            first.correction[np.newaxis, :],
            recovar["window_indices"],
            full_image_size=full_image_size,
            current_size=recovar["current_size"],
        )[0].real
        normalization = float(full_image_size**2)
        native_shifted = (
            -native_shifted_raw
            * native_correction_raw[np.newaxis, :]
            / (normalization * recovar["half_weights"][np.newaxis, :])
        )
        native_correction = native_correction_raw / (
            normalization**2 * recovar["half_weights"]
        )
        expected_phase = (
            -2.0 * np.pi * recovar["translation_phase_source"] / full_image_size
        )
        translation_phase_max_abs = float(
            np.max(np.abs(first.translations[:2].T - expected_phase))
        )
        _require(
            translation_phase_max_abs <= TRANSLATION_PHASE_MAX_ABS_GATE,
            "translation-phase mapping exceeds gate",
        )

        recovar_norm, recovar_cross = _class_components(
            recovar["references"],
            recovar["shifted_data"],
            recovar["ctf2_data"],
            recovar["half_weights"],
        )
        reference_norm, reference_cross = _class_components(
            native_references_array,
            recovar["shifted_data"],
            recovar["ctf2_data"],
            recovar["half_weights"],
        )
        shifted_norm, shifted_cross = _class_components(
            recovar["references"],
            native_shifted,
            recovar["ctf2_data"],
            recovar["half_weights"],
        )
        correction_norm, correction_cross = _class_components(
            recovar["references"],
            recovar["shifted_data"],
            native_correction,
            recovar["half_weights"],
        )
        all_native_norm, all_native_cross = _class_components(
            native_references_array,
            native_shifted,
            native_correction,
            recovar["half_weights"],
        )
        raw_surfaces = {
            "recovar_operands": recovar_norm + recovar_cross,
            "native_projected_reference": reference_norm + reference_cross,
            "native_shifted_image": shifted_norm + shifted_cross,
            "native_correction": correction_norm + correction_cross,
            "all_native_operands": all_native_norm + all_native_cross,
        }
        support_factorial, score_energy = _support_factorial(
            native=native, recovar=recovar, raw_surfaces=raw_surfaces
        )
        records.append(
            {
                "native_particle_id_zero_based": part_id,
                "dataset_index_zero_based": recovar["dataset_index"],
                "stack_index_one_based": score_capture.header[6],
                "stratum": expected_particles[part_id]["stratum"],
                "candidate_count": int(np.prod(native["raw_score"].shape)),
                "mapping": {
                    "native_directions": n_directions,
                    "native_psi": n_psi,
                    "euler_transpose_max_abs": euler_max_abs,
                    "translation_phase_max_abs": translation_phase_max_abs,
                },
                "operand_comparison": {
                    "projected_reference": _array_comparison(
                        native_references_array, recovar["references"]
                    ),
                    "shifted_image": _array_comparison(
                        native_shifted, recovar["shifted_data"]
                    ),
                    "correction": _array_comparison(
                        native_correction, recovar["ctf2_data"]
                    ),
                },
                "score_replay": {
                    "recovar_operands_vs_recovar_capture": _centered_comparison(
                        raw_surfaces["recovar_operands"], recovar["raw_score"]
                    ),
                    "all_native_operands_vs_native_capture": _centered_comparison(
                        raw_surfaces["all_native_operands"], native["raw_score"]
                    ),
                },
                "support_counterfactuals": support_factorial,
                "score_residual_energy": score_energy,
                "artifacts": {
                    "recovar": {
                        "path": str(recovar["path"]),
                        "sha256": recovar["sha256"],
                    },
                    "native_score": {
                        "path": str(score_capture.path.resolve()),
                        "sha256": score_capture.sha256,
                    },
                    "native_operands": [
                        {
                            "path": str(item.path.resolve()),
                            "sha256": item.sha256,
                        }
                        for item in particle_operands
                    ],
                },
            }
        )

    support_fields = tuple(records[0]["support_counterfactuals"])
    aggregate_support = {
        field: _aggregate_support(records, field) for field in support_fields
    }
    aggregate_energy = _aggregate_energy(records)
    summary = {
        "particle_records": len(records),
        "candidate_values": sum(item["candidate_count"] for item in records),
        "support_counterfactuals": aggregate_support,
        "score_residual_energy": aggregate_energy,
        "operand_relative_l2": {
            field: _quantiles(
                item["operand_comparison"][field]["relative_l2"]
                for item in records
            )
            for field in ("projected_reference", "shifted_image", "correction")
        },
        "mapping": {
            "euler_transpose_max_abs": max(
                item["mapping"]["euler_transpose_max_abs"] for item in records
            ),
            "translation_phase_max_abs": max(
                item["mapping"]["translation_phase_max_abs"] for item in records
            ),
        },
    }
    summary["classification"] = classify(summary, len(records))
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "scientific_scope": (
            "16 frozen EMPIAR-10076 shared-200 particles at K=4 iteration 1/coarse "
            "current-size 20; all 4 classes, 576 rotations, 29 translations, and "
            "three independent paired observer-inertness repeats"
        ),
        "metric_policy": (
            "exact native significant-support recovery and centered squared-residual "
            "energy are causal metrics; operand relative L2 is descriptive; no "
            "correlation metric is used for acceptance"
        ),
        "fixed_gates": {
            "euler_transpose_max_abs": EULER_MAX_ABS_GATE,
            "translation_phase_max_abs": TRANSLATION_PHASE_MAX_ABS_GATE,
            "projected_reference_fraction_baseline_energy_remaining": (
                REFERENCE_ENERGY_REMAINING_CEILING
            ),
            "nonreference_fraction_baseline_energy_remaining": (
                NONREFERENCE_ENERGY_REMAINING_FLOOR
            ),
        },
        "inputs": {
            "capture_directory": str(capture_dir),
            "significance_directory": str(significance_dir.resolve()),
            "support_report": str(support_report_path.resolve()),
            "support_report_sha256": _sha256(support_report_path),
            "operand_validation": str(operand_validation_path.resolve()),
            "operand_validation_sha256": _sha256(operand_validation_path),
            "repeatability_gate": str(repeatability_path.resolve()),
            "repeatability_gate_sha256": _sha256(repeatability_path),
            "full_image_size": full_image_size,
        },
        "summary": summary,
        "particles": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--significance-dir", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--operand-validation", type=Path, required=True)
    parser.add_argument("--repeatability-gate", type=Path, required=True)
    parser.add_argument("--full-image-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    report = build_report(
        capture_dir=args.capture_dir,
        significance_dir=args.significance_dir,
        support_report_path=args.support_report,
        operand_validation_path=args.operand_validation,
        repeatability_path=args.repeatability_gate,
        full_image_size=args.full_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
