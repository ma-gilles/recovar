#!/usr/bin/env python3
"""Localize the real-data K=4 coarse gap to likelihood score components.

This analyzer joins passive RECOVAR reference-norm/cross-term surfaces with a
paired native RELION coarse-score/component capture.  It replaces one
likelihood component at a time and measures exact significant-support recovery.
The counterfactuals do not alter either engine's production trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_real_k4_coarse_score_support import (
    _stable_top_count_mask,
    _support_metric,
)
from scripts.analyze_em_real_k4_native_coarse_scores import (
    SUPPORT_SCHEMA,
    _aggregate_comparison,
    _aggregate_support,
    _centered_comparison,
    _load_recovar_dump,
    _native_surface,
    _quantiles,
    _require,
    _sha256,
)
from scripts.validate_relion_coarse_component_capture import (
    CoarseComponentCapture,
    load_coarse_component_capture,
)
from scripts.validate_relion_coarse_score_capture import load_coarse_score_capture

SCHEMA = "recovar.em_real_k4_native_coarse_components.v1"
COMPONENT_VALIDATION_SCHEMA = "relion-coarse-component-validation-v1"
CAPTURE_INERTNESS_SCHEMA = "recovar.relion_k4_coarse_score_capture_inertness.v1"
COMPONENT_CLOSURE_RMS_CEILING = 1.0e-3


def _native_component_surface(
    capture: CoarseComponentCapture,
    field: str,
) -> np.ndarray:
    """Convert one native direction-major component to RECOVAR score order."""

    class_min, n_classes, n_directions, n_psi, n_translations = capture.header[9:14]
    _require(class_min == 0, "native component capture does not start at class zero")
    native_shape = (n_classes, n_directions, n_psi, n_translations)
    recovar_shape = (n_classes, n_psi * n_directions, n_translations)
    values = np.asarray(capture.candidates[field], dtype=np.float64).reshape(native_shape)
    # RELION diff2 components become likelihood scores after changing sign.
    return -values.transpose(0, 2, 1, 3).reshape(recovar_shape)


def _load_recovar_component_surfaces(
    path: Path,
    expected_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "projected_reference_rotation_ids",
            "projected_reference_norm_score_per_class",
            "projected_cross_score_per_class",
        }
        _require(
            required <= set(payload.files),
            f"RECOVAR dump lacks component fields {sorted(required - set(payload.files))}",
        )
        rotation_ids = np.asarray(
            payload["projected_reference_rotation_ids"], dtype=np.int32
        )
        _require(
            np.array_equal(rotation_ids, np.arange(expected_shape[1], dtype=np.int32)),
            "RECOVAR component dump does not cover every rotation in production order",
        )
        reference_norm = np.asarray(
            payload["projected_reference_norm_score_per_class"], dtype=np.float64
        )
        cross_term = np.asarray(
            payload["projected_cross_score_per_class"], dtype=np.float64
        )
    _require(
        reference_norm.shape == cross_term.shape == expected_shape,
        "RECOVAR component topology differs",
    )
    _require(
        np.isfinite(reference_norm).all() and np.isfinite(cross_term).all(),
        "RECOVAR component surface is non-finite",
    )
    return {"reference_norm_score": reference_norm, "cross_term_score": cross_term}


def _centered(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array - float(np.mean(array))


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _residual_decomposition(
    *,
    recovar_raw: np.ndarray,
    native_raw: np.ndarray,
    recovar_norm: np.ndarray,
    native_norm: np.ndarray,
    recovar_cross: np.ndarray,
    native_cross: np.ndarray,
) -> dict[str, float | int]:
    """Decompose the centered raw-score residual into norm and cross residuals."""

    arrays = (
        recovar_raw,
        native_raw,
        recovar_norm,
        native_norm,
        recovar_cross,
        native_cross,
    )
    shape = np.asarray(arrays[0]).shape
    _require(all(np.asarray(array).shape == shape for array in arrays), "component shape differs")
    raw_residual = _centered(recovar_raw - native_raw)
    norm_residual = _centered(recovar_norm - native_norm)
    cross_residual = _centered(recovar_cross - native_cross)
    closure = raw_residual - norm_residual - cross_residual
    recovar_closure = _centered(recovar_raw - recovar_norm - recovar_cross)
    native_closure = _centered(native_raw - native_norm - native_cross)
    raw_energy = float(np.sum(raw_residual * raw_residual))
    norm_energy = float(np.sum(norm_residual * norm_residual))
    cross_energy = float(np.sum(cross_residual * cross_residual))
    _require(raw_energy > 0, "raw-score residual is only an additive constant")
    return {
        "value_count": int(raw_residual.size),
        "raw_residual_centered_rms": _rms(raw_residual),
        "norm_residual_centered_rms": _rms(norm_residual),
        "cross_residual_centered_rms": _rms(cross_residual),
        "component_residual_closure_centered_rms": _rms(closure),
        "component_residual_closure_centered_max_abs": float(np.max(np.abs(closure))),
        "recovar_component_closure_centered_rms": _rms(recovar_closure),
        "native_component_closure_centered_rms": _rms(native_closure),
        "raw_residual_centered_squared_energy": raw_energy,
        "norm_residual_centered_squared_energy": norm_energy,
        "cross_residual_centered_squared_energy": cross_energy,
        "fraction_raw_residual_energy_remaining_after_native_cross": norm_energy
        / raw_energy,
        "fraction_raw_residual_energy_remaining_after_native_norm": cross_energy
        / raw_energy,
    }


def _component_support_counterfactuals(
    *,
    native: dict[str, np.ndarray],
    recovar: dict[str, np.ndarray],
    native_norm: np.ndarray,
    native_cross: np.ndarray,
    recovar_norm: np.ndarray,
    recovar_cross: np.ndarray,
) -> dict[str, Any]:
    reference = native["significant"]
    count = int(np.count_nonzero(reference))
    surfaces = {
        "native_captured_combined": native["combined_score"],
        "recovar_captured_combined": recovar["combined_score"],
        "recovar_norm_native_cross_native_prior": (
            recovar_norm + native_cross + native["total_prior"]
        ),
        "native_norm_recovar_cross_native_prior": (
            native_norm + recovar_cross + native["total_prior"]
        ),
        "recovar_norm_native_cross_recovar_prior": (
            recovar_norm + native_cross + recovar["total_prior"]
        ),
        "native_norm_recovar_cross_recovar_prior": (
            native_norm + recovar_cross + recovar["total_prior"]
        ),
    }
    report: dict[str, Any] = {}
    selected: dict[str, np.ndarray] = {}
    for name, surface in surfaces.items():
        mask, cutoff = _stable_top_count_mask(surface, count)
        selected[name] = mask
        report[name] = {**_support_metric(reference, mask), **cutoff}
    report["set_identities"] = {
        "cross_swap_is_prior_invariant": bool(
            np.array_equal(
                selected["recovar_norm_native_cross_native_prior"],
                selected["recovar_norm_native_cross_recovar_prior"],
            )
        ),
        "norm_swap_is_prior_invariant": bool(
            np.array_equal(
                selected["native_norm_recovar_cross_native_prior"],
                selected["native_norm_recovar_cross_recovar_prior"],
            )
        ),
    }
    return report


def _aggregate_residual_decomposition(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [record["residual_decomposition"] for record in records]
    raw_energy = sum(
        float(metric["raw_residual_centered_squared_energy"]) for metric in metrics
    )
    norm_energy = sum(
        float(metric["norm_residual_centered_squared_energy"]) for metric in metrics
    )
    cross_energy = sum(
        float(metric["cross_residual_centered_squared_energy"]) for metric in metrics
    )
    quantile_fields = (
        "raw_residual_centered_rms",
        "norm_residual_centered_rms",
        "cross_residual_centered_rms",
        "component_residual_closure_centered_rms",
        "recovar_component_closure_centered_rms",
        "native_component_closure_centered_rms",
        "fraction_raw_residual_energy_remaining_after_native_cross",
        "fraction_raw_residual_energy_remaining_after_native_norm",
    )
    return {
        "records": len(metrics),
        "value_count": sum(int(metric["value_count"]) for metric in metrics),
        "pooled_fraction_raw_residual_energy_remaining_after_native_cross": (
            norm_energy / raw_energy
        ),
        "pooled_fraction_raw_residual_energy_remaining_after_native_norm": (
            cross_energy / raw_energy
        ),
        "component_residual_closure_centered_max_abs": max(
            float(metric["component_residual_closure_centered_max_abs"])
            for metric in metrics
        ),
        **{
            field: _quantiles(float(metric[field]) for metric in metrics)
            for field in quantile_fields
        },
    }


def _classification(records: list[dict[str, Any]]) -> str:
    native = _aggregate_support(records, "native_captured_combined")
    baseline = _aggregate_support(records, "recovar_captured_combined")
    cross_swap = _aggregate_support(
        records, "recovar_norm_native_cross_native_prior"
    )
    norm_swap = _aggregate_support(records, "native_norm_recovar_cross_native_prior")
    if (
        native["exact_records"] == len(records)
        and cross_swap["exact_records"] > baseline["exact_records"]
        and cross_swap["exact_records"] > norm_swap["exact_records"]
        and cross_swap["jaccard"] > norm_swap["jaccard"]
    ):
        return "cross_term_is_dominant_component_of_first_material_likelihood_difference"
    if (
        native["exact_records"] == len(records)
        and norm_swap["exact_records"] > baseline["exact_records"]
        and norm_swap["jaccard"] > cross_swap["jaccard"]
    ):
        return "reference_norm_is_dominant_component_of_first_material_likelihood_difference"
    if native["exact_records"] == len(records):
        return "norm_and_cross_components_both_material_or_not_separated"
    return "native_component_capture_does_not_reproduce_native_support"


def _load_gate(path: Path, schema: str, status_field: str, expected: Any) -> dict[str, Any]:
    report = json.loads(path.read_text())
    _require(report.get("schema") == schema, f"gate schema differs: {path}")
    _require(report.get(status_field) == expected, f"gate failed: {path}")
    return report


def build_report(
    *,
    capture_dir: Path,
    significance_dir: Path,
    support_report_path: Path,
    component_validation_path: Path,
    capture_inertness_path: Path,
) -> dict[str, Any]:
    support_report = json.loads(support_report_path.read_text())
    _require(support_report.get("schema") == SUPPORT_SCHEMA, "support report schema differs")
    component_validation = _load_gate(
        component_validation_path,
        COMPONENT_VALIDATION_SCHEMA,
        "capture_ready",
        True,
    )
    capture_inertness = _load_gate(
        capture_inertness_path,
        CAPTURE_INERTNESS_SCHEMA,
        "status",
        "pass",
    )

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

    component_paths = sorted(capture_dir.glob("*.coarse-components-v1.bin"))
    components = [load_coarse_component_capture(path) for path in component_paths]
    component_by_particle = {capture.header[5]: capture for capture in components}
    _require(len(component_by_particle) == len(components), "duplicate native component capture")
    _require(set(component_by_particle) == set(expected_particles), "native component panel differs")
    _require(
        int(component_validation["particle_count"]) == len(components),
        "component-validation particle count differs",
    )

    dump_paths = sorted(significance_dir.glob("significance_*.npz"))
    dumps = [_load_recovar_dump(path) for path in dump_paths]
    dump_by_dataset = {dump["dataset_index"]: dump for dump in dumps}
    _require(len(dump_by_dataset) == len(dumps), "duplicate RECOVAR component dump")
    _require(set(dump_by_dataset) == set(dataset_by_particle.values()), "RECOVAR panel differs")

    records = []
    for particle_id in sorted(expected_particles):
        expected = expected_particles[particle_id]
        component_capture = component_by_particle[particle_id]
        score_path = component_capture.path.with_name(
            component_capture.path.name.replace(
                ".coarse-components-v1.bin", ".coarse-score-v1.bin"
            )
        )
        score_capture = load_coarse_score_capture(score_path)
        _require(
            component_capture.header[4:15] == score_capture.header[4:15],
            "native score/component identities differ",
        )
        native = _native_surface(score_capture)
        recovar = dump_by_dataset[dataset_by_particle[particle_id]]
        shape = native["raw_score"].shape
        expected_shape = (
            recovar["n_classes"],
            recovar["n_rotations"],
            recovar["n_translations"],
        )
        _require(shape == expected_shape, "native/RECOVAR component topology differs")
        _require(
            score_capture.header[6] == expected["stack_index_one_based"],
            "native stack join differs",
        )
        recovar_components = _load_recovar_component_surfaces(
            recovar["path"], expected_shape
        )
        native_norm = _native_component_surface(component_capture, "reference_norm")
        native_cross = _native_component_surface(component_capture, "cross_term")
        recovar_norm = recovar_components["reference_norm_score"]
        recovar_cross = recovar_components["cross_term_score"]
        residual = _residual_decomposition(
            recovar_raw=recovar["raw_score"],
            native_raw=native["raw_score"],
            recovar_norm=recovar_norm,
            native_norm=native_norm,
            recovar_cross=recovar_cross,
            native_cross=native_cross,
        )
        _require(
            float(residual["component_residual_closure_centered_rms"])
            <= COMPONENT_CLOSURE_RMS_CEILING,
            "paired component residual does not close to the captured raw score",
        )
        support_counterfactuals = _component_support_counterfactuals(
            native=native,
            recovar=recovar,
            native_norm=native_norm,
            native_cross=native_cross,
            recovar_norm=recovar_norm,
            recovar_cross=recovar_cross,
        )
        records.append(
            {
                "native_particle_id_zero_based": particle_id,
                "dataset_index_zero_based": recovar["dataset_index"],
                "stack_index_one_based": score_capture.header[6],
                "stratum": expected["stratum"],
                "candidate_count": int(np.prod(shape)),
                "native_component_capture": {
                    "path": str(component_capture.path.resolve()),
                    "sha256": component_capture.sha256,
                },
                "native_score_capture": {
                    "path": str(score_capture.path.resolve()),
                    "sha256": score_capture.sha256,
                },
                "recovar_dump": {
                    "path": str(recovar["path"]),
                    "sha256": recovar["sha256"],
                },
                "component_comparison": {
                    "reference_norm_score": _centered_comparison(
                        recovar_norm, native_norm
                    ),
                    "cross_term_score": _centered_comparison(
                        recovar_cross, native_cross
                    ),
                    "raw_likelihood_score": _centered_comparison(
                        recovar["raw_score"], native["raw_score"]
                    ),
                },
                "residual_decomposition": residual,
                "support_counterfactuals": support_counterfactuals,
            }
        )

    support_fields = (
        "native_captured_combined",
        "recovar_captured_combined",
        "recovar_norm_native_cross_native_prior",
        "native_norm_recovar_cross_native_prior",
        "recovar_norm_native_cross_recovar_prior",
        "native_norm_recovar_cross_recovar_prior",
    )
    comparison_fields = (
        "reference_norm_score",
        "cross_term_score",
        "raw_likelihood_score",
    )
    aggregate_support = {
        field: _aggregate_support(records, field) for field in support_fields
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_scope": (
            "16 frozen EMPIAR-10076 shared-200 particles at K=4 iteration 1/coarse "
            "current-size 20; all 4 classes, 576 rotations, and 29 translations"
        ),
        "metric_policy": (
            "exact native significant-support recovery is the causal metric; component "
            "residuals remove only one per-particle additive image constant; correlation "
            "is not an acceptance metric"
        ),
        "thresholds": {
            "component_residual_closure_centered_rms_ceiling": (
                COMPONENT_CLOSURE_RMS_CEILING
            )
        },
        "inputs": {
            "capture_directory": str(capture_dir.resolve()),
            "significance_directory": str(significance_dir.resolve()),
            "support_report": str(support_report_path.resolve()),
            "support_report_sha256": _sha256(support_report_path),
            "component_validation": str(component_validation_path.resolve()),
            "component_validation_sha256": _sha256(component_validation_path),
            "capture_inertness": str(capture_inertness_path.resolve()),
            "capture_inertness_sha256": _sha256(capture_inertness_path),
            "capture_inertness_classification": capture_inertness["classification"],
        },
        "summary": {
            "classification": _classification(records),
            "particle_records": len(records),
            "candidate_values": sum(record["candidate_count"] for record in records),
            "support_counterfactuals": aggregate_support,
            "component_comparison": {
                field: _aggregate_comparison(records, field)
                for field in comparison_fields
            },
            "residual_decomposition": _aggregate_residual_decomposition(records),
            "prior_invariance": {
                "cross_swap_exact_records": sum(
                    bool(
                        record["support_counterfactuals"]["set_identities"][
                            "cross_swap_is_prior_invariant"
                        ]
                    )
                    for record in records
                ),
                "norm_swap_exact_records": sum(
                    bool(
                        record["support_counterfactuals"]["set_identities"][
                            "norm_swap_is_prior_invariant"
                        ]
                    )
                    for record in records
                ),
            },
        },
        "particles": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--significance-dir", required=True, type=Path)
    parser.add_argument("--support-report", required=True, type=Path)
    parser.add_argument("--component-validation", required=True, type=Path)
    parser.add_argument("--capture-inertness", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    report = build_report(
        capture_dir=args.capture_dir,
        significance_dir=args.significance_dir,
        support_report_path=args.support_report,
        component_validation_path=args.component_validation,
        capture_inertness_path=args.capture_inertness,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
