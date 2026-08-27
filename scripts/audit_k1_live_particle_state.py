#!/usr/bin/env python3
"""Audit one live RECOVAR intermediate particle state against RELION."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_em_particle_state_distribution import (
    AuditError,
    _angular_error_deg,
    _cohort_metrics,
    _identity_array,
    _identity_sha256,
    _load_relion_state,
    _particle_table,
)

_TOP_DISCREPANCY_COUNT = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_half(path: Path, *, expected_iteration: int, expected_half: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise AuditError(f"missing live particle-state archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "rotation_eulers_deg",
            "absolute_translations_pixels",
            "max_posterior",
            "significant_counts",
            "original_image_indices",
            "zero_based_iteration",
            "one_based_iteration",
            "half",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise AuditError(f"{path} is missing required fields: {missing}")
        result = {key: np.asarray(archive[key]) for key in required}

    scalar_expectations = {
        "zero_based_iteration": expected_iteration,
        "one_based_iteration": expected_iteration + 1,
        "half": expected_half,
    }
    for key, expected in scalar_expectations.items():
        values = np.asarray(result[key]).reshape(-1)
        if values.size != 1 or int(values[0]) != expected:
            raise AuditError(f"{path}:{key}={values.tolist()}, expected [{expected}]")

    indices = np.asarray(result["original_image_indices"], dtype=np.int64).reshape(-1)
    n_half = indices.size
    expected_shapes = {
        "rotation_eulers_deg": (n_half, 3),
        "absolute_translations_pixels": (n_half, 2),
        "max_posterior": (n_half,),
        "significant_counts": (n_half,),
    }
    for key, shape in expected_shapes.items():
        if result[key].shape != shape:
            raise AuditError(f"{path}:{key} shape={result[key].shape}, expected {shape}")
        if not np.isfinite(result[key]).all():
            raise AuditError(f"{path}:{key} contains non-finite values")
    counts = np.asarray(result["significant_counts"], dtype=np.float64)
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise AuditError(f"{path}:significant_counts contains invalid counts")
    result["original_image_indices"] = indices
    return result


def _top_discrepancy_rows(
    values: np.ndarray,
    *,
    identities: np.ndarray,
    half_labels: np.ndarray,
    recovar_values: np.ndarray,
    relion_values: np.ndarray,
    signed: bool,
) -> list[dict[str, object]]:
    metric = np.asarray(values, dtype=np.float64).reshape(-1)
    recovar = np.asarray(recovar_values)
    relion = np.asarray(relion_values)
    if metric.size != identities.size or recovar.shape[0] != metric.size or relion.shape[0] != metric.size:
        raise AuditError("top-discrepancy arrays are not particle aligned")
    ranking = np.argsort(-np.abs(metric), kind="stable")[:_TOP_DISCREPANCY_COUNT]
    rows = []
    for index in ranking:
        recovar_value = np.asarray(recovar[index])
        relion_value = np.asarray(relion[index])
        rows.append(
            {
                "source_row_zero_based": int(index),
                "identity": str(identities[index]),
                "half": int(half_labels[index]),
                "metric": float(metric[index]),
                "metric_is_signed": bool(signed),
                "recovar": recovar_value.item() if recovar_value.ndim == 0 else recovar_value.tolist(),
                "relion": relion_value.item() if relion_value.ndim == 0 else relion_value.tolist(),
            }
        )
    return rows


def audit_live_particle_state(
    *,
    intermediates_dir: Path,
    recovar_particles_star: Path,
    relion_star: Path,
    recovar_iteration: int,
    pixel_size_angstrom: float,
) -> dict[str, object]:
    intermediates_dir = intermediates_dir.resolve()
    recovar_particles_star = recovar_particles_star.resolve()
    relion_star = relion_star.resolve()
    if recovar_iteration < 0:
        raise AuditError("RECOVAR iteration must be non-negative")
    if not np.isfinite(pixel_size_angstrom) or pixel_size_angstrom <= 0.0:
        raise AuditError("pixel size must be finite and positive")

    source_table = _particle_table(recovar_particles_star)
    identities = _identity_array(source_table, source=recovar_particles_star)
    n_images = identities.size
    tag = f"it{recovar_iteration:03d}"
    half_paths = {
        half: intermediates_dir / f"{tag}_particle_state_half{half}.npz"
        for half in (1, 2)
    }
    halves = {
        half: _load_half(path, expected_iteration=recovar_iteration, expected_half=half)
        for half, path in half_paths.items()
    }

    all_indices = np.concatenate(
        [halves[half]["original_image_indices"] for half in (1, 2)]
    )
    if all_indices.size != n_images:
        raise AuditError(
            f"live halves contain {all_indices.size} rows, input STAR contains {n_images}"
        )
    if not np.array_equal(np.sort(all_indices), np.arange(n_images, dtype=np.int64)):
        raise AuditError("live half indices are not a disjoint complete source-row partition")

    recovar_state = {
        "pmax": np.empty(n_images, dtype=np.float64),
        "support": np.empty(n_images, dtype=np.int64),
        "eulers": np.empty((n_images, 3), dtype=np.float64),
        "translations": np.empty((n_images, 2), dtype=np.float64),
        "translation_units": None,
        "classes": None,
    }
    half_labels = np.empty(n_images, dtype=np.int8)
    for half in (1, 2):
        state = halves[half]
        indices = state["original_image_indices"]
        recovar_state["pmax"][indices] = state["max_posterior"]
        recovar_state["support"][indices] = state["significant_counts"]
        recovar_state["eulers"][indices] = state["rotation_eulers_deg"]
        recovar_state["translations"][indices] = state["absolute_translations_pixels"]
        half_labels[indices] = half

    relion_state, _ = _load_relion_state(relion_star, identities)
    if relion_state["translation_units"] == "angstrom":
        recovar_state["translations"] *= float(pixel_size_angstrom)
        recovar_state["translation_units"] = "angstrom"
    elif relion_state["translation_units"] == "pixel":
        recovar_state["translation_units"] = "pixel"
    elif relion_state["translations"] is not None:
        raise AuditError("RELION translation units could not be resolved")
    else:
        recovar_state["translations"] = None

    all_metrics = _cohort_metrics(
        np.ones(n_images, dtype=bool), recovar_state, relion_state
    )
    half_metrics = {
        str(half): _cohort_metrics(half_labels == half, recovar_state, relion_state)
        for half in (1, 2)
    }
    pmax_delta = np.asarray(recovar_state["pmax"]) - np.asarray(relion_state["pmax"])
    support_delta = np.asarray(recovar_state["support"]) - np.asarray(relion_state["support"])
    angular_error = _angular_error_deg(recovar_state["eulers"], relion_state["eulers"])
    largest_discrepancies = {
        "pmax_signed_delta": _top_discrepancy_rows(
            pmax_delta,
            identities=identities,
            half_labels=half_labels,
            recovar_values=recovar_state["pmax"],
            relion_values=relion_state["pmax"],
            signed=True,
        ),
        "support_signed_delta": _top_discrepancy_rows(
            support_delta,
            identities=identities,
            half_labels=half_labels,
            recovar_values=recovar_state["support"],
            relion_values=relion_state["support"],
            signed=True,
        ),
        "angular_error_deg": _top_discrepancy_rows(
            angular_error,
            identities=identities,
            half_labels=half_labels,
            recovar_values=recovar_state["eulers"],
            relion_values=relion_state["eulers"],
            signed=False,
        ),
    }
    if recovar_state["translations"] is not None and relion_state["translations"] is not None:
        translation_error = np.linalg.norm(
            np.asarray(recovar_state["translations"]) - np.asarray(relion_state["translations"]),
            axis=1,
        )
        largest_discrepancies["translation_error"] = _top_discrepancy_rows(
            translation_error,
            identities=identities,
            half_labels=half_labels,
            recovar_values=recovar_state["translations"],
            relion_values=relion_state["translations"],
            signed=False,
        )
    return {
        "schema": "recovar.em.k1_live_particle_state_audit.v2",
        "status": "complete",
        "quality_metric_policy": (
            "Exact identity-aligned particle-state distributions; correlation is not computed."
        ),
        "recovar_iteration": recovar_iteration,
        "relion_iteration": recovar_iteration + 1,
        "n_images": int(n_images),
        "identity_sha256": _identity_sha256(identities),
        "half_counts": {
            str(half): int(np.count_nonzero(half_labels == half)) for half in (1, 2)
        },
        "recovar_vs_relion": all_metrics,
        "per_half": half_metrics,
        "largest_discrepancies": largest_discrepancies,
        "sources": {
            "intermediates_dir": str(intermediates_dir),
            "recovar_particles_star": str(recovar_particles_star),
            "relion_star": str(relion_star),
            "pixel_size_angstrom": float(pixel_size_angstrom),
        },
        "input_sha256": {
            "recovar_particles_star": _sha256(recovar_particles_star),
            "relion_star": _sha256(relion_star),
            **{
                f"recovar_half{half}": _sha256(path)
                for half, path in half_paths.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediates-dir", type=Path, required=True)
    parser.add_argument("--recovar-particles-star", type=Path, required=True)
    parser.add_argument("--relion-star", type=Path, required=True)
    parser.add_argument("--recovar-iteration", type=int, required=True)
    parser.add_argument("--pixel-size-angstrom", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    report = audit_live_particle_state(
        intermediates_dir=args.intermediates_dir,
        recovar_particles_star=args.recovar_particles_star,
        relion_star=args.relion_star,
        recovar_iteration=args.recovar_iteration,
        pixel_size_angstrom=args.pixel_size_angstrom,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
