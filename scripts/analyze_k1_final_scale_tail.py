#!/usr/bin/env python3
"""Measure whether final-boundary scale errors enrich K=1 particle-state tails."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import starfile


EXPOSURE_QUANTILES = (0.90, 0.95, 0.99, 0.999)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stack_indices(image_names) -> np.ndarray:
    return np.asarray(
        [int(str(name).split("@", maxsplit=1)[0]) - 1 for name in image_names],
        dtype=np.int64,
    )


def exposure_panel(values: np.ndarray, targets: dict[str, np.ndarray]) -> dict[str, object]:
    magnitudes = np.asarray(values, dtype=np.float64)
    if magnitudes.ndim != 1 or not np.isfinite(magnitudes).all() or np.any(magnitudes < 0):
        raise ValueError("exposure values must be one-dimensional, finite, and nonnegative")
    normalized_targets: dict[str, np.ndarray] = {}
    for name, target in targets.items():
        array = np.asarray(target, dtype=bool)
        if array.shape != magnitudes.shape:
            raise ValueError(f"target {name} shape differs from exposure values")
        normalized_targets[name] = array

    panel: dict[str, object] = {}
    for quantile in EXPOSURE_QUANTILES:
        threshold = float(np.quantile(magnitudes, quantile))
        exposed = magnitudes >= threshold
        if np.all(exposed) or not np.any(exposed):
            raise ValueError(f"quantile {quantile} does not split the exposure population")
        outcomes = {}
        for name, target in normalized_targets.items():
            exposed_fraction = float(np.mean(target[exposed]))
            remainder_fraction = float(np.mean(target[~exposed]))
            outcomes[name] = {
                "exposed_fraction": exposed_fraction,
                "remainder_fraction": remainder_fraction,
                "enrichment": (
                    exposed_fraction / remainder_fraction
                    if remainder_fraction > 0.0
                    else None
                ),
            }
        panel[f"q{quantile:g}"] = {
            "threshold": threshold,
            "exposed_count": int(np.count_nonzero(exposed)),
            "outcomes": outcomes,
        }
    return panel


def _model_table(path: Path, name: str):
    tables = starfile.read(path, always_dict=True)
    if name not in tables:
        raise ValueError(f"{path} has no {name} table")
    return tables[name]


def analyze(
    *,
    recovar_results: Path,
    recovar_manifest_half1: Path,
    recovar_manifest_half2: Path,
    particle_state_arrays: Path,
    relion_data_star: Path,
    relion_model_half1: Path,
    relion_model_half2: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(recovar_results, allow_pickle=False) as results:
        half_indices = [
            np.asarray(results["half1_indices"], dtype=np.int64),
            np.asarray(results["half2_indices"], dtype=np.int64),
        ]
        n_images = int(np.asarray(results["n_images"]).item())
    joined = np.concatenate(half_indices)
    if joined.size != n_images or np.unique(joined).size != n_images:
        raise ValueError("RECOVAR half indices are not a complete disjoint particle partition")

    data = _model_table(relion_data_star, "particles")
    source_indices = _stack_indices(data["rlnImageName"])
    if source_indices.size != n_images or not np.array_equal(
        np.sort(source_indices), np.arange(n_images, dtype=np.int64)
    ):
        raise ValueError("RELION data STAR stack identities are not a complete zero-based range")
    source_to_relion_row = np.empty(n_images, dtype=np.int64)
    source_to_relion_row[source_indices] = np.arange(n_images, dtype=np.int64)
    norm_corrections = np.asarray(data["rlnNormCorrection"], dtype=np.float64)
    group_numbers = np.asarray(data["rlnGroupNumber"], dtype=np.int64)

    scale_abs = np.empty(n_images, dtype=np.float64)
    image_correction_abs = np.empty(n_images, dtype=np.float64)
    manifests = (recovar_manifest_half1, recovar_manifest_half2)
    models = (relion_model_half1, relion_model_half2)
    per_half = []
    for half_index, (indices, manifest_path, model_path) in enumerate(
        zip(half_indices, manifests, models, strict=True)
    ):
        general = _model_table(model_path, "model_general")
        groups = _model_table(model_path, "model_groups")
        average_norm = float(np.asarray(general["rlnNormCorrectionAverage"]).reshape(-1)[0])
        group_scales = np.asarray(groups["rlnGroupScaleCorrection"], dtype=np.float64)
        relion_rows = source_to_relion_row[indices]
        relion_scale = group_scales[
            np.clip(group_numbers[relion_rows] - 1, 0, group_scales.size - 1)
        ].astype(np.float32)
        relion_image_correction = (
            (average_norm / norm_corrections[relion_rows])
            * relion_scale.astype(np.float64)
        ).astype(np.float32)
        with np.load(manifest_path, allow_pickle=False) as manifest:
            if int(np.asarray(manifest["half_index"]).item()) != half_index:
                raise ValueError(f"manifest {manifest_path} has the wrong half index")
            recovar_scale = np.asarray(manifest["scale_corrections"], dtype=np.float64)
            recovar_image_correction = np.asarray(
                manifest["image_corrections"], dtype=np.float64
            )
        if recovar_scale.shape != relion_scale.shape:
            raise ValueError(f"half-{half_index + 1} scale-correction shape differs")
        scale_delta = recovar_scale - relion_scale.astype(np.float64)
        correction_delta = recovar_image_correction - relion_image_correction.astype(np.float64)
        scale_abs[indices] = np.abs(scale_delta)
        image_correction_abs[indices] = np.abs(correction_delta)
        per_half.append(
            {
                "half": half_index + 1,
                "n": int(indices.size),
                "scale_relative_l2": float(
                    np.linalg.norm(scale_delta) / np.linalg.norm(relion_scale.astype(np.float64))
                ),
                "scale_max_abs": float(np.max(np.abs(scale_delta))),
                "scale_p95_abs": float(np.quantile(np.abs(scale_delta), 0.95)),
                "image_correction_relative_l2": float(
                    np.linalg.norm(correction_delta)
                    / np.linalg.norm(relion_image_correction.astype(np.float64))
                ),
                "image_correction_max_abs": float(np.max(np.abs(correction_delta))),
                "image_correction_p95_abs": float(np.quantile(np.abs(correction_delta), 0.95)),
            }
        )

    with np.load(particle_state_arrays, allow_pickle=False) as state:
        identity_rows = np.asarray(state["identity_row_index"], dtype=np.int64)
        if identity_rows.size != n_images or np.unique(identity_rows).size != n_images:
            raise ValueError("particle-state identities are not unique and complete")
        targets_by_audit_row = {
            "rotation_gt_0p01_deg": np.asarray(state["it015_rotation_geodesic_deg"]) > 0.01,
            "translation_gt_0p01_angstrom": np.asarray(state["it015_translation_l2"]) > 0.01,
            "support_count_changed": np.asarray(state["it015_support_delta"]) != 0,
        }
        pmax_abs_by_audit_row = np.abs(np.asarray(state["it015_pmax_delta"], dtype=np.float64))
    targets = {}
    pmax_abs = np.empty(n_images, dtype=np.float64)
    pmax_abs[identity_rows] = pmax_abs_by_audit_row
    for name, values in targets_by_audit_row.items():
        target = np.empty(n_images, dtype=bool)
        target[identity_rows] = values
        targets[name] = target

    scale_panel = exposure_panel(scale_abs, targets)
    correction_panel = exposure_panel(image_correction_abs, targets)
    for panel, values in ((scale_panel, scale_abs), (correction_panel, image_correction_abs)):
        for quantile in EXPOSURE_QUANTILES:
            entry = panel[f"q{quantile:g}"]
            exposed = values >= entry["threshold"]
            entry["pmax_abs_mean_exposed"] = float(np.mean(pmax_abs[exposed]))
            entry["pmax_abs_mean_remainder"] = float(np.mean(pmax_abs[~exposed]))

    sources = {
        "recovar_results": recovar_results,
        "recovar_manifest_half1": recovar_manifest_half1,
        "recovar_manifest_half2": recovar_manifest_half2,
        "particle_state_arrays": particle_state_arrays,
        "relion_data_star": relion_data_star,
        "relion_model_half1": relion_model_half1,
        "relion_model_half2": relion_model_half2,
    }
    report = {
        "schema": "recovar.em.k1_final_scale_tail.v1",
        "status": "complete",
        "n_images": n_images,
        "metric_policy": "tail enrichment only; no correlation and no acceptance gate",
        "per_half": per_half,
        "scale_absolute_error_exposure": scale_panel,
        "image_correction_absolute_error_exposure": correction_panel,
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
    }
    arrays = {
        "source_row": np.arange(n_images, dtype=np.int64),
        "scale_correction_abs_error": scale_abs,
        "image_correction_abs_error": image_correction_abs,
        "pmax_abs_error": pmax_abs,
        **{name: value for name, value in targets.items()},
    }
    return report, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-results", required=True, type=Path)
    parser.add_argument("--recovar-manifest-half1", required=True, type=Path)
    parser.add_argument("--recovar-manifest-half2", required=True, type=Path)
    parser.add_argument("--particle-state-arrays", required=True, type=Path)
    parser.add_argument("--relion-data-star", required=True, type=Path)
    parser.add_argument("--relion-model-half1", required=True, type=Path)
    parser.add_argument("--relion-model-half2", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_npz.exists():
        raise FileExistsError("refusing to overwrite an existing output artifact")
    report, arrays = analyze(
        recovar_results=args.recovar_results,
        recovar_manifest_half1=args.recovar_manifest_half1,
        recovar_manifest_half2=args.recovar_manifest_half2,
        particle_state_arrays=args.particle_state_arrays,
        relion_data_star=args.relion_data_star,
        relion_model_half1=args.relion_model_half1,
        relion_model_half2=args.relion_model_half2,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    report["output_arrays"] = {
        "path": str(args.output_npz.resolve()),
        "sha256": _sha256(args.output_npz),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
