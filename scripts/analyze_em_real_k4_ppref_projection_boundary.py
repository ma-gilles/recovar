#!/usr/bin/env python3
"""Replay a native K=4 PPref through RECOVAR's CUDA texture projector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np

from scripts.analyze_em_k1_live_reference_counterfactual import (
    relion_reference_on_recovar_window,
)
from scripts.analyze_em_real_k4_native_coarse_operands import (
    OPERAND_VALIDATION_SCHEMA,
    REPEATABILITY_SCHEMA,
    _array_comparison,
    _load_recovar_operands,
    _native_rotation_order,
)
from scripts.analyze_em_real_k4_ppref_sources import SCHEMA as PPREF_SOURCE_SCHEMA
from scripts.validate_relion_coarse_score_capture import load_coarse_score_capture
from scripts.validate_relion_k4_coarse_operand_capture import load_artifact
from scripts.validate_relion_k4_ppref_capture import load_ppref

SCHEMA = "recovar.em_real_k4_ppref_projection_boundary.v1"
MATCH_RELATIVE_L2_CEILING = 1.0e-7
MATERIAL_RELATIVE_L2_FLOOR = 1.0e-4
SEPARATION_RATIO_FLOOR = 100.0
EULER_MAX_ABS_CEILING = 1.0e-5


class AnalysisError(RuntimeError):
    """Raised when independently captured panels cannot be joined exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, schema: str, status: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing gate: {path}")
    report = json.loads(path.read_text())
    _require(report.get("schema") == schema, f"gate schema differs: {path}")
    _require(report.get("status") == status, f"gate status differs: {path}")
    return report


def _read_raw_mrc(path: Path) -> np.ndarray:
    _require(path.is_file(), f"missing source map: {path}")
    with mrcfile.open(path, permissive=True) as stream:
        values = np.asarray(stream.data, dtype=np.float32).copy()
    _require(values.ndim == 3 and values.size > 0, f"invalid source map: {path}")
    _require(np.isfinite(values).all(), f"non-finite source map: {path}")
    return values


def compare_source_map_panels(
    ppref_source_maps: list[Path],
    operand_source_maps: list[Path],
) -> list[dict[str, Any]]:
    """Require map payload equality while allowing irrelevant MRC-header drift."""

    _require(len(ppref_source_maps) == len(operand_source_maps), "map panel sizes differ")
    _require(bool(ppref_source_maps), "source map panel is empty")
    records = []
    for class_index, (ppref_path, operand_path) in enumerate(
        zip(ppref_source_maps, operand_source_maps, strict=True)
    ):
        ppref_map = _read_raw_mrc(ppref_path)
        operand_map = _read_raw_mrc(operand_path)
        _require(ppref_map.shape == operand_map.shape, f"class {class_index} map shape differs")
        _require(
            np.array_equal(ppref_map, operand_map),
            f"class {class_index} map payload differs",
        )
        records.append(
            {
                "model_zero_based": class_index,
                "payload_shape": list(ppref_map.shape),
                "payload_bitwise_equal": True,
                "ppref_source_map": {
                    "path": str(ppref_path.resolve()),
                    "sha256": _sha256(ppref_path),
                },
                "operand_source_map": {
                    "path": str(operand_path.resolve()),
                    "sha256": _sha256(operand_path),
                },
            }
        )
    return records


def classify(
    *,
    texture_vs_native_l2: float,
    texture_vs_recovar_l2: float,
    native_vs_recovar_l2: float,
) -> str:
    """Classify which side of the frozen PPref projection boundary differs."""

    values = (texture_vs_native_l2, texture_vs_recovar_l2, native_vs_recovar_l2)
    _require(all(np.isfinite(value) and value >= 0 for value in values), "invalid error")
    if native_vs_recovar_l2 < MATERIAL_RELATIVE_L2_FLOOR:
        return "captured_projected_reference_difference_is_not_material"
    if (
        texture_vs_recovar_l2 <= MATCH_RELATIVE_L2_CEILING
        and texture_vs_native_l2
        >= SEPARATION_RATIO_FLOOR
        * max(texture_vs_recovar_l2, np.finfo(np.float64).tiny)
    ):
        return "native_vs_recovar_texture_projection_is_first_material_ppref_downstream_difference"
    if (
        texture_vs_native_l2 <= MATCH_RELATIVE_L2_CEILING
        and texture_vs_recovar_l2
        >= SEPARATION_RATIO_FLOOR
        * max(texture_vs_native_l2, np.finfo(np.float64).tiny)
    ):
        return "recovar_projected_reference_wiring_differs_after_matching_texture_primitive"
    return "ppref_projection_boundary_is_mixed_or_unresolved"


def _common_recovar_panel(significance_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = sorted(significance_dir.glob("significance_*.npz"))
    records = [_load_recovar_operands(path) for path in paths]
    _require(len(records) == 16, "RECOVAR significance denominator differs from 16")
    first = records[0]
    _require(first["n_classes"] == 4, "RECOVAR class count differs from four")
    _require(len({record["dataset_index"] for record in records}) == 16, "duplicate dataset index")
    for record in records[1:]:
        for field in ("n_classes", "n_rotations", "n_translations", "current_size"):
            _require(record[field] == first[field], f"RECOVAR {field} differs across particles")
        for field in ("rotations", "window_indices", "references"):
            _require(
                np.array_equal(record[field], first[field]),
                f"RECOVAR {field} differs across particles",
            )
    return first, records


def _native_reference_panel(
    *,
    capture_dir: Path,
    recovar: dict[str, Any],
    full_image_size: int,
) -> tuple[np.ndarray, dict[str, Any], list[Any]]:
    operand_paths = sorted(capture_dir.glob("*.coarse-operands-v1.bin"))
    operands = [load_artifact(path) for path in operand_paths]
    by_key = {(item.part_id, item.class_one_based): item for item in operands}
    part_ids = sorted({item.part_id for item in operands})
    _require(len(operands) == 64 and len(by_key) == 64, "native operand panel differs from 64")
    _require(len(part_ids) == 16, "native particle denominator differs from 16")
    _require(
        set(by_key) == {(part_id, cls) for part_id in part_ids for cls in range(1, 5)},
        "native particle/class panel is incomplete",
    )

    scores = [
        load_coarse_score_capture(path)
        for path in sorted(capture_dir.glob("*.coarse-score-v1.bin"))
    ]
    score_by_part = {int(item.header[5]): item for item in scores}
    _require(set(score_by_part) == set(part_ids), "native score/operand particles differ")
    topology = {(int(item.header[11]), int(item.header[12])) for item in scores}
    _require(len(topology) == 1, "native direction/psi topology differs across particles")
    n_directions, n_psi = topology.pop()
    _require(
        n_directions * n_psi == recovar["n_rotations"],
        "native/RECOVAR rotation counts differ",
    )

    references = []
    maximum_euler_error = 0.0
    first_part = part_ids[0]
    for class_one_based in range(1, 5):
        reference_artifact = by_key[(first_part, class_one_based)]
        for part_id in part_ids[1:]:
            candidate = by_key[(part_id, class_one_based)]
            for field in (
                "rotation_keys",
                "euler_matrices",
                "reference_real",
                "reference_imag",
            ):
                _require(
                    np.array_equal(getattr(candidate, field), getattr(reference_artifact, field)),
                    f"native class {class_one_based} {field} differs across particles",
                )
        order = _native_rotation_order(
            reference_artifact,
            n_directions=n_directions,
            n_psi=n_psi,
        )
        eulers = reference_artifact.euler_matrices[order].transpose(0, 2, 1)
        maximum_euler_error = max(
            maximum_euler_error,
            float(np.max(np.abs(eulers - recovar["rotations"]))),
        )
        native = relion_reference_on_recovar_window(
            reference_artifact.reference_real[order].astype(np.float64)
            + 1j * reference_artifact.reference_imag[order].astype(np.float64),
            recovar["window_indices"],
            full_image_size=full_image_size,
            current_size=recovar["current_size"],
        )
        references.append(np.asarray(native, dtype=np.complex64))
    _require(maximum_euler_error <= EULER_MAX_ABS_CEILING, "Euler mapping exceeds gate")
    return (
        np.stack(references),
        {
            "particle_count": len(part_ids),
            "class_count": 4,
            "n_directions": n_directions,
            "n_psi": n_psi,
            "n_rotations": n_directions * n_psi,
            "maximum_euler_abs": maximum_euler_error,
        },
        operands,
    )


def analyze(
    *,
    ppref_dir: Path,
    ppref_source_report_path: Path,
    operand_source_maps: list[Path],
    capture_dir: Path,
    significance_dir: Path,
    operand_validation_path: Path,
    repeatability_path: Path,
    full_image_size: int,
) -> dict[str, Any]:
    """Build the class-complete frozen-boundary report on a CUDA device."""

    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block,
    )

    _require(jax.default_backend() == "gpu", "PPref projection replay requires a GPU")
    _require(cuda_backproject.cuda_available(), "RECOVAR CUDA projector is unavailable")
    _require(full_image_size > 0 and full_image_size % 2 == 0, "invalid full image size")
    ppref_dir = ppref_dir.resolve()
    capture_dir = capture_dir.resolve()
    significance_dir = significance_dir.resolve()

    source_report = _load_json(
        ppref_source_report_path,
        schema=PPREF_SOURCE_SCHEMA,
        status="complete",
    )
    _require(
        source_report.get("classification") == "map_to_ppref_matches_native",
        "map-to-PPref source gate did not pass",
    )
    _require(
        Path(source_report["native_validation"]["capture_dir"]).resolve() == ppref_dir,
        "source gate names another native PPref directory",
    )
    source_map_records = source_report["source_maps"]
    _require(len(source_map_records) == 4, "PPref source map panel differs from four")
    ppref_source_maps = [Path(record["path"]) for record in source_map_records]
    for path, record in zip(ppref_source_maps, source_map_records, strict=True):
        _require(_sha256(path) == record["sha256"], f"PPref source map hash differs: {path}")
    source_map_equivalence = compare_source_map_panels(
        ppref_source_maps,
        operand_source_maps,
    )

    operand_validation = _load_json(
        operand_validation_path,
        schema=OPERAND_VALIDATION_SCHEMA,
        status="pass",
    )
    repeatability = _load_json(
        repeatability_path,
        schema=REPEATABILITY_SCHEMA,
        status="pass",
    )
    _require(
        Path(operand_validation["directory"]).resolve() == capture_dir,
        "operand validation names another capture directory",
    )
    admitted_roots = {
        Path(record["run_root"]).resolve() for record in repeatability["repeats"]
    }
    _require(capture_dir.parent in admitted_roots, "operand capture is not repeatability-admitted")

    recovar, recovar_records = _common_recovar_panel(significance_dir)
    native_references, native_topology, native_operands = _native_reference_panel(
        capture_dir=capture_dir,
        recovar=recovar,
        full_image_size=full_image_size,
    )
    recovar_references = np.asarray(recovar["references"], dtype=np.complex64)

    pprefs = []
    ppref_metadata = []
    source_ppref_by_model = {
        int(record["model_zero_based"]): record for record in source_report["classes"]
    }
    _require(set(source_ppref_by_model) == set(range(4)), "source PPref class panel differs")
    for model in range(4):
        path = ppref_dir / f"ppref_iter001_rank000_model{model:03d}.bin"
        values, metadata = load_ppref(path)
        source_record = source_ppref_by_model[model]["native"]
        _require(Path(source_record["path"]).resolve() == path, "source PPref path differs")
        _require(_sha256(path) == source_record["sha256"], "source PPref hash differs")
        pprefs.append(values)
        ppref_metadata.append(metadata)
    for metadata in ppref_metadata[1:]:
        for field in ("current_size", "shape_zyx", "r_max", "padding_factor"):
            _require(metadata[field] == ppref_metadata[0][field], f"PPref {field} differs")

    texture_references = []
    for ppref in pprefs:
        projected, _ = compute_relion_projector_projections_block(
            jnp.asarray(ppref),
            jnp.asarray(recovar["rotations"], dtype=jnp.float32),
            (full_image_size, full_image_size),
            r_max=int(ppref_metadata[0]["r_max"]),
            padding_factor=int(ppref_metadata[0]["padding_factor"]),
            return_abs2=False,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=int(recovar["current_size"]),
            pixel_indices=jnp.asarray(recovar["window_indices"], dtype=jnp.int32),
            relion_texture_interp=True,
        )
        texture_references.append(
            np.asarray(jax.block_until_ready(projected), dtype=np.complex64)
        )
    texture_references = np.stack(texture_references)

    class_records = []
    for model in range(4):
        class_records.append(
            {
                "model_zero_based": model,
                "comparisons": {
                    "recovar_texture_of_native_ppref_vs_native_capture": _array_comparison(
                        texture_references[model], native_references[model]
                    ),
                    "recovar_texture_of_native_ppref_vs_recovar_capture": _array_comparison(
                        texture_references[model], recovar_references[model]
                    ),
                    "native_capture_vs_recovar_capture": _array_comparison(
                        native_references[model], recovar_references[model]
                    ),
                },
            }
        )

    def maximum_relative_l2(field: str) -> float:
        return max(float(record["comparisons"][field]["relative_l2"]) for record in class_records)

    texture_vs_native_l2 = maximum_relative_l2(
        "recovar_texture_of_native_ppref_vs_native_capture"
    )
    texture_vs_recovar_l2 = maximum_relative_l2(
        "recovar_texture_of_native_ppref_vs_recovar_capture"
    )
    native_vs_recovar_l2 = maximum_relative_l2("native_capture_vs_recovar_capture")
    classification = classify(
        texture_vs_native_l2=texture_vs_native_l2,
        texture_vs_recovar_l2=texture_vs_recovar_l2,
        native_vs_recovar_l2=native_vs_recovar_l2,
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification": classification,
        "classification_ready": True,
        "scientific_scope": (
            "EMPIAR-10076 shared-200 K=4 iteration-1 coarse projected references; "
            "four classes, 576 rotations, 16 independently captured particles"
        ),
        "metric_policy": "complex64 direct relative L2 and exact panel joins; no correlation",
        "fixed_gates": {
            "match_relative_l2_ceiling": MATCH_RELATIVE_L2_CEILING,
            "material_relative_l2_floor": MATERIAL_RELATIVE_L2_FLOOR,
            "separation_ratio_floor": SEPARATION_RATIO_FLOOR,
            "euler_max_abs_ceiling": EULER_MAX_ABS_CEILING,
        },
        "summary": {
            "class_count": 4,
            "particle_count": 16,
            "rotation_count": int(recovar["n_rotations"]),
            "projection_pixel_count": int(recovar["window_indices"].size),
            "recovar_texture_of_native_ppref_vs_native_capture_max_relative_l2": (
                texture_vs_native_l2
            ),
            "recovar_texture_of_native_ppref_vs_recovar_capture_max_relative_l2": (
                texture_vs_recovar_l2
            ),
            "native_capture_vs_recovar_capture_max_relative_l2": native_vs_recovar_l2,
        },
        "identity": {
            "full_image_size": full_image_size,
            "projection_current_size": int(recovar["current_size"]),
            "ppref_current_size": int(ppref_metadata[0]["current_size"]),
            "ppref_shape_zyx": ppref_metadata[0]["shape_zyx"],
            "ppref_r_max": int(ppref_metadata[0]["r_max"]),
            "ppref_padding_factor": float(ppref_metadata[0]["padding_factor"]),
            "native_topology": native_topology,
        },
        "source_map_equivalence": source_map_equivalence,
        "classes": class_records,
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_library": os.environ.get("RECOVAR_CUDA_LIB"),
        },
        "inputs": {
            "ppref_directory": str(ppref_dir),
            "ppref_source_report": str(ppref_source_report_path.resolve()),
            "ppref_source_report_sha256": _sha256(ppref_source_report_path),
            "capture_directory": str(capture_dir),
            "significance_directory": str(significance_dir),
            "operand_validation": str(operand_validation_path.resolve()),
            "operand_validation_sha256": _sha256(operand_validation_path),
            "repeatability_gate": str(repeatability_path.resolve()),
            "repeatability_gate_sha256": _sha256(repeatability_path),
            "recovar_captures": [
                {"path": str(record["path"]), "sha256": record["sha256"]}
                for record in recovar_records
            ],
            "native_operand_sha256": {
                str(item.path.resolve()): item.sha256 for item in native_operands
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppref-dir", type=Path, required=True)
    parser.add_argument("--ppref-source-report", type=Path, required=True)
    parser.add_argument("--operand-source-map", type=Path, action="append", required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--significance-dir", type=Path, required=True)
    parser.add_argument("--operand-validation", type=Path, required=True)
    parser.add_argument("--repeatability-gate", type=Path, required=True)
    parser.add_argument("--full-image-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    report = analyze(
        ppref_dir=args.ppref_dir,
        ppref_source_report_path=args.ppref_source_report,
        operand_source_maps=args.operand_source_map,
        capture_dir=args.capture_dir,
        significance_dir=args.significance_dir,
        operand_validation_path=args.operand_validation,
        repeatability_path=args.repeatability_gate,
        full_image_size=args.full_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
