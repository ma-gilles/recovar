#!/usr/bin/env python3
"""Localize a K=1 fine projection difference across map, PPref, and texture stages.

This GPU diagnostic binds four already sealed inputs:

1. the two engines' iteration-start real-space maps,
2. RELION's frozen ``PPref`` and fine-reference capture,
3. RECOVAR's captured fine projections, and
4. the prior top-pair operand-attribution report.

It rebuilds both map-derived ``PPref`` arrays through the exact RELION binding,
then projects them through RECOVAR's production CUDA texture path.  The report
uses bitwise equality and direct array errors for boundary closure, plus
shellwise map FSC/FSC-AUC as the primary map-quality diagnostic.  It does not
use correlation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_em_k1_fine_top_pair_operands import (
    SCHEMA as OPERAND_REPORT_SCHEMA,
)
from scripts.analyze_em_k1_fine_top_pair_operands import (
    _require,
    _sha256,
)
from scripts.analyze_em_k1_live_reference_counterfactual import (
    relion_reference_on_recovar_window,
)
from scripts.audit_em_k1_membership_capture_inertness import _metrics as _map_metrics
from scripts.compare_relion_recovar_estep_dump import (
    _nearest_rotation_rows_by_matrix,
)

SCHEMA = "em-k1-fine-ppref-source-boundary-v1"
REBUILD_RELATIVE_L2_GATE = 1.0e-7
SOURCE_SEPARATION_RATIO_GATE = 100.0


def _array_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    _require(lhs.shape == rhs.shape, f"array shapes differ: {lhs.shape} != {rhs.shape}")
    denominator = float(np.linalg.norm(rhs))
    _require(denominator > 0.0, "array comparison target has zero norm")
    absolute = np.abs(lhs - rhs)
    return {
        "shape": list(lhs.shape),
        "array_equal": bool(np.array_equal(lhs, rhs)),
        "relative_l2_lhs_minus_rhs_over_rhs": float(np.linalg.norm(lhs - rhs) / denominator),
        "max_abs": float(np.max(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "mean_abs": float(np.mean(absolute)),
    }


def classify_source_boundary(
    *,
    frozen_relion_texture_exact: bool,
    relion_map_rebuild_relative_l2: float,
    recovar_map_replay_relative_l2: float,
    cross_engine_projection_relative_l2: float,
    map_states_equal: bool,
) -> str:
    """Classify the first open boundary using fixed closure and separation gates."""

    closures = (
        float(relion_map_rebuild_relative_l2),
        float(recovar_map_replay_relative_l2),
    )
    cross = float(cross_engine_projection_relative_l2)
    _require(all(value >= 0.0 for value in closures) and cross >= 0.0, "array errors must be non-negative")
    if (
        frozen_relion_texture_exact
        and max(closures) <= REBUILD_RELATIVE_L2_GATE
        and cross > SOURCE_SEPARATION_RATIO_GATE * max(max(closures), np.finfo(np.float64).tiny)
        and not map_states_equal
    ):
        return "fine_projection_difference_is_iteration_start_map_state"
    if not frozen_relion_texture_exact:
        return "texture_projection_boundary_remains_open"
    if max(closures) > REBUILD_RELATIVE_L2_GATE:
        return "map_to_ppref_or_replay_boundary_remains_open"
    return "fine_projection_source_boundary_is_mixed"


def _read_relion_map(path: Path) -> np.ndarray:
    from recovar.utils.helpers import relion_volume_to_recovar

    with mrcfile.open(path, permissive=False) as handle:
        raw = np.asarray(handle.data, dtype=np.float32).copy()
    _require(raw.ndim == 3 and len(set(raw.shape)) == 1, f"RELION map is not cubic: {path}")
    _require(np.all(np.isfinite(raw)), f"RELION map contains non-finite values: {path}")
    return np.asarray(relion_volume_to_recovar(raw), dtype=np.float32)


def _read_recovar_map(path: Path) -> np.ndarray:
    from recovar.utils.helpers import load_mrc

    volume = np.asarray(load_mrc(path), dtype=np.float32)
    _require(volume.ndim == 3 and len(set(volume.shape)) == 1, f"RECOVAR map is not cubic: {path}")
    _require(np.all(np.isfinite(volume)), f"RECOVAR map contains non-finite values: {path}")
    return volume


def _module_sha256(module: Any) -> tuple[str, str]:
    path = Path(module.__file__).resolve()
    return str(path), _sha256(path)


def _load_inputs(
    *,
    relion_npz: Path,
    recovar_npz: Path,
    ordered_top_keys: tuple[tuple[int, int], tuple[int, int]],
    full_image_size: int,
    expected_current_size: int,
) -> dict[str, Any]:
    with np.load(recovar_npz, allow_pickle=False) as recovar_payload:
        required_recovar = {
            "original_index",
            "current_size",
            "rotations",
            "candidate_mask",
            "proj_half",
            "window_indices",
        }
        _require(
            required_recovar <= set(recovar_payload.files),
            f"missing RECOVAR fields: {sorted(required_recovar - set(recovar_payload.files))}",
        )
        recovar_original_index = int(np.asarray(recovar_payload["original_index"]).item())
        recovar_current_size = int(np.asarray(recovar_payload["current_size"]).item())
        recovar_rotations = np.asarray(recovar_payload["rotations"], dtype=np.float32)
        recovar_candidate_mask = np.asarray(recovar_payload["candidate_mask"], dtype=bool)
        recovar_references = np.asarray(recovar_payload["proj_half"], dtype=np.complex64)
        window_indices = np.asarray(recovar_payload["window_indices"], dtype=np.int32)
    _require(recovar_current_size == expected_current_size, "RECOVAR current-size mismatch")
    top_rotation_rows = np.asarray([key[0] for key in ordered_top_keys], dtype=np.int64)
    for rotation_row, translation_row in ordered_top_keys:
        _require(
            bool(recovar_candidate_mask[rotation_row, translation_row]),
            f"RECOVAR top key {(rotation_row, translation_row)} is outside candidate support",
        )
    recovar_top_references = recovar_references[top_rotation_rows]

    with np.load(relion_npz, allow_pickle=False) as relion_payload:
        required_relion = {
            "pass1_acc_stack_index",
            "pass1_acc_rot_idx",
            "pass1_acc_trans_idx",
            "pass1_class0_fine_eulers",
            "pass1_class0_fine_ref_real",
            "pass1_class0_fine_ref_imag",
            "pass1_class0_ppref_dims",
            "pass1_class0_ppref_padding_factor",
            "pass1_class0_ppref_real",
            "pass1_class0_ppref_imag",
            "pass1_img0_exp_current_image_size",
        }
        _require(
            required_relion <= set(relion_payload.files),
            f"missing RELION fields: {sorted(required_relion - set(relion_payload.files))}",
        )
        relion_stack_index = int(np.asarray(relion_payload["pass1_acc_stack_index"]).item())
        relion_current_size = int(np.asarray(relion_payload["pass1_img0_exp_current_image_size"]).item())
        _require(relion_current_size == expected_current_size, "RELION current-size mismatch")
        relion_rotation_rows = np.asarray(relion_payload["pass1_acc_rot_idx"], dtype=np.int64).reshape(-1)
        relion_translation_rows = np.asarray(relion_payload["pass1_acc_trans_idx"], dtype=np.int64).reshape(-1)
        relion_rotation_matrices = np.asarray(
            relion_payload["pass1_class0_fine_eulers"],
            dtype=np.float64,
        ).reshape(-1, 3, 3)
        ppref_dims = np.asarray(relion_payload["pass1_class0_ppref_dims"], dtype=np.int64).reshape(-1)
        _require(ppref_dims.size == 7, "RELION PPref dimensions must have seven entries")
        ppref_shape = (int(ppref_dims[2]), int(ppref_dims[1]), int(ppref_dims[0]))
        frozen_ppref = (
            np.asarray(relion_payload["pass1_class0_ppref_real"], dtype=np.float32)
            + 1j * np.asarray(relion_payload["pass1_class0_ppref_imag"], dtype=np.float32)
        ).astype(np.complex64).reshape(ppref_shape)
        padding_factor = int(np.asarray(relion_payload["pass1_class0_ppref_padding_factor"]).item())
        r_max = int(ppref_dims[6])
        n_current_pixels = expected_current_size * (expected_current_size // 2 + 1)
        relion_fine_references = (
            np.asarray(relion_payload["pass1_class0_fine_ref_real"], dtype=np.float64)
            + 1j * np.asarray(relion_payload["pass1_class0_fine_ref_imag"], dtype=np.float64)
        ).reshape(-1, n_current_pixels)

    nearest_rows, rotation_distances, rotation_orientation = _nearest_rotation_rows_by_matrix(
        relion_rotation_matrices,
        recovar_rotations,
    )
    _require(np.unique(nearest_rows).size == relion_rotation_matrices.shape[0], "rotation map is not one-to-one")
    relion_candidate_rows: list[int] = []
    for rotation_row, translation_row in ordered_top_keys:
        matching = np.flatnonzero(
            (nearest_rows[relion_rotation_rows] == rotation_row)
            & (relion_translation_rows == translation_row)
        )
        _require(matching.size == 1, f"RELION top key {(rotation_row, translation_row)} is not unique")
        relion_candidate_rows.append(int(matching[0]))
    relion_top_references = relion_reference_on_recovar_window(
        relion_fine_references[relion_candidate_rows],
        window_indices,
        full_image_size=full_image_size,
        current_size=expected_current_size,
    ).astype(np.complex64)
    return {
        "recovar_original_index": recovar_original_index,
        "relion_stack_index": relion_stack_index,
        "recovar_rotations": recovar_rotations[top_rotation_rows],
        "window_indices": window_indices,
        "recovar_top_references": recovar_top_references,
        "relion_top_references": relion_top_references,
        "frozen_relion_ppref": frozen_ppref,
        "padding_factor": padding_factor,
        "r_max": r_max,
        "ppref_dims": ppref_dims,
        "rotation_orientation": rotation_orientation,
        "rotation_median_frobenius": float(np.median(rotation_distances)),
        "rotation_max_frobenius": float(np.max(rotation_distances)),
    }


def build_report(
    *,
    relion_npz: Path,
    recovar_npz: Path,
    operand_report_json: Path,
    relion_map_mrc: Path,
    recovar_map_mrc: Path,
    expected_binding_sha256: str,
    expected_original_index: int,
    expected_current_size: int,
    full_image_size: int,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from recovar.cuda_backproject import cuda_available
    from recovar.em.dense_single_volume.helpers.projection import (
        _relion_projector_texture_enabled,
        compute_relion_projector_projections_block,
    )
    from recovar.em.initial_model.dense_adapter import (
        reference_to_relion_projector_half_maps,
    )
    from recovar.relion_bind import _relion_bind_core as relion_bind

    paths = [
        Path(path).resolve()
        for path in (
            relion_npz,
            recovar_npz,
            operand_report_json,
            relion_map_mrc,
            recovar_map_mrc,
        )
    ]
    relion_npz, recovar_npz, operand_report_json, relion_map_mrc, recovar_map_mrc = paths
    operand_report = json.loads(operand_report_json.read_text())
    _require(operand_report.get("schema") == OPERAND_REPORT_SCHEMA, "operand-report schema mismatch")
    _require(operand_report.get("status") == "pass", "operand report is not passing")
    _require(
        operand_report.get("classification") == "fine_winner_flip_is_projected_reference_determined",
        "operand report does not isolate the projected-reference boundary",
    )
    input_hashes = operand_report.get("input_artifacts", {})
    _require(input_hashes.get(str(relion_npz)) == _sha256(relion_npz), "RELION NPZ hash differs from operand report")
    _require(input_hashes.get(str(recovar_npz)) == _sha256(recovar_npz), "RECOVAR NPZ hash differs from operand report")
    ordered_top_keys = tuple(
        tuple(int(value) for value in key)
        for key in operand_report["identity"]["ordered_top_keys"]
    )
    _require(len(ordered_top_keys) == 2, "operand report must bind exactly two top keys")

    _require(jax.default_backend() == "gpu", "PPref source replay requires a GPU backend")
    _require(cuda_available(), "RECOVAR CUDA projector is unavailable")
    binding_path, binding_sha256 = _module_sha256(relion_bind)
    _require(binding_sha256 == expected_binding_sha256, "RELION binding hash mismatch")
    inputs = _load_inputs(
        relion_npz=relion_npz,
        recovar_npz=recovar_npz,
        ordered_top_keys=ordered_top_keys,
        full_image_size=full_image_size,
        expected_current_size=expected_current_size,
    )
    _require(inputs["recovar_original_index"] == expected_original_index, "RECOVAR particle mismatch")
    _require(inputs["relion_stack_index"] == expected_original_index + 1, "RELION stack-index mismatch")
    _require(inputs["padding_factor"] == 2, "source replay requires the explicit RELION padding factor 2")
    _require(
        _relion_projector_texture_enabled(
            jnp.asarray(inputs["frozen_relion_ppref"]),
            r_max=inputs["r_max"],
            padding_factor=inputs["padding_factor"],
            enabled=True,
        ),
        "production CUDA texture projector did not activate",
    )

    relion_map = _read_relion_map(relion_map_mrc)
    recovar_map = _read_recovar_map(recovar_map_mrc)
    _require(relion_map.shape == recovar_map.shape == (full_image_size,) * 3, "map geometry mismatch")
    rebuilt_relion_ppref, rebuilt_relion_r_max = reference_to_relion_projector_half_maps(
        relion_map[np.newaxis, ...],
        current_size=expected_current_size,
        padding_factor=inputs["padding_factor"],
    )
    rebuilt_recovar_ppref, rebuilt_recovar_r_max = reference_to_relion_projector_half_maps(
        recovar_map[np.newaxis, ...],
        current_size=expected_current_size,
        padding_factor=inputs["padding_factor"],
    )
    _require(
        rebuilt_relion_r_max == rebuilt_recovar_r_max == inputs["r_max"],
        "rebuilt PPref r_max differs from frozen capture",
    )
    rebuilt_relion_ppref = np.asarray(rebuilt_relion_ppref[0], dtype=np.complex64)
    rebuilt_recovar_ppref = np.asarray(rebuilt_recovar_ppref[0], dtype=np.complex64)

    def project(ppref: np.ndarray) -> np.ndarray:
        projection, _ = compute_relion_projector_projections_block(
            jnp.asarray(ppref),
            jnp.asarray(inputs["recovar_rotations"]),
            (full_image_size, full_image_size),
            r_max=inputs["r_max"],
            padding_factor=inputs["padding_factor"],
            return_abs2=False,
            centered_rows=True,
            dense_scale=True,
            projector_output_size=expected_current_size,
            pixel_indices=jnp.asarray(inputs["window_indices"]),
            relion_texture_interp=True,
        )
        return np.asarray(jax.block_until_ready(projection), dtype=np.complex64)

    frozen_relion_projection = project(inputs["frozen_relion_ppref"])
    rebuilt_relion_projection = project(rebuilt_relion_ppref)
    rebuilt_recovar_projection = project(rebuilt_recovar_ppref)
    comparisons = {
        "frozen_relion_ppref_texture_vs_captured_relion_projection": _array_metrics(
            frozen_relion_projection,
            inputs["relion_top_references"],
        ),
        "relion_map_rebuilt_ppref_vs_frozen_relion_ppref": _array_metrics(
            rebuilt_relion_ppref,
            inputs["frozen_relion_ppref"],
        ),
        "relion_map_rebuilt_projection_vs_captured_relion_projection": _array_metrics(
            rebuilt_relion_projection,
            inputs["relion_top_references"],
        ),
        "recovar_map_rebuilt_projection_vs_captured_recovar_projection": _array_metrics(
            rebuilt_recovar_projection,
            inputs["recovar_top_references"],
        ),
        "captured_relion_projection_vs_captured_recovar_projection": _array_metrics(
            inputs["relion_top_references"],
            inputs["recovar_top_references"],
        ),
        "frozen_relion_ppref_vs_recovar_map_rebuilt_ppref": _array_metrics(
            inputs["frozen_relion_ppref"],
            rebuilt_recovar_ppref,
        ),
    }
    frozen_texture = comparisons[
        "frozen_relion_ppref_texture_vs_captured_relion_projection"
    ]
    relion_rebuild = comparisons[
        "relion_map_rebuilt_ppref_vs_frozen_relion_ppref"
    ]
    recovar_replay = comparisons[
        "recovar_map_rebuilt_projection_vs_captured_recovar_projection"
    ]
    cross_projection = comparisons[
        "captured_relion_projection_vs_captured_recovar_projection"
    ]
    classification = classify_source_boundary(
        frozen_relion_texture_exact=bool(frozen_texture["array_equal"]),
        relion_map_rebuild_relative_l2=float(
            relion_rebuild["relative_l2_lhs_minus_rhs_over_rhs"]
        ),
        recovar_map_replay_relative_l2=float(
            recovar_replay["relative_l2_lhs_minus_rhs_over_rhs"]
        ),
        cross_engine_projection_relative_l2=float(
            cross_projection["relative_l2_lhs_minus_rhs_over_rhs"]
        ),
        map_states_equal=bool(np.array_equal(relion_map, recovar_map)),
    )
    _require(
        classification == "fine_projection_difference_is_iteration_start_map_state",
        f"PPref source boundary did not close: {classification}",
    )
    map_comparison = _map_metrics(relion_map, recovar_map)
    source_separation_ratio = float(
        cross_projection["relative_l2_lhs_minus_rhs_over_rhs"]
        / max(
            float(relion_rebuild["relative_l2_lhs_minus_rhs_over_rhs"]),
            float(recovar_replay["relative_l2_lhs_minus_rhs_over_rhs"]),
            np.finfo(np.float64).tiny,
        )
    )
    return {
        "schema": SCHEMA,
        "status": "pass",
        "classification": classification,
        "closed_boundaries": [
            "RECOVAR CUDA texture interpolation and coordinates",
            "RELION serialized map to PPref construction",
            "RECOVAR serialized map to PPref construction and texture replay",
        ],
        "first_open_boundary": "iteration-start half-map state entering physical iteration 3",
        "metric_policy": (
            "bitwise equality and direct array errors gate PPref/projection closure; "
            "shellwise FSC/FSC-AUC is the primary map-quality metric; no correlation"
        ),
        "identity": {
            "recovar_original_index_zero_based": inputs["recovar_original_index"],
            "relion_stack_index_one_based": inputs["relion_stack_index"],
            "ordered_top_keys": [list(key) for key in ordered_top_keys],
            "current_size": expected_current_size,
            "full_image_size": full_image_size,
            "ppref_dims_xyz_and_origins_rmax": inputs["ppref_dims"].tolist(),
            "ppref_padding_factor": inputs["padding_factor"],
            "ppref_r_max": inputs["r_max"],
        },
        "map_comparison_relion_iteration2_half1_vs_recovar_iteration1_half1": map_comparison,
        "comparisons": comparisons,
        "fixed_gates": {
            "rebuild_relative_l2_gate": REBUILD_RELATIVE_L2_GATE,
            "source_separation_ratio_gate": SOURCE_SEPARATION_RATIO_GATE,
            "observed_source_separation_ratio": source_separation_ratio,
            "frozen_relion_texture_requires_array_equal": True,
        },
        "rotation_matrix_match": {
            "orientation": inputs["rotation_orientation"],
            "median_frobenius": inputs["rotation_median_frobenius"],
            "max_frobenius": inputs["rotation_max_frobenius"],
        },
        "runtime_provenance": {
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "recovar_relion_projector_texture_interp": os.environ.get(
                "RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP"
            ),
            "relion_bind_module": binding_path,
            "relion_bind_module_sha256": binding_sha256,
        },
        "input_artifacts": {
            str(path): _sha256(path)
            for path in (
                relion_npz,
                recovar_npz,
                operand_report_json,
                relion_map_mrc,
                recovar_map_mrc,
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-npz", type=Path, required=True)
    parser.add_argument("--recovar-pass2-npz", type=Path, required=True)
    parser.add_argument("--operand-report-json", type=Path, required=True)
    parser.add_argument("--relion-map-mrc", type=Path, required=True)
    parser.add_argument("--recovar-map-mrc", type=Path, required=True)
    parser.add_argument("--expected-binding-sha256", required=True)
    parser.add_argument("--expected-original-index", type=int, required=True)
    parser.add_argument("--expected-current-size", type=int, required=True)
    parser.add_argument("--full-image-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_json.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    report = build_report(
        relion_npz=args.relion_npz,
        recovar_npz=args.recovar_pass2_npz,
        operand_report_json=args.operand_report_json,
        relion_map_mrc=args.relion_map_mrc,
        recovar_map_mrc=args.recovar_map_mrc,
        expected_binding_sha256=args.expected_binding_sha256,
        expected_original_index=args.expected_original_index,
        expected_current_size=args.expected_current_size,
        full_image_size=args.full_image_size,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
