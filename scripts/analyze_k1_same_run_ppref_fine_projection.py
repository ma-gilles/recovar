#!/usr/bin/env python3
"""Join same-run native RELION PPref and fine projected-reference operands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.projection import (
    compute_relion_projector_projections_block,
)
from scripts.analyze_k1_exact_ppref_fine_boundary import _load_ppref
from scripts.analyze_k1_fine_operand_tuple import _sass_tree_raw_diff2
from scripts.compare_k4_relion_recovar_fine_operands import _infer_current_size, _metric
from scripts.validate_relion_fine_operand_capture import (
    load_fine_operand_capture,
    validate_capture,
)


SCHEMA = "recovar.em.k1_same_run_ppref_fine_projection.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_pixel_indices(pixels: np.ndarray, physical_image_size: int) -> np.ndarray:
    """Map RELION signed-y fine pixels into RECOVAR centered physical rows."""

    x = np.asarray(pixels["x"], dtype=np.int64)
    y = np.asarray(pixels["y"], dtype=np.int64)
    half_width = int(physical_image_size) // 2 + 1
    _require(np.all((x >= 0) & (x < half_width)), "native x coordinate is outside the half image")
    _require(
        np.all((y >= -int(physical_image_size) // 2) & (y <= int(physical_image_size) // 2)),
        "native y coordinate is outside the physical image",
    )
    indices = (y + int(physical_image_size) // 2) * half_width + x
    _require(np.unique(indices).size == indices.size, "native fine pixels are not unique")
    return indices.astype(np.int64, copy=False)


def _classification(expected_arm: dict[str, Any], alternatives: dict[str, dict[str, Any]]) -> str:
    projection = expected_arm["projected_reference"]
    score = expected_arm["raw_diff2"]
    if projection["exact_equal"] and score["exact_equal"]:
        return "same_run_ppref_and_texture_projection_are_bit_exact"
    if score["exact_equal"]:
        return "same_run_texture_projection_score_is_exact_despite_pixel_rounding"
    expected_l2 = float(projection["relative_l2_over_relion"])
    alternative_l2 = min(
        float(arm["projected_reference"]["relative_l2_over_relion"])
        for arm in alternatives.values()
    )
    if expected_l2 < alternative_l2:
        return "same_run_texture_transpose_is_closest_but_not_exact"
    return "same_run_ppref_does_not_select_expected_projection_semantics"


def analyze(
    *,
    ppref_path: Path,
    fine_operand_path: Path,
    candidate_index: int,
    physical_image_size: int,
) -> dict[str, Any]:
    _require(jax.default_backend() == "gpu", "same-run PPref projection audit requires a GPU")
    ppref, ppref_metadata = _load_ppref(ppref_path)
    capture = load_fine_operand_capture(fine_operand_path)
    validation = validate_capture(capture)
    _require(0 <= candidate_index < capture.candidates.size, "candidate index is outside capture")
    _require(int(ppref_metadata["iteration"]) == capture.iteration, "iteration differs")
    _require(float(ppref_metadata["padding_factor"]) == 2.0, "padding factor differs")

    candidate = capture.candidates[candidate_index]
    pixels = capture.pixels.reshape(capture.candidates.size, capture.image_size)[candidate_index]
    current_size = _infer_current_size(capture.image_size)
    _require(int(ppref_metadata["current_size"]) == current_size, "current size differs")
    _require(int(ppref_metadata["r_max"]) == current_size // 2, "projector radius differs")
    physical_shape = (int(physical_image_size), int(physical_image_size))
    physical_indices = _native_pixel_indices(pixels, int(physical_image_size))

    native_reference = (
        np.asarray(pixels["reference_real"], dtype=np.float32)
        + np.complex64(1j) * np.asarray(pixels["reference_imag"], dtype=np.float32)
    ).astype(np.complex64)
    native_shifted = (
        np.asarray(pixels["shifted_real"], dtype=np.float32)
        + np.complex64(1j) * np.asarray(pixels["shifted_imag"], dtype=np.float32)
    ).astype(np.complex64)
    native_corr = np.asarray(pixels["corr"], dtype=np.float32)
    native_sum = np.float32(candidate["sum_init"])
    native_raw = np.float32(candidate["production_raw_diff2"])
    native_matrix = np.asarray(candidate["matrix"], dtype=np.float32).reshape(3, 3)

    def project(*, transpose: bool, texture: bool) -> np.ndarray:
        matrix = native_matrix.T if transpose else native_matrix
        projected, _ = compute_relion_projector_projections_block(
            jnp.asarray(ppref),
            jnp.asarray(matrix[None], dtype=jnp.float32),
            physical_shape,
            r_max=int(ppref_metadata["r_max"]),
            padding_factor=int(ppref_metadata["padding_factor"]),
            return_abs2=False,
            centered_rows=True,
            dense_scale=False,
            projector_output_size=current_size,
            relion_texture_interp=texture,
        )
        full = np.asarray(jax.block_until_ready(projected), dtype=np.complex64)[0]
        return full[physical_indices]

    arms: dict[str, dict[str, Any]] = {}
    for name, transpose, texture in (
        ("texture_transpose", True, True),
        ("texture_native_matrix", False, True),
        ("manual_transpose", True, False),
        ("manual_native_matrix", False, False),
    ):
        reference = project(transpose=transpose, texture=texture)
        replay_raw = _sass_tree_raw_diff2(
            reference,
            native_shifted,
            native_corr,
            native_sum,
        )[0]
        arms[name] = {
            "projected_reference": _metric(native_reference, reference),
            "raw_diff2": _metric(np.asarray([native_raw]), np.asarray([replay_raw])),
            "replay_raw_diff2": float(replay_raw),
        }

    expected = arms["texture_transpose"]
    alternatives = {name: arm for name, arm in arms.items() if name != "texture_transpose"}
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification": _classification(expected, alternatives),
        "identity": {
            "stack_index_one_based": capture.stack_index,
            "particle_id": capture.particle_id,
            "physical_iteration": capture.iteration,
            "class_one_based": capture.class_one_based,
            "candidate_index": int(candidate_index),
            "rotation_local": int(candidate["rotation_local"]),
            "translation_id": int(candidate["translation_id"]),
            "ppref_rank": int(ppref_metadata["rank"]),
        },
        "geometry": {
            "physical_image_size": int(physical_image_size),
            "current_size": current_size,
            "pixel_count": capture.image_size,
            "projector_r_max": int(ppref_metadata["r_max"]),
            "padding_factor": int(ppref_metadata["padding_factor"]),
            "expected_rotation_input": "native captured matrix transpose",
            "output_rows": "physical centered y rows gathered by native signed (x,y)",
        },
        "native_production_raw_diff2": float(native_raw),
        "arms": arms,
        "native_capture_validation": validation,
        "artifacts": {
            "ppref": str(ppref_path.resolve()),
            "ppref_sha256": _sha256(ppref_path),
            "fine_operand": str(fine_operand_path.resolve()),
            "fine_operand_sha256": _sha256(fine_operand_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppref", type=Path, required=True)
    parser.add_argument("--fine-operand", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        ppref_path=args.ppref,
        fine_operand_path=args.fine_operand,
        candidate_index=args.candidate_index,
        physical_image_size=args.physical_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
