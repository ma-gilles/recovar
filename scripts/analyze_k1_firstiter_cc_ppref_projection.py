#!/usr/bin/env python3
"""Split a fresh firstiter-CC reference mismatch at PPref vs projection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_fine_ppref_source_boundary import _array_metrics
from scripts.analyze_em_k1_live_reference_counterfactual import (
    relion_reference_on_recovar_window,
)
from scripts.analyze_k1_firstiter_cc_native_operands import _rotation_map
from scripts.analyze_k1_native_cc_translation_tie import _pass_field, _sha256
from scripts.parse_relion_dump_dir import parse_dump_dir


SCHEMA = "recovar.em.k1_firstiter_cc_ppref_projection.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def classify(*, frozen_texture_l2: float, captured_cross_l2: float) -> str:
    _require(frozen_texture_l2 >= 0.0 and captured_cross_l2 >= 0.0, "negative error")
    if frozen_texture_l2 <= 1e-7 and captured_cross_l2 > 100.0 * max(
        frozen_texture_l2, np.finfo(np.float64).tiny
    ):
        return "projected_reference_difference_enters_before_texture_projection"
    if frozen_texture_l2 >= 0.1 * captured_cross_l2:
        return "texture_projection_boundary_remains_open"
    return "ppref_and_texture_boundaries_are_mixed"


def analyze(
    *,
    relion_dump_dir: Path,
    recovar_capture: Path,
    physical_image_size: int,
    rotation_gate_deg: float,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.projection import (
        compute_relion_projector_projections_block,
    )

    _require(jax.default_backend() == "gpu", "PPref projection replay requires a GPU")
    _require(cuda_backproject.cuda_available(), "RECOVAR CUDA projector is unavailable")
    payload = parse_dump_dir(relion_dump_dir)
    raw_score = _pass_field(
        payload, "firstiter_cc_exp_Mweight_raw_preonehot"
    ).reshape(-1)
    native_rotation_rows = _pass_field(payload, "firstiter_cc_raw_rot_idx").reshape(-1)
    candidate_count = int(raw_score.size)
    native_image_size = int(_pass_field(payload, "image_size").reshape(-1)[0])
    _require(native_image_size > 0, "invalid native image size")
    native_reference = (
        _pass_field(payload, "fine_ref_real")
        + np.complex128(1j) * _pass_field(payload, "fine_ref_imag")
    ).reshape(candidate_count, native_image_size)
    native_rotations = np.transpose(
        _pass_field(payload, "fine_eulers").reshape(-1, 3, 3),
        (0, 2, 1),
    )
    ppref_dims = _pass_field(payload, "ppref_dims").reshape(-1).astype(np.int64)
    _require(ppref_dims.size == 7, "native PPref dimensions changed")
    ppref_shape = (int(ppref_dims[2]), int(ppref_dims[1]), int(ppref_dims[0]))
    frozen_ppref = (
        _pass_field(payload, "ppref_real")
        + np.complex128(1j) * _pass_field(payload, "ppref_imag")
    ).astype(np.complex64).reshape(ppref_shape)
    padding_factor = int(_pass_field(payload, "ppref_padding_factor").reshape(-1)[0])
    r_max = int(ppref_dims[6])

    with np.load(recovar_capture, allow_pickle=False) as archive:
        recovar_rotations = np.asarray(archive["rotations"], dtype=np.float32)
        recovar_reference = np.asarray(archive["raw_operand_proj_half"], dtype=np.complex64)
        window_indices = np.asarray(archive["window_indices"], dtype=np.int32)
        current_size = int(archive["current_size"])
        source_row = int(archive["original_index"])

    rotation_map, rotation_errors = _rotation_map(native_rotations, recovar_rotations)
    _require(
        float(np.max(rotation_errors, initial=0.0)) <= rotation_gate_deg,
        "rotation mapping exceeds gate",
    )
    native_reference_by_rotation = []
    for native_rotation_row in range(native_rotations.shape[0]):
        rows = np.flatnonzero(native_rotation_rows == native_rotation_row)
        _require(rows.size > 0, f"native rotation {native_rotation_row} has no candidates")
        reference_rows = native_reference[rows].astype(np.complex64)
        _require(
            all(np.array_equal(reference_rows[0], row) for row in reference_rows[1:]),
            f"native reference changes across translations for rotation {native_rotation_row}",
        )
        native_reference_by_rotation.append(reference_rows[0])
    native_reference_by_rotation = relion_reference_on_recovar_window(
        np.asarray(native_reference_by_rotation),
        window_indices,
        full_image_size=physical_image_size,
        current_size=current_size,
    ).astype(np.complex64)
    recovar_reference_by_rotation = recovar_reference[rotation_map]

    frozen_texture, _ = compute_relion_projector_projections_block(
        jnp.asarray(frozen_ppref),
        jnp.asarray(recovar_rotations[rotation_map]),
        (physical_image_size, physical_image_size),
        r_max=r_max,
        padding_factor=padding_factor,
        return_abs2=False,
        centered_rows=True,
        dense_scale=True,
        projector_output_size=current_size,
        pixel_indices=jnp.asarray(window_indices),
        relion_texture_interp=True,
    )
    frozen_texture = np.asarray(jax.block_until_ready(frozen_texture), dtype=np.complex64)
    comparisons = {
        "frozen_relion_ppref_texture_vs_native_reference": _array_metrics(
            frozen_texture, native_reference_by_rotation
        ),
        "frozen_relion_ppref_texture_vs_recovar_reference": _array_metrics(
            frozen_texture, recovar_reference_by_rotation
        ),
        "native_reference_vs_recovar_reference": _array_metrics(
            native_reference_by_rotation, recovar_reference_by_rotation
        ),
    }
    frozen_l2 = comparisons[
        "frozen_relion_ppref_texture_vs_native_reference"
    ]["relative_l2_lhs_minus_rhs_over_rhs"]
    cross_l2 = comparisons[
        "native_reference_vs_recovar_reference"
    ]["relative_l2_lhs_minus_rhs_over_rhs"]
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification": classify(
            frozen_texture_l2=float(frozen_l2),
            captured_cross_l2=float(cross_l2),
        ),
        "metric_policy": "bitwise equality and direct relative L2; no correlation",
        "identity": {
            "source_row_zero_based": source_row,
            "candidate_count": candidate_count,
            "rotation_count": int(native_rotations.shape[0]),
            "current_size": current_size,
            "ppref_shape_zyx": list(ppref_shape),
            "ppref_dims_xyz_and_origins_rmax": ppref_dims.tolist(),
            "padding_factor": padding_factor,
            "r_max": r_max,
            "rotation_map_max_error_deg": float(np.max(rotation_errors, initial=0.0)),
        },
        "comparisons": comparisons,
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_library": os.environ.get("RECOVAR_CUDA_LIB"),
        },
        "artifacts": {
            "relion_dump_dir": str(relion_dump_dir.resolve()),
            "relion_dump_sha256": {
                path.name: _sha256(path) for path in sorted(relion_dump_dir.glob("*.bin"))
            },
            "recovar_capture": str(recovar_capture.resolve()),
            "recovar_capture_sha256": _sha256(recovar_capture),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-dump-dir", required=True, type=Path)
    parser.add_argument("--recovar-capture", required=True, type=Path)
    parser.add_argument("--physical-image-size", required=True, type=int)
    parser.add_argument("--rotation-gate-deg", type=float, default=1e-3)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        relion_dump_dir=args.relion_dump_dir,
        recovar_capture=args.recovar_capture,
        physical_image_size=args.physical_image_size,
        rotation_gate_deg=args.rotation_gate_deg,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
