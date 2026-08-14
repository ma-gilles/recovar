#!/usr/bin/env python3
"""Qualify fused RELION translation+fine-diff2 on one captured rotation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_translation_angles_f32,
)
from scripts.validate_relion_fine_operand_capture import load_fine_operand_capture
from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    _require(reference.shape == candidate.shape, "metric shapes differ")
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    denominator = float(np.linalg.norm(reference.astype(np.float64)))
    return {
        "shape": list(reference.shape),
        "exact_equal": bool(np.array_equal(reference, candidate)),
        "mismatch_count": int(np.count_nonzero(reference != candidate)),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "relative_l2_over_reference": (
            float(np.linalg.norm(delta) / denominator) if denominator else 0.0
        ),
    }


def analyze(
    *,
    fine_operand_path: Path,
    fine_score_path: Path,
    pass2_path: Path,
    recovar_global_rotation: int,
    physical_image_size: int,
) -> dict[str, Any]:
    operand = load_fine_operand_capture(fine_operand_path)
    score = load_fine_score_capture(fine_score_path)
    _require(operand.candidates.size == 1, "fine-operand capture must contain one tuple")
    target = operand.candidates[0]
    _require(operand.stack_index == score.stack_index, "native capture identities differ")

    required = {
        "current_size",
        "fine_translations",
        "oversampled_rot_indices",
        "raw_operand_proj_half",
        "raw_operand_corr_img_score",
        "raw_operand_half_weights",
        "raw_operand_relion_full_to_compact",
        "raw_operand_highres_xi2_half",
    }
    with np.load(pass2_path, allow_pickle=False) as archive:
        _require(required <= set(archive.files), "pass-2 capture misses fused-score inputs")
        recovar = {name: np.asarray(archive[name]) for name in required}

    rotation_rows = np.flatnonzero(
        np.asarray(recovar["oversampled_rot_indices"], dtype=np.int64)
        == int(recovar_global_rotation)
    )
    _require(rotation_rows.size == 1, "RECOVAR target rotation is not unique")
    rotation_row = int(rotation_rows[0])
    current_size = int(np.asarray(recovar["current_size"]).item())
    lookup = np.asarray(recovar["raw_operand_relion_full_to_compact"], dtype=np.int32)
    compact_pixel_count = int(recovar["raw_operand_proj_half"].shape[-1])
    _require(
        lookup.shape == (current_size * (current_size // 2 + 1),),
        "RELION full lookup and current size differ",
    )
    supported_full = np.flatnonzero(lookup >= 0)
    supported_compact = lookup[supported_full]
    _require(
        np.array_equal(np.sort(supported_compact), np.arange(compact_pixel_count)),
        "RELION full lookup does not cover compact pixels exactly once",
    )

    native_pixels = operand.pixels.reshape(1, operand.image_size)[0]
    native_image_full = (
        np.asarray(native_pixels["image_real"], dtype=np.float32)
        + np.complex64(1j) * np.asarray(native_pixels["image_imag"], dtype=np.float32)
    ).astype(np.complex64)
    n2 = np.float32(physical_image_size**2)
    image_compact = np.empty(compact_pixel_count, dtype=np.complex64)
    image_compact[supported_compact] = -native_image_full[supported_full] * n2

    reference = np.asarray(
        recovar["raw_operand_proj_half"][rotation_row : rotation_row + 1],
        dtype=np.complex64,
    )[None, :, :]
    weight = np.multiply(
        np.asarray(recovar["raw_operand_corr_img_score"], dtype=np.float32),
        np.asarray(recovar["raw_operand_half_weights"], dtype=np.float32),
        dtype=np.float32,
    )[None, :]
    translations = np.asarray(recovar["fine_translations"], dtype=np.float32)
    translation_angles = _relion_translation_angles_f32(
        translations,
        (physical_image_size, physical_image_size),
    )
    fused = cuda_backproject.relion_fine_diff2_fused_translate_rectangular_f32(
        jnp.asarray(reference),
        jnp.asarray(image_compact[None, :]),
        jnp.asarray(translation_angles),
        jnp.asarray(weight),
        jnp.asarray(lookup),
        current_size=current_size,
    )
    fused_lane_sum = np.asarray(jax.block_until_ready(fused), dtype=np.float32)[0, 0]
    recovar_highres = np.float32(
        np.asarray(recovar["raw_operand_highres_xi2_half"]).item()
    )
    native_highres = np.float32(target["sum_init"])
    fused_recovar_highres = np.add(
        fused_lane_sum,
        recovar_highres,
        dtype=np.float32,
    )
    fused_native_highres = np.add(
        fused_lane_sum,
        native_highres,
        dtype=np.float32,
    )

    active = (score.candidates["flags"] & ACTIVE) != 0
    same_rotation = active & (
        score.candidates["rotation_local"] == np.uint64(target["rotation_local"])
    )
    native_candidates = score.candidates[same_rotation]
    _require(native_candidates.size > 0, "native fine score has no matched rotation")
    native_translation = np.asarray(native_candidates["translation_id"], dtype=np.int64)
    _require(
        np.all(
            (native_translation >= 0)
            & (native_translation < fused_recovar_highres.size)
        ),
        "native translation ids are outside the fused result",
    )
    native_raw = np.asarray(native_candidates["raw_diff2"], dtype=np.float32)
    fused_raw_recovar_highres = fused_recovar_highres[native_translation]
    fused_raw_native_highres = fused_native_highres[native_translation]
    target_translation = int(target["translation_id"])
    target_native_raw = np.float32(target["production_raw_diff2"])
    target_fused_recovar_highres = np.float32(
        fused_recovar_highres[target_translation]
    )
    target_fused_native_highres = np.float32(
        fused_native_highres[target_translation]
    )
    return {
        "schema": "recovar.em.k1_fused_translate_tuple.v1",
        "status": "accepted",
        "device": str(jax.devices()[0]),
        "identity": {
            "stack_index_one_based": operand.stack_index,
            "native_particle_id": operand.particle_id,
            "native_rotation_local": int(target["rotation_local"]),
            "recovar_global_rotation": int(recovar_global_rotation),
            "recovar_rotation_row": rotation_row,
            "target_translation": target_translation,
        },
        "topology": {
            "current_size": current_size,
            "compact_pixel_count": compact_pixel_count,
            "translation_count": int(translations.shape[0]),
            "native_same_rotation_active_count": int(native_candidates.size),
            "cuda_block_size": 256,
            "cuda_translation_capacity": 7,
            "deployed_ref3d_job_chunk": 4,
        },
        "target": {
            "native_raw_diff2": float(target_native_raw),
            "fused_recovar_highres_raw_diff2": float(
                target_fused_recovar_highres
            ),
            "fused_native_highres_raw_diff2": float(target_fused_native_highres),
            "recovar_highres_exact_equal": bool(
                target_native_raw == target_fused_recovar_highres
            ),
            "native_highres_exact_equal": bool(
                target_native_raw == target_fused_native_highres
            ),
            "recovar_highres_float32_ulp_distance": int(
                abs(
                    int(target_native_raw.view(np.uint32))
                    - int(target_fused_recovar_highres.view(np.uint32))
                )
            ),
            "native_highres_float32_ulp_distance": int(
                abs(
                    int(target_native_raw.view(np.uint32))
                    - int(target_fused_native_highres.view(np.uint32))
                )
            ),
        },
        "highres_sum_init": {
            "native": float(native_highres),
            "recovar": float(recovar_highres),
            "absolute_delta": float(
                abs(np.float64(recovar_highres) - np.float64(native_highres))
            ),
            "exact_equal": bool(native_highres == recovar_highres),
        },
        "same_rotation_raw_diff2_recovar_highres": _metric(
            native_raw,
            fused_raw_recovar_highres,
        ),
        "same_rotation_raw_diff2_native_highres": _metric(
            native_raw,
            fused_raw_native_highres,
        ),
        "artifacts": {
            "fine_operand": str(fine_operand_path.resolve()),
            "fine_operand_sha256": _sha256(fine_operand_path),
            "fine_score": str(fine_score_path.resolve()),
            "fine_score_sha256": _sha256(fine_score_path),
            "recovar_pass2": str(pass2_path.resolve()),
            "recovar_pass2_sha256": _sha256(pass2_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine-operand", type=Path, required=True)
    parser.add_argument("--fine-score", type=Path, required=True)
    parser.add_argument("--pass2", type=Path, required=True)
    parser.add_argument("--recovar-global-rotation", type=int, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        fine_operand_path=args.fine_operand,
        fine_score_path=args.fine_score,
        pass2_path=args.pass2,
        recovar_global_rotation=args.recovar_global_rotation,
        physical_image_size=args.physical_image_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
