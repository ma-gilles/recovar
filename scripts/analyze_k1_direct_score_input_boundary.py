#!/usr/bin/env python3
"""Compare one native/RECOVAR fine-score image before and after translation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar import cuda_backproject
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_translation_angles_f32,
)
from scripts.analyze_k1_exact_ppref_operand_tuple import _float32_ulp_stats
from scripts.analyze_k1_fine_operand_tuple import _metric, _translation_alignment
from scripts.validate_relion_fine_operand_capture import (
    load_fine_operand_capture,
    validate_capture,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scatter_compact(
    compact: np.ndarray,
    *,
    supported_full: np.ndarray,
    supported_compact: np.ndarray,
    full_size: int,
) -> np.ndarray:
    result = np.zeros(full_size, dtype=np.complex64)
    result[supported_full] = np.asarray(compact, dtype=np.complex64)[supported_compact]
    return result


def _translate(
    image_compact: np.ndarray,
    angle: np.ndarray,
    window_indices: np.ndarray,
    physical_image_size: int,
) -> np.ndarray:
    translated = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(np.asarray(image_compact, dtype=np.complex64)[None, :]),
        jnp.asarray(np.asarray(angle, dtype=np.float32)[None, :]),
        jnp.asarray(window_indices, dtype=jnp.int32),
        (physical_image_size, physical_image_size),
    )
    return np.asarray(jax.block_until_ready(translated), dtype=np.complex64).reshape(-1)


def analyze(
    *,
    capture_path: Path,
    recovar_path: Path,
    physical_image_size: int,
) -> dict[str, object]:
    _require(jax.default_backend() == "gpu", "translation replay requires a GPU")
    _require(cuda_backproject.cuda_available(), str(cuda_backproject.cuda_unavailable_error()))
    capture = load_fine_operand_capture(capture_path)
    validation = validate_capture(capture)
    _require(capture.candidates.size == 1, "native capture must contain one tuple")
    candidate = capture.candidates[0]
    pixels = capture.pixels.reshape(1, capture.image_size)[0]

    with np.load(recovar_path, allow_pickle=False) as archive:
        recovar = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "current_size",
        "fine_translations",
        "window_indices",
        "direct_score_input",
        "shifted_corrected",
    }
    _require(required <= set(recovar), f"RECOVAR dump misses {sorted(required - set(recovar))}")
    direct_score_input = np.asarray(recovar["direct_score_input"], dtype=np.complex64)
    _require(direct_score_input.ndim == 1 and direct_score_input.size > 0, "direct score input is empty")

    current_size = int(np.asarray(recovar["current_size"]).item())
    expected_full_size = current_size * (current_size // 2 + 1)
    _require(capture.image_size == expected_full_size, "native current-size geometry differs")
    window_indices = np.asarray(recovar["window_indices"], dtype=np.int32)
    _require(direct_score_input.shape == window_indices.shape, "direct input/window topology differs")
    lookup = np.asarray(
        _relion_cuda_fine_full_to_compact_lookup(
            (physical_image_size, physical_image_size),
            current_size,
            window_indices,
        ),
        dtype=np.int32,
    )
    supported_full = np.flatnonzero(lookup >= 0)
    supported_compact = lookup[supported_full]
    _require(
        np.array_equal(np.sort(supported_compact), np.arange(window_indices.size)),
        "full/compact score lookup is incomplete",
    )

    translation_row, translation_error = _translation_alignment(
        candidate["translation"],
        recovar["fine_translations"],
        physical_image_size,
    )
    _require(translation_error <= 1.0e-6, "native/RECOVAR translation differs")
    angles = _relion_translation_angles_f32(
        recovar["fine_translations"],
        (physical_image_size, physical_image_size),
    )
    angle = np.asarray(angles[translation_row], dtype=np.float32)
    # The native fine-operand capture stores the angle already passed to
    # translatePixel, not the source translation in pixels.
    native_angle = np.asarray(candidate["translation"][:2], dtype=np.float32)

    native_image = (
        np.asarray(pixels["image_real"], dtype=np.float32)
        + np.complex64(1j) * np.asarray(pixels["image_imag"], dtype=np.float32)
    ).astype(np.complex64)
    native_shifted = (
        np.asarray(pixels["shifted_real"], dtype=np.float32)
        + np.complex64(1j) * np.asarray(pixels["shifted_imag"], dtype=np.float32)
    ).astype(np.complex64)

    n2 = np.float32(physical_image_size**2)
    recovar_image = _scatter_compact(
        -direct_score_input / n2,
        supported_full=supported_full,
        supported_compact=supported_compact,
        full_size=capture.image_size,
    )
    recovar_shifted = _scatter_compact(
        -np.asarray(recovar["shifted_corrected"][translation_row], dtype=np.complex64) / n2,
        supported_full=supported_full,
        supported_compact=supported_compact,
        full_size=capture.image_size,
    )

    native_compact = np.zeros(window_indices.size, dtype=np.complex64)
    native_compact[supported_compact] = native_image[supported_full]
    native_replay_compact = _translate(
        native_compact,
        angle,
        window_indices,
        physical_image_size,
    )
    recovar_replay_compact = _translate(
        direct_score_input,
        angle,
        window_indices,
        physical_image_size,
    )
    native_replay = _scatter_compact(
        native_replay_compact,
        supported_full=supported_full,
        supported_compact=supported_compact,
        full_size=capture.image_size,
    )
    recovar_replay = _scatter_compact(
        -recovar_replay_compact / n2,
        supported_full=supported_full,
        supported_compact=supported_compact,
        full_size=capture.image_size,
    )

    stages = {
        "unshifted_score_input": (native_image[supported_full], recovar_image[supported_full]),
        "captured_shifted_image": (native_shifted[supported_full], recovar_shifted[supported_full]),
        "native_input_cuda_replay": (native_shifted[supported_full], native_replay[supported_full]),
        "recovar_input_cuda_replay": (native_shifted[supported_full], recovar_replay[supported_full]),
        "recovar_cuda_replay_vs_production": (
            recovar_shifted[supported_full],
            recovar_replay[supported_full],
        ),
    }
    return {
        "schema": "recovar.em.k1_direct_score_input_boundary.v1",
        "status": "complete",
        "identity": {
            "stack_index_one_based": capture.stack_index,
            "original_index_zero_based": int(np.asarray(recovar["original_index"]).item()),
            "native_rotation_local": int(candidate["rotation_local"]),
            "native_translation": int(candidate["translation_id"]),
            "recovar_translation_row": translation_row,
        },
        "alignment": {
            "translation_max_abs": translation_error,
            "native_angle": native_angle.tolist(),
            "recovar_angle": angle.tolist(),
            "angle_exact_equal": bool(np.array_equal(native_angle, angle)),
            "supported_pixel_count": int(supported_full.size),
        },
        "stage_metrics": {
            name: _metric(native, candidate_value)
            for name, (native, candidate_value) in stages.items()
        },
        "stage_float32_ulp_metrics": {
            name: _float32_ulp_stats(native, candidate_value)
            for name, (native, candidate_value) in stages.items()
        },
        "first_exact_unequal_boundary": next(
            (
                name
                for name, (native, candidate_value) in stages.items()
                if not np.array_equal(native, candidate_value)
            ),
            None,
        ),
        "native_capture_validation": validation,
        "artifacts": {
            "native_capture": str(capture_path.resolve()),
            "native_capture_sha256": _sha256(capture_path),
            "recovar": str(recovar_path.resolve()),
            "recovar_sha256": _sha256(recovar_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        capture_path=args.capture,
        recovar_path=args.recovar,
        physical_image_size=args.physical_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
