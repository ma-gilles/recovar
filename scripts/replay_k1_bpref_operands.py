#!/usr/bin/env python3
"""Replay the first K=1 BPref particle with native and RECOVAR operands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject
from scripts.analyze_k1_bpref_operands import _load_flat, _metrics, _native_operands


def _half_coordinates(height: int) -> list[tuple[int, int]]:
    width = height // 2 + 1
    return [
        (x, row if row <= height // 2 else row - height)
        for row in range(height)
        for x in range(width)
    ]


def _reshape_height(values: np.ndarray) -> int:
    size = int(np.asarray(values).size)
    for height in range(2, 4097, 2):
        if height * (height // 2 + 1) == size:
            return height
    raise ValueError(f"cannot infer FFTW half height from {size} values")


def _remap(values: np.ndarray, source_height: int, target_height: int, fill=0) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    source_coordinates = _half_coordinates(source_height)
    target_coordinates = _half_coordinates(target_height)
    source_lookup = {coordinate: index for index, coordinate in enumerate(source_coordinates)}
    result = np.full(len(target_coordinates), fill, dtype=values.dtype)
    for target_index, coordinate in enumerate(target_coordinates):
        source_index = source_lookup.get(coordinate)
        if source_index is not None:
            result[target_index] = values[source_index]
    return result


def _replace_common(native: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    native = np.asarray(native).reshape(-1).copy()
    native_height = _reshape_height(native)
    replacement = np.asarray(replacement).reshape(-1)
    replacement_height = _reshape_height(replacement)
    replacement_lookup = {
        coordinate: index for index, coordinate in enumerate(_half_coordinates(replacement_height))
    }
    for native_index, coordinate in enumerate(_half_coordinates(native_height)):
        replacement_index = replacement_lookup.get(coordinate)
        if replacement_index is not None:
            native[native_index] = replacement[replacement_index]
    return native


def _native_prefix(native_directory: Path, half: int, internal_index: int) -> tuple[np.ndarray, np.ndarray]:
    metadata = sorted(native_directory.glob(
        f"half{half}_part{internal_index}_stack*_bpref_prefix_metadata.bin"
    ))
    if len(metadata) != 1:
        raise ValueError("native prefix identity is ambiguous")
    prefix = metadata[0].name.removesuffix("metadata.bin")
    real = _load_flat(native_directory / f"{prefix}real.bin", np.float32)
    imag = _load_flat(native_directory / f"{prefix}imag.bin", np.float32)
    weight = _load_flat(native_directory / f"{prefix}weight.bin", np.float32)
    return np.asarray(real + np.complex64(1j) * imag, dtype=np.complex64), weight


def _run(image, ctf, noise, posterior, translations, eulers, threshold, norm, image_height):
    volume_shape = (115, 115, 115)
    volume_size = 115 * 115 * 58
    data, weight = cuda_backproject.relion_firstiter_bpref_fused_x_half(
        jnp.zeros(volume_size, dtype=jnp.complex64),
        jnp.zeros(volume_size, dtype=jnp.float32),
        jnp.asarray(image, dtype=jnp.complex64),
        jnp.asarray(ctf, dtype=jnp.float32),
        jnp.asarray(noise, dtype=jnp.float32),
        jnp.asarray(posterior, dtype=jnp.float32),
        jnp.asarray(translations, dtype=jnp.float32),
        jnp.asarray(eulers, dtype=jnp.float32),
        jnp.asarray([threshold], dtype=jnp.float32),
        jnp.asarray([norm], dtype=jnp.float32),
        (image_height, image_height),
        volume_shape,
        28.0,
    )
    return np.asarray(data, dtype=np.complex64), np.asarray(weight, dtype=np.float32)


def replay_half(native_directory: Path, recovar_path: Path, half: int, internal_index: int) -> dict:
    native = _native_operands(native_directory, half, internal_index)
    native_data, native_weight = _native_prefix(native_directory, half, internal_index)
    with np.load(recovar_path, allow_pickle=False) as values:
        rec_image = np.asarray(values["operand_source_image"], dtype=np.complex64)
        rec_ctf = np.asarray(values["operand_ctf"], dtype=np.float32)
        rec_noise = np.asarray(values["operand_minvsigma2"], dtype=np.float32)
        rec_posterior = np.asarray(values["operand_posterior"], dtype=np.float32)
        rec_translations = np.asarray(values["operand_translation_angles"], dtype=np.float32)
        rec_eulers = np.asarray(values["operand_eulers"], dtype=np.float32)
        rec_threshold = np.float32(values["operand_threshold"].reshape(-1)[0])
        rec_isolated_data = -np.asarray(values["isolated_data"], dtype=np.complex64).reshape(-1)
        rec_isolated_weight = np.asarray(values["isolated_weight"], dtype=np.float32).reshape(-1)

    native_height = _reshape_height(native["image"])
    rec_height = _reshape_height(rec_image)
    native_with_rec_noise = _replace_common(native["minvsigma2"], rec_noise)
    native_with_rec_image = _replace_common(native["image"], rec_image)
    # The two programs expose opposite CTF signs at this boundary; using
    # negative RECOVAR CTF values preserves RELION's native numerator sign.
    native_with_rec_ctf = _replace_common(native["ctf"], -rec_ctf)
    arms = {
        "native_all": (
            native["image"], native["ctf"], native["minvsigma2"], native["weights"],
            native["translations"], native["eulers"], native["significant_weight"],
            native["weight_norm"], native_height,
        ),
        "rec_noise_only": (
            native["image"], native["ctf"], native_with_rec_noise, native["weights"],
            native["translations"], native["eulers"], native["significant_weight"],
            native["weight_norm"], native_height,
        ),
        "rec_translation_only": (
            native["image"], native["ctf"], native["minvsigma2"], native["weights"],
            rec_translations, native["eulers"], native["significant_weight"],
            native["weight_norm"], native_height,
        ),
        "rec_noise_translation": (
            native["image"], native["ctf"], native_with_rec_noise, native["weights"],
            rec_translations, native["eulers"], native["significant_weight"],
            native["weight_norm"], native_height,
        ),
        "all_rec_aligned_native_topology": (
            native_with_rec_image, native_with_rec_ctf, native_with_rec_noise, rec_posterior,
            rec_translations, rec_eulers, rec_threshold, np.float32(1), native_height,
        ),
        "native_operands_rec_topology": (
            _remap(native["image"], native_height, rec_height),
            _remap(native["ctf"], native_height, rec_height),
            _remap(native["minvsigma2"], native_height, rec_height),
            native["weights"], native["translations"], native["eulers"],
            native["significant_weight"], native["weight_norm"], rec_height,
        ),
    }
    report_arms = {}
    for name, operands in arms.items():
        data, weight = _run(*operands)
        report_arms[name] = {
            "data": _metrics(data, native_data),
            "weight": _metrics(weight, native_weight),
        }

    production_data, production_weight = _run(
        rec_image, rec_ctf, rec_noise, rec_posterior, rec_translations, rec_eulers,
        rec_threshold, np.float32(1), rec_height,
    )
    report_arms["production_rec_replay"] = {
        "data_vs_native_sign_aligned": _metrics(-production_data, native_data),
        "weight_vs_native": _metrics(production_weight, native_weight),
        "data_vs_captured_rec": _metrics(-production_data, rec_isolated_data),
        "weight_vs_captured_rec": _metrics(production_weight, rec_isolated_weight),
    }
    return {
        "half": half,
        "native_internal_index": internal_index,
        "native_image_height": native_height,
        "recovar_image_height": rec_height,
        "arms": report_arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-directory", required=True, type=Path)
    parser.add_argument("--recovar-half1", required=True, type=Path)
    parser.add_argument("--recovar-half2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "schema": "recovar-k1-bpref-native-operand-replay-v1",
        "halves": [
            replay_half(args.native_directory, args.recovar_half1, 1, 79970),
            replay_half(args.native_directory, args.recovar_half2, 2, 29717),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
