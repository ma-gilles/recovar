#!/usr/bin/env python3
"""Locate the first unequal K=1 BPref production operand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCHEMA = "recovar-k1-bpref-operand-comparison-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_flat(path: Path, dtype) -> np.ndarray:
    with path.open("rb") as stream:
        count = np.fromfile(stream, dtype=np.uint64, count=1)
        _require(count.size == 1, f"missing native count in {path}")
        values = np.fromfile(stream, dtype=dtype)
    _require(values.size == int(count[0]), f"native count mismatch in {path}")
    return values


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    _require(candidate.shape == reference.shape, f"shape mismatch: {candidate.shape} != {reference.shape}")
    comparison_dtype = np.complex128 if (
        np.iscomplexobj(candidate) or np.iscomplexobj(reference)
    ) else np.float64
    difference = candidate.astype(comparison_dtype) - reference.astype(comparison_dtype)
    unequal = np.flatnonzero(candidate.reshape(-1) != reference.reshape(-1))
    denominator = max(float(np.linalg.norm(reference.astype(comparison_dtype))), np.finfo(np.float64).tiny)
    return {
        "shape": list(candidate.shape),
        "bitwise_equal": bool(unequal.size == 0),
        "unequal_count": int(unequal.size),
        "first_unequal_flat_index": None if unequal.size == 0 else int(unequal[0]),
        "max_abs": float(np.abs(difference).max(initial=0.0)),
        "relative_l2": float(np.linalg.norm(difference) / denominator),
    }


def _variants(reference: np.ndarray, candidates: dict[str, np.ndarray]) -> dict:
    reports = {name: _metrics(value, reference) for name, value in candidates.items()}
    best = min(reports, key=lambda name: reports[name]["relative_l2"])
    return {"best": best, "variants": reports}


def _half_height(size: int) -> int:
    for height in range(2, 4097, 2):
        if height * (height // 2 + 1) == int(size):
            return height
    raise ValueError(f"cannot infer an even FFTW half height from {size} values")


def _half_coordinates(height: int) -> list[tuple[int, int]]:
    return [
        (x, row if row <= height // 2 else row - height)
        for row in range(height)
        for x in range(height // 2 + 1)
    ]


def _aligned_active(native, recovar, *, max_r: float, recovar_scale=1):
    native = np.asarray(native).reshape(-1)
    recovar = np.asarray(recovar).reshape(-1)
    native_lookup = {
        coordinate: index
        for index, coordinate in enumerate(_half_coordinates(_half_height(native.size)))
    }
    recovar_lookup = {
        coordinate: index
        for index, coordinate in enumerate(_half_coordinates(_half_height(recovar.size)))
    }
    coordinates = sorted(
        (
            coordinate for coordinate in native_lookup.keys() & recovar_lookup.keys()
            if coordinate[0] * coordinate[0] + coordinate[1] * coordinate[1] <= max_r * max_r
            and not (coordinate[0] == 0 and coordinate[1] < 0)
        ),
        key=lambda coordinate: (coordinate[1], coordinate[0]),
    )
    return (
        np.asarray([recovar_scale * recovar[recovar_lookup[value]] for value in coordinates]),
        np.asarray([native[native_lookup[value]] for value in coordinates]),
    )


def _native_operands(native_directory: Path, half: int, internal_index: int) -> dict:
    matches = sorted(native_directory.glob(
        f"half{half}_part{internal_index}_stack*_bpref_prefix_metadata.bin"
    ))
    _require(len(matches) == 1, f"expected one native capture for half {half}, row {internal_index}")
    metadata_path = matches[0]
    prefix = metadata_path.name.removesuffix("metadata.bin")
    load = lambda suffix, dtype=np.float32: _load_flat(native_directory / f"{prefix}{suffix}.bin", dtype)
    metadata = load("metadata", np.uint64)
    _require(metadata.size == 15 and int(metadata[0]) == 1, "native metadata schema changed")
    image = np.asarray(load("operand_image_real") + np.complex64(1j) * load("operand_image_imag"), dtype=np.complex64)
    tx = load("operand_translation_x")
    ty = load("operand_translation_y")
    tz = load("operand_translation_z")
    weights = load("operand_weights")
    eulers = load("operand_eulers")
    controls = load("operand_controls")
    _require(tx.size == ty.size == tz.size and tx.size > 0, "native translation shape changed")
    _require(weights.size % tx.size == 0, "native posterior shape changed")
    n_rotations = weights.size // tx.size
    _require(eulers.size == n_rotations * 9, "native Euler shape changed")
    _require(controls.size == 2 and float(controls[1]) > 0, "native controls changed")
    return {
        "stack_index_1based": int(metadata[4]),
        "image_height": _half_height(image.size),
        "image": image,
        "translations": np.stack((tx, ty), axis=1),
        "translation_z": tz,
        "weights": weights.reshape(n_rotations, tx.size),
        "minvsigma2": load("operand_minvsigma2"),
        "ctf": load("operand_ctf"),
        "eulers": eulers.reshape(n_rotations, 3, 3),
        "significant_weight": np.float32(controls[0]),
        "weight_norm": np.float32(controls[1]),
    }


def analyze(selection_path: Path, native_directory: Path, recovar_directories: dict[int, Path]) -> dict:
    selection = json.loads(selection_path.read_text())
    _require(selection["schema"] == "recovar-k1-bpref-prefix-selection-v2", "selection schema changed")
    rows = []
    for half in (1, 2):
        selected = selection[f"half{half}"]
        _require(len(selected["original_indices"]) == 1, "operand discriminator requires one particle per half")
        original_index = int(selected["original_indices"][0])
        internal_index = int(selected["native_internal_indices"][0])
        stack_index = int(selected["stack_indices_1based"][0])
        native = _native_operands(native_directory, half, internal_index)
        _require(native["stack_index_1based"] == stack_index, "immutable native identity changed")
        recovar_path = recovar_directories[half] / (
            f"bpref_accumulator_delta_it001_h{half}_orig{original_index:06d}.npz"
        )
        with np.load(recovar_path, allow_pickle=False) as recovar:
            image = np.asarray(recovar["operand_source_image"], dtype=np.complex64).reshape(-1)
            ctf = np.asarray(recovar["operand_ctf"], dtype=np.float32).reshape(-1)
            minvsigma2 = np.asarray(recovar["operand_minvsigma2"], dtype=np.float32).reshape(-1)
            posterior = np.asarray(recovar["operand_posterior"], dtype=np.float32)
            translations = np.asarray(recovar["operand_translation_angles"], dtype=np.float32)
            eulers = np.asarray(recovar["operand_eulers"], dtype=np.float32)
            threshold = np.float32(np.asarray(recovar["operand_threshold"]).reshape(-1)[0])
            weight_norm = np.float32(np.asarray(recovar["operand_weight_norm"]).reshape(-1)[0])
            max_r = float(np.asarray(recovar["max_r"]).reshape(()))

        native_normalized_weights = np.asarray(native["weights"] / native["weight_norm"], dtype=np.float32)
        native_normalized_threshold = np.float32(native["significant_weight"] / native["weight_norm"])
        image_aligned, native_image_aligned = _aligned_active(
            native["image"], image, max_r=max_r
        )
        ctf_aligned, native_ctf_aligned = _aligned_active(
            native["ctf"], ctf, max_r=max_r
        )
        noise_aligned, native_noise_aligned = _aligned_active(
            native["minvsigma2"], minvsigma2, max_r=max_r
        )
        native_support = native["weights"] >= native["significant_weight"]
        row = {
            "half": half,
            "original_index": original_index,
            "native_internal_index": internal_index,
            "stack_index_1based": stack_index,
            "native_image_height": native["image_height"],
            "recovar_image_height": _half_height(image.size),
            "active_pixel_count": int(image_aligned.size),
            "image": _variants(native_image_aligned, {
                "direct": image_aligned,
                "negative": -image_aligned,
                "conjugate": np.conjugate(image_aligned),
                "negative_conjugate": -np.conjugate(image_aligned),
            }),
            "ctf": _variants(native_ctf_aligned, {
                "direct": ctf_aligned,
                "negative": -ctf_aligned,
            }),
            "minvsigma2": _metrics(noise_aligned, native_noise_aligned),
            "translations_xy": _variants(native["translations"], {
                "direct": translations,
                "negative": -translations,
            }),
            "native_translation_z_zero": _metrics(
                np.zeros_like(native["translation_z"]), native["translation_z"]
            ),
            "eulers": _variants(native["eulers"], {
                "direct": eulers,
                "transpose_each": np.swapaxes(eulers, -1, -2),
            }),
            "posterior_normalized_active": _metrics(
                posterior[native_support], native_normalized_weights[native_support]
            ),
            "support_mask": _metrics(
                posterior >= threshold,
                native["weights"] >= native["significant_weight"],
            ),
            "normalized_threshold": _metrics(
                np.asarray([threshold], dtype=np.float32),
                np.asarray([native_normalized_threshold], dtype=np.float32),
            ),
            "controls": {
                "native_significant_weight": float(native["significant_weight"]),
                "native_weight_norm": float(native["weight_norm"]),
                "native_normalized_threshold": float(native_normalized_threshold),
                "recovar_threshold": float(threshold),
                "recovar_weight_norm": float(weight_norm),
            },
        }
        rows.append(row)
    return {
        "schema": SCHEMA,
        "selection": str(selection_path.resolve()),
        "native_directory": str(native_directory.resolve()),
        "recovar_directories": {
            str(half): str(path.resolve()) for half, path in recovar_directories.items()
        },
        "rows": rows,
    }


def _markdown(report: dict) -> str:
    lines = [
        "# K=1 first-particle BPref operand comparison",
        "",
        "| Half | Image transform | Image relL2 | CTF relL2 | Noise relL2 | Posterior relL2 | Support exact | Euler transform | Euler relL2 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for row in report["rows"]:
        image_best = row["image"]["best"]
        ctf_best = row["ctf"]["best"]
        euler_best = row["eulers"]["best"]
        lines.append(
            f"| {row['half']} | {image_best} | {row['image']['variants'][image_best]['relative_l2']:.9g} | "
            f"{row['ctf']['variants'][ctf_best]['relative_l2']:.9g} | {row['minvsigma2']['relative_l2']:.9g} | "
            f"{row['posterior_normalized_active']['relative_l2']:.9g} | "
            f"{row['support_mask']['bitwise_equal']} | {euler_best} | "
            f"{row['eulers']['variants'][euler_best]['relative_l2']:.9g} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--native-directory", required=True, type=Path)
    parser.add_argument("--recovar-half1-directory", required=True, type=Path)
    parser.add_argument("--recovar-half2-directory", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        args.selection,
        args.native_directory,
        {1: args.recovar_half1_directory, 2: args.recovar_half2_directory},
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
