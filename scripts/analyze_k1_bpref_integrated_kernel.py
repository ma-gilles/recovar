#!/usr/bin/env python3
"""Test the integrated native-unit firstiter BPref CUDA boundary.

The input is the passive RELION factor capture.  The candidate kernel forms
translation/CTF/noise source rows and scatters them in the same block, then is
compared directly with RELION's independent zero-prefix shadow accumulator.
No production RECOVAR state is read or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject

if __package__:
    from .validate_relion_bpref_factor_capture import load_factor_capture
else:
    from validate_relion_bpref_factor_capture import load_factor_capture  # type: ignore[no-redef]


SCHEMA = "recovar.em.k1_bpref_integrated_kernel.v1"
_NATIVE_PATTERN = re.compile(
    r"rank-(?P<rank>\d+)_thread(?P<thread>\d+)_part(?P<part>\d+)_"
    r"stack(?P<stack>\d+)_class(?P<class_>\d+)_bpref_shadow_metadata\.bin$"
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


def _load_counted(path: Path, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    with path.open("rb") as stream:
        count_values = np.fromfile(stream, dtype=np.uint64, count=1)
        _require(count_values.size == 1, f"truncated count in {path}")
        values = np.fromfile(stream, dtype=dtype)
    _require(values.size == int(count_values[0]), f"count mismatch in {path}")
    return values


def _native_prefix(metadata_path: Path) -> Path:
    suffix = "metadata.bin"
    _require(metadata_path.name.endswith(suffix), "native metadata suffix changed")
    return metadata_path.with_name(metadata_path.name[: -len(suffix)])


def _load_native_shadow(metadata_path: Path) -> dict[str, Any]:
    prefix = _native_prefix(metadata_path)
    metadata = _load_counted(metadata_path, np.uint64)
    _require(metadata.size == 18 and int(metadata[0]) == 1, "native shadow schema changed")
    real = _load_counted(Path(f"{prefix}real.bin"), np.float32)
    imag = _load_counted(Path(f"{prefix}imag.bin"), np.float32)
    weight = _load_counted(Path(f"{prefix}weight.bin"), np.float32)
    count = int(metadata[14])
    _require(real.size == imag.size == weight.size == count, "native shadow size changed")
    data = real.astype(np.complex64)
    data.imag = imag
    return {
        "iteration": int(metadata[1]),
        "stack_index_one_based": int(metadata[3]),
        "rank": int(metadata[4]),
        "thread": int(metadata[5]),
        "volume_shape": (int(metadata[9]), int(metadata[8]), 2 * int(metadata[7]) - 1),
        "native_half_shape": (int(metadata[9]), int(metadata[8]), int(metadata[7])),
        "max_r": int(metadata[12]),
        "max_r2": int(metadata[13]),
        "data": data,
        "weight": weight,
        "metadata_path": metadata_path,
    }


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference).reshape(-1)
    right = np.asarray(candidate).reshape(-1)
    _require(left.shape == right.shape and left.size > 0, "metric topology changed")
    delta = right.astype(np.complex128) - left.astype(np.complex128)
    denominator = max(float(np.linalg.norm(left.astype(np.complex128))), np.finfo(np.float64).tiny)
    return {
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_reference": float(np.linalg.norm(delta) / denominator),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "support_mismatch_count": int(np.count_nonzero((left != 0) != (right != 0))),
    }


def _float32_from_bits(value: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", int(value) & 0xFFFFFFFF))[0])


def _run_particle(factor_path: Path, native: dict[str, Any], repeats: int) -> dict[str, Any]:
    factor = load_factor_capture(factor_path)
    _require(factor.stack_index == native["stack_index_one_based"], "factor/shadow identity changed")
    _require(int(factor.header[9]) == native["iteration"] == 1, "physical iteration changed")
    image_h = int(factor.header[17])
    image_w = int(factor.header[16])
    _require(image_w == image_h // 2 + 1, "native FFTW image shape changed")
    _require(factor.pixels.size == image_h * image_w, "factor pixel panel is incomplete")
    _require(np.prod(native["native_half_shape"]) == native["data"].size, "native half shape changed")

    image = (
        factor.pixels["image_re"] + np.complex64(1j) * factor.pixels["image_im"]
    ).astype(np.complex64)
    posterior = factor.hypotheses["posterior"].reshape(
        factor.rotations.size, factor.translations.size
    ).astype(np.float32)
    translation_angles = np.stack(
        (factor.translations["x"], factor.translations["y"]), axis=1
    ).astype(np.float32)
    native_eulers = factor.rotations["matrix"].reshape(-1, 3, 3).astype(np.float32)
    significant_weight = np.asarray([_float32_from_bits(factor.header[25])], dtype=np.float32)
    weight_norm = np.asarray([_float32_from_bits(factor.header[26])], dtype=np.float32)
    _require(significant_weight[0] > 0.0 and weight_norm[0] > 0.0, "invalid weight controls")

    comparisons = []
    outputs = []
    for _ in range(repeats):
        data_volume = jnp.zeros(native["data"].shape, dtype=jnp.complex64)
        weight_volume = jnp.zeros(native["weight"].shape, dtype=jnp.float32)
        output_data, output_weight = cuda_backproject.relion_firstiter_bpref_fused_x_half(
            data_volume,
            weight_volume,
            jnp.asarray(image),
            jnp.asarray(factor.pixels["ctf"], dtype=jnp.float32),
            jnp.asarray(factor.pixels["minvsigma2"], dtype=jnp.float32),
            jnp.asarray(posterior),
            jnp.asarray(translation_angles),
            jnp.asarray(native_eulers),
            jnp.asarray(significant_weight),
            jnp.asarray(weight_norm),
            (image_h, image_h),
            native["volume_shape"],
            float(native["max_r"]),
        )
        output_data, output_weight = jax.block_until_ready((output_data, output_weight))
        output_pair = (np.asarray(output_data), np.asarray(output_weight))
        outputs.append(output_pair)
        comparisons.append(
            {
                "numerator_native_units": _metric(native["data"], output_pair[0]),
                "denominator_native_units": _metric(native["weight"], output_pair[1]),
            }
        )
    repeat_comparisons = [
        {
            "numerator_native_units": _metric(outputs[0][0], outputs[index][0]),
            "denominator_native_units": _metric(outputs[0][1], outputs[index][1]),
        }
        for index in range(1, repeats)
    ]
    return {
        "stack_index_one_based": factor.stack_index,
        "native_rank": native["rank"],
        "native_thread": native["thread"],
        "image_shape": [image_h, image_h],
        "volume_shape": list(native["volume_shape"]),
        "rotation_count": int(factor.rotations.size),
        "translation_count": int(factor.translations.size),
        "significant_weight": float(significant_weight[0]),
        "weight_norm": float(weight_norm[0]),
        "comparisons": comparisons,
        "repeat_comparisons": repeat_comparisons,
        "factor_capture": str(factor_path.resolve()),
        "factor_capture_sha256": factor.sha256,
        "native_metadata": str(native["metadata_path"].resolve()),
        "native_metadata_sha256": _sha256(native["metadata_path"]),
    }


def _summary(particles: list[dict[str, Any]], field: str) -> dict[str, Any]:
    metrics = [particle["comparisons"][0][field] for particle in particles]
    values = np.asarray([metric["relative_l2_over_reference"] for metric in metrics])
    return {
        "count": len(metrics),
        "exact_count": int(sum(metric["exact_equal"] for metric in metrics)),
        "support_exact_count": int(sum(metric["support_mismatch_count"] == 0 for metric in metrics)),
        "relative_l2_minimum": float(values.min()),
        "relative_l2_median": float(np.median(values)),
        "relative_l2_maximum": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-directory", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    _require(args.repeats >= 2, "at least two candidate repeats are required")

    native_by_stack: dict[int, dict[str, Any]] = {}
    for metadata_path in sorted(args.native_directory.glob("*bpref_shadow_metadata.bin")):
        match = _NATIVE_PATTERN.fullmatch(metadata_path.name)
        _require(match is not None, f"unexpected native metadata name {metadata_path.name}")
        native = _load_native_shadow(metadata_path)
        stack = native["stack_index_one_based"]
        _require(stack not in native_by_stack, f"duplicate native stack {stack}")
        native_by_stack[stack] = native
    _require(native_by_stack, "native shadow directory is empty")

    factor_paths = sorted(args.factor_directory.glob("*.bpre-v2.bin"))
    factor_by_stack = {load_factor_capture(path).stack_index: path for path in factor_paths}
    _require(set(factor_by_stack) == set(native_by_stack), "factor/native stack panels differ")
    particles = [
        _run_particle(factor_by_stack[stack], native_by_stack[stack], args.repeats)
        for stack in sorted(native_by_stack)
    ]
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "device": str(jax.devices()[0]),
        "metric_policy": "exact and scale-sensitive native-unit relative-L2; no correlation",
        "particle_count": len(particles),
        "candidate_repeat_count": args.repeats,
        "summary": {
            field: _summary(particles, field)
            for field in ("numerator_native_units", "denominator_native_units")
        },
        "particles": particles,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    _require(not args.output_json.exists(), f"refusing to overwrite {args.output_json}")
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
