#!/usr/bin/env python3
"""Validate and compare a bounded RELION fine-image-input capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
)

if __package__:
    from scripts.validate_relion_fine_operand_capture import load_fine_operand_capture
    from scripts.validate_relion_preprocess_capture import load_artifact
else:
    from validate_relion_fine_operand_capture import load_fine_operand_capture
    from validate_relion_preprocess_capture import load_artifact


HEADER_MAGIC = b"RLNFIMGV1HEADER\0"
FOOTER_MAGIC = b"RLNFIMGV1FOOTER\0"
HEADER = struct.Struct("<16s40Q")
FOOTER = struct.Struct("<16sQQ")
PIXEL_DTYPE = np.dtype(
    [
        ("pixel", "<u4"),
        ("flags", "<u4"),
        ("fourier_real", "<f4"),
        ("fourier_imag", "<f4"),
        ("local_fctf", "<f4"),
        ("pixel_correction", "<f4"),
        ("corrected_real", "<f4"),
        ("corrected_imag", "<f4"),
    ]
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


def _float32_from_bits(value: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", value & 0xFFFFFFFF), dtype="<f4")[0]


def _ordered_float_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return np.where(
        (bits & np.uint32(0x80000000)) != 0,
        ~bits,
        bits | np.uint32(0x80000000),
    ).astype(np.uint32)


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    _require(reference.shape == candidate.shape, "metric shape mismatch")
    left = np.ascontiguousarray(reference).view(np.float32).reshape(-1)
    right = np.ascontiguousarray(candidate).view(np.float32).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    ulp = np.abs(
        _ordered_float_bits(left[finite]).astype(np.int64)
        - _ordered_float_bits(right[finite]).astype(np.int64)
    )
    mismatch = left.view(np.uint32) != right.view(np.uint32)
    difference = right.astype(np.float64) - left.astype(np.float64)
    denominator = np.linalg.norm(left.astype(np.float64))
    first = np.flatnonzero(mismatch)
    return {
        "component_count": int(left.size),
        "bit_exact_component_count": int(np.count_nonzero(~mismatch)),
        "mismatch_component_count": int(np.count_nonzero(mismatch)),
        "first_mismatch_component": int(first[0]) if first.size else None,
        "relative_l2": float(np.linalg.norm(difference) / denominator)
        if denominator
        else 0.0,
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "max_ulp": int(np.max(ulp, initial=0)),
        "p95_ulp": float(np.percentile(ulp, 95)) if ulp.size else None,
    }


def _load_capture(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    payload = path.read_bytes()
    _require(len(payload) >= HEADER.size + FOOTER.size, "truncated capture")
    magic, *raw_header = HEADER.unpack_from(payload)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, "header magic mismatch")
    _require(header[0] == 1, "schema mismatch")
    _require(header[1] == HEADER.size, "header size mismatch")
    _require(header[2] == PIXEL_DTYPE.itemsize, "pixel size mismatch")
    _require(header[3] == FOOTER.size, "footer size mismatch")
    pixel_count = header[10]
    expected = HEADER.size + pixel_count * PIXEL_DTYPE.itemsize + FOOTER.size
    _require(len(payload) == expected, "artifact byte-count mismatch")
    _require(header[16] == expected, "header byte-count mismatch")
    _require(header[20:22] == (1, 1), "passive/full-pixel flags missing")
    pixels = np.frombuffer(
        payload, dtype=PIXEL_DTYPE, count=pixel_count, offset=HEADER.size
    ).copy()
    footer_magic, footer_count, footer_reserved = FOOTER.unpack_from(
        payload, HEADER.size + pixels.nbytes
    )
    _require(footer_magic == FOOTER_MAGIC, "footer magic mismatch")
    _require(footer_count == pixel_count and footer_reserved == 0, "footer mismatch")
    _require(
        np.array_equal(pixels["pixel"], np.arange(pixel_count, dtype=np.uint32)),
        "pixel identities are not dense and ordered",
    )
    _require(np.all(np.isfinite(pixels["local_fctf"])), "non-finite CTF values")
    _require(
        np.all(np.isfinite(pixels["pixel_correction"])),
        "non-finite pixel corrections",
    )
    return header, pixels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--direct-dump", type=Path, required=True)
    parser.add_argument("--native-fine-capture", type=Path, required=True)
    parser.add_argument("--native-preprocess-capture", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    header, pixels = _load_capture(args.capture)
    fourier = (
        pixels["fourier_real"] + np.complex64(1j) * pixels["fourier_imag"]
    ).astype(np.complex64)
    corrected = (
        pixels["corrected_real"] + np.complex64(1j) * pixels["corrected_imag"]
    ).astype(np.complex64)
    correction = np.asarray(pixels["pixel_correction"], dtype=np.float32)
    closure = (fourier * correction).astype(np.complex64)

    fine = load_fine_operand_capture(args.native_fine_capture)
    _require(fine.candidates.size == 1, "fine capture must contain one candidate")
    _require(fine.image_size == pixels.size, "native fine topology mismatch")
    fine_pixels = fine.pixels.reshape(1, fine.image_size)[0]
    native_device_image = (
        fine_pixels["image_real"]
        + np.complex64(1j) * fine_pixels["image_imag"]
    ).astype(np.complex64)
    preprocess = load_artifact(args.native_preprocess_capture)
    preprocess_fourier = np.asarray(
        preprocess.masked_fourier_post_optics, dtype=np.complex64
    ).reshape(-1)
    _require(preprocess_fourier.shape == fourier.shape, "preprocess topology mismatch")

    with np.load(args.direct_dump, allow_pickle=False) as archive:
        current_size = int(np.asarray(archive["current_size"]).item())
        window_indices = np.asarray(archive["window_indices"], dtype=np.int32)
        live_preprocessed = np.asarray(
            archive["direct_preprocessed_score_input"], dtype=np.complex64
        )
        live_correction = np.asarray(
            archive["direct_pixel_correction"], dtype=np.float32
        )
        live_score_input = np.asarray(archive["direct_score_input"], dtype=np.complex64)
    _require(header[11] == current_size, "current-size mismatch")
    _require(
        pixels.size == current_size * (current_size // 2 + 1),
        "RELION current-size Fourier topology mismatch",
    )
    lookup = _relion_cuda_fine_full_to_compact_lookup(
        (args.physical_image_size, args.physical_image_size),
        current_size,
        window_indices,
    )
    supported_full = np.flatnonzero(lookup >= 0)
    supported_compact = lookup[supported_full]
    _require(
        np.array_equal(np.sort(supported_compact), np.arange(live_score_input.size)),
        "compact RECOVAR rows do not cover the RELION support",
    )
    n2 = np.float32(args.physical_image_size**2)
    live_preprocessed_relion = (
        live_preprocessed[supported_compact] / n2
    ).astype(np.complex64)
    live_corrected_relion = (-live_score_input[supported_compact] / n2).astype(
        np.complex64
    )

    report = {
        "schema": "recovar.em.k1_fine_image_input_boundary.v1",
        "status": "complete",
        "identity": {
            "iteration": header[4],
            "part_id": header[5],
            "stack_index": header[6],
            "mpi_rank": header[7],
            "thread": header[8],
            "image_id": header[9],
            "pixel_count": header[10],
            "current_size": header[11],
            "scale_correction": float(_float32_from_bits(header[12])),
            "do_scale_correction": bool(header[13]),
            "do_ctf_correction": bool(header[14]),
            "refs_are_ctf_corrected": bool(header[15]),
        },
        "native_host_product_closure": _metric(corrected, closure),
        "native_preprocess_vs_native_host_fourier": _metric(
            preprocess_fourier, fourier
        ),
        "negative_native_preprocess_vs_native_host_fourier": _metric(
            -preprocess_fourier, fourier
        ),
        "native_host_corrected_vs_native_device_image": _metric(
            corrected, native_device_image
        ),
        "supported_boundary": {
            "pixel_count": int(supported_full.size),
            "native_host_fourier_vs_recovar_preprocessed": _metric(
                fourier[supported_full], live_preprocessed_relion
            ),
            "native_pixel_correction_vs_recovar": _metric(
                correction[supported_full], live_correction[supported_compact]
            ),
            "native_pixel_correction_vs_negative_recovar": _metric(
                correction[supported_full], -live_correction[supported_compact]
            ),
            "native_host_corrected_vs_recovar_score_input": _metric(
                corrected[supported_full], live_corrected_relion
            ),
            "native_device_image_vs_recovar_score_input": _metric(
                native_device_image[supported_full], live_corrected_relion
            ),
        },
        "capture": str(args.capture.resolve()),
        "capture_sha256": _sha256(args.capture),
        "native_fine_capture": str(args.native_fine_capture.resolve()),
        "native_fine_capture_sha256": _sha256(args.native_fine_capture),
        "native_preprocess_capture": str(args.native_preprocess_capture.resolve()),
        "native_preprocess_capture_sha256": _sha256(args.native_preprocess_capture),
        "direct_dump": str(args.direct_dump.resolve()),
        "direct_dump_sha256": _sha256(args.direct_dump),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
