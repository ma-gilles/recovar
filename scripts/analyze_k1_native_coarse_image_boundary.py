#!/usr/bin/env python3
"""Compare native RELION and RECOVAR K=1 coarse image-score operands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
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


def _one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    _require(len(matches) == 1, f"expected one {pattern!r} file, found {len(matches)}")
    return matches[0]


def _load_flat_real(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        count_raw = stream.read(4)
        _require(len(count_raw) == 4, f"truncated flat-real header: {path}")
        count = int(np.frombuffer(count_raw, dtype="<i4", count=1)[0])
        payload = stream.read()
    _require(count >= 0, f"negative flat-real count in {path}")
    _require(len(payload) == count * 8, f"flat-real payload size mismatch in {path}")
    return np.frombuffer(payload, dtype="<f8", count=count).astype(np.float32)


def _ordered_float_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return np.where(
        (bits & np.uint32(0x80000000)) != 0,
        ~bits,
        bits | np.uint32(0x80000000),
    ).astype(np.uint32)


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    ref = np.ascontiguousarray(reference)
    cand = np.ascontiguousarray(candidate)
    _require(ref.shape == cand.shape, f"metric shape mismatch: {ref.shape} != {cand.shape}")
    _require(ref.dtype == cand.dtype, f"metric dtype mismatch: {ref.dtype} != {cand.dtype}")
    ref_components = ref.view(np.float32).reshape(-1)
    cand_components = cand.view(np.float32).reshape(-1)
    finite = np.isfinite(ref_components) & np.isfinite(cand_components)
    _require(bool(np.all(finite)), "metric operands contain non-finite components")
    mismatch = ref_components.view(np.uint32) != cand_components.view(np.uint32)
    difference = cand_components.astype(np.float64) - ref_components.astype(np.float64)
    denominator = float(np.linalg.norm(ref_components.astype(np.float64)))
    ulp = np.abs(
        _ordered_float_bits(ref_components).astype(np.int64)
        - _ordered_float_bits(cand_components).astype(np.int64)
    )
    first = np.flatnonzero(mismatch)
    return {
        "component_count": int(ref_components.size),
        "bit_exact_component_count": int(np.count_nonzero(~mismatch)),
        "mismatch_component_count": int(np.count_nonzero(mismatch)),
        "first_mismatch_component": int(first[0]) if first.size else None,
        "relative_l2": float(np.linalg.norm(difference) / denominator)
        if denominator
        else float(np.linalg.norm(difference)),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "max_ulp": int(np.max(ulp, initial=0)),
        "p95_ulp": float(np.percentile(ulp, 95)) if ulp.size else None,
    }


def _scalar_fit(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.complex64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.complex64).reshape(-1)
    denominator = float(np.vdot(ref.astype(np.complex128), ref.astype(np.complex128)).real)
    _require(denominator > 0.0, "scalar-fit reference has zero norm")
    alpha = float(
        np.vdot(ref.astype(np.complex128), cand.astype(np.complex128)).real
        / denominator
    )
    residual = cand.astype(np.complex128) - alpha * ref.astype(np.complex128)
    return {
        "candidate_over_reference_real_alpha": alpha,
        "alpha_minus_one": alpha - 1.0,
        "residual_relative_l2_over_reference": float(
            np.linalg.norm(residual) / np.sqrt(denominator)
        ),
    }


def _load_capture(path: Path) -> dict[str, np.ndarray]:
    required = {
        "current_size",
        "original_index",
        "coarse_gaussian_score_indices",
        "coarse_gaussian_unshifted_corrected",
        "coarse_gaussian_pixel_weight",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        _require(not missing, f"coarse capture misses fields: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def analyze(
    *,
    native_dump_dir: Path,
    exact_path: Path,
    live_path: Path,
    physical_image_size: int,
) -> dict[str, object]:
    exact = _load_capture(exact_path)
    live = _load_capture(live_path)
    for field in ("current_size", "original_index", "coarse_gaussian_score_indices"):
        _require(np.array_equal(exact[field], live[field]), f"capture field {field} differs")
    current_size = int(np.asarray(live["current_size"]).item())
    original_index = int(np.asarray(live["original_index"]).item())
    _require(physical_image_size > 0, "physical image size must be positive")

    real_path = _one(native_dump_dir, "pass1_img*_Fimg_corrected_real.bin")
    imag_path = _one(native_dump_dir, "pass1_img*_Fimg_corrected_imag.bin")
    corr_path = _one(native_dump_dir, "pass1_img*_corr_img.bin")
    native_raw_image = (
        _load_flat_real(real_path)
        + np.complex64(1j) * _load_flat_real(imag_path)
    ).astype(np.complex64)
    native_raw_weight = _load_flat_real(corr_path).astype(np.float32)
    native_pixel_count = current_size * (current_size // 2 + 1)
    _require(
        native_raw_image.shape == native_raw_weight.shape == (native_pixel_count,),
        "native image and weight do not match current-size rFFT topology",
    )

    image_shape = (physical_image_size, physical_image_size)
    score_indices = np.asarray(live["coarse_gaussian_score_indices"], dtype=np.int32)
    full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
        image_shape,
        current_size,
        score_indices,
    ).astype(np.int64, copy=False)
    _require(
        full_to_compact.shape == (native_pixel_count,),
        "RECOVAR lookup does not match native current-size topology",
    )
    _require(np.all(full_to_compact >= 0), "square coarse support misses native pixels")
    _require(
        np.unique(full_to_compact).size == native_pixel_count,
        "native-to-RECOVAR pixel map is not bijective",
    )

    scale = np.float32(physical_image_size**2)
    weight_divisor = np.float32(physical_image_size**4)
    native_image = np.asarray(-scale * native_raw_image, dtype=np.complex64)
    native_weight = np.asarray(native_raw_weight / weight_divisor, dtype=np.float32)

    def aligned(payload: dict[str, np.ndarray], field: str, dtype) -> np.ndarray:
        compact = np.asarray(payload[field], dtype=dtype).reshape(-1)
        _require(
            compact.shape == (native_pixel_count,),
            f"RECOVAR field {field} does not cover the square support",
        )
        return compact[full_to_compact]

    exact_image = aligned(exact, "coarse_gaussian_unshifted_corrected", np.complex64)
    live_image = aligned(live, "coarse_gaussian_unshifted_corrected", np.complex64)
    exact_weight = aligned(exact, "coarse_gaussian_pixel_weight", np.float32)
    live_weight = aligned(live, "coarse_gaussian_pixel_weight", np.float32)

    image_metrics = {
        "native_vs_exact": _metric(native_image, exact_image),
        "native_vs_live": _metric(native_image, live_image),
        "exact_vs_live": _metric(exact_image, live_image),
    }
    weight_metrics = {
        "native_vs_exact": _metric(native_weight, exact_weight),
        "native_vs_live": _metric(native_weight, live_weight),
        "exact_vs_live": _metric(exact_weight, live_weight),
    }
    return {
        "schema": "recovar.em.k1_native_coarse_image_boundary.v2",
        "status": "complete",
        "identity": {
            "source_row_zero_based": original_index,
            "stack_index_one_based": original_index + 1,
            "current_size": current_size,
            "physical_image_size": physical_image_size,
            "native_pixel_count": native_pixel_count,
        },
        "unit_conversion": {
            "native_complex_multiplier": float(-scale),
            "native_weight_divisor": float(weight_divisor),
        },
        "corrected_image": {
            "metrics": image_metrics,
            "native_to_exact_scalar_fit": _scalar_fit(native_image, exact_image),
            "native_to_live_scalar_fit": _scalar_fit(native_image, live_image),
            "comparison_qualification": (
                "unqualified until the native RELION and RECOVAR corrected-image "
                "phase and Fourier-coordinate conventions are aligned"
            ),
            "closer_arm_by_relative_l2_unqualified": min(
                ("exact", "live"),
                key=lambda arm: image_metrics[f"native_vs_{arm}"]["relative_l2"],
            ),
        },
        "pixel_weight": {
            "metrics": weight_metrics,
            "closer_arm_by_relative_l2": min(
                ("exact", "live"),
                key=lambda arm: weight_metrics[f"native_vs_{arm}"]["relative_l2"],
            ),
        },
        "artifacts": {
            "native_real": {"path": str(real_path.resolve()), "sha256": _sha256(real_path)},
            "native_imag": {"path": str(imag_path.resolve()), "sha256": _sha256(imag_path)},
            "native_weight": {"path": str(corr_path.resolve()), "sha256": _sha256(corr_path)},
            "exact": {"path": str(exact_path.resolve()), "sha256": _sha256(exact_path)},
            "live": {"path": str(live_path.resolve()), "sha256": _sha256(live_path)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dump-dir", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        native_dump_dir=args.native_dump_dir,
        exact_path=args.exact,
        live_path=args.live,
        physical_image_size=args.physical_image_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
