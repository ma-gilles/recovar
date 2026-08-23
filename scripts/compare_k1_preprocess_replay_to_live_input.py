#!/usr/bin/env python3
"""Join native preprocessing replays to one live RECOVAR fine-score input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
)
if __package__:
    from scripts.validate_relion_fine_operand_capture import load_fine_operand_capture
    from scripts.validate_relion_preprocess_capture import (
        load_artifact as load_preprocess_capture,
    )
else:
    from validate_relion_fine_operand_capture import (  # type: ignore[no-redef]
        load_fine_operand_capture,
    )
    from validate_relion_preprocess_capture import (  # type: ignore[no-redef]
        load_artifact as load_preprocess_capture,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference, dtype=np.complex64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.complex64).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    difference = candidate.astype(np.complex128) - reference.astype(np.complex128)
    denominator = np.linalg.norm(reference.astype(np.complex128))
    complex_exact = reference.view(np.float32).reshape(-1, 2).view(
        np.uint32
    ) == candidate.view(np.float32).reshape(-1, 2).view(np.uint32)
    return {
        "complex_value_count": int(reference.size),
        "complex_bit_exact_count": int(np.count_nonzero(np.all(complex_exact, axis=1))),
        "complex_mismatch_count": int(np.count_nonzero(~np.all(complex_exact, axis=1))),
        "relative_l2": float(np.linalg.norm(difference) / denominator) if denominator else 0.0,
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-dump", type=Path, required=True)
    parser.add_argument(
        "--stage-npz",
        metavar="MODE=PATH",
        action="append",
        default=[],
        help="stage replay archive, for example default=/path/to/report.npz",
    )
    parser.add_argument("--physical-image-size", type=int, default=128)
    parser.add_argument("--native-fine-capture", type=Path)
    parser.add_argument("--native-preprocess-capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stage_paths: dict[str, Path] = {}
    for item in args.stage_npz:
        mode, separator, path = item.partition("=")
        if not separator or not mode or mode in stage_paths:
            raise ValueError(f"invalid or duplicate MODE=PATH value: {item}")
        stage_paths[mode] = Path(path)

    with np.load(args.direct_dump, allow_pickle=False) as archive:
        current_size = int(np.asarray(archive["current_size"]).item())
        window_indices = np.asarray(archive["window_indices"], dtype=np.int32)
        direct = np.asarray(archive["direct_score_input"], dtype=np.complex64)
        live_preprocessed = np.asarray(
            archive["direct_preprocessed_score_input"], dtype=np.complex64
        )
        pixel_correction = np.asarray(
            archive["direct_pixel_correction"], dtype=np.float32
        )
        normalization_factor = float(
            np.asarray(archive["relion_preprocess_normalization_factor"]).item()
        )
        integer_pre_shift = np.asarray(
            archive["relion_integer_pre_shift"], dtype=np.int32
        )
        batch_image_correction = float(
            np.asarray(archive["batch_image_correction"]).item()
        )
        batch_scale_correction = float(
            np.asarray(archive["batch_scale_correction"]).item()
        )
    lookup = _relion_cuda_fine_full_to_compact_lookup(
        (args.physical_image_size, args.physical_image_size),
        current_size,
        window_indices,
    )
    supported_full = np.flatnonzero(lookup >= 0)
    supported_compact = lookup[supported_full]
    if not np.array_equal(np.sort(supported_compact), np.arange(direct.size)):
        raise ValueError("live compact input does not cover the supported current-size rows")
    if live_preprocessed.shape != direct.shape or pixel_correction.shape != direct.shape:
        raise ValueError("live preprocessing/correction topology differs from score input")
    # RECOVAR retains cuFFT's unnormalised output.  RELION's preprocessing
    # capture stores the same image after its explicit 1/N^2 scale.
    live_preprocessed_relion_scale = (
        live_preprocessed[supported_compact] / np.float32(args.physical_image_size**2)
    ).astype(np.complex64)

    native_preprocess_operands = None
    native_reference = None
    if args.native_preprocess_capture is not None:
        native_preprocess = load_preprocess_capture(args.native_preprocess_capture)
        native_full = np.asarray(
            native_preprocess.masked_fourier_post_optics, dtype=np.complex64
        ).reshape(-1)
        if native_full.size != lookup.size:
            raise ValueError(
                "native preprocessing/current-size topology mismatch: "
                f"{native_full.size} != {lookup.size}"
            )
        native_reference = native_full[supported_full]
        native_preprocess_operands = {
            "normalization_factor": native_preprocess.norm_correction,
            "integer_pre_shift": np.asarray(
                native_preprocess.old_offset[:2], dtype=np.int32
            ).tolist(),
            "normalization_factor_float32_ulp_delta_live_minus_native": int(
                np.float32(normalization_factor).view(np.uint32)
            )
            - int(np.float32(native_preprocess.norm_correction).view(np.uint32)),
            "native_vs_live_preprocessed": _metric(
                native_reference, live_preprocessed_relion_scale
            ),
            "capture": str(args.native_preprocess_capture.resolve()),
            "capture_sha256": _sha256(args.native_preprocess_capture),
        }

    modes: dict[str, object] = {}
    for mode, path in stage_paths.items():
        with np.load(path, allow_pickle=False) as archive:
            native = np.asarray(
                archive["native_masked_fourier_post_optics"][0], dtype=np.complex64
            ).reshape(-1)[supported_full]
            replay = np.asarray(
                archive["recovar_masked_fourier"][0], dtype=np.complex64
            ).reshape(-1)[supported_full]
        if native_reference is None:
            native_reference = native
        elif not np.array_equal(native_reference, native):
            raise ValueError(
                "stage archive and preprocessing capture do not contain the same "
                "native operand"
            )
        modes[mode] = {
            "native_vs_replay": _metric(native, replay),
            "native_vs_live_preprocessed": _metric(
                native, live_preprocessed_relion_scale
            ),
            "replay_vs_live_preprocessed": _metric(
                replay, live_preprocessed_relion_scale
            ),
            "stage_npz": str(path.resolve()),
            "stage_npz_sha256": _sha256(path),
        }

    native_fine_metrics = None
    if args.native_fine_capture is not None:
        fine = load_fine_operand_capture(args.native_fine_capture)
        if fine.candidates.size != 1:
            raise ValueError("native fine capture must contain exactly one tuple")
        pixels = fine.pixels.reshape(1, fine.image_size)[0]
        native_fine = (
            np.asarray(pixels["image_real"], dtype=np.float32)
            + np.complex64(1j) * np.asarray(pixels["image_imag"], dtype=np.float32)
        ).astype(np.complex64)[supported_full]
        live_fine_convention = (
            -direct[supported_compact] / np.float32(args.physical_image_size**2)
        ).astype(np.complex64)
        assert native_reference is not None
        exact_preprocess_counterfactual = (
            -native_reference * pixel_correction[supported_compact]
        ).astype(np.complex64)
        native_fine_metrics = {
            "native_vs_live": _metric(native_fine, live_fine_convention),
            "native_vs_exact_preprocess_counterfactual": _metric(
                native_fine, exact_preprocess_counterfactual
            ),
            "live_vs_exact_preprocess_counterfactual": _metric(
                live_fine_convention, exact_preprocess_counterfactual
            ),
            "capture": str(args.native_fine_capture.resolve()),
            "capture_sha256": _sha256(args.native_fine_capture),
        }

    if native_reference is None:
        raise ValueError(
            "at least one --stage-npz or --native-preprocess-capture is required"
        )

    report = {
        "schema": "recovar.em.k1_preprocess_replay_to_live_input.v1",
        "status": "complete",
        "physical_image_size": args.physical_image_size,
        "current_size": current_size,
        "supported_pixel_count": int(supported_full.size),
        "live_product_closure": _metric(
            direct,
            (live_preprocessed * pixel_correction).astype(np.complex64),
        ),
        "live_preprocess_operands": {
            "normalization_factor": normalization_factor,
            "integer_pre_shift": integer_pre_shift.tolist(),
            "batch_image_correction": batch_image_correction,
            "batch_scale_correction": batch_scale_correction,
        },
        "native_fine_metrics": native_fine_metrics,
        "native_preprocess_operands": native_preprocess_operands,
        "modes": modes,
        "direct_dump": str(args.direct_dump.resolve()),
        "direct_dump_sha256": _sha256(args.direct_dump),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
