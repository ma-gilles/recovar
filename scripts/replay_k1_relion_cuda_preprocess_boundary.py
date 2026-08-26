#!/usr/bin/env python3
"""Replay three RELION CUDA preprocessing operands without running EM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mrcfile
import numpy as np
import starfile

from recovar.cuda_backproject import relion_preprocess_real_f32
from recovar.data_io.image_backends import _centered_rfft2_jax
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _relion_cuda_fine_full_to_compact_lookup,
)

if __package__:
    from .compare_k1_relion_recovar_fine_operands import _json_default, _metric
    from .validate_relion_bpref_factor_capture import load_factor_capture
    from .validate_relion_fine_operand_capture import load_fine_operand_capture
    from .validate_relion_preprocess_capture import load_artifact as load_preprocess_capture
else:
    from compare_k1_relion_recovar_fine_operands import (  # type: ignore[no-redef]
        _json_default,
        _metric,
    )
    from validate_relion_bpref_factor_capture import (  # type: ignore[no-redef]
        load_factor_capture,
    )
    from validate_relion_fine_operand_capture import (  # type: ignore[no-redef]
        load_fine_operand_capture,
    )
    from validate_relion_preprocess_capture import (  # type: ignore[no-redef]
        load_artifact as load_preprocess_capture,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_particle(value: str) -> tuple[int, int]:
    try:
        stack_text, original_text = value.split(":", maxsplit=1)
        return int(stack_text), int(original_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("particle must be STACK:ZERO_BASED_ORIGINAL") from error


def _particle_table(path: Path):
    document = starfile.read(path)
    return document["particles"] if isinstance(document, dict) else document


def _model_average_norm(path: Path) -> float:
    document = starfile.read(path)
    general = document["model_general"]
    return float(general["rlnNormCorrectionAverage"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--recovar-dir", type=Path, required=True)
    parser.add_argument("--particles-mrcs", type=Path, required=True)
    parser.add_argument("--source-state-star", type=Path, required=True)
    parser.add_argument("--source-model-star", type=Path, required=True)
    parser.add_argument("--native-preprocess-capture-dir", type=Path)
    parser.add_argument("--particle", type=_parse_particle, action="append", required=True)
    parser.add_argument("--physical-image-size", type=int, required=True)
    parser.add_argument("--pixel-size", type=float, required=True)
    parser.add_argument("--particle-diameter", type=float, required=True)
    parser.add_argument("--mask-edge-pixels", type=float, default=5.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()

    devices = jax.devices("gpu")
    _require(len(devices) == 1, "preprocessing replay requires exactly one visible GPU")
    table = _particle_table(args.source_state_star)
    identities = table["rlnImageName"].astype(str)
    average_norm = _model_average_norm(args.source_model_star)
    with mrcfile.open(args.particles_mrcs, permissive=False) as stack_file:
        raw_stack = np.asarray(stack_file.data)

    raw_images = []
    normalization_factors = []
    integer_shift_rows = []
    records = []
    native_targets = []
    native_target_rows = []
    compact_indices = []
    native_preprocess_captures = []
    for stack, original in args.particle:
        identity_rows = np.flatnonzero(identities.str.startswith(f"{stack}@").to_numpy())
        _require(identity_rows.size == 1, f"stack {stack}: source state identity is not unique")
        state_row = table.iloc[int(identity_rows[0])]
        normalization = np.float32(average_norm / float(state_row["rlnNormCorrection"]))
        origins_pixels = np.asarray(
            [state_row["rlnOriginXAngst"], state_row["rlnOriginYAngst"]],
            dtype=np.float64,
        ) / float(args.pixel_size)
        # RELION translates the normalized real-space image by ROUND(old_offset)
        # before applying its soft mask.  The selected values are not half-way
        # cases, so rint and RELION's ROUND macro have the same result here.
        integer_shift = np.rint(origins_pixels).astype(np.int32)
        preprocess_capture = None
        if args.native_preprocess_capture_dir is not None:
            matches = sorted(
                args.native_preprocess_capture_dir.glob(f"*_stack{stack}.preprocess-v1.bin")
            )
            _require(len(matches) == 1, f"stack {stack}: expected one native preprocess capture")
            preprocess_capture = load_preprocess_capture(matches[0])
            _require(preprocess_capture.iteration == 2, f"stack {stack}: capture iteration changed")
            normalization = np.float32(preprocess_capture.norm_correction)
            integer_shift = np.asarray(preprocess_capture.old_offset[:2], dtype=np.int32)
        fine_path = next(args.capture_dir.glob(f"*_stack{stack}_class1.fine-operand-v1.bin"))
        factor_path = next(args.capture_dir.glob(f"*_stack{stack}_img0_class1.bpre-v2.bin"))
        recovar_path = next(args.recovar_dir.glob(f"pass2_orig{original:06d}_cs*.npz"))
        fine = load_fine_operand_capture(fine_path)
        factor = load_factor_capture(factor_path)
        pixels = fine.pixels.reshape(fine.candidates.size, fine.image_size)[0]
        native_image = (
            np.asarray(pixels["image_real"], dtype=np.float32)
            + np.complex64(1j) * np.asarray(pixels["image_imag"], dtype=np.float32)
        ).astype(np.complex64)
        with np.load(recovar_path, allow_pickle=False) as recovar:
            compact = np.asarray(recovar["window_indices"], dtype=np.int32)
            direct_ctf = (
                np.asarray(recovar["direct_ctf_rfloat_score"], dtype=np.float32)
                if "direct_ctf_rfloat_score" in recovar.files
                else None
            )
        current_size = next(
            size
            for size in range(2, args.physical_image_size + 1, 2)
            if size * (size // 2 + 1) == fine.image_size
        )
        full_to_compact = _relion_cuda_fine_full_to_compact_lookup(
            (args.physical_image_size, args.physical_image_size),
            current_size,
            compact,
        )
        supported_full = np.flatnonzero(full_to_compact >= 0)
        supported_compact = full_to_compact[supported_full]
        _require(
            np.array_equal(np.sort(supported_compact), np.arange(compact.size)),
            f"stack {stack}: fine/current-size support does not cover score window",
        )
        native_rows = np.empty(compact.size, dtype=np.int32)
        native_rows[supported_compact] = supported_full
        if factor.pixels.shape == pixels.shape:
            _require(
                np.array_equal(factor.pixels["x"], pixels["x"])
                and np.array_equal(factor.pixels["y"], pixels["y"]),
                f"stack {stack}: native factor/fine pixel coordinates changed",
            )
            native_processed = np.empty(compact.size, dtype=np.complex64)
            native_processed[supported_compact] = (
                native_image[supported_full] * factor.pixels["ctf"][supported_full]
            ).astype(np.complex64)
            native_ctf_source = "native_bpref_factor_capture"
        else:
            _require(
                factor.pixels.size == 0 and direct_ctf is not None,
                f"stack {stack}: incomplete native factor capture has no CTF fallback",
            )
            _require(
                direct_ctf.shape == compact.shape,
                f"stack {stack}: direct CTF/score window topology changed",
            )
            native_processed = np.empty(compact.size, dtype=np.complex64)
            native_processed[supported_compact] = (
                native_image[supported_full] * direct_ctf[supported_compact]
            ).astype(np.complex64)
            native_ctf_source = "recovar_source_faithful_direct_ctf_fallback"
        raw_images.append(np.asarray(raw_stack[stack - 1], dtype=np.float32))
        normalization_factors.append(normalization)
        integer_shift_rows.append(integer_shift)
        native_targets.append(native_processed)
        native_target_rows.append(native_rows)
        compact_indices.append(compact)
        native_preprocess_captures.append(preprocess_capture)
        records.append(
            {
                "stack_index_one_based": stack,
                "original_index_zero_based": original,
                "normalization_factor": float(normalization),
                "origin_pixels": origins_pixels.tolist(),
                "integer_shift": integer_shift.tolist(),
                "normalization_source": (
                    "native_preprocess_capture"
                    if preprocess_capture is not None
                    else "serialized_star"
                ),
                "native_ctf_source": native_ctf_source,
                "native_preprocess_capture": (
                    str(preprocess_capture.path.resolve())
                    if preprocess_capture is not None
                    else None
                ),
                "fine_capture": str(fine_path.resolve()),
                "factor_capture": str(factor_path.resolve()),
                "recovar_capture": str(recovar_path.resolve()),
            }
        )

    raw = jnp.asarray(np.stack(raw_images), dtype=jnp.float32)
    normalization = jnp.asarray(normalization_factors, dtype=jnp.float32)
    integer_shifts = jnp.asarray(np.stack(integer_shift_rows), dtype=jnp.int32)
    radius = args.particle_diameter / (2.0 * args.pixel_size)
    replay_arrays = {}
    replay_real_arrays = {}
    for native_lane in (False, True):
        normalized_real, processed_real = relion_preprocess_real_f32(
            raw,
            normalization,
            integer_shifts,
            radius,
            args.mask_edge_pixels,
            True,
            native_lane_reduction=native_lane,
        )
        processed_half = _centered_rfft2_jax(processed_real).reshape(len(records), -1)
        processed_half = np.asarray(jax.block_until_ready(processed_half), dtype=np.complex64)
        mode = "relion_cuda_native_lane" if native_lane else "relion_cuda"
        replay_arrays[mode] = processed_half
        replay_real_arrays[mode] = (
            np.asarray(jax.block_until_ready(normalized_real), dtype=np.float32),
            np.asarray(jax.block_until_ready(processed_real), dtype=np.float32),
        )

    n2 = np.float32(args.physical_image_size**2)
    dump = {}
    for row, (record, native_compact, native_rows, compact, preprocess_capture) in enumerate(
        zip(
            records,
            native_targets,
            native_target_rows,
            compact_indices,
            native_preprocess_captures,
            strict=True,
        )
    ):
        record["metrics"] = {}
        if preprocess_capture is not None:
            record["native_repeat_metrics"] = {
                "raw_input": _metric(
                    preprocess_capture.raw_input_real.reshape(args.physical_image_size, -1),
                    raw_images[row].reshape(args.physical_image_size, -1),
                ),
                "fine_processed_fourier": _metric(
                    native_compact,
                    preprocess_capture.masked_fourier_post_optics.reshape(-1)[native_rows],
                ),
            }
        dump[f"stack{record['stack_index_one_based']}_native_processed"] = native_compact
        for mode, replay in replay_arrays.items():
            replay_compact = (replay[row, compact] / n2).astype(np.complex64)
            record["metrics"][mode] = _metric(native_compact, replay_compact)
            if preprocess_capture is not None:
                normalized_real, processed_real = replay_real_arrays[mode]
                record["metrics"][mode]["native_real_stages"] = {
                    "normalized_shifted": _metric(
                        preprocess_capture.normalized_shifted_real.reshape(
                            args.physical_image_size, -1
                        ),
                        normalized_real[row].reshape(args.physical_image_size, -1),
                    ),
                    "masked": _metric(
                        preprocess_capture.masked_real.reshape(args.physical_image_size, -1),
                        processed_real[row].reshape(args.physical_image_size, -1),
                    ),
                }
            dump[f"stack{record['stack_index_one_based']}_{mode}"] = replay_compact

    report = {
        "schema": "recovar.em.k1_relion_cuda_preprocess_replay.v1",
        "status": "complete",
        "gpu_device": str(devices[0]),
        "physical_image_size": args.physical_image_size,
        "particle_diameter_angstrom": args.particle_diameter,
        "mask_edge_pixels": args.mask_edge_pixels,
        "source_average_norm_correction": average_norm,
        "particles": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **dump)
    rendered = json.dumps(report, indent=2, sort_keys=True, default=_json_default)
    args.output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
