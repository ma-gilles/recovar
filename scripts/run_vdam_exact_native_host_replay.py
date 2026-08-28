#!/usr/bin/env python3
"""Execute one packed VDAM M-step in a fresh native CUDA process."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np


class ReplayArguments(ctypes.Structure):
    _fields_ = [
        ("projector_full", ctypes.c_void_p),
        ("images", ctypes.c_void_p),
        ("ctf", ctypes.c_void_p),
        ("minvsigma2", ctypes.c_void_p),
        ("posterior_over_weight_norm", ctypes.c_void_p),
        ("translation_angles", ctypes.c_void_p),
        ("projector_eulers", ctypes.c_void_p),
        ("compact_rotations", ctypes.c_void_p),
        ("reconstruction_group_ids", ctypes.c_void_p),
        ("worker_lane_ids", ctypes.c_void_p),
        ("particle_trace_ids", ctypes.c_void_p),
        ("rotation_replay_order", ctypes.c_void_p),
        ("rotation_replay_counts", ctypes.c_void_p),
        ("particle_start_offsets_ns", ctypes.c_void_p),
        ("data_real_volume", ctypes.c_void_p),
        ("data_imag_volume", ctypes.c_void_p),
        ("weight_volume", ctypes.c_void_p),
        ("denominator_sum", ctypes.c_void_p),
        ("quiesced_prelaunch_data_real", ctypes.c_void_p),
        ("quiesced_prelaunch_data_imag", ctypes.c_void_p),
        ("quiesced_prelaunch_weight", ctypes.c_void_p),
        ("quiesced_prelaunch_found", ctypes.c_void_p),
        ("quiesced_prelaunch_particle_row", ctypes.c_void_p),
        ("quiesced_prelaunch_worker_lane", ctypes.c_void_p),
        ("quiesced_prelaunch_reconstruction_group", ctypes.c_void_p),
        ("projector_size", ctypes.c_int64),
        ("n_particles", ctypes.c_int64),
        ("rotation_count", ctypes.c_int64),
        ("translation_count", ctypes.c_int64),
        ("pixel_count", ctypes.c_int64),
        ("image_h", ctypes.c_int64),
        ("image_w", ctypes.c_int64),
        ("volume_n0", ctypes.c_int64),
        ("volume_n1", ctypes.c_int64),
        ("volume_n2", ctypes.c_int64),
        ("upsampling", ctypes.c_int64),
        ("max_r2_x4", ctypes.c_int64),
        ("physical_image_size", ctypes.c_int32),
        ("projector_max_r", ctypes.c_int32),
        ("projection_padding_factor", ctypes.c_int32),
        ("reconstruction_group_count", ctypes.c_int32),
        ("parallel_worker_replay", ctypes.c_int32),
        ("quiesced_prelaunch_target_particle_id", ctypes.c_int64),
    ]


def _array(bundle, name: str, dtype, shape=None) -> np.ndarray:
    value = np.ascontiguousarray(bundle[name], dtype=dtype)
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
    return value


def _scalar(bundle, name: str) -> int:
    value = np.asarray(bundle[name])
    if value.size != 1:
        raise ValueError(f"{name} must be scalar")
    return int(value.reshape(()))


def _pointer(value: np.ndarray) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(value.ctypes.data))


def _optional_pointer(value: np.ndarray | None) -> ctypes.c_void_p:
    return ctypes.c_void_p() if value is None else _pointer(value)


def run_replay(input_path: Path, output_path: Path, library_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not os.environ.get("RECOVAR_VDAM_EXACT_NATIVE_PTX", "").strip():
        raise RuntimeError("RECOVAR_VDAM_EXACT_NATIVE_PTX is required")
    if not library_path.is_file():
        raise FileNotFoundError(library_path)
    prelaunch_capture_dir_text = os.environ.get(
        "RECOVAR_VDAM_QUIESCED_PRELAUNCH_CAPTURE_DIR", ""
    ).strip()
    prelaunch_target_text = os.environ.get(
        "RECOVAR_VDAM_QUIESCED_PRELAUNCH_PARTICLE_ID", ""
    ).strip()
    if bool(prelaunch_capture_dir_text) != bool(prelaunch_target_text):
        raise RuntimeError(
            "quiesced prelaunch capture requires both directory and particle ID"
        )
    prelaunch_target = -1
    if prelaunch_target_text:
        try:
            prelaunch_target = int(prelaunch_target_text)
        except ValueError as exc:
            raise ValueError(
                "quiesced prelaunch particle ID must be an integer"
            ) from exc
        if prelaunch_target < 0:
            raise ValueError("quiesced prelaunch particle ID must be nonnegative")

    with np.load(input_path, allow_pickle=False) as bundle:
        projector_size = _scalar(bundle, "projector_size")
        n_particles = _scalar(bundle, "n_particles")
        rotation_count = _scalar(bundle, "rotation_count")
        translation_count = _scalar(bundle, "translation_count")
        pixel_count = _scalar(bundle, "pixel_count")
        image_h = _scalar(bundle, "image_h")
        image_w = _scalar(bundle, "image_w")
        volume_n0 = _scalar(bundle, "volume_n0")
        volume_n1 = _scalar(bundle, "volume_n1")
        volume_n2 = _scalar(bundle, "volume_n2")
        reconstruction_group_count = _scalar(bundle, "reconstruction_group_count")

        projector = _array(
            bundle,
            "projector_full",
            np.complex64,
            (projector_size, projector_size, projector_size),
        )
        images = _array(bundle, "images", np.complex64, (n_particles, pixel_count))
        ctf = _array(bundle, "ctf", np.float32, images.shape)
        minvsigma2 = _array(bundle, "minvsigma2", np.float32, images.shape)
        posterior = _array(
            bundle,
            "posterior_over_weight_norm",
            np.float32,
            (n_particles, rotation_count, translation_count),
        )
        translations = _array(
            bundle, "translation_angles", np.float32, (translation_count, 2)
        )
        eulers = _array(
            bundle,
            "projector_eulers",
            np.float32,
            (n_particles, rotation_count, 9),
        )
        compact_rotations = _array(
            bundle,
            "compact_rotations",
            np.float32,
            (n_particles, rotation_count, 6),
        )
        particle_shape = (n_particles,)
        reconstruction_groups = _array(
            bundle, "reconstruction_group_ids", np.int32, particle_shape
        )
        worker_lanes = _array(bundle, "worker_lane_ids", np.int32, particle_shape)
        particle_trace_ids = _array(
            bundle, "particle_trace_ids", np.int32, particle_shape
        )
        rotation_order = _array(
            bundle,
            "rotation_replay_order",
            np.int32,
            (n_particles, rotation_count),
        )
        rotation_counts = _array(
            bundle, "rotation_replay_counts", np.int32, particle_shape
        )
        particle_offsets = _array(
            bundle, "particle_start_offsets_ns", np.int32, particle_shape
        )
        accumulator_shape = (
            reconstruction_group_count,
            volume_n0 * volume_n1 * (volume_n2 // 2 + 1),
        )
        data_real = _array(bundle, "data_real_volume", np.float32, accumulator_shape)
        data_imag = _array(bundle, "data_imag_volume", np.float32, accumulator_shape)
        weight = _array(bundle, "weight_volume", np.float32, accumulator_shape)
        prelaunch_shape = (accumulator_shape[1],)
        prelaunch_real = (
            np.empty(prelaunch_shape, dtype=np.float32)
            if prelaunch_target >= 0
            else None
        )
        prelaunch_imag = (
            np.empty(prelaunch_shape, dtype=np.float32)
            if prelaunch_target >= 0
            else None
        )
        prelaunch_weight = (
            np.empty(prelaunch_shape, dtype=np.float32)
            if prelaunch_target >= 0
            else None
        )
        prelaunch_found = np.zeros(1, dtype=np.int32)
        prelaunch_particle_row = np.full(1, -1, dtype=np.int32)
        prelaunch_worker_lane = np.full(1, -1, dtype=np.int32)
        prelaunch_reconstruction_group = np.full(1, -1, dtype=np.int32)

        denominator = np.empty(
            (n_particles, rotation_count, pixel_count), dtype=np.float32
        )
        arguments = ReplayArguments(
            _pointer(projector),
            _pointer(images),
            _pointer(ctf),
            _pointer(minvsigma2),
            _pointer(posterior),
            _pointer(translations),
            _pointer(eulers),
            _pointer(compact_rotations),
            _pointer(reconstruction_groups),
            _pointer(worker_lanes),
            _pointer(particle_trace_ids),
            _pointer(rotation_order),
            _pointer(rotation_counts),
            _pointer(particle_offsets),
            _pointer(data_real),
            _pointer(data_imag),
            _pointer(weight),
            _pointer(denominator),
            _optional_pointer(prelaunch_real),
            _optional_pointer(prelaunch_imag),
            _optional_pointer(prelaunch_weight),
            _pointer(prelaunch_found),
            _pointer(prelaunch_particle_row),
            _pointer(prelaunch_worker_lane),
            _pointer(prelaunch_reconstruction_group),
            projector_size,
            n_particles,
            rotation_count,
            translation_count,
            pixel_count,
            image_h,
            image_w,
            volume_n0,
            volume_n1,
            volume_n2,
            _scalar(bundle, "upsampling"),
            _scalar(bundle, "max_r2_x4"),
            _scalar(bundle, "physical_image_size"),
            _scalar(bundle, "projector_max_r"),
            _scalar(bundle, "projection_padding_factor"),
            reconstruction_group_count,
            _scalar(bundle, "parallel_worker_replay"),
            prelaunch_target,
        )

    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    replay = library.recovar_relion_vdam_exact_native_host_replay
    replay.argtypes = [ctypes.POINTER(ReplayArguments)]
    replay.restype = ctypes.c_int
    error = int(replay(ctypes.byref(arguments)))
    if error != 0:
        raise RuntimeError(f"clean-process CUDA replay failed with error {error}")

    prelaunch_capture_path = None
    if int(prelaunch_found[0]) != 0:
        capture_dir = Path(prelaunch_capture_dir_text).expanduser().resolve()
        capture_dir.mkdir(parents=True, exist_ok=True)
        prelaunch_capture_path = capture_dir / (
            f"particle-{prelaunch_target}-quiesced-prelaunch.npz"
        )
        if prelaunch_capture_path.exists():
            raise FileExistsError(f"refusing to overwrite {prelaunch_capture_path}")
        np.savez(
            prelaunch_capture_path,
            schema=np.asarray("recovar.vdam_quiesced_prelaunch.v1"),
            target_particle_id=np.int64(prelaunch_target),
            particle_row=prelaunch_particle_row,
            worker_lane=prelaunch_worker_lane,
            reconstruction_group=prelaunch_reconstruction_group,
            data_real_volume=prelaunch_real,
            data_imag_volume=prelaunch_imag,
            weight_volume=prelaunch_weight,
        )

    np.savez(
        output_path,
        data_real_volume=data_real,
        data_imag_volume=data_imag,
        weight_volume=weight,
        denominator_sum=denominator,
    )
    return {
        "schema": "recovar.vdam_exact_native_host_replay.v1",
        "status": "complete",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "library": str(library_path.resolve()),
        "n_particles": n_particles,
        "rotation_count": rotation_count,
        "translation_count": translation_count,
        "pixel_count": pixel_count,
        "quiesced_prelaunch_found": bool(prelaunch_found[0]),
        "quiesced_prelaunch_capture": (
            None if prelaunch_capture_path is None else str(prelaunch_capture_path)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite {args.report}")
    report = run_replay(args.input, args.output, args.library)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
