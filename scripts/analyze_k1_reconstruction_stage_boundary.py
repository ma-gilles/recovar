#!/usr/bin/env python3
"""Locate the first K=1 reconstruction-stage mismatch against native RELION."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils, mask, padding
from recovar.reconstruction import relion_functions
from recovar.utils import helpers


def _load(path: Path, dtype: np.dtype) -> np.ndarray:
    raw = path.read_bytes()
    shape = struct.unpack_from("<3q", raw)
    values = np.frombuffer(raw, dtype=dtype, offset=64).copy()
    if values.size != int(np.prod(shape)):
        raise ValueError(f"payload size mismatch for {path}")
    return values.reshape(shape)


def _stage_path(root: Path, half: int, stage: str, call_index: int = 0) -> Path:
    paths = sorted(
        root.glob(
            f"reconstruct_rank{half:02d}_pid*_call{call_index:04d}_{stage}.bin"
        )
    )
    if len(paths) != 1:
        raise ValueError(
            f"expected one half-{half} call-{call_index} {stage} dump, found {paths}"
        )
    return paths[0]


def _public_full_to_relion_half(values: np.ndarray, side: int) -> np.ndarray:
    public = np.asarray(values).reshape((side, side, side))
    centered = public.transpose(2, 1, 0)[:, :, side // 2 :]
    return np.ascontiguousarray(np.fft.ifftshift(centered, axes=(0, 1)))


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    delta = actual - expected
    denominator = float(np.linalg.norm(expected.reshape(-1)))
    return {
        "count": int(expected.size),
        "bitwise_equal_count": int(np.count_nonzero(actual == expected)),
        "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / denominator),
        "max_absolute": float(np.max(np.abs(delta))),
    }


def _native_floor(regularized: np.ndarray, radial_floor: np.ndarray, padding_factor: int) -> np.ndarray:
    side = regularized.shape[0]
    freq = np.fft.fftfreq(side) * side
    kz, iy, jx = np.meshgrid(freq, freq, np.arange(regularized.shape[2]), indexing="ij")
    shells = np.floor(np.sqrt(kz * kz + iy * iy + jx * jx) / float(padding_factor)).astype(np.int64)
    shells = np.minimum(shells, radial_floor.size - 1)
    return np.maximum(regularized, radial_floor[shells])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-accumulator", type=Path, required=True)
    parser.add_argument("--native-stage-dir", type=Path, required=True)
    parser.add_argument("--native-call-index", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.native_call_index < 0:
        raise ValueError("native call index must be non-negative")

    with np.load(args.recovar_accumulator, allow_pickle=False) as archive:
        grid_size = int(archive["grid_size"])
        current_size = int(archive["current_size"])
        padding_factor = int(archive["padding_factor"])
        volume_shape = tuple(int(value) for value in archive["volume_shape"])
        accumulator_shape = tuple(int(value) for value in archive["mstep_accumulator_shape"])
        numerators = [np.asarray(archive[f"Ft_y_{index}"]) for index in (0, 1)]
        weights = [np.asarray(archive[f"Ft_ctf_{index}"]) for index in (0, 1)]

    accumulator_side = accumulator_shape[0]
    if accumulator_shape != (accumulator_side,) * 3:
        raise ValueError(f"non-cubic accumulator {accumulator_shape}")
    n2 = float(grid_size**2)
    n4 = float(grid_size**4)
    wiener_radius = padding_factor * (current_size // 2)
    valid = relion_functions._relion_current_size_decenter_mask(
        accumulator_shape, wiener_radius, half_volume=False
    ).reshape(-1)
    reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
        volume_shape, padding_factor
    )

    halves: dict[str, object] = {}
    for half in (1, 2):
        native_tau = _load(
            _stage_path(
                args.native_stage_dir, half, "tau2", args.native_call_index
            ),
            np.dtype("<f8"),
        ).reshape(-1)
        native_weight = _load(
            _stage_path(
                args.native_stage_dir,
                half,
                "fweight_decentered",
                args.native_call_index,
            ),
            np.dtype("<f8"),
        )
        native_regularized = _load(
            _stage_path(
                args.native_stage_dir,
                half,
                "fweight_regularized",
                args.native_call_index,
            ),
            np.dtype("<f8"),
        )
        native_radial_floor = _load(
            _stage_path(
                args.native_stage_dir,
                half,
                "radavg_weight",
                args.native_call_index,
            ),
            np.dtype("<f8"),
        ).reshape(-1)
        native_denominator = _native_floor(native_regularized, native_radial_floor, padding_factor)
        native_divided = _load(
            _stage_path(
                args.native_stage_dir,
                half,
                "fconv_divided",
                args.native_call_index,
            ),
            np.dtype("<c16"),
        )
        native_before = helpers.relion_volume_to_recovar(
            _load(
                _stage_path(
                    args.native_stage_dir,
                    half,
                    "volume_before_gridding",
                    args.native_call_index,
                ),
                np.dtype("<f8"),
            )
        )
        native_after = helpers.relion_volume_to_recovar(
            _load(
                _stage_path(
                    args.native_stage_dir,
                    half,
                    "volume_after_gridding",
                    args.native_call_index,
                ),
                np.dtype("<f8"),
            )
        )

        weight = jnp.asarray(weights[half - 1])
        numerator = jnp.asarray(numerators[half - 1])

        def replay(
            tau_recovar: jnp.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            denominator = relion_functions.adjust_regularization_relion_style(
                weight.real,
                accumulator_shape,
                tau=tau_recovar,
                padding_factor=padding_factor,
                max_res_shell=current_size // 2,
                half_volume=False,
                tau2_fudge=1.0,
                minres_map=5,
                relion_native_shell_floor=True,
                native_volume_shape=volume_shape,
                tau_is_1d=True,
            )
            divided = numerator * valid.astype(numerator.real.dtype) / denominator
            divided_half = fourier_transform_utils.full_volume_to_half_volume(
                divided.reshape(accumulator_shape), accumulator_shape
            )
            divided_half = relion_functions._relion_window_centered_half_fourier(
                divided_half, accumulator_shape, reconstruction_shape
            )
            before = fourier_transform_utils.get_idft3_real(
                divided_half,
                volume_shape=reconstruction_shape,
            )
            before = padding.unpad_volume_spatial_domain(
                before, reconstruction_shape[0] - volume_shape[0]
            )
            before, _ = mask.soft_mask_outside_map(before, cosine_width=3)
            after, _ = relion_functions.griddingCorrect(
                before.reshape(volume_shape), volume_shape[0], padding_factor, order=1
            )
            return np.asarray(denominator), np.asarray(divided), np.asarray(before), np.asarray(after)

        tau_recovar_exact = jnp.asarray(native_tau * n4, dtype=jnp.float64)
        denominator, divided, before_np, after_np = replay(tau_recovar_exact)
        tau_recovar_f32 = jnp.asarray(
            np.asarray(native_tau * n4, dtype=np.float32), dtype=jnp.float64
        )
        rounded_denominator, rounded_divided, rounded_before, rounded_after = replay(tau_recovar_f32)

        rec_weight_native = _public_full_to_relion_half(
            np.asarray(weight * valid.astype(weight.dtype)), accumulator_side
        ) * n4
        rec_denominator_native = _public_full_to_relion_half(
            denominator, accumulator_side
        ) * n4
        rec_divided_native = _public_full_to_relion_half(
            divided, accumulator_side
        ) / (-n2)

        halves[str(half)] = {
            "native_tau_head": native_tau[:8].tolist(),
            "stages": {
                "fweight_decentered": _metrics(rec_weight_native, native_weight),
                "wiener_denominator_after_floor": _metrics(
                    rec_denominator_native, native_denominator
                ),
                "fconv_divided": _metrics(rec_divided_native, native_divided),
                "volume_before_gridding": _metrics(before_np, native_before),
                "volume_after_gridding": _metrics(after_np, native_after),
            },
            "float32_rounded_tau_replay": {
                "wiener_denominator_after_floor": _metrics(
                    _public_full_to_relion_half(rounded_denominator, accumulator_side) * n4,
                    native_denominator,
                ),
                "fconv_divided": _metrics(
                    _public_full_to_relion_half(rounded_divided, accumulator_side) / (-n2),
                    native_divided,
                ),
                "volume_before_gridding": _metrics(rounded_before, native_before),
                "volume_after_gridding": _metrics(rounded_after, native_after),
            },
        }

    report = {
        "schema": "recovar.em.k1_reconstruction_stage_boundary.v1",
        "recovar_accumulator": str(args.recovar_accumulator.resolve()),
        "native_stage_dir": str(args.native_stage_dir.resolve()),
        "native_call_index": args.native_call_index,
        "metric_policy": "scale-sensitive relative-L2; no fitted rescaling",
        "halves": halves,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
