#!/usr/bin/env python3
"""Compare one RECOVAR BPref aggregate with the matching native RELION stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_bpref_substitution import (
    load_relion_raw,
    relion_raw_to_recovar_full,
)
from scripts.compare_iter1_bpref_accum import (
    _apply_recovar_frame,
    downsample_recovar_accumulator,
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


def _load_native_bpref(path: Path, *, value_dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Load either the state-v1 three-shape header or the older raw header."""

    dtype = np.dtype(value_dtype)
    with path.open("rb") as stream:
        shape = np.fromfile(stream, dtype=np.int64, count=3)
    if shape.size == 3 and np.all(shape > 0):
        count = int(np.prod(shape, dtype=np.int64))
        if path.stat().st_size == 3 * np.dtype(np.int64).itemsize + count * dtype.itemsize:
            with path.open("rb") as stream:
                actual_shape = np.fromfile(stream, dtype=np.int64, count=3)
                values = np.fromfile(stream, dtype=dtype, count=count)
            _require(np.array_equal(actual_shape, shape), "native state-v1 shape changed while reading")
            return shape, values.reshape(tuple(int(value) for value in shape))
    return load_relion_raw(path, value_dtype=dtype)


def _metric(source: np.ndarray, target: np.ndarray, *, allow_sign: bool) -> dict[str, Any]:
    source = np.asarray(source).reshape(-1)
    target = np.asarray(target).reshape(-1)
    _require(source.shape == target.shape and source.size > 0, "metric topology mismatch")
    _require(np.all(np.isfinite(source)) and np.all(np.isfinite(target)), "non-finite metric values")
    target_norm = float(np.linalg.norm(target))
    source_energy = float(np.vdot(source, source).real)
    _require(target_norm > 0.0 and source_energy > 0.0, "metric has zero energy")
    sign = 1
    if allow_sign and np.linalg.norm(-source - target) < np.linalg.norm(source - target):
        sign = -1
    source = sign * source
    relative_l2 = float(np.linalg.norm(source - target) / target_norm)
    scale = float(np.vdot(source, target).real / source_energy)
    scaled_relative_l2 = float(np.linalg.norm(scale * source - target) / target_norm)
    return {
        "sign_applied_to_source": sign,
        "relative_l2": relative_l2,
        "source_to_target_least_squares_scale": scale,
        "relative_l2_after_scale": scaled_relative_l2,
        "max_absolute": float(np.max(np.abs(source - target))),
    }


def _load_recovar(path: Path, *, half: int = 1) -> dict[str, Any]:
    _require(half in (1, 2), "half must be 1 or 2")
    half_index = half - 1
    with np.load(path, allow_pickle=False) as archive:
        schema = str(np.asarray(archive["schema"]).item())
        _require(
            schema in {"recovar-bpref-prejoin-v2", "recovar-bpref-accum-v2"},
            "unexpected RECOVAR BPref schema",
        )
        return {
            "stage": "pre_lowres_join" if schema == "recovar-bpref-prejoin-v2" else "post_lowres_join",
            "iteration": int(archive["iteration"]),
            "current_size": int(archive["current_size"]),
            "padding_factor": int(archive["padding_factor"]),
            "grid_size": int(archive["grid_size"]),
            "volume_shape": tuple(int(value) for value in archive["volume_shape"]),
            "accumulator_shape": tuple(
                int(value) for value in archive["mstep_accumulator_shape"]
            ),
            "half": half,
            "numerator": np.asarray(archive[f"Ft_y_{half_index}"]),
            "weight": np.asarray(archive[f"Ft_ctf_{half_index}"]),
        }


def _downsample(
    numerator: np.ndarray,
    weight: np.ndarray,
    *,
    grid_size: int,
    volume_shape: tuple[int, int, int],
    accumulator_shape: tuple[int, int, int],
    padding_factor: int,
    max_shell: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    average, down_weight, radius, _, _, actual_max_shell = downsample_recovar_accumulator(
        numerator,
        weight,
        volume_shape,
        padding_factor,
        max_shell,
        accumulator_shape=accumulator_shape,
    )
    _require(actual_max_shell == max_shell, "downsample max-shell mismatch")
    average, down_weight = _apply_recovar_frame(
        average,
        down_weight,
        grid_size=grid_size,
        recovar_frame="relion",
    )
    return average, down_weight, radius


def _comparison(
    source_average: np.ndarray,
    source_weight: np.ndarray,
    target_average: np.ndarray,
    target_weight: np.ndarray,
    *,
    radius: int,
    first_shell: int,
    max_shell: int,
) -> dict[str, Any]:
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    x_axis = np.arange(source_average.shape[2], dtype=np.float64)
    z, y, x = np.meshgrid(axis, axis, x_axis, indexing="ij")
    radial = np.sqrt(x * x + y * y + z * z)
    shell = np.where(radial >= 0.0, np.floor(radial + 0.5), 0).astype(np.int64)
    selected = (radial <= float(max_shell)) & (shell >= int(first_shell))
    _require(np.any(selected), "selected shell band is empty")
    source_numerator = source_average * source_weight
    target_numerator = target_average * target_weight
    source_support = source_weight > 0.0
    target_support = target_weight > 0.0
    support_union = np.count_nonzero((source_support | target_support) & selected)
    report = {
        "shell_band": [int(first_shell), int(max_shell)],
        "coordinate_count": int(np.count_nonzero(selected)),
        "support_mismatch_count": int(
            np.count_nonzero((source_support != target_support) & selected)
        ),
        "support_jaccard": float(
            np.count_nonzero(source_support & target_support & selected)
            / max(int(support_union), 1)
        ),
        "average": _metric(source_average[selected], target_average[selected], allow_sign=True),
        "numerator": _metric(source_numerator[selected], target_numerator[selected], allow_sign=True),
        "denominator": _metric(source_weight[selected], target_weight[selected], allow_sign=False),
        "per_shell": {},
    }
    for shell_index in range(int(first_shell), int(max_shell) + 1):
        shell_mask = selected & (shell == shell_index)
        if not np.any(shell_mask):
            continue
        report["per_shell"][str(shell_index)] = {
            "numerator_relative_l2": _metric(
                source_numerator[shell_mask], target_numerator[shell_mask], allow_sign=True
            )["relative_l2"],
            "denominator_relative_l2": _metric(
                source_weight[shell_mask], target_weight[shell_mask], allow_sign=False
            )["relative_l2"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-prejoin", type=Path, required=True)
    parser.add_argument("--recovar-repeat", type=Path)
    parser.add_argument("--native-data", type=Path, required=True)
    parser.add_argument("--native-weight", type=Path, required=True)
    parser.add_argument("--native-repeat-data", type=Path)
    parser.add_argument("--native-repeat-weight", type=Path)
    parser.add_argument("--half", type=int, choices=(1, 2), default=1)
    parser.add_argument("--expected-local-iteration", type=int, default=2)
    parser.add_argument("--physical-iteration", type=int, default=2)
    parser.add_argument("--first-shell", type=int, default=15)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    recovar = _load_recovar(args.recovar_prejoin, half=args.half)
    _require(
        recovar["iteration"] == args.expected_local_iteration,
        "RECOVAR dump local-iteration label differs",
    )
    max_shell = recovar["current_size"] // 2
    native_data_header, native_data = _load_native_bpref(
        args.native_data, value_dtype=np.complex128
    )
    native_weight_header, native_weight = _load_native_bpref(
        args.native_weight, value_dtype=np.float64
    )
    _require(np.array_equal(native_data_header, native_weight_header), "native raw headers differ")
    native_numerator, native_denominator = relion_raw_to_recovar_full(
        native_data, native_weight, grid_size=recovar["grid_size"]
    )
    rec_average, rec_weight, radius = _downsample(
        recovar["numerator"], recovar["weight"], max_shell=max_shell, **{
            key: recovar[key]
            for key in ("grid_size", "volume_shape", "accumulator_shape", "padding_factor")
        }
    )
    native_average, native_down_weight, native_radius = _downsample(
        native_numerator, native_denominator, max_shell=max_shell, **{
            key: recovar[key]
            for key in ("grid_size", "volume_shape", "accumulator_shape", "padding_factor")
        }
    )
    _require(radius == native_radius, "downsample radii differ")
    report: dict[str, Any] = {
        "schema": "recovar.em.k1_half1_raw_accumulator.v1",
        "metric_policy": "scale-sensitive relative-L2 on matched BPref intermediates; no correlation",
        "bpref_stage": recovar["stage"],
        "physical_iteration": args.physical_iteration,
        "recovar_local_iteration_label": recovar["iteration"],
        "half": recovar["half"],
        "current_size": recovar["current_size"],
        "raw_accumulator": {
            "numerator": _metric(
                recovar["numerator"], native_numerator, allow_sign=True
            ),
            "denominator": _metric(
                recovar["weight"], native_denominator, allow_sign=False
            ),
        },
        "recovar_vs_native": _comparison(
            rec_average,
            rec_weight,
            native_average,
            native_down_weight,
            radius=radius,
            first_shell=args.first_shell,
            max_shell=max_shell,
        ),
        "artifacts": {
            str(path.resolve()): _sha256(path)
            for path in (args.recovar_prejoin, args.native_data, args.native_weight)
        },
    }
    if args.recovar_repeat is not None:
        repeat_recovar = _load_recovar(args.recovar_repeat, half=args.half)
        for key in (
            "stage",
            "iteration",
            "current_size",
            "padding_factor",
            "grid_size",
            "volume_shape",
            "accumulator_shape",
            "half",
        ):
            _require(
                repeat_recovar[key] == recovar[key],
                f"RECOVAR repeat metadata differs for {key}",
            )
        repeat_average, repeat_weight, repeat_radius = _downsample(
            repeat_recovar["numerator"],
            repeat_recovar["weight"],
            max_shell=max_shell,
            **{
                key: repeat_recovar[key]
                for key in (
                    "grid_size",
                    "volume_shape",
                    "accumulator_shape",
                    "padding_factor",
                )
            },
        )
        _require(radius == repeat_radius, "RECOVAR repeat downsample radius differs")
        report["recovar_vs_recovar_repeat"] = {
            "raw_accumulator": {
                "numerator": _metric(
                    recovar["numerator"],
                    repeat_recovar["numerator"],
                    allow_sign=True,
                ),
                "denominator": _metric(
                    recovar["weight"],
                    repeat_recovar["weight"],
                    allow_sign=False,
                ),
            },
            "downsampled": _comparison(
                rec_average,
                rec_weight,
                repeat_average,
                repeat_weight,
                radius=radius,
                first_shell=args.first_shell,
                max_shell=max_shell,
            ),
        }
        report["artifacts"][str(args.recovar_repeat.resolve())] = _sha256(
            args.recovar_repeat
        )
    if (args.native_repeat_data is None) != (args.native_repeat_weight is None):
        raise ValueError("native repeat data and weight must be supplied together")
    if args.native_repeat_data is not None and args.native_repeat_weight is not None:
        repeat_data_header, repeat_data = _load_native_bpref(
            args.native_repeat_data, value_dtype=np.complex128
        )
        repeat_weight_header, repeat_weight = _load_native_bpref(
            args.native_repeat_weight, value_dtype=np.float64
        )
        _require(np.array_equal(repeat_data_header, repeat_weight_header), "repeat raw headers differ")
        repeat_numerator, repeat_denominator = relion_raw_to_recovar_full(
            repeat_data, repeat_weight, grid_size=recovar["grid_size"]
        )
        repeat_average, repeat_down_weight, repeat_radius = _downsample(
            repeat_numerator, repeat_denominator, max_shell=max_shell, **{
                key: recovar[key]
                for key in ("grid_size", "volume_shape", "accumulator_shape", "padding_factor")
            }
        )
        _require(radius == repeat_radius, "repeat downsample radius differs")
        report["native_vs_native_repeat"] = _comparison(
            repeat_average,
            repeat_down_weight,
            native_average,
            native_down_weight,
            radius=radius,
            first_shell=args.first_shell,
            max_shell=max_shell,
        )
        report["artifacts"].update(
            {
                str(path.resolve()): _sha256(path)
                for path in (args.native_repeat_data, args.native_repeat_weight)
            }
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
