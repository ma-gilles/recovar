#!/usr/bin/env python3
"""Compare one RECOVAR BPref aggregate with the matching native RELION stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from recovar.reconstruction import regularization
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
    mismatch_indices = np.flatnonzero(source != target)
    absolute_residual = np.abs(source - target)
    relative_l2 = float(np.linalg.norm(source - target) / target_norm)
    scale = float(np.vdot(source, target).real / source_energy)
    scaled_relative_l2 = float(np.linalg.norm(scale * source - target) / target_norm)
    return {
        "sign_applied_to_source": sign,
        "exact_equal": bool(mismatch_indices.size == 0),
        "mismatch_count": int(mismatch_indices.size),
        "first_mismatch_flat_index": (
            None if mismatch_indices.size == 0 else int(mismatch_indices[0])
        ),
        "absolute_residual_p95": float(np.percentile(absolute_residual, 95)),
        "relative_l2": relative_l2,
        "source_to_target_least_squares_scale": scale,
        "relative_l2_after_scale": scaled_relative_l2,
        "max_absolute": float(np.max(np.abs(source - target))),
    }


def _centered_real_inner_correlation(
    source: np.ndarray,
    target: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float:
    """Return a diagnostic centered correlation using the real inner product."""

    source = np.asarray(source).reshape(-1)
    target = np.asarray(target).reshape(-1)
    _require(source.shape == target.shape and source.size > 0, "correlation topology mismatch")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        _require(mask.shape == source.shape and np.any(mask), "correlation mask mismatch")
        source = source[mask]
        target = target[mask]
    _require(
        np.all(np.isfinite(source)) and np.all(np.isfinite(target)),
        "non-finite correlation values",
    )
    source = source - np.mean(source)
    target = target - np.mean(target)
    denominator = np.sqrt(
        np.vdot(source, source).real * np.vdot(target, target).real
    )
    _require(denominator > 0.0, "correlation has zero centered energy")
    return float(np.vdot(source, target).real / denominator)


def _paired_raw_residual_structure(
    recovar_numerator1: np.ndarray,
    native_numerator1: np.ndarray,
    recovar_denominator1: np.ndarray,
    native_denominator1: np.ndarray,
    recovar_numerator2: np.ndarray,
    native_numerator2: np.ndarray,
    recovar_denominator2: np.ndarray,
    native_denominator2: np.ndarray,
) -> dict[str, Any]:
    """Distinguish cross-half signal coherence from residual coherence."""

    numerator = _paired_field_residual_structure(
        recovar_numerator1,
        native_numerator1,
        recovar_numerator2,
        native_numerator2,
        allow_sign=True,
        support_union=(native_numerator1 != 0) | (native_numerator2 != 0),
    )
    denominator = _paired_field_residual_structure(
        recovar_denominator1,
        native_denominator1,
        recovar_denominator2,
        native_denominator2,
        allow_sign=False,
        support_union=(native_denominator1 != 0) | (native_denominator2 != 0),
    )
    return {
        "policy": (
            "diagnostic centered real-inner-product correlation; acceptance remains "
            "scale-sensitive relative-L2/FSC-AUC"
        ),
        "numerator_sign_applied_to_recovar": numerator["sign_applied_to_recovar"],
        "native_signal": {
            "numerator_centered_correlation": numerator[
                "native_signal_centered_correlation"
            ],
            "denominator_centered_correlation": denominator[
                "native_signal_centered_correlation"
            ],
        },
        "recovar_minus_native_residual": {
            "numerator_centered_correlation": numerator[
                "residual_centered_correlation"
            ],
            "numerator_centered_correlation_on_support_union": numerator[
                "residual_centered_correlation_on_support_union"
            ],
            "denominator_centered_correlation": denominator[
                "residual_centered_correlation"
            ],
            "denominator_centered_correlation_on_support_union": denominator[
                "residual_centered_correlation_on_support_union"
            ],
        },
    }


def _raw_accumulator_region_metrics(
    recovar_numerator: np.ndarray,
    native_numerator: np.ndarray,
    recovar_denominator: np.ndarray,
    native_denominator: np.ndarray,
    *,
    accumulator_shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Partition raw residual energy into the public x=0 plane and its complement."""

    expected_size = int(np.prod(accumulator_shape, dtype=np.int64))
    fields = {
        "numerator": (
            np.asarray(recovar_numerator).reshape(-1),
            np.asarray(native_numerator).reshape(-1),
            True,
        ),
        "denominator": (
            np.asarray(recovar_denominator).reshape(-1),
            np.asarray(native_denominator).reshape(-1),
            False,
        ),
    }
    x0 = np.zeros(accumulator_shape, dtype=bool)
    x0[accumulator_shape[0] // 2, :, :] = True
    masks = {"x0": x0.reshape(-1), "off_x0": (~x0).reshape(-1)}
    report: dict[str, Any] = {
        "x0_public_axis": 0,
        "x0_public_index": accumulator_shape[0] // 2,
        "coordinate_count": {
            name: int(np.count_nonzero(mask)) for name, mask in masks.items()
        },
        "regions": {},
    }
    for field, (recovar, native, allow_sign) in fields.items():
        _require(
            recovar.size == native.size == expected_size,
            f"{field} raw-region topology mismatch",
        )
        sign = _metric(recovar, native, allow_sign=allow_sign)[
            "sign_applied_to_source"
        ]
        aligned = sign * recovar
        total_residual_l2 = float(np.linalg.norm(aligned - native))
        _require(total_residual_l2 > 0.0, f"{field} raw residual has zero energy")
        report["regions"][field] = {
            name: {
                **_metric(aligned[mask], native[mask], allow_sign=False),
                "residual_l2_fraction_of_total": float(
                    np.linalg.norm((aligned - native)[mask]) / total_residual_l2
                ),
            }
            for name, mask in masks.items()
        }
    return report


def _intervention_projection_on_gap(
    recovar: np.ndarray,
    native_control: np.ndarray,
    native_intervention: np.ndarray,
    *,
    allow_sign: bool,
) -> dict[str, Any]:
    """Measure whether a native intervention moves along the RECOVAR-native gap."""

    recovar = np.asarray(recovar).reshape(-1)
    native_control = np.asarray(native_control).reshape(-1)
    native_intervention = np.asarray(native_intervention).reshape(-1)
    _require(
        recovar.shape == native_control.shape == native_intervention.shape,
        "intervention projection topology mismatch",
    )
    sign = _metric(recovar, native_control, allow_sign=allow_sign)[
        "sign_applied_to_source"
    ]
    gap = sign * recovar - native_control
    intervention = native_intervention - native_control
    gap_norm = float(np.linalg.norm(gap))
    intervention_norm = float(np.linalg.norm(intervention))
    _require(gap_norm > 0.0, "RECOVAR-native gap has zero energy")
    if intervention_norm == 0.0:
        cosine = 0.0
        projection_scale = 0.0
    else:
        inner = float(np.vdot(intervention, gap).real)
        cosine = inner / (intervention_norm * gap_norm)
        projection_scale = inner / (intervention_norm * intervention_norm)
    residual_ratio = float(
        np.linalg.norm(sign * recovar - native_intervention) / gap_norm
    )
    return {
        "sign_applied_to_recovar": sign,
        "intervention_to_gap_norm_ratio": intervention_norm / gap_norm,
        "real_inner_product_cosine": cosine,
        "least_squares_intervention_scale_to_gap": projection_scale,
        "gap_norm_ratio_after_full_intervention": residual_ratio,
        "squared_gap_fraction_removed_by_full_intervention": 1.0
        - residual_ratio * residual_ratio,
    }


def _paired_field_residual_structure(
    recovar1: np.ndarray,
    native1: np.ndarray,
    recovar2: np.ndarray,
    native2: np.ndarray,
    *,
    allow_sign: bool,
    support_union: np.ndarray,
) -> dict[str, Any]:
    """Compare cross-half signal and aligned residual correlations for one field."""

    sign1 = _metric(recovar1, native1, allow_sign=allow_sign)[
        "sign_applied_to_source"
    ]
    sign2 = _metric(recovar2, native2, allow_sign=allow_sign)[
        "sign_applied_to_source"
    ]
    residual1 = sign1 * np.asarray(recovar1) - native1
    residual2 = sign2 * np.asarray(recovar2) - native2
    return {
        "sign_applied_to_recovar": [sign1, sign2],
        "native_signal_centered_correlation": _centered_real_inner_correlation(
            native1, native2
        ),
        "residual_centered_correlation": _centered_real_inner_correlation(
            residual1, residual2
        ),
        "residual_centered_correlation_on_support_union": (
            _centered_real_inner_correlation(
                residual1, residual2, mask=support_union
            )
        ),
    }


def _paired_downsampled_residual_structure(
    recovar_average1: np.ndarray,
    recovar_weight1: np.ndarray,
    native_average1: np.ndarray,
    native_weight1: np.ndarray,
    recovar_average2: np.ndarray,
    recovar_weight2: np.ndarray,
    native_average2: np.ndarray,
    native_weight2: np.ndarray,
) -> dict[str, Any]:
    """Locate residual coherence after BPref downsampling and division."""

    support_union = (native_weight1 > 0) | (native_weight2 > 0)
    return {
        "policy": (
            "diagnostic centered real-inner-product correlation; acceptance remains "
            "scale-sensitive relative-L2/FSC-AUC"
        ),
        "average": _paired_field_residual_structure(
            recovar_average1,
            native_average1,
            recovar_average2,
            native_average2,
            allow_sign=True,
            support_union=support_union,
        ),
        "numerator": _paired_field_residual_structure(
            recovar_average1 * recovar_weight1,
            native_average1 * native_weight1,
            recovar_average2 * recovar_weight2,
            native_average2 * native_weight2,
            allow_sign=True,
            support_union=support_union,
        ),
        "denominator": _paired_field_residual_structure(
            recovar_weight1,
            native_weight1,
            recovar_weight2,
            native_weight2,
            allow_sign=False,
            support_union=support_union,
        ),
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


def _require_matching_bpref_stage(recovar_stage: str, native_stage: str) -> None:
    _require(
        recovar_stage == native_stage,
        "BPref stage mismatch: RECOVAR is "
        f"{recovar_stage!r}, while the native capture is {native_stage!r}",
    )


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


def _fsc_float64_from_downsampled_average(
    average0: np.ndarray,
    average1: np.ndarray,
    *,
    radius: int,
    max_shell: int,
    shell_count: int,
) -> np.ndarray:
    """Mirror RELION's signed FSC reduction without the production float32 cast."""

    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    x_axis = np.arange(average0.shape[2], dtype=np.float64)
    z, y, x = np.meshgrid(axis, axis, x_axis, indexing="ij")
    radial = np.sqrt(x * x + y * y + z * z)
    shell = np.floor(radial + 0.5).astype(np.int64)
    selected = radial <= float(max_shell)
    labels = shell[selected].reshape(-1)
    values0 = average0[selected].reshape(-1)
    values1 = average1[selected].reshape(-1)
    numerator = np.bincount(
        labels,
        weights=(np.conj(values0) * values1).real,
        minlength=shell_count,
    )[:shell_count]
    denominator0 = np.bincount(
        labels,
        weights=np.abs(values0) ** 2,
        minlength=shell_count,
    )[:shell_count]
    denominator1 = np.bincount(
        labels,
        weights=np.abs(values1) ** 2,
        minlength=shell_count,
    )[:shell_count]
    fsc = np.zeros(shell_count, dtype=np.float64)
    nonzero = denominator0 * denominator1 > 0.0
    fsc[nonzero] = numerator[nonzero] / np.sqrt(
        denominator0[nonzero] * denominator1[nonzero]
    )
    fsc[0] = 1.0
    return fsc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recovar-bpref",
        "--recovar-prejoin",
        dest="recovar_bpref",
        type=Path,
        required=True,
        help=(
            "RECOVAR BPref dump. --recovar-prejoin remains as a compatibility alias; "
            "the archive schema determines the actual stage."
        ),
    )
    parser.add_argument("--recovar-repeat", type=Path)
    parser.add_argument("--native-data", type=Path, required=True)
    parser.add_argument("--native-weight", type=Path, required=True)
    parser.add_argument("--native-repeat-data", type=Path)
    parser.add_argument("--native-repeat-weight", type=Path)
    parser.add_argument(
        "--native-data-half2",
        type=Path,
        help="Optional matching half-2 raw data dump for a two-half FSC comparison.",
    )
    parser.add_argument(
        "--native-weight-half2",
        type=Path,
        help="Optional matching half-2 raw weight dump for a two-half FSC comparison.",
    )
    parser.add_argument(
        "--recovar-fsc",
        type=Path,
        help="Optional production FSC array used to validate the paired accumulator replay.",
    )
    parser.add_argument(
        "--native-stage",
        choices=("pre_lowres_join", "post_lowres_join"),
        default="post_lowres_join",
        help=(
            "Stage of the native buffer. BackProjector::getDownsampledAverage raw "
            "captures are post_lowres_join; the MPI state hook labels both stages."
        ),
    )
    parser.add_argument("--half", type=int, choices=(1, 2), default=1)
    parser.add_argument("--expected-local-iteration", type=int, default=2)
    parser.add_argument("--physical-iteration", type=int, default=2)
    parser.add_argument("--first-shell", type=int, default=15)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    recovar = _load_recovar(args.recovar_bpref, half=args.half)
    _require_matching_bpref_stage(recovar["stage"], args.native_stage)
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
        "stage_gate": {
            "recovar": recovar["stage"],
            "native": args.native_stage,
            "exact": True,
        },
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
        "raw_accumulator_regions": _raw_accumulator_region_metrics(
            recovar["numerator"],
            native_numerator,
            recovar["weight"],
            native_denominator,
            accumulator_shape=recovar["accumulator_shape"],
        ),
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
            for path in (args.recovar_bpref, args.native_data, args.native_weight)
        },
    }
    if (args.native_data_half2 is None) != (args.native_weight_half2 is None):
        raise ValueError("native half-2 data and weight must be supplied together")
    if args.recovar_fsc is not None and args.native_data_half2 is None:
        raise ValueError("--recovar-fsc requires the matching native half-2 data and weight")
    if args.native_data_half2 is not None and args.native_weight_half2 is not None:
        _require(args.half == 1, "the paired FSC comparison must be anchored by --half=1")
        recovar_half2 = _load_recovar(args.recovar_bpref, half=2)
        for key in (
            "stage",
            "iteration",
            "current_size",
            "padding_factor",
            "grid_size",
            "volume_shape",
            "accumulator_shape",
        ):
            _require(recovar_half2[key] == recovar[key], f"RECOVAR half metadata differs for {key}")
        native_data2_header, native_data2 = _load_native_bpref(
            args.native_data_half2, value_dtype=np.complex128
        )
        native_weight2_header, native_weight2 = _load_native_bpref(
            args.native_weight_half2, value_dtype=np.float64
        )
        _require(
            np.array_equal(native_data2_header, native_weight2_header),
            "native half-2 raw headers differ",
        )
        native_numerator2, native_denominator2 = relion_raw_to_recovar_full(
            native_data2, native_weight2, grid_size=recovar["grid_size"]
        )
        report["paired_raw_residual_structure"] = _paired_raw_residual_structure(
            recovar["numerator"],
            native_numerator,
            recovar["weight"],
            native_denominator,
            recovar_half2["numerator"],
            native_numerator2,
            recovar_half2["weight"],
            native_denominator2,
        )
        rec_average2, rec_weight2, rec_radius2 = _downsample(
            recovar_half2["numerator"],
            recovar_half2["weight"],
            max_shell=max_shell,
            **{
                key: recovar_half2[key]
                for key in ("grid_size", "volume_shape", "accumulator_shape", "padding_factor")
            },
        )
        native_average2, native_down_weight2, native_radius2 = _downsample(
            native_numerator2,
            native_denominator2,
            max_shell=max_shell,
            **{
                key: recovar[key]
                for key in ("grid_size", "volume_shape", "accumulator_shape", "padding_factor")
            },
        )
        _require(
            radius == rec_radius2 == native_radius2,
            "paired FSC downsample radii differ",
        )
        report["paired_downsampled_residual_structure"] = (
            _paired_downsampled_residual_structure(
                rec_average,
                rec_weight,
                native_average,
                native_down_weight,
                rec_average2,
                rec_weight2,
                native_average2,
                native_down_weight2,
            )
        )
        accumulator_shape = tuple(int(value) for value in recovar["accumulator_shape"])
        x0_mask = np.zeros(accumulator_shape, dtype=bool)
        x0_mask[accumulator_shape[0] // 2, :, :] = True
        x0_mask = x0_mask.reshape(-1)
        denominator_factorial: dict[str, Any] = {}
        for arm, replacement_mask in (
            ("control", None),
            ("native_denominator_x0_only", x0_mask),
            ("native_denominator_off_x0", ~x0_mask),
        ):
            denominator1 = np.asarray(recovar["weight"], dtype=np.float64).copy()
            denominator2 = np.asarray(recovar_half2["weight"], dtype=np.float64).copy()
            if replacement_mask is not None:
                denominator1[replacement_mask] = native_denominator[replacement_mask]
                denominator2[replacement_mask] = native_denominator2[replacement_mask]
            arm_average1, arm_weight1, arm_radius1 = _downsample(
                recovar["numerator"],
                denominator1,
                max_shell=max_shell,
                **{
                    key: recovar[key]
                    for key in (
                        "grid_size",
                        "volume_shape",
                        "accumulator_shape",
                        "padding_factor",
                    )
                },
            )
            arm_average2, arm_weight2, arm_radius2 = _downsample(
                recovar_half2["numerator"],
                denominator2,
                max_shell=max_shell,
                **{
                    key: recovar_half2[key]
                    for key in (
                        "grid_size",
                        "volume_shape",
                        "accumulator_shape",
                        "padding_factor",
                    )
                },
            )
            _require(
                radius == arm_radius1 == arm_radius2,
                "x0 denominator factorial downsample radii differ",
            )
            denominator_factorial[arm] = {
                "half1": _comparison(
                    arm_average1,
                    arm_weight1,
                    native_average,
                    native_down_weight,
                    radius=radius,
                    first_shell=args.first_shell,
                    max_shell=max_shell,
                ),
                "half2": _comparison(
                    arm_average2,
                    arm_weight2,
                    native_average2,
                    native_down_weight2,
                    radius=radius,
                    first_shell=args.first_shell,
                    max_shell=max_shell,
                ),
                "paired_average_residual": _paired_field_residual_structure(
                    arm_average1,
                    native_average,
                    arm_average2,
                    native_average2,
                    allow_sign=True,
                    support_union=(native_down_weight > 0)
                    | (native_down_weight2 > 0),
                ),
            }
        report["x0_denominator_factorial"] = {
            "intervention": (
                "replace only the named RECOVAR raw denominator region with "
                "source-aligned native values before downsampling/division"
            ),
            "x0_public_axis": 0,
            "x0_public_index": accumulator_shape[0] // 2,
            "arms": denominator_factorial,
        }
        shell_count = recovar["grid_size"] // 2 + 1
        recovar_fsc_float64 = _fsc_float64_from_downsampled_average(
            rec_average,
            rec_average2,
            radius=radius,
            max_shell=max_shell,
            shell_count=shell_count,
        )
        native_fsc_float64 = _fsc_float64_from_downsampled_average(
            native_average,
            native_average2,
            radius=radius,
            max_shell=max_shell,
            shell_count=shell_count,
        )
        recovar_fsc = np.asarray(
            regularization.compute_relion_fsc_from_backprojector(
                recovar["numerator"],
                recovar_half2["numerator"],
                recovar["weight"],
                recovar_half2["weight"],
                recovar["volume_shape"],
                padding_factor=recovar["padding_factor"],
                r_max=max_shell,
                accumulator_volume_shape=recovar["accumulator_shape"],
            ),
            dtype=np.float64,
        )
        native_fsc = np.asarray(
            regularization.compute_relion_fsc_from_backprojector(
                native_numerator,
                native_numerator2,
                native_denominator,
                native_denominator2,
                recovar["volume_shape"],
                padding_factor=recovar["padding_factor"],
                r_max=max_shell,
                accumulator_volume_shape=recovar["accumulator_shape"],
            ),
            dtype=np.float64,
        )
        report["two_half_fsc"] = {
            "metrics": _metric(recovar_fsc, native_fsc, allow_sign=False),
            "recovar": recovar_fsc.tolist(),
            "native": native_fsc.tolist(),
            "float64_metrics": _metric(
                recovar_fsc_float64,
                native_fsc_float64,
                allow_sign=False,
            ),
            "recovar_float64": recovar_fsc_float64.tolist(),
            "native_float64": native_fsc_float64.tolist(),
            "recovar_float32_cast_effect": _metric(
                recovar_fsc,
                recovar_fsc_float64,
                allow_sign=False,
            ),
            "native_float32_cast_effect": _metric(
                native_fsc,
                native_fsc_float64,
                allow_sign=False,
            ),
        }
        if args.recovar_fsc is not None:
            production_fsc = np.asarray(
                np.load(args.recovar_fsc, allow_pickle=False), dtype=np.float64
            ).reshape(-1)
            report["two_half_fsc"]["recovar_replay_vs_production"] = _metric(
                recovar_fsc, production_fsc, allow_sign=False
            )
            report["two_half_fsc"]["production"] = production_fsc.tolist()
            report["artifacts"][str(args.recovar_fsc.resolve())] = _sha256(args.recovar_fsc)
        for path in (args.native_data_half2, args.native_weight_half2):
            report["artifacts"][str(path.resolve())] = _sha256(path)
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
        report["recovar_vs_native_repeat"] = _comparison(
            rec_average,
            rec_weight,
            repeat_average,
            repeat_down_weight,
            radius=radius,
            first_shell=args.first_shell,
            max_shell=max_shell,
        )
        report["native_raw_vs_native_repeat"] = {
            "numerator": _metric(
                repeat_numerator,
                native_numerator,
                allow_sign=True,
            ),
            "denominator": _metric(
                repeat_denominator,
                native_denominator,
                allow_sign=False,
            ),
        }
        report["native_repeat_projection_on_recovar_gap"] = {
            "policy": (
                "raw-field real-inner-product projection; positive removed-gap "
                "fraction means the native alternate moves toward RECOVAR"
            ),
            "numerator": _intervention_projection_on_gap(
                recovar["numerator"],
                native_numerator,
                repeat_numerator,
                allow_sign=True,
            ),
            "denominator": _intervention_projection_on_gap(
                recovar["weight"],
                native_denominator,
                repeat_denominator,
                allow_sign=False,
            ),
        }
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
