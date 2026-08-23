#!/usr/bin/env python3
"""Replay one FSC shell from saved post-backprojection BPref aggregates.

This boundary can eliminate aggregate-to-native downsampling and FSC reduction
precision/order. It cannot distinguish upstream operand, geometry, or GPU
atomic-order differences; those require versioned BPref contribution captures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import starfile

from recovar.reconstruction import regularization

SCHEMA = "recovar-saved-bpref-fsc-precision-audit-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _round_away(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values >= 0,
        np.floor(values + 0.5),
        np.ceil(values - 0.5),
    ).astype(np.int64)


def _scatter(values, labels, size: int, *, dtype, reverse: bool) -> np.ndarray:
    values = np.asarray(values, dtype=dtype)
    labels = np.asarray(labels, dtype=np.int64)
    if reverse:
        values = values[::-1]
        labels = labels[::-1]
    out = np.zeros(size, dtype=dtype)
    np.add.at(out, labels, values)
    return out


def replay_shell_from_saved_aggregates(
    data0,
    data1,
    weight0,
    weight1,
    *,
    padding_factor: int,
    current_size: int,
    accumulator_size: int,
    shell: int,
) -> dict[str, object]:
    """Replay aggregate downsampling and shell sums at two precisions/orders."""

    cube_shape = (accumulator_size,) * 3
    arrays = [
        np.asarray(value).reshape(cube_shape)
        for value in (data0, data1, weight0, weight1)
    ]
    logical = [np.transpose(value, (1, 2, 0)) for value in arrays]
    axis = np.arange(
        -(accumulator_size // 2),
        accumulator_size - accumulator_size // 2,
        dtype=np.float64,
    )
    down_radius = current_size // 2 + 1
    down_size = 2 * down_radius + 1
    down_xsize = down_size // 2 + 1
    target_count = down_size * down_size * down_xsize
    down_axis = _round_away(axis / padding_factor)

    labels_parts: list[np.ndarray] = []
    values_parts: list[list[np.ndarray]] = [[], [], [], []]
    identity_parts: list[np.ndarray] = []
    yy, xx = np.meshgrid(
        np.arange(accumulator_size),
        np.arange(accumulator_size),
        indexing="ij",
    )
    dy = down_axis[yy]
    dx = down_axis[xx]
    for z_index, dz in enumerate(down_axis):
        radius = np.sqrt(float(dz * dz) + dy * dy + dx * dx)
        selected = (
            (dz >= -down_radius)
            & (dz <= down_radius)
            & (dy >= -down_radius)
            & (dy <= down_radius)
            & (dx >= 0)
            & (dx < down_xsize)
            & (radius <= float(current_size // 2))
            & (_round_away(radius) == int(shell))
        )
        rows, cols = np.nonzero(selected)
        if rows.size == 0:
            continue
        labels_parts.append(
            ((dz + down_radius) * down_size + (dy[rows, cols] + down_radius))
            * down_xsize
            + dx[rows, cols]
        )
        identity_parts.append(
            ((z_index * accumulator_size + rows) * accumulator_size + cols).astype(
                np.int64
            )
        )
        for index, grid in enumerate(logical):
            values_parts[index].append(np.asarray(grid[z_index, rows, cols]))

    labels = np.concatenate(labels_parts)
    identities = np.concatenate(identity_parts)
    values = [np.concatenate(parts) for parts in values_parts]
    if np.unique(identities).size != identities.size:
        raise ValueError("saved aggregate voxel identities are not unique")
    if not np.all(np.diff(identities) > 0):
        raise ValueError("traversal is not canonical logical z/y/x order")

    downsampled: dict[tuple[str, str], list[np.ndarray]] = {}
    for precision, real_dtype, complex_dtype in (
        ("float32", np.float32, np.complex64),
        ("float64", np.float64, np.complex128),
    ):
        for order, reverse in (("canonical", False), ("reverse", True)):
            averages = []
            for data_values, weight_values in (
                (values[0], values[2]),
                (values[1], values[3]),
            ):
                sum_real = _scatter(
                    data_values.real,
                    labels,
                    target_count,
                    dtype=real_dtype,
                    reverse=reverse,
                )
                sum_imag = _scatter(
                    data_values.imag,
                    labels,
                    target_count,
                    dtype=real_dtype,
                    reverse=reverse,
                )
                sum_weight = _scatter(
                    weight_values,
                    labels,
                    target_count,
                    dtype=real_dtype,
                    reverse=reverse,
                )
                current = (
                    sum_real + complex_dtype(1j) * sum_imag
                ).astype(complex_dtype, copy=False)
                nonzero = sum_weight > 0
                current[nonzero] /= sum_weight[nonzero]
                current[~nonzero] = 0
                averages.append(current)
            downsampled[(precision, order)] = averages

    targets = np.unique(labels)
    modes: dict[str, float] = {}
    for (downsample_precision, downsample_order), averages in downsampled.items():
        for shell_precision, real_dtype, complex_dtype in (
            ("float32", np.float32, np.complex64),
            ("float64", np.float64, np.complex128),
        ):
            left = averages[0][targets].astype(complex_dtype, copy=False)
            right = averages[1][targets].astype(complex_dtype, copy=False)
            terms = (
                (np.conj(left) * right).real.astype(real_dtype, copy=False),
                (np.abs(left) ** 2).astype(real_dtype, copy=False),
                (np.abs(right) ** 2).astype(real_dtype, copy=False),
            )
            for shell_order, shell_reverse in (
                ("canonical", False),
                ("reverse", True),
            ):
                shell_labels = np.zeros(terms[0].size, dtype=np.int64)
                numerator, denom0, denom1 = (
                    _scatter(
                        term,
                        shell_labels,
                        1,
                        dtype=real_dtype,
                        reverse=shell_reverse,
                    )[0]
                    for term in terms
                )
                mode = (
                    f"downsample_{downsample_precision}_{downsample_order}__"
                    f"shell_{shell_precision}_{shell_order}"
                )
                modes[mode] = float(numerator / np.sqrt(denom0 * denom1))

    values = np.asarray(list(modes.values()), dtype=np.float64)

    return {
        "shell": int(shell),
        "saved_voxel_contribution_count": int(labels.size),
        "native_target_count": int(np.unique(labels).size),
        "canonical_identity": "logical_relion_z_y_x_source_coordinate",
        "modes": modes,
        "mode_min": float(np.min(values)),
        "mode_max": float(np.max(values)),
        "mode_range": float(np.ptp(values)),
    }


def _star_fsc(path: Path, shell: int) -> float:
    table = starfile.read(path)["model_class_1"]
    rows = table.loc[
        table["rlnSpectralIndex"] == int(shell),
        "rlnGoldStandardFsc",
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one shell {shell} row in {path}")
    return float(rows.iloc[0])


def audit(args) -> dict[str, object]:
    directory = args.intermediates.resolve()
    prefix = directory / f"it{args.iteration:03d}"
    paths = {
        name: Path(f"{prefix}_{name}.npy")
        for name in ("Ft_y_0", "Ft_y_1", "Ft_ctf_0", "Ft_ctf_1", "fsc", "meta")
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    arrays = {
        name: np.load(
            path,
            mmap_mode=None if name in {"fsc", "meta"} else "r",
            allow_pickle=name == "meta",
        )
        for name, path in paths.items()
    }
    meta = arrays["meta"].item()
    accumulator_size = round(np.cbrt(arrays["Ft_y_0"].size))
    if accumulator_size**3 != arrays["Ft_y_0"].size:
        raise ValueError("saved Ft_y is not a cubic full accumulator")
    if int(meta["current_size"]) != int(args.current_size):
        raise ValueError(
            f"current-size mismatch: capture={meta['current_size']} "
            f"CLI={args.current_size}"
        )

    production = np.asarray(
        regularization.compute_relion_fsc_from_backprojector(
            arrays["Ft_y_0"],
            arrays["Ft_y_1"],
            arrays["Ft_ctf_0"],
            arrays["Ft_ctf_1"],
            (args.volume_size,) * 3,
            padding_factor=args.padding_factor,
            r_max=args.current_size // 2,
            accumulator_volume_shape=(accumulator_size,) * 3,
        )
    )
    saved = np.asarray(arrays["fsc"])
    replay = replay_shell_from_saved_aggregates(
        arrays["Ft_y_0"],
        arrays["Ft_y_1"],
        arrays["Ft_ctf_0"],
        arrays["Ft_ctf_1"],
        padding_factor=args.padding_factor,
        current_size=args.current_size,
        accumulator_size=accumulator_size,
        shell=args.shell,
    )
    controls = {
        str(path.resolve()): _star_fsc(path.resolve(), args.shell)
        for path in args.relion_model
    }
    recovar_value = float(saved[args.shell])
    closest_gap = min(abs(recovar_value - value) for value in controls.values())
    downstream_values = np.asarray(
        [recovar_value, float(production[args.shell]), *replay["modes"].values()],
        dtype=np.float64,
    )
    downstream_bound = float(
        np.max(np.abs(downstream_values - recovar_value))
    )
    downstream_range = float(np.ptp(downstream_values))
    threshold_side_stable = bool(
        np.all(downstream_values > args.threshold)
        or np.all(downstream_values < args.threshold)
    )
    threshold_margin = float(
        np.min(np.abs(downstream_values - args.threshold))
    )
    replay_closed = bool(np.array_equal(saved, production) and threshold_side_stable)
    contribution_artifacts = sorted(
        directory.parent.rglob("*bpref*contribution*.npz")
    )
    signature_artifacts = sorted(directory.parent.rglob("*device*signature*.npz"))
    return {
        "schema": SCHEMA,
        "metric_policy": (
            "exact/array metrics for intermediates; FSC/FSC-AUC for maps; "
            "correlation not computed"
        ),
        "scope": "post-backprojection saved aggregates through downsampled FSC",
        "capture_commit": args.capture_commit,
        "inputs": {
            str(path.resolve()): _sha256(path) for path in paths.values()
        },
        "saved_fsc": {
            "shell": args.shell,
            "value": recovar_value,
            "production_recompute_value": float(production[args.shell]),
            "production_curve_bitwise_equal": bool(
                np.array_equal(saved, production)
            ),
            "production_curve_max_abs": float(
                np.max(
                    np.abs(
                        saved.astype(np.float64)
                        - production.astype(np.float64)
                    )
                )
            ),
        },
        "target_shell_replay": replay,
        "relion_controls": controls,
        "closest_relion_control_abs_gap": closest_gap,
        "scheduling_threshold": args.threshold,
        "downstream_max_abs_from_saved": downstream_bound,
        "downstream_control_range": downstream_range,
        "downstream_threshold_side_stable": threshold_side_stable,
        "downstream_min_abs_threshold_margin": threshold_margin,
        "gap_to_downstream_bound_ratio": (
            None if downstream_bound == 0 else closest_gap / downstream_bound
        ),
        "classification": (
            "unresolved_upstream_of_saved_aggregate"
            if replay_closed
            else "downstream_saved_aggregate_replay_not_closed"
        ),
        "eliminated": (
            [
                "saved_FSC_recompute",
                "aggregate_to_native_reduction_precision",
                "aggregate_to_native_reduction_order",
                "downsampled_FSC_shell_reduction_precision",
                "downsampled_FSC_shell_reduction_order",
            ]
            if replay_closed
            else []
        ),
        "not_classifiable_from_capture": [
            "operand_generation",
            "backprojection_geometry",
            "GPU_atomic_reduction_order",
        ],
        "capture_inventory": {
            "bpref_contribution_artifacts": [
                str(path.resolve()) for path in contribution_artifacts
            ],
            "device_signature_artifacts": [
                str(path.resolve()) for path in signature_artifacts
            ],
            "complete_per_contribution_capture_available": bool(
                contribution_artifacts and signature_artifacts
            ),
        },
        "required_next_capture": [
            "exact particle/image identity and launch ordinal",
            "posterior weight and pre-fold complex source value",
            "eight device indices and interpolation coefficients",
            "row and neighbor Hermitian-conjugation flags",
            "raw image, CTF, preprocessing, pose and support operands",
        ],
        "captured_float32_cast_limitation": (
            "complex64/float32 aggregates cannot recover precision lost during "
            "operand generation or GPU scatter"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediates", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--shell", type=int, required=True)
    parser.add_argument("--volume-size", type=int, required=True)
    parser.add_argument("--current-size", type=int, required=True)
    parser.add_argument("--padding-factor", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--relion-model", type=Path, action="append", required=True)
    parser.add_argument("--capture-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
