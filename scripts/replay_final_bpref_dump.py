#!/usr/bin/env python
"""Replay a dumped final all-data RELION-style backprojection.

This is a debugging aid for RECOVAR EM parity work. It takes the
``recovar_final_bpref_accum.npz`` file written by
``RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR`` and re-runs the final tau2 update plus
Wiener reconstruction with the currently checked-out code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dump", type=Path, required=True, help="Path to recovar_final_bpref_accum.npz.")
    parser.add_argument("--gt-mrc", type=Path, help="Optional ground-truth MRC in RECOVAR convention.")
    parser.add_argument("--gt-relion-convention", action="store_true", help="Load --gt-mrc as a RELION MRC.")
    parser.add_argument("--relion-mrc", type=Path, help="Optional RELION final map MRC for comparison.")
    parser.add_argument("--output-json", type=Path, help="Optional output JSON summary.")
    parser.add_argument("--output-mrc", type=Path, help="Optional replayed map MRC in RECOVAR convention.")
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Set JAX_PLATFORMS=cpu before importing RECOVAR/JAX.",
    )
    parser.add_argument(
        "--recompute-fsc",
        action="store_true",
        help="Recompute FSC from half accumulators instead of using dumped fsc_shells.",
    )
    parser.add_argument(
        "--grid-correct",
        choices=("dump", "on", "off"),
        default="dump",
        help="Final gridding correction mode for replay.",
    )
    parser.add_argument(
        "--tau2-fsc-mode",
        choices=("whole", "half"),
        default="whole",
        help=(
            "FSC-to-SSNR conversion for the final tau2 replay. 'whole' matches "
            "the current RECOVAR final all-data path; 'half' is a diagnostic "
            "for RELION parity debugging."
        ),
    )
    parser.add_argument(
        "--tau2-full-half-axis",
        type=int,
        help=(
            "Half-complex axis for full-volume tau2 shell statistics. Defaults "
            "to mstep_full_half_axis in the dump, or -1 for old/native dumps."
        ),
    )
    parser.add_argument("--minres-map", type=int, default=5, help="RELION minres_map shell.")
    parser.add_argument(
        "--projection-padding-factor",
        type=int,
        help="Gridding correction padding factor. Defaults to dump value or reconstruction padding factor.",
    )
    parser.add_argument(
        "--relion-bpref-prefix",
        type=Path,
        help=(
            "Optional RELION dump prefix ending before _bpref_data.bin and "
            "_bpref_weight.bin. Replaces the RECOVAR joined accumulator."
        ),
    )
    parser.add_argument(
        "--compare-relion-boundary-spectra",
        action="store_true",
        help=(
            "Require and compare _tau2.bin, _sigma2.bin, _data_vs_prior.bin, "
            "_fsc.bin, and _fourier_coverage.bin at --relion-bpref-prefix."
        ),
    )
    parser.add_argument(
        "--relion-tau2-bin",
        type=Path,
        help="Optional RELION length-prefixed tau2 spectrum. Replaces the RECOVAR-derived tau2.",
    )
    return parser.parse_args(argv)


def resolve_tau2_full_half_axis(dump: Any, override: int | None = None) -> int:
    """Return the tau2 shell-stat half-complex axis for replay dumps."""

    if override is not None:
        return int(override)
    if "mstep_full_half_axis" not in dump:
        return -1
    return int(np.asarray(dump["mstep_full_half_axis"]).reshape(()).item())


def resolve_mstep_accumulator_shape(dump: Any, volume_shape: tuple[int, int, int], padding_factor: int):
    """Return the BPref accumulator shape recorded by new dumps, or the old even-grid default."""

    if "mstep_accumulator_shape" in dump:
        return tuple(int(v) for v in np.asarray(dump["mstep_accumulator_shape"]).reshape(-1))
    return tuple(int(v) * int(padding_factor) for v in volume_shape)


def read_relion_bpref_array(path: Path, *, dtype: np.dtype) -> np.ndarray:
    """Read a RELION BPref dump with a three-int64 shape header."""

    with path.open("rb") as stream:
        header = stream.read(3 * np.dtype(np.int64).itemsize)
        if len(header) != 3 * np.dtype(np.int64).itemsize:
            raise ValueError(f"truncated RELION BPref header: {path}")
        shape = tuple(int(value) for value in struct.unpack("qqq", header))
        count = int(np.prod(shape))
        values = np.fromfile(stream, dtype=dtype, count=count)
        if values.size != count or stream.read(1):
            raise ValueError(f"RELION BPref payload size does not match {shape}: {path}")
    return values.reshape(shape)


def read_relion_spectrum(path: Path) -> np.ndarray:
    """Read a RELION float64 spectrum with a one-int64 length header."""

    with path.open("rb") as stream:
        header = stream.read(np.dtype(np.int64).itemsize)
        if len(header) != np.dtype(np.int64).itemsize:
            raise ValueError(f"truncated RELION spectrum header: {path}")
        (count,) = struct.unpack("q", header)
        values = np.fromfile(stream, dtype=np.float64, count=int(count))
        if values.size != int(count) or stream.read(1):
            raise ValueError(f"RELION spectrum payload size does not match length {count}: {path}")
    return values


def compare_relion_boundary_spectra(
    dump: Any,
    prefix: Path,
    *,
    grid_size: int,
) -> dict[str, Any]:
    """Compare native RELION updateSSNRarrays outputs with RECOVAR equivalents."""

    required_dump_keys = {
        "tau2": "tau2_prior_shells",
        "sigma2": "tau2_sigma2_shells",
        "data_vs_prior": "tau2_ssnr_shells",
        "fsc": "fsc_shells",
    }
    missing = [key for key in required_dump_keys.values() if key not in dump]
    if missing:
        raise ValueError(f"RECOVAR final BPref dump is missing spectrum fields: {missing}")

    native = {
        name: read_relion_spectrum(Path(f"{prefix}_{name}.bin"))
        for name in (*required_dump_keys, "fourier_coverage")
    }
    n4 = float(int(grid_size) ** 4)
    converted_native = {
        "tau2": native["tau2"] * n4,
        "sigma2": native["sigma2"] * n4,
        "data_vs_prior": native["data_vs_prior"],
        "fsc": native["fsc"],
    }
    comparisons = {
        name: streaming_field_metrics(
            np.asarray(dump[dump_key], dtype=np.float64).reshape(-1),
            np.asarray(converted_native[name], dtype=np.float64).reshape(-1),
        )
        for name, dump_key in required_dump_keys.items()
    }
    coverage = np.asarray(native["fourier_coverage"], dtype=np.float64).reshape(-1)
    if coverage.size == 0 or not np.all(np.isfinite(coverage)):
        raise ValueError("native RELION Fourier-coverage spectrum is empty or non-finite")
    return {
        "policy": (
            "streamed float64 comparison; native RELION tau2/sigma2 multiplied by N^4; "
            "data-vs-prior and FSC are dimensionless"
        ),
        "grid_size": int(grid_size),
        "native_to_recovar_tau2_sigma2_scale": n4,
        "comparisons": comparisons,
        "native_fourier_coverage": {
            "element_count": int(coverage.size),
            "minimum": float(np.min(coverage)),
            "maximum": float(np.max(coverage)),
            "mean": float(np.mean(coverage)),
        },
    }


def centered_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=np.float64).reshape(-1)
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        return float("nan")
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(np.dot(a, b) / denom)


def streaming_field_metrics(
    source: np.ndarray,
    target: np.ndarray,
    *,
    chunk_size: int = 1 << 20,
) -> dict[str, Any]:
    """Compare large real or complex fields without materializing a full residual."""

    source_flat = np.asarray(source).reshape(-1)
    target_flat = np.asarray(target).reshape(-1)
    if source_flat.shape != target_flat.shape or source_flat.size == 0:
        raise ValueError(
            f"field topology mismatch: source={source_flat.shape}, target={target_flat.shape}"
        )
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    work_dtype = (
        np.complex128
        if np.iscomplexobj(source_flat) or np.iscomplexobj(target_flat)
        else np.float64
    )
    source_energy = 0.0
    target_energy = 0.0
    source_target_inner_real = 0.0
    residual_energy = 0.0
    maximum_absolute_residual = 0.0
    mismatch_count = 0
    first_mismatch_flat_index: int | None = None
    for start in range(0, source_flat.size, int(chunk_size)):
        stop = min(start + int(chunk_size), source_flat.size)
        source_chunk = np.asarray(source_flat[start:stop], dtype=work_dtype)
        target_chunk = np.asarray(target_flat[start:stop], dtype=work_dtype)
        if not np.all(np.isfinite(source_chunk)) or not np.all(np.isfinite(target_chunk)):
            raise ValueError(f"field contains non-finite values in flat range [{start}, {stop})")
        residual = source_chunk - target_chunk
        source_energy += float(np.vdot(source_chunk, source_chunk).real)
        target_energy += float(np.vdot(target_chunk, target_chunk).real)
        source_target_inner_real += float(np.vdot(source_chunk, target_chunk).real)
        residual_energy += float(np.vdot(residual, residual).real)
        if residual.size:
            maximum_absolute_residual = max(
                maximum_absolute_residual,
                float(np.max(np.abs(residual))),
            )
        unequal = source_chunk != target_chunk
        chunk_mismatch_count = int(np.count_nonzero(unequal))
        if chunk_mismatch_count and first_mismatch_flat_index is None:
            first_mismatch_flat_index = start + int(np.flatnonzero(unequal)[0])
        mismatch_count += chunk_mismatch_count

    if source_energy <= 0.0 or target_energy <= 0.0:
        raise ValueError("field comparison requires nonzero source and target energy")
    least_squares_scale = source_target_inner_real / source_energy
    scaled_residual_energy = max(
        least_squares_scale * least_squares_scale * source_energy
        - 2.0 * least_squares_scale * source_target_inner_real
        + target_energy,
        0.0,
    )
    return {
        "element_count": int(source_flat.size),
        "exact_equal": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "first_mismatch_flat_index": first_mismatch_flat_index,
        "maximum_absolute_residual": maximum_absolute_residual,
        "relative_l2": float(math.sqrt(residual_energy / target_energy)),
        "source_to_target_least_squares_scale": float(least_squares_scale),
        "relative_l2_after_scale": float(
            math.sqrt(scaled_residual_energy / target_energy)
        ),
    }


def shell_fsc(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return canonical RECOVAR FSC shells, excluding Nyquist edges."""
    a = np.asarray(lhs, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or len(set(a.shape)) != 1:
        return np.asarray([], dtype=np.float64)

    n = int(a.shape[0])
    fa = np.fft.fftn(a)
    fb = np.fft.fftn(b)
    freqs = np.fft.fftfreq(n) * n
    z, y, x = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    shells = np.rint(np.sqrt(x * x + y * y + z * z)).astype(np.int32).ravel()
    product = (fa * np.conj(fb)).ravel()
    numerator = np.bincount(shells, weights=np.real(product))
    lhs_power = np.bincount(shells, weights=(np.abs(fa) ** 2).ravel())
    rhs_power = np.bincount(shells, weights=(np.abs(fb) ** 2).ravel())
    denom = np.sqrt(lhs_power * rhs_power)
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denom, out=out, where=denom > 0.0)
    return out[: n // 2 - 1]


def normalized_fsc_auc(values: Any, axis: Any | None = None) -> float:
    fsc = np.asarray(values, dtype=np.float64).reshape(-1)
    if fsc.size == 0:
        return float("nan")
    if axis is None:
        x = np.arange(fsc.size, dtype=np.float64)
    else:
        x = np.asarray(axis, dtype=np.float64).reshape(-1)
        if x.size != fsc.size:
            return float("nan")
    finite = np.isfinite(fsc) & np.isfinite(x)
    if finite.size:
        finite[0] = False
    x = x[finite]
    y = fsc[finite]
    if y.size == 0:
        return float("nan")
    if y.size == 1:
        return float(y[0])
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])
    if span <= 0.0 or not math.isfinite(span):
        return float(np.mean(y))
    x_norm = (x - x[0]) / span
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x_norm))


def first_shell_below(values: np.ndarray, threshold: float) -> int | None:
    values = np.asarray(values, dtype=np.float64)
    for shell in range(1, values.size):
        if np.isfinite(values[shell]) and float(values[shell]) < float(threshold):
            return int(shell)
    return None


def map_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    fsc = shell_fsc(lhs, rhs)
    shells = np.arange(fsc.size, dtype=np.float64)
    return {
        "corr": centered_corr(lhs, rhs),
        "fsc_auc": normalized_fsc_auc(fsc, shells),
        "fsc": fsc,
        "shell_05": first_shell_below(fsc, 0.5),
        "shell_0143": first_shell_below(fsc, 0.143),
    }


def _tail(values: Any, n: int = 10) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return []
    return [float(v) for v in arr[-min(n, arr.size) :]]


def _head(values: Any, n: int = 10) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return []
    return [float(v) for v in arr[: min(n, arr.size)]]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.force_cpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from recovar.core import fourier_transform_utils
    from recovar.em.dense_single_volume.helpers import half_volume_mstep
    from recovar.reconstruction import regularization, relion_functions
    from recovar.utils import helpers

    dump_path = args.dump.resolve()
    with np.load(dump_path) as dump:
        volume_shape = tuple(int(v) for v in np.asarray(dump["volume_shape"]).reshape(-1))
        padding_factor = int(np.asarray(dump["padding_factor"]).item())
        projection_padding_factor = int(
            args.projection_padding_factor
            if args.projection_padding_factor is not None
            else np.asarray(dump["projection_padding_factor"]).item()
            if "projection_padding_factor" in dump
            else padding_factor
        )
        current_size = int(np.asarray(dump["current_size"]).item())
        mstep_accumulator_shape = resolve_mstep_accumulator_shape(dump, volume_shape, padding_factor)
        voxel_size = float(np.asarray(dump["voxel_size"]).item()) if "voxel_size" in dump else None
        tau2_fudge = float(np.asarray(dump["tau2_fudge"]).item()) if "tau2_fudge" in dump else 1.0
        if args.grid_correct == "dump":
            grid_correct = bool(np.asarray(dump["grid_correct"]).item()) if "grid_correct" in dump else False
        else:
            grid_correct = args.grid_correct == "on"

        if args.recompute_fsc or "fsc_shells" not in dump:
            fsc_shells = regularization.compute_relion_fsc_from_backprojector(
                dump["Ft_y_0"],
                dump["Ft_y_1"],
                dump["Ft_ctf_0"],
                dump["Ft_ctf_1"],
                volume_shape,
                padding_factor=padding_factor,
                r_max=current_size // 2,
                accumulator_volume_shape=mstep_accumulator_shape,
            )
        else:
            fsc_shells = np.asarray(dump["fsc_shells"], dtype=np.float64)

        tau2_full_half_axis = resolve_tau2_full_half_axis(dump, args.tau2_full_half_axis)
        final_tau, tau2_fsc_used, tau_details = regularization.compute_relion_tau2_from_weights(
            dump["Ft_ctf_0"] if "Ft_ctf_0" in dump else dump["Ft_ctf"],
            dump["Ft_ctf_1"] if "Ft_ctf_1" in dump else dump["Ft_ctf"],
            fsc_shells,
            volume_shape,
            tau2_fudge=tau2_fudge,
            padding_factor=padding_factor,
            r_max=current_size // 2,
            is_whole_instead_of_half=args.tau2_fsc_mode == "whole",
            return_details=True,
            full_half_axis=tau2_full_half_axis,
            accumulator_volume_shape=mstep_accumulator_shape,
            weight_combination="sum",
        )

        reconstruction_Ft_ctf = np.asarray(dump["Ft_ctf"])
        reconstruction_Ft_y = np.asarray(dump["Ft_y"])
        accumulator_source = "recovar_dump"
        accumulator_comparison = None
        spectrum_comparison = None
        if args.relion_bpref_prefix is not None:
            prefix = args.relion_bpref_prefix.resolve()
            relion_data = read_relion_bpref_array(
                Path(f"{prefix}_bpref_data.bin"),
                dtype=np.dtype(np.complex128),
            )
            relion_weight = read_relion_bpref_array(
                Path(f"{prefix}_bpref_weight.bin"),
                dtype=np.dtype(np.float64),
            )
            expected_half_shape = (
                mstep_accumulator_shape[0],
                mstep_accumulator_shape[1],
                mstep_accumulator_shape[2] // 2 + 1,
            )
            if relion_data.shape != expected_half_shape or relion_weight.shape != expected_half_shape:
                raise ValueError(
                    "RELION BPref shape mismatch: "
                    f"data={relion_data.shape}, weight={relion_weight.shape}, expected={expected_half_shape}"
                )
            n = int(volume_shape[0])
            reconstruction_Ft_y = (
                half_volume_mstep.relion_x_half_volume_to_native_half(
                    relion_data.reshape(-1),
                    mstep_accumulator_shape,
                )
                / (n**2)
            )
            reconstruction_Ft_ctf = (
                half_volume_mstep.relion_x_half_volume_to_native_half(
                    relion_weight.reshape(-1),
                    mstep_accumulator_shape,
                ).real
                / (n**4)
            )
            accumulator_comparison = {
                "policy": (
                    "streamed float64/complex128 comparison after RELION-to-RECOVAR "
                    "layout and unit conversion"
                ),
                "source": "recovar_joined_accumulator",
                "target": "native_relion_joined_accumulator",
                "numerator": streaming_field_metrics(
                    dump["Ft_y"], reconstruction_Ft_y
                ),
                "denominator": streaming_field_metrics(
                    dump["Ft_ctf"], reconstruction_Ft_ctf
                ),
            }
            accumulator_source = str(prefix)
            if args.compare_relion_boundary_spectra:
                spectrum_comparison = compare_relion_boundary_spectra(
                    dump,
                    prefix,
                    grid_size=int(volume_shape[0]),
                )
        elif args.compare_relion_boundary_spectra:
            raise ValueError(
                "--compare-relion-boundary-spectra requires --relion-bpref-prefix"
            )

        reconstruction_tau = final_tau
        reconstruction_tau_is_1d = False
        tau2_source = "recovar_derived"
        if args.relion_tau2_bin is not None:
            relion_tau2 = read_relion_spectrum(args.relion_tau2_bin.resolve())
            if relion_tau2.size < volume_shape[0] // 2 - 1:
                raise ValueError(
                    f"RELION tau2 has {relion_tau2.size} shells; expected at least {volume_shape[0] // 2 - 1}"
                )
            reconstruction_tau = relion_tau2 * float(volume_shape[0] ** 4)
            reconstruction_tau_is_1d = True
            tau2_source = str(args.relion_tau2_bin.resolve())

        # Match the production output path exactly.  The final refinement keeps
        # the reconstruction in Fourier space, then run_full_refinement applies
        # get_idft3 immediately before writing the MRC.  Returning real space
        # directly takes a different large-grid FFT path and is not an inert
        # replay at box 384.
        replay_map_ft = relion_functions.post_process_from_filter_v2(
            reconstruction_Ft_ctf,
            reconstruction_Ft_y,
            volume_shape,
            padding_factor,
            tau=reconstruction_tau,
            kernel="triangular",
            use_spherical_mask=True,
            grid_correct=grid_correct,
            gridding_correct="radial",
            kernel_width=1,
            tau2_fudge=tau2_fudge,
            gridding_padding_factor=projection_padding_factor,
            minres_map=int(args.minres_map),
            current_size=current_size,
            return_real_space=False,
            accumulator_volume_shape=mstep_accumulator_shape,
            tau_is_1d=reconstruction_tau_is_1d,
        )
        replay_map = fourier_transform_utils.get_idft3(
            replay_map_ft.reshape(volume_shape)
        ).real
        replay_map_np = np.asarray(replay_map, dtype=np.float32).reshape(volume_shape)
        summary: dict[str, Any] = {
            "dump": str(dump_path),
            "volume_shape": list(volume_shape),
            "mstep_accumulator_shape": list(mstep_accumulator_shape),
            "current_size": current_size,
            "padding_factor": padding_factor,
            "projection_padding_factor": projection_padding_factor,
            "grid_correct": grid_correct,
            "tau2_fudge": tau2_fudge,
            "tau2_fsc_mode": args.tau2_fsc_mode,
            "tau2_weight_combination": "sum",
            "tau2_full_half_axis": tau2_full_half_axis,
            "voxel_size": voxel_size,
            "minres_map": int(args.minres_map),
            "accumulator_source": accumulator_source,
            "tau2_source": tau2_source,
            "tau2_is_1d": reconstruction_tau_is_1d,
            "fsc_source": "recomputed" if args.recompute_fsc or "fsc_shells" not in dump else "dump",
            "fsc_head": _head(fsc_shells),
            "fsc_tail": _tail(fsc_shells),
            "tau2_fsc_used_tail": _tail(tau2_fsc_used),
            "tau2_prior_shells_head": _head(tau_details["prior_shells"]),
            "tau2_prior_shells_tail": _tail(tau_details["prior_shells"]),
            "tau2_sigma2_shells_tail": _tail(tau_details["sigma2_shells"]),
            "tau2_avg_weight_shells_tail": _tail(tau_details["avg_weight_shells"]),
            "tau2_shell_count_tail": _tail(tau_details["shell_count"]),
        }
        if accumulator_comparison is not None:
            summary["accumulator_comparison"] = accumulator_comparison
        if spectrum_comparison is not None:
            summary["spectrum_comparison"] = spectrum_comparison
        if "tau2_prior_shells" in dump:
            dumped_tau = np.asarray(dump["tau2_prior_shells"], dtype=np.float64)
            replay_tau = np.asarray(tau_details["prior_shells"], dtype=np.float64)
            n = min(dumped_tau.size, replay_tau.size)
            diff = replay_tau[:n] - dumped_tau[:n]
            summary["dumped_tau2_prior_shells_tail"] = _tail(dumped_tau)
            summary["replay_minus_dumped_tau2_prior_shells_absmax"] = float(np.max(np.abs(diff))) if n else float("nan")

    if args.output_mrc is not None:
        args.output_mrc.parent.mkdir(parents=True, exist_ok=True)
        helpers.write_mrc(str(args.output_mrc), replay_map_np, voxel_size=summary.get("voxel_size"))
        summary["output_mrc"] = str(args.output_mrc)

    if args.gt_mrc is not None:
        gt = (
            helpers.load_relion_volume(str(args.gt_mrc))
            if args.gt_relion_convention
            else helpers.load_mrc(str(args.gt_mrc))
        )
        summary["metrics_vs_gt"] = map_metrics(replay_map_np, gt)

    if args.relion_mrc is not None:
        relion = helpers.load_relion_volume(str(args.relion_mrc))
        summary["metrics_vs_relion"] = map_metrics(replay_map_np, relion)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
