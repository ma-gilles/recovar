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


def shell_fsc(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
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
    return out[: n // 2 + 1]


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

        replay_map = relion_functions.post_process_from_filter_v2(
            dump["Ft_ctf"],
            dump["Ft_y"],
            volume_shape,
            padding_factor,
            tau=final_tau,
            kernel="triangular",
            use_spherical_mask=True,
            grid_correct=grid_correct,
            gridding_correct="radial",
            kernel_width=1,
            tau2_fudge=tau2_fudge,
            gridding_padding_factor=projection_padding_factor,
            minres_map=int(args.minres_map),
            current_size=current_size,
            return_real_space=True,
            accumulator_volume_shape=mstep_accumulator_shape,
        )

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
