#!/usr/bin/env python3
"""Rebuild the untapered K=1 updateSSNRarrays spectrum from saved BPref arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from recovar.reconstruction import regularization


SCHEMA = "em_k1_pre_reconstruction_tau2_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_odd_cube(size: int) -> tuple[int, int, int]:
    side = int(round(float(size) ** (1.0 / 3.0)))
    if side**3 != int(size) or side % 2 != 1:
        raise ValueError(f"expected an odd full-cube accumulator, got {size} values")
    return (side, side, side)


def _load_flat(path: Path, *, complex_values: bool) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    if values.ndim != 1:
        raise ValueError(f"expected a flat accumulator in {path}, got {values.shape}")
    if complex_values and not np.issubdtype(values.dtype, np.complexfloating):
        raise TypeError(f"expected a complex accumulator in {path}, got {values.dtype}")
    if not complex_values and not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"expected a real accumulator in {path}, got {values.dtype}")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ft-y-half1", type=Path, required=True)
    parser.add_argument("--ft-y-half2", type=Path, required=True)
    parser.add_argument("--ft-ctf-half1", type=Path, required=True)
    parser.add_argument("--ft-ctf-half2", type=Path, required=True)
    parser.add_argument("--refinement-results", type=Path, required=True)
    parser.add_argument("--iteration-index", type=int, required=True)
    parser.add_argument("--padding-factor", type=int, default=2)
    parser.add_argument(
        "--full-half-axis",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help=(
            "Public full-volume axis containing RELION's stored half-complex values. "
            "Saved x-half BPref accumulators are transposed to public axis 0."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inputs = (
        args.ft_y_half1,
        args.ft_y_half2,
        args.ft_ctf_half1,
        args.ft_ctf_half2,
        args.refinement_results,
    )
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"missing input: {path}")
    if args.iteration_index < 0:
        raise ValueError("iteration index must be nonnegative")
    if args.padding_factor <= 0:
        raise ValueError("padding factor must be positive")

    ft_y = [
        _load_flat(args.ft_y_half1, complex_values=True),
        _load_flat(args.ft_y_half2, complex_values=True),
    ]
    ft_ctf = [
        _load_flat(args.ft_ctf_half1, complex_values=False),
        _load_flat(args.ft_ctf_half2, complex_values=False),
    ]
    sizes = {array.size for array in (*ft_y, *ft_ctf)}
    if len(sizes) != 1:
        raise ValueError(f"BPref accumulator sizes disagree: {sorted(sizes)}")
    accumulator_shape = _infer_odd_cube(sizes.pop())

    with np.load(args.refinement_results, allow_pickle=False) as results:
        current_sizes = np.asarray(results["current_sizes"], dtype=np.int64)
        if args.iteration_index >= current_sizes.size:
            raise ValueError("requested iteration is unavailable")
        current_size = int(current_sizes[args.iteration_index])
        volume_shape = tuple(int(value) for value in np.asarray(results["volume_shape"]))
        tau2_fudge = float(np.asarray(results["tau2_fudge"]))
        git_commit = str(np.asarray(results["git_commit"]).item())

    r_max = current_size // 2
    fsc = regularization.compute_relion_fsc_from_backprojector(
        ft_y[0],
        ft_y[1],
        ft_ctf[0],
        ft_ctf[1],
        volume_shape,
        padding_factor=args.padding_factor,
        r_max=r_max,
        accumulator_volume_shape=accumulator_shape,
    )
    tau2_shells = []
    details = []
    for weight in ft_ctf:
        _, _, detail = regularization.compute_relion_tau2_from_weights(
            weight,
            weight,
            fsc,
            volume_shape,
            tau2_fudge=tau2_fudge,
            padding_factor=args.padding_factor,
            r_max=r_max,
            return_details=True,
            full_half_axis=args.full_half_axis,
            accumulator_volume_shape=accumulator_shape,
        )
        tau2_shells.append(np.asarray(detail["prior_shells"], dtype=np.float32))
        details.append(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for half, values in enumerate(tau2_shells, start=1):
        path = args.output_dir / f"tau2_pre_reconstruction_half{half}.npy"
        np.save(path, values)
        output_paths.append(path)
    fsc_path = args.output_dir / "backprojector_fsc.npy"
    np.save(fsc_path, np.asarray(fsc, dtype=np.float64))
    output_paths.append(fsc_path)

    report = {
        "schema": SCHEMA,
        "status": "complete",
        "provenance": "recomputed from sealed post-join BPref accumulators before firstiter tau2 taper",
        "artifact_git_commit": git_commit,
        "parameters": {
            "iteration_index": args.iteration_index,
            "current_size": current_size,
            "r_max": r_max,
            "padding_factor": args.padding_factor,
            "volume_shape": list(volume_shape),
            "accumulator_shape": list(accumulator_shape),
            "tau2_fudge": tau2_fudge,
            "full_half_axis": args.full_half_axis,
        },
        "metrics": {
            "fsc_shell_count": int(np.asarray(fsc).size),
            "fsc_shell1": float(np.asarray(fsc)[1]),
            "tau2_half1_max": float(np.max(tau2_shells[0])),
            "tau2_half2_max": float(np.max(tau2_shells[1])),
            "tau2_shell_count": int(tau2_shells[0].size),
        },
        "inputs": {
            path.name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in inputs
        },
        "outputs": {
            path.name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in output_paths
        },
    }
    report_path = args.output_dir / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
