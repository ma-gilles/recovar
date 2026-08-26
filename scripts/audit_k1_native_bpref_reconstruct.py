#!/usr/bin/env python3
"""Reconstruct one saved K=1 BPref half with RELION's native solver.

This is a boundary diagnostic.  It converts RECOVAR's saved odd, centered,
full Fourier accumulator back to RELION's packed-x BPref layout, calls the
pinned ``BackProjector::reconstruct`` binding, and compares that result with
the recorded RECOVAR and RELION numbered maps using signed shellwise FSC.

RELION's first-iteration CC path reconstructs with the untapered spectrum
returned by ``updateSSNRarrays`` and only then tapers the model-state tau2.
Consequently this tool requires an explicit pre-reconstruction tau2 artifact;
``tau2_radial_iter_000`` from ``refinement_results.npz`` is not a valid input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from recovar.core import mask
from recovar.utils import helpers
from scripts.summarize_em_completion_bench import normalized_fsc_auc, shell_fsc


SCHEMA = "em_k1_native_bpref_reconstruct_audit_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _metric(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    shellwise: dict[str, np.ndarray],
    key: str,
) -> dict[str, Any]:
    curve = np.asarray(shell_fsc(candidate, reference), dtype=np.float64)
    shellwise[key] = curve
    return {
        "signed_fsc_auc": _finite(normalized_fsc_auc(curve)),
        "sign_flipped_fsc_auc": _finite(normalized_fsc_auc(-curve)),
        "n_shells": int(curve.size),
        "shellwise_key": key,
    }


def _infer_odd_cube(size: int) -> int:
    side = int(round(float(size) ** (1.0 / 3.0)))
    _require(side**3 == int(size) and side % 2 == 1, f"expected odd cube, got {size}")
    return side


def _public_full_to_relion_x_half(values: np.ndarray, side: int) -> np.ndarray:
    """Invert RECOVAR's RELION-x-half-to-public-full layout conversion."""

    public_full = np.asarray(values).reshape(side, side, side)
    relion_full = public_full.transpose(2, 1, 0)
    packed_x = np.arange(side // 2, side, dtype=np.intp)
    return np.ascontiguousarray(relion_full[:, :, packed_x])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ft-y", type=Path, required=True)
    parser.add_argument("--ft-ctf", type=Path, required=True)
    parser.add_argument("--tau2-pre-reconstruction", type=Path, required=True)
    parser.add_argument("--refinement-results", type=Path, required=True)
    parser.add_argument("--recovar-map", type=Path, required=True)
    parser.add_argument("--relion-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration-index", type=int, required=True)
    parser.add_argument("--particle-diameter-ang", type=float, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--padding-factor", type=int, default=2)
    parser.add_argument("--minres-map", type=float, default=5.0)
    parser.add_argument("--max-iter-preweight", type=int, default=10)
    parser.add_argument("--skip-gridding", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inputs = (
        args.ft_y,
        args.ft_ctf,
        args.tau2_pre_reconstruction,
        args.refinement_results,
        args.recovar_map,
        args.relion_map,
    )
    for path in inputs:
        _require(path.is_file(), f"missing input: {path}")
    _require(args.iteration_index >= 0, "iteration index must be nonnegative")

    ft_y = np.load(args.ft_y, mmap_mode="r")
    ft_ctf = np.load(args.ft_ctf, mmap_mode="r")
    _require(ft_y.ndim == ft_ctf.ndim == 1, "K=1 accumulators must be flat")
    _require(ft_y.shape == ft_ctf.shape, "accumulator shapes disagree")
    _require(np.issubdtype(ft_y.dtype, np.complexfloating), "Ft_y must be complex")
    accumulator_side = _infer_odd_cube(ft_y.size)

    with np.load(args.refinement_results, allow_pickle=False) as results:
        current_sizes = np.asarray(results["current_sizes"], dtype=np.int64)
        _require(args.iteration_index < current_sizes.size, "iteration index is unavailable")
        current_size = int(current_sizes[args.iteration_index])
        tau2_fudge = float(np.asarray(results["tau2_fudge"]))
        git_commit = str(np.asarray(results["git_commit"]).item())

    tau2_recovar = np.asarray(np.load(args.tau2_pre_reconstruction), dtype=np.float64)
    _require(tau2_recovar.ndim == 1, "pre-reconstruction K=1 tau2 must be one-dimensional")
    _require(np.all(np.isfinite(tau2_recovar)), "pre-reconstruction tau2 must be finite")
    _require(np.all(tau2_recovar >= 0.0), "pre-reconstruction tau2 must be nonnegative")

    r_max = current_size // 2
    expected_side = 2 * (int(args.padding_factor * r_max + 0.5) + 1) + 1
    _require(
        accumulator_side == expected_side,
        f"accumulator side {accumulator_side} != RELION side {expected_side}",
    )

    recovar_map = np.asarray(helpers.load_mrc(str(args.recovar_map)), dtype=np.float64)
    relion_map = np.asarray(helpers.load_relion_volume(str(args.relion_map)), dtype=np.float64)
    _require(recovar_map.shape == relion_map.shape, "numbered map shapes disagree")
    _require(recovar_map.ndim == 3 and len(set(recovar_map.shape)) == 1, "maps must be cubic")
    ori_size = int(recovar_map.shape[0])
    n2 = float(ori_size**2)
    n4 = float(ori_size**4)

    data_native = np.asarray(
        _public_full_to_relion_x_half(ft_y, accumulator_side) * (-n2),
        dtype=np.complex128,
    )
    weight_native = np.asarray(
        _public_full_to_relion_x_half(ft_ctf, accumulator_side).real * n4,
        dtype=np.float64,
    )
    weight_native[np.abs(weight_native) < 1.0e-15] = 0.0
    tau2_relion = tau2_recovar / n4

    from recovar.relion_bind import _relion_bind_core as relion_bind

    reconstructed_relion = np.asarray(
        relion_bind.reconstruct_from_bpref(
            data_native,
            weight_native,
            tau2_relion,
            ori_size=ori_size,
            padding_factor=int(args.padding_factor),
            interpolator=relion_bind.TRILINEAR,
            do_map=True,
            max_iter_preweight=int(args.max_iter_preweight),
            tau2_fudge=tau2_fudge,
            skip_gridding=bool(args.skip_gridding),
            current_size=current_size,
            r_max=r_max,
            normalise=1.0,
            minres_map=float(args.minres_map),
        ),
        dtype=np.float64,
    )
    _require(reconstructed_relion.shape == recovar_map.shape, "native result shape changed")
    reconstructed_recovar = helpers.relion_volume_to_recovar(reconstructed_relion)

    flatten_radius = float(args.particle_diameter_ang) / (2.0 * float(args.voxel_size))
    solvent_mask = np.asarray(
        mask.raised_cosine_mask(
            reconstructed_recovar.shape,
            radius=flatten_radius,
            radius_p=flatten_radius + 5.0,
            offset=np.zeros(3),
        ),
        dtype=np.float64,
    )
    reconstructed_masked = reconstructed_recovar * solvent_mask

    shellwise: dict[str, np.ndarray] = {}
    metrics = {
        "native_raw_vs_recovar": _metric(
            reconstructed_recovar, recovar_map, shellwise=shellwise, key="native_raw_vs_recovar"
        ),
        "native_masked_vs_recovar": _metric(
            reconstructed_masked,
            recovar_map,
            shellwise=shellwise,
            key="native_masked_vs_recovar",
        ),
        "native_raw_vs_relion": _metric(
            reconstructed_recovar, relion_map, shellwise=shellwise, key="native_raw_vs_relion"
        ),
        "native_masked_vs_relion": _metric(
            reconstructed_masked,
            relion_map,
            shellwise=shellwise,
            key="native_masked_vs_relion",
        ),
        "recovar_vs_relion": _metric(
            recovar_map, relion_map, shellwise=shellwise, key="recovar_vs_relion"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_dir / "native_reconstruction_recovar_frame.npy",
        reconstructed_recovar.astype(np.float32),
    )
    np.save(
        args.output_dir / "native_reconstruction_masked_recovar_frame.npy",
        reconstructed_masked.astype(np.float32),
    )
    np.savez(args.output_dir / "shellwise_fsc.npz", **shellwise)
    binding_path = Path(relion_bind.__file__).resolve()
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "classification": "diagnostic_only_no_production_change",
        "quality_metric_policy": "signed shellwise FSC and normalized non-DC FSC-AUC",
        "source": {
            "artifact_git_commit": git_commit,
            "binding_path": str(binding_path),
            "binding_sha256": _sha256(binding_path),
        },
        "inputs": {str(path.name): {"path": str(path.resolve()), "sha256": _sha256(path)} for path in inputs},
        "parameters": {
            "iteration_index": args.iteration_index,
            "ori_size": ori_size,
            "current_size": current_size,
            "r_max": r_max,
            "padding_factor": args.padding_factor,
            "accumulator_side": accumulator_side,
            "packed_shape": list(data_native.shape),
            "tau2_fudge": tau2_fudge,
            "tau2_provenance": "explicit_live_pre_reconstruction_updateSSNRarrays",
            "data_frame_scale": -n2,
            "weight_frame_scale": n4,
            "tau2_frame_scale": 1.0 / n4,
            "particle_diameter_ang": args.particle_diameter_ang,
            "voxel_size": args.voxel_size,
            "minres_map": args.minres_map,
            "max_iter_preweight": args.max_iter_preweight,
            "skip_gridding": args.skip_gridding,
        },
        "metrics": metrics,
    }
    report_path = args.output_dir / "AUDIT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
