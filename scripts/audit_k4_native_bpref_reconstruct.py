#!/usr/bin/env python3
"""Route a saved K-class RECOVAR accumulator through RELION reconstruction.

The dense Class3D path exposes an odd, centered, full Fourier accumulator in
RECOVAR public ``(x, y, z)`` order.  This diagnostic inverts the production
RELION-x-half expansion, converts the unnormalised RECOVAR FFT frame to native
RELION BPref units, and calls the compiled
``BackProjector::reconstruct`` binding.  Quality is reported only with
shellwise FSC and normalized FSC-AUC.
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
from recovar.relion_bind import _relion_bind_core as relion_bind
from recovar.utils import helpers

if __package__:
    from scripts.summarize_em_completion_bench import normalized_fsc_auc, shell_fsc
else:
    from summarize_em_completion_bench import normalized_fsc_auc, shell_fsc


SCHEMA = "em_k4_native_bpref_reconstruct_audit_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ft-y", type=Path, required=True)
    parser.add_argument("--ft-ctf", type=Path, required=True)
    parser.add_argument("--refinement-results", type=Path, required=True)
    parser.add_argument("--recovar-map", type=Path, required=True)
    parser.add_argument("--relion-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration-index", type=int, required=True)
    parser.add_argument("--class-index", type=int, required=True, help="One-based class index.")
    parser.add_argument("--particle-diameter-ang", type=float, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--padding-factor", type=int, default=2)
    parser.add_argument("--minres-map", type=float, default=5.0)
    parser.add_argument("--max-iter-preweight", type=int, default=10)
    parser.add_argument("--skip-gridding", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _metric(lhs: np.ndarray, rhs: np.ndarray, *, shellwise: dict[str, np.ndarray], key: str) -> dict[str, Any]:
    curve = np.asarray(shell_fsc(lhs, rhs), dtype=np.float64)
    shellwise[key] = curve
    return {
        "fsc_auc": _finite(normalized_fsc_auc(curve)),
        "n_shells": int(curve.size),
        "shellwise_key": key,
    }


def _infer_odd_cube(size: int) -> int:
    side = int(round(float(size) ** (1.0 / 3.0)))
    if side**3 != int(size) or side % 2 != 1:
        raise ValueError(f"expected an odd cubic accumulator, got flat size={size}")
    return side


def _public_full_to_relion_x_half(values: np.ndarray, side: int) -> np.ndarray:
    """Invert ``relion_x_half_volume_to_full`` for an odd accumulator."""

    public_full = np.asarray(values).reshape(side, side, side)
    relion_full = public_full.transpose(2, 1, 0)
    packed_x = np.arange(side // 2, side, dtype=np.intp)
    return np.ascontiguousarray(relion_full[:, :, packed_x])


def main() -> int:
    args = _parse_args()
    for path in (
        args.ft_y,
        args.ft_ctf,
        args.refinement_results,
        args.recovar_map,
        args.relion_map,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.class_index < 1:
        raise ValueError("--class-index must be one-based and positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_offset = int(args.class_index) - 1
    ft_y_all = np.load(args.ft_y, mmap_mode="r")
    ft_ctf_all = np.load(args.ft_ctf, mmap_mode="r")
    if ft_y_all.shape != ft_ctf_all.shape or ft_y_all.ndim != 2:
        raise ValueError(f"accumulator shapes disagree: {ft_y_all.shape} vs {ft_ctf_all.shape}")
    if class_offset >= ft_y_all.shape[0]:
        raise ValueError(f"class {args.class_index} unavailable in shape {ft_y_all.shape}")

    accumulator_side = _infer_odd_cube(ft_y_all.shape[1])
    with np.load(args.refinement_results, allow_pickle=False) as results:
        current_sizes = np.asarray(results["current_sizes"], dtype=np.int64)
        tau2_fudge = float(np.asarray(results["tau2_fudge"]))
        git_commit = str(np.asarray(results["git_commit"]).item())
        tau2_key = f"tau2_radial_iter_{args.iteration_index:03d}"
        if tau2_key not in results.files:
            raise KeyError(f"missing {tau2_key} in {args.refinement_results}")
        tau2_recovar = np.asarray(results[tau2_key][class_offset], dtype=np.float64)
    if args.iteration_index < 0 or args.iteration_index >= current_sizes.size:
        raise ValueError(f"iteration index {args.iteration_index} outside current_sizes")
    current_size = int(current_sizes[args.iteration_index])
    r_max = current_size // 2
    expected_side = 2 * (int(args.padding_factor * r_max + 0.5) + 1) + 1
    if accumulator_side != expected_side:
        raise ValueError(
            f"accumulator side {accumulator_side} != explicit RELION pad size {expected_side} "
            f"for current_size={current_size}, r_max={r_max}, padding_factor={args.padding_factor}"
        )

    ori_size = int(np.asarray(helpers.load_mrc(str(args.recovar_map))).shape[0])
    n2 = float(ori_size**2)
    n4 = float(ori_size**4)
    data_native = _public_full_to_relion_x_half(ft_y_all[class_offset], accumulator_side)
    weight_native = _public_full_to_relion_x_half(ft_ctf_all[class_offset], accumulator_side).real
    bp_data = np.asarray(data_native * (-n2), dtype=np.complex128)
    bp_weight = np.asarray(weight_native * n4, dtype=np.float64)
    bp_weight[np.abs(bp_weight) < 1.0e-15] = 0.0
    tau2_relion = np.asarray(tau2_recovar / n4, dtype=np.float64)

    reconstructed_relion = np.asarray(
        relion_bind.reconstruct_from_bpref(
            bp_data,
            bp_weight,
            tau2_relion,
            ori_size=ori_size,
            padding_factor=int(args.padding_factor),
            interpolator=relion_bind.TRILINEAR,
            do_map=True,
            max_iter_preweight=int(args.max_iter_preweight),
            tau2_fudge=float(tau2_fudge),
            skip_gridding=bool(args.skip_gridding),
            current_size=current_size,
            r_max=r_max,
            normalise=1.0,
            minres_map=float(args.minres_map),
        ),
        dtype=np.float64,
    )
    if reconstructed_relion.shape != (ori_size, ori_size, ori_size):
        raise ValueError(f"native reconstruction returned unexpected shape {reconstructed_relion.shape}")
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
    recovar_map = np.asarray(helpers.load_mrc(str(args.recovar_map)), dtype=np.float64)
    relion_map = np.asarray(helpers.load_relion_volume(str(args.relion_map)), dtype=np.float64)

    shellwise: dict[str, np.ndarray] = {}
    metrics = {
        "native_raw_vs_recovar": _metric(
            reconstructed_recovar,
            recovar_map,
            shellwise=shellwise,
            key="native_raw_vs_recovar",
        ),
        "native_masked_vs_recovar": _metric(
            reconstructed_masked,
            recovar_map,
            shellwise=shellwise,
            key="native_masked_vs_recovar",
        ),
        "native_raw_vs_relion": _metric(
            reconstructed_recovar,
            relion_map,
            shellwise=shellwise,
            key="native_raw_vs_relion",
        ),
        "native_masked_vs_relion": _metric(
            reconstructed_masked,
            relion_map,
            shellwise=shellwise,
            key="native_masked_vs_relion",
        ),
        "recovar_vs_relion": _metric(
            recovar_map,
            relion_map,
            shellwise=shellwise,
            key="recovar_vs_relion",
        ),
    }
    np.save(args.output_dir / "native_reconstruction_recovar_frame.npy", reconstructed_recovar.astype(np.float32))
    np.save(args.output_dir / "native_reconstruction_masked_recovar_frame.npy", reconstructed_masked.astype(np.float32))
    np.savez(args.output_dir / "shellwise_fsc.npz", **shellwise)

    binding_path = Path(relion_bind.__file__).resolve()
    report = {
        "schema": SCHEMA,
        "quality_metric_policy": "shellwise FSC and normalized FSC-AUC only; correlation is not computed",
        "classification": "diagnostic_only_no_production_change",
        "source": {
            "artifact_git_commit": git_commit,
            "binding_path": str(binding_path),
            "binding_sha256": _sha256(binding_path),
        },
        "inputs": {
            "ft_y": str(args.ft_y.resolve()),
            "ft_y_sha256": _sha256(args.ft_y),
            "ft_ctf": str(args.ft_ctf.resolve()),
            "ft_ctf_sha256": _sha256(args.ft_ctf),
            "refinement_results": str(args.refinement_results.resolve()),
            "recovar_map": str(args.recovar_map.resolve()),
            "recovar_map_sha256": _sha256(args.recovar_map),
            "relion_map": str(args.relion_map.resolve()),
            "relion_map_sha256": _sha256(args.relion_map),
        },
        "parameters": {
            "iteration_index": int(args.iteration_index),
            "class_index": int(args.class_index),
            "ori_size": ori_size,
            "current_size": current_size,
            "r_max": r_max,
            "padding_factor": int(args.padding_factor),
            "accumulator_side": accumulator_side,
            "packed_shape": list(bp_data.shape),
            "tau2_fudge": tau2_fudge,
            "minres_map": float(args.minres_map),
            "max_iter_preweight": int(args.max_iter_preweight),
            "skip_gridding": bool(args.skip_gridding),
            "particle_diameter_ang": float(args.particle_diameter_ang),
            "voxel_size": float(args.voxel_size),
            "tau2_key": tau2_key,
            "data_frame_scale": -n2,
            "weight_frame_scale": n4,
            "tau2_frame_scale": 1.0 / n4,
        },
        "metrics": metrics,
    }
    report_path = args.output_dir / "AUDIT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
