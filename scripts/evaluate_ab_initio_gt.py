#!/usr/bin/env python
"""Evaluate ab-initio / InitialModel MRC outputs against a GT map.

Native InitialModel writes RELION-frame MRCs, while benchmark GT maps are
usually stored in recovar/cryoSPARC frame.  This script makes the frame choice
explicit and reports both raw and alignment-aware GT metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.em.initial_model.gt_metrics import (
    DEFAULT_GT_ALIGN_HEALPIX_ORDER,
    DEFAULT_GT_ALIGN_MAX_SHELL,
    align_volume_to_reference,
    centered_correlation,
    first_shell_below_threshold,
    relion_alignment_rotations,
)
from recovar.reconstruction import regularization
from recovar.utils import helpers


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--volume", action="append", required=True, help="MRC volume to evaluate. Repeatable.")
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional metric label for each --volume. Defaults to sanitized MRC stems.",
    )
    parser.add_argument("--gt_volume", required=True, help="Ground-truth MRC volume.")
    parser.add_argument(
        "--volume_frame",
        choices=("relion", "recovar"),
        default="relion",
        help="Frame convention for --volume inputs. Native InitialModel outputs use relion.",
    )
    parser.add_argument(
        "--gt_frame",
        choices=("relion", "recovar"),
        default="recovar",
        help="Frame convention for --gt_volume.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Voxel size in Angstrom. If omitted, use GT header then volume header then 1.0.",
    )
    parser.add_argument("--output_npz", default=None, help="Optional .npz path for full FSC curves and metrics.")
    parser.add_argument("--output_json", default=None, help="Optional JSON path for scalar metric summary.")
    parser.add_argument("--gt_align", action="store_true", help="Also compute alignment-aware GT metrics.")
    parser.add_argument(
        "--gt_align_healpix_order",
        type=int,
        default=DEFAULT_GT_ALIGN_HEALPIX_ORDER,
        help="RELION/RECOVAR rotation-grid order used for GT alignment.",
    )
    parser.add_argument(
        "--gt_align_max_shell",
        type=int,
        default=DEFAULT_GT_ALIGN_MAX_SHELL,
        help="Maximum Fourier shell used to score coarse GT alignment.",
    )
    parser.add_argument(
        "--gt_align_no_mirror",
        action="store_true",
        help="Do not test the x-axis mirror handedness ambiguity during GT alignment.",
    )
    parser.add_argument(
        "--gt_align_allow_sign",
        action="store_true",
        help="Allow a global sign flip during GT alignment. Off by default.",
    )
    parser.add_argument(
        "--gt_align_refine_orders",
        type=int,
        nargs="*",
        default=[3, 4],
        help=("HEALPix orders for local rotation refinement after the coarse pass. Empty list disables refinement."),
    )
    parser.add_argument(
        "--gt_align_refine_sigma_deg",
        type=float,
        default=30.0,
        help="Angular radius (deg) used to keep nearby rotations during local refinement.",
    )
    parser.add_argument(
        "--print_per_shell_fsc",
        action="store_true",
        help="Print full per-shell FSC curve for each volume.",
    )
    return parser.parse_args(argv)


def _voxel_size_value(raw: Any) -> float | None:
    if raw is None:
        return None
    candidates: list[Any] = []
    if hasattr(raw, "x"):
        candidates.append(raw.x)
    candidates.append(raw)
    for candidate in candidates:
        try:
            value = float(np.asarray(candidate).reshape(-1)[0])
        except Exception:
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return None


def _load_volume(path: str | Path, frame: str) -> tuple[np.ndarray, float | None]:
    if frame == "relion":
        vol, voxel = helpers.load_relion_volume(str(path), return_voxel_size=True)
    elif frame == "recovar":
        vol, voxel = helpers.load_mrc(str(path), return_voxel_size=True)
    else:
        raise ValueError(f"Unknown volume frame: {frame!r}")
    return np.asarray(vol, dtype=np.float64), _voxel_size_value(voxel)


def _sanitize_label(label: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", label).strip("_")
    return sanitized or "volume"


def _labels_for(volumes: list[str], labels: list[str] | None) -> list[str]:
    if labels is not None and len(labels) != len(volumes):
        raise ValueError(f"Expected either zero labels or one per volume, got {len(labels)} labels for {len(volumes)}")
    raw_labels = labels if labels is not None else [Path(v).stem for v in volumes]
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in raw_labels:
        base = _sanitize_label(raw)
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count + 1}")
    return out


def _real_to_ft(volume: np.ndarray) -> np.ndarray:
    return np.asarray(ftu.get_dft3(jnp.asarray(volume))).reshape(-1)


def _fsc_against(volume_ft: np.ndarray, reference_ft: np.ndarray, volume_shape: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(
        regularization.get_fsc_gpu(
            jnp.asarray(volume_ft),
            jnp.asarray(reference_ft),
            volume_shape,
        ),
        dtype=np.float64,
    )


def _shell_resolution(shell_index: int, volume_size: int, voxel_size: float) -> float:
    if int(shell_index) <= 0:
        return float("nan")
    return float(volume_size) * float(voxel_size) / float(shell_index)


def _mean_fsc(fsc: np.ndarray, first_shell: int, last_shell: int) -> float:
    values = np.asarray(fsc, dtype=np.float64)
    lo = max(0, int(first_shell))
    hi = min(values.size, int(last_shell) + 1)
    if hi <= lo:
        return float("nan")
    return float(np.nanmean(values[lo:hi]))


def _add_metric_set(
    *,
    prefix: str,
    volume: np.ndarray,
    reference: np.ndarray,
    reference_ft: np.ndarray,
    volume_shape: tuple[int, int, int],
    voxel_size: float,
    npz_payload: dict[str, Any],
    json_payload: dict[str, Any],
) -> None:
    volume_ft = _real_to_ft(volume)
    fsc = _fsc_against(volume_ft, reference_ft, volume_shape)
    shell_05 = first_shell_below_threshold(fsc, 0.5)
    shell_0143 = first_shell_below_threshold(fsc, 0.143)

    corr = centered_correlation(volume, reference)
    npz_payload[f"{prefix}_corr_vs_gt"] = np.float64(corr)
    npz_payload[f"{prefix}_fsc_vs_gt"] = fsc
    npz_payload[f"{prefix}_shell_05"] = np.int32(shell_05)
    npz_payload[f"{prefix}_shell_0143"] = np.int32(shell_0143)
    npz_payload[f"{prefix}_resolution_05_A"] = np.float64(_shell_resolution(shell_05, volume_shape[0], voxel_size))
    npz_payload[f"{prefix}_resolution_0143_A"] = np.float64(_shell_resolution(shell_0143, volume_shape[0], voxel_size))
    npz_payload[f"{prefix}_mean_fsc_1_8"] = np.float64(_mean_fsc(fsc, 1, 8))
    npz_payload[f"{prefix}_mean_fsc_1_16"] = np.float64(_mean_fsc(fsc, 1, 16))

    json_payload.update(
        {
            "corr_vs_gt": float(corr),
            "shell_05": int(shell_05),
            "shell_0143": int(shell_0143),
            "resolution_05_A": _shell_resolution(shell_05, volume_shape[0], voxel_size),
            "resolution_0143_A": _shell_resolution(shell_0143, volume_shape[0], voxel_size),
            "mean_fsc_1_8": _mean_fsc(fsc, 1, 8),
            "mean_fsc_1_16": _mean_fsc(fsc, 1, 16),
            "fsc_vs_gt": [float(v) for v in fsc],
        }
    )


def evaluate(
    *,
    volume_paths: list[str],
    labels: list[str],
    gt_volume_path: str,
    volume_frame: str,
    gt_frame: str,
    voxel_size_override: float | None,
    gt_align: bool,
    gt_align_healpix_order: int,
    gt_align_max_shell: int,
    gt_align_allow_mirror: bool,
    gt_align_allow_sign: bool,
    gt_align_refine_orders: tuple[int, ...] = (),
    gt_align_refine_sigma_deg: float = 30.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_real, gt_voxel = _load_volume(gt_volume_path, gt_frame)
    if gt_real.ndim != 3 or len(set(gt_real.shape)) != 1:
        raise ValueError(f"GT volume must be cubic 3D, got shape {gt_real.shape}")

    volume_shape = tuple(int(x) for x in gt_real.shape)
    gt_ft = _real_to_ft(gt_real)
    rotations = relion_alignment_rotations(gt_align_healpix_order) if gt_align else None

    npz_payload: dict[str, Any] = {
        "gt_volume": np.asarray(str(gt_volume_path)),
        "gt_frame": np.asarray(str(gt_frame)),
        "volume_frame": np.asarray(str(volume_frame)),
        "volume_shape": np.asarray(volume_shape, dtype=np.int32),
        "gt_align_enabled": np.bool_(gt_align),
        "gt_align_healpix_order": np.int32(gt_align_healpix_order),
        "gt_align_max_shell": np.int32(gt_align_max_shell),
        "gt_align_allow_mirror": np.bool_(gt_align_allow_mirror),
        "gt_align_allow_sign": np.bool_(gt_align_allow_sign),
    }
    json_summary: dict[str, Any] = {
        "gt_volume": str(gt_volume_path),
        "gt_frame": gt_frame,
        "volume_frame": volume_frame,
        "volume_shape": list(volume_shape),
        "gt_align_enabled": bool(gt_align),
        "gt_align_healpix_order": int(gt_align_healpix_order),
        "gt_align_max_shell": int(gt_align_max_shell),
        "gt_align_allow_mirror": bool(gt_align_allow_mirror),
        "gt_align_allow_sign": bool(gt_align_allow_sign),
        "volumes": [],
    }

    for label, path in zip(labels, volume_paths):
        real, vol_voxel = _load_volume(path, volume_frame)
        if real.shape != gt_real.shape:
            raise ValueError(f"{path} shape {real.shape} does not match GT shape {gt_real.shape}")
        voxel_size = voxel_size_override or gt_voxel or vol_voxel or 1.0
        npz_payload["voxel_size"] = np.float64(voxel_size)
        json_summary["voxel_size"] = float(voxel_size)

        per_volume: dict[str, Any] = {"label": label, "path": str(path)}
        npz_payload[f"{label}_path"] = np.asarray(str(path))
        _add_metric_set(
            prefix=label,
            volume=real,
            reference=gt_real,
            reference_ft=gt_ft,
            volume_shape=volume_shape,
            voxel_size=float(voxel_size),
            npz_payload=npz_payload,
            json_payload=per_volume,
        )

        if gt_align:
            assert rotations is not None
            alignment = align_volume_to_reference(
                real,
                gt_real,
                rotations,
                score_max_shell=int(gt_align_max_shell),
                allow_mirror=bool(gt_align_allow_mirror),
                allow_sign=bool(gt_align_allow_sign),
                refine_orders=tuple(int(o) for o in gt_align_refine_orders) or None,
                refine_sigma_deg=float(gt_align_refine_sigma_deg),
            )
            aligned_prefix = f"{label}_aligned"
            _add_metric_set(
                prefix=aligned_prefix,
                volume=alignment.aligned_volume,
                reference=gt_real,
                reference_ft=gt_ft,
                volume_shape=volume_shape,
                voxel_size=float(voxel_size),
                npz_payload=npz_payload,
                json_payload=per_volume.setdefault("aligned", {}),
            )
            npz_payload[f"{label}_gt_align_rotation_index"] = np.int32(alignment.rotation_index)
            npz_payload[f"{label}_gt_align_rotation_matrix"] = alignment.rotation_matrix
            npz_payload[f"{label}_gt_align_mirror_x"] = np.bool_(alignment.mirror_x)
            npz_payload[f"{label}_gt_align_sign"] = np.int32(alignment.sign)
            per_volume["aligned"].update(
                {
                    "rotation_index": int(alignment.rotation_index),
                    "rotation_matrix": np.asarray(alignment.rotation_matrix).tolist(),
                    "mirror_x": bool(alignment.mirror_x),
                    "sign": int(alignment.sign),
                    "score_vs_gt": float(alignment.score),
                }
            )

        json_summary["volumes"].append(per_volume)

    return npz_payload, json_summary


def _print_per_shell(label: str, fsc: list[float]) -> None:
    n = len(fsc)
    # Print every shell up to where FSC drops below 0.143, then sample to end.
    for chunk_start in range(0, n, 16):
        chunk_end = min(chunk_start + 16, n)
        header = "    " + "shell:" + " ".join(f"{s:>5d}" for s in range(chunk_start, chunk_end))
        body = "    " + f"{label:<6s}" + " ".join(f"{fsc[s]:>5.2f}" for s in range(chunk_start, chunk_end))
        print(header)
        print(body)


def _print_summary(summary: dict[str, Any], *, print_per_shell_fsc: bool = False) -> None:
    align = "aligned" if summary["gt_align_enabled"] else "raw-only"
    print(
        "Ab-initio GT evaluation: "
        f"frame={summary['volume_frame']} gt_frame={summary['gt_frame']} "
        f"voxel={summary.get('voxel_size', 1.0):.6g} A align={align}"
    )
    header = f"{'label':<28s} {'corr':>10s} {'fsc1-8':>10s} {'fsc1-16':>10s} {'0.5 shell':>9s} {'0.143 shell':>11s}"
    print(header)
    print("-" * len(header))
    for item in summary["volumes"]:
        print(
            f"{item['label']:<28s} {item['corr_vs_gt']:10.6f} {item['mean_fsc_1_8']:10.6f} "
            f"{item['mean_fsc_1_16']:10.6f} {item['shell_05']:9d} {item['shell_0143']:11d}"
        )
        aligned = item.get("aligned")
        if aligned:
            print(
                f"{item['label'] + ' aligned':<28s} {aligned['corr_vs_gt']:10.6f} "
                f"{aligned['mean_fsc_1_8']:10.6f} {aligned['mean_fsc_1_16']:10.6f} "
                f"{aligned['shell_05']:9d} {aligned['shell_0143']:11d}"
            )
        if print_per_shell_fsc:
            print(f"  per-shell FSC vs GT — {item['label']}:")
            _print_per_shell("raw", item.get("fsc_vs_gt", []))
            if aligned and "fsc_vs_gt" in aligned:
                print(f"  per-shell FSC vs GT — {item['label']} aligned:")
                _print_per_shell("alig", aligned["fsc_vs_gt"])


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    labels = _labels_for(args.volume, args.label)
    npz_payload, json_summary = evaluate(
        volume_paths=[str(v) for v in args.volume],
        labels=labels,
        gt_volume_path=str(args.gt_volume),
        volume_frame=str(args.volume_frame),
        gt_frame=str(args.gt_frame),
        voxel_size_override=args.voxel_size,
        gt_align=bool(args.gt_align),
        gt_align_healpix_order=int(args.gt_align_healpix_order),
        gt_align_max_shell=int(args.gt_align_max_shell),
        gt_align_allow_mirror=not bool(args.gt_align_no_mirror),
        gt_align_allow_sign=bool(args.gt_align_allow_sign),
        gt_align_refine_orders=tuple(int(o) for o in (args.gt_align_refine_orders or [])),
        gt_align_refine_sigma_deg=float(args.gt_align_refine_sigma_deg),
    )

    if args.output_npz:
        out_npz = Path(args.output_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, **npz_payload)
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(json_summary, indent=2, sort_keys=True) + "\n")

    _print_summary(json_summary, print_per_shell_fsc=bool(args.print_per_shell_fsc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
