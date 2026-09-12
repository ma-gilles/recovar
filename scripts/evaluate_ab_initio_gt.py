#!/usr/bin/env python
"""Evaluate ab-initio / InitialModel MRC outputs against a GT map.

Native InitialModel writes RELION-frame MRCs, while benchmark GT maps are
usually stored in recovar/cryoSPARC frame.  This script makes the frame choice
explicit and reports both raw and alignment-aware GT metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.em.vdam.gt_metrics import (
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
        "--gt_align_rigid",
        action="store_true",
        help=(
            "Opt in to rigid translation/rotation fitting with fixed contrast sign +1. "
            "Requires --gt_align; use shared transform options to fit once for a trajectory. Discrete refinement "
            "orders/sigma are inactive in this mode. Effective fit controls are saved."
        ),
    )
    shared = parser.add_mutually_exclusive_group()
    shared.add_argument(
        "--gt_align_fit_label",
        help="Fit this input label to GT once and apply its rigid transform to every input volume.",
    )
    shared.add_argument(
        "--gt_align_transform_json",
        help="Apply a previously saved rigid transform to every input volume without fitting.",
    )
    parser.add_argument(
        "--gt_align_transform_output",
        help="Save the common transform JSON; requires --gt_align_fit_label or --gt_align_transform_json.",
    )
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
    args = parser.parse_args(argv)
    if args.gt_align_rigid and not args.gt_align:
        parser.error("--gt_align_rigid requires --gt_align")
    if args.gt_align_rigid and args.gt_align_allow_sign:
        parser.error("--gt_align_rigid fixes contrast sign +1; omit --gt_align_allow_sign")
    if (
        args.gt_align_fit_label or args.gt_align_transform_json or args.gt_align_transform_output
    ) and not args.gt_align_rigid:
        parser.error("Shared transform options require --gt_align_rigid")
    if args.gt_align_transform_output and not (args.gt_align_fit_label or args.gt_align_transform_json):
        parser.error("--gt_align_transform_output requires a common transform")
    return args


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


def _load_volume(path: str | Path, frame: str, *, strict_voxel_grid: bool = False) -> tuple[np.ndarray, float | None]:
    if frame == "relion":
        vol, voxel = helpers.load_relion_volume(str(path), return_voxel_size=True)
    elif frame == "recovar":
        vol, voxel = helpers.load_mrc(str(path), return_voxel_size=True)
    else:
        raise ValueError(f"Unknown volume frame: {frame!r}")
    if strict_voxel_grid:
        axes = np.asarray([getattr(voxel, axis) for axis in "xyz"], dtype=np.float64)
        if not np.isfinite(axes).all() or np.any(axes <= 0) or not np.all(axes == axes[0]):
            raise ValueError(f"{path}: rigid registration requires a finite positive isotropic voxel grid")
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


def _common_rigid_transform(document: dict[str, Any]):
    """Load the evaluator's transport record without invoking the fitter."""
    from recovar.em.vdam.gt_registration import RigidVolumeTransform

    expected = {"transform", "identity_sha256", "fit_receipt", "fit_reference"}
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError("Invalid common rigid transform document")
    transform = RigidVolumeTransform.from_dict(document["transform"])
    if document["identity_sha256"] != transform.identity_sha256:
        raise ValueError("Rigid transform identity does not match its contents")
    if not isinstance(document["fit_receipt"], dict) or not isinstance(document["fit_reference"], dict):
        raise ValueError("Rigid transform fit metadata must be objects")
    return transform


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
    gt_align_rigid: bool = False,
    gt_align_fit_label: str | None = None,
    gt_align_transform: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if gt_align_rigid and (not gt_align or gt_align_allow_sign):
        raise ValueError("Rigid alignment requires gt_align=True and gt_align_allow_sign=False")
    if (gt_align_fit_label is not None or gt_align_transform is not None) and not gt_align_rigid:
        raise ValueError("Shared transforms require rigid alignment")
    if gt_align_fit_label is not None and gt_align_transform is not None:
        raise ValueError("Select either a fit label or an existing transform")
    if gt_align_rigid and (not volume_paths or len(volume_paths) != len(labels) or len(set(labels)) != len(labels)):
        raise ValueError("Rigid alignment requires one unique label per input volume")
    if (
        gt_align_rigid
        and voxel_size_override is not None
        and (not math.isfinite(voxel_size_override) or voxel_size_override <= 0)
    ):
        raise ValueError("Voxel size override must be finite and positive")
    gt_real, gt_voxel = _load_volume(gt_volume_path, gt_frame, strict_voxel_grid=gt_align_rigid)
    if gt_real.ndim != 3 or len(set(gt_real.shape)) != 1:
        raise ValueError(f"GT volume must be cubic 3D, got shape {gt_real.shape}")

    volume_shape = tuple(int(x) for x in gt_real.shape)
    gt_ft = _real_to_ft(gt_real)
    rotations = relion_alignment_rotations(gt_align_healpix_order) if gt_align and gt_align_transform is None else None

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

    shared_transform = None
    shared_document = gt_align_transform
    if gt_align_rigid:
        from recovar.em.vdam.gt_registration import (
            RigidFitControls,
            RigidVolumeTransform,
            align_volume_rigid_to_reference,
        )

        gt_identity = hashlib.sha256(np.ascontiguousarray(gt_real, dtype="<f8").tobytes()).hexdigest()
        if gt_align_fit_label is not None:
            if gt_align_fit_label not in labels:
                raise ValueError(f"Unknown rigid fit label: {gt_align_fit_label!r}")
            fit_path = volume_paths[labels.index(gt_align_fit_label)]
            fit_volume, fit_voxel = _load_volume(fit_path, volume_frame, strict_voxel_grid=True)
            if fit_volume.shape != gt_real.shape or fit_voxel != gt_voxel:
                raise ValueError("Rigid fit reference and GT must have matching shapes and voxel grids")
            fitted = align_volume_rigid_to_reference(
                fit_volume,
                gt_real,
                rotations,
                controls=RigidFitControls(
                    score_max_shell=int(gt_align_max_shell), allow_mirror=bool(gt_align_allow_mirror)
                ),
            )
            shared_transform = RigidVolumeTransform.from_alignment(
                fitted,
                volume_shape=volume_shape,
                voxel_size=float(voxel_size_override or gt_voxel),
                gt_sha256=gt_identity,
            )
            shared_document = {
                "transform": shared_transform.to_dict(),
                "identity_sha256": shared_transform.identity_sha256,
                "fit_receipt": json.loads(json.dumps(asdict(fitted.receipt), allow_nan=False)),
                "fit_reference": {
                    "label": gt_align_fit_label,
                    "path": str(fit_path),
                    "lowpass_objective": float(fitted.score),
                },
            }
        elif shared_document is not None:
            shared_transform = _common_rigid_transform(shared_document)
        json_summary["gt_align_options_applied"] = gt_align_transform is None
        json_summary["gt_align_rigid_mode"] = (
            "shared_fit"
            if gt_align_fit_label is not None
            else "shared_transform"
            if shared_transform is not None
            else "independent"
        )
        if shared_document is not None:
            json_summary["rigid_transform"] = shared_document
            npz_payload["gt_align_rigid_transform_json"] = np.asarray(
                json.dumps(shared_document, sort_keys=True, allow_nan=False)
            )

    for label, path in zip(labels, volume_paths):
        real, vol_voxel = _load_volume(path, volume_frame, strict_voxel_grid=gt_align_rigid)
        if real.shape != gt_real.shape:
            raise ValueError(f"{path} shape {real.shape} does not match GT shape {gt_real.shape}")
        voxel_size = voxel_size_override or gt_voxel or vol_voxel or 1.0
        if gt_align_rigid and vol_voxel != gt_voxel:
            raise ValueError(f"{path}: rigid comparison requires the same voxel grid as GT")
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
            if shared_transform is not None:
                aligned_volume = shared_transform.apply(real, voxel_size=float(voxel_size), gt_sha256=gt_identity)
                rotation_index = int(shared_document["fit_receipt"]["coarse_rotation_index"])
                rotation_matrix = np.asarray(shared_transform.rotation_matrix)
                mirror_x, sign, score = shared_transform.mirror_x, 1, None
                translation = np.asarray(shared_transform.translation_voxels)
            elif gt_align_rigid:
                from recovar.em.vdam.gt_registration import RigidFitControls, align_volume_rigid_to_reference

                alignment = align_volume_rigid_to_reference(
                    real,
                    gt_real,
                    rotations,
                    controls=RigidFitControls(
                        score_max_shell=int(gt_align_max_shell),
                        allow_mirror=bool(gt_align_allow_mirror),
                    ),
                )
            else:
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
            if shared_transform is None:
                aligned_volume = alignment.aligned_volume
                rotation_index, rotation_matrix = alignment.rotation_index, alignment.rotation_matrix
                mirror_x, sign, score = alignment.mirror_x, alignment.sign, float(alignment.score)
                if gt_align_rigid:
                    translation = alignment.translation_voxels
            aligned_prefix = f"{label}_aligned"
            _add_metric_set(
                prefix=aligned_prefix,
                volume=aligned_volume,
                reference=gt_real,
                reference_ft=gt_ft,
                volume_shape=volume_shape,
                voxel_size=float(voxel_size),
                npz_payload=npz_payload,
                json_payload=per_volume.setdefault("aligned", {}),
            )
            npz_payload[f"{label}_gt_align_rotation_index"] = np.int32(rotation_index)
            npz_payload[f"{label}_gt_align_rotation_matrix"] = rotation_matrix
            npz_payload[f"{label}_gt_align_mirror_x"] = np.bool_(mirror_x)
            npz_payload[f"{label}_gt_align_sign"] = np.int32(sign)
            per_volume["aligned"].update(
                {
                    "rotation_index": int(rotation_index),
                    "rotation_matrix": np.asarray(rotation_matrix).tolist(),
                    "mirror_x": bool(mirror_x),
                    "sign": int(sign),
                    "score_vs_gt": score,
                }
            )

            if gt_align_rigid:
                receipt = (
                    dict(shared_document["fit_receipt"]) if shared_transform is not None else asdict(alignment.receipt)
                )
                receipt.update(
                    translation_voxels=translation.tolist(),
                    translation_A=(translation * float(voxel_size)).tolist(),
                    coordinate_frame="recovar_array_axes_0_1_2",
                    forward_transform="y=c+R*M*(x-c)+t; c=(shape-1)/2; M reflects axis0",
                    sign=1,
                    independently_fitted_per_volume=shared_transform is None,
                    discrete_refinement_used=False,
                    quality_accepted=False,
                )
                if shared_transform is not None:
                    receipt["transform_identity_sha256"] = shared_transform.identity_sha256
                    receipt["fit_reference"] = shared_document["fit_reference"]
                per_volume["aligned"]["rigid_registration"] = receipt
                npz_payload[f"{label}_gt_align_translation_voxels"] = translation
                npz_payload[f"{label}_gt_align_rigid_receipt_json"] = np.asarray(
                    json.dumps(receipt, sort_keys=True, allow_nan=False)
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
        gt_align_rigid=bool(args.gt_align_rigid),
        gt_align_fit_label=args.gt_align_fit_label,
        gt_align_transform=json.loads(Path(args.gt_align_transform_json).read_text())
        if args.gt_align_transform_json
        else None,
    )

    if args.gt_align_transform_output:
        transform_path = Path(args.gt_align_transform_output)
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        transform_path.write_text(
            json.dumps(json_summary["rigid_transform"], sort_keys=True, indent=2, allow_nan=False) + "\n"
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
