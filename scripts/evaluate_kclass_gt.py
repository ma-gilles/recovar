#!/usr/bin/env python
"""K-class GT FSC evaluation with best-permutation matching.

Wraps :mod:`recovar.em.initial_model.gt_metrics` to handle the K-class
case end-to-end:

* takes ``--volume`` MRCs (one per class) and ``--gt_volume`` MRCs (one
  per GT class);
* runs the same coarse + local-refinement rotation alignment as
  ``evaluate_ab_initio_gt.py`` (HEALPix-2 coarse, refined at orders 3/4
  by default, locks mirror+sign at coarse stage);
* solves the best one-to-one assignment of (recovar class -> GT class) using
  normalized FSC-AUC, and records the old mean fsc(shells 1..8) assignment
  as a diagnostic;
* prints per-class FSC curves so the user can see whether a single
  class is dragging the mean (use ``--print_per_shell_fsc``).

Mirrors the per-class needs the upstream ab-initio benchmarking has —
"per-class FSCs not just mean" — without running the legacy
``evaluate_ab_initio_gt.py`` once per (volume, gt) pair.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from recovar.em.initial_model.gt_metrics import (
    DEFAULT_GT_ALIGN_HEALPIX_ORDER,
    DEFAULT_GT_ALIGN_MAX_SHELL,
    align_volume_to_reference,
    first_shell_below_threshold,
    relion_alignment_rotations,
)
from recovar.utils import helpers


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--volume",
        action="append",
        required=True,
        help="K recovar MRCs (one per class). Pass --volume once per class.",
    )
    parser.add_argument(
        "--gt_volume",
        action="append",
        required=True,
        help="K GT MRCs (one per class). Pass --gt_volume once per class.",
    )
    parser.add_argument(
        "--label",
        default="primary",
        help=(
            "Label for the --volume set (printed in the report). When --compare_with is "
            "supplied, the report adds a side-by-side delta table between this label and "
            "each comparison label."
        ),
    )
    parser.add_argument(
        "--compare_with",
        action="append",
        default=None,
        metavar="LABEL=PATH1,PATH2,...",
        help=(
            "Add another K-class volume set to evaluate against the same --gt_volume(s) and "
            "produce a side-by-side per-class FSC delta table. Repeatable. Each value is "
            "'<LABEL>=<comma-separated MRC paths>' with exactly K paths. Example: "
            "--compare_with 'RELION=run_it100_class001.mrc,run_it100_class002.mrc'."
        ),
    )
    parser.add_argument("--volume_frame", choices=("relion", "recovar"), default="relion")
    parser.add_argument("--gt_frame", choices=("relion", "recovar"), default="recovar")
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument(
        "--gt_align_healpix_order",
        type=int,
        default=DEFAULT_GT_ALIGN_HEALPIX_ORDER,
        help="Coarse HEALPix order for the rotation grid passed to align_volume_to_reference.",
    )
    parser.add_argument("--gt_align_max_shell", type=int, default=DEFAULT_GT_ALIGN_MAX_SHELL)
    parser.add_argument("--gt_align_no_mirror", action="store_true")
    parser.add_argument(
        "--gt_align_allow_sign",
        action="store_true",
        help="Allow a global sign flip during alignment (set this when comparing native InitialModel outputs to recovar-frame GT — they have opposite contrast convention).",
    )
    parser.add_argument(
        "--gt_align_refine_orders",
        type=int,
        nargs="*",
        default=[3, 4],
        help="HEALPix orders for local refinement; empty list disables refinement.",
    )
    parser.add_argument("--gt_align_refine_sigma_deg", type=float, default=30.0)
    parser.add_argument("--print_per_shell_fsc", action="store_true")
    parser.add_argument("--output_json", default=None, help="Optional JSON path for the report.")
    return parser.parse_args(argv)


def _voxel_size_value(raw: Any) -> float | None:
    """Robust voxel-size unpacking — handles numpy recarrays, scalars, tuples."""
    import math

    if raw is None:
        return None
    candidates: list[Any] = []
    if hasattr(raw, "x"):
        candidates.append(raw.x)  # mrcfile recarray: voxel_size.x
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
        raise ValueError(f"unknown frame {frame!r}")
    return np.asarray(vol, dtype=np.float64), _voxel_size_value(voxel)


def _fsc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.fft.fftshift(np.fft.fftn(a))
    B = np.fft.fftshift(np.fft.fftn(b))
    n = a.shape[0]
    c = n // 2
    z, y, x = np.indices(a.shape)
    r = np.round(np.sqrt((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2)).astype(np.int32)
    # Match regularization.get_fsc_gpu: shell indices [0, N // 2 - 2].
    # The Nyquist-edge shells are not complete spherical shells on this grid.
    out = np.zeros(c - 1)
    for s in range(c - 1):
        m = r == s
        if not m.any():
            continue
        num = np.real(np.sum(A[m] * np.conj(B[m])))
        den = np.sqrt(np.sum(np.abs(A[m]) ** 2) * np.sum(np.abs(B[m]) ** 2))
        out[s] = float(num / den) if den > 0 else 0.0
    return out


def _mean_fsc(fsc: np.ndarray, lo: int, hi: int) -> float:
    lo = max(0, lo)
    hi = min(len(fsc), hi)
    if hi <= lo:
        return float("nan")
    return float(np.mean(fsc[lo:hi]))


def _normalized_fsc_auc(fsc: np.ndarray) -> float:
    values = np.asarray(fsc, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return float("nan")
    finite = np.isfinite(values)
    finite[0] = False
    if not finite.any():
        return float("nan")
    x = np.arange(values.size, dtype=np.float64)[finite]
    y = values[finite]
    if y.size == 1:
        return float(y[0])
    span = float(x[-1] - x[0])
    if span <= 0.0 or not np.isfinite(span):
        return float(np.mean(y))
    x = (x - x[0]) / span
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x))


def _best_permutation_from_score_matrix(score_matrix: np.ndarray) -> tuple[tuple[int, ...], float]:
    """Return the max-score one-to-one class assignment.

    This is equivalent to maximizing over all permutations, but uses the
    Hungarian algorithm instead of enumerating K! assignments. K=16 already
    makes factorial enumeration impractical in validation runs.
    """

    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"score_matrix must be square, got shape {scores.shape}")
    if scores.shape[0] == 0:
        raise ValueError("score_matrix must be non-empty")
    finite = np.isfinite(scores)
    if not finite.any(axis=1).all() or not finite.any(axis=0).all():
        raise ValueError("score_matrix must have at least one finite score in every row and column")

    finite_scores = scores[finite]
    penalty = float(np.min(finite_scores) - max(1.0, np.ptp(finite_scores), np.max(np.abs(finite_scores))))
    safe_scores = np.where(finite, scores, penalty)
    rows, cols = linear_sum_assignment(-safe_scores)
    if set(int(row) for row in rows) != set(range(scores.shape[0])):
        raise RuntimeError("linear_sum_assignment did not assign every class")
    perm = [0] * scores.shape[0]
    for row, col in zip(rows, cols):
        perm[int(row)] = int(col)
    chosen = scores[np.arange(scores.shape[0]), np.asarray(perm, dtype=np.int64)]
    return tuple(perm), float(np.mean(chosen))


def _mean_score_for_perm(score_matrix: np.ndarray, perm: tuple[int, ...]) -> float:
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"score_matrix must be square, got shape {scores.shape}")
    if len(perm) != scores.shape[0]:
        raise ValueError(f"perm length {len(perm)} does not match score matrix shape {scores.shape}")
    chosen = scores[np.arange(scores.shape[0]), np.asarray(perm, dtype=np.int64)]
    return float(np.mean(chosen))


def _pairwise_score_matrices(fsc_table: list[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    K = len(fsc_table)
    if K == 0 or any(len(row) != K for row in fsc_table):
        raise ValueError("fsc_table must be a non-empty square table")
    mean_fsc_1_8 = np.array([[_mean_fsc(fsc_table[i][j], 1, 8) for j in range(K)] for i in range(K)])
    fsc_auc = np.array([[_normalized_fsc_auc(fsc_table[i][j]) for j in range(K)] for i in range(K)])
    return mean_fsc_1_8, fsc_auc


def _assignment_summary_from_fsc_table(fsc_table: list[list[np.ndarray]]) -> dict[str, Any]:
    mean_fsc_1_8_matrix, fsc_auc_matrix = _pairwise_score_matrices(fsc_table)
    best_perm, best_mean_fsc_auc = _best_permutation_from_score_matrix(fsc_auc_matrix)
    best_perm_by_mean_fsc_1_8, best_mean_fsc_1_8_by_mean_fsc_1_8 = _best_permutation_from_score_matrix(
        mean_fsc_1_8_matrix
    )
    return {
        "best_perm": best_perm,
        "best_perm_score_key": "fsc_auc",
        "best_mean_fsc_auc": float(best_mean_fsc_auc),
        "best_mean_fsc_1_8": _mean_score_for_perm(mean_fsc_1_8_matrix, best_perm),
        "best_perm_by_mean_fsc_1_8": best_perm_by_mean_fsc_1_8,
        "best_mean_fsc_1_8_by_mean_fsc_1_8": float(best_mean_fsc_1_8_by_mean_fsc_1_8),
        "pairwise_mean_fsc_1_8": mean_fsc_1_8_matrix,
        "pairwise_fsc_auc": fsc_auc_matrix,
    }


def _shell_resolution(shell_index: int, volume_size: int, voxel_size: float) -> float:
    if int(shell_index) <= 0:
        return float("nan")
    return float(volume_size) * float(voxel_size) / float(shell_index)


def _evaluate_one_set(
    *,
    label: str,
    rec_paths: list[str],
    rec_vols: list[np.ndarray],
    gt_paths: list[str],
    gt_vols: list[np.ndarray],
    voxel_size: float,
    rotations: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the K^2 alignment + best-permutation pick for one volume set."""
    K = len(rec_paths)
    refine_orders = tuple(int(o) for o in (args.gt_align_refine_orders or [])) or None
    print(
        f"\n[{label}] K-class GT eval: K={K}, voxel={voxel_size:.4g} A, refine_orders={refine_orders}",
        flush=True,
    )
    print(
        f"[{label}]   pairwise aligning {K * K} pairs "
        f"(coarse HEALPix-{args.gt_align_healpix_order} + refine {refine_orders})...",
        flush=True,
    )
    alignments: list[list[Any]] = [[None] * K for _ in range(K)]
    fsc_table: list[list[np.ndarray]] = [[np.array([])] * K for _ in range(K)]
    t0 = time.time()
    for i in range(K):
        for j in range(K):
            a = align_volume_to_reference(
                rec_vols[i],
                gt_vols[j],
                rotations,
                score_max_shell=int(args.gt_align_max_shell),
                allow_mirror=not bool(args.gt_align_no_mirror),
                allow_sign=bool(args.gt_align_allow_sign),
                refine_orders=refine_orders,
                refine_sigma_deg=float(args.gt_align_refine_sigma_deg),
            )
            alignments[i][j] = a
            fsc_table[i][j] = _fsc(a.aligned_volume, gt_vols[j])
            print(
                f"[{label}]     align rec[{i}] -> gt[{j}]: "
                f"corr={a.corr:.4f} mean_fsc(1-8)={_mean_fsc(fsc_table[i][j], 1, 8):.4f} "
                f"fsc_auc={_normalized_fsc_auc(fsc_table[i][j]):.4f}",
                flush=True,
            )
    print(f"[{label}]   alignment done in {time.time() - t0:.1f}s", flush=True)

    assignment = _assignment_summary_from_fsc_table(fsc_table)
    best_perm = assignment["best_perm"]
    pairwise_corr = np.array([[float(alignments[i][j].corr) for j in range(K)] for i in range(K)])

    shape = rec_vols[0].shape
    per_class = []
    for i in range(K):
        j = best_perm[i]
        a = alignments[i][j]
        fsc = fsc_table[i][j]
        fsc_auc = float(_normalized_fsc_auc(fsc))
        sh05 = int(first_shell_below_threshold(fsc, 0.5))
        sh143 = int(first_shell_below_threshold(fsc, 0.143))
        per_class.append(
            {
                "class": int(i),
                "matched_gt_class": int(j),
                "rec_path": str(rec_paths[i]),
                "gt_path": str(gt_paths[j]),
                "corr": float(a.corr),
                "fsc_auc": fsc_auc,
                "mean_fsc_1_8": float(_mean_fsc(fsc, 1, 8)),
                "mean_fsc_1_16": float(_mean_fsc(fsc, 1, 16)),
                "shell_05": sh05,
                "shell_0143": sh143,
                "resolution_05_A": _shell_resolution(sh05, shape[0], voxel_size),
                "resolution_0143_A": _shell_resolution(sh143, shape[0], voxel_size),
                "fsc_vs_gt": [float(v) for v in fsc],
                "rotation_matrix": np.asarray(a.rotation_matrix).tolist(),
                "mirror_x": bool(a.mirror_x),
                "sign": int(a.sign),
            }
        )
    mean_fsc_auc = float(assignment["best_mean_fsc_auc"])

    return {
        "label": label,
        "K": K,
        "voxel_size": float(voxel_size),
        "best_perm": list(best_perm),
        "best_perm_score_key": str(assignment["best_perm_score_key"]),
        "best_mean_fsc_auc": float(assignment["best_mean_fsc_auc"]),
        "best_mean_fsc_1_8": float(assignment["best_mean_fsc_1_8"]),
        "best_perm_by_mean_fsc_1_8": list(assignment["best_perm_by_mean_fsc_1_8"]),
        "best_mean_fsc_1_8_by_mean_fsc_1_8": float(assignment["best_mean_fsc_1_8_by_mean_fsc_1_8"]),
        "pairwise_fsc_auc": np.asarray(assignment["pairwise_fsc_auc"], dtype=np.float64).tolist(),
        "pairwise_mean_fsc_1_8": np.asarray(assignment["pairwise_mean_fsc_1_8"], dtype=np.float64).tolist(),
        "pairwise_corr": pairwise_corr.tolist(),
        "mean_fsc_auc": mean_fsc_auc,
        "mean_fsc_auc_1_nyquist": mean_fsc_auc,
        "per_class": per_class,
    }


def _print_set_table(result: dict[str, Any]) -> None:
    print(
        f"\n[{result['label']}] best permutation by {result.get('best_perm_score_key', 'fsc_auc')} "
        f"(rec -> gt): {tuple(result['best_perm'])}, "
        f"mean fsc_auc = {result['mean_fsc_auc']:.6f}, "
        f"mean fsc(1-8) = {result['best_mean_fsc_1_8']:.6f}"
    )
    legacy_perm = tuple(result.get("best_perm_by_mean_fsc_1_8", result["best_perm"]))
    if legacy_perm != tuple(result["best_perm"]):
        print(
            f"[{result['label']}] mean-fsc(1-8) diagnostic permutation: {legacy_perm}, "
            f"mean fsc(1-8) = {result['best_mean_fsc_1_8_by_mean_fsc_1_8']:.6f}"
        )
    header = (
        f"{'class':<6s} {'rec_path':<60s} {'gt_path':<60s} "
        f"{'corr':>8s} {'auc':>8s} {'fsc1-8':>8s} {'fsc1-16':>8s} {'sh@0.5':>7s} {'sh@.143':>8s}"
    )
    print(header)
    print("-" * len(header))
    for entry in result["per_class"]:
        print(
            f"{entry['class']:<6d} {Path(entry['rec_path']).name[:60]:<60s} "
            f"{Path(entry['gt_path']).name[:60]:<60s} "
            f"{entry['corr']:>8.4f} {entry['fsc_auc']:>8.4f} "
            f"{entry['mean_fsc_1_8']:>8.4f} {entry['mean_fsc_1_16']:>8.4f} "
            f"{entry['shell_05']:>7d} {entry['shell_0143']:>8d}"
        )


def _print_side_by_side_delta(primary: dict[str, Any], comparison: dict[str, Any]) -> None:
    """Print a per-class delta table primary - comparison.

    Best permutations are evaluated INDEPENDENTLY for each set. Rows are joined
    by matched GT class, so an unrelated raw class-order difference does not
    create a fake per-class delta.
    """
    if primary["K"] != comparison["K"]:
        print(f"\n[delta] skipping side-by-side: K mismatch primary={primary['K']} vs comparison={comparison['K']}")
        return
    print(f"\n[delta] {primary['label']} vs {comparison['label']} (per matched GT class, best-permutation per set):")
    header = (
        f"{'class':<6s} {'gt':<4s} {comparison['label'] + '_class':>15s} "
        f"{primary['label'] + '_corr':>15s} {comparison['label'] + '_corr':>15s} {'Δcorr':>8s}  "
        f"{primary['label'] + '_auc':>15s} {comparison['label'] + '_auc':>15s} {'Δauc':>8s}  "
        f"{primary['label'] + '_fsc1-8':>17s} {comparison['label'] + '_fsc1-8':>17s} {'Δfsc1-8':>9s}  "
        f"{primary['label'] + '_sh@.143':>17s} {comparison['label'] + '_sh@.143':>17s} {'Δshell':>8s}"
    )
    print(header)
    print("-" * len(header))
    comparison_by_gt = {int(entry["matched_gt_class"]): entry for entry in comparison["per_class"]}
    for p in primary["per_class"]:
        gt_class = int(p["matched_gt_class"])
        c = comparison_by_gt.get(gt_class)
        if c is None:
            print(f"{p['class']:<6d} {gt_class:<4d} {'missing':>15s}")
            continue
        d_corr = p["corr"] - c["corr"]
        d_auc = p["fsc_auc"] - c["fsc_auc"]
        d_fsc18 = p["mean_fsc_1_8"] - c["mean_fsc_1_8"]
        d_sh143 = p["shell_0143"] - c["shell_0143"]
        print(
            f"{p['class']:<6d} {gt_class:<4d} {c['class']:>15d} "
            f"{p['corr']:>15.4f} {c['corr']:>15.4f} {d_corr:>+8.4f}  "
            f"{p['fsc_auc']:>15.4f} {c['fsc_auc']:>15.4f} {d_auc:>+8.4f}  "
            f"{p['mean_fsc_1_8']:>17.4f} {c['mean_fsc_1_8']:>17.4f} {d_fsc18:>+9.4f}  "
            f"{p['shell_0143']:>17d} {c['shell_0143']:>17d} {d_sh143:>+8d}"
        )
    # Aggregate row
    p_mean_fsc = primary["best_mean_fsc_1_8"]
    c_mean_fsc = comparison["best_mean_fsc_1_8"]
    p_mean_auc = primary["mean_fsc_auc"]
    c_mean_auc = comparison["mean_fsc_auc"]
    print(
        f"{'mean':<6s} "
        f"{'':<4s} {'':>15s} "
        f"{'':>15s} {'':>15s} {'':>8s}  "
        f"{p_mean_auc:>15.4f} {c_mean_auc:>15.4f} {p_mean_auc - c_mean_auc:>+8.4f}  "
        f"{p_mean_fsc:>17.4f} {c_mean_fsc:>17.4f} {p_mean_fsc - c_mean_fsc:>+9.4f}  "
        f"{'':>17s} {'':>17s} {'':>8s}"
    )


def _parse_compare_with(spec: str, K: int) -> tuple[str, list[str]]:
    """Parse 'LABEL=path1,path2,...' into (label, [paths])."""
    if "=" not in spec:
        raise SystemExit(f"--compare_with must be 'LABEL=path1,path2,...': got {spec!r}")
    label, paths_str = spec.split("=", 1)
    label = label.strip()
    paths = [p.strip() for p in paths_str.split(",") if p.strip()]
    if not label:
        raise SystemExit(f"--compare_with missing LABEL: {spec!r}")
    if len(paths) != K:
        raise SystemExit(f"--compare_with {label!r}: expected {K} paths to match --volume count, got {len(paths)}")
    return label, paths


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rec_paths = list(args.volume)
    gt_paths = list(args.gt_volume)
    if len(rec_paths) != len(gt_paths):
        raise SystemExit(f"--volume count {len(rec_paths)} must equal --gt_volume count {len(gt_paths)}")
    K = len(rec_paths)

    # Load primary volume set + GT once; share across comparisons.
    rec_pairs = [_load_volume(p, args.volume_frame) for p in rec_paths]
    gt_pairs = [_load_volume(p, args.gt_frame) for p in gt_paths]
    primary_vols = [pair[0] for pair in rec_pairs]
    gt_vols = [pair[0] for pair in gt_pairs]
    voxel_size = (
        args.voxel_size
        or next((v for _, v in gt_pairs if v is not None), None)
        or next((v for _, v in rec_pairs if v is not None), None)
        or 1.0
    )

    shape = primary_vols[0].shape
    for v in primary_vols + gt_vols:
        if v.shape != shape:
            raise SystemExit(f"all volumes must share shape; got {[u.shape for u in primary_vols + gt_vols]}")

    rotations = relion_alignment_rotations(int(args.gt_align_healpix_order))

    primary_result = _evaluate_one_set(
        label=str(args.label),
        rec_paths=rec_paths,
        rec_vols=primary_vols,
        gt_paths=gt_paths,
        gt_vols=gt_vols,
        voxel_size=voxel_size,
        rotations=rotations,
        args=args,
    )
    _print_set_table(primary_result)

    comparison_results: list[dict[str, Any]] = []
    for spec in args.compare_with or []:
        cmp_label, cmp_paths = _parse_compare_with(spec, K)
        cmp_pairs = [_load_volume(p, args.volume_frame) for p in cmp_paths]
        cmp_vols = [pair[0] for pair in cmp_pairs]
        for v in cmp_vols:
            if v.shape != shape:
                raise SystemExit(f"--compare_with {cmp_label!r} volume shape {v.shape} must match primary {shape}")
        cmp_result = _evaluate_one_set(
            label=cmp_label,
            rec_paths=cmp_paths,
            rec_vols=cmp_vols,
            gt_paths=gt_paths,
            gt_vols=gt_vols,
            voxel_size=voxel_size,
            rotations=rotations,
            args=args,
        )
        _print_set_table(cmp_result)
        _print_side_by_side_delta(primary_result, cmp_result)
        comparison_results.append(cmp_result)

    if args.print_per_shell_fsc:
        print()
        print(f"[{primary_result['label']}] per-shell FSC vs GT (best permutation):")
        for entry in primary_result["per_class"]:
            fsc = entry["fsc_vs_gt"]
            print(f"  class {entry['class']} (matched gt {entry['matched_gt_class']}):")
            for chunk_start in range(0, len(fsc), 16):
                chunk_end = min(chunk_start + 16, len(fsc))
                print("    shell: " + " ".join(f"{s:>5d}" for s in range(chunk_start, chunk_end)))
                print("    fsc:   " + " ".join(f"{fsc[s]:>5.2f}" for s in range(chunk_start, chunk_end)))

    summary: dict[str, Any] = {
        "primary": primary_result,
        "comparisons": comparison_results,
        "voxel_size": float(voxel_size),
        "K": K,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
