#!/usr/bin/env python
"""Quality + perf comparison: recovar auto-refine vs RELION auto-refine.

Outputs a single ``comparison.json`` + ``comparison_table.txt`` summarising:

  Quality (FSC vs ground-truth volume):
    - FSC area mean
    - FSC@0.5 shell + Angstrom resolution
    - FSC@0.143 shell + Angstrom resolution
    - half-map gold-standard FSC@0.143 (resolution self-estimate)

  Perf (wall time, GPU memory):
    - total wall clock seconds
    - per-iter average seconds
    - n iterations to convergence

For k=1: compare against a single RELION 3D auto-refine run (one final
half1/half2 pair).
For k=N: compare against a RELION Class3D multi-iter run (K final classes),
Hungarian-matched against GT classes.

Usage::

    # k=1
    python compare_recovar_relion_autorefine.py \\
        --mode k1 \\
        --recovar-dir /scratch/.../autorefine_k1_2026_05_02/run_NNNNN \\
        --relion-dir /scratch/.../data_100k_512/relion_ref_full_autorefine \\
        --gt-mrc /scratch/.../data_100k_512/reference_gt.mrc \\
        --output comparison.json

    # k=N
    python compare_recovar_relion_autorefine.py \\
        --mode kN \\
        --recovar-dir /scratch/.../autorefine_kN_2026_05_02/run_NNNNN_K4 \\
        --relion-dir /scratch/.../ribosembly_allk_g256_n100000_snr1_cubic/relion_class3d_k4_autorefine \\
        --gt-dir /scratch/.../ribosembly_allk_g256_n100000_snr1_cubic \\
        --output comparison.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import mrcfile
import numpy as np

# ---------------------------------------------------------------------------
# Volume helpers
# ---------------------------------------------------------------------------


def load_mrc(path: Path) -> tuple[np.ndarray, float]:
    with mrcfile.open(str(path), permissive=True) as mrc:
        return mrc.data.copy().astype(np.float32), float(mrc.voxel_size.x)


def crop_or_downsample(volume: np.ndarray, target: int) -> np.ndarray:
    """Match a reference volume to a target grid size by mean-pooling
    (downsample) or zero-padding (upsample) the leading-axis cube.
    Used so volumes prepared at different grids can be FSC'd."""
    src = volume.shape[0]
    if src == target:
        return volume
    if src > target:
        ratio = src // target
        if src != target * ratio:
            raise ValueError(f"non-integer downsample ratio: {src} → {target}")
        return volume.reshape(target, ratio, target, ratio, target, ratio).mean(axis=(1, 3, 5))
    # Upsample: zero-pad (rare, best avoided).
    pad = (target - src) // 2
    return np.pad(volume, ((pad, pad), (pad, pad), (pad, pad)))


def fsc(pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Return (fsc_curve, fsc_area_mean, shell@0.5, shell@0.143)."""
    from recovar.reconstruction.regularization import get_fsc

    vol_shape = pred.shape
    curve = np.asarray(get_fsc(pred.reshape(-1), ref.reshape(-1), vol_shape))
    area = float(np.mean(curve))
    s05 = int(np.argmax(curve < 0.5)) if (curve < 0.5).any() else len(curve)
    s0143 = int(np.argmax(curve < 0.143)) if (curve < 0.143).any() else len(curve)
    return curve, area, s05, s0143


def shell_to_angstrom(shell: int, grid_size: int, voxel_size: float) -> float:
    return float(grid_size) * float(voxel_size) / max(int(shell), 1)


# ---------------------------------------------------------------------------
# Wall-time extraction from RELION
# ---------------------------------------------------------------------------


def relion_walltime_from_iters(relion_dir: Path) -> tuple[float, list[float]]:
    """Return (total_seconds, per_iter_seconds) by reading mtimes of
    ``run_itNNN_optimiser.star``. RELION writes one optimiser per iter,
    so consecutive mtime diffs give per-iter wall."""
    optimisers = sorted(relion_dir.glob("run_it*_optimiser.star"))
    if len(optimisers) < 2:
        return float("nan"), []
    mtimes = [p.stat().st_mtime for p in optimisers]
    per_iter = [mtimes[i + 1] - mtimes[i] for i in range(len(mtimes) - 1)]
    total = mtimes[-1] - mtimes[0]
    return total, per_iter


def relion_final_iter(relion_dir: Path) -> int:
    """Return the highest itNNN index for which both half1+half2 mrcs exist."""
    halfs = list(relion_dir.glob("run_it*_half1_class*.mrc"))
    if not halfs:
        return -1
    iters = []
    for p in halfs:
        m = re.search(r"run_it(\d+)_", p.name)
        if m:
            iters.append(int(m.group(1)))
    return max(iters) if iters else -1


# ---------------------------------------------------------------------------
# Hungarian matching for k>1
# ---------------------------------------------------------------------------


def hungarian_match(pred_vols: np.ndarray, gt_vols: np.ndarray) -> dict:
    from scipy.optimize import linear_sum_assignment

    K = min(pred_vols.shape[0], gt_vols.shape[0])
    fsc_areas = np.zeros((pred_vols.shape[0], gt_vols.shape[0]), dtype=np.float32)
    fsc_05_shell = np.zeros_like(fsc_areas, dtype=np.int32)
    fsc_0143_shell = np.zeros_like(fsc_areas, dtype=np.int32)
    for i in range(pred_vols.shape[0]):
        for j in range(gt_vols.shape[0]):
            curve, area, s05, s0143 = fsc(pred_vols[i], gt_vols[j])
            fsc_areas[i, j] = area
            fsc_05_shell[i, j] = s05
            fsc_0143_shell[i, j] = s0143
    row, col = linear_sum_assignment(-fsc_areas[:K, :K])
    matched = list(zip(map(int, row), map(int, col)))
    return {
        "assignment": matched,
        "fsc_area_per_class": [float(fsc_areas[i, j]) for i, j in matched],
        "fsc_05_shell_per_class": [int(fsc_05_shell[i, j]) for i, j in matched],
        "fsc_0143_shell_per_class": [int(fsc_0143_shell[i, j]) for i, j in matched],
        "fsc_area_mean": float(np.mean([fsc_areas[i, j] for i, j in matched])),
    }


# ---------------------------------------------------------------------------
# k=1 mode
# ---------------------------------------------------------------------------


def compare_k1(args) -> dict:
    recovar_dir = Path(args.recovar_dir)
    relion_dir = Path(args.relion_dir)
    gt_path = Path(args.gt_mrc)

    gt, gt_voxel = load_mrc(gt_path)

    # ---- recovar final volume.
    rec_mean = sorted(recovar_dir.glob("iter_*_mean.mrc"))
    if not rec_mean:
        rec_mean = sorted(recovar_dir.glob("**/iter_*_mean.mrc"))
    if not rec_mean:
        raise SystemExit(f"no iter_*_mean.mrc in {recovar_dir}")
    rec_vol, rec_voxel = load_mrc(rec_mean[-1])
    gt_for_rec = crop_or_downsample(gt, rec_vol.shape[0])
    rec_curve, rec_area, rec_s05, rec_s0143 = fsc(rec_vol, gt_for_rec)
    grid = rec_vol.shape[0]

    # ---- RELION final half maps + averaged map.
    rel_iter = relion_final_iter(relion_dir)
    rel_h1, _ = load_mrc(relion_dir / f"run_it{rel_iter:03d}_half1_class001.mrc")
    rel_h2, rel_voxel = load_mrc(relion_dir / f"run_it{rel_iter:03d}_half2_class001.mrc")
    rel_avg = 0.5 * (rel_h1 + rel_h2)
    gt_for_rel = crop_or_downsample(gt, rel_avg.shape[0])
    rel_curve, rel_area, rel_s05, rel_s0143 = fsc(rel_avg, gt_for_rel)
    rel_h1_h2_curve, rel_h1_h2_area, rel_gs_05, rel_gs_0143 = fsc(rel_h1, rel_h2)

    # ---- Recovar half maps (if dumped).
    rec_h1_files = sorted(recovar_dir.glob("iter_*_half1.mrc"))
    rec_gs_block: dict = {}
    if rec_h1_files:
        rec_h1, _ = load_mrc(rec_h1_files[-1])
        rec_h2_path = recovar_dir / rec_h1_files[-1].name.replace("half1", "half2")
        if rec_h2_path.exists():
            rec_h2, _ = load_mrc(rec_h2_path)
            _, _, rec_gs_05, rec_gs_0143 = fsc(rec_h1, rec_h2)
            rec_gs_block = {
                "fsc_at_05_shell": rec_gs_05,
                "fsc_at_0143_shell": rec_gs_0143,
                "fsc_at_05_angstrom": shell_to_angstrom(rec_gs_05, grid, rec_voxel),
                "fsc_at_0143_angstrom": shell_to_angstrom(rec_gs_0143, grid, rec_voxel),
            }

    # ---- Wall time (recovar from results.npz, RELION from mtimes).
    rec_results = list(recovar_dir.glob("results.npz"))
    if not rec_results:
        rec_results = list(recovar_dir.glob("**/results.npz"))
    rec_total = float("nan")
    rec_per_iter: list[float] = []
    rec_n_iter = -1
    if rec_results:
        rec_npz = np.load(rec_results[0], allow_pickle=True)
        rec_total = float(rec_npz.get("total_time", float("nan")))
        rec_n_iter = int(rec_npz.get("n_iterations", -1))
        rec_walls = rec_npz.get("wall_times", None)
        if rec_walls is not None:
            rec_per_iter = [float(w) for w in np.asarray(rec_walls)]
    rel_total, rel_per_iter = relion_walltime_from_iters(relion_dir)

    summary = {
        "mode": "k1",
        "ground_truth": str(gt_path),
        "recovar": {
            "dir": str(recovar_dir),
            "voxel_size": rec_voxel,
            "grid_size": grid,
            "n_iter": rec_n_iter,
            "wall_total_s": rec_total,
            "wall_per_iter_s": rec_per_iter,
            "fsc_vs_gt": {
                "area_mean": rec_area,
                "fsc_at_05_shell": rec_s05,
                "fsc_at_0143_shell": rec_s0143,
                "fsc_at_05_angstrom": shell_to_angstrom(rec_s05, grid, rec_voxel),
                "fsc_at_0143_angstrom": shell_to_angstrom(rec_s0143, grid, rec_voxel),
            },
            "fsc_halfmap": rec_gs_block,
        },
        "relion": {
            "dir": str(relion_dir),
            "voxel_size": rel_voxel,
            "grid_size": rel_avg.shape[0],
            "final_iter": rel_iter,
            "wall_total_s": rel_total,
            "wall_per_iter_s": rel_per_iter,
            "fsc_vs_gt": {
                "area_mean": rel_area,
                "fsc_at_05_shell": rel_s05,
                "fsc_at_0143_shell": rel_s0143,
                "fsc_at_05_angstrom": shell_to_angstrom(rel_s05, rel_avg.shape[0], rel_voxel),
                "fsc_at_0143_angstrom": shell_to_angstrom(rel_s0143, rel_avg.shape[0], rel_voxel),
            },
            "fsc_halfmap": {
                "fsc_at_05_shell": rel_gs_05,
                "fsc_at_0143_shell": rel_gs_0143,
                "fsc_at_05_angstrom": shell_to_angstrom(rel_gs_05, rel_avg.shape[0], rel_voxel),
                "fsc_at_0143_angstrom": shell_to_angstrom(rel_gs_0143, rel_avg.shape[0], rel_voxel),
            },
        },
    }
    return summary


# ---------------------------------------------------------------------------
# k=N mode
# ---------------------------------------------------------------------------


def compare_kN(args) -> dict:
    recovar_dir = Path(args.recovar_dir)
    relion_dir = Path(args.relion_dir)
    gt_dir = Path(args.gt_dir)

    # Ground truth: load first K reference_gt_class*.mrc
    gt_paths = sorted(gt_dir.glob("reference_gt_class*.mrc"))
    if not gt_paths:
        raise SystemExit(f"no reference_gt_class*.mrc in {gt_dir}")
    gt_vols = []
    gt_voxel = None
    for p in gt_paths:
        v, vx = load_mrc(p)
        gt_vols.append(v)
        gt_voxel = vx if gt_voxel is None else gt_voxel
    gt_vols = np.stack(gt_vols, axis=0)

    # Recovar final per-class volumes (latest iter).
    iter_dirs = sorted(p for p in recovar_dir.glob("iter_*") if p.is_dir())
    if not iter_dirs:
        raise SystemExit(f"no iter_*/ in {recovar_dir}")
    last_iter = iter_dirs[-1]
    rec_classes = sorted(last_iter.glob("class_*.mrc"))
    rec_vols, rec_voxel = [], None
    for p in rec_classes:
        v, vx = load_mrc(p)
        rec_vols.append(v)
        rec_voxel = vx if rec_voxel is None else rec_voxel
    rec_vols = np.stack(rec_vols, axis=0)

    # RELION final classes (latest iter).
    rel_iter = relion_final_iter(relion_dir)
    rel_classes = sorted(relion_dir.glob(f"run_it{rel_iter:03d}_class*.mrc"))
    rel_vols, rel_voxel = [], None
    for p in rel_classes:
        v, vx = load_mrc(p)
        rel_vols.append(v)
        rel_voxel = vx if rel_voxel is None else rel_voxel
    rel_vols = np.stack(rel_vols, axis=0)

    # Match all to the recovar grid for fair FSC.
    grid = rec_vols.shape[1]
    gt_for_match = np.stack([crop_or_downsample(v, grid) for v in gt_vols])
    rel_for_match = np.stack([crop_or_downsample(v, grid) for v in rel_vols])

    rec_match = hungarian_match(rec_vols, gt_for_match)
    rel_match = hungarian_match(rel_for_match, gt_for_match)

    # Wall times.
    rec_summary_path = recovar_dir / "summary.json"
    rec_total = float("nan")
    rec_n_iter = -1
    if rec_summary_path.exists():
        rec_summary = json.loads(rec_summary_path.read_text())
        rec_total = float(rec_summary.get("runtime_s", float("nan")))
        rec_n_iter = int(rec_summary.get("n_iters_completed", -1))
    rel_total, rel_per_iter = relion_walltime_from_iters(relion_dir)

    return {
        "mode": "kN",
        "n_classes_recovar": rec_vols.shape[0],
        "n_classes_relion": rel_vols.shape[0],
        "n_classes_gt": gt_vols.shape[0],
        "ground_truth_dir": str(gt_dir),
        "recovar": {
            "dir": str(recovar_dir),
            "voxel_size": rec_voxel,
            "grid_size": grid,
            "n_iter": rec_n_iter,
            "wall_total_s": rec_total,
            **{f"hungarian_{k}": v for k, v in rec_match.items()},
        },
        "relion": {
            "dir": str(relion_dir),
            "voxel_size": rel_voxel,
            "grid_size": grid,
            "final_iter": rel_iter,
            "wall_total_s": rel_total,
            "wall_per_iter_s": rel_per_iter,
            **{f"hungarian_{k}": v for k, v in rel_match.items()},
        },
    }


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------


def render_table(summary: dict) -> str:
    lines: list[str] = []
    if summary["mode"] == "k1":
        rec = summary["recovar"]
        rel = summary["relion"]

        def hours(s):
            return f"{s / 3600:.2f}h" if s == s and s > 0 else "—"

        lines += [
            "=" * 64,
            "k=1 auto-refine: recovar vs RELION",
            "=" * 64,
            "",
            f"{'metric':<32} {'recovar':>14} {'RELION':>14}",
            "-" * 64,
            f"{'iters to convergence':<32} {rec['n_iter']:>14d} {rel['final_iter']:>14d}",
            f"{'wall total':<32} {hours(rec['wall_total_s']):>14} {hours(rel['wall_total_s']):>14}",
            f"{'wall / iter (mean)':<32} "
            f"{(rec['wall_total_s'] / max(rec['n_iter'], 1)) / 60:>11.1f} min "
            f"{(rel['wall_total_s'] / max(rel['final_iter'], 1)) / 60:>11.1f} min",
            "",
            "FSC vs GT:",
            f"{'  area mean':<32} {rec['fsc_vs_gt']['area_mean']:>14.4f} {rel['fsc_vs_gt']['area_mean']:>14.4f}",
            f"{'  resolution @ 0.5  (Å)':<32} "
            f"{rec['fsc_vs_gt']['fsc_at_05_angstrom']:>14.2f} "
            f"{rel['fsc_vs_gt']['fsc_at_05_angstrom']:>14.2f}",
            f"{'  resolution @ 0.143 (Å)':<32} "
            f"{rec['fsc_vs_gt']['fsc_at_0143_angstrom']:>14.2f} "
            f"{rel['fsc_vs_gt']['fsc_at_0143_angstrom']:>14.2f}",
            "",
            "Half-map gold-standard FSC:",
            f"{'  resolution @ 0.143 (Å)':<32} "
            f"{(rec['fsc_halfmap'].get('fsc_at_0143_angstrom', float('nan'))):>14.2f} "
            f"{rel['fsc_halfmap']['fsc_at_0143_angstrom']:>14.2f}",
            "",
        ]
    else:  # kN
        rec = summary["recovar"]
        rel = summary["relion"]

        def hours(s):
            return f"{s / 3600:.2f}h" if s == s and s > 0 else "—"

        lines += [
            "=" * 64,
            f"k={summary['n_classes_recovar']} auto-refine: recovar vs RELION",
            "=" * 64,
            "",
            f"{'metric':<32} {'recovar':>14} {'RELION':>14}",
            "-" * 64,
            f"{'wall total':<32} {hours(rec['wall_total_s']):>14} {hours(rel['wall_total_s']):>14}",
            "",
            "FSC vs GT (Hungarian-matched):",
            f"{'  area mean':<32} {rec['hungarian_fsc_area_mean']:>14.4f} {rel['hungarian_fsc_area_mean']:>14.4f}",
            "",
            "Per-class FSC area:",
            f"{'  recovar':<32} {rec['hungarian_fsc_area_per_class']}",
            f"{'  RELION':<32}  {rel['hungarian_fsc_area_per_class']}",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--mode", choices=["k1", "kN"], required=True)
    p.add_argument("--recovar-dir", required=True)
    p.add_argument("--relion-dir", required=True)
    p.add_argument("--gt-mrc", help="(k1) reference_gt.mrc")
    p.add_argument("--gt-dir", help="(kN) directory containing reference_gt_class*.mrc")
    p.add_argument("--output", default="comparison.json")
    args = p.parse_args()

    if args.mode == "k1" and not args.gt_mrc:
        sys.exit("--gt-mrc required in k1 mode")
    if args.mode == "kN" and not args.gt_dir:
        sys.exit("--gt-dir required in kN mode")

    summary = compare_k1(args) if args.mode == "k1" else compare_kN(args)
    summary["generated_at"] = datetime.now().isoformat()

    out_json = Path(args.output)
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out_json}")

    table = render_table(summary)
    out_txt = out_json.with_suffix(".txt")
    out_txt.write_text(table)
    print(f"wrote {out_txt}")
    print()
    print(table)


if __name__ == "__main__":
    main()
