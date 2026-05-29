#!/usr/bin/env python3
"""Score decoded benchmark volumes against matching GT volumes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np


DEFAULT_SOURCE_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/"
    "n00100000/runs/n00100000_seed0000"
)
DEFAULT_BENCH_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_benchmark_100k_20260517")


@dataclass(frozen=True)
class DecodedSet:
    method: str
    run_name: str
    decoded_dir: Path
    labels: list[int]


def _load_volume(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        data = np.asarray(handle.data, dtype=np.float32)
    # Match recovar.utils.helpers.load_mrc convention for cubic cryoSPARC/cryoDRGN volumes.
    if data.ndim == 3 and np.isclose(data.shape, data.shape[0]).all():
        data = np.transpose(data, (2, 1, 0))
    return data


def _voxel_size(path: Path) -> float:
    with mrcfile.open(path, permissive=True) as handle:
        voxel = handle.voxel_size
        return float(voxel.x) if float(voxel.x) > 0 else 1.0


def _shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    coords = np.indices(shape, sparse=False, dtype=np.float32)
    center = np.array(shape, dtype=np.float32)[:, None, None, None] // 2
    radius = np.sqrt(((coords - center) ** 2).sum(axis=0))
    labels = np.rint(radius).astype(np.int32)
    n_shells = max(1, min(shape) // 2 - 1)
    return np.clip(labels, 0, n_shells - 1), n_shells


def _masked_fsc(volume: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    labels, n_shells = _shell_labels(target.shape)
    ft = np.fft.fftshift(np.fft.fftn(volume * mask))
    target_ft = np.fft.fftshift(np.fft.fftn(target * mask))
    top = np.bincount(labels.ravel(), weights=np.real(np.conj(ft).ravel() * target_ft.ravel()), minlength=n_shells)
    bot1 = np.bincount(labels.ravel(), weights=np.abs(ft).ravel() ** 2, minlength=n_shells)
    bot2 = np.bincount(labels.ravel(), weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = top / np.sqrt(bot1 * bot2)
    fsc[~np.isfinite(fsc)] = 0.0
    if fsc.size > 1:
        fsc[0] = fsc[1]
    return fsc.astype(np.float32)


def _resolution_at_threshold(fsc: np.ndarray, voxel_size: float, box_size: int, threshold: float) -> float:
    below = np.flatnonzero(fsc[1:] < threshold)
    if below.size == 0:
        shell = max(1, len(fsc) - 1)
    else:
        shell = int(below[0]) + 1
    if shell <= 0:
        return math.nan
    return float(box_size * voxel_size / shell)


def _real_space_metrics(volume: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    active = mask > 0
    if not np.any(active):
        return math.nan, math.nan
    v = volume[active].astype(np.float64)
    t = target[active].astype(np.float64)
    rel_l2 = float(np.linalg.norm(v - t) / max(np.linalg.norm(t), 1e-30))
    if v.size < 2 or np.std(v) == 0 or np.std(t) == 0:
        corr = math.nan
    else:
        corr = float(np.corrcoef(v, t)[0, 1])
    return rel_l2, corr


def _local_resolution_summary(
    volume: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    voxel_size: float,
    sampling_angstrom: float,
    radius_angstrom: float,
    max_points: int,
) -> tuple[float, float, int]:
    step = max(1, int(round(sampling_angstrom / voxel_size)))
    radius = max(3, int(round(radius_angstrom / voxel_size)))
    active = np.argwhere(mask > 0.5)
    if active.size == 0:
        return math.nan, math.nan, 0
    mins = np.maximum(active.min(axis=0), radius)
    maxs = np.minimum(active.max(axis=0), np.array(mask.shape) - radius - 1)
    grids = [np.arange(mins[axis], maxs[axis] + 1, step, dtype=np.int32) for axis in range(3)]
    centers = np.array(np.meshgrid(*grids, indexing="ij")).reshape(3, -1).T
    centers = np.array([center for center in centers if mask[tuple(center)] > 0.5], dtype=np.int32)
    if centers.size == 0:
        return math.nan, math.nan, 0
    if centers.shape[0] > max_points:
        rng = np.random.default_rng(0)
        centers = centers[rng.choice(centers.shape[0], size=max_points, replace=False)]

    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    sphere = ((xx**2 + yy**2 + zz**2) <= radius**2).astype(np.float32)
    values: list[float] = []
    for center in centers:
        slices = tuple(slice(int(c) - radius, int(c) + radius + 1) for c in center)
        local_mask = sphere * (mask[slices] > 0.5)
        if local_mask.sum() < 32:
            continue
        fsc = _masked_fsc(volume[slices], target[slices], local_mask)
        values.append(_resolution_at_threshold(fsc, voxel_size, local_mask.shape[0], 1.0 / 7.0))
    valid = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float32)
    if valid.size == 0:
        return math.nan, math.nan, 0
    return float(np.median(valid)), float(np.percentile(valid, 90)), int(valid.size)


def _label_manifest(decoded_dir: Path) -> list[int] | None:
    mean_root = decoded_dir.parents[1] / "mean_embeddings"
    for manifest in sorted(mean_root.glob("labels_mean_z*.json")):
        try:
            data = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            continue
        labels = data.get("labels")
        if isinstance(labels, list):
            return [int(x) for x in labels]
    return None


def _discover_decoded_sets(evaluation_root: Path) -> list[DecodedSet]:
    sets: list[DecodedSet] = []
    for decoded_dir in sorted(evaluation_root.glob("*/*/decoded_volumes/*")):
        if not decoded_dir.is_dir():
            continue
        volumes = sorted(decoded_dir.glob("*.mrc"))
        if not volumes:
            continue
        labels = _label_manifest(decoded_dir) or list(range(len(volumes)))
        sets.append(
            DecodedSet(
                method=decoded_dir.parents[2].name,
                run_name=decoded_dir.parents[1].name,
                decoded_dir=decoded_dir,
                labels=labels,
            )
        )
    return sets


def _volume_index(path: Path) -> int | None:
    match = re.search(r"(\d+)(?=\.mrc$)", path.name)
    return int(match.group(1)) if match else None


def _write_plots(rows: list[dict[str, object]], fsc_curves: dict[str, list[tuple[str, np.ndarray]]], out_dir: Path) -> None:
    if not rows:
        return
    for mask_name, curves in fsc_curves.items():
        if not curves:
            continue
        plt.figure(figsize=(8, 5))
        grouped: dict[str, list[np.ndarray]] = {}
        for name, fsc in curves:
            grouped.setdefault(name, []).append(fsc)
            plt.plot(np.arange(len(fsc)), fsc, color="0.75", alpha=0.15, linewidth=0.7)
        for name, values in grouped.items():
            min_len = min(len(v) for v in values)
            mean = np.mean([v[:min_len] for v in values], axis=0)
            plt.plot(np.arange(min_len), mean, linewidth=2.0, label=name)
        plt.axhline(0.5, color="0.35", linestyle="--", linewidth=0.8)
        plt.axhline(1.0 / 7.0, color="0.35", linestyle=":", linewidth=0.8)
        plt.xlabel("Fourier shell")
        plt.ylabel("FSC vs GT")
        plt.title(f"{mask_name} masked FSC")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"fsc_curves_{mask_name}.png", dpi=160)
        plt.close()

    labels = []
    values = []
    for row in rows:
        if row["metric_mask"] != "focus":
            continue
        labels.append(f"{row['method']}/{row['run_name']}")
        values.append(float(row["fsc_auc"]))
    if values:
        plt.figure(figsize=(max(6, 0.4 * len(values)), 4))
        plt.bar(np.arange(len(values)), values)
        plt.xticks(np.arange(len(values)), labels, rotation=45, ha="right", fontsize=7)
        plt.ylabel("Mean masked FSC AUC")
        plt.tight_layout()
        plt.savefig(out_dir / "focus_fsc_auc_by_volume.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=None, help="Default: BENCH_ROOT/evaluation")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: EVALUATION_ROOT/metrics")
    parser.add_argument("--max-volumes", type=int, default=None, help="Smoke-test limit per decoded set.")
    parser.add_argument("--compute-locres", action="store_true", help="Compute sampled local FSC resolution summaries.")
    parser.add_argument("--locres-sampling", type=float, default=24.0)
    parser.add_argument("--locres-radius", type=float, default=24.0)
    parser.add_argument("--locres-max-points", type=int, default=400)
    args = parser.parse_args()

    evaluation_root = args.evaluation_root or (args.bench_root / "evaluation")
    out_dir = args.out_dir or (evaluation_root / "metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = {
        "focus": _load_volume(args.source_run / "05_masks" / "focus_mask_moving.mrc"),
        "global": _load_volume(args.source_run / "05_masks" / "volume_mask_union.mrc"),
    }
    decoded_sets = _discover_decoded_sets(evaluation_root)
    if not decoded_sets:
        print(f"SKIP scoring: no decoded volume sets found under {evaluation_root}")
        return

    rows: list[dict[str, object]] = []
    fsc_curves: dict[str, list[tuple[str, np.ndarray]]] = {name: [] for name in masks}
    for decoded_set in decoded_sets:
        volumes = sorted(decoded_set.decoded_dir.glob("*.mrc"))
        if args.max_volumes is not None:
            volumes = volumes[: args.max_volumes]
        for volume_path in volumes:
            index = _volume_index(volume_path)
            if index is None:
                print(f"SKIP {volume_path}: cannot parse label index")
                continue
            gt_label = decoded_set.labels[index] if index < len(decoded_set.labels) else index
            gt_path = args.source_run / "04_ground_truth" / f"gt_vol{gt_label:04d}.mrc"
            if not gt_path.exists():
                print(f"SKIP {volume_path}: missing GT {gt_path}")
                continue
            pred = _load_volume(volume_path)
            target = _load_volume(gt_path)
            voxel = _voxel_size(gt_path)
            if pred.shape != target.shape:
                print(f"SKIP {volume_path}: shape {pred.shape} != GT shape {target.shape}")
                continue
            for mask_name, mask in masks.items():
                if mask.shape != target.shape:
                    print(f"SKIP mask {mask_name}: shape {mask.shape} != target shape {target.shape}")
                    continue
                metric_mask = (mask > 0.5).astype(np.float32)
                fsc = _masked_fsc(pred, target, metric_mask)
                rel_l2, corr = _real_space_metrics(pred, target, metric_mask)
                locres_median = locres_p90 = math.nan
                locres_points = 0
                if args.compute_locres:
                    locres_median, locres_p90, locres_points = _local_resolution_summary(
                        pred,
                        target,
                        metric_mask,
                        voxel,
                        args.locres_sampling,
                        args.locres_radius,
                        args.locres_max_points,
                    )
                rows.append(
                    {
                        "method": decoded_set.method,
                        "run_name": decoded_set.run_name,
                        "decoded_dir": str(decoded_set.decoded_dir),
                        "volume": str(volume_path),
                        "gt_label": int(gt_label),
                        "gt_volume": str(gt_path),
                        "metric_mask": mask_name,
                        "voxel_size": voxel,
                        "fsc_auc": float(np.mean(np.clip(fsc[1:], -1, 1))) if fsc.size > 1 else math.nan,
                        "fsc_res_0p5_A": _resolution_at_threshold(fsc, voxel, target.shape[0], 0.5),
                        "fsc_res_1over7_A": _resolution_at_threshold(fsc, voxel, target.shape[0], 1.0 / 7.0),
                        "masked_rel_l2": rel_l2,
                        "masked_corr": corr,
                        "locres_median_A": locres_median,
                        "locres_90pct_A": locres_p90,
                        "locres_points": locres_points,
                    }
                )
                fsc_curves[mask_name].append((f"{decoded_set.method}/{decoded_set.run_name}", fsc))

    csv_path = out_dir / "decoded_volume_metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out_dir / "decoded_volume_fsc_curves.npz",
        **{
            f"{mask}_{idx:05d}": fsc
            for mask, curves in fsc_curves.items()
            for idx, (_name, fsc) in enumerate(curves)
        },
    )
    _write_plots(rows, fsc_curves, out_dir)
    print(f"WROTE {csv_path}")
    print(f"WROTE {out_dir / 'decoded_volume_fsc_curves.npz'}")


if __name__ == "__main__":
    main()
