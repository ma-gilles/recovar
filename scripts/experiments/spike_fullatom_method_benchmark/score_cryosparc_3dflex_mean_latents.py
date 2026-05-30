#!/usr/bin/env python3
"""Score 3DFlex generated maps at GT-state mean latent coordinates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np


DEFAULT_SOURCE_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/"
    "n00100000/runs/n00100000_seed0000"
)
DEFAULT_EVAL_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sanity_100k_noise10_b100_20260529/"
    "evaluation_3dflex_mean_latents"
)
DEFAULT_PROJECT_DIR = Path("/tigress/CRYOEM/singerlab/mg6942/CS-testres")


def load_volume(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        data = np.asarray(handle.data, dtype=np.float32)
    if data.ndim == 3 and len(set(data.shape)) == 1:
        data = np.transpose(data, (2, 1, 0))
    return data


def shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    coords = np.indices(shape, dtype=np.float32)
    center = np.array(shape, dtype=np.float32)[:, None, None, None] // 2
    radius = np.sqrt(((coords - center) ** 2).sum(axis=0))
    n_shells = max(1, min(shape) // 2 - 1)
    labels = np.clip(np.rint(radius).astype(np.int32), 0, n_shells - 1)
    return labels, n_shells


def masked_fsc(volume: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    labels, n_shells = shell_labels(target.shape)
    volume_ft = np.fft.fftshift(np.fft.fftn(volume * mask))
    target_ft = np.fft.fftshift(np.fft.fftn(target * mask))
    cross = np.bincount(
        labels.ravel(),
        weights=np.real(np.conj(volume_ft).ravel() * target_ft.ravel()),
        minlength=n_shells,
    )
    vol_power = np.bincount(
        labels.ravel(), weights=np.abs(volume_ft).ravel() ** 2, minlength=n_shells
    )
    target_power = np.bincount(
        labels.ravel(), weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(vol_power * target_power)
    fsc[~np.isfinite(fsc)] = 0.0
    if fsc.size > 1:
        fsc[0] = fsc[1]
    return fsc.astype(np.float32)


def resolution_at_threshold(
    fsc: np.ndarray, voxel_size: float, box_size: int, threshold: float
) -> float:
    valid = np.isfinite(fsc)
    valid[0] = False
    below = np.flatnonzero(valid & (fsc < threshold))
    if below.size == 0:
        shell = int(np.flatnonzero(valid)[-1]) if np.any(valid) else 0
    else:
        shell = int(below[0])
    if shell <= 0:
        return math.nan
    return float(box_size * voxel_size / shell)


def real_space_metrics(volume: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    active = mask > 0
    if not np.any(active):
        return math.nan, math.nan
    v = volume[active].astype(np.float64)
    t = target[active].astype(np.float64)
    rel_l2 = float(np.linalg.norm(v - t) / max(np.linalg.norm(t), 1e-30))
    corr = math.nan
    if v.size > 1 and np.std(v) > 0 and np.std(t) > 0:
        corr = float(np.corrcoef(v, t)[0, 1])
    return rel_l2, corr


def read_manifest(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def series_frames(project_dir: Path, job_uid: str) -> list[Path]:
    job_dir = project_dir / job_uid
    frames = sorted(job_dir.glob(f"{job_uid}_series_*/{job_uid}_series_*_frame_*.mrc"))
    return frames


def frame_index(path: Path) -> int:
    match = re.search(r"_frame_(\d+)\.mrc$", path.name)
    if not match:
        raise ValueError(f"Cannot parse frame index from {path}")
    return int(match.group(1))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(
    rows: list[dict[str, object]],
    curves: dict[str, np.ndarray],
    out_dir: Path,
    voxel_size: float,
    box_size: int,
) -> None:
    if not rows:
        return
    colors = {
        "default_mask_k1": "#1b9e77",
        "default_mask_k2": "#d95f02",
        "j397_mask_k1": "#7570b3",
        "j397_mask_k2": "#e7298a",
    }
    masks = sorted({str(row["metric_mask"]) for row in rows})
    states = sorted({int(row["state"]) for row in rows})
    freq = np.arange(next(iter(curves.values())).size, dtype=np.float64) / (
        box_size * voxel_size
    )
    for mask_name in masks:
        for state in states:
            subset = [
                row
                for row in rows
                if row["metric_mask"] == mask_name and int(row["state"]) == state
            ]
            if not subset:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
            for row in subset:
                key = str(row["curve_key"])
                model = str(row["model"])
                label = f"{model}: {float(row['fsc05_resolution_A']):.2f} A"
                fsc = curves[key]
                axes[0].plot(freq, fsc, lw=2, color=colors.get(model), label=label)
                axes[1].semilogy(
                    freq,
                    np.clip(1.0 - fsc, 1e-4, None),
                    lw=2,
                    color=colors.get(model),
                    label=model,
                )
            axes[0].axhline(0.5, color="0.35", ls="--", lw=0.8)
            axes[0].axhline(1.0 / 7.0, color="0.35", ls=":", lw=0.8)
            axes[0].set_ylim(-0.05, 1.02)
            axes[0].set_ylabel("FSC vs GT")
            axes[1].set_ylabel("FSC error (1 - FSC)")
            for ax in axes:
                ax.set_xlabel("spatial frequency (1/A)")
                ax.grid(True, color="0.9", lw=0.5)
            axes[0].set_title(f"3DFlex mean-latent FSC, state {state}, {mask_name}")
            axes[1].set_title("FSC error")
            axes[0].legend(fontsize=7)
            fig.tight_layout()
            stem = f"3dflex_mean_latent_{mask_name}_state{state:04d}_fsc_error"
            fig.savefig(out_dir / f"{stem}.png", dpi=180)
            fig.savefig(out_dir / f"{stem}.pdf")
            plt.close(fig)

    for mask_name in masks:
        subset = [row for row in rows if row["metric_mask"] == mask_name]
        models = sorted({str(row["model"]) for row in subset})
        x = np.arange(len(states), dtype=np.float64)
        width = 0.8 / max(1, len(models))
        fig, ax = plt.subplots(figsize=(7, 4))
        for idx, model in enumerate(models):
            values = []
            for state in states:
                match = [
                    row
                    for row in subset
                    if str(row["model"]) == model and int(row["state"]) == state
                ]
                values.append(float(match[0]["fsc_auc"]) if match else math.nan)
            ax.bar(x + (idx - (len(models) - 1) / 2) * width, values, width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels([str(state) for state in states])
        ax.set_xlabel("GT state")
        ax.set_ylabel("FSC AUC, higher is better")
        ax.set_title(f"3DFlex mean-latent FSC AUC ({mask_name})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"3dflex_mean_latent_{mask_name}_auc_by_state.png", dpi=180)
        fig.savefig(out_dir / f"3dflex_mean_latent_{mask_name}_auc_by_state.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=1.25)
    parser.add_argument("--box-size", type=int, default=256)
    args = parser.parse_args()

    eval_root = args.eval_root.resolve()
    manifest_path = args.manifest or (eval_root / "3dflex_generate_mean_latents_manifest.json")
    manifest = read_manifest(manifest_path)
    out_dir = eval_root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = {
        "focus": np.clip(
            load_volume(args.source_run / "05_masks/focus_mask_moving.mrc"), 0.0, 1.0
        ),
        "global": np.clip(
            load_volume(args.source_run / "05_masks/volume_mask_union.mrc"), 0.0, 1.0
        ),
    }
    gt_by_state = {
        0: load_volume(args.source_run / "04_ground_truth/gt_vol0000.mrc"),
        25: load_volume(args.source_run / "04_ground_truth/gt_vol0025.mrc"),
        50: load_volume(args.source_run / "04_ground_truth/gt_vol0050.mrc"),
    }

    rows: list[dict[str, object]] = []
    curves: dict[str, np.ndarray] = {}
    skipped: list[dict[str, str]] = []
    for model in manifest["models"]:
        model_name = str(model["model"])
        job_uid = str(model.get("mean_latents_generate", ""))
        states = [int(x) for x in model["states"]]
        if not job_uid:
            skipped.append({"model": model_name, "reason": "no mean_latents_generate job"})
            continue
        frames = series_frames(args.project_dir.resolve(), job_uid)
        if len(frames) < len(states):
            skipped.append(
                {
                    "model": model_name,
                    "reason": f"found {len(frames)} frames for {len(states)} states",
                }
            )
            continue
        for frame in frames[: len(states)]:
            idx = frame_index(frame)
            state = states[idx]
            volume = load_volume(frame)
            target = gt_by_state[state]
            for mask_name, mask in masks.items():
                fsc = masked_fsc(volume, target, mask)
                rel_l2, corr = real_space_metrics(volume, target, mask)
                curve_key = f"{model_name}_{job_uid}_state{state:04d}_{mask_name}"
                curves[curve_key] = fsc
                rows.append(
                    {
                        "model": model_name,
                        "train_job": model["train_job"],
                        "highres_job": model["highres_job"],
                        "generate_job": job_uid,
                        "mask_mode": model["mask_mode"],
                        "metric_mask": mask_name,
                        "state": state,
                        "frame_index": idx,
                        "volume_path": str(frame),
                        "fsc_auc": float(np.mean(fsc)),
                        "fsc_error_auc": float(np.mean(1.0 - fsc)),
                        "fsc05_resolution_A": resolution_at_threshold(
                            fsc, args.voxel_size, args.box_size, 0.5
                        ),
                        "fsc_1_7_resolution_A": resolution_at_threshold(
                            fsc, args.voxel_size, args.box_size, 1.0 / 7.0
                        ),
                        "rel_l2": rel_l2,
                        "corr": corr,
                        "curve_key": curve_key,
                    }
                )

    metrics_csv = out_dir / "3dflex_mean_latent_vs_gt_metrics.csv"
    write_csv(metrics_csv, rows)
    np.savez_compressed(
        out_dir / "3dflex_mean_latent_vs_gt_fsc_curves.npz",
        **curves,
        skipped=json.dumps(skipped),
    )
    plot_curves(rows, curves, out_dir, args.voxel_size, args.box_size)
    summary = {
        "manifest": str(manifest_path),
        "metrics_csv": str(metrics_csv),
        "n_rows": len(rows),
        "skipped": skipped,
    }
    (out_dir / "3dflex_mean_latent_scoring_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
