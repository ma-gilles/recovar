#!/usr/bin/env python3
"""Score nonuniform spike method maps with corrected trajectory masks.

The state-50 broad mask is useful for the middle conformation, but it is not a
valid trajectory-wide scoring mask.  This script provides two corrected
trajectory scores:

* ``moving``: reuse the source run's soft moving/focus mask for every state.
* ``tracked_atoms``: select atoms that land inside the state-50 broad mask,
  track those atom indices through the morph PDB trajectory, and build a soft
  per-state mask around their new positions.
"""

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
import numpy as np
from scipy import ndimage

from recovar import utils
from recovar.core import fourier_transform_utils as ftu


DEFAULT_SOURCE_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_nonuniform_B70_noise3_b80_300k_20260602/"
    "n00300000/runs/n00300000_seed0000"
)
DEFAULT_BENCH_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_nonuniform_B70_noise3_b80_300k_20260602"
)
DEFAULT_BROAD_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
)
DEFAULT_PDB_DIR = Path("/projects/CRYOEM/singerlab/mg6942/spike_morph_pdbs")
DEFAULT_FLEX_PROJECT_DIR = Path("/projects/CRYOEM/singerlab/mg6942/CS-testres")
DEFAULT_STATES = (0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90)
METHOD_COLORS = {
    "recovar": "#3B7EA1",
    "cryodrgn": "#2AA876",
    "3dflex": "#C17D11",
}


def dft3(volume: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(volume)))


def shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, int]:
    labels = np.asarray(ftu.get_grid_of_radial_distances(shape, rounded=True), dtype=np.int32)
    n_shells = shape[0] // 2 - 1
    valid = (labels >= 0) & (labels < n_shells)
    return labels, valid, n_shells


def shell_metrics(estimate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    labels, valid, n_shells = shell_labels(estimate.shape)
    estimate_ft = dft3(estimate * mask)
    target_ft = dft3(target * mask)
    diff_ft = estimate_ft - target_ft
    flat = labels[valid].ravel()
    cross = np.bincount(
        flat,
        weights=np.real(np.conj(estimate_ft[valid]).ravel() * target_ft[valid].ravel()),
        minlength=n_shells,
    )
    estimate_power = np.bincount(flat, weights=np.abs(estimate_ft[valid]).ravel() ** 2, minlength=n_shells)
    target_power = np.bincount(flat, weights=np.abs(target_ft[valid]).ravel() ** 2, minlength=n_shells)
    diff_power = np.bincount(flat, weights=np.abs(diff_ft[valid]).ravel() ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(estimate_power * target_power)
        relerr = diff_power / target_power
        cum_relerr = np.cumsum(diff_power) / np.cumsum(target_power)
    fsc[~np.isfinite(fsc)] = 0.0
    relerr[~np.isfinite(relerr)] = np.nan
    cum_relerr[~np.isfinite(cum_relerr)] = np.nan
    if fsc.size > 1:
        fsc[0] = fsc[1]
    return {
        "fsc": fsc.astype(np.float32),
        "relative_error": relerr.astype(np.float32),
        "cumulative_relative_error": cum_relerr.astype(np.float32),
        "target_power": target_power.astype(np.float64),
        "diff_power": diff_power.astype(np.float64),
    }


def last_good_resolution(
    frequency: np.ndarray,
    values: np.ndarray,
    threshold: float,
    higher_is_better: bool,
) -> float:
    valid = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0)
    frequency = np.asarray(frequency, dtype=np.float64)[valid]
    values = np.asarray(values, dtype=np.float64)[valid]
    if frequency.size == 0:
        return math.nan
    good = values >= threshold if higher_is_better else values <= threshold
    indices = np.flatnonzero(good)
    if indices.size == 0:
        return math.nan
    return float(1.0 / frequency[int(indices[-1])])


def first_crossing_resolution(
    frequency: np.ndarray,
    values: np.ndarray,
    threshold: float,
    higher_is_better: bool,
) -> float:
    valid = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0)
    frequency = np.asarray(frequency, dtype=np.float64)[valid]
    values = np.asarray(values, dtype=np.float64)[valid]
    if frequency.size == 0:
        return math.nan
    bad = values < threshold if higher_is_better else values > threshold
    indices = np.flatnonzero(bad)
    if indices.size == 0:
        return float(1.0 / frequency[-1])
    index = int(indices[0])
    if index == 0:
        return float(1.0 / frequency[index])
    x0 = frequency[index - 1]
    x1 = frequency[index]
    y0 = values[index - 1]
    y1 = values[index]
    if np.isfinite(y0) and np.isfinite(y1) and y0 != y1:
        frac = float((threshold - y0) / (y1 - y0))
        frac = min(1.0, max(0.0, frac))
        crossing_frequency = x0 + frac * (x1 - x0)
        if crossing_frequency > 0:
            return float(1.0 / crossing_frequency)
    return float(1.0 / x1)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_states(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def parse_pdb_coords(path: Path) -> np.ndarray:
    coords: list[tuple[float, float, float]] = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                coords.append(
                    (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    )
                )
    if not coords:
        raise ValueError(f"No ATOM/HETATM coordinates found in {path}")
    return np.asarray(coords, dtype=np.float32)


def pdb_for_state(pdb_dir: Path, state: int) -> Path:
    candidates = [
        pdb_dir / f"morph_{state:03d}.pdb",
        pdb_dir / f"morph_{state + 1:03d}.pdb",
        pdb_dir / f"morph_{max(1, state):03d}.pdb",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(candidates[0])


def coords_to_voxels(coords: np.ndarray, center: np.ndarray, shape: tuple[int, int, int], voxel_size: float) -> np.ndarray:
    # RECOVAR's utils.load_mrc convention transposes cube MRCs into x/y/z order.
    return np.rint((coords - center.reshape(1, 3)) / voxel_size + np.asarray(shape) / 2.0).astype(np.int32)


def sample_volume_at_points(volume: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = np.asarray(volume.shape)
    in_bounds = np.all((points >= 0) & (points < shape.reshape(1, 3)), axis=1)
    values = np.zeros(points.shape[0], dtype=np.float32)
    if np.any(in_bounds):
        values[in_bounds] = volume[tuple(points[in_bounds].T)]
    return values, in_bounds


def cosine_soft_mask(binary: np.ndarray, edge_voxels: float) -> np.ndarray:
    binary = np.asarray(binary, dtype=bool)
    outside_distance = ndimage.distance_transform_edt(~binary)
    mask = np.zeros(binary.shape, dtype=np.float32)
    mask[binary] = 1.0
    edge = (~binary) & (outside_distance <= edge_voxels)
    mask[edge] = 0.5 * (1.0 + np.cos(np.pi * outside_distance[edge] / edge_voxels))
    return mask


def rasterize_tracked_mask(
    coords: np.ndarray,
    selected_indices: np.ndarray,
    *,
    shape: tuple[int, int, int],
    voxel_size: float,
    atom_radius_a: float,
    soft_edge_voxels: float,
) -> np.ndarray:
    center = coords.mean(axis=0, dtype=np.float64).astype(np.float32)
    points = coords_to_voxels(coords[selected_indices], center, shape, voxel_size)
    in_bounds = np.all((points >= 0) & (points < np.asarray(shape).reshape(1, 3)), axis=1)
    impulses = np.zeros(shape, dtype=bool)
    if np.any(in_bounds):
        impulses[tuple(points[in_bounds].T)] = True
    distance = ndimage.distance_transform_edt(~impulses)
    hard = distance <= (atom_radius_a / voxel_size)
    return cosine_soft_mask(hard, soft_edge_voxels)


def build_masks(args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Path], dict[int, Path], dict[str, object]]:
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    moving_src = args.source_run / "05_masks/focus_mask_moving.mrc"
    moving = np.clip(np.asarray(utils.load_mrc(moving_src), dtype=np.float32), 0.0, 1.0)
    moving_path = mask_dir / "moving_mask_soft.mrc"
    utils.write_mrc(moving_path, moving, voxel_size=args.voxel_size)

    broad = np.clip(np.asarray(utils.load_mrc(args.broad_mask), dtype=np.float32), 0.0, 1.0)
    ref_pdb = pdb_for_state(args.pdb_dir, args.reference_state)
    ref_coords = parse_pdb_coords(ref_pdb)
    ref_center = ref_coords.mean(axis=0, dtype=np.float64).astype(np.float32)
    ref_points = coords_to_voxels(ref_coords, ref_center, broad.shape, args.voxel_size)
    broad_values, in_bounds = sample_volume_at_points(broad, ref_points)
    selected = np.flatnonzero(in_bounds & (broad_values > args.broad_threshold)).astype(np.int64)
    if selected.size == 0:
        raise RuntimeError("No reference atoms were selected by the broad mask")
    np.save(mask_dir / "tracked_atom_indices.npy", selected)

    tracked: dict[int, Path] = {}
    for state in args.states:
        pdb_path = pdb_for_state(args.pdb_dir, state)
        coords = parse_pdb_coords(pdb_path)
        if coords.shape[0] <= int(selected.max()):
            raise RuntimeError(
                f"{pdb_path} has {coords.shape[0]} atoms, but selected index {int(selected.max())}"
            )
        mask = rasterize_tracked_mask(
            coords,
            selected,
            shape=broad.shape,
            voxel_size=args.voxel_size,
            atom_radius_a=args.atom_radius_a,
            soft_edge_voxels=args.soft_edge_voxels,
        )
        path = mask_dir / f"state{state:04d}_tracked_atoms_soft.mrc"
        utils.write_mrc(path, mask, voxel_size=args.voxel_size)
        tracked[state] = path

    manifest: dict[str, object] = {
        "moving_mask_source": str(moving_src),
        "moving_mask_copy": str(moving_path),
        "broad_mask_for_atom_selection": str(args.broad_mask),
        "reference_state": int(args.reference_state),
        "reference_pdb": str(ref_pdb),
        "coordinate_convention": "RECOVAR load_mrc x/y/z, centered by per-state all-atom mean",
        "broad_threshold": float(args.broad_threshold),
        "selected_atom_count": int(selected.size),
        "atom_radius_A": float(args.atom_radius_a),
        "soft_edge_voxels": float(args.soft_edge_voxels),
        "states": list(args.states),
        "tracked_masks": {str(k): str(v) for k, v in tracked.items()},
    }
    (mask_dir / "mask_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"moving": moving_path}, tracked, manifest


def flex_frame_map(manifest_path: Path | None, project_dir: Path) -> dict[int, Path]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    frame_map: dict[int, Path] = {}
    model_records = manifest.get("models")
    if isinstance(model_records, list):
        records = model_records
    else:
        records = [manifest]
    for model in records:
        job_uid = str(model.get("mean_latents_generate", ""))
        if not job_uid:
            continue
        states = [int(x) for x in model.get("states", manifest.get("states", []))]
        frames = sorted((project_dir / job_uid).glob(f"{job_uid}_series_*/{job_uid}_series_*_frame_*.mrc"))
        for frame in frames:
            match = re.search(r"_frame_(\d+)\.mrc$", frame.name)
            if not match:
                continue
            idx = int(match.group(1))
            if idx < len(states):
                frame_map[states[idx]] = frame
    return frame_map


def estimate_path(args: argparse.Namespace, method: str, state: int, flex_frames: dict[int, Path]) -> Path | None:
    if method == "recovar":
        path = (
            args.bench_root
            / "recovar"
            / "n00300000"
            / "compute_state_zdim4_noreg_focus"
            / f"state{state:04d}"
            / "state000.mrc"
        )
        return path if path.exists() else None
    if method == "cryodrgn":
        for evaluation_dir in ("evaluation_extra_states", "evaluation"):
            path = (
                args.bench_root
                / "n00300000"
                / evaluation_dir
                / "cryodrgn"
                / "zdim1"
                / "decoded_volumes"
                / "labels_mean_z_epoch019_state_names"
                / f"gt_label_{state:03d}.mrc"
            )
            if path.exists():
                return path
            path = path.parent.parent / "labels_mean_z_epoch019" / f"gt_label_{state:03d}.mrc"
            if path.exists():
                return path
        return None
    if method == "3dflex":
        return flex_frames.get(state)
    raise ValueError(method)


def write_shell_csv(path: Path, metrics: dict[str, np.ndarray], frequency: np.ndarray) -> None:
    resolution = np.divide(1.0, frequency, out=np.full_like(frequency, np.inf), where=frequency > 0)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shell",
                "frequency_1_per_A",
                "resolution_A",
                "fsc_vs_gt",
                "relative_error_per_shell",
                "cumulative_relative_error",
                "target_power",
                "diff_power",
            ]
        )
        for idx in range(metrics["fsc"].size):
            writer.writerow(
                [
                    idx,
                    frequency[idx],
                    resolution[idx],
                    metrics["fsc"][idx],
                    metrics["relative_error"][idx],
                    metrics["cumulative_relative_error"][idx],
                    metrics["target_power"][idx],
                    metrics["diff_power"][idx],
                ]
            )


def score(args: argparse.Namespace, out_dir: Path, moving_masks: dict[str, Path], tracked_masks: dict[int, Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    flex_frames = flex_frame_map(args.flex_manifest, args.flex_project_dir)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    methods = ("recovar", "cryodrgn", "3dflex")
    for state in args.states:
        gt_path = args.source_run / "04_ground_truth" / f"gt_vol{state:04d}.mrc"
        if not gt_path.exists():
            skipped.append({"state": state, "reason": "missing_gt", "path": str(gt_path)})
            continue
        target = np.asarray(utils.load_mrc(gt_path), dtype=np.float32)
        masks = {
            "moving": moving_masks["moving"],
            "tracked_atoms": tracked_masks[state],
        }
        for method in methods:
            est_path = estimate_path(args, method, state, flex_frames)
            if est_path is None or not est_path.exists():
                skipped.append({"method": method, "state": state, "reason": "missing_estimate"})
                continue
            estimate = np.asarray(utils.load_mrc(est_path), dtype=np.float32)
            if estimate.shape != target.shape:
                skipped.append(
                    {
                        "method": method,
                        "state": state,
                        "reason": "shape_mismatch",
                        "estimate_shape": str(estimate.shape),
                        "target_shape": str(target.shape),
                    }
                )
                continue
            for mask_name, mask_path in masks.items():
                mask = np.clip(np.asarray(utils.load_mrc(mask_path), dtype=np.float32), 0.0, 1.0)
                metrics = shell_metrics(estimate, target, mask)
                frequency = np.arange(metrics["fsc"].size, dtype=np.float64) / (
                    estimate.shape[0] * args.voxel_size
                )
                fsc05_first = first_crossing_resolution(frequency, metrics["fsc"], 0.5, True)
                relerr05_first = first_crossing_resolution(
                    frequency, metrics["relative_error"], 0.5, False
                )
                fsc05_last_good = last_good_resolution(frequency, metrics["fsc"], 0.5, True)
                relerr05_last_good = last_good_resolution(
                    frequency, metrics["relative_error"], 0.5, False
                )
                shell_csv = out_dir / mask_name / "shell_metrics" / method / f"state{state:04d}.csv"
                shell_csv.parent.mkdir(parents=True, exist_ok=True)
                write_shell_csv(shell_csv, metrics, frequency)
                curve_npz = out_dir / mask_name / "curves" / method / f"state{state:04d}.npz"
                curve_npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(curve_npz, frequency=frequency, **metrics)
                rows.append(
                    {
                        "method": method,
                        "state": int(state),
                        "mask": mask_name,
                        "estimate": str(est_path),
                        "gt": str(gt_path),
                        "mask_path": str(mask_path),
                        "fsc05_resolution_A": fsc05_first,
                        "relerr05_resolution_A": relerr05_first,
                        "fsc05_first_crossing_resolution_A": fsc05_first,
                        "relerr05_first_crossing_resolution_A": relerr05_first,
                        "fsc05_last_good_resolution_A": fsc05_last_good,
                        "relerr05_last_good_resolution_A": relerr05_last_good,
                        "fsc_auc": float(np.nanmean(metrics["fsc"])),
                        "relerr_auc": float(np.nanmean(metrics["relative_error"])),
                        "shell_metrics_csv": str(shell_csv),
                        "curve_npz": str(curve_npz),
                    }
                )
    return rows, skipped


def load_curve(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def plot_curves(rows: list[dict[str, object]], out_dir: Path) -> None:
    for mask_name in sorted({str(row["mask"]) for row in rows}):
        for method in ("recovar", "cryodrgn", "3dflex"):
            subset = [
                row for row in rows if row["mask"] == mask_name and row["method"] == method
            ]
            if not subset:
                continue
            subset.sort(key=lambda r: int(r["state"]))
            colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(subset)))
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)
            for color, row in zip(colors, subset):
                curve = load_curve(Path(str(row["curve_npz"])))
                label = f"s{int(row['state'])}: {float(row['fsc05_resolution_A']):.2f} A"
                axes[0].plot(curve["frequency"], curve["fsc"], color=color, lw=1.8, label=label)
                axes[1].semilogy(
                    curve["frequency"],
                    curve["relative_error"],
                    color=color,
                    lw=1.8,
                    label=f"s{int(row['state'])}",
                )
            axes[0].axhline(0.5, color="0.35", ls="--", lw=1.0)
            axes[0].set_ylim(-0.05, 1.03)
            axes[0].set_ylabel("masked FSC vs GT")
            axes[0].set_title("FSC")
            axes[1].axhline(0.5, color="0.35", ls="--", lw=1.0)
            axes[1].set_ylim(1e-3, 1e3)
            axes[1].set_ylabel("relative Fourier shell error")
            axes[1].set_title("relative shell error")
            for ax in axes:
                ax.set_xlim(0.0, 0.40)
                ax.set_xlabel("spatial frequency (1/A)")
                ax.grid(True, alpha=0.25)
            axes[0].legend(title="state, FSC0.5", fontsize=7, ncol=2)
            fig.suptitle(f"{method} | {mask_name} corrected mask", weight="bold")
            stem = f"{mask_name}_{method}_all_states_fsc_relerr"
            plot_dir = out_dir / mask_name / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_dir / f"{stem}.png", dpi=220)
            fig.savefig(plot_dir / f"{stem}.pdf")
            plt.close(fig)


def plot_resolution(rows: list[dict[str, object]], out_dir: Path) -> None:
    for mask_name in sorted({str(row["mask"]) for row in rows}):
        subset = [row for row in rows if row["mask"] == mask_name]
        if not subset:
            continue
        states = sorted({int(row["state"]) for row in subset})
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
        for method in ("recovar", "cryodrgn", "3dflex"):
            method_rows = {int(row["state"]): row for row in subset if row["method"] == method}
            if not method_rows:
                continue
            y_fsc = [float(method_rows[s]["fsc05_resolution_A"]) if s in method_rows else math.nan for s in states]
            y_err = [float(method_rows[s]["relerr05_resolution_A"]) if s in method_rows else math.nan for s in states]
            color = METHOD_COLORS[method]
            axes[0].plot(states, y_fsc, marker="o", lw=2.2, color=color, label=method)
            axes[1].plot(states, y_err, marker="o", lw=2.2, color=color, label=method)
            for ax, values in zip(axes, (y_fsc, y_err)):
                for x, y in zip(states, values):
                    if np.isfinite(y):
                        ax.text(x, y, f"{y:.1f}", fontsize=7, color=color, ha="center", va="bottom")
        axes[0].set_title("FSC=0.5 resolution")
        axes[0].set_ylabel("resolution (A), lower is better")
        axes[1].set_title("relative shell error <= 0.5 resolution")
        axes[1].set_ylabel("resolution (A), lower is better")
        for ax in axes:
            ax.set_xlabel("GT state")
            ax.set_xticks(states)
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
        fig.suptitle(f"Nonuniform 300k noise3 | {mask_name} corrected mask", weight="bold")
        plot_dir = out_dir / mask_name / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / f"{mask_name}_resolution_by_state_all_methods.png", dpi=240)
        fig.savefig(plot_dir / f"{mask_name}_resolution_by_state_all_methods.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--broad-mask", type=Path, default=DEFAULT_BROAD_MASK)
    parser.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    parser.add_argument("--flex-project-dir", type=Path, default=DEFAULT_FLEX_PROJECT_DIR)
    parser.add_argument("--flex-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--states", default=",".join(str(x) for x in DEFAULT_STATES))
    parser.add_argument("--reference-state", type=int, default=50)
    parser.add_argument("--broad-threshold", type=float, default=0.01)
    parser.add_argument("--atom-radius-a", type=float, default=4.0)
    parser.add_argument("--soft-edge-voxels", type=float, default=6.0)
    parser.add_argument("--voxel-size", type=float, default=1.25)
    args = parser.parse_args()
    args.source_run = args.source_run.resolve()
    args.bench_root = args.bench_root.resolve()
    args.broad_mask = args.broad_mask.resolve()
    args.pdb_dir = args.pdb_dir.resolve()
    args.flex_project_dir = args.flex_project_dir.resolve()
    if args.flex_manifest is not None:
        args.flex_manifest = args.flex_manifest.resolve()
    args.states = parse_states(args.states)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else args.bench_root / "corrected_nonuniform_mask_scoring_20260603"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    moving_masks, tracked_masks, mask_manifest = build_masks(args, out_dir)
    rows, skipped = score(args, out_dir, moving_masks, tracked_masks)
    write_csv(out_dir / "corrected_mask_metrics.csv", rows)
    write_csv(out_dir / "corrected_mask_skipped.csv", skipped)
    plot_curves(rows, out_dir)
    plot_resolution(rows, out_dir)
    summary = {
        "source_run": str(args.source_run),
        "bench_root": str(args.bench_root),
        "flex_manifest": str(args.flex_manifest) if args.flex_manifest else None,
        "out_dir": str(out_dir),
        "n_metric_rows": len(rows),
        "n_skipped": len(skipped),
        "mask_manifest": mask_manifest,
        "metrics_csv": str(out_dir / "corrected_mask_metrics.csv"),
        "skipped_csv": str(out_dir / "corrected_mask_skipped.csv"),
    }
    (out_dir / "corrected_mask_scoring_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
