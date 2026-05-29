#!/usr/bin/env python3
"""Plot IgG-1D no-reg oracle compute_state controls for selected states.

This intentionally recomputes latent distances from the compute_state command
and current pipeline embedding.  It does not read cached GT embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "spike_fullatom_state_sweeps"))
from gt_embedding_controls import (  # noqa: E402
    parse_compute_state_inputs,
    raw_gt_pc_coordinates,
    recompute_latent_distances_from_compute_state,
)


VOXEL_SIZE_A = 1.25
FREQ_MAX = 0.40
N_STATES = 50
ZDIM = 4
NEAREST_COUNTS = (100, 1000)
GT_WINDOWS = (0, 5, 10, 15)


def load_mrc(path: Path) -> np.ndarray:
    return np.asarray(utils.load_mrc(path), dtype=np.float32)


def fft3_centered(volume: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(volume.astype(np.float32, copy=False))))


def shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    labels = np.asarray(ftu.get_grid_of_radial_distances(shape, rounded=True), dtype=np.int32)
    n_shells = shape[0] // 2 - 1
    return np.clip(labels, 0, n_shells - 1).ravel(), n_shells


def masked_metrics(
    volume: np.ndarray,
    mask: np.ndarray,
    labels_flat: np.ndarray,
    n_shells: int,
    target_ft: np.ndarray,
    target_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    volume_ft = fft3_centered(volume * mask)
    cross = np.bincount(
        labels_flat,
        weights=np.real(np.conj(volume_ft).ravel() * target_ft.ravel()),
        minlength=n_shells,
    )
    volume_power = np.bincount(labels_flat, weights=np.abs(volume_ft).ravel() ** 2, minlength=n_shells)
    diff_power = np.bincount(
        labels_flat,
        weights=np.abs(volume_ft.ravel() - target_ft.ravel()) ** 2,
        minlength=n_shells,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(volume_power * target_power)
        relerr = diff_power / np.maximum(target_power, 1e-30)
    fsc[~np.isfinite(fsc)] = np.nan
    relerr[~np.isfinite(relerr)] = np.nan
    if fsc.size > 1 and np.isfinite(fsc[1]):
        fsc[0] = fsc[1]
    return fsc.astype(np.float32), relerr.astype(np.float32)


def fsc05_resolution(freq: np.ndarray, fsc: np.ndarray) -> float:
    valid = np.isfinite(fsc) & (freq > 0)
    x = freq[valid]
    y = fsc[valid]
    if x.size == 0:
        return float("nan")
    below = np.flatnonzero(y < 0.5)
    if below.size == 0:
        return float(1.0 / x[-1])
    idx = int(below[0])
    if idx == 0:
        return float(1.0 / x[0])
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    crossing = x1 if y0 == y1 else x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    return float(1.0 / crossing)


def relerr_resolution(freq: np.ndarray, relerr: np.ndarray, threshold: float = 0.1) -> float:
    valid = np.isfinite(relerr) & (freq > 0)
    x = freq[valid]
    y = relerr[valid]
    if x.size == 0:
        return float("nan")
    above = np.flatnonzero(y > threshold)
    if above.size == 0:
        return float(1.0 / x[-1])
    return float(1.0 / x[int(above[0])])


def state_dir(compute_dir: Path) -> Path:
    return compute_dir / "diagnostics" / "state000"


def candidate_grid(compute_dir: Path) -> np.ndarray:
    with (state_dir(compute_dir) / "params.pkl").open("rb") as handle:
        params = pickle.load(handle)
    return np.asarray(params["heterogeneity_bins"], dtype=np.float64)


def candidate_paths(compute_dir: Path) -> list[tuple[Path, Path]]:
    base = state_dir(compute_dir)
    half1 = sorted((base / "estimates_half1_unfil").glob("*.mrc"))
    half2 = sorted((base / "estimates_half2_unfil").glob("*.mrc"))
    if len(half1) != len(half2) or not half1:
        raise FileNotFoundError(f"missing unfiltered candidate halfmaps in {base}")
    return list(zip(half1, half2))


def average_candidate(paths: tuple[Path, Path]) -> np.ndarray:
    return 0.5 * (load_mrc(paths[0]) + load_mrc(paths[1]))


def gt_paths(run_dir: Path) -> list[Path]:
    paths = sorted((run_dir / "04_ground_truth").glob("gt_vol[0-9][0-9][0-9][0-9].mrc"))
    if len(paths) != N_STATES:
        raise FileNotFoundError(f"expected {N_STATES} GT volumes in {run_dir / '04_ground_truth'}; found {len(paths)}")
    return paths


def weighted_gt(run_dir: Path, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    paths = gt_paths(run_dir)
    out = np.zeros_like(load_mrc(paths[0]), dtype=np.float32)
    for state, weight in enumerate(weights):
        if weight:
            out += np.float32(weight) * load_mrc(paths[state])
    return out


def state_weights_from_indices(run_dir: Path, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(np.load(run_dir / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    counts = np.bincount(states[indices], minlength=N_STATES).astype(np.float64)
    return counts / counts.sum(), counts


def nearest_indices(distances: np.ndarray, n_nearest: int) -> np.ndarray:
    n = min(int(n_nearest), distances.size)
    return np.argpartition(distances, n - 1)[:n] if n < distances.size else np.arange(distances.size)


def support_indices(distances: np.ndarray, h: float) -> np.ndarray:
    return np.flatnonzero(np.isfinite(distances) & (distances <= h))


def write_mrc(path: Path, volume: np.ndarray) -> None:
    utils.write_mrc(str(path), volume.astype(np.float32), voxel_size=VOXEL_SIZE_A)


def curve_row(label: str, fsc: np.ndarray, err: np.ndarray, color: str, ls="-", lw=2.0) -> dict[str, object]:
    return {"label": label, "fsc": fsc, "err": err, "color": color, "ls": ls, "lw": lw}


def plot_curves(out_dir: Path, state: int, freq: np.ndarray, candidates: tuple[np.ndarray, np.ndarray], rows: list[dict[str, object]]) -> Path:
    cand_fsc, cand_err = candidates
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.8), constrained_layout=True)
    fig.suptitle(f"IgG-1D no-reg oracle | state {state} | unfiltered only | moving focus mask", fontweight="bold")
    for idx in range(cand_fsc.shape[0]):
        color = cmap(idx / max(1, cand_fsc.shape[0] - 1))
        axes[0].plot(freq, cand_fsc[idx], color=color, alpha=0.22, lw=0.75)
        axes[1].semilogy(freq, np.maximum(cand_err[idx], 1e-30), color=color, alpha=0.22, lw=0.75)
    for row in rows:
        axes[0].plot(freq, row["fsc"], color=row["color"], ls=row["ls"], lw=row["lw"], label=row["label"])
        axes[1].semilogy(freq, np.maximum(row["err"], 1e-30), color=row["color"], ls=row["ls"], lw=row["lw"], label=row["label"])
    axes[0].axhline(0.5, color="0.45", ls=":", lw=1.0)
    axes[1].axhline(0.1, color="0.45", ls=":", lw=1.0)
    axes[0].set_title("FSC vs GT state")
    axes[1].set_title("Relative Fourier error vs GT state")
    axes[0].set_ylabel("masked FSC")
    axes[1].set_ylabel("masked relative error")
    axes[0].set_ylim(-0.08, 1.03)
    axes[1].set_ylim(1e-3, 1e3)
    for ax in axes:
        ax.set_xlim(0.0, FREQ_MAX)
        ax.set_xticks(np.arange(0, FREQ_MAX + 1e-6, 0.05))
        ax.set_xlabel("spatial frequency (1/A)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7.4, loc="best")
    path = out_dir / f"state{state:04d}_noreg_all50_fsc_error.png"
    fig.savefig(path, dpi=190)
    fig.savefig(out_dir / f"state{state:04d}_noreg_all50_fsc_error.pdf")
    plt.close(fig)
    return path


def plot_embedding(
    out_dir: Path,
    state: int,
    z: np.ndarray,
    target_z: np.ndarray,
    states: np.ndarray,
    nearest: dict[int, np.ndarray],
    raw_gt_by_state: np.ndarray | None,
) -> Path:
    rng = np.random.default_rng(20260528 + state)
    sample = rng.choice(z.shape[0], size=min(180_000, z.shape[0]), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.0), constrained_layout=True)
    for ax, (a, b) in zip(axes, [(0, 1), (2, 3)]):
        sc = ax.scatter(z[sample, a], z[sample, b], c=states[sample], cmap="turbo", s=1.1, alpha=0.22, rasterized=True)
        if 1000 in nearest:
            ax.scatter(z[nearest[1000], a], z[nearest[1000], b], s=9, facecolors="none", edgecolors="#1b9e77", linewidths=0.45, label="nearest 1000")
        if 100 in nearest:
            ax.scatter(z[nearest[100], a], z[nearest[100], b], s=35, facecolors="none", edgecolors="black", linewidths=1.05, label="nearest 100")
        ax.scatter([target_z[a]], [target_z[b]], s=130, c="red", marker="*", edgecolors="black", linewidths=0.7, label="target z")
        state_mean = z[states == state].mean(axis=0)
        ax.scatter([state_mean[a]], [state_mean[b]], s=85, c="white", marker="X", edgecolors="black", linewidths=0.9, label=f"mean z | state {state}")
        if raw_gt_by_state is not None:
            ax.scatter([raw_gt_by_state[state, a]], [raw_gt_by_state[state, b]], s=90, c="#ff7f00", marker="D", edgecolors="black", linewidths=0.7, label="raw GT-PC state")
        zoom = nearest[100] if 100 in nearest else next(iter(nearest.values()))
        xpad = max(1e-6, 0.22 * np.ptp(z[zoom, a]))
        ypad = max(1e-6, 0.22 * np.ptp(z[zoom, b]))
        ax.set_xlim(float(np.min(z[zoom, a]) - xpad), float(np.max(z[zoom, a]) + xpad))
        ax.set_ylim(float(np.min(z[zoom, b]) - ypad), float(np.max(z[zoom, b]) + ypad))
        ax.set_xlabel(f"z{a}")
        ax.set_ylabel(f"z{b}")
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=7.2, loc="best")
    cbar = fig.colorbar(sc, ax=axes, shrink=0.84, pad=0.01)
    cbar.set_label("GT state label")
    fig.suptitle(f"IgG no-reg embedding | state {state} | nearest controls", fontweight="bold")
    path = out_dir / f"state{state:04d}_embedding_nearest_controls.png"
    fig.savefig(path, dpi=190)
    fig.savefig(out_dir / f"state{state:04d}_embedding_nearest_controls.pdf")
    plt.close(fig)
    return path


def write_curves(path: Path, freq: np.ndarray, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["frequency_1_per_A"]
        for row in rows:
            key = str(row["label"]).replace(" ", "_").replace("/", "_")
            header.extend([f"{key}_fsc", f"{key}_relative_error"])
        writer.writerow(header)
        for shell, frequency in enumerate(freq):
            record = [float(frequency)]
            for row in rows:
                record.extend([float(np.asarray(row["fsc"])[shell]), float(np.asarray(row["err"])[shell])])
            writer.writerow(record)


def symlink_force(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def process_state(args: argparse.Namespace, state: int, raw_gt_by_state: np.ndarray | None) -> tuple[Path, Path, list[dict[str, object]]]:
    run_dir = args.run_dir
    compute_dir = run_dir / f"07_compute_state_oracle_noreg_state{state:04d}_lazy"
    out_dir = args.out_dir / f"state{state:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    info = parse_compute_state_inputs(compute_dir)
    if info.coords_entry != "latent_coords_noreg" or info.precision_entry != "latent_precision_noreg":
        raise RuntimeError(f"{compute_dir} is not a no-reg compute_state output: {info}")
    if info.zdim != ZDIM:
        raise RuntimeError(f"expected zdim {ZDIM}, got {info.zdim}")

    distances, _ = recompute_latent_distances_from_compute_state(compute_dir)
    saved = np.asarray(np.loadtxt(state_dir(compute_dir) / "heterogeneity_distances.txt"), dtype=np.float64).reshape(-1)
    finite = np.isfinite(distances) & np.isfinite(saved)
    distance_check = {
        "max_abs_distance_diff": float(np.max(np.abs(distances[finite] - saved[finite]))),
        "max_rel_distance_diff": float(np.max(np.abs(distances[finite] - saved[finite]) / np.maximum(1.0, np.abs(saved[finite])))),
        "saved_distance_path": str(state_dir(compute_dir) / "heterogeneity_distances.txt"),
    }

    z = np.asarray(np.load(info.pipeline / "model" / f"zdim_{ZDIM}" / "latent_coords_noreg.npy"), dtype=np.float64)[:, :ZDIM]
    states = np.asarray(np.load(run_dir / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    target_z = np.asarray(info.target, dtype=np.float64).reshape(ZDIM)

    mask = np.clip(load_mrc(args.mask), 0.0, 1.0)
    target_vol = load_mrc(run_dir / "04_ground_truth" / f"gt_vol{state:04d}.mrc")
    labels_flat, n_shells = shell_labels(target_vol.shape)
    target_ft = fft3_centered(target_vol * mask)
    target_power = np.bincount(labels_flat, weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)
    freq = np.arange(n_shells, dtype=np.float64) / (target_vol.shape[0] * VOXEL_SIZE_A)

    grid = candidate_grid(compute_dir)
    cand_fsc, cand_err = [], []
    cand_vols = []
    for idx, paths in enumerate(candidate_paths(compute_dir)):
        volume = average_candidate(paths)
        cand_vols.append(volume if idx == 0 else None)
        fsc, err = masked_metrics(volume, mask, labels_flat, n_shells, target_ft, target_power)
        cand_fsc.append(fsc)
        cand_err.append(err)
        if idx % 10 == 9 or idx == grid.size - 1:
            print(f"state {state}: candidate metrics {idx + 1}/{grid.size}", flush=True)
    cand_fsc = np.stack(cand_fsc)
    cand_err = np.stack(cand_err)
    fsc_res = np.asarray([fsc05_resolution(freq, curve) for curve in cand_fsc])
    err_res = np.asarray([relerr_resolution(freq, curve, 0.1) for curve in cand_err])
    best_fsc_idx = int(np.nanargmin(fsc_res))
    best_err_idx = int(np.nanargmin(err_res))

    final_fsc, final_err = masked_metrics(load_mrc(compute_dir / "state000_unfil.mrc"), mask, labels_flat, n_shells, target_ft, target_power)

    nearest: dict[int, np.ndarray] = {n: nearest_indices(distances, n) for n in NEAREST_COUNTS}
    rows = [
        curve_row("final shell oracle unfil", final_fsc, final_err, "black", "-", 2.7),
        curve_row(f"best FSC0.5 candidate #{best_fsc_idx + 1}", cand_fsc[best_fsc_idx], cand_err[best_fsc_idx], "#1f77b4", "--", 2.35),
        curve_row(f"best relerr<0.1 candidate #{best_err_idx + 1}", cand_fsc[best_err_idx], cand_err[best_err_idx], "#2ca02c", "-.", 2.25),
    ]

    gt_info: dict[str, object] = {}
    for n, color in zip(NEAREST_COUNTS, ("#d95f02", "#1b9e77")):
        weights, counts = state_weights_from_indices(run_dir, nearest[n])
        vol = weighted_gt(run_dir, weights)
        write_mrc(out_dir / f"nearest{n}_gt_mix.mrc", vol)
        fsc, err = masked_metrics(vol, mask, labels_flat, n_shells, target_ft, target_power)
        rows.append(curve_row(f"nearest {n} GT mix", fsc, err, color, ":", 2.55 if n == 100 else 2.15))
        top = " ".join(f"{int(s)}:{int(counts[s])}" for s in np.argsort(counts)[::-1][:12] if counts[s] > 0)
        gt_info[f"nearest{n}_top_states"] = top
        gt_info[f"nearest{n}_target_state_fraction"] = float(weights[state])
        gt_info[f"nearest{n}_radius"] = float(np.max(distances[nearest[n]]))

    best_h = float(grid[best_fsc_idx])
    support = support_indices(distances, best_h)
    if support.size:
        weights, counts = state_weights_from_indices(run_dir, support)
        vol = weighted_gt(run_dir, weights)
        write_mrc(out_dir / "best_fsc_support_gt_mix.mrc", vol)
        fsc, err = masked_metrics(vol, mask, labels_flat, n_shells, target_ft, target_power)
        rows.append(curve_row("GT mix over best h support", fsc, err, "#7570b3", (0, (3, 1, 1, 1)), 2.05))
        gt_info["best_support_count"] = int(support.size)
        gt_info["best_support_top_states"] = " ".join(f"{int(s)}:{int(counts[s])}" for s in np.argsort(counts)[::-1][:12] if counts[s] > 0)

        half_n = max(1, support.size // 2)
        half = support[np.argpartition(distances[support], half_n - 1)[:half_n]]
        weights, counts = state_weights_from_indices(run_dir, half)
        vol = weighted_gt(run_dir, weights)
        write_mrc(out_dir / "nearest_half_best_fsc_support_gt_mix.mrc", vol)
        fsc, err = masked_metrics(vol, mask, labels_flat, n_shells, target_ft, target_power)
        rows.append(curve_row("nearest half of best h support GT mix", fsc, err, "#e7298a", (0, (1, 1)), 2.05))
        gt_info["nearest_half_best_support_count"] = int(half.size)
        gt_info["nearest_half_best_support_top_states"] = " ".join(f"{int(s)}:{int(counts[s])}" for s in np.argsort(counts)[::-1][:12] if counts[s] > 0)

    for half_width, color in zip(GT_WINDOWS, ("#8c510a", "#bf812d", "#dfc27d", "#80cdc1")):
        lo = max(0, state - half_width)
        hi = min(N_STATES - 1, state + half_width)
        weights = np.zeros(N_STATES, dtype=np.float64)
        weights[lo : hi + 1] = 1.0
        vol = weighted_gt(run_dir, weights)
        write_mrc(out_dir / f"gt_avg_state{state:04d}_pm{half_width:02d}.mrc", vol)
        fsc, err = masked_metrics(vol, mask, labels_flat, n_shells, target_ft, target_power)
        label = f"GT state {state}" if half_width == 0 else f"GT avg state {state} +/- {half_width}"
        rows.append(curve_row(label, fsc, err, color, (0, (5, 2)), 1.85))

    uniform = weighted_gt(run_dir, np.ones(N_STATES))
    write_mrc(out_dir / "uniform_gt_mean.mrc", uniform)
    uniform_fsc, uniform_err = masked_metrics(uniform, mask, labels_flat, n_shells, target_ft, target_power)
    rows.append(curve_row("uniform GT mean", uniform_fsc, uniform_err, "0.35", (0, (2, 2)), 2.0))

    best_vol = average_candidate(candidate_paths(compute_dir)[best_fsc_idx])
    write_mrc(out_dir / f"best_fsc_candidate_{best_fsc_idx + 1:04d}_unfil.mrc", best_vol)
    write_mrc(out_dir / f"gt_state{state:04d}.mrc", target_vol)

    plot_path = plot_curves(out_dir, state, freq, (cand_fsc, cand_err), rows)
    embed_path = plot_embedding(out_dir, state, z, target_z, states, nearest, raw_gt_by_state)
    write_curves(out_dir / f"state{state:04d}_curves.csv", freq, rows)

    summary_rows: list[dict[str, object]] = []
    for row in rows:
        summary_rows.append(
            {
                "state": state,
                "curve": row["label"],
                "fsc05_resolution_A": fsc05_resolution(freq, np.asarray(row["fsc"])),
                "relerr10_resolution_A": relerr_resolution(freq, np.asarray(row["err"]), 0.1),
            }
        )
    summary_rows.append(
        {
            "state": state,
            "curve": "metadata",
            "fsc05_resolution_A": "",
            "relerr10_resolution_A": "",
            "best_fsc_candidate_1based": best_fsc_idx + 1,
            "best_fsc_h": best_h,
            "best_error_candidate_1based": best_err_idx + 1,
            "best_error_h": float(grid[best_err_idx]),
            "target_z": " ".join(f"{x:.8g}" for x in target_z),
            "mean_z_for_state": " ".join(f"{x:.8g}" for x in z[states == state].mean(axis=0)),
            "raw_gt_z_for_state": "" if raw_gt_by_state is None else " ".join(f"{x:.8g}" for x in raw_gt_by_state[state]),
            **distance_check,
            **gt_info,
        }
    )
    keys = sorted({key for row in summary_rows for key in row})
    with (out_dir / f"state{state:04d}_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    np.savez_compressed(
        out_dir / f"state{state:04d}_curves.npz",
        frequency_1_per_A=freq,
        candidate_grid=grid,
        candidate_fsc=cand_fsc,
        candidate_relative_error=cand_err,
        **{str(row["label"]).replace(" ", "_").replace("/", "_") + "_fsc": row["fsc"] for row in rows},
        **{str(row["label"]).replace(" ", "_").replace("/", "_") + "_relative_error": row["err"] for row in rows},
    )

    bundle = out_dir / "download_bundle"
    bundle.mkdir(exist_ok=True)
    symlink_force(compute_dir / "state000_unfil.mrc", bundle / f"compute_state_state{state:04d}_unfil.mrc")
    symlink_force(args.mask, bundle / "focus_mask_moving.mrc")
    for src in sorted(out_dir.glob("*.mrc")):
        symlink_force(src, bundle / src.name)
    for src in (plot_path, embed_path, out_dir / f"state{state:04d}_summary.csv", out_dir / f"state{state:04d}_curves.csv"):
        symlink_force(src, bundle / src.name)

    metadata = {
        "state": state,
        "run_dir": str(run_dir),
        "compute_dir": str(compute_dir),
        "pipeline_dir": str(info.pipeline),
        "mask": str(args.mask),
        "no_reg_entries": {"coords": info.coords_entry, "precision": info.precision_entry},
        "distance_definition": "cov_dist=(z-target)^T latent_precision_noreg (z-target), recomputed from current pipeline output",
        "distance_check": distance_check,
        "gt_controls": gt_info,
        "plot": str(plot_path),
        "embedding_plot": str(embed_path),
        "download_bundle": str(bundle),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"STATE {state} PLOT {plot_path}", flush=True)
    print(f"STATE {state} EMBEDDING {embed_path}", flush=True)
    print(f"STATE {state} BUNDLE {bundle}", flush=True)
    return plot_path, embed_path, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "/scratch/gpfs/CRYOEM/gilleslab/tmp/igg1d_first50_noise30_b80_1m_20260528/"
            "n01000000/runs/n01000000_seed0000"
        ),
    )
    parser.add_argument("--states", default="25,12,0")
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.mask = args.mask or (args.run_dir / "05_masks" / "focus_mask_moving.mrc")
    args.out_dir = args.out_dir or (args.run_dir / "08_noreg_state_controls_20260528")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    states = [int(x) for x in args.states.split(",") if x.strip()]

    raw_gt_by_state = None
    try:
        raw_gt_by_state = raw_gt_pc_coordinates(args.run_dir, args.run_dir / "06_pipeline", ZDIM)
    except Exception as exc:
        print(f"WARNING: could not compute raw GT-PC coordinates: {exc}", flush=True)

    all_rows: list[dict[str, object]] = []
    outputs = []
    for state in states:
        plot_path, embed_path, rows = process_state(args, state, raw_gt_by_state)
        outputs.append({"state": state, "plot": str(plot_path), "embedding": str(embed_path)})
        all_rows.extend(rows)

    keys = sorted({key for row in all_rows for key in row})
    with (args.out_dir / "all_states_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    (args.out_dir / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print(f"SUMMARY {args.out_dir / 'all_states_summary.csv'}")


if __name__ == "__main__":
    main()
