"""Generate the spike kernel-regression comparison report.

This command collects the post-processing diagnostics used for the synthetic
spike kernel-regression experiments. It compares one standard compute_state
run against one deconvolved compute_state run on the same target state and
ground-truth mask, then writes an ``08_*`` style report directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.heterogeneity.deconvolved_kernel_regression import (
    deconvolution_weights_1d_many,
    epanechnikov_deconvolution_kernel_1d,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpikeKernelReportConfig:
    standard_root: Path
    deconvolved_root: Path
    target_volume: Path
    mask: Path
    out_dir: Path
    pipeline_root: Path | None = None
    target_point: Path | None = None
    state_label: str = "state000"
    report_title: str = "spike kernel regression comparison"
    expected_candidates: int | None = None
    score_frequency_max: float = 0.35
    plot_frequency_max: float = 0.40
    selected_count: int = 10
    package_masked_unfiltered: bool = True
    package_lowpass: bool = False
    lowpass_cutoff: float = 0.15


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _state_dir(root: Path, state_label: str) -> Path:
    root = Path(root)
    if (root / "diagnostics" / state_label / "params.pkl").exists():
        return root / "diagnostics" / state_label
    if (root / "params.pkl").exists():
        return root
    raise FileNotFoundError(f"Could not find {state_label}/params.pkl below {root}")


def _candidate_paths(root: Path, state_label: str, subdir: str, expected: int | None = None) -> list[Path]:
    state_dir = _state_dir(root, state_label)
    paths = sorted((state_dir / subdir).glob("*.mrc"))
    if expected is not None and len(paths) != expected:
        raise RuntimeError(f"Expected {expected} candidate volumes in {state_dir / subdir}, found {len(paths)}")
    if not paths:
        raise RuntimeError(f"No candidate volumes found in {state_dir / subdir}")
    return paths


def _candidate_grid(root: Path, state_label: str) -> np.ndarray:
    params = utils.pickle_load(_state_dir(root, state_label) / "params.pkl")
    if "lambda_grid" in params:
        return np.asarray(params["lambda_grid"], dtype=np.float64)
    return np.asarray(params["heterogeneity_bins"], dtype=np.float64)


def _voxel_size(root: Path, state_label: str) -> float:
    params = utils.pickle_load(_state_dir(root, state_label) / "params.pkl")
    return float(params["voxel_size"])


def _numpy_dft3(volume: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(volume)))


def _inverse_numpy_dft3(volume_ft: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(volume_ft))).real


def _shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    labels = np.asarray(ftu.get_grid_of_radial_distances(shape, rounded=True), dtype=np.int32)
    n_shells = shape[0] // 2 - 1
    labels = np.clip(labels, 0, n_shells - 1)
    return labels, n_shells


def _masked_metrics_for_volume(
    volume: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    n_shells: int,
    target_ft: np.ndarray,
    target_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ft = _numpy_dft3(np.asarray(volume, dtype=np.float32) * mask)
    top = np.bincount(labels.ravel(), weights=np.real(np.conj(ft).ravel() * target_ft.ravel()), minlength=n_shells)
    bot1 = np.bincount(labels.ravel(), weights=np.abs(ft).ravel() ** 2, minlength=n_shells)
    bot2 = np.bincount(labels.ravel(), weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = top / np.sqrt(bot1 * bot2)
    fsc[~np.isfinite(fsc)] = 0.0
    if fsc.size > 1:
        fsc[0] = fsc[1]

    diff_ft = _numpy_dft3((np.asarray(volume, dtype=np.float32) - target) * mask)
    err = np.bincount(labels.ravel(), weights=np.abs(diff_ft).ravel() ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = err / np.maximum(target_power, 1e-30)
    rel_err[~np.isfinite(rel_err)] = np.inf
    return fsc, rel_err


def _mean_volume_from_stack(path: Path, target_shape: tuple[int, int, int]) -> np.ndarray:
    stack = np.load(path, mmap_mode="r")
    if stack.ndim != 4 or tuple(stack.shape[1:]) != target_shape:
        raise ValueError(f"Expected {path} to have shape (n, {target_shape}), got {stack.shape}")

    acc = np.zeros(target_shape, dtype=np.float64)
    for idx in range(stack.shape[0]):
        acc += np.asarray(stack[idx], dtype=np.float64)
    return (acc / float(stack.shape[0])).astype(np.float32)


def _load_distribution_mean_volume(cfg: SpikeKernelReportConfig, target_shape: tuple[int, int, int]) -> tuple[np.ndarray, Path, str] | None:
    stack_path = cfg.target_volume.parent / "gt_volumes_used_by_simulator.npy"
    if stack_path.exists():
        return _mean_volume_from_stack(stack_path, target_shape), stack_path, "mean of gt_volumes_used_by_simulator.npy"

    if cfg.pipeline_root is not None:
        mean_path = cfg.pipeline_root / "output" / "volumes" / "mean.mrc"
        if mean_path.exists():
            mean_volume = np.asarray(utils.load_mrc(mean_path), dtype=np.float32)
            if mean_volume.shape != target_shape:
                raise ValueError(f"Expected {mean_path} to have shape {target_shape}, got {mean_volume.shape}")
            return mean_volume, mean_path, "pipeline output mean.mrc"

    logger.warning("Could not find a distribution mean volume for FSC/error reference plot.")
    return None


def _all_masked_metrics(
    root: Path,
    state_label: str,
    target: np.ndarray,
    mask: np.ndarray,
    expected: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    labels, n_shells = _shell_labels(target.shape)
    target_ft = _numpy_dft3(target * mask)
    target_power = np.bincount(labels.ravel(), weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)

    fscs = []
    errors = []
    for path in _candidate_paths(root, state_label, "estimates_filt", expected):
        volume = np.asarray(utils.load_mrc(path), dtype=np.float32)
        fsc, error = _masked_metrics_for_volume(volume, target, mask, labels, n_shells, target_ft, target_power)
        fscs.append(fsc)
        errors.append(error)
    return np.asarray(fscs), np.asarray(errors)


def _shell_oracle_volume(
    choices: np.ndarray,
    roots: list[Path],
    state_label: str,
    out_path: Path,
    voxel_size: float,
) -> np.ndarray:
    first = np.asarray(utils.load_mrc(_candidate_paths(roots[0], state_label, "estimates_filt")[0]), dtype=np.float32)
    labels, _ = _shell_labels(first.shape)
    oracle_ft = np.zeros(first.shape, dtype=np.complex128)
    flat_labels = labels.ravel()

    offset = 0
    for root in roots:
        paths = _candidate_paths(root, state_label, "estimates_filt")
        for local_idx, path in enumerate(paths):
            global_idx = offset + local_idx
            if not np.any(choices == global_idx):
                continue
            ft = _numpy_dft3(np.asarray(utils.load_mrc(path), dtype=np.float32))
            selected = choices[flat_labels] == global_idx
            oracle_ft.ravel()[selected] = ft.ravel()[selected]
        offset += len(paths)

    oracle = _inverse_numpy_dft3(oracle_ft).astype(np.float32)
    utils.write_mrc(str(out_path), oracle, voxel_size=voxel_size)
    return oracle


def _unfiltered_candidate_count(state_dir: Path) -> int:
    half1 = sorted((state_dir / "estimates_half1_unfil").glob("*.mrc"))
    half2 = sorted((state_dir / "estimates_half2_unfil").glob("*.mrc"))
    if not half1:
        raise RuntimeError(f"No unfiltered half-map candidates found in {state_dir}")
    if len(half1) != len(half2):
        raise RuntimeError(
            f"Mismatched unfiltered half-map candidates in {state_dir}: "
            f"{len(half1)} half1 vs {len(half2)} half2"
        )
    return len(half1)


def _shell_oracle_unfiltered_half_average(choices: np.ndarray, state_dirs: list[Path]) -> np.ndarray:
    first, _ = _load_half_average(state_dirs[0], 0)
    labels, _ = _shell_labels(first.shape)
    oracle_ft = np.zeros(first.shape, dtype=np.complex128)
    flat_labels = labels.ravel()

    offset = 0
    for state_dir in state_dirs:
        n_candidates = _unfiltered_candidate_count(state_dir)
        for local_idx in range(n_candidates):
            global_idx = offset + local_idx
            if not np.any(choices == global_idx):
                continue
            volume, _ = _load_half_average(state_dir, local_idx)
            selected = choices[flat_labels] == global_idx
            oracle_ft.ravel()[selected] = _numpy_dft3(volume).ravel()[selected]
        offset += n_candidates

    return _inverse_numpy_dft3(oracle_ft).astype(np.float32)


def _top_candidate_indices(scores: np.ndarray, best: int, count: int = 5) -> list[int]:
    finite_scores = np.where(np.isfinite(scores), scores, np.inf)
    indices = list(np.argsort(finite_scores)[:count])
    if best not in indices:
        indices.insert(0, best)
    return sorted(set(int(idx) for idx in indices))


def _select_interesting(error: np.ndarray, freq: np.ndarray, max_lines: int, score_frequency_max: float) -> tuple[np.ndarray, np.ndarray]:
    shell_mask = (freq > 0.0) & (freq <= score_frequency_max)
    score = np.nanmedian(error[:, shell_mask], axis=1)
    shell_choice = np.argmin(error[:, shell_mask], axis=0)
    winners, counts = np.unique(shell_choice, return_counts=True)

    selected = set(np.argsort(score)[:5].tolist())
    for idx in winners[np.argsort(-counts)[:8]]:
        selected.add(int(idx))

    score_rank = np.empty_like(score, dtype=np.int64)
    score_rank[np.argsort(score)] = np.arange(score.size)
    count_by_idx = np.zeros(score.size, dtype=np.int64)
    count_by_idx[winners] = counts
    ordered = sorted(selected, key=lambda idx: (score_rank[idx], -count_by_idx[idx], idx))
    return np.asarray(ordered[:max_lines], dtype=np.int32), score


def _standard_kernel(diff: np.ndarray, bin_value: float, precision_ref: float) -> np.ndarray:
    if bin_value <= 0 or precision_ref <= 0:
        return np.zeros_like(diff, dtype=np.float64)
    u = np.sqrt(np.maximum(precision_ref * diff**2 / (2.0 * bin_value), 0.0))
    return np.where(np.abs(u) < 1.0, 0.75 * (1.0 - u**2), 0.0)


def _deconv_kernel(diff: np.ndarray, lambda_value: float, sigma_ref: float) -> np.ndarray:
    h = float(lambda_value) * float(sigma_ref)
    return epanechnikov_deconvolution_kernel_1d(diff / h, lambda_value)


def _plot_selected(
    ax: plt.Axes,
    freq: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    selected: np.ndarray,
    score: np.ndarray,
    label_prefix: str,
    colors: list,
    linestyle: str,
    best_color: str,
    *,
    logy: bool = False,
) -> None:
    best = int(np.nanargmin(score))
    for rank, idx in enumerate(selected):
        color = colors[rank % len(colors)]
        linewidth = 1.5
        alpha = 0.78
        label = f"{label_prefix} #{idx + 1}: {grid[idx]:.3g}"
        if idx == best:
            linewidth = 3.0
            alpha = 1.0
            color = best_color
            label = f"{label_prefix} best #{idx + 1}: {grid[idx]:.3g}"
        y = np.maximum(values[idx], 1e-30) if logy else values[idx]
        if logy:
            ax.semilogy(freq, y, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)
        else:
            ax.plot(freq, y, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)


def _safe_token(value: float) -> str:
    return f"{value:.3g}".replace(".", "p").replace("-", "m")


def _spherical_lowpass(volume: np.ndarray, voxel_size: float, cutoff_frequency: float) -> np.ndarray:
    n = int(volume.shape[0])
    if volume.shape != (n, n, n):
        raise ValueError(f"Expected cubic volume, got {volume.shape}")
    freq = np.fft.fftfreq(n, d=voxel_size)
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
    keep = (kx * kx + ky * ky + kz * kz) <= cutoff_frequency * cutoff_frequency
    ft = np.fft.fftn(volume)
    ft *= keep
    return np.fft.ifftn(ft).real.astype(np.float32)


def _load_half_average(state_dir: Path, idx: int) -> tuple[np.ndarray, dict[str, str]]:
    filename = f"{idx:04d}.mrc"
    half1 = state_dir / "estimates_half1_unfil" / filename
    half2 = state_dir / "estimates_half2_unfil" / filename
    if not half1.exists() or not half2.exists():
        raise FileNotFoundError(f"Missing half-map pair for {state_dir} index {idx}")
    volume = 0.5 * (np.asarray(utils.load_mrc(half1), dtype=np.float32) + np.asarray(utils.load_mrc(half2), dtype=np.float32))
    return volume, {"half1_unfil": str(half1), "half2_unfil": str(half2)}


def _write_lowpass_candidate(
    method: str,
    state_dir: Path,
    idx: int,
    parameter_name: str,
    parameter_value: float,
    voxel_size: float,
    cutoff_frequency: float,
    out_dir: Path,
) -> dict[str, object]:
    volume, inputs = _load_half_average(state_dir, idx)
    processed = _spherical_lowpass(volume, voxel_size, cutoff_frequency)
    out_name = (
        f"{method}_{idx:04d}_{parameter_name}{_safe_token(parameter_value)}_"
        f"halfavg_unfil_lp{_safe_token(cutoff_frequency)}.mrc"
    )
    out_path = out_dir / out_name
    utils.write_mrc(str(out_path), processed, voxel_size=voxel_size)
    return {
        "method": method,
        "index_0based": int(idx),
        parameter_name: float(parameter_value),
        "input_half_maps": inputs,
        "output": str(out_path),
    }


def _write_masked_unfiltered_candidate(
    method: str,
    state_dir: Path,
    idx: int,
    parameter_name: str,
    parameter_value: float,
    mask: np.ndarray,
    voxel_size: float,
    out_dir: Path,
) -> dict[str, object]:
    volume, inputs = _load_half_average(state_dir, idx)
    out_name = (
        f"{method}_{idx:04d}_{parameter_name}{_safe_token(parameter_value)}_"
        "halfavg_unfil_masked.mrc"
    )
    out_path = out_dir / out_name
    utils.write_mrc(str(out_path), volume * mask, voxel_size=voxel_size)
    return {
        "method": method,
        "index_0based": int(idx),
        parameter_name: float(parameter_value),
        "input_half_maps": inputs,
        "output": str(out_path),
    }


def _candidate_shells(choices: np.ndarray, idx: int, freq: np.ndarray) -> dict[str, object]:
    shells = np.flatnonzero(choices == idx)
    return {
        "selected_shell_indices": [int(shell) for shell in shells],
        "selected_freq_1_per_A": [float(value) for value in freq[shells]],
    }


def _package_masked_unfiltered_oracle_volumes(
    cfg: SpikeKernelReportConfig,
    all_report: dict,
    target: np.ndarray,
    mask: np.ndarray,
    standard_grid: np.ndarray,
    deconv_grid: np.ndarray,
    voxel_size: float,
) -> dict:
    out_dir = cfg.out_dir / "selected_masked_unfil_oracle_shell"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    metrics = np.load(all_report["metrics_npz"])
    freq = metrics["freq"]
    standard_choice = metrics["standard_oracle_choice"].astype(np.int64)
    deconv_choice = metrics["deconv_oracle_choice"].astype(np.int64)
    combined_choice = metrics["combined_oracle_choice"].astype(np.int64)

    standard_state_dir = _state_dir(cfg.standard_root, cfg.state_label)
    deconv_state_dir = _state_dir(cfg.deconvolved_root, cfg.state_label)
    manifest: dict[str, object] = {
        "description": (
            "Masked, unfiltered volume package. Candidate volumes are half1/half2 "
            "averages from estimates_half*_unfil multiplied by the report mask. "
            "Shell-oracle composites are assembled in Fourier shells from those "
            "unfiltered half averages, then multiplied by the same mask. No "
            "low-pass filtering and no downsampling are applied."
        ),
        "selection_rule": "true-GT per-frequency-shell oracle choices from the report metrics",
        "standard_root": str(standard_state_dir),
        "deconvolved_root": str(deconv_state_dir),
        "target_volume": str(cfg.target_volume),
        "mask": str(cfg.mask),
        "shape": list(target.shape),
        "voxel_size_A": voxel_size,
        "standard_oracle_candidate_indices_0based": [int(idx) for idx in np.unique(standard_choice)],
        "deconvolved_oracle_candidate_indices_0based": [int(idx) for idx in np.unique(deconv_choice)],
        "combined_choice_encoding": "0..n_standard-1 standard, n_standard..n_standard+n_deconvolved-1 deconvolved",
        "outputs": [],
        "candidates": [],
    }

    def write_output(name: str, volume: np.ndarray, role: str, **metadata: object) -> str:
        path = out_dir / name
        utils.write_mrc(str(path), np.asarray(volume, dtype=np.float32), voxel_size=voxel_size)
        manifest["outputs"].append({"role": role, "output": str(path), **metadata})
        return str(path)

    write_output("gt_masked_unfil.mrc", target * mask, "masked_ground_truth_target")
    write_output("mask.mrc", mask, "mask_applied_to_all_volumes")

    write_output(
        "standard_shell_oracle_halfavg_unfil_masked.mrc",
        _shell_oracle_unfiltered_half_average(standard_choice, [standard_state_dir]) * mask,
        "standard_shell_oracle_from_unfiltered_half_averages",
    )
    write_output(
        "deconvolved_shell_oracle_halfavg_unfil_masked.mrc",
        _shell_oracle_unfiltered_half_average(deconv_choice, [deconv_state_dir]) * mask,
        "deconvolved_shell_oracle_from_unfiltered_half_averages",
    )
    write_output(
        "combined_shell_oracle_halfavg_unfil_masked.mrc",
        _shell_oracle_unfiltered_half_average(combined_choice, [standard_state_dir, deconv_state_dir]) * mask,
        "combined_shell_oracle_from_unfiltered_half_averages",
    )

    for idx in np.unique(standard_choice):
        item = _write_masked_unfiltered_candidate(
            "standard",
            standard_state_dir,
            int(idx),
            "h",
            float(standard_grid[idx]),
            mask,
            voxel_size,
            out_dir,
        )
        item.update(_candidate_shells(standard_choice, int(idx), freq))
        manifest["candidates"].append(item)

    for idx in np.unique(deconv_choice):
        item = _write_masked_unfiltered_candidate(
            "deconvolved",
            deconv_state_dir,
            int(idx),
            "lambda",
            float(deconv_grid[idx]),
            mask,
            voxel_size,
            out_dir,
        )
        item.update(_candidate_shells(deconv_choice, int(idx), freq))
        manifest["candidates"].append(item)

    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    tar_path = out_dir.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    manifest["tarball"] = str(tar_path)
    _write_json(manifest_path, manifest)
    return {"directory": str(out_dir), "tarball": str(tar_path), "manifest": str(manifest_path)}


def _save_all_candidate_report(
    cfg: SpikeKernelReportConfig,
    target: np.ndarray,
    mask: np.ndarray,
    standard_grid: np.ndarray,
    deconv_grid: np.ndarray,
    standard_fsc: np.ndarray,
    deconv_fsc: np.ndarray,
    standard_err: np.ndarray,
    deconv_err: np.ndarray,
    freq: np.ndarray,
    voxel_size: float,
    plots_dir: Path,
    oracle_dir: Path,
) -> dict:
    shell_mask = (freq > 0) & (freq <= cfg.score_frequency_max)
    standard_score = np.nanmedian(standard_err[:, shell_mask], axis=1)
    deconv_score = np.nanmedian(deconv_err[:, shell_mask], axis=1)
    standard_best = int(np.nanargmin(standard_score))
    deconv_best = int(np.nanargmin(deconv_score))
    standard_top = _top_candidate_indices(standard_score, standard_best)
    deconv_top = _top_candidate_indices(deconv_score, deconv_best)

    standard_choice = np.argmin(standard_err, axis=0)
    deconv_choice = np.argmin(deconv_err, axis=0)
    combined_err = np.vstack([standard_err, deconv_err])
    combined_choice = np.argmin(combined_err, axis=0)

    labels, n_shells = _shell_labels(target.shape)
    target_ft = _numpy_dft3(target * mask)
    target_power = np.bincount(labels.ravel(), weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)

    standard_oracle = _shell_oracle_volume(
        standard_choice,
        [cfg.standard_root],
        cfg.state_label,
        oracle_dir / "standard_shell_oracle_estimator.mrc",
        voxel_size,
    )
    deconv_oracle = _shell_oracle_volume(
        deconv_choice,
        [cfg.deconvolved_root],
        cfg.state_label,
        oracle_dir / "deconvolved_shell_oracle_estimator.mrc",
        voxel_size,
    )
    combined_oracle = _shell_oracle_volume(
        combined_choice,
        [cfg.standard_root, cfg.deconvolved_root],
        cfg.state_label,
        oracle_dir / "combined_standard_deconvolved_shell_oracle_estimator.mrc",
        voxel_size,
    )
    standard_oracle_fsc, standard_oracle_err = _masked_metrics_for_volume(
        standard_oracle, target, mask, labels, n_shells, target_ft, target_power
    )
    deconv_oracle_fsc, deconv_oracle_err = _masked_metrics_for_volume(
        deconv_oracle, target, mask, labels, n_shells, target_ft, target_power
    )
    combined_oracle_fsc, combined_oracle_err = _masked_metrics_for_volume(
        combined_oracle, target, mask, labels, n_shells, target_ft, target_power
    )
    distribution_mean = _load_distribution_mean_volume(cfg, target.shape)
    if distribution_mean is not None:
        distribution_mean_volume, distribution_mean_path, distribution_mean_source = distribution_mean
        distribution_mean_fsc, distribution_mean_err = _masked_metrics_for_volume(
            distribution_mean_volume, target, mask, labels, n_shells, target_ft, target_power
        )
        distribution_mean_score = float(np.nanmedian(distribution_mean_err[shell_mask]))
    else:
        distribution_mean_path = None
        distribution_mean_source = None
        distribution_mean_fsc = None
        distribution_mean_err = None
        distribution_mean_score = None

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True, sharex=True)
    fig.suptitle(f"All candidates | standard vs deconvolved | true GT metrics | {cfg.report_title}", fontsize=15, fontweight="bold")

    std_norm = mcolors.LogNorm(vmin=max(float(np.nanmin(standard_grid[standard_grid > 0])), 1e-12), vmax=float(np.nanmax(standard_grid)))
    dec_norm = mcolors.LogNorm(vmin=max(float(np.nanmin(deconv_grid[deconv_grid > 0])), 1e-12), vmax=float(np.nanmax(deconv_grid)))
    std_cmap = plt.cm.turbo
    dec_cmap = plt.cm.nipy_spectral
    xticks = np.arange(0.0, cfg.plot_frequency_max + 1e-9, 0.05)
    error_ylim_max = float(
        np.nanpercentile(
            np.concatenate(
                [
                    standard_err[standard_top][:, shell_mask].ravel(),
                    deconv_err[deconv_top][:, shell_mask].ravel(),
                    standard_oracle_err[shell_mask],
                    deconv_oracle_err[shell_mask],
                    combined_oracle_err[shell_mask],
                    distribution_mean_err[shell_mask] if distribution_mean_err is not None else np.asarray([], dtype=np.float64),
                ]
            ),
            99.0,
        )
    )
    error_ylim_max = max(error_ylim_max * 1.8, 1e-2)

    ax = axes[0, 0]
    for idx in range(standard_fsc.shape[0]):
        ax.plot(freq, standard_fsc[idx], color="0.80", linewidth=0.6, alpha=0.45)
    for idx in standard_top:
        ax.plot(freq, standard_fsc[idx], color=std_cmap(std_norm(standard_grid[idx])), linewidth=1.4, alpha=0.95)
    ax.plot(freq, standard_fsc[standard_best], color="black", linewidth=2.5, label=f"best median: #{standard_best + 1}, h={standard_grid[standard_best]:.3g}")
    ax.plot(freq, standard_oracle_fsc, color="magenta", linewidth=2.0, linestyle="-.", label="standard shell oracle")
    ax.plot(freq, combined_oracle_fsc, color="cyan", linewidth=2.0, linestyle=":", label="combined shell oracle")
    if distribution_mean_fsc is not None:
        ax.plot(freq, distribution_mean_fsc, color="forestgreen", linewidth=2.0, linestyle="--", label="distribution mean")
    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=0.9)
    ax.axhline(1 / 7, color="0.4", linestyle=":", linewidth=0.9)
    ax.set_ylabel("masked FSC vs GT")
    ax.set_title("Standard FSC vs GT")
    ax.set_ylim(-0.08, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[0, 1]
    for idx in range(deconv_fsc.shape[0]):
        ax.plot(freq, deconv_fsc[idx], color="0.80", linewidth=0.6, alpha=0.45)
    for idx in deconv_top:
        ax.plot(freq, deconv_fsc[idx], color=dec_cmap(dec_norm(deconv_grid[idx])), linewidth=1.4, alpha=0.95)
    ax.plot(freq, deconv_fsc[deconv_best], color="black", linewidth=2.5, label=f"best median: #{deconv_best + 1}, lambda={deconv_grid[deconv_best]:.3g}")
    ax.plot(freq, deconv_oracle_fsc, color="magenta", linewidth=2.0, linestyle="-.", label="deconv shell oracle")
    ax.plot(freq, combined_oracle_fsc, color="cyan", linewidth=2.0, linestyle=":", label="combined shell oracle")
    if distribution_mean_fsc is not None:
        ax.plot(freq, distribution_mean_fsc, color="forestgreen", linewidth=2.0, linestyle="--", label="distribution mean")
    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=0.9)
    ax.axhline(1 / 7, color="0.4", linestyle=":", linewidth=0.9)
    ax.set_title("Deconvolved FSC vs GT")
    ax.set_ylim(-0.08, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[1, 0]
    for idx in range(standard_err.shape[0]):
        ax.semilogy(freq, np.maximum(standard_err[idx], 1e-30), color="0.80", linewidth=0.6, alpha=0.45)
    for idx in standard_top:
        ax.semilogy(freq, np.maximum(standard_err[idx], 1e-30), color=std_cmap(std_norm(standard_grid[idx])), linewidth=1.4, alpha=0.95)
    ax.semilogy(freq, np.maximum(standard_err[standard_best], 1e-30), color="black", linewidth=2.5, label=f"best median: #{standard_best + 1}, h={standard_grid[standard_best]:.3g}")
    ax.semilogy(freq, np.maximum(standard_oracle_err, 1e-30), color="magenta", linewidth=2.0, linestyle="-.", label="standard shell oracle")
    ax.semilogy(freq, np.maximum(combined_oracle_err, 1e-30), color="cyan", linewidth=2.0, linestyle=":", label="combined shell oracle")
    if distribution_mean_err is not None:
        ax.semilogy(freq, np.maximum(distribution_mean_err, 1e-30), color="forestgreen", linewidth=2.0, linestyle="--", label="distribution mean")
    ax.set_ylim(1e-4, error_ylim_max)
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_ylabel("masked relative Fourier error vs GT")
    ax.set_title("Standard true error vs GT; gray candidates may clip above y-limit")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=7)

    ax = axes[1, 1]
    for idx in range(deconv_err.shape[0]):
        ax.semilogy(freq, np.maximum(deconv_err[idx], 1e-30), color="0.80", linewidth=0.6, alpha=0.45)
    for idx in deconv_top:
        ax.semilogy(freq, np.maximum(deconv_err[idx], 1e-30), color=dec_cmap(dec_norm(deconv_grid[idx])), linewidth=1.4, alpha=0.95)
    ax.semilogy(freq, np.maximum(deconv_err[deconv_best], 1e-30), color="black", linewidth=2.5, label=f"best median: #{deconv_best + 1}, lambda={deconv_grid[deconv_best]:.3g}")
    ax.semilogy(freq, np.maximum(deconv_oracle_err, 1e-30), color="magenta", linewidth=2.0, linestyle="-.", label="deconv shell oracle")
    ax.semilogy(freq, np.maximum(combined_oracle_err, 1e-30), color="cyan", linewidth=2.0, linestyle=":", label="combined shell oracle")
    if distribution_mean_err is not None:
        ax.semilogy(freq, np.maximum(distribution_mean_err, 1e-30), color="forestgreen", linewidth=2.0, linestyle="--", label="distribution mean")
    ax.set_ylim(1e-4, error_ylim_max)
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_title("Deconvolved true error vs GT; gray candidates may clip above y-limit")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=7)

    for ax in axes.ravel():
        ax.set_xlim(0.0, cfg.plot_frequency_max)
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", which="both", labelbottom=True)

    sm_std = plt.cm.ScalarMappable(norm=std_norm, cmap=std_cmap)
    sm_std.set_array([])
    fig.colorbar(sm_std, ax=[axes[0, 0], axes[1, 0]], shrink=0.82, pad=0.01).set_label("standard h")
    sm_dec = plt.cm.ScalarMappable(norm=dec_norm, cmap=dec_cmap)
    sm_dec.set_array([])
    fig.colorbar(sm_dec, ax=[axes[0, 1], axes[1, 1]], shrink=0.82, pad=0.01).set_label("deconvolved lambda")

    proxy = [
        Line2D([0], [0], color="0.80", lw=1.0, label="all candidates"),
        Line2D([0], [0], color="black", lw=2.5, label="best median true error"),
        Line2D([0], [0], color="magenta", lw=2.0, linestyle="-.", label="method shell oracle"),
        Line2D([0], [0], color="cyan", lw=2.0, linestyle=":", label="combined shell oracle"),
        Line2D([0], [0], color="0.4", lw=0.9, linestyle="--", label="FSC 0.5"),
        Line2D([0], [0], color="0.4", lw=0.9, linestyle=":", label="FSC 1/7"),
    ]
    if distribution_mean_fsc is not None:
        proxy.insert(4, Line2D([0], [0], color="forestgreen", lw=2.0, linestyle="--", label="distribution mean"))
    fig.legend(handles=proxy, loc="outside lower center", ncol=6, frameon=True)

    png = plots_dir / "all_candidates_true_gt_fsc_error.png"
    pdf = plots_dir / "all_candidates_true_gt_fsc_error.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)

    metrics_npz = plots_dir / "all_candidates_true_gt_fsc_error.npz"
    np.savez_compressed(
        metrics_npz,
        freq=freq,
        standard_grid=standard_grid,
        deconv_grid=deconv_grid,
        standard_fsc=standard_fsc,
        deconv_fsc=deconv_fsc,
        standard_gt_errors=standard_err,
        deconv_gt_errors=deconv_err,
        standard_score=standard_score,
        deconv_score=deconv_score,
        standard_top_indices=np.asarray(standard_top, dtype=np.int32),
        deconv_top_indices=np.asarray(deconv_top, dtype=np.int32),
        standard_oracle_fsc=standard_oracle_fsc,
        standard_oracle_gt_errors=standard_oracle_err,
        deconv_oracle_fsc=deconv_oracle_fsc,
        deconv_oracle_gt_errors=deconv_oracle_err,
        combined_oracle_fsc=combined_oracle_fsc,
        combined_oracle_gt_errors=combined_oracle_err,
        standard_oracle_choice=standard_choice,
        deconv_oracle_choice=deconv_choice,
        combined_oracle_choice=combined_choice,
        distribution_mean_fsc=np.asarray(distribution_mean_fsc if distribution_mean_fsc is not None else [], dtype=np.float64),
        distribution_mean_gt_errors=np.asarray(distribution_mean_err if distribution_mean_err is not None else [], dtype=np.float64),
        distribution_mean_path=str(distribution_mean_path) if distribution_mean_path is not None else "",
        distribution_mean_source=str(distribution_mean_source) if distribution_mean_source is not None else "",
    )
    np.savez_compressed(
        oracle_dir / "shell_oracle_choices_and_metrics.npz",
        freq=freq,
        standard_grid=standard_grid,
        deconv_grid=deconv_grid,
        standard_oracle_choice=standard_choice,
        deconv_oracle_choice=deconv_choice,
        combined_oracle_choice=combined_choice,
        standard_oracle_fsc=standard_oracle_fsc,
        standard_oracle_gt_errors=standard_oracle_err,
        deconv_oracle_fsc=deconv_oracle_fsc,
        deconv_oracle_gt_errors=deconv_oracle_err,
        combined_oracle_fsc=combined_oracle_fsc,
        combined_oracle_gt_errors=combined_oracle_err,
        distribution_mean_fsc=np.asarray(distribution_mean_fsc if distribution_mean_fsc is not None else [], dtype=np.float64),
        distribution_mean_gt_errors=np.asarray(distribution_mean_err if distribution_mean_err is not None else [], dtype=np.float64),
        distribution_mean_path=str(distribution_mean_path) if distribution_mean_path is not None else "",
        distribution_mean_source=str(distribution_mean_source) if distribution_mean_source is not None else "",
    )

    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    fig2.suptitle("Best-candidate scores and shell oracle choices", fontsize=14, fontweight="bold")
    axes2[0].loglog(standard_grid, standard_score, marker="o", markersize=3.5, label="standard median true error")
    axes2[0].loglog(deconv_grid, deconv_score, marker="s", markersize=3.5, label="deconv median true error")
    axes2[0].axvline(standard_grid[standard_best], color="C0", linestyle="--", label=f"best standard h={standard_grid[standard_best]:.3g}")
    axes2[0].axvline(deconv_grid[deconv_best], color="C1", linestyle="--", label=f"best deconv lambda={deconv_grid[deconv_best]:.3g}")
    if distribution_mean_score is not None:
        axes2[0].axhline(distribution_mean_score, color="forestgreen", linestyle=":", label="distribution mean")
    axes2[0].set_xlabel("bandwidth parameter")
    axes2[0].set_ylabel(f"median relative Fourier error vs GT, 0 < freq <= {cfg.score_frequency_max:g} 1/A")
    axes2[0].grid(True, which="both", alpha=0.25)
    axes2[0].legend(fontsize=8)

    combined_is_deconv = combined_choice >= standard_grid.size
    combined_value = np.empty_like(freq)
    combined_value[combined_is_deconv] = deconv_grid[combined_choice[combined_is_deconv] - standard_grid.size]
    combined_value[~combined_is_deconv] = standard_grid[combined_choice[~combined_is_deconv]]
    axes2[1].plot(freq, standard_grid[standard_choice], color="C0", linewidth=1.5, label="standard oracle h")
    axes2[1].plot(freq, deconv_grid[deconv_choice], color="C1", linewidth=1.5, label="deconv oracle lambda")
    axes2[1].plot(freq, combined_value, color="black", linewidth=1.2, label="combined oracle parameter")
    axes2[1].scatter(freq[combined_is_deconv], combined_value[combined_is_deconv], s=10, color="C1", label="combined chose deconv")
    axes2[1].scatter(freq[~combined_is_deconv], combined_value[~combined_is_deconv], s=10, color="C0", label="combined chose standard")
    axes2[1].set_xlim(0, cfg.plot_frequency_max)
    axes2[1].set_xticks(xticks)
    axes2[1].set_yscale("log")
    axes2[1].set_xlabel("spatial frequency (1/A)")
    axes2[1].set_ylabel("oracle-selected parameter")
    axes2[1].grid(True, which="both", alpha=0.25)
    axes2[1].legend(fontsize=8)

    score_png = plots_dir / "best_scores_and_oracle_choices.png"
    score_pdf = plots_dir / "best_scores_and_oracle_choices.pdf"
    fig2.savefig(score_png, dpi=180)
    fig2.savefig(score_pdf)
    plt.close(fig2)

    return {
        "metrics_npz": str(metrics_npz),
        "all_candidates_png": str(png),
        "all_candidates_pdf": str(pdf),
        "score_png": str(score_png),
        "score_pdf": str(score_pdf),
        "standard_best_index_0based": standard_best,
        "standard_best_h": float(standard_grid[standard_best]),
        "standard_best_score": float(standard_score[standard_best]),
        "deconv_best_index_0based": deconv_best,
        "deconv_best_lambda": float(deconv_grid[deconv_best]),
        "deconv_best_score": float(deconv_score[deconv_best]),
        "standard_top_indices_0based": [int(idx) for idx in standard_top],
        "deconv_top_indices_0based": [int(idx) for idx in deconv_top],
        "oracle_outputs": {
            "standard": str(oracle_dir / "standard_shell_oracle_estimator.mrc"),
            "deconvolved": str(oracle_dir / "deconvolved_shell_oracle_estimator.mrc"),
            "combined": str(oracle_dir / "combined_standard_deconvolved_shell_oracle_estimator.mrc"),
            "choices_npz": str(oracle_dir / "shell_oracle_choices_and_metrics.npz"),
        },
        "distribution_mean": {
            "path": str(distribution_mean_path) if distribution_mean_path is not None else None,
            "source": distribution_mean_source,
            "score": distribution_mean_score,
        },
    }


def _save_selected_overlay(
    cfg: SpikeKernelReportConfig,
    metrics_npz: Path,
    plots_dir: Path,
) -> dict:
    data = np.load(metrics_npz)
    freq = data["freq"]
    standard_grid = data["standard_grid"]
    deconv_grid = data["deconv_grid"]
    standard_fsc = data["standard_fsc"]
    deconv_fsc = data["deconv_fsc"]
    standard_err = data["standard_gt_errors"]
    deconv_err = data["deconv_gt_errors"]

    standard_selected, standard_score = _select_interesting(
        standard_err, freq, cfg.selected_count, cfg.score_frequency_max
    )
    deconv_selected, deconv_score = _select_interesting(deconv_err, freq, cfg.selected_count, cfg.score_frequency_max)
    shell_mask = (freq > 0.0) & (freq <= cfg.score_frequency_max)

    oracle = {}
    for name in ["standard_oracle", "deconv_oracle", "combined_oracle"]:
        fsc_key = f"{name}_fsc"
        err_key = f"{name}_gt_errors"
        if fsc_key in data.files and err_key in data.files:
            oracle[name] = (data[fsc_key], data[err_key])
    distribution_mean = None
    if "distribution_mean_fsc" in data.files and "distribution_mean_gt_errors" in data.files:
        mean_fsc = data["distribution_mean_fsc"]
        mean_err = data["distribution_mean_gt_errors"]
        if mean_fsc.shape == freq.shape and mean_err.shape == freq.shape:
            distribution_mean = (mean_fsc, mean_err)

    fig, axes = plt.subplots(2, 1, figsize=(15, 11), constrained_layout=True)
    fig.suptitle(f"Selected candidates on the same axes | true GT metrics | {cfg.report_title}", fontsize=15, fontweight="bold")

    std_colors = list(plt.cm.Blues(np.linspace(0.40, 0.90, max(standard_selected.size, 2))))
    dec_colors = list(plt.cm.Oranges(np.linspace(0.40, 0.95, max(deconv_selected.size, 2))))

    ax = axes[0]
    _plot_selected(ax, freq, standard_fsc, standard_grid, standard_selected, standard_score, "std h", std_colors, "-", "navy")
    _plot_selected(ax, freq, deconv_fsc, deconv_grid, deconv_selected, deconv_score, "dec lambda", dec_colors, "--", "darkorange")
    if "standard_oracle" in oracle:
        ax.plot(freq, oracle["standard_oracle"][0], color="dodgerblue", linewidth=2.1, linestyle="-.", label="std shell oracle")
    if "deconv_oracle" in oracle:
        ax.plot(freq, oracle["deconv_oracle"][0], color="orangered", linewidth=2.1, linestyle="-.", label="dec shell oracle")
    if "combined_oracle" in oracle:
        ax.plot(freq, oracle["combined_oracle"][0], color="black", linewidth=2.4, linestyle=":", label="combined shell oracle")
    if distribution_mean is not None:
        ax.plot(freq, distribution_mean[0], color="forestgreen", linewidth=2.3, linestyle="--", label="distribution mean")
    ax.axhline(0.5, color="0.35", linestyle="--", linewidth=0.9)
    ax.axhline(1.0 / 7.0, color="0.35", linestyle=":", linewidth=0.9)
    ax.set_ylabel("masked FSC vs GT")
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_ylim(-0.08, 1.03)
    ax.set_title("FSC vs GT, selected standard and deconvolved candidates overlaid")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    _plot_selected(
        ax, freq, standard_err, standard_grid, standard_selected, standard_score, "std h", std_colors, "-", "navy", logy=True
    )
    _plot_selected(
        ax, freq, deconv_err, deconv_grid, deconv_selected, deconv_score, "dec lambda", dec_colors, "--", "darkorange", logy=True
    )
    if "standard_oracle" in oracle:
        ax.semilogy(freq, np.maximum(oracle["standard_oracle"][1], 1e-30), color="dodgerblue", linewidth=2.1, linestyle="-.", label="std shell oracle")
    if "deconv_oracle" in oracle:
        ax.semilogy(freq, np.maximum(oracle["deconv_oracle"][1], 1e-30), color="orangered", linewidth=2.1, linestyle="-.", label="dec shell oracle")
    if "combined_oracle" in oracle:
        ax.semilogy(freq, np.maximum(oracle["combined_oracle"][1], 1e-30), color="black", linewidth=2.4, linestyle=":", label="combined shell oracle")
    if distribution_mean is not None:
        ax.semilogy(freq, np.maximum(distribution_mean[1], 1e-30), color="forestgreen", linewidth=2.3, linestyle="--", label="distribution mean")
    selected_errors = np.concatenate([standard_err[standard_selected][:, shell_mask].ravel(), deconv_err[deconv_selected][:, shell_mask].ravel()])
    if oracle:
        selected_errors = np.concatenate([selected_errors] + [value[1][shell_mask] for value in oracle.values()])
    if distribution_mean is not None:
        selected_errors = np.concatenate([selected_errors, distribution_mean[1][shell_mask]])
    finite = selected_errors[np.isfinite(selected_errors)]
    ax.set_ylim(1e-4, max(float(np.nanpercentile(finite, 99.0)) * 1.6, 1e-2))
    ax.set_ylabel("masked relative Fourier error vs GT")
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_title("True relative Fourier error vs GT, selected candidates overlaid")
    ax.grid(True, which="both", alpha=0.25)

    xticks = np.arange(0.0, cfg.plot_frequency_max + 1e-9, 0.05)
    for ax in axes:
        ax.set_xlim(0.0, cfg.plot_frequency_max)
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", which="both", labelbottom=True)

    handles, labels = axes[0].get_legend_handles_labels()
    err_handles, err_labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels + err_labels, handles + err_handles))
    fig.legend(by_label.values(), by_label.keys(), loc="outside lower center", ncol=4, fontsize=8, frameon=True)

    png = plots_dir / "selected_standard_vs_deconvolved_true_gt_overlay.png"
    pdf = plots_dir / "selected_standard_vs_deconvolved_true_gt_overlay.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)

    return {
        "png": str(png),
        "pdf": str(pdf),
        "standard_selected_indices_0based": [int(v) for v in standard_selected],
        "standard_selected_h": [float(standard_grid[v]) for v in standard_selected],
        "standard_best_index_0based": int(np.nanargmin(standard_score)),
        "standard_best_h": float(standard_grid[int(np.nanargmin(standard_score))]),
        "deconv_selected_indices_0based": [int(v) for v in deconv_selected],
        "deconv_selected_lambda": [float(deconv_grid[v]) for v in deconv_selected],
        "deconv_best_index_0based": int(np.nanargmin(deconv_score)),
        "deconv_best_lambda": float(deconv_grid[int(np.nanargmin(deconv_score))]),
    }


def _save_real_space_kernels(
    cfg: SpikeKernelReportConfig,
    standard_grid: np.ndarray,
    deconv_grid: np.ndarray,
    plots_dir: Path,
) -> dict | None:
    if cfg.pipeline_root is None or cfg.target_point is None:
        logger.info("Skipping real-space kernel plots because pipeline root or target point is not set.")
        return None

    latent_path = cfg.pipeline_root / "model/zdim_1/latent_coords_noreg.npy"
    precision_path = cfg.pipeline_root / "model/zdim_1/latent_precision_noreg.npy"
    if not latent_path.exists() or not precision_path.exists():
        logger.warning("Skipping real-space kernel plots; missing %s or %s", latent_path, precision_path)
        return None

    deconv_params = utils.pickle_load(_state_dir(cfg.deconvolved_root, cfg.state_label) / "params.pkl")
    sigma_ref = float(deconv_params["sigma_ref"])
    zs = np.load(latent_path).reshape(-1)
    precision = np.load(precision_path).reshape(-1)
    target_point = float(np.loadtxt(cfg.target_point).reshape(-1)[0])
    latent_diff = zs - target_point
    finite_diff = np.isfinite(latent_diff)
    x_extent = float(np.quantile(np.abs(latent_diff[finite_diff]), 0.995))
    x_grid = np.linspace(-x_extent, x_extent, 3000)
    precision_ref = float(np.median(precision[np.isfinite(precision) & (precision > 0)]))

    standard_kernels = np.stack([_standard_kernel(x_grid, value, precision_ref) for value in standard_grid])
    deconv_kernels = np.stack([_deconv_kernel(x_grid, value, sigma_ref) for value in deconv_grid])
    standard_norm = standard_kernels / np.maximum(np.max(np.abs(standard_kernels), axis=1, keepdims=True), 1e-12)
    deconv_norm = deconv_kernels / np.maximum(np.max(np.abs(deconv_kernels), axis=1, keepdims=True), 1e-12)

    std_colors = plt.cm.Blues(np.linspace(0.25, 0.95, standard_grid.size))
    dec_colors = plt.cm.Oranges(np.linspace(0.25, 0.95, deconv_grid.size))

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    fig.suptitle(f"All real-space latent kernels | {cfg.report_title}", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    for idx, color in enumerate(std_colors):
        ax.plot(x_grid, standard_kernels[idx], color=color, linewidth=1.0, alpha=0.75)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_xlabel("latent difference z - z*")
    ax.set_ylabel("kernel weight")
    ax.set_title("Standard kernels, raw")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    for idx, color in enumerate(dec_colors):
        ax.plot(x_grid, deconv_kernels[idx], color=color, linewidth=1.0, alpha=0.75)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_xlabel("latent difference z - z*")
    ax.set_ylabel("kernel weight, symlog")
    ax.set_title("Deconvolved kernels, raw symlog")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 0]
    for idx, color in enumerate(std_colors):
        ax.plot(x_grid, standard_norm[idx], color=color, linewidth=1.0, alpha=0.75)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_xlabel("latent difference z - z*")
    ax.set_ylabel("kernel / max(abs(kernel))")
    ax.set_title("Standard kernels, normalized shape")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    for idx, color in enumerate(dec_colors):
        ax.plot(x_grid, deconv_norm[idx], color=color, linewidth=1.0, alpha=0.75)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_xlabel("latent difference z - z*")
    ax.set_ylabel("kernel / max(abs(kernel))")
    ax.set_title("Deconvolved kernels, normalized shape")
    ax.grid(True, alpha=0.25)

    sm_std = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=1, vmax=standard_grid.size))
    sm_dec = plt.cm.ScalarMappable(cmap="Oranges", norm=plt.Normalize(vmin=1, vmax=deconv_grid.size))
    fig.colorbar(sm_std, ax=[axes[0, 0], axes[1, 0]], shrink=0.8, pad=0.01).set_label("standard candidate index")
    fig.colorbar(sm_dec, ax=[axes[0, 1], axes[1, 1]], shrink=0.8, pad=0.01).set_label("deconvolved candidate index")

    png = plots_dir / "real_space_kernels.png"
    pdf = plots_dir / "real_space_kernels.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)

    npz = plots_dir / "real_space_kernels.npz"
    np.savez_compressed(
        npz,
        x_grid=x_grid,
        standard_grid=standard_grid,
        deconv_grid=deconv_grid,
        standard_kernels=standard_kernels,
        deconv_kernels=deconv_kernels,
        standard_norm=standard_norm,
        deconv_norm=deconv_norm,
        precision_ref=precision_ref,
        sigma_ref=sigma_ref,
        target_point=target_point,
        latent_diff_abs_p995=x_extent,
    )
    return {"png": str(png), "pdf": str(pdf), "npz": str(npz)}


def _nearest_indices(values: np.ndarray, targets: list[float]) -> list[int]:
    return sorted({int(np.argmin(np.abs(values - target))) for target in targets})


def _save_deconv_kernels_over_embedding(
    cfg: SpikeKernelReportConfig,
    deconv_grid: np.ndarray,
    plots_dir: Path,
) -> dict | None:
    if cfg.pipeline_root is None or cfg.target_point is None:
        logger.info("Skipping deconvolved embedding-kernel plot because pipeline root or target point is not set.")
        return None

    latent_path = cfg.pipeline_root / "model/zdim_1/latent_coords_noreg.npy"
    precision_path = cfg.pipeline_root / "model/zdim_1/latent_precision_noreg.npy"
    if not latent_path.exists() or not precision_path.exists():
        logger.warning("Skipping deconvolved embedding-kernel plot; missing %s or %s", latent_path, precision_path)
        return None

    params = utils.pickle_load(_state_dir(cfg.deconvolved_root, cfg.state_label) / "params.pkl")
    sigma_ref = float(params["sigma_ref"])
    h_grid = deconv_grid * sigma_ref

    z = np.load(latent_path).reshape(-1).astype(np.float64)
    precision = np.load(precision_path).reshape(-1).astype(np.float64)
    target_z = float(np.loadtxt(cfg.target_point).reshape(-1)[0])
    valid = np.isfinite(z) & np.isfinite(precision) & (precision > 0)
    z = z[valid]
    precision = precision[valid]
    precision_ref = float(np.median(precision))

    z_lo, z_hi = np.quantile(z, [0.001, 0.999])
    pad = 0.08 * (z_hi - z_lo)
    x_grid = np.linspace(z_lo - pad, z_hi + pad, 3500)
    dense_precision = np.full_like(x_grid, precision_ref)
    dense_weights = deconvolution_weights_1d_many(x_grid - target_z, dense_precision, h_grid)
    dense_norm = dense_weights / np.maximum(np.max(np.abs(dense_weights), axis=1, keepdims=True), 1e-30)

    actual_weights = deconvolution_weights_1d_many(z - target_z, precision, h_grid)
    abs_sum = np.sum(np.abs(actual_weights), axis=1)
    signed_sum_ratio = np.abs(np.sum(actual_weights, axis=1)) / np.maximum(abs_sum, 1e-30)
    weight_variation = np.std(actual_weights, axis=1) / np.maximum(np.mean(np.abs(actual_weights), axis=1), 1e-30)
    negative_mass_fraction = np.sum(np.abs(np.minimum(actual_weights, 0.0)), axis=1) / np.maximum(abs_sum, 1e-30)
    central_weight = dense_weights[:, int(np.argmin(np.abs(x_grid - target_z)))]

    selected_targets = [0.2, 0.3, 0.42, 0.56, 0.68, 0.9, 1.2, 1.6, 2.5, 5.0, 10.0, 20.0]
    selected = _nearest_indices(deconv_grid, selected_targets)
    selected_colors = list(plt.cm.tab20(np.linspace(0, 1, len(selected))))

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 16),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.8, 2.0, 2.1, 1.5]},
    )
    fig.suptitle(f"Deconvolved kernels over observed embedding space | {cfg.report_title}", fontsize=16, fontweight="bold")

    ax = axes[0]
    ax.hist(z, bins=120, color="0.72", edgecolor="0.50", linewidth=0.3)
    ax.axvline(target_z, color="black", linewidth=2.0, label=f"target z*={target_z:.3g}")
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_ylabel("particle count")
    ax.set_title("Observed noisy embedding distribution")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    for row in dense_weights:
        ax.plot(x_grid, row, color="0.82", linewidth=0.6, alpha=0.50)
    for color, idx in zip(selected_colors, selected):
        ax.plot(x_grid, dense_weights[idx], color=color, linewidth=1.9, label=f"#{idx + 1}: lambda={deconv_grid[idx]:.3g}")
    ax.axvline(target_z, color="black", linewidth=1.2)
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylabel("raw kernel weight, symlog")
    ax.set_title("Raw signed kernels; gray lines are all candidates")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper left", ncol=3, fontsize=8)

    ax = axes[2]
    image = ax.imshow(
        dense_norm,
        aspect="auto",
        origin="lower",
        extent=[x_grid[0], x_grid[-1], deconv_grid[0], deconv_grid[-1]],
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    ax.axvline(target_z, color="black", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("lambda")
    ax.set_title("Normalized kernel shape heatmap: K / max(abs(K))")
    fig.colorbar(image, ax=ax, pad=0.01).set_label("normalized signed weight")

    ax = axes[3]
    ax.loglog(deconv_grid, np.maximum(weight_variation, 1e-30), marker="o", label="std(weight) / mean(abs(weight))")
    ax.loglog(deconv_grid, np.maximum(signed_sum_ratio, 1e-30), marker="s", label="abs(sum weight) / sum(abs weight)")
    ax.semilogx(deconv_grid, negative_mass_fraction, marker="^", label="negative abs-mass fraction")
    ax2 = ax.twinx()
    ax2.semilogx(deconv_grid, central_weight, color="black", marker="x", linewidth=1.2, label="central raw weight")
    ax.set_xlabel("lambda")
    ax.set_ylabel("actual image-weight diagnostics")
    ax2.set_ylabel("central raw kernel weight")
    ax.set_title("Useful range diagnostics from actual particle weights")
    ax.grid(True, which="both", alpha=0.22)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)

    for ax in axes[:3]:
        ax.set_xlabel("observed embedding coordinate z")

    png = plots_dir / "deconvolved_kernels_over_embedding.png"
    pdf = plots_dir / "deconvolved_kernels_over_embedding.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)

    zoom_mask = (deconv_grid >= max(0.2, deconv_grid[0])) & (deconv_grid <= min(2.0, deconv_grid[-1]))
    zoom_indices = np.flatnonzero(zoom_mask)
    zoom_selected = [idx for idx in _nearest_indices(deconv_grid, [0.2, 0.29, 0.42, 0.56, 0.68, 0.9, 1.19, 1.58, 2.0]) if zoom_mask[idx]]
    zoom_colors = list(plt.cm.tab10(np.linspace(0, 1, max(len(zoom_selected), 1))))

    fig_zoom, axes_zoom = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True, sharex=True)
    fig_zoom.suptitle("Deconvolved kernel shapes over embedding space | lambda zoom", fontsize=15, fontweight="bold")
    axes_zoom[0].hist(z, bins=120, color="0.83", edgecolor="0.55", linewidth=0.25)
    axes_zoom[0].axvline(target_z, color="black", linewidth=2.0, label=f"target z*={target_z:.3g}")
    axes_zoom[0].set_ylabel("particle count")
    axes_zoom[0].set_title("Observed noisy embedding distribution")
    axes_zoom[0].grid(True, alpha=0.2)
    axes_zoom[0].legend(loc="upper right", fontsize=9)

    ax = axes_zoom[1]
    for idx in zoom_indices:
        ax.plot(x_grid, dense_norm[idx], color="0.82", linewidth=0.7, alpha=0.55)
    for color, idx in zip(zoom_colors, zoom_selected):
        ax.plot(x_grid, dense_norm[idx], color=color, linewidth=2.0, label=f"#{idx + 1}: lambda={deconv_grid[idx]:.3g}")
    ax.axvline(target_z, color="black", linewidth=1.2)
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("observed embedding coordinate z")
    ax.set_ylabel("K / max(abs(K))")
    ax.set_title("Normalized signed kernel shapes")
    ax.grid(True, alpha=0.24)
    ax.legend(loc="lower left", ncol=3, fontsize=8)

    zoom_png = plots_dir / "deconvolved_kernels_over_embedding_zoom.png"
    zoom_pdf = plots_dir / "deconvolved_kernels_over_embedding_zoom.pdf"
    fig_zoom.savefig(zoom_png, dpi=180)
    fig_zoom.savefig(zoom_pdf)
    plt.close(fig_zoom)

    npz = plots_dir / "deconvolved_kernels_over_embedding.npz"
    np.savez_compressed(
        npz,
        x_grid=x_grid,
        lambda_grid=deconv_grid,
        h_grid=h_grid,
        dense_weights=dense_weights,
        dense_norm=dense_norm,
        target_z=target_z,
        z_quantile_0p1_99p9=np.asarray([z_lo, z_hi]),
        precision_ref=precision_ref,
        sigma_ref=sigma_ref,
        selected_indices=np.asarray(selected, dtype=np.int32),
        signed_sum_ratio=signed_sum_ratio,
        weight_variation=weight_variation,
        negative_mass_fraction=negative_mass_fraction,
        central_weight=central_weight,
    )
    return {
        "png": str(png),
        "pdf": str(pdf),
        "npz": str(npz),
        "zoom_png": str(zoom_png),
        "zoom_pdf": str(zoom_pdf),
        "selected_indices_0based": [int(i) for i in selected],
        "selected_lambdas": [float(deconv_grid[i]) for i in selected],
    }


def _package_lowpass_candidates(
    cfg: SpikeKernelReportConfig,
    selected_summary: dict,
    target: np.ndarray,
    voxel_size: float,
) -> dict:
    out_dir = cfg.out_dir / f"selected_unfil_lowpass_{_safe_token(cfg.lowpass_cutoff)}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    standard_state_dir = _state_dir(cfg.standard_root, cfg.state_label)
    deconv_state_dir = _state_dir(cfg.deconvolved_root, cfg.state_label)
    manifest: dict[str, object] = {
        "description": "Half1/half2 unfiltered candidate averages, full-size spherical Fourier low-pass filtered. No mask and no resampling/downsampling.",
        "standard_root": str(standard_state_dir),
        "deconvolved_root": str(deconv_state_dir),
        "target_volume": str(cfg.target_volume),
        "shape": list(target.shape),
        "voxel_size_A": voxel_size,
        "lowpass_cutoff_1_per_A": cfg.lowpass_cutoff,
        "candidates": [],
    }

    gt_out = out_dir / f"gt_lp{_safe_token(cfg.lowpass_cutoff)}.mrc"
    utils.write_mrc(str(gt_out), _spherical_lowpass(target, voxel_size, cfg.lowpass_cutoff), voxel_size=voxel_size)
    manifest["gt_output"] = str(gt_out)

    for idx, value in zip(selected_summary["standard_selected_indices_0based"], selected_summary["standard_selected_h"]):
        manifest["candidates"].append(
            _write_lowpass_candidate("standard", standard_state_dir, int(idx), "h", float(value), voxel_size, cfg.lowpass_cutoff, out_dir)
        )
    for idx, value in zip(selected_summary["deconv_selected_indices_0based"], selected_summary["deconv_selected_lambda"]):
        manifest["candidates"].append(
            _write_lowpass_candidate("deconvolved", deconv_state_dir, int(idx), "lambda", float(value), voxel_size, cfg.lowpass_cutoff, out_dir)
        )

    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    tar_path = out_dir.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    manifest["tarball"] = str(tar_path)
    _write_json(manifest_path, manifest)
    return {"directory": str(out_dir), "tarball": str(tar_path), "manifest": str(manifest_path)}


def _write_report_readme(cfg: SpikeKernelReportConfig, summary: dict) -> None:
    text = f"""# Spike Kernel Regression Report

Generated by `recovar spike_kernel_report`.

- standard root: `{cfg.standard_root}`
- deconvolved root: `{cfg.deconvolved_root}`
- target volume: `{cfg.target_volume}`
- mask: `{cfg.mask}`
- pipeline root: `{cfg.pipeline_root}`
- target point: `{cfg.target_point}`

Primary plots live in `plots/`.

Key outputs:

- all candidates: `{summary["all_candidate_report"]["all_candidates_png"]}`
- selected overlay: `{summary["selected_overlay"]["png"]}`
- shell oracle volumes: `{summary["all_candidate_report"]["oracle_outputs"]}`
- masked unfiltered oracle/candidate volumes: `{summary["masked_unfiltered_package"]}`
"""
    (cfg.out_dir / "README.md").write_text(text)


def generate_report(cfg: SpikeKernelReportConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cfg.out_dir / "plots"
    tables_dir = cfg.out_dir / "tables"
    oracle_dir = cfg.out_dir / "oracle_estimators"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)

    target = np.asarray(utils.load_mrc(cfg.target_volume), dtype=np.float32)
    mask = np.clip(np.asarray(utils.load_mrc(cfg.mask), dtype=np.float32), 0.0, 1.0)
    if target.shape != mask.shape:
        raise ValueError(f"Target volume and mask shapes differ: {target.shape} vs {mask.shape}")

    voxel_size = _voxel_size(cfg.standard_root, cfg.state_label)
    standard_grid = _candidate_grid(cfg.standard_root, cfg.state_label)
    deconv_grid = _candidate_grid(cfg.deconvolved_root, cfg.state_label)

    standard_fsc, standard_err = _all_masked_metrics(
        cfg.standard_root, cfg.state_label, target, mask, cfg.expected_candidates
    )
    deconv_fsc, deconv_err = _all_masked_metrics(
        cfg.deconvolved_root, cfg.state_label, target, mask, cfg.expected_candidates
    )
    freq = np.arange(standard_fsc.shape[1], dtype=np.float64) / (target.shape[0] * voxel_size)

    all_report = _save_all_candidate_report(
        cfg,
        target,
        mask,
        standard_grid,
        deconv_grid,
        standard_fsc,
        deconv_fsc,
        standard_err,
        deconv_err,
        freq,
        voxel_size,
        plots_dir,
        oracle_dir,
    )
    selected = _save_selected_overlay(cfg, Path(all_report["metrics_npz"]), plots_dir)
    real_space = _save_real_space_kernels(cfg, standard_grid, deconv_grid, plots_dir)
    deconv_embedding = _save_deconv_kernels_over_embedding(cfg, deconv_grid, plots_dir)
    masked_unfiltered = (
        _package_masked_unfiltered_oracle_volumes(
            cfg,
            all_report,
            target,
            mask,
            standard_grid,
            deconv_grid,
            voxel_size,
        )
        if cfg.package_masked_unfiltered
        else None
    )
    lowpass = _package_lowpass_candidates(cfg, selected, target, voxel_size) if cfg.package_lowpass else None

    candidate_map = tables_dir / "candidate_volume_index_map.csv"
    with candidate_map.open("w") as f:
        f.write("method,index_0based,index_1based,parameter_name,parameter_value\n")
        for idx, value in enumerate(standard_grid):
            f.write(f"standard,{idx},{idx + 1},h,{float(value):.9g}\n")
        for idx, value in enumerate(deconv_grid):
            f.write(f"deconvolved,{idx},{idx + 1},lambda,{float(value):.9g}\n")

    summary = {
        "standard_root": str(cfg.standard_root),
        "deconvolved_root": str(cfg.deconvolved_root),
        "target_volume": str(cfg.target_volume),
        "mask": str(cfg.mask),
        "out_dir": str(cfg.out_dir),
        "pipeline_root": str(cfg.pipeline_root) if cfg.pipeline_root else None,
        "target_point": str(cfg.target_point) if cfg.target_point else None,
        "state_label": cfg.state_label,
        "report_title": cfg.report_title,
        "voxel_size_A": voxel_size,
        "frequency_range_1_per_A": [float(freq[0]), float(freq[-1])],
        "standard_h_min_max": [float(np.nanmin(standard_grid)), float(np.nanmax(standard_grid))],
        "deconv_lambda_min_max": [float(np.nanmin(deconv_grid)), float(np.nanmax(deconv_grid))],
        "all_candidate_report": all_report,
        "selected_overlay": selected,
        "real_space_kernels": real_space,
        "deconvolved_kernels_over_embedding": deconv_embedding,
        "masked_unfiltered_package": masked_unfiltered,
        "lowpass_package": lowpass,
        "candidate_map": str(candidate_map),
    }
    _write_json(cfg.out_dir / "summary.json", summary)
    _write_report_readme(cfg, summary)
    return summary


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--standard-root", type=Path, required=True, help="compute_state output dir for standard mode.")
    parser.add_argument("--deconvolved-root", type=Path, required=True, help="compute_state output dir for deconvolved mode.")
    parser.add_argument("--target-volume", type=Path, required=True, help="GT target volume MRC.")
    parser.add_argument("--mask", type=Path, required=True, help="Mask used for FSC/error metrics.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Report directory to create, usually an 08_* folder.")
    parser.add_argument("--pipeline-root", type=Path, default=None, help="Pipeline output root, needed for kernel-over-embedding plots.")
    parser.add_argument("--target-point", type=Path, default=None, help="Target latent point text file.")
    parser.add_argument("--state-label", type=str, default="state000")
    parser.add_argument("--report-title", type=str, default="spike kernel regression comparison")
    parser.add_argument("--expected-candidates", type=int, default=None)
    parser.add_argument("--score-frequency-max", type=float, default=0.35)
    parser.add_argument("--plot-frequency-max", type=float, default=0.40)
    parser.add_argument("--selected-count", type=int, default=10)
    parser.add_argument(
        "--no-package-masked-unfiltered",
        dest="package_masked_unfiltered",
        action="store_false",
        help="Disable the default masked, unfiltered shell-oracle/candidate volume package.",
    )
    parser.add_argument("--package-lowpass", action="store_true")
    parser.add_argument("--lowpass-cutoff", type=float, default=0.15)
    parser.set_defaults(package_masked_unfiltered=True)
    return parser


def _config_from_args(args: argparse.Namespace) -> SpikeKernelReportConfig:
    return SpikeKernelReportConfig(
        standard_root=Path(args.standard_root),
        deconvolved_root=Path(args.deconvolved_root),
        target_volume=Path(args.target_volume),
        mask=Path(args.mask),
        out_dir=Path(args.out_dir),
        pipeline_root=Path(args.pipeline_root) if args.pipeline_root else None,
        target_point=Path(args.target_point) if args.target_point else None,
        state_label=str(args.state_label),
        report_title=str(args.report_title),
        expected_candidates=args.expected_candidates,
        score_frequency_max=float(args.score_frequency_max),
        plot_frequency_max=float(args.plot_frequency_max),
        selected_count=int(args.selected_count),
        package_masked_unfiltered=bool(args.package_masked_unfiltered),
        package_lowpass=bool(args.package_lowpass),
        lowpass_cutoff=float(args.lowpass_cutoff),
    )


def main() -> None:
    parser = add_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s", level=logging.INFO)
    summary = generate_report(_config_from_args(args))
    logger.info("Finished spike kernel report: %s", summary["out_dir"])
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
