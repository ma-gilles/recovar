"""Helpers for the 1D kernel-bandwidth benchmark.

This module keeps the benchmark command small and isolates the pure pieces:

- trajectory PCA / optional projection
- bandwidth-bin selection
- shellwise oracle and CV scoring
- lightweight plotting and summary exports
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import utils
from recovar.heterogeneity import heterogeneity_volume as hv
from recovar.heterogeneity import kernel_regression_reconstruction as kernel_recon
from recovar.output import plot_style


@dataclass(frozen=True)
class VolumePcaResult:
    """PCA metadata for a finite set of trajectory volumes."""

    mean: np.ndarray
    scores: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_energy: np.ndarray


def fit_volume_pca(volumes: np.ndarray) -> VolumePcaResult:
    """Fit an exact PCA model to a small stack of volumes.

    Parameters
    ----------
    volumes:
        Array of shape ``(n_states, *volume_shape)``.
    """
    vols = np.asarray(volumes, dtype=np.float64)
    if vols.ndim < 2:
        raise ValueError(f"volumes must have at least 2 dimensions, got shape {vols.shape}")

    n_states = vols.shape[0]
    flat = vols.reshape(n_states, -1)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean

    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = u * s
    energy = np.square(s)
    total = float(np.sum(energy))
    explained = energy / total if total > 0 else np.zeros_like(energy)

    return VolumePcaResult(
        mean=mean.reshape(vols.shape[1:]),
        scores=scores,
        components=vt.reshape((vt.shape[0],) + vols.shape[1:]),
        singular_values=s,
        explained_energy=explained,
    )


def project_volume_trajectory(volumes: np.ndarray, n_pcs: int) -> tuple[np.ndarray, dict]:
    """Project trajectory volumes onto the first ``n_pcs`` principal components."""
    vols = np.asarray(volumes, dtype=np.float64)
    pca = fit_volume_pca(vols)
    n_keep = int(np.clip(n_pcs, 0, pca.components.shape[0]))

    if n_keep == 0:
        projected = vols.copy()
    else:
        flat = pca.mean.reshape(1, -1) + pca.scores[:, :n_keep] @ pca.components[:n_keep].reshape(n_keep, -1)
        projected = flat.reshape(vols.shape)

    meta = {
        "n_pcs": n_keep,
        "mean": np.asarray(pca.mean, dtype=np.float32),
        "scores": np.asarray(pca.scores, dtype=np.float32),
        "components": np.asarray(pca.components, dtype=np.float32),
        "singular_values": np.asarray(pca.singular_values, dtype=np.float32),
        "explained_energy": np.asarray(pca.explained_energy, dtype=np.float32),
    }
    return np.asarray(projected, dtype=np.float32), meta


def make_state_distribution(n_states: int, kind: str = "uniform") -> np.ndarray:
    """Return a probability vector over trajectory states."""
    if n_states <= 0:
        raise ValueError(f"n_states must be positive, got {n_states}")

    if kind == "uniform":
        weights = np.ones(n_states, dtype=np.float64)
    elif kind == "vonmises":
        from scipy.stats import vonmises

        theta = np.linspace(0.0, 2.0 * np.pi, n_states, endpoint=False)
        means = [np.pi / 2, np.pi, 3 * np.pi / 2]
        kappas = [6.0, 6.0, 6.0]
        mix = np.array([2.0, 1.0, 2.0], dtype=np.float64)
        mix /= mix.sum()
        weights = np.zeros_like(theta)
        for weight, loc, kappa in zip(mix, means, kappas, strict=True):
            weights += weight * vonmises.pdf(theta, loc=loc, kappa=kappa)
    else:
        raise ValueError(f"Unknown state distribution kind: {kind}")

    weights = np.asarray(weights, dtype=np.float64)
    weights /= np.sum(weights)
    return weights.astype(np.float32)


def write_volume_prefix(volumes: np.ndarray, prefix: str | Path, voxel_size: float) -> str:
    """Write ``volumes`` to ``prefix0000.mrc`` style files and return the prefix."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for idx, volume in enumerate(np.asarray(volumes)):
        utils.write_mrc(f"{prefix}{idx:04d}.mrc", np.asarray(volume, dtype=np.float32), voxel_size=voxel_size)
    return str(prefix)


def shell_labels_for_volume(volume_shape: tuple[int, int, int]) -> np.ndarray:
    """Integer Fourier-shell labels for the centered 3-D FFT grid."""
    labels = ftu.get_grid_of_radial_distances(volume_shape, rounded=True)
    return np.asarray(labels, dtype=np.int32)


def shell_sums(values: np.ndarray, labels: np.ndarray, n_shells: int) -> np.ndarray:
    """Sum ``values`` within each shell index."""
    return np.bincount(labels.ravel(), weights=np.asarray(values).ravel(), minlength=n_shells)


def choose_bandwidth_bins(
    distances_by_half: list[np.ndarray],
    n_bandwidths: int = 50,
    n_min_particles: int = 200,
) -> np.ndarray:
    """Choose candidate bandwidth thresholds using the production RECOVAR rule.

    The real reconstruction path picks bins from the second half-set's latent
    distance cloud via :func:`pick_heterogeneity_bins2`, which uses the
    minimum-size threshold and the 95th percentile of that same cloud.
    """
    if len(distances_by_half) < 2:
        raise ValueError("distances_by_half must contain two half-set arrays")

    bins = hv.pick_heterogeneity_bins2(
        -1,
        np.asarray(distances_by_half[1]),
        q=0.5,
        min_images=int(n_min_particles),
        n_bins=int(n_bandwidths),
    )
    bins = np.asarray(bins, dtype=np.float32)
    if bins.ndim != 1 or bins.size == 0:
        raise ValueError(f"Unexpected bandwidth bin array shape {bins.shape}")
    return bins


def compute_candidate_estimates(
    dataset,
    distances_by_half: list[np.ndarray],
    bins: np.ndarray,
    *,
    batch_size: int | None = None,
    heterogeneity_kernel: str = "parabola",
    use_fast_rfft: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Compute all candidate estimates plus the small-bin CV target."""
    vol_shape = tuple(dataset.volume_shape)
    kernel = "square" if heterogeneity_kernel == "flat" else heterogeneity_kernel

    estimates_by_half: list[np.ndarray] = []
    cv_by_half: list[np.ndarray] = []
    lhs_by_half: list[np.ndarray] = []

    for half in range(2):
        half_ds = dataset.get_halfset(half)
        estimates = kernel_recon.estimate_standard_kernel_volumes(
            half_ds,
            np.asarray(distances_by_half[half]),
            np.asarray(bins),
            batch_size=batch_size,
            tau=None,
            grid_correct=False,
            disc_type="linear_interp",
            use_spherical_mask=False,
            return_lhs_rhs=False,
            heterogeneity_kernel=kernel,
            upsampling_factor=1,
            return_real_space=True,
            use_fast_rfft=use_fast_rfft,
        )
        estimates = np.asarray(estimates).reshape(len(bins), *vol_shape).astype(np.float32)
        estimates_by_half.append(estimates)

        cv_est, lhs_half, _rhs = kernel_recon.estimate_standard_kernel_volumes(
            half_ds,
            np.asarray(distances_by_half[half]),
            np.asarray(bins[:1]),
            batch_size=batch_size,
            tau=None,
            grid_correct=False,
            disc_type="linear_interp",
            use_spherical_mask=False,
            return_lhs_rhs=True,
            heterogeneity_kernel=kernel,
            upsampling_factor=1,
            return_real_space=True,
            use_fast_rfft=use_fast_rfft,
        )
        cv_est = np.asarray(cv_est).reshape(1, *vol_shape)[0].astype(np.float32)
        lhs_full = np.asarray(ftu.half_volume_to_full_volume(lhs_half[0], vol_shape)).reshape(vol_shape)
        if hasattr(dataset, "get_valid_frequency_indices"):
            lhs_full = lhs_full * np.asarray(dataset.get_valid_frequency_indices()).reshape(vol_shape)
        cv_by_half.append(cv_est)
        lhs_by_half.append(np.asarray(lhs_full, dtype=np.float32))

    return estimates_by_half, cv_by_half, lhs_by_half


def compute_shellwise_oracle_and_cv(
    estimates_by_half: list[np.ndarray],
    cv_by_half: list[np.ndarray],
    lhs_by_half: list[np.ndarray],
    target_volume: np.ndarray,
    relative_error: bool = True,
    cv_focus_mask: np.ndarray | None = None,
) -> dict:
    """Compute shellwise oracle errors and shellwise CV scores."""
    est0, est1 = [np.asarray(x) for x in estimates_by_half]
    cv0, cv1 = [np.asarray(x) for x in cv_by_half]
    lhs0, lhs1 = [np.asarray(x) for x in lhs_by_half]
    target_volume = np.asarray(target_volume)

    if est0.shape != est1.shape:
        raise ValueError(f"Half-set estimate shapes differ: {est0.shape} vs {est1.shape}")
    if target_volume.shape != est0.shape[1:]:
        raise ValueError(f"Target volume shape {target_volume.shape} does not match estimates {est0.shape[1:]}")

    if cv_focus_mask is not None:
        cv_focus_mask = np.asarray(cv_focus_mask)
        if cv_focus_mask.shape != target_volume.shape:
            raise ValueError(
                f"CV focus mask shape {cv_focus_mask.shape} does not match target volume shape {target_volume.shape}"
            )
        cv_focus_mask = cv_focus_mask.astype(np.float32, copy=False)

    labels = shell_labels_for_volume(target_volume.shape)
    n_shells = int(labels.max()) + 1
    shell_counts = np.bincount(labels.ravel(), minlength=n_shells)
    shell_counts = np.maximum(shell_counts, 1)

    oracle_target_volume = target_volume if cv_focus_mask is None else target_volume * cv_focus_mask
    target_fft = np.asarray(ftu.get_dft3(oracle_target_volume))
    target_shell_power = shell_sums(np.abs(target_fft) ** 2, labels, n_shells)
    target_shell_power = np.maximum(target_shell_power, 1e-12)

    oracle_error = np.zeros((est0.shape[0], n_shells), dtype=np.float64)
    oracle_error_abs = np.zeros_like(oracle_error)
    cv_score = np.zeros_like(oracle_error)

    for m in range(est0.shape[0]):
        est_avg = 0.5 * (est0[m] + est1[m])
        diff_true = est_avg - target_volume
        if cv_focus_mask is not None:
            diff_true = diff_true * cv_focus_mask
        diff_true_fft = np.asarray(ftu.get_dft3(diff_true))
        err_shell = shell_sums(np.abs(diff_true_fft) ** 2, labels, n_shells)
        oracle_error_abs[m] = err_shell
        oracle_error[m] = err_shell / target_shell_power if relative_error else err_shell

        d10 = est1[m] - cv0
        d01 = est0[m] - cv1
        if cv_focus_mask is not None:
            d10 = d10 * cv_focus_mask
            d01 = d01 * cv_focus_mask
        d10_fft = np.asarray(ftu.get_dft3(d10))
        d01_fft = np.asarray(ftu.get_dft3(d01))
        score_density = np.maximum(lhs0, 0.0) * np.abs(d10_fft) ** 2 + np.maximum(lhs1, 0.0) * np.abs(d01_fft) ** 2
        cv_score[m] = shell_sums(score_density, labels, n_shells)

    cv_score_per_voxel = cv_score / shell_counts[None, :]
    oracle_choice = np.argmin(oracle_error, axis=0)
    cv_choice = np.argmin(cv_score_per_voxel, axis=0)
    best_oracle_error = oracle_error[oracle_choice, np.arange(n_shells)]
    regret = oracle_error[cv_choice, np.arange(n_shells)] / np.maximum(best_oracle_error, 1e-12)

    return {
        "shell_labels": labels,
        "shell_counts": shell_counts,
        "target_shell_power": target_shell_power,
        "oracle_error": oracle_error,
        "oracle_error_abs": oracle_error_abs,
        "cv_score": cv_score,
        "cv_score_per_voxel": cv_score_per_voxel,
        "oracle_choice": oracle_choice,
        "cv_choice": cv_choice,
        "regret": regret,
        "cv_focus_mask_used": bool(cv_focus_mask is not None),
        "cv_focus_mask_fraction": None if cv_focus_mask is None else float(np.mean(cv_focus_mask > 0)),
    }


def summarize_shellwise_results(result: dict) -> dict:
    """Return scalar summary metrics for the benchmark."""
    oracle_choice = np.asarray(result["oracle_choice"])
    cv_choice = np.asarray(result["cv_choice"])
    regret = np.asarray(result["regret"])
    shell_counts = np.asarray(result["shell_counts"])
    resolved = shell_counts > 1
    if not np.any(resolved):
        resolved = np.ones_like(shell_counts, dtype=bool)

    choice_match_rate = float(np.mean(cv_choice[resolved] == oracle_choice[resolved]))
    return {
        "choice_match_rate": choice_match_rate,
        "median_regret": float(np.median(regret[resolved])),
        "p90_regret": float(np.quantile(regret[resolved], 0.90)),
        "mean_choice_offset": float(np.mean(np.abs(cv_choice[resolved] - oracle_choice[resolved]))),
        "max_regret": float(np.max(regret[resolved])),
        "n_resolved_shells": int(np.sum(resolved)),
    }


def save_summary_csv(path: str | Path, result: dict, candidate_bins: np.ndarray) -> None:
    """Write a per-shell summary table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    oracle_choice = np.asarray(result["oracle_choice"])
    cv_choice = np.asarray(result["cv_choice"])
    regret = np.asarray(result["regret"])
    oracle_error = np.asarray(result["oracle_error"])
    cv_score = np.asarray(result["cv_score_per_voxel"])
    shell_counts = np.asarray(result["shell_counts"])

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shell",
                "shell_count",
                "oracle_bandwidth_idx",
                "cv_bandwidth_idx",
                "oracle_error",
                "cv_score_per_voxel",
                "regret",
                "candidate_bandwidth",
            ],
        )
        writer.writeheader()
        for shell in range(len(shell_counts)):
            writer.writerow(
                {
                    "shell": shell,
                    "shell_count": int(shell_counts[shell]),
                    "oracle_bandwidth_idx": int(oracle_choice[shell]),
                    "cv_bandwidth_idx": int(cv_choice[shell]),
                    "oracle_error": float(oracle_error[oracle_choice[shell], shell]),
                    "cv_score_per_voxel": float(cv_score[cv_choice[shell], shell]),
                    "regret": float(regret[shell]),
                    "candidate_bandwidth": float(candidate_bins[int(cv_choice[shell])]),
                }
            )


def save_plots(
    plot_dir: str | Path,
    candidate_bins: np.ndarray,
    result: dict,
    *,
    max_selected_shells: int = 6,
) -> None:
    """Save benchmark figures as PDFs."""
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    oracle_error = np.asarray(result["oracle_error"])
    cv_score = np.asarray(result["cv_score_per_voxel"])
    oracle_choice = np.asarray(result["oracle_choice"])
    cv_choice = np.asarray(result["cv_choice"])
    regret = np.asarray(result["regret"])
    shell_counts = np.asarray(result["shell_counts"])

    candidate_bins = np.asarray(candidate_bins)
    valid_shells = np.flatnonzero(shell_counts > 1)
    if valid_shells.size == 0:
        valid_shells = np.arange(shell_counts.size)
    selected = np.unique(np.linspace(valid_shells[0], valid_shells[-1], num=min(max_selected_shells, valid_shells.size), dtype=int))

    fig, ax = plt.subplots(figsize=(10, 6))
    for shell in selected:
        ax.plot(candidate_bins, oracle_error[:, shell], label=f"oracle shell {shell}")
        ax.plot(candidate_bins, cv_score[:, shell], linestyle="--", label=f"cv shell {shell}")
    ax.set_xlabel("Bandwidth threshold")
    ax.set_ylabel("Relative shell error / CV score")
    ax.set_title("Selected shell curves")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "shell_curves_selected.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(oracle_choice, label="oracle")
    ax.plot(cv_choice, label="cv", linestyle="--")
    ax.set_xlabel("Shell")
    ax.set_ylabel("Chosen bandwidth index")
    ax.set_title("Bandwidth choice by shell")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "bandwidth_choice_by_shell.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(regret, color="black")
    ax.set_xlabel("Shell")
    ax.set_ylabel("Oracle regret")
    ax.set_title("CV regret relative to oracle")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "cv_vs_oracle_regret.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(shell_counts, label="shell count")
    ax.set_xlabel("Shell")
    ax.set_ylabel("Count")
    ax.set_title("Shell population")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "shell_population.pdf")
    plt.close(fig)


def dump_config(path: str | Path, config: dict) -> None:
    """Write a JSON config file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)


def write_json(path: str | Path, payload: dict) -> None:
    """Write a JSON payload with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def _format_scalar(value) -> str:
    """Format a scalar value for markdown output."""
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.6g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return str(value)


def build_notebook_style_report(
    *,
    metadata: dict,
    result: dict,
    candidate_bins: np.ndarray,
    stage_records: list[dict],
) -> str:
    """Render a notebook-like markdown report for the benchmark run."""
    candidate_bins = np.asarray(candidate_bins)
    lines: list[str] = []
    lines.append("# Kernel bandwidth benchmark report")
    lines.append("")
    lines.append("This report mirrors the step-by-step structure of the diagnosis notebook.")
    lines.append("Each stage names the artifact paths and the metric being computed.")
    lines.append("")

    lines.append("## Run overview")
    lines.append("")
    for key in [
        "trajectory_source",
        "embedding_source",
        "diagnostic_embedding_source",
        "pc_project",
        "target_state",
        "n_states",
        "n_images",
        "grid_size",
        "noise_level",
        "contrast_std",
    ]:
        if key in metadata:
            lines.append(f"- `{key}`: {_format_scalar(metadata[key])}")
    if "contrast_std_requested" in metadata:
        lines.append(f"- `contrast_std_requested`: {_format_scalar(metadata['contrast_std_requested'])}")
    lines.append(f"- `candidate_bins`: {int(candidate_bins.size)}")
    if candidate_bins.size:
        lines.append(f"- `bandwidth_range`: [{_format_scalar(candidate_bins[0])}, {_format_scalar(candidate_bins[-1])}]")
    if "volume_prefix" in metadata:
        lines.append(f"- `volume_prefix`: `{metadata['volume_prefix']}`")
    if "dataset_dir" in metadata:
        lines.append(f"- `dataset_dir`: `{metadata['dataset_dir']}`")
    if metadata.get("pipeline_output_dir") is not None:
        lines.append(f"- `pipeline_output_dir`: `{metadata['pipeline_output_dir']}`")
    lines.append("")

    lines.append("## Staged flow")
    lines.append("")
    for record in stage_records:
        lines.append(f"### Step {record['step']}: {record['title']}")
        lines.append("")
        if record.get("summary"):
            lines.append(record["summary"])
            lines.append("")
        if record.get("artifacts"):
            lines.append("Artifacts:")
            for name, value in record["artifacts"].items():
                lines.append(f"- `{name}`: `{value}`")
            lines.append("")
        if record.get("parameters"):
            lines.append("Parameters:")
            for name, value in record["parameters"].items():
                lines.append(f"- `{name}`: {_format_scalar(value)}")
            lines.append("")
        if record.get("metrics"):
            lines.append("Metrics:")
            for name, value in record["metrics"].items():
                lines.append(f"- `{name}`: {_format_scalar(value)}")
            lines.append("")

    lines.append("## Metric definitions")
    lines.append("")
    lines.append("- `oracle_error`: relative shellwise Fourier error of the oracle half-set average against the true target volume.")
    lines.append("- `cv_score_per_voxel`: shellwise cross-validation score after dividing the weighted Fourier quadratic by shell population.")
    lines.append("- `oracle_choice`: bandwidth index minimizing `oracle_error` in each shell.")
    lines.append("- `cv_choice`: bandwidth index minimizing `cv_score_per_voxel` in each shell.")
    lines.append("- `regret`: `oracle_error[cv_choice] / oracle_error[oracle_choice]` per shell.")
    lines.append("- `choice_match_rate`: fraction of resolved shells where `cv_choice == oracle_choice`.")
    lines.append("- `median_regret`, `p90_regret`, `max_regret`: summary of the regret distribution over resolved shells.")
    lines.append("")

    lines.append("## How compute_state picks h")
    lines.append("")
    lines.append(
        "The benchmark oracle is global and compares the candidate average against the true target volume shell by shell."
    )
    lines.append(
        "`compute_state` uses the same candidate-bandwidth grid, but the final selection is local: "
        "`choice_most_likely_split(...)` scores each candidate against the two half-set CV targets with a masked noisy error, "
        "then picks a bandwidth index per shell."
    )
    lines.append(
        "So if the benchmark oracle prefers the largest h, that does not by itself imply a bug in `compute_state`; "
        "the real path is solving a different local selection problem."
    )
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append("- `candidate_bins.npy`: candidate bandwidth thresholds.")
    lines.append("- `oracle_error.npy`: shellwise oracle error curves.")
    lines.append("- `cv_score.npy`: shellwise cross-validation score curves.")
    lines.append("- `oracle_choice.npy` / `cv_choice.npy`: selected bandwidth index per shell.")
    lines.append("- `regret.npy`: regret ratio per shell.")
    lines.append("- `summary.csv`: one row per Fourier shell.")
    lines.append("- `summary.json`: scalar summary and provenance.")
    lines.append("- `presentation/bandwidth_ladder.pdf`: visual grid of the candidate h ladder.")
    lines.append("- `presentation/latent_bin_population.pdf`: how many particles fall under each candidate h.")
    lines.append("- `presentation/compute_state_bandwidth_selection.pdf`: compute_state's per-shell h choice and bin populations.")
    lines.append("- `presentation/compute_state_shell_choice.pdf`: chosen h/bin and images-per-bin versus shell.")
    lines.append("- `presentation/compute_state_shell_choices.csv`: shell-by-shell selected h/bin in the moving-mask subvolume.")
    lines.append("- `presentation/compute_state_subset_embedding.pdf`: latent-space view of the mask-selected image subset.")
    lines.append("- `presentation/compute_state_subset_montage.pdf`: montage of the selected images.")
    lines.append("- `plots/*.pdf`: shell curves, selected bandwidths, regret, and shell population.")
    lines.append("")

    if metadata.get("compute_state_output_dir") is not None:
        lines.append("## compute_state inspection")
        lines.append("")
        lines.append(f"- `compute_state_output_dir`: `{metadata['compute_state_output_dir']}`")
        if metadata.get("compute_state_summary"):
            for key, value in metadata["compute_state_summary"].items():
                lines.append(f"- `{key}`: {_format_scalar(value)}")
        lines.append("")

    if "summary" in metadata:
        lines.append("## Summary metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for key, value in metadata["summary"].items():
            lines.append(f"| `{key}` | {_format_scalar(value)} |")
        lines.append("")

    return "\n".join(lines)


def write_markdown_report(path: str | Path, text: str) -> None:
    """Write a markdown report to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _save_figure(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _central_xy(volume: np.ndarray) -> np.ndarray:
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {vol.shape}")
    return np.real(vol[vol.shape[0] // 2])


def _affine_align_points(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Affine-align ``source`` to ``target`` in least-squares sense.

    This is purely for visualization: it lets us overlay GT state scores on top
    of a diagnostic embedding even when the two coordinate systems are not
    directly comparable.
    """
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if src.ndim != 2 or tgt.ndim != 2:
        raise ValueError(f"source and target must be 2-D arrays, got {src.shape} and {tgt.shape}")
    if src.shape[0] != tgt.shape[0]:
        raise ValueError(f"source and target must have the same number of rows, got {src.shape[0]} and {tgt.shape[0]}")
    if src.shape[0] == 0:
        return np.zeros((0, tgt.shape[1]), dtype=np.float32)

    design = np.concatenate([src, np.ones((src.shape[0], 1), dtype=np.float64)], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, tgt, rcond=None)
    aligned = design @ coeffs
    return np.asarray(aligned, dtype=np.float32)


def _state_mean_path(points: np.ndarray, labels: np.ndarray, n_states: int) -> np.ndarray:
    """Average point clouds state-by-state into a trajectory path."""
    pts = np.asarray(points, dtype=np.float64)
    lab = np.asarray(labels).reshape(-1)
    if pts.ndim != 2:
        raise ValueError(f"points must be a 2-D array, got shape {pts.shape}")
    if pts.shape[0] != lab.shape[0]:
        raise ValueError(f"points and labels must have matching rows, got {pts.shape[0]} and {lab.shape[0]}")

    path = np.full((int(n_states), pts.shape[1]), np.nan, dtype=np.float64)
    for state in range(int(n_states)):
        mask = lab == state
        if np.any(mask):
            path[state] = np.mean(pts[mask], axis=0)
    return path


def _volume_triptych(volume: np.ndarray) -> list[np.ndarray]:
    """Return the three orthogonal central slices of a 3-D volume."""
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {vol.shape}")
    return [
        np.real(vol[vol.shape[0] // 2]),
        np.real(vol[:, vol.shape[1] // 2]),
        np.real(vol[:, :, vol.shape[2] // 2]),
    ]


def _save_triptych_figure(
    path: str | Path,
    title: str,
    volumes: list[np.ndarray],
    row_labels: list[str],
    *,
    cmap: str = "gray",
) -> None:
    """Save a small grid of orthogonal central slices for a set of volumes."""
    import matplotlib.pyplot as plt

    if len(volumes) != len(row_labels):
        raise ValueError(f"volumes and row_labels must have the same length, got {len(volumes)} and {len(row_labels)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(volumes), 3, figsize=(12, 3.2 * len(volumes)))
    if len(volumes) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, (vol, row_label) in enumerate(zip(volumes, row_labels, strict=True)):
        slices = _volume_triptych(vol)
        for col, (sl, slab_title) in enumerate(zip(slices, ["XY", "XZ", "YZ"], strict=True)):
            ax = axes[row, col]
            vmax = np.max(np.abs(sl)) or 1.0
            ax.imshow(sl, cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{row_label} {slab_title}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontweight="bold")
    _save_figure(fig, path)


def _save_bandwidth_ladder_figure(
    path: str | Path,
    candidate_bins: np.ndarray,
    estimates_by_half: list[np.ndarray],
) -> None:
    """Save a grid of central slices for all candidate bandwidths."""
    import matplotlib.pyplot as plt

    if len(estimates_by_half) != 2:
        raise ValueError(f"estimates_by_half must contain exactly two half-sets, got {len(estimates_by_half)}")

    est0, est1 = [np.asarray(x) for x in estimates_by_half]
    if est0.shape != est1.shape:
        raise ValueError(f"Half-set estimate shapes differ: {est0.shape} vs {est1.shape}")
    if est0.ndim != 4:
        raise ValueError(f"Expected estimate stacks with shape (n_bins, n, n, n), got {est0.shape}")

    candidate_bins = np.asarray(candidate_bins)
    if candidate_bins.ndim != 1 or candidate_bins.size != est0.shape[0]:
        raise ValueError(
            f"candidate_bins shape {candidate_bins.shape} does not match estimate stack length {est0.shape[0]}"
        )

    avg_estimates = 0.5 * (est0 + est1)
    n_bins = avg_estimates.shape[0]
    n_cols = min(5, n_bins)
    n_rows = int(np.ceil(n_bins / n_cols))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.8 * n_rows), squeeze=False)
    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        if idx >= n_bins:
            ax.axis("off")
            continue
        slice_xy = np.real(avg_estimates[idx, avg_estimates.shape[1] // 2])
        vmax = np.max(np.abs(slice_xy)) or 1.0
        ax.imshow(slice_xy, cmap="gray", origin="lower", vmin=-vmax, vmax=vmax)
        ax.set_title(f"h[{idx}]={candidate_bins[idx]:.4g}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Bandwidth ladder: average half-set estimate at each candidate h", fontweight="bold")
    _save_figure(fig, path)


def save_latent_bin_population_figure(
    path: str | Path,
    candidate_bins: np.ndarray,
    particle_zs: np.ndarray,
    particle_cov_zs: np.ndarray,
    target_z: np.ndarray,
) -> dict[str, np.ndarray]:
    """Plot how many particles fall under each candidate latent-distance threshold."""
    import matplotlib.pyplot as plt

    candidate_bins = np.asarray(candidate_bins, dtype=np.float64).reshape(-1)
    zs = np.asarray(particle_zs, dtype=np.float64)
    covs = np.asarray(particle_cov_zs, dtype=np.float64)
    target = np.asarray(target_z, dtype=np.float64).reshape(1, -1)

    if zs.ndim != 2:
        raise ValueError(f"particle_zs must be 2-D, got shape {zs.shape}")
    if covs.ndim != 3 or covs.shape[0] != zs.shape[0]:
        raise ValueError(f"particle_cov_zs must have shape (n, d, d), got {covs.shape}")
    if target.shape[1] != zs.shape[1]:
        raise ValueError(f"target_z dim {target.shape[1]} does not match particle_zs dim {zs.shape[1]}")

    dz = zs - target
    if covs.shape[1] == covs.shape[2] == 1:
        var = np.maximum(covs[:, 0, 0], 1e-12)
        distances = (dz[:, 0] ** 2) / var
    else:
        inv_cov = np.linalg.inv(covs)
        distances = np.einsum("ni,nij,nj->n", dz, inv_cov, dz)

    cumulative = np.array([(distances < b).sum() for b in candidate_bins], dtype=np.int64)
    interval = np.diff(np.concatenate([[0], cumulative]))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(candidate_bins, cumulative, marker="o", color="tab:blue")
    ax.set_ylabel("# images with d < h")
    ax.set_title("Latent-bin population under candidate h thresholds")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(np.arange(candidate_bins.size), interval, color="tab:orange")
    ax.set_ylabel("# newly added images")
    ax.set_xlabel("candidate h index")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(np.arange(candidate_bins.size)[:: max(1, candidate_bins.size // 10)])
    ax.set_xticklabels([f"{b:.3g}" for b in candidate_bins[:: max(1, candidate_bins.size // 10)]], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"distances": distances, "cumulative": cumulative, "interval": interval}


def save_presentation_plots(
    plot_dir: str | Path,
    *,
    raw_volumes: np.ndarray,
    used_volumes: np.ndarray,
    candidate_bins: np.ndarray | None = None,
    candidate_estimates_by_half: list[np.ndarray] | None = None,
    cheat_volumes: np.ndarray,
    image_stack: np.ndarray,
    particle_zs: np.ndarray,
    particle_labels: np.ndarray,
    gt_state_scores: np.ndarray,
    target_state: int,
    target_volume: np.ndarray,
    moving_mask: np.ndarray | None = None,
    cv_by_half: list[np.ndarray] | None = None,
    diagnostic_zs: np.ndarray | None,
) -> dict[str, str]:
    """Save student-friendly walk-through figures.

    The output intentionally emphasizes interpretation over compactness:
    raw trajectory volumes, PC-projected "cheat" volumes, a state-index
    PC path plot, image montages, and the diagnostic embedding if provided.
    """
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    raw_volumes = np.asarray(raw_volumes)
    used_volumes = np.asarray(used_volumes)
    cheat_volumes = np.asarray(cheat_volumes)
    image_stack = np.asarray(image_stack)
    particle_zs = np.asarray(particle_zs)
    particle_labels = np.asarray(particle_labels).reshape(-1)
    gt_state_scores = np.asarray(gt_state_scores)
    target_volume = np.asarray(target_volume)
    moving_mask = None if moving_mask is None else np.asarray(moving_mask)

    n_states = raw_volumes.shape[0]
    selected_states = np.unique([0, target_state, n_states // 2, n_states - 1])

    # 1) Raw trajectory and projected "cheat" trajectory volumes.
    for tag, vols in [("raw", raw_volumes), ("used", used_volumes), ("cheat_pc1", cheat_volumes)]:
        fig, axes = plt.subplots(len(selected_states), 3, figsize=(12, 3.2 * len(selected_states)))
        if len(selected_states) == 1:
            axes = np.expand_dims(axes, axis=0)
        for row, state in enumerate(selected_states):
            vol = vols[state]
            slices = [np.real(vol[vol.shape[0] // 2]), np.real(vol[:, vol.shape[1] // 2]), np.real(vol[:, :, vol.shape[2] // 2])]
            titles = ["XY", "XZ", "YZ"]
            for col, (sl, title) in enumerate(zip(slices, titles, strict=True)):
                ax = axes[row, col]
                vmax = np.max(np.abs(sl)) or 1.0
                ax.imshow(sl, cmap="gray", origin="lower", vmin=-vmax, vmax=vmax)
                ax.set_title(f"{tag} state {int(state)} {title}")
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"{tag} trajectory volumes", fontweight="bold")
        path = plot_dir / f"{tag}_trajectory_volumes.pdf"
        _save_figure(fig, path)
        saved[path.name] = str(path)

    # 2) Difference between the used target volume and the raw target volume.
    fig = plot_style.volume_slices(used_volumes[target_state] - raw_volumes[target_state], title="used - raw target volume", cmap="coolwarm")
    path = plot_dir / "target_volume_difference.pdf"
    _save_figure(fig, path)
    saved[path.name] = str(path)

    fig = plot_style.volume_slices(target_volume, title=f"target volume (state {target_state})", cmap="gray")
    path = plot_dir / "target_volume_slices.pdf"
    _save_figure(fig, path)
    saved[path.name] = str(path)

    if moving_mask is not None:
        mask_path = plot_dir / "moving_mask_slices.pdf"
        _save_triptych_figure(
            mask_path,
            f"moving mask (state {target_state})",
            [moving_mask.astype(np.float32)],
            [f"mask state {target_state}"],
            cmap="magma",
        )
        saved[mask_path.name] = str(mask_path)

    if cv_by_half is not None:
        cv_by_half = [np.asarray(x) for x in cv_by_half]
        if len(cv_by_half) != 2:
            raise ValueError(f"cv_by_half must contain exactly two volumes, got {len(cv_by_half)}")
        cv_path = plot_dir / "cv_estimates.pdf"
        _save_triptych_figure(
            cv_path,
            "CV estimates for the two half-sets",
            cv_by_half,
            ["CV half 0", "CV half 1"],
            cmap="gray",
        )
        saved[cv_path.name] = str(cv_path)

    if candidate_bins is not None and candidate_estimates_by_half is not None:
        ladder_path = plot_dir / "bandwidth_ladder.pdf"
        _save_bandwidth_ladder_figure(ladder_path, candidate_bins, candidate_estimates_by_half)
        saved[ladder_path.name] = str(ladder_path)

    # 3) PCA summary on the raw trajectory volumes.
    raw_pca = fit_volume_pca(raw_volumes)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    cum_energy = np.cumsum(raw_pca.explained_energy)
    ax.plot(np.arange(1, len(cum_energy) + 1), cum_energy, marker="o")
    ax.axvline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative explained energy")
    ax.set_title("Trajectory PCA scree")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    scores = raw_pca.scores
    if scores.shape[1] >= 2:
        scatter = ax.scatter(scores[:, 0], scores[:, 1], c=np.arange(n_states), cmap="viridis", s=40)
        fig.colorbar(scatter, ax=ax, label="state index")
        ax.set_xlabel("PC1 score")
        ax.set_ylabel("PC2 score")
    else:
        ax.plot(np.arange(n_states), scores[:, 0], marker="o")
        ax.set_xlabel("state index")
        ax.set_ylabel("PC1 score")
    ax.set_title("Trajectory states in PCA space")
    ax.grid(True, alpha=0.3)
    path = plot_dir / "trajectory_pca_summary.pdf"
    _save_figure(fig, path)
    saved[path.name] = str(path)

    # 4) One simple state-wise path plot in GT-PC coordinates.
    fig, ax = plt.subplots(figsize=(8, 4))
    if gt_state_scores.ndim == 2 and gt_state_scores.shape[1] >= 2:
        ax.plot(gt_state_scores[:, 0], gt_state_scores[:, 1], marker="o")
        for idx, (x, y) in enumerate(gt_state_scores[:, :2]):
            ax.annotate(str(idx), (x, y), fontsize=8)
        ax.set_xlabel("GT PC1")
        ax.set_ylabel("GT PC2")
    else:
        ax.plot(np.arange(gt_state_scores.shape[0]), gt_state_scores[:, 0], marker="o")
        ax.set_xlabel("state index")
        ax.set_ylabel("GT PC1")
    ax.set_title("Ground-truth latent path")
    ax.grid(True, alpha=0.3)
    path = plot_dir / "gt_latent_path.pdf"
    _save_figure(fig, path)
    saved[path.name] = str(path)

    # 5) Particle image montage.
    n_show = min(16, image_stack.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, idx in zip(axes.ravel(), range(n_show), strict=False):
        ax.imshow(np.real(image_stack[idx]), cmap="gray", origin="lower")
        ax.set_title(f"img {idx} / state {int(particle_labels[idx])}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n_show:]:
        ax.axis("off")
    fig.suptitle("Example synthetic particle images", fontweight="bold")
    path = plot_dir / "image_montage.pdf"
    _save_figure(fig, path)
    saved[path.name] = str(path)

    # 6) Benchmark latent coordinates, colored by state.
    fig, ax = plt.subplots(figsize=(8, 6))
    if particle_zs.ndim == 2 and particle_zs.shape[1] >= 2:
        scatter = ax.scatter(particle_zs[:, 0], particle_zs[:, 1], c=particle_labels, cmap="turbo", s=8, alpha=0.7)
        fig.colorbar(scatter, ax=ax, label="state")
        ax.set_xlabel("GT PC1")
        ax.set_ylabel("GT PC2")
    else:
        ax.scatter(np.arange(particle_zs.shape[0]), particle_zs[:, 0], c=particle_labels, cmap="turbo", s=8, alpha=0.7)
        ax.set_xlabel("particle index")
        ax.set_ylabel("GT PC1")
    ax.set_title("Benchmark latent coordinates")
    ax.grid(True, alpha=0.3)
    path = plot_dir / "benchmark_latent_scatter.pdf"
    _save_figure(fig, path)
    saved[path.name] = str(path)

    # 7) Diagnostic embedding, if available.
    if diagnostic_zs is not None:
        diagnostic_zs = np.asarray(diagnostic_zs)
        fig, ax = plt.subplots(figsize=(8, 6))
        if diagnostic_zs.shape[1] >= 2:
            scatter = ax.scatter(diagnostic_zs[:, 0], diagnostic_zs[:, 1], c=particle_labels, cmap="turbo", s=8, alpha=0.6)
            fig.colorbar(scatter, ax=ax, label="state")
            ax.set_xlabel("embedding dim 1")
            ax.set_ylabel("embedding dim 2")

            gt_path = np.asarray(gt_state_scores, dtype=np.float64)
            if gt_path.ndim != 2:
                gt_path = gt_path.reshape(-1, 1)
            diag_state_path = _state_mean_path(diagnostic_zs[:, :2], particle_labels, gt_path.shape[0])
            valid = np.isfinite(diag_state_path).all(axis=1) & np.isfinite(gt_path).all(axis=1)
            if np.count_nonzero(valid) >= 2:
                aligned_gt_path = _affine_align_points(gt_path[valid], diag_state_path[valid])
                full_gt_path = np.full((gt_path.shape[0], 2), np.nan, dtype=np.float32)
                full_gt_path[valid] = aligned_gt_path
                ax.plot(
                    full_gt_path[:, 0],
                    full_gt_path[:, 1],
                    color="black",
                    linewidth=3.0,
                    alpha=1.0,
                    label="GT path",
                    zorder=10,
                    marker="o",
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=1.0,
                )
                for idx, (x, y) in enumerate(full_gt_path):
                    if np.isfinite(x) and np.isfinite(y):
                        ax.annotate(str(idx), (x, y), fontsize=8, color="black", xytext=(4, 4), textcoords="offset points", zorder=11)
                ax.legend(loc="best", frameon=True)
        else:
            ax.scatter(np.arange(diagnostic_zs.shape[0]), diagnostic_zs[:, 0], c=particle_labels, cmap="turbo", s=8, alpha=0.7)
            ax.set_xlabel("particle index")
            ax.set_ylabel("embedding dim 1")
            gt_path = np.asarray(gt_state_scores, dtype=np.float64)
            if gt_path.ndim == 2 and gt_path.shape[1] >= 1:
                gt_state_line = _state_mean_path(diagnostic_zs[:, :1], particle_labels, gt_path.shape[0])[:, 0]
                valid = np.isfinite(gt_state_line) & np.isfinite(gt_path[:, 0])
                if np.count_nonzero(valid) >= 2:
                    aligned_gt = _affine_align_points(gt_path[valid, :1], gt_state_line[valid, None])
                    ax.plot(
                        np.where(valid)[0],
                        aligned_gt[:, 0],
                        color="black",
                        linewidth=3.0,
                        alpha=1.0,
                        label="GT path",
                        zorder=10,
                        marker="o",
                        markersize=5.5,
                        markerfacecolor="white",
                        markeredgecolor="black",
                        markeredgewidth=1.0,
                    )
                    ax.legend(loc="best", frameon=True)
        ax.set_title("Diagnostic RECOVAR embedding with GT overlay")
        ax.grid(True, alpha=0.3)
        path = plot_dir / "diagnostic_embedding_scatter.pdf"
        _save_figure(fig, path)
        saved[path.name] = str(path)

    return saved


def save_compute_state_walkthrough_plots(
    plot_dir: str | Path,
    compute_state_root: str | Path,
    *,
    image_stack: np.ndarray,
    particle_zs: np.ndarray,
    particle_labels: np.ndarray,
    gt_state_scores: np.ndarray,
    moving_mask: np.ndarray | None = None,
    diagnostic_zs: np.ndarray | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    """Save notebook-style figures derived from a real ``compute_state`` run.

    The goal is to mirror the production selection path:

    - read the per-volume ``params.pkl`` and ``ml_choice`` diagnostics
    - use the documented mask-based subset extraction path
    - plot the chosen subvolume in latent space
    - show the candidate-h ladder and the per-bin image counts
    - show the actual selected particle images
    The key output is a per-shell view of the chosen bandwidth in the moving-mask
    subvolume.  This helper writes both plots and a CSV table so the selection
    can be inspected shell by shell.
    """
    import matplotlib.pyplot as plt
    from recovar.commands import extract_image_subset as extract_subset
    from recovar.heterogeneity import locres

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    root = Path(compute_state_root)
    diagnostics_root = root / "diagnostics"
    if not diagnostics_root.exists():
        return {}, {}

    state_dirs = sorted([p for p in diagnostics_root.iterdir() if p.is_dir() and p.name.startswith("state")])
    if not state_dirs:
        return {}, {}

    state_dir = state_dirs[len(state_dirs) // 2]
    params_path = state_dir / "params.pkl"
    locres_path = state_dir / "local_resolution.mrc"
    het_dist_path = state_dir / "heterogeneity_distances.txt"
    if not params_path.exists() or not locres_path.exists() or not het_dist_path.exists():
        return {}, {}

    params = utils.pickle_load(params_path)
    heterogeneity_bins = np.asarray(params["heterogeneity_bins"], dtype=np.float64)
    n_images_per_bin = np.asarray(params["n_images_per_bin"], dtype=np.int64)
    ml_choice = np.asarray(params["ml_choice"], dtype=np.int64)
    ml_errors = np.asarray(params["ml_errors"], dtype=np.float64)
    voxel_size = float(params["voxel_size"])

    moving_mask = None if moving_mask is None else np.asarray(moving_mask, dtype=np.float32)
    if moving_mask is None:
        return {}, {}
    if moving_mask.shape != tuple(utils.load_mrc(locres_path).shape):
        raise ValueError(
            f"moving_mask shape {moving_mask.shape} does not match local_resolution shape {utils.load_mrc(locres_path).shape}"
        )

    saved: dict[str, str] = {}

    mask_path = plot_dir / "compute_state_moving_mask.mrc"
    utils.write_mrc(mask_path, moving_mask.astype(np.float32), voxel_size=voxel_size)

    subset_indices_path = plot_dir / "compute_state_subset_indices.pkl"
    extract_subset.extract_image_subset(str(state_dir), str(subset_indices_path), None, str(mask_path), None)
    subset_indices = np.asarray(utils.pickle_load(subset_indices_path), dtype=np.int64).reshape(-1)

    grid_size = int(utils.load_mrc(locres_path).shape[0])
    sampling_points = locres.get_sampling_points(grid_size, params["locres_sampling"], params["locres_maskrad"], voxel_size)
    subvolume_idx = extract_subset.decide_subvolume_index_from_mask(moving_mask, sampling_points)
    shell_choice = np.asarray(ml_choice[subvolume_idx])
    shell_errors = np.asarray(ml_errors[:, subvolume_idx, :])
    chosen_bin_thresholds = np.asarray(heterogeneity_bins[shell_choice], dtype=np.float64)
    chosen_bin_counts = np.asarray(n_images_per_bin[shell_choice], dtype=np.int64)
    shell_indices = np.arange(shell_choice.size, dtype=np.int64)
    shell_min_error = np.min(shell_errors, axis=0)

    csv_path = plot_dir / "compute_state_shell_choices.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "shell_idx",
                "chosen_bin_idx",
                "chosen_h",
                "chosen_n_images",
                "min_log10_error",
                "subvolume_idx",
            ]
        )
        for shell_idx, bin_idx, h, n_images, err in zip(
            shell_indices, shell_choice, chosen_bin_thresholds, chosen_bin_counts, shell_min_error, strict=True
        ):
            writer.writerow([int(shell_idx), int(bin_idx), float(h), int(n_images), float(np.log10(max(err, 1e-12))), int(subvolume_idx)])
    saved[csv_path.name] = str(csv_path)

    # Bandwidth selection overview.
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    ax = axes[0]
    ax.plot(np.arange(heterogeneity_bins.size), n_images_per_bin, marker="o", color="tab:blue")
    ax.set_title("Compute-state image counts per candidate h")
    ax.set_xlabel("Bandwidth index")
    ax.set_ylabel("# images")
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    im = ax2.imshow(np.log10(np.maximum(shell_errors, 1e-12)), aspect="auto", origin="lower", cmap="magma")
    ax2.plot(shell_indices, shell_choice, color="cyan", linewidth=2.0, label="chosen h/bin")
    ax2.set_xlabel("Fourier shell")
    ax2.set_ylabel("Bandwidth index")
    ax2.set_title(f"Compute-state shellwise ML errors at subvolume {subvolume_idx}")
    fig.colorbar(im, ax=ax2, label="log10(error)")
    ax2.legend(loc="upper right", frameon=True)
    path = plot_dir / "compute_state_bandwidth_selection.pdf"
    _save_figure(fig, path)
    saved[path.name] = str(path)

    # Compact per-shell summary of the actual selected h/bin.
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(shell_indices, shell_choice, marker="o", color="tab:blue")
    axes[0].set_ylabel("chosen bin idx")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Compute-state h/bin chosen in moving-mask subvolume {subvolume_idx}")

    axes[1].plot(shell_indices, chosen_bin_thresholds, marker="o", color="tab:green")
    axes[1].set_ylabel("chosen h")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(shell_indices, chosen_bin_counts, marker="o", color="tab:orange")
    axes[2].set_ylabel("# images")
    axes[2].set_xlabel("Fourier shell")
    axes[2].grid(True, alpha=0.3)
    path = plot_dir / "compute_state_shell_choice.pdf"
    _save_figure(fig, path)
    saved[path.name] = str(path)

    # Latent-space subset highlight.
    scatter_zs = np.asarray(diagnostic_zs) if diagnostic_zs is not None else np.asarray(particle_zs)
    selected_mask = np.zeros(scatter_zs.shape[0], dtype=bool)
    selected_mask[subset_indices[subset_indices < selected_mask.size]] = True
    fig, ax = plt.subplots(figsize=(8, 6))
    if scatter_zs.ndim == 2 and scatter_zs.shape[1] >= 2:
        ax.scatter(
            scatter_zs[~selected_mask, 0],
            scatter_zs[~selected_mask, 1],
            c="lightgray",
            s=8,
            alpha=0.35,
            label="all particles",
        )
        ax.scatter(
            scatter_zs[selected_mask, 0],
            scatter_zs[selected_mask, 1],
            c="crimson",
            s=16,
            alpha=0.9,
            label="selected subset",
        )
        gt_path = np.asarray(gt_state_scores, dtype=np.float64)
        if gt_path.ndim != 2:
            gt_path = gt_path.reshape(-1, 1)
        diag_state_path = _state_mean_path(scatter_zs[:, :2], particle_labels, gt_path.shape[0])
        valid = np.isfinite(diag_state_path).all(axis=1) & np.isfinite(gt_path).all(axis=1)
        if np.count_nonzero(valid) >= 2:
            aligned_gt_path = _affine_align_points(gt_path[valid], diag_state_path[valid])
            full_gt_path = np.full((gt_path.shape[0], 2), np.nan, dtype=np.float32)
            full_gt_path[valid] = aligned_gt_path
            ax.plot(
                full_gt_path[:, 0],
                full_gt_path[:, 1],
                color="black",
                linewidth=2.5,
                alpha=1.0,
                label="GT path",
                zorder=10,
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.0,
            )
        ax.set_xlabel("embedding dim 1")
        ax.set_ylabel("embedding dim 2")
    else:
        ax.scatter(np.arange(scatter_zs.shape[0])[~selected_mask], scatter_zs[~selected_mask, 0], c="lightgray", s=8, alpha=0.35)
        ax.scatter(np.arange(scatter_zs.shape[0])[selected_mask], scatter_zs[selected_mask, 0], c="crimson", s=16, alpha=0.9)
        ax.set_xlabel("particle index")
        ax.set_ylabel("embedding dim 1")
    ax.set_title(f"Particles selected by the mask-based compute_state subset ({subset_indices.size} images)")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    path = plot_dir / "compute_state_subset_embedding.pdf"
    _save_figure(fig, path)
    saved[path.name] = str(path)

    # Montage of the actual selected particles.
    n_show = min(16, subset_indices.size)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, idx in zip(axes.ravel(), subset_indices[:n_show], strict=False):
        ax.imshow(np.real(image_stack[int(idx)]), cmap="gray", origin="lower")
        ax.set_title(f"img {int(idx)} / state {int(particle_labels[int(idx)])}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n_show:]:
        ax.axis("off")
    fig.suptitle("Images selected by compute_state + mask", fontweight="bold")
    path = plot_dir / "compute_state_subset_montage.pdf"
    _save_figure(fig, path)
    saved[path.name] = str(path)

    return saved, {
        "compute_state_state": state_dir.name,
        "compute_state_subvolume_idx": int(subvolume_idx),
        "compute_state_subset_count": int(subset_indices.size),
        "compute_state_n_images_per_bin_head": n_images_per_bin[:10].tolist(),
        "compute_state_shell_choice_head": shell_choice[:10].tolist(),
        "compute_state_shell_h_head": chosen_bin_thresholds[:10].tolist(),
        "compute_state_shell_n_images_head": chosen_bin_counts[:10].tolist(),
        "compute_state_shell_choices_csv": str(csv_path),
    }
