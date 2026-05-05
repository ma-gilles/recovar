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
from recovar.heterogeneity import adaptive_kernel_discretization as akd


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
    q_max: float = 0.95,
    eps: float = 1e-8,
) -> np.ndarray:
    """Choose candidate bandwidth thresholds from the observed distance cloud."""
    d = np.concatenate([np.asarray(x).ravel() for x in distances_by_half])
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("Cannot choose bandwidth bins from an empty distance set")
    d = np.maximum(d, 0.0)

    kth = min(max(int(n_min_particles), 1), d.size - 1)
    b_min = float(np.partition(d, kth)[kth])
    b_max = float(np.quantile(d, q_max))
    if b_max <= b_min:
        b_max = b_min + eps

    r_min = np.sqrt(max(b_min, eps))
    r_max = np.sqrt(max(b_max, b_min + eps))
    radii = np.linspace(r_min, r_max, int(n_bandwidths))
    return np.square(radii)


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
        estimates = akd.even_less_naive_heterogeneity_scheme_relion_style(
            half_ds,
            None,
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

        cv_est, lhs_half, _rhs = akd.even_less_naive_heterogeneity_scheme_relion_style(
            half_ds,
            None,
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

    labels = shell_labels_for_volume(target_volume.shape)
    n_shells = int(labels.max()) + 1
    shell_counts = np.bincount(labels.ravel(), minlength=n_shells)
    shell_counts = np.maximum(shell_counts, 1)

    target_fft = np.asarray(ftu.get_dft3(target_volume))
    target_shell_power = shell_sums(np.abs(target_fft) ** 2, labels, n_shells)
    target_shell_power = np.maximum(target_shell_power, 1e-12)

    oracle_error = np.zeros((est0.shape[0], n_shells), dtype=np.float64)
    oracle_error_abs = np.zeros_like(oracle_error)
    cv_score = np.zeros_like(oracle_error)

    for m in range(est0.shape[0]):
        est_avg = 0.5 * (est0[m] + est1[m])
        diff_true_fft = np.asarray(ftu.get_dft3(est_avg - target_volume))
        err_shell = shell_sums(np.abs(diff_true_fft) ** 2, labels, n_shells)
        oracle_error_abs[m] = err_shell
        oracle_error[m] = err_shell / target_shell_power if relative_error else err_shell

        d10_fft = np.asarray(ftu.get_dft3(est1[m] - cv0))
        d01_fft = np.asarray(ftu.get_dft3(est0[m] - cv1))
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
