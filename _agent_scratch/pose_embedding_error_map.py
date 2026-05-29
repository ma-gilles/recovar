from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from recovar import utils


def _load_array(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, mmap_mode="r"))


def _load_simulation_info(run_dir: Path) -> dict:
    sim_path = run_dir / "03_dataset" / "simulation_info.pkl"
    if not sim_path.exists():
        raise FileNotFoundError(f"Missing simulation info: {sim_path}")
    return utils.pickle_load(sim_path)


def _load_rotations(run_dir: Path) -> np.ndarray:
    poses_path = run_dir / "03_dataset" / "poses.pkl"
    if poses_path.exists():
        poses = utils.pickle_load(poses_path)
        return np.asarray(poses[0], dtype=np.float64)
    sim_info = _load_simulation_info(run_dir)
    if "rots" not in sim_info:
        raise KeyError(f"Could not find rotations in {poses_path} or simulation_info.pkl")
    return np.asarray(sim_info["rots"], dtype=np.float64)


def _load_state_assignment(run_dir: Path) -> np.ndarray:
    state_path = run_dir / "03_dataset" / "state_assignment.npy"
    if state_path.exists():
        return np.asarray(np.load(state_path), dtype=np.int64)
    sim_info = _load_simulation_info(run_dir)
    if "image_assignment" not in sim_info:
        raise KeyError("simulation_info.pkl has no image_assignment")
    return np.asarray(sim_info["image_assignment"], dtype=np.int64)


def _metadata_candidates(run_dir: Path, pipeline_dir: Path) -> list[Path]:
    candidates = sorted(run_dir.glob("*embedding_metadata.json"))
    candidates += sorted(run_dir.glob("*noisy_embedding_metadata.json"))
    seen = set()
    unique = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    matching = []
    pipeline_resolved = pipeline_dir.resolve()
    for path in unique:
        try:
            metadata = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        value = metadata.get("pipeline")
        if value and Path(value).resolve() == pipeline_resolved:
            matching.append(path)
    return matching


def _load_metadata(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _true_latent_by_state_from_metadata(metadata: dict, run_dir: Path, n_states: int) -> np.ndarray | None:
    for key in ("state_latent_by_state_path", "true_scaled_by_state_path", "reference_gtpc1_reference_gtpc1_by_state_path"):
        value = metadata.get(key)
        if value and Path(value).exists():
            arr = np.asarray(np.load(value), dtype=np.float64).reshape(-1)
            if arr.size >= n_states:
                return arr[:n_states]

    if "scaled_index_slope" in metadata and "scaled_index_intercept" in metadata:
        states = np.arange(n_states, dtype=np.float64)
        return float(metadata["scaled_index_slope"]) * states + float(metadata["scaled_index_intercept"])

    if "gt_scaling_slope" in metadata and "gt_scaling_intercept" in metadata:
        pca_scores = run_dir / "02_active_volumes" / "pca_scores.npy"
        if pca_scores.exists():
            scores = np.asarray(np.load(pca_scores), dtype=np.float64)
            pc0 = scores[:n_states, 0]
            return float(metadata["gt_scaling_slope"]) * pc0 + float(metadata["gt_scaling_intercept"])

    return None


def _fit_true_latent_by_state_from_active_pca(run_dir: Path, z: np.ndarray, states: np.ndarray) -> np.ndarray:
    pca_scores = run_dir / "02_active_volumes" / "pca_scores.npy"
    if not pca_scores.exists():
        raise FileNotFoundError(
            "No metadata true latent found and no active-volume PCA scores available at "
            f"{pca_scores}"
        )
    scores = np.asarray(np.load(pca_scores), dtype=np.float64)
    n_states = int(states.max()) + 1
    raw = np.asarray(scores[:n_states, 0], dtype=np.float64)
    means = np.full(n_states, np.nan, dtype=np.float64)
    for state in range(n_states):
        mask = states == state
        if np.any(mask):
            means[state] = np.nanmean(z[mask, 0])
    valid = np.isfinite(raw) & np.isfinite(means)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("Not enough states to fit active PCA latent scale to estimated embedding")
    slope, intercept = np.polyfit(raw[valid], means[valid], deg=1)
    return slope * raw + intercept


def _load_true_latent(
    run_dir: Path,
    pipeline_dir: Path,
    z: np.ndarray,
    states: np.ndarray,
    metadata_path: Path | None,
) -> tuple[np.ndarray, Path | None, str]:
    if metadata_path is None:
        candidates = _metadata_candidates(run_dir, pipeline_dir)
        metadata_path = candidates[0] if candidates else None
    metadata = _load_metadata(metadata_path)
    n_states = int(states.max()) + 1

    by_state = None
    source = "active_pca_fit_to_state_mean_estimated_z"
    if metadata is not None:
        by_state = _true_latent_by_state_from_metadata(metadata, run_dir, n_states)
        if by_state is not None:
            source = f"metadata:{metadata_path}"
    if by_state is None:
        by_state = _fit_true_latent_by_state_from_active_pca(run_dir, z, states)

    if np.any(states < 0) or np.any(states >= by_state.size):
        raise ValueError(f"State assignments are outside true latent table: min={states.min()} max={states.max()}")
    return by_state[states], metadata_path, source


def _covariance_summary(precision: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision = np.asarray(precision, dtype=np.float64)
    if precision.ndim == 1:
        precision = precision[:, None, None]
    elif precision.ndim == 2:
        precision = precision[:, :, None] if precision.shape[1] == 1 else np.asarray([np.diag(row) for row in precision])

    n, d, _ = precision.shape
    if d == 1:
        p = precision[:, 0, 0]
        variance = np.full(n, np.nan, dtype=np.float64)
        valid = np.isfinite(p) & (p > 0)
        variance[valid] = 1.0 / p[valid]
        sigma = np.sqrt(variance)
        precision_trace = p
        return variance, sigma, precision_trace

    cov_trace = np.full(n, np.nan, dtype=np.float64)
    sigma_rms = np.full(n, np.nan, dtype=np.float64)
    precision_trace = np.trace(precision, axis1=1, axis2=2)
    for i in range(n):
        mat = precision[i]
        if np.all(np.isfinite(mat)):
            try:
                cov = np.linalg.inv(mat)
            except np.linalg.LinAlgError:
                continue
            trace = float(np.trace(cov))
            if np.isfinite(trace) and trace > 0:
                cov_trace[i] = trace
                sigma_rms[i] = np.sqrt(trace / d)
    return cov_trace, sigma_rms, precision_trace


def _viewing_angles_from_rotations(rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # RECOVAR rotates row-vector Fourier plane coordinates as coords @ R.
    # The viewing direction is therefore the rotated +Z plane normal: [0,0,1] @ R.
    view = np.asarray(rotations[:, 2, :], dtype=np.float64)
    norm = np.linalg.norm(view, axis=1)
    valid = np.isfinite(norm) & (norm > 0)
    view[valid] /= norm[valid, None]
    view[~valid] = np.nan

    phi = np.degrees(np.arctan2(view[:, 1], view[:, 0]))
    theta = np.degrees(np.arccos(np.clip(view[:, 2], -1.0, 1.0)))
    return phi, theta, view


def _bin_stat(phi: np.ndarray, theta: np.ndarray, values: np.ndarray, n_phi: int, n_theta: int, reducer: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi_edges = np.linspace(-180.0, 180.0, n_phi + 1)
    theta_edges = np.linspace(0.0, 180.0, n_theta + 1)
    phi_idx = np.searchsorted(phi_edges, phi, side="right") - 1
    theta_idx = np.searchsorted(theta_edges, theta, side="right") - 1
    phi_idx = np.clip(phi_idx, 0, n_phi - 1)
    theta_idx = np.clip(theta_idx, 0, n_theta - 1)
    valid = (
        np.isfinite(phi)
        & np.isfinite(theta)
        & np.isfinite(values)
        & (phi >= -180)
        & (phi <= 180)
        & (theta >= 0)
        & (theta <= 180)
    )
    flat = theta_idx[valid] * n_phi + phi_idx[valid]
    vals = values[valid]
    out = np.full(n_theta * n_phi, np.nan, dtype=np.float64)
    counts = np.bincount(flat, minlength=n_theta * n_phi).astype(np.int64)

    if reducer == "mean":
        sums = np.bincount(flat, weights=vals, minlength=n_theta * n_phi)
        nonzero = counts > 0
        out[nonzero] = sums[nonzero] / counts[nonzero]
    elif reducer == "median":
        order = np.argsort(flat, kind="stable")
        flat_sorted = flat[order]
        vals_sorted = vals[order]
        unique, start = np.unique(flat_sorted, return_index=True)
        end = np.r_[start[1:], flat_sorted.size]
        for bin_id, lo, hi in zip(unique, start, end):
            out[bin_id] = np.nanmedian(vals_sorted[lo:hi])
    elif reducer == "count":
        out = counts.astype(np.float64)
    else:
        raise ValueError(f"Unknown reducer {reducer}")

    return out.reshape(n_theta, n_phi), counts.reshape(n_theta, n_phi), (phi_edges, theta_edges)


def _plot_map(ax, grid: np.ndarray, phi_edges: np.ndarray, theta_edges: np.ndarray, title: str, label: str, cmap: str = "viridis"):
    masked = np.ma.masked_invalid(grid)
    image = ax.pcolormesh(phi_edges, theta_edges, masked, shading="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("view azimuth phi (deg)")
    ax.set_ylabel("view polar theta / RELION tilt-like angle (deg)")
    ax.set_xlim(phi_edges[0], phi_edges[-1])
    ax.set_ylim(theta_edges[0], theta_edges[-1])
    ax.grid(alpha=0.15, linewidth=0.5)
    plt.colorbar(image, ax=ax, label=label)


def _finite_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Map embedding error and latent covariance over viewing direction.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Experiment root containing 03_dataset.")
    parser.add_argument("--pipeline-dir", type=Path, default=None, help="Pipeline root containing model/zdim_*/ arrays.")
    parser.add_argument("--zdim", type=int, default=1)
    parser.add_argument("--embedding-kind", choices=["noreg", "regularized"], default="noreg")
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument(
        "--true-latent-mode",
        choices=["metadata", "active-pca-fit"],
        default="metadata",
        help="Use metadata truth if available, or fit active-volume PC0 to per-state mean estimated z.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-phi", type=int, default=72)
    parser.add_argument("--n-theta", type=int, default=36)
    parser.add_argument("--max-scatter", type=int, default=200000)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    pipeline_dir = (args.pipeline_dir or (run_dir / "06_pipeline")).resolve()
    out_dir = args.out_dir or (run_dir / f"09_pose_embedding_diagnostics_{pipeline_dir.name}_zdim{args.zdim}_{args.embedding_kind}")
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_noreg" if args.embedding_kind == "noreg" else ""
    z_path = pipeline_dir / "model" / f"zdim_{args.zdim}" / f"latent_coords{suffix}.npy"
    precision_path = pipeline_dir / "model" / f"zdim_{args.zdim}" / f"latent_precision{suffix}.npy"
    if not z_path.exists():
        raise FileNotFoundError(z_path)
    if not precision_path.exists():
        raise FileNotFoundError(precision_path)

    z = _load_array(z_path).astype(np.float64)
    if z.ndim == 1:
        z = z[:, None]
    precision = _load_array(precision_path)
    rotations = _load_rotations(run_dir)
    states = _load_state_assignment(run_dir)

    n = z.shape[0]
    if rotations.shape[0] != n or states.shape[0] != n:
        raise ValueError(f"Length mismatch: z={z.shape}, rotations={rotations.shape}, states={states.shape}")

    metadata_path = args.metadata_json
    if args.true_latent_mode == "active-pca-fit":
        true_by_state = _fit_true_latent_by_state_from_active_pca(run_dir, z, states)
        true_z = true_by_state[states]
        true_source = "active_pca_fit_to_state_mean_estimated_z"
    else:
        true_z, metadata_path, true_source = _load_true_latent(run_dir, pipeline_dir, z, states, metadata_path)
    z_error_vec = z[:, :1] - true_z[:, None]
    signed_error = z_error_vec[:, 0]
    abs_error = np.abs(signed_error)
    variance_trace, sigma_rms, precision_trace = _covariance_summary(precision)
    phi, theta, viewing_direction = _viewing_angles_from_rotations(rotations)

    median_abs_error_grid, counts, edges = _bin_stat(phi, theta, abs_error, args.n_phi, args.n_theta, "median")
    mean_signed_error_grid, _, _ = _bin_stat(phi, theta, signed_error, args.n_phi, args.n_theta, "mean")
    median_sigma_grid, _, _ = _bin_stat(phi, theta, sigma_rms, args.n_phi, args.n_theta, "median")
    count_grid, _, _ = _bin_stat(phi, theta, abs_error, args.n_phi, args.n_theta, "count")
    phi_edges, theta_edges = edges

    np.savez_compressed(
        out_dir / "pose_embedding_per_image_stats.npz",
        phi_deg=phi.astype(np.float32),
        theta_deg=theta.astype(np.float32),
        viewing_direction=viewing_direction.astype(np.float32),
        state_assignment=states.astype(np.int32),
        z_est=z.astype(np.float32),
        z_true=true_z.astype(np.float32),
        z_error=signed_error.astype(np.float32),
        z_abs_error=abs_error.astype(np.float32),
        latent_variance_trace=variance_trace.astype(np.float32),
        latent_sigma_rms=sigma_rms.astype(np.float32),
        latent_precision_trace=precision_trace.astype(np.float32),
        median_abs_error_grid=median_abs_error_grid.astype(np.float32),
        mean_signed_error_grid=mean_signed_error_grid.astype(np.float32),
        median_sigma_grid=median_sigma_grid.astype(np.float32),
        count_grid=count_grid.astype(np.int32),
        phi_edges=phi_edges.astype(np.float32),
        theta_edges=theta_edges.astype(np.float32),
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 8.8), constrained_layout=True)
    _plot_map(axes[0, 0], median_abs_error_grid, phi_edges, theta_edges, "Median |embedding error|", "|z - z_gt|", "magma")
    _plot_map(axes[0, 1], mean_signed_error_grid, phi_edges, theta_edges, "Mean signed embedding error", "z - z_gt", "coolwarm")
    _plot_map(axes[1, 0], median_sigma_grid, phi_edges, theta_edges, "Median posterior latent sigma", "sqrt(trace(cov)/zdim)", "viridis")
    _plot_map(axes[1, 1], np.log10(np.maximum(count_grid, 1.0)), phi_edges, theta_edges, "Image count per view bin", "log10(count)", "cividis")
    fig.suptitle(f"Pose dependence of embedding error and latent covariance | {run_dir.name} | {pipeline_dir.name}")
    fig.savefig(out_dir / "pose_embedding_error_covariance_maps.png", dpi=180)
    plt.close(fig)

    if args.max_scatter > 0:
        rng = np.random.default_rng(0)
        take_n = min(n, int(args.max_scatter))
        take = rng.choice(n, size=take_n, replace=False) if take_n < n else np.arange(n)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        sc0 = axes[0].scatter(phi[take], theta[take], c=abs_error[take], s=2, alpha=0.35, cmap="magma")
        axes[0].set_title("Per-image |embedding error|")
        axes[0].set_xlabel("view azimuth phi (deg)")
        axes[0].set_ylabel("view polar theta (deg)")
        plt.colorbar(sc0, ax=axes[0], label="|z - z_gt|")
        sc1 = axes[1].scatter(phi[take], theta[take], c=sigma_rms[take], s=2, alpha=0.35, cmap="viridis")
        axes[1].set_title("Per-image posterior latent sigma")
        axes[1].set_xlabel("view azimuth phi (deg)")
        axes[1].set_ylabel("view polar theta (deg)")
        plt.colorbar(sc1, ax=axes[1], label="sqrt(trace(cov)/zdim)")
        fig.savefig(out_dir / "pose_embedding_error_covariance_scatter.png", dpi=180)
        plt.close(fig)

    corr_inputs = np.vstack([abs_error, sigma_rms]).T
    finite_corr = np.all(np.isfinite(corr_inputs), axis=1)
    corr = float(np.corrcoef(abs_error[finite_corr], sigma_rms[finite_corr])[0, 1]) if np.count_nonzero(finite_corr) > 2 else None
    summary = {
        "run_dir": str(run_dir),
        "pipeline_dir": str(pipeline_dir),
        "zdim": int(args.zdim),
        "embedding_kind": args.embedding_kind,
        "z_path": str(z_path),
        "precision_path": str(precision_path),
        "metadata_json": str(metadata_path) if metadata_path else None,
        "true_latent_source": true_source,
        "n_images": int(n),
        "n_phi_bins": int(args.n_phi),
        "n_theta_bins": int(args.n_theta),
        "angle_convention": {
            "relion_names": "rlnAngleRot and rlnAngleTilt define viewing orientation; rlnAnglePsi is in-plane",
            "map_coordinates": "phi/theta are computed from RECOVAR viewing vector [0,0,1] @ R, so in-plane rotation is ignored",
        },
        "embedding_error": _finite_stats(signed_error),
        "embedding_abs_error": _finite_stats(abs_error),
        "latent_variance_trace": _finite_stats(variance_trace),
        "latent_sigma_rms": _finite_stats(sigma_rms),
        "latent_precision_trace": _finite_stats(precision_trace),
        "abs_error_vs_sigma_correlation": corr,
        "outputs": {
            "per_image_npz": str(out_dir / "pose_embedding_per_image_stats.npz"),
            "maps_png": str(out_dir / "pose_embedding_error_covariance_maps.png"),
            "scatter_png": str(out_dir / "pose_embedding_error_covariance_scatter.png"),
        },
    }
    (out_dir / "pose_embedding_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
