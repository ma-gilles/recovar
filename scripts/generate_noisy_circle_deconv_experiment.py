"""Generate a noisy-circle toy benchmark for deconvolved kernel regression.

The experiment is intentionally small and explicit:

1. Build a circular 1D trajectory of synthetic 3D density volumes.
2. Sample particle images from that trajectory with a non-uniform state prior.
3. Save the generated images in RECOVAR-compatible format:
   ``particles.<D>.mrcs``, ``particles.star``, ``poses.pkl``, ``ctf.pkl``,
   and ``simulation_info.pkl``.
4. Add large Gaussian noise to the scalar angle coordinate.
5. Reconstruct target states with both standard noisy-coordinate kernel
   regression and Gaussian-error deconvolved kernel regression.
6. Score all candidates against the known target volume.

Example
-------
pixi run python scripts/generate_noisy_circle_deconv_experiment.py \
  --output-dir /scratch/gpfs/GILLES/mg6942/runs/noisy_circle_deconv_toy \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import recovar.jax_config  # noqa: F401 - initialize JAX config before RECOVAR modules use it
from recovar import utils
from recovar.data_io import cryoem_dataset
from recovar.heterogeneity import deconvolved_kernel_regression
from recovar.heterogeneity import kernel_regression_reconstruction
from recovar.simulation import simulator

logger = logging.getLogger(__name__)


def _parse_float_grid(value: str | None, default: np.ndarray) -> np.ndarray:
    if value is None or value.strip() == "":
        return default.astype(np.float32)
    arr = np.asarray([float(part) for part in value.split(",") if part.strip()], dtype=np.float32)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"Expected a comma-separated list of positive finite values, got {value!r}")
    return arr


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _gaussian_blob(grid, center, sigma):
    diff2 = sum((axis - coord) ** 2 for axis, coord in zip(grid, center))
    return np.exp(-0.5 * diff2 / (sigma**2))


def make_circle_volume(theta: float, grid_size: int) -> np.ndarray:
    """Return a simple positive 3D density whose moving lobes trace a circle."""
    coords = np.linspace(-1.0, 1.0, grid_size, endpoint=False, dtype=np.float32)
    grid = np.meshgrid(coords, coords, coords, indexing="ij")

    c = float(np.cos(theta))
    s = float(np.sin(theta))
    volume = 0.95 * _gaussian_blob(grid, (0.0, 0.0, 0.0), 0.24)
    volume += 0.60 * _gaussian_blob(grid, (0.48 * c, 0.48 * s, 0.16 * np.sin(2.0 * theta)), 0.12)
    volume += 0.42 * _gaussian_blob(grid, (-0.40 * c, -0.40 * s, -0.14 * np.cos(theta)), 0.11)
    volume += 0.18 * _gaussian_blob(grid, (0.22 * np.cos(theta + 1.7), -0.25, 0.24 * s), 0.10)
    volume -= float(volume.min())
    norm = float(np.linalg.norm(volume.reshape(-1)))
    if norm > 0:
        volume /= norm
    return volume.astype(np.float32)


def make_nonuniform_state_distribution(thetas: np.ndarray) -> np.ndarray:
    """A fixed von-Mises mixture on the circle, discretized over state angles."""
    centers = np.asarray([0.20, 0.54, 1.34], dtype=np.float64) * (2.0 * np.pi)
    kappas = np.asarray([8.0, 14.0, 5.0], dtype=np.float64)
    weights = np.asarray([0.42, 0.38, 0.20], dtype=np.float64)

    density = np.zeros_like(thetas, dtype=np.float64)
    for weight, kappa, center in zip(weights, kappas, centers):
        density += weight * np.exp(kappa * np.cos(thetas - center))
    density += 0.02 * float(np.max(density))
    density /= np.sum(density)
    return density.astype(np.float64)


def write_circle_volumes(out_dir: Path, grid_size: int, n_states: int, voxel_size: float) -> tuple[Path, np.ndarray]:
    volume_dir = out_dir / "00_volumes"
    volume_dir.mkdir(parents=True, exist_ok=True)
    volume_prefix = volume_dir / "vol"
    state_thetas = np.linspace(0.0, 2.0 * np.pi, n_states, endpoint=False, dtype=np.float64)
    for state_idx, theta in enumerate(state_thetas):
        volume = make_circle_volume(float(theta), grid_size)
        utils.write_mrc(str(volume_dir / f"vol{state_idx:04d}.mrc"), volume, voxel_size=voxel_size)
    return volume_prefix, state_thetas


def generate_dataset(args, output_dir: Path, volume_prefix: Path, state_distribution: np.ndarray) -> tuple[Path, dict]:
    dataset_dir = output_dir / "01_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(args.seed))
    logger.info("Generating RECOVAR-format synthetic dataset in %s", dataset_dir)
    _image_stack, sim_info = simulator.generate_synthetic_dataset(
        str(dataset_dir),
        float(args.voxel_size),
        str(volume_prefix),
        int(args.n_images),
        outlier_file_input=None,
        grid_size=int(args.grid_size),
        volume_distribution=state_distribution,
        dataset_params_option=str(args.dataset_params_option),
        noise_level=float(args.image_noise_level),
        noise_model=str(args.image_noise_model),
        put_extra_particles=False,
        percent_outliers=0.0,
        volume_radius=0.70,
        trailing_zero_format_in_vol_name=True,
        noise_scale_std=0.0,
        contrast_std=0.0,
        disc_type=str(args.disc_type),
        image_dtype=np.float32,
        per_particle_contrast=True,
        premultiplied_ctf=False,
    )
    return dataset_dir, sim_info


def load_generated_dataset(args, dataset_dir: Path, sim_info: dict):
    ds = cryoem_dataset.load_dataset(
        str(dataset_dir / f"particles.{args.grid_size}.mrcs"),
        poses_file=str(dataset_dir / "poses.pkl"),
        ctf_file=str(dataset_dir / "ctf.pkl"),
        lazy=True,
        premultiplied_ctf=False,
    )
    ds.set_radial_noise_model(np.asarray(sim_info["noise_variance"], dtype=np.float32))
    return ds


def make_noisy_latents(args, output_dir: Path, sim_info: dict, state_thetas: np.ndarray) -> dict:
    assignments = np.asarray(sim_info["image_assignment"], dtype=np.int64).reshape(-1)
    if np.any(assignments < 0):
        raise ValueError("This toy script expects no outlier assignments.")

    true_theta = state_thetas[assignments].astype(np.float32)
    true_circle_xy = np.stack([np.cos(true_theta), np.sin(true_theta)], axis=1).astype(np.float32)

    rng = np.random.default_rng(int(args.seed) + 1)
    observed_theta = true_theta + rng.normal(0.0, float(args.latent_noise_std), size=true_theta.shape).astype(np.float32)
    observed_circle_xy = true_circle_xy + rng.normal(
        0.0,
        float(args.circle_xy_noise_std),
        size=true_circle_xy.shape,
    ).astype(np.float32)
    latent_precision = np.full(
        true_theta.shape,
        1.0 / max(float(args.latent_noise_std) ** 2, np.finfo(np.float32).tiny),
        dtype=np.float32,
    )

    target_fractions = np.linspace(0.15, 0.85, int(args.n_targets), dtype=np.float64)
    target_state_indices = np.unique(np.round(target_fractions * (len(state_thetas) - 1)).astype(np.int32))
    target_theta = state_thetas[target_state_indices].astype(np.float32)

    latent_dir = output_dir / "02_latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        latent_dir / "noisy_circle_latents.npz",
        state_assignment=assignments.astype(np.int32),
        state_thetas=state_thetas.astype(np.float32),
        true_theta=true_theta,
        observed_theta=observed_theta,
        latent_precision=latent_precision,
        true_circle_xy=true_circle_xy,
        observed_circle_xy=observed_circle_xy,
        target_state_indices=target_state_indices,
        target_theta=target_theta,
    )
    np.savetxt(latent_dir / "observed_theta.txt", observed_theta.reshape(-1, 1))
    np.savetxt(latent_dir / "latent_precision.txt", latent_precision.reshape(-1, 1))
    np.savetxt(latent_dir / "target_theta.txt", target_theta.reshape(-1, 1))
    np.save(latent_dir / "state_assignment.npy", assignments.astype(np.int32))

    return {
        "assignments": assignments,
        "true_theta": true_theta,
        "observed_theta": observed_theta,
        "latent_precision": latent_precision,
        "true_circle_xy": true_circle_xy,
        "observed_circle_xy": observed_circle_xy,
        "target_state_indices": target_state_indices,
        "target_theta": target_theta,
        "latent_dir": latent_dir,
    }


def _strictly_increasing(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = np.sort(values[np.isfinite(values)])
    values = values[values > 0]
    if values.size == 0:
        raise ValueError("Could not build a positive candidate grid.")
    eps = np.finfo(np.float32).eps
    out = values.copy()
    for idx in range(1, out.size):
        if out[idx] <= out[idx - 1]:
            out[idx] = np.nextafter(out[idx - 1], np.float32(np.inf)) + eps
    return out


def standard_distance_bins(distance: np.ndarray, n_bins: int, q_min: float, q_max: float) -> np.ndarray:
    quantiles = np.linspace(float(q_min), float(q_max), int(n_bins), dtype=np.float64)
    bins = np.quantile(np.asarray(distance, dtype=np.float64), quantiles)
    return _strictly_increasing(bins.astype(np.float32))


def score_candidates(candidates: np.ndarray, target: np.ndarray) -> list[dict]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    target_norm2 = float(np.dot(target, target))
    if target_norm2 <= 0:
        raise ValueError("Target volume has zero norm.")

    rows = []
    for candidate_idx, candidate in enumerate(np.asarray(candidates)):
        est = np.asarray(candidate, dtype=np.float64).reshape(-1)
        err = est - target
        est_norm2 = float(np.dot(est, est))
        dot = float(np.dot(est, target))
        corr = dot / np.sqrt(max(est_norm2 * target_norm2, np.finfo(np.float64).tiny))
        scale = dot / max(est_norm2, np.finfo(np.float64).tiny)
        scaled_err = scale * est - target
        rows.append(
            {
                "candidate_index": int(candidate_idx),
                "mse": float(np.mean(err**2)),
                "relative_mse": float(np.dot(err, err) / target_norm2),
                "scaled_relative_mse": float(np.dot(scaled_err, scaled_err) / target_norm2),
                "correlation": float(corr),
                "least_squares_scale": float(scale),
            }
        )
    return rows


def _best_row(rows: list[dict]) -> dict:
    return min(rows, key=lambda row: row["scaled_relative_mse"])


def _write_candidate_volumes(
    out_dir: Path,
    candidates: np.ndarray,
    grid_size: int,
    voxel_size: float,
    best_index: int,
    save_all_candidates: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_all_candidates:
        for idx, candidate in enumerate(candidates):
            utils.write_mrc(
                str(out_dir / f"candidate_{idx:03d}.mrc"),
                np.asarray(candidate, dtype=np.float32).reshape((grid_size, grid_size, grid_size)),
                voxel_size=voxel_size,
            )
    utils.write_mrc(
        str(out_dir / "best.mrc"),
        np.asarray(candidates[best_index], dtype=np.float32).reshape((grid_size, grid_size, grid_size)),
        voxel_size=voxel_size,
    )


def run_reconstructions(args, output_dir: Path, ds, latents: dict, state_thetas: np.ndarray) -> tuple[list[dict], dict]:
    lambda_grid = _parse_float_grid(
        args.deconv_lambda_grid,
        deconvolved_kernel_regression.DEFAULT_DECONV_LAMBDA_GRID,
    )
    recon_dir = output_dir / "03_reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    all_metric_rows = []
    target_summaries = {}
    scale_vol = float(args.simulated_volume_scale)

    for target_idx, state_idx in enumerate(latents["target_state_indices"]):
        target_theta = float(state_thetas[int(state_idx)])
        target_dir = recon_dir / f"target_{target_idx:03d}_state_{int(state_idx):04d}"
        target_dir.mkdir(parents=True, exist_ok=True)

        target_gt = make_circle_volume(target_theta, int(args.grid_size)).reshape(-1) * scale_vol
        utils.write_mrc(
            str(target_dir / "target_gt.mrc"),
            target_gt.reshape((args.grid_size, args.grid_size, args.grid_size)).astype(np.float32),
            voxel_size=float(args.voxel_size),
        )

        latent_diff = np.asarray(latents["observed_theta"] - target_theta, dtype=np.float32)
        latent_precision = np.asarray(latents["latent_precision"], dtype=np.float32)
        standard_distance = np.square(latent_diff) * latent_precision
        bins = standard_distance_bins(
            standard_distance,
            int(args.n_standard_bins),
            float(args.standard_quantile_min),
            float(args.standard_quantile_max),
        )

        logger.info(
            "Target %d/%d: state=%d theta=%.4f standard_bins=%s lambda_grid=%s",
            target_idx + 1,
            len(latents["target_state_indices"]),
            int(state_idx),
            target_theta,
            bins,
            lambda_grid,
        )

        standard_estimates = kernel_regression_reconstruction.estimate_standard_kernel_volumes(
            ds,
            standard_distance,
            bins,
            batch_size=int(args.batch_size),
            tau=args.tau,
            grid_correct=bool(args.grid_correct),
            disc_type=str(args.disc_type),
            use_spherical_mask=bool(args.use_spherical_mask),
            return_lhs_rhs=False,
            heterogeneity_kernel=str(args.standard_kernel),
            upsampling_factor=1,
            return_real_space=True,
            use_fast_rfft=bool(args.use_fast_rfft),
        )
        standard_rows = score_candidates(standard_estimates, target_gt)
        for row in standard_rows:
            row.update(
                {
                    "target_index": int(target_idx),
                    "target_state_index": int(state_idx),
                    "target_theta": target_theta,
                    "method": "standard",
                    "grid_value": float(bins[row["candidate_index"]]),
                    "grid_name": "distance_bin",
                }
            )

        deconv_estimates = deconvolved_kernel_regression.estimate_deconvolved_kernel_volumes(
            ds,
            latent_diff,
            latent_precision,
            lambda_grid=lambda_grid,
            batch_size=int(args.batch_size),
            tau=args.tau,
            grid_correct=bool(args.grid_correct),
            disc_type=str(args.disc_type),
            use_spherical_mask=bool(args.use_spherical_mask),
            return_lhs_rhs=False,
            upsampling_factor=1,
            return_real_space=True,
            use_fast_rfft=bool(args.use_fast_rfft),
            lambda_batch_size=args.lambda_batch_size,
        )
        deconv_rows = score_candidates(deconv_estimates, target_gt)
        for row in deconv_rows:
            row.update(
                {
                    "target_index": int(target_idx),
                    "target_state_index": int(state_idx),
                    "target_theta": target_theta,
                    "method": "deconvolved",
                    "grid_value": float(lambda_grid[row["candidate_index"]]),
                    "grid_name": "lambda",
                }
            )

        best_standard = _best_row(standard_rows)
        best_deconv = _best_row(deconv_rows)
        _write_candidate_volumes(
            target_dir / "standard",
            standard_estimates,
            int(args.grid_size),
            float(args.voxel_size),
            int(best_standard["candidate_index"]),
            bool(args.save_all_candidates),
        )
        _write_candidate_volumes(
            target_dir / "deconvolved",
            deconv_estimates,
            int(args.grid_size),
            float(args.voxel_size),
            int(best_deconv["candidate_index"]),
            bool(args.save_all_candidates),
        )
        np.savetxt(target_dir / "standard_bins.txt", bins.reshape(-1, 1))
        np.savetxt(target_dir / "deconv_lambda_grid.txt", lambda_grid.reshape(-1, 1))

        target_summaries[str(target_idx)] = {
            "target_state_index": int(state_idx),
            "target_theta": target_theta,
            "best_standard": best_standard,
            "best_deconvolved": best_deconv,
        }
        logger.info(
            "Target %d best scaled rel-MSE: standard=%.4g deconvolved=%.4g",
            target_idx,
            best_standard["scaled_relative_mse"],
            best_deconv["scaled_relative_mse"],
        )
        all_metric_rows.extend(standard_rows)
        all_metric_rows.extend(deconv_rows)

    return all_metric_rows, target_summaries


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_index",
        "target_state_index",
        "target_theta",
        "method",
        "grid_name",
        "grid_value",
        "candidate_index",
        "mse",
        "relative_mse",
        "scaled_relative_mse",
        "correlation",
        "least_squares_scale",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def run(args) -> dict:
    output_dir = Path(args.output_dir).resolve()
    _prepare_output_dir(output_dir, bool(args.overwrite))
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    start = time.monotonic()

    utils.set_gpu_memory_limit(float(args.memory_gb))

    volume_prefix, state_thetas = write_circle_volumes(
        output_dir,
        int(args.grid_size),
        int(args.n_states),
        float(args.voxel_size),
    )
    state_distribution = make_nonuniform_state_distribution(state_thetas)
    np.save(output_dir / "00_volumes" / "state_thetas.npy", state_thetas.astype(np.float32))
    np.save(output_dir / "00_volumes" / "state_distribution.npy", state_distribution.astype(np.float32))

    dataset_dir, sim_info = generate_dataset(args, output_dir, volume_prefix, state_distribution)
    args.simulated_volume_scale = float(sim_info["scale_vol"])
    latents = make_noisy_latents(args, output_dir, sim_info, state_thetas)

    metric_rows = []
    target_summaries = {}
    if not args.skip_reconstruction:
        ds = load_generated_dataset(args, dataset_dir, sim_info)
        metric_rows, target_summaries = run_reconstructions(args, output_dir, ds, latents, state_thetas)
        write_metrics_csv(output_dir / "metrics.csv", metric_rows)

    summary = {
        "output_dir": output_dir,
        "dataset_dir": dataset_dir,
        "particles": dataset_dir / f"particles.{args.grid_size}.mrcs",
        "particles_star": dataset_dir / "particles.star",
        "poses": dataset_dir / "poses.pkl",
        "ctf": dataset_dir / "ctf.pkl",
        "simulation_info": dataset_dir / "simulation_info.pkl",
        "latent_npz": latents["latent_dir"] / "noisy_circle_latents.npz",
        "metrics_csv": output_dir / "metrics.csv",
        "state_distribution": state_distribution,
        "target_summaries": target_summaries,
        "args": vars(args),
        "wall_seconds": round(time.monotonic() - start, 2),
        "reproduce": {
            "command": "pixi run python scripts/generate_noisy_circle_deconv_experiment.py "
            f"--output-dir {output_dir} --overwrite",
            "direct_function": "recovar.heterogeneity.deconvolved_kernel_regression.estimate_deconvolved_kernel_volumes",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    logger.info("Wrote summary to %s", output_dir / "summary.json")
    return summary


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--n-states", type=int, default=64)
    parser.add_argument("--n-images", type=int, default=4000)
    parser.add_argument("--n-targets", type=int, default=5)
    parser.add_argument("--voxel-size", type=float, default=4.25)
    parser.add_argument("--image-noise-level", type=float, default=6.0)
    parser.add_argument("--image-noise-model", choices=("white", "radial1"), default="white")
    parser.add_argument("--latent-noise-std", type=float, default=1.25)
    parser.add_argument("--circle-xy-noise-std", type=float, default=0.75)
    parser.add_argument("--dataset-params-option", default="noctf")
    parser.add_argument("--disc-type", default="linear_interp")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-gb", type=float, default=8.0)
    parser.add_argument("--n-standard-bins", type=int, default=10)
    parser.add_argument("--standard-quantile-min", type=float, default=0.02)
    parser.add_argument("--standard-quantile-max", type=float, default=0.40)
    parser.add_argument("--standard-kernel", choices=("square", "parabola", "triangle"), default="parabola")
    parser.add_argument("--deconv-lambda-grid", default=None)
    parser.add_argument("--lambda-batch-size", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--grid-correct", action="store_true")
    parser.add_argument("--use-spherical-mask", action="store_true")
    parser.add_argument("--use-fast-rfft", action="store_true")
    parser.add_argument("--save-all-candidates", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def validate_args(args) -> None:
    if args.grid_size <= 0 or args.grid_size % 2 != 0:
        raise ValueError(f"--grid-size must be a positive even integer, got {args.grid_size}")
    if args.n_states < 4:
        raise ValueError("--n-states must be at least 4")
    if args.n_images <= 0:
        raise ValueError("--n-images must be positive")
    if args.n_targets <= 0:
        raise ValueError("--n-targets must be positive")
    if args.image_noise_level < 0:
        raise ValueError("--image-noise-level must be nonnegative")
    if args.latent_noise_std <= 0 or not np.isfinite(args.latent_noise_std):
        raise ValueError("--latent-noise-std must be finite and positive")
    if args.circle_xy_noise_std < 0 or not np.isfinite(args.circle_xy_noise_std):
        raise ValueError("--circle-xy-noise-std must be finite and nonnegative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.memory_gb <= 0:
        raise ValueError("--memory-gb must be positive")
    if args.n_standard_bins <= 0:
        raise ValueError("--n-standard-bins must be positive")
    if not (0.0 < args.standard_quantile_min < args.standard_quantile_max < 1.0):
        raise ValueError("--standard-quantile-min/max must satisfy 0 < min < max < 1")


def main(argv: list[str] | None = None) -> int:
    parser = add_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    validate_args(args)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
