"""Build a small-motion synthetic dataset and two compute_state references.

This is intended as a self-contained sandbox for experimenting with
``compute_state`` kernel-regression changes.  It generates a simple 1D PDB
trajectory, simulates low-noise/no-contrast/no-CTF particle images, runs the
standard RECOVAR pipeline, then runs ``compute_state`` twice on the middle
trajectory state:

1. using RECOVAR's own zdim=1 embedding
2. using a ground-truth PC1 embedding written into a cloned pipeline output

Example
-------
pixi run python scripts/run_compute_state_deconv_sandbox.py \
  --output-dir /scratch/gpfs/GILLES/mg6942/runs/deconv_compute_state_sandbox
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import recovar.jax_config  # noqa: F401 - initialize JAX config before RECOVAR modules use it
from recovar import utils
from recovar.commands import compute_state, pipeline
from recovar.commands.run_test_all_metrics import setup_logging
from recovar.output import metrics, output
from recovar.output.output_paths import ResultPaths
from recovar.simulation import simulator, synthetic_dataset
from recovar.simulation.trajectory_generation import generate_trajectory_volumes

logger = logging.getLogger(__name__)


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


def _hardlink_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _clone_pipeline_output(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing cloned pipeline output: {dst}")
    shutil.copytree(src, dst, copy_function=_hardlink_or_copy)


def _count_volume_files(prefix: Path) -> int:
    count = 0
    while Path(f"{prefix}{count:04d}.mrc").exists():
        count += 1
    return count


def _make_uniform_distribution(n_states: int) -> np.ndarray:
    if n_states <= 0:
        raise ValueError(f"n_states must be positive, got {n_states}")
    return np.full(n_states, 1.0 / n_states, dtype=np.float64)


def _generate_dataset(args) -> tuple[Path, Path, dict]:
    output_dir = Path(args.output_dir)
    generated_prefix = output_dir / "generated_volumes" / "vol"
    dataset_dir = output_dir / "test_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating small-motion PDB trajectory: n_states=%d grid=%d max_rotation=%.3f",
        args.n_states,
        args.grid_size,
        args.max_rotation_degrees,
    )
    volume_prefix = Path(
        generate_trajectory_volumes(
            output_dir=str(output_dir),
            grid_size=args.grid_size,
            n_volumes=args.n_states,
            voxel_size=args.voxel_size,
            Bfactor=args.bfactor,
            max_rotation_degrees=args.max_rotation_degrees,
            pdb_path=args.pdb_path,
            prefix_name=generated_prefix.name,
            output_prefix=str(generated_prefix),
        )
    )

    n_states = _count_volume_files(volume_prefix)
    volume_distribution = _make_uniform_distribution(n_states)
    logger.info(
        "Simulating low-noise/no-contrast/no-CTF images: n_images=%d noise=%.4g",
        args.n_images,
        args.noise_level,
    )
    np.random.seed(args.seed)
    _image_stack, sim_info = simulator.generate_synthetic_dataset(
        str(dataset_dir),
        args.voxel_size,
        str(volume_prefix),
        int(args.n_images),
        outlier_file_input=None,
        grid_size=args.grid_size,
        volume_distribution=volume_distribution,
        dataset_params_option="noctf",
        noise_level=args.noise_level,
        noise_model="white",
        put_extra_particles=False,
        percent_outliers=0.0,
        volume_radius=0.7,
        trailing_zero_format_in_vol_name=True,
        noise_scale_std=0.0,
        contrast_std=0.0,
        disc_type="cubic",
        image_dtype=np.float32,
        per_particle_contrast=True,
        premultiplied_ctf=False,
    )
    return dataset_dir, volume_prefix, sim_info


def _write_gt_union_mask(dataset_dir: Path, sim_info_path: Path, grid_size: int, voxel_size: float) -> Path:
    mask_dir = dataset_dir / "gt_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / "gt_union_mask.mrc"

    gt_hvd = synthetic_dataset.load_heterogeneous_reconstruction(str(sim_info_path))
    soft_mask, binary_mask = metrics.make_union_gt_mask_from_hvd(gt_hvd, (grid_size, grid_size, grid_size))
    utils.write_mrc(mask_path, soft_mask.astype(np.float32), voxel_size=voxel_size)
    logger.info("Wrote GT union mask to %s (%.2f%% voxels)", mask_path, 100.0 * binary_mask.mean())
    return mask_path


def _run_pipeline(args, dataset_dir: Path, mask_path: Path) -> Path:
    pipeline_output_dir = dataset_dir / "pipeline_output"
    pipeline_cmd = [
        str(dataset_dir / f"particles.{args.grid_size}.mrcs"),
        "--poses",
        str(dataset_dir / "poses.pkl"),
        "--ctf",
        str(dataset_dir / "ctf.pkl"),
        "-o",
        str(pipeline_output_dir),
        "--mask",
        str(mask_path),
        "--noise-model",
        "radial",
        "--zdim",
        "1",
        "--ignore-zero-frequency",
    ]
    if args.accept_cpu:
        pipeline_cmd.append("--accept-cpu")
    if args.low_memory_option:
        pipeline_cmd.append("--low-memory-option")

    parser = pipeline.add_args(argparse.ArgumentParser())
    pipeline_args = parser.parse_args(pipeline_cmd)
    logger.info("Running pipeline as: recovar pipeline %s", " ".join(pipeline_cmd))
    pipeline.standard_recovar_pipeline(pipeline_args)
    return pipeline_output_dir


def _middle_state_target_from_embedding(pipeline_output_dir: Path, sim_info: dict, middle_state: int) -> np.ndarray:
    po = output.PipelineOutput(str(pipeline_output_dir))
    raw_zs = np.asarray(po.get_unsorted_embedding_component("latent_coords", 1), dtype=np.float32)
    assignments = np.asarray(sim_info["image_assignment"], dtype=np.int64).reshape(-1)
    mask = assignments == int(middle_state)
    if not np.any(mask):
        raise ValueError(f"No particles were assigned to middle state {middle_state}")
    target = np.mean(raw_zs[mask], axis=0, dtype=np.float64).astype(np.float32)
    return np.atleast_1d(target)


def _load_real_trajectory_volumes(volume_prefix: Path, n_states: int) -> np.ndarray:
    volumes = []
    for idx in range(n_states):
        path = Path(f"{volume_prefix}{idx:04d}.mrc")
        if not path.exists():
            raise FileNotFoundError(f"Missing trajectory volume: {path}")
        volumes.append(np.asarray(utils.load_mrc(path), dtype=np.float32).reshape(-1))
    return np.stack(volumes, axis=0)


def _compute_gt_pc1_embedding(volume_prefix: Path, sim_info: dict, n_states: int) -> tuple[np.ndarray, np.ndarray, dict]:
    volumes = _load_real_trajectory_volumes(volume_prefix, n_states)
    assignments = np.asarray(sim_info["image_assignment"], dtype=np.int64).reshape(-1)
    centered = volumes - np.mean(volumes, axis=0, keepdims=True)
    _u, s, vh = np.linalg.svd(centered, full_matrices=False)
    state_scores = (centered @ vh[0]).astype(np.float32)
    if state_scores[-1] < state_scores[0]:
        state_scores *= -1.0
        vh[0] *= -1.0

    if np.any(assignments < 0) or np.any(assignments >= n_states):
        raise ValueError("GT PC embedding expects all assignments to reference generated trajectory states.")
    particle_scores = state_scores[assignments][:, None].astype(np.float32)
    explained = np.square(s) / max(float(np.sum(np.square(s))), np.finfo(np.float32).eps)
    meta = {
        "state_scores": state_scores,
        "singular_values": s[:10].astype(np.float64),
        "explained_variance_head": explained[:10].astype(np.float64),
        "pc1_explained_variance": float(explained[0]) if explained.size else None,
    }
    return particle_scores, state_scores, meta


def _write_gt_pc_pipeline_output(
    src_pipeline_output_dir: Path,
    dst_pipeline_output_dir: Path,
    gt_particle_zs: np.ndarray,
    gt_covariance: float,
) -> None:
    _clone_pipeline_output(src_pipeline_output_dir, dst_pipeline_output_dir)

    paths = ResultPaths(str(dst_pipeline_output_dir))
    zdim_dir = Path(paths.embedding_zdim_dir(1))
    zdim_dir.mkdir(parents=True, exist_ok=True)

    n_particles = int(gt_particle_zs.shape[0])
    cov_zs = np.repeat(
        (np.eye(1, dtype=np.float32) * np.float32(gt_covariance))[None, :, :],
        n_particles,
        axis=0,
    )
    contrasts = np.ones(n_particles, dtype=np.float32)
    replacements = {
        "latent_coords": gt_particle_zs,
        "latent_coords_noreg": gt_particle_zs,
        "latent_precision": cov_zs,
        "latent_precision_noreg": cov_zs,
        "contrasts": contrasts,
        "contrasts_noreg": contrasts,
    }
    for field, arr in replacements.items():
        path = zdim_dir / f"{field}.npy"
        if path.exists():
            path.unlink()
        np.save(path, np.asarray(arr, dtype=np.float32))

    logger.info("Wrote GT PC1 embedding into cloned pipeline output: %s", dst_pipeline_output_dir)


def _run_compute_state(
    result_dir: Path,
    outdir: Path,
    latent_points_path: Path,
    args,
) -> None:
    cs_args = argparse.Namespace(
        result_dir=str(result_dir),
        outdir=str(outdir),
        project=None,
        Bfactor=float(args.compute_state_bfactor),
        n_bins=int(args.n_bins),
        maskrad_fraction=float(args.maskrad_fraction),
        n_min_particles=int(args.n_min_particles),
        zdim1=True,
        no_z_regularization=False,
        lazy=True,
        particles=None,
        datadir=None,
        strip_prefix=None,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
        latent_points=str(latent_points_path),
        save_all_estimates=bool(args.save_all_estimates),
    )
    logger.info("Running compute_state: result_dir=%s outdir=%s target=%s", result_dir, outdir, latent_points_path)
    compute_state.compute_state(cs_args)


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run(args) -> dict:
    output_dir = Path(args.output_dir).resolve()
    _prepare_output_dir(output_dir, args.overwrite)
    setup_logging(str(output_dir))
    logger.info("Writing sandbox output to %s", output_dir)

    start = time.monotonic()
    dataset_dir, volume_prefix, sim_info = _generate_dataset(args)
    sim_info_path = dataset_dir / "simulation_info.pkl"
    mask_path = _write_gt_union_mask(dataset_dir, sim_info_path, args.grid_size, args.voxel_size)
    pipeline_output_dir = _run_pipeline(args, dataset_dir, mask_path)

    middle_state = int(args.target_state) if args.target_state is not None else int(args.n_states // 2)
    if not (0 <= middle_state < args.n_states):
        raise ValueError(f"--target-state must be in [0, {args.n_states - 1}], got {middle_state}")

    recovar_target = _middle_state_target_from_embedding(pipeline_output_dir, sim_info, middle_state)
    recovar_target_path = output_dir / "latent_points_recovar_middle.txt"
    np.savetxt(recovar_target_path, recovar_target.reshape(1, -1))
    _run_compute_state(
        pipeline_output_dir,
        output_dir / "compute_state_recovar_embedding",
        recovar_target_path,
        args,
    )

    gt_particle_zs, gt_state_scores, gt_pc_meta = _compute_gt_pc1_embedding(volume_prefix, sim_info, args.n_states)
    gt_pipeline_output_dir = dataset_dir / "pipeline_output_gt_pc_embedding"
    _write_gt_pc_pipeline_output(
        pipeline_output_dir,
        gt_pipeline_output_dir,
        gt_particle_zs,
        gt_covariance=args.gt_pc_covariance,
    )
    gt_target = np.asarray([gt_state_scores[middle_state]], dtype=np.float32)
    gt_target_path = output_dir / "latent_points_gt_pc_middle.txt"
    np.savetxt(gt_target_path, gt_target.reshape(1, -1))
    _run_compute_state(
        gt_pipeline_output_dir,
        output_dir / "compute_state_gt_pc_embedding",
        gt_target_path,
        args,
    )

    embedding_dir = output_dir / "gt_pc_embedding"
    embedding_dir.mkdir(exist_ok=True)
    with (embedding_dir / "embedding.pkl").open("wb") as f:
        pickle.dump(gt_particle_zs, f)
    np.save(embedding_dir / "particle_zs.npy", gt_particle_zs)
    np.save(embedding_dir / "state_scores.npy", gt_state_scores)
    _write_json(embedding_dir / "metadata.json", gt_pc_meta)

    summary = {
        "output_dir": output_dir,
        "dataset_dir": dataset_dir,
        "volume_prefix": str(volume_prefix),
        "pipeline_output_dir": pipeline_output_dir,
        "gt_pc_pipeline_output_dir": gt_pipeline_output_dir,
        "compute_state_recovar_output_dir": output_dir / "compute_state_recovar_embedding",
        "compute_state_gt_pc_output_dir": output_dir / "compute_state_gt_pc_embedding",
        "recovar_latent_points": recovar_target_path,
        "gt_pc_latent_points": gt_target_path,
        "gt_pc_embedding_dir": embedding_dir,
        "middle_state": middle_state,
        "recovar_target": recovar_target,
        "gt_pc_target": gt_target,
        "gt_pc": gt_pc_meta,
        "args": vars(args),
        "wall_seconds": round(time.monotonic() - start, 2),
    }
    _write_json(output_dir / "summary.json", summary)
    logger.info("Sandbox complete. Summary: %s", output_dir / "summary.json")
    return summary


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-states", type=int, default=81)
    parser.add_argument("--n-images", type=int, default=20000)
    parser.add_argument("--noise-level", type=float, default=0.02)
    parser.add_argument("--voxel-size", type=float, default=4.25)
    parser.add_argument("--bfactor", type=float, default=80.0)
    parser.add_argument("--max-rotation-degrees", type=float, default=2.0)
    parser.add_argument("--pdb-path", default=None)
    parser.add_argument("--target-state", type=int, default=None)
    parser.add_argument("--gt-pc-covariance", type=float, default=1.0e-4)
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--n-min-particles", type=int, default=100)
    parser.add_argument("--maskrad-fraction", type=float, default=20.0)
    parser.add_argument("--compute-state-bfactor", type=float, default=0.0)
    parser.add_argument("--save-all-estimates", action="store_true")
    parser.add_argument("--low-memory-option", action="store_true")
    parser.add_argument("--accept-cpu", action="store_true")
    return parser


def _validate_args(args) -> None:
    if args.grid_size <= 0:
        raise ValueError(f"--grid-size must be positive, got {args.grid_size}")
    if args.n_states <= 1:
        raise ValueError(f"--n-states must be > 1, got {args.n_states}")
    if args.n_images <= 0:
        raise ValueError(f"--n-images must be positive, got {args.n_images}")
    if args.noise_level < 0:
        raise ValueError(f"--noise-level must be non-negative, got {args.noise_level}")
    if args.gt_pc_covariance <= 0:
        raise ValueError(f"--gt-pc-covariance must be positive, got {args.gt_pc_covariance}")
    if args.n_bins <= 0:
        raise ValueError(f"--n-bins must be positive, got {args.n_bins}")
    if args.n_min_particles <= 0:
        raise ValueError(f"--n-min-particles must be positive, got {args.n_min_particles}")
    if args.maskrad_fraction <= 0:
        raise ValueError(f"--maskrad-fraction must be positive, got {args.maskrad_fraction}")
    if args.target_state is not None and not (0 <= args.target_state < args.n_states):
        raise ValueError(f"--target-state must be in [0, {args.n_states - 1}], got {args.target_state}")


def main(argv: list[str] | None = None) -> int:
    parser = add_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    _validate_args(args)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
