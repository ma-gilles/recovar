"""Standalone benchmark for shellwise kernel-bandwidth selection on a 1D trajectory."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

import recovar.jax_config  # noqa: F401 - ensure JAX config is initialized early
import recovar.core.fourier_transform_utils as ftu
from recovar import utils
from recovar.commands import pipeline
from recovar.data_io import halfsets
from recovar.heterogeneity import latent_density
from recovar.heterogeneity import kernel_bandwidth_benchmark as kb
from recovar.output import metrics, output as o
from recovar.project.job_context import job_context
from recovar.simulation import simulator, synthetic_dataset
from recovar.simulation.trajectory_generation import generate_trajectory_volumes
from recovar.utils.helpers import RobustFileHandler, RobustStreamHandler

logger = logging.getLogger(__name__)


def _toy_trajectory_volumes(n_states: int, grid_size: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    vols = []
    for idx in range(n_states):
        t = 2.0 * np.pi * idx / max(n_states, 1)
        vol = np.exp(
            -((xx - 0.3 * np.cos(t)) ** 2 + (yy - 0.25 * np.sin(t)) ** 2 + (zz - 0.2 * np.cos(2 * t)) ** 2)
            / (2 * 0.18**2)
        ) + 0.7 * np.exp(
            -((xx + 0.25 * np.sin(1.3 * t)) ** 2 + (yy - 0.2 * np.cos(1.1 * t)) ** 2 + (zz + 0.2 * np.sin(t)) ** 2)
            / (2 * 0.16**2)
        )
        vol = vol.astype(np.float32)
        vol -= vol.mean()
        denom = np.linalg.norm(vol.ravel())
        if denom > 0:
            vol /= denom
        vols.append(vol)
    return np.stack(vols, axis=0)


def _load_volume_stack(prefix: str | Path, n_states: int) -> np.ndarray:
    prefix = Path(prefix)
    vols = []
    for idx in range(n_states):
        path = Path(f"{prefix}{idx:04d}.mrc")
        if not path.exists():
            raise FileNotFoundError(f"Expected trajectory volume file {path}")
        vols.append(np.asarray(utils.load_mrc(path), dtype=np.float32))
    return np.stack(vols, axis=0)


def _make_dataset_args(dataset_dir: Path, grid_size: int):
    return argparse.Namespace(
        particles=str(dataset_dir / "particles.star"),
        poses=str(dataset_dir / "poses.pkl"),
        ctf=str(dataset_dir / "ctf.pkl"),
        datadir=None,
        strip_prefix=None,
        tilt_series=False,
        tilt_series_ctf="cryoem",
        angle_per_tilt=None,
        dose_per_tilt=None,
        premultiplied_ctf=False,
        n_images=-1,
        padding=0,
        downsample=grid_size,
        uninvert_data="automatic",
    )


def _write_dataset(
    volume_prefix: str,
    output_dir: Path,
    *,
    voxel_size: float,
    n_images: int,
    grid_size: int,
    volume_distribution: np.ndarray,
    noise_level: float,
    contrast_std: float,
    noise_model: str,
    seed: int,
    premultiplied_ctf: bool,
):
    np.random.seed(seed)
    output_folder = output_dir / "test_dataset"
    output_folder.mkdir(parents=True, exist_ok=True)
    image_stack, sim_info = simulator.generate_synthetic_dataset(
        str(output_folder),
        voxel_size,
        volume_prefix,
        int(n_images),
        outlier_file_input=None,
        grid_size=grid_size,
        volume_distribution=volume_distribution,
        dataset_params_option="uniform",
        noise_level=noise_level,
        noise_model=noise_model,
        put_extra_particles=False,
        percent_outliers=0.0,
        volume_radius=0.7,
        trailing_zero_format_in_vol_name=True,
        noise_scale_std=0.0,
        contrast_std=contrast_std,
        disc_type="cubic",
        n_tilts=-1,
        premultiplied_ctf=premultiplied_ctf,
    )
    return image_stack, sim_info


def _fit_gt_latents(gt_volumes_real: np.ndarray, zdim: int, gt_z_sigma: float) -> tuple[np.ndarray, np.ndarray, dict]:
    pca = kb.fit_volume_pca(gt_volumes_real)
    scores = np.asarray(pca.scores[:, :zdim], dtype=np.float32)
    cov = np.eye(zdim, dtype=np.float32) * np.square(gt_z_sigma)
    cov_zs = np.repeat(cov[None, :, :], scores.shape[0], axis=0)
    meta = {
        "pca": {
            "mean": np.asarray(pca.mean, dtype=np.float32),
            "singular_values": np.asarray(pca.singular_values, dtype=np.float32),
            "explained_energy": np.asarray(pca.explained_energy, dtype=np.float32),
        }
    }
    return scores, cov_zs, meta


def _prepare_trajectory_volumes(args, benchmark_dir: Path, voxel_size: float) -> tuple[str, dict]:
    raw_dir = benchmark_dir / "volumes_raw"
    raw_prefix = raw_dir / "vol"

    trajectory_source = "pdb5nrl" if args.trajectory_source in {"recovar-default", "pdb5nrl"} else args.trajectory_source

    if trajectory_source == "pdb5nrl":
        generate_trajectory_volumes(
            output_dir=str(benchmark_dir),
            grid_size=args.grid_size,
            n_volumes=args.n_states,
            voxel_size=voxel_size,
            Bfactor=args.bfactor,
            max_rotation_degrees=args.max_rotation_degrees,
            pdb_path=args.pdb_path,
            prefix_name="vol",
            output_prefix=str(raw_prefix),
        )
        raw_volumes = _load_volume_stack(raw_prefix, args.n_states)
    elif trajectory_source == "toy":
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_volumes = _toy_trajectory_volumes(args.n_states, args.grid_size)
        kb.write_volume_prefix(raw_volumes, raw_prefix, voxel_size)
    else:
        raise ValueError(f"Unknown trajectory source: {args.trajectory_source}")

    projected_meta = {"n_pcs": 0}
    volume_prefix = str(raw_prefix)
    if args.pc_project > 0:
        projected_dir = benchmark_dir / "volumes_pc_projected"
        projected_prefix = projected_dir / "vol"
        projected_volumes, projected_meta = kb.project_volume_trajectory(raw_volumes, args.pc_project)
        kb.write_volume_prefix(projected_volumes, projected_prefix, voxel_size)
        volume_prefix = str(projected_prefix)

    projection_summary = {"n_pcs": int(projected_meta.get("n_pcs", 0))}
    if "singular_values" in projected_meta:
        projection_summary["singular_values_head"] = np.asarray(projected_meta["singular_values"])[:10].tolist()
    if "explained_energy" in projected_meta:
        projection_summary["explained_energy_head"] = np.asarray(projected_meta["explained_energy"])[:10].tolist()

    return volume_prefix, {
        "raw_prefix": str(raw_prefix),
        "pc_project": int(args.pc_project),
        "projection": projection_summary,
    }


def _run_pipeline_if_needed(args, dataset_dir: Path, sim_info: dict, voxel_size: float) -> str | None:
    if args.embedding_source != "recovar":
        return None

    if args.pipeline_output is not None:
        return str(Path(args.pipeline_output).resolve())

    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_info)
    mask_dir = dataset_dir / "gt_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / "gt_union_mask.mrc"
    soft_mask, binary_mask = metrics.make_union_gt_mask_from_hvd(gt, (args.grid_size, args.grid_size, args.grid_size))
    utils.write_mrc(mask_path, soft_mask.astype(np.float32), voxel_size=voxel_size)
    logger.info("GT union mask written to %s (%.1f%% voxels)", mask_path, 100.0 * binary_mask.mean())

    pipeline_cmd = [
        str(dataset_dir / "particles.star"),
        "--poses",
        str(dataset_dir / "poses.pkl"),
        "--ctf",
        str(dataset_dir / "ctf.pkl"),
        "-o",
        str(dataset_dir / "pipeline_output"),
        "--mask",
        str(mask_path),
        "--noise-model",
        args.noise_model,
    ]
    if args.contrast_std > 0:
        pipeline_cmd.append("--correct-contrast")
    if args.lazy:
        pipeline_cmd.append("--lazy")
    if args.premultiplied_ctf:
        pipeline_cmd.append("--premultiplied-ctf")

    pipeline_parser = pipeline.add_args(argparse.ArgumentParser())
    pipeline_args = pipeline_parser.parse_args(pipeline_cmd)
    logger.info("Running pipeline as: recovar pipeline %s", " ".join(pipeline_cmd))
    pipeline.standard_recovar_pipeline(pipeline_args)
    return str(dataset_dir / "pipeline_output")


def _load_embedding_components(pipeline_output_dir: str, zdim: int, lazy: bool):
    po = o.PipelineOutput(pipeline_output_dir)
    coords_entry = "latent_coords"
    precision_entry = "latent_precision"
    if not lazy:
        dataset = po.get("dataset")
    else:
        dataset = po.get("lazy_dataset")
    zs = po.get_embedding_component(coords_entry, zdim)
    cov_zs = po.get_embedding_component(precision_entry, zdim)
    return dataset, np.asarray(zs, dtype=np.float32), np.asarray(cov_zs, dtype=np.float32)


def _build_particle_latents_from_gt(
    sim_info: dict, gt_scores: np.ndarray, zdim: int, gt_z_sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    image_assignment = np.asarray(sim_info["image_assignment"], dtype=np.int32).reshape(-1)
    zs = gt_scores[image_assignment]
    sigma = np.eye(zdim, dtype=np.float32) * np.square(gt_z_sigma)
    cov_zs = np.repeat(sigma[None, :, :], zs.shape[0], axis=0)
    return zs.astype(np.float32), cov_zs


def _make_target_state(args, n_states: int) -> int:
    if args.target_state is not None:
        if not (0 <= args.target_state < n_states):
            raise ValueError(f"--target-state must be in [0, {n_states - 1}], got {args.target_state}")
        return int(args.target_state)
    return n_states // 2


def run_benchmark(args, benchmark_dir: Path) -> dict:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    voxel_size = 4.25 * 128 / args.grid_size

    state_distribution = kb.make_state_distribution(args.n_states, args.volume_distribution)
    volume_prefix, volume_meta = _prepare_trajectory_volumes(args, benchmark_dir, voxel_size)
    image_stack, sim_info = _write_dataset(
        volume_prefix,
        benchmark_dir,
        voxel_size=voxel_size,
        n_images=args.n_images,
        grid_size=args.grid_size,
        volume_distribution=state_distribution,
        noise_level=args.noise_level,
        contrast_std=args.contrast_std,
        noise_model=args.noise_model,
        seed=args.seed,
        premultiplied_ctf=args.premultiplied_ctf,
    )
    image_stack_shape = list(np.asarray(image_stack).shape)
    del image_stack

    dataset_dir = benchmark_dir / "test_dataset"
    sim_info_path = dataset_dir / "simulation_info.pkl"
    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_info_path)
    gt_shape = tuple(gt.volume_shape)
    gt_volumes_real = np.asarray(
        [
            np.asarray(ftu.get_idft3(gt.volumes[i].reshape(gt_shape)).real, dtype=np.float32)
            for i in range(args.n_states)
        ]
    )
    gt_scores, _gt_cov_zs, gt_meta = _fit_gt_latents(gt_volumes_real, args.zdim, args.gt_z_sigma)

    target_state = _make_target_state(args, args.n_states)
    target_volume = gt_volumes_real[target_state]

    dataset_args = _make_dataset_args(dataset_dir, args.grid_size)
    dataset = halfsets.load_halfset_dataset_from_args(dataset_args, lazy=args.lazy)

    if args.embedding_source == "recovar":
        pipeline_output_dir = _run_pipeline_if_needed(args, dataset_dir, sim_info, voxel_size)
        if pipeline_output_dir is None:
            raise RuntimeError("embedding_source=recovar requires a pipeline output directory")
        dataset, particle_zs, particle_cov_zs = _load_embedding_components(pipeline_output_dir, args.zdim, args.lazy)
        target_assignment = np.asarray(sim_info["image_assignment"], dtype=np.int32).reshape(-1)
        target_mask = target_assignment == target_state
        if np.any(target_mask):
            target_z = np.mean(particle_zs[target_mask], axis=0)
        else:
            target_z = gt_scores[target_state]
    else:
        particle_zs, particle_cov_zs = _build_particle_latents_from_gt(sim_info, gt_scores, args.zdim, args.gt_z_sigma)
        target_z = gt_scores[target_state]
        pipeline_output_dir = None

    particle_zs_halves = dataset.split_halfset_array(np.asarray(particle_zs))
    particle_cov_zs_halves = dataset.split_halfset_array(np.asarray(particle_cov_zs))
    target_pts = np.asarray(target_z, dtype=np.float32).reshape(1, -1)
    quad0 = latent_density.compute_latent_quadratic_forms_in_batch(target_pts, particle_zs_halves[0], particle_cov_zs_halves[0])
    quad1 = latent_density.compute_latent_quadratic_forms_in_batch(target_pts, particle_zs_halves[1], particle_cov_zs_halves[1])
    distances_by_half = [quad0[:, 0], quad1[:, 0]]

    candidate_bins = kb.choose_bandwidth_bins(
        distances_by_half,
        n_bandwidths=args.n_bandwidths,
        n_min_particles=args.n_min_particles,
        q_max=args.q_max,
    )
    estimates_by_half, cv_by_half, lhs_by_half = kb.compute_candidate_estimates(
        dataset,
        distances_by_half,
        candidate_bins,
        batch_size=args.batch_size,
        heterogeneity_kernel=args.heterogeneity_kernel,
    )
    shellwise = kb.compute_shellwise_oracle_and_cv(
        estimates_by_half,
        cv_by_half,
        lhs_by_half,
        target_volume,
        relative_error=True,
    )

    out_dir = benchmark_dir / "bandwidth_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    kb.dump_config(
        benchmark_dir / "config.json",
        {
            "args": vars(args),
            "volume_meta": volume_meta,
            "gt_meta": gt_meta,
            "target_state": int(target_state),
            "target_z": np.asarray(target_z, dtype=np.float32).tolist(),
            "state_distribution": state_distribution.tolist(),
        },
    )
    np.save(out_dir / "candidate_bins.npy", np.asarray(candidate_bins, dtype=np.float32))
    np.save(out_dir / "shell_labels.npy", np.asarray(shellwise["shell_labels"], dtype=np.int32))
    np.save(out_dir / "oracle_error.npy", np.asarray(shellwise["oracle_error"], dtype=np.float32))
    np.save(out_dir / "oracle_error_abs.npy", np.asarray(shellwise["oracle_error_abs"], dtype=np.float32))
    np.save(out_dir / "cv_score.npy", np.asarray(shellwise["cv_score"], dtype=np.float32))
    np.save(out_dir / "cv_score_per_voxel.npy", np.asarray(shellwise["cv_score_per_voxel"], dtype=np.float32))
    np.save(out_dir / "oracle_choice.npy", np.asarray(shellwise["oracle_choice"], dtype=np.int32))
    np.save(out_dir / "cv_choice.npy", np.asarray(shellwise["cv_choice"], dtype=np.int32))
    np.save(out_dir / "regret.npy", np.asarray(shellwise["regret"], dtype=np.float32))
    np.save(out_dir / "particle_zs.npy", np.asarray(particle_zs, dtype=np.float32))
    np.save(out_dir / "particle_cov_zs.npy", np.asarray(particle_cov_zs, dtype=np.float32))
    np.save(out_dir / "target_volume.npy", np.asarray(target_volume, dtype=np.float32))
    np.save(out_dir / "target_z.npy", np.asarray(target_z, dtype=np.float32))

    if args.save_candidate_volumes:
        np.save(out_dir / "candidate_estimates_half0.npy", np.asarray(estimates_by_half[0], dtype=np.float32))
        np.save(out_dir / "candidate_estimates_half1.npy", np.asarray(estimates_by_half[1], dtype=np.float32))
        np.save(out_dir / "cv_estimate_half0.npy", np.asarray(cv_by_half[0], dtype=np.float32))
        np.save(out_dir / "cv_estimate_half1.npy", np.asarray(cv_by_half[1], dtype=np.float32))
        np.save(out_dir / "lhs_half0.npy", np.asarray(lhs_by_half[0], dtype=np.float32))
        np.save(out_dir / "lhs_half1.npy", np.asarray(lhs_by_half[1], dtype=np.float32))

    kb.save_summary_csv(out_dir / "summary.csv", shellwise, candidate_bins)
    kb.save_plots(out_dir / "plots", candidate_bins, shellwise)

    summary = kb.summarize_shellwise_results(shellwise)
    summary["pipeline_output_dir"] = pipeline_output_dir
    summary["benchmark_dir"] = str(benchmark_dir)
    summary["dataset_dir"] = str(dataset_dir)
    summary["target_state"] = int(target_state)
    summary["trajectory_source"] = args.trajectory_source
    summary["embedding_source"] = args.embedding_source
    summary["pc_project"] = int(args.pc_project)
    summary["candidate_bins"] = int(np.asarray(candidate_bins).size)
    summary["image_stack_shape"] = image_stack_shape
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--output-dir", dest="output_dir", type=os.path.abspath, default=None)
    parser.add_argument("--project", type=os.path.abspath, default=None)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-states", type=int, default=50)
    parser.add_argument("--n-images", type=int, default=10000)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--contrast-std", type=float, default=0.1)
    parser.add_argument("--noise-model", choices=["radial1", "radial"], default="radial1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory-source", choices=["recovar-default", "pdb5nrl", "toy"], default="pdb5nrl")
    parser.add_argument("--pc-project", type=int, default=0, help="0 means no projection; otherwise keep first k PCs")
    parser.add_argument("--volume-distribution", choices=["uniform", "vonmises"], default="uniform")
    parser.add_argument("--embedding-source", choices=["gt-pc", "recovar"], default="gt-pc")
    parser.add_argument("--zdim", type=int, default=1)
    parser.add_argument("--gt-z-sigma", type=float, default=0.05)
    parser.add_argument("--target-state", type=int, default=None)
    parser.add_argument("--n-bandwidths", type=int, default=50)
    parser.add_argument("--n-min-particles", type=int, default=200)
    parser.add_argument("--q-max", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--heterogeneity-kernel", choices=["parabola", "triangle", "square", "flat"], default="parabola")
    parser.add_argument("--save-candidate-volumes", action="store_true")
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--premultiplied-ctf", action="store_true")
    parser.add_argument("--bfactor", type=float, default=80.0)
    parser.add_argument("--max-rotation-degrees", type=float, default=10.0)
    parser.add_argument("--pdb-path", type=str, default=None)
    parser.add_argument("--pipeline-output", type=os.path.abspath, default=None)
    return parser


def main():
    parser = add_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    with job_context(args, "benchmark_kernel_bandwidth_1d") as ctx:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
            level=logging.INFO,
            handlers=[
                RobustFileHandler(os.path.join(ctx.output_dir, "run.log")),
                RobustStreamHandler(),
            ],
        )
        logger.info("Running benchmark in %s", ctx.output_dir)
        run_benchmark(args, Path(ctx.output_dir))


if __name__ == "__main__":
    main()
