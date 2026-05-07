"""Standalone benchmark for shellwise kernel-bandwidth selection on a 1D trajectory."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np

import recovar.jax_config  # noqa: F401 - ensure JAX config is initialized early
import recovar.core.fourier_transform_utils as ftu
from recovar import utils
from recovar.commands import compute_state as compute_state_cmd
from recovar.commands import pipeline
from recovar.data_io import halfsets
from recovar.heterogeneity import embedding
from recovar.heterogeneity import latent_density
from recovar.heterogeneity import kernel_bandwidth_benchmark as kb
from recovar.output import metrics, output as o
from recovar.project.job_context import job_context
from recovar.simulation import simulator, synthetic_dataset
from recovar.simulation.trajectory_generation import generate_trajectory_volumes
from recovar.utils.helpers import RobustFileHandler, RobustStreamHandler

logger = logging.getLogger(__name__)


def _load_volume_stack(prefix: str | Path, n_states: int) -> np.ndarray:
    prefix = Path(prefix)
    vols = []
    for idx in range(n_states):
        path = Path(f"{prefix}{idx:04d}.mrc")
        if not path.exists():
            raise FileNotFoundError(f"Expected trajectory volume file {path}")
        vols.append(np.asarray(utils.load_mrc(path), dtype=np.float32))
    return np.stack(vols, axis=0)


def _trajectory_volume_signature(args, voxel_size: float) -> dict:
    return {
        "trajectory_source": "pdb5nrl" if args.trajectory_source in {"recovar-default", "pdb5nrl"} else args.trajectory_source,
        "grid_size": int(args.grid_size),
        "n_states": int(args.n_states),
        "voxel_size": float(voxel_size),
        "bfactor": float(args.bfactor),
        "max_rotation_degrees": float(args.max_rotation_degrees),
        "pdb_path": None if args.pdb_path is None else str(args.pdb_path),
    }


def _load_cached_signature(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _volume_cache_is_valid(meta_path: Path, expected_signature: dict) -> bool:
    payload = _load_cached_signature(meta_path)
    return bool(payload and payload.get("signature") == expected_signature)


def _make_dataset_args(dataset_dir: Path, grid_size: int):
    return argparse.Namespace(
        particles=str(dataset_dir / "particles.star"),
        poses=str(dataset_dir / "poses.pkl"),
        ctf=str(dataset_dir / "ctf.pkl"),
        datadir=None,
        halfsets=None,
        ind=None,
        tilt_ind=None,
        ntilts=None,
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
    particles_path = output_folder / f"particles.{grid_size}.mrcs"
    sim_info_path = output_folder / "simulation_info.pkl"
    if particles_path.exists() and sim_info_path.exists():
        with sim_info_path.open("rb") as f:
            sim_info = pickle.load(f)
        image_stack = np.asarray(utils.load_mrc(particles_path), dtype=np.float32)
        return image_stack, sim_info
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
        contrast_std=0.0,
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


def _prepare_trajectory_volumes(args, benchmark_dir: Path, voxel_size: float) -> tuple[str, dict, np.ndarray, np.ndarray]:
    raw_dir = benchmark_dir / "volumes_raw"
    raw_prefix = raw_dir / "vol"
    raw_meta_path = raw_dir / "trajectory_meta.json"
    projected_prefix = None

    trajectory_source = "pdb5nrl" if args.trajectory_source in {"recovar-default", "pdb5nrl"} else args.trajectory_source
    raw_signature = _trajectory_volume_signature(args, voxel_size)

    def _cached_volume_stack(prefix: Path) -> np.ndarray | None:
        first = Path(f"{prefix}{0:04d}.mrc")
        if not first.exists():
            return None
        vols = []
        for idx in range(args.n_states):
            path = Path(f"{prefix}{idx:04d}.mrc")
            if not path.exists():
                return None
            vols.append(np.asarray(utils.load_mrc(path), dtype=np.float32))
        return np.stack(vols, axis=0)

    raw_volumes = _cached_volume_stack(raw_prefix) if _volume_cache_is_valid(raw_meta_path, raw_signature) else None
    if trajectory_source == "pdb5nrl":
        if raw_volumes is None:
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
            kb.write_json(raw_meta_path, {"signature": raw_signature})
    else:
        raise ValueError(f"Unknown trajectory source: {args.trajectory_source}")

    projected_meta = {"n_pcs": 0}
    volume_prefix = str(raw_prefix)
    used_volumes = raw_volumes
    if args.pc_project > 0:
        projected_dir = benchmark_dir / "volumes_pc_projected"
        projected_prefix = projected_dir / "vol"
        projected_meta_path = projected_dir / "trajectory_meta.json"
        projected_signature = {"raw_signature": raw_signature, "pc_project": int(args.pc_project)}
        projected_volumes = _cached_volume_stack(projected_prefix) if _volume_cache_is_valid(projected_meta_path, projected_signature) else None
        if projected_volumes is None:
            projected_volumes, projected_meta = kb.project_volume_trajectory(raw_volumes, args.pc_project)
            kb.write_volume_prefix(projected_volumes, projected_prefix, voxel_size)
            kb.write_json(projected_meta_path, {"signature": projected_signature})
        else:
            projected_meta = kb.project_volume_trajectory(raw_volumes, args.pc_project)[1]
        volume_prefix = str(projected_prefix)
        used_volumes = projected_volumes

    projection_summary = {"n_pcs": int(projected_meta.get("n_pcs", 0))}
    if "singular_values" in projected_meta:
        projection_summary["singular_values_head"] = np.asarray(projected_meta["singular_values"])[:10].tolist()
    if "explained_energy" in projected_meta:
        projection_summary["explained_energy_head"] = np.asarray(projected_meta["explained_energy"])[:10].tolist()

    return volume_prefix, {
        "trajectory_source": trajectory_source,
        "raw_prefix": str(raw_prefix),
        "projected_prefix": None if projected_prefix is None else str(projected_prefix),
        "volume_prefix": volume_prefix,
        "pc_project": int(args.pc_project),
        "projection": projection_summary,
    }, raw_volumes, used_volumes


def _run_pipeline_if_needed(args, dataset_dir: Path, sim_info: dict, voxel_size: float, *, enabled: bool) -> str | None:
    if not enabled:
        return None

    if args.pipeline_output is not None:
        return str(Path(args.pipeline_output).resolve())

    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_info)
    mask_dir = dataset_dir / "gt_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    union_mask_path = mask_dir / "gt_union_mask.mrc"
    moving_mask_path = mask_dir / "gt_moving_mask.mrc"
    soft_union_mask, binary_union_mask = metrics.make_union_gt_mask_from_hvd(gt, (args.grid_size, args.grid_size, args.grid_size))
    soft_moving_mask, binary_moving_mask = metrics.make_moving_gt_mask_from_hvd(gt, (args.grid_size, args.grid_size, args.grid_size))
    utils.write_mrc(union_mask_path, soft_union_mask.astype(np.float32), voxel_size=voxel_size)
    utils.write_mrc(moving_mask_path, soft_moving_mask.astype(np.float32), voxel_size=voxel_size)
    logger.info("GT union mask written to %s (%.1f%% voxels)", union_mask_path, 100.0 * binary_union_mask.mean())
    logger.info("GT moving mask written to %s (%.1f%% voxels)", moving_mask_path, 100.0 * binary_moving_mask.mean())

    pipeline_cmd = [
        str(dataset_dir / "particles.star"),
        "--poses",
        str(dataset_dir / "poses.pkl"),
        "--ctf",
        str(dataset_dir / "ctf.pkl"),
        "-o",
        str(dataset_dir / "pipeline_output"),
        "--mask",
        str(union_mask_path),
        "--focus-mask",
        str(moving_mask_path),
        "--no-correct-contrast",
        "--noise-model",
        "radial" if args.noise_model == "radial1" else args.noise_model,
    ]
    if args.lazy:
        pipeline_cmd.append("--lazy")
    if getattr(args, "low_memory_option", False):
        pipeline_cmd.append("--low-memory-option")
    if getattr(args, "very_low_memory_option", False):
        pipeline_cmd.append("--very-low-memory-option")
    if args.premultiplied_ctf:
        pipeline_cmd.append("--premultiplied-ctf")

    pipeline_parser = pipeline.add_args(argparse.ArgumentParser())
    pipeline_args = pipeline_parser.parse_args(pipeline_cmd)
    logger.info("Running pipeline as: recovar pipeline %s", " ".join(pipeline_cmd))
    pipeline.standard_recovar_pipeline(pipeline_args)
    return str(dataset_dir / "pipeline_output")


def _load_embedding_components(pipeline_output_dir: str, zdim: int, lazy: bool):
    po = o.PipelineOutput(pipeline_output_dir)
    dataset = po.get("lazy_dataset") if lazy else po.get("dataset")
    input_args = po.get("input_args")
    mean = np.asarray(po.get("mean"), dtype=np.complex64)
    u = np.asarray(po.get("u"), dtype=np.complex64)
    s = np.asarray(po.get("s"), dtype=np.float32)
    volume_mask = np.asarray(po.get("volume_mask"), dtype=np.float32)
    if u.ndim == 2 and hasattr(dataset, "volume_size"):
        if u.shape[0] != dataset.volume_size and u.shape[1] == dataset.volume_size:
            u = u.T
    gpu_memory = utils.get_gpu_memory_total()
    ignore_zero_frequency = bool(getattr(input_args, "ignore_zero_frequency", False))
    zs, cov_zs, contrasts, _ = embedding.get_per_image_embedding(
        mean,
        u,
        s,
        zdim,
        dataset,
        volume_mask,
        gpu_memory,
        "linear_interp",
        contrast_grid=None,
        contrast_option="none",
        ignore_zero_frequency=ignore_zero_frequency,
    )
    return dataset, np.asarray(zs, dtype=np.float32), np.asarray(cov_zs, dtype=np.float32), np.asarray(contrasts)


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


def _run_compute_state_walkthrough(
    args,
    benchmark_dir: Path,
    pipeline_output_dir: str,
    *,
    target_z: np.ndarray,
    image_stack: np.ndarray,
    particle_zs: np.ndarray,
    particle_labels: np.ndarray,
    gt_state_scores: np.ndarray,
    moving_mask: np.ndarray | None,
    diagnostic_zs: np.ndarray | None,
) -> tuple[str, dict[str, str], dict[str, object]]:
    """Run ``compute_state`` on the middle latent point and build inspection plots."""
    compute_state_root = benchmark_dir / "compute_state"
    compute_state_root.mkdir(parents=True, exist_ok=True)
    latent_points_path = benchmark_dir / "compute_state_latent_points.txt"
    np.savetxt(latent_points_path, np.asarray(target_z, dtype=np.float32).reshape(1, -1))

    compute_state_args = argparse.Namespace(
        result_dir=str(pipeline_output_dir),
        outdir=str(compute_state_root),
        project=None,
        Bfactor=float(args.bfactor),
        n_bins=int(args.n_bins),
        maskrad_fraction=float(args.compute_state_maskrad_fraction),
        n_min_particles=int(args.compute_state_n_min_particles),
        zdim1=(np.asarray(target_z).ndim == 0 or np.asarray(target_z).shape[-1] == 1),
        no_z_regularization=False,
        lazy=bool(args.lazy),
        particles=None,
        datadir=None,
        strip_prefix=None,
        apply_global_filtering=False,
        fsc_mask_radius=None,
        fsc_mask_edgewidth=None,
        latent_points=str(latent_points_path),
        save_all_estimates=bool(args.compute_state_save_all_estimates),
    )
    logger.info("Running compute_state walkthrough as a post-processing step")
    compute_state_cmd.compute_state(compute_state_args)
    saved, summary = kb.save_compute_state_walkthrough_plots(
        benchmark_dir / "presentation",
        compute_state_root,
        image_stack=image_stack,
        particle_zs=particle_zs,
        particle_labels=particle_labels,
        gt_state_scores=gt_state_scores,
        moving_mask=moving_mask,
        diagnostic_zs=diagnostic_zs,
    )
    summary["compute_state_output_dir"] = str(compute_state_root)
    summary["compute_state_latent_points"] = str(latent_points_path)
    return str(compute_state_root), saved, summary


def run_benchmark(args, benchmark_dir: Path) -> dict:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    voxel_size = 4.25 * 128 / args.grid_size
    stage_records: list[dict] = []

    def add_stage(step: int, title: str, *, summary: str = "", artifacts=None, parameters=None, metrics=None):
        record = {
            "step": int(step),
            "title": title,
            "summary": summary,
        }
        logger.info("[step %d] %s", step, title)
        if summary:
            logger.info("         %s", summary)
        if artifacts:
            record["artifacts"] = {str(k): str(v) for k, v in artifacts.items()}
        if parameters:
            record["parameters"] = dict(parameters)
        if metrics:
            record["metrics"] = dict(metrics)
        stage_records.append(record)
        return record

    state_distribution = kb.make_state_distribution(args.n_states, args.volume_distribution)
    volume_prefix, volume_meta, raw_trajectory_volumes, used_trajectory_volumes = _prepare_trajectory_volumes(
        args, benchmark_dir, voxel_size
    )
    add_stage(
        1,
        "Generate trajectory volumes",
        summary=(
            "Generate the 1D conformational trajectory volumes from the selected source. "
            "This is where the benchmark creates the volume stack that later becomes the synthetic image source."
        ),
        artifacts={
            "raw_volume_prefix": volume_meta["raw_prefix"],
            "projected_volume_prefix": volume_meta.get("projected_prefix") or "(none)",
            "active_volume_prefix": volume_meta.get("volume_prefix", volume_prefix),
        },
        parameters={
            "trajectory_source": volume_meta.get("trajectory_source", args.trajectory_source),
            "pc_project": volume_meta.get("pc_project", args.pc_project),
            "grid_size": args.grid_size,
            "n_states": args.n_states,
            "voxel_size": voxel_size,
            "bfactor": args.bfactor,
            "max_rotation_degrees": args.max_rotation_degrees,
        },
    )
    image_stack, sim_info = _write_dataset(
        volume_prefix,
        benchmark_dir,
        voxel_size=voxel_size,
        n_images=args.n_images,
        grid_size=args.grid_size,
        volume_distribution=state_distribution,
        noise_level=args.noise_level,
        contrast_std=0.0,
        noise_model=args.noise_model,
        seed=args.seed,
        premultiplied_ctf=args.premultiplied_ctf,
    )
    image_stack_shape = list(np.asarray(image_stack).shape)
    add_stage(
        2,
        "Generate the synthetic image dataset",
        summary=(
            "Sample particle images from the generated trajectory volumes using RECOVAR's simulator. "
            "This stage produces the STAR/pose/CTF metadata and the synthetic stack used for the benchmark."
        ),
        artifacts={
            "dataset_dir": str(benchmark_dir / "test_dataset"),
            "simulation_info": str(benchmark_dir / "test_dataset" / "simulation_info.pkl"),
        },
        parameters={
            "n_images": args.n_images,
            "noise_level": args.noise_level,
            "contrast_std_requested": args.contrast_std,
            "contrast_std_used": 0.0,
            "noise_model": args.noise_model,
            "volume_distribution": args.volume_distribution,
            "premultiplied_ctf": args.premultiplied_ctf,
        },
        metrics={
            "image_stack_shape": image_stack_shape,
        },
    )

    dataset_dir = benchmark_dir / "test_dataset"
    sim_info_path = dataset_dir / "simulation_info.pkl"
    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_info_path)
    gt_shape = tuple(gt.volume_shape)
    noise_variance = sim_info.get("noise_variance")
    if noise_variance is None:
        noise_variance = np.asarray(args.noise_level, dtype=np.float32)
    else:
        noise_variance = np.asarray(noise_variance, dtype=np.float32)
    gt_volumes_real = np.asarray(
        [
            np.asarray(ftu.get_idft3(gt.volumes[i].reshape(gt_shape)).real, dtype=np.float32)
            for i in range(args.n_states)
        ]
    )
    gt_scores, _gt_cov_zs, gt_meta = _fit_gt_latents(gt_volumes_real, args.zdim, args.gt_z_sigma)
    _moving_soft_mask, moving_binary_mask = metrics.make_moving_gt_mask_from_hvd(gt, gt_shape)
    moving_mask = np.asarray(moving_binary_mask, dtype=bool)
    moving_mask_fraction = float(np.mean(moving_mask))
    add_stage(
        3,
        "Load ground-truth volumes and fit latent coordinates",
        summary=(
            "Recover the ground-truth heterogeneous volume stack from the simulation metadata and fit a small PCA model. "
            "In gt-pc mode, the target latent coordinate and per-particle latents come from these PCA scores."
        ),
        artifacts={
            "simulation_info": str(sim_info_path),
        },
        parameters={
            "zdim": args.zdim,
            "gt_z_sigma": args.gt_z_sigma,
            "noise_variance": float(np.asarray(noise_variance).reshape(-1)[0]),
        },
        metrics={
            "gt_volume_shape": gt_shape,
            "gt_explained_energy_head": gt_meta["pca"]["explained_energy"][: min(5, len(gt_meta["pca"]["explained_energy"]))].tolist(),
            **({"moving_mask_fraction": moving_mask_fraction} if moving_mask_fraction is not None else {}),
        },
    )

    target_state = _make_target_state(args, args.n_states)
    target_volume = gt_volumes_real[target_state]

    dataset_args = _make_dataset_args(dataset_dir, args.grid_size)
    dataset = halfsets.load_halfset_dataset_from_args(dataset_args, lazy=args.lazy)
    if hasattr(dataset, "set_noise"):
        dataset.set_noise(noise_variance)

    need_pipeline = args.embedding_source == "recovar" or args.diagnostic_embedding_source == "recovar"
    pipeline_output_dir = _run_pipeline_if_needed(args, dataset_dir, sim_info, voxel_size, enabled=need_pipeline)

    diagnostic_zs = None
    diagnostic_cov_zs = None
    diagnostic_contrasts = None
    if args.diagnostic_embedding_source == "recovar":
        if pipeline_output_dir is None:
            raise RuntimeError("diagnostic_embedding_source=recovar requires a pipeline output directory")
        _diag_dataset, diagnostic_zs, diagnostic_cov_zs, diagnostic_contrasts = _load_embedding_components(
            pipeline_output_dir, args.diagnostic_zdim, args.lazy
        )

    if args.embedding_source == "recovar":
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
    add_stage(
        4,
        "Choose the target state and latent target",
        summary=(
            "Select a target state on the 1D trajectory, then define the latent point whose kernel regression we will benchmark. "
            "The shellwise oracle is computed against this target volume."
        ),
        artifacts={
            "target_volume": str(benchmark_dir / "bandwidth_benchmark" / "target_volume.npy"),
            "target_z": str(benchmark_dir / "bandwidth_benchmark" / "target_z.npy"),
        },
        parameters={
            "target_state": target_state,
            "embedding_source": args.embedding_source,
        },
        metrics={
            "target_z_norm": float(np.linalg.norm(np.asarray(target_z).ravel())),
        },
    )

    particle_zs_halves = dataset.split_halfset_array(np.asarray(particle_zs))
    particle_cov_zs_halves = dataset.split_halfset_array(np.asarray(particle_cov_zs))
    target_pts = np.asarray(target_z, dtype=np.float32).reshape(1, -1)
    quad0 = latent_density.compute_latent_quadratic_forms_in_batch(target_pts, particle_zs_halves[0], particle_cov_zs_halves[0])
    quad1 = latent_density.compute_latent_quadratic_forms_in_batch(target_pts, particle_zs_halves[1], particle_cov_zs_halves[1])
    distances_by_half = [quad0[:, 0], quad1[:, 0]]
    add_stage(
        5,
        "Compute target distances and choose candidate bandwidths",
        summary=(
            "Measure latent quadratic distances from the target point to particles in each half-set, then derive the bandwidth grid "
            "from that observed distance cloud."
        ),
        metrics={
            "half0_distance_min": float(np.min(distances_by_half[0])),
            "half0_distance_median": float(np.median(distances_by_half[0])),
            "half0_distance_max": float(np.max(distances_by_half[0])),
            "half1_distance_min": float(np.min(distances_by_half[1])),
            "half1_distance_median": float(np.median(distances_by_half[1])),
            "half1_distance_max": float(np.max(distances_by_half[1])),
        },
    )

    candidate_bins = kb.choose_bandwidth_bins(
        distances_by_half,
        n_bandwidths=args.n_bandwidths,
        n_min_particles=args.n_min_particles,
    )
    add_stage(
        6,
        "Compute candidate kernel estimates and the CV target",
        summary=(
            "Run RECOVAR's adaptive kernel regression for every candidate bandwidth, once per half-set, and also compute the small-bin "
            "cross-validation estimate used for the shellwise score. Full candidate volume dumps are written only when "
            "`--save-candidate-volumes` is enabled."
        ),
        artifacts={
            "candidate_bins": str(benchmark_dir / "bandwidth_benchmark" / "candidate_bins.npy"),
            "candidate_estimates_half0": str(benchmark_dir / "bandwidth_benchmark" / "candidate_estimates_half0.npy"),
            "candidate_estimates_half1": str(benchmark_dir / "bandwidth_benchmark" / "candidate_estimates_half1.npy"),
        },
        parameters={
            "n_bandwidths": args.n_bandwidths,
            "n_min_particles": args.n_min_particles,
            "batch_size": args.batch_size if args.batch_size is not None else -1,
            "heterogeneity_kernel": args.heterogeneity_kernel,
            "cv_focus_moving_mask": bool(args.cv_focus_moving_mask),
            "save_candidate_volumes": args.save_candidate_volumes,
        },
    )
    estimates_by_half, cv_by_half, lhs_by_half = kb.compute_candidate_estimates(
        dataset,
        distances_by_half,
        candidate_bins,
        noise_variance=noise_variance,
        batch_size=args.batch_size,
        heterogeneity_kernel=args.heterogeneity_kernel,
    )
    shellwise = kb.compute_shellwise_oracle_and_cv(
        estimates_by_half,
        cv_by_half,
        lhs_by_half,
        target_volume,
        relative_error=True,
        cv_focus_mask=moving_mask,
    )
    particle_labels = np.asarray(sim_info["image_assignment"], dtype=np.int32).reshape(-1)
    cheat_volumes, _cheat_meta = kb.project_volume_trajectory(raw_trajectory_volumes, 1)
    add_stage(
        7,
        "Compute shellwise oracle and CV metrics",
        summary=(
            "Compare every bandwidth candidate against the oracle target and compute the shellwise regret curve. "
            "The main metrics are the oracle choice, CV choice, regret ratio, and the agreement rate between them."
        ),
        metrics=kb.summarize_shellwise_results(shellwise),
    )

    out_dir = benchmark_dir / "bandwidth_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    presentation_saved = kb.save_presentation_plots(
        out_dir / "presentation",
        raw_volumes=raw_trajectory_volumes,
        used_volumes=used_trajectory_volumes,
        candidate_bins=candidate_bins,
        candidate_estimates_by_half=estimates_by_half,
        cheat_volumes=cheat_volumes,
        image_stack=image_stack if isinstance(image_stack, np.ndarray) else np.asarray(image_stack),
        particle_zs=particle_zs,
        particle_labels=particle_labels,
        gt_state_scores=gt_scores,
        target_state=target_state,
        target_volume=target_volume,
        moving_mask=moving_mask,
        cv_by_half=cv_by_half,
        diagnostic_zs=diagnostic_zs,
    )
    compute_state_saved = {}
    compute_state_summary = {}
    compute_state_output_dir = None
    if pipeline_output_dir is not None and not args.skip_compute_state:
        compute_state_output_dir, compute_state_saved, compute_state_summary = _run_compute_state_walkthrough(
            args,
            benchmark_dir,
            pipeline_output_dir,
            target_z=target_z,
            image_stack=image_stack,
            particle_zs=particle_zs,
            particle_labels=particle_labels,
            gt_state_scores=gt_scores,
            moving_mask=moving_mask,
            diagnostic_zs=diagnostic_zs,
        )
        presentation_saved.update(compute_state_saved)
        add_stage(
            8,
            "Run compute_state and inspect the mask-selected subset",
            summary=(
                "Run RECOVAR's production compute_state path on the middle latent point, then use the documented mask-based "
                "image-subset extraction to inspect which particles are actually contributing to the selected subvolume."
            ),
            artifacts={
                "compute_state_output_dir": compute_state_output_dir,
                "compute_state_subset_indices": str(out_dir / "presentation" / "compute_state_subset_indices.pkl"),
            },
            parameters={
                "maskrad_fraction": args.compute_state_maskrad_fraction,
                "n_min_particles": args.compute_state_n_min_particles,
            },
            metrics=compute_state_summary,
        )
    latent_bin_stats = kb.save_latent_bin_population_figure(
        out_dir / "presentation" / "latent_bin_population.pdf",
        candidate_bins,
        particle_zs,
        particle_cov_zs,
        target_z,
    )
    del image_stack
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
    if diagnostic_zs is not None:
        np.save(out_dir / "diagnostic_embedding.npy", np.asarray(diagnostic_zs, dtype=np.float32))
    if diagnostic_cov_zs is not None:
        np.save(out_dir / "diagnostic_embedding_cov.npy", np.asarray(diagnostic_cov_zs, dtype=np.float32))
    if diagnostic_contrasts is not None:
        np.save(out_dir / "diagnostic_contrasts.npy", np.asarray(diagnostic_contrasts, dtype=np.float32))
    if moving_mask is not None:
        np.save(out_dir / "moving_mask.npy", np.asarray(moving_mask, dtype=np.bool_))

    report_metadata = {
        "trajectory_source": args.trajectory_source,
        "embedding_source": args.embedding_source,
        "pc_project": int(args.pc_project),
        "target_state": int(target_state),
        "n_states": int(args.n_states),
        "n_images": int(args.n_images),
        "grid_size": int(args.grid_size),
        "noise_level": float(args.noise_level),
        "contrast_std_requested": float(args.contrast_std),
        "contrast_std": 0.0,
        "volume_prefix": volume_meta.get("volume_prefix", volume_prefix),
        "dataset_dir": str(dataset_dir),
        "pipeline_output_dir": pipeline_output_dir,
        "diagnostic_embedding_source": args.diagnostic_embedding_source,
        "diagnostic_zdim": int(args.diagnostic_zdim),
        "cv_focus_moving_mask": bool(args.cv_focus_moving_mask),
        "skip_compute_state": bool(args.skip_compute_state),
        "moving_mask_fraction": moving_mask_fraction,
        "compute_state_output_dir": compute_state_output_dir,
        "compute_state_summary": compute_state_summary,
        "summary": kb.summarize_shellwise_results(shellwise),
        "latent_bin_population": {
            "cumulative_head": latent_bin_stats["cumulative"][:10].tolist(),
            "interval_head": latent_bin_stats["interval"][:10].tolist(),
            "max_cumulative": int(np.max(latent_bin_stats["cumulative"])),
        },
    }
    report_text = kb.build_notebook_style_report(
        metadata=report_metadata,
        result=shellwise,
        candidate_bins=candidate_bins,
        stage_records=stage_records,
    )
    if presentation_saved:
        report_text += "\n\n## Presentation figures\n\n"
        for name, path in sorted(presentation_saved.items()):
            report_text += f"- `{name}`: `{path}`\n"
    kb.write_markdown_report(out_dir / "report.md", report_text)
    kb.write_json(out_dir / "trace.json", {"metadata": report_metadata, "stages": stage_records})

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
    summary["diagnostic_embedding_source"] = args.diagnostic_embedding_source
    summary["pc_project"] = int(args.pc_project)
    summary["skip_compute_state"] = bool(args.skip_compute_state)
    summary["compute_state_output_dir"] = compute_state_output_dir
    summary["compute_state_summary"] = compute_state_summary
    summary["candidate_bins"] = int(np.asarray(candidate_bins).size)
    summary["image_stack_shape"] = image_stack_shape
    summary["presentation_dir"] = str(out_dir / "presentation")
    summary["presentation_figures"] = sorted(presentation_saved.keys())
    summary["report_path"] = str(out_dir / "report.md")
    summary["trace_path"] = str(out_dir / "trace.json")
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
    parser.add_argument(
        "--contrast-std",
        type=float,
        default=0.0,
        help="Requested simulator contrast std. The benchmark currently forces this to zero.",
    )
    parser.add_argument("--noise-model", choices=["radial1", "radial"], default="radial1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory-source", choices=["recovar-default", "pdb5nrl"], default="pdb5nrl")
    parser.add_argument("--pc-project", type=int, default=0, help="0 means no projection; otherwise keep first k PCs")
    parser.add_argument("--volume-distribution", choices=["uniform", "vonmises"], default="uniform")
    parser.add_argument("--embedding-source", choices=["gt-pc", "recovar"], default="gt-pc")
    parser.add_argument(
        "--diagnostic-embedding-source",
        choices=["none", "recovar"],
        default="recovar",
        help="Compute a diagnostic embedding for plots without changing the benchmark latent coordinates.",
    )
    parser.add_argument("--diagnostic-zdim", type=int, default=3)
    parser.add_argument("--zdim", type=int, default=1)
    parser.add_argument("--gt-z-sigma", type=float, default=0.05)
    parser.add_argument("--target-state", type=int, default=None)
    parser.add_argument("--n-bandwidths", type=int, default=50)
    parser.add_argument("--n-min-particles", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--heterogeneity-kernel", choices=["parabola", "triangle", "square", "flat"], default="parabola")
    parser.add_argument(
        "--cv-focus-moving-mask",
        action="store_true",
        help="Restrict the CV score to the GT moving mask before the Fourier shell comparison.",
    )
    parser.add_argument(
        "--skip-compute-state",
        action="store_true",
        help="Skip the notebook-style compute_state walkthrough and subset plots.",
    )
    parser.add_argument(
        "--compute-state-maskrad-fraction",
        type=float,
        default=0.5,
        help="Mask radius fraction used for the compute_state walkthrough.",
    )
    parser.add_argument(
        "--compute-state-n-min-particles",
        type=int,
        default=100,
        help="Minimum particles per bin for the compute_state walkthrough.",
    )
    parser.add_argument(
        "--compute-state-save-all-estimates",
        action="store_true",
        help="Save all intermediate estimates when running the compute_state walkthrough.",
    )
    parser.add_argument("--save-candidate-volumes", action="store_true")
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--low-memory-option", action="store_true")
    parser.add_argument("--very-low-memory-option", action="store_true")
    parser.add_argument("--premultiplied-ctf", action="store_true")
    parser.add_argument("--bfactor", type=float, default=80.0)
    parser.add_argument("--max-rotation-degrees", type=float, default=5.0)
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
