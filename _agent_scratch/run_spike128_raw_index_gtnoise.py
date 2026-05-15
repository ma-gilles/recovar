from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from scipy.ndimage import center_of_mass

from recovar import utils
from recovar.commands import spike_kernel_report as skr
from recovar.commands import render_spike_morph_volumes as render_cmd
from recovar.commands import spike_walkthrough as sw
from recovar.core import mask as mask_utils
from recovar.heterogeneity import embedding, local_polynomial_regression
from recovar.output import output as output_mod
from recovar.reconstruction import regularization
from recovar.simulation import simulator


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_mrc_stack_header(path: Path, shape: tuple[int, int, int], voxel_size: float) -> None:
    """Write a valid MRC stack header without memory-mapping the stack data."""
    import mrcfile

    with mrcfile.new(path, overwrite=True) as mrc:
        header = mrc.header
        header.nx = int(shape[2])
        header.ny = int(shape[1])
        header.nz = int(shape[0])
        header.mode = 2  # float32
        header.mx = int(shape[2])
        header.my = int(shape[1])
        header.mz = int(shape[0])
        header.cella.x = float(shape[2] * voxel_size)
        header.cella.y = float(shape[1] * voxel_size)
        header.cella.z = float(shape[0] * voxel_size)
        header.dmin = 0.0
        header.dmax = -1.0
        header.dmean = -2.0
        header.rms = -1.0
        mrc.voxel_size = voxel_size


def write_array_chunk(file_obj, start_index: int, images: np.ndarray) -> None:
    image_stride = int(images.shape[-2] * images.shape[-1] * np.dtype(np.float32).itemsize)
    offset = 1024 + int(start_index) * image_stride
    file_obj.seek(offset)
    file_obj.write(np.asarray(images, dtype="<f4", order="C").tobytes(order="C"))


def copy_raw_volumes(reference_raw_dir: Path, raw_dir: Path) -> bool:
    if not reference_raw_dir.exists():
        return False
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied = False
    for src in sorted(reference_raw_dir.glob("vol[0-9][0-9][0-9][0-9].mrc")):
        dst = raw_dir / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        copied = True
    return copied or sw._count_raw_volumes(raw_dir) > 0


def ensure_source_pipeline(args: argparse.Namespace) -> Path:
    out = args.run_dir
    source_pipeline = out / "06_pipeline"
    if (source_pipeline / "model" / "params.pkl").exists():
        print(f"Reusing existing source oracle pipeline: {source_pipeline}", flush=True)
        return source_pipeline

    raw_dir = out / "01_raw_volumes"
    raw_count = sw._count_raw_volumes(raw_dir)
    if raw_count == 0 and args.reference_raw_dir is not None:
        copied = copy_raw_volumes(args.reference_raw_dir, raw_dir)
        if copied:
            raw_count = sw._count_raw_volumes(raw_dir)
            print(f"Reused {raw_count} raw volumes from {args.reference_raw_dir}", flush=True)

    if raw_count == 0:
        render_cmd.render_stack(
            pdb_dir=args.pdb_dir,
            out_dir=out,
            grid_size=args.grid_size,
            voxel_size=args.voxel_size,
            bfactor=args.render_bfactor,
            glob_pattern="morph_*.pdb",
        )
        raw_count = sw._count_raw_volumes(raw_dir)
    if raw_count < args.n_states:
        raise RuntimeError(f"Need {args.n_states} raw volumes, found {raw_count} in {raw_dir}")

    sw_args = SimpleNamespace(
        grid_size=args.grid_size,
        n_images=args.n_images,
        n_states=args.n_states,
        seed=0,
        noise_level=args.noise_level,
        noise_model=args.noise_model,
        no_ctf=args.no_ctf,
        premultiplied_ctf=False,
        pc_project=0,
        overwrite=False,
        pipeline_gpu_memory=None,
        zdim=1,
    )
    raw_volumes = sw._load_volume_stack(raw_dir / "vol", args.n_states)
    active_prefix, _active_volumes, pca_meta = sw._write_active_volumes(raw_volumes, 0, out, args.voxel_size)
    if args.stream_simulation:
        _image_stack, sim_info = simulate_dataset_streaming(sw_args, out, active_prefix, args.voxel_size)
    else:
        _image_stack, sim_info = sw._simulate_dataset(sw_args, out, active_prefix, args.voxel_size)
    if args.noise_variance_floor is not None and args.noise_variance_floor > 0:
        actual_noise_variance = np.asarray(sim_info["noise_variance"], dtype=np.float32)
        sim_info["actual_image_noise_variance"] = actual_noise_variance
        sim_info["noise_variance"] = np.maximum(actual_noise_variance, float(args.noise_variance_floor)).astype(np.float32)
        sim_info["noise_variance_floor_for_reconstruction"] = float(args.noise_variance_floor)
        utils.pickle_dump(sim_info, out / "03_dataset/simulation_info.pkl")
        print(
            "Applied positive reconstruction noise variance floor "
            f"{args.noise_variance_floor:g} to noiseless/small-noise simulation metadata",
            flush=True,
        )
    mask_paths = sw._write_gt_masks_and_volumes(
        out,
        sim_info,
        args.grid_size,
        args.voxel_size,
        mask_dilation_iters=None,
        focus_mask_percentile=95.0,
    )
    pipeline_dir = sw._run_oracle_pipeline(sw_args, out, mask_paths, sim_info, args.voxel_size)
    summary = {
        "output_dir": str(out),
        "raw_volumes": str(raw_dir),
        "active_volumes": str(out / "02_active_volumes"),
        "dataset": str(out / "03_dataset"),
        "ground_truth": mask_paths["gt_dir"],
        "masks": str(out / "05_masks"),
        "pipeline": str(pipeline_dir),
        "target_state": args.target_state,
        "grid_size": args.grid_size,
        "voxel_size": args.voxel_size,
        "box_edge_A": args.grid_size * args.voxel_size,
        "n_states": args.n_states,
        "n_images": args.n_images,
        "noise_level": args.noise_level,
        "noise_variance_floor_for_reconstruction": args.noise_variance_floor,
        "no_ctf": bool(args.no_ctf),
        "pc_project": 0,
        "true_embedding_source": "state index 0..n_states-1",
        "pca_explained_energy_head": np.asarray(pca_meta["explained_energy"])[:10].astype(float).tolist(),
        "raw_volume_prefix": str(raw_dir / "vol"),
        "active_volume_prefix": str(active_prefix),
        "render_bfactor": args.render_bfactor,
        "use_oracle_pipeline": True,
    }
    write_json(out / "config_dataset_pipeline.json", {"args": vars(args), "summary": summary})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return pipeline_dir


def simulate_dataset_streaming(args, out: Path, volume_prefix: Path, voxel_size: float) -> tuple[None, dict]:
    dataset_dir = out / "03_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    particles_path = dataset_dir / f"particles.{args.grid_size}.mrcs"
    sim_info_path = dataset_dir / "simulation_info.pkl"
    if particles_path.exists() and sim_info_path.exists() and not args.overwrite:
        return None, utils.pickle_load(sim_info_path)

    volumes = simulator.load_volumes_from_folder(
        str(volume_prefix),
        args.grid_size,
        trailing_zero_format_in_vol_name=True,
        normalize=False,
    )
    scale_vol = 1.0 / np.mean(np.linalg.norm(volumes, axis=(-1)))
    volumes *= scale_vol
    volume_distribution = sw._uniform_state_distribution(args.n_states)
    dataset_params_option = "noctf" if args.no_ctf else "uniform"
    dataset_param_generator = simulator.get_pose_ctf_generator(dataset_params_option)
    noise_variance = simulator.get_noise_model(args.noise_model, args.grid_size) / 50000 * args.noise_level

    calibration_stack, *_ = simulator.generate_simulated_dataset(
        volumes,
        voxel_size,
        volume_distribution,
        10,
        noise_variance,
        noise_scale_std=0.0,
        contrast_std=0.0,
        put_extra_particles=False,
        percent_outliers=0.0,
        dataset_param_generator=dataset_param_generator,
        volume_radius=0.7,
        disc_type="cubic",
        premultiplied_ctf=args.premultiplied_ctf,
    )
    norm_image = float(np.mean(calibration_stack**2))
    if not np.isfinite(norm_image) or norm_image <= 0:
        raise RuntimeError(f"Invalid streaming simulation normalization: {norm_image}")
    noise_variance = noise_variance / norm_image
    volumes = volumes / np.sqrt(norm_image)
    scale_vol = scale_vol / np.sqrt(norm_image)

    ctf_params, rots, trans = dataset_param_generator(args.n_images, args.grid_size)
    trans *= 0
    per_image_contrast = np.ones(args.n_images, dtype=np.float64)
    per_image_noise_scale = np.ones(args.n_images, dtype=np.float64)
    image_assignments = np.random.choice(np.arange(volumes.shape[0]), size=args.n_images, p=volume_distribution)
    ctf_evaluator = simulator.core.CTFEvaluator()
    dataset = simulator.cryoem_dataset.CryoEMDataset(
        None,
        voxel_size,
        simulator.cryoem_dataset.ImageMetadata(rots, trans, ctf_params),
        ctf_evaluator=ctf_evaluator,
        grid_size=args.grid_size,
    )
    batch_size = int(0.5 * utils.get_image_batch_size(args.grid_size, utils.get_gpu_memory_total()))
    logger_msg = (
        f"Streaming {args.n_images} images to {particles_path} as float32; "
        f"expected data bytes={args.n_images * args.grid_size * args.grid_size * 4}"
    )
    print(logger_msg, flush=True)

    write_mrc_stack_header(particles_path, (args.n_images, args.grid_size, args.grid_size), voxel_size)
    sequential_chunk_size = max(1, min(args.n_images, max(batch_size, 32768)))
    with particles_path.open("r+b", buffering=0) as stack_file:
        for chunk_id, chunk_start in enumerate(range(0, args.n_images, sequential_chunk_size)):
            chunk_end = min(args.n_images, chunk_start + sequential_chunk_size)
            chunk_slice = slice(chunk_start, chunk_end)
            chunk_dataset = simulator.cryoem_dataset.CryoEMDataset(
                None,
                voxel_size,
                simulator.cryoem_dataset.ImageMetadata(
                    rots[chunk_slice],
                    trans[chunk_slice],
                    ctf_params[chunk_slice],
                ),
                ctf_evaluator=ctf_evaluator,
                grid_size=args.grid_size,
            )
            chunk_images = simulator.simulate_data(
                chunk_dataset,
                volumes,
                noise_variance,
                batch_size,
                image_assignments[chunk_slice],
                per_image_contrast[chunk_slice],
                per_image_noise_scale[chunk_slice],
                seed=chunk_id,
                disc_type="cubic",
                mrc_file=None,
                premultiplied_ctf=args.premultiplied_ctf,
            )
            write_array_chunk(stack_file, chunk_start, chunk_images)
            if chunk_id == 0 or chunk_end == args.n_images or chunk_id % 10 == 0:
                print(f"Streamed {chunk_end}/{args.n_images} images", flush=True)

    sim_info = {
        "ctf_params": ctf_params,
        "rots": rots,
        "trans": trans,
        "per_image_contrast": per_image_contrast,
        "per_image_noise_scale": per_image_noise_scale,
        "per_image_offset": np.zeros(args.n_images, dtype=np.float32),
        "image_assignment": image_assignments,
        "noise_variance": noise_variance.astype(np.float32),
        "voxel_size": voxel_size,
        "tilt_series_assignment": None,
        "tilt_groups": None,
        "per_tilt_contrast": None,
        "noise_increase_per_tilt": None,
        "dose_indices": None,
        "dose_per_tilt": None,
        "angle_per_tilt": None,
        "n_tilts": None,
        "volumes_path_root": str(volume_prefix),
        "trailing_zero_format_in_vol_name": True,
        "scale_vol": scale_vol,
        "grid_size": args.grid_size,
        "dataset_params_option": dataset_params_option,
        "outlier_file_input": None,
        "noise_model": args.noise_model,
        "noise_level": args.noise_level,
        "no_ctf": bool(args.no_ctf),
        "streaming_simulation": True,
    }
    poses = (rots.astype(np.float32), trans.astype(np.float32))
    utils.pickle_dump(poses, dataset_dir / "poses.pkl")
    simulator.save_ctf_params(str(dataset_dir), args.grid_size, ctf_params, voxel_size)
    utils.pickle_dump(sim_info, sim_info_path)
    utils.write_starfile(
        ctf_params,
        rots.astype(np.float32),
        trans.astype(np.float32),
        voxel_size,
        args.grid_size,
        f"particles.{args.grid_size}.mrcs",
        str(dataset_dir / "particles.star"),
        halfset_indices=None,
        tilt_groups=None,
    )
    np.save(dataset_dir / "state_assignment.npy", image_assignments.astype(np.int32))
    write_json(
        dataset_dir / "README.json",
        {
            "description": "Synthetic particle stack and metadata generated from 02_active_volumes using streaming simulation.",
            "contrast_std": 0.0,
            "noise_scale_std": 0.0,
            "noise_level": float(args.noise_level),
            "no_ctf": bool(args.no_ctf),
            "particles": str(particles_path),
            "particle_dtype": "float32",
            "star": str(dataset_dir / "particles.star"),
            "poses": str(dataset_dir / "poses.pkl"),
            "ctf": str(dataset_dir / "ctf.pkl"),
            "state_assignment": str(dataset_dir / "state_assignment.npy"),
        },
    )
    return None, sim_info


def scale_state_index_embedding(args: argparse.Namespace, source_pipeline: Path, gt_pipeline: Path) -> None:
    if (gt_pipeline / "model/zdim_1/latent_coords_noreg.npy").exists():
        print(f"Reusing existing index-noisy pipeline: {gt_pipeline}", flush=True)
        return

    if gt_pipeline.exists():
        shutil.rmtree(gt_pipeline)
    shutil.copytree(source_pipeline, gt_pipeline)

    out = args.run_dir
    state_assignment = np.load(out / "03_dataset/state_assignment.npy").astype(np.int64).reshape(-1)
    source_z = np.load(source_pipeline / "model/zdim_1/latent_coords_noreg.npy").astype(np.float64).reshape(-1)
    source_precision = np.load(source_pipeline / "model/zdim_1/latent_precision_noreg.npy").astype(np.float64).reshape(-1)

    state_index = np.arange(args.n_states, dtype=np.float64)
    true_index = state_index[state_assignment]

    reference = "source oracle embedding mean/std"
    ref_mean = float(np.mean(source_z))
    ref_std = float(np.std(source_z))
    ref_meta = None
    if args.reference_gt_metadata is not None and args.reference_gt_metadata.exists():
        ref_meta = json.loads(args.reference_gt_metadata.read_text())
        ref_stats = ref_meta.get("stats", {})
        if "true_scaled_mean" in ref_stats and "true_scaled_std" in ref_stats:
            ref_mean = float(ref_stats["true_scaled_mean"])
            ref_std = float(ref_stats["true_scaled_std"])
            reference = str(args.reference_gt_metadata)
    if not np.isfinite(ref_std) or ref_std <= 0:
        raise RuntimeError(f"Invalid reference embedding std: {ref_std}")

    index_mean = float(np.mean(true_index))
    index_std = float(np.std(true_index))
    if index_std <= 0:
        raise RuntimeError("State-index embedding has zero std")
    slope = ref_std / index_std
    intercept = ref_mean - slope * index_mean
    true_scaled = slope * true_index + intercept
    true_scaled_by_state = slope * state_index + intercept

    sigma_source = "mean sqrt(1 / latent_precision_noreg) from raw-volume source oracle pipeline"
    if args.embedding_noise_std is not None:
        sigma_noise = float(args.embedding_noise_std)
        sigma_source = "explicit --embedding-noise-std"
    elif args.embedding_noise_from_reference_metadata and ref_meta is not None and "sigma_noise" in ref_meta:
        sigma_noise = float(ref_meta["sigma_noise"])
        sigma_source = f"sigma_noise from {args.reference_gt_metadata}"
    else:
        valid_precision = np.isfinite(source_precision) & (source_precision > 0)
        sigma_noise = float(np.mean(np.sqrt(1.0 / source_precision[valid_precision])))
    precision_constant = float(1.0 / sigma_noise**2)
    rng = np.random.default_rng(args.embedding_seed)
    added_noise = rng.normal(loc=0.0, scale=sigma_noise, size=true_scaled.shape)
    noisy_z = (true_scaled + added_noise).astype(np.float32).reshape(-1, 1)
    precision = np.full((noisy_z.shape[0], 1, 1), precision_constant, dtype=np.float32)

    zdir = gt_pipeline / "model/zdim_1"
    for name in ("latent_coords.npy", "latent_coords_noreg.npy"):
        np.save(zdir / name, noisy_z)
    for name in ("latent_precision.npy", "latent_precision_noreg.npy"):
        np.save(zdir / name, precision)

    target_true = float(true_scaled_by_state[args.target_state])
    target_point = out / f"target_latent_point_index_state{args.target_state:04d}.txt"
    np.savetxt(target_point, np.asarray([[target_true]], dtype=np.float32))
    np.save(out / "index_latent_true_scaled.npy", true_scaled.astype(np.float32))
    np.save(out / "index_latent_added_noise.npy", added_noise.astype(np.float32))

    metadata = {
        "embedding_source": "state index assigned to each particle",
        "state_index_min": float(np.min(state_index)),
        "state_index_max": float(np.max(state_index)),
        "scaling_reference": reference,
        "scaled_index_mean_target": ref_mean,
        "scaled_index_std_target": ref_std,
        "scaled_index_slope": float(slope),
        "scaled_index_intercept": float(intercept),
        "sigma_source": sigma_source,
        "sigma_noise": sigma_noise,
        "precision_constant": precision_constant,
        "rng_seed": args.embedding_seed,
        "target_state": int(args.target_state),
        "target_latent_point_true_state": target_true,
        "target_latent_point_noisy_centroid_state": float(np.mean(noisy_z[state_assignment == args.target_state, 0])),
        "source_pipeline": str(source_pipeline),
        "pipeline": str(gt_pipeline),
        "stats": {
            "true_scaled_min": float(np.min(true_scaled)),
            "true_scaled_max": float(np.max(true_scaled)),
            "true_scaled_mean": float(np.mean(true_scaled)),
            "true_scaled_std": float(np.std(true_scaled)),
            "noisy_z_min": float(np.min(noisy_z)),
            "noisy_z_max": float(np.max(noisy_z)),
            "noisy_z_mean": float(np.mean(noisy_z)),
            "noisy_z_std": float(np.std(noisy_z)),
        },
    }
    write_json(out / "index_noisy_embedding_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


def write_crafted_mask(args: argparse.Namespace) -> Path:
    out = args.run_dir
    mask_path = out / "05_masks/mask_crafted_level0p0126_cosine3_128.mrc"
    if mask_path.exists():
        return mask_path

    crafted = np.asarray(utils.load_mrc(args.crafted_mask), dtype=np.float32)
    binary = crafted > args.crafted_mask_level
    if crafted.shape != (args.grid_size, args.grid_size, args.grid_size):
        if all(s % args.grid_size == 0 for s in crafted.shape):
            factors = tuple(s // args.grid_size for s in crafted.shape)
            binary = binary.reshape(
                args.grid_size,
                factors[0],
                args.grid_size,
                factors[1],
                args.grid_size,
                factors[2],
            ).any(axis=(1, 3, 5))
        else:
            raise ValueError(f"Cannot downsample crafted mask shape {crafted.shape} to grid {args.grid_size}")

    binary = binary.astype(np.float32)
    soft = mask_utils.soften_volume_mask(binary, kern_rad=args.crafted_mask_cosine_width).astype(np.float32)
    binary_path = out / "05_masks/mask_crafted_level0p0126_binary_downsampled128.mrc"
    utils.write_mrc(str(binary_path), binary, voxel_size=args.voxel_size)
    utils.write_mrc(str(mask_path), soft, voxel_size=args.voxel_size)

    target = np.asarray(utils.load_mrc(out / f"04_ground_truth/gt_vol{args.target_state:04d}.mrc"), dtype=np.float32)
    if soft.sum() > 0:
        center = tuple(int(round(x)) for x in center_of_mass(soft))
    else:
        center = (args.grid_size // 2,) * 3
    center = tuple(max(0, min(args.grid_size - 1, c)) for c in center)
    preview_path = out / "05_masks/mask_crafted_level0p0126_cosine3_128_preview.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    planes = [
        ("z", target[center[0], :, :], soft[center[0], :, :], center[0]),
        ("y", target[:, center[1], :], soft[:, center[1], :], center[1]),
        ("x", target[:, :, center[2]], soft[:, :, center[2]], center[2]),
    ]
    for ax, (name, target_slice, mask_slice, idx) in zip(axes, planes):
        vmax = float(np.quantile(np.abs(target_slice), 0.995))
        vmax = vmax if vmax > 0 else float(np.max(np.abs(target)))
        ax.imshow(target_slice, cmap="gray", origin="lower", vmin=-vmax, vmax=vmax)
        ax.imshow(np.ma.masked_where(mask_slice <= 0, mask_slice), cmap="autumn", origin="lower", alpha=0.42, vmin=0, vmax=1)
        if np.any(mask_slice > 0.5):
            ax.contour(mask_slice, levels=[0.5], colors=["cyan"], linewidths=0.8, origin="lower")
        ax.set_title(f"{name}-slice index {idx}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("mask_crafted > 0.0126, downsampled to 128, cosine edge 3 voxels; over raw-volume GT state 50")
    fig.savefig(preview_path, dpi=180)
    plt.close(fig)

    stats = {
        "source_mask": str(args.crafted_mask),
        "threshold": args.crafted_mask_level,
        "binary_voxels": int(binary.sum()),
        "soft_nonzero_voxels": int((soft > 0).sum()),
        "soft_gt_0p5_voxels": int((soft > 0.5).sum()),
        "soft_sum": float(soft.sum()),
        "soft_edge_voxels": args.crafted_mask_cosine_width,
        "soft_edge_A": args.crafted_mask_cosine_width * args.voxel_size,
        "binary_path": str(binary_path),
        "soft_path": str(mask_path),
        "preview_png": str(preview_path),
        "slice_center_zyx": list(center),
    }
    write_json(out / "05_masks/mask_crafted_level0p0126_cosine3_128_stats.json", stats)
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    return mask_path


def run_command(cmd: list[str]) -> None:
    print("COMMAND " + " ".join(cmd), flush=True)
    start = time.time()
    subprocess.run(cmd, check=True)
    print(f"COMMAND_DONE elapsed_s={time.time() - start:.1f}", flush=True)


def _state_dir(root: Path) -> Path:
    return root / "diagnostics" / "state000"


def _candidate_paths(root: Path) -> list[Path]:
    paths = sorted((_state_dir(root) / "estimates_filt").glob("*.mrc"))
    if not paths:
        raise RuntimeError(f"No filtered candidate volumes in {_state_dir(root) / 'estimates_filt'}")
    return paths


def _candidate_complete(root: Path) -> bool:
    try:
        return (_state_dir(root) / "params.pkl").exists() and len(_candidate_paths(root)) > 0
    except RuntimeError:
        return False


def _candidate_grid(root: Path) -> np.ndarray:
    params = utils.pickle_load(_state_dir(root) / "params.pkl")
    if "lambda_grid" in params:
        return np.asarray(params["lambda_grid"], dtype=np.float64)
    local_poly = params.get("local_poly", {})
    if isinstance(local_poly, dict) and "h_grid" in local_poly:
        return np.asarray(local_poly["h_grid"], dtype=np.float64)
    return np.asarray(params["heterogeneity_bins"], dtype=np.float64)


def _load_candidate_metrics(root: Path, target: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, n_shells = skr._shell_labels(target.shape)
    target_ft = skr._numpy_dft3(target * mask)
    target_power = np.bincount(labels.ravel(), weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)
    fscs = []
    errors = []
    for path in _candidate_paths(root):
        volume = np.asarray(utils.load_mrc(path), dtype=np.float32)
        fsc, error = skr._masked_metrics_for_volume(volume, target, mask, labels, n_shells, target_ft, target_power)
        fscs.append(fsc)
        errors.append(error)
    voxel_size = float(utils.pickle_load(_state_dir(root) / "params.pkl")["voxel_size"])
    freq = np.arange(n_shells, dtype=np.float64) / (target.shape[0] * voxel_size)
    return freq, np.asarray(fscs), np.asarray(errors)


def write_gt_shell_voxel_polynomial_diagnostic(
    args: argparse.Namespace,
    mask_path: Path,
    report_out: Path,
    h_values: list[float],
) -> Path | None:
    gt_dir = args.run_dir / "04_ground_truth"
    gt_paths = [gt_dir / f"gt_vol{idx:04d}.mrc" for idx in range(args.n_states)]
    if not all(path.exists() for path in gt_paths):
        return None

    metadata_path = args.run_dir / "index_noisy_embedding_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        slope = float(metadata["scaled_index_slope"])
        intercept = float(metadata["scaled_index_intercept"])
    else:
        slope, intercept = 1.0, 0.0
    states = np.arange(args.n_states, dtype=np.float64)
    z_scaled = slope * states + intercept
    target_z = float(slope * args.target_state + intercept)

    mask = np.asarray(utils.load_mrc(mask_path), dtype=np.float32)
    labels, _ = skr._shell_labels(mask.shape)
    shell = 32
    shell_indices = np.flatnonzero(labels.ravel() == shell)
    if shell_indices.size == 0:
        return None

    coeffs = []
    for path in gt_paths:
        vol = np.asarray(utils.load_mrc(path), dtype=np.float32)
        coeffs.append(skr._numpy_dft3(vol * mask).ravel()[shell_indices])
    coeffs = np.asarray(coeffs)
    voxel_local = int(np.nanargmax(np.std(np.abs(coeffs), axis=0)))
    voxel_flat = int(shell_indices[voxel_local])
    voxel_zyx = tuple(int(x) for x in np.unravel_index(voxel_flat, mask.shape))
    y = coeffs[:, voxel_local]

    dense_states = np.linspace(0, args.n_states - 1, 600)
    dense_z = slope * dense_states + intercept

    def _poly_fit(values: np.ndarray, h: float) -> np.ndarray:
        t = (z_scaled - target_z) / h
        x_dense = (dense_z - target_z) / h
        design = np.stack([t**r / math.factorial(r) for r in range(args.local_poly_degree + 1)], axis=1)
        design_dense = np.stack(
            [x_dense**r / math.factorial(r) for r in range(args.local_poly_degree + 1)],
            axis=1,
        )
        weights = np.exp(-0.5 * ((z_scaled - target_z) / h) ** 2)
        lhs = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * values)
        theta = np.linalg.solve(lhs + 1e-8 * np.eye(lhs.shape[0]), rhs)
        return design_dense @ theta

    plot_path = report_out / "plots" / "gt_shell32_voxel_polynomial_fit.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True, constrained_layout=True)
    components = [("real", y.real), ("imag", y.imag)]
    colors = ["tab:orange", "tab:red", "tab:purple"]
    for ax, (name, values) in zip(axes, components):
        ax.plot(states, values, "o", markersize=3.5, color="black", alpha=0.72, label="GT states")
        for color, h in zip(colors, h_values):
            fit = _poly_fit(values, float(h))
            ax.plot(dense_states, fit, linewidth=2.0, color=color, label=f"degree {args.local_poly_degree} fit, h={h:g}")
        ax.axvline(args.target_state, color="0.35", linestyle="--", linewidth=1.0, label="target state" if name == "real" else None)
        ax.set_ylabel(f"{name} coefficient")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("GT state index")
    fig.suptitle(
        f"GT Fourier voxel on shell {shell}, zyx={voxel_zyx}; polynomial fits use scaled index embedding",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    write_json(
        report_out / "gt_shell32_voxel_polynomial_fit.json",
        {
            "plot": str(plot_path),
            "shell": shell,
            "voxel_zyx": voxel_zyx,
            "target_state": int(args.target_state),
            "target_scaled_z": target_z,
            "h_values": [float(x) for x in h_values],
            "description": "GT masked Fourier voxel values over all state volumes with local degree-3 polynomial fits.",
        },
    )
    return plot_path


def _local_polynomial_state_coefficients(
    z_scaled: np.ndarray,
    target_z: float,
    h_scaled: float,
    degree: int,
    *,
    include_state: np.ndarray,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, float, float]:
    """Return coefficients mapping GT state volumes to theta_0."""
    z_fit = z_scaled[include_state]
    t = (z_fit - target_z) / float(h_scaled)
    weights = np.exp(-0.5 * t**2)
    design = np.stack([t**r / math.factorial(r) for r in range(degree + 1)], axis=1)
    lhs = design.T @ (weights[:, None] * design)
    scale = max(float(np.trace(lhs)) / max(lhs.shape[0], 1), 1.0)
    rhs_matrix = design.T * weights[None, :]
    smoother = np.linalg.solve(lhs + ridge * scale * np.eye(lhs.shape[0]), rhs_matrix)[0]
    coeff = np.zeros_like(z_scaled, dtype=np.float64)
    coeff[include_state] = smoother
    effective_n = float(weights.sum() ** 2 / max(float(np.sum(weights**2)), 1e-30))
    coeff_l1 = float(np.sum(np.abs(coeff)))
    return coeff, effective_n, coeff_l1


def _masked_metrics_from_ft(
    pred_ft_flat: np.ndarray,
    target_ft_flat: np.ndarray,
    target_power: np.ndarray,
    labels_flat: np.ndarray,
    n_shells: int,
) -> tuple[np.ndarray, np.ndarray]:
    top = np.bincount(
        labels_flat,
        weights=np.real(np.conj(pred_ft_flat) * target_ft_flat),
        minlength=n_shells,
    )
    bot1 = np.bincount(labels_flat, weights=np.abs(pred_ft_flat) ** 2, minlength=n_shells)
    bot2 = target_power
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = top / np.sqrt(bot1 * bot2)
    fsc[~np.isfinite(fsc)] = 0.0
    if fsc.size > 1:
        fsc[0] = fsc[1]
    diff_ft = pred_ft_flat - target_ft_flat
    err = np.bincount(labels_flat, weights=np.abs(diff_ft) ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = err / np.maximum(target_power, 1e-30)
    rel_err[~np.isfinite(rel_err)] = np.inf
    return fsc, rel_err


def write_gt_polynomial_sweep(
    args: argparse.Namespace,
    mask_path: Path,
    report_out: Path,
) -> dict | None:
    """Fit GT state volumes locally over state index and score target recovery."""
    gt_dir = args.run_dir / "04_ground_truth"
    stack_path = gt_dir / "gt_volumes_used_by_simulator.npy"
    if not stack_path.exists():
        return None

    metadata_path = args.run_dir / "index_noisy_embedding_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        slope = float(metadata["scaled_index_slope"])
        intercept = float(metadata["scaled_index_intercept"])
    else:
        slope, intercept = 1.0, 0.0
    states = np.arange(args.n_states, dtype=np.float64)
    z_scaled = slope * states + intercept
    target_state = int(args.target_state)
    target_z = float(z_scaled[target_state])

    mask = np.asarray(utils.load_mrc(mask_path), dtype=np.float32)
    labels, n_shells = skr._shell_labels(mask.shape)
    labels_flat = labels.ravel()
    freq = np.arange(n_shells, dtype=np.float64) / (mask.shape[0] * args.voxel_size)
    score_mask = (freq > 0.0) & (freq <= args.report_score_frequency_max)

    stack = np.load(stack_path, mmap_mode="r")
    if stack.shape[0] < args.n_states or tuple(stack.shape[1:]) != tuple(mask.shape):
        raise ValueError(f"Unexpected GT stack shape {stack.shape}; expected ({args.n_states}, {mask.shape})")

    print("GT_POLY_SWEEP precomputing masked Fourier GT stack", flush=True)
    ft_stack = np.empty((args.n_states, mask.size), dtype=np.complex64)
    for idx in range(args.n_states):
        ft_stack[idx] = skr._numpy_dft3(np.asarray(stack[idx], dtype=np.float32) * mask).reshape(-1).astype(np.complex64)
    target_ft = ft_stack[target_state]
    target_power = np.bincount(labels_flat, weights=np.abs(target_ft) ** 2, minlength=n_shells)

    base_h_state = np.asarray(
        [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 32.0, 40.0, 50.0, 64.0, 80.0],
        dtype=np.float64,
    )
    local_poly_params = utils.pickle_load(_state_dir(local_poly_output_dir(args)) / "params.pkl")
    local_h_scaled = np.asarray(local_poly_params["local_poly"]["h_grid"], dtype=np.float64)
    h_state_grid = np.unique(np.round(np.concatenate([base_h_state, local_h_scaled / slope]), 8))
    degree_grid = np.asarray([0, 1, 2, 3, 4, 5, 6, 8], dtype=np.int32)

    rows = []
    include_all = np.ones(args.n_states, dtype=bool)
    include_leave_target_out = include_all.copy()
    include_leave_target_out[target_state] = False
    for degree in degree_grid:
        for h_state in h_state_grid:
            h_scaled = float(h_state * slope)
            for fit_mode, include_state in (
                ("in_sample", include_all),
                ("leave_target_out", include_leave_target_out),
            ):
                coeff, effective_n, coeff_l1 = _local_polynomial_state_coefficients(
                    z_scaled,
                    target_z,
                    h_scaled,
                    int(degree),
                    include_state=include_state,
                )
                pred_ft = coeff.astype(np.complex64) @ ft_stack
                fsc, rel_err = _masked_metrics_from_ft(pred_ft, target_ft, target_power, labels_flat, n_shells)
                rows.append(
                    {
                        "fit_mode": fit_mode,
                        "degree": int(degree),
                        "h_state": float(h_state),
                        "h_scaled": float(h_scaled),
                        "effective_n": effective_n,
                        "coeff_l1": coeff_l1,
                        "mean_fsc_0_to_score_max": float(np.nanmean(fsc[score_mask])),
                        "median_fsc_0_to_score_max": float(np.nanmedian(fsc[score_mask])),
                        "median_relative_error_0_to_score_max": float(np.nanmedian(rel_err[score_mask])),
                        "fsc_shell32": float(fsc[32]) if fsc.size > 32 else np.nan,
                        "relative_error_shell32": float(rel_err[32]) if rel_err.size > 32 else np.nan,
                    }
                )

    import csv

    csv_path = report_out / "gt_polynomial_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def _best_row(mode: str, selector: str = "mean_fsc_0_to_score_max") -> dict:
        candidates = [r for r in rows if r["fit_mode"] == mode]
        if selector == "median_relative_error_0_to_score_max":
            return min(candidates, key=lambda r: r[selector])
        return max(candidates, key=lambda r: r[selector])

    best_leave = _best_row("leave_target_out")
    best_leave_err = _best_row("leave_target_out", "median_relative_error_0_to_score_max")
    best_in = _best_row("in_sample")
    best_degree3 = max(
        [r for r in rows if r["fit_mode"] == "leave_target_out" and r["degree"] == args.local_poly_degree],
        key=lambda r: r["mean_fsc_0_to_score_max"],
    )
    current_local_h_state = float(local_h_scaled[0] / slope)
    current_degree3 = min(
        [r for r in rows if r["fit_mode"] == "leave_target_out" and r["degree"] == args.local_poly_degree],
        key=lambda r: abs(r["h_state"] - current_local_h_state),
    )

    plots_dir = report_out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _plot_metric(metric: str, ylabel: str, name: str, fit_mode: str = "leave_target_out") -> Path:
        fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
        for degree in degree_grid:
            degree_rows = sorted(
                [r for r in rows if r["fit_mode"] == fit_mode and r["degree"] == int(degree)],
                key=lambda r: r["h_state"],
            )
            ax.plot(
                [r["h_state"] for r in degree_rows],
                [r[metric] for r in degree_rows],
                marker="o",
                markersize=3.5,
                linewidth=1.5,
                label=f"degree {degree}",
            )
        ax.axvline(current_local_h_state, color="0.25", linestyle="--", linewidth=1.0, label="current local_poly h_min")
        ax.set_xscale("log")
        ax.set_xlabel("bandwidth h in GT state-index units")
        ax.set_ylabel(ylabel)
        ax.set_title(f"GT polynomial sweep ({fit_mode.replace('_', ' ')})")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        out_path = plots_dir / name
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return out_path

    mean_fsc_plot = _plot_metric(
        "mean_fsc_0_to_score_max",
        f"mean masked FSC, 0-{args.report_score_frequency_max:g} 1/A",
        "gt_polynomial_sweep_leave_target_out_mean_fsc.png",
    )
    shell32_plot = _plot_metric(
        "fsc_shell32",
        "masked FSC on shell 32",
        "gt_polynomial_sweep_leave_target_out_shell32_fsc.png",
    )
    err_plot = _plot_metric(
        "median_relative_error_0_to_score_max",
        f"median relative error, 0-{args.report_score_frequency_max:g} 1/A",
        "gt_polynomial_sweep_leave_target_out_relative_error.png",
    )

    summary = {
        "csv": str(csv_path),
        "plots": {
            "leave_target_out_mean_fsc": str(mean_fsc_plot),
            "leave_target_out_shell32_fsc": str(shell32_plot),
            "leave_target_out_relative_error": str(err_plot),
        },
        "degree_grid": [int(x) for x in degree_grid],
        "h_state_grid": [float(x) for x in h_state_grid],
        "slope_scaled_z_per_state": float(slope),
        "current_local_poly_h_state": current_local_h_state,
        "best_leave_target_out_by_mean_fsc": best_leave,
        "best_leave_target_out_by_median_relative_error": best_leave_err,
        "best_in_sample_by_mean_fsc": best_in,
        "best_leave_target_out_degree3_by_mean_fsc": best_degree3,
        "current_degree3_nearest_local_poly_h": current_degree3,
    }
    write_json(report_out / "gt_polynomial_sweep_summary.json", summary)
    print("GT_POLY_SWEEP_DONE " + json.dumps(summary["best_leave_target_out_by_mean_fsc"], sort_keys=True), flush=True)
    return summary


def _parse_float_list(value: str | None, default: np.ndarray) -> np.ndarray:
    if value is None or str(value).strip() == "":
        return np.asarray(default, dtype=np.float32)
    return np.asarray([float(x) for x in str(value).split(",") if x.strip()], dtype=np.float32)


def _tag_token(tag: str | None) -> str:
    if tag is None or str(tag).strip() == "":
        return ""
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "p" for ch in str(tag).strip())
    return f"_{safe}"


def _local_poly_config_token(args: argparse.Namespace) -> str:
    tokens = []
    basis = str(getattr(args, "local_poly_basis", "monomial"))
    reg_type = str(getattr(args, "local_poly_pol_reg_type", "none"))
    eta = float(getattr(args, "local_poly_pol_reg_eta", 0.0))
    power = float(getattr(args, "local_poly_pol_reg_power", 2.0))
    if basis != "monomial":
        tokens.append(f"basis{basis}")
    if reg_type != "none" or eta != 0.0:
        tokens.append(f"reg{reg_type}_eta{eta:g}_pow{power:g}".replace(".", "p"))
    return "" if not tokens else "_" + "_".join(tokens)


def local_poly_output_dir(args: argparse.Namespace) -> Path:
    suffix = "_lazy" if args.compute_state_lazy else ""
    local_poly_maskrad_token = f"_maskrad{args.local_poly_maskrad_fraction:g}".replace(".", "p")
    return args.run_dir / (
        f"07_compute_state_local_poly_index_gtnoise_degree{args.local_poly_degree}"
        f"{local_poly_maskrad_token}{_local_poly_config_token(args)}{_tag_token(args.local_poly_output_tag)}{suffix}"
    )


def local_poly_em_output_dir(args: argparse.Namespace) -> Path:
    suffix = "_lazy" if args.compute_state_lazy else ""
    local_poly_maskrad_token = f"_maskrad{args.local_poly_maskrad_fraction:g}".replace(".", "p")
    return args.run_dir / (
        f"07_compute_state_local_poly_em_index_gtnoise_degree{args.local_poly_degree}"
        f"_iter{args.local_poly_em_iterations}_quad{args.local_poly_em_quadrature}"
        f"{local_poly_maskrad_token}{_local_poly_config_token(args)}{_tag_token(args.local_poly_output_tag)}{suffix}"
    )


def three_mode_report_dir(args: argparse.Namespace) -> Path:
    suffix = "_lazy" if args.compute_state_lazy else ""
    return args.run_dir / f"08_kernel_report_mask_crafted_level0p0126_cosine3_local_poly{_tag_token(args.local_poly_output_tag)}{suffix}"


def compute_local_poly_candidates_direct(
    args: argparse.Namespace,
    gt_pipeline: Path,
    local_poly_out: Path,
    target_point: Path,
) -> None:
    """Write local_poly candidate volumes without RECOVAR's final recombination pass.

    The full compute_state path currently finishes the candidate local polynomial
    reconstructions but can OOM when it reruns all candidates at upsampling=2 for
    local-resolution recombination. For this controlled raw-index experiment we
    only need per-candidate FSC curves, so save the upsampling=1 halfmaps and
    their average in the same diagnostics layout used by the report code.
    """
    if _candidate_complete(local_poly_out):
        print(f"Reusing local_poly direct candidates at {local_poly_out}", flush=True)
        return
    if local_poly_out.exists():
        print(f"Removing incomplete local_poly output at {local_poly_out}", flush=True)
        shutil.rmtree(local_poly_out)

    state_dir = _state_dir(local_poly_out)
    state_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(state_dir / "latent_coords.txt", np.loadtxt(target_point, ndmin=2))

    po = output_mod.PipelineOutput(str(gt_pipeline))
    dataset = po.get("lazy_dataset") if args.compute_state_lazy else po.get("dataset")
    zs = np.asarray(po.get_embedding_component("latent_coords_noreg", 1), dtype=np.float32)
    if zs.ndim == 1:
        zs = zs[:, None]
    latent_precision = local_polynomial_regression.coerce_1d_latent_precision(
        np.asarray(po.get_embedding_component("latent_precision_noreg", 1), dtype=np.float32)
    )
    contrasts = np.asarray(po.get_embedding_component("contrasts_noreg", 1), dtype=np.float32)
    embedding.set_contrasts_in_cryos(dataset, contrasts)

    target = np.loadtxt(target_point, ndmin=2).astype(np.float32).reshape(1, 1)
    latent_diff = np.asarray(zs[:, 0] - target[0, 0], dtype=np.float32)
    multipliers = _parse_float_list(
        args.local_poly_bandwidth_multipliers,
        local_polynomial_regression.DEFAULT_LOCAL_POLY_BANDWIDTH_MULTIPLIERS,
    )
    _, default_h_grid, sigma_ref, h_min, r_min = local_polynomial_regression.local_poly_bandwidth_grid_info_1d(
        latent_diff,
        latent_precision,
        100,
        multipliers=multipliers,
    )
    if args.local_poly_h_scaled_grid:
        h_grid = _parse_float_list(args.local_poly_h_scaled_grid, default_h_grid)
    elif args.local_poly_h_state_grid:
        metadata_path = args.run_dir / "index_noisy_embedding_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        slope = float(metadata["scaled_index_slope"])
        h_grid = _parse_float_list(args.local_poly_h_state_grid, default_h_grid / slope) * slope
    else:
        h_grid = default_h_grid
    h_grid = np.asarray(h_grid, dtype=np.float32).reshape(-1)
    if h_grid.size == 0 or not np.all(np.isfinite(h_grid)) or np.any(h_grid <= 0):
        raise ValueError(f"Invalid local_poly h_grid: {h_grid}")

    n_images_per_h = []
    for h in h_grid:
        moments, _ = local_polynomial_regression.gaussian_window_polynomial_moments_1d(
            latent_diff,
            latent_precision,
            float(h),
            0,
        )
        n_images_per_h.append(int(np.count_nonzero(moments[:, 0] > 0)))

    split_diff = dataset.split_halfset_array(latent_diff, per_particle=dataset.tilt_series_flag)
    split_precision = dataset.split_halfset_array(latent_precision, per_particle=dataset.tilt_series_flag)
    half_estimates = []
    for half_idx in (0, 1):
        print(f"LOCAL_POLY_DIRECT half={half_idx + 1} h_grid={h_grid.tolist()}", flush=True)
        estimates = local_polynomial_regression.estimate_local_polynomial_volumes(
            dataset.get_halfset(half_idx),
            split_diff[half_idx],
            split_precision[half_idx],
            h_grid,
            degree=args.local_poly_degree,
            tau=None,
            grid_correct=True,
            use_spherical_mask=True,
            upsampling_factor=1,
            return_real_space=True,
            use_fast_rfft=True,
            basis=args.local_poly_basis,
            pol_reg_type=args.local_poly_pol_reg_type,
            pol_reg_eta=args.local_poly_pol_reg_eta,
            pol_reg_power=args.local_poly_pol_reg_power,
        )
        half_estimates.append(estimates.reshape(-1, *dataset.volume_shape).astype(np.float32))

    avg_estimates = ((half_estimates[0] + half_estimates[1]) * 0.5).astype(np.float32)
    for subdir, volumes in (
        ("estimates_half1_unfil", half_estimates[0]),
        ("estimates_half2_unfil", half_estimates[1]),
        ("estimates_filt", avg_estimates),
    ):
        out_dir = state_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        output_mod.save_volumes(
            volumes,
            os.path.join(out_dir, ""),
            dataset.volume_shape,
            from_ft=False,
            voxel_size=dataset.voxel_size,
        )

    params = {
        "kernel_regression_mode": "local_poly",
        "heterogeneity_bins": h_grid,
        "n_images_per_bin": n_images_per_h,
        "voxel_size": float(dataset.voxel_size),
        "volume_shape": tuple(dataset.volume_shape),
        "candidate_note": (
            "Direct local_poly upsampling=1 candidate averages. Stored in estimates_filt "
            "for report compatibility; this intentionally skips compute_state final recombination."
        ),
        "local_poly": {
            "degree": int(args.local_poly_degree),
            "basis": str(args.local_poly_basis),
            "pol_reg_type": str(args.local_poly_pol_reg_type),
            "pol_reg_eta": float(args.local_poly_pol_reg_eta),
            "pol_reg_power": float(args.local_poly_pol_reg_power),
            "h_grid": h_grid,
            "h_state_grid": (
                (h_grid / float(json.loads((args.run_dir / "index_noisy_embedding_metadata.json").read_text())["scaled_index_slope"]))
                if (args.run_dir / "index_noisy_embedding_metadata.json").exists()
                else None
            ),
            "requested_h_state_grid": args.local_poly_h_state_grid,
            "requested_h_scaled_grid": args.local_poly_h_scaled_grid,
            "sigma_ref": float(sigma_ref),
            "h_min": float(h_min),
            "r_min": float(r_min),
        },
    }
    utils.pickle_dump(params, state_dir / "params.pkl")
    write_json(
        local_poly_out / "job.json",
        {
            "mode": "local_poly_direct_candidates",
            "pipeline": str(gt_pipeline),
            "target_point": str(target_point),
            "candidate_count": int(h_grid.size),
            "params": str(state_dir / "params.pkl"),
        },
    )
    print(f"LOCAL_POLY_DIRECT_DONE out={local_poly_out}", flush=True)


def make_local_poly_em_tau(
    args: argparse.Namespace,
    po: output_mod.PipelineOutput,
    dataset,
) -> tuple[np.ndarray | None, dict]:
    """Construct an optional radial Fourier prior for EM local-poly solves."""
    mode = str(args.local_poly_em_tau_mode)
    metadata = {
        "mode": mode,
        "scale": float(args.local_poly_em_tau_scale),
        "floor_relative": float(args.local_poly_em_tau_floor_relative),
    }
    if mode == "none":
        return None, metadata

    if mode == "mean_power":
        source = np.asarray(po.get("mean")).reshape(dataset.volume_shape)
        source_note = "pipeline mean Fourier power"
    elif mode == "gt_target_power":
        target_path = args.run_dir / f"04_ground_truth/gt_vol{args.target_state:04d}.mrc"
        source = skr._numpy_dft3(np.asarray(utils.load_mrc(target_path), dtype=np.float32))
        source_note = f"GT target Fourier power ({target_path})"
    else:
        raise ValueError(f"Unknown --local-poly-em-tau-mode {mode!r}")

    power = np.asarray(np.abs(source) ** 2, dtype=np.float64).reshape(dataset.volume_shape)
    shell_power = np.array(
        regularization.average_over_shells(power, dataset.volume_shape),
        dtype=np.float64,
        copy=True,
    )
    if shell_power.size > 1:
        shell_power[0] = shell_power[1]
    positive = shell_power[np.isfinite(shell_power) & (shell_power > 0)]
    if positive.size == 0:
        raise ValueError(f"Cannot build local_poly EM tau from {source_note}: power spectrum is nonpositive")
    floor = float(np.max(positive) * args.local_poly_em_tau_floor_relative)
    shell_power = np.maximum(np.nan_to_num(shell_power, nan=floor, posinf=floor, neginf=floor), floor)
    shell_power *= float(args.local_poly_em_tau_scale)
    tau = np.asarray(utils.make_radial_image(shell_power, dataset.volume_shape), dtype=np.float32).reshape(-1)
    metadata.update(
        {
            "source": source_note,
            "tau_min": float(np.min(tau)),
            "tau_max": float(np.max(tau)),
            "tau_median": float(np.median(tau)),
        }
    )
    print(f"LOCAL_POLY_EM_TAU {json.dumps(metadata, sort_keys=True)}", flush=True)
    return tau, metadata


def compute_local_poly_em_candidates_direct(
    args: argparse.Namespace,
    gt_pipeline: Path,
    local_poly_em_out: Path,
    target_point: Path,
) -> None:
    """Write EM local_poly candidate volumes for this controlled experiment."""
    if _candidate_complete(local_poly_em_out):
        print(f"Reusing local_poly EM direct candidates at {local_poly_em_out}", flush=True)
        return
    if local_poly_em_out.exists():
        print(f"Removing incomplete local_poly EM output at {local_poly_em_out}", flush=True)
        shutil.rmtree(local_poly_em_out)

    state_dir = _state_dir(local_poly_em_out)
    state_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(state_dir / "latent_coords.txt", np.loadtxt(target_point, ndmin=2))

    po = output_mod.PipelineOutput(str(gt_pipeline))
    dataset = po.get("lazy_dataset") if args.compute_state_lazy else po.get("dataset")
    zs = np.asarray(po.get_embedding_component("latent_coords_noreg", 1), dtype=np.float32)
    if zs.ndim == 1:
        zs = zs[:, None]
    latent_precision = local_polynomial_regression.coerce_1d_latent_precision(
        np.asarray(po.get_embedding_component("latent_precision_noreg", 1), dtype=np.float32)
    )
    contrasts = np.asarray(po.get_embedding_component("contrasts_noreg", 1), dtype=np.float32)
    embedding.set_contrasts_in_cryos(dataset, contrasts)
    tau, tau_metadata = make_local_poly_em_tau(args, po, dataset)

    target = np.loadtxt(target_point, ndmin=2).astype(np.float32).reshape(1, 1)
    latent_diff = np.asarray(zs[:, 0] - target[0, 0], dtype=np.float32)
    multipliers = _parse_float_list(
        args.local_poly_bandwidth_multipliers,
        local_polynomial_regression.DEFAULT_LOCAL_POLY_BANDWIDTH_MULTIPLIERS,
    )
    _, default_h_grid, sigma_ref, h_min, r_min = local_polynomial_regression.local_poly_bandwidth_grid_info_1d(
        latent_diff,
        latent_precision,
        100,
        multipliers=multipliers,
    )
    if args.local_poly_h_scaled_grid:
        h_grid = _parse_float_list(args.local_poly_h_scaled_grid, default_h_grid)
    elif args.local_poly_h_state_grid:
        metadata_path = args.run_dir / "index_noisy_embedding_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        slope = float(metadata["scaled_index_slope"])
        h_grid = _parse_float_list(args.local_poly_h_state_grid, default_h_grid / slope) * slope
    else:
        h_grid = default_h_grid
    h_grid = np.asarray(h_grid, dtype=np.float32).reshape(-1)
    if h_grid.size == 0 or not np.all(np.isfinite(h_grid)) or np.any(h_grid <= 0):
        raise ValueError(f"Invalid local_poly EM h_grid: {h_grid}")

    n_images_per_h = []
    for h in h_grid:
        moments, _ = local_polynomial_regression.gaussian_window_polynomial_moments_1d(
            latent_diff,
            latent_precision,
            float(h),
            0,
        )
        n_images_per_h.append(int(np.count_nonzero(moments[:, 0] > 0)))

    split_diff = dataset.split_halfset_array(latent_diff, per_particle=dataset.tilt_series_flag)
    split_precision = dataset.split_halfset_array(latent_precision, per_particle=dataset.tilt_series_flag)
    half_estimates = []
    half_diagnostics = []
    for half_idx in (0, 1):
        print(
            "LOCAL_POLY_EM_DIRECT "
            f"half={half_idx + 1} degree={args.local_poly_degree} "
            f"iters={args.local_poly_em_iterations} quad={args.local_poly_em_quadrature} "
            f"temperature={args.local_poly_em_temperature:g} "
            f"prior_mix={args.local_poly_em_prior_mix:g} "
            f"update_damping={args.local_poly_em_update_damping:g} "
            f"basis={args.local_poly_basis} "
            f"pol_reg={args.local_poly_pol_reg_type}:{args.local_poly_pol_reg_eta:g} "
            f"tau_mode={args.local_poly_em_tau_mode} "
            f"h_grid={h_grid.tolist()}",
            flush=True,
        )
        estimates, diagnostics = local_polynomial_regression.estimate_local_polynomial_volumes_em(
            dataset.get_halfset(half_idx),
            split_diff[half_idx],
            split_precision[half_idx],
            h_grid,
            degree=args.local_poly_degree,
            n_iterations=args.local_poly_em_iterations,
            n_quadrature=args.local_poly_em_quadrature,
            tau=tau,
            grid_correct=True,
            use_spherical_mask=True,
            upsampling_factor=1,
            return_real_space=True,
            use_fast_rfft=True,
            return_diagnostics=True,
            em_temperature=args.local_poly_em_temperature,
            em_prior_mix=args.local_poly_em_prior_mix,
            em_update_damping=args.local_poly_em_update_damping,
            basis=args.local_poly_basis,
            pol_reg_type=args.local_poly_pol_reg_type,
            pol_reg_eta=args.local_poly_pol_reg_eta,
            pol_reg_power=args.local_poly_pol_reg_power,
        )
        half_estimates.append(estimates.reshape(-1, *dataset.volume_shape).astype(np.float32))
        half_diagnostics.append(diagnostics)

    avg_estimates = ((half_estimates[0] + half_estimates[1]) * 0.5).astype(np.float32)
    for subdir, volumes in (
        ("estimates_half1_unfil", half_estimates[0]),
        ("estimates_half2_unfil", half_estimates[1]),
        ("estimates_filt", avg_estimates),
    ):
        out_dir = state_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        output_mod.save_volumes(
            volumes,
            os.path.join(out_dir, ""),
            dataset.volume_shape,
            from_ft=False,
            voxel_size=dataset.voxel_size,
        )

    metadata = {}
    metadata_path = args.run_dir / "index_noisy_embedding_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    params = {
        "kernel_regression_mode": "local_poly_em",
        "heterogeneity_bins": h_grid,
        "n_images_per_bin": n_images_per_h,
        "voxel_size": float(dataset.voxel_size),
        "volume_shape": tuple(dataset.volume_shape),
        "candidate_note": (
            "Direct local_poly EM upsampling=1 candidate averages. Stored in estimates_filt "
            "for report compatibility; this intentionally skips compute_state final recombination."
        ),
        "local_poly_em": {
            "degree": int(args.local_poly_degree),
            "n_iterations": int(args.local_poly_em_iterations),
            "n_quadrature": int(args.local_poly_em_quadrature),
            "basis": str(args.local_poly_basis),
            "pol_reg_type": str(args.local_poly_pol_reg_type),
            "pol_reg_eta": float(args.local_poly_pol_reg_eta),
            "pol_reg_power": float(args.local_poly_pol_reg_power),
            "em_temperature": float(args.local_poly_em_temperature),
            "em_prior_mix": float(args.local_poly_em_prior_mix),
            "em_update_damping": float(args.local_poly_em_update_damping),
            "tau": tau_metadata,
            "h_grid": h_grid,
            "h_state_grid": (
                (h_grid / float(metadata["scaled_index_slope"]))
                if "scaled_index_slope" in metadata
                else None
            ),
            "requested_h_state_grid": args.local_poly_h_state_grid,
            "requested_h_scaled_grid": args.local_poly_h_scaled_grid,
            "sigma_ref": float(sigma_ref),
            "h_min": float(h_min),
            "r_min": float(r_min),
            "half_diagnostics": half_diagnostics,
        },
    }
    utils.pickle_dump(params, state_dir / "params.pkl")
    write_json(
        local_poly_em_out / "job.json",
        {
            "mode": "local_poly_em_direct_candidates",
            "pipeline": str(gt_pipeline),
            "target_point": str(target_point),
            "candidate_count": int(h_grid.size),
            "params": str(state_dir / "params.pkl"),
        },
    )
    print(f"LOCAL_POLY_EM_DIRECT_DONE out={local_poly_em_out}", flush=True)


def write_three_mode_report(
    args: argparse.Namespace,
    gt_pipeline: Path,
    standard_out: Path,
    deconv_out: Path,
    local_poly_out: Path,
    local_poly_em_out: Path | None,
    target_point: Path,
    mask_path: Path,
) -> None:
    report_out = three_mode_report_dir(args)
    if (report_out / "summary.json").exists() and not args.overwrite_report:
        print(f"Reusing three-mode report at {report_out}", flush=True)
        return

    plots_dir = report_out / "plots"
    report_out.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    target_vol = args.run_dir / f"04_ground_truth/gt_vol{args.target_state:04d}.mrc"
    target = np.asarray(utils.load_mrc(target_vol), dtype=np.float32)
    mask = np.asarray(utils.load_mrc(mask_path), dtype=np.float32)
    if target.shape != mask.shape:
        raise ValueError(f"Target shape {target.shape} and mask shape {mask.shape} differ")

    roots = {
        "standard": standard_out,
        "deconvolved": deconv_out,
        "local_poly": local_poly_out,
    }
    if local_poly_em_out is not None and (local_poly_em_out / "job.json").exists():
        roots["local_poly_em"] = local_poly_em_out
    colors = {
        "standard": "tab:green",
        "deconvolved": "tab:blue",
        "local_poly": "tab:orange",
        "local_poly_em": "tab:red",
    }
    cmaps = {
        "standard": plt.cm.Greens,
        "deconvolved": plt.cm.Blues,
        "local_poly": plt.cm.Oranges,
        "local_poly_em": plt.cm.Reds,
    }

    all_metrics = {}
    rows = []
    score_mask = None
    voxel_size = None
    for mode, root in roots.items():
        grid = _candidate_grid(root)
        freq, fsc, err = _load_candidate_metrics(root, target, mask)
        if voxel_size is None:
            voxel_size = float(utils.pickle_load(_state_dir(root) / "params.pkl")["voxel_size"])
        score_mask = (freq > 0.0) & (freq <= args.report_score_frequency_max)
        mean_fsc = np.nanmean(fsc[:, score_mask], axis=1)
        median_fsc = np.nanmedian(fsc[:, score_mask], axis=1)
        median_error = np.nanmedian(err[:, score_mask], axis=1)
        all_metrics[mode] = {
            "grid": grid,
            "freq": freq,
            "fsc": fsc,
            "err": err,
            "mean_fsc": mean_fsc,
            "median_fsc": median_fsc,
            "median_error": median_error,
        }
        for idx in range(fsc.shape[0]):
            rows.append(
                {
                    "mode": mode,
                    "candidate_index_0based": int(idx),
                    "candidate_index_1based": int(idx + 1),
                    "parameter": float(grid[idx]) if idx < grid.size else np.nan,
                    "mean_fsc_0_to_score_max": float(mean_fsc[idx]),
                    "median_fsc_0_to_score_max": float(median_fsc[idx]),
                    "median_relative_error_0_to_score_max": float(median_error[idx]),
                    "fsc_shell32": float(fsc[idx, 32]) if fsc.shape[1] > 32 else np.nan,
                    "relative_error_shell32": float(err[idx, 32]) if err.shape[1] > 32 else np.nan,
                    "path": str(_candidate_paths(root)[idx]),
                }
            )

    mean_fsc = None
    mean_err = None
    mean_stack = args.run_dir / "04_ground_truth/gt_volumes_used_by_simulator.npy"
    if mean_stack.exists():
        distribution_mean = skr._mean_volume_from_stack(mean_stack, target.shape)
        labels, n_shells = skr._shell_labels(target.shape)
        target_ft = skr._numpy_dft3(target * mask)
        target_power = np.bincount(labels.ravel(), weights=np.abs(target_ft).ravel() ** 2, minlength=n_shells)
        mean_fsc, mean_err = skr._masked_metrics_for_volume(distribution_mean, target, mask, labels, n_shells, target_ft, target_power)

    import csv

    csv_path = report_out / "candidate_metrics.csv"
    fieldnames = [
        "mode",
        "candidate_index_0based",
        "candidate_index_1based",
        "parameter",
        "mean_fsc_0_to_score_max",
        "median_fsc_0_to_score_max",
        "median_relative_error_0_to_score_max",
        "fsc_shell32",
        "relative_error_shell32",
        "path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = {}
    for mode, metrics in all_metrics.items():
        best_fsc = int(np.nanargmax(metrics["mean_fsc"]))
        best_err = int(np.nanargmin(metrics["median_error"]))
        best[mode] = {
            "best_by_mean_fsc_index_0based": best_fsc,
            "best_by_mean_fsc_parameter": float(metrics["grid"][best_fsc]),
            "best_by_mean_fsc": float(metrics["mean_fsc"][best_fsc]),
            "best_by_mean_fsc_shell32": float(metrics["fsc"][best_fsc, 32]) if metrics["fsc"].shape[1] > 32 else None,
            "best_by_median_error_index_0based": best_err,
            "best_by_median_error_parameter": float(metrics["grid"][best_err]),
            "best_by_median_error": float(metrics["median_error"][best_err]),
            "best_by_median_error_shell32_fsc": float(metrics["fsc"][best_err, 32]) if metrics["fsc"].shape[1] > 32 else None,
        }

    oracle_rows = []
    oracle_summary = {}
    shell_indices = np.arange(next(iter(all_metrics.values()))["freq"].size)
    combined_candidates = []
    combined_labels = []
    combined_freq = None
    for mode, metrics in all_metrics.items():
        fsc = metrics["fsc"]
        err = metrics["err"]
        freq = metrics["freq"]
        combined_freq = freq
        combined_candidates.append(fsc)
        combined_labels.extend([(mode, idx) for idx in range(fsc.shape[0])])

        fsc_choice = np.nanargmax(np.where(np.isfinite(fsc), fsc, -np.inf), axis=0)
        err_choice = np.nanargmin(np.where(np.isfinite(err), err, np.inf), axis=0)
        fsc_oracle = fsc[fsc_choice, shell_indices]
        err_choice_fsc = fsc[err_choice, shell_indices]
        err_oracle = err[err_choice, shell_indices]
        oracle_summary[mode] = {
            "fsc_oracle_mean_0_to_score_max": float(np.nanmean(fsc_oracle[score_mask])),
            "fsc_oracle_shell32": float(fsc_oracle[32]) if fsc_oracle.size > 32 else None,
            "error_choice_fsc_mean_0_to_score_max": float(np.nanmean(err_choice_fsc[score_mask])),
            "error_choice_fsc_shell32": float(err_choice_fsc[32]) if err_choice_fsc.size > 32 else None,
            "error_oracle_median_relative_error_0_to_score_max": float(np.nanmedian(err_oracle[score_mask])),
            "fsc_oracle_candidate_indices_0based": [int(x) for x in np.unique(fsc_choice[score_mask])],
            "error_oracle_candidate_indices_0based": [int(x) for x in np.unique(err_choice[score_mask])],
        }

        for shell, freq_value, fsc_idx, err_idx, fsc_value, err_fsc_value, err_value in zip(
            shell_indices,
            freq,
            fsc_choice,
            err_choice,
            fsc_oracle,
            err_choice_fsc,
            err_oracle,
        ):
            oracle_rows.append(
                {
                    "mode": mode,
                    "shell": int(shell),
                    "frequency_1_per_A": float(freq_value),
                    "fsc_oracle_candidate_index_0based": int(fsc_idx),
                    "fsc_oracle_parameter": float(metrics["grid"][fsc_idx]),
                    "fsc_oracle_fsc": float(fsc_value),
                    "error_oracle_candidate_index_0based": int(err_idx),
                    "error_oracle_parameter": float(metrics["grid"][err_idx]),
                    "error_oracle_fsc": float(err_fsc_value),
                    "error_oracle_relative_error": float(err_value),
                }
            )

    combined_stack = np.concatenate(combined_candidates, axis=0)
    combined_choice = np.nanargmax(np.where(np.isfinite(combined_stack), combined_stack, -np.inf), axis=0)
    combined_oracle_fsc = combined_stack[combined_choice, shell_indices]
    combined_modes = [combined_labels[int(idx)][0] for idx in combined_choice]
    combined_indices = [combined_labels[int(idx)][1] for idx in combined_choice]
    oracle_summary["combined_fsc_oracle"] = {
        "fsc_oracle_mean_0_to_score_max": float(np.nanmean(combined_oracle_fsc[score_mask])),
        "fsc_oracle_shell32": float(combined_oracle_fsc[32]) if combined_oracle_fsc.size > 32 else None,
        "selected_modes_0_to_score_max": sorted(set(combined_modes[idx] for idx in np.flatnonzero(score_mask))),
    }

    oracle_csv_path = report_out / "oracle_shell_choices.csv"
    oracle_fields = [
        "mode",
        "shell",
        "frequency_1_per_A",
        "fsc_oracle_candidate_index_0based",
        "fsc_oracle_parameter",
        "fsc_oracle_fsc",
        "error_oracle_candidate_index_0based",
        "error_oracle_parameter",
        "error_oracle_fsc",
        "error_oracle_relative_error",
    ]
    with oracle_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=oracle_fields)
        writer.writeheader()
        writer.writerows(oracle_rows)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    for mode, metrics in all_metrics.items():
        fsc = metrics["fsc"]
        fsc_choice = np.nanargmax(np.where(np.isfinite(fsc), fsc, -np.inf), axis=0)
        fsc_oracle = fsc[fsc_choice, shell_indices]
        ax.plot(metrics["freq"], fsc_oracle, color=colors[mode], linewidth=2.2, label=f"{mode} per-shell FSC oracle")
    ax.plot(combined_freq, combined_oracle_fsc, color="black", linewidth=2.4, linestyle="--", label="combined per-shell FSC oracle")
    if mean_fsc is not None:
        ax.plot(combined_freq, mean_fsc, color="0.25", linestyle=":", linewidth=2.0, label="distribution mean")
    ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.8)
    ax.axhline(1 / 7, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_xlim(0.0, args.report_plot_frequency_max)
    ax.set_ylim(-0.08, 1.03)
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_ylabel("FSC vs GT")
    ax.set_title("Per-shell oracle FSC curves: best candidate at each frequency")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    oracle_fsc_plot = plots_dir / "oracle_fsc_curves_by_method.png"
    fig.savefig(oracle_fsc_plot, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    for mode, metrics in all_metrics.items():
        fsc = metrics["fsc"]
        err = metrics["err"]
        err_choice = np.nanargmin(np.where(np.isfinite(err), err, np.inf), axis=0)
        err_choice_fsc = fsc[err_choice, shell_indices]
        ax.plot(metrics["freq"], err_choice_fsc, color=colors[mode], linewidth=2.2, label=f"{mode} min-error candidate FSC")
    if mean_fsc is not None:
        ax.plot(combined_freq, mean_fsc, color="0.25", linestyle=":", linewidth=2.0, label="distribution mean")
    ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.8)
    ax.axhline(1 / 7, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_xlim(0.0, args.report_plot_frequency_max)
    ax.set_ylim(-0.08, 1.03)
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_ylabel("FSC vs GT")
    ax.set_title("Per-shell oracle by minimum relative error, plotted as FSC")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    oracle_error_plot = plots_dir / "oracle_min_error_choice_fsc_curves_by_method.png"
    fig.savefig(oracle_error_plot, dpi=180)
    plt.close(fig)
    write_json(
        report_out / "oracle_summary.json",
        {
            "oracle_shell_choices_csv": str(oracle_csv_path),
            "oracle": oracle_summary,
            "plots": {
                "oracle_fsc_curves_by_method": str(oracle_fsc_plot),
                "oracle_min_error_choice_fsc_curves_by_method": str(oracle_error_plot),
            },
        },
    )

    fig, axes = plt.subplots(
        len(all_metrics),
        1,
        figsize=(11, 3.8 * len(all_metrics)),
        sharex=True,
        constrained_layout=True,
    )
    if len(all_metrics) == 1:
        axes = [axes]
    for ax, (mode, metrics) in zip(axes, all_metrics.items()):
        grid = metrics["grid"]
        fsc = metrics["fsc"]
        cmap = cmaps[mode]
        norm = mcolors.Normalize(vmin=0, vmax=max(fsc.shape[0] - 1, 1))
        for idx in range(fsc.shape[0]):
            ax.plot(metrics["freq"], fsc[idx], color=cmap(norm(idx)), linewidth=0.9, alpha=0.78)
        best_idx = best[mode]["best_by_mean_fsc_index_0based"]
        ax.plot(
            metrics["freq"],
            fsc[best_idx],
            color="black",
            linewidth=2.2,
            label=f"best mean FSC #{best_idx + 1}, param={grid[best_idx]:.4g}",
        )
        if mean_fsc is not None:
            ax.plot(metrics["freq"], mean_fsc, color="0.25", linestyle="--", linewidth=1.8, label="distribution mean")
        ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.8)
        ax.axhline(1 / 7, color="0.55", linestyle=":", linewidth=0.8)
        ax.set_xlim(0.0, args.report_plot_frequency_max)
        ax.set_ylim(-0.08, 1.03)
        ax.set_ylabel("FSC vs GT")
        ax.set_title(f"{mode}: every candidate FSC vs GT")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.01).set_label("candidate index")
    axes[-1].set_xlabel("spatial frequency (1/A)")
    fig.suptitle("All candidates, crafted mask, true GT FSC", fontsize=14, fontweight="bold")
    fig.savefig(plots_dir / "all_candidate_fsc_three_modes.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for mode, metrics in all_metrics.items():
        envelope = np.nanmax(metrics["fsc"], axis=0)
        ax.plot(metrics["freq"], envelope, color=colors[mode], linewidth=2.0, label=f"{mode} max over candidates")
    if mean_fsc is not None:
        ax.plot(next(iter(all_metrics.values()))["freq"], mean_fsc, color="0.25", linestyle="--", linewidth=2.0, label="distribution mean")
    ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.8)
    ax.axhline(1 / 7, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_xlim(0.0, args.report_plot_frequency_max)
    ax.set_ylim(-0.08, 1.03)
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_ylabel("FSC vs GT")
    ax.set_title("FSC envelope: max over all candidates at each shell")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(plots_dir / "fsc_envelope_max_over_candidates.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for mode, metrics in all_metrics.items():
        idx = best[mode]["best_by_mean_fsc_index_0based"]
        ax.plot(metrics["freq"], metrics["fsc"][idx], color=colors[mode], linewidth=2.0, label=f"{mode} best mean FSC")
    if mean_fsc is not None:
        ax.plot(next(iter(all_metrics.values()))["freq"], mean_fsc, color="0.25", linestyle="--", linewidth=2.0, label="distribution mean")
    ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.8)
    ax.axhline(1 / 7, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_xlim(0.0, args.report_plot_frequency_max)
    ax.set_ylim(-0.08, 1.03)
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_ylabel("FSC vs GT")
    ax.set_title("Best candidate by mean FSC, compared across modes")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(plots_dir / "best_mean_fsc_candidates.png", dpi=180)
    plt.close(fig)

    gt_sweep = None
    if not args.skip_gt_polynomial_sweep:
        gt_sweep = write_gt_polynomial_sweep(args, mask_path, report_out)

    local_poly_grid = all_metrics["local_poly"]["grid"]
    diagnostic_h_values = [float(local_poly_grid[0])]
    if gt_sweep is not None:
        diagnostic_h_values.append(float(gt_sweep["best_leave_target_out_degree3_by_mean_fsc"]["h_scaled"]))
    if local_poly_grid.size > 2:
        diagnostic_h_values.append(float(local_poly_grid[-1]))
    diagnostic_h_values = sorted(set(round(float(x), 8) for x in diagnostic_h_values))
    voxel_poly_plot = write_gt_shell_voxel_polynomial_diagnostic(
        args,
        mask_path,
        report_out,
        diagnostic_h_values,
    )

    summary = {
        "run_dir": str(args.run_dir),
        "pipeline": str(gt_pipeline),
        "target_point": str(target_point),
        "target_volume": str(target_vol),
        "mask": str(mask_path),
        "score_frequency_max": float(args.report_score_frequency_max),
        "plot_frequency_max": float(args.report_plot_frequency_max),
        "candidate_metrics_csv": str(csv_path),
        "best": best,
        "oracle_shell_choices_csv": str(oracle_csv_path),
        "oracle": oracle_summary,
        "plots": {
            "all_candidate_fsc_three_modes": str(plots_dir / "all_candidate_fsc_three_modes.png"),
            "fsc_envelope_max_over_candidates": str(plots_dir / "fsc_envelope_max_over_candidates.png"),
            "best_mean_fsc_candidates": str(plots_dir / "best_mean_fsc_candidates.png"),
            "oracle_fsc_curves_by_method": str(oracle_fsc_plot),
            "oracle_min_error_choice_fsc_curves_by_method": str(oracle_error_plot),
            "gt_shell32_voxel_polynomial_fit": str(voxel_poly_plot) if voxel_poly_plot is not None else None,
        },
        "gt_polynomial_sweep": gt_sweep,
    }
    write_json(report_out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def run_compute_and_report(args: argparse.Namespace, gt_pipeline: Path, target_point: Path, mask_path: Path) -> None:
    suffix = "_lazy" if args.compute_state_lazy else ""
    standard_out = args.run_dir / f"07_compute_state_standard_index_gtnoise{suffix}"
    deconv_out = args.run_dir / f"07_compute_state_deconvolved_index_gtnoise_lam0p2_20{suffix}"
    local_poly_out = local_poly_output_dir(args)
    local_poly_em_out = local_poly_em_output_dir(args) if args.local_poly_em else None
    report_out = args.run_dir / f"08_kernel_report_mask_crafted_level0p0126_cosine3{suffix}"
    target_vol = args.run_dir / f"04_ground_truth/gt_vol{args.target_state:04d}.mrc"

    if not (standard_out / "job.json").exists():
        cmd = [
            sys.executable,
            "-m",
            "recovar.commands.compute_state",
            str(gt_pipeline),
            "-o",
            str(standard_out),
            "--latent-points",
            str(target_point),
            "--zdim1",
            "--save-all-estimates",
            "--kernel-regression-mode",
            "standard",
        ]
        if args.compute_state_lazy:
            cmd.append("--lazy")
        run_command(cmd)
    else:
        print(f"Reusing standard compute_state at {standard_out}", flush=True)

    if not (deconv_out / "job.json").exists():
        cmd = [
            sys.executable,
            "-m",
            "recovar.commands.compute_state",
            str(gt_pipeline),
            "-o",
            str(deconv_out),
            "--latent-points",
            str(target_point),
            "--zdim1",
            "--save-all-estimates",
            "--kernel-regression-mode",
            "deconvolved",
            "--deconv-lambda-grid",
            args.lambda_grid,
        ]
        if args.compute_state_lazy:
            cmd.append("--lazy")
        run_command(cmd)
    else:
        print(f"Reusing deconvolved compute_state at {deconv_out}", flush=True)

    compute_local_poly_candidates_direct(args, gt_pipeline, local_poly_out, target_point)
    if args.local_poly_em:
        compute_local_poly_em_candidates_direct(args, gt_pipeline, local_poly_em_out, target_point)

    if not (report_out / "summary.json").exists():
        run_command(
            [
                sys.executable,
                "-m",
                "recovar.commands.spike_kernel_report",
                "--standard-root",
                str(standard_out),
                "--deconvolved-root",
                str(deconv_out),
                "--pipeline-root",
                str(gt_pipeline),
                "--target-point",
                str(target_point),
                "--target-volume",
                str(target_vol),
                "--mask",
                str(mask_path),
                "--out-dir",
                str(report_out),
                "--expected-candidates",
                "50",
                "--score-frequency-max",
                "0.18",
                "--plot-frequency-max",
                "0.20",
                "--report-title",
                (
                    f"grid{args.grid_size} raw volumes {args.n_images // 1000}k "
                    f"{'no CTF, ' if args.no_ctf else ''}"
                    f"image noise{args.noise_level:g} state-index embedding + artificial noise, crafted mask"
                ),
            ]
        )
    else:
        print(f"Reusing report at {report_out}", flush=True)

    write_three_mode_report(
        args,
        gt_pipeline,
        standard_out,
        deconv_out,
        local_poly_out,
        local_poly_em_out,
        target_point,
        mask_path,
    )

    print(f"RUN_DIR={args.run_dir}", flush=True)
    print(f"GT_PIPELINE={gt_pipeline}", flush=True)
    print(f"TARGET_POINT={target_point}", flush=True)
    print(f"STANDARD_OUT={standard_out}", flush=True)
    print(f"DECONV_OUT={deconv_out}", flush=True)
    print(f"LOCAL_POLY_OUT={local_poly_out}", flush=True)
    if local_poly_em_out is not None:
        print(f"LOCAL_POLY_EM_OUT={local_poly_em_out}", flush=True)
    print(f"REPORT_OUT={report_out}", flush=True)
    print(f"THREE_MODE_REPORT_OUT={three_mode_report_dir(args)}", flush=True)
    print(f"MASK={mask_path}", flush=True)
    print(f"TARGET_VOL={target_vol}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pdb-dir", type=Path, default=Path("/home/mg6942/myscratch/spike_pdb_motion"))
    parser.add_argument("--reference-raw-dir", type=Path, default=None)
    parser.add_argument("--reference-gt-metadata", type=Path, default=None)
    parser.add_argument("--crafted-mask", type=Path, default=Path("/scratch/gpfs/GILLES/mg6942/uploads/mask_crafted.mrc"))
    parser.add_argument("--crafted-mask-level", type=float, default=0.0126)
    parser.add_argument("--crafted-mask-cosine-width", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--voxel-size", type=float, default=2.5)
    parser.add_argument("--n-states", type=int, default=100)
    parser.add_argument("--n-images", type=int, default=300000)
    parser.add_argument("--noise-level", type=float, default=30.0)
    parser.add_argument("--noise-variance-floor", type=float, default=None)
    parser.add_argument("--noise-model", type=str, default="radial1")
    parser.add_argument("--no-ctf", action="store_true")
    parser.add_argument("--render-bfactor", type=float, default=100.0)
    parser.add_argument("--target-state", type=int, default=50)
    parser.add_argument("--embedding-seed", type=int, default=20260513)
    parser.add_argument("--embedding-noise-std", type=float, default=None)
    parser.add_argument("--embedding-noise-from-reference-metadata", action="store_true")
    parser.add_argument("--stream-simulation", action="store_true")
    parser.add_argument("--compute-state-lazy", action="store_true")
    parser.add_argument("--local-poly-degree", type=int, default=3)
    parser.add_argument(
        "--local-poly-basis",
        choices=local_polynomial_regression.LOCAL_POLY_BASIS_OPTIONS,
        default=local_polynomial_regression.DEFAULT_LOCAL_POLY_BASIS,
    )
    parser.add_argument(
        "--local-poly-pol-reg-type",
        choices=local_polynomial_regression.LOCAL_POLY_POL_REG_TYPES,
        default="none",
    )
    parser.add_argument("--local-poly-pol-reg-eta", type=float, default=0.0)
    parser.add_argument("--local-poly-pol-reg-power", type=float, default=2.0)
    parser.add_argument("--local-poly-maskrad-fraction", type=float, default=4.0)
    parser.add_argument("--local-poly-output-tag", type=str, default=None)
    parser.add_argument("--local-poly-em", action="store_true")
    parser.add_argument("--local-poly-em-iterations", type=int, default=2)
    parser.add_argument("--local-poly-em-quadrature", type=int, default=5)
    parser.add_argument("--local-poly-em-temperature", type=float, default=1.0)
    parser.add_argument("--local-poly-em-prior-mix", type=float, default=0.0)
    parser.add_argument("--local-poly-em-update-damping", type=float, default=1.0)
    parser.add_argument(
        "--local-poly-em-tau-mode",
        choices=["none", "mean_power", "gt_target_power"],
        default="none",
    )
    parser.add_argument("--local-poly-em-tau-scale", type=float, default=1.0)
    parser.add_argument("--local-poly-em-tau-floor-relative", type=float, default=1e-8)
    parser.add_argument(
        "--local-poly-h-state-grid",
        type=str,
        default=None,
        help="Comma-separated local_poly bandwidths in raw GT state-index units; converted to scaled latent units.",
    )
    parser.add_argument(
        "--local-poly-h-scaled-grid",
        type=str,
        default=None,
        help="Comma-separated local_poly bandwidths directly in scaled latent units.",
    )
    parser.add_argument(
        "--local-poly-bandwidth-multipliers",
        type=str,
        default="1,1.25,1.5,2,3,4,6,8,12,16",
    )
    parser.add_argument("--report-score-frequency-max", type=float, default=0.18)
    parser.add_argument("--report-plot-frequency-max", type=float, default=0.20)
    parser.add_argument("--overwrite-report", action="store_true")
    parser.add_argument(
        "--skip-gt-polynomial-sweep",
        action="store_true",
        help="Skip the expensive GT state polynomial diagnostic in per-parameter reports.",
    )
    parser.add_argument(
        "--lambda-grid",
        type=str,
        default=(
            "0.2,0.21970823,0.24135853,0.26514227,0.2912697,0.31997174,"
            "0.35150212,0.38613955,0.42419018,0.46599036,0.51190958,"
            "0.56235374,0.61776872,0.67864435,0.74551874,0.81898301,"
            "0.89968653,0.98834267,1.0857351,1.1927247,1.3102571,"
            "1.4393713,1.5812086,1.7370227,1.908191,2.0962263,2.3027908,"
            "2.5297104,2.778991,3.0528359,3.3536659,3.6841399,4.0471793,"
            "4.445993,4.8841062,5.3653916,5.8941034,6.4749151,7.1129606,"
            "7.8138799,8.5838685,9.4297327,10.358949,11.379732,12.501104,"
            "13.732977,15.08624,16.572855,18.205964,20"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    source_pipeline = ensure_source_pipeline(args)
    gt_pipeline = args.run_dir / f"06_pipeline_index_scaled_meansigma_seed{args.embedding_seed}"
    scale_state_index_embedding(args, source_pipeline, gt_pipeline)
    mask_path = write_crafted_mask(args)
    target_point = args.run_dir / f"target_latent_point_index_state{args.target_state:04d}.txt"
    run_compute_and_report(args, gt_pipeline, target_point, mask_path)


if __name__ == "__main__":
    main()
