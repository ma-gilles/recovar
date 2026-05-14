from __future__ import annotations

import argparse
import json
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
import mrcfile
import numpy as np
from scipy.ndimage import center_of_mass

from recovar import utils
from recovar.commands import render_spike_morph_volumes as render_cmd
from recovar.commands import spike_walkthrough as sw
from recovar.core import mask as mask_utils
from recovar.simulation import simulator


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


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

    with mrcfile.new_mmap(
        particles_path,
        shape=(args.n_images, args.grid_size, args.grid_size),
        mrc_mode=2,
        overwrite=True,
    ) as mrc:
        mrc.voxel_size = voxel_size
        simulator.simulate_data(
            dataset,
            volumes,
            noise_variance,
            batch_size,
            image_assignments,
            per_image_contrast,
            per_image_noise_scale,
            seed=0,
            disc_type="cubic",
            mrc_file=mrc,
            premultiplied_ctf=args.premultiplied_ctf,
        )
        mrc.flush()

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


def run_compute_and_report(args: argparse.Namespace, gt_pipeline: Path, target_point: Path, mask_path: Path) -> None:
    suffix = "_lazy" if args.compute_state_lazy else ""
    standard_out = args.run_dir / f"07_compute_state_standard_index_gtnoise{suffix}"
    deconv_out = args.run_dir / f"07_compute_state_deconvolved_index_gtnoise_lam0p2_20{suffix}"
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
                "--package-lowpass",
                "--lowpass-cutoff",
                "0.15",
            ]
        )
    else:
        print(f"Reusing report at {report_out}", flush=True)

    print(f"RUN_DIR={args.run_dir}", flush=True)
    print(f"GT_PIPELINE={gt_pipeline}", flush=True)
    print(f"TARGET_POINT={target_point}", flush=True)
    print(f"STANDARD_OUT={standard_out}", flush=True)
    print(f"DECONV_OUT={deconv_out}", flush=True)
    print(f"REPORT_OUT={report_out}", flush=True)
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
