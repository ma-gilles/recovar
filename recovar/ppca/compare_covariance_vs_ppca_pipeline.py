#!/usr/bin/env python
"""Compare covariance-PCA and direct PPCA pipeline runs on matched synthetic datasets.

This driver generates a synthetic CryoBench-style dataset from the same source
volumes used by the historical PPCA sweep, runs the standard covariance
pipeline and the new direct-PPCA pipeline on the exact same particles, then
scores both against the GT structural subspace.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from PIL import Image

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import utils
from recovar.commands import compute_state
from recovar.commands import run_test_all_metrics as test_metrics
from recovar.output import metrics
from recovar.output import output as output_module
from recovar.output import plot_utils
from recovar.simulation import simulator, synthetic_dataset


DEFAULT_BASE_DIR = "/home/mg6942/mytigress/cryobench2"
DEFAULT_RESULTS_ROOT = "/scratch/gpfs/GILLES/mg6942/ppca_pipeline_compare"
logger = logging.getLogger(__name__)

METHOD_ORDER = ("covariance", "ppca", "ppca_projected_covariance")
METHOD_LABELS = {
    "covariance": "Covariance",
    "ppca": "PPCA",
    "ppca_projected_covariance": "PPCA+ProjCov",
}
METHOD_COLORS = {
    "covariance": "#d55e00",
    "ppca": "#0072b2",
    "ppca_projected_covariance": "#009e73",
}


def _with_trailing_separator(path: str) -> str:
    return path if path.endswith("/") else path + "/"


def _contrast_tag(contrast_std: float) -> str:
    return f"c{contrast_std:.2f}".replace(".", "p")


def _build_run_name(dataset_name: str, grid_size: int, n_images: int, noise_level: float, contrast_std: float, zdim: int, seed: int) -> str:
    return (
        f"{dataset_name}_g{grid_size}_n{n_images}_snr{noise_level}"
        f"_{_contrast_tag(contrast_std)}_z{zdim}_seed{seed}"
    )


def _find_cryobench_dataset(base_dir: str, dataset_name: str) -> dict:
    dataset_dir = os.path.join(base_dir, dataset_name)
    vol_dir = os.path.join(dataset_dir, "vols", "128_org")
    if not os.path.isdir(vol_dir):
        raise FileNotFoundError(f"Could not find CryoBench volume directory: {vol_dir}")
    mrc_files = sorted(Path(vol_dir).glob("*.mrc"))
    if not mrc_files:
        raise FileNotFoundError(f"No .mrc volumes found in {vol_dir}")
    return {"name": dataset_name, "vol_dir": vol_dir, "n_volumes": len(mrc_files)}


def _ensure_fixed_volume_headers(dataset_info: dict, fixed_vol_dir: str, voxel_size: float) -> None:
    os.makedirs(fixed_vol_dir, exist_ok=True)
    for idx, mrc_path in enumerate(sorted(Path(dataset_info["vol_dir"]).glob("*.mrc"))):
        fixed_path = os.path.join(fixed_vol_dir, f"{idx}.mrc")
        if os.path.isfile(fixed_path):
            continue
        with mrcfile.open(mrc_path, permissive=True) as src:
            data = src.data.copy()
        with mrcfile.new(fixed_path, overwrite=True) as dst:
            dst.set_data(data)
            dst.voxel_size = voxel_size


def generate_dataset(
    dataset_info: dict,
    output_root: str,
    grid_size: int,
    n_images: int,
    noise_level: float,
    contrast_std: float,
    seed: int,
) -> str:
    """Generate or reuse a synthetic dataset rooted at ``output_root/simulated_data``."""
    sim_dir = os.path.join(output_root, "simulated_data")
    particles_file = os.path.join(sim_dir, f"particles.{grid_size}.mrcs")
    sim_info_file = os.path.join(sim_dir, "simulation_info.pkl")
    if os.path.isfile(particles_file) and os.path.isfile(sim_info_file):
        return sim_dir

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(sim_dir, exist_ok=True)

    np.random.seed(seed)
    voxel_size = 4.25 * 128 / grid_size
    fixed_vol_dir = os.path.join(output_root, "volumes_fixed")
    _ensure_fixed_volume_headers(dataset_info, fixed_vol_dir, voxel_size)

    simulator.generate_synthetic_dataset(
        sim_dir,
        voxel_size,
        _with_trailing_separator(fixed_vol_dir),
        n_images,
        outlier_file_input=None,
        grid_size=grid_size,
        volume_distribution=None,
        dataset_params_option="uniform",
        noise_level=noise_level,
        noise_model="white",
        put_extra_particles=False,
        percent_outliers=0.0,
        volume_radius=0.7,
        trailing_zero_format_in_vol_name=False,
        noise_scale_std=0.0,
        contrast_std=contrast_std,
        disc_type="nufft",
    )

    run_info = {
        "dataset": dataset_info["name"],
        "grid_size": grid_size,
        "n_images": n_images,
        "noise_level": noise_level,
        "contrast_std": contrast_std,
        "seed": seed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_root, "dataset_config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_info, fh, indent=2)
    return sim_dir


def _pipeline_output_dir(method_root: str) -> str:
    return os.path.join(method_root, "result")


def _halfsets_path(output_root: str) -> str:
    return os.path.join(output_root, "halfsets.pkl")


def ensure_halfsets(output_root: str, n_images: int, seed: int) -> str:
    """Create one deterministic halfset split shared by covariance and PPCA runs."""
    halfsets_path = _halfsets_path(output_root)
    if os.path.isfile(halfsets_path):
        return halfsets_path

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_images)
    split = n_images // 2
    halfsets = [
        np.sort(perm[:split]).astype(np.int32),
        np.sort(perm[split:]).astype(np.int32),
    ]
    utils.pickle_dump(halfsets, halfsets_path)
    return halfsets_path


def build_pipeline_command(
    method: str,
    sim_dir: str,
    method_root: str,
    grid_size: int,
    halfsets_path: str,
    zdim: int,
    ppca_em_iters: int,
    use_contrast: bool,
    gpu_gb: float | None,
    low_memory_option: bool,
    very_low_memory_option: bool,
    lazy: bool,
    force: bool,
) -> tuple[list[str], str, str]:
    """Build one pipeline command and return ``(cmd, result_dir, log_path)``."""
    result_dir = _pipeline_output_dir(method_root)
    log_path = os.path.join(method_root, f"{method}.log")
    cmd = [
        sys.executable,
        "-m",
        "recovar.commands.pipeline",
        os.path.join(sim_dir, f"particles.{grid_size}.mrcs"),
        "-o",
        result_dir,
        "--mask",
        "from_halfmaps",
        "--poses",
        os.path.join(sim_dir, "poses.pkl"),
        "--ctf",
        os.path.join(sim_dir, "ctf.pkl"),
        "--halfsets",
        halfsets_path,
        "--zdim",
        str(zdim),
    ]
    if gpu_gb is not None:
        cmd.extend(["--gpu-gb", str(gpu_gb)])
    if low_memory_option:
        cmd.append("--low-memory-option")
    if very_low_memory_option:
        cmd.append("--very-low-memory-option")
    if lazy:
        cmd.append("--lazy")
    if use_contrast:
        cmd.append("--correct-contrast")
    if method == "ppca":
        cmd.extend(["--use-ppca", "--ppca-zdim", str(zdim), "--ppca-em-iters", str(ppca_em_iters)])
    elif method == "ppca_projected_covariance":
        cmd.extend(
            [
                "--use-ppca",
                "--ppca-zdim",
                str(zdim),
                "--ppca-em-iters",
                str(ppca_em_iters),
                "--ppca-projected-covariance",
            ]
        )
    elif method != "covariance":
        raise ValueError(f"Unknown method: {method}")
    return cmd, result_dir, log_path


def run_pipeline_method(
    method: str,
    sim_dir: str,
    method_root: str,
    grid_size: int,
    halfsets_path: str,
    zdim: int,
    ppca_em_iters: int,
    use_contrast: bool,
    gpu_gb: float | None,
    low_memory_option: bool,
    very_low_memory_option: bool,
    lazy: bool,
    force: bool,
) -> tuple[str, str, float | None]:
    """Run one pipeline method and return ``(result_dir, log_path, runtime_seconds)``."""
    cmd, result_dir, log_path = build_pipeline_command(
        method=method,
        sim_dir=sim_dir,
        method_root=method_root,
        grid_size=grid_size,
        halfsets_path=halfsets_path,
        zdim=zdim,
        ppca_em_iters=ppca_em_iters,
        use_contrast=use_contrast,
        gpu_gb=gpu_gb,
        low_memory_option=low_memory_option,
        very_low_memory_option=very_low_memory_option,
        lazy=lazy,
        force=force,
    )
    params_path = os.path.join(result_dir, "model", "params.pkl")
    if os.path.isfile(params_path) and not force:
        return result_dir, log_path, None

    if force and os.path.isdir(method_root):
        shutil.rmtree(method_root)
    os.makedirs(method_root, exist_ok=True)

    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log_fh:
        subprocess.run(cmd, check=True, stdout=log_fh, stderr=subprocess.STDOUT)
    runtime_seconds = time.time() - started
    return result_dir, log_path, runtime_seconds


def _write_method_runner_script(output_root: str, method_specs: list[dict]) -> str:
    script_path = os.path.join(output_root, "run_methods.sh")
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "run_one() {",
        "  local method=\"$1\"",
        "  local log_path=\"$2\"",
        "  local status_path=\"$3\"",
        "  shift 3",
        "  local -a cmd=(\"$@\")",
        "  echo \"[compare] running ${method}: ${cmd[*]}\"",
        "  set +e",
        "  \"${cmd[@]}\" >\"${log_path}\" 2>&1",
        "  local code=$?",
        "  set -e",
        "  printf '%s\\n' \"$code\" >\"${status_path}\"",
        "  if [ \"$code\" -ne 0 ]; then",
        "    echo \"[compare] method ${method} failed with exit code ${code}\" >&2",
        "  fi",
        "}",
        "",
    ]
    for spec in method_specs:
        cmd_parts = " ".join(shlex.quote(part) for part in spec["cmd"])
        params_path = os.path.join(spec["result_dir"], "model", "params.pkl")
        lines.extend(
            [
                f"mkdir -p {shlex.quote(os.path.dirname(spec['log_path']))} {shlex.quote(spec['result_dir'])}",
                f"if [ -f {shlex.quote(params_path)} ]; then",
                f"  echo \"[compare] skipping {spec['method']}; found existing {params_path}\"",
                f"  printf '0\\n' > {shlex.quote(spec['status_path'])}",
                "else",
                f"  run_one {shlex.quote(spec['method'])} {shlex.quote(spec['log_path'])} {shlex.quote(spec['status_path'])} {cmd_parts}",
                "fi",
                "",
            ]
        )
    Path(script_path).write_text("\n".join(lines), encoding="utf-8")
    os.chmod(script_path, 0o755)
    return script_path


def prepare_compare_run(args) -> dict:
    dataset_info = _find_cryobench_dataset(args.base_dir, args.dataset)
    run_name = _build_run_name(
        dataset_info["name"],
        args.grid_size,
        args.n_images,
        args.noise_level,
        args.contrast_std,
        args.zdim,
        args.seed,
    )
    output_root = os.path.join(args.results_root, run_name)
    sim_dir = generate_dataset(
        dataset_info,
        output_root,
        args.grid_size,
        args.n_images,
        args.noise_level,
        args.contrast_std,
        args.seed,
    )
    halfsets_path = ensure_halfsets(output_root, args.n_images, args.seed)
    use_contrast = args.contrast_std > 0

    method_specs = []
    for method in METHOD_ORDER:
        method_root = os.path.join(output_root, method)
        if args.force and os.path.isdir(method_root):
            shutil.rmtree(method_root)
        os.makedirs(method_root, exist_ok=True)
        method_gpu_gb = args.covariance_gpu_gb if method == "covariance" else args.ppca_gpu_gb
        method_low_memory = args.covariance_low_memory_option if method == "covariance" else args.ppca_low_memory_option
        method_very_low_memory = (
            args.covariance_very_low_memory_option if method == "covariance" else args.ppca_very_low_memory_option
        )
        cmd, result_dir, log_path = build_pipeline_command(
            method=method,
            sim_dir=sim_dir,
            method_root=method_root,
            grid_size=args.grid_size,
            halfsets_path=halfsets_path,
            zdim=args.zdim,
            ppca_em_iters=args.ppca_em_iters,
            use_contrast=use_contrast,
            gpu_gb=method_gpu_gb,
            low_memory_option=method_low_memory,
            very_low_memory_option=method_very_low_memory,
            lazy=args.lazy,
            force=args.force,
        )
        result_path = Path(result_dir)
        result_path.mkdir(parents=True, exist_ok=True)
        command_path = result_path / "command.txt"
        command_path.write_text(shlex.join(cmd) + "\n", encoding="utf-8")
        status_path = os.path.join(method_root, f"{method}.exitcode")
        method_specs.append(
            {
                "method": method,
                "cmd": cmd,
                "result_dir": result_dir,
                "log_path": log_path,
                "status_path": status_path,
                "command_path": str(command_path),
            }
        )

    runner_script = _write_method_runner_script(output_root, method_specs)
    manifest = {
        "dataset": dataset_info["name"],
        "run_root": output_root,
        "sim_dir": sim_dir,
        "halfsets_path": halfsets_path,
        "runner_script": runner_script,
        "method_specs": [
            {
                "method": spec["method"],
                "result_dir": spec["result_dir"],
                "log_path": spec["log_path"],
                "status_path": spec["status_path"],
                "command_path": spec["command_path"],
            }
            for spec in method_specs
        ],
    }
    manifest_path = os.path.join(output_root, "compare_run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    manifest["manifest_path"] = manifest_path
    return manifest


def _read_runtime_seconds(result_dir: str) -> float | None:
    job_path = os.path.join(result_dir, "job.json")
    if not os.path.isfile(job_path):
        return None
    with open(job_path, "r", encoding="utf-8") as fh:
        job = json.load(fh)
    return job.get("timing", {}).get("duration_seconds")


def _maybe_float(value):
    if value is None:
        return None
    return float(value)


def _build_metric_context(gt_results, sim_info: dict, volume_shape: tuple[int, int, int]) -> dict:
    gt_mean = np.asarray(gt_results.get_mean())
    gt_union_soft_mask, gt_union_binary_mask = metrics.make_union_gt_mask_from_hvd(gt_results, volume_shape)
    u_gt_real, s_gt_real, _ = gt_results.get_vol_svd(contrasted=False, real_space=True, random_svd_pcs=200)
    particle_assignment = sim_info["image_assignment"]
    preferred_labels = [0, (int(np.max(particle_assignment)) + 1) // 2]
    return {
        "gt_mean": gt_mean,
        "gt_union_soft_mask": gt_union_soft_mask,
        "gt_union_binary_mask": gt_union_binary_mask,
        "u_gt_real": u_gt_real,
        "s_gt_real": s_gt_real,
        "particle_assignment": particle_assignment,
        "preferred_labels": preferred_labels,
        "gt_contrasts": np.asarray(gt_results.contrasts),
        "sim_info": sim_info,
    }


def _compute_pipeline_like_metrics(
    result_dir: str,
    method_root: str,
    gt_results,
    metric_context: dict,
    zdim: int,
) -> tuple[dict, dict]:
    po = output_module.PipelineOutput(result_dir)
    plots_dir = os.path.join(method_root, "pipeline_metric_plots")
    os.makedirs(plots_dir, exist_ok=True)
    ds = po.get("lazy_dataset")
    volume_shape = ds.volume_shape
    all_scores: dict[str, object] = {}
    plot_paths: dict[str, str] = {}

    gt_mean = metric_context["gt_mean"]
    gt_union_binary_mask = metric_context["gt_union_binary_mask"]
    gt_contrasts = metric_context["gt_contrasts"]
    particle_assignment = metric_context["particle_assignment"]
    preferred_labels = metric_context["preferred_labels"]
    sim_info = metric_context["sim_info"]

    mean = np.asarray(po.get("mean"))
    mean_fsc_path = os.path.join(plots_dir, "fsc_mean.png")
    _, mean_fsc = plot_utils.plot_fsc_new(
        gt_mean,
        mean,
        np.array(volume_shape),
        ds.voxel_size,
        threshold=0.5,
        filename=mean_fsc_path,
        name="Mean FSC",
        fmat="",
    )
    all_scores["mean_fsc"] = float(mean_fsc)
    plot_paths["mean_fsc_plot"] = mean_fsc_path

    gt_spatial_variance = gt_results.get_spatial_variances(contrasted=False)
    estimated_spatial_variance = np.asarray(po.get("variance"))
    gt_sp_dft = fourier_transform_utils.get_dft3(gt_spatial_variance.reshape(volume_shape)).reshape(-1)
    est_sp_dft = fourier_transform_utils.get_dft3(estimated_spatial_variance.reshape(volume_shape)).reshape(-1)
    var_spatial_path = os.path.join(plots_dir, "fsc_variance_spatial.png")
    _, score_spatial = plot_utils.plot_fsc_new(
        gt_sp_dft,
        est_sp_dft,
        np.array(volume_shape),
        ds.voxel_size,
        threshold=0.5,
        filename=var_spatial_path,
        name="Variance Spatial FSC",
        fmat="",
    )
    all_scores["variance_spatial_fsc"] = float(score_spatial)
    all_scores["variance_fsc"] = float(score_spatial)
    plot_paths["variance_spatial_fsc_plot"] = var_spatial_path

    if hasattr(gt_results, "get_covariance_square_root"):
        cov_sqrt_fourier = gt_results.get_covariance_square_root(contrasted=False)
        gt_fourier_variance = np.sum(np.abs(cov_sqrt_fourier) ** 2, axis=-1)
        u_fourier_all = np.asarray(po.get("u"))
        s_all_var = np.asarray(po.get("s"))
        n_pcs_var = min(20, u_fourier_all.shape[0])
        est_fourier_variance = utils.estimate_variance(u_fourier_all[:n_pcs_var, :], s_all_var[:n_pcs_var])
        var_fourier_path = os.path.join(plots_dir, "fsc_variance_fourier.png")
        _, score_fourier = plot_utils.plot_fsc_new(
            gt_fourier_variance,
            est_fourier_variance,
            np.array(volume_shape),
            ds.voxel_size,
            threshold=0.5,
            filename=var_fourier_path,
            name="Variance Fourier FSC",
            fmat="",
        )
        all_scores["variance_fourier_fsc"] = float(score_fourier)
        plot_paths["variance_fourier_fsc_plot"] = var_fourier_path

    take_n_pcs = max(20, zdim)
    u_real_subset = test_metrics.load_u_real_for_metrics(po, take_n_pcs)
    take_n_pcs_eff = int(u_real_subset.shape[0])
    vol_norm = np.sqrt(np.prod(po.get("volume_shape")))
    u_est_real = np.array(u_real_subset.reshape(take_n_pcs_eff, -1)).T * vol_norm
    s_all = np.asarray(po.get("s"))
    s_est = s_all[:take_n_pcs_eff] / (vol_norm**2)
    variance, rel_var, norm_var = metrics.get_all_variance_scores(
        u_est_real, metric_context["u_gt_real"], metric_context["s_gt_real"]
    )
    all_scores["svd_relative_variance_4"] = _maybe_float(rel_var[4]) if rel_var.size > 4 else None
    all_scores["svd_relative_variance_10"] = _maybe_float(rel_var[10]) if rel_var.size > 10 else None
    all_scores["svd_relative_variance_curve"] = np.asarray(rel_var, dtype=np.float32).tolist()
    all_scores["svd_normalized_variance_curve"] = np.asarray(norm_var, dtype=np.float32).tolist()
    all_scores["svd_variance_score_curve"] = np.asarray(variance, dtype=np.float32).tolist()
    all_scores["eigenvalues"] = np.asarray(s_est, dtype=np.float32).tolist()

    embedding_cache = {}
    if po.has_embedding_key("latent_coords", zdim):
        unsorted_zs = test_metrics.load_unsorted_embedding_component(po, "latent_coords", zdim, cache=embedding_cache)
        _, averaged_variance = metrics.variance_of_zs(unsorted_zs, particle_assignment)
        all_scores[f"embedding_squared_error_{zdim}"] = float(averaged_variance)

        zs_assignment, labels_to_plot = test_metrics.select_state_target_latent_points(
            unsorted_zs=unsorted_zs,
            particle_assignment=particle_assignment,
            preferred_labels=preferred_labels,
            max_points=2,
        )
        latent_points_path = os.path.join(method_root, f"latent_points_z{zdim}.txt")
        np.savetxt(latent_points_path, zs_assignment)
        state_dir = os.path.join(method_root, f"state_z{zdim}")
        cs_parser = compute_state.add_args(argparse.ArgumentParser())
        cs_args = cs_parser.parse_args(
            [
                result_dir,
                "-o",
                state_dir,
                "--latent-points",
                latent_points_path,
                "--save-all-estimates",
            ]
        )
        compute_state.compute_state(cs_args)
        all_scores["state_dir"] = state_dir
        for l_idx, label in enumerate(labels_to_plot):
            gt_map = fourier_transform_utils.get_idft3(gt_results.volumes[label].reshape(volume_shape)).real
            estimate_map = utils.load_mrc(Path(state_dir, f"state{l_idx:03d}.mrc"))
            errors_metrics = metrics.compute_volume_error_metrics_from_gt(
                gt_map,
                estimate_map,
                ds.voxel_size,
                gt_union_binary_mask,
                partial_mask=None,
                normalize_by_map1=True,
            )
            all_scores[f"state_{l_idx}_label"] = int(label)
            all_scores[f"state_{l_idx}_locres_90pct"] = _maybe_float(errors_metrics.get("ninety_pc_locres"))
            all_scores[f"state_{l_idx}_locres_median"] = _maybe_float(errors_metrics.get("median_locres"))

    for entry, suffix in (("contrasts", str(zdim)), ("contrasts_noreg", f"{zdim}_noreg")):
        if po.has_embedding_key(entry, zdim):
            unsorted_contrast = test_metrics.load_unsorted_embedding_component(po, entry, zdim, cache=embedding_cache)
            contrast_abs_error = np.mean(np.abs(gt_contrasts - unsorted_contrast))
            all_scores[f"contrast_abs_error_{suffix}"] = float(contrast_abs_error)

    all_scores.update(
        test_metrics.compute_noise_variance_metrics(
            sim_info.get("noise_variance"),
            po.get("noise_var_used"),
            plots_dir,
            logger,
            dose_indices=sim_info.get("dose_indices"),
            noise_increase_per_tilt=sim_info.get("noise_increase_per_tilt"),
        )
    )
    return test_metrics.normalize_scores_for_json(all_scores), plot_paths


def score_pipeline_output(result_dir: str, method_root: str, gt_results, metric_context: dict, zdim: int) -> dict:
    po = output_module.PipelineOutput(result_dir)
    gt_mean = metric_context["gt_mean"]
    u_gt, s_gt, _ = gt_results.get_vol_svd(contrasted=False)
    u_est = np.asarray(po.get_u(zdim)).T[:, :zdim]
    s_est = np.asarray(po.get("s"), dtype=np.float32)[:zdim]
    mean_est = np.asarray(po.get("mean"))
    variance, rel_var, norm_var = metrics.get_all_variance_scores(u_est, u_gt[:, :zdim], s_gt[:zdim] ** 2)
    mean_error = float(np.linalg.norm(mean_est - gt_mean) / np.linalg.norm(gt_mean))
    pipeline_metrics, pipeline_metric_plots = _compute_pipeline_like_metrics(
        result_dir=result_dir,
        method_root=method_root,
        gt_results=gt_results,
        metric_context=metric_context,
        zdim=zdim,
    )
    return {
        "heterogeneity_method": po.get("heterogeneity_method"),
        "ppca_info": po.get("ppca_info") if "ppca_info" in po.keys() else None,
        "rel_var_mean": float(np.mean(rel_var)),
        "rel_var_per_pc": np.asarray(rel_var, dtype=np.float32).tolist(),
        "variance_score_per_pc": np.asarray(variance, dtype=np.float32).tolist(),
        "norm_var_per_pc": np.asarray(norm_var, dtype=np.float32).tolist(),
        "s_est": s_est.tolist(),
        "s_gt": np.asarray(s_gt[:zdim] ** 2, dtype=np.float32).tolist(),
        "mean_error": mean_error,
        "pipeline_metrics": pipeline_metrics,
        "pipeline_metric_plots": pipeline_metric_plots,
        "result_dir": result_dir,
    }


def _copy_if_exists(src: str, dst: str) -> None:
    if os.path.isfile(src):
        Image.open(src).save(dst)


def _merge_images_horizontally(image_paths: list[str], out_path: str) -> str | None:
    existing = [p for p in image_paths if os.path.isfile(p)]
    if not existing:
        return None
    images = [Image.open(p).convert("RGB") for p in existing]
    target_h = max(im.size[1] for im in images)
    resized = []
    for im in images:
        if im.size[1] == target_h:
            resized.append(im)
            continue
        new_w = int(round(im.size[0] * target_h / im.size[1]))
        resized.append(im.resize((new_w, target_h), Image.Resampling.LANCZOS))
    canvas = Image.new("RGB", (sum(im.size[0] for im in resized), target_h), color=(255, 255, 255))
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.size[0]
    canvas.save(out_path)
    return out_path


def _merge_images_vertically(image_paths: list[str], out_path: str) -> str | None:
    existing = [p for p in image_paths if p and os.path.isfile(p)]
    if not existing:
        return None
    images = [Image.open(p).convert("RGB") for p in existing]
    target_w = max(im.size[0] for im in images)
    resized = []
    for im in images:
        if im.size[0] == target_w:
            resized.append(im)
            continue
        new_h = int(round(im.size[1] * target_w / im.size[0]))
        resized.append(im.resize((target_w, new_h), Image.Resampling.LANCZOS))
    canvas = Image.new("RGB", (target_w, sum(im.size[1] for im in resized)), color=(255, 255, 255))
    y = 0
    for im in resized:
        canvas.paste(im, (0, y))
        y += im.size[1]
    canvas.save(out_path)
    return out_path


def write_comparison_plots(output_root: str, scores: dict, zdim: int) -> dict:
    plot_dir = os.path.join(output_root, "comparison_plots")
    os.makedirs(plot_dir, exist_ok=True)

    methods = [method for method in METHOD_ORDER if method in scores]
    labels = [METHOD_LABELS[method] for method in methods]
    colors = [METHOD_COLORS[method] for method in methods]
    x = np.arange(1, zdim + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    width = 0.8 / max(len(methods), 1)
    center_offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    for offset, method, label, color in zip(center_offsets, methods, labels, colors):
        axes[0].bar(
            x + offset,
            scores[method]["rel_var_per_pc"],
            width=width,
            label=label,
            color=color,
            alpha=0.85,
        )
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("RelVar")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Per-PC Relative Variance")
    axes[0].legend(loc="best")

    axes[1].plot(x, scores[methods[0]]["s_gt"], "k--", linewidth=2, label="GT")
    markers = ["o", "s", "^", "d", "v"]
    for marker, method, label, color in zip(markers, methods, labels, colors):
        axes[1].plot(x, scores[method]["s_est"], marker=marker, color=color, label=label)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("PC")
    axes[1].set_ylabel("Eigenvalue")
    axes[1].set_title("Eigenvalues")
    axes[1].legend(loc="best")

    relvar_vals = [scores[m]["rel_var_mean"] for m in methods]
    mean_err_vals = [scores[m]["mean_error"] for m in methods]
    axes[2].bar(np.arange(len(methods)), relvar_vals, color=colors, alpha=0.85)
    axes[2].set_xticks(np.arange(len(methods)), labels)
    axes[2].set_ylabel("Mean RelVar")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("Overall RelVar")
    ax2 = axes[2].twinx()
    ax2.plot(np.arange(len(methods)), mean_err_vals, "ko--", linewidth=1.5)
    ax2.set_ylabel("Mean Relative Error")
    ax2.set_ylim(0.0, max(mean_err_vals + [1e-3]) * 1.3)

    plt.tight_layout()
    metrics_plot = os.path.join(plot_dir, "metrics_comparison.png")
    plt.savefig(metrics_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_images = [
        os.path.join(scores[method]["result_dir"], "output", "plots", "pipeline_summary.png") for method in methods
    ]
    summary_row = _merge_images_horizontally(summary_images, os.path.join(plot_dir, "pipeline_summaries.png"))

    pc_images = [
        os.path.join(scores[method]["result_dir"], "output", "plots", "principal_component_space_analysis.png")
        for method in methods
    ]
    pc_row = _merge_images_horizontally(pc_images, os.path.join(plot_dir, "pc_space_comparison.png"))

    combined = _merge_images_vertically(
        [summary_row, pc_row, metrics_plot],
        os.path.join(plot_dir, "comparison_combined.png"),
    )
    return {
        "metrics_plot": metrics_plot,
        "summary_row": summary_row,
        "pc_row": pc_row,
        "combined_plot": combined,
    }


def _score_existing_run(args) -> dict:
    prep = prepare_compare_run(args)
    dataset_info = _find_cryobench_dataset(args.base_dir, args.dataset)
    output_root = prep["run_root"]
    sim_dir = prep["sim_dir"]
    sim_info_path = os.path.join(sim_dir, "simulation_info.pkl")
    sim_info = utils.pickle_load(sim_info_path)
    gt_results = synthetic_dataset.load_heterogeneous_reconstruction(os.path.join(sim_dir, "simulation_info.pkl"))
    metric_context = _build_metric_context(gt_results, sim_info, (args.grid_size, args.grid_size, args.grid_size))
    scores = {}
    logs = {}
    runtimes = {}
    status_paths = {}
    missing_methods = []
    failed_methods = {}
    for spec in prep["method_specs"]:
        method = spec["method"]
        method_root = os.path.join(output_root, method)
        result_dir, log_path, runtime_seconds = run_pipeline_method(
            method=method,
            sim_dir=sim_dir,
            method_root=method_root,
            grid_size=args.grid_size,
            halfsets_path=prep["halfsets_path"],
            zdim=args.zdim,
            ppca_em_iters=args.ppca_em_iters,
            use_contrast=args.contrast_std > 0,
            gpu_gb=args.covariance_gpu_gb if method == "covariance" else args.ppca_gpu_gb,
            low_memory_option=args.covariance_low_memory_option if method == "covariance" else args.ppca_low_memory_option,
            very_low_memory_option=(
                args.covariance_very_low_memory_option if method == "covariance" else args.ppca_very_low_memory_option
            ),
            lazy=args.lazy,
            force=args.force,
        )
        logs[method] = log_path
        status_paths[method] = spec["status_path"]
        if runtime_seconds is not None:
            runtimes[method] = runtime_seconds
        else:
            runtimes[method] = _read_runtime_seconds(result_dir)
        params_path = os.path.join(result_dir, "model", "params.pkl")
        if os.path.isfile(params_path):
            scores[method] = score_pipeline_output(result_dir, method_root, gt_results, metric_context, args.zdim)
            continue
        missing_methods.append(method)
        if os.path.isfile(spec["status_path"]):
            with open(spec["status_path"], "r", encoding="utf-8") as fh:
                failed_methods[method] = fh.read().strip()

    plot_paths = write_comparison_plots(output_root, scores, args.zdim) if scores else {}
    summary = {
        "dataset": dataset_info["name"],
        "n_volumes": dataset_info["n_volumes"],
        "grid_size": args.grid_size,
        "n_images": args.n_images,
        "noise_level": args.noise_level,
        "contrast_std": args.contrast_std,
        "zdim": args.zdim,
        "ppca_em_iters": args.ppca_em_iters,
        "seed": args.seed,
        "use_contrast_correction": args.contrast_std > 0,
        "covariance_gpu_gb": args.covariance_gpu_gb,
        "ppca_gpu_gb": args.ppca_gpu_gb,
        "covariance_low_memory_option": args.covariance_low_memory_option,
        "covariance_very_low_memory_option": args.covariance_very_low_memory_option,
        "ppca_low_memory_option": args.ppca_low_memory_option,
        "ppca_very_low_memory_option": args.ppca_very_low_memory_option,
        "scores": scores,
        "logs": logs,
        "status_paths": status_paths,
        "missing_methods": missing_methods,
        "failed_methods": failed_methods,
        "runtimes_seconds": runtimes,
        "plots": plot_paths,
        "sim_dir": sim_dir,
        "halfsets_path": prep["halfsets_path"],
        "run_root": output_root,
    }
    summary_path = os.path.join(output_root, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def compare_methods(args) -> dict:
    prep = prepare_compare_run(args)
    for spec in prep["method_specs"]:
        method = spec["method"]
        method_root = os.path.join(prep["run_root"], method)
        run_pipeline_method(
            method=method,
            sim_dir=prep["sim_dir"],
            method_root=method_root,
            grid_size=args.grid_size,
            halfsets_path=prep["halfsets_path"],
            zdim=args.zdim,
            ppca_em_iters=args.ppca_em_iters,
            use_contrast=args.contrast_std > 0,
            gpu_gb=args.covariance_gpu_gb if method == "covariance" else args.ppca_gpu_gb,
            low_memory_option=args.covariance_low_memory_option if method == "covariance" else args.ppca_low_memory_option,
            very_low_memory_option=(
                args.covariance_very_low_memory_option if method == "covariance" else args.ppca_very_low_memory_option
            ),
            lazy=args.lazy,
            force=args.force,
        )
    return _score_existing_run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare covariance-PCA pipeline and direct PPCA pipeline on one synthetic CryoBench dataset.",
    )
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--n-images", type=int, default=100000)
    parser.add_argument("--noise-level", type=float, default=1.0)
    parser.add_argument("--contrast-std", type=float, default=0.0)
    parser.add_argument("--zdim", type=int, default=10)
    parser.add_argument("--ppca-em-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-gb", type=float, default=None)
    parser.add_argument("--covariance-gpu-gb", type=float, default=None)
    parser.add_argument("--ppca-gpu-gb", type=float, default=None)
    parser.add_argument("--covariance-low-memory-option", action="store_true")
    parser.add_argument("--covariance-very-low-memory-option", action="store_true")
    parser.add_argument("--ppca-low-memory-option", action="store_true")
    parser.add_argument("--ppca-very-low-memory-option", action="store_true")
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage", choices=("full", "prepare", "score"), default="full")
    parser.add_argument("--print-run-root", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.covariance_gpu_gb is None:
        args.covariance_gpu_gb = args.gpu_gb
    if args.ppca_gpu_gb is None:
        args.ppca_gpu_gb = args.gpu_gb
    if args.stage == "prepare":
        prep = prepare_compare_run(args)
        if args.print_run_root:
            print(prep["run_root"])
        else:
            print(json.dumps(prep, indent=2))
        return
    if args.stage == "score":
        _score_existing_run(args)
        return
    compare_methods(args)


if __name__ == "__main__":
    main()
