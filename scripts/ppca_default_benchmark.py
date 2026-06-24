#!/usr/bin/env python
"""Manifest-driven PPCA-vs-default pipeline benchmark.

This is intentionally an orchestration script, not production library code. It
builds matched covariance/PPCA runs for synthetic CryoBench PDB datasets and
RECOVAR-paper real datasets, then writes per-case summaries that can be
aggregated after the Slurm array finishes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils as ftu
from recovar.output import metrics
from recovar.ppca import compare_covariance_vs_ppca_pipeline as compare_mod
from recovar.project.project import RecovarProject
from recovar.simulation import simulate_scattering_potential as ssp
from recovar.simulation import simulator, synthetic_dataset
from recovar.simulation.trajectory_generation import compute_bfactor_scaling

METHODS = ("covariance", "ppca")
DEFAULT_CRYOBENCH_ROOT = Path("/home/mg6942/mytigress/cryobench2")
DEFAULT_RECOVAR_DATASETS_ROOT = Path("/tigress/CRYOEM/singerlab/mg6942/RECOVAR_datasets")

SYNTHETIC_DATASETS = ("Ribosembly", "IgG-1D", "IgG-RL", "Tomotwin-100")

SYNTHETIC_TEMPLATES = (
    # case_suffix, n_images, contrast_std, correct_contrast, solvent, focus, complement
    ("nomask_c0_cc0_n100k", 100_000, 0.0, False, False, False, False),
    ("solvent_c0_cc0_n100k", 100_000, 0.0, False, True, False, False),
    ("focus_c0_cc0_n200k", 200_000, 0.0, False, False, True, False),
    ("solvent_focus_c0_cc0_n200k", 200_000, 0.0, False, True, True, False),
    ("solvent_c03_cc0_n200k", 200_000, 0.3, False, True, False, False),
    ("solvent_focus_complement_c03_cc1_n300k", 300_000, 0.3, True, True, True, True),
)

REAL_TEMPLATES = (
    # case_suffix, correct_contrast, solvent, focus, complement, compute_state_from_centers
    ("nomask_cc0", False, False, False, False, False),
    ("solvent_cc0", False, True, False, False, False),
    ("focus_cc1", True, False, True, False, False),
    ("solvent_focus_cc1", True, True, True, False, True),
    ("solvent_focus_complement_cc1", True, True, True, True, True),
)

REAL_DATASETS = {
    "10073": {
        "particles": "/tigress/CRYOEM/singerlab/mg6942/10073/recovar_data/particles.256.mrcs",
        "poses": "10073/poses.pkl",
        "ctf": "10073/ctf.pkl",
        "mask": "10073/mask.mrc",
        "focus_mask": "10073/focus_mask.mrc",
        "ind": "10073/ind.pkl",
    },
    "10076": {
        "particles": "/tigress/CRYOEM/singerlab/mg6942/10076/particles.256.mrcs",
        "poses": "10076/poses.pkl",
        "ctf": "10076/ctf.pkl",
        "mask": "10076/mask.mrc",
        "focus_mask": "/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar/empiar10076/inputs/recovar_masks/mask_10076.mrc",
        "ind": "",
    },
    "10180": {
        "particles": "/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar/empiar10180/inputs/particles.256.mrcs",
        "poses": "10180/poses.pkl",
        "ctf": "10180/ctf.pkl",
        "mask": "10180/mask.mrc",
        "focus_mask": "10180/focus_mask.mrc",
        "ind": "10180/filtered.ind.pkl",
    },
    "10345": {
        "particles": "/tigress/CRYOEM/singerlab/mg6942/10345/recovar_data/particles.256.mrcs",
        "poses": "10345/poses.pkl",
        "ctf": "10345/ctf.pkl",
        "mask": "10345/mask.mrc",
        "focus_mask": "10345/focus_mask.mrc",
        "ind": "10345/ind.pkl",
    },
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _repo_head(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _rel_or_abs(root: Path, value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def _case_root(results_root: Path, case: dict) -> Path:
    return results_root / "cases" / case["case_id"]


def _mask_mode(solvent: bool, focus: bool, complement: bool) -> str:
    if solvent and focus and complement:
        return "solvent_focus_complement"
    if solvent and focus:
        return "solvent_focus"
    if focus:
        return "focus"
    if solvent:
        return "solvent"
    return "none"


def _pipeline_done(result_dir: Path) -> bool:
    return (result_dir / "model" / "params.pkl").is_file()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def build_manifest(args: argparse.Namespace) -> dict:
    repo_root = args.repo_root.resolve()
    results_root = args.results_root.resolve()
    cryobench_root = args.cryobench_root.resolve()
    recovar_datasets_root = args.recovar_datasets_root.resolve()

    cases: list[dict] = []
    seed_base = int(args.seed)
    for d_idx, dataset in enumerate(SYNTHETIC_DATASETS):
        pdb_dir = cryobench_root / dataset / "pdbs"
        for t_idx, (suffix, n_images, contrast_std, correct, solvent, focus, complement) in enumerate(
            SYNTHETIC_TEMPLATES
        ):
            seed = seed_base + d_idx * 100 + t_idx
            cases.append(
                {
                    "kind": "synthetic",
                    "case_id": f"syn_{dataset}_{suffix}".replace("/", "_"),
                    "dataset": dataset,
                    "pdb_dir": str(pdb_dir),
                    "grid_size": args.grid_size,
                    "n_images": n_images,
                    "noise_level": args.noise_level,
                    "contrast_std": contrast_std,
                    "correct_contrast": correct,
                    "use_solvent_mask": solvent,
                    "use_focus_mask": focus,
                    "use_complement_mask": complement,
                    "mask_mode": _mask_mode(solvent, focus, complement),
                    "zdim": args.zdim,
                    "ppca_em_iters": args.ppca_em_iters,
                    "seed": seed,
                    "pdb_bfactor": args.pdb_bfactor,
                }
            )

    for dataset_id, spec in REAL_DATASETS.items():
        resolved = {
            key: _rel_or_abs(recovar_datasets_root, value)
            for key, value in spec.items()
        }
        for suffix, correct, solvent, focus, complement, compute_state_centers in REAL_TEMPLATES:
            cases.append(
                {
                    "kind": "real",
                    "case_id": f"real_{dataset_id}_{suffix}",
                    "dataset": dataset_id,
                    "grid_size": args.grid_size,
                    "correct_contrast": correct,
                    "use_solvent_mask": solvent,
                    "use_focus_mask": focus,
                    "use_complement_mask": complement,
                    "mask_mode": _mask_mode(solvent, focus, complement),
                    "zdim": args.zdim,
                    "ppca_em_iters": args.ppca_em_iters,
                    "run_analyze": True,
                    "compute_state_from_centers": compute_state_centers,
                    **resolved,
                }
            )

    case_kind = getattr(args, "case_kind", "all")
    if case_kind != "all":
        cases = [case for case in cases if case["kind"] == case_kind]

    manifest = {
        "created_at": _now(),
        "repo_root": str(repo_root),
        "repo_head": _repo_head(repo_root),
        "results_root": str(results_root),
        "cryobench_root": str(cryobench_root),
        "recovar_datasets_root": str(recovar_datasets_root),
        "methods": list(METHODS),
        "n_cases": len(cases),
        "n_method_runs": len(cases) * len(METHODS),
        "cases": cases,
    }
    return manifest


def write_manifest(args: argparse.Namespace) -> None:
    manifest = build_manifest(args)
    ensure_benchmark_project(Path(manifest["results_root"]))
    out = args.output.resolve()
    _write_json(out, manifest)
    print(json.dumps({"manifest": str(out), "n_cases": manifest["n_cases"], "n_method_runs": manifest["n_method_runs"]}, indent=2))


def _pdb_files(pdb_dir: Path) -> list[Path]:
    files = sorted(path for path in pdb_dir.glob("*.pdb") if not path.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"No PDB files found in {pdb_dir}")
    return files


def _acquire_lock(lock_dir: Path, *, timeout_s: int = 12 * 3600) -> None:
    start = time.time()
    while True:
        try:
            lock_dir.mkdir(parents=True)
            return
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for lock {lock_dir}")
            time.sleep(30)


def _release_lock(lock_dir: Path) -> None:
    try:
        lock_dir.rmdir()
    except FileNotFoundError:
        pass


def ensure_benchmark_project(results_root: Path) -> Path:
    project_root = results_root / "project_cache"
    if (project_root / "project.json").exists():
        return project_root
    lock_dir = project_root.with_name(project_root.name + ".init.lock")
    _acquire_lock(lock_dir, timeout_s=30 * 60)
    try:
        RecovarProject.init(str(project_root), name=f"ppca_default_benchmark_{results_root.name}")
    finally:
        _release_lock(lock_dir)
    return project_root


def ensure_pdb_volumes(case: dict, results_root: Path) -> Path:
    pdb_dir = Path(case["pdb_dir"])
    pdbs = _pdb_files(pdb_dir)
    grid_size = int(case["grid_size"])
    bfactor = float(case["pdb_bfactor"])
    shared_root = results_root / "shared" / "cryobench_pdb_volumes"
    volume_dir = shared_root / f"{case['dataset']}_g{grid_size}_bf{bfactor:g}"
    prefix = volume_dir / "vol"
    expected = [Path(f"{prefix}{idx:04d}.mrc") for idx in range(len(pdbs))]
    if all(path.exists() for path in expected):
        return prefix

    lock_dir = volume_dir.with_name(volume_dir.name + ".lock")
    _acquire_lock(lock_dir)
    try:
        if all(path.exists() for path in expected):
            return prefix
        volume_dir.mkdir(parents=True, exist_ok=True)
        voxel_size = 4.25 * 128 / grid_size
        scaling = compute_bfactor_scaling((grid_size, grid_size, grid_size), voxel_size, bfactor)
        manifest = []
        for idx, pdb_path in enumerate(pdbs):
            out_path = expected[idx]
            if out_path.exists():
                continue
            print(f"[pdb-volumes] {case['dataset']} {idx + 1}/{len(pdbs)} {pdb_path}", flush=True)
            ft_mol = ssp.generate_molecule_spectrum_from_pdb_id(
                str(pdb_path),
                voxel_size=voxel_size,
                grid_size=grid_size,
                force_symmetry=True,
            )
            ft_mol = ft_mol.reshape((grid_size, grid_size, grid_size)) * scaling
            vol = np.real(ftu.get_idft3(jnp.asarray(ft_mol))).astype(np.float32)
            utils.write_mrc(str(out_path), vol, voxel_size=voxel_size)
            manifest.append({"class_index": idx, "pdb_path": str(pdb_path), "volume_path": str(out_path)})
        _write_json(volume_dir / "pdb_volume_manifest.json", {"dataset": case["dataset"], "volumes": manifest})
    finally:
        _release_lock(lock_dir)
    return prefix


def ensure_synthetic_dataset(case: dict, run_root: Path, results_root: Path) -> dict:
    dataset_dir = run_root / "simulated_data"
    sim_info_path = dataset_dir / "simulation_info.pkl"
    grid_size = int(case["grid_size"])
    if not sim_info_path.exists():
        volume_prefix = ensure_pdb_volumes(case, results_root)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(int(case["seed"]))
        simulator.generate_synthetic_dataset(
            str(dataset_dir),
            4.25 * 128 / grid_size,
            str(volume_prefix),
            int(case["n_images"]),
            outlier_file_input=None,
            grid_size=grid_size,
            volume_distribution=None,
            dataset_params_option="uniform",
            noise_level=float(case["noise_level"]),
            noise_model="white",
            put_extra_particles=False,
            percent_outliers=0.0,
            volume_radius=0.7,
            trailing_zero_format_in_vol_name=True,
            noise_scale_std=0.0,
            contrast_std=float(case["contrast_std"]),
            disc_type="nufft",
            image_dtype=np.float32,
        )

    halfsets_path = compare_mod.ensure_halfsets(str(run_root), int(case["n_images"]), int(case["seed"]))
    gt = synthetic_dataset.load_heterogeneous_reconstruction(str(sim_info_path))
    volume_shape = (grid_size, grid_size, grid_size)
    masks_dir = dataset_dir / "gt_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    union_mask_path = masks_dir / "gt_union_mask.mrc"
    moving_mask_path = masks_dir / "gt_moving_mask.mrc"
    if not union_mask_path.exists():
        union_soft, _ = metrics.make_union_gt_mask_from_hvd(gt, volume_shape)
        utils.write_mrc(str(union_mask_path), np.asarray(union_soft, dtype=np.float32), voxel_size=4.25 * 128 / grid_size)
    if not moving_mask_path.exists():
        moving_soft, _ = metrics.make_moving_gt_mask_from_hvd(gt, volume_shape)
        utils.write_mrc(str(moving_mask_path), np.asarray(moving_soft, dtype=np.float32), voxel_size=4.25 * 128 / grid_size)
    return {
        "particles": str(dataset_dir / f"particles.{grid_size}.mrcs"),
        "poses": str(dataset_dir / "poses.pkl"),
        "ctf": str(dataset_dir / "ctf.pkl"),
        "halfsets": halfsets_path,
        "mask": str(union_mask_path),
        "focus_mask": str(moving_mask_path),
        "sim_info": str(sim_info_path),
    }


def _method_dir(run_root: Path, method: str) -> Path:
    return run_root / method


def _result_dir(run_root: Path, method: str) -> Path:
    return _method_dir(run_root, method) / "result"


def build_pipeline_cmd(case: dict, inputs: dict, run_root: Path, method: str, gpu_gb: float | None) -> list[str]:
    zdim = int(case["zdim"])
    method_root = _method_dir(run_root, method)
    result_dir = _result_dir(run_root, method)
    mask_arg = inputs["mask"] if case["use_solvent_mask"] else "none"
    cmd = [
        sys.executable,
        "-m",
        "recovar.commands.pipeline",
        inputs["particles"],
        "-o",
        str(result_dir),
        "--poses",
        inputs["poses"],
        "--ctf",
        inputs["ctf"],
        "--mask",
        mask_arg,
        "--zdim",
        str(zdim),
        "--downsample",
        str(int(case["grid_size"])),
        "--lazy",
    ]
    if case.get("kind") == "real":
        cmd.extend(["--project", str(ensure_benchmark_project(run_root.parent.parent))])
    if inputs.get("halfsets"):
        cmd.extend(["--halfsets", inputs["halfsets"]])
    if inputs.get("ind"):
        cmd.extend(["--ind", inputs["ind"]])
    if gpu_gb is not None:
        cmd.extend(["--gpu-budget-gb", str(gpu_gb)])
    if case["correct_contrast"]:
        cmd.append("--correct-contrast")
    if case["use_solvent_mask"] or case["use_focus_mask"]:
        cmd.append("--keep-input-mask")
    if case["use_focus_mask"]:
        cmd.extend(["--focus-mask", inputs["focus_mask"]])
    if case["use_complement_mask"]:
        cmd.append("--use-complement-mask")
    if method == "ppca":
        cmd.extend(["--use-ppca", "--ppca-zdim", str(zdim), "--ppca-em-iters", str(case["ppca_em_iters"])])
        if case["use_complement_mask"]:
            cmd.extend(["--ppca-pcs-per-mask", f"{zdim},{zdim}"])
    elif method != "covariance":
        raise ValueError(f"Unknown method {method}")
    method_root.mkdir(parents=True, exist_ok=True)
    (method_root / "pipeline_command.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    return cmd


def run_logged(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"started_at={_now()}\n")
        fh.write(f"command={shlex.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
        fh.write(f"\nfinished_at={_now()}\n")
        fh.write(f"exit_code={proc.returncode}\n")
        fh.write(f"wall_seconds={time.time() - started:.3f}\n")
    return int(proc.returncode)


def run_pipeline_methods(case: dict, inputs: dict, run_root: Path, gpu_gb: float | None) -> dict:
    methods = {}
    for method in METHODS:
        result_dir = _result_dir(run_root, method)
        method_root = _method_dir(run_root, method)
        log_path = method_root / "pipeline.log"
        if _pipeline_done(result_dir):
            exit_code = 0
            skipped = True
        else:
            cmd = build_pipeline_cmd(case, inputs, run_root, method, gpu_gb)
            exit_code = run_logged(cmd, log_path)
            skipped = False
        methods[method] = {
            "result_dir": str(result_dir),
            "log_path": str(log_path),
            "exit_code": exit_code,
            "skipped_existing": skipped,
        }
    return methods


def score_synthetic_case(case: dict, inputs: dict, run_root: Path, method_status: dict) -> dict:
    failed = {method: status for method, status in method_status.items() if status.get("exit_code") != 0}
    if failed:
        return {"scores": {}, "plots": {}, "score_skipped": f"pipeline_failed:{sorted(failed)}"}
    grid_size = int(case["grid_size"])
    sim_info = utils.pickle_load(inputs["sim_info"])
    gt = synthetic_dataset.load_heterogeneous_reconstruction(inputs["sim_info"])
    metric_context = build_metric_context(gt, sim_info, (grid_size, grid_size, grid_size), int(case["zdim"]))
    scores = {}
    for method in METHODS:
        result_dir = Path(method_status[method]["result_dir"])
        if not _pipeline_done(result_dir):
            continue
        scores[method] = compare_mod.score_pipeline_output(
            str(result_dir),
            str(_method_dir(run_root, method)),
            gt,
            metric_context,
            int(case["zdim"]),
        )
    plots = {}
    plot_error = None
    if scores:
        try:
            plots = compare_mod.write_comparison_plots(str(run_root), scores, int(case["zdim"]))
        except Exception as exc:  # Plotting should not invalidate benchmark metrics.
            plot_error = f"{type(exc).__name__}: {exc}"
    result = {"scores": scores, "plots": plots}
    if plot_error is not None:
        result["plot_error"] = plot_error
    return result


def build_metric_context(gt_results, sim_info: dict, volume_shape: tuple[int, int, int], zdim: int) -> dict:
    """Build the comparison metric context without the legacy 200-PC GT SVD."""
    gt_mean = np.asarray(gt_results.get_mean())
    gt_union_soft_mask, gt_union_binary_mask = metrics.make_union_gt_mask_from_hvd(gt_results, volume_shape)
    random_svd_pcs = max(20, int(zdim)) + 1
    u_gt_real, s_gt_real, _ = gt_results.get_vol_svd(
        contrasted=False,
        real_space=True,
        random_svd_pcs=random_svd_pcs,
    )
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


def read_runtime(result_dir: Path) -> float | None:
    job_path = result_dir / "job.json"
    if not job_path.exists():
        return None
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return job.get("timing", {}).get("duration_seconds")


def run_real_analysis(case: dict, run_root: Path, method_status: dict) -> dict:
    artifacts = {}
    for method in METHODS:
        if method_status.get(method, {}).get("exit_code") != 0:
            artifacts[method] = {"skipped": "pipeline_failed"}
            continue
        result_dir = Path(method_status[method]["result_dir"])
        method_root = _method_dir(run_root, method)
        if not _pipeline_done(result_dir):
            continue
        analyze_log = method_root / "analyze.log"
        analysis_dir = result_dir / f"analysis_{case['zdim']}"
        if not (analysis_dir / "kmeans" / "centers.txt").exists():
            cmd = [
                sys.executable,
                "-m",
                "recovar.commands.analyze",
                str(result_dir),
                "--zdim",
                str(case["zdim"]),
                "--n-clusters",
                "20",
                "--n-trajectories",
                "2",
                "--n-vols-along-path",
                "6",
            ]
            run_logged(cmd, analyze_log)
        centers = analysis_dir / "kmeans" / "centers.txt"
        compute_state_dir = method_root / "compute_state_from_centers"
        compute_state_log = method_root / "compute_state_from_centers.log"
        if case.get("compute_state_from_centers") and centers.exists() and not (compute_state_dir / "state000.mrc").exists():
            cmd = [
                sys.executable,
                "-m",
                "recovar.commands.compute_state",
                str(result_dir),
                "-o",
                str(compute_state_dir),
                "--latent-points",
                str(centers),
                "--save-all-estimates",
            ]
            run_logged(cmd, compute_state_log)
        artifacts[method] = {
            "analysis_dir": str(analysis_dir),
            "centers": str(centers) if centers.exists() else "",
            "analyze_log": str(analyze_log),
            "compute_state_dir": str(compute_state_dir) if compute_state_dir.exists() else "",
            "compute_state_log": str(compute_state_log) if compute_state_log.exists() else "",
        }
    return artifacts


def real_inputs(case: dict) -> dict:
    return {
        "particles": case["particles"],
        "poses": case["poses"],
        "ctf": case["ctf"],
        "mask": case["mask"],
        "focus_mask": case["focus_mask"],
        "ind": case.get("ind", ""),
        "halfsets": "",
    }


def validate_case_inputs(case: dict, inputs: dict) -> None:
    required = ["particles", "poses", "ctf"]
    if case["use_solvent_mask"]:
        required.append("mask")
    if case["use_focus_mask"]:
        required.append("focus_mask")
    if inputs.get("halfsets"):
        required.append("halfsets")
    if inputs.get("ind"):
        required.append("ind")
    missing = [key for key in required if not inputs.get(key) or not Path(inputs[key]).exists()]
    if missing:
        detail = {key: inputs.get(key, "") for key in missing}
        raise FileNotFoundError(f"Missing inputs for {case['case_id']}: {detail}")


def run_case(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    case = manifest["cases"][args.case_index]
    results_root = Path(manifest["results_root"])
    run_root = _case_root(results_root, case)
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "case.json", case)

    if case["kind"] == "synthetic":
        inputs = ensure_synthetic_dataset(case, run_root, results_root)
    elif case["kind"] == "real":
        inputs = real_inputs(case)
    else:
        raise ValueError(f"Unknown case kind {case['kind']}")
    validate_case_inputs(case, inputs)
    _write_json(run_root / "inputs.json", inputs)

    method_status = run_pipeline_methods(case, inputs, run_root, args.gpu_gb)
    summary = {
        "case": case,
        "inputs": inputs,
        "methods": method_status,
        "started_or_resumed_at": _now(),
        "run_root": str(run_root),
    }
    if case["kind"] == "synthetic":
        summary.update(score_synthetic_case(case, inputs, run_root, method_status))
    else:
        summary["artifacts"] = run_real_analysis(case, run_root, method_status)
    for method in METHODS:
        summary["methods"][method]["runtime_seconds"] = read_runtime(Path(summary["methods"][method]["result_dir"]))
    _write_json(run_root / "case_summary.json", summary)
    print(json.dumps({"case_id": case["case_id"], "summary": str(run_root / "case_summary.json")}, indent=2))


def _get_nested(data: dict, path: str):
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def summarize(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results_root = Path(manifest["results_root"])
    rows = []
    metric_paths = {
        "relvar_mean": "rel_var_mean",
        "mean_error": "mean_error",
        "mean_fsc": "pipeline_metrics.mean_fsc",
        "variance_spatial_fsc": "pipeline_metrics.variance_spatial_fsc",
        "variance_fourier_fsc": "pipeline_metrics.variance_fourier_fsc",
        "svd_relative_variance_4": "pipeline_metrics.svd_relative_variance_4",
        "svd_relative_variance_10": "pipeline_metrics.svd_relative_variance_10",
        "embedding_squared_error_20": "pipeline_metrics.embedding_squared_error_20",
        "state_0_locres_90pct": "pipeline_metrics.state_0_locres_90pct",
        "state_1_locres_90pct": "pipeline_metrics.state_1_locres_90pct",
    }
    for case in manifest["cases"]:
        summary_path = _case_root(results_root, case) / "case_summary.json"
        if not summary_path.exists():
            rows.append({"case_id": case["case_id"], "kind": case["kind"], "status": "missing"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "case_id": case["case_id"],
            "kind": case["kind"],
            "dataset": case["dataset"],
            "mask_mode": case["mask_mode"],
            "correct_contrast": case["correct_contrast"],
            "summary_path": str(summary_path),
            "status": "complete",
        }
        if case["kind"] == "synthetic":
            row.update(
                {
                    "n_images": case["n_images"],
                    "contrast_std": case["contrast_std"],
                    "noise_level": case["noise_level"],
                }
            )
        scores = summary.get("scores", {})
        for method in METHODS:
            method_summary = summary.get("methods", {}).get(method, {})
            row[f"{method}_exit_code"] = method_summary.get("exit_code")
            row[f"{method}_runtime_seconds"] = method_summary.get("runtime_seconds")
            row[f"{method}_result_dir"] = method_summary.get("result_dir", "")
            method_scores = scores.get(method, {})
            for out_key, metric_path in metric_paths.items():
                value = _get_nested(method_scores, metric_path)
                if isinstance(value, (int, float)):
                    row[f"{method}_{out_key}"] = value
        if "covariance" in scores and "ppca" in scores:
            cov = scores["covariance"].get("pipeline_metrics", {})
            ppca = scores["ppca"].get("pipeline_metrics", {})
            for key in ("mean_fsc", "variance_spatial_fsc", "variance_fourier_fsc", "svd_relative_variance_4", "svd_relative_variance_10"):
                if isinstance(cov.get(key), (int, float)) and isinstance(ppca.get(key), (int, float)):
                    row[f"ppca_minus_covariance_{key}"] = ppca[key] - cov[key]
        rows.append(row)

    out_dir = args.output_dir or (results_root / "aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "case_table.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        "created_at": _now(),
        "manifest": str(args.manifest),
        "results_root": str(results_root),
        "n_cases": len(manifest["cases"]),
        "n_completed_or_started": sum(1 for row in rows if row["status"] == "complete"),
        "case_table": str(csv_path),
        "rows": rows,
    }
    _write_json(out_dir / "aggregate_summary.json", aggregate)
    print(json.dumps(aggregate, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_manifest = sub.add_parser("write-manifest")
    p_manifest.add_argument("--repo-root", type=Path, default=Path.cwd())
    p_manifest.add_argument("--results-root", type=Path, required=True)
    p_manifest.add_argument("--output", type=Path, required=True)
    p_manifest.add_argument("--cryobench-root", type=Path, default=DEFAULT_CRYOBENCH_ROOT)
    p_manifest.add_argument("--recovar-datasets-root", type=Path, default=DEFAULT_RECOVAR_DATASETS_ROOT)
    p_manifest.add_argument("--grid-size", type=int, default=256)
    p_manifest.add_argument("--zdim", type=int, default=20)
    p_manifest.add_argument("--ppca-em-iters", type=int, default=20)
    p_manifest.add_argument("--noise-level", type=float, default=1.0)
    p_manifest.add_argument("--pdb-bfactor", type=float, default=80.0)
    p_manifest.add_argument("--seed", type=int, default=42)
    p_manifest.add_argument("--case-kind", choices=("all", "synthetic", "real"), default="all")
    p_manifest.set_defaults(func=write_manifest)

    p_run = sub.add_parser("run-case")
    p_run.add_argument("--manifest", type=Path, required=True)
    p_run.add_argument("--case-index", type=int, required=True)
    p_run.add_argument("--gpu-gb", type=float, default=None)
    p_run.set_defaults(func=run_case)

    p_summary = sub.add_parser("summarize")
    p_summary.add_argument("--manifest", type=Path, required=True)
    p_summary.add_argument("--output-dir", type=Path, default=None)
    p_summary.set_defaults(func=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
