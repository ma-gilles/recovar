#!/usr/bin/env python
"""Prepare/run no-contrast CryoBench PDB benchmarks for SOLVAR."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar import utils
from recovar.ppca import compare_covariance_vs_ppca_pipeline as compare
from recovar.simulation import synthetic_dataset

from scripts.prepare_cryobench_pdb_multiclass_relion_parity_benchmark import prepare_benchmark


DEFAULT_CRYOBENCH_ROOT = Path("/home/mg6942/mytigress/cryobench2")
DEFAULT_RESULTS_ROOT = Path("/projects/CRYOEM/singerlab/mg6942/solvar_no_contrast_benchmark")
DEFAULT_DATASETS = ("Ribosembly", "IgG-1D", "IgG-RL", "Tomotwin-100")
METHODS = ("covariance", "solvar-ls", "solvar-mle")
compare.METHOD_ORDER = METHODS
compare.METHOD_LABELS.update({"solvar-ls": "SOLVAR LS", "solvar-mle": "SOLVAR MLE"})
compare.METHOD_COLORS.update({"solvar-ls": "#9467bd", "solvar-mle": "#2ca02c"})


def _run_name(dataset: str, grid_size: int, n_images: int, noise_level: float, zdim: int, seed: int) -> str:
    snr = str(noise_level).replace(".", "p")
    return f"{dataset}_g{grid_size}_n{n_images}_snr{snr}_z{zdim}_seed{seed}_no_contrast"


def _dataset_root(args, dataset: str) -> Path:
    return args.results_root / _run_name(dataset, args.grid_size, args.n_images, args.noise_level, args.zdim, args.seed)


def _prepare_dataset(args, dataset: str) -> dict:
    run_root = _dataset_root(args, dataset)
    sim_dir = run_root / "simulated_data"
    pdb_dir = args.cryobench_root / dataset / "pdbs"
    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"missing CryoBench PDB directory: {pdb_dir}")
    prepare_benchmark(
        sim_dir,
        pdb_dir=pdb_dir,
        n_images=args.n_images,
        grid_size=args.grid_size,
        voxel_size=args.voxel_size,
        noise_level=args.noise_level,
        init_radius=args.init_radius,
        pdb_bfactor=args.pdb_bfactor,
        relion_normalize=False,
        streaming_mmap=args.streaming_mmap,
        streaming_chunk_size=args.streaming_chunk_size,
        disc_type=args.sim_disc_type,
        seed=args.seed,
        force_volumes=args.force_volumes,
    )
    halfsets_path = compare.ensure_halfsets(str(run_root), args.n_images, args.seed)
    return {"dataset": dataset, "run_root": str(run_root), "sim_dir": str(sim_dir), "halfsets_path": halfsets_path}


def _pipeline_command(args, method: str, sim_dir: str, halfsets_path: str, result_dir: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "recovar.commands.pipeline",
        os.path.join(sim_dir, f"particles.{args.grid_size}.mrcs"),
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
        str(args.zdim),
        "--no-downsample",
    ]
    if args.gpu_gb is not None:
        cmd.extend(["--gpu-budget-gb", str(args.gpu_gb)])
    if args.lazy:
        cmd.append("--lazy")
    if method == "solvar-ls":
        cmd.extend(
            [
                "--use-solvar",
                "--solvar-objective",
                "ls",
                "--solvar-zdim",
                str(args.zdim),
                "--solvar-iters",
                str(args.solvar_iters),
                "--solvar-learning-rate",
                str(args.solvar_learning_rate),
                "--solvar-init",
                args.solvar_init,
                "--solvar-warm-start-n-pcs",
                str(args.solvar_warm_start_n_pcs),
                "--solvar-batch-size",
                str(args.solvar_batch_size),
            ]
        )
    elif method == "solvar-mle":
        cmd.extend(
            [
                "--use-solvar",
                "--solvar-objective",
                "mle",
                "--solvar-zdim",
                str(args.zdim),
                "--solvar-iters",
                str(args.solvar_iters),
                "--solvar-learning-rate",
                str(args.solvar_learning_rate),
                "--solvar-init",
                args.solvar_init,
                "--solvar-warm-start-n-pcs",
                str(args.solvar_warm_start_n_pcs),
                "--solvar-batch-size",
                str(args.solvar_batch_size),
            ]
        )
    elif method != "covariance":
        raise ValueError(f"unknown method {method}")
    return cmd


def _write_runner(run_root: Path, method_specs: list[dict], *, continue_on_method_error: bool = False) -> str:
    runner = run_root / "run_methods.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for spec in method_specs:
        cmd = " ".join(shlex.quote(part) for part in spec["cmd"])
        params_path = Path(spec["result_dir"]) / "model" / "params.pkl"
        if continue_on_method_error:
            run_lines = [
                f"  if ! {cmd} > {shlex.quote(spec['log_path'])} 2>&1; then",
                f"    echo '[solvar-bench] {spec['method']} failed; continuing' >&2",
                "  fi",
            ]
        else:
            run_lines = [f"  {cmd} > {shlex.quote(spec['log_path'])} 2>&1"]
        lines.extend(
            [
                f"mkdir -p {shlex.quote(str(Path(spec['log_path']).parent))}",
                f"if [ -f {shlex.quote(str(params_path))} ]; then",
                f"  echo '[solvar-bench] skipping {spec['method']}; found {params_path}'",
                "else",
                f"  echo '[solvar-bench] running {spec['method']}'",
                *run_lines,
                "fi",
                "",
            ]
        )
    runner.write_text("\n".join(lines), encoding="utf-8")
    runner.chmod(0o755)
    return str(runner)


def prepare_run(args, dataset: str) -> dict:
    prep = _prepare_dataset(args, dataset)
    run_root = Path(prep["run_root"])
    method_specs = []
    selected_methods = tuple(args.methods or METHODS)
    for method in selected_methods:
        method_root = run_root / method
        result_dir = method_root / "result"
        method_root.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        cmd = _pipeline_command(args, method, prep["sim_dir"], prep["halfsets_path"], str(result_dir))
        command_path = result_dir / "command.txt"
        command_path.write_text(shlex.join(cmd) + "\n", encoding="utf-8")
        method_specs.append(
            {
                "method": method,
                "cmd": cmd,
                "result_dir": str(result_dir),
                "log_path": str(method_root / f"{method}.log"),
                "command_path": str(command_path),
            }
        )
    manifest = {
        **prep,
        "methods": list(selected_methods),
        "method_specs": [
            {k: v for k, v in spec.items() if k != "cmd"} | {"cmd": spec["cmd"]} for spec in method_specs
        ],
        "runner_script": _write_runner(
            run_root,
            method_specs,
            continue_on_method_error=bool(getattr(args, "continue_on_method_error", False)),
        ),
        "no_contrast": True,
        "solvar_init": args.solvar_init,
        "source": "CryoBench pdbs/ regenerated through prepare_cryobench_pdb_multiclass_relion_parity_benchmark.py",
    }
    manifest_path = run_root / "solvar_no_contrast_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def run_methods(
    manifest: dict,
    force: bool = False,
    continue_on_method_error: bool = False,
) -> dict[str, float | None]:
    runtimes = {}
    for spec in manifest["method_specs"]:
        result_params = Path(spec["result_dir"]) / "model" / "params.pkl"
        if result_params.exists() and not force:
            runtimes[spec["method"]] = compare._read_runtime_seconds(spec["result_dir"])
            continue
        started = time.time()
        with open(spec["log_path"], "w", encoding="utf-8") as log_fh:
            try:
                subprocess.run(spec["cmd"], check=True, stdout=log_fh, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                runtimes[spec["method"]] = None
                if continue_on_method_error:
                    continue
                raise
        runtimes[spec["method"]] = time.time() - started
    return runtimes


def score_run(args, manifest: dict, runtimes: dict[str, float | None] | None = None) -> dict:
    sim_info_path = Path(manifest["sim_dir"]) / "simulation_info.pkl"
    sim_info = utils.pickle_load(sim_info_path)
    gt_results = synthetic_dataset.load_heterogeneous_reconstruction(str(sim_info_path))
    volume_shape = (args.grid_size, args.grid_size, args.grid_size)
    metric_context = _build_metric_context(gt_results, sim_info, volume_shape, args.zdim)
    scores = {}
    for spec in manifest["method_specs"]:
        params_path = Path(spec["result_dir"]) / "model" / "params.pkl"
        if params_path.exists():
            scores[spec["method"]] = compare.score_pipeline_output(
                spec["result_dir"],
                str(Path(spec["result_dir"]).parent),
                gt_results,
                metric_context,
                args.zdim,
            )
    plot_paths = {}
    plot_error = None
    if scores:
        try:
            plot_paths = compare.write_comparison_plots(manifest["run_root"], scores, args.zdim)
        except Exception as exc:  # Plotting should not invalidate benchmark metrics.
            plot_error = f"{type(exc).__name__}: {exc}"
    summary = {
        "dataset": manifest["dataset"],
        "grid_size": args.grid_size,
        "n_images": args.n_images,
        "noise_level": args.noise_level,
        "zdim": args.zdim,
        "seed": args.seed,
        "no_contrast": True,
        "scores": scores,
        "runtimes_seconds": runtimes or {},
        "plots": plot_paths,
        "manifest_path": manifest["manifest_path"],
    }
    if plot_error is not None:
        summary["plot_error"] = plot_error
    summary_path = Path(manifest["run_root"]) / "solvar_no_contrast_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_metric_context(gt_results, sim_info: dict, volume_shape: tuple[int, int, int], zdim: int) -> dict:
    gt_mean = np.asarray(gt_results.get_mean())
    gt_union_soft_mask, gt_union_binary_mask = compare.metrics.make_union_gt_mask_from_hvd(gt_results, volume_shape)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cryobench-root", type=Path, default=DEFAULT_CRYOBENCH_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--dataset", action="append", choices=DEFAULT_DATASETS, help="Dataset to run; repeatable.")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--n-images", type=int, default=100000)
    parser.add_argument("--noise-level", type=float, default=1.0)
    parser.add_argument("--zdim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--init-radius", type=int, default=10)
    parser.add_argument("--pdb-bfactor", type=float, default=80.0)
    parser.add_argument("--sim-disc-type", choices=["nufft", "linear_interp", "pdb"], default="nufft")
    parser.add_argument("--streaming-mmap", action="store_true")
    parser.add_argument("--streaming-chunk-size", type=int, default=2048)
    parser.add_argument("--gpu-gb", type=float, default=None)
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--solvar-iters", type=int, default=40)
    parser.add_argument("--solvar-learning-rate", type=float, default=1e-6)
    parser.add_argument("--solvar-init", choices=("covariance", "random"), default="covariance")
    parser.add_argument("--solvar-warm-start-n-pcs", type=int, default=0)
    parser.add_argument("--solvar-batch-size", type=int, default=200)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--continue-on-method-error",
        action="store_true",
        help="Continue later methods if one method fails; useful for isolating SOLVAR from a flaky covariance baseline.",
    )
    parser.add_argument("--force-volumes", action="store_true")
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--stage", choices=("prepare", "full", "score"), default="prepare")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = args.dataset or list(DEFAULT_DATASETS)
    summaries = []
    for dataset in datasets:
        if args.stage == "score":
            manifest_path = _dataset_root(args, dataset) / "solvar_no_contrast_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["manifest_path"] = str(manifest_path)
            else:
                manifest = prepare_run(args, dataset)
        else:
            manifest = prepare_run(args, dataset)
        runtimes = {}
        if args.stage == "full":
            runtimes = run_methods(
                manifest,
                force=args.force_run,
                continue_on_method_error=args.continue_on_method_error,
            )
            summaries.append(score_run(args, manifest, runtimes))
        elif args.stage == "score":
            summaries.append(score_run(args, manifest, runtimes))
        else:
            summaries.append(manifest)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
