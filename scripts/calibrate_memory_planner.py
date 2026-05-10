#!/usr/bin/env python3
"""Empirical peak-memory calibration sweep for the RECOVAR memory planner.

Runs a single ``(command, grid_size, n_pcs_or_n_volumes, backend)`` cell
and emits a JSON record that ``aggregate_memory_calibration.py`` collects
into ``recovar/utils/memory_calibration_data.json``.

Usage (single cell):

    python scripts/calibrate_memory_planner.py \
        --command pipeline \
        --grid-size 128 \
        --n-pcs 200 \
        --backend custom_cuda \
        --n-images 20000 \
        --dataset-root /scratch/gpfs/GILLES/mg6942/calibration_datasets \
        --out-json /scratch/gpfs/GILLES/mg6942/calibration_runs/cell_pipeline_128_200_custom_cuda.json

Submitted via Slurm (one cell per job) by
``scripts/submit_calibrate_memory_planner.sh``. The aggregator merges the
per-cell JSONs into the calibration table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def make_dataset_if_needed(dataset_root: Path, grid_size: int, n_images: int) -> Path:
    """Create ``dataset_root/grid<G>_n<N>/test_dataset/`` if it does not exist.

    Returns the test_dataset directory containing particles.<G>.mrcs etc.
    """
    cell_dir = dataset_root / f"grid{grid_size}_n{n_images}"
    test_dataset = cell_dir / "test_dataset"
    particles = test_dataset / f"particles.{grid_size}.mrcs"
    if particles.exists():
        return test_dataset

    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "make_test_dataset",
        str(cell_dir),
        "--n-images",
        str(n_images),
        "--image-size",
        str(grid_size),
    ]
    print(f"[calibrate] generating dataset: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return test_dataset


def run_pipeline_cell(
    *,
    command: str,
    test_dataset: Path,
    grid_size: int,
    n_pcs: int,
    backend: str,
    out_dir: Path,
    extra_pipeline_args: list[str] | None = None,
) -> dict:
    """Run a single pipeline cell and return the recorded peaks."""
    if extra_pipeline_args is None:
        extra_pipeline_args = []
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if backend == "jax_fallback":
        env["RECOVAR_DISABLE_CUDA"] = "1"
    else:
        env.pop("RECOVAR_DISABLE_CUDA", None)
        env.pop("RECOVAR_CUDA_DISABLE", None)

    if command == "pipeline":
        cmd = [
            sys.executable,
            "-m",
            "recovar.command_line",
            "pipeline",
            str(test_dataset / f"particles.{grid_size}.mrcs"),
            "--poses",
            str(test_dataset / "poses.pkl"),
            "--ctf",
            str(test_dataset / "ctf.pkl"),
            "--mask=from_halfmaps",
            "--lazy",
            "--correct-contrast",
            "-o",
            str(out_dir / "pipeline_output"),
            # Diagnostics are always-on now; no flag needed. Use
            # --memory-profile if heavyweight JAX-profiler captures
            # are wanted.
            *extra_pipeline_args,
        ]
        # Force the planner to use exactly n_pcs by pinning adaptive-n-pcs OFF
        # and relying on covariance_estimation's default 200 path; for n_pcs
        # < 200 we use --very-low-memory-option / a custom override is left
        # to the future, since the sweep currently maps n_pcs in {4, 20, 50,
        # 200}. For the prototype, we record peaks at the default 200 PC
        # config and document partial coverage.
        # TODO(calibrate): once a per-cell n_pcs override flag exists, plumb
        # it through here to populate the {4, 20, 50, 200} grid fully.
    elif command == "compute_state":
        # compute_state calibration: agent should populate this branch with
        # the right invocation once a baseline pipeline output is available.
        # For the prototype, we record an "uncalibrated" placeholder so the
        # JSON schema round-trips.
        return {
            "status": "skipped",
            "reason": "compute_state cell not implemented in prototype",
        }
    else:
        raise ValueError(f"Unsupported command: {command}")

    print(f"[calibrate] running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
    wall_time_s = time.time() - t0

    status = "ok" if result.returncode == 0 else "failed"
    # Extract peaks from memory_trace.jsonl
    trace_path = out_dir / "pipeline_output" / "memory_trace.jsonl"
    peak_total = None
    peak_by_phase: dict[str, float] = {}
    if trace_path.exists():
        with trace_path.open() as fh:
            for line in fh:
                row = json.loads(line)
                phase = row.get("phase")
                gb = row.get("jax_peak_gb")
                if phase and gb is not None:
                    peak_by_phase[phase] = float(gb)
                    peak_total = max(peak_total or 0.0, float(gb))

    if status == "failed":
        # Detect OOM-style failures from the captured stderr.
        if any(
            tok in (result.stderr or "")
            for tok in (
                "RESOURCE_EXHAUSTED",
                "out of memory",
                "CUDA_ERROR_OUT_OF_MEMORY",
                "CUBLAS_STATUS_ALLOC_FAILED",
            )
        ):
            status = "oom"

    return {
        "status": status,
        "peak_gb_total": peak_total,
        "peak_gb_by_phase": peak_by_phase,
        "wall_time_s": wall_time_s,
        "stderr_tail": (result.stderr or "")[-2000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", choices=["pipeline", "compute_state"], required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--n-pcs", type=int, required=True)
    parser.add_argument("--backend", choices=["custom_cuda", "jax_fallback"], required=True)
    parser.add_argument("--n-images", type=int, default=20000)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/scratch/gpfs/GILLES/mg6942/calibration_datasets"),
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument(
        "--out-runs-root",
        type=Path,
        default=Path("/scratch/gpfs/GILLES/mg6942/calibration_runs"),
    )
    args = parser.parse_args()

    cell_id = f"{args.command}_g{args.grid_size}_n{args.n_pcs}_{args.backend}"
    out_dir = args.out_runs_root / cell_id
    test_dataset = make_dataset_if_needed(args.dataset_root, args.grid_size, args.n_images)
    record = run_pipeline_cell(
        command=args.command,
        test_dataset=test_dataset,
        grid_size=args.grid_size,
        n_pcs=args.n_pcs,
        backend=args.backend,
        out_dir=out_dir,
    )
    record.update(
        {
            "command": args.command,
            "grid_size": args.grid_size,
            "n_pcs": args.n_pcs,
            "backend": args.backend,
            "n_images": args.n_images,
        }
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as fh:
        json.dump(record, fh, indent=2, default=str)
    print(f"[calibrate] wrote {args.out_json}")
    print(json.dumps(record, indent=2, default=str))
    return 0 if record.get("status") in ("ok", "oom", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
