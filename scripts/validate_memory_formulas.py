#!/usr/bin/env python3
"""Two-pass memory-model validation sweep for RECOVAR.

  --mode record:   run a cell, capture observed peaks, write a JSON record
  --mode validate: read previously-recorded peaks, assert predictions
                   match within slack
  --mode discover: a focused n_pcs sweep at fixed (grid, backend, pipeline)
                   to fit the exponent of the SVD-workspace term

A "cell" is one tuple of (pipeline, grid_size, backend, n_pcs). Each
cell runs ``recovar pipeline ...`` (or ``recovar run_test_dataset``)
under ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` so observed peaks reflect
actual demand rather than JAX's preallocated slab.

Output JSON schema:

    {
      "schema_version": 1,
      "host": {
        "gpu_kind": "NVIDIA H100",
        "driver":   "...",
        "jax":      "...",
        "git_head": "61bba9ca...",
      },
      "cells": [
        {
          "pipeline": "spa",
          "grid_size": 128,
          "n_pcs": 200,
          "backend": "custom_cuda",
          "status": "ok" | "oom" | "error",
          "observed_peaks_gb": {
            "after_mean": 6.2,
            "after_covariance": 31.4,
            "after_embedding": 12.8
          },
          "predicted_peaks_gb": {
            "after_mean": 5.9,
            "after_covariance": 78.3,
            "after_embedding": 12.4
          },
          "wall_time_s": 412.0
        },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def host_info() -> dict:
    info = {"gpu_kind": "unknown", "driver": "unknown", "jax": "unknown", "git_head": "unknown"}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, driver = (s.strip() for s in out.stdout.strip().splitlines()[0].split(","))
            info["gpu_kind"] = name
            info["driver"] = driver
    except Exception:
        pass
    try:
        import jax

        info["jax"] = jax.__version__
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            info["git_head"] = out.stdout.strip()
    except Exception:
        pass
    return info


def build_dataset_if_needed(*, dataset_root: Path, pipeline: str, grid_size: int) -> Path:
    """Generate a tiny synthetic dataset for the cell. Reuses across cells
    of the same (pipeline, grid_size)."""
    if pipeline == "spa":
        cell_dir = dataset_root / f"spa_grid{grid_size}_n2000"
        particles = cell_dir / "test_dataset" / f"particles.{grid_size}.mrcs"
        if particles.exists():
            return cell_dir / "test_dataset"
        cell_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "recovar.command_line",
            "make_test_dataset",
            str(cell_dir),
            "--n-images",
            "2000",
            "--image-size",
            str(grid_size),
        ]
        subprocess.run(cmd, check=True)
        return cell_dir / "test_dataset"
    elif pipeline == "tilt_series":
        cell_dir = dataset_root / f"et_grid{grid_size}_p50t41"
        particles = cell_dir / "test_dataset" / "particles.star"
        if particles.exists():
            return cell_dir / "test_dataset"
        cell_dir.mkdir(parents=True, exist_ok=True)
        # n-images = particles × tilts; tilt-series make_test_dataset accepts
        # a particle count via --n-images.
        n_images = 50
        cmd = [
            sys.executable,
            "-m",
            "recovar.command_line",
            "make_test_dataset",
            str(cell_dir),
            "--n-images",
            str(n_images),
            "--tilt-series",
            "--image-size",
            str(grid_size),
        ]
        subprocess.run(cmd, check=True)
        return cell_dir / "test_dataset"
    else:
        raise ValueError(f"unknown pipeline {pipeline}")


def build_pipeline_argv(
    *,
    pipeline: str,
    grid_size: int,
    n_pcs: int,
    backend: str,
    dataset_dir: Path,
    out_dir: Path,
    budget_gb: float | None = None,
) -> list[str]:
    """argv for one ``recovar pipeline`` invocation.

    If ``budget_gb`` is None the cell does NOT pass ``--gpu-budget-gb``
    and we observe the actual peak under the user's (or our) allocator
    settings. If ``budget_gb`` is given we constrain the planner to
    that budget (saturation sweeps need this to measure peak vs budget).
    """
    base = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "pipeline",
    ]
    if pipeline == "spa":
        particles = dataset_dir / f"particles.{grid_size}.mrcs"
        argv = base + [
            str(particles),
            "--poses",
            str(dataset_dir / "poses.pkl"),
            "--ctf",
            str(dataset_dir / "ctf.pkl"),
            "--mask=from_halfmaps",
            "--lazy",
            "--correct-contrast",
            "-o",
            str(out_dir),
        ]
    else:
        particles = dataset_dir / "particles.star"
        argv = base + [
            str(particles),
            "--poses",
            str(dataset_dir / "poses.pkl"),
            "--ctf",
            str(dataset_dir / "ctf.pkl"),
            "--tilt-series",
            "--tilt-series-ctf=relion5",
            "--mask=from_halfmaps",
            "--lazy",
            "--correct-contrast",
            "-o",
            str(out_dir),
        ]
    # n_pcs is forced by ``RECOVAR_DEBUG_FORCE_N_PCS`` in the subprocess
    # env; not by argv (no user-facing CLI flag for it).
    # Always pass --memory-profile in sweeps so memory_trace.jsonl
    # is written. (Production runs default to no trace to avoid
    # the JAX memory_stats() + nvidia-smi probe cost.)
    argv.append("--memory-profile")
    if budget_gb is not None:
        argv.extend(["--gpu-budget-gb", str(budget_gb)])
    return argv


def run_cell(
    *,
    pipeline: str,
    grid_size: int,
    n_pcs: int,
    backend: str,
    dataset_root: Path,
    runs_root: Path,
    budget_gb: float | None = None,
) -> dict:
    """Run one cell. Returns a record dict (status, peaks, wall time)."""
    budget_tag = f"_b{int(budget_gb)}" if budget_gb is not None else ""
    cell_id = f"{pipeline}_g{grid_size}_n{n_pcs}_{backend}{budget_tag}"
    cell_out = runs_root / cell_id
    cell_out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # observe actual demand
    # Force this cell's n_pcs into the covariance pipeline. Honored by
    # recovar/commands/pipeline.py post-planner-override.
    env["RECOVAR_DEBUG_FORCE_N_PCS"] = str(n_pcs)
    if backend == "jax_fallback":
        env["RECOVAR_DISABLE_CUDA"] = "1"
    else:
        env.pop("RECOVAR_DISABLE_CUDA", None)
        env.pop("RECOVAR_CUDA_DISABLE", None)

    dataset_dir = build_dataset_if_needed(
        dataset_root=dataset_root,
        pipeline=pipeline,
        grid_size=grid_size,
    )

    argv = build_pipeline_argv(
        pipeline=pipeline,
        grid_size=grid_size,
        n_pcs=n_pcs,
        backend=backend,
        dataset_dir=dataset_dir,
        out_dir=cell_out / "pipeline_out",
        budget_gb=budget_gb,
    )

    print(f"[{cell_id}] running: {' '.join(shlex.quote(a) for a in argv)}", flush=True)
    t0 = time.time()
    result = subprocess.run(argv, env=env, capture_output=True, text=True, check=False)
    wall_time_s = time.time() - t0

    status = "ok" if result.returncode == 0 else "error"
    if status == "error":
        # Detect OOM signatures.
        combined = (result.stderr or "") + (result.stdout or "")
        oom_tokens = ["RESOURCE_EXHAUSTED", "out of memory", "CUDA_ERROR_OUT_OF_MEMORY"]
        if any(t in combined for t in oom_tokens):
            status = "oom"

    # Read observed peaks from memory_trace.jsonl.
    observed: dict[str, float] = {}
    trace_paths = list((cell_out / "pipeline_out").rglob("memory_trace.jsonl"))
    if not trace_paths:
        # Fallback: maybe in _diagnostics/
        trace_paths = list((cell_out / "pipeline_out").rglob("_diagnostics/memory_trace.jsonl"))
    for tp in trace_paths:
        for line in tp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            phase = row.get("phase")
            peak = row.get("jax_peak_gb")
            if phase and isinstance(peak, (int, float)):
                observed[phase] = max(observed.get(phase, 0.0), float(peak))

    # Read planner predictions from memory_plan.json.
    predicted: dict[str, float] = {}
    breakdown_predictions: dict[str, dict] = {}
    plan_paths = list((cell_out / "pipeline_out").rglob("memory_plan.json"))
    if not plan_paths:
        plan_paths = list((cell_out / "pipeline_out").rglob("_diagnostics/memory_plan.json"))
    if plan_paths:
        try:
            plan = json.loads(plan_paths[0].read_text())
            preds = plan.get("predicted_peak_gb_by_phase", {}) or {}
            predicted = {f"after_{k}": float(v) for k, v in preds.items()}
        except Exception:
            pass

    return {
        "pipeline": pipeline,
        "grid_size": grid_size,
        "n_pcs": n_pcs,
        "backend": backend,
        "budget_gb": budget_gb,
        "cell_id": cell_id,
        "status": status,
        "wall_time_s": wall_time_s,
        "observed_peaks_gb": observed,
        "predicted_peaks_gb": predicted,
        "stderr_tail": (result.stderr or "")[-2000:] if status != "ok" else "",
    }


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def discovery_cells() -> list[dict]:
    """Phase 2: focused n_pcs sweep at one fixed cell."""
    return [
        {"pipeline": "spa", "grid_size": 128, "n_pcs": n, "backend": "custom_cuda"}
        for n in (10, 20, 40, 80, 120, 160, 200)
    ]


def validation_cells() -> list[dict]:
    """Phase 3: full validation matrix."""
    cells = []
    for pipeline in ("spa", "tilt_series"):
        for grid_size in (64, 128, 256):
            for backend in ("custom_cuda", "jax_fallback"):
                for n_pcs in (20, 50, 100, 200):
                    cells.append(
                        {
                            "pipeline": pipeline,
                            "grid_size": grid_size,
                            "n_pcs": n_pcs,
                            "backend": backend,
                        }
                    )
    return cells


def saturation_cells() -> list[dict]:
    """Phase A: measure HEADROOM_SATURATION = peak / budget per (grid, backend).

    Three sub-sweeps:

      A1 — grid × backend at full JAX budget (6 cells). Observes peak
           when planner has the whole GPU. Yields the per-(grid, backend)
           "natural" peak ratio.

      A2 — budget sweep at fixed grid=128 custom_cuda, n_pcs=50 (5 cells).
           Confirms peak tracks budget linearly. If linear, a single
           SATURATION multiplier suffices; if sub-linear the legacy
           formula's grid-scaling is the limiter.

      A3 — jax_fallback at constrained budgets (6 cells). Cells in A1
           that OOM under full budget reappear here with --gpu-budget-gb
           forcing the planner to pick smaller batches. We measure peak
           there to derive jax_fallback's saturation.

    Total: 17 cells. Some A1 cells (jax_fallback at g≥128) will OOM —
    that's expected; their measurement comes from A3.
    """
    cells: list[dict] = []

    # A1: grid × backend at full budget. n_pcs=50 fixed (constant in
    # n_pcs per discovery sweep).
    for grid in (64, 128, 256):
        for backend in ("custom_cuda", "jax_fallback"):
            cells.append(
                {
                    "pipeline": "spa",
                    "grid_size": grid,
                    "n_pcs": 50,
                    "backend": backend,
                    "budget_gb": None,
                }
            )

    # A2: budget sweep at the well-understood grid=128 custom_cuda cell.
    for budget in (16, 24, 40, 60, 76):
        cells.append(
            {
                "pipeline": "spa",
                "grid_size": 128,
                "n_pcs": 50,
                "backend": "custom_cuda",
                "budget_gb": float(budget),
            }
        )

    # A3: jax_fallback at constrained budgets (g=64 and g=128).
    for grid in (64, 128):
        for budget in (12, 24, 40):
            cells.append(
                {
                    "pipeline": "spa",
                    "grid_size": grid,
                    "n_pcs": 50,
                    "backend": "jax_fallback",
                    "budget_gb": float(budget),
                }
            )

    return cells


def fast_cells() -> list[dict]:
    """A 4-cell smoke test for harness debugging — not for fitting."""
    return [
        {"pipeline": "spa", "grid_size": 64, "n_pcs": 50, "backend": "custom_cuda"},
        {"pipeline": "spa", "grid_size": 128, "n_pcs": 50, "backend": "custom_cuda"},
        {"pipeline": "tilt_series", "grid_size": 64, "n_pcs": 50, "backend": "custom_cuda"},
        {"pipeline": "tilt_series", "grid_size": 128, "n_pcs": 50, "backend": "custom_cuda"},
    ]


def cmd_record(args) -> int:
    cells_to_run = {
        "discover": discovery_cells,
        "validate": validation_cells,
        "saturation": saturation_cells,
        "fast": fast_cells,
    }[args.cells]()

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    records = []
    for cell in cells_to_run:
        rec = run_cell(
            pipeline=cell["pipeline"],
            grid_size=cell["grid_size"],
            n_pcs=cell["n_pcs"],
            backend=cell["backend"],
            dataset_root=dataset_root,
            runs_root=runs_root,
            budget_gb=cell.get("budget_gb"),
        )
        records.append(rec)
        print(
            f"[{rec['cell_id']}] status={rec['status']} "
            f"observed={rec['observed_peaks_gb']} "
            f"predicted={rec['predicted_peaks_gb']} "
            f"wall_time={rec['wall_time_s']:.1f}s",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "host": host_info(),
        "cells_set": args.cells,
        "cells": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    return 0


def cmd_validate(args) -> int:
    """Read a recorded sweep, assert predictions ≤ observed × slack."""
    payload = json.loads(Path(args.baseline).read_text())
    slack = args.slack
    overpred_threshold = 2.0  # predicted > 2× observed = warning

    failures = []
    overpreds = []
    for rec in payload["cells"]:
        if rec["status"] != "ok":
            continue
        for phase, observed in rec["observed_peaks_gb"].items():
            predicted = rec["predicted_peaks_gb"].get(phase, 0.0)
            if predicted == 0.0:
                continue
            underpred = observed / predicted
            overpred = predicted / observed
            if underpred > slack:
                failures.append(
                    f"{rec['cell_id']} phase={phase}: predicted={predicted:.2f} GB "
                    f"observed={observed:.2f} GB ratio={underpred:.2f} > slack={slack}"
                )
            if overpred > overpred_threshold:
                overpreds.append(
                    f"{rec['cell_id']} phase={phase}: predicted={predicted:.2f} GB "
                    f"observed={observed:.2f} GB ratio={overpred:.2f} (overprediction)"
                )

    print(f"Validation against {args.baseline} with slack {slack}:")
    print(f"  underprediction failures: {len(failures)}")
    for f in failures[:20]:
        print(f"    FAIL: {f}")
    print(f"  overprediction warnings: {len(overpreds)}")
    for o in overpreds[:10]:
        print(f"    WARN: {o}")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    rec = sub.add_parser("record", help="run cells and write JSON record")
    rec.add_argument("--cells", choices=("discover", "validate", "saturation", "fast"), default="fast")
    rec.add_argument("--runs-root", default="/scratch/gpfs/GILLES/mg6942/_agent_scratch/sweep_runs")
    rec.add_argument("--dataset-root", default="/scratch/gpfs/GILLES/mg6942/_agent_scratch/sweep_datasets")
    rec.add_argument("--output", required=True)
    rec.set_defaults(func=cmd_record)

    val = sub.add_parser("validate", help="check predictions against a recorded sweep")
    val.add_argument("--baseline", required=True)
    val.add_argument(
        "--slack", type=float, default=1.20, help="development slack; use 1.10 for pinned H100, 1.30 for cross-hardware"
    )
    val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
