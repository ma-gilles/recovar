"""GPU integration matrix for the memory planner / diagnostics.

Two flavors:

  * "fast" (under ``--run-gpu``): 3 budgets x custom_cuda, ~5 min total.
    Verifies the planner -> run_test_dataset wiring on real GPU.

  * "full" (under ``--long-test``, marker ``gpu_memory_matrix``): 7
    budgets x 2 backends, run via Slurm submitter
    ``scripts/run_gpu_memory_matrix.sh`` rather than inline pytest, so
    pytest just shells out to a single representative cell here.

Each cell:
  1. ``recovar run_test_dataset --gpu-gb N --memory-diagnostics --no-delete``
  2. asserts ``memory_plan.json`` exists under the output dir
  3. asserts ``memory_trace.jsonl`` exists and the recorded peaks fit
     within ``budget * 1.2``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _resolve_output_dir(tmp_path: Path, name: str) -> Path:
    base = os.environ.get("LONG_METRICS_OUTPUT_BASE")
    if base:
        out = Path(base) / "pytest_gpu_memory_matrix" / name
    else:
        out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _run_cell(out_dir: Path, *, gpu_gb: int, backend: str) -> None:
    from conftest import gpu_subprocess_env  # noqa: F401  (path-injected)

    env = gpu_subprocess_env()
    if backend == "jax_fallback":
        env["RECOVAR_DISABLE_CUDA"] = "1"
    else:
        env.pop("RECOVAR_DISABLE_CUDA", None)
        env.pop("RECOVAR_CUDA_DISABLE", None)

    cmd = [
        sys.executable,
        "-m",
        "recovar.command_line",
        "run_test_dataset",
        "--output-dir",
        str(out_dir),
        "--gpu-gb",
        str(gpu_gb),
        "--memory-diagnostics",
        "--no-delete",
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.stderr.write(result.stdout)
        pytest.fail(f"run_test_dataset --gpu-gb {gpu_gb} backend={backend} exited {result.returncode}")


def _assert_diagnostics(out_dir: Path, *, gpu_gb: float) -> None:
    plan_paths = list(out_dir.rglob("memory_plan.json"))
    assert plan_paths, f"No memory_plan.json under {out_dir}"
    for p in plan_paths:
        data = json.loads(p.read_text())
        assert "budget" in data
        assert data["budget"]["effective_budget_gb"] <= gpu_gb * 1.2 + 1e-3

    trace_paths = list(out_dir.rglob("memory_trace.jsonl"))
    assert trace_paths, f"No memory_trace.jsonl under {out_dir}"
    for p in trace_paths:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        peaks = [
            float(row["jax_peak_gb"]) for row in rows if row.get("jax_memory_stats_available") and "jax_peak_gb" in row
        ]
        if peaks:
            # 5% slack same as production --fail-on-memory-exceed
            assert max(peaks) <= gpu_gb * 1.05 + 1e-3, (
                f"peak {max(peaks)} GB exceeded {gpu_gb} GB by more than 5% slack"
            )


@pytest.mark.parametrize("gpu_gb", [16, 40, 75])
def test_memory_matrix_fast(tmp_path, gpu_gb):
    """Fast set: small dataset, default custom_cuda backend."""
    out_dir = _resolve_output_dir(tmp_path, f"fast_{gpu_gb}gb")
    _run_cell(out_dir, gpu_gb=gpu_gb, backend="custom_cuda")
    _assert_diagnostics(out_dir, gpu_gb=gpu_gb)


@pytest.mark.long_test
@pytest.mark.gpu_memory_matrix
@pytest.mark.parametrize(
    "gpu_gb,backend",
    [
        (8, "custom_cuda"),
        (12, "custom_cuda"),
        (16, "custom_cuda"),
        (24, "custom_cuda"),
        (40, "custom_cuda"),
        (60, "custom_cuda"),
        (75, "custom_cuda"),
        (8, "jax_fallback"),
        (12, "jax_fallback"),
        (16, "jax_fallback"),
        (24, "jax_fallback"),
        (40, "jax_fallback"),
        (60, "jax_fallback"),
        (75, "jax_fallback"),
    ],
)
def test_memory_matrix_long(tmp_path, gpu_gb, backend):
    """Long set: the real proof, 14 cells. Runs under ``--long-test``."""
    out_dir = _resolve_output_dir(tmp_path, f"long_{backend}_{gpu_gb}gb")
    _run_cell(out_dir, gpu_gb=gpu_gb, backend=backend)
    _assert_diagnostics(out_dir, gpu_gb=gpu_gb)
