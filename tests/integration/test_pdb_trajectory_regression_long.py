"""Long regression test using PDB-based 5nrl trajectory volumes.

Generates a synthetic cryo-EM dataset from rigid-body subcomplex motions of
PDB 5nrl (pre-catalytic spliceosome), runs the recovar pipeline, and compares
all metrics against a committed baseline.

The baseline was established by running the old pipeline (commit 911604e,
~/recovar) on a PDB-generated dataset with noise_level=0.1 (high SNR).
This test verifies that the current code matches or exceeds the old
pipeline's quality on realistic PDB-derived data.

The dataset is **reproducible**: given the same volumes, the simulator uses
deterministic seeds (seed=0) and fixed parameters.  The PDB volumes themselves
are deterministic given the 5nrl_atoms.npz asset + trajectory parameters.

Environment variables:
    PDB_REGRESSION_OUTPUT_BASE : str
        Base directory for large outputs (recommended: /scratch).
        Defaults to pytest tmp_path if unset.
    PDB_REGRESSION_BASELINE_JSON : str
        Path to baseline JSON.  Defaults to in-repo baseline.
    PDB_REGRESSION_TOL_FRAC : float
        Tolerated relative degradation (default 0.10 = 10%).
    PDB_REGRESSION_REUSE_DATASET : str
        Set to "1" to reuse an existing dataset in the output dir.
    PDB_REGRESSION_RUN_ARGS : str
        Extra args for run_test_all_metrics (overrides defaults).
"""

import json
import os
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers.metrics_regression import compare_metric, metric_direction, log_comparison_table
from helpers.perf_regression import perf_snapshot, stage_perf, build_perf_record, check_perf_regression, run_tracked_subprocess

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.io,
    pytest.mark.long_test,
]

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Default baseline: established from old pipeline (commit 911604e) on PDB dataset
# with noise_level=0.1 (high SNR).
_DEFAULT_PDB_BASELINE_JSON = (
    _REPO_ROOT / "tests" / "baselines" / "run_test_all_metrics" / "pdb_5nrl_old_pipeline" / "all_scores.json"
)

# Default run args matching the PDB dataset generation parameters.
_DEFAULT_RUN_ARGS = (
    "--grid-size 128 --n-images 50000 --noise-level 0.1 --contrast-std 0.1 "
    "--generate-pdb-volumes --generated-n-volumes 50 "
    "--pdb-bfactor 80.0 --pdb-max-rotation 10.0"
)

# Dataset generation parameters — stored here for reproducibility documentation.
# To regenerate the canonical dataset from scratch:
#   python -m recovar.commands.run_test_all_metrics \
#       --output-dir <OUT> \
#       --grid-size 128 --n-images 50000 --noise-level 0.1 --contrast-std 0.1 \
#       --generate-pdb-volumes --generated-n-volumes 50 \
#       --pdb-bfactor 80.0 --pdb-max-rotation 10.0 --no-delete
#
# The volumes are generated from recovar/assets/5nrl_atoms.npz using:
#   recovar.simulation.trajectory_generation.generate_trajectory_volumes(
#       grid_size=128, n_volumes=50, voxel_size=4.25,
#       Bfactor=80, max_rotation_degrees=10.0,
#       path_fn=path_symmetric,  # both B and Db rotate together
#   )
#
# The simulator uses seed=0 (hardcoded in simulator.py), so given the same
# volumes + parameters, the particles/poses/CTF are deterministic.


def _resolve_output_dir(tmp_path: Path, name: str) -> Path:
    base = os.environ.get("PDB_REGRESSION_OUTPUT_BASE")
    if base:
        out_dir = Path(base) / "pytest_pdb_regression" / name
    else:
        out_dir = tmp_path / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _run_pdb_metrics(output_dir, run_args, reuse_dataset=False):
    """Run run_test_all_metrics with PDB volume generation."""
    cmd = [
        sys.executable,
        "-m",
        "recovar.commands.run_test_all_metrics",
        "--output-dir",
        str(output_dir),
    ]
    if reuse_dataset:
        cmd += ["--reuse-dataset"]
    if run_args:
        cmd.extend(shlex.split(run_args))
    from conftest import gpu_subprocess_env

    run_tracked_subprocess(cmd, check=True, env=gpu_subprocess_env())
    score_path = output_dir / "test_dataset" / "metrics_plot" / "all_scores.json"
    assert score_path.exists(), f"missing score file at {score_path}"
    with open(score_path, "r") as f:
        return json.load(f)


def test_pdb_trajectory_regression(tmp_path):
    """
    Long regression test (~30min–1h) using PDB-based 5nrl trajectory volumes.

    Generates a reproducible synthetic dataset from 5nrl spliceosome
    rigid-body motions, runs the current pipeline, and compares metrics
    against the committed baseline (current pipeline on same data).

    This test catches regressions: any code change that degrades
    reconstruction quality on realistic PDB-derived data beyond the
    tolerance threshold will fail.
    """
    baseline_json = Path(os.environ.get("PDB_REGRESSION_BASELINE_JSON", str(_DEFAULT_PDB_BASELINE_JSON)))
    run_args = os.environ.get("PDB_REGRESSION_RUN_ARGS", _DEFAULT_RUN_ARGS)
    tol_frac = float(os.environ.get("PDB_REGRESSION_TOL_FRAC", "0.10"))

    output_dir = _resolve_output_dir(tmp_path, "pdb_current")
    reuse = (
        os.environ.get("PDB_REGRESSION_REUSE_DATASET", "0") == "1"
        and (output_dir / "test_dataset" / "simulation_info.pkl").exists()
    )

    snap_before = perf_snapshot()
    current = _run_pdb_metrics(output_dir, run_args, reuse_dataset=reuse)
    snap_after = perf_snapshot()

    perf_stages = {"pdb_trajectory": stage_perf(snap_before, snap_after)}
    perf_record = build_perf_record(perf_stages)
    perf_baseline_path = str(
        _REPO_ROOT / "tests" / "baselines" / "run_test_all_metrics" / "pdb_5nrl_old_pipeline" / "perf_baseline.json"
    )
    check_perf_regression(perf_record, perf_baseline_path, "test_pdb_trajectory_regression")

    assert baseline_json.exists(), f"PDB baseline not found: {baseline_json}"
    baseline = _load_json(baseline_json)

    checked, failures = log_comparison_table(current, baseline, tol_frac, title="PDB Trajectory Regression")
    assert checked > 0, "no metrics compared"
    assert not failures, "regressions:\n" + "\n".join(failures)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)
