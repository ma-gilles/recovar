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
    PDB_REGRESSION_WRITE_BASELINE : str
        Set to "1" to (re)write baseline from current run.
    PDB_REGRESSION_REUSE_DATASET : str
        Set to "1" to reuse an existing dataset in the output dir.
    PDB_REGRESSION_RUN_ARGS : str
        Extra args for run_test_all_metrics (overrides defaults).
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers.metrics_regression import compare_metric, metric_direction

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
    _REPO_ROOT / "tests" / "baselines" / "run_test_all_metrics"
    / "pdb_5nrl_old_pipeline" / "all_scores.json"
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
    subprocess.run(cmd, check=True, env=gpu_subprocess_env())
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
    baseline_json = Path(
        os.environ.get("PDB_REGRESSION_BASELINE_JSON", str(_DEFAULT_PDB_BASELINE_JSON))
    )
    run_args = os.environ.get("PDB_REGRESSION_RUN_ARGS", _DEFAULT_RUN_ARGS)
    tol_frac = float(os.environ.get("PDB_REGRESSION_TOL_FRAC", "0.10"))
    write_baseline = os.environ.get("PDB_REGRESSION_WRITE_BASELINE", "0") == "1"

    output_dir = _resolve_output_dir(tmp_path, "pdb_current")
    reuse = (
        not write_baseline
        and os.environ.get("PDB_REGRESSION_REUSE_DATASET", "0") == "1"
        and (output_dir / "test_dataset" / "simulation_info.pkl").exists()
    )
    current = _run_pdb_metrics(output_dir, run_args, reuse_dataset=reuse)

    if write_baseline or (not baseline_json.exists()):
        baseline_json.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_json, "w") as f:
            json.dump(current, f, indent=2, sort_keys=True)
        pytest.skip(f"PDB baseline written to {baseline_json}")

    baseline = _load_json(baseline_json)

    failures = []
    checked = 0
    shared_keys = sorted(set(current.keys()) & set(baseline.keys()))
    for key in shared_keys:
        cur = current[key]
        base = baseline[key]
        if not isinstance(cur, (int, float)) or not isinstance(base, (int, float)):
            continue
        direction = metric_direction(key)
        if direction == "ignore":
            continue
        ok, msg = compare_metric(float(cur), float(base), direction, tol_frac=tol_frac)
        checked += 1
        if not ok:
            failures.append(f"{key}: current={cur} baseline={base} ({msg})")

    assert checked > 0, "no numeric metrics were checked; verify baseline/current score files"
    assert not failures, "PDB trajectory metric regressions:\n" + "\n".join(failures)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)
