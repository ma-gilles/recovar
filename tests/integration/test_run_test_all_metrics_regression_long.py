import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.metrics_regression import compare_metric, metric_direction


pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.io]


def _require_env(name):
    val = os.environ.get(name)
    if not val:
        pytest.skip(f"set {name} to run this long regression test")
    return val


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _run_metrics(output_dir, volumes_prefix, run_args):
    cmd = [
        sys.executable,
        "-m",
        "recovar.commands.run_test_all_metrics",
        "--volume-input",
        volumes_prefix,
        "--output-dir",
        str(output_dir),
    ]
    if run_args:
        cmd.extend(run_args.split())
    subprocess.run(cmd, check=True)
    score_path = output_dir / "test_dataset" / "metrics_plot" / "all_scores.json"
    assert score_path.exists(), f"missing score file at {score_path}"
    return _load_json(score_path)


def test_run_test_all_metrics_regression_against_baseline(tmp_path):
    """
    Very long regression test (typically ~1h+) for run_test_all_metrics.

    Required env:
    - LONG_METRICS_VOLUMES_DIR: prefix path to volumes (expects <prefix>0000.mrc)
    - LONG_METRICS_BASELINE_JSON: where baseline metrics are stored/read

    Optional env:
    - LONG_METRICS_RUN_ARGS: extra args for run_test_all_metrics
    - LONG_METRICS_TOL_FRAC: tolerated relative degradation (default 0.10)
    - LONG_METRICS_WRITE_BASELINE: set to 1 to (re)write baseline from current run
    """
    volumes_prefix = _require_env("LONG_METRICS_VOLUMES_DIR")
    baseline_json = Path(_require_env("LONG_METRICS_BASELINE_JSON"))
    run_args = os.environ.get("LONG_METRICS_RUN_ARGS", "--grid-size 128 --n-images 50000 --noise-level 1.0 --contrast-std 0.1")
    tol_frac = float(os.environ.get("LONG_METRICS_TOL_FRAC", "0.10"))
    write_baseline = os.environ.get("LONG_METRICS_WRITE_BASELINE", "0") == "1"

    if not Path(f"{volumes_prefix}0000.mrc").exists():
        pytest.skip(f"invalid LONG_METRICS_VOLUMES_DIR prefix: {volumes_prefix}")

    current = _run_metrics(tmp_path / "current", volumes_prefix, run_args)

    if write_baseline or (not baseline_json.exists()):
        baseline_json.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_json, "w") as f:
            json.dump(current, f, indent=2, sort_keys=True)
        pytest.skip(f"baseline written to {baseline_json}")

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
    assert not failures, "metric regressions:\n" + "\n".join(failures)
