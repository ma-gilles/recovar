import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers.metrics_regression import compare_metric, metric_direction, metric_tolerance, log_comparison_table
from helpers.perf_regression import perf_snapshot, stage_perf, build_perf_record, check_perf_regression


pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.gpu, pytest.mark.io, pytest.mark.long_test]

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Default in-repo baseline paths (auto-created on first run).
_DEFAULT_LONG_METRICS_BASELINE_JSON = (
    _REPO_ROOT / "tests" / "baselines" / "run_test_all_metrics" / "long_generated" / "all_scores.json"
)
_DEFAULT_LONG_METRICS_ET_BASELINE_JSON = (
    _REPO_ROOT / "tests" / "baselines" / "run_test_all_metrics" / "long_generated" / "all_scores_cryo_et.json"
)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _run_metrics(output_dir, run_args, volumes_prefix=None, reuse_dataset=False):
    """Run run_test_all_metrics; generate PDB volumes when volumes_prefix is None."""
    cmd = [
        sys.executable,
        "-m",
        "recovar.commands.run_test_all_metrics",
        "--output-dir",
        str(output_dir),
    ]
    if reuse_dataset:
        cmd += ["--reuse-dataset"]
    if volumes_prefix is not None:
        cmd += ["--volume-input", volumes_prefix]
    # Let the test itself handle regression comparison; disable the
    # subprocess's internal check so it doesn't exit(1) on its own.
    cmd += ["--skip-metrics-regression-check"]
    if run_args:
        cmd.extend(shlex.split(run_args))
    from conftest import gpu_subprocess_env

    subprocess.run(cmd, check=True, env=gpu_subprocess_env())
    score_path = output_dir / "test_dataset" / "metrics_plot" / "all_scores.json"
    assert score_path.exists(), f"missing score file at {score_path}"
    return _load_json(score_path)


def _resolve_output_dir(tmp_path: Path, name: str) -> Path:
    """
    Resolve large-output directory for long tests.

    If LONG_METRICS_OUTPUT_BASE is set, write under that base (recommended on
    /scratch or /tigress). Otherwise, fall back to pytest tmp_path.
    """
    base = os.environ.get("LONG_METRICS_OUTPUT_BASE")
    if base:
        out_dir = Path(base) / "pytest_long_metrics" / name
    else:
        out_dir = tmp_path / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _assert_cryo_et_subsampling_consistency(particles_star: Path):
    """Validate tilt/image/ntilts subsampling invariants on a real generated ET STAR."""
    from recovar.data_io import halfsets
    from recovar.data_io import image_backends as cryo_dataset

    particles_to_tilts, _ = cryo_dataset.TiltSeriesDataset.parse_particle_tilt(str(particles_star))
    n_particles = len(particles_to_tilts)
    assert n_particles > 0, "expected at least one particle group in tilt STAR"

    keep_a = 0
    keep_b = 1 if n_particles > 1 else 0

    tilt_ind_file = np.array([keep_a, keep_b, keep_a, -3, n_particles + 100], dtype=np.int32)
    candidate_images = np.concatenate(
        [
            np.asarray(particles_to_tilts[keep_a], dtype=np.int32),
            np.asarray(particles_to_tilts[keep_b], dtype=np.int32),
        ]
    )
    assert candidate_images.size > 0, "expected non-empty tilt indices for selected particles"

    ind_file = np.array(
        [
            int(candidate_images[-1]),
            int(candidate_images[0]),
            int(candidate_images[-1]),
            int(candidate_images[0]) + 100_000,  # out-of-range image id (should be ignored)
        ],
        dtype=np.int32,
    )

    split = halfsets.get_split_tilt_indices(
        particles_file=str(particles_star),
        ind_file=ind_file,
        tilt_ind_file=tilt_ind_file,
        particle_halfset_indices_file=[
            np.array([keep_b, keep_a, keep_b], dtype=np.int32),
            np.array([], dtype=np.int32),
        ],
    )
    half0 = np.asarray(split[0], dtype=np.int32)
    half1 = np.asarray(split[1], dtype=np.int32)

    assert np.intersect1d(half0, half1).size == 0
    allowed_image_values = set(candidate_images.tolist()) & set(ind_file.tolist())
    assert set(half0.tolist()).issubset(allowed_image_values)
    assert half1.size == 0

    # ntilts subsampling: at most one tilt kept per selected particle when ntilts=1.
    split_ntilts = halfsets.get_split_tilt_indices(
        particles_file=str(particles_star),
        tilt_ind_file=np.array([keep_a, keep_b], dtype=np.int32),
        ntilts=1,
        particle_halfset_indices_file=[
            np.array([keep_a, keep_b], dtype=np.int32),
            np.array([], dtype=np.int32),
        ],
    )
    kept = np.asarray(split_ntilts[0], dtype=np.int32)
    for pidx in np.unique(np.array([keep_a, keep_b], dtype=np.int32)):
        particle_tilts = np.asarray(particles_to_tilts[int(pidx)], dtype=np.int32)
        assert np.intersect1d(kept, particle_tilts).size <= 1


def test_run_test_all_metrics_regression_against_baseline(tmp_path):
    """
    Very long regression test (typically ~1h+) for run_test_all_metrics.

    Volumes are generated from PDB 5nrl trajectory so no external
    data path is required; tests can be run from any machine with a GPU.

    Optional env:
    - LONG_METRICS_VOLUMES_DIR: if set, use real volumes instead of synthetic ones
    - LONG_METRICS_BASELINE_JSON: baseline path; defaults to in-repo
      tests/baselines/run_test_all_metrics/long_generated/all_scores.json
    - LONG_METRICS_RUN_ARGS: extra args for run_test_all_metrics
    - LONG_METRICS_TOL_FRAC: tolerated relative degradation (default 0.01)
    """
    volumes_prefix = os.environ.get("LONG_METRICS_VOLUMES_DIR") or None
    if volumes_prefix and not Path(f"{volumes_prefix}0000.mrc").exists():
        pytest.skip(f"invalid LONG_METRICS_VOLUMES_DIR prefix: {volumes_prefix}")
    baseline_json = Path(os.environ.get("LONG_METRICS_BASELINE_JSON", str(_DEFAULT_LONG_METRICS_BASELINE_JSON)))
    run_args = os.environ.get(
        "LONG_METRICS_RUN_ARGS", "--grid-size 128 --n-images 50000 --noise-level 0.1 --contrast-std 0.1"
    )
    tol_frac = float(os.environ.get("LONG_METRICS_TOL_FRAC", "0.01"))

    output_dir = _resolve_output_dir(tmp_path, "current")
    reuse = (output_dir / "test_dataset" / "simulation_info.pkl").exists()

    snap_before = perf_snapshot()
    current = _run_metrics(output_dir, run_args, volumes_prefix=volumes_prefix, reuse_dataset=reuse)
    snap_after = perf_snapshot()

    # Build perf record from all_scores["perf"] if available, else from wall-clock timing
    perf_stages = current.get("perf", {})
    if not perf_stages:
        perf_stages = {"run_test_all_metrics_spa": stage_perf(snap_before, snap_after)}
    perf_record = build_perf_record(perf_stages)
    # Save current perf record for extract_regression_tables.py
    import json as _json
    _perf_out = output_dir / "current_perf_record_spa.json"
    _perf_out.parent.mkdir(parents=True, exist_ok=True)
    _perf_out.write_text(_json.dumps(perf_record, indent=2))
    perf_baseline_path = str(
        _REPO_ROOT / "tests" / "baselines" / "run_test_all_metrics" / "long_generated" / "perf_baseline_spa.json"
    )
    check_perf_regression(perf_record, perf_baseline_path, "test_run_test_all_metrics_spa")

    assert baseline_json.exists(), f"baseline not found: {baseline_json}"
    baseline = _load_json(baseline_json)

    checked, failures = log_comparison_table(current, baseline, tol_frac, title="SPA Metrics Regression")
    assert checked > 0, "no metrics compared"
    assert not failures, "regressions:\n" + "\n".join(failures)


def test_run_test_all_metrics_cryo_et_subsampling_regression_against_baseline(tmp_path):
    """
    Very long cryo-ET regression + subsampling consistency test.

    Volumes are generated synthetically so no external data path is required.
    Runs run_test_all_metrics with ET settings and validates:
    - tilt/image subsampling behavior on generated particles.star
    - numeric metric regression against a dedicated ET baseline

    Optional env:
    - LONG_METRICS_VOLUMES_DIR: if set, use real volumes instead of synthetic ones
    - LONG_METRICS_ET_BASELINE_JSON: baseline path; defaults to in-repo
      tests/baselines/run_test_all_metrics/long_generated/all_scores_cryo_et.json
    - LONG_METRICS_ET_RUN_ARGS / LONG_METRICS_ET_TOL_FRAC: see SPA test above
    """
    volumes_prefix = os.environ.get("LONG_METRICS_VOLUMES_DIR") or None
    if volumes_prefix and not Path(f"{volumes_prefix}0000.mrc").exists():
        pytest.skip(f"invalid LONG_METRICS_VOLUMES_DIR prefix: {volumes_prefix}")
    baseline_json = Path(os.environ.get("LONG_METRICS_ET_BASELINE_JSON", str(_DEFAULT_LONG_METRICS_ET_BASELINE_JSON)))
    run_args = os.environ.get(
        "LONG_METRICS_ET_RUN_ARGS",
        "--grid-size 128 --n-images 50000 --noise-level 0.1 --contrast-std 0.1 --tomo-tilts 7 --noise-model radial",
    )
    tol_frac = float(os.environ.get("LONG_METRICS_ET_TOL_FRAC", os.environ.get("LONG_METRICS_TOL_FRAC", "0.01")))

    output_dir = _resolve_output_dir(tmp_path, "current_cryo_et")
    reuse = (output_dir / "test_dataset" / "simulation_info.pkl").exists()

    snap_before = perf_snapshot()
    current = _run_metrics(output_dir, run_args, volumes_prefix=volumes_prefix, reuse_dataset=reuse)
    snap_after = perf_snapshot()

    particles_star = output_dir / "test_dataset" / "particles.star"
    assert particles_star.exists(), f"expected cryo-ET particles.star at {particles_star}"
    _assert_cryo_et_subsampling_consistency(particles_star)

    assert baseline_json.exists(), f"ET baseline not found: {baseline_json}"
    baseline = _load_json(baseline_json)

    checked, failures = log_comparison_table(current, baseline, tol_frac, title="Cryo-ET Metrics Regression")
    assert checked > 0, "no metrics compared"
    assert not failures, "regressions:\n" + "\n".join(failures)

    # Build perf record from all_scores["perf"] if available, else from wall-clock timing
    perf_stages = current.get("perf", {})
    if not perf_stages:
        perf_stages = {"run_test_all_metrics_cryo_et": stage_perf(snap_before, snap_after)}
    perf_record = build_perf_record(perf_stages)
    # Save current perf record for extract_regression_tables.py
    import json as _json
    _perf_out = output_dir / "current_perf_record_cryo_et.json"
    _perf_out.parent.mkdir(parents=True, exist_ok=True)
    _perf_out.write_text(_json.dumps(perf_record, indent=2))
    perf_baseline_path = str(
        _REPO_ROOT / "tests" / "baselines" / "run_test_all_metrics" / "long_generated" / "perf_baseline_cryo_et.json"
    )
    check_perf_regression(perf_record, perf_baseline_path, "test_run_test_all_metrics_cryo_et")
