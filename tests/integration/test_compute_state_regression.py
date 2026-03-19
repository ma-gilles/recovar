"""Regression test: OLD vs NEW compute_state on identical inputs.

Generates a synthetic dataset (SPA and ET), runs the NEW pipeline, then runs
both OLD (~/recovar) and NEW compute_state on the same pipeline output and
latent points.  Compares local resolution and FSC-vs-GT metrics.

Run with:
    pytest tests/integration/test_compute_state_regression.py --long-test

Requires:
    - ~/recovar (old recovar code)
    - GPU (submitted via Slurm in practice)
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from helpers.metrics_regression import log_comparison_table
from helpers.perf_regression import (
    build_perf_record,
    check_perf_regression,
    perf_snapshot,
    stage_perf,
)

pytestmark = [
    pytest.mark.long_test,
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.slow,
]

_GRID = int(os.environ.get("CS_REGR_GRID", "128"))
_N_IMAGES_SPA = int(os.environ.get("CS_REGR_N_IMAGES_SPA", "20000"))
_N_IMAGES_ET = int(os.environ.get("CS_REGR_N_IMAGES_ET", "20000"))
_N_TILTS = int(os.environ.get("CS_REGR_N_TILTS", "7"))
_ZDIM = 4
_NOISE = 0.1
_SEED = 42
_TOL_FRAC = float(os.environ.get("CS_REGR_TOL_FRAC", "0.05"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = _REPO_ROOT / "tests" / "baselines" / "compute_state_regression"
_PERF_BASELINE_DIR = _REPO_ROOT / "tests" / "baselines" / "perf"
_OLD_RECOVAR = Path.home() / "recovar"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_base(tmp_path: Path, name: str) -> Path:
    """Use LONG_METRICS_OUTPUT_BASE if set, else tmp_path."""
    base = os.environ.get("LONG_METRICS_OUTPUT_BASE")
    if base:
        out = Path(base) / "compute_state_regression" / name
    else:
        out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _gpu_env():
    """Environment for GPU subprocesses."""
    from conftest import gpu_subprocess_env
    return gpu_subprocess_env()


def _run(cmd, **kwargs):
    """Run a subprocess, fail on error with informative output."""
    kwargs.setdefault("env", _gpu_env())
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-80:])
        pytest.fail(
            f"Command failed (rc={result.returncode}):\n  {' '.join(str(c) for c in cmd[:6])}...\n"
            f"--- stderr (last 80 lines) ---\n{tail}"
        )
    return result


def _make_dataset(output_dir: Path, tilt_series: bool = False) -> Path:
    """Generate synthetic dataset using NEW code."""
    cmd = [
        sys.executable, "-m", "recovar.commands.make_test_dataset",
        str(output_dir),
        "--grid-size", str(_GRID),
        "--n-images", str(_N_IMAGES_ET if tilt_series else _N_IMAGES_SPA),
        "--noise-level", str(_NOISE),
        "--seed", str(_SEED),
    ]
    if tilt_series:
        cmd += ["--tilt-series", "--n-tilts", str(_N_TILTS)]
    _run(cmd)
    ds = output_dir / "test_dataset"
    assert ds.exists(), f"Dataset not created at {ds}"
    return ds


def _run_pipeline(dataset_dir: Path, output_dir: Path) -> Path:
    """Run NEW pipeline on dataset."""
    from recovar.simulation import synthetic_dataset
    from recovar.output import metrics
    from recovar import utils

    # Compute GT mask
    sim_info_path = str(dataset_dir / "simulation_info.pkl")
    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_info_path)
    volume_shape = (_GRID, _GRID, _GRID)
    gt_mask, _ = metrics.make_union_gt_mask_from_hvd(gt, volume_shape)
    mask_path = str(output_dir / "gt_union_mask.mrc")
    utils.write_mrc(mask_path, gt_mask)

    mrcs = dataset_dir / f"particles.{_GRID}.mrcs"
    pipeline_out = output_dir / "pipeline_output"
    cmd = [
        sys.executable, "-m", "recovar.command_line", "pipeline",
        str(mrcs),
        "--poses", str(dataset_dir / "poses.pkl"),
        "--ctf", str(dataset_dir / "ctf.pkl"),
        "--correct-contrast",
        "-o", str(pipeline_out),
        "--mask", mask_path,
        "--lazy",
        "--zdim", str(_ZDIM),
    ]
    _run(cmd)
    assert (pipeline_out / "model" / "params.pkl").exists()
    return pipeline_out


def _select_latent_points(pipeline_output_dir: Path, dataset_dir: Path,
                          tilt_series: bool = False) -> tuple[np.ndarray, list[int]]:
    """Select representative latent points from GT labels."""
    from recovar.output import output
    from recovar.commands.run_test_all_metrics import (
        select_state_target_latent_points,
        load_unsorted_embedding_component,
    )

    sim_info_path = str(dataset_dir / "simulation_info.pkl")
    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)

    po = output.PipelineOutput(str(pipeline_output_dir))
    legacy_cache = {}
    unsorted_zs = load_unsorted_embedding_component(
        po, "latent_coords", _ZDIM, legacy_cache=legacy_cache
    )

    # For ET: latent_coords may be per-particle or per-image depending on
    # pipeline version.  Use tilt_series_assignment if sizes match, else
    # fall back to image_assignment (which is always per-image).
    n_zs = unsorted_zs.shape[0]
    if tilt_series:
        ts_assign = np.asarray(sim_info["tilt_series_assignment"]).ravel()
        img_assign = np.asarray(sim_info["image_assignment"]).ravel()
        if ts_assign.size == n_zs:
            assignment = ts_assign
        else:
            assignment = img_assign
    else:
        assignment = np.asarray(sim_info["image_assignment"]).ravel()

    max_classes = int(np.max(sim_info["image_assignment"])) + 1
    requested_labels = [0, max_classes // 2]

    points, labels = select_state_target_latent_points(
        unsorted_zs=unsorted_zs,
        particle_assignment=assignment,
        preferred_labels=requested_labels,
        max_points=2,
    )
    return points, labels


def _run_new_compute_state(pipeline_output_dir: Path, latent_points: np.ndarray,
                           output_dir: Path) -> Path:
    """Run NEW compute_state."""
    pts_path = output_dir / "latent_points.txt"
    np.savetxt(str(pts_path), latent_points)

    state_out = output_dir / "new_compute_state"
    cmd = [
        sys.executable, "-m", "recovar.command_line", "compute_state",
        str(pipeline_output_dir),
        "-o", str(state_out),
        "--latent-points", str(pts_path),
        "--lazy",
    ]
    _run(cmd)
    return state_out


def _convert_for_old_code(pipeline_output_dir: Path, output_dir: Path) -> Path:
    """Convert pipeline output to OLD format."""
    converted = output_dir / "converted_pipeline_output"
    convert_script = _REPO_ROOT / "scripts" / "convert_pipeline_output_for_old_code.py"
    cmd = [
        sys.executable, str(convert_script),
        str(pipeline_output_dir),
        str(converted),
    ]
    _run(cmd)
    assert converted.exists()
    return converted


def _run_old_compute_state(converted_pipeline_dir: Path, latent_points: np.ndarray,
                           output_dir: Path) -> Path:
    """Run OLD compute_state as subprocess with ~/recovar on sys.path."""
    pts_path = output_dir / "latent_points.txt"
    if not pts_path.exists():
        np.savetxt(str(pts_path), latent_points)

    state_out = output_dir / "old_compute_state"

    # Write a small runner script to avoid module conflicts
    runner_script = output_dir / "_run_old_compute_state.py"
    runner_script.write_text(textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, {str(_OLD_RECOVAR)!r})

        import argparse
        from recovar.commands import compute_state

        parser = argparse.ArgumentParser()
        parser = compute_state.add_args(parser)
        args = parser.parse_args()
        compute_state.compute_state(args)
    """))

    cmd = [
        sys.executable, str(runner_script),
        str(converted_pipeline_dir),
        "-o", str(state_out),
        "--latent-points", str(pts_path),
        "--lazy",
    ]
    _run(cmd)
    return state_out


def _extract_locres_metrics(state_dir: Path, prefix: str, n_vols: int,
                            style: str = "new") -> dict:
    """Extract local resolution metrics from compute_state output.

    Parameters
    ----------
    style : "new" or "old"
        "new" → diagnostics/state{i:03d}/local_resolution.mrc
        "old" → vol{i:04d}/locres.mrc  (or all_volumes/locres{i:04d}.mrc)
    """
    metrics = {}
    for i in range(n_vols):
        if style == "new":
            locres_path = state_dir / "diagnostics" / f"state{i:03d}" / "local_resolution.mrc"
        else:
            # OLD: try vol{i:04d}/locres.mrc first, then all_volumes/locres{i:04d}.mrc
            locres_path = state_dir / f"vol{i:04d}" / "locres.mrc"
            if not locres_path.exists():
                locres_path = state_dir / "all_volumes" / f"locres{i:04d}.mrc"

        if not locres_path.exists():
            print(f"WARNING: locres not found at {locres_path}")
            continue

        with mrcfile.open(str(locres_path), mode="r") as mrc:
            locres_map = np.array(mrc.data, dtype=np.float32)

        valid = locres_map[np.isfinite(locres_map) & (locres_map > 0)]
        if valid.size == 0:
            continue

        metrics[f"{prefix}_state_{i}_locres_median"] = float(np.median(valid))
        metrics[f"{prefix}_state_{i}_locres_90pct"] = float(np.percentile(valid, 90))

    return metrics


def _extract_volumes_real_space(state_dir: Path, n_vols: int,
                                style: str = "new") -> list[np.ndarray]:
    """Load reconstructed filtered volumes in real space."""
    from recovar import utils

    vols = []
    for i in range(n_vols):
        if style == "new":
            vol_path = state_dir / f"state{i:03d}.mrc"
        else:
            # OLD: all_volumes/vol{i:04d}.mrc or vol{i:04d}/locres_filtered.mrc
            vol_path = state_dir / "all_volumes" / f"vol{i:04d}.mrc"
            if not vol_path.exists():
                vol_path = state_dir / f"vol{i:04d}" / "locres_filtered.mrc"

        if not vol_path.exists():
            print(f"WARNING: volume not found at {vol_path}")
            vols.append(None)
            continue

        vols.append(utils.load_mrc(str(vol_path)))
    return vols


def _compute_fsc_vs_gt(reconstructed_vols: list[np.ndarray | None],
                        gt_volumes_fourier: np.ndarray,
                        volume_shape: tuple, voxel_size: float,
                        prefix: str) -> dict:
    """Compute FSC AUC of each reconstructed volume vs closest GT volume.

    GT volumes are in Fourier space (shape: n_gt_vols x vol_size).
    Reconstructed volumes are in real space (shape: volume_shape).
    """
    from recovar.core.fourier_transform_utils import get_dft3
    from recovar.output.plot_utils import FSC, fsc_score

    metrics = {}
    grid_size = volume_shape[0]
    n_gt = gt_volumes_fourier.shape[0]

    for i, recon_vol in enumerate(reconstructed_vols):
        if recon_vol is None:
            continue

        # Convert reconstructed volume to Fourier space
        recon_ft = get_dft3(recon_vol).reshape(-1)

        # Find best-matching GT volume by FSC AUC
        best_auc = -1.0
        best_gt_idx = 0
        for g in range(n_gt):
            gt_vol_ft = gt_volumes_fourier[g].reshape(volume_shape)
            recon_vol_ft_3d = recon_ft.reshape(volume_shape)
            fsc_curve = FSC(np.array(gt_vol_ft), np.array(recon_vol_ft_3d))
            # AUC: average FSC over shells (excluding DC)
            auc = float(np.mean(fsc_curve[1:grid_size // 2]))
            if auc > best_auc:
                best_auc = auc
                best_gt_idx = g

        metrics[f"{prefix}_state_{i}_fsc_auc_vs_gt"] = best_auc
        metrics[f"{prefix}_state_{i}_best_gt_idx"] = best_gt_idx

    return metrics


def _run_full_comparison(tmp_path: Path, tilt_series: bool = False) -> dict:
    """Run the full OLD vs NEW comparison for a single mode (SPA or ET).

    Returns combined metrics dict.
    """
    mode = "et" if tilt_series else "spa"
    out_base = _output_base(tmp_path, mode)

    perf_stages = {}

    # 1. Generate dataset
    snap0 = perf_snapshot()
    dataset_dir = _make_dataset(out_base / "data", tilt_series=tilt_series)
    perf_stages["make_dataset"] = stage_perf(snap0, perf_snapshot())

    # 2. Run pipeline
    snap1 = perf_snapshot()
    pipeline_dir = _run_pipeline(dataset_dir, out_base)
    perf_stages["pipeline"] = stage_perf(snap1, perf_snapshot())

    # 3. Select latent points
    latent_points, labels = _select_latent_points(
        pipeline_dir, dataset_dir, tilt_series=tilt_series
    )
    n_vols = latent_points.shape[0]
    print(f"\n[{mode.upper()}] Selected {n_vols} latent points for labels {labels}")

    # 4. Run NEW compute_state
    snap2 = perf_snapshot()
    new_state_dir = _run_new_compute_state(pipeline_dir, latent_points, out_base)
    perf_stages["new_compute_state"] = stage_perf(snap2, perf_snapshot())

    # 5. Convert pipeline output for OLD code
    converted_dir = _convert_for_old_code(pipeline_dir, out_base)

    # 6. Run OLD compute_state
    if not _OLD_RECOVAR.is_dir():
        pytest.skip(f"Old recovar not found at {_OLD_RECOVAR}")

    snap3 = perf_snapshot()
    old_state_dir = _run_old_compute_state(converted_dir, latent_points, out_base)
    perf_stages["old_compute_state"] = stage_perf(snap3, perf_snapshot())

    # 7. Extract metrics
    new_metrics = _extract_locres_metrics(new_state_dir, "new", n_vols, style="new")
    old_metrics = _extract_locres_metrics(old_state_dir, "old", n_vols, style="old")

    # 8. Load GT volumes for FSC comparison
    sim_info_path = str(dataset_dir / "simulation_info.pkl")
    from recovar.simulation import synthetic_dataset
    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_info_path)
    gt_vols_ft = gt.volumes  # (n_vols, vol_size), Fourier space

    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)

    # Get voxel_size from pipeline output params
    from recovar.output import output
    po = output.PipelineOutput(str(pipeline_dir))
    volume_shape = po.get("volume_shape")
    voxel_size = po.get("voxel_size")

    # Compute FSC vs GT for both OLD and NEW
    new_vols = _extract_volumes_real_space(new_state_dir, n_vols, style="new")
    old_vols = _extract_volumes_real_space(old_state_dir, n_vols, style="old")

    new_fsc = _compute_fsc_vs_gt(new_vols, gt_vols_ft, volume_shape, voxel_size, "new")
    old_fsc = _compute_fsc_vs_gt(old_vols, gt_vols_ft, volume_shape, voxel_size, "old")

    # 9. Build comparison table
    # Rename metrics so OLD and NEW share the same keys for comparison
    comparison_new = {}
    comparison_old = {}
    for key, val in new_metrics.items():
        clean_key = key.replace("new_", "", 1)
        comparison_new[clean_key] = val
    for key, val in old_metrics.items():
        clean_key = key.replace("old_", "", 1)
        comparison_old[clean_key] = val
    for key, val in new_fsc.items():
        clean_key = key.replace("new_", "", 1)
        comparison_new[clean_key] = val
    for key, val in old_fsc.items():
        clean_key = key.replace("old_", "", 1)
        comparison_old[clean_key] = val

    # 10. Performance tracking
    perf_record = build_perf_record(perf_stages)
    perf_path = str(
        _PERF_BASELINE_DIR / f"perf_compute_state_regression_{mode}.json"
    )
    check_perf_regression(perf_record, perf_path,
                          test_name=f"compute_state_regression_{mode}")

    return {
        "new": comparison_new,
        "old": comparison_old,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compute_state_regression_spa(tmp_path):
    """Compare OLD vs NEW compute_state on SPA synthetic data."""
    result = _run_full_comparison(tmp_path, tilt_series=False)

    checked, failures = log_comparison_table(
        result["new"], result["old"], _TOL_FRAC,
        title="compute_state regression: SPA (NEW vs OLD baseline)",
    )

    # Print raw metrics for reference
    print("\n--- SPA Raw Metrics ---")
    print(f"NEW: {json.dumps(result['new'], indent=2, default=float)}")
    print(f"OLD: {json.dumps(result['old'], indent=2, default=float)}")

    if checked > 0:
        assert not failures, (
            f"NEW compute_state regressed vs OLD on SPA:\n" + "\n".join(failures)
        )


def test_compute_state_regression_et(tmp_path):
    """Compare OLD vs NEW compute_state on ET synthetic data."""
    result = _run_full_comparison(tmp_path, tilt_series=True)

    checked, failures = log_comparison_table(
        result["new"], result["old"], _TOL_FRAC,
        title="compute_state regression: ET (NEW vs OLD baseline)",
    )

    print("\n--- ET Raw Metrics ---")
    print(f"NEW: {json.dumps(result['new'], indent=2, default=float)}")
    print(f"OLD: {json.dumps(result['old'], indent=2, default=float)}")

    if checked > 0:
        assert not failures, (
            f"NEW compute_state regressed vs OLD on ET:\n" + "\n".join(failures)
        )
