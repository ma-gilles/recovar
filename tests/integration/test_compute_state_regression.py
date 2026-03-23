"""Regression test: compute_state quality vs committed baselines.

Generates a synthetic dataset (SPA and ET), runs the NEW pipeline + compute_state,
and compares locres/FSC metrics against committed baselines (from OLD ~/recovar code).

By default compares against stored baselines in tests/baselines/compute_state_regression/.
Set CS_REGR_REGENERATE_BASELINE=1 to re-run OLD code and overwrite baselines.

Run with:
    pytest tests/integration/test_compute_state_regression.py --long-test
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path


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
_TOL_FRAC = float(os.environ.get("CS_REGR_TOL_FRAC", "0.01"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = _REPO_ROOT / "tests" / "baselines" / "compute_state_regression"
_PERF_BASELINE_DIR = _REPO_ROOT / "tests" / "baselines" / "perf"
_OLD_RECOVAR = Path.home() / "recovar"
_REGENERATE = os.environ.get("CS_REGR_REGENERATE_BASELINE", "") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_base(tmp_path: Path, name: str) -> Path:
    base = os.environ.get("LONG_METRICS_OUTPUT_BASE")
    if base:
        out = Path(base) / "compute_state_regression" / name
    else:
        out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _gpu_env():
    from conftest import gpu_subprocess_env
    return gpu_subprocess_env()


def _run(cmd, **kwargs):
    kwargs.setdefault("env", _gpu_env())
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-80:])
        pytest.fail(
            f"Command failed (rc={result.returncode}):\n  {' '.join(str(c) for c in cmd[:6])}...\n"
            f"--- stderr (last 80 lines) ---\n{tail}"
        )
    return result


def _make_dataset(output_dir: Path, tilt_series: bool = False,
                   premultiplied_ctf: bool = False) -> Path:
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
    if premultiplied_ctf:
        cmd += ["--premultiplied-ctf"]
    _run(cmd)
    ds = output_dir / "test_dataset"
    assert ds.exists(), f"Dataset not created at {ds}"
    return ds


def _run_pipeline(dataset_dir: Path, output_dir: Path,
                   premultiplied_ctf: bool = False,
                   noise_model: str | None = None,
                   tilt_series: bool = False) -> Path:
    from recovar.simulation import synthetic_dataset
    from recovar.output import metrics
    from recovar import utils

    sim_info_path = str(dataset_dir / "simulation_info.pkl")
    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_info_path)
    volume_shape = (_GRID, _GRID, _GRID)
    gt_mask, _ = metrics.make_union_gt_mask_from_hvd(gt, volume_shape)
    mask_path = str(output_dir / "gt_union_mask.mrc")
    utils.write_mrc(mask_path, gt_mask)

    if tilt_series:
        particles = dataset_dir / "particles.star"
    else:
        particles = dataset_dir / f"particles.{_GRID}.mrcs"
    pipeline_out = output_dir / "pipeline_output"
    cmd = [
        sys.executable, "-m", "recovar.command_line", "pipeline",
        str(particles),
        "--poses", str(dataset_dir / "poses.pkl"),
        "--ctf", str(dataset_dir / "ctf.pkl"),
        "--correct-contrast",
        "-o", str(pipeline_out),
        "--mask", mask_path,
        "--lazy",
        "--zdim", str(_ZDIM),
    ]
    if tilt_series:
        cmd += ["--tilt-series"]
    if premultiplied_ctf:
        cmd += ["--premultiplied-ctf"]
    if noise_model:
        cmd += ["--noise-model", noise_model]
    _run(cmd)
    assert (pipeline_out / "model" / "params.pkl").exists()
    return pipeline_out


def _select_latent_points(pipeline_output_dir: Path, dataset_dir: Path,
                          tilt_series: bool = False) -> tuple[np.ndarray, list[int]]:
    from recovar.output import output
    from recovar.commands.run_test_all_metrics import (
        select_state_target_latent_points,
        load_unsorted_embedding_component,
    )

    sim_info_path = str(dataset_dir / "simulation_info.pkl")
    with open(sim_info_path, "rb") as f:
        sim_info = pickle.load(f)

    po = output.PipelineOutput(str(pipeline_output_dir))
    embedding_cache = {}
    unsorted_zs = load_unsorted_embedding_component(
        po, "latent_coords", _ZDIM, cache=embedding_cache
    )

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
    pts_path = output_dir / "latent_points.txt"
    if not pts_path.exists():
        np.savetxt(str(pts_path), latent_points)

    state_out = output_dir / "old_compute_state"

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


def _extract_volumes_real_space(state_dir: Path, n_vols: int,
                                style: str = "new") -> list[np.ndarray]:
    from recovar import utils

    vols = []
    for i in range(n_vols):
        if style == "new":
            vol_path = state_dir / f"state{i:03d}.mrc"
        else:
            vol_path = state_dir / "all_volumes" / f"vol{i:04d}.mrc"
            if not vol_path.exists():
                vol_path = state_dir / f"vol{i:04d}" / "locres_filtered.mrc"

        if not vol_path.exists():
            vols.append(None)
            continue

        vols.append(utils.load_mrc(str(vol_path)))
    return vols


def _compute_gt_volume_metrics(reconstructed_vols: list[np.ndarray | None],
                               gt_thing,
                               volume_shape: tuple,
                               voxel_size: float,
                               state_labels: list[int]) -> dict:
    from recovar.core.fourier_transform_utils import get_idft3
    from recovar.output import metrics as output_metrics

    _, moving_mask = output_metrics.make_moving_gt_mask_from_hvd(gt_thing, volume_shape)

    metrics = {}
    for i, recon_vol in enumerate(reconstructed_vols):
        if recon_vol is None:
            continue
        gt_label = int(state_labels[i])
        gt_map = get_idft3(gt_thing.volumes[gt_label].reshape(volume_shape)).real
        gt_metrics = output_metrics.compute_volume_error_metrics_from_gt(
            gt_map,
            recon_vol,
            voxel_size,
            mask=None,
            partial_mask=moving_mask,
            normalize_by_map1=True,
        )
        metrics[f"state_{i}_locres_median"] = float(gt_metrics["median_locres"])
        metrics[f"state_{i}_locres_90pct"] = float(gt_metrics["ninety_pc_locres"])
        metrics[f"state_{i}_moving_piece_locres_median"] = float(gt_metrics["partial_median_locres"])
        metrics[f"state_{i}_moving_piece_locres_90pct"] = float(gt_metrics["partial_ninety_pc_locres"])
        metrics[f"state_{i}_moving_piece_error_median"] = float(gt_metrics["partial_median_error"])
        metrics[f"state_{i}_moving_piece_error_90pct"] = float(gt_metrics["partial_ninety_pc_error"])
    return metrics


def _load_baseline(mode: str) -> dict:
    path = _BASELINE_DIR / f"{mode}.json"
    if not path.exists():
        pytest.fail(
            f"Baseline missing: {path}\n"
            f"Generate with: CS_REGR_REGENERATE_BASELINE=1 pytest ... --long-test"
        )
    with open(path) as f:
        return json.load(f)


def _save_baseline(mode: str, data: dict):
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_BASELINE_DIR / f"{mode}.json", "w") as f:
        json.dump(data, f, indent=2, default=float)


def _run_and_collect_metrics(tmp_path: Path, tilt_series: bool = False,
                              premultiplied_ctf: bool = False,
                              noise_model: str | None = None,
                              mode_override: str | None = None,
                              pipeline_tilt_series: bool = False) -> dict:
    """Generate dataset, run pipeline + NEW compute_state, return metrics.

    ``pipeline_tilt_series`` loads particles.star with ``--tilt-series``
    (needed for radial_per_tilt which requires dose indices).
    """
    mode = mode_override or ("et" if tilt_series else "spa")
    out_base = _output_base(tmp_path, mode)

    perf_stages = {}

    snap0 = perf_snapshot()
    dataset_dir = _make_dataset(
        out_base / "data", tilt_series=tilt_series,
        premultiplied_ctf=premultiplied_ctf,
    )
    perf_stages["make_dataset"] = stage_perf(snap0, perf_snapshot())

    snap1 = perf_snapshot()
    pipeline_dir = _run_pipeline(
        dataset_dir, out_base,
        premultiplied_ctf=premultiplied_ctf,
        noise_model=noise_model,
        tilt_series=pipeline_tilt_series,
    )
    perf_stages["pipeline"] = stage_perf(snap1, perf_snapshot())

    latent_points, labels = _select_latent_points(
        pipeline_dir, dataset_dir, tilt_series=tilt_series
    )
    n_vols = latent_points.shape[0]
    print(f"\n[{mode.upper()}] Selected {n_vols} latent points for labels {labels}")

    snap2 = perf_snapshot()
    new_state_dir = _run_new_compute_state(pipeline_dir, latent_points, out_base)
    perf_stages["compute_state"] = stage_perf(snap2, perf_snapshot())

    sim_info_path = str(dataset_dir / "simulation_info.pkl")
    from recovar.simulation import synthetic_dataset
    gt = synthetic_dataset.load_heterogeneous_reconstruction(sim_info_path)

    from recovar.output import output
    po = output.PipelineOutput(str(pipeline_dir))
    volume_shape = po.get("volume_shape")
    voxel_size = po.get("voxel_size")

    new_vols = _extract_volumes_real_space(new_state_dir, n_vols, style="new")
    current = _compute_gt_volume_metrics(new_vols, gt, volume_shape, voxel_size, labels)

    # If regenerating, also run OLD code to produce new baselines
    if _REGENERATE and _OLD_RECOVAR.is_dir():
        converted_dir = _convert_for_old_code(pipeline_dir, out_base)
        old_state_dir = _run_old_compute_state(converted_dir, latent_points, out_base)
        old_vols = _extract_volumes_real_space(old_state_dir, n_vols, style="old")
        baseline = _compute_gt_volume_metrics(old_vols, gt, volume_shape, voxel_size, labels)
        _save_baseline(mode, baseline)
        print(f"Saved NEW baseline to {_BASELINE_DIR / f'{mode}.json'}")

    # Performance tracking
    perf_record = build_perf_record(perf_stages)
    perf_path = str(
        _PERF_BASELINE_DIR / f"perf_compute_state_regression_{mode}.json"
    )
    check_perf_regression(perf_record, perf_path,
                          test_name=f"compute_state_regression_{mode}")

    return current


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compute_state_regression_spa(tmp_path):
    """Check compute_state quality against committed baseline (SPA)."""
    current = _run_and_collect_metrics(tmp_path, tilt_series=False)
    baseline = _load_baseline("spa")

    checked, failures = log_comparison_table(
        current, baseline, _TOL_FRAC,
        title="compute_state regression: SPA",
    )

    print("\n--- SPA Raw Metrics ---")
    print(f"NEW: {json.dumps(current, indent=2, default=float)}")
    print(f"BASELINE: {json.dumps(baseline, indent=2, default=float)}")

    if checked > 0:
        assert not failures, (
            f"compute_state SPA regressed vs baseline:\n" + "\n".join(failures)
        )


def test_compute_state_regression_et(tmp_path):
    """Check compute_state quality against committed baseline (ET)."""
    current = _run_and_collect_metrics(tmp_path, tilt_series=True)
    baseline = _load_baseline("et")

    checked, failures = log_comparison_table(
        current, baseline, _TOL_FRAC,
        title="compute_state regression: ET",
    )

    print("\n--- ET Raw Metrics ---")
    print(f"NEW: {json.dumps(current, indent=2, default=float)}")
    print(f"BASELINE: {json.dumps(baseline, indent=2, default=float)}")

    if checked > 0:
        assert not failures, (
            f"compute_state ET regressed vs baseline:\n" + "\n".join(failures)
        )


def test_compute_state_regression_et_premultiplied_radial_per_tilt(tmp_path):
    """Check compute_state quality with premultiplied-CTF + radial_per_tilt.

    Tests the most common real cryo-ET configuration (Warp/M output) where
    images arrive pre-multiplied by their CTF.  This exercises:
    - ``skip_ctf=True`` path in the heterogeneity kernel
    - per-tilt noise estimation (``radial_per_tilt``)
    - contrast estimation under premultiplied-CTF

    Compares against a dedicated baseline; regenerate with
    ``CS_REGR_REGENERATE_BASELINE=1``.
    """
    current = _run_and_collect_metrics(
        tmp_path,
        tilt_series=True,
        premultiplied_ctf=True,
        noise_model="radial_per_tilt",
        mode_override="et_premultiplied",
        pipeline_tilt_series=True,
    )
    baseline = _load_baseline("et_premultiplied")

    checked, failures = log_comparison_table(
        current, baseline, _TOL_FRAC,
        title="compute_state regression: ET premultiplied + radial_per_tilt",
    )

    print("\n--- ET Premultiplied Raw Metrics ---")
    print(f"NEW: {json.dumps(current, indent=2, default=float)}")
    print(f"BASELINE: {json.dumps(baseline, indent=2, default=float)}")

    if checked > 0:
        assert not failures, (
            f"compute_state ET premultiplied regressed vs baseline:\n"
            + "\n".join(failures)
        )
