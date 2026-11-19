"""Measure fixed-pose PPCA and covariance on the same synthetic SPA stack."""

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import mrcfile
import numpy as np
import pytest
from conftest import gpu_subprocess_env

from recovar.output.output import PipelineOutput
from recovar.utils.helpers import pickle_load

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]


def _label_r2(embedding, groups):
    centered = embedding - embedding.mean(axis=0)
    total = np.sum(centered**2)
    assert total > 0, "The latent embedding collapsed"
    counts = np.bincount(groups)
    sums = np.zeros((counts.size, centered.shape[1]), dtype=centered.dtype)
    np.add.at(sums, groups, centered)
    return float(np.sum(sums**2 / counts[:, None]) / total)


def _run(*args, log, env):
    with log.open("w") as stream:
        subprocess.run(
            [sys.executable, "-m", "recovar.command_line", *map(str, args)],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _embedding_scores(result, zdim, groups, gt_contrast):
    z = np.asarray(result.get_embedding_component("latent_coords", zdim))
    contrast = np.asarray(result.get_embedding_component("contrasts", zdim)).reshape(-1)
    assert z.shape == (len(groups), zdim) and contrast.shape == gt_contrast.shape
    assert np.isfinite(z).all() and np.isfinite(contrast).all()
    rng = np.random.default_rng(0)
    return {
        "state_label_r2": _label_r2(z, groups),
        "permuted_label_r2_max": max(_label_r2(z, rng.permutation(groups)) for _ in range(100)),
        "contrast_mae": float(np.mean(np.abs(contrast - gt_contrast))),
    }


def test_ppca_pipeline_metrics(tmp_path):
    env = gpu_subprocess_env()

    _run(
        "make_test_dataset",
        tmp_path / "input",
        "--n-images",
        2000,
        "--grid-size",
        64,
        "--noise-level",
        0.05,
        "--seed",
        42,
        log=tmp_path / "generate.log",
        env=env,
    )
    dataset = tmp_path / "input" / "test_dataset"
    truth = pickle_load(dataset / "simulation_info.pkl")
    labels, groups = np.unique(truth["image_assignment"], return_inverse=True)
    gt_contrast = np.asarray(truth["per_image_contrast"]).reshape(-1)
    assert len(labels) > 1 and np.all(labels >= 0)
    assert groups.shape == gt_contrast.shape == (2000,)
    results = {}
    for method, extra in (
        ("covariance", []),
        ("ppca", ["--use-ppca", "--ppca-zdim", "4", "--ppca-em-iters", "8"]),
    ):
        started = perf_counter()
        _run(
            "pipeline",
            dataset / "particles.64.mrcs",
            "--poses",
            dataset / "poses.pkl",
            "--ctf",
            dataset / "ctf.pkl",
            "-o",
            tmp_path / method,
            "--mask",
            "sphere",
            "--correct-contrast",
            "--lazy",
            "--zdim",
            4,
            *extra,
            log=tmp_path / f"{method}.log",
            env=env,
        )
        wall_seconds = perf_counter() - started
        result = PipelineOutput(str(tmp_path / method))
        assert result.get("heterogeneity_method") == method
        results[method] = {
            **_embedding_scores(result, 4, groups, gt_contrast),
            "wall_seconds": wall_seconds,
        }

    (tmp_path / "ppca_metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print("| Metric | Covariance | PPCA | Change |", file=sys.stderr)
    print("|---|---:|---:|---:|", file=sys.stderr)
    for metric, covariance in results["covariance"].items():
        ppca = results["ppca"][metric]
        change = f"{100 * (ppca - covariance) / covariance:+.2f}%" if covariance else "n/a"
        print(f"| {metric} | {covariance:.6g} | {ppca:.6g} | {change} |", file=sys.stderr)
    for method, metrics in results.items():
        assert metrics["state_label_r2"] > metrics["permuted_label_r2_max"], method


@pytest.mark.long_test
@pytest.mark.io
def test_ppca_pipeline_full_resolution(tmp_path):
    """Fit rank-10 PPCA for 20 iterations on the standard long SPA fixture."""
    from recovar.output import metrics

    env = gpu_subprocess_env()
    output_base = Path(os.environ.get("LONG_METRICS_OUTPUT_BASE", tmp_path))
    run_dir = output_base / "pytest_long_metrics" / "ppca_spa"
    run_dir.mkdir(parents=True, exist_ok=True)
    reuse = os.environ.get("LONG_PPCA_DATASET")
    if reuse:
        dataset = Path(reuse).resolve()
    else:
        _run(
            "run_test_all_metrics",
            "--dataset-only",
            "--output-dir",
            run_dir,
            "--grid-size",
            128,
            "--n-images",
            50_000,
            "--noise-level",
            0.1,
            "--contrast-std",
            0.1,
            log=run_dir / "generate.log",
            env=env,
        )
        dataset = run_dir / "test_dataset"

    stack = dataset / "particles.128.mrcs"
    with mrcfile.open(stack, header_only=True) as mrc:
        assert tuple(int(getattr(mrc.header, axis)) for axis in ("nz", "ny", "nx")) == (50_000, 128, 128)
    truth = pickle_load(dataset / "simulation_info.pkl")
    labels, groups = np.unique(truth["image_assignment"], return_inverse=True)
    gt_contrast = np.asarray(truth["per_image_contrast"]).reshape(-1)
    assert len(labels) > 1 and np.all(labels >= 0)
    assert groups.shape == gt_contrast.shape == (50_000,)

    zdim = 10
    ppca = ["--use-ppca", "--ppca-zdim", str(zdim), "--ppca-em-iters", "20"]
    results = {}
    for name, options, box in (
        ("covariance", [], 128),
        ("ppca_full", [*ppca, "--ppca-downsample-resolution", "off"], 128),
        ("ppca_downsampled", [*ppca, "--ppca-downsample-resolution", "12"], 94),
    ):
        started = perf_counter()
        _run(
            "pipeline",
            stack,
            "--poses",
            dataset / "poses.pkl",
            "--ctf",
            dataset / "ctf.pkl",
            "--mask",
            dataset / "gt_masks" / "gt_union_mask.mrc",
            "--noise-model",
            "radial",
            "--correct-contrast",
            "--lazy",
            "--zdim",
            zdim,
            "-o",
            run_dir / name,
            *options,
            log=run_dir / f"{name}.log",
            env=env,
        )
        wall_seconds = perf_counter() - started
        result = PipelineOutput(str(run_dir / name))
        assert tuple(result.get("volume_shape")) == (128,) * 3
        assert result.get("heterogeneity_method") == ("ppca" if options else "covariance")
        if options:
            info = result.get("ppca_info")
            assert info["em_iters"] == 20
            assert info["downsample"]["fitting_box_size"] == box
            loadings = np.load(run_dir / name / "model" / "ppca_loadings.npy", mmap_mode="r")
            assert loadings.shape == (128 * 128 * 65, zdim)
            assert np.isfinite(loadings).all()

        z = result.get_unsorted_embedding_component("latent_coords", zdim)
        _, embedding_error = metrics.variance_of_zs(z, groups)
        results[name] = {
            **_embedding_scores(result, zdim, groups, gt_contrast),
            "embedding_squared_error_10": float(embedding_error),
            "fitting_box_size": box,
            "wall_seconds": wall_seconds,
        }

    (run_dir / "ppca_metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print("| Metric | Covariance | PPCA full | PPCA downsampled |", file=sys.stderr)
    print("|---|---:|---:|---:|", file=sys.stderr)
    for key in results["covariance"]:
        values = " | ".join(f"{scores[key]:.6g}" for scores in results.values())
        print(f"| {key} | {values} |", file=sys.stderr)
    for method, scores in results.items():
        assert scores["state_label_r2"] > scores["permuted_label_r2_max"], method
