"""Measure fixed-pose PPCA and covariance on the same synthetic SPA stack."""

import json
import subprocess
import sys
from time import perf_counter

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


def test_ppca_pipeline_metrics(tmp_path):
    env = gpu_subprocess_env()

    def run(*args, log):
        with (tmp_path / log).open("w") as stream:
            subprocess.run(
                [sys.executable, "-m", "recovar.command_line", *map(str, args)],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    run(
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
        log="generate.log",
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
        run(
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
            log=f"{method}.log",
        )
        wall_seconds = perf_counter() - started
        result = PipelineOutput(str(tmp_path / method))
        assert result.get("heterogeneity_method") == method
        z = np.asarray(result.get_embedding_component("latent_coords", 4))
        contrast = np.asarray(result.get_embedding_component("contrasts", 4)).reshape(-1)
        assert z.shape == (len(groups), 4) and contrast.shape == gt_contrast.shape
        assert np.isfinite(z).all() and np.isfinite(contrast).all()
        rng = np.random.default_rng(0)
        results[method] = {
            "state_label_r2": _label_r2(z, groups),
            "permuted_label_r2_max": max(_label_r2(z, rng.permutation(groups)) for _ in range(100)),
            "contrast_mae": float(np.mean(np.abs(contrast - gt_contrast))),
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
