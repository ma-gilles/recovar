"""Tests for the bounded K=4 replay repeatability audit."""

from __future__ import annotations

import json
from pathlib import Path

import mrcfile
import numpy as np

from scripts.audit_em_k4_replay_repeatability import build_report


def _write_replay(root: Path, *, array_delta: float = 0.0) -> None:
    output = root / "output"
    output.mkdir(parents=True)
    summary = {
        "n_classes": 4,
        "n_images": 8,
        "current_size": 12,
        "healpix_order": 1,
        "n_rotations": 24,
        "n_translations": 5,
        "random_perturbation": -0.125,
        "coarse_rotation_source": "relion_cuda_make_eulers_3d",
        "image_fourier_backend": "relion_cuda",
    }
    (output / "summary.json").write_text(json.dumps(summary))
    np.savez(
        output / "k_class_parity_arrays.npz",
        assignments=np.asarray([0.0 + array_delta, 1.0], dtype=np.float32),
        weights=np.asarray([0.25] * 4, dtype=np.float32),
    )
    rng = np.random.default_rng(20260902)
    for prefix in ("recovar_class", "recovar_best_variant_class"):
        for class_one_based in range(1, 5):
            values = rng.standard_normal((8, 8, 8)).astype(np.float32)
            with mrcfile.new(
                output / f"{prefix}{class_one_based:03d}.mrc", overwrite=True
            ) as stream:
                stream.set_data(values)


def test_repeatability_accepts_exact_state_and_maps(tmp_path: Path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_replay(baseline)
    _write_replay(candidate)

    report = build_report(baseline, candidate)

    assert report["status"] == "pass"
    assert report["configuration_exact"] is True
    assert report["parity_arrays_bitwise"] is True
    assert report["summary"]["map_count"] == 8
    assert report["summary"]["map_bitwise_count"] == 8
    assert report["summary"]["map_fsc_auc_minimum"] >= 1.0 - 1e-14


def test_repeatability_fails_on_parity_array_drift(tmp_path: Path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_replay(baseline)
    _write_replay(candidate, array_delta=1.0)

    report = build_report(baseline, candidate)

    assert report["status"] == "fail"
    assert report["parity_arrays_bitwise"] is False
    assert report["parity_array_results"]["assignments"] is False
