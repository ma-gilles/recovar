from __future__ import annotations

import json

import numpy as np

from recovar.utils import helpers
from scripts.evaluate_ab_initio_gt import main


def _asymmetric_test_volume(n: int = 17) -> np.ndarray:
    grid = np.indices((n, n, n), dtype=np.float64)
    z, y, x = [(axis - (n - 1) / 2.0) / n for axis in grid]
    first = np.exp(-90.0 * ((x - 0.12) ** 2 + (y + 0.07) ** 2 + (z - 0.03) ** 2))
    second = 0.35 * np.exp(-130.0 * ((x + 0.18) ** 2 + (y - 0.11) ** 2 + (z + 0.09) ** 2))
    ridge = 0.08 * x + 0.03 * y - 0.05 * z
    return first + second + ridge


def test_evaluate_ab_initio_gt_cli_handles_relion_frame_outputs(tmp_path):
    gt = _asymmetric_test_volume()
    gt_path = tmp_path / "reference_gt.mrc"
    native_path = tmp_path / "run_it003_class001.mrc"
    out_npz = tmp_path / "metrics.npz"
    out_json = tmp_path / "metrics.json"

    helpers.write_mrc(str(gt_path), gt, voxel_size=2.5)
    helpers.write_relion_mrc(str(native_path), gt, voxel_size=2.5)

    rc = main(
        [
            "--volume",
            str(native_path),
            "--label",
            "native_it003",
            "--gt_volume",
            str(gt_path),
            "--volume_frame",
            "relion",
            "--gt_frame",
            "recovar",
            "--gt_align",
            "--gt_align_healpix_order",
            "0",
            "--gt_align_max_shell",
            "4",
            "--output_npz",
            str(out_npz),
            "--output_json",
            str(out_json),
        ]
    )

    assert rc == 0
    metrics = np.load(out_npz)
    summary = json.loads(out_json.read_text())

    assert summary["volume_frame"] == "relion"
    assert summary["gt_frame"] == "recovar"
    assert summary["gt_align_enabled"] is True
    assert summary["volumes"][0]["label"] == "native_it003"
    assert summary["volumes"][0]["corr_vs_gt"] > 0.999
    assert "aligned" in summary["volumes"][0]
    assert np.isfinite(summary["volumes"][0]["aligned"]["corr_vs_gt"])
    assert summary["volumes"][0]["aligned"]["sign"] == 1
    assert float(metrics["native_it003_corr_vs_gt"]) > 0.999
    assert np.isfinite(float(metrics["native_it003_aligned_corr_vs_gt"]))
    assert np.all(np.asarray(metrics["native_it003_fsc_vs_gt"])[1:5] > 0.999)
