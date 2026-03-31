import sys
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.commands import estimate_conformational_density

pytestmark = pytest.mark.unit


def test_estimate_conformational_density_raises_for_missing_result_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        estimate_conformational_density.estimate_conformational_density(str(missing))


def test_estimate_conformational_density_forwards_percentile_and_defaults(monkeypatch, tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    out_dir = tmp_path / "density_out"

    calls = {}

    class FakePipelineOutput:
        def get(self, key):
            if key == "input_args":
                return SimpleNamespace(zdim=[2, 4, 8])
            raise KeyError(key)

    def fake_get_deconvolved_density(
        pipeline_output,
        zdim,
        noreg,
        pca_dim_max,
        percentile_reject,
        kernel_option,
        num_points,
        alphas,
        percentile_bound,
        save_to_file,
    ):
        calls["args"] = {
            "zdim": zdim,
            "noreg": noreg,
            "pca_dim_max": pca_dim_max,
            "percentile_reject": percentile_reject,
            "num_points": num_points,
            "percentile_bound": percentile_bound,
            "alphas": np.array(alphas),
        }
        sols = [np.ones((4, 4), dtype=np.float32) * 0.5, np.ones((4, 4), dtype=np.float32) * 0.8]
        alphas_out = np.array([1e-2, 1e-1], dtype=np.float64)
        cost = np.array([2.0, 4.0], dtype=np.float64)
        reg_cost = np.array([0.0, 0.0], dtype=np.float64)
        density = np.ones((4, 4), dtype=np.float32)
        total_covar = np.eye(2, dtype=np.float32)
        grids = None
        bounds = {"x": [-1, 1], "y": [-1, 1]}
        return sols, alphas_out, cost, reg_cost, density, total_covar, grids, bounds

    monkeypatch.setattr(estimate_conformational_density.output, "PipelineOutput", lambda _p: FakePipelineOutput())
    monkeypatch.setattr(
        estimate_conformational_density.deconvolve_density,
        "get_deconvolved_density",
        fake_get_deconvolved_density,
    )
    monkeypatch.setattr(
        estimate_conformational_density.deconvolve_density,
        "plot_density",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "kneed",
        SimpleNamespace(KneeLocator=lambda *args, **kwargs: SimpleNamespace(knee=1.0)),
    )

    estimate_conformational_density.estimate_conformational_density(
        recovar_result_dir=str(result_dir),
        output_dir=str(out_dir),
        pca_dim=2,
        z_dim_used=None,
        percentile_reject=37,
        num_disc_points=None,
        alphas=[1e-2, 1e-1],
        percentile_bound=9,
    )

    assert calls["args"]["zdim"] == 2
    assert calls["args"]["noreg"] is True
    assert calls["args"]["pca_dim_max"] == 2
    assert calls["args"]["percentile_reject"] == 37
    assert calls["args"]["num_points"] == 200  # pca_dim=2 default
    assert calls["args"]["percentile_bound"] == 9
    assert np.allclose(calls["args"]["alphas"], np.array([1e-2, 1e-1]))
