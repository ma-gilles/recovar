import sys
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.commands import compute_embedding, estimate_conformational_density

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

    assert calls["args"]["zdim"] == "2_noreg"
    assert calls["args"]["pca_dim_max"] == 2
    assert calls["args"]["percentile_reject"] == 37
    assert calls["args"]["num_points"] == 200  # pca_dim=2 default
    assert calls["args"]["percentile_bound"] == 9
    assert np.allclose(calls["args"]["alphas"], np.array([1e-2, 1e-1]))


def test_compute_embedding_uses_saved_z_keys(monkeypatch):
    fake_results = {
        "input_args": SimpleNamespace(zdim=[4, 10]),
        "means": {"combined": np.zeros(4, dtype=np.float32)},
        "u": {"rescaled": np.zeros((4, 2), dtype=np.float32)},
        "s": {"rescaled": np.ones(2, dtype=np.float32)},
        "cov_noise": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones(4, dtype=np.float32),
        "zs": {10: np.zeros((3, 10)), 4: np.zeros((3, 4))},
    }

    calls = []

    monkeypatch.setattr(compute_embedding.o, "load_results_new", lambda _p: fake_results)
    monkeypatch.setattr(compute_embedding.dataset, "load_dataset_from_args", lambda _a: "cryos")
    monkeypatch.setattr(compute_embedding.utils, "make_algorithm_options", lambda _a: {"contrast": "none"})
    monkeypatch.setattr(compute_embedding.utils, "get_gpu_memory_total", lambda: 16)

    def fake_get_per_image_embedding(
        mean,
        u,
        s,
        zdim,
        cov_noise,
        cryos,
        volume_mask,
        gpu_memory,
        disc_type,
        contrast_grid,
        contrast_option,
    ):
        calls.append(zdim)
        n = 5
        return np.zeros((n, zdim)), np.zeros((n, zdim, zdim)), np.ones(n)

    monkeypatch.setattr(compute_embedding.embedding, "get_per_image_embedding", fake_get_per_image_embedding)

    zs, cov_zs, contrasts = compute_embedding.compute_embedding("/tmp/fake")

    assert calls == [4, 10]
    assert set(zs.keys()) == {4, 10}
    assert cov_zs[4].shape == (5, 4, 4)
    assert contrasts[10].shape == (5,)


def test_compute_embedding_falls_back_to_input_args_zdim(monkeypatch):
    fake_results = {
        "input_args": SimpleNamespace(zdim=[6]),
        "means": {"combined": np.zeros(4, dtype=np.float32)},
        "u": {"rescaled": np.zeros((4, 2), dtype=np.float32)},
        "s": {"rescaled": np.ones(2, dtype=np.float32)},
        "cov_noise": np.ones(4, dtype=np.float32),
        "volume_mask": np.ones(4, dtype=np.float32),
        "zs": {},
    }

    calls = []

    monkeypatch.setattr(compute_embedding.o, "load_results_new", lambda _p: fake_results)
    monkeypatch.setattr(compute_embedding.dataset, "load_dataset_from_args", lambda _a: "cryos")
    monkeypatch.setattr(compute_embedding.utils, "make_algorithm_options", lambda _a: {"contrast": "none"})
    monkeypatch.setattr(compute_embedding.utils, "get_gpu_memory_total", lambda: 16)
    monkeypatch.setattr(
        compute_embedding.embedding,
        "get_per_image_embedding",
        lambda *_args, **_kwargs: (calls.append(_args[3]) or (np.zeros((2, _args[3])), np.zeros((2, _args[3], _args[3])), np.zeros(2))),
    )

    zs, cov_zs, _ = compute_embedding.compute_embedding("/tmp/fake")
    assert calls == [6]
    assert 6 in zs and 6 in cov_zs


def test_compute_embedding_main_not_implemented():
    with pytest.raises(NotImplementedError):
        compute_embedding.main()
