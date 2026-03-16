from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pytest
import jax.numpy as jnp

pytest.importorskip("jax")

from recovar.commands import run_test_all_metrics as rtam

pytestmark = pytest.mark.unit


def _logger():
    return SimpleNamespace(info=lambda *_: None, warning=lambda *_: None)


def _install_main_runtime_stubs(monkeypatch, tmp_path, *, mean_fsc=0.5, variance_fsc=0.4):
    class _ArgsParser:
        def parse_args(self, _cmd):
            return SimpleNamespace()

    monkeypatch.setattr(rtam.pipeline, "add_args", lambda _p: _ArgsParser())
    monkeypatch.setattr(rtam.pipeline, "standard_recovar_pipeline", lambda _args: None)
    monkeypatch.setattr(rtam.compute_state, "add_args", lambda _p: _ArgsParser())
    monkeypatch.setattr(rtam.compute_state, "compute_state", lambda _args: None)

    monkeypatch.setattr(
        rtam,
        "make_big_test_dataset",
        lambda *_args, **_kwargs: {
            "image_assignment": np.array([0, 1], dtype=np.int32),
            "noise_variance": np.array([1.0, 1.0], dtype=np.float32),
            "dose_indices": None,
        },
    )
    monkeypatch.setattr(
        rtam,
        "compute_noise_variance_metrics",
        lambda *_args, **_kwargs: {
            "noise_mean_relative_error": 0.1,
            "noise_correlation": 0.9,
        },
    )

    def _fake_plot_fsc_new(*_args, **kwargs):
        name = kwargs.get("name", "")
        if name == "Mean FSC":
            return None, float(mean_fsc)
        if name in ("Variance FSC", "Variance Spatial FSC"):
            return None, float(variance_fsc)
        return None, 0.0

    monkeypatch.setattr(rtam.plot_utils, "plot_fsc_new", _fake_plot_fsc_new)
    monkeypatch.setattr(
        rtam.metrics,
        "get_all_variance_scores",
        lambda *_args, **_kwargs: (
            None,
            np.linspace(0.0, 1.0, 20, dtype=np.float32),
            np.linspace(0.0, 1.0, 20, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(rtam.metrics, "variance_of_zs", lambda *_args, **_kwargs: (None, 0.01))
    monkeypatch.setattr(
        rtam.metrics,
        "make_union_gt_mask_from_hvd",
        lambda *_args, **_kwargs: (np.ones((2, 2, 2), dtype=np.float32), np.ones((2, 2, 2), dtype=bool)),
    )
    monkeypatch.setattr(
        rtam.metrics,
        "compute_volume_error_metrics_from_gt",
        lambda *_args, **_kwargs: {"ninety_pc_locres": 1.0, "median_locres": 1.0, "mask": None},
    )
    monkeypatch.setattr(
        rtam.metrics,
        "subspace_angles",
        lambda *_args, **_kwargs: np.linspace(0.0, 1.0, 20, dtype=np.float32),
    )
    def _fake_idft3(arr):
        a = np.asarray(arr)
        n = round(a.size ** (1/3))
        return a.reshape(n, n, n)
    monkeypatch.setattr(rtam.fourier_transform_utils, "get_idft3", _fake_idft3)
    monkeypatch.setattr(
        rtam.utils,
        "load_mrc",
        lambda _path: np.zeros((2, 2, 2), dtype=np.float32),
    )

    class _PipelineOutput:
        def __init__(self, _path):
            self._embedding = {
                "latent_coords": {
                    4: np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                    10: np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                },
                "contrasts": {
                    4: np.array([0.25, 0.75], dtype=np.float32),
                    10: np.array([0.30, 0.70], dtype=np.float32),
                },
                "contrasts_noreg": {
                    4: np.array([0.20, 0.80], dtype=np.float32),
                    10: np.array([0.35, 0.65], dtype=np.float32),
                },
            }
            self._u_real = np.ones((20, 2, 2, 2), dtype=np.float32)
            self._s = np.linspace(1.0, 2.0, 20, dtype=np.float32)
            self._cryos = SimpleNamespace(volume_shape=(2, 2, 2), voxel_size=1.0)

        def get_embedding_component(self, entry, key):
            return self._embedding[entry][key]

        def get_u_real(self, n_pcs):
            return self._u_real[: int(n_pcs)]

        def get(self, key):
            mapping = {
                "lazy_dataset": self._cryos,
                "mean": np.zeros((2, 2, 2), dtype=np.float32),
                "variance": np.zeros((2, 2, 2), dtype=np.float32),
                "variance_est": {"combined": np.zeros(8, dtype=np.float32)},
                "volume_shape": (2, 2, 2),
                "s": self._s,
                "noise_var_used": np.array([1.0, 1.0], dtype=np.float32),
                "unsorted_embedding": self._embedding,
            }
            return mapping[key]

    monkeypatch.setattr(rtam.output, "PipelineOutput", _PipelineOutput)

    class _SyntheticGT:
        def __init__(self):
            self.contrasts = np.array([0.3, 0.7], dtype=np.float32)
            self.volumes = np.stack(
                [
                    np.zeros((8,), dtype=np.complex64),
                    np.ones((8,), dtype=np.complex64),
                ],
                axis=0,
            )

        def get_mean(self):
            return np.zeros((2, 2, 2), dtype=np.float32)

        def get_spatial_variances(self, contrasted=False):
            _ = contrasted
            return np.zeros((2, 2, 2), dtype=np.float32)

        def get_vol_svd(self, contrasted=False, real_space=True, random_svd_pcs=200):
            _ = (contrasted, real_space, random_svd_pcs)
            return (
                np.zeros((8, 20), dtype=np.float32),
                np.ones((20,), dtype=np.float32),
                np.zeros((20, 8), dtype=np.float32),
            )

    monkeypatch.setattr(
        rtam.synthetic_dataset,
        "load_heterogeneous_reconstruction",
        lambda _path: _SyntheticGT(),
    )


def test_main_generated_run_writes_default_baseline_when_missing(monkeypatch, tmp_path):
    _install_main_runtime_stubs(monkeypatch, tmp_path, mean_fsc=0.55, variance_fsc=0.45)
    monkeypatch.setattr(
        rtam.sys,
        "argv",
        [
            "run_test_all_metrics",
            "--output-dir",
            str(tmp_path),
            "--cpu",
            "--generate-pdb-volumes",
            "--generated-n-volumes",
            "3",
            "--grid-size",
            "8",
        ],
    )

    rtam.main()

    baseline_path = tmp_path / "generated_volumes" / "metrics_baseline_pdb_grid8_nvol3.json"
    report_path = tmp_path / "test_dataset" / "metrics_plot" / "metrics_regression_report.json"
    assert baseline_path.exists()
    assert report_path.exists()

    with open(baseline_path, "r") as f:
        baseline_scores = json.load(f)
    with open(report_path, "r") as f:
        report = json.load(f)

    assert baseline_scores["mean_fsc"] == pytest.approx(0.55)
    assert baseline_scores["variance_fsc"] == pytest.approx(0.45)
    assert report["status"] == "baseline_written"


def test_main_regression_fail_raises_without_skip(monkeypatch, tmp_path):
    _install_main_runtime_stubs(monkeypatch, tmp_path, mean_fsc=0.50, variance_fsc=0.40)
    baseline_path = tmp_path / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump({"mean_fsc": 0.99, "variance_fsc": 0.99}, f, indent=2)

    monkeypatch.setattr(
        rtam.sys,
        "argv",
        [
            "run_test_all_metrics",
            "--output-dir",
            str(tmp_path),
            "--cpu",
            "--volume-input",
            str(tmp_path / "fixed_vol_prefix_"),
            "--metrics-baseline-json",
            str(baseline_path),
            "--metrics-regression-tol-frac",
            "0.01",
        ],
    )

    with pytest.raises(SystemExit, match="1"):
        rtam.main()

    report_path = tmp_path / "test_dataset" / "metrics_plot" / "metrics_regression_report.json"
    assert report_path.exists()
    with open(report_path, "r") as f:
        report = json.load(f)
    assert report["status"] == "checked"
    assert report["checked_metrics"] >= 2
    assert len(report["failures"]) >= 1


def test_main_regression_fail_is_allowed_with_skip_flag(monkeypatch, tmp_path):
    _install_main_runtime_stubs(monkeypatch, tmp_path, mean_fsc=0.50, variance_fsc=0.40)
    baseline_path = tmp_path / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump({"mean_fsc": 0.99, "variance_fsc": 0.99}, f, indent=2)

    monkeypatch.setattr(
        rtam.sys,
        "argv",
        [
            "run_test_all_metrics",
            "--output-dir",
            str(tmp_path),
            "--cpu",
            "--volume-input",
            str(tmp_path / "fixed_vol_prefix_"),
            "--metrics-baseline-json",
            str(baseline_path),
            "--metrics-regression-tol-frac",
            "0.01",
            "--skip-metrics-regression-check",
        ],
    )

    rtam.main()

    report_path = tmp_path / "test_dataset" / "metrics_plot" / "metrics_regression_report.json"
    with open(report_path, "r") as f:
        report = json.load(f)
    assert report["status"] == "checked"
    assert len(report["failures"]) >= 1


def test_main_overwrite_metrics_baseline_forces_rewrite(monkeypatch, tmp_path):
    _install_main_runtime_stubs(monkeypatch, tmp_path, mean_fsc=0.61, variance_fsc=0.51)
    baseline_path = tmp_path / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump({"mean_fsc": 0.1, "variance_fsc": 0.1}, f, indent=2)

    monkeypatch.setattr(
        rtam.sys,
        "argv",
        [
            "run_test_all_metrics",
            "--output-dir",
            str(tmp_path),
            "--cpu",
            "--volume-input",
            str(tmp_path / "fixed_vol_prefix_"),
            "--metrics-baseline-json",
            str(baseline_path),
            "--overwrite-metrics-baseline",
        ],
    )

    rtam.main()

    report_path = tmp_path / "test_dataset" / "metrics_plot" / "metrics_regression_report.json"
    with open(baseline_path, "r") as f:
        baseline_scores = json.load(f)
    with open(report_path, "r") as f:
        report = json.load(f)

    assert baseline_scores["mean_fsc"] == pytest.approx(0.61)
    assert baseline_scores["variance_fsc"] == pytest.approx(0.51)
    assert report["status"] == "baseline_written"


def test_main_regression_pass_writes_checked_report_without_failures(monkeypatch, tmp_path):
    _install_main_runtime_stubs(monkeypatch, tmp_path, mean_fsc=0.62, variance_fsc=0.52)
    baseline_path = tmp_path / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(
            {
                "mean_fsc": 0.60,
                "variance_fsc": 0.50,
                "noise_mean_relative_error": 0.20,
            },
            f,
            indent=2,
        )

    monkeypatch.setattr(
        rtam.sys,
        "argv",
        [
            "run_test_all_metrics",
            "--output-dir",
            str(tmp_path),
            "--cpu",
            "--volume-input",
            str(tmp_path / "fixed_vol_prefix_"),
            "--metrics-baseline-json",
            str(baseline_path),
            "--metrics-regression-tol-frac",
            "0.001",
        ],
    )

    rtam.main()

    report_path = tmp_path / "test_dataset" / "metrics_plot" / "metrics_regression_report.json"
    with open(report_path, "r") as f:
        report = json.load(f)

    assert report["status"] == "checked"
    assert report["checked_metrics"] >= 3
    assert report["failures"] == []


def test_main_explicit_volume_without_baseline_skips_regression_report(monkeypatch, tmp_path):
    _install_main_runtime_stubs(monkeypatch, tmp_path, mean_fsc=0.55, variance_fsc=0.45)
    monkeypatch.setattr(
        rtam.sys,
        "argv",
        [
            "run_test_all_metrics",
            "--output-dir",
            str(tmp_path),
            "--cpu",
            "--volume-input",
            str(tmp_path / "fixed_vol_prefix_"),
        ],
    )

    rtam.main()

    scores_path = tmp_path / "test_dataset" / "metrics_plot" / "all_scores.json"
    report_path = tmp_path / "test_dataset" / "metrics_plot" / "metrics_regression_report.json"
    assert scores_path.exists()
    assert not report_path.exists()


def test_main_emits_canonical_keys_and_legacy_aliases_and_perf(monkeypatch, tmp_path):
    """Check that a run emits both new canonical keys and legacy aliases, plus perf."""
    _install_main_runtime_stubs(monkeypatch, tmp_path, mean_fsc=0.55, variance_fsc=0.45)
    monkeypatch.setattr(
        rtam.sys,
        "argv",
        [
            "run_test_all_metrics",
            "--output-dir",
            str(tmp_path),
            "--cpu",
            "--volume-input",
            str(tmp_path / "fixed_vol_prefix_"),
        ],
    )

    rtam.main()

    scores_path = tmp_path / "test_dataset" / "metrics_plot" / "all_scores.json"
    with open(scores_path, "r") as f:
        scores = json.load(f)

    # Canonical keys present
    assert "svd_relative_variance_4" in scores
    assert "svd_relative_variance_10" in scores
    assert "contrast_abs_error_4" in scores
    assert "contrast_abs_error_4_noreg" in scores
    assert "state_0_locres_90pct" in scores
    assert "state_0_locres_median" in scores

    # Legacy aliases also present
    assert "pcs_relative_variance_4" in scores
    assert "contrasts_4" in scores
    assert "constrasts_4" in scores
    assert "state_0_ninety_pc_locres" in scores
    assert "state_0_median_locres" in scores

    # Canonical and legacy have same values
    assert scores["svd_relative_variance_4"] == scores["pcs_relative_variance_4"]
    assert scores["contrast_abs_error_4"] == scores["contrasts_4"]
    assert scores["state_0_locres_90pct"] == scores["state_0_ninety_pc_locres"]

    # Perf dict present with expected structure
    perf = scores.get("perf")
    assert isinstance(perf, dict)
    assert "gpu_name" in perf
    for stage in ["dataset_generation", "pipeline", "compute_state", "metrics"]:
        assert stage in perf, f"missing perf stage: {stage}"
        assert "wall_seconds" in perf[stage]
        assert "peak_gpu_memory_gb" in perf[stage]
        assert "peak_cpu_memory_gb" in perf[stage]


def test_main_regression_with_zero_comparable_metrics_writes_checked_report(monkeypatch, tmp_path):
    _install_main_runtime_stubs(monkeypatch, tmp_path, mean_fsc=0.55, variance_fsc=0.45)
    baseline_path = tmp_path / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump({"unknown_metric_name": 123.0}, f, indent=2)

    monkeypatch.setattr(
        rtam.sys,
        "argv",
        [
            "run_test_all_metrics",
            "--output-dir",
            str(tmp_path),
            "--cpu",
            "--volume-input",
            str(tmp_path / "fixed_vol_prefix_"),
            "--metrics-baseline-json",
            str(baseline_path),
        ],
    )

    rtam.main()

    report_path = tmp_path / "test_dataset" / "metrics_plot" / "metrics_regression_report.json"
    with open(report_path, "r") as f:
        report = json.load(f)

    assert report["status"] == "checked"
    assert report["checked_metrics"] == 0
    assert report["failures"] == []


def test_compute_noise_variance_metrics_single_noise(tmp_path):
    gt = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    est = np.array([1.1, 1.8, 2.9], dtype=np.float64)

    scores = rtam.compute_noise_variance_metrics(gt, est, str(tmp_path), _logger())

    assert "noise_mean_relative_error" in scores
    assert "noise_median_relative_error" in scores
    assert "noise_max_relative_error" in scores
    assert "noise_correlation" in scores
    assert (tmp_path / "noise_variance_comparison.png").exists()


def test_compute_noise_variance_metrics_per_tilt(tmp_path):
    gt = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    est = np.array(
        [
            [1.0, 2.1, 2.9],
            [1.2, 2.2, 3.4],
        ],
        dtype=np.float64,
    )
    dose_indices = np.array([0, 0, 1, 1], dtype=np.int64)

    scores = rtam.compute_noise_variance_metrics(
        gt,
        est,
        str(tmp_path),
        _logger(),
        dose_indices=dose_indices,
        noise_increase_per_tilt=0.1,
    )

    assert "noise_correlation_per_tilt" in scores
    assert len(scores["noise_correlation_per_tilt"]) == 2
    assert len(scores["noise_mean_error_per_tilt"]) == 2
    assert len(scores["noise_median_error_per_tilt"]) == 2
    assert (tmp_path / "noise_variance_comparison_per_tilt.png").exists()


def test_compute_noise_variance_metrics_missing_gt_returns_empty(tmp_path):
    scores = rtam.compute_noise_variance_metrics(None, np.array([1.0]), str(tmp_path), _logger())
    assert scores == {}


def test_compute_noise_variance_metrics_missing_estimate_returns_empty(tmp_path):
    gt = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    scores = rtam.compute_noise_variance_metrics(gt, None, str(tmp_path), _logger())
    assert scores == {}


def test_compute_noise_variance_metrics_truncates_mismatched_lengths(tmp_path):
    gt = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    est = np.array([1.1, 1.9], dtype=np.float64)
    scores = rtam.compute_noise_variance_metrics(gt, est, str(tmp_path), _logger())
    assert "noise_mean_relative_error" in scores
    assert np.isfinite(scores["noise_mean_relative_error"])
    assert (tmp_path / "noise_variance_comparison.png").exists()


def test_make_big_test_dataset_uses_detected_state_count(monkeypatch, tmp_path):
    captured = {}

    def fake_isfile(path):
        return any(path.endswith(f"{i:04d}.mrc") for i in range(3))

    def fake_generate_synthetic_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return None, {"simulated": True}

    monkeypatch.setattr(rtam.os.path, "isfile", fake_isfile)
    monkeypatch.setattr(rtam.output, "mkdir_safe", lambda *_: None)
    monkeypatch.setattr(rtam.simulator, "generate_synthetic_dataset", fake_generate_synthetic_dataset)

    sim_info = rtam.make_big_test_dataset(
        input_dir="/tmp/vol_prefix_",
        output_dir=str(tmp_path),
        n_images=25.0,
        grid_size=32,
        contrast_std=0.0,
        n_tilts=-1,
    )

    assert sim_info == {"simulated": True}
    volume_distribution = captured["kwargs"]["volume_distribution"]
    assert volume_distribution.shape == (3,)
    assert np.isclose(np.sum(volume_distribution), 1.0)
    assert captured["args"][3] == 25


def test_make_big_test_dataset_raises_when_no_input_volumes(monkeypatch, tmp_path):
    monkeypatch.setattr(rtam.os.path, "isfile", lambda _p: False)
    monkeypatch.setattr(rtam.output, "mkdir_safe", lambda *_: None)
    with pytest.raises(ValueError, match="No volumes found for prefix"):
        rtam.make_big_test_dataset(
            input_dir="/tmp/nonexistent_prefix_",
            output_dir=str(tmp_path),
            n_images=10,
            grid_size=16,
        )


def test_validate_storage_args_for_generated_volumes_requires_explicit_outdir():
    args = SimpleNamespace(volume_input=None)
    with pytest.raises(ValueError, match="must pass --output-dir/-o explicitly"):
        rtam.validate_storage_args_for_generated_volumes(args, argv=["--grid-size", "32"])

    rtam.validate_storage_args_for_generated_volumes(args, argv=["--output-dir", "/scratch/tmp/out"])
    rtam.validate_storage_args_for_generated_volumes(args, argv=["-o", "/scratch/tmp/out"])

    args_with_input = SimpleNamespace(volume_input="/scratch/vol")
    rtam.validate_storage_args_for_generated_volumes(args_with_input, argv=["--grid-size", "32"])


def test_validate_storage_args_for_generated_volumes_accepts_equals_style_flags():
    args = SimpleNamespace(volume_input=None)
    rtam.validate_storage_args_for_generated_volumes(args, argv=["--output-dir=/scratch/tmp/out"])
    rtam.validate_storage_args_for_generated_volumes(args, argv=["-o=/scratch/tmp/out"])


@pytest.mark.parametrize(
    ("argv_extra", "msg"),
    [
        (["--grid-size", "0"], "--grid-size must be positive"),
        (["--n-images", "0"], "--n-images must be positive"),
        (["--noise-level", "-1"], "--noise-level must be non-negative"),
        (["--contrast-std", "-0.1"], "--contrast-std must be non-negative"),
        (["--metrics-regression-tol-frac", "-0.01"], "--metrics-regression-tol-frac must be non-negative"),
    ],
)
def test_main_rejects_invalid_numeric_cli_args_early(monkeypatch, tmp_path, argv_extra, msg):
    monkeypatch.setattr(
        rtam.sys,
        "argv",
        [
            "run_test_all_metrics",
            "--volume-input",
            str(tmp_path / "vol_prefix_"),
            "--output-dir",
            str(tmp_path),
            "--cpu",
            *argv_extra,
        ],
    )
    with pytest.raises(ValueError, match=msg):
        rtam.main()


def test_main_rejects_nonpositive_generated_n_volumes_when_generating(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rtam.sys,
        "argv",
        [
            "run_test_all_metrics",
            "--output-dir",
            str(tmp_path),
            "--cpu",
            "--generate-pdb-volumes",
            "--generated-n-volumes",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="--generated-n-volumes must be positive"):
        rtam.main()


def test_resolve_metrics_baseline_path_defaults_for_generated_volumes(tmp_path):
    args = SimpleNamespace(
        metrics_baseline_json=None,
        generate_pdb_volumes=True,
        output_dir=str(tmp_path),
        grid_size=128,
        generated_n_volumes=50,
    )
    path = rtam.resolve_metrics_baseline_path(args)
    assert path == tmp_path / "generated_volumes" / "metrics_baseline_pdb_grid128_nvol50.json"


def test_resolve_metrics_baseline_path_explicit_wins(tmp_path):
    explicit = tmp_path / "my_baseline.json"
    args = SimpleNamespace(
        metrics_baseline_json=str(explicit),
        generate_pdb_volumes=True,
        output_dir=str(tmp_path),
        grid_size=64,
        generated_n_volumes=10,
    )
    assert rtam.resolve_metrics_baseline_path(args) == explicit


def test_resolve_metrics_baseline_path_returns_none_when_not_generated_and_not_explicit(tmp_path):
    args = SimpleNamespace(
        metrics_baseline_json=None,
        generate_pdb_volumes=False,
        output_dir=str(tmp_path),
        grid_size=64,
        generated_n_volumes=10,
    )
    assert rtam.resolve_metrics_baseline_path(args) is None


def test_normalize_scores_for_json_handles_numpy_scalars_and_arrays():
    inp = {
        "a": np.float32(1.25),
        "b": np.int64(7),
        "c": np.array([1.0, 2.0], dtype=np.float32),
        "d": "ok",
        "flag_py": True,
        "flag_np": np.bool_(False),
    }
    out = rtam.normalize_scores_for_json(inp)
    assert out["a"] == 1.25
    assert out["b"] == 7.0
    assert out["c"] == [1.0, 2.0]
    assert out["d"] == "ok"
    assert out["flag_py"] is True
    assert out["flag_np"] is False


def test_normalize_scores_for_json_handles_nested_dicts():
    inp = {
        "mean_fsc": np.float32(0.5),
        "perf": {
            "gpu_name": "NVIDIA H100",
            "pipeline": {
                "wall_seconds": np.float64(123.4),
                "peak_gpu_memory_gb": 5.2,
            },
        },
    }
    out = rtam.normalize_scores_for_json(inp)
    assert out["mean_fsc"] == pytest.approx(0.5)
    assert out["perf"]["gpu_name"] == "NVIDIA H100"
    assert out["perf"]["pipeline"]["wall_seconds"] == pytest.approx(123.4)
    assert out["perf"]["pipeline"]["peak_gpu_memory_gb"] == pytest.approx(5.2)


def test_compare_scores_against_baseline_applies_direction_and_tolerance():
    baseline = {
        "mean_fsc": 0.80,  # higher is better
        "noise_mean_relative_error": 0.20,  # lower is better
        "some_unknown_metric": 3.0,  # ignored
    }
    current_ok = {
        "mean_fsc": 0.79,  # slight drop (1.25%)
        "noise_mean_relative_error": 0.205,  # slight increase (2.5%)
        "some_unknown_metric": 10.0,
    }
    checked, failures, details = rtam.compare_scores_against_baseline(
        current_ok,
        baseline,
        tol_frac=0.03,
    )
    assert checked == 2
    assert failures == []
    assert details["mean_fsc"]["ok"]
    assert details["noise_mean_relative_error"]["ok"]

    current_bad = dict(current_ok)
    current_bad["mean_fsc"] = 0.70
    checked, failures, details = rtam.compare_scores_against_baseline(
        current_bad,
        baseline,
        tol_frac=0.03,
    )
    assert checked == 2
    assert len(failures) == 1
    assert "mean_fsc" in failures[0]
    assert not details["mean_fsc"]["ok"]


def test_compare_metric_handles_non_finite_values():
    ok, msg = rtam.compare_metric(np.nan, 1.0, direction="higher", tol_frac=0.05)
    assert not ok
    assert "non-finite values" in msg

    ok, msg = rtam.compare_metric(1.0, np.inf, direction="lower", tol_frac=0.05)
    assert not ok
    assert "non-finite values" in msg


def test_metric_direction_recognizes_known_tokens_and_defaults_to_ignore():
    assert rtam.metric_direction("mean_fsc") == "higher"
    assert rtam.metric_direction("noise_mean_relative_error") == "lower"
    # Keep backward compatibility for existing typo-ed keys.
    assert rtam.metric_direction("constrasts_4") == "lower"
    assert rtam.metric_direction("unclassified_metric_name") == "ignore"
    # Canonical renamed keys
    assert rtam.metric_direction("svd_relative_variance_4") == "higher"
    assert rtam.metric_direction("contrast_abs_error_10_noreg") == "lower"
    assert rtam.metric_direction("state_0_locres_90pct") == "lower"
    assert rtam.metric_direction("state_1_locres_median") == "lower"


def test_load_u_real_for_metrics_prefers_selective_api():
    class _PO:
        def get_u_real(self, n_pcs):
            assert n_pcs == 3
            return np.ones((3, 2, 2, 2), dtype=np.float32)

        def get(self, key):
            raise AssertionError("legacy get('u_real') should not be called when get_u_real exists")

    out = rtam.load_u_real_for_metrics(_PO(), 3)
    assert out.shape == (3, 2, 2, 2)


def test_load_u_real_for_metrics_falls_back_to_legacy_get():
    class _PO:
        def get(self, key):
            assert key == "u_real"
            return np.arange(5 * 2, dtype=np.float32).reshape(5, 2)

    out = rtam.load_u_real_for_metrics(_PO(), 3)
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out, np.arange(6, dtype=np.float32).reshape(3, 2))


def test_load_u_real_for_metrics_rejects_nonpositive_request():
    class _PO:
        def get(self, key):
            return np.zeros((1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="n_pcs must be positive"):
        rtam.load_u_real_for_metrics(_PO(), 0)


def test_load_unsorted_embedding_component_caches_by_component():
    """get('unsorted_embedding') is called once; components are cached by (entry, key)."""
    class _PO:
        def __init__(self):
            self.get_calls = 0
            self.root = {
                "latent_coords": {
                    10: np.array([1.0, 10.0], dtype=np.float32),
                    4: np.array([2.0, 4.0], dtype=np.float32),
                },
            }

        def get(self, key):
            assert key == "unsorted_embedding"
            self.get_calls += 1
            return self.root

    po = _PO()
    cache = {}

    first = rtam.load_unsorted_embedding_component(po, "latent_coords", 10, cache)
    second = rtam.load_unsorted_embedding_component(po, "latent_coords", 10, cache)
    third = rtam.load_unsorted_embedding_component(po, "latent_coords", 4, cache)

    # Root loaded once; same (entry, key) returns cached result.
    assert po.get_calls == 1
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_load_unsorted_embedding_component_legacy_fallback_caches_root_and_component():
    class _PO:
        def __init__(self):
            self.get_calls = 0
            self.root = {
                "latent_coords": {
                    4: np.array([[1.0, 2.0]], dtype=np.float32),
                },
                "contrasts": {
                    4: np.array([0.5], dtype=np.float32),
                },
            }

        def get(self, key):
            assert key == "unsorted_embedding"
            self.get_calls += 1
            return self.root

    po = _PO()
    cache = {}
    z_first = rtam.load_unsorted_embedding_component(po, "latent_coords", 4, cache)
    z_second = rtam.load_unsorted_embedding_component(po, "latent_coords", 4, cache)
    c_first = rtam.load_unsorted_embedding_component(po, "contrasts", 4, cache)
    c_second = rtam.load_unsorted_embedding_component(po, "contrasts", 4, cache)

    assert po.get_calls == 1
    np.testing.assert_array_equal(z_first, z_second)
    np.testing.assert_array_equal(c_first, c_second)


def test_select_state_target_latent_points_falls_back_to_nonempty_labels():
    unsorted_zs = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
        ],
        dtype=np.float32,
    )
    particle_assignment = np.array([1, 1, 3, 3], dtype=np.int64)

    points, labels = rtam.select_state_target_latent_points(
        unsorted_zs=unsorted_zs,
        particle_assignment=particle_assignment,
        preferred_labels=[0, 25],
        max_points=2,
    )

    assert labels == [1, 3]
    np.testing.assert_allclose(points, np.array([[1.0, 0.0], [5.0, 0.0]], dtype=np.float32))


def test_select_state_target_latent_points_filters_nonfinite_rows():
    unsorted_zs = np.array(
        [
            [1.0, 2.0],
            [np.nan, 0.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    particle_assignment = np.array([0, 0, 1], dtype=np.int64)

    points, labels = rtam.select_state_target_latent_points(
        unsorted_zs=unsorted_zs,
        particle_assignment=particle_assignment,
        preferred_labels=[0, 1],
        max_points=2,
    )

    assert labels == [0, 1]
    assert np.all(np.isfinite(points))
    np.testing.assert_allclose(points[0], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(points[1], np.array([3.0, 4.0], dtype=np.float32))


def test_select_state_target_latent_points_rejects_length_mismatch():
    unsorted_zs = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    particle_assignment = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match="Length mismatch"):
        rtam.select_state_target_latent_points(
            unsorted_zs=unsorted_zs,
            particle_assignment=particle_assignment,
            preferred_labels=[0, 1],
            max_points=2,
        )


def test_select_state_target_latent_points_rejects_nonpositive_max_points():
    unsorted_zs = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    particle_assignment = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="max_points must be positive"):
        rtam.select_state_target_latent_points(
            unsorted_zs=unsorted_zs,
            particle_assignment=particle_assignment,
            preferred_labels=[0, 1],
            max_points=0,
        )


def test_select_state_target_latent_points_rejects_non_integer_like_assignments():
    unsorted_zs = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    particle_assignment = np.array([0.1, 1.9], dtype=np.float32)
    with pytest.raises(ValueError, match="integer-like"):
        rtam.select_state_target_latent_points(
            unsorted_zs=unsorted_zs,
            particle_assignment=particle_assignment,
            preferred_labels=[0, 1],
            max_points=2,
        )


def test_select_state_target_latent_points_accepts_integer_like_float_assignments():
    unsorted_zs = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=np.float32)
    particle_assignment = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    points, labels = rtam.select_state_target_latent_points(
        unsorted_zs=unsorted_zs,
        particle_assignment=particle_assignment,
        preferred_labels=[1, 0],
        max_points=2,
    )
    assert labels == [1, 0]
    np.testing.assert_allclose(points, np.array([[5.0, 0.0], [1.0, 0.0]], dtype=np.float32))


def test_select_state_target_latent_points_rejects_all_nonfinite_rows():
    unsorted_zs = np.array([[np.nan, 0.0], [np.inf, -1.0]], dtype=np.float32)
    particle_assignment = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="All rows in unsorted_zs are non-finite"):
        rtam.select_state_target_latent_points(
            unsorted_zs=unsorted_zs,
            particle_assignment=particle_assignment,
            preferred_labels=[0, 1],
            max_points=2,
        )


def test_load_u_real_for_metrics_legacy_get_handles_shorter_arrays():
    class _PO:
        def get(self, key):
            assert key == "u_real"
            return np.arange(2 * 3, dtype=np.float32).reshape(2, 3)

    out = rtam.load_u_real_for_metrics(_PO(), 10)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, np.arange(2 * 3, dtype=np.float32).reshape(2, 3))


def test_compare_scores_against_baseline_deduplicates_aliased_keys():
    """Aliased metric keys (e.g. pcs_relative_variance_4 / svd_relative_variance_4)
    should only be checked once, not counted twice."""
    current = {
        "svd_relative_variance_4": 0.90,
        "pcs_relative_variance_4": 0.90,  # legacy alias for same metric
        "mean_fsc": 0.80,
    }
    baseline = {
        "svd_relative_variance_4": 0.89,
        "pcs_relative_variance_4": 0.89,
        "mean_fsc": 0.79,
    }
    checked, failures, details = rtam.compare_scores_against_baseline(current, baseline, tol_frac=0.05)
    # svd_relative_variance_4 should be checked once (not twice counting the alias)
    assert checked == 2  # mean_fsc + one of the variance keys
    assert failures == []


def test_compare_scores_against_baseline_deduplicates_dynamic_locres_aliases():
    current = {
        "state_0_locres_90pct": 5.0,
        "state_0_ninety_pc_locres": 5.0,
        "state_0_locres_median": 4.0,
        "state_0_median_locres": 4.0,
    }
    baseline = dict(current)
    checked, failures, details = rtam.compare_scores_against_baseline(current, baseline, tol_frac=0.01)
    assert checked == 2  # one for each unique locres metric


def test_compare_scores_against_baseline_skips_non_numeric_values():
    current = {
        "mean_fsc": 0.80,
        "meta_list": [1, 2, 3],   # should be ignored
        "meta_dict": {"a": 1},    # should be ignored
    }
    baseline = {
        "mean_fsc": 0.79,
        "meta_list": [1, 2, 3],
        "meta_dict": {"a": 1},
    }
    checked, failures, details = rtam.compare_scores_against_baseline(current, baseline, tol_frac=0.01)
    assert checked == 1
    assert failures == []
    assert sorted(details.keys()) == ["mean_fsc"]


def test_compare_scores_against_baseline_with_no_comparable_metrics():
    current = {"meta_a": [1, 2], "meta_b": {"x": 1}}
    baseline = {"meta_a": [1, 2], "meta_b": {"x": 1}}
    checked, failures, details = rtam.compare_scores_against_baseline(current, baseline, tol_frac=0.01)
    assert checked == 0
    assert failures == []
    assert details == {}


def test_compare_scores_against_baseline_skips_boolean_values():
    current = {"flag_metric": True, "mean_fsc": 0.8}
    baseline = {"flag_metric": False, "mean_fsc": 0.79}
    checked, failures, details = rtam.compare_scores_against_baseline(current, baseline, tol_frac=0.01)
    assert checked == 1
    assert failures == []
    assert "flag_metric" not in details
    assert "mean_fsc" in details


def test_compare_scores_against_baseline_skips_numpy_boolean_values():
    current = {"flag_metric": np.bool_(True), "mean_fsc": 0.8}
    baseline = {"flag_metric": np.bool_(False), "mean_fsc": 0.79}
    checked, failures, details = rtam.compare_scores_against_baseline(current, baseline, tol_frac=0.01)
    assert checked == 1
    assert failures == []
    assert "flag_metric" not in details
    assert "mean_fsc" in details


def test_compute_noise_variance_metrics_per_tilt_without_dose_indices_returns_empty(tmp_path):
    gt = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    est = np.array(
        [
            [1.0, 2.1, 2.9],
            [1.2, 2.2, 3.4],
        ],
        dtype=np.float64,
    )
    scores = rtam.compute_noise_variance_metrics(
        gt,
        est,
        str(tmp_path),
        _logger(),
        dose_indices=None,
    )
    assert scores == {}


def test_compute_noise_variance_metrics_per_tilt_with_empty_dose_indices_returns_empty(tmp_path):
    gt = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    est = np.array(
        [
            [1.0, 2.1, 2.9],
            [1.2, 2.2, 3.4],
        ],
        dtype=np.float64,
    )
    scores = rtam.compute_noise_variance_metrics(
        gt,
        est,
        str(tmp_path),
        _logger(),
        dose_indices=np.array([], dtype=np.int32),
    )
    assert scores == {}


def test_compute_noise_variance_metrics_per_tilt_handles_noncontiguous_dose_labels(tmp_path):
    gt = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    # Two modeled tilt rows, but dose labels are 3 and 7.
    est = np.array(
        [
            [1.0, 2.1, 2.9],
            [1.2, 2.2, 3.4],
        ],
        dtype=np.float64,
    )
    dose_indices = np.array([3, 3, 7, 7], dtype=np.int64)

    scores = rtam.compute_noise_variance_metrics(
        gt,
        est,
        str(tmp_path),
        _logger(),
        dose_indices=dose_indices,
        noise_increase_per_tilt=0.05,
    )
    assert "noise_correlation_per_tilt" in scores
    assert len(scores["noise_correlation_per_tilt"]) == 2
    assert (tmp_path / "noise_variance_comparison_per_tilt.png").exists()


def test_compute_noise_variance_metrics_per_tilt_skips_unmatched_labels_without_zero_bias(tmp_path):
    gt = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    # Only one estimated row is available.
    est = np.array(
        [
            [1.0, 2.1, 2.9],
        ],
        dtype=np.float64,
    )
    # Two labels present; second label cannot be matched and should be skipped.
    dose_indices = np.array([0, 0, 5, 5], dtype=np.int64)

    scores = rtam.compute_noise_variance_metrics(
        gt,
        est,
        str(tmp_path),
        _logger(),
        dose_indices=dose_indices,
        noise_increase_per_tilt=0.0,
    )
    assert "noise_correlation_per_tilt" in scores
    assert len(scores["noise_correlation_per_tilt"]) == 1
    assert len(scores["noise_mean_error_per_tilt"]) == 1
    assert len(scores["noise_median_error_per_tilt"]) == 1


def test_compute_noise_variance_metrics_per_tilt_all_unmatched_returns_empty(tmp_path):
    gt = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    est = np.array([], dtype=np.float64).reshape(0, 3)
    dose_indices = np.array([4, 4, 9, 9], dtype=np.int64)

    scores = rtam.compute_noise_variance_metrics(
        gt,
        est,
        str(tmp_path),
        _logger(),
        dose_indices=dose_indices,
        noise_increase_per_tilt=0.0,
    )
    assert scores == {}


def test_compute_noise_variance_metrics_single_case_zero_overlap_returns_empty(tmp_path):
    gt = np.array([], dtype=np.float64)
    est = np.array([], dtype=np.float64)
    scores = rtam.compute_noise_variance_metrics(gt, est, str(tmp_path), _logger())
    assert scores == {}


def test_compute_noise_variance_metrics_per_tilt_zero_overlap_skips_and_returns_empty(tmp_path):
    gt = np.array([], dtype=np.float64)
    est = np.array([[]], dtype=np.float64)  # shape (1, 0)
    dose_indices = np.array([0, 0], dtype=np.int64)
    scores = rtam.compute_noise_variance_metrics(
        gt,
        est,
        str(tmp_path),
        _logger(),
        dose_indices=dose_indices,
        noise_increase_per_tilt=0.0,
    )
    assert scores == {}


def test_compute_noise_variance_metrics_single_case_nonfinite_corr_is_stabilized(tmp_path):
    # Constant vectors produce NaN correlation from np.corrcoef.
    gt = np.ones(5, dtype=np.float64)
    est = np.ones(5, dtype=np.float64)
    scores = rtam.compute_noise_variance_metrics(gt, est, str(tmp_path), _logger())
    assert "noise_correlation" in scores
    assert np.isfinite(scores["noise_correlation"])
    assert scores["noise_correlation"] == 0.0


def test_compute_noise_variance_metrics_per_tilt_nonfinite_corr_is_stabilized(tmp_path):
    gt = np.ones(4, dtype=np.float64)
    est = np.ones((2, 4), dtype=np.float64)
    dose_indices = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = rtam.compute_noise_variance_metrics(
        gt,
        est,
        str(tmp_path),
        _logger(),
        dose_indices=dose_indices,
        noise_increase_per_tilt=0.0,
    )
    assert "noise_correlation_per_tilt" in scores
    assert all(np.isfinite(c) for c in scores["noise_correlation_per_tilt"])
    assert scores["noise_correlation_per_tilt"] == [0.0, 0.0]


def test_compute_noise_variance_metrics_accepts_jax_arrays_for_per_tilt_branch(tmp_path):
    gt = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    est = jnp.array(
        [
            [1.0, 2.1, 2.9],
            [1.1, 2.0, 3.1],
        ],
        dtype=jnp.float32,
    )
    dose_indices = np.array([0, 0, 1, 1], dtype=np.int32)

    scores = rtam.compute_noise_variance_metrics(
        gt,
        est,
        str(tmp_path),
        _logger(),
        dose_indices=dose_indices,
        noise_increase_per_tilt=0.0,
    )

    assert "noise_correlation_per_tilt" in scores
    assert len(scores["noise_correlation_per_tilt"]) == 2
    assert (tmp_path / "noise_variance_comparison_per_tilt.png").exists()


def test_compute_noise_variance_metrics_accepts_jax_arrays_for_single_noise(tmp_path):
    gt = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    est = jnp.array([0.9, 2.1, 2.8, 4.2], dtype=jnp.float32)

    scores = rtam.compute_noise_variance_metrics(gt, est, str(tmp_path), _logger())

    assert "noise_mean_relative_error" in scores
    assert "noise_correlation" in scores
    assert np.isfinite(scores["noise_correlation"])
    assert (tmp_path / "noise_variance_comparison.png").exists()


def test_compute_noise_variance_metrics_accepts_scalar_estimate(tmp_path):
    gt = np.array([1.0], dtype=np.float64)
    est = np.float64(0.8)
    scores = rtam.compute_noise_variance_metrics(gt, est, str(tmp_path), _logger())
    assert "noise_mean_relative_error" in scores
    assert np.isfinite(scores["noise_correlation"])


def test_compute_noise_variance_metrics_accepts_python_lists_for_per_tilt(tmp_path):
    gt = [1.0, 2.0, 3.0]
    est = [[1.0, 2.1, 2.9], [0.9, 2.0, 3.2]]
    dose_indices = [0, 0, 1, 1]

    scores = rtam.compute_noise_variance_metrics(
        gt,
        est,
        str(tmp_path),
        _logger(),
        dose_indices=dose_indices,
        noise_increase_per_tilt=0.0,
    )
    assert "noise_correlation_per_tilt" in scores
    assert len(scores["noise_correlation_per_tilt"]) == 2
