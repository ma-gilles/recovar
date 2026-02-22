from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.commands import run_test_all_metrics as rtam

pytestmark = pytest.mark.unit


def _logger():
    return SimpleNamespace(info=lambda *_: None, warning=lambda *_: None)


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


def test_generate_compact_support_test_volumes_has_high_freq_content(tmp_path):
    prefix = rtam.generate_compact_support_test_volumes(
        output_dir=str(tmp_path),
        grid_size=32,
        n_volumes=10,
        voxel_size=1.0,
        output_prefix=str(tmp_path / "vol"),
    )
    p0 = Path(f"{prefix}0000.mrc")
    p5 = Path(f"{prefix}0005.mrc")
    assert p0.exists() and p5.exists()

    v0 = rtam.utils.load_mrc(p0)
    v5 = rtam.utils.load_mrc(p5)
    assert v0.shape == (32, 32, 32)
    # Moving component should change volume over sequence.
    assert np.linalg.norm(v0 - v5) > 1e-3

    # Check that a non-trivial fraction of spectral energy is in higher frequencies.
    fv = np.fft.fftn(v0)
    power = np.abs(fv) ** 2
    freqs = np.fft.fftfreq(v0.shape[0])
    fx, fy, fz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    fr = np.sqrt(fx**2 + fy**2 + fz**2)
    hi = power[fr > 0.22].sum()
    lo = power[fr <= 0.22].sum() + 1e-12
    hi_lo_ratio = hi / lo
    assert hi_lo_ratio > 0.08


def test_validate_storage_args_for_generated_volumes_requires_explicit_outdir():
    args = SimpleNamespace(volume_input=None)
    with pytest.raises(ValueError, match="must pass --output-dir/-o explicitly"):
        rtam.validate_storage_args_for_generated_volumes(args, argv=["--grid-size", "32"])

    rtam.validate_storage_args_for_generated_volumes(args, argv=["--output-dir", "/scratch/tmp/out"])
    rtam.validate_storage_args_for_generated_volumes(args, argv=["-o", "/scratch/tmp/out"])

    args_with_input = SimpleNamespace(volume_input="/scratch/vol")
    rtam.validate_storage_args_for_generated_volumes(args_with_input, argv=["--grid-size", "32"])


def test_resolve_metrics_baseline_path_defaults_for_generated_volumes(tmp_path):
    args = SimpleNamespace(
        metrics_baseline_json=None,
        generate_volumes=True,
        output_dir=str(tmp_path),
        grid_size=128,
        generated_n_volumes=50,
    )
    path = rtam.resolve_metrics_baseline_path(args)
    assert path == tmp_path / "generated_volumes" / "metrics_baseline_grid128_nvol50.json"


def test_resolve_metrics_baseline_path_explicit_wins(tmp_path):
    explicit = tmp_path / "my_baseline.json"
    args = SimpleNamespace(
        metrics_baseline_json=str(explicit),
        generate_volumes=True,
        output_dir=str(tmp_path),
        grid_size=64,
        generated_n_volumes=10,
    )
    assert rtam.resolve_metrics_baseline_path(args) == explicit


def test_resolve_metrics_baseline_path_returns_none_when_not_generated_and_not_explicit(tmp_path):
    args = SimpleNamespace(
        metrics_baseline_json=None,
        generate_volumes=False,
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
    }
    out = rtam.normalize_scores_for_json(inp)
    assert out["a"] == 1.25
    assert out["b"] == 7.0
    assert out["c"] == [1.0, 2.0]
    assert out["d"] == "ok"


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
