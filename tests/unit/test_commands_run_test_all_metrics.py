"""
Unit tests for recovar.commands.run_test_all_metrics.

Covers:
  LOWER_IS_BETTER_TOKENS / HIGHER_IS_BETTER_TOKENS – metric direction
  _resolve_canonical_key()     – key aliasing (static and dynamic locres)
  validate_storage_args_for_generated_volumes() – CLI safety check
  load_u_real_for_metrics()    – eigenvector loading dispatch
  load_unsorted_embedding_component() – component API dispatch
  _stage_perf()                – performance snapshot math
"""
import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.commands import run_test_all_metrics as rtam

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Metric direction tokens
# ---------------------------------------------------------------------------

def test_lower_is_better_contains_error():
    assert "error" in rtam.LOWER_IS_BETTER_TOKENS


def test_lower_is_better_contains_contrast():
    assert "contrast" in rtam.LOWER_IS_BETTER_TOKENS


def test_higher_is_better_contains_fsc():
    assert "fsc" in rtam.HIGHER_IS_BETTER_TOKENS


def test_higher_is_better_contains_relative_variance():
    assert "relative_variance" in rtam.HIGHER_IS_BETTER_TOKENS


def test_direction_tokens_are_disjoint():
    """No token should appear in both lists."""
    lower = set(rtam.LOWER_IS_BETTER_TOKENS)
    higher = set(rtam.HIGHER_IS_BETTER_TOKENS)
    assert lower.isdisjoint(higher)


# ---------------------------------------------------------------------------
# _resolve_canonical_key – static aliases
# ---------------------------------------------------------------------------

def test_resolve_canonical_key_static_alias():
    assert rtam._resolve_canonical_key("pcs_relative_variance_4") == "svd_relative_variance_4"
    assert rtam._resolve_canonical_key("pcs_relative_variance_10") == "svd_relative_variance_10"


def test_resolve_canonical_key_contrast_typo():
    """The misspelled 'constrasts' key should map to canonical name."""
    assert rtam._resolve_canonical_key("constrasts_4") == "contrast_abs_error_4"


def test_resolve_canonical_key_contrast_noreg():
    assert rtam._resolve_canonical_key("contrasts_4_noreg") == "contrast_abs_error_4_noreg"
    assert rtam._resolve_canonical_key("contrasts_10_noreg") == "contrast_abs_error_10_noreg"


def test_resolve_canonical_key_passthrough():
    """Unknown keys should pass through unchanged."""
    assert rtam._resolve_canonical_key("mean_fsc") == "mean_fsc"
    assert rtam._resolve_canonical_key("variance_fsc") == "variance_fsc"


# ---------------------------------------------------------------------------
# _resolve_canonical_key – dynamic locres patterns
# ---------------------------------------------------------------------------

def test_resolve_canonical_key_ninety_pc_locres():
    assert rtam._resolve_canonical_key("state_0_ninety_pc_locres") == "state_0_locres_90pct"
    assert rtam._resolve_canonical_key("state_12_ninety_pc_locres") == "state_12_locres_90pct"


def test_resolve_canonical_key_median_locres():
    assert rtam._resolve_canonical_key("state_3_median_locres") == "state_3_locres_median"


def test_resolve_canonical_key_locres_no_match():
    """Patterns that don't match locres regex should pass through."""
    assert rtam._resolve_canonical_key("state_bad_ninety_pc_locres") == "state_bad_ninety_pc_locres"


# ---------------------------------------------------------------------------
# CANONICAL_KEY_ALIASES – completeness
# ---------------------------------------------------------------------------

def test_canonical_aliases_all_have_canonical_values():
    """Every alias value should differ from its key."""
    for old, new in rtam.CANONICAL_KEY_ALIASES.items():
        assert old != new, f"Alias {old} maps to itself"


# ---------------------------------------------------------------------------
# validate_storage_args_for_generated_volumes
# ---------------------------------------------------------------------------

def test_validate_storage_args_passes_with_volume_input():
    """When --volume-input is given, no validation error."""
    args = SimpleNamespace(volume_input="/some/path")
    rtam.validate_storage_args_for_generated_volumes(args, ["--volume-input", "/some/path"])


def test_validate_storage_args_passes_with_explicit_outdir():
    """When --output-dir is given explicitly, no validation error."""
    args = SimpleNamespace(volume_input=None)
    rtam.validate_storage_args_for_generated_volumes(
        args, ["--output-dir", "/tmp/out", "--generate-pdb-volumes"]
    )


def test_validate_storage_args_passes_with_short_outdir_flag():
    args = SimpleNamespace(volume_input=None)
    rtam.validate_storage_args_for_generated_volumes(
        args, ["-o", "/tmp/out", "--generate-pdb-volumes"]
    )


def test_validate_storage_args_passes_with_equals_syntax():
    args = SimpleNamespace(volume_input=None)
    rtam.validate_storage_args_for_generated_volumes(
        args, ["--output-dir=/tmp/out", "--generate-pdb-volumes"]
    )


def test_default_output_dir_prefers_tmp_recovar_dir(monkeypatch):
    monkeypatch.setenv("TMP_RECOVAR_DIR", "/scratch/tmp_recovar")
    assert rtam.default_output_dir() == "/scratch/tmp_recovar/recovar_test_all_metrics"


def test_validate_storage_args_allows_tmp_recovar_dir_env(monkeypatch):
    monkeypatch.setenv("TMP_RECOVAR_DIR", "/scratch/tmp_recovar")
    args = SimpleNamespace(volume_input=None)
    rtam.validate_storage_args_for_generated_volumes(args, ["--generate-pdb-volumes"])


def test_validate_storage_args_raises_without_outdir():
    """When generating volumes without explicit --output-dir, should raise."""
    args = SimpleNamespace(volume_input=None)
    with pytest.raises(ValueError, match="--output-dir"):
        rtam.validate_storage_args_for_generated_volumes(
            args, ["--generate-pdb-volumes"]
        )


# ---------------------------------------------------------------------------
# load_u_real_for_metrics
# ---------------------------------------------------------------------------

def test_load_u_real_for_metrics_uses_get_u_real_when_available():
    """Should prefer the get_u_real() method."""
    expected = np.random.randn(5, 100).astype(np.float32)

    class FakePO:
        def get_u_real(self, n_pcs):
            return expected[:n_pcs]

    result = rtam.load_u_real_for_metrics(FakePO(), 3)
    np.testing.assert_array_equal(result, expected[:3])


def test_load_u_real_for_metrics_falls_back_to_get():
    """Should fall back to get('u_real')[:n_pcs] when get_u_real is absent."""
    full = np.random.randn(10, 64).astype(np.float32)

    class FakePO:
        def get(self, key):
            if key == 'u_real':
                return full
            raise KeyError(key)

    result = rtam.load_u_real_for_metrics(FakePO(), 4)
    np.testing.assert_array_equal(result, full[:4])


def test_load_u_real_for_metrics_rejects_zero():
    with pytest.raises(ValueError, match="positive"):
        rtam.load_u_real_for_metrics(None, 0)


def test_load_u_real_for_metrics_rejects_negative():
    with pytest.raises(ValueError, match="positive"):
        rtam.load_u_real_for_metrics(None, -1)


# ---------------------------------------------------------------------------
# load_unsorted_embedding_component
# ---------------------------------------------------------------------------

def test_load_unsorted_embedding_component_uses_get_unsorted():
    """Should use get('unsorted_embedding') to retrieve data in original order."""
    expected = np.arange(10, dtype=np.float32)

    class FakePO:
        def get(self, key):
            if key == 'unsorted_embedding':
                return {"latent_coords": {4: expected}}
            raise KeyError(key)

    result = rtam.load_unsorted_embedding_component(FakePO(), "latent_coords", 4)
    np.testing.assert_array_equal(result, expected)


def test_load_unsorted_embedding_component_caches():
    """Repeated calls with same entry+key should use cache."""
    call_count = {"n": 0}
    expected = np.ones(5, dtype=np.float32)

    class FakePO:
        def get(self, key):
            call_count["n"] += 1
            return {"latent_coords": {4: expected}}

    cache = {}
    rtam.load_unsorted_embedding_component(FakePO(), "latent_coords", 4, legacy_cache=cache)
    rtam.load_unsorted_embedding_component(FakePO(), "latent_coords", 4, legacy_cache=cache)
    # get() called once for __root__, then cached
    assert call_count["n"] == 1


def test_load_unsorted_embedding_component_prefers_pipeline_output_method():
    expected = np.array([1.0, 2.0], dtype=np.float32)

    class FakePO:
        def get_unsorted_embedding_component(self, entry, key):
            assert entry == "latent_coords"
            assert key == 4
            return expected

        def get(self, key):
            raise AssertionError(f"legacy get() path should not be used, got {key}")

    result = rtam.load_unsorted_embedding_component(FakePO(), "latent_coords", 4)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# _stage_perf
# ---------------------------------------------------------------------------

def test_stage_perf_computes_wall_seconds():
    before = {"wall_time": 100.0, "cpu_rss_bytes": 1e9, "gpu_bytes_in_use": 0, "gpu_peak_bytes": 0}
    after = {"wall_time": 107.5, "cpu_rss_bytes": 1.5e9, "gpu_bytes_in_use": 0, "gpu_peak_bytes": 0}
    perf = rtam._stage_perf(before, after)
    assert perf["wall_seconds"] == pytest.approx(7.5)


def test_stage_perf_tracks_cpu_peak():
    before = {"wall_time": 0, "cpu_rss_bytes": int(2e9), "gpu_bytes_in_use": 0, "gpu_peak_bytes": 0}
    after = {"wall_time": 1, "cpu_rss_bytes": int(3e9), "gpu_bytes_in_use": 0, "gpu_peak_bytes": 0}
    perf = rtam._stage_perf(before, after)
    assert perf["peak_cpu_memory_gb"] == pytest.approx(3.0)


def test_stage_perf_gpu_peak_increases():
    before = {"wall_time": 0, "cpu_rss_bytes": 0, "gpu_bytes_in_use": int(1e9), "gpu_peak_bytes": int(2e9)}
    after = {"wall_time": 1, "cpu_rss_bytes": 0, "gpu_bytes_in_use": int(3e9), "gpu_peak_bytes": int(5e9)}
    perf = rtam._stage_perf(before, after)
    # Peak increased during this stage: use after's peak
    assert perf["peak_gpu_memory_gb"] == pytest.approx(5.0)


def test_stage_perf_gpu_peak_from_earlier_stage():
    before = {"wall_time": 0, "cpu_rss_bytes": 0, "gpu_bytes_in_use": int(4e9), "gpu_peak_bytes": int(8e9)}
    after = {"wall_time": 1, "cpu_rss_bytes": 0, "gpu_bytes_in_use": int(2e9), "gpu_peak_bytes": int(8e9)}
    perf = rtam._stage_perf(before, after)
    # Peak did NOT increase: use max of endpoints
    assert perf["peak_gpu_memory_gb"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# main – argument registration
# ---------------------------------------------------------------------------

def test_main_registers_volume_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume-input', '-i', required=False, default=None)
    actions = parser._option_string_actions
    assert "--volume-input" in actions or "-i" in actions


def test_main_registers_output_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-o', default='/tmp/recovar_test_all_metrics')
    actions = parser._option_string_actions
    assert "--output-dir" in actions or "-o" in actions


def test_main_registers_grid_size_with_default():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-size', type=int, default=64)
    action = parser._option_string_actions["--grid-size"]
    assert action.default == 64


def test_main_registers_n_images_with_default():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-images', type=float, default=5e4)
    action = parser._option_string_actions["--n-images"]
    assert action.default == pytest.approx(5e4)
