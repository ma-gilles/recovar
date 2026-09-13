"""Noise-only bootstrap must not require or enable full-state oracle replay."""

import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_full_refinement as driver

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('option, expected', [([], 'pipeline'), (['--initial-noise-bootstrap', 'relion'], 'relion')])
def test_noise_bootstrap_cli_is_opt_in_without_full_state_replay(monkeypatch, option, expected):
    class Parsed(BaseException):
        pass

    original = argparse.ArgumentParser.parse_args

    def stop_after_parse(parser, *args, **kwargs):
        parsed = original(parser, *args, **kwargs)
        assert parsed.initial_noise_bootstrap == expected
        assert parsed.relion_init_dir is None
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', stop_after_parse)
    monkeypatch.setattr(sys, 'argv', ['run_full_refinement.py', *option])
    with pytest.raises(Parsed):
        driver.main()


def _args(**overrides):
    return SimpleNamespace(**(dict(n_classes=1, init_relion_iteration=0,
        perturb_replay_relion_dir=None, relion_init_dir=None,
        init_noise_from_npz=None, initial_noise_cache_dir=None,
        relion_half_sets="particles.star") | overrides))


def test_noise_only_bootstrap_preserves_order_and_float32_boundary(monkeypatch):
    dataset = SimpleNamespace(grid_size=8)
    rows = np.array([4, 1, 2], dtype=np.int64)
    optics = np.array([7, 7, 7], dtype=np.int64)
    sigma = np.array([[.5, .4, .3, .2, .1]], dtype=np.float64)
    calls = []

    def compute(ds, **kwargs):
        assert ds is dataset
        np.testing.assert_array_equal(kwargs['source_rows'], rows)
        np.testing.assert_array_equal(kwargs['optics_group_ids'], optics)
        assert kwargs['particle_diameter_ang'] == 12
        assert kwargs['width_mask_edge_px'] == 3
        calls.append(kwargs)
        return sigma

    monkeypatch.setattr(driver, '_compute_relion_fresh_k1_initial_sigma2', compute)
    radial, noise = driver._compute_relion_noise_only_bootstrap(
        dataset, args=_args(), frozen_boundary=None, source_rows=rows,
        optics_group_ids=optics, mask_params=(12., 3))
    assert len(calls) == 1
    assert radial.dtype == np.float64 and noise.dtype == np.float32
    np.testing.assert_array_equal(radial, sigma[0] * 8**4)
    np.testing.assert_array_equal(noise, driver._relion_sigma2_to_native_noise_variance(
        sigma[0], grid_size=8, output_dtype=np.float32))


@pytest.mark.parametrize('overrides', [
    dict(n_classes=4), dict(init_relion_iteration=1),
    dict(perturb_replay_relion_dir='replay'), dict(relion_init_dir='oracle'),
    dict(init_noise_from_npz='noise.npz'), dict(initial_noise_cache_dir='cache'),
    dict(relion_half_sets=None),
])
def test_noise_only_bootstrap_rejects_conflicting_modes(overrides):
    with pytest.raises(ValueError, match='noise-only bootstrap'):
        driver._compute_relion_noise_only_bootstrap(SimpleNamespace(grid_size=8),
            args=_args(**overrides), frozen_boundary=None,
            source_rows=np.arange(3), optics_group_ids=np.ones(3), mask_params=(12., 3))


@pytest.mark.parametrize('overrides', [
    dict(frozen_boundary=object()), dict(source_rows=None),
    dict(optics_group_ids=None), dict(mask_params=None),
    dict(optics_group_ids=np.array([1, 2, 1])),
])
def test_noise_only_bootstrap_rejects_missing_or_unsupported_inputs(overrides):
    params = dict(frozen_boundary=None, source_rows=np.arange(3),
                  optics_group_ids=np.ones(3), mask_params=(12., 3)) | overrides
    with pytest.raises(ValueError, match='noise-only bootstrap'):
        driver._compute_relion_noise_only_bootstrap(SimpleNamespace(grid_size=8),
            args=_args(), **params)
