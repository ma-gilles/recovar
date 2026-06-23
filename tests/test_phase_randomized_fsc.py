from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "spike_fullatom_method_benchmark"
    / "compare_uniform_noise3_state50_halfmap_vs_mapmodel.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_uniform_halfmap", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase_randomization_preserves_amplitudes_and_low_frequencies():
    module = _load_module()
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(16, 16, 16)).astype(np.float32)
    randomized = module._phase_randomize_volume(
        volume,
        cutoff_A=4.0,
        voxel_size=1.0,
        rng=np.random.default_rng(1),
    )

    original_ft = np.fft.fftn(volume)
    randomized_ft = np.fft.fftn(randomized)
    radius = module._fftfreq_radius(volume.shape, voxel_size=1.0)
    low = radius < 0.25
    high = radius >= 0.25

    np.testing.assert_allclose(np.abs(randomized_ft), np.abs(original_ft), rtol=2e-5, atol=2e-4)
    np.testing.assert_allclose(randomized_ft[low], original_ft[low], rtol=2e-5, atol=2e-4)
    assert np.mean(np.abs(randomized_ft[high] - original_ft[high])) > 1.0


def test_mask_corrected_fsc_uses_relion_formula_only_above_cutoff():
    module = _load_module()
    frequency = np.array([0.02, 0.08, 0.10, 0.20, 0.30], dtype=np.float64)
    masked = np.array([0.95, 0.90, 0.70, 0.50, 0.20], dtype=np.float32)
    randomized = np.array([0.60, 0.50, 0.25, 0.20, -0.10], dtype=np.float32)

    corrected = module._mask_corrected_fsc(masked, randomized, frequency, cutoff_A=10.0)
    expected = masked.copy()
    high = frequency >= 0.10
    expected[high] = (masked[high] - randomized[high]) / (1.0 - randomized[high])

    np.testing.assert_allclose(corrected, expected, rtol=1e-6, atol=1e-6)


def test_independent_phase_randomization_destroys_high_frequency_fsc_without_mask():
    module = _load_module()
    rng = np.random.default_rng(2)
    volume = rng.normal(size=(32, 32, 32)).astype(np.float32)
    mask = np.ones_like(volume, dtype=np.float32)
    labels, n_shells = module._shell_labels(volume.shape)
    frequency = np.arange(n_shells, dtype=np.float64) / 32.0
    randomized_fsc = module._phase_randomized_masked_fsc(
        volume,
        volume,
        mask,
        labels,
        n_shells,
        cutoff_A=4.0,
        voxel_size=1.0,
        seed=3,
    )
    high_away_from_cutoff = frequency >= 0.35
    assert np.nanmax(np.abs(randomized_fsc[high_away_from_cutoff])) < 0.12
