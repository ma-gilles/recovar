"""RELION SamplingPerturbation of a trial grid and its M-step source angles have one rule each.

Both the regular iterations and the final all-data pass of the refinement
controller use ``_relion_mstep_source_eulers`` and ``_perturbed_trial_grid``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import recovar.em.refinement.iteration_loop as iteration_loop
from recovar.em.sampling import (
    apply_relion_rotation_perturbation_to_eulers,
    apply_relion_translation_perturbation,
    relion_angular_sampling_deg,
    rotation_grid_size,
)

pytestmark = pytest.mark.unit

ORDER = 1
N_ROT = rotation_grid_size(ORDER)


def _canonical_eulers(order):
    """Deterministic stand-in for RELION's native grid angles (no native binding needed)."""
    n = rotation_grid_size(order)
    return np.stack([np.linspace(0.0, 350.0, n), np.linspace(10.0, 170.0, n), np.linspace(5.0, 355.0, n)], axis=1)


@pytest.fixture(autouse=True)
def _fake_canonical_grid(monkeypatch):
    monkeypatch.setattr(iteration_loop, "_get_relion_rotation_grid_eulers_float64", _canonical_eulers)


def test_mstep_source_uses_the_canonical_grid_at_matching_size():
    eulers = np.zeros((N_ROT, 3), dtype=np.float32)
    source = iteration_loop._relion_mstep_source_eulers(eulers, ORDER)
    expected = _canonical_eulers(ORDER)
    assert source.dtype == np.float64 and source.tobytes() == expected.tobytes()


def test_mstep_source_falls_back_to_the_grid_angles_when_sizes_differ():
    eulers = np.arange(15, dtype=np.float32).reshape(5, 3)
    source = iteration_loop._relion_mstep_source_eulers(eulers, ORDER)
    assert source.dtype == np.float64 and source.tolist() == eulers.astype(np.float64).tolist()


def test_sealed_grid_supplies_its_own_angles(monkeypatch):
    eulers = np.arange(3 * N_ROT, dtype=np.float32).reshape(-1, 3)
    monkeypatch.setattr(
        iteration_loop, "_get_relion_rotation_grid_eulers_float64", lambda order: pytest.fail("canonical grid must not be built")
    )
    source = iteration_loop._relion_mstep_source_eulers(eulers, ORDER, use_grid_eulers=True)
    assert source.dtype == np.float64 and source.tolist() == eulers.astype(np.float64).tolist()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_perturbed_trial_grid_matches_the_separate_relion_calls(dtype):
    eulers = _canonical_eulers(ORDER).astype(np.float32)
    base_translations = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    angsamp = relion_angular_sampling_deg(ORDER, adaptive_oversampling=0)
    grid = iteration_loop._perturbed_trial_grid(
        rotation_eulers=eulers,
        mstep_source_eulers=_canonical_eulers(ORDER),
        base_translations=base_translations,
        translation_step=2.0,
        random_perturbation=0.25,
        angular_sampling_deg=angsamp,
        dtype=dtype,
    )
    rotations, public_eulers = apply_relion_rotation_perturbation_to_eulers(eulers, 0.25, angsamp, dtype=dtype)
    _, _, mstep = apply_relion_rotation_perturbation_to_eulers(
        _canonical_eulers(ORDER), 0.25, angsamp, return_mstep_rotations=True, dtype=dtype
    )
    translations = jnp.asarray(apply_relion_translation_perturbation(base_translations, 0.25, 2.0), dtype=dtype)
    for got, want in ((grid.rotations, rotations), (grid.rotation_eulers, public_eulers), (grid.mstep_rotations, mstep)):
        assert got.dtype == want.dtype and got.shape == want.shape
        assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
    assert grid.translations.dtype == translations.dtype
    assert np.asarray(grid.translations).tobytes() == np.asarray(translations).tobytes()


def test_perturbed_trial_grid_records_call_order(monkeypatch):
    calls = []

    def fake_rot(eulers, rp, angsamp, *, return_mstep_rotations=False, dtype=np.float32):
        calls.append(("rot", return_mstep_rotations, rp, angsamp, dtype))
        n = np.asarray(eulers).shape[0]
        out = (np.zeros((n, 3, 3), dtype=dtype), np.asarray(eulers, dtype=dtype))
        return out + ((np.ones((n, 3, 3), dtype=dtype),) if return_mstep_rotations else ())

    def fake_trans(base, rp, step):
        calls.append(("trans", rp, step))
        return np.asarray(base) + rp

    monkeypatch.setattr(iteration_loop, "apply_relion_rotation_perturbation_to_eulers", fake_rot)
    monkeypatch.setattr(iteration_loop, "apply_relion_translation_perturbation", fake_trans)
    grid = iteration_loop._perturbed_trial_grid(
        rotation_eulers=np.zeros((4, 3)),
        mstep_source_eulers=np.zeros((4, 3)),
        base_translations=np.zeros((2, 2)),
        translation_step=1.5,
        random_perturbation=0.5,
        angular_sampling_deg=30.0,
        dtype=np.float32,
    )
    assert calls == [("rot", False, 0.5, 30.0, np.float32), ("rot", True, 0.5, 30.0, np.float32), ("trans", 0.5, 1.5)]
    assert grid.mstep_rotations.shape == (4, 3, 3) and float(grid.mstep_rotations[0, 0, 0]) == 1.0
    assert grid.translations.shape == (2, 2) and float(grid.translations[0, 0]) == 0.5
