"""The exact fine local-search grid and its M-step matrices have one owner in the controller module.

Both the regular iterations and the final all-data pass call
``_exact_local_fine_grid`` (RELION SamplingPerturbation of the materialized
fine grid plus the exact M-step rotations) and ``_local_search_mstep_rotations``
(reuse of a scoring grid's M-step matrices); the final pass sizes its parent
pass with ``relion_local_pass1_current_size`` only under adaptive oversampling.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import recovar.em.dense_single_volume.iteration_loop as iteration_loop
from recovar.em.sampling import (
    _relion_mstep_rotations_from_eulers,
    apply_relion_rotation_perturbation_to_eulers,
    relion_angular_sampling_deg,
    rotation_grid_size,
)

pytestmark = pytest.mark.unit

ORDER = 1
N_ROT = rotation_grid_size(ORDER)
ANGULAR_SAMPLING = relion_angular_sampling_deg(ORDER, adaptive_oversampling=0)


def _canonical_eulers(order):
    n = rotation_grid_size(order)
    return np.stack([np.linspace(0.0, 350.0, n), np.linspace(10.0, 170.0, n), np.linspace(5.0, 355.0, n)], axis=1) + 0.1234567


def _fake_grid(order, dtype=np.float32):
    source = _canonical_eulers(order)
    return _relion_mstep_rotations_from_eulers(source, dtype=dtype), source.astype(dtype)


def _same(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


@pytest.fixture(autouse=True)
def _fake_canonical_grid(monkeypatch):
    monkeypatch.setattr(iteration_loop, "_get_relion_rotation_grid_eulers_float64", _canonical_eulers)
    monkeypatch.setattr(iteration_loop, "_relion_rotation_grid_float32", _fake_grid)


@pytest.mark.parametrize("random_perturbation", [0.3, -0.125, 0.0])
def test_fine_grid_is_perturbed_with_exact_mstep_rotations(random_perturbation):
    rotations, eulers, mstep = iteration_loop._exact_local_fine_grid(
        healpix_order=ORDER, angular_sampling_deg=ANGULAR_SAMPLING, random_perturbation=random_perturbation
    )
    _, grid_eulers = _fake_grid(ORDER)
    exp_rot, exp_eulers = apply_relion_rotation_perturbation_to_eulers(grid_eulers, random_perturbation, ANGULAR_SAMPLING)
    _, _, exp_mstep = apply_relion_rotation_perturbation_to_eulers(
        _canonical_eulers(ORDER), random_perturbation, ANGULAR_SAMPLING, return_mstep_rotations=True
    )
    assert _same(rotations, exp_rot) and _same(eulers, exp_eulers) and _same(mstep, exp_mstep)
    assert rotations.dtype == np.float32 and rotations.shape == (N_ROT, 3, 3)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pass_without_perturbation_keeps_the_grid_matrices(dtype):
    rotations, eulers, mstep = iteration_loop._exact_local_fine_grid(
        healpix_order=ORDER, angular_sampling_deg=ANGULAR_SAMPLING, random_perturbation=None, dtype=dtype
    )
    grid_rot, grid_eulers = _fake_grid(ORDER, dtype=dtype)
    assert _same(rotations, grid_rot) and _same(eulers, grid_eulers)
    _, _, exp_mstep = apply_relion_rotation_perturbation_to_eulers(
        _canonical_eulers(ORDER), 0.0, ANGULAR_SAMPLING, return_mstep_rotations=True
    )
    assert _same(mstep, exp_mstep)


def test_reused_grid_keeps_its_own_mstep_rotations():
    mstep = np.repeat(np.eye(3, dtype=np.float32)[None], N_ROT, axis=0)
    eulers = np.zeros((N_ROT, 3), dtype=np.float32)
    assert iteration_loop._local_search_mstep_rotations(mstep, eulers, ORDER) is mstep


@pytest.mark.parametrize("n_rows", [N_ROT, N_ROT + 3])
def test_reused_grid_without_mstep_rotations_rebuilds_them_from_source_angles(n_rows):
    eulers = (np.arange(3 * n_rows, dtype=np.float64).reshape(n_rows, 3) / 3.0).astype(np.float32)
    got = iteration_loop._local_search_mstep_rotations(None, eulers, ORDER)
    _, _, expected = apply_relion_rotation_perturbation_to_eulers(
        iteration_loop._relion_mstep_source_eulers(eulers, ORDER), 0.0, ANGULAR_SAMPLING, return_mstep_rotations=True
    )
    assert _same(got, expected) and got.shape == (n_rows, 3, 3)


def test_controller_builds_local_search_grids_through_the_owners():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    assert source.count("_exact_local_fine_grid(") == 2
    assert source.count("_local_search_mstep_rotations(") == 2
    assert source.count("relion_local_pass1_current_size(") == 2
    # The remaining coarse-size arithmetic belongs to the adaptive (non-local) pass-1 sizing of each pass.
    assert source.count("compute_coarse_image_size(") == 2
    assert source.count("clamp_relion_coarse_image_size(") == 2
