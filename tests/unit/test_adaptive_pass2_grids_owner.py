"""The K=1 and K-class dense routes materialize RELION's two-pass trial grids through one owner."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from recovar.em.dense import half_scoring
from recovar.em.helpers.oversampling import build_adaptive_pass2_grids
from recovar.em.sampling import apply_relion_translation_perturbation, rotation_grid_size

pytestmark = pytest.mark.unit

ORDER = 1
N_ROT = rotation_grid_size(ORDER)
BASE = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)


def _same(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


@pytest.mark.parametrize("oversampling", [0, 1])
def test_owner_returns_the_builder_grids_and_perturbed_phase_source(oversampling):
    rot = np.repeat(np.eye(3, dtype=np.float32)[None], N_ROT, axis=0)
    grids = half_scoring._adaptive_pass2_grids(
        rot, BASE.astype(np.float32), BASE, healpix_order=ORDER, adaptive_oversampling=oversampling,
        translation_step=1.0, random_perturbation=0.25, coarse_rotation_ids=None,
    )
    expected = build_adaptive_pass2_grids(rot, BASE.astype(np.float32), BASE, ORDER, oversampling, 1.0, 0.25, return_mstep_rotations=True)
    for got, exp in zip(grids[:7], expected):
        assert _same(got, exp)
    assert _same(grids.coarse_translation_phase_source, apply_relion_translation_perturbation(BASE, 0.25, 1.0))
    assert grids.n_fine_translations == int(expected[3].shape[0]) and isinstance(grids.n_fine_translations, int)


def test_zero_oversampling_keeps_the_coarse_rotation_grid():
    rot = np.repeat(np.eye(3, dtype=np.float32)[None], N_ROT, axis=0)
    grids = half_scoring._adaptive_pass2_grids(
        rot, BASE.astype(np.float32), BASE, healpix_order=ORDER, adaptive_oversampling=0,
        translation_step=1.0, random_perturbation=0.0, coarse_rotation_ids=None,
    )
    assert _same(grids.fine_rotations, rot) and np.array_equal(grids.rotation_parent_map, np.arange(N_ROT))


def test_both_dense_routes_use_the_owner():
    source = inspect.getsource(half_scoring._score_half_dense)
    assert source.count("pass2_grids = _adaptive_pass2_grids(") == 2
    assert source.count("fine_rotations_for_pose = _adaptive_pass2_grids(") == 1
    assert "build_adaptive_pass2_grids(" not in source
