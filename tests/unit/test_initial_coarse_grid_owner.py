"""The refinement controller materializes its coarse trial grids through one owner.

``_initial_coarse_grids`` builds the first exhaustive grid (sealed capture,
caller translation table, or RELION translation grid) and
``_relion_base_translation_grid`` is the only unperturbed RELION translation
grid construction in ``iteration_loop``.
"""

from __future__ import annotations

import inspect
import logging

import jax.numpy as jnp
import numpy as np
import pytest

import recovar.em.refinement.iteration_loop as iteration_loop
from recovar.em.diagnostics.relion_replay import _sealed_sampling_base_grids
from recovar.em.sampling import _translation_grid_for_class_count

pytestmark = pytest.mark.unit

SEALED = {
    "directions_ipix": np.asarray([7, 19, 503], dtype=np.int64),
    "rot_angles_deg": np.asarray([10.0, 20.0, 30.0]),
    "tilt_angles_deg": np.asarray([40.0, 50.0, 60.0]),
    "psi_angles_deg": np.asarray([0.0, 90.0]),
    "translations_x_angstrom": np.asarray([-2.0, 0.0, 2.0]),
    "translations_y_angstrom": np.asarray([0.0, 1.0, 0.0]),
    "healpix_order_original": 3,
}


def _fake_rotation_grid(order, dtype=np.float32):
    n = 3 * (int(order) + 1)
    rotations = np.repeat(np.eye(3, dtype=dtype)[None], n, axis=0)
    return rotations, np.arange(3 * n, dtype=dtype).reshape(n, 3)


def _same(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def _initial_grids(**overrides):
    kwargs = dict(
        healpix_order=3,
        sealed_sampling_state=None,
        translations=None,
        init_healpix_order=3,
        init_translation_range=4.25,
        init_translation_step=1.416667,
        n_classes=1,
        voxel_size=2.0,
        log=logging.getLogger("test_initial_coarse_grid_owner"),
    )
    kwargs.update(overrides)
    return iteration_loop._initial_coarse_grids(**kwargs)


@pytest.mark.parametrize("n_classes", [1, 4])
def test_base_translation_grid_is_host_float64_in_source_units(n_classes):
    grid = iteration_loop._relion_base_translation_grid(4.25, 1.416667, n_classes=n_classes, voxel_size=2.0)
    expected = _translation_grid_for_class_count(4.25, 1.416667, n_classes=n_classes, source_units_per_pixel=2.0)
    assert isinstance(grid, np.ndarray) and grid.dtype == np.float64
    assert _same(grid, expected.astype(np.float64))


def test_base_translation_grid_falls_back_to_pixel_units_without_a_voxel_size():
    fallback = iteration_loop._relion_base_translation_grid(5.0, 1.0, n_classes=1, voxel_size=0.0)
    pixel_units = iteration_loop._relion_base_translation_grid(5.0, 1.0, n_classes=1, voxel_size=1.0)
    assert _same(fallback, pixel_units)


def test_sealed_state_supplies_the_initial_grid(caplog):
    caplog.set_level(logging.INFO, logger="test_initial_coarse_grid_owner")
    grids = _initial_grids(sealed_sampling_state=SEALED)
    rotations, eulers, translations = _sealed_sampling_base_grids(SEALED, voxel_size_angstrom=2.0, dtype=np.float32)
    assert _same(grids.rotations, rotations) and _same(grids.rotation_eulers, eulers)
    assert isinstance(grids.translations, jnp.ndarray) and _same(grids.translations, translations)
    assert grids.base_translations.dtype == np.float64
    assert _same(grids.base_translations, np.asarray(translations, dtype=np.float64))
    assert grids.healpix_order == 3 and isinstance(grids.healpix_order, int)
    assert "Frozen-boundary v3 directly materialized 6 Euler rows and 3 translations" in caplog.text


def test_sealed_state_must_sit_at_the_initialized_order():
    with pytest.raises(ValueError, match="sealed=3 init=2"):
        _initial_grids(sealed_sampling_state=SEALED, init_healpix_order=2)


def test_relion_translation_grid_pairs_with_the_canonical_rotation_grid(monkeypatch):
    monkeypatch.setattr(iteration_loop, "_relion_rotation_grid_float32", _fake_rotation_grid)
    grids = _initial_grids(n_classes=4)
    rotations, eulers = _fake_rotation_grid(3, dtype=iteration_loop._dense_global_scoring_dtype())
    assert _same(grids.rotations, rotations) and _same(grids.rotation_eulers, eulers)
    expected = iteration_loop._relion_base_translation_grid(4.25, 1.416667, n_classes=4, voxel_size=2.0)
    assert _same(grids.base_translations, expected)
    assert isinstance(grids.translations, jnp.ndarray)
    assert _same(grids.translations, jnp.asarray(expected, dtype=iteration_loop._dense_global_scoring_dtype()))
    assert grids.healpix_order == 3


def test_caller_translation_table_is_kept_as_the_base_grid(monkeypatch):
    monkeypatch.setattr(iteration_loop, "_relion_rotation_grid_float32", _fake_rotation_grid)
    table = np.asarray([[0.5, -1.0], [0.0, 0.0]], dtype=np.float32)
    grids = _initial_grids(translations=table)
    assert grids.base_translations.dtype == np.float64 and _same(grids.base_translations, table.astype(np.float64))
    assert _same(grids.translations, jnp.asarray(table, dtype=iteration_loop._dense_global_scoring_dtype()))


def test_controller_materializes_coarse_grids_through_the_owners():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    assert source.count("_initial_coarse_grids(") == 1
    assert "_sealed_sampling_base_grids(" not in source
    assert "_translation_grid_for_class_count(" not in source
    assert source.count("_relion_base_translation_grid(") == 6
