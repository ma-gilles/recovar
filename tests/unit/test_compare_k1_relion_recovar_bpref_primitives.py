from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from scripts.compare_k1_relion_recovar_bpref_primitives import (
    _load_live_initial_sigma2,
    _load_selection,
)


def _write_relion_real_2d(path, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float64)
    path.write_bytes(struct.pack("<ii", *array.shape) + array.tobytes())


@pytest.mark.unit
def test_load_live_initial_sigma2_accepts_lossless_npy_and_relion_bin(tmp_path) -> None:
    expected = np.asarray([0.25, 0.5, 1.0], dtype=np.float64)
    npy_path = tmp_path / "sigma2.npy"
    bin_path = tmp_path / "sigma2_noise.bin"
    np.save(npy_path, expected)
    _write_relion_real_2d(bin_path, expected[None, :])

    np.testing.assert_array_equal(_load_live_initial_sigma2(npy_path), expected)
    np.testing.assert_array_equal(_load_live_initial_sigma2(bin_path), expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "values",
    [
        np.asarray([0.25, np.nan], dtype=np.float64),
        np.asarray([0.25, 0.0], dtype=np.float64),
        np.asarray([0.25, -0.5], dtype=np.float64),
    ],
)
def test_load_live_initial_sigma2_rejects_invalid_values(tmp_path, values) -> None:
    path = tmp_path / "sigma2.npy"
    np.save(path, values)

    with pytest.raises(ValueError):
        _load_live_initial_sigma2(path)


@pytest.mark.unit
def test_load_live_initial_sigma2_rejects_multirow_relion_bin(tmp_path) -> None:
    path = tmp_path / "sigma2_noise.bin"
    _write_relion_real_2d(path, np.ones((2, 3), dtype=np.float64))

    with pytest.raises(ValueError, match="exactly one spectrum row"):
        _load_live_initial_sigma2(path)


@pytest.mark.unit
def test_load_selection_requires_production_preprocess_geometry(tmp_path) -> None:
    path = tmp_path / "selection.json"
    expected = {
        "physical_image_size": 256,
        "particle_diameter_ang": 200.0,
        "width_mask_edge_px": 5,
        "targets": [{"stack_index_one_based": 38595}],
    }
    path.write_text(json.dumps(expected))

    selection, image_size, particle_diameter_ang, width_mask_edge_px = _load_selection(path)

    assert selection == expected
    assert image_size == 256
    assert particle_diameter_ang == 200.0
    assert width_mask_edge_px == 5


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("physical_image_size", 0, "physical image size"),
        ("particle_diameter_ang", 0.0, "particle diameter"),
        ("width_mask_edge_px", 0, "mask-edge width"),
    ],
)
def test_load_selection_rejects_invalid_preprocess_geometry(tmp_path, field, value, message) -> None:
    selection = {
        "physical_image_size": 256,
        "particle_diameter_ang": 200.0,
        "width_mask_edge_px": 5,
        "targets": [],
    }
    selection[field] = value
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))

    with pytest.raises(ValueError, match=message):
        _load_selection(path)
