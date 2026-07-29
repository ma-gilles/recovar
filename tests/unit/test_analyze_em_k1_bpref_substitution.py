from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_em_k1_bpref_substitution import (
    load_relion_raw,
    relion_raw_to_recovar_full,
)


def _write_raw(path: Path, values: np.ndarray, *, rank: int = 1, call: int = 1):
    zsize, ysize, xsize = values.shape
    radius = zsize // 2
    header = np.asarray(
        [rank, call, xsize, ysize, zsize, 0, -radius, -radius],
        dtype=np.int64,
    )
    with path.open("wb") as stream:
        header.tofile(stream)
        values.tofile(stream)


@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
def test_load_relion_raw_round_trip(tmp_path, dtype):
    values = np.arange(75, dtype=np.float64).reshape(5, 5, 3)
    values = values.astype(dtype)
    path = tmp_path / "raw.bin"
    _write_raw(path, values)

    header, loaded = load_relion_raw(path, value_dtype=dtype)

    assert header.tolist() == [1, 1, 3, 5, 5, 0, -2, -2]
    np.testing.assert_array_equal(loaded, values)


def test_load_relion_raw_fails_closed_on_truncated_payload(tmp_path):
    values = np.ones((5, 5, 3), dtype=np.float64)
    path = tmp_path / "raw.bin"
    _write_raw(path, values)
    path.write_bytes(path.read_bytes()[:-8])

    with pytest.raises(ValueError, match="payload"):
        load_relion_raw(path, value_dtype=np.float64)


def test_relion_raw_to_recovar_full_maps_axes_units_and_hermitian_partner():
    raw_data = np.zeros((5, 5, 3), dtype=np.complex128)
    raw_weight = np.zeros((5, 5, 3), dtype=np.float64)
    # RELION coordinates k=-1, i=+2, j=+1.
    raw_data[1, 4, 1] = 3.0 + 4.0j
    raw_weight[1, 4, 1] = 7.0

    data, weight = relion_raw_to_recovar_full(
        raw_data,
        raw_weight,
        grid_size=2,
    )
    data = data.reshape((5, 5, 5))
    weight = weight.reshape((5, 5, 5))

    # RECOVAR axes are [j, i, k], centered at index 2.
    assert data[3, 4, 1] == pytest.approx((3.0 + 4.0j) / 4.0)
    assert weight[3, 4, 1] == pytest.approx(7.0 / 16.0)
    # Hermitian partner coordinates are (-j, -i, -k).
    assert data[1, 0, 3] == pytest.approx((3.0 - 4.0j) / 4.0)
    assert weight[1, 0, 3] == pytest.approx(7.0 / 16.0)
