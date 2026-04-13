import numpy as np
import pytest

from recovar.data_io._index_utils import load_index_like

pytestmark = pytest.mark.unit


def test_load_index_like_accepts_npy(tmp_path):
    indices = np.array([1, 3, 5], dtype=np.int32)
    path = tmp_path / "indices.npy"
    np.save(path, indices)

    out = load_index_like(str(path))

    np.testing.assert_array_equal(out, indices)


def test_load_index_like_accepts_multiarray_npz(tmp_path):
    half0 = np.array([0, 2], dtype=np.int32)
    half1 = np.array([1, 3], dtype=np.int32)
    path = tmp_path / "halfsets.npz"
    np.savez(path, halfset_0=half0, halfset_1=half1)

    out = load_index_like(str(path))

    assert len(out) == 2
    np.testing.assert_array_equal(out[0], half0)
    np.testing.assert_array_equal(out[1], half1)


def test_load_index_like_accepts_txt(tmp_path):
    path = tmp_path / "indices.txt"
    np.savetxt(path, np.array([2, 4, 6], dtype=np.int32), fmt="%d")

    out = load_index_like(str(path))

    np.testing.assert_array_equal(out, np.array([2, 4, 6], dtype=np.int64))
