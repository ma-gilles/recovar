import mrcfile
import numpy as np
import pytest

from recovar import utils


@pytest.mark.unit
def test_write_mrc_stack_roundtrip_float32(tmp_path):
    stack = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    path = tmp_path / "particles.5.mrcs"

    utils.write_mrc_stack(path, stack, voxel_size=1.25, chunk_size=2)

    with mrcfile.open(path, permissive=False) as mrc:
        assert mrc.data.shape == stack.shape
        assert mrc.data.dtype == np.float32
        assert int(mrc.header.ispg) == 0
        assert int(mrc.header.mz) == 1
        assert mrc.voxel_size.x == pytest.approx(1.25)
        assert mrc.voxel_size.y == pytest.approx(1.25)
        assert mrc.voxel_size.z == pytest.approx(1.25)
        np.testing.assert_array_equal(np.asarray(mrc.data), stack)


@pytest.mark.unit
def test_write_mrc_stack_preserves_requested_float16_dtype(tmp_path):
    stack = np.linspace(-1.0, 1.0, num=2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)
    path = tmp_path / "particles.4.mrcs"

    utils.write_mrc_stack(path, stack, voxel_size=2.0, dtype=np.float16, chunk_size=1)

    with mrcfile.open(path, permissive=False) as mrc:
        assert mrc.data.shape == stack.shape
        assert mrc.data.dtype == np.float16
        assert int(mrc.header.mode) == 12
        np.testing.assert_array_equal(np.asarray(mrc.data), stack)
