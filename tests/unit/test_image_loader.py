import numpy as np
import pytest

from recovar import image_loader

pytestmark = pytest.mark.unit


class _DummyLoader(image_loader.ImageLoader):
    def __init__(self, n=6, D=4, dtype=np.float32):
        super().__init__(n, D, dtype=dtype)
        self._store = np.arange(n * D * D, dtype=dtype).reshape(n, D, D)

    def _load(self, indices: np.ndarray) -> np.ndarray:
        return self._store[indices]


def test_load_images_rejects_unknown_extension():
    with pytest.raises(ValueError, match="Unsupported format"):
        image_loader.load_images("particles.unknown")


def test_parse_indices_int_slice_array_and_bool():
    loader = _DummyLoader(n=8, D=2)
    assert np.all(loader._parse_indices(None) == np.arange(8))
    assert np.all(loader._parse_indices(3) == np.array([3]))
    assert np.all(loader._parse_indices(slice(2, 6, 2)) == np.array([2, 4]))
    assert np.all(loader._parse_indices(np.array([1, 5], dtype=np.int32)) == np.array([1, 5]))
    assert np.all(loader._parse_indices(np.array([True, False, True, False, False, False, False, False])) == np.array([0, 2]))


def test_parse_indices_rejects_non_integer_ndarray():
    loader = _DummyLoader(n=4, D=2)
    with pytest.raises(TypeError, match="integer or bool dtype"):
        loader._parse_indices(np.array([0.0, 1.0], dtype=np.float32))


def test_get_and_cached_loading_and_iter_batches():
    loader = _DummyLoader(n=5, D=3, dtype=np.float32)
    out = loader.get([1, 3])
    assert out.shape == (2, 3, 3)
    # Load cache and ensure get uses it.
    loader.load_all()
    out2 = loader.get(np.array([0, 4], dtype=np.int32))
    assert np.allclose(out2, loader._store[[0, 4]])

    batches = list(loader.iter_batches(batch_size=2))
    assert len(batches) == 3
    idx0, imgs0 = batches[0]
    assert np.all(idx0 == np.array([0, 1]))
    assert imgs0.shape == (2, 3, 3)
