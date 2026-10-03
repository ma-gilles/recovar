"""``RECOVAR_PREREAD_IMAGES``: MRC stacks read once into host memory (RELION ``--preread_images``)."""

import numpy as np
import pytest

from recovar.data_io import image_loader
from recovar.data_io.image_loader import MRCLoader, MultiMRCLoader


def _write_stack(tmp_path, n=12, d=8, seed=0):
    import mrcfile

    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, d, d)).astype(np.float32)
    path = tmp_path / "stack.mrcs"
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.set_image_stack()
    return path, data


def test_preread_flag_parsing(monkeypatch):
    monkeypatch.delenv(image_loader.PREREAD_IMAGES_ENV, raising=False)
    assert image_loader.preread_images_requested() is False
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    assert image_loader.preread_images_requested() is True
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "yes")
    with pytest.raises(ValueError):
        image_loader.preread_images_requested()


def test_preread_serves_selected_rows_from_memory(tmp_path, monkeypatch):
    path, data = _write_stack(tmp_path)
    selection = np.asarray([9, 2, 5, 0, 11], dtype=np.int64)
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    monkeypatch.setenv("RECOVAR_CACHE_DIR", "")
    preread = MRCLoader(str(path), indices=selection, skip_staging=True)
    assert preread._cached is not None
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "0")
    lazy = MRCLoader(str(path), indices=selection, skip_staging=True)
    assert lazy._cached is None
    request = np.asarray([3, 0, 3, 4, 1], dtype=np.int64)
    np.testing.assert_array_equal(preread.get(request), lazy.get(request))
    np.testing.assert_array_equal(preread.get(request), data[selection][request])
    # Multi-file wrappers call ``_load`` directly; the preread cache must serve that path too.
    np.testing.assert_array_equal(preread._load(request), lazy._load(request))
    np.testing.assert_array_equal(preread.get(None), data[selection])


def test_preread_respects_host_memory_cap(tmp_path, monkeypatch):
    path, _ = _write_stack(tmp_path)
    monkeypatch.setenv(image_loader.PREREAD_IMAGES_ENV, "1")
    monkeypatch.setenv(image_loader.PREREAD_MAX_GB_ENV, "0.000001")
    monkeypatch.setenv("RECOVAR_CACHE_DIR", "")
    loader = MRCLoader(str(path), skip_staging=True)
    assert loader._cached is None
    assert loader.get(np.asarray([1, 2])).shape == (2, 8, 8)
