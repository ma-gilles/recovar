import numpy as np
import pandas as pd
import pytest

from recovar import image_loader
from recovar import starfile
from recovar import utils

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
    # Compatibility alias.
    chunk0 = next(loader.chunks(chunksize=2))
    np.testing.assert_array_equal(chunk0[0], np.array([0, 1]))


def test_mrc_loader_reads_real_file_sequential_and_random(tmp_path):
    data = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    mrc_path = tmp_path / "particles.mrcs"
    utils.write_mrc(str(mrc_path), data)

    loader = image_loader.MRCLoader(str(mrc_path), lazy=True)
    seq = loader.get(np.array([0, 1, 2], dtype=np.int32))
    rnd = loader.get(np.array([4, 2, 0], dtype=np.int32))
    assert seq.shape == (3, 4, 4)
    assert rnd.shape == (3, 4, 4)
    np.testing.assert_allclose(seq, data[[0, 1, 2]])
    np.testing.assert_allclose(rnd, data[[4, 2, 0]])


def test_mrc_loader_rejects_non_square_images(tmp_path):
    import mrcfile

    bad = np.zeros((3, 4, 5), dtype=np.float32)
    mrc_path = tmp_path / "bad.mrcs"
    with mrcfile.new(str(mrc_path), overwrite=True) as m:
        m.set_data(bad)
    with pytest.raises(ValueError, match="Non-square images not supported"):
        image_loader.MRCLoader(str(mrc_path), lazy=True)


def test_multi_mrc_loader_from_txt_and_batches(tmp_path):
    a = (np.arange(3 * 4 * 4, dtype=np.float32) + 10).reshape(3, 4, 4)
    b = (np.arange(2 * 4 * 4, dtype=np.float32) + 100).reshape(2, 4, 4)
    a_path = tmp_path / "a.mrcs"
    b_path = tmp_path / "b.mrcs"
    utils.write_mrc(str(a_path), a)
    utils.write_mrc(str(b_path), b)

    txt_path = tmp_path / "files.txt"
    txt_path.write_text(f"{a_path.name}\n{b_path.name}\n")

    loader = image_loader.MultiMRCLoader.from_txt(str(txt_path), lazy=True, max_threads=2)
    # Expect concatenation of all images from a then b
    assert len(loader) == 5
    imgs = loader.get(np.array([0, 2, 3, 4], dtype=np.int32))
    assert imgs.shape == (4, 4, 4)
    np.testing.assert_allclose(imgs[0], a[0])
    np.testing.assert_allclose(imgs[1], a[2])
    np.testing.assert_allclose(imgs[2], b[0])
    np.testing.assert_allclose(imgs[3], b[1])


def test_multi_mrc_loader_indices_filtering_preserves_order(tmp_path):
    a = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    b = (100 + np.arange(2 * 4 * 4, dtype=np.float32)).reshape(2, 4, 4)
    a_path = tmp_path / "a.mrcs"
    b_path = tmp_path / "b.mrcs"
    utils.write_mrc(str(a_path), a)
    utils.write_mrc(str(b_path), b)

    df = pd.DataFrame(
        {
            "mrc_file": [str(a_path), str(a_path), str(a_path), str(b_path), str(b_path)],
            "mrc_index": [0, 1, 2, 0, 1],
        }
    )
    loader = image_loader.MultiMRCLoader(df, indices=np.array([4, 1, 3], dtype=np.int32), lazy=True, max_threads=2)
    out = loader.get(np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_allclose(out[0], b[1])
    np.testing.assert_allclose(out[1], a[1])
    np.testing.assert_allclose(out[2], b[0])


def test_multi_mrc_loader_get_with_duplicate_indices(tmp_path):
    a = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    b = (100 + np.arange(2 * 4 * 4, dtype=np.float32)).reshape(2, 4, 4)
    a_path = tmp_path / "a.mrcs"
    b_path = tmp_path / "b.mrcs"
    utils.write_mrc(str(a_path), a)
    utils.write_mrc(str(b_path), b)

    df = pd.DataFrame(
        {
            "mrc_file": [str(a_path), str(a_path), str(a_path), str(b_path), str(b_path)],
            "mrc_index": [0, 1, 2, 0, 1],
        }
    )
    loader = image_loader.MultiMRCLoader(df, lazy=True, max_threads=2)
    req = np.array([4, 1, 4, 0], dtype=np.int32)
    out = loader.get(req)
    assert out.shape == (4, 4, 4)
    np.testing.assert_allclose(out[0], b[1])
    np.testing.assert_allclose(out[1], a[1])
    np.testing.assert_allclose(out[2], b[1])  # duplicate preserved
    np.testing.assert_allclose(out[3], a[0])


def test_star_loader_and_strip_prefix(tmp_path):
    data = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    rel_dir = tmp_path / "rel"
    rel_dir.mkdir()
    mrc_path = rel_dir / "stack.mrcs"
    utils.write_mrc(str(mrc_path), data)

    df = pd.DataFrame(
        {
            "_rlnImageName": [
                "1@prefix/stack.mrcs",
                "2@prefix/stack.mrcs",
                "3@prefix/stack.mrcs",
            ]
        }
    )
    star_path = tmp_path / "particles.star"
    starfile.write_star(str(star_path), data=df)

    loader = image_loader.StarLoader(
        str(star_path),
        datadir=str(rel_dir),
        strip_prefix="prefix/",
        lazy=True,
    )
    out = loader.get(np.array([0, 2], dtype=np.int32))
    assert out.shape == (2, 4, 4)
    np.testing.assert_allclose(out[0], data[0])
    np.testing.assert_allclose(out[1], data[2])

    with pytest.raises(ValueError, match="strip_prefix"):
        image_loader.StarLoader(str(star_path), datadir=str(rel_dir), strip_prefix="does_not_match/", lazy=True)


def test_star_loader_uses_star_parent_when_datadir_missing(tmp_path):
    data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    mrc_path = tmp_path / "stack.mrcs"
    utils.write_mrc(str(mrc_path), data)
    df = pd.DataFrame({"_rlnImageName": [f"1@{mrc_path.name}", f"2@{mrc_path.name}"]})
    star_path = tmp_path / "particles.star"
    starfile.write_star(str(star_path), data=df)
    loader = image_loader.StarLoader(str(star_path), lazy=True)
    out = loader.get(np.array([1], dtype=np.int32))
    np.testing.assert_allclose(out[0], data[1])


def test_cryosparc_loader_reads_structured_cs(tmp_path):
    data = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    mrc_path = tmp_path / "stack.mrcs"
    utils.write_mrc(str(mrc_path), data)

    cs_dtype = np.dtype([("blob/idx", np.int32), ("blob/path", "U64")])
    cs = np.zeros(3, dtype=cs_dtype)
    cs["blob/idx"] = np.array([0, 1, 2], dtype=np.int32)
    cs["blob/path"] = np.array([">stack.mrcs", ">stack.mrcs", ">stack.mrcs"])
    cs_path = tmp_path / "particles.cs"
    with open(cs_path, "wb") as f:
        np.save(f, cs)

    loader = image_loader.CryoSparcLoader(str(cs_path), datadir=str(tmp_path), lazy=True)
    out = loader.get(np.array([1, 2], dtype=np.int32))
    assert out.shape == (2, 4, 4)
    np.testing.assert_allclose(out[0], data[1])
    np.testing.assert_allclose(out[1], data[2])


def test_image_loader_from_file_alias_and_dtype_preserved():
    loader = _DummyLoader(n=3, D=2, dtype=np.float32)
    out = loader.images(np.array([0, 2], dtype=np.int32), require_contiguous=True)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, loader._store[[0, 2]])


def test_parse_indices_scalar_ndarray_and_bounds_errors():
    loader = _DummyLoader(n=4, D=2)
    np.testing.assert_array_equal(loader._parse_indices(np.array(2, dtype=np.int64)), np.array([2]))
    with pytest.raises(IndexError, match="out of range"):
        loader._parse_indices(9)
    with pytest.raises(IndexError, match="out of range"):
        loader._parse_indices(np.array(-1, dtype=np.int64))


def test_image_source_from_file_delegates_to_load_images(monkeypatch):
    sentinel = object()
    captured = {}

    def _fake_load(filepath, indices=None, datadir="", lazy=True, max_threads=1, strip_prefix=None):
        captured["args"] = (filepath, indices, datadir, lazy, max_threads, strip_prefix)
        return sentinel

    monkeypatch.setattr(image_loader, "load_images", _fake_load)
    out = image_loader.ImageSource.from_file(
        "particles.star",
        lazy=False,
        indices=np.array([0, 3], dtype=np.int32),
        datadir="/tmp/x",
        max_threads=8,
        strip_prefix="abc/",
    )
    assert out is sentinel
    assert captured["args"][0] == "particles.star"
    np.testing.assert_array_equal(captured["args"][1], np.array([0, 3], dtype=np.int32))
    assert captured["args"][2:] == ("/tmp/x", False, 8, "abc/")


def test_image_loader_parse_indices_rejects_bad_type():
    loader = _DummyLoader(n=4, D=2)
    with pytest.raises(TypeError, match="Cannot index with type"):
        loader._parse_indices({"bad": "type"})
