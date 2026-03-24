import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from recovar.data_io import image_loader
from recovar.data_io import starfile
from recovar import utils
from recovar.data_io import image_backends as cryo_dataset

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
    parsed_none = loader._parse_indices(None)
    assert parsed_none.dtype == np.int32
    assert np.all(parsed_none == np.arange(8))

    parsed_int = loader._parse_indices(3)
    assert parsed_int.dtype == np.int32
    assert np.all(parsed_int == np.array([3]))

    parsed_slice = loader._parse_indices(slice(2, 6, 2))
    assert parsed_slice.dtype == np.int32
    assert np.all(parsed_slice == np.array([2, 4]))

    parsed_array = loader._parse_indices(np.array([1, 5], dtype=np.int32))
    assert parsed_array.dtype == np.int32
    assert np.all(parsed_array == np.array([1, 5]))

    parsed_bool = loader._parse_indices(np.array([True, False, True, False, False, False, False, False]))
    assert parsed_bool.dtype == np.int32
    assert np.all(parsed_bool == np.array([0, 2]))


def test_parse_indices_rejects_non_1d_arrays_and_bad_boolean_masks():
    loader = _DummyLoader(n=4, D=2)
    with pytest.raises(ValueError, match="Indices array must be 1D"):
        loader._parse_indices(np.array([[0, 1], [2, 3]], dtype=np.int32))

    with pytest.raises(ValueError, match="Boolean indices must be a 1D mask"):
        loader._parse_indices(np.array([[True, False], [False, True]], dtype=bool))

    with pytest.raises(ValueError, match="must match number of images"):
        loader._parse_indices(np.array([True, False], dtype=bool))


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


def test_mrc_loader_duplicate_indices_preserve_order_and_duplicates(tmp_path):
    data = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    mrc_path = tmp_path / "particles.mrcs"
    utils.write_mrc(str(mrc_path), data)

    loader = image_loader.MRCLoader(str(mrc_path), lazy=True)
    req = np.array([4, 2, 4, 0], dtype=np.int32)
    out = loader.get(req)
    np.testing.assert_allclose(out, data[req])


def test_mrc_loader_empty_request_returns_empty_batch(tmp_path):
    data = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    mrc_path = tmp_path / "particles.mrcs"
    utils.write_mrc(str(mrc_path), data)

    loader = image_loader.MRCLoader(str(mrc_path), lazy=True)
    out = loader.get(np.array([], dtype=np.int32))
    assert out.shape == (0, 4, 4)
    assert out.dtype == data.dtype


def test_mrc_loader_deduplicates_disk_reads_for_duplicate_indices(monkeypatch, tmp_path):
    data = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
    mrc_path = tmp_path / "particles.mrcs"
    utils.write_mrc(str(mrc_path), data)

    loader = image_loader.MRCLoader(str(mrc_path), lazy=True)
    req = np.array([5, 2, 5, 0], dtype=np.int32)  # unique non-sequential: [0,2,5]

    orig_fromfile = np.fromfile
    calls = {"n": 0}

    def _counting_fromfile(*args, **kwargs):
        calls["n"] += 1
        return orig_fromfile(*args, **kwargs)

    monkeypatch.setattr(np, "fromfile", _counting_fromfile)
    out = loader.get(req)

    np.testing.assert_allclose(out, data[req])
    # Random-access path should read once per unique index, not per request element.
    assert calls["n"] == 3


def test_mrc_loader_deduplicates_disk_reads_when_all_indices_are_same(monkeypatch, tmp_path):
    data = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
    mrc_path = tmp_path / "particles.mrcs"
    utils.write_mrc(str(mrc_path), data)

    loader = image_loader.MRCLoader(str(mrc_path), lazy=True)
    req = np.array([3, 3, 3, 3], dtype=np.int32)

    orig_fromfile = np.fromfile
    calls = {"n": 0}

    def _counting_fromfile(*args, **kwargs):
        calls["n"] += 1
        return orig_fromfile(*args, **kwargs)

    monkeypatch.setattr(np, "fromfile", _counting_fromfile)
    out = loader.get(req)

    np.testing.assert_allclose(out, data[req])
    # Single contiguous read for one unique index.
    assert calls["n"] == 1


def test_mrc_loader_constructor_accepts_boolean_subset_mask(tmp_path):
    data = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    mrc_path = tmp_path / "particles.mrcs"
    utils.write_mrc(str(mrc_path), data)

    keep = np.array([True, False, True, False, True], dtype=bool)
    loader = image_loader.MRCLoader(str(mrc_path), indices=keep, lazy=True)
    out = loader.get(np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_allclose(out, data[[0, 2, 4]])


def test_mrc_loader_constructor_rejects_bad_subset_masks(tmp_path):
    data = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    mrc_path = tmp_path / "particles.mrcs"
    utils.write_mrc(str(mrc_path), data)

    with pytest.raises(ValueError, match="boolean mask must be 1D"):
        image_loader.MRCLoader(str(mrc_path), indices=np.array([[True, False, True, False, True]], dtype=bool), lazy=True)

    with pytest.raises(ValueError, match="boolean mask length.*must match total size"):
        image_loader.MRCLoader(str(mrc_path), indices=np.array([True, False], dtype=bool), lazy=True)


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


def test_multi_mrc_loader_constructor_accepts_boolean_subset_mask(tmp_path):
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
    mask = np.array([False, True, False, True, True], dtype=bool)
    loader = image_loader.MultiMRCLoader(df, indices=mask, lazy=True, max_threads=2)
    out = loader.get(np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_allclose(out[0], a[1])
    np.testing.assert_allclose(out[1], b[0])
    np.testing.assert_allclose(out[2], b[1])


def test_multi_mrc_loader_constructor_rejects_bad_subset_masks(tmp_path):
    a = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    a_path = tmp_path / "a.mrcs"
    utils.write_mrc(str(a_path), a)
    df = pd.DataFrame(
        {
            "mrc_file": [str(a_path), str(a_path), str(a_path)],
            "mrc_index": [0, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="boolean mask must be 1D"):
        image_loader.MultiMRCLoader(df, indices=np.array([[True, False, True]], dtype=bool), lazy=True, max_threads=1)

    with pytest.raises(ValueError, match="boolean mask length.*must match total size"):
        image_loader.MultiMRCLoader(df, indices=np.array([True, False], dtype=bool), lazy=True, max_threads=1)

    with pytest.raises(ValueError, match="must be 1D"):
        image_loader.MultiMRCLoader(df, indices=np.array([[0, 1]], dtype=np.int32), lazy=True, max_threads=1)


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


def test_multi_mrc_loader_empty_request_returns_empty_batch(tmp_path):
    a = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    a_path = tmp_path / "a.mrcs"
    utils.write_mrc(str(a_path), a)
    df = pd.DataFrame({"mrc_file": [str(a_path), str(a_path), str(a_path)], "mrc_index": [0, 1, 2]})
    loader = image_loader.MultiMRCLoader(df, lazy=True, max_threads=1)
    out = loader.get(np.array([], dtype=np.int32))
    assert out.shape == (0, 4, 4)
    assert out.dtype == a.dtype


def test_multi_mrc_loader_indices_only_load_selected_files(tmp_path):
    a = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    a_path = tmp_path / "a.mrcs"
    utils.write_mrc(str(a_path), a)

    missing_path = tmp_path / "missing.mrcs"  # intentionally not created
    df = pd.DataFrame(
        {
            "mrc_file": [str(a_path), str(a_path), str(missing_path)],
            "mrc_index": [0, 1, 0],
        }
    )

    # Select only rows that point to the existing file.
    loader = image_loader.MultiMRCLoader(df, indices=np.array([1, 0], dtype=np.int32), lazy=True, max_threads=2)
    out = loader.get(np.array([0, 1], dtype=np.int32))
    np.testing.assert_allclose(out[0], a[1])
    np.testing.assert_allclose(out[1], a[0])


def test_multi_mrc_loader_empty_selection_raises_clear_error(tmp_path):
    a = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    a_path = tmp_path / "a.mrcs"
    utils.write_mrc(str(a_path), a)
    df = pd.DataFrame({"mrc_file": [str(a_path), str(a_path)], "mrc_index": [0, 1]})
    with pytest.raises(ValueError, match="No images selected"):
        image_loader.MultiMRCLoader(df, indices=np.array([], dtype=np.int32), lazy=True, max_threads=1)


def test_multi_mrc_loader_rejects_out_of_range_mrc_indices_at_construction(tmp_path):
    a = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    a_path = tmp_path / "a.mrcs"
    utils.write_mrc(str(a_path), a)
    df = pd.DataFrame(
        {
            "mrc_file": [str(a_path), str(a_path)],
            "mrc_index": [0, 2],  # second index is out of range for 2-image stack
        }
    )
    with pytest.raises(ValueError, match="out of range"):
        image_loader.MultiMRCLoader(df, lazy=True, max_threads=1)


def test_multi_mrc_loader_rejects_negative_mrc_index_values_at_construction(tmp_path):
    a = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    a_path = tmp_path / "a.mrcs"
    utils.write_mrc(str(a_path), a)
    df = pd.DataFrame(
        {
            "mrc_file": [str(a_path), str(a_path)],
            "mrc_index": [0, -1],
        }
    )
    with pytest.raises(ValueError, match="must be non-negative"):
        image_loader.MultiMRCLoader(df, lazy=True, max_threads=1)


def test_multi_mrc_loader_rejects_noninteger_mrc_index_values_at_construction(tmp_path):
    a = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    a_path = tmp_path / "a.mrcs"
    utils.write_mrc(str(a_path), a)
    df = pd.DataFrame(
        {
            "mrc_file": [str(a_path), str(a_path)],
            "mrc_index": [0.0, 1.0],  # float metadata should be rejected
        }
    )
    with pytest.raises(ValueError, match="must be integers"):
        image_loader.MultiMRCLoader(df, lazy=True, max_threads=1)


def test_tilt_series_dataset_strip_prefix_with_tiny_real_files(tmp_path):
    from helpers import tiny_synthetic

    files = tiny_synthetic.make_tiny_loader_files(tmp_path / "tiny", grid_size=8, n_images=6, n_particles=3)
    bad_prefix = "/unmounted/prefix"
    mrcs_name = (tmp_path / "tiny" / "particles.mrcs").name

    sf = starfile.StarFile.load(files["particles_star"])
    sf.df["_rlnImageName"] = [f"{i+1}@{bad_prefix}/{mrcs_name}" for i in range(files["n_images"])]
    prefixed_star = tmp_path / "prefixed.star"
    starfile.write_star(str(prefixed_star), data=sf.df)

    # Without strip_prefix the prefixed path is unresolved.
    with pytest.raises(Exception):
        cryo_dataset.TiltSeriesDataset(str(prefixed_star), datadir=str(tmp_path / "tiny"), lazy=True, tilt_file_option="relion5")

    ds = cryo_dataset.TiltSeriesDataset(
        str(prefixed_star),
        datadir=str(tmp_path / "tiny"),
        strip_prefix=bad_prefix + "/",
        lazy=True,
        tilt_file_option="relion5",
        random_tilts=False,
        num_tilts=1,
    )
    assert len(ds) == 3
    batches = list(ds.get_image_subset_generator(batch_size=2, subset_indices=np.array([5, 1, 4], dtype=np.int32)))
    got = []
    for _imgs, _pidx, tidx in batches:
        got.extend(np.array(tidx).reshape(-1).tolist())
    assert got == [5, 1, 4]


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


def test_star_loader_rejects_missing_rlnimagename_column(tmp_path):
    df = pd.DataFrame({"_rlnDefocusU": [10000, 11000]})
    star_path = tmp_path / "particles.star"
    starfile.write_star(str(star_path), data=df)
    with pytest.raises(ValueError, match="_rlnImageName"):
        image_loader.StarLoader(str(star_path), lazy=True)


def test_star_loader_rejects_nonnumeric_image_index_tokens(tmp_path):
    data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    rel_dir = tmp_path / "rel"
    rel_dir.mkdir()
    mrc_path = rel_dir / "stack.mrcs"
    utils.write_mrc(str(mrc_path), data)

    # Numeric-looking prefix is required; nonnumeric token should fail clearly.
    df_bad = pd.DataFrame({"_rlnImageName": [f"one@{mrc_path.name}", f"2@{mrc_path.name}"]})
    bad_star = tmp_path / "bad_nonnumeric_token.star"
    starfile.write_star(str(bad_star), data=df_bad)
    with pytest.raises(ValueError, match="index part is not an integer"):
        image_loader.StarLoader(str(bad_star), datadir=str(rel_dir), lazy=True)


def test_star_loader_rejects_malformed_image_name_entries(tmp_path):
    data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    rel_dir = tmp_path / "rel"
    rel_dir.mkdir()
    mrc_path = rel_dir / "stack.mrcs"
    utils.write_mrc(str(mrc_path), data)

    # Missing '@' separator.
    df_bad = pd.DataFrame({"_rlnImageName": [f"1{mrc_path.name}", f"2@{mrc_path.name}"]})
    bad_star = tmp_path / "bad.star"
    starfile.write_star(str(bad_star), data=df_bad)
    with pytest.raises(ValueError, match="Malformed _rlnImageName"):
        image_loader.StarLoader(str(bad_star), datadir=str(rel_dir), lazy=True)

    # Zero index (must be >= 1 in STAR convention).
    df_zero = pd.DataFrame({"_rlnImageName": [f"0@{mrc_path.name}", f"1@{mrc_path.name}"]})
    zero_star = tmp_path / "zero.star"
    starfile.write_star(str(zero_star), data=df_zero)
    with pytest.raises(ValueError, match="indices must be >= 1"):
        image_loader.StarLoader(str(zero_star), datadir=str(rel_dir), lazy=True)


def test_star_loader_rejects_non_integer_image_indices(tmp_path):
    data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    rel_dir = tmp_path / "rel"
    rel_dir.mkdir()
    mrc_path = rel_dir / "stack.mrcs"
    utils.write_mrc(str(mrc_path), data)

    df_bad = pd.DataFrame({"_rlnImageName": [f"a@{mrc_path.name}", f"2@{mrc_path.name}"]})
    bad_star = tmp_path / "bad_non_integer.star"
    starfile.write_star(str(bad_star), data=df_bad)

    with pytest.raises(ValueError, match="index part is not an integer"):
        image_loader.StarLoader(str(bad_star), datadir=str(rel_dir), lazy=True)


def test_star_loader_rejects_image_indices_out_of_range_for_stack(tmp_path):
    data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    rel_dir = tmp_path / "rel"
    rel_dir.mkdir()
    mrc_path = rel_dir / "stack.mrcs"
    utils.write_mrc(str(mrc_path), data)

    # Third image requested from a two-image stack.
    df_bad = pd.DataFrame({"_rlnImageName": [f"1@{mrc_path.name}", f"3@{mrc_path.name}"]})
    bad_star = tmp_path / "bad_out_of_range.star"
    starfile.write_star(str(bad_star), data=df_bad)

    with pytest.raises(ValueError, match="out of range"):
        image_loader.StarLoader(str(bad_star), datadir=str(rel_dir), lazy=True)


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


def test_cryosparc_loader_supports_byte_blob_paths(tmp_path):
    data = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    mrc_path = tmp_path / "stack.mrcs"
    utils.write_mrc(str(mrc_path), data)

    cs_dtype = np.dtype([("blob/idx", np.int32), ("blob/path", "S64")])
    cs = np.zeros(3, dtype=cs_dtype)
    cs["blob/idx"] = np.array([0, 2, 1], dtype=np.int32)
    cs["blob/path"] = np.array([b">stack.mrcs", b">stack.mrcs", b">stack.mrcs"], dtype="S64")
    cs_path = tmp_path / "particles_bytes.cs"
    with open(cs_path, "wb") as f:
        np.save(f, cs)

    loader = image_loader.CryoSparcLoader(str(cs_path), datadir=str(tmp_path), lazy=True)
    out = loader.get(np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_allclose(out[0], data[0])
    np.testing.assert_allclose(out[1], data[2])
    np.testing.assert_allclose(out[2], data[1])


def test_cryosparc_loader_rejects_missing_required_fields(tmp_path):
    cs_dtype = np.dtype([("blob/path", "U64")])
    cs = np.zeros(2, dtype=cs_dtype)
    cs["blob/path"] = np.array([">a.mrcs", ">a.mrcs"])
    cs_path = tmp_path / "bad.cs"
    with open(cs_path, "wb") as f:
        np.save(f, cs)
    with pytest.raises(ValueError, match="blob/idx"):
        image_loader.CryoSparcLoader(str(cs_path), datadir=str(tmp_path), lazy=True)


def test_cryosparc_loader_rejects_blob_indices_out_of_range_for_stack(tmp_path):
    data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    mrc_path = tmp_path / "stack.mrcs"
    utils.write_mrc(str(mrc_path), data)

    cs_dtype = np.dtype([("blob/idx", np.int32), ("blob/path", "U64")])
    cs = np.zeros(2, dtype=cs_dtype)
    cs["blob/idx"] = np.array([0, 2], dtype=np.int32)  # second index out of range
    cs["blob/path"] = np.array([">stack.mrcs", ">stack.mrcs"])
    cs_path = tmp_path / "bad_out_of_range.cs"
    with open(cs_path, "wb") as f:
        np.save(f, cs)

    with pytest.raises(ValueError, match="out of range"):
        image_loader.CryoSparcLoader(str(cs_path), datadir=str(tmp_path), lazy=True)


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
    out = image_loader.ImageLoader.from_file(
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


def test_load_images_dispatches_to_expected_loader(monkeypatch):
    sentinels = {
        "mrc": object(),
        "star": object(),
        "txt": object(),
        "cs": object(),
    }
    calls = {}

    def _fake_mrc(filepath, indices, lazy, skip_staging=False):
        calls["mrc"] = (filepath, indices, lazy)
        return sentinels["mrc"]

    def _fake_star(filepath, indices, datadir, lazy, max_threads, strip_prefix, skip_staging=False):
        calls["star"] = (filepath, indices, datadir, lazy, max_threads, strip_prefix)
        return sentinels["star"]

    def _fake_cs(filepath, indices, datadir, lazy, max_threads, strip_prefix=None, skip_staging=False):
        calls["cs"] = (filepath, indices, datadir, lazy, max_threads, strip_prefix)
        return sentinels["cs"]

    class _FakeMulti:
        @staticmethod
        def from_txt(filepath, indices, lazy, max_threads, skip_staging=False):
            calls["txt"] = (filepath, indices, lazy, max_threads)
            return sentinels["txt"]

    monkeypatch.setattr(image_loader, "MRCLoader", _fake_mrc)
    monkeypatch.setattr(image_loader, "StarLoader", _fake_star)
    monkeypatch.setattr(image_loader, "CryoSparcLoader", _fake_cs)
    monkeypatch.setattr(image_loader, "MultiMRCLoader", _FakeMulti)

    idx = np.array([0, 2], dtype=np.int32)
    assert image_loader.load_images("particles.mrcs", indices=idx, lazy=False) is sentinels["mrc"]
    assert image_loader.load_images("particles.mrc", indices=idx, lazy=True) is sentinels["mrc"]
    assert image_loader.load_images(
        "particles.star",
        indices=idx,
        datadir="/tmp/data",
        lazy=True,
        max_threads=8,
        strip_prefix="old/",
    ) is sentinels["star"]
    assert image_loader.load_images("particles.txt", indices=idx, lazy=True, max_threads=4) is sentinels["txt"]
    assert image_loader.load_images("particles.cs", indices=idx, datadir="/tmp/cs", lazy=False, max_threads=2) is sentinels["cs"]

    np.testing.assert_array_equal(calls["mrc"][1], idx)
    np.testing.assert_array_equal(calls["star"][1], idx)
    np.testing.assert_array_equal(calls["txt"][1], idx)
    np.testing.assert_array_equal(calls["cs"][1], idx)
    assert calls["star"][2:] == ("/tmp/data", True, 8, "old/")
    assert calls["txt"][2:] == (True, 4)
    assert calls["cs"][2:] == ("/tmp/cs", False, 2, None)


def test_cryosparc_loader_rejects_negative_blob_indices(tmp_path):
    cs_dtype = np.dtype([("blob/idx", np.int32), ("blob/path", "U64")])
    cs = np.zeros(2, dtype=cs_dtype)
    cs["blob/idx"] = np.array([0, -1], dtype=np.int32)
    cs["blob/path"] = np.array([">stack.mrcs", ">stack.mrcs"])
    cs_path = tmp_path / "bad_negative.cs"
    with open(cs_path, "wb") as f:
        np.save(f, cs)

    with pytest.raises(ValueError, match="negative blob/idx"):
        image_loader.CryoSparcLoader(str(cs_path), datadir=str(tmp_path), lazy=True)


def test_cryosparc_loader_resolves_relative_datadir_against_cs_file(tmp_path):
    data = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    run_dir = tmp_path / "job" / "run"
    data_dir = tmp_path / "job" / "data"
    run_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    mrc_path = data_dir / "stack.mrcs"
    utils.write_mrc(str(mrc_path), data)

    cs_dtype = np.dtype([("blob/idx", np.int32), ("blob/path", "U64")])
    cs = np.zeros(3, dtype=cs_dtype)
    cs["blob/idx"] = np.array([2, 0, 1], dtype=np.int32)
    cs["blob/path"] = np.array([">stack.mrcs", ">stack.mrcs", ">stack.mrcs"])
    cs_path = run_dir / "particles.cs"
    with open(cs_path, "wb") as f:
        np.save(f, cs)

    loader = image_loader.CryoSparcLoader(str(cs_path), datadir="../data", lazy=True)
    out = loader.get(np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_allclose(out[0], data[2])
    np.testing.assert_allclose(out[1], data[0])
    np.testing.assert_allclose(out[2], data[1])


def test_load_images_star_from_simulator_tiny_tilt_preserves_duplicates_and_order(tmp_path):
    from helpers import tiny_synthetic

    files = tiny_synthetic.make_tiny_tilt_loader_files_from_simulator(
        tmp_path / "sim_tiny_tilt",
        grid_size=8,
        n_images=18,
        n_tilts=3,
        n_volumes=4,
    )
    datadir = str(Path(files["particles_star"]).parent)
    req = np.array([5, 1, 5, 0], dtype=np.int32)

    star_loader = image_loader.load_images(files["particles_star"], indices=None, datadir=datadir, lazy=True)
    mrc_loader = image_loader.MRCLoader(files["particles_mrcs"], lazy=True)
    got = star_loader.get(req)
    expected = mrc_loader.get(req)
    np.testing.assert_allclose(got, expected)


def test_load_images_star_with_constructor_indices_from_simulator_preserves_local_order_and_duplicates(tmp_path):
    from helpers import tiny_synthetic

    files = tiny_synthetic.make_tiny_tilt_loader_files_from_simulator(
        tmp_path / "sim_tiny_tilt_subset",
        grid_size=8,
        n_images=18,
        n_tilts=3,
        n_volumes=4,
    )
    datadir = str(Path(files["particles_star"]).parent)
    selected_global = np.array([7, 2, 7, 1], dtype=np.int32)
    local_request = np.array([3, 0, 2, 1], dtype=np.int32)

    loader = image_loader.load_images(
        files["particles_star"],
        indices=selected_global,
        datadir=datadir,
        lazy=True,
    )
    got = loader.get(local_request)

    mrc_loader = image_loader.MRCLoader(files["particles_mrcs"], lazy=True)
    expected = mrc_loader.get(selected_global[local_request])
    np.testing.assert_allclose(got, expected)


def test_load_images_star_with_boolean_constructor_mask_from_simulator(tmp_path):
    from helpers import tiny_synthetic

    files = tiny_synthetic.make_tiny_tilt_loader_files_from_simulator(
        tmp_path / "sim_tiny_tilt_mask",
        grid_size=8,
        n_images=18,
        n_tilts=3,
        n_volumes=4,
    )
    datadir = str(Path(files["particles_star"]).parent)
    mask = np.zeros(files["n_images"], dtype=bool)
    mask[[7, 2, 1]] = True
    local_request = np.array([2, 0, 1], dtype=np.int32)

    loader = image_loader.load_images(
        files["particles_star"],
        indices=mask,
        datadir=datadir,
        lazy=True,
    )
    got = loader.get(local_request)

    mrc_loader = image_loader.MRCLoader(files["particles_mrcs"], lazy=True)
    # mask selects global indices [1,2,7] in ascending order; local_request [2,0,1]
    # should map to global [7,1,2].
    expected = mrc_loader.get(np.array([7, 1, 2], dtype=np.int32))
    np.testing.assert_allclose(got, expected)


def test_load_images_star_rejects_wrong_length_boolean_constructor_mask(tmp_path):
    from helpers import tiny_synthetic

    files = tiny_synthetic.make_tiny_tilt_loader_files_from_simulator(
        tmp_path / "sim_tiny_tilt_badmask",
        grid_size=8,
        n_images=18,
        n_tilts=3,
        n_volumes=4,
    )
    datadir = str(Path(files["particles_star"]).parent)

    with pytest.raises(ValueError, match="boolean mask length.*must match total size"):
        image_loader.load_images(
            files["particles_star"],
            indices=np.array([True, False, True], dtype=bool),
            datadir=datadir,
            lazy=True,
        )


def test_multi_mrc_loader_mrc_to_mrcs_extension_fallback(tmp_path):
    """When file_map references .mrc but the file on disk is .mrcs, the loader should find it."""
    data = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    mrcs_path = tmp_path / "stack.mrcs"
    utils.write_mrc(str(mrcs_path), data)

    # file_map references .mrc (without the 's'), but the actual file is .mrcs
    mrc_ref = str(tmp_path / "stack.mrc")
    df = pd.DataFrame({
        'mrc_file': [mrc_ref, mrc_ref, mrc_ref],
        'mrc_index': [0, 1, 2],
    })

    loader = image_loader.MultiMRCLoader(df, lazy=True)
    out = loader.get(np.array([0, 2], dtype=np.int32))
    assert out.shape == (2, 4, 4)
    np.testing.assert_allclose(out[0], data[0])
    np.testing.assert_allclose(out[1], data[2])


def test_multi_mrc_loader_mrcs_to_mrc_extension_fallback(tmp_path):
    """When file_map references .mrcs but the file on disk is .mrc, the loader should find it."""
    data = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    mrc_path = tmp_path / "stack.mrc"
    utils.write_mrc(str(mrc_path), data)

    # file_map references .mrcs, but the actual file is .mrc
    mrcs_ref = str(tmp_path / "stack.mrcs")
    df = pd.DataFrame({
        'mrc_file': [mrcs_ref, mrcs_ref, mrcs_ref],
        'mrc_index': [0, 1, 2],
    })

    loader = image_loader.MultiMRCLoader(df, lazy=True)
    out = loader.get(np.array([1], dtype=np.int32))
    assert out.shape == (1, 4, 4)
    np.testing.assert_allclose(out[0], data[1])


def test_cryosparc_loader_mrc_mrcs_extension_fallback(tmp_path):
    """CryoSparcLoader should find .mrcs files when CS references .mrc."""
    data = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    mrcs_path = tmp_path / "stack.mrcs"
    utils.write_mrc(str(mrcs_path), data)

    # CS file references stack.mrc (no 's'), but file on disk is stack.mrcs
    cs_dtype = np.dtype([("blob/idx", np.int32), ("blob/path", "U64")])
    cs = np.zeros(3, dtype=cs_dtype)
    cs["blob/idx"] = np.array([0, 1, 2], dtype=np.int32)
    cs["blob/path"] = np.array([">stack.mrc", ">stack.mrc", ">stack.mrc"])
    cs_path = tmp_path / "particles.cs"
    with open(cs_path, "wb") as f:
        np.save(f, cs)

    loader = image_loader.CryoSparcLoader(str(cs_path), datadir=str(tmp_path), lazy=True)
    out = loader.get(np.array([0, 2], dtype=np.int32))
    assert out.shape == (2, 4, 4)
    np.testing.assert_allclose(out[0], data[0])
    np.testing.assert_allclose(out[1], data[2])


def test_cryosparc_loader_strip_prefix_with_extension_fallback(tmp_path):
    """CryoSparcLoader with strip-prefix + .mrc->.mrcs fallback."""
    data = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    mrcs_path = tmp_path / "stack.mrcs"
    utils.write_mrc(str(mrcs_path), data)

    # CS file has paths like J3/imported/stack.mrc
    cs_dtype = np.dtype([("blob/idx", np.int32), ("blob/path", "U64")])
    cs = np.zeros(3, dtype=cs_dtype)
    cs["blob/idx"] = np.array([0, 1, 2], dtype=np.int32)
    cs["blob/path"] = np.array([
        ">J3/imported/stack.mrc",
        ">J3/imported/stack.mrc",
        ">J3/imported/stack.mrc",
    ])
    cs_path = tmp_path / "particles.cs"
    with open(cs_path, "wb") as f:
        np.save(f, cs)

    loader = image_loader.CryoSparcLoader(
        str(cs_path),
        datadir=str(tmp_path),
        strip_prefix="J3/imported",
        lazy=True,
    )
    out = loader.get(np.array([0, 1, 2], dtype=np.int32))
    assert out.shape == (3, 4, 4)
    np.testing.assert_allclose(out, data)


def test_error_hint_shows_raw_metadata_path(tmp_path):
    """Error hint for missing files should suggest strip-prefix based on raw metadata path."""
    cs_dtype = np.dtype([("blob/idx", np.int32), ("blob/path", "U64")])
    cs = np.zeros(1, dtype=cs_dtype)
    cs["blob/idx"] = np.array([0], dtype=np.int32)
    cs["blob/path"] = np.array([">J3/imported/nonexistent.mrc"])
    cs_path = tmp_path / "particles.cs"
    with open(cs_path, "wb") as f:
        np.save(f, cs)

    with pytest.raises(FileNotFoundError, match="--strip-prefix J3/imported"):
        image_loader.CryoSparcLoader(str(cs_path), datadir=str(tmp_path), lazy=True)


def test_swap_mrc_ext_helper():
    """Test the _swap_mrc_ext helper function."""
    assert image_loader._swap_mrc_ext("/path/to/file.mrc") == "/path/to/file.mrcs"
    assert image_loader._swap_mrc_ext("/path/to/file.mrcs") == "/path/to/file.mrc"
    assert image_loader._swap_mrc_ext("/path/to/file.txt") is None
    assert image_loader._swap_mrc_ext("/path/to/file.star") is None
