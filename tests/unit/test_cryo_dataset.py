from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from recovar import cryo_dataset

pytestmark = pytest.mark.unit


class _DummySource:
    def __init__(self, n=6, D=8):
        self.n = n
        self.D = D
        self._store = np.arange(n * D * D, dtype=np.float32).reshape(n, D, D)

    def images(self, index, require_contiguous=False):
        _ = require_contiguous
        if isinstance(index, (int, np.integer)):
            return self._store[int(index)]
        if isinstance(index, slice):
            idx = np.arange(*index.indices(self.n))
            return self._store[idx]
        return self._store[np.asarray(index)]


def test_particle_image_dataset_basic_getitem_and_preprocess(monkeypatch):
    monkeypatch.setattr(cryo_dataset.ImageSource, "from_file", lambda *args, **kwargs: _DummySource(n=4, D=8))
    ds = cryo_dataset.ParticleImageDataset("dummy.mrcs", lazy=True, invert_data=True)

    imgs, p_idx, t_idx = ds[1]
    assert imgs.shape == (1, 8, 8)
    assert p_idx == 1 and t_idx == 1

    processed = ds.process_images(imgs, apply_image_mask=False)
    assert processed.shape[0] == 1
    assert processed.dtype == np.complex64


def test_tiltseries_parse_particle_tilt_and_reverse_map(monkeypatch):
    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g2", "g1", "g2", "g1", "g3"],
        }
    )
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))
    p2t, t2p = cryo_dataset.TiltSeriesDataset.parse_particle_tilt("dummy.star")
    assert len(p2t) == 3
    # Canonical sorted groups: g1,g2,g3
    assert np.all(p2t[0] == np.array([1, 3]))
    assert np.all(p2t[1] == np.array([0, 2]))
    assert np.all(p2t[2] == np.array([4]))
    assert t2p[1] == 0 and t2p[0] == 1 and t2p[4] == 2


def test_tiltseries_parse_micrograph_tilt_mapping(monkeypatch):
    df = pd.DataFrame(
        {
            "_rlnTiltName": ["tA", "tB", "tA", "tC"],
        }
    )
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))
    groups, reverse = cryo_dataset.TiltSeriesDataset.parse_micrograph_tilt_mapping("dummy.star")
    assert len(groups) == 3
    assert sorted(reverse.keys()) == [0, 1, 2, 3]


def test_tiltseries_parse_micrograph_tilt_mapping_alt_column_name(monkeypatch):
    df = pd.DataFrame(
        {
            "rlnTiltName": ["tA", "tA", "tB"],
        }
    )
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))
    groups, reverse = cryo_dataset.TiltSeriesDataset.parse_micrograph_tilt_mapping("dummy.star")
    assert len(groups) == 2
    assert set(reverse.keys()) == {0, 1, 2}


def test_tiltseries_parse_micrograph_tilt_mapping_missing_column(monkeypatch):
    df = pd.DataFrame({"foo": [1, 2, 3]})
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))
    with pytest.raises(ValueError, match="No tilt name column found"):
        cryo_dataset.TiltSeriesDataset.parse_micrograph_tilt_mapping("dummy.star")


def test_particles_tilts_conversion_helpers_and_canonical_groups():
    p2t = [np.array([0, 1]), np.array([2]), np.array([3, 4])]
    out = cryo_dataset.TiltSeriesDataset.particles_to_tilts(p2t, np.array([2, 0]))
    assert np.all(out == np.array([3, 4, 0, 1]))

    t2p = {0: 5, 1: 2, 7: 2}
    outp = cryo_dataset.TiltSeriesDataset.tilts_to_particles(t2p, np.array([7, 1, 7]))
    assert np.all(outp == np.array([2]))

    df = pd.DataFrame({"_rlnGroupName": ["b", "a", "c", "a"]})
    names = cryo_dataset.get_canonical_group_names(df)
    assert names == ["a", "b", "c"]


def test_tilt_series_to_images_with_subset(monkeypatch):
    monkeypatch.setattr(
        cryo_dataset.TiltSeriesDataset,
        "parse_particle_tilt",
        lambda _p: ([np.array([0, 1]), np.array([2, 3]), np.array([4])], {0: 0}),
    )
    out = cryo_dataset.tilt_series_to_images(np.array([0, 2]), "dummy.star", image_subset=np.array([1, 4]))
    assert np.all(out == np.array([1, 4]))


def test_tilt_series_to_images_with_subset_preserves_order_and_duplicates(monkeypatch):
    monkeypatch.setattr(
        cryo_dataset.TiltSeriesDataset,
        "parse_particle_tilt",
        lambda _p: ([np.array([0, 1]), np.array([2, 3]), np.array([4])], {0: 0}),
    )
    out = cryo_dataset.tilt_series_to_images(
        np.array([1, 1], dtype=np.int32),
        "dummy.star",
        image_subset=np.array([3, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(out, np.array([2, 3, 2, 3], dtype=np.int32))


def test_tilt_series_to_images_empty_and_duplicate_particle_indices(monkeypatch):
    monkeypatch.setattr(
        cryo_dataset.TiltSeriesDataset,
        "parse_particle_tilt",
        lambda _p: ([np.array([0, 1]), np.array([2, 3]), np.array([4])], {0: 0}),
    )
    empty = cryo_dataset.tilt_series_to_images(np.array([], dtype=np.int32), "dummy.star")
    assert empty.shape == (0,)
    assert empty.dtype == np.int32

    dup = cryo_dataset.tilt_series_to_images(np.array([1, 1], dtype=np.int32), "dummy.star")
    np.testing.assert_array_equal(dup, np.array([2, 3, 2, 3]))

    with pytest.raises(IndexError):
        cryo_dataset.tilt_series_to_images(np.array([9], dtype=np.int32), "dummy.star")


def test_image_count_batch_loader_batches_and_padding():
    class _FakeTiltDataset:
        def __init__(self):
            self.particle_groups = {"a": np.array([0, 1]), "b": np.array([2]), "c": np.array([3, 4, 5])}
            self.num_tilts = None

        def __len__(self):
            return 3

        def __getitem__(self, idx):
            groups = [np.array([0, 1]), np.array([2]), np.array([3, 4, 5])]
            t = groups[idx]
            imgs = np.ones((len(t), 8, 8), dtype=np.float32) * (idx + 1)
            return imgs, idx, t

    loader = cryo_dataset.ImageCountBatchLoader(_FakeTiltDataset(), batch_size=4, pad_to_batch=True)
    batches = list(loader)
    assert len(batches) >= 1
    b0_img, b0_pid, b0_tid = batches[0]
    assert b0_img.shape[0] == 4
    assert b0_pid.shape[0] == 4
    assert b0_tid.shape[0] == 4


def test_image_count_batch_loader_handles_oversized_single_particle():
    class _OversizedDataset:
        def __init__(self):
            self.particle_groups = {"a": np.arange(5)}
            self.num_tilts = None

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            imgs = np.ones((5, 8, 8), dtype=np.float32)
            t = np.arange(5, dtype=np.int32)
            return imgs, idx, t

    loader = cryo_dataset.ImageCountBatchLoader(_OversizedDataset(), batch_size=2, pad_to_batch=False)
    batches = list(loader)
    assert len(batches) == 1
    imgs, pidx, tidx = batches[0]
    assert imgs.shape[0] == 5
    np.testing.assert_array_equal(pidx, np.zeros(5, dtype=np.int32))
    np.testing.assert_array_equal(tidx, np.arange(5, dtype=np.int32))


def test_image_count_batch_loader_padding_marks_invalid_entries():
    class _TinyTiltDataset:
        def __init__(self):
            self.particle_groups = {"a": np.array([0]), "b": np.array([1])}
            self.num_tilts = None

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            imgs = np.ones((1, 8, 8), dtype=np.float32)
            return imgs, idx, np.array([idx], dtype=np.int32)

    loader = cryo_dataset.ImageCountBatchLoader(_TinyTiltDataset(), batch_size=3, pad_to_batch=True)
    imgs, pidx, tidx = next(iter(loader))
    assert imgs.shape[0] == 3
    np.testing.assert_array_equal(pidx, np.array([0, 1, -1], dtype=np.int32))
    np.testing.assert_array_equal(tidx, np.array([0, 1, -1], dtype=np.int32))


def test_tiltseries_dataset_getitem_deterministic_selection(monkeypatch):
    class _Source:
        def __init__(self):
            self.n = 6
            self.D = 8
            self._store = np.arange(self.n * self.D * self.D, dtype=np.float32).reshape(self.n, self.D, self.D)

        def images(self, index, require_contiguous=False):
            _ = require_contiguous
            if isinstance(index, (int, np.integer)):
                return self._store[int(index)]
            return self._store[np.asarray(index)]

    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g1", "g1", "g1", "g2", "g2", "g2"],
            "_rlnMicrographPreExposure": [1.0, 3.0, 2.0, 2.0, 5.0, 1.0],
            "_rlnCtfScalefactor": [1, 1, 1, 1, 1, 1],
            "_rlnCtfBfactor": [-1, -2, -3, -4, -5, -6],
        }
    )
    monkeypatch.setattr(cryo_dataset.ImageSource, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))

    ds = cryo_dataset.TiltSeriesDataset("dummy.star", num_tilts=2, random_tilts=False, tilt_file_option="relion5")
    imgs, pidx, selected = ds[0]

    assert pidx == 0
    assert imgs.shape == (2, 8, 8)
    assert selected.shape == (2,)
    # Deterministic order induced by _compute_tilt_ordering implementation.
    np.testing.assert_array_equal(selected, np.array([0, 2]))


def test_tiltseries_dataset_getitem_deterministic_selection_warp(monkeypatch):
    class _Source:
        def __init__(self):
            self.n = 4
            self.D = 8
            self._store = np.arange(self.n * self.D * self.D, dtype=np.float32).reshape(self.n, self.D, self.D)

        def images(self, index, require_contiguous=False):
            _ = require_contiguous
            if isinstance(index, (int, np.integer)):
                return self._store[int(index)]
            return self._store[np.asarray(index)]

    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g1", "g1", "g1", "g2"],
            "_rlnMicrographPreExposure": [1.0, 2.0, 3.0, 4.0],
            "_rlnCtfScalefactor": [1, 1, 1, 1],
            "_rlnCtfBfactor": [-5.0, -1.0, -3.0, -2.0],
        }
    )
    monkeypatch.setattr(cryo_dataset.ImageSource, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))

    ds = cryo_dataset.TiltSeriesDataset("dummy.star", num_tilts=2, random_tilts=False, tilt_file_option="warp")
    _imgs, pidx, selected = ds[0]
    assert pidx == 0
    # Stable expectation from current B-factor ordering implementation.
    np.testing.assert_array_equal(selected, np.array([1, 2]))


def test_tiltseries_get_tilt_returns_single_image_tuple(monkeypatch):
    class _Source:
        def __init__(self):
            self.n = 3
            self.D = 8
            self._store = np.arange(self.n * self.D * self.D, dtype=np.float32).reshape(self.n, self.D, self.D)

        def images(self, index, require_contiguous=False):
            _ = require_contiguous
            if isinstance(index, (int, np.integer)):
                return self._store[int(index)]
            return self._store[np.asarray(index)]

    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g1", "g1", "g2"],
            "_rlnMicrographPreExposure": [1.0, 2.0, 3.0],
            "_rlnCtfScalefactor": [1, 1, 1],
            "_rlnCtfBfactor": [-1.0, -2.0, -3.0],
        }
    )
    monkeypatch.setattr(cryo_dataset.ImageSource, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))
    ds = cryo_dataset.TiltSeriesDataset("dummy.star", num_tilts=None, random_tilts=False, tilt_file_option="relion5")

    img, pidx, tidx = ds.get_tilt(2)
    assert img.shape == (1, 8, 8)
    assert pidx == 2
    assert tidx == 2


def test_tiltseries_dataset_images_mode_batch_size_validation(monkeypatch):
    class _Source:
        def __init__(self):
            self.n = 4
            self.D = 8
            self._store = np.zeros((self.n, self.D, self.D), dtype=np.float32)

        def images(self, index, require_contiguous=False):
            _ = require_contiguous
            if isinstance(index, (int, np.integer)):
                return self._store[int(index)]
            return self._store[np.asarray(index)]

    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g1", "g1", "g1", "g2"],
            "_rlnMicrographPreExposure": [1.0, 2.0, 3.0, 4.0],
            "_rlnCtfScalefactor": [1, 1, 1, 1],
            "_rlnCtfBfactor": [-1, -1, -1, -1],
        }
    )
    monkeypatch.setattr(cryo_dataset.ImageSource, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))

    ds = cryo_dataset.TiltSeriesDataset("dummy.star", num_tilts=None, random_tilts=False, tilt_file_option="relion5")
    # max tilts per particle is 3; batch_size=2 should fail in images mode.
    with pytest.raises(ValueError, match="Batch size"):
        ds.get_dataset_generator(batch_size=2, mode="images")


def test_tiltseries_subset_generator_images_mode_batch_size_validation(monkeypatch):
    class _Source:
        def __init__(self):
            self.n = 5
            self.D = 8
            self._store = np.zeros((self.n, self.D, self.D), dtype=np.float32)

        def images(self, index, require_contiguous=False):
            _ = require_contiguous
            if isinstance(index, (int, np.integer)):
                return self._store[int(index)]
            return self._store[np.asarray(index)]

    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g1", "g1", "g1", "g2", "g2"],
            "_rlnMicrographPreExposure": [1.0, 2.0, 3.0, 4.0, 5.0],
            "_rlnCtfScalefactor": [1, 1, 1, 1, 1],
            "_rlnCtfBfactor": [-1, -1, -1, -1, -1],
        }
    )
    monkeypatch.setattr(cryo_dataset.ImageSource, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))

    ds = cryo_dataset.TiltSeriesDataset("dummy.star", num_tilts=None, random_tilts=False, tilt_file_option="relion5")
    # Subset includes particle g1 with 3 tilts; batch_size=2 should fail for images mode.
    with pytest.raises(ValueError, match="Batch size"):
        ds.get_dataset_subset_generator(batch_size=2, subset_indices=np.array([0], dtype=np.int32), mode="images")


def test_simple_dataloader_forces_batch_size_one(monkeypatch):
    class _Source:
        def __init__(self):
            self.n = 4
            self.D = 8
            self._store = np.zeros((self.n, self.D, self.D), dtype=np.float32)

        def images(self, index, require_contiguous=False):
            _ = require_contiguous
            if isinstance(index, (int, np.integer)):
                return self._store[int(index)]
            return self._store[np.asarray(index)]

    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g1", "g1", "g2", "g2"],
            "_rlnMicrographPreExposure": [1.0, 2.0, 3.0, 4.0],
            "_rlnCtfScalefactor": [1, 1, 1, 1],
            "_rlnCtfBfactor": [-1, -1, -1, -1],
        }
    )
    monkeypatch.setattr(cryo_dataset.ImageSource, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))

    ds = cryo_dataset.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="relion5")
    loader = cryo_dataset.simple_dataloader(ds, batch_size=99)
    assert loader.batch_size == 1


def test_tiltseries_dataset_invalid_mode_raises(monkeypatch):
    class _Source:
        def __init__(self):
            self.n = 2
            self.D = 8
            self._store = np.zeros((2, 8, 8), dtype=np.float32)

        def images(self, index, require_contiguous=False):
            _ = require_contiguous
            if isinstance(index, (int, np.integer)):
                return self._store[int(index)]
            return self._store[np.asarray(index)]

    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g1", "g1"],
            "_rlnMicrographPreExposure": [1.0, 2.0],
            "_rlnCtfScalefactor": [1, 1],
            "_rlnCtfBfactor": [-1, -1],
        }
    )
    monkeypatch.setattr(cryo_dataset.ImageSource, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))
    ds = cryo_dataset.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="relion5")
    with pytest.raises(ValueError, match="Invalid mode"):
        ds.get_dataset_generator(batch_size=2, mode="bad_mode")


def test_tiltseries_dataset_invalid_tilt_file_option_raises(monkeypatch):
    class _Source:
        def __init__(self):
            self.n = 2
            self.D = 8
            self._store = np.zeros((2, 8, 8), dtype=np.float32)

        def images(self, index, require_contiguous=False):
            _ = require_contiguous
            if isinstance(index, (int, np.integer)):
                return self._store[int(index)]
            return self._store[np.asarray(index)]

    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g1", "g1"],
            "_rlnMicrographPreExposure": [1.0, 2.0],
            "_rlnCtfScalefactor": [1, 1],
            "_rlnCtfBfactor": [-1, -1],
        }
    )
    monkeypatch.setattr(cryo_dataset.ImageSource, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))
    with pytest.raises(ValueError, match="Invalid tilt ordering method"):
        cryo_dataset.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="bad")


def test_tiltseries_image_subset_generator_none_matches_full(monkeypatch):
    class _Source:
        def __init__(self):
            self.n = 4
            self.D = 8
            self._store = np.arange(self.n * self.D * self.D, dtype=np.float32).reshape(self.n, self.D, self.D)

        def images(self, index, require_contiguous=False):
            _ = require_contiguous
            if isinstance(index, (int, np.integer)):
                return self._store[int(index)]
            return self._store[np.asarray(index)]

    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g1", "g1", "g2", "g2"],
            "_rlnMicrographPreExposure": [1.0, 2.0, 3.0, 4.0],
            "_rlnCtfScalefactor": [1, 1, 1, 1],
            "_rlnCtfBfactor": [-1, -1, -1, -1],
        }
    )
    monkeypatch.setattr(cryo_dataset.ImageSource, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))
    ds = cryo_dataset.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="relion5")
    full = list(ds.get_image_generator(batch_size=2))
    none_subset = list(ds.get_image_subset_generator(batch_size=2, subset_indices=None))
    assert len(full) == len(none_subset)
    for a, b in zip(full, none_subset):
        np.testing.assert_array_equal(np.array(a[2]), np.array(b[2]))


def test_collate_to_jax_handles_none_scalar_ndarray_and_nested_sequences():
    # ndarray branch
    arr = [np.ones((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32)]
    out = cryo_dataset.collate_to_jax(arr)
    np.testing.assert_array_equal(np.array(out), np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32))

    # scalar branch
    out_scalar = cryo_dataset.collate_to_jax([1, 2, 3])
    np.testing.assert_array_equal(np.array(out_scalar), np.array([1, 2, 3]))

    # none branch
    assert cryo_dataset.collate_to_jax(None) is None

    # tuple/list recursion branch
    nested = [
        (np.ones((1, 1), dtype=np.float32), np.array([5], dtype=np.int32)),
        (np.zeros((1, 1), dtype=np.float32), np.array([7], dtype=np.int32)),
    ]
    out_nested = cryo_dataset.collate_to_jax(nested)
    assert isinstance(out_nested, list)
    np.testing.assert_array_equal(np.array(out_nested[0]), np.array([[1.0], [0.0]], dtype=np.float32))
    np.testing.assert_array_equal(np.array(out_nested[1]), np.array([5, 7], dtype=np.int32))


def test_tiltseries_subset_generator_preserves_subset_order_and_duplicates(monkeypatch):
    class _Source:
        def __init__(self):
            self.n = 6
            self.D = 8
            self._store = np.arange(self.n * self.D * self.D, dtype=np.float32).reshape(self.n, self.D, self.D)

        def images(self, index, require_contiguous=False):
            _ = require_contiguous
            if isinstance(index, (int, np.integer)):
                return self._store[int(index)]
            return self._store[np.asarray(index)]

    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "_rlnMicrographPreExposure": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "_rlnCtfScalefactor": [1, 1, 1, 1, 1, 1],
            "_rlnCtfBfactor": [-1, -2, -1, -2, -1, -2],
        }
    )
    monkeypatch.setattr(cryo_dataset.ImageSource, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))

    ds = cryo_dataset.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="relion5")
    subset_indices = np.array([2, 0, 2], dtype=np.int32)
    loader = ds.get_dataset_subset_generator(batch_size=8, subset_indices=subset_indices, mode="tilt_series")
    batches = list(loader)

    # simple_dataloader uses batch_size=1 for tilt_series mode.
    assert len(batches) == 3
    out_particle_ids = [int(np.array(b[1]).reshape(-1)[0]) for b in batches]
    out_tilt_ids = [int(np.array(b[2]).reshape(-1)[0]) for b in batches]
    assert out_particle_ids == [2, 0, 2]
    # For this deterministic setup with num_tilts=1 and relion5 ordering.
    assert out_tilt_ids == [4, 0, 4]


def test_tiltseries_image_subset_generator_preserves_requested_image_order_and_duplicates(monkeypatch):
    class _Source:
        def __init__(self):
            self.n = 5
            self.D = 8
            self._store = np.arange(self.n * self.D * self.D, dtype=np.float32).reshape(self.n, self.D, self.D)

        def images(self, index, require_contiguous=False):
            _ = require_contiguous
            if isinstance(index, (int, np.integer)):
                return self._store[int(index)]
            return self._store[np.asarray(index)]

    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g1", "g1", "g2", "g2", "g3"],
            "_rlnMicrographPreExposure": [1.0, 2.0, 1.0, 2.0, 1.0],
            "_rlnCtfScalefactor": [1, 1, 1, 1, 1],
            "_rlnCtfBfactor": [-1, -2, -1, -2, -1],
        }
    )
    monkeypatch.setattr(cryo_dataset.ImageSource, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))

    ds = cryo_dataset.TiltSeriesDataset("dummy.star", num_tilts=None, random_tilts=False, tilt_file_option="relion5")
    subset_indices = np.array([3, 1, 3], dtype=np.int32)
    loader = ds.get_image_subset_generator(batch_size=2, subset_indices=subset_indices)
    batches = list(loader)

    got = []
    for _imgs, _pidx, tidx in batches:
        got.extend(np.array(tidx).reshape(-1).tolist())
    assert got == [3, 1, 3]


def test_tiltseries_dataset_ind_subset_preserves_image_tilt_alignment(monkeypatch):
    class _Source:
        def __init__(self, indices=None):
            base_n = 6
            self.D = 8
            full = np.arange(base_n * self.D * self.D, dtype=np.float32).reshape(base_n, self.D, self.D)
            if indices is None:
                self._store = full
            else:
                self._store = full[np.asarray(indices)]
            self.n = self._store.shape[0]

        def images(self, index, require_contiguous=False):
            _ = require_contiguous
            if isinstance(index, (int, np.integer)):
                return self._store[int(index)]
            return self._store[np.asarray(index)]

    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "_rlnMicrographPreExposure": [1.0, 2.0, 1.0, 2.0, 5.0, 6.0],
            "_rlnCtfScalefactor": [1, 1, 1, 1, 1, 1],
            "_rlnCtfBfactor": [-1, -2, -1, -2, -1, -2],
        }
    )
    monkeypatch.setattr(
        cryo_dataset.ImageSource,
        "from_file",
        lambda *_args, **kwargs: _Source(indices=kwargs.get("indices")),
    )
    monkeypatch.setattr(cryo_dataset.starfile.Starfile, "load", lambda _p: SimpleNamespace(df=df))

    # Reordered subset over original row indices.
    subset = np.array([5, 2, 4, 1], dtype=np.int32)
    ds = cryo_dataset.TiltSeriesDataset("dummy.star", ind=subset, num_tilts=1, random_tilts=False, tilt_file_option="relion5")

    # Direct tilt fetch should map to subset-local order: local 0 == original 5.
    img0, _pidx0, tidx0 = ds.get_tilt(0)
    assert int(tidx0) == 0
    np.testing.assert_array_equal(img0[0], ds.source._store[0])

    # For g3 in subset (locals 0 and 2), relion5 ordering picks lower dose first -> local 2.
    imgs, particle_idx, selected = ds[2]  # canonical order: g1, g2, g3
    assert int(particle_idx) == 2
    np.testing.assert_array_equal(selected, np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(imgs[0], ds.source._store[2])
