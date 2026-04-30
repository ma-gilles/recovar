from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from recovar.data_io import image_backends
from recovar.core import fourier_transform_utils, mask
from helpers import tiny_synthetic

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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _DummySource(n=4, D=8))
    ds = image_backends.ParticleImageDataset("dummy.mrcs", lazy=True, invert_data=True)

    imgs, p_idx, t_idx = ds[1]
    assert imgs.shape == (1, 8, 8)
    assert p_idx == 1 and t_idx == 1

    processed = ds.process_images(imgs, apply_image_mask=False)
    assert processed.shape[0] == 1
    assert processed.dtype == np.complex64


def test_particle_image_dataset_process_images_half_uses_native_rfft(monkeypatch):
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _DummySource(n=4, D=8))
    ds = image_backends.ParticleImageDataset("dummy.mrcs", lazy=True, invert_data=False)

    imgs, _p_idx, _t_idx = ds[2]
    expected_half = np.asarray(fourier_transform_utils.get_dft2_real(imgs).reshape((1, -1))).astype(np.complex64)
    processed_half = ds.process_images_half(imgs, apply_image_mask=False)

    np.testing.assert_array_equal(processed_half, expected_half)


def test_particle_image_dataset_relion_background_fill_mask_mode(monkeypatch):
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _DummySource(n=4, D=8))
    ds = image_backends.ParticleImageDataset("dummy.mrcs", lazy=True, invert_data=False)

    imgs, _p_idx, _t_idx = ds[2]
    ds.image_mask = mask.relion_soft_image_mask(
        image_size=ds.image_size,
        pixel_size=1.0,
        particle_diameter_ang=6.0,
        width_mask_edge_px=2.0,
    )
    ds.image_mask_mode = "relion_background_fill"

    processed = ds.process_images(imgs, apply_image_mask=True)
    masked = image_backends._apply_relion_soft_image_mask_numpy(imgs, ds.image_mask)
    expected = image_backends._centered_fft2_numpy(masked).reshape((1, -1)).astype(np.complex64)

    np.testing.assert_allclose(processed, expected, atol=1e-6)


def test_particle_image_dataset_process_images_half_pair_matches_individual_calls(monkeypatch):
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _DummySource(n=4, D=8))
    ds = image_backends.ParticleImageDataset("dummy.mrcs", lazy=True, invert_data=False)
    ds.image_mask = mask.relion_soft_image_mask(
        image_size=ds.image_size,
        pixel_size=1.0,
        particle_diameter_ang=6.0,
        width_mask_edge_px=2.0,
    )
    ds.image_mask_mode = "relion_background_fill"

    imgs, _p_idx, _t_idx = ds[[1, 2]]
    masked_pair, raw_pair = ds.process_images_half_pair(
        imgs,
        apply_image_mask_a=True,
        apply_image_mask_b=False,
    )

    np.testing.assert_allclose(masked_pair, ds.process_images_half(imgs, apply_image_mask=True), atol=1e-6)
    np.testing.assert_allclose(raw_pair, ds.process_images_half(imgs, apply_image_mask=False), atol=1e-6)


def test_particle_image_dataset_subset_generators_preserve_order_and_duplicates(monkeypatch):
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _DummySource(n=5, D=8))
    ds = image_backends.ParticleImageDataset("dummy.mrcs", lazy=True, invert_data=False)

    subset = np.array([3, 1, 3], dtype=np.int32)
    batches = list(ds.get_dataset_subset_generator(batch_size=2, subset_indices=subset))
    got = []
    for _imgs, _pidx, tidx in batches:
        got.extend(np.array(tidx).reshape(-1).tolist())
    assert got == [3, 1, 3]

    image_batches = list(ds.get_image_subset_generator(batch_size=2, subset_indices=subset))
    got_img = []
    for _imgs, _pidx, tidx in image_batches:
        got_img.extend(np.array(tidx).reshape(-1).tolist())
    assert got_img == [3, 1, 3]


def test_particle_image_dataset_subset_generators_accept_boolean_mask(monkeypatch):
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _DummySource(n=5, D=8))
    ds = image_backends.ParticleImageDataset("dummy.mrcs", lazy=True, invert_data=False)

    mask = np.array([False, True, False, True, False], dtype=bool)
    batches = list(ds.get_dataset_subset_generator(batch_size=2, subset_indices=mask))
    got = []
    for _imgs, _pidx, tidx in batches:
        got.extend(np.array(tidx).reshape(-1).tolist())
    assert got == [1, 3]


def test_particle_image_dataset_subset_generators_reject_invalid_masks(monkeypatch):
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _DummySource(n=5, D=8))
    ds = image_backends.ParticleImageDataset("dummy.mrcs", lazy=True, invert_data=False)

    with pytest.raises(ValueError, match="boolean mask must be 1D"):
        list(ds.get_dataset_subset_generator(batch_size=2, subset_indices=np.array([[True, False]], dtype=bool)))
    with pytest.raises(ValueError, match="must match total size"):
        list(ds.get_dataset_subset_generator(batch_size=2, subset_indices=np.array([True, False], dtype=bool)))
    with pytest.raises(IndexError, match="negative"):
        list(ds.get_dataset_subset_generator(batch_size=2, subset_indices=np.array([0, -1], dtype=np.int32)))
    with pytest.raises(IndexError, match="out-of-range"):
        list(ds.get_dataset_subset_generator(batch_size=2, subset_indices=np.array([5], dtype=np.int32)))


def test_tiltseries_parse_particle_tilt_and_reverse_map(monkeypatch):
    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g2", "g1", "g2", "g1", "g3"],
        }
    )
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))
    p2t, t2p = image_backends.TiltSeriesDataset.parse_particle_tilt("dummy.star")
    assert len(p2t) == 3
    # Canonical sorted groups: g1,g2,g3
    assert np.all(p2t[0] == np.array([1, 3]))
    assert np.all(p2t[1] == np.array([0, 2]))
    assert np.all(p2t[2] == np.array([4]))
    assert t2p[1] == 0 and t2p[0] == 1 and t2p[4] == 2


def test_tiltseries_parse_particle_tilt_accepts_boolean_indices(monkeypatch):
    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g2", "g1", "g2", "g1", "g3"],
        }
    )
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))
    mask = np.array([False, True, False, True, True], dtype=bool)
    p2t, t2p = image_backends.TiltSeriesDataset.parse_particle_tilt("dummy.star", indices=mask)

    assert len(p2t) == 2
    np.testing.assert_array_equal(p2t[0], np.array([1, 3], dtype=int))  # g1
    np.testing.assert_array_equal(p2t[1], np.array([4], dtype=int))  # g3
    assert t2p == {1: 0, 3: 0, 4: 1}


def test_tiltseries_parse_particle_tilt_rejects_bad_subset_indices(monkeypatch):
    df = pd.DataFrame(
        {
            "_rlnGroupName": ["g2", "g1", "g2", "g1", "g3"],
        }
    )
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    with pytest.raises(ValueError, match="must match total size"):
        image_backends.TiltSeriesDataset.parse_particle_tilt(
            "dummy.star",
            indices=np.array([True, False], dtype=bool),
        )
    with pytest.raises(TypeError, match="integer indices or boolean mask"):
        image_backends.TiltSeriesDataset.parse_particle_tilt(
            "dummy.star",
            indices=np.array([0.0, 1.0], dtype=np.float32),
        )


def test_tiltseries_parse_particle_tilt_missing_group_column_raises(monkeypatch):
    df = pd.DataFrame({"_rlnImageName": ["1@a.mrcs", "2@a.mrcs"]})
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))
    with pytest.raises(ValueError, match="_rlnGroupName"):
        image_backends.TiltSeriesDataset.parse_particle_tilt("dummy.star")


def test_tiltseries_parse_micrograph_tilt_mapping(monkeypatch):
    df = pd.DataFrame(
        {
            "_rlnTiltName": ["tA", "tB", "tA", "tC"],
        }
    )
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))
    groups, reverse = image_backends.TiltSeriesDataset.parse_micrograph_tilt_mapping("dummy.star")
    assert len(groups) == 3
    assert sorted(reverse.keys()) == [0, 1, 2, 3]


def test_tiltseries_parse_micrograph_tilt_mapping_alt_column_name(monkeypatch):
    df = pd.DataFrame(
        {
            "rlnTiltName": ["tA", "tA", "tB"],
        }
    )
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))
    groups, reverse = image_backends.TiltSeriesDataset.parse_micrograph_tilt_mapping("dummy.star")
    assert len(groups) == 2
    assert set(reverse.keys()) == {0, 1, 2}


def test_tiltseries_parse_micrograph_tilt_mapping_missing_column(monkeypatch):
    df = pd.DataFrame({"foo": [1, 2, 3]})
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))
    with pytest.raises(ValueError, match="No tilt name column found"):
        image_backends.TiltSeriesDataset.parse_micrograph_tilt_mapping("dummy.star")


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

    loader = image_backends._ImageCountBatchLoader(_FakeTiltDataset(), batch_size=4, pad_to_batch=True)
    batches = list(loader)
    assert len(batches) >= 1
    b0_img, b0_pid, b0_tid = batches[0]
    assert b0_img.shape[0] == 4
    assert b0_pid.shape[0] == 4
    assert b0_tid.shape[0] == 4


def test_image_count_batch_loader_rejects_nonpositive_or_noninteger_batch_size():
    class _FakeTiltDataset:
        def __init__(self):
            self.particle_groups = {"a": np.array([0, 1])}
            self.num_tilts = None

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            _ = idx
            imgs = np.ones((2, 8, 8), dtype=np.float32)
            t = np.array([0, 1], dtype=np.int32)
            return imgs, 0, t

    with pytest.raises(ValueError, match="batch_size must be positive"):
        image_backends._ImageCountBatchLoader(_FakeTiltDataset(), batch_size=0, pad_to_batch=False)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        image_backends._ImageCountBatchLoader(_FakeTiltDataset(), batch_size=-3, pad_to_batch=False)
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        image_backends._ImageCountBatchLoader(_FakeTiltDataset(), batch_size=2.5, pad_to_batch=False)


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

    loader = image_backends._ImageCountBatchLoader(_OversizedDataset(), batch_size=2, pad_to_batch=False)
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

    loader = image_backends._ImageCountBatchLoader(_TinyTiltDataset(), batch_size=3, pad_to_batch=True)
    imgs, pidx, tidx = next(iter(loader))
    assert imgs.shape[0] == 3
    np.testing.assert_array_equal(pidx, np.array([0, 1, -1], dtype=np.int32))
    np.testing.assert_array_equal(tidx, np.array([0, 1, -1], dtype=np.int32))


def test_image_count_batch_loader_skips_zero_image_particles():
    class _SparseTiltDataset:
        def __init__(self):
            self.particle_groups = {
                "p0": np.array([], dtype=np.int32),  # 0 selected
                "p1": np.array([10, 11], dtype=np.int32),  # 2 selected
                "p2": np.array([], dtype=np.int32),  # 0 selected
                "p3": np.array([30], dtype=np.int32),  # 1 selected
            }
            self.num_tilts = None

        def __len__(self):
            return 4

        def __getitem__(self, idx):
            tilts = list(self.particle_groups.values())[int(idx)]
            images = np.ones((len(tilts), 8, 8), dtype=np.float32) * (int(idx) + 1)
            return images, int(idx), tilts

    loader = image_backends._ImageCountBatchLoader(_SparseTiltDataset(), batch_size=2, pad_to_batch=False)

    assert loader.total_images == 3
    assert len(loader) == 2

    batches = list(loader)
    assert len(batches) == len(loader)
    np.testing.assert_array_equal(
        [np.asarray(batch[0]).shape[0] for batch in batches],
        np.array([2, 1], dtype=np.int32),
    )
    emitted_particles = np.concatenate([np.asarray(b[1]).reshape(-1) for b in batches], axis=0)
    np.testing.assert_array_equal(emitted_particles, np.array([1, 1, 3], dtype=np.int32))


def test_image_count_batch_loader_all_zero_particles_emits_no_batches():
    class _AllZeroTiltDataset:
        def __init__(self):
            self.particle_groups = {
                "p0": np.array([], dtype=np.int32),
                "p1": np.array([], dtype=np.int32),
            }
            self.num_tilts = None

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            _ = idx
            return (
                np.zeros((0, 8, 8), dtype=np.float32),
                -1,
                np.zeros((0,), dtype=np.int32),
            )

    loader = image_backends._ImageCountBatchLoader(_AllZeroTiltDataset(), batch_size=3, pad_to_batch=True)
    assert loader.total_images == 0
    assert len(loader) == 0
    assert list(loader) == []


def test_image_count_batch_loader_subset_wrapper_preserves_duplicate_parent_mapping():
    class _ParentTiltDataset:
        def __init__(self):
            self._particle_tilts = [
                np.array([10, 11], dtype=np.int32),  # 2 images
                np.array([20, 21, 22], dtype=np.int32),  # 3 images
                np.array([30], dtype=np.int32),  # 1 image
            ]
            self.num_tilts = None

        def __len__(self):
            return len(self._particle_tilts)

        def __getitem__(self, idx):
            tilt_ids = self._particle_tilts[int(idx)]
            images = np.ones((len(tilt_ids), 8, 8), dtype=np.float32) * (idx + 1)
            return images, int(idx), tilt_ids

    parent = _ParentTiltDataset()
    # Exercise Subset path with duplicate/reordered parent indices: [2, 0, 2].
    subset = image_backends._SimpleSubset(parent, [2, 0, 2])
    loader = image_backends._ImageCountBatchLoader(subset, batch_size=3, pad_to_batch=False)

    assert loader.total_images == 4  # 1 + 2 + 1
    assert len(loader) == 2

    batches = list(loader)
    assert len(batches) == len(loader)

    emitted_particles = np.concatenate([np.asarray(b[1]).reshape(-1) for b in batches], axis=0)
    emitted_tilts = np.concatenate([np.asarray(b[2]).reshape(-1) for b in batches], axis=0)

    np.testing.assert_array_equal(emitted_particles, np.array([2, 0, 0, 2], dtype=np.int32))
    np.testing.assert_array_equal(emitted_tilts, np.array([30, 10, 11, 30], dtype=np.int32))


def test_image_count_batch_loader_subset_wrapper_uses_parent_num_tilts_cap():
    class _ParentTiltDataset:
        def __init__(self):
            self._particle_tilts = [
                np.array([0, 1, 2], dtype=np.int32),
                np.array([10, 11], dtype=np.int32),
            ]
            self.num_tilts = 1

        def __len__(self):
            return len(self._particle_tilts)

        def __getitem__(self, idx):
            tilt_ids = self._particle_tilts[int(idx)]
            tilt_ids = tilt_ids[: self.num_tilts]
            images = np.ones((len(tilt_ids), 8, 8), dtype=np.float32)
            return images, int(idx), tilt_ids

    subset = image_backends._SimpleSubset(_ParentTiltDataset(), [0, 1, 0])
    loader = image_backends._ImageCountBatchLoader(subset, batch_size=2, pad_to_batch=False)

    assert loader.total_images == 3
    assert len(loader) == 2
    batches = list(loader)
    assert len(batches) == len(loader)
    emitted_tilts = np.concatenate([np.asarray(b[2]).reshape(-1) for b in batches], axis=0)
    np.testing.assert_array_equal(emitted_tilts, np.array([0, 10, 0], dtype=np.int32))


def test_image_count_batch_loader_nested_subset_wrapper_preserves_mapping_and_duplicates():
    class _ParentTiltDataset:
        def __init__(self):
            self._particle_tilts = [
                np.array([0, 1], dtype=np.int32),  # 2 images
                np.array([10], dtype=np.int32),  # 1 image
                np.array([20, 21, 22], dtype=np.int32),  # 3 images
                np.array([30], dtype=np.int32),  # 1 image
            ]
            self.num_tilts = None

        def __len__(self):
            return len(self._particle_tilts)

        def __getitem__(self, idx):
            tilt_ids = self._particle_tilts[int(idx)]
            images = np.ones((len(tilt_ids), 8, 8), dtype=np.float32) * (int(idx) + 1)
            return images, int(idx), tilt_ids

    parent = _ParentTiltDataset()
    subset_lvl1 = image_backends._SimpleSubset(parent, [3, 1, 0])  # maps local->[3,1,0]
    subset_lvl2 = image_backends._SimpleSubset(subset_lvl1, [2, 0, 2])  # maps to base [0,3,0]

    loader = image_backends._ImageCountBatchLoader(subset_lvl2, batch_size=3, pad_to_batch=False)

    assert loader.total_images == 5  # base[0]=2 + base[3]=1 + base[0]=2
    assert len(loader) == 2

    batches = list(loader)
    assert len(batches) == len(loader)
    emitted_particles = np.concatenate([np.asarray(b[1]).reshape(-1) for b in batches], axis=0)
    emitted_tilts = np.concatenate([np.asarray(b[2]).reshape(-1) for b in batches], axis=0)
    np.testing.assert_array_equal(emitted_particles, np.array([0, 0, 3, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(emitted_tilts, np.array([0, 1, 30, 0, 1], dtype=np.int32))


def test_max_tilts_per_dataset_view_uses_parent_particle_tilts_without_particle_groups():
    class _ParentTiltDataset:
        def __init__(self):
            self._particle_tilts = [
                np.array([0, 1, 2], dtype=np.int32),
                np.array([10, 11], dtype=np.int32),
            ]
            self.num_tilts = 2

        def __len__(self):
            return len(self._particle_tilts)

        def __getitem__(self, idx):
            tilt_ids = self._particle_tilts[int(idx)][: self.num_tilts]
            images = np.ones((len(tilt_ids), 8, 8), dtype=np.float32)
            return images, int(idx), tilt_ids

    subset = image_backends._SimpleSubset(_ParentTiltDataset(), np.array([1, 0, 1], dtype=np.int32))
    assert image_backends._max_tilts_per_dataset_view(subset) == 2


def test_max_tilts_per_dataset_view_respects_zero_num_tilts():
    class _ParentTiltDataset:
        def __init__(self):
            self._particle_tilts = [
                np.array([0, 1, 2], dtype=np.int32),
                np.array([10, 11], dtype=np.int32),
            ]
            self.num_tilts = 0

        def __len__(self):
            return len(self._particle_tilts)

        def __getitem__(self, idx):
            _ = idx
            return np.zeros((0, 8, 8), dtype=np.float32), int(idx), np.zeros((0,), dtype=np.int32)

    subset = image_backends._SimpleSubset(_ParentTiltDataset(), np.array([1, 0, 1], dtype=np.int32))
    assert image_backends._max_tilts_per_dataset_view(subset) == 0


def test_max_tilts_per_dataset_view_raises_when_parent_has_no_group_metadata():
    class _ParentNoGroups:
        num_tilts = None

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            images = np.ones((1, 8, 8), dtype=np.float32)
            return images, int(idx), np.array([int(idx)], dtype=np.int32)

    subset = image_backends._SimpleSubset(_ParentNoGroups(), np.array([0], dtype=np.int32))
    with pytest.raises(AttributeError, match="must expose _particle_tilts"):
        image_backends._max_tilts_per_dataset_view(subset)


def test_image_count_batch_loader_rejects_dataset_without_group_metadata():
    class _NoGroupDataset:
        num_tilts = None

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return np.ones((1, 8, 8), dtype=np.float32), int(idx), np.array([0], dtype=np.int32)

    with pytest.raises(AttributeError, match="must expose _particle_tilts"):
        image_backends._ImageCountBatchLoader(_NoGroupDataset(), batch_size=2, pad_to_batch=False)


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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=2, random_tilts=False, tilt_file_option="relion5")
    imgs, pidx, selected = ds[0]

    assert pidx == 0
    assert imgs.shape == (2, 8, 8)
    assert selected.shape == (2,)
    # Deterministic order induced by _compute_tilt_ordering implementation.
    np.testing.assert_array_equal(selected, np.array([0, 2]))


def test_tiltseries_dataset_random_selection_clamps_when_num_tilts_exceeds_available(monkeypatch):
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
            "_rlnGroupName": ["g1", "g1", "g2", "g2", "g2"],
            "_rlnMicrographPreExposure": [1.0, 2.0, 1.0, 2.0, 3.0],
            "_rlnCtfScalefactor": [1, 1, 1, 1, 1],
            "_rlnCtfBfactor": [-1.0, -2.0, -3.0, -4.0, -5.0],
        }
    )
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    np.random.seed(0)
    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=10, random_tilts=True, tilt_file_option="relion5")

    # g1 has only 2 tilts; should clamp instead of raising from np.random.choice.
    imgs0, pidx0, selected0 = ds[0]
    assert pidx0 == 0
    assert len(selected0) == 2
    assert len(np.unique(selected0)) == 2
    np.testing.assert_allclose(imgs0, _Source()._store[selected0])

    # g2 has 3 tilts; clamp there as well.
    imgs1, pidx1, selected1 = ds[1]
    assert pidx1 == 1
    assert len(selected1) == 3
    assert len(np.unique(selected1)) == 3
    np.testing.assert_allclose(imgs1, _Source()._store[selected1])


def test_tiltseries_dataset_negative_num_tilts_matches_all_tilts(monkeypatch):
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds_all = image_backends.TiltSeriesDataset(
        "dummy.star", num_tilts=None, random_tilts=False, tilt_file_option="relion5"
    )
    ds_neg = image_backends.TiltSeriesDataset(
        "dummy.star", num_tilts=-1, random_tilts=False, tilt_file_option="relion5"
    )

    assert ds_neg.num_tilts is None
    for pidx in range(len(ds_all)):
        _imgs_all, _pid_all, sel_all = ds_all[pidx]
        _imgs_neg, _pid_neg, sel_neg = ds_neg[pidx]
        np.testing.assert_array_equal(sel_neg, sel_all)


def test_tiltseries_dataset_non_integer_num_tilts_raises(monkeypatch):
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    with pytest.raises(TypeError, match="num_tilts must be an integer or None"):
        image_backends.TiltSeriesDataset("dummy.star", num_tilts=1.5, random_tilts=False, tilt_file_option="relion5")


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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=2, random_tilts=False, tilt_file_option="warp")
    _imgs, pidx, selected = ds[0]
    assert pidx == 0
    # Stable expectation from current B-factor ordering implementation.
    np.testing.assert_array_equal(selected, np.array([1, 2]))


def test_particle_image_backend_getitem_returns_single_image_tuple(monkeypatch):
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))
    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=None, random_tilts=False, tilt_file_option="relion5")
    img, pidx, tidx = image_backends.ParticleImageDataset.__getitem__(ds, 2)
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=None, random_tilts=False, tilt_file_option="relion5")
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=None, random_tilts=False, tilt_file_option="relion5")
    # Subset includes particle g1 with 3 tilts; batch_size=2 should fail for images mode.
    with pytest.raises(ValueError, match="Batch size"):
        ds.get_dataset_subset_generator(batch_size=2, subset_indices=np.array([0], dtype=np.int32), mode="images")


def test_tilt_series_generator_forces_batch_size_one(monkeypatch):
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="relion5")
    loader = ds.get_dataset_generator(batch_size=99, mode="tilt_series")
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))
    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="relion5")
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))
    with pytest.raises(ValueError, match="Invalid tilt ordering method"):
        image_backends.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="bad")


def test_tiltseries_dataset_missing_group_column_raises(monkeypatch):
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
            "_rlnMicrographPreExposure": [1.0, 2.0],
            "_rlnCtfScalefactor": [1.0, 1.0],
            "_rlnCtfBfactor": [-1.0, -1.0],
        }
    )
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    with pytest.raises(ValueError, match="_rlnGroupName"):
        image_backends.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="relion5")


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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))
    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="relion5")
    full = list(ds.get_image_generator(batch_size=2))
    none_subset = list(ds.get_image_subset_generator(batch_size=2, subset_indices=None))
    assert len(full) == len(none_subset)
    for a, b in zip(full, none_subset):
        np.testing.assert_array_equal(np.array(a[2]), np.array(b[2]))


def test_collate_to_jax_handles_none_scalar_ndarray_and_nested_sequences():
    # ndarray branch
    arr = [np.ones((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32)]
    out = image_backends._collate_batch_to_jax(arr)
    np.testing.assert_array_equal(np.array(out), np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32))

    # scalar branch
    out_scalar = image_backends._collate_batch_to_jax([1, 2, 3])
    np.testing.assert_array_equal(np.array(out_scalar), np.array([1, 2, 3]))

    # none branch
    assert image_backends._collate_batch_to_jax(None) is None

    # tuple/list recursion branch
    nested = [
        (np.ones((1, 1), dtype=np.float32), np.array([5], dtype=np.int32)),
        (np.zeros((1, 1), dtype=np.float32), np.array([7], dtype=np.int32)),
    ]
    out_nested = image_backends._collate_batch_to_jax(nested)
    assert isinstance(out_nested, list)
    np.testing.assert_array_equal(np.array(out_nested[0]), np.array([[1.0], [0.0]], dtype=np.float32))
    np.testing.assert_array_equal(np.array(out_nested[1]), np.array([5, 7], dtype=np.int32))


def test_collate_to_jax_single_numpy_batch_skips_concatenate(monkeypatch):
    arr = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    monkeypatch.setattr(
        image_backends.np,
        "concatenate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("single-item fast path should skip concatenate")
        ),
    )

    out = image_backends._collate_batch_to_jax([arr])
    np.testing.assert_array_equal(np.array(out), arr)


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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="relion5")
    subset_indices = np.array([2, 0, 2], dtype=np.int32)
    loader = ds.get_dataset_subset_generator(batch_size=8, subset_indices=subset_indices, mode="tilt_series")
    batches = list(loader)

    # Tilt-series iteration remains particle-at-a-time.
    assert len(batches) == 3
    out_particle_ids = [int(np.array(b[1]).reshape(-1)[0]) for b in batches]
    out_tilt_ids = [int(np.array(b[2]).reshape(-1)[0]) for b in batches]
    assert out_particle_ids == [2, 0, 2]
    # For this deterministic setup with num_tilts=1 and relion5 ordering.
    assert out_tilt_ids == [4, 0, 4]


def test_tiltseries_subset_generator_accepts_boolean_particle_mask(monkeypatch):
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="relion5")
    particle_mask = np.array([True, False, True], dtype=bool)
    loader = ds.get_dataset_subset_generator(batch_size=8, subset_indices=particle_mask, mode="tilt_series")
    batches = list(loader)
    out_particle_ids = [int(np.array(b[1]).reshape(-1)[0]) for b in batches]
    assert out_particle_ids == [0, 2]


def test_tiltseries_subset_generator_images_mode_accepts_boolean_particle_mask(monkeypatch):
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="relion5")
    particle_mask = np.array([True, False, True], dtype=bool)
    loader = ds.get_dataset_subset_generator(batch_size=4, subset_indices=particle_mask, mode="images")
    batches = list(loader)

    emitted_particles = np.concatenate([np.asarray(b[1]).reshape(-1) for b in batches], axis=0)
    # num_tilts=1 means one emitted image per selected particle.
    np.testing.assert_array_equal(np.sort(emitted_particles), np.array([0, 2], dtype=np.int32))


def test_tiltseries_subset_generator_rejects_bad_particle_masks(monkeypatch):
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=1, random_tilts=False, tilt_file_option="relion5")
    with pytest.raises(ValueError, match="boolean mask must be 1D"):
        list(
            ds.get_dataset_subset_generator(
                batch_size=8, subset_indices=np.array([[True, False, True]], dtype=bool), mode="tilt_series"
            )
        )
    with pytest.raises(ValueError, match="must match total size"):
        list(
            ds.get_dataset_subset_generator(
                batch_size=8, subset_indices=np.array([True, False], dtype=bool), mode="tilt_series"
            )
        )
    with pytest.raises(IndexError, match="negative"):
        list(
            ds.get_dataset_subset_generator(
                batch_size=8, subset_indices=np.array([-1], dtype=np.int32), mode="tilt_series"
            )
        )


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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=None, random_tilts=False, tilt_file_option="relion5")
    subset_indices = np.array([3, 1, 3], dtype=np.int32)
    loader = ds.get_image_subset_generator(batch_size=2, subset_indices=subset_indices)
    batches = list(loader)

    got = []
    for _imgs, _pidx, tidx in batches:
        got.extend(np.array(tidx).reshape(-1).tolist())
    assert got == [3, 1, 3]


def test_tiltseries_image_subset_generator_accepts_boolean_image_mask(monkeypatch):
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=None, random_tilts=False, tilt_file_option="relion5")
    mask = np.array([False, True, False, True, False], dtype=bool)
    loader = ds.get_image_subset_generator(batch_size=2, subset_indices=mask)
    batches = list(loader)
    got = []
    for _imgs, _pidx, tidx in batches:
        got.extend(np.array(tidx).reshape(-1).tolist())
    assert got == [1, 3]


def test_tiltseries_image_subset_generator_rejects_bad_image_masks(monkeypatch):
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
    monkeypatch.setattr(image_backends.ImageLoader, "from_file", lambda *args, **kwargs: _Source())
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    ds = image_backends.TiltSeriesDataset("dummy.star", num_tilts=None, random_tilts=False, tilt_file_option="relion5")
    with pytest.raises(ValueError, match="boolean mask must be 1D"):
        list(ds.get_image_subset_generator(batch_size=2, subset_indices=np.array([[True, False]], dtype=bool)))
    with pytest.raises(ValueError, match="must match total size"):
        list(ds.get_image_subset_generator(batch_size=2, subset_indices=np.array([True, False], dtype=bool)))
    with pytest.raises(IndexError, match="out-of-range"):
        list(ds.get_image_subset_generator(batch_size=2, subset_indices=np.array([9], dtype=np.int32)))


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
        image_backends.ImageLoader,
        "from_file",
        lambda *_args, **kwargs: _Source(indices=kwargs.get("indices")),
    )
    monkeypatch.setattr(image_backends.starfile.StarFile, "load", lambda _p: SimpleNamespace(df=df))

    # Reordered subset over original row indices.
    subset = np.array([5, 2, 4, 1], dtype=np.int32)
    ds = image_backends.TiltSeriesDataset(
        "dummy.star", ind=subset, num_tilts=1, random_tilts=False, tilt_file_option="relion5"
    )
    assert ds.dataset_tilt_indices == [0, 1, 2]

    # Direct image fetch should map to subset-local order: local 0 == original 5.
    img0, _pidx0, tidx0 = image_backends.ParticleImageDataset.__getitem__(ds, 0)
    assert int(tidx0) == 0
    np.testing.assert_array_equal(img0[0], ds.source._store[0])

    # For g3 in subset (locals 0 and 2), relion5 ordering picks lower dose first -> local 2.
    imgs, particle_idx, selected = ds[2]  # canonical order: g1, g2, g3
    assert int(particle_idx) == 2
    np.testing.assert_array_equal(selected, np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(imgs[0], ds.source._store[2])


def test_parse_particle_tilt_real_tiny_star_preserves_original_indices_with_subset(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    subset = np.array([5, 1, 4], dtype=np.int32)

    p2t, t2p = image_backends.TiltSeriesDataset.parse_particle_tilt(files["particles_star"], indices=subset)

    # Original STAR groups are cyclic: g1,g2,g3,g1,g2,g3.
    # Keeping rows [5,1,4] leaves groups g2:[1,4], g3:[5] in canonical order.
    np.testing.assert_array_equal(p2t[0], np.array([1, 4], dtype=np.int32))
    np.testing.assert_array_equal(p2t[1], np.array([5], dtype=np.int32))
    assert t2p[1] == 0 and t2p[4] == 0 and t2p[5] == 1


def test_parse_particle_tilt_real_tiny_star_subset_with_duplicate_tilt_indices(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    subset = np.array([5, 1, 5, 4], dtype=np.int32)

    p2t, t2p = image_backends.TiltSeriesDataset.parse_particle_tilt(files["particles_star"], indices=subset)

    # Groups after subset are g2:[1,4] and g3:[5,5], preserving duplicate tilt requests.
    np.testing.assert_array_equal(p2t[0], np.array([1, 4], dtype=np.int32))
    np.testing.assert_array_equal(p2t[1], np.array([5, 5], dtype=np.int32))
    assert t2p[1] == 0 and t2p[4] == 0 and t2p[5] == 1
