from types import SimpleNamespace

import numpy as np
import pytest
import pickle

pytest.importorskip("jax")

import recovar.dataset as dataset
from recovar import core, load_utils

pytestmark = pytest.mark.unit


def test_split_index_list_deterministic_and_disjoint():
    indices = np.arange(11)
    s1 = dataset.split_index_list(indices, split_random_seed=7)
    s2 = dataset.split_index_list(indices, split_random_seed=7)
    np.testing.assert_array_equal(s1[0], s2[0])
    np.testing.assert_array_equal(s1[1], s2[1])
    assert len(np.intersect1d(s1[0], s1[1])) == 0
    np.testing.assert_array_equal(np.sort(np.concatenate(s1)), indices)


def test_split_index_list_accepts_python_list():
    split = dataset.split_index_list([5, 2, 9, 1], split_random_seed=0)
    merged = np.sort(np.concatenate(split))
    np.testing.assert_array_equal(merged, np.array([1, 2, 5, 9]))


def test_split_index_list_rejects_empty():
    with pytest.raises(ValueError):
        dataset.split_index_list(np.array([], dtype=int))


def test_get_split_indices_from_ndarray_ind_file():
    ind = np.array([10, 30, 20, 40], dtype=int)
    split = dataset.get_split_indices("unused.star", ind_file=ind, validate_split=True)
    merged = np.sort(np.concatenate(split))
    np.testing.assert_array_equal(merged, np.sort(ind))


def test_get_split_indices_from_boolean_mask_ind_file(monkeypatch):
    monkeypatch.setattr(dataset, "get_num_images_in_dataset", lambda *args, **kwargs: 6)
    mask = np.array([True, False, True, False, False, True], dtype=bool)
    split = dataset.get_split_indices("unused.star", ind_file=mask, validate_split=True, split_random_seed=0)
    merged = np.sort(np.concatenate(split))
    np.testing.assert_array_equal(merged, np.array([0, 2, 5], dtype=np.int32))


def test_get_split_indices_rejects_non_1d_boolean_mask(monkeypatch):
    monkeypatch.setattr(dataset, "get_num_images_in_dataset", lambda *args, **kwargs: 4)
    with pytest.raises(ValueError, match="boolean mask must be 1D"):
        dataset.get_split_indices("unused.star", ind_file=np.array([[True, False], [False, True]], dtype=bool))


def test_get_split_indices_rejects_negative_indices():
    with pytest.raises(ValueError, match="must be non-negative"):
        dataset.get_split_indices("unused.star", ind_file=np.array([0, -1, 2], dtype=np.int32))


def test_get_split_indices_deduplicates_duplicate_indices():
    ind = np.array([5, 1, 5, 2], dtype=np.int32)
    split = dataset.get_split_indices("unused.star", ind_file=ind, validate_split=True, split_random_seed=0)
    merged = np.sort(np.concatenate(split))
    np.testing.assert_array_equal(merged, np.array([1, 2, 5], dtype=np.int32))


def test_get_split_indices_accepts_python_list_ind_file():
    split = dataset.get_split_indices("unused.star", ind_file=[8, 3, 1], validate_split=True, split_random_seed=0)
    merged = np.sort(np.concatenate(split))
    np.testing.assert_array_equal(merged, np.array([1, 3, 8], dtype=np.int32))


def test_reorder_to_original_indexing_from_halfsets():
    arr = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    halfsets = [np.array([2, 0]), np.array([4])]
    out = dataset.reorder_to_original_indexing_from_halfsets(arr, halfsets, num_images=5)
    assert out.shape == (5,)
    assert np.isnan(out[1]) and np.isnan(out[3])
    assert out[0] == pytest.approx(2.2)
    assert out[2] == pytest.approx(1.1)
    assert out[4] == pytest.approx(3.3)


def test_reorder_to_original_indexing_with_fake_cryos():
    arr = np.array([10, 20, 30])
    cryos = [
        SimpleNamespace(dataset_indices=np.array([2, 0]), image_stack=SimpleNamespace(dataset_tilt_indices=np.array([7, 5]))),
        SimpleNamespace(dataset_indices=np.array([4]), image_stack=SimpleNamespace(dataset_tilt_indices=np.array([9]))),
    ]
    out = dataset.reorder_to_original_indexing(arr, cryos, use_tilt_indices=False)
    assert out.shape == (5,)
    assert out[0] == 20 and out[2] == 10 and out[4] == 30

    out_tilt = dataset.reorder_to_original_indexing(arr, cryos, use_tilt_indices=True)
    assert out_tilt.shape == (10,)
    assert out_tilt[5] == 20 and out_tilt[7] == 10 and out_tilt[9] == 30


def test_reorder_to_original_indexing_from_halfsets_rejects_duplicate_indices():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    halfsets = [np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)]
    with pytest.raises(ValueError, match="contain duplicates"):
        dataset.reorder_to_original_indexing_from_halfsets(arr, halfsets)


def test_reorder_to_original_indexing_from_halfsets_rejects_length_mismatch():
    arr = np.array([1.0], dtype=np.float32)
    halfsets = [np.array([0, 1], dtype=np.int32), np.array([], dtype=np.int32)]
    with pytest.raises(ValueError, match="must match number of dataset indices"):
        dataset.reorder_to_original_indexing_from_halfsets(arr, halfsets)


def test_reorder_to_original_indexing_from_halfsets_rejects_negative_indices():
    arr = np.array([1.0], dtype=np.float32)
    halfsets = [np.array([-1], dtype=np.int32), np.array([], dtype=np.int32)]
    with pytest.raises(ValueError, match="must be non-negative"):
        dataset.reorder_to_original_indexing_from_halfsets(arr, halfsets)


def test_reorder_to_original_indexing_from_halfsets_rejects_too_small_num_images():
    arr = np.array([1.0], dtype=np.float32)
    halfsets = [np.array([2], dtype=np.int32), np.array([], dtype=np.int32)]
    with pytest.raises(ValueError, match="smaller than required size"):
        dataset.reorder_to_original_indexing_from_halfsets(arr, halfsets, num_images=2)


def test_reorder_to_original_indexing_from_halfsets_empty_inputs_return_empty():
    arr = np.array([], dtype=np.float32)
    halfsets = [np.array([], dtype=np.int32), np.array([], dtype=np.int32)]
    out = dataset.reorder_to_original_indexing_from_halfsets(arr, halfsets)
    assert out.shape == (0,)


def test_reorder_to_original_indexing_from_halfsets_empty_list_input_return_empty():
    arr = []
    halfsets = [np.array([], dtype=np.int32), np.array([], dtype=np.int32)]
    out = dataset.reorder_to_original_indexing_from_halfsets(arr, halfsets)
    assert out.shape == (0,)


def test_make_dataset_loader_dict_uninvert_parsing():
    args = SimpleNamespace(
        particles="p.star",
        ctf="c.pkl",
        poses="r.pkl",
        datadir=None,
        n_images=-1,
        ind=None,
        padding=0,
        uninvert_data="automatic",
        strip_prefix=None,
        tilt_series=False,
        tilt_series_ctf="cryoem",
        angle_per_tilt=3.0,
        dose_per_tilt=2.9,
        premultiplied_ctf=False,
    )
    out = dataset.make_dataset_loader_dict(args)
    assert out["uninvert_data"] is False

    args.uninvert_data = "true"
    out2 = dataset.make_dataset_loader_dict(args)
    assert out2["uninvert_data"] is True

    args.uninvert_data = "bad"
    with pytest.raises(ValueError):
        dataset.make_dataset_loader_dict(args)


def test_cryoemdataset_minimal_and_noise_access():
    ctf_params = np.zeros((3, 9), dtype=np.float32)
    rots = np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))
    trans = np.zeros((3, 2), dtype=np.float32)

    def ctf_fun(params, image_shape, voxel_size):
        return np.ones((params.shape[0], image_shape[0] * image_shape[1]), dtype=np.float64)

    ds = dataset.CryoEMDataset(
        image_stack=None,
        voxel_size=1.0,
        rotation_matrices=rots,
        translations=trans,
        CTF_params=ctf_params,
        CTF_fun=ctf_fun,
        grid_size=4,
    )
    assert ds.n_images == 3
    assert ds.image_shape == (4, 4)
    assert ds.get_noise_variance(np.array([0])) is None

    class _Noise:
        @staticmethod
        def get(indices):
            return np.asarray(indices) + 1

    ds.noise = _Noise()
    np.testing.assert_array_equal(ds.get_noise_variance(np.array([0, 2])), np.array([1, 3]))

    ctf = ds.CTF_fun(ds.CTF_params[:1], ds.image_shape, ds.voxel_size)
    assert ctf.dtype == ds.CTF_dtype


def test_cryoemdataset_casts_arrays_to_expected_dtypes():
    ctf_params = np.zeros((2, 9), dtype=np.float64)
    rots = np.tile(np.eye(3, dtype=np.float64), (2, 1, 1))
    trans = np.array([[1, 2], [3, 4]], dtype=np.float64)

    ds = dataset.CryoEMDataset(
        image_stack=None,
        voxel_size=1.0,
        rotation_matrices=rots,
        translations=trans,
        CTF_params=ctf_params,
        grid_size=4,
    )

    assert ds.rotation_matrices.dtype == np.float32
    assert ds.translations.dtype == np.float32
    assert ds.CTF_params.dtype == np.float32


def test_get_default_dataset_option_has_expected_keys():
    d = dataset.get_default_dataset_option()
    for key in [
        "particles_file",
        "ctf_file",
        "poses_file",
        "tilt_series",
        "tilt_series_ctf",
        "uninvert_data",
        "premultiplied_ctf",
    ]:
        assert key in d


def test_figure_out_halfsets_random_path(monkeypatch):
    args = SimpleNamespace(
        halfsets=None,
        tilt_series=False,
        tilt_series_ctf="cryoem",
        particles="particles.star",
        ind=None,
        tilt_ind=None,
        ntilts=None,
        datadir=None,
        strip_prefix=None,
        n_images=-1,
    )
    expected = [np.array([0, 2]), np.array([1, 3])]
    monkeypatch.setattr(dataset, "get_split_indices", lambda *a, **k: expected)
    out = dataset.figure_out_halfsets(args)
    np.testing.assert_array_equal(out[0], expected[0])
    np.testing.assert_array_equal(out[1], expected[1])


def test_figure_out_halfsets_n_images_limit(monkeypatch):
    args = SimpleNamespace(
        halfsets=None,
        tilt_series=False,
        tilt_series_ctf="cryoem",
        particles="particles.star",
        ind=None,
        tilt_ind=None,
        ntilts=None,
        datadir=None,
        strip_prefix=None,
        n_images=2,
    )
    monkeypatch.setattr(dataset, "get_split_indices", lambda *a, **k: [np.array([0, 2]), np.array([1, 3])])
    out = dataset.figure_out_halfsets(args)
    np.testing.assert_array_equal(out[0], np.array([0]))
    np.testing.assert_array_equal(out[1], np.array([1]))


def test_figure_out_halfsets_from_file_intersects_with_ind(tmp_path):
    halfsets_file = tmp_path / "halfsets.pkl"
    ind_file = tmp_path / "ind.pkl"
    with open(halfsets_file, "wb") as f:
        pickle.dump([np.array([0, 1, 2]), np.array([3, 4, 5])], f)
    with open(ind_file, "wb") as f:
        pickle.dump(np.array([1, 2, 4], dtype=int), f)

    args = SimpleNamespace(
        halfsets=str(halfsets_file),
        tilt_series=False,
        tilt_series_ctf="cryoem",
        particles="particles.star",
        ind=str(ind_file),
        tilt_ind=None,
        ntilts=None,
        datadir=None,
        strip_prefix=None,
        n_images=-1,
    )
    out = dataset.figure_out_halfsets(args)
    np.testing.assert_array_equal(out[0], np.array([1, 2]))
    np.testing.assert_array_equal(out[1], np.array([4]))


def test_figure_out_halfsets_from_file_intersects_with_ind_ndarray(tmp_path):
    halfsets_file = tmp_path / "halfsets.pkl"
    with open(halfsets_file, "wb") as f:
        pickle.dump([np.array([0, 1, 2]), np.array([3, 4, 5])], f)

    args = SimpleNamespace(
        halfsets=str(halfsets_file),
        tilt_series=False,
        tilt_series_ctf="cryoem",
        particles="particles.star",
        ind=np.array([2, 3, 5], dtype=np.int32),
        tilt_ind=None,
        ntilts=None,
        datadir=None,
        strip_prefix=None,
        n_images=-1,
    )
    out = dataset.figure_out_halfsets(args)
    np.testing.assert_array_equal(out[0], np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([3, 5], dtype=np.int32))


def test_figure_out_halfsets_from_file_intersects_with_ind_preserves_halfset_order(tmp_path):
    halfsets_file = tmp_path / "halfsets.pkl"
    with open(halfsets_file, "wb") as f:
        pickle.dump([np.array([5, 2, 4]), np.array([3, 1, 0])], f)

    args = SimpleNamespace(
        halfsets=str(halfsets_file),
        tilt_series=False,
        tilt_series_ctf="cryoem",
        particles="particles.star",
        ind=np.array([4, 5, 0, 1], dtype=np.int32),
        tilt_ind=None,
        ntilts=None,
        datadir=None,
        strip_prefix=None,
        n_images=-1,
    )
    out = dataset.figure_out_halfsets(args)
    np.testing.assert_array_equal(out[0], np.array([5, 4], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([1, 0], dtype=np.int32))


def test_figure_out_halfsets_from_file_intersects_with_boolean_mask_ind(tmp_path, monkeypatch):
    halfsets_file = tmp_path / "halfsets.pkl"
    with open(halfsets_file, "wb") as f:
        pickle.dump([np.array([5, 2, 4]), np.array([3, 1, 0])], f)

    monkeypatch.setattr(dataset, "get_num_images_in_dataset", lambda *args, **kwargs: 6)

    args = SimpleNamespace(
        halfsets=str(halfsets_file),
        tilt_series=False,
        tilt_series_ctf="cryoem",
        particles="particles.star",
        ind=np.array([True, True, False, False, True, True], dtype=bool),
        tilt_ind=None,
        ntilts=None,
        datadir=None,
        strip_prefix=None,
        n_images=-1,
    )
    out = dataset.figure_out_halfsets(args)
    np.testing.assert_array_equal(out[0], np.array([5, 4], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([1, 0], dtype=np.int32))


def test_figure_out_halfsets_from_file_accepts_boolean_halfset_masks(tmp_path, monkeypatch):
    halfsets_file = tmp_path / "halfsets.pkl"
    with open(halfsets_file, "wb") as f:
        pickle.dump(
            [
                np.array([True, False, False, True, False, False], dtype=bool),
                np.array([False, True, True, False, True, True], dtype=bool),
            ],
            f,
        )

    monkeypatch.setattr(dataset, "get_num_images_in_dataset", lambda *args, **kwargs: 6)

    args = SimpleNamespace(
        halfsets=str(halfsets_file),
        tilt_series=False,
        tilt_series_ctf="cryoem",
        particles="particles.star",
        ind=None,
        tilt_ind=None,
        ntilts=None,
        datadir=None,
        strip_prefix=None,
        n_images=-1,
    )
    out = dataset.figure_out_halfsets(args)
    np.testing.assert_array_equal(out[0], np.array([0, 3], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([1, 2, 4, 5], dtype=np.int32))


def test_figure_out_halfsets_from_file_rejects_non_two_halfsets(tmp_path):
    halfsets_file = tmp_path / "halfsets.pkl"
    with open(halfsets_file, "wb") as f:
        pickle.dump([np.array([0, 1], dtype=np.int32)], f)

    args = SimpleNamespace(
        halfsets=str(halfsets_file),
        tilt_series=False,
        tilt_series_ctf="cryoem",
        particles="particles.star",
        ind=None,
        tilt_ind=None,
        ntilts=None,
        datadir=None,
        strip_prefix=None,
        n_images=-1,
    )
    with pytest.raises(ValueError, match="exactly two halfsets"):
        dataset.figure_out_halfsets(args)


class _FakeImageStack:
    def __init__(self, n_images=4, D=4, padding=0, Np=2):
        self.n_images = n_images
        self.D = D
        self.unpadded_D = D
        self.padding = padding
        self.image_shape = (D, D)
        self.Np = Np
        self.mask = np.ones((D, D), dtype=np.float32)

    def get_dataset_generator(self, batch_size, num_workers=0, **kwargs):
        return ("dataset", batch_size, num_workers, kwargs)

    def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
        return ("subset", batch_size, tuple(subset_indices), num_workers, kwargs)

    def get_image_subset_generator(self, batch_size, subset_indices, num_workers=0):
        return ("image_subset", batch_size, tuple(subset_indices), num_workers)

    def get_image_generator(self, batch_size, num_workers=0):
        return ("image", batch_size, num_workers)

    def __getitem__(self, i):
        img = np.ones((self.D * self.D,), dtype=np.complex64) * (i + 1)
        return img[None], None, np.array([i], dtype=int)

    def process_images(self, image, apply_image_mask=True):
        return np.asarray(image)


def _fake_load_ctf_params(D, ctf_file):
    n = 4
    # [voxel, dfu, dfv, dfang, volt, cs, w, phase]
    ctf = np.zeros((n, 8), dtype=np.float32)
    ctf[:, 0] = 1.5
    ctf[:, 4] = 300.0
    ctf[:, 5] = 2.7
    ctf[:, 6] = 0.1
    return ctf


def _fake_load_poses(poses_file, n_images, D, ind=None):
    rots = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
    trans = np.zeros((n_images, 2), dtype=np.float32)
    return rots, trans, None


def test_load_cryodrgn_dataset_cryoem_branch(monkeypatch):
    fake_stack = _FakeImageStack(n_images=4, D=8, padding=2)
    monkeypatch.setattr(dataset, "LazyMRCDataMod", lambda *a, **k: fake_stack)
    monkeypatch.setattr(load_utils, "load_ctf_params", _fake_load_ctf_params)
    monkeypatch.setattr(load_utils, "load_poses", _fake_load_poses)

    out = dataset.load_cryodrgn_dataset(
        particles_file="p.mrcs",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
        lazy=True,
        ind=np.array([0, 2], dtype=int),
        tilt_series=False,
        tilt_series_ctf="cryoem",
    )
    assert out.CTF_fun_inp is core.evaluate_ctf_wrapper
    assert out.n_images == 4
    assert out.CTF_params.shape[1] == 9
    np.testing.assert_array_equal(out.dataset_indices, np.array([0, 2], dtype=int))


def test_load_cryodrgn_dataset_rejects_ctf_count_mismatch_when_no_subset(monkeypatch):
    fake_stack = _FakeImageStack(n_images=4, D=8, padding=0)
    monkeypatch.setattr(dataset, "LazyMRCDataMod", lambda *a, **k: fake_stack)
    monkeypatch.setattr(load_utils, "load_poses", _fake_load_poses)

    def _short_ctf(D, ctf_file):
        _ = (D, ctf_file)
        ctf = np.zeros((3, 8), dtype=np.float32)
        ctf[:, 0] = 1.5
        ctf[:, 4] = 300.0
        ctf[:, 5] = 2.7
        ctf[:, 6] = 0.1
        return ctf

    monkeypatch.setattr(load_utils, "load_ctf_params", _short_ctf)

    with pytest.raises(ValueError, match="CTF parameter count"):
        dataset.load_cryodrgn_dataset(
            particles_file="p.mrcs",
            poses_file="poses.pkl",
            ctf_file="ctf.pkl",
            lazy=True,
            tilt_series=False,
            tilt_series_ctf="cryoem",
        )


def test_load_cryodrgn_dataset_propagates_premultiplied_ctf_flag(monkeypatch):
    fake_stack = _FakeImageStack(n_images=4, D=8, padding=0)
    monkeypatch.setattr(dataset, "LazyMRCDataMod", lambda *a, **k: fake_stack)
    monkeypatch.setattr(load_utils, "load_ctf_params", _fake_load_ctf_params)
    monkeypatch.setattr(load_utils, "load_poses", _fake_load_poses)

    out = dataset.load_cryodrgn_dataset(
        particles_file="p.mrcs",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="cryoem",
        premultiplied_ctf=True,
    )
    assert out.premultiplied_ctf is True


def test_load_cryodrgn_dataset_rotation_only_pose_defaults_zero_translations(monkeypatch):
    fake_stack = _FakeImageStack(n_images=4, D=8, padding=0)
    monkeypatch.setattr(dataset, "LazyMRCDataMod", lambda *a, **k: fake_stack)
    monkeypatch.setattr(load_utils, "load_ctf_params", _fake_load_ctf_params)

    def _poses_no_trans(_poses_file, n_images, _D, ind=None):
        _ = ind
        rots = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        return rots, None, None

    monkeypatch.setattr(load_utils, "load_poses", _poses_no_trans)

    out = dataset.load_cryodrgn_dataset(
        particles_file="p.mrcs",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="cryoem",
    )
    np.testing.assert_array_equal(out.translations, np.zeros((4, 2), dtype=np.float32))


def test_load_cryodrgn_dataset_rejects_translation_shape_mismatch(monkeypatch):
    fake_stack = _FakeImageStack(n_images=4, D=8, padding=0)
    monkeypatch.setattr(dataset, "LazyMRCDataMod", lambda *a, **k: fake_stack)
    monkeypatch.setattr(load_utils, "load_ctf_params", _fake_load_ctf_params)

    def _poses_bad_trans(_poses_file, n_images, _D, ind=None):
        _ = ind
        rots = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        trans = np.zeros((n_images, 3), dtype=np.float32)
        return rots, trans, None

    monkeypatch.setattr(load_utils, "load_poses", _poses_bad_trans)

    with pytest.raises(ValueError, match="Translation array must have shape"):
        dataset.load_cryodrgn_dataset(
            particles_file="p.mrcs",
            poses_file="poses.pkl",
            ctf_file="ctf.pkl",
            lazy=True,
            tilt_series=False,
            tilt_series_ctf="cryoem",
        )


def test_load_cryodrgn_dataset_from_dose_branch(monkeypatch):
    fake_stack = _FakeImageStack(n_images=4, D=8, padding=0)
    monkeypatch.setattr(dataset, "LazyMRCDataMod", lambda *a, **k: fake_stack)
    monkeypatch.setattr(load_utils, "load_ctf_params", _fake_load_ctf_params)
    monkeypatch.setattr(load_utils, "load_poses", _fake_load_poses)

    class _FakeTiltSeriesData:
        def __init__(self, *args, **kwargs):
            self.ctfscalefactor = np.array([1.0, 0.9, 1.1, 1.0], dtype=np.float32)
            self.ctfBfactor = np.array([-2.0, -4.0, -6.0, -8.0], dtype=np.float32)
            self.tilt_numbers = np.array([0, 1, 2, 3], dtype=np.int32)

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    sentinel_ctf_fun = object()
    monkeypatch.setattr(core, "get_cryo_ET_CTF_fun", lambda dose_per_tilt, angle_per_tilt: sentinel_ctf_fun)

    out = dataset.load_cryodrgn_dataset(
        particles_file="p.mrcs",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="from_dose",
        dose_per_tilt=2.9,
        angle_per_tilt=3.0,
    )
    assert out.CTF_fun_inp is sentinel_ctf_fun
    # from_dose appends tilt-number channel after baseline CTF fields.
    assert out.CTF_params.shape[1] == 10
    np.testing.assert_array_equal(out.CTF_params[:, -1], np.array([0, 1, 2, 3], dtype=np.float32))


def test_load_cryodrgn_dataset_tilt_series_relion5_branch(monkeypatch):
    class _FakeTiltSeriesData(_FakeImageStack):
        def __init__(self, *args, **kwargs):
            super().__init__(n_images=4, D=8, padding=0, Np=2)
            self.ctfscalefactor = np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float32)
            self.dose = np.array([0.0, 1.5, 3.0, 4.5], dtype=np.float32)
            self.tilt_numbers = np.array([0, 1, 2, 3], dtype=np.float32)

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(load_utils, "load_ctf_params", _fake_load_ctf_params)
    monkeypatch.setattr(load_utils, "load_poses", _fake_load_poses)

    out = dataset.load_cryodrgn_dataset(
        particles_file="p.star",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )
    assert out.tilt_series_flag is True
    assert out.n_units == 2  # Np from fake tilt dataset
    assert out.CTF_fun_inp is core.evaluate_ctf_wrapper_tilt_series_v2
    # Baseline 8 + bfactor + contrast + (dose,angle) then drop voxel_size => 11.
    assert out.CTF_params.shape[1] == 11
    np.testing.assert_array_equal(
        out.CTF_params[:, core.CTFParamIndex.CONTRAST],
        np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float32),
    )


def test_load_cryodrgn_dataset_tilt_series_warp_alias_maps_to_v2_scale(monkeypatch):
    called = {}

    class _FakeTiltSeriesData(_FakeImageStack):
        def __init__(self, *args, **kwargs):
            called["tilt_file_option"] = kwargs.get("tilt_file_option")
            super().__init__(n_images=4, D=8, padding=0, Np=2)
            self.ctfscalefactor = np.array([0.95, 1.05, 0.9, 1.1], dtype=np.float32)
            # WARP convention used by loader: dose ~ -B/4.
            self.ctfBfactor = np.array([-4.0, -8.0, -12.0, -16.0], dtype=np.float32)
            self.tilt_numbers = np.array([0, 1, 2, 3], dtype=np.float32)

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(load_utils, "load_ctf_params", _fake_load_ctf_params)
    monkeypatch.setattr(load_utils, "load_poses", _fake_load_poses)

    out = dataset.load_cryodrgn_dataset(
        particles_file="p.star",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="warp",  # alias branch
        angle_per_tilt=3.0,
    )
    assert called["tilt_file_option"] == "warp"
    assert out.tilt_series_flag is True
    assert out.CTF_fun_inp is core.evaluate_ctf_wrapper_tilt_series_v2
    # Baseline 8 + bfactor + contrast + (dose,angle) then drop voxel_size => 11.
    assert out.CTF_params.shape[1] == 11
    np.testing.assert_array_equal(
        out.CTF_params[:, core.CTFParamIndex.CONTRAST],
        np.array([0.95, 1.05, 0.9, 1.1], dtype=np.float32),
    )


def test_load_cryodrgn_dataset_defaults_tilt_series_ctf_by_mode(monkeypatch):
    calls = {"tilt_init": 0}

    class _FakeTiltSeriesData(_FakeImageStack):
        def __init__(self, *args, **kwargs):
            calls["tilt_init"] += 1
            calls["tilt_file_option"] = kwargs.get("tilt_file_option")
            super().__init__(n_images=4, D=8, padding=0, Np=2)
            self.ctfscalefactor = np.ones(4, dtype=np.float32)
            self.ctfBfactor = -4 * np.ones(4, dtype=np.float32)
            self.tilt_numbers = np.arange(4, dtype=np.float32)
            self.dose = np.arange(4, dtype=np.float32)

    monkeypatch.setattr(dataset, "LazyMRCDataMod", lambda *a, **k: _FakeImageStack(n_images=4, D=8, padding=0))
    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(load_utils, "load_ctf_params", _fake_load_ctf_params)
    monkeypatch.setattr(load_utils, "load_poses", _fake_load_poses)

    # tilt_series=False with tilt_series_ctf=None should default to cryoem and not instantiate TiltSeriesData.
    out_spa = dataset.load_cryodrgn_dataset(
        particles_file="p.mrcs",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
        lazy=True,
        tilt_series=False,
        tilt_series_ctf=None,
    )
    assert out_spa.CTF_fun_inp is core.evaluate_ctf_wrapper
    assert calls["tilt_init"] == 0

    # tilt_series=True with tilt_series_ctf=None should default to relion5.
    out_tilt = dataset.load_cryodrgn_dataset(
        particles_file="p.star",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
        lazy=True,
        tilt_series=True,
        tilt_series_ctf=None,
    )
    assert out_tilt.CTF_fun_inp is core.evaluate_ctf_wrapper_tilt_series_v2
    assert calls["tilt_init"] == 1
    assert calls["tilt_file_option"] == "relion5"


def test_load_cryodrgn_dataset_from_star_branch_sets_contrast_and_bfactor(monkeypatch):
    captured = {}

    class _FakeTiltSeriesData:
        def __init__(self, *args, **kwargs):
            captured["sort_with_Bfac"] = kwargs.get("sort_with_Bfac")
            self.ctfscalefactor = np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float32)
            # STAR convention: negative B-factors.
            self.ctfBfactor = np.array([-5.0, -10.0, -15.0, -20.0], dtype=np.float32)
            self.tilt_numbers = np.array([0, 1, 2, 3], dtype=np.float32)

    monkeypatch.setattr(dataset, "LazyMRCDataMod", lambda *a, **k: _FakeImageStack(n_images=4, D=8, padding=0))
    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(load_utils, "load_ctf_params", _fake_load_ctf_params)
    monkeypatch.setattr(load_utils, "load_poses", _fake_load_poses)

    out = dataset.load_cryodrgn_dataset(
        particles_file="p.mrcs",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="from_star",
        sort_with_Bfac=True,
    )
    assert captured["sort_with_Bfac"] is True
    assert out.CTF_fun_inp is core.evaluate_ctf_wrapper
    assert out.CTF_params.shape[1] == 9
    np.testing.assert_array_equal(
        out.CTF_params[:, core.CTFParamIndex.CONTRAST],
        np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float32),
    )
    # Loader flips sign to positive in-memory.
    np.testing.assert_array_equal(
        out.CTF_params[:, core.CTFParamIndex.BFACTOR],
        np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float32),
    )


def test_load_cryodrgn_dataset_v2_scale_from_star_uses_star_scaling_and_zero_angles(monkeypatch):
    class _FakeTiltSeriesData:
        def __init__(self, *args, **kwargs):
            self.ctfscalefactor = np.array([0.7, 0.8, 0.9, 1.0], dtype=np.float32)
            self.ctfBfactor = np.array([-4.0, -8.0, -12.0, -16.0], dtype=np.float32)
            self.tilt_numbers = np.array([0, 1, 2, 3], dtype=np.float32)

    monkeypatch.setattr(dataset, "LazyMRCDataMod", lambda *a, **k: _FakeImageStack(n_images=4, D=8, padding=0))
    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(load_utils, "load_ctf_params", _fake_load_ctf_params)
    monkeypatch.setattr(load_utils, "load_poses", _fake_load_poses)

    out = dataset.load_cryodrgn_dataset(
        particles_file="p.mrcs",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="v2_scale_from_star",
        dose_per_tilt=None,
        angle_per_tilt=3.0,
    )
    assert out.CTF_fun_inp is core.evaluate_ctf_wrapper_tilt_series_v2
    # Baseline 8 + bfactor + contrast + (dose,angle) then drop voxel_size => 11.
    assert out.CTF_params.shape[1] == 11
    np.testing.assert_array_equal(
        out.CTF_params[:, core.CTFParamIndex.CONTRAST],
        np.array([0.7, 0.8, 0.9, 1.0], dtype=np.float32),
    )
    # scale_from_star forces angle_per_tilt=0 => angle channel should be all zeros.
    np.testing.assert_array_equal(out.CTF_params[:, -1], np.zeros(4, dtype=np.float32))
    # dose_per_tilt=None branch uses dose from star: -Bfactor/4.
    np.testing.assert_array_equal(
        out.CTF_params[:, core.CTFParamIndex.DOSE],
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )


def test_load_cryodrgn_dataset_warp_alias_for_non_tilt_series_maps_to_v2_scale(monkeypatch):
    class _FakeTiltSeriesData:
        def __init__(self, *args, **kwargs):
            self.ctfscalefactor = np.array([1.0, 1.1, 0.9, 1.2], dtype=np.float32)
            self.ctfBfactor = np.array([-4.0, -8.0, -12.0, -16.0], dtype=np.float32)
            self.tilt_numbers = np.array([0, 1, 2, 3], dtype=np.float32)

    monkeypatch.setattr(dataset, "LazyMRCDataMod", lambda *a, **k: _FakeImageStack(n_images=4, D=8, padding=0))
    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(load_utils, "load_ctf_params", _fake_load_ctf_params)
    monkeypatch.setattr(load_utils, "load_poses", _fake_load_poses)

    out = dataset.load_cryodrgn_dataset(
        particles_file="p.mrcs",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
        lazy=True,
        tilt_series=False,
        tilt_series_ctf="warp",  # alias -> v2_scale_from_star
    )
    assert out.CTF_fun_inp is core.evaluate_ctf_wrapper_tilt_series_v2
    assert out.CTF_params.shape[1] == 11
    np.testing.assert_array_equal(
        out.CTF_params[:, core.CTFParamIndex.CONTRAST],
        np.array([1.0, 1.1, 0.9, 1.2], dtype=np.float32),
    )
    np.testing.assert_array_equal(out.CTF_params[:, -1], np.zeros(4, dtype=np.float32))


def test_cryoemdataset_predicted_image_and_generators(monkeypatch):
    stack = _FakeImageStack(n_images=3, D=4, padding=0, Np=2)
    ctf_params = np.zeros((3, 9), dtype=np.float32)
    rots = np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))
    trans = np.zeros((3, 2), dtype=np.float32)
    ds = dataset.CryoEMDataset(
        image_stack=stack,
        voxel_size=1.0,
        rotation_matrices=rots,
        translations=trans,
        CTF_params=ctf_params,
        tilt_series_flag=True,
    )
    assert ds.n_units == stack.Np
    assert ds.get_dataset_generator(2)[0] == "dataset"
    assert ds.get_dataset_subset_generator(2, np.array([0, 2]))[0] == "subset"
    assert ds.get_image_generator(2)[0] == "image"
    assert ds.get_image_subset_generator(2, np.array([1]))[0] == "image_subset"

    called = {}
    import recovar.core_forward as core_forward_mod

    def _fake_fwd(config, volume, ctf, rots, skip_ctf=False):
        called["disc_type"] = config.disc_type
        called["skip_ctf"] = skip_ctf
        return np.ones((len(ctf), config.image_shape[0] * config.image_shape[1]), dtype=np.complex64) * (2 + 3j)

    monkeypatch.setattr(core_forward_mod, "forward_model", _fake_fwd)
    monkeypatch.setattr(dataset.fourier_transform_utils, "get_idft2", lambda x: x)
    pred = ds.get_predicted_image(np.array([0, 2]), volume=np.zeros(ds.volume_size), skip_ctf=True, spatial=True)
    assert pred.shape == (2, 4, 4)
    assert called["disc_type"] == "linear_interp"
    assert called["skip_ctf"] is True
    np.testing.assert_array_equal(pred, np.full((2, 4, 4), 2.0, dtype=np.float32))


def test_get_split_tilt_indices_with_filters(tmp_path, monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
            tilts_to_particles = [0, 0, 1, 1, 2, 2]
            return particles_to_tilts, tilts_to_particles

        def __init__(self, particles_file, datadir=None):
            self.tilt_numbers = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3]), np.array([4, 5])][i] for i in particle_ind]
        ),
    )
    monkeypatch.setattr(dataset, "split_index_list", lambda valid_particles: [np.array([0]), np.array([2])])

    ind_file = tmp_path / "ind.pkl"
    tilt_ind_file = tmp_path / "tilt_ind.pkl"
    with open(ind_file, "wb") as f:
        pickle.dump(np.array([0, 1, 4, 5], dtype=int), f)
    with open(tilt_ind_file, "wb") as f:
        pickle.dump(np.array([0, 2], dtype=int), f)

    out = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        ind_file=str(ind_file),
        tilt_ind_file=str(tilt_ind_file),
        ntilts=1,
        datadir=None,
    )
    # ntilts=1 keeps only tilt number 0 from each selected particle.
    np.testing.assert_array_equal(out[0], np.array([0]))
    np.testing.assert_array_equal(out[1], np.array([4]))


def test_get_split_tilt_indices_with_precomputed_halfsets_handles_empty_half(tmp_path, monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
            tilts_to_particles = [0, 0, 1, 1, 2, 2]
            return particles_to_tilts, tilts_to_particles

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3]), np.array([4, 5])][i] for i in particle_ind]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    # Keep only particle 2 through tilt_ind_file, but precomputed halfsets request [0] and [1].
    # Both halves become empty after intersection with valid particles; should not crash.
    tilt_ind_file = tmp_path / "tilt_ind.pkl"
    halfsets_file = tmp_path / "halfsets.pkl"
    with open(tilt_ind_file, "wb") as f:
        pickle.dump(np.array([2], dtype=np.int32), f)
    with open(halfsets_file, "wb") as f:
        pickle.dump([np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)], f)

    out = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        tilt_ind_file=str(tilt_ind_file),
        particle_halfset_indices_file=str(halfsets_file),
        datadir=None,
    )
    np.testing.assert_array_equal(out[0], np.array([], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([], dtype=np.int32))


def test_get_split_tilt_indices_no_allowed_images_returns_empty_halfsets(tmp_path, monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [np.array([0, 1]), np.array([2, 3])]
            tilts_to_particles = [0, 0, 1, 1]
            return particles_to_tilts, tilts_to_particles

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3])][i] for i in particle_ind]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    # ind_file excludes every image.
    ind_file = tmp_path / "ind_none.pkl"
    with open(ind_file, "wb") as f:
        pickle.dump(np.array([], dtype=np.int32), f)

    out = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        ind_file=str(ind_file),
        datadir=None,
    )
    assert len(out) == 2
    np.testing.assert_array_equal(out[0], np.array([], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([], dtype=np.int32))


def test_get_split_indices_from_pickle_file_path(tmp_path, monkeypatch):
    ind_file = tmp_path / "ind.pkl"
    with open(ind_file, "wb") as f:
        pickle.dump(np.array([7, 1, 5, 3], dtype=int), f)

    monkeypatch.setattr(dataset, "split_index_list", lambda idx, split_random_seed=0: [np.sort(idx[:2]), np.sort(idx[2:])])
    out = dataset.get_split_indices("unused.star", ind_file=str(ind_file), validate_split=True)
    np.testing.assert_array_equal(out[0], np.array([1, 7]))
    np.testing.assert_array_equal(out[1], np.array([3, 5]))


def test_get_split_indices_raises_on_overlapping_split(monkeypatch):
    monkeypatch.setattr(dataset, "get_num_images_in_dataset", lambda *args, **kwargs: 4)
    monkeypatch.setattr(dataset, "split_index_list", lambda *_args, **_kwargs: [np.array([0, 1]), np.array([1, 2])])
    with pytest.raises(ValueError, match="overlapping indices"):
        dataset.get_split_indices("particles.star", validate_split=True)


def test_load_dataset_from_dict_delegates(monkeypatch):
    captured = {}

    def _fake_load_cryodrgn_dataset(**kwargs):
        captured["kwargs"] = kwargs
        return "cryo"

    monkeypatch.setattr(dataset, "load_cryodrgn_dataset", _fake_load_cryodrgn_dataset)
    out = dataset.load_dataset_from_dict({"particles_file": "p", "poses_file": "r", "ctf_file": "c"}, lazy=False)
    assert out == "cryo"
    assert captured["kwargs"]["lazy"] is False
    assert captured["kwargs"]["particles_file"] == "p"


def test_load_dataset_from_args_uses_given_split(monkeypatch):
    args = SimpleNamespace(
        particles="p.star",
        ctf="c.pkl",
        poses="r.pkl",
        datadir=None,
        n_images=-1,
        ind=None,
        padding=0,
        uninvert_data="false",
        strip_prefix=None,
        tilt_series=False,
        tilt_series_ctf="cryoem",
        angle_per_tilt=3.0,
        dose_per_tilt=2.9,
        premultiplied_ctf=False,
    )
    given_split = [np.array([0, 2]), np.array([1, 3])]
    captured = {}

    monkeypatch.setattr(dataset, "make_dataset_loader_dict", lambda _a: {"particles_file": "p.star"})
    monkeypatch.setattr(
        dataset,
        "get_split_datasets_from_dict",
        lambda loader_dict, ind_split, lazy=False: captured.setdefault("call", (loader_dict, ind_split, lazy)) or ["a", "b"],
    )

    out = dataset.load_dataset_from_args(args, lazy=True, ind_split=given_split)
    assert out == captured["call"]
    assert captured["call"][0] == {"particles_file": "p.star"}
    assert captured["call"][1] is given_split
    assert captured["call"][2] is True


def test_load_dataset_from_args_computes_split_when_missing(monkeypatch):
    args = SimpleNamespace(
        particles="p.star",
        ctf="c.pkl",
        poses="r.pkl",
        datadir=None,
        n_images=-1,
        ind=None,
        padding=0,
        uninvert_data="false",
        strip_prefix=None,
        tilt_series=False,
        tilt_series_ctf="cryoem",
        angle_per_tilt=3.0,
        dose_per_tilt=2.9,
        premultiplied_ctf=False,
    )
    computed = [np.array([0, 2]), np.array([1, 3])]
    monkeypatch.setattr(dataset, "figure_out_halfsets", lambda _a: computed)
    monkeypatch.setattr(dataset, "make_dataset_loader_dict", lambda _a: {"particles_file": "p.star"})
    called = {}

    def _fake_get_split(loader_dict, ind_split, lazy=False):
        called["v"] = (loader_dict, ind_split, lazy)
        return ["ok"]

    monkeypatch.setattr(dataset, "get_split_datasets_from_dict", _fake_get_split)
    out = dataset.load_dataset_from_args(args, lazy=False, ind_split=None)
    assert out == ["ok"]
    assert called["v"][1] is computed
    assert called["v"][2] is False


def test_make_dataset_loader_dict_without_strip_prefix_attr():
    args = SimpleNamespace(
        particles="p.star",
        ctf="c.pkl",
        poses="r.pkl",
        datadir=None,
        n_images=-1,
        ind=None,
        padding=0,
        uninvert_data="false",
        tilt_series=False,
        tilt_series_ctf="cryoem",
        angle_per_tilt=3.0,
        dose_per_tilt=2.9,
        premultiplied_ctf=False,
    )
    out = dataset.make_dataset_loader_dict(args)
    assert out["strip_prefix"] is None


def test_get_split_datasets_calls_loader_once_per_halfset(monkeypatch):
    calls = []

    def _fake_load(*args, **kwargs):
        calls.append(kwargs)
        return kwargs["ind"]

    monkeypatch.setattr(dataset, "load_dataset", _fake_load)
    ind_split = [np.array([2, 0], dtype=np.int32), np.array([5], dtype=np.int32)]
    out = dataset.get_split_datasets(
        particles_file="particles.mrcs",
        poses_file="poses.pkl",
        ctf_file="ctf.pkl",
        datadir="/tmp/data",
        ind_split=ind_split,
        lazy=True,
        tilt_series=True,
        tilt_series_ctf="relion5",
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(out[0], ind_split[0])
    np.testing.assert_array_equal(out[1], ind_split[1])
    assert calls[0]["lazy"] is True
    assert calls[0]["tilt_series"] is True
    assert calls[0]["tilt_series_ctf"] == "relion5"


def test_subsample_cryoem_dataset_reindexes_and_slices_metadata():
    class _GenImageStack:
        def __init__(self, n=5, D=4):
            self.n_images = n
            self.Np = n
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            imgs = np.zeros((subset_indices.size, self.D * self.D), dtype=np.complex64)
            yield imgs, subset_indices, subset_indices

    n = 5
    stack = _GenImageStack(n=n, D=4)
    ctf_params = np.zeros((n, 9), dtype=np.float32)
    rots = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    trans = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    cryo = dataset.CryoEMDataset(
        image_stack=stack,
        voxel_size=1.5,
        rotation_matrices=rots,
        translations=trans,
        CTF_params=ctf_params,
    )

    sub = dataset.subsample_cryoem_dataset(cryo, np.array([True, False, True, False, True]))
    np.testing.assert_array_equal(sub.dataset_indices, np.array([0, 2, 4], dtype=np.int32))
    assert sub.n_images == 3
    np.testing.assert_array_equal(sub.translations, trans[[0, 2, 4]])

    batch = next(sub.get_dataset_generator(batch_size=2))
    _, particle_idx, image_idx = batch
    # Reindexed to contiguous local ids.
    np.testing.assert_array_equal(particle_idx, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(image_idx, np.array([0, 1], dtype=np.int32))


def test_subsample_cryoem_dataset_preserves_premultiplied_ctf_flag():
    n = 4

    class _GenImageStack:
        def __init__(self, n=4, D=4):
            self.n_images = n
            self.Np = n
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            imgs = np.zeros((subset_indices.size, self.D * self.D), dtype=np.complex64)
            yield imgs, subset_indices, subset_indices

    stack = _GenImageStack(n=n, D=4)
    ctf_params = np.zeros((n, 9), dtype=np.float32)
    rots = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    trans = np.zeros((n, 2), dtype=np.float32)
    cryo = dataset.CryoEMDataset(
        image_stack=stack,
        voxel_size=1.5,
        rotation_matrices=rots,
        translations=trans,
        CTF_params=ctf_params,
        premultiplied_ctf=True,
    )

    sub = dataset.subsample_cryoem_dataset(cryo, np.array([0, 2], dtype=np.int32))
    assert sub.premultiplied_ctf is True


def test_subsampled_image_stack_subset_generator_maps_local_to_original_indices():
    class _BackingStack:
        def __init__(self, D=4):
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            # Emit indices back so caller can verify mapping.
            imgs = np.zeros((subset_indices.size, self.D * self.D), dtype=np.complex64)
            yield imgs, subset_indices, subset_indices

    backing = _BackingStack(D=4)
    wrapped = dataset._SubsampledImageStack(backing, subset_indices=np.array([7, 3, 5], dtype=np.int32))
    gen = wrapped.get_dataset_subset_generator(batch_size=8, subset_indices=np.array([2, 0], dtype=np.int32))
    imgs, local_pidx, local_iidx = next(gen)
    assert imgs.shape[0] == 2
    # Returned indices are local (contiguous) by wrapper contract.
    np.testing.assert_array_equal(local_pidx, np.array([2, 0], dtype=np.int32))
    np.testing.assert_array_equal(local_iidx, np.array([2, 0], dtype=np.int32))


def test_subsampled_image_stack_image_subset_generator_alias():
    class _BackingStack:
        def __init__(self, D=4):
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            imgs = np.zeros((subset_indices.size, self.D * self.D), dtype=np.complex64)
            yield imgs, subset_indices, subset_indices

    wrapped = dataset._SubsampledImageStack(_BackingStack(), subset_indices=np.array([4, 8], dtype=np.int32))
    gen = wrapped.get_image_subset_generator(batch_size=4, subset_indices=np.array([1], dtype=np.int32))
    _imgs, pidx, iidx = next(gen)
    np.testing.assert_array_equal(pidx, np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(iidx, np.array([1], dtype=np.int32))


def test_subsampled_image_stack_image_subset_generator_none_emits_full_local_range():
    class _BackingStack:
        def __init__(self, D=4):
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            for start in range(0, subset_indices.size, batch_size):
                chunk = subset_indices[start:start + batch_size]
                imgs = np.zeros((chunk.size, self.D * self.D), dtype=np.complex64)
                yield imgs, chunk, chunk

    wrapped = dataset._SubsampledImageStack(_BackingStack(), subset_indices=np.array([4, 8, 2], dtype=np.int32))
    got = []
    for _imgs, pidx, iidx in wrapped.get_image_subset_generator(batch_size=2, subset_indices=None):
        np.testing.assert_array_equal(pidx, iidx)
        got.extend(np.asarray(iidx).reshape(-1).tolist())
    assert got == [0, 1, 2]


def test_subsampled_image_stack_image_generator_alias_emits_all_local_indices():
    class _BackingStack:
        def __init__(self, D=4):
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            for start in range(0, subset_indices.size, batch_size):
                chunk = subset_indices[start:start + batch_size]
                imgs = np.zeros((chunk.size, self.D * self.D), dtype=np.complex64)
                yield imgs, chunk, chunk

    wrapped = dataset._SubsampledImageStack(_BackingStack(), subset_indices=np.array([9, 5, 1, 6], dtype=np.int32))
    got = []
    for _imgs, pidx, iidx in wrapped.get_image_generator(batch_size=3):
        np.testing.assert_array_equal(pidx, iidx)
        got.extend(np.asarray(iidx).reshape(-1).tolist())
    assert got == [0, 1, 2, 3]


def test_subsampled_image_stack_prefers_backing_image_subset_generator_when_available():
    class _BackingStack:
        def __init__(self, D=4):
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            # Simulate particle-index semantics (wrong for image remapping).
            bad = subset_indices + 1000
            imgs = np.zeros((subset_indices.size, self.D * self.D), dtype=np.complex64)
            yield imgs, bad, bad

        def get_image_subset_generator(self, batch_size, subset_indices, num_workers=0):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            imgs = np.zeros((subset_indices.size, self.D * self.D), dtype=np.complex64)
            yield imgs, subset_indices, subset_indices

    wrapped = dataset._SubsampledImageStack(_BackingStack(), subset_indices=np.array([7, 2, 7, 1], dtype=np.int32))
    req = np.array([3, 0, 2], dtype=np.int32)
    gen = wrapped.get_dataset_subset_generator(batch_size=2, subset_indices=req)
    got = []
    for _imgs, pidx, iidx in gen:
        np.testing.assert_array_equal(pidx, iidx)
        got.extend(np.asarray(iidx).reshape(-1).tolist())
    assert got == [3, 0, 2]


def test_subsampled_image_stack_subset_generator_accepts_boolean_mask():
    class _BackingStack:
        def __init__(self, D=4):
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            imgs = np.zeros((subset_indices.size, self.D * self.D), dtype=np.complex64)
            yield imgs, subset_indices, subset_indices

    wrapped = dataset._SubsampledImageStack(_BackingStack(), subset_indices=np.array([4, 8, 2, 9], dtype=np.int32))
    mask = np.array([False, True, False, True], dtype=bool)
    _imgs, pidx, iidx = next(wrapped.get_dataset_subset_generator(batch_size=8, subset_indices=mask))
    np.testing.assert_array_equal(pidx, np.array([1, 3], dtype=np.int32))
    np.testing.assert_array_equal(iidx, np.array([1, 3], dtype=np.int32))


def test_subsampled_image_stack_subset_generator_rejects_bad_subset_indices():
    class _BackingStack:
        def __init__(self, D=4):
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            imgs = np.zeros((subset_indices.size, self.D * self.D), dtype=np.complex64)
            yield imgs, subset_indices, subset_indices

    wrapped = dataset._SubsampledImageStack(_BackingStack(), subset_indices=np.array([4, 8, 2, 9], dtype=np.int32))
    with pytest.raises(ValueError, match="boolean mask must be 1D"):
        list(wrapped.get_dataset_subset_generator(batch_size=8, subset_indices=np.array([[True, False, True, False]], dtype=bool)))
    with pytest.raises(ValueError, match="must match number of images"):
        list(wrapped.get_dataset_subset_generator(batch_size=8, subset_indices=np.array([True, False], dtype=bool)))
    with pytest.raises(ValueError, match="non-negative"):
        list(wrapped.get_dataset_subset_generator(batch_size=8, subset_indices=np.array([-1], dtype=np.int32)))
    with pytest.raises(ValueError, match="number of images"):
        list(wrapped.get_dataset_subset_generator(batch_size=8, subset_indices=np.array([4], dtype=np.int32)))
    with pytest.raises(TypeError, match="integer or boolean mask"):
        list(wrapped.get_dataset_subset_generator(batch_size=8, subset_indices=np.array([1.5], dtype=np.float32)))


def test_subsampled_image_stack_subset_generator_handles_multiple_underlying_batches():
    class _BackingStack:
        def __init__(self, D=4):
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            for start in range(0, subset_indices.size, batch_size):
                chunk = subset_indices[start:start + batch_size]
                imgs = np.zeros((chunk.size, self.D * self.D), dtype=np.complex64)
                # Return original-image indices from backing stack.
                yield imgs, chunk, chunk

    wrapped = dataset._SubsampledImageStack(_BackingStack(), subset_indices=np.array([7, 3, 5, 9], dtype=np.int32))
    gen = wrapped.get_dataset_subset_generator(batch_size=2, subset_indices=np.array([3, 0, 2, 1], dtype=np.int32))

    got = []
    for _imgs, pidx, iidx in gen:
        np.testing.assert_array_equal(pidx, iidx)
        got.extend(np.asarray(iidx).reshape(-1).tolist())

    # Must preserve requested local subset order across all emitted batches.
    assert got == [3, 0, 2, 1]


def test_subsampled_image_stack_dataset_generator_emits_all_local_indices():
    class _BackingStack:
        def __init__(self, D=4):
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            imgs = np.zeros((subset_indices.size, self.D * self.D), dtype=np.complex64)
            yield imgs, subset_indices, subset_indices

    wrapped = dataset._SubsampledImageStack(_BackingStack(), subset_indices=np.array([10, 4, 8, 2, 6], dtype=np.int32))
    got = []
    for _imgs, pidx, iidx in wrapped.get_dataset_generator(batch_size=2):
        np.testing.assert_array_equal(pidx, iidx)
        got.extend(np.asarray(iidx).reshape(-1).tolist())

    np.testing.assert_array_equal(np.asarray(got, dtype=np.int32), np.array([0, 1, 2, 3, 4], dtype=np.int32))


def test_subsampled_image_stack_dataset_generator_preserves_duplicate_original_indices():
    class _BackingStack:
        def __init__(self, D=4):
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            for start in range(0, subset_indices.size, batch_size):
                chunk = subset_indices[start:start + batch_size]
                imgs = np.zeros((chunk.size, self.D * self.D), dtype=np.complex64)
                yield imgs, chunk, chunk

    # Original index 7 appears twice in the kept subset.
    wrapped = dataset._SubsampledImageStack(_BackingStack(), subset_indices=np.array([7, 3, 7, 5], dtype=np.int32))
    got = []
    for _imgs, pidx, iidx in wrapped.get_dataset_generator(batch_size=2):
        np.testing.assert_array_equal(pidx, iidx)
        got.extend(np.asarray(iidx).reshape(-1).tolist())

    np.testing.assert_array_equal(np.asarray(got, dtype=np.int32), np.array([0, 1, 2, 3], dtype=np.int32))


def test_subsampled_image_stack_subset_generator_preserves_duplicate_requests():
    class _BackingStack:
        def __init__(self, D=4):
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            for start in range(0, subset_indices.size, batch_size):
                chunk = subset_indices[start:start + batch_size]
                imgs = np.zeros((chunk.size, self.D * self.D), dtype=np.complex64)
                yield imgs, chunk, chunk

    wrapped = dataset._SubsampledImageStack(_BackingStack(), subset_indices=np.array([7, 3, 7, 5], dtype=np.int32))
    gen = wrapped.get_dataset_subset_generator(batch_size=2, subset_indices=np.array([2, 0, 2, 1], dtype=np.int32))

    got = []
    for _imgs, pidx, iidx in gen:
        np.testing.assert_array_equal(pidx, iidx)
        got.extend(np.asarray(iidx).reshape(-1).tolist())

    # Must preserve requested local order and duplicates.
    assert got == [2, 0, 2, 1]


def test_subsampled_image_stack_generator_raises_on_unmapped_underlying_index():
    class _BackingStack:
        def __init__(self, D=4):
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            imgs = np.zeros((subset_indices.size, self.D * self.D), dtype=np.complex64)
            # Emit an index not present in subset_indices to simulate backend inconsistency.
            bad = subset_indices.copy()
            bad[0] = 999
            yield imgs, bad, bad

    wrapped = dataset._SubsampledImageStack(_BackingStack(), subset_indices=np.array([7, 3, 5], dtype=np.int32))
    gen = wrapped.get_dataset_subset_generator(batch_size=4, subset_indices=np.array([0, 1], dtype=np.int32))
    with pytest.raises(KeyError, match="Original index 999"):
        next(gen)


def test_subsample_cryoem_dataset_preserves_duplicate_requested_indices():
    n = 5

    class _GenImageStack:
        def __init__(self, n=5, D=4):
            self.n_images = n
            self.Np = n
            self.D = D
            self.unpadded_D = D
            self.padding = 0
            self.image_shape = (D, D)
            self.mask = np.ones((D, D), dtype=np.float32)

        def process_images(self, images, apply_image_mask=True):
            return np.asarray(images)

        def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers=0, **kwargs):
            subset_indices = np.asarray(subset_indices, dtype=np.int32)
            imgs = np.zeros((subset_indices.size, self.D * self.D), dtype=np.complex64)
            yield imgs, subset_indices, subset_indices

    stack = _GenImageStack(n=n, D=4)
    ctf_params = np.arange(n * 9, dtype=np.float32).reshape(n, 9)
    rots = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    rots[:, 0, 0] = np.arange(n, dtype=np.float32)
    trans = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    cryo = dataset.CryoEMDataset(
        image_stack=stack,
        voxel_size=1.5,
        rotation_matrices=rots,
        translations=trans,
        CTF_params=ctf_params,
    )

    requested = np.array([4, 1, 4], dtype=np.int32)
    sub = dataset.subsample_cryoem_dataset(cryo, requested)
    np.testing.assert_array_equal(sub.dataset_indices, requested)
    np.testing.assert_array_equal(sub.translations, trans[requested])
    np.testing.assert_array_equal(sub.CTF_params, ctf_params[requested])
    # Duplicate source index must be duplicated in metadata, not deduplicated.
    np.testing.assert_array_equal(sub.rotation_matrices[:, 0, 0], np.array([4.0, 1.0, 4.0], dtype=np.float32))


def test_subsample_cryoem_dataset_rejects_non_1d_boolean_mask():
    n = 3
    ctf_params = np.zeros((n, 9), dtype=np.float32)
    rots = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    trans = np.zeros((n, 2), dtype=np.float32)
    cryo = dataset.CryoEMDataset(
        image_stack=None,
        voxel_size=1.5,
        rotation_matrices=rots,
        translations=trans,
        CTF_params=ctf_params,
        grid_size=4,
    )

    with pytest.raises(ValueError, match="boolean mask must be 1D"):
        dataset.subsample_cryoem_dataset(cryo, np.array([[True, False, True]], dtype=bool))


def test_subsample_cryoem_dataset_rejects_wrong_length_boolean_mask():
    n = 3
    ctf_params = np.zeros((n, 9), dtype=np.float32)
    rots = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    trans = np.zeros((n, 2), dtype=np.float32)
    cryo = dataset.CryoEMDataset(
        image_stack=None,
        voxel_size=1.5,
        rotation_matrices=rots,
        translations=trans,
        CTF_params=ctf_params,
        grid_size=4,
    )

    with pytest.raises(ValueError, match="must match number of images"):
        dataset.subsample_cryoem_dataset(cryo, np.array([True, False], dtype=bool))


def test_subsample_cryoem_dataset_rejects_out_of_range_indices():
    n = 3
    ctf_params = np.zeros((n, 9), dtype=np.float32)
    rots = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    trans = np.zeros((n, 2), dtype=np.float32)
    cryo = dataset.CryoEMDataset(
        image_stack=None,
        voxel_size=1.5,
        rotation_matrices=rots,
        translations=trans,
        CTF_params=ctf_params,
        grid_size=4,
    )

    with pytest.raises(ValueError, match=">= number of images"):
        dataset.subsample_cryoem_dataset(cryo, np.array([0, 3], dtype=np.int32))

def test_get_split_indices_from_empty_ind_file_raises(tmp_path):
    ind_file = tmp_path / "empty.pkl"
    with open(ind_file, "wb") as f:
        pickle.dump(np.array([], dtype=np.int32), f)
    with pytest.raises(ValueError, match="No valid indices found"):
        dataset.get_split_indices("unused.star", ind_file=str(ind_file))


def test_get_split_tilt_indices_accepts_array_inputs_for_all_index_args(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [
                np.array([0, 1], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
                np.array([4, 5], dtype=np.int32),
            ]
            tilts_to_particles = [0, 0, 1, 1, 2, 2]
            return particles_to_tilts, tilts_to_particles

        def __init__(self, particles_file, datadir=None):
            self.tilt_numbers = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3]), np.array([4, 5])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    out = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        # Keep only particles 0 and 2.
        tilt_ind_file=np.array([0, 2], dtype=np.int32),
        # Keep only these image ids globally.
        ind_file=np.array([0, 1, 4, 5], dtype=np.int32),
        # Precomputed particle halfsets as an in-memory value (not file path).
        particle_halfset_indices_file=[np.array([0], dtype=np.int32), np.array([2], dtype=np.int32)],
        ntilts=1,
    )
    np.testing.assert_array_equal(out[0], np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([4], dtype=np.int32))


def test_get_split_tilt_indices_accepts_boolean_masks_for_particle_and_image_filters(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [
                np.array([0, 1], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
                np.array([4, 5], dtype=np.int32),
            ]
            tilts_to_particles = [0, 0, 1, 1, 2, 2]
            return particles_to_tilts, tilts_to_particles

        def __init__(self, particles_file, datadir=None):
            self.tilt_numbers = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3]), np.array([4, 5])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    split = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        # Keep particles 0 and 2.
        tilt_ind_file=np.array([True, False, True], dtype=bool),
        # Keep only image ids 0 and 4 (boolean mask over 6 images).
        ind_file=np.array([True, False, False, False, True, False], dtype=bool),
        particle_halfset_indices_file=[np.array([0], dtype=np.int32), np.array([2], dtype=np.int32)],
    )
    np.testing.assert_array_equal(split[0], np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(split[1], np.array([4], dtype=np.int32))


def test_get_split_tilt_indices_rejects_wrong_length_boolean_masks(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [
                np.array([0, 1], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
                np.array([4, 5], dtype=np.int32),
            ]
            tilts_to_particles = [0, 0, 1, 1, 2, 2]
            return particles_to_tilts, tilts_to_particles

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3]), np.array([4, 5])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="must match number of particles"):
        dataset.get_split_tilt_indices(
            particles_file="particles.star",
            tilt_ind_file=np.array([True, False], dtype=bool),
        )

    with pytest.raises(ValueError, match="must match number of images"):
        dataset.get_split_tilt_indices(
            particles_file="particles.star",
            # valid particle mask length (3)
            tilt_ind_file=np.array([True, False, True], dtype=bool),
            # invalid image mask length (should be 6)
            ind_file=np.array([True, False, True], dtype=bool),
        )


def test_get_split_tilt_indices_rejects_non_1d_mask_and_index_arrays(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [
                np.array([0, 1], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
                np.array([4, 5], dtype=np.int32),
            ]
            tilts_to_particles = [0, 0, 1, 1, 2, 2]
            return particles_to_tilts, tilts_to_particles

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3]), np.array([4, 5])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="boolean mask must be 1D"):
        dataset.get_split_tilt_indices(
            particles_file="particles.star",
            tilt_ind_file=np.array([[True, False, True]], dtype=bool),
        )

    with pytest.raises(ValueError, match="ids must be 1D"):
        dataset.get_split_tilt_indices(
            particles_file="particles.star",
            tilt_ind_file=np.array([[0, 2]], dtype=np.int32),
        )

    with pytest.raises(ValueError, match="ind_file image ids must be 1D"):
        dataset.get_split_tilt_indices(
            particles_file="particles.star",
            tilt_ind_file=np.array([0, 2], dtype=np.int32),
            ind_file=np.array([[0, 4]], dtype=np.int32),
        )


def test_get_split_tilt_indices_sanitizes_tilt_ind_file_values(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [
                np.array([0, 1], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
                np.array([4, 5], dtype=np.int32),
            ]
            tilts_to_particles = [0, 0, 1, 1, 2, 2]
            return particles_to_tilts, tilts_to_particles

        def __init__(self, particles_file, datadir=None):
            self.tilt_numbers = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3]), np.array([4, 5])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    out = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        # Includes duplicate and out-of-range particle ids.
        tilt_ind_file=np.array([2, 2, -1, 99, 0], dtype=np.int32),
        particle_halfset_indices_file=[np.array([2, 0], dtype=np.int32), np.array([], dtype=np.int32)],
    )
    np.testing.assert_array_equal(out[0], np.array([4, 5, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([], dtype=np.int32))


def test_get_split_tilt_indices_with_only_invalid_tilt_ind_file_returns_empty(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)]
            tilts_to_particles = [0, 0, 1, 1]
            return particles_to_tilts, tilts_to_particles

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    out = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        tilt_ind_file=np.array([-10, 50], dtype=np.int32),
    )
    np.testing.assert_array_equal(out[0], np.array([], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([], dtype=np.int32))


def test_get_split_tilt_indices_zero_tilts_skips_tilt_dataset_instantiation(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [
                np.array([0, 1], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
            ]
            tilts_to_particles = [0, 0, 1, 1]
            return particles_to_tilts, tilts_to_particles

        def __init__(self, *args, **kwargs):
            raise AssertionError("TiltSeriesData should not be instantiated for ntilts <= 0")

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    out = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        ntilts=0,
        particle_halfset_indices_file=[
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
        ],
    )
    np.testing.assert_array_equal(out[0], np.array([], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([], dtype=np.int32))


def test_get_split_tilt_indices_negative_tilts_returns_empty_without_tilt_dataset(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [
                np.array([0, 1], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
            ]
            tilts_to_particles = [0, 0, 1, 1]
            return particles_to_tilts, tilts_to_particles

        def __init__(self, *args, **kwargs):
            raise AssertionError("TiltSeriesData should not be instantiated for ntilts <= 0")

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    out = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        ntilts=-1,
        particle_halfset_indices_file=[
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
        ],
    )
    np.testing.assert_array_equal(out[0], np.array([], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([], dtype=np.int32))


def test_get_split_tilt_indices_rejects_invalid_halfset_container(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)]
            tilts_to_particles = [0, 0, 1, 1]
            return particles_to_tilts, tilts_to_particles

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="exactly two halfsets"):
        dataset.get_split_tilt_indices(
            particles_file="particles.star",
            particle_halfset_indices_file=[np.array([0], dtype=np.int32)],
        )


def test_get_split_tilt_indices_ind_filter_preserves_allowed_image_order(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [
                np.array([5, 1], dtype=np.int32),
                np.array([4, 2], dtype=np.int32),
            ]
            tilts_to_particles = [0, 0, 1, 1, 1, 0]
            return particles_to_tilts, tilts_to_particles

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([5, 1]), np.array([4, 2])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    split = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        tilt_ind_file=np.array([0, 1], dtype=np.int32),
        ind_file=np.array([2, 5, 4], dtype=np.int32),
        particle_halfset_indices_file=[np.array([0, 1], dtype=np.int32), np.array([], dtype=np.int32)],
    )
    # Preserve source ordering from selected particles/tilts: [5,1,4,2] -> keep {2,5,4} => [5,4,2].
    np.testing.assert_array_equal(split[0], np.array([5, 4, 2], dtype=np.int32))
    np.testing.assert_array_equal(split[1], np.array([], dtype=np.int32))


def test_get_split_tilt_indices_preserves_precomputed_particle_halfset_order(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [
                np.array([0], dtype=np.int32),
                np.array([1], dtype=np.int32),
                np.array([2], dtype=np.int32),
            ]
            tilts_to_particles = [0, 1, 2]
            return particles_to_tilts, tilts_to_particles

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0]), np.array([1]), np.array([2])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    split = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        # valid particles are only 1 and 2
        tilt_ind_file=np.array([1, 2], dtype=np.int32),
        # halfset keeps custom order [2,0,1] and should become [2,1] after filtering
        particle_halfset_indices_file=[np.array([2, 0, 1], dtype=np.int32), np.array([], dtype=np.int32)],
    )
    np.testing.assert_array_equal(split[0], np.array([2, 1], dtype=np.int32))
    np.testing.assert_array_equal(split[1], np.array([], dtype=np.int32))


def test_get_split_tilt_indices_ignores_out_of_range_precomputed_particle_ids(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [
                np.array([0, 1], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
                np.array([4, 5], dtype=np.int32),
            ]
            tilts_to_particles = [0, 0, 1, 1, 2, 2]
            return particles_to_tilts, tilts_to_particles

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0, 1]), np.array([2, 3]), np.array([4, 5])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    split = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        # Halfset includes invalid ids: -2 and 99 should be dropped.
        particle_halfset_indices_file=[np.array([2, -2, 0, 99], dtype=np.int32), np.array([1], dtype=np.int32)],
    )
    # Valid first-half order should remain [2, 0] after dropping invalid ids.
    np.testing.assert_array_equal(split[0], np.array([4, 5, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(split[1], np.array([2, 3], dtype=np.int32))


def test_get_split_tilt_indices_deduplicates_precomputed_particle_ids_preserving_order(monkeypatch):
    class _FakeTiltSeriesData:
        @staticmethod
        def parse_particle_tilt(_particles_file):
            particles_to_tilts = [
                np.array([0], dtype=np.int32),
                np.array([1], dtype=np.int32),
                np.array([2], dtype=np.int32),
            ]
            tilts_to_particles = [0, 1, 2]
            return particles_to_tilts, tilts_to_particles

    monkeypatch.setattr(dataset.tilt_dataset, "TiltSeriesData", _FakeTiltSeriesData)
    monkeypatch.setattr(
        dataset.tilt_dataset,
        "tilt_series_indices_to_image_indices",
        lambda particle_ind, particles_file: np.concatenate(
            [[np.array([0]), np.array([1]), np.array([2])][int(i)] for i in np.asarray(particle_ind)]
        ) if len(particle_ind) > 0 else np.array([], dtype=np.int32),
    )

    split = dataset.get_split_tilt_indices(
        particles_file="particles.star",
        # duplicates in first halfset should be removed but order preserved -> [2,0,1]
        particle_halfset_indices_file=[np.array([2, 0, 2, 1, 1], dtype=np.int32), np.array([], dtype=np.int32)],
    )
    np.testing.assert_array_equal(split[0], np.array([2, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(split[1], np.array([], dtype=np.int32))


def test_figure_out_halfsets_tilt_series_with_halfsets_file_uses_tilt_splitter(monkeypatch):
    captured = {}

    def _fake_get_split_tilt_indices(*args, **kwargs):
        captured["kwargs"] = kwargs
        return [np.array([0, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)]

    args = SimpleNamespace(
        halfsets="halfsets.pkl",
        tilt_series=True,
        tilt_series_ctf="relion5",
        particles="particles.star",
        ind=np.array([0, 1, 2, 3], dtype=np.int32),
        tilt_ind=np.array([0, 2], dtype=np.int32),
        ntilts=2,
        datadir="/tmp/data",
        strip_prefix=None,
        n_images=-1,
    )

    monkeypatch.setattr(dataset, "get_split_tilt_indices", _fake_get_split_tilt_indices)
    out = dataset.figure_out_halfsets(args)

    np.testing.assert_array_equal(out[0], np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.array([1, 3], dtype=np.int32))
    assert captured["kwargs"]["ind_file"] is args.ind
    assert captured["kwargs"]["tilt_ind_file"] is args.tilt_ind
    assert captured["kwargs"]["ntilts"] == 2
    assert captured["kwargs"]["particle_halfset_indices_file"] == "halfsets.pkl"
