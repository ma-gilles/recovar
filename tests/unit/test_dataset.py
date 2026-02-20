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

    def _fake_fwd(volume, ctf, rots, image_shape, volume_shape, voxel_size, ctf_fun, disc_type, skip_ctf):
        called["disc_type"] = disc_type
        called["skip_ctf"] = skip_ctf
        return np.ones((len(ctf), image_shape[0] * image_shape[1]), dtype=np.complex64) * (2 + 3j)

    monkeypatch.setattr(core, "forward_model_from_map", _fake_fwd)
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
