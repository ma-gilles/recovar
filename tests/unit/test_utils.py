from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("mrcfile")
pytest.importorskip("starfile")

import recovar.fourier_transform_utils as fourier_transform_utils
from recovar import utils

pytestmark = pytest.mark.unit


def test_make_radial_image_matches_manual_indexing():
    average = jnp.array([10.0, 20.0], dtype=jnp.float32)
    radial_distances = fourier_transform_utils.get_grid_of_radial_distances((3, 3), scaled=False).astype(int).reshape(-1)
    expected = np.asarray(average)[np.asarray(radial_distances)]

    out = utils.make_radial_image(average, (3, 3), extend_last_frequency=False)
    np.testing.assert_allclose(np.asarray(out), expected)


def test_make_radial_image_extend_last_frequency_true():
    average = jnp.array([10.0, 20.0], dtype=jnp.float32)
    out = utils.make_radial_image(average, (5, 5), extend_last_frequency=True)
    assert out.shape == (25,)
    # Outer ring indices should map to appended last value.
    assert np.max(np.asarray(out)) == pytest.approx(20.0)


def test_batch_iter_helpers():
    assert [list(x) for x in utils.index_batch_iter(10, 4)] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert [list(x) for x in utils.subset_batch_iter([5, 1, 9, 2], 2)] == [[5, 1], [9, 2]]
    paired = list(utils.subset_and_indices_batch_iter([5, 1, 9, 2], 2))
    assert [list(x) for x, _ in paired] == [[0, 1], [2, 3]]
    assert [list(y) for _, y in paired] == [[5, 1], [9, 2]]


def test_estimate_variance():
    u = np.array([[1 + 1j, 2], [3, 4 - 1j]], dtype=np.complex64)
    s = np.array([2.0, 3.0], dtype=np.float32)
    out = utils.estimate_variance(u, s)
    expected = np.sum(np.abs(u) ** 2 * s[..., None], axis=0)
    np.testing.assert_allclose(out, expected)


def test_get_gpu_memory_helpers_without_gpu(monkeypatch):
    monkeypatch.setattr(utils, "GPU_MEMORY_LIMIT", None)
    monkeypatch.setattr(utils, "jax_has_gpu", lambda: False)
    assert utils.get_gpu_memory_total() == 80
    assert utils.get_gpu_memory_used() == 0
    assert utils.get_peak_gpu_memory_used() == 0


def test_get_gpu_memory_total_limit_override(monkeypatch):
    monkeypatch.setattr(utils, "GPU_MEMORY_LIMIT", 12)
    assert utils.get_gpu_memory_total() == 12


def test_get_gpu_memory_helpers_with_mocked_stats(monkeypatch):
    class FakeDevice:
        @staticmethod
        def memory_stats():
            return {
                "bytes_limit": 42 * 10**9,
                "bytes_in_use": 7 * 10**9,
                "peak_bytes_in_use": 9 * 10**9,
            }

    monkeypatch.setattr(utils, "GPU_MEMORY_LIMIT", None)
    monkeypatch.setattr(utils, "jax_has_gpu", lambda: True)
    monkeypatch.setattr(utils.jax, "local_devices", lambda: [FakeDevice()])
    assert utils.get_gpu_memory_total() == 42
    assert utils.get_gpu_memory_used() == 7
    assert utils.get_peak_gpu_memory_used() == 9


def test_report_memory_device_with_logger(monkeypatch):
    captured = {"msg": None}
    logger = SimpleNamespace(info=lambda msg: captured.__setitem__("msg", msg))
    monkeypatch.setattr(utils, "get_gpu_memory_used", lambda device=0: 1)
    monkeypatch.setattr(utils, "get_peak_gpu_memory_used", lambda device=0: 2)
    monkeypatch.setattr(utils, "get_gpu_memory_total", lambda device=0: 3)
    monkeypatch.setattr(utils, "get_process_memory_used", lambda: 4)
    utils.report_memory_device(logger=logger)
    assert "GPU mem in use:1; peak:2; total available:3, process mem in use:4" == captured["msg"]


def test_report_memory_device_without_logger(monkeypatch, capsys):
    monkeypatch.setattr(utils, "get_gpu_memory_used", lambda device=0: 1)
    monkeypatch.setattr(utils, "get_peak_gpu_memory_used", lambda device=0: 2)
    monkeypatch.setattr(utils, "get_gpu_memory_total", lambda device=0: 3)
    monkeypatch.setattr(utils, "get_process_memory_used", lambda: 4)
    utils.report_memory_device(logger=None)
    captured = capsys.readouterr()
    assert "GPU mem in use:1; peak:2; total available:3, process mem in use:4" in captured.out


def test_misc_size_and_shape_helpers():
    arr = np.zeros((10,), dtype=np.float64)
    assert utils.get_size_in_gb(arr) == arr.size * arr.itemsize / 1e9
    assert utils.guess_grid_size_from_vol_size(27) == 3
    assert utils.guess_vol_shape_from_vol_size(27) == (3, 3, 3)


def test_write_and_load_mrc_roundtrip(tmp_path):
    vol = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    out = tmp_path / "vol.mrc"
    utils.write_mrc(str(out), vol, voxel_size=2.5)
    loaded, voxel = utils.load_mrc(str(out), return_voxel_size=True)
    np.testing.assert_allclose(loaded, vol)
    assert float(voxel.x) == pytest.approx(2.5)


def test_symmetrize_ft_volume_shape_and_dtype():
    vol = jnp.ones((8,), dtype=jnp.complex64)
    out = utils.symmetrize_ft_volume(vol, (2, 2, 2))
    assert out.shape == (8,)
    assert jnp.iscomplexobj(out)


def test_dataset_index_helpers():
    cryos = [SimpleNamespace(dataset_indices=np.array([2, 5])), SimpleNamespace(dataset_indices=np.array([1, 3]))]
    all_idx = utils.get_all_dataset_indices(cryos)
    inv_idx = utils.get_inverse_dataset_indices(cryos)
    np.testing.assert_array_equal(all_idx, np.array([2, 5, 1, 3]))
    np.testing.assert_array_equal(all_idx[inv_idx], np.sort(all_idx))


def test_batch_size_helpers_and_validation():
    assert utils.get_image_batch_size(256, 10) > 0
    assert utils.get_vol_batch_size(128, 10) > 0
    assert utils.get_column_batch_size(128, 10) > 0
    with pytest.raises(ValueError):
        utils.get_image_batch_size(0, 10)
    with pytest.raises(ValueError):
        utils.get_vol_batch_size(8, 0)
    with pytest.raises(ValueError):
        utils.get_column_batch_size(-1, 10)


def test_latent_and_embedding_batch_size(caplog):
    test_pts = np.zeros((100, 3), dtype=np.float32)
    assert utils.get_latent_density_batch_size(test_pts, zdim=2, gpu_memory=4) >= 1

    basis = np.zeros((2000, 2000), dtype=np.float64)
    image_size = 64 * 64
    contrast_grid = np.zeros((16,), dtype=np.float64)
    with caplog.at_level("WARNING"):
        out = utils.get_embedding_batch_size(basis, image_size, contrast_grid, zdim=8, gpu_memory=1)
    assert out == 1
    out2 = utils.get_embedding_batch_size(
        basis=np.zeros((10, 10), dtype=np.float64),
        image_size=image_size,
        contrast_grid=contrast_grid,
        zdim=2,
        gpu_memory=16,
    )
    assert out2 >= 1


def test_options_and_pickle_helpers(tmp_path):
    args = SimpleNamespace(
        mask="none",
        zdim=8,
        correct_contrast=True,
        ignore_zero_frequency=False,
        keep_intermediate=True,
    )
    opts = utils.make_algorithm_options(args)
    assert opts["contrast"] == "contrast_qr"
    assert opts["zs_dim_to_test"] == 8

    path = tmp_path / "obj.pkl"
    obj = {"a": 1, "b": [1, 2]}
    utils.pickle_dump(obj, path)
    assert utils.pickle_load(path) == obj


def test_get_variances_and_batch_index_helpers():
    cov = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
    picked = np.array([2, 0, 1])
    out = utils.get_variances(cov, picked_frequencies=picked)
    np.testing.assert_array_equal(out, np.array([31, 12, 23]))

    assert utils.get_number_of_index_batch(10, 3) == 4
    assert utils.get_number_of_index_batch(9, 3) == 3
    assert utils.get_batch_of_indices(10, 3, 2) == (6, 9)
    np.testing.assert_array_equal(utils.get_batch_of_indices_arange(10, 3, 2), np.array([6, 7, 8]))
    assert utils.get_batch_of_indices(10, 3, 3) == (9, 10)


def test_jax_has_gpu_returns_bool():
    assert isinstance(utils.jax_has_gpu(), bool)


def test_dtype_to_real():
    assert utils.dtype_to_real(np.dtype(np.complex64)) == np.dtype(np.float32)
    assert utils.dtype_to_real(np.dtype(np.complex128)) == np.dtype(np.float64)


def test_relion_rotation_roundtrip():
    scipy_rot = pytest.importorskip("scipy.spatial.transform").Rotation
    mats = scipy_rot.random(5, random_state=0).as_matrix()
    euler = utils.R_to_relion(mats, degrees=True)
    back = utils.R_from_relion(euler, degrees=True)
    np.testing.assert_allclose(back, mats, atol=1e-6, rtol=1e-6)


def test_relion_rotation_input_shapes():
    ident = np.eye(3)
    out = utils.R_to_relion(ident)
    assert out.shape == (1, 3)
    with pytest.raises(ValueError):
        utils.R_to_relion(np.ones((2, 2)))


def test_write_starfile_minimal(tmp_path):
    n = 3
    ctf = np.zeros((n, 7), dtype=np.float32)
    ctf[:, 3] = 300.0
    ctf[:, 4] = 2.7
    ctf[:, 5] = 0.1
    rots = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    trans = np.zeros((n, 2), dtype=np.float32)
    out = tmp_path / "particles.star"
    utils.write_starfile(
        CTF_params=ctf,
        rotation_matrices=rots,
        translations=trans,
        voxel_size=1.5,
        grid_size=64,
        particles_file="particles.mrcs",
        output_filename=str(out),
        halfset_indices=np.array([1, 2, 1]),
    )
    assert out.exists()


def test_downsample_vol_by_fourier_truncation():
    vol = np.ones((8, 8, 8), dtype=np.float32)
    out = utils.downsample_vol_by_fourier_truncation(vol, target_grid_size=4)
    assert out.shape == (4, 4, 4)
    out_same = utils.downsample_vol_by_fourier_truncation(vol, target_grid_size=8)
    assert out_same.shape == (8, 8, 8)
    out_odd = utils.downsample_vol_by_fourier_truncation(vol, target_grid_size=5)
    assert out_odd.shape == (5, 5, 5)


def test_downsample_vol_by_fourier_truncation_validates_target():
    vol = np.ones((8, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        utils.downsample_vol_by_fourier_truncation(vol, target_grid_size=0)
    with pytest.raises(ValueError):
        utils.downsample_vol_by_fourier_truncation(vol, target_grid_size=9)


def test_basic_config_logger_and_duplicate_filter(tmp_path):
    utils.basic_config_logger(str(tmp_path))
    logger = utils.logging.getLogger("recovar-test")
    logger.info("hello")
    assert (tmp_path / "run.log").exists()

    f = utils.DuplicateFilter()
    r1 = SimpleNamespace(msg="same")
    r2 = SimpleNamespace(msg="same")
    assert f.filter(r1) is True
    assert f.filter(r2) is False
