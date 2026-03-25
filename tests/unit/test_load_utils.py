import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.data_io import load_utils
from helpers import tiny_synthetic
from recovar import utils

pytestmark = pytest.mark.unit


def test_load_ctf_params_rescales_apix_and_drops_size_column(monkeypatch):
    ctf = np.array(
        [
            [128, 1.5, 10000, 11000, 0.0, 300, 2.7, 0.1, 0.0],
            [128, 1.5, 12000, 13000, 5.0, 300, 2.7, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: ctf.copy())

    out = load_utils.load_ctf_params(D=64, ctf_params_pkl="dummy.pkl")

    assert out.shape == (2, 8)
    # original_D * original_Apix / D = 128 * 1.5 / 64 = 3.0
    assert np.allclose(out[:, 0], 3.0)
    assert np.allclose(out[:, 1], [10000, 12000])


def test_load_ctf_params_rejects_odd_dimension():
    with pytest.raises(ValueError, match="must be even"):
        load_utils.load_ctf_params(D=65, ctf_params_pkl="dummy.pkl")


def test_load_ctf_params_rejects_bad_param_width(monkeypatch):
    bad = np.zeros((3, 8), dtype=np.float32)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: bad)
    with pytest.raises(ValueError, match="Expected 9 CTF parameters"):
        load_utils.load_ctf_params(D=64, ctf_params_pkl="dummy.pkl")


def test_load_ctf_params_accepts_list_payload(monkeypatch):
    ctf_list = [
        [128, 1.5, 10000, 11000, 0.0, 300, 2.7, 0.1, 0.0],
        [128, 1.5, 12000, 13000, 5.0, 300, 2.7, 0.1, 0.0],
    ]
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: ctf_list)

    out = load_utils.load_ctf_params(D=64, ctf_params_pkl="dummy.pkl")
    assert out.shape == (2, 8)
    assert np.allclose(out[:, 0], 3.0)


def test_load_ctf_params_rejects_non_2d_payload(monkeypatch):
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: np.zeros((9,), dtype=np.float32))
    with pytest.raises(ValueError, match="must be a 2D array"):
        load_utils.load_ctf_params(D=64, ctf_params_pkl="dummy.pkl")


def test_load_ctf_params_rejects_nonnumeric_payload(monkeypatch):
    bad = np.array([["a"] * 9], dtype=object)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: bad)
    with pytest.raises(ValueError, match="must be numeric"):
        load_utils.load_ctf_params(D=64, ctf_params_pkl="dummy.pkl")


def test_load_ctf_params_rejects_empty_payload(monkeypatch):
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: np.zeros((0, 9), dtype=np.float32))
    with pytest.raises(ValueError, match="are empty"):
        load_utils.load_ctf_params(D=64, ctf_params_pkl="dummy.pkl")


def test_load_ctf_params_rejects_nonfinite_payload(monkeypatch):
    bad = np.array([[128, 1.5, np.nan, 11000, 0.0, 300, 2.7, 0.1, 0.0]], dtype=np.float32)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: bad)
    with pytest.raises(ValueError, match="non-finite"):
        load_utils.load_ctf_params(D=64, ctf_params_pkl="dummy.pkl")


def test_load_poses_single_file_with_translations_scales_by_D(monkeypatch):
    rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 3, axis=0)
    trans_frac = np.array([[0.0, 0.5], [1.0, 0.25], [0.1, 0.2]], dtype=np.float32)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots, trans_frac))

    rots_out, trans_out, D_out = load_utils.load_poses("poses.pkl", Nimg=3, D=128)

    assert D_out == 128
    assert rots_out.shape == (3, 3, 3)
    assert trans_out.shape == (3, 2)
    assert np.allclose(trans_out, trans_frac * 128)


def test_load_poses_two_file_input(monkeypatch):
    rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0)
    trans_frac = np.array([[0.25, 0.75], [0.5, 0.5]], dtype=np.float32)

    def fake_pickle_load(path):
        if path == "rots.pkl":
            return rots
        if path == "trans.pkl":
            return trans_frac
        raise AssertionError("unexpected path")

    monkeypatch.setattr(load_utils.utils, "pickle_load", fake_pickle_load)
    rots_out, trans_out, _ = load_utils.load_poses(["rots.pkl", "trans.pkl"], Nimg=2, D=64)
    assert np.allclose(rots_out, rots)
    assert np.allclose(trans_out, trans_frac * 64)


def test_load_poses_accepts_list_payload_and_applies_indices(monkeypatch):
    rots_list = [np.eye(3, dtype=np.float32) * (i + 1) for i in range(4)]
    trans_list = [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots_list, trans_list))

    ind = np.array([3, 1, 3], dtype=np.int32)
    rots_out, trans_out, D_out = load_utils.load_poses("poses.pkl", Nimg=3, D=10, ind=ind)
    assert D_out == 10
    np.testing.assert_allclose(rots_out[:, 0, 0], np.array([4.0, 2.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(trans_out, np.array([[0.6, 0.7], [0.2, 0.3], [0.6, 0.7]], dtype=np.float32) * 10.0)


def test_load_poses_index_filter_applies_when_input_is_longer(monkeypatch):
    rots_all = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 5, axis=0)
    trans_all = np.linspace(0.0, 0.9, 10, dtype=np.float32).reshape(5, 2)
    ind = np.array([0, 2, 4], dtype=np.int32)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots_all, trans_all))

    rots_out, trans_out, _ = load_utils.load_poses("poses.pkl", Nimg=3, D=100, ind=ind)
    assert rots_out.shape == (3, 3, 3)
    assert trans_out.shape == (3, 2)
    assert np.allclose(trans_out, trans_all[ind] * 100)


def test_load_poses_index_filter_accepts_boolean_mask_when_input_is_longer(monkeypatch):
    rots_all = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 5, axis=0)
    trans_all = np.linspace(0.0, 0.9, 10, dtype=np.float32).reshape(5, 2)
    mask = np.array([True, False, True, False, True], dtype=bool)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots_all, trans_all))

    rots_out, trans_out, _ = load_utils.load_poses("poses.pkl", Nimg=3, D=80, ind=mask)
    assert rots_out.shape == (3, 3, 3)
    np.testing.assert_allclose(trans_out, trans_all[[0, 2, 4]] * 80.0)


def test_load_poses_applies_index_filter_when_lengths_match_for_duplicate_permuted_indices(monkeypatch):
    rots_all = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 6, axis=0)
    rots_all[:, 0, 0] = np.arange(1, 7, dtype=np.float32)
    trans_all = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
            [0.1, 0.0],
        ],
        dtype=np.float32,
    )
    requested = np.array([5, 0, 5, 1, 2, 3], dtype=np.int32)  # len == Nimg, with duplicates/permutation
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots_all, trans_all))

    rots_out, trans_out, _ = load_utils.load_poses("poses.pkl", Nimg=6, D=20, ind=requested)

    np.testing.assert_allclose(rots_out[:, 0, 0], rots_all[requested, 0, 0])
    np.testing.assert_allclose(trans_out, trans_all[requested] * 20.0)


def test_load_poses_index_filter_rejects_bad_masks_or_indices(monkeypatch):
    rots_all = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 5, axis=0)
    trans_all = np.linspace(0.0, 0.9, 10, dtype=np.float32).reshape(5, 2)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots_all, trans_all))

    with pytest.raises(ValueError, match="boolean mask must be 1D"):
        load_utils.load_poses("poses.pkl", Nimg=3, D=64, ind=np.array([[True, False, True, False, True]], dtype=bool))
    with pytest.raises(ValueError, match="boolean mask length.*must match total size"):
        load_utils.load_poses("poses.pkl", Nimg=3, D=64, ind=np.array([True, False], dtype=bool))
    with pytest.raises(IndexError, match="negative"):
        load_utils.load_poses("poses.pkl", Nimg=3, D=64, ind=np.array([0, -1, 2], dtype=np.int32))
    with pytest.raises(IndexError, match="out-of-range"):
        load_utils.load_poses("poses.pkl", Nimg=3, D=64, ind=np.array([0, 2, 9], dtype=np.int32))


def test_load_poses_rejects_old_pixel_translation_format(monkeypatch):
    rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0)
    trans_pixels = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots, trans_pixels))
    with pytest.raises(ValueError, match="fractional units"):
        load_utils.load_poses("poses.pkl", Nimg=2, D=64)


def test_load_poses_rejects_negative_values_with_abs_gt_one(monkeypatch):
    rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0)
    trans_bad = np.array([[-1.2, 0.1], [0.2, -1.1]], dtype=np.float32)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots, trans_bad))
    with pytest.raises(ValueError, match=r"\|value\| <= 1"):
        load_utils.load_poses("poses.pkl", Nimg=2, D=64)


def test_load_poses_rejects_nonnumeric_rotation_or_translation(monkeypatch):
    rots_bad = np.array([[["a"] * 3] * 3] * 2, dtype=object)
    trans_ok = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots_bad, trans_ok))
    with pytest.raises(ValueError, match="Rotation array must be numeric"):
        load_utils.load_poses("poses.pkl", Nimg=2, D=64)

    rots_ok = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0)
    trans_bad = np.array([["x", "y"], ["u", "v"]], dtype=object)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots_ok, trans_bad))
    with pytest.raises(ValueError, match="Translation array must be numeric"):
        load_utils.load_poses("poses.pkl", Nimg=2, D=64)


def test_load_poses_rejects_nonfinite_rotation_or_translation(monkeypatch):
    rots_bad = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0)
    rots_bad[1, 0, 0] = np.nan
    trans_ok = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots_bad, trans_ok))
    with pytest.raises(ValueError, match="Rotation array contains non-finite"):
        load_utils.load_poses("poses.pkl", Nimg=2, D=64)

    rots_ok = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0)
    trans_bad = np.array([[0.1, 0.2], [np.inf, 0.4]], dtype=np.float32)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots_ok, trans_bad))
    with pytest.raises(ValueError, match="Translation array contains non-finite"):
        load_utils.load_poses("poses.pkl", Nimg=2, D=64)


def test_load_poses_rejects_bad_shapes(monkeypatch):
    bad_rots = np.zeros((2, 2, 2), dtype=np.float32)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: bad_rots)
    with pytest.raises(ValueError, match="Rotation array has shape"):
        load_utils.load_poses("poses.pkl", Nimg=2, D=64)


def test_load_poses_rejects_invalid_number_of_input_files():
    with pytest.raises(ValueError, match="Expected 1 or 2 input files"):
        load_utils.load_poses(["a.pkl", "b.pkl", "c.pkl"], Nimg=2, D=64)


def test_load_poses_rotations_only_returns_none_translation(monkeypatch):
    rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 3, axis=0)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: rots)
    rots_out, trans_out, D_out = load_utils.load_poses("poses.pkl", Nimg=3, D=32)
    assert D_out == 32
    np.testing.assert_array_equal(rots_out, rots)
    assert trans_out is None


def test_load_ctf_params_from_tiny_file_roundtrip(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=5, n_particles=2)
    out = load_utils.load_ctf_params(D=8, ctf_params_pkl=files["ctf_pkl"])
    assert out.shape == (5, 8)
    # Column 0 in return is Apix and should stay 1.5 when D is unchanged.
    assert np.allclose(out[:, 0], 1.5)
    assert np.allclose(out[:, 4], 300.0)  # voltage


def test_load_poses_from_tiny_file_roundtrip(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    rots, trans, D_out = load_utils.load_poses(files["poses_pkl"], Nimg=6, D=8)
    assert D_out == 8
    assert rots.shape == (6, 3, 3)
    assert trans.shape == (6, 2)
    # tiny helper writes zero fractional shifts; conversion to pixels keeps zeros.
    np.testing.assert_array_equal(trans, np.zeros((6, 2), dtype=np.float32))


def test_load_poses_two_file_real_pickles(tmp_path):
    rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 4, axis=0)
    trans_frac = np.array([[0.0, 0.0], [0.25, 0.5], [0.5, 0.25], [1.0, 1.0]], dtype=np.float32)
    rots_pkl = tmp_path / "rots.pkl"
    trans_pkl = tmp_path / "trans.pkl"
    utils.pickle_dump(rots, str(rots_pkl))
    utils.pickle_dump(trans_frac, str(trans_pkl))
    rots_out, trans_out, D_out = load_utils.load_poses([str(rots_pkl), str(trans_pkl)], Nimg=4, D=10)
    assert D_out == 10
    np.testing.assert_array_equal(rots_out, rots)
    np.testing.assert_allclose(trans_out, trans_frac * 10.0)


def test_load_poses_rejects_rotation_translation_count_mismatch(monkeypatch):
    rots = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 3, axis=0)
    trans_frac = np.zeros((4, 2), dtype=np.float32)
    monkeypatch.setattr(load_utils.utils, "pickle_load", lambda _: (rots, trans_frac))

    with pytest.raises(ValueError, match="count mismatch"):
        load_utils.load_poses("poses.pkl", Nimg=3, D=8)


def test_load_poses_from_tiny_tilt_simulator_subset_preserves_duplicates_and_alignment(tmp_path):
    files = tiny_synthetic.make_tiny_tilt_loader_files_from_simulator(
        tmp_path / "tiny_tilt_pose_subset",
        grid_size=8,
        n_images=12,
        n_tilts=3,
        n_volumes=3,
    )
    requested = np.array([5, 1, 5, 9], dtype=np.int32)

    rots_all, trans_all, _ = load_utils.load_poses(files["poses_pkl"], Nimg=12, D=8)
    rots_sub, trans_sub, D_out = load_utils.load_poses(
        files["poses_pkl"],
        Nimg=len(requested),
        D=8,
        ind=requested,
    )

    assert D_out == 8
    np.testing.assert_allclose(rots_sub, rots_all[requested])
    np.testing.assert_allclose(trans_sub, trans_all[requested])


def test_load_poses_from_tiny_tilt_simulator_boolean_mask_subset_matches_integer_selection(tmp_path):
    files = tiny_synthetic.make_tiny_tilt_loader_files_from_simulator(
        tmp_path / "tiny_tilt_pose_mask",
        grid_size=8,
        n_images=10,
        n_tilts=2,
        n_volumes=3,
    )
    mask = np.array([True, False, True, False, False, True, False, True, False, False], dtype=bool)
    requested = np.flatnonzero(mask)

    rots_mask, trans_mask, _ = load_utils.load_poses(
        files["poses_pkl"],
        Nimg=int(mask.sum()),
        D=8,
        ind=mask,
    )
    rots_idx, trans_idx, _ = load_utils.load_poses(
        files["poses_pkl"],
        Nimg=len(requested),
        D=8,
        ind=requested,
    )

    np.testing.assert_allclose(rots_mask, rots_idx)
    np.testing.assert_allclose(trans_mask, trans_idx)
