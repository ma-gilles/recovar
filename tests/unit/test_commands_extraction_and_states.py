import argparse
import os
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.commands import (
    estimate_stable_states,
    extract_image_subset,
    extract_image_subset_from_kmeans,
)

pytestmark = pytest.mark.unit


def test_center_of_mass_3d_weighted():
    arr = np.zeros((3, 3, 3), dtype=np.float32)
    arr[0, 0, 0] = 1.0
    arr[2, 2, 2] = 3.0
    com = extract_image_subset.center_of_mass(arr)
    assert np.allclose(com, (1.5, 1.5, 1.5))


def test_nearest_point_index_returns_expected_index():
    point = np.array([2.0, 2.0, 2.0])
    pts = np.array([[0, 0, 0], [2.1, 2.0, 2.0], [5, 5, 5]], dtype=np.float32)
    idx = extract_image_subset.nearest_point_index(point, pts)
    assert idx == 1


def test_decide_subvolume_index_from_mask_uses_center_of_mass():
    mask = np.zeros((5, 5, 5), dtype=np.float32)
    mask[1, 1, 1] = 1.0
    mask[3, 3, 3] = 1.0
    sampling_points = np.array([[1, 1, 1], [2, 2, 2], [4, 4, 4]], dtype=np.float32)
    idx = extract_image_subset.decide_subvolume_index_from_mask(mask, sampling_points)
    assert idx == 1


def test_extract_image_subset_uses_coordinate_to_pick_subvolume(monkeypatch, tmp_path):
    captured = {}

    def fake_pickle_load(path):
        assert path.endswith("/params.pkl")
        return {"locres_sampling": 3, "locres_maskrad": 1.0, "voxel_size": 1.5}

    def fake_load_mrc(path):
        if path.endswith("locres.mrc"):
            return np.zeros((16, 16, 16), dtype=np.float32)
        raise AssertionError("unexpected path")

    def fake_get_sampling_points(grid_size, sampling, maskrad, voxel_size):
        assert (grid_size, sampling, maskrad, voxel_size) == (16, 3, 1.0, 1.5)
        return np.array([[0, 0, 0], [3, 3, 3], [7, 7, 7]], dtype=np.float32)

    def fake_get_inds_for_subvolume(input_dir, subvolume_idx):
        captured["subvolume_idx"] = subvolume_idx
        assert input_dir == str(tmp_path / "in")
        return np.array([1, 3, 5], dtype=np.int32)

    def fake_pickle_dump(indices, out_path):
        captured["indices"] = np.array(indices)
        captured["out_path"] = out_path

    monkeypatch.setattr(extract_image_subset.utils, "pickle_load", fake_pickle_load)
    monkeypatch.setattr(extract_image_subset.utils, "load_mrc", fake_load_mrc)
    monkeypatch.setattr(extract_image_subset.locres, "get_sampling_points", fake_get_sampling_points)
    monkeypatch.setattr(extract_image_subset.heterogeneity_volume, "get_inds_for_subvolume", fake_get_inds_for_subvolume)
    monkeypatch.setattr(extract_image_subset.utils, "pickle_dump", fake_pickle_dump)

    extract_image_subset.extract_image_subset(
        input_dir=str(tmp_path / "in"),
        output_path=str(tmp_path / "subset.pkl"),
        subvolume_idx=None,
        mask=None,
        coordinate=[10, 10, 10],
    )

    assert captured["subvolume_idx"] == 1
    assert np.array_equal(captured["indices"], np.array([1, 3, 5], dtype=np.int32))
    assert captured["out_path"] == str(tmp_path / "subset.pkl")


def test_extract_image_subset_uses_mask_center_of_mass(monkeypatch, tmp_path):
    captured = {}
    mask_arr = np.zeros((6, 6, 6), dtype=np.float32)
    mask_arr[4, 2, 2] = 1.0

    monkeypatch.setattr(extract_image_subset.utils, "load_mrc", lambda path: mask_arr if path.endswith("mask.mrc") else np.zeros((16, 16, 16), dtype=np.float32))
    monkeypatch.setattr(extract_image_subset.utils, "pickle_load", lambda _p: {"locres_sampling": 2, "locres_maskrad": 0.5, "voxel_size": 1.0})
    monkeypatch.setattr(extract_image_subset.locres, "get_sampling_points", lambda *_args, **_kwargs: np.array([[0, 0, 0], [4, 2, 2]], dtype=np.float32) - 8)
    monkeypatch.setattr(extract_image_subset.heterogeneity_volume, "get_inds_for_subvolume", lambda _d, idx: np.array([idx], dtype=np.int32))
    monkeypatch.setattr(extract_image_subset.utils, "pickle_dump", lambda inds, _p: captured.setdefault("inds", inds))

    extract_image_subset.extract_image_subset(
        input_dir=str(tmp_path / "in"),
        output_path=str(tmp_path / "subset.pkl"),
        subvolume_idx=None,
        mask=str(tmp_path / "mask.mrc"),
        coordinate=None,
    )

    assert np.array_equal(captured["inds"], np.array([1], dtype=np.int32))


def test_extract_image_subset_main_validates_input(monkeypatch):
    args = SimpleNamespace(input_dir="/tmp/in", output="/tmp/out.pkl", subvol_idx=None, mask=None, coordinate=None)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)

    with pytest.raises(ValueError, match="either a subvolume index"):
        extract_image_subset.main()


def test_extract_image_subset_from_kmeans_basic_and_inverse(tmp_path):
    centers_pkl = tmp_path / "centers.pkl"
    out_pkl = tmp_path / "subset.pkl"
    labels = np.array([0, 1, np.nan, 2, 1, np.nan], dtype=np.float32)

    extract_image_subset_from_kmeans.utils.pickle_dump({"labels": labels}, str(centers_pkl))

    extract_image_subset_from_kmeans.extract_image_subset_from_kmeans(str(centers_pkl), [1], False, str(out_pkl))
    kept = extract_image_subset_from_kmeans.utils.pickle_load(str(out_pkl))
    assert np.array_equal(kept, np.array([1, 4]))

    extract_image_subset_from_kmeans.extract_image_subset_from_kmeans(str(centers_pkl), [1], True, str(out_pkl))
    kept_inv = extract_image_subset_from_kmeans.utils.pickle_load(str(out_pkl))
    assert np.array_equal(kept_inv, np.array([0, 3]))


def test_estimate_stable_states_writes_outputs(monkeypatch, tmp_path):
    latent_pts_z = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    latent_pts_grid = np.array([[5, 6], [7, 8]], dtype=np.float32)
    called = {}

    def fake_find_local_maxs_of_density(density, latent_space_bounds, percent_top, n_local_maxs, plot_folder):
        called["args"] = (density.shape, latent_space_bounds, percent_top, n_local_maxs, plot_folder)
        return latent_pts_z, latent_pts_grid

    monkeypatch.setattr(
        estimate_stable_states.deconvolve_density,
        "find_local_maxs_of_density",
        fake_find_local_maxs_of_density,
    )
    monkeypatch.setattr(
        estimate_stable_states.output,
        "plot_over_density",
        lambda density, points, annotate, plot_folder, cmap: called.setdefault("plot", (density.shape, points.shape, annotate, plot_folder, cmap)),
    )

    density = np.zeros((8, 8), dtype=np.float32)
    estimate_stable_states.estimate_stable_states(density, {"x": [-1, 1]}, percent_top=5, n_local_maxs=2, file_path=str(tmp_path))

    assert called["args"][2] == 5
    assert called["args"][3] == 2
    assert os.path.exists(tmp_path / "stable_state_all_coords.txt")
    assert os.path.exists(tmp_path / "stable_state_0_coords.txt")
    assert os.path.exists(tmp_path / "stable_state_1_coords.txt")


def test_estimate_stable_states_main_loads_pkl_and_dispatches(monkeypatch, tmp_path):
    density_pkl = tmp_path / "density.pkl"
    extract_image_subset_from_kmeans.utils.pickle_dump(
        {"density": np.ones((4, 4), dtype=np.float32), "latent_space_bounds": {"x": [0, 1]}},
        str(density_pkl),
    )

    args = SimpleNamespace(
        density=str(density_pkl),
        file_path=str(tmp_path / "out"),
        percent_top=3.5,
        n_local_maxs=7,
    )

    called = {}
    monkeypatch.setattr(estimate_stable_states, "parse_args", lambda: args)
    monkeypatch.setattr(
        estimate_stable_states,
        "estimate_stable_states",
        lambda density, latent_space_bounds, percent_top, n_local_maxs, file_path: called.setdefault(
            "args", (density.shape, latent_space_bounds, percent_top, n_local_maxs, file_path)
        ),
    )

    estimate_stable_states.main()
    assert called["args"][0] == (4, 4)
    assert called["args"][2:] == (3.5, 7, str(tmp_path / "out"))
