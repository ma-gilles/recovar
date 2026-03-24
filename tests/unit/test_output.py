import os

import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.output import output
from recovar.output import plot_utils

pytestmark = pytest.mark.unit


def test_get_resampled_distances_and_resample_trajectory():
    vols = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    d = output.get_resampled_distances(vols)
    idx = output.resample_trajectory(vols, n_vols_along_path=3)
    assert np.all(d[1:] >= d[:-1])
    assert idx.shape == (3,)
    assert idx[0] == 0
    assert idx[-1] == 4


def test_sum_over_other_and_slice_helpers():
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    summed = output.sum_over_other(x, use_axis=[0, 2])
    assert summed.shape == (2, 4)
    assert np.allclose(summed, x.sum(axis=1))

    density = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    half = output.half_slice_other(density, axes=[0, 2])
    assert half.shape == (5, 7)
    assert np.allclose(half, density[:, density.shape[1] // 2, :])

    pt = [1, 2, 3]
    sl = output.slice_at_point(density, axes=[0, 2], point=pt)
    assert sl.shape == (5, 7)
    assert np.allclose(sl, density[:, pt[1], :])


def test_save_covar_output_volumes_clamps_to_available_pcs(monkeypatch, tmp_path):
    # Regression test for IndexError when available PCs < default us_to_save.
    d = 8
    vol_size = d**3
    n_pcs = 30

    mean = np.zeros(vol_size, dtype=np.complex64)
    u = np.random.randn(vol_size, n_pcs).astype(np.float32)
    s = {"rescaled": np.abs(np.random.randn(n_pcs)).astype(np.float32) + 1e-3}
    volume_mask = np.ones((d, d, d), dtype=np.float32)

    calls = {"n_eigs": []}

    monkeypatch.setattr(output, "save_volumes", lambda vols, *args, **kwargs: calls.setdefault("saved_vols", len(vols)))
    monkeypatch.setattr(output, "save_volume", lambda *args, **kwargs: None)
    monkeypatch.setattr(output.linalg, "batch_idft3", lambda arr, volume_shape, vol_batch_size: np.asarray(arr, dtype=np.float32))
    monkeypatch.setattr(output.utils, "estimate_variance", lambda u_t, svals: np.ones(u_t.shape[-1], dtype=np.float32))

    output.save_covar_output_volumes(
        str(tmp_path) + "/",
        mean=mean,
        u=u,
        s=s,
        mask=volume_mask,
        volume_shape=(d, d, d),
        us_to_save=50,          # intentionally larger than available
        us_to_var=[4, 10, 20],  # all <= n_pcs
        voxel_size=1.0,
    )

    # Should save exactly n_pcs eigenvectors, not fail on index n_pcs.
    assert calls["saved_vols"] == n_pcs


def test_save_covar_output_volumes_skips_variance_when_no_pcs(monkeypatch, tmp_path):
    d = 8
    vol_size = d**3

    mean = np.zeros(vol_size, dtype=np.complex64)
    u = np.zeros((vol_size, 0), dtype=np.float32)  # no PCs available
    s = {"rescaled": np.zeros((0,), dtype=np.float32)}
    volume_mask = np.ones((d, d, d), dtype=np.float32)

    calls = {"save_volumes": 0, "save_volume": 0}

    monkeypatch.setattr(
        output,
        "save_volumes",
        lambda vols, *args, **kwargs: calls.__setitem__("save_volumes", calls["save_volumes"] + 1),
    )
    monkeypatch.setattr(
        output,
        "save_volume",
        lambda *args, **kwargs: calls.__setitem__("save_volume", calls["save_volume"] + 1),
    )
    monkeypatch.setattr(
        output.linalg,
        "batch_idft3",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("batch_idft3 should not run for 0 PCs")),
    )

    output.save_covar_output_volumes(
        str(tmp_path) + "/",
        mean=mean,
        u=u,
        s=s,
        mask=volume_mask,
        volume_shape=(d, d, d),
        us_to_save=50,
        us_to_var=[4, 10, 20],
        voxel_size=1.0,
    )

    # Called once for eigen_pos (empty list is fine) and once for mean volume.
    assert calls["save_volumes"] == 1
    assert calls["save_volume"] == 1


def test_save_covar_output_volumes_clamps_variance_rank_to_available_svals(monkeypatch, tmp_path):
    d = 8
    vol_size = d**3
    n_pcs = 8
    n_svals = 3

    mean = np.zeros(vol_size, dtype=np.complex64)
    u = np.random.randn(vol_size, n_pcs).astype(np.float32)
    s = {"rescaled": np.abs(np.random.randn(n_svals)).astype(np.float32) + 1e-3}
    volume_mask = np.ones((d, d, d), dtype=np.float32)

    calls = {"ranks": []}

    monkeypatch.setattr(output, "save_volumes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(output, "save_volume", lambda *_args, **_kwargs: None)

    def _fake_batch_idft3(arr, volume_shape, vol_batch_size):
        _ = volume_shape, vol_batch_size
        calls["ranks"].append(int(arr.shape[-1]))
        return np.asarray(arr, dtype=np.float32)

    def _fake_estimate_variance(u_t, svals):
        assert int(u_t.shape[0]) == int(len(svals))
        return np.ones(u_t.shape[-1], dtype=np.float32)

    monkeypatch.setattr(output.linalg, "batch_idft3", _fake_batch_idft3)
    monkeypatch.setattr(output.utils, "estimate_variance", _fake_estimate_variance)

    output.save_covar_output_volumes(
        str(tmp_path) + "/",
        mean=mean,
        u=u,
        s=s,
        mask=volume_mask,
        volume_shape=(d, d, d),
        us_to_save=5,
        us_to_var=[2, 5, 7],  # requests above n_svals must clamp to n_svals
        voxel_size=1.0,
    )

    assert calls["ranks"] == [2, n_svals, n_svals]


def test_save_covar_output_volumes_negative_us_to_save_yields_no_eigenvectors(monkeypatch, tmp_path):
    d = 8
    vol_size = d**3
    n_pcs = 4

    mean = np.zeros(vol_size, dtype=np.complex64)
    u = np.random.randn(vol_size, n_pcs).astype(np.float32)
    s = {"rescaled": np.abs(np.random.randn(n_pcs)).astype(np.float32) + 1e-3}
    volume_mask = np.ones((d, d, d), dtype=np.float32)

    calls = {"save_volumes_sizes": []}

    monkeypatch.setattr(
        output,
        "save_volumes",
        lambda vols, *_args, **_kwargs: calls["save_volumes_sizes"].append(len(vols)),
    )
    monkeypatch.setattr(output, "save_volume", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(output.linalg, "batch_idft3", lambda arr, *_args, **_kwargs: np.asarray(arr, dtype=np.float32))
    monkeypatch.setattr(output.utils, "estimate_variance", lambda u_t, svals: np.ones(u_t.shape[-1], dtype=np.float32))

    output.save_covar_output_volumes(
        str(tmp_path) + "/",
        mean=mean,
        u=u,
        s=s,
        mask=volume_mask,
        volume_shape=(d, d, d),
        us_to_save=-5,
        us_to_var=[],
        voxel_size=1.0,
    )

    # First call corresponds to eigenvectors export.
    assert calls["save_volumes_sizes"][0] == 0


def test_save_covar_output_volumes_large_grid_clamps_batch_size_to_one(monkeypatch, tmp_path):
    mean = np.zeros(8, dtype=np.complex64)
    u = np.random.randn(8, 2).astype(np.float32)
    s = {"rescaled": np.array([1.0, 0.5], dtype=np.float32)}
    volume_mask = np.ones((2, 2, 2), dtype=np.float32)

    seen = {"vol_batch_size": []}

    monkeypatch.setattr(output, "save_volumes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(output, "save_volume", lambda *_args, **_kwargs: None)

    def _fake_batch_idft3(arr, volume_shape, vol_batch_size):
        _ = arr, volume_shape
        seen["vol_batch_size"].append(int(vol_batch_size))
        return np.ones((2, int(arr.shape[-1])), dtype=np.float32)

    monkeypatch.setattr(output.linalg, "batch_idft3", _fake_batch_idft3)
    monkeypatch.setattr(output.utils, "estimate_variance", lambda u_t, svals: np.ones(u_t.shape[-1], dtype=np.float32))

    output.save_covar_output_volumes(
        str(tmp_path) + "/",
        mean=mean,
        u=u,
        s=s,
        mask=volume_mask,
        # Very large grid would previously make int((2**24)/(grid_size**3)) == 0.
        volume_shape=(1024, 1024, 1024),
        us_to_save=1,
        us_to_var=[1],
        voxel_size=1.0,
    )

    assert seen["vol_batch_size"] == [1]


def test_get_nearest_point_finds_closest():
    data = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    query = np.array([[0.1, 0.0], [1.9, 0.1]], dtype=np.float32)
    points, indices = output.get_nearest_point(data, query)
    assert indices[0] == 0
    assert indices[1] == 2
    np.testing.assert_allclose(points[0], data[0])
    np.testing.assert_allclose(points[1], data[2])


def test_get_nearest_point_single_data_point():
    data = np.array([[3.0, 4.0]], dtype=np.float32)
    query = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    points, indices = output.get_nearest_point(data, query)
    assert list(indices) == [0, 0]
    np.testing.assert_allclose(points, np.tile(data, (2, 1)))


def test_get_nearest_point_chunked_matches_dense_path():
    data = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    query = np.array([[0.1, 0.0], [2.9, 0.0], [1.2, 0.0]], dtype=np.float32)
    dense_points, dense_indices = output.get_nearest_point(data, query)
    chunked_points, chunked_indices = output.get_nearest_point(data, query, chunk_size=1)
    np.testing.assert_array_equal(chunked_indices, dense_indices)
    np.testing.assert_allclose(chunked_points, dense_points)


def test_standard_pipeline_plots_uses_embedding_component_api(monkeypatch, tmp_path):
    calls = []

    class _PO:
        def get(self, key):
            if key in {"latent_coords", "contrasts"}:
                raise AssertionError(f"standard_pipeline_plots should not call get('{key}')")
            if key == "s":
                return np.ones(4, dtype=np.float32)
            raise KeyError(key)

        def get_embedding_keys(self, entry):
            assert entry == "latent_coords"
            return [4]

        def get_embedding_component(self, entry, key):
            calls.append((entry, key))
            if entry == "contrasts":
                return np.ones(12, dtype=np.float32)
            if entry == "latent_coords":
                return np.arange(48, dtype=np.float32).reshape(12, 4)
            raise KeyError(entry)

    monkeypatch.setattr(output, "mkdir_safe", lambda path: tmp_path.mkdir(exist_ok=True))
    monkeypatch.setattr(plot_utils, "plot_summary_t", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plot_utils, "plot_contrast_histogram", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plot_utils, "plot_eigenvalues", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plot_utils, "plot_mean_fsc", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plot_utils, "plot_pipeline_summary", lambda *_args, **_kwargs: None)

    output.standard_pipeline_plots(_PO(), 4, str(tmp_path))

    assert ("contrasts", 4) in calls
    assert ("latent_coords", 4) in calls


def test_volume_output_paths_places_primary_and_diag_correctly(tmp_path):
    """VolumeOutputPaths should put primary outputs at root and diagnostics in subdir."""
    from recovar.output.output_paths import VolumeOutputPaths

    vp = VolumeOutputPaths(str(tmp_path), "state", 0)
    vp.ensure_dirs()

    # Primary outputs should be in tmp_path directly
    assert vp.filtered == str(tmp_path / "state000.mrc")
    assert vp.half1_unfil == str(tmp_path / "state000_half1_unfil.mrc")
    assert vp.half2_unfil == str(tmp_path / "state000_half2_unfil.mrc")

    # Diagnostics should be in diagnostics/state000/
    assert "diagnostics" in vp.params
    assert "diagnostics" in vp.split_choice
    assert os.path.isdir(vp.diag_dir)


def test_make_trajectory_plots_from_results_uses_pipeline_output_helpers(monkeypatch, tmp_path):
    captured = {}
    fake_dataset = object()
    zs = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    cov_zs = np.zeros((2, 2, 2), dtype=np.float32)
    density = np.ones((8, 8), dtype=np.float32)
    bounds = np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=np.float32)
    contrasts = np.array([0.2, 0.8], dtype=np.float32)

    class FakePO:
        def get_embedding_component(self, entry, key):
            captured.setdefault("embedding_calls", []).append((entry, key))
            values = {
                ("latent_coords", 2): zs,
                ("latent_precision", 2): cov_zs,
                ("contrasts", 2): contrasts,
            }
            return values[(entry, key)]

        def get(self, key):
            captured.setdefault("get_calls", []).append(key)
            values = {
                "dataset": fake_dataset,
                "density": density,
            }
            return values[key]

    monkeypatch.setattr(output.ld, "compute_latent_space_bounds", lambda _zs: bounds)
    monkeypatch.setattr(
        output.embedding,
        "set_contrasts_in_cryos",
        lambda ds, vals: captured.setdefault("contrast_call", (ds, np.array(vals))),
    )

    def fake_make_trajectory_plots(density_arg, zs_arg, cov_arg, *args, **kwargs):
        captured["make_args"] = (density_arg, zs_arg, cov_arg, args, kwargs)
        return "full_path", "subsampled_path"

    monkeypatch.setattr(output, "make_trajectory_plots", fake_make_trajectory_plots)

    result = output.make_trajectory_plots_from_results(
        pipeline_output=FakePO(),
        basis_size=2,
        output_folder=str(tmp_path) + "/",
        z_st=np.array([0.0, 0.0], dtype=np.float32),
        z_end=np.array([1.0, 1.0], dtype=np.float32),
        plot_llh=True,
    )

    assert result == ("full_path", "subsampled_path")
    assert captured["contrast_call"][0] is fake_dataset
    np.testing.assert_array_equal(captured["contrast_call"][1], contrasts)
    np.testing.assert_array_equal(captured["make_args"][0], density)
    np.testing.assert_array_equal(captured["make_args"][1], zs)
    np.testing.assert_array_equal(captured["make_args"][2], cov_zs)


def test_cluster_kmeans_shapes_and_label_count():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((60, 2)).astype(np.float32)
    labels, centers = output.cluster_kmeans(z, K=3)
    assert labels.shape == (60,)
    assert centers.shape == (3, 2)
    assert set(labels) == {0, 1, 2}


def test_cluster_kmeans_reorder_false_skips_sort():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((40, 2)).astype(np.float32)
    labels, centers = output.cluster_kmeans(z, K=2, reorder=False)
    assert labels.shape == (40,)
    assert centers.shape == (2, 2)


def test_scatter_annotate_returns_figure_axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    # annotate=False so no centers are needed
    fig, ax = output.scatter_annotate(x, y, annotate=False)
    assert hasattr(fig, "savefig")
    assert hasattr(ax, "scatter")
    plt.close(fig)


def test_scatter_annotate_with_centers_ind():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    fig, ax = output.scatter_annotate(x, y, centers_ind=np.array([0, 2]), annotate=True)
    assert hasattr(fig, "savefig")
    plt.close(fig)


def test_scatter_annotate_with_explicit_centers():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    centers = np.array([[0.25, 0.25], [0.75, 0.75]])
    fig, ax = output.scatter_annotate(x, y, centers=centers, annotate=False)
    assert hasattr(fig, "savefig")
    plt.close(fig)
