import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.output import output

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


