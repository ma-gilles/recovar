import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import output

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
