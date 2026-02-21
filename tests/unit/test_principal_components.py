import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.principal_components as pc
import recovar.core as core
from recovar import dataset
from helpers.tiny_synthetic import make_tiny_simulation

pytestmark = pytest.mark.unit


def test_zero_boundary_mask_has_expected_structure():
    m = pc.get_zero_boundary_mask((4, 4, 4), dtype=np.float32).reshape(4, 4, 4)
    assert np.all(m[0, :, :] == 0)
    assert np.all(m[:, 0, :] == 0)
    assert np.all(m[:, :, 0] == 0)
    assert np.all(m[1:, 1:, 1:] == 1)


def test_minus_index_negates_frequencies():
    volume_shape = (4, 4, 4)
    idx = np.arange(np.prod(volume_shape))
    minus_idx = np.asarray(pc.get_minus_vec_index(idx, volume_shape))
    freqs = np.asarray(core.vec_indices_to_frequencies(idx, volume_shape))
    minus_freqs = np.asarray(core.vec_indices_to_frequencies(minus_idx, volume_shape))
    # Exclude Nyquist boundary where mapping is not strictly invertible on the wrapped grid.
    interior = np.all(np.abs(freqs) < (volume_shape[0] // 2), axis=1)
    np.testing.assert_array_equal(minus_freqs[interior], -freqs[interior])


def test_make_symmetric_columns_matches_numpy_version():
    volume_shape = (4, 4, 4)
    vol_size = int(np.prod(volume_shape))
    n_cols = 5
    rng = np.random.default_rng(0)

    columns = (rng.normal(size=(vol_size, n_cols)) + 1j * rng.normal(size=(vol_size, n_cols))).astype(np.complex64)
    picked = rng.choice(vol_size, size=n_cols, replace=False)

    cols_a, minus_a, good_a = pc.make_symmetric_columns(columns, picked, volume_shape)
    cols_b, minus_b, good_b = pc.make_symmetric_columns_np(columns, picked, volume_shape)

    np.testing.assert_allclose(np.asarray(cols_a), np.asarray(cols_b), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(minus_a), np.asarray(minus_b))
    np.testing.assert_array_equal(np.asarray(good_a), np.asarray(good_b))


def test_flip_vec_and_batch_flip_vec2_shapes():
    volume_shape = (4, 4, 4)
    vol_size = int(np.prod(volume_shape))
    col = np.arange(vol_size, dtype=np.float32).astype(np.complex64)
    flipped = np.asarray(pc.flip_vec(col, volume_shape))
    assert flipped.shape == (vol_size,)

    cols = np.stack([col, col * 2], axis=1).astype(np.complex64)
    out = pc.batch_flip_vec2(cols, volume_shape)
    assert out.shape == (2, vol_size)


def test_idft_from_both_sides_calls_batch_idft3(monkeypatch):
    calls = {"n": 0}

    def _fake_batch_idft3(x, shape, batch_size):
        calls["n"] += 1
        return x

    monkeypatch.setattr(pc.linalg, "batch_idft3", _fake_batch_idft3)
    x = np.ones((4, 3), dtype=np.complex64)
    out = pc.IDFT_from_both_sides(x, (2, 2, 1), (2, 2, 1), 2, 2)
    assert out.shape == x.shape
    assert calls["n"] == 2


def test_get_all_copied_columns_shapes():
    volume_shape = (4, 4, 4)
    vol_size = int(np.prod(volume_shape))
    rng = np.random.default_rng(0)
    cols = (rng.normal(size=(vol_size, 4)) + 1j * rng.normal(size=(vol_size, 4))).astype(np.complex64)
    picked = rng.choice(vol_size, size=4, replace=False)

    all_cols, all_freqs = pc.get_all_copied_columns(cols, picked, volume_shape)
    assert all_cols.shape[0] == vol_size
    assert all_cols.shape[1] >= cols.shape[1]
    assert all_freqs.shape[0] == all_cols.shape[1]


def test_get_cov_svds_delegates_to_randomized_svd(monkeypatch):
    called = {}

    def fake_randomized_real_svd_of_columns(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        u = np.eye(3, dtype=np.float32)
        s = np.array([3.0, 2.0, 1.0], dtype=np.float32)
        vh = np.eye(3, dtype=np.float32)
        return u, s, vh

    monkeypatch.setattr(pc, "randomized_real_svd_of_columns", fake_randomized_real_svd_of_columns)

    cov_cols = {"est_mask": np.ones((3, 2), dtype=np.complex64)}
    u, s = pc.get_cov_svds(
        covariance_cols=cov_cols,
        picked_frequencies=np.array([0, 1]),
        volume_mask=np.ones(3, dtype=np.float32),
        volume_shape=(1, 1, 3),
        vol_batch_size=2,
        gpu_memory_to_use=8,
        ignore_zero_frequency=False,
        randomized_sketch_size=5,
    )
    assert "real" in u and "real" in s
    np.testing.assert_array_equal(u["real"], np.eye(3, dtype=np.float32))
    np.testing.assert_array_equal(s["real"], np.array([3.0, 2.0, 1.0], dtype=np.float32))
    assert called["kwargs"]["test_size"] == 5


def test_pca_by_projected_covariance_sorts_and_clamps_eigs(monkeypatch):
    cryos = [type("Cryo", (), {"image_size": 16})()]
    basis = np.eye(3, dtype=np.float32)
    mean = np.zeros(3, dtype=np.complex64)
    volume_mask = np.ones(3, dtype=np.float32)

    monkeypatch.setattr(pc.utils, "get_gpu_memory_total", lambda: 100.0)
    monkeypatch.setattr(pc.utils, "get_embedding_batch_size", lambda *args, **kwargs: 2)

    def fake_projected_covariance(*args, **kwargs):
        # Eigenvalues are [3, -1, 2] => sorted desc [3,2,-1], then clamp last to EPSILON.
        return np.diag(np.array([3.0, -1.0, 2.0], dtype=np.float64))

    monkeypatch.setattr(pc.covariance_estimation, "compute_projected_covariance", fake_projected_covariance)

    u, s = pc.pca_by_projected_covariance(
        cryos=cryos,
        basis=basis,
        mean=mean,
        volume_mask=volume_mask,
        disc_type="linear_interp",
        disc_type_u="linear_interp",
        gpu_memory_to_use=8,
        n_pcs_to_compute=3,
    )
    assert u.shape == (3, 3)
    assert s.shape == (3,)
    assert s[0] >= s[1] >= s[2]
    assert s[2] == pytest.approx(pc.constants.EPSILON)


def test_estimate_principal_components_high_snr_from_var_est_requires_variance():
    cryos = [
        type(
            "Cryo",
            (),
            {
                "volume_shape": (2, 2, 2),
                "grid_size": 2,
                "volume_size": 8,
                "dtype": np.complex64,
                "image_size": 4,
            },
        )()
    ]
    means = {
        "lhs": np.ones(8, dtype=np.float32),
        "prior": np.ones(8, dtype=np.float32),
        "combined": np.ones(8, dtype=np.complex64),
        "combined_regularized": np.ones(8, dtype=np.complex64),
    }
    options = {"keep_intermediate": True, "contrast": "none", "ignore_zero_frequency": False}
    cov_options = {
        "column_sampling_scheme": "high_snr_from_var_est",
        "sampling_n_cols": 2,
        "sampling_avoid_in_radius": 0,
        "randomize_column_sampling": False,
        "randomized_sketch_size": 2,
        "disc_type": "linear_interp",
        "disc_type_u": "linear_interp",
        "mask_images_in_proj": False,
        "n_pcs_to_compute": 2,
    }

    with pytest.raises(ValueError, match="variance_estimate must be provided"):
        pc.estimate_principal_components(
            cryos=cryos,
            options=options,
            means=means,
            mean_prior=np.zeros(8, dtype=np.complex64),
            volume_mask=np.ones(8, dtype=np.float32),
            dilated_volume_mask=np.ones(8, dtype=np.float32),
            valid_idx=np.arange(2),
            batch_size=1,
            gpu_memory_to_use=8,
            covariance_options=cov_options,
            variance_estimate=None,
        )


def test_estimate_principal_components_low_freqs_pipeline(monkeypatch):
    cryos = [
        type(
            "Cryo",
            (),
            {
                "volume_shape": (2, 2, 2),
                "grid_size": 2,
                "volume_size": 8,
                "dtype": np.complex64,
                "image_size": 4,
            },
        )()
    ]
    means = {
        "lhs": np.ones(8, dtype=np.float32),
        "prior": np.ones(8, dtype=np.float32),
        "combined": np.ones(8, dtype=np.complex64),
        "combined_regularized": np.ones(8, dtype=np.complex64),
    }
    options = {"keep_intermediate": False, "contrast": "none", "ignore_zero_frequency": False}
    cov_options = {
        "column_sampling_scheme": "low_freqs",
        "column_radius": 1,
        "randomized_sketch_size": 2,
        "disc_type": "linear_interp",
        "disc_type_u": "linear_interp",
        "mask_images_in_proj": False,
        "n_pcs_to_compute": 2,
    }

    monkeypatch.setattr(pc.utils, "get_vol_batch_size", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        pc.covariance_estimation,
        "compute_regularized_covariance_columns_in_batch",
        lambda *args, **kwargs: (
            {"est_mask": np.ones((8, 2), dtype=np.complex64)},
            np.array([0, 1]),
            np.array([0.1, 0.2]),
        ),
    )
    monkeypatch.setattr(
        pc,
        "get_cov_svds",
        lambda *args, **kwargs: (
            {"real": np.ones((8, 2), dtype=np.float32)},
            {"real": np.array([2.0, 1.0], dtype=np.float32)},
        ),
    )
    monkeypatch.setattr(
        pc,
        "pca_by_projected_covariance",
        lambda *args, **kwargs: (
            np.ones((8, 2), dtype=np.complex64),
            np.array([2.0, 1.0], dtype=np.float32),
        ),
    )

    u, s, covariance_cols, picked_frequencies, column_fscs = pc.estimate_principal_components(
        cryos=cryos,
        options=options,
        means=means,
        mean_prior=np.zeros(8, dtype=np.complex64),
        volume_mask=np.ones(8, dtype=np.float32),
        dilated_volume_mask=np.ones(8, dtype=np.float32),
        valid_idx=np.arange(2),
        batch_size=1,
        gpu_memory_to_use=8,
        covariance_options=cov_options,
    )

    assert u["rescaled"].shape == (8, 2)
    np.testing.assert_array_equal(s["rescaled"], np.array([2.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(picked_frequencies, np.array([0, 1]))
    np.testing.assert_array_equal(column_fscs, np.array([0.1, 0.2]))
    assert covariance_cols["est_mask"] is None


def test_estimate_principal_components_with_real_tiny_dataset(monkeypatch):
    # Build a real tiny CryoEMDataset from simulator outputs.
    _, ctf_params, rots, trans, _, voxel_size, _ = make_tiny_simulation(grid_size=4, n_images=6, seed=0)
    cryo = dataset.CryoEMDataset(
        image_stack=None,
        voxel_size=voxel_size,
        rotation_matrices=rots,
        translations=trans,
        CTF_params=ctf_params,
        CTF_fun=core.evaluate_ctf_wrapper,
        dataset_indices=None,
        grid_size=4,
    )
    cryos = [cryo]

    means = {
        "lhs": np.ones(cryo.volume_size, dtype=np.float32),
        "prior": np.ones(cryo.volume_size, dtype=np.float32),
        "combined": np.ones(cryo.volume_size, dtype=np.complex64),
        "combined_regularized": np.ones(cryo.volume_size, dtype=np.complex64),
    }
    options = {"keep_intermediate": False, "contrast": "none", "ignore_zero_frequency": False}
    cov_options = {
        "column_sampling_scheme": "low_freqs",
        "column_radius": 1,
        "randomized_sketch_size": 2,
        "disc_type": "linear_interp",
        "disc_type_u": "linear_interp",
        "mask_images_in_proj": False,
        "n_pcs_to_compute": 2,
    }

    monkeypatch.setattr(pc.utils, "get_vol_batch_size", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        pc.covariance_estimation,
        "compute_regularized_covariance_columns_in_batch",
        lambda *args, **kwargs: (
            {"est_mask": np.ones((cryo.volume_size, 2), dtype=np.complex64)},
            np.array([0, 1]),
            np.array([0.1, 0.2]),
        ),
    )
    monkeypatch.setattr(
        pc,
        "get_cov_svds",
        lambda *args, **kwargs: (
            {"real": np.ones((cryo.volume_size, 2), dtype=np.float32)},
            {"real": np.array([2.0, 1.0], dtype=np.float32)},
        ),
    )
    monkeypatch.setattr(
        pc,
        "pca_by_projected_covariance",
        lambda *args, **kwargs: (
            np.ones((cryo.volume_size, 2), dtype=np.complex64),
            np.array([2.0, 1.0], dtype=np.float32),
        ),
    )

    u, s, covariance_cols, picked_frequencies, column_fscs = pc.estimate_principal_components(
        cryos=cryos,
        options=options,
        means=means,
        mean_prior=np.zeros(cryo.volume_size, dtype=np.complex64),
        volume_mask=np.ones(cryo.volume_size, dtype=np.float32),
        dilated_volume_mask=np.ones(cryo.volume_size, dtype=np.float32),
        valid_idx=np.arange(cryo.n_images),
        batch_size=2,
        gpu_memory_to_use=8,
        covariance_options=cov_options,
    )

    assert u["rescaled"].shape == (cryo.volume_size, 2)
    np.testing.assert_array_equal(s["rescaled"], np.array([2.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(picked_frequencies, np.array([0, 1]))
    np.testing.assert_array_equal(column_fscs, np.array([0.1, 0.2]))
    assert covariance_cols["est_mask"] is None
