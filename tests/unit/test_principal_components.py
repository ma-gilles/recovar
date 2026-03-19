import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.heterogeneity.principal_components as pc
import recovar.core as core
from recovar.data_io import dataset
from recovar.reconstruction.homogeneous import MeanEstimate
from recovar.utils.helpers import AlgorithmOptions
from helpers.tiny_synthetic import make_tiny_simulation, make_tiny_cryo_dataset_with_images


def _make_means(vol_size, dtype_real=np.float32, dtype_complex=np.complex64):
    """Create a MeanEstimate with dummy data for testing."""
    return MeanEstimate(
        combined=np.ones(vol_size, dtype=dtype_complex),
        corrected0=np.ones(vol_size, dtype=dtype_complex),
        corrected1=np.ones(vol_size, dtype=dtype_complex),
        corrected0reg=np.ones(vol_size, dtype=dtype_complex),
        corrected1reg=np.ones(vol_size, dtype=dtype_complex),
        lhs=np.ones(vol_size, dtype=dtype_real),
        prior=np.ones(vol_size, dtype=dtype_real),
    )

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


@pytest.mark.parametrize("grid_size", [4, 8, 16, 32])
def test_flip_columns_structured_matches_batch_flip_vec2(grid_size):
    """flip_columns_structured (JAX CPU flip+conj) must return the same result
    as the legacy batch_flip_vec2 (fancy-indexed flip) for all grid sizes."""
    volume_shape = (grid_size, grid_size, grid_size)
    vol_size = int(np.prod(volume_shape))
    n_cols = 5

    rng = np.random.default_rng(42)
    columns = (rng.standard_normal((vol_size, n_cols)) +
               1j * rng.standard_normal((vol_size, n_cols))).astype(np.complex64)

    old_result = pc.batch_flip_vec2(columns, volume_shape).T  # (vol_size, n_cols)
    new_result = pc.flip_columns_structured(columns, volume_shape)  # (vol_size, n_cols)

    assert old_result.shape == new_result.shape
    np.testing.assert_allclose(new_result, old_result, rtol=1e-6, atol=1e-7)


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
    cryos = type("Cryo", (), {"image_size": 16})()
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
        dataset=cryos,
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
    assert s[2] == pytest.approx(pc.jax_config.EPSILON)


def test_estimate_principal_components_high_snr_from_var_est_requires_variance():
    mock_cryo = type(
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
    cryos = mock_cryo
    means = _make_means(8)
    options = AlgorithmOptions(volume_mask_option="none", zs_dim_to_test=[4], contrast="none", ignore_zero_frequency=False, keep_intermediate=True)
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
            dataset=cryos,
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
    mock_cryo = type(
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
    cryos = mock_cryo
    means = _make_means(8)
    options = AlgorithmOptions(volume_mask_option="none", zs_dim_to_test=[4], contrast="none", ignore_zero_frequency=False, keep_intermediate=False)
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
        dataset=cryos,
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
    metadata = dataset.Metadata(rots, trans, ctf_params)
    cryo = dataset.CryoEMDataset(
        image_source=None,
        voxel_size=voxel_size,
        metadata=metadata,
        ctf_evaluator=core.CTFEvaluator(),
        grid_size=4,
    )
    cryos = cryo

    means = _make_means(cryo.volume_size)
    options = AlgorithmOptions(volume_mask_option="none", zs_dim_to_test=[4], contrast="none", ignore_zero_frequency=False, keep_intermediate=False)
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
        dataset=cryos,
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


def test_pca_by_projected_covariance_real_tiny_dataset_runs():
    # grid_size>=6 required: simulator noise interpolation produces NaN at grid_size=4
    # (only 1 radial frequency bin → scipy interp1d divides by zero)
    cryo = make_tiny_cryo_dataset_with_images(grid_size=6, n_images=6, seed=0)
    basis = np.eye(cryo.volume_size, 4, dtype=np.complex64)

    u, s = pc.pca_by_projected_covariance(
        dataset=cryo,
        basis=basis,
        mean=np.zeros(cryo.volume_size, dtype=np.complex64),
        volume_mask=np.ones(cryo.volume_shape, dtype=np.float32),
        disc_type="linear_interp",
        disc_type_u="linear_interp",
        gpu_memory_to_use=8,
        use_mask=False,
        n_pcs_to_compute=2,
    )

    assert u.shape == (cryo.volume_size, 2)
    assert s.shape == (2,)
    assert np.isfinite(s).all()
    assert np.all(s >= pc.jax_config.EPSILON)
    assert np.all(s[:-1] >= s[1:])
    u_np = np.asarray(u)
    assert np.isfinite(u_np).all()
    gram = u_np.T @ u_np
    np.testing.assert_allclose(gram, np.eye(2), atol=5e-4, rtol=5e-4)


def test_right_matvec_with_spatial_sigma_v2_matches_v1_small_problem(monkeypatch):
    monkeypatch.setattr(pc.utils, "report_memory_device", lambda *args, **kwargs: None)

    volume_shape = (4, 4, 4)
    vol_size = int(np.prod(volume_shape))
    rng = np.random.default_rng(0)
    picked = rng.choice(vol_size, size=8, replace=False).astype(np.int32)
    columns = (rng.normal(size=(vol_size, picked.size)) + 1j * rng.normal(size=(vol_size, picked.size))).astype(np.complex64)

    picked_freqs = core.vec_indices_to_frequencies(picked, volume_shape)
    smaller_size = int(2 * (np.max(np.abs(np.asarray(picked_freqs))) + 1))
    smaller_vol_size = smaller_size ** 3
    test_mat = rng.normal(size=(smaller_vol_size, 3)).astype(np.float32)

    out_v1 = pc.right_matvec_with_spatial_Sigma(
        test_mat=test_mat,
        columns=columns,
        picked_frequency_indices=picked,
        volume_shape=volume_shape,
        vol_batch_size=2,
        memory_to_use=8,
    )
    out_v2 = pc.right_matvec_with_spatial_Sigma_v2(
        test_mat=test_mat,
        columns=columns,
        picked_frequency_indices=picked,
        volume_shape=volume_shape,
        vol_batch_size=2,
        memory_to_use=8,
    )

    np.testing.assert_allclose(np.asarray(out_v1), np.asarray(out_v2), atol=2e-5, rtol=2e-5)


def test_left_matvec_with_spatial_sigma_v2_matches_v1_small_problem(monkeypatch):
    monkeypatch.setattr(pc.utils, "report_memory_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(pc, "report_memory", False)

    volume_shape = (4, 4, 4)
    vol_size = int(np.prod(volume_shape))
    rng = np.random.default_rng(1)
    picked = rng.choice(vol_size, size=7, replace=False).astype(np.int32)
    columns = (rng.normal(size=(vol_size, picked.size)) + 1j * rng.normal(size=(vol_size, picked.size))).astype(np.complex64)

    Q = (rng.normal(size=(vol_size, 3)) + 1j * rng.normal(size=(vol_size, 3))).astype(np.complex64)

    out_v1 = pc.left_matvec_with_spatial_Sigma(
        Q=Q,
        columns=columns,
        picked_frequency_indices=picked,
        volume_shape=volume_shape,
        vol_batch_size=2,
        memory_to_use=8,
    )
    out_v2 = pc.left_matvec_with_spatial_Sigma_v2(
        Q=Q,
        columns=columns,
        picked_frequency_indices=picked,
        volume_shape=volume_shape,
        vol_batch_size=2,
        memory_to_use=8,
    )

    np.testing.assert_allclose(np.asarray(out_v1), np.asarray(out_v2), atol=2e-5, rtol=2e-5)


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_flip_vec_gpu(gpu_device):
    volume_shape = (4, 4, 4)
    vol_size = int(np.prod(volume_shape))
    col = np.arange(vol_size, dtype=np.float32).astype(np.complex64)

    cpu_flipped = np.asarray(pc.flip_vec(col, volume_shape))

    with jax.default_device(gpu_device):
        col_g = jax.device_put(jnp.array(col), gpu_device)
        gpu_flipped = np.asarray(pc.flip_vec(col_g, volume_shape))

    np.testing.assert_allclose(cpu_flipped, gpu_flipped, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_batch_flip_vec2_gpu(gpu_device):
    volume_shape = (4, 4, 4)
    vol_size = int(np.prod(volume_shape))
    col = np.arange(vol_size, dtype=np.float32).astype(np.complex64)
    cols = np.stack([col, col * 2], axis=1).astype(np.complex64)

    cpu_out = np.asarray(pc.batch_flip_vec2(cols, volume_shape))

    with jax.default_device(gpu_device):
        cols_g = jax.device_put(jnp.array(cols), gpu_device)
        gpu_out = np.asarray(pc.batch_flip_vec2(cols_g, volume_shape))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_make_symmetric_columns_gpu(gpu_device):
    volume_shape = (4, 4, 4)
    vol_size = int(np.prod(volume_shape))
    n_cols = 5
    rng = np.random.default_rng(0)

    columns = (rng.normal(size=(vol_size, n_cols)) + 1j * rng.normal(size=(vol_size, n_cols))).astype(np.complex64)
    picked = rng.choice(vol_size, size=n_cols, replace=False)

    cpu_cols, cpu_minus, cpu_good = pc.make_symmetric_columns(columns, picked, volume_shape)
    cpu_cols = np.asarray(cpu_cols)
    cpu_minus = np.asarray(cpu_minus)
    cpu_good = np.asarray(cpu_good)

    with jax.default_device(gpu_device):
        columns_g = jax.device_put(jnp.array(columns), gpu_device)
        picked_g = jax.device_put(jnp.array(picked), gpu_device)
        gpu_cols, gpu_minus, gpu_good = pc.make_symmetric_columns(columns_g, picked_g, volume_shape)
        gpu_cols = np.asarray(gpu_cols)
        gpu_minus = np.asarray(gpu_minus)
        gpu_good = np.asarray(gpu_good)

    np.testing.assert_allclose(cpu_cols, gpu_cols, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(cpu_minus, gpu_minus)
    np.testing.assert_array_equal(cpu_good, gpu_good)
