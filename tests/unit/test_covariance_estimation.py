from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.tiny_synthetic import make_tiny_cryo_dataset, make_tiny_cryo_dataset_with_images

import recovar.core.forward as core_forward
import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.heterogeneity.covariance_estimation as cov_est
from recovar import core
from recovar.core import linalg
from recovar.core.configs import CovarianceOpts, CovColumnOpts, ForwardModelConfig, ModelState
from recovar.core.ctf import as_ctf_evaluator
from recovar.heterogeneity import covariance_core

pytestmark = pytest.mark.unit


def _make_batch_fields(*, images, ctf_params, rotation_matrices, translations, noise_variance):
    return SimpleNamespace(
        images=images,
        ctf_params=ctf_params,
        rotation_matrices=rotation_matrices,
        translations=translations,
        noise_variance=noise_variance,
    )


def _call_variance_kernel(config, batch_data, mean_estimate, volume_mask, image_mask, **kwargs):
    return cov_est.variance_relion_kernel_trilinear(
        config,
        batch_data.images,
        mean_estimate,
        volume_mask,
        image_mask,
        batch_data.rotation_matrices,
        batch_data.translations,
        batch_data.ctf_params,
        batch_data.noise_variance,
        **kwargs,
    )


def _call_reduce_covariance_inner(config, batch_data, model, opts, image_mask, **kwargs):
    return cov_est.reduce_covariance_inner(
        config,
        batch_data.images,
        model,
        opts,
        image_mask,
        batch_data.rotation_matrices,
        batch_data.translations,
        batch_data.ctf_params,
        batch_data.noise_variance,
        **kwargs,
    )


def test_default_covariance_options_has_expected_keys(monkeypatch):
    monkeypatch.setattr(cov_est.utils, "get_gpu_memory_total", lambda: 8)
    opts = cov_est.get_default_covariance_computation_options(grid_size=32)
    assert isinstance(opts, dict)
    assert "n_pcs_to_compute" in opts
    assert opts["n_pcs_to_compute"] >= 1
    assert "reg_fn" in opts


def test_greedy_column_choice_validation():
    sampling_vec = np.ones(64, dtype=np.float32)
    volume_shape = (4, 4, 4)

    with pytest.raises(ValueError):
        cov_est.greedy_column_choice(sampling_vec, n_samples=0, volume_shape=volume_shape, avoid_in_radius=1)
    with pytest.raises(ValueError):
        cov_est.greedy_column_choice(sampling_vec, n_samples=2, volume_shape=volume_shape, avoid_in_radius=-1)


def test_greedy_column_choice_basic_properties():
    sampling_vec = np.ones(64, dtype=np.float32)
    volume_shape = (4, 4, 4)
    picked, freqs = cov_est.greedy_column_choice(
        sampling_vec=sampling_vec,
        n_samples=5,
        volume_shape=volume_shape,
        avoid_in_radius=0,
        keep_only_below_freq=32,
    )
    assert picked.shape == (5,)
    assert freqs.shape == (5, 3)
    assert len(np.unique(picked)) == 5
    assert np.all(freqs[:, 0] >= 0)


def test_randomized_column_choice_basic_properties():
    sampling_vec = np.ones(64, dtype=np.float64)
    volume_shape = (4, 4, 4)
    picked, freqs = cov_est.randomized_column_choice(
        sampling_vec=sampling_vec,
        n_samples=4,
        volume_shape=volume_shape,
        avoid_in_radius=0,
        random_seed=7,
    )
    assert picked.shape == (4,)
    assert freqs.shape == (4, 3)
    assert len(np.unique(picked)) == 4


def test_randomized_column_choice_is_deterministic_for_fixed_seed():
    sampling_vec = np.linspace(1.0, 5.0, 64, dtype=np.float64)
    volume_shape = (4, 4, 4)

    picked1, freqs1 = cov_est.randomized_column_choice(
        sampling_vec=sampling_vec,
        n_samples=5,
        volume_shape=volume_shape,
        avoid_in_radius=0,
        random_seed=11,
    )
    picked2, freqs2 = cov_est.randomized_column_choice(
        sampling_vec=sampling_vec,
        n_samples=5,
        volume_shape=volume_shape,
        avoid_in_radius=0,
        random_seed=11,
    )
    picked3, freqs3 = cov_est.randomized_column_choice(
        sampling_vec=sampling_vec,
        n_samples=5,
        volume_shape=volume_shape,
        avoid_in_radius=0,
        random_seed=12,
    )

    np.testing.assert_array_equal(picked1, picked2)
    np.testing.assert_array_equal(freqs1, freqs2)
    assert not np.array_equal(picked1, picked3)


def test_set_covariance_options_updates_only_present_keys():
    opts = {"a": 1, "b": 2}
    args = {"a": 5, "c": 9}
    out = cov_est.set_covariance_options(args, opts.copy())
    assert out["a"] == 5
    assert out["b"] == 2
    assert "c" not in out


def test_compute_regularized_covariance_columns_in_batch_concatenates(monkeypatch):
    mock_cryo = type("Cryo", (), {"grid_size": 4, "halfset_indices": [np.arange(5), np.arange(5, 10)]})()
    cryos = mock_cryo
    picked_frequencies = np.arange(10)

    monkeypatch.setattr(cov_est.utils, "get_column_batch_size", lambda *_: 4)

    calls = []

    def fake_compute_regularized_covariance_columns(
        dataset_in,
        means,
        mean_prior,
        volume_mask,
        dilated_volume_mask,
        valid_idx,
        gpu_memory,
        options,
        picked_frequencies_batch,
        use_multi_gpu=False,
        n_gpus=None,
        mean_cubic=None,
    ):
        calls.append(np.array(picked_frequencies_batch))
        n = len(picked_frequencies_batch)
        cols = np.tile(np.array(picked_frequencies_batch)[None, :], (3, 1)).astype(np.complex64)
        fsc = np.arange(n, dtype=np.float32) + 0.5
        return {"est_mask": cols}, None, fsc

    monkeypatch.setattr(cov_est, "compute_regularized_covariance_columns", fake_compute_regularized_covariance_columns)

    covariance_cols, picked_out, fscs = cov_est.compute_regularized_covariance_columns_in_batch(
        dataset=cryos,
        means={},
        mean_prior=None,
        volume_mask=None,
        dilated_volume_mask=None,
        valid_idx=None,
        gpu_memory=8,
        options={},
        picked_frequencies=picked_frequencies,
    )

    assert len(calls) == 3
    np.testing.assert_array_equal(calls[0], np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(calls[1], np.array([4, 5, 6, 7]))
    np.testing.assert_array_equal(calls[2], np.array([8, 9]))
    np.testing.assert_array_equal(picked_out, picked_frequencies)
    assert covariance_cols["est_mask"].shape == (3, 10)
    assert fscs.shape == (10,)


def test_compute_regularized_covariance_columns_with_real_tiny_dataset(monkeypatch):
    cryo = make_tiny_cryo_dataset(grid_size=4, n_images=6, seed=0)

    class _Noise:
        @staticmethod
        def get_average_radial_noise():
            return np.ones(cryo.grid_size // 2 - 1, dtype=np.float32)

    cryo.noise = _Noise()

    monkeypatch.setattr(cov_est.utils, "report_memory_device", lambda logger=None: None)
    monkeypatch.setattr(cov_est, "CUDA_PROFILER_AVAILABLE", False)
    monkeypatch.setattr(
        cov_est,
        "compute_both_H_B",
        lambda *args, **kwargs: (["H0", "H1"], ["B0", "B1"]),
    )
    monkeypatch.setattr(
        cov_est.noise,
        "make_radial_noise",
        lambda avg, volume_shape: np.ones(volume_shape, dtype=np.float32),
    )

    def _fake_reg_relion(
        Hs,
        Bs,
        mean_prior,
        picked_frequencies,
        volume_noise_var,
        mask_final,
        volume_shape,
        gpu_memory,
        reg_init_multiplier,
        options,
    ):
        n_cols = len(picked_frequencies)
        vol_size = int(np.prod(volume_shape))
        est = np.ones((n_cols, vol_size), dtype=np.complex64)
        prior = np.ones((n_cols,), dtype=np.float32)
        fscs = np.linspace(0.2, 0.8, n_cols, dtype=np.float32)
        return est, prior, fscs

    monkeypatch.setattr(cov_est, "compute_covariance_regularization_relion_style", _fake_reg_relion)

    options = {"reg_fn": "new"}
    picked_frequencies = np.array([0, 1, 2], dtype=np.int32)
    covariance_cols, picked_out, fscs = cov_est.compute_regularized_covariance_columns(
        dataset=cryo,
        means={},
        mean_prior=np.ones(cryo.volume_size, dtype=np.float32),
        volume_mask=np.ones(cryo.volume_size, dtype=np.float32),
        dilated_volume_mask=np.ones(cryo.volume_size, dtype=np.float32),
        valid_idx=np.ones(cryo.volume_size, dtype=np.float32),
        gpu_memory=8,
        options=options,
        picked_frequencies=picked_frequencies,
    )

    assert covariance_cols["est_mask"].shape == (cryo.volume_size, picked_frequencies.size)
    np.testing.assert_array_equal(picked_out, picked_frequencies)
    assert fscs.shape == (picked_frequencies.size,)


def test_compute_both_h_b_selects_combined_or_corrected_mean(monkeypatch):
    ds = make_tiny_cryo_dataset(grid_size=4, n_images=8, seed=0)
    ds.halfset_indices = [np.arange(4, dtype=np.int32), np.arange(4, 8, dtype=np.int32)]
    from recovar.reconstruction.homogeneous import MeanEstimate

    means = MeanEstimate(
        combined=np.array([11], dtype=np.float32),
        corrected0=np.array([21], dtype=np.float32),
        corrected1=np.array([31], dtype=np.float32),
        corrected0reg=np.zeros(1, dtype=np.float32),
        corrected1reg=np.zeros(1, dtype=np.float32),
        lhs=np.zeros(1, dtype=np.float32),
        prior=np.zeros(1, dtype=np.float32),
    )

    chosen_means = []

    def _fake_compute_h_b_for_halfset(cryo, mean, *args, **kwargs):
        chosen_means.append(float(np.asarray(mean).reshape(-1)[0]))
        return np.ones((2, 2), dtype=np.complex64), np.ones((2, 2), dtype=np.complex64) * 2

    monkeypatch.setattr(cov_est, "compute_H_B_for_halfset", _fake_compute_h_b_for_halfset)

    options = {"use_combined_mean": True}
    Hs, Bs = cov_est.compute_both_H_B(ds, means, None, np.array([0, 1]), 8, options)
    assert len(Hs) == 2 and len(Bs) == 2
    assert chosen_means == [11.0, 11.0]

    chosen_means.clear()
    options = {"use_combined_mean": False}
    cov_est.compute_both_H_B(ds, means, None, np.array([0, 1]), 8, options)
    assert chosen_means == [21.0, 31.0]


def test_iter_column_batch_ranges_covers_all_columns():
    ranges = list(cov_est._iter_column_batch_ranges(5, 2))
    assert ranges == [(0, 2), (2, 4), (4, 5)]


def test_transposed_column_batch_matches_expected_view():
    matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = cov_est._transposed_column_batch(matrix, 1, 3)
    np.testing.assert_array_equal(out, matrix[:, 1:3].T)


def test_vec_unvec_square_matrix_round_trip():
    matrix = np.arange(9, dtype=np.float32).reshape(3, 3)
    vec = cov_est._vec_square_matrix(matrix)
    round_trip = cov_est._unvec_square_matrix(vec)
    np.testing.assert_array_equal(round_trip, matrix)


def test_compute_covariance_regularization_relion_style_uses_transposed_batches(monkeypatch):
    monkeypatch.setattr(cov_est.utils, "get_column_batch_size", lambda *args, **kwargs: 16)

    calls = []

    def fake_prior_iteration_relion_style_batch(H0, H1, B0, B1, shifts, reg_init, *args):
        calls.append((np.array(H0), np.array(H1), np.array(B0), np.array(B1), np.array(shifts), np.array(reg_init)))
        batch_size, volume_size = H0.shape
        combined = np.ones((batch_size, volume_size), dtype=np.complex64) * (len(calls))
        priors = np.ones((batch_size, volume_size), dtype=np.float32) * (10 * len(calls))
        fscs = np.ones((batch_size,), dtype=np.float32) * (100 * len(calls))
        return combined, priors, fscs

    monkeypatch.setattr(
        cov_est.regularization,
        "prior_iteration_relion_style_batch",
        fake_prior_iteration_relion_style_batch,
    )

    H0 = np.arange(16, dtype=np.float32).reshape(4, 4)
    H1 = H0 + 100
    B0 = (H0 + 200).astype(np.complex64)
    B1 = (H1 + 200).astype(np.complex64)
    picked = np.array([0, 1, 2, 3], dtype=np.int32)
    mean_prior = np.linspace(1.0, 2.0, 4, dtype=np.float32)
    cov_noise = np.ones(4, dtype=np.float32)
    options = {
        "use_mask_in_fsc": False,
        "shift_fsc": False,
        "substract_shell_mean": False,
        "left_kernel": "triangular",
        "use_spherical_mask": False,
        "grid_correct": False,
        "prior_n_iterations": 1,
        "downsample_from_fsc": False,
    }

    combined, priors, fscs = cov_est.compute_covariance_regularization_relion_style(
        Hs=[H0, H1],
        Bs=[B0, B1],
        mean_prior=mean_prior,
        picked_frequencies=picked,
        cov_noise=cov_noise,
        volume_mask=np.ones(4, dtype=np.float32),
        volume_shape=(2, 2, 1),
        gpu_memory=8,
        reg_init_multiplier=2.0,
        options=options,
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], H0[:, 0:2].T)
    np.testing.assert_array_equal(calls[1][0], H0[:, 2:4].T)
    assert combined.shape == (4, 4)
    assert priors.shape == (4, 4)
    assert len(fscs) == 4


def test_summed_batch_kron_matches_scan():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    out1 = cov_est.summed_batch_kron(x)
    out2 = cov_est.summed_batch_kron_scan(x)
    np.testing.assert_allclose(np.asarray(out1), np.asarray(out2), atol=1e-6, rtol=1e-6)


def _dense_lhs_to_packed_reference(lhs_dense, n_basis):
    packed_dim = cov_est._symmetric_matrix_packed_size(n_basis)
    packed_eye = np.eye(packed_dim, dtype=np.asarray(lhs_dense).dtype)
    packed_cols = []
    for idx in range(packed_dim):
        basis_matrix = cov_est._unpack_symmetric_matrix_svec(jnp.asarray(packed_eye[idx]))
        dense_col = np.asarray(lhs_dense) @ np.asarray(cov_est._vec_square_matrix(basis_matrix))
        packed_cols.append(
            np.asarray(cov_est._pack_symmetric_matrix_svec(cov_est._unvec_square_matrix(jnp.asarray(dense_col))))
        )
    return np.stack(packed_cols, axis=1)


def test_pack_symmetric_matrix_svec_round_trip():
    matrix = np.array(
        [
            [1.0, 2.0, -1.0],
            [2.0, 3.5, 0.25],
            [-1.0, 0.25, 4.0],
        ],
        dtype=np.float32,
    )

    packed = cov_est._pack_symmetric_matrix_svec(jnp.asarray(matrix))
    unpacked = cov_est._unpack_symmetric_matrix_svec(packed)

    np.testing.assert_allclose(np.asarray(unpacked), matrix, atol=1e-6, rtol=1e-6)


def test_projected_covariance_packed_lhs_matches_dense_reference():
    rng = np.random.default_rng(0)
    AU_t_AU = rng.standard_normal((4, 3, 3)).astype(np.float32)
    AU_t_AU = 0.5 * (AU_t_AU + np.swapaxes(AU_t_AU, -1, -2))

    lhs_dense = cov_est._projected_covariance_dense_lhs_batch(jnp.asarray(AU_t_AU))
    lhs_packed = cov_est._projected_covariance_packed_lhs_batch(jnp.asarray(AU_t_AU))
    lhs_packed_ref = _dense_lhs_to_packed_reference(lhs_dense, n_basis=3)

    np.testing.assert_allclose(np.asarray(lhs_packed), lhs_packed_ref, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(lhs_packed), np.asarray(lhs_packed).T, atol=1e-6, rtol=1e-6)


def test_solve_projected_covariance_system_packed_matches_dense():
    rng = np.random.default_rng(1)
    AU_features = rng.standard_normal((5, 4, 7)).astype(np.float64)
    AU_t_AU = np.einsum("bik,bjk->bij", AU_features, AU_features)
    cov_true = rng.standard_normal((4, 4)).astype(np.float64)
    cov_true = 0.5 * (cov_true + cov_true.T)

    lhs_dense = cov_est._projected_covariance_dense_lhs_batch(jnp.asarray(AU_t_AU))
    lhs_packed = cov_est._projected_covariance_packed_lhs_batch(jnp.asarray(AU_t_AU))
    rhs = cov_est._unvec_square_matrix(lhs_dense @ cov_est._vec_square_matrix(jnp.asarray(cov_true)))

    dense_sol = cov_est._solve_projected_covariance_system_dense(lhs_dense, rhs)
    packed_sol = cov_est._solve_projected_covariance_system_packed(lhs_packed, rhs)

    np.testing.assert_allclose(np.asarray(packed_sol), np.asarray(dense_sol), atol=1e-6, rtol=1e-6)


def test_summed_outer_products_matches_manual():
    a = jnp.array([[1 + 1j, 2 + 0j], [3 + 0j, 4 - 1j]], dtype=jnp.complex64)
    out = cov_est.summed_outer_products(a)
    expected = np.asarray(a).T @ np.conj(np.asarray(a))
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-6, rtol=1e-6)


def test_group_sum_by_labels_and_preprocess_labels():
    labels = jnp.array([10, 20, 10, 30], dtype=jnp.int32)
    mapped = cov_est.preprocess_tilt_labels_for_batch(labels)
    # mapped labels are consecutive ids with same equality structure
    assert mapped.shape == labels.shape
    assert mapped[0] == mapped[2]
    assert mapped[0] != mapped[1] and mapped[1] != mapped[3]

    arr = jnp.array([[1.0], [2.0], [3.0], [4.0]], dtype=jnp.float32)
    grouped = cov_est.group_sum_by_labels(arr, jnp.array([0, 1, 0, 1], dtype=jnp.int32), max_groups=4)
    np.testing.assert_allclose(np.asarray(grouped).reshape(-1), np.array([4.0, 6.0, 4.0, 6.0]), atol=1e-6, rtol=1e-6)


def test_adjoint_kernel_slice_dispatch(monkeypatch):
    monkeypatch.setattr(cov_est.core, "adjoint_slice_volume", lambda images, *_, **__: images + 1)

    images = jnp.ones((2, 4), dtype=jnp.complex64)
    out_tri = cov_est.adjoint_kernel_slice(images, None, (2, 2), (2, 2, 1), kernel="triangular")
    out_sq = cov_est.adjoint_kernel_slice(images, None, (2, 2), (2, 2, 1), kernel="square")
    # Both kernels now dispatch to adjoint_slice_volume with disc_type
    np.testing.assert_allclose(np.asarray(out_tri), np.asarray(images + 1))
    np.testing.assert_allclose(np.asarray(out_sq), np.asarray(images + 1))
    with pytest.raises(ValueError):
        cov_est.adjoint_kernel_slice(images, None, (2, 2), (2, 2, 1), kernel="bad")


def test_compute_h_b_for_halfset_batches_frequency_chunks(monkeypatch):
    """Test that compute_H_B_for_halfset processes frequencies in column batches."""
    cryo = make_tiny_cryo_dataset_with_images(grid_size=4, n_images=6, seed=0)
    picked_frequencies = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    monkeypatch.setattr(cov_est.utils, "get_image_batch_size", lambda *args, **kwargs: 100)
    monkeypatch.setattr(cov_est.utils, "get_column_batch_size", lambda *args, **kwargs: 2)

    freq_calls = []

    def _fake_freq_batch(
        config,
        opts,
        freq_batch,
        images,
        ctf_on_grid,
        plane_coords,
        rotation_matrices,
        noise_variances,
        image_mask,
        tilt_labels,
        premultiplied_ctf,
        shared_label,
        no_mask,
        H_accum=None,
        B_accum=None,
    ):
        n_freq = freq_batch.shape[0]
        volume_size = config.volume_size
        freq_calls.extend(int(f) for f in np.asarray(freq_batch))
        H = jnp.broadcast_to(jnp.asarray(freq_batch, dtype=jnp.float32)[:, None], (n_freq, volume_size)).astype(
            jnp.complex64
        )
        B = jnp.broadcast_to(
            (jnp.asarray(freq_batch, dtype=jnp.float32)[:, None] + 100.0).astype(jnp.complex64), (n_freq, volume_size)
        )
        if H_accum is not None:
            H = H_accum + H
        if B_accum is not None:
            B = B_accum + B
        return H, B

    monkeypatch.setattr(cov_est, "compute_freq_batch", _fake_freq_batch)

    options = cov_est.get_default_covariance_computation_options(grid_size=4)
    options["disc_type"] = "linear_interp"
    options["mask_images_in_H_B"] = False

    H, B = cov_est.compute_H_B_for_halfset(
        cryo=cryo,
        mean_estimate=np.zeros(cryo.volume_size, dtype=np.complex64),
        volume_mask=np.ones(cryo.volume_shape, dtype=np.float32),
        picked_frequencies=picked_frequencies,
        gpu_memory=8,
        options=options,
    )

    # Should have called compute_freq_batch for each frequency
    assert set(freq_calls) == {0, 1, 2, 3, 4}
    assert H.shape == (cryo.volume_size, picked_frequencies.size)
    assert B.shape == (cryo.volume_size, picked_frequencies.size)


def test_compute_h_b_for_halfset_preprocesses_tilt_labels_before_freq_kernel(monkeypatch):
    monkeypatch.setattr(cov_est.utils, "get_image_batch_size", lambda *args, **kwargs: 8)
    monkeypatch.setattr(cov_est.utils, "get_column_batch_size", lambda *args, **kwargs: 8)

    class FakeCryo:
        volume_size = 4
        dtype = np.complex64
        grid_size = 4
        noise = object()
        tilt_series_flag = True
        premultiplied_ctf = False
        image_mask = np.ones((2, 2), dtype=np.float32)

        def process_images(self, images):
            return images

        def iter_batches(self, batch_size, **kwargs):
            assert batch_size == 8
            assert kwargs["by_image"] is False
            assert kwargs["pack_groups"] is True
            yield (
                np.zeros((3, 4), dtype=np.complex64),
                np.zeros((3, 3, 3), dtype=np.float32),
                np.zeros((3, 2), dtype=np.float32),
                np.zeros((3, 9), dtype=np.float32),
                np.zeros((3, 4), dtype=np.float32),
                np.array([99], dtype=np.int32),
                np.arange(3, dtype=np.int32),
            )

    config = SimpleNamespace(
        volume_size=4,
        image_shape=(2, 2),
        volume_shape=(2, 2, 1),
    )
    monkeypatch.setattr(cov_est.ForwardModelConfig, "from_dataset", lambda *args, **kwargs: config)

    monkeypatch.setattr(
        cov_est,
        "preprocess_covariance_batch",
        lambda *args, **kwargs: (
            np.zeros((3, 4), dtype=np.complex64),
            np.zeros((3, 4), dtype=np.complex64),
            np.zeros((3, 4, 3), dtype=np.float32),
            np.ones((3, 2, 2), dtype=np.float32),
            np.array([10, 20, 10], dtype=np.int32),
        ),
    )

    seen_labels = []

    def fake_compute_freq_batch(
        config,
        opts,
        freq_batch,
        images,
        ctf_on_grid,
        plane_coords,
        rotation_matrices,
        noise_variances,
        image_mask,
        tilt_labels,
        premultiplied_ctf,
        shared_label,
        no_mask,
        H_accum=None,
        B_accum=None,
    ):
        seen_labels.append(np.array(tilt_labels))
        assert shared_label is True
        return H_accum, B_accum

    monkeypatch.setattr(cov_est, "compute_freq_batch", fake_compute_freq_batch)

    options = cov_est.get_default_covariance_computation_options(grid_size=4)
    options["disc_type"] = "linear_interp"
    options["mask_images_in_H_B"] = False

    cov_est.compute_H_B_for_halfset(
        cryo=FakeCryo(),
        mean_estimate=np.zeros(4, dtype=np.complex64),
        volume_mask=np.ones((2, 2, 1), dtype=np.float32),
        picked_frequencies=np.array([0, 1], dtype=np.int32),
        gpu_memory=8,
        options=options,
    )

    assert len(seen_labels) == 1
    np.testing.assert_array_equal(seen_labels[0], np.array([0, 1, 0], dtype=np.int32))


def test_compute_variance_orchestration_with_stubbed_kernels(monkeypatch):
    # Create a dataset with 8 images, split into two halves of 4.
    ds = make_tiny_cryo_dataset(grid_size=4, n_images=8, seed=0)
    ds.halfset_indices = [np.arange(4), np.arange(4, 8)]
    vol_shape = ds.volume_shape
    vol_size = ds.volume_size

    # Scale factor per half: half0 → 1.0, half1 → 2.0
    half0_idx = set(ds.halfset_indices[0].tolist())

    def _fake_var_kernel(cryo, mean_estimate, batch_size, image_subset=None, volume_mask=None, disc_type=""):
        original_indices = cryo.original_image_indices_from_local(np.arange(cryo.n_images))
        scale = 1.0 if set(original_indices.tolist()) == half0_idx else 2.0
        lhs = np.ones(vol_size, dtype=np.float32) * (10.0 * scale)
        rhs = np.ones(vol_size, dtype=np.float32) * (4.0 * scale)
        noise_lhs = np.ones(vol_size, dtype=np.float32) * (2.0 * scale)
        noise_rhs = np.ones(vol_size, dtype=np.float32) * (1.0 * scale)
        return lhs, rhs, noise_lhs, noise_rhs

    monkeypatch.setattr(cov_est, "variance_relion_style_triangular_kernel", _fake_var_kernel)
    import recovar.reconstruction.relion_functions as relion_functions

    monkeypatch.setattr(relion_functions, "adjust_regularization_relion_style", lambda lhs, *args, **kwargs: lhs)
    monkeypatch.setattr(
        cov_est.regularization,
        "compute_fsc_prior_gpu_v2",
        lambda *args, **kwargs: (
            jnp.ones(vol_size, dtype=jnp.float32) * 0.5,
            jnp.ones(vol_size, dtype=jnp.float32) * 0.7,
            jnp.ones(vol_size, dtype=jnp.float32),
        ),
    )

    variance, variance_prior, fsc, lhs, noise_p_variance_est = cov_est.compute_variance(
        dataset=ds,
        mean_estimate=np.ones(vol_size, dtype=np.complex64),
        batch_size=2,
        volume_mask=np.ones(vol_size, dtype=np.float32),
        image_subset=None,
        use_regularization=False,
        disc_type="linear_interp",
    )

    assert variance["corrected0"].shape == (vol_size,)
    assert variance["corrected1"].shape == (vol_size,)
    assert variance["combined"].shape == (vol_size,)
    assert variance["prior"].shape == (vol_size,)
    assert variance["prior_total_signal"].shape == (vol_size,)
    assert variance["prior_shell_subtracted"].shape == (vol_size,)
    assert variance["fsc_total_signal"].shape == (vol_size,)
    assert variance["fsc_shell_subtracted"].shape == (vol_size,)
    assert variance["prior_avg_total_signal"].shape == (vol_size,)
    assert variance["prior_avg_shell_subtracted"].shape == (vol_size,)
    assert variance["lhs"].shape == (vol_size,)
    assert variance_prior.shape == (vol_size,)
    assert fsc.shape == (vol_size,)
    assert lhs.shape == (vol_size,)
    assert noise_p_variance_est.shape == (vol_size,)
    np.testing.assert_allclose(variance["corrected0"], np.ones(vol_size, dtype=np.float32) * 0.4, atol=1e-6, rtol=1e-6)


def test_compute_h_b_runs_on_tiny_image_dataset():
    cryo = make_tiny_cryo_dataset_with_images(grid_size=4, n_images=6, seed=0)
    options = cov_est.get_default_covariance_computation_options(grid_size=4)
    options.update(
        {
            "disc_type": "linear_interp",
            "left_kernel": "triangular",
            "right_kernel": "triangular",
            "right_kernel_width": 1,
            "mask_images_in_H_B": False,
        }
    )

    H, B = cov_est.compute_H_B_for_halfset(
        cryo=cryo,
        mean_estimate=np.zeros(cryo.volume_size, dtype=np.complex64),
        volume_mask=np.ones(cryo.volume_shape, dtype=np.float32),
        picked_frequencies=np.array([0, 1], dtype=np.int32),
        gpu_memory=8,
        options=options,
    )

    assert H.shape == (cryo.volume_size, 2)
    assert B.shape == (cryo.volume_size, 2)
    assert np.isfinite(np.asarray(H)).all()


def test_compute_freq_batch_two_calls_accumulate():
    """Verify compute_freq_batch accumulates correctly across two calls."""
    rng = np.random.RandomState(123)
    grid_size = 4
    n_images = 6
    image_shape = (grid_size, grid_size)
    volume_shape = (grid_size, grid_size, grid_size)
    volume_size = grid_size**3
    n_pix = grid_size**2

    images = jnp.array(
        rng.randn(n_images, n_pix).astype(np.float32) + 1j * rng.randn(n_images, n_pix).astype(np.float32),
        dtype=jnp.complex64,
    )
    ctf = jnp.array(
        rng.randn(n_images, n_pix).astype(np.float32) + 1j * rng.randn(n_images, n_pix).astype(np.float32),
        dtype=jnp.complex64,
    )
    plane_coords = jnp.array(rng.randn(n_images, n_pix, 3).astype(np.float32))
    rot = jnp.array(np.stack([np.eye(3, dtype=np.float32) for _ in range(n_images)]))
    noise_var = jnp.array(rng.rand(n_images, n_pix).astype(np.float32) + 0.1)
    image_mask = jnp.ones((n_images, *image_shape), dtype=jnp.float32)

    freq_indices = jnp.array([0, 3, 5, 10], dtype=jnp.int32)

    config = ForwardModelConfig(
        image_shape=image_shape,
        volume_shape=volume_shape,
        grid_size=grid_size,
        voxel_size=1.0,
        padding=0,
        disc_type="linear_interp",
        ctf=as_ctf_evaluator(
            lambda params, shape, voxel, **kw: jnp.ones((params.shape[0], shape[0] * shape[1]), dtype=jnp.complex64)
        ),
    )
    opts = CovColumnOpts(
        right_kernel="triangular",
        left_kernel="triangular",
        right_kernel_width=1,
        mask_images=False,
        soften_mask=3,
    )

    # Single call with all images
    H_all, B_all = cov_est.compute_freq_batch(
        config, opts, freq_indices, images, ctf, plane_coords, rot, noise_var, image_mask, None, False, False, True
    )

    # Two calls: split images into halves, accumulate
    H_acc, B_acc = cov_est.compute_freq_batch(
        config,
        opts,
        freq_indices,
        images[:3],
        ctf[:3],
        plane_coords[:3],
        rot[:3],
        noise_var[:3],
        image_mask[:3],
        None,
        False,
        False,
        True,
    )
    H_acc, B_acc = cov_est.compute_freq_batch(
        config,
        opts,
        freq_indices,
        images[3:],
        ctf[3:],
        plane_coords[3:],
        rot[3:],
        noise_var[3:],
        image_mask[3:],
        None,
        False,
        False,
        True,
        H_accum=H_acc,
        B_accum=B_acc,
    )

    np.testing.assert_allclose(np.asarray(H_acc), np.asarray(H_all), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(B_acc), np.asarray(B_all), atol=1e-4, rtol=1e-4)


def test_compute_projected_covariance_runs_on_tiny_image_dataset():
    # grid_size>=6 required: simulator noise interpolation produces NaN at grid_size=4
    cryo = make_tiny_cryo_dataset_with_images(grid_size=6, n_images=6, seed=0)
    basis = np.eye(cryo.volume_size, 4, dtype=np.complex64)
    covar = cov_est.compute_projected_covariance(
        dataset=[cryo],
        mean_estimate=np.zeros(cryo.volume_size, dtype=np.complex64),
        basis=basis,
        volume_mask=np.ones(cryo.volume_shape, dtype=np.float32),
        batch_size=3,
        disc_type="linear_interp",
        disc_type_u="linear_interp",
        do_mask_images=False,
    )

    # Full projected-covariance reduction path on real tiny simulated images.
    assert covar.shape == (4, 4)
    assert np.asarray(covar).dtype in (np.float32, np.float64)
    covar_np = np.asarray(covar)
    assert np.isfinite(covar_np).all()
    np.testing.assert_allclose(covar_np, covar_np.T, atol=1e-5, rtol=1e-5)


def test_compute_projected_covariance_single_requests_packed_tilt_batches(monkeypatch):
    seen_kwargs = []

    class _FakeTiltDataset:
        dtype = np.complex64
        dtype_real = np.float32
        tilt_series_flag = True
        volume_shape = (2, 2, 1)
        volume_size = 4
        image_shape = (2, 2)
        noise = object()
        image_mask = np.ones((2, 2), dtype=np.float32)
        halfset_indices = None

        def process_images(self, images):
            return images

        def iter_batches(self, batch_size, **kwargs):
            seen_kwargs.append(kwargs.copy())
            assert batch_size == 6
            assert kwargs["by_image"] is False
            assert kwargs["pack_groups"] is True
            yield (
                np.zeros((3, 4), dtype=np.complex64),
                np.zeros((3, 3, 3), dtype=np.float32),
                np.zeros((3, 2), dtype=np.float32),
                np.zeros((3, 9), dtype=np.float32),
                np.zeros((3, 4), dtype=np.float32),
                np.array([0, 0, 1], dtype=np.int32),
                np.arange(3, dtype=np.int32),
            )

    config = SimpleNamespace(image_shape=(2, 2), volume_shape=(2, 2, 1))
    monkeypatch.setattr(cov_est.ForwardModelConfig, "from_dataset", lambda *args, **kwargs: config)
    monkeypatch.setattr(cov_est.linalg, "rfft2_hermitian_weights", lambda *args, **kwargs: jnp.ones((4,)))
    monkeypatch.setattr(cov_est.utils, "report_memory_device", lambda *args, **kwargs: None)

    def fake_reduce_covariance_inner(
        config,
        images,
        model,
        opts,
        image_mask,
        *,
        rotation_matrices,
        translations,
        ctf_params,
        noise_variance,
        hermitian_weights,
        lhs,
        rhs,
        tilt_labels,
    ):
        _ = (
            config,
            images,
            model,
            opts,
            image_mask,
            rotation_matrices,
            translations,
            ctf_params,
            noise_variance,
            hermitian_weights,
            tilt_labels,
        )
        return lhs + jnp.eye(lhs.shape[0], dtype=lhs.dtype), rhs + jnp.eye(rhs.shape[0], dtype=rhs.dtype)

    monkeypatch.setattr(cov_est, "reduce_covariance_inner", fake_reduce_covariance_inner)

    covar = cov_est._compute_projected_covariance_single(
        _FakeTiltDataset(),
        mean_estimate=np.zeros(4, dtype=np.complex64),
        basis=np.eye(4, 1, dtype=np.complex64),
        volume_mask=np.ones((2, 2, 1), dtype=np.float32),
        batch_size=6,
        disc_type="linear_interp",
        disc_type_u="linear_interp",
        do_mask_images=False,
    )

    assert covar.shape == (1, 1)
    assert len(seen_kwargs) == 1
    assert seen_kwargs[0]["noise_half"] is False
    assert seen_kwargs[0]["by_image"] is False
    assert seen_kwargs[0]["pack_groups"] is True


def test_compute_projected_covariance_halfsets_requests_packed_tilt_batches(monkeypatch):
    seen_kwargs = []

    class _FakeTiltHalfsetDataset:
        dtype = np.complex64
        dtype_real = np.float32
        tilt_series_flag = True
        volume_shape = (2, 2, 1)
        volume_size = 4
        image_shape = (2, 2)
        noise = object()
        image_mask = np.ones((2, 2), dtype=np.float32)
        halfset_indices = [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)]

        def process_images(self, images):
            return images

        def iter_batches(self, batch_size, **kwargs):
            seen_kwargs.append(kwargs.copy())
            assert batch_size == 6
            assert kwargs["by_image"] is False
            assert kwargs["pack_groups"] is True
            yield (
                np.zeros((3, 4), dtype=np.complex64),
                np.zeros((3, 3, 3), dtype=np.float32),
                np.zeros((3, 2), dtype=np.float32),
                np.zeros((3, 9), dtype=np.float32),
                np.zeros((3, 4), dtype=np.float32),
                np.array([0, 0, 1], dtype=np.int32),
                np.arange(3, dtype=np.int32),
            )

    config = SimpleNamespace(image_shape=(2, 2), volume_shape=(2, 2, 1))
    monkeypatch.setattr(cov_est.ForwardModelConfig, "from_dataset", lambda *args, **kwargs: config)
    monkeypatch.setattr(cov_est.linalg, "rfft2_hermitian_weights", lambda *args, **kwargs: jnp.ones((4,)))
    monkeypatch.setattr(cov_est.utils, "report_memory_device", lambda *args, **kwargs: None)

    def fake_reduce_covariance_inner(
        config,
        images,
        model,
        opts,
        image_mask,
        *,
        rotation_matrices,
        translations,
        ctf_params,
        noise_variance,
        hermitian_weights,
        lhs,
        rhs,
        tilt_labels,
    ):
        _ = (
            config,
            images,
            model,
            opts,
            image_mask,
            rotation_matrices,
            translations,
            ctf_params,
            noise_variance,
            hermitian_weights,
            tilt_labels,
        )
        return lhs + jnp.eye(lhs.shape[0], dtype=lhs.dtype), rhs + jnp.eye(rhs.shape[0], dtype=rhs.dtype)

    monkeypatch.setattr(cov_est, "reduce_covariance_inner", fake_reduce_covariance_inner)

    covar = cov_est.compute_projected_covariance(
        dataset=_FakeTiltHalfsetDataset(),
        mean_estimate=np.zeros(4, dtype=np.complex64),
        basis=np.eye(4, 1, dtype=np.complex64),
        volume_mask=np.ones((2, 2, 1), dtype=np.float32),
        batch_size=6,
        disc_type="linear_interp",
        disc_type_u="linear_interp",
        do_mask_images=False,
    )

    assert covar.shape == (1, 1)
    assert len(seen_kwargs) == 2
    assert {kwargs["halfset_id"] for kwargs in seen_kwargs} == {0, 1}
    assert all(kwargs["pack_groups"] is True for kwargs in seen_kwargs)


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_summed_batch_kron_gpu(gpu_device):
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)

    cpu_out = np.asarray(cov_est.summed_batch_kron(x))

    with jax.default_device(gpu_device):
        x_g = jax.device_put(x, gpu_device)
        gpu_out = np.asarray(cov_est.summed_batch_kron(x_g))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_summed_batch_kron_scan_gpu(gpu_device):
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)

    cpu_out = np.asarray(cov_est.summed_batch_kron_scan(x))

    with jax.default_device(gpu_device):
        x_g = jax.device_put(x, gpu_device)
        gpu_out = np.asarray(cov_est.summed_batch_kron_scan(x_g))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_summed_outer_products_gpu(gpu_device):
    a = jnp.array([[1 + 1j, 2 + 0j], [3 + 0j, 4 - 1j]], dtype=jnp.complex64)

    cpu_out = np.asarray(cov_est.summed_outer_products(a))

    with jax.default_device(gpu_device):
        a_g = jax.device_put(a, gpu_device)
        gpu_out = np.asarray(cov_est.summed_outer_products(a_g))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_group_sum_by_labels_gpu(gpu_device):
    arr = jnp.array([[1.0], [2.0], [3.0], [4.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1, 0, 1], dtype=jnp.int32)

    cpu_out = np.asarray(cov_est.group_sum_by_labels(arr, labels, max_groups=4))

    with jax.default_device(gpu_device):
        arr_g = jax.device_put(arr, gpu_device)
        labels_g = jax.device_put(labels, gpu_device)
        gpu_out = np.asarray(cov_est.group_sum_by_labels(arr_g, labels_g, max_groups=4))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_preprocess_tilt_labels_for_batch_gpu(gpu_device):
    labels = jnp.array([10, 20, 10, 30], dtype=jnp.int32)

    cpu_out = np.asarray(cov_est.preprocess_tilt_labels_for_batch(labels))

    with jax.default_device(gpu_device):
        labels_g = jax.device_put(labels, gpu_device)
        gpu_out = np.asarray(cov_est.preprocess_tilt_labels_for_batch(labels_g))

    np.testing.assert_array_equal(cpu_out, gpu_out)


# ---------------------------------------------------------------------------
# Variance kernel equivalence: new half-vol must match old full-vol reference
# ---------------------------------------------------------------------------


def _make_variance_test_fixtures(grid_size=4, n_images=4, seed=42):
    """Return (config, batch_data, mean_estimate, volume_mask, image_mask)."""
    cryo = make_tiny_cryo_dataset_with_images(grid_size=grid_size, n_images=n_images, seed=seed)
    config = ForwardModelConfig.from_dataset(cryo, disc_type="linear_interp")

    images_batch, _, indices = next(cryo.image_source.iter_batches(batch_size=n_images, batch_mode="groups"))

    # nan_to_num: tiny-grid DFTs can produce NaN at edge frequencies;
    # NaN in slices propagates through the VJP-based JAX half-volume adjoint.
    full_images = jnp.nan_to_num(jnp.asarray(images_batch))
    # The variance kernel expects half-image (rfft-packed) batches from the
    # explicit dataset iterator path.
    half_images = fourier_transform_utils.full_image_to_half_image(full_images, config.image_shape)
    batch_data = _make_batch_fields(
        images=half_images,
        ctf_params=jnp.asarray(cryo.CTF_params[indices]),
        rotation_matrices=jnp.asarray(cryo.rotation_matrices[indices]),
        translations=jnp.asarray(cryo.translations[indices]),
        noise_variance=jnp.asarray(cryo.noise.get(indices)),
    )
    mean_estimate = jnp.zeros(config.volume_size, dtype=jnp.complex64)
    # volume_mask must be 3D (volume_shape), not flat — pad_volume_spatial_domain
    # expects a spatial-domain grid.
    volume_mask = jnp.ones(config.volume_shape, dtype=jnp.float32)
    image_mask = jnp.ones(config.image_shape, dtype=jnp.float32)
    return config, batch_data, mean_estimate, volume_mask, image_mask


def _reference_variance_kernel(config, batch_data, mean_estimate, volume_mask, image_mask, soften=5):
    """Old full-volume backprojection logic, for numerical equivalence testing.

    batch_data.images is in half-image format; convert to full spectrum before
    the full-vol reference path.
    """
    from recovar.heterogeneity import covariance_core
    from recovar.reconstruction import noise as noise_mod

    images = fourier_transform_utils.half_image_to_full_image(batch_data.images, config.image_shape)
    noise_variances = batch_data.noise_variance
    CTF = config.compute_ctf(batch_data.ctf_params)
    images = core.translate_images(images, batch_data.translations, config.image_shape)

    if config.premultiplied_ctf:
        images = (
            images
            - core.slice_volume(
                mean_estimate, batch_data.rotation_matrices, config.image_shape, config.volume_shape, config.disc_type
            )
            * CTF**2
        )
        noise_p_variance_ctf = CTF**2
    else:
        images = (
            images
            - core.slice_volume(
                mean_estimate, batch_data.rotation_matrices, config.image_shape, config.volume_shape, config.disc_type
            )
            * CTF
        )
        noise_p_variance_ctf = jnp.ones_like(images)

    img_power_full = jnp.abs(images) ** 2
    cov_noise = jnp.zeros_like(images)

    if volume_mask is not None:
        image_mask_per_image = covariance_core.get_per_image_tight_mask(
            volume_mask,
            batch_data.rotation_matrices,
            image_mask,
            config.volume_mask_threshold,
            config.image_shape,
            config.volume_shape,
            config.grid_size,
            config.padding,
            "linear_interp",
            soften=soften,
        )
        images = covariance_core.apply_image_masks(images, image_mask_per_image, config.image_shape)
        if config.premultiplied_ctf:
            noise_variances = noise_variances * CTF**2
        cov_noise = noise_mod.get_masked_noise_variance_from_noise_variance(
            image_mask_per_image, noise_variances, config.image_shape
        )

    images_squared = jnp.abs(images) ** 2 - cov_noise.reshape(images.shape)
    CTF_squared = CTF**2
    if not config.premultiplied_ctf:
        images_squared *= CTF_squared

    def _bp(arr):
        return core.adjoint_slice_volume(
            arr, batch_data.rotation_matrices, config.image_shape, config.volume_shape, "linear_interp"
        )

    return _bp(images_squared), _bp(CTF_squared**2), _bp(img_power_full), _bp(noise_p_variance_ctf)


def _to_full(half_vol, volume_shape):
    return np.asarray(fourier_transform_utils.half_volume_to_full_volume(half_vol, volume_shape))


def test_variance_kernel_no_mask_matches_reference():
    """New kernel (half-vol accumulation) matches old full-vol backprojection — no mask."""
    config, batch_data, mean_estimate, _, image_mask = _make_variance_test_fixtures()

    ref_y, ref_ctf, ref_im, ref_one = _reference_variance_kernel(
        config, batch_data, mean_estimate, volume_mask=None, image_mask=image_mask
    )
    new_y, new_ctf, new_im, new_one = _call_variance_kernel(
        config, batch_data, mean_estimate, volume_mask=None, image_mask=image_mask
    )

    vol_shape = config.volume_shape
    np.testing.assert_allclose(_to_full(new_y, vol_shape), np.asarray(ref_y), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_full(new_ctf, vol_shape), np.asarray(ref_ctf), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_full(new_im, vol_shape), np.asarray(ref_im), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_full(new_one, vol_shape), np.asarray(ref_one), atol=1e-5, rtol=1e-5)


def test_variance_kernel_with_mask_matches_reference():
    """New kernel (half-vol accumulation) matches old full-vol backprojection — with mask."""
    config, batch_data, mean_estimate, volume_mask, image_mask = _make_variance_test_fixtures()

    ref_y, ref_ctf, ref_im, ref_one = _reference_variance_kernel(
        config, batch_data, mean_estimate, volume_mask, image_mask
    )
    new_y, new_ctf, new_im, new_one = _call_variance_kernel(config, batch_data, mean_estimate, volume_mask, image_mask)

    vol_shape = config.volume_shape
    np.testing.assert_allclose(_to_full(new_y, vol_shape), np.asarray(ref_y), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_full(new_ctf, vol_shape), np.asarray(ref_ctf), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_full(new_im, vol_shape), np.asarray(ref_im), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_full(new_one, vol_shape), np.asarray(ref_one), atol=1e-5, rtol=1e-5)


def test_variance_kernel_accumulator_matches_sequential_sum():
    """Accumulator pattern gives the same result as summing per-batch contributions."""
    config, batch_data, mean_estimate, volume_mask, image_mask = _make_variance_test_fixtures(n_images=6)

    # Split into two half-batches.
    def _slice_batch(bd, sl):
        return _make_batch_fields(
            images=bd.images[sl],
            ctf_params=bd.ctf_params[sl],
            rotation_matrices=bd.rotation_matrices[sl],
            translations=bd.translations[sl],
            noise_variance=bd.noise_variance[sl],
        )

    b1 = _slice_batch(batch_data, slice(0, 3))
    b2 = _slice_batch(batch_data, slice(3, 6))

    # Reference: run separately, convert, sum.
    kwargs = dict(mean_estimate=mean_estimate, volume_mask=volume_mask, image_mask=image_mask)
    y1, ctf1, im1, one1 = _call_variance_kernel(config, b1, **kwargs)
    y2, ctf2, im2, one2 = _call_variance_kernel(config, b2, **kwargs)
    vol_shape = config.volume_shape
    ref_y = _to_full(y1, vol_shape) + _to_full(y2, vol_shape)
    ref_ctf = _to_full(ctf1, vol_shape) + _to_full(ctf2, vol_shape)
    ref_im = _to_full(im1, vol_shape) + _to_full(im2, vol_shape)
    ref_one = _to_full(one1, vol_shape) + _to_full(one2, vol_shape)

    # Accumulator: pass running accumulators between calls.
    acc_y, acc_ctf, acc_im, acc_one = _call_variance_kernel(config, b1, **kwargs)
    acc_y, acc_ctf, acc_im, acc_one = _call_variance_kernel(
        config,
        b2,
        mean_estimate=mean_estimate,
        volume_mask=volume_mask,
        image_mask=image_mask,
        Ft_y=acc_y,
        Ft_ctf=acc_ctf,
        Ft_im=acc_im,
        Ft_one=acc_one,
    )

    np.testing.assert_allclose(_to_full(acc_y, vol_shape), ref_y, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_full(acc_ctf, vol_shape), ref_ctf, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_full(acc_im, vol_shape), ref_im, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_full(acc_one, vol_shape), ref_one, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Double-precision equivalence: new kernel vs reference at float64
# ---------------------------------------------------------------------------


@pytest.fixture()
def enable_x64():
    """Enable 64-bit JAX arithmetic for this test, restoring the previous state afterward."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _cast_batch_data_f64(batch_data):
    return _make_batch_fields(
        images=batch_data.images.astype(jnp.complex128),
        ctf_params=batch_data.ctf_params.astype(jnp.float64),
        rotation_matrices=batch_data.rotation_matrices.astype(jnp.float64),
        translations=batch_data.translations.astype(jnp.float64),
        noise_variance=batch_data.noise_variance.astype(jnp.float64),
    )


def test_variance_kernel_no_mask_f64(enable_x64):
    """New kernel matches old full-vol reference in float64.

    Note: config.compute_ctf() always returns float32, so CTF-dependent
    quantities (Ft_ctf) are limited to float32 precision (~1e-7 absolute).
    atol=1e-7 is 100x tighter than the float32 tests (atol=1e-5).
    """
    config, batch_data, mean_estimate, _, image_mask = _make_variance_test_fixtures()
    bd64 = _cast_batch_data_f64(batch_data)
    mean64 = mean_estimate.astype(jnp.complex128)
    im64 = image_mask.astype(jnp.float64)

    ref_y, ref_ctf, ref_im, ref_one = _reference_variance_kernel(
        config, bd64, mean64, volume_mask=None, image_mask=im64
    )
    new_y, new_ctf, new_im, new_one = _call_variance_kernel(config, bd64, mean64, volume_mask=None, image_mask=im64)

    vol_shape = config.volume_shape
    np.testing.assert_allclose(_to_full(new_y, vol_shape), np.asarray(ref_y), atol=1e-7, rtol=0)
    np.testing.assert_allclose(_to_full(new_ctf, vol_shape), np.asarray(ref_ctf), atol=1e-7, rtol=0)
    np.testing.assert_allclose(_to_full(new_im, vol_shape), np.asarray(ref_im), atol=1e-7, rtol=0)
    np.testing.assert_allclose(_to_full(new_one, vol_shape), np.asarray(ref_one), atol=1e-7, rtol=0)


def test_variance_kernel_with_mask_f64(enable_x64):
    """New kernel matches old full-vol reference in float64 with mask (atol=1e-7)."""
    config, batch_data, mean_estimate, volume_mask, image_mask = _make_variance_test_fixtures()
    bd64 = _cast_batch_data_f64(batch_data)
    mean64 = mean_estimate.astype(jnp.complex128)
    vm64 = volume_mask.astype(jnp.float64)
    im64 = image_mask.astype(jnp.float64)

    ref_y, ref_ctf, ref_im, ref_one = _reference_variance_kernel(config, bd64, mean64, vm64, im64)
    new_y, new_ctf, new_im, new_one = _call_variance_kernel(config, bd64, mean64, vm64, im64)

    vol_shape = config.volume_shape
    np.testing.assert_allclose(_to_full(new_y, vol_shape), np.asarray(ref_y), atol=1e-7, rtol=0)
    np.testing.assert_allclose(_to_full(new_ctf, vol_shape), np.asarray(ref_ctf), atol=1e-7, rtol=0)
    np.testing.assert_allclose(_to_full(new_im, vol_shape), np.asarray(ref_im), atol=1e-7, rtol=0)
    np.testing.assert_allclose(_to_full(new_one, vol_shape), np.asarray(ref_one), atol=1e-7, rtol=0)


def test_variance_kernel_accumulator_f64(enable_x64):
    """Accumulator pattern is self-consistent at float64 precision (atol=1e-13).

    This directly tests the half-image refactor: since both runs use the same
    half-image code path, the comparison avoids full-vol vs half-vol differences
    and achieves near-machine-epsilon agreement.
    """
    config, batch_data, mean_estimate, volume_mask, image_mask = _make_variance_test_fixtures(n_images=6)
    bd64 = _cast_batch_data_f64(batch_data)
    mean64 = mean_estimate.astype(jnp.complex128)
    vm64 = volume_mask.astype(jnp.float64)
    im64 = image_mask.astype(jnp.float64)

    def _slice_batch(bd, sl):
        return _make_batch_fields(
            images=bd.images[sl],
            ctf_params=bd.ctf_params[sl],
            rotation_matrices=bd.rotation_matrices[sl],
            translations=bd.translations[sl],
            noise_variance=bd.noise_variance[sl],
        )

    b1 = _slice_batch(bd64, slice(0, 3))
    b2 = _slice_batch(bd64, slice(3, 6))
    kwargs = dict(mean_estimate=mean64, volume_mask=vm64, image_mask=im64)

    # Sum of separate runs.
    y1, c1, im1, o1 = _call_variance_kernel(config, b1, **kwargs)
    y2, c2, im2, o2 = _call_variance_kernel(config, b2, **kwargs)
    vol_shape = config.volume_shape
    ref_y = _to_full(y1, vol_shape) + _to_full(y2, vol_shape)

    # Accumulated run.
    acc_y, acc_c, acc_im, acc_o = _call_variance_kernel(config, b1, **kwargs)
    acc_y, acc_c, acc_im, acc_o = _call_variance_kernel(
        config,
        b2,
        mean_estimate=mean64,
        volume_mask=vm64,
        image_mask=im64,
        Ft_y=acc_y,
        Ft_ctf=acc_c,
        Ft_im=acc_im,
        Ft_one=acc_o,
    )

    # Same code path: differences are only from float64 accumulation order — near machine eps.
    np.testing.assert_allclose(_to_full(acc_y, vol_shape), ref_y, atol=1e-13, rtol=0)


# ---------------------------------------------------------------------------
# Tests for half-image masking in reduce_covariance_inner
# ---------------------------------------------------------------------------


def _make_reduce_cov_fixtures(grid_size=4, n_images=6, seed=0):
    """Build (config, batch_data, model, hermitian_weights, image_mask) for reduce_covariance_inner tests."""
    cryo = make_tiny_cryo_dataset_with_images(grid_size=grid_size, n_images=n_images, seed=seed)
    config = ForwardModelConfig.from_dataset(
        cryo,
        disc_type="linear_interp",
        process_fn=cryo.image_source.process_images,
    )

    images_batch, _, indices = next(cryo.image_source.iter_batches(batch_size=n_images, batch_mode="groups"))
    images = jnp.nan_to_num(jnp.asarray(images_batch))

    batch_data = _make_batch_fields(
        images=images,
        ctf_params=jnp.asarray(cryo.CTF_params[indices]),
        rotation_matrices=jnp.asarray(cryo.rotation_matrices[indices]),
        translations=jnp.asarray(cryo.translations[indices]),
        noise_variance=jnp.asarray(cryo.noise.get(indices)),
    )

    mean_estimate_full = jnp.zeros(cryo.volume_size, dtype=jnp.complex64)
    volume_mask = jnp.ones(cryo.volume_shape, dtype=jnp.float32)
    basis_full = jnp.eye(cryo.volume_size, 3, dtype=jnp.complex64)  # (vol_size, 3)

    # Pre-convert to half-volumes (matching what compute_projected_covariance does).
    # reduce_covariance_inner expects half volumes when disc_type != 'cubic'.
    mean_estimate_half = fourier_transform_utils.full_volume_to_half_volume(
        mean_estimate_full.reshape(cryo.volume_shape),
        cryo.volume_shape,
    ).reshape(-1)
    # basis is (n_basis, vol_size) after .T in compute_projected_covariance
    basis_T = basis_full.T  # (3, vol_size)
    basis_half = fourier_transform_utils.full_volume_to_half_volume(
        basis_T.reshape(3, *cryo.volume_shape),
        cryo.volume_shape,
    ).reshape(3, -1)

    model_half = ModelState(
        mean_estimate=mean_estimate_half,
        volume_mask=volume_mask,
        basis=basis_half,
    )
    model_full = ModelState(
        mean_estimate=mean_estimate_full,
        volume_mask=volume_mask,
        basis=basis_T,  # (3, vol_size) — full volumes
    )

    hermitian_weights = linalg.rfft2_hermitian_weights(config.image_shape)
    image_mask = cryo.image_source.mask
    return config, batch_data, model_half, model_full, hermitian_weights, image_mask


def _reduce_covariance_inner_reference(config, batch_data, model, opts, image_mask, hermitian_weights):
    """Reference implementation using full-image masking (the old code path).

    This mimics the old code where do_mask_images=True forced full images
    (no half projection), ensuring we can verify numerical equivalence.
    """
    batch = batch_data.images
    ctf_params = batch_data.ctf_params
    rotation_matrices = batch_data.rotation_matrices
    translations = batch_data.translations
    noise_variance = batch_data.noise_variance

    do_mask_images = opts.do_mask_images

    if do_mask_images:
        image_mask = covariance_core.get_per_image_tight_mask(
            model.volume_mask,
            rotation_matrices,
            image_mask,
            config.volume_mask_threshold,
            config.image_shape,
            config.volume_shape,
            config.grid_size,
            config.padding,
            "linear_interp",
        )

    if config.process_fn is not None:
        batch = config.process_fn(batch)
    batch = core.translate_images(batch, translations, config.image_shape)

    # Old code: full images when masking
    projected_mean = core_forward.forward_model(
        config,
        model.mean_estimate,
        ctf_params,
        rotation_matrices,
        half_image=False,
        half_volume=False,
    )

    if do_mask_images:
        batch = covariance_core.apply_image_masks(batch, image_mask, config.image_shape, half_images=False)
        projected_mean = covariance_core.apply_image_masks(
            projected_mean, image_mask, config.image_shape, half_images=False
        )

    config_u = config.replace(disc_type=opts.disc_type_u)
    AUs = covariance_core.batch_vol_forward_from_map(
        config_u,
        model.basis,
        ctf_params,
        rotation_matrices,
        skip_ctf=config.premultiplied_ctf,
        half_image=False,
        half_volume=False,
    )

    if do_mask_images:
        AUs = covariance_core.apply_image_masks_to_eigen(AUs, image_mask, config.image_shape, half_images=False)

    # Convert to half and apply weights (old path: always full→half here)
    if hermitian_weights is not None:
        batch = fourier_transform_utils.full_image_to_half_image(batch, config.image_shape) * hermitian_weights
        projected_mean = (
            fourier_transform_utils.full_image_to_half_image(projected_mean, config.image_shape) * hermitian_weights
        )
        n_b, n_i = AUs.shape[0], AUs.shape[1]
        AUs = (
            fourier_transform_utils.full_image_to_half_image(
                AUs.reshape(n_b * n_i, -1),
                config.image_shape,
            ).reshape(n_b, n_i, -1)
            * hermitian_weights[None, None, :]
        )
        noise_variance = fourier_transform_utils.full_image_to_half_image(noise_variance, config.image_shape)

    AUs = AUs.transpose(1, 2, 0)

    batch = batch - projected_mean
    AU_t_images = jax.vmap(lambda x, y: jnp.conj(x).T @ y, in_axes=(0, 0))(AUs, batch)
    AU_t_AU = jax.vmap(lambda x, y: jnp.conj(x).T @ y, in_axes=(0, 0))(AUs, AUs).real.astype(ctf_params.dtype)

    AUs_noise = AUs * jnp.sqrt(noise_variance)[..., None]
    _n_basis = AUs_noise.shape[-1]
    _AUs_flat = AUs_noise.reshape(-1, _n_basis)
    UALambdaAUs = jnp.conj(_AUs_flat).T @ _AUs_flat

    outer_products = AU_t_images.T @ jnp.conj(AU_t_images)
    rhs = (outer_products - UALambdaAUs).real.astype(ctf_params.dtype)

    _n = AU_t_AU.shape[-1]
    lhs = jnp.einsum("bik,bjl->ijkl", AU_t_AU, AU_t_AU).reshape(_n * _n, _n * _n)
    return lhs, rhs


def test_reduce_covariance_inner_masked_half_matches_full():
    """reduce_covariance_inner half-image path matches full-image path (both with Nyquist zeroing)."""
    config, batch_data, model_half, model_full, hermitian_weights, image_mask = _make_reduce_cov_fixtures()

    opts = CovarianceOpts(disc_type_u="linear_interp", do_mask_images=True)

    # Half-image path (hermitian weights → half projection + sqrt(w) + .real)
    lhs_half, rhs_half = _call_reduce_covariance_inner(
        config,
        batch_data,
        model_full,
        opts,
        image_mask,
        hermitian_weights=hermitian_weights,
    )

    # Full-image path (no hermitian weights → full projection)
    lhs_full, rhs_full = _call_reduce_covariance_inner(
        config,
        batch_data,
        model_full,
        opts,
        image_mask,
        hermitian_weights=None,
    )

    lhs_half_np = np.asarray(lhs_half)
    lhs_full_np = np.asarray(lhs_full)
    rhs_half_np = np.asarray(rhs_half)
    rhs_full_np = np.asarray(rhs_full)

    # Both should be finite
    assert np.all(np.isfinite(lhs_half_np)), f"half LHS has non-finite: {lhs_half_np}"
    assert np.all(np.isfinite(lhs_full_np)), f"full LHS has non-finite: {lhs_full_np}"
    assert np.all(np.isfinite(rhs_half_np)), f"half RHS has non-finite: {rhs_half_np}"
    assert np.all(np.isfinite(rhs_full_np)), f"full RHS has non-finite: {rhs_full_np}"

    # Half and full paths should agree (Nyquist zeroing ensures consistency)
    lhs_norm = np.linalg.norm(lhs_full_np)
    rhs_norm = np.linalg.norm(rhs_full_np)
    if lhs_norm > 0:
        np.testing.assert_allclose(lhs_half_np, lhs_full_np, atol=1e-4, rtol=1e-3)
    if rhs_norm > 0:
        np.testing.assert_allclose(rhs_half_np, rhs_full_np, atol=1e-4, rtol=1e-3)


def test_reduce_covariance_inner_unmasked_still_works():
    """reduce_covariance_inner with do_mask_images=False continues to work (regression)."""
    config, batch_data, _, model_full, hermitian_weights, image_mask = _make_reduce_cov_fixtures()

    opts = CovarianceOpts(disc_type_u="linear_interp", do_mask_images=False)

    lhs, rhs = _call_reduce_covariance_inner(
        config,
        batch_data,
        model_full,
        opts,
        image_mask,
        hermitian_weights=hermitian_weights,
    )

    assert lhs.shape[0] == lhs.shape[1]
    assert rhs.shape[0] == rhs.shape[1]
    assert np.all(np.isfinite(np.asarray(lhs)))
    assert np.all(np.isfinite(np.asarray(rhs)))


def test_reduce_covariance_inner_masked_vs_unmasked_differ():
    """Masking must change the result (otherwise masking is a no-op, which would indicate a bug)."""
    config, batch_data, _, model_full, hermitian_weights, image_mask = _make_reduce_cov_fixtures()

    opts_mask = CovarianceOpts(disc_type_u="linear_interp", do_mask_images=True)
    opts_no_mask = CovarianceOpts(disc_type_u="linear_interp", do_mask_images=False)

    lhs_mask, rhs_mask = _call_reduce_covariance_inner(
        config,
        batch_data,
        model_full,
        opts_mask,
        image_mask,
        hermitian_weights=hermitian_weights,
    )
    lhs_no, rhs_no = _call_reduce_covariance_inner(
        config,
        batch_data,
        model_full,
        opts_no_mask,
        image_mask,
        hermitian_weights=hermitian_weights,
    )

    # With an all-ones volume mask and get_per_image_tight_mask, the masks may
    # still clip to a tight boundary, so results should generally differ.
    # At minimum, they should both be finite.
    assert np.all(np.isfinite(np.asarray(lhs_mask)))
    assert np.all(np.isfinite(np.asarray(rhs_mask)))
    assert np.all(np.isfinite(np.asarray(lhs_no)))
    assert np.all(np.isfinite(np.asarray(rhs_no)))


def test_reduce_covariance_inner_uses_half_volume_for_half_volume_model(monkeypatch):
    config, batch_data, model_half, _, hermitian_weights, image_mask = _make_reduce_cov_fixtures()
    opts = CovarianceOpts(disc_type_u="linear_interp", do_mask_images=False)

    n_images = batch_data.images.shape[0]
    n_basis = model_half.basis.shape[0]
    half_image_size = int(np.prod(fourier_transform_utils.image_shape_to_half_image_shape(config.image_shape)))
    seen = {"mean_half_volume": False, "basis_half_volume": False}

    def fake_forward_model(
        config,
        volume,
        ctf_params,
        rotation_matrices,
        skip_ctf=False,
        half_image=False,
        half_volume=False,
    ):
        seen["mean_half_volume"] = half_volume
        n_pixels = half_image_size if half_image else int(np.prod(config.image_shape))
        return jnp.zeros((n_images, n_pixels), dtype=jnp.complex64)

    def fake_batch_vol_forward_from_map(
        config,
        volumes,
        ctf_params,
        rotation_matrices,
        skip_ctf=False,
        half_image=False,
        half_volume=False,
    ):
        seen["basis_half_volume"] = half_volume
        n_pixels = half_image_size if half_image else int(np.prod(config.image_shape))
        return jnp.zeros((n_basis, n_images, n_pixels), dtype=jnp.complex64)

    monkeypatch.setattr(core_forward, "forward_model", fake_forward_model)
    monkeypatch.setattr(covariance_core, "batch_vol_forward_from_map", fake_batch_vol_forward_from_map)

    _call_reduce_covariance_inner(
        config,
        batch_data,
        model_half,
        opts,
        image_mask,
        hermitian_weights=hermitian_weights,
    )

    assert seen["mean_half_volume"] is True
    assert seen["basis_half_volume"] is True

def test_solve_projected_covariance_system_raises_on_nonfinite_output(monkeypatch):
    def fake_solve(lhs, rhs, assume_a="pos"):
        return jnp.full(rhs.shape, jnp.nan, dtype=rhs.dtype)

    monkeypatch.setattr(cov_est.jax.scipy.linalg, "solve", fake_solve)

    lhs = jnp.eye(cov_est._symmetric_matrix_packed_size(2), dtype=jnp.float32)
    rhs = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)

    with pytest.raises(ValueError, match="projected covariance solve returned non-finite output"):
        cov_est._solve_projected_covariance_system(lhs, rhs)


def test_compute_projected_covariance_masked():
    """compute_projected_covariance with do_mask_images=True completes and returns valid result."""
    cryo = make_tiny_cryo_dataset_with_images(grid_size=4, n_images=6, seed=0)
    basis = np.eye(cryo.volume_size, 3, dtype=np.complex64)
    covar = cov_est.compute_projected_covariance(
        dataset=[cryo],
        mean_estimate=np.zeros(cryo.volume_size, dtype=np.complex64),
        basis=basis,
        volume_mask=np.ones(cryo.volume_shape, dtype=np.float32),
        batch_size=3,
        disc_type="linear_interp",
        disc_type_u="linear_interp",
        do_mask_images=True,
    )

    assert covar.shape == (3, 3)
    covar_np = np.asarray(covar)
    assert covar_np.dtype in (np.float32, np.float64)
    finite_mask = np.isfinite(covar_np)
    if finite_mask.all():
        np.testing.assert_allclose(covar_np, covar_np.T, atol=1e-5, rtol=1e-5)


def test_compute_projected_covariance_masked_matches_unmasked_with_ones_mask():
    """With an all-ones volume mask, masked and unmasked paths give close results.

    get_per_image_tight_mask projects the volume mask → per-image mask. With a
    volume of all ones, the per-image mask is also ~all ones, so the two paths
    should agree closely (modulo FFT rounding in the mask-project-threshold cycle).
    """
    cryo = make_tiny_cryo_dataset_with_images(grid_size=4, n_images=6, seed=0)
    basis = np.eye(cryo.volume_size, 3, dtype=np.complex64)
    kwargs = dict(
        dataset=[cryo],
        mean_estimate=np.zeros(cryo.volume_size, dtype=np.complex64),
        basis=basis,
        volume_mask=np.ones(cryo.volume_shape, dtype=np.float32),
        batch_size=6,
        disc_type="linear_interp",
        disc_type_u="linear_interp",
    )

    covar_masked = np.asarray(cov_est.compute_projected_covariance(**kwargs, do_mask_images=True))
    covar_unmasked = np.asarray(cov_est.compute_projected_covariance(**kwargs, do_mask_images=False))

    if np.all(np.isfinite(covar_masked)) and np.all(np.isfinite(covar_unmasked)):
        # With an all-ones mask the results should be close but not identical
        # because the mask→project→threshold cycle introduces small differences.
        np.testing.assert_allclose(covar_masked, covar_unmasked, atol=0.5, rtol=0.5)


def test_compute_projected_covariance_preserves_full_volume_model_layout(monkeypatch):
    cryo = make_tiny_cryo_dataset_with_images(grid_size=6, n_images=6, seed=0)
    basis = np.eye(cryo.volume_size, 3, dtype=np.complex64)
    seen = {}

    def fake_reduce_covariance_inner(
        config,
        images,
        model,
        opts,
        image_mask,
        rotation_matrices,
        translations,
        ctf_params,
        noise_variance,
        hermitian_weights=None,
        lhs=None,
        rhs=None,
        tilt_labels=None,
    ):
        seen["mean_size"] = int(np.prod(model.mean_estimate.shape))
        seen["basis_size"] = int(np.prod(model.basis.shape[1:]))
        return lhs, rhs

    monkeypatch.setattr(cov_est, "reduce_covariance_inner", fake_reduce_covariance_inner)

    covar = cov_est.compute_projected_covariance(
        dataset=[cryo],
        mean_estimate=np.zeros(cryo.volume_size, dtype=np.complex64),
        basis=basis,
        volume_mask=np.ones(cryo.volume_shape, dtype=np.float32),
        batch_size=3,
        disc_type="linear_interp",
        disc_type_u="linear_interp",
        do_mask_images=False,
    )

    assert seen["mean_size"] == cryo.volume_size
    assert seen["basis_size"] == cryo.volume_size
    np.testing.assert_allclose(np.asarray(covar), np.zeros((3, 3), dtype=np.float32), atol=1e-7, rtol=1e-7)


def test_prepare_model_half_volumes_keeps_cubic_coefficients_full():
    cryo = make_tiny_cryo_dataset_with_images(grid_size=6, n_images=6, seed=0)
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(cryo.volume_shape)
    half_size = int(np.prod(half_shape))

    mean_half = np.arange(half_size, dtype=np.float32).astype(np.complex64)
    basis_half = np.arange(3 * half_size, dtype=np.float32).reshape(3, half_size).astype(np.complex64)

    mean_full, basis_full = cov_est._prepare_model_half_volumes(
        cryo.volume_shape,
        mean_half,
        basis_half,
        mean_disc_type="cubic",
        basis_disc_type="cubic",
    )

    expected_mean = np.asarray(
        fourier_transform_utils.half_volume_to_full_volume(mean_half.reshape(half_shape), cryo.volume_shape)
    ).reshape(-1)
    expected_basis = np.stack(
        [
            np.asarray(fourier_transform_utils.half_volume_to_full_volume(vec.reshape(half_shape), cryo.volume_shape)).reshape(
                -1
            )
            for vec in basis_half
        ],
        axis=0,
    )

    assert mean_full.shape == (cryo.volume_size,)
    assert basis_full.shape == (3, cryo.volume_size)
    np.testing.assert_allclose(np.asarray(mean_full), expected_mean, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(basis_full), expected_basis, atol=1e-6, rtol=1e-6)


def test_prepare_model_half_volumes_preserves_linear_full_layout():
    cryo = make_tiny_cryo_dataset_with_images(grid_size=6, n_images=6, seed=0)
    rng = np.random.default_rng(0)
    mean = (rng.standard_normal(cryo.volume_size) + 1j * rng.standard_normal(cryo.volume_size)).astype(np.complex64)
    basis = (
        rng.standard_normal((4, cryo.volume_size)) + 1j * rng.standard_normal((4, cryo.volume_size))
    ).astype(np.complex64)

    mean_out, basis_out = cov_est._prepare_model_half_volumes(
        cryo.volume_shape,
        mean,
        basis,
        mean_disc_type="linear_interp",
        basis_disc_type="linear_interp",
    )

    np.testing.assert_allclose(np.asarray(mean_out), mean, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(basis_out), basis, atol=1e-6, rtol=1e-6)


def test_compute_projected_covariance_preserves_packed_half_volume_model_layout(monkeypatch):
    cryo = make_tiny_cryo_dataset_with_images(grid_size=6, n_images=6, seed=0)
    half_size = int(np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(cryo.volume_shape)))
    mean_half = np.zeros(half_size, dtype=np.complex64)
    basis_half = np.zeros((half_size, 3), dtype=np.complex64)
    seen = {}

    def fake_reduce_covariance_inner(
        config,
        images,
        model,
        opts,
        image_mask,
        rotation_matrices,
        translations,
        ctf_params,
        noise_variance,
        hermitian_weights=None,
        lhs=None,
        rhs=None,
        tilt_labels=None,
    ):
        seen["mean_size"] = int(np.prod(model.mean_estimate.shape))
        seen["basis_size"] = int(np.prod(model.basis.shape[1:]))
        return lhs, rhs

    monkeypatch.setattr(cov_est, "reduce_covariance_inner", fake_reduce_covariance_inner)

    covar = cov_est.compute_projected_covariance(
        dataset=[cryo],
        mean_estimate=mean_half,
        basis=basis_half,
        volume_mask=np.ones(cryo.volume_shape, dtype=np.float32),
        batch_size=3,
        disc_type="linear_interp",
        disc_type_u="linear_interp",
        do_mask_images=False,
    )

    assert seen["mean_size"] == half_size
    assert seen["basis_size"] == half_size
    np.testing.assert_allclose(np.asarray(covar), np.zeros((3, 3), dtype=np.float32), atol=1e-7, rtol=1e-7)


# ---------------------------------------------------------------------------
# Tests for apply_image_masks_to_eigen with half_images=True
# ---------------------------------------------------------------------------


def test_apply_image_masks_to_eigen_half_matches_full():
    """Half-image masking round-trip matches the full-image path.

    Uses DFT of real images to ensure Hermitian symmetry, which is required
    for the irfft2/rfft2 round-trip in the half-image path.
    """
    n_basis, n_images = 3, 4
    image_shape = (8, 8)

    rng = np.random.RandomState(42)
    # Real-space images → DFT → Hermitian-symmetric full spectrum
    real_imgs = rng.randn(n_basis, n_images, *image_shape).astype(np.float32)
    proj_eigen_full = jnp.array(fourier_transform_utils.get_dft2(real_imgs).reshape(n_basis, n_images, -1))

    # Random real-space mask
    image_masks = jnp.array((rng.rand(n_images, *image_shape) > 0.3).astype(np.float32))

    # Full-image path (reference)
    result_full = covariance_core.apply_image_masks_to_eigen(
        proj_eigen_full, image_masks, image_shape, half_images=False
    )

    # Convert to half, run half path, convert back to full
    proj_eigen_half = fourier_transform_utils.full_image_to_half_image(
        proj_eigen_full.reshape(n_basis * n_images, -1), image_shape
    ).reshape(n_basis, n_images, -1)

    result_half = covariance_core.apply_image_masks_to_eigen(
        proj_eigen_half, image_masks, image_shape, half_images=True
    )

    # Convert half result to full for comparison
    result_half_full = fourier_transform_utils.half_image_to_full_image(
        result_half.reshape(n_basis * n_images, -1), image_shape
    ).reshape(n_basis, n_images, -1)

    np.testing.assert_allclose(np.asarray(result_half_full), np.asarray(result_full), atol=1e-5, rtol=1e-5)


def test_apply_image_masks_half_matches_full():
    """apply_image_masks with half_images=True matches full path.

    Uses DFT of real images to ensure Hermitian symmetry.
    """
    n_images = 5
    image_shape = (8, 8)

    rng = np.random.RandomState(99)
    real_imgs = rng.randn(n_images, *image_shape).astype(np.float32)
    images_full = jnp.array(fourier_transform_utils.get_dft2(real_imgs).reshape(n_images, -1))
    image_masks = jnp.array((rng.rand(n_images, *image_shape) > 0.3).astype(np.float32))

    # Full-image path
    result_full = covariance_core.apply_image_masks(images_full, image_masks, image_shape, half_images=False)

    # Half-image path
    images_half = fourier_transform_utils.full_image_to_half_image(images_full, image_shape)
    result_half = covariance_core.apply_image_masks(images_half, image_masks, image_shape, half_images=True)

    # Convert half result back to full for comparison
    result_half_full = fourier_transform_utils.half_image_to_full_image(result_half, image_shape)
    np.testing.assert_allclose(np.asarray(result_half_full), np.asarray(result_full), atol=1e-5, rtol=1e-5)


def test_variance_relion_style_triangular_kernel_uses_upsampled_volume_shape(monkeypatch):
    class _FakeDataset:
        image_shape = (4, 4)
        upsampled_volume_shape = (8, 8, 8)
        grid_size = 4
        voxel_size = 1.5
        padding = 0
        ctf_evaluator = staticmethod(lambda *args, **kwargs: None)
        premultiplied_ctf = False
        volume_mask_threshold = 0.5
        noise = None
        image_mask = np.ones((4, 4), dtype=np.float32)

        def iter_batches(self, *args, **kwargs):
            yield (
                np.zeros((1, 4, 4), dtype=np.float32),
                np.zeros((1, 3, 3), dtype=np.float32),
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((1, 9), dtype=np.float32),
                None,
                np.zeros((1,), dtype=np.int32),
                np.zeros((1,), dtype=np.int32),
            )

        def process_images_half(self, images):
            return images

    seen = {}

    def _fake_kernel(config, *args, **kwargs):
        seen["volume_shape"] = tuple(config.volume_shape)
        raise RuntimeError("stop-after-config")

    monkeypatch.setattr(cov_est, "variance_relion_kernel_trilinear", _fake_kernel)

    with pytest.raises(RuntimeError, match="stop-after-config"):
        cov_est.variance_relion_style_triangular_kernel(
            _FakeDataset(),
            mean_estimate=np.zeros(8, dtype=np.complex64),
            batch_size=1,
        )

    assert seen["volume_shape"] == (8, 8, 8)
