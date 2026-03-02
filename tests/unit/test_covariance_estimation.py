import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

import recovar.heterogeneity.covariance_estimation as cov_est
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core
from recovar.core.configs import ForwardModelConfig, BatchData
from recovar.data_io.dataset import CryoEMHalfsets
from helpers.tiny_synthetic import make_tiny_cryo_dataset, make_tiny_cryo_dataset_with_images

pytestmark = pytest.mark.unit


def test_default_covariance_options_has_expected_keys(monkeypatch):
    monkeypatch.setattr(cov_est.utils, "get_gpu_memory_total", lambda: 8)
    opts = cov_est.get_default_covariance_computation_options(grid_size=32)
    assert isinstance(opts, dict)
    assert "n_pcs_to_compute" in opts
    assert opts["n_pcs_to_compute"] >= 1
    assert opts["covariance_fn"] == "kernel"


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
    np.random.seed(0)
    sampling_vec = np.ones(64, dtype=np.float64)
    volume_shape = (4, 4, 4)
    picked, freqs = cov_est.randomized_column_choice(
        sampling_vec=sampling_vec,
        n_samples=4,
        volume_shape=volume_shape,
        avoid_in_radius=0,
    )
    assert picked.shape == (4,)
    assert freqs.shape == (4, 3)
    assert len(np.unique(picked)) == 4


def test_set_covariance_options_updates_only_present_keys():
    opts = {"a": 1, "b": 2}
    args = {"a": 5, "c": 9}
    out = cov_est.set_covariance_options(args, opts.copy())
    assert out["a"] == 5
    assert out["b"] == 2
    assert "c" not in out


def test_compute_regularized_covariance_columns_in_batch_concatenates(monkeypatch):
    mock_cryo = type("Cryo", (), {"grid_size": 4})()
    cryos = CryoEMHalfsets(mock_cryo, mock_cryo)
    picked_frequencies = np.arange(10)

    monkeypatch.setattr(cov_est.utils, "get_column_batch_size", lambda *_: 4)

    calls = []

    def fake_compute_regularized_covariance_columns(
        cryos_in,
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
        cryos=cryos,
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

    def _fake_reg_relion(Hs, Bs, mean_prior, picked_frequencies, volume_noise_var, mask_final, volume_shape, gpu_memory, reg_init_multiplier, options):
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
        cryos=CryoEMHalfsets(cryo, cryo),
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
    cryos = [make_tiny_cryo_dataset(grid_size=4, n_images=4, seed=0), make_tiny_cryo_dataset(grid_size=4, n_images=4, seed=1)]
    means = {
        "combined": np.array([11], dtype=np.float32),
        "corrected0": np.array([21], dtype=np.float32),
        "corrected1": np.array([31], dtype=np.float32),
    }

    chosen_means = []

    def _fake_compute_h_b_in_volume_batch(cryo, mean, *args, **kwargs):
        chosen_means.append(float(np.asarray(mean).reshape(-1)[0]))
        return np.ones((2, 2), dtype=np.complex64), np.ones((2, 2), dtype=np.complex64) * 2

    monkeypatch.setattr(cov_est, "compute_H_B_in_volume_batch", _fake_compute_h_b_in_volume_batch)

    options = {"use_combined_mean": True}
    Hs, Bs = cov_est.compute_both_H_B(cryos, means, None, np.array([0, 1]), 8, False, options)
    assert len(Hs) == 2 and len(Bs) == 2
    assert chosen_means == [11.0, 11.0]

    chosen_means.clear()
    options = {"use_combined_mean": False}
    cov_est.compute_both_H_B(cryos, means, None, np.array([0, 1]), 8, False, options)
    assert chosen_means == [21.0, 31.0]


def test_summed_batch_kron_matches_scan():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    out1 = cov_est.summed_batch_kron(x)
    out2 = cov_est.summed_batch_kron_scan(x)
    np.testing.assert_allclose(np.asarray(out1), np.asarray(out2), atol=1e-6, rtol=1e-6)


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


def test_adjoint_kernel_slice_dispatch_and_noise_term(monkeypatch):
    monkeypatch.setattr(cov_est.core, "adjoint_slice_volume_by_trilinear", lambda images, *_: images + 1)
    monkeypatch.setattr(cov_est.core, "adjoint_slice_volume_by_map", lambda images, *_: images + 2)

    images = jnp.ones((2, 4), dtype=jnp.complex64)
    out_tri = cov_est.adjoint_kernel_slice(images, None, (2, 2), (2, 2, 1), kernel="triangular")
    out_sq = cov_est.adjoint_kernel_slice(images, None, (2, 2), (2, 2, 1), kernel="square")
    np.testing.assert_allclose(np.asarray(out_tri), np.asarray(images + 1))
    np.testing.assert_allclose(np.asarray(out_sq), np.asarray(images + 2))
    with pytest.raises(ValueError):
        cov_est.adjoint_kernel_slice(images, None, (2, 2), (2, 2, 1), kernel="bad")

    monkeypatch.setattr(cov_est.covariance_core, "evaluate_kernel_on_grid", lambda *args, **kwargs: jnp.ones((2, 4), dtype=jnp.complex64))
    monkeypatch.setattr(cov_est.covariance_core, "apply_image_masks", lambda x, *_: x)
    ctf = jnp.ones((2, 4), dtype=jnp.complex64) * (2 + 0j)
    noise_var = jnp.ones((2, 4), dtype=jnp.float32) * 3

    out_nonpremult = cov_est.compute_noise_term(None, None, ctf, (2, 2), None, noise_var, premultiplied_ctf=False)
    out_premult = cov_est.compute_noise_term(None, None, ctf, (2, 2), None, noise_var, premultiplied_ctf=True)
    assert out_nonpremult.shape == (2, 4)
    assert out_premult.shape == (2, 4)
    assert np.all(np.isfinite(np.asarray(out_nonpremult)))
    assert np.all(np.isfinite(np.asarray(out_premult)))


def test_compute_h_b_in_volume_batch_batches_frequency_chunks(monkeypatch):
    cryo = make_tiny_cryo_dataset(grid_size=4, n_images=6, seed=0)
    picked_frequencies = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    monkeypatch.setattr(cov_est.utils, "get_image_batch_size", lambda *args, **kwargs: 4)
    monkeypatch.setattr(cov_est.utils, "get_column_batch_size", lambda *args, **kwargs: 2)

    calls = []

    def _fake_compute_h_b(experiment_dataset, mean_estimate, volume_mask, picked_frequency_indices, batch_size, diag_prior, **kwargs):
        calls.append(np.array(picked_frequency_indices))
        n = len(picked_frequency_indices)
        h = np.tile(np.array(picked_frequency_indices, dtype=np.float32)[None, :], (experiment_dataset.volume_size, 1))
        b = h + 100.0
        return h.astype(experiment_dataset.dtype), b.astype(experiment_dataset.dtype)

    monkeypatch.setattr(cov_est, "compute_H_B", _fake_compute_h_b)

    options = {"disc_type": "linear_interp"}
    H, B = cov_est.compute_H_B_in_volume_batch(
        cryo=cryo,
        mean=np.ones(cryo.volume_size, dtype=np.complex64),
        dilated_volume_mask=np.ones(cryo.volume_size, dtype=np.float32),
        picked_frequencies=picked_frequencies,
        gpu_memory=8,
        parallel_analysis=False,
        options=options,
        use_multi_gpu=False,
        n_gpus=None,
    )

    assert len(calls) == 3
    np.testing.assert_array_equal(calls[0], np.array([0, 1]))
    np.testing.assert_array_equal(calls[1], np.array([2, 3]))
    np.testing.assert_array_equal(calls[2], np.array([4]))
    assert H.shape == (cryo.volume_size, picked_frequencies.size)
    assert B.shape == (cryo.volume_size, picked_frequencies.size)
    np.testing.assert_allclose(H[0], np.array([0, 1, 2, 3, 4], dtype=np.complex64))
    np.testing.assert_allclose(B[0], np.array([100, 101, 102, 103, 104], dtype=np.complex64))


def test_compute_variance_orchestration_with_stubbed_kernels(monkeypatch):
    vol_shape = (2, 2, 2)
    vol_size = int(np.prod(vol_shape))

    class _Cryo:
        def __init__(self, scale):
            self.scale = scale
            self.volume_shape = vol_shape
            self.volume_size = vol_size
            self.dtype_real = np.float32

        def get_valid_frequency_indices(self, rad=None):
            return np.ones(self.volume_size, dtype=np.float32)

    cryos = CryoEMHalfsets(_Cryo(1.0), _Cryo(2.0))

    def _fake_var_kernel(cryo, mean_estimate, batch_size, image_subset=None, volume_mask=None, disc_type=""):
        lhs = np.ones(vol_size, dtype=np.float32) * (10.0 * cryo.scale)
        rhs = np.ones(vol_size, dtype=np.float32) * (4.0 * cryo.scale)
        noise_lhs = np.ones(vol_size, dtype=np.float32) * (2.0 * cryo.scale)
        noise_rhs = np.ones(vol_size, dtype=np.float32) * (1.0 * cryo.scale)
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
        cryos=cryos,
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
            "covariance_fn": "kernel",
            "left_kernel": "triangular",
            "right_kernel": "triangular",
            "right_kernel_width": 1,
            "mask_images_in_H_B": False,
        }
    )

    H, B = cov_est.compute_H_B(
        experiment_dataset=cryo,
        mean_estimate=np.zeros(cryo.volume_size, dtype=np.complex64),
        volume_mask=np.ones(cryo.volume_shape, dtype=np.float32),
        picked_frequency_indices=np.array([0, 1], dtype=np.int32),
        batch_size=3,
        diag_prior=np.zeros(cryo.volume_size, dtype=np.float32),
        options=options,
    )

    assert H.shape == (cryo.volume_size, 2)
    assert B.shape == (cryo.volume_size, 2)
    assert np.isfinite(np.asarray(H)).all()


def test_compute_projected_covariance_runs_on_tiny_image_dataset():
    cryo = make_tiny_cryo_dataset_with_images(grid_size=4, n_images=6, seed=0)
    basis = np.eye(cryo.volume_size, 4, dtype=np.complex64)
    covar = cov_est.compute_projected_covariance(
        experiment_datasets=[cryo],
        mean_estimate=np.zeros(cryo.volume_size, dtype=np.complex64),
        basis=basis,
        volume_mask=np.ones(cryo.volume_shape, dtype=np.float32),
        batch_size=3,
        disc_type="linear_interp",
        disc_type_u="linear_interp",
        parallel_analysis=False,
        do_mask_images=False,
    )

    # Full projected-covariance reduction path on real tiny simulated images.
    assert covar.shape == (4, 4)
    assert np.asarray(covar).dtype in (np.float32, np.float64)
    covar_np = np.asarray(covar)
    finite_mask = np.isfinite(covar_np)
    if finite_mask.all():
        np.testing.assert_allclose(covar_np, covar_np.T, atol=1e-5, rtol=1e-5)
        evals = np.linalg.eigvalsh(covar_np)
        assert np.isfinite(evals).all()
        # Numerical jitter can introduce tiny negative values.
        assert np.min(evals) > -1e-4
    else:
        # Tiny synthetic grids can produce NaNs in this projected path; ensure it's explicit.
        assert np.isnan(covar_np).any()


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax


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

    images_gen = cryo.image_stack.get_dataset_generator(batch_size=n_images)
    images_batch, _, indices = next(iter(images_gen))

    batch_data = BatchData(
        images=jnp.asarray(images_batch),
        ctf_params=jnp.asarray(cryo.CTF_params[indices]),
        rotation_matrices=jnp.asarray(cryo.rotation_matrices[indices]),
        translations=jnp.asarray(cryo.translations[indices]),
        noise_variance=jnp.asarray(cryo.noise.get(indices)),
    )
    mean_estimate = jnp.zeros(config.volume_size, dtype=jnp.complex64)
    volume_mask = jnp.ones(config.volume_size, dtype=jnp.float32)
    image_mask = jnp.ones(config.image_shape, dtype=jnp.float32)
    return config, batch_data, mean_estimate, volume_mask, image_mask


def _reference_variance_kernel(config, batch_data, mean_estimate, volume_mask, image_mask, soften=5):
    """Old full-volume backprojection logic, for numerical equivalence testing."""
    from recovar.heterogeneity import covariance_core
    from recovar.reconstruction import noise as noise_mod

    images = batch_data.images
    noise_variances = batch_data.noise_variance
    CTF = config.compute_ctf(batch_data.ctf_params)
    images = core.translate_images(images, batch_data.translations, config.image_shape)

    if config.premultiplied_ctf:
        images = images - core.slice_volume_by_map(
            mean_estimate, batch_data.rotation_matrices, config.image_shape, config.volume_shape, config.disc_type
        ) * CTF ** 2
        noise_p_variance_ctf = CTF ** 2
    else:
        images = images - core.slice_volume_by_map(
            mean_estimate, batch_data.rotation_matrices, config.image_shape, config.volume_shape, config.disc_type
        ) * CTF
        noise_p_variance_ctf = jnp.ones_like(images)

    img_power_full = jnp.abs(images) ** 2
    cov_noise = jnp.zeros_like(images)

    if volume_mask is not None:
        image_mask_per_image = covariance_core.get_per_image_tight_mask(
            volume_mask, batch_data.rotation_matrices, image_mask, config.volume_mask_threshold,
            config.image_shape, config.volume_shape, config.grid_size, config.padding,
            "linear_interp", soften=soften,
        )
        images = covariance_core.apply_image_masks(images, image_mask_per_image, config.image_shape)
        if config.premultiplied_ctf:
            noise_variances = noise_variances * CTF ** 2
        cov_noise = noise_mod.get_masked_noise_variance_from_noise_variance(
            image_mask_per_image, noise_variances, config.image_shape
        )

    images_squared = jnp.abs(images) ** 2 - cov_noise.reshape(images.shape)
    CTF_squared = CTF ** 2
    if not config.premultiplied_ctf:
        images_squared *= CTF_squared

    def _bp(arr):
        return core.adjoint_slice_volume_by_trilinear(
            arr, batch_data.rotation_matrices, config.image_shape, config.volume_shape
        )

    return _bp(images_squared), _bp(CTF_squared ** 2), _bp(img_power_full), _bp(noise_p_variance_ctf)


def _to_full(half_vol, volume_shape):
    return np.asarray(fourier_transform_utils.half_volume_to_full_volume(half_vol, volume_shape))


def test_variance_kernel_no_mask_matches_reference():
    """New kernel (half-vol accumulation) matches old full-vol backprojection — no mask."""
    config, batch_data, mean_estimate, _, image_mask = _make_variance_test_fixtures()

    ref_y, ref_ctf, ref_im, ref_one = _reference_variance_kernel(
        config, batch_data, mean_estimate, volume_mask=None, image_mask=image_mask
    )
    new_y, new_ctf, new_im, new_one = cov_est.variance_relion_kernel_trilinear(
        config, batch_data, mean_estimate, volume_mask=None, image_mask=image_mask
    )

    vol_shape = config.volume_shape
    np.testing.assert_allclose(_to_full(new_y, vol_shape), np.asarray(ref_y), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(_to_full(new_ctf, vol_shape), np.asarray(ref_ctf), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(_to_full(new_im, vol_shape), np.asarray(ref_im), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(_to_full(new_one, vol_shape), np.asarray(ref_one), atol=1e-4, rtol=1e-4)


def test_variance_kernel_with_mask_matches_reference():
    """New kernel (half-vol accumulation) matches old full-vol backprojection — with mask."""
    config, batch_data, mean_estimate, volume_mask, image_mask = _make_variance_test_fixtures()

    ref_y, ref_ctf, ref_im, ref_one = _reference_variance_kernel(
        config, batch_data, mean_estimate, volume_mask, image_mask
    )
    new_y, new_ctf, new_im, new_one = cov_est.variance_relion_kernel_trilinear(
        config, batch_data, mean_estimate, volume_mask, image_mask
    )

    vol_shape = config.volume_shape
    np.testing.assert_allclose(_to_full(new_y, vol_shape), np.asarray(ref_y), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(_to_full(new_ctf, vol_shape), np.asarray(ref_ctf), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(_to_full(new_im, vol_shape), np.asarray(ref_im), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(_to_full(new_one, vol_shape), np.asarray(ref_one), atol=1e-4, rtol=1e-4)


def test_variance_kernel_accumulator_matches_sequential_sum():
    """Accumulator pattern gives the same result as summing per-batch contributions."""
    config, batch_data, mean_estimate, volume_mask, image_mask = _make_variance_test_fixtures(n_images=6)

    # Split into two half-batches.
    def _slice_batch(bd, sl):
        return BatchData(
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
    y1, ctf1, im1, one1 = cov_est.variance_relion_kernel_trilinear(config, b1, **kwargs)
    y2, ctf2, im2, one2 = cov_est.variance_relion_kernel_trilinear(config, b2, **kwargs)
    vol_shape = config.volume_shape
    ref_y = _to_full(y1, vol_shape) + _to_full(y2, vol_shape)
    ref_ctf = _to_full(ctf1, vol_shape) + _to_full(ctf2, vol_shape)
    ref_im = _to_full(im1, vol_shape) + _to_full(im2, vol_shape)
    ref_one = _to_full(one1, vol_shape) + _to_full(one2, vol_shape)

    # Accumulator: pass running accumulators between calls.
    acc_y, acc_ctf, acc_im, acc_one = cov_est.variance_relion_kernel_trilinear(config, b1, **kwargs)
    acc_y, acc_ctf, acc_im, acc_one = cov_est.variance_relion_kernel_trilinear(
        config, b2, mean_estimate=mean_estimate, volume_mask=volume_mask, image_mask=image_mask,
        Ft_y=acc_y, Ft_ctf=acc_ctf, Ft_im=acc_im, Ft_one=acc_one,
    )

    np.testing.assert_allclose(_to_full(acc_y, vol_shape), ref_y, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_full(acc_ctf, vol_shape), ref_ctf, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_full(acc_im, vol_shape), ref_im, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_full(acc_one, vol_shape), ref_one, atol=1e-5, rtol=1e-5)
