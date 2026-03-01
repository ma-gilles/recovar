import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.heterogeneity import embedding
from recovar.data_io.dataset import CryoEMHalfsets

pytestmark = pytest.mark.unit


class _Cryo:
    def __init__(self, n_images):
        self.n_images = n_images


def test_split_weights_partitions_by_cryo_sizes():
    w = np.arange(10, dtype=np.float32)
    cryos = [_Cryo(3), _Cryo(4), _Cryo(3)]
    out = embedding.split_weights(w, cryos)
    assert len(out) == 3
    assert np.allclose(out[0], [0, 1, 2])
    assert np.allclose(out[1], [3, 4, 5, 6])
    assert np.allclose(out[2], [7, 8, 9])


def test_generate_conformation_from_reprojection_linear_combination():
    # mean: (1, vol_size), u: (vol_size, latent_dim), xs: (n_states, latent_dim)
    mean = np.array([[10.0, 20.0]], dtype=np.float32)
    u = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    xs = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)

    out = embedding.generate_conformation_from_reprojection(xs, mean, u)
    expected = np.array(
        [
            [11.0, 24.0],  # mean + [1,4]
            [9.0, 21.0],   # mean + [-1,1]
        ],
        dtype=np.float32,
    )
    assert out.shape == expected.shape
    assert np.allclose(out, expected)


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@pytest.mark.gpu
def test_generate_conformation_from_reprojection_gpu(gpu_device):
    mean = np.array([[10.0, 20.0]], dtype=np.float32)
    u = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    xs = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)

    cpu_out = np.asarray(embedding.generate_conformation_from_reprojection(xs, mean, u))

    with jax.default_device(gpu_device):
        xs_g = jax.device_put(jnp.array(xs), gpu_device)
        mean_g = jax.device_put(jnp.array(mean), gpu_device)
        u_g = jax.device_put(jnp.array(u), gpu_device)
        gpu_out = np.asarray(embedding.generate_conformation_from_reprojection(xs_g, mean_g, u_g))

    np.testing.assert_allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)


class _DummyCryo:
    def __init__(self, volume_size=4, image_size=16, n_images=3, dtype=np.complex64):
        self.volume_size = volume_size
        self.image_size = image_size
        self.n_images = n_images
        self.dtype = dtype


def test_get_per_image_embedding_clamps_batch_size_to_at_least_one(monkeypatch):
    cryo0 = _DummyCryo(volume_size=4, image_size=16, n_images=3)
    cryo1 = _DummyCryo(volume_size=4, image_size=16, n_images=2)
    mean = np.zeros((4,), dtype=np.complex64)
    u = np.zeros((4, 2), dtype=np.complex64)
    s = np.ones((2,), dtype=np.float32)
    volume_mask = np.ones((4,), dtype=np.float32)

    monkeypatch.setattr(embedding, "USE_CUBIC", False)
    monkeypatch.setattr(embedding.utils, "get_embedding_batch_size", lambda *_args, **_kwargs: 0)

    captured = {"batch_sizes": []}

    def fake_get_coords(
        experiment_dataset,
        mean_estimate,
        basis,
        eigenvalues,
        volume_mask_in,
        contrast_grid,
        batch_size,
        disc_type,
        **kwargs,
    ):
        _ = (mean_estimate, eigenvalues, volume_mask_in, contrast_grid, disc_type, kwargs)
        captured["batch_sizes"].append(batch_size)
        bsz = basis.shape[0]
        n = experiment_dataset.n_images
        xs = np.zeros((n, bsz), dtype=np.complex64)
        cov = np.zeros((n, bsz, bsz), dtype=np.complex64)
        contrasts = np.ones((n,), dtype=np.float32)
        bias = np.zeros((n, bsz, bsz), dtype=np.complex64)
        return xs, cov, contrasts, bias

    monkeypatch.setattr(embedding, "get_coords_in_basis_and_contrast_3", fake_get_coords)

    zs, cov_zs, est_contrasts, bias = embedding.get_per_image_embedding(
        mean=mean,
        u=u,
        s=s,
        basis_size=2,
        cryos=CryoEMHalfsets(cryo0, cryo1),
        volume_mask=volume_mask,
        gpu_memory=1,
        disc_type="linear_interp",
        contrast_option="none",
        to_real=True,
        compute_covariances=True,
        compute_bias=True,
    )

    assert captured["batch_sizes"] == [1, 1]
    assert zs.shape == (5, 2)
    assert cov_zs.shape == (5, 2, 2)
    assert est_contrasts.shape == (5,)
    assert bias.shape == (5, 2, 2)
    assert np.isrealobj(zs)
    assert np.isrealobj(cov_zs)
    assert np.isrealobj(bias)


def test_get_per_image_embedding_ignore_zero_frequency_overrides_volume_mask(monkeypatch):
    cryo0 = _DummyCryo(volume_size=4, image_size=16, n_images=2)
    cryo1 = _DummyCryo(volume_size=4, image_size=16, n_images=1)
    mean = np.zeros((4,), dtype=np.complex64)
    u = np.zeros((4, 1), dtype=np.complex64)
    s = np.ones((1,), dtype=np.float32)
    volume_mask = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(embedding, "USE_CUBIC", False)
    monkeypatch.setattr(embedding.utils, "get_embedding_batch_size", lambda *_args, **_kwargs: 10)

    captured = {"masks": [], "contrast_grid_size": []}

    def fake_get_coords(
        experiment_dataset,
        mean_estimate,
        basis,
        eigenvalues,
        volume_mask_in,
        contrast_grid,
        batch_size,
        disc_type,
        **kwargs,
    ):
        _ = (mean_estimate, eigenvalues, batch_size, disc_type, kwargs)
        captured["masks"].append(np.asarray(volume_mask_in))
        captured["contrast_grid_size"].append(int(np.asarray(contrast_grid).size))
        n = experiment_dataset.n_images
        bsz = basis.shape[0]
        return (
            np.zeros((n, bsz), dtype=np.complex64),
            np.zeros((n, bsz, bsz), dtype=np.complex64),
            np.ones((n,), dtype=np.float32),
            np.zeros((n, bsz, bsz), dtype=np.complex64),
        )

    monkeypatch.setattr(embedding, "get_coords_in_basis_and_contrast_3", fake_get_coords)

    embedding.get_per_image_embedding(
        mean=mean,
        u=u,
        s=s,
        basis_size=1,
        cryos=CryoEMHalfsets(cryo0, cryo1),
        volume_mask=volume_mask,
        gpu_memory=1,
        disc_type="linear_interp",
        contrast_option="none",
        ignore_zero_frequency=True,
        to_real=True,
        compute_covariances=False,
        compute_bias=False,
    )

    assert len(captured["masks"]) == 2
    np.testing.assert_allclose(captured["masks"][0], np.ones((4,), dtype=np.float32))
    np.testing.assert_allclose(captured["masks"][1], np.ones((4,), dtype=np.float32))
    assert captured["contrast_grid_size"] == [1, 1]


def test_get_per_image_embedding_supports_single_cryo_list(monkeypatch):
    cryo = _DummyCryo(volume_size=4, image_size=16, n_images=2)
    mean = np.zeros((4,), dtype=np.complex64)
    u = np.zeros((4, 1), dtype=np.complex64)
    s = np.ones((1,), dtype=np.float32)
    volume_mask = np.ones((4,), dtype=np.float32)

    monkeypatch.setattr(embedding, "USE_CUBIC", False)
    monkeypatch.setattr(embedding.utils, "get_embedding_batch_size", lambda *_args, **_kwargs: 10)
    monkeypatch.setattr(
        embedding,
        "get_coords_in_basis_and_contrast_3",
        lambda experiment_dataset, _mean_estimate, basis, _eigenvalues, _volume_mask_in, _contrast_grid, _batch_size, _disc_type, **_kwargs: (
            np.zeros((experiment_dataset.n_images, basis.shape[0]), dtype=np.complex64),
            np.zeros((experiment_dataset.n_images, basis.shape[0], basis.shape[0]), dtype=np.complex64),
            np.ones((experiment_dataset.n_images,), dtype=np.float32),
            np.zeros((experiment_dataset.n_images, basis.shape[0], basis.shape[0]), dtype=np.complex64),
        ),
    )

    zs, cov_zs, est_contrasts, bias = embedding.get_per_image_embedding(
        mean=mean,
        u=u,
        s=s,
        basis_size=1,
        cryos=CryoEMHalfsets(cryo, cryo),
        volume_mask=volume_mask,
        gpu_memory=1,
        disc_type="linear_interp",
        contrast_option="none",
        to_real=True,
        compute_covariances=True,
        compute_bias=True,
    )

    assert zs.shape == (4, 1)
    assert cov_zs.shape == (4, 1, 1)
    assert est_contrasts.shape == (4,)
    assert bias.shape == (4, 1, 1)


# ---------------------------------------------------------------------------
# Tests for solve_contrast_linear_system
# ---------------------------------------------------------------------------


def test_solve_contrast_linear_system_identity_case():
    """With identity AU_t_AU and unit eigenvalues, solution should be simple."""
    zdim = 2
    AU_t_images = jnp.array([1.0, 2.0], dtype=jnp.float32)
    AU_t_Amean = jnp.zeros(zdim, dtype=jnp.float32)
    AU_t_AU = jnp.eye(zdim, dtype=jnp.float32)
    eigenvalues = jnp.ones(zdim, dtype=jnp.float32)
    contrast = 1.0

    sol = np.asarray(embedding.solve_contrast_linear_system(
        AU_t_images, AU_t_Amean, AU_t_AU, eigenvalues, contrast
    ))
    # A = I + I = 2I, b = AU_t_images => sol = AU_t_images / 2
    np.testing.assert_allclose(sol, np.array([0.5, 1.0]), atol=1e-5)


def test_solve_contrast_linear_system_zero_contrast():
    """With zero contrast, A = diag(1/eigenvalues), b = 0, so solution should be 0."""
    zdim = 3
    AU_t_images = jnp.ones(zdim, dtype=jnp.float32)
    AU_t_Amean = jnp.zeros(zdim, dtype=jnp.float32)
    AU_t_AU = jnp.eye(zdim, dtype=jnp.float32)
    eigenvalues = jnp.ones(zdim, dtype=jnp.float32)
    contrast = 0.0

    sol = np.asarray(embedding.solve_contrast_linear_system(
        AU_t_images, AU_t_Amean, AU_t_AU, eigenvalues, contrast
    ))
    np.testing.assert_allclose(sol, np.zeros(zdim), atol=1e-5)


# ---------------------------------------------------------------------------
# Tests for compute_contrast_residual_fast_2
# ---------------------------------------------------------------------------


def test_compute_contrast_residual_fast_2_at_zero_xs():
    """With zero xs, fit residual should be image_norms_sq (per contrast)."""
    zdim = 2
    n_contrast = 1
    # xs has shape (n_contrast, zdim)
    xs = jnp.zeros((n_contrast, zdim), dtype=jnp.float32)
    AU_t_images = jnp.array([1.0, 2.0], dtype=jnp.float32)
    image_norms_sq = jnp.array(5.0, dtype=jnp.float32)
    AU_t_Amean = jnp.zeros(zdim, dtype=jnp.float32)
    Amean_norms_sq = jnp.array(0.0, dtype=jnp.float32)
    image_T_A_mean = jnp.array(0.0, dtype=jnp.float32)
    AU_t_AU = jnp.eye(zdim, dtype=jnp.float32)
    eigenvalues = jnp.ones(zdim, dtype=jnp.float32)
    contrast = jnp.array([1.0], dtype=jnp.float32)

    fit_res, prior_res = embedding.compute_contrast_residual_fast_2(
        xs, AU_t_images, image_norms_sq, AU_t_Amean, Amean_norms_sq,
        image_T_A_mean, AU_t_AU, eigenvalues, contrast
    )
    # With xs=0: p1=0, p2=0, p3=image_norms_sq, p4=0, p5=0, p6=0
    np.testing.assert_allclose(float(fit_res[0]), 5.0, atol=1e-5)
    np.testing.assert_allclose(float(prior_res[0]), 0.0, atol=1e-5)


def test_compute_contrast_residual_fast_2_prior_scales_with_eigenvalues():
    """Prior residual = xs^T (xs / eigenvalues), should scale inversely with eigenvalues."""
    zdim = 2
    n_contrast = 1
    xs = jnp.ones((n_contrast, zdim), dtype=jnp.float32)
    AU_t_images = jnp.zeros(zdim, dtype=jnp.float32)
    image_norms_sq = jnp.array(0.0, dtype=jnp.float32)
    AU_t_Amean = jnp.zeros(zdim, dtype=jnp.float32)
    Amean_norms_sq = jnp.array(0.0, dtype=jnp.float32)
    image_T_A_mean = jnp.array(0.0, dtype=jnp.float32)
    AU_t_AU = jnp.eye(zdim, dtype=jnp.float32)
    eigenvalues = jnp.array([2.0, 4.0], dtype=jnp.float32)
    contrast = jnp.array([1.0], dtype=jnp.float32)

    _, prior_res = embedding.compute_contrast_residual_fast_2(
        xs, AU_t_images, image_norms_sq, AU_t_Amean, Amean_norms_sq,
        image_T_A_mean, AU_t_AU, eigenvalues, contrast
    )
    # prior = xs^T diag(1/eigenvalues) xs = 1/2 + 1/4 = 0.75
    np.testing.assert_allclose(float(prior_res[0]), 0.75, atol=1e-5)
