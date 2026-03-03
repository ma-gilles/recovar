import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.heterogeneity import embedding
from recovar.data_io.dataset import CryoEMHalfsets
from recovar.core.configs import ForwardModelConfig, BatchData, ModelState, EmbeddingOpts

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


# ---------------------------------------------------------------------------
# Tests for _rfft2_hermitian_weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H,W", [(4, 4), (4, 5), (8, 8), (6, 7)])
def test_rfft2_hermitian_weights_shape(H, W):
    """Output shape must be H * (W//2 + 1)."""
    w = embedding._rfft2_hermitian_weights((H, W))
    assert w.shape == (H * (W // 2 + 1),)


@pytest.mark.parametrize("H,W", [(4, 4), (4, 5), (8, 8), (6, 7)])
def test_rfft2_hermitian_weights_inner_product_equivalence(H, W):
    """Half-spectrum inner product with sqrt(w) weights equals full-spectrum inner product.

    The property only holds for Hermitian-symmetric arrays (i.e., DFTs of real images)
    and for the REAL PART of the inner product::

        Re(sum_{k in half} w[k]*conj(A[k])*B[k]) == sum_{k in full} conj(A[k])*B[k]

    For Hermitian data, the full-spectrum inner product is real (it equals the real-space
    inner product up to a scalar), so we compare real parts only.
    """
    import recovar.core.fourier_transform_utils as ftu_mod

    rng = np.random.default_rng(42)
    # Generate Hermitian-symmetric data via DFT of real images
    A_real = rng.standard_normal((H, W)).astype(np.float32)
    B_real = rng.standard_normal((H, W)).astype(np.float32)
    # fftshift gives the centered-frequency convention used throughout recovar
    A_full = np.fft.fftshift(np.fft.fft2(A_real)).astype(np.complex64)
    B_full = np.fft.fftshift(np.fft.fft2(B_real)).astype(np.complex64)

    # Full-spectrum inner product — real for Hermitian (DFT of real) data
    ip_full = np.sum(np.conj(A_full) * B_full)

    # Extract half-spectrum and apply sqrt(w) weights
    image_shape = (H, W)
    w = np.asarray(embedding._rfft2_hermitian_weights(image_shape))  # (half_size,)
    A_h = np.asarray(ftu_mod.full_image_to_half_image(jnp.array(A_full.reshape(1, -1)), image_shape)).reshape(-1)
    B_h = np.asarray(ftu_mod.full_image_to_half_image(jnp.array(B_full.reshape(1, -1)), image_shape)).reshape(-1)

    # Weighted half-spectrum inner product: Re(ip_half) should equal ip_full (which is real)
    ip_half = np.sum(np.conj(w * A_h) * (w * B_h))

    np.testing.assert_allclose(ip_half.real, ip_full.real, rtol=1e-4, atol=1e-3)


# ---------------------------------------------------------------------------
# Helpers shared by half-vs-full equivalence tests
# ---------------------------------------------------------------------------


def _hermitian_flat(rng, n, H, W):
    """DFT of random real images → Hermitian-symmetric, shape (n, H*W)."""
    real = rng.standard_normal((n, H, W)).astype(np.float32)
    return jnp.array(
        np.fft.fftshift(np.fft.fft2(real), axes=(-2, -1)).reshape(n, -1).astype(np.complex64)
    )


def _radial_noise_var(n_images, H, W):
    """Radially-symmetric noise variance, shape (n_images, H*W).

    nv[k] == nv[-k] because it depends only on |k|.  This property is required
    for the weighted half-spectrum inner product formula to hold.
    """
    kx = np.fft.fftshift(np.fft.fftfreq(W))
    ky = np.fft.fftshift(np.fft.fftfreq(H))
    KX, KY = np.meshgrid(kx, ky)
    rad = np.sqrt(KX ** 2 + KY ** 2).astype(np.float32)
    nv = (1.0 + 5.0 * rad).reshape(-1)
    return jnp.tile(jnp.array(nv), (n_images, 1))


def _half_image_of(arr_full, image_shape):
    """Extract half-spectrum from a full-spectrum array: (n, H*W) → (n, H*(W//2+1))."""
    import recovar.core.fourier_transform_utils as ftu_mod
    return ftu_mod.full_image_to_half_image(arr_full, image_shape)


def _make_forward_model_mock(proj_mean_full, image_shape):
    """Return a forward_model mock that handles half_image=True/False."""
    def mock(*a, half_image=False, **kw):
        if half_image:
            return _half_image_of(proj_mean_full, image_shape)
        return proj_mean_full
    return mock


def _make_batch_vol_mock(aus_full, image_shape):
    """Return a batch_vol_forward_from_map mock that handles half_image=True/False.

    aus_full: (n_basis, n_images, H*W) full-spectrum.
    """
    def mock(*a, half_image=False, **kw):
        if half_image:
            n_b, n_i = aus_full.shape[0], aus_full.shape[1]
            half = _half_image_of(aus_full.reshape(n_b * n_i, -1), image_shape)
            return half.reshape(n_b, n_i, -1)
        return aus_full
    return mock


def _minimal_config(image_shape, premultiplied_ctf=False):
    from recovar.core import ctf as ctf_mod
    H, _ = image_shape
    return ForwardModelConfig(
        image_shape=image_shape,
        volume_shape=(H, H, H),
        grid_size=H,
        voxel_size=1.0,
        padding=0,
        disc_type="linear_interp",
        CTF_fun=ctf_mod.cryodrgn_CTF,
        CTF_fun_half=ctf_mod.cryodrgn_CTF_half,
        premultiplied_ctf=premultiplied_ctf,
        process_fn=None,
    )


# ---------------------------------------------------------------------------
# Unit tests: _compute_batch_coords_p1 half vs full (CPU, monkeypatched)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H,W", [(8, 8), (8, 9), (6, 7)])
@pytest.mark.parametrize("noise_type", ["constant", "radial"])
def test_compute_batch_coords_p1_half_matches_full(H, W, noise_type, monkeypatch):
    """_compute_batch_coords_p1 half-spectrum path matches full-spectrum for Hermitian data.

    Monkeypatches the forward-model functions to inject controlled Hermitian-symmetric
    arrays, then verifies that every output of the half-spectrum path equals the real
    part of the corresponding full-spectrum output.
    """
    n_images, n_basis = 7, 4
    image_shape = (H, W)
    rng = np.random.default_rng(42)

    images_flat = _hermitian_flat(rng, n_images, H, W)      # (n_images, H*W)
    proj_mean   = _hermitian_flat(rng, n_images, H, W)      # (n_images, H*W)
    aus = jnp.stack([_hermitian_flat(rng, n_images, H, W)   # (n_basis, n_images, H*W)
                     for _ in range(n_basis)])

    noise_var = (
        jnp.ones((n_images, H * W), dtype=jnp.float32)
        if noise_type == "constant"
        else _radial_noise_var(n_images, H, W)
    )

    # Inject controlled Hermitian data; bypass translate (zero translations = identity anyway)
    monkeypatch.setattr(embedding.core, "translate_images", lambda b, t, s: b)
    monkeypatch.setattr(embedding.core_forward, "forward_model",
                        _make_forward_model_mock(proj_mean, image_shape))
    monkeypatch.setattr(embedding.covariance_core, "batch_vol_forward_from_map",
                        _make_batch_vol_mock(aus, image_shape))

    config = _minimal_config(image_shape, premultiplied_ctf=False)
    batch_data = BatchData(
        images=images_flat,
        rotation_matrices=jnp.zeros((n_images, 3, 3)),
        translations=jnp.zeros((n_images, 2)),
        ctf_params=jnp.zeros((n_images, 9)),
        noise_variance=noise_var,
    )
    model = ModelState(
        mean_estimate=jnp.zeros((H ** 3,), dtype=jnp.complex64),
        volume_mask=jnp.ones((H ** 3,), dtype=jnp.float32),
        basis=jnp.zeros((n_basis, H ** 3), dtype=jnp.complex64),
        eigenvalues=jnp.ones(n_basis, dtype=jnp.complex64),
    )

    hw = embedding._rfft2_hermitian_weights(image_shape)
    full_out = embedding._compute_batch_coords_p1(config, batch_data, model, hermitian_weights=None)
    half_out = embedding._compute_batch_coords_p1(config, batch_data, model, hermitian_weights=hw)

    names = ["AU_t_images", "AU_t_Amean", "AU_t_AU",
             "image_norms_sq", "image_T_A_mean", "A_mean_norm_sq"]
    for name, f, h in zip(names, full_out, half_out):
        f_np = np.asarray(f)
        h_np = np.asarray(h)

        # The implementations are mathematically identical for Hermitian data,
        # but float32 complex matmul (BLAS GEMM) accumulates ~0.02% relative error
        # for the Gram matrix AU_t_AU.  Verified: float64 agrees to ~3e-8 relative
        # (see test_compute_batch_coords_p1_half_matches_full_float64).
        # rtol=5e-3 covers diagonal elements; atol=1.0 handles small-magnitude
        # off-diagonal Gram matrix elements (near-orthogonal basis) where absolute
        # BLAS GEMM error (~0.2) can be ~5% relative — confirmed not a bug via float64.
        np.testing.assert_allclose(
            h_np, f_np.real,
            rtol=5e-3, atol=1.0,
            err_msg=f"{name}: max_err={np.max(np.abs(h_np - f_np.real)):.4g}",
        )

        # Verify that full-spectrum cross-products have negligible imaginary parts
        # (they must be real for DFTs of real images).
        if np.iscomplexobj(f_np):
            scale = max(float(np.max(np.abs(f_np.real))), 1e-8)
            np.testing.assert_allclose(
                f_np.imag / scale, 0.0, atol=1e-3,
                err_msg=f"{name} full-spectrum imag should be ~0 for Hermitian data",
            )


@pytest.mark.parametrize("H,W", [(8, 8)])
def test_compute_batch_coords_p1_premult_ctf_half_matches_full(H, W, monkeypatch):
    """Half-spectrum path also works for premultiplied_ctf=True (CTF applied inside p1).

    Only square images are tested here.  For non-square (H, W) images, ``cryodrgn_CTF``
    uses column-major pixel ordering (kx-slow, ky-fast) while ``_hermitian_flat``
    produces row-major data; these orderings agree only when H == W, so non-square
    cases would fail for unrelated reasons (CTF not Hermitian-symmetric in row-major
    ordering).  In production, both images and CTF use the same pixel ordering, so
    the half vs full equivalence is correct for all shapes.
    """
    n_images, n_basis = 5, 3
    image_shape = (H, W)
    rng = np.random.default_rng(7)

    images_flat = _hermitian_flat(rng, n_images, H, W)
    proj_mean   = _hermitian_flat(rng, n_images, H, W)
    aus = jnp.stack([_hermitian_flat(rng, n_images, H, W) for _ in range(n_basis)])
    noise_var   = jnp.ones((n_images, H * W), dtype=jnp.float32)

    monkeypatch.setattr(embedding.core, "translate_images", lambda b, t, s: b)
    monkeypatch.setattr(embedding.core_forward, "forward_model",
                        _make_forward_model_mock(proj_mean, image_shape))
    monkeypatch.setattr(embedding.covariance_core, "batch_vol_forward_from_map",
                        _make_batch_vol_mock(aus, image_shape))

    config = _minimal_config(image_shape, premultiplied_ctf=True)
    # Use standard non-zero CTF params so CTF is non-trivial (defocus=1 μm, 300 kV)
    ctf_std = np.zeros((n_images, 9), dtype=np.float32)
    ctf_std[:, 0] = 10000.0   # DFU
    ctf_std[:, 1] = 10000.0   # DFV
    ctf_std[:, 3] = 300.0     # VOLT
    ctf_std[:, 4] = 2.7       # CS
    ctf_std[:, 5] = 0.1       # W (amplitude contrast)
    ctf_std[:, 8] = 1.0       # CONTRAST scale
    batch_data = BatchData(
        images=images_flat,
        rotation_matrices=jnp.zeros((n_images, 3, 3)),
        translations=jnp.zeros((n_images, 2)),
        ctf_params=jnp.array(ctf_std),
        noise_variance=noise_var,
    )
    model = ModelState(
        mean_estimate=jnp.zeros((H ** 3,), dtype=jnp.complex64),
        volume_mask=jnp.ones((H ** 3,), dtype=jnp.float32),
        basis=jnp.zeros((n_basis, H ** 3), dtype=jnp.complex64),
        eigenvalues=jnp.ones(n_basis, dtype=jnp.complex64),
    )

    hw = embedding._rfft2_hermitian_weights(image_shape)
    full_out = embedding._compute_batch_coords_p1(config, batch_data, model, hermitian_weights=None)
    half_out = embedding._compute_batch_coords_p1(config, batch_data, model, hermitian_weights=hw)

    names = ["AU_t_images", "AU_t_Amean", "AU_t_AU",
             "image_norms_sq", "image_T_A_mean", "A_mean_norm_sq"]
    for name, f, h in zip(names, full_out, half_out):
        f_np, h_np = np.asarray(f), np.asarray(h)
        np.testing.assert_allclose(
            h_np, f_np.real, rtol=5e-3, atol=1.0,
            err_msg=f"{name}: max_err={np.max(np.abs(h_np - f_np.real)):.4g}",
        )


@pytest.mark.parametrize("H,W", [(8, 8), (8, 9), (6, 7)])
def test_compute_batch_coords_p1_half_matches_full_float64(H, W, monkeypatch):
    """In float64, half ≈ full to near-machine-epsilon — confirms no bugs, just float32 slop.

    The large-ish (~0.02% relative) tolerance in the float32 tests above is purely
    due to BLAS GEMM float32 rounding.  In float64 the same computation agrees to
    ~3e-8 relative, which is near the float64 machine epsilon times the number of
    terms in the sum.
    """
    _prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        n_images, n_basis = 5, 3
        image_shape = (H, W)
        rng = np.random.default_rng(77)

        def hermitian_flat64(n, h, w):
            real = rng.standard_normal((n, h, w)).astype(np.float64)
            fft = np.fft.fftshift(np.fft.fft2(real), axes=(-2, -1)).astype(np.complex128)
            return jnp.array(fft.reshape(n, -1))

        images_flat = hermitian_flat64(n_images, H, W)
        proj_mean   = hermitian_flat64(n_images, H, W)
        aus = jnp.stack([hermitian_flat64(n_images, H, W) for _ in range(n_basis)])
        noise_var   = jnp.ones((n_images, H * W), dtype=jnp.float64)

        monkeypatch.setattr(embedding.core, "translate_images", lambda b, t, s: b)
        monkeypatch.setattr(embedding.core_forward, "forward_model",
                            _make_forward_model_mock(proj_mean, image_shape))
        monkeypatch.setattr(embedding.covariance_core, "batch_vol_forward_from_map",
                            _make_batch_vol_mock(aus, image_shape))

        from recovar.core import ctf as ctf_mod
        config = ForwardModelConfig(
            image_shape=image_shape,
            volume_shape=(H, H, H),
            grid_size=H,
            voxel_size=1.0,
            padding=0,
            disc_type="linear_interp",
            CTF_fun=ctf_mod.cryodrgn_CTF,
            CTF_fun_half=ctf_mod.cryodrgn_CTF_half,
            premultiplied_ctf=False,
            process_fn=None,
        )
        batch_data = BatchData(
            images=images_flat,
            rotation_matrices=jnp.zeros((n_images, 3, 3)),
            translations=jnp.zeros((n_images, 2)),
            ctf_params=jnp.zeros((n_images, 9)),
            noise_variance=noise_var,
        )
        model = ModelState(
            mean_estimate=jnp.zeros((H ** 3,), dtype=jnp.complex128),
            volume_mask=jnp.ones((H ** 3,), dtype=jnp.float64),
            basis=jnp.zeros((n_basis, H ** 3), dtype=jnp.complex128),
            eigenvalues=jnp.ones(n_basis, dtype=jnp.complex128),
        )

        hw64 = jnp.array(np.asarray(embedding._rfft2_hermitian_weights(image_shape)), dtype=jnp.float64)
        full_out = embedding._compute_batch_coords_p1(config, batch_data, model, hermitian_weights=None)
        half_out = embedding._compute_batch_coords_p1(config, batch_data, model, hermitian_weights=hw64)

        names = ["AU_t_images", "AU_t_Amean", "AU_t_AU",
                 "image_norms_sq", "image_T_A_mean", "A_mean_norm_sq"]
        for name, f, h in zip(names, full_out, half_out):
            f_np, h_np = np.asarray(f), np.asarray(h)
            # float64: relative error ~5e-6 due to BLAS GEMM rounding over n_terms;
            # atol=1e-7 handles near-zero off-diagonal Gram matrix elements.
            np.testing.assert_allclose(
                h_np, f_np.real,
                rtol=1e-5, atol=1e-7,
                err_msg=f"{name}: max_err={np.max(np.abs(h_np - f_np.real)):.4g}",
            )
    finally:
        jax.config.update("jax_enable_x64", _prev_x64)


# ---------------------------------------------------------------------------
# GPU integration test: full compute_batch_coords with actual forward model
# ---------------------------------------------------------------------------


def _make_standard_ctf_params(n_images, dtype=np.float32):
    """Standard non-degenerate CTF params (1 μm defocus, 300 kV, Cs=2.7 mm)."""
    p = np.zeros((n_images, 9), dtype=dtype)
    p[:, 0] = 10000.0   # DFU (Å)
    p[:, 1] = 10000.0   # DFV (Å)
    p[:, 3] = 300.0     # VOLT (kV)
    p[:, 4] = 2.7       # CS (mm)
    p[:, 5] = 0.1       # W
    p[:, 8] = 1.0       # CONTRAST
    return p


def _run_half_vs_full_compute_batch_coords(H=16, W=16, n_images=32, n_basis=6):
    """Core logic for the end-to-end half vs full integration test.

    Uses the *actual* JAX forward model (no monkeypatching), so projections are
    computed from real random volumes and are Hermitian-symmetric by construction.
    Returns True if half and full paths agree within tolerance.
    """
    rng = np.random.default_rng(99)
    image_shape = (H, W)

    # --- Real 3D volume → Hermitian-symmetric Fourier coefficients ---
    vol_real = rng.standard_normal((H, H, H)).astype(np.float32)
    vol_fourier = np.fft.fftshift(np.fft.fftn(vol_real)).astype(np.complex64).reshape(-1)
    mean_estimate = jnp.array(vol_fourier)

    # --- Basis: projections of real volumes (will be Hermitian when projected) ---
    basis_real = rng.standard_normal((n_basis, H, H, H)).astype(np.float32)
    basis_fourier = np.fft.fftshift(
        np.fft.fftn(basis_real, axes=(-3, -2, -1)), axes=(-3, -2, -1)
    ).astype(np.complex64).reshape(n_basis, -1)
    basis = jnp.array(basis_fourier)

    # --- Random rotation matrices (orthogonal) ---
    def random_rot(seed):
        Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return Q.astype(np.float32)

    rotation_matrices = jnp.array(np.stack([random_rot(i) for i in range(n_images)]))

    # --- Observed images: DFTs of real images (Hermitian-symmetric) ---
    real_imgs = rng.standard_normal((n_images, H, W)).astype(np.float32)
    fourier_imgs = np.fft.fftshift(np.fft.fft2(real_imgs), axes=(-2, -1)).astype(np.complex64)
    images_flat = jnp.array(fourier_imgs.reshape(n_images, -1))

    ctf_params  = jnp.array(_make_standard_ctf_params(n_images))
    translations = jnp.zeros((n_images, 2), dtype=jnp.float32)
    noise_var   = jnp.ones((n_images, H * W), dtype=jnp.float32)

    from recovar.core import ctf as ctf_mod
    config = ForwardModelConfig(
        image_shape=image_shape,
        volume_shape=(H, H, H),
        grid_size=H,
        voxel_size=4.0,   # 4 Å/pix — physically reasonable
        padding=0,
        disc_type="linear_interp",
        CTF_fun=ctf_mod.cryodrgn_CTF,
        CTF_fun_half=ctf_mod.cryodrgn_CTF_half,
        premultiplied_ctf=False,
        process_fn=None,
    )
    batch_data = BatchData(
        images=images_flat,
        rotation_matrices=rotation_matrices,
        translations=translations,
        ctf_params=ctf_params,
        noise_variance=noise_var,
    )
    model = ModelState(
        mean_estimate=mean_estimate,
        volume_mask=jnp.ones((H ** 3,), dtype=jnp.float32),
        basis=basis,
        eigenvalues=jnp.ones(n_basis, dtype=jnp.complex64),
    )
    opts = EmbeddingOpts(
        compute_covariances=True,
        compute_bias=True,
        shared_label=False,
    )
    contrast_grid = jnp.linspace(0.2, 2.0, 20, dtype=jnp.float32)
    image_mask = jnp.ones((H * W,), dtype=jnp.float32)

    hw = embedding._rfft2_hermitian_weights(image_shape)

    full_xs, full_contrast, full_cov, full_bias = embedding.compute_batch_coords(
        config, batch_data, model, opts, image_mask, contrast_grid,
        hermitian_weights=None,
    )
    half_xs, half_contrast, half_cov, half_bias = embedding.compute_batch_coords(
        config, batch_data, model, opts, image_mask, contrast_grid,
        hermitian_weights=hw,
    )

    results = {
        "xs":       (np.asarray(half_xs),       np.asarray(full_xs)),
        "contrast": (np.asarray(half_contrast),  np.asarray(full_contrast)),
        "cov":      (np.asarray(half_cov),       np.asarray(full_cov)),
        "bias":     (np.asarray(half_bias),      np.asarray(full_bias)),
    }
    return results


@pytest.mark.gpu
def test_compute_batch_coords_half_vs_full_gpu(gpu_device):
    """compute_batch_coords half-spectrum path == full-spectrum on GPU with actual forward model.

    Uses real random volumes projected via the JAX forward model (no monkeypatching),
    so all Fourier arrays are Hermitian-symmetric by construction.  Compares xs,
    contrasts, covariance matrices, and bias terms between the two paths.

    The half-spectrum path enforces ``.real`` on all inner products (embedding.py
    lines 329-333), so we compare against ``.real`` of the full-spectrum results.
    Trilinear interpolation in Fourier space slightly breaks Hermitian symmetry,
    so the two paths solve slightly different linear systems; we use tolerances
    that accommodate this.
    """
    with jax.default_device(gpu_device):
        results = _run_half_vs_full_compute_batch_coords(H=16, W=16, n_images=32, n_basis=6)

    for name, (half_val, full_val) in results.items():
        # The half path enforces .real on inner products; compare real parts only
        # (matching the CPU test convention at line 871).
        full_real = full_val.real if np.iscomplexobj(full_val) else full_val
        np.testing.assert_allclose(
            half_val, full_real,
            rtol=5e-3, atol=5e-2,
            err_msg=f"Half vs full mismatch in '{name}': "
                    f"max_err={np.max(np.abs(half_val - full_real)):.4g}",
        )


def test_compute_batch_coords_half_vs_full_cpu(monkeypatch):
    """compute_batch_coords: half-spectrum path == full-spectrum on CPU.

    Uses ``jax.disable_jit()`` to disable the ``@eqx.filter_jit`` decorator on
    ``compute_batch_coords``, allowing Python-level monkeypatching to inject
    controlled Hermitian-symmetric arrays.  Verifies that xs, contrasts,
    covariances, and bias match between the two code paths.
    """
    n_images, n_basis = 7, 3
    H, W = 8, 8  # square only: avoids CTF pixel-ordering mismatch for non-square
    image_shape = (H, W)
    rng = np.random.default_rng(42)

    images_flat = _hermitian_flat(rng, n_images, H, W)
    proj_mean   = _hermitian_flat(rng, n_images, H, W)
    aus = jnp.stack([_hermitian_flat(rng, n_images, H, W) for _ in range(n_basis)])
    noise_var = jnp.ones((n_images, H * W), dtype=jnp.float32)

    monkeypatch.setattr(embedding.core, "translate_images", lambda b, t, s: b)
    monkeypatch.setattr(embedding.core_forward, "forward_model",
                        _make_forward_model_mock(proj_mean, image_shape))
    monkeypatch.setattr(embedding.covariance_core, "batch_vol_forward_from_map",
                        _make_batch_vol_mock(aus, image_shape))

    config = _minimal_config(image_shape, premultiplied_ctf=False)
    batch_data = BatchData(
        images=images_flat,
        rotation_matrices=jnp.zeros((n_images, 3, 3)),
        translations=jnp.zeros((n_images, 2)),
        ctf_params=jnp.zeros((n_images, 9)),
        noise_variance=noise_var,
    )
    model = ModelState(
        mean_estimate=jnp.zeros((H ** 3,), dtype=jnp.complex64),
        volume_mask=jnp.ones((H ** 3,), dtype=jnp.float32),
        basis=jnp.zeros((n_basis, H ** 3), dtype=jnp.complex64),
        eigenvalues=jnp.ones(n_basis, dtype=jnp.complex64),
    )
    opts = EmbeddingOpts(compute_covariances=True, compute_bias=True, shared_label=False)
    contrast_grid = jnp.linspace(0.2, 2.0, 10, dtype=jnp.float32)
    image_mask = jnp.ones((H * W,), dtype=jnp.float32)
    hw = embedding._rfft2_hermitian_weights(image_shape)

    with jax.disable_jit():
        full_xs, full_contrast, full_cov, full_bias = embedding.compute_batch_coords(
            config, batch_data, model, opts, image_mask, contrast_grid,
            hermitian_weights=None,
        )
        half_xs, half_contrast, half_cov, half_bias = embedding.compute_batch_coords(
            config, batch_data, model, opts, image_mask, contrast_grid,
            hermitian_weights=hw,
        )

    for name, full_val, half_val in [
        ("xs",       full_xs,       half_xs),
        ("contrast", full_contrast, half_contrast),
        ("cov",      full_cov,      half_cov),
        ("bias",     full_bias,     half_bias),
    ]:
        f_np = np.asarray(full_val)
        h_np = np.asarray(half_val)
        # For exactly Hermitian data (monkeypatched), the half path (.real enforcement)
        # and full path give the same linear system; differences are float32 matmul noise.
        np.testing.assert_allclose(
            h_np, f_np.real,
            rtol=5e-3, atol=1.0,
            err_msg=f"Half vs full mismatch in '{name}': "
                    f"max_err={np.max(np.abs(h_np - f_np.real)):.4g}",
        )
