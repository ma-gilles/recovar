"""Tests for the pure-JAX 3D wavelet transform (recovar.ppca.wavelets).

Validates correctness against pywt (ground truth), roundtrip accuracy,
complex input support, JIT compatibility, and backend equivalence through
the Wavelet_multilvl / WaveletL1 integration surface.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import pywt

from recovar.ppca.wavelets import wavedec3, waverec3

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wavelet", ["db1", "db2", "db4", "db8", "sym3", "sym6", "coif2"])
def test_roundtrip_real(wavelet):
    """wavedec3 -> waverec3 reconstructs real input."""
    rng = np.random.default_rng(42)
    x = jnp.array(rng.normal(size=(2, 16, 16, 16)).astype(np.float32))
    coeffs = wavedec3(x, wavelet, mode="symmetric")
    recon = waverec3(coeffs, wavelet)
    np.testing.assert_allclose(np.asarray(recon), np.asarray(x), atol=1e-5)


@pytest.mark.parametrize("wavelet", ["db1", "db4", "sym3"])
def test_roundtrip_complex(wavelet):
    """wavedec3 -> waverec3 reconstructs complex input natively."""
    rng = np.random.default_rng(7)
    x = jnp.array((rng.normal(size=(2, 16, 16, 16)) + 1j * rng.normal(size=(2, 16, 16, 16))).astype(np.complex64))
    coeffs = wavedec3(x, wavelet, mode="symmetric")
    recon = waverec3(coeffs, wavelet)
    np.testing.assert_allclose(np.asarray(recon), np.asarray(x), atol=1e-5)


@pytest.mark.parametrize("size", [8, 16, 32])
def test_roundtrip_various_sizes(size):
    """Roundtrip works for different volume sizes."""
    rng = np.random.default_rng(99)
    x = jnp.array(rng.normal(size=(2, size, size, size)).astype(np.float32))
    coeffs = wavedec3(x, "db1", mode="symmetric")
    recon = waverec3(coeffs, "db1")
    np.testing.assert_allclose(np.asarray(recon), np.asarray(x), atol=1e-5)


def test_roundtrip_batch_of_one():
    """Works with a single volume in batch dim (batch=1)."""
    rng = np.random.default_rng(55)
    x = jnp.array(rng.normal(size=(1, 8, 8, 8)).astype(np.float32))
    coeffs = wavedec3(x, "db1", mode="symmetric")
    recon = waverec3(coeffs, "db1")
    np.testing.assert_allclose(np.asarray(recon), np.asarray(x), atol=1e-5)


# ---------------------------------------------------------------------------
# Cross-validation against pywt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wavelet", ["db1", "db2", "sym3"])
def test_coefficients_match_pywt(wavelet):
    """Our coefficients match pywt.wavedecn (ground truth)."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2, 16, 16, 16)).astype(np.float32)

    ours = wavedec3(jnp.array(x), wavelet, mode="symmetric")
    theirs = pywt.wavedecn(x, wavelet, mode="symmetric", axes=(-3, -2, -1))

    # Same number of levels
    assert len(ours) == len(theirs)

    # Compare approx
    np.testing.assert_allclose(np.asarray(ours[0]), np.asarray(theirs[0]), atol=1e-5)

    # Compare detail bands
    for i in range(1, len(ours)):
        for key in ours[i]:
            np.testing.assert_allclose(
                np.asarray(ours[i][key]),
                np.asarray(theirs[i][key]),
                atol=1e-5,
                err_msg=f"level {i} key {key}",
            )


# ---------------------------------------------------------------------------
# Coefficient format
# ---------------------------------------------------------------------------


def test_coefficient_format():
    """Output format matches jaxwt convention (list of approx + dicts)."""
    x = jnp.ones((2, 8, 8, 8))
    coeffs = wavedec3(x, "db1", mode="symmetric", level=2)
    assert len(coeffs) == 3  # approx + 2 detail levels
    assert isinstance(coeffs[0], jnp.ndarray)
    for i in range(1, len(coeffs)):
        assert isinstance(coeffs[i], dict)
        assert set(coeffs[i].keys()) == {"aad", "ada", "add", "daa", "dad", "dda", "ddd"}


def test_explicit_level():
    """level parameter controls decomposition depth."""
    x = jnp.ones((2, 32, 32, 32))
    for level in [1, 2, 3]:
        coeffs = wavedec3(x, "db1", mode="symmetric", level=level)
        assert len(coeffs) == level + 1


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


def test_jit_roundtrip():
    """wavedec3 + waverec3 work inside jax.jit."""
    rng = np.random.default_rng(77)
    x = jnp.array(rng.normal(size=(2, 16, 16, 16)).astype(np.float32))

    @jax.jit
    def roundtrip(data):
        c = wavedec3(data, "db1", mode="symmetric", level=3)
        return waverec3(c, "db1")

    recon = roundtrip(x)
    np.testing.assert_allclose(np.asarray(recon), np.asarray(x), atol=1e-5)


# ---------------------------------------------------------------------------
# Float64 precision test
# ---------------------------------------------------------------------------


def test_roundtrip_f64():
    """Float64 roundtrip achieves much tighter tolerance than float32."""
    import os

    os.environ["JAX_ENABLE_X64"] = "1"
    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(42)
    x = jnp.array(rng.normal(size=(2, 16, 16, 16)).astype(np.float64))
    coeffs = wavedec3(x, "db4", mode="symmetric")
    recon = waverec3(coeffs, "db4")
    np.testing.assert_allclose(np.asarray(recon), np.asarray(x), atol=1e-12)


# ---------------------------------------------------------------------------
# Wavelet_multilvl backend="jax" integration
# ---------------------------------------------------------------------------


def test_wavelet_multilvl_jax_backend():
    """Wavelet_multilvl with backend='jax' roundtrips correctly."""
    from recovar.ppca.sparse_PCA import Wavelet_multilvl

    rng = np.random.default_rng(42)
    volume_shape = (16, 16, 16)
    real_vols = rng.normal(size=(3, *volume_shape)).astype(np.float32)
    ft_vols = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(real_vols, axes=(-3, -2, -1)), axes=(-3, -2, -1)),
        axes=(-3, -2, -1),
    )
    ft_flat = jnp.array(ft_vols.reshape(3, -1).astype(np.complex64))

    w = Wavelet_multilvl(volume_shape, "db1", backend="jax")
    coeffs = w.to_basis(ft_flat)
    recon = w.to_image(coeffs)

    rel_err = float(jnp.max(jnp.abs(ft_flat - recon)) / jnp.max(jnp.abs(ft_flat)))
    assert rel_err < 1e-5


def test_wavelet_multilvl_jax_matches_jaxwt():
    """backend='jax' produces identical coefficients to backend='jaxwt'."""
    from recovar.ppca.sparse_PCA import Wavelet_multilvl

    rng = np.random.default_rng(7)
    volume_shape = (16, 16, 16)
    real_vols = rng.normal(size=(3, *volume_shape)).astype(np.float32)
    ft_vols = np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(real_vols, axes=(-3, -2, -1)), axes=(-3, -2, -1)),
        axes=(-3, -2, -1),
    )
    ft_flat = jnp.array(ft_vols.reshape(3, -1).astype(np.complex64))

    w_jax = Wavelet_multilvl(volume_shape, "db1", backend="jax")
    w_jaxwt = Wavelet_multilvl(volume_shape, "db1", backend="jaxwt")

    c_jax = w_jax.to_basis(ft_flat)
    c_jaxwt = w_jaxwt.to_basis(ft_flat)

    np.testing.assert_allclose(np.asarray(c_jax), np.asarray(c_jaxwt), atol=1e-5)


# ---------------------------------------------------------------------------
# WaveletL1 integration
# ---------------------------------------------------------------------------


def test_wavelet_l1_jax_backend():
    """WaveletL1 proximal operator works with backend='jax'."""
    from recovar.ppca.admm_test import WaveletL1

    rng = np.random.default_rng(42)
    volume_shape = (16, 16, 16)
    volume_size = int(np.prod(volume_shape))
    dim = (volume_size, 2)
    x = jnp.array(rng.normal(size=dim).astype(np.float32).flatten())

    wl1_jax = WaveletL1(dim, volume_shape, "db1", sigma=0.1, backend="jax")
    wl1_jaxwt = WaveletL1(dim, volume_shape, "db1", sigma=0.1, backend="jaxwt")

    r_jax = np.asarray(wl1_jax.prox(x, 1.0))
    r_jaxwt = np.asarray(wl1_jaxwt.prox(x, 1.0))

    np.testing.assert_allclose(r_jax, r_jaxwt, atol=1e-5)


def test_invalid_backend_raises():
    """Unknown backend name raises ValueError."""
    from recovar.ppca.sparse_PCA import Wavelet_multilvl

    with pytest.raises(ValueError, match="Unknown wavelet backend"):
        Wavelet_multilvl((8, 8, 8), "db1", backend="invalid")
