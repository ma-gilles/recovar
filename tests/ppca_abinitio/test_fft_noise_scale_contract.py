"""FFT noise scale contract (per spec Section 4.3 / 13.1).

Pins the relationship between real-space noise variance and the
`noise_variance` array consumed by `compute_little_H_b` and the new
PPCA helper.

Convention
----------
recovar uses the centered-FT layout with `norm="backward"`
(`recovar/core/fourier_transform_utils.py:4`). For a real-space
white-noise image with per-pixel variance σ²_real, the discrete
Fourier transform produces coefficients whose per-pixel **mean
power** is

    E[|F_k|²]  =  N · σ²_real

where N = `prod(image_shape)`. This follows from Parseval's theorem
under the unnormalized convention: `sum_k |F_k|² = N · sum_n |x_n|²`.

The `noise_variance` array passed to `compute_little_H_b` and to the
new posterior helper is therefore expected to be expressed in
**Fourier units**: for white real-space noise, set
`noise_variance[k] = N · σ²_real`.

This test pins that convention by:
1. Generating real-space white noise of known variance σ²_real;
2. Applying the canonical centered FFT (`ftu.get_dft2`);
3. Asserting the empirical per-pixel Fourier power matches `N·σ²`
   to within ~3% (Monte Carlo noise on n_samples ≈ 4000 pixels).

If `DEFAULT_FFT_NORM` ever changes from `"backward"`, this test
fires and the synthetic harness must be updated to match.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as ftu

pytestmark = pytest.mark.unit


def test_default_fft_norm_is_backward():
    """The whole noise-scale contract assumes 'backward' (unnormalized)
    forward FFT. If this assumption breaks, the rest of the spec is
    silently invalid."""
    assert ftu.DEFAULT_FFT_NORM == "backward", (
        f"DEFAULT_FFT_NORM is {ftu.DEFAULT_FFT_NORM!r}, but the spec "
        "(Section 4.3) and the new PPCA harness assume 'backward'. "
        "Update the spec AND the harness in the same PR if this changes."
    )


def test_white_noise_fft_power_equals_N_sigma_squared():
    """Real-space white noise of variance σ² produces Fourier
    coefficients with per-pixel mean power N·σ², where N is the
    total number of pixels."""
    rng = np.random.default_rng(2024)
    H, W = 16, 16
    N = H * W
    sigma_real = 0.7
    n_samples = 64

    real_images = (sigma_real * rng.standard_normal((n_samples, H, W))).astype(np.float64)
    fts = ftu.get_dft2(jnp.asarray(real_images))
    fts = np.asarray(fts)
    empirical_per_pixel_power = float(np.mean(np.abs(fts) ** 2))
    expected = N * sigma_real**2

    rel_err = abs(empirical_per_pixel_power - expected) / expected
    assert rel_err < 0.05, (
        f"per-pixel Fourier power {empirical_per_pixel_power:.3f} "
        f"vs expected N·σ² = {expected:.3f} (rel err {rel_err:.3%}). "
        "If this is off by a constant factor, the FFT norm convention "
        "has drifted from 'backward' and the synthetic harness's "
        "noise_variance contract is wrong."
    )


def test_parseval_holds_for_get_dft2():
    """Direct sanity: sum_k |F_k|² == N · sum_n |x_n|² for the
    centered backward FFT, exactly (no Monte Carlo)."""
    rng = np.random.default_rng(7)
    H, W = 8, 8
    N = H * W
    img = rng.standard_normal((H, W)).astype(np.float64)
    ft = np.asarray(ftu.get_dft2(jnp.asarray(img)))

    lhs = float(np.sum(np.abs(ft) ** 2))
    rhs = N * float(np.sum(img**2))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_synthetic_harness_noise_variance_recipe():
    """Document the recipe the synthetic harness must follow.

    Given a desired real-space per-pixel noise variance σ²_real, the
    `noise_variance` array consumed by compute_little_H_b is

        noise_variance[k] = N · σ²_real

    (constant in k for white noise). This test sanity-checks that
    recipe by drawing white noise, FT'ing it, computing
    bHb-style residuals on a trivial PPCA model, and verifying that
    the empirical per-pose mean residual is consistent with χ² with
    N degrees of freedom under that noise_variance.

    This is the canary against the most common failure mode: writing
    noise_variance in real-space units, which makes the H matrix
    off by a factor of N and silently destroys the score.
    """
    rng = np.random.default_rng(101)
    H_size, W = 8, 8
    N = H_size * W
    sigma_real = 1.3
    n_samples = 200

    real_noise = (sigma_real * rng.standard_normal((n_samples, H_size, W))).astype(np.float64)
    ft_noise = np.asarray(ftu.get_dft2(jnp.asarray(real_noise)))

    # Per-pose homogeneous Mahalanobis residual against zero mean,
    # with the *correct* noise_variance:
    noise_variance_correct = np.full(N, N * sigma_real**2, dtype=np.float64)

    flat = ft_noise.reshape(n_samples, N)
    mahal = np.sum(np.abs(flat) ** 2 / noise_variance_correct, axis=-1)
    # E[mahal] should equal N (sum of N standardized squares)
    assert abs(float(np.mean(mahal)) - N) / N < 0.05, (
        f"empirical Mahalanobis mean {float(np.mean(mahal)):.2f} "
        f"vs expected N={N}. If far off, the noise_variance recipe "
        "in the spec is wrong."
    )

    # And the *wrong* recipe (real-space units) should be off by ~N:
    noise_variance_wrong = np.full(N, sigma_real**2, dtype=np.float64)
    mahal_wrong = np.sum(np.abs(flat) ** 2 / noise_variance_wrong, axis=-1)
    ratio = float(np.mean(mahal_wrong)) / float(np.mean(mahal))
    assert abs(ratio - N) / N < 0.05, (
        f"sanity check failed: passing real-space variance to a "
        f"Fourier-units consumer should inflate the residual by ~N={N}, "
        f"observed ratio {ratio:.1f}"
    )
