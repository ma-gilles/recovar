"""Test 3: deterministic Cauchy upper bound on the BnB high-frequency score.

The cryoSPARC bound (Suppl Eq 22) is probabilistic (4 sigma → 0.999936
probability of holding). To get a no-false-pruning safety check, we derive a
*deterministic* Cauchy-Schwarz upper bound:

    s_H(r) ≤ image_high_power                            if image_high_power ≤ P^max
    s_H(r) ≤ -P^max + 2·sqrt(P^max · image_high_power)   otherwise

This bound must hold for every pose, not just most poses. The cryoSPARC
probabilistic test (violation rate < 1e-3 on noisy particles) is deferred to
Phase 2, where the engine and a synthetic-particle fixture are available.

For Phase 1 we test pure linear-algebra correctness: synthetic random
volumes, projected via the existing slicer, scored via the existing kernel,
with P^max computed by our new ``compute_high_model_pmax_per_image``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.bnb.bounds import (
    cauchy_score_upper_correction,
    compute_high_model_pmax_per_image,
    compute_image_high_power_per_image,
)
from recovar.em.dense_single_volume.bnb.frequency import (
    fourier_window_spec_from_indices,
    make_bnb_high_indices_np,
)
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import make_half_image_weights
from recovar.em.dense_single_volume.helpers.projection import (
    compute_projections_block,
)
from recovar.em.dense_single_volume.helpers.scoring import _score_rotation_block


def _random_rotations(n: int, seed: int) -> np.ndarray:
    """n random SO(3) matrices (Haar-uniform via QR on Gaussian)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, 3, 3))
    rotations = np.empty_like(M, dtype=np.float32)
    for i in range(n):
        Q, R = np.linalg.qr(M[i])
        # Make the QR sign canonical (det = +1 for proper rotation).
        d = np.sign(np.diag(R))
        d[d == 0] = 1
        Q = Q * d
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        rotations[i] = Q.astype(np.float32)
    return rotations


def _score(
    inputs,
    window_spec,
    proj_half,
    proj_abs2_half,
) -> np.ndarray:
    """Score (n_images, n_rot, n_trans=1) for the given window and projections."""
    shifted = window_spec.score_values(inputs["shifted_half"])
    ctf2 = window_spec.score_values(inputs["ctf2_over_nv_half"])
    precision_policy = DensePrecisionPolicy()
    scores = _score_rotation_block(
        window_spec,
        shifted_score=shifted,
        batch_norm=jnp.zeros((inputs["n_images"], 1)),
        score_weight=ctf2,
        proj_half=proj_half,
        proj_abs2_half=proj_abs2_half,
        half_weights=inputs["half_weights"],
        n_images=inputs["n_images"],
        n_trans=1,
        image_shape=inputs["image_shape"],
        volume_shape=inputs["volume_shape"],
        score_mode="gaussian",
        precision_policy=precision_policy,
    )
    return np.asarray(scores)


CAUCHY_CASES = [
    # (image_shape, current_size, L, n_rot, seed)
    ((16, 16), 16, 4, 8, 11),
    ((16, 16), 16, 6, 8, 13),
    ((16, 16), 12, 4, 12, 17),
    ((32, 32), 32, 8, 16, 19),
    ((32, 32), 32, 12, 16, 23),
    ((32, 32), 24, 6, 12, 29),
]


@pytest.mark.parametrize(("image_shape", "current_size", "L", "n_rot", "seed"), CAUCHY_CASES)
def test_cauchy_bound_holds_for_all_poses(image_shape, current_size, L, n_rot, seed):
    """For every (image, rotation), the deterministic Cauchy bound holds."""
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    volume_shape = (H, H, H)
    volume_size = H * H * H

    rng = np.random.default_rng(seed)

    # Random complex volume in centered Fourier layout. compute_projections_block
    # takes flat (volume_size,) complex.
    mean = (
        rng.standard_normal(volume_size).astype(np.float32)
        + 1j * rng.standard_normal(volume_size).astype(np.float32)
    )
    mean = jnp.asarray(mean, dtype=jnp.complex64)

    rotations = _random_rotations(n_rot, seed=seed + 1)

    # Project the mean for all candidate rotations.
    proj_half, proj_abs2_half = compute_projections_block(
        mean, rotations, image_shape, volume_shape, "linear_interp", return_abs2=True,
    )

    n_images = 2
    # Random complex "images" on the half-spectrum.
    image_X = (
        rng.standard_normal((n_images, n_half)).astype(np.float32)
        + 1j * rng.standard_normal((n_images, n_half)).astype(np.float32)
    )
    # Constant CTF / unit noise to keep the algebra clean. The bound is
    # derived for arbitrary C, sigma — the unit-test only needs to exercise
    # the score-vs-bound arithmetic, not the CTF model.
    ctf2_over_nv = np.ones((n_images, n_half), dtype=np.float32)
    half_weights = np.asarray(make_half_image_weights(image_shape), dtype=np.float32)

    # In RECOVAR preprocessing: shifted_half = X * C / sigma^2 with one trans.
    # With C=1, sigma^2=1: shifted_half = X.
    shifted_half = jnp.asarray(image_X, dtype=jnp.complex64)
    ctf2_over_nv_half = jnp.asarray(ctf2_over_nv, dtype=jnp.float32)
    half_weights_j = jnp.asarray(half_weights, dtype=jnp.float32)

    inputs = {
        "shifted_half": shifted_half,
        "ctf2_over_nv_half": ctf2_over_nv_half,
        "half_weights": half_weights_j,
        "n_images": n_images,
        "n_trans": 1,
        "image_shape": image_shape,
        "volume_shape": volume_shape,
    }

    # Window specs for the three bands.
    final_idx, _ = make_fourier_window_indices_np(
        image_shape, current_size, square=False, include_dc=False,
    )
    low_idx, _ = make_fourier_window_indices_np(
        image_shape, 2 * L, square=False, include_dc=False,
    )
    high_idx = make_bnb_high_indices_np(final_idx, low_idx)

    if high_idx.size == 0:
        pytest.skip("Empty high band — Cauchy bound is trivially 0; nothing to test.")

    final_spec = fourier_window_spec_from_indices(final_idx)
    low_spec = fourier_window_spec_from_indices(low_idx)

    # s_full and s_low using _score_rotation_block.
    s_full = _score(inputs, final_spec, proj_half, proj_abs2_half)
    s_low = _score(inputs, low_spec, proj_half, proj_abs2_half)
    s_H = s_full - s_low  # shape (n_images, n_rot, 1)

    # Compute P^max_H using the new bound primitive.
    pmax_per_image = compute_high_model_pmax_per_image(
        mean,
        rotations,
        ctf2_over_nv_half,
        half_weights_j,
        jnp.asarray(high_idx, dtype=jnp.int32),
        image_shape=image_shape,
        volume_shape=volume_shape,
        disc_type="linear_interp",
    )
    pmax_per_image_np = np.asarray(pmax_per_image)

    # Compute image_high_power = 1/2 sum_l h_l |X_l|^2 / sigma_l^2 over high.
    # With sigma^2=1: image_high_power = 1/2 sum_l h_l |X_l|^2.
    image_power_over_sigma2_high = (np.abs(image_X) ** 2)[:, high_idx]
    half_weights_high = half_weights[high_idx]
    image_high_power = compute_image_high_power_per_image(
        jnp.asarray(image_power_over_sigma2_high, dtype=jnp.float32),
        jnp.asarray(half_weights_high, dtype=jnp.float32),
    )
    image_high_power_np = np.asarray(image_high_power)

    # Cauchy upper bound — per image.
    cauchy_upper = np.asarray(cauchy_score_upper_correction(
        image_high_power,
        pmax_per_image,
    ))

    # Verify: s_H[i, r, 0] <= cauchy_upper[i] for all i, r.
    s_H_max_per_image = s_H[..., 0].max(axis=1)  # max over rotations
    margin = 1e-4 * (np.abs(cauchy_upper) + np.abs(s_H_max_per_image) + 1.0)
    violation = s_H_max_per_image - cauchy_upper - margin
    if np.any(violation > 0):
        idx = int(np.argmax(violation))
        raise AssertionError(
            f"Cauchy upper bound violated for image {idx}: "
            f"s_H_max={s_H_max_per_image[idx]:.6f}, cauchy_upper={cauchy_upper[idx]:.6f}, "
            f"P^max={pmax_per_image_np[idx]:.6f}, image_high_power={image_high_power_np[idx]:.6f}, "
            f"L={L}, current_size={current_size}"
        )


def test_cauchy_bound_invariants():
    """Edge cases of the Cauchy upper-bound formula.

    Recall the bound: s_H(r) ≤ -P^max(r) + 2·sqrt(P^max(r)·H), maximized over
    P^max(r) ∈ [0, P^max], where H = image_high_power. The maximizer is
    P^max(r)=H if H ≤ P^max, else P^max(r)=P^max (the boundary).
    """
    # P^max = 0: model has zero high-frequency power, so s_H is identically 0.
    # The bound is 0, NOT image_high_power.
    upper = cauchy_score_upper_correction(
        jnp.asarray([1.0, 5.0, 0.0]), jnp.asarray([0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(np.asarray(upper), [0.0, 0.0, 0.0], atol=1e-7)

    # image_high_power = 0: image has zero high-frequency content, so s_H ≤ 0.
    upper2 = cauchy_score_upper_correction(
        jnp.asarray([0.0, 0.0, 0.0]), jnp.asarray([1.0, 5.0, 100.0]),
    )
    np.testing.assert_allclose(np.asarray(upper2), [0.0, 0.0, 0.0])

    # At the breakeven image_high_power == P^max, both regimes give the same
    # answer (= image_high_power = P^max).
    a = jnp.asarray([2.5])
    upper3 = cauchy_score_upper_correction(a, a)
    np.testing.assert_allclose(np.asarray(upper3), [2.5], atol=1e-7)

    # H < P^max: bound = H.
    upper4 = cauchy_score_upper_correction(
        jnp.asarray([1.0, 2.0, 0.5]), jnp.asarray([4.0, 9.0, 1.0]),
    )
    np.testing.assert_allclose(np.asarray(upper4), [1.0, 2.0, 0.5], atol=1e-7)

    # H > P^max: bound = -P^max + 2·sqrt(P^max · H).
    H = np.array([10.0, 20.0])
    P = np.array([1.0, 4.0])
    upper5 = cauchy_score_upper_correction(jnp.asarray(H), jnp.asarray(P))
    expected = -P + 2.0 * np.sqrt(P * H)
    np.testing.assert_allclose(np.asarray(upper5), expected, atol=1e-6)


def test_cryosparc_bound_formula():
    """cryosparc_score_upper_correction = P^max + tau * sqrt(P^max)."""
    from recovar.em.dense_single_volume.bnb.bounds import cryosparc_score_upper_correction

    pmax = jnp.asarray([0.0, 1.0, 4.0, 9.0])
    delta = cryosparc_score_upper_correction(pmax, tau_sigma=4.0)
    expected = np.array([0.0, 1.0 + 4.0 * 1.0, 4.0 + 4.0 * 2.0, 9.0 + 4.0 * 3.0])
    np.testing.assert_allclose(np.asarray(delta), expected, atol=1e-6)

    # Non-default tau.
    delta2 = cryosparc_score_upper_correction(pmax, tau_sigma=2.0)
    expected2 = np.array([0.0, 1.0 + 2.0 * 1.0, 4.0 + 2.0 * 2.0, 9.0 + 2.0 * 3.0])
    np.testing.assert_allclose(np.asarray(delta2), expected2, atol=1e-6)


def test_rms_ctf_mode_returns_same_value_per_image():
    """RMS-CTF mode broadcasts a single P^max across all images that share
    the noise spectrum (Phase-6 cryoSPARC speed optimization)."""
    image_shape = (16, 16)
    H, W = image_shape
    n_half = H * (W // 2 + 1)
    volume_shape = (H, H, H)
    volume_size = H * H * H

    rng = np.random.default_rng(101)
    mean = jnp.asarray(
        rng.standard_normal(volume_size).astype(np.float32)
        + 1j * rng.standard_normal(volume_size).astype(np.float32),
        dtype=jnp.complex64,
    )
    rotations = _random_rotations(4, seed=102)

    n_images = 3
    # Per-image CTF/noise differing across images (so the exact bound
    # produces 3 different Pmax values).
    ctf2_over_nv = rng.uniform(0.3, 1.5, size=(n_images, n_half)).astype(np.float32)
    half_weights = np.asarray(make_half_image_weights(image_shape), dtype=np.float32)

    final_idx, _ = make_fourier_window_indices_np(
        image_shape, 16, square=False, include_dc=False,
    )
    low_idx, _ = make_fourier_window_indices_np(
        image_shape, 8, square=False, include_dc=False,
    )
    high_idx = make_bnb_high_indices_np(final_idx, low_idx)

    # Shared noise spectrum across all images.
    noise_variance_half = np.ones(n_half, dtype=np.float32) * 0.7

    # Exact mode: Pmax varies across images.
    pmax_exact = np.asarray(compute_high_model_pmax_per_image(
        mean,
        rotations,
        jnp.asarray(ctf2_over_nv),
        jnp.asarray(half_weights),
        jnp.asarray(high_idx, dtype=jnp.int32),
        image_shape=image_shape,
        volume_shape=volume_shape,
        disc_type="linear_interp",
    ))
    assert np.std(pmax_exact) > 1e-6, "Exact-mode Pmax should differ across images"

    # RMS mode with shared noise_variance_half: Pmax broadcast to one value.
    pmax_rms = np.asarray(compute_high_model_pmax_per_image(
        mean,
        rotations,
        jnp.asarray(ctf2_over_nv),
        jnp.asarray(half_weights),
        jnp.asarray(high_idx, dtype=jnp.int32),
        image_shape=image_shape,
        volume_shape=volume_shape,
        disc_type="linear_interp",
        use_rms_ctf_approximation=True,
        rms_ctf_squared=0.5,
        noise_variance_half=jnp.asarray(noise_variance_half),
    ))
    # All RMS Pmax values must agree (shared noise spectrum -> shared bound).
    np.testing.assert_allclose(pmax_rms, pmax_rms[0], atol=1e-6)
