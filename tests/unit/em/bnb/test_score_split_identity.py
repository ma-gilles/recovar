"""Test 2: score(full) == score(low) + score(high) on the BnB partition.

cryoSPARC's bound (Suppl Eq 12) hinges on splitting the squared-error sum
``E = A_L + B_L`` exactly: low and high are disjoint and exhaustive over the
final score support. RECOVAR's score is a linear function of that pixel set
(it's a sum of pixel-level contributions), so the identity

    score(full) == score(low) + score(high)

must hold to floating-point precision for any image, any projection, any
CTF/noise. If a future refactor breaks this identity (e.g. by introducing a
window-dependent normalization), the BnB bound silently becomes invalid.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.bnb.frequency import (
    fourier_window_spec_from_indices,
    make_bnb_high_indices_np,
)
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import make_half_image_weights
from recovar.em.dense_single_volume.helpers.scoring import _score_rotation_block


def _build_synthetic_inputs(
    image_shape: tuple[int, int],
    n_rot: int,
    n_images: int,
    n_trans: int,
    seed: int,
) -> dict:
    """Make a tiny self-consistent score-kernel input bundle.

    No physical model — just complex/real arrays of the right shapes so the
    score kernel's algebra exercises every code path.
    """
    rng = np.random.default_rng(seed)
    H, W = image_shape
    n_half = H * (W // 2 + 1)

    shifted = rng.standard_normal((n_images * n_trans, n_half)) + 1j * rng.standard_normal(
        (n_images * n_trans, n_half),
    )
    proj_half = rng.standard_normal((n_rot, n_half)) + 1j * rng.standard_normal((n_rot, n_half))
    # ctf^2/sigma^2 strictly positive.
    ctf2_over_nv = rng.uniform(0.2, 1.5, size=(n_images, n_half))
    half_weights = np.asarray(make_half_image_weights(image_shape))

    return {
        "shifted_half": jnp.asarray(shifted, dtype=jnp.complex64),
        "proj_half": jnp.asarray(proj_half, dtype=jnp.complex64),
        "proj_abs2_half": jnp.asarray(np.abs(proj_half) ** 2, dtype=jnp.float32),
        "ctf2_over_nv_half": jnp.asarray(ctf2_over_nv, dtype=jnp.float32),
        "half_weights": jnp.asarray(half_weights, dtype=jnp.float32),
        "n_images": n_images,
        "n_trans": n_trans,
        "image_shape": image_shape,
        "volume_shape": (H, H, H),
        "n_half": n_half,
    }


def _score_with_window(inputs: dict, window_spec) -> np.ndarray:
    """Run _score_rotation_block once with the given window_spec.

    Returns a (n_images, n_rot, n_trans) array of scores. Pre-windows the
    shifted/ctf arrays the way the existing callers do.
    """
    shifted_full = inputs["shifted_half"]
    ctf2_full = inputs["ctf2_over_nv_half"]

    shifted_windowed = window_spec.score_values(shifted_full)
    ctf2_windowed = window_spec.score_values(ctf2_full)

    precision_policy = DensePrecisionPolicy()
    scores = _score_rotation_block(
        window_spec,
        shifted_score=shifted_windowed,
        batch_norm=jnp.zeros((inputs["n_images"] * inputs["n_trans"], 1)),
        score_weight=ctf2_windowed,
        proj_half=inputs["proj_half"],
        proj_abs2_half=inputs["proj_abs2_half"],
        half_weights=inputs["half_weights"],
        n_images=inputs["n_images"],
        n_trans=inputs["n_trans"],
        image_shape=inputs["image_shape"],
        volume_shape=inputs["volume_shape"],
        score_mode="gaussian",
        precision_policy=precision_policy,
    )
    return np.asarray(scores)


SPLIT_CASES = [
    # (image_shape, current_size, L)
    ((16, 16), 16, 4),
    ((16, 16), 16, 6),
    ((16, 16), 12, 4),
    ((32, 32), 32, 8),
    ((32, 32), 32, 12),
    ((32, 32), 24, 6),
]


@pytest.mark.parametrize(("image_shape", "current_size", "L"), SPLIT_CASES)
def test_score_low_plus_high_equals_full(image_shape, current_size, L):
    """For random inputs: score(final) ≈ score(low) + score(high)."""
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    final_idx, _ = make_fourier_window_indices_np(
        image_shape, current_size, square=False, include_dc=False,
    )
    low_idx, _ = make_fourier_window_indices_np(
        image_shape, 2 * L, square=False, include_dc=False,
    )
    high_idx = make_bnb_high_indices_np(final_idx, low_idx)

    final_spec = fourier_window_spec_from_indices(final_idx)
    low_spec = fourier_window_spec_from_indices(low_idx)
    high_spec = fourier_window_spec_from_indices(high_idx)

    inputs = _build_synthetic_inputs(
        image_shape=image_shape,
        n_rot=3,
        n_images=2,
        n_trans=2,
        seed=int(current_size * 1009 + L * 17),
    )

    s_full = _score_with_window(inputs, final_spec)
    s_low = _score_with_window(inputs, low_spec)
    s_high = _score_with_window(inputs, high_spec)

    np.testing.assert_allclose(
        s_full,
        s_low + s_high,
        rtol=1e-5,
        atol=1e-4,
        err_msg=(
            f"BnB low/high partition violates linearity for image_shape="
            f"{image_shape}, current_size={current_size}, L={L}"
        ),
    )


def test_score_split_with_zero_high():
    """When low covers the whole final support, score(high) == 0."""
    image_shape = (16, 16)
    final_idx, _ = make_fourier_window_indices_np(
        image_shape, 16, square=False, include_dc=False,
    )
    # Choose 2L = current_size so low == final.
    low_idx, _ = make_fourier_window_indices_np(
        image_shape, 16, square=False, include_dc=False,
    )
    high_idx = make_bnb_high_indices_np(final_idx, low_idx)
    assert high_idx.size == 0

    final_spec = fourier_window_spec_from_indices(final_idx)
    low_spec = fourier_window_spec_from_indices(low_idx)

    inputs = _build_synthetic_inputs(
        image_shape=image_shape, n_rot=2, n_images=1, n_trans=1, seed=42,
    )
    s_full = _score_with_window(inputs, final_spec)
    s_low = _score_with_window(inputs, low_spec)
    np.testing.assert_allclose(s_full, s_low, rtol=1e-6, atol=1e-6)
