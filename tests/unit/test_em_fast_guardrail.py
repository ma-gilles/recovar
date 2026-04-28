"""Fast guardrail checks for dense/local EM refactor helpers.

The full dense/local guardrail command lives in ``scripts/run_em_fast_guard.sh``.
This file keeps the helper-specific assertions small enough to run alongside
existing dense and local exact EM equivalence tests.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec

pytestmark = pytest.mark.unit


def test_dense_precision_policy_casts_score_and_projection_dtypes():
    score = jnp.asarray([1.0 + 2.0j], dtype=jnp.complex128)
    score_weight = jnp.asarray([1.5], dtype=jnp.float64)
    recon = jnp.asarray([3.0 + 4.0j], dtype=jnp.complex64)
    volume = jnp.asarray([5.0 + 6.0j], dtype=jnp.complex64)

    default_policy = DensePrecisionPolicy()
    cast_score, cast_weight, cast_recon = default_policy.cast_scoring_inputs(
        score,
        score_weight,
        recon,
    )

    assert default_policy.cast_projection_volume(volume) is volume
    assert default_policy.normalization_real_dtype == jnp.float64
    assert cast_score.dtype == jnp.complex64
    assert cast_weight.dtype == jnp.float32
    assert cast_recon.dtype == jnp.complex64

    precise_policy = DensePrecisionPolicy(
        use_float64_scoring=True,
        use_float64_projections=True,
    )
    cast_score, cast_weight, cast_recon = precise_policy.cast_scoring_inputs(
        score.astype(jnp.complex64),
        score_weight.astype(jnp.float32),
        recon,
    )

    assert precise_policy.cast_projection_volume(volume).dtype == jnp.complex128
    assert precise_policy.normalization_real_dtype == jnp.float64
    assert cast_score.dtype == jnp.complex128
    assert cast_weight.dtype == jnp.float64
    assert cast_recon.dtype == jnp.complex128

    float32_normalization_policy = DensePrecisionPolicy(use_float64_normalization=False)
    assert float32_normalization_policy.normalization_real_dtype == jnp.float32


def test_dense_precision_policy_casts_projection_scores_only_for_float32_scoring():
    proj = jnp.asarray([1.0 + 2.0j], dtype=jnp.complex128)
    proj_abs2 = jnp.asarray([3.0], dtype=jnp.float64)

    proj32, abs32 = DensePrecisionPolicy().cast_projection_scores(proj, proj_abs2)
    assert proj32.dtype == jnp.complex64
    assert abs32.dtype == jnp.float32

    proj64, abs64 = DensePrecisionPolicy(use_float64_scoring=True).cast_projection_scores(
        proj,
        proj_abs2,
    )
    assert proj64.dtype == jnp.complex128
    assert abs64.dtype == jnp.float64


def test_dense_precision_policy_casts_local_score_inputs():
    score = jnp.asarray([1.0 + 2.0j], dtype=jnp.complex128)
    recon = jnp.asarray([3.0 + 4.0j], dtype=jnp.complex64)
    noise = jnp.asarray([5.0 + 6.0j], dtype=jnp.complex64)
    score_weight = jnp.asarray([1.0], dtype=jnp.float64)
    recon_weight = jnp.asarray([2.0], dtype=jnp.float32)
    proj_abs2 = jnp.asarray([3.0], dtype=jnp.float64)

    default_policy = DensePrecisionPolicy()
    cast_score, cast_recon, cast_noise, cast_score_weight, cast_recon_weight = (
        default_policy.cast_local_preprocessed_inputs(
            score,
            recon,
            noise,
            score_weight,
            recon_weight,
        )
    )
    proj, proj_noise, abs2, abs2_noise = default_policy.cast_local_projection_scores(
        score,
        recon,
        proj_abs2,
        proj_abs2,
    )

    assert cast_score.dtype == jnp.complex64
    assert cast_score_weight.dtype == jnp.float32
    assert cast_recon.dtype == jnp.complex64
    assert cast_noise.dtype == jnp.complex64
    assert cast_recon_weight.dtype == jnp.float32
    assert proj.dtype == jnp.complex64
    assert proj_noise.dtype == jnp.complex64
    assert abs2.dtype == jnp.float32
    assert abs2_noise.dtype == jnp.float64

    precise_policy = DensePrecisionPolicy(use_float64_scoring=True)
    cast_score, cast_recon, cast_noise, cast_score_weight, cast_recon_weight = (
        precise_policy.cast_local_preprocessed_inputs(
            score.astype(jnp.complex64),
            recon,
            noise,
            score_weight.astype(jnp.float32),
            recon_weight,
        )
    )
    proj, proj_noise, abs2, abs2_noise = precise_policy.cast_local_projection_scores(
        score.astype(jnp.complex64),
        recon,
        proj_abs2.astype(jnp.float32),
        proj_abs2.astype(jnp.float32),
    )

    assert cast_score.dtype == jnp.complex128
    assert cast_score_weight.dtype == jnp.float64
    assert cast_recon.dtype == jnp.complex128
    assert cast_noise.dtype == jnp.complex128
    assert cast_recon_weight.dtype == jnp.float64
    assert proj.dtype == jnp.complex128
    assert proj_noise.dtype == jnp.complex128
    assert abs2.dtype == jnp.float64
    assert abs2_noise.dtype == jnp.float64


def test_dense_precision_policy_casts_local_big_jit_inputs():
    complex64 = jnp.asarray([1.0 + 2.0j], dtype=jnp.complex64)
    complex128 = jnp.asarray([3.0 + 4.0j], dtype=jnp.complex128)
    real32 = jnp.asarray([1.0], dtype=jnp.float32)
    real64 = jnp.asarray([2.0], dtype=jnp.float64)

    default_casts = DensePrecisionPolicy().cast_local_big_jit_inputs(
        complex128,
        complex64,
        complex128,
        real64,
        real32,
        complex128,
        complex64,
    )
    assert [value.dtype for value in default_casts] == [
        jnp.complex64,
        jnp.complex64,
        jnp.complex64,
        jnp.float32,
        jnp.float32,
        jnp.complex64,
        jnp.complex64,
    ]

    precise_casts = DensePrecisionPolicy(use_float64_scoring=True).cast_local_big_jit_inputs(
        complex64,
        complex64,
        complex64,
        real32,
        real32,
        complex64,
        complex64,
    )
    assert [value.dtype for value in precise_casts] == [
        jnp.complex128,
        jnp.complex128,
        jnp.complex128,
        jnp.float64,
        jnp.float64,
        jnp.complex128,
        jnp.complex128,
    ]


def test_fourier_window_spec_gathers_last_axis_for_batched_values():
    image_shape = (8, 8)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    spec = make_fourier_window_spec(
        image_shape,
        current_size=6,
        n_half=n_half,
        include_recon_window=True,
    )
    values = jnp.arange(2 * 3 * n_half, dtype=jnp.float32).reshape(2, 3, n_half)

    np.testing.assert_array_equal(
        np.asarray(spec.score_values(values)),
        np.asarray(values)[..., np.asarray(spec.score_indices_np)],
    )
    np.testing.assert_array_equal(
        np.asarray(spec.recon_values(values)),
        np.asarray(values)[..., np.asarray(spec.recon_indices_np)],
    )
