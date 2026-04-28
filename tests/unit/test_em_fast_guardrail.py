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
