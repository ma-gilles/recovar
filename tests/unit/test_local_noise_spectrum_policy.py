"""Keep the PR179 spectrum policy distinct from InitialModel's high-shell rule."""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import local_big_jit, local_em_engine

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("owns_high_shell", [False, True])
def test_image_power_policy_matches_frozen_pr179_expression(dtype, owns_high_shell):
    # PR179 d756dccf local_big_jit.py:1357-1361 weights all spectrum pixels
    # by support_mass; its norm-correction policy is separate. Dyadic values
    # make the two scientific expectations exact in either precision.
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
    processed = jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=complex_dtype)
    support = jnp.asarray([0.25, 0.0], dtype=dtype)
    shells = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    kwargs = dict(
        shell_count=3,
        image_shape=(4, 4),
        current_size=None,
        include_unweighted_high_shell=owns_high_shell,
    )
    weighted, weighted_norm = local_big_jit._noise_image_power_shells_and_per_image(
        processed,
        support,
        shells,
        jnp.asarray([True, True]),
        1,
        unweighted_high_shell_image_power=False,
        **kwargs,
    )
    unweighted, unweighted_norm = local_big_jit._noise_image_power_shells_and_per_image(
        processed,
        support,
        shells,
        jnp.asarray([True, True]),
        1,
        unweighted_high_shell_image_power=True,
        **kwargs,
    )
    # Literal frozen producer expression, independently of the new helper.
    old_pixels = jnp.sum((jnp.abs(processed) ** 2) * support[:, None], axis=0).astype(dtype)
    np.testing.assert_array_equal(weighted, old_pixels)
    np.testing.assert_array_equal(weighted, np.asarray([0.25, 1.0, 2.25], dtype=dtype))
    np.testing.assert_array_equal(unweighted, np.asarray([0.25, 1.0, 45.0 if owns_high_shell else 0.0], dtype=dtype))
    assert weighted.dtype == unweighted.dtype == jnp.dtype(dtype)
    np.testing.assert_array_equal(weighted_norm, unweighted_norm)


@pytest.mark.parametrize("split", [False, True], ids=["bigjit", "split"])
def test_local_engine_defaults_to_pr179_spectrum_and_forwards_opt_in(monkeypatch, split):
    # Reuse the established 3-image, 8-pixel numerical fixture, not its oracle.
    spec = importlib.util.spec_from_file_location(
        "_local_spectrum_fixture", Path(__file__).with_name("test_refine_relion_mode.py")
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    args = fixture._sparse_big_jit_local_case(np.random.default_rng(891))
    monkeypatch.setenv("RECOVAR_DISABLE_LOCAL_BIG_JIT", "1" if split else "0")
    monkeypatch.setenv("RECOVAR_EXACT_LOCAL_PROCESSED_HALF_CACHE_MAX_GB", "0")
    options = dict(
        image_batch_size=3,
        rotation_block_size=8,
        current_size=4,
        accumulate_noise=True,
        score_with_masked_images=False,
        half_spectrum_scoring=True,
        reconstruct_significant_only=True,
        max_significants=1,
        adaptive_fraction=0.5,
        return_profile=True,
    )
    default = local_em_engine.run_local_em_exact(*args, "linear_interp", **options)
    weighted = local_em_engine.run_local_em_exact(
        *args, "linear_interp", unweighted_high_shell_image_power=False, **options
    )
    unweighted = local_em_engine.run_local_em_exact(
        *args, "linear_interp", unweighted_high_shell_image_power=True, **options
    )
    assert (int(default.profile["big_jit_bucket_count"]) > 0) is not split
    np.testing.assert_array_equal(default.noise_stats.wsum_img_power, weighted.noise_stats.wsum_img_power)
    assert np.any(
        np.asarray(unweighted.noise_stats.wsum_img_power)[3:] != np.asarray(weighted.noise_stats.wsum_img_power)[3:]
    )
    # This policy changes only the high-shell image-power numerator; normcorr,
    # posterior/scoring and reconstruction remain independently controlled.
    np.testing.assert_array_equal(weighted.noise_stats.wsum_img_power[:3], unweighted.noise_stats.wsum_img_power[:3])
    np.testing.assert_array_equal(
        weighted.noise_stats.wsum_norm_correction, unweighted.noise_stats.wsum_norm_correction
    )
    np.testing.assert_array_equal(weighted.hard_assignments, unweighted.hard_assignments)
    np.testing.assert_array_equal(weighted.Ft_y, unweighted.Ft_y)
    np.testing.assert_array_equal(weighted.Ft_ctf, unweighted.Ft_ctf)
