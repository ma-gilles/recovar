"""Noise-only tail packing preserves active operands and published statistics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_shared_local_exact_noise import _call_deferred_wrapper, _make_noise_inputs

from recovar.em.dense.deferred_noise_pack import _pad_noise_pixels, pack_noise_pixel_capacity
from recovar.em.helpers.env_flags import parse_env_binary_flag
from recovar.em.local import local_em_engine as engine

pytestmark = pytest.mark.unit
PIXELS = ("pixel_reconstruction_probs", "pixel_proj_for_noise", "pixel_ctf_probs", "bucket_image_indices")


@pytest.mark.parametrize("token,expected", [(None, False), ("0", False), ("1", True), (" 1 ", True)])
def test_selector(monkeypatch, token, expected):
    monkeypatch.delenv(engine.EXACT_LOCAL_NOISE_PIXEL_CAPACITY_ENV, raising=False)
    if token is not None:
        monkeypatch.setenv(engine.EXACT_LOCAL_NOISE_PIXEL_CAPACITY_ENV, token)
    assert parse_env_binary_flag(engine.EXACT_LOCAL_NOISE_PIXEL_CAPACITY_ENV) is expected


@pytest.mark.parametrize("token", ["", "true", "false", "2", "-1"])
def test_invalid_selector(monkeypatch, token):
    monkeypatch.setenv(engine.EXACT_LOCAL_NOISE_PIXEL_CAPACITY_ENV, token)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        parse_env_binary_flag(engine.EXACT_LOCAL_NOISE_PIXEL_CAPACITY_ENV)


def test_requires_norm_capacity_before_dataset_access(monkeypatch):
    monkeypatch.setenv(engine.EXACT_LOCAL_NOISE_PIXEL_CAPACITY_ENV, "1")
    monkeypatch.setenv(engine.EXACT_LOCAL_NOISE_NORM_CAPACITY_ENV, "0")
    with pytest.raises(ValueError, match="noise pixel capacity requires"):
        engine.run_local_em_exact(
            None,
            None,
            None,
            None,
            None,
            "linear_interp",
            image_batch_size=1,
            rotation_block_size=1,
            current_size=8,
        )


@pytest.mark.parametrize("batch", [1, 32, 41, 42])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_padding_preserves_every_original_byte(batch, dtype):
    inputs = _make_noise_inputs(batch)
    values = [inputs[k] for k in PIXELS]
    values[0] = values[0].astype(dtype).at[0, 0, 0].set(-0.0)
    result = pack_noise_pixel_capacity(*values, target_batch=42, n_images=64, norm_capacity=1024)
    for before, after in zip(values[:3], result[:3], strict=True):
        assert before.dtype == after.dtype and after.shape[0] == 42
        assert np.asarray(before).tobytes() == np.asarray(after[:batch]).tobytes()
        assert not np.any(np.asarray(after[batch:]) != 0)
    np.testing.assert_array_equal(result[3][:batch], values[3])
    np.testing.assert_array_equal(result[3][batch:], np.full(42 - batch, 64, dtype=np.int32))


@pytest.mark.parametrize("batch,n_images,capacity", [(32, 1024, 1024), (42, 200, 1024)])
def test_no_spare_or_full_batch_returns_original_objects(batch, n_images, capacity):
    inputs = _make_noise_inputs(batch)
    values = tuple(inputs[k] for k in PIXELS)
    result = pack_noise_pixel_capacity(*values, target_batch=42, n_images=n_images, norm_capacity=capacity)
    assert all(a is b for a, b in zip(values, result, strict=True))


@pytest.mark.parametrize("target,n_images,capacity", [(31, 200, 1024), (42, -1, 1024), (42, 1025, 1024)])
def test_invalid_dimensions_rejected(target, n_images, capacity):
    inputs = _make_noise_inputs(32)
    with pytest.raises(ValueError):
        pack_noise_pixel_capacity(
            *(inputs[k] for k in PIXELS), target_batch=target, n_images=n_images, norm_capacity=capacity
        )


@pytest.mark.parametrize("batch", [32, 41])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_actual_noise_outputs_and_norm_prefix_with_nonzero_carry(batch, dtype):
    inputs = _make_noise_inputs(batch)
    # Dyadic inputs make this an exact arithmetic fixture. Real CUDA operands
    # are separately qualified with the existing repeated-run science policy.
    for key, value in inputs.items():
        if isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.inexact):
            if jnp.issubdtype(value.dtype, jnp.complexfloating):
                inputs[key] = (jnp.round(value.real * 4) / 4 + 1j * jnp.round(value.imag * 4) / 4).astype(
                    jnp.complex64 if dtype == jnp.float32 else jnp.complex128
                )
            else:
                inputs[key] = (jnp.round(value * 4) / 4).astype(dtype)
    inputs["noise_variance_for_noise"] = jnp.ones(4, dtype=dtype)
    inputs["noise_norm_correction"] = jnp.concatenate((jnp.arange(64, dtype=dtype), jnp.zeros(960, dtype=dtype)))
    padded = dict(inputs)
    values = pack_noise_pixel_capacity(*(inputs[k] for k in PIXELS), target_batch=42, n_images=64, norm_capacity=1024)
    padded.update(zip(PIXELS, values, strict=True))
    expected = _call_deferred_wrapper(inputs, source_faithful_spectrum_norm=dtype == jnp.float64)
    actual = _call_deferred_wrapper(padded, source_faithful_spectrum_norm=dtype == jnp.float64)
    for a, b in zip(actual, expected, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    assert not np.any(np.asarray(actual[6][64:]) != 0)


def test_small_padding_keys_keep_heavy_noise_key_fixed():
    from recovar.em.local.local_big_jit import run_deferred_local_exact_noise_jit as noise

    noise.clear_cache()
    _pad_noise_pixels.clear_cache()
    try:
        for batch in (32, 33, 41, 42):
            inputs = _make_noise_inputs(batch)
            inputs.update(
                zip(
                    PIXELS,
                    pack_noise_pixel_capacity(
                        *(inputs[k] for k in PIXELS),
                        target_batch=42,
                        n_images=42,
                        norm_capacity=64,
                    ),
                    strict=True,
                )
            )
            jax.block_until_ready(_call_deferred_wrapper(inputs))
        assert noise._cache_size() == 1
        assert _pad_noise_pixels._cache_size() == 3
    finally:
        noise.clear_cache()
        _pad_noise_pixels.clear_cache()
