"""Opt-in zero-skipping indexed backprojection: gate, registration and GPU equivalence."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from recovar import cuda_backproject as cb

pytestmark = pytest.mark.unit


def test_skip_zero_gate_default_off(monkeypatch):
    monkeypatch.delenv(cb._BACKPROJECT_SKIP_ZERO_ENV, raising=False)
    assert cb.backproject_skip_zero_requested() is False
    monkeypatch.setenv(cb._BACKPROJECT_SKIP_ZERO_ENV, "1")
    assert cb.backproject_skip_zero_requested() is True


def test_skip_zero_target_is_optional_registration():
    symbol, _message = cb._OPTIONAL_FFI_REGISTRATIONS[cb._TARGET_BACKPROJECT_INDEXED_SKIP_ZERO]
    assert symbol == "BackprojectIndexedSkipZero"
    assert cb._TARGET_BACKPROJECT_INDEXED_SKIP_ZERO not in dict(cb._FFI_REGISTRATIONS)


def _operands(seed, n_rows, zero_fraction, image_shape=(64, 64), volume_shape=(64, 64, 64), max_r=28.0):
    rng = np.random.default_rng(seed)
    height, width = image_shape
    half_width = width // 2 + 1
    ky, kx = np.meshgrid(np.arange(height), np.arange(half_width), indexing="ij")
    ky_signed = np.where(ky <= height // 2, ky, ky - height)
    keep = (ky_signed**2 + kx**2) <= max_r**2
    pixel_indices = (ky * half_width + kx)[keep].astype(np.int32)
    n_pixels = int(pixel_indices.size)
    data = (rng.normal(size=(n_rows, n_pixels)) + 1j * rng.normal(size=(n_rows, n_pixels))).astype(np.complex64)
    weight = np.abs(rng.normal(size=(n_rows, n_pixels))).astype(np.float32)
    zero_rows = rng.random(n_rows) < zero_fraction
    data[zero_rows] = 0
    weight[zero_rows] = 0
    # Scatter a few exact zeros inside live rows too.
    weight[~zero_rows, ::7] = 0
    data[~zero_rows, ::11] = 0
    q = rng.normal(size=(n_rows, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    rotations = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], 1).astype(np.float32)
    volume_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    return data, weight, pixel_indices, rotations, volume_size, zero_rows


def _run(target_skip, data, weight, pixel_indices, rotations, volume_size, image_shape, volume_shape, max_r, monkeypatch):
    monkeypatch.setenv(cb._BACKPROJECT_SKIP_ZERO_ENV, "1" if target_skip else "0")
    # backproject_indexed is jitted and bakes the chosen FFI target into the
    # cached executable, so the same shapes would silently reuse the previous
    # arm's kernel and make this comparison vacuous.
    jax.clear_caches()
    kwargs = dict(image_shape=image_shape, volume_shape=volume_shape, order=1, half_volume=True, half_image=True, max_r=max_r, relion_x_half=True)
    y = cb.backproject_indexed(jnp.zeros(volume_size, jnp.complex64), jnp.asarray(data), jnp.asarray(pixel_indices), jnp.asarray(rotations), **kwargs)
    w = cb.backproject_indexed(jnp.zeros(volume_size, jnp.float32), jnp.asarray(weight), jnp.asarray(pixel_indices), jnp.asarray(rotations), **kwargs)
    return np.asarray(jax.block_until_ready(y)), np.asarray(jax.block_until_ready(w))


@pytest.mark.gpu
@pytest.mark.parametrize("n_rows,zero_fraction", [(1, 0.0), (5, 1.0), (40, 0.8), (200, 0.85)])
def test_skip_zero_matches_reference_scatter(monkeypatch, n_rows, zero_fraction):
    """Skipping exact zeros changes only the number of atomics, not the sums."""

    assert jax.default_backend() == "gpu"
    cb._ensure_ffi()
    if not hasattr(cb._get_lib(), "BackprojectIndexedSkipZero"):
        pytest.skip("loaded CUDA library lacks BackprojectIndexedSkipZero")
    image_shape, volume_shape, max_r = (64, 64), (64, 64, 64), 28.0
    data, weight, pixel_indices, rotations, volume_size, zero_rows = _operands(3 + n_rows, n_rows, zero_fraction, image_shape, volume_shape, max_r)
    y_ref, w_ref = _run(False, data, weight, pixel_indices, rotations, volume_size, image_shape, volume_shape, max_r, monkeypatch)
    y_skip, w_skip = _run(True, data, weight, pixel_indices, rotations, volume_size, image_shape, volume_shape, max_r, monkeypatch)
    # Atomic accumulation order is not fixed, so compare against the float32
    # rounding band of the reference rather than bitwise.
    scale_w = max(float(np.abs(w_ref).max()), 1e-30)
    scale_y = max(float(np.abs(y_ref).max()), 1e-30)
    assert float(np.abs(w_skip - w_ref).max()) / scale_w <= 2e-6
    assert float(np.abs(y_skip - y_ref).max()) / scale_y <= 2e-6
    if zero_fraction >= 1.0:
        np.testing.assert_array_equal(w_skip, 0)
        np.testing.assert_array_equal(y_skip, 0)
    # Rows that were zero contribute nothing: dropping them from the reference operands gives the same volume.
    live = ~zero_rows
    if live.any() and not live.all():
        y_live, w_live = _run(False, data[live], weight[live], pixel_indices, rotations[live], volume_size, image_shape, volume_shape, max_r, monkeypatch)
        assert float(np.abs(w_skip - w_live).max()) / scale_w <= 2e-6
        assert float(np.abs(y_skip - y_live).max()) / scale_y <= 2e-6


def test_flag_off_selects_the_original_target(monkeypatch):
    """With the gate off the accepted path is chosen and no optional ABI is touched."""
    requested = []
    monkeypatch.delenv(cb._BACKPROJECT_SKIP_ZERO_ENV, raising=False)
    monkeypatch.setattr(cb, "_ensure_optional_ffi", requested.append)
    assert cb._backproject_indexed_target(False) == cb._TARGET_BACKPROJECT_INDEXED
    assert cb._backproject_indexed_target(True) == cb._TARGET_BACKPROJECT_INDEXED
    assert requested == []


def test_flag_on_selects_the_skip_zero_target(monkeypatch):
    requested = []
    monkeypatch.setenv(cb._BACKPROJECT_SKIP_ZERO_ENV, "1")
    monkeypatch.setattr(cb, "_ensure_optional_ffi", requested.append)
    assert cb._backproject_indexed_target(False) == cb._TARGET_BACKPROJECT_INDEXED_SKIP_ZERO
    assert requested == [cb._TARGET_BACKPROJECT_INDEXED_SKIP_ZERO]


def test_block_topology_keeps_the_original_target(monkeypatch):
    """The RELION block topology re-expands operands, so the skip variant is refused."""
    requested = []
    monkeypatch.setenv(cb._BACKPROJECT_SKIP_ZERO_ENV, "1")
    monkeypatch.setattr(cb, "_ensure_optional_ffi", requested.append)
    assert cb._backproject_indexed_target(True) == cb._TARGET_BACKPROJECT_INDEXED
    assert requested == []
