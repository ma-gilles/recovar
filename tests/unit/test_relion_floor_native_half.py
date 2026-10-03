"""RELION's reconstruct() denominator floor from a native packed half.

RELION stores its accumulator as the x-half (``kx >= 0``) and its radial
average visits every stored entry once, so Hermitian pairs on the ``kx = 0``
plane count twice and all other pairs once. ``_relion_reconstruct_floor_volume``
must give that average whether it is handed the full ``(x, y, z)`` cube or its
native packed half ``(x, y, z >= 0)``, whose own-mate plane is ``kz = 0``.
The reference below is computed in NumPy from the x-half storage alone.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.core import fourier_transform_utils
from recovar.reconstruction import relion_functions

pytestmark = pytest.mark.unit

# (accumulator size, padding factor): RELION's BPref grid is odd
# (2 * padding * r_max + 3); production pads by 2, or by 1 with --pad 1.
_GRIDS = [(15, 1), (15, 2), (27, 2), (16, 1), (16, 2)]


def _centered_frequencies(size):
    return np.arange(size) - size // 2


def _hermitian_mate(values):
    """``values[-k]`` on a centered grid; an even axis' Nyquist bin is its own mate."""

    for axis, size in enumerate(values.shape):
        mate = (size // 2 - _centered_frequencies(size)) % size
        values = np.take(values, mate, axis=axis)
    return values


def _anisotropic_filter(size, dtype=np.float64):
    """A real Hermitian weight that differs strongly between the kx = 0 and kz = 0 planes."""

    rng = np.random.default_rng(11)
    noise = rng.uniform(0.5, 1.5, size=(size,) * 3)
    noise = 0.5 * (noise + _hermitian_mate(noise))
    k = _centered_frequencies(size).astype(np.float64)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    profile = 1.0 + 40.0 * np.exp(-(kx**2) / 2.0) + 5.0 * np.exp(-(kz**2) / 2.0) + 0.1 * np.abs(ky)
    values = noise * profile
    np.testing.assert_array_equal(values, _hermitian_mate(values))
    return values.astype(dtype)


def _relion_shell_average(values, padding_factor, max_res_shell):
    """RELION's rule: every stored entry of the x-half once, shell ``floor(r / padding)``."""

    size = values.shape[0]
    k = _centered_frequencies(size)
    stored_x = np.flatnonzero((k >= 0) | ((size % 2 == 0) & (k == -(size // 2))))
    shell_sum = np.zeros(max_res_shell, dtype=np.float64)
    shell_count = np.zeros(max_res_shell, dtype=np.float64)
    for ix in stored_x:
        for iy in range(size):
            for iz in range(size):
                shell = int(np.floor(np.sqrt(float(k[ix] ** 2 + k[iy] ** 2 + k[iz] ** 2)) / padding_factor))
                if shell < max_res_shell:
                    shell_sum[shell] += values[ix, iy, iz]
                    shell_count[shell] += 1.0
    return np.where(shell_count > 0, shell_sum / np.maximum(shell_count, 1.0), 0.0)


def _relion_floor_volume(values, padding_factor, max_res_shell):
    size = values.shape[0]
    k = _centered_frequencies(size).astype(np.float64)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    shell = np.floor(np.sqrt(kx**2 + ky**2 + kz**2) / padding_factor).astype(np.int64)
    average = _relion_shell_average(values, padding_factor, max_res_shell)
    return average[np.minimum(shell, max_res_shell - 1)] / 1000.0


def _to_half(values):
    shape = values.shape
    return np.asarray(fourier_transform_utils.full_volume_to_half_volume(jnp.asarray(values), shape))


def _floor_kwargs(case, size, padding_factor):
    """``max_res_shell`` as an int, as None, and traced below a static bound."""

    default = size // (2 * padding_factor)
    if case == "int":
        shell = max(2, default - 1)
        return shell, dict(max_res_shell=shell)
    if case == "none":
        return max(1, default), dict(max_res_shell=None)
    shell = max(2, default - 1)
    return shell, dict(max_res_shell=jnp.int32(shell), max_res_shell_bound=default + 2)


@pytest.mark.parametrize("size, padding_factor", _GRIDS)
@pytest.mark.parametrize("case", ["int", "none", "traced"])
def test_floor_volume_matches_relion_x_half_rule_in_both_layouts(size, padding_factor, case):
    shape = (size,) * 3
    values = _anisotropic_filter(size)
    max_res_shell, kwargs = _floor_kwargs(case, size, padding_factor)
    expected = _relion_floor_volume(values, padding_factor, max_res_shell)

    full = relion_functions._relion_reconstruct_floor_volume(
        jnp.asarray(values), shape, padding_factor, half_volume=False, **kwargs
    )
    half = relion_functions._relion_reconstruct_floor_volume(
        jnp.asarray(_to_half(values)), shape, padding_factor, half_volume=True, **kwargs
    )

    np.testing.assert_allclose(np.asarray(full), expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(half), _to_half(expected), rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(half), _to_half(np.asarray(full)), rtol=1e-12, atol=0)


@pytest.mark.parametrize("size, padding_factor", [(15, 2), (16, 2)])
def test_traced_max_res_shell_serves_every_shell_with_one_program(size, padding_factor):
    shape = (size,) * 3
    values = _anisotropic_filter(size)
    bound = size // (2 * padding_factor)
    traces = []

    @jax.jit
    def floor(half_filter, max_res_shell):
        traces.append(1)
        return relion_functions._relion_reconstruct_floor_volume(
            half_filter,
            shape,
            padding_factor,
            half_volume=True,
            max_res_shell=max_res_shell,
            max_res_shell_bound=bound,
        )

    half_filter = jnp.asarray(_to_half(values))
    for max_res_shell in range(1, bound + 1):
        expected = _to_half(_relion_floor_volume(values, padding_factor, max_res_shell))
        np.testing.assert_allclose(
            np.asarray(floor(half_filter, jnp.int32(max_res_shell))), expected, rtol=1e-12, atol=0
        )
    assert len(traces) == 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_floor_volume_keeps_the_filter_dtype_and_layout(dtype):
    size, padding_factor = 15, 2
    shape = (size,) * 3
    half_filter = _to_half(_anisotropic_filter(size, dtype=dtype))
    for given in (half_filter, half_filter.reshape(-1)):
        floor = relion_functions._relion_reconstruct_floor_volume(
            jnp.asarray(given), shape, padding_factor, half_volume=True, max_res_shell=3
        )
        assert floor.dtype == dtype
        assert floor.shape == given.shape


@pytest.mark.parametrize("size, padding_factor", [(15, 2), (16, 2)])
def test_non_finite_filter_outside_the_averaged_shells_does_not_reach_the_floor(size, padding_factor):
    """Only shells below ``max_res_shell`` are averaged; a value outside them is masked, not weighted by zero."""

    shape = (size,) * 3
    max_res_shell = 2
    values = _anisotropic_filter(size)
    expected = _to_half(_relion_floor_volume(values, padding_factor, max_res_shell))
    half_filter = _to_half(values).copy()
    # The corner (-k_max, -k_max, k_max) of the native half lies beyond every averaged shell.
    half_filter[0, 0, -1] = np.inf

    half = relion_functions._relion_reconstruct_floor_volume(
        jnp.asarray(half_filter), shape, padding_factor, half_volume=True, max_res_shell=max_res_shell
    )

    assert np.all(np.isfinite(np.asarray(half)))
    np.testing.assert_allclose(np.asarray(half), expected, rtol=1e-12, atol=0)


def test_floor_volume_single_precision_matches_relion_x_half_rule():
    """Float32 filter, as the large-grid route passes: tolerance limited by single precision.

    ``test_floor_volume_matches_relion_x_half_rule_in_both_layouts`` is the float64 companion.
    """

    size, padding_factor, max_res_shell = 27, 2, 5
    values = _anisotropic_filter(size, dtype=np.float32)
    expected = _relion_floor_volume(values.astype(np.float64), padding_factor, max_res_shell)
    half = relion_functions._relion_reconstruct_floor_volume(
        jnp.asarray(_to_half(values)), (size,) * 3, padding_factor, half_volume=True, max_res_shell=max_res_shell
    )
    np.testing.assert_allclose(np.asarray(half), _to_half(expected), rtol=1e-5, atol=0)


@pytest.mark.parametrize("size, padding_factor", [(15, 2), (27, 2), (16, 2)])
def test_regularized_filter_agrees_between_full_and_native_half(size, padding_factor):
    """Through ``adjust_regularization_relion_style``, where the floor replaces empty voxels."""

    shape = (size,) * 3
    values = _anisotropic_filter(size)
    k = _centered_frequencies(size)
    # Unsampled voxels, symmetric under k -> -k, so the floor is what the caller divides by.
    empty = (np.abs(k)[:, None, None] + np.abs(k)[None, :, None] + np.abs(k)[None, None, :]) % 3 == 0
    values = np.where(empty, 0.0, values)
    max_res_shell = size // (2 * padding_factor) - 1
    kwargs = dict(padding_factor=padding_factor, max_res_shell=max_res_shell, relion_native_shell_floor=True)

    full = relion_functions.adjust_regularization_relion_style(jnp.asarray(values), shape, half_volume=False, **kwargs)
    half = relion_functions.adjust_regularization_relion_style(
        jnp.asarray(_to_half(values)), shape, half_volume=True, **kwargs
    )

    expected = np.maximum(values, _relion_floor_volume(values, padding_factor, max_res_shell))
    assert np.any(expected > values)
    np.testing.assert_allclose(np.asarray(full), expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(half), _to_half(np.asarray(full)), rtol=1e-12, atol=0)
