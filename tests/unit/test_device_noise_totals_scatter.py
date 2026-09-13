"""Device noise-total accumulation must equal the host ``np.add.at`` with padded duplicate image indices.

Jobs 13837222/13837223 (2026-09-13) failed RELION's non-negative norm-correction guard when the
device accumulator scattered padded rows (which repeat the last real image's index) with
``unique_indices=True``.  These tests pin the accumulator against the host order on the
default device and, marked ``gpu``, on the accelerator where the unique-index hint is exploited.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as bucketed_mod


def _host_reference(wsum, norm, image_indices, shells, residual):
    wsum = wsum.copy(); norm = norm.copy()
    wsum += np.asarray(shells, dtype=np.float64)
    np.add.at(norm, image_indices, np.asarray(residual, dtype=np.float64))
    return wsum, norm


def _case(rng, n_images, n_real, n_alloc, n_shells):
    real = np.sort(rng.choice(n_images, size=n_real, replace=False)).astype(np.int64)
    image_indices = np.concatenate([real, np.repeat(real[-1:], n_alloc - n_real)])
    shells = rng.standard_normal(n_shells).astype(np.float32)
    # padded rows carry non-zero residuals of both signs, as a masked copy of the last image may
    residual = rng.standard_normal(n_alloc).astype(np.float32) * np.array([1.0] * n_real + [3.0] * (n_alloc - n_real), dtype=np.float32)
    wsum = rng.standard_normal(n_shells)
    norm = rng.standard_normal(n_images)
    return image_indices, shells, residual, wsum, norm


def _run_device(device, image_indices, shells, residual, wsum, norm, n_real):
    with jax.default_device(device):
        w, n = bucketed_mod._accumulate_noise_totals_device(
            jnp.asarray(wsum, dtype=jnp.float64),
            jnp.asarray(norm, dtype=jnp.float64),
            jnp.asarray(image_indices, dtype=jnp.int32),
            jnp.int32(n_real),
            jnp.asarray(shells),
            jnp.asarray(residual),
        )
        return np.asarray(w), np.asarray(n)


@pytest.mark.parametrize("seed", range(4))
def test_device_noise_totals_match_host_add_at_with_padded_duplicates(seed):
    rng = np.random.default_rng(seed)
    image_indices, shells, residual, wsum, norm = _case(rng, n_images=40, n_real=11, n_alloc=16, n_shells=9)
    expected = _host_reference(wsum, norm, image_indices, shells, residual)
    got = _run_device(jax.devices("cpu")[0], image_indices, shells, residual, wsum, norm, 11)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


@pytest.mark.gpu
@pytest.mark.parametrize("seed", range(4))
def test_device_noise_totals_match_host_add_at_on_gpu(gpu_device, seed):
    rng = np.random.default_rng(100 + seed)
    image_indices, shells, residual, wsum, norm = _case(rng, n_images=300, n_real=27, n_alloc=32, n_shells=129)
    expected = _host_reference(wsum, norm, image_indices, shells, residual)
    got = _run_device(gpu_device, image_indices, shells, residual, wsum, norm, 27)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


@pytest.mark.gpu
def test_unique_index_scatter_with_duplicates_is_not_the_host_sum_on_gpu(gpu_device):
    """Documents the defect: the unique-index hint with repeated indices does not reproduce
    the host sum on the accelerator (job 13837222). Skipped, not failed, if the backend happens
    to reproduce it, so the pin above remains the contract."""
    rng = np.random.default_rng(7)
    idx = np.concatenate([np.arange(27), np.repeat(26, 5)]).astype(np.int32)
    vals = rng.standard_normal(32)
    expected = np.zeros(300); np.add.at(expected, idx, vals)
    with jax.default_device(gpu_device):
        got = np.asarray(jnp.zeros(300, jnp.float64).at[jnp.asarray(idx)].add(jnp.asarray(vals), unique_indices=True))
    if np.array_equal(got, expected):
        pytest.skip("this backend reproduces the host sum despite the unique-index hint")
    assert not np.array_equal(got, expected)
