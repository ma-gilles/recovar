"""The memoised scaled half lattice must be byte-identical to the direct device computation."""

import numpy as np
import jax.numpy as jnp

from recovar.core import fourier_transform_utils
from recovar.em.sparse_pass2 import sparse_pass2_bucket_io as bio


def test_cached_lattice_matches_direct_and_is_reused():
    shape = (16, 16)
    direct = np.asarray(fourier_transform_utils.get_k_coordinate_of_each_pixel_half(shape, voxel_size=1, scaled=True))
    cached = bio._scaled_half_lattice_cached(shape)
    np.testing.assert_array_equal(np.asarray(cached), direct)
    assert bio._scaled_half_lattice_cached(shape) is cached  # memoised
    assert bio._scaled_half_lattice_cached((16, 16)) is cached  # same key for equal shapes


def test_phase_table_unchanged_by_memoisation():
    shape = (16, 16)
    translations = jnp.asarray([[0.0, 0.0], [1.5, -2.0], [0.25, 3.0]], dtype=jnp.float32)
    pixel_indices = jnp.arange(0, 16 * 9, 7, dtype=jnp.int32)
    got = bio._half_translation_phase_table_for_indices(translations, shape, pixel_indices)
    lattice = jnp.asarray(fourier_transform_utils.get_k_coordinate_of_each_pixel_half(shape, voxel_size=1, scaled=True))
    import jax
    want = jnp.exp(-2j * jnp.pi * jnp.einsum("td,pd->tp", translations, lattice[pixel_indices], precision=jax.lax.Precision.HIGHEST))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
