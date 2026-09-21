"""The memoised scaled half lattice must carry RELION's row labels.

The lattice was byte-identical to ``get_k_coordinate_of_each_pixel_half``
until P4-B (2026-09-20). That core helper labels the packed Nyquist row
``ky = -N/2``, RECOVAR's centered convention; RELION labels the same physical
row of an uncropped half image ``+N/2`` (``fftw.h:99-109``), and the EM
translate and scoring kernels all use RELION's label. A phase built on the
centered label is therefore the conjugate of RELION's on that row for any
shift that is not a whole pixel. The EM lattice now applies RELION's label and
the core helper keeps its own for its non-EM callers.
"""

import numpy as np
import jax
import jax.numpy as jnp

from recovar.core import fourier_transform_utils
from recovar.em.helpers.preprocessing import relion_half_translation_lattice
from recovar.em.sparse_pass2 import sparse_pass2_bucket_io as bio


def _core_lattice(shape):
    return np.asarray(
        fourier_transform_utils.get_k_coordinate_of_each_pixel_half(
            shape, voxel_size=1, scaled=True
        )
    )


def test_cached_lattice_matches_the_relion_labelled_lattice_and_is_reused():
    shape = (16, 16)
    cached = bio._scaled_half_lattice_cached(shape)
    np.testing.assert_array_equal(
        np.asarray(cached), np.asarray(relion_half_translation_lattice(shape))
    )
    assert bio._scaled_half_lattice_cached(shape) is cached  # memoised
    assert bio._scaled_half_lattice_cached((16, 16)) is cached  # same key for equal shapes


def test_phase_table_unchanged_by_memoisation():
    shape = (16, 16)
    translations = jnp.asarray([[0.0, 0.0], [1.5, -2.0], [0.25, 3.0]], dtype=jnp.float32)
    pixel_indices = jnp.arange(0, 16 * 9, 7, dtype=jnp.int32)
    got = bio._half_translation_phase_table_for_indices(translations, shape, pixel_indices)
    lattice = bio._scaled_half_lattice_cached(shape)
    want = jnp.exp(
        -2j
        * jnp.pi
        * jnp.einsum(
            "td,pd->tp",
            translations,
            lattice[pixel_indices],
            precision=jax.lax.Precision.HIGHEST,
        )
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_only_the_packed_nyquist_row_differs_from_the_core_lattice():
    """Every other row keeps the core helper's label, at several box sizes."""

    for size in (8, 16, 32, 64, 256):
        shape = (size, size)
        core = _core_lattice(shape)
        relion = np.asarray(relion_half_translation_lattice(shape))
        ky = np.rint(core[:, 1] * size).astype(int)
        nyquist = ky == -(size // 2)
        assert nyquist.any(), size
        np.testing.assert_array_equal(relion[~nyquist], core[~nyquist])
        np.testing.assert_array_equal(relion[nyquist, 1], -core[nyquist, 1])
        np.testing.assert_array_equal(relion[nyquist, 0], core[nyquist, 0])
        # RELION's labels for an uncropped half image run -N/2+1 .. +N/2.
        labels = np.rint(relion[:, 1] * size).astype(int)
        assert labels.min() == -(size // 2) + 1 and labels.max() == size // 2


def test_a_cropped_window_never_selects_the_relabelled_row():
    """The label only matters where RELION keeps that row: the uncropped box.

    ``windowFourierTransform`` keeps ``ip = -(cs/2-1)..+cs/2``
    (``fftw.h:849-855``), so every current size below the box drops the packed
    Nyquist row from the scoring window; at the box it is kept, which is the
    only place the two labels can disagree.
    """

    from recovar.em.helpers.fourier_window import (
        make_fourier_window_indices_np,
        make_frequency_coords_half_np,
    )

    size = 64
    coords = make_frequency_coords_half_np((size, size))
    ky = np.rint(coords[:, 1]).astype(int)
    nyquist = ky == -(size // 2)
    assert nyquist.any()
    for current_size in (16, 32, 48, 62):
        window, _ = make_fourier_window_indices_np((size, size), current_size)
        selected = np.zeros(coords.shape[0], dtype=bool)
        selected[window] = True
        assert not np.any(selected & nyquist), current_size
    # At the box there is no crop: the unwindowed consumer scores the whole
    # packed half, that row included, which is where the label matters. (What
    # the radial window does at the box belongs to the window branch, which
    # this repair does not depend on.)


def test_the_dense_batch_preparation_consumes_the_relabelled_table():
    """The dense engine's batch inputs must carry RELION's label too.

    ``_dense_batch_half_inputs`` builds its translation phases with the same EM
    helper, so the dense path at ``current_size == ori_size`` gets the repaired
    table rather than the centered one.
    """

    import inspect

    from recovar.em.helpers import preprocessing

    source = inspect.getsource(preprocessing._dense_batch_half_inputs)
    assert "half_translation_phase_table(" in source
    table_source = inspect.getsource(preprocessing.half_translation_phase_table)
    assert "relion_half_translation_lattice(image_shape)" in table_source
    assert "get_k_coordinate_of_each_pixel_half" not in table_source

    shape = (16, 16)
    translations = jnp.asarray([[0.0, 0.5]], dtype=jnp.float32)
    table = np.asarray(preprocessing.half_translation_phase_table(translations, shape))
    core = _core_lattice(shape)
    ky = np.rint(core[:, 1] * shape[0]).astype(int)
    nyquist = np.flatnonzero(ky == -(shape[0] // 2))
    centered = np.exp(
        -2j * np.pi * (core[nyquist, 0] * 0.0 + core[nyquist, 1] * 0.5)
    ).astype(np.complex64)
    np.testing.assert_allclose(
        table[0, nyquist], np.conj(centered), rtol=2e-6, atol=2e-6
    )
