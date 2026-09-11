"""Both centered-row RELION projectors take their FFTW block from one owner."""

import inspect

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers import projection


def test_fftw_block_clamps_the_projector_size_and_transposes_rotations():
    n = 8
    rng = np.random.default_rng(1)
    volume = jnp.asarray((rng.normal(size=(n, n, n // 2 + 1)) + 1j * rng.normal(size=(n, n, n // 2 + 1))).astype(np.complex128))
    rot = jnp.asarray(np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])[None])
    block, size = projection._relion_projector_fftw_block(volume, rot, n, n // 2, 1, None, False)
    assert size == n and block.shape == (1, n, n // 2 + 1)
    raw = projection.project_relion_projector_half_spectrum(volume, jnp.swapaxes(rot, -1, -2), (n, n), n // 2, 1, False)
    assert np.array_equal(np.asarray(block).reshape(1, -1), np.asarray(raw).reshape(1, -1))
    cropped, size = projection._relion_projector_fftw_block(volume, rot, n, 3, 1, None, False)
    assert size == 6 and cropped.shape == (1, 6, 4)
    _, size = projection._relion_projector_fftw_block(volume, rot, n, 3, 1, 4, False)
    assert size == 4
    _, size = projection._relion_projector_fftw_block(volume, rot, n, 3, 1, 12, False)
    assert size == n


def test_centered_row_projectors_use_the_owner():
    for fn in (
        projection.project_relion_projector_half_spectrum_centered_rows,
        projection.project_relion_projector_half_spectrum_centered_rows_at_indices,
    ):
        source = inspect.getsource(fn)
        assert source.count("proj_fftw, projector_image_size = _relion_projector_fftw_block(") == 1
        assert "jnp.swapaxes(rotations_block" not in source
        assert "project_relion_projector_half_spectrum(" not in source
