"""RELION ``powerClass`` reproductions share one operand owner.

``_relion_powerclass_packed_image`` repacks centred rfft images into RELION's
unshifted ``Faux`` layout with RELION's amplitude convention,
``_relion_powerclass_operands`` adds the CUDA shell map and pixel validity,
and ``_relion_powerclass_native_spectrum_highres`` feeds the native kernels.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import sparse_pass2_scoring as sp

from recovar.em.dense_single_volume.helpers import sparse_pass2_scoring

pytestmark = pytest.mark.unit

SIZE = 8
HALF = SIZE // 2 + 1


def _images(dtype, n=2):
    rng = np.random.default_rng(1)
    return (rng.standard_normal((n, SIZE * HALF)) + 1j * rng.standard_normal((n, SIZE * HALF))).astype(dtype)


@pytest.mark.parametrize("dtype,expected_real", [(np.complex64, jnp.float32), (np.complex128, jnp.float64)])
def test_packed_image_follows_relion_layout_and_amplitude(dtype, expected_real):
    x = _images(dtype)
    image, real_dtype, height, width, half = sparse_pass2_scoring._relion_powerclass_packed_image(x, image_shape=(SIZE, SIZE))
    assert (height, width, half) == (SIZE, SIZE, HALF) and real_dtype == expected_real
    expected = np.roll(x.reshape(-1, SIZE, HALF), -(SIZE // 2), axis=1).reshape(x.shape[0], -1) / (SIZE * SIZE)
    np.testing.assert_allclose(np.asarray(image), expected.astype(dtype), rtol=0, atol=0)
    assert image.dtype == jnp.dtype(dtype)


def test_forced_complex64_matches_native_kernel_precision():
    image, real_dtype, *_ = sparse_pass2_scoring._relion_powerclass_packed_image(_images(np.complex128), image_shape=(SIZE, SIZE), dtype=jnp.complex64)
    assert image.dtype == jnp.complex64 and real_dtype == jnp.float32


@pytest.mark.parametrize("bad,shape,match", [
    (np.zeros((2, 40), np.complex64), (8, 10), "square images"),
    (np.zeros((2, 39), np.complex64), (8, 8), "flattened centred rfft"),
    (np.zeros((40,), np.complex64), (8, 8), "flattened centred rfft"),
])
def test_packed_image_rejects_wrong_layouts(bad, shape, match):
    with pytest.raises(ValueError, match=match):
        sparse_pass2_scoring._relion_powerclass_packed_image(bad, image_shape=shape)


def test_resolution_limit_is_static_or_traced():
    assert sparse_pass2_scoring._relion_powerclass_resolution_limit(None, None, image_width=SIZE) == SIZE // 2 + 1
    assert sparse_pass2_scoring._relion_powerclass_resolution_limit(6, None, image_width=SIZE) == 4
    traced = sparse_pass2_scoring._relion_powerclass_resolution_limit(6, np.asarray(4, dtype=np.int32), image_width=SIZE)
    assert isinstance(traced, jnp.ndarray) and int(traced) == 3


def test_operands_shell_and_validity_follow_the_cuda_kernel():
    ops = sparse_pass2_scoring._relion_powerclass_operands(_images(np.complex64), image_shape=(SIZE, SIZE), current_size=None, runtime_current_size=None)
    rows = np.arange(SIZE)[:, None]
    cols = np.arange(HALF)[None, :]
    signed = np.where(rows < HALF, rows, rows - SIZE)
    shell = np.rint(np.sqrt((cols * cols + signed * signed).astype(np.float32))).astype(np.int32)
    assert ops.shell.dtype == np.int32 and np.array_equal(ops.shell, shell)
    valid = ((shell > 0) & (shell < HALF) & ~((cols == 0) & (signed < 0))).reshape(-1)
    assert np.array_equal(ops.valid, valid) and ops.resolution_limit == HALF
    assert ops.relion_image.shape == (2, SIZE * HALF)


def test_reproductions_and_native_wrappers_use_the_owners():
    for name in ("_relion_cuda_powerclass_highres_xi2_half", "_relion_cuda_powerclass_spectrum_highres_norm_units"):
        source = inspect.getsource(getattr(sp, name))
        assert source.count("_relion_powerclass_operands(") == 1 and "jnp.roll(" not in source
    for name in ("_relion_cuda_powerclass_highres_xi2_half_atomic", "_relion_cuda_powerclass_spectrum_norm_units"):
        source = inspect.getsource(getattr(sp, name))
        assert source.count("_relion_powerclass_native_spectrum_highres(") == 1 and "cuda_backproject" not in source
    module_source = inspect.getsource(sp)
    assert module_source.count("-(image_height // 2),") == 1
