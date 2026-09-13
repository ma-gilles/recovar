"""Host row plans preserve bit patterns when composed into one JIT."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.helpers.deferred_vdam_host_pack import _gather_deferred_vdam_host_plan, pack_deferred_vdam_host_plan

pytestmark = pytest.mark.unit


def _case(batch):
    rng = np.random.default_rng(643)
    posterior = rng.normal(size=(7, 16, 3)).astype(np.float32)
    sums = rng.normal(size=(7, 16)).astype(np.float32)
    images = (rng.normal(size=(7, 11)) + 1j * rng.normal(size=(7, 11))).astype(np.complex64)
    ctf = rng.normal(size=(7, 11)).astype(np.float32)
    inverse = rng.uniform(size=(7, 11)).astype(np.float32)
    projections = (rng.normal(size=(31, 11)) + 1j * rng.normal(size=(31, 11))).astype(np.complex64)
    take = rng.integers(1, 16, size=(batch, 4), dtype=np.int32)
    flat = rng.integers(1, 31, size=(batch, 4), dtype=np.int32)
    take[:, 0] = 1
    flat[:, 0] = 1
    posterior[:, 1, 0] = np.float32(-0.0)
    sums[:, 1] = np.float32(-0.0)
    projections[1, 0] = np.array([0x80000000, 0x80000000], dtype=np.uint32).view(np.complex64)[0]
    mask = np.ones((batch, 4), dtype=bool)
    mask[:, -1] = False
    mask[0] = False
    take[~mask] = 0
    flat[~mask] = 0
    # Masked padding must not leak NaNs; fringe images must be sliced away.
    posterior[:, 0] = np.nan
    sums[:, 0] = np.nan
    projections[0] = complex(np.nan, np.nan)
    return tuple(map(jnp.asarray, (posterior, sums, images, ctf, inverse, projections, take, mask, flat)))


def _numpy_expected(args):
    posterior, sums, images, ctf, inverse, projection, take, mask, flat = map(np.asarray, args)
    batch = len(take)
    return (
        np.where(mask[..., None], np.take_along_axis(posterior[:batch], take[..., None], axis=1), np.float32(0)),
        np.where(mask, np.take_along_axis(sums[:batch], take, axis=1), np.float32(0)),
        images[:batch],
        ctf[:batch],
        inverse[:batch],
        np.where(mask[..., None], projection[flat], np.complex64(0)),
    )


def _assert_bits(actual, expected):
    for got, want in zip(actual, expected, strict=True):
        got, want = np.asarray(got), np.asarray(want)
        assert got.shape == want.shape and got.dtype == want.dtype
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize("batch", [3, 7])
def test_host_plan_preserves_rows_masks_and_fringe_bits(batch):
    args = _case(batch)
    @jax.jit
    def compiled(*operands):
        return _gather_deferred_vdam_host_plan(*operands)
    _assert_bits(compiled(*args), _numpy_expected(args))
    changed = list(args)
    changed[6] = args[6][:, ::-1]
    changed[7] = args[7][:, ::-1]
    changed[8] = args[8][:, ::-1]
    _assert_bits(compiled(*changed), _numpy_expected(changed))
    assert compiled._cache_size() == 1


@pytest.mark.gpu
@pytest.mark.parametrize("batch", [3, 7])
def test_composed_denominator_matches_existing_eager_sequence_bitwise(batch):
    from recovar import cuda_backproject

    args = _case(batch)
    eager = _gather_deferred_vdam_host_plan(*args)
    denominator = cuda_backproject.relion_vdam_mstep_denominator_f32(eager[3], eager[4], eager[0])
    denominator = jnp.where(args[7][..., None], denominator, 0.0)
    _assert_bits(pack_deferred_vdam_host_plan(*args), (*eager, denominator))
