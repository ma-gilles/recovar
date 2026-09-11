"""``_packed_reconstruction_rows`` is the one gather-and-zero of packed reconstruction rows in the local engine."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume import local_em_engine


def test_owner_gathers_rows_and_zeroes_padding():
    values = jnp.asarray(np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2))
    take = jnp.asarray([[2, 0], [1, 1]], dtype=jnp.int32)
    mask = jnp.asarray([[True, False], [True, True]])
    out = local_em_engine._packed_reconstruction_rows(values, take, mask)
    assert out.shape == (2, 2, 2)
    assert out[0, 0].tolist() == values[0, 2].tolist() and out[0, 1].tolist() == [0.0, 0.0]
    assert out[1, 0].tolist() == values[1, 1].tolist() and out[1, 1].tolist() == values[1, 1].tolist()
    expected = jnp.where(mask[:, :, None], jnp.take_along_axis(values, take[:, :, None], axis=1), 0.0)

    def inline(v, t, m):
        packed = jnp.take_along_axis(v, t[:, :, None], axis=1)
        return jnp.where(m[:, :, None], packed, 0.0)

    assert str(jax.make_jaxpr(local_em_engine._packed_reconstruction_rows)(values, take, mask)) == str(
        jax.make_jaxpr(inline)(values, take, mask)
    )
    assert np.array_equal(np.asarray(out), np.asarray(expected))


def test_local_engine_sites_use_the_owner():
    src = inspect.getsource(local_em_engine.run_local_em_exact)
    assert src.count("_packed_reconstruction_rows(") == 11
    # no remaining inline gather-then-zero of a (batch, rotation, pixel) operand on the packed take indices
    # the one remaining inline take feeds a conditional zeroing (source-VDAM ctf probs) and stays
    assert src.count("reconstruction_take_indices_jnp[:, :, None]") == 1
    assert src.count("chunk_take_indices[:, :, None]") == 0
