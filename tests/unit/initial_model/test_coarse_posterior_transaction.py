"""Explicit CUDA transaction: interface checks and same-score GPU oracle."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cuda
from recovar.em.dense_single_volume.helpers.coarse_publication import _posterior_statistics

pytestmark = pytest.mark.unit


def operands(shape=(3, 17)):
    return (jax.ShapeDtypeStruct(shape, jnp.float32),
            jax.ShapeDtypeStruct((shape[0],), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.int32))


def test_policy_keeps_parsed_float32_fraction_and_unlimited_support():
    fraction, maximum = cuda._validate_coarse_posterior_transaction(*operands(), .999, None)
    assert fraction.dtype == np.float32 and fraction == np.float32(.999)
    assert maximum == 0


@pytest.mark.parametrize('which,value,error', [
    (0, jax.ShapeDtypeStruct((3,17), jnp.float64), TypeError),
    (0, jax.ShapeDtypeStruct((0,17), jnp.float32), ValueError),
    (0, jax.ShapeDtypeStruct((65536,65536), jnp.float32), ValueError),
    (1, jax.ShapeDtypeStruct((1,), jnp.float32), TypeError),
    (2, jax.ShapeDtypeStruct((1,), jnp.int32), TypeError),
    (2, jax.ShapeDtypeStruct((), jnp.int64), TypeError),
    (3, float('nan'), ValueError), (3, 0, ValueError), (3, 1.01, ValueError),
    (4, -1, ValueError), (4, 1.5, TypeError), (4, 2**31, ValueError),
])
def test_invalid_contract_rejected_before_gpu_dispatch(which, value, error):
    args = [*operands(), .999, 500]
    args[which] = value
    with pytest.raises(error):
        cuda._validate_coarse_posterior_transaction(*args)


@pytest.mark.gpu
@pytest.mark.parametrize('case,width,maxsig', [
    ('random', 257, 19), ('ties', 1025, 3), ('underflow', 1025, None),
    ('empty', 257, 7), ('nonfinite', 257, 7), ('offset', 13312, 500),
])
def test_same_scores_match_cuda_posterior_bitwise(monkeypatch, case, width, maxsig):
    assert jax.default_backend() == 'gpu'
    assert cuda.cuda_available() and cuda.custom_cuda_requested()
    monkeypatch.setenv('RECOVAR_RELION_BATCHED_POSTERIOR_PRIMITIVES', '1')
    rng = np.random.default_rng(293)
    values = rng.normal(-100, 4, (4,width)).astype(np.float32)
    raw_max = np.max(values, axis=1)
    if case == 'ties':
        values.fill(-7); raw_max.fill(-7)
    elif case == 'underflow':
        values[:] = np.linspace(-150, 0, width, dtype=np.float32); raw_max.fill(0)
    elif case == 'empty':
        values[0] = -np.inf; raw_max[0] = 0
    elif case == 'nonfinite':
        values[0,0] = np.nan; values[1,0] = np.inf
    elif case == 'offset':
        values += np.float32(-12345.5); raw_max += np.float32(-12345.5)
    values[-1] = -np.inf; raw_max[-1] = 0  # padded row
    scores, maxima = jnp.asarray(values), jnp.asarray(raw_max)
    reference = jax.device_get(_posterior_statistics(scores, maxima, None,
        adaptive_fraction=.999, max_significants=maxsig, tie_score_ulps=0))
    statistics, indices, support, count = jax.device_get(cuda.relion_coarse_posterior_transaction_f32(
        scores, maxima, jnp.asarray(3, jnp.int32), adaptive_fraction=.999, max_significants=maxsig))
    for column, key in enumerate(('best_score','pmax','sum_weight','threshold')):
        want = np.asarray(reference[key], np.float32)
        np.testing.assert_array_equal(statistics[:,column], want, err_msg=key)
        finite = np.isfinite(want)
        np.testing.assert_array_equal(statistics[:,column][finite].view(np.uint32), want[finite].view(np.uint32), err_msg=key)
    for column, key in enumerate(('best_pose','winner','n_significant','cutoff_count')):
        np.testing.assert_array_equal(indices[:,column], reference[key], err_msg=key)
    expected = np.flatnonzero(reference['mask'].reshape(-1)).astype(np.int32)
    assert count == len(expected) == indices[:,2].sum()
    np.testing.assert_array_equal(support[:int(count)], expected)
    assert np.all(support[int(count):] == -1)
    if case == 'ties':
        assert np.all(indices[:3,2] == width)  # maxsig cannot truncate exact ties
