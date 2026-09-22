"""Bounded dispatch waits preserve all fused outputs."""
import numpy as np
import pytest
from recovar.em.sparse_pass2.sparse_pass2_policy import sparse_kclass_pipeline_depth
from test_compact_capacity_integration import _fused_kclass_capacity_fixture, _fused_kclass_result_arrays, bucketed_mod

@pytest.mark.parametrize('value,expected', [(None,4),('0',0),('1',1),('7',7)])
def test_depth_policy(monkeypatch, value, expected):
    monkeypatch.delenv('RECOVAR_SPARSE_KCLASS_PIPELINE_DEPTH', raising=False)
    if value is not None:
        monkeypatch.setenv('RECOVAR_SPARSE_KCLASS_PIPELINE_DEPTH', value)
    assert sparse_kclass_pipeline_depth() == expected

@pytest.mark.parametrize('value', ['-1','typo','1.5'])
def test_invalid_depth(monkeypatch, value):
    monkeypatch.setenv('RECOVAR_SPARSE_KCLASS_PIPELINE_DEPTH', value)
    with pytest.raises(ValueError):
        sparse_kclass_pipeline_depth()

@pytest.mark.parametrize('noise', [False, True])
def test_throttle_executes_and_preserves_outputs(monkeypatch, noise):
    monkeypatch.setenv('RECOVAR_DISABLE_CUDA', '1')
    monkeypatch.setenv('RECOVAR_SPARSE_KCLASS_DEFERRED_HOST_STATS', '1')
    monkeypatch.setenv('RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS', '1')
    monkeypatch.setenv('RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE', '1')
    kwargs = _fused_kclass_capacity_fixture()
    kwargs['accumulate_noise'] = noise
    original = bucketed_mod.jax.block_until_ready
    waits = []
    def record(value):
        waits.append(1)
        return original(value)
    monkeypatch.setattr(bucketed_mod.jax, 'block_until_ready', record)
    monkeypatch.setenv('RECOVAR_SPARSE_KCLASS_PIPELINE_DEPTH', '0')
    expected = _fused_kclass_result_arrays(bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs))
    baseline_waits = len(waits)
    waits.clear()
    monkeypatch.setenv('RECOVAR_SPARSE_KCLASS_PIPELINE_DEPTH', '1')
    actual = _fused_kclass_result_arrays(bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs))
    assert len(waits) > baseline_waits
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
