"""Shadow replay must detect stale transfers and changed bucket metadata."""
from typing import NamedTuple
import jax
import numpy as np
import pytest
from recovar.em.classification.k_class_results import DeferredHostUpdates
from recovar.em.sparse_pass2.sparse_pass2_policy import deferred_host_statistics_mode

class Stats(NamedTuple):
    totals: np.ndarray
    def add(self, *, image_indices, values):
        np.add.at(self.totals, image_indices, values)


def test_shadow_replay_is_bounded_and_duplicate_safe():
    target = Stats(np.zeros(3))
    queue = DeferredHostUpdates(max_records=2, check=True)
    for _ in range(5):
        queue.append(target.add, host=dict(image_indices=np.array([1, 1])), device=dict(values=np.array([2., 3.])))
        assert len(queue.check.records) < 2
    queue.flush()
    np.testing.assert_array_equal(target.totals, [0., 25., 0.])
    assert not queue.check.owners and not queue.check.records


def test_changed_host_metadata_is_named():
    target = Stats(np.zeros(3)); indices = np.array([0])
    queue = DeferredHostUpdates(check=True)
    queue.append(target.add, host=dict(image_indices=indices), device=dict(values=np.array([2.])))
    indices[0] = 1
    with pytest.raises(RuntimeError, match='HOST METADATA mutated: image_indices'):
        queue.flush()
    assert not target.totals.any()


def test_stale_device_leaf_is_named(monkeypatch):
    target = Stats(np.zeros(3)); queue = DeferredHostUpdates(check=True)
    queue.append(target.add, host=dict(image_indices=np.array([0])), device=dict(values=np.array([2.])))
    monkeypatch.setattr(jax, 'device_get', lambda _: [dict(values=np.array([3.]))])
    with pytest.raises(RuntimeError, match='DEVICE LEAF differs at replay: values'):
        queue.flush()
    assert not target.totals.any()


def test_shadow_output_difference_is_named():
    target = Stats(np.zeros(3)); queue = DeferredHostUpdates(check=True)
    queue.append(target.add, host=dict(image_indices=np.array([0])), device=dict(values=np.array([2.])))
    target.totals[2] = 1
    with pytest.raises(RuntimeError, match='shadow statistics differ: totals'):
        queue.flush()

@pytest.mark.parametrize('mode', ['0', '1', 'check'])
def test_mode(monkeypatch, mode):
    monkeypatch.setenv('RECOVAR_SPARSE_KCLASS_DEFERRED_HOST_STATS', mode)
    assert deferred_host_statistics_mode() == mode

def test_invalid_mode(monkeypatch):
    monkeypatch.setenv('RECOVAR_SPARSE_KCLASS_DEFERRED_HOST_STATS', 'typo')
    with pytest.raises(ValueError):
        deferred_host_statistics_mode()


@pytest.mark.parametrize('field', ['best_argmax', 'best_log_score_bucket', 'max_posterior_bucket', 'class_log_z', 'probs_sum_t_jax'])
def test_statistics_row_shape_guard_names_bad_leaf(field):
    import inspect
    from recovar.em.classification.k_class_results import SparseKClassHostStatistics
    target = SparseKClassHostStatistics(*([None] * len(SparseKClassHostStatistics._fields)))
    kwargs = {name: None for name in inspect.signature(target.update_bucket).parameters}
    kwargs.update(class_index=0, batch=2, image_indices=np.arange(2))
    for name in ['best_argmax', 'best_log_score_bucket', 'max_posterior_bucket', 'class_log_z', 'probs_sum_t_jax']:
        kwargs[name] = np.zeros(3 if name == field else 2)
    with pytest.raises(RuntimeError, match='deferred statistics leaf .* bucket has 2 real images'):
        target.update_bucket(**kwargs)
