"""The three local debug dump writers select their rows through one owner."""

import inspect
from types import SimpleNamespace

import numpy as np

from recovar.em.dense_single_volume import local_debug


def _bucket(ids):
    return SimpleNamespace(image_indices=np.asarray(ids, dtype=np.int32))


def _dataset(calls):
    def original(ids):
        calls.append(np.asarray(ids).copy())
        return 100 + np.asarray(ids)

    return SimpleNamespace(original_image_indices_from_local=original)


def test_requested_dump_rows_gates_before_touching_the_dataset(tmp_path):
    calls = []
    common = dict(current_size=8, debug_iteration=9, requested_current_sizes=None, requested_iterations=None)
    assert local_debug._requested_dump_rows(_dataset(calls), _bucket([0, 1]), dump_dir=None, pending_targets={100}, **common) is None
    assert local_debug._requested_dump_rows(_dataset(calls), _bucket([0, 1]), dump_dir=tmp_path, pending_targets=set(), **common) is None
    assert local_debug._requested_dump_rows(
        _dataset(calls), _bucket([0, 1]), dump_dir=tmp_path, pending_targets={100},
        current_size=8, debug_iteration=9, requested_current_sizes={16}, requested_iterations=None,
    ) is None
    assert local_debug._requested_dump_rows(
        _dataset(calls), _bucket([0, 1]), dump_dir=tmp_path, pending_targets={100},
        current_size=8, debug_iteration=9, requested_current_sizes=None, requested_iterations={3},
    ) is None
    assert calls == []
    assert local_debug._requested_dump_rows(_dataset(calls), _bucket([0, 1]), dump_dir=tmp_path, pending_targets={555}, **common) is None
    assert len(calls) == 1


def test_requested_dump_rows_returns_original_ids_and_matching_rows(tmp_path):
    calls = []
    selected = local_debug._requested_dump_rows(
        _dataset(calls), _bucket([0, 1, 2]), dump_dir=tmp_path, pending_targets={101, 102, 999},
        current_size=8, debug_iteration=9, requested_current_sizes={8}, requested_iterations={9},
    )
    original, rows = selected
    assert original.dtype == np.int64 and original.tolist() == [100, 101, 102]
    assert rows == [1, 2]
    assert len(calls) == 1


def test_dump_writers_use_the_row_owner():
    for fn in (
        local_debug.maybe_write_debug_fused_posterior_dump,
        local_debug.maybe_write_debug_score_dump,
        local_debug.maybe_write_debug_noise_component_dump,
    ):
        source = inspect.getsource(fn)
        assert source.count("selected = _requested_dump_rows(") == 1
        assert "original_image_indices, target_rows = selected" in source
        assert "original_image_indices_from_local" not in source
        assert "current_size_matches_request(" not in source
        assert "requested_iterations is not None" not in source
