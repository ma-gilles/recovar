"""Ordered EM prefetch must preserve outputs and release stopped producers."""

import threading

import pytest

from recovar.em.helpers.batch_fetch import prefetched_batches, prefetch_depth

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_order_and_producer_error(depth):
    def source():
        yield 1
        yield 2
        raise RuntimeError("producer failed")

    values = []
    with pytest.raises(RuntimeError, match="producer failed"):
        with prefetched_batches(source(), depth=depth) as batches:
            values.extend(batches)
    assert values == [1, 2]
    with prefetched_batches(range(20), depth=depth) as batches:
        assert list(batches) == list(range(20))


@pytest.mark.parametrize("consumer_error", [False, True])
def test_exit_releases_full_queue(consumer_error):
    blocked = threading.Event()
    stopped = threading.Event()

    def source():
        try:
            yield 0
            yield 1
            blocked.set()
            yield 2
        finally:
            stopped.set()

    try:
        with prefetched_batches(source(), depth=1) as batches:
            assert next(batches) == 0
            assert blocked.wait(timeout=2)
            if consumer_error:
                raise RuntimeError("consumer failed")
    except RuntimeError as exc:
        assert consumer_error and str(exc) == "consumer failed"
    assert stopped.wait(timeout=2)


@pytest.mark.parametrize("value", ["-1", "invalid", "1.5"])
def test_invalid_depth_rejected(monkeypatch, value):
    monkeypatch.setenv("RECOVAR_EM_PREFETCH_BATCHES", value)
    with pytest.raises(ValueError, match="non-negative integer"):
        prefetch_depth()


@pytest.mark.parametrize("device_scalars", [False, True])
def test_fused_pass2_prefetch_preserves_all_outputs(monkeypatch, device_scalars):
    from test_compact_real_rows_integration import (
        _fused_kclass_multibucket_fixture,
        _fused_kclass_result_arrays,
        _assert_fused_arrays_identical,
    )
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as engine

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", "4")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_DEVICE_CHUNK_SCALARS", str(int(device_scalars)))
    kwargs = _fused_kclass_multibucket_fixture(n_images=13)
    kwargs.update(accumulate_noise=True, relion_f32_fine_posterior=True)
    threads = []
    original = engine.fetch_indexed_batch

    def capture(*args, **kw):
        threads.append(threading.get_ident())
        return original(*args, **kw)

    monkeypatch.setattr(engine, "fetch_indexed_batch", capture)
    monkeypatch.setenv("RECOVAR_EM_PREFETCH_BATCHES", "0")
    expected = _fused_kclass_result_arrays(engine.compute_k_class_pass2_stats_sparse_fused(**kwargs))
    assert threads and set(threads) == {threading.get_ident()}
    threads.clear()
    monkeypatch.setenv("RECOVAR_EM_PREFETCH_BATCHES", "2")
    actual = _fused_kclass_result_arrays(engine.compute_k_class_pass2_stats_sparse_fused(**kwargs))
    assert threads and threading.get_ident() not in threads
    _assert_fused_arrays_identical(expected, actual, "prefetch")


def test_coarse_prefetch_preserves_all_outputs(monkeypatch):
    from test_em_stage_glue_programs import _significance_call, _assert_significance_results_identical
    from recovar.em.scoring import significance

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    args, kwargs = _significance_call()
    threads = []
    original = args[0].iter_batches

    def capture(*a, **kw):
        for batch in original(*a, **kw):
            threads.append(threading.get_ident())
            yield batch

    monkeypatch.setattr(args[0], "iter_batches", capture)
    monkeypatch.setenv("RECOVAR_EM_PREFETCH_BATCHES", "0")
    expected = significance._compute_k_class_significance_batched(*args, **kwargs)
    assert threads and set(threads) == {threading.get_ident()}
    threads.clear()
    monkeypatch.setenv("RECOVAR_EM_PREFETCH_BATCHES", "2")
    actual = significance._compute_k_class_significance_batched(*args, **kwargs)
    assert threads and threading.get_ident() not in threads
    _assert_significance_results_identical(actual, expected)
