"""Coarse Pmax publication preserves active rows across physical batches."""

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.scoring import significance

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("active", [0, 1, 8, 24, 192, 200])
@pytest.mark.parametrize("tail", [0.0, 7.0, np.nan, np.inf, -np.inf])
def test_physical_pmax_preserves_active_bytes(active, tail):
    weights = np.random.default_rng(829).uniform(size=(200, 129)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    weights[0] = 0
    weights[1] = -np.arange(129, dtype=np.float32)
    weights[active:] = tail
    device_weights = jnp.asarray(weights)
    control = significance._coarse_max_posterior_for_host(device_weights, active)
    candidate = significance._coarse_max_posterior_for_host(
        device_weights, active, physical_batch=True,
    )
    assert candidate.shape == control.shape == (active,)
    assert candidate.dtype == control.dtype == np.dtype(np.float32)
    assert candidate.tobytes() == control.tobytes()
    np.testing.assert_array_equal(candidate, weights[:active].max(axis=1))


@pytest.mark.parametrize("token, expected", [(None, False), ("0", False), ("1", True), (" 1 ", True)])
def test_physical_pmax_selector(monkeypatch, token, expected):
    name = "RECOVAR_COARSE_MAX_POSTERIOR_PHYSICAL_BATCH"
    if token is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, token)
    assert significance._coarse_max_posterior_physical_batch_enabled() is expected


@pytest.mark.parametrize("token", ["", "2", "true", "false", "-1"])
def test_physical_pmax_selector_rejects_invalid(monkeypatch, token):
    monkeypatch.setenv("RECOVAR_COARSE_MAX_POSTERIOR_PHYSICAL_BATCH", token)
    with pytest.raises(ValueError, match="RECOVAR_COARSE_MAX_POSTERIOR_PHYSICAL_BATCH"):
        significance._coarse_max_posterior_physical_batch_enabled()


@pytest.mark.gpu
@pytest.mark.parametrize("width", [37888, 50176, 29696, 21504])
def test_physical_pmax_gpu_shapes(width, monkeypatch):
    """Measure real fringe widths and prove one reduction acquisition per arm."""
    import inspect
    import json
    import os
    import time
    from pathlib import Path

    import jax
    from jax._src import compiler

    assert jax.default_backend() == "gpu"
    active_sizes = (1, 8, 24, 40, 96, 192, 200)
    weights = np.random.default_rng(593).uniform(size=(200, width)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    weights[0] = 0
    device_weights = jax.device_put(weights)
    device_weights.block_until_ready()
    expected = weights.max(axis=1)
    original = compiler.compile_or_get_cached
    signature = inspect.signature(original)
    acquisitions = []

    def observe(*args, **kwargs):
        computation = signature.bind(*args, **kwargs).arguments["computation"]
        acquisitions.append({
            "module": compiler.ir.StringAttr(computation.operation.attributes["sym_name"]).value,
            "types": [str(op.attributes["function_type"]) for op in computation.body.operations
                      if "function_type" in op.attributes],
        })
        return original(*args, **kwargs)

    monkeypatch.setattr(compiler, "compile_or_get_cached", observe)
    panels = []
    for physical in (False, True, True, False):
        jax.clear_caches()
        acquisitions.clear()
        for active in active_sizes:
            actual = significance._coarse_max_posterior_for_host(
                device_weights, active, physical_batch=physical,
            )
            assert actual.tobytes() == expected[:active].tobytes()
        reductions = [x for x in acquisitions if x["module"] == "jit__reduce_max"]
        slices = [x for x in acquisitions if x["module"] == "jit_dynamic_slice"]
        assert len(reductions) == (1 if physical else len(active_sizes))
        assert len(slices) == (0 if physical else len(active_sizes) - 1)
        acquired = list(acquisitions)
        elapsed = {str(active): [] for active in active_sizes}
        for repeat in range(21):
            for active in active_sizes[::1 if repeat % 2 == 0 else -1]:
                started = time.perf_counter_ns()
                actual = significance._coarse_max_posterior_for_host(
                    device_weights, active, physical_batch=physical,
                )
                elapsed[str(active)].append((time.perf_counter_ns() - started) / 1e6)
                assert actual.tobytes() == expected[:active].tobytes()
        assert acquisitions == acquired  # no hidden acquisition during warm timing
        panels.append({"physical_batch": physical, "acquisitions": acquired,
                       "warm_ms": elapsed})
    if "COARSE_GPU_ROOT" in os.environ:
        output = Path(os.environ["COARSE_GPU_ROOT"]) / f"pmax_width_{width}.json"
        with output.open("x") as stream:
            json.dump({"width": width, "physical_rows": 200, "logical_rows": active_sizes,
                       "input_bytes": weights.nbytes, "panels": panels}, stream, indent=2)
