"""CPU control-flow tests use explicit doubles; marked GPU tests use real CUDA."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cb
from recovar.em.dense_single_volume.helpers import coarse_device_rescore as module
from recovar.em.dense_single_volume.helpers.coarse_device_selection import decode_device_coarse_selection
from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    assemble_coarse_gemm_hybrid_compact_scores_f32,
    plan_coarse_gemm_certificate_topology,
    select_coarse_gemm_hybrid_rotation_blocks,
)
from recovar.em.dense_single_volume.helpers import scoring

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolate_rescore_executables():
    """Never reuse a graph traced with a CPU scorer double in an oracle test."""
    module._rescore_coarse_rotation_blocks_jit.clear_cache()
    try:
        yield
    finally:
        module._rescore_coarse_rotation_blocks_jit.clear_cache()


def case(priors="none", mapping=None, omitted=()):
    # Dyadic small operands make native atomic additions exactly representable;
    # these tests check composition, not a tolerance for unordered reductions.
    rng = np.random.default_rng(401)
    batch, rotations, translations, pixels = 3, 80, 3, 11

    def complex_data(shape):
        return ((rng.integers(-4, 5, shape) + 1j * rng.integers(-4, 5, shape)) / 4).astype(np.complex64)

    reference = complex_data((rotations, pixels))
    shifted = complex_data((batch, translations, pixels))
    weight = np.full((batch, pixels), 0.5, np.float32)
    weight[:, list(omitted)] = 0
    initial = np.ones(batch, np.float32)
    shifted[-1] = np.nan
    weight[-1] = np.inf
    initial[-1] = -1
    if mapping is None:
        mapping = np.arange(pixels, dtype=np.int32)
    topology = plan_coarse_gemm_certificate_topology(
        np.asarray(mapping, np.int32), compact_pixel_count=pixels, translation_count=translations
    )
    kwargs = dict(topology=topology, class_log_prior=np.float32(-0.5), chunk_rows=32, block_capacity=8)
    if priors != "none":
        kwargs["rotation_log_prior"] = rng.integers(-4, 5, rotations).astype(np.float32) / 8
        shape = (batch, translations) if priors == "batched" else (translations,)
        kwargs["translation_log_prior"] = rng.integers(-4, 5, shape).astype(np.float32) / 8
    return (reference, shifted, weight, initial), kwargs


def install_cpu_double(monkeypatch, *, invalid=False, execution_log=None):
    """Deliberately no CUDA arithmetic claim: manufactured positive diff2."""

    def selected(reference, shifted, weight, initial, ids, mapping, *logical):
        if execution_log is not None:
            jax.debug.callback(lambda: execution_log.append("executed"), ordered=True)
        shape = (shifted.shape[0], ids.shape[1], 16, shifted.shape[1])
        values = jnp.broadcast_to(jnp.arange(16, dtype=jnp.float32)[None, None, :, None] / 8 + 1, shape)
        result = jnp.where(ids[:, :, None, None] >= 0, values, jnp.inf)
        if invalid:
            result = result.at[0, 0, 0, 0].set(jnp.nan)
        return result

    monkeypatch.setattr(cb, "relion_coarse_diff2_rotation_blocks_f32", selected)
    monkeypatch.setattr(cb, "relion_coarse_diff2_rotation_blocks_runtime_f32", selected)
    module._rescore_coarse_rotation_blocks_jit.clear_cache()
    return selected


def assert_failed(result, reason):
    assert not bool(result.status[module.STATUS_VALID]) and int(result.status[module.STATUS_REASON]) == reason
    compact = result.compact_scores
    assert np.isneginf(np.asarray(compact.posterior_scores_flat)).all()
    assert np.isneginf(np.asarray(compact.best_score)).all()
    np.testing.assert_array_equal(compact.source_block_ids, -1)
    np.testing.assert_array_equal(compact.block_count, 0)
    np.testing.assert_array_equal(compact.best_pose, 0)
    assert not np.asarray(compact.selected_output_valid).any()


@pytest.mark.parametrize("priors", ["none", "shared", "batched"])
def test_cpu_composed_control_flow_and_shared_assembly(monkeypatch, priors):
    fake = install_cpu_double(monkeypatch)
    operands, kwargs = case(priors)
    result = module.rescore_coarse_rotation_blocks(*operands, jnp.int32(2), **kwargs)
    host_selection = decode_device_coarse_selection(result.selection)
    expected = assemble_coarse_gemm_hybrid_compact_scores_f32(
        fake(*operands, jnp.asarray(host_selection.block_ids), jnp.asarray(kwargs["topology"].full_to_compact)),
        host_selection,
        actual_image_count=2,
        n_rotations=80,
        **{k: v for k, v in kwargs.items() if k.endswith("log_prior")},
    )
    assert bool(result.status[module.STATUS_VALID]) and bool(result.status[module.STATUS_USED_SELECTED_RESCORE])
    assert int(result.status[module.STATUS_BLOCK_SUM]) == host_selection.block_count.sum()
    assert int(result.status[module.STATUS_BLOCK_MAX]) == host_selection.block_count.max()
    for left, right in zip(result.compact_scores, expected):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    "failure,reason", [("overflow", 10), ("bad_count", 1), ("bad_certificate", 7), ("bad_prefix", 12)]
)
def test_cpu_failed_route_does_not_execute_scorer(monkeypatch, failure, reason):
    executions = []
    install_cpu_double(monkeypatch, execution_log=executions)
    operands, kwargs = case()
    count = jnp.int32(2)
    if failure == "overflow":
        kwargs["block_capacity"] = 1
    if failure == "bad_count":
        count = jnp.int32(0)
    if failure == "bad_certificate":
        operands[0][0, 0] = np.nan
    if failure == "bad_prefix":
        kwargs["logical_full_pixel_count"] = jnp.int32(5)
    result = module.rescore_coarse_rotation_blocks(*operands, count, **kwargs)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert_failed(result, reason)
    assert not bool(result.status[module.STATUS_USED_SELECTED_RESCORE]) and executions == []


def test_cpu_invalid_selected_output_clears_whole_batch(monkeypatch):
    install_cpu_double(monkeypatch, invalid=True)
    operands, kwargs = case()
    result = module.rescore_coarse_rotation_blocks(*operands, 2, **kwargs)
    assert_failed(result, 11)
    assert bool(result.selection.eligible) and bool(result.status[module.STATUS_USED_SELECTED_RESCORE])


@pytest.mark.parametrize("capacity,invalid", [(8, False), (1, False), (8, True)])
def test_cpu_optional_capture_retains_only_same_scorer_output(monkeypatch, capacity, invalid):
    executions = []
    install_cpu_double(monkeypatch, invalid=invalid, execution_log=executions)
    operands, kwargs = case()
    kwargs["block_capacity"] = capacity
    result = module.rescore_coarse_rotation_blocks(*operands, 2, capture_selected_diff2=True, **kwargs)
    jax.block_until_ready(result)
    jax.effects_barrier()
    status = np.asarray(result.status)
    assert status.shape == (5,) and status.dtype == np.int64 and status.nbytes == 40
    assert result.selected_diff2.shape == (3, capacity, 16, 3)
    if capacity == 1:
        assert executions == [] and np.isposinf(np.asarray(result.selected_diff2)).all()
    else:
        assert executions == ["executed"]
        if invalid:
            assert np.isnan(np.asarray(result.selected_diff2)[0, 0, 0, 0])
        else:
            assert np.isfinite(np.asarray(result.selected_diff2)[0, 0]).all()


@pytest.mark.parametrize(
    "mapping,omitted,valid",
    [
        ([2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10], tuple(range(5, 11)), True),
        ([0, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], tuple(range(4, 11)), True),
        ([0, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], tuple(range(5, 11)), False),
        ([0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10], tuple(range(4, 11)), False),
    ],
)
def test_cpu_general_runtime_prefix_contract(monkeypatch, mapping, omitted, valid):
    install_cpu_double(monkeypatch)
    operands, kwargs = case(mapping=mapping, omitted=omitted)
    result = module.rescore_coarse_rotation_blocks(
        *operands, jnp.int32(2), logical_full_pixel_count=jnp.int32(5), **kwargs
    )
    if valid:
        assert bool(result.status[module.STATUS_VALID])
    else:
        assert_failed(result, 12)


def test_cpu_outer_jit_dynamic_counts_and_no_host_callbacks(monkeypatch):
    install_cpu_double(monkeypatch)
    operands, kwargs = case(omitted=tuple(range(5, 11)))
    traces = []

    @jax.jit
    def outer(*values):
        traces.append(1)
        return module.rescore_coarse_rotation_blocks(
            *values[:4], values[4], logical_full_pixel_count=values[5], **kwargs
        )

    for actual, logical in ((1, 5), (2, 6), (0, 5), (2, 0), (2, 12)):
        result = outer(*operands, jnp.int32(actual), jnp.int32(logical))
        if actual in (1, 2) and logical in (5, 6):
            assert bool(result.status[module.STATUS_VALID])
        else:
            assert_failed(result, 1 if actual == 0 else 12)
    assert len(traces) == 1
    assert "callback" not in str(jax.make_jaxpr(outer)(*operands, jnp.int32(2), jnp.int32(5)))


@pytest.mark.parametrize("field", ["mapping", "translations", "capacity", "logical_dtype", "logical_shape"])
def test_admission_rejects_before_compilation(field):
    operands, kwargs = case()
    if field == "mapping":
        kwargs["topology"] = kwargs["topology"]._replace(full_to_compact_sha256="bad")
    if field == "translations":
        operands = (operands[0], np.zeros((3, 129, 11), np.complex64), operands[2], operands[3])
    if field == "capacity":
        kwargs["block_capacity"] = 0
    if field == "logical_dtype":
        kwargs["logical_full_pixel_count"] = jnp.float32(5)
    if field == "logical_shape":
        kwargs["logical_full_pixel_count"] = jnp.ones(1, jnp.int32)
    with pytest.raises((ValueError, TypeError)):
        module.rescore_coarse_rotation_blocks(*operands, 2, **kwargs)


def host_oracle(operands, kwargs, actual, logical):
    certificate_kwargs = {k: v for k, v in kwargs.items() if k != "block_capacity"}
    # The accepted host loop is deliberately retained as the reference chain.
    reference, shifted, weight, initial = operands
    image_batch = scoring._prepare_relion_coarse_gaussian_gemm_f64_image_batch(shifted, weight, initial, actual)
    from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import initialize_coarse_gemm_hybrid_interval_state

    state = initialize_coarse_gemm_hybrid_interval_state(shifted.shape[0], reference.shape[0])
    for offset in range(0, reference.shape[0], kwargs["chunk_rows"]):
        prior = kwargs.get("rotation_log_prior")
        state = scoring._relion_coarse_gaussian_gemm_update_certificate_state(
            state,
            reference[offset : offset + kwargs["chunk_rows"]],
            image_batch,
            topology=kwargs["topology"],
            rotation_offset=offset,
            class_log_prior=kwargs["class_log_prior"],
            rotation_log_prior=None if prior is None else prior[offset : offset + kwargs["chunk_rows"]],
            translation_log_prior=kwargs.get("translation_log_prior"),
        )
    selection = select_coarse_gemm_hybrid_rotation_blocks(
        state,
        actual_image_count=actual,
        n_rotations=reference.shape[0],
        n_translations=shifted.shape[1],
        certificate_valid=True,
        block_capacity=kwargs["block_capacity"],
    )
    if not selection.eligible:
        return selection, None
    diff2 = scoring._relion_coarse_diff2_rotation_blocks_from_topology_f32(
        *operands, jnp.asarray(selection.block_ids), topology=kwargs["topology"], logical_full_pixel_count=logical
    )
    compact = assemble_coarse_gemm_hybrid_compact_scores_f32(
        diff2,
        selection,
        actual_image_count=actual,
        n_rotations=reference.shape[0],
        **{k: v for k, v in certificate_kwargs.items() if k.endswith("log_prior")},
    )
    return selection, compact


@pytest.mark.gpu
@pytest.mark.parametrize("priors", ["none", "shared", "batched"])
@pytest.mark.parametrize("runtime", [False, True])
def test_gpu_existing_host_chain_exact(priors, runtime, custom_cuda_lib, gpu_device, monkeypatch):
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    operands, kwargs = case(priors, omitted=tuple(range(7, 11)) if runtime else ())
    with jax.default_device(gpu_device), jax.enable_x64(True):
        operands = tuple(jnp.asarray(v) for v in operands)
        logical = jnp.int32(7) if runtime else None
        expected_selection, expected = host_oracle(operands, kwargs, 2, logical)
        actual = module.rescore_coarse_rotation_blocks(
            *operands, jnp.int32(2), logical_full_pixel_count=logical, **kwargs
        )
        assert bool(actual.status[module.STATUS_VALID]) and expected_selection.eligible
        decoded = decode_device_coarse_selection(actual.selection)
        for a, b in zip(decoded, expected_selection):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(actual.compact_scores, expected):
            np.testing.assert_array_equal(a, b)


@pytest.mark.gpu
def test_gpu_dynamic_prefix_and_failure_reuses_executable(custom_cuda_lib, gpu_device, monkeypatch):
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    operands, kwargs = case(omitted=tuple(range(5, 11)))
    with jax.default_device(gpu_device), jax.enable_x64(True):
        operands = tuple(jnp.asarray(v) for v in operands)
        module._rescore_coarse_rotation_blocks_jit.clear_cache()
        for count, logical in ((1, 5), (2, 7), (0, 5), (2, -1)):
            result = module.rescore_coarse_rotation_blocks(
                *operands, jnp.int32(count), logical_full_pixel_count=jnp.int32(logical), **kwargs
            )
            if count > 0 and logical > 0:
                _, expected = host_oracle(operands, kwargs, count, jnp.int32(logical))
                assert bool(result.status[module.STATUS_VALID])
                for a, b in zip(result.compact_scores, expected):
                    np.testing.assert_array_equal(a, b)
            else:
                assert_failed(result, 1 if count == 0 else 12)
        assert module._rescore_coarse_rotation_blocks_jit._cache_size() == 1


@pytest.mark.gpu
@pytest.mark.parametrize("failure", ["overflow", "invalid_certificate"])
def test_gpu_whole_batch_selector_failure(failure, custom_cuda_lib, gpu_device, monkeypatch):
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    operands, kwargs = case()
    if failure == "overflow":
        kwargs["block_capacity"] = 1
    else:
        operands[0][0, 0] = np.nan
    with jax.default_device(gpu_device), jax.enable_x64(True):
        operands = tuple(jnp.asarray(v) for v in operands)
        expected, _ = host_oracle(operands, kwargs, 2, None)
        result = module.rescore_coarse_rotation_blocks(*operands, jnp.int32(2), capture_selected_diff2=True, **kwargs)
        assert_failed(result, 10 if failure == "overflow" else 7)
        assert not expected.eligible and not bool(result.status[module.STATUS_USED_SELECTED_RESCORE])
        assert np.isposinf(np.asarray(result.selected_diff2)).all()


@pytest.mark.gpu
def test_gpu_runtime_lookup_holes_and_diagnostic_capture(custom_cuda_lib, gpu_device, monkeypatch):
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    operands, kwargs = case(mapping=[0, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], omitted=tuple(range(4, 11)))
    with jax.default_device(gpu_device), jax.enable_x64(True):
        operands = tuple(jnp.asarray(v) for v in operands)
        selection, expected = host_oracle(operands, kwargs, 2, jnp.int32(5))
        result = module.rescore_coarse_rotation_blocks(
            *operands, jnp.int32(2), logical_full_pixel_count=jnp.int32(5), capture_selected_diff2=True, **kwargs
        )
        assert bool(result.status[module.STATUS_VALID])
        for a, b in zip(result.compact_scores, expected):
            np.testing.assert_array_equal(a, b)
        old = scoring._relion_coarse_diff2_rotation_blocks_from_topology_f32(
            *operands,
            jnp.asarray(selection.block_ids),
            topology=kwargs["topology"],
            logical_full_pixel_count=jnp.int32(5),
        )
        np.testing.assert_array_equal(result.selected_diff2, old)
