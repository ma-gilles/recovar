"""CPU routing contracts; the transaction itself has separate GPU tests."""

import sys
from types import ModuleType, SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import significance
from recovar.em.dense_single_volume.helpers.coarse_device_selection import SELECTION_REASONS
from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    CoarseGemmHybridCompactScores,
    plan_coarse_gemm_certificate_topology,
)

pytestmark = pytest.mark.unit
MODULE = "recovar.em.dense_single_volume.helpers.coarse_device_rescore"
VARIABLE = "RECOVAR_COARSE_GAUSSIAN_GEMM_DEVICE_TRANSACTION"


class _DeviceOnly:
    def __array__(self, *args, **kwargs):
        raise AssertionError("routing must not download interval/selection arrays")

    def __bool__(self):
        raise AssertionError("routing must use the compact status")


def _fixture(monkeypatch, status=None):
    topology = plan_coarse_gemm_certificate_topology(
        np.asarray([2, 0, 3, 1], np.int32), compact_pixel_count=4, translation_count=3
    )
    operands = (
        jnp.zeros((1, 32, 4), jnp.complex64),
        jnp.zeros((2, 3, 4), jnp.complex64),
        jnp.ones((2, 4), jnp.float32),
        jnp.zeros(2, jnp.float32),
    )
    kwargs = dict(
        topology=topology,
        actual_image_count=1,
        class_log_prior=np.float32(-0.25),
        block_capacity=1,
        certificate_chunk_rows=16,
        compact_posterior=True,
    )
    selection = SimpleNamespace(
        eligible=_DeviceOnly(),
        block_ids=_DeviceOnly(),
        block_count=_DeviceOnly(),
        posterior_block_count=_DeviceOnly(),
        raw_max_block_count=_DeviceOnly(),
    )
    scores = jnp.full((2, 48), -2.0, dtype=jnp.float32)
    compact = CoarseGemmHybridCompactScores(
        posterior_scores_flat=scores,
        source_block_ids=jnp.asarray([[0], [-1]], jnp.int32),
        block_count=jnp.asarray([1, 0], jnp.int32),
        raw_score_max=jnp.asarray([-1.75, -jnp.inf], jnp.float32),
        min_diff2_offsets=jnp.asarray([1.75, 0], jnp.float32),
        best_score=jnp.asarray([-2, -jnp.inf], jnp.float32),
        best_pose=jnp.zeros(2, jnp.int32),
        selected_output_valid=jnp.ones(2, jnp.bool_),
    )
    raw = jnp.ones((2, 1, 16, 3), jnp.float32)
    result = SimpleNamespace(
        selection=selection,
        compact_scores=compact,
        status=np.asarray([1, 0, 1, 1, 1], np.int64) if status is None else status,
        selected_diff2=raw,
    )
    calls = []
    module = ModuleType(MODULE)
    module.RESCORE_REASONS = SELECTION_REASONS + ("invalid_selected_exact_output", "invalid_runtime_prefix_contract")

    def rescore(*args, **options):
        calls.append((args, options))
        return result

    module.rescore_coarse_rotation_blocks = rescore
    monkeypatch.setitem(sys.modules, MODULE, module)

    def forbidden(*args, **kwargs):
        raise AssertionError("device route re-entered host certificate or selected scorer")

    for name in (
        "_prepare_relion_coarse_gaussian_gemm_f64_image_batch",
        "_relion_coarse_gaussian_gemm_update_certificate_state",
        "select_coarse_gemm_hybrid_rotation_blocks",
        "_relion_coarse_diff2_rotation_blocks_from_topology_f32",
        "assemble_coarse_gemm_hybrid_compact_scores_f32",
    ):
        monkeypatch.setattr(significance, name, forbidden)
    return operands, kwargs, result, calls


def test_switch_is_explicit_default_off(monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)
    assert not significance._coarse_gaussian_gemm_device_transaction_enabled()
    monkeypatch.setenv(VARIABLE, "1")
    assert significance._coarse_gaussian_gemm_device_transaction_enabled()
    monkeypatch.setenv(VARIABLE, "0")
    assert not significance._coarse_gaussian_gemm_device_transaction_enabled()
    for value in ("", "true", "auto", "2"):
        monkeypatch.setenv(VARIABLE, value)
        with pytest.raises(ValueError, match=VARIABLE):
            significance._coarse_gaussian_gemm_device_transaction_enabled()


def test_selected_transaction_keeps_device_arrays_and_forwards_operands(monkeypatch):
    operands, kwargs, expected, calls = _fixture(monkeypatch)
    rotation_prior = jnp.arange(32, dtype=jnp.float32)
    translation_prior = jnp.zeros((2, 3), jnp.float32)
    logical = jnp.int32(4)
    actual = significance._compute_coarse_gaussian_gemm_hybrid_batch(
        *operands,
        **kwargs,
        device_transaction=True,
        rotation_log_prior=rotation_prior,
        translation_log_prior=translation_prior,
        logical_full_pixel_count=logical,
        capture_selected_diff2=True,
        full_dense_diff2_fn=lambda: pytest.fail("selected route invoked full fallback"),
    )
    assert len(calls) == 1
    args, options = calls[0]
    assert args[0].shape == (32, 4)
    assert all(left is right for left, right in zip(args[1:4], operands[1:]))
    assert args[4].dtype == jnp.int32 and int(args[4]) == 1
    assert options["rotation_log_prior"] is rotation_prior
    assert options["translation_log_prior"] is translation_prior
    assert options["logical_full_pixel_count"] is logical
    assert options["topology"] is kwargs["topology"]
    assert options["chunk_rows"] == 16 and options["block_capacity"] == 1
    assert options["capture_selected_diff2"] is True
    assert actual.compact_scores is expected.compact_scores
    assert actual.selection is expected.selection
    assert actual.diagnostic_selected_diff2 is expected.selected_diff2
    assert actual.scores is None and actual.scores_include_priors
    assert actual.used_selected_rescore and actual.fallback_reason is None
    assert actual.selected_block_count == actual.max_selected_blocks == 1
    np.testing.assert_array_equal(actual.raw_score_max, [-1.75, 0])


@pytest.mark.parametrize("reason", [6, 10, 11, 12])
@pytest.mark.parametrize("capture", [False, True])
def test_transaction_failure_uses_one_existing_full_fallback(monkeypatch, reason, capture):
    operands, kwargs, expected, calls = _fixture(
        monkeypatch, np.asarray([0, reason, int(reason == 11), 0, 0], np.int64)
    )
    full_calls = []

    def full():
        full_calls.append(True)
        return jnp.full((2, 32, 3), 2.0, jnp.float32)

    actual = significance._compute_coarse_gaussian_gemm_hybrid_batch(
        *operands,
        **kwargs,
        device_transaction=True,
        full_dense_diff2_fn=full,
        capture_selected_diff2=capture,
    )
    assert len(calls) == len(full_calls) == 1
    assert actual.selection is expected.selection
    assert not actual.used_selected_rescore and actual.compact_scores is None
    assert actual.score_representation == "dense_full_direct_dynamic_fallback"
    assert actual.full_dense_backend == "fused_projector"
    assert actual.fallback_reason == sys.modules[MODULE].RESCORE_REASONS[reason]
    assert actual.scores.shape == (2, 32, 3)
    assert actual.diagnostic_selected_diff2 is None


@pytest.mark.parametrize("latched", [False, True])
def test_static_capacity_or_prior_overflow_skips_transaction(monkeypatch, latched):
    operands, kwargs, _expected, calls = _fixture(monkeypatch)
    kwargs["block_capacity"] = 1 if latched else 2
    actual = significance._compute_coarse_gaussian_gemm_hybrid_batch(
        *operands,
        **kwargs,
        device_transaction=True,
        force_static_dense_after_overflow=latched,
        full_dense_diff2_fn=lambda: jnp.zeros((2, 32, 3), jnp.float32),
    )
    assert calls == []
    assert actual.score_representation == "dense_full_direct_static_capacity"
    assert actual.fallback_reason == (
        "prior_batch_block_capacity_overflow" if latched else "compact_physical_capacity_not_smaller_than_dense"
    )


@pytest.mark.parametrize(
    "status",
    [
        np.zeros(4, np.int64),
        np.zeros(5, np.int32),
        np.asarray([2, 0, 1, 1, 1], np.int64),
        np.asarray([1, 99, 1, 1, 1], np.int64),
        np.asarray([1, 1, 1, 1, 1], np.int64),
        np.asarray([1, 0, 0, 1, 1], np.int64),
        np.asarray([1, 0, 1, 3, 1], np.int64),
        np.asarray([1, 0, 1, 1, 2], np.int64),
    ],
)
def test_malformed_status_cannot_publish_scores(monkeypatch, status):
    operands, kwargs, _expected, _calls = _fixture(monkeypatch, status)
    with pytest.raises(RuntimeError, match="[Dd]evice coarse"):
        significance._compute_coarse_gaussian_gemm_hybrid_batch(*operands, **kwargs, device_transaction=True)


def test_device_route_rejects_noncompact_request(monkeypatch):
    operands, kwargs, _expected, calls = _fixture(monkeypatch)
    kwargs["compact_posterior"] = False
    with pytest.raises(ValueError, match="requires compact_posterior"):
        significance._compute_coarse_gaussian_gemm_hybrid_batch(*operands, **kwargs, device_transaction=True)
    assert calls == []
