"""Focused contract tests for half-volume BPref finalisation."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from recovar.em.helpers import half_volume_mstep

pytestmark = pytest.mark.unit


def test_c1_finalizer_is_exact_historical_x0_callthrough(monkeypatch):
    data_result = object()
    weight_result = object()
    calls = []

    def fake_x0(data, weight, shape, *, logger, label):
        calls.append((data, weight, shape, logger, label))
        return data_result, weight_result

    def forbidden_cuda(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError("C1 must not dispatch point-group CUDA")

    monkeypatch.setattr(half_volume_mstep, "enforce_half_volume_x0", fake_x0)
    from recovar.em.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(
        em_cuda_kernels,
        "relion_point_group_symmetrise_bpref",
        forbidden_cuda,
    )
    data = object()
    weight = object()
    logger = logging.getLogger("test_c1_finalizer")

    actual_data, actual_weight = half_volume_mstep.finalize_half_volume_bpref(
        data,
        weight,
        (7, 7, 7),
        logger=logger,
        label="test",
        symmetry_label="c1",
        relion_x_half=True,
    )

    assert actual_data is data_result
    assert actual_weight is weight_result
    assert calls == [(data, weight, (7, 7, 7), logger, "test")]


def test_c1_finalizer_is_bitwise_equal_to_historical_x0_result():
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(20260830)
    shape = (7, 7, 7)
    half_size = shape[0] * shape[1] * (shape[2] // 2 + 1)
    data = jnp.asarray(
        (rng.standard_normal(half_size) + 1j * rng.standard_normal(half_size)).astype(
            np.complex64
        )
    )
    weight = jnp.asarray(rng.standard_normal(half_size).astype(np.float32))
    logger = logging.getLogger("test_c1_bitwise")

    expected_data, expected_weight = half_volume_mstep.enforce_half_volume_x0(
        data,
        weight,
        shape,
        logger=logger,
        label="historical",
    )
    actual_data, actual_weight = half_volume_mstep.finalize_half_volume_bpref(
        data,
        weight,
        shape,
        logger=logger,
        label="new",
        symmetry_label="C1",
        relion_x_half=True,
    )

    np.testing.assert_array_equal(jax.device_get(actual_data), jax.device_get(expected_data))
    np.testing.assert_array_equal(jax.device_get(actual_weight), jax.device_get(expected_weight))


def test_non_c1_native_half_volume_fails_closed_before_cuda():
    c2 = np.stack(
        [
            np.eye(3),
            np.diag([-1.0, -1.0, 1.0]),
        ]
    )
    with pytest.raises(NotImplementedError, match="requires RELION x-half"):
        half_volume_mstep.finalize_half_volume_bpref(
            np.zeros(8, dtype=np.complex64),
            np.zeros(8, dtype=np.float32),
            (2, 2, 2),
            logger=logging.getLogger("test_native_fail_closed"),
            label="test",
            symmetry_label="C2",
            symmetry_operators=c2,
            relion_x_half=False,
        )


def test_c1_rejects_a_nonidentity_operator_payload():
    c2 = np.stack(
        [
            np.eye(3),
            np.diag([-1.0, -1.0, 1.0]),
        ]
    )
    with pytest.raises(ValueError, match="identity only"):
        half_volume_mstep.finalize_half_volume_bpref(
            np.zeros(8, dtype=np.complex64),
            np.zeros(8, dtype=np.float32),
            (2, 2, 2),
            logger=logging.getLogger("test_c1_payload"),
            label="test",
            symmetry_label="C1",
            symmetry_operators=c2,
            relion_x_half=False,
        )
