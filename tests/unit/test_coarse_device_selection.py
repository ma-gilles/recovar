"""Exact host-oracle tests for the standalone device source-block selector."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.scoring.coarse_device_selection import (
    SELECTION_REASONS,
    decode_device_coarse_selection,
    select_coarse_rotation_blocks_device,
)
from recovar.em.scoring.coarse_gemm_hybrid import (
    CoarseGemmHybridIntervalState,
    select_coarse_gemm_hybrid_rotation_blocks,
)

pytestmark = pytest.mark.unit


def fixture_state(batch=3, blocks=8, translations=3, dtype=np.float64):
    rng = np.random.default_rng(71)
    raw = rng.normal(size=(batch, blocks)).astype(dtype)
    posterior = (rng.normal(size=(batch, blocks)) * 200).astype(dtype)
    return CoarseGemmHybridIntervalState(
        raw,
        raw + dtype(0.2),
        posterior,
        posterior + dtype(0.1),
        np.ones(blocks * 16, dtype=np.int32),
        np.full(batch, blocks * 16 * translations, dtype=np.int64),
        np.zeros(batch, dtype=np.int64),
    )


def config(state, **overrides):
    return (
        dict(
            actual_image_count=state.raw_block_lower_max.shape[0] - 1,
            n_rotations=state.rotation_visit_count.size,
            n_translations=3,
            certificate_valid=True,
            block_capacity=8,
        )
        | overrides
    )


def check_oracle(state, *, compiled=True, **overrides):
    kwargs = config(state, **overrides)
    expected = select_coarse_gemm_hybrid_rotation_blocks(state, **kwargs)
    fn = partial(select_coarse_rotation_blocks_device, **kwargs)
    result = (jax.jit(fn) if compiled else fn)(state)
    assert all(isinstance(x, jax.Array) for x in result)
    actual = decode_device_coarse_selection(result)
    assert actual.eligible == expected.eligible
    assert actual.fallback_reason == expected.fallback_reason
    assert result.eligible.dtype == jnp.bool_
    assert result.reason_code.dtype == jnp.int32
    for name in ("block_ids", "block_count", "posterior_block_count", "raw_max_block_count"):
        left, right = getattr(actual, name), getattr(expected, name)
        assert left.dtype == right.dtype == np.int32
        np.testing.assert_array_equal(left, right)
    return actual


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("capacity", [1, 2, 8, 12])
def test_exact_union_and_whole_batch_overflow(dtype, capacity):
    check_oracle(fixture_state(dtype=dtype), block_capacity=capacity)


@pytest.mark.parametrize("span", [0.0, 138.0, np.nextafter(138.0, 0.0), np.nextafter(138.0, np.inf)])
def test_fp64_threshold_neighbors_and_sorted_union(span):
    state = fixture_state(blocks=8)
    lower = np.tile(
        np.array([200.0, 62.0, np.nextafter(62.0, -np.inf), np.nextafter(62.0, np.inf), -200.0, 0.0, 61.0, 63.0]),
        (3, 1),
    )
    raw = np.tile(np.array([-20.0, -20.0, -20.0, -20.0, 9.0, -20.0, -20.0, -20.0]), (3, 1))
    state = state._replace(
        raw_block_lower_max=raw,
        raw_block_upper_max=raw,
        posterior_block_lower_max=lower,
        posterior_block_upper_max=lower,
    )
    check_oracle(state, nonzero_score_span=span)


def test_inactive_image_rows_are_ignored_and_cleared():
    state = fixture_state()
    for field in state[:4]:
        field[-1] = np.nan
    state.candidate_count[-1] = -8
    state.invalid_candidate_count[-1] = 2
    result = check_oracle(state)
    assert result.eligible
    np.testing.assert_array_equal(result.block_ids[-1], -1)
    assert result.block_count[-1] == result.posterior_block_count[-1] == result.raw_max_block_count[-1] == 0


@pytest.mark.parametrize(
    "field", ["raw_block_lower_max", "raw_block_upper_max", "posterior_block_lower_max", "posterior_block_upper_max"]
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_endpoint_clears_entire_batch(field, value):
    state = fixture_state()
    getattr(state, field)[1, 4] = value
    result = check_oracle(state)
    assert result.fallback_reason == "invalid_or_nonfinite_block_interval"


@pytest.mark.parametrize("field", ["raw", "posterior"])
def test_reversed_endpoint(field):
    state = fixture_state()
    getattr(state, field + "_block_lower_max")[1, 2] = 1e6
    check_oracle(state)


@pytest.mark.parametrize("failure", ["visits_zero", "visits_duplicate", "counts", "invalid", "overflow"])
def test_runtime_failure_precedence(failure):
    state = fixture_state()
    # All rows select every block, so capacity failure is also present.
    state = state._replace(raw_block_lower_max=np.zeros((3, 8)), raw_block_upper_max=np.zeros((3, 8)))
    if failure != "overflow":
        state.posterior_block_upper_max[1, 0] = np.nan
    if failure in ("visits_zero", "visits_duplicate", "counts", "invalid"):
        state.invalid_candidate_count[1] = 1
    if failure in ("visits_zero", "visits_duplicate", "counts"):
        state.candidate_count[1] -= 1
    if failure.startswith("visits"):
        state.rotation_visit_count[0] = 0 if failure == "visits_zero" else 2
    result = check_oracle(state, block_capacity=2)
    expected = {"visits_zero": 5, "visits_duplicate": 5, "counts": 6, "invalid": 7, "overflow": 10}
    assert result.fallback_reason == SELECTION_REASONS[expected[failure]]


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"certificate_valid": None}, 2),
        ({"certificate_valid": False, "n_rotations": 17}, 2),
        ({"certificate_valid": 1}, 2),
        ({"n_rotations": 17, "block_capacity": 0}, 3),
        ({"n_rotations": 0}, 3),
        ({"actual_image_count": 0}, 1),
        ({"actual_image_count": 4}, 1),
        ({"block_capacity": 0}, 1),
        ({"block_capacity": -3}, 1),
        ({"n_translations": 0}, 1),
        ({"nonzero_score_span": -1}, 1),
        ({"nonzero_score_span": np.inf}, 1),
        ({"nonzero_score_span": np.nan}, 1),
        ({"actual_image_count": 1.5}, 1),
        ({"block_capacity": "bad", "certificate_valid": False}, 1),
    ],
)
def test_static_admission_matches_host_reason_and_buffer_width(overrides, reason):
    result = check_oracle(fixture_state(), **overrides)
    assert result.fallback_reason == SELECTION_REASONS[reason]


@pytest.mark.parametrize("field", CoarseGemmHybridIntervalState._fields)
def test_incomplete_field_shape(field):
    state = fixture_state()
    state = state._replace(**{field: getattr(state, field)[..., :-1]})
    result = check_oracle(state, n_rotations=128)
    assert result.fallback_reason == "incomplete_block_table"


def test_outer_jit_changes_inputs_and_status_without_retracing():
    state = fixture_state()
    kwargs = config(state)
    traces = []

    @jax.jit
    def composed(value):
        traces.append(1)
        # Both preparation and consumption occur under the same trace.
        prepared = value._replace(raw_block_upper_max=value.raw_block_upper_max + jnp.float64(0))
        selected = select_coarse_rotation_blocks_device(prepared, **kwargs)
        return selected, jnp.sum(jnp.where(selected.block_ids >= 0, selected.block_ids, 0))

    first, checksum = composed(state)
    assert bool(first.eligible)
    assert int(checksum) == np.maximum(np.asarray(first.block_ids), 0).sum()
    changed = state._replace(candidate_count=state.candidate_count - 1)
    second, checksum = composed(changed)
    assert not bool(second.eligible) and int(second.reason_code) == 6
    assert int(checksum) == 0 and len(traces) == 1
    jaxpr = str(jax.make_jaxpr(composed)(state))
    assert "callback" not in jaxpr


def test_fp64_is_required_instead_of_silent_threshold_rounding():
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="requires JAX x64"):
            select_coarse_rotation_blocks_device(fixture_state(), **config(fixture_state()))


def test_empty_selection_is_unreachable_for_finite_ordered_nonempty_intervals():
    # Host validation guarantees at least its raw maximum block survives.
    state = fixture_state(blocks=1)
    result = check_oracle(state, block_capacity=1)
    assert result.eligible
    np.testing.assert_array_equal(result.block_count, [1, 1, 0])


def test_runtime_image_prefix_composes_and_reuses_one_trace():
    state = fixture_state()
    traces = []
    kwargs = config(state)
    del kwargs["actual_image_count"]

    @jax.jit
    def composed(value, prefix):
        traces.append(1)
        return select_coarse_rotation_blocks_device(value, actual_image_count=prefix, **kwargs)

    for prefix in (1, 3, 2, 0, -1, 4):
        actual = decode_device_coarse_selection(composed(state, jnp.int32(prefix)))
        expected = select_coarse_gemm_hybrid_rotation_blocks(state, actual_image_count=prefix, **kwargs)
        assert actual.eligible == expected.eligible
        assert actual.fallback_reason == expected.fallback_reason
        for name in ("block_ids", "block_count", "posterior_block_count", "raw_max_block_count"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert len(traces) == 1


def test_runtime_invalid_prefix_precedes_bad_table():
    state = fixture_state()._replace(candidate_count=np.zeros(1, dtype=np.int64))
    kwargs = config(state)
    del kwargs["actual_image_count"]
    fn = jax.jit(lambda prefix: select_coarse_rotation_blocks_device(state, actual_image_count=prefix, **kwargs))
    assert int(fn(jnp.int32(0)).reason_code) == 1
    assert int(fn(jnp.int32(2)).reason_code) == 4


def test_reference_integer_casts_and_unrepresentable_candidate_total():
    state = fixture_state()
    cast_state = state._replace(
        rotation_visit_count=state.rotation_visit_count.astype(np.float64) + 0.25,
        candidate_count=state.candidate_count.astype(np.float64) + 0.5,
        invalid_candidate_count=state.invalid_candidate_count.astype(np.float64) + 0.5,
    )
    assert check_oracle(cast_state).eligible
    assert check_oracle(state, n_translations=2**63).fallback_reason == "incomplete_candidate_coverage"


@pytest.mark.parametrize("dtype,shape", [(np.int64, ()), (np.float32, ()), (np.int32, (1,))])
def test_runtime_prefix_metadata_rejects_non_s32_scalar(dtype, shape):
    state = fixture_state()
    kwargs = config(state)
    kwargs["actual_image_count"] = jnp.asarray(np.ones(shape, dtype=dtype))
    result = select_coarse_rotation_blocks_device(state, **kwargs)
    assert not bool(result.eligible) and int(result.reason_code) == 1
    assert result.block_ids.shape == (3, 1)
