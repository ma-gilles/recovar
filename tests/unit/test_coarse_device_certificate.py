"""Same-input comparisons of composed and host-streamed certificate state."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import scoring
from recovar.em.dense_single_volume.helpers.coarse_device_certificate import (
    _coarse_certificate_state_jit,
    _certify_coarse_rotation_blocks_jit,
    certify_coarse_rotation_blocks,
    prepare_coarse_certificate_state,
)
from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
    initialize_coarse_gemm_hybrid_interval_state,
    plan_coarse_gemm_certificate_topology,
    select_coarse_gemm_hybrid_rotation_blocks,
)

pytestmark = pytest.mark.unit


def _case(rotations=80, prior_kind="batched", poison=False):
    rng = np.random.default_rng(413)
    batch, translations, pixels = 3, 3, 11

    def complex_values(shape):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)

    projections = complex_values((rotations, pixels))
    shifted = complex_values((batch, translations, pixels))
    weight = rng.uniform(0.1, 2, (batch, pixels)).astype(np.float32)
    initial = rng.uniform(0, 1, batch).astype(np.float32)
    # Garbage in the padded image must not poison valid rows or their counts.
    shifted[-1] = np.nan
    weight[-1] = np.inf
    initial[-1] = -1
    if poison:
        projections[rotations - 1, 2] = np.nan
    mapping = np.concatenate((np.arange(pixels, dtype=np.int32), [-1, -1])).astype(np.int32)
    topology = plan_coarse_gemm_certificate_topology(
        mapping, compact_pixel_count=pixels, translation_count=translations
    )
    kwargs = dict(topology=topology, class_log_prior=np.float32(-0.37))
    if prior_kind != "none":
        kwargs["rotation_log_prior"] = rng.normal(size=rotations).astype(np.float32)
        shape = (batch, translations) if prior_kind == "batched" else (translations,)
        kwargs["translation_log_prior"] = rng.normal(size=shape).astype(np.float32)
    return (projections, shifted, weight, initial), kwargs


def _host_loop(operands, kwargs, actual, chunk):
    projections, shifted, weight, initial = operands
    image_batch = scoring._prepare_relion_coarse_gaussian_gemm_f64_image_batch(shifted, weight, initial, actual)
    state = initialize_coarse_gemm_hybrid_interval_state(shifted.shape[0], projections.shape[0])
    for offset in range(0, projections.shape[0], chunk):
        block_kwargs = dict(kwargs)
        if "rotation_log_prior" in kwargs:
            block_kwargs["rotation_log_prior"] = kwargs["rotation_log_prior"][offset : offset + chunk]
        state = scoring._relion_coarse_gaussian_gemm_update_certificate_state(
            state, projections[offset : offset + chunk], image_batch, rotation_offset=offset, **block_kwargs
        )
    return state


@pytest.mark.parametrize("rotations,chunk", [(16, 32), (64, 32), (80, 32)])
@pytest.mark.parametrize("prior_kind", ["none", "shared", "batched"])
def test_composed_state_matches_host_chunks_exactly(rotations, chunk, prior_kind):
    operands, kwargs = _case(rotations, prior_kind)
    with jax.enable_x64(True):
        expected = _host_loop(operands, kwargs, 2, chunk)
        actual = prepare_coarse_certificate_state(*operands, 2, chunk_rows=chunk, **kwargs)
        for field in expected._fields:
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field), err_msg=field)
        assert np.array_equal(actual.rotation_visit_count, np.ones(rotations, np.int32))
        assert np.array_equal(actual.candidate_count, [rotations * 3, rotations * 3, 0])
        config = dict(
            actual_image_count=2,
            n_rotations=rotations,
            n_translations=3,
            certificate_valid=True,
            block_capacity=rotations // 16,
        )
        for left, right in zip(
            select_coarse_gemm_hybrid_rotation_blocks(expected, **config),
            select_coarse_gemm_hybrid_rotation_blocks(actual, **config),
        ):
            np.testing.assert_array_equal(left, right)


def test_invalid_tail_is_retained_for_selector_fallback():
    operands, kwargs = _case(poison=True)
    with jax.enable_x64(True):
        expected = _host_loop(operands, kwargs, 2, 32)
        actual = prepare_coarse_certificate_state(*operands, 2, chunk_rows=32, **kwargs)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
        selection = select_coarse_gemm_hybrid_rotation_blocks(
            actual,
            actual_image_count=2,
            n_rotations=80,
            n_translations=3,
            certificate_valid=True,
        )
    assert not selection.eligible
    assert selection.fallback_reason == "invalid_or_nonfinite_candidate_interval"


def test_device_active_count_reuses_executable_and_composes():
    operands, kwargs = _case()
    with jax.enable_x64(True):
        device_operands = tuple(jnp.asarray(value) for value in operands)
        _coarse_certificate_state_jit.clear_cache()
        for count in (1, 2):
            actual = prepare_coarse_certificate_state(*device_operands, jnp.int32(count), chunk_rows=32, **kwargs)
            expected = _host_loop(operands, kwargs, count, 32)
            for left, right in zip(actual, expected):
                np.testing.assert_array_equal(left, right)
        assert _coarse_certificate_state_jit._cache_size() == 1
        # The public admission reads topology and abstract array metadata only.
        composed = jax.jit(lambda *args: prepare_coarse_certificate_state(*args, chunk_rows=32, **kwargs))
        state = composed(*device_operands, jnp.int32(2))
        np.testing.assert_array_equal(state.candidate_count, [240, 240, 0])


@pytest.mark.parametrize("chunk", [0, -16, 17, 1.5])
def test_invalid_chunk_is_rejected(chunk):
    operands, kwargs = _case()
    with pytest.raises(ValueError, match="chunk_rows"):
        prepare_coarse_certificate_state(*operands, 2, chunk_rows=chunk, **kwargs)


@pytest.mark.parametrize("count", [0, 4, -1, 2**40])
def test_invalid_host_active_count_is_rejected(count):
    operands, kwargs = _case()
    with pytest.raises(ValueError, match="actual_image_count"):
        prepare_coarse_certificate_state(*operands, count, **kwargs)


def test_stored_input_precision_is_not_silently_changed():
    operands, kwargs = _case()
    with jax.enable_x64(True), pytest.raises(TypeError, match="complex64/float32"):
        prepare_coarse_certificate_state(operands[0].astype(np.complex128), *operands[1:], 2, **kwargs)


def test_topology_mismatch_is_rejected():
    operands, kwargs = _case()
    kwargs["topology"] = plan_coarse_gemm_certificate_topology(
        np.arange(11, dtype=np.int32), compact_pixel_count=11, translation_count=4
    )
    with pytest.raises(ValueError, match="topology"):
        prepare_coarse_certificate_state(*operands, 2, **kwargs)


@pytest.mark.parametrize("capacity", [1, 5, 8])
@pytest.mark.parametrize("poison", [False, True])
def test_complete_device_certificate_selection_matches_host(capacity, poison):
    from recovar.em.dense_single_volume.helpers.coarse_device_selection import (
        decode_device_coarse_selection,
    )

    operands, kwargs = _case(poison=poison)
    with jax.enable_x64(True):
        expected = select_coarse_gemm_hybrid_rotation_blocks(
            _host_loop(operands, kwargs, 2, 32),
            actual_image_count=2,
            n_rotations=80,
            n_translations=3,
            certificate_valid=True,
            block_capacity=capacity,
        )
        actual = decode_device_coarse_selection(
            certify_coarse_rotation_blocks(*operands, jnp.int32(2), chunk_rows=32, block_capacity=capacity, **kwargs)
        )
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def test_combined_selection_validates_dynamic_count_without_new_executables():
    from recovar.em.dense_single_volume.helpers.coarse_device_selection import (
        decode_device_coarse_selection,
    )

    operands, kwargs = _case()
    with jax.enable_x64(True):
        _certify_coarse_rotation_blocks_jit.clear_cache()
        for count in (1, 2, 0, -1, 4):
            actual = decode_device_coarse_selection(
                certify_coarse_rotation_blocks(*operands, jnp.int32(count), chunk_rows=32, block_capacity=5, **kwargs)
            )
            if count in (1, 2):
                expected = select_coarse_gemm_hybrid_rotation_blocks(
                    _host_loop(operands, kwargs, count, 32),
                    actual_image_count=count,
                    n_rotations=80,
                    n_translations=3,
                    certificate_valid=True,
                    block_capacity=5,
                )
                for left, right in zip(actual, expected):
                    np.testing.assert_array_equal(left, right)
            else:
                assert actual.fallback_reason == "invalid_selection_configuration"
                assert not actual.eligible
                np.testing.assert_array_equal(actual.block_ids, np.full((3, 5), -1, np.int32))
                np.testing.assert_array_equal(actual.block_count, np.zeros(3, np.int32))
        assert _certify_coarse_rotation_blocks_jit._cache_size() == 1
