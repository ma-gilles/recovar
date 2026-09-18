"""Accumulator-only BPref must preserve scatter inputs and its legacy default."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cb

pytestmark = pytest.mark.unit


def arguments(grouped, stable):
    b = 2 if grouped else 1
    shape = (2, 726) if grouped else (726,)
    carry = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 1024
    values = dict(
        data_volume=(carry + 1j * (carry + 0.125)).astype(np.complex64),
        weight_volume=carry,
        images=np.full((b, 1), 1.25 - 0.625j, np.complex64),
        ctf=np.full((b, 1), -0.75, np.float32),
        minvsigma2=np.full((b, 1), 1.5, np.float32),
        posterior_over_weight_norm=np.full((b, 1, 1), 0.5, np.float32),
        translation_angles=np.zeros((1, 2), np.float32),
        pixel_indices=np.ones(1, np.int32),
        projector_full=np.zeros((9, 9, 9), np.complex64),
        rotation_matrices=np.broadcast_to(np.eye(3, dtype=np.float32), (b, 1, 3, 3)).copy(),
        image_shape=(32, 32), volume_shape=(11, 11, 11), max_r=4.0,
        projector_max_r=3, projection_padding_factor=1,
    )
    if grouped:
        values['reconstruction_group_ids'] = np.arange(b, dtype=np.int32)
    if stable:
        values.update(stable_dense_positions=np.ones(1, np.int32), logical_current_size=np.asarray(8, np.int32))
    return values


def device(values):
    return {k: jax.device_put(v.copy()) if isinstance(v, np.ndarray) else v for k, v in values.items()}


@pytest.mark.parametrize('value', [None, 0, 1, 'false', np.bool_(False)])
def test_selector_rejects_non_boolean_before_cuda(monkeypatch, value):
    monkeypatch.setattr(cb, '_ensure_ffi', lambda: pytest.fail('CUDA loaded before selector validation'))
    with pytest.raises(TypeError, match='Python bool'):
        cb.relion_vdam_mstep_fused_projector_x_half.__wrapped__(**arguments(False, True), return_denominator=value)


def test_external_replay_rejects_omitted_output_before_cuda(monkeypatch):
    monkeypatch.setenv('RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY', '/nonexistent')
    monkeypatch.setattr(cb, '_ensure_ffi', lambda: pytest.fail('CUDA loaded before mode validation'))
    with pytest.raises(ValueError, match='external host replay'):
        cb.relion_vdam_mstep_fused_projector_x_half.__wrapped__(**arguments(False, True), return_denominator=False)


@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('stable', [False, True])
def test_optional_result_preserves_all_ffi_operands_attributes_and_aliases(monkeypatch, grouped, stable):
    monkeypatch.delenv('RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY', raising=False)
    monkeypatch.setattr(cb, '_ensure_ffi', lambda: None)
    records = []

    def fake_ffi(target, output_types, **options):
        def call(*operands, **attrs):
            records.append((target, output_types, options, operands, attrs))
            return (*operands[14:17], jnp.ones(output_types[3].shape, jnp.float32))
        return call

    monkeypatch.setattr(jax.ffi, 'ffi_call', fake_ffi)
    fn = cb.relion_vdam_mstep_fused_projector_x_half.__wrapped__
    values = device(arguments(grouped, stable))
    full = fn(**values)
    compact = fn(**values, return_denominator=False)
    assert compact[2] is None
    np.testing.assert_array_equal(np.asarray(full[0]), np.asarray(compact[0]))
    np.testing.assert_array_equal(np.asarray(full[1]), np.asarray(compact[1]))
    left, right = records
    assert left[0] == right[0] and left[2] == right[2] and left[4] == right[4]
    assert left[2]['input_output_aliases'] == {14: 0, 15: 1, 16: 2}
    assert left[1][:3] == right[1][:3]
    assert left[1][3].shape == (2 if grouped else 1, 1, 40)
    assert right[1][3].shape == (0,) and right[1][3].dtype == np.float32
    for old, new in zip(left[3], right[3], strict=True):
        assert old.shape == new.shape and old.dtype == new.dtype
        assert np.asarray(old).tobytes() == np.asarray(new).tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('stable', [False, True])
def test_gpu_single_active_row_accumulators_are_bitwise_exact(grouped, stable):
    assert jax.default_backend() == 'gpu'
    fn = cb.relion_vdam_mstep_fused_projector_x_half
    values = arguments(grouped, stable)
    full = jax.block_until_ready(fn(**device(values)))
    compact = jax.block_until_ready(fn(**device(values), return_denominator=False))
    assert compact[2] is None
    assert full[2].shape == (2 if grouped else 1, 1, 1)
    for old, new in zip(full[:2], compact[:2], strict=True):
        assert np.isfinite(np.asarray(new)).all()
        assert np.asarray(old).tobytes() == np.asarray(new).tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize('bad_shape', [(1,), (0, 0), (1, 1, 0)])
def test_gpu_empty_output_abi_rejects_other_shapes(monkeypatch, bad_shape):
    assert jax.default_backend() == 'gpu'
    original = jax.ffi.ffi_call

    def bad_ffi(target, outputs, **options):
        return original(target, (*outputs[:3], jax.ShapeDtypeStruct(bad_shape, jnp.float32)), **options)

    monkeypatch.setattr(jax.ffi, 'ffi_call', bad_ffi)
    fn = cb.relion_vdam_mstep_fused_projector_x_half
    fn.clear_cache()
    try:
        with pytest.raises(RuntimeError, match='inconsistent topology'):
            jax.block_until_ready(fn(**device(arguments(False, True)), return_denominator=False))
    finally:
        fn.clear_cache()
