"""Focused tests for the RELION CUDA make-eulers diagnostic FFI.

Production accelerated EM receives host-generated matrices; this helper is
retained to reproduce and classify CUDA arithmetic in isolation.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.unit


_EULER_BITS = np.asarray(
    [
        0xC32D9249,
        0x42276F67,
        0x438E8000,
        0x432C8000,
        0x420EA2E8,
        0x43960000,
        0x432D9249,
        0x42276F67,
        0x43924000,
        0xC32D9249,
        0x42276F67,
        0x438E8000,
    ],
    dtype=np.uint32,
)
_RIGHT_MATRIX_BITS = np.asarray(
    [
        0x3F7E33B2,
        0x3DD8B1C8,
        0xBD58D8D0,
        0xBDD8B1C8,
        0x3F7E8FDC,
        0x3B383321,
        0x3D58D8D0,
        0x3B383321,
        0x3F7FA3D6,
    ],
    dtype=np.uint32,
)
_RELION_INVERSE_PROJECTOR_BITS = np.asarray(
    [
        0xBECFE1AF,
        0xBF2DF283,
        0xBF1C7199,
        0x3F669AF7,
        0xBED209B0,
        0xBE11C756,
        0xBE1DA86A,
        0xBF1BB8DA,
        0x3F4754ED,
        0xBECC9734,
        0xBF3DD491,
        0xBF09F8AE,
        0x3F600302,
        0xBEF7ADE2,
        0x3C8992EF,
        0xBE8BDCE3,
        0xBEEE06DB,
        0x3F579871,
        0xBE95CA1D,
        0xBF38D979,
        0xBF207DC3,
        0x3F6CC69D,
        0xBEC2A27E,
        0x3BCC03F7,
        0xBE78A4DC,
        0xBF13F915,
        0x3F477073,
        0xBECFE1AF,
        0xBF2DF283,
        0xBF1C7199,
        0x3F669AF7,
        0xBED209B0,
        0xBE11C756,
        0xBE1DA86A,
        0xBF1BB8DA,
        0x3F4754ED,
    ],
    dtype=np.uint32,
)


def _f32_from_bits(bits, shape):
    return bits.view(np.float32).reshape(shape)


@pytest.mark.gpu
def test_relion_scoring_rotations_match_frozen_device_replay_bitwise(monkeypatch, custom_cuda_lib, gpu_device):
    """All 36 scorer floats equal transpose(RELION inverse-projector dump)."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    eulers = _f32_from_bits(_EULER_BITS, (4, 3))
    right_matrix = _f32_from_bits(_RIGHT_MATRIX_BITS, (3, 3))
    relion_inverse = _f32_from_bits(_RELION_INVERSE_PROJECTOR_BITS, (4, 3, 3))

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_make_scoring_rotations_f32(jnp.asarray(eulers), jnp.asarray(right_matrix))

    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        relion_inverse.swapaxes(1, 2).view(np.uint32),
    )


@pytest.mark.gpu
def test_relion_scoring_rotations_support_empty_batch(monkeypatch, custom_cuda_lib, gpu_device):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_make_scoring_rotations_f32(
            jnp.empty((0, 3), dtype=jnp.float32),
            jnp.eye(3, dtype=jnp.float32),
        )

    assert actual.shape == (0, 3, 3)
    assert actual.dtype == jnp.float32


@pytest.mark.parametrize(
    "eulers,right_matrix,error_type,error",
    [
        (np.zeros((2, 3), np.float64), np.eye(3, dtype=np.float32), TypeError, "eulers_deg must be float32"),
        (np.zeros((2, 3), np.float32), np.eye(3, dtype=np.float64), TypeError, "right_matrix must be float32"),
        (np.zeros((2, 2), np.float32), np.eye(3, dtype=np.float32), ValueError, r"shape \(N, 3\)"),
        (np.zeros((2, 3), np.float32), np.zeros((3, 2), np.float32), ValueError, r"shape \(3, 3\)"),
    ],
)
def test_relion_scoring_rotations_reject_invalid_inputs(eulers, right_matrix, error_type, error):
    import recovar.cuda_backproject as cuda_backproject

    with pytest.raises(error_type, match=error):
        cuda_backproject.relion_make_scoring_rotations_f32.__wrapped__(jnp.asarray(eulers), jnp.asarray(right_matrix))


def test_relion_scoring_rotations_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_make_scoring_rotations_f32.__wrapped__(
            jnp.zeros((1, 3), dtype=jnp.float32),
            jnp.eye(3, dtype=jnp.float32),
        )


def test_relion_scoring_rotations_fails_closed_when_custom_cuda_disabled(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    with pytest.raises(RuntimeError, match="custom CUDA is disabled"):
        cuda_backproject.relion_make_scoring_rotations_f32.__wrapped__(
            jnp.zeros((1, 3), dtype=jnp.float32),
            jnp.eye(3, dtype=jnp.float32),
        )


def test_relion_scoring_rotations_ffi_has_no_aliases(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    call_options = {}

    def fake_ffi_call(_target, out_type, **options):
        call_options.update(options)

        def invoke(_eulers, _right_matrix, **_attrs):
            return jnp.empty(out_type.shape, out_type.dtype)

        return invoke

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda_backproject, "custom_cuda_requested", lambda: True)
    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda_backproject.jax.ffi, "ffi_call", fake_ffi_call)
    result = cuda_backproject.relion_make_scoring_rotations_f32.__wrapped__(
        jnp.zeros((2, 3), dtype=jnp.float32),
        jnp.eye(3, dtype=jnp.float32),
        do_right=False,
    )

    assert result.shape == (2, 3, 3)
    assert "input_output_aliases" not in call_options


def test_relion_scoring_rotations_f64_ffi_dtype_and_target(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    call = {}

    def fake_ffi_call(target, out_type, **options):
        call.update(target=target, out_type=out_type, options=options)

        def invoke(_eulers, _right_matrix, **attrs):
            call["attrs"] = attrs
            return jnp.empty(out_type.shape, out_type.dtype)

        return invoke

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda_backproject, "custom_cuda_requested", lambda: True)
    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cuda_backproject.jax.ffi, "ffi_call", fake_ffi_call)
    result = cuda_backproject.relion_make_scoring_rotations_f64.__wrapped__(
        jnp.zeros((2, 3), dtype=jnp.float64),
        jnp.eye(3, dtype=jnp.float64),
        do_right=False,
    )

    assert result.shape == (2, 3, 3)
    assert result.dtype == jnp.float64
    assert call["target"] == "cuda_relion_make_scoring_rotations_f64"
    assert call["attrs"]["do_right"] == np.int64(0)
    assert "input_output_aliases" not in call["options"]


@pytest.mark.parametrize(
    "eulers,right_matrix,error",
    [
        (np.zeros((2, 3), np.float32), np.eye(3, dtype=np.float64), "eulers_deg must be float64"),
        (np.zeros((2, 3), np.float64), np.eye(3, dtype=np.float32), "right_matrix must be float64"),
    ],
)
def test_relion_scoring_rotations_f64_rejects_narrow_inputs(monkeypatch, eulers, right_matrix, error):
    import recovar.cuda_backproject as cuda_backproject

    with pytest.raises(TypeError, match=error):
        cuda_backproject.relion_make_scoring_rotations_f64.__wrapped__(
            jnp.asarray(eulers), jnp.asarray(right_matrix)
        )
