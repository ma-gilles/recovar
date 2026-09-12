"""The M transaction must give C2R a real DC coefficient in both precisions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.relion import relion_vdam_mstep as mstep

pytestmark = pytest.mark.unit


def _imaginary_dc_case(dtype, sign):
    """Real projector DC16 plus an imaginary-only M gradient; final real DC8."""
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
    data = np.zeros((19, 19, 10), complex_dtype)
    data[9, 9, 0] = sign * 2j
    return dict(
        reference_relion=np.ones((16, 16, 16), dtype),
        data_h0=data.copy(),
        weight_h0=np.ones(data.shape, dtype),
        data_h1=data.copy(),
        weight_h1=np.ones(data.shape, dtype),
        mom1_h0=np.zeros((16, 16, 9), complex_dtype),
        mom1_h1=np.zeros((16, 16, 9), complex_dtype),
        mom2=np.ones((16, 16, 9), complex_dtype),
        fsc_reconstruct=np.full(9, 0.5, dtype),
        tau2=np.ones(9, np.float64),
        grad_stepsize=dtype(1),
        tau2_fudge=dtype(1),
        r_max=np.int32(4),
        first_initializes_h0=np.bool_(True),
        first_initializes_h1=np.bool_(True),
        ori_size=16,
        padding_factor=1,
        pseudo_halfsets=True,
        compute_dtype=dtype,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sign", [-1, 1])
def test_m_step_c2r_input_has_real_dc(dtype, sign, monkeypatch):
    """Inspect the executed inverse-FFT boundary, before a backend can hide it."""
    calls = []

    def inspect_inverse(spectrum, volume_shape, *, norm):
        actual = np.asarray(spectrum)
        calls.append(actual.shape)
        assert volume_shape == (16, 16, 16)
        assert norm == "forward"
        assert actual.dtype == np.dtype(np.complex64 if dtype == np.float32 else np.complex128)
        # Only y/z are centered in this RFFT layout; x=0 contains DC.
        assert actual[8, 8, 0].imag == 0
        assert actual[8, 8, 0].real == 8
        return jnp.zeros(volume_shape, dtype=dtype)

    monkeypatch.setattr(mstep.ftu, "get_idft3_real", inspect_inverse)
    with jax.disable_jit():
        mstep.relion_vdam_m_step_device(**_imaginary_dc_case(dtype, sign))
    assert calls == [(16, 16, 9)]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sign", [-1, 1])
def test_m_step_imaginary_dc_preserves_exact_constant_map(dtype, sign):
    """Actual GPU C2R must preserve real DC and ignore an invalid imaginary DC."""
    assert jax.default_backend() == "gpu"
    result = mstep.relion_vdam_m_step_device(**_imaginary_dc_case(dtype, sign))
    control = mstep.relion_vdam_m_step_device(**_imaginary_dc_case(dtype, 0))
    # Unit step and FSC1/2 leave projector DC8. Inverse norm='forward'
    # yields constant8; the transaction divides by N16, giving exact1/2.
    expected = np.full((16, 16, 16), 0.5, dtype)
    for output in (result, control):
        assert output["iref"].dtype == np.dtype(dtype)
        assert not output["_invalid_sigma2"]
        assert not output["_invalid_tau2"]
        np.testing.assert_array_equal(output["iref"], expected)
    np.testing.assert_array_equal(result["iref"], control["iref"])
