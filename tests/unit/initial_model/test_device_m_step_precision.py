"""Explicit M-step precision controls; no trajectory-quality acceptance."""

from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.relion.relion_vdam_mstep import relion_vdam_m_step_device, relion_vdam_m_step_host

pytestmark = pytest.mark.unit


def _case(dtype=np.float32):
    size = 16
    capacity = size + 3
    half = (capacity, capacity, capacity // 2 + 1)
    moment = (size, size, size // 2 + 1)
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
    return dict(
        reference_relion=np.ones((size,) * 3, dtype),
        data_h0=np.zeros(half, complex_dtype),
        weight_h0=np.ones(half, dtype),
        data_h1=np.zeros(half, complex_dtype),
        weight_h1=np.ones(half, dtype),
        mom1_h0=np.zeros(moment, complex_dtype),
        mom1_h1=np.zeros(moment, complex_dtype),
        mom2=np.ones(moment, complex_dtype),
        fsc_reconstruct=np.zeros(size // 2 + 1, dtype),
        tau2=np.full(size // 2 + 1, np.nextafter(1.0, 2.0), np.float64),
        grad_stepsize=dtype(0),
        tau2_fudge=dtype(1),
        r_max=np.int32(4),
        first_initializes_h0=np.bool_(True),
        first_initializes_h1=np.bool_(True),
        ori_size=size,
        return_intermediates=True,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_constant_reference_zero_update_and_dtype_contract(dtype):
    case = _case(dtype)
    original = deepcopy(case)
    result = relion_vdam_m_step_device(**case, compute_dtype=dtype)
    real_fields = ("iref", "sigma2", "data_vs_prior", "fourier_coverage", "mom1_noise_power", "fsc_estimate")
    complex_fields = ("mom1_h0", "mom1_h1", "mom2", "gradient", "projector", "updated_projector")
    for key in real_fields:
        assert result[key].dtype == np.dtype(dtype), key
    for key in complex_fields:
        expected = np.complex64 if dtype == np.float32 else np.complex128
        assert result[key].dtype == np.dtype(expected), key
    # A constant map has only DC; zero stepsize preserves it through FFT/mask.
    np.testing.assert_array_equal(result["iref"], case["reference_relion"])
    np.testing.assert_array_equal(result["mom1_h0"], 0)
    np.testing.assert_array_equal(result["mom1_h1"], 0)
    np.testing.assert_array_equal(result["mom1_noise_power"], 0)
    assert result["sigma2"][0] == dtype(1)
    assert result["fourier_coverage"][0] == dtype(1)
    assert not result["_invalid_sigma2"]
    assert not result["_invalid_tau2"]
    # update_tau2_with_fsc=false must not round the authoritative F64 vector.
    assert result["tau2"].dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(result["tau2"], original["tau2"])
    assert np.asarray(result["tau2"])[0] != float(np.float32(original["tau2"][0]))
    for key, value in original.items():
        np.testing.assert_array_equal(case[key], value)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_explicit_first_moment_certificate_and_inactive_moments(dtype):
    case = _case(dtype)
    case["data_h0"].fill(2)
    case["data_h1"].fill(2)
    case["mom1_h0"].fill(4)
    case["mom1_h1"].fill(4)
    case["first_initializes_h0"] = np.bool_(True)
    case["first_initializes_h1"] = np.bool_(False)
    result = relion_vdam_m_step_device(**case, compute_dtype=dtype)
    center = (8, 8, 0)
    # Explicit certificates, rather than a recomputed parallel sum, choose EMA.
    assert result["mom1_h0"][center] == dtype(2)
    expected = dtype(dtype(0.9) * dtype(4)) + dtype(dtype(1.0 - 0.9) * dtype(2))
    assert result["mom1_h1"][center] == expected
    assert result["mom1_h0"][0, 0, 0] == dtype(4)
    assert result["mom1_h1"][0, 0, 0] == dtype(4)


def test_default_precision_is_explicit_float64():
    case = _case(np.float64)
    default = relion_vdam_m_step_device(**case)
    explicit = relion_vdam_m_step_device(**case, compute_dtype=jnp.float64)
    for key in default:
        np.testing.assert_array_equal(default[key], explicit[key])


@pytest.mark.parametrize("dtype", [np.float16, np.int32, np.complex64, np.bool_])
def test_invalid_precision_rejected_before_native_or_device_work(dtype):
    case = _case()
    with pytest.raises(ValueError, match="compute_dtype"):
        relion_vdam_m_step_device(**case, compute_dtype=dtype)
    host = dict(case)
    host.pop("first_initializes_h0")
    host.pop("first_initializes_h1")
    host.pop("return_intermediates")
    host["fsc_ssnr"] = case["fsc_reconstruct"]
    with pytest.raises(ValueError, match="compute_dtype"):
        relion_vdam_m_step_host(**host, compute_dtype=dtype)


def test_float32_small_fft_cannot_fall_back_to_native_double():
    case = _case()
    case["ori_size"] = 8
    with pytest.raises(ValueError, match="FFT grid >=16"):
        relion_vdam_m_step_device(**case, compute_dtype=np.float32)
    case.pop("first_initializes_h0")
    case.pop("first_initializes_h1")
    case.pop("return_intermediates")
    case["fsc_ssnr"] = case["fsc_reconstruct"]
    with pytest.raises(ValueError, match="native fallback is float64"):
        relion_vdam_m_step_host(**case, compute_dtype=np.float32)


def test_float32_fft_ir_has_no_double_payload():
    case = _case()
    ir = str(relion_vdam_m_step_device.lower(**case, compute_dtype=np.float32).compiler_ir())
    fft_lines = [line for line in ir.splitlines() if "stablehlo.fft " in line]
    assert len(fft_lines) == 2, fft_lines
    assert all("f32" in line and "f64" not in line for line in fft_lines)
    assert "complex<f64>" not in ir
    # F64 is allowed only for authoritative tau2 passthrough and coordinate
    # shell rounding. Neither may introduce a high-precision payload reduction.
    for line in ir.splitlines():
        if "stablehlo." in line and "f64" in line:
            forbidden = ("stablehlo.multiply", "stablehlo.reduce", "stablehlo.scatter", "stablehlo.complex")
            assert not any(op in line for op in forbidden), line


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_authoritative_negative_tau_cannot_round_away(dtype):
    case = _case(dtype)
    case["tau2"][0] = -1e-100
    result = relion_vdam_m_step_device(**case, compute_dtype=dtype)
    assert result["_invalid_tau2"]
    np.testing.assert_array_equal(result["tau2"], case["tau2"])
