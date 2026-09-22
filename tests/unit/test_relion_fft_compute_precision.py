"""Explicit transform precision, with independent analytic DC checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from recovar.reconstruction.relion_functions import post_process_from_filter_v2
pytestmark = pytest.mark.unit


def _fft_dtypes(jaxpr):
    if hasattr(jaxpr, "jaxpr"): jaxpr = jaxpr.jaxpr
    result = []
    for equation in getattr(jaxpr, "eqns", []):
        if equation.primitive.name == "fft":
            result.extend(np.dtype(v.aval.dtype) for v in equation.outvars)
        for value in equation.params.values():
            for child in value if isinstance(value, (list, tuple)) else [value]:
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    result.extend(_fft_dtypes(child))
    return result


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128], ids=["production-f32", "diagnostic-f64"])
@pytest.mark.parametrize("packed", [False, True])
def test_reconstruction_fft_precision_and_exact_dc(dtype, packed):
    shape = (16, 16, 9) if packed else (16, 16, 16)
    numerator = np.zeros(shape, dtype=np.complex64)
    numerator[8, 8, 0 if packed else 8] = 4096
    numerator = jnp.asarray(numerator.reshape(-1))
    weight = jnp.ones(numerator.shape, dtype=jnp.float32)
    def compute(w, y):
        return post_process_from_filter_v2(
            w, y, (8, 8, 8), 2, tau=None, use_spherical_mask=False,
            grid_correct=False, input_half_volume=packed, preserve_output_precision=True,
            fft_compute_dtype=dtype,
        )
    result = np.asarray(compute(weight, numerator)).reshape(8, 8, 8)
    assert result.dtype == dtype
    expected = np.zeros((8, 8, 8), dtype=dtype)
    expected[4, 4, 4] = 512
    np.testing.assert_array_equal(result, expected)
    dtypes = _fft_dtypes(jax.make_jaxpr(compute)(weight, numerator))
    assert len(dtypes) == 2
    expected_bits = 32 if dtype == np.complex64 else 64
    assert all(dt.itemsize * 8 == expected_bits * (2 if dt.kind == "c" else 1) for dt in dtypes)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128], ids=["production-f32", "diagnostic-f64"])
def test_em_reconstruction_keeps_fft_precision_with_double_tau_and_gridding(dtype):
    from recovar.em.refinement.mean_helpers import _reconstruct_volume_eager
    numerator = jnp.ones(4096, dtype=dtype)
    weight = jnp.ones(4096, dtype=jnp.float32 if dtype == np.complex64 else jnp.float64)
    tau = jnp.linspace(0.1, 2.0, 512, dtype=jnp.float64)
    def compute(w, y, t):
        return _reconstruct_volume_eager(w, y, (8, 8, 8), 2, t, 1.0, 2,
                                        preserve_output_precision=True, relion_filter_scale=8**4)
    result = compute(weight, numerator, tau)
    assert result.dtype == dtype
    assert np.all(np.isfinite(np.asarray(result)))
    assert _fft_dtypes(jax.make_jaxpr(compute)(weight, numerator, tau)) == [np.dtype(dtype), np.dtype(dtype)]
    assert tau.dtype == jnp.float64
