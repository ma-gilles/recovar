"""Native residual ABI, masks and mixed-precision arithmetic contracts."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_noise_residual as nr
from recovar.em.helpers import projection
from recovar.em.helpers.half_spectrum import bin_shell_values_jax

pytestmark = pytest.mark.unit


def operands(dtype=np.float32, shape=(3, 259, 33), *, dyadic=False):
    rng = np.random.default_rng(7193)
    complex_dtype = np.complex128 if dtype == np.float64 else np.complex64
    def values():
        return ((rng.integers(-8,9,shape)/8) if dyadic else rng.normal(size=shape)).astype(dtype)
    proj = (values()+1j*values()).astype(complex_dtype)
    summed = (values()+1j*values()).astype(complex_dtype)
    ctf = np.abs(values())
    abs2 = (proj.real**2 + proj.imag**2).astype(dtype)
    variance = (1+np.arange(shape[-1])%4).astype(np.float64)/4
    mask = np.arange(shape[-1])%3 != 0
    summed[:,::7,:] = 0
    ctf[:,::5,:] = 0
    return proj, abs2, summed, ctf, variance, mask


@pytest.mark.parametrize("dtype", [np.float32,np.float64])
def test_reference_reconstructs_existing_helpers(dtype):
    args = jax.tree.map(jnp.asarray, operands(dtype, shape=(2,3,9), dyadic=True))
    proj, abs2, summed, ctf, variance, mask = args
    p_a2, p_cross, i_a2, i_xa, s_a2, s_xa = nr.reference_statistics(*args, compute_scale=True)
    shells = jnp.arange(9, dtype=jnp.int32)%3
    xa = jnp.where(p_cross.real != 0, variance*p_cross.real, 0.)
    actual = tuple(bin_shell_values_jax(v.astype(jnp.float32),shells,3)
                   for v in (p_a2-2*xa,p_a2,xa))
    expected = projection.compute_noise_block(proj.reshape(-1,9),abs2.reshape(-1,9),
        summed.reshape(-1,9),ctf.reshape(-1,9),variance,shells,3,True)
    for a,b in zip(actual,expected,strict=True):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal((i_a2-2*i_xa).astype(jnp.float32),
        projection.compute_norm_residual_per_image(proj,abs2,summed,ctf,variance))
    scale = jnp.array([.5,2.],dtype=dtype)
    sx,sa = projection.compute_scale_correction_terms_per_image(proj,abs2,summed,ctf,variance,scale,mask)
    np.testing.assert_array_equal((s_xa/scale).astype(jnp.float32),sx)
    np.testing.assert_array_equal((s_a2/scale**2).astype(jnp.float32),sa)


@pytest.mark.parametrize("case", ["real_projection","shape","variance_dtype","mask_dtype","empty","grid"])
def test_bad_abi_rejected(case):
    args = list(jax.tree.map(lambda a:jax.ShapeDtypeStruct(a.shape,a.dtype),operands()))
    if case == "real_projection": args[0] = jax.ShapeDtypeStruct(args[0].shape,jnp.float32)
    if case == "shape": args[1] = jax.ShapeDtypeStruct((1,2,3),jnp.float32)
    if case == "variance_dtype": args[4] = jax.ShapeDtypeStruct(args[4].shape,jnp.float32)
    if case == "mask_dtype": args[5] = jax.ShapeDtypeStruct(args[5].shape,jnp.int32)
    if case == "empty": args[0] = jax.ShapeDtypeStruct((0,3,5),jnp.complex64)
    if case == "grid": args[0] = jax.ShapeDtypeStruct((65535,65535,33),jnp.complex64)
    with pytest.raises(ValueError): nr._output_shapes(*args)


@pytest.mark.parametrize("dtype", [np.float32,np.float64])
@pytest.mark.parametrize("scale", [False,True])
@pytest.mark.gpu
def test_native_f32_with_f64_companion(dtype,scale):
    assert jax.default_backend() == "gpu"
    host = operands(dtype)
    args = jax.tree.map(jnp.asarray,host)
    expected = nr.reference_statistics(*args,compute_scale=scale)
    actual = nr.residual_statistics(*args,compute_scale=scale)
    tolerance = 1e-9 if dtype == np.float64 else 1e-6
    for a,b in zip(actual,expected,strict=True):
        # Normalized aggregate error, matching the scientific panel's scale.
        a,b=np.asarray(a),np.asarray(b)
        denominator=max(float(np.max(np.abs(b))),1.)
        assert float(np.max(np.abs(a-b)))/denominator <= tolerance
    for a,b in zip(args,host,strict=True): np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize("dtype", [np.float32,np.float64])
@pytest.mark.gpu
def test_masked_nonfinite_terms(dtype):
    args=list(operands(dtype,shape=(2,3,5),dyadic=True))
    args[0][:]=np.nan+1j*np.nan
    args[1][:]=np.inf
    args[2][:]=0
    args[3][:]=0
    # Masked products must remain zero, while unguarded variance*0 yields NaN.
    args[4][2]=np.inf
    args=jax.tree.map(jnp.asarray,args)
    actual=nr.residual_statistics(*args,compute_scale=True)
    expected=nr.reference_statistics(*args,compute_scale=True)
    for a,b in zip(actual,expected,strict=True):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize("case", ["mask_dtype", "scratch", "scale_policy"])
@pytest.mark.gpu
def test_native_rejects_malformed_buffers(case):
    args = list(jax.tree.map(jnp.asarray,operands(shape=(2,3,5))))
    outputs = list(nr._output_shapes(*args))
    nr._ensure_registered()
    scale = 0
    if case == "mask_dtype": args[-1] = args[-1].astype(jnp.int32)
    if case == "scratch": outputs[-1] = jax.ShapeDtypeStruct((2,1,3),jnp.float64)
    if case == "scale_policy": scale = 2
    with pytest.raises((ValueError,RuntimeError),match="NoiseResidual:"):
        result = jax.ffi.ffi_call(nr._TARGET,tuple(outputs),vmap_method="sequential")(
            *args,compute_scale=np.int64(scale))
        jax.block_until_ready(result)
