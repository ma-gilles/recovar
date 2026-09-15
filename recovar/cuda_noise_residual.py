"""Native EM residual statistics, selected only by an explicit diagnostic option."""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject as cb

_TARGET = "recovar_noise_residual_statistics"
_registered = False


def _output_shapes(proj, abs2, summed, ctf, variance, mask):
    if proj.ndim != 3 or proj.dtype not in (jnp.complex64, jnp.complex128):
        raise ValueError("Residual projections must be complex B,R,P")
    b, r, p = proj.shape
    if not (0 < b <= 65535 and 0 < r <= 65535 and 0 < p <= 65535):
        raise ValueError("Residual geometry is empty or exceeds CUDA bounds")
    if b*r > 65535*256 or r*p > 65535*8192:
        raise ValueError("Residual scratch launch exceeds CUDA grid bounds")
    real = jnp.float64 if proj.dtype == jnp.complex128 else jnp.float32
    for value, shape, dtype in ((abs2, proj.shape, real),
                                (summed, proj.shape, proj.dtype),
                                (ctf, proj.shape, real),
                                (variance, (p,), jnp.float64),
                                (mask, (p,), jnp.bool_)):
        if value.shape != shape or value.dtype != dtype:
            raise ValueError("Residual operand shape/dtype mismatch")
    rt, it = (b*r+255)//256, (r*p+8191)//8192
    return tuple(jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in (
        ((p,), jnp.float64), ((p,), proj.dtype),
        ((b,), jnp.float64), ((b,), jnp.float64),
        ((b,), jnp.float64), ((b,), jnp.float64),
        ((rt,p), jnp.float64), ((rt,p), proj.dtype), ((b,it,4), jnp.float64)))


def _ensure_registered():
    global _registered
    cb._ensure_ffi()
    if _registered:
        return
    with cb._ffi_lock:
        if _registered:
            return
        symbol = getattr(cb._get_lib(), "NoiseResidualStatistics", None)
        if symbol is None:
            raise RuntimeError("Explicit CUDA build with NoiseResidualStatistics required")
        jax.ffi.register_ffi_target(_TARGET, jax.ffi.pycapsule(symbol), platform="CUDA")
        _registered = True


@functools.partial(jax.jit, static_argnames=("compute_scale",))
def residual_statistics(proj, abs2, summed, ctf, variance, mask, *, compute_scale=False):
    """Return pixel A2/cross and image A2/XA/masked-A2/masked-XA sums.

    Projection abs2 is supplied to preserve the original JAX arithmetic. Cross
    terms and pixel cross reduction use the input complex precision. A2 and
    image reductions retain the original float64 variance-promoted precision.
    Native reduction trees require qualification against the accepted path.

    Scratch consists of bounded reduction tiles, owned by XLA and overwritten
    completely by CUDA. Inputs are read-only. No internal allocation or host
    transfer is performed. Optional scale outputs are zero when disabled.
    """
    if type(compute_scale) is not bool:
        raise TypeError("compute_scale must be a static Python bool")
    outputs = _output_shapes(proj, abs2, summed, ctf, variance, mask)
    if jax.default_backend() != "gpu" or not cb.custom_cuda_requested():
        raise RuntimeError("Native residual statistics require enabled CUDA")
    _ensure_registered()
    result = jax.ffi.ffi_call(_TARGET, outputs, vmap_method="sequential")(
        proj, abs2, summed, ctf, variance, mask,
        compute_scale=np.int64(compute_scale))
    return tuple(result[:6])


@functools.partial(jax.jit, static_argnames=("compute_scale",))
def reference_statistics(proj, abs2, summed, ctf, variance, mask, *, compute_scale=False):
    """Original general-noise term/reduction arithmetic, exposed for diagnostics."""
    mass = ctf != 0.0
    raw = jnp.where(mass, ctf * variance[None,None,:], 0.0)
    a2 = jnp.where(mass, abs2 * raw, 0.0)
    cross = jnp.where(summed != 0.0, proj * jnp.conj(summed), 0.0)
    pixel_a2 = jnp.sum(a2.reshape(-1, a2.shape[-1]), axis=0)
    pixel_cross = jnp.sum(cross.reshape(-1, cross.shape[-1]), axis=0)
    image_a2 = jnp.sum(a2, axis=(1,2))
    image_xa = jnp.sum(variance[None,None,:] * cross.real, axis=(1,2))
    if compute_scale:
        smass = mass & mask[None,None,:]
        sraw = jnp.where(smass, ctf * variance[None,None,:], 0.0)
        sa2 = jnp.where(smass, abs2 * sraw, 0.0)
        scross = jnp.where((summed != 0.0) & mask[None,None,:],
                           proj * jnp.conj(summed), 0.0)
        scale_a2 = jnp.sum(sa2, axis=(1,2))
        scale_xa = jnp.sum(variance[None,None,:] * scross.real, axis=(1,2))
    else:
        scale_a2 = jnp.zeros_like(image_a2)
        scale_xa = jnp.zeros_like(image_xa)
    return pixel_a2, pixel_cross, image_a2, image_xa, scale_a2, scale_xa
