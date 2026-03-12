"""Linear algebra helpers: batch SVD, QR, eigendecomposition on CPU/GPU."""

import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import utils

# Batch linear algebra on GPU.

logger = logging.getLogger(__name__)

# Module-level JIT kernels — compiled once, reused across calls.
_conj_t_matmul = jax.jit(lambda y, x: jnp.conj(y).T @ x)
_gram = jax.jit(lambda x: jnp.conj(x).T @ x)
_matmul = jax.jit(lambda x, y: x @ y)


def batch_st_end(k, batch_size, n_rows):
    batch_st = int(k * batch_size)
    batch_end = int(np.min([(k+1) * batch_size, n_rows]))
    return batch_st, batch_end

def blockwise_Y_T_X(Y, X, batch_size=None, memory_to_use=10):
    # X and Y are tall and skinny; result Y^T @ X is small.
    if batch_size is None:
        size_of_X = utils.get_size_in_gb(X)
        size_of_Y = utils.get_size_in_gb(Y)
        n_blocks = np.ceil(4 * (size_of_X + size_of_Y) / memory_to_use).astype(int)
        batch_size = np.floor(X.shape[0] / n_blocks)

    n_rows = X.shape[0]
    YX = jnp.zeros((Y.shape[-1], X.shape[-1]), dtype=X.dtype)
    n_blocks = int(np.ceil(n_rows / batch_size))
    logger.info("Y^T @ X %d blocks", n_blocks)
    for k in range(n_blocks):
        batch_st, batch_end = batch_st_end(k, batch_size, n_rows)
        YX += _conj_t_matmul(Y[batch_st:batch_end], X[batch_st:batch_end])
    return np.array(YX)


def blockwise_X_T_X(X, batch_size=None, memory_to_use=10):
    # Result X^T @ X is small (cols × cols); accumulate on device.
    if batch_size is None:
        size_of_X = utils.get_size_in_gb(X)
        n_blocks = np.ceil(size_of_X / memory_to_use).astype(int)
        batch_size = np.floor(X.shape[0] / n_blocks)

    n_rows = X.shape[0]
    XX = jnp.zeros((X.shape[-1], X.shape[-1]), dtype=X.dtype)
    n_blocks = int(np.ceil(n_rows / batch_size))
    logger.info("X^T @ X in %d blocks", n_blocks)

    for k in range(n_blocks):
        batch_st, batch_end = batch_st_end(k, batch_size, n_rows)
        # JAX auto-transfers the numpy slice; no need for explicit jnp.array()
        XX += _gram(X[batch_st:batch_end])

    return np.array(XX)


def blockwise_A_X(A, X, batch_size=None, memory_to_use=10):
    if batch_size is None:
        size_of_X = utils.get_size_in_gb(X)
        usable_memory = memory_to_use - size_of_X
        max_item_size = (A[0, 0] * X[0, 0]).itemsize

        size_of_A = A.shape[0] * (np.max([A.shape[1], X.shape[1]])) * max_item_size / 1e9
        n_blocks = np.ceil(4 * size_of_A / usable_memory).astype(int)
        batch_size = np.floor(A.shape[0] / n_blocks)

    n_rows = A.shape[0]
    # Output is large (n_rows × X.cols) — must stay on CPU.
    Z = np.zeros((A.shape[0], X.shape[-1]), dtype=np.result_type(A.dtype, X.dtype))
    utils.report_memory_device(logger=logger)
    # Transfer X once; it stays resident on device for all blocks.
    X_dev = jnp.array(X)

    n_blocks = int(np.ceil(n_rows / batch_size))
    logger.info("A@X in %d blocks", n_blocks)
    for k in range(n_blocks):
        batch_st, batch_end = batch_st_end(k, batch_size, n_rows)
        Z[batch_st:batch_end] = np.array(_matmul(A[batch_st:batch_end], X_dev))
    return Z        



# Legacy SVD methods used only for debugging; prefer randomized_svd for production
def thin_svd_in_blocks(X, np = np, memory_to_use = 5, epsilon = 1e-8, n_components = -1):
    '''
    This is an unstable method for SVD but that can run on GPU.
    This should be only (maybe) used for matrices that cannot fit on the GPU.
    
    Same as SVD but dispatch matrices in blocks to gpu (also pretty unstable)
    '''
    
    Y = blockwise_X_T_X(X, memory_to_use = memory_to_use)
    Ys, Yu = np.linalg.eigh(Y)#, full_matrices = True)
    sigma = np.sqrt(np.where(Ys > 0, Ys, 0))
    # Avoid an annoying warning?
    sigma_pos = np.where( sigma > epsilon , sigma, epsilon)
    sigma_inv = np.where( sigma > epsilon , 1/ sigma_pos, 0)
    
    U = blockwise_A_X(X, Yu, memory_to_use = memory_to_use)
    U = U * sigma_inv
    V = Yu
    
    n_components = n_components if n_components > 0 else X.shape[0]
    
    return np.flip(U, axis =1)[:,:n_components], np.flip(sigma)[:n_components], np.conj(np.flip(V, axis =1))[:,:n_components].T



def thin_svd(X, np = np, epsilon = 1e-8):
    '''
    For some reason, the built in svd seems to allocate a lot more memory than necessary.
    '''
    Y = np.conj(X).T @ X
    Ys, Yu = np.linalg.eigh(Y)#, full_matrices = True)
    sigma = np.sqrt(np.where(Ys > 0, Ys, 0))
    safe_sigma = np.where(sigma > epsilon, sigma, 1)
    sigma_inv = np.where(sigma > epsilon, 1 / safe_sigma, 0)
    U = (X @ Yu) * sigma_inv
    V = Yu
    return np.flip(U, axis =1), np.flip(sigma), np.flip(V, axis =1)


_qr_jit = jax.jit(jnp.linalg.qr)
_svd_full_jit = jax.jit(lambda X: jnp.linalg.svd(X, full_matrices=True))


def randomized_svd(A, n_pcs=200):
    """Randomized SVD for large matrices that don't fit on GPU."""
    n_pcs = n_pcs if n_pcs < A.shape[1] else A.shape[1]
    rng = np.random.default_rng(0)
    gauss = rng.standard_normal((A.shape[1], n_pcs))
    Agauss = blockwise_A_X(A, gauss, memory_to_use=utils.get_gpu_memory_total() // 3)
    Q, _ = _qr_jit(Agauss)
    logger.info("QR done")
    Y = blockwise_Y_T_X(Q, A)
    logger.info("Q^TA done")
    U, S, Vh = _svd_full_jit(Y)
    QU = blockwise_A_X(Q, U, memory_to_use=utils.get_gpu_memory_total() // 3)

    return QU, S, Vh


#### batching IDFT

# Assumes input are of size (vol_size, n_vol)
@functools.partial(jax.jit, static_argnums = [1])
def idft3(x, vec_shape ):
    x = x.reshape([*vec_shape, x.shape[-1]])
    x = fourier_transform_utils.get_idft3(x, axes =(0,1,2))
    x = x.reshape([-1, x.shape[-1]])
    return x

## TODO could the batched ones be replaced with jax.numpy cpu  code rather than numpy?
# Would that be faster?
# I.e. still send data back and forth between cpu/gpu but use jax.numpy as the cpu backend as well
# .e.g batch_idft3 is allocated on numpy
@functools.partial(jax.jit, static_argnums = [1])
def dft3(x, vec_shape):
    x = x.reshape([*vec_shape, x.shape[-1]])
    x = fourier_transform_utils.get_dft3(x, axes =(0,1,2))
    x = x.reshape([-1, x.shape[-1]])
    return x
def batch_idft3(x, vec_shape, batch_size):
    x_out = np.empty_like(x)
    n_tot = x.shape[-1]
    n_blocks = int(np.ceil(n_tot / batch_size))
    logger.info("batch_idft3 in %d blocks", n_blocks)
    for k in range(n_blocks):
        batch_st, batch_end = batch_st_end(k, batch_size, n_tot)
        x_out[:, batch_st:batch_end] = jax.device_get(
            idft3(x[:, batch_st:batch_end], vec_shape=vec_shape)
        )
    return x_out


def batch_dft3(x, vec_shape, batch_size):
    x_out = np.empty(x.shape, dtype=np.complex64)
    n_tot = x.shape[-1]
    n_blocks = int(np.ceil(n_tot / batch_size))
    logger.info("batch_dft3 in %d blocks", n_blocks)
    for k in range(n_blocks):
        batch_st, batch_end = batch_st_end(k, batch_size, n_tot)
        x_out[:, batch_st:batch_end] = jax.device_get(
            dft3(x[:, batch_st:batch_end], vec_shape=vec_shape)
        )
    return x_out

## TODO: are these two functions re-implemented several times in the codebase in different places? Probably should be streamline
## Also is this the most efficient way to implement this?
def broadcast_dot(x,y):
    return jax.lax.batch_matmul(jnp.conj(x[...,None,:]),y[...,:,None])[...,0,0]

def broadcast_outer(x,y):
    return jax.lax.batch_matmul(x[...,:,None],jnp.conj(y[...,None,:]))


def inner_product(x, y):
    """Conjugate inner product for non-batched arrays of identical shape."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    return jnp.vdot(x, y)


def batch_inner_product(x, y):
    """Conjugate inner product over batch dimension 0.

    Accepts arrays shaped `(B, ...)` and returns `(B,)`.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    if x.ndim < 2:
        raise ValueError(f"batch_inner_product expects batched input with ndim>=2, got {x.ndim}")
    x_flat = x.reshape((x.shape[0], -1))
    y_flat = y.reshape((y.shape[0], -1))
    return jnp.sum(jnp.conj(x_flat) * y_flat, axis=-1)


## TODO similarly, similar functions are implemented elsewher ein codes I think. It should be cleaned up.
def _half_spectrum_to_full_spectrum(x_half, full_shape):
    full_shape = tuple(int(s) for s in full_shape)
    if len(full_shape) == 2:
        return fourier_transform_utils.half_image_to_full_image(x_half, full_shape)
    if len(full_shape) == 3:
        return fourier_transform_utils.half_volume_to_full_volume(x_half, full_shape)
    raise ValueError(f"full_shape must have 2 or 3 dims, got {full_shape}")


def half_spectrum_last_axis_weights(last_axis_size, dtype=jnp.float32):
    """Weights to recover full-spectrum inner products from packed real FFT coefficients."""
    n = int(last_axis_size)
    if n <= 0:
        raise ValueError(f"last_axis_size must be positive, got {n}")
    m = n // 2 + 1
    w = jnp.ones((m,), dtype=dtype)
    if n % 2 == 0:
        if m > 2:
            w = w.at[1:-1].set(2)
    else:
        if m > 1:
            w = w.at[1:].set(2)
    return w

def nyquist_mask(shape, half=False, dtype=jnp.float32):
    """Binary mask that zeros Nyquist frequencies (-N/2) in a spectrum.

    For even-sized axes, the Nyquist frequency has no symmetric partner,
    breaking Hermitian symmetry of interpolated slices.  Zeroing these
    frequencies restores exact symmetry for interior pixels.

    Works for 2-D images (``shape = (H, W)``) and 3-D volumes
    (``shape = (N0, N1, N2)``).

    Args:
        shape: ``(H, W)`` or ``(N0, N1, N2)`` tuple.
        half: If True, return the mask for the packed rfft layout (last axis
              has size ``N//2+1``).  If False, return the full centered-FFT mask.
        dtype: dtype for the returned array (default float32).

    Returns:
        Flattened JAX array with values in ``{0, 1}``.
    """
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)

    if half:
        out_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    else:
        out_shape = shape

    mask = jnp.ones(out_shape, dtype=dtype)
    for ax in range(ndim):
        if shape[ax] % 2 == 0:
            if half and ax == ndim - 1:
                # Last axis in half format: Nyquist is the last element
                idx = [slice(None)] * ndim
                idx[ax] = -1
                mask = mask.at[tuple(idx)].set(0)
            else:
                # Centered FFT: Nyquist is index 0
                idx = [slice(None)] * ndim
                idx[ax] = 0
                mask = mask.at[tuple(idx)].set(0)
    return mask.ravel()


def rfft2_nyquist_mask(image_shape, dtype=jnp.float32):
    """Binary mask that zeros Nyquist frequencies in a 2-D half-spectrum.

    Convenience wrapper around :func:`nyquist_mask` for 2-D half-images.
    """
    return nyquist_mask(image_shape, half=True, dtype=dtype)


## TODO this is also reimplemented elsewher eI believe
def rfft2_hermitian_weights(image_shape, dtype=jnp.float32):
    """Precompute ``sqrt(w)`` weights for 2-D half-spectrum (rfft2) inner products.

    For Hermitian-symmetric arrays ``A``, ``B`` of shape ``(H, W)`` stored in
    rfft2 half-spectrum format (shape ``(H, W//2+1)``), the weighted inner
    product over the half equals the full-spectrum inner product::

        sum_{k in half} w[k] * conj(A[k]) * B[k]  ==  sum_{k in full} conj(A[k]) * B[k]

    Built from :func:`half_spectrum_last_axis_weights`, then tiled over the H rows.

    Args:
        image_shape: ``(H, W)`` tuple.
        dtype: dtype for the returned array (default float32).

    Returns:
        JAX array of shape ``(H * (W // 2 + 1),)`` with values in ``{1, sqrt(2)}``.
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    w1d = jnp.sqrt(half_spectrum_last_axis_weights(W, dtype=dtype))  # (W//2+1,)
    return jnp.tile(w1d, H)


def _coerce_half_grid(arr, full_shape, name):
    full_shape = tuple(int(s) for s in full_shape)
    if len(full_shape) == 2:
        half_shape = fourier_transform_utils.image_shape_to_half_image_shape(full_shape)
    elif len(full_shape) == 3:
        half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(full_shape)
    else:
        raise ValueError(f"full_shape must have 2 or 3 dims, got {full_shape}")

    arr = jnp.asarray(arr)
    flat_size = int(np.prod(half_shape))
    if arr.ndim == len(half_shape) and tuple(arr.shape) == half_shape:
        return arr, False
    if arr.ndim == 1 and int(arr.shape[0]) == flat_size:
        return arr.reshape(half_shape), True
    if arr.ndim == len(half_shape) + 1 and tuple(arr.shape[-len(half_shape):]) == half_shape:
        return arr, False
    if arr.ndim == 2 and int(arr.shape[-1]) == flat_size:
        return arr.reshape((arr.shape[0],) + half_shape), True

    raise ValueError(
        f"{name} must be half-spectrum grid {half_shape}, flat {flat_size}, batched grid (B,*half_shape), "
        f"or batched flat (B,{flat_size}); got {arr.shape}"
    )


def _weighted_half_inner_products(x_half_grid, y_half_grid, full_shape, batched):
    # For packed real-FFT spectra:
    # - edge bins (k=0 and Nyquist when present) are kept as-is;
    # - interior bins represent one element of a Hermitian pair, so contribute
    #   a + conj(a) = 2*Re(a), not 2*a.
    w = half_spectrum_last_axis_weights(full_shape[-1], dtype=x_half_grid.real.dtype)
    prod = jnp.conj(x_half_grid) * y_half_grid
    edge_mask = (w == 1).astype(prod.dtype)
    interior_mask = (w == 2).astype(prod.dtype)
    weighted = prod * edge_mask + (prod + jnp.conj(prod)) * interior_mask
    if batched:
        axes = tuple(range(1, weighted.ndim))
    else:
        axes = tuple(range(weighted.ndim))
    return jnp.sum(weighted, axis=axes)


def half_spectrum_inner_product(x_half, y_half, full_shape):
    """Inner product from packed half-spectrum inputs, equivalent to full-spectrum inner product.

    This computes the weighted packed-spectrum sum directly (no full-spectrum
    reconstruction), applying factor-2 to non-edge frequencies on the packed axis.
    """
    full_shape = tuple(int(s) for s in full_shape)
    x_half_grid, _ = _coerce_half_grid(x_half, full_shape, "x_half")
    y_half_grid, _ = _coerce_half_grid(y_half, full_shape, "y_half")
    if x_half_grid.shape != y_half_grid.shape:
        raise ValueError(f"x_half and y_half must have matching shapes, got {x_half_grid.shape} and {y_half_grid.shape}")
    if x_half_grid.ndim != len(full_shape):
        raise ValueError(
            f"half_spectrum_inner_product expects non-batched input with ndim={len(full_shape)}, "
            f"got shape {x_half_grid.shape}. Use batch_half_spectrum_inner_product for batched input."
        )
    return _weighted_half_inner_products(x_half_grid, y_half_grid, full_shape, batched=False)


def batch_half_spectrum_inner_product(x_half, y_half, full_shape):
    """Batched inner products from packed half-spectrum inputs.

    Equivalent to computing full-spectrum batched inner products.
    """
    full_shape = tuple(int(s) for s in full_shape)
    x_half_grid, _ = _coerce_half_grid(x_half, full_shape, "x_half")
    y_half_grid, _ = _coerce_half_grid(y_half, full_shape, "y_half")
    if x_half_grid.shape != y_half_grid.shape:
        raise ValueError(f"x_half and y_half must have matching shapes, got {x_half_grid.shape} and {y_half_grid.shape}")
    if x_half_grid.ndim != len(full_shape) + 1:
        raise ValueError(
            f"batch_half_spectrum_inner_product expects batched input with ndim={len(full_shape)+1}, "
            f"got shape {x_half_grid.shape}"
        )
    return _weighted_half_inner_products(x_half_grid, y_half_grid, full_shape, batched=True)

def multiply_along_axis(A, B, axis):
    return jnp.swapaxes(jnp.swapaxes(A, axis, -1) * B, -1, axis)


def batch_hermitian_linear_solver(A,b):
    return jax.scipy.linalg.solve(A,b, assume_a = 'pos')

def batch_linear_solver(A,b):
    return jax.scipy.linalg.solve(A,b)




def solve_by_SVD(A,b, hermitian = False):
    U,S,Vh = jax.numpy.linalg.svd(A, hermitian = False)

    if b.ndim == A.ndim -1:
        expand = True
        b = b[...,None]
    else:
        expand = False
    
    Uhb = jax.lax.batch_matmul(jnp.conj(U.swapaxes(-1,-2)),b)/ S[...,None]
    x = jax.lax.batch_matmul(jnp.conj(Vh.swapaxes(-1,-2)),Uhb)

    if expand:
        x = x[...,0]

    return x


def l2_distance(X,Y):
    x_norm = jnp.sum(jnp.abs(X) ** 2, axis=-1, keepdims=True)
    y_norm = jnp.sum(jnp.abs(Y) ** 2, axis=-1)[None, :]
    cross = jnp.conj(X) @ Y.T
    l2_dist = x_norm - 2 * cross + y_norm
    return l2_dist.real
