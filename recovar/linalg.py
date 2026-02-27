import logging
import jax
import jax.numpy as jnp
import numpy as np
import functools
import recovar.fourier_transform_utils as fourier_transform_utils
from recovar import utils

# Some functions that do linear algera on batch in GPU. I find it strange that there is not already a decent library to do this, but I couldn't find one.

logger = logging.getLogger(__name__)

def batch_st_end(k, batch_size, n_rows):
    batch_st = int(k * batch_size)
    batch_end = int(np.min( [(k+1) * batch_size, n_rows]))
    return batch_st, batch_end

def blockwise_Y_T_X(Y,X, batch_size = None, memory_to_use = 10):
    # X and Y are tall and skinny
    if batch_size is None:
        size_of_X = utils.get_size_in_gb(X)
        size_of_Y = utils.get_size_in_gb(Y)
        n_blocks = np.ceil(4 * (size_of_X + size_of_Y) / memory_to_use).astype(int)
        batch_size = np.floor(X.shape[0] / n_blocks)
        
    n_rows = X.shape[0]
    YX = jnp.zeros_like(X, shape =[Y.shape[-1], X.shape[-1]])
    square_jit = jax.jit( lambda y, x: jnp.conj(y).T @ x)
    logger.info(f"Y^T @ X {int(np.ceil(n_rows/batch_size))} blocks") 
    for k in range(0, int(np.ceil(n_rows/batch_size))):
        batch_st, batch_end = batch_st_end(k, batch_size, n_rows)
        YX += square_jit(Y[batch_st:batch_end], X[batch_st:batch_end]) #jnp.conj(Z).T @ Z
        # utils.report_memory_device(logger =logger)
    return np.array(YX)


def blockwise_X_T_X(X, batch_size = None, memory_to_use = 10):
    # X
    if batch_size is None:
        size_of_X = utils.get_size_in_gb(X)
        n_blocks = np.ceil(size_of_X / memory_to_use).astype(int)
        batch_size = np.floor(X.shape[0] / n_blocks)
        
    n_rows = X.shape[0]
    XX = jnp.zeros_like(X, shape =[X.shape[-1], X.shape[-1]])
    square_jit = jax.jit( lambda x: jnp.conj(x).T @ x)
    logger.info(f"X^T @ X in {int(np.ceil(n_rows / batch_size))} blocks")
    
    for k in range(0, int(np.ceil(n_rows/batch_size))):
        batch_st, batch_end = batch_st_end(k, batch_size, n_rows)
        Z = jnp.array(X[batch_st:batch_end])
        XX += square_jit(Z) #jnp.conj(Z).T @ Z

    return np.array(XX)        
            

# Sometimes this takes hours for no apparent reason...
def blockwise_A_X(A, X, batch_size = None, memory_to_use = 10):
    # Blockwise multiply where # A is very tall, and X is square-ish

    if batch_size is None:
        size_of_X = utils.get_size_in_gb(X)
        usable_memory = memory_to_use - size_of_X 
        # max_item_size = A.itemsize if A.itemsize > X.itemsize else X.itemsize
        max_item_size = (A[0,0] * X[0,0]).itemsize # item size of product

        size_of_A = A.shape[0]* ( np.max([A.shape[1], X.shape[1]])) * max_item_size / 1e9
        n_blocks = np.ceil( 4 * size_of_A / usable_memory).astype(int)
        batch_size = np.floor(A.shape[0] / n_blocks)
        
    n_rows = A.shape[0]
    # Compute the bottom of fraction.
    Z = np.zeros_like(A, shape = [A.shape[0], X.shape[-1]])
    utils.report_memory_device(logger =logger)
    X = jnp.array(X)
    
    mat_mat_jit = jax.jit( lambda x, y: x @ y)
    logger.info(f"A@X in {int(np.ceil(n_rows/batch_size))} blocks") 
    for k in range(0, int(np.ceil(n_rows/batch_size))):
        batch_st, batch_end = batch_st_end(k, batch_size, n_rows)
        Z[batch_st:batch_end] = np.array(mat_mat_jit( A[batch_st:batch_end],  X))
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
    sigma_inv = np.where( sigma > epsilon , 1/ sigma, 0)
    U = (X @ Yu) * sigma_inv
    V = Yu
    return np.flip(U, axis =1), np.flip(sigma), np.flip(V, axis =1)


def randomized_svd(A, n_pcs = 200):
    '''
    For some reason, the built in svd seems to allocate a lot more memory than necessary.
    '''
    n_pcs = n_pcs if n_pcs < A.shape[1] else A.shape[1]
    gauss = np.random.randn(A.shape[1], n_pcs)
    Agauss = blockwise_A_X(A, gauss, memory_to_use = utils.get_gpu_memory_total()//3)
    qr_cpu = jax.jit(jnp.linalg.qr, backend='cpu')
    Q, _ = qr_cpu(Agauss)
    logger.info("QR done")
    Y = blockwise_Y_T_X(Q,A) #np.conj(Q).T @ A
    logger.info("Q^TA done")
    svd_cpu = jax.jit( lambda X :jnp.linalg.svd(X, full_matrices = True), backend='cpu')
    U, S, Vh = svd_cpu(Y)
    QU = blockwise_A_X(Q, U, memory_to_use = utils.get_gpu_memory_total()//3)

    return QU, S, Vh


#### batching IDFT

# Assumes input are of size (vol_size, n_vol)
# This seems like a crazy amount of reshaping/transposing
@functools.partial(jax.jit, static_argnums = [1])    
def idft3(x, vec_shape ):
    x = x.reshape([*vec_shape, x.shape[-1]]) 
    # x = x.T
    x = fourier_transform_utils.get_idft3(x, axes =(0,1,2))
    # x = x.T
    x = x.reshape([-1, x.shape[-1]])
    return x

@functools.partial(jax.jit, static_argnums = [1])
def dft3(x, vec_shape):
    x = x.reshape([*vec_shape, x.shape[-1]])
    # x = x.T
    x = fourier_transform_utils.get_dft3(x, axes =(0,1,2))
    # x = x.T
    x = x.reshape([-1, x.shape[-1]])
    return x

# Could cut down by a factor of 2 here?
def batch_idft3(x, vec_shape, batch_size):
    x_out = np.zeros_like(x) 
    n_tot = x.shape[-1]
    logger.info(f"batch_idft3 in {int(np.ceil(n_tot/batch_size))} blocks") 
    for k in range(0, int(np.ceil(n_tot/batch_size))):
        batch_st, batch_end = batch_st_end(k, batch_size, n_tot)
        x_out[:,batch_st:batch_end] = np.array(idft3(x[:,batch_st:batch_end], vec_shape = vec_shape))
    return x_out
    
    
def batch_dft3(x, vec_shape, batch_size):
    x_out = np.zeros_like(x, dtype = 'complex64')
    n_tot = x.shape[-1]
    logger.info(f"batch_dft3 in {int(np.ceil(n_tot/batch_size))} blocks")
    for k in range(0, int(np.ceil(n_tot/batch_size))):
        batch_st, batch_end = batch_st_end(k, batch_size, n_tot)
        x_out[:,batch_st:batch_end] = np.array(dft3(x[:,batch_st:batch_end], vec_shape = vec_shape))
    return x_out


def batch_dft3_2(x, vec_shape, batch_size):
    x_out = jnp.empty(x.shape, dtype = np.complex64, device =jax.devices("cpu")[0])
    n_tot = x.shape[-1]
    logger.info(f"batch_dft3 in {int(np.ceil(n_tot/batch_size))} blocks")
    for k in range(0, int(np.ceil(n_tot/batch_size))):
        batch_st, batch_end = batch_st_end(k, batch_size, n_tot)
        x_out[:,batch_st:batch_end] = (dft3(x[:,batch_st:batch_end], vec_shape = vec_shape))
    return x_out


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
