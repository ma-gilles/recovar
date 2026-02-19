import jax.numpy as jnp

DEFAULT_FFT_NORM = "backward"


def get_1d_frequency_grid(n, voxel_size=1, scaled=False):
    # Equivalent to the old even/odd linspace logic, but cheaper and exact on integer steps.
    half = n // 2
    grid = jnp.arange(-half, n - half, dtype=jnp.float32)
    if scaled:
        grid = grid / (n * voxel_size)
    return grid


def get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled=True):
    one_d_grids = [get_1d_frequency_grid(sh, voxel_size, scaled) for sh in image_shape]
    grids = jnp.meshgrid(*one_d_grids, indexing="xy")
    return jnp.transpose(jnp.vstack([jnp.reshape(g, -1) for g in grids])).astype(one_d_grids[0].dtype)


def get_k_coordinate_of_each_pixel_3d(image_shape, voxel_size, scaled=True):
    one_d_grids = [get_1d_frequency_grid(sh, voxel_size, scaled) for sh in image_shape]
    grids = jnp.meshgrid(*one_d_grids, indexing="ij")
    return jnp.transpose(jnp.vstack([jnp.reshape(g, -1) for g in grids])).astype(one_d_grids[0].dtype)


def get_grid_of_radial_distances(image_shape, voxel_size=1, scaled=False, frequency_shift=0, rounded=True):
    # Build squared distances with broadcasting per axis to avoid materializing
    # a full (..., ndim) coordinate stack, which saves GPU memory.
    ndim = len(image_shape)
    one_d_grids = [get_1d_frequency_grid(sh, voxel_size, scaled) for sh in image_shape]

    shift = jnp.asarray(frequency_shift, dtype=one_d_grids[0].dtype)
    if shift.ndim == 0:
        shift = jnp.full((ndim,), shift, dtype=one_d_grids[0].dtype)

    radial_sq = jnp.zeros(tuple(image_shape), dtype=one_d_grids[0].dtype)
    for axis, (g, s) in enumerate(zip(one_d_grids, shift)):
        shape = [1] * ndim
        shape[axis] = image_shape[axis]
        radial_sq = radial_sq + (g.reshape(shape) - s) ** 2

    radial = jnp.sqrt(radial_sq)
    if rounded and not scaled:
        return jnp.round(radial).astype(jnp.int32)
    return radial


def get_dft(img, norm=DEFAULT_FFT_NORM):
    return jnp.fft.fftshift(jnp.fft.fft(jnp.fft.fftshift(img, axes=(-1,)), norm=norm), axes=(-1,))


def get_idft(img, norm=DEFAULT_FFT_NORM):
    return jnp.fft.ifftshift(jnp.fft.ifft(jnp.fft.ifftshift(img, axes=(-1,)), norm=norm), axes=(-1,))


def get_dft2(img, norm=DEFAULT_FFT_NORM):
    return jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.fftshift(img, axes=(-2, -1)), norm=norm),
        axes=(-2, -1),
    )


def get_idft2(img, norm=DEFAULT_FFT_NORM):
    return jnp.fft.ifftshift(
        jnp.fft.ifft2(jnp.fft.ifftshift(img, axes=(-2, -1)), norm=norm),
        axes=(-2, -1),
    )


def get_dft3(img, norm=DEFAULT_FFT_NORM, axes=(-3, -2, -1)):
    img = jnp.fft.fftshift(img, axes=axes)
    img = jnp.fft.fftn(img, axes=axes, norm=norm)
    img = jnp.fft.fftshift(img, axes=axes)
    return img


def get_idft3(img, norm=DEFAULT_FFT_NORM, axes=(-3, -2, -1)):
    img = jnp.fft.ifftshift(img, axes=axes)
    img = jnp.fft.ifftn(img, axes=axes, norm=norm)
    img = jnp.fft.ifftshift(img, axes=axes)
    return img
