"""Sparse-PCA basis utilities used by the PPCA compatibility layer."""

import jax.numpy as jax_numpy
import jaxwt
import numpy as np
import pywt

import recovar.core.fourier_transform_utils as ftu
from recovar import utils

use_np = False
use_np_ft = False  # Always use JAX FFT for GPU acceleration
if use_np:
    jnp = np
else:
    jnp = jax_numpy

use_jaxwt = True


# Keep these for backward compatibility (exported in __init__.py)
def jax_ift(u, volume_shape):
    """Inverse FFT - wrapper for compatibility."""
    return get_ft_U(u, volume_shape, inverse=True)


def jax_ft(u, volume_shape):
    """Forward FFT - wrapper for compatibility."""
    return get_ft_U(u, volume_shape, inverse=False)


def get_ft_U(u, volume_shape, inverse=True):
    """
    Optimized Fourier transform utility.

    Args:
        u: Input array of shape (batch, volume_size)
        volume_shape: 3D volume shape tuple
        inverse: If True, compute inverse FFT; else forward FFT

    Returns:
        Transformed array of shape (batch, volume_size)
    """
    expected_vol_size = np.prod(volume_shape)

    if use_np_ft:
        # CPU FFT
        if inverse:
            return np.fft.ifftshift(
                np.fft.ifftn(
                    np.fft.ifftshift(u.reshape([u.shape[0], *volume_shape]), axes=(-3, -2, -1)),
                    axes=(-3, -2, -1),
                ),
                axes=(-3, -2, -1),
            ).reshape([u.shape[0], expected_vol_size])
        return np.fft.fftshift(
            np.fft.fftn(
                np.fft.fftshift(u.reshape([u.shape[0], *volume_shape]), axes=(-3, -2, -1)),
                axes=(-3, -2, -1),
            ),
            axes=(-3, -2, -1),
        ).reshape([u.shape[0], expected_vol_size])
    else:
        # GPU FFT (optimized path)
        # Convert to JAX if needed (check type to avoid unnecessary conversion)
        if not isinstance(u, jnp.ndarray):
            u_jax = jnp.array(u)
        else:
            u_jax = u

        # Ensure correct shape for FFT
        if u_jax.shape[1] != expected_vol_size:
            raise ValueError(
                f"Input shape {u_jax.shape} doesn't match volume_shape {volume_shape}. "
                f"Expected ({u_jax.shape[0]}, {expected_vol_size}) but got {u_jax.shape}"
            )

        if inverse:
            u_ft = ftu.get_idft3(u_jax.reshape([u_jax.shape[0], *volume_shape]))
        else:
            u_ft = ftu.get_dft3(u_jax.reshape([u_jax.shape[0], *volume_shape]))

        # Reshape back to (batch, volume_size)
        return u_ft.reshape([u_ft.shape[0], expected_vol_size])


def get_sparse_PCA_in_basis(u, sigma, to_basis_fun, from_basis_fun, percentile_used):
    u = np.array(u.T).copy()
    sigma = np.array(sigma).copy()

    basis_u = to_basis_fun(u)
    basis_variance = utils.estimate_variance(basis_u, sigma)
    del u

    # Wavelet threshold
    variance_threshold = np.percentile(basis_variance, percentile_used)
    valid_variance_indices = np.abs(basis_variance) > variance_threshold

    thresh_wavelet_vars = basis_variance.copy()
    thresh_wavelet_vars[~valid_variance_indices] = 0

    basis_var_img = from_basis_fun(basis_variance[None, ...])[0]
    basis_var_img_thresholded = from_basis_fun(thresh_wavelet_vars[None, ...])[0]
    sparse_basis_u = basis_u[..., valid_variance_indices]

    # EBU * Sigma^{1/2}
    sparse_basis_u_sigma = sparse_basis_u * np.sqrt(sigma[..., None])

    # EBU * Sigma^{1/2} = V G W^T =>
    sparse_V, sparse_sigma, _ = jnp.linalg.svd(sparse_basis_u_sigma.T, full_matrices=False)
    n_columns_trunc = np.min(sparse_V.shape)

    # E^T V
    expanded_sparse_V = jnp.zeros([valid_variance_indices.shape[0], n_columns_trunc], sparse_V.dtype)
    if use_np:
        expanded_sparse_V[valid_variance_indices] = sparse_V
    else:
        expanded_sparse_V = expanded_sparse_V.at[valid_variance_indices].set(sparse_V)
    # B^* E^T V

    Binv_expanded_sparse_V = from_basis_fun(expanded_sparse_V.T)

    sparsified_u = Binv_expanded_sparse_V
    sparsified_s = sparse_sigma**2
    return (
        np.array(sparsified_u).T,
        np.array(sparsified_s),
        np.array(basis_var_img),
        np.array(basis_var_img_thresholded),
    )


def measure_orthogonality(KK):
    XX = jnp.linalg.norm(jnp.conj(KK).T @ KK - jnp.eye(KK.shape[-1]))
    return XX


## Wavelet transforms
def wavelet_dict_to_wavelet_vec(wavelet_dict, keys_order):
    wavelet_vec = [wavelet_dict[key].reshape([wavelet_dict[key].shape[0], -1]) for key in keys_order]
    wavelet_vec = np.concatenate(wavelet_vec, axis=-1)
    return wavelet_vec


def wavelet_vec_to_wavelet_dict(wavelet_vec, keys_order, wavelet_dict_shape):
    current_idx = 0
    wavelet_dict = {}

    for key in keys_order:
        size_of_block = np.prod(wavelet_dict_shape[key])
        wavelet_dict[key] = wavelet_vec[:, current_idx : current_idx + size_of_block].reshape(
            [-1] + list(wavelet_dict_shape[key])
        )

        #         if use_np:
        #             wavelet_dict[key] = wavelet_vec[:,current_idx:current_idx+size_of_block].reshape([-1] + list( wavelet_dict_shape[key]))
        #         else:
        #             wavelet_dict[key] = wavelet_vec.at[:,current_idx:current_idx+size_of_block].get().reshape([-1] + list( wavelet_dict_shape[key]))

        current_idx += size_of_block
    return wavelet_dict


class Basis:
    def to_basis(self, image):
        raise NotImplementedError

    def to_image(self, basis_coeffs):
        raise NotImplementedError


class Wavelet(Basis):
    def __init__(self, volume_shape, wavelet_type, wavelet_mode="symmetric"):
        wavelet_dict_tmp = pywt.dwtn(np.zeros(volume_shape), wavelet=wavelet_type, mode=wavelet_mode)

        keys_order = list(wavelet_dict_tmp.keys())
        wavelet_dict_shape = {}
        for key in keys_order:
            wavelet_dict_shape[key] = wavelet_dict_tmp[key].shape
        self.wavelet_type = wavelet_type
        self.wavelet_mode = wavelet_mode
        self.keys_order = keys_order
        self.wavelet_dict_shape = wavelet_dict_shape
        self.volume_shape = volume_shape
        self.volume_size = np.prod(volume_shape)
        self.base_name = "wavelet"

    def basis_to_image_ft_single(self, wavelet_vec):
        # print(keys_order)

        wavelet_dict = wavelet_vec_to_wavelet_dict(wavelet_vec, self.keys_order, self.wavelet_dict_shape)
        real_image = pywt.idwtn(
            wavelet_dict, wavelet=self.wavelet_type, mode=self.wavelet_mode, axes=(-3, -2, -1)
        ).reshape([wavelet_vec.shape[0], -1])
        return real_image

    def image_ft_to_basis_single(self, real_image):
        wavelet_dict = pywt.dwtn(
            real_image.reshape([-1] + list(self.volume_shape)),
            wavelet=self.wavelet_type,
            mode=self.wavelet_mode,
            axes=(-3, -2, -1),
        )
        wavelet_vec = wavelet_dict_to_wavelet_vec(wavelet_dict, self.keys_order)
        return wavelet_vec

    def to_basis(self, images):
        cols_real = get_ft_U(images, self.volume_shape, inverse=True) * np.sqrt(self.volume_size)
        if np.linalg.norm(cols_real.imag) > 1e-6 * np.linalg.norm(cols_real.real):
            print(
                "Imaginary part is non-zero! ratio of imaginary to real:",
                np.linalg.norm(cols_real.imag) / np.linalg.norm(cols_real.real),
            )
        cols_all_wavelet = self.image_ft_to_basis_single(cols_real.real)
        return cols_all_wavelet

    def to_image(self, basis_coeffs):
        BT_ET_EBFU = self.basis_to_image_ft_single(basis_coeffs)
        FT_BT_ET_EBFU = get_ft_U(BT_ET_EBFU, self.volume_shape, inverse=False) / np.sqrt(self.volume_size)
        return FT_BT_ET_EBFU

    def name(self):
        return self.base_name + self.wavelet_type


class Wavelet_multilvl(Basis):
    def __init__(self, volume_shape, wavelet_type, wavelet_mode="symmetric", mask=None, backend=None):
        """
        Multi-level wavelet basis.

        Parameters
        ----------
        backend : ``"jax"`` | ``"jaxwt"`` | ``"pywt"`` | None
            Which library to use for 3-D wavelet transforms.
            *None* (default) uses ``"jax"`` (pure JAX, no external deps).

            ``"jax"`` — pure JAX implementation (``recovar.ppca.wavelets``),
            GPU-native, JIT-compilable, handles complex data natively.
            ``"jaxwt"`` — legacy jaxwt library.
            ``"pywt"`` — CPU-only PyWavelets.
        """
        self.volume_shape = volume_shape
        self.wavelet_type = wavelet_type
        self.wavelet_mode = wavelet_mode
        self.volume_size = np.prod(volume_shape)
        self.base_name = "wavelet_multilvl"
        self.from_ft = True
        self.mask = mask

        # Resolve backend
        if backend is None:
            self.backend = "jax"
        else:
            if backend not in ("jax", "jaxwt", "pywt"):
                raise ValueError(f"Unknown wavelet backend: {backend!r}")
            self.backend = backend

        # Initialize metadata by doing a dummy transform
        dummy = np.zeros([1, *volume_shape])
        dummy_coeffs = self._wavedec3(dummy)
        _, self.shapes_info = jaxwt_coeffs_to_flat_vector(dummy_coeffs)

    # -- backend dispatch helpers ------------------------------------------

    def _wavedec3(self, data):
        """Forward 3-D wavelet transform dispatched by *self.backend*."""
        wt, mode = self.wavelet_type, self.wavelet_mode
        if self.backend == "jax":
            from recovar.ppca.wavelets import wavedec3 as _jax_wavedec3

            return _jax_wavedec3(data, wavelet=wt, mode=mode)
        elif self.backend == "jaxwt":
            return jaxwt.wavedec3(data, wavelet=wt, mode=mode, axes=(-3, -2, -1))
        else:
            return pywt.wavedecn(data, wavelet=wt, mode=mode, axes=(-3, -2, -1))

    def _waverec3(self, coeffs):
        """Inverse 3-D wavelet transform dispatched by *self.backend*."""
        wt, mode = self.wavelet_type, self.wavelet_mode
        if self.backend == "jax":
            from recovar.ppca.wavelets import waverec3 as _jax_waverec3

            return _jax_waverec3(coeffs, wavelet=wt)
        elif self.backend == "jaxwt":
            return jaxwt.waverec3(coeffs, wavelet=wt, axes=(-3, -2, -1))
        else:
            return pywt.waverecn(coeffs, wavelet=wt, mode=mode, axes=(-3, -2, -1))

    # -- forward / inverse -------------------------------------------------

    def image_ft_to_basis_single(self, real_image):
        """Convert real-space images to wavelet coefficient vectors."""
        # Handle both batched and non-batched inputs
        if real_image.ndim == 1:
            # Single image without batch dimension: (volume_size,)
            real_image = real_image[None, :]  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        # Reshape to (batch, d, d, d)
        batch_size = real_image.shape[0]
        real_image_3d = real_image.reshape([batch_size] + list(self.volume_shape))

        # Wavelet transform
        wavelet_coeffs = self._wavedec3(real_image_3d)

        # Convert to flat vector
        flat_vec, _ = jaxwt_coeffs_to_flat_vector(wavelet_coeffs)

        # Remove batch dimension if input didn't have one
        if squeeze_output:
            flat_vec = flat_vec[0]

        return flat_vec

    def to_basis(self, images):
        """Convert Fourier-space images to wavelet coefficient vectors."""
        # Handle both batched and non-batched inputs
        input_is_1d = images.ndim == 1
        if input_is_1d:
            images = images[None, :]  # Add batch dimension

        if self.from_ft:
            # Inverse Fourier transform to real space
            cols_real = get_ft_U(images, self.volume_shape, inverse=True) * np.sqrt(self.volume_size)
        else:
            cols_real = images

        if self.mask is not None:
            cols_real *= self.mask[None]

        cols_all_wavelet = self.image_ft_to_basis_single(cols_real)

        # Remove batch dimension if input didn't have one
        if input_is_1d:
            cols_all_wavelet = cols_all_wavelet[0]

        return cols_all_wavelet

    def to_image_ft_single(self, wavelet_vec):
        """Convert wavelet coefficient vectors back to real-space images."""
        # Handle both batched and non-batched inputs
        if wavelet_vec.ndim == 1:
            # Single vector without batch dimension
            wavelet_vec = wavelet_vec[None, :]  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        # Convert flat vector to wavelet coefficients
        use_jax = self.backend in ("jax", "jaxwt")
        wavelet_coeffs = flat_vector_to_jaxwt_coeffs(wavelet_vec, self.shapes_info, use_jax=use_jax)

        # Inverse wavelet transform
        image = self._waverec3(wavelet_coeffs)

        # Reshape to (batch, volume_size)
        image_flat = image.reshape(-1, np.prod(self.volume_shape))

        # Remove batch dimension if input didn't have one
        if squeeze_output:
            image_flat = image_flat[0]

        return image_flat

    def to_image(self, basis_coeffs):
        """Convert wavelet coefficient vectors back to Fourier-space images."""
        # Handle both batched and non-batched inputs
        input_is_1d = basis_coeffs.ndim == 1
        if input_is_1d:
            basis_coeffs = basis_coeffs[None, :]  # Add batch dimension

        # Inverse wavelet transform to real space
        BT_ET_EBFU = self.to_image_ft_single(basis_coeffs)

        if self.from_ft:
            # Forward Fourier transform to Fourier space
            FT_BT_ET_EBFU = get_ft_U(BT_ET_EBFU, self.volume_shape, inverse=False) / np.sqrt(self.volume_size)
        else:
            FT_BT_ET_EBFU = BT_ET_EBFU

        # Remove batch dimension if input didn't have one
        if input_is_1d:
            FT_BT_ET_EBFU = FT_BT_ET_EBFU[0]

        return FT_BT_ET_EBFU

    def name(self):
        return self.base_name + self.wavelet_type


class Spatial(Basis):
    def __init__(self, volume_shape, mask=None):
        self.volume_shape = volume_shape
        self.volume_size = np.prod(volume_shape)
        self.base_name = "spatial"
        self.mask = np.ones(self.volume_size) if mask is None else mask

    def to_basis(self, images):
        cols_real = get_ft_U(images, self.volume_shape, inverse=True) * np.sqrt(self.volume_size) * self.mask[None]
        return cols_real

    def to_image(self, basis_coeffs):
        FT_BT_ET_EBFU = get_ft_U(basis_coeffs, self.volume_shape, inverse=False) / np.sqrt(self.volume_size)
        return FT_BT_ET_EBFU

    def name(self):
        return self.base_name


class Identity(Basis):
    def __init__(self, volume_shape, mask=None):
        self.volume_shape = volume_shape
        self.volume_size = np.prod(volume_shape)
        self.base_name = "identity"
        self.mask = np.ones(self.volume_size) if mask is None else mask

    def to_basis(self, images):
        return images

    def to_image(self, basis_coeffs):
        return basis_coeffs

    def name(self):
        return self.base_name


def coeffs_to_array(coeff_dict, axes=(-3, -2, -1)):

    if use_jaxwt:
        for i in range(len(coeff_dict)):
            if isinstance(coeff_dict[i], dict):
                # Handle dictionary case
                for k in coeff_dict[i].keys():
                    coeff_dict[i][k] = np.array(coeff_dict[i][k])
            else:
                # Handle non-dictionary case (e.g. approximation coefficients)
                coeff_dict[i] = np.array(coeff_dict[i])

    coeffs_array, _ = pywt.coeffs_to_array(coeff_dict, axes=axes)
    return coeffs_array, _


def wavelet_coeffs_to_vector(coeff_list):
    """
    Convert jaxwt wavedec3 output (list of arrays/dicts) to a flat vector.
    Works with batches and is JAX-jittable.

    Args:
        coeff_list: Output from jaxwt.wavedec3, a list where:
            - coeff_list[0] is approximation coefficients with shape (batch, d0, d1, d2)
            - coeff_list[i] for i>0 are dicts with keys like 'aad', 'ada', etc.
              each containing arrays of shape (batch, d0, d1, d2)

    Returns:
        vector: Flattened array of shape (batch, total_coeffs)
        metadata: Dictionary with shapes and structure info for reconstruction
    """
    batch_size = coeff_list[0].shape[0]
    vectors = []
    metadata = {"batch_size": batch_size, "structure": []}

    for level_idx, level_coeffs in enumerate(coeff_list):
        if isinstance(level_coeffs, dict):
            # Detail coefficients - store in consistent order
            level_meta = {"type": "dict", "keys": []}
            for key in sorted(level_coeffs.keys()):  # Sort for consistency
                arr = level_coeffs[key]
                level_meta["keys"].append((key, arr.shape))
                # Reshape: (batch, d0, d1, d2) -> (batch, d0*d1*d2)
                vectors.append(arr.reshape(batch_size, -1))
            metadata["structure"].append(level_meta)
        else:
            # Approximation coefficients
            level_meta = {"type": "array", "shape": level_coeffs.shape}
            vectors.append(level_coeffs.reshape(batch_size, -1))
            metadata["structure"].append(level_meta)

    # Concatenate all vectors along the feature dimension
    vector = jnp.concatenate(vectors, axis=1) if not use_np else np.concatenate(vectors, axis=1)

    return vector, metadata


def vector_to_wavelet_coeffs(vector, metadata):
    """
    Convert flat vector back to jaxwt wavedec3 format.
    Works with batches and is JAX-jittable.

    Args:
        vector: Flattened array of shape (batch, total_coeffs)
        metadata: Dictionary returned by wavelet_coeffs_to_vector

    Returns:
        coeff_list: List structure compatible with jaxwt.waverec3
    """
    batch_size = metadata["batch_size"]
    coeff_list = []
    current_idx = 0

    for level_meta in metadata["structure"]:
        if level_meta["type"] == "dict":
            # Reconstruct detail coefficients dictionary
            level_dict = {}
            for key, shape in level_meta["keys"]:
                n_elements = np.prod(shape[1:])  # Elements per sample
                level_dict[key] = vector[:, current_idx : current_idx + n_elements].reshape(shape)
                current_idx += n_elements
            coeff_list.append(level_dict)
        else:
            # Reconstruct approximation coefficients
            shape = level_meta["shape"]
            n_elements = np.prod(shape[1:])
            coeff_list.append(vector[:, current_idx : current_idx + n_elements].reshape(shape))
            current_idx += n_elements

    return coeff_list


# JAX-jittable versions using static metadata
def create_wavelet_transform_functions(volume_shape, wavelet_type="db1", wavelet_mode="symmetric"):
    """
    Create JAX-jittable forward and inverse wavelet transform functions with fixed metadata.

    Args:
        volume_shape: Shape of 3D volume (d, d, d)
        wavelet_type: Wavelet type (e.g., 'db1')
        wavelet_mode: Mode for padding (e.g., 'symmetric')

    Returns:
        to_vector: Function (batch, d, d, d) -> (batch, n_coeffs)
        from_vector: Function (batch, n_coeffs) -> (batch, d, d, d)
        metadata: Static metadata for the transforms
    """
    # Get metadata from a dummy transform
    dummy = np.zeros([1, *volume_shape])
    if use_jaxwt:
        dummy_coeffs = jaxwt.wavedec3(dummy, wavelet=wavelet_type, mode=wavelet_mode, axes=(-3, -2, -1))
    else:
        dummy_coeffs = pywt.wavedecn(dummy, wavelet=wavelet_type, mode=wavelet_mode, axes=(-3, -2, -1))

    _, metadata = wavelet_coeffs_to_vector(dummy_coeffs)

    def to_vector(images):
        """Convert images to wavelet coefficient vectors."""
        if use_jaxwt:
            coeffs = jaxwt.wavedec3(images, wavelet=wavelet_type, mode=wavelet_mode, axes=(-3, -2, -1))
        else:
            coeffs = pywt.wavedecn(images, wavelet=wavelet_type, mode=wavelet_mode, axes=(-3, -2, -1))
        vec, _ = wavelet_coeffs_to_vector(coeffs)
        return vec

    def from_vector(vector):
        """Convert wavelet coefficient vectors back to images."""
        coeffs = vector_to_wavelet_coeffs(vector, metadata)
        if use_jaxwt:
            images = jaxwt.waverec3(coeffs, wavelet=wavelet_type, mode=wavelet_mode, axes=(-3, -2, -1))
        else:
            images = pywt.waverecn(coeffs, wavelet=wavelet_type, mode=wavelet_mode, axes=(-3, -2, -1))
        return images

    return to_vector, from_vector, metadata


def wavelet_avg_square_by_level_both(volume, wavelet_type="db1"):
    """
    Compute average square of wavelet coefficients by level for a single volume.
    Returns both:
    1. Array where each entry has the average square value for its level
    2. Regular dictionary with level information
    """

    if use_jaxwt:
        coeff_dict = jaxwt.wavedec3(volume[None], wavelet=wavelet_type, mode="symmetric", axes=(-3, -2, -1))
    else:
        coeff_dict = pywt.wavedecn(volume, wavelet=wavelet_type, mode="periodization")
    # coeff_dict2 = pywt.wavedecn(volume, wavelet=wavelet_type, mode='periodization')

    # for key in coeff_dict:
    #     if key in coeff_dict2:
    #         if np.linalg.norm(coeff_dict[key] - coeff_dict2[key]) > 1e-6:
    #             print("BAD")
    # import ipdb; ipdb.set_trace()
    # Get all coefficients as a flat array
    coeffs_array, _ = coeffs_to_array(coeff_dict, axes=(-3, -2, -1))
    total_coeffs = coeffs_array.size

    # Initialize result array
    avg_square_array = np.zeros(total_coeffs)
    current_idx = 0

    # Initialize level results dictionary
    level_results = {}

    for level_idx, coeff_data in enumerate(coeff_dict):
        # Determine level name and type
        if level_idx == 0:
            level_key = f"a{len(coeff_dict) - 1}"  # Approximation level
            level_type = "Approximation"
        else:
            level_key = f"d{len(coeff_dict) - level_idx}"  # Detail level
            level_type = "Detail"

        # Extract coefficients for this level
        if isinstance(coeff_data, dict):
            coeffs = np.concatenate([arr.flatten() for arr in coeff_data.values()])
        else:
            coeffs = coeff_data.flatten()

        # Compute average square for this level
        avg_square = np.mean(coeffs**2)
        total_energy = np.sum(coeffs**2)
        n_coefficients = len(coeffs)

        # Fill the array with the average square value for this level
        avg_square_array[current_idx : current_idx + n_coefficients] = avg_square
        current_idx += n_coefficients

        # Store in level results dictionary
        level_results[level_key] = {
            "avg_square": avg_square,
            "total_energy": total_energy,
            "n_coefficients": n_coefficients,
            "level_type": level_type,
            "coeffs": coeffs,
        }

        print(f"{level_key} ({level_type}): {n_coefficients} coeffs, avg square: {avg_square:.6f}")

    return avg_square_array, level_results


# =============================================================================
# Standalone wavelet coefficient conversion functions for linear algebra
# =============================================================================


def jaxwt_coeffs_to_flat_vector(coeff_list):
    """
    Convert jaxwt.wavedec3 coefficient list to a flat vector for linear algebra.
    Works with batches. JAX-jittable if coeff_list contains JAX arrays.

    Args:
        coeff_list: Output from jaxwt.wavedec3, a list where:
            - coeff_list[0]: approximation coefficients, shape (batch, d0, d1, d2)
            - coeff_list[i] (i>0): dict with keys like 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd'
              each containing arrays of shape (batch, d0, d1, d2)

    Returns:
        flat_vector: Array of shape (batch, total_coeffs)
        shapes_info: List of (type, shape_or_keys) for reconstruction
    """
    batch_size = coeff_list[0].shape[0]
    flat_parts = []
    shapes_info = []

    for level_idx, level_data in enumerate(coeff_list):
        if isinstance(level_data, dict):
            # Detail coefficients - process in sorted key order for consistency
            keys_sorted = sorted(level_data.keys())
            for key in keys_sorted:
                arr = level_data[key]
                spatial_shape = arr.shape[1:]  # (d0, d1, d2)
                flat_parts.append(arr.reshape(batch_size, -1))
                shapes_info.append(("dict_key", key, spatial_shape))
        else:
            # Approximation coefficients
            spatial_shape = level_data.shape[1:]  # (d0, d1, d2)
            flat_parts.append(level_data.reshape(batch_size, -1))
            shapes_info.append(("approx", spatial_shape))

    # Concatenate along feature dimension
    if use_np:
        flat_vector = np.concatenate(flat_parts, axis=1)
    else:
        flat_vector = jnp.concatenate(flat_parts, axis=1)

    return flat_vector, shapes_info


def flat_vector_to_jaxwt_coeffs(flat_vector, shapes_info, use_jax=None):
    """
    Convert flat vector back to jaxwt.wavedec3 coefficient list format.
    Works with batches. JAX-jittable if flat_vector is a JAX array.

    Args:
        flat_vector: Array of shape (batch, total_coeffs)
        shapes_info: List of (type, shape_or_keys) from jaxwt_coeffs_to_flat_vector
        use_jax: If True, convert to JAX arrays. If None, infer from input type.

    Returns:
        coeff_list: List structure compatible with jaxwt.waverec3
    """
    batch_size = flat_vector.shape[0]
    coeff_list = []
    current_idx = 0
    current_dict = None
    prev_key = None

    # Determine if we should use JAX arrays
    if use_jax is None:
        # Check if input is JAX array
        use_jax = hasattr(flat_vector, "__array_namespace__") or str(type(flat_vector)).find("jax") >= 0

    # Import jax if needed
    if use_jax:
        import jax.numpy as jnp_local

    for info in shapes_info:
        if info[0] == "approx":
            # Approximation coefficients - always add any pending dict first
            if current_dict is not None:
                coeff_list.append(current_dict)
                current_dict = None
                prev_key = None

            spatial_shape = info[1]
            n_elements = np.prod(spatial_shape)
            arr = flat_vector[:, current_idx : current_idx + n_elements].reshape(batch_size, *spatial_shape)
            if use_jax:
                arr = jnp_local.array(arr)
            coeff_list.append(arr)
            current_idx += n_elements
        elif info[0] == "dict_key":
            # Detail coefficients
            key = info[1]
            spatial_shape = info[2]
            n_elements = np.prod(spatial_shape)
            arr = flat_vector[:, current_idx : current_idx + n_elements].reshape(batch_size, *spatial_shape)
            if use_jax:
                arr = jnp_local.array(arr)

            # Start a new dict if:
            # 1. No current dict exists, OR
            # 2. The key cycles back (e.g., 'ddd' -> 'aad' indicates new level)
            if current_dict is None or (prev_key is not None and key <= prev_key):
                if current_dict is not None:
                    coeff_list.append(current_dict)
                current_dict = {}

            current_dict[key] = arr
            prev_key = key
            current_idx += n_elements

    # Don't forget the last dict
    if current_dict is not None:
        coeff_list.append(current_dict)

    return coeff_list


def test_wavelet_vector_conversion(
    volume_shape=(32, 32, 32), batch_size=3, wavelet_type="db1", wavelet_mode="symmetric"
):
    """
    Test function to verify the conversion works correctly.

    Args:
        volume_shape: Shape of 3D volumes
        batch_size: Number of volumes in batch
        wavelet_type: Wavelet type
        wavelet_mode: Wavelet mode

    Returns:
        bool: True if test passes
    """
    print("Testing wavelet vector conversion...")
    print(f"  Volume shape: {volume_shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Wavelet: {wavelet_type}, mode: {wavelet_mode}")

    # Create random test volumes
    test_volumes = np.random.randn(batch_size, *volume_shape)

    # Forward transform
    if use_jaxwt:
        coeffs = jaxwt.wavedec3(test_volumes, wavelet=wavelet_type, mode=wavelet_mode, axes=(-3, -2, -1))
    else:
        coeffs = pywt.wavedecn(test_volumes, wavelet=wavelet_type, mode=wavelet_mode, axes=(-3, -2, -1))

    print("\nOriginal coeffs structure:")
    for i, c in enumerate(coeffs):
        if isinstance(c, dict):
            print(f"  Level {i}: dict with keys {list(c.keys())}, shapes: {[c[k].shape for k in c.keys()]}")
        else:
            print(f"  Level {i}: array with shape {c.shape}")

    # Convert to flat vector
    flat_vec, shapes_info = jaxwt_coeffs_to_flat_vector(coeffs)
    print(f"\nFlat vector shape: {flat_vec.shape}")
    print(f"Shapes info: {len(shapes_info)} entries")

    # Convert back (use JAX arrays for jaxwt compatibility)
    coeffs_reconstructed = flat_vector_to_jaxwt_coeffs(flat_vec, shapes_info, use_jax=use_jaxwt)

    print("\nReconstructed coeffs structure:")
    for i, c in enumerate(coeffs_reconstructed):
        if isinstance(c, dict):
            print(f"  Level {i}: dict with keys {list(c.keys())}, shapes: {[c[k].shape for k in c.keys()]}")
        else:
            print(f"  Level {i}: array with shape {c.shape}")

    # Verify reconstruction
    max_error = 0
    for i, (orig, recon) in enumerate(zip(coeffs, coeffs_reconstructed)):
        if isinstance(orig, dict):
            for key in orig.keys():
                error = np.linalg.norm(orig[key] - recon[key])
                max_error = max(max_error, error)
                if error > 1e-10:
                    print(f"  Level {i}, key {key}: error = {error}")
        else:
            error = np.linalg.norm(orig - recon)
            max_error = max(max_error, error)
            if error > 1e-10:
                print(f"  Level {i}: error = {error}")

    print(f"\nMax reconstruction error: {max_error}")

    # Inverse transform
    if use_jaxwt:
        volumes_reconstructed = jaxwt.waverec3(coeffs_reconstructed, wavelet=wavelet_type, axes=(-3, -2, -1))
    else:
        volumes_reconstructed = pywt.waverecn(
            coeffs_reconstructed, wavelet=wavelet_type, mode=wavelet_mode, axes=(-3, -2, -1)
        )

    # Check final reconstruction
    volume_error = np.linalg.norm(test_volumes - volumes_reconstructed) / np.linalg.norm(test_volumes)
    print(f"Volume reconstruction relative error: {volume_error}")

    success = max_error < 1e-8 and volume_error < 1e-8
    print(f"\n{'✓ TEST PASSED' if success else '✗ TEST FAILED'}")

    return success, flat_vec, shapes_info
