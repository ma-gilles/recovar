import pywt
from recovar.fourier_transform_utils import fourier_transform_utils
import numpy as np
import functools
import jax
use_np = True
use_np_ft = use_np
if use_np:
    jnp = np
else:
    import jax.numpy as jnp
    
import jaxwt 
ftu_np = fourier_transform_utils(np)
ftu_jax = fourier_transform_utils(jnp)

use_jaxwt = True

def jax_ift(u, volume_shape):
    return ftu_jax.get_idft3(u.reshape([u.shape[0], *volume_shape])).reshape(u.shape )

def jax_ft(u, volume_shape):
    return ftu_jax.get_dft3(u.reshape([u.shape[0], *volume_shape])).reshape(u.shape )

debug_ft = False

def get_ft_U(u, volume_shape, inverse = True):
    if debug_ft:
        print("shape:", u.shape)
        u = u.astype(np.complex64)
        print(u.dtype)

    # vol_shape = 
    if inverse:
        def ift(u):
            return jax_ift(u, volume_shape)

        if use_np_ft:
            u_ft = ftu_np.get_idft3(u.reshape([u.shape[0], *volume_shape])).reshape([u.shape[0], np.prod(volume_shape)])
        else:
            u_ft = np.array(jax.jit(ift)(u))
            
        if debug_ft:
            u_ft4 = np.array(jax.jit(ift)(u))

            u_ftnp = ftu_np.get_idft3(u.reshape([u.shape[0], *volume_shape])).reshape([u.shape[0], -1])
            u_ft2 = np.array(jax_ift(jnp.array(u),volume_shape))
            u_ft3 = np.array(jax_ift(jnp.array(u.copy()),volume_shape))

            print("reverse", np.linalg.norm(u_ftnp - u_ft2) / np.linalg.norm(u_ftnp))
            print("reverse", np.linalg.norm(u_ft2 - u_ft3) / np.linalg.norm(u_ftnp))
            print("reverse", np.linalg.norm(u_ftnp - u_ft4) / np.linalg.norm(u_ftnp))
            print("reverse", np.linalg.norm(u_ft - u_ftnp) / np.linalg.norm(u_ftnp))

            # if np.linalg.norm(u_ft - u_ft2) / np.linalg.norm(u_ft) > 0.1:
            #     import pdb; pdb.set_trace()

        
    else:
        def ft(u):
            return jax_ft(u, volume_shape)

        
        if use_np_ft:
            u_ft = ftu_np.get_dft3(u.reshape([u.shape[0], *volume_shape])).reshape([u.shape[0], np.prod(volume_shape)])
        else:
            u_ft = np.array(jax.jit(ft)(jnp.array(u)))
        
        if debug_ft:
            u_ft = ftu_np.get_dft3(u.reshape([u.shape[0], *volume_shape])).reshape([u.shape[0], -1])
            u_ft2 = np.array(jax_ft(jnp.array(u),volume_shape))
            u_ft3 = np.array(jax_ft(jnp.array(u.copy()),volume_shape))

            print("forward", np.linalg.norm(u_ft - u_ft2) / np.linalg.norm(u_ft))
            print("forward", np.linalg.norm(u_ft3 - u_ft2) / np.linalg.norm(u_ft3))
            print("forward", np.linalg.norm(u_ft - u_ft3) / np.linalg.norm(u_ft2))

            
    return u_ft


def get_sparse_PCA_in_basis(u, sigma, to_basis_fun, from_basis_fun, percentile_used):
    volume_size = u.shape[0]
    
    # Everything here should be numpy
    
    
    u = np.array(u.T).copy()
    sigma = np.array(sigma).copy()

    # print(u.shape)
    # print(np.linalg.norm(u ))
    # print(np.linalg.norm(sigma))

    basis_u = to_basis_fun(u)
    basis_variance = estimate_variance(basis_u, sigma)
    # import pdb; pdb.set_trace()
    del u
    # print(np.linalg.norm(basis_u))
    # print(np.linalg.norm(basis_variance))

    
    # Wavelet threshold
    var_treshold = np.percentile(basis_variance, percentile_used)
    valid_variance_indices = np.abs(basis_variance) > var_treshold

    thresh_wavelet_vars = basis_variance.copy()
    thresh_wavelet_vars[~valid_variance_indices] = 0 
    
    # thresh_wavelet_vars = thresh_wavelet_vars.at[~valid_variance_indices].set(0)
    basis_var_img = from_basis_fun(basis_variance[None,...])[0]#...,0]
    basis_var_img_thresholded = from_basis_fun(thresh_wavelet_vars[None,...])[0]#,...]

    # Thresholded wavelet images
    # BU:
    sparse_basis_u = basis_u[...,valid_variance_indices]

    # EBU * Sigma^{1/2}
    sparse_basis_u_sigma = sparse_basis_u * np.sqrt(sigma[...,None])

    # Do SVD in JAX?
    # EBU * Sigma^{1/2} = V G W^T => 
    sparse_V, sparse_sigma, _ = jnp.linalg.svd(sparse_basis_u_sigma.T, full_matrices = False)
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
    sparsified_s = sparse_sigma **2 
    return np.array(sparsified_u).T, np.array(sparsified_s), np.array(basis_var_img), np.array(basis_var_img_thresholded)


def measure_orthogonality(KK):
    XX = jnp.linalg.norm(jnp.conj(KK).T @ KK - jnp.eye(KK.shape[-1]))
    return XX

## Wavelet transforms
def wavelet_dict_to_wavelet_vec(wavelet_dict, keys_order):
    wavelet_vec = [ wavelet_dict[key].reshape([wavelet_dict[key].shape[0], -1]) for key in keys_order ]
    wavelet_vec = np.concatenate(wavelet_vec, axis = -1)
    return wavelet_vec

def wavelet_vec_to_wavelet_dict(wavelet_vec, keys_order, wavelet_dict_shape):
    current_idx = 0 
    wavelet_dict = {}
    
    for key in keys_order:
        size_of_block = np.prod(wavelet_dict_shape[key])
        wavelet_dict[key] = wavelet_vec[:,current_idx:current_idx+size_of_block].reshape([-1] + list( wavelet_dict_shape[key]))

#         if use_np:
#             wavelet_dict[key] = wavelet_vec[:,current_idx:current_idx+size_of_block].reshape([-1] + list( wavelet_dict_shape[key]))
#         else:
#             wavelet_dict[key] = wavelet_vec.at[:,current_idx:current_idx+size_of_block].get().reshape([-1] + list( wavelet_dict_shape[key]))

        current_idx += size_of_block
    return  wavelet_dict   


class Basis():
    def to_basis(self, image):
        pass
    def to_image(self, basis_coeffs):
        pass


class Wavelet(Basis):
    def __init__(self, volume_shape, wavelet_type, wavelet_mode = 'periodization'):
        wavelet_dict_tmp  = pywt.dwtn(np.zeros(volume_shape), wavelet = wavelet_type, mode = wavelet_mode ) 

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

        wavelet_dict = wavelet_vec_to_wavelet_dict(wavelet_vec, self.keys_order, self.wavelet_dict_shape )
        real_image = pywt.idwtn(wavelet_dict,
                                wavelet = self.wavelet_type,
                                mode = self.wavelet_mode, axes = (-3, -2, -1)).reshape([wavelet_vec.shape[0], -1])
        return real_image

    def image_ft_to_basis_single(self, real_image):
        wavelet_dict = pywt.dwtn(real_image.reshape([-1] + list(self.volume_shape)),
                                 wavelet = self.wavelet_type,
                                 mode = self.wavelet_mode, axes = (-3, -2, -1))
        wavelet_vec = wavelet_dict_to_wavelet_vec(wavelet_dict, self.keys_order )
        return wavelet_vec

    def to_basis(self, images):
        cols_real =  get_ft_U(images, self.volume_shape, inverse = True) * np.sqrt(self.volume_size)
        if np.linalg.norm(cols_real.imag) > 1e-6 * np.linalg.norm(cols_real.real):
            print("Imaginary part is non-zero! ratio of imaginary to real:", np.linalg.norm(cols_real.imag) / np.linalg.norm(cols_real.real))
        cols_all_wavelet = self.image_ft_to_basis_single(cols_real.real)
        return cols_all_wavelet

    def to_image(self, basis_coeffs):
        BT_ET_EBFU = self.basis_to_image_ft_single(basis_coeffs)
        FT_BT_ET_EBFU = get_ft_U(BT_ET_EBFU, self.volume_shape, inverse = False) / np.sqrt(self.volume_size)
        return FT_BT_ET_EBFU

    def name(self):
        return self.base_name + self.wavelet_type

class Wavelet_multilvl(Basis):
    def __init__(self, volume_shape, wavelet_type, wavelet_mode = 'periodization', mask=None):

        if use_jaxwt:
            wavelet_decn_dict  = jaxwt.wavedec3(np.zeros(volume_shape), wavelet = wavelet_type, mode = wavelet_mode, axes = (-3,-2,-1))
        else:
            wavelet_decn_dict  = pywt.wavedecn(np.zeros(volume_shape), wavelet = wavelet_type, mode = wavelet_mode ) 
        arr, coeff_slices = coeffs_to_array(wavelet_decn_dict, axes = (-3,-2,-1))
        wavelet_decn_arr_shape = arr.shape
        self.volume_shape = volume_shape
        self.wavelet_type = wavelet_type
        self.wavelet_mode = wavelet_mode
        self.coeff_slices = coeff_slices
        self.wavelet_decn_arr_shape = wavelet_decn_arr_shape
        self.volume_size = np.prod(volume_shape)
        self.coeffs_slices_batch = None
        self.base_name = "wavelet_multilvl"
        self.from_ft = True
        self.mask = mask
        
    def image_ft_to_basis_single(self, real_image):
        if use_jaxwt:
            wavelet_dict  = jaxwt.wavedec3(real_image.reshape([-1] + list(self.volume_shape)), wavelet = self.wavelet_type, mode = self.wavelet_mode, axes = (-3,-2,-1))
        else:            
            wavelet_dict  = pywt.wavedecn(real_image.reshape([-1] + list(self.volume_shape)), wavelet = self.wavelet_type, mode = self.wavelet_mode , axes = (-3, -2, -1))
        arr, sizes = coeffs_to_array(wavelet_dict, axes = (-3,-2,-1))
        if real_image.shape[0] > 0:
            self.coeffs_slices_batch = sizes
        return arr.reshape([real_image.shape[0], -1])

    def to_basis(self, images):
        if self.from_ft:
            cols_real =  (get_ft_U(images, self.volume_shape, inverse = True) * np.sqrt(self.volume_size))#.real
        else:
            cols_real = images

        if self.mask is not None:
            cols_real *= self.mask[None]

        cols_all_wavelet = self.image_ft_to_basis_single(cols_real)
        return cols_all_wavelet

    def to_image_ft_single(self, wavelet_vec):
        
        # # Doesn't seem to work for a batch of 1...?
        # wavelet_dict_2 = pywt.array_to_coeffs(wavelet_vec.reshape([-1] + list(self.wavelet_decn_arr_shape)), self.coeff_slices)
        # image  = pywt.waverecn(wavelet_dict_2, wavelet = self.wavelet_type, mode = self.wavelet_mode, axes = (-3,-2,-1) )

        if wavelet_vec.shape[0] > 1:
            wavelet_dict_2 = pywt.array_to_coeffs(wavelet_vec.reshape([-1] + list(self.wavelet_decn_arr_shape)), self.coeffs_slices_batch)
            if use_jaxwt:
                image  = pywt.waverec3(wavelet_dict_2, wavelet = self.wavelet_type, mode = self.wavelet_mode, axes = (-3,-2,-1) )
            else:
                image  = pywt.waverecn(wavelet_dict_2, wavelet = self.wavelet_type, mode = self.wavelet_mode, axes = (-3,-2,-1) )
        else:
            wavelet_dict_2 = pywt.array_to_coeffs(wavelet_vec.reshape( list(self.wavelet_decn_arr_shape)), self.coeff_slices)
            if use_jaxwt:
                image  = pywt.waverec3(wavelet_dict_2, wavelet = self.wavelet_type, mode = self.wavelet_mode, axes = (-3,-2,-1) )
            else:
                image  = pywt.waverecn(wavelet_dict_2, wavelet = self.wavelet_type, mode = self.wavelet_mode, axes = (-3,-2,-1) )

        return image.reshape(-1, np.prod(self.volume_shape))

    def to_image(self, basis_coeffs):
        # Thresholded wavelet to real 
        BT_ET_EBFU = self.to_image_ft_single(basis_coeffs)

        if self.from_ft:
            FT_BT_ET_EBFU = get_ft_U(BT_ET_EBFU, self.volume_shape, inverse = False) / np.sqrt(self.volume_size)
        else:
            FT_BT_ET_EBFU = BT_ET_EBFU

        return FT_BT_ET_EBFU

    def name(self):
        return self.base_name + self.wavelet_type


class Spatial(Basis):
    def __init__(self, volume_shape, mask = None):
        self.volume_shape = volume_shape
        self.volume_size = np.prod(volume_shape)
        self.base_name = "spatial"
        self.mask = np.ones(self.volume_size) if mask is None else mask
        
    def to_basis(self, images):
        cols_real =  get_ft_U(images, self.volume_shape, inverse = True) * np.sqrt(self.volume_size) * self.mask[None]
        return cols_real

    def to_image(self, basis_coeffs):
        FT_BT_ET_EBFU = get_ft_U(basis_coeffs, self.volume_shape,  inverse = False) / np.sqrt(self.volume_size)
        return FT_BT_ET_EBFU

    def name(self):
        return self.base_name



class Identity(Basis):
    def __init__(self, volume_shape, mask = None):
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



def coeffs_to_array(coeff_dict, axes = (-3,-2,-1)):

    if use_jaxwt:
        for i in range(len(coeff_dict)):
            if isinstance(coeff_dict[i], dict):
                # Handle dictionary case
                for k in coeff_dict[i].keys():
                    coeff_dict[i][k] = np.array(coeff_dict[i][k])
            else:
                # Handle non-dictionary case (e.g. approximation coefficients)
                coeff_dict[i] = np.array(coeff_dict[i])

    coeffs_array, _ = pywt.coeffs_to_array(coeff_dict, axes = axes)
    return coeffs_array


def wavelet_avg_square_by_level_both(volume, wavelet_type='db1'):
    """
    Compute average square of wavelet coefficients by level for a single volume.
    Returns both:
    1. Array where each entry has the average square value for its level
    2. Regular dictionary with level information
    """

    if use_jaxwt:
        coeff_dict = jaxwt.wavedec3(volume[None], wavelet=wavelet_type, mode='symmetric', axes = (-3,-2,-1))
    else:
        coeff_dict = pywt.wavedecn(volume, wavelet=wavelet_type, mode='periodization')
    # coeff_dict2 = pywt.wavedecn(volume, wavelet=wavelet_type, mode='periodization')

    # for key in coeff_dict:
    #     if key in coeff_dict2:
    #         if np.linalg.norm(coeff_dict[key] - coeff_dict2[key]) > 1e-6:
    #             print("BAD")
    # import ipdb; ipdb.set_trace()
    # Get all coefficients as a flat array
    coeffs_array, _ = coeffs_to_array(coeff_dict, axes = (-3,-2,-1))
    total_coeffs = coeffs_array.size
    
    # Initialize result array
    avg_square_array = np.zeros(total_coeffs)
    current_idx = 0
    
    # Initialize level results dictionary
    level_results = {}
    
    for level_idx, coeff_data in enumerate(coeff_dict):
        # Determine level name and type
        if level_idx == 0:
            level_key = f"a{len(coeff_dict)-1}"  # Approximation level
            level_type = "Approximation"
        else:
            level_key = f"d{len(coeff_dict)-level_idx}"  # Detail level
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
        avg_square_array[current_idx:current_idx + n_coefficients] = avg_square
        current_idx += n_coefficients
        
        # Store in level results dictionary
        level_results[level_key] = {
            'avg_square': avg_square,
            'total_energy': total_energy,
            'n_coefficients': n_coefficients,
            'level_type': level_type,
            'coeffs': coeffs
        }
        
        print(f"{level_key} ({level_type}): {n_coefficients} coeffs, avg square: {avg_square:.6f}")
    
    return avg_square_array, level_results