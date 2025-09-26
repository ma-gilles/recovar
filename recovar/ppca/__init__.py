# Main PPCA functions
from .ppca import EM, EM_step, E_M_step_batch

# Basis classes for sparse PCA
from .sparse_PCA import (
    Basis, 
    Wavelet, 
    Wavelet_multilvl, 
    Spatial, 
    Identity
)

# Utility functions for sparse PCA
from .sparse_PCA import (
    get_sparse_PCA_in_basis,
    measure_orthogonality,
    wavelet_dict_to_wavelet_vec,
    wavelet_vec_to_wavelet_dict,
    get_ft_U,
    jax_ft,
    jax_ift
)

# Batch processing utilities
from .ppca import batch_vec, batch_unvec, check_imaginary_part

__all__ = [
    # Main PPCA functions
    'EM',
    'EM_step', 
    'E_M_step_batch',
    
    # Basis classes
    'Basis',
    'Wavelet',
    'Wavelet_multilvl', 
    'Spatial',
    'Identity',
    
    # Sparse PCA utilities
    'get_sparse_PCA_in_basis',
    'measure_orthogonality',
    'wavelet_dict_to_wavelet_vec',
    'wavelet_vec_to_wavelet_dict',
    'get_ft_U',
    'jax_ft',
    'jax_ift',
    
    # Batch processing utilities
    'batch_vec',
    'batch_unvec', 
    'check_imaginary_part'
]