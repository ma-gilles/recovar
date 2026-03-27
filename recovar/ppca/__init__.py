"""Public PPCA compatibility surface."""

from .ppca import (
    EM,
    E_M_step_batch,
    E_M_step_batch_half,
    EM_step,
    _tri_size,
    batch_unvec,
    batch_vec,
    check_imaginary_part,
    unpack_tri_to_full,
)
from .sparse_PCA import (
    Basis,
    Identity,
    Spatial,
    Wavelet,
    Wavelet_multilvl,
    get_ft_U,
    get_sparse_PCA_in_basis,
    jax_ft,
    jax_ift,
    measure_orthogonality,
    wavelet_dict_to_wavelet_vec,
    wavelet_vec_to_wavelet_dict,
)

__all__ = [
    # Main PPCA functions
    "EM",
    "EM_step",
    "E_M_step_batch",
    "E_M_step_batch_half",
    "unpack_tri_to_full",
    "_tri_size",
    # Basis classes
    "Basis",
    "Wavelet",
    "Wavelet_multilvl",
    "Spatial",
    "Identity",
    # Sparse PCA utilities
    "get_sparse_PCA_in_basis",
    "measure_orthogonality",
    "wavelet_dict_to_wavelet_vec",
    "wavelet_vec_to_wavelet_dict",
    "get_ft_U",
    "jax_ft",
    "jax_ift",
    # Batch processing utilities
    "batch_vec",
    "batch_unvec",
    "check_imaginary_part",
]
