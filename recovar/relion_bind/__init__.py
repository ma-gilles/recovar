"""RELION pybind11 bindings for exact parity testing.

Provides Python wrappers around RELION's C++ subfunctions so that
recovar and RELION can be called on identical inputs and their outputs
diffed numerically.

Submodules
----------
conversions : Layout conversion functions (pure Python, no C++ dependency)
"""

from recovar.relion_bind.conversions import (
    compute_relion_pad_size,
    fftw_half_to_recovar_centered,
    fftw_half_to_relion_projector,
    recovar_centered_to_fftw_half,
    recovar_centered_to_relion_projector,
    recovar_real_to_relion_real,
    relion_projector_to_fftw_half,
    relion_projector_to_recovar_centered,
    relion_real_to_recovar_real,
)

__all__ = [
    "compute_relion_pad_size",
    "fftw_half_to_recovar_centered",
    "fftw_half_to_relion_projector",
    "recovar_centered_to_fftw_half",
    "recovar_centered_to_relion_projector",
    "recovar_real_to_relion_real",
    "relion_projector_to_fftw_half",
    "relion_projector_to_recovar_centered",
    "relion_real_to_recovar_real",
]
