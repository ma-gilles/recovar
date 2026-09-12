"""The VDAM M-step accumulator record shared by the single-class M-step and the multi-class driver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VdamAccumulator:
    """Per-class raw backprojection accumulator.

    The E-step adapter produces one `VdamAccumulator` per `(class, halfset)`
    pair (so `2K` total when `pseudo_halfsets` is active). `data` and
    `weight` have the padded Fourier shape `(N_pad, N_pad, N_pad // 2 + 1)`
    at `padding_factor=1`.
    """

    data: np.ndarray  # complex128, shape (Nz_pad, Ny_pad, Nx_pad_half)
    weight: np.ndarray  # float64, same shape
    class_idx: int
    halfset_idx: int
