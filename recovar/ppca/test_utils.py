"""Utility helpers shared by PPCA experiments."""

import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.ppca.sparse_PCA import wavelet_avg_square_by_level_both


def compute_level_sigma(W_init, volume_shape, scale, mode="avg_all", wavelet_type="db1"):
    """Estimate per-wavelet-coefficient L1 scales from the PCA warm start."""
    volume_size = int(np.prod(volume_shape))
    n_basis = int(W_init.shape[1])
    volumes = ftu.get_idft3(np.asarray(W_init).T.reshape(n_basis, *volume_shape)).real
    volumes *= np.sqrt(volume_size)

    per_basis = []
    for volume in volumes:
        avg_square, _ = wavelet_avg_square_by_level_both(volume, wavelet_type=wavelet_type)
        per_basis.append(np.asarray(avg_square, dtype=np.float64))
    per_basis = np.asarray(per_basis, dtype=np.float64)

    if mode == "avg_all":
        avg_square = np.mean(per_basis, axis=0)
    elif mode == "per_basis":
        avg_square = per_basis
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    return scale * np.sqrt(np.maximum(avg_square, 1e-12) / 2.0)
