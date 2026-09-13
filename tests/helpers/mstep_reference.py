"""Independent NumPy reference for RELION-order float32 M-step sums."""

import numpy as np


def numpy_relion_f32_mstep_sums(probs, shifted, ctf2_over_nv):
    probs = np.asarray(probs, dtype=np.float32)
    shifted = np.asarray(shifted, dtype=np.complex64)
    ctf2_over_nv = np.asarray(ctf2_over_nv, dtype=np.float32)
    numerator = np.zeros((probs.shape[0], probs.shape[1], shifted.shape[-1]), dtype=np.complex64)
    denominator = np.zeros(numerator.shape, dtype=np.float32)
    for trans_idx in range(probs.shape[-1]):
        weight = probs[:, :, trans_idx, None]
        numerator += weight * shifted[:, None, trans_idx, :]
        denominator += weight * ctf2_over_nv[:, None, :]
    return numerator, denominator

