"""Tests for the single-particle BPref scatter boundary diagnostic."""

import numpy as np

from scripts.compare_single_particle_bpref_scatter import (
    _active_mstep_rows,
    _fftw_indices,
)


def test_contribution_rows_are_already_active_and_fftw_indexed():
    summed = np.asarray([[1.0 + 2.0j, 3.0 + 4.0j]], dtype=np.complex128)
    ctf = np.asarray([[5.0, 6.0]], dtype=np.float64)
    rotations = np.eye(3, dtype=np.float32)[None]
    indices = np.asarray([7, 11], dtype=np.int32)
    contribution = {
        "active_summed": summed,
        "active_ctf_probs": ctf,
        "active_rotations": rotations,
        "window_indices": indices,
    }

    actual_summed, actual_ctf, actual_rotations = _active_mstep_rows(contribution)

    np.testing.assert_array_equal(actual_summed, summed)
    np.testing.assert_array_equal(actual_ctf, ctf)
    np.testing.assert_array_equal(actual_rotations, rotations.astype(np.float64))
    np.testing.assert_array_equal(_fftw_indices(contribution), indices)
