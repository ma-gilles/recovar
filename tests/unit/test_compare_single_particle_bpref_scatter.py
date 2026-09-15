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
    np.testing.assert_array_equal(_fftw_indices(contribution, ori_size=8), indices)


def test_legacy_pass2_dump_still_reduces_and_converts_centered_indices():
    legacy = {
        "reconstruction_probs": np.asarray(
            [[0.75, 0.25], [0.0, 0.0]], dtype=np.float64
        ),
        "shifted_recon": np.asarray(
            [[1.0 + 0.0j], [3.0 + 0.0j]], dtype=np.complex64
        ),
        "ctf2_over_nv_recon": np.asarray([2.0], dtype=np.float64),
        "rotations": np.broadcast_to(
            np.eye(3, dtype=np.float32), (2, 3, 3)
        ).copy(),
        "recon_window_indices": np.asarray([0], dtype=np.int32),
    }

    summed, ctf, rotations = _active_mstep_rows(legacy)

    np.testing.assert_allclose(summed, [[1.5 + 0.0j]])
    np.testing.assert_allclose(ctf, [[2.0]])
    np.testing.assert_array_equal(rotations, np.eye(3, dtype=np.float64)[None])
    assert _fftw_indices(legacy, ori_size=8).shape == (1,)
