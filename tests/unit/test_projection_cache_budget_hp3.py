"""Fine-projection cache admission for the HEALPix-3 / current_size-92 regime.

The 10k EMPIAR-10097 convergence run (job 14031616) skipped the sparse pass-2
projection cache in every hp3 iteration ("estimated transient 18.36 GiB
exceeds cap 7.96 GiB") and recomputed fine projections per image chunk.
These checks pin the admission rule that lets an 80 GB device cache the
294912-rotation hp3 projections while a 40 GB device still declines.
"""

import numpy as np

from recovar.em.sparse_pass2.sparse_pass2_budget import (
    _projection_cache_fits_budget,
    _projection_cache_max_bytes_for_pass,
    _projection_cache_transient_bytes,
    _projection_call_max_bytes_for_pass,
)

_HP3_FINE_ROTATIONS = 294912
_CS92_HALF_PIXELS = 3386  # windowed half-spectrum score pixels at current_size 92, 256^2
_H100_BYTES = int(79.65 * 1024**3)
_A100_40_BYTES = 40 * 1024**3


def _clear_env(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", raising=False)


def test_hp3_cs92_cache_estimate_matches_logged_value():
    transient = _projection_cache_transient_bytes(
        _HP3_FINE_ROTATIONS, _CS92_HALF_PIXELS, projection_complex_dtype=np.complex64, include_abs2=True,
    )
    # score (complex64) + recon (complex64 shares the score cache when not
    # windowed) is not double counted here; the logged 18.36 GiB includes the
    # windowed score+recon+abs2 triple, so only check the single-cache term.
    assert transient == _HP3_FINE_ROTATIONS * _CS92_HALF_PIXELS * (8 + 4)


def test_hp3_cache_admitted_on_80gb_rejected_on_40gb(monkeypatch):
    _clear_env(monkeypatch)
    logged_hp3_estimate = int(18.36 * 1024**3)
    assert _projection_cache_fits_budget(logged_hp3_estimate, _projection_cache_max_bytes_for_pass(_H100_BYTES))
    assert not _projection_cache_fits_budget(logged_hp3_estimate, _projection_cache_max_bytes_for_pass(_A100_40_BYTES))


def test_cache_cap_is_quarter_of_device_memory_without_override(monkeypatch):
    _clear_env(monkeypatch)
    assert _projection_cache_max_bytes_for_pass(_H100_BYTES) == int(_H100_BYTES * 0.25)


def test_cache_cap_change_leaves_per_call_rotation_budget_alone(monkeypatch):
    _clear_env(monkeypatch)
    # The per-call projected-rotation budget (809 rotations per call at cs 92 on
    # an H100 in the measured runs) must not grow with the cache admission cap.
    assert _projection_call_max_bytes_for_pass(_H100_BYTES) == int(_H100_BYTES * 0.040)
    assert _projection_call_max_bytes_for_pass(_H100_BYTES) < _projection_cache_max_bytes_for_pass(_H100_BYTES)


def test_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", "654321")
    assert _projection_cache_max_bytes_for_pass(_H100_BYTES) == 654321


def test_cache_build_rotations_per_call_scales_scoring_budget(monkeypatch):
    from recovar.em.sparse_pass2.sparse_pass2_budget import _projection_cache_build_max_rotations_per_call

    _clear_env(monkeypatch)
    assert _projection_cache_build_max_rotations_per_call(809, 294912) == 4 * 809
    # never more rotations than the fine grid holds, never below one
    assert _projection_cache_build_max_rotations_per_call(809, 1000) == 1000
    assert _projection_cache_build_max_rotations_per_call(None, 294912) is None
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", "4096")
    assert _projection_cache_build_max_rotations_per_call(809, 294912) == 4096
