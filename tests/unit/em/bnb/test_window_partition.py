"""Test 1: window partition for cryoSPARC BnB low/high split.

Verifies the disjoint, complete coverage property:
    low_indices ∪ high_indices == final_score_indices
    low_indices ∩ high_indices == ∅

This is purely a NumPy-level check against
``helpers.fourier_window.make_fourier_window_indices_np`` and
``bnb.frequency.make_bnb_high_indices_np``.
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.dense_single_volume.bnb.frequency import (
    make_bnb_frequency_schedule,
    make_bnb_high_indices_np,
)
from recovar.em.dense_single_volume.bnb.options import BranchBoundOptions
from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
)


# (image_shape, final_current_size, L)
PARTITION_CASES = [
    ((32, 32), 32, 4),
    ((32, 32), 32, 8),
    ((32, 32), 32, 12),
    ((32, 32), 32, 16),
    ((64, 64), 64, 12),
    ((64, 64), 64, 24),
    ((64, 64), 64, 32),
    ((64, 64), 48, 12),
    ((64, 64), 48, 24),
    ((128, 128), 64, 12),
    ((128, 128), 64, 24),
    ((128, 128), 128, 12),
    ((128, 128), 128, 24),
    ((128, 128), 128, 48),
    ((128, 128), 128, 64),
]


@pytest.mark.parametrize(("image_shape", "current_size", "L"), PARTITION_CASES)
def test_low_high_disjoint_and_exhaustive(image_shape, current_size, L):
    """low ∪ high == final  and  low ∩ high == ∅."""
    final_idx, _ = make_fourier_window_indices_np(
        image_shape, current_size, square=False, include_dc=False,
    )
    low_idx, _ = make_fourier_window_indices_np(
        image_shape, 2 * L, square=False, include_dc=False,
    )

    # The low window is bounded above by the final window: every low pixel is
    # a final pixel.
    low_subset = np.intersect1d(low_idx, final_idx, assume_unique=False)
    assert np.array_equal(np.sort(low_subset), np.sort(low_idx)), (
        f"low pixels not a subset of final for image_shape={image_shape}, "
        f"current_size={current_size}, L={L}"
    )

    high_idx = make_bnb_high_indices_np(final_idx, low_idx)

    # Disjoint.
    assert np.intersect1d(low_idx, high_idx, assume_unique=False).size == 0

    # Exhaustive.
    union = np.union1d(low_idx, high_idx)
    assert np.array_equal(np.sort(union), np.sort(final_idx))


def test_high_band_is_bounded_by_final_resolution():
    """High band must not extend beyond current refinement resolution."""
    image_shape = (128, 128)
    current_size = 64
    L = 12
    final_idx, _ = make_fourier_window_indices_np(
        image_shape, current_size, square=False, include_dc=False,
    )
    low_idx, _ = make_fourier_window_indices_np(
        image_shape, 2 * L, square=False, include_dc=False,
    )
    high_idx = make_bnb_high_indices_np(final_idx, low_idx)

    # Every high pixel must be in the final score support, NOT in the full
    # half-spectrum complement of the low band.
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    naive_complement = np.setdiff1d(np.arange(n_half), low_idx)
    assert high_idx.size < naive_complement.size, (
        "high band escaped the final resolution shell"
    )

    final_set = set(int(x) for x in final_idx)
    for idx in high_idx:
        assert int(idx) in final_set, f"high pixel {idx} not in final support"


def test_frequency_schedule_paper_default():
    """Default schedule [12, 24, 48, ..., L_max] for paper-faithful settings."""
    opts = BranchBoundOptions()  # 7 subdivisions, L0=12, growth=2.0
    schedule = make_bnb_frequency_schedule(
        final_current_size=128,
        image_shape=(128, 128),
        options=opts,
    )
    # L_max = 64, so schedule is [12, 24, 48, 64]: doubled until > 64, capped.
    assert schedule == [12, 24, 48, 64]


def test_frequency_schedule_lmax_clamping():
    """L schedule clamps at L_max and ends exactly there."""
    opts = BranchBoundOptions(n_subdivisions=7, initial_fourier_radius=12)

    # current_size=32 -> L_max=16, expect [12, 16]
    s32 = make_bnb_frequency_schedule(32, (32, 32), opts)
    assert s32 == [12, 16]

    # current_size=24 -> L_max=12, expect [12] (initial radius already at cap)
    s24 = make_bnb_frequency_schedule(24, (24, 24), opts)
    assert s24 == [12]

    # current_size=None -> L_max=image_shape[0]//2
    sN = make_bnb_frequency_schedule(None, (256, 256), opts)
    assert sN[-1] == 128
    # Doublings: 12, 24, 48, 96, 128 (stop after exceeding)
    assert sN == [12, 24, 48, 96, 128]


def test_frequency_schedule_monotone_and_bounded():
    """Schedule is strictly increasing and respects n_subdivisions+1 bound."""
    opts = BranchBoundOptions(n_subdivisions=7, initial_fourier_radius=12, fourier_radius_growth=2.0)
    schedule = make_bnb_frequency_schedule(512, (512, 512), opts)
    # Strictly increasing.
    assert all(schedule[i] < schedule[i + 1] for i in range(len(schedule) - 1))
    # Length bounded by n_subdivisions + 1 = 8.
    assert len(schedule) <= 8


def test_frequency_schedule_bad_growth():
    """Growth <= 1.0 should raise."""
    with pytest.raises(ValueError, match="fourier_radius_growth"):
        make_bnb_frequency_schedule(
            128, (128, 128),
            options=BranchBoundOptions(fourier_radius_growth=1.0),
        )


def test_frequency_schedule_bad_lmax():
    """L_max < 1 should raise."""
    with pytest.raises(ValueError, match="L_max"):
        make_bnb_frequency_schedule(0, (8, 8), options=BranchBoundOptions())
