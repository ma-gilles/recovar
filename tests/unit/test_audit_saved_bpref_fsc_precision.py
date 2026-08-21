from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_saved_bpref_fsc_precision import (
    replay_shell_from_saved_aggregates,
)

pytestmark = pytest.mark.unit


def test_saved_aggregate_shell_replay_float64_is_order_stable():
    rng = np.random.default_rng(3)
    accumulator_size = 11
    shape = (accumulator_size,) * 3
    data0 = (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ).astype(np.complex64)
    data1 = (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ).astype(np.complex64)
    weight0 = rng.uniform(0.2, 2.0, size=shape).astype(np.float32)
    weight1 = rng.uniform(0.2, 2.0, size=shape).astype(np.float32)

    report = replay_shell_from_saved_aggregates(
        data0,
        data1,
        weight0,
        weight1,
        padding_factor=2,
        current_size=4,
        accumulator_size=accumulator_size,
        shell=1,
    )

    assert report["saved_voxel_contribution_count"] > 0
    assert report["native_target_count"] > 0
    assert len(report["modes"]) == 16
    assert report["mode_range"] < 1.0e-5
    assert report["mode_min"] <= report["mode_max"]
    assert report["modes"][
        "downsample_float64_canonical__shell_float64_canonical"
    ] == pytest.approx(
        report["modes"][
            "downsample_float64_canonical__shell_float64_reverse"
        ],
        abs=1.0e-14,
    )
