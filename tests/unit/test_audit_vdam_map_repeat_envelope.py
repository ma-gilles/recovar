from __future__ import annotations

import numpy as np

from scripts.audit_vdam_map_repeat_envelope import compare_repeat_envelope


def test_candidate_equal_to_native_repeat_is_inside_fsc_diameter():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(16, 16, 16)).astype(np.float32)
    second = base + 1.0e-3 * rng.normal(size=base.shape).astype(np.float32)
    shellwise: dict[str, np.ndarray] = {}

    report = compare_repeat_envelope(
        base.copy(),
        [base, second],
        iteration=20,
        shellwise=shellwise,
    )

    assert report["inside_native_repeat_fsc_diameter"] is True
    assert report["candidate_to_native_diameter_ratio"] == 0.0
    assert len(shellwise) == 3


def test_unrelated_candidate_is_outside_tight_native_repeat_envelope():
    rng = np.random.default_rng(11)
    base = rng.normal(size=(16, 16, 16)).astype(np.float32)
    second = base + 1.0e-4 * rng.normal(size=base.shape).astype(np.float32)
    candidate = rng.normal(size=base.shape).astype(np.float32)

    report = compare_repeat_envelope(
        candidate,
        [base, second],
        iteration=40,
        shellwise={},
    )

    assert report["inside_native_repeat_fsc_diameter"] is False
    assert report["candidate_to_native_diameter_ratio"] > 1.0
