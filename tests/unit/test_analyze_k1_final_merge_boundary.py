from __future__ import annotations

import numpy as np

from scripts.analyze_k1_final_merge_boundary import build_report_from_arrays


def test_merge_boundary_localizes_explicit_only_failure() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(size=(16, 16, 16))
    bad_merged = base + 2.0 * rng.normal(size=base.shape)

    report, shellwise = build_report_from_arrays(
        recovar_half1=base,
        recovar_half2=base,
        recovar_merged=bad_merged,
        relion_half1=base,
        relion_half2=base,
        relion_merged=base,
    )

    assert report["half_average_cross_engine_passes"]
    assert not report["explicit_merged_cross_engine_passes"]
    assert report["classification"] == (
        "failure_appears_only_at_explicit_merged_product_comparison"
    )
    assert set(shellwise) == set(report["comparisons"])


def test_merge_boundary_reports_no_failure_for_identical_maps() -> None:
    rng = np.random.default_rng(5)
    base = rng.normal(size=(12, 12, 12))

    report, _ = build_report_from_arrays(
        recovar_half1=base,
        recovar_half2=base,
        recovar_merged=base,
        relion_half1=base,
        relion_half2=base,
        relion_merged=base,
    )

    assert report["classification"] == "no_fixed_merged_fsc_gate_failure"
    assert report["half_average_cross_engine_passes"]
    assert report["explicit_merged_cross_engine_passes"]
