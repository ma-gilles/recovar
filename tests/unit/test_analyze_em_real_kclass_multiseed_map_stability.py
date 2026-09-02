import numpy as np
import pytest

from scripts import analyze_em_real_kclass_multiseed_map_stability as analyzer


def _row(values):
    return {"matched_per_class_fsc_auc": values}


def test_summarize_separation_requires_every_class_to_clear_seed_variability() -> None:
    same = [
        _row([0.95, 0.96, 0.85, 0.94]),
        _row([0.94, 0.97, 0.84, 0.95]),
        _row([0.96, 0.95, 0.87, 0.93]),
    ]
    within = [
        _row([0.82, 0.88, 0.64, 0.81]),
        _row([0.80, 0.84, 0.54, 0.83]),
        _row([0.81, 0.86, 0.57, 0.82]),
    ]

    summary = analyzer.summarize_separation(same, within)

    assert summary["separated_for_every_class"] is True
    assert [row["class"] for row in summary["per_class"]] == [1, 2, 3, 4]
    assert summary["per_class"][2]["same_seed_cross_engine"]["min"] == pytest.approx(0.84)
    assert summary["per_class"][2]["within_engine_cross_seed"]["max"] == pytest.approx(0.64)


def test_summarize_separation_fails_if_one_class_overlaps() -> None:
    same = [_row([0.95, 0.96, 0.63, 0.94])] * 3
    within = [_row([0.82, 0.88, 0.64, 0.81])] * 3

    summary = analyzer.summarize_separation(same, within)

    assert summary["separated_for_every_class"] is False
    assert summary["per_class"][2]["same_seed_min_exceeds_within_engine_cross_seed_max"] is False


def test_canonical_order_rejects_duplicate_class_mapping() -> None:
    audit = {
        "class_matching": {
            "source_class_for_anchor": {"relion_half1": [1, 2, 2, 4]}
        }
    }

    with pytest.raises(ValueError, match="not a K-class permutation"):
        analyzer._canonical_order(audit, "relion_half1")


def test_identity_alignment_preserves_f32_values() -> None:
    maps = [np.arange(27, dtype=np.float32).reshape(3, 3, 3)]

    aligned = analyzer._alignment_transform(
        maps,
        {"identity_anchor": True},
        interpolation_order=1,
    )

    np.testing.assert_array_equal(aligned[0], maps[0])
    assert aligned[0].dtype == np.float32
