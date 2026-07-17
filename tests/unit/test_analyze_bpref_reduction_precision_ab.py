import numpy as np

from scripts.analyze_bpref_reduction_precision_ab import _factor_comparisons


def test_factor_ab_requires_only_production_numerator_to_change(tmp_path):
    default = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64)
    highest = np.array([[1.001 + 2j, 3 + 4.001j]], dtype=np.complex64)
    common = {
        "numerator_highest_f32": highest,
        "numerator_sequential_f32": highest,
        "term_f32": np.ones((1, 2), dtype=np.complex64),
    }
    control = tmp_path / "control.npz"
    fixed = tmp_path / "fixed.npz"
    np.savez(control, numerator_f32=default, **common)
    np.savez(fixed, numerator_f32=highest, **common)

    result = _factor_comparisons(control, fixed)

    assert result["changed_arrays"] == ["numerator_f32"]
    assert result["fixed_matches_highest_exactly"] is True
