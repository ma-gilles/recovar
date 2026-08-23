from scripts.compare_relion_recovar_bpref_factors import _classify


def _summary(default, highest, sequential, f64):
    return {
        "relion_vs_recovar_default_f32": {"global": {"relative_l2_over_lhs": default}},
        "relion_vs_recovar_highest_f32": {"global": {"relative_l2_over_lhs": highest}},
        "relion_vs_recovar_sequential_f32": {"global": {"relative_l2_over_lhs": sequential}},
        "relion_vs_recovar_genuine_f64": {"global": {"relative_l2_over_lhs": f64}},
    }


def test_classifies_reduced_precision_only_when_controls_close():
    values = _summary(2.0e-4, 4.0e-7, 4.0e-7, 3.0e-7)

    assert _classify(values, factors_close=True, relion_terms_close=True) == "recovar_default_gemm_reduced_precision"
    assert _classify(values, factors_close=False, relion_terms_close=True) == "unresolved"
    assert _classify(values, factors_close=True, relion_terms_close=False) == "unresolved"


def test_does_not_classify_small_or_unclosed_default_difference():
    assert _classify(_summary(2.0e-6, 4.0e-7, 4.0e-7, 3.0e-7), True, True) == "unresolved"
    assert _classify(_summary(2.0e-4, 2.0e-5, 4.0e-7, 3.0e-7), True, True) == "unresolved"
