from scripts.analyze_k1_verbose_preprocess_repeat import _summarize_repeats


def test_summarize_repeats_tracks_native_hits_and_stage_envelopes():
    def metric(relative_l2, max_abs, mismatches):
        return {
            "relative_l2_over_reference": relative_l2,
            "max_abs": max_abs,
            "value_mismatch_count": mismatches,
        }

    records = [
        {
            "background": 1.0,
            "normalized_shifted_real": metric(0.0, 0.0, 0),
            "masked_real": metric(1e-7, 2e-7, 3),
            "masked_fourier_pre_optics": metric(2e-7, 4e-7, 5),
        },
        {
            "background": 1.0000001192092896,
            "normalized_shifted_real": metric(0.0, 0.0, 0),
            "masked_real": metric(3e-7, 5e-7, 7),
            "masked_fourier_pre_optics": metric(4e-7, 6e-7, 9),
        },
    ]

    report = _summarize_repeats(records, 1.0)

    assert report["repeat_count"] == 2
    assert report["exact_native_background_hits"] == 1
    assert len(report["unique_background_bits"]) == 2
    assert report["stage_envelopes"]["masked_real"] == {
        "min_relative_l2": 1e-7,
        "max_relative_l2": 3e-7,
        "min_max_abs": 2e-7,
        "max_max_abs": 5e-7,
        "min_value_mismatch_count": 3,
        "max_value_mismatch_count": 7,
    }
