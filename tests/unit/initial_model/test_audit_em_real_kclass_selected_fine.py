from __future__ import annotations

from scripts.audit_em_real_kclass_selected_fine import summarize_particle


def _class_record(
    class_index: int,
    *,
    relion_mass: float,
    recovar_mass: float,
    relion_only: int,
    recovar_only: int,
    raw_max: float,
    translation_prior_max: float,
) -> dict[str, object]:
    return {
        "class_index_zero_based": class_index,
        "relion_probability_mass": relion_mass,
        "recovar_probability_mass": recovar_mass,
        "relion_only_count": relion_only,
        "recovar_only_count": recovar_only,
        "common_score_pre_prior_centered_diff": {"max_abs": raw_max},
        "common_rotation_log_prior_centered_diff": {"max_abs": 0.0},
        "common_translation_log_prior_centered_diff": {"max_abs": translation_prior_max},
    }


def test_selected_fine_summary_localizes_class_flip_after_coarse_support_divergence() -> None:
    selection = {
        "recovar_source_row_zero_based": 132,
        "stack_index_one_based": 1839,
    }
    coarse_record = {
        "classification": "coarse_joint_winner_exact",
        "comparison": {
            "joint_winner_exact": True,
            "significance_support_mismatch_count": 70,
        },
    }
    records = [
        _class_record(
            0,
            relion_mass=0.0,
            recovar_mass=0.0,
            relion_only=0,
            recovar_only=0,
            raw_max=0.0,
            translation_prior_max=0.0,
        ),
        _class_record(
            1,
            relion_mass=0.9855,
            recovar_mass=0.0100,
            relion_only=0,
            recovar_only=832,
            raw_max=0.0146,
            translation_prior_max=2.79,
        ),
        _class_record(
            2,
            relion_mass=0.0145,
            recovar_mass=0.9787,
            relion_only=0,
            recovar_only=1376,
            raw_max=0.0141,
            translation_prior_max=2.49,
        ),
        _class_record(
            3,
            relion_mass=0.0,
            recovar_mass=0.0,
            relion_only=0,
            recovar_only=0,
            raw_max=0.0,
            translation_prior_max=0.0,
        ),
    ]

    result = summarize_particle(
        selection=selection,
        coarse_record=coarse_record,
        class_records=records,
    )

    assert result["coarse_joint_winner_exact"] is True
    assert result["first_observed_nonidentical_boundary"] == "coarse_significance_support"
    assert result["fine_relion_class_winner_zero_based"] == 1
    assert result["fine_recovar_class_winner_zero_based"] == 2
    assert result["fine_class_winner_exact"] is False
    assert result["fine_support_symmetric_difference_count"] == 2208
    assert result["common_centered_raw_score_max_abs"] == 0.0146
    assert result["common_centered_rotation_log_prior_max_abs"] == 0.0
    assert result["common_centered_translation_log_prior_max_abs"] == 2.79
    assert result["classification"] == "fine_class_winner_mismatch_with_support_and_prior_differences"
