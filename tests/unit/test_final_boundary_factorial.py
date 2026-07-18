import inspect

import pytest

from recovar.em.dense_single_volume.iteration_loop import (
    _run_relion_iteration_loop,
    refine_single_volume,
)
from scripts.run_full_refinement import _select_final_replay_override


def _source_override():
    return {
        "previous_best_translations": "translations",
        "previous_best_rotations": "rotations",
        "previous_best_rotation_eulers": "eulers",
        "translation_sigma_angstrom": "sigma",
        "translation_sigma_angstrom_per_half": "sigma_halves",
        "noise_variance": "noise",
        "direction_prior": "prior",
        "image_corrections": "image_corr",
        "serialized_scale_corrections": "scale_corr",
        "class_tau2": "not_selected_for_k1_final_factorial",
    }


@pytest.mark.parametrize(
    ("groups", "expected_keys"),
    [
        (
            "poses",
            {
                "previous_best_translations",
                "previous_best_rotations",
                "previous_best_rotation_eulers",
            },
        ),
        (
            "sampling",
            {"translation_sigma_angstrom", "translation_sigma_angstrom_per_half"},
        ),
        (
            "corrections",
            {
                "noise_variance",
                "direction_prior",
                "image_corrections",
                "serialized_scale_corrections",
            },
        ),
    ],
)
def test_final_replay_group_selection_is_disjoint(groups, expected_keys):
    selected_groups, override = _select_final_replay_override(_source_override(), groups)
    assert selected_groups == {groups}
    assert set(override) == expected_keys


def test_final_replay_all_is_union_without_unrelated_fields():
    groups, override = _select_final_replay_override(_source_override(), "all")
    assert groups == {"poses", "sampling", "corrections"}
    assert "class_tau2" not in override
    assert set(override) == set(_source_override()) - {"class_tau2"}


@pytest.mark.parametrize("value", ["", "poses,unknown"])
def test_final_replay_group_selection_fails_closed(value):
    with pytest.raises(ValueError, match="final-replay-fields"):
        _select_final_replay_override(_source_override(), value)


def test_final_only_options_reach_both_iteration_loop_boundaries():
    for function in (refine_single_volume, _run_relion_iteration_loop):
        parameters = inspect.signature(function).parameters
        assert "final_replay_override" in parameters
        assert "final_sampling_replay_relion_dir" in parameters
