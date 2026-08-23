import copy

import numpy as np
import pytest

from scripts import classify_relion_capture_target_tie as classifier


def _inertness():
    exact_fields = {
        field: {
            "exact": True,
            "mismatch_count": 0,
            "max_abs": 0.0,
            "mismatch_examples": [],
        }
        for field in classifier.EXPECTED_EXACT_FIELDS
    }
    exact_fields["rlnOriginYAngst"] = {
        "exact": False,
        "mismatch_count": 1,
        "max_abs": 1.0,
        "mismatch_examples": [{"image_identity": "2@particles.mrcs"}],
    }
    return {
        "schema": classifier.INERTNESS_SCHEMA,
        "status": "rejected",
        "scorecard_change_admissible": False,
        "particle_count": 3,
        "expected_particle_count": 3,
        "perturbation_exact": True,
        "strict_gate": {
            "particle_count_exact": True,
            "sampling_perturbation_exact": True,
            "all_half_map_fsc_auc_at_least_threshold": True,
            "all_particle_fields_exact": False,
        },
        "fields": exact_fields,
        "target": {
            "original_index_zero_based": 1,
            "image_identity": "2@particles.mrcs",
            "mismatch_fields": ["rlnOriginYAngst"],
            "control_fields": {
                "rlnOriginXAngst": 2.0,
                "rlnOriginYAngst": 1.0,
            },
            "capture_fields": {
                "rlnOriginXAngst": 2.0,
                "rlnOriginYAngst": 0.0,
            },
        },
    }


def _recovar():
    scores = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    return {
        "original_index": np.array(1),
        "fine_translations": np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "scores_pre_prior": scores,
        "candidate_mask": np.ones_like(scores, dtype=bool),
    }


def _relion():
    return {
        "raw_scores": np.array([0.5, 0.5, 0.2], dtype=np.float64),
        "raw_rot_idx": np.array([1, 1, 0], dtype=np.int64),
        "raw_trans_idx": np.array([1, 2, 0], dtype=np.int64),
        "argmax_index": np.array(0, dtype=np.int64),
    }


def _classify(inertness=None, recovar=None, relion=None):
    return classifier.classify(
        _inertness() if inertness is None else inertness,
        _recovar() if recovar is None else recovar,
        _relion() if relion is None else relion,
        pixel_size=1.0,
    )


def test_exact_target_tie_is_localized_but_not_scorecard_admissible():
    report = _classify()

    assert report["classification"] == "observer_sensitive_exact_tie_winner_flip"
    assert report["target"]["control_key"] == [1, 2]
    assert report["target"]["capture_key"] == [1, 1]
    assert report["exact_raw_score_tie"] == {"relion": True, "recovar": True}
    assert report["diagnostic_admissibility"]["capture_inertness_passed"] is False
    assert report["scorecard_change_admissible"] is False


def test_non_tied_relion_candidates_are_rejected():
    relion = _relion()
    relion["raw_scores"][1] = 0.499

    with pytest.raises(ValueError, match="not an exact raw-score tie"):
        _classify(relion=relion)


def test_non_tied_recovar_candidates_are_rejected():
    recovar = _recovar()
    recovar["scores_pre_prior"][1, 2] = 0.499

    with pytest.raises(ValueError, match="RECOVAR.*not an exact raw-score tie"):
        _classify(recovar=recovar)


def test_mismatch_must_be_the_requested_target_only():
    inertness = _inertness()
    inertness["fields"]["rlnOriginYAngst"]["mismatch_examples"][0][
        "image_identity"
    ] = "3@particles.mrcs"

    with pytest.raises(ValueError, match="not the requested target"):
        _classify(inertness=inertness)


def test_capture_metadata_must_match_relion_selected_translation():
    relion = copy.deepcopy(_relion())
    relion["argmax_index"] = np.array(1, dtype=np.int64)

    with pytest.raises(ValueError, match="does not match captured target metadata"):
        _classify(relion=relion)


def test_translation_mapping_is_fail_closed():
    inertness = _inertness()
    inertness["target"]["capture_fields"]["rlnOriginYAngst"] = 0.1

    with pytest.raises(ValueError, match="does not map within"):
        _classify(inertness=inertness)
