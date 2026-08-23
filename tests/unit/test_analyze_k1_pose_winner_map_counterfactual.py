from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_k1_pose_winner_map_counterfactual import (
    _match_target_candidate,
    _rotation_distances_deg,
)


pytestmark = pytest.mark.unit


def test_rotation_distances_are_geodesic() -> None:
    identity = np.eye(3)
    quarter_turn = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    distances = _rotation_distances_deg(np.stack((identity, quarter_turn)), quarter_turn)
    np.testing.assert_allclose(distances, [90.0, 0.0], atol=1e-12)


def test_match_target_candidate_respects_particle_and_integer_pre_shift() -> None:
    identity = np.eye(3)
    quarter_turn = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = _match_target_candidate(
        active_particle_rows=np.asarray([0, 1, 1]),
        active_rotation_rows=np.asarray([4, 2, 7]),
        active_rotations=np.stack((quarter_turn, identity, quarter_turn)),
        particle_row=1,
        target_rotation=quarter_turn,
        fine_translations=np.asarray([[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]),
        integer_pre_shift=np.asarray([2.0, -3.0]),
        target_translation_pixels=np.asarray([2.5, -3.0]),
    )
    assert result["rotation_row"] == 7
    assert result["rotation_error_deg"] == pytest.approx(0.0)
    assert result["translation_index"] == 2
    assert result["translation_error_pixels"] == pytest.approx(0.0)
