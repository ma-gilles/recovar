import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import select_k1_native_operand_target as selector
from scripts.validate_relion_fine_score_capture import ACTIVE, CANDIDATE_DTYPE


def _stage_join(tmp_path: Path, *, boundary: str = "preprior_score_centered") -> Path:
    recovar = tmp_path / "pass2.npz"
    np.savez(
        recovar,
        rotations=np.zeros((2, 3, 3), dtype=np.float32),
        fine_translations=np.zeros((3, 2), dtype=np.float32),
    )
    factor = tmp_path / "factor.bin"
    score = tmp_path / "score.bin"
    factor.write_bytes(b"factor")
    score.write_bytes(b"score")
    stage = {
        "first_exact_unequal_boundary": boundary,
        "first_mismatch": {
            "preprior_score_centered": {"tuple_key": [1, 2]},
        },
        "first_support_native_only_key": [1, 2],
        "first_support_recovar_only_key": None,
        "native_winner": {"recovar_rotation_row": 1, "recovar_translation_row": 2},
        "native_factor": str(factor),
        "native_fine_score": str(score),
        "recovar_capture": str(recovar),
        "physical_image_size": 256,
    }
    join = {
        "schema": "recovar.em.k1_top1_stage_join.v1",
        "status": "complete",
        "target": {"stack_index_one_based": 31560},
        "stage_analysis": stage,
    }
    path = tmp_path / "join.json"
    path.write_text(json.dumps(join))
    return path


@pytest.mark.parametrize("boundary", ["preprior_score_centered", "significant_support", "hard_winner"])
def test_selector_maps_first_unequal_tuple_to_native_ids(
    tmp_path: Path, monkeypatch, boundary: str
) -> None:
    join = _stage_join(tmp_path, boundary=boundary)
    candidates = np.zeros(2, dtype=CANDIDATE_DTYPE)
    candidates["rotation_local"] = [0, 1]
    candidates["translation_id"] = [0, 2]
    candidates["raw_diff2"] = [10.0, 11.0]
    candidates["flags"] = ACTIVE
    factor = SimpleNamespace(stack_index=31560, rotations=np.zeros(2), translations=np.zeros(3))
    score = SimpleNamespace(stack_index=31560, candidates=candidates)
    monkeypatch.setattr(selector, "load_factor_capture", lambda _path: factor)
    monkeypatch.setattr(selector, "load_fine_score_capture", lambda _path: score)
    monkeypatch.setattr(selector, "_rotation_map", lambda *_args: (np.asarray([0, 1]), 0.0))
    monkeypatch.setattr(selector, "_translation_map", lambda *_args, **_kwargs: (np.asarray([0, 1, 2]), 0.0))

    report = selector.select_target(stage_join_json=join)

    assert report["first_exact_unequal_boundary"] == boundary
    assert report["native_rotation_local"] == 1
    assert report["native_translation_id"] == 2
    assert report["native_raw_diff2"] == 11.0
    assert report["recovar_rotation_row"] == 1
    assert report["recovar_translation_row"] == 2


def test_selector_rejects_boundary_without_operand_tuple(tmp_path: Path) -> None:
    join = _stage_join(tmp_path, boundary="all_stages_exact")

    with pytest.raises(ValueError, match="cannot select an operand tuple"):
        selector.select_target(stage_join_json=join)


def test_first_unequal_tuple_accepts_coordinate_fields() -> None:
    stage = {
        "first_exact_unequal_boundary": "raw_diff2",
        "first_mismatch": {
            "raw_diff2": {
                "recovar_rotation_row": 4,
                "recovar_translation_row": 9,
            }
        },
    }

    assert selector._first_unequal_tuple(stage) == ("raw_diff2", (4, 9))
