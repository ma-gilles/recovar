import json

import pytest

from scripts.summarize_k1_selected_treatment_prefix import FIXED_CASES, summarize


def _movement(path, *, candidate_pmax=0.1, gate=True):
    arm = {
        "pmax": {"relative_l2": 0.2},
        "support": {"mismatch_count": 3},
        "pose_error_gt_0p01_deg_count": 2,
        "translation_error_gt_0p01_angstrom_count": 1,
        "merged_signed_fsc_auc": 0.999,
    }
    candidate = json.loads(json.dumps(arm))
    candidate["pmax"]["relative_l2"] = candidate_pmax
    report = {
        "schema": "recovar.em.k1_autonomous_boundary_movement.v1",
        "classification": "moves_toward_relion_without_measured_regression",
        "arms": {"baseline": arm, "candidate": candidate},
        "movement": {"support": {"new_source_rows": []}},
        "gates": {"example": gate},
    }
    path.write_text(json.dumps(report))


def test_summarize_uses_fixed_denominator_and_all_gates(tmp_path):
    paths = {}
    for case_id in FIXED_CASES:
        path = tmp_path / f"case{case_id}.json"
        _movement(path, gate=case_id != 5)
        paths[case_id] = path

    report = summarize(paths)

    assert report["counts"] == {"passed": 2, "total": 3}
    assert not report["terminal_run_eligible"]
    assert [item["case"] for item in report["cases"]] == [4, 5, 10]
    assert report["cases"][0]["candidate"]["pmax_relative_l2"] == 0.1


def test_summarize_rejects_changed_denominator_and_schema(tmp_path):
    path = tmp_path / "case4.json"
    _movement(path)
    with pytest.raises(ValueError, match="requires cases"):
        summarize({4: path})

    paths = {}
    for case_id in FIXED_CASES:
        item = tmp_path / f"bad{case_id}.json"
        _movement(item)
        paths[case_id] = item
    payload = json.loads(paths[10].read_text())
    payload["schema"] = "wrong"
    paths[10].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unexpected movement schema"):
        summarize(paths)
