"""Unit guards for sealing positive K-class benchmark campaigns."""

from __future__ import annotations

import copy
import json

import pytest

import scripts.seal_em_kclass_campaign as sealer

pytestmark = pytest.mark.unit


def _class_row(
    *,
    recovar_class: int,
    relion_class: int,
    gt_class: int,
    direct: float,
    recovar_gt: float,
    relion_gt: float,
) -> dict:
    return {
        "recovar_class": recovar_class,
        "relion_class": relion_class,
        "gt_class": gt_class,
        "cross_engine": {"fsc_auc": direct},
        "vs_gt": {
            "recovar": {"fsc_auc": recovar_gt},
            "relion": {"fsc_auc": relion_gt},
        },
        "gt_fsc_auc_delta": recovar_gt - relion_gt,
    }


def _audit(*, status: str = "pass") -> dict:
    first = [
        _class_row(
            recovar_class=1,
            relion_class=2,
            gt_class=1,
            direct=0.999,
            recovar_gt=0.30,
            relion_gt=0.3001,
        ),
        _class_row(
            recovar_class=2,
            relion_class=1,
            gt_class=2,
            direct=0.998,
            recovar_gt=0.31,
            relion_gt=0.3102,
        ),
    ]
    second = copy.deepcopy(first)
    return {
        "status": status,
        "thresholds": {
            "per_class_direct_fsc_auc_min": 0.995,
            "per_class_recovar_minus_relion_gt_fsc_auc_min": -0.002,
            "class_assignment_agreement_min_when_available": 0.99,
        },
        "numbered_iterations": [
            {"classes": first, "class_agreement": {"agreement": 1.0}},
            {"classes": second, "class_agreement": {"agreement": 0.98}},
        ],
        "final": {"classes": copy.deepcopy(second)},
        "earliest_failure": None if status == "pass" else "it002 class agreement failed",
    }


def _endpoint(label: str) -> dict:
    matched_gt = (0, 1) if label == "RECOVAR" else (1, 0)
    classes = [
        {
            "class": 0,
            "matched_gt_class": matched_gt[0],
            "resolution_0143_A": 20.0 if label == "RECOVAR" else 21.0,
        },
        {
            "class": 1,
            "matched_gt_class": matched_gt[1],
            "resolution_0143_A": 22.0 if label == "RECOVAR" else 23.0,
        },
    ]
    return {"primary": {"per_class": classes}}


def test_class_cell_count_and_science_gate_are_distinct_from_assignment_gate():
    audit = _audit(status="fail")
    assert sealer._passing_class_cells(audit) == 4
    assert sealer._science_passes(audit)

    audit["numbered_iterations"][0]["classes"][0]["gt_fsc_auc_delta"] = -0.0021
    assert sealer._passing_class_cells(audit) == 3
    assert not sealer._science_passes(audit)


def test_outcome_preserves_trajectory_and_science_equivalence_labels():
    final_rows = [{"cross_engine_fsc_auc": 0.998, "gt_fsc_auc_delta": -0.0002}]
    clear = {"status": "CLEAR"}

    exact = sealer._outcome(_audit(status="pass"), clear, final_rows)
    assert exact["classification"] == "TRAJECTORY_EXACT"
    assert (exact["formal_status"], exact["science_status"]) == ("PASS", "PASS")

    science = sealer._outcome(_audit(status="fail"), clear, final_rows)
    assert science["classification"] == "SCIENCE_EQUIVALENT"
    assert (science["formal_status"], science["science_status"]) == ("FAIL", "PASS")

    failed_audit = _audit(status="fail")
    failed_audit["final"]["classes"][0]["gt_fsc_auc_delta"] = -0.01
    unresolved = sealer._outcome(failed_audit, clear, final_rows)
    assert unresolved["classification"] == "UNRESOLVED_TRAJECTORY_FAILURE"
    assert unresolved["science_status"] == "UNRESOLVED"


def test_outcome_marks_near_collapse_and_rejects_zero_class():
    final_rows = [{"cross_engine_fsc_auc": 0.999, "gt_fsc_auc_delta": 0.0}]
    near = sealer._outcome(_audit(), {"status": "NEAR_COLLAPSE"}, final_rows)
    assert near["classification"] == "TRAJECTORY_EXACT_NEAR_COLLAPSE"
    assert near["science_status"] == "BOUNDARY"
    with pytest.raises(ValueError, match="zero-population"):
        sealer._outcome(_audit(), {"status": "ZERO_CLASS"}, final_rows)


def test_final_rows_join_trajectory_metrics_to_endpoint_resolutions():
    rows = sealer._final_class_rows(_audit(), _endpoint("RECOVAR"), _endpoint("RELION"))
    assert [(row["recovar_class"], row["relion_class"], row["gt_class"]) for row in rows] == [
        (1, 2, 1),
        (2, 1, 2),
    ]
    assert [row["recovar_gt_resolution_0_143_angstrom"] for row in rows] == [20.0, 22.0]
    assert [row["relion_gt_resolution_0_143_angstrom"] for row in rows] == [23.0, 21.0]


def test_final_rows_fail_closed_on_endpoint_matching_disagreement():
    relion = _endpoint("RELION")
    relion["primary"]["per_class"][1]["matched_gt_class"] = 1
    with pytest.raises(ValueError, match="endpoint evaluator disagrees"):
        sealer._final_class_rows(_audit(), _endpoint("RECOVAR"), relion)


def test_occupancy_flags_are_mechanical(monkeypatch, tmp_path):
    monkeypatch.setattr(sealer, "_class_counts", lambda *args, **kwargs: ([991, 9], [990, 10]))
    occupancy = sealer._occupancy(tmp_path, n_classes=2, iterations=5, particles=1000)
    assert occupancy["recovar_flagged_classes"] == [2]
    assert occupancy["relion_flagged_classes"] == []
    assert occupancy["status"] == "NEAR_COLLAPSE"


def test_trajectory_manifest_is_deterministic_and_tamper_evident(tmp_path):
    case_root = tmp_path / "case"
    for path in (
        case_root / "recovar/intermediates/it000_half1_class1_reg.mrc",
        case_root / "recovar/intermediates/it000_half2_class1_reg.mrc",
        case_root / "relion_ref/run_it001_class001.mrc",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    (case_root / "trajectory_analysis").mkdir()

    manifest = sealer._write_trajectory_manifest(case_root, iterations=1, n_classes=1)
    first = manifest.read_text()
    assert sealer._write_trajectory_manifest(case_root, iterations=1, n_classes=1) == manifest
    payload = json.loads(first)
    assert len(payload["files"]) == 3

    manifest.write_text(first.replace('"n_classes": 1', '"n_classes": 2'))
    with pytest.raises(ValueError, match="manifest differs"):
        sealer._write_trajectory_manifest(case_root, iterations=1, n_classes=1)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2", (2,)), ("2,8,16", (2, 8, 16))],
)
def test_parse_seeds(value, expected):
    assert sealer._parse_seeds(value) == expected


@pytest.mark.parametrize("value", ["", "8,2", "2,2"])
def test_parse_seeds_rejects_ambiguous_values(value):
    with pytest.raises(Exception, match="sorted unique"):
        sealer._parse_seeds(value)


def test_job_reference_requires_terminal_exact_allocation(monkeypatch, tmp_path):
    log = tmp_path / "job.out"
    log.write_text("ok\n")
    base = {
        "job_id": "123",
        "status": "COMPLETED",
        "elapsed_s": 2,
        "exit_code": "0:0",
        "node": "node1",
        "req_tres": "cpu=4,mem=64G,node=1",
        "alloc_tres": "cpu=4,mem=64G,node=1",
        "max_rss_kib": 10,
    }
    monkeypatch.setattr(sealer, "_sacct_job", lambda _: copy.deepcopy(base))
    reference = sealer._job_reference("123", role="audit", logs=[log], gpu_model=None, gpu_uuid=None)
    assert reference["role"] == "audit"
    assert reference["logs"] == [str(log.resolve())]

    mismatch = copy.deepcopy(base)
    mismatch["alloc_tres"] = "cpu=8,mem=64G,node=1"
    monkeypatch.setattr(sealer, "_sacct_job", lambda _: mismatch)
    with pytest.raises(ValueError, match="resources differ"):
        sealer._job_reference("123", role="audit", logs=[log], gpu_model=None, gpu_uuid=None)
