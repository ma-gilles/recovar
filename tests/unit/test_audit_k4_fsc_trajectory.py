from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import audit_k4_fsc_trajectory as auditor


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path.resolve()


def _make_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    recovar_indices: tuple[int, ...] = (0, 1),
    relion_iterations: tuple[int, ...] = (1, 2),
    missing_numbered: tuple[str, int, int, int] | None = None,
    missing_final: tuple[str, int] | None = None,
) -> tuple[Path, dict[Path, np.ndarray]]:
    case_root = tmp_path / "case"
    recovar_dir = case_root / "recovar"
    intermediates = recovar_dir / "intermediates"
    relion_dir = case_root / "relion_ref"
    data_dir = case_root / "data"
    rng = np.random.default_rng(29)
    gt = [rng.normal(size=(8, 8, 8)).astype(np.float64) for _ in range(4)]
    # RELION classes contain GT classes [3, 1, 4, 2], so the correct
    # zero-based RECOVAR-to-RELION assignment is [1, 3, 0, 2].
    relion_source = [2, 0, 3, 1]
    arrays: dict[Path, np.ndarray] = {}

    for class_id, volume in enumerate(gt, start=1):
        arrays[_touch(data_dir / f"reference_gt_class{class_id:03d}.mrc")] = volume
    for iteration in recovar_indices:
        for half in (1, 2):
            for class_id, volume in enumerate(gt, start=1):
                if missing_numbered == ("recovar", iteration, half, class_id):
                    continue
                # Production debug dumps intentionally spell this class1, not class001.
                arrays[_touch(intermediates / f"it{iteration:03d}_half{half}_class{class_id}_reg.mrc")] = volume
    for iteration in relion_iterations:
        for half in (1, 2):
            for class_id, source_id in enumerate(relion_source, start=1):
                if missing_numbered == ("relion", iteration, half, class_id):
                    continue
                arrays[_touch(relion_dir / f"run_it{iteration:03d}_half{half}_class{class_id:03d}.mrc")] = gt[source_id]
    for class_id, volume in enumerate(gt, start=1):
        if missing_final != ("recovar", class_id):
            arrays[_touch(recovar_dir / f"final_class{class_id:03d}.mrc")] = volume
    for class_id, source_id in enumerate(relion_source, start=1):
        if missing_final != ("relion", class_id):
            arrays[_touch(relion_dir / f"run_class{class_id:03d}.mrc")] = gt[source_id]

    recovar_dir.mkdir(parents=True, exist_ok=True)
    np.savez(recovar_dir / "refinement_results.npz", current_sizes=np.arange(len(recovar_indices)))

    def load(path: Path) -> np.ndarray:
        return arrays[Path(path).resolve()]

    monkeypatch.setattr(auditor, "_load_recovar_volume", load)
    monkeypatch.setattr(auditor, "_load_relion_volume", load)
    return case_root, arrays


def _run(case_root: Path, tmp_path: Path, *extra: str) -> tuple[int, dict, Path]:
    output_json = tmp_path / "report.json"
    output_markdown = tmp_path / "report.md"
    output_npz = tmp_path / "curves.npz"
    status = auditor.main(
        [
            "--case-root",
            str(case_root),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
            "--output-shellwise-npz",
            str(output_npz),
            *extra,
        ]
    )
    return status, json.loads(output_json.read_text()), output_npz


@pytest.mark.unit
def test_complete_k4_trajectory_matches_each_iteration_and_final_by_fsc_auc(tmp_path, monkeypatch):
    case_root, _ = _make_case(tmp_path, monkeypatch)

    status, report, output_npz = _run(case_root, tmp_path)

    assert status == 0
    assert report["status"] == "pass"
    assert report["numbered_iteration_count"] == 2
    assert [row["recovar_index"] for row in report["numbered_iterations"]] == [0, 1]
    assert [row["relion_iteration"] for row in report["numbered_iterations"]] == [1, 2]
    assert report["numbered_iterations"][0]["recovar_to_relion_assignment"] == [2, 4, 1, 3]
    assert report["numbered_iterations"][0]["matched_pair_to_gt_assignment"] == [1, 2, 3, 4]
    assert len(report["numbered_iterations"][0]["classes"]) == 4
    assert report["final"]["recovar_to_relion_assignment"] == [2, 4, 1, 3]
    assert all(
        item["cross_engine"]["merged"]["fsc_auc"] == pytest.approx(1.0)
        for item in report["numbered_iterations"][0]["classes"]
    )
    with np.load(output_npz, allow_pickle=False) as curves:
        assert "it001_rec001_rel002_cross_half1" in curves.files
        assert "it002_rec004_gt004_merged" in curves.files
        assert "final_rec003_rel001_cross" in curves.files
    markdown = (tmp_path / "report.md").read_text()
    assert "Correlation was not computed" in markdown
    assert "REC class" in markdown


@pytest.mark.unit
def test_missing_numbered_class_fails_closed(tmp_path, monkeypatch):
    case_root, _ = _make_case(
        tmp_path,
        monkeypatch,
        recovar_indices=(0,),
        relion_iterations=(1,),
        missing_numbered=("recovar", 0, 2, 4),
    )

    status, report, output_npz = _run(case_root, tmp_path)

    assert status == 2
    assert report["status"] == "error"
    assert "numbered K=4 topology is incomplete" in report["earliest_failure"]
    with np.load(output_npz, allow_pickle=False) as curves:
        assert curves.files == []


@pytest.mark.unit
def test_noncontiguous_iteration_fails_closed(tmp_path, monkeypatch):
    case_root, _ = _make_case(
        tmp_path,
        monkeypatch,
        recovar_indices=(0, 2),
        relion_iterations=(1, 2),
    )

    status, report, _ = _run(case_root, tmp_path)

    assert status == 2
    assert "not contiguous zero-based" in report["earliest_failure"]


@pytest.mark.unit
def test_incomplete_final_class_products_fail_closed(tmp_path, monkeypatch):
    case_root, _ = _make_case(
        tmp_path,
        monkeypatch,
        recovar_indices=(0,),
        relion_iterations=(1,),
        missing_final=("relion", 4),
    )

    status, report, _ = _run(case_root, tmp_path)

    assert status == 2
    assert "RELION final K=4 products are incomplete" in report["earliest_failure"]


@pytest.mark.unit
def test_one_bad_class_fails_direct_fsc_gate_without_mean_hiding_it(tmp_path, monkeypatch):
    case_root, arrays = _make_case(tmp_path, monkeypatch, recovar_indices=(0,), relion_iterations=(1,))
    rng = np.random.default_rng(91)
    arrays[(case_root / "relion_ref" / "run_it001_half1_class002.mrc").resolve()] = rng.normal(size=(8, 8, 8))
    arrays[(case_root / "relion_ref" / "run_it001_half2_class002.mrc").resolve()] = rng.normal(size=(8, 8, 8))

    status, report, _ = _run(case_root, tmp_path)

    assert status == 2
    assert report["status"] == "fail"
    assert any("it001" in failure and "direct FSC-AUC" in failure for failure in report["failures"])


@pytest.mark.unit
def test_class_agreement_aligns_by_image_identity_and_map_permutation(tmp_path, monkeypatch):
    refinement_results = tmp_path / "refinement_results.npz"
    np.savez(
        refinement_results,
        class_assignments_by_image_iter_000=np.asarray([0, 1, 2, 3], dtype=np.int32),
    )
    fixture_star = _touch(tmp_path / "particles.star")
    relion_star = _touch(tmp_path / "run_it001_data.star")
    fixture = pd.DataFrame({"rlnImageName": ["a", "b", "c", "d"]})
    # Rows are shuffled; class values map REC classes through [1, 3, 0, 2].
    relion = pd.DataFrame(
        {
            "rlnImageName": ["d", "b", "a", "c"],
            "rlnClassNumber": [3, 4, 2, 1],
        }
    )
    monkeypatch.setattr(auditor, "_particle_table", lambda path: fixture if path == fixture_star else relion)

    result = auditor._class_agreement(
        refinement_results=refinement_results,
        relion_data_star=relion_star,
        fixture_particles_star=fixture_star,
        recovar_iteration=0,
        rel_for_rec=[1, 3, 0, 2],
    )

    assert result["status"] == "available"
    assert result["matched_count"] == 4
    assert result["agreement"] == pytest.approx(1.0)


@pytest.mark.unit
def test_available_class_agreement_below_99_percent_is_gated(tmp_path, monkeypatch):
    case_root, _ = _make_case(tmp_path, monkeypatch, recovar_indices=(0,), relion_iterations=(1,))
    monkeypatch.setattr(
        auditor,
        "_class_agreement",
        lambda **_kwargs: {"status": "available", "agreement": 0.98, "matched_count": 100},
    )

    status, report, _ = _run(case_root, tmp_path)

    assert status == 2
    assert any("class agreement 0.980000000 < 0.990000000" in failure for failure in report["failures"])


@pytest.mark.unit
def test_each_class_gt_delta_is_gated_individually():
    passing_class = {
        "recovar_class": 1,
        "relion_class": 1,
        "cross_engine": {"merged": {"fsc_auc": 0.999}},
        "merged_gt_fsc_auc_delta": 0.0,
    }
    failing_class = {
        **passing_class,
        "recovar_class": 4,
        "relion_class": 2,
        "merged_gt_fsc_auc_delta": -0.003,
    }
    row = {
        "relion_iteration": 1,
        "classes": [passing_class, passing_class, passing_class, failing_class],
        "class_agreement": {"status": "unavailable", "reason": "fixture intentionally omitted"},
    }
    final_class = {
        "recovar_class": 1,
        "relion_class": 1,
        "cross_engine": {"fsc_auc": 0.999},
        "gt_fsc_auc_delta": 0.0,
    }

    failures = auditor._apply_gates(
        [row],
        {"classes": [final_class] * 4},
        min_cross=0.995,
        min_gt_delta=-0.002,
        min_class_agreement=0.99,
    )

    assert failures == ["it001 RECOVAR class 4 / RELION class 2 GT FSC-AUC delta -0.003 < -0.002000000"]


@pytest.mark.unit
def test_missing_assignment_artifacts_are_reported_not_invented(tmp_path, monkeypatch):
    case_root, _ = _make_case(tmp_path, monkeypatch, recovar_indices=(0,), relion_iterations=(1,))

    status, report, _ = _run(case_root, tmp_path)

    assert status == 0
    assert report["numbered_iterations"][0]["class_agreement"]["status"] == "unavailable"
    assert "missing fixture particle STAR" in report["class_agreement_unavailable"][0]["reason"]
