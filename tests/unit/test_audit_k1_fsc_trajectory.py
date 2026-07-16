from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_k1_fsc_trajectory as auditor


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
    missing_recovar_half: tuple[int, int] | None = None,
    noctf: bool = False,
    include_final: bool = False,
    invert_gt: bool = False,
) -> tuple[Path, dict[Path, np.ndarray]]:
    case_root = tmp_path / "case"
    recovar_dir = case_root / "recovar"
    intermediates = recovar_dir / "intermediates"
    relion_dir = case_root / "relion_ref"
    gt_path = _touch(case_root / "data" / "reference_gt.mrc")
    rng = np.random.default_rng(17)
    gt = rng.normal(size=(8, 8, 8)).astype(np.float64)
    map_volume = -gt if invert_gt else gt
    arrays: dict[Path, np.ndarray] = {gt_path: gt}

    for iteration in recovar_indices:
        for half in (1, 2):
            if missing_recovar_half == (iteration, half):
                continue
            path = _touch(intermediates / f"it{iteration:03d}_half{half}_reg.mrc")
            arrays[path] = map_volume
    for iteration in relion_iterations:
        for half in (1, 2):
            path = _touch(relion_dir / f"run_it{iteration:03d}_half{half}_class001.mrc")
            arrays[path] = map_volume

    recovar_dir.mkdir(parents=True, exist_ok=True)
    np.savez(recovar_dir / "refinement_results.npz", current_sizes=np.arange(len(recovar_indices)))
    (case_root / "case_config.json").write_text(
        json.dumps({"dataset_params_option": "noctf" if noctf else "uniform"}) + "\n"
    )

    if include_final:
        for name in ("final_half1.mrc", "final_half2.mrc", "final_merged.mrc"):
            arrays[_touch(recovar_dir / name)] = map_volume
        for name in (
            "run_half1_class001_unfil.mrc",
            "run_half2_class001_unfil.mrc",
            "run_class001.mrc",
        ):
            arrays[_touch(relion_dir / name)] = map_volume

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
def test_complete_numbered_and_final_trajectory_outputs_all_fsc_series(tmp_path, monkeypatch):
    case_root, _ = _make_case(tmp_path, monkeypatch, include_final=True)

    status, report, output_npz = _run(case_root, tmp_path)

    assert status == 0
    assert report["status"] == "pass"
    assert report["numbered_iteration_count"] == 2
    assert [row["recovar_index"] for row in report["numbered_iterations"]] == [0, 1]
    assert [row["relion_iteration"] for row in report["numbered_iterations"]] == [1, 2]
    first = report["numbered_iterations"][0]
    assert set(first["cross_engine"]) == {"half1", "half2", "merged"}
    assert set(first["vs_gt"]["recovar"]) == {"half1", "half2", "merged"}
    assert set(first["vs_gt"]["relion"]) == {"half1", "half2", "merged"}
    assert first["cross_engine"]["merged"]["fsc_auc"] == pytest.approx(1.0, abs=1e-12)
    assert first["merged_gt_fsc_auc_delta"] == pytest.approx(0.0, abs=1e-12)
    assert set(report["final"]["cross_engine"]) == {"half1", "half2", "merged"}

    with np.load(output_npz, allow_pickle=False) as curves:
        expected_numbered = {
            f"it{iteration:03d}_{kind}_{label}"
            for iteration in (1, 2)
            for kind, labels in (
                ("cross", ("half1", "half2", "merged")),
                ("recovar_gt", ("half1", "half2", "merged")),
                ("relion_gt", ("half1", "half2", "merged")),
            )
            for label in labels
        }
        assert expected_numbered.issubset(curves.files)
        assert "final_cross_merged" in curves.files
        assert "final_recovar_gt_half1" in curves.files
        assert "final_relion_gt_half2" in curves.files
    markdown = (tmp_path / "report.md").read_text()
    assert "Correlation was not computed" in markdown
    assert "Cross half1" in markdown


@pytest.mark.unit
def test_noctf_auto_mode_uses_sign_invariant_gt_but_signed_cross_engine(tmp_path, monkeypatch):
    case_root, _ = _make_case(
        tmp_path,
        monkeypatch,
        recovar_indices=(0,),
        relion_iterations=(1,),
        noctf=True,
        invert_gt=True,
    )

    status, report, _ = _run(case_root, tmp_path)

    assert status == 0
    assert report["gt_sign_policy"] == {
        "requested": "auto",
        "used": "sign_invariant",
        "reason": "auto_noctf",
    }
    row = report["numbered_iterations"][0]
    assert row["cross_engine"]["merged"]["sign_mode_used"] == "signed"
    assert row["cross_engine"]["merged"]["fsc_auc"] == pytest.approx(1.0, abs=1e-12)
    rec_gt = row["vs_gt"]["recovar"]["merged"]
    assert rec_gt["signed_fsc_auc"] == pytest.approx(-1.0, abs=1e-12)
    assert rec_gt["fsc_auc"] == pytest.approx(1.0, abs=1e-12)
    assert rec_gt["sign_invariant_best_sign"] == -1


@pytest.mark.unit
def test_noncontiguous_numbered_maps_fail_closed_and_still_write_artifacts(tmp_path, monkeypatch):
    case_root, _ = _make_case(
        tmp_path,
        monkeypatch,
        recovar_indices=(0, 2),
        relion_iterations=(1, 2),
    )

    status, report, output_npz = _run(case_root, tmp_path)

    assert status == 2
    assert report["status"] == "error"
    assert "not contiguous zero-based" in report["earliest_failure"]
    assert (tmp_path / "report.md").is_file()
    with np.load(output_npz, allow_pickle=False) as curves:
        assert curves.files == []


@pytest.mark.unit
def test_incomplete_half_pair_fails_before_map_loading(tmp_path, monkeypatch):
    case_root, _ = _make_case(
        tmp_path,
        monkeypatch,
        recovar_indices=(0,),
        relion_iterations=(1,),
        missing_recovar_half=(0, 2),
    )

    status, report, _ = _run(case_root, tmp_path)

    assert status == 2
    assert report["status"] == "error"
    assert "half-map pairs are incomplete" in report["earliest_failure"]


@pytest.mark.unit
def test_cross_engine_iteration_count_mismatch_fails_closed(tmp_path, monkeypatch):
    case_root, _ = _make_case(
        tmp_path,
        monkeypatch,
        recovar_indices=(0, 1),
        relion_iterations=(1,),
    )

    status, report, _ = _run(case_root, tmp_path)

    assert status == 2
    assert report["status"] == "error"
    assert "numbered iteration count mismatch" in report["earliest_failure"]


@pytest.mark.unit
def test_refinement_result_iteration_count_mismatch_fails_closed(tmp_path, monkeypatch):
    case_root, _ = _make_case(tmp_path, monkeypatch, recovar_indices=(0,), relion_iterations=(1,))
    np.savez(case_root / "recovar" / "refinement_results.npz", current_sizes=np.arange(2))

    status, report, _ = _run(case_root, tmp_path)

    assert status == 2
    assert report["status"] == "error"
    assert "map/result iteration count mismatch" in report["earliest_failure"]


@pytest.mark.unit
def test_partial_final_products_fail_closed(tmp_path, monkeypatch):
    case_root, arrays = _make_case(tmp_path, monkeypatch, recovar_indices=(0,), relion_iterations=(1,))
    partial_final = _touch(case_root / "recovar" / "final_merged.mrc")
    arrays[partial_final] = next(iter(arrays.values()))

    status, report, _ = _run(case_root, tmp_path)

    assert status == 2
    assert report["status"] == "error"
    assert "final products are partially present" in report["earliest_failure"]
