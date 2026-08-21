from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recovar.data_io.starfile import write_star
from scripts import audit_vdam_fsc_trajectory as audit_module

pytestmark = pytest.mark.unit


DEFINITION = {
    "source_em_case_id": "k1-25",
    "nr_classes": 1,
    "nr_iter": 8,
    "random_seed": 0,
    "tau2_fudge": 4.0,
    "healpix_order": 1,
    "oversampling": 1,
    "offset_range_px": 6.0,
    "offset_step_px": 2.0,
    "padding_factor": 1,
}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    fixture_dir = tmp_path / "fixture"
    recovar_dir = tmp_path / "recovar"
    relion_dir = tmp_path / "relion"
    for directory in (fixture_dir, recovar_dir, relion_dir):
        directory.mkdir()

    scorecard = {
        "schema": audit_module.SCORECARD_SCHEMA,
        "suite_id": "vdam-k1-gui-grid0-fixed12",
        "source_fixture_manifest": {"sha256": "a" * 64},
        "acceptance_contract": {
            "required_checkpoints": list(audit_module.CHECKPOINTS),
            "cross_engine_fsc_auc_min": 0.999,
            "recovar_minus_relion_gt_fsc_auc_min": -0.002,
            "correlation_used": False,
        },
        "cases": [{"id": "vdam-08", "definition": DEFINITION}],
    }
    scorecard_path = tmp_path / "scorecard.json"
    _write_json(scorecard_path, scorecard)
    _write_json(
        fixture_dir / "fixture_materialization.json",
        {
            "schema": "recovar.em_k1_fixture_materialization.v1",
            "manifest_sha256": "a" * 64,
            "case_id": "k1-25",
            "files": [{"name": "particles.star"}],
        },
    )
    (fixture_dir / "reference_gt_relion.mrc").touch()
    write_star(
        str(fixture_dir / "particles.star"),
        pd.DataFrame({"_rlnImageName": [f"image-{idx}@stack.mrcs" for idx in range(6)]}),
    )
    _write_json(
        recovar_dir / "run_native_options.json",
        {
            "nr_classes": 1,
            "nr_iter": 8,
            "random_seed": 0,
            "tau2_fudge": 4.0,
            "healpix_order": 1,
            "oversampling": 1,
            "offset_range_px": 6.0,
            "offset_step_px": 2.0,
            "padding_factor": 1,
            "sym_name": "C1",
            "particle_diameter": 200.0,
        },
    )
    _write_json(
        relion_dir / "relion_command.json",
        {
            "argv": [
                "relion_refine",
                "--grad",
                "--denovo_3dref",
                "--grad_write_iter",
                "1",
                "--flatten_solvent",
                "--zero_mask",
                "--auto_sampling",
                "--sym",
                "C1",
                "--particle_diameter",
                "200.0",
                "--K",
                "1",
                "--iter",
                "8",
                "--random_seed",
                "0",
                "--tau2_fudge",
                "4.0",
                "--healpix_order",
                "1",
                "--oversampling",
                "1",
                "--offset_range",
                "6.0",
                "--offset_step",
                "2.0",
                "--pad",
                "1",
            ]
        },
    )
    for iteration in audit_module.CHECKPOINTS:
        for directory in (recovar_dir, relion_dir):
            for path in audit_module._artifact_paths(directory, iteration).values():
                path.touch()
    _write_json(
        recovar_dir / "run_it001_recovar_meta.json",
        {
            "selected_particle_ids": [0, 2, 1, 3],
            "halfset_0_class_assignments": [0, 0],
            "halfset_1_class_assignments": [0, 0],
        },
    )
    write_star(
        str(relion_dir / "run_it001_data.star"),
        pd.DataFrame(
            {
                "_rlnImageName": [f"image-{idx}@stack.mrcs" for idx in range(6)],
                "_rlnMaxValueProbDistribution": [0.5, 0.4, 0.3, 0.2, 0.0, 0.0],
            }
        ),
    )
    gpu_report = tmp_path / "paired_gpu_uuid.json"
    _write_json(
        gpu_report,
        {
            "physical_gpu_uuid": "GPU-fixed",
            "relion_gpu_uuid": "GPU-fixed",
            "recovar_gpu_uuid": "GPU-fixed",
        },
    )
    return {
        "scorecard_path": scorecard_path,
        "fixture_dir": fixture_dir,
        "recovar_dir": recovar_dir,
        "relion_dir": relion_dir,
        "paired_gpu_report_path": gpu_report,
    }


def _audit(paths: dict[str, Path]):
    return audit_module.audit(case_id="vdam-08", **paths)


def test_identical_fixed_checkpoint_maps_pass_without_correlation(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    volume = np.random.default_rng(0).normal(size=(8, 8, 8))
    monkeypatch.setattr(audit_module, "_load_relion_volume", lambda _path: volume)

    report, shellwise = _audit(paths)

    assert report["result"] == "pass"
    assert report["correlation_used"] is False
    for checkpoint in report["checkpoints"]:
        assert all(
            "corr" not in key.lower()
            for metric in ("cross_engine", "recovar_gt", "relion_gt")
            for key in checkpoint[metric]
        )
    assert [row["iteration"] for row in report["checkpoints"]] == list(audit_module.CHECKPOINTS)
    assert report["iteration_one_particle_subset"] == {
        "exact": True,
        "identity": "_rlnImageName",
        "particle_count": 4,
        "first_image_name": "image-0@stack.mrcs",
        "last_image_name": "image-3@stack.mrcs",
    }
    assert len(shellwise) == 3 * len(audit_module.CHECKPOINTS)


def test_iteration_one_particle_identity_mismatch_is_rejected(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    _write_json(
        paths["recovar_dir"] / "run_it001_recovar_meta.json",
        {
            "selected_particle_ids": [0, 2, 4, 1],
            "halfset_0_class_assignments": [0, 0],
            "halfset_1_class_assignments": [0, 0],
        },
    )
    monkeypatch.setattr(
        audit_module,
        "_load_relion_volume",
        lambda _path: np.random.default_rng(6).normal(size=(8, 8, 8)),
    )

    with pytest.raises(audit_module.AuditError, match="iteration-1 particle subsets differ"):
        _audit(paths)


def test_one_failed_checkpoint_fails_the_whole_trajectory(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    volume = np.random.default_rng(1).normal(size=(8, 8, 8))

    def load(path: Path):
        if path.parent == paths["relion_dir"] and path.name.startswith("run_it004_"):
            return -volume
        return volume

    monkeypatch.setattr(audit_module, "_load_relion_volume", load)

    report, _ = _audit(paths)

    assert report["result"] == "fail"
    assert next(row for row in report["checkpoints"] if row["iteration"] == 4)["pass"] is False


def test_missing_common_artifact_is_rejected(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    paths["recovar_dir"].joinpath("run_it002_data.star").unlink()
    monkeypatch.setattr(
        audit_module,
        "_load_relion_volume",
        lambda _path: np.random.default_rng(2).normal(size=(8, 8, 8)),
    )

    with pytest.raises(audit_module.AuditError, match="missing required artifacts"):
        _audit(paths)


def test_same_physical_gpu_is_mandatory(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    _write_json(
        paths["paired_gpu_report_path"],
        {
            "physical_gpu_uuid": "GPU-a",
            "relion_gpu_uuid": "GPU-a",
            "recovar_gpu_uuid": "GPU-b",
        },
    )
    monkeypatch.setattr(
        audit_module,
        "_load_relion_volume",
        lambda _path: np.random.default_rng(3).normal(size=(8, 8, 8)),
    )

    with pytest.raises(audit_module.AuditError, match="identical physical GPU"):
        _audit(paths)


def test_relion_command_must_match_frozen_case(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    command_path = paths["relion_dir"] / "relion_command.json"
    command = json.loads(command_path.read_text())
    command["argv"][command["argv"].index("--iter") + 1] = "7"
    _write_json(command_path, command)
    monkeypatch.setattr(
        audit_module,
        "_load_relion_volume",
        lambda _path: np.random.default_rng(4).normal(size=(8, 8, 8)),
    )

    with pytest.raises(audit_module.AuditError, match="does not match frozen nr_iter"):
        _audit(paths)


def test_fixture_manifest_digest_is_mandatory(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    materialization_path = paths["fixture_dir"] / "fixture_materialization.json"
    materialization = json.loads(materialization_path.read_text())
    materialization["manifest_sha256"] = "b" * 64
    _write_json(materialization_path, materialization)
    monkeypatch.setattr(
        audit_module,
        "_load_relion_volume",
        lambda _path: np.random.default_rng(5).normal(size=(8, 8, 8)),
    )

    with pytest.raises(audit_module.AuditError, match="manifest digest"):
        _audit(paths)


def test_extended_suite_can_freeze_a_longer_checkpoint_contract(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    scorecard = json.loads(paths["scorecard_path"].read_text())
    scorecard["schema"] = audit_module.SUITE_SCHEMA
    scorecard["acceptance_contract"]["required_checkpoints"] = [0, 1, 3]
    scorecard["cases"][0]["definition"]["nr_iter"] = 3
    _write_json(paths["scorecard_path"], scorecard)

    native_options_path = paths["recovar_dir"] / "run_native_options.json"
    native_options = json.loads(native_options_path.read_text())
    native_options["nr_iter"] = 3
    _write_json(native_options_path, native_options)
    relion_command_path = paths["relion_dir"] / "relion_command.json"
    relion_command = json.loads(relion_command_path.read_text())
    relion_command["argv"][relion_command["argv"].index("--iter") + 1] = "3"
    _write_json(relion_command_path, relion_command)
    for directory in (paths["recovar_dir"], paths["relion_dir"]):
        for path in audit_module._artifact_paths(directory, 3).values():
            path.touch()

    volume = np.random.default_rng(7).normal(size=(8, 8, 8))
    monkeypatch.setattr(audit_module, "_load_relion_volume", lambda _path: volume)

    report, _shellwise = _audit(paths)

    assert report["result"] == "pass"
    assert [row["iteration"] for row in report["checkpoints"]] == [0, 1, 3]
