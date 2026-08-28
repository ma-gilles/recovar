from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_vdam_candidate_native_envelope as audit_module
from scripts.audit_vdam_candidate_native_envelope import (
    CandidateEnvelopeError,
    audit_candidate_envelope,
    classify_candidate_checkpoint,
)


def test_candidate_checkpoint_accepts_any_native_mode_and_best_native_gt():
    result = classify_candidate_checkpoint(
        candidate_native_fsc_auc=[0.991, 0.9995, 0.992],
        candidate_gt_fsc_auc=0.42,
        native_gt_fsc_auc=[0.418, 0.421, 0.419],
        cross_engine_min=0.999,
        gt_delta_min=-0.002,
    )

    assert result["pass"] is True
    assert result["candidate_best_native_fsc_auc"] == pytest.approx(0.9995)
    assert result["candidate_minus_best_native_gt_fsc_auc"] == pytest.approx(-0.001)


def test_candidate_checkpoint_rejects_missing_native_mode():
    result = classify_candidate_checkpoint(
        candidate_native_fsc_auc=[0.9989, 0.997],
        candidate_gt_fsc_auc=0.5,
        native_gt_fsc_auc=[0.49, 0.48],
        cross_engine_min=0.999,
        gt_delta_min=-0.002,
    )

    assert result["pass"] is False
    assert result["checks"]["candidate_matches_a_native_mode_at_frozen_gate"] is False


def test_candidate_checkpoint_uses_best_native_as_conservative_gt_control():
    result = classify_candidate_checkpoint(
        candidate_native_fsc_auc=[1.0, 0.99],
        candidate_gt_fsc_auc=0.5,
        native_gt_fsc_auc=[0.499, 0.503],
        cross_engine_min=0.999,
        gt_delta_min=-0.002,
    )

    assert result["pass"] is False
    assert result["candidate_minus_best_native_gt_fsc_auc"] == pytest.approx(-0.003)
    assert result["checks"]["candidate_meets_gt_nondegradation_vs_best_native"] is False


def test_candidate_checkpoint_rejects_misaligned_native_metrics():
    with pytest.raises(CandidateEnvelopeError, match="must align"):
        classify_candidate_checkpoint(
            candidate_native_fsc_auc=[1.0, 0.99],
            candidate_gt_fsc_auc=0.5,
            native_gt_fsc_auc=[0.5],
            cross_engine_min=0.999,
            gt_delta_min=-0.002,
        )


def _touch_artifacts(root: Path, engine: str) -> None:
    directory = root / engine
    directory.mkdir(parents=True)
    for iteration in (0, 1):
        for suffix in audit_module.ARTIFACT_SUFFIXES:
            (directory / f"run_it{iteration:03d}_{suffix}").touch()


def _native_root(root: Path, index: int) -> Path:
    native = root / f"native-{index}"
    _touch_artifacts(native, "relion")
    (native / "trajectory_audit.json").write_text(
        json.dumps(
            {
                "schema": audit_module.TRAJECTORY_SCHEMA,
                "suite_id": "suite",
                "case_id": "case",
                "artifact_topology_exact": True,
                "checkpoints": [{"iteration": 0}, {"iteration": 1}],
            }
        )
    )
    (native / "run_provenance.json").write_text(
        json.dumps(
            {
                "git_head": "a" * 40,
                "relion_reference": {"executable_sha256": "b" * 64},
            }
        )
    )
    (native / "paired_gpu_uuid.json").write_text(
        json.dumps(
            {
                "physical_gpu_uuid": "GPU-acde",
                "relion_gpu_uuid": "GPU-acde",
                "recovar_gpu_uuid": "GPU-acde",
            }
        )
    )
    return native


def test_candidate_envelope_audit_is_provenance_complete_and_fail_closed(tmp_path, monkeypatch):
    scorecard = tmp_path / "scorecard.json"
    scorecard.write_text(
        json.dumps(
            {
                "schema": audit_module.SUITE_SCHEMA,
                "suite_id": "suite",
                "source_fixture_manifest": {"sha256": "fixture-sha"},
                "acceptance_contract": {
                    "required_checkpoints": [0, 1],
                    "cross_engine_fsc_auc_min": 0.999,
                    "recovar_minus_relion_gt_fsc_auc_min": -0.002,
                },
                "cases": [
                    {
                        "id": "case",
                        "name": "synthetic",
                        "definition": {"source_em_case_id": "k1", "nr_iter": 1},
                    }
                ],
            }
        )
    )
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "fixture_materialization.json").write_text(
        json.dumps({"case_id": "k1", "manifest_sha256": "fixture-sha"})
    )
    (fixture / "reference_gt_relion.mrc").touch()
    candidate = tmp_path / "candidate"
    _touch_artifacts(candidate, "recovar")
    provenance = candidate / "provenance"
    provenance.mkdir()
    (provenance / "repo_head.txt").write_text("c" * 40 + "\n")
    (provenance / "input_sha256.txt").write_text(
        f"{'d' * 64}  /immutable/libcuda_backproject.so\n"
    )
    (provenance / "nvidia_smi.txt").write_text("GPU UUID : GPU-acde\n")
    native_roots = [_native_root(tmp_path, 1), _native_root(tmp_path, 2)]

    monkeypatch.setattr(audit_module, "_load_relion_volume", lambda path: np.zeros((2, 2, 2)))

    def fake_metric(lhs, rhs, *, key, shellwise):
        shellwise[key] = np.asarray([1.0, 1.0])
        if "candidate_native" in key:
            return 0.9995
        if "candidate_gt" in key:
            return 0.5
        if key.endswith("_gt"):
            return 0.501
        return 0.997

    monkeypatch.setattr(audit_module, "_metric", fake_metric)
    report, shells = audit_candidate_envelope(
        scorecard_path=scorecard,
        case_id="case",
        candidate_root=candidate,
        native_roots=native_roots,
        fixture_dir=fixture,
    )

    assert report["result"] == "pass"
    assert report["candidate_provenance"]["source_head"] == "c" * 40
    assert report["native_panel_provenance"]["repeat_count"] == 2
    assert report["minimum_candidate_best_native_fsc_auc"] == pytest.approx(0.9995)
    assert report["minimum_candidate_minus_best_native_gt_fsc_auc"] == pytest.approx(-0.001)
    assert len(shells) == 12


def test_candidate_envelope_rejects_cross_gpu_evidence():
    candidate = {"physical_gpu_uuid": "GPU-aaaa"}
    native = {"physical_gpu_uuid": "GPU-bbbb"}
    with pytest.raises(CandidateEnvelopeError, match="physical GPU differs"):
        audit_module.require_same_physical_gpu(candidate, native)


def test_candidate_provenance_rejects_ambiguous_cuda_digest(tmp_path):
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "repo_head.txt").write_text("a" * 40)
    (provenance / "input_sha256.txt").write_text(
        f"{'b' * 64}  /one/libcuda_backproject.so\n{'c' * 64}  /two/libcuda_backproject.so\n"
    )
    (provenance / "nvidia_smi.txt").write_text("GPU UUID : GPU-one\n")

    with pytest.raises(CandidateEnvelopeError, match="exactly one CUDA"):
        audit_module._candidate_provenance(tmp_path)


def test_candidate_provenance_accepts_current_paired_run_format(tmp_path):
    (tmp_path / "run_provenance.json").write_text(
        json.dumps(
            {
                "git_head": "a" * 40,
                "recovar_native_extensions": {
                    "cuda_backproject": {"sha256": "b" * 64}
                },
            }
        )
    )
    (tmp_path / "paired_gpu_uuid.json").write_text(
        json.dumps(
            {
                "physical_gpu_uuid": "GPU-one",
                "relion_gpu_uuid": "GPU-one",
                "recovar_gpu_uuid": "GPU-one",
            }
        )
    )

    report = audit_module._candidate_provenance(tmp_path)

    assert report == {
        "source_head": "a" * 40,
        "cuda_library_sha256": "b" * 64,
        "physical_gpu_uuid": "GPU-one",
        "source_format": "paired_run_provenance.v1",
    }


def test_candidate_provenance_rejects_mixed_paired_gpu_report(tmp_path):
    (tmp_path / "run_provenance.json").write_text(
        json.dumps(
            {
                "git_head": "a" * 40,
                "recovar_native_extensions": {
                    "cuda_backproject": {"sha256": "b" * 64}
                },
            }
        )
    )
    (tmp_path / "paired_gpu_uuid.json").write_text(
        json.dumps(
            {
                "physical_gpu_uuid": "GPU-one",
                "relion_gpu_uuid": "GPU-one",
                "recovar_gpu_uuid": "GPU-two",
            }
        )
    )

    with pytest.raises(CandidateEnvelopeError, match="paired GPU identity"):
        audit_module._candidate_provenance(tmp_path)
