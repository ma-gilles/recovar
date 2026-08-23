from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_vdam_kclass_repeatability as repeatability


def _write_pair_report(root: Path, *, gpu_uuid: str = "GPU-repeat") -> None:
    root.mkdir(parents=True)
    report = {
        "schema": "recovar.vdam_kclass_pair.v1",
        "git_head": "a" * 40,
        "git_dirty": False,
        "physical_gpu_uuid": gpu_uuid,
        "fixture_dir": "/fixture",
        "fixture_sha256": {"particles.star": "b" * 64},
        "relion_sha256": "c" * 64,
        "audit": {
            "K": 4,
            "checkpoints": [0, 1, 2],
            "result": "pass",
            "thresholds": {
                "minimum_per_class_fsc_auc": 0.999,
                "minimum_class_assignment_accuracy": 0.995,
            },
        },
    }
    (root / "pair_report.json").write_text(json.dumps(report))


def test_repeatability_audits_both_native_engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    roots = (tmp_path / "repeat-1", tmp_path / "repeat-2")
    for root in roots:
        _write_pair_report(root)
    calls = []

    def fake_audit_trajectory(**kwargs):
        calls.append(kwargs)
        return {"result": "pass"}, {"curve": np.ones(3)}

    monkeypatch.setattr(repeatability, "audit_trajectory", fake_audit_trajectory)
    report, shellwise = repeatability.audit_repeatability(pair_roots=roots)

    assert report["result"] == "pass"
    assert calls[0]["candidate_dir"] == roots[1] / "recovar"
    assert calls[0]["reference_dir"] == roots[0] / "recovar"
    assert calls[1]["candidate_dir"] == roots[1] / "relion"
    assert calls[1]["reference_dir"] == roots[0] / "relion"
    assert set(shellwise) == {"recovar_repeat_curve", "relion_repeat_curve"}


def test_repeatability_rejects_cross_gpu_panel(tmp_path: Path):
    roots = (tmp_path / "repeat-1", tmp_path / "repeat-2")
    _write_pair_report(roots[0], gpu_uuid="GPU-first")
    _write_pair_report(roots[1], gpu_uuid="GPU-second")

    with pytest.raises(repeatability.RepeatabilityError, match="contracts, source, GPU"):
        repeatability._validated_contract(roots)
