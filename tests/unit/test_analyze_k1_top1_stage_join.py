import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_k1_top1_stage_join as joiner


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    capture = tmp_path / "pass2_orig031559_cs100.npz"
    np.savez(
        capture,
        original_index=np.asarray(31559, dtype=np.int64),
        current_size=np.asarray(100, dtype=np.int64),
        iteration=np.asarray(2, dtype=np.int64),
    )
    (tmp_path / "part_stack31560.fine-score-v1.bin").write_bytes(b"fine")
    (tmp_path / "part_stack31560.bpre-v2.bin").write_bytes(b"factor")
    manifest = {
        "schema": joiner.MANIFEST_SCHEMA,
        "status": "preregistered_waiting_for_captures",
        "target": {
            "source_row_zero_based": 31559,
            "stack_index_one_based": 31560,
            "current_size": 100,
            "relion_iteration": 2,
            "relion_significant_count": 4,
            "recovar_significant_count": 5,
        },
        "expected_outputs": {
            "recovar_pass2": str(capture),
            "native_fine_score_glob": str(tmp_path / "*stack31560*.fine-score-v1.bin"),
            "native_factor_glob": str(tmp_path / "*stack31560*.bpre-v2.bin"),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, capture


def test_join_validates_identity_and_wraps_stage_report(tmp_path: Path, monkeypatch) -> None:
    manifest, capture = _fixture(tmp_path)
    seen = {}

    def fake_analyze(**kwargs):
        seen.update(kwargs)
        return {
            "stack_index_one_based": 31560,
            "native_significant_count": 20,
            "recovar_significant_count": 20,
            "first_exact_unequal_boundary": "preprior_score_centered",
        }

    monkeypatch.setattr(joiner, "analyze", fake_analyze)
    report = joiner.run_join(manifest_path=manifest, physical_image_size=256, top_count=64)

    assert report["status"] == "complete"
    assert report["stage_analysis"]["first_exact_unequal_boundary"] == "preprior_score_centered"
    assert seen["recovar_capture"] == capture
    assert seen["physical_image_size"] == 256
    assert seen["top_count"] == 64
    assert report["manifest_sha256"] == joiner._sha256(manifest)
    semantics = report["significant_count_semantics"]
    assert semantics["metadata"]["native_relion"] == 4
    assert semantics["metadata"]["recovar"] == 5
    assert semantics["captured_fine_support"]["native_relion"] == 20
    assert semantics["captured_fine_support"]["recovar"] == 20
    assert semantics["cross_semantic_equality_asserted"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("original_index", 1, "source row changed"),
        ("current_size", 98, "current size changed"),
        ("iteration", 3, "physical iteration changed"),
    ],
)
def test_join_rejects_recovar_boundary_drift(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    manifest, capture = _fixture(tmp_path)
    values = {"original_index": 31559, "current_size": 100, "iteration": 2}
    values[field] = value
    np.savez(capture, **{name: np.asarray(item, dtype=np.int64) for name, item in values.items()})

    with pytest.raises(ValueError, match=message):
        joiner.run_join(manifest_path=manifest, physical_image_size=256, top_count=64)


def test_join_rejects_ambiguous_native_capture(tmp_path: Path) -> None:
    manifest, _ = _fixture(tmp_path)
    (tmp_path / "other_stack31560.fine-score-v1.bin").write_bytes(b"duplicate")

    with pytest.raises(ValueError, match="exactly one native fine-score capture"):
        joiner.run_join(manifest_path=manifest, physical_image_size=256, top_count=64)
