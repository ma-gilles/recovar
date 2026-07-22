import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build_em_k1_fixture_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_em_k1_fixture_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scorecard(path: Path) -> Path:
    scorecard = {
        "schema": MODULE.SCORECARD_SCHEMA,
        "suite_id": "fixed-suite",
        "suite_version": 1,
        "frozen_denominator": 1,
        "frozen_case_definitions_sha256": "a" * 64,
        "cases": [{"id": "k1-01", "name": "case_one"}],
    }
    path.write_text(json.dumps(scorecard))
    return path


@pytest.mark.unit
def test_build_manifest_hashes_every_fixture_file(tmp_path):
    root = tmp_path / "fixture-root"
    data = root / "run" / "cases" / "1_case_one" / "data"
    data.mkdir(parents=True)
    particle = data / "particles.128.mrcs"
    particle.write_bytes(b"particles")
    metadata = data / "particles.star"
    metadata.write_bytes(b"metadata")
    scorecard = _scorecard(tmp_path / "scorecard.json")
    output = tmp_path / "manifest.json"

    manifest = MODULE.build_manifest(scorecard, root, root / "run", output)

    assert output.exists()
    case = manifest["cases"][0]
    assert manifest["frozen_case_definitions_sha256"] == "a" * 64
    assert case["source_data_dir"] == "run/cases/1_case_one/data"
    assert case["files"] == [
        {"name": "particles.128.mrcs", "size": 9, "sha256": _sha256(particle)},
        {"name": "particles.star", "size": 8, "sha256": _sha256(metadata)},
    ]


@pytest.mark.unit
def test_build_manifest_rejects_source_outside_fixture_root(tmp_path):
    root = tmp_path / "fixture-root"
    root.mkdir()
    data = tmp_path / "outside"
    data.mkdir()
    (data / "particles.128.mrcs").write_bytes(b"particles")
    scorecard = _scorecard(tmp_path / "scorecard.json")

    with pytest.raises(ValueError, match="outside fixture root"):
        MODULE.build_manifest(
            scorecard,
            root,
            root / "unused",
            tmp_path / "manifest.json",
            source_overrides={"k1-01": data},
        )
