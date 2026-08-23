import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "materialize_em_k1_fixture.py"
SPEC = importlib.util.spec_from_file_location("materialize_em_k1_fixture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path, *, recorded_sha256: str | None = None) -> tuple[Path, Path]:
    fixture_root = tmp_path / "fixtures"
    source_data = fixture_root / "cases" / "20_case" / "data"
    source_data.mkdir(parents=True)
    particle_file = source_data / "particles.mrcs"
    particle_file.write_bytes(b"immutable-particles")
    manifest = {
        "schema": MODULE.SCHEMA,
        "suite_id": "fixed-test-suite",
        "cases": [
            {
                "id": "k1-20",
                "name": "case",
                "source_data_dir": "cases/20_case/data",
                "files": [
                    {
                        "name": particle_file.name,
                        "size": particle_file.stat().st_size,
                        "sha256": recorded_sha256 or _sha256(particle_file),
                    }
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, fixture_root


@pytest.mark.unit
def test_materialize_verifies_and_symlinks_without_copying(tmp_path):
    manifest_path, fixture_root = _manifest(tmp_path)
    output_dir = tmp_path / "run" / "data"

    report = MODULE.materialize(
        manifest_path,
        fixture_root,
        output_dir,
        case_id="k1-20",
        case_name="case",
    )

    destination = output_dir / "particles.mrcs"
    assert destination.is_symlink()
    assert destination.read_bytes() == b"immutable-particles"
    assert report["files"][0]["sha256"] == _sha256(destination)
    recorded = json.loads((output_dir / "fixture_materialization.json").read_text())
    assert recorded == report


@pytest.mark.unit
def test_materialize_fails_closed_on_digest_mismatch(tmp_path):
    manifest_path, fixture_root = _manifest(tmp_path, recorded_sha256="0" * 64)

    with pytest.raises(MODULE.FixtureError, match="SHA-256 mismatch"):
        MODULE.materialize(
            manifest_path,
            fixture_root,
            tmp_path / "run" / "data",
            case_id="k1-20",
            case_name="case",
        )

    assert not (tmp_path / "run" / "data" / "particles.mrcs").exists()


@pytest.mark.unit
def test_materialize_rejects_path_traversal(tmp_path):
    manifest_path, fixture_root = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0]["source_data_dir"] = "../outside"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(MODULE.FixtureError, match="unsafe fixture-relative path"):
        MODULE.materialize(
            manifest_path,
            fixture_root,
            tmp_path / "run" / "data",
            case_id="k1-20",
            case_name="case",
        )


@pytest.mark.unit
def test_materialize_refuses_to_replace_existing_output(tmp_path):
    manifest_path, fixture_root = _manifest(tmp_path)
    output_dir = tmp_path / "run" / "data"
    output_dir.mkdir(parents=True)
    (output_dir / "particles.mrcs").write_bytes(b"different")

    with pytest.raises(MODULE.FixtureError, match="refusing to replace"):
        MODULE.materialize(
            manifest_path,
            fixture_root,
            output_dir,
            case_id="k1-20",
            case_name="case",
        )
