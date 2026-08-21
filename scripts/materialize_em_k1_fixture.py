#!/usr/bin/env python3
"""Verify and materialize one immutable K=1 parity fixture.

The bulky fixture files remain in their canonical scratch root.  A run gets a
directory of read-only symlinks only after every recorded size and SHA-256
digest has been verified.  This prevents a regenerated synthetic stack from
silently replacing a fixed-suite input while keeping run-local derived files
separate from the canonical fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "recovar.em_k1_fixture_manifest.v1"


class FixtureError(RuntimeError):
    """Raised when a fixture cannot be verified or materialized safely."""


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_source(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise FixtureError(f"unsafe fixture-relative path: {relative_path!r}")
    source = (root / relative).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise FixtureError(f"fixture path escapes root: {relative_path!r}") from exc
    return source


def load_case(manifest_path: Path, *, case_id: str, case_name: str) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise FixtureError(f"unsupported fixture manifest schema: {manifest.get('schema')!r}")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise FixtureError("fixture manifest cases must be a list")
    matches = [case for case in cases if case.get("id") == case_id]
    if len(matches) != 1:
        raise FixtureError(f"expected exactly one manifest row for {case_id}, found {len(matches)}")
    case = matches[0]
    if case.get("name") != case_name:
        raise FixtureError(
            f"fixture name mismatch for {case_id}: expected {case_name!r}, found {case.get('name')!r}"
        )
    files = case.get("files")
    if not isinstance(files, list) or not files:
        raise FixtureError(f"{case_id}: files must be a non-empty list")
    names = [entry.get("name") for entry in files]
    if len(set(names)) != len(names) or not all(isinstance(name, str) and name for name in names):
        raise FixtureError(f"{case_id}: fixture file names must be non-empty and unique")
    return manifest, case


def materialize(
    manifest_path: Path,
    fixture_root: Path,
    output_dir: Path,
    *,
    case_id: str,
    case_name: str,
) -> dict:
    manifest_path = manifest_path.resolve()
    fixture_root = fixture_root.resolve()
    output_dir = output_dir.resolve()
    if not manifest_path.is_file():
        raise FixtureError(f"missing fixture manifest: {manifest_path}")
    if not fixture_root.is_dir():
        raise FixtureError(f"missing fixture root: {fixture_root}")
    manifest, case = load_case(manifest_path, case_id=case_id, case_name=case_name)
    source_data_dir = _safe_source(fixture_root, str(case.get("source_data_dir", "")))
    if not source_data_dir.is_dir():
        raise FixtureError(f"missing canonical fixture directory: {source_data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    verified_files = []
    for entry in case["files"]:
        name = entry.get("name")
        expected_size = entry.get("size")
        expected_sha256 = entry.get("sha256")
        if Path(name).name != name:
            raise FixtureError(f"{case_id}: fixture file name must be a basename: {name!r}")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise FixtureError(f"{case_id}/{name}: invalid size")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise FixtureError(f"{case_id}/{name}: invalid SHA-256")
        source = _safe_source(source_data_dir, name)
        if not source.is_file():
            raise FixtureError(f"missing canonical fixture file: {source}")
        actual_size = source.stat().st_size
        if actual_size != expected_size:
            raise FixtureError(
                f"{case_id}/{name}: size mismatch expected={expected_size} actual={actual_size}"
            )
        actual_sha256 = sha256_file(source)
        if actual_sha256 != expected_sha256:
            raise FixtureError(
                f"{case_id}/{name}: SHA-256 mismatch expected={expected_sha256} actual={actual_sha256}"
            )

        verified_files.append(
            {
                "name": name,
                "size": actual_size,
                "sha256": actual_sha256,
                "source": str(source),
                "destination": str(output_dir / name),
            }
        )

    # Do not expose a partially verified fixture. Validate every source digest
    # and every existing destination before creating any new symlink.
    for verified in verified_files:
        destination = Path(verified["destination"])
        source = Path(verified["source"])
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or destination.resolve() != source:
                raise FixtureError(f"refusing to replace existing noncanonical output: {destination}")
    for verified in verified_files:
        destination = Path(verified["destination"])
        if not destination.is_symlink():
            destination.symlink_to(Path(verified["source"]))

    report = {
        "schema": "recovar.em_k1_fixture_materialization.v1",
        "suite_id": manifest.get("suite_id"),
        "case_id": case_id,
        "case_name": case_name,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "fixture_root": str(fixture_root),
        "source_data_dir": str(source_data_dir),
        "output_dir": str(output_dir),
        "files": verified_files,
    }
    report_path = output_dir / "fixture_materialization.json"
    temporary_path = output_dir / f".{report_path.name}.{os.getpid()}.tmp"
    temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = materialize(
        args.manifest,
        args.fixture_root,
        args.output_dir,
        case_id=args.case_id,
        case_name=args.case_name,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
