#!/usr/bin/env python3
"""Build a hash-pinned manifest for the fixed K=1 parity fixtures.

This is intentionally a separate, explicit operation from the matrix launcher:
launches may verify a reviewed manifest, but never refresh its hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "recovar.em_k1_fixture_manifest.v1"
SCORECARD_SCHEMA = "recovar.em_relion_parity_scorecard.v1"


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _case_sources(values: list[str]) -> dict[str, Path]:
    sources = {}
    for value in values:
        case_id, separator, path = value.partition("=")
        if not separator or not case_id or not path:
            raise ValueError(f"invalid --case-source {value!r}; expected CASE_ID=DATA_DIR")
        if case_id in sources:
            raise ValueError(f"duplicate --case-source for {case_id}")
        sources[case_id] = Path(path)
    return sources


def _relative_to_root(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError as exc:
        raise ValueError(f"fixture directory is outside fixture root: {resolved}") from exc


def build_manifest(
    scorecard_path: Path,
    fixture_root: Path,
    default_run_root: Path,
    output_path: Path,
    *,
    source_overrides: dict[str, Path] | None = None,
    selected_case_ids: set[str] | None = None,
) -> dict:
    scorecard_path = scorecard_path.resolve()
    fixture_root = fixture_root.resolve()
    default_run_root = default_run_root.resolve()
    source_overrides = source_overrides or {}
    scorecard = json.loads(scorecard_path.read_text())
    if scorecard.get("schema") != SCORECARD_SCHEMA:
        raise ValueError(f"unsupported scorecard schema: {scorecard.get('schema')!r}")

    scorecard_cases = scorecard.get("cases")
    if not isinstance(scorecard_cases, list) or not scorecard_cases:
        raise ValueError("scorecard cases must be a non-empty list")
    frozen_case_definitions_sha256 = scorecard.get("frozen_case_definitions_sha256")
    if not isinstance(frozen_case_definitions_sha256, str) or len(frozen_case_definitions_sha256) != 64:
        raise ValueError("scorecard must record a frozen_case_definitions_sha256 digest")
    known_ids = {case.get("id") for case in scorecard_cases}
    unknown_overrides = sorted(set(source_overrides) - known_ids)
    if unknown_overrides:
        raise ValueError(f"case-source overrides are not in the scorecard: {unknown_overrides}")
    if selected_case_ids:
        unknown_selected = sorted(selected_case_ids - known_ids)
        if unknown_selected:
            raise ValueError(f"selected cases are not in the scorecard: {unknown_selected}")

    cases = []
    for scorecard_case in scorecard_cases:
        case_id = scorecard_case["id"]
        if selected_case_ids and case_id not in selected_case_ids:
            continue
        case_index = int(case_id.split("-")[-1])
        case_name = scorecard_case["name"]
        data_dir = source_overrides.get(
            case_id,
            default_run_root / "cases" / f"{case_index}_{case_name}" / "data",
        ).resolve()
        if not data_dir.is_dir():
            raise ValueError(f"missing fixture data directory for {case_id}: {data_dir}")
        source_data_dir = _relative_to_root(data_dir, fixture_root)
        source_files = sorted(path for path in data_dir.iterdir() if path.is_file())
        if not source_files:
            raise ValueError(f"fixture data directory is empty for {case_id}: {data_dir}")
        particle_stacks = [
            path for path in source_files if path.name.startswith("particles.") and path.suffix == ".mrcs"
        ]
        if len(particle_stacks) != 1:
            raise ValueError(f"expected one particle stack for {case_id}, found {len(particle_stacks)}")
        files = []
        for path in source_files:
            files.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        cases.append(
            {
                "id": case_id,
                "name": case_name,
                "source_data_dir": source_data_dir,
                "files": files,
            }
        )

    manifest = {
        "schema": SCHEMA,
        "suite_id": f"{scorecard['suite_id']}-artifact-pinned-v2",
        "suite_version": 2,
        "frozen_denominator": scorecard["frozen_denominator"],
        "frozen_case_definitions_sha256": frozen_case_definitions_sha256,
        "source_scorecard": str(scorecard_path.relative_to(scorecard_path.parents[2])),
        "source_scorecard_sha256": sha256_file(scorecard_path),
        "fixture_root_policy": "Runtime root supplied separately; source_data_dir is root-relative.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(output_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--default-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-source", action="append", default=[], metavar="CASE_ID=DATA_DIR")
    parser.add_argument("--case", action="append", default=[], dest="selected_cases")
    args = parser.parse_args(argv)
    manifest = build_manifest(
        args.scorecard,
        args.fixture_root,
        args.default_run_root,
        args.output,
        source_overrides=_case_sources(args.case_source),
        selected_case_ids=set(args.selected_cases) or None,
    )
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(manifest["cases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
