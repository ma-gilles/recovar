#!/usr/bin/env python3
"""Freeze STAR particle identities and source-stack provenance for BPref replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import starfile


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _particle_table(star_path: Path):
    tables = starfile.read(star_path)
    if not isinstance(tables, dict):
        if "rlnImageName" not in tables.columns:
            raise ValueError(f"{star_path} has no rlnImageName column")
        return tables
    candidates = [table for table in tables.values() if "rlnImageName" in table.columns]
    if len(candidates) != 1:
        raise ValueError(
            f"{star_path} must contain exactly one particle table with rlnImageName; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def prepare_metadata(*, star_path: Path, image_names_path: Path, manifest_path: Path) -> dict:
    star_path = star_path.expanduser().resolve()
    image_names_path = image_names_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    table = _particle_table(star_path)
    resolved_identities = []
    stack_paths = set()
    for raw_identity in table["rlnImageName"].astype(str).tolist():
        stack_index, separator, raw_stack_path = raw_identity.partition("@")
        if not separator or not stack_index.isdigit() or int(stack_index) <= 0:
            raise ValueError(f"invalid rlnImageName identity: {raw_identity!r}")
        stack_path = Path(raw_stack_path).expanduser()
        if not stack_path.is_absolute():
            stack_path = star_path.parent / stack_path
        stack_path = stack_path.resolve()
        if not stack_path.is_file():
            raise FileNotFoundError(stack_path)
        stack_paths.add(stack_path)
        resolved_identities.append(f"{int(stack_index)}@{stack_path}")
    if len(stack_paths) != 1:
        raise ValueError(
            "BPref contribution schema v3 requires one frozen source stack; "
            f"found {len(stack_paths)}"
        )
    stack_path = next(iter(stack_paths))
    stack_sha256 = _sha256_file(stack_path)
    image_names_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(image_names_path, np.asarray(resolved_identities), allow_pickle=False)
    manifest = {
        "schema": "recovar-bpref-contribution-metadata-v1",
        "star_path": str(star_path),
        "image_names_npy": str(image_names_path),
        "particle_count": len(resolved_identities),
        "source_stack_path": str(stack_path),
        "source_stack_sha256": stack_sha256,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--star", type=Path, required=True)
    parser.add_argument("--image-names-npy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_metadata(
        star_path=args.star,
        image_names_path=args.image_names_npy,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
