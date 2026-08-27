#!/usr/bin/env python3
"""Seal a captured process-resident K=1 reference for continuation replay."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_boundary(source: Path, output: Path, expected_size: int) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite boundary: {output}")
    with np.load(source, allow_pickle=False) as payload:
        required = {"iteration", "half", "stage", "value_fourier"}
        if not required <= set(payload.files):
            raise ValueError(f"lifecycle capture lacks {sorted(required - set(payload.files))}")
        iteration = int(payload["iteration"])
        half = int(payload["half"])
        stage = str(payload["stage"])
        volume = np.asarray(payload["value_fourier"])
    if iteration != 1 or half not in (1, 2) or stage != "post_mask":
        raise ValueError(
            f"expected iteration-1 half-1/2 post_mask capture, got "
            f"iteration={iteration} half={half} stage={stage!r}"
        )
    expected_elements = expected_size**3
    if volume.size != expected_elements:
        raise ValueError(f"reference has {volume.size} elements, expected {expected_elements}")
    if not np.issubdtype(volume.dtype, np.complexfloating) or not np.isfinite(volume).all():
        raise ValueError("reference must contain finite complex Fourier values")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        mean_vol_ft=volume.reshape(-1),
        source_path=np.asarray(str(source)),
        source_sha256=np.asarray(_sha256(source)),
        source_iteration=np.asarray(iteration, dtype=np.int32),
        source_half=np.asarray(half, dtype=np.int32),
        source_stage=np.asarray(stage),
        volume_size=np.asarray(expected_size, dtype=np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-size", required=True, type=int)
    args = parser.parse_args()
    if args.expected_size <= 0:
        parser.error("--expected-size must be positive")
    seal_boundary(args.source, args.output, args.expected_size)
    print(f"sealed {args.output.resolve()} sha256={_sha256(args.output.resolve())}")


if __name__ == "__main__":
    main()
