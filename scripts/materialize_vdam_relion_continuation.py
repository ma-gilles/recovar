#!/usr/bin/env python3
"""Materialize a RELION continuation optimiser with an explicit sampling STAR."""

from __future__ import annotations

import argparse
from pathlib import Path


def materialize(source: Path, sampling: Path, output: Path) -> None:
    source = source.resolve(strict=True)
    sampling = sampling.resolve(strict=True)
    lines = source.read_text().splitlines()
    matches = [
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("_rlnOrientSamplingStarFile")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one sampling-file row in {source}")
    index = matches[0]
    label = lines[index].split(maxsplit=1)[0]
    lines[index] = f"{label:<55} {sampling}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-optimiser", type=Path, required=True)
    parser.add_argument("--sampling-star", type=Path, required=True)
    parser.add_argument("--output-optimiser", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source_optimiser, args.sampling_star, args.output_optimiser)


if __name__ == "__main__":
    main()
