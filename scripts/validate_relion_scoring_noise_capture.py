#!/usr/bin/env python3
"""Validate passive RELION scoring-noise captures and their float32 inverse."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"RLNSIGMAV1"
MAGIC_SIZE = 16
HEADER_WORDS = 16
HEADER_SIZE = MAGIC_SIZE + HEADER_WORDS * np.dtype("<u8").itemsize


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ScoringNoiseCapture:
    path: Path
    iteration: int
    rank: int
    optics_group_zero_based: int
    sigma2_fudge: np.float32
    sigma2: np.ndarray
    inverse_sigma2_f32: np.ndarray


def load_capture(path: Path) -> ScoringNoiseCapture:
    path = path.resolve()
    payload = path.read_bytes()
    _require(len(payload) >= HEADER_SIZE, f"{path}: capture is truncated")
    _require(payload[: len(MAGIC)] == MAGIC, f"{path}: magic changed")
    _require(
        payload[len(MAGIC) : MAGIC_SIZE] == bytes(MAGIC_SIZE - len(MAGIC)),
        f"{path}: magic padding is not zero",
    )
    header = np.frombuffer(
        payload,
        dtype="<u8",
        count=HEADER_WORDS,
        offset=MAGIC_SIZE,
    )
    _require(int(header[0]) == 1, f"{path}: unsupported schema version {header[0]}")
    _require(int(header[4]) > 0, f"{path}: shell count must be positive")
    _require(int(header[5]) == 8, f"{path}: RELION RFLOAT is not double")
    _require(int(header[6]) == 8, f"{path}: captured double size changed")
    _require(int(header[7]) == 4, f"{path}: captured float size changed")
    _require(np.all(header[9:] == 0), f"{path}: reserved header words are nonzero")

    shell_count = int(header[4])
    expected_size = HEADER_SIZE + shell_count * (np.dtype("<f8").itemsize + np.dtype("<f4").itemsize)
    _require(len(payload) == expected_size, f"{path}: capture size changed")
    sigma2 = np.frombuffer(
        payload,
        dtype="<f8",
        count=shell_count,
        offset=HEADER_SIZE,
    ).copy()
    inverse_sigma2_f32 = np.frombuffer(
        payload,
        dtype="<f4",
        count=shell_count,
        offset=HEADER_SIZE + shell_count * np.dtype("<f8").itemsize,
    ).copy()
    _require(np.all(np.isfinite(sigma2) & (sigma2 > 0)), f"{path}: sigma2 must be finite and positive")
    _require(np.all(np.isfinite(inverse_sigma2_f32)), f"{path}: inverse sigma2 must be finite")
    fudge_bits = np.asarray([header[8]], dtype="<u8").astype("<u4", copy=False)[0]
    sigma2_fudge = np.asarray([fudge_bits], dtype="<u4").view("<f4")[0]
    _require(
        np.isfinite(sigma2_fudge) and sigma2_fudge > 0,
        f"{path}: sigma2 fudge must be finite and positive",
    )
    expected_inverse = np.asarray(
        np.float64(1.0) / (np.float64(sigma2_fudge) * sigma2),
        dtype=np.float32,
    )
    _require(
        np.array_equal(expected_inverse, inverse_sigma2_f32),
        f"{path}: captured inverse sigma2 does not replay exactly",
    )
    return ScoringNoiseCapture(
        path=path,
        iteration=int(header[1]),
        rank=int(header[2]),
        optics_group_zero_based=int(header[3]),
        sigma2_fudge=sigma2_fudge,
        sigma2=sigma2,
        inverse_sigma2_f32=inverse_sigma2_f32,
    )


def validate_capture(capture: ScoringNoiseCapture) -> dict[str, object]:
    return {
        "schema": "relion.scoring_noise_capture.v1",
        "status": "accepted",
        "path": str(capture.path),
        "sha256": _sha256(capture.path),
        "iteration": capture.iteration,
        "rank": capture.rank,
        "optics_group_zero_based": capture.optics_group_zero_based,
        "shell_count": int(capture.sigma2.size),
        "sigma2_fudge_float32": float(capture.sigma2_fudge),
        "sigma2_fudge_bits": f"0x{capture.sigma2_fudge.view(np.uint32).item():08x}",
        "inverse_sigma2_replay_exact": True,
        "sigma2_min": float(np.min(capture.sigma2)),
        "sigma2_max": float(np.max(capture.sigma2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [validate_capture(load_capture(path)) for path in args.capture]
    ranks = [(row["iteration"], row["rank"], row["optics_group_zero_based"]) for row in reports]
    _require(len(set(ranks)) == len(ranks), "capture iteration/rank/optics identities are not unique")
    report = {
        "schema": "relion.scoring_noise_capture_set.v1",
        "status": "accepted",
        "capture_count": len(reports),
        "captures": reports,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
