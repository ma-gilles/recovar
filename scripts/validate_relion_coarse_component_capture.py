#!/usr/bin/env python3
"""Fail-closed validation for paired RELION coarse likelihood components."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts import validate_relion_coarse_score_capture as score_validator

HEADER_MAGIC = b"RLNCRCP1HEADER".ljust(16, b"\0")
FOOTER_MAGIC = b"RLNCRCP1FOOTER".ljust(16, b"\0")
HEADER_STRUCT = struct.Struct("<16s64Q")
FOOTER_STRUCT = struct.Struct("<16s2Q")
CANDIDATE_DTYPE = np.dtype(
    {
        "names": ("flat_index", "reference_norm", "cross_term"),
        "formats": ("<u8", "<f8", "<f8"),
        "offsets": (0, 8, 16),
        "itemsize": 24,
    }
)
FILE_NAME = re.compile(
    r"part(?P<part>\d+)_stack(?P<stack>\d+)\.coarse-components-v1\.bin"
)


@dataclass(frozen=True)
class CoarseComponentCapture:
    path: Path
    sha256: str
    header: tuple[int, ...]
    candidates: np.ndarray

    @property
    def stack_index(self) -> int:
        return self.header[6]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_coarse_component_capture(path: Path) -> CoarseComponentCapture:
    """Load one component artifact and verify its self-contained contract."""

    path = Path(path)
    match = FILE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected coarse-component file name: {path.name}")
    payload = path.read_bytes()
    _require(
        len(payload) >= HEADER_STRUCT.size + FOOTER_STRUCT.size,
        f"truncated coarse-component capture: {path}",
    )
    magic, *raw_header = HEADER_STRUCT.unpack_from(payload)
    header = tuple(int(value) for value in raw_header)
    _require(magic == HEADER_MAGIC, f"coarse-component header magic mismatch: {path}")
    _require(
        header[:4]
        == (1, HEADER_STRUCT.size, CANDIDATE_DTYPE.itemsize, FOOTER_STRUCT.size),
        f"coarse-component schema/record sizes changed: {path}",
    )
    candidate_count = header[14]
    expected_bytes = (
        HEADER_STRUCT.size
        + candidate_count * CANDIDATE_DTYPE.itemsize
        + FOOTER_STRUCT.size
    )
    _require(
        len(payload) == expected_bytes,
        f"coarse-component byte count mismatch: {path}",
    )
    _require(
        header[19] == expected_bytes,
        f"coarse-component byte estimate changed: {path}",
    )
    candidates = np.frombuffer(
        payload,
        dtype=CANDIDATE_DTYPE,
        count=candidate_count,
        offset=HEADER_STRUCT.size,
    ).copy()
    footer = FOOTER_STRUCT.unpack_from(
        payload, HEADER_STRUCT.size + candidate_count * CANDIDATE_DTYPE.itemsize
    )
    _require(footer[0] == FOOTER_MAGIC, f"coarse-component footer magic mismatch: {path}")
    finite_count = int(
        np.count_nonzero(
            np.isfinite(candidates["reference_norm"])
            & np.isfinite(candidates["cross_term"])
        )
    )
    _require(
        tuple(int(value) for value in footer[1:])
        == (candidate_count, finite_count),
        f"coarse-component footer counts changed: {path}",
    )
    _require(
        finite_count == candidate_count,
        f"coarse-component capture contains non-finite candidates: {path}",
    )
    assert match is not None
    _require(
        int(match["part"]) == header[5],
        f"coarse-component part identity mismatch: {path}",
    )
    _require(
        int(match["stack"]) == header[6],
        f"coarse-component stack identity mismatch: {path}",
    )
    class_count, nr_dir, nr_psi, nr_trans = header[10:14]
    _require(
        header[4] > 0
        and class_count > 0
        and nr_dir > 0
        and nr_psi > 0
        and nr_trans > 0,
        f"invalid coarse-component runtime dimensions: {path}",
    )
    _require(
        candidate_count == class_count * nr_dir * nr_psi * nr_trans,
        f"coarse-component topology changed: {path}",
    )
    _require(
        np.array_equal(
            candidates["flat_index"], np.arange(candidate_count, dtype=np.uint64)
        ),
        f"coarse-component flattened order changed: {path}",
    )
    _require(
        header[15] and header[16] and header[17] and header[23],
        f"coarse-component identity hash is zero: {path}",
    )
    _require(header[18] > 0 and header[20] > 0, f"invalid coarse-component cap: {path}")
    _require(
        header[21] == 8 and header[24:27] == (1, 1, 1),
        f"coarse-component capture path/layout changed: {path}",
    )
    return CoarseComponentCapture(
        path=path,
        sha256=_sha256(path),
        header=header,
        candidates=candidates,
    )


def _pair_metrics(
    component: CoarseComponentCapture,
    score: score_validator.CoarseScoreCapture,
) -> dict[str, float | int]:
    active = (score.candidates["flags"] & score_validator.ACTIVE) != 0
    _require(np.any(active), "paired coarse-score artifact has no active candidates")
    reference = component.candidates["reference_norm"]
    cross = component.candidates["cross_term"]
    raw = score.candidates["raw_diff2"].astype(np.float64)
    image_term = raw[active] - reference[active] - cross[active]
    image_constant = float(np.median(image_term))
    centered = image_term - image_constant
    nr_trans = component.header[13]
    translation_spread = np.ptp(reference.reshape(-1, nr_trans), axis=1)
    return {
        "active_candidate_count": int(np.count_nonzero(active)),
        "image_constant_median": image_constant,
        "centered_replay_p95_abs": float(np.percentile(np.abs(centered), 95)),
        "centered_replay_max_abs": float(np.max(np.abs(centered))),
        "reference_norm_translation_spread_max": float(
            np.max(translation_spread, initial=0.0)
        ),
    }


def _validate_pair(
    component: CoarseComponentCapture,
    score: score_validator.CoarseScoreCapture,
) -> None:
    component_header = component.header
    score_header = score.header
    _require(
        component_header[4:15] == score_header[4:15],
        "coarse component/score runtime identities differ",
    )
    for component_field, score_field, label in (
        (15, 28, "stack-name hash"),
        (16, 29, "image-name hash"),
        (17, 30, "selected-stack hash"),
        (18, 31, "expected-particle count"),
        (20, 33, "byte cap"),
        (22, 36, "body identity"),
        (23, 37, "current-image hash"),
    ):
        _require(
            component_header[component_field] == score_header[score_field],
            f"coarse component/score {label} differs",
        )
    _require(
        np.array_equal(
            component.candidates["flat_index"], score.candidates["flat_index"]
        ),
        "coarse component/score flattened orders differ",
    )
    bytes_per_particle = component_header[19] + score_header[32]
    _require(
        bytes_per_particle * component_header[18] <= component_header[20],
        "paired coarse component/score byte cap exceeded",
    )


def _parse_stacks(text: str) -> tuple[int, ...]:
    values = tuple(int(token) for token in text.split(",") if token)
    _require(
        values and len(values) == len(set(values)) and min(values) > 0,
        "invalid expected stacks",
    )
    return values


def validate_directory(
    directory: Path,
    expected_stacks: tuple[int, ...],
    *,
    expected_iteration: int | None = None,
) -> dict[str, object]:
    """Validate a complete identity-paired score/component panel."""

    directory = Path(directory)
    paths = sorted(directory.glob("*.coarse-components-v1.bin"))
    _require(
        not list(directory.glob("*.tmp.*")),
        "coarse-component directory contains temporary files",
    )
    _require(
        len(paths) == len(expected_stacks),
        "coarse-component file count differs from target set",
    )
    components = tuple(load_coarse_component_capture(path) for path in paths)
    _require(
        {capture.stack_index for capture in components} == set(expected_stacks),
        "coarse-component stack set is incomplete or duplicated",
    )
    selected_hash = score_validator.fnv1a64(
        ",".join(str(stack) for stack in expected_stacks)
    )
    pairs = []
    for component in components:
        score_path = component.path.with_name(
            component.path.name.replace(
                ".coarse-components-v1.bin", ".coarse-score-v1.bin"
            )
        )
        _require(score_path.is_file(), f"paired coarse-score artifact is missing: {score_path}")
        score = score_validator.load_coarse_score_capture(score_path)
        _validate_pair(component, score)
        _require(
            component.header[17] == selected_hash,
            "coarse-component selected-stack hash changed",
        )
        pairs.append((component, score, _pair_metrics(component, score)))
    iterations = {component.header[4] for component in components}
    _require(len(iterations) == 1, "coarse-component iteration changed across panel")
    iteration = next(iter(iterations))
    if expected_iteration is not None:
        _require(
            iteration == expected_iteration,
            "coarse-component iteration differs from expectation",
        )
    return {
        "schema": "relion-coarse-component-validation-v1",
        "capture_ready": True,
        "directory": str(directory.resolve()),
        "iteration": iteration,
        "particle_count": len(pairs),
        "candidate_count": int(
            sum(component.candidates.size for component, _, _ in pairs)
        ),
        "notes": [
            "Production diff2 remains binary32; reference/cross diagnostics accumulate in binary64.",
            "Centered replay is diagnostic and is not an exact-closure acceptance gate.",
        ],
        "files": [
            {
                "component_path": str(component.path.resolve()),
                "component_sha256": component.sha256,
                "score_path": str(score.path.resolve()),
                "score_sha256": score.sha256,
                "part_id": component.header[5],
                "stack_index_one_based": component.stack_index,
                "mpi_rank": component.header[7],
                "candidate_count": int(component.candidates.size),
                "metrics": metrics,
            }
            for component, score, metrics in pairs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--expected-stacks", required=True)
    parser.add_argument("--expected-iteration", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_directory(
        args.capture_dir,
        _parse_stacks(args.expected_stacks),
        expected_iteration=args.expected_iteration,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
