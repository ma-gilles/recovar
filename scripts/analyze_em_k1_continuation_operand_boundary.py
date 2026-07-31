#!/usr/bin/env python3
"""Compare uninterrupted and continued RELION K=1 iteration-2 operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_relion_coarse_operand_capture import (  # noqa: E402
    CoarseOperandCapture,
)
from scripts.validate_relion_coarse_operand_capture import (  # noqa: E402
    load_artifact as load_operand,
)
from scripts.validate_relion_preprocess_capture import (  # noqa: E402
    RelionPreprocessCapture,
)
from scripts.validate_relion_preprocess_capture import (  # noqa: E402
    load_artifact as load_preprocess,
)

SCHEMA = "recovar.em_k1_continuation_operand_boundary.v1"
EXPECTED_PARTICLES = 14
STAR_VALUE = re.compile(r"^(?P<label>_rln\S+)\s+(?P<value>\S+)\s*$")
PREPROCESS_FIELDS = (
    "norm_correction",
    "old_offset",
    "raw_input_real",
    "normalized_shifted_real",
    "unmasked_fourier_pre_optics",
    "unmasked_fourier_post_optics",
    "masked_real",
    "masked_fourier_pre_optics",
    "masked_fourier_post_optics",
)
OPERAND_FIELDS = (
    "rotation_keys",
    "local_rotation_indices",
    "euler_matrices",
    "reference_real",
    "reference_imag",
    "image_real",
    "image_imag",
    "correction",
    "translations",
    "shifted_real",
    "shifted_imag",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_star_float(path: Path, label: str) -> float:
    """Read one scalar STAR-list value without normalizing its source file."""

    matches: list[str] = []
    for line in Path(path).read_text().splitlines():
        match = STAR_VALUE.fullmatch(line)
        if match is not None and match["label"] == label:
            matches.append(match["value"])
    _require(len(matches) == 1, f"{path}: expected one {label}, found {len(matches)}")
    value = float(matches[0])
    _require(np.isfinite(value), f"{path}: {label} is not finite")
    return value


def array_metrics(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Return scale-sensitive cross-arm metrics; correlation is absent."""

    source = np.asarray(source)
    target = np.asarray(target)
    _require(source.shape == target.shape, "paired array shapes differ")
    _require(source.dtype == target.dtype, "paired array dtypes differ")
    _require(source.size > 0, "paired arrays are empty")
    exact = bool(np.array_equal(source, target, equal_nan=True))
    unequal = ~np.equal(source, target)
    if source.dtype.kind in {"f", "c"}:
        unequal &= ~(np.isnan(source) & np.isnan(target))
        _require(
            np.array_equal(np.isfinite(source), np.isfinite(target)),
            "paired finite-value masks differ",
        )
        finite = np.isfinite(source)
        source_values = source[finite].astype(
            np.complex128 if source.dtype.kind == "c" else np.float64,
        )
        target_values = target[finite].astype(
            np.complex128 if target.dtype.kind == "c" else np.float64,
        )
        difference = np.abs(source_values - target_values)
        source_norm = float(np.linalg.norm(source_values))
        difference_norm = float(np.linalg.norm(difference))
        return {
            "bitwise_equal": exact,
            "mismatch_elements": int(np.count_nonzero(unequal)),
            "evaluated_elements": int(source.size),
            "max_abs": float(np.max(difference, initial=0.0)),
            "p95_abs": float(np.percentile(difference, 95)),
            "relative_l2": (difference_norm / source_norm if source_norm > 0 else difference_norm),
        }
    return {
        "bitwise_equal": exact,
        "mismatch_elements": int(np.count_nonzero(unequal)),
        "evaluated_elements": int(source.size),
        "max_abs": None,
        "p95_abs": None,
        "relative_l2": None,
    }


def summarize_field(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one fixed field over paired particles."""

    _require(bool(metrics), "field metric cohort is empty")
    numeric = [record for record in metrics if record["relative_l2"] is not None]
    summary: dict[str, Any] = {
        "bitwise_equal_particles": sum(record["bitwise_equal"] for record in metrics),
        "evaluated_particles": len(metrics),
        "mismatch_elements": sum(record["mismatch_elements"] for record in metrics),
        "evaluated_elements": sum(record["evaluated_elements"] for record in metrics),
    }
    if numeric:
        relative_l2 = np.asarray(
            [record["relative_l2"] for record in numeric],
            dtype=np.float64,
        )
        summary.update(
            {
                "relative_l2_min": float(np.min(relative_l2)),
                "relative_l2_median": float(np.median(relative_l2)),
                "relative_l2_max": float(np.max(relative_l2)),
                "max_abs": float(max(record["max_abs"] for record in numeric)),
                "p95_abs_max": float(max(record["p95_abs"] for record in numeric)),
            }
        )
    return summary


def _load_pairs(
    fresh_directory: Path,
    restart_directory: Path,
    *,
    suffix: str,
    loader: Callable[[Path], Any],
    expected_particles: int,
) -> tuple[tuple[Any, Any], ...]:
    fresh_paths = sorted(Path(fresh_directory).glob(f"*.{suffix}"))
    restart_paths = sorted(Path(restart_directory).glob(f"*.{suffix}"))
    _require(
        len(fresh_paths) == expected_particles,
        f"fresh {suffix} denominator changed",
    )
    _require(
        len(restart_paths) == expected_particles,
        f"restart {suffix} denominator changed",
    )
    _require(
        [path.name for path in fresh_paths] == [path.name for path in restart_paths],
        f"paired {suffix} identities changed",
    )
    pairs = tuple(
        (loader(fresh_path), loader(restart_path))
        for fresh_path, restart_path in zip(
            fresh_paths,
            restart_paths,
            strict=True,
        )
    )
    identities = [(fresh.part_id, fresh.stack_index) for fresh, _ in pairs]
    _require(
        len(set(identities)) == expected_particles,
        f"paired {suffix} identities are not unique",
    )
    return pairs


def _compare_fields(
    pairs: tuple[
        tuple[RelionPreprocessCapture, RelionPreprocessCapture] | tuple[CoarseOperandCapture, CoarseOperandCapture],
        ...,
    ],
    fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    _require(bool(pairs), "capture pair cohort is empty")
    for fresh, restart in pairs:
        _require(fresh.part_id == restart.part_id, "paired part identity changed")
        _require(
            fresh.stack_index == restart.stack_index,
            "paired stack identity changed",
        )
    return {
        field: summarize_field(
            [array_metrics(getattr(fresh, field), getattr(restart, field)) for fresh, restart in pairs]
        )
        for field in fields
    }


def classify_boundary(
    preprocess: dict[str, dict[str, Any]],
    operands: dict[str, dict[str, Any]],
    sampling: dict[str, float],
    *,
    expected_particles: int,
) -> str:
    """Classify the fixed fresh-versus-continuation operand boundary."""

    for name in PREPROCESS_FIELDS:
        _require(
            preprocess[name]["evaluated_particles"] == expected_particles,
            f"{name}: denominator changed",
        )
    for name in OPERAND_FIELDS:
        _require(
            operands[name]["evaluated_particles"] == expected_particles,
            f"{name}: denominator changed",
        )
    raw_exact = preprocess["raw_input_real"]["bitwise_equal_particles"]
    rotation_exact = operands["rotation_keys"]["bitwise_equal_particles"]
    local_exact = operands["local_rotation_indices"]["bitwise_equal_particles"]
    euler_exact = operands["euler_matrices"]["bitwise_equal_particles"]
    translation_exact = operands["translations"]["bitwise_equal_particles"]
    source_state_discarded = not np.isclose(
        sampling["source_iteration1"],
        sampling["restart_output_iteration1"],
        rtol=0.0,
        atol=1.0e-8,
    )
    iteration2_diverged = not np.isclose(
        sampling["fresh_iteration2"],
        sampling["restart_iteration2"],
        rtol=0.0,
        atol=1.0e-8,
    )
    if (
        raw_exact == expected_particles
        and rotation_exact == expected_particles
        and local_exact == expected_particles
        and euler_exact == 0
        and translation_exact == 0
        and source_state_discarded
        and iteration2_diverged
    ):
        return "serialized_sampling_perturbation_discarded_before_euler_and_translation_geometry"
    return "continuation_operand_boundary_not_uniquely_classified"


def analyze(
    *,
    fresh_capture_directory: Path,
    restart_capture_directory: Path,
    source_iteration1_sampling: Path,
    restart_output_iteration1_sampling: Path,
    fresh_iteration2_sampling: Path,
    restart_iteration2_sampling: Path,
    expected_particles: int = EXPECTED_PARTICLES,
) -> dict[str, Any]:
    """Build the deterministic cross-arm operand-boundary report."""

    _require(expected_particles > 0, "expected particle count must be positive")
    preprocess_pairs = _load_pairs(
        fresh_capture_directory,
        restart_capture_directory,
        suffix="preprocess-v1.bin",
        loader=load_preprocess,
        expected_particles=expected_particles,
    )
    operand_pairs = _load_pairs(
        fresh_capture_directory,
        restart_capture_directory,
        suffix="p1-op-v2.bin",
        loader=load_operand,
        expected_particles=expected_particles,
    )
    preprocess = _compare_fields(preprocess_pairs, PREPROCESS_FIELDS)
    operands = _compare_fields(operand_pairs, OPERAND_FIELDS)
    sampling_paths = {
        "source_iteration1": source_iteration1_sampling,
        "restart_output_iteration1": restart_output_iteration1_sampling,
        "fresh_iteration2": fresh_iteration2_sampling,
        "restart_iteration2": restart_iteration2_sampling,
    }
    sampling = {name: read_star_float(path, "_rlnSamplingPerturbInstance") for name, path in sampling_paths.items()}
    classification = classify_boundary(
        preprocess,
        operands,
        sampling,
        expected_particles=expected_particles,
    )
    identities = [
        {
            "part_id": fresh.part_id,
            "stack_index": fresh.stack_index,
            "fresh_preprocess_sha256": fresh.sha256,
            "restart_preprocess_sha256": restart.sha256,
        }
        for fresh, restart in preprocess_pairs
    ]
    operand_by_identity = {
        (fresh.part_id, fresh.stack_index): {
            "fresh_operand_sha256": fresh.sha256,
            "restart_operand_sha256": restart.sha256,
        }
        for fresh, restart in operand_pairs
    }
    _require(
        set(operand_by_identity) == {(identity["part_id"], identity["stack_index"]) for identity in identities},
        "preprocess and operand cohorts differ",
    )
    for identity in identities:
        identity.update(operand_by_identity[(identity["part_id"], identity["stack_index"])])
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification": classification,
        "correlation_computed": False,
        "fixed_metric": {
            "expected_particles": expected_particles,
            "evaluated_particles": len(preprocess_pairs),
            "raw_input_bitwise_equal": preprocess["raw_input_real"]["bitwise_equal_particles"],
            "norm_correction_bitwise_equal": preprocess["norm_correction"]["bitwise_equal_particles"],
            "old_offset_bitwise_equal": preprocess["old_offset"]["bitwise_equal_particles"],
            "rotation_keys_bitwise_equal": operands["rotation_keys"]["bitwise_equal_particles"],
            "local_rotation_indices_bitwise_equal": operands["local_rotation_indices"]["bitwise_equal_particles"],
            "euler_matrices_bitwise_equal": operands["euler_matrices"]["bitwise_equal_particles"],
            "translation_values_bitwise_equal": operands["translations"]["bitwise_equal_particles"],
        },
        "sampling_perturbation": sampling,
        "preprocess_fields": preprocess,
        "operand_fields": operands,
        "artifacts": identities,
        "sampling_inputs": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": _sha256(path),
            }
            for name, path in sampling_paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-capture-directory", type=Path, required=True)
    parser.add_argument("--restart-capture-directory", type=Path, required=True)
    parser.add_argument("--source-iteration1-sampling", type=Path, required=True)
    parser.add_argument(
        "--restart-output-iteration1-sampling",
        type=Path,
        required=True,
    )
    parser.add_argument("--fresh-iteration2-sampling", type=Path, required=True)
    parser.add_argument("--restart-iteration2-sampling", type=Path, required=True)
    parser.add_argument("--expected-particles", type=int, default=EXPECTED_PARTICLES)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = analyze(
        fresh_capture_directory=args.fresh_capture_directory,
        restart_capture_directory=args.restart_capture_directory,
        source_iteration1_sampling=args.source_iteration1_sampling,
        restart_output_iteration1_sampling=args.restart_output_iteration1_sampling,
        fresh_iteration2_sampling=args.fresh_iteration2_sampling,
        restart_iteration2_sampling=args.restart_iteration2_sampling,
        expected_particles=args.expected_particles,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
