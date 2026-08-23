#!/usr/bin/env python3
"""Audit exact repeatability of K=4 pass-2 contribution diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "recovar.em_k4_contribution_repeatability.v1"
GROUPS = ("pass2", "contribution", "device_signature")
DEVICE_IGNORED_KEYS = frozenset({"companion_contribution_path"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _element_bytes(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return contiguous.view(np.uint8).reshape(-1, contiguous.dtype.itemsize)


def _json_value(value: np.ndarray) -> Any:
    scalar = value.item()
    if isinstance(scalar, complex):
        return {"real": float(scalar.real), "imag": float(scalar.imag)}
    if isinstance(scalar, (bytes, np.bytes_)):
        return scalar.decode(errors="backslashreplace")
    if isinstance(scalar, np.generic):
        return scalar.item()
    return scalar


def _first_mismatch(
    observed: np.ndarray,
    reference: np.ndarray,
    mismatched_elements: np.ndarray,
) -> dict[str, Any] | None:
    flat_index = int(np.flatnonzero(mismatched_elements)[0])
    index = [int(component) for component in np.unravel_index(flat_index, observed.shape)]
    observed_flat = observed.reshape(-1)
    reference_flat = reference.reshape(-1)
    observed_bytes = _element_bytes(observed)[flat_index]
    reference_bytes = _element_bytes(reference)[flat_index]
    return {
        "index": index,
        "observed": _json_value(observed_flat[flat_index]),
        "reference": _json_value(reference_flat[flat_index]),
        "observed_bytes_hex": observed_bytes.tobytes().hex(),
        "reference_bytes_hex": reference_bytes.tobytes().hex(),
    }


def compare_array(
    observed: np.ndarray,
    reference: np.ndarray,
    *,
    gate_included: bool = True,
) -> dict[str, Any]:
    observed = np.asarray(observed)
    reference = np.asarray(reference)
    shape_equal = observed.shape == reference.shape
    dtype_equal = observed.dtype == reference.dtype
    report: dict[str, Any] = {
        "gate_included": gate_included,
        "observed_shape": list(observed.shape),
        "reference_shape": list(reference.shape),
        "shape_equal": shape_equal,
        "observed_dtype": observed.dtype.str,
        "reference_dtype": reference.dtype.str,
        "dtype_equal": dtype_equal,
        "element_count": int(observed.size),
        "byte_equal": False,
        "mismatch_count": None,
        "first_mismatch": None,
    }
    if not shape_equal or not dtype_equal:
        return report

    observed_bytes = _element_bytes(observed)
    reference_bytes = _element_bytes(reference)
    mismatched_elements = np.any(observed_bytes != reference_bytes, axis=1)
    mismatch_count = int(np.count_nonzero(mismatched_elements))
    report["byte_equal"] = mismatch_count == 0
    report["mismatch_count"] = mismatch_count
    value_mismatch = np.not_equal(observed, reference).reshape(-1)
    report["value_mismatch_count"] = int(np.count_nonzero(value_mismatch))
    if np.issubdtype(observed.dtype, np.inexact):
        paired_nan = (np.isnan(observed) & np.isnan(reference)).reshape(-1)
        report["paired_nan_count"] = int(np.count_nonzero(paired_nan))
        report["paired_nan_byte_mismatch_count"] = int(np.count_nonzero(paired_nan & mismatched_elements))
        report["equal_nan_value_mismatch_count"] = int(np.count_nonzero(value_mismatch & ~paired_nan))
    if np.issubdtype(observed.dtype, np.floating):
        paired_zero = ((observed == 0) & (reference == 0)).reshape(-1)
        sign_mismatch = (np.signbit(observed) != np.signbit(reference)).reshape(-1)
        report["signed_zero_byte_mismatch_count"] = int(
            np.count_nonzero(paired_zero & sign_mismatch & mismatched_elements)
        )
    if mismatch_count:
        report["first_mismatch"] = _first_mismatch(
            observed,
            reference,
            mismatched_elements,
        )
        if np.issubdtype(observed.dtype, np.number):
            with np.errstate(invalid="ignore", over="ignore"):
                delta = np.abs(observed.astype(np.complex128) - reference.astype(np.complex128))
            finite = np.isfinite(delta)
            report["max_abs_difference"] = float(np.max(delta[finite])) if np.any(finite) else None
    return report


def compare_archives(
    observed_path: Path,
    reference_path: Path,
    *,
    ignored_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    observed = _load_archive(observed_path)
    reference = _load_archive(reference_path)
    observed_keys = set(observed)
    reference_keys = set(reference)
    common_keys = sorted(observed_keys & reference_keys)
    comparisons = {
        key: compare_array(
            observed[key],
            reference[key],
            gate_included=key not in ignored_keys,
        )
        for key in common_keys
    }
    gated_keys = [key for key in common_keys if key not in ignored_keys]
    exact = (
        observed_keys == reference_keys
        and ignored_keys.issubset(common_keys)
        and all(comparisons[key]["byte_equal"] for key in gated_keys)
    )
    return {
        "observed": {
            "path": str(observed_path.resolve()),
            "sha256": _sha256(observed_path),
        },
        "reference": {
            "path": str(reference_path.resolve()),
            "sha256": _sha256(reference_path),
        },
        "keys_equal": observed_keys == reference_keys,
        "missing_keys": sorted(reference_keys - observed_keys),
        "unexpected_keys": sorted(observed_keys - reference_keys),
        "ignored_value_keys": sorted(ignored_keys),
        "arrays": comparisons,
        "strict_byte_equal": exact,
    }


def audit_repeatability(
    paths: dict[str, tuple[Path, Path]],
    *,
    owner_job_id: int | None = None,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for group in GROUPS:
        observed_path, reference_path = paths[group]
        ignored_keys = DEVICE_IGNORED_KEYS if group == "device_signature" else frozenset()
        try:
            comparisons[group] = compare_archives(
                observed_path,
                reference_path,
                ignored_keys=ignored_keys,
            )
        except Exception as error:
            errors[group] = f"{type(error).__name__}: {error}"
            comparisons[group] = {
                "observed": {"path": str(observed_path.resolve())},
                "reference": {"path": str(reference_path.resolve())},
                "strict_byte_equal": False,
            }

    gates = {group: bool(comparisons[group]["strict_byte_equal"]) for group in GROUPS}
    passing = sum(gates.values())
    accepted = not errors and passing == len(gates)
    if errors:
        classification = "observer_archives_incomplete"
    elif accepted:
        classification = "same_observer_archives_repeat_bit_for_bit"
    else:
        classification = "same_observer_archives_do_not_repeat_bit_for_bit"
    return {
        "schema": SCHEMA,
        "status": "complete" if not errors else "incomplete",
        "classification": classification,
        "accepted": accepted,
        "scorecard_change_admissible": False,
        "metric_policy": (
            "fixed three-gate archive byte equality; identical keys, shapes, "
            "dtypes, and per-element bytes required; no tolerance, scale, "
            "sign, threshold, map metric, FSC claim, or correlation"
        ),
        "owner_job_id": owner_job_id,
        "fixed_metric": {
            "passing": passing,
            "evaluated": len(gates),
            "gates": gates,
        },
        "errors": errors,
        "comparisons": comparisons,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for group in GROUPS:
        parser.add_argument(
            f"--observed-{group.replace('_', '-')}",
            required=True,
            type=Path,
        )
        parser.add_argument(
            f"--reference-{group.replace('_', '-')}",
            required=True,
            type=Path,
        )
    parser.add_argument("--owner-job-id", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--require-accepted",
        action="store_true",
        help="Return nonzero after writing the report unless all gates pass.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = {
        group: (
            getattr(args, f"observed_{group}"),
            getattr(args, f"reference_{group}"),
        )
        for group in GROUPS
    }
    report = audit_repeatability(paths, owner_job_id=args.owner_job_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(args.require_accepted and not report["accepted"])


if __name__ == "__main__":
    raise SystemExit(main())
