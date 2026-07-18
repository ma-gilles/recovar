"""Sealed single-component variants of a frozen refinement boundary.

This module is diagnostic-only.  A variant is derived from an already sealed
and validated frozen boundary; it never reconstructs a boundary from STAR or
MRC files.  That restriction is deliberate: a partial or inferred control
boundary cannot support a causal one-component experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.frozen_boundary import (
    FROZEN_BOUNDARY_FILENAME,
    FROZEN_BOUNDARY_MANIFEST,
    load_frozen_refinement_boundary,
)

FROZEN_BOUNDARY_VARIANT_SCHEMA = "recovar.em.frozen_boundary_variant.v1"
FROZEN_BOUNDARY_VARIANT_ATTESTATION = "FROZEN_BOUNDARY_VARIANT_V1.json"
FROZEN_BOUNDARY_VARIANT_MANIFEST = "FROZEN_BOUNDARY_VARIANT_V1_SHA256SUMS"

_COMPONENT_PAYLOAD_KEYS = {
    "tau2": frozenset({"mean_variance"}),
}
_TAU2_SOURCE_PATTERN = re.compile(r"it(?P<iteration>\d{3})_tau2\.npy")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    """Hash array metadata and C-order payload bytes without dtype coercion."""

    array = np.asarray(value)
    digest = hashlib.sha256(b"recovar-frozen-boundary-array-v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _load_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _payload_hashes(payload: dict[str, np.ndarray]) -> dict[str, str]:
    return {key: _array_sha256(value) for key, value in sorted(payload.items())}


def _changed_payload_keys(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
) -> list[str]:
    if set(before) != set(after):
        missing = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        raise ValueError(
            f"frozen-boundary variant payload keys differ from the sealed base: missing={missing}, added={added}"
        )
    return [
        key
        for key in sorted(before)
        if before[key].dtype != after[key].dtype
        or before[key].shape != after[key].shape
        or not np.array_equal(before[key], after[key])
    ]


def _load_source_commit(results_path: Path) -> str:
    if not results_path.is_file():
        raise ValueError(f"RECOVAR source results NPZ does not exist: {results_path}")
    with np.load(results_path, allow_pickle=False) as results:
        if "git_commit" not in results.files:
            raise ValueError("RECOVAR source results NPZ is missing git_commit")
        commit_value = np.asarray(results["git_commit"])
        if commit_value.shape != () or commit_value.dtype.kind != "U":
            raise ValueError("RECOVAR source results git_commit must be a Unicode scalar")
        commit = str(commit_value.item())
        if _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
            raise ValueError("RECOVAR source results git_commit must be 40 lowercase hex digits")
        if "git_dirty_count" not in results.files:
            raise ValueError("RECOVAR source results NPZ is missing git_dirty_count")
        dirty_count = np.asarray(results["git_dirty_count"])
        if dirty_count.shape != () or dirty_count.dtype != np.dtype(np.int64):
            raise ValueError("RECOVAR source results git_dirty_count must be an int64 scalar")
        if int(dirty_count.item()) != 0:
            raise ValueError("RECOVAR component source must come from a clean worktree")
    return commit


def _load_tau2_source(
    source_path: Path,
    *,
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    match = _TAU2_SOURCE_PATTERN.fullmatch(source_path.name)
    if match is None:
        raise ValueError("tau2 source must be named itNNN_tau2.npy")
    if source_path.parent.name != "intermediates":
        raise ValueError("tau2 source must live in an intermediates directory")
    if not source_path.is_file():
        raise ValueError(f"tau2 source NPY does not exist: {source_path}")
    value = np.load(source_path, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise ValueError("tau2 source must contain one NumPy array")
    if value.dtype != np.dtype(np.float32):
        raise ValueError(f"tau2 source has dtype {value.dtype}; expected float32")
    if value.shape != expected_shape:
        raise ValueError(f"tau2 source has shape {value.shape}; expected sealed shape {expected_shape}")
    if not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("tau2 source must be finite and nonnegative")
    return np.asarray(value), int(match.group("iteration"))


def _validate_source_layout(component_source: Path, source_results: Path) -> None:
    expected_results = component_source.parent.parent / "refinement_results.npz"
    if source_results != expected_results:
        raise ValueError(
            "component source and refinement results must share the canonical run layout: "
            f"expected {expected_results}, got {source_results}"
        )


def _write_single_file_manifest(path: Path, sealed_path: Path) -> None:
    path.write_text(f"{_sha256(sealed_path)}  {sealed_path.name}\n", encoding="utf-8")


def _load_attestation(output_dir: Path) -> dict[str, object]:
    attestation_path = output_dir / FROZEN_BOUNDARY_VARIANT_ATTESTATION
    manifest_path = output_dir / FROZEN_BOUNDARY_VARIANT_MANIFEST
    if not attestation_path.is_file() or not manifest_path.is_file():
        raise ValueError("frozen-boundary variant attestation or manifest is missing")
    lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("variant attestation manifest must contain exactly one entry")
    fields = lines[0].split(maxsplit=1)
    if len(fields) != 2 or fields[1].lstrip("*") != FROZEN_BOUNDARY_VARIANT_ATTESTATION:
        raise ValueError("variant attestation manifest seals the wrong file")
    if _SHA256_PATTERN.fullmatch(fields[0]) is None:
        raise ValueError("variant attestation manifest contains an invalid SHA-256")
    if _sha256(attestation_path) != fields[0]:
        raise ValueError("variant attestation SHA-256 mismatch")
    value = json.loads(attestation_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("variant attestation must be a JSON object")
    return value


def validate_frozen_boundary_variant(output_dir: str | Path) -> dict[str, object]:
    """Validate a sealed variant and prove exactly one declared component changed."""

    output_dir = Path(output_dir).expanduser().resolve()
    attestation = _load_attestation(output_dir)
    required_keys = {
        "schema",
        "component",
        "source_iteration_zero_based",
        "source_git_commit",
        "base_boundary_dir",
        "base_boundary_sha256",
        "base_manifest_sha256",
        "variant_boundary_sha256",
        "variant_manifest_sha256",
        "component_source_path",
        "component_source_sha256",
        "source_results_path",
        "source_results_sha256",
        "expected_changed_payload_keys",
        "actual_changed_payload_keys",
        "base_payload_sha256",
        "variant_payload_sha256",
    }
    if set(attestation) != required_keys:
        raise ValueError(
            "variant attestation keys do not match schema: "
            f"missing={sorted(required_keys - set(attestation))}, "
            f"unknown={sorted(set(attestation) - required_keys)}"
        )
    if attestation["schema"] != FROZEN_BOUNDARY_VARIANT_SCHEMA:
        raise ValueError("unsupported frozen-boundary variant schema")
    component = str(attestation["component"])
    if component not in _COMPONENT_PAYLOAD_KEYS:
        raise ValueError(f"unsupported frozen-boundary variant component {component!r}")

    base_dir = Path(str(attestation["base_boundary_dir"])).resolve()
    base_boundary = load_frozen_refinement_boundary(base_dir)
    variant_boundary = load_frozen_refinement_boundary(output_dir)
    if base_boundary.boundary_sha256 != attestation["base_boundary_sha256"]:
        raise ValueError("base boundary SHA-256 no longer matches the attestation")
    if base_boundary.source_manifest_sha256 != attestation["base_manifest_sha256"]:
        raise ValueError("base manifest SHA-256 no longer matches the attestation")
    if variant_boundary.boundary_sha256 != attestation["variant_boundary_sha256"]:
        raise ValueError("variant boundary SHA-256 does not match the attestation")
    if variant_boundary.source_manifest_sha256 != attestation["variant_manifest_sha256"]:
        raise ValueError("variant manifest SHA-256 does not match the attestation")

    component_source = Path(str(attestation["component_source_path"])).resolve()
    source_results = Path(str(attestation["source_results_path"])).resolve()
    _validate_source_layout(component_source, source_results)
    if _sha256(component_source) != attestation["component_source_sha256"]:
        raise ValueError("component source SHA-256 no longer matches the attestation")
    if _sha256(source_results) != attestation["source_results_sha256"]:
        raise ValueError("source results SHA-256 no longer matches the attestation")
    if _load_source_commit(source_results) != attestation["source_git_commit"]:
        raise ValueError("source git commit no longer matches the attestation")
    expected_source_iteration = base_boundary.completed_relion_iteration - 1
    if attestation["source_iteration_zero_based"] != expected_source_iteration:
        raise ValueError(
            "component source iteration is not the RECOVAR boundary paired with the completed RELION iteration"
        )

    base_payload = _load_payload(base_dir / FROZEN_BOUNDARY_FILENAME)
    variant_payload = _load_payload(output_dir / FROZEN_BOUNDARY_FILENAME)
    expected_changed = sorted(_COMPONENT_PAYLOAD_KEYS[component])
    actual_changed = _changed_payload_keys(base_payload, variant_payload)
    if attestation["expected_changed_payload_keys"] != expected_changed:
        raise ValueError("attested expected component keys are incorrect")
    if attestation["actual_changed_payload_keys"] != actual_changed:
        raise ValueError("attested actual component keys are incorrect")
    if actual_changed != expected_changed:
        raise ValueError(
            "variant does not change exactly one declared component: "
            f"expected={expected_changed}, actual={actual_changed}"
        )
    base_hashes = _payload_hashes(base_payload)
    variant_hashes = _payload_hashes(variant_payload)
    if attestation["base_payload_sha256"] != base_hashes:
        raise ValueError("base payload hashes do not match the attestation")
    if attestation["variant_payload_sha256"] != variant_hashes:
        raise ValueError("variant payload hashes do not match the attestation")
    if variant_hashes["mean_variance"] != _array_sha256(np.load(component_source, allow_pickle=False)):
        raise ValueError("variant tau2 does not exactly equal the sealed component source")
    return attestation


def build_frozen_boundary_variant(
    *,
    base_boundary_dir: str | Path,
    output_dir: str | Path,
    component: str,
    component_source: str | Path,
    source_results: str | Path,
) -> dict[str, object]:
    """Create and validate one sealed single-component boundary variant."""

    if component not in _COMPONENT_PAYLOAD_KEYS:
        raise ValueError(f"unsupported frozen-boundary variant component {component!r}")
    base_dir = Path(base_boundary_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    component_source = Path(component_source).expanduser().resolve()
    source_results = Path(source_results).expanduser().resolve()
    if output_dir == base_dir:
        raise ValueError("variant output directory must differ from the sealed base")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"variant output directory is not empty: {output_dir}")

    base_boundary = load_frozen_refinement_boundary(base_dir)
    _validate_source_layout(component_source, source_results)
    source_commit = _load_source_commit(source_results)
    base_payload = _load_payload(base_dir / FROZEN_BOUNDARY_FILENAME)
    variant_payload = {key: value.copy() for key, value in base_payload.items()}
    if component == "tau2":
        tau2, source_iteration = _load_tau2_source(
            component_source,
            expected_shape=base_payload["mean_variance"].shape,
        )
        variant_payload["mean_variance"] = tau2.copy()
    else:  # pragma: no cover - guarded by the component registry
        raise AssertionError(component)

    expected_source_iteration = base_boundary.completed_relion_iteration - 1
    if source_iteration != expected_source_iteration:
        raise ValueError(
            "tau2 source iteration does not match the sealed boundary: "
            f"source={source_iteration}, expected={expected_source_iteration}"
        )

    actual_changed = _changed_payload_keys(base_payload, variant_payload)
    expected_changed = sorted(_COMPONENT_PAYLOAD_KEYS[component])
    if actual_changed != expected_changed:
        raise ValueError(
            "component source does not produce exactly the declared change: "
            f"expected={expected_changed}, actual={actual_changed}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_path = output_dir / FROZEN_BOUNDARY_FILENAME
    np.savez(boundary_path, **variant_payload)
    boundary_manifest = output_dir / FROZEN_BOUNDARY_MANIFEST
    _write_single_file_manifest(boundary_manifest, boundary_path)
    variant_boundary = load_frozen_refinement_boundary(output_dir)

    attestation: dict[str, object] = {
        "schema": FROZEN_BOUNDARY_VARIANT_SCHEMA,
        "component": component,
        "source_iteration_zero_based": source_iteration,
        "source_git_commit": source_commit,
        "base_boundary_dir": os.fspath(base_dir),
        "base_boundary_sha256": base_boundary.boundary_sha256,
        "base_manifest_sha256": base_boundary.source_manifest_sha256,
        "variant_boundary_sha256": variant_boundary.boundary_sha256,
        "variant_manifest_sha256": variant_boundary.source_manifest_sha256,
        "component_source_path": os.fspath(component_source),
        "component_source_sha256": _sha256(component_source),
        "source_results_path": os.fspath(source_results),
        "source_results_sha256": _sha256(source_results),
        "expected_changed_payload_keys": expected_changed,
        "actual_changed_payload_keys": actual_changed,
        "base_payload_sha256": _payload_hashes(base_payload),
        "variant_payload_sha256": _payload_hashes(variant_payload),
    }
    attestation_path = output_dir / FROZEN_BOUNDARY_VARIANT_ATTESTATION
    attestation_path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_single_file_manifest(
        output_dir / FROZEN_BOUNDARY_VARIANT_MANIFEST,
        attestation_path,
    )
    return validate_frozen_boundary_variant(output_dir)
