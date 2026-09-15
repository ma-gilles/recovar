#!/usr/bin/env python3
"""Analyze the sealed eight-arm VDAM late-iteration crossed pair.

The benchmark is a two-by-two factorial comparison of the canonical/native-
atomic reduction and serial/eight-stream dispatch, with two mirrored repeats.
This analyzer proves that the requested selector actually executed, checks
exact discrete-state parity, and bounds map changes by the observed repeat
envelope.  It never widens a numerical tolerance: a zero repeat envelope
requires exact map equality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import sqlite3
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

import mrcfile
import numpy as np
import starfile

from recovar.em.diagnostics.coarse_score_diagnostics import _validate_coarse_selector_audit
from scripts import summarize_vdam_nsys_sqlite as nsys_sqlite
from scripts.analyze_vdam_coarse_combined_true200 import (
    _has_nonfinite_numeric,
    _values_equal,
)

SCHEMA = "recovar.vdam_coarse_multistream_late_pair_analysis.v1"
RUN_SCHEMA = "recovar.vdam_coarse_atomic_multistream_crossed.v3"
FOCUSED_GATE_SCHEMA = "recovar.vdam_coarse_multistream_focused_gpu_gate.v2"
PROFILE_SCHEMA = "recovar.vdam_late_iteration_profile.v1"
NSIGHT_SCHEMA = "recovar.vdam_nsys_sqlite_summary.v1"
PROFILED_ITERATION = 181
MATERIAL_WALL_WIN_PERCENT = -5.0
COARSE_KERNEL_NAME = "relion_coarse_diff2_projector_f32_kernel"
PROFILE_METRICS = (
    "warm_wall_s",
    "warm_expectation_s",
    "pass1_s",
    "pass2_s",
    "gpu_kernel_union_s",
    "coarse_kernel_sum_s",
    "coarse_kernel_interval_union_s",
    "coarse_kernel_launch_count",
)
FOCUSED_TEST_NODE = (
    "tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_vdam_multistream_atomic_stays_in_lane_envelope"
)
ACCEPTANCE_RULE = {
    "mathematically_equivalent_reduction_required": True,
    "numeric_delta_within_repeat_envelope_required": True,
    "nondirectional_signed_delta_required": True,
    "repeat_growth_forbidden": True,
    "discrete_decisions_preserved_required": True,
    "basin_preserved_required": True,
    "material_end_to_end_runtime_win_required": True,
}

ARM_SPECS = (
    ("canonical_serial_1", 0, False, 1),
    ("atomic_serial_1", 0, True, 1),
    ("atomic_multistream_1", 8, True, 1),
    ("canonical_multistream_1", 8, False, 1),
    ("canonical_multistream_2", 8, False, 2),
    ("atomic_multistream_2", 8, True, 2),
    ("atomic_serial_2", 0, True, 2),
    ("canonical_serial_2", 0, False, 2),
)
ARM_LABELS = tuple(spec[0] for spec in ARM_SPECS)
CONFIGURATIONS = (
    "canonical_serial",
    "atomic_serial",
    "canonical_multistream",
    "atomic_multistream",
)
DISCRETE_META_KEYS = (
    "selected_particle_ids",
    "best_pose_rotation_ids",
    "best_pose_rotations",
    "best_pose_translations",
    "class_assignments",
    "max_posterior_per_image",
    "pose_assignments",
    "halfset_0_class_assignments",
)
_HALFSET_PROFILE_RE = re.compile(r"^halfset_(?P<halfset>\d+)_profile_summary$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class LatePairSetupError(RuntimeError):
    """Raised when benchmark evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LatePairSetupError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise LatePairSetupError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LatePairSetupError(f"cannot read {label} at {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must contain one JSON object: {path}")
    return value


def _git_rev_parse(repo: Path, revision: str, label: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", revision],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise LatePairSetupError(f"cannot resolve {label} in {repo}: {exc}") from exc
    _require(
        completed.returncode == 0,
        f"cannot resolve {label} in --repo: {completed.stderr.strip()}",
    )
    rows = completed.stdout.splitlines()
    _require(len(rows) == 1 and bool(_GIT_OBJECT_RE.fullmatch(rows[0])), f"resolved {label} is invalid")
    return rows[0]


def _resolved_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise LatePairSetupError(f"{label} escapes benchmark root: {path}") from exc
    return resolved


def _read_single_line(path: Path, label: str) -> str:
    try:
        rows = path.read_text().splitlines()
    except OSError as exc:
        raise LatePairSetupError(f"cannot read {label} at {path}: {exc}") from exc
    _require(len(rows) == 1 and bool(rows[0].strip()), f"{label} must contain one value")
    return rows[0].strip()


def _parse_sha_line(path: Path, label: str, *, root: Path) -> tuple[str, Path]:
    line = _read_single_line(path, label)
    try:
        digest, raw_path = line.split(maxsplit=1)
    except ValueError as exc:
        raise LatePairSetupError(f"{label} has malformed sha256sum output") from exc
    _require(bool(_SHA256_RE.fullmatch(digest)), f"{label} has an invalid digest")
    artifact = _resolved_inside(Path(raw_path), root, label)
    _require(artifact.is_file(), f"{label} target is missing: {artifact}")
    _require(_sha256(artifact) == digest, f"{label} target digest differs: {artifact}")
    return digest, artifact


def _validate_manifest(
    path: Path,
    *,
    expected_digest: str,
    relative_base: Path | None,
    label: str,
) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    _require(bool(_SHA256_RE.fullmatch(expected_digest)), f"{label} expected digest is invalid")
    _require(_sha256(path) == expected_digest, f"{label} digest differs")
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise LatePairSetupError(f"cannot read {label}: {exc}") from exc
    _require(bool(lines), f"{label} is empty")
    entries = []
    seen: set[Path] = set()
    for line in lines:
        try:
            digest, raw_path = line.split(maxsplit=1)
        except ValueError as exc:
            raise LatePairSetupError(f"{label} contains a malformed row") from exc
        _require(bool(_SHA256_RE.fullmatch(digest)), f"{label} row has an invalid digest")
        candidate = Path(raw_path)
        if candidate.is_absolute():
            artifact = candidate.resolve()
        else:
            _require(relative_base is not None, f"{label} contains an unresolved relative path")
            _require(".." not in candidate.parts, f"{label} path escapes its base: {candidate}")
            artifact = (relative_base / candidate).resolve()
            try:
                artifact.relative_to(relative_base.resolve())
            except ValueError as exc:
                raise LatePairSetupError(f"{label} path escapes its base: {candidate}") from exc
        _require(artifact not in seen, f"{label} repeats an artifact: {artifact}")
        seen.add(artifact)
        _require(artifact.is_file(), f"{label} artifact is missing: {artifact}")
        _require(_sha256(artifact) == digest, f"{label} artifact digest differs: {artifact}")
        entries.append({"path": str(artifact), "sha256": digest})
    return {"sha256": expected_digest, "entries": entries}


def _validate_junit(path: Path) -> None:
    _require(path.is_file(), f"missing focused GPU JUnit XML: {path}")
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise LatePairSetupError(f"cannot parse focused GPU JUnit XML: {exc}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    _require(len(suites) == 1, "focused GPU JUnit XML must contain one test suite")
    suite = suites[0]
    expected = {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
    observed = {name: int(suite.attrib.get(name, 0)) for name in expected}
    _require(observed == expected, f"focused GPU gate did not pass exactly once: {observed}")


def _validate_execution_order(path: Path, root: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing execution order: {path}")
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
    except OSError as exc:
        raise LatePairSetupError(f"cannot read execution order: {exc}") from exc
    expected_columns = (
        "order",
        "label",
        "workers",
        "single_lane_canonical",
        "native_atomic_reduction",
        "nsys_base",
    )
    _require(
        tuple(rows[0].keys()) == expected_columns if rows else False,
        "execution-order columns differ",
    )
    _require(len(rows) == len(ARM_SPECS), "execution order does not contain eight arms")
    result = []
    for order, (row, spec) in enumerate(zip(rows, ARM_SPECS, strict=True), start=1):
        label, workers, atomic, _ = spec
        expected = {
            "order": str(order),
            "label": label,
            "workers": str(workers),
            "single_lane_canonical": "0",
            "native_atomic_reduction": str(int(atomic)),
        }
        mismatches = {key: (row.get(key), value) for key, value in expected.items() if row.get(key) != value}
        _require(not mismatches, f"execution-order arm {order} differs: {mismatches}")
        nsys_base = _resolved_inside(Path(row["nsys_base"]), root, f"{label} Nsight base")
        _require(
            nsys_base.name == f"{label}_it{PROFILED_ITERATION:03d}_warm",
            f"{label} Nsight base has the wrong name",
        )
        result.append({**expected, "nsys_base": str(nsys_base)})
    return result


def _validate_command(path: Path, *, workers: int, atomic: bool, label: str) -> str:
    _require(path.is_file(), f"missing command ledger for {label}: {path}")
    try:
        tokens = shlex.split(path.read_text())
    except (OSError, ValueError) as exc:
        raise LatePairSetupError(f"cannot parse command ledger for {label}: {exc}") from exc
    expected_assignments = {
        "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": str(workers),
        "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL": "0",
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": str(int(atomic)),
    }
    for name, value in expected_assignments.items():
        matches = [token for token in tokens if token.startswith(f"{name}=")]
        _require(matches == [f"{name}={value}"], f"{label} command has the wrong {name}")
    _require(
        tokens.count("scripts.run_vdam_late_iteration_profile") == 1,
        f"{label} command does not invoke the late-iteration profiler exactly once",
    )
    return _sha256(path)


def _validate_resource_files(
    report_path: Path,
    demangled_path: Path,
    *,
    cuda_sha256: str,
    report_sha256: str,
    demangled_sha256: str,
    label: str,
) -> dict[str, Any]:
    _require(_sha256(report_path) == report_sha256, f"{label} resource report digest differs")
    _require(_sha256(demangled_path) == demangled_sha256, f"{label} demangled resource digest differs")
    report = _load_json(report_path, f"{label} resource report")
    expected_resources = {
        "canonical": {"registers_per_thread": 56, "static_shared_bytes": 15232},
        "atomic": {"registers_per_thread": 48, "static_shared_bytes": 7040},
    }
    _require(report.get("binary_sha256") == cuda_sha256, f"{label} resource report CUDA differs")
    _require(
        report.get("demangled_resource_report_sha256") == demangled_sha256,
        f"{label} resource report demangled digest differs",
    )
    _require(report.get("resources") == expected_resources, f"{label} kernel resources differ")
    return {
        "report_path": str(report_path.resolve()),
        "report_sha256": report_sha256,
        "demangled_path": str(demangled_path.resolve()),
        "demangled_sha256": demangled_sha256,
        "resources": expected_resources,
    }


def _validate_focused_gate(
    gate_root: Path,
    *,
    repo: Path,
    late_run: dict[str, Any],
) -> dict[str, Any]:
    _require(gate_root.is_dir(), "qualified focused GPU gate root is missing")
    _require((gate_root / "COMPLETED").is_file(), "qualified focused GPU gate is incomplete")
    provenance = gate_root / "provenance"
    gate_run_path = provenance / "run.json"
    gate = _load_json(gate_run_path, "qualified focused GPU gate run provenance")
    expected = {
        "schema": FOCUSED_GATE_SCHEMA,
        "classification": "performance_only_qualification",
        "git_head": late_run["git_head"],
        "git_tree": late_run["git_tree"],
        "interpreter_sha256": late_run["interpreter_sha256"],
        "gpu_uuid": late_run["gpu_uuid"],
        "node": late_run["node"],
        "gpu_name": late_run["gpu_name"],
        "cuda_sha256": late_run["cuda_sha256"],
        "source_manifest_scope": "selected_high_risk_files",
        "tests": 3,
        "passed": 3,
        "science_promotion_allowed": False,
    }
    mismatches = {
        key: {"expected": value, "observed": gate.get(key)} for key, value in expected.items() if gate.get(key) != value
    }
    _require(not mismatches, f"qualified focused GPU gate differs: {mismatches}")
    _require(isinstance(gate.get("job_id"), str) and gate["job_id"], "qualified gate job ID is invalid")
    _require(
        isinstance(gate.get("source_manifest_sha256"), str)
        and bool(_SHA256_RE.fullmatch(gate["source_manifest_sha256"])),
        "qualified gate source manifest digest is invalid",
    )
    for name in ("cuda_stage_wall_s", "test_wall_s"):
        value = gate.get(name)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0,
            f"qualified gate {name} is invalid",
        )
    _require(isinstance(gate.get("pytest_node"), str) and gate["pytest_node"], "qualified gate test list is empty")

    recorded = {
        "git_head": _read_single_line(provenance / "repo_head.txt", "qualified gate repo HEAD"),
        "git_tree": _read_single_line(provenance / "repo_tree.txt", "qualified gate repo tree"),
        "job_id": _read_single_line(provenance / "slurm_job_id.txt", "qualified gate Slurm job"),
        "gpu_uuid": _read_single_line(provenance / "selected_gpu_uuid.txt", "qualified gate GPU UUID"),
        "node": _read_single_line(provenance / "node.txt", "qualified gate node"),
        "gpu_name": _read_single_line(provenance / "gpu_name.txt", "qualified gate GPU name"),
    }
    _require(
        all(recorded[name] == gate[name] for name in recorded),
        f"qualified gate scalar provenance differs: {recorded}",
    )
    _require((provenance / "repo_status.txt").read_text() == "", "qualified gate repository was dirty")
    gate_diff = _read_single_line(provenance / "repo_diff.sha256", "qualified gate repo diff").split()[0]
    _require(gate_diff == _EMPTY_SHA256, "qualified gate repository diff was nonempty")
    for name in ("allocated_gpu_uuids.csv", "visible_gpu_uuids.csv"):
        uuids = [value.strip() for value in _read_single_line(provenance / name, name).split(",")]
        _require(uuids == [gate["gpu_uuid"]], f"qualified gate {name} does not prove single-GPU ownership")

    source_path = provenance / "source_manifest.sha256"
    source = _validate_manifest(
        source_path,
        expected_digest=gate["source_manifest_sha256"],
        relative_base=repo,
        label="qualified gate source manifest",
    )
    final_source = provenance / "source_manifest.final.sha256"
    _require(final_source.is_file(), "qualified gate final source manifest is missing")
    _require(final_source.read_bytes() == source_path.read_bytes(), "qualified gate source changed during execution")

    interpreter_sha, interpreter_path = _parse_sha_line(
        provenance / "interpreter.sha256", "qualified gate interpreter", root=repo
    )
    _require(interpreter_sha == gate["interpreter_sha256"], "qualified gate interpreter differs")
    cuda_sha, cuda_path = _parse_sha_line(
        provenance / "qualified_cuda.sha256", "qualified gate CUDA binary", root=gate_root
    )
    _require(cuda_sha == gate["cuda_sha256"], "qualified gate CUDA binary differs")
    resources = _validate_resource_files(
        provenance / "coarse_kernel_resources.json",
        provenance / "cuda_resource_usage.demangled.txt",
        cuda_sha256=gate["cuda_sha256"],
        report_sha256=late_run["resource_report_sha256"],
        demangled_sha256=late_run["resource_demangled_sha256"],
        label="qualified gate",
    )
    return {
        "root": str(gate_root),
        "run_json_sha256": _sha256(gate_run_path),
        "job_id": gate["job_id"],
        "interpreter": {"path": str(interpreter_path), "sha256": interpreter_sha},
        "cuda_binary": {"path": str(cuda_path), "sha256": cuda_sha},
        "source_manifest": source,
        "resources": resources,
    }


def _validate_provenance(root: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(root.is_dir(), f"late-pair root does not exist: {root}")
    _require((root / "COMPLETED").is_file(), "late-pair root is not complete")
    _require(not (root / "provenance" / "failure.txt").exists(), "late-pair root records a failure")
    provenance = root / "provenance"
    run_path = provenance / "run.json"
    run = _load_json(run_path, "run provenance")
    expected_run = {
        "schema": RUN_SCHEMA,
        "classification": "diagnostic_performance_only",
        "execution_order": list(ARM_LABELS),
        "coarse_multistream_workers": [spec[1] for spec in ARM_SPECS],
        "single_lane_canonical": [False] * len(ARM_SPECS),
        "native_atomic_reduction": [spec[2] for spec in ARM_SPECS],
        "raw_image_cache": "off",
        "exact_local_bucket_radix": 4,
        "exact_local_physical_order_chunk_size": 0,
        "source_manifest_scope": "selected_high_risk_files",
        "focused_test_node": FOCUSED_TEST_NODE,
        "acceptance_rule": ACCEPTANCE_RULE,
        "science_promotion_allowed": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": run.get(key)}
        for key, expected in expected_run.items()
        if run.get(key) != expected
    }
    _require(not mismatches, f"run provenance differs: {mismatches}")
    for name in ("git_head", "git_tree"):
        _require(
            isinstance(run.get(name), str) and bool(_GIT_OBJECT_RE.fullmatch(run[name])),
            f"run provenance {name} is invalid",
        )
    resolved_head = _git_rev_parse(repo, f"{run['git_head']}^{{commit}}", "run git_head")
    _require(resolved_head == run["git_head"], "run git_head did not resolve to the recorded commit")
    resolved_tree = _git_rev_parse(repo, f"{run['git_head']}^{{tree}}", "run git tree")
    _require(resolved_tree == run["git_tree"], "run git tree differs from the recorded git_tree")
    for name in (
        "cuda_sha256",
        "source_manifest_sha256",
        "input_manifest_sha256",
        "relion_bind_sha256",
        "interpreter_sha256",
        "resource_report_sha256",
        "resource_demangled_sha256",
    ):
        _require(
            isinstance(run.get(name), str) and bool(_SHA256_RE.fullmatch(run[name])),
            f"run provenance {name} is invalid",
        )
    _require(isinstance(run.get("job_id"), str) and run["job_id"], "run job ID is invalid")
    _require(isinstance(run.get("gpu_uuid"), str) and run["gpu_uuid"].startswith("GPU-"), "run GPU UUID is invalid")
    _require("H100" in str(run.get("gpu_name", "")), "run was not recorded on an H100")
    _require(isinstance(run.get("node"), str) and run["node"], "run node is invalid")

    recorded = {
        "git_head": _read_single_line(provenance / "repo_head.txt", "recorded repo HEAD"),
        "git_tree": _read_single_line(provenance / "repo_tree.txt", "recorded repo tree"),
        "job_id": _read_single_line(provenance / "slurm_job_id.txt", "recorded Slurm job"),
        "gpu_uuid": _read_single_line(provenance / "selected_gpu_uuid.txt", "selected GPU UUID"),
        "node": _read_single_line(provenance / "node.txt", "recorded node"),
        "gpu_name": _read_single_line(provenance / "gpu_name.txt", "recorded GPU name"),
    }
    _require(
        all(recorded[name] == run[name] for name in recorded),
        f"recorded scalar provenance differs from run.json: {recorded}",
    )
    _require((provenance / "repo_status.txt").read_text() == "", "run repository was dirty")
    repo_diff = _read_single_line(provenance / "repo_diff.sha256", "repo diff digest").split()[0]
    _require(repo_diff == _EMPTY_SHA256, "run repository diff was nonempty")
    for name in ("allocated_gpu_uuids.csv", "visible_gpu_uuids.csv"):
        uuids = [value.strip() for value in _read_single_line(provenance / name, name).split(",")]
        _require(uuids == [run["gpu_uuid"]], f"{name} does not prove single-GPU ownership")

    source = _validate_manifest(
        provenance / "source_manifest.sha256",
        expected_digest=run["source_manifest_sha256"],
        relative_base=repo,
        label="source manifest",
    )
    final_source_path = provenance / "source_manifest.final.sha256"
    _require(final_source_path.is_file(), "missing final source manifest")
    _require(
        final_source_path.read_bytes() == (provenance / "source_manifest.sha256").read_bytes(),
        "source changed during the late-pair run",
    )
    inputs = _validate_manifest(
        provenance / "input_manifest.sha256",
        expected_digest=run["input_manifest_sha256"],
        relative_base=None,
        label="input manifest",
    )
    cuda_sha, cuda_path = _parse_sha_line(provenance / "qualified_cuda.sha256", "qualified CUDA binary", root=root)
    _require(cuda_sha == run["cuda_sha256"], "qualified CUDA digest differs from run.json")
    relion_sha, relion_path = _parse_sha_line(provenance / "relion_bind.sha256", "RELION binding", root=root)
    _require(relion_sha == run["relion_bind_sha256"], "RELION binding differs from run.json")
    interpreter_sha, interpreter_path = _parse_sha_line(
        provenance / "interpreter.sha256", "late-pair interpreter", root=repo
    )
    _require(interpreter_sha == run["interpreter_sha256"], "late-pair interpreter differs from run.json")
    resources = _validate_resource_files(
        provenance / "coarse_atomic_resources.json",
        provenance / "cuda_resource_usage.demangled.txt",
        cuda_sha256=run["cuda_sha256"],
        report_sha256=run["resource_report_sha256"],
        demangled_sha256=run["resource_demangled_sha256"],
        label="late pair",
    )

    qualified_root = Path(str(run.get("qualified_gpu_gate_root", ""))).resolve()
    qualified_gate = _validate_focused_gate(
        qualified_root,
        repo=repo,
        late_run=run,
    )

    _validate_junit(root / "focused_pytest.junit.xml")
    execution = _validate_execution_order(provenance / "execution_order.tsv", root)
    command_hashes = {}
    for label, workers, atomic, _ in ARM_SPECS:
        command_hashes[label] = _validate_command(
            provenance / f"{label}_command.sh",
            workers=workers,
            atomic=atomic,
            label=label,
        )
    run_dirs = {path.name for path in (root / "runs").iterdir() if path.is_dir()}
    _require(run_dirs == set(ARM_LABELS), f"run-directory topology differs: {sorted(run_dirs)}")

    return run, {
        "run_json_sha256": _sha256(run_path),
        "source_manifest": source,
        "input_manifest": inputs,
        "cuda_binary": {"path": str(cuda_path), "sha256": cuda_sha},
        "relion_binding": {"path": str(relion_path), "sha256": relion_sha},
        "interpreter": {"path": str(interpreter_path), "sha256": interpreter_sha},
        "resources": resources,
        "qualified_gpu_gate": qualified_gate,
        "git_repository": {
            "path": str(repo),
            "resolved_head": resolved_head,
            "resolved_tree": resolved_tree,
        },
        "execution_order": execution,
        "command_sha256": command_hashes,
    }


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
        f"{label} must be an integer",
    )
    return int(value)


def _coarse_translation_count(metadata: dict[str, Any], label: str) -> int:
    """Recover the pass-1 grid size from the serialized fine sampling plan."""

    fine_count = _integer(metadata.get("n_translations"), f"{label} n_translations")
    oversampling = _integer(metadata.get("oversampling"), f"{label} oversampling")
    _require(fine_count > 0, f"{label} n_translations must be positive")
    _require(oversampling >= 0, f"{label} oversampling must be non-negative")
    _require(oversampling <= 8, f"{label} oversampling is implausibly large")
    children_per_parent = 4**oversampling
    _require(
        fine_count % children_per_parent == 0,
        f"{label} fine translation count is not divisible by its oversampling factor",
    )
    return fine_count // children_per_parent


def _validate_selector_audit(
    audit: Any,
    *,
    workers: int,
    atomic: bool,
    translations: int,
    label: str,
) -> dict[str, Any]:
    _require(isinstance(audit, dict), f"{label} selector audit is missing")
    required = {
        "score_mode",
        "translation_count",
        "requested_fused",
        "effective_fused",
        "requested_workers",
        "effective_workers",
        "requested_atomic",
        "effective_atomic",
        "wrapper",
        "target",
        "counts",
    }
    _require(set(audit) == required, f"{label} selector audit fields differ")
    try:
        audit = _validate_coarse_selector_audit(audit)
    except (TypeError, ValueError) as exc:
        raise LatePairSetupError(f"{label} selector audit is invalid: {exc}") from exc
    multistream = workers == 8
    expected_wrapper = (
        "relion_coarse_diff2_projector_multistream_f32" if multistream else "relion_coarse_diff2_projector_f32"
    )
    expected_target = (
        "cuda_relion_coarse_diff2_projector_multistream_f32"
        if multistream
        else "cuda_relion_coarse_diff2_projector_f32"
    )
    observed = {
        "score_mode": audit["score_mode"],
        "translation_count": audit["translation_count"],
        "requested_fused": audit["requested_fused"],
        "effective_fused": audit["effective_fused"],
        "requested_workers": audit["requested_workers"],
        "effective_workers": audit["effective_workers"],
        "requested_atomic": audit["requested_atomic"],
        "effective_atomic": audit["effective_atomic"],
        "wrapper": audit["wrapper"],
        "target": audit["target"],
    }
    expected = {
        "score_mode": "gaussian",
        "translation_count": translations,
        "requested_fused": True,
        "effective_fused": True,
        "requested_workers": workers,
        "effective_workers": workers,
        "requested_atomic": atomic,
        "effective_atomic": atomic,
        "wrapper": expected_wrapper,
        "target": expected_target,
    }
    mismatches = {
        key: {"expected": value, "observed": observed[key]} for key, value in expected.items() if observed[key] != value
    }
    _require(not mismatches, f"{label} effective selector differs: {mismatches}")
    normalized = audit["counts"]
    fused_calls = normalized["fused_calls"]
    _require(fused_calls > 0, f"{label} fused selector recorded zero calls")
    _require(normalized["actual_rows"] >= fused_calls, f"{label} selector row count is invalid")
    _require(
        normalized["multistream_calls"] == (fused_calls if multistream else 0),
        f"{label} multistream execution count is invalid",
    )
    _require(
        normalized["native_atomic_selected_calls"] == (fused_calls if atomic else 0),
        f"{label} native-atomic execution count is invalid",
    )
    return {**observed, "counts": normalized}


def _load_star(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        value = starfile.read(path, always_dict=True)
    except Exception as exc:
        raise LatePairSetupError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict) and "particles" in value, f"{label} has no particles table")
    particles = value["particles"]
    _require(len(particles) > 0, f"{label} has no particle rows")
    for column in particles.columns:
        values = particles[column].to_numpy()
        _require(not _has_nonfinite_numeric(values), f"{label} column {column} is non-finite")
    _require("rlnImageName" in particles.columns, f"{label} lacks rlnImageName")
    image_names = [str(item).strip() for item in particles["rlnImageName"].to_numpy()]
    _require(all(image_names), f"{label} has an empty image identity")
    _require(len(set(image_names)) == len(image_names), f"{label} has duplicate image identities")
    return value


def _star_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        left_value, right_value = left[key], right[key]
        if hasattr(left_value, "columns") or hasattr(right_value, "columns"):
            if not hasattr(left_value, "columns") or not hasattr(right_value, "columns"):
                return False
            if list(left_value.columns) != list(right_value.columns):
                return False
            if not _values_equal(left_value.to_numpy(), right_value.to_numpy()):
                return False
        elif not _values_equal(left_value, right_value):
            return False
    return True


def _load_map(path: Path, label: str) -> np.ndarray:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        with mrcfile.open(path, permissive=False) as stream:
            value = np.asarray(stream.data, dtype=np.float64).copy()
    except Exception as exc:
        raise LatePairSetupError(f"cannot read {label}: {exc}") from exc
    _require(value.ndim == 3 and len(set(value.shape)) == 1, f"{label} is not a cubic map")
    _require(np.all(np.isfinite(value)), f"{label} contains non-finite values")
    return value


def _map_delta(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    _require(left.shape == right.shape, f"map shapes differ: {left.shape} != {right.shape}")
    delta = right - left
    left_norm = float(np.linalg.norm(left.reshape(-1)))
    right_norm = float(np.linalg.norm(right.reshape(-1)))
    denominator = math.sqrt(0.5 * (left_norm * left_norm + right_norm * right_norm))
    delta_norm = float(np.linalg.norm(delta.reshape(-1)))
    relative_l2 = delta_norm / denominator if denominator else (0.0 if delta_norm == 0.0 else math.inf)
    rms = float(np.sqrt(np.mean(np.square(delta))))
    signed_mean = float(np.mean(delta))
    return {
        "relative_l2": relative_l2,
        "max_abs": float(np.max(np.abs(delta))),
        "signed_mean": signed_mean,
        "signed_mean_over_delta_rms": signed_mean / rms if rms else 0.0,
        "relative_scale_drift": right_norm / left_norm - 1.0 if left_norm else (0.0 if right_norm == 0.0 else math.inf),
    }


def _numeric_scalars(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        parsed = float(item)
        _require(math.isfinite(parsed) and parsed >= 0.0, f"stage timing {key} is invalid")
        result[str(key)] = parsed
    return result


def _load_nsight_profile(root: Path, label: str) -> dict[str, Any]:
    """Verify a raw Nsight export against its summary and extract overlap-aware timings."""

    nsight_root = root / "nsight"
    summary_path = nsight_root / f"{label}_summary.json"
    sqlite_path = nsight_root / f"{label}.sqlite"
    _require(summary_path.is_file(), f"missing {label} Nsight summary: {summary_path}")
    _require(sqlite_path.is_file(), f"missing {label} Nsight SQLite export: {sqlite_path}")
    summary = _load_json(summary_path, f"{label} Nsight summary")
    _require(summary.get("schema") == NSIGHT_SCHEMA, f"{label} Nsight schema differs")
    recorded_sqlite = summary.get("sqlite")
    _require(isinstance(recorded_sqlite, str) and recorded_sqlite, f"{label} Nsight SQLite path is missing")
    resolved_sqlite = _resolved_inside(Path(recorded_sqlite), root, f"{label} Nsight SQLite path")
    _require(resolved_sqlite == sqlite_path.resolve(), f"{label} Nsight SQLite path differs")

    sqlite_sha256 = _sha256(sqlite_path)
    try:
        connection = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            tables = nsys_sqlite._tables(connection)
            kernel_table = "CUPTI_ACTIVITY_KIND_KERNEL"
            _require(kernel_table in tables, f"{label} Nsight SQLite export has no kernel table")
            columns = nsys_sqlite._columns(connection, kernel_table)
            required_columns = {"start", "end", "deviceId"}
            _require(
                required_columns.issubset(columns),
                f"{label} Nsight kernel table lacks columns {sorted(required_columns - columns)}",
            )
            strings = nsys_sqlite._string_ids(connection, tables)
            rows = list(connection.execute(f'SELECT * FROM "{kernel_table}"'))
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise LatePairSetupError(f"cannot read {label} Nsight SQLite export: {exc}") from exc
    _require(rows, f"{label} Nsight SQLite export has no CUDA kernels")
    _require(_sha256(sqlite_path) == sqlite_sha256, f"{label} Nsight SQLite export changed during analysis")

    intervals_by_device: dict[int, list[tuple[int, int]]] = {}
    coarse_intervals: list[tuple[int, int]] = []
    coarse_sum_ns = 0
    for row in rows:
        start, end = int(row["start"]), int(row["end"])
        _require(end >= start, f"{label} Nsight kernel interval has negative duration")
        device = int(row["deviceId"])
        interval = (start, end)
        intervals_by_device.setdefault(device, []).append(interval)
        name = nsys_sqlite._name(
            row,
            candidates=("shortName", "demangledName", "mangledName", "name"),
            strings=strings,
        )
        if name == COARSE_KERNEL_NAME:
            coarse_intervals.append(interval)
            coarse_sum_ns += end - start

    _require(len(intervals_by_device) == 1, f"{label} raw Nsight device topology differs")
    _require(coarse_intervals, f"{label} raw Nsight export has no {COARSE_KERNEL_NAME}")
    device_id, all_intervals = next(iter(intervals_by_device.items()))
    total_union_ns = nsys_sqlite._union_ns(all_intervals)
    coarse_union_ns = nsys_sqlite._union_ns(coarse_intervals)
    _require(total_union_ns > 0, f"{label} raw total GPU interval union is invalid")
    _require(coarse_union_ns > 0, f"{label} raw coarse GPU interval union is invalid")

    devices = summary.get("devices")
    _require(isinstance(devices, dict) and set(devices) == {str(device_id)}, f"{label} Nsight device topology differs")
    device = devices[str(device_id)]
    _require(isinstance(device, dict), f"{label} Nsight device summary is invalid")
    _require(
        device.get("kernel_count") == len(all_intervals),
        f"{label} Nsight summary kernel count differs from raw SQLite",
    )
    _require(
        device.get("gpu_busy_ns") == total_union_ns,
        f"{label} Nsight summary GPU union differs from raw SQLite",
    )
    kernels = summary.get("kernels")
    _require(isinstance(kernels, list), f"{label} Nsight kernel summary is missing")
    coarse_rows = [row for row in kernels if isinstance(row, dict) and row.get("name") == COARSE_KERNEL_NAME]
    _require(len(coarse_rows) == 1, f"{label} Nsight coarse kernel summary row differs")
    coarse_row = coarse_rows[0]
    _require(
        coarse_row.get("count") == len(coarse_intervals),
        f"{label} Nsight coarse launch count differs from raw SQLite",
    )
    _require(
        coarse_row.get("total_ns") == coarse_sum_ns,
        f"{label} Nsight coarse kernel sum differs from raw SQLite",
    )
    return {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "sqlite_path": str(sqlite_path.resolve()),
        "sqlite_sha256": sqlite_sha256,
        "total_gpu_interval_union_s": total_union_ns / 1e9,
        "coarse_kernel_sum_s": coarse_sum_ns / 1e9,
        "coarse_kernel_interval_union_s": coarse_union_ns / 1e9,
        "coarse_kernel_launch_count": len(coarse_intervals),
        "summary_crosscheck_exact": True,
    }


def _load_arm(root: Path, spec: tuple[str, int, bool, int]) -> dict[str, Any]:
    label, workers, atomic, repeat = spec
    profile_root = root / "runs" / label / "profile"
    summary_path = profile_root / "profile_summary.json"
    summary = _load_json(summary_path, f"{label} profile summary")
    expected_summary = {
        "schema": PROFILE_SCHEMA,
        "classification": "diagnostic_performance_only",
        "checkpoint_iteration": PROFILED_ITERATION - 1,
        "profiled_iteration": PROFILED_ITERATION,
        "nr_iter_schedule": 200,
        "exact_local_bucket_radix": 4,
        "exact_local_physical_order_chunk_size": 0,
        "cuda_profiler_range": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": summary.get(key)}
        for key, expected in expected_summary.items()
        if summary.get(key) != expected
    }
    _require(not mismatches, f"{label} profile summary differs: {mismatches}")
    _require(set(summary).issuperset({"cold", "warm"}), f"{label} profile phases are incomplete")
    warm = summary["warm"]
    _require(isinstance(warm, dict), f"{label} warm profile is invalid")
    warm_root = profile_root / "warm"
    meta_path = warm_root / f"run_it{PROFILED_ITERATION:03d}_recovar_meta.json"
    recorded_meta = _resolved_inside(Path(str(warm.get("meta_path", ""))), root, f"{label} warm metadata")
    _require(recorded_meta == meta_path.resolve(), f"{label} warm metadata path differs")
    _require(_sha256(meta_path) == warm.get("meta_sha256"), f"{label} warm metadata digest differs")
    metadata = _load_json(meta_path, f"{label} warm metadata")
    translations = _coarse_translation_count(metadata, label)
    _require(
        metadata.get("joint_halfset_particle_stream") is True,
        f"{label} did not record the joint-halfset particle stream",
    )
    _require(
        _values_equal(metadata.get("halfset_ids"), [0, 1]),
        f"{label} joint-halfset IDs differ from [0, 1]",
    )
    profile_keys = sorted(key for key in metadata if _HALFSET_PROFILE_RE.fullmatch(str(key)))
    _require(
        profile_keys == ["halfset_0_profile_summary"],
        f"{label} joint-halfset profile topology differs: {profile_keys}",
    )
    selector_audits = []
    for key in profile_keys:
        halfset_profile = metadata[key]
        _require(isinstance(halfset_profile, dict), f"{label} {key} is not a profile mapping")
        selector_audits.append(
            {
                "halfset": int(_HALFSET_PROFILE_RE.fullmatch(key).group("halfset")),
                "halfset_ids": [0, 1],
                "audit": _validate_selector_audit(
                    halfset_profile.get("coarse_selector_audit"),
                    workers=workers,
                    atomic=atomic,
                    translations=translations,
                    label=f"{label} {key}",
                ),
            }
        )
    extra_meta = sorted(warm_root.glob("run_it*_recovar_meta.json"))
    _require(extra_meta == [meta_path], f"{label} warm metadata topology differs")

    wall_s = warm.get("wall_s")
    _require(isinstance(wall_s, (int, float)) and not isinstance(wall_s, bool), f"{label} warm wall is invalid")
    wall_s = float(wall_s)
    _require(math.isfinite(wall_s) and wall_s > 0.0, f"{label} warm wall is invalid")
    iteration_stages = _numeric_scalars(warm.get("iteration_profile"))
    _require("expectation_time_s" in iteration_stages, f"{label} has no warm expectation timing")
    sparse_stages = _numeric_scalars(metadata.get("sparse_pass2_profile_summary"))
    _require("pass1_time_s" in sparse_stages, f"{label} has no warm pass-1 timing")
    _require("pass2_time_s" in sparse_stages, f"{label} has no warm pass-2 timing")
    halfset_stages = {key: _numeric_scalars(metadata[key]) for key in profile_keys}
    nsight = _load_nsight_profile(root, label)

    star_path = warm_root / f"run_it{PROFILED_ITERATION:03d}_data.star"
    map_path = warm_root / f"run_it{PROFILED_ITERATION:03d}_class001.mrc"
    return {
        "label": label,
        "configuration": label.rsplit("_", 1)[0],
        "repeat": repeat,
        "workers": workers,
        "atomic": atomic,
        "profile_summary_sha256": _sha256(summary_path),
        "meta_sha256": _sha256(meta_path),
        "selector_audits": selector_audits,
        "metadata": metadata,
        "star": _load_star(star_path, f"{label} warm particle STAR"),
        "star_sha256": _sha256(star_path),
        "map": _load_map(map_path, f"{label} warm map"),
        "map_sha256": _sha256(map_path),
        "performance": {
            "warm_wall_s": wall_s,
            "warm_expectation_s": iteration_stages["expectation_time_s"],
            "pass1_s": sparse_stages["pass1_time_s"],
            "pass2_s": sparse_stages["pass2_time_s"],
            "iteration_stages_s": iteration_stages,
            "sparse_stages_s": sparse_stages,
            "halfset_stages_s": halfset_stages,
            "gpu_kernel_union_s": nsight["total_gpu_interval_union_s"],
            "coarse_kernel_sum_s": nsight["coarse_kernel_sum_s"],
            "coarse_kernel_interval_union_s": nsight["coarse_kernel_interval_union_s"],
            "coarse_kernel_launch_count": nsight["coarse_kernel_launch_count"],
            "nsight_summary_sha256": nsight["summary_sha256"],
            "nsight": nsight,
        },
    }


def _paired_science(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparisons = {}
    all_exact = True
    maps_by_config = {config: [arms[f"{config}_1"]["map"], arms[f"{config}_2"]["map"]] for config in CONFIGURATIONS}
    shapes = {tuple(value.shape) for values in maps_by_config.values() for value in values}
    _require(len(shapes) == 1, f"warm maps have different shapes: {sorted(shapes)}")
    repeat_deltas = {config: _map_delta(values[0], values[1]) for config, values in maps_by_config.items()}
    repeat_envelope = max(row["relative_l2"] for row in repeat_deltas.values())
    _require(math.isfinite(repeat_envelope), "map repeat envelope is non-finite")
    paired_map_rows = {}
    all_maps_bounded = True
    for repeat in (1, 2):
        control_label = f"canonical_serial_{repeat}"
        control = arms[control_label]
        for config in CONFIGURATIONS[1:]:
            candidate_label = f"{config}_{repeat}"
            candidate = arms[candidate_label]
            meta_exact = {
                key: (
                    key in control["metadata"]
                    and key in candidate["metadata"]
                    and _values_equal(control["metadata"][key], candidate["metadata"][key])
                )
                for key in DISCRETE_META_KEYS
            }
            star_exact = _star_equal(control["star"], candidate["star"])
            pair_exact = all(meta_exact.values()) and star_exact
            all_exact &= pair_exact
            map_delta = _map_delta(control["map"], candidate["map"])
            map_delta["repeat_envelope_relative_l2"] = repeat_envelope
            map_delta["within_repeat_envelope"] = bool(
                map_delta["relative_l2"] <= np.nextafter(repeat_envelope, math.inf)
                if repeat_envelope > 0.0
                else map_delta["relative_l2"] == 0.0
            )
            all_maps_bounded &= map_delta["within_repeat_envelope"]
            key = f"{control_label}__{candidate_label}"
            comparisons[key] = {
                "metadata_exact": meta_exact,
                "particle_star_exact": star_exact,
                "all_discrete_exact": pair_exact,
                "map": map_delta,
            }
            paired_map_rows[key] = map_delta
    return {
        "discrete_meta_keys": list(DISCRETE_META_KEYS),
        "all_factorial_discrete_and_star_exact": all_exact,
        "repeat_map_deltas": repeat_deltas,
        "repeat_relative_l2_envelope": repeat_envelope,
        "paired_factorial_comparisons": comparisons,
        "all_factorial_maps_within_repeat_envelope": all_maps_bounded,
        "candidate_pairs": {key: value for key, value in comparisons.items() if "__atomic_multistream_" in key},
        "pass": all_exact and all_maps_bounded,
    }


def _percent_change(candidate: float, control: float) -> float:
    _require(math.isfinite(candidate) and candidate > 0.0, "candidate runtime is invalid")
    _require(math.isfinite(control) and control > 0.0, "control runtime is invalid")
    return 100.0 * (candidate / control - 1.0)


def _summarize_performance(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    medians = {}
    profiling_configurations = {}
    for config in CONFIGURATIONS:
        rows = [arms[f"{config}_{repeat}"]["performance"] for repeat in (1, 2)]
        raw = {f"R{repeat}": {name: rows[repeat - 1][name] for name in PROFILE_METRICS} for repeat in (1, 2)}
        values = {name: float(median([row[name] for row in rows])) for name in PROFILE_METRICS}
        sparse_names = sorted(set(rows[0]["sparse_stages_s"]).intersection(rows[1]["sparse_stages_s"]))
        values["sparse_stages_s"] = {
            name: float(median([row["sparse_stages_s"][name] for row in rows])) for name in sparse_names
        }
        medians[config] = values
        repeat_span = {name: abs(float(raw["R2"][name]) - float(raw["R1"][name])) for name in PROFILE_METRICS}
        repeat_span_percent = {
            name: 100.0 * repeat_span[name] / values[name] if values[name] else 0.0 for name in PROFILE_METRICS
        }
        profiling_configurations[config] = {
            "raw": raw,
            "median": {name: values[name] for name in PROFILE_METRICS},
            "repeat_span": repeat_span,
            "repeat_span_percent": repeat_span_percent,
        }
    control = medians["canonical_serial"]
    changes = {}
    for config, values in medians.items():
        changes[config] = {}
        for name in PROFILE_METRICS:
            changes[config][name] = _percent_change(values[name], control[name])
        profiling_configurations[config]["percent_change_vs_canonical_serial"] = dict(changes[config])
    candidate_wall_change = changes["atomic_multistream"]["warm_wall_s"]
    material_wall_win = bool(candidate_wall_change <= MATERIAL_WALL_WIN_PERCENT)
    profiling_arms = {}
    for label, arm in arms.items():
        performance = arm["performance"]
        nsight = performance["nsight"]
        profiling_arms[label] = {
            **{name: performance[name] for name in PROFILE_METRICS},
            "nsight_sqlite": {
                "path": nsight["sqlite_path"],
                "sha256": nsight["sqlite_sha256"],
                "summary_path": nsight["summary_path"],
                "summary_sha256": nsight["summary_sha256"],
                "total_gpu_union_crosscheck_exact": nsight["summary_crosscheck_exact"],
            },
        }
    return {
        "arms": {label: arm["performance"] for label, arm in arms.items()},
        "configuration_medians": medians,
        "percent_change_vs_canonical_serial": changes,
        "profiling": {
            "metric_order": list(PROFILE_METRICS),
            "arms": profiling_arms,
            "configurations": profiling_configurations,
            "all_sqlite_summary_crosschecks_exact": all(
                arm["nsight_sqlite"]["total_gpu_union_crosscheck_exact"] for arm in profiling_arms.values()
            ),
        },
        "material_warm_wall_threshold_percent": MATERIAL_WALL_WIN_PERCENT,
        "atomic_multistream_material_warm_wall_win": material_wall_win,
        "pass": material_wall_win,
    }


def _markdown(report: dict[str, Any]) -> str:
    performance = report["performance"]
    science = report["science"]
    acceptance = report["acceptance"]
    dashboard = report["dashboard"]
    profiling = performance["profiling"]
    overall = dashboard["overall_status"]

    def mark(value: bool) -> str:
        return "PASS" if value else "FAIL"

    def metric_value(name: str, value: float) -> str:
        return f"{value:.0f}" if name == "coarse_kernel_launch_count" else f"{value:.6f}"

    metric_labels = {
        "warm_wall_s": "warm wall (s)",
        "warm_expectation_s": "expectation (s)",
        "pass1_s": "pass 1 (s)",
        "pass2_s": "pass 2 (s)",
        "gpu_kernel_union_s": "total GPU union (s)",
        "coarse_kernel_sum_s": "coarse kernel sum (s)",
        "coarse_kernel_interval_union_s": "coarse interval union (s)",
        "coarse_kernel_launch_count": "coarse launches",
    }
    candidate_wall_change = performance["percent_change_vs_canonical_serial"]["atomic_multistream"]["warm_wall_s"]
    rows = [
        f"# OVERALL: {overall} — VDAM coarse multistream one-transition gate",
        "",
        f"> **One-transition gate: {overall}. The candidate remains default-off.**",
        "",
        "- Scope: `diagnostic_performance_only`, covering only the frozen iteration 180 → 181 transition.",
        "- This report cannot promote scientific correctness or change a production default.",
        f"- `one_transition_gate_pass={str(dashboard['one_transition_gate_pass']).lower()}`",
        "- `long_trajectory_no_growth_evaluated=false`",
        "- `default_enablement_allowed=false`",
        "",
        "## Gate dashboard",
        "",
        "| Gate | Requirement | Result | Status |",
        "|---|---|---|---:|",
        "| Topology and sealed provenance | Exact | Verified | PASS |",
        f"| Effective selector audits | 8/8 exact | {acceptance['selector_audit_count']}/8 | "
        f"{mark(acceptance['effective_selector_audits'])} |",
        f"| Raw SQLite ↔ Nsight summary GPU union | 8/8 exact | "
        f"{sum(arm['nsight_sqlite']['total_gpu_union_crosscheck_exact'] for arm in profiling['arms'].values())}/8 | "
        f"{mark(profiling['all_sqlite_summary_crosschecks_exact'])} |",
        f"| Discrete state and particle STAR | All six pairs exact | "
        f"{science['all_factorial_discrete_and_star_exact']} | "
        f"{mark(acceptance['exact_discrete_and_star_parity'])} |",
        f"| Map arithmetic | All six pairs ≤ repeat envelope | "
        f"{science['all_factorial_maps_within_repeat_envelope']} | "
        f"{mark(acceptance['map_deltas_within_repeat_envelope'])} |",
        f"| Atomic + 8-stream warm wall | ≤ {MATERIAL_WALL_WIN_PERCENT:+.1f}% vs canonical serial | "
        f"{candidate_wall_change:+.2f}% | {mark(acceptance['material_warm_wall_win'])} |",
        f"| One-transition gate | Every gate above passes | {acceptance['one_transition_gate_pass']} | "
        f"{mark(acceptance['one_transition_gate_pass'])} |",
        "| Long-trajectory no-growth | Required before enablement | Not evaluated | OPEN |",
        "| Default enablement | Requires long-trajectory qualification | `false` | BLOCKED |",
        "",
        "## Median performance",
        "",
        "| Configuration | Warm wall (s) | Δ wall | Expectation (s) | Pass 1 (s) | Pass 2 (s) | "
        "Total GPU union (s) | Coarse sum (s) | Coarse union (s) | Coarse launches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    medians = performance["configuration_medians"]
    changes = performance["percent_change_vs_canonical_serial"]
    for config in CONFIGURATIONS:
        values = medians[config]
        rows.append(
            f"| {config} | {values['warm_wall_s']:.6f} | {changes[config]['warm_wall_s']:+.2f}% | "
            f"{values['warm_expectation_s']:.6f} | {values['pass1_s']:.6f} | {values['pass2_s']:.6f} | "
            f"{values['gpu_kernel_union_s']:.6f} | {values['coarse_kernel_sum_s']:.6f} | "
            f"{values['coarse_kernel_interval_union_s']:.6f} | "
            f"{values['coarse_kernel_launch_count']:.0f} |"
        )
    rows.extend(
        (
            "",
            "## Repeat stability (raw R1/R2)",
            "",
            "| Configuration | Metric | R1 | R2 | Median | Repeat span | Span / median | Δ vs canonical |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for config in CONFIGURATIONS:
        stats = profiling["configurations"][config]
        for name in PROFILE_METRICS:
            rows.append(
                f"| {config} | {metric_labels[name]} | {metric_value(name, stats['raw']['R1'][name])} | "
                f"{metric_value(name, stats['raw']['R2'][name])} | "
                f"{metric_value(name, stats['median'][name])} | "
                f"{metric_value(name, stats['repeat_span'][name])} | "
                f"{stats['repeat_span_percent'][name]:.3f}% | "
                f"{stats['percent_change_vs_canonical_serial'][name]:+.2f}% |"
            )
    rows.extend(
        (
            "",
            "### Map repeat envelope",
            "",
            "| Configuration | R1 ↔ R2 rel-L2 | Max abs | Signed mean | Relative scale drift |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for config in CONFIGURATIONS:
        value = science["repeat_map_deltas"][config]
        rows.append(
            f"| {config} | {value['relative_l2']:.3e} | {value['max_abs']:.3e} | "
            f"{value['signed_mean']:.3e} | {value['relative_scale_drift']:.3e} |"
        )
    rows.extend(
        (
            "",
            f"Observed repeat rel-L2 envelope: `{science['repeat_relative_l2_envelope']:.6e}`.",
            "",
            "## All six factorial numerical comparisons",
            "",
            "| Control → factorial arm | Discrete/STAR exact | Map rel-L2 | Max abs | Signed mean | "
            "Signed mean / RMS | Scale drift | Repeat envelope | Bounded |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for key, value in science["paired_factorial_comparisons"].items():
        map_row = value["map"]
        rows.append(
            f"| {key} | {value['all_discrete_exact']} | {map_row['relative_l2']:.3e} | "
            f"{map_row['max_abs']:.3e} | {map_row['signed_mean']:.3e} | "
            f"{map_row['signed_mean_over_delta_rms']:.3e} | {map_row['relative_scale_drift']:.3e} | "
            f"{map_row['repeat_envelope_relative_l2']:.3e} | {map_row['within_repeat_envelope']} |"
        )
    provenance = report["provenance"]
    qualified = provenance["qualified_gpu_gate"]
    rows.extend(
        (
            "",
            "## Compact provenance",
            "",
            "| Evidence | Sealed value |",
            "|---|---|",
            f"| Result root | `{provenance['root']}` |",
            f"| Slurm job | `{provenance['job_id']}` |",
            f"| Git commit / tree | `{provenance['git_head']}` / `{provenance['git_tree']}` |",
            f"| Node / GPU | `{provenance['node']}` / `{provenance['gpu_name']}` |",
            f"| GPU UUID | `{provenance['gpu_uuid']}` |",
            f"| CUDA binary SHA-256 | `{provenance['cuda_sha256']}` |",
            f"| Source / input manifests | `{provenance['source_manifest_sha256']}` / "
            f"`{provenance['input_manifest_sha256']}` |",
            f"| Qualified focused GPU gate | job `{qualified['job_id']}`, `{qualified['root']}` |",
            f"| Analyzer SHA-256 | `{provenance['analyzer_source_sha256']}` |",
            "",
            "### Nsight SQLite seals",
            "",
            "Each path below is absolute in `report.json`; the compact path is relative to the sealed result root.",
            "",
            "| Arm | SQLite | SQLite SHA-256 | Summary SHA-256 | Union cross-check |",
            "|---|---|---|---|---:|",
        )
    )
    root_path = Path(provenance["root"])
    for label in ARM_LABELS:
        seal = profiling["arms"][label]["nsight_sqlite"]
        compact_path = Path(seal["path"]).relative_to(root_path)
        rows.append(
            f"| {label} | `{compact_path}` | `{seal['sha256']}` | `{seal['summary_sha256']}` | "
            f"{seal['total_gpu_union_crosscheck_exact']} |"
        )
    transition_conclusion = (
        "This run passes only the bounded one-transition gate; it does not qualify default enablement."
        if acceptance["one_transition_gate_pass"]
        else "This frozen result does not pass even the one-transition gate."
    )
    rows.extend(
        (
            "",
            "## Promotion boundary",
            "",
            "The atomic + 8-stream selector remains **default-off**. Passing this one-transition diagnostic would not "
            "establish long-trajectory numerical no-growth, basin preservation across a full run, or production "
            f"enablement. {transition_conclusion}",
            "",
        )
    )
    return "\n".join(rows)


def analyze(root: Path, *, repo: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    repo = (repo or Path(__file__).resolve().parents[1]).resolve()
    run, provenance_details = _validate_provenance(root, repo)
    nsight_summaries = {path.name.removesuffix("_summary.json") for path in (root / "nsight").glob("*_summary.json")}
    nsight_sqlites = {path.stem for path in (root / "nsight").glob("*.sqlite")}
    _require(
        nsight_summaries == set(ARM_LABELS) and nsight_sqlites == set(ARM_LABELS),
        "Nsight summaries/SQLite exports are only partially present or have unexpected arms",
    )
    arms = {spec[0]: _load_arm(root, spec) for spec in ARM_SPECS}
    science = _paired_science(arms)
    performance = _summarize_performance(arms)
    selector_count = sum(len(arm["selector_audits"]) for arm in arms.values())
    one_transition_gate_pass = science["pass"] and performance["pass"]
    report = {
        "schema": SCHEMA,
        "provenance": {
            "root": str(root),
            "job_id": run["job_id"],
            "git_head": run["git_head"],
            "git_tree": run["git_tree"],
            "gpu_uuid": run["gpu_uuid"],
            "gpu_name": run["gpu_name"],
            "node": run["node"],
            "cuda_sha256": run["cuda_sha256"],
            "source_manifest_sha256": run["source_manifest_sha256"],
            "input_manifest_sha256": run["input_manifest_sha256"],
            "analyzer_source_sha256": _sha256(Path(__file__).resolve()),
            **provenance_details,
        },
        "selector_audits": {label: arm["selector_audits"] for label, arm in arms.items()},
        "science": science,
        "performance": performance,
        "acceptance": {
            "topology_and_provenance": True,
            "effective_selector_audits": True,
            "selector_audit_count": selector_count,
            "exact_discrete_and_star_parity": science["all_factorial_discrete_and_star_exact"],
            "map_deltas_within_repeat_envelope": science["all_factorial_maps_within_repeat_envelope"],
            "material_warm_wall_win": performance["atomic_multistream_material_warm_wall_win"],
            "one_transition_gate_pass": one_transition_gate_pass,
            "long_trajectory_no_growth_evaluated": False,
            "default_enablement_allowed": False,
            "pass": one_transition_gate_pass,
        },
        "dashboard": {
            "overall_status": "PASS" if one_transition_gate_pass else "FAIL",
            "scope": "diagnostic_performance_only",
            "transition": f"iteration_{PROFILED_ITERATION - 1}_to_{PROFILED_ITERATION}",
            "candidate": "atomic_multistream",
            "candidate_default_state": "off",
            "one_transition_gate_pass": one_transition_gate_pass,
            "long_trajectory_no_growth_evaluated": False,
            "default_enablement_allowed": False,
        },
    }
    report["markdown"] = _markdown(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze(args.root, repo=args.repo)
    except LatePairSetupError as exc:
        parser.error(str(exc))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(report)
    markdown = serializable.pop("markdown")
    args.output_json.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(markdown)
    print(markdown, end="")
    return 0 if report["acceptance"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
