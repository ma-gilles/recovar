#!/usr/bin/env python3
"""Qualify the fused GF46 K=1 certificate-state scorer on one H100.

This is deliberately an analysis-first harness.  It lowers and compiles from
``ShapeDtypeStruct`` inputs, persists the complete XLA memory/HLO evidence,
and only then allocates the exact full-size deterministic operands.  A runtime
allocation or execution failure therefore cannot erase the mandatory static
memory evidence and is never hidden by reducing the production geometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Sequence


class GF46Geometry(NamedTuple):
    batch_size: int
    translation_count: int
    pixel_count: int
    rotation_block: int
    total_rotations: int
    source_rotation_block: int


GF46_GEOMETRY = GF46Geometry(
    batch_size=500,
    translation_count=29,
    pixel_count=5100,
    rotation_block=4608,
    total_rotations=36864,
    source_rotation_block=16,
)
EXPECTED_DOT_COUNT = 2
EXPECTED_LOGICAL_ARGUMENT_BYTES = 1_428_827_344

SOURCE_FILES = (
    "recovar/em/dense_single_volume/helpers/coarse_gemm_hybrid.py",
    "recovar/em/dense_single_volume/helpers/scoring.py",
    "scripts/qualify_vdam_gf46_certificate_state_h100.py",
    "scripts/run_vdam_gf46_certificate_state_h100.sbatch",
    "scripts/vdam_gpu_selection.sh",
    "tests/unit/initial_model/test_vdam_gf46_certificate_state_h100.py",
    "pixi.lock",
    "pixi.toml",
    "pyproject.toml",
)

REQUIRED_MEMORY_FIELDS = (
    "generated_code_size_in_bytes",
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
    "peak_memory_in_bytes",
    "host_generated_code_size_in_bytes",
    "host_argument_size_in_bytes",
    "host_output_size_in_bytes",
    "host_alias_size_in_bytes",
    "host_temp_size_in_bytes",
)

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GPU_UUID_RE = re.compile(r"GPU-[0-9a-fA-F-]{36}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _assert_exact_geometry(geometry: GF46Geometry = GF46_GEOMETRY) -> None:
    expected = GF46Geometry(500, 29, 5100, 4608, 36864, 16)
    if geometry != expected:
        raise RuntimeError(f"GF46 geometry drifted: expected={expected}, observed={geometry}")
    if geometry.rotation_block % geometry.source_rotation_block:
        raise RuntimeError("rotation block is not source-16 aligned")
    if geometry.total_rotations % geometry.rotation_block:
        raise RuntimeError("total rotations are not exactly tiled by the rotation block")
    if geometry.total_rotations // geometry.rotation_block != 8:
        raise RuntimeError("GF46 must contain exactly eight 4608-rotation blocks")


def _source_manifest(repo_root: Path) -> tuple[bytes, list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    lines: list[str] = []
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"source-manifest file is missing: {relative}")
        digest = _sha256_file(path)
        entries.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
        lines.append(f"{digest}  {relative}\n")
    return "".join(lines).encode(), entries


def _repository_provenance(
    repo_root: Path,
    *,
    expected_head: str,
    expected_tree: str,
) -> dict[str, Any]:
    if not _SHA1_RE.fullmatch(expected_head):
        raise ValueError("expected repository head must be a lowercase 40-hex commit")
    if not _SHA1_RE.fullmatch(expected_tree):
        raise ValueError("expected repository tree must be a lowercase 40-hex tree")
    observed_root = Path(_run(["git", "rev-parse", "--show-toplevel"], cwd=repo_root).strip())
    if observed_root.resolve() != repo_root:
        raise RuntimeError(f"repository root mismatch: {observed_root} != {repo_root}")
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root).strip()
    tree = _run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root).strip()
    status = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
    ).splitlines()
    if head != expected_head:
        raise RuntimeError(f"repository head mismatch: expected={expected_head}, observed={head}")
    if tree != expected_tree:
        raise RuntimeError(f"repository tree mismatch: expected={expected_tree}, observed={tree}")
    if status:
        raise RuntimeError(f"qualification requires a clean repository: {status}")
    branch_result = subprocess.run(
        ["git", "symbolic-ref", "--short", "-q", "HEAD"],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if branch_result.returncode not in (0, 1):
        raise RuntimeError(f"failed to resolve repository branch: {branch_result.stderr.strip()}")
    branch = branch_result.stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "root": str(repo_root),
        "head": head,
        "tree": tree,
        "branch": branch or None,
        "status_porcelain_v1": status,
        "head_diff_sha256": _sha256_bytes(diff),
    }


def _operand_generator_manifest(
    resolved_coefficients: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Describe every dynamic buffer without constructing full GF46 arrays."""

    g = GF46_GEOMETRY
    blocks = g.total_rotations // g.source_rotation_block
    energy = g.pixel_count * ((0.125**2 + 0.0625**2) * 0.75)
    entries = [
        ("state.raw_block_lower_max", (g.batch_size, blocks), "float64", "-1e100"),
        ("state.raw_block_upper_max", (g.batch_size, blocks), "float64", "-1e100"),
        ("state.posterior_block_lower_max", (g.batch_size, blocks), "float64", "-1e100"),
        ("state.posterior_block_upper_max", (g.batch_size, blocks), "float64", "-1e100"),
        ("state.rotation_visit_count", (g.total_rotations,), "int32", "0"),
        ("state.candidate_count", (g.batch_size,), "int32", "0"),
        ("state.invalid_candidate_count", (g.batch_size,), "int32", "0"),
        ("projected_reference", (g.rotation_block, g.pixel_count), "complex64", "0.03125-0.015625j"),
        (
            "image_batch.weighted_shifted",
            (g.batch_size * g.translation_count, g.pixel_count),
            "complex128",
            "0.09375+0.046875j",
        ),
        ("image_batch.pixel_weight", (g.batch_size, g.pixel_count), "float64", "0.75"),
        ("image_batch.image_energy", (g.batch_size, g.translation_count), "float64", repr(energy)),
        ("image_batch.initial_diff2", (g.batch_size,), "float64", "0.125"),
        ("image_batch.image_component_abs_max", (g.batch_size,), "float64", "0.125"),
        ("image_batch.weight_max", (g.batch_size,), "float64", "0.75"),
        ("image_batch.active", (g.batch_size,), "bool", "true"),
        ("image_batch.stored_inputs_valid", (g.batch_size,), "bool", "true"),
        ("image_batch.actual_image_count", (), "int32", str(g.batch_size)),
        ("class_log_prior", (), "float32", "-0.25"),
        ("rotation_log_prior", (g.rotation_block,), "float32", "-0.03125"),
        ("translation_log_prior", (g.batch_size, g.translation_count), "float32", "-0.015625"),
        ("cross_gamma", (), "float64", "certified_f64_expanded_score_gammas(5100).cross"),
        ("energy_gamma", (), "float64", "certified_f64_expanded_score_gammas(5100).energy"),
        (
            "initial_diff2_gamma",
            (),
            "float64",
            "certified_f64_expanded_score_gammas(5100).initial_diff2",
        ),
        (
            "energy_envelope_gamma",
            (),
            "float64",
            "certified_f64_expanded_score_gammas(5100).energy_envelope",
        ),
        ("direct_gamma", (), "float64", "certified_f32_dot_product_gamma(5100,29)"),
        ("traversed_full_position_count", (), "int32", str(g.pixel_count)),
        ("rotation_offset", (), "int32", "0"),
    ]
    dtype_bytes = {
        "bool": 1,
        "int32": 4,
        "float32": 4,
        "float64": 8,
        "complex64": 8,
        "complex128": 16,
    }
    coefficient_names = {
        "cross_gamma",
        "energy_gamma",
        "initial_diff2_gamma",
        "energy_envelope_gamma",
        "direct_gamma",
    }
    if resolved_coefficients is not None:
        if set(resolved_coefficients) != coefficient_names:
            raise RuntimeError(f"certificate coefficient names drifted: {sorted(resolved_coefficients)}")
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in resolved_coefficients.values()):
            raise RuntimeError(f"certificate coefficients are not finite and nonnegative: {resolved_coefficients}")
    buffers = []
    logical_bytes = 0
    for name, shape, dtype, fill in entries:
        element_count = math.prod(shape) if shape else 1
        nbytes = element_count * dtype_bytes[dtype]
        logical_bytes += nbytes
        if dtype == "bool" or "certified_" in fill:
            finite_by_construction = True
        else:
            literal = complex(fill)
            finite_by_construction = math.isfinite(literal.real) and math.isfinite(literal.imag)
        buffers.append(
            {
                "name": name,
                "shape": list(shape),
                "dtype": dtype,
                "generator": "jax.numpy.asarray" if not shape else "jax.numpy.full",
                "fill": fill,
                "finite_by_construction": finite_by_construction,
                "logical_size_bytes": nbytes,
            }
        )
    payload = {
        "schema": "recovar.vdam.gf46_certificate_state_operands.v1",
        "geometry": GF46_GEOMETRY._asdict(),
        "construction": "constant device fills; no pseudorandom generator",
        "buffers": buffers,
        "logical_dynamic_argument_bytes": logical_bytes,
        "resolved_coefficients": resolved_coefficients,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["generator_manifest_sha256"] = _sha256_bytes(encoded)
    return payload


def _named_argument_leaves(arguments: Sequence[Any]) -> list[tuple[str, Any]]:
    if len(arguments) != 13:
        raise RuntimeError(f"fused scorer argument count drifted: {len(arguments)}")
    state, projected, image_batch, *tail = arguments
    tail_names = (
        "class_log_prior",
        "rotation_log_prior",
        "translation_log_prior",
        "cross_gamma",
        "energy_gamma",
        "initial_diff2_gamma",
        "energy_envelope_gamma",
        "direct_gamma",
        "traversed_full_position_count",
        "rotation_offset",
    )
    if len(tail) != len(tail_names):
        raise RuntimeError("fused scorer scalar/prior signature drifted")
    state_fields = tuple(getattr(state, "_fields", ()))
    batch_fields = tuple(getattr(image_batch, "_fields", ()))
    if len(state_fields) != len(state) or len(batch_fields) != len(image_batch):
        raise RuntimeError("fused scorer named-tuple input structure drifted")
    return [
        *((f"state.{name}", value) for name, value in zip(state_fields, state, strict=True)),
        ("projected_reference", projected),
        *((f"image_batch.{name}", value) for name, value in zip(batch_fields, image_batch, strict=True)),
        *zip(tail_names, tail, strict=True),
    ]


def _argument_contract(arguments: Sequence[Any], generator_manifest: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    expected = {entry["name"]: entry for entry in generator_manifest["buffers"]}
    if not expected or not all(entry["finite_by_construction"] for entry in expected.values()):
        raise RuntimeError("GF46 operand generator includes a non-finite fill")
    observed = []
    for name, value in _named_argument_leaves(arguments):
        dtype = str(np.dtype(value.dtype))
        shape = tuple(int(size) for size in value.shape)
        element_count = math.prod(shape) if shape else 1
        logical_size_bytes = element_count * np.dtype(dtype).itemsize
        observed.append(
            {
                "name": name,
                "shape": list(shape),
                "dtype": dtype,
                "logical_size_bytes": logical_size_bytes,
            }
        )
    if [entry["name"] for entry in observed] != list(expected):
        raise RuntimeError("fused scorer argument names/order drifted")
    mismatches = {
        entry["name"]: {
            "expected": {key: expected[entry["name"]][key] for key in ("shape", "dtype", "logical_size_bytes")},
            "observed": {key: entry[key] for key in ("shape", "dtype", "logical_size_bytes")},
        }
        for entry in observed
        if any(entry[key] != expected[entry["name"]][key] for key in ("shape", "dtype", "logical_size_bytes"))
    }
    logical_bytes = sum(entry["logical_size_bytes"] for entry in observed)
    if mismatches:
        raise RuntimeError(f"GF46 fused scorer shapes/dtypes differ: {mismatches}")
    if logical_bytes != EXPECTED_LOGICAL_ARGUMENT_BYTES:
        raise RuntimeError(
            "GF46 fused scorer logical argument size differs: "
            f"expected={EXPECTED_LOGICAL_ARGUMENT_BYTES}, observed={logical_bytes}"
        )
    return {
        "schema": "recovar.vdam.gf46_certificate_state_argument_contract.v1",
        "buffers": observed,
        "logical_dynamic_argument_bytes": logical_bytes,
        "pass": True,
    }


def _shape_arguments():
    import jax
    import numpy as np

    from recovar.em.dense_single_volume.helpers import scoring
    from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
        CoarseGemmHybridIntervalState,
        certified_f32_dot_product_gamma,
        certified_f64_expanded_score_gammas,
    )

    g = GF46_GEOMETRY
    blocks = g.total_rotations // g.source_rotation_block
    shape = jax.ShapeDtypeStruct
    state = CoarseGemmHybridIntervalState(
        *(shape((g.batch_size, blocks), np.float64) for _ in range(4)),
        shape((g.total_rotations,), np.int32),
        shape((g.batch_size,), np.int32),
        shape((g.batch_size,), np.int32),
    )
    image_batch = scoring.RelionCoarseGaussianGemmF64ImageBatch(
        weighted_shifted=shape((g.batch_size * g.translation_count, g.pixel_count), np.complex128),
        pixel_weight=shape((g.batch_size, g.pixel_count), np.float64),
        image_energy=shape((g.batch_size, g.translation_count), np.float64),
        initial_diff2=shape((g.batch_size,), np.float64),
        image_component_abs_max=shape((g.batch_size,), np.float64),
        weight_max=shape((g.batch_size,), np.float64),
        active=shape((g.batch_size,), np.bool_),
        stored_inputs_valid=shape((g.batch_size,), np.bool_),
        actual_image_count=shape((), np.int32),
    )
    gamma = certified_f64_expanded_score_gammas(g.pixel_count)
    return (
        state,
        shape((g.rotation_block, g.pixel_count), np.complex64),
        image_batch,
        shape((), np.float32),
        shape((g.rotation_block,), np.float32),
        shape((g.batch_size, g.translation_count), np.float32),
        shape((), np.float64),
        shape((), np.float64),
        shape((), np.float64),
        shape((), np.float64),
        shape((), np.float64),
        shape((), np.int32),
        shape((), np.int32),
    ), {
        "cross_gamma": gamma.cross,
        "energy_gamma": gamma.energy,
        "initial_diff2_gamma": gamma.initial_diff2,
        "energy_envelope_gamma": gamma.energy_envelope,
        "direct_gamma": certified_f32_dot_product_gamma(g.pixel_count, g.translation_count),
    }


def _concrete_arguments(coefficients: dict[str, float]):
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.helpers import scoring
    from recovar.em.dense_single_volume.helpers.coarse_gemm_hybrid import (
        CoarseGemmHybridIntervalState,
    )

    g = GF46_GEOMETRY
    blocks = g.total_rotations // g.source_rotation_block
    state = CoarseGemmHybridIntervalState(
        *(jnp.full((g.batch_size, blocks), -1e100, dtype=jnp.float64) for _ in range(4)),
        jnp.zeros((g.total_rotations,), dtype=jnp.int32),
        jnp.zeros((g.batch_size,), dtype=jnp.int32),
        jnp.zeros((g.batch_size,), dtype=jnp.int32),
    )
    energy = g.pixel_count * ((0.125**2 + 0.0625**2) * 0.75)
    image_batch = scoring.RelionCoarseGaussianGemmF64ImageBatch(
        weighted_shifted=jnp.full(
            (g.batch_size * g.translation_count, g.pixel_count),
            0.09375 + 0.046875j,
            dtype=jnp.complex128,
        ),
        pixel_weight=jnp.full((g.batch_size, g.pixel_count), 0.75, dtype=jnp.float64),
        image_energy=jnp.full((g.batch_size, g.translation_count), energy, dtype=jnp.float64),
        initial_diff2=jnp.full((g.batch_size,), 0.125, dtype=jnp.float64),
        image_component_abs_max=jnp.full((g.batch_size,), 0.125, dtype=jnp.float64),
        weight_max=jnp.full((g.batch_size,), 0.75, dtype=jnp.float64),
        active=jnp.full((g.batch_size,), True, dtype=jnp.bool_),
        stored_inputs_valid=jnp.full((g.batch_size,), True, dtype=jnp.bool_),
        actual_image_count=jnp.asarray(g.batch_size, dtype=jnp.int32),
    )
    return (
        state,
        jnp.full(
            (g.rotation_block, g.pixel_count),
            0.03125 - 0.015625j,
            dtype=jnp.complex64,
        ),
        image_batch,
        jnp.asarray(-0.25, dtype=jnp.float32),
        jnp.full((g.rotation_block,), -0.03125, dtype=jnp.float32),
        jnp.full((g.batch_size, g.translation_count), -0.015625, dtype=jnp.float32),
        jnp.asarray(coefficients["cross_gamma"], dtype=jnp.float64),
        jnp.asarray(coefficients["energy_gamma"], dtype=jnp.float64),
        jnp.asarray(coefficients["initial_diff2_gamma"], dtype=jnp.float64),
        jnp.asarray(coefficients["energy_envelope_gamma"], dtype=jnp.float64),
        jnp.asarray(coefficients["direct_gamma"], dtype=jnp.float64),
        jnp.asarray(g.pixel_count, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )


def _extract_balanced_json(text: str, marker: str = "backend_config=") -> tuple[str | None, Any | None]:
    marker_index = text.find(marker)
    if marker_index < 0:
        return None, None
    start = text.find("{", marker_index + len(marker))
    if start < 0:
        return None, None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                raw = text[start : index + 1]
                try:
                    return raw, json.loads(raw)
                except json.JSONDecodeError:
                    return raw, None
    return text[start:], None


def _interesting_backend_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    selected: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            lowered = key.lower()
            if any(token in lowered for token in ("algorithm", "precision", "epilogue", "math_type")):
                selected[path] = child
            selected.update(_interesting_backend_fields(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            selected.update(_interesting_backend_fields(child, f"{prefix}[{index}]"))
    return selected


def _hlo_evidence(stablehlo: str, pre_optimized_hlo: str, optimized_hlo: str) -> dict[str, Any]:
    stable_count = len(re.findall(r"\bstablehlo\.dot_general\b", stablehlo))
    pre_dot_lines = [
        {"line_number": number, "text": line}
        for number, line in enumerate(pre_optimized_hlo.splitlines(), 1)
        if re.search(r"\bdot\(", line)
    ]
    gemm_calls = []
    target_pattern = re.compile(r'custom_call_target="([^"]+)"')
    for number, line in enumerate(optimized_hlo.splitlines(), 1):
        target_match = target_pattern.search(line)
        if target_match is None:
            continue
        target = target_match.group(1)
        lowered = target.lower()
        if "cublas" not in lowered or not ("gemm" in lowered or "matmul" in lowered):
            continue
        raw_config, parsed_config = _extract_balanced_json(line)
        gemm_calls.append(
            {
                "line_number": number,
                "custom_call_target": target,
                "instruction": line,
                "backend_config_raw": raw_config,
                "backend_config": parsed_config,
                "selected_backend_fields": _interesting_backend_fields(parsed_config),
            }
        )
    gate_checks = {
        "stablehlo_dot_general_count_is_two": stable_count == EXPECTED_DOT_COUNT,
        "pre_optimized_hlo_dot_count_is_two": len(pre_dot_lines) == EXPECTED_DOT_COUNT,
        "optimized_cublas_gemm_like_count_is_two": len(gemm_calls) == EXPECTED_DOT_COUNT,
        "every_gemm_has_parseable_backend_config": bool(gemm_calls)
        and all(isinstance(call["backend_config"], dict) for call in gemm_calls),
    }
    return {
        "schema": "recovar.vdam.gf46_certificate_state_hlo_evidence.v1",
        "stablehlo_dot_general_count": stable_count,
        "pre_optimized_hlo_dot_count": len(pre_dot_lines),
        "pre_optimized_hlo_dot_instructions": pre_dot_lines,
        "optimized_cublas_gemm_like_count": len(gemm_calls),
        "optimized_cublas_gemm_like_calls": gemm_calls,
        "gate_checks": gate_checks,
        "pass": all(gate_checks.values()),
    }


def _memory_analysis_payload(stats: Any, proto_path: Path) -> dict[str, Any]:
    if stats is None:
        raise RuntimeError("XLA compiled memory_analysis() is unavailable")
    missing = [name for name in REQUIRED_MEMORY_FIELDS if not hasattr(stats, name)]
    if missing:
        raise RuntimeError(f"XLA memory_analysis fields are missing: {missing}")
    fields = {name: int(getattr(stats, name)) for name in REQUIRED_MEMORY_FIELDS}
    proto = getattr(stats, "serialized_buffer_assignment_proto", None)
    if not isinstance(proto, (bytes, bytearray, memoryview)):
        raise RuntimeError("serialized XLA buffer assignment is unavailable")
    proto_bytes = bytes(proto)
    if not proto_bytes:
        raise RuntimeError("serialized XLA buffer assignment is empty")
    proto_path.write_bytes(proto_bytes)
    public_attributes = sorted(name for name in dir(stats) if not name.startswith("_"))
    public_attribute_values = {}
    for name in public_attributes:
        if name == "serialized_buffer_assignment_proto":
            public_attribute_values[name] = {
                "stored_separately": proto_path.name,
                "size_bytes": len(proto_bytes),
                "sha256": _sha256_bytes(proto_bytes),
            }
            continue
        value = getattr(stats, name)
        if isinstance(value, (bool, int, float, str)) or value is None:
            public_attribute_values[name] = value
        else:
            public_attribute_values[name] = {
                "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "repr": repr(value),
            }
    return {
        "schema": "recovar.vdam.xla_memory_analysis.v1",
        "python_type": f"{type(stats).__module__}.{type(stats).__qualname__}",
        "all_public_attributes": public_attributes,
        "all_public_attribute_values": public_attribute_values,
        "scalar_fields": fields,
        "serialized_buffer_assignment": {
            "path": proto_path.name,
            "size_bytes": len(proto_bytes),
            "sha256": _sha256_bytes(proto_bytes),
        },
    }


def _runtime_provenance(repo_root: Path, expected_gpu_uuid: str) -> dict[str, Any]:
    if not _GPU_UUID_RE.fullmatch(expected_gpu_uuid):
        raise ValueError("expected GPU UUID must have the form GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    if os.environ.get("JAX_ENABLE_X64") != "1":
        raise RuntimeError("JAX_ENABLE_X64 must be exactly 1")
    if os.environ.get("JAX_PLATFORMS") != "cuda":
        raise RuntimeError("JAX_PLATFORMS must be exactly cuda")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the pinned physical GPU UUID")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise RuntimeError("H100 qualification must run inside a Slurm job")

    import jax
    import jax.extend.backend as jax_backend
    import jaxlib

    import recovar

    if not bool(jax.config.x64_enabled):
        raise RuntimeError("JAX x64 is disabled")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX default backend is not GPU: {jax.default_backend()}")
    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise RuntimeError(f"expected exactly one visible JAX GPU, observed {devices}")
    if "H100" not in str(devices[0].device_kind):
        raise RuntimeError(f"visible JAX device is not an H100: {devices[0]}")
    recovar_path = Path(recovar.__file__).resolve()
    if not recovar_path.is_relative_to(repo_root):
        raise RuntimeError(f"RECOVAR imported outside the sealed repository: {recovar_path}")

    query_fields = "uuid,name,index,pci.bus_id,memory.total,driver_version"
    nvidia_query = _run(
        [
            "nvidia-smi",
            "-i",
            expected_gpu_uuid,
            f"--query-gpu={query_fields}",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    rows = list(csv.reader([nvidia_query], skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise RuntimeError(f"unexpected nvidia-smi query result: {nvidia_query!r}")
    uuid, name, index, pci_bus_id, memory_total_mib, driver_version = rows[0]
    if uuid != expected_gpu_uuid or "H100" not in name:
        raise RuntimeError(f"nvidia-smi did not resolve the pinned H100: {rows[0]}")
    backend = jax_backend.get_backend()
    return {
        "timestamp_utc": _utc_now(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "executable_sha256": _sha256_file(Path(sys.executable).resolve()),
        },
        "jax": {
            "version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "jax_file": str(Path(jax.__file__).resolve()),
            "jaxlib_file": str(Path(jaxlib.__file__).resolve()),
            "x64_enabled": bool(jax.config.x64_enabled),
            "default_backend": jax.default_backend(),
            "platform_version": str(backend.platform_version),
        },
        "device": {
            "jax_repr": str(devices[0]),
            "jax_device_kind": str(devices[0].device_kind),
            "jax_platform": str(devices[0].platform),
            "visible_device_count": len(devices),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "uuid": uuid,
            "name": name,
            "nvidia_smi_index": index,
            "pci_bus_id": pci_bus_id,
            "memory_total_mib": int(memory_total_mib),
            "driver_version": driver_version,
        },
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
                "SLURMD_NODENAME",
                "SLURM_JOB_GPUS",
                "SLURM_STEP_GPUS",
                "SLURM_CPUS_PER_TASK",
            )
        },
        "recovar_file": str(recovar_path),
    }


def _device_memory_stats(device: Any) -> dict[str, int] | None:
    try:
        stats = device.memory_stats()
    except Exception:  # pragma: no cover - backend-dependent diagnostic
        return None
    if stats is None:
        return None
    return {str(key): int(value) for key, value in stats.items() if isinstance(value, int)}


def _block_tree(value: Any) -> Any:
    import jax

    return jax.tree.map(lambda leaf: leaf.block_until_ready(), value)


def _output_digest(output: Any) -> dict[str, Any]:
    import jax
    import numpy as np

    field_names = tuple(getattr(output, "_fields", ()))
    leaves = jax.tree.leaves(output)
    if len(field_names) != len(leaves):
        raise RuntimeError("compiled output no longer matches the seven-field interval state")
    fields = []
    combined = hashlib.sha256()
    for name, leaf in zip(field_names, leaves, strict=True):
        array = np.asarray(jax.device_get(leaf))
        raw = array.tobytes(order="C")
        digest = _sha256_bytes(raw)
        combined.update(name.encode() + b"\0" + digest.encode() + b"\0")
        finite = bool(np.all(np.isfinite(array)))
        fields.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "size_bytes": array.nbytes,
                "sha256": digest,
                "all_finite": finite,
                "minimum": float(np.min(array)),
                "maximum": float(np.max(array)),
            }
        )
    return {"fields": fields, "combined_sha256": combined.hexdigest()}


def _execution(
    compiled: Any,
    coefficients: dict[str, float],
    *,
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, Any]:
    import jax
    import numpy as np

    device = jax.devices("gpu")[0]
    allocation_started = time.perf_counter_ns()
    arguments = _concrete_arguments(coefficients)
    argument_contract = _argument_contract(
        arguments,
        _operand_generator_manifest(coefficients),
    )
    _block_tree(arguments)
    allocation_finished = time.perf_counter_ns()

    warmups = []
    for _ in range(warmup_runs):
        started = time.perf_counter_ns()
        warm_output = _block_tree(compiled(*arguments))
        finished = time.perf_counter_ns()
        warmups.append((finished - started) / 1e9)
    del warm_output

    timed_seconds = []
    retained_outputs = []
    for index in range(timed_runs):
        started = time.perf_counter_ns()
        output = _block_tree(compiled(*arguments))
        finished = time.perf_counter_ns()
        timed_seconds.append((finished - started) / 1e9)
        if index in (0, timed_runs - 1):
            retained_outputs.append(output)

    first_digest = _output_digest(retained_outputs[0])
    final_digest = _output_digest(retained_outputs[-1])
    deterministic = first_digest["combined_sha256"] == final_digest["combined_sha256"]
    all_finite = all(field["all_finite"] for field in final_digest["fields"])
    if not deterministic:
        raise RuntimeError("repeated synchronized scorer outputs are not byte-identical")
    if not all_finite:
        raise RuntimeError("scorer output contains a non-finite value")

    expected_candidates = GF46_GEOMETRY.rotation_block * GF46_GEOMETRY.translation_count
    candidate_field = np.asarray(jax.device_get(retained_outputs[-1].candidate_count))
    invalid_field = np.asarray(jax.device_get(retained_outputs[-1].invalid_candidate_count))
    visits = np.asarray(jax.device_get(retained_outputs[-1].rotation_visit_count))
    semantic_checks = {
        "candidate_count_exact": bool(np.all(candidate_field == expected_candidates)),
        "invalid_candidate_count_zero": bool(np.all(invalid_field == 0)),
        "visited_block_exactly_once": bool(np.all(visits[: GF46_GEOMETRY.rotation_block] == 1)),
        "unvisited_rotations_zero": bool(np.all(visits[GF46_GEOMETRY.rotation_block :] == 0)),
    }
    if not all(semantic_checks.values()):
        raise RuntimeError(f"deterministic scorer semantic checks failed: {semantic_checks}")
    return {
        "schema": "recovar.vdam.gf46_certificate_state_execution.v1",
        "synchronization": "every scorer call blocked on every output leaf before stopping timer",
        "allocation_seconds": (allocation_finished - allocation_started) / 1e9,
        "concrete_argument_contract": argument_contract,
        "warmup_seconds": warmups,
        "warm_timed_seconds": timed_seconds,
        "warm_timed_summary_seconds": {
            "minimum": min(timed_seconds),
            "median": statistics.median(timed_seconds),
            "mean": statistics.fmean(timed_seconds),
            "maximum": max(timed_seconds),
        },
        "repeat_output_byte_identical": deterministic,
        "first_output": first_digest,
        "final_output": final_digest,
        "semantic_checks": semantic_checks,
        "device_memory_stats_after_execution": _device_memory_stats(device),
    }


def _write_artifact_manifest(output_root: Path) -> None:
    manifest = output_root / "SHA256SUMS"
    digest_path = output_root / "SHA256SUMS.sha256"
    paths = sorted(path for path in output_root.rglob("*") if path.is_file() and path not in (manifest, digest_path))
    lines = [f"{_sha256_file(path)}  {path.relative_to(output_root)}\n" for path in paths]
    manifest.write_text("".join(lines))
    digest_path.write_text(f"{_sha256_file(manifest)}  SHA256SUMS\n")


def _seal_output(output_root: Path) -> None:
    for path in output_root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(output_root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    output_root.chmod(0o555)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-repo-head")
    parser.add_argument("--expected-repo-tree")
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=5)
    parser.add_argument("--print-source-manifest-sha256", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    if args.print_source_manifest_sha256:
        manifest, _entries = _source_manifest(repo_root)
        print(_sha256_bytes(manifest))
        return 0

    required = {
        "output_root": args.output_root,
        "expected_repo_head": args.expected_repo_head,
        "expected_repo_tree": args.expected_repo_tree,
        "expected_source_manifest_sha256": args.expected_source_manifest_sha256,
        "expected_gpu_uuid": args.expected_gpu_uuid,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"qualification arguments are required: {missing}")
    if not _SHA256_RE.fullmatch(args.expected_source_manifest_sha256):
        raise SystemExit("expected source-manifest SHA-256 must be lowercase 64-hex")
    if args.warmup_runs < 1 or args.timed_runs < 2:
        raise SystemExit("qualification requires at least one warmup and two timed runs")
    if Path.cwd().resolve() != repo_root:
        raise SystemExit(f"run from the sealed repository root: {repo_root}")

    if not args.output_root.is_absolute():
        raise SystemExit(f"output root must be an absolute path: {args.output_root}")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"output root must be a new absolute path: {output_root}")
    if not output_root.parent.is_dir():
        raise SystemExit(f"output parent does not exist: {output_root.parent}")
    if output_root.is_relative_to(repo_root):
        raise SystemExit(f"output root must be outside the repository: {output_root}")

    output_root.mkdir()
    (output_root / "provenance").mkdir()
    (output_root / "analysis").mkdir()
    (output_root / "results").mkdir()
    (output_root / "SAFE_TO_DELETE").write_text("Disposable GF46 H100 certificate-state qualification artifact.\n")
    events_path = output_root / "results" / "events.jsonl"

    def event(stage: str, **fields: Any) -> None:
        record = {"timestamp_utc": _utc_now(), "stage": stage, **fields}
        with events_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    failure: BaseException | None = None
    stage = "initial_validation"
    report: dict[str, Any] = {
        "schema": "recovar.vdam.gf46_certificate_state_h100_qualification.v1",
        "classification": "incomplete",
        "geometry": GF46_GEOMETRY._asdict(),
        "expected_dot_count": EXPECTED_DOT_COUNT,
    }
    initial_manifest: bytes | None = None
    try:
        event(stage)
        _assert_exact_geometry()
        repository = _repository_provenance(
            repo_root,
            expected_head=args.expected_repo_head,
            expected_tree=args.expected_repo_tree,
        )
        initial_manifest, source_entries = _source_manifest(repo_root)
        source_digest = _sha256_bytes(initial_manifest)
        if source_digest != args.expected_source_manifest_sha256:
            raise RuntimeError(
                f"source-manifest mismatch: expected={args.expected_source_manifest_sha256}, observed={source_digest}"
            )
        provenance = output_root / "provenance"
        (provenance / "source_manifest.sha256").write_bytes(initial_manifest)
        _write_json(
            provenance / "source_manifest.json",
            {
                "sha256": source_digest,
                "scope": "selected qualification, scorer, environment, and test sources",
                "entries": source_entries,
            },
        )
        _write_json(provenance / "repository.json", repository)
        _write_json(
            provenance / "invocation.json",
            {
                "argv": sys.argv,
                "shell_command": shlex.join(sys.argv),
                "cwd": str(Path.cwd()),
                "timestamp_utc": _utc_now(),
            },
        )
        environment_keys = (
            "CUDA_VISIBLE_DEVICES",
            "JAX_COMPILATION_CACHE_DIR",
            "JAX_ENABLE_X64",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
            "JAX_PLATFORMS",
            "PYTHONNOUSERSITE",
            "PYTHONUNBUFFERED",
            "SLURMD_NODENAME",
            "SLURM_CPUS_PER_TASK",
            "SLURM_JOB_GPUS",
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_PARTITION",
            "SLURM_STEP_GPUS",
            "TMPDIR",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_FLAGS",
        )
        _write_json(
            provenance / "environment.json",
            {key: os.environ.get(key) for key in environment_keys},
        )

        stage = "runtime_gate"
        event(stage)
        runtime = _runtime_provenance(repo_root, args.expected_gpu_uuid)
        _write_json(provenance / "runtime.json", runtime)
        try:
            (provenance / "nvidia_smi_q.txt").write_text(_run(["nvidia-smi", "-i", args.expected_gpu_uuid, "-q"]))
            (provenance / "slurm_job.txt").write_text(_run(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]]))
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"failed to capture GPU/Slurm provenance: {error}") from error

        from recovar.em.dense_single_volume.helpers import scoring

        stage = "lowering"
        event(stage)
        shape_arguments, coefficients = _shape_arguments()
        operand_manifest = _operand_generator_manifest(coefficients)
        shape_argument_contract = _argument_contract(shape_arguments, operand_manifest)
        _write_json(output_root / "analysis" / "operand_generator_manifest.json", operand_manifest)
        _write_json(output_root / "analysis" / "shape_argument_contract.json", shape_argument_contract)
        lowering_started_utc = _utc_now()
        lowering_started_ns = time.perf_counter_ns()
        lowered = scoring._relion_coarse_gaussian_gemm_update_certificate_state_jit.lower(*shape_arguments)
        lowering_finished_ns = time.perf_counter_ns()
        lowering_finished_utc = _utc_now()
        stablehlo = str(lowered.compiler_ir("stablehlo"))
        pre_hlo_computation = lowered.compiler_ir("hlo")
        pre_optimized_hlo = pre_hlo_computation.as_hlo_text()
        (output_root / "analysis" / "stablehlo_pre_optimization.mlir").write_text(stablehlo)
        (output_root / "analysis" / "hlo_pre_optimization.txt").write_text(pre_optimized_hlo)
        (output_root / "analysis" / "hlo_pre_optimization.pb").write_bytes(
            pre_hlo_computation.as_serialized_hlo_module_proto()
        )

        stage = "compilation"
        event(stage)
        compile_started_utc = _utc_now()
        compile_started_ns = time.perf_counter_ns()
        compiled = lowered.compile()
        compile_finished_ns = time.perf_counter_ns()
        compile_finished_utc = _utc_now()
        optimized_hlo = compiled.as_text()
        runtime_executable = compiled.runtime_executable()
        runtime_hlo = runtime_executable.get_hlo_text()
        runtime_modules = runtime_executable.hlo_modules()
        (output_root / "analysis" / "hlo_optimized_compiled.txt").write_text(optimized_hlo)
        (output_root / "analysis" / "hlo_optimized_runtime.txt").write_text(runtime_hlo)
        runtime_module_records = []
        for index, module in enumerate(runtime_modules):
            module_text = module.to_string()
            path = output_root / "analysis" / f"hlo_optimized_runtime_module_{index:03d}.txt"
            path.write_text(module_text)
            runtime_module_records.append(
                {"index": index, "path": path.name, "sha256": _sha256_bytes(module_text.encode())}
            )

        memory = _memory_analysis_payload(
            compiled.memory_analysis(),
            output_root / "analysis" / "xla_buffer_assignment.pb",
        )
        _write_json(output_root / "analysis" / "xla_memory_analysis.json", memory)
        hlo = _hlo_evidence(stablehlo, pre_optimized_hlo, optimized_hlo)
        _write_json(output_root / "analysis" / "hlo_gemm_evidence.json", hlo)
        compile_record = {
            "schema": "recovar.vdam.gf46_certificate_state_compile.v1",
            "lowering_started_utc": lowering_started_utc,
            "lowering_finished_utc": lowering_finished_utc,
            "lowering_seconds": (lowering_finished_ns - lowering_started_ns) / 1e9,
            "compile_started_utc": compile_started_utc,
            "compile_finished_utc": compile_finished_utc,
            "compile_seconds": (compile_finished_ns - compile_started_ns) / 1e9,
            "lowering_input": "jax.ShapeDtypeStruct only",
            "runtime_hlo_matches_compiled_hlo": runtime_hlo == optimized_hlo,
            "runtime_hlo_modules": runtime_module_records,
            "artifact_sha256": {
                path.name: _sha256_file(path)
                for path in sorted((output_root / "analysis").glob("*hlo*"))
                if path.is_file()
            },
        }
        _write_json(output_root / "analysis" / "compile.json", compile_record)
        report.update(
            {
                "repository": repository,
                "source_manifest_sha256": source_digest,
                "runtime": runtime,
                "operand_generator_manifest_sha256": operand_manifest["generator_manifest_sha256"],
                "shape_argument_contract": shape_argument_contract,
                "compile": compile_record,
                "xla_memory_analysis": memory,
                "hlo_evidence": hlo,
            }
        )
        _write_json(output_root / "results" / "qualification.partial.json", report)
        if not hlo["pass"]:
            raise RuntimeError(f"HLO GEMM structure gate failed: {hlo['gate_checks']}")

        stage = "full_geometry_execution"
        event(stage, mandatory_analysis_persisted=True)
        execution = _execution(
            compiled,
            coefficients,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )
        _write_json(output_root / "results" / "execution.json", execution)
        report["execution"] = execution

        stage = "final_source_verification"
        event(stage)
        final_manifest, _final_entries = _source_manifest(repo_root)
        (provenance / "source_manifest.final.sha256").write_bytes(final_manifest)
        if final_manifest != initial_manifest:
            raise RuntimeError("source manifest changed during qualification")
        final_repository = _repository_provenance(
            repo_root,
            expected_head=args.expected_repo_head,
            expected_tree=args.expected_repo_tree,
        )
        if final_repository != repository:
            raise RuntimeError("repository provenance changed during qualification")
        report["classification"] = "qualified_exact_gf46_h100"
        report["completed_utc"] = _utc_now()
        _write_json(output_root / "results" / "qualification.json", report)
        (output_root / "COMPLETED").write_text("Exact GF46 H100 qualification completed.\n")
        event("complete")
    except BaseException as error:  # Preserve static evidence even after an execution OOM.
        failure = error
        failure_record = {
            "schema": "recovar.vdam.gf46_certificate_state_failure.v1",
            "stage": stage,
            "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp_utc": _utc_now(),
            "mandatory_analysis_persisted": (output_root / "analysis" / "xla_memory_analysis.json").is_file(),
            "geometry_was_not_reduced": True,
        }
        _write_json(output_root / "results" / "failure.json", failure_record)
        report["classification"] = (
            "full_geometry_execution_failed_after_mandatory_analysis"
            if stage == "full_geometry_execution" and failure_record["mandatory_analysis_persisted"]
            else "qualification_failed"
        )
        report["failure"] = failure_record
        _write_json(output_root / "results" / "qualification.json", report)
        (output_root / "FAILED").write_text(f"{stage}: {type(error).__name__}: {error}\n")
        event("failure", failed_stage=stage, exception_type=type(error).__name__)
    finally:
        try:
            final_manifest, _final_entries = _source_manifest(repo_root)
            final_path = output_root / "provenance" / "source_manifest.final.sha256"
            if not final_path.exists():
                final_path.write_bytes(final_manifest)
            _write_artifact_manifest(output_root)
            _seal_output(output_root)
        except BaseException as finalization_error:
            if failure is None:
                failure = finalization_error
            print(f"artifact finalization failed: {finalization_error}", file=sys.stderr)

    if failure is not None:
        print(f"GF46 H100 qualification failed at {stage}: {failure}", file=sys.stderr)
        return 1
    print(f"GF46 H100 qualification complete: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
