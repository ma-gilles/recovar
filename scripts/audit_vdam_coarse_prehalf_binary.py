#!/usr/bin/env python3
"""Audit the Hopper binary contract for the default-off coarse pre-half experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA = "recovar.vdam_coarse_prehalf_binary_audit.v1"
DEFAULT_KERNEL = "relion_coarse_diff2_projector_f32_kernel"
PREHALF_KERNEL = "relion_coarse_diff2_projector_prehalf_f32_kernel"
BASELINE_TOKEN = "ILi16ELb0ELb0ELb0EE"
DEFAULT_TOKEN = BASELINE_TOKEN
PREHALF_TOKEN = "ILi16ELb0EE"
BASELINE_DEMANGLED = "<16,false,false,false>("
DEFAULT_DEMANGLED = BASELINE_DEMANGLED
PREHALF_DEMANGLED = "<16,false>("


class BinaryAuditError(ValueError):
    """Raised when a CUDA artifact cannot prove the requested contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BinaryAuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _matching_block(
    text: str,
    *,
    kernel: str,
    token: str,
    demangled_token: str,
    header_pattern: re.Pattern[str],
    label: str,
) -> tuple[str, str]:
    matches = list(header_pattern.finditer(text))
    selected: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        header = match.group("header")
        normalized = _normalized(header)
        if kernel not in header:
            continue
        if token not in header and demangled_token not in normalized:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        selected.append((header, text[match.end() : end]))
    _require(len(selected) == 1, f"{label} matched {len(selected)} kernel blocks, expected one")
    return selected[0]


_SASS_HEADER = re.compile(r"(?m)^\s*Function\s+:\s+(?P<header>.+)$")
_RESOURCE_HEADER = re.compile(r"(?m)^\s*Function\s+(?P<header>[^\n]+):\s*$")
_SASS_INSTRUCTION = re.compile(
    r"/\*[0-9a-fA-F]+\*/\s+(?P<instruction>.*?)\s*;\s*"
    r"/\*\s*(?P<encoding0>0x[0-9a-fA-F]+)\s*\*/\s*\n\s*"
    r"/\*\s*(?P<encoding1>0x[0-9a-fA-F]+)\s*\*/",
)
_RESOURCE_VALUES = re.compile(
    r"REG:(?P<registers>\d+)\s+STACK:(?P<stack>\d+)\s+"
    r"SHARED:(?P<shared>\d+)\s+LOCAL:(?P<local>\d+)",
)


def _sass_signature(
    text: str,
    *,
    kernel: str,
    token: str,
    demangled_token: str,
    label: str,
) -> dict[str, Any]:
    header, block = _matching_block(
        text,
        kernel=kernel,
        token=token,
        demangled_token=demangled_token,
        header_pattern=_SASS_HEADER,
        label=label,
    )
    instructions = [
        {
            "instruction": " ".join(match.group("instruction").split()),
            "encoding": [match.group("encoding0").lower(), match.group("encoding1").lower()],
        }
        for match in _SASS_INSTRUCTION.finditer(block)
    ]
    _require(instructions, f"{label} has no parsed SASS instructions")
    half_fmul_count = sum(
        bool(
            re.search(
                r"(?:^|\s)FMUL(?:\.[A-Z0-9_.]+)?\s+.*?,\s*0\.5$",
                row["instruction"],
            )
        )
        for row in instructions
    )
    sequence_sha256 = hashlib.sha256(
        json.dumps(instructions, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "header": header,
        "instruction_count": len(instructions),
        "half_fmul_count": half_fmul_count,
        "instruction_sequence_sha256": sequence_sha256,
        "instructions": instructions,
    }


def _sass_comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_instructions = reference["instructions"]
    candidate_instructions = candidate["instructions"]
    common_count = min(len(reference_instructions), len(candidate_instructions))
    text_mismatches = [
        index
        for index in range(common_count)
        if reference_instructions[index]["instruction"] != candidate_instructions[index]["instruction"]
    ]
    encoding_mismatches = [
        index
        for index in range(common_count)
        if reference_instructions[index]["encoding"] != candidate_instructions[index]["encoding"]
    ]
    unmatched = list(range(common_count, max(len(reference_instructions), len(candidate_instructions))))
    text_mismatches.extend(unmatched)
    encoding_mismatches.extend(unmatched)
    return {
        "exact": reference_instructions == candidate_instructions,
        "instruction_count_delta": len(candidate_instructions) - len(reference_instructions),
        "instruction_text_mismatch_count": len(text_mismatches),
        "encoding_mismatch_count": len(encoding_mismatches),
        "first_instruction_text_mismatch_indices": text_mismatches[:16],
        "first_encoding_mismatch_indices": encoding_mismatches[:16],
        "baseline_instruction_sequence_sha256": reference["instruction_sequence_sha256"],
        "candidate_instruction_sequence_sha256": candidate["instruction_sequence_sha256"],
    }


def _resources(
    text: str,
    *,
    kernel: str,
    token: str,
    demangled_token: str,
    label: str,
) -> dict[str, int | str]:
    header, block = _matching_block(
        text,
        kernel=kernel,
        token=token,
        demangled_token=demangled_token,
        header_pattern=_RESOURCE_HEADER,
        label=label,
    )
    match = _RESOURCE_VALUES.search(block)
    _require(match is not None, f"{label} has no CUDA resource row")
    return {
        "header": header,
        "registers_per_thread": int(match.group("registers")),
        "stack_bytes_per_thread": int(match.group("stack")),
        "static_shared_bytes": int(match.group("shared")),
        "local_bytes_per_thread": int(match.group("local")),
    }


def _analyze_dumps(
    *,
    baseline_sass: str,
    candidate_sass: str,
    baseline_resources: str,
    candidate_resources: str,
) -> dict[str, Any]:
    baseline = _sass_signature(
        baseline_sass,
        kernel=DEFAULT_KERNEL,
        token=BASELINE_TOKEN,
        demangled_token=BASELINE_DEMANGLED,
        label="baseline atomic kernel",
    )
    default = _sass_signature(
        candidate_sass,
        kernel=DEFAULT_KERNEL,
        token=DEFAULT_TOKEN,
        demangled_token=DEFAULT_DEMANGLED,
        label="candidate default atomic kernel",
    )
    prehalf = _sass_signature(
        candidate_sass,
        kernel=PREHALF_KERNEL,
        token=PREHALF_TOKEN,
        demangled_token=PREHALF_DEMANGLED,
        label="candidate pre-half atomic kernel",
    )
    baseline_resource = _resources(
        baseline_resources,
        kernel=DEFAULT_KERNEL,
        token=BASELINE_TOKEN,
        demangled_token=BASELINE_DEMANGLED,
        label="baseline atomic resources",
    )
    default_resource = _resources(
        candidate_resources,
        kernel=DEFAULT_KERNEL,
        token=DEFAULT_TOKEN,
        demangled_token=DEFAULT_DEMANGLED,
        label="candidate default atomic resources",
    )
    prehalf_resource = _resources(
        candidate_resources,
        kernel=PREHALF_KERNEL,
        token=PREHALF_TOKEN,
        demangled_token=PREHALF_DEMANGLED,
        label="candidate pre-half atomic resources",
    )

    default_sass_comparison = _sass_comparison(baseline, default)
    default_sass_exact = default_sass_comparison["exact"]
    default_resources_exact = default_resource == {
        **baseline_resource,
        "header": default_resource["header"],
    }
    half_multiply_contract = (
        baseline["half_fmul_count"] == 16 and default["half_fmul_count"] == 16 and prehalf["half_fmul_count"] == 1
    )
    resource_nonregression = (
        prehalf_resource["registers_per_thread"] <= default_resource["registers_per_thread"]
        and prehalf_resource["stack_bytes_per_thread"] <= default_resource["stack_bytes_per_thread"]
        and prehalf_resource["static_shared_bytes"] == default_resource["static_shared_bytes"]
        and prehalf_resource["local_bytes_per_thread"] <= default_resource["local_bytes_per_thread"]
    )
    gates = {
        "default_atomic_sass_exact": default_sass_exact,
        "default_atomic_resources_exact": default_resources_exact,
        "half_multiply_16_to_1": half_multiply_contract,
        "prehalf_resource_nonregression": resource_nonregression,
    }
    return {
        "sass": {
            "baseline": {key: value for key, value in baseline.items() if key != "instructions"},
            "candidate_default": {key: value for key, value in default.items() if key != "instructions"},
            "candidate_prehalf": {key: value for key, value in prehalf.items() if key != "instructions"},
        },
        "resources": {
            "baseline": baseline_resource,
            "candidate_default": default_resource,
            "candidate_prehalf": prehalf_resource,
        },
        "default_atomic_sass_comparison": default_sass_comparison,
        "gates": gates,
        "pass": all(gates.values()),
        "default_enablement_allowed": False,
        "production_wiring_evaluated": False,
    }


def _run_cuobjdump(cuobjdump: Path, flag: str, binary: Path) -> str:
    result = subprocess.run(
        [str(cuobjdump), "--gpu-architecture", "sm_90", flag, str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(result.returncode == 0, f"cuobjdump {flag} failed for {binary}: {result.stderr.strip()}")
    _require(bool(result.stdout), f"cuobjdump {flag} returned no output for {binary}")
    return result.stdout


def audit(*, baseline_binary: Path, candidate_binary: Path, cuobjdump: Path) -> dict[str, Any]:
    paths = {
        "baseline_binary": baseline_binary.resolve(),
        "candidate_binary": candidate_binary.resolve(),
        "cuobjdump": cuobjdump.resolve(),
    }
    for label, path in paths.items():
        _require(path.is_file(), f"{label} is not a file: {path}")
    result = _analyze_dumps(
        baseline_sass=_run_cuobjdump(paths["cuobjdump"], "--dump-sass", paths["baseline_binary"]),
        candidate_sass=_run_cuobjdump(paths["cuobjdump"], "--dump-sass", paths["candidate_binary"]),
        baseline_resources=_run_cuobjdump(
            paths["cuobjdump"],
            "--dump-resource-usage",
            paths["baseline_binary"],
        ),
        candidate_resources=_run_cuobjdump(
            paths["cuobjdump"],
            "--dump-resource-usage",
            paths["candidate_binary"],
        ),
    )
    version = subprocess.run(
        [str(paths["cuobjdump"]), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(version.returncode == 0, "cuobjdump --version failed")
    return {
        "schema": SCHEMA,
        "classification": "diagnostic_primitive_qualification",
        "architecture": "sm_90",
        "artifacts": {
            "baseline_binary": {
                "path": str(paths["baseline_binary"]),
                "sha256": _sha256(paths["baseline_binary"]),
            },
            "candidate_binary": {
                "path": str(paths["candidate_binary"]),
                "sha256": _sha256(paths["candidate_binary"]),
            },
            "cuobjdump": {
                "path": str(paths["cuobjdump"]),
                "version": version.stdout.strip(),
            },
        },
        **result,
    }


def _markdown(report: dict[str, Any]) -> str:
    status = "PASS" if report["pass"] else "FAIL"
    rows = [
        f"# VDAM coarse pre-half Hopper binary audit: {status}",
        "",
        "> Diagnostic primitive qualification only; production wiring and default enablement were not evaluated.",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    rows.extend(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in report["gates"].items())
    rows.extend(
        (
            "",
            "| Kernel | Instructions | `FMUL ..., 0.5` | Registers | Shared B | Stack B | Local B |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for key, label in (
        ("baseline", "sealed baseline"),
        ("candidate_default", "rebuilt default"),
        ("candidate_prehalf", "pre-half experiment"),
    ):
        sass = report["sass"][key]
        resource = report["resources"][key]
        rows.append(
            f"| {label} | {sass['instruction_count']} | {sass['half_fmul_count']} | "
            f"{resource['registers_per_thread']} | {resource['static_shared_bytes']} | "
            f"{resource['stack_bytes_per_thread']} | {resource['local_bytes_per_thread']} |"
        )
    rows.extend(
        (
            "",
            f"- Baseline SHA-256: `{report['artifacts']['baseline_binary']['sha256']}`",
            f"- Candidate SHA-256: `{report['artifacts']['candidate_binary']['sha256']}`",
            "- Default SASS text mismatches: "
            f"`{report['default_atomic_sass_comparison']['instruction_text_mismatch_count']}`",
            "- Default SASS encoding mismatches: "
            f"`{report['default_atomic_sass_comparison']['encoding_mismatch_count']}`",
            "- `default_enablement_allowed=false`",
            "- `production_wiring_evaluated=false`",
            "",
        )
    )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _require(not args.output_json.exists(), f"output already exists: {args.output_json}")
        _require(not args.output_markdown.exists(), f"output already exists: {args.output_markdown}")
        report = audit(
            baseline_binary=args.baseline_binary,
            candidate_binary=args.candidate_binary,
            cuobjdump=args.cuobjdump,
        )
    except BinaryAuditError as exc:
        parser.error(str(exc))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(_markdown(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
