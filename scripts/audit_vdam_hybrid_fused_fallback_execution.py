#!/usr/bin/env python3
"""Audit actual mature-fused execution for a forced VDAM hybrid fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "recovar.vdam_hybrid_fused_fallback_execution.v1"
FUSED_TARGET = "cuda_relion_coarse_diff2_projector_f32"
FUSED_WRAPPER = "relion_coarse_diff2_projector_f32"
ARM_ORDER = ("stable_off_1", "stable_on_1", "stable_on_2", "stable_off_2")
CANDIDATE_ARMS = ("stable_on_1", "stable_on_2")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class FallbackExecutionAuditError(RuntimeError):
    """Raised when forced-fallback execution evidence is incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FallbackExecutionAuditError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FallbackExecutionAuditError(f"cannot read {label} at {path}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_field(values: dict[str, Any], key: str, *, prefix: str) -> int:
    value = values.get(key)
    _require(type(value) is int, f"{prefix} {key} is not an integer")
    return value


def validate_fused_fallback_observation(
    meta: dict[str, Any],
    *,
    arm: str,
    iteration: int,
    expected_capacity: int,
) -> dict[str, Any]:
    """Require one recorded significance call to use only full fused fallback."""

    profile = meta.get("halfset_0_profile_summary")
    _require(isinstance(profile, dict), f"{arm} iteration {iteration} has no half-set profile")
    hybrid = profile.get("coarse_gaussian_gemm_hybrid")
    selector = profile.get("coarse_selector_audit")
    _require(isinstance(hybrid, dict), f"{arm} iteration {iteration} has no hybrid audit")
    _require(isinstance(selector, dict), f"{arm} iteration {iteration} has no selector audit")
    counts = selector.get("counts")
    layout = hybrid.get("coarse_square_layout")
    _require(isinstance(counts, dict), f"{arm} iteration {iteration} has no selector counts")
    _require(isinstance(layout, dict), f"{arm} iteration {iteration} has no square-layout audit")

    prefix = f"{arm} iteration {iteration}"
    full_batches = _integer_field(hybrid, "full_dense_batch_count", prefix=prefix)
    full_images = _integer_field(hybrid, "full_dense_image_count", prefix=prefix)
    _require(
        _integer_field(hybrid, "selected_block_capacity", prefix=prefix)
        == int(expected_capacity),
        f"{prefix} selected-block capacity differs",
    )
    _require(full_batches > 0 and full_images > 0, f"{prefix} did not exercise full fallback")
    _require(
        hybrid.get("all_full_dense_batches_used_fused") is True,
        f"{prefix} did not route every full batch through fused scoring",
    )
    _require(
        hybrid.get("whole_batch_fail_closed_fallback") is True,
        f"{prefix} did not report fail-closed whole-batch fallback",
    )
    _require(
        hybrid.get("full_fallback_backend_requested") == "fused_projector"
        and hybrid.get("full_fallback_backend_armed") == "fused_projector",
        f"{prefix} did not request and arm fused fallback",
    )
    _require(
        hybrid.get("full_fallback_backend_effective") == "fused_projector",
        f"{prefix} effective fallback backend differs",
    )
    _require(
        _integer_field(hybrid, "fused_full_fallback_batch_count", prefix=prefix)
        == full_batches,
        f"{prefix} fused/full batch counts differ",
    )
    _require(
        _integer_field(hybrid, "fused_full_fallback_image_count", prefix=prefix)
        == full_images,
        f"{prefix} fused/full image counts differ",
    )
    _require(
        _integer_field(hybrid, "rectangular_full_fallback_batch_count", prefix=prefix)
        == 0
        and _integer_field(hybrid, "rectangular_full_fallback_image_count", prefix=prefix)
        == 0,
        f"{prefix} executed a rectangular fallback",
    )
    _require(
        _integer_field(hybrid, "selected_rescore_batch_count", prefix=prefix) == 0,
        f"{prefix} unexpectedly executed selected rescoring",
    )
    fallback_reasons = hybrid.get("fallback_reasons")
    _require(
        isinstance(fallback_reasons, dict)
        and set(fallback_reasons) == {"block_capacity_overflow"}
        and _integer_field(
            fallback_reasons,
            "block_capacity_overflow",
            prefix=f"{prefix} fallback reasons",
        )
        > 0,
        f"{prefix} fallback reason was not solely capacity overflow",
    )
    _require(
        _integer_field(hybrid, "overflow_latch_activation_count", prefix=prefix) == 1
        and hybrid.get("overflow_latch_active_at_return") is True,
        f"{prefix} overflow latch evidence differs",
    )
    _require(
        selector.get("requested_fused") is True and selector.get("effective_fused") is True,
        f"{prefix} fused selector was not requested and effective",
    )
    _require(selector.get("target") == FUSED_TARGET, f"{prefix} selector target differs")
    _require(selector.get("wrapper") == FUSED_WRAPPER, f"{prefix} selector wrapper differs")
    _require(
        _integer_field(counts, "fused_calls", prefix=prefix) == full_batches,
        f"{prefix} fused selector call count differs",
    )
    _require(
        _integer_field(counts, "actual_rows", prefix=prefix) == full_images,
        f"{prefix} fused selector row count differs",
    )
    _require(
        _integer_field(counts, "multistream_calls", prefix=prefix) == 0
        and _integer_field(counts, "prehalf_selected_calls", prefix=prefix) == 0
        and _integer_field(counts, "native_atomic_selected_calls", prefix=prefix) == 0,
        f"{prefix} executed an unsupported fused variant",
    )

    stable_requested = arm in CANDIDATE_ARMS
    _require(
        "stable_fourier_window_shapes_requested" in layout,
        f"{prefix} omitted its stable-shape request",
    )
    _require(
        layout.get("stable_fourier_window_shapes_requested") is stable_requested,
        f"{prefix} stable-shape request differs",
    )
    logical_pixels = int(layout.get("logical_square_pixels", -1))
    physical_pixels = int(layout.get("physical_square_pixels", -1))
    _require(logical_pixels > 0, f"{prefix} logical square size is invalid")
    _require(physical_pixels >= logical_pixels, f"{prefix} physical square is too small")
    _require(
        int(layout.get("executed_square_pixels", -1)) == logical_pixels
        and layout.get("logical_issue_stream_is_prefix") is True,
        f"{prefix} did not execute the logical lookup prefix",
    )
    if physical_pixels > logical_pixels:
        _require(stable_requested, f"{prefix} padded a control lookup")
        _require(
            layout.get("physical_tail_skipped_by_runtime_count") is True,
            f"{prefix} did not skip its physical lookup tail",
        )

    return {
        "arm": arm,
        "iteration": int(iteration),
        "selected_block_capacity": int(expected_capacity),
        "full_dense_batch_count": full_batches,
        "full_dense_image_count": full_images,
        "fused_full_fallback_batch_count": full_batches,
        "fused_full_fallback_image_count": full_images,
        "rectangular_full_fallback_batch_count": 0,
        "selector_target": FUSED_TARGET,
        "selector_wrapper": FUSED_WRAPPER,
        "logical_square_pixels": logical_pixels,
        "physical_square_pixels": physical_pixels,
        "fallback_reasons": dict(fallback_reasons),
    }


def audit_fused_fallback_root(
    root: Path,
    *,
    iterations: Sequence[int],
    expected_capacity: int = 1,
    expected_head: str | None = None,
) -> dict[str, Any]:
    """Validate a four-arm forced-fallback execution root."""

    root = root.resolve()
    iteration_values = tuple(int(value) for value in iterations)
    _require(
        iteration_values
        and iteration_values == tuple(sorted(set(iteration_values)))
        and iteration_values[0] > 0,
        "fallback iterations must be sorted, unique, and positive",
    )
    _require(int(expected_capacity) > 0, "expected fallback capacity must be positive")
    provenance = root / "provenance"
    source_head = (provenance / "repo_head.txt").read_text().strip()
    _require(_HEX40.fullmatch(source_head) is not None, "fallback source head is invalid")
    if expected_head is not None:
        _require(source_head == expected_head, "fallback source head differs")
    _require(
        (provenance / "coarse_mode.txt").read_text().strip() == "hybrid_fused_fallback",
        "fallback coarse mode differs",
    )
    _require(
        int((provenance / "hybrid_block_capacity.txt").read_text().strip())
        == int(expected_capacity),
        "fallback provenance capacity differs",
    )

    observations = []
    command_hashes = {}
    for arm in ARM_ORDER:
        run_root = root / "runs" / arm
        _require((run_root / "SCIENCE_COMPLETED").is_file(), f"{arm} science sentinel is missing")
        command_path = run_root / "command.json"
        command = _load_json(command_path, f"{arm} command")
        _require(
            int(command.get("hybrid_block_capacity", -1)) == int(expected_capacity),
            f"{arm} command fallback capacity differs",
        )
        command_hashes[arm] = _sha256(command_path)
        for iteration in iteration_values:
            meta_path = run_root / "output" / f"run_it{iteration:03d}_recovar_meta.json"
            meta = _load_json(meta_path, f"{arm} iteration {iteration} metadata")
            observation = validate_fused_fallback_observation(
                meta,
                arm=arm,
                iteration=iteration,
                expected_capacity=int(expected_capacity),
            )
            observation["metadata_path"] = str(meta_path.resolve())
            observation["metadata_sha256"] = _sha256(meta_path)
            observations.append(observation)

    return {
        "schema": SCHEMA,
        "result": "pass",
        "scope": (
            "actual mature-fused fallback execution; independent of stable-shape "
            "speed and repeat-oracle gates"
        ),
        "root": str(root),
        "source_head": source_head,
        "coarse_mode": "hybrid_fused_fallback",
        "hybrid_block_capacity": int(expected_capacity),
        "arms": list(ARM_ORDER),
        "iterations": list(iteration_values),
        "observation_count": len(observations),
        "command_sha256": command_hashes,
        "observations": observations,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, action="append", required=True)
    parser.add_argument("--expected-capacity", type=int, default=1)
    parser.add_argument("--expected-head")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = audit_fused_fallback_root(
            args.root,
            iterations=args.iteration,
            expected_capacity=args.expected_capacity,
            expected_head=args.expected_head,
        )
    except (FallbackExecutionAuditError, OSError, ValueError) as error:
        print(f"SETUP ERROR: {error}")
        return 2
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
