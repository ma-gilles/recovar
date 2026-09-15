#!/usr/bin/env python3
"""Run a prewarmed ABBA+BAAB VDAM dense-representation panel."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import jax

from scripts.run_vdam_late_iteration_profile import (
    _all_optimized_q32_environment,
    _recovar_argv,
    _sha256,
)

SCHEMA = "recovar.vdam_schedule_representation_panel.v1"
ARM_SPECS = (
    ("dynamic_1", 64),
    ("oracle_1", 288),
    ("oracle_2", 288),
    ("dynamic_2", 64),
    ("oracle_3", 288),
    ("dynamic_3", 64),
    ("dynamic_4", 64),
    ("oracle_4", 288),
)
PREWARM_SPECS = (("prewarm_dynamic", 64), ("prewarm_oracle", 288))


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-star", type=Path, required=True)
    parser.add_argument("--checkpoint-optimiser", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-iteration", type=int, required=True)
    parser.add_argument("--nr-iter", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=29)
    parser.add_argument("--image-batch-size", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--exact-local-bucket-radix", type=int, default=4)
    parser.add_argument(
        "--exact-local-physical-order-chunk-size", type=int, default=0
    )
    parser.add_argument("--stable-fourier-window-shapes", action="store_true")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_arguments(parser)
    return parser.parse_args(argv)


def _run_arm(
    args: argparse.Namespace,
    label: str,
    capacity: int,
    *,
    expected_representation: str | None = None,
    expected_fallback: int | None = None,
) -> dict:
    from recovar.commands.initial_model import main as initial_model_command

    output_prefix = args.output_root / "arms" / label / "run"
    output_prefix.parent.mkdir(parents=True, exist_ok=False)
    os.environ["RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY"] = str(
        capacity
    )
    started = time.perf_counter()
    status = int(initial_model_command(_recovar_argv(args=args, output_prefix=output_prefix)))
    jax.effects_barrier()
    wall_s = float(time.perf_counter() - started)
    if status != 0:
        raise RuntimeError(f"{label} exited with status {status}")
    iteration = int(args.checkpoint_iteration) + 1
    meta_path = output_prefix.with_name(
        f"{output_prefix.name}_it{iteration:03d}_recovar_meta.json"
    )
    meta = json.loads(meta_path.read_text())
    hybrid = meta["halfset_0_profile_summary"]["coarse_gaussian_gemm_hybrid"]
    if (expected_representation is None) != (expected_fallback is None):
        raise ValueError("representation and fallback expectations must be paired")
    if expected_representation is not None and (
        hybrid.get("score_representation_batch_counts")
        != {expected_representation: 1}
        or hybrid.get("fallback_batch_count") != expected_fallback
    ):
        raise RuntimeError(
            f"{label} representation mismatch: "
            f"{hybrid.get('score_representation_batch_counts')!r}, "
            f"fallback={hybrid.get('fallback_batch_count')!r}"
        )
    return {
        "label": label,
        "block_capacity": capacity,
        "wall_s": wall_s,
        "output_prefix": str(output_prefix.resolve()),
        "artifact_prefix": str(
            meta_path.with_name(meta_path.name.removesuffix("_recovar_meta.json")).resolve()
        ),
        "meta_path": str(meta_path.resolve()),
        "score_representation_batch_counts": hybrid[
            "score_representation_batch_counts"
        ],
        "fallback_batch_count": hybrid["fallback_batch_count"],
        "fallback_reasons": hybrid["fallback_reasons"],
    }


def _prepare_run(args: argparse.Namespace) -> dict[str, str]:
    args.input_star = args.input_star.resolve(strict=True)
    args.checkpoint_optimiser = args.checkpoint_optimiser.resolve(strict=True)
    args.data_dir = args.data_dir.resolve(strict=True)
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    (args.output_root / "SAFE_TO_DELETE").touch()
    if not args.stable_fourier_window_shapes:
        raise ValueError("the optimized representation panel requires stable shapes")

    expected_environment = _all_optimized_q32_environment()
    for name, value in expected_environment.items():
        if name == "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY":
            continue
        if os.environ.get(name) != value:
            raise RuntimeError(
                f"optimized environment mismatch: {name}={os.environ.get(name)!r}, "
                f"expected {value!r}"
            )
    os.environ["RECOVAR_INITIAL_MODEL_PROFILE"] = "1"
    return expected_environment


def _expected_arm(args: argparse.Namespace, label: str, capacity: int) -> dict:
    dynamic = label.startswith("dynamic_") or label == "prewarm_dynamic"
    return _run_arm(
        args,
        label,
        capacity,
        expected_representation=(
            "dense_full_direct_dynamic_fallback"
            if dynamic
            else "dense_full_direct_static_capacity"
        ),
        expected_fallback=1 if dynamic else 0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    expected_environment = _prepare_run(args)

    prewarm = [
        _expected_arm(args, label, capacity) for label, capacity in PREWARM_SPECS
    ]
    arms = [_expected_arm(args, label, capacity) for label, capacity in ARM_SPECS]
    report = {
        "schema": SCHEMA,
        "classification": "diagnostic_schedule_representation_qualification",
        "input_star": str(args.input_star),
        "input_star_sha256": _sha256(args.input_star),
        "checkpoint_optimiser": str(args.checkpoint_optimiser),
        "checkpoint_optimiser_sha256": _sha256(args.checkpoint_optimiser),
        "checkpoint_iteration": int(args.checkpoint_iteration),
        "profiled_iteration": int(args.checkpoint_iteration) + 1,
        "environment_without_block_capacity": {
            name: value
            for name, value in sorted(expected_environment.items())
            if name != "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY"
        },
        "prewarm": prewarm,
        "arm_order": [label for label, _capacity in ARM_SPECS],
        "arms": {arm["label"]: arm for arm in arms},
    }
    report_path = args.output_root / "panel.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
