#!/usr/bin/env python3
"""Analyze the frozen GF46 coarse expanded-GEMM qualification gate.

The gate has two deliberately separate parts:

* one immutable, timing-ineligible paired score-surface capture for selected
  original particles 1 and 2160, one from each pseudo-halfset; and
* one diagnostic-unset A/B/B/A late-iteration timing panel, where A is the
  mature rectangular scorer and B is the expanded shared-GEMM scorer.

This analyzer is performance-only.  Passing it does not establish trajectory
quality, permit default enablement, or alter the frozen VDAM parity score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import starfile

from scripts.analyze_vdam_coarse_multistream_late_pair import (
    LatePairSetupError as GemmGateSetupError,
)
from scripts.analyze_vdam_coarse_multistream_late_pair import (
    _git_rev_parse,
    _load_json,
    _load_map,
    _load_star,
    _map_delta,
    _read_single_line,
    _require,
    _resolved_inside,
    _sha256,
    _star_equal,
    _validate_manifest,
    _values_equal,
)

SCHEMA = "recovar.vdam_coarse_gemm_gf46_gate_analysis.v1"
RUN_SCHEMA = "recovar.vdam_coarse_gemm_gf46_gate.v1"
PROFILE_SCHEMA = "recovar.vdam_late_iteration_profile.v1"
PROFILED_ITERATION = 181
MATERIAL_END_TO_END_RATIO = 0.90
TARGET_NODE = "della-h21g4"
TARGET_GPU_UUID = "GPU-099c0d77-bb85-f2e9-f628-148b733c9176"
EXPECTED_INPUT_MANIFEST_SHA256 = (
    "de224471a690d1faaae4067217dbcc90b632269d62b0b3372b20aafa69157d91"
)
EXPECTED_CHECKPOINT_OPTIMISER_SHA256 = (
    "e55c86262ab1800eef5da19845833dac852c8d0b01b6940018dbb4ad95558606"
)
EXPECTED_INPUT_STAR_SHA256 = (
    "90d4b8cf9413d81d71dc91cb3bf36c56cfe74b218fffc52e8adc0452c64d99c0"
)
EXPECTED_PARTICLE_STACK_SHA256 = (
    "804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9"
)
EXPECTED_CUDA_SHA256 = (
    "2af7bf1e4cbdc10705948d907c087d1662db612fe8d57362f1390033ac6c047b"
)
EXPECTED_RELION_BIND_SHA256 = (
    "9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980"
)
EXPECTED_INTERPRETER_SHA256 = (
    "48556a44c0dd1570866beb838e6fcbea771bce93d413acd4197bb2f254b72d23"
)
EXPECTED_CUSPARSE_SHA256 = (
    "58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf"
)
GF46_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "vdam_full_expansion_v3_984637b7d_87274be_20260826/"
    "vdam-gf46/repeat-01/vdam-gf46"
)
EXPECTED_CHECKPOINT_OPTIMISER = GF46_ROOT / "relion/run_it180_optimiser.star"
EXPECTED_INPUT_STAR = GF46_ROOT / "relion/run_it180_data.star"
EXPECTED_DATA_DIR = GF46_ROOT / "data"
EXPECTED_PARTICLE_STACK = EXPECTED_DATA_DIR / "particles.128.mrcs"
RAW_TARGETS = (1, 2160)
RAW_TARGET_POSITIONS = (72, 999)
RAW_TARGET_PART_IDS = (1, 2160)
RAW_TARGET_HALFSETS = (1, 0)
EXPECTED_SELECTED_IDS_INT64_SHA256 = (
    "c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be"
)
EXPECTED_SCHEDULE = {
    "current_size": 128,
    "healpix_order": 3,
    "n_rotations": 294_912,
    "n_translations": 116,
    "subset_size": 1_000,
    "random_perturbation": 0.4751259684562683,
}
# The continuation scores iteration 181 with the checkpoint's box size (100),
# then advances the persisted schedule to 128 for the emitted iteration-181
# state.  Raw paired-score artifacts therefore record the scoring box, while
# profile summaries and output metadata record the post-iteration schedule.
EXPECTED_RAW_SCORE_CURRENT_SIZE = 100
ARM_SPECS = (
    ("direct_1", 0, 1),
    ("gemm_1", 1, 1),
    ("gemm_2", 1, 2),
    ("direct_2", 0, 2),
)
ARM_LABELS = tuple(spec[0] for spec in ARM_SPECS)
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
TIMING_METRICS = (
    "warm_wall_s",
    "warm_expectation_s",
    "warm_pass1_s",
    "warm_pass2_s",
    "peak_rss_gib",
)
ENVELOPE_METRICS = (
    "relative_l2",
    "max_abs",
    "abs_relative_scale_drift",
    "abs_signed_mean_over_delta_rms",
)
COMMON_SELECTOR_ENV = {
    "RECOVAR_K1_COARSE_FUSED_PROJECTOR": "0",
    "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION": "0",
    "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": "0",
    "RECOVAR_K1_COARSE_PREHALF_WEIGHT": "0",
    "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL": "0",
    "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": "0",
    "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE": "0",
    "RECOVAR_K1_COARSE_GAUSSIAN_FFI": "1",
    "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF": "1",
    "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS": "1",
    "RECOVAR_K1_RELION_F32_COARSE_SUPPORT": "1",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB": "2",
    "RECOVAR_EM_RAW_IMAGE_CACHE": "auto",
    "RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB": "16",
}
DIAGNOSTIC_DIR_ENV = "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR"
DIAGNOSTIC_INDICES_ENV = (
    "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES"
)
GEMM_ENV = "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO"
GEMM_LOG_MARKER = "Opt-in shared coarse projection-once/GEMM macro enabled"
SOURCE_REQUIRED = frozenset(
    {
        "recovar/em/dense_single_volume/helpers/scoring.py",
        "recovar/em/dense_single_volume/helpers/significance.py",
        "recovar/em/initial_model/dense_adapter.py",
        "recovar/em/initial_model/driver.py",
        "recovar/em/initial_model/iteration_loop.py",
        "scripts/run_ab_initio.py",
        "scripts/run_vdam_late_iteration_profile.py",
        "scripts/analyze_vdam_coarse_gemm_gf46_gate.py",
        "scripts/run_vdam_coarse_gemm_gf46_gate.sbatch",
        "tests/unit/initial_model/test_vdam_coarse_gemm_gf46_analyzer.py",
        "tests/unit/initial_model/test_vdam_coarse_gemm_gf46_runner.py",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    parsed = float(value)
    _require(math.isfinite(parsed), f"{label} is non-finite")
    if positive:
        _require(parsed > 0.0, f"{label} must be positive")
    return parsed


def _integer(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} is not an integer")
    return int(value)


def _np_scalar(archive: Any, key: str) -> Any:
    _require(key in archive.files, f"raw score artifact lacks {key}")
    value = np.asarray(archive[key])
    _require(value.shape == (), f"raw score artifact {key} is not scalar")
    return value.item()


def _within(value: float, envelope: float) -> bool:
    _require(math.isfinite(value) and math.isfinite(envelope), "repeat-envelope value is non-finite")
    return bool(value <= np.nextafter(envelope, math.inf) if envelope > 0.0 else value == 0.0)


def _delta_envelope(delta: dict[str, float]) -> dict[str, float]:
    return {
        "relative_l2": float(delta["relative_l2"]),
        "max_abs": float(delta["max_abs"]),
        "abs_relative_scale_drift": abs(float(delta["relative_scale_drift"])),
        "abs_signed_mean_over_delta_rms": abs(float(delta["signed_mean_over_delta_rms"])),
    }


def _against_envelope(delta: dict[str, float], envelope: dict[str, float]) -> dict[str, Any]:
    values = _delta_envelope(delta)
    checks = {key: _within(values[key], envelope[key]) for key in ENVELOPE_METRICS}
    return {
        "values": values,
        "control_repeat_envelope": envelope,
        "bounded_metrics": checks,
        "within_control_repeat_envelope": all(checks.values()),
    }


def _parse_sha_record(path: Path, expected: str, label: str) -> dict[str, str]:
    _require(bool(_SHA256_RE.fullmatch(expected)), f"{label} expected digest is invalid")
    line = _read_single_line(path, label)
    try:
        digest, raw_path = line.split(maxsplit=1)
    except ValueError as exc:
        raise GemmGateSetupError(f"{label} has malformed sha256sum output") from exc
    _require(digest == expected, f"{label} digest differs from run.json")
    artifact = Path(raw_path).resolve()
    _require(artifact.is_file(), f"{label} target is missing: {artifact}")
    _require(_sha256(artifact) == digest, f"{label} target digest differs")
    return {"path": str(artifact), "sha256": digest}


def _validate_raw_target_preflight(root: Path) -> dict[str, Any]:
    path = root / "provenance" / "raw_target_preflight.json"
    payload = _load_json(path, "raw target preflight")
    expected = {
        "schema": "recovar.vdam_coarse_gemm_gf46_raw_target_preflight.v1",
        "checkpoint": str(EXPECTED_CHECKPOINT_OPTIMISER.resolve()),
        "input_star": str(EXPECTED_INPUT_STAR.resolve()),
        "checkpoint_iteration": 180,
        "profiled_iteration": 181,
        "random_seed": 29,
        "native_shuffle_seed": 210,
        "particle_count": 3_000,
        "subset_size": 1_000,
        "selected_particle_ids_int64_sha256": EXPECTED_SELECTED_IDS_INT64_SHA256,
        "joint_halfset_particle_stream": True,
        "requested_image_batch_size": 500,
        "targets": [
            {
                "original_index": target,
                "selected_position": position,
                "part_id": part_id,
                "pseudo_halfset_id": halfset,
                "requested_batch_index": batch_index,
            }
            for target, position, part_id, halfset, batch_index in zip(
                RAW_TARGETS,
                RAW_TARGET_POSITIONS,
                RAW_TARGET_PART_IDS,
                RAW_TARGET_HALFSETS,
                (0, 1),
                strict=True,
            )
        ],
    }
    _require(payload == expected, f"raw target preflight differs: {payload}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "state": payload}


def _parse_env_command(path: Path, label: str) -> tuple[dict[str, str], set[str], list[str]]:
    _require(path.is_file(), f"missing {label} command: {path}")
    try:
        tokens = shlex.split(path.read_text())
    except (OSError, ValueError) as exc:
        raise GemmGateSetupError(f"cannot parse {label} command: {exc}") from exc
    _require(tokens and tokens[0] == "env", f"{label} command does not start with env")
    assignments: dict[str, str] = {}
    unsets: set[str] = set()
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-u":
            _require(index + 1 < len(tokens), f"{label} command has incomplete env -u")
            unsets.add(tokens[index + 1])
            index += 2
            continue
        if "=" not in token:
            break
        name, value = token.split("=", 1)
        _require(name and name not in assignments, f"{label} command repeats env {name}")
        assignments[name] = value
        index += 1
    argv = tokens[index:]
    _require(len(argv) >= 3 and argv[1] == "-m", f"{label} command lacks python -m")
    return assignments, unsets, argv


def _flag_value(argv: list[str], flag: str, label: str) -> str:
    _require(argv.count(flag) == 1, f"{label} command does not contain exactly one {flag}")
    index = argv.index(flag)
    _require(index + 1 < len(argv), f"{label} command has incomplete {flag}")
    return argv[index + 1]


def _validate_common_command_env(assignments: dict[str, str], label: str) -> None:
    mismatches = {
        name: {"expected": value, "observed": assignments.get(name)}
        for name, value in COMMON_SELECTOR_ENV.items()
        if assignments.get(name) != value
    }
    _require(not mismatches, f"{label} common selector environment differs: {mismatches}")


def _validate_raw_command(root: Path) -> dict[str, Any]:
    path = root / "provenance" / "raw_score_diagnostic_command.sh"
    assignments, unsets, argv = _parse_env_command(path, "raw diagnostic")
    _validate_common_command_env(assignments, "raw diagnostic")
    diagnostic_root = (root / "raw_score_diagnostic" / "artifacts").resolve()
    _require(assignments.get(GEMM_ENV) == "1", "raw diagnostic did not request GEMM")
    _require(assignments.get(DIAGNOSTIC_DIR_ENV) == str(diagnostic_root), "raw diagnostic directory differs")
    _require(assignments.get(DIAGNOSTIC_INDICES_ENV) == "1,2160", "raw diagnostic indices differ")
    _require(DIAGNOSTIC_DIR_ENV not in unsets and DIAGNOSTIC_INDICES_ENV not in unsets, "raw diagnostic env was unset")
    _require(argv[2] == "scripts.run_ab_initio", "raw diagnostic did not call scripts.run_ab_initio")
    _require("scripts.run_vdam_late_iteration_profile" not in argv, "raw diagnostic used the two-pass profiler")
    expected_flags = {
        "--i": str(EXPECTED_INPUT_STAR),
        "--o": str((root / "raw_score_diagnostic" / "output" / "run").resolve()),
        "--nr_iter": "200",
        "--grad_write_iter": "1",
        "--K": "1",
        "--tau2_fudge": "4",
        "--sym": "C1",
        "--do_run_C1": "1",
        "--particle_diameter": "200",
        "--random_seed": "29",
        "--healpix_order": "1",
        "--oversampling": "1",
        "--offset_range": "6",
        "--offset_step": "2",
        "--padding_factor": "1",
        "--image_batch_size": "500",
        "--datadir": str(EXPECTED_DATA_DIR),
        "--gpu": "0",
        "--diagnostic_continue_optimiser": str(EXPECTED_CHECKPOINT_OPTIMISER),
        "--diagnostic_stop_after_iteration": "181",
    }
    mismatches = {
        flag: {"expected": expected, "observed": _flag_value(argv, flag, "raw diagnostic")}
        for flag, expected in expected_flags.items()
        if _flag_value(argv, flag, "raw diagnostic") != expected
    }
    _require(not mismatches, f"raw diagnostic command differs: {mismatches}")
    _require(argv.count("--require_custom_cuda") == 1, "raw diagnostic does not require custom CUDA")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "environment": assignments}


def _validate_timing_command(root: Path, label: str, macro: int) -> dict[str, Any]:
    path = root / "provenance" / f"{label}_command.sh"
    assignments, unsets, argv = _parse_env_command(path, label)
    _validate_common_command_env(assignments, label)
    _require(assignments.get(GEMM_ENV) == str(macro), f"{label} GEMM mode differs")
    _require(
        {DIAGNOSTIC_DIR_ENV, DIAGNOSTIC_INDICES_ENV}.issubset(unsets),
        f"{label} did not explicitly unset both diagnostic variables",
    )
    _require(DIAGNOSTIC_DIR_ENV not in assignments and DIAGNOSTIC_INDICES_ENV not in assignments, f"{label} assigned a diagnostic variable")
    _require(argv[2] == "scripts.run_vdam_late_iteration_profile", f"{label} did not call the late profiler")
    expected_flags = {
        "--checkpoint-optimiser": str(EXPECTED_CHECKPOINT_OPTIMISER),
        "--input-star": str(EXPECTED_INPUT_STAR),
        "--data-dir": str(EXPECTED_DATA_DIR),
        "--output-root": str((root / "runs" / label / "profile").resolve()),
        "--checkpoint-iteration": "180",
        "--nr-iter": "200",
        "--random-seed": "29",
        "--image-batch-size": "500",
        "--exact-local-bucket-radix": "4",
        "--exact-local-physical-order-chunk-size": "0",
    }
    mismatches = {
        flag: {"expected": expected, "observed": _flag_value(argv, flag, label)}
        for flag, expected in expected_flags.items()
        if _flag_value(argv, flag, label) != expected
    }
    _require(not mismatches, f"{label} timing command differs: {mismatches}")
    _require("--cuda-profiler-range" not in argv, f"{label} enabled the CUDA profiler without Nsight")
    _require("--audit-raw-image-cache" not in argv, f"{label} enabled cache instrumentation")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "environment": assignments}


def _validate_execution_order(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing execution order: {path}")
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
    except OSError as exc:
        raise GemmGateSetupError(f"cannot read execution order: {exc}") from exc
    _require(bool(rows), "execution order is empty")
    _require(list(rows[0]) == ["order", "label", "gemm_macro"], "execution-order columns differ")
    expected = [
        {"order": str(index + 1), "label": label, "gemm_macro": str(macro)}
        for index, (label, macro, _) in enumerate(ARM_SPECS)
    ]
    _require(rows == expected, f"execution order differs: {rows}")
    return rows


def _validate_provenance(root: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(root.is_dir(), f"GF46 GEMM gate root does not exist: {root}")
    _require((root / "RAW_COMPLETED").is_file(), "raw score diagnostic is incomplete")
    _require((root / "ARMS_COMPLETED").is_file(), "timing panel is incomplete")
    _require(not (root / "provenance" / "failure.txt").exists(), "gate root records a setup failure")
    provenance = root / "provenance"
    run_path = provenance / "run.json"
    run = _load_json(run_path, "run provenance")
    expected_run = {
        "schema": RUN_SCHEMA,
        "classification": "diagnostic_performance_only",
        "execution_order": list(ARM_LABELS),
        "gemm_macro_modes": [spec[1] for spec in ARM_SPECS],
        "raw_score_diagnostic_original_indices": list(RAW_TARGETS),
        "raw_score_diagnostic_selected_positions": list(RAW_TARGET_POSITIONS),
        "raw_score_diagnostic_part_ids": list(RAW_TARGET_PART_IDS),
        "raw_score_diagnostic_pseudo_halfset_ids": list(RAW_TARGET_HALFSETS),
        "raw_score_diagnostic_joint_halfset_particle_stream": True,
        "selected_particle_ids_int64_sha256": EXPECTED_SELECTED_IDS_INT64_SHA256,
        "raw_score_diagnostic_timing_eligible": False,
        "timing_diagnostic_variables_unset": True,
        "checkpoint_iteration": 180,
        "profiled_iteration": 181,
        "nr_iter_schedule": 200,
        "random_seed": 29,
        "image_batch_size": 500,
        "raw_image_cache": "auto",
        "raw_image_cache_max_gb": 16.0,
        "exact_local_bucket_radix": 4,
        "exact_local_physical_order_chunk_size": 0,
        "material_end_to_end_ratio": MATERIAL_END_TO_END_RATIO,
        "source_manifest_scope": "selected_high_risk_files",
        "science_promotion_allowed": False,
        "default_enablement_allowed": False,
        "node": TARGET_NODE,
        "gpu_uuid": TARGET_GPU_UUID,
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA256,
        "particle_stack_sha256": EXPECTED_PARTICLE_STACK_SHA256,
        "cuda_sha256": EXPECTED_CUDA_SHA256,
        "relion_bind_sha256": EXPECTED_RELION_BIND_SHA256,
        "interpreter_sha256": EXPECTED_INTERPRETER_SHA256,
        "cusparse_sha256": EXPECTED_CUSPARSE_SHA256,
    }
    mismatches = {
        key: {"expected": expected, "observed": run.get(key)}
        for key, expected in expected_run.items()
        if run.get(key) != expected
    }
    _require(not mismatches, f"run provenance differs: {mismatches}")
    for key in ("git_head", "git_tree"):
        _require(isinstance(run.get(key), str) and bool(_GIT_OBJECT_RE.fullmatch(run[key])), f"run {key} is invalid")
    for key in ("source_manifest_sha256",):
        _require(isinstance(run.get(key), str) and bool(_SHA256_RE.fullmatch(run[key])), f"run {key} is invalid")
    _require(isinstance(run.get("job_id"), str) and run["job_id"], "run job ID is invalid")
    _require("H100" in str(run.get("gpu_name", "")), "run was not recorded on an H100")

    resolved_head = _git_rev_parse(repo, f"{run['git_head']}^{{commit}}", "run git_head")
    resolved_tree = _git_rev_parse(repo, f"{run['git_head']}^{{tree}}", "run git tree")
    _require(resolved_head == run["git_head"], "recorded git head did not resolve exactly")
    _require(resolved_tree == run["git_tree"], "recorded git tree differs")
    recorded = {
        "git_head": _read_single_line(provenance / "repo_head.txt", "recorded repo HEAD"),
        "git_tree": _read_single_line(provenance / "repo_tree.txt", "recorded repo tree"),
        "job_id": _read_single_line(provenance / "slurm_job_id.txt", "recorded Slurm job"),
        "node": _read_single_line(provenance / "node.txt", "recorded node"),
        "gpu_uuid": _read_single_line(provenance / "selected_gpu_uuid.txt", "selected GPU UUID"),
        "gpu_name": _read_single_line(provenance / "gpu_name.txt", "recorded GPU name"),
    }
    _require(all(recorded[key] == run[key] for key in recorded), "scalar provenance differs from run.json")
    _require((provenance / "repo_status.txt").read_text() == "", "run repository was dirty")
    diff_digest = _read_single_line(provenance / "repo_diff.sha256", "repo diff digest").split()[0]
    _require(diff_digest == _EMPTY_SHA256, "run repository diff was nonempty")
    for name in ("allocated_gpu_uuids.csv", "visible_gpu_uuids.csv"):
        _require(_read_single_line(provenance / name, name) == TARGET_GPU_UUID, f"{name} does not prove exclusive GPU visibility")

    source = _validate_manifest(
        provenance / "source_manifest.sha256",
        expected_digest=run["source_manifest_sha256"],
        relative_base=repo,
        label="source manifest",
    )
    source_names = {str(Path(row["path"]).resolve().relative_to(repo)) for row in source["entries"]}
    _require(SOURCE_REQUIRED.issubset(source_names), f"source manifest lacks required files: {sorted(SOURCE_REQUIRED - source_names)}")
    final_source = provenance / "source_manifest.final.sha256"
    _require(final_source.is_file(), "final source manifest is missing")
    _require(final_source.read_bytes() == (provenance / "source_manifest.sha256").read_bytes(), "source changed during the gate")
    inputs = _validate_manifest(
        provenance / "input_manifest.sha256",
        expected_digest=run["input_manifest_sha256"],
        relative_base=None,
        label="input manifest",
    )
    input_paths = {Path(row["path"]).resolve() for row in inputs["entries"]}
    _require(EXPECTED_PARTICLE_STACK.resolve() in input_paths, "input manifest lacks the GF46 particle stack")

    runtime = {
        "cuda": _parse_sha_record(provenance / "qualified_cuda.sha256", run["cuda_sha256"], "qualified CUDA"),
        "relion_bind": _parse_sha_record(provenance / "relion_bind.sha256", run["relion_bind_sha256"], "RELION binding"),
        "interpreter": _parse_sha_record(provenance / "interpreter.sha256", run["interpreter_sha256"], "pixi interpreter"),
        "cusparse": _parse_sha_record(provenance / "cusparse.sha256", run["cusparse_sha256"], "cuSPARSE preload"),
    }
    execution = _validate_execution_order(provenance / "execution_order.tsv")
    raw_target_preflight = _validate_raw_target_preflight(root)
    raw_command = _validate_raw_command(root)
    timing_commands = {
        label: _validate_timing_command(root, label, macro)
        for label, macro, _ in ARM_SPECS
    }
    run_dirs = {path.name for path in (root / "runs").iterdir() if path.is_dir()}
    _require(run_dirs == set(ARM_LABELS), f"timing run topology differs: {sorted(run_dirs)}")
    timing_diagnostics = [
        path
        for path in (root / "runs").rglob("*")
        if path.is_file() and path.name.startswith("coarse_gemm")
    ]
    _require(not timing_diagnostics, f"timing panel contains paired diagnostic artifacts: {timing_diagnostics}")
    return run, {
        "run_json_sha256": _sha256(run_path),
        "source_manifest": source,
        "input_manifest": inputs,
        "runtime": runtime,
        "git_repository": {"path": str(repo), "resolved_head": resolved_head, "resolved_tree": resolved_tree},
        "execution_order": execution,
        "raw_target_preflight": raw_target_preflight,
        "raw_command": raw_command,
        "timing_commands": timing_commands,
    }


def _raw_artifact_summary(path: Path, *, expected_run_id: str, expected_call_id: str) -> dict[str, Any]:
    _require(path.is_file(), f"raw score artifact is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required_arrays = (
                "direct_scores_pre_prior",
                "macro_scores_pre_prior",
                "direct_scores_with_prior",
                "macro_scores_with_prior",
                "direct_support",
                "macro_support",
                "score_delta",
                "ulp_score_delta",
                "argmax_equal",
                "support_equal",
                "original_indices",
                "automatic_no_go_reasons",
                "pending_qualification_gates",
            )
            missing = [key for key in required_arrays if key not in archive.files]
            _require(not missing, f"raw score artifact lacks arrays: {missing}")
            _require(_np_scalar(archive, "diagnostic_run_id") == expected_run_id, "raw artifact run ID differs")
            _require(_np_scalar(archive, "diagnostic_call_id") == expected_call_id, "raw artifact call ID differs")
            _require(_np_scalar(archive, "diagnostic_selection_policy") == "explicit_call_scope_intersection", "raw artifact selection policy differs")
            _require(_np_scalar(archive, "layout") == "image,class,rotation,translation", "raw score layout differs")
            _require(_np_scalar(archive, "qualification_status") == "NO_GO_UNQUALIFIED", "raw diagnostic qualification status differs")
            _require(bool(_np_scalar(archive, "paired_capture_active")), "paired capture was not active")
            _require(not bool(_np_scalar(archive, "clean_timing_eligible")), "raw paired capture was marked timing eligible")
            _require(not bool(_np_scalar(archive, "requires_bitwise_score_identity")), "raw policy unexpectedly requires bitwise scores")
            _require(bool(_np_scalar(archive, "requires_exact_discrete_identity")), "raw policy did not require exact discretes")
            _require(_integer(_np_scalar(archive, "debug_iteration"), "raw debug iteration") == 181, "raw diagnostic iteration differs")
            _require(
                _integer(_np_scalar(archive, "current_size"), "raw current size")
                == EXPECTED_RAW_SCORE_CURRENT_SIZE,
                "raw current size differs",
            )
            reasons = [str(value) for value in np.asarray(archive["automatic_no_go_reasons"]).reshape(-1)]
            _require(not reasons, f"raw score diagnostic has automatic NO-GO reasons: {reasons}")
            pending = [str(value) for value in np.asarray(archive["pending_qualification_gates"]).reshape(-1)]
            _require(bool(pending), "raw diagnostic incorrectly claims all empirical gates complete")

            arrays = {
                key: np.asarray(archive[key])
                for key in (
                    "direct_scores_pre_prior",
                    "macro_scores_pre_prior",
                    "direct_scores_with_prior",
                    "macro_scores_with_prior",
                )
            }
            shapes = {value.shape for value in arrays.values()}
            _require(len(shapes) == 1, f"paired raw score shapes differ: {shapes}")
            shape = next(iter(shapes))
            _require(
                len(shape) == 4
                and shape[0] > 0
                and shape[1] == 1
                and shape[2] > 0
                and shape[3] > 0,
                f"raw score shape is not a non-empty K=1 particle batch: {shape}",
            )
            _require(all(np.all(np.isfinite(value)) for value in arrays.values()), "raw score artifact contains non-finite scores")
            direct_support = np.asarray(archive["direct_support"], dtype=bool)
            macro_support = np.asarray(archive["macro_support"], dtype=bool)
            _require(direct_support.shape == shape and macro_support.shape == shape, "raw support shape differs")
            _require(np.array_equal(direct_support, macro_support), "raw direct/GEMM support differs")
            _require(np.all(np.asarray(archive["support_equal"], dtype=bool)), "raw support_equal is false")
            _require(np.all(np.asarray(archive["argmax_equal"], dtype=bool)), "raw argmax_equal is false")
            direct = arrays["direct_scores_with_prior"]
            macro = arrays["macro_scores_with_prior"]
            expected_delta = macro.astype(np.float64) - direct.astype(np.float64)
            _require(np.array_equal(np.asarray(archive["score_delta"]), expected_delta), "raw score_delta does not match paired scores")
            direct_argmax = np.argmax(direct.reshape(shape[0], -1), axis=1)
            macro_argmax = np.argmax(macro.reshape(shape[0], -1), axis=1)
            _require(np.array_equal(direct_argmax, macro_argmax), "raw paired argmax differs")
            _require(_integer(_np_scalar(archive, "macro_only_negative_implied_diff2_count"), "negative implied diff2 count") == 0, "GEMM introduced negative implied diff2")
            _require(not np.any(np.asarray(archive["exact_zero_direct_nonzero_macro_per_image"], dtype=bool)), "GEMM drifted from an exactly zero direct surface")
            for resource in (
                "resource_full_centered_projection_bytes",
                "resource_compact_projection_bytes",
                "resource_compact_projection_abs2_bytes",
                "resource_predicted_peak_projection_bytes",
                "resource_projected_transient_budget_bytes",
            ):
                _require(_integer(_np_scalar(archive, resource), resource) > 0, f"{resource} must be positive")
            predicted = _integer(_np_scalar(archive, "resource_predicted_peak_projection_bytes"), "predicted projection bytes")
            budget = _integer(_np_scalar(archive, "resource_projected_transient_budget_bytes"), "projection budget")
            _require(predicted <= budget, "raw score projection transient exceeds its budget")
            _require(_integer(_np_scalar(archive, "resource_pixel_index_device_to_host_materializations"), "pixel index materializations") == 0, "raw scorer materialized pixel indices device-to-host")
            original_indices = [int(value) for value in np.asarray(archive["original_indices"]).reshape(-1)]
            _require(
                len(original_indices) == shape[0]
                and len(set(original_indices)) == len(original_indices),
                "raw artifact target identities do not match its image axis",
            )
            pre_delta = arrays["macro_scores_pre_prior"].astype(np.float64) - arrays["direct_scores_pre_prior"].astype(np.float64)
            ulps = np.asarray(archive["ulp_score_delta"], dtype=np.uint64)
            return {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "original_indices": original_indices,
                "shape": list(shape),
                "pending_qualification_gates": pending,
                "with_prior": {
                    "max_abs_delta": float(np.max(np.abs(expected_delta))),
                    "rms_delta": float(np.sqrt(np.mean(np.square(expected_delta)))),
                    "signed_mean_delta": float(np.mean(expected_delta)),
                    "nonzero_delta_count": int(np.count_nonzero(expected_delta)),
                    "max_ulp_delta": int(np.max(ulps)),
                },
                "pre_prior": {
                    "max_abs_delta": float(np.max(np.abs(pre_delta))),
                    "rms_delta": float(np.sqrt(np.mean(np.square(pre_delta)))),
                    "signed_mean_delta": float(np.mean(pre_delta)),
                    "nonzero_delta_count": int(np.count_nonzero(pre_delta)),
                },
                "argmax_exact": True,
                "support_exact": True,
                "automatic_no_go_reasons": [],
                "clean_timing_eligible": False,
                "resource_predicted_peak_projection_bytes": predicted,
                "resource_projected_transient_budget_bytes": budget,
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, GemmGateSetupError):
            raise
        raise GemmGateSetupError(f"cannot read raw score artifact {path}: {exc}") from exc


def _validate_raw_score_diagnostic(root: Path) -> dict[str, Any]:
    diagnostic_root = root / "raw_score_diagnostic" / "artifacts"
    _require(diagnostic_root.is_dir(), "raw score diagnostic directory is missing")
    aggregate_paths = sorted(diagnostic_root.glob("coarse_gemm_manifest_*.json"))
    scope_paths = sorted(diagnostic_root.glob("coarse_gemm_scope_*.json"))
    artifact_paths = sorted(diagnostic_root.glob("coarse_gemm_ab_*.npz"))
    _require(len(aggregate_paths) == 1, f"raw diagnostic must have one aggregate manifest: {aggregate_paths}")
    _require(len(scope_paths) == 1, f"raw diagnostic must have one joint-halfset scope: {scope_paths}")
    _require(len(artifact_paths) == 2, f"raw diagnostic must have two target artifacts: {artifact_paths}")
    aggregate = _load_json(aggregate_paths[0], "raw aggregate manifest")
    _require(aggregate.get("schema_version") == 1, "raw aggregate schema differs")
    expected_counts = {str(target): 1 for target in RAW_TARGETS}
    _require(aggregate.get("requested_original_indices") == list(RAW_TARGETS), "raw aggregate target indices differ")
    _require(aggregate.get("captured_target_counts") == expected_counts, "raw aggregate target counts differ")
    _require(aggregate.get("all_requested_captured_exactly_once") is True, "raw aggregate is not exact-once")
    run_id = aggregate.get("run_id")
    call_ids = aggregate.get("expected_call_ids")
    scope_records = aggregate.get("scope_records")
    _require(isinstance(run_id, str) and run_id, "raw aggregate run ID is invalid")
    _require(
        isinstance(call_ids, list)
        and len(call_ids) == 1
        and len(set(call_ids)) == 1,
        "raw aggregate call topology differs",
    )
    _require(
        isinstance(scope_records, list) and len(scope_records) == len(call_ids),
        "raw aggregate scope records differ",
    )
    _require(len(scope_paths) == len(call_ids), "raw persisted scope count differs")
    summaries = []
    seen_artifacts: set[str] = set()
    for call_id, record, scope_path in zip(call_ids, scope_records, scope_paths, strict=True):
        persisted = _load_json(scope_path, f"raw scope {call_id}")
        _require(persisted == record, f"raw aggregate scope record differs from {scope_path.name}")
        _require(record.get("run_id") == run_id and record.get("call_id") == call_id, "raw scope identity differs")
        _require(record.get("expected_call_ids") == call_ids, "raw scope expected calls differ")
        _require(record.get("requested_original_indices") == list(RAW_TARGETS), "raw scope requested targets differ")
        _require(record.get("targets_in_scope") == list(RAW_TARGETS), "raw joint scope targets differ")
        _require(record.get("targets_explicitly_out_of_scope") == [], "raw joint scope excluded a target")
        _require(record.get("selection_policy") == "explicit_call_scope_intersection", "raw scope selection policy differs")
        counts = record.get("captured_target_counts")
        _require(counts == expected_counts, "raw scope target counts differ")
        names = record.get("artifact_paths")
        _require(isinstance(names, list) and len(names) == 2, "raw joint scope must name two artifacts")
        for name in names:
            _require(
                isinstance(name, str)
                and Path(name).name == name
                and name not in seen_artifacts,
                "raw scope artifact name is unsafe or repeated",
            )
            seen_artifacts.add(name)
            artifact = diagnostic_root / name
            _require(artifact in artifact_paths, f"raw scope names an unexpected artifact: {name}")
            summaries.append(
                _raw_artifact_summary(
                    artifact,
                    expected_run_id=run_id,
                    expected_call_id=call_id,
                )
            )
    _require(
        seen_artifacts == {path.name for path in artifact_paths},
        "raw score artifacts are not exactly covered by scope manifests",
    )
    _require(
        all(len(summary["original_indices"]) == 1 for summary in summaries),
        "each raw artifact must capture exactly one pinned target",
    )
    captured = sorted(index for summary in summaries for index in summary["original_indices"])
    _require(captured == list(RAW_TARGETS), f"raw artifacts captured the wrong particles: {captured}")

    output_root = root / "raw_score_diagnostic" / "output"
    meta_path = output_root / "run_it181_recovar_meta.json"
    _require(meta_path.is_file(), "raw one-shot metadata is missing")
    _require(sorted(output_root.glob("run_it*_recovar_meta.json")) == [meta_path], "raw diagnostic wrote more than iteration 181")
    metadata = _load_json(meta_path, "raw one-shot metadata")
    _require(metadata.get("joint_halfset_particle_stream") is True, "raw diagnostic was not joint-halfset")
    selected = metadata.get("selected_particle_ids")
    _require(
        isinstance(selected, list)
        and len(selected) == 1_000
        and len(set(selected)) == 1_000
        and all(isinstance(value, int) and not isinstance(value, bool) for value in selected),
        "raw selected-particle state differs",
    )
    _require(
        [selected[position] for position in RAW_TARGET_POSITIONS] == list(RAW_TARGETS),
        "raw target positions differ from the frozen continuation preflight",
    )
    recorded_aggregate = _resolved_inside(
        Path(str(metadata.get("coarse_gaussian_gemm_aggregate_manifest_path", ""))),
        root,
        "raw aggregate path",
    )
    _require(recorded_aggregate == aggregate_paths[0].resolve(), "raw metadata aggregate path differs")
    return {
        "aggregate_manifest": {"path": str(aggregate_paths[0].resolve()), "sha256": _sha256(aggregate_paths[0])},
        "scope_manifest_count": len(scope_paths),
        "artifact_count": len(artifact_paths),
        "captured_original_indices": captured,
        "all_requested_captured_exactly_once": True,
        "artifacts": summaries,
        "all_scores_finite": True,
        "all_argmax_exact": True,
        "all_support_exact": True,
        "automatic_no_go_reasons": [],
        "timing_eligible": False,
        "pass": True,
    }


def _validate_schedule(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} schedule is missing")
    _require(value == EXPECTED_SCHEDULE, f"{label} GF46 iteration-181 schedule differs: {value}")
    return dict(value)


def _model_state(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        blocks = starfile.read(path, always_dict=True)
    except Exception as exc:
        raise GemmGateSetupError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(blocks, dict) and blocks, f"{label} is empty")
    identity: dict[str, Any] = {}
    continuous: list[tuple[str, float]] = []
    for block_name, block in blocks.items():
        if isinstance(block, dict):
            for key, value in block.items():
                item_key = f"{block_name}.{key}"
                if isinstance(value, bool) or isinstance(value, (int, np.integer)):
                    identity[item_key] = int(value)
                elif isinstance(value, (float, np.floating)):
                    parsed = float(value)
                    _require(math.isfinite(parsed), f"{label} {item_key} is non-finite")
                    continuous.append((item_key, parsed))
                else:
                    text = str(value)
                    identity[item_key] = Path(text).name if key == "rlnReferenceImage" else text
            continue
        _require(hasattr(block, "columns"), f"{label} block {block_name} has unsupported type")
        identity[f"{block_name}.__rows__"] = int(len(block))
        identity[f"{block_name}.__columns__"] = [str(column) for column in block.columns]
        for column in block.columns:
            values = block[column].to_numpy()
            item_prefix = f"{block_name}.{column}"
            if np.issubdtype(values.dtype, np.integer):
                identity[item_prefix] = [int(value) for value in values]
            elif np.issubdtype(values.dtype, np.floating):
                parsed_values = np.asarray(values, dtype=np.float64)
                _require(np.all(np.isfinite(parsed_values)), f"{label} {item_prefix} is non-finite")
                continuous.extend((f"{item_prefix}[{index}]", float(value)) for index, value in enumerate(parsed_values))
            else:
                normalized = [
                    Path(str(value)).name if str(column) == "rlnReferenceImage" else str(value)
                    for value in values
                ]
                identity[item_prefix] = normalized
    _require(bool(continuous), f"{label} has no continuous model state")
    continuous.sort(key=lambda row: row[0])
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "identity": identity,
        "continuous_keys": [row[0] for row in continuous],
        "continuous_values": np.asarray([row[1] for row in continuous], dtype=np.float64),
    }


def _parse_process_time(path: Path, label: str) -> int:
    _require(path.is_file() and path.stat().st_size > 0, f"{label} process.time is missing")
    matches = re.findall(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text())
    _require(len(matches) == 1, f"{label} process.time lacks one peak RSS")
    value = int(matches[0])
    _require(value > 0, f"{label} process.time peak RSS is invalid")
    return value


def _load_phase(root: Path, label: str, phase_name: str, phase: dict[str, Any]) -> dict[str, Any]:
    phase_root = root / "runs" / label / "profile" / phase_name
    meta_path = phase_root / "run_it181_recovar_meta.json"
    recorded_meta = _resolved_inside(Path(str(phase.get("meta_path", ""))), root, f"{label} {phase_name} metadata")
    _require(recorded_meta == meta_path.resolve(), f"{label} {phase_name} metadata path differs")
    _require(_sha256(meta_path) == phase.get("meta_sha256"), f"{label} {phase_name} metadata digest differs")
    metadata = _load_json(meta_path, f"{label} {phase_name} metadata")
    schedule = _validate_schedule(phase.get("schedule"), f"{label} {phase_name}")
    for key, value in schedule.items():
        _require(_values_equal(metadata.get(key), value), f"{label} {phase_name} metadata schedule differs for {key}")
    _require("coarse_gaussian_gemm_aggregate_manifest_path" not in metadata, f"{label} {phase_name} contains a timing-ineligible GEMM diagnostic")
    selected = metadata.get("selected_particle_ids")
    _require(isinstance(selected, list) and len(selected) == 1_000 and len(set(selected)) == 1_000, f"{label} {phase_name} selected-particle state differs")
    for key in DISCRETE_META_KEYS:
        values = metadata.get(key)
        _require(isinstance(values, list) and len(values) == 1_000, f"{label} {phase_name} {key} length differs")
    _require(metadata.get("joint_halfset_particle_stream") is True, f"{label} {phase_name} is not joint-halfset")
    _require(_values_equal(metadata.get("halfset_ids"), [0, 1]), f"{label} {phase_name} halfsets differ")
    profile_keys = sorted(key for key in metadata if re.fullmatch(r"halfset_\d+_profile_summary", str(key)))
    _require(profile_keys == ["halfset_0_profile_summary"], f"{label} {phase_name} halfset profile topology differs")
    sparse = metadata.get("sparse_pass2_profile_summary")
    _require(isinstance(sparse, dict), f"{label} {phase_name} sparse profile is missing")
    pass1 = _finite_number(sparse.get("pass1_time_s"), f"{label} {phase_name} pass1", positive=True)
    pass2 = _finite_number(sparse.get("pass2_time_s"), f"{label} {phase_name} pass2", positive=True)
    iteration = phase.get("iteration_profile")
    _require(isinstance(iteration, dict), f"{label} {phase_name} iteration profile is missing")
    expectation = _finite_number(iteration.get("expectation_time_s"), f"{label} {phase_name} expectation", positive=True)
    resources = phase.get("process_resources")
    _require(isinstance(resources, dict) and isinstance(resources.get("after"), dict), f"{label} {phase_name} process resources are missing")
    after = resources["after"]
    high_water_kb = _integer(after.get("high_water_rss_kb"), f"{label} {phase_name} high-water RSS")
    max_rss_kb = _integer(after.get("max_rss_kb"), f"{label} {phase_name} max RSS")
    current_rss_kb = _integer(after.get("current_rss_kb"), f"{label} {phase_name} current RSS")
    _require(high_water_kb > 0 and max_rss_kb > 0 and high_water_kb >= current_rss_kb, f"{label} {phase_name} RSS counters are invalid")
    meta_files = sorted(phase_root.glob("run_it*_recovar_meta.json"))
    _require(meta_files == [meta_path], f"{label} {phase_name} wrote unexpected iterations")
    star_path = phase_root / "run_it181_data.star"
    model_path = phase_root / "run_it181_model.star"
    map_path = phase_root / "run_it181_class001.mrc"
    star = _load_star(star_path, f"{label} {phase_name} particle STAR")
    _require(len(star["particles"]) == 3_000, f"{label} {phase_name} particle STAR row count differs")
    return {
        "metadata": metadata,
        "schedule": schedule,
        "star": star,
        "star_sha256": _sha256(star_path),
        "map": _load_map(map_path, f"{label} {phase_name} map"),
        "map_sha256": _sha256(map_path),
        "model_state": _model_state(model_path, f"{label} {phase_name} model state"),
        "timing": {
            "wall_s": _finite_number(phase.get("wall_s"), f"{label} {phase_name} wall", positive=True),
            "expectation_s": expectation,
            "pass1_s": pass1,
            "pass2_s": pass2,
            "profile_high_water_rss_kb": high_water_kb,
            "profile_max_rss_kb": max_rss_kb,
        },
    }


def _load_arm(root: Path, spec: tuple[str, int, int]) -> dict[str, Any]:
    label, macro, repeat = spec
    run_root = root / "runs" / label
    for name in ("runner.stdout", "runner.stderr", "process.time"):
        _require((run_root / name).is_file(), f"{label} lacks {name}")
    summary_path = run_root / "profile" / "profile_summary.json"
    summary = _load_json(summary_path, f"{label} profile summary")
    expected = {
        "schema": PROFILE_SCHEMA,
        "classification": "diagnostic_performance_only",
        "checkpoint_iteration": 180,
        "profiled_iteration": 181,
        "nr_iter_schedule": 200,
        "cuda_profiler_range": False,
        "raw_image_cache_audit_enabled": False,
        "exact_local_bucket_radix": 4,
        "exact_local_physical_order_chunk_size": 0,
        "checkpoint_optimiser": str(EXPECTED_CHECKPOINT_OPTIMISER),
        "checkpoint_optimiser_sha256": EXPECTED_CHECKPOINT_OPTIMISER_SHA256,
        "input_star": str(EXPECTED_INPUT_STAR),
        "input_star_sha256": EXPECTED_INPUT_STAR_SHA256,
        "data_dir": str(EXPECTED_DATA_DIR),
    }
    mismatches = {key: {"expected": value, "observed": summary.get(key)} for key, value in expected.items() if summary.get(key) != value}
    _require(not mismatches, f"{label} profile summary differs: {mismatches}")
    _require(isinstance(summary.get("cold"), dict) and isinstance(summary.get("warm"), dict), f"{label} profile phases differ")
    cold = _load_phase(root, label, "cold", summary["cold"])
    warm = _load_phase(root, label, "warm", summary["warm"])
    _require(cold["schedule"] == warm["schedule"], f"{label} cold/warm schedule differs")
    logs = (run_root / "runner.stdout").read_text() + "\n" + (run_root / "runner.stderr").read_text()
    marker_count = logs.count(GEMM_LOG_MARKER)
    if macro:
        _require(marker_count > 0, f"{label} has no effective GEMM execution marker")
    else:
        _require(marker_count == 0, f"{label} unexpectedly executed GEMM")
    time_peak_kb = _parse_process_time(run_root / "process.time", label)
    warm_timing = warm["timing"]
    peak_kb = max(time_peak_kb, warm_timing["profile_high_water_rss_kb"], warm_timing["profile_max_rss_kb"])
    performance = {
        "warm_wall_s": warm_timing["wall_s"],
        "warm_expectation_s": warm_timing["expectation_s"],
        "warm_pass1_s": warm_timing["pass1_s"],
        "warm_pass2_s": warm_timing["pass2_s"],
        "peak_rss_gib": float(peak_kb) / (1024.0**2),
        "profile_high_water_rss_kb": warm_timing["profile_high_water_rss_kb"],
        "profile_max_rss_kb": warm_timing["profile_max_rss_kb"],
        "time_max_rss_kb": time_peak_kb,
    }
    return {
        "label": label,
        "macro": bool(macro),
        "repeat": repeat,
        "profile_summary_sha256": _sha256(summary_path),
        "gemm_execution_marker_count": marker_count,
        "cold": cold,
        "warm": warm,
        "performance": performance,
    }


def _discrete_checks(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cold_warm: dict[str, Any] = {}
    all_exact = True
    for label in ARM_LABELS:
        arm = arms[label]
        metadata = {
            key: key in arm["cold"]["metadata"]
            and key in arm["warm"]["metadata"]
            and _values_equal(arm["cold"]["metadata"][key], arm["warm"]["metadata"][key])
            for key in DISCRETE_META_KEYS
        }
        star_exact = _star_equal(arm["cold"]["star"], arm["warm"]["star"])
        model_identity_exact = arm["cold"]["model_state"]["identity"] == arm["warm"]["model_state"]["identity"]
        exact = all(metadata.values()) and star_exact and model_identity_exact
        all_exact &= exact
        cold_warm[label] = {
            "metadata_exact": metadata,
            "particle_star_exact": star_exact,
            "model_discrete_identity_exact": model_identity_exact,
            "all_exact": exact,
        }
    reference = arms["direct_1"]["warm"]
    warm_panel: dict[str, Any] = {}
    for label in ARM_LABELS[1:]:
        candidate = arms[label]["warm"]
        metadata = {
            key: key in reference["metadata"]
            and key in candidate["metadata"]
            and _values_equal(reference["metadata"][key], candidate["metadata"][key])
            for key in DISCRETE_META_KEYS
        }
        star_exact = _star_equal(reference["star"], candidate["star"])
        model_identity_exact = reference["model_state"]["identity"] == candidate["model_state"]["identity"]
        model_keys_exact = reference["model_state"]["continuous_keys"] == candidate["model_state"]["continuous_keys"]
        exact = all(metadata.values()) and star_exact and model_identity_exact and model_keys_exact
        all_exact &= exact
        warm_panel[f"direct_1__{label}"] = {
            "metadata_exact": metadata,
            "particle_star_exact": star_exact,
            "model_discrete_identity_exact": model_identity_exact,
            "model_continuous_keys_exact": model_keys_exact,
            "all_exact": exact,
        }
    return {
        "metadata_keys": list(DISCRETE_META_KEYS),
        "cold_warm": cold_warm,
        "warm_panel": warm_panel,
        "all_schedule_particle_star_and_discrete_state_exact": all_exact,
        "pass": all_exact,
    }


def _numeric_repeat_panel(
    arms: dict[str, dict[str, Any]],
    *,
    field: str,
) -> dict[str, Any]:
    def value(label: str) -> np.ndarray:
        warm = arms[label]["warm"]
        if field == "map":
            return np.asarray(warm["map"], dtype=np.float64)
        _require(field == "model_state", f"unknown numeric field {field}")
        return np.asarray(warm["model_state"]["continuous_values"], dtype=np.float64)

    direct_repeat = _map_delta(value("direct_1"), value("direct_2"))
    gemm_repeat = _map_delta(value("gemm_1"), value("gemm_2"))
    envelope = _delta_envelope(direct_repeat)
    gemm_repeat_check = _against_envelope(gemm_repeat, envelope)
    cross = {}
    signed_means = []
    all_cross_bounded = True
    for repeat in (1, 2):
        delta = _map_delta(value(f"direct_{repeat}"), value(f"gemm_{repeat}"))
        check = _against_envelope(delta, envelope)
        signed_means.append(float(delta["signed_mean"]))
        all_cross_bounded &= bool(check["within_control_repeat_envelope"])
        cross[f"direct_{repeat}__gemm_{repeat}"] = {**delta, **check}
    signed_nondirectional = min(signed_means) <= 0.0 <= max(signed_means)
    panel_pass = (
        bool(gemm_repeat_check["within_control_repeat_envelope"])
        and all_cross_bounded
        and signed_nondirectional
    )
    return {
        "direct_repeat": direct_repeat,
        "gemm_repeat": {**gemm_repeat, **gemm_repeat_check},
        "direct_control_repeat_envelope": envelope,
        "cross_arm": cross,
        "cross_arm_signed_means": signed_means,
        "cross_arm_signed_drift_opposes_or_is_zero": signed_nondirectional,
        "gemm_repeat_within_direct_envelope": gemm_repeat_check["within_control_repeat_envelope"],
        "all_cross_arm_deltas_within_direct_envelope": all_cross_bounded,
        "pass": panel_pass,
    }


def _science(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    discrete = _discrete_checks(arms)
    maps = _numeric_repeat_panel(arms, field="map")
    model_state = _numeric_repeat_panel(arms, field="model_state")
    return {
        "discrete": discrete,
        "map_repeat_envelope": maps,
        "model_state_repeat_envelope": model_state,
        "long_trajectory_no_growth_evaluated": False,
        "pass": discrete["pass"] and maps["pass"] and model_state["pass"],
    }


def _performance(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    modes = {
        "direct": [arms["direct_1"]["performance"], arms["direct_2"]["performance"]],
        "gemm": [arms["gemm_1"]["performance"], arms["gemm_2"]["performance"]],
    }
    raw = {
        mode: {
            f"R{index + 1}": {metric: float(row[metric]) for metric in TIMING_METRICS}
            for index, row in enumerate(rows)
        }
        for mode, rows in modes.items()
    }
    medians = {
        mode: {
            metric: float(median([row[metric] for row in rows]))
            for metric in TIMING_METRICS
        }
        for mode, rows in modes.items()
    }
    ratios = {
        metric: medians["gemm"][metric] / medians["direct"][metric]
        for metric in TIMING_METRICS
    }
    wall_ratio = ratios["warm_wall_s"]
    return {
        "metric_order": list(TIMING_METRICS),
        "raw": raw,
        "medians": medians,
        "gemm_over_direct_ratio": ratios,
        "material_end_to_end_ratio_threshold": MATERIAL_END_TO_END_RATIO,
        "median_warm_wall_ratio": wall_ratio,
        "median_warm_wall_speedup": 1.0 / wall_ratio,
        "crossed_median_end_to_end_gate_pass": wall_ratio <= MATERIAL_END_TO_END_RATIO,
        "pass": wall_ratio <= MATERIAL_END_TO_END_RATIO,
    }


def _markdown(report: dict[str, Any]) -> str:
    acceptance = report["acceptance"]
    performance = report["performance"]
    science = report["science"]
    raw = report["raw_score_diagnostic"]
    status = "PASS" if acceptance["pass"] else "FAIL"

    def mark(value: bool) -> str:
        return "PASS" if value else "FAIL"

    rows = [
        f"# OVERALL: {status} — GF46 coarse shared-GEMM gate",
        "",
        f"> **Performance-only one-transition gate: {status}. The GEMM path remains default-off.**",
        "",
        "- Frozen transition: GF46 iteration `180 → 181`, K=1, pinned H100/UUID.",
        "- Timing order: `direct_1 → gemm_1 → gemm_2 → direct_2`.",
        "- The paired score capture is separate and timing-ineligible.",
        "- Long-trajectory no-growth and production default enablement are not evaluated.",
        "",
        "## Gate dashboard",
        "",
        "| Gate | Requirement | Result | Status |",
        "|---|---|---|---:|",
        f"| Raw paired score capture | particles 1,2160 exact-once in one joint-halfset scope; finite; support/argmax exact; no automatic NO-GO | {raw['artifact_count']} artifacts | {mark(acceptance['raw_score_surface'])} |",
        f"| Timing diagnostics | Both diagnostic variables explicitly unset in all four arms | {acceptance['timing_diagnostics_unset']} | {mark(acceptance['timing_diagnostics_unset'])} |",
        f"| Schedule and discrete state | Cold/warm and all arms exact | {science['discrete']['pass']} | {mark(acceptance['schedule_and_discrete_state_exact'])} |",
        f"| Map repeat envelope | GEMM repeat and both crossed pairs ≤ direct repeat envelope | {science['map_repeat_envelope']['pass']} | {mark(acceptance['map_repeat_envelope'])} |",
        f"| Model-state repeat envelope | GEMM repeat and both crossed pairs ≤ direct repeat envelope | {science['model_state_repeat_envelope']['pass']} | {mark(acceptance['model_state_repeat_envelope'])} |",
        f"| Median warm end-to-end | GEMM/direct ≤ {MATERIAL_END_TO_END_RATIO:.2f} | {performance['median_warm_wall_ratio']:.4f} | {mark(acceptance['material_end_to_end_runtime'])} |",
        f"| Bounded gate | Every gate above passes | {acceptance['pass']} | {mark(acceptance['pass'])} |",
        "| Long trajectory / default | Required before enablement | Not evaluated / false | OPEN |",
        "",
        "## Warm A/B/B/A performance",
        "",
        "| Mode | R1 wall (s) | R2 wall (s) | Median wall (s) | Expectation (s) | Pass 1 (s) | Peak RSS (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("direct", "gemm"):
        mode_raw = performance["raw"][mode]
        mode_median = performance["medians"][mode]
        rows.append(
            f"| {mode} | {mode_raw['R1']['warm_wall_s']:.6f} | {mode_raw['R2']['warm_wall_s']:.6f} | "
            f"{mode_median['warm_wall_s']:.6f} | {mode_median['warm_expectation_s']:.6f} | "
            f"{mode_median['warm_pass1_s']:.6f} | {mode_median['peak_rss_gib']:.3f} |"
        )
    rows.extend(
        (
            "",
            f"Median GEMM/direct warm wall ratio: `{performance['median_warm_wall_ratio']:.6f}` "
            f"(`{performance['median_warm_wall_speedup']:.3f}x` speedup).",
            "",
            "## Raw score-surface diagnostic",
            "",
            "| Original index | Shape | Max abs Δ (with prior) | RMS Δ | Signed mean Δ | Max ULP | Support | Argmax |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for artifact in raw["artifacts"]:
        metrics = artifact["with_prior"]
        rows.append(
            f"| {','.join(str(value) for value in artifact['original_indices'])} | `{artifact['shape']}` | {metrics['max_abs_delta']:.3e} | "
            f"{metrics['rms_delta']:.3e} | {metrics['signed_mean_delta']:.3e} | "
            f"{metrics['max_ulp_delta']} | {artifact['support_exact']} | {artifact['argmax_exact']} |"
        )
    rows.extend(("", "## Repeat-envelope summary", "", "| State | Direct repeat rel-L2 | GEMM repeat rel-L2 | Cross R1 rel-L2 | Cross R2 rel-L2 | Pass |", "|---|---:|---:|---:|---:|---:|"))
    for name, label in (("map_repeat_envelope", "map"), ("model_state_repeat_envelope", "model state")):
        panel = science[name]
        rows.append(
            f"| {label} | {panel['direct_repeat']['relative_l2']:.3e} | "
            f"{panel['gemm_repeat']['relative_l2']:.3e} | "
            f"{panel['cross_arm']['direct_1__gemm_1']['relative_l2']:.3e} | "
            f"{panel['cross_arm']['direct_2__gemm_2']['relative_l2']:.3e} | {panel['pass']} |"
        )
    provenance = report["provenance"]
    rows.extend(
        (
            "",
            "## Compact provenance",
            "",
            "| Evidence | Value |",
            "|---|---|",
            f"| Result root | `{provenance['root']}` |",
            f"| Slurm job | `{provenance['job_id']}` |",
            f"| Commit / tree | `{provenance['git_head']}` / `{provenance['git_tree']}` |",
            f"| Node / GPU UUID | `{provenance['node']}` / `{provenance['gpu_uuid']}` |",
            f"| Source manifest | `{provenance['source_manifest_sha256']}` |",
            f"| Input manifest | `{provenance['input_manifest_sha256']}` |",
            f"| Analyzer SHA-256 | `{provenance['analyzer_source_sha256']}` |",
            "",
            "## Promotion boundary",
            "",
            "Even a PASS is only a bounded iteration-180→181 performance result. It does not establish "
            "long-trajectory no-growth, cross-dataset basin/quality neutrality, or default enablement.",
            "",
        )
    )
    return "\n".join(rows)


def analyze(root: Path, *, repo: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    repo = (repo or Path(__file__).resolve().parents[1]).resolve()
    run, provenance_details = _validate_provenance(root, repo)
    raw = _validate_raw_score_diagnostic(root)
    arms = {spec[0]: _load_arm(root, spec) for spec in ARM_SPECS}
    schedules = [arms[label][phase]["schedule"] for label in ARM_LABELS for phase in ("cold", "warm")]
    _require(all(schedule == EXPECTED_SCHEDULE for schedule in schedules), "GF46 schedule differs across the panel")
    science = _science(arms)
    performance = _performance(arms)
    # Command validation above already proved the explicit ``env -u`` contract.
    timing_unset = True
    gate_pass = raw["pass"] and science["pass"] and performance["pass"] and timing_unset
    report = {
        "schema": SCHEMA,
        "provenance": {
            "root": str(root),
            "job_id": run["job_id"],
            "git_head": run["git_head"],
            "git_tree": run["git_tree"],
            "node": run["node"],
            "gpu_uuid": run["gpu_uuid"],
            "gpu_name": run["gpu_name"],
            "source_manifest_sha256": run["source_manifest_sha256"],
            "input_manifest_sha256": run["input_manifest_sha256"],
            "cuda_sha256": run["cuda_sha256"],
            "relion_bind_sha256": run["relion_bind_sha256"],
            "interpreter_sha256": run["interpreter_sha256"],
            "cusparse_sha256": run["cusparse_sha256"],
            "analyzer_source_sha256": _sha256(Path(__file__).resolve()),
            **provenance_details,
        },
        "raw_score_diagnostic": raw,
        "arms": {
            label: {
                "macro": arm["macro"],
                "repeat": arm["repeat"],
                "profile_summary_sha256": arm["profile_summary_sha256"],
                "gemm_execution_marker_count": arm["gemm_execution_marker_count"],
                "warm_map_sha256": arm["warm"]["map_sha256"],
                "warm_model_sha256": arm["warm"]["model_state"]["sha256"],
                "warm_particle_star_sha256": arm["warm"]["star_sha256"],
                "performance": arm["performance"],
            }
            for label, arm in arms.items()
        },
        "schedule": EXPECTED_SCHEDULE,
        "science": science,
        "performance": performance,
        "acceptance": {
            "topology_and_sealed_provenance": True,
            "raw_score_surface": raw["pass"],
            "timing_diagnostics_unset": timing_unset,
            "schedule_and_discrete_state_exact": science["discrete"]["pass"],
            "map_repeat_envelope": science["map_repeat_envelope"]["pass"],
            "model_state_repeat_envelope": science["model_state_repeat_envelope"]["pass"],
            "material_end_to_end_runtime": performance["pass"],
            "long_trajectory_no_growth_evaluated": False,
            "default_enablement_allowed": False,
            "pass": gate_pass,
        },
        "dashboard": {
            "overall_status": "PASS" if gate_pass else "FAIL",
            "scope": "diagnostic_performance_only",
            "transition": "iteration_180_to_181",
            "candidate": "coarse_gaussian_gemm_macro",
            "candidate_default_state": "off",
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
    except GemmGateSetupError as exc:
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
