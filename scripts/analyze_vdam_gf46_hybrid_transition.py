#!/usr/bin/env python3
"""Analyze the sealed GF46 direct/integrated-hybrid iteration-180->181 gate.

The two support-audit arms prove exact ordered coarse-candidate identity.  A
separate direct/hybrid/hybrid/direct panel measures steady-state performance
without the support audit.  This is a bounded one-transition qualification;
it cannot promote the default or change the frozen trajectory scorecard.
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

from scripts.analyze_vdam_coarse_gemm_gf46_gate import (
    DISCRETE_META_KEYS as _LEGACY_DISCRETE_META_KEYS,
    EXPECTED_CHECKPOINT_OPTIMISER,
    EXPECTED_CHECKPOINT_OPTIMISER_SHA256,
    EXPECTED_DATA_DIR,
    EXPECTED_INPUT_MANIFEST_SHA256,
    EXPECTED_INPUT_STAR,
    EXPECTED_INPUT_STAR_SHA256,
    EXPECTED_PARTICLE_STACK,
    EXPECTED_PARTICLE_STACK_SHA256,
    EXPECTED_SCHEDULE,
    PROFILE_SCHEMA,
    _against_envelope,
    _delta_envelope,
    _finite_number,
    _load_phase,
    _model_state,
    _parse_process_time,
)
from scripts.analyze_vdam_coarse_multistream_late_pair import (
    LatePairSetupError as HybridTransitionSetupError,
    _git_rev_parse,
    _load_json,
    _map_delta,
    _read_single_line,
    _require,
    _sha256,
    _star_equal,
    _validate_manifest,
    _values_equal,
)

SCHEMA = "recovar.vdam_gf46_hybrid_transition_analysis.v1"
RUN_SCHEMA = "recovar.vdam_gf46_hybrid_transition.v1"
MATERIAL_WALL_RATIO = 0.90
AUDIT_SPECS = (("audit_direct", False), ("audit_hybrid", True))
TIMING_SPECS = (
    ("direct_1", False, 1),
    ("hybrid_1", True, 1),
    ("hybrid_2", True, 2),
    ("direct_2", False, 2),
)
ALL_SPECS = tuple((name, hybrid) for name, hybrid in AUDIT_SPECS) + tuple(
    (name, hybrid) for name, hybrid, _repeat in TIMING_SPECS
)
TIMING_LABELS = tuple(name for name, _hybrid, _repeat in TIMING_SPECS)
DISCRETE_META_KEYS = tuple(_LEGACY_DISCRETE_META_KEYS) + ("significant_counts",)
TIMING_METRICS = (
    "warm_wall_s",
    "warm_expectation_s",
    "warm_pass1_s",
    "warm_pass2_s",
    "peak_rss_gib",
)
SUPPORT_SCHEMA = "recovar.coarse_significance_support_audit.v1"
EXPECTED_SUPPORT_ENCODING = (
    "class-major/image-major; uint64 row-byte-length; "
    "int64-le header(class,image,total,count); int64-le sorted IDs"
)
EXPECTED_SELECTED_IDS_INT64_SHA256 = (
    "c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be"
)
EXPECTED_RELION_BIND_SHA256 = (
    "9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980"
)
EXPECTED_INTERPRETER_SHA256 = (
    "1e43e23601e6369d52fd56b0405882463297c9ba6ad5238b460da90db6771b9c"
)
EXPECTED_CUSPARSE_SHA256 = (
    "58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf"
)
SOURCE_FILES = (
    "recovar/cuda/Makefile",
    "recovar/cuda/cuda_backproject.cu",
    "recovar/cuda/relion_coarse_diff2_projector_body.inc",
    "recovar/cuda_backproject.py",
    "recovar/data_io/image_loader.py",
    "recovar/em/dense_single_volume/batch_planning.py",
    "recovar/em/dense_single_volume/helpers/coarse_gemm_hybrid.py",
    "recovar/em/dense_single_volume/helpers/projection.py",
    "recovar/em/dense_single_volume/helpers/scoring.py",
    "recovar/em/dense_single_volume/helpers/significance.py",
    "recovar/em/dense_single_volume/k_class.py",
    "recovar/em/initial_model/dense_adapter.py",
    "recovar/em/initial_model/driver.py",
    "recovar/em/initial_model/iteration_loop.py",
    "recovar/em/initial_model/m_step.py",
    "recovar/em/initial_model/schedules.py",
    "scripts/analyze_vdam_coarse_gemm_gf46_gate.py",
    "scripts/analyze_vdam_coarse_multistream_late_pair.py",
    "scripts/analyze_vdam_gf46_hybrid_transition.py",
    "scripts/run_vdam_gf46_hybrid_transition.sbatch",
    "scripts/run_vdam_late_iteration_profile.py",
    "scripts/vdam_gpu_selection.sh",
    "tests/unit/initial_model/test_vdam_gf46_hybrid_transition.py",
    "pixi.lock",
    "pixi.toml",
    "pyproject.toml",
)
SOURCE_REQUIRED = frozenset(SOURCE_FILES)
COMMON_ENV = {
    "RECOVAR_EM_RAW_IMAGE_CACHE": "auto",
    "RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB": "16",
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
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB": "4",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY": "64",
}
HYBRID_ENV = {
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO": "1",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE": "1",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID": "1",
}
DIRECT_ENV = {
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO": "0",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE": "0",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID": "0",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _source_manifest(repo: Path) -> tuple[bytes, list[dict[str, Any]]]:
    lines: list[str] = []
    entries: list[dict[str, Any]] = []
    for relative in SOURCE_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"hybrid transition source is missing: {relative}")
        digest = _sha256(path)
        lines.append(f"{digest}  {relative}\n")
        entries.append({"path": relative, "sha256": digest, "size_bytes": path.stat().st_size})
    return "".join(lines).encode(), entries


def _parse_env_command(path: Path, label: str) -> tuple[dict[str, str], set[str], list[str]]:
    _require(path.is_file(), f"missing {label} command: {path}")
    try:
        tokens = shlex.split(path.read_text())
    except (OSError, ValueError) as exc:
        raise HybridTransitionSetupError(f"cannot parse {label} command: {exc}") from exc
    _require(tokens and tokens[0] == "env", f"{label} command does not start with env")
    assignments: dict[str, str] = {}
    unsets: set[str] = set()
    index = 1
    while index < len(tokens):
        if tokens[index] == "-u":
            _require(index + 1 < len(tokens), f"{label} has incomplete env -u")
            unsets.add(tokens[index + 1])
            index += 2
        elif "=" in tokens[index]:
            name, value = tokens[index].split("=", 1)
            _require(name and name not in assignments, f"{label} repeats env {name}")
            assignments[name] = value
            index += 1
        else:
            break
    return assignments, unsets, tokens[index:]


def _flag(argv: list[str], name: str, label: str) -> str:
    _require(argv.count(name) == 1, f"{label} must contain exactly one {name}")
    index = argv.index(name)
    _require(index + 1 < len(argv), f"{label} has incomplete {name}")
    return argv[index + 1]


def _validate_command(root: Path, label: str, *, hybrid: bool, audit: bool) -> dict[str, Any]:
    path = root / "provenance" / f"{label}_command.sh"
    assignments, unsets, argv = _parse_env_command(path, label)
    expected_env = {**COMMON_ENV, **(HYBRID_ENV if hybrid else DIRECT_ENV)}
    mismatch = {
        key: {"expected": value, "observed": assignments.get(key)}
        for key, value in expected_env.items()
        if assignments.get(key) != value
    }
    _require(not mismatch, f"{label} scorer environment differs: {mismatch}")
    audit_name = "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT"
    if audit:
        _require(assignments.get(audit_name) == "1", f"{label} did not enable support audit")
        _require(audit_name not in unsets, f"{label} also unsets support audit")
    else:
        _require(audit_name in unsets, f"{label} did not explicitly unset support audit")
        _require(audit_name not in assignments, f"{label} assigns support audit")
    _require(len(argv) >= 3 and argv[1:3] == ["-m", "scripts.run_vdam_late_iteration_profile"], f"{label} command target differs")
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
    mismatch = {
        name: {"expected": value, "observed": _flag(argv, name, label)}
        for name, value in expected_flags.items()
        if _flag(argv, name, label) != value
    }
    _require(not mismatch, f"{label} command flags differ: {mismatch}")
    _require("--cuda-profiler-range" not in argv and "--audit-raw-image-cache" not in argv, f"{label} enables timing instrumentation")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "environment": assignments}


def _validate_execution_order(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), "execution order is missing")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    expected = [
        {"order": str(index + 1), "label": label, "hybrid": str(int(hybrid)), "support_audit": str(int(label.startswith("audit_")))}
        for index, (label, hybrid) in enumerate(ALL_SPECS)
    ]
    _require(rows == expected, f"execution order differs: {rows}")
    return rows


def _validate_sha_record(path: Path, expected: str, label: str) -> dict[str, str]:
    _require(bool(_SHA256_RE.fullmatch(expected)), f"{label} digest is invalid")
    line = _read_single_line(path, label)
    digest, raw_path = line.split(maxsplit=1)
    _require(digest == expected, f"{label} digest record differs")
    target = Path(raw_path).resolve()
    _require(target.is_file() and _sha256(target) == digest, f"{label} target differs")
    return {"path": str(target), "sha256": digest}


def _validate_provenance(root: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require((root / "AUDITS_COMPLETED").is_file(), "support audits are incomplete")
    _require((root / "ARMS_COMPLETED").is_file(), "timing arms are incomplete")
    _require(not (root / "provenance/failure.txt").exists(), "run records a setup failure")
    run_path = root / "provenance/run.json"
    run = _load_json(run_path, "run provenance")
    expected = {
        "schema": RUN_SCHEMA,
        "classification": "diagnostic_one_transition_only",
        "execution_order": [name for name, _hybrid in ALL_SPECS],
        "support_audit_arms": [name for name, _hybrid in AUDIT_SPECS],
        "timing_arms": list(TIMING_LABELS),
        "checkpoint_iteration": 180,
        "profiled_iteration": 181,
        "nr_iter_schedule": 200,
        "random_seed": 29,
        "image_batch_size": 500,
        "effective_significance_image_batch_size": 187,
        "selected_particle_ids_int64_sha256": EXPECTED_SELECTED_IDS_INT64_SHA256,
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA256,
        "particle_stack_sha256": EXPECTED_PARTICLE_STACK_SHA256,
        "relion_bind_sha256": EXPECTED_RELION_BIND_SHA256,
        "interpreter_sha256": EXPECTED_INTERPRETER_SHA256,
        "cusparse_sha256": EXPECTED_CUSPARSE_SHA256,
        "cuda_toolkit": "/usr/local/cuda-12.6",
        "cuda_arch": "sm_90",
        "material_warm_wall_ratio": MATERIAL_WALL_RATIO,
        "support_audit_timing_eligible": False,
        "clean_timing_support_audit_unset": True,
        "science_promotion_allowed": False,
        "default_enablement_allowed": False,
    }
    mismatch = {key: {"expected": value, "observed": run.get(key)} for key, value in expected.items() if run.get(key) != value}
    _require(not mismatch, f"run provenance differs: {mismatch}")
    for key in ("git_head", "git_tree"):
        _require(isinstance(run.get(key), str) and bool(_GIT_RE.fullmatch(run[key])), f"run {key} is invalid")
    for key in ("source_manifest_sha256", "cuda_sha256"):
        _require(isinstance(run.get(key), str) and bool(_SHA256_RE.fullmatch(run[key])), f"run {key} is invalid")
    _require(isinstance(run.get("job_id"), str) and run["job_id"], "job id is invalid")
    _require("H100" in str(run.get("gpu_name", "")), "run did not use an H100")
    head = _git_rev_parse(repo, f"{run['git_head']}^{{commit}}", "run commit")
    tree = _git_rev_parse(repo, f"{run['git_head']}^{{tree}}", "run tree")
    _require(head == run["git_head"] and tree == run["git_tree"], "run commit/tree differs")
    provenance = root / "provenance"
    recorded = {
        "git_head": _read_single_line(provenance / "repo_head.txt", "repo head"),
        "git_tree": _read_single_line(provenance / "repo_tree.txt", "repo tree"),
        "job_id": _read_single_line(provenance / "slurm_job_id.txt", "Slurm job"),
        "node": _read_single_line(provenance / "node.txt", "node"),
        "gpu_uuid": _read_single_line(provenance / "selected_gpu_uuid.txt", "GPU UUID"),
        "gpu_name": _read_single_line(provenance / "gpu_name.txt", "GPU name"),
    }
    _require(all(recorded[key] == run[key] for key in recorded), "scalar provenance differs")
    _require((provenance / "repo_status.txt").read_text() == "", "run repository was dirty")
    _require(_read_single_line(provenance / "repo_diff.sha256", "repo diff").split()[0] == _EMPTY_SHA256, "run repository diff was nonempty")
    for name in ("allocated_gpu_uuids.csv", "visible_gpu_uuids.csv"):
        _require(_read_single_line(provenance / name, name) == run["gpu_uuid"], f"{name} does not prove sole GPU visibility")
    source = _validate_manifest(
        provenance / "source_manifest.sha256",
        expected_digest=run["source_manifest_sha256"],
        relative_base=repo,
        label="source manifest",
    )
    source_names = {str(Path(row["path"]).resolve().relative_to(repo)) for row in source["entries"]}
    _require(SOURCE_REQUIRED.issubset(source_names), f"source manifest lacks {sorted(SOURCE_REQUIRED - source_names)}")
    _require((provenance / "source_manifest.final.sha256").read_bytes() == (provenance / "source_manifest.sha256").read_bytes(), "source changed during run")
    inputs = _validate_manifest(
        provenance / "input_manifest.sha256",
        expected_digest=run["input_manifest_sha256"],
        relative_base=None,
        label="input manifest",
    )
    input_paths = {Path(row["path"]).resolve() for row in inputs["entries"]}
    _require(EXPECTED_PARTICLE_STACK.resolve() in input_paths, "input manifest lacks particle stack")
    commands = {
        label: _validate_command(root, label, hybrid=hybrid, audit=label.startswith("audit_"))
        for label, hybrid in ALL_SPECS
    }
    run_dirs = {path.name for path in (root / "runs").iterdir() if path.is_dir()}
    _require(run_dirs == {name for name, _hybrid in ALL_SPECS}, f"run topology differs: {sorted(run_dirs)}")
    runtime = {
        "cuda": _validate_sha_record(provenance / "cuda.sha256", run["cuda_sha256"], "CUDA binary"),
        "relion_bind": _validate_sha_record(provenance / "relion_bind.sha256", run["relion_bind_sha256"], "RELION binding"),
        "interpreter": _validate_sha_record(provenance / "interpreter.sha256", run["interpreter_sha256"], "interpreter"),
        "cusparse": _validate_sha_record(provenance / "cusparse.sha256", run["cusparse_sha256"], "cuSPARSE"),
    }
    return run, {
        "run_json_sha256": _sha256(run_path),
        "source_manifest": source,
        "input_manifest": inputs,
        "runtime": runtime,
        "commands": commands,
        "execution_order": _validate_execution_order(provenance / "execution_order.tsv"),
    }


def _load_arm(root: Path, label: str, *, hybrid: bool, audit: bool, repeat: int | None) -> dict[str, Any]:
    run_root = root / "runs" / label
    for name in ("runner.stdout", "runner.stderr", "process.time"):
        _require((run_root / name).is_file(), f"{label} lacks {name}")
    summary_path = run_root / "profile/profile_summary.json"
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
    mismatch = {key: {"expected": value, "observed": summary.get(key)} for key, value in expected.items() if summary.get(key) != value}
    _require(not mismatch, f"{label} profile summary differs: {mismatch}")
    cold = _load_phase(root, label, "cold", summary["cold"])
    warm = _load_phase(root, label, "warm", summary["warm"])
    for phase_name, phase in (("cold", cold), ("warm", warm)):
        metadata = phase["metadata"]
        selected = np.ascontiguousarray(
            np.asarray(metadata.get("selected_particle_ids"), dtype="<i8"),
        )
        _require(
            selected.shape == (1_000,)
            and hashlib.sha256(selected.tobytes(order="C")).hexdigest()
            == EXPECTED_SELECTED_IDS_INT64_SHA256,
            f"{label} {phase_name} selected-particle identity differs",
        )
        counts = metadata.get("significant_counts")
        _require(isinstance(counts, list) and len(counts) == 1_000, f"{label} {phase_name} significant_counts differs")
        profile = metadata["halfset_0_profile_summary"]
        has_support = "coarse_significance_support_audit" in profile
        has_hybrid = "coarse_gaussian_gemm_hybrid" in profile
        _require(has_support is audit, f"{label} {phase_name} support-audit presence differs")
        _require(has_hybrid is hybrid, f"{label} {phase_name} hybrid telemetry presence differs")
    time_peak_kb = _parse_process_time(run_root / "process.time", label)
    warm_timing = warm["timing"]
    peak_kb = max(time_peak_kb, warm_timing["profile_high_water_rss_kb"], warm_timing["profile_max_rss_kb"])
    return {
        "label": label,
        "hybrid": hybrid,
        "audit": audit,
        "repeat": repeat,
        "profile_summary_sha256": _sha256(summary_path),
        "cold": cold,
        "warm": warm,
        "performance": {
            "warm_wall_s": warm_timing["wall_s"],
            "warm_expectation_s": warm_timing["expectation_s"],
            "warm_pass1_s": warm_timing["pass1_s"],
            "warm_pass2_s": warm_timing["pass2_s"],
            "peak_rss_gib": float(peak_kb) / (1024.0**2),
        },
    }


def _validated_support(metadata: dict[str, Any], label: str) -> dict[str, Any]:
    profile = metadata.get("halfset_0_profile_summary")
    _require(isinstance(profile, dict), f"{label} profile is missing")
    audit = profile.get("coarse_significance_support_audit")
    _require(isinstance(audit, dict), f"{label} support audit is missing")
    expected = {
        "schema": SUPPORT_SCHEMA,
        "classification": "diagnostic_only",
        "canonical_encoding": EXPECTED_SUPPORT_ENCODING,
        "n_classes": 1,
        "n_images": 1_000,
        "samples_per_class": 36_864 * 29,
    }
    mismatch = {key: {"expected": value, "observed": audit.get(key)} for key, value in expected.items() if audit.get(key) != value}
    _require(not mismatch, f"{label} support audit geometry differs: {mismatch}")
    for key in ("aggregate_support_sha256", "per_class_image_selected_counts_sha256"):
        _require(isinstance(audit.get(key), str) and bool(_SHA256_RE.fullmatch(audit[key])), f"{label} {key} is invalid")
    counts = audit.get("per_class_image_selected_counts")
    digests = audit.get("per_class_image_support_sha256")
    _require(isinstance(counts, list) and len(counts) == 1 and isinstance(counts[0], list) and len(counts[0]) == 1_000, f"{label} support counts topology differs")
    _require(isinstance(digests, list) and len(digests) == 1 and isinstance(digests[0], list) and len(digests[0]) == 1_000, f"{label} support hashes topology differs")
    _require(all(isinstance(value, int) and 0 <= value <= expected["samples_per_class"] for value in counts[0]), f"{label} support counts are invalid")
    _require(all(isinstance(value, str) and bool(_SHA256_RE.fullmatch(value)) for value in digests[0]), f"{label} per-row support hash is invalid")
    _require(sum(counts[0]) == audit.get("selected_count_sum"), f"{label} selected count sum differs")
    _require(min(counts[0]) == audit.get("selected_count_min") and max(counts[0]) == audit.get("selected_count_max"), f"{label} selected count range differs")
    _require(metadata.get("significant_counts") == counts[0], f"{label} persisted significant_counts differs from support audit")
    return audit


def _validated_hybrid(metadata: dict[str, Any], label: str) -> dict[str, Any]:
    stats = metadata["halfset_0_profile_summary"].get("coarse_gaussian_gemm_hybrid")
    _require(isinstance(stats, dict), f"{label} hybrid telemetry is missing")
    expected = {
        "enabled": True,
        "default_enabled": False,
        "published_score_source": "exact_relion_source16_or_full_rectangular",
        "expanded_gemm_scores_published": False,
        "whole_batch_fail_closed_fallback": True,
        "selected_block_capacity": 64,
        "certificate_chunk_rows": 4_608,
        "certificate_chunk_count_per_batch": 8,
    }
    mismatch = {key: {"expected": value, "observed": stats.get(key)} for key, value in expected.items() if stats.get(key) != value}
    _require(not mismatch, f"{label} hybrid contract differs: {mismatch}")
    integer_keys = (
        "batch_count", "selected_rescore_batch_count", "fallback_batch_count",
        "selected_rescore_image_count", "fallback_image_count",
        "selected_source16_block_count", "selected_exact_candidate_count",
        "full_candidate_count_for_selected_images", "max_selected_blocks_per_image",
    )
    _require(all(isinstance(stats.get(key), int) and stats[key] >= 0 for key in integer_keys), f"{label} hybrid counters are invalid")
    _require(stats["batch_count"] == 6, f"{label} hybrid batch count differs")
    _require(stats["selected_rescore_batch_count"] + stats["fallback_batch_count"] == 6, f"{label} hybrid batch accounting differs")
    _require(stats["selected_rescore_image_count"] + stats["fallback_image_count"] == 1_000, f"{label} hybrid image accounting differs")
    reasons = stats.get("fallback_reasons")
    _require(isinstance(reasons, dict) and all(isinstance(key, str) and isinstance(value, int) and value > 0 for key, value in reasons.items()), f"{label} fallback reasons differ")
    _require(sum(reasons.values()) == stats["fallback_batch_count"], f"{label} fallback reasons do not cover fallback batches")
    _require(isinstance(stats.get("topology_full_to_compact_sha256"), str) and bool(_SHA256_RE.fullmatch(stats["topology_full_to_compact_sha256"])), f"{label} topology digest is invalid")
    return stats


def _support_gate(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    captures: dict[str, dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    exact = True
    for label, _hybrid in AUDIT_SPECS:
        for phase_name in ("cold", "warm"):
            metadata = arms[label][phase_name]["metadata"]
            audit = _validated_support(metadata, f"{label} {phase_name}")
            if label == "audit_hybrid":
                _validated_hybrid(metadata, f"{label} {phase_name}")
            captures[f"{label}_{phase_name}"] = {
                "aggregate_support_sha256": audit["aggregate_support_sha256"],
                "selected_count_sum": audit["selected_count_sum"],
                "selected_count_min": audit["selected_count_min"],
                "selected_count_max": audit["selected_count_max"],
            }
            if reference is None:
                reference = audit
            else:
                exact &= all(
                    audit[key] == reference[key]
                    for key in (
                        "aggregate_support_sha256",
                        "per_class_image_selected_counts_sha256",
                        "per_class_image_selected_counts",
                        "per_class_image_support_sha256",
                    )
                )
    assert reference is not None
    return {
        "captures": captures,
        "aggregate_support_sha256": reference["aggregate_support_sha256"],
        "all_direct_hybrid_cold_warm_support_exact": exact,
        "pass": exact,
    }


def _discrete_gate(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = arms["direct_1"]["warm"]
    comparisons: dict[str, Any] = {}
    exact = True
    for label in TIMING_LABELS:
        for phase_name in ("cold", "warm"):
            candidate = arms[label][phase_name]
            metadata = {
                key: key in reference["metadata"]
                and key in candidate["metadata"]
                and _values_equal(reference["metadata"][key], candidate["metadata"][key])
                for key in DISCRETE_META_KEYS
            }
            state_exact = reference["model_state"]["identity"] == candidate["model_state"]["identity"]
            state_keys_exact = reference["model_state"]["continuous_keys"] == candidate["model_state"]["continuous_keys"]
            row_exact = all(metadata.values()) and _star_equal(reference["star"], candidate["star"]) and state_exact and state_keys_exact
            comparisons[f"direct_1_warm__{label}_{phase_name}"] = {
                "metadata_exact": metadata,
                "particle_star_exact": _star_equal(reference["star"], candidate["star"]),
                "model_identity_exact": state_exact,
                "model_continuous_keys_exact": state_keys_exact,
                "all_exact": row_exact,
            }
            exact &= row_exact
    return {"metadata_keys": list(DISCRETE_META_KEYS), "comparisons": comparisons, "pass": exact}


def _numeric_panel(arms: dict[str, dict[str, Any]], field: str) -> dict[str, Any]:
    def value(label: str) -> np.ndarray:
        warm = arms[label]["warm"]
        if field == "map":
            return np.asarray(warm["map"], dtype=np.float64)
        return np.asarray(warm["model_state"]["continuous_values"], dtype=np.float64)

    direct_repeat = _map_delta(value("direct_1"), value("direct_2"))
    hybrid_repeat = _map_delta(value("hybrid_1"), value("hybrid_2"))
    envelope = _delta_envelope(direct_repeat)
    hybrid_check = _against_envelope(hybrid_repeat, envelope)
    crossed: dict[str, Any] = {}
    signed_means: list[float] = []
    bounded = True
    for repeat in (1, 2):
        delta = _map_delta(value(f"direct_{repeat}"), value(f"hybrid_{repeat}"))
        check = _against_envelope(delta, envelope)
        crossed[f"direct_{repeat}__hybrid_{repeat}"] = {**delta, **check}
        signed_means.append(float(delta["signed_mean"]))
        bounded &= bool(check["within_control_repeat_envelope"])
    nondirectional = min(signed_means) <= 0.0 <= max(signed_means)
    passed = bool(hybrid_check["within_control_repeat_envelope"]) and bounded and nondirectional
    return {
        "direct_repeat": direct_repeat,
        "hybrid_repeat": {**hybrid_repeat, **hybrid_check},
        "direct_repeat_envelope": envelope,
        "crossed": crossed,
        "crossed_signed_means": signed_means,
        "crossed_signed_drift_opposes_or_is_zero": nondirectional,
        "pass": passed,
    }


def _performance(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "direct": [arms["direct_1"]["performance"], arms["direct_2"]["performance"]],
        "hybrid": [arms["hybrid_1"]["performance"], arms["hybrid_2"]["performance"]],
    }
    raw = {mode: {f"R{index + 1}": {key: float(row[key]) for key in TIMING_METRICS} for index, row in enumerate(rows)} for mode, rows in groups.items()}
    medians = {mode: {key: float(median(row[key] for row in rows)) for key in TIMING_METRICS} for mode, rows in groups.items()}
    ratios = {key: medians["hybrid"][key] / medians["direct"][key] for key in TIMING_METRICS}
    wall_ratio = ratios["warm_wall_s"]
    return {
        "raw": raw,
        "medians": medians,
        "hybrid_over_direct_ratio": ratios,
        "material_warm_wall_ratio_threshold": MATERIAL_WALL_RATIO,
        "median_warm_wall_ratio": wall_ratio,
        "median_warm_wall_speedup": 1.0 / wall_ratio,
        "pass": wall_ratio <= MATERIAL_WALL_RATIO,
    }


def _hybrid_summary(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    selected_images = fallback_images = 0
    for label in ("audit_hybrid", "hybrid_1", "hybrid_2"):
        for phase_name in ("cold", "warm"):
            stats = _validated_hybrid(arms[label][phase_name]["metadata"], f"{label} {phase_name}")
            rows[f"{label}_{phase_name}"] = stats
            selected_images += stats["selected_rescore_image_count"]
            fallback_images += stats["fallback_image_count"]
    return {
        "profiles": rows,
        "selected_rescore_image_count_across_profiles": selected_images,
        "fallback_image_count_across_profiles": fallback_images,
        "any_selected_rescore": selected_images > 0,
    }


def _markdown(report: dict[str, Any]) -> str:
    status = "PASS" if report["acceptance"]["pass"] else "FAIL"
    support = report["support_identity"]
    science = report["science"]
    perf = report["performance"]
    hybrid = report["hybrid_telemetry"]
    mark = lambda value: "PASS" if value else "FAIL"
    rows = [
        f"# OVERALL: {status} — GF46 integrated-hybrid transition",
        "",
        f"> **Bounded iteration 180→181 diagnostic: {status}. Default enablement and frozen parity scores are unchanged.**",
        "",
        "| Gate | Result | Status |",
        "|---|---:|---:|",
        f"| Ordered support identity (direct/hybrid, cold/warm) | `{support['aggregate_support_sha256']}` | {mark(support['pass'])} |",
        f"| Clean discrete/STAR identity | exact={science['discrete']['pass']} | {mark(science['discrete']['pass'])} |",
        f"| Map repeat envelope | bounded={science['map_repeat_envelope']['pass']} | {mark(science['map_repeat_envelope']['pass'])} |",
        f"| Model-state repeat envelope | bounded={science['model_state_repeat_envelope']['pass']} | {mark(science['model_state_repeat_envelope']['pass'])} |",
        f"| Median warm wall | hybrid/direct `{perf['median_warm_wall_ratio']:.4f}` (≤ `{MATERIAL_WALL_RATIO:.2f}`) | {mark(perf['pass'])} |",
        f"| Hybrid selected/fallback images (6 profiles) | {hybrid['selected_rescore_image_count_across_profiles']} / {hybrid['fallback_image_count_across_profiles']} | INFO |",
        "| Long trajectory/default gate | not evaluated | OPEN |",
        "",
        "## Clean ABBA timing",
        "",
        "| Mode | R1 wall (s) | R2 wall (s) | Median wall (s) | Median expectation (s) | Median pass 1 (s) | Peak RSS (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("direct", "hybrid"):
        raw = perf["raw"][mode]
        med = perf["medians"][mode]
        rows.append(
            f"| {mode} | {raw['R1']['warm_wall_s']:.6f} | {raw['R2']['warm_wall_s']:.6f} | {med['warm_wall_s']:.6f} | {med['warm_expectation_s']:.6f} | {med['warm_pass1_s']:.6f} | {med['peak_rss_gib']:.3f} |"
        )
    provenance = report["provenance"]
    rows.extend((
        "",
        f"Median speedup: `{perf['median_warm_wall_speedup']:.3f}x`.",
        "",
        "## Provenance",
        "",
        f"- Slurm `{provenance['job_id']}` on `{provenance['node']}` / `{provenance['gpu_uuid']}`",
        f"- Commit/tree `{provenance['git_head']}` / `{provenance['git_tree']}`",
        f"- Source/input manifests `{provenance['source_manifest_sha256']}` / `{provenance['input_manifest_sha256']}`",
        "",
        "A PASS qualifies only this real GF46 one-iteration transition. Long-trajectory no-growth, other datasets, RELION wall-time parity, and default enablement remain separate gates.",
        "",
    ))
    return "\n".join(rows)


def analyze(root: Path, *, repo: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    repo = (repo or Path(__file__).resolve().parents[1]).resolve()
    run, provenance_details = _validate_provenance(root, repo)
    arms: dict[str, dict[str, Any]] = {}
    for label, hybrid in AUDIT_SPECS:
        arms[label] = _load_arm(root, label, hybrid=hybrid, audit=True, repeat=None)
    for label, hybrid, repeat in TIMING_SPECS:
        arms[label] = _load_arm(root, label, hybrid=hybrid, audit=False, repeat=repeat)
    support = _support_gate(arms)
    discrete = _discrete_gate(arms)
    maps = _numeric_panel(arms, "map")
    models = _numeric_panel(arms, "model_state")
    science = {"discrete": discrete, "map_repeat_envelope": maps, "model_state_repeat_envelope": models, "pass": discrete["pass"] and maps["pass"] and models["pass"]}
    performance = _performance(arms)
    hybrid = _hybrid_summary(arms)
    passed = support["pass"] and science["pass"] and performance["pass"]
    report = {
        "schema": SCHEMA,
        "classification": "diagnostic_one_transition_only",
        "provenance": {
            "root": str(root), "job_id": run["job_id"], "git_head": run["git_head"], "git_tree": run["git_tree"],
            "node": run["node"], "gpu_uuid": run["gpu_uuid"], "gpu_name": run["gpu_name"],
            "source_manifest_sha256": run["source_manifest_sha256"], "input_manifest_sha256": run["input_manifest_sha256"],
            "cuda_sha256": run["cuda_sha256"], "analyzer_source_sha256": _sha256(Path(__file__).resolve()), **provenance_details,
        },
        "support_identity": support,
        "science": science,
        "performance": performance,
        "hybrid_telemetry": hybrid,
        "arms": {
            label: {
                "hybrid": arm["hybrid"], "audit": arm["audit"], "repeat": arm["repeat"],
                "profile_summary_sha256": arm["profile_summary_sha256"], "performance": arm["performance"],
                "warm_map_sha256": arm["warm"]["map_sha256"], "warm_model_sha256": arm["warm"]["model_state"]["sha256"],
                "warm_particle_star_sha256": arm["warm"]["star_sha256"],
            }
            for label, arm in arms.items()
        },
        "acceptance": {
            "sealed_topology_and_provenance": True,
            "exact_ordered_support_identity": support["pass"],
            "clean_discrete_and_star_identity": discrete["pass"],
            "map_repeat_envelope": maps["pass"],
            "model_state_repeat_envelope": models["pass"],
            "material_warm_wall_speedup": performance["pass"],
            "long_trajectory_no_growth_evaluated": False,
            "default_enablement_allowed": False,
            "pass": passed,
        },
        "dashboard": {
            "overall_status": "PASS" if passed else "FAIL",
            "transition": "GF46_iteration_180_to_181",
            "candidate": "integrated_certified_source16_hybrid",
            "candidate_default_state": "off",
        },
    }
    report["markdown"] = _markdown(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--print-source-manifest-sha256", action="store_true")
    args = parser.parse_args(argv)
    repo = (args.repo or Path(__file__).resolve().parents[1]).resolve()
    if args.print_source_manifest_sha256:
        manifest, _entries = _source_manifest(repo)
        print(hashlib.sha256(manifest).hexdigest())
        return 0
    if args.root is None or args.output_json is None or args.output_markdown is None:
        parser.error("--root, --output-json, and --output-markdown are required")
    try:
        report = analyze(args.root, repo=repo)
    except HybridTransitionSetupError as exc:
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
