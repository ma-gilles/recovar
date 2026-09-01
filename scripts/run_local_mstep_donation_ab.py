#!/usr/bin/env python3
"""Run one frozen late-VDAM continuation with or without local M-step donation.

This is a same-source diagnostic A/B.  The donated arm uses the production
``run_local_bucket_big_jit`` wrapper.  The control wraps that exact
``__wrapped__`` function with the same static arguments and no donated inputs.
No numerical implementation, operand, or launch-order option differs between
the arms.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from scripts.run_vdam_late_iteration_profile import (
    _effects_barrier,
    _process_resource_delta,
    _process_resource_snapshot,
    _profile_metadata,
    _recovar_argv,
    _sha256,
)

SCHEMA = "recovar.local_mstep_donation_ab.v4"
INPUT_MANIFEST_SCHEMA = "recovar.local_mstep_donation_gf46_inputs.v2"
RESOLVED_INPUT_CONTRACT_SCHEMA = "recovar.local_mstep_donation_gf46_resolved_inputs.v1"
LAUNCH_MANIFEST_SCHEMA = "recovar.local_mstep_donation_launch_manifest.v1"
SEALED_ARM_ENVIRONMENT_SCHEMA = "recovar.local_mstep_donation_arm_environment.v1"
DONATED_POSITIONAL_NAMES = ("Ft_y", "Ft_ctf")
DONATED_ARGNUMS = (7, 8)
GF46_CHECKPOINT_ITERATION = 180
GF46_PROFILED_ITERATION = 181
GF46_NR_ITER_SCHEDULE = 200
GF46_RANDOM_SEED = 29
GF46_IMAGE_BATCH_SIZE = 500
GF46_EXACT_LOCAL_BUCKET_RADIX = 4
GF46_EXACT_LOCAL_PHYSICAL_ORDER_CHUNK_SIZE = 0

SEALED_ARM_ENVIRONMENT_NAMES = frozenset(
    {
        "CUDA_LAUNCH_BLOCKING",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_LOG_COMPILES",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PATH",
        "PIXI_HOME",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONUNBUFFERED",
        "PYTHONUTF8",
        "RATTLER_CACHE_DIR",
        "RECOVAR_CUDA_LIB",
        "RECOVAR_ENABLE_CUSTOM_CUDA",
        "RECOVAR_EXPECTED_REPO_ROOT",
        "RECOVAR_INITIAL_MODEL_PROFILE",
        "RECOVAR_RELION_BIND_BUILD_DIR",
        "TMPDIR",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    }
)
_SEALED_ARM_STATIC_ENVIRONMENT = {
    "CUDA_LAUNCH_BLOCKING": "0",
    "JAX_LOG_COMPILES": "1",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONUTF8": "1",
    "RECOVAR_ENABLE_CUSTOM_CUDA": "1",
    "RECOVAR_INITIAL_MODEL_PROFILE": "1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}

SEALED_NORMALIZED_OPTIONS = {
    "checkpoint_iteration": GF46_CHECKPOINT_ITERATION,
    "profiled_iteration": GF46_PROFILED_ITERATION,
    "nr_iter_schedule": GF46_NR_ITER_SCHEDULE,
    "random_seed": GF46_RANDOM_SEED,
    "image_batch_size": GF46_IMAGE_BATCH_SIZE,
    "exact_local_bucket_radix": GF46_EXACT_LOCAL_BUCKET_RADIX,
    "exact_local_physical_order_chunk_size": GF46_EXACT_LOCAL_PHYSICAL_ORDER_CHUNK_SIZE,
}

SEALED_NORMALIZED_RECOVAR_ARGV = [
    "--i",
    "<INPUT_STAR>",
    "--o",
    "<OUTPUT_PREFIX>",
    "--nr_iter",
    "200",
    "--grad_write_iter",
    "1",
    "--K",
    "1",
    "--tau2_fudge",
    "4",
    "--sym",
    "C1",
    "--do_run_C1",
    "1",
    "--particle_diameter",
    "200.0",
    "--random_seed",
    "29",
    "--healpix_order",
    "1",
    "--oversampling",
    "1",
    "--offset_range",
    "6",
    "--offset_step",
    "2",
    "--padding_factor",
    "1",
    "--image_batch_size",
    "500",
    "--datadir",
    "<DATA_DIR>",
    "--gpu",
    "0",
    "--require_custom_cuda",
    "--diagnostic_continue_optimiser",
    "<CHECKPOINT_OPTIMISER>",
    "--diagnostic_stop_after_iteration",
    "181",
]

# A source change that alters the numeric boundary must update this diagnostic
# deliberately.  Deriving this tuple from the candidate wrapper would make a
# changed static/dynamic contract invisible to the A/B.
SEALED_STATIC_ARGNAMES = (
    "mask_mode",
    "score_with_masked_images",
    "apply_integer_pre_shift",
    "apply_fourier_pre_shift",
    "half_spectrum_scoring",
    "use_float64_scoring",
    "use_float64_normalization",
    "use_window",
    "reconstruct_significant_only",
    "use_relion_f32_fine_posterior",
    "adaptive_fraction",
    "max_significants",
    "image_shape",
    "proj_volume_shape",
    "recon_volume_shape",
    "disc_type",
    "projection_half_volume",
    "projection_max_r",
    "mstep_max_r",
    "use_compact_relion_projector_projection",
    "use_relion_projection_cache",
    "relion_projector_output_size",
    "projection_relion_texture_interp",
    "projection_force_jax",
    "projection_mask_current_image_disk",
    "relion_exact_bpref_operands",
    "relion_exact_fine_diff2",
    "relion_wavg_sequential_cuda",
    "relion_cuda_preprocess_radius",
    "relion_cuda_preprocess_cosine_width",
    "mstep_subtract_ctf_projection",
    "mstep_relion_x_half",
    "relion_sequential_mstep_reduction",
    "disable_adjoint_y",
    "disable_adjoint_ctf",
    "accumulate_noise",
    "accumulate_scale_correction",
    "return_noise_split",
    "return_mstep_tensors",
    "return_source_vdam_operands",
    "return_deferred_mstep_inputs",
    "return_deferred_noise_inputs",
    "n_shells",
    "norm_current_size",
    "include_unweighted_norm_high_shell",
    "has_normalization_log_z",
    "has_normalization_log_evidence",
    "has_normalization_max_posterior",
    "has_reconstruction_probability_threshold",
    "score_only",
    "use_relion_projector",
    "relion_projector_r_max",
    "projection_padding_factor",
    "return_debug_arrays",
    "return_debug_scores",
    "return_debug_operands",
)

_MEMORY_ANALYSIS_FIELDS = (
    "generated_code_size_in_bytes",
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
    "host_generated_code_size_in_bytes",
    "host_argument_size_in_bytes",
    "host_output_size_in_bytes",
    "host_alias_size_in_bytes",
    "host_temp_size_in_bytes",
)

_LAUNCH_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "output_root",
        "repo_root",
        "checkpoint_optimiser",
        "input_star",
        "data_dir",
        "particle_stacks",
        "expected_repo_head",
        "expected_source_manifest_sha256",
        "expected_input_manifest_sha256",
        "expected_node_name",
        "target_gpu_uuid",
        "relion_bind_binary",
        "expected_relion_bind_sha256",
        "expected_focused_test_count",
    }
)
_LAUNCH_PATH_KEYS = (
    "output_root",
    "repo_root",
    "checkpoint_optimiser",
    "input_star",
    "data_dir",
    "relion_bind_binary",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "donated"), required=True)
    parser.add_argument("--checkpoint-optimiser", type=Path, required=True)
    parser.add_argument("--input-star", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-particle-stack",
        action="append",
        type=Path,
        required=True,
        help="Canonical particle-stack path; repeat for every unique stack consumed by the STAR.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-iteration", type=int, default=GF46_CHECKPOINT_ITERATION)
    parser.add_argument("--nr-iter", type=int, default=GF46_NR_ITER_SCHEDULE)
    parser.add_argument("--random-seed", type=int, default=GF46_RANDOM_SEED)
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=GF46_IMAGE_BATCH_SIZE,
    )
    parser.add_argument(
        "--exact-local-bucket-radix",
        type=int,
        choices=(2, 4),
        default=GF46_EXACT_LOCAL_BUCKET_RADIX,
    )
    parser.add_argument(
        "--exact-local-physical-order-chunk-size",
        type=int,
        default=GF46_EXACT_LOCAL_PHYSICAL_ORDER_CHUNK_SIZE,
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--expected-launch-manifest-sha256", required=True)
    parser.add_argument("--expected-jax-cache-dir", type=Path, required=True)
    parser.add_argument("--expected-cuda-lib", type=Path, required=True)
    parser.add_argument("--expected-relion-bind-dir", type=Path, required=True)
    parser.add_argument("--expected-cusparse-library", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    return parser.parse_args(argv)


def _checkpoint_family_paths(checkpoint_optimiser: Path) -> list[tuple[str, Path]]:
    suffix = "_optimiser.star"
    checkpoint_text = str(checkpoint_optimiser)
    if not checkpoint_text.endswith(suffix):
        raise ValueError(f"checkpoint optimiser must end in {suffix}: {checkpoint_optimiser}")
    prefix = Path(checkpoint_text[: -len(suffix)])
    members = (
        ("optimiser.star", checkpoint_optimiser),
        ("model.star", Path(f"{prefix}_model.star")),
        ("data.star", Path(f"{prefix}_data.star")),
        ("sampling.star", Path(f"{prefix}_sampling.star")),
        ("class001.mrc", Path(f"{prefix}_class001.mrc")),
        ("1moment001.mrc", Path(f"{prefix}_1moment001.mrc")),
        ("1moment002.mrc", Path(f"{prefix}_1moment002.mrc")),
        ("2moment001.mrc", Path(f"{prefix}_2moment001.mrc")),
    )
    return [(f"checkpoint/{name}", path) for name, path in members]


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _resolve_gf46_consumed_inputs(
    checkpoint_optimiser: Path,
    input_star: Path,
    data_dir: Path,
    expected_particle_stacks: Sequence[Path],
) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    """Resolve and pin the exact transitive inputs consumed by continuation.

    The continuation loader follows paths embedded in the optimiser/model STAR
    files, while the image loader separately resolves every ``rlnImageName``
    through ``--datadir`` and its extension fallbacks.  Reusing those exact
    resolvers here prevents a conventional same-prefix file from being hashed
    while execution silently consumes another target.
    """

    import starfile

    from recovar.data_io.image_loader import StarLoader
    from recovar.em.initial_model.driver import (
        _relion_star_list_value,
        _resolve_relion_checkpoint_path,
        _second_pseudo_half_moment_path,
    )

    checkpoint_optimiser = checkpoint_optimiser.resolve(strict=True)
    input_star = input_star.resolve(strict=True)
    data_dir = data_dir.resolve(strict=True)
    canonical = {
        name: path.resolve(strict=True)
        for name, path in _checkpoint_family_paths(checkpoint_optimiser)
    }
    optimiser_text = checkpoint_optimiser.read_text()
    actual_model = _resolve_relion_checkpoint_path(
        _relion_star_list_value(optimiser_text, "rlnModelStarFile"),
        owner=checkpoint_optimiser,
    )
    actual_data = _resolve_relion_checkpoint_path(
        _relion_star_list_value(optimiser_text, "rlnExperimentalDataStarFile"),
        owner=checkpoint_optimiser,
    )
    actual_sampling = _resolve_relion_checkpoint_path(
        _relion_star_list_value(optimiser_text, "rlnOrientSamplingStarFile"),
        owner=checkpoint_optimiser,
    )
    if actual_data != input_star:
        raise RuntimeError(
            "GF46 optimiser data STAR differs from the pinned --input-star: "
            f"{actual_data} != {input_star}"
        )

    model = starfile.read(actual_model, always_dict=True)
    general = model.get("model_general")
    classes = model.get("model_classes")
    if (
        not isinstance(general, dict)
        or int(general.get("rlnNrClasses", -1)) != 1
        or classes is None
        or len(classes) != 1
    ):
        raise RuntimeError("donation A/B requires the canonical K=1 GF46 model_classes table")
    row = classes.iloc[0]
    actual_reference = _resolve_relion_checkpoint_path(
        str(row["rlnReferenceImage"]), owner=actual_model
    )
    actual_moment1 = _resolve_relion_checkpoint_path(
        str(row["rlnGradMoment1"]), owner=actual_model
    )
    actual_moment1_pseudo_half = _second_pseudo_half_moment_path(
        actual_moment1,
        nr_classes=1,
    )
    actual_moment2 = _resolve_relion_checkpoint_path(
        str(row["rlnGradMoment2"]), owner=actual_model
    )
    actual_checkpoint_paths = {
        "checkpoint/optimiser.star": checkpoint_optimiser,
        "checkpoint/model.star": actual_model,
        "checkpoint/data.star": actual_data,
        "checkpoint/sampling.star": actual_sampling,
        "checkpoint/class001.mrc": actual_reference,
        "checkpoint/1moment001.mrc": actual_moment1,
        "checkpoint/1moment002.mrc": actual_moment1_pseudo_half,
        "checkpoint/2moment001.mrc": actual_moment2,
    }
    mismatches = {
        role: {"expected": str(canonical[role]), "resolved": str(path)}
        for role, path in actual_checkpoint_paths.items()
        if path != canonical[role]
    }
    if mismatches:
        raise RuntimeError(
            "GF46 continuation references non-canonical checkpoint targets: "
            f"{mismatches}"
        )

    image_loader = StarLoader(
        str(input_star),
        datadir=str(data_dir),
        lazy=True,
        max_threads=1,
        skip_staging=True,
    )
    try:
        resolved_particle_stacks = tuple(
            sorted(
                {
                    Path(path).resolve(strict=True)
                    for path in image_loader._file_map["mrc_file"].unique()
                },
                key=lambda path: path.as_posix(),
            )
        )
    finally:
        image_loader.close()
    expected_paths_raw = [Path(path).resolve(strict=True) for path in expected_particle_stacks]
    expected_paths = tuple(sorted(set(expected_paths_raw), key=lambda path: path.as_posix()))
    if len(expected_paths) != len(expected_paths_raw):
        raise RuntimeError("pinned particle-stack list contains duplicate canonical paths")
    if resolved_particle_stacks != expected_paths:
        raise RuntimeError(
            "GF46 rlnImageName particle stacks differ from the pinned canonical targets: "
            f"resolved={[str(path) for path in resolved_particle_stacks]}, "
            f"expected={[str(path) for path in expected_paths]}"
        )

    named_paths = [
        *actual_checkpoint_paths.items(),
        (f"input/{input_star.name}", input_star),
        *(
            (f"particles/{index:03d}/{path.name}", path)
            for index, path in enumerate(resolved_particle_stacks)
        ),
    ]
    resolved_contract = {
        "schema": RESOLVED_INPUT_CONTRACT_SCHEMA,
        "data_dir": str(data_dir),
        "consumed": [
            {"role": role, "path": str(path.resolve(strict=True))}
            for role, path in named_paths
        ],
        "particle_stacks": [str(path) for path in resolved_particle_stacks],
    }
    return named_paths, resolved_contract


def _input_manifest_payload_from_named_paths(
    named_paths: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    entries = []
    seen_names: set[str] = set()
    for relative_name, raw_path in named_paths:
        if relative_name in seen_names:
            raise RuntimeError(f"duplicate input-manifest name: {relative_name}")
        seen_names.add(relative_name)
        path = raw_path.resolve(strict=True)
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"sealed GF46 input is not a non-empty file: {path}")
        entries.append(
            {
                "relative_name": relative_name,
                "source_name": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return {
        "schema": INPUT_MANIFEST_SCHEMA,
        "entries": entries,
    }


def gf46_input_manifest_payload(
    checkpoint_optimiser: Path,
    input_star: Path,
    data_dir: Path,
    expected_particle_stacks: Sequence[Path],
) -> dict[str, Any]:
    """Hash every exact consumed GF46 input without embedding host roots."""

    named_paths, _resolved_contract = _resolve_gf46_consumed_inputs(
        checkpoint_optimiser,
        input_star,
        data_dir,
        expected_particle_stacks,
    )
    return _input_manifest_payload_from_named_paths(named_paths)


def write_gf46_input_manifest(
    output: Path,
    checkpoint_optimiser: Path,
    input_star: Path,
    data_dir: Path,
    expected_particle_stacks: Sequence[Path],
    *,
    resolved_output: Path | None = None,
) -> str:
    named_paths, resolved_contract = _resolve_gf46_consumed_inputs(
        checkpoint_optimiser,
        input_star,
        data_dir,
        expected_particle_stacks,
    )
    payload = _input_manifest_payload_from_named_paths(named_paths)
    encoded = _canonical_json_bytes(payload)
    output.write_bytes(encoded)
    if resolved_output is not None:
        # Include hashes here as path-specific provenance without making the
        # reviewed content manifest depend on one filesystem root.
        resolved_by_role = {
            role: {
                "path": str(path.resolve(strict=True)),
                "sha256": _sha256(path.resolve(strict=True)),
            }
            for role, path in named_paths
        }
        resolved_output.write_bytes(
            _canonical_json_bytes(
                {
                    **resolved_contract,
                    "resolved_by_role": resolved_by_role,
                }
            )
        )
    return hashlib.sha256(encoded).hexdigest()


def _verify_input_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str,
    checkpoint_optimiser: Path,
    input_star: Path,
    data_dir: Path,
    expected_particle_stacks: Sequence[Path],
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    observed_sha256 = _sha256(manifest_path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"GF46 input-manifest SHA mismatch: expected {expected_sha256}, got {observed_sha256}"
        )
    observed = json.loads(manifest_path.read_text())
    named_paths, resolved_contract = _resolve_gf46_consumed_inputs(
        checkpoint_optimiser,
        input_star,
        data_dir,
        expected_particle_stacks,
    )
    expected = _input_manifest_payload_from_named_paths(named_paths)
    if observed != expected:
        raise RuntimeError("GF46 inputs no longer match the reviewed input manifest")
    return {
        "path": str(manifest_path),
        "sha256": observed_sha256,
        "schema": INPUT_MANIFEST_SCHEMA,
        "entries": expected["entries"],
        "resolved_inputs": resolved_contract,
        "hashes": {
            str(entry["relative_name"]): str(entry["sha256"])
            for entry in expected["entries"]
        },
    }


def _validate_launch_manifest_payload(payload: Any) -> dict[str, Any]:
    """Validate the reviewed, positional launch contract without defaults."""

    if not isinstance(payload, dict) or set(payload) != _LAUNCH_MANIFEST_KEYS:
        observed = set(payload) if isinstance(payload, dict) else set()
        raise RuntimeError(
            "donation launch manifest keys drifted: "
            f"missing={sorted(_LAUNCH_MANIFEST_KEYS - observed)}, "
            f"unexpected={sorted(observed - _LAUNCH_MANIFEST_KEYS)}"
        )
    if payload.get("schema") != LAUNCH_MANIFEST_SCHEMA:
        raise RuntimeError(f"donation launch manifest schema drifted: {payload.get('schema')!r}")
    for key, value in payload.items():
        if key in {"particle_stacks", "expected_focused_test_count"}:
            continue
        if not isinstance(value, str) or not value or any(character in value for character in "\0\r\n"):
            raise RuntimeError(f"donation launch manifest {key} must be one non-empty line")
    for key in (
        "expected_source_manifest_sha256",
        "expected_input_manifest_sha256",
        "expected_relion_bind_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", payload[key]) is None:
            raise RuntimeError(f"donation launch manifest {key} is not a lowercase SHA-256")
    if re.fullmatch(r"[0-9a-f]{40,64}", payload["expected_repo_head"]) is None:
        raise RuntimeError("donation launch manifest expected_repo_head is not a full commit ID")
    if re.fullmatch(r"GPU-[0-9A-Fa-f-]+", payload["target_gpu_uuid"]) is None:
        raise RuntimeError("donation launch manifest target_gpu_uuid is not a physical GPU UUID")
    focused_count = payload["expected_focused_test_count"]
    if isinstance(focused_count, bool) or not isinstance(focused_count, int) or focused_count <= 0:
        raise RuntimeError("donation launch manifest expected_focused_test_count must be positive")

    particles = payload["particle_stacks"]
    if not isinstance(particles, list) or not particles:
        raise RuntimeError("donation launch manifest particle_stacks must be a non-empty list")
    if any(
        not isinstance(value, str)
        or not value
        or any(character in value for character in "\0\r\n")
        for value in particles
    ):
        raise RuntimeError("donation launch manifest particle_stacks contains an invalid path")

    normalized = dict(payload)
    for key in _LAUNCH_PATH_KEYS:
        path = Path(payload[key])
        if not path.is_absolute():
            raise RuntimeError(f"donation launch manifest {key} is not absolute: {path}")
        strict = key != "output_root"
        resolved = path.resolve(strict=strict)
        if resolved != path:
            raise RuntimeError(
                f"donation launch manifest {key} is not canonical: {path} != {resolved}"
            )
        normalized[key] = str(resolved)
    normalized_particles = [str(Path(value).resolve(strict=True)) for value in particles]
    if normalized_particles != particles or normalized_particles != sorted(set(normalized_particles)):
        raise RuntimeError(
            "donation launch manifest particle_stacks must be sorted, unique canonical paths"
        )
    normalized["particle_stacks"] = normalized_particles
    return normalized


def _verify_launch_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise RuntimeError("expected launch-manifest digest is not a lowercase SHA-256")
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "donation launch-manifest SHA mismatch: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )
    payload = _validate_launch_manifest_payload(json.loads(path.read_text()))
    return {
        "path": str(path),
        "sha256": observed_sha256,
        "schema": LAUNCH_MANIFEST_SCHEMA,
        "payload": payload,
    }


def _validate_sealed_gf46_options(args: argparse.Namespace) -> dict[str, int]:
    observed = {
        "checkpoint_iteration": int(args.checkpoint_iteration),
        "profiled_iteration": int(args.checkpoint_iteration) + 1,
        "nr_iter_schedule": int(args.nr_iter),
        "random_seed": int(args.random_seed),
        "image_batch_size": int(args.image_batch_size),
        "exact_local_bucket_radix": int(args.exact_local_bucket_radix),
        "exact_local_physical_order_chunk_size": int(args.exact_local_physical_order_chunk_size),
    }
    if observed != SEALED_NORMALIZED_OPTIONS:
        raise RuntimeError(
            "donation A/B only accepts the reviewed GF46 iteration-181 schedule: "
            f"expected {SEALED_NORMALIZED_OPTIONS}, got {observed}"
        )
    return observed


def _normalized_recovar_argv(
    command: list[str],
    *,
    args: argparse.Namespace,
    output_prefix: Path,
) -> list[str]:
    replacements = {
        str(args.input_star): "<INPUT_STAR>",
        str(output_prefix): "<OUTPUT_PREFIX>",
        str(args.data_dir): "<DATA_DIR>",
        str(args.checkpoint_optimiser): "<CHECKPOINT_OPTIMISER>",
    }
    normalized = [replacements.get(token, token) for token in command]
    if normalized != SEALED_NORMALIZED_RECOVAR_ARGV:
        raise RuntimeError(
            "RECOVAR continuation argv drifted from the reviewed GF46 contract: "
            f"expected {SEALED_NORMALIZED_RECOVAR_ARGV}, got {normalized}"
        )
    return normalized


def _directory_manifest(root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative_name = path.relative_to(root).as_posix()
        if path.is_dir():
            entries.append({"relative_name": relative_name, "type": "directory"})
        elif path.is_file():
            entries.append(
                {
                    "relative_name": relative_name,
                    "type": "file",
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
        else:
            raise RuntimeError(f"unsupported entry in JAX cache: {path}")
    payload = {"entries": entries}
    payload["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def _assert_fresh_jax_cache(output_root: Path, expected_cache_dir: Path) -> dict[str, Any]:
    cache_dir = expected_cache_dir.resolve(strict=True)
    env_value = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if env_value is None or Path(env_value).resolve() != cache_dir:
        raise RuntimeError(
            "JAX_COMPILATION_CACHE_DIR does not match --expected-jax-cache-dir: "
            f"env={env_value!r}, expected={cache_dir}"
        )
    if cache_dir.parent != output_root.parent:
        raise RuntimeError(
            f"per-run JAX cache must be a sibling of result/: {cache_dir}, {output_root}"
        )
    initial_entries = list(cache_dir.iterdir())
    if initial_entries:
        raise RuntimeError(f"per-run JAX cache is not empty: {initial_entries[:5]}")
    return {
        "path": str(cache_dir),
        "initial_empty": True,
        "initial_entry_count": 0,
    }


def _assert_sealed_arm_environment(
    *,
    repo_root: Path,
    cache_dir: Path,
    cuda_lib: Path,
    relion_bind_dir: Path,
    cusparse_library: Path,
    gpu_uuid: str,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Reject every variable outside the explicitly reconstructed arm env."""

    observed = dict(os.environ if environment is None else environment)
    observed_names = set(observed)
    if observed_names != SEALED_ARM_ENVIRONMENT_NAMES:
        missing = sorted(SEALED_ARM_ENVIRONMENT_NAMES - observed_names)
        unexpected = sorted(observed_names - SEALED_ARM_ENVIRONMENT_NAMES)
        raise RuntimeError(
            "arm process environment is not the sealed allowlist: "
            f"missing={missing}, unexpected={unexpected}"
        )
    expected_values = {
        **_SEALED_ARM_STATIC_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "JAX_COMPILATION_CACHE_DIR": str(cache_dir.resolve(strict=True)),
        "LD_PRELOAD": str(cusparse_library.resolve(strict=True)),
        "RECOVAR_CUDA_LIB": str(cuda_lib.resolve(strict=True)),
        "RECOVAR_EXPECTED_REPO_ROOT": str(repo_root.resolve(strict=True)),
        "RECOVAR_RELION_BIND_BUILD_DIR": str(relion_bind_dir.resolve(strict=True)),
    }
    drift = {
        name: {"expected": expected, "observed": observed.get(name)}
        for name, expected in expected_values.items()
        if observed.get(name) != expected
    }
    if drift:
        raise RuntimeError(f"arm process environment values drifted: {drift}")
    for name in ("HOME", "PIXI_HOME", "RATTLER_CACHE_DIR", "TMPDIR"):
        path = Path(observed[name])
        if not path.is_absolute() or not path.is_dir():
            raise RuntimeError(
                f"sealed arm environment {name} is not an existing absolute directory: {path}"
            )
    for name in ("PATH", "LD_LIBRARY_PATH"):
        if not observed[name]:
            raise RuntimeError(f"sealed arm environment {name} is empty")
    return {
        "schema": SEALED_ARM_ENVIRONMENT_SCHEMA,
        "exact_allowlist": True,
        "allowed_names": sorted(SEALED_ARM_ENVIRONMENT_NAMES),
        "observed": {name: observed[name] for name in sorted(observed)},
        "unexpected_names": [],
        "iref_replay_forbidden": True,
    }


def _runtime_provenance(repo_root: Path) -> dict[str, Any]:
    import jax

    import recovar
    from recovar.utils.parity_provenance import (
        REQUIRED_PARITY_ANCESTORS,
        assert_parity_ancestors,
    )

    pixi_env = (repo_root / ".pixi" / "envs" / "default").resolve(strict=True)
    python_executable = Path(sys.executable).resolve(strict=True)
    python_prefix = Path(sys.prefix).resolve(strict=True)
    jax_path = Path(jax.__file__).resolve(strict=True)
    recovar_path = Path(recovar.__file__).resolve(strict=True)
    if python_prefix != pixi_env:
        raise RuntimeError(f"Python prefix is outside the exact worktree pixi env: {python_prefix} != {pixi_env}")
    if not python_executable.is_relative_to(pixi_env):
        raise RuntimeError(f"Python executable is outside the exact worktree pixi env: {python_executable}")
    if not jax_path.is_relative_to(pixi_env):
        raise RuntimeError(f"JAX import is outside the exact worktree pixi env: {jax_path}")
    if not recovar_path.is_relative_to(repo_root):
        raise RuntimeError(f"RECOVAR import is outside the exact worktree: {recovar_path}")
    assert_parity_ancestors()
    return {
        "repo_root": str(repo_root),
        "pixi_env": str(pixi_env),
        "python_executable": str(python_executable),
        "python_prefix": str(python_prefix),
        "jax_path": str(jax_path),
        "recovar_path": str(recovar_path),
        "parity_ancestors_verified": True,
        "required_parity_ancestors": [sha for sha, _description in REQUIRED_PARITY_ANCESTORS],
        # The worktree interpreter/import paths are pinned, but every shared
        # object loaded by that interpreter is not recursively content-hashed.
        # Consequently this experiment may report only a same-job preliminary
        # timing signal; it cannot make an archival/replayable speed claim.
        "installed_runtime_content_hash_complete": False,
        "runtime_claim_scope": "same_job_preliminary_only_runtime_contents_not_archivally_hashed",
        "reproducible_runtime_speed_claim_allowed": False,
    }


def _jsonable_static(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable_static(item) for item in value]
    return repr(value)


def _dynamic_signature(args: tuple[Any, ...]) -> tuple[str, list[dict[str, Any]]]:
    import jax

    leaves, tree = jax.tree_util.tree_flatten(args)
    specs = []
    for leaf in leaves:
        shape = getattr(leaf, "shape", None)
        dtype = getattr(leaf, "dtype", None)
        specs.append(
            {
                "shape": None if shape is None else [int(value) for value in shape],
                "dtype": None if dtype is None else str(dtype),
                "python_type": f"{type(leaf).__module__}.{type(leaf).__qualname__}",
            }
        )
    # Equinox static PyTree metadata includes process-local function/object
    # addresses in its repr.  They are relevant to JAX's in-process cache but
    # not to the crossed-process source contract, so redact addresses before
    # deriving the persistent program key.
    stable_tree = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", str(tree))
    return stable_tree, specs


def _program_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tree, leaves = _dynamic_signature(args)
    static = {name: _jsonable_static(kwargs[name]) for name in SEALED_STATIC_ARGNAMES}
    payload = {"dynamic_tree": tree, "dynamic_leaves": leaves, "static": static}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _memory_analysis_dict(stats: Any) -> dict[str, int]:
    if stats is None:
        raise RuntimeError("XLA compiled memory analysis is unavailable")
    return {name: int(getattr(stats, name)) for name in _MEMORY_ANALYSIS_FIELDS}


def _array_nbytes(value: Any) -> int:
    size = int(getattr(value, "size", 0))
    dtype = getattr(value, "dtype", None)
    return 0 if dtype is None else size * int(dtype.itemsize)


def _source_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


class LocalMstepDonationMonitor:
    """Compile and execute each exact local signature under one sealed arm."""

    def __init__(self, arm: str):
        import jax

        from recovar.em.dense_single_volume import local_big_jit

        production = local_big_jit.run_local_bucket_big_jit
        info = production._jit_info
        parameter_names = tuple(inspect.signature(production).parameters)
        if parameter_names[7:9] != DONATED_POSITIONAL_NAMES:
            raise RuntimeError("local accumulator signature positions changed")
        if tuple(info.static_argnames) != SEALED_STATIC_ARGNAMES:
            raise RuntimeError("local big-JIT static argument contract changed")
        if tuple(info.donate_argnums) != DONATED_ARGNUMS:
            raise RuntimeError("production local big-JIT donation contract changed")
        if production.__wrapped__ is not production._fun:
            raise RuntimeError("production local big-JIT wrapped source identity changed")
        if arm == "donated":
            selected = production
        elif arm == "control":
            selected = jax.jit(
                production.__wrapped__,
                static_argnames=SEALED_STATIC_ARGNAMES,
                donate_argnums=(),
            )
            if selected.__wrapped__ is not production.__wrapped__:
                raise RuntimeError("control wrapper does not share production numeric source")
            if tuple(selected._jit_info.static_argnames) != SEALED_STATIC_ARGNAMES:
                raise RuntimeError("control wrapper static argument contract changed")
            if tuple(selected._jit_info.donate_argnums):
                raise RuntimeError("control wrapper unexpectedly donates inputs")
        else:
            raise ValueError(f"unknown donation arm: {arm}")

        self.arm = arm
        self.production = production
        self.selected = selected
        self.executables: dict[str, Any] = {}
        self.records: dict[str, dict[str, Any]] = {}
        self.call_keys: list[str] = []
        self.phase = "setup"

    def set_phase(self, phase: str) -> None:
        self.phase = str(phase)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if set(kwargs) != set(SEALED_STATIC_ARGNAMES):
            missing = sorted(set(SEALED_STATIC_ARGNAMES) - set(kwargs))
            extra = sorted(set(kwargs) - set(SEALED_STATIC_ARGNAMES))
            raise RuntimeError(f"local big-JIT static kwargs changed: missing={missing}, extra={extra}")
        key, signature = _program_key(args, kwargs)
        executable = self.executables.get(key)
        if executable is None:
            lowered = self.selected.lower(*args, **kwargs)
            executable = lowered.compile()
            self.executables[key] = executable
            self.records[key] = {
                "program_key": key,
                "first_phase": self.phase,
                "signature": signature,
                "accumulator_bytes": {
                    "Ft_y": _array_nbytes(args[7]),
                    "Ft_ctf": _array_nbytes(args[8]),
                    "total": _array_nbytes(args[7]) + _array_nbytes(args[8]),
                },
                "adjoints_enabled": bool((not kwargs["disable_adjoint_y"]) or (not kwargs["disable_adjoint_ctf"])),
                "mstep_relion_x_half": bool(kwargs["mstep_relion_x_half"]),
                "memory_analysis": _memory_analysis_dict(executable.memory_analysis()),
            }
        self.call_keys.append(key)
        # All keyword arguments are static and were captured by ``lower``.
        # The compiled executable therefore accepts only the positional dynamic
        # pytree.  This also prevents the control from drifting to another JIT.
        return executable(*args)

    def contract(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "same_wrapped_numeric_source": self.selected.__wrapped__ is self.production.__wrapped__,
            "numeric_source_sha256": _source_sha256(self.production.__wrapped__),
            "parameter_names": list(inspect.signature(self.production).parameters),
            "static_argnames": list(SEALED_STATIC_ARGNAMES),
            "production_donate_argnums": list(DONATED_ARGNUMS),
            "selected_donate_argnums": list(self.selected._jit_info.donate_argnums),
            "selected_donate_argnames": list(self.selected._jit_info.donate_argnames),
        }

    def report(self) -> dict[str, Any]:
        return {
            "contract": self.contract(),
            "unique_program_count": len(self.records),
            "call_count": len(self.call_keys),
            "call_program_keys": list(self.call_keys),
            "programs": [self.records[key] for key in sorted(self.records)],
        }


@contextmanager
def installed_local_mstep_donation_arm(
    arm: str,
) -> Iterator[LocalMstepDonationMonitor]:
    from recovar.em.dense_single_volume import local_em_engine

    original = local_em_engine._invoke_local_bucket_big_jit
    monitor = LocalMstepDonationMonitor(arm)
    local_em_engine._invoke_local_bucket_big_jit = monitor
    try:
        yield monitor
    finally:
        local_em_engine._invoke_local_bucket_big_jit = original


def _gpu_uuid(expected: str | None) -> str | None:
    import jax

    if jax.default_backend() != "gpu":
        if expected is not None:
            raise RuntimeError(f"expected GPU backend, got {jax.default_backend()}")
        return None
    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise RuntimeError(f"donation A/B requires one visible GPU, got {devices}")
    command = ["/usr/bin/nvidia-smi"]
    if expected is not None:
        command.append(f"--id={expected}")
    command.extend(("--query-gpu=uuid", "--format=csv,noheader"))
    observed = subprocess.check_output(command, text=True).strip()
    if expected is not None and observed != expected:
        raise RuntimeError(f"GPU UUID mismatch: expected {expected}, got {observed}")
    return observed


def _device_memory_stats() -> dict[str, int]:
    import jax

    stats = jax.devices("gpu")[0].memory_stats()
    if stats is None:
        raise RuntimeError("JAX GPU memory stats are unavailable")
    return {str(key): int(value) for key, value in stats.items()}


def _science_outputs(prefix: Path, iteration: int) -> dict[str, dict[str, Any]]:
    paths = {
        "class_map": Path(f"{prefix}_it{iteration:03d}_class001.mrc"),
        "initial_model_map": prefix.parent / "initial_model.mrc",
        "data_star": Path(f"{prefix}_it{iteration:03d}_data.star"),
        "model_star": Path(f"{prefix}_it{iteration:03d}_model.star"),
        "recovar_meta": Path(f"{prefix}_it{iteration:03d}_recovar_meta.json"),
    }
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"continuation output is missing {name}: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    result["output_prefix"] = {"path": str(prefix.resolve())}
    return result


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    sealed_options = _validate_sealed_gf46_options(args)
    observed_head = _git_head(repo_root)
    if observed_head != args.expected_repo_head:
        raise RuntimeError(f"repository head mismatch: expected {args.expected_repo_head}, got {observed_head}")
    args.checkpoint_optimiser = args.checkpoint_optimiser.resolve(strict=True)
    args.input_star = args.input_star.resolve(strict=True)
    args.data_dir = args.data_dir.resolve(strict=True)
    args.expected_particle_stack = [
        path.resolve(strict=True) for path in args.expected_particle_stack
    ]
    args.expected_cuda_lib = args.expected_cuda_lib.resolve(strict=True)
    args.expected_relion_bind_dir = args.expected_relion_bind_dir.resolve(strict=True)
    args.expected_cusparse_library = args.expected_cusparse_library.resolve(strict=True)
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")
    launch_manifest = _verify_launch_manifest(
        args.launch_manifest,
        args.expected_launch_manifest_sha256,
    )
    launch = launch_manifest["payload"]
    expected_launch_values = {
        "repo_root": str(repo_root.resolve(strict=True)),
        "checkpoint_optimiser": str(args.checkpoint_optimiser),
        "input_star": str(args.input_star),
        "data_dir": str(args.data_dir),
        "particle_stacks": [str(path) for path in args.expected_particle_stack],
        "expected_repo_head": args.expected_repo_head,
        "expected_input_manifest_sha256": args.expected_input_manifest_sha256,
        "target_gpu_uuid": args.expected_gpu_uuid,
    }
    launch_drift = {
        name: {"manifest": launch.get(name), "runner": expected}
        for name, expected in expected_launch_values.items()
        if launch.get(name) != expected
    }
    launch_root = Path(launch["output_root"])
    if not args.output_root.is_relative_to(launch_root / "runs"):
        launch_drift["output_root"] = {
            "manifest_runs_root": str(launch_root / "runs"),
            "runner": str(args.output_root),
        }
    if launch_drift:
        raise RuntimeError(f"arm arguments drifted from the positional launch manifest: {launch_drift}")
    sealed_environment = _assert_sealed_arm_environment(
        repo_root=repo_root,
        cache_dir=args.expected_jax_cache_dir,
        cuda_lib=args.expected_cuda_lib,
        relion_bind_dir=args.expected_relion_bind_dir,
        cusparse_library=args.expected_cusparse_library,
        gpu_uuid=args.expected_gpu_uuid,
    )
    cache_contract = _assert_fresh_jax_cache(
        args.output_root,
        args.expected_jax_cache_dir,
    )
    input_manifest = _verify_input_manifest(
        args.input_manifest,
        expected_sha256=args.expected_input_manifest_sha256,
        checkpoint_optimiser=args.checkpoint_optimiser,
        input_star=args.input_star,
        data_dir=args.data_dir,
        expected_particle_stacks=args.expected_particle_stack,
    )
    runtime_provenance = _runtime_provenance(repo_root)
    gpu_uuid = _gpu_uuid(args.expected_gpu_uuid)
    args.output_root.mkdir(parents=True)
    (args.output_root / "SAFE_TO_DELETE").touch()

    from scripts.run_ab_initio import main as run_ab_initio

    target_iteration = int(args.checkpoint_iteration) + 1
    phase_reports: dict[str, dict[str, Any]] = {}
    with installed_local_mstep_donation_arm(args.arm) as monitor:
        for phase in ("cold", "warm"):
            monitor.set_phase(phase)
            prefix = args.output_root / phase / "run"
            prefix.parent.mkdir(parents=True, exist_ok=False)
            command = _recovar_argv(args=args, output_prefix=prefix)
            normalized_argv = _normalized_recovar_argv(
                command,
                args=args,
                output_prefix=prefix,
            )
            if phase == "warm":
                _effects_barrier()
                gc.collect()
                # Installed JAX exposes cumulative allocator stats but no
                # peak-reset API.  Keep before/after values explicitly labeled
                # whole-process diagnostics; the Slurm runner's nvidia-smi
                # sampler is delimited by the two timestamps below but remains
                # diagnostic-only because sparse polling cannot prove a HWM.
                memory_before = _device_memory_stats()
                timed_start_unix_ns = time.time_ns()
                (args.output_root / "TIMED_BEGIN").write_text(f"{timed_start_unix_ns}\n")
            else:
                memory_before = None
                timed_start_unix_ns = None

            resources_before = _process_resource_snapshot()
            started = time.perf_counter()
            status = int(run_ab_initio(command))
            _effects_barrier()
            wall_s = float(time.perf_counter() - started)
            resources_after = _process_resource_snapshot()
            if status != 0:
                raise RuntimeError(f"{phase} continuation exited with status {status}")
            if phase == "warm":
                timed_end_unix_ns = time.time_ns()
                (args.output_root / "TIMED_END").write_text(f"{timed_end_unix_ns}\n")
                memory_after = _device_memory_stats()
            else:
                timed_end_unix_ns = None
                memory_after = None
            phase_reports[phase] = {
                "wall_s": wall_s,
                "argv": command,
                "normalized_argv": normalized_argv,
                "process_resources": {
                    "before": resources_before,
                    "after": resources_after,
                    "delta": _process_resource_delta(resources_before, resources_after),
                },
                "timed_start_unix_ns": timed_start_unix_ns,
                "timed_end_unix_ns": timed_end_unix_ns,
                "jax_memory_before": memory_before,
                "jax_memory_after": memory_after,
                "jax_memory_scope": ("whole_process_cumulative_not_warm_isolated" if phase == "warm" else None),
                "science_outputs": _science_outputs(prefix, target_iteration),
                **_profile_metadata(prefix, target_iteration),
            }

        monitor_report = monitor.report()

    cache_contract["final"] = _directory_manifest(Path(cache_contract["path"]))

    report = {
        "schema": SCHEMA,
        "classification": "diagnostic_performance_only",
        "passed": True,
        "arm": args.arm,
        "git_head": observed_head,
        "gpu_uuid": gpu_uuid,
        "runtime_provenance": runtime_provenance,
        "sealed_environment": sealed_environment,
        "launch_manifest": launch_manifest,
        "jax_persistent_cache": cache_contract,
        "input_manifest": input_manifest,
        "input_hashes": input_manifest["hashes"],
        "normalized_options": sealed_options,
        "normalized_recovar_argv": list(SEALED_NORMALIZED_RECOVAR_ARGV),
        "checkpoint_iteration": int(args.checkpoint_iteration),
        "profiled_iteration": target_iteration,
        "nr_iter_schedule": int(args.nr_iter),
        "checkpoint_optimiser": str(args.checkpoint_optimiser),
        "checkpoint_optimiser_sha256": _sha256(args.checkpoint_optimiser),
        "input_star": str(args.input_star),
        "input_star_sha256": _sha256(args.input_star),
        "data_dir": str(args.data_dir),
        "particle_stacks": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in args.expected_particle_stack
        ],
        "numeric_policy": {
            "mathematically_equivalent": True,
            "arithmetic_changed": False,
            "strict_exactness_is_strong_evidence_not_universal_requirement": True,
            "phase_matched_continuous_repeat_noise_may_be_compatible": True,
            "arbitrary_absolute_or_relative_tolerance_used": False,
            "broader_optimized_arithmetic_policy": {
                "stable_repeat_bounded_noise_allowed": True,
                "directional_bias_allowed": False,
                "iteration_amplified_drift_allowed": False,
                "exact_discrete_choices_and_trajectory_basin_required": True,
                "material_final_quality_loss_allowed": False,
                "material_end_to_end_runtime_gain_required": True,
            },
        },
        "speed_claim_allowed": False,
        "same_job_preliminary_speed_signal_allowed": False,
        "runtime_claim_scope": "same_job_preliminary_only_runtime_contents_not_archivally_hashed",
        "memory_claim_allowed": False,
        "default_promotion_allowed": False,
        "cold": phase_reports["cold"],
        "warm": phase_reports["warm"],
        "compiled_local_programs": monitor_report,
    }
    report_path = args.output_root / "donation_arm_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
