#!/usr/bin/env python3
"""Run one cold and one warm RECOVAR VDAM continuation under one CUDA context.

This is a diagnostic-only performance harness.  Both executions continue the
same native RELION checkpoint for exactly one next iteration.  The optional
CUDA profiler range encloses only the second execution so Nsight Systems can
measure steady-state work without importing, data-loading, or JIT compilation
from the first execution.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import hashlib
import inspect
import json
import math
import os
import resource
import threading
import time
import traceback
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

_PROFILE_BOOLEAN_TRUE = frozenset({"1", "true", "yes", "on"})
_PROFILE_BOOLEAN_FALSE = frozenset({"0", "false", "no", "off"})
_PROFILE_GEMM_MACRO_ENV = "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO"
_PROFILE_FFI_ENV = "RECOVAR_K1_COARSE_GAUSSIAN_FFI"
_PROFILE_SINCOSF_ENV = "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF"
_PROFILE_FUSED_PROJECTOR_ENV = "RECOVAR_K1_COARSE_FUSED_PROJECTOR"
_PROFILE_CANONICAL_REDUCTION_ENV = "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION"
_PROFILE_NATIVE_ATOMIC_ENV = "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION"
_PROFILE_SINGLE_LANE_ENV = "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL"
_PROFILE_MULTISTREAM_ENV = "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS"
_PROFILE_NATIVE_TEXTURE_ENV = "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE"
_PROFILE_CONTRACT_CHOICES = ("diagnostic", "default", "all_optimized_q32")
_STAGE_PROFILE_CHOICES = ("on", "off", "certify-off")
_PROFILE_RUNTIME_ENVIRONMENT = frozenset(
    {
        "RECOVAR_CUDA_LIB",
        "RECOVAR_EXPECTED_REPO_ROOT",
        "RECOVAR_RELION_BIND_BUILD_DIR",
        "RECOVAR_SELECTED_GPU_UUID",
    }
)
_PROFILE_ALL_OPTIMIZED_Q32_EXTRA_ENVIRONMENT = {
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY": "64",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE": "200",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB": "2",
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB": "40",
    "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT": "0",
    "RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT_IDS": "0",
    "RECOVAR_K1_COARSE_FUSED_PROJECTOR": "0",
    "RECOVAR_K1_COARSE_GAUSSIAN_FFI": "1",
    "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE": "0",
    "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF": "1",
    "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": "0",
    "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": "0",
    "RECOVAR_K1_COARSE_PREHALF_WEIGHT": "0",
    "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL": "0",
    "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS": "1",
    "RECOVAR_K1_RELION_F32_COARSE_SUPPORT": "1",
    "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION": "0",
    "RECOVAR_RELION_BATCHED_POSTERIOR_PRIMITIVES": "1",
    "RECOVAR_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM": "32",
}
_NATIVE_REUSE_RELOCATABLE_SUFFIXES = (
    Path("scripts/vdam_relion_one_iteration.gdb"),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-optimiser", type=Path, required=True)
    parser.add_argument("--input-star", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--static-input-manifest",
        type=Path,
        help="sha256sum manifest rechecked before and after each arm.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-iteration", type=int, default=180)
    parser.add_argument("--nr-iter", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=29)
    parser.add_argument("--image-batch-size", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--exact-local-bucket-radix", type=int, choices=(2, 4), default=4)
    parser.add_argument("--exact-local-physical-order-chunk-size", type=int, default=0)
    parser.add_argument(
        "--stable-fourier-window-shapes",
        action="store_true",
        help="Use the production candidate's stable physical Fourier ABI.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Call cudaProfilerStart/Stop around only the warm execution.",
    )
    parser.add_argument(
        "--audit-raw-image-cache",
        action="store_true",
        help="Record every ImageLoader.load_all call without changing cache policy.",
    )
    parser.add_argument(
        "--cold-compile-callsite-log",
        type=Path,
        help=(
            "Diagnostic-only JSONL trace of cold compile_or_get_cached calls; "
            "this adds traceback overhead and invalidates timing truth."
        ),
    )
    parser.add_argument(
        "--execution-contract",
        choices=_PROFILE_CONTRACT_CHOICES,
        default="diagnostic",
        help=(
            "Fail closed on the emitted default or complete optimized-q32 "
            "execution profile; diagnostic preserves the legacy unchecked mode."
        ),
    )
    parser.add_argument(
        "--initial-model-stage-profile",
        choices=_STAGE_PROFILE_CHOICES,
        default="on",
        help=(
            "Enable the synchronization-heavy RECOVAR initial-model stage profiler. "
            "Use off to measure both arms with only the final output barrier, or "
            "certify-off to profile-certify the cold route before measuring the "
            "otherwise identical warm production topology."
        ),
    )
    parser.add_argument(
        "--expected-schedule-json",
        help="Exact schedule JSON required by the certify-off qualification mode.",
    )
    parser.add_argument(
        "--expected-selected-particle-sha256",
        help="Ordered little-endian int64 particle-ID digest required by certify-off.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_reused_native_inputs(
    manifest_path: Path,
    current_paths: list[Path],
) -> dict[str, list[dict[str, str]]]:
    """Validate capture inputs while allowing a content-identical worktree move."""

    expected: dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        digest, raw_path = line.split(maxsplit=1)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"invalid reused native digest for {raw_path}")
        if raw_path in expected and expected[raw_path] != digest:
            raise RuntimeError(f"conflicting reused native digests for {raw_path}")
        expected[raw_path] = digest

    records: list[dict[str, str]] = []
    for path in current_paths:
        raw_path = str(path)
        source_path = raw_path
        role = "absolute_path"
        if raw_path not in expected:
            matching_suffixes = [
                suffix
                for suffix in _NATIVE_REUSE_RELOCATABLE_SUFFIXES
                if path.parts[-len(suffix.parts) :] == suffix.parts
            ]
            if len(matching_suffixes) != 1:
                raise RuntimeError(f"reused native manifest omitted {raw_path}")
            suffix = matching_suffixes[0]
            source_matches = [
                candidate
                for candidate in expected
                if Path(candidate).parts[-len(suffix.parts) :] == suffix.parts
            ]
            if len(source_matches) != 1:
                raise RuntimeError(
                    "reused native manifest has ambiguous relocated role "
                    f"{suffix}: {source_matches}"
                )
            source_path = source_matches[0]
            role = str(suffix)

        digest = _sha256(path)
        if digest != expected[source_path]:
            raise RuntimeError(f"reused native input changed: {raw_path}")
        records.append(
            {
                "path": raw_path,
                "source_manifest_path": source_path,
                "role": role,
                "sha256": digest,
            }
        )
    return {"inputs": records}


def _validate_static_input_manifest(manifest_path: Path) -> dict[str, str]:
    """Recompute every exact input named by one sha256sum manifest."""

    observed: dict[str, str] = {}
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        try:
            expected, raw_path = line.split(maxsplit=1)
        except ValueError as error:
            raise RuntimeError(
                f"invalid static-input manifest line {line_number}"
            ) from error
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise RuntimeError(
                f"invalid static-input digest on line {line_number}"
            )
        path = Path(raw_path.strip()).resolve(strict=True)
        key = str(path)
        if key in observed:
            if observed[key] != expected:
                raise RuntimeError(
                    f"conflicting duplicate static-input manifest path: {key}"
                )
            continue
        digest = _sha256(path)
        if digest != expected:
            raise RuntimeError(f"static input changed: {key}")
        observed[key] = digest
    if not observed:
        raise RuntimeError("static-input manifest is empty")
    return dict(sorted(observed.items()))


def _profile_environment_bool(
    environment: Mapping[str, str],
    name: str,
    *,
    default: bool,
) -> bool:
    token = environment.get(name, "1" if default else "0").strip().lower()
    if token in _PROFILE_BOOLEAN_FALSE:
        return False
    if token in _PROFILE_BOOLEAN_TRUE:
        return True
    raise ValueError(f"Unsupported {name}={token!r}")


def _validate_profile_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Resolve late-profile coarse selectors before any expensive capture.

    The continuation is the guarded K=1 Gaussian path, so absent fused and
    canonical selectors are effectively enabled.  An explicitly requested
    GEMM macro must therefore carry the same explicit opt-outs as the accepted
    optimized q32 configuration instead of relying on ambient defaults.
    """

    environ = os.environ if environment is None else environment
    ffi = _profile_environment_bool(environ, _PROFILE_FFI_ENV, default=True)
    sincosf = _profile_environment_bool(
        environ,
        _PROFILE_SINCOSF_ENV,
        default=ffi,
    )
    fused_projector = _profile_environment_bool(
        environ,
        _PROFILE_FUSED_PROJECTOR_ENV,
        default=ffi and sincosf,
    )
    canonical_reduction = _profile_environment_bool(
        environ,
        _PROFILE_CANONICAL_REDUCTION_ENV,
        default=fused_projector,
    )
    native_atomic = _profile_environment_bool(
        environ,
        _PROFILE_NATIVE_ATOMIC_ENV,
        default=False,
    )
    single_lane = _profile_environment_bool(
        environ,
        _PROFILE_SINGLE_LANE_ENV,
        default=False,
    )
    native_texture = _profile_environment_bool(
        environ,
        _PROFILE_NATIVE_TEXTURE_ENV,
        default=False,
    )
    multistream_token = environ.get(_PROFILE_MULTISTREAM_ENV, "0").strip()
    try:
        multistream_workers = int(multistream_token)
    except ValueError as error:
        raise ValueError(
            f"{_PROFILE_MULTISTREAM_ENV} must be 0 or 8, got {multistream_token!r}",
        ) from error
    if multistream_workers not in {0, 8}:
        raise ValueError(
            f"{_PROFILE_MULTISTREAM_ENV} must be 0 or 8, got {multistream_token!r}",
        )
    gemm_macro = _profile_environment_bool(
        environ,
        _PROFILE_GEMM_MACRO_ENV,
        default=False,
    )
    competing = {
        _PROFILE_FUSED_PROJECTOR_ENV: fused_projector,
        _PROFILE_CANONICAL_REDUCTION_ENV: canonical_reduction,
        _PROFILE_NATIVE_ATOMIC_ENV: native_atomic,
        _PROFILE_SINGLE_LANE_ENV: single_lane,
        _PROFILE_MULTISTREAM_ENV: multistream_workers > 0,
        _PROFILE_NATIVE_TEXTURE_ENV: native_texture,
    }
    conflicts = [name for name, enabled in competing.items() if enabled]
    if gemm_macro and conflicts:
        rendered = ", ".join(
            f"{name}={environ.get(name, '<effective default>')}" for name in conflicts
        )
        raise ValueError(
            f"{_PROFILE_GEMM_MACRO_ENV}=1 conflicts with {rendered}; "
            "set every competing selector explicitly to 0 before profiling",
        )
    return {
        "schema": "recovar.vdam_late_profile_environment_preflight.v1",
        "gemm_macro_requested": gemm_macro,
        "resolved_backend": "gemm_macro" if gemm_macro else "non_gemm",
        "effective_selectors": {
            _PROFILE_FFI_ENV: ffi,
            _PROFILE_SINCOSF_ENV: sincosf,
            _PROFILE_FUSED_PROJECTOR_ENV: fused_projector,
            _PROFILE_CANONICAL_REDUCTION_ENV: canonical_reduction,
            _PROFILE_NATIVE_ATOMIC_ENV: native_atomic,
            _PROFILE_SINGLE_LANE_ENV: single_lane,
            _PROFILE_MULTISTREAM_ENV: multistream_workers,
            _PROFILE_NATIVE_TEXTURE_ENV: native_texture,
        },
        "explicit_values": {
            name: environ[name]
            for name in (
                _PROFILE_GEMM_MACRO_ENV,
                _PROFILE_FFI_ENV,
                _PROFILE_SINCOSF_ENV,
                *competing,
            )
            if name in environ
        },
    }


def _all_optimized_q32_environment() -> dict[str, str]:
    """Return the complete qualified stack, including post-ten-seam removals."""

    from scripts import run_vdam_hybrid_same_state_transition as same_state

    requested = same_state._candidate_environment(
        "exact_compact_preprocess",
        enabled=True,
    )
    requested[same_state.EXACT_COARSE_ASSEMBLY_PROFILE_ENVIRONMENT] = "1"
    requested.update(_PROFILE_ALL_OPTIMIZED_Q32_EXTRA_ENVIRONMENT)
    return requested


def _validate_profile_contract_environment(
    contract_mode: str,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Prove the process environment names the profile it is about to run."""

    if contract_mode not in _PROFILE_CONTRACT_CHOICES:
        raise ValueError(f"unsupported late-profile contract {contract_mode!r}")
    environ = os.environ if environment is None else environment
    candidate = _all_optimized_q32_environment()
    if contract_mode == "diagnostic":
        return {
            "mode": contract_mode,
            "environment_checked": False,
            "effective_environment": {
                name: environ.get(name) for name in sorted(candidate)
            },
        }
    if contract_mode == "default":
        unexpected = {
            name: environ[name] for name in sorted(candidate) if name in environ
        }
        if unexpected:
            raise RuntimeError(
                "default late-profile contract requires candidate selectors to "
                f"be absent, got {unexpected!r}"
            )
        return {
            "mode": contract_mode,
            "environment_checked": True,
            "required_absent": sorted(candidate),
            "effective_environment": {},
        }

    effective = {name: environ.get(name) for name in sorted(candidate)}
    if effective != dict(sorted(candidate.items())):
        raise RuntimeError(
            "all_optimized_q32 late-profile environment differs from the "
            f"qualified stack: expected={candidate!r}, effective={effective!r}"
        )
    return {
        "mode": contract_mode,
        "environment_checked": True,
        "requested_environment": dict(sorted(candidate.items())),
        "effective_environment": effective,
    }


def _validate_sealed_recovar_environment(
    contract_mode: str,
    *,
    stage_profile_enabled: bool,
    expected_runtime_environment: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Require the exact qualified RECOVAR route plus runtime bindings."""

    environ = os.environ if environment is None else environment
    runtime_source = (
        environ
        if expected_runtime_environment is None
        else expected_runtime_environment
    )
    missing_runtime = sorted(
        name
        for name in _PROFILE_RUNTIME_ENVIRONMENT
        if not isinstance(runtime_source.get(name), str) or not runtime_source[name]
    )
    if missing_runtime:
        raise RuntimeError(
            "late-profile environment lacks required runtime bindings: "
            f"{missing_runtime}"
        )

    candidate = _all_optimized_q32_environment()
    if contract_mode == "all_optimized_q32":
        expected_candidate = candidate
    elif contract_mode == "default":
        expected_candidate = {}
    elif contract_mode == "diagnostic":
        expected_candidate = {
            name: environ[name] for name in sorted(candidate) if name in environ
        }
    else:
        raise ValueError(f"unsupported late-profile contract {contract_mode!r}")

    extra_runtime = sorted(set(runtime_source) - _PROFILE_RUNTIME_ENVIRONMENT)
    if extra_runtime:
        raise RuntimeError(
            f"unexpected names in expected runtime environment: {extra_runtime}"
        )
    expected = {
        name: runtime_source[name]
        for name in sorted(_PROFILE_RUNTIME_ENVIRONMENT)
    }
    expected.update(expected_candidate)
    if stage_profile_enabled:
        expected["RECOVAR_INITIAL_MODEL_PROFILE"] = "1"
    effective = {
        name: environ[name]
        for name in sorted(environ)
        if name.startswith("RECOVAR_")
    }
    if effective != dict(sorted(expected.items())):
        raise RuntimeError(
            "late-profile RECOVAR environment is not sealed: "
            f"expected={dict(sorted(expected.items()))!r}, effective={effective!r}"
        )
    contract = _validate_profile_contract_environment(contract_mode, environ)
    return {
        "environment_exact": True,
        "stage_profile_enabled": bool(stage_profile_enabled),
        "expected_recovar_environment": dict(sorted(expected.items())),
        "present_recovar_environment": effective,
        "execution_contract": contract,
    }


def _qualified_profile_runtime_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve and validate the four non-algorithm RECOVAR bindings once."""

    environ = os.environ if environment is None else environment
    runtime = {
        name: environ.get(name, "") for name in sorted(_PROFILE_RUNTIME_ENVIRONMENT)
    }
    missing = [name for name, value in runtime.items() if not value]
    if missing:
        raise RuntimeError(f"late-profile runtime bindings are missing: {missing}")
    expected_repo = Path(__file__).resolve().parents[1]
    if Path(runtime["RECOVAR_EXPECTED_REPO_ROOT"]).resolve() != expected_repo:
        raise RuntimeError(
            "RECOVAR_EXPECTED_REPO_ROOT does not name the executing checkout"
        )
    cuda_library = Path(runtime["RECOVAR_CUDA_LIB"]).resolve(strict=True)
    if not cuda_library.is_file():
        raise RuntimeError("RECOVAR_CUDA_LIB is not a file")
    bind_dir = Path(runtime["RECOVAR_RELION_BIND_BUILD_DIR"]).resolve(strict=True)
    if not bind_dir.is_dir():
        raise RuntimeError("RECOVAR_RELION_BIND_BUILD_DIR is not a directory")
    selected_uuid = runtime["RECOVAR_SELECTED_GPU_UUID"]
    if not selected_uuid.startswith("GPU-") or len(selected_uuid) <= 4:
        raise RuntimeError("RECOVAR_SELECTED_GPU_UUID is not a physical GPU UUID")
    return runtime


def _validate_optimized_row_totals(
    estep_meta: dict[str, object],
    *,
    label: str,
) -> dict[str, object]:
    """Validate every published compact row vector and its scalar total."""

    import numpy as np

    array_to_sum = {
        "chunk_flat_score_rows": "sum_flat_score_rows",
        "chunk_padded_rotations": "sum_padded_rows",
        "chunk_planned_padded_rotations": "sum_planned_padded_rows",
        "chunk_reconstruction_rows": "sum_reconstruction_rows",
        "chunk_nonzero_posterior_rows": "sum_nonzero_posterior_rows",
    }
    profiles: dict[str, object] = {}
    for name, raw_profile in sorted(estep_meta.items()):
        if not isinstance(raw_profile, dict) or "chunk_padded_rotations" not in raw_profile:
            continue
        arrays: dict[str, list[int]] = {}
        totals: dict[str, int] = {}
        for array_name, sum_name in array_to_sum.items():
            if array_name not in raw_profile or sum_name not in raw_profile:
                raise RuntimeError(
                    f"{label} profile {name} omitted {array_name} or {sum_name}"
                )
            values = np.asarray(raw_profile[array_name], dtype=np.int64)
            if values.ndim != 1 or values.size == 0 or np.any(values < 0):
                raise RuntimeError(
                    f"{label} profile {name} has invalid {array_name}"
                )
            observed_sum = int(raw_profile[sum_name])
            expected_sum = int(np.sum(values, dtype=np.int64))
            if observed_sum != expected_sum:
                raise RuntimeError(
                    f"{label} profile {name} reported {sum_name}={observed_sum}, "
                    f"expected {expected_sum} from {array_name}"
                )
            arrays[array_name] = values.tolist()
            totals[sum_name] = observed_sum
        if arrays["chunk_flat_score_rows"] != arrays["chunk_padded_rotations"]:
            raise RuntimeError(
                f"{label} profile {name} flat and padded row ABIs differ"
            )
        if arrays["chunk_planned_padded_rotations"] != arrays["chunk_padded_rotations"]:
            raise RuntimeError(
                f"{label} profile {name} planned and executed row ABIs differ"
            )
        if arrays["chunk_reconstruction_rows"] != arrays["chunk_nonzero_posterior_rows"]:
            raise RuntimeError(
                f"{label} profile {name} reconstruction and posterior rows differ"
            )
        if totals["sum_reconstruction_rows"] <= 0:
            raise RuntimeError(f"{label} profile {name} reconstructed no rows")
        profiles[name] = {"arrays": arrays, "totals": totals}
    if not profiles:
        raise RuntimeError(f"{label} has no local row profile")
    return {"profile_exact": True, "profiles": profiles}


def _validate_late_hybrid_image_batch(
    estep_meta: dict[str, object],
    *,
    label: str,
    n_translations: int,
    oversampling: int,
    requested_image_batch_size: int = 500,
) -> dict[str, object]:
    """Prove late profiling used one real 200-image matrix-matrix batch."""

    from scripts import run_vdam_hybrid_same_state_transition as same_state

    profiles = same_state._coarse_hybrid_profiles(estep_meta)
    if not profiles:
        raise RuntimeError(f"{label} did not publish a compact-hybrid profile")
    observed: dict[str, object] = {}
    exact = {
        "input_image_batch_size": int(requested_image_batch_size),
        "requested_hybrid_image_batch_size": 200,
        "effective_image_batch_size": 200,
        "batch_count": 1,
        "fallback_batch_count": 0,
        "fallback_image_count": 0,
        "actual_image_batch_sizes": [200],
        "physical_image_batch_sizes": [200],
    }
    for name, profile in sorted(profiles.items()):
        current: dict[str, object] = {}
        for field, expected in exact.items():
            value = profile.get(field)
            if value != expected:
                raise RuntimeError(
                    f"{label} profile {name} reported {field}={value!r}, "
                    f"expected {expected!r}"
                )
            current[field] = value
        selected_batches = profile.get("selected_rescore_batch_count")
        static_dense_batches = profile.get("static_dense_batch_count")
        selected_images = profile.get("selected_rescore_image_count")
        static_dense_images = profile.get("static_dense_image_count")
        adaptive_counts = {
            "selected_rescore_batch_count": selected_batches,
            "static_dense_batch_count": static_dense_batches,
            "selected_rescore_image_count": selected_images,
            "static_dense_image_count": static_dense_images,
        }
        invalid_adaptive_counts = [
            field
            for field, value in adaptive_counts.items()
            if not isinstance(value, int) or value < 0
        ]
        if invalid_adaptive_counts:
            raise RuntimeError(
                f"{label} profile {name} has invalid adaptive counts "
                f"{invalid_adaptive_counts}"
            )
        if selected_batches + static_dense_batches != 1:
            raise RuntimeError(
                f"{label} profile {name} did not represent the profiled batch"
            )
        if selected_images + static_dense_images != 200:
            raise RuntimeError(
                f"{label} profile {name} did not represent all profiled images"
            )
        current.update(
            selected_rescore_batch_count=selected_batches,
            static_dense_batch_count=static_dense_batches,
            selected_rescore_image_count=selected_images,
            static_dense_image_count=static_dense_images,
            static_preferred_score_representation=profile.get(
                "static_preferred_score_representation"
            ),
            score_representation_batch_counts=profile.get(
                "score_representation_batch_counts"
            ),
        )
        chunk_rows = profile.get("certificate_chunk_rows")
        chunk_count = profile.get("certificate_chunk_count_per_batch")
        if not isinstance(chunk_rows, int) or chunk_rows <= 0:
            raise RuntimeError(f"{label} profile {name} has invalid certificate_chunk_rows")
        if not isinstance(chunk_count, int) or chunk_count <= 0:
            raise RuntimeError(
                f"{label} profile {name} has invalid certificate_chunk_count_per_batch"
            )
        oversampling_factor = 4 ** int(oversampling)
        if int(n_translations) % oversampling_factor:
            raise RuntimeError(
                f"{label} n_translations={n_translations} is not divisible by "
                f"the oversampling factor {oversampling_factor}"
            )
        coarse_translation_count = int(n_translations) // oversampling_factor
        expected_streamed = 200 * chunk_rows * coarse_translation_count
        if profile.get("streamed_certificate_candidate_count_at_effective_batch") != expected_streamed:
            raise RuntimeError(
                f"{label} profile {name} has stale streamed certificate geometry"
            )
        current.update(
            certificate_chunk_rows=chunk_rows,
            certificate_chunk_count_per_batch=chunk_count,
            coarse_translation_count=coarse_translation_count,
            oversampling_factor=oversampling_factor,
            streamed_certificate_candidate_count_at_effective_batch=expected_streamed,
        )
        observed[name] = current
    return {"profile_exact": True, "profiles": observed}


def _validate_profile_execution_contract(
    estep_meta: dict[str, object],
    *,
    contract_mode: str,
    image_shape: tuple[int, int],
    requested_image_batch_size: int = 500,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Fail closed on effective metadata before a profile can be labeled."""

    environment_contract = _validate_profile_contract_environment(
        contract_mode,
        environment,
    )
    if contract_mode == "diagnostic":
        return {
            "mode": contract_mode,
            "profile_checked": False,
            "environment": environment_contract,
        }

    from scripts import run_vdam_hybrid_same_state_transition as same_state

    if contract_mode == "default":
        composed = same_state._validate_all_optimized_profiles(
            estep_meta,
            enabled=False,
            label="default",
            image_shape=image_shape,
            stable_fourier_window_quantum=8,
        )
        if same_state._coarse_hybrid_profiles(estep_meta):
            raise RuntimeError("default profile unexpectedly used the GEMM hybrid")
        fused_pair = same_state._validate_fused_pair_fine_profiles(
            estep_meta,
            enabled=False,
            label="default",
        )
        return {
            "mode": contract_mode,
            "profile_checked": True,
            "profile_exact": True,
            "environment": environment_contract,
            "all_optimized": composed,
            "fused_pair_fine": fused_pair,
        }

    requested = _all_optimized_q32_environment()
    effective = {
        name: (os.environ if environment is None else environment).get(name)
        for name in requested
    }
    compact = same_state._validate_arm_execution_contract(
        candidate_mode="exact_compact_preprocess",
        candidate_enabled=True,
        requested_environment=requested,
        effective_environment=effective,
        estep_meta=estep_meta,
    )
    hybrid_invariants = _validate_hybrid_fallback_overflow(estep_meta)
    composed = same_state._validate_all_optimized_profiles(
        estep_meta,
        enabled=True,
        label="all_optimized_q32",
        image_shape=image_shape,
        stable_fourier_window_quantum=32,
    )
    exact_coarse = same_state._validate_exact_coarse_single_translate_profiles(
        estep_meta,
        enabled=True,
        label="all_optimized_q32",
    )
    exact_compact = same_state._validate_exact_compact_preprocess_profiles(
        estep_meta,
        enabled=True,
        label="all_optimized_q32",
    )
    fused_pair = same_state._validate_fused_pair_fine_profiles(
        estep_meta,
        enabled=False,
        label="all_optimized_q32",
    )
    n_translations = estep_meta.get("n_translations")
    if not isinstance(n_translations, int) or n_translations <= 0:
        raise RuntimeError("all_optimized_q32 metadata has invalid n_translations")
    oversampling = estep_meta.get("oversampling")
    if not isinstance(oversampling, int) or oversampling < 0:
        raise RuntimeError("all_optimized_q32 metadata has invalid oversampling")
    image_batch = _validate_late_hybrid_image_batch(
        estep_meta,
        label="all_optimized_q32",
        n_translations=n_translations,
        oversampling=oversampling,
        requested_image_batch_size=requested_image_batch_size,
    )
    rows = _validate_optimized_row_totals(
        estep_meta,
        label="all_optimized_q32",
    )
    return {
        "mode": contract_mode,
        "profile_checked": True,
        "profile_exact": True,
        "environment": environment_contract,
        "compact_hybrid": compact,
        "hybrid_fallback_overflow": hybrid_invariants,
        "all_optimized": composed,
        "exact_coarse_single_translate": exact_coarse,
        "exact_compact_preprocess": exact_compact,
        "fused_pair_fine": fused_pair,
        "image_batch": image_batch,
        "row_totals": rows,
    }


def _json_values_exact(left: object, right: object) -> bool:
    """Compare JSON-like values without treating booleans as integers."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        assert isinstance(right, dict)
        return left.keys() == right.keys() and all(
            _json_values_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        assert isinstance(right, list)
        return len(left) == len(right) and all(
            _json_values_exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _validate_hybrid_fallback_overflow(
    estep_meta: dict[str, object],
) -> dict[str, dict[str, object]]:
    from scripts import run_vdam_hybrid_same_state_transition as same_state

    invariants: dict[str, dict[str, object]] = {}
    for name, profile in sorted(same_state._coarse_hybrid_profiles(estep_meta).items()):
        exact_hybrid = {
            "fallback_reasons": {},
            "overflow_latch_scope": (
                "current_significance_call_exact_geometry_and_capacity"
            ),
            "overflow_latch_active_at_return": False,
            "overflow_latch_activation_count": 0,
            "overflow_latch_static_dense_batch_count": 0,
            "overflow_latch_static_dense_image_count": 0,
            "selected_block_capacity": 64,
        }
        observed_hybrid = {field: profile.get(field) for field in exact_hybrid}
        if not _json_values_exact(observed_hybrid, exact_hybrid):
            raise RuntimeError(
                f"optimized hybrid {name} fallback/overflow route differs: "
                f"observed={observed_hybrid!r}, expected={exact_hybrid!r}"
            )
        max_blocks = profile.get("max_selected_blocks_per_image")
        if (
            isinstance(max_blocks, bool)
            or not isinstance(max_blocks, int)
            or not 0 < max_blocks <= 64
        ):
            raise RuntimeError(
                f"optimized hybrid {name} has invalid selected-block maximum"
            )
        invariants[name] = {
            **observed_hybrid,
            "max_selected_blocks_per_image": max_blocks,
        }
    if not invariants:
        raise RuntimeError("optimized route has no hybrid fallback telemetry")
    return invariants


def _validate_profile_free_effective_route(
    estep_meta: dict[str, object],
    *,
    image_shape: tuple[int, int],
    requested_image_batch_size: int,
    exact_local_bucket_radix: int,
    exact_local_physical_order_chunk_size: int,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Certify the optimized route using counters emitted without stage waits."""

    from scripts import run_vdam_hybrid_same_state_transition as same_state

    expected_fields: dict[str, object] = {
        "sparse_pass2": True,
        "pass2_engine": "local",
        "joint_halfset_particle_stream": True,
        "halfset_ids": [0, 1],
        "oversampling": 1,
        "max_significants": 100,
        "requested_image_batch_size": int(requested_image_batch_size),
        "effective_image_batch_size": int(requested_image_batch_size),
        "requested_relion_wavg_sequential_cuda": True,
        "effective_relion_wavg_sequential_cuda": True,
        "requested_exact_local_bucket_radix": int(exact_local_bucket_radix),
        "effective_exact_local_bucket_radix": int(exact_local_bucket_radix),
        "requested_exact_local_physical_order_chunk_size": int(
            exact_local_physical_order_chunk_size
        ),
        "effective_exact_local_physical_order_chunk_size": int(
            exact_local_physical_order_chunk_size
        ),
        "requested_stable_fourier_window_shapes": True,
        "effective_stable_fourier_window_shapes": True,
        "requested_stable_flat_row_capacity": True,
        "effective_stable_flat_row_capacity": True,
        "requested_fused_pair_fine_score": False,
        "effective_fused_pair_fine_score": False,
    }
    observed_fields = {name: estep_meta.get(name) for name in expected_fields}
    mismatches = {
        name: {"expected": expected, "observed": observed_fields[name]}
        for name, expected in expected_fields.items()
        if not _json_values_exact(observed_fields[name], expected)
    }
    if mismatches:
        raise RuntimeError(
            "all_optimized_q32 profile-free route metadata differs from the "
            f"qualified production route: {mismatches!r}"
        )

    requested = _all_optimized_q32_environment()
    environ = os.environ if environment is None else environment
    effective = {name: environ.get(name) for name in requested}
    compact = same_state._validate_arm_execution_contract(
        candidate_mode="compact_posterior",
        candidate_enabled=True,
        requested_environment=requested,
        effective_environment=effective,
        estep_meta=estep_meta,
    )
    hybrid_invariants = _validate_hybrid_fallback_overflow(estep_meta)
    exact_coarse = same_state._validate_exact_coarse_single_translate_profiles(
        estep_meta,
        enabled=True,
        label="all_optimized_q32_profile_free",
    )
    exact_compact = same_state._validate_exact_compact_preprocess_profiles(
        estep_meta,
        enabled=True,
        label="all_optimized_q32_profile_free",
    )
    stable_coarse = same_state._validate_stable_coarse_square_profiles(
        estep_meta,
        enabled=True,
        label="all_optimized_q32_profile_free",
        image_size=int(image_shape[0]),
        stable_fourier_window_quantum=32,
    )
    fused_coarse = same_state._validate_fused_coarse_projector_profiles(
        estep_meta,
        enabled=False,
        label="all_optimized_q32_profile_free",
    )
    n_translations = estep_meta.get("n_translations")
    if isinstance(n_translations, bool) or not isinstance(n_translations, int):
        raise RuntimeError("profile-free route has invalid n_translations")
    if n_translations <= 0 or n_translations % 4:
        raise RuntimeError(
            "profile-free route translation count is incompatible with oversampling=1"
        )
    expected_coarse_translations = n_translations // 4
    for name, audit in fused_coarse["profiles"].items():
        if audit.get("translation_count") != expected_coarse_translations:
            raise RuntimeError(
                f"profile-free selector {name} has stale translation count"
            )
    image_batch = _validate_late_hybrid_image_batch(
        estep_meta,
        label="all_optimized_q32_profile_free",
        n_translations=n_translations,
        oversampling=1,
        requested_image_batch_size=requested_image_batch_size,
    )
    profile_names = sorted(same_state._coarse_hybrid_profiles(estep_meta))
    if profile_names != ["halfset_0_profile_summary"]:
        raise RuntimeError(
            "profile-free joint-halfset route must emit exactly one halfset-0 "
            f"coarse profile, got {profile_names!r}"
        )
    for name, validation in (
        ("exact coarse", exact_coarse),
        ("exact compact", exact_compact),
        ("stable coarse", stable_coarse),
        ("fused coarse", fused_coarse),
        ("image batch", image_batch),
    ):
        if sorted(validation["profiles"]) != profile_names:
            raise RuntimeError(
                f"profile-free {name} telemetry does not match {profile_names!r}"
            )

    return {
        "mode": "all_optimized_q32",
        "effective_route_checked": True,
        "effective_route_exact": True,
        "metadata_fields": observed_fields,
        "halfset_profile_names": profile_names,
        "compact_hybrid": compact,
        "hybrid_fallback_overflow": hybrid_invariants,
        "exact_coarse_single_translate": exact_coarse,
        "exact_compact_preprocess": exact_compact,
        "stable_coarse_significance": stable_coarse,
        "fused_coarse_projector": fused_coarse,
        "image_batch": image_batch,
    }


def _normalized_profile_argv(argv: object) -> list[str]:
    if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
        raise RuntimeError("late-profile argv is not a list of strings")
    output_positions = [index for index, value in enumerate(argv) if value == "--o"]
    if len(output_positions) != 1 or output_positions[0] + 1 >= len(argv):
        raise RuntimeError("late-profile argv must contain exactly one --o value")
    normalized = list(argv)
    normalized[output_positions[0] + 1] = "<ARM_OUTPUT_PREFIX>"
    return normalized


def _validate_profile_argv_output(argv: object, expected_output: object) -> None:
    if not isinstance(argv, list) or not isinstance(expected_output, str):
        raise RuntimeError("late-profile argv/output metadata is invalid")
    output_positions = [index for index, value in enumerate(argv) if value == "--o"]
    if len(output_positions) != 1 or output_positions[0] + 1 >= len(argv):
        raise RuntimeError("late-profile argv must contain exactly one --o value")
    if argv[output_positions[0] + 1] != expected_output:
        raise RuntimeError("late-profile argv --o differs from its metadata prefix")


def _validate_certify_schedule(schedule: object, *, label: str) -> dict[str, object]:
    if not isinstance(schedule, dict):
        raise RuntimeError(f"certify-off {label} schedule is not a mapping")
    expected_fields = {
        "current_size",
        "healpix_order",
        "n_rotations",
        "n_translations",
        "subset_size",
        "random_perturbation",
    }
    if set(schedule) != expected_fields:
        raise RuntimeError(
            f"certify-off {label} schedule fields differ: "
            f"expected={sorted(expected_fields)!r}, observed={sorted(schedule)!r}"
        )
    positive_ints = ("current_size", "n_rotations", "n_translations", "subset_size")
    for field in positive_ints:
        value = schedule.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"certify-off {label} schedule has invalid {field}")
    healpix_order = schedule.get("healpix_order")
    if (
        isinstance(healpix_order, bool)
        or not isinstance(healpix_order, int)
        or healpix_order < 0
    ):
        raise RuntimeError(f"certify-off {label} schedule has invalid healpix_order")
    perturbation = schedule.get("random_perturbation")
    if (
        isinstance(perturbation, bool)
        or not isinstance(perturbation, (int, float))
        or not math.isfinite(float(perturbation))
    ):
        raise RuntimeError(
            f"certify-off {label} schedule has invalid random_perturbation"
        )
    return dict(schedule)


def _certify_profile_free_warm(
    cold: Mapping[str, object],
    warm: Mapping[str, object],
    *,
    expected_schedule: Mapping[str, object],
    expected_selected_particle_sha256: str,
) -> dict[str, object]:
    """Bind a profile-free warm timing arm to its profiled cold route."""

    cold_contract = cold.get("execution_contract")
    warm_contract = warm.get("execution_contract")
    if not isinstance(cold_contract, dict) or not isinstance(warm_contract, dict):
        raise RuntimeError("certify-off arms lack execution contracts")
    if not (
        cold_contract.get("mode") == "all_optimized_q32"
        and cold_contract.get("profile_checked") is True
        and cold_contract.get("profile_exact") is True
    ):
        raise RuntimeError("certify-off cold arm lacks an exact profiled contract")
    if not (
        warm_contract.get("mode") == "all_optimized_q32"
        and warm_contract.get("profile_checked") is False
        and warm_contract.get("profile_exact") is False
    ):
        raise RuntimeError("certify-off warm arm incorrectly claims profiled validation")
    if not isinstance(cold.get("iteration_profile"), dict):
        raise RuntimeError("certify-off cold arm omitted stage timings")
    if warm.get("iteration_profile") is not None:
        raise RuntimeError("certify-off warm arm emitted synchronization-heavy stage timings")
    expected_inputs: dict[str, str] | None = None
    for label, arm in (("cold", cold), ("warm", warm)):
        inputs = arm.get("static_input_sha256")
        if not isinstance(inputs, dict):
            raise RuntimeError(f"certify-off {label} lacks static-input hashes")
        before = inputs.get("before")
        after = inputs.get("after")
        if not isinstance(before, dict) or not _json_values_exact(before, after):
            raise RuntimeError(f"certify-off {label} static inputs changed")
        if expected_inputs is None:
            expected_inputs = before
        elif not _json_values_exact(expected_inputs, before):
            raise RuntimeError("certify-off cold and warm static inputs differ")

    routes: dict[str, dict[str, object]] = {}
    for label, contract in (("cold", cold_contract), ("warm", warm_contract)):
        route = contract.get("production_route")
        if not isinstance(route, dict) or route.get("effective_route_exact") is not True:
            raise RuntimeError(f"certify-off {label} production route is not exact")
        routes[label] = route
    if not _json_values_exact(routes["cold"], routes["warm"]):
        raise RuntimeError("certify-off cold and warm production routes differ")

    normalized_argv = {
        "cold": _normalized_profile_argv(cold.get("argv")),
        "warm": _normalized_profile_argv(warm.get("argv")),
    }
    _validate_profile_argv_output(cold.get("argv"), cold.get("output_prefix"))
    _validate_profile_argv_output(warm.get("argv"), warm.get("output_prefix"))
    cold_output = Path(str(cold["output_prefix"])).resolve()
    warm_output = Path(str(warm["output_prefix"])).resolve()
    if (
        cold_output.name != "run"
        or warm_output.name != "run"
        or cold_output.parent.name != "cold"
        or warm_output.parent.name != "warm"
        or cold_output.parent.parent != warm_output.parent.parent
    ):
        raise RuntimeError("certify-off arm outputs are not one cold/warm root")
    if normalized_argv["cold"] != normalized_argv["warm"]:
        raise RuntimeError("certify-off cold and warm algorithm argv differ")
    cold_schedule = _validate_certify_schedule(cold.get("schedule"), label="cold")
    warm_schedule = _validate_certify_schedule(warm.get("schedule"), label="warm")
    if not _json_values_exact(cold_schedule, warm_schedule):
        raise RuntimeError("certify-off cold and warm schedules differ")
    qualified_schedule = _validate_certify_schedule(
        dict(expected_schedule),
        label="expected",
    )
    if not _json_values_exact(cold_schedule, qualified_schedule):
        raise RuntimeError("certify-off observed schedule differs from its pinned value")
    if not _json_values_exact(
        cold.get("selected_particle_fingerprint"),
        warm.get("selected_particle_fingerprint"),
    ):
        raise RuntimeError("certify-off cold and warm particle streams differ")
    fingerprint = cold.get("selected_particle_fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError("certify-off selected-particle fingerprint is invalid")
    if fingerprint.get("count") != cold_schedule["subset_size"]:
        raise RuntimeError("certify-off selected-particle count differs from subset_size")
    if fingerprint.get("encoding") != "ordered_little_endian_int64":
        raise RuntimeError("certify-off selected-particle encoding is not canonical")
    if fingerprint.get("sha256") != expected_selected_particle_sha256:
        raise RuntimeError(
            "certify-off selected-particle digest differs from its pinned value"
        )

    clean_environments: dict[str, dict[str, str]] = {}
    for label, arm, expected_profile in (
        ("cold", cold, True),
        ("warm", warm, False),
    ):
        sealed = arm.get("sealed_recovar_environment")
        if not isinstance(sealed, dict):
            raise RuntimeError(f"certify-off {label} lacks a sealed environment")
        before = sealed.get("before")
        after = sealed.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise RuntimeError(f"certify-off {label} environment snapshots are invalid")
        if before.get("environment_exact") is not True or after.get("environment_exact") is not True:
            raise RuntimeError(f"certify-off {label} environment is not exact")
        if before.get("stage_profile_enabled") is not expected_profile:
            raise RuntimeError(f"certify-off {label} stage-profile selector is wrong")
        before_map = before.get("present_recovar_environment")
        after_map = after.get("present_recovar_environment")
        if not isinstance(before_map, dict) or not _json_values_exact(before_map, after_map):
            raise RuntimeError(f"certify-off {label} RECOVAR environment changed during execution")
        expected_runtime = {
            name: before_map.get(name) for name in _PROFILE_RUNTIME_ENVIRONMENT
        }
        _validate_sealed_recovar_environment(
            "all_optimized_q32",
            stage_profile_enabled=expected_profile,
            expected_runtime_environment=expected_runtime,
            environment=before_map,
        )
        arm_map = dict(before_map)
        profile_value = arm_map.pop("RECOVAR_INITIAL_MODEL_PROFILE", None)
        if expected_profile and profile_value != "1":
            raise RuntimeError("certify-off cold profile selector is not exactly 1")
        if not expected_profile and profile_value is not None:
            raise RuntimeError("certify-off warm profile selector must be absent")
        clean_environments[label] = arm_map
    if clean_environments["cold"] != clean_environments["warm"]:
        raise RuntimeError("certify-off cold and warm RECOVAR routes differ")

    route_json = json.dumps(routes["warm"], sort_keys=True, separators=(",", ":"))
    argv_json = json.dumps(normalized_argv["warm"], separators=(",", ":"))
    return {
        "classification": "diagnostic_performance_only",
        "certificate_checked": True,
        "certificate_exact": True,
        "certificate_scope": (
            "profiled cold full contract bound to warm top-level and always-emitted "
            "coarse route telemetry; warm local profiler remains disabled"
        ),
        "cold_stage_profile_enabled": True,
        "warm_stage_profile_enabled": False,
        "warm_direct_profile_checked": False,
        "warm_direct_profile_exact": False,
        "normalized_algorithm_argv": normalized_argv["warm"],
        "normalized_algorithm_argv_sha256": hashlib.sha256(argv_json.encode()).hexdigest(),
        "schedule": cold_schedule,
        "selected_particle_fingerprint": fingerprint,
        "production_route_sha256": hashlib.sha256(route_json.encode()).hexdigest(),
        "recovar_environment_without_profile": clean_environments["warm"],
        "static_input_sha256": expected_inputs,
    }


def _load_cuda_profiler() -> tuple[Callable[[], None], Callable[[], None]]:
    try:
        cudart = ctypes.CDLL("libcudart.so")
        start = cudart.cudaProfilerStart
        stop = cudart.cudaProfilerStop
    except (OSError, AttributeError) as exc:
        raise RuntimeError("CUDA profiler API is unavailable") from exc
    start.restype = ctypes.c_int
    start.argtypes = []
    stop.restype = ctypes.c_int
    stop.argtypes = []

    def _checked(function: Callable[[], int], name: str) -> None:
        status = int(function())
        if status != 0:
            raise RuntimeError(f"{name} returned CUDA error code {status}")

    return lambda: _checked(start, "cudaProfilerStart"), lambda: _checked(stop, "cudaProfilerStop")


def _recovar_argv(
    *,
    args: argparse.Namespace,
    output_prefix: Path,
) -> list[str]:
    stop_iteration = int(args.checkpoint_iteration) + 1
    command = [
        "--jax-compilation-cache" if os.environ.get("JAX_COMPILATION_CACHE_DIR") else "--no-jax-compilation-cache",
        "--i",
        str(args.input_star),
        "--o",
        str(output_prefix),
        "--nr_iter",
        str(args.nr_iter),
        "--grad_write_iter",
        "1",
        "--K",
        "1",
        "--tau2_fudge",
        "4",
        "--sym",
        "C1",
        "--run-in-c1",
        "--particle_diameter",
        "200.0",
        "--random_seed",
        str(args.random_seed),
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
        str(args.image_batch_size),
        "--datadir",
        str(args.data_dir),
        "--gpu",
        "0",
        "--require-custom-cuda",
        "--diagnostic-continue-optimiser",
        str(args.checkpoint_optimiser),
        "--diagnostic-stop-after-iteration",
        str(stop_iteration),
    ]
    # The frozen pre-candidate control does not expose these knobs.  Omit its
    # qualified defaults, and pass only non-default candidate values once the
    # integrated source provides the corresponding CLI options.
    if int(args.exact_local_bucket_radix) != 4:
        command.extend(("--exact-local-bucket-radix", str(args.exact_local_bucket_radix)))
    if int(args.exact_local_physical_order_chunk_size) > 0:
        command.extend(
            (
                "--exact-local-physical-order-chunk-size",
                str(args.exact_local_physical_order_chunk_size),
            )
        )
    if args.stable_fourier_window_shapes:
        command.append("--stable-fourier-window-shapes")
    return command


def _process_resource_snapshot() -> dict[str, object]:
    """Capture monotonic process I/O counters and resident-memory state."""

    usage = resource.getrusage(resource.RUSAGE_SELF)
    proc_io: dict[str, int] = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":", 1)
        proc_io[key] = int(value.strip())
    proc_status: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, unit = line.split()
            if unit != "kB":
                raise RuntimeError(f"unexpected /proc/self/status unit: {line}")
            proc_status[key.rstrip(":")] = int(value)
    return {
        "user_cpu_s": float(usage.ru_utime),
        "system_cpu_s": float(usage.ru_stime),
        "max_rss_kb": int(usage.ru_maxrss),
        "minor_faults": int(usage.ru_minflt),
        "major_faults": int(usage.ru_majflt),
        "input_blocks": int(usage.ru_inblock),
        "output_blocks": int(usage.ru_oublock),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "current_rss_kb": int(proc_status["VmRSS"]),
        "high_water_rss_kb": int(proc_status["VmHWM"]),
        "proc_io": proc_io,
    }


def _process_resource_delta(
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    monotonic = (
        "user_cpu_s",
        "system_cpu_s",
        "minor_faults",
        "major_faults",
        "input_blocks",
        "output_blocks",
        "voluntary_context_switches",
        "involuntary_context_switches",
    )
    delta: dict[str, object] = {key: float(after[key]) - float(before[key]) for key in monotonic}
    before_io = before["proc_io"]
    after_io = after["proc_io"]
    assert isinstance(before_io, dict) and isinstance(after_io, dict)
    delta["proc_io"] = {key: int(after_io[key]) - int(before_io[key]) for key in sorted(after_io)}
    return delta


def _profile_metadata(
    output_prefix: Path,
    iteration: int,
    *,
    execution_contract: str = "diagnostic",
    image_shape: tuple[int, int] = (128, 128),
    initial_model_stage_profile_enabled: bool = True,
    requested_image_batch_size: int = 500,
    exact_local_bucket_radix: int = 4,
    exact_local_physical_order_chunk_size: int = 0,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    meta_path = Path(f"{output_prefix}_it{iteration:03d}_recovar_meta.json")
    continuation_path = Path(f"{output_prefix}_diagnostic_continuation.json")
    if not meta_path.is_file() or not continuation_path.is_file():
        raise RuntimeError(
            f"diagnostic continuation did not write its required metadata: {meta_path}, {continuation_path}"
        )
    unexpected = sorted(output_prefix.parent.glob(f"{output_prefix.name}_it*_recovar_meta.json"))
    if unexpected != [meta_path]:
        raise RuntimeError(f"diagnostic continuation must write exactly one iteration metadata file: {unexpected}")
    meta = json.loads(meta_path.read_text())
    continuation = json.loads(continuation_path.read_text())
    if continuation.get("classification") != "diagnostic_performance_only":
        raise RuntimeError("continuation output is not classified diagnostic_performance_only")
    continuation_iteration = continuation.get("iteration")
    if (
        isinstance(continuation_iteration, bool)
        or not isinstance(continuation_iteration, int)
        or continuation_iteration + 1 != iteration
    ):
        raise RuntimeError("continuation metadata does not identify the incoming iteration")
    profile = meta.get("vdam_iteration_profile_summary")
    if initial_model_stage_profile_enabled:
        if not isinstance(profile, dict) or not profile:
            raise RuntimeError("RECOVAR_INITIAL_MODEL_PROFILE did not emit stage timings")
    else:
        forbidden_profile_keys = {
            "vdam_iteration_profile_summary",
            "sparse_pass2_profile_summary",
            "dense_adapter_profile_summary",
            "native_expectation_profile_summary",
        }
        present_profile_keys = sorted(forbidden_profile_keys.intersection(meta))
        if present_profile_keys:
            raise RuntimeError(
                "RECOVAR_INITIAL_MODEL_PROFILE emitted profile-only metadata while "
                f"explicitly disabled: {present_profile_keys}"
            )
        if execution_contract == "all_optimized_q32":
            profile_free_halfsets = {
                key: value
                for key, value in meta.items()
                if key.startswith("halfset_")
                and key.endswith("_profile_summary")
            }
            if sorted(profile_free_halfsets) != ["halfset_0_profile_summary"]:
                raise RuntimeError(
                    "profile-free joint-halfset execution emitted the wrong halfset profiles"
                )
            coarse_only_keys = {
                "coarse_selector_audit",
                "coarse_gaussian_gemm_hybrid",
                "exact_coarse_operand_assembly",
            }
            halfset_profile = profile_free_halfsets["halfset_0_profile_summary"]
            if not isinstance(halfset_profile, dict) or set(halfset_profile) != coarse_only_keys:
                raise RuntimeError(
                    "profile-free halfset metadata is not the exact coarse-only telemetry set"
                )
    schedule_keys = (
        "current_size",
        "healpix_order",
        "n_rotations",
        "n_translations",
        "subset_size",
        "random_perturbation",
    )
    missing_schedule = [key for key in schedule_keys if key not in meta]
    if missing_schedule:
        raise RuntimeError(f"iteration metadata lacks required schedule fields: {missing_schedule}")
    raw_subset_size = meta["subset_size"]
    if isinstance(raw_subset_size, bool) or not isinstance(raw_subset_size, int):
        raise RuntimeError("iteration metadata subset_size is invalid")
    subset_size = int(raw_subset_size)
    if subset_size == 0 or subset_size < -1:
        raise RuntimeError(f"iteration metadata subset_size is invalid: {subset_size}")
    selected_particle_ids = meta.get("selected_particle_ids")
    if not isinstance(selected_particle_ids, list):
        raise RuntimeError("iteration metadata lacks selected_particle_ids")
    if not selected_particle_ids or any(
        isinstance(particle_id, bool)
        or not isinstance(particle_id, int)
        or particle_id < 0
        or particle_id > 2**63 - 1
        for particle_id in selected_particle_ids
    ):
        raise RuntimeError("iteration metadata selected_particle_ids are invalid")
    if len(set(selected_particle_ids)) != len(selected_particle_ids):
        raise RuntimeError("iteration metadata selected_particle_ids are not unique")
    if subset_size > 0 and len(selected_particle_ids) != subset_size:
        raise RuntimeError(
            "iteration metadata subset_size does not match selected_particle_ids: "
            f"{subset_size} != {len(selected_particle_ids)}"
        )
    production_route = None
    if execution_contract == "all_optimized_q32":
        production_route = _validate_profile_free_effective_route(
            meta,
            image_shape=image_shape,
            requested_image_batch_size=requested_image_batch_size,
            exact_local_bucket_radix=exact_local_bucket_radix,
            exact_local_physical_order_chunk_size=(
                exact_local_physical_order_chunk_size
            ),
            environment=environment,
        )
    if initial_model_stage_profile_enabled:
        contract = _validate_profile_execution_contract(
            meta,
            contract_mode=execution_contract,
            image_shape=image_shape,
            requested_image_batch_size=requested_image_batch_size,
            environment=environment,
        )
    else:
        contract = {
            "mode": execution_contract,
            "profile_checked": False,
            "profile_exact": False,
            "environment": _validate_profile_contract_environment(
                execution_contract,
                environment,
            ),
            "reason": "initial-model stage profiler disabled for production-topology timing",
        }
    if production_route is not None:
        contract["production_route"] = production_route
    import numpy as np

    selected_ids_le = np.asarray(selected_particle_ids, dtype="<i8")
    particle_fingerprint = {
        "count": len(selected_particle_ids),
        "encoding": "ordered_little_endian_int64",
        "sha256": hashlib.sha256(
            selected_ids_le.tobytes(order="C")
        ).hexdigest(),
    }
    return {
        "output_prefix": str(output_prefix.resolve()),
        "meta_path": str(meta_path.resolve()),
        "meta_sha256": _sha256(meta_path),
        "continuation_path": str(continuation_path.resolve()),
        "iteration_profile": profile,
        "sparse_pass2_profile": meta.get("sparse_pass2_profile_summary"),
        "halfset_profiles": {
            key: value for key, value in meta.items() if key.startswith("halfset_") and key.endswith("_profile_summary")
        },
        "schedule": {key: meta[key] for key in schedule_keys},
        "selected_particle_fingerprint": particle_fingerprint,
        "execution_contract": contract,
    }


def _configure_initial_model_stage_profile(enabled: bool) -> None:
    """Select profiling by presence, matching the production driver's contract."""

    if enabled:
        os.environ["RECOVAR_INITIAL_MODEL_PROFILE"] = "1"
    else:
        os.environ.pop("RECOVAR_INITIAL_MODEL_PROFILE", None)


def _stage_profile_enabled_by_arm(mode: str) -> dict[str, bool]:
    if mode not in _STAGE_PROFILE_CHOICES:
        raise ValueError(f"unsupported stage-profile mode {mode!r}")
    return {
        "cold": mode in {"on", "certify-off"},
        "warm": mode == "on",
    }


def _effects_barrier() -> None:
    import jax

    barrier = getattr(jax, "effects_barrier", None)
    if barrier is not None:
        barrier()


def _raw_image_cache_loader_topology(loader) -> dict[str, object]:
    """Describe the concrete metadata wrapper and leaf loaders being cached."""
    import numpy as np

    file_map = getattr(loader, "_file_map", None)
    mapped_files: list[str] = []
    mapped_indices = np.empty(0, dtype="<i8")
    if file_map is not None and {"mrc_file", "mrc_index"}.issubset(file_map.columns):
        mapped_files = sorted({str(path) for path in file_map["mrc_file"].tolist()})
        mapped_indices = np.asarray(file_map["mrc_index"], dtype="<i8")

    unique_indices = np.unique(mapped_indices)
    mapping_is_unique = bool(unique_indices.size == mapped_indices.size)
    mapping_is_contiguous_set = bool(
        mapping_is_unique
        and unique_indices.size > 0
        and int(unique_indices[-1]) - int(unique_indices[0]) + 1 == unique_indices.size
    )
    mapping_is_strictly_ascending = bool(
        mapped_indices.size <= 1 or np.all(np.diff(mapped_indices) == 1)
    )

    raw_leaves = getattr(loader, "_loaders", {})
    leaf_items = sorted(raw_leaves.items(), key=lambda item: str(item[0])) if isinstance(raw_leaves, dict) else []
    leaf_loaders = []
    leaf_cached = []
    for path, leaf in leaf_items:
        selection = np.asarray(getattr(leaf, "selection_indices", []), dtype="<i8")
        leaf_loaders.append(
            {
                "path": str(path),
                "io_path": str(getattr(leaf, "_filepath", "")),
                "loader_type": f"{type(leaf).__module__}.{type(leaf).__qualname__}",
                "num_images": int(getattr(leaf, "num_images")),
                "image_size": int(getattr(leaf, "image_size")),
                "dtype": np.dtype(getattr(leaf, "_dtype", np.float32)).str,
                "selection_indices_sha256": hashlib.sha256(selection.tobytes(order="C")).hexdigest(),
            }
        )
        leaf_cached.append(getattr(leaf, "_cached", None) is not None)

    return {
        "mapped_rows": int(mapped_indices.size),
        "mapped_files": mapped_files,
        "mapped_file_count": len(mapped_files),
        "mapping_unique_index_count": int(unique_indices.size),
        "mapping_min_index": int(unique_indices[0]) if unique_indices.size else None,
        "mapping_max_index": int(unique_indices[-1]) if unique_indices.size else None,
        "mapping_is_unique": mapping_is_unique,
        "mapping_is_contiguous_set": mapping_is_contiguous_set,
        "mapping_is_strictly_ascending": mapping_is_strictly_ascending,
        "mapping_mrc_indices_sha256": hashlib.sha256(mapped_indices.tobytes(order="C")).hexdigest(),
        "leaf_loader_count": len(leaf_loaders),
        "leaf_loaders": leaf_loaders,
        "leaf_cached": leaf_cached,
    }


@contextmanager
def _capture_raw_image_cache_loads(
    enabled: bool,
) -> Iterator[list[dict[str, object]] | None]:
    """Observe diagnostic ``load_all`` calls without changing cache policy."""

    if not enabled:
        yield None
        return

    import numpy as np

    from recovar.data_io.image_loader import ImageLoader

    events: list[dict[str, object]] = []
    original = ImageLoader.load_all

    def audited_load_all(loader):
        cached_before = getattr(loader, "_cached", None)
        num_images = int(getattr(loader, "num_images"))
        image_size = int(getattr(loader, "image_size"))
        dtype = np.dtype(getattr(loader, "_dtype", np.float32))
        topology_before = _raw_image_cache_loader_topology(loader)
        resources_before = _process_resource_snapshot()
        started = time.perf_counter()
        result = original(loader)
        elapsed_s = float(time.perf_counter() - started)
        resources_after = _process_resource_snapshot()
        cached_after = getattr(loader, "_cached", None)
        topology_after = _raw_image_cache_loader_topology(loader)
        leaf_cached_before = topology_before.pop("leaf_cached")
        leaf_cached_after = topology_after.pop("leaf_cached")
        if topology_after != topology_before:
            raise RuntimeError("raw-image cache loader topology changed while loading")
        topology_before["leaf_cached_before"] = leaf_cached_before
        topology_before["leaf_cached_after"] = leaf_cached_after
        rss_before = int(resources_before["current_rss_kb"]) * 1024
        rss_after = int(resources_after["current_rss_kb"]) * 1024
        hwm_before = int(resources_before["high_water_rss_kb"]) * 1024
        hwm_after = int(resources_after["high_water_rss_kb"]) * 1024
        events.append(
            {
                "loader_type": f"{type(loader).__module__}.{type(loader).__qualname__}",
                "num_images": num_images,
                "image_size": image_size,
                "dtype": dtype.str,
                "estimated_bytes": int(num_images * image_size * image_size * dtype.itemsize),
                "cached_before": cached_before is not None,
                "cached_after": cached_after is not None,
                "cached_nbytes": int(getattr(cached_after, "nbytes", 0)),
                "cached_shape": list(getattr(cached_after, "shape", ())),
                "cached_dtype": np.dtype(getattr(cached_after, "dtype", dtype)).str,
                "cached_c_contiguous": bool(getattr(getattr(cached_after, "flags", None), "c_contiguous", False)),
                "cached_writeable": bool(getattr(getattr(cached_after, "flags", None), "writeable", False)),
                "loader_topology": topology_before,
                "elapsed_s": elapsed_s,
                "current_rss_before_bytes": rss_before,
                "current_rss_after_bytes": rss_after,
                "current_rss_delta_bytes": rss_after - rss_before,
                "high_water_rss_before_bytes": hwm_before,
                "high_water_rss_after_bytes": hwm_after,
                "high_water_rss_delta_bytes": hwm_after - hwm_before,
            }
        )
        return result

    ImageLoader.load_all = audited_load_all
    try:
        yield events
    finally:
        ImageLoader.load_all = original


def _nearest_repo_frame(stack: list[traceback.FrameSummary]) -> dict[str, object] | None:
    repo_root = Path(__file__).resolve().parents[1]
    this_file = Path(__file__).resolve()
    for frame in reversed(stack):
        path = Path(frame.filename).resolve()
        if path == this_file:
            continue
        try:
            relative = path.relative_to(repo_root)
        except ValueError:
            continue
        return {
            "file": str(relative),
            "line": int(frame.lineno),
            "function": frame.name,
            "source": frame.line,
        }
    return None


@contextmanager
def _capture_cold_compile_calls(path: Path | None) -> Iterator[list[dict[str, object]] | None]:
    if path is None:
        yield None
        return
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from jax._src import compiler

    original_compile = compiler.compile_or_get_cached
    original_cache_read = compiler._cache_read
    signature = inspect.signature(original_compile)
    thread_state = threading.local()
    write_lock = threading.Lock()
    records: list[dict[str, object]] = []

    @functools.wraps(original_cache_read)
    def traced_cache_read(*args, **kwargs):
        result = original_cache_read(*args, **kwargs)
        current = getattr(thread_state, "compile_record", None)
        if current is not None:
            current["cache_lookup"] = True
            current["cache_hit"] = result[0] is not None
        return result

    @functools.wraps(original_compile)
    def traced_compile(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        computation = bound.arguments["computation"]
        try:
            sym_name = computation.operation.attributes["sym_name"]
            module_name = compiler.ir.StringAttr(sym_name).value
        except Exception:
            module_name = str(getattr(computation, "name", "<unknown>"))
        record: dict[str, object] = {
            "module": module_name,
            "cache_lookup": False,
            "cache_hit": None,
            "thread_id": threading.get_ident(),
            "thread_name": threading.current_thread().name,
            "callsite": _nearest_repo_frame(traceback.extract_stack()[:-1]),
        }
        started = time.perf_counter()
        thread_state.compile_record = record
        try:
            result = original_compile(*args, **kwargs)
        except BaseException as error:
            record["exception"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["elapsed_s"] = float(time.perf_counter() - started)
            if record["cache_lookup"]:
                record["cache_status"] = "hit" if record["cache_hit"] else "miss"
            else:
                record["cache_status"] = "not_checked"
            thread_state.compile_record = None
            with write_lock:
                record["sequence"] = len(records)
                records.append(record)
                with path.open("a") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
        return result

    compiler._cache_read = traced_cache_read
    compiler.compile_or_get_cached = traced_compile
    try:
        yield records
    finally:
        compiler.compile_or_get_cached = original_compile
        compiler._cache_read = original_cache_read


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoint = args.checkpoint_optimiser.resolve(strict=True)
    input_star = args.input_star.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    static_input_manifest = (
        args.static_input_manifest.resolve(strict=True)
        if args.static_input_manifest is not None
        else None
    )
    output_root = args.output_root.resolve()
    args.checkpoint_optimiser = checkpoint
    args.input_star = input_star
    args.data_dir = data_dir
    if int(args.checkpoint_iteration) < 0:
        raise ValueError("checkpoint-iteration must be non-negative")
    if int(args.nr_iter) <= int(args.checkpoint_iteration):
        raise ValueError("nr-iter must exceed checkpoint-iteration")
    if int(args.exact_local_physical_order_chunk_size) < 0:
        raise ValueError("exact-local-physical-order-chunk-size must be non-negative")
    if int(args.image_size) <= 0:
        raise ValueError("image-size must be positive")
    if args.execution_contract == "all_optimized_q32" and not args.stable_fourier_window_shapes:
        raise ValueError(
            "all_optimized_q32 requires --stable-fourier-window-shapes"
        )
    if args.execution_contract == "default" and args.stable_fourier_window_shapes:
        raise ValueError("default execution contract forbids stable Fourier shapes")
    if (
        args.initial_model_stage_profile == "certify-off"
        and args.execution_contract != "all_optimized_q32"
    ):
        raise ValueError(
            "certify-off requires the all_optimized_q32 execution contract"
        )
    expected_schedule: dict[str, object] | None = None
    expected_particle_sha256: str | None = None
    if args.initial_model_stage_profile == "certify-off":
        if static_input_manifest is None:
            raise ValueError("certify-off requires --static-input-manifest")
        if not args.expected_schedule_json:
            raise ValueError("certify-off requires --expected-schedule-json")
        if not args.expected_selected_particle_sha256:
            raise ValueError(
                "certify-off requires --expected-selected-particle-sha256"
            )
        try:
            raw_expected_schedule = json.loads(args.expected_schedule_json)
        except json.JSONDecodeError as error:
            raise ValueError("expected-schedule-json is not valid JSON") from error
        expected_schedule = _validate_certify_schedule(
            raw_expected_schedule,
            label="expected",
        )
        expected_particle_sha256 = args.expected_selected_particle_sha256
        if not (
            len(expected_particle_sha256) == 64
            and all(character in "0123456789abcdef" for character in expected_particle_sha256)
        ):
            raise ValueError(
                "expected-selected-particle-sha256 must be 64 lowercase hex characters"
            )
    elif args.expected_schedule_json or args.expected_selected_particle_sha256:
        raise ValueError("pinned certify-off expectations require certify-off mode")
    contract_environment = _validate_profile_contract_environment(
        args.execution_contract,
    )
    runtime_environment = _qualified_profile_runtime_environment()
    if output_root.exists():
        if any(output_root.iterdir()):
            raise FileExistsError(f"output root is not empty: {output_root}")
    else:
        output_root.mkdir(parents=True)

    # The stage profiler synchronizes each major phase.  The off mode measures
    # production topology and retains only the final output barrier below;
    # certify-off profiles only the cold route and certifies the warm route
    # from always-emitted counters.
    stage_profile_enabled_by_arm = _stage_profile_enabled_by_arm(
        args.initial_model_stage_profile
    )
    os.environ.setdefault("JAX_LOG_COMPILES", "1")

    profiler_start: Callable[[], None] | None = None
    profiler_stop: Callable[[], None] | None = None
    if args.cuda_profiler_range:
        profiler_start, profiler_stop = _load_cuda_profiler()

    reports: dict[str, dict[str, object]] = {}
    target_iteration = int(args.checkpoint_iteration) + 1
    qualified_static_manifest_sha256 = (
        _sha256(static_input_manifest)
        if static_input_manifest is not None
        else None
    )
    qualified_static_inputs = (
        _validate_static_input_manifest(static_input_manifest)
        if static_input_manifest is not None
        else {
            str(checkpoint): _sha256(checkpoint),
            str(input_star): _sha256(input_star),
        }
    )
    initial_model_command: Callable[[list[str]], int] | None = None
    with _capture_raw_image_cache_loads(bool(args.audit_raw_image_cache)) as cache_events:
        for label in ("cold", "warm"):
            stage_profile_enabled = stage_profile_enabled_by_arm[label]
            _configure_initial_model_stage_profile(stage_profile_enabled)
            sealed_before = _validate_sealed_recovar_environment(
                args.execution_contract,
                stage_profile_enabled=stage_profile_enabled,
                expected_runtime_environment=runtime_environment,
            )
            if (
                static_input_manifest is not None
                and _sha256(static_input_manifest) != qualified_static_manifest_sha256
            ):
                raise RuntimeError(f"{label} static-input manifest changed")
            static_inputs_before = (
                _validate_static_input_manifest(static_input_manifest)
                if static_input_manifest is not None
                else {
                    str(checkpoint): _sha256(checkpoint),
                    str(input_star): _sha256(input_star),
                }
            )
            if static_inputs_before != qualified_static_inputs:
                raise RuntimeError(f"{label} static inputs changed before execution")
            prefix = output_root / label / "run"
            prefix.parent.mkdir(parents=True, exist_ok=False)
            command = _recovar_argv(args=args, output_prefix=prefix)
            capture = label == "warm" and profiler_start is not None
            event_start = len(cache_events) if cache_events is not None else 0
            resources_before = _process_resource_snapshot()
            started = time.perf_counter()
            if capture:
                profiler_start()
            compile_log = args.cold_compile_callsite_log if label == "cold" else None
            with _capture_cold_compile_calls(compile_log) as compile_records:
                if initial_model_command is None:
                    from recovar.commands.initial_model import main as imported_initial_model_command

                    initial_model_command = imported_initial_model_command
                try:
                    status = int(initial_model_command(command))
                    _effects_barrier()
                finally:
                    if capture:
                        assert profiler_stop is not None
                        profiler_stop()
            wall_s = float(time.perf_counter() - started)
            resources_after = _process_resource_snapshot()
            if status != 0:
                raise RuntimeError(f"{label} continuation exited with status {status}")
            sealed_after = _validate_sealed_recovar_environment(
                args.execution_contract,
                stage_profile_enabled=stage_profile_enabled,
                expected_runtime_environment=runtime_environment,
            )
            if (
                static_input_manifest is not None
                and _sha256(static_input_manifest) != qualified_static_manifest_sha256
            ):
                raise RuntimeError(f"{label} static-input manifest changed")
            static_inputs_after = (
                _validate_static_input_manifest(static_input_manifest)
                if static_input_manifest is not None
                else {
                    str(checkpoint): _sha256(checkpoint),
                    str(input_star): _sha256(input_star),
                }
            )
            if static_inputs_after != qualified_static_inputs:
                raise RuntimeError(f"{label} static inputs changed during execution")
            reports[label] = {
                "wall_s": wall_s,
                "argv": command,
                "sealed_recovar_environment": {
                    "before": sealed_before,
                    "after": sealed_after,
                    "unchanged": True,
                },
                "static_input_sha256": {
                    "before": static_inputs_before,
                    "after": static_inputs_after,
                },
                "process_resources": {
                    "before": resources_before,
                    "after": resources_after,
                    "delta": _process_resource_delta(resources_before, resources_after),
                },
                **_profile_metadata(
                    prefix,
                    target_iteration,
                    execution_contract=args.execution_contract,
                    image_shape=(int(args.image_size), int(args.image_size)),
                    initial_model_stage_profile_enabled=stage_profile_enabled,
                    requested_image_batch_size=int(args.image_batch_size),
                    exact_local_bucket_radix=int(args.exact_local_bucket_radix),
                    exact_local_physical_order_chunk_size=int(
                        args.exact_local_physical_order_chunk_size
                    ),
                    environment=sealed_before["present_recovar_environment"],
                ),
            }
            if cache_events is not None:
                reports[label]["raw_image_cache_audit"] = {
                    "mode": os.environ.get("RECOVAR_EM_RAW_IMAGE_CACHE", "auto"),
                    "max_gb": float(os.environ.get("RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB", "16")),
                    "load_all_events": [dict(event) for event in cache_events[event_start:]],
                }
            if compile_records is not None:
                reports[label]["cold_compile_callsite_count"] = len(compile_records)

    production_topology_certificate = (
        _certify_profile_free_warm(
            reports["cold"],
            reports["warm"],
            expected_schedule=expected_schedule,
            expected_selected_particle_sha256=expected_particle_sha256,
        )
        if args.initial_model_stage_profile == "certify-off"
        else None
    )
    report = {
        "schema": "recovar.vdam_late_iteration_profile.v1",
        "classification": "diagnostic_performance_only",
        "checkpoint_iteration": int(args.checkpoint_iteration),
        "profiled_iteration": target_iteration,
        "nr_iter_schedule": int(args.nr_iter),
        "checkpoint_optimiser": str(checkpoint),
        "checkpoint_optimiser_sha256": _sha256(checkpoint),
        "input_star": str(input_star),
        "input_star_sha256": _sha256(input_star),
        "data_dir": str(data_dir),
        "static_input_manifest": (
            str(static_input_manifest) if static_input_manifest is not None else None
        ),
        "static_input_manifest_sha256": (
            qualified_static_manifest_sha256
        ),
        "cuda_profiler_range": bool(args.cuda_profiler_range),
        "initial_model_stage_profile_mode": args.initial_model_stage_profile,
        "initial_model_stage_profile_enabled_by_arm": stage_profile_enabled_by_arm,
        "raw_image_cache_audit_enabled": bool(args.audit_raw_image_cache),
        "cold_compile_callsite_log": (
            str(args.cold_compile_callsite_log.resolve())
            if args.cold_compile_callsite_log is not None
            else None
        ),
        "execution_contract_mode": args.execution_contract,
        "execution_contract_environment": contract_environment,
        "image_shape": [int(args.image_size), int(args.image_size)],
        "exact_local_bucket_radix": int(args.exact_local_bucket_radix),
        "exact_local_physical_order_chunk_size": int(args.exact_local_physical_order_chunk_size),
        "production_topology_certificate": production_topology_certificate,
        "cold": reports["cold"],
        "warm": reports["warm"],
        "cold_minus_warm_wall_s": float(reports["cold"]["wall_s"]) - float(reports["warm"]["wall_s"]),
    }
    report_path = output_root / "profile_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
