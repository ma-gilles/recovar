#!/usr/bin/env python3
"""Diagnostic A/B timing for the default-off VDAM coarse prehalf primitive.

The default shape reproduces GF46 iteration 181's physical coarse call.  CLI
overrides may only make that workload smaller.  This script deliberately does
not make an acceptance decision: it records alternating native-atomic timing,
exact discrete comparisons, and numerical distances for later gated review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

SCHEMA = "recovar.vdam_coarse_prehalf_benchmark.v1"
CLASSIFICATION = "diagnostic_only_no_decision"
API_NAMES = ("serial", "multistream")
PREHALF_SEQUENCE = (False, True)

GF46_CURRENT_SIZE = 128
GF46_MODEL_MAX_R = 64
GF46_ROTATION_COUNT = 36_864
GF46_TRANSLATION_COUNT = 29
GF46_PHYSICAL_BATCH_SIZE = 187
MAX_REPETITIONS = 20
MAX_TOP_SUPPORT_SIZE = 256


class BenchmarkError(RuntimeError):
    """Raised when the diagnostic cannot produce trustworthy evidence."""


@dataclass(frozen=True)
class BenchmarkOptions:
    current_size: int = GF46_CURRENT_SIZE
    model_max_r: int = GF46_MODEL_MAX_R
    rotation_count: int = GF46_ROTATION_COUNT
    translation_count: int = GF46_TRANSLATION_COUNT
    physical_batch_size: int = GF46_PHYSICAL_BATCH_SIZE
    actual_batch_size: int = GF46_PHYSICAL_BATCH_SIZE
    repetitions: int = 5
    top_support_size: int = 32
    seed: int = 46


@dataclass(frozen=True)
class Workload:
    projector_full: Any
    rotation_matrices: Any
    images: Any
    translation_angles: Any
    weight: Any
    initial_diff2: Any
    full_to_compact: Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkError(message)


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
        f"{label} must be an integer",
    )
    return int(value)


def _validate_options(options: BenchmarkOptions) -> None:
    values = {name: _integer(value, name) for name, value in asdict(options).items()}
    _require(1 <= values["current_size"] <= GF46_CURRENT_SIZE, "current_size is outside 1--128")
    _require(
        1 <= values["model_max_r"] <= min(GF46_MODEL_MAX_R, max(1, values["current_size"] // 2)),
        "model_max_r exceeds the bounded current-size radius",
    )
    _require(
        1 <= values["rotation_count"] <= GF46_ROTATION_COUNT,
        "rotation_count is outside the bounded GF46 range",
    )
    _require(
        1 <= values["translation_count"] <= GF46_TRANSLATION_COUNT,
        "translation_count is outside the bounded GF46 range",
    )
    _require(
        1 <= values["physical_batch_size"] <= GF46_PHYSICAL_BATCH_SIZE,
        "physical_batch_size is outside the bounded GF46 range",
    )
    _require(
        1 <= values["actual_batch_size"] <= values["physical_batch_size"],
        "actual_batch_size must be inside the physical batch",
    )
    _require(
        2 <= values["repetitions"] <= MAX_REPETITIONS,
        f"repetitions must be in 2--{MAX_REPETITIONS}",
    )
    _require(
        1
        <= values["top_support_size"]
        <= min(MAX_TOP_SUPPORT_SIZE, values["rotation_count"] * values["translation_count"]),
        "top_support_size exceeds the bounded flattened score count",
    )
    _require(0 <= values["seed"] < 2**32, "seed is outside the uint32 range")


def _expected_output_shape(options: BenchmarkOptions) -> tuple[int, int, int]:
    return (
        options.physical_batch_size,
        options.rotation_count,
        options.translation_count,
    )


def _validate_workload(workload: Workload, options: BenchmarkOptions) -> None:
    packed_pixels = options.current_size * (options.current_size // 2 + 1)
    projector_size = 2 * options.model_max_r + 3
    expected = {
        "projector_full": ((projector_size,) * 3, np.dtype(np.complex64)),
        "rotation_matrices": ((options.rotation_count, 3, 3), np.dtype(np.float32)),
        "images": ((options.physical_batch_size, packed_pixels), np.dtype(np.complex64)),
        "translation_angles": ((options.translation_count, 2), np.dtype(np.float32)),
        "weight": ((options.physical_batch_size, packed_pixels), np.dtype(np.float32)),
        "initial_diff2": ((options.physical_batch_size,), np.dtype(np.float32)),
        "full_to_compact": ((packed_pixels,), np.dtype(np.int32)),
    }
    for name, (shape, dtype) in expected.items():
        value = getattr(workload, name)
        observed_shape = tuple(int(item) for item in getattr(value, "shape", ()))
        try:
            observed_dtype = np.dtype(getattr(value, "dtype"))
        except (TypeError, ValueError) as exc:
            raise BenchmarkError(f"{name} has no valid dtype") from exc
        _require(
            observed_shape == shape and observed_dtype == dtype,
            f"{name} shape/dtype differs: expected={shape}/{dtype}, observed={observed_shape}/{observed_dtype}",
        )


def _rotation_matrices(rng: np.random.Generator, count: int) -> np.ndarray:
    quaternions = rng.standard_normal((count, 4), dtype=np.float32)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    _require(bool(np.all(np.isfinite(norms))) and bool(np.all(norms > 0)), "quaternion draw is invalid")
    quaternions /= norms
    w, x, y, z = quaternions.T
    rotations = np.empty((count, 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - z * w)
    rotations[:, 0, 2] = 2 * (x * z + y * w)
    rotations[:, 1, 0] = 2 * (x * y + z * w)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - x * w)
    rotations[:, 2, 0] = 2 * (x * z - y * w)
    rotations[:, 2, 1] = 2 * (y * z + x * w)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotations


def _complex_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    *,
    scale: float,
) -> np.ndarray:
    real = rng.standard_normal(shape, dtype=np.float32)
    imaginary = rng.standard_normal(shape, dtype=np.float32)
    return np.asarray(scale * real + np.complex64(1j * scale) * imaginary, dtype=np.complex64)


def _generate_host_workload(options: BenchmarkOptions) -> Workload:
    _validate_options(options)
    rng = np.random.default_rng(options.seed)
    packed_pixels = options.current_size * (options.current_size // 2 + 1)
    projector_size = 2 * options.model_max_r + 3
    workload = Workload(
        projector_full=_complex_normal(rng, (projector_size,) * 3, scale=0.02),
        rotation_matrices=_rotation_matrices(rng, options.rotation_count),
        images=_complex_normal(
            rng,
            (options.physical_batch_size, packed_pixels),
            scale=0.02,
        ),
        translation_angles=rng.uniform(
            -0.2,
            0.2,
            (options.translation_count, 2),
        ).astype(np.float32),
        weight=rng.uniform(
            0.1,
            3.0,
            (options.physical_batch_size, packed_pixels),
        ).astype(np.float32),
        initial_diff2=rng.uniform(5.0, 15.0, options.physical_batch_size).astype(np.float32),
        full_to_compact=np.arange(packed_pixels, dtype=np.int32),
    )
    _validate_workload(workload, options)
    for name in workload.__dataclass_fields__:
        _require(bool(np.all(np.isfinite(getattr(workload, name)))), f"{name} contains non-finite values")
    return workload


def _device_workload(host: Workload, jnp: Any) -> Workload:
    return Workload(**{name: jnp.asarray(getattr(host, name)) for name in host.__dataclass_fields__})


def _synchronize(output: Any) -> Any:
    block_until_ready = getattr(output, "block_until_ready", None)
    _require(callable(block_until_ready), "projector output cannot be synchronized")
    synchronized = block_until_ready()
    return output if synchronized is None else synchronized


def _validate_output_shape(output: Any, expected_shape: tuple[int, int, int], label: str) -> None:
    observed = tuple(int(value) for value in getattr(output, "shape", ()))
    _require(observed == expected_shape, f"{label} output shape differs: {observed} != {expected_shape}")


def _comparison_summary(
    reference: Any,
    candidate: Any,
    *,
    active_batch_size: int,
    top_support_size: int,
) -> dict[str, Any]:
    """Reduce one full GPU output pair to bounded numerical/discrete evidence."""

    import jax
    import jax.numpy as jnp

    _require(tuple(reference.shape) == tuple(candidate.shape), "comparison output shapes differ")
    _require(reference.ndim == 3, "comparison outputs must have rank three")
    _require(1 <= active_batch_size <= reference.shape[0], "comparison active batch is invalid")
    flattened_count = int(reference.shape[1]) * int(reference.shape[2])
    _require(1 <= top_support_size <= flattened_count, "comparison support size is invalid")
    left = reference[:active_batch_size].reshape(active_batch_size, flattened_count)
    right = candidate[:active_batch_size].reshape(active_batch_size, flattened_count)
    delta = right - left
    tiny = np.float32(np.finfo(np.float32).tiny)
    reference_norm = jnp.linalg.norm(left)
    delta_norm = jnp.linalg.norm(delta)
    max_reference = jnp.max(jnp.abs(left))
    max_error = jnp.max(jnp.abs(delta))
    left_support = jnp.sort(jax.lax.top_k(-left, top_support_size)[1], axis=1)
    right_support = jnp.sort(jax.lax.top_k(-right, top_support_size)[1], axis=1)
    reduced = jax.block_until_ready(
        (
            jnp.all(jnp.isfinite(left)) & jnp.all(jnp.isfinite(right)),
            delta_norm / jnp.maximum(reference_norm, tiny),
            max_error,
            max_error / jnp.maximum(max_reference, tiny),
            jnp.array_equal(left, right),
            jnp.array_equal(jnp.argmin(left, axis=1), jnp.argmin(right, axis=1)),
            jnp.array_equal(left_support, right_support),
        )
    )
    finite, relative_l2, max_abs, max_relative, exact, argmin, support = (np.asarray(value).item() for value in reduced)
    _require(bool(finite), "comparison contains non-finite projector output")
    numeric = tuple(float(value) for value in (relative_l2, max_abs, max_relative))
    _require(all(math.isfinite(value) and value >= 0 for value in numeric), "comparison metrics are invalid")
    return {
        "active_batch_size": int(active_batch_size),
        "relative_l2_error": numeric[0],
        "max_abs_error": numeric[1],
        "max_relative_error": numeric[2],
        "value_exact_equal": bool(exact),
        "argmin_equal": bool(argmin),
        "fixed_top_support_size": int(top_support_size),
        "fixed_top_support_equal": bool(support),
    }


def _validate_comparison(summary: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "active_batch_size",
        "relative_l2_error",
        "max_abs_error",
        "max_relative_error",
        "value_exact_equal",
        "argmin_equal",
        "fixed_top_support_size",
        "fixed_top_support_equal",
    }
    _require(set(summary) == required, "comparison summary fields differ")
    numeric = {}
    for name in ("relative_l2_error", "max_abs_error", "max_relative_error"):
        value = float(summary[name])
        _require(math.isfinite(value) and value >= 0, f"comparison {name} is invalid")
        numeric[name] = value
    booleans = {}
    for name in ("value_exact_equal", "argmin_equal", "fixed_top_support_equal"):
        _require(isinstance(summary[name], (bool, np.bool_)), f"comparison {name} must be boolean")
        booleans[name] = bool(summary[name])
    return {
        "active_batch_size": _integer(
            summary["active_batch_size"],
            "comparison active_batch_size",
        ),
        **numeric,
        **booleans,
        "fixed_top_support_size": _integer(
            summary["fixed_top_support_size"],
            "comparison fixed_top_support_size",
        ),
    }


def _timing_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = {
        flag: [float(row["elapsed_seconds"]) for row in samples if row["prehalf_weight"] is flag]
        for flag in PREHALF_SEQUENCE
    }
    _require(all(grouped.values()), "timing samples do not cover both static modes")
    for values in grouped.values():
        _require(all(math.isfinite(value) and value > 0 for value in values), "timing sample is invalid")
    control_median = float(statistics.median(grouped[False]))
    candidate_median = float(statistics.median(grouped[True]))

    def mode(values: list[float]) -> dict[str, Any]:
        return {
            "raw_seconds": values,
            "median_seconds": float(statistics.median(values)),
            "span_seconds": float(max(values) - min(values)),
        }

    return {
        "prehalf_off": mode(grouped[False]),
        "prehalf_on": mode(grouped[True]),
        "candidate_minus_control_median_seconds": candidate_median - control_median,
        "candidate_over_control_median": candidate_median / control_median,
        "control_minus_candidate_fraction": (control_median - candidate_median) / control_median,
    }


def _benchmark_api(
    name: str,
    invoke: Callable[[bool], Any],
    options: BenchmarkOptions,
    *,
    synchronize: Callable[[Any], Any] = _synchronize,
    compare: Callable[..., dict[str, Any]] = _comparison_summary,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    _require(name in API_NAMES, f"unknown benchmark API {name!r}")
    expected_shape = _expected_output_shape(options)
    warm: dict[bool, Any] = {}
    for prehalf_weight in PREHALF_SEQUENCE:
        output = synchronize(invoke(prehalf_weight))
        _validate_output_shape(output, expected_shape, f"{name} warmup prehalf={prehalf_weight}")
        warm[prehalf_weight] = output
    warm_comparison = _validate_comparison(
        compare(
            warm[False],
            warm[True],
            active_batch_size=options.actual_batch_size,
            top_support_size=options.top_support_size,
        )
    )
    samples: list[dict[str, Any]] = []
    repeats = {False: [], True: []}
    for repetition in range(options.repetitions):
        for prehalf_weight in PREHALF_SEQUENCE:
            start = _integer(clock_ns(), "start clock")
            output = synchronize(invoke(prehalf_weight))
            stop = _integer(clock_ns(), "stop clock")
            _validate_output_shape(
                output,
                expected_shape,
                f"{name} repetition {repetition} prehalf={prehalf_weight}",
            )
            _require(stop > start, "benchmark clock did not advance")
            elapsed_ns = stop - start
            samples.append(
                {
                    "repetition": repetition,
                    "prehalf_weight": prehalf_weight,
                    "elapsed_ns": elapsed_ns,
                    "elapsed_seconds": elapsed_ns / 1e9,
                }
            )
            repeats[prehalf_weight].append(
                _validate_comparison(
                    compare(
                        warm[prehalf_weight],
                        output,
                        active_batch_size=options.actual_batch_size,
                        top_support_size=options.top_support_size,
                    )
                )
            )
    return {
        "warmup_sequence": [False, True],
        "timed_sequence": [row["prehalf_weight"] for row in samples],
        "samples": samples,
        "timing": _timing_summary(samples),
        "numerical": {
            "warmup_control_vs_candidate": warm_comparison,
            "repeat_stability": {
                "prehalf_off": repeats[False],
                "prehalf_on": repeats[True],
            },
        },
    }


_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    _require(
        set(provenance).issuperset({"repository", "cuda", "device", "runtime", "script"}),
        "provenance sections are incomplete",
    )
    repository = provenance["repository"]
    cuda = provenance["cuda"]
    device = provenance["device"]
    _require(isinstance(repository, Mapping), "repository provenance is invalid")
    _require(isinstance(cuda, Mapping), "CUDA provenance is invalid")
    _require(isinstance(device, Mapping), "device provenance is invalid")
    _require(bool(_GIT_OBJECT_RE.fullmatch(str(repository.get("head", "")))), "repository head is invalid")
    _require(bool(_GIT_OBJECT_RE.fullmatch(str(repository.get("tree", "")))), "repository tree is invalid")
    _require(bool(_SHA256_RE.fullmatch(str(cuda.get("sha256", "")))), "CUDA SHA256 is invalid")
    _require(device.get("backend") == "gpu", "device provenance is not GPU-backed")


def _run_panel(
    options: BenchmarkOptions,
    invokers: Mapping[str, Callable[[bool], Any]],
    provenance: Mapping[str, Any],
    *,
    synchronize: Callable[[Any], Any] = _synchronize,
    compare: Callable[..., dict[str, Any]] = _comparison_summary,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    generated_at: str | None = None,
) -> dict[str, Any]:
    _validate_options(options)
    _require(set(invokers) == set(API_NAMES), "benchmark must exercise serial and multistream APIs")
    _validate_provenance(provenance)
    output_bytes = int(np.prod(_expected_output_shape(options), dtype=np.int64)) * 4
    exact_gf46 = options == BenchmarkOptions(
        repetitions=options.repetitions,
        top_support_size=options.top_support_size,
        seed=options.seed,
    )
    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "generated_at_utc": generated_at or datetime.now(timezone.utc).isoformat(),
        "configuration": {
            **asdict(options),
            "profile": "gf46_it181_geometry" if exact_gf46 else "bounded_smaller_override",
            "packed_pixel_count": options.current_size * (options.current_size // 2 + 1),
            "output_shape": list(_expected_output_shape(options)),
            "output_bytes_per_call": output_bytes,
            "canonical_reduction": False,
            "single_lane_canonical": False,
            "prehalf_default": False,
        },
        "protocol": {
            "compile_warmup_before_timing": True,
            "alternation": [False, True],
            "synchronizes_every_projector_output": True,
            "timings_exclude_numerical_reductions": True,
            "only_static_kernel_delta": "prehalf_weight=False/True",
            "relative_error_denominator": "max(control magnitude, float32 tiny)",
        },
        "provenance": dict(provenance),
        "apis": {
            name: _benchmark_api(
                name,
                invokers[name],
                options,
                synchronize=synchronize,
                compare=compare,
                clock_ns=clock_ns,
            )
            for name in API_NAMES
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise BenchmarkError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _collect_git_provenance(
    repo: Path,
    *,
    git_output: Callable[..., str] = _git_output,
) -> dict[str, Any]:
    resolved = repo.expanduser().resolve(strict=True)
    top = Path(git_output(resolved, "rev-parse", "--show-toplevel")).resolve(strict=True)
    _require(top == resolved, f"repo is not the resolved Git top level: {top} != {resolved}")
    head = git_output(resolved, "rev-parse", "HEAD")
    tree = git_output(resolved, "rev-parse", "HEAD^{tree}")
    status = git_output(resolved, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    diff = git_output(resolved, "diff", "--binary", "HEAD", "--")
    return {
        "path": str(resolved),
        "head": head,
        "tree": tree,
        "status_porcelain_v1": status,
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def _environment_flag(value: str | None) -> bool:
    return bool(value) and value.strip().lower() not in {"0", "false", "no", "off"}


def _runtime_contract(jax_module: Any, cuda_module: Any, environ: Mapping[str, str]) -> tuple[Any, Path]:
    _require(jax_module.default_backend() == "gpu", "benchmark requires the JAX GPU backend")
    _require(
        not _environment_flag(environ.get("RECOVAR_DISABLE_CUDA")),
        "custom CUDA is disabled by RECOVAR_DISABLE_CUDA",
    )
    _require(bool(cuda_module.custom_cuda_requested()), "RECOVAR custom CUDA was not requested")
    cuda_value = environ.get("RECOVAR_CUDA_LIB", "").strip()
    _require(bool(cuda_value), "RECOVAR_CUDA_LIB must pin the benchmark CUDA binary")
    cuda_path = Path(cuda_value).expanduser().resolve()
    _require(cuda_path.is_file(), f"RECOVAR_CUDA_LIB does not name a file: {cuda_path}")
    devices = list(jax_module.devices("gpu"))
    _require(len(devices) == 1, f"benchmark requires exactly one visible GPU, observed {len(devices)}")
    return devices[0], cuda_path


def _verify_loaded_cuda(cuda_module: Any, expected: Path) -> None:
    loaded = getattr(cuda_module, "_loaded_lib_path", None)
    _require(loaded is not None, "custom CUDA library was not loaded")
    _require(Path(loaded).resolve() == expected, f"loaded CUDA library differs: {loaded} != {expected}")


def _device_provenance(jax_module: Any, device: Any) -> dict[str, Any]:
    return {
        "backend": str(jax_module.default_backend()),
        "jax_version": str(jax_module.__version__),
        "device": str(device),
        "device_kind": str(getattr(device, "device_kind", "")),
        "device_id": int(getattr(device, "id", -1)),
        "process_index": int(getattr(device, "process_index", -1)),
        "local_hardware_id": int(getattr(device, "local_hardware_id", -1)),
        "platform": str(getattr(device, "platform", "")),
        "platform_version": str(getattr(getattr(device, "client", None), "platform_version", "")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }


def _projector_invokers(
    cuda_module: Any,
    workload: Workload,
    options: BenchmarkOptions,
    actual_batch_size: Any,
) -> dict[str, Callable[[bool], Any]]:
    operands = (
        workload.projector_full,
        workload.rotation_matrices,
        workload.images,
        workload.translation_angles,
        workload.weight,
        workload.initial_diff2,
        workload.full_to_compact,
    )
    common = {
        "current_size": options.current_size,
        "physical_image_size": options.current_size,
        "model_max_r": options.model_max_r,
        "canonical_reduction": False,
        "single_lane_canonical": False,
    }
    return {
        "serial": lambda prehalf_weight: cuda_module.relion_coarse_diff2_projector_f32(
            *operands,
            prehalf_weight=prehalf_weight,
            **common,
        ),
        "multistream": lambda prehalf_weight: cuda_module.relion_coarse_diff2_projector_multistream_f32(
            *operands,
            actual_batch_size=actual_batch_size,
            prehalf_weight=prehalf_weight,
            **common,
        ),
    }


def _live_report(options: BenchmarkOptions, repo: Path) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject

    device, cuda_path = _runtime_contract(jax, cuda_backproject, os.environ)
    cuda_backproject._ensure_ffi()
    _verify_loaded_cuda(cuda_backproject, cuda_path)
    host = _generate_host_workload(options)
    with jax.default_device(device):
        workload = _device_workload(host, jnp)
        _validate_workload(workload, options)
        actual_batch_size = jnp.asarray(options.actual_batch_size, dtype=jnp.int32)
        invokers = _projector_invokers(
            cuda_backproject,
            workload,
            options,
            actual_batch_size,
        )
        script_path = Path(__file__).resolve(strict=True)
        provenance = {
            "repository": _collect_git_provenance(repo),
            "cuda": {
                "path": str(cuda_path),
                "loaded_path": str(Path(cuda_backproject._loaded_lib_path).resolve()),
                "sha256": _sha256(cuda_path),
                "size_bytes": cuda_path.stat().st_size,
            },
            "device": _device_provenance(jax, device),
            "runtime": {
                "python_executable": str(Path(sys.executable).resolve()),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
            },
            "script": {
                "path": str(script_path),
                "sha256": _sha256(script_path),
            },
        }
        return _run_panel(options, invokers, provenance)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--current-size", type=int, default=GF46_CURRENT_SIZE)
    parser.add_argument("--model-max-r", type=int)
    parser.add_argument("--rotation-count", type=int, default=GF46_ROTATION_COUNT)
    parser.add_argument("--translation-count", type=int, default=GF46_TRANSLATION_COUNT)
    parser.add_argument("--physical-batch-size", type=int, default=GF46_PHYSICAL_BATCH_SIZE)
    parser.add_argument("--actual-batch-size", type=int)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--top-support-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=46)
    return parser.parse_args(argv)


def _options_from_args(args: argparse.Namespace) -> BenchmarkOptions:
    current_size = int(args.current_size)
    model_max_r = (
        int(args.model_max_r) if args.model_max_r is not None else min(GF46_MODEL_MAX_R, max(1, current_size // 2))
    )
    physical_batch_size = int(args.physical_batch_size)
    actual_batch_size = int(args.actual_batch_size) if args.actual_batch_size is not None else physical_batch_size
    options = BenchmarkOptions(
        current_size=current_size,
        model_max_r=model_max_r,
        rotation_count=int(args.rotation_count),
        translation_count=int(args.translation_count),
        physical_batch_size=physical_batch_size,
        actual_batch_size=actual_batch_size,
        repetitions=int(args.repetitions),
        top_support_size=int(args.top_support_size),
        seed=int(args.seed),
    )
    _validate_options(options)
    return options


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output_json.expanduser().resolve()
    if output.exists():
        raise BenchmarkError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    options = _options_from_args(args)
    report = _live_report(options, args.repo)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
