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
import hashlib
import json
import os
import resource
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-optimiser", type=Path, required=True)
    parser.add_argument("--input-star", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-iteration", type=int, default=180)
    parser.add_argument("--nr-iter", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=29)
    parser.add_argument("--image-batch-size", type=int, default=500)
    parser.add_argument("--exact-local-bucket-radix", type=int, choices=(2, 4), default=4)
    parser.add_argument("--exact-local-physical-order-chunk-size", type=int, default=0)
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
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "--do_run_C1",
        "1",
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
        "--require_custom_cuda",
        "--diagnostic_continue_optimiser",
        str(args.checkpoint_optimiser),
        "--diagnostic_stop_after_iteration",
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


def _profile_metadata(output_prefix: Path, iteration: int) -> dict[str, object]:
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
    if int(continuation.get("iteration", -1)) + 1 != iteration:
        raise RuntimeError("continuation metadata does not identify the incoming iteration")
    profile = meta.get("vdam_iteration_profile_summary")
    if not isinstance(profile, dict) or not profile:
        raise RuntimeError("RECOVAR_INITIAL_MODEL_PROFILE did not emit stage timings")
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
    try:
        subset_size = int(meta["subset_size"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("iteration metadata subset_size is invalid") from exc
    if subset_size == 0 or subset_size < -1:
        raise RuntimeError(f"iteration metadata subset_size is invalid: {subset_size}")
    selected_particle_ids = meta.get("selected_particle_ids")
    if not isinstance(selected_particle_ids, list):
        raise RuntimeError("iteration metadata lacks selected_particle_ids")
    if any(
        isinstance(particle_id, bool) or not isinstance(particle_id, int) or particle_id < 0
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
    return {
        "meta_path": str(meta_path.resolve()),
        "meta_sha256": _sha256(meta_path),
        "continuation_path": str(continuation_path.resolve()),
        "iteration_profile": profile,
        "sparse_pass2_profile": meta.get("sparse_pass2_profile_summary"),
        "halfset_profiles": {
            key: value for key, value in meta.items() if key.startswith("halfset_") and key.endswith("_profile_summary")
        },
        "schedule": {key: meta[key] for key in schedule_keys},
    }


def _effects_barrier() -> None:
    import jax

    barrier = getattr(jax, "effects_barrier", None)
    if barrier is not None:
        barrier()


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
        resources_before = _process_resource_snapshot()
        started = time.perf_counter()
        result = original(loader)
        elapsed_s = float(time.perf_counter() - started)
        resources_after = _process_resource_snapshot()
        cached_after = getattr(loader, "_cached", None)
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoint = args.checkpoint_optimiser.resolve(strict=True)
    input_star = args.input_star.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
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
    if output_root.exists():
        if any(output_root.iterdir()):
            raise FileExistsError(f"output root is not empty: {output_root}")
    else:
        output_root.mkdir(parents=True)

    # The stage profiler synchronizes each major phase, which is intentional:
    # the run is for attribution and cannot be promoted as a science result.
    os.environ["RECOVAR_INITIAL_MODEL_PROFILE"] = "1"
    os.environ.setdefault("JAX_LOG_COMPILES", "1")

    from scripts.run_ab_initio import main as run_ab_initio

    profiler_start: Callable[[], None] | None = None
    profiler_stop: Callable[[], None] | None = None
    if args.cuda_profiler_range:
        profiler_start, profiler_stop = _load_cuda_profiler()

    reports: dict[str, dict[str, object]] = {}
    target_iteration = int(args.checkpoint_iteration) + 1
    with _capture_raw_image_cache_loads(bool(args.audit_raw_image_cache)) as cache_events:
        for label in ("cold", "warm"):
            prefix = output_root / label / "run"
            prefix.parent.mkdir(parents=True, exist_ok=False)
            command = _recovar_argv(args=args, output_prefix=prefix)
            capture = label == "warm" and profiler_start is not None
            event_start = len(cache_events) if cache_events is not None else 0
            resources_before = _process_resource_snapshot()
            started = time.perf_counter()
            if capture:
                profiler_start()
            try:
                status = int(run_ab_initio(command))
                _effects_barrier()
            finally:
                if capture:
                    assert profiler_stop is not None
                    profiler_stop()
            wall_s = float(time.perf_counter() - started)
            resources_after = _process_resource_snapshot()
            if status != 0:
                raise RuntimeError(f"{label} continuation exited with status {status}")
            reports[label] = {
                "wall_s": wall_s,
                "argv": command,
                "process_resources": {
                    "before": resources_before,
                    "after": resources_after,
                    "delta": _process_resource_delta(resources_before, resources_after),
                },
                **_profile_metadata(prefix, target_iteration),
            }
            if cache_events is not None:
                reports[label]["raw_image_cache_audit"] = {
                    "mode": os.environ.get("RECOVAR_EM_RAW_IMAGE_CACHE", "auto"),
                    "max_gb": float(os.environ.get("RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB", "16")),
                    "load_all_events": [dict(event) for event in cache_events[event_start:]],
                }

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
        "cuda_profiler_range": bool(args.cuda_profiler_range),
        "raw_image_cache_audit_enabled": bool(args.audit_raw_image_cache),
        "exact_local_bucket_radix": int(args.exact_local_bucket_radix),
        "exact_local_physical_order_chunk_size": int(args.exact_local_physical_order_chunk_size),
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
