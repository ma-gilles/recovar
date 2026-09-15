#!/usr/bin/env python3
"""Measure raw-cache allocation topology outside the timed VDAM benchmark.

The timed profiler runs cold and warm continuations in one process, so Linux's
lifetime ``VmHWM`` cannot attribute a warm cache load after JAX has already set
a larger high-water mark.  This fresh-process canary uses Python's allocation
tracer around only ``StarLoader.load_all`` and then independently streams the
same STAR table without caching to prove byte-for-byte logical-order identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

from scripts.run_vdam_late_iteration_profile import (
    _process_resource_snapshot,
    _raw_image_cache_loader_topology,
)

SCHEMA = "recovar.vdam_raw_cache_memory_probe.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_bytes(array: np.ndarray) -> memoryview:
    contiguous = np.ascontiguousarray(array)
    return memoryview(contiguous).cast("B")


def probe(
    *,
    input_star: Path,
    data_dir: Path,
    output_json: Path,
    comparison_batch_size: int,
) -> dict[str, object]:
    """Trace one full cache load and compare it to streamed uncached reads."""

    if sys.flags.optimize != 0:
        raise RuntimeError("raw-cache memory probe requires Python assertions")
    if comparison_batch_size <= 0:
        raise ValueError("comparison-batch-size must be positive")
    if os.environ.get("RECOVAR_CACHE_DIR") != "":
        raise RuntimeError("RECOVAR_CACHE_DIR must be explicitly empty")

    from recovar.data_io.image_loader import StarLoader

    input_star = input_star.resolve(strict=True)
    data_dir = data_dir.resolve(strict=True)
    output_json = output_json.resolve()
    if output_json.exists():
        raise FileExistsError(f"output already exists: {output_json}")

    loader = StarLoader(
        str(input_star),
        datadir=str(data_dir),
        lazy=True,
        max_threads=1,
        skip_staging=True,
    )
    comparison = None
    try:
        topology_before = _raw_image_cache_loader_topology(loader)
        resources_before = _process_resource_snapshot()
        tracemalloc.start()
        tracemalloc.reset_peak()
        traced_baseline_current, traced_baseline_peak = tracemalloc.get_traced_memory()
        started = time.perf_counter()
        loader.load_all()
        elapsed_s = float(time.perf_counter() - started)
        traced_after_current, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        resources_after = _process_resource_snapshot()

        cached = getattr(loader, "_cached", None)
        if not isinstance(cached, np.ndarray):
            raise RuntimeError("StarLoader.load_all did not retain a NumPy cache")
        topology_after = _raw_image_cache_loader_topology(loader)

        cached_digest = hashlib.sha256(_array_bytes(cached)).hexdigest()
        comparison = StarLoader(
            str(input_star),
            datadir=str(data_dir),
            lazy=True,
            max_threads=1,
            skip_staging=True,
        )
        comparison_topology_before = _raw_image_cache_loader_topology(comparison)
        streamed_digest = hashlib.sha256()
        exact = True
        first_mismatch_index: int | None = None
        compared_images = 0
        batch_count = 0
        for start in range(0, int(loader.num_images), comparison_batch_size):
            stop = min(start + comparison_batch_size, int(loader.num_images))
            indices = np.arange(start, stop, dtype=np.int64)
            batch = comparison.get(indices)
            streamed_digest.update(_array_bytes(batch))
            batch_count += 1
            compared_images += int(batch.shape[0])
            cached_bytes = cached[start:stop].view(np.uint8).reshape(stop - start, -1)
            batch_bytes = np.ascontiguousarray(batch).view(np.uint8).reshape(stop - start, -1)
            equal_rows = np.all(cached_bytes == batch_bytes, axis=1)
            if not bool(np.all(equal_rows)):
                exact = False
                if first_mismatch_index is None:
                    first_mismatch_index = start + int(np.flatnonzero(~equal_rows)[0])
        comparison_topology_after = _raw_image_cache_loader_topology(comparison)

        rss_before = int(resources_before["current_rss_kb"]) * 1024
        rss_after = int(resources_after["current_rss_kb"]) * 1024
        hwm_before = int(resources_before["high_water_rss_kb"]) * 1024
        hwm_after = int(resources_after["high_water_rss_kb"]) * 1024
        payload: dict[str, object] = {
            "schema": SCHEMA,
            "classification": "untimed_memory_and_bitwise_equivalence_canary",
            "input_star": str(input_star),
            "input_star_sha256": _sha256(input_star),
            "data_dir": str(data_dir),
            "cache_dir_env": os.environ.get("RECOVAR_CACHE_DIR"),
            "comparison_batch_size": int(comparison_batch_size),
            "loader": {
                "loader_type": f"{type(loader).__module__}.{type(loader).__qualname__}",
                "num_images": int(loader.num_images),
                "image_size": int(loader.image_size),
                "dtype": np.dtype(getattr(loader, "_dtype", cached.dtype)).str,
                "estimated_bytes": int(loader.num_images * loader.image_size**2 * cached.dtype.itemsize),
                "cached_nbytes": int(cached.nbytes),
                "cached_shape": list(cached.shape),
                "cached_dtype": cached.dtype.str,
                "cached_c_contiguous": bool(cached.flags.c_contiguous),
                "cached_writeable": bool(cached.flags.writeable),
                "topology_before": topology_before,
                "topology_after": topology_after,
            },
            "tracemalloc": {
                "baseline_current_bytes": int(traced_baseline_current),
                "baseline_peak_bytes": int(traced_baseline_peak),
                "after_current_bytes": int(traced_after_current),
                "peak_bytes": int(traced_peak),
                "retained_delta_bytes": int(traced_after_current - traced_baseline_current),
                "peak_above_baseline_bytes": int(traced_peak - traced_baseline_current),
                "elapsed_s": elapsed_s,
            },
            "rss_diagnostic": {
                "current_before_bytes": rss_before,
                "current_after_bytes": rss_after,
                "current_delta_bytes": rss_after - rss_before,
                "high_water_before_bytes": hwm_before,
                "high_water_after_bytes": hwm_after,
                "high_water_delta_bytes": hwm_after - hwm_before,
            },
            "bitwise_equivalence": {
                "exact": exact,
                "cached_sha256": cached_digest,
                "streamed_uncached_sha256": streamed_digest.hexdigest(),
                "compared_images": compared_images,
                "batch_count": batch_count,
                "first_mismatch_index": first_mismatch_index,
                "comparison_loader_cached": getattr(comparison, "_cached", None) is not None,
                "comparison_topology_before": comparison_topology_before,
                "comparison_topology_after": comparison_topology_after,
            },
        }
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        loader.close()
        if comparison is not None:
            comparison.close()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-star", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--comparison-batch-size", type=int, default=500)
    args = parser.parse_args(argv)
    payload = probe(
        input_star=args.input_star,
        data_dir=args.data_dir,
        output_json=args.output_json,
        comparison_batch_size=args.comparison_batch_size,
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
