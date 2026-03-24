from __future__ import annotations

import argparse
import json
import time

import numpy as np

from recovar.heterogeneity import fast_marching


def _median_runtime(fn, repeat):
    timings = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - t0)
    return float(np.median(timings)), result


def _benchmark_case(shape, order, repeat, anisotropic=False):
    rng = np.random.default_rng(sum(shape) + order)
    speed = 0.5 + rng.random(shape)
    start = tuple(axis // 2 for axis in shape)
    dx = np.linspace(0.4, 1.6, num=len(shape), dtype=np.float64) if anisotropic else np.ones(len(shape), dtype=np.float64)

    phi = np.ones(shape, dtype=np.float64)
    phi[start] = -1.0

    timings = {}

    timings["python_seconds"], python_out = _median_runtime(
        lambda: fast_marching._python_travel_time(phi, speed, dx, order),
        repeat,
    )

    native_out = None
    if fast_marching.native_available():
        timings["native_seconds"], native_out = _median_runtime(
            lambda: fast_marching._native_travel_time(phi, speed, dx, order),
            repeat,
        )
        timings["native_max_abs_error"] = float(np.max(np.abs(native_out - python_out)))
        timings["native_over_python_ratio"] = timings["native_seconds"] / timings["python_seconds"]
        timings["native_speedup_vs_python"] = timings["python_seconds"] / timings["native_seconds"]

    try:
        import skfmm
    except ImportError:
        skfmm = None

    if skfmm is not None:
        timings["skfmm_seconds"], skfmm_out = _median_runtime(
            lambda: np.asarray(skfmm.travel_time(phi, speed=speed, dx=dx, order=order), dtype=np.float64),
            repeat,
        )
        timings["skfmm_max_abs_error"] = float(np.max(np.abs(skfmm_out - python_out)))
        timings["skfmm_over_python_ratio"] = timings["skfmm_seconds"] / timings["python_seconds"]
        timings["skfmm_speedup_vs_python"] = timings["python_seconds"] / timings["skfmm_seconds"]
        if native_out is not None:
            timings["native_over_skfmm_ratio"] = timings["native_seconds"] / timings["skfmm_seconds"]
            timings["native_vs_skfmm_max_abs_error"] = float(np.max(np.abs(native_out - skfmm_out)))

    return timings


def main():
    parser = argparse.ArgumentParser(description="Benchmark RECOVAR fast marching backends")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--repeat", type=int, default=5, help="number of timing repeats per backend")
    args = parser.parse_args()

    cases = {
        "2d_64_order2": _benchmark_case((64, 64), order=2, repeat=args.repeat),
        "2d_96_anisotropic_order2": _benchmark_case((96, 96), order=2, repeat=args.repeat, anisotropic=True),
        "3d_24_order2": _benchmark_case((24, 24, 24), order=2, repeat=args.repeat),
        "4d_8_order2": _benchmark_case((8, 8, 8, 8), order=2, repeat=args.repeat),
    }

    if args.json:
        print(json.dumps(cases, indent=2, sort_keys=True))
        return

    for name, metrics in cases.items():
        print(name)
        for key, value in metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
