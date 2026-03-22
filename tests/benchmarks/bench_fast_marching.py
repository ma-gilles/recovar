from __future__ import annotations

import argparse
import json
import time

import numpy as np

from recovar.heterogeneity import fast_marching


def _benchmark_case(shape, order, anisotropic=False):
    rng = np.random.default_rng(sum(shape) + order)
    speed = 0.5 + rng.random(shape)
    start = tuple(axis // 2 for axis in shape)
    dx = np.linspace(0.4, 1.6, num=len(shape), dtype=np.float64) if anisotropic else np.ones(len(shape), dtype=np.float64)

    phi = np.ones(shape, dtype=np.float64)
    phi[start] = -1.0

    timings = {}

    t0 = time.perf_counter()
    python_out = fast_marching._python_travel_time(phi, speed, dx, order)
    timings["python_seconds"] = time.perf_counter() - t0

    if fast_marching.native_available():
        t0 = time.perf_counter()
        native_out = fast_marching._native_travel_time(phi, speed, dx, order)
        timings["native_seconds"] = time.perf_counter() - t0
        timings["native_max_abs_error"] = float(np.max(np.abs(native_out - python_out)))

    try:
        import skfmm
    except ImportError:
        skfmm = None

    if skfmm is not None:
        t0 = time.perf_counter()
        skfmm_out = np.asarray(skfmm.travel_time(phi, speed=speed, dx=dx, order=order), dtype=np.float64)
        timings["skfmm_seconds"] = time.perf_counter() - t0
        timings["skfmm_max_abs_error"] = float(np.max(np.abs(skfmm_out - python_out)))

    return timings


def main():
    parser = argparse.ArgumentParser(description="Benchmark RECOVAR fast marching backends")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    cases = {
        "2d_64_order2": _benchmark_case((64, 64), order=2),
        "2d_96_anisotropic_order2": _benchmark_case((96, 96), order=2, anisotropic=True),
        "3d_24_order2": _benchmark_case((24, 24, 24), order=2),
        "4d_8_order2": _benchmark_case((8, 8, 8, 8), order=2),
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
