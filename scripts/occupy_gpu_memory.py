#!/usr/bin/env python3
"""Allocate a configurable amount of GPU memory and sleep.

Used in the manual conflict-process integration test:

    python scripts/occupy_gpu_memory.py --gb 30 --seconds 600

Then in another shell:

    recovar run_test_dataset --gpu-budget-gb 70 --memory-diagnostics --no-delete

The wrapper should detect the conflict via the new gpu_preflight probe
and surface a "GPU appears to be partially occupied" hint instead of a
raw XLA OOM stack trace.
"""

from __future__ import annotations

import argparse
import os
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gb", type=float, default=8.0)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

    import jax
    import jax.numpy as jnp

    devices = jax.devices("gpu")
    if not devices:
        raise SystemExit("No GPU device available")
    dev = devices[args.device]

    n_floats = int(args.gb * 1e9 / 4)  # float32
    print(f"Allocating ~{args.gb:.1f} GB on {dev}...", flush=True)
    block = jax.device_put(jnp.zeros((n_floats,), dtype=jnp.float32), dev)
    block.block_until_ready()
    print(f"Holding allocation for {args.seconds} s...", flush=True)
    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        del block
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
