#!/usr/bin/env python3
"""Aggregate per-cell calibration JSONs into the planner's table.

Reads every ``cell_*.json`` produced by ``calibrate_memory_planner.py``
and emits ``recovar/utils/memory_calibration_data.json`` in the schema
the planner expects.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells-dir",
        type=Path,
        default=Path("/scratch/gpfs/GILLES/mg6942/calibration_runs/cells"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/scratch/gpfs/GILLES/mg6942/recovar_wt_agent_20260507_170447_778531/recovar/utils/memory_calibration_data.json"
        ),
    )
    parser.add_argument("--gpu-kind", default="NVIDIA H100")
    parser.add_argument("--driver", default="auto")
    parser.add_argument("--jax-version", default="auto")
    args = parser.parse_args()

    if args.driver == "auto":
        args.driver = (
            os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null").read().strip()
            or "unknown"
        )
    if args.jax_version == "auto":
        try:
            import jax

            args.jax_version = jax.__version__
        except Exception:
            args.jax_version = "unknown"

    tables: dict[str, list[dict]] = {"pipeline": [], "compute_state": []}
    for path in sorted(args.cells_dir.glob("*.json")):
        with path.open() as fh:
            cell = json.load(fh)
        if cell.get("status") not in ("ok", "oom"):
            continue
        cmd = cell.get("command")
        if cmd not in tables:
            continue
        tables[cmd].append(
            {
                "grid_size": cell["grid_size"],
                "backend": cell["backend"],
                "n_pcs": cell["n_pcs"],
                "peak_gb_total": cell.get("peak_gb_total") or 0.0,
                "peak_gb_by_phase": cell.get("peak_gb_by_phase") or {},
                "status": cell["status"],
            }
        )

    payload = {
        "schema_version": 1,
        "calibrated_on": {
            "gpu_kind": args.gpu_kind,
            "driver": args.driver,
            "jax": args.jax_version,
        },
        "tables": tables,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(
        f"[aggregate] wrote {args.out}: "
        f"pipeline cells={len(tables['pipeline'])}, "
        f"compute_state cells={len(tables['compute_state'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
