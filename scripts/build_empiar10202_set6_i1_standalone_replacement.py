#!/usr/bin/env python3
"""Build a sealed standalone RECOVAR-full replacement record after completion.

The builder never submits or modifies a job/run root.  It refuses to overwrite
its output, and the audit adapter requires every terminal artifact before this
payload can be built successfully.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

ADAPTER_PATH = Path(__file__).resolve().with_name("audit_empiar10202_set6_i1_native_harness.py")


def _load_adapter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_empiar10202_native_adapter_for_builder", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load audit adapter: {ADAPTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-launch-manifest", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--particle-stack", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.output.is_absolute():
        raise ValueError("--output must be absolute")
    if not args.output.parent.is_dir():
        raise ValueError(f"--output parent does not exist: {args.output.parent}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite replacement record: {args.output}")
    adapter = _load_adapter()
    payload = adapter.build_standalone_replacement_payload(
        args.parent_launch_manifest,
        profile_name=args.profile,
        data_dir=args.data_dir,
        particle_stack=args.particle_stack,
        reason=args.reason,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"replacement_record": str(args.output), "sha256": adapter.sha256_file(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
