#!/usr/bin/env python3
"""Localize active-particle VDAM escapes against direct native run paths.

This focused diagnostic intentionally audits particle state only.  Frozen
score promotion still requires ``audit_vdam_candidate_state_envelope``, which
also seals provenance, physical-GPU identity, and the controller schedule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recovar.data_io.starfile import read_star

if __package__:
    from scripts.audit_vdam_candidate_state_envelope import audit_particle_state_checkpoints
else:
    from audit_vdam_candidate_state_envelope import audit_particle_state_checkpoints


SCHEMA = "recovar.vdam_direct_particle_state_envelope.v1"


def audit_direct_particle_state_envelope(
    *,
    candidate_dir: Path,
    native_dirs: list[Path],
    fixture_star: Path,
    iterations: tuple[int, ...],
) -> dict[str, Any]:
    fixture, _ = read_star(str(fixture_star))
    rows = audit_particle_state_checkpoints(
        candidate_dir=candidate_dir,
        native_dirs=native_dirs,
        fixture=fixture,
        iterations=iterations,
    )
    failures = [row for row in rows if not row["pass"]]
    return {
        "schema": SCHEMA,
        "result": "fail" if failures else "pass",
        "scope": "diagnostic active-particle state coverage; not a frozen-suite promotion",
        "candidate_dir": str(candidate_dir),
        "native_dirs": [str(path) for path in native_dirs],
        "fixture_star": str(fixture_star),
        "native_repeat_count": len(native_dirs),
        "first_failure_iteration": None if not failures else failures[0]["iteration"],
        "failure_count": len(failures),
        "checkpoints": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, action="append", required=True)
    parser.add_argument("--fixture-star", type=Path, required=True)
    parser.add_argument("--iterations", type=int, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_direct_particle_state_envelope(
        candidate_dir=args.candidate_dir.resolve(),
        native_dirs=[path.resolve() for path in args.native_dir],
        fixture_star=args.fixture_star.resolve(),
        iterations=tuple(args.iterations),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
