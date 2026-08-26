#!/usr/bin/env python3
"""Audit the sealed K=1 controller prefix without requiring terminal results.

RECOVAR ``itNNN_meta.npy`` is paired with RELION physical iteration ``NNN+1``
by default.  This audit is deliberately narrower than the terminal intermediate
audit: it checks only controller topology that is already sealed on disk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from scripts.audit_k1_intermediate_trajectory import _general, _model, _sampling_general

SCHEMA = "em_k1_live_controller_prefix_audit_v1"
META_PATTERN = re.compile(r"it([0-9]{3})_meta[.]npy")


class AuditError(RuntimeError):
    """Raised when the sealed prefix is absent or malformed."""


def _read_relion_controller(relion_dir: Path, iteration: int) -> tuple[int, int]:
    prefix = relion_dir / f"run_it{iteration:03d}"
    general = _general(_model(Path(f"{prefix}_half1_model.star")))
    sampling = _sampling_general(Path(f"{prefix}_sampling.star"))
    return int(general["rlnCurrentImageSize"]), int(sampling["rlnHealpixOrder"])


def _sealed_meta_paths(recovar_dir: Path) -> list[tuple[int, Path]]:
    intermediate_dir = recovar_dir / "intermediates"
    if not intermediate_dir.is_dir():
        raise AuditError(f"missing {intermediate_dir}")
    indexed = []
    for path in intermediate_dir.glob("it*_meta.npy"):
        match = META_PATTERN.fullmatch(path.name)
        if match is not None:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    if not indexed:
        raise AuditError(f"no sealed iteration metadata in {intermediate_dir}")
    indices = [index for index, _ in indexed]
    expected = list(range(len(indices)))
    if indices != expected:
        raise AuditError(f"RECOVAR metadata indices are not contiguous: found {indices}, expected {expected}")
    return indexed


def audit(recovar_dir: Path, relion_dir: Path, *, relion_start_iteration: int = 1) -> dict:
    recovar_dir = recovar_dir.resolve()
    relion_dir = relion_dir.resolve()
    rows = []
    failures = []
    for index, path in _sealed_meta_paths(recovar_dir):
        try:
            meta = np.load(path, allow_pickle=True).item()
        except (OSError, ValueError) as exc:
            raise AuditError(f"failed to load {path}: {exc}") from exc
        if not isinstance(meta, dict):
            raise AuditError(f"expected dictionary metadata in {path}")
        required = {"iteration", "current_size", "healpix_order"}
        missing = sorted(required.difference(meta))
        if missing:
            raise AuditError(f"missing metadata fields {missing} in {path}")

        relion_iteration = relion_start_iteration + index
        relion_size, relion_healpix = _read_relion_controller(relion_dir, relion_iteration)
        recovar_iteration = int(meta["iteration"])
        recovar_size = int(meta["current_size"])
        recovar_healpix = int(meta["healpix_order"])
        exact = {
            "iteration": recovar_iteration == index,
            "current_size": recovar_size == relion_size,
            "healpix_order": recovar_healpix == relion_healpix,
        }
        for field, matches in exact.items():
            if not matches:
                failures.append(f"it{relion_iteration:03d} {field} mismatch")
        rows.append(
            {
                "recovar_index": index,
                "relion_iteration": relion_iteration,
                "meta_path": str(path.resolve()),
                "recovar_iteration_field": recovar_iteration,
                "current_size": {
                    "recovar": recovar_size,
                    "relion": relion_size,
                    "exact_equal": exact["current_size"],
                },
                "healpix_order": {
                    "recovar": recovar_healpix,
                    "relion": relion_healpix,
                    "exact_equal": exact["healpix_order"],
                },
                "iteration_field_exact": exact["iteration"],
                "local_search": bool(meta.get("local_search", False)),
                "n_rotations": None if "n_rotations" not in meta else int(meta["n_rotations"]),
                "n_translations": None
                if "n_translations" not in meta
                else int(meta["n_translations"]),
            }
        )

    return {
        "schema": SCHEMA,
        "status": "pass" if not failures else "fail",
        "completion_claim": False,
        "scope": "sealed numbered controller prefix only; terminal trajectory is not claimed",
        "recovar_dir": str(recovar_dir),
        "relion_dir": str(relion_dir),
        "relion_start_iteration": relion_start_iteration,
        "sealed_iteration_count": len(rows),
        "all_controller_topology_exact": not failures,
        "failures": failures,
        "earliest_failure": failures[0] if failures else None,
        "iterations": rows,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# K=1 live controller-prefix audit",
        "",
        f"Status: `{report['status']}`.",
        "",
        "This covers sealed numbered metadata only and does not claim terminal completion.",
        "",
        "| RELION iteration | Current size exact | HEALPix exact | Local search |",
        "|---:|:---:|:---:|:---:|",
    ]
    for row in report.get("iterations", []):
        lines.append(
            f"| {row['relion_iteration']} | {row['current_size']['exact_equal']} | "
            f"{row['healpix_order']['exact_equal']} | {row['local_search']} |"
        )
    if report.get("failures"):
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in report["failures"])
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-dir", type=Path, required=True)
    parser.add_argument("--relion-dir", type=Path, required=True)
    parser.add_argument("--relion-start-iteration", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for path in (args.output_json, args.output_markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = audit(
            args.recovar_dir,
            args.relion_dir,
            relion_start_iteration=args.relion_start_iteration,
        )
    except AuditError as exc:
        report = {
            "schema": SCHEMA,
            "status": "error",
            "completion_claim": False,
            "scope": "sealed numbered controller prefix only; terminal trajectory is not claimed",
            "failures": [str(exc)],
            "earliest_failure": str(exc),
            "iterations": [],
        }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(render_markdown(report))
    print(json.dumps({"status": report["status"], "output": str(args.output_json.resolve())}))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
