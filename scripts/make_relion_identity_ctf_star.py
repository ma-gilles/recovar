#!/usr/bin/env python
"""Write a RELION-only identity-CTF STAR for simulator no-CTF diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import starfile


RELION_IDENTITY_PHASE_SHIFT_DEG = 180.0


def _set_existing_column(df: Any, names: list[str], value: float) -> str | None:
    for name in names:
        if name in df.columns:
            df[name] = value
            return name
    return None


def _set_required_column(df: Any, names: list[str], value: float) -> str:
    column = _set_existing_column(df, names, value)
    if column is not None:
        return column
    column = names[0]
    df[column] = value
    return column


def write_identity_ctf_star(
    input_star: Path,
    output_star: Path,
    *,
    manifest_path: Path | None = None,
    phase_shift_deg: float = 0.0,
) -> None:
    """Create a RELION-compatible STAR whose CTF evaluates as identity.

    RECOVAR's simulator represents no-CTF SPA images with the internal sentinel
    ``W=-1``. RELION rejects that amplitude contrast under ``--ctf``, so this
    diagnostic STAR rewrites only RELION metadata and leaves the original
    RECOVAR input files unchanged.
    """
    tables_raw = starfile.read(input_star)
    if not isinstance(tables_raw, dict):
        raise RuntimeError(f"Expected RELION 3.1 STAR with optics/particles tables: {input_star}")
    tables = {name: table.copy() for name, table in tables_raw.items()}
    optics = tables.get("optics")
    particles = tables.get("particles")
    if optics is None or particles is None:
        raise RuntimeError(f"Expected optics and particles tables in {input_star}; got {sorted(tables)}")

    # RELION rejects a zero-defocus, zero-Cs, zero-amplitude CTF as all-zero.
    # Pure amplitude contrast plus a 180 degree phase shift gives a constant
    # +1 transfer function, matching RECOVAR's no-CTF sentinel W=-1.
    _set_required_column(optics, ["rlnAmplitudeContrast", "_rlnAmplitudeContrast"], 1.0)
    _set_existing_column(optics, ["rlnSphericalAberration", "_rlnSphericalAberration"], 0.0)
    _set_existing_column(optics, ["rlnVoltage", "_rlnVoltage"], 300.0)
    _set_required_column(particles, ["rlnDefocusU", "_rlnDefocusU"], 0.0)
    _set_required_column(particles, ["rlnDefocusV", "_rlnDefocusV"], 0.0)
    _set_required_column(particles, ["rlnDefocusAngle", "_rlnDefocusAngle"], 0.0)
    _set_required_column(particles, ["rlnPhaseShift", "_rlnPhaseShift"], float(phase_shift_deg))
    _set_existing_column(particles, ["rlnCtfBfactor", "_rlnCtfBfactor"], 0.0)
    _set_existing_column(particles, ["rlnCtfScalefactor", "_rlnCtfScalefactor"], 1.0)

    output_star.parent.mkdir(parents=True, exist_ok=True)
    starfile.write(tables, output_star, overwrite=True)

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "source_star": str(input_star),
                    "output_star": str(output_star),
                    "purpose": "RELION-only identity CTF metadata for simulator no-CTF diagnostic",
                    "recovar_raw_noctf_sentinel": {"rlnAmplitudeContrast": -1.0},
                    "relion_identity_ctf": {
                        "rlnAmplitudeContrast": 1.0,
                        "rlnDefocusU": 0.0,
                        "rlnDefocusV": 0.0,
                        "rlnDefocusAngle": 0.0,
                        "rlnPhaseShift": float(phase_shift_deg),
                    },
                    "raw_star_left_unchanged": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-star", required=True, type=Path)
    parser.add_argument("--output-star", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--phase-shift-deg", type=float, default=RELION_IDENTITY_PHASE_SHIFT_DEG)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    write_identity_ctf_star(
        args.input_star,
        args.output_star,
        manifest_path=args.manifest,
        phase_shift_deg=args.phase_shift_deg,
    )
    print(f"wrote {args.output_star} for RELION no-CTF --ctf diagnostic")


if __name__ == "__main__":
    main()
