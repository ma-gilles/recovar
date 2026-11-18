#!/usr/bin/env python3
"""Simulate a filament-attachment cryo-ET dataset (RELION 5 2D stacks + ground truth) from a ladder preset.

Presets (``recovar.simulation.filament_attachments.stage_config``):

- ``stage0``: asymmetric landmark only, no core, no noise (sign/axis/centre checks);
- ``stage1``: core + one rigid attachment per band (eta = 1, q = 0);
- ``stage2``: sparse occupancy, ``--eta`` (grid 0.05, 0.10, 0.30; default 0.10);
- ``stage3``: q = 2 known modes.

``--geometry toy`` is 96 px at 8 A/px with 40 A spacing and a 240 A band; ``dogic`` is 128 px at
4.8515625 A/px with 42.2 A spacing and a 2 x 42.2 A band. Any other field of
``FilamentAttachmentConfig`` is set with ``--set name=value`` (JSON values, e.g.
``--set core_radii=[115,125,135]``). ``--dry-run`` builds the scene and prints the counts and
the rendering box without simulating images.

Run with this checkout on ``PYTHONPATH`` (``python scripts/...`` does not add the repo root)::

    PYTHONPATH=$PWD python scripts/simulate_filament_attachments.py --preset stage1 -o OUT --n-filaments 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time


def _parse_set(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects name=value, got {item!r}")
        name, value = item.split("=", 1)
        try:
            out[name] = json.loads(value)
        except json.JSONDecodeError:
            out[name] = value
    return out


def build_config(args):
    from recovar.simulation import filament_attachments as fa

    overrides = {}
    for name in ("n_filaments", "windows_per_filament", "n_tomograms", "eta", "q", "seed"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if args.snr is not None:
        overrides["snr"] = None if args.snr.lower() in ("none", "inf") else float(args.snr)
    overrides.update(_parse_set(args.set))
    fields = {f.name for f in dataclasses.fields(fa.FilamentAttachmentConfig)}
    unknown = sorted(set(overrides) - fields)
    if unknown:
        raise SystemExit(f"unknown config fields: {unknown}")
    for name, value in list(overrides.items()):
        if isinstance(value, list):
            overrides[name] = tuple(tuple(v) if isinstance(v, list) else v for v in value)
    return fa.stage_config(args.preset, geometry=args.geometry, **overrides)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", required=True, choices=["stage0", "stage1", "stage2", "stage3"])
    parser.add_argument("--geometry", default="toy", choices=["toy", "dogic"])
    parser.add_argument("-o", "--output", required=True, help="Output folder (RELION project + truth/)")
    parser.add_argument("--n-filaments", dest="n_filaments", type=int)
    parser.add_argument("--windows-per-filament", dest="windows_per_filament", type=int)
    parser.add_argument("--n-tomograms", dest="n_tomograms", type=int)
    parser.add_argument("--eta", type=float, help="Fraction of windows whose band is occupied")
    parser.add_argument("--q", type=int, choices=[0, 1, 2])
    parser.add_argument("--snr", help="Image SNR (total signal over noise); 'none' for noise-free")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="Build the scene only and print its summary")
    parser.add_argument("--overwrite", action="store_true", help="Allow a non-empty output folder")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from recovar.simulation import filament_attachments as fa

    config = build_config(args)
    if args.dry_run:
        scene = fa.build_scene(config)
        windows = scene["windows"]
        summary = {
            "render_grid_size": scene["render_grid_size"],
            "filaments": len(scene["filaments"]),
            "windows": len(windows),
            "attachments": len(scene["attachments"]),
            "occupied_fraction": float(windows["band_occupied"].mean()),
            "max_rendered_attachments_per_window": int(windows["n_rendered_attachments"].max()),
            "tilts": int(round(2 * config.max_tilt / config.tilt_step)) + 1,
            "config": config.to_json_dict(),
        }
        print(json.dumps(summary, indent=2))
        return 0
    if os.path.isdir(args.output) and os.listdir(args.output) and not args.overwrite:
        raise SystemExit(f"{args.output} is not empty (pass --overwrite)")
    start = time.time()
    result = fa.generate_filament_attachment_dataset(args.output, config)
    record = result["record"]
    logging.info(
        "Done in %.0f s: %s windows, occupied fraction %.3f, render box %d px, git %s%s",
        time.time() - start,
        record["counts"]["windows"],
        record["counts"]["occupied_fraction"],
        record["render_grid_size"],
        record["recovar_git_sha"],
        " (dirty)" if record["recovar_git_dirty"] else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
