#!/usr/bin/env python3
"""Render MRC maps from a CSV manifest inside ChimeraX.

Run with ChimeraX, not normal Python:

    chimerax --nogui --offscreen --script "render_chimerax_manifest.py --manifest manifest.csv --output-dir renders"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from chimerax.core.commands import run


def _parse_csv_ints(value: str | None) -> set[int] | None:
    if value is None or value == "":
        return None
    return {int(x) for x in value.split(",") if x.strip()}


def _parse_csv_strings(value: str | None) -> set[str] | None:
    if value is None or value == "":
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def _quote(path: Path | str) -> str:
    return '"' + str(path).replace('"', '\\"') + '"'


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _filtered_rows(args: argparse.Namespace, all_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    noise_levels = _parse_csv_ints(args.noise_levels)
    n_images = _parse_csv_ints(args.n_images)
    states = _parse_csv_ints(args.states)
    methods = _parse_csv_strings(args.methods)
    roles = _parse_csv_strings(args.roles)

    rows: list[dict[str, str]] = []
    for row in all_rows:
        if noise_levels is not None and int(row["noise_level"]) not in noise_levels:
            continue
        if n_images is not None and int(row["n_images"]) not in n_images:
            continue
        if states is not None and int(row["state"]) not in states:
            continue
        if methods is not None and row["method"] not in methods:
            continue
        if roles is not None and row["role"] not in roles:
            continue
        if args.overlay_gt and (row["method"] == "ground_truth" or row["role"] != "estimate"):
            continue
        rows.append(row)
    return rows


def _gt_lookup_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("noise_level", ""),
        row.get("bfactor", ""),
        row.get("collection", ""),
        row.get("n_images", ""),
        row.get("state", ""),
    )


def _build_gt_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("method") == "ground_truth" and row.get("role") == "gt":
            out.setdefault(_gt_lookup_key(row), row)
    return out


def _set_scene(args: argparse.Namespace) -> None:
    run(session, f"set bgColor {args.background}")
    if args.orthographic:
        run(session, "camera ortho")
    run(session, "lighting soft")
    run(session, "graphics silhouettes true")


def _render_name(row: dict[str, str], overlay_gt: bool) -> str:
    name = row["render_name"]
    if not overlay_gt:
        return name
    path = Path(name)
    return f"{path.stem}_gt_overlay{path.suffix}"


def _style_volume(model_spec: str, row: dict[str, str], color: str, args: argparse.Namespace) -> None:
    level = _float_or_none(row.get("contour_level"))
    if level is None:
        level = args.fallback_level
    run(session, f"volume {model_spec} style surface")
    run(session, f"volume {model_spec} level {level:.8g}")
    run(session, f"color {model_spec} {color}")


def _apply_view(args: argparse.Namespace) -> None:
    if args.camera_matrix:
        run(session, f"view matrix camera {args.camera_matrix}")
        if args.zoom != 1.0:
            run(session, f"zoom {args.zoom}")
    else:
        run(session, "view")
        run(session, f"turn y {args.yaw}")
        run(session, f"turn x {args.pitch}")
        if args.roll:
            run(session, f"turn z {args.roll}")
        run(session, "view")
        if args.zoom != 1.0:
            run(session, f"zoom {args.zoom}")


def _render_one(row: dict[str, str], args: argparse.Namespace, gt_row: dict[str, str] | None = None) -> None:
    volume_path = Path(row["volume_path"])
    out_path = args.output_dir / _render_name(row, gt_row is not None)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run(session, "close")
    run(session, f"open {_quote(volume_path)}")
    _set_scene(args)
    color = row.get("color") or args.default_color

    _style_volume("#1", row, color, args)
    if gt_row is not None:
        run(session, f"open {_quote(Path(gt_row['volume_path']))}")
        _style_volume("#2", gt_row, args.gt_color, args)
        transparency = max(0.0, min(100.0, (1.0 - args.gt_opacity) * 100.0))
        run(session, f"transparency #2 {transparency:.6g} surfaces")
    _apply_view(args)

    save_cmd = (
        f"save {_quote(out_path)} width {args.width} height {args.height} "
        f"supersample {args.supersample}"
    )
    if args.transparent:
        save_cmd += " transparentBackground true"
    run(session, save_cmd)
    print(f"rendered {out_path}", flush=True)


def _write_render_settings(args: argparse.Namespace, n_rows: int) -> None:
    settings = {
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "n_render_rows": n_rows,
        "filters": {
            "noise_levels": args.noise_levels,
            "n_images": args.n_images,
            "states": args.states,
            "methods": args.methods,
            "roles": args.roles,
            "limit": args.limit,
        },
        "image": {
            "width": args.width,
            "height": args.height,
            "supersample": args.supersample,
            "transparent": args.transparent,
        },
        "view": {
            "camera_matrix": args.camera_matrix,
            "zoom": args.zoom,
            "yaw": args.yaw,
            "pitch": args.pitch,
            "roll": args.roll,
            "orthographic": args.orthographic,
        },
        "scene": {
            "background": args.background,
            "default_color": args.default_color,
            "fallback_level": args.fallback_level,
            "overlay_gt": args.overlay_gt,
            "gt_color": args.gt_color,
            "gt_opacity": args.gt_opacity,
        },
    }
    path = args.output_dir / "render_settings.json"
    path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--view-json",
        type=Path,
        help="View settings JSON from extract_chimerax_cxs_view.py.",
    )
    parser.add_argument(
        "--use-view-size",
        action="store_true",
        help="Use width/height saved in --view-json.",
    )
    parser.add_argument("--noise-levels")
    parser.add_argument("--n-images")
    parser.add_argument("--states")
    parser.add_argument("--methods")
    parser.add_argument("--roles", default="estimate,gt")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--supersample", type=int, default=3)
    parser.add_argument("--yaw", type=float, default=-35.0)
    parser.add_argument("--pitch", type=float, default=-18.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--zoom", type=float, default=0.92)
    parser.add_argument(
        "--camera-matrix",
        default="",
        help="Camera matrix payload from ChimeraX `view matrix camera ...`.",
    )
    parser.add_argument("--background", default="white")
    parser.add_argument("--orthographic", action="store_true")
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--overlay-gt", action="store_true")
    parser.add_argument("--gt-color", default="#777777")
    parser.add_argument("--gt-opacity", type=float, default=0.3)
    parser.add_argument("--fallback-level", type=float, default=0.02)
    parser.add_argument("--default-color", default="#6f8fd6")
    parser.add_argument("--limit", type=int, default=0, help="Render only the first N matching rows.")
    args = parser.parse_args(argv)

    if args.view_json is not None:
        view_settings = json.loads(args.view_json.read_text())
        if not args.camera_matrix:
            args.camera_matrix = view_settings["camera_matrix"]
        if args.use_view_size and view_settings.get("window_size"):
            args.width, args.height = [int(x) for x in view_settings["window_size"]]

    all_rows = _read_rows(args.manifest)
    rows = _filtered_rows(args, all_rows)
    gt_lookup = _build_gt_lookup(all_rows)
    if args.limit:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_render_settings(args, len(rows))
    print(f"rendering {len(rows)} rows from {args.manifest}", flush=True)
    for row in rows:
        gt_row = gt_lookup.get(_gt_lookup_key(row)) if args.overlay_gt else None
        if args.overlay_gt and gt_row is None:
            print(f"skipping overlay without matching GT: {row}", flush=True)
            continue
        _render_one(row, args, gt_row)
    run(session, "exit")


main(sys.argv[1:])
