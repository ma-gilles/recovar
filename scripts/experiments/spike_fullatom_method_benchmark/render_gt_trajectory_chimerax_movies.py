#!/usr/bin/env python3
"""Render GT spike trajectory movies with a saved ChimeraX camera.

The highlight movie does not render the mask surface.  It uses the moving mask
only to split each GT volume into two density maps:

* inside mask: full opacity
* outside mask: low opacity

Run with normal Python.  The script invokes ChimeraX for rendering.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_GT_DIR = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise3_b80_20260531/"
    "n00300000/runs/n00300000_seed0000/04_ground_truth"
)
DEFAULT_MOVING_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise3_b80_20260531/"
    "n00300000/runs/n00300000_seed0000/05_masks/focus_mask_moving.mrc"
)
DEFAULT_VIEW_CXS = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "chimerax_state50_method_progression_moving_mask_zoomed_view_20260601/"
    "animations/full_view.cxs"
)
DEFAULT_VIEW_JSON = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "chimerax_state50_method_progression_moving_mask_zoomed_view_20260601/"
    "zoomed_moving_view_extracted.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "gt_trajectory_chimerax_movies_full_view_20260603"
)


def _quote(path: str | Path) -> str:
    return '"' + str(path).replace('"', '\\"') + '"'


def _load_view(path: Path) -> dict[str, Any]:
    view = json.loads(path.read_text())
    if not view.get("camera_matrix"):
        raise RuntimeError(f"view JSON has no camera_matrix: {path}")
    return view


def _load_mask(path: Path) -> Any:
    import mrcfile
    import numpy as np

    with mrcfile.open(path, permissive=True) as handle:
        mask = np.asarray(handle.data, dtype=np.float32).copy()
    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(mask, 0.0, 1.0)


def _write_like(src_path: Path, dst_path: Path, data: Any) -> None:
    import mrcfile

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.open(src_path, permissive=True) as src:
        with mrcfile.new(dst_path, overwrite=True) as dst:
            dst.set_data(data)
            dst.voxel_size = src.voxel_size
            for field in ("nxstart", "nystart", "nzstart", "mapc", "mapr", "maps"):
                try:
                    setattr(dst.header, field, getattr(src.header, field))
                except Exception:
                    pass
            try:
                dst.header.origin = src.header.origin
            except Exception:
                pass
            dst.update_header_from_data()
            dst.update_header_stats()


def _prepare_split_volumes(args: argparse.Namespace, states: list[int]) -> list[dict[str, str | int]]:
    import mrcfile
    import numpy as np

    mask = _load_mask(args.moving_mask)
    split_dir = args.output_dir / "split_volumes"
    frames: list[dict[str, str | int]] = []
    for state in states:
        src = args.gt_dir / f"gt_vol{state:04d}.mrc"
        if not src.exists():
            raise FileNotFoundError(src)
        inside = split_dir / "inside_moving_mask" / f"gt_vol{state:04d}_inside_moving_mask.mrc"
        outside = split_dir / "outside_moving_mask" / f"gt_vol{state:04d}_outside_moving_mask.mrc"
        if args.overwrite_split or not inside.exists() or not outside.exists():
            with mrcfile.open(src, permissive=True) as handle:
                vol = np.asarray(handle.data, dtype=np.float32)
                if vol.shape != mask.shape:
                    raise RuntimeError(
                        f"mask shape {mask.shape} does not match {src} shape {vol.shape}"
                    )
                inside_data = np.asarray(vol * mask, dtype=np.float32)
                outside_data = np.asarray(vol * (1.0 - mask), dtype=np.float32)
            _write_like(src, inside, inside_data)
            _write_like(src, outside, outside_data)
        frames.append(
            {
                "state": state,
                "whole_volume": str(src),
                "inside_volume": str(inside),
                "outside_volume": str(outside),
                "whole_png": str(args.output_dir / "frames_raw" / "whole" / f"state_{state:04d}.png"),
                "highlight_png": str(
                    args.output_dir / "frames_raw" / "moving_mask_highlight" / f"state_{state:04d}.png"
                ),
            }
        )
    return frames


def _write_plan(args: argparse.Namespace, frames: list[dict[str, str | int]], view: dict[str, Any]) -> Path:
    width = args.width
    height = args.height
    if args.use_view_size and view.get("window_size"):
        width, height = [int(x) for x in view["window_size"]]
    plan = {
        "source": {
            "gt_dir": str(args.gt_dir),
            "moving_mask": str(args.moving_mask),
            "view_cxs": str(args.view_cxs),
            "view_json": str(args.view_json),
        },
        "render": {
            "camera_matrix": view["camera_matrix"],
            "field_of_view": view.get("field_of_view"),
            "width": width,
            "height": height,
            "supersample": args.supersample,
            "background": args.background,
            "contour_level": args.contour_level,
            "whole_color": args.whole_color,
            "inside_color": args.inside_color,
            "outside_color": args.outside_color,
            "outside_opacity": args.outside_opacity,
            "silhouettes": args.silhouettes,
        },
        "movie": {"fps": args.fps, "duration_seconds": len(frames) / args.fps},
        "frames": frames,
    }
    out = args.output_dir / "render_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return out


def _chimerax_style_volume(run: Any, model: str, color: str, level: float, opacity: float) -> None:
    run(session, f"volume {model} style surface")
    run(session, f"volume {model} level {level:.8g}")
    run(session, f"color {model} {color}")
    opacity = max(0.0, min(1.0, opacity))
    if opacity < 1.0:
        run(session, f"transparency {model} {(1.0 - opacity) * 100.0:.6g} surfaces")


def _chimerax_apply_scene(run: Any, render: dict[str, Any]) -> None:
    run(session, f"set bgColor {render['background']}")
    run(session, "lighting soft")
    run(session, f"graphics silhouettes {'true' if render['silhouettes'] else 'false'}")


def _chimerax_apply_view(run: Any, render: dict[str, Any]) -> None:
    run(session, f"view matrix camera {render['camera_matrix']}")


def _chimerax_save(run: Any, out_path: Path, render: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        session,
        f"save {_quote(out_path)} width {int(render['width'])} height {int(render['height'])} "
        f"supersample {int(render['supersample'])}",
    )


def _internal_render(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args(argv)

    from chimerax.core.commands import run

    plan = json.loads(args.plan.read_text())
    render = plan["render"]
    level = float(render["contour_level"])
    for frame in plan["frames"]:
        state = int(frame["state"])

        run(session, "close")
        run(session, f"open {_quote(frame['whole_volume'])}")
        _chimerax_apply_scene(run, render)
        _chimerax_style_volume(run, "#1", render["whole_color"], level, 1.0)
        _chimerax_apply_view(run, render)
        _chimerax_save(run, Path(frame["whole_png"]), render)
        print(f"rendered whole state {state:04d}", flush=True)

        run(session, "close")
        run(session, f"open {_quote(frame['outside_volume'])}")
        run(session, f"open {_quote(frame['inside_volume'])}")
        _chimerax_apply_scene(run, render)
        _chimerax_style_volume(
            run, "#1", render["outside_color"], level, float(render["outside_opacity"])
        )
        _chimerax_style_volume(run, "#2", render["inside_color"], level, 1.0)
        _chimerax_apply_view(run, render)
        _chimerax_save(run, Path(frame["highlight_png"]), render)
        print(f"rendered highlight state {state:04d}", flush=True)

    run(session, "exit")


def _run_chimerax(args: argparse.Namespace, plan: Path) -> Path:
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = f"{Path(__file__).resolve()} --internal-render --plan {plan}"
    command = (
        f"module purge; module load {shlex.quote(args.chimerax_module)}; "
        f"chimerax --nogui --offscreen --script {shlex.quote(payload)}"
    )
    log_path = log_dir / "chimerax_render_gt_trajectory.log"
    with log_path.open("w") as log:
        log.write(f"COMMAND={command}\n")
        log.flush()
        proc = subprocess.run(["bash", "-lc", command], stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"ChimeraX render failed with code {proc.returncode}; see {log_path}")
    return log_path


def _run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write("COMMAND=" + shlex.join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"RETURNCODE={proc.returncode}\n")
        return proc.returncode


def _assemble_movie(args: argparse.Namespace, frames_dir: Path, stem: str) -> dict[str, str | int]:
    movie_dir = args.output_dir / "movies"
    movie_dir.mkdir(parents=True, exist_ok=True)
    palette = movie_dir / f"{stem}_palette.png"
    gif = movie_dir / f"{stem}.gif"
    mp4 = movie_dir / f"{stem}.mp4"
    log = args.output_dir / "logs" / f"{stem}_ffmpeg.log"
    pattern = frames_dir / "state_%04d.png"
    rc1 = _run_logged(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{args.fps:g}",
            "-i",
            str(pattern),
            "-vf",
            "palettegen",
            str(palette),
        ],
        log,
    )
    rc2 = _run_logged(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{args.fps:g}",
            "-i",
            str(pattern),
            "-i",
            str(palette),
            "-lavfi",
            "paletteuse",
            "-loop",
            "0",
            str(gif),
        ],
        log,
    )
    rc3 = _run_logged(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{args.fps:g}",
            "-i",
            str(pattern),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ],
        log,
    )
    if rc1 or rc2 or rc3:
        raise RuntimeError(f"ffmpeg failed for {stem}; see {log}")
    return {
        "gif": str(gif),
        "mp4": str(mp4),
        "palette": str(palette),
        "log": str(log),
        "returncodes": [rc1, rc2, rc3],
    }


def _write_audit(
    args: argparse.Namespace,
    plan: Path,
    render_log: Path,
    movies: dict[str, dict[str, str | int]],
) -> Path:
    audit = {
        "script": str(Path(__file__).resolve()),
        "output_dir": str(args.output_dir),
        "plan": str(plan),
        "chimerax_log": str(render_log),
        "movies": movies,
        "states": [args.state_start, args.state_end],
        "notes": [
            "moving mask is not rendered as a surface",
            "highlight movie uses inside=volume*mask and outside=volume*(1-mask)",
        ],
    }
    out = args.output_dir / "render_gt_trajectory_audit.json"
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return out


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-render", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--moving-mask", type=Path, default=DEFAULT_MOVING_MASK)
    parser.add_argument("--view-cxs", type=Path, default=DEFAULT_VIEW_CXS)
    parser.add_argument("--view-json", type=Path, default=DEFAULT_VIEW_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-start", type=int, default=0)
    parser.add_argument("--state-end", type=int, default=99)
    parser.add_argument("--contour-level", type=float, default=0.013)
    parser.add_argument("--whole-color", default="#6f8fd6")
    parser.add_argument("--inside-color", default="#f28e2b")
    parser.add_argument("--outside-color", default="#909090")
    parser.add_argument("--outside-opacity", type=float, default=0.3)
    parser.add_argument("--background", default="white")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--use-view-size", action="store_true")
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--chimerax-module", default="chimerax/1.9")
    parser.add_argument("--silhouettes", action="store_true")
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--skip-chimerax", action="store_true")
    parser.add_argument("--skip-movies", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = _parse_args(argv)
    if args.internal_render:
        if args.plan is None:
            raise SystemExit("--internal-render requires --plan")
        _internal_render(["--plan", str(args.plan)])
        return

    args.gt_dir = args.gt_dir.resolve()
    args.moving_mask = args.moving_mask.resolve()
    args.view_cxs = args.view_cxs.resolve()
    args.view_json = args.view_json.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.gt_dir, args.moving_mask, args.view_json):
        if not path.exists():
            raise FileNotFoundError(path)

    states = list(range(args.state_start, args.state_end + 1))
    view = _load_view(args.view_json)
    frames = _prepare_split_volumes(args, states)
    plan = _write_plan(args, frames, view)
    render_log = args.output_dir / "logs" / "chimerax_render_gt_trajectory.log"
    if not args.skip_chimerax:
        render_log = _run_chimerax(args, plan)

    movies: dict[str, dict[str, str | int]] = {}
    if not args.skip_movies:
        movies["whole"] = _assemble_movie(args, args.output_dir / "frames_raw" / "whole", "gt_state_trajectory_whole")
        movies["moving_mask_highlight"] = _assemble_movie(
            args,
            args.output_dir / "frames_raw" / "moving_mask_highlight",
            "gt_state_trajectory_moving_mask_highlight",
        )
    audit = _write_audit(args, plan, render_log, movies)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "plan": str(plan),
                "audit": str(audit),
                "render_log": str(render_log),
                "movies": movies,
            },
            indent=2,
            sort_keys=True,
        )
    )


main(sys.argv[1:])
