#!/usr/bin/env python3
"""Render 300k spike method trajectory movies with saved ChimeraX views.

This script renders the noise=3, n=300k trajectory state maps for RECOVAR,
cryoDRGN, and 3DFlex.  It writes two movie families:

* full view: the whole reconstruction at the saved full-volume camera
* moving-mask highlight: outside the moving mask at low opacity, inside at
  full opacity, using the saved zoomed moving-piece camera

The mask is never rendered as a surface.  It is only used to split density
maps into inside/outside volumes.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_STATES = (0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99)
METHODS = ("recovar", "cryodrgn", "3dflex")
METHOD_COLORS = {
    "recovar": "#1b9e77",
    "cryodrgn": "#d95f02",
    "3dflex": "#7570b3",
}
GT_COLOR = "#6f6f6f"

DEFAULT_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531"
)
DEFAULT_TRAJECTORY_ROOT = DEFAULT_ROOT / "state_trajectory_n00300000_20260601"
DEFAULT_SOURCE_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise3_b80_20260531/"
    "n00300000/runs/n00300000_seed0000"
)
DEFAULT_CRYODRGN_DIR = (
    DEFAULT_ROOT
    / "n00300000/evaluation/cryodrgn/zdim1/decoded_volumes/labels_mean_z_epoch019"
)
DEFAULT_THREEDFLEX_DIR = Path("/projects/CRYOEM/singerlab/mg6942/CS-testres/J530/J530_series_000")
DEFAULT_FULL_VIEW_JSON = (
    DEFAULT_ROOT
    / "chimerax_state50_method_progression_moving_mask_zoomed_view_20260601/"
    "animations/full_view_extracted_20260603.json"
)
DEFAULT_MOVING_VIEW_JSON = (
    DEFAULT_ROOT
    / "chimerax_state50_method_progression_moving_mask_zoomed_view_20260601/"
    "zoomed_moving_view_extracted.json"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_TRAJECTORY_ROOT
    / "chimerax_method_trajectory_movies_full_and_moving_20260603"
)


def _quote(path: str | Path) -> str:
    return '"' + str(path).replace('"', '\\"') + '"'


def _parse_states(value: str) -> list[int]:
    if value.strip().lower() == "default":
        return list(DEFAULT_STATES)
    return [int(x) for x in value.split(",") if x.strip()]


def _load_view(path: Path) -> dict[str, Any]:
    view = json.loads(path.read_text())
    if not view.get("camera_matrix"):
        raise RuntimeError(f"view JSON has no camera_matrix: {path}")
    return view


def _method_volume(args: argparse.Namespace, method: str, state: int) -> Path:
    if method == "recovar":
        return args.recovar_compute_dir / f"state{state:04d}" / "state000.mrc"
    if method == "cryodrgn":
        return args.cryodrgn_dir / f"gt_label_{state:03d}.mrc"
    if method == "3dflex":
        if state not in args.threedflex_states:
            raise ValueError(
                f"3DFlex trajectory only has generated frames for states "
                f"{tuple(args.threedflex_states)}; got {state}"
            )
        frame_index = args.threedflex_states.index(state)
        series_name = args.threedflex_dir.name
        return args.threedflex_dir / f"{series_name}_frame_{frame_index:03d}.mrc"
    raise ValueError(method)


def _gt_volume(args: argparse.Namespace, state: int) -> Path:
    return args.source_run / "04_ground_truth" / f"gt_vol{state:04d}.mrc"


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


def _prepare_split_volumes(args: argparse.Namespace, frames: list[dict[str, Any]]) -> None:
    import mrcfile
    import numpy as np

    mask = _load_mask(args.moving_mask)
    for frame in frames:
        src = Path(frame["volume"])
        inside = Path(frame["inside_volume"])
        outside = Path(frame["outside_volume"])
        if not args.overwrite_split and inside.exists() and outside.exists():
            continue
        with mrcfile.open(src, permissive=True) as handle:
            vol = np.asarray(handle.data, dtype=np.float32)
            if vol.shape != mask.shape:
                raise RuntimeError(f"mask shape {mask.shape} does not match {src} shape {vol.shape}")
            inside_data = np.asarray(vol * mask, dtype=np.float32)
            outside_data = np.asarray(vol * (1.0 - mask), dtype=np.float32)
        _write_like(src, inside, inside_data)
        _write_like(src, outside, outside_data)


def _build_frames(args: argparse.Namespace, states: list[int]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for frame_index, state in enumerate(states):
        for method in METHODS:
            volume = _method_volume(args, method, state)
            if not volume.exists():
                raise FileNotFoundError(volume)
            gt_volume = _gt_volume(args, state)
            if not gt_volume.exists():
                raise FileNotFoundError(gt_volume)
            split_root = args.output_dir / "split_volumes" / method
            inside = split_root / "inside_moving_mask" / f"state{state:04d}.mrc"
            outside = split_root / "outside_moving_mask" / f"state{state:04d}.mrc"
            frames.append(
                {
                    "frame_index": frame_index,
                    "state": state,
                    "method": method,
                    "color": METHOD_COLORS[method],
                    "volume": str(volume),
                    "gt_color": args.gt_color,
                    "gt_volume": str(gt_volume),
                    "inside_volume": str(inside),
                    "outside_volume": str(outside),
                    "full_png": str(args.output_dir / "frames_raw" / "full_view" / method / f"state_{state:04d}.png"),
                    "moving_png": str(
                        args.output_dir
                        / "frames_raw"
                        / "moving_mask_highlight"
                        / method
                        / f"state_{state:04d}.png"
                    ),
                }
            )
    return frames


def _write_plan(args: argparse.Namespace, frames: list[dict[str, Any]], states: list[int]) -> Path:
    full_view = _load_view(args.full_view_json)
    moving_view = _load_view(args.moving_view_json)
    width = args.width
    height = args.height
    if args.use_view_size and full_view.get("window_size"):
        width, height = [int(x) for x in full_view["window_size"]]
    plan = {
        "source": {
            "trajectory_root": str(args.trajectory_root),
            "source_run": str(args.source_run),
            "moving_mask": str(args.moving_mask),
            "cryodrgn_dir": str(args.cryodrgn_dir),
            "threedflex_dir": str(args.threedflex_dir),
            "full_view_json": str(args.full_view_json),
            "moving_view_json": str(args.moving_view_json),
        },
        "states": states,
        "methods": list(METHODS),
        "render": {
            "width": width,
            "height": height,
            "supersample": args.supersample,
            "background": args.background,
            "full_view_camera_matrix": full_view["camera_matrix"],
            "moving_view_camera_matrix": moving_view["camera_matrix"],
            "full_contour_level": args.full_contour_level,
            "moving_contour_level": args.moving_contour_level,
            "outside_color": args.outside_color,
            "outside_opacity": args.outside_opacity,
            "method_colors": METHOD_COLORS,
            "gt_color": args.gt_color,
            "overlay_gt": args.overlay_gt,
            "gt_contour_level": args.gt_contour_level,
            "gt_opacity": args.gt_opacity,
            "silhouettes": args.silhouettes,
        },
        "movie": {
            "fps": args.fps,
            "duration_seconds": len(states) / args.fps,
        },
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
    for frame in plan["frames"]:
        state = int(frame["state"])
        method = frame["method"]
        color = frame["color"]

        run(session, "close")
        if render["overlay_gt"]:
            run(session, f"open {_quote(frame['gt_volume'])}")
            run(session, f"open {_quote(frame['volume'])}")
            gt_model = "#1"
            estimate_model = "#2"
        else:
            run(session, f"open {_quote(frame['volume'])}")
            gt_model = None
            estimate_model = "#1"
        _chimerax_apply_scene(run, render)
        if gt_model is not None:
            _chimerax_style_volume(
                run,
                gt_model,
                render["gt_color"],
                float(render["gt_contour_level"]),
                float(render["gt_opacity"]),
            )
        _chimerax_style_volume(run, estimate_model, color, float(render["full_contour_level"]), 1.0)
        run(session, f"view matrix camera {render['full_view_camera_matrix']}")
        _chimerax_save(run, Path(frame["full_png"]), render)
        print(f"rendered full {method} state {state:04d}", flush=True)

        run(session, "close")
        if render["overlay_gt"]:
            run(session, f"open {_quote(frame['gt_volume'])}")
            run(session, f"open {_quote(frame['outside_volume'])}")
            run(session, f"open {_quote(frame['inside_volume'])}")
            gt_model = "#1"
            outside_model = "#2"
            inside_model = "#3"
        else:
            run(session, f"open {_quote(frame['outside_volume'])}")
            run(session, f"open {_quote(frame['inside_volume'])}")
            gt_model = None
            outside_model = "#1"
            inside_model = "#2"
        _chimerax_apply_scene(run, render)
        if gt_model is not None:
            _chimerax_style_volume(
                run,
                gt_model,
                render["gt_color"],
                float(render["gt_contour_level"]),
                float(render["gt_opacity"]),
            )
        _chimerax_style_volume(
            run,
            outside_model,
            render["outside_color"],
            float(render["moving_contour_level"]),
            float(render["outside_opacity"]),
        )
        _chimerax_style_volume(run, inside_model, color, float(render["moving_contour_level"]), 1.0)
        run(session, f"view matrix camera {render['moving_view_camera_matrix']}")
        _chimerax_save(run, Path(frame["moving_png"]), render)
        print(f"rendered moving {method} state {state:04d}", flush=True)

    run(session, "exit")


def _run_chimerax(args: argparse.Namespace, plan: Path) -> Path:
    log_path = args.output_dir / "logs" / "chimerax_render_method_trajectory.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{Path(__file__).resolve()} --internal-render --plan {plan}"
    command = (
        "unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_ROOT CONDA_SHLVL "
        "CONDA_PROMPT_MODIFIER CONDA_PREFIX PYTHONPATH PYTHONHOME VIRTUAL_ENV; "
        f"module purge; module load {shlex.quote(args.chimerax_module)}; "
        f"chimerax --nogui --offscreen --script {shlex.quote(payload)}"
    )
    with log_path.open("w") as log:
        log.write(f"COMMAND={command}\n")
        log.flush()
        proc = subprocess.run(["bash", "-lc", command], stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
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


def _copy_sequence(frames: list[Path], seq_dir: Path) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    for old in seq_dir.glob("frame_*.png"):
        old.unlink()
    for index, frame in enumerate(frames):
        if not frame.exists():
            raise FileNotFoundError(frame)
        shutil.copy2(frame, seq_dir / f"frame_{index:04d}.png")


def _assemble_movie(args: argparse.Namespace, frames: list[Path], stem: str) -> dict[str, str]:
    movie_dir = args.output_dir / "movies"
    seq_dir = args.output_dir / "frames_seq" / stem
    movie_dir.mkdir(parents=True, exist_ok=True)
    _copy_sequence(frames, seq_dir)
    palette = movie_dir / f"{stem}_palette.png"
    gif = movie_dir / f"{stem}.gif"
    mp4 = movie_dir / f"{stem}.mp4"
    log = args.output_dir / "logs" / f"{stem}_ffmpeg.log"
    pattern = seq_dir / "frame_%04d.png"
    rc1 = _run_logged(
        ["ffmpeg", "-y", "-framerate", f"{args.fps:g}", "-i", str(pattern), "-vf", "palettegen", str(palette)],
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
    return {"gif": str(gif), "mp4": str(mp4), "palette": str(palette), "log": str(log)}


def _compose_horizontal(images: list[Path], out: Path) -> None:
    from PIL import Image

    opened = [Image.open(path).convert("RGB") for path in images]
    height = max(img.height for img in opened)
    width = sum(img.width for img in opened)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for img in opened:
        y = (height - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def _assemble_all_movies(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, dict[str, str]]:
    states = [int(s) for s in plan["states"]]
    frame_lookup = {
        (frame["method"], int(frame["state"])): {
            "full": Path(frame["full_png"]),
            "moving": Path(frame["moving_png"]),
        }
        for frame in plan["frames"]
    }
    movies: dict[str, dict[str, str]] = {}
    for view_key, stem_part in (("full", "full_view"), ("moving", "moving_mask_highlight")):
        for method in METHODS:
            frames = [frame_lookup[(method, state)][view_key] for state in states]
            movies[f"{method}_{stem_part}"] = _assemble_movie(args, frames, f"{method}_{stem_part}")

        combined_frames: list[Path] = []
        for state in states:
            out = args.output_dir / "frames_raw" / f"all_methods_{stem_part}_horizontal" / f"state_{state:04d}.png"
            _compose_horizontal([frame_lookup[(method, state)][view_key] for method in METHODS], out)
            combined_frames.append(out)
        movies[f"all_methods_{stem_part}_horizontal"] = _assemble_movie(
            args,
            combined_frames,
            f"all_methods_{stem_part}_horizontal",
        )
    return movies


def _write_audit(
    args: argparse.Namespace,
    plan_path: Path,
    render_log: Path,
    movies: dict[str, dict[str, str]],
) -> Path:
    audit = {
        "script": str(Path(__file__).resolve()),
        "output_dir": str(args.output_dir),
        "plan": str(plan_path),
        "render_log": str(render_log),
        "movies": movies,
        "notes": [
            "3DFlex frames map by order to the states listed in render_plan.json",
            "moving mask is not rendered as a surface",
            "moving-mask highlight uses inside=volume*mask and outside=volume*(1-mask)",
            "combined horizontal movies are ordered left-to-right: recovar, cryodrgn, 3dflex",
            "fixed colors are stored in render_plan.json under render.method_colors and render.gt_color",
        ],
    }
    out = args.output_dir / "render_method_trajectory_audit.json"
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return out


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-render", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument(
        "--recovar-compute-dir",
        type=Path,
        help="Directory containing RECOVAR stateXXXX/state000.mrc outputs. Defaults to trajectory-root/recovar/compute_state.",
    )
    parser.add_argument("--cryodrgn-dir", type=Path, default=DEFAULT_CRYODRGN_DIR)
    parser.add_argument("--threedflex-dir", type=Path, default=DEFAULT_THREEDFLEX_DIR)
    parser.add_argument(
        "--threedflex-states",
        default="default",
        help="Comma-separated state labels corresponding to 3DFlex frame_000, frame_001, ...; default is the 21-state trajectory list.",
    )
    parser.add_argument("--moving-mask", type=Path)
    parser.add_argument("--full-view-json", type=Path, default=DEFAULT_FULL_VIEW_JSON)
    parser.add_argument("--moving-view-json", type=Path, default=DEFAULT_MOVING_VIEW_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--states", default="default")
    parser.add_argument("--full-contour-level", type=float, default=0.022)
    parser.add_argument("--moving-contour-level", type=float, default=0.013)
    parser.add_argument("--outside-color", default="#909090")
    parser.add_argument("--outside-opacity", type=float, default=0.3)
    parser.add_argument("--overlay-gt", action="store_true")
    parser.add_argument("--gt-color", default=GT_COLOR)
    parser.add_argument("--gt-contour-level", type=float, default=0.011)
    parser.add_argument("--gt-opacity", type=float, default=0.5)
    parser.add_argument("--background", default="white")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--use-view-size", action="store_true")
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--fps", type=float, default=5.25)
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

    args.trajectory_root = args.trajectory_root.resolve()
    args.source_run = args.source_run.resolve()
    if args.recovar_compute_dir is None:
        args.recovar_compute_dir = args.trajectory_root / "recovar" / "compute_state"
    args.recovar_compute_dir = args.recovar_compute_dir.resolve()
    args.cryodrgn_dir = args.cryodrgn_dir.resolve()
    args.threedflex_dir = args.threedflex_dir.resolve()
    args.threedflex_states = _parse_states(args.threedflex_states)
    args.full_view_json = args.full_view_json.resolve()
    args.moving_view_json = args.moving_view_json.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.moving_mask is None:
        args.moving_mask = args.source_run / "05_masks" / "focus_mask_moving.mrc"
    args.moving_mask = args.moving_mask.resolve()
    for path in (
        args.trajectory_root,
        args.source_run,
        args.recovar_compute_dir,
        args.cryodrgn_dir,
        args.threedflex_dir,
        args.moving_mask,
        args.full_view_json,
        args.moving_view_json,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    states = _parse_states(args.states)
    frames = _build_frames(args, states)
    _prepare_split_volumes(args, frames)
    plan_path = _write_plan(args, frames, states)
    render_log = args.output_dir / "logs" / "chimerax_render_method_trajectory.log"
    if not args.skip_chimerax:
        render_log = _run_chimerax(args, plan_path)

    movies: dict[str, dict[str, str]] = {}
    if not args.skip_movies:
        plan = json.loads(plan_path.read_text())
        movies = _assemble_all_movies(args, plan)
    audit = _write_audit(args, plan_path, render_log, movies)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "plan": str(plan_path),
                "audit": str(audit),
                "render_log": str(render_log),
                "movies": movies,
            },
            indent=2,
            sort_keys=True,
        )
    )


main(sys.argv[1:])
