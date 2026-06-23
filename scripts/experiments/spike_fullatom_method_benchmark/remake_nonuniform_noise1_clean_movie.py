#!/usr/bin/env python3
"""Compose clean nonuniform noise=1 volume/FSC movies.

This script reuses existing ChimeraX renders and the shell-fixed v13 FSC
curves.  It intentionally does not rerender volumes; the only work here is
presentation composition so the nonuniform trajectory matches the newer
uniform-noise movie style.

The highest-frequency edge shell is optionally dropped before plotting.  That
shell can be dominated by grid/Nyquist artifacts and should not drive a visible
rebound in presentation movies.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BENCH_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_nonuniform_B70_noise1_b80_300k_20260604"
)
DEFAULT_RENDER_ROOT = (
    BENCH_ROOT
    / "nonuniform_noise1_300k_threshold_sweep_shellfix_20260606"
    / "contour_0p014"
    / "render"
    / "frames_raw"
    / "moving_mask_highlight"
)
DEFAULT_CURVE_ROOT = BENCH_ROOT / "corrected_nonuniform_v13_solid_mask_scoring_shellfix_20260605"
DEFAULT_OUT_DIR = BENCH_ROOT / "nonuniform_noise1_300k_clean_style_volume_fsc_20260614"

METHODS = ("recovar", "cryodrgn", "3dflex")
METHOD_LABELS = {"recovar": "RECOVAR", "cryodrgn": "cryoDRGN", "3dflex": "3DFlex"}
METHOD_COLORS = {"recovar": "#1b9e77", "cryodrgn": "#d95f02", "3dflex": "#7570b3"}


def _parse_states(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_STATE = _font(70, True)


def _first_crossing_resolution(frequency: np.ndarray, fsc: np.ndarray, threshold: float = 0.5) -> float:
    valid = np.isfinite(frequency) & np.isfinite(fsc) & (frequency > 0)
    x = np.asarray(frequency[valid], dtype=np.float64)
    y = np.asarray(fsc[valid], dtype=np.float64)
    if x.size == 0:
        return math.nan
    below = np.flatnonzero(y < threshold)
    if below.size == 0:
        return float(1.0 / x[-1])
    idx = int(below[0])
    if idx == 0:
        return math.nan
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    if y1 == y0:
        cross = x1
    else:
        cross = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    return float(1.0 / cross) if cross > 0 else math.nan


def _load_curve(curve_root: Path, mask: str, method: str, state: int) -> tuple[np.ndarray, np.ndarray, float, Path]:
    path = curve_root / mask / "curves" / method / f"state{state:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        frequency = np.asarray(data["frequency"], dtype=np.float64)
        fsc = np.asarray(data["fsc"], dtype=np.float64)
    return frequency, fsc, _first_crossing_resolution(frequency, fsc), path


def _trim_white(image: Image.Image, threshold: int = 248, pad: int = 24) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    nonwhite = np.any(arr < threshold, axis=2)
    if not np.any(nonwhite):
        return image
    ys, xs = np.where(nonwhite)
    left = max(int(xs.min()) - pad, 0)
    right = min(int(xs.max()) + pad + 1, image.width)
    top = max(int(ys.min()) - pad, 0)
    bottom = min(int(ys.max()) + pad + 1, image.height)
    return image.crop((left, top, right, bottom))


def _fit_tile(path: Path, size: tuple[int, int]) -> Image.Image:
    image = _trim_white(Image.open(path).convert("RGB"))
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _render_fsc_tile(
    *,
    curve_root: Path,
    mask: str,
    method: str,
    states: list[int],
    current_index: int,
    size: tuple[int, int],
    x_max: float,
    drop_last_shells: int,
) -> tuple[Image.Image, dict[str, object]]:
    current_state = states[current_index]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(states)))
    fig, ax = plt.subplots(figsize=(size[0] / 180, size[1] / 180), dpi=180)
    fig.patch.set_facecolor("white")
    curve_paths: dict[str, str] = {}
    current_resolution = math.nan
    ax.axhline(0.5, color="0.50", ls=":", lw=1.4)
    for idx, (state, color) in enumerate(zip(states, colors, strict=True)):
        if idx > current_index:
            continue
        frequency, fsc, resolution, path = _load_curve(curve_root, mask, method, state)
        if drop_last_shells > 0:
            frequency = frequency[:-drop_last_shells]
            fsc = fsc[:-drop_last_shells]
        plot_mask = frequency <= x_max
        curve_paths[f"{state:04d}"] = str(path)
        if state == current_state:
            current_resolution = resolution
        ax.plot(
            frequency[plot_mask][1:],
            fsc[plot_mask][1:],
            color=color,
            lw=3.0 if state == current_state else 1.8,
            alpha=1.0 if state == current_state else 0.55,
        )
    if math.isfinite(current_resolution):
        ax.text(
            0.94,
            0.10,
            f"{current_resolution:.2f} A",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=15,
            weight="bold",
        )
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(-0.04, 1.03)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, color="0.86", lw=0.7)
    ax.tick_params(labelsize=9, width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.18")
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.14, top=0.98)
    # Avoid a temporary file so repeated runs do not leave partial artifacts.
    buffer = BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    plt.close(fig)
    return image.resize(size, Image.Resampling.LANCZOS), {
        "method": method,
        "current_state": current_state,
        "current_resolution_A": current_resolution,
        "curve_paths": curve_paths,
    }


def _draw_centered(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) // 2, y), text, fill=(0, 0, 0), font=font)


def _volume_path(render_root: Path, method: str, state: int) -> Path:
    path = render_root / method / f"state_{state:04d}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _ffmpeg_from_frames(frames_dir: Path, stem: str, fps: float) -> dict[str, object]:
    gif = frames_dir.parent / f"{stem}.gif"
    mp4 = frames_dir.parent / f"{stem}.mp4"
    palette = frames_dir.parent / f"{stem}_palette.png"
    log_path = frames_dir.parent / f"{stem}_ffmpeg.log"
    pattern = str(frames_dir / "frame_%04d.png")
    commands = [
        [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:g}",
            "-i",
            pattern,
            "-vf",
            "palettegen=stats_mode=diff",
            str(palette),
        ],
        [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:g}",
            "-i",
            pattern,
            "-i",
            str(palette),
            "-lavfi",
            "paletteuse=dither=bayer:bayer_scale=3",
            str(gif),
        ],
        [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:g}",
            "-i",
            pattern,
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "14",
            "-preset",
            "slow",
            "-movflags",
            "+faststart",
            str(mp4),
        ],
    ]
    returncodes: list[int] = []
    with log_path.open("w") as handle:
        for command in commands:
            handle.write("$ " + " ".join(command) + "\n")
            proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            returncodes.append(proc.returncode)
    if any(returncodes):
        raise RuntimeError(f"ffmpeg failed with {returncodes}; see {log_path}")
    return {
        "gif": str(gif),
        "mp4": str(mp4),
        "palette": str(palette),
        "ffmpeg_log": str(log_path),
        "returncodes": returncodes,
    }


def build_movie(args: argparse.Namespace, mask: str) -> dict[str, object]:
    states = _parse_states(args.states)
    out_dir = args.out_dir / mask
    frames_unique = out_dir / "frames_unique"
    frames_video = out_dir / "frames_video"
    frames_unique.mkdir(parents=True, exist_ok=True)
    frames_video.mkdir(parents=True, exist_ok=True)

    vol_tile = (760, 570)
    fsc_tile = (760, 420)
    gap = 28
    top = 96
    width = len(METHODS) * vol_tile[0] + (len(METHODS) - 1) * gap
    height = top + vol_tile[1] + 16 + fsc_tile[1]

    frame_records: list[dict[str, object]] = []
    unique_paths: list[Path] = []
    for state_index, state in enumerate(states):
        frame = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(frame)
        _draw_centered(draw, width // 2, 12, f"{state:02d}", FONT_STATE)
        method_records: list[dict[str, object]] = []
        for col, method in enumerate(METHODS):
            x = col * (vol_tile[0] + gap)
            volume_path = _volume_path(args.render_root, method, state)
            frame.paste(_fit_tile(volume_path, vol_tile), (x, top))
            fsc_tile_image, fsc_record = _render_fsc_tile(
                curve_root=args.curve_root,
                mask=mask,
                method=method,
                states=states,
                current_index=state_index,
                size=fsc_tile,
                x_max=args.x_max,
                drop_last_shells=args.drop_last_shells,
            )
            frame.paste(fsc_tile_image, (x, top + vol_tile[1] + 16))
            fsc_record["volume_frame"] = str(volume_path)
            method_records.append(fsc_record)
        unique_path = frames_unique / f"frame_{state_index:04d}_state{state:04d}.png"
        frame.save(unique_path, quality=98)
        unique_paths.append(unique_path)
        frame_records.append({"state": state, "frame": str(unique_path), "methods": method_records})

    repeats = max(1, int(round(args.seconds_per_state * args.fps)))
    video_index = 0
    for unique_path in unique_paths:
        image = Image.open(unique_path).convert("RGB")
        for _ in range(repeats):
            image.save(frames_video / f"frame_{video_index:04d}.png", quality=98)
            video_index += 1

    final_png = out_dir / f"nonuniform_noise1_300k_{mask}_clean_volume_fsc_final.png"
    Image.open(unique_paths[-1]).save(final_png, quality=98)
    stem = f"nonuniform_noise1_300k_{mask}_clean_volume_fsc"
    movies = _ffmpeg_from_frames(frames_video, stem, args.fps)
    summary = {
        "script": str(Path(__file__).resolve()),
        "bench_root": str(BENCH_ROOT),
        "render_root": str(args.render_root.resolve()),
        "curve_root": str(args.curve_root.resolve()),
        "render_contour_source": "contour_0p014",
        "mask": mask,
        "states": states,
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "method_colors": METHOD_COLORS,
        "top_number_is_state_index": True,
        "drop_last_shells_for_plot_only": int(args.drop_last_shells),
        "seconds_per_state": args.seconds_per_state,
        "fps": args.fps,
        "frames_unique": str(frames_unique.resolve()),
        "frames_video": str(frames_video.resolve()),
        "final_png": str(final_png),
        "movies": movies,
        "frame_records": frame_records,
    }
    (out_dir / "README.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--curve-root", type=Path, default=DEFAULT_CURVE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--states", default="0,5,10,20,25,30,40,50,60,70,80,90")
    parser.add_argument("--mask", choices=("tracked_atoms", "moving", "both"), default="both")
    parser.add_argument("--seconds-per-state", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--x-max", type=float, default=0.39)
    parser.add_argument("--drop-last-shells", type=int, default=1)
    args = parser.parse_args()

    masks = ("tracked_atoms", "moving") if args.mask == "both" else (args.mask,)
    summaries = {mask: build_movie(args, mask) for mask in masks}
    top_summary = {
        "script": str(Path(__file__).resolve()),
        "outputs": {mask: summary["movies"] for mask, summary in summaries.items()},
        "summaries": {mask: str((args.out_dir / mask / "README.json").resolve()) for mask in masks},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "README.json").write_text(json.dumps(top_summary, indent=2) + "\n")
    print(json.dumps(top_summary, indent=2))


if __name__ == "__main__":
    main()
