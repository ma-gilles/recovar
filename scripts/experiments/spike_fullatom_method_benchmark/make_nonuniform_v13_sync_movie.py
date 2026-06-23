#!/usr/bin/env python3
"""Build a synced nonuniform trajectory volume/FSC movie from existing frames.

This script intentionally does not rerender ChimeraX volumes.  It pairs an
existing per-state horizontal volume frame with cumulative FSC curves read from
the v13 state-specific mask scoring directory.  The output is meant to make the
FSC panel match the static v13 scoring figure exactly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_BENCH_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_nonuniform_B70_noise1_b80_300k_20260604"
)
DEFAULT_VOLUME_FRAMES = (
    DEFAULT_BENCH_ROOT
    / "chimerax_moving_view_movies_corrected_masks_20260604"
    / "frames_raw/all_methods_moving_mask_highlight_horizontal"
)
DEFAULT_CURVE_ROOT = (
    DEFAULT_BENCH_ROOT / "corrected_nonuniform_v13_solid_mask_scoring_shellfix_20260605"
)
DEFAULT_OUT_DIR = DEFAULT_BENCH_ROOT / "v13_state_mask_volume_fsc_sync_shellfix_20260605"

METHODS = ("recovar", "cryodrgn", "3dflex")
METHOD_TITLES = {"recovar": "RECOVAR", "cryodrgn": "cryoDRGN", "3dflex": "3DFlex"}
METHOD_COLORS = {"recovar": "#22a884", "cryodrgn": "#d95f02", "3dflex": "#7570b3"}


def parse_states(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def first_crossing_resolution(frequency: np.ndarray, fsc: np.ndarray, threshold: float = 0.5) -> float:
    valid = np.isfinite(frequency) & np.isfinite(fsc) & (frequency > 0)
    frequency = np.asarray(frequency, dtype=np.float64)[valid]
    fsc = np.asarray(fsc, dtype=np.float64)[valid]
    if frequency.size == 0:
        return float("nan")
    bad = np.flatnonzero(fsc < threshold)
    if bad.size == 0:
        return float(1.0 / frequency[-1])
    idx = int(bad[0])
    if idx == 0:
        return float(1.0 / frequency[idx])
    x0, x1 = frequency[idx - 1], frequency[idx]
    y0, y1 = fsc[idx - 1], fsc[idx]
    if np.isfinite(y0) and np.isfinite(y1) and y0 != y1:
        frac = float((threshold - y0) / (y1 - y0))
        frac = max(0.0, min(1.0, frac))
        x = x0 + frac * (x1 - x0)
        if x > 0:
            return float(1.0 / x)
    return float(1.0 / x1)


def load_curve(curve_root: Path, mask: str, method: str, state: int) -> tuple[np.ndarray, np.ndarray, float]:
    path = curve_root / mask / "curves" / method / f"state{state:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        frequency = np.asarray(data["frequency"], dtype=np.float64)
        fsc = np.asarray(data["fsc"], dtype=np.float64)
    return frequency, fsc, first_crossing_resolution(frequency, fsc)


def render_fsc_panel(
    *,
    curve_root: Path,
    mask: str,
    states: list[int],
    upto_index: int,
    out_path: Path,
    width_px: int,
    height_px: int,
    x_max: float,
) -> None:
    dpi = 200
    fig_w = width_px / dpi
    fig_h = height_px / dpi
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), sharey=True, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(states)))
    active = states[: upto_index + 1]
    for ax, method in zip(axes, METHODS, strict=True):
        for state, color in zip(states, colors):
            if state not in active:
                continue
            frequency, fsc, resolution = load_curve(curve_root, mask, method, state)
            alpha = 1.0 if state == active[-1] else 0.42
            linewidth = 3.0 if state == active[-1] else 1.7
            label = f"{state:02d}  {resolution:.2f} A"
            ax.plot(frequency[1:], fsc[1:], color=color, alpha=alpha, lw=linewidth, label=label)
        ax.axhline(0.5, color="0.25", ls=":", lw=1.3)
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(True, color="0.88", lw=0.8)
        ax.set_title(METHOD_TITLES[method], color=METHOD_COLORS[method], fontsize=19, weight="bold")
        ax.set_xlabel("spatial frequency (1/A)", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(title="state  FSC0.5", title_fontsize=10, fontsize=8.5, frameon=False, loc="lower left")
    axes[0].set_ylabel("masked FSC", fontsize=14)
    fig.suptitle(f"FSC vs GT, cumulative v13 state-specific mask through state {active[-1]}", fontsize=22, weight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def compose_frame(volume_path: Path, fsc_path: Path, out_path: Path, state: int, n_images: str) -> None:
    volume = Image.open(volume_path).convert("RGB")
    fsc = Image.open(fsc_path).convert("RGB")
    if fsc.width != volume.width:
        new_h = int(round(fsc.height * (volume.width / fsc.width)))
        fsc = fsc.resize((volume.width, new_h), Image.Resampling.LANCZOS)
    label_h = 118
    canvas = Image.new("RGB", (volume.width, label_h + volume.height + fsc.height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((45, 24), f"{n_images} images    state {state}", fill="black", font=font(58))
    canvas.paste(volume, (0, label_h))
    canvas.paste(fsc, (0, label_h + volume.height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def run_ffmpeg(frames_dir: Path, movie_dir: Path, stem: str, fps: float) -> dict[str, object]:
    movie_dir.mkdir(parents=True, exist_ok=True)
    palette = movie_dir / f"{stem}_palette.png"
    gif = movie_dir / f"{stem}.gif"
    mp4 = movie_dir / f"{stem}.mp4"
    log = movie_dir / f"{stem}_ffmpeg.log"
    pattern = frames_dir / "frame_%04d.png"
    commands = [
        ["ffmpeg", "-y", "-framerate", f"{fps:g}", "-i", str(pattern), "-vf", "palettegen", str(palette)],
        [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:g}",
            "-i",
            str(pattern),
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
            str(pattern),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "17",
            str(mp4),
        ],
    ]
    returncodes: list[int] = []
    with log.open("w") as handle:
        for command in commands:
            handle.write("$ " + " ".join(command) + "\n")
            proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            returncodes.append(proc.returncode)
    if any(returncodes):
        raise RuntimeError(f"ffmpeg failed; see {log}")
    return {"gif": str(gif), "mp4": str(mp4), "palette": str(palette), "ffmpeg_log": str(log)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-frame-dir", type=Path, default=DEFAULT_VOLUME_FRAMES)
    parser.add_argument("--curve-root", type=Path, default=DEFAULT_CURVE_ROOT)
    parser.add_argument("--mask", default="tracked_atoms", choices=("tracked_atoms", "moving"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--states", default="0,5,10,20,25,30,40,50,60,70,80,90")
    parser.add_argument("--n-images-label", default="300k")
    parser.add_argument("--seconds-per-state", type=float, default=1.5)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--fsc-height-px", type=int, default=1150)
    parser.add_argument("--x-max", type=float, default=0.39)
    args = parser.parse_args()

    states = parse_states(args.states)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fsc_dir = args.out_dir / "fsc_panels"
    unique_dir = args.out_dir / "frames_unique"
    repeated_dir = args.out_dir / "frames_repeated"

    manifest_rows: list[dict[str, object]] = []
    unique_paths: list[Path] = []
    for idx, state in enumerate(states):
        volume = args.volume_frame_dir / f"state_{state:04d}.png"
        if not volume.exists():
            raise FileNotFoundError(volume)
        width = Image.open(volume).width
        fsc = fsc_dir / f"fsc_cumulative_v13_{args.mask}_state{state:04d}.png"
        render_fsc_panel(
            curve_root=args.curve_root,
            mask=args.mask,
            states=states,
            upto_index=idx,
            out_path=fsc,
            width_px=width,
            height_px=args.fsc_height_px,
            x_max=args.x_max,
        )
        frame = unique_dir / f"frame_{idx:04d}_state{state:04d}.png"
        compose_frame(volume, fsc, frame, state, args.n_images_label)
        unique_paths.append(frame)
        manifest_rows.append({"state": state, "volume_frame": str(volume), "fsc_panel": str(fsc), "synced_frame": str(frame)})

    repeats = max(1, int(round(args.seconds_per_state * args.fps)))
    frame_idx = 0
    repeated_dir.mkdir(parents=True, exist_ok=True)
    for frame in unique_paths:
        image = Image.open(frame).convert("RGB")
        for _ in range(repeats):
            image.save(repeated_dir / f"frame_{frame_idx:04d}.png", quality=95)
            frame_idx += 1

    stem = f"nonuniform_noise1_300k_{args.mask}_v13_volume_fsc_sync_shellfix"
    movies = run_ffmpeg(repeated_dir, args.out_dir / "movies", stem, args.fps)
    summary = {
        "volume_frame_dir": str(args.volume_frame_dir.resolve()),
        "curve_root": str(args.curve_root.resolve()),
        "mask": args.mask,
        "states": states,
        "seconds_per_state": args.seconds_per_state,
        "fps": args.fps,
        "frames_unique": str(unique_dir.resolve()),
        "frames_repeated": str(repeated_dir.resolve()),
        "fsc_panels": str(fsc_dir.resolve()),
        "movies": movies,
        "manifest": manifest_rows,
    }
    (args.out_dir / "README.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
