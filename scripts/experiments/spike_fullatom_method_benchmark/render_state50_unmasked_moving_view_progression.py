#!/usr/bin/env python3
"""Render state-50 method progression with the moving-region camera and no mask.

This uses the camera matrix pasted from ChimeraX for the moving part of the
spike, but renders full volumes directly.  No focus/not-moving mask is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY_CSV = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "not_moving_softmask_fsc_resolution_20260601/"
    "not_moving_softmask_noise3_fsc_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "chimerax_state50_method_progression_unmasked_moving_view_20260601"
)
RENDERER = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "spike_fullatom_method_benchmark"
    / "render_chimerax_manifest.py"
)
MOVING_CAMERA_MATRIX = (
    "-0.16946,-0.046152,0.98446,429.28,"
    "0.97109,0.16259,0.17478,184.06,"
    "-0.16813,0.98561,0.017264,208.79"
)

METHOD_ORDER = ["recovar", "cryodrgn", "3dflex"]
METHOD_LABELS = {"recovar": "RECOVAR", "cryodrgn": "cryoDRGN", "3dflex": "3DFlex"}
METHOD_COLORS = {"recovar": "#1f77b4", "cryodrgn": "#d62728", "3dflex": "#2ca02c"}
N_ORDER = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
N_LABELS = {
    10_000: "10k",
    30_000: "30k",
    100_000: "100k",
    300_000: "300k",
    1_000_000: "1M",
    3_000_000: "3M",
}


@dataclass(frozen=True)
class RenderEntry:
    method: str
    n_images: int
    n_label: str
    state: int
    estimate: Path
    gt: Path


def _recovar_filtered_path(path: Path) -> Path:
    if path.name == "state000_unfil.mrc":
        filtered = path.with_name("state000.mrc")
        if filtered.exists():
            return filtered
    return path


def _read_entries(summary_csv: Path, state: int) -> list[RenderEntry]:
    entries: list[RenderEntry] = []
    with summary_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["state"]) != state:
                continue
            method = row["method"]
            if method not in METHOD_ORDER:
                continue
            estimate = Path(row["estimate"])
            if method == "recovar":
                estimate = _recovar_filtered_path(estimate)
            entries.append(
                RenderEntry(
                    method=method,
                    n_images=int(row["n_images"]),
                    n_label=row["n_label"],
                    state=state,
                    estimate=estimate,
                    gt=Path(row["gt"]),
                )
            )
    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    entries.sort(key=lambda e: (method_order[e.method], e.n_images))
    return entries


def _write_manifest(args: argparse.Namespace, entries: list[RenderEntry]) -> Path:
    manifest = args.output_dir / "manifest_chimerax_state50_unmasked_moving_view.csv"
    fields = [
        "noise_level",
        "bfactor",
        "collection",
        "n_images",
        "n_label",
        "state",
        "method",
        "method_label",
        "role",
        "volume_path",
        "render_name",
        "contour_level",
        "color",
        "mask_policy",
    ]
    rows: list[dict[str, str]] = []
    collection = "noise3_b80_state50_unmasked_moving_view"
    for entry in entries:
        n_tag = f"n{entry.n_images:08d}"
        rows.append(
            {
                "noise_level": "3",
                "bfactor": "80",
                "collection": collection,
                "n_images": str(entry.n_images),
                "n_label": entry.n_label,
                "state": str(entry.state),
                "method": entry.method,
                "method_label": METHOD_LABELS[entry.method],
                "role": "estimate",
                "volume_path": str(entry.estimate),
                "render_name": f"{entry.method}/state{entry.state:04d}_noise3_{entry.method}_{n_tag}_unmasked_moving_view.png",
                "contour_level": f"{args.contour_level:.8g}",
                "color": METHOD_COLORS[entry.method],
                "mask_policy": "none",
            }
        )
        rows.append(
            {
                "noise_level": "3",
                "bfactor": "80",
                "collection": collection,
                "n_images": str(entry.n_images),
                "n_label": entry.n_label,
                "state": str(entry.state),
                "method": "ground_truth",
                "method_label": "GT",
                "role": "gt",
                "volume_path": str(entry.gt),
                "render_name": f"ground_truth/state{entry.state:04d}_noise3_gt_{n_tag}_unmasked_moving_view.png",
                "contour_level": f"{args.contour_level:.8g}",
                "color": args.gt_color,
                "mask_policy": "none",
            }
        )
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _run_chimerax(args: argparse.Namespace, manifest: Path) -> Path:
    raw_dir = args.output_dir / "png_raw"
    log_dir = args.output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = (
        f"{RENDERER} "
        f"--manifest {manifest} "
        f"--output-dir {raw_dir} "
        f"--camera-matrix={shlex.quote(args.camera_matrix)} "
        f"--zoom 1.0 "
        f"--width {args.width} "
        f"--height {args.height} "
        f"--supersample {args.supersample} "
        f"--background {shlex.quote(args.background)} "
        f"--roles estimate "
        f"--states {args.state} "
        f"--overlay-gt "
        f"--gt-color {shlex.quote(args.gt_color)} "
        f"--gt-opacity {args.gt_opacity} "
        f"--fallback-level {args.contour_level}"
    )
    command = (
        f"module purge; module load {shlex.quote(args.chimerax_module)}; "
        f"chimerax --nogui --offscreen --script {shlex.quote(payload)}"
    )
    log = log_dir / "chimerax_render.log"
    with log.open("w") as handle:
        proc = subprocess.run(["bash", "-lc", command], stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"ChimeraX render failed with code {proc.returncode}; see {log}")
    return log


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_BIG = _font(42, True)
FONT_MED = _font(34, True)
FONT_SMALL = _font(26, False)
FONT_CELL = _font(22, True)
DURATION_MS = 1500


def _read_estimate_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["role"] == "estimate"]


def _raw_path(raw_dir: Path, row: dict[str, str]) -> Path:
    render_name = Path(row["render_name"])
    return raw_dir / f"{render_name.stem}_gt_overlay{render_name.suffix}"


def _labeled_path(labeled_dir: Path, row: dict[str, str]) -> Path:
    return labeled_dir / row["method"] / _raw_path(Path("."), row).name


def _label_one(raw_dir: Path, labeled_dir: Path, row: dict[str, str], args: argparse.Namespace) -> Path:
    src = _raw_path(raw_dir, row)
    if not src.exists():
        raise FileNotFoundError(src)
    dst = _labeled_path(labeled_dir, row)
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    title = f"{METHOD_LABELS[row['method']]}  {row['n_label']}  state {row['state']}"
    subtitle = f"unmasked moving-region view; contour {args.contour_level:.4f}; GT opacity {args.gt_opacity:.1f}"
    margin = 28
    pad_x, pad_y = 22, 14
    title_box = draw.textbbox((0, 0), title, font=FONT_BIG)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=FONT_SMALL)
    width = max(title_box[2], subtitle_box[2]) + 2 * pad_x
    height = (title_box[3] - title_box[1]) + (subtitle_box[3] - subtitle_box[1]) + 3 * pad_y
    draw.rounded_rectangle(
        (margin, margin, margin + width, margin + height),
        radius=10,
        fill=(255, 255, 255, 210),
        outline=(0, 0, 0, 90),
        width=2,
    )
    draw.text((margin + pad_x, margin + pad_y), title, fill=(0, 0, 0, 255), font=FONT_BIG)
    draw.text(
        (margin + pad_x, margin + 2 * pad_y + title_box[3] - title_box[1]),
        subtitle,
        fill=(40, 40, 40, 255),
        font=FONT_SMALL,
    )
    image.convert("RGB").save(dst, quality=95)
    return dst


def _thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _placeholder(size: tuple[int, int], text: str) -> Image.Image:
    image = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(190, 190, 190), width=2)
    box = draw.textbbox((0, 0), text, font=FONT_MED)
    draw.text(
        ((size[0] - (box[2] - box[0])) // 2, (size[1] - (box[3] - box[1])) // 2),
        text,
        fill=(90, 90, 90),
        font=FONT_MED,
    )
    return image


def _contact_sheet(index: dict[tuple[str, int], Path], panels_dir: Path) -> Path:
    panels_dir.mkdir(parents=True, exist_ok=True)
    cell = (420, 315)
    left = 170
    top = 80
    sheet = Image.new("RGB", (left + cell[0] * len(N_ORDER), top + cell[1] * len(METHOD_ORDER)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 18), "Spike benchmark noise=3, state 50: unmasked moving-region view over GT", fill=(0, 0, 0), font=FONT_MED)
    for j, n_images in enumerate(N_ORDER):
        label = N_LABELS[n_images]
        box = draw.textbbox((0, 0), label, font=FONT_CELL)
        draw.text((left + j * cell[0] + (cell[0] - (box[2] - box[0])) // 2, 48), label, fill=(0, 0, 0), font=FONT_CELL)
    for i, method in enumerate(METHOD_ORDER):
        y = top + i * cell[1]
        draw.text((20, y + cell[1] // 2 - 14), METHOD_LABELS[method], fill=(0, 0, 0), font=FONT_CELL)
        for j, n_images in enumerate(N_ORDER):
            path = index.get((method, n_images))
            tile = _thumb(path, cell) if path else _placeholder(cell, "not available")
            x = left + j * cell[0]
            sheet.paste(tile, (x, y))
            draw.rectangle((x, y, x + cell[0] - 1, y + cell[1] - 1), outline=(210, 210, 210), width=1)
    output = panels_dir / "state50_noise3_unmasked_moving_view_method_by_n_contact_sheet.png"
    sheet.save(output, quality=95)
    return output


def _save_gif(paths: list[Path], output: Path, size: tuple[int, int] | None = None) -> None:
    images: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if size is not None:
            image.thumbnail(size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", size, "white")
            canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
            image = canvas
        images.append(image)
    if images:
        output.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(output, save_all=True, append_images=images[1:], duration=DURATION_MS, loop=0, optimize=True)


def _frames(index: dict[tuple[str, int], Path], frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cell = (640, 480)
    left = 160
    top = 70
    paths = []
    for n_images in N_ORDER:
        frame = Image.new("RGB", (left + cell[0], top + len(METHOD_ORDER) * cell[1]), "white")
        draw = ImageDraw.Draw(frame)
        draw.text((20, 18), f"noise=3  state 50  unmasked moving view  n={N_LABELS[n_images]}", fill=(0, 0, 0), font=FONT_MED)
        for i, method in enumerate(METHOD_ORDER):
            y = top + i * cell[1]
            draw.text((20, y + cell[1] // 2 - 14), METHOD_LABELS[method], fill=(0, 0, 0), font=FONT_CELL)
            path = index.get((method, n_images))
            tile = _thumb(path, cell) if path else _placeholder(cell, "not available")
            frame.paste(tile, (left, y))
            draw.rectangle((left, y, left + cell[0] - 1, y + cell[1] - 1), outline=(210, 210, 210), width=1)
        output = frames_dir / f"all_methods_n{n_images:08d}.png"
        frame.save(output, quality=95)
        paths.append(output)
    return paths


def _compose(args: argparse.Namespace, manifest: Path) -> dict[str, object]:
    raw_dir = args.output_dir / "png_raw"
    labeled_dir = args.output_dir / "png_labeled"
    panels_dir = args.output_dir / "panels"
    animations_dir = args.output_dir / "animations"
    frames_dir = args.output_dir / "animation_frames"
    rows = _read_estimate_rows(manifest)
    labeled = [_label_one(raw_dir, labeled_dir, row, args) for row in rows]
    index = {(row["method"], int(row["n_images"])): _labeled_path(labeled_dir, row) for row in rows}
    contact = _contact_sheet(index, panels_dir)
    per_method = {}
    for method in METHOD_ORDER:
        paths = [index[(method, n_images)] for n_images in N_ORDER if (method, n_images) in index]
        output = animations_dir / f"state50_noise3_unmasked_moving_view_{method}_progression.gif"
        _save_gif(paths, output, size=(960, 720))
        if output.exists():
            per_method[method] = str(output)
    frame_paths = _frames(index, frames_dir)
    all_gif = animations_dir / "state50_noise3_unmasked_moving_view_all_methods_progression.gif"
    _save_gif(frame_paths, all_gif, size=(800, 1800))
    all_mp4 = animations_dir / "state50_noise3_unmasked_moving_view_all_methods_progression.mp4"
    ffmpeg_log = args.output_dir / "logs" / "ffmpeg_all_methods.log"
    ffmpeg_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(1 / 1.5),
        "-pattern_type",
        "glob",
        "-i",
        str(frames_dir / "all_methods_n*.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-movflags",
        "+faststart",
        str(all_mp4),
    ]
    with ffmpeg_log.open("w") as handle:
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    return {
        "labeled_png_dir": str(labeled_dir),
        "n_labeled_png": len(labeled),
        "contact_sheet": str(contact),
        "per_method_gifs": per_method,
        "all_methods_gif": str(all_gif),
        "all_methods_mp4": str(all_mp4) if proc.returncode == 0 else None,
        "all_method_frames": [str(path) for path in frame_paths],
        "duration_ms": DURATION_MS,
        "ffmpeg_returncode": proc.returncode,
        "ffmpeg_command": command,
        "ffmpeg_log": str(ffmpeg_log),
    }


def _write_audit(args: argparse.Namespace, entries: list[RenderEntry], manifest: Path, outputs: dict[str, object] | None, render_log: Path | None) -> Path:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.method] = counts.get(entry.method, 0) + 1
    missing = {
        method: [N_LABELS[n] for n in N_ORDER if not any(e.method == method and e.n_images == n for e in entries)]
        for method in METHOD_ORDER
    }
    audit = args.output_dir / "COMMAND_AUDIT.md"
    lines = [
        "# State 50 Unmasked Moving-View ChimeraX Render Audit",
        "",
        f"- Repo script: `{Path(__file__).resolve()}`",
        f"- Summary CSV: `{args.summary_csv}`",
        f"- Output dir: `{args.output_dir}`",
        "- Mask policy: `none`",
        f"- Camera matrix: `{args.camera_matrix}`",
        f"- ChimeraX renderer: `{RENDERER}`",
        f"- State: `{args.state}`",
        f"- Contour level for estimate and GT: `{args.contour_level}`",
        f"- GT overlay opacity: `{args.gt_opacity}`",
        "- RECOVAR source: filtered `state000.mrc` when the source CSV points at `state000_unfil.mrc`",
        f"- Counts: `{json.dumps(counts, sort_keys=True)}`",
        f"- Missing n by method: `{json.dumps(missing, sort_keys=True)}`",
        f"- Manifest: `{manifest}`",
        f"- ChimeraX log: `{render_log}`",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(shlex.quote(item) for item in sys.argv),
        "```",
    ]
    if outputs is not None:
        lines.extend(["", "## Outputs", "", "```json", json.dumps(outputs, indent=2, sort_keys=True), "```"])
    audit.write_text("\n".join(lines) + "\n")
    return audit


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state", type=int, default=50)
    parser.add_argument("--camera-matrix", default=MOVING_CAMERA_MATRIX)
    parser.add_argument("--contour-level", type=float, default=0.013)
    parser.add_argument("--gt-color", default="#777777")
    parser.add_argument("--gt-opacity", type=float, default=0.3)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--background", default="white")
    parser.add_argument("--chimerax-module", default="chimerax/1.9")
    parser.add_argument("--run-chimerax", action="store_true")
    parser.add_argument("--skip-compose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = _read_entries(args.summary_csv, args.state)
    if not entries:
        raise RuntimeError(f"no state {args.state} entries found in {args.summary_csv}")
    for entry in entries:
        if not entry.estimate.exists():
            raise FileNotFoundError(entry.estimate)
        if not entry.gt.exists():
            raise FileNotFoundError(entry.gt)
    manifest = _write_manifest(args, entries)
    render_log = _run_chimerax(args, manifest) if args.run_chimerax else None
    outputs = None if args.skip_compose else _compose(args, manifest)
    if outputs is not None and render_log is not None:
        outputs["chimerax_log"] = str(render_log)
    audit = _write_audit(args, entries, manifest, outputs, render_log)
    summary = {
        "output_dir": str(args.output_dir),
        "manifest": str(manifest),
        "audit": str(audit),
        "render_log": str(render_log) if render_log else None,
        "n_entries": len(entries),
        "outputs": outputs,
    }
    (args.output_dir / "render_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
