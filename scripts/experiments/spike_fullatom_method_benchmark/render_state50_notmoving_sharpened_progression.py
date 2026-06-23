#!/usr/bin/env python3
"""Render sharpened state-50 method progression with the not-moving mask.

The input method volumes are sharpened in Fourier space, then multiplied by the
static/not-moving soft mask before ChimeraX rendering.  This keeps the original
volumes untouched while making a reproducible render package for slides/movies.
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

import mrcfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY_CSV = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "not_moving_softmask_fsc_resolution_20260601/"
    "not_moving_softmask_noise3_fsc_summary.csv"
)
DEFAULT_VIEW_JSON = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "chimerax_state50_method_progression_notmoving_view_20260601/"
    "not_moving_view_extracted.json"
)
DEFAULT_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/not_moving_spike_mask_soft_20260601.mrc"
)
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_method_sweep_noise3_b80_20260531/"
    "chimerax_state50_method_progression_notmoving_sharpenB80_20260601"
)
RENDERER = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "spike_fullatom_method_benchmark"
    / "render_chimerax_manifest.py"
)

METHOD_ORDER = ["recovar", "cryodrgn", "3dflex"]
METHOD_LABELS = {
    "recovar": "RECOVAR",
    "cryodrgn": "cryoDRGN",
    "3dflex": "3DFlex",
}
METHOD_COLORS = {
    "recovar": "#1f77b4",
    "cryodrgn": "#d62728",
    "3dflex": "#2ca02c",
}
METHOD_LEVELS = {
    "recovar": 0.0214,
    "cryodrgn": 0.0213,
    "3dflex": 0.0213,
}
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


def _load_mrc(path: Path) -> tuple[np.ndarray, object, object]:
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
        voxel_size = mrc.voxel_size
        origin = mrc.header.origin.copy()
    return data, voxel_size, origin


def _write_mrc(path: Path, data: np.ndarray, voxel_size: object, origin: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
        mrc.voxel_size = voxel_size
        mrc.header.origin = origin


def _sharpen_volume(
    volume: np.ndarray,
    voxel_size_a: float,
    sharpen_bfactor_a2: float,
    max_gain: float,
) -> np.ndarray:
    shape = volume.shape
    if len(shape) != 3 or len(set(shape)) != 1:
        raise ValueError(f"expected cubic 3D volume, got {shape}")
    freqs = np.fft.fftfreq(shape[0], d=voxel_size_a).astype(np.float32)
    fx, fy, fz = np.meshgrid(freqs, freqs, freqs, indexing="ij", sparse=True)
    radius = np.sqrt(fx * fx + fy * fy + fz * fz)
    nyquist = 0.5 / voxel_size_a
    radius = np.minimum(radius, nyquist)
    gain = np.exp((sharpen_bfactor_a2 / 4.0) * radius * radius)
    gain = np.minimum(gain, max_gain).astype(np.float32)
    sharpened = np.fft.ifftn(np.fft.fftn(volume) * gain).real
    return np.asarray(sharpened, dtype=np.float32)


def _processed_paths(out_dir: Path, entry: RenderEntry, noise_level: str) -> tuple[Path, Path]:
    n_tag = f"n{entry.n_images:08d}"
    estimate = (
        out_dir
        / "processed_mrc"
        / entry.method
        / f"state{entry.state:04d}_noise{noise_level}_{entry.method}_{n_tag}_notmoving_sharpened_masked.mrc"
    )
    gt = (
        out_dir
        / "processed_mrc"
        / "ground_truth"
        / n_tag
        / f"gt_state{entry.state:04d}_{n_tag}_notmoving_sharpened_masked.mrc"
    )
    return estimate, gt


def _prepare_processed_volumes(args: argparse.Namespace, entries: list[RenderEntry]) -> dict[tuple[str, int], tuple[Path, Path]]:
    mask, _, _ = _load_mrc(args.mask)
    mask = np.asarray(mask, dtype=np.float32)
    out: dict[tuple[str, int], tuple[Path, Path]] = {}
    for entry in entries:
        for path in (entry.estimate, entry.gt):
            if not path.exists():
                raise FileNotFoundError(path)
        estimate_out, gt_out = _processed_paths(args.output_dir, entry, args.noise_level)
        if not estimate_out.exists():
            estimate, voxel_size, origin = _load_mrc(entry.estimate)
            if estimate.shape != mask.shape:
                raise ValueError(f"mask shape {mask.shape} does not match {entry.estimate}: {estimate.shape}")
            sharpened = _sharpen_volume(estimate, args.voxel_size, args.sharpen_bfactor, args.max_gain)
            _write_mrc(estimate_out, sharpened * mask, voxel_size, origin)
        if not gt_out.exists():
            gt, voxel_size, origin = _load_mrc(entry.gt)
            if gt.shape != mask.shape:
                raise ValueError(f"mask shape {mask.shape} does not match {entry.gt}: {gt.shape}")
            sharpened_gt = _sharpen_volume(gt, args.voxel_size, args.sharpen_bfactor, args.max_gain)
            _write_mrc(gt_out, sharpened_gt * mask, voxel_size, origin)
        out[(entry.method, entry.n_images)] = (estimate_out, gt_out)
    return out


def _write_manifest(
    args: argparse.Namespace,
    entries: list[RenderEntry],
    processed: dict[tuple[str, int], tuple[Path, Path]],
) -> Path:
    manifest = args.output_dir / "manifest_chimerax_state50_notmoving_sharpened.csv"
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
        "source_volume_path",
        "mask_path",
        "render_name",
        "contour_level",
        "color",
        "sharpen_bfactor_A2",
        "max_gain",
    ]
    rows: list[dict[str, str]] = []
    collection = f"noise{args.noise_level}_b{args.bfactor}_state50_notmoving_sharpened"
    for entry in entries:
        estimate_out, gt_out = processed[(entry.method, entry.n_images)]
        n_tag = f"n{entry.n_images:08d}"
        rows.append(
            {
                "noise_level": args.noise_level,
                "bfactor": args.bfactor,
                "collection": collection,
                "n_images": str(entry.n_images),
                "n_label": entry.n_label,
                "state": str(entry.state),
                "method": entry.method,
                "method_label": METHOD_LABELS[entry.method],
                "role": "estimate",
                "volume_path": str(estimate_out),
                "source_volume_path": str(entry.estimate),
                "mask_path": str(args.mask),
                "render_name": f"{entry.method}/state{entry.state:04d}_noise{args.noise_level}_{entry.method}_{n_tag}_notmoving_sharpened.png",
                "contour_level": f"{METHOD_LEVELS[entry.method]:.8g}",
                "color": METHOD_COLORS[entry.method],
                "sharpen_bfactor_A2": f"{args.sharpen_bfactor:.8g}",
                "max_gain": f"{args.max_gain:.8g}",
            }
        )
        rows.append(
            {
                "noise_level": args.noise_level,
                "bfactor": args.bfactor,
                "collection": collection,
                "n_images": str(entry.n_images),
                "n_label": entry.n_label,
                "state": str(entry.state),
                "method": "ground_truth",
                "method_label": "GT",
                "role": "gt",
                "volume_path": str(gt_out),
                "source_volume_path": str(entry.gt),
                "mask_path": str(args.mask),
                "render_name": f"ground_truth/state{entry.state:04d}_noise{args.noise_level}_gt_{n_tag}_notmoving_sharpened.png",
                "contour_level": f"{args.gt_contour_level:.8g}",
                "color": args.gt_color,
                "sharpen_bfactor_A2": f"{args.sharpen_bfactor:.8g}",
                "max_gain": f"{args.max_gain:.8g}",
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
        f"--view-json {args.view_json} "
        f"--width {args.width} "
        f"--height {args.height} "
        f"--supersample {args.supersample} "
        f"--background {shlex.quote(args.background)} "
        f"--roles estimate "
        f"--states {args.state} "
        f"--overlay-gt "
        f"--gt-color {shlex.quote(args.gt_color)} "
        f"--gt-opacity {args.gt_opacity} "
        f"--fallback-level {args.gt_contour_level}"
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
    subtitle = (
        f"not-moving mask; sharpen +{args.sharpen_bfactor:g} A^2 "
        f"(gain cap {args.max_gain:g}); GT gray opacity {args.gt_opacity:.1f}"
    )
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


def _contact_sheet(index: dict[tuple[str, int], Path], panels_dir: Path, noise_level: str, state: int) -> Path:
    panels_dir.mkdir(parents=True, exist_ok=True)
    cell = (420, 315)
    left = 170
    top = 80
    sheet = Image.new("RGB", (left + cell[0] * len(N_ORDER), top + cell[1] * len(METHOD_ORDER)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 18), f"Spike benchmark noise={noise_level}, state {state}: sharpened not-moving region over GT", fill=(0, 0, 0), font=FONT_MED)
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
    output = panels_dir / f"state{state:04d}_noise{noise_level}_notmoving_sharpened_method_by_n_contact_sheet.png"
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


def _progression_frames(index: dict[tuple[str, int], Path], frames_dir: Path, noise_level: str, state: int) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cell = (640, 480)
    left = 160
    top = 70
    frames = []
    for n_images in N_ORDER:
        frame = Image.new("RGB", (left + cell[0], top + len(METHOD_ORDER) * cell[1]), "white")
        draw = ImageDraw.Draw(frame)
        draw.text((20, 18), f"noise={noise_level}  state {state}  sharpened not-moving  n={N_LABELS[n_images]}", fill=(0, 0, 0), font=FONT_MED)
        for i, method in enumerate(METHOD_ORDER):
            y = top + i * cell[1]
            draw.text((20, y + cell[1] // 2 - 14), METHOD_LABELS[method], fill=(0, 0, 0), font=FONT_CELL)
            path = index.get((method, n_images))
            tile = _thumb(path, cell) if path else _placeholder(cell, "not available")
            frame.paste(tile, (left, y))
            draw.rectangle((left, y, left + cell[0] - 1, y + cell[1] - 1), outline=(210, 210, 210), width=1)
        output = frames_dir / f"all_methods_n{n_images:08d}.png"
        frame.save(output, quality=95)
        frames.append(output)
    return frames


def _compose(args: argparse.Namespace, manifest: Path) -> dict[str, object]:
    raw_dir = args.output_dir / "png_raw"
    labeled_dir = args.output_dir / "png_labeled"
    panels_dir = args.output_dir / "panels"
    animations_dir = args.output_dir / "animations"
    frames_dir = args.output_dir / "animation_frames"
    rows = _read_estimate_rows(manifest)
    labeled = [_label_one(raw_dir, labeled_dir, row, args) for row in rows]
    index = {(row["method"], int(row["n_images"])): _labeled_path(labeled_dir, row) for row in rows}
    contact = _contact_sheet(index, panels_dir, args.noise_level, args.state)
    per_method = {}
    for method in METHOD_ORDER:
        paths = [index[(method, n_images)] for n_images in N_ORDER if (method, n_images) in index]
        output = animations_dir / f"state{args.state:04d}_noise{args.noise_level}_notmoving_sharpened_{method}_progression.gif"
        _save_gif(paths, output, size=(960, 720))
        if output.exists():
            per_method[method] = str(output)
    frames = _progression_frames(index, frames_dir, args.noise_level, args.state)
    all_gif = animations_dir / f"state{args.state:04d}_noise{args.noise_level}_notmoving_sharpened_all_methods_progression.gif"
    _save_gif(frames, all_gif, size=(800, 1800))
    all_mp4 = animations_dir / f"state{args.state:04d}_noise{args.noise_level}_notmoving_sharpened_all_methods_progression.mp4"
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
        "all_method_frames": [str(path) for path in frames],
        "duration_ms": DURATION_MS,
        "ffmpeg_returncode": proc.returncode,
        "ffmpeg_command": command,
        "ffmpeg_log": str(ffmpeg_log),
    }


def _write_audit(
    args: argparse.Namespace,
    entries: list[RenderEntry],
    manifest: Path,
    outputs: dict[str, object] | None,
    render_log: Path | None,
) -> Path:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.method] = counts.get(entry.method, 0) + 1
    missing = {
        method: [N_LABELS[n] for n in N_ORDER if not any(e.method == method and e.n_images == n for e in entries)]
        for method in METHOD_ORDER
    }
    audit = args.output_dir / "COMMAND_AUDIT.md"
    lines = [
        "# State 50 Not-Moving Sharpened ChimeraX Render Audit",
        "",
        f"- Repo script: `{Path(__file__).resolve()}`",
        f"- Summary CSV: `{args.summary_csv}`",
        f"- Output dir: `{args.output_dir}`",
        f"- Static mask: `{args.mask}`",
        f"- View JSON: `{args.view_json}`",
        f"- ChimeraX renderer: `{RENDERER}`",
        f"- State: `{args.state}`",
        f"- Sharpening: inverse B-factor `+{args.sharpen_bfactor:g} A^2`, max gain `{args.max_gain:g}`",
        f"- Contour levels: `{json.dumps(METHOD_LEVELS, sort_keys=True)}`, GT `{args.gt_contour_level}`",
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
    parser.add_argument("--view-json", type=Path, default=DEFAULT_VIEW_JSON)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--state", type=int, default=50)
    parser.add_argument("--noise-level", default="3")
    parser.add_argument("--bfactor", default="80")
    parser.add_argument("--voxel-size", type=float, default=1.25)
    parser.add_argument("--sharpen-bfactor", type=float, default=80.0)
    parser.add_argument("--max-gain", type=float, default=25.0)
    parser.add_argument("--gt-contour-level", type=float, default=0.0194)
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
    args.noise_level = str(args.noise_level)
    args.bfactor = str(args.bfactor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = _read_entries(args.summary_csv, args.state)
    if not entries:
        raise RuntimeError(f"no state {args.state} entries found in {args.summary_csv}")
    if not args.mask.exists():
        raise FileNotFoundError(args.mask)
    processed = _prepare_processed_volumes(args, entries)
    manifest = _write_manifest(args, entries, processed)
    render_log = _run_chimerax(args, manifest) if args.run_chimerax else None
    outputs = None if args.skip_compose else _compose(args, manifest)
    if outputs is not None and render_log is not None:
        outputs["chimerax_log"] = str(render_log)
    audit = _write_audit(args, entries, manifest, outputs, render_log)
    summary = {
        "output_dir": str(args.output_dir),
        "manifest": str(manifest),
        "audit": str(audit),
        "render_log": str(render_log) if render_log is not None else None,
        "n_entries": len(entries),
        "outputs": outputs,
    }
    (args.output_dir / "render_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
