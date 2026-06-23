#!/usr/bin/env python3
"""Render a uniform noise=3 state-50 RECOVAR/theory/GT still.

The figure uses the same moving/broad-mask style as the June 7 state-50 movie:
colored density is rendered only inside ``broad_mask.mrc`` while the full GT map
is overlaid in gray at low opacity.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
RENDERER = (
    REPO_ROOT
    / "scripts"
    / "experiments"
    / "spike_fullatom_method_benchmark"
    / "render_chimerax_manifest.py"
)

SWEEP_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531")
THEORY_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_gt_embeddings_noise1_dataset_20260601/download_noise3oracle_sigz23")
BROAD_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
)
VIEW_JSON = (
    SWEEP_ROOT
    / "chimerax_state50_method_progression_moving_mask_zoomed_view_20260601"
    / "zoomed_moving_view_extracted.json"
)
RECOVAR_3M = SWEEP_ROOT / "recovar/n03000000/compute_state_zdim4_noreg_focus/state0050/state000.mrc"
GT_3M = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise3_b80_20260531/"
    "n03000000/runs/n03000000_seed0000/04_ground_truth/gt_vol0050.mrc"
)
THEORY_STANDARD = THEORY_ROOT / "recon_standard_convolution.mrc"
THEORY_DECONV = THEORY_ROOT / "recon_deconvolution.mrc"

COLORS = {
    "recovar_3m": "#1b9e77",
    "theory_inf": "#2b6cb0",
    "gt": "#555555",
}
GT_OVERLAY_COLOR = "#8f8f8f"


@dataclass(frozen=True)
class Panel:
    key: str
    title: str
    source: Path
    color: str


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


TITLE_FONT = _font(54, True)
SUBTITLE_FONT = _font(28, False)


def _load_mrc(path: Path) -> tuple[np.ndarray, object, object]:
    with mrcfile.open(path, permissive=True) as handle:
        data = np.asarray(handle.data, dtype=np.float32)
        voxel_size = handle.voxel_size.copy()
        origin = handle.header.origin.copy()
    return data, voxel_size, origin


def _write_mrc(path: Path, data: np.ndarray, voxel_size: object, origin: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.asarray(data, dtype=np.float32))
        handle.voxel_size = voxel_size
        handle.header.origin = origin
        handle.update_header_stats()


def _prepare_masked_mrcs(panels: list[Panel], output_dir: Path) -> dict[str, Path]:
    mask, _, _ = _load_mrc(BROAD_MASK)
    out: dict[str, Path] = {}
    for panel in panels:
        volume, voxel_size, origin = _load_mrc(panel.source)
        if volume.shape != mask.shape:
            raise ValueError(f"shape mismatch for {panel.source}: volume={volume.shape}, mask={mask.shape}")
        path = output_dir / "masked_mrc" / f"{panel.key}_broad_masked.mrc"
        _write_mrc(path, volume * mask, voxel_size, origin)
        out[panel.key] = path
    return out


def _write_manifest(panels: list[Panel], masked: dict[str, Path], output_dir: Path, contour_level: float) -> Path:
    manifest = output_dir / "manifest_uniform_noise3_state50_recovar_theory_gt_still.csv"
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
    ]
    rows: list[dict[str, str]] = []
    collection = "uniform_noise3_b80_state50_recovar_theory_gt_broadmask"
    rows.append(
        {
            "noise_level": "3",
            "bfactor": "80",
            "collection": collection,
            "n_images": "3000000",
            "n_label": "3M",
            "state": "50",
            "method": "ground_truth",
            "method_label": "GT",
            "role": "gt",
            "volume_path": str(GT_3M),
            "source_volume_path": str(GT_3M),
            "mask_path": str(BROAD_MASK),
            "render_name": "gt_full_overlay.png",
            "contour_level": f"{contour_level:.8g}",
            "color": GT_OVERLAY_COLOR,
        }
    )
    for panel in panels:
        rows.append(
            {
                "noise_level": "3",
                "bfactor": "80",
                "collection": collection,
                "n_images": "3000000",
                "n_label": "3M",
                "state": "50",
                "method": panel.key,
                "method_label": panel.title,
                "role": "estimate",
                "volume_path": str(masked[panel.key]),
                "source_volume_path": str(panel.source),
                "mask_path": str(BROAD_MASK),
                "render_name": f"{panel.key}.png",
                "contour_level": f"{contour_level:.8g}",
                "color": panel.color,
            }
        )
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _run_chimerax(args: argparse.Namespace, manifest: Path) -> Path:
    raw_dir = args.output_dir / "renders_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "logs" / "chimerax_render.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        f"{RENDERER} "
        f"--manifest {manifest} "
        f"--output-dir {raw_dir} "
        f"--view-json {VIEW_JSON} "
        f"--width {args.render_width} "
        f"--height {args.render_height} "
        f"--supersample {args.supersample} "
        f"--background white "
        f"--roles estimate "
        f"--states 50 "
        f"--overlay-gt "
        f"--gt-color {GT_OVERLAY_COLOR} "
        f"--gt-opacity {args.gt_opacity:.6g} "
        f"--fallback-level {args.contour_level:.8g}"
    )
    command = (
        f"module purge; module load {shlex.quote(args.chimerax_module)}; "
        f"chimerax --nogui --offscreen --script {shlex.quote(payload)}"
    )
    with log_path.open("w") as log:
        log.write(command + "\n\n")
        proc = subprocess.run(["bash", "-lc", command], stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"ChimeraX render failed with code {proc.returncode}; see {log_path}")
    return log_path


def _rendered_path(output_dir: Path, key: str) -> Path:
    return output_dir / "renders_raw" / f"{key}_gt_overlay.png"


def _trim_white(image: Image.Image, threshold: int = 248, pad: int = 24) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    mask = np.any(arr < threshold, axis=2)
    if not np.any(mask):
        return image
    ys, xs = np.where(mask)
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


def _centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    x = left + (right - left - (bbox[2] - bbox[0])) // 2
    y = top + (bottom - top - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill="black", font=font)


def _assemble(panels: list[Panel], output_dir: Path) -> dict[str, Path]:
    tile_size = (780, 670)
    title_h = 96
    subtitle_h = 48
    margin = 34
    gutter = 38
    width = margin * 2 + len(panels) * tile_size[0] + (len(panels) - 1) * gutter
    height = title_h + subtitle_h + tile_size[1] + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, panel in enumerate(panels):
        left = margin + i * (tile_size[0] + gutter)
        title_box = (left, 0, left + tile_size[0], title_h)
        _centered_text(draw, title_box, panel.title, TITLE_FONT)
        tile = _fit_tile(_rendered_path(output_dir, panel.key), tile_size)
        canvas.paste(tile, (left, title_h + subtitle_h))
    _centered_text(
        draw,
        (0, title_h - 2, width, title_h + subtitle_h),
        "uniform noise=3, state 50, broad mask; full GT shown in gray at opacity 0.3",
        SUBTITLE_FONT,
    )
    png = output_dir / "uniform_noise3_state50_recovar3m_theory_inf_gt_broadmask_still.png"
    canvas.save(png)
    pdf = png.with_suffix(".pdf")
    canvas.save(pdf, "PDF", resolution=220.0)
    return {"png": png, "pdf": pdf}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SWEEP_ROOT / "state50_recovar3m_theory_inf_gt_broadmask_still_20260614",
    )
    parser.add_argument("--contour-level", type=float, default=0.013)
    parser.add_argument("--gt-opacity", type=float, default=0.3)
    parser.add_argument("--render-width", type=int, default=1600)
    parser.add_argument("--render-height", type=int, default=1200)
    parser.add_argument("--supersample", type=int, default=3)
    parser.add_argument("--chimerax-module", default="chimerax/1.9")
    parser.add_argument(
        "--theory-mode",
        choices=("standard", "deconvolution", "polynomial"),
        default="standard",
        help="Which infinite-images theory map to render.",
    )
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    theory_sources = {
        "standard": (THEORY_STANDARD, "theory n=inf", "standard/convolution"),
        "deconvolution": (THEORY_DECONV, "deconv n=inf", "deconvolution"),
        "polynomial": (THEORY_ROOT / "recon_polynomial_fit.mrc", "poly n=inf", "polynomial fit"),
    }
    theory_source, theory_title, theory_label = theory_sources[args.theory_mode]
    panels = [
        Panel("recovar_3m", "RECOVAR 3M", RECOVAR_3M, COLORS["recovar_3m"]),
        Panel("theory_inf", theory_title, theory_source, COLORS["theory_inf"]),
        Panel("gt", "GT", GT_3M, COLORS["gt"]),
    ]
    for path in [BROAD_MASK, VIEW_JSON, GT_3M, *[panel.source for panel in panels]]:
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masked = _prepare_masked_mrcs(panels, args.output_dir)
    manifest = _write_manifest(panels, masked, args.output_dir, args.contour_level)
    render_log = None
    if not args.skip_render:
        render_log = _run_chimerax(args, manifest)
    outputs = _assemble(panels, args.output_dir)
    if args.theory_mode != "standard":
        renamed_outputs: dict[str, Path] = {}
        for key, path in outputs.items():
            renamed = path.with_name(path.stem + f"_{args.theory_mode}" + path.suffix)
            path.replace(renamed)
            renamed_outputs[key] = renamed
        outputs = renamed_outputs
    audit = {
        "outputs": {k: str(v) for k, v in outputs.items()},
        "manifest": str(manifest),
        "render_log": str(render_log) if render_log is not None else None,
        "mask": str(BROAD_MASK),
        "view_json": str(VIEW_JSON),
        "contour_level": args.contour_level,
        "gt_opacity": args.gt_opacity,
        "panels": [
            {
                "key": panel.key,
                "title": panel.title,
                "source": str(panel.source),
                "masked_mrc": str(masked[panel.key]),
                "render": str(_rendered_path(args.output_dir, panel.key)),
            }
            for panel in panels
        ],
        "theory_note": (
            f"theory_inf uses the {theory_label} infinite-images estimator "
            "from download_noise3oracle_sigz23."
        ),
    }
    audit_path = args.output_dir / "uniform_noise3_state50_recovar3m_theory_inf_gt_broadmask_still_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
