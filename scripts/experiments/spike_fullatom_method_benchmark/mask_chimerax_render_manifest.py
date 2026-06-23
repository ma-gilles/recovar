#!/usr/bin/env python3
"""Create a new render manifest whose MRCs are multiplied by moving masks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load_mrc(path: Path) -> tuple[np.ndarray, float | None]:
    import mrcfile

    with mrcfile.open(path, permissive=True) as mrc:
        voxel_size = float(mrc.voxel_size.x) if mrc.voxel_size is not None else None
        return np.asarray(mrc.data, dtype=np.float32).copy(), voxel_size


def _write_mrc(path: Path, data: np.ndarray, voxel_size: float | None) -> None:
    import mrcfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
        if voxel_size is not None:
            mrc.voxel_size = voxel_size


def _estimate_level(path: Path, percentile: float, sigma: float, max_voxels: int) -> tuple[float, float, float]:
    data, _ = _load_mrc(path)
    if data.size > max_voxels:
        step = int(np.ceil((data.size / max_voxels) ** (1.0 / max(data.ndim, 1))))
        data = data[tuple(slice(None, None, step) for _ in range(data.ndim))]
    values = np.asarray(data, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    std = float(values.std())
    positive = values[values > 0]
    level_values = positive if positive.size > 100 else values
    return max(float(np.percentile(level_values, percentile)), mean + sigma * std), mean, std


def _moving_mask_from_volume(volume_path: Path) -> Path | None:
    parts = list(volume_path.parts)
    if "04_ground_truth" in parts:
        idx = parts.index("04_ground_truth")
        run_root = Path(*parts[:idx])
        mask = run_root / "05_masks" / "focus_mask_moving.mrc"
        return mask if mask.exists() else None
    return None


def _moving_mask_from_source_csv(source_csv: Path) -> Path | None:
    try:
        with source_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    for row in rows:
        gt = row.get("gt")
        if gt:
            mask = _moving_mask_from_volume(Path(gt))
            if mask is not None:
                return mask
    return None


def _mask_for_row(row: dict[str, str]) -> Path | None:
    mask = _moving_mask_from_volume(Path(row["volume_path"]))
    if mask is not None:
        return mask
    return _moving_mask_from_source_csv(Path(row["source_csv"]))


def _selected(row: dict[str, str], methods: set[str], roles: set[str]) -> bool:
    return row["method"] in methods and row["role"] in roles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--masked-volume-dir", required=True, type=Path)
    parser.add_argument("--methods", default="recovar,ground_truth")
    parser.add_argument("--roles", default="estimate,gt")
    parser.add_argument("--level-percentile", type=float, default=70.0)
    parser.add_argument("--level-sigma", type=float, default=1.5)
    parser.add_argument("--max-level-voxels", type=int, default=2_000_000)
    args = parser.parse_args()

    methods = {x.strip() for x in args.methods.split(",") if x.strip()}
    roles = {x.strip() for x in args.roles.split(",") if x.strip()}
    with args.manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    out_rows: list[dict[str, str]] = []
    for row in rows:
        if not _selected(row, methods, roles):
            continue
        mask_path = _mask_for_row(row)
        if mask_path is None:
            print(f"skip missing moving mask: {row['volume_path']}")
            continue
        volume_path = Path(row["volume_path"])
        data, voxel_size = _load_mrc(volume_path)
        mask, _ = _load_mrc(mask_path)
        if data.shape != mask.shape:
            print(f"skip shape mismatch: {volume_path} {data.shape} vs {mask_path} {mask.shape}")
            continue
        masked_path = args.masked_volume_dir / f"{Path(row['render_name']).stem}_movingmask.mrc"
        _write_mrc(masked_path, data * mask, voxel_size)
        level, mean, std = _estimate_level(masked_path, args.level_percentile, args.level_sigma, args.max_level_voxels)
        new_row = dict(row)
        new_row["collection"] = f"{row.get('collection', 'collection')}_movingmask"
        new_row["volume_path"] = str(masked_path.resolve())
        new_row["render_name"] = f"{Path(row['render_name']).stem}_movingmask.png"
        new_row["contour_level"] = f"{level:.8g}"
        new_row["data_mean"] = f"{mean:.8g}"
        new_row["data_std"] = f"{std:.8g}"
        out_rows.append(new_row)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {args.output_manifest} with {len(out_rows)} rows")


if __name__ == "__main__":
    main()
