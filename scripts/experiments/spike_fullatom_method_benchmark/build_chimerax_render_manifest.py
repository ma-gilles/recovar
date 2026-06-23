#!/usr/bin/env python3
"""Build a manifest of spike sweep volumes to render with ChimeraX.

The manifest is intentionally plain CSV so rendering can be rerun without
rediscovering outputs.  It points at unfiltered estimate maps where available
and includes matching GT maps as separate rows.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_ROOTS = {
    "noise1_method": (
        1,
        80,
        Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise1_b80_20260530"),
    ),
    "noise3_method": (
        3,
        80,
        Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531"),
    ),
    "noise10_recovar": (
        10,
        100,
        Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_true_pipeline_sweep_noise10_b100_dev2_20260529"),
    ),
    "noise10_methods": (
        10,
        100,
        Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise10_b100_20260528"),
    ),
    "noise30_recovar": (
        30,
        80,
        Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise30_b80_20260528"),
    ),
}

METHOD_COLORS = {
    "recovar": "#3b6fb6",
    "cryodrgn": "#2aa876",
    "3dflex": "#d88a24",
    "ground_truth": "#777777",
}


@dataclass(frozen=True)
class RenderRow:
    collection: str
    noise_level: int
    bfactor: int
    method: str
    n_images: int
    state: int
    role: str
    volume_path: Path
    source_csv: Path
    render_name: str
    color: str
    contour_level: float | None
    data_mean: float | None
    data_std: float | None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _n_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"n(\d{8})", part)
        if match:
            return int(match.group(1))
    return None


def _state_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"state(\d{4})", part)
        if match:
            return int(match.group(1))
    return None


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _estimate_level(
    path: Path,
    percentile: float,
    sigma: float,
    max_voxels: int,
) -> tuple[float | None, float | None, float | None]:
    try:
        import mrcfile
    except Exception:
        return None, None, None

    try:
        with mrcfile.open(path, permissive=True) as mrc:
            data = np.asarray(mrc.data, dtype=np.float32)
            if data.size > max_voxels:
                step = int(np.ceil((data.size / max_voxels) ** (1.0 / max(data.ndim, 1))))
                data = data[tuple(slice(None, None, step) for _ in range(data.ndim))]
            values = np.asarray(data, dtype=np.float64).ravel()
    except Exception:
        return None, None, None

    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None, None
    mean = float(values.mean())
    std = float(values.std())
    positive = values[values > 0]
    level_values = positive if positive.size > 100 else values
    percentile_level = float(np.percentile(level_values, percentile))
    sigma_level = mean + sigma * std
    level = max(percentile_level, sigma_level)
    return level, mean, std


def _make_name(collection: str, noise: int, bfactor: int, method: str, n_images: int, state: int, role: str) -> str:
    return f"noise{noise:02d}_b{bfactor}_{collection}_{method}_n{n_images:08d}_state{state:04d}_{role}.png"


def _row(
    *,
    noise_level: int,
    bfactor: int,
    collection: str,
    method: str,
    n_images: int,
    state: int,
    role: str,
    volume_path: Path,
    source_csv: Path,
    compute_levels: bool,
    level_percentile: float,
    level_sigma: float,
    max_level_voxels: int,
) -> RenderRow | None:
    volume_path = volume_path.resolve()
    if not volume_path.exists():
        return None
    level = mean = std = None
    if compute_levels:
        level, mean, std = _estimate_level(volume_path, level_percentile, level_sigma, max_level_voxels)
    return RenderRow(
        collection=collection,
        noise_level=noise_level,
        bfactor=bfactor,
        method=method,
        n_images=n_images,
        state=state,
        role=role,
        volume_path=volume_path,
        source_csv=source_csv.resolve(),
        render_name=_make_name(collection, noise_level, bfactor, method, n_images, state, role),
        color=METHOD_COLORS.get(method, "#8a8a8a"),
        contour_level=level,
        data_mean=mean,
        data_std=std,
    )


def _collect_recovar_benchmark_root(
    root: Path,
    noise_level: int,
    bfactor: int,
    collection: str,
    states: set[int],
    compute_levels: bool,
    level_percentile: float,
    level_sigma: float,
    max_level_voxels: int,
) -> Iterable[RenderRow]:
    for summary in sorted(root.glob("recovar/n*/metrics_zdim4_noreg_broadmask/state*/summary.csv")):
        n_images = _n_from_path(summary)
        state = _state_from_path(summary)
        if n_images is None or state is None or state not in states:
            continue
        for entry in _read_csv(summary):
            for method, role, key in (
                ("recovar", "estimate", "estimate"),
                ("ground_truth", "gt", "gt"),
            ):
                path = entry.get(key)
                if not path:
                    continue
                row = _row(
                    noise_level=noise_level,
                    bfactor=bfactor,
                    collection=collection,
                    method=method,
                    n_images=n_images,
                    state=state,
                    role=role,
                    volume_path=Path(path),
                    source_csv=summary,
                    compute_levels=compute_levels,
                    level_percentile=level_percentile,
                    level_sigma=level_sigma,
                    max_level_voxels=max_level_voxels,
                )
                if row is not None:
                    yield row


def _collect_recovar_run_root(
    root: Path,
    noise_level: int,
    bfactor: int,
    collection: str,
    states: set[int],
    compute_levels: bool,
    level_percentile: float,
    level_sigma: float,
    max_level_voxels: int,
) -> Iterable[RenderRow]:
    for summary in sorted(root.glob("n*/runs/*/08_metrics_zdim4_noreg/state*/summary.csv")):
        n_images = _n_from_path(summary)
        state = _state_from_path(summary)
        if n_images is None or state is None or state not in states:
            continue
        for entry in _read_csv(summary):
            for method, role, key in (
                ("recovar", "estimate", "estimate"),
                ("ground_truth", "gt", "gt"),
            ):
                path = entry.get(key)
                if not path:
                    continue
                row = _row(
                    noise_level=noise_level,
                    bfactor=bfactor,
                    collection=collection,
                    method=method,
                    n_images=n_images,
                    state=state,
                    role=role,
                    volume_path=Path(path),
                    source_csv=summary,
                    compute_levels=compute_levels,
                    level_percentile=level_percentile,
                    level_sigma=level_sigma,
                    max_level_voxels=max_level_voxels,
                )
                if row is not None:
                    yield row


def _collect_recovar_flat_summary(
    root: Path,
    noise_level: int,
    bfactor: int,
    collection: str,
    states: set[int],
    compute_levels: bool,
    level_percentile: float,
    level_sigma: float,
    max_level_voxels: int,
) -> Iterable[RenderRow]:
    summary = root / "recovar" / "recovar_state_sweep_summary.csv"
    if not summary.exists():
        return
    for entry in _read_csv(summary):
        state = int(float(entry["state"]))
        if state not in states:
            continue
        n_images = int(float(entry["n_images"]))
        for method, role, key in (
            ("recovar", "estimate", "estimate"),
            ("ground_truth", "gt", "gt"),
        ):
            path = entry.get(key)
            if not path:
                continue
            row = _row(
                noise_level=noise_level,
                bfactor=bfactor,
                collection=collection,
                method=method,
                n_images=n_images,
                state=state,
                role=role,
                volume_path=Path(path),
                source_csv=summary,
                compute_levels=compute_levels,
                level_percentile=level_percentile,
                level_sigma=level_sigma,
                max_level_voxels=max_level_voxels,
            )
            if row is not None:
                yield row


def _collect_cryodrgn(
    root: Path,
    noise_level: int,
    bfactor: int,
    collection: str,
    states: set[int],
    compute_levels: bool,
    level_percentile: float,
    level_sigma: float,
    max_level_voxels: int,
) -> Iterable[RenderRow]:
    for metrics in sorted(root.glob("n*/evaluation/metrics/decoded_volume_metrics.csv")):
        n_images = _n_from_path(metrics)
        if n_images is None:
            continue
        for entry in _read_csv(metrics):
            if entry.get("method") != "cryodrgn" or entry.get("metric_mask") != "broad":
                continue
            state = int(float(entry["gt_label"]))
            if state not in states:
                continue
            row = _row(
                noise_level=noise_level,
                bfactor=bfactor,
                collection=collection,
                method="cryodrgn",
                n_images=n_images,
                state=state,
                role="estimate",
                volume_path=Path(entry["volume"]),
                source_csv=metrics,
                compute_levels=compute_levels,
                level_percentile=level_percentile,
                level_sigma=level_sigma,
                max_level_voxels=max_level_voxels,
            )
            if row is not None:
                yield row


def _collect_3dflex(
    root: Path,
    noise_level: int,
    bfactor: int,
    collection: str,
    states: set[int],
    compute_levels: bool,
    level_percentile: float,
    level_sigma: float,
    max_level_voxels: int,
) -> Iterable[RenderRow]:
    for metrics in sorted(root.glob("evaluation_3dflex_mean_latents/n*/metrics/3dflex_mean_latent_vs_gt_metrics.csv")):
        n_images = _n_from_path(metrics)
        if n_images is None:
            continue
        for entry in _read_csv(metrics):
            if entry.get("metric_mask") != "broad":
                continue
            state = int(float(entry["state"]))
            if state not in states:
                continue
            row = _row(
                noise_level=noise_level,
                bfactor=bfactor,
                collection=collection,
                method="3dflex",
                n_images=n_images,
                state=state,
                role="estimate",
                volume_path=Path(entry["volume_path"]),
                source_csv=metrics,
                compute_levels=compute_levels,
                level_percentile=level_percentile,
                level_sigma=level_sigma,
                max_level_voxels=max_level_voxels,
            )
            if row is not None:
                yield row


def _parse_root_specs(specs: list[str]) -> dict[str, tuple[int, int, Path]]:
    roots = dict(DEFAULT_ROOTS)
    for spec in specs:
        try:
            name, noise, bfactor, path = spec.split(":", 3)
        except ValueError as exc:
            raise SystemExit(
                f"Bad --root {spec!r}; expected name:noise_level:bfactor:/path"
            ) from exc
        roots[name] = (int(noise), int(bfactor), Path(path))
    return roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--states", default="0,25,50", help="Comma-separated GT state labels.")
    parser.add_argument(
        "--methods",
        default="recovar,cryodrgn,3dflex,ground_truth",
        help="Comma-separated methods to include.",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Add/override root as name:noise_level:bfactor:/path.",
    )
    parser.add_argument("--level-percentile", type=float, default=70.0)
    parser.add_argument("--level-sigma", type=float, default=1.5)
    parser.add_argument("--max-level-voxels", type=int, default=2_000_000)
    parser.add_argument("--no-compute-levels", action="store_true")
    args = parser.parse_args()

    states = {int(x) for x in args.states.split(",") if x.strip()}
    methods = {x.strip() for x in args.methods.split(",") if x.strip()}
    roots = _parse_root_specs(args.root)
    compute_levels = not args.no_compute_levels

    rows: list[RenderRow] = []
    for name, (noise_level, bfactor, root) in roots.items():
        if not root.exists():
            continue
        if "recovar" in methods or "ground_truth" in methods:
            rows.extend(
                _collect_recovar_benchmark_root(
                    root,
                    noise_level,
                    bfactor,
                    name,
                    states,
                    compute_levels,
                    args.level_percentile,
                    args.level_sigma,
                    args.max_level_voxels,
                )
            )
            rows.extend(
                _collect_recovar_run_root(
                    root,
                    noise_level,
                    bfactor,
                    name,
                    states,
                    compute_levels,
                    args.level_percentile,
                    args.level_sigma,
                    args.max_level_voxels,
                )
            )
            rows.extend(
                _collect_recovar_flat_summary(
                    root,
                    noise_level,
                    bfactor,
                    name,
                    states,
                    compute_levels,
                    args.level_percentile,
                    args.level_sigma,
                    args.max_level_voxels,
                )
            )
        if "cryodrgn" in methods:
            rows.extend(
                _collect_cryodrgn(
                    root,
                    noise_level,
                    bfactor,
                    name,
                    states,
                    compute_levels,
                    args.level_percentile,
                    args.level_sigma,
                    args.max_level_voxels,
                )
            )
        if "3dflex" in methods:
            rows.extend(
                _collect_3dflex(
                    root,
                    noise_level,
                    bfactor,
                    name,
                    states,
                    compute_levels,
                    args.level_percentile,
                    args.level_sigma,
                    args.max_level_voxels,
                )
            )

    # Deduplicate rows created from overlapping roots/summaries.
    dedup: dict[tuple[int, int, str, int, int, str, str], RenderRow] = {}
    for row in rows:
        if row.method not in methods:
            continue
        key = (
            row.noise_level,
            row.bfactor,
            row.collection,
            row.method,
            row.n_images,
            row.state,
            row.role,
            str(row.volume_path),
        )
        dedup.setdefault(key, row)
    rows = sorted(
        dedup.values(),
        key=lambda r: (r.noise_level, r.bfactor, r.method, r.n_images, r.state, r.role, str(r.volume_path)),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "noise_level",
        "bfactor",
        "collection",
        "method",
        "n_images",
        "state",
        "role",
        "volume_path",
        "source_csv",
        "render_name",
        "color",
        "contour_level",
        "data_mean",
        "data_std",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "noise_level": row.noise_level,
                    "bfactor": row.bfactor,
                    "collection": row.collection,
                    "method": row.method,
                    "n_images": row.n_images,
                    "state": row.state,
                    "role": row.role,
                    "volume_path": str(row.volume_path),
                    "source_csv": str(row.source_csv),
                    "render_name": row.render_name,
                    "color": row.color,
                    "contour_level": "" if row.contour_level is None else f"{row.contour_level:.8g}",
                    "data_mean": "" if row.data_mean is None else f"{row.data_mean:.8g}",
                    "data_std": "" if row.data_std is None else f"{row.data_std:.8g}",
                }
            )
    print(f"wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
