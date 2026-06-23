#!/usr/bin/env python3
"""Score the nonuniform spike benchmark with solid trajectory masks.

This is the follow-up to ``score_nonuniform_corrected_masks.py``.  The first
corrected tracked-atom masks were atom-ball masks and were visually holey.  The
mask used here is the v13 construction:

1. Advect the state-50 broad mask along the PDB trajectory.
2. Take the compact hard core of that advected mask.
3. Gaussian-smooth that hard support only as a morphology cleanup step.
4. Threshold, apply a small binary close, and fill enclosed cavities.
5. Apply the final raised-cosine soft edge.

The final scoring mask is cosine-softened; the Gaussian field is only an
intermediate binary-support cleanup.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from recovar import utils

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.spike_fullatom_method_benchmark.score_nonuniform_corrected_masks import (
    DEFAULT_BENCH_ROOT,
    DEFAULT_BROAD_MASK,
    DEFAULT_FLEX_PROJECT_DIR,
    DEFAULT_PDB_DIR,
    DEFAULT_SOURCE_RUN,
    DEFAULT_STATES,
    build_masks,
    coords_to_voxels,
    parse_pdb_coords,
    parse_states,
    pdb_for_state,
    plot_curves,
    plot_resolution,
    score,
    write_csv,
)


def largest_component(binary: np.ndarray) -> np.ndarray:
    labels, n_components = ndimage.label(binary)
    if n_components == 0:
        return np.asarray(binary, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def fill_enclosed_cavities(binary: np.ndarray) -> np.ndarray:
    binary = largest_component(np.asarray(binary, dtype=bool))
    labels, n_components = ndimage.label(~binary)
    if n_components == 0:
        return binary
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    outside = labels == int(np.argmax(counts))
    return largest_component(~outside)


def distance_dilate(binary: np.ndarray, radius: float) -> np.ndarray:
    return ndimage.distance_transform_edt(~binary) <= radius


def distance_erode(binary: np.ndarray, radius: float) -> np.ndarray:
    return ndimage.distance_transform_edt(binary) > radius


def distance_close(binary: np.ndarray, radius: float) -> np.ndarray:
    return distance_erode(distance_dilate(binary, radius), radius)


def cosine_soft_mask(binary: np.ndarray, edge_voxels: float) -> np.ndarray:
    binary = np.asarray(binary, dtype=bool)
    distance = ndimage.distance_transform_edt(~binary)
    mask = np.zeros(binary.shape, dtype=np.float32)
    mask[binary] = 1.0
    edge = (~binary) & (distance <= edge_voxels)
    mask[edge] = 0.5 * (1.0 + np.cos(np.pi * distance[edge] / edge_voxels))
    return mask


def advect_broad_mask(
    broad_mask: np.ndarray,
    *,
    pdb_dir: Path,
    reference_state: int,
    state: int,
    voxel_size: float,
    atom_selection_threshold: float,
) -> np.ndarray:
    ref_coords = parse_pdb_coords(pdb_for_state(pdb_dir, reference_state))
    state_coords = parse_pdb_coords(pdb_for_state(pdb_dir, state))
    if state_coords.shape != ref_coords.shape:
        raise ValueError(
            f"State {state} has {state_coords.shape[0]} atoms, "
            f"reference has {ref_coords.shape[0]}"
        )

    ref_center = ref_coords.mean(axis=0, dtype=np.float64).astype(np.float32)
    state_center = state_coords.mean(axis=0, dtype=np.float64).astype(np.float32)
    ref_voxels = coords_to_voxels(ref_coords, ref_center, broad_mask.shape, voxel_size).astype(
        np.float32
    )
    state_voxels = coords_to_voxels(
        state_coords, state_center, broad_mask.shape, voxel_size
    ).astype(np.float32)

    ref_points = np.rint(ref_voxels).astype(np.int32)
    in_bounds = np.all(
        (ref_points >= 0) & (ref_points < np.asarray(broad_mask.shape).reshape(1, 3)),
        axis=1,
    )
    ref_values = np.zeros(ref_points.shape[0], dtype=np.float32)
    ref_values[in_bounds] = broad_mask[tuple(ref_points[in_bounds].T)]
    selected = np.flatnonzero(in_bounds & (ref_values > atom_selection_threshold))
    if selected.size == 0:
        raise RuntimeError("No atoms selected by broad-mask threshold")

    nonzero = np.argwhere(broad_mask > 0)
    values = broad_mask[tuple(nonzero.T)]
    tree = cKDTree(ref_voxels[selected])
    _, nearest = tree.query(nonzero.astype(np.float32), k=1)
    atom_indices = selected[nearest]
    target = nonzero.astype(np.float32) + (state_voxels[atom_indices] - ref_voxels[atom_indices])

    advected = np.zeros_like(broad_mask, dtype=np.float32)
    base = np.floor(target).astype(np.int32)
    frac = target - base
    shape = np.asarray(broad_mask.shape).reshape(1, 3)
    for dz in (0, 1):
        wz = (1.0 - frac[:, 0]) if dz == 0 else frac[:, 0]
        for dy in (0, 1):
            wy = (1.0 - frac[:, 1]) if dy == 0 else frac[:, 1]
            for dx in (0, 1):
                wx = (1.0 - frac[:, 2]) if dx == 0 else frac[:, 2]
                idx = base + np.array([dz, dy, dx], dtype=np.int32)
                ok = np.all((idx >= 0) & (idx < shape), axis=1)
                if np.any(ok):
                    np.add.at(advected, tuple(idx[ok].T), values[ok] * wz[ok] * wy[ok] * wx[ok])
    return np.clip(advected, 0.0, 1.0)


def make_v13_mask(
    advected_mask: np.ndarray,
    *,
    core_threshold: float,
    gaussian_sigma_voxels: float,
    gaussian_threshold: float,
    close_radius_voxels: float,
    soft_edge_voxels: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    core = advected_mask >= core_threshold
    core = fill_enclosed_cavities(core)
    cleanup_field = ndimage.gaussian_filter(core.astype(np.float32), sigma=gaussian_sigma_voxels)
    thresholded = largest_component(cleanup_field >= gaussian_threshold)
    closed = largest_component(distance_close(thresholded, close_radius_voxels))
    hard = fill_enclosed_cavities(closed)
    soft = cosine_soft_mask(hard, soft_edge_voxels)
    return soft, {
        "advected": advected_mask,
        "core": core.astype(np.float32),
        "cleanup_field": cleanup_field.astype(np.float32),
        "thresholded": thresholded.astype(np.float32),
        "closed": closed.astype(np.float32),
        "hard": hard.astype(np.float32),
    }


def build_v13_masks(args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Path], dict[int, Path], dict[str, object]]:
    moving_masks, _old_tracked_masks, old_manifest = build_masks(args, out_dir)

    mask_dir = out_dir / "masks"
    broad = np.clip(np.asarray(utils.load_mrc(args.broad_mask), dtype=np.float32), 0.0, 1.0)
    tracked: dict[int, Path] = {}
    rows: list[dict[str, object]] = []
    for state in args.states:
        advected = advect_broad_mask(
            broad,
            pdb_dir=args.pdb_dir,
            reference_state=args.reference_state,
            state=state,
            voxel_size=args.voxel_size,
            atom_selection_threshold=args.broad_threshold,
        )
        mask, intermediates = make_v13_mask(
            advected,
            core_threshold=args.core_threshold,
            gaussian_sigma_voxels=args.gaussian_sigma_voxels,
            gaussian_threshold=args.gaussian_threshold,
            close_radius_voxels=args.close_radius_voxels,
            soft_edge_voxels=args.soft_edge_voxels,
        )

        state_prefix = mask_dir / f"state{state:04d}_v13"
        for name, volume in intermediates.items():
            utils.write_mrc(state_prefix.with_name(f"{state_prefix.name}_{name}.mrc"), volume, voxel_size=args.voxel_size)
        mask_path = state_prefix.with_name(f"{state_prefix.name}_gauss_cleanup_closed_cosine_soft.mrc")
        utils.write_mrc(mask_path, mask, voxel_size=args.voxel_size)
        tracked[state] = mask_path
        hard = intermediates["hard"] > 0.5
        labels, n_components = ndimage.label(hard)
        rows.append(
            {
                "state": state,
                "mask": str(mask_path),
                "mask_sum": float(mask.sum()),
                "hard_voxels": int(hard.sum()),
                "hard_components": int(n_components),
                "advected_sum": float(advected.sum()),
                "core_voxels": int((intermediates["core"] > 0.5).sum()),
                "thresholded_voxels": int((intermediates["thresholded"] > 0.5).sum()),
                "closed_voxels": int((intermediates["closed"] > 0.5).sum()),
            }
        )

    stats_path = mask_dir / "v13_solid_mask_stats.csv"
    with stats_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        **old_manifest,
        "tracked_mask_type": "v13_advected_broad_gauss_cleanup_close_cosine",
        "core_threshold": float(args.core_threshold),
        "gaussian_sigma_voxels": float(args.gaussian_sigma_voxels),
        "gaussian_threshold": float(args.gaussian_threshold),
        "close_radius_voxels": float(args.close_radius_voxels),
        "soft_edge_voxels": float(args.soft_edge_voxels),
        "softening": "final raised-cosine distance-transform taper; Gaussian only cleans binary support",
        "v13_stats_csv": str(stats_path),
        "tracked_masks": {str(k): str(v) for k, v in tracked.items()},
    }
    (mask_dir / "v13_mask_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return moving_masks, tracked, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--broad-mask", type=Path, default=DEFAULT_BROAD_MASK)
    parser.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    parser.add_argument("--flex-project-dir", type=Path, default=DEFAULT_FLEX_PROJECT_DIR)
    parser.add_argument("--flex-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--states", default=",".join(str(x) for x in DEFAULT_STATES))
    parser.add_argument("--reference-state", type=int, default=50)
    parser.add_argument("--broad-threshold", type=float, default=0.01)
    parser.add_argument("--core-threshold", type=float, default=0.999)
    parser.add_argument("--gaussian-sigma-voxels", type=float, default=2.0)
    parser.add_argument("--gaussian-threshold", type=float, default=0.35)
    parser.add_argument("--close-radius-voxels", type=float, default=2.0)
    parser.add_argument("--soft-edge-voxels", type=float, default=7.45)
    parser.add_argument("--atom-radius-a", type=float, default=4.0)
    parser.add_argument("--voxel-size", type=float, default=1.25)
    args = parser.parse_args()

    args.source_run = args.source_run.resolve()
    args.bench_root = args.bench_root.resolve()
    args.broad_mask = args.broad_mask.resolve()
    args.pdb_dir = args.pdb_dir.resolve()
    args.flex_project_dir = args.flex_project_dir.resolve()
    if args.flex_manifest is not None:
        args.flex_manifest = args.flex_manifest.resolve()
    args.states = parse_states(args.states)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else args.bench_root / "corrected_nonuniform_v13_solid_mask_scoring_20260605"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    moving_masks, tracked_masks, mask_manifest = build_v13_masks(args, out_dir)
    rows, skipped = score(args, out_dir, moving_masks, tracked_masks)
    write_csv(out_dir / "v13_solid_mask_metrics.csv", rows)
    write_csv(out_dir / "v13_solid_mask_skipped.csv", skipped)
    # Keep compatibility with scripts that expect the older names.
    write_csv(out_dir / "corrected_mask_metrics.csv", rows)
    write_csv(out_dir / "corrected_mask_skipped.csv", skipped)
    plot_curves(rows, out_dir)
    plot_resolution(rows, out_dir)
    summary = {
        "source_run": str(args.source_run),
        "bench_root": str(args.bench_root),
        "flex_manifest": str(args.flex_manifest) if args.flex_manifest else None,
        "out_dir": str(out_dir),
        "n_metric_rows": len(rows),
        "n_skipped": len(skipped),
        "mask_manifest": mask_manifest,
        "metrics_csv": str(out_dir / "v13_solid_mask_metrics.csv"),
        "skipped_csv": str(out_dir / "v13_solid_mask_skipped.csv"),
    }
    (out_dir / "v13_solid_mask_scoring_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
