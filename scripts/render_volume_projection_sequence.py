#!/usr/bin/env python3
"""
Render projection images from a sequence of 3D volumes.

By default it expects files like:
  <volume_prefix>0000.mrc, <volume_prefix>0001.mrc, ...

It can also auto-generate compact-support test volumes using
recovar.commands.run_test_all_metrics.generate_compact_support_test_volumes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.commands.run_test_all_metrics import generate_compact_support_test_volumes


def discover_n_volumes(prefix: str, max_try: int = 2000) -> int:
    n = 0
    while n < max_try and Path(f"{prefix}{n:04d}.mrc").exists():
        n += 1
    return n


def render_projection(vol: np.ndarray, axis: int = 2) -> np.ndarray:
    proj = np.sum(vol, axis=axis)
    proj = proj - np.mean(proj)
    denom = np.std(proj)
    if denom > 0:
        proj = proj / denom
    return proj


def main() -> None:
    p = argparse.ArgumentParser(description="Render projection sequence from volume stack.")
    p.add_argument("--volume-prefix", default=None, help="Volume prefix, e.g. /path/to/vol")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-volumes", type=int, default=None, help="If omitted, auto-discovers from prefix.")
    p.add_argument("--projection-axis", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--generate-if-missing", action="store_true", help="Generate synthetic volumes if prefix is missing.")
    p.add_argument("--generate-grid-size", type=int, default=128)
    p.add_argument("--generate-n-volumes", type=int, default=50)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    volume_prefix = args.volume_prefix
    if volume_prefix is None:
        volume_prefix = str(out_dir / "generated_volumes" / "vol")
        args.generate_if_missing = True

    if args.generate_if_missing and not Path(f"{volume_prefix}0000.mrc").exists():
        generate_compact_support_test_volumes(
            output_dir=str(Path(volume_prefix).parent),
            grid_size=args.generate_grid_size,
            n_volumes=args.generate_n_volumes,
            voxel_size=4.25 * 128 / args.generate_grid_size,
            output_prefix=volume_prefix,
        )

    if args.n_volumes is None:
        n_volumes = discover_n_volumes(volume_prefix)
    else:
        n_volumes = args.n_volumes
    if n_volumes <= 0:
        raise ValueError(f"No volumes found for prefix {volume_prefix}")

    projections = []
    vmin = np.inf
    vmax = -np.inf
    for i in range(n_volumes):
        vol = utils.load_mrc(f"{volume_prefix}{i:04d}.mrc")
        proj = render_projection(vol, axis=args.projection_axis)
        projections.append(proj)
        vmin = min(vmin, float(np.min(proj)))
        vmax = max(vmax, float(np.max(proj)))

    seq_dir = out_dir / "projection_frames"
    seq_dir.mkdir(parents=True, exist_ok=True)
    for i, proj in enumerate(projections):
        plt.figure(figsize=(4, 4))
        plt.imshow(proj, cmap="gray", vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(seq_dir / f"proj_{i:04d}.png", dpi=120, bbox_inches="tight", pad_inches=0)
        plt.close()

    # Save an overview grid for quick visual inspection.
    n_cols = 10
    n_rows = int(np.ceil(n_volumes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.axis("off")
        if i < n_volumes:
            ax.imshow(projections[i], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{i:02d}", fontsize=8)
    plt.tight_layout()
    overview_path = out_dir / "projection_overview.png"
    plt.savefig(overview_path, dpi=150)
    plt.close(fig)

    print(f"volume_prefix={volume_prefix}")
    print(f"n_volumes={n_volumes}")
    print(f"frames_dir={seq_dir}")
    print(f"overview_png={overview_path}")


if __name__ == "__main__":
    main()
