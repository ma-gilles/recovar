#!/usr/bin/env python
"""Plot PPCA mean and PC maps for initializer and saved iterations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recovar.core import fourier_transform_utils as ftu


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _half_to_real(half_volume, volume_shape):
    full = ftu.half_volume_to_full_volume(jnp.asarray(half_volume), tuple(volume_shape))
    return np.asarray(ftu.get_idft3(full.reshape(tuple(volume_shape))).real)


def _load_stages(init_npz: Path, run_dir: Path, volume_shape):
    init = np.load(init_npz, allow_pickle=True)
    stages = [
        {
            "label": "init",
            "mu": np.asarray(init["mu"], dtype=np.float32),
            "W": np.asarray(init["W"], dtype=np.float32),
        }
    ]
    for iter_path in sorted(run_dir.glob("iter*.npz")):
        data = np.load(iter_path)
        W_half = np.asarray(data["W_half"])
        stages.append(
            {
                "label": iter_path.stem,
                "mu": _half_to_real(np.asarray(data["mu_half"]), volume_shape).astype(np.float32),
                "W": np.stack(
                    [_half_to_real(W_half[:, pc_idx], volume_shape).astype(np.float32) for pc_idx in range(W_half.shape[1])],
                    axis=0,
                ),
            }
        )
    return stages


def _central_slice(volume):
    volume = np.asarray(volume)
    return volume[volume.shape[0] // 2]


def plot_iteration_maps(*, init_npz: Path, run_dir: Path, output_dir: Path, volume_shape, q: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = _load_stages(init_npz, run_dir, tuple(volume_shape))
    n_rows = len(stages)
    n_cols = int(q) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.8 * n_rows), dpi=160, squeeze=False)
    for col in range(n_cols):
        if col == 0:
            values = np.concatenate([_central_slice(stage["mu"]).reshape(-1) for stage in stages])
        else:
            values = np.concatenate([_central_slice(stage["W"][col - 1]).reshape(-1) for stage in stages])
        vmax = float(np.percentile(np.abs(values), 99.5)) if values.size else 1.0
        vmax = max(vmax, np.finfo(np.float32).eps)
        for row, stage in enumerate(stages):
            ax = axes[row, col]
            image = _central_slice(stage["mu"] if col == 0 else stage["W"][col - 1])
            ax.imshow(image, cmap="gray" if col == 0 else "coolwarm", vmin=(-vmax if col else None), vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title("mean" if col == 0 else f"PC {col}")
            if col == 0:
                ax.set_ylabel(stage["label"])
    fig.tight_layout()
    panel_path = output_dir / "ppca_mean_pcs_by_iteration.png"
    fig.savefig(panel_path)
    plt.close(fig)

    labels = [stage["label"] for stage in stages]
    mu_rms = [float(np.sqrt(np.mean(np.asarray(stage["mu"]) ** 2))) for stage in stages]
    W_rms = np.asarray(
        [[float(np.sqrt(np.mean(np.asarray(stage["W"][pc_idx]) ** 2))) for pc_idx in range(int(q))] for stage in stages],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    x = np.arange(len(stages))
    ax.plot(x, mu_rms, marker="o", label="mean")
    for pc_idx in range(int(q)):
        ax.plot(x, W_rms[:, pc_idx], marker="o", label=f"PC {pc_idx + 1}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("real-space RMS")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    rms_path = output_dir / "ppca_mean_pcs_rms_by_iteration.png"
    fig.savefig(rms_path)
    plt.close(fig)

    summary = {
        "passed": True,
        "init_npz": init_npz,
        "run_dir": run_dir,
        "output_dir": output_dir,
        "panel_path": panel_path,
        "rms_path": rms_path,
        "stages": labels,
        "mu_rms": mu_rms,
        "W_rms": W_rms,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    return _jsonable(summary)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-npz", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--volume-shape", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--q", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = plot_iteration_maps(
        init_npz=Path(args.init_npz),
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir),
        volume_shape=tuple(args.volume_shape),
        q=int(args.q),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
