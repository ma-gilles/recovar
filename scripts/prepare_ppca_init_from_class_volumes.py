#!/usr/bin/env python
"""Build a PPCA refinement init NPZ from a small set of class volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovar.utils import helpers


def prepare_ppca_init_from_class_volumes(
    class_volumes: list[str | Path],
    output_dir: str | Path,
    *,
    q: int,
    voxel_size: float | None = None,
) -> Path:
    paths = [Path(p).expanduser().resolve() for p in class_volumes]
    if not paths:
        raise ValueError("at least one class volume is required")
    q = int(q)
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")

    volumes = [np.asarray(helpers.load_mrc(path), dtype=np.float32) for path in paths]
    shape = volumes[0].shape
    if any(vol.shape != shape for vol in volumes):
        shapes = [vol.shape for vol in volumes]
        raise ValueError(f"class volume shapes differ: {shapes}")

    stack = np.stack(volumes, axis=0)
    mu = np.mean(stack, axis=0, dtype=np.float64).astype(np.float32)
    centered = (stack - mu[None, ...]).reshape(len(volumes), -1).astype(np.float64)
    centered /= np.sqrt(float(len(volumes)))
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    nonzero = int(np.count_nonzero(eigenvalues > max(float(eigenvalues[0]) * 1e-10, 1e-12))) if eigenvalues.size else 0
    n_rank = min(q, nonzero)
    W_flat = np.zeros((q, int(np.prod(shape))), dtype=np.float32)
    if n_rank:
        # If centered = U S V^T, then W_i = sqrt(lambda_i) V_i = S_i V_i
        # equals U[:, i]^T @ centered.  This avoids materializing a wide SVD.
        W_flat[:n_rank] = (eigenvectors[:, :n_rank].T @ centered).astype(np.float32)
    W = W_flat.reshape(q, *shape)
    singular_values = np.sqrt(eigenvalues)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "ppca_init.npz"
    payload = {
        "mu": mu,
        "W": W,
        "class_volume_paths": np.asarray([str(path) for path in paths]),
        "volume_shape": np.asarray(shape, dtype=np.int32),
        "q_requested": np.int32(q),
        "rank": np.int32(n_rank),
        "singular_values": singular_values.astype(np.float32),
        "eigenvalues_equal_class_prior": (singular_values**2).astype(np.float32),
        "W_scaling": np.asarray("sqrt_equal_class_covariance_eigenvalue"),
    }
    if voxel_size is not None:
        payload["voxel_size"] = np.float32(voxel_size)
    np.savez_compressed(npz_path, **payload)

    summary = {
        "class_volume_paths": [str(path) for path in paths],
        "ppca_init": str(npz_path),
        "q": q,
        "rank": n_rank,
        "volume_shape": list(shape),
        "voxel_size": None if voxel_size is None else float(voxel_size),
        "singular_values": [float(x) for x in singular_values],
        "mu_rms": float(np.sqrt(np.mean(np.asarray(mu, dtype=np.float64) ** 2))),
        "W_rms": [float(np.sqrt(np.mean(np.asarray(w, dtype=np.float64) ** 2))) for w in W],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return npz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-volume", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=None)
    args = parser.parse_args()
    print(
        prepare_ppca_init_from_class_volumes(
            args.class_volume,
            args.output_dir,
            q=int(args.q),
            voxel_size=args.voxel_size,
        )
    )


if __name__ == "__main__":
    main()
