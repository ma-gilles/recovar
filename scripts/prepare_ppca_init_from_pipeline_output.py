#!/usr/bin/env python
"""Convert fixed-pose RECOVAR pipeline output into a PPCA refinement init NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovar.output.output import PipelineOutput
from recovar.utils import helpers


def _load_rescaled_eigenvalues(po: PipelineOutput) -> np.ndarray:
    s = po.get("s")
    if isinstance(s, dict):
        if "rescaled" in s:
            return np.asarray(s["rescaled"], dtype=np.float32).reshape(-1)
        if "s" in s:
            return np.asarray(s["s"], dtype=np.float32).reshape(-1)
    return np.asarray(s, dtype=np.float32).reshape(-1)


def prepare_ppca_init_from_pipeline_output(
    pipeline_output: str | Path,
    output_dir: str | Path,
    *,
    q: int,
    scale_by_sqrt_eigenvalue: bool = True,
) -> Path:
    po = PipelineOutput(str(Path(pipeline_output).expanduser().resolve()))
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    q = int(q)
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    mu = np.asarray(helpers.load_mrc(po.paths.mean_volume), dtype=np.float32)
    u_real = np.asarray(po.get_u_real(q), dtype=np.float32)
    if u_real.shape[0] < q:
        raise ValueError(f"pipeline output only has {u_real.shape[0]} saved eigenvectors, requested q={q}")
    s_rescaled = _load_rescaled_eigenvalues(po)
    if s_rescaled.shape[0] < q:
        raise ValueError(f"pipeline output only has {s_rescaled.shape[0]} eigenvalues, requested q={q}")
    W = u_real[:q].copy()
    if scale_by_sqrt_eigenvalue:
        W *= np.sqrt(np.maximum(s_rescaled[:q], 0.0)).astype(np.float32)[:, None, None, None]

    npz_path = output_dir / "ppca_init.npz"
    payload = {
        "mu": mu,
        "W": W,
        "s_rescaled": s_rescaled[:q],
        "volume_shape": np.asarray(mu.shape, dtype=np.int32),
        "voxel_size": np.float32(po.params.get("voxel_size", np.nan)),
        "pipeline_output": np.asarray(str(Path(pipeline_output).expanduser().resolve())),
        "W_scaling": np.asarray("sqrt_s_rescaled" if scale_by_sqrt_eigenvalue else "unit_eigenvectors"),
    }
    if "noise_var_used" in po.params:
        payload["noise_var_used"] = np.asarray(po.params["noise_var_used"], dtype=np.float32)
    np.savez_compressed(npz_path, **payload)

    summary = {
        "pipeline_output": str(Path(pipeline_output).expanduser().resolve()),
        "ppca_init": str(npz_path),
        "q": q,
        "volume_shape": list(mu.shape),
        "scale_by_sqrt_eigenvalue": bool(scale_by_sqrt_eigenvalue),
        "s_rescaled": [float(x) for x in s_rescaled[:q]],
        "mu_rms": float(np.sqrt(np.mean(np.asarray(mu, dtype=np.float64) ** 2))),
        "W_rms": [float(np.sqrt(np.mean(np.asarray(W_i, dtype=np.float64) ** 2))) for W_i in W],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return npz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument(
        "--scale-by-sqrt-eigenvalue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use W_k = sqrt(s_rescaled[k]) * eigenvector_k, matching a unit latent prior.",
    )
    args = parser.parse_args()
    print(
        prepare_ppca_init_from_pipeline_output(
            args.pipeline_output,
            args.output_dir,
            q=int(args.q),
            scale_by_sqrt_eigenvalue=bool(args.scale_by_sqrt_eigenvalue),
        )
    )


if __name__ == "__main__":
    main()
