#!/usr/bin/env python
"""Run dense PPCA EM iterations from a precomputed PPCA init npz."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import time
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.em.ppca_refinement.dense_dataset import run_dense_ppca_fused_em_iteration
from recovar.em.sampling import get_rotation_grid_at_order, get_translation_grid
from recovar.reconstruction import noise as recon_noise
from recovar.utils import helpers


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


def _load_noise_variance(simulation_info: str | Path | None, image_shape) -> np.ndarray:
    if simulation_info is None:
        return np.ones(int(np.prod(image_shape)), dtype=np.float32)
    with Path(simulation_info).open("rb") as f:
        info = pickle.load(f)
    radial = np.asarray(info["noise_variance"], dtype=np.float32).reshape(-1)
    return np.asarray(recon_noise.make_radial_noise(radial, tuple(image_shape)), dtype=np.float32).reshape(-1)


def _half_size(volume_shape) -> int:
    return int(np.prod(ftu.volume_shape_to_half_volume_shape(tuple(volume_shape))))


def _write_half_volume_mrc(path: Path, half_volume, volume_shape, *, voxel_size: float | None) -> None:
    full = ftu.half_volume_to_full_volume(jnp.asarray(half_volume), tuple(volume_shape))
    real = np.asarray(ftu.get_idft3(full.reshape(tuple(volume_shape))).real)
    helpers.write_mrc(path, real, voxel_size=voxel_size)


def _load_init(init_npz: str | Path, *, q_override: int | None):
    init_path = Path(init_npz)
    init = np.load(init_path, allow_pickle=True)
    mu = np.asarray(init["mu"], dtype=np.float32)
    W = np.asarray(init["W"], dtype=np.float32)
    if W.ndim != 4:
        raise ValueError(f"expected W shaped [q, N, N, N], got {W.shape}")
    if q_override is not None:
        q = int(q_override)
        if q > int(W.shape[0]):
            raise ValueError(f"q override {q} exceeds init W rank {W.shape[0]}")
        W = W[:q]
    return mu, W, int(W.shape[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-star", required=True, help="Input particles.star")
    parser.add_argument("--simulation-info", help="Synthetic simulation_info.pkl with radial noise variance")
    parser.add_argument("--init-npz", required=True, help="ppca_init.npz from prepare_random_volume_ppca_init.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--q", type=int, default=None, help="Optional truncation from init W")
    parser.add_argument("--n-iters", type=int, default=1)
    parser.add_argument("--n-images", type=int, default=None, help="Default: all images")
    parser.add_argument("--healpix-order", type=int, default=1)
    parser.add_argument("--max-rotations", type=int, default=None, help="Debug cap; default uses full HEALPix grid")
    parser.add_argument("--offset-range-px", type=float, default=6.0)
    parser.add_argument("--offset-step-px", type=float, default=2.0)
    parser.add_argument("--max-translations", type=int, default=None, help="Debug cap; default uses full translation grid")
    parser.add_argument("--current-size", type=int, default=16)
    parser.add_argument("--image-batch-size", type=int, default=50)
    parser.add_argument("--rotation-block-size", type=int, default=72)
    parser.add_argument("--mstep-chunk-size", type=int, default=65536)
    parser.add_argument("--mean-prior-variance", type=float, default=1.0)
    parser.add_argument("--W-prior-variance", type=float, default=1.0)
    parser.add_argument("--save-mrc", action="store_true", help="Write final mean and PC maps in RECOVAR MRC convention")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(str(args.data_star))
    mu, W, q = _load_init(args.init_npz, q_override=args.q)
    if tuple(mu.shape) != tuple(dataset.volume_shape):
        raise ValueError(f"init mu shape {mu.shape} does not match dataset volume shape {dataset.volume_shape}")

    rotations = np.asarray(get_rotation_grid_at_order(int(args.healpix_order)), dtype=np.float32)
    if args.max_rotations is not None:
        rotations = rotations[: int(args.max_rotations)]
    translations = np.asarray(get_translation_grid(float(args.offset_range_px), float(args.offset_step_px)), dtype=np.float32)
    if args.max_translations is not None:
        translations = translations[: int(args.max_translations)]
    n_images = int(dataset.n_images) if args.n_images is None else min(int(args.n_images), int(dataset.n_images))
    image_indices = np.arange(n_images, dtype=np.int64)
    noise_variance = _load_noise_variance(args.simulation_info, dataset.image_shape)

    half_size = _half_size(dataset.volume_shape)
    mean_prior = jnp.full((half_size,), float(args.mean_prior_variance), dtype=jnp.float32)
    W_prior = jnp.full((half_size, q), float(args.W_prior_variance), dtype=jnp.float32)

    iter_summaries = []
    volume_domain = "real"
    current_mu = mu
    current_W = W
    final_result = None
    t_all = time.time()
    for iter_idx in range(1, int(args.n_iters) + 1):
        t0 = time.time()
        result = run_dense_ppca_fused_em_iteration(
            dataset,
            current_mu,
            current_W,
            mean_prior=mean_prior,
            W_prior=W_prior,
            noise_variance=noise_variance,
            rotations=rotations,
            translations=translations,
            image_batch_size=int(args.image_batch_size),
            rotation_block_size=int(args.rotation_block_size),
            current_size=int(args.current_size),
            q=q,
            volume_domain=volume_domain,
            image_indices=image_indices,
            mstep_chunk_size=int(args.mstep_chunk_size),
            half_spectrum_scoring=False,
            square_window=False,
        )
        jax.block_until_ready(result.mu_half)
        jax.block_until_ready(result.W_half)
        elapsed_s = float(time.time() - t0)
        iter_npz = output_dir / f"iter{iter_idx:03d}.npz"
        np.savez_compressed(
            iter_npz,
            mu_half=np.asarray(result.mu_half),
            W_half=np.asarray(result.W_half),
            best_rotation_idx=np.asarray(result.diagnostics["best_rotation_idx"]),
            best_translation_idx=np.asarray(result.diagnostics["best_translation_idx"]),
            log_likelihood=np.asarray(result.diagnostics["log_likelihood"]),
            logZ_mean=np.asarray(result.diagnostics["logZ_mean"]),
            pmax_mean=np.asarray(result.diagnostics["pmax_mean"]),
            nsig_mean=np.asarray(result.diagnostics["nsig_mean"]),
        )
        iter_summary = {
            "iteration": iter_idx,
            "elapsed_s": elapsed_s,
            "npz_path": iter_npz,
            "diagnostics": {
                "log_likelihood": float(result.diagnostics["log_likelihood"]),
                "logZ_mean": float(result.diagnostics["logZ_mean"]),
                "pmax_mean": float(result.diagnostics["pmax_mean"]),
                "nsig_mean": float(result.diagnostics["nsig_mean"]),
                "n_images_accumulated": int(result.stats.n_images),
            },
            "output_stats": {
                "mu_half_shape": list(np.asarray(result.mu_half).shape),
                "W_half_shape": list(np.asarray(result.W_half).shape),
                "mu_half_finite": bool(np.all(np.isfinite(np.asarray(result.mu_half)))),
                "W_half_finite": bool(np.all(np.isfinite(np.asarray(result.W_half)))),
                "mu_half_rms": float(jnp.sqrt(jnp.mean(jnp.abs(result.mu_half) ** 2))),
                "W_half_rms": float(jnp.sqrt(jnp.mean(jnp.abs(result.W_half) ** 2))) if q else 0.0,
            },
        }
        iter_summaries.append(iter_summary)
        print(json.dumps(_jsonable(iter_summary), indent=2, sort_keys=True), flush=True)
        current_mu = np.asarray(result.mu_half)
        current_W = np.asarray(result.W_half)
        volume_domain = "fourier_half"
        final_result = result

    if final_result is None:
        raise SystemExit("no iterations were run")

    final_npz = output_dir / "final_ppca_dense.npz"
    np.savez_compressed(
        final_npz,
        mu_half=np.asarray(final_result.mu_half),
        W_half=np.asarray(final_result.W_half),
        best_rotation_idx=np.asarray(final_result.diagnostics["best_rotation_idx"]),
        best_translation_idx=np.asarray(final_result.diagnostics["best_translation_idx"]),
    )

    written_mrcs = []
    if args.save_mrc:
        voxel_size = float(getattr(dataset, "voxel_size", 1.0))
        mean_path = output_dir / "final_mu.mrc"
        _write_half_volume_mrc(mean_path, final_result.mu_half, dataset.volume_shape, voxel_size=voxel_size)
        written_mrcs.append(mean_path)
        for pc_idx in range(q):
            pc_path = output_dir / f"final_W{pc_idx + 1:02d}.mrc"
            _write_half_volume_mrc(pc_path, final_result.W_half[:, pc_idx], dataset.volume_shape, voxel_size=voxel_size)
            written_mrcs.append(pc_path)

    summary = {
        "passed": bool(
            np.all(np.isfinite(np.asarray(final_result.mu_half)))
            and np.all(np.isfinite(np.asarray(final_result.W_half)))
            and int(final_result.stats.n_images) == n_images
        ),
        "data_star": Path(args.data_star),
        "simulation_info": None if args.simulation_info is None else Path(args.simulation_info),
        "init_npz": Path(args.init_npz),
        "output_dir": output_dir,
        "final_npz": final_npz,
        "written_mrcs": written_mrcs,
        "n_images": n_images,
        "image_shape": list(dataset.image_shape),
        "volume_shape": list(dataset.volume_shape),
        "q": q,
        "n_iters": int(args.n_iters),
        "healpix_order": int(args.healpix_order),
        "n_rotations": int(rotations.shape[0]),
        "offset_range_px": float(args.offset_range_px),
        "offset_step_px": float(args.offset_step_px),
        "n_translations": int(translations.shape[0]),
        "current_size": int(args.current_size),
        "image_batch_size": int(args.image_batch_size),
        "rotation_block_size": int(args.rotation_block_size),
        "elapsed_s": float(time.time() - t_all),
        "iterations": iter_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
