#!/usr/bin/env python
"""Benchmark dense K-class EM from a PPCA initializer.

This is a timing harness, not a scientific initializer.  It builds ``K``
reference volumes from ``mu`` and ``W`` in a PPCA init NPZ, then runs the
current dense K-class EM path on a fixed image subset and pose grid.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.em.dense_single_volume.k_class import run_dense_k_class_em
from recovar.em.sampling import get_rotation_grid_at_order, get_translation_grid
from recovar.reconstruction import noise as recon_noise


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


def _make_kclass_references(mu: np.ndarray, W: np.ndarray, *, k: int, pc_scale: float) -> np.ndarray:
    """Build ``K`` real-space references as deterministic sign combinations."""

    mu = np.asarray(mu, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 4:
        raise ValueError(f"expected W shaped [q, N, N, N], got {W.shape}")
    q = int(W.shape[0])
    if q == 0:
        return np.repeat(mu[None, ...], int(k), axis=0)

    refs = []
    for class_idx in range(int(k)):
        coeffs = np.asarray([1.0 if ((class_idx >> pc_idx) & 1) else -1.0 for pc_idx in range(q)], dtype=np.float32)
        refs.append(mu + float(pc_scale) * np.tensordot(coeffs, W, axes=(0, 0)) / np.sqrt(float(q)))
    return np.stack(refs, axis=0).astype(np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-star", required=True)
    parser.add_argument("--simulation-info")
    parser.add_argument("--init-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--k", type=int, default=None, help="Default: q*q")
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--healpix-order", type=int, default=3)
    parser.add_argument("--max-rotations", type=int, default=None, help="Debug cap; default uses the full HEALPix grid")
    parser.add_argument("--offset-range-px", type=float, default=6.0)
    parser.add_argument("--offset-step-px", type=float, default=2.0)
    parser.add_argument("--max-translations", type=int, default=None, help="Debug cap; default uses the full translation grid")
    parser.add_argument("--current-size", type=int, default=64)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--rotation-block-size", type=int, default=72)
    parser.add_argument("--mean-prior-variance", type=float, default=1.0)
    parser.add_argument("--pc-scale", type=float, default=1.0)
    parser.add_argument("--sparse-pass2", action="store_true", help="Enable dense engine sparse pass-2 block skipping")
    parser.add_argument("--score-with-masked-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(str(args.data_star))
    init = np.load(args.init_npz, allow_pickle=True)
    mu = np.asarray(init["mu"], dtype=np.float32)
    W = np.asarray(init["W"], dtype=np.float32)[: int(args.q)]
    k = int(args.k if args.k is not None else int(args.q) * int(args.q))
    refs_real = _make_kclass_references(mu, W, k=k, pc_scale=float(args.pc_scale))
    if tuple(refs_real.shape[1:]) != tuple(dataset.volume_shape):
        raise ValueError(f"reference shape {refs_real.shape[1:]} does not match dataset volume shape {dataset.volume_shape}")

    means = jnp.stack([ftu.get_dft3(jnp.asarray(ref)) for ref in refs_real], axis=0).reshape(k, -1)
    mean_variance = jnp.ones((k, int(np.prod(dataset.volume_shape))), dtype=jnp.float32) * float(args.mean_prior_variance)
    noise_variance = _load_noise_variance(args.simulation_info, dataset.image_shape)
    rotations = np.asarray(get_rotation_grid_at_order(int(args.healpix_order)), dtype=np.float32)
    if args.max_rotations is not None:
        rotations = rotations[: int(args.max_rotations)]
    translations = np.asarray(get_translation_grid(float(args.offset_range_px), float(args.offset_step_px)), dtype=np.float32)
    if args.max_translations is not None:
        translations = translations[: int(args.max_translations)]
    n_images = min(int(args.n_images), int(dataset.n_images))
    image_indices = np.arange(n_images, dtype=np.int64)

    t0 = time.time()
    result = run_dense_k_class_em(
        dataset,
        means,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        "linear_interp",
        class_log_priors=np.zeros(k, dtype=np.float32),
        accumulate_noise=False,
        image_batch_size=int(args.image_batch_size),
        rotation_block_size=int(args.rotation_block_size),
        current_size=int(args.current_size),
        image_indices=image_indices,
        score_with_masked_images=bool(args.score_with_masked_images),
        half_spectrum_scoring=False,
        square_window=False,
        sparse_pass2=bool(args.sparse_pass2),
        use_float64_scoring=False,
        use_float64_projections=False,
    )
    jax.block_until_ready(result.new_means)
    elapsed_s = float(time.time() - t0)

    summary = {
        "passed": bool(np.all(np.isfinite(np.asarray(result.new_means)))),
        "data_star": Path(args.data_star),
        "simulation_info": None if args.simulation_info is None else Path(args.simulation_info),
        "init_npz": Path(args.init_npz),
        "output_dir": output_dir,
        "elapsed_s": elapsed_s,
        "n_images": n_images,
        "image_shape": list(dataset.image_shape),
        "volume_shape": list(dataset.volume_shape),
        "q_source": int(args.q),
        "k": k,
        "healpix_order": int(args.healpix_order),
        "n_rotations": int(rotations.shape[0]),
        "n_translations": int(translations.shape[0]),
        "current_size": int(args.current_size),
        "image_batch_size": int(args.image_batch_size),
        "rotation_block_size": int(args.rotation_block_size),
        "sparse_pass2": bool(args.sparse_pass2),
        "diagnostics": {
            "class_assignment_counts": np.bincount(np.asarray(result.class_assignments), minlength=k),
            "pmax_mean": float(np.mean([np.mean(np.asarray(s.max_posterior_per_image)) for s in result.per_class_stats])),
            "best_log_score_mean": float(np.mean([np.mean(np.asarray(s.best_log_score_per_image)) for s in result.per_class_stats])),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
