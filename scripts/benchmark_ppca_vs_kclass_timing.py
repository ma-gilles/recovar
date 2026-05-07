#!/usr/bin/env python
"""Benchmark dense PPCA refinement against dense K-class EM on one fixture."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.em.dense_single_volume.k_class import run_dense_k_class_em
from recovar.em.ppca_refinement.dense_dataset import run_dense_ppca_fused_em_iteration
from recovar.em.ppca_refinement.initialization import (
    loading_row_norm_variance_prior,
    volume_power_variance_prior,
)
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


def _load_simulation_info(path: str | Path | None):
    if path is None:
        return None
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _load_noise_variance(simulation_info, image_shape) -> np.ndarray:
    if simulation_info is None:
        return np.ones(int(np.prod(image_shape)), dtype=np.float32)
    radial = np.asarray(simulation_info["noise_variance"], dtype=np.float32).reshape(-1)
    return np.asarray(recon_noise.make_radial_noise(radial, tuple(image_shape)), dtype=np.float32).reshape(-1)


def _half_size(volume_shape) -> int:
    return int(np.prod(ftu.volume_shape_to_half_volume_shape(tuple(volume_shape))))


def _load_init(init_npz: str | Path, *, q: int, k_classes: int):
    init = np.load(init_npz, allow_pickle=True)
    mu = np.asarray(init["mu"], dtype=np.float32)
    W = np.asarray(init["W"], dtype=np.float32)[: int(q)]
    aligned = np.asarray(init["aligned_volumes"], dtype=np.float32)[: int(k_classes)]
    if aligned.shape[0] != int(k_classes):
        raise ValueError(f"init only has {aligned.shape[0]} aligned volumes, requested K={k_classes}")
    return mu, W, aligned


def _make_rotations_translations(args, simulation_info, n_images):
    if args.rotation_source == "simulation-info":
        if simulation_info is None:
            raise ValueError("--rotation-source=simulation-info requires --simulation-info")
        rotations = np.asarray(simulation_info["rots"], dtype=np.float32)[: int(n_images)]
    else:
        rotations = np.asarray(get_rotation_grid_at_order(int(args.healpix_order)), dtype=np.float32)
        if args.max_rotations is not None:
            rotations = rotations[: int(args.max_rotations)]
    if args.translation_source == "simulation-info-unique":
        if simulation_info is None:
            raise ValueError("--translation-source=simulation-info-unique requires --simulation-info")
        translations = np.unique(np.asarray(simulation_info["trans"], dtype=np.float32)[: int(n_images)], axis=0)
    else:
        translations = np.asarray(get_translation_grid(float(args.offset_range_px), float(args.offset_step_px)), dtype=np.float32)
        if args.max_translations is not None:
            translations = translations[: int(args.max_translations)]
    return rotations, translations


def _block_ppca(result):
    jax.block_until_ready(result.mu_half)
    jax.block_until_ready(result.W_half)
    jax.block_until_ready(result.diagnostics["best_rotation_idx"])


def _block_kclass(result):
    if result.new_means is not None:
        jax.block_until_ready(result.new_means)
    jax.block_until_ready(result.Ft_y)
    jax.block_until_ready(result.Ft_ctf)
    jax.block_until_ready(result.pose_assignments)


def _time_call(label, fn, block_fn, *, warmups: int, repeats: int):
    for _ in range(int(warmups)):
        result = fn()
        block_fn(result)
    elapsed = []
    last = None
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        last = fn()
        block_fn(last)
        elapsed.append(time.perf_counter() - t0)
    return {
        "label": label,
        "elapsed_s": elapsed,
        "min_s": float(np.min(elapsed)) if elapsed else float("nan"),
        "median_s": float(np.median(elapsed)) if elapsed else float("nan"),
        "mean_s": float(np.mean(elapsed)) if elapsed else float("nan"),
        "last": last,
    }


def main() -> None:
    args = _parse_args()
    output = None if args.output is None else Path(args.output)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(str(args.data_star))
    n_images = int(dataset.n_images) if args.n_images is None else min(int(args.n_images), int(dataset.n_images))
    image_indices = np.arange(n_images, dtype=np.int64)
    simulation_info = _load_simulation_info(args.simulation_info)
    rotations, translations = _make_rotations_translations(args, simulation_info, n_images)
    rotation_translation_mask = None
    if args.exact_pose_support_mask:
        if args.rotation_source != "simulation-info" or args.translation_source != "simulation-info-unique":
            raise ValueError("--exact-pose-support-mask requires simulator rotation and translation sources")
        gt_trans = np.asarray(simulation_info["trans"], dtype=np.float32)[:n_images]
        _translations_check, expected_translation_idx = np.unique(gt_trans, axis=0, return_inverse=True)
        rotation_translation_mask = np.zeros((n_images, int(rotations.shape[0]), int(translations.shape[0])), dtype=bool)
        rows = np.arange(n_images, dtype=np.int64)
        rotation_translation_mask[rows, rows, np.asarray(expected_translation_idx, dtype=np.int64)] = True
    noise_variance = _load_noise_variance(simulation_info, dataset.image_shape)
    mu, W, class_volumes = _load_init(args.init_npz, q=int(args.q), k_classes=int(args.k_classes))

    half_size = _half_size(dataset.volume_shape)
    if args.ppca_prior == "gt-row-norm":
        prior_init_npz = args.init_npz if args.prior_init_npz is None else args.prior_init_npz
        prior_mu, prior_W, _prior_class_volumes = _load_init(
            prior_init_npz,
            q=int(args.q),
            k_classes=int(args.k_classes),
        )
        mean_prior = jnp.asarray(
            volume_power_variance_prior(
                prior_mu,
                volume_shape=dataset.volume_shape,
                volume_domain="real",
                box_size_power=float(args.gt_prior_box_power),
                floor=float(args.gt_prior_floor),
                shell_average=bool(args.gt_prior_shell_average),
            ),
            dtype=jnp.float32,
        )
        W_prior = jnp.asarray(
            loading_row_norm_variance_prior(
                prior_W,
                volume_shape=dataset.volume_shape,
                volume_domain="real",
                q=int(args.q),
                box_size_power=float(args.gt_prior_box_power),
                floor=float(args.gt_prior_floor),
                shell_average=bool(args.gt_prior_shell_average),
            ),
            dtype=jnp.float32,
        )
    else:
        prior_init_npz = None
        mean_prior = jnp.full((half_size,), float(args.mean_prior_variance), dtype=jnp.float32)
        W_prior = jnp.full((half_size, int(args.q)), float(args.W_prior_variance), dtype=jnp.float32)

    class_means = jnp.asarray(
        np.stack([np.asarray(ftu.get_dft3(volume), dtype=np.complex64).reshape(-1) for volume in class_volumes], axis=0),
        dtype=jnp.complex64,
    )
    volume_size = int(np.prod(tuple(dataset.volume_shape)))
    mean_variance = jnp.full((int(args.k_classes), volume_size), float(args.kclass_mean_variance), dtype=jnp.float32)

    common = {
        "image_batch_size": int(args.image_batch_size),
        "rotation_block_size": int(args.rotation_block_size),
        "current_size": int(args.current_size),
        "image_indices": image_indices,
        "score_with_masked_images": False,
        "half_spectrum_scoring": False,
        "square_window": False,
    }

    def run_ppca():
        return run_dense_ppca_fused_em_iteration(
            dataset,
            mu,
            W,
            mean_prior=mean_prior,
            W_prior=W_prior,
            noise_variance=noise_variance,
            rotations=rotations,
            translations=translations,
            q=int(args.q),
            volume_domain="real",
            relion_texture_interp=False,
            rotation_translation_mask=rotation_translation_mask,
            skip_empty_pose_blocks=rotation_translation_mask is not None,
            sparse_pass2=bool(args.ppca_sparse_pass2),
            sparse_pass2_log_threshold=float(args.ppca_sparse_pass2_log_threshold),
            mstep_chunk_size=int(args.mstep_chunk_size),
            **common,
        )

    def run_kclass():
        return run_dense_k_class_em(
            dataset,
            class_means,
            mean_variance,
            noise_variance,
            rotations,
            translations,
            "linear_interp",
            sparse_pass2=bool(args.kclass_sparse_pass2),
            relion_half_volume_mstep=bool(args.kclass_half_volume_mstep),
            **common,
        )

    ppca = _time_call("ppca", run_ppca, _block_ppca, warmups=int(args.warmups), repeats=int(args.repeats))
    kclass = _time_call("kclass", run_kclass, _block_kclass, warmups=int(args.warmups), repeats=int(args.repeats))
    ppca_result = ppca.pop("last")
    kclass_result = kclass.pop("last")

    result = {
        "passed": True,
        "data_star": Path(args.data_star),
        "simulation_info": None if args.simulation_info is None else Path(args.simulation_info),
        "init_npz": Path(args.init_npz),
        "jax_devices": [str(device) for device in jax.devices()],
        "params": {
            "n_images": n_images,
            "q": int(args.q),
            "k_classes": int(args.k_classes),
            "q2_over_2": 0.5 * float(args.q) ** 2,
            "n_rotations": int(rotations.shape[0]),
            "n_translations": int(translations.shape[0]),
            "current_size": int(args.current_size),
            "image_batch_size": int(args.image_batch_size),
            "rotation_block_size": int(args.rotation_block_size),
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
            "ppca_sparse_pass2": bool(args.ppca_sparse_pass2),
            "ppca_sparse_pass2_log_threshold": float(args.ppca_sparse_pass2_log_threshold),
            "kclass_sparse_pass2": bool(args.kclass_sparse_pass2),
            "kclass_half_volume_mstep": bool(args.kclass_half_volume_mstep),
            "exact_pose_support_mask": bool(args.exact_pose_support_mask),
            "prior_init_npz": None if prior_init_npz is None else str(prior_init_npz),
        },
        "timing": {
            "ppca": ppca,
            "kclass": kclass,
            "ppca_over_kclass_median": float(ppca["median_s"] / kclass["median_s"]),
            "ppca_over_kclass_q2_half_adjusted": float(ppca["median_s"] / kclass["median_s"] * int(args.k_classes) / (0.5 * float(args.q) ** 2)),
        },
        "diagnostics": {
            "ppca_logZ_mean": float(ppca_result.diagnostics["logZ_mean"]),
            "ppca_pmax_mean": float(ppca_result.diagnostics["pmax_mean"]),
            "ppca_sparse_pass2_skipped_blocks": int(ppca_result.diagnostics.get("sparse_pass2_skipped_blocks", 0)),
            "ppca_sparse_pass2_total_blocks": int(ppca_result.diagnostics.get("sparse_pass2_total_blocks", 0)),
            "ppca_sparse_pass2_skipped_fraction": float(ppca_result.diagnostics.get("sparse_pass2_skipped_fraction", 0.0)),
            "ppca_score_fourier_size": int(ppca_result.diagnostics.get("score_fourier_size", 0)),
            "ppca_recon_fourier_size": int(ppca_result.diagnostics.get("recon_fourier_size", 0)),
            "ppca_full_half_fourier_size": int(ppca_result.diagnostics.get("full_half_fourier_size", 0)),
            "ppca_uses_fourier_window": bool(ppca_result.diagnostics.get("uses_fourier_window", False)),
            "kclass_pmax_mean": float(jnp.mean(kclass_result.stats.max_posterior_per_image)),
            "kclass_class_posterior_sums": np.asarray(kclass_result.class_posterior_sums).astype(float),
        },
    }
    text = json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.write_text(text)
    print(text, end="")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-star", required=True)
    parser.add_argument("--simulation-info")
    parser.add_argument("--init-npz", required=True)
    parser.add_argument(
        "--prior-init-npz",
        default=None,
        help="When --ppca-prior=gt-row-norm, compute PPCA priors from this init npz instead of --init-npz.",
    )
    parser.add_argument("--output")
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--k-classes", type=int, default=4)
    parser.add_argument("--n-images", type=int, default=128)
    parser.add_argument("--rotation-source", choices=("healpix", "simulation-info"), default="simulation-info")
    parser.add_argument("--translation-source", choices=("grid", "simulation-info-unique"), default="simulation-info-unique")
    parser.add_argument("--exact-pose-support-mask", action="store_true")
    parser.add_argument("--healpix-order", type=int, default=1)
    parser.add_argument("--max-rotations", type=int, default=None)
    parser.add_argument("--offset-range-px", type=float, default=6.0)
    parser.add_argument("--offset-step-px", type=float, default=2.0)
    parser.add_argument("--max-translations", type=int, default=None)
    parser.add_argument("--current-size", type=int, default=64)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--rotation-block-size", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mstep-chunk-size", type=int, default=65536)
    parser.add_argument("--ppca-prior", choices=("constant", "gt-row-norm"), default="gt-row-norm")
    parser.add_argument("--mean-prior-variance", type=float, default=1.0)
    parser.add_argument("--W-prior-variance", type=float, default=1.0)
    parser.add_argument("--gt-prior-box-power", type=float, default=2.0)
    parser.add_argument("--gt-prior-floor", type=float, default=1.0e-8)
    parser.add_argument("--gt-prior-shell-average", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ppca-sparse-pass2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ppca-sparse-pass2-log-threshold", type=float, default=float(np.log(1.0e-6)))
    parser.add_argument("--kclass-mean-variance", type=float, default=1.0)
    parser.add_argument("--kclass-sparse-pass2", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kclass-half-volume-mstep", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
