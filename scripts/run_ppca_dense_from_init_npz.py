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
from recovar.em.ppca_refinement.dense_dataset import coerce_augmented_half_volumes, run_dense_ppca_fused_em_iteration
from recovar.em.ppca_refinement.initialization import (
    loading_row_norm_variance_prior,
    volume_power_variance_prior,
)
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


def _load_simulation_info(simulation_info: str | Path | None) -> dict[str, Any] | None:
    if simulation_info is None:
        return None
    with Path(simulation_info).open("rb") as f:
        return pickle.load(f)


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


def _regularization_penalty(mu, W, *, mean_prior, W_prior, volume_shape, volume_domain: str, q: int) -> float:
    augmented_half, _q = coerce_augmented_half_volumes(
        mu,
        W,
        volume_shape=tuple(volume_shape),
        q=int(q),
        volume_domain=volume_domain,
    )
    mu_half = jnp.asarray(augmented_half[0])
    W_half = jnp.swapaxes(jnp.asarray(augmented_half[1:]), 0, 1) if q else jnp.zeros((mu_half.shape[0], 0), dtype=mu_half.dtype)
    mean_prior = jnp.asarray(mean_prior)
    W_prior = jnp.asarray(W_prior)
    mu_penalty = jnp.sum(jnp.abs(mu_half) ** 2 / mean_prior)
    W_penalty = jnp.sum(jnp.abs(W_half) ** 2 / W_prior) if q else jnp.asarray(0.0, dtype=mu_penalty.dtype)
    return float(-0.5 * (mu_penalty + W_penalty))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-star", required=True, help="Input particles.star")
    parser.add_argument("--simulation-info", help="Synthetic simulation_info.pkl with radial noise variance")
    parser.add_argument("--init-npz", required=True, help="ppca_init.npz from prepare_random_volume_ppca_init.py")
    parser.add_argument(
        "--prior-init-npz",
        default=None,
        help=(
            "When --prior-from-init=gt-row-norm, compute mean/W priors from this init npz "
            "instead of --init-npz. Use this to test random W with GT-derived W_prior."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--q", type=int, default=None, help="Optional truncation from init W")
    parser.add_argument("--n-iters", type=int, default=1)
    parser.add_argument("--n-images", type=int, default=None, help="Default: all images")
    parser.add_argument(
        "--rotation-source",
        choices=("healpix", "simulation-info"),
        default="healpix",
        help="Use a HEALPix grid or the first n simulator rotations as candidates.",
    )
    parser.add_argument("--healpix-order", type=int, default=1)
    parser.add_argument("--max-rotations", type=int, default=None, help="Debug cap; default uses full HEALPix grid")
    parser.add_argument(
        "--translation-source",
        choices=("grid", "simulation-info-unique"),
        default="grid",
        help="Use regular translation grid or unique GT translations from simulation_info.",
    )
    parser.add_argument(
        "--exact-pose-support-mask",
        action="store_true",
        help=(
            "With simulator rotations and translations, score only the known GT pose for each image. "
            "This is a controlled diagnostic path, not a pose-search run."
        ),
    )
    parser.add_argument("--offset-range-px", type=float, default=6.0)
    parser.add_argument("--offset-step-px", type=float, default=2.0)
    parser.add_argument("--max-translations", type=int, default=None, help="Debug cap; default uses full translation grid")
    parser.add_argument("--current-size", type=int, default=16)
    parser.add_argument("--image-batch-size", type=int, default=50)
    parser.add_argument("--rotation-block-size", type=int, default=512)
    parser.add_argument("--mstep-chunk-size", type=int, default=65536)
    parser.add_argument(
        "--sparse-pass2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip dense pass-2 rotation blocks whose posterior mass upper bound is negligible. "
            "Pass 1 still normalizes over all candidate poses."
        ),
    )
    parser.add_argument("--sparse-pass2-log-threshold", type=float, default=float(np.log(1.0e-6)))
    parser.add_argument(
        "--relion-texture-interp",
        action="store_true",
        help="Enable RELION texture interpolation in projection/backprojection helpers.",
    )
    parser.add_argument("--mean-prior-variance", type=float, default=1.0)
    parser.add_argument("--W-prior-variance", type=float, default=1.0)
    parser.add_argument(
        "--prior-from-init",
        choices=("constant", "gt-row-norm"),
        default="constant",
        help=(
            "constant uses scalar prior variances. gt-row-norm is a synthetic/debug mode "
            "that derives variance-like priors from the initial GT-scaled mean and W."
        ),
    )
    parser.add_argument(
        "--gt-prior-box-power",
        type=float,
        default=2.0,
        help="When --prior-from-init=gt-row-norm, divide raw DFT power by N**this value.",
    )
    parser.add_argument("--gt-w-prior-scale", type=float, default=1.0)
    parser.add_argument("--gt-mean-prior-scale", type=float, default=1.0)
    parser.add_argument("--gt-prior-floor", type=float, default=1.0e-8)
    parser.add_argument(
        "--gt-prior-shell-average",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shell-average GT-derived mean/W power priors before broadcasting back to half-Fourier coefficients.",
    )
    parser.add_argument(
        "--gt-w-prior-divide-by-q-total",
        action="store_true",
        help="Divide GT shell row-norm-squared W prior by q, matching make_shell_w_prior's per-component convention.",
    )
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

    n_images = int(dataset.n_images) if args.n_images is None else min(int(args.n_images), int(dataset.n_images))
    image_indices = np.arange(n_images, dtype=np.int64)
    simulation_info = _load_simulation_info(args.simulation_info)
    if args.rotation_source == "simulation-info":
        if simulation_info is None:
            raise ValueError("--rotation-source=simulation-info requires --simulation-info")
        rotations = np.asarray(simulation_info["rots"], dtype=np.float32)[:n_images]
    else:
        rotations = np.asarray(get_rotation_grid_at_order(int(args.healpix_order)), dtype=np.float32)
        if args.max_rotations is not None:
            rotations = rotations[: int(args.max_rotations)]
    if args.translation_source == "simulation-info-unique":
        if simulation_info is None:
            raise ValueError("--translation-source=simulation-info-unique requires --simulation-info")
        gt_trans = np.asarray(simulation_info["trans"], dtype=np.float32)[:n_images]
        translations, expected_translation_idx = np.unique(gt_trans, axis=0, return_inverse=True)
    else:
        translations = np.asarray(get_translation_grid(float(args.offset_range_px), float(args.offset_step_px)), dtype=np.float32)
        if args.max_translations is not None:
            translations = translations[: int(args.max_translations)]
        expected_translation_idx = None
    rotation_translation_mask = None
    if args.exact_pose_support_mask:
        if args.rotation_source != "simulation-info" or expected_translation_idx is None:
            raise ValueError("--exact-pose-support-mask requires simulator rotations and simulator unique translations")
        rotation_translation_mask = np.zeros((n_images, int(rotations.shape[0]), int(translations.shape[0])), dtype=bool)
        image_rows = np.arange(n_images, dtype=np.int64)
        rotation_translation_mask[image_rows, image_rows, np.asarray(expected_translation_idx, dtype=np.int64)] = True
    noise_variance = _load_noise_variance(args.simulation_info, dataset.image_shape)

    half_size = _half_size(dataset.volume_shape)
    if args.prior_from_init == "gt-row-norm":
        prior_init_npz = args.init_npz if args.prior_init_npz is None else args.prior_init_npz
        prior_mu, prior_W, prior_q = _load_init(prior_init_npz, q_override=q)
        if tuple(prior_mu.shape) != tuple(dataset.volume_shape):
            raise ValueError(
                f"prior init mu shape {prior_mu.shape} does not match dataset volume shape {dataset.volume_shape}",
            )
        if int(prior_q) != int(q):
            raise ValueError(f"prior init q={prior_q} does not match run q={q}")
        mean_prior = jnp.asarray(
            volume_power_variance_prior(
                prior_mu,
                volume_shape=dataset.volume_shape,
                volume_domain="real",
                box_size_power=float(args.gt_prior_box_power),
                scale=float(args.gt_mean_prior_scale),
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
                q=q,
                box_size_power=float(args.gt_prior_box_power),
                scale=float(args.gt_w_prior_scale),
                floor=float(args.gt_prior_floor),
                shell_average=bool(args.gt_prior_shell_average),
                divide_by_q_total=bool(args.gt_w_prior_divide_by_q_total),
                q_total=q,
            ),
            dtype=jnp.float32,
        )
    else:
        prior_init_npz = None
        mean_prior = jnp.full((half_size,), float(args.mean_prior_variance), dtype=jnp.float32)
        W_prior = jnp.full((half_size, q), float(args.W_prior_variance), dtype=jnp.float32)
    prior_summary = {
        "mode": str(args.prior_from_init),
        "prior_init_npz": None if prior_init_npz is None else str(prior_init_npz),
        "mean_prior_min": float(jnp.min(mean_prior)) if half_size else 0.0,
        "mean_prior_median": float(jnp.median(mean_prior)) if half_size else 0.0,
        "mean_prior_max": float(jnp.max(mean_prior)) if half_size else 0.0,
        "W_prior_min": float(jnp.min(W_prior)) if q else 0.0,
        "W_prior_median": float(jnp.median(W_prior)) if q else 0.0,
        "W_prior_max": float(jnp.max(W_prior)) if q else 0.0,
        "gt_prior_box_power": float(args.gt_prior_box_power),
        "gt_w_prior_scale": float(args.gt_w_prior_scale),
        "gt_mean_prior_scale": float(args.gt_mean_prior_scale),
        "gt_prior_floor": float(args.gt_prior_floor),
        "gt_prior_shell_average": bool(args.gt_prior_shell_average),
        "gt_w_prior_divide_by_q_total": bool(args.gt_w_prior_divide_by_q_total),
        "latent_prior": "identity",
        "W_prior_source": (
            "shell_average_sum_k_abs_W_xi_k_squared"
            if args.prior_from_init == "gt-row-norm" and args.gt_prior_shell_average
            else "sum_k_abs_W_xi_k_squared"
            if args.prior_from_init == "gt-row-norm"
            else "constant"
        ),
    }

    iter_summaries = []
    volume_domain = "real"
    current_mu = mu
    current_W = W
    final_result = None
    t_all = time.time()
    for iter_idx in range(1, int(args.n_iters) + 1):
        t0 = time.time()
        input_prior_penalty = _regularization_penalty(
            current_mu,
            current_W,
            mean_prior=mean_prior,
            W_prior=W_prior,
            volume_shape=dataset.volume_shape,
            volume_domain=volume_domain,
            q=q,
        )
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
            rotation_translation_mask=rotation_translation_mask,
            mstep_chunk_size=int(args.mstep_chunk_size),
            half_spectrum_scoring=False,
            square_window=False,
            relion_texture_interp=bool(args.relion_texture_interp),
            skip_empty_pose_blocks=rotation_translation_mask is not None,
            sparse_pass2=bool(args.sparse_pass2),
            sparse_pass2_log_threshold=float(args.sparse_pass2_log_threshold),
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
        gt_pose_diagnostics = None
        if args.rotation_source == "simulation-info":
            best_rot = np.asarray(result.diagnostics["best_rotation_idx"], dtype=np.int64)
            expected_rot = np.arange(n_images, dtype=np.int64)
            gt_pose_diagnostics = {
                "rotation_exact_fraction": float(np.mean(best_rot == expected_rot)),
            }
            if expected_translation_idx is not None:
                best_trans = np.asarray(result.diagnostics["best_translation_idx"], dtype=np.int64)
                gt_pose_diagnostics["translation_exact_fraction"] = float(
                    np.mean(best_trans == np.asarray(expected_translation_idx, dtype=np.int64))
                )
        iter_summary = {
            "iteration": iter_idx,
            "elapsed_s": elapsed_s,
            "npz_path": iter_npz,
            "diagnostics": {
                "log_likelihood": float(result.diagnostics["log_likelihood"]),
                "logZ_mean": float(result.diagnostics["logZ_mean"]),
                "input_prior_penalty": float(input_prior_penalty),
                "input_prior_penalty_mean": float(input_prior_penalty / n_images),
                "input_regularized_objective": float(result.diagnostics["log_likelihood"] + input_prior_penalty),
                "input_regularized_objective_mean": float((result.diagnostics["log_likelihood"] + input_prior_penalty) / n_images),
                "pmax_mean": float(result.diagnostics["pmax_mean"]),
                "nsig_mean": float(result.diagnostics["nsig_mean"]),
                "n_images_accumulated": int(result.stats.n_images),
                "sparse_pass2_total_blocks": int(result.diagnostics.get("sparse_pass2_total_blocks", 0)),
                "sparse_pass2_skipped_blocks": int(result.diagnostics.get("sparse_pass2_skipped_blocks", 0)),
                "sparse_pass2_skipped_fraction": float(result.diagnostics.get("sparse_pass2_skipped_fraction", 0.0)),
                "sparse_pass2_omitted_mass_upper_mean": float(
                    result.diagnostics.get("sparse_pass2_omitted_mass_upper_mean", 0.0)
                ),
                "sparse_pass2_omitted_mass_upper_max": float(
                    result.diagnostics.get("sparse_pass2_omitted_mass_upper_max", 0.0)
                ),
                "score_fourier_size": int(result.diagnostics.get("score_fourier_size", 0)),
                "recon_fourier_size": int(result.diagnostics.get("recon_fourier_size", 0)),
                "full_half_fourier_size": int(result.diagnostics.get("full_half_fourier_size", 0)),
                "uses_fourier_window": bool(result.diagnostics.get("uses_fourier_window", False)),
            },
            "output_stats": {
                "mu_half_shape": list(np.asarray(result.mu_half).shape),
                "W_half_shape": list(np.asarray(result.W_half).shape),
                "mu_half_finite": bool(np.all(np.isfinite(np.asarray(result.mu_half)))),
                "W_half_finite": bool(np.all(np.isfinite(np.asarray(result.W_half)))),
                "mu_half_rms": float(jnp.sqrt(jnp.mean(jnp.abs(result.mu_half) ** 2))),
                "W_half_rms": float(jnp.sqrt(jnp.mean(jnp.abs(result.W_half) ** 2))) if q else 0.0,
                "output_prior_penalty": float(
                    _regularization_penalty(
                        np.asarray(result.mu_half),
                        np.asarray(result.W_half),
                        mean_prior=mean_prior,
                        W_prior=W_prior,
                        volume_shape=dataset.volume_shape,
                        volume_domain="fourier_half",
                        q=q,
                    )
                ),
            },
        }
        if gt_pose_diagnostics is not None:
            iter_summary["gt_pose_diagnostics"] = gt_pose_diagnostics
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
        "rotation_source": str(args.rotation_source),
        "healpix_order": int(args.healpix_order),
        "n_rotations": int(rotations.shape[0]),
        "translation_source": str(args.translation_source),
        "exact_pose_support_mask": bool(args.exact_pose_support_mask),
        "offset_range_px": float(args.offset_range_px),
        "offset_step_px": float(args.offset_step_px),
        "n_translations": int(translations.shape[0]),
        "current_size": int(args.current_size),
        "relion_texture_interp": bool(args.relion_texture_interp),
        "image_batch_size": int(args.image_batch_size),
        "rotation_block_size": int(args.rotation_block_size),
        "sparse_pass2": bool(args.sparse_pass2),
        "sparse_pass2_log_threshold": float(args.sparse_pass2_log_threshold),
        "prior": prior_summary,
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
