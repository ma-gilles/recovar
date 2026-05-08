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
from recovar.em.ppca_refinement.mean_regularization import (
    KCLASS_RELION_MINRES_MAP,
    relion_style_mean_precision_from_stats,
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
    if "mu" in init and "W" in init:
        mu = np.asarray(init["mu"], dtype=np.float32)
        W = np.asarray(init["W"], dtype=np.float32)
        if W.ndim != 4:
            raise ValueError(f"expected W shaped [q, N, N, N], got {W.shape}")
        if q_override is not None:
            q = int(q_override)
            if q > int(W.shape[0]):
                raise ValueError(f"q override {q} exceeds init W rank {W.shape[0]}")
            W = W[:q]
        return mu, W, int(W.shape[0]), "real"
    if "mu_half" in init and "W_half" in init:
        mu = np.asarray(init["mu_half"], dtype=np.complex64).reshape(-1)
        W = np.asarray(init["W_half"], dtype=np.complex64)
        if W.ndim != 2:
            raise ValueError(f"expected W_half shaped [half_size, q], got {W.shape}")
        if q_override is not None:
            q = int(q_override)
            if q > int(W.shape[1]):
                raise ValueError(f"q override {q} exceeds init W_half rank {W.shape[1]}")
            W = W[:, :q]
        return mu, W, int(W.shape[1]), "fourier_half"
    raise ValueError(f"{init_path} must contain either mu/W or mu_half/W_half arrays")


def _parse_int_schedule(value: str | None, *, name: str) -> list[int] | None:
    if value is None or str(value).strip() == "":
        return None
    schedule = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not schedule:
        raise ValueError(f"{name} must contain at least one integer")
    if any(item <= 0 for item in schedule):
        raise ValueError(f"{name} entries must be positive, got {schedule}")
    return schedule


def _parse_float_schedule(value: str | None, *, name: str) -> list[float] | None:
    if value is None or str(value).strip() == "":
        return None
    schedule = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not schedule:
        raise ValueError(f"{name} must contain at least one number")
    if any(item < 0.0 for item in schedule):
        raise ValueError(f"{name} entries must be nonnegative, got {schedule}")
    return schedule


def _schedule_value(schedule: list[int] | None, fallback: int, iteration_index: int) -> int:
    if schedule is None:
        return int(fallback)
    return int(schedule[min(int(iteration_index), len(schedule) - 1)])


def _float_schedule_value(schedule: list[float] | None, fallback: float, iteration_index: int) -> float:
    if schedule is None:
        return float(fallback)
    return float(schedule[min(int(iteration_index), len(schedule) - 1)])


def _regularization_penalty(
    mu,
    W,
    *,
    mean_prior,
    W_prior,
    volume_shape,
    volume_domain: str,
    q: int,
    mean_precision=None,
) -> float:
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
    if mean_precision is None:
        mu_penalty = jnp.sum(jnp.abs(mu_half) ** 2 / mean_prior)
    else:
        mu_penalty = jnp.sum(jnp.abs(mu_half) ** 2 * jnp.asarray(mean_precision).real)
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
        choices=("healpix", "simulation-info", "simulation-info-plus-healpix"),
        default="healpix",
        help=(
            "Use a HEALPix grid, the first n simulator rotations, or simulator rotations "
            "with an appended HEALPix distractor grid."
        ),
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
    parser.add_argument(
        "--image-scale-source",
        choices=("none", "simulation-info-contrast"),
        default="none",
        help=(
            "Known per-image signal scale correction. simulation-info-contrast uses synthetic "
            "per_image_contrast so cross/RHS terms scale by s and LHS/template terms by s^2."
        ),
    )
    parser.add_argument("--current-size", type=int, default=16)
    parser.add_argument(
        "--current-size-schedule",
        default=None,
        help=(
            "Comma-separated per-iteration current_size schedule, e.g. '32,48,64'. "
            "If shorter than --n-iters, the last value is reused."
        ),
    )
    parser.add_argument(
        "--freeze-mean-iters",
        type=int,
        default=0,
        help=(
            "For the first N iterations, keep the incoming mean fixed and solve "
            "only W from the conditional augmented normal equations."
        ),
    )
    parser.add_argument(
        "--score-W-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply W by this nonnegative factor only in the PPCA scoring/E-step model. "
            "This is an explicit tempering knob for random-W warm starts; 1.0 is the native model."
        ),
    )
    parser.add_argument(
        "--score-W-scale-schedule",
        default=None,
        help=(
            "Comma-separated per-iteration W scoring scale schedule, e.g. '0.05,0.1,0.25,0.5,1'. "
            "If shorter than --n-iters, the last value is reused."
        ),
    )
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
        "--mean-regularization-style",
        choices=("relion-tau", "variance"),
        default="relion-tau",
        help=(
            "relion-tau treats mean_prior as K-class/RELION tau and converts it "
            "through the same denominator regularization strategy. variance uses "
            "the legacy direct 1/mean_prior diagonal."
        ),
    )
    parser.add_argument("--mean-tau2-fudge", type=float, default=1.0)
    parser.add_argument("--mean-minres-map", type=int, default=KCLASS_RELION_MINRES_MAP)
    parser.add_argument(
        "--postprocess-strategy",
        choices=("none", "mean-only", "mean-and-w-mask"),
        default="mean-and-w-mask",
        help=(
            "Post-solve PPCA volume heuristic. Default applies RELION-style "
            "background-fill masking/grid correction to mu and zero-fill masking/grid "
            "correction to W. This is explicit heuristic behavior, not yet a masked PCG objective."
        ),
    )
    parser.add_argument(
        "--postprocess-mask-radius-px",
        type=float,
        default=None,
        help="Soft-mask radius in pixels. Default uses the RELION-style half-box map mask.",
    )
    parser.add_argument("--postprocess-cosine-width-px", type=float, default=3.0)
    parser.add_argument(
        "--postprocess-grid-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply RELION-style real-space grid correction after post-solve masking.",
    )
    parser.add_argument("--postprocess-gridding-padding-factor", type=float, default=1.0)
    parser.add_argument("--postprocess-gridding-order", type=int, default=1)
    parser.add_argument(
        "--postprocess-gridding-correct",
        choices=("radial", "square"),
        default="radial",
    )
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
        default=0.0,
        help=(
            "When --prior-from-init=gt-row-norm, divide raw DFT power by N**this "
            "value. Default 0 keeps the prior on the same half-Fourier scale as "
            "the augmented M-step unknowns; 2 is the legacy over-shrinking "
            "diagnostic setting."
        ),
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
    current_size_schedule = _parse_int_schedule(args.current_size_schedule, name="current_size_schedule")
    score_W_scale_schedule = _parse_float_schedule(args.score_W_scale_schedule, name="score_W_scale_schedule")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(str(args.data_star))
    mu, W, q, init_volume_domain = _load_init(args.init_npz, q_override=args.q)
    expected_mu_shape = (
        tuple(dataset.volume_shape)
        if init_volume_domain == "real"
        else (_half_size(dataset.volume_shape),)
    )
    if tuple(mu.shape) != expected_mu_shape:
        raise ValueError(f"init mu shape {mu.shape} does not match dataset volume shape {dataset.volume_shape}")

    n_images = int(dataset.n_images) if args.n_images is None else min(int(args.n_images), int(dataset.n_images))
    image_indices = np.arange(n_images, dtype=np.int64)
    simulation_info = _load_simulation_info(args.simulation_info)
    if args.rotation_source == "simulation-info":
        if simulation_info is None:
            raise ValueError("--rotation-source=simulation-info requires --simulation-info")
        rotations = np.asarray(simulation_info["rots"], dtype=np.float32)[:n_images]
    elif args.rotation_source == "simulation-info-plus-healpix":
        if simulation_info is None:
            raise ValueError("--rotation-source=simulation-info-plus-healpix requires --simulation-info")
        simulator_rotations = np.asarray(simulation_info["rots"], dtype=np.float32)[:n_images]
        healpix_rotations = np.asarray(get_rotation_grid_at_order(int(args.healpix_order)), dtype=np.float32)
        if args.max_rotations is not None:
            healpix_rotations = healpix_rotations[: int(args.max_rotations)]
        rotations = np.concatenate([simulator_rotations, healpix_rotations], axis=0)
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
        if args.rotation_source not in ("simulation-info", "simulation-info-plus-healpix") or expected_translation_idx is None:
            raise ValueError(
                "--exact-pose-support-mask requires simulator-containing rotations and simulator unique translations"
            )
        rotation_translation_mask = np.zeros((n_images, int(rotations.shape[0]), int(translations.shape[0])), dtype=bool)
        image_rows = np.arange(n_images, dtype=np.int64)
        rotation_translation_mask[image_rows, image_rows, np.asarray(expected_translation_idx, dtype=np.int64)] = True
    noise_variance = _load_noise_variance(args.simulation_info, dataset.image_shape)
    image_scale_corrections = None
    if args.image_scale_source == "simulation-info-contrast":
        if simulation_info is None or "per_image_contrast" not in simulation_info:
            raise ValueError("--image-scale-source=simulation-info-contrast requires simulation_info['per_image_contrast']")
        image_scale_corrections = np.asarray(simulation_info["per_image_contrast"], dtype=np.float32)
        if image_scale_corrections.shape[0] < n_images:
            raise ValueError(
                f"per_image_contrast has {image_scale_corrections.shape[0]} entries, need at least {n_images}"
            )

    half_size = _half_size(dataset.volume_shape)
    if args.prior_from_init == "gt-row-norm":
        prior_init_npz = args.init_npz if args.prior_init_npz is None else args.prior_init_npz
        prior_mu, prior_W, prior_q, prior_volume_domain = _load_init(prior_init_npz, q_override=q)
        expected_prior_mu_shape = (
            tuple(dataset.volume_shape)
            if prior_volume_domain == "real"
            else (_half_size(dataset.volume_shape),)
        )
        if tuple(prior_mu.shape) != expected_prior_mu_shape:
            raise ValueError(
                f"prior init mu shape {prior_mu.shape} does not match dataset volume shape {dataset.volume_shape}",
            )
        if int(prior_q) != int(q):
            raise ValueError(f"prior init q={prior_q} does not match run q={q}")
        mean_prior = jnp.asarray(
            volume_power_variance_prior(
                prior_mu,
                volume_shape=dataset.volume_shape,
                volume_domain=prior_volume_domain,
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
                volume_domain=prior_volume_domain,
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
        "prior_volume_domain": None if prior_init_npz is None else prior_volume_domain,
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
        "mean_regularization_style": str(args.mean_regularization_style),
        "mean_tau2_fudge": float(args.mean_tau2_fudge),
        "mean_minres_map": int(args.mean_minres_map),
        "postprocess_strategy": str(args.postprocess_strategy),
        "postprocess_mask_radius_px": None
        if args.postprocess_mask_radius_px is None
        else float(args.postprocess_mask_radius_px),
        "postprocess_cosine_width_px": float(args.postprocess_cosine_width_px),
        "postprocess_grid_correct": bool(args.postprocess_grid_correct),
        "postprocess_gridding_padding_factor": float(args.postprocess_gridding_padding_factor),
        "postprocess_gridding_order": int(args.postprocess_gridding_order),
        "postprocess_gridding_correct": str(args.postprocess_gridding_correct),
        "postprocess_warning": "heuristic_post_solve_mask_grid_correction_not_masked_pcg_objective",
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
    volume_domain = init_volume_domain
    current_mu = mu
    current_W = W
    final_result = None
    t_all = time.time()
    for iter_idx in range(1, int(args.n_iters) + 1):
        iter_current_size = _schedule_value(current_size_schedule, int(args.current_size), iter_idx - 1)
        iter_score_W_scale = _float_schedule_value(score_W_scale_schedule, float(args.score_W_scale), iter_idx - 1)
        freeze_mean = iter_idx <= int(args.freeze_mean_iters)
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
            mean_regularization_style=str(args.mean_regularization_style).replace("-", "_"),
            mean_tau2_fudge=float(args.mean_tau2_fudge),
            mean_minres_map=int(args.mean_minres_map),
            postprocess_strategy=str(args.postprocess_strategy).replace("-", "_"),
            postprocess_mask_radius_px=args.postprocess_mask_radius_px,
            postprocess_cosine_width_px=float(args.postprocess_cosine_width_px),
            postprocess_grid_correct=bool(args.postprocess_grid_correct),
            postprocess_gridding_padding_factor=float(args.postprocess_gridding_padding_factor),
            postprocess_gridding_order=int(args.postprocess_gridding_order),
            postprocess_gridding_correct=str(args.postprocess_gridding_correct),
            image_batch_size=int(args.image_batch_size),
            rotation_block_size=int(args.rotation_block_size),
            current_size=iter_current_size,
            q=q,
            volume_domain=volume_domain,
            image_indices=image_indices,
            rotation_translation_mask=rotation_translation_mask,
            image_scale_corrections=image_scale_corrections,
            mstep_chunk_size=int(args.mstep_chunk_size),
            freeze_mean=freeze_mean,
            half_spectrum_scoring=False,
            square_window=False,
            relion_texture_interp=bool(args.relion_texture_interp),
            skip_empty_pose_blocks=rotation_translation_mask is not None,
            sparse_pass2=bool(args.sparse_pass2),
            sparse_pass2_log_threshold=float(args.sparse_pass2_log_threshold),
            score_W_scale=float(iter_score_W_scale),
        )
        jax.block_until_ready(result.mu_half)
        jax.block_until_ready(result.W_half)
        if args.mean_regularization_style == "relion-tau":
            mean_precision_for_penalty = relion_style_mean_precision_from_stats(
                result.stats,
                mean_prior,
                dataset.volume_shape,
                tau2_fudge=float(args.mean_tau2_fudge),
                minres_map=int(args.mean_minres_map),
            )
        else:
            mean_precision_for_penalty = None
        mstep_objective_diagnostics = {}
        for key, value in result.diagnostics.items():
            if not str(key).startswith("mstep_objective_"):
                continue
            if isinstance(value, (bool, str)):
                mstep_objective_diagnostics[str(key)] = value
            else:
                mstep_objective_diagnostics[str(key)] = float(value)
        input_prior_penalty = _regularization_penalty(
            current_mu,
            np.asarray(current_W) * np.asarray(iter_score_W_scale, dtype=np.float32),
            mean_prior=mean_prior,
            W_prior=W_prior,
            volume_shape=dataset.volume_shape,
            volume_domain=volume_domain,
            q=q,
            mean_precision=mean_precision_for_penalty,
        )
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
        if args.rotation_source in ("simulation-info", "simulation-info-plus-healpix"):
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
            "current_size": int(iter_current_size),
            "score_W_scale": float(iter_score_W_scale),
            "mean_frozen": bool(freeze_mean),
            "elapsed_s": elapsed_s,
            "npz_path": iter_npz,
            "diagnostics": {
                "log_likelihood": float(result.diagnostics["log_likelihood"]),
                "logZ_mean": float(result.diagnostics["logZ_mean"]),
                "input_prior_penalty": float(input_prior_penalty),
                "input_prior_penalty_mean": float(input_prior_penalty / n_images),
                "input_regularized_objective": float(result.diagnostics["log_likelihood"] + input_prior_penalty),
                "input_regularized_objective_mean": float((result.diagnostics["log_likelihood"] + input_prior_penalty) / n_images),
                "legacy_logZ_plus_input_prior": float(result.diagnostics["log_likelihood"] + input_prior_penalty),
                "legacy_logZ_plus_input_prior_mean": float(
                    (result.diagnostics["log_likelihood"] + input_prior_penalty) / n_images
                ),
                "objective_accounting_note": (
                    "mstep_objective_* fields are fixed-statistics augmented quadratic terms; "
                    "legacy_logZ_plus_input_prior mixes E-step logZ with prior penalty and is diagnostic-only"
                ),
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
                "score_W_scale": float(result.diagnostics.get("score_W_scale", iter_score_W_scale)),
                "score_W_tempered": bool(result.diagnostics.get("score_W_tempered", iter_score_W_scale != 1.0)),
                "uses_image_scale_corrections": bool(result.diagnostics.get("uses_image_scale_corrections", False)),
                "image_scale_min": float(result.diagnostics.get("image_scale_min", 1.0)),
                "image_scale_max": float(result.diagnostics.get("image_scale_max", 1.0)),
                "mean_regularization_style": str(result.diagnostics.get("mean_regularization_style", "")),
                "mean_tau2_fudge": float(result.diagnostics.get("mean_tau2_fudge", 1.0)),
                "mean_minres_map": int(result.diagnostics.get("mean_minres_map", 0)),
                "mean_frozen": bool(result.diagnostics.get("mean_frozen", False)),
                "mstep_mode": str(result.diagnostics.get("mstep_mode", "")),
                "postprocess_strategy": str(result.diagnostics.get("postprocess_strategy", "")),
                "postprocess_warning": str(result.diagnostics.get("postprocess_warning", "")),
                "postprocess_mask_radius_px": result.diagnostics.get("postprocess_mask_radius_px"),
                "postprocess_cosine_width_px": float(result.diagnostics.get("postprocess_cosine_width_px", 0.0)),
                "postprocess_grid_correct": bool(result.diagnostics.get("postprocess_grid_correct", False)),
                "postprocess_gridding_padding_factor": float(
                    result.diagnostics.get("postprocess_gridding_padding_factor", 0.0)
                ),
                "postprocess_gridding_order": int(result.diagnostics.get("postprocess_gridding_order", 0)),
                "postprocess_gridding_correct": str(result.diagnostics.get("postprocess_gridding_correct", "")),
                "postprocess_mask_mean": float(result.diagnostics.get("postprocess_mask_mean", 0.0)),
                "postprocess_bandlimit_max_r": result.diagnostics.get("postprocess_bandlimit_max_r"),
                "postprocess_bandlimit_fraction": result.diagnostics.get("postprocess_bandlimit_fraction"),
                "postprocess_cap_W_shell_power": bool(
                    result.diagnostics.get("postprocess_cap_W_shell_power", False)
                ),
                "postprocess_W_shell_power_scale_min": float(
                    result.diagnostics.get("postprocess_W_shell_power_scale_min", 1.0)
                ),
                "postprocess_W_shell_power_scale_mean": float(
                    result.diagnostics.get("postprocess_W_shell_power_scale_mean", 1.0)
                ),
                "postprocess_W_shell_power_input_sum": float(
                    result.diagnostics.get("postprocess_W_shell_power_input_sum", 0.0)
                ),
                "postprocess_W_shell_power_pre_cap_sum": float(
                    result.diagnostics.get("postprocess_W_shell_power_pre_cap_sum", 0.0)
                ),
                "postprocess_W_shell_power_output_sum": float(
                    result.diagnostics.get("postprocess_W_shell_power_output_sum", 0.0)
                ),
                **mstep_objective_diagnostics,
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
                        mean_precision=mean_precision_for_penalty,
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

    final_current_size = int(iter_summaries[-1]["current_size"]) if iter_summaries else int(args.current_size)
    summary = {
        "passed": bool(
            np.all(np.isfinite(np.asarray(final_result.mu_half)))
            and np.all(np.isfinite(np.asarray(final_result.W_half)))
            and int(final_result.stats.n_images) == n_images
        ),
        "data_star": Path(args.data_star),
        "simulation_info": None if args.simulation_info is None else Path(args.simulation_info),
        "init_npz": Path(args.init_npz),
        "init_volume_domain": init_volume_domain,
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
        "image_scale_source": str(args.image_scale_source),
        "exact_pose_support_mask": bool(args.exact_pose_support_mask),
        "offset_range_px": float(args.offset_range_px),
        "offset_step_px": float(args.offset_step_px),
        "n_translations": int(translations.shape[0]),
        "current_size": final_current_size,
        "requested_current_size": int(args.current_size),
        "current_size_schedule": current_size_schedule,
        "score_W_scale": float(args.score_W_scale),
        "score_W_scale_schedule": score_W_scale_schedule,
        "freeze_mean_iters": int(args.freeze_mean_iters),
        "relion_texture_interp": bool(args.relion_texture_interp),
        "image_batch_size": int(args.image_batch_size),
        "rotation_block_size": int(args.rotation_block_size),
        "sparse_pass2": bool(args.sparse_pass2),
        "sparse_pass2_log_threshold": float(args.sparse_pass2_log_threshold),
        "postprocess_strategy": str(args.postprocess_strategy),
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
