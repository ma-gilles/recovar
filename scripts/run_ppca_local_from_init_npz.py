#!/usr/bin/env python
"""Run exact-local PPCA EM iterations from a PPCA init/result npz."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from recovar.data_io.cryoem_dataset import load_dataset
from recovar.em.dense_single_volume.local_layout import build_local_hypothesis_layout
from recovar.em.ppca_refinement.dense_dataset import coerce_augmented_half_volumes
from recovar.em.ppca_refinement.initialization import loading_row_norm_variance_prior, volume_power_variance_prior
from recovar.em.ppca_refinement.local_dataset import run_local_ppca_fused_em_iteration
from recovar.em.ppca_refinement.mean_regularization import KCLASS_RELION_MINRES_MAP, relion_style_mean_precision_from_stats
from recovar.em.sampling import build_local_search_grid_metadata, get_rotation_grid_at_order, get_translation_grid

from run_ppca_dense_from_init_npz import (
    _float_schedule_value,
    _half_size,
    _jsonable,
    _load_init,
    _load_noise_variance,
    _load_simulation_info,
    _parse_float_schedule,
    _parse_int_schedule,
    _regularization_penalty,
    _schedule_value,
    _write_half_volume_mrc,
)


def _candidate_rotations(
    *,
    source: str,
    simulation_info,
    healpix_order: int,
    n_images: int,
) -> np.ndarray:
    if source == "healpix":
        return np.asarray(get_rotation_grid_at_order(int(healpix_order)), dtype=np.float32)
    if source == "simulation-info":
        if simulation_info is None:
            raise ValueError("--prior-rotation-source=simulation-info requires --simulation-info")
        return np.asarray(simulation_info["rots"], dtype=np.float32)[:n_images]
    if source == "simulation-info-plus-healpix":
        if simulation_info is None:
            raise ValueError("--prior-rotation-source=simulation-info-plus-healpix requires --simulation-info")
        return np.concatenate(
            [
                np.asarray(simulation_info["rots"], dtype=np.float32)[:n_images],
                np.asarray(get_rotation_grid_at_order(int(healpix_order)), dtype=np.float32),
            ],
            axis=0,
        )
    raise ValueError(f"unknown prior rotation source {source!r}")


def _prior_rotations_from_pose_npz(
    pose_npz,
    *,
    source: str,
    simulation_info,
    healpix_order: int,
    n_images: int,
) -> np.ndarray:
    if source == "auto":
        source = "result-matrices" if "best_rotation_matrix" in pose_npz else "healpix"
    if source == "result-matrices":
        if "best_rotation_matrix" not in pose_npz:
            raise ValueError("--prior-rotation-source=result-matrices requires best_rotation_matrix in --pose-npz")
        rotations = np.asarray(pose_npz["best_rotation_matrix"], dtype=np.float32)
        if rotations.shape[0] < n_images:
            raise ValueError(f"best_rotation_matrix has {rotations.shape[0]} rows, need {n_images}")
        return rotations[:n_images]
    best_idx = np.asarray(pose_npz["best_rotation_idx"], dtype=np.int64)[:n_images]
    candidates = _candidate_rotations(
        source=source,
        simulation_info=simulation_info,
        healpix_order=int(healpix_order),
        n_images=n_images,
    )
    if int(np.max(best_idx, initial=0)) >= int(candidates.shape[0]):
        raise ValueError("best_rotation_idx exceeds prior candidate rotation count")
    return candidates[best_idx]


def _translations_from_source(args, simulation_info, n_images: int):
    if args.translation_source == "simulation-info-unique":
        if simulation_info is None:
            raise ValueError("--translation-source=simulation-info-unique requires --simulation-info")
        gt_trans = np.asarray(simulation_info["trans"], dtype=np.float32)[:n_images]
        translations = np.unique(gt_trans, axis=0)
    else:
        translations = np.asarray(get_translation_grid(float(args.offset_range_px), float(args.offset_step_px)), dtype=np.float32)
        if args.max_translations is not None:
            translations = translations[: int(args.max_translations)]
    return translations


def _prior_translations_from_pose_npz(pose_npz, translations: np.ndarray, n_images: int) -> np.ndarray:
    if "best_translation" in pose_npz:
        best = np.asarray(pose_npz["best_translation"], dtype=np.float32)
        if best.shape[0] < n_images:
            raise ValueError(f"best_translation has {best.shape[0]} rows, need {n_images}")
        return best[:n_images]
    best_idx = np.asarray(pose_npz["best_translation_idx"], dtype=np.int64)[:n_images]
    if int(np.max(best_idx, initial=0)) >= int(translations.shape[0]):
        raise ValueError("best_translation_idx exceeds translation candidate count")
    return np.asarray(translations, dtype=np.float32)[best_idx]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-star", required=True)
    parser.add_argument("--simulation-info")
    parser.add_argument("--init-npz", required=True, help="Starting PPCA model, usually a dense PPCA result npz")
    parser.add_argument("--pose-npz", default=None, help="NPZ with prior best poses; default is --init-npz")
    parser.add_argument("--prior-init-npz", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--n-iters", type=int, default=1)
    parser.add_argument("--n-images", type=int, default=None)
    parser.add_argument(
        "--prior-rotation-source",
        choices=("auto", "healpix", "simulation-info", "simulation-info-plus-healpix", "result-matrices"),
        default="auto",
    )
    parser.add_argument("--prior-healpix-order", type=int, default=3)
    parser.add_argument("--local-healpix-order", type=int, default=4)
    parser.add_argument("--sigma-rot-deg", type=float, default=5.0)
    parser.add_argument("--sigma-psi-deg", type=float, default=5.0)
    parser.add_argument("--sigma-offset-angstrom", type=float, default=3.0)
    parser.add_argument("--translation-source", choices=("grid", "simulation-info-unique"), default="grid")
    parser.add_argument("--offset-range-px", type=float, default=6.0)
    parser.add_argument("--offset-step-px", type=float, default=2.0)
    parser.add_argument("--max-translations", type=int, default=None)
    parser.add_argument("--image-scale-source", choices=("none", "simulation-info-contrast"), default="none")
    parser.add_argument("--current-size", type=int, default=64)
    parser.add_argument("--current-size-schedule", default=None)
    parser.add_argument("--freeze-mean-iters", type=int, default=0)
    parser.add_argument("--score-W-scale", type=float, default=1.0)
    parser.add_argument("--score-W-scale-schedule", default=None)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--rotation-block-size", type=int, default=512)
    parser.add_argument("--max-hypotheses-per-microbatch", type=int, default=32768)
    parser.add_argument("--mstep-chunk-size", type=int, default=65536)
    parser.add_argument("--mean-prior-variance", type=float, default=1.0)
    parser.add_argument("--W-prior-variance", type=float, default=1.0)
    parser.add_argument("--mean-regularization-style", choices=("relion-tau", "variance"), default="relion-tau")
    parser.add_argument("--mean-tau2-fudge", type=float, default=1.0)
    parser.add_argument("--mean-minres-map", type=int, default=KCLASS_RELION_MINRES_MAP)
    parser.add_argument("--postprocess-strategy", choices=("none", "mean-only", "mean-and-w-mask"), default="mean-and-w-mask")
    parser.add_argument("--postprocess-mask-radius-px", type=float, default=None)
    parser.add_argument("--postprocess-cosine-width-px", type=float, default=3.0)
    parser.add_argument("--postprocess-grid-correct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess-gridding-padding-factor", type=float, default=1.0)
    parser.add_argument("--postprocess-gridding-order", type=int, default=1)
    parser.add_argument("--postprocess-gridding-correct", choices=("radial", "square"), default="radial")
    parser.add_argument("--prior-from-init", choices=("constant", "gt-row-norm"), default="constant")
    parser.add_argument("--gt-prior-box-power", type=float, default=0.0)
    parser.add_argument("--gt-w-prior-scale", type=float, default=1.0)
    parser.add_argument("--gt-mean-prior-scale", type=float, default=1.0)
    parser.add_argument("--gt-prior-floor", type=float, default=1.0e-8)
    parser.add_argument("--gt-prior-shell-average", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gt-w-prior-divide-by-q-total", action="store_true")
    parser.add_argument("--save-mrc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    current_size_schedule = _parse_int_schedule(args.current_size_schedule, name="current_size_schedule")
    score_W_scale_schedule = _parse_float_schedule(args.score_W_scale_schedule, name="score_W_scale_schedule")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(str(args.data_star))
    mu, W, q, init_volume_domain = _load_init(args.init_npz, q_override=args.q)
    n_images = int(dataset.n_images) if args.n_images is None else min(int(args.n_images), int(dataset.n_images))
    image_indices = np.arange(n_images, dtype=np.int64)
    simulation_info = _load_simulation_info(args.simulation_info)
    pose_npz_path = Path(args.init_npz if args.pose_npz is None else args.pose_npz)
    pose_npz = np.load(pose_npz_path, allow_pickle=True)
    prior_rotations = _prior_rotations_from_pose_npz(
        pose_npz,
        source=str(args.prior_rotation_source),
        simulation_info=simulation_info,
        healpix_order=int(args.prior_healpix_order),
        n_images=n_images,
    )
    translations = _translations_from_source(args, simulation_info, n_images)
    prior_translations = _prior_translations_from_pose_npz(pose_npz, translations, n_images)
    noise_variance = _load_noise_variance(args.simulation_info, dataset.image_shape)

    image_scale_corrections = None
    if args.image_scale_source == "simulation-info-contrast":
        if simulation_info is None or "per_image_contrast" not in simulation_info:
            raise ValueError("--image-scale-source=simulation-info-contrast requires simulation_info['per_image_contrast']")
        image_scale_corrections = np.asarray(simulation_info["per_image_contrast"], dtype=np.float32)
        if image_scale_corrections.shape[0] < n_images:
            raise ValueError(f"per_image_contrast has {image_scale_corrections.shape[0]} entries, need {n_images}")

    half_size = _half_size(dataset.volume_shape)
    if args.prior_from_init == "gt-row-norm":
        prior_init_npz = args.init_npz if args.prior_init_npz is None else args.prior_init_npz
        prior_mu, prior_W, prior_q, prior_volume_domain = _load_init(prior_init_npz, q_override=q)
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
        prior_volume_domain = None
        mean_prior = jnp.full((half_size,), float(args.mean_prior_variance), dtype=jnp.float32)
        W_prior = jnp.full((half_size, q), float(args.W_prior_variance), dtype=jnp.float32)

    local_grid_metadata = build_local_search_grid_metadata(int(args.local_healpix_order))
    current_mu = mu
    current_W = W
    volume_domain = init_volume_domain
    final_result = None
    iter_summaries = []
    t_all = time.time()
    for iter_idx in range(1, int(args.n_iters) + 1):
        iter_current_size = _schedule_value(current_size_schedule, int(args.current_size), iter_idx - 1)
        iter_score_W_scale = _float_schedule_value(score_W_scale_schedule, float(args.score_W_scale), iter_idx - 1)
        freeze_mean = iter_idx <= int(args.freeze_mean_iters)
        local_layout = build_local_hypothesis_layout(
            prior_rotations,
            None,
            np.deg2rad(float(args.sigma_rot_deg)),
            np.deg2rad(float(args.sigma_psi_deg)),
            int(args.local_healpix_order),
            translations,
            prior_translations,
            float(args.sigma_offset_angstrom),
            None,
            float(getattr(dataset, "voxel_size", 1.0)),
            grid_metadata=local_grid_metadata,
            translation_prior_reference_translations=translations,
        )
        fixed_mean_half = None
        if freeze_mean:
            fixed_mean_half = coerce_augmented_half_volumes(
                current_mu,
                current_W,
                volume_shape=dataset.volume_shape,
                q=q,
                volume_domain=volume_domain,
            )[0][0]

        t0 = time.time()
        result = run_local_ppca_fused_em_iteration(
            dataset,
            current_mu,
            current_W,
            mean_prior=mean_prior,
            W_prior=W_prior,
            noise_variance=noise_variance,
            local_layout=local_layout,
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
            current_size=iter_current_size,
            q=q,
            volume_domain=volume_domain,
            image_indices=image_indices,
            image_batch_size=int(args.image_batch_size),
            rotation_block_size=int(args.rotation_block_size),
            max_hypotheses_per_microbatch=int(args.max_hypotheses_per_microbatch),
            image_scale_corrections=image_scale_corrections,
            score_W_scale=float(iter_score_W_scale),
            mstep_chunk_size=int(args.mstep_chunk_size),
            fixed_mean_half=fixed_mean_half,
        )
        jax.block_until_ready(result.mu_half)
        jax.block_until_ready(result.W_half)
        elapsed_s = float(time.time() - t0)
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

        best_rotation_matrix = np.asarray(result.diagnostics["best_rotation_matrix"], dtype=np.float32)
        best_translation = np.asarray(result.diagnostics["best_translation"], dtype=np.float32)
        iter_npz = output_dir / f"iter{iter_idx:03d}.npz"
        np.savez_compressed(
            iter_npz,
            mu_half=np.asarray(result.mu_half),
            W_half=np.asarray(result.W_half),
            best_rotation_idx=np.asarray(result.diagnostics["best_rotation_idx"]),
            best_rotation_id=np.asarray(result.diagnostics["best_rotation_id"]),
            best_rotation_matrix=best_rotation_matrix,
            best_translation_idx=np.asarray(result.diagnostics["best_translation_idx"]),
            best_translation=best_translation,
            image_indices=np.asarray(result.diagnostics["image_indices"]),
            log_likelihood=np.asarray(result.diagnostics["log_likelihood"]),
            logZ_mean=np.asarray(result.diagnostics["logZ_mean"]),
            pmax_mean=np.asarray(result.diagnostics["pmax_mean"]),
            nsig_mean=np.asarray(result.diagnostics["nsig_mean"]),
        )
        iter_summary = {
            "iteration": iter_idx,
            "elapsed_s": elapsed_s,
            "npz_path": iter_npz,
            "current_size": int(iter_current_size),
            "score_W_scale": float(iter_score_W_scale),
            "mean_frozen": bool(freeze_mean),
            "local_rotation_count_min": int(np.min(local_layout.rotation_counts)) if local_layout.n_images else 0,
            "local_rotation_count_median": float(np.median(local_layout.rotation_counts)) if local_layout.n_images else 0.0,
            "local_rotation_count_max": int(np.max(local_layout.rotation_counts)) if local_layout.n_images else 0,
            "diagnostics": {
                "log_likelihood": float(result.diagnostics["log_likelihood"]),
                "logZ_mean": float(result.diagnostics["logZ_mean"]),
                "input_prior_penalty": float(input_prior_penalty),
                "input_regularized_objective": float(result.diagnostics["log_likelihood"] + input_prior_penalty),
                "pmax_mean": float(result.diagnostics["pmax_mean"]),
                "nsig_mean": float(result.diagnostics["nsig_mean"]),
                "n_images_accumulated": int(result.stats.n_images),
                "mean_frozen": bool(result.diagnostics.get("mean_frozen", False)),
                "mstep_mode": str(result.diagnostics.get("mstep_mode", "")),
                "postprocess_strategy": str(result.diagnostics.get("postprocess_strategy", "")),
                "postprocess_warning": str(result.diagnostics.get("postprocess_warning", "")),
                "mstep_objective_output_total_per_image": float(
                    result.diagnostics.get("mstep_objective_output_total_per_image", float("nan"))
                ),
                "mstep_objective_postprocess_delta_per_image": float(
                    result.diagnostics.get("mstep_objective_postprocess_delta_per_image", float("nan"))
                ),
            },
            "output_stats": {
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
        prior_rotations = best_rotation_matrix
        prior_translations = best_translation
        final_result = result

    if final_result is None:
        raise SystemExit("no iterations were run")

    final_npz = output_dir / "final_ppca_local.npz"
    np.savez_compressed(
        final_npz,
        mu_half=np.asarray(final_result.mu_half),
        W_half=np.asarray(final_result.W_half),
        best_rotation_idx=np.asarray(final_result.diagnostics["best_rotation_idx"]),
        best_rotation_id=np.asarray(final_result.diagnostics["best_rotation_id"]),
        best_rotation_matrix=np.asarray(final_result.diagnostics["best_rotation_matrix"]),
        best_translation_idx=np.asarray(final_result.diagnostics["best_translation_idx"]),
        best_translation=np.asarray(final_result.diagnostics["best_translation"]),
        image_indices=np.asarray(final_result.diagnostics["image_indices"]),
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
        "pose_npz": pose_npz_path,
        "init_volume_domain": init_volume_domain,
        "output_dir": output_dir,
        "final_npz": final_npz,
        "written_mrcs": written_mrcs,
        "n_images": int(n_images),
        "q": int(q),
        "n_iters": int(args.n_iters),
        "prior_rotation_source": str(args.prior_rotation_source),
        "prior_healpix_order": int(args.prior_healpix_order),
        "local_healpix_order": int(args.local_healpix_order),
        "sigma_rot_deg": float(args.sigma_rot_deg),
        "sigma_psi_deg": float(args.sigma_psi_deg),
        "sigma_offset_angstrom": float(args.sigma_offset_angstrom),
        "translation_source": str(args.translation_source),
        "n_translations": int(translations.shape[0]),
        "image_scale_source": str(args.image_scale_source),
        "current_size_schedule": current_size_schedule,
        "score_W_scale_schedule": score_W_scale_schedule,
        "freeze_mean_iters": int(args.freeze_mean_iters),
        "image_batch_size": int(args.image_batch_size),
        "rotation_block_size": int(args.rotation_block_size),
        "max_hypotheses_per_microbatch": int(args.max_hypotheses_per_microbatch),
        "prior": {
            "mode": str(args.prior_from_init),
            "prior_init_npz": None if prior_init_npz is None else str(prior_init_npz),
            "prior_volume_domain": prior_volume_domain,
            "mean_regularization_style": str(args.mean_regularization_style),
            "mean_tau2_fudge": float(args.mean_tau2_fudge),
            "mean_minres_map": int(args.mean_minres_map),
        },
        "elapsed_s": float(time.time() - t_all),
        "iterations": iter_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
