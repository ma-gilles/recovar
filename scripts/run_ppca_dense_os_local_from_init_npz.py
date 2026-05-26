#!/usr/bin/env python
"""Run PPCA dense coarse -> adaptive oversampled -> top-p local refinement.

This is the high-resolution experiment driver for already-initialized PPCA
models.  The warmup path only scores poses:

    dense HEALPix order 3 -> adaptive/oversampled order 4 -> top-p local

No mean or W update is performed unless ``--em-iters`` is positive.  EM
iterations start from the input mean/W and rebuild top-p local supports from the
previous iteration diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovar.data_io.cryoem_dataset import load_dataset
from recovar.em.dense_single_volume.local_layout import build_pass2_hypothesis_layout
from recovar.em.sampling import get_relion_rotation_grid
from recovar.em.ppca_refinement.config import (
    GeometryConfig,
    PoseSelectionConfig,
    ScheduleConfig,
    ScoringConfig,
    SparsePass2Config,
)
from recovar.em.ppca_refinement.dense_dataset import compute_dense_ppca_adaptive_significance
from recovar.em.ppca_refinement.highres_refinement import build_top_p_local_hypothesis_layout
from recovar.em.ppca_refinement.local_dataset import (
    run_local_ppca_fused_em_iteration,
    run_local_ppca_pose_scoring_iteration,
)
from recovar.em.ppca_refinement.mean_regularization import (
    KCLASS_RELION_MINRES_MAP,
    MeanRegularizationConfig,
    relion_style_mean_precision_from_stats,
)
from recovar.em.ppca_refinement.postprocess import PostprocessConfig
from recovar.em.ppca_refinement.initialization import (
    loading_row_norm_variance_prior,
    pipeline_variance_W_prior,
    volume_power_variance_prior,
)
from scripts.run_ppca_local_from_init_npz import (
    _half_size,
    _image_ordered_pose_arrays,
    _jsonable,
    _load_init,
    _load_noise_variance,
    _load_simulation_info,
    _regularization_penalty,
    _translations_from_source,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-star", required=True)
    parser.add_argument("--simulation-info")
    parser.add_argument("--init-npz", required=True)
    parser.add_argument("--prior-init-npz", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--n-images", type=int, default=None)
    parser.add_argument("--dense-healpix-order", type=int, default=3)
    parser.add_argument("--os-healpix-order", type=int, default=4)
    parser.add_argument("--final-local-healpix-order", type=int, default=6)
    parser.add_argument("--current-size", type=int, default=256)
    parser.add_argument("--top-p-poses", type=int, default=4)
    parser.add_argument("--top-p-report-widths", default="2,3,4")
    parser.add_argument("--top-p-candidate-pool-factor", type=int, default=8)
    parser.add_argument("--top-p-min-candidate-pool", type=int, default=32)
    parser.add_argument("--top-p-max-log-score-gap", type=float, default=25.0)
    parser.add_argument("--top-p-min-angle-deg", type=float, default=1.0)
    parser.add_argument("--top-p-min-translation-px", type=float, default=0.0)
    parser.add_argument("--adaptive-fraction", type=float, default=0.999)
    parser.add_argument("--max-significants", type=int, default=-1)
    parser.add_argument("--adaptive-translation-step-px", type=float, default=None)
    parser.add_argument(
        "--final-translation-source",
        choices=("coarse", "adaptive"),
        default="coarse",
        help=(
            "Translation grid for the final high-resolution local/EM stage. "
            "'adaptive' keeps OS translation children; 'coarse' keeps the original "
            "grid but centers priors at the OS translation winners."
        ),
    )
    parser.add_argument("--translation-source", choices=("grid", "simulation-info-unique"), default="simulation-info-unique")
    parser.add_argument("--offset-range-px", type=float, default=0.0)
    parser.add_argument("--offset-step-px", type=float, default=1.0)
    parser.add_argument("--max-translations", type=int, default=None)
    parser.add_argument("--image-scale-source", choices=("none", "simulation-info-contrast"), default="simulation-info-contrast")
    parser.add_argument("--sigma-rot-deg", type=float, default=2.0)
    parser.add_argument("--sigma-psi-deg", type=float, default=2.0)
    parser.add_argument("--sigma-offset-angstrom", type=float, default=3.0)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--rotation-block-size", type=int, default=512)
    parser.add_argument("--max-hypotheses-per-microbatch", type=int, default=32768)
    parser.add_argument("--mstep-chunk-size", type=int, default=65536)
    parser.add_argument("--em-iters", type=int, default=0)
    parser.add_argument("--freeze-mean-iters", type=int, default=0)
    parser.add_argument("--local-mstep-top-k", type=int, default=0)
    parser.add_argument("--local-mstep-min-pmax", type=float, default=0.999)
    parser.add_argument("--mean-prior-variance", type=float, default=1.0)
    parser.add_argument("--W-prior-variance", type=float, default=1.0)
    parser.add_argument("--mean-regularization-style", choices=("relion-tau", "variance"), default="relion-tau")
    parser.add_argument("--mean-tau2-fudge", type=float, default=1.0)
    parser.add_argument("--mean-minres-map", type=int, default=KCLASS_RELION_MINRES_MAP)
    parser.add_argument(
        "--prior-from-init",
        choices=(
            "constant",
            "gt-row-norm",
            "pipeline-variance-shell",
            "pipeline-variance-voxel",
            "pipeline-mean-prior",
        ),
        default="gt-row-norm",
        help=(
            "Prior source. 'pipeline-mean-prior' replicates the pipeline's "
            "covariance Wiener form: prior_W = mean_prior * REG_INIT_MULTIPLIER "
            "/ noise / n_pcs. Requires init built with prepare_ppca_init_from_pipeline_output_v2.py."
        ),
    )
    parser.add_argument("--gt-prior-box-power", type=float, default=0.0)
    parser.add_argument("--gt-w-prior-scale", type=float, default=1.0)
    parser.add_argument("--gt-mean-prior-scale", type=float, default=1.0)
    parser.add_argument("--gt-prior-floor", type=float, default=1.0e-8)
    parser.add_argument("--gt-prior-shell-average", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gt-w-prior-divide-by-q-total", action="store_true")
    parser.add_argument(
        "--postprocess-strategy",
        choices=("none", "mean-only", "mean-and-w-mask", "w-only-mask"),
        default="none",
    )
    parser.add_argument(
        "--postprocess-mask-source",
        choices=("none", "radius", "init-volume-mask", "init-volume-mask-dilated"),
        default="none",
        help=(
            "External mask for postprocess. 'init-volume-mask[-dilated]' loads the "
            "saved pipeline solvent mask from the v2 init NPZ. See "
            "scripts/run_ppca_hp6_resume_from_os_npz.py for matching semantics."
        ),
    )
    parser.add_argument("--postprocess-mask-radius-px", type=float, default=None)
    parser.add_argument("--postprocess-cosine-width-px", type=float, default=3.0)
    parser.add_argument("--postprocess-grid-correct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess-gridding-padding-factor", type=float, default=1.0)
    parser.add_argument("--postprocess-gridding-order", type=int, default=1)
    parser.add_argument("--postprocess-gridding-correct", choices=("radial", "square"), default="radial")
    return parser.parse_args()


def _pose_selection(args: argparse.Namespace) -> PoseSelectionConfig:
    return PoseSelectionConfig(
        top_p_poses=int(args.top_p_poses),
        candidate_pool_factor=int(args.top_p_candidate_pool_factor),
        min_candidate_pool=int(args.top_p_min_candidate_pool),
        top_pose_max_log_score_gap=float(args.top_p_max_log_score_gap),
        top_pose_min_angle_deg=float(args.top_p_min_angle_deg),
        top_pose_min_translation_px=float(args.top_p_min_translation_px),
    )


def _pose_arrays_from_dense(sig, rotations: np.ndarray, translations: np.ndarray) -> dict[str, np.ndarray]:
    diag = sig.diagnostics
    best_trans_idx = np.asarray(diag["best_translation_idx"], dtype=np.int64)
    arrays = {
        "best_rotation_idx": np.asarray(diag["best_rotation_idx"], dtype=np.int32),
        "best_rotation_id": np.asarray(diag["best_rotation_id"], dtype=np.int32),
        "best_rotation_matrix": np.asarray(diag["best_rotation_matrix"], dtype=np.float32),
        "best_translation_idx": best_trans_idx.astype(np.int32),
        "best_translation": np.asarray(translations, dtype=np.float32)[best_trans_idx],
        "top_rotation_idx": np.asarray(diag["top_rotation_idx"], dtype=np.int32),
        "top_rotation_id": np.asarray(diag["top_rotation_id"], dtype=np.int32),
        "top_rotation_matrix": np.asarray(diag["top_rotation_matrix"], dtype=np.float32),
        "top_translation_idx": np.asarray(diag["top_translation_idx"], dtype=np.int32),
        "top_log_score": np.asarray(diag["top_log_score"], dtype=np.float32),
        "top_log_score_per_image": np.asarray(diag["top_log_score_per_image"], dtype=np.float32),
        "top_posterior": np.asarray(diag["top_posterior"], dtype=np.float32),
        "top_posterior_per_image": np.asarray(diag["top_posterior_per_image"], dtype=np.float32),
        "max_posterior_per_image": np.asarray(diag["max_posterior_per_image"], dtype=np.float32),
        "n_significant_per_image": np.asarray(sig.n_significant_per_image, dtype=np.int32),
    }
    return arrays


def _save_pose_npz(path: Path, *, mu, W, image_indices, pose_arrays: dict[str, np.ndarray], extra: dict | None = None) -> None:
    payload = {
        "mu_half": np.asarray(mu),
        "W_half": np.asarray(W),
        "image_indices": np.asarray(image_indices, dtype=np.int64),
    }
    payload.update({key: np.asarray(value) for key, value in pose_arrays.items()})
    if extra:
        payload.update({key: np.asarray(value) for key, value in extra.items()})
    np.savez_compressed(path, **payload)


def _build_top_p_layout_from_arrays(
    pose_arrays: dict[str, np.ndarray],
    *,
    center_rotation_grid: np.ndarray,
    target_rotation_grid: np.ndarray,
    healpix_order: int,
    translations: np.ndarray,
    center_translation_grid: np.ndarray | None = None,
    sigma_rot_deg: float,
    sigma_psi_deg: float,
    sigma_offset_angstrom: float,
    voxel_size: float,
):
    return build_top_p_local_hypothesis_layout(
        np.asarray(pose_arrays["top_rotation_id"], dtype=np.int64),
        np.asarray(pose_arrays["top_translation_idx"], dtype=np.int64),
        center_rotation_grid=center_rotation_grid,
        top_rotation_matrices=np.asarray(pose_arrays.get("top_rotation_matrix"), dtype=np.float32)
        if "top_rotation_matrix" in pose_arrays
        else None,
        center_translation_grid=center_translation_grid,
        target_rotation_grid=target_rotation_grid,
        healpix_order=int(healpix_order),
        translations=np.asarray(translations, dtype=np.float32),
        sigma_rot=np.deg2rad(float(sigma_rot_deg)),
        sigma_psi=np.deg2rad(float(sigma_psi_deg)),
        sigma_offset_angstrom=float(sigma_offset_angstrom),
        voxel_size=float(voxel_size),
    )


def _layout_summary(layout) -> dict:
    counts = np.asarray(layout.rotation_counts, dtype=np.int64)
    return {
        "n_images": int(layout.n_images),
        "n_translations": int(layout.translation_grid.shape[0]),
        "rotation_count_min": int(np.min(counts)) if counts.size else 0,
        "rotation_count_median": float(np.median(counts)) if counts.size else 0.0,
        "rotation_count_max": int(np.max(counts)) if counts.size else 0,
        "rotation_count_mean": float(np.mean(counts)) if counts.size else 0.0,
    }


def _top_p_subset_summary(pose_arrays: dict[str, np.ndarray], widths: list[int]) -> dict:
    out = {}
    post = np.asarray(pose_arrays.get("top_posterior_per_image", pose_arrays.get("top_posterior")), dtype=np.float32)
    rot = np.asarray(pose_arrays.get("top_rotation_id", pose_arrays.get("top_rotation_idx")), dtype=np.int32)
    for width in widths:
        width = min(int(width), int(post.shape[1]) if post.ndim == 2 else 0)
        if width <= 0:
            continue
        valid = rot[:, :width] >= 0
        out[f"top{width}_valid_mean"] = float(np.mean(np.sum(valid, axis=1)))
        out[f"top{width}_posterior_mass_mean"] = float(np.mean(np.sum(post[:, :width] * valid, axis=1)))
    return out


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(str(args.data_star))
    mu, W, q, volume_domain = _load_init(args.init_npz, q_override=args.q)
    n_images = int(dataset.n_images) if args.n_images is None else min(int(args.n_images), int(dataset.n_images))
    image_indices = np.arange(n_images, dtype=np.int64)
    simulation_info = _load_simulation_info(args.simulation_info)
    translations = _translations_from_source(args, simulation_info, n_images)
    noise_variance = _load_noise_variance(args.simulation_info, dataset.image_shape)
    pose_selection = _pose_selection(args)
    image_scale_corrections = None
    if args.image_scale_source == "simulation-info-contrast":
        if simulation_info is None or "per_image_contrast" not in simulation_info:
            raise ValueError("--image-scale-source=simulation-info-contrast requires per_image_contrast in simulation_info")
        image_scale_corrections = np.asarray(simulation_info["per_image_contrast"], dtype=np.float32)

    dense_rotations = np.asarray(get_relion_rotation_grid(int(args.dense_healpix_order)), dtype=np.float32)
    os_order = int(args.os_healpix_order)
    final_order = int(args.final_local_healpix_order)
    if os_order < int(args.dense_healpix_order):
        raise ValueError("--os-healpix-order must be >= --dense-healpix-order")
    adaptive_oversampling = os_order - int(args.dense_healpix_order)
    if adaptive_oversampling < 1:
        raise ValueError("OS stage requires --os-healpix-order at least one level above dense order")

    geometry = GeometryConfig(current_size=int(args.current_size), q=q, volume_domain=volume_domain)
    schedule = ScheduleConfig(
        image_batch_size=int(args.image_batch_size),
        rotation_block_size=int(args.rotation_block_size),
        mstep_chunk_size=int(args.mstep_chunk_size),
    )
    scoring = ScoringConfig(image_scale_corrections=image_scale_corrections)
    mean_reg = MeanRegularizationConfig(
        style=str(args.mean_regularization_style).replace("-", "_"),
        tau2_fudge=float(args.mean_tau2_fudge),
        minres_map=int(args.mean_minres_map),
    )

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
    elif args.prior_from_init in ("pipeline-variance-shell", "pipeline-variance-voxel"):
        prior_init_npz = args.init_npz if args.prior_init_npz is None else args.prior_init_npz
        with np.load(prior_init_npz, allow_pickle=False) as z:
            field = (
                "pipeline_variance_prior_half_shell"
                if args.prior_from_init == "pipeline-variance-shell"
                else "pipeline_variance_prior_half_voxel"
            )
            if field not in z.files:
                raise ValueError(
                    f"--prior-from-init {args.prior_from_init} requires '{field}' in {prior_init_npz}; "
                    "rebuild with prepare_ppca_init_from_pipeline_output_v2.py"
                )
            variance_array = np.asarray(z[field], dtype=np.float64)
        if variance_array.shape != (half_size,):
            raise ValueError(
                f"pipeline variance prior in {prior_init_npz} has shape {variance_array.shape}, "
                f"expected ({half_size},)"
            )
        mean_prior = jnp.asarray(
            np.maximum(variance_array * float(args.gt_mean_prior_scale), float(args.gt_prior_floor)).astype(np.float32),
            dtype=jnp.float32,
        )
        W_prior = jnp.asarray(
            pipeline_variance_W_prior(
                variance_array,
                q=q,
                divide_by_q=True,
                floor=float(args.gt_prior_floor),
                scale=float(args.gt_w_prior_scale),
            ),
            dtype=jnp.float32,
        )
    elif args.prior_from_init == "pipeline-mean-prior":
        # ``mean_prior`` here is the FSC-derived per-Fourier-voxel signal-
        # variance prior the pipeline computes from the mean half-maps
        # (recovar.reconstruction.regularization.compute_relion_prior). It's
        # already in image-likelihood signal-variance units — same units the
        # M-step's W prior expects. The user's intended Wiener regularization
        # for the PPCA W M-step is simply:
        #   W_prior[xi, k] = mean_prior[xi] / n_pcs
        # i.e., the per-voxel signal-variance budget split equally across q
        # PCs. The M-step solver evaluates penalty = |W|^2 / W_prior, so the
        # prior precision per voxel per PC is n_pcs / mean_prior[xi]. This
        # matches the user's stated form "D diagonal with 1/n_pcs * variance".
        # We do NOT divide by noise or multiply by REG_INIT_MULTIPLIER here —
        # those appear in the pipeline's covariance Wiener form (regularization_init
        # = mean_prior * REG_INIT_MULTIPLIER / cov_noise) which uses
        # dimensionless SNR-style priors against a CTF^2/noise^2 accumulator;
        # the M-step W solve uses image-likelihood units directly so the
        # natural prior shape is just signal_variance / n_pcs.
        prior_init_npz = args.init_npz if args.prior_init_npz is None else args.prior_init_npz
        with np.load(prior_init_npz, allow_pickle=False) as z:
            if "pipeline_mean_prior_half_voxel" not in z.files:
                raise ValueError(
                    f"--prior-from-init pipeline-mean-prior requires 'pipeline_mean_prior_half_voxel' "
                    f"in {prior_init_npz}; rebuild with prepare_ppca_init_from_pipeline_output_v2.py "
                    "(it recomputes mean_prior from saved half-maps if pipeline didn't save it)"
                )
            mean_prior_half = np.asarray(z["pipeline_mean_prior_half_voxel"], dtype=np.float64).reshape(-1)
        per_pc_prior_variance = mean_prior_half / float(max(1, q))
        per_pc_prior_variance = per_pc_prior_variance * float(args.gt_w_prior_scale)
        if float(args.gt_prior_floor) > 0.0:
            per_pc_prior_variance = np.maximum(per_pc_prior_variance, float(args.gt_prior_floor))
        # Mean prior uses the full per-voxel signal variance (no /n_pcs split since
        # mu is a single scalar per voxel, not q of them).
        mean_prior = jnp.asarray(
            np.maximum(mean_prior_half * float(args.gt_mean_prior_scale), float(args.gt_prior_floor)).astype(np.float32),
            dtype=jnp.float32,
        )
        W_prior = jnp.asarray(
            np.repeat(per_pc_prior_variance.astype(np.float32)[:, None], int(q), axis=1),
            dtype=jnp.float32,
        )
    else:
        mean_prior = jnp.full((half_size,), float(args.mean_prior_variance), dtype=jnp.float32)
        W_prior = jnp.full((half_size, q), float(args.W_prior_variance), dtype=jnp.float32)

    # External pipeline mask for postprocess (--postprocess-mask-source).
    external_mask_volume = None
    if args.postprocess_mask_source not in ("none", "radius"):
        mask_npz = args.init_npz if args.prior_init_npz is None else args.prior_init_npz
        mask_field = (
            "volume_mask_dilated"
            if args.postprocess_mask_source == "init-volume-mask-dilated"
            else "volume_mask"
        )
        with np.load(mask_npz, allow_pickle=False) as z:
            if mask_field not in z.files:
                raise ValueError(
                    f"--postprocess-mask-source {args.postprocess_mask_source} requires '{mask_field}' "
                    f"in {mask_npz}; rebuild with prepare_ppca_init_from_pipeline_output_v2.py"
                )
            external_mask_volume = np.asarray(z[mask_field], dtype=np.float32)
        if external_mask_volume.shape != tuple(dataset.volume_shape):
            raise ValueError(
                f"external mask shape {external_mask_volume.shape} != volume_shape {dataset.volume_shape}"
            )

    report_widths = [int(x) for x in str(args.top_p_report_widths).split(",") if str(x).strip()]
    summary: dict = {
        "data_star": Path(args.data_star),
        "simulation_info": None if args.simulation_info is None else Path(args.simulation_info),
        "init_npz": Path(args.init_npz),
        "output_dir": output_dir,
        "n_images": int(n_images),
        "q": int(q),
        "current_size": int(args.current_size),
        "dense_healpix_order": int(args.dense_healpix_order),
        "os_healpix_order": os_order,
        "final_local_healpix_order": final_order,
        "adaptive_oversampling": int(adaptive_oversampling),
        "top_p_poses": int(args.top_p_poses),
        "translation_source": str(args.translation_source),
        "final_translation_source": str(args.final_translation_source),
        "n_coarse_translations": int(translations.shape[0]),
        "mean_regularization": {
            "style": str(args.mean_regularization_style),
            "tau2_fudge": float(args.mean_tau2_fudge),
            "minres_map": int(args.mean_minres_map),
        },
        "W_prior_source": str(args.prior_from_init),
        "stages": [],
    }

    t0 = time.time()
    dense_sig = compute_dense_ppca_adaptive_significance(
        dataset,
        mu,
        W,
        noise_variance=noise_variance,
        rotations=dense_rotations,
        translations=translations,
        adaptive_fraction=float(args.adaptive_fraction),
        max_significants=int(args.max_significants),
        geometry=geometry,
        schedule=schedule,
        scoring=scoring,
        image_indices=image_indices,
        pose_selection=pose_selection,
    )
    dense_pose = _pose_arrays_from_dense(dense_sig, dense_rotations, translations)
    dense_npz = output_dir / "stage01_dense_hp3_pose_only.npz"
    _save_pose_npz(
        dense_npz,
        mu=mu,
        W=W,
        image_indices=image_indices,
        pose_arrays=dense_pose,
        extra={
            "rotation_grid": dense_rotations,
            "translation_grid": translations,
            "logZ": dense_sig.logZ,
        },
    )
    dense_elapsed = float(time.time() - t0)
    dense_stage = {
        "stage": "dense_pose_only",
        "elapsed_s": dense_elapsed,
        "npz_path": dense_npz,
        "pmax_mean": float(dense_sig.diagnostics["pmax_mean"]),
        "nsig_mean": float(dense_sig.diagnostics["nsig_mean"]),
        "logZ_mean": float(dense_sig.diagnostics["logZ_mean"]),
        **_top_p_subset_summary(dense_pose, report_widths),
    }
    summary["stages"].append(dense_stage)
    print(json.dumps(_jsonable(dense_stage), indent=2, sort_keys=True), flush=True)

    os_layout = build_pass2_hypothesis_layout(
        dense_sig.significant_sample_indices,
        int(dense_rotations.shape[0]),
        int(translations.shape[0]),
        int(args.dense_healpix_order),
        translations,
        oversampling_order=int(adaptive_oversampling),
        translation_step=args.adaptive_translation_step_px,
        allow_empty=False,
    )
    t0 = time.time()
    os_result = run_local_ppca_pose_scoring_iteration(
        dataset,
        mu,
        W,
        noise_variance=noise_variance,
        local_layout=os_layout,
        geometry=geometry,
        schedule=schedule,
        scoring=scoring,
        mean_reg=mean_reg,
        image_indices=image_indices,
        max_hypotheses_per_microbatch=int(args.max_hypotheses_per_microbatch),
        top_pose_count=int(args.top_p_poses),
    )
    jax.block_until_ready(os_result.diagnostics["best_rotation_matrix"])
    os_pose, os_image_indices = _image_ordered_pose_arrays(os_result.diagnostics)
    os_npz = output_dir / "stage02_adaptive_os4_pose_only.npz"
    _save_pose_npz(
        os_npz,
        mu=mu,
        W=W,
        image_indices=os_image_indices,
        pose_arrays=os_pose,
        extra={
            "translation_grid": np.asarray(os_layout.translation_grid, dtype=np.float32),
            "log_likelihood": os_result.diagnostics["log_likelihood"],
        },
    )
    os_elapsed = float(time.time() - t0)
    os_stage = {
        "stage": "adaptive_os_pose_only",
        "elapsed_s": os_elapsed,
        "npz_path": os_npz,
        "layout": _layout_summary(os_layout),
        "pmax_mean": float(os_result.diagnostics["pmax_mean"]),
        "nsig_mean": float(os_result.diagnostics["nsig_mean"]),
        "logZ_mean": float(os_result.diagnostics["logZ_mean"]),
        **_top_p_subset_summary(os_pose, report_widths),
    }
    summary["stages"].append(os_stage)
    print(json.dumps(_jsonable(os_stage), indent=2, sort_keys=True), flush=True)

    current_mu = np.asarray(mu)
    current_W = np.asarray(W)
    current_volume_domain = volume_domain
    current_pose = os_pose
    current_center_grid = np.asarray(get_relion_rotation_grid(os_order), dtype=np.float32)
    current_translations = np.asarray(os_layout.translation_grid, dtype=np.float32)
    final_translations = current_translations if args.final_translation_source == "adaptive" else np.asarray(translations, dtype=np.float32)

    if final_order > os_order:
        final_grid = np.asarray(get_relion_rotation_grid(final_order), dtype=np.float32)
        local_layout = _build_top_p_layout_from_arrays(
            current_pose,
            center_rotation_grid=current_center_grid,
            target_rotation_grid=final_grid,
            healpix_order=final_order,
            translations=final_translations,
            center_translation_grid=current_translations,
            sigma_rot_deg=float(args.sigma_rot_deg),
            sigma_psi_deg=float(args.sigma_psi_deg),
            sigma_offset_angstrom=float(args.sigma_offset_angstrom),
            voxel_size=float(getattr(dataset, "voxel_size", 1.0)),
        )
        t0 = time.time()
        final_pose_result = run_local_ppca_pose_scoring_iteration(
            dataset,
            current_mu,
            current_W,
            noise_variance=noise_variance,
            local_layout=local_layout,
            geometry=GeometryConfig(current_size=int(args.current_size), q=q, volume_domain=current_volume_domain),
            schedule=schedule,
            scoring=scoring,
            mean_reg=mean_reg,
            image_indices=image_indices,
            max_hypotheses_per_microbatch=int(args.max_hypotheses_per_microbatch),
            top_pose_count=int(args.top_p_poses),
        )
        jax.block_until_ready(final_pose_result.diagnostics["best_rotation_matrix"])
        current_pose, final_image_indices = _image_ordered_pose_arrays(final_pose_result.diagnostics)
        final_pose_npz = output_dir / f"stage03_local_hp{final_order}_pose_only.npz"
        _save_pose_npz(
            final_pose_npz,
            mu=current_mu,
            W=current_W,
            image_indices=final_image_indices,
            pose_arrays=current_pose,
            extra={"translation_grid": final_translations, "log_likelihood": final_pose_result.diagnostics["log_likelihood"]},
        )
        final_pose_elapsed = float(time.time() - t0)
        final_pose_stage = {
            "stage": "top_p_local_pose_only",
            "elapsed_s": final_pose_elapsed,
            "npz_path": final_pose_npz,
            "layout": _layout_summary(local_layout),
            "pmax_mean": float(final_pose_result.diagnostics["pmax_mean"]),
            "nsig_mean": float(final_pose_result.diagnostics["nsig_mean"]),
            "logZ_mean": float(final_pose_result.diagnostics["logZ_mean"]),
            **_top_p_subset_summary(current_pose, report_widths),
        }
        summary["stages"].append(final_pose_stage)
        print(json.dumps(_jsonable(final_pose_stage), indent=2, sort_keys=True), flush=True)
        current_center_grid = final_grid
        current_translations = final_translations
    else:
        final_pose_npz = os_npz
        final_grid = current_center_grid

    em_results = []
    for iter_idx in range(1, int(args.em_iters) + 1):
        local_layout = _build_top_p_layout_from_arrays(
            current_pose,
            center_rotation_grid=current_center_grid,
            target_rotation_grid=final_grid,
            healpix_order=final_order,
            translations=current_translations,
            center_translation_grid=current_translations,
            sigma_rot_deg=float(args.sigma_rot_deg),
            sigma_psi_deg=float(args.sigma_psi_deg),
            sigma_offset_angstrom=float(args.sigma_offset_angstrom),
            voxel_size=float(getattr(dataset, "voxel_size", 1.0)),
        )
        fixed_mean_half = None
        if iter_idx <= int(args.freeze_mean_iters):
            from recovar.em.ppca_refinement.dense_dataset import coerce_augmented_half_volumes

            fixed_mean_half = coerce_augmented_half_volumes(
                current_mu,
                current_W,
                volume_shape=dataset.volume_shape,
                q=q,
                volume_domain=current_volume_domain,
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
            mean_reg=mean_reg,
            postprocess=PostprocessConfig(
                strategy=str(args.postprocess_strategy).replace("-", "_"),
                mask_radius_px=args.postprocess_mask_radius_px,
                cosine_width_px=float(args.postprocess_cosine_width_px),
                grid_correct=bool(args.postprocess_grid_correct),
                gridding_padding_factor=float(args.postprocess_gridding_padding_factor),
                gridding_order=int(args.postprocess_gridding_order),
                gridding_correct=str(args.postprocess_gridding_correct),
                external_mask_volume=external_mask_volume,
            ),
            geometry=GeometryConfig(current_size=int(args.current_size), q=q, volume_domain=current_volume_domain),
            schedule=schedule,
            scoring=scoring,
            sparse_pass2=SparsePass2Config(
                enabled=int(args.local_mstep_top_k) > 0,
                local_mstep_top_k=int(args.local_mstep_top_k),
                local_mstep_min_pmax=float(args.local_mstep_min_pmax),
            ),
            image_indices=image_indices,
            max_hypotheses_per_microbatch=int(args.max_hypotheses_per_microbatch),
            fixed_mean_half=fixed_mean_half,
            top_pose_count=int(args.top_p_poses),
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
        input_prior_penalty = _regularization_penalty(
            current_mu,
            np.asarray(current_W),
            mean_prior=mean_prior,
            W_prior=W_prior,
            volume_shape=dataset.volume_shape,
            volume_domain=current_volume_domain,
            q=q,
            mean_precision=mean_precision_for_penalty,
        )
        current_pose, em_image_indices = _image_ordered_pose_arrays(result.diagnostics)
        current_mu = np.asarray(result.mu_half)
        current_W = np.asarray(result.W_half)
        current_volume_domain = "fourier_half"
        em_npz = output_dir / f"stage04_em_iter{iter_idx:03d}.npz"
        _save_pose_npz(
            em_npz,
            mu=current_mu,
            W=current_W,
            image_indices=em_image_indices,
            pose_arrays=current_pose,
            extra={"translation_grid": current_translations, "log_likelihood": result.diagnostics["log_likelihood"]},
        )
        elapsed = float(time.time() - t0)
        em_stage = {
            "stage": "local_em",
            "iteration": iter_idx,
            "elapsed_s": elapsed,
            "npz_path": em_npz,
            "layout": _layout_summary(local_layout),
            "pmax_mean": float(result.diagnostics["pmax_mean"]),
            "nsig_mean": float(result.diagnostics["nsig_mean"]),
            "logZ_mean": float(result.diagnostics["logZ_mean"]),
            "input_prior_penalty": float(input_prior_penalty),
            "postprocess_strategy": str(result.diagnostics.get("postprocess_strategy", "")),
            "mstep_objective_output_total_per_image": float(
                result.diagnostics.get("mstep_objective_output_total_per_image", float("nan"))
            ),
            **_top_p_subset_summary(current_pose, report_widths),
        }
        em_results.append(em_stage)
        summary["stages"].append(em_stage)
        print(json.dumps(_jsonable(em_stage), indent=2, sort_keys=True), flush=True)

    final_npz = output_dir / "final_ppca_dense_os_local.npz"
    _save_pose_npz(
        final_npz,
        mu=current_mu,
        W=current_W,
        image_indices=image_indices,
        pose_arrays=current_pose,
        extra={"translation_grid": current_translations},
    )
    summary["final_npz"] = final_npz
    summary["final_pose_only_npz"] = final_pose_npz
    summary["em_iters_completed"] = int(len(em_results))
    summary["passed"] = bool(np.all(np.isfinite(current_mu)) and np.all(np.isfinite(current_W)))
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable({"summary": summary_path, "final_npz": final_npz, "passed": summary["passed"]}), indent=2))


if __name__ == "__main__":
    main()
