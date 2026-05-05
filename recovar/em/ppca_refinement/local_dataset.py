"""Exact-local PPCA refinement over ``LocalHypothesisLayout`` supports."""

from __future__ import annotations

from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout
from recovar.em.dense_single_volume.helpers.preprocessing import (
    prepare_reconstruction_batch,
    preprocess_batch,
)
from recovar.em.ppca_refinement.dense_dataset import prepare_dense_ppca_dataset_inputs
from recovar.em.ppca_refinement.dense_engine import DensePPCAFusedBlock, run_dense_ppca_fused_refinement_blocks
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState
from recovar.reconstruction import noise as noise_utils


def _local_translation_log_prior(layout: LocalHypothesisLayout, image_index: int) -> np.ndarray:
    prior = np.asarray(layout.translation_log_priors, dtype=np.float32)
    if prior.ndim == 1:
        return prior
    if prior.ndim == 2:
        return prior[int(image_index)]
    raise ValueError(f"translation_log_priors must be 1D or 2D, got {prior.shape}")


def _fetch_single_image_batch(experiment_dataset, image_index: int):
    batch_iter = experiment_dataset.iter_batches(
        1,
        indices=np.asarray([int(image_index)], dtype=np.int64),
        by_image=False,
    )
    try:
        return next(batch_iter)
    except StopIteration as exc:
        raise ValueError(f"Could not fetch image index {image_index}") from exc


def _project_local_augmented(augmented_half_volumes, rotations, image_shape, volume_shape, disc_type, *, max_r):
    from recovar import core

    kwargs = {}
    if max_r is not None:
        kwargs["max_r"] = max_r
    projections = core.batch_slice_volume(
        jnp.asarray(augmented_half_volumes),
        jnp.asarray(rotations),
        image_shape,
        volume_shape,
        disc_type,
        half_volume=True,
        half_image=True,
        **kwargs,
    )
    return jnp.swapaxes(projections, 0, 1)


def iter_local_ppca_dataset_blocks(
    experiment_dataset,
    mu,
    W=None,
    noise_variance=None,
    local_layout: LocalHypothesisLayout | None = None,
    *,
    disc_type: str = "linear_interp",
    current_size: int | None = None,
    q: int | None = None,
    volume_domain: str = "auto",
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    class_log_prior: float = 0.0,
) -> Iterable[tuple[int, np.ndarray, DensePPCAFusedBlock]]:
    """Yield one exact-local PPCA block per image.

    The support is entirely defined by ``LocalHypothesisLayout``. Pruning is
    support-only: candidate masks add ``-inf`` to the same PPCA score algebra
    used by the dense path.
    """

    if local_layout is None:
        raise ValueError("local_layout is required")
    if noise_variance is None:
        raise ValueError("noise_variance is required")
    if int(local_layout.n_images) != int(getattr(experiment_dataset, "n_units", experiment_dataset.n_images)):
        raise ValueError(
            f"local_layout.n_images={local_layout.n_images} does not match dataset image count",
        )

    resolved = prepare_dense_ppca_dataset_inputs(
        experiment_dataset,
        mu,
        W,
        q=q,
        volume_domain=volume_domain,
        current_size=current_size,
        half_spectrum_scoring=half_spectrum_scoring,
        square_window=square_window,
    )
    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, resolved.image_shape).squeeze()
    n_trans = int(local_layout.translation_grid.shape[0])

    for image_index in range(int(local_layout.n_images)):
        start = int(local_layout.rotation_offsets[image_index])
        end = int(local_layout.rotation_offsets[image_index + 1])
        if end <= start:
            continue
        rotations = np.asarray(local_layout.rotations_flat[start:end], dtype=np.float32)
        rotation_ids = np.asarray(local_layout.rotation_ids_flat[start:end], dtype=np.int32)
        batch_data, _rots, _trans, ctf_params, _noise, _particle_indices, _indices = _fetch_single_image_batch(
            experiment_dataset,
            image_index,
        )
        shifted_score_half, batch_norm, ctf2_over_nv_half = preprocess_batch(
            experiment_dataset,
            batch_data,
            ctf_params,
            noise_variance_half,
            local_layout.translation_grid,
            config,
            score_with_masked_images=score_with_masked_images,
        )
        if score_with_masked_images:
            shifted_recon_half = prepare_reconstruction_batch(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                local_layout.translation_grid,
                config,
            )
        else:
            shifted_recon_half = shifted_score_half

        F = int(shifted_score_half.shape[-1])
        Y1_score = shifted_score_half.reshape(1, n_trans, F) * resolved.score_mask[None, None, :]
        ctf2_score = ctf2_over_nv_half * resolved.score_mask[None, :]
        Y1_recon = shifted_recon_half.reshape(1, n_trans, F) * resolved.recon_mask[None, None, :]
        ctf2_recon = ctf2_over_nv_half * resolved.recon_mask[None, :]
        proj_aug = _project_local_augmented(
            resolved.augmented_half_volumes,
            rotations,
            resolved.image_shape,
            resolved.volume_shape,
            disc_type,
            max_r=resolved.projection_max_r,
        )

        rotation_prior = np.asarray(local_layout.rotation_log_priors_flat[start:end], dtype=np.float32)
        translation_prior = _local_translation_log_prior(local_layout, image_index)
        pose_prior = rotation_prior[:, None] + translation_prior[None, :] + float(class_log_prior)
        if local_layout.sample_mask_flat is not None:
            sample_mask = np.asarray(local_layout.sample_mask_flat[start:end], dtype=bool)
            if sample_mask.shape != (end - start, n_trans):
                raise ValueError(f"sample_mask shape {sample_mask.shape} != ({end - start}, {n_trans})")
            pose_prior = np.where(sample_mask, pose_prior, -np.inf)

        yield (
            image_index,
            rotation_ids,
            DensePPCAFusedBlock(
                Y1=Y1_score,
                proj_aug=proj_aug,
                ctf2_over_noise=ctf2_score,
                y_norm=jnp.asarray(batch_norm).reshape(1),
                rotations=jnp.asarray(rotations),
                pose_log_prior=jnp.asarray(pose_prior[None, :, :], dtype=jnp.float32),
                Y1_recon=Y1_recon,
                ctf2_over_noise_recon=ctf2_recon,
            ),
        )


def run_local_ppca_fused_em_iteration(
    experiment_dataset,
    mu,
    W=None,
    *,
    mean_prior,
    W_prior,
    noise_variance,
    local_layout: LocalHypothesisLayout,
    disc_type: str = "linear_interp",
    current_size: int | None = None,
    q: int | None = None,
    volume_domain: str = "auto",
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    class_log_prior: float = 0.0,
    enforce_x0: bool = True,
    mstep_chunk_size: int | None = None,
):
    """Run one exact-local PPCA EM update over a ``LocalHypothesisLayout``."""

    blocks = []
    local_rotation_id_maps = []
    image_indices = []
    for image_index, rotation_ids, block in iter_local_ppca_dataset_blocks(
        experiment_dataset,
        mu,
        W,
        noise_variance,
        local_layout,
        disc_type=disc_type,
        current_size=current_size,
        q=q,
        volume_domain=volume_domain,
        score_with_masked_images=score_with_masked_images,
        half_spectrum_scoring=half_spectrum_scoring,
        square_window=square_window,
        class_log_prior=class_log_prior,
    ):
        image_indices.append(int(image_index))
        local_rotation_id_maps.append(np.asarray(rotation_ids, dtype=np.int32))
        blocks.append(block)

    result = run_dense_ppca_fused_refinement_blocks(
        blocks,
        q=int(jnp.asarray(W_prior).shape[1]),
        image_shape=tuple(int(x) for x in experiment_dataset.image_shape),
        volume_shape=tuple(int(x) for x in experiment_dataset.volume_shape),
        mean_prior=mean_prior,
        W_prior=W_prior,
        disc_type_backproject=disc_type,
        enforce_x0=enforce_x0,
        mstep_chunk_size=mstep_chunk_size,
    )
    best_local = np.asarray(result.diagnostics["best_rotation_idx"], dtype=np.int64)
    best_global = np.zeros_like(best_local, dtype=np.int32)
    for row, local_idx in enumerate(best_local.tolist()):
        best_global[row] = int(local_rotation_id_maps[row][local_idx])
    diagnostics = dict(result.diagnostics)
    diagnostics["image_indices"] = jnp.asarray(image_indices, dtype=jnp.int32)
    diagnostics["best_rotation_id"] = jnp.asarray(best_global, dtype=jnp.int32)
    return result._replace(diagnostics=diagnostics)


def run_local_ppca_halfset_fused_em_iteration(
    state: PoseMarginalPPCAEMState,
    halfset_datasets,
    halfset_local_layouts,
    *,
    disc_type: str = "linear_interp",
    current_size: int | None = None,
    volume_domain: str = "fourier_half",
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    mstep_chunk_size: int | None = None,
) -> PoseMarginalPPCAEMState:
    """Run one exact-local PPCA iteration for two halfsets."""

    from recovar.em.ppca_refinement.dense_dataset import combine_halfset_scoring_model

    if len(halfset_datasets) != 2 or len(halfset_local_layouts) != 2:
        raise ValueError("halfset_datasets and halfset_local_layouts must each have length 2")
    results = []
    for half_dataset, half_layout in zip(halfset_datasets, halfset_local_layouts, strict=True):
        results.append(
            run_local_ppca_fused_em_iteration(
                half_dataset,
                state.mu_score,
                state.W_score,
                mean_prior=state.mean_prior,
                W_prior=state.W_prior,
                noise_variance=state.noise_variance,
                local_layout=half_layout,
                disc_type=disc_type,
                current_size=current_size,
                volume_domain=volume_domain,
                score_with_masked_images=score_with_masked_images,
                half_spectrum_scoring=half_spectrum_scoring,
                square_window=square_window,
                mstep_chunk_size=mstep_chunk_size,
            )
        )
    mu_half = (results[0].mu_half, results[1].mu_half)
    W_half = (results[0].W_half, results[1].W_half)
    mu_score, W_score = combine_halfset_scoring_model(mu_half, W_half)
    pose_diagnostics = {
        "halfset0": results[0].diagnostics,
        "halfset1": results[1].diagnostics,
        "delta_rms_mu": float(jnp.sqrt(jnp.mean(jnp.abs(mu_score - jnp.asarray(state.mu_score)) ** 2))),
        "delta_rms_W": float(jnp.sqrt(jnp.mean(jnp.abs(W_score - jnp.asarray(state.W_score)) ** 2)))
        if jnp.asarray(state.W_score).size
        else 0.0,
    }
    return state.replace(
        mu_half=mu_half,
        W_half=W_half,
        mu_score=mu_score,
        W_score=W_score,
        pose_diagnostics=pose_diagnostics,
    )
