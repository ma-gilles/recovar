"""Dataset-facing dense PPCA refinement iteration.

This module is the bridge between the current K-class dense EM infrastructure
and PPCA-specific score/moment algebra. It deliberately reuses the dense
single-volume preprocessing, Fourier-window, CTF/noise, translation, and
projection helpers rather than recreating the stale PPCA branch engine layout.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar import core
import recovar.core.fourier_transform_utils as ftu
from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers.fourier_window import make_fourier_window_spec
from recovar.em.dense_single_volume.helpers.half_spectrum import make_scoring_half_image_weights
from recovar.em.dense_single_volume.helpers.preprocessing import (
    prepare_reconstruction_batch,
    preprocess_batch,
)
from recovar.em.ppca_refinement.dense_engine import (
    DensePPCAFusedBlock,
    DensePPCAFusedEMResult,
    _enforce_augmented_x0,
    dense_pose_ppca_E_step_blocked,
    fused_dense_pose_ppca_block,
)
from recovar.em.ppca_refinement.initialization import real_volume_to_centered_fourier_half
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState
from recovar.ppca import AugmentedPPCAStats, solve_augmented_ppca_mstep
from recovar.ppca.triangular import _tri_size
from recovar.reconstruction import noise as noise_utils


class DensePPCADatasetBlockInputs(NamedTuple):
    """Resolved static inputs for a dense PPCA dataset iteration."""

    augmented_half_volumes: jax.Array
    q: int
    image_shape: tuple[int, int]
    volume_shape: tuple[int, int, int]
    half_volume_size: int
    score_mask: jax.Array
    recon_mask: jax.Array
    projection_max_r: float | None


def _half_volume_size(volume_shape) -> int:
    return int(np.prod(ftu.volume_shape_to_half_volume_shape(tuple(volume_shape))))


def _full_volume_size(volume_shape) -> int:
    return int(np.prod(tuple(volume_shape)))


def _coerce_one_volume_to_half(volume, volume_shape, *, volume_domain: str) -> jax.Array:
    """Return one flattened half-Fourier volume.

    ``volume_domain`` is explicit by design. The ``auto`` mode is restricted to
    obvious unit-test and handoff cases: already half-Fourier, complex full
    Fourier, or real-space grid-shaped volumes.
    """

    volume_shape = tuple(int(x) for x in volume_shape)
    half_size = _half_volume_size(volume_shape)
    full_size = _full_volume_size(volume_shape)
    arr = np.asarray(volume)
    domain = str(volume_domain)
    if domain == "auto":
        if arr.size == half_size:
            domain = "fourier_half"
        elif arr.size == full_size and np.iscomplexobj(arr):
            domain = "fourier_full"
        elif tuple(arr.shape) == volume_shape and not np.iscomplexobj(arr):
            domain = "real"
        else:
            raise ValueError(
                "Could not infer PPCA volume_domain. Pass one of "
                "'fourier_half', 'fourier_full', or 'real'."
            )

    if domain == "fourier_half":
        if arr.size != half_size:
            raise ValueError(f"fourier_half volume has {arr.size} elements, expected {half_size}")
        return jnp.asarray(arr.reshape(-1), dtype=jnp.complex64)
    if domain == "fourier_full":
        if arr.size != full_size:
            raise ValueError(f"fourier_full volume has {arr.size} elements, expected {full_size}")
        full = jnp.asarray(arr.reshape(volume_shape), dtype=jnp.complex64)
        return jnp.asarray(ftu.full_volume_to_half_volume(full, volume_shape).reshape(-1), dtype=jnp.complex64)
    if domain == "real":
        if tuple(arr.shape) != volume_shape:
            raise ValueError(f"real volume shape {arr.shape} != {volume_shape}")
        return jnp.asarray(real_volume_to_centered_fourier_half(arr), dtype=jnp.complex64)
    raise ValueError("volume_domain must be 'auto', 'fourier_half', 'fourier_full', or 'real'")


def _coerce_loading_matrix(W, volume_shape, *, q: int | None) -> tuple[np.ndarray, int]:
    if W is None:
        q_resolved = 0 if q is None else int(q)
        return np.zeros((q_resolved, _half_volume_size(volume_shape)), dtype=np.complex64), q_resolved
    arr = np.asarray(W)
    if q is None:
        if arr.ndim >= 2 and tuple(arr.shape[1:]) == tuple(volume_shape):
            q = int(arr.shape[0])
        elif arr.ndim == 2:
            half_size = _half_volume_size(volume_shape)
            full_size = _full_volume_size(volume_shape)
            if arr.shape[0] in {half_size, full_size}:
                q = int(arr.shape[1])
            else:
                q = int(arr.shape[0])
        else:
            raise ValueError("W must be shaped [q, volume...] or [volume_size, q]")
    return arr, int(q)


def coerce_augmented_half_volumes(
    mu,
    W=None,
    *,
    volume_shape,
    q: int | None = None,
    volume_domain: str = "auto",
) -> tuple[jax.Array, int]:
    """Stack ``[mu, W_1, ..., W_q]`` as flattened half-Fourier volumes."""

    volume_shape = tuple(int(x) for x in volume_shape)
    W_arr, q = _coerce_loading_matrix(W, volume_shape, q=q)
    mu_half = _coerce_one_volume_to_half(mu, volume_shape, volume_domain=volume_domain)
    half_size = _half_volume_size(volume_shape)
    loading_halves = []
    if q:
        if W_arr.ndim >= 2 and tuple(W_arr.shape[1:]) == volume_shape:
            loadings = [W_arr[k] for k in range(q)]
        elif W_arr.ndim == 2 and W_arr.shape == (half_size, q):
            loadings = [W_arr[:, k] for k in range(q)]
        elif W_arr.ndim == 2 and W_arr.shape[0] == q:
            loadings = [W_arr[k] for k in range(q)]
        elif W_arr.ndim == 2 and W_arr.shape[1] == q:
            loadings = [W_arr[:, k] for k in range(q)]
        else:
            raise ValueError(f"Cannot interpret W shape {W_arr.shape} for q={q}")
        loading_halves = [
            _coerce_one_volume_to_half(loading, volume_shape, volume_domain=volume_domain) for loading in loadings
        ]
    aug = jnp.concatenate([mu_half[None, :], jnp.stack(loading_halves, axis=0)], axis=0) if q else mu_half[None, :]
    return aug, q


def _mask_from_indices(n_half: int, indices, *, dtype=jnp.float32) -> jax.Array:
    if indices is None:
        return jnp.ones((int(n_half),), dtype=dtype)
    mask = jnp.zeros((int(n_half),), dtype=dtype)
    return mask.at[jnp.asarray(indices, dtype=jnp.int32)].set(jnp.asarray(1.0, dtype=dtype))


def prepare_dense_ppca_dataset_inputs(
    experiment_dataset,
    mu,
    W=None,
    *,
    volume_shape=None,
    q: int | None = None,
    volume_domain: str = "auto",
    current_size: int | None = None,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
) -> DensePPCADatasetBlockInputs:
    """Resolve augmented volumes and Fourier-window masks for dense PPCA."""

    image_shape = tuple(int(x) for x in experiment_dataset.image_shape)
    volume_shape = tuple(int(x) for x in (experiment_dataset.volume_shape if volume_shape is None else volume_shape))
    H, W_img = image_shape
    n_half = int(H * (W_img // 2 + 1))
    augmented_half, q = coerce_augmented_half_volumes(
        mu,
        W,
        volume_shape=volume_shape,
        q=q,
        volume_domain=volume_domain,
    )
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=True,
    )
    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
    )
    score_mask = _mask_from_indices(n_half, window_spec.score_indices) * half_weights
    recon_mask = _mask_from_indices(n_half, window_spec.recon_indices)
    return DensePPCADatasetBlockInputs(
        augmented_half_volumes=augmented_half,
        q=q,
        image_shape=image_shape,
        volume_shape=volume_shape,
        half_volume_size=_half_volume_size(volume_shape),
        score_mask=score_mask.astype(jnp.float32),
        recon_mask=recon_mask.astype(jnp.float32),
        projection_max_r=window_spec.max_r,
    )


def _project_augmented_half_volumes(
    augmented_half_volumes,
    rotations_block,
    image_shape,
    volume_shape,
    disc_type,
    *,
    max_r,
    relion_texture_interp: bool,
) -> jax.Array:
    kwargs = {}
    if max_r is not None:
        kwargs["max_r"] = max_r
    projections = core.batch_slice_volume(
        jnp.asarray(augmented_half_volumes),
        jnp.asarray(rotations_block),
        image_shape,
        volume_shape,
        disc_type,
        half_volume=True,
        half_image=True,
        relion_texture_interp=relion_texture_interp,
        **kwargs,
    )
    return jnp.swapaxes(projections, 0, 1)


def _per_image_pose_prior_block(
    *,
    batch_start: int,
    batch_count: int,
    r0: int,
    r1: int,
    n_trans: int,
    rotation_log_prior,
    translation_log_prior,
    rotation_translation_mask,
    class_log_prior: float,
) -> jax.Array | None:
    R = int(r1 - r0)
    prior = None
    if rotation_log_prior is not None:
        rot = np.asarray(rotation_log_prior)
        if rot.ndim == 1:
            rot_block = np.broadcast_to(rot[r0:r1][None, :, None], (batch_count, R, n_trans))
        elif rot.ndim == 2:
            rot_block = np.broadcast_to(
                rot[batch_start : batch_start + batch_count, r0:r1, None],
                (batch_count, R, n_trans),
            )
        else:
            raise ValueError("rotation_log_prior must be 1D or 2D")
        prior = rot_block if prior is None else prior + rot_block
    if translation_log_prior is not None:
        trans = np.asarray(translation_log_prior)
        if trans.ndim == 1:
            trans_block = np.broadcast_to(trans[None, None, :], (batch_count, R, n_trans))
        elif trans.ndim == 2:
            trans_block = np.broadcast_to(
                trans[batch_start : batch_start + batch_count, None, :],
                (batch_count, R, n_trans),
            )
        else:
            raise ValueError("translation_log_prior must be 1D or 2D")
        prior = trans_block if prior is None else prior + trans_block
    if rotation_translation_mask is not None:
        mask = np.asarray(rotation_translation_mask, dtype=bool)
        if mask.ndim == 2:
            mask_block = np.broadcast_to(mask[r0:r1][None, :, :], (batch_count, R, n_trans))
        elif mask.ndim == 3:
            mask_block = mask[batch_start : batch_start + batch_count, r0:r1, :]
        else:
            raise ValueError("rotation_translation_mask must be 2D or 3D")
        mask_prior = np.where(mask_block, 0.0, -np.inf)
        prior = mask_prior if prior is None else prior + mask_prior
    if class_log_prior != 0.0:
        class_prior = np.full((batch_count, R, n_trans), float(class_log_prior), dtype=np.float32)
        prior = class_prior if prior is None else prior + class_prior
    return None if prior is None else jnp.asarray(prior, dtype=jnp.float32)


def iter_dense_ppca_dataset_blocks(
    experiment_dataset,
    mu,
    W=None,
    noise_variance=None,
    rotations=None,
    translations=None,
    *,
    disc_type: str = "linear_interp",
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    current_size: int | None = None,
    q: int | None = None,
    volume_domain: str = "auto",
    image_indices: np.ndarray | None = None,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    rotation_translation_mask: np.ndarray | None = None,
    class_log_prior: float = 0.0,
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    relion_texture_interp: bool = True,
) -> Iterable[DensePPCAFusedBlock]:
    """Yield prepared dense PPCA fused blocks directly from a dataset."""

    if rotations is None or translations is None:
        raise ValueError("rotations and translations are required")
    if noise_variance is None:
        raise ValueError("noise_variance is required")
    rotations = np.asarray(rotations, dtype=np.float32)
    translations = np.asarray(translations, dtype=np.float32)
    n_rot = int(rotations.shape[0])
    n_trans = int(translations.shape[0])
    if translations.ndim != 2 or translations.shape[1] != 2:
        raise ValueError(f"translations must have shape [T, 2], got {translations.shape}")

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
    image_indices = np.arange(getattr(experiment_dataset, "n_units", experiment_dataset.n_images)) if image_indices is None else np.asarray(image_indices)
    batch_iter = experiment_dataset.iter_batches(
        image_batch_size,
        indices=image_indices,
        by_image=False,
    )

    batch_start = 0
    for batch_data, _rots, _trans, ctf_params, _noise, _particle_indices, indices in batch_iter:
        batch_count = int(len(indices))
        shifted_score_half, batch_norm, ctf2_over_nv_half = preprocess_batch(
            experiment_dataset,
            batch_data,
            ctf_params,
            noise_variance_half,
            translations,
            config,
            score_with_masked_images=score_with_masked_images,
        )
        if score_with_masked_images:
            shifted_recon_half = prepare_reconstruction_batch(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
            )
        else:
            shifted_recon_half = shifted_score_half

        F = int(shifted_score_half.shape[-1])
        Y1_score = shifted_score_half.reshape(batch_count, n_trans, F) * resolved.score_mask[None, None, :]
        ctf2_score = ctf2_over_nv_half * resolved.score_mask[None, :]
        Y1_recon = shifted_recon_half.reshape(batch_count, n_trans, F) * resolved.recon_mask[None, None, :]
        ctf2_recon = ctf2_over_nv_half * resolved.recon_mask[None, :]
        y_norm = jnp.asarray(batch_norm).reshape(batch_count)

        for r0 in range(0, n_rot, int(rotation_block_size)):
            r1 = min(r0 + int(rotation_block_size), n_rot)
            rotations_block = rotations[r0:r1]
            proj_aug = _project_augmented_half_volumes(
                resolved.augmented_half_volumes,
                rotations_block,
                resolved.image_shape,
                resolved.volume_shape,
                disc_type,
                max_r=resolved.projection_max_r,
                relion_texture_interp=relion_texture_interp,
            )
            pose_log_prior = _per_image_pose_prior_block(
                batch_start=batch_start,
                batch_count=batch_count,
                r0=r0,
                r1=r1,
                n_trans=n_trans,
                rotation_log_prior=rotation_log_prior,
                translation_log_prior=translation_log_prior,
                rotation_translation_mask=rotation_translation_mask,
                class_log_prior=class_log_prior,
            )
            yield DensePPCAFusedBlock(
                Y1=Y1_score,
                proj_aug=proj_aug,
                ctf2_over_noise=ctf2_score,
                y_norm=y_norm,
                rotations=jnp.asarray(rotations_block),
                pose_log_prior=pose_log_prior,
                Y1_recon=Y1_recon,
                ctf2_over_noise_recon=ctf2_recon,
            )
        batch_start += batch_count


def iter_dense_ppca_dataset_block_groups(
    experiment_dataset,
    mu,
    W=None,
    noise_variance=None,
    rotations=None,
    translations=None,
    **kwargs,
) -> Iterable[tuple[DensePPCAFusedBlock, ...]]:
    """Yield all rotation blocks for one image batch as a normalization group."""

    if rotations is None:
        raise ValueError("rotations are required")
    rotation_block_size = int(kwargs.get("rotation_block_size", 5000))
    n_rot = int(np.asarray(rotations).shape[0])
    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    group = []
    for block in iter_dense_ppca_dataset_blocks(
        experiment_dataset,
        mu,
        W,
        noise_variance,
        rotations,
        translations,
        **kwargs,
    ):
        group.append(block)
        if len(group) == n_blocks:
            yield tuple(group)
            group = []
    if group:
        yield tuple(group)


def run_dense_ppca_fused_em_iteration(
    experiment_dataset,
    mu,
    W=None,
    *,
    mean_prior,
    W_prior,
    noise_variance,
    rotations,
    translations,
    disc_type: str = "linear_interp",
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    current_size: int | None = None,
    q: int | None = None,
    volume_domain: str = "auto",
    image_indices: np.ndarray | None = None,
    rotation_log_prior: np.ndarray | None = None,
    translation_log_prior: np.ndarray | None = None,
    rotation_translation_mask: np.ndarray | None = None,
    class_log_prior: float = 0.0,
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    relion_texture_interp: bool = True,
    enforce_x0: bool = True,
    mstep_chunk_size: int | None = None,
) -> DensePPCAFusedEMResult:
    """Run one dataset-backed dense PPCA EM iteration."""

    block_groups = iter_dense_ppca_dataset_block_groups(
        experiment_dataset,
        mu,
        W,
        noise_variance,
        rotations,
        translations,
        disc_type=disc_type,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        current_size=current_size,
        q=q,
        volume_domain=volume_domain,
        image_indices=image_indices,
        rotation_log_prior=rotation_log_prior,
        translation_log_prior=translation_log_prior,
        rotation_translation_mask=rotation_translation_mask,
        class_log_prior=class_log_prior,
        score_with_masked_images=score_with_masked_images,
        half_spectrum_scoring=half_spectrum_scoring,
        square_window=square_window,
        relion_texture_interp=relion_texture_interp,
    )
    q_resolved = int(jnp.asarray(W_prior).shape[1])
    P = q_resolved + 1
    tri = _tri_size(P)
    image_shape = tuple(int(x) for x in experiment_dataset.image_shape)
    volume_shape = tuple(int(x) for x in experiment_dataset.volume_shape)
    mean_prior = jnp.asarray(mean_prior)
    W_prior = jnp.asarray(W_prior)
    if W_prior.shape != (mean_prior.shape[0], q_resolved):
        raise ValueError(f"W_prior shape {W_prior.shape} != ({mean_prior.shape[0]}, {q_resolved})")

    rhs_volume = jnp.zeros((P, mean_prior.shape[0]), dtype=jnp.complex64)
    lhs_tri_volume = jnp.zeros((tri, mean_prior.shape[0]), dtype=jnp.float32)
    log_likelihood = 0.0
    n_images = 0
    pmax_values = []
    nsig_values = []
    best_rotations = []
    best_translations = []

    for group in block_groups:
        if not group:
            continue
        batch_size = int(group[0].Y1.shape[0])
        block_logZ = []
        for block in group:
            _stats, diag = dense_pose_ppca_E_step_blocked(
                block.Y1,
                block.proj_aug,
                block.ctf2_over_noise,
                block.y_norm,
                block.pose_log_prior,
            )
            block_logZ.append(diag.logZ)
        logZ = block_logZ[0]
        for next_logZ in block_logZ[1:]:
            logZ = jnp.logaddexp(logZ, next_logZ)

        log_likelihood += float(jnp.sum(logZ))
        n_images += batch_size
        batch_pmax = jnp.zeros((batch_size,), dtype=jnp.float32)
        batch_nsig = jnp.zeros((batch_size,), dtype=jnp.int32)
        batch_best_score = jnp.full((batch_size,), -jnp.inf)
        batch_best_rotation = jnp.zeros((batch_size,), dtype=jnp.int32)
        batch_best_translation = jnp.zeros((batch_size,), dtype=jnp.int32)

        rotation_offset = 0
        for block in group:
            rhs_volume, lhs_tri_volume, diag = fused_dense_pose_ppca_block(
                block.Y1,
                block.proj_aug,
                block.ctf2_over_noise,
                block.y_norm,
                block.rotations,
                image_shape,
                volume_shape,
                rhs_volume,
                lhs_tri_volume,
                block.pose_log_prior,
                Y1_recon=block.Y1_recon,
                ctf2_over_noise_recon=block.ctf2_over_noise_recon,
                normalization_logZ=logZ,
                disc_type_backproject=disc_type,
            )
            batch_pmax = jnp.maximum(batch_pmax, diag.pmax)
            batch_nsig = batch_nsig + diag.n_significant_per_image
            improve = diag.best_log_score_per_image > batch_best_score
            batch_best_score = jnp.where(improve, diag.best_log_score_per_image, batch_best_score)
            batch_best_rotation = jnp.where(
                improve,
                diag.best_rotation_idx + jnp.asarray(rotation_offset, dtype=jnp.int32),
                batch_best_rotation,
            )
            batch_best_translation = jnp.where(improve, diag.best_translation_idx, batch_best_translation)
            rotation_offset += int(block.rotations.shape[0])

        pmax_values.append(batch_pmax)
        nsig_values.append(batch_nsig)
        best_rotations.append(batch_best_rotation)
        best_translations.append(batch_best_translation)

    if enforce_x0:
        rhs_volume = _enforce_augmented_x0(rhs_volume, volume_shape)
        lhs_tri_volume = _enforce_augmented_x0(lhs_tri_volume.astype(jnp.complex64), volume_shape).real.astype(jnp.float32)

    diagnostics = {
        "pmax_mean": float(jnp.mean(jnp.concatenate(pmax_values))) if pmax_values else float("nan"),
        "nsig_mean": float(jnp.mean(jnp.concatenate(nsig_values))) if nsig_values else float("nan"),
        "log_likelihood": float(log_likelihood),
        "logZ_mean": float(log_likelihood / n_images) if n_images else float("nan"),
        "best_rotation_idx": jnp.concatenate(best_rotations) if best_rotations else jnp.zeros((0,), dtype=jnp.int32),
        "best_translation_idx": jnp.concatenate(best_translations) if best_translations else jnp.zeros((0,), dtype=jnp.int32),
    }
    stats = AugmentedPPCAStats(
        rhs=jnp.swapaxes(rhs_volume, 0, 1),
        lhs_tri=jnp.swapaxes(lhs_tri_volume, 0, 1),
        log_likelihood=log_likelihood,
        n_images=n_images,
        diagnostics=diagnostics,
    )
    mu_half, W_half = solve_augmented_ppca_mstep(
        stats,
        mean_prior=mean_prior,
        W_prior=W_prior,
        chunk_size=mstep_chunk_size,
    )
    return DensePPCAFusedEMResult(mu_half=mu_half, W_half=W_half, stats=stats, diagnostics=diagnostics)


def combine_halfset_scoring_model(mu_half, W_half):
    """Combine halfset PPCA estimates for scoring with per-column sign alignment."""

    mu0, mu1 = (jnp.asarray(mu_half[0]), jnp.asarray(mu_half[1]))
    W0, W1 = (jnp.asarray(W_half[0]), jnp.asarray(W_half[1]))
    mu_score = 0.5 * (mu0 + mu1)
    if W0.shape != W1.shape:
        raise ValueError(f"halfset W shapes differ: {W0.shape} vs {W1.shape}")
    if W0.shape[-1] == 0:
        return mu_score, W0
    dots = jnp.sum(jnp.conj(W0) * W1, axis=0).real
    signs = jnp.where(dots < 0.0, -1.0, 1.0).astype(W1.real.dtype)
    W_score = 0.5 * (W0 + W1 * signs[None, :])
    return mu_score, W_score


def run_dense_ppca_halfset_fused_em_iteration(
    state: PoseMarginalPPCAEMState,
    experiment_dataset,
    *,
    rotations,
    translations,
    disc_type: str = "linear_interp",
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    current_size: int | None = None,
    volume_domain: str = "fourier_half",
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    mstep_chunk_size: int | None = None,
) -> PoseMarginalPPCAEMState:
    """Run one gold-standard halfset dense PPCA iteration and update state."""

    half_datasets = (experiment_dataset.get_halfset(0), experiment_dataset.get_halfset(1))
    results = []
    for half_dataset in half_datasets:
        results.append(
            run_dense_ppca_fused_em_iteration(
                half_dataset,
                state.mu_score,
                state.W_score,
                mean_prior=state.mean_prior,
                W_prior=state.W_prior,
                noise_variance=state.noise_variance,
                rotations=rotations,
                translations=translations,
                disc_type=disc_type,
                image_batch_size=image_batch_size,
                rotation_block_size=rotation_block_size,
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
    old_mu = jnp.asarray(state.mu_score)
    old_W = jnp.asarray(state.W_score)
    pose_diagnostics = {
        "halfset0": results[0].diagnostics,
        "halfset1": results[1].diagnostics,
        "delta_rms_mu": float(jnp.sqrt(jnp.mean(jnp.abs(mu_score - old_mu) ** 2))),
        "delta_rms_W": float(jnp.sqrt(jnp.mean(jnp.abs(W_score - old_W) ** 2))) if old_W.size else 0.0,
    }
    return state.replace(
        mu_half=mu_half,
        W_half=W_half,
        mu_score=mu_score,
        W_score=W_score,
        pose_diagnostics=pose_diagnostics,
    )
