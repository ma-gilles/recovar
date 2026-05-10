"""Dataset-facing dense PPCA refinement iteration.

This module is the bridge between the current K-class dense EM infrastructure
and PPCA-specific score/moment algebra. It deliberately reuses the dense
single-volume preprocessing, Fourier-window, CTF/noise, translation, and
projection helpers rather than recreating the stale PPCA branch engine layout.
"""

from __future__ import annotations

from functools import partial
from typing import Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core
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
    dense_pose_ppca_score_stats_blocked,
    fused_dense_pose_ppca_block,
)
from recovar.em.ppca_refinement.initialization import real_volume_to_centered_fourier_half
from recovar.em.ppca_refinement.mean_regularization import (
    KCLASS_RELION_MINRES_MAP,
    relion_style_mean_precision_from_stats,
)
from recovar.em.ppca_refinement.postprocess import postprocess_ppca_half_volumes
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState
from recovar.ppca import AugmentedPPCAStats, augmented_ppca_mstep_objective, solve_augmented_ppca_mstep
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
    score_indices: jax.Array | None
    recon_indices: jax.Array | None
    use_window: bool
    backprojection_max_r: float | None
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
                "Could not infer PPCA volume_domain. Pass one of 'fourier_half', 'fourier_full', or 'real'."
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
        score_indices=window_spec.score_indices,
        recon_indices=window_spec.recon_indices,
        use_window=bool(window_spec.use_window),
        backprojection_max_r=window_spec.max_r,
        projection_max_r=window_spec.max_r,
    )


@partial(
    jax.jit,
    static_argnames=(
        "image_shape",
        "volume_shape",
        "disc_type",
        "max_r",
        "relion_texture_interp",
    ),
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


def _pose_mask_block_has_support(
    rotation_translation_mask, *, batch_start: int, batch_count: int, r0: int, r1: int
) -> bool:
    if rotation_translation_mask is None:
        return True
    mask = np.asarray(rotation_translation_mask, dtype=bool)
    if mask.ndim == 2:
        return bool(np.any(mask[r0:r1, :]))
    if mask.ndim == 3:
        return bool(np.any(mask[batch_start : batch_start + batch_count, r0:r1, :]))
    raise ValueError("rotation_translation_mask must be 2D or 3D")


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
    image_scale_corrections: np.ndarray | None = None,
    class_log_prior: float = 0.0,
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    relion_texture_interp: bool = True,
    skip_empty_pose_blocks: bool = False,
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
    image_indices = (
        np.arange(getattr(experiment_dataset, "n_units", experiment_dataset.n_images))
        if image_indices is None
        else np.asarray(image_indices)
    )
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
        if image_scale_corrections is None:
            batch_scale = jnp.ones((batch_count,), dtype=shifted_score_half.real.dtype)
        else:
            scale_arr = np.asarray(image_scale_corrections, dtype=np.float32)
            batch_scale = jnp.asarray(
                scale_arr[np.asarray(indices, dtype=np.int64)], dtype=shifted_score_half.real.dtype
            )
        batch_scale_sq = batch_scale**2
        Y1_score_full = (
            shifted_score_half.reshape(batch_count, n_trans, F)
            * batch_scale[:, None, None]
            * resolved.score_mask[None, None, :]
        )
        ctf2_score_full = ctf2_over_nv_half * batch_scale_sq[:, None] * resolved.score_mask[None, :]
        Y1_recon_full = (
            shifted_recon_half.reshape(batch_count, n_trans, F)
            * batch_scale[:, None, None]
            * resolved.recon_mask[None, None, :]
        )
        ctf2_recon_full = ctf2_over_nv_half * batch_scale_sq[:, None] * resolved.recon_mask[None, :]
        if resolved.score_indices is None:
            Y1_score = Y1_score_full
            ctf2_score = ctf2_score_full
        else:
            Y1_score = Y1_score_full[:, :, resolved.score_indices]
            ctf2_score = ctf2_score_full[:, resolved.score_indices]
        if resolved.recon_indices is None:
            Y1_recon = Y1_recon_full
            ctf2_recon = ctf2_recon_full
        else:
            Y1_recon = Y1_recon_full[:, :, resolved.recon_indices]
            ctf2_recon = ctf2_recon_full[:, resolved.recon_indices]
        y_norm = jnp.asarray(batch_norm).reshape(batch_count)

        for r0 in range(0, n_rot, int(rotation_block_size)):
            r1 = min(r0 + int(rotation_block_size), n_rot)
            if skip_empty_pose_blocks and not _pose_mask_block_has_support(
                rotation_translation_mask,
                batch_start=batch_start,
                batch_count=batch_count,
                r0=r0,
                r1=r1,
            ):
                continue
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
            if resolved.score_indices is not None:
                proj_aug = proj_aug[:, :, resolved.score_indices]
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
                recon_window_indices=resolved.recon_indices,
                use_recon_window=bool(resolved.use_window),
                backprojection_max_r=resolved.backprojection_max_r,
                batch_start=batch_start,
                rotation_start=r0,
            )
        batch_start += batch_count


def iter_dense_ppca_dataset_block_groups(
    experiment_dataset,
    mu,
    W=None,
    noise_variance=None,
    rotations=None,
    translations=None,
    image_scale_corrections=None,
    **kwargs,
) -> Iterable[tuple[DensePPCAFusedBlock, ...]]:
    """Yield all retained rotation blocks for one image batch as a normalization group."""

    group = []
    current_batch_start = None
    for block in iter_dense_ppca_dataset_blocks(
        experiment_dataset,
        mu,
        W,
        noise_variance,
        rotations,
        translations,
        image_scale_corrections=image_scale_corrections,
        **kwargs,
    ):
        block_batch_start = int(block.batch_start)
        if current_batch_start is None:
            current_batch_start = block_batch_start
        elif block_batch_start != current_batch_start:
            if group:
                yield tuple(group)
            group = []
            current_batch_start = block_batch_start
        group.append(block)
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
    mean_regularization_style: str = "relion_tau",
    mean_tau2_fudge: float = 1.0,
    mean_minres_map: int = KCLASS_RELION_MINRES_MAP,
    postprocess_strategy: str = "mean_and_w_mask",
    postprocess_mask_radius_px: float | None = None,
    postprocess_cosine_width_px: float = 3.0,
    postprocess_grid_correct: bool = True,
    postprocess_gridding_padding_factor: float = 1.0,
    postprocess_gridding_order: int = 1,
    postprocess_gridding_correct: str = "radial",
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
    image_scale_corrections: np.ndarray | None = None,
    class_log_prior: float = 0.0,
    score_with_masked_images: bool = False,
    half_spectrum_scoring: bool = False,
    square_window: bool = False,
    relion_texture_interp: bool = True,
    enforce_x0: bool = True,
    mstep_chunk_size: int | None = None,
    freeze_mean: bool = False,
    skip_empty_pose_blocks: bool = False,
    sparse_pass2: bool = True,
    sparse_pass2_log_threshold: float = float(np.log(1.0e-6)),
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
        image_scale_corrections=image_scale_corrections,
        class_log_prior=class_log_prior,
        score_with_masked_images=score_with_masked_images,
        half_spectrum_scoring=half_spectrum_scoring,
        square_window=square_window,
        relion_texture_interp=relion_texture_interp,
        skip_empty_pose_blocks=skip_empty_pose_blocks,
    )
    q_resolved = int(jnp.asarray(W_prior).shape[1])
    P = q_resolved + 1
    tri = _tri_size(P)
    image_shape = tuple(int(x) for x in experiment_dataset.image_shape)
    volume_shape = tuple(int(x) for x in experiment_dataset.volume_shape)
    if image_scale_corrections is not None:
        scale_arr = np.asarray(image_scale_corrections, dtype=np.float32)
        selected_scale = scale_arr if image_indices is None else scale_arr[np.asarray(image_indices, dtype=np.int64)]
        image_scale_min = float(np.min(selected_scale)) if selected_scale.size else float("nan")
        image_scale_max = float(np.max(selected_scale)) if selected_scale.size else float("nan")
    else:
        image_scale_min = 1.0
        image_scale_max = 1.0
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
    sparse_pass2_total_blocks = 0
    sparse_pass2_skipped_blocks = 0
    sparse_pass2_omitted_mass_upper_sum = 0.0
    sparse_pass2_omitted_mass_upper_max = 0.0
    sparse_pass2_omitted_mass_upper_image_count = 0
    score_fourier_size = None
    recon_fourier_size = None
    postprocess_bandlimit_max_r = None

    for group in block_groups:
        if not group:
            continue
        batch_size = int(group[0].Y1.shape[0])
        if postprocess_bandlimit_max_r is None and bool(group[0].use_recon_window):
            postprocess_bandlimit_max_r = group[0].backprojection_max_r
        if score_fourier_size is None:
            score_fourier_size = int(group[0].Y1.shape[-1])
            recon_fourier_size = (
                int(group[0].Y1_recon.shape[-1]) if group[0].Y1_recon is not None else score_fourier_size
            )
        block_logZ = []
        block_score_stats = []
        block_pose_counts = []
        for block in group:
            score_stats = dense_pose_ppca_score_stats_blocked(
                block.Y1,
                block.proj_aug,
                block.ctf2_over_noise,
                block.y_norm,
                block.pose_log_prior,
            )
            block_score_stats.append(score_stats)
            block_logZ.append(score_stats.logZ)
            block_pose_counts.append(int(block.Y1.shape[1]) * int(block.rotations.shape[0]))
        logZ = block_logZ[0]
        for next_logZ in block_logZ[1:]:
            logZ = jnp.logaddexp(logZ, next_logZ)

        log_likelihood += float(jnp.sum(logZ))
        n_images += batch_size
        batch_nsig = jnp.zeros((batch_size,), dtype=jnp.int32)
        batch_best_score = jnp.full((batch_size,), -jnp.inf)
        batch_best_rotation = jnp.zeros((batch_size,), dtype=jnp.int32)
        batch_best_translation = jnp.zeros((batch_size,), dtype=jnp.int32)

        for block, score_stats in zip(group, block_score_stats, strict=True):
            rotation_offset = int(block.rotation_start)
            improve = score_stats.best_log_score_per_image > batch_best_score
            batch_best_score = jnp.where(improve, score_stats.best_log_score_per_image, batch_best_score)
            batch_best_rotation = jnp.where(
                improve,
                score_stats.best_rotation_idx + jnp.asarray(rotation_offset, dtype=jnp.int32),
                batch_best_rotation,
            )
            batch_best_translation = jnp.where(improve, score_stats.best_translation_idx, batch_best_translation)

        batch_pmax = jnp.exp(batch_best_score - logZ).astype(jnp.float32)
        retained_group = tuple(group)
        if sparse_pass2 and len(group) > 1:
            block_best_matrix = jnp.stack(
                [stats.best_log_score_per_image for stats in block_score_stats],
                axis=0,
            )
            block_log_pose_counts = jnp.log(jnp.asarray(block_pose_counts, dtype=logZ.dtype))[:, None]
            finite_log_z = jnp.isfinite(logZ)
            log_omitted_mass_upper = jnp.where(
                finite_log_z[None, :],
                block_log_pose_counts + block_best_matrix.astype(logZ.dtype) - logZ[None, :].astype(logZ.dtype),
                jnp.inf,
            )
            skip_candidate = log_omitted_mass_upper < float(sparse_pass2_log_threshold)
            skip_pass2_block = np.asarray(jnp.all(skip_candidate, axis=1), dtype=bool)
            if np.all(skip_pass2_block):
                best_block_idx = int(np.argmax(np.asarray(jnp.max(block_best_matrix, axis=1))))
                skip_pass2_block[best_block_idx] = False
            sparse_pass2_total_blocks += len(group)
            sparse_pass2_skipped_blocks += int(skip_pass2_block.sum())
            if np.any(skip_pass2_block):
                skipped_mass_upper = jnp.sum(
                    jnp.where(
                        jnp.asarray(skip_pass2_block)[:, None],
                        jnp.exp(jnp.minimum(log_omitted_mass_upper, 50.0)),
                        0.0,
                    ),
                    axis=0,
                )
                skipped_mass_upper_np = np.asarray(skipped_mass_upper, dtype=np.float64)
                sparse_pass2_omitted_mass_upper_sum += float(np.sum(skipped_mass_upper_np))
                sparse_pass2_omitted_mass_upper_max = max(
                    sparse_pass2_omitted_mass_upper_max,
                    float(np.max(skipped_mass_upper_np)),
                )
                sparse_pass2_omitted_mass_upper_image_count += batch_size
                retained_group = tuple(block for block, skip in zip(group, skip_pass2_block, strict=True) if not skip)

        for block in retained_group:
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
                recon_window_indices=block.recon_window_indices,
                use_recon_window=block.use_recon_window,
                backprojection_max_r=block.backprojection_max_r,
            )
            batch_nsig = batch_nsig + diag.n_significant_per_image
        pmax_values.append(batch_pmax)
        nsig_values.append(batch_nsig)
        best_rotations.append(batch_best_rotation)
        best_translations.append(batch_best_translation)

    if enforce_x0:
        rhs_volume = _enforce_augmented_x0(rhs_volume, volume_shape)
        lhs_tri_volume = _enforce_augmented_x0(lhs_tri_volume.astype(jnp.complex64), volume_shape).real.astype(
            jnp.float32
        )

    diagnostics = {
        "pmax_mean": float(jnp.mean(jnp.concatenate(pmax_values))) if pmax_values else float("nan"),
        "nsig_mean": float(jnp.mean(jnp.concatenate(nsig_values))) if nsig_values else float("nan"),
        "log_likelihood": float(log_likelihood),
        "logZ_mean": float(log_likelihood / n_images) if n_images else float("nan"),
        "best_rotation_idx": jnp.concatenate(best_rotations) if best_rotations else jnp.zeros((0,), dtype=jnp.int32),
        "best_translation_idx": jnp.concatenate(best_translations)
        if best_translations
        else jnp.zeros((0,), dtype=jnp.int32),
        "sparse_pass2_enabled": bool(sparse_pass2),
        "sparse_pass2_log_threshold": float(sparse_pass2_log_threshold),
        "sparse_pass2_total_blocks": int(sparse_pass2_total_blocks),
        "sparse_pass2_skipped_blocks": int(sparse_pass2_skipped_blocks),
        "sparse_pass2_skipped_fraction": (
            float(sparse_pass2_skipped_blocks / sparse_pass2_total_blocks) if sparse_pass2_total_blocks else 0.0
        ),
        "sparse_pass2_omitted_mass_upper_sum": float(sparse_pass2_omitted_mass_upper_sum),
        "sparse_pass2_omitted_mass_upper_max": float(sparse_pass2_omitted_mass_upper_max),
        "sparse_pass2_omitted_mass_upper_mean": (
            float(sparse_pass2_omitted_mass_upper_sum / sparse_pass2_omitted_mass_upper_image_count)
            if sparse_pass2_omitted_mass_upper_image_count
            else 0.0
        ),
        "score_fourier_size": int(score_fourier_size) if score_fourier_size is not None else 0,
        "recon_fourier_size": int(recon_fourier_size) if recon_fourier_size is not None else 0,
        "full_half_fourier_size": int(image_shape[0] * (image_shape[1] // 2 + 1)),
        "uses_fourier_window": bool(
            score_fourier_size is not None and int(score_fourier_size) < int(image_shape[0] * (image_shape[1] // 2 + 1))
        ),
        "mean_regularization_style": str(mean_regularization_style),
        "mean_tau2_fudge": float(mean_tau2_fudge),
        "mean_minres_map": int(mean_minres_map),
        "uses_image_scale_corrections": bool(image_scale_corrections is not None),
        "image_scale_min": float(image_scale_min),
        "image_scale_max": float(image_scale_max),
    }
    stats = AugmentedPPCAStats(
        rhs=jnp.swapaxes(rhs_volume, 0, 1),
        lhs_tri=jnp.swapaxes(lhs_tri_volume, 0, 1),
        log_likelihood=log_likelihood,
        n_images=n_images,
        diagnostics=diagnostics,
    )
    if mean_regularization_style == "variance":
        mean_precision = None
    elif mean_regularization_style == "relion_tau":
        mean_precision = relion_style_mean_precision_from_stats(
            stats,
            mean_prior,
            volume_shape,
            tau2_fudge=float(mean_tau2_fudge),
            minres_map=int(mean_minres_map),
        )
    else:
        raise ValueError(
            f"mean_regularization_style must be 'variance' or 'relion_tau', got {mean_regularization_style!r}"
        )
    input_augmented_half, _input_q = coerce_augmented_half_volumes(
        mu,
        W,
        volume_shape=volume_shape,
        q=q_resolved,
        volume_domain=volume_domain,
    )
    input_mu_half = input_augmented_half[0]
    input_W_half = (
        jnp.swapaxes(input_augmented_half[1:], 0, 1)
        if q_resolved
        else jnp.zeros((mean_prior.shape[0], 0), dtype=input_mu_half.dtype)
    )
    input_objective = augmented_ppca_mstep_objective(
        stats,
        input_mu_half,
        input_W_half,
        mean_prior=mean_prior,
        W_prior=W_prior,
        mean_precision=mean_precision,
        chunk_size=mstep_chunk_size,
    )
    fixed_mean_half = None
    if freeze_mean:
        fixed_mean_half = input_mu_half
    mu_half, W_half = solve_augmented_ppca_mstep(
        stats,
        mean_prior=mean_prior,
        W_prior=W_prior,
        mean_precision=mean_precision,
        fixed_mean=fixed_mean_half,
        chunk_size=mstep_chunk_size,
    )
    solved_objective = augmented_ppca_mstep_objective(
        stats,
        mu_half,
        W_half,
        mean_prior=mean_prior,
        W_prior=W_prior,
        mean_precision=mean_precision,
        chunk_size=mstep_chunk_size,
    )
    postprocessed = postprocess_ppca_half_volumes(
        mu_half,
        W_half,
        volume_shape,
        strategy=postprocess_strategy,
        mask_radius_px=postprocess_mask_radius_px,
        cosine_width_px=postprocess_cosine_width_px,
        grid_correct=postprocess_grid_correct,
        gridding_padding_factor=postprocess_gridding_padding_factor,
        gridding_order=postprocess_gridding_order,
        gridding_correct=postprocess_gridding_correct,
        bandlimit_max_r=postprocess_bandlimit_max_r,
    )
    diagnostics.update(postprocessed.diagnostics)
    mu_half, W_half = postprocessed.mu_half, postprocessed.W_half
    diagnostics["mean_frozen"] = bool(freeze_mean)
    diagnostics["mstep_mode"] = "fixed_mean_conditional_W" if freeze_mean else "joint_mu_W"
    if freeze_mean:
        mu_half = fixed_mean_half
    output_objective = augmented_ppca_mstep_objective(
        stats,
        mu_half,
        W_half,
        mean_prior=mean_prior,
        W_prior=W_prior,
        mean_precision=mean_precision,
        chunk_size=mstep_chunk_size,
    )
    diagnostics.update(input_objective.diagnostics("mstep_objective_input", n_images=n_images))
    diagnostics.update(solved_objective.diagnostics("mstep_objective_solved", n_images=n_images))
    diagnostics.update(output_objective.diagnostics("mstep_objective_output", n_images=n_images))
    diagnostics["mstep_objective_solved_delta"] = float(solved_objective.total - input_objective.total)
    diagnostics["mstep_objective_solved_delta_per_image"] = (
        float((solved_objective.total - input_objective.total) / n_images) if n_images else float("nan")
    )
    diagnostics["mstep_objective_output_delta"] = float(output_objective.total - input_objective.total)
    diagnostics["mstep_objective_output_delta_per_image"] = (
        float((output_objective.total - input_objective.total) / n_images) if n_images else float("nan")
    )
    diagnostics["mstep_objective_postprocess_delta"] = float(output_objective.total - solved_objective.total)
    diagnostics["mstep_objective_postprocess_delta_per_image"] = (
        float((output_objective.total - solved_objective.total) / n_images) if n_images else float("nan")
    )
    diagnostics["mstep_objective_scope"] = "fixed_e_step_augmented_quadratic_without_constants"
    diagnostics["mstep_objective_postprocess_in_objective"] = False
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
    image_scale_corrections: np.ndarray | None = None,
    mean_regularization_style: str = "relion_tau",
    mean_tau2_fudge: float = 1.0,
    mean_minres_map: int = KCLASS_RELION_MINRES_MAP,
    postprocess_strategy: str = "mean_and_w_mask",
    postprocess_mask_radius_px: float | None = None,
    postprocess_cosine_width_px: float = 3.0,
    postprocess_grid_correct: bool = True,
    postprocess_gridding_padding_factor: float = 1.0,
    postprocess_gridding_order: int = 1,
    postprocess_gridding_correct: str = "radial",
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
                image_scale_corrections=image_scale_corrections,
                mean_regularization_style=mean_regularization_style,
                mean_tau2_fudge=mean_tau2_fudge,
                mean_minres_map=mean_minres_map,
                postprocess_strategy=postprocess_strategy,
                postprocess_mask_radius_px=postprocess_mask_radius_px,
                postprocess_cosine_width_px=postprocess_cosine_width_px,
                postprocess_grid_correct=postprocess_grid_correct,
                postprocess_gridding_padding_factor=postprocess_gridding_padding_factor,
                postprocess_gridding_order=postprocess_gridding_order,
                postprocess_gridding_correct=postprocess_gridding_correct,
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
