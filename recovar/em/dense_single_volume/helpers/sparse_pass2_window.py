"""Scoring window, RELION flags and per-pass budgets of the sparse bucketed pass 2.

The scoring-window setup shared by both pass-2 entry points, the RELION
behaviour flags they resolve once per pass, the half-spectrum scoring
weights, the projection budget, the two-dimensional fine translation prior,
the shared K-class noise variance and the projected-reference subtraction
applied to sparse M-step sums.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core.configs import ForwardModelConfig
from recovar.em.dense_single_volume.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_flag
from recovar.em.dense_single_volume.helpers.fourier_window import (
    centered_half_indices_to_fftw_half_indices,
    make_fourier_window_spec,
)
from recovar.em.dense_single_volume.helpers.half_spectrum import make_scoring_half_image_weights
from recovar.em.dense_single_volume.helpers.sparse_pass2_budget import (
    _device_memory_limit_bytes,
    _max_projected_rotations_per_call_for_pass,
    _projection_budget_pixels_for_pass,
    _projection_cache_budget_complex_dtype,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_policy import (
    _SPARSE_PASS2_WINDOWED_PREPARE_ENV,
    _windowed_prepare_enabled_for_pass,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_posterior import relion_x_half_f32_fine_posterior_enabled
from recovar.em.dense_single_volume.helpers.sparse_pass2_scoring import _RELION_FINE_DIFF2_FUSED_FFI_ENV
from recovar.em.dense_single_volume.helpers.translation_prior import expand_fine_translation_prior

logger = logging.getLogger(__name__)


class _SparsePass2WindowSetup(NamedTuple):
    """Forward-model configuration and Fourier windows of one sparse pass-2 scorer."""

    config: object
    window_spec: object
    use_window: bool
    window_indices_np: object
    window_indices: object
    recon_window_indices: object
    relion_x_half_recon_indices: object
    windowed_prepare: bool
    n_windowed: int
    n_recon_windowed: int


def _sparse_pass2_window_setup(
    experiment_dataset,
    *,
    disc_type,
    image_shape,
    current_size,
    n_half,
    mstep_current_size,
    square_window,
    window_spec_kwargs,
    use_relion_x_half_mstep,
    log_label,
) -> _SparsePass2WindowSetup:
    """Build the forward model and score/reconstruction windows of a sparse pass 2.

    The RELION x-half M-step addresses its reconstruction window in FFTW half
    order, so the centred reconstruction indices are converted when that
    layout is active. Windowed prepare is logged once per pass with the
    caller's label. The single-class and fused K-class sparse scorers share
    this setup.
    """

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        reconstruction_current_size=mstep_current_size,
        square=square_window,
        include_recon_window=True,
        **window_spec_kwargs,
    )
    use_window = window_spec.use_window
    recon_window_indices = window_spec.recon_indices
    relion_x_half_recon_indices = None
    if use_relion_x_half_mstep:
        centered_recon_indices = (
            recon_window_indices
            if recon_window_indices is not None
            else jnp.arange(int(n_half), dtype=jnp.int32)
        )
        relion_x_half_recon_indices = centered_half_indices_to_fftw_half_indices(
            image_shape,
            centered_recon_indices,
        )
    windowed_prepare = _windowed_prepare_enabled_for_pass(use_window)
    n_windowed = window_spec.n_score
    n_recon_windowed = window_spec.n_recon
    if windowed_prepare:
        logger.info(
            "%s windowed prepare enabled; set %s=0 to disable "
            "(score_pixels=%d recon_pixels=%d full_half_pixels=%d)",
            log_label,
            _SPARSE_PASS2_WINDOWED_PREPARE_ENV,
            int(n_windowed),
            int(n_recon_windowed),
            int(n_half),
        )
    return _SparsePass2WindowSetup(
        config,
        window_spec,
        use_window,
        window_spec.score_indices_np,
        window_spec.score_indices,
        recon_window_indices,
        relion_x_half_recon_indices,
        windowed_prepare,
        n_windowed,
        n_recon_windowed,
    )


@jax.jit
def subtract_projected_reference_from_sparse_mstep_sums(
    summed,
    reconstruction_probs,
    projected_reference,
    ctf2_over_noise,
):
    """Form RELION VDAM's residual backprojection operand on compact support."""

    reconstruction_probs_sum_t = jnp.sum(reconstruction_probs, axis=-1)
    return subtract_projected_reference_from_sparse_mstep_rotation_sums(
        summed,
        reconstruction_probs_sum_t,
        projected_reference,
        ctf2_over_noise,
    )


@jax.jit
def subtract_projected_reference_from_sparse_mstep_rotation_sums(
    summed,
    posterior_mass_by_rotation,
    projected_reference,
    ctf2_over_noise,
):
    """Subtract reference signal after dense or pair-sparse translation sums."""

    projected_reference_weighted = projected_reference * ctf2_over_noise[:, None, :]
    projected_reference_delta = jnp.where(
        posterior_mass_by_rotation[..., None] != 0.0,
        posterior_mass_by_rotation[..., None] * projected_reference_weighted,
        0.0,
    )
    return summed - projected_reference_delta


def _pass2_relion_flags(*, relion_exact_fine_gaussian, relion_firstiter_score_mode, relion_fine_diff2_fused_ffi, relion_f32_fine_posterior):
    """The three RELION fine-scoring flags of a pass 2: exact Gaussian scoring, fused-FFI diff2, float32 fine posterior.

    The fused-FFI and float32-posterior flags may also be switched on by their
    environment variables, read here in the same order as before.
    """

    use_exact_relion_gaussian = bool(
        relion_exact_fine_gaussian
        and relion_firstiter_score_mode == "gaussian"
    )
    use_relion_fine_diff2_fused_ffi = bool(
        relion_fine_diff2_fused_ffi
        or parse_env_flag(_RELION_FINE_DIFF2_FUSED_FFI_ENV, default=False)
    )
    use_relion_f32_fine_posterior = bool(
        relion_f32_fine_posterior
        or relion_x_half_f32_fine_posterior_enabled()
    )
    return use_exact_relion_gaussian, use_relion_fine_diff2_fused_ffi, use_relion_f32_fine_posterior


def _pass2_half_weights(image_shape, window_spec, *, half_spectrum_scoring: bool, relion_firstiter_score_mode, use_float64_scoring: bool):
    """Scoring half-image weights of a pass 2, full and windowed, in the scoring precision."""

    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
        exclude_relion_redundant_x0=relion_firstiter_score_mode != "normalized_cc",
    )
    half_weights_windowed = window_spec.score_values(half_weights)
    if use_float64_scoring:
        half_weights = half_weights.astype(jnp.float64)
        half_weights_windowed = window_spec.score_values(half_weights)
    return half_weights, half_weights_windowed


def _pass2_projection_budget(
    mean_dtype,
    precision_policy: DensePrecisionPolicy,
    *,
    n_half: int,
    use_relion_projector: bool,
    budget_window_spec,
    device_memory_bytes,
    include_abs2: bool,
):
    """Projection cache dtype, budget pixels and rotations per projection call of a pass 2.

    ``include_abs2`` says whether the projection call also materializes the
    squared magnitudes; the single-volume route skips them for score-only
    passes, the fused K-class route only under a window.
    """

    projection_complex_dtype = _projection_cache_budget_complex_dtype(
        mean_dtype,
        precision_policy.score_complex_dtype,
        use_relion_projector=use_relion_projector,
    )
    projection_budget_pixels = _projection_budget_pixels_for_pass(
        n_half,
        use_window=budget_window_spec.use_window,
        use_relion_projector=use_relion_projector,
    )
    max_projected_rotations_per_projection_call = _max_projected_rotations_per_call_for_pass(
        device_memory_bytes=device_memory_bytes,
        n_projection_pixels=projection_budget_pixels,
        projection_complex_dtype=projection_complex_dtype,
        include_abs2=include_abs2,
    )
    return projection_complex_dtype, projection_budget_pixels, max_projected_rotations_per_projection_call


class _Pass2WindowSetup(NamedTuple):
    """Window, memory and precision setup shared by both bucketed pass-2 entry points."""

    mstep_current_size: int | None
    n_half: int
    window_spec_kwargs: dict
    budget_window_spec: object
    device_memory_bytes: int | None
    precision_policy: DensePrecisionPolicy


def _pass2_window_setup(
    image_shape,
    *,
    current_size,
    reconstruction_current_size,
    half_spectrum_scoring: bool,
    square_window: bool,
    relion_firstiter_score_mode,
    use_exact_relion_gaussian: bool,
    use_float64_scoring: bool,
) -> _Pass2WindowSetup:
    """Resolve the M-step current size, the score/recon window and the precision policy of a pass 2."""

    H, W = image_shape
    mstep_current_size = (
        current_size
        if reconstruction_current_size is None
        else int(reconstruction_current_size)
    )
    if (
        use_exact_relion_gaussian
        and current_size is not None
        and int(current_size) < int(W)
        and (not half_spectrum_scoring or square_window)
    ):
        raise NotImplementedError(
            "exact RELION fine Gaussian high-resolution scoring requires "
            "half_spectrum_scoring=True and square_window=False"
        )
    n_half = H * (W // 2 + 1)
    window_spec_kwargs = {}
    if relion_firstiter_score_mode == "normalized_cc":
        window_spec_kwargs = {
            "score_square": True,
            "score_include_dc": True,
        }
    budget_window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        reconstruction_current_size=mstep_current_size,
        square=square_window,
        include_recon_window=True,
        **window_spec_kwargs,
    )
    return _Pass2WindowSetup(
        mstep_current_size=mstep_current_size,
        n_half=n_half,
        window_spec_kwargs=window_spec_kwargs,
        budget_window_spec=budget_window_spec,
        device_memory_bytes=_device_memory_limit_bytes(),
        precision_policy=DensePrecisionPolicy(use_float64_scoring=use_float64_scoring),
    )


def _fine_translation_prior_2d(translation_log_prior, fine_translation_parent, *, n_images, n_fine_trans, dtype):
    """Expand the coarse translation log-prior onto the fine translation grid, or ``None`` without a prior."""

    if translation_log_prior is None:
        return None
    translation_log_prior_np = np.asarray(translation_log_prior, dtype=dtype)
    return expand_fine_translation_prior(
        translation_log_prior_np,
        fine_translation_parent,
        n_images=n_images,
        n_fine_trans=n_fine_trans,
        dtype=dtype,
    )


def _shared_k_class_noise_variance(noise_variance, n_classes: int):
    noise_np = np.asarray(noise_variance)
    if noise_np.ndim >= 2 and int(noise_np.shape[0]) == int(n_classes):
        first = noise_np[0]
        if not np.allclose(noise_np, first[None, ...], rtol=0.0, atol=0.0):
            return None
        return first
    return noise_variance
