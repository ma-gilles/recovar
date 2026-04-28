"""Optimized dense single-volume EM engine (v2) with half-spectrum GEMMs.

Key optimizations over engine_fused.py:
1. Two-pass blockwise posterior normalization -- no full (batch, n_rot, n_trans) tensor
2. JIT-compiled per-block E-step and M-step kernels -- eliminates Python dispatch overhead
3. E-step scores computed twice (pass1: logsumexp stats, pass2: normalize+accumulate M-step)
   This trades 2x E-step compute for eliminating the giant residual tensor and
   enabling much larger rotation block sizes.
4. Half-spectrum GEMMs: operate on N_half = H * (W//2+1) instead of N = H*W,
   giving ~2x speedup on all GEMMs (Phase 1 of RELION-parity plan).
5. Coordinate-preserving Fourier windowing: when current_size < full resolution,
   GEMMs operate on only the low-frequency subset of the half-spectrum (N_windowed
   instead of N_half).  This gives ~15x fewer FLOPs at current_size=32.
   (Phase 3 of RELION-parity plan.)

Translation handling (see docs/math/translation_handling_analysis.md):
   Both E-step and M-step use GEMM with explicit shifted-image copies.
   The n_trans factor inflates the GEMM matrices but enables 200x better
   data reuse vs the FFT alternative (1.5 GB vs 327 GB memory traffic).
   GEMM: 45 ms at 47 TFLOPS.  FFT: 1500 ms at 0.7 TFLOPS.  Same result.
   FFT wins only for single-rotation refinement (2x faster per rotation).

Half-spectrum inner product identity (for real-valued images):
   Re<a, b>_full = Re[sum_half w(k) * conj(a(k)) * b(k)]
   where w(k) = 1 for DC and Nyquist columns, w(k) = 2 for interior columns.
   The weights are absorbed into projections (precomputed once per rotation block)
   to avoid extra elementwise multiplies in the hot GEMM loops.

Fourier windowing:
   At low current_size, only frequencies with radius <= current_size//2 are
   included in the GEMMs.  The window is applied as a gather from the full
   half-spectrum after phase shifting (shift-then-gather), preserving correct
   physical frequency spacing.  For the M-step, the windowed GEMM result is
   scattered back to a full half-spectrum before adjoint_slice_volume.
"""

import logging
import os
import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.core.configs import ForwardModelConfig
from recovar.reconstruction import noise as noise_utils

from .dense_big_jit import run_dense_bucket_big_jit
from .helpers.backprojection import (
    adjoint_slice_volume_half as _adjoint_slice_volume_half,
    adjoint_slice_volume_windowed as _adjoint_slice_volume_windowed,
    batch_adjoint_slice_volume_half as _batch_adjoint_slice_volume_half,
    batch_adjoint_slice_volume_windowed as _batch_adjoint_slice_volume_windowed,
)
from .helpers.dtype_policy import DensePrecisionPolicy
from .helpers.env_flags import parse_env_bool
from .helpers.fourier_window import make_fourier_window_spec
from .helpers.half_spectrum import (
    half_spectrum_dc_index,
    make_half_image_weights,
    make_relion_noise_shell_indices_half,
    make_scoring_half_image_weights,
    make_shell_indices_half,
)
from .helpers.image_shifts import apply_relion_integer_pre_shifts, integer_pre_shifts_or_none
from .helpers.jax_runtime import block_until_ready as _block_until_ready
from .helpers.preprocessing import (
    prepare_reconstruction_batch as _prepare_reconstruction_batch,
    process_half_image,
    preprocess_batch as _preprocess_batch,
    preprocess_batch_firstiter_cc as _preprocess_batch_firstiter_cc,
)
from .helpers.projection import (
    DEFAULT_PROJECTION_MAX_R as _DEFAULT_PROJECTION_MAX_R,
    compute_noise_block as _compute_noise_block,
    compute_projections_block as _compute_projections_block,
)
from .helpers.scoring import (
    _e_step_block_scores,
    _e_step_block_scores_windowed,
    _merge_block_logsumexp,
    _m_step_block_compute,
    _m_step_block_windowed,
    _score_rotation_block,
    _update_logsumexp,
    _winner_take_all_probs_for_block,
)
from .helpers.translation_prior import (
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from .helpers.types import EMProfileStats, NoiseStats, RelionStats
from .local_debug import (
    maybe_write_dense_per_pose_score_dump,
    parse_dense_noise_component_dump_request,
    parse_dense_per_pose_score_dump_request,
)
from .shape_buckets import pad_axis, pad_batch_data_ctf_and_valid_mask

logger = logging.getLogger(__name__)

# TRACKED TODOs: DENSE_ENGINE_BOUNDARY
# TODO(DENSE_ENGINE_BOUNDARY/E001): this file is dense/global-only, local search belongs in local_em_engine.py
# TODO(DENSE_ENGINE_BOUNDARY/E002): extract shared primitives, do not let local logic grow back here
# TODO(DENSE_ENGINE_BOUNDARY/E004): dtype-policy cleanup needed, reduce ad hoc casts and flags
# TODO(DENSE_ENGINE_BOUNDARY/E005): audit em_engine.py vs local_em_engine.py for copied implementations
# See docs/relion_local_engine_refactor.md


def _noise_split_diagnostics_requested() -> bool:
    """Return whether per-shell A2/XA noise split diagnostics are needed."""
    return bool(
        os.environ.get("RECOVAR_NOISE_DEBUG_DUMP_DIR")
        or os.environ.get("RECOVAR_DENSE_NOISE_COMPONENT_DUMP_DIR")
    )


def _dense_big_jit_enabled() -> bool:
    """Return whether the experimental dense/global bucket big-JIT is enabled.

    The default is on. Unsupported dense variants still fall back before the
    batch loop, so the main RELION path gets the compiled bucket boundary where
    eligible without mixing it into sparse/local/debug code paths.
    """

    return parse_env_bool("RECOVAR_RELION_DENSE_BIG_JIT", default=True)


def _dense_big_jit_disabled_reason(
    *,
    relion_firstiter_winner_take_all: bool,
    accumulate_noise: bool,
    dense_noise_component_dump_enabled: bool,
    per_pose_debug_dump_enabled: bool,
) -> str | None:
    """Return the dense big-JIT fallback reason, or ``None`` if eligible."""

    if relion_firstiter_winner_take_all:
        return "winner_take_all"
    if accumulate_noise:
        return "noise_accumulation"
    if dense_noise_component_dump_enabled:
        return "dense_noise_component_dump"
    if per_pose_debug_dump_enabled:
        return "per_pose_debug_dump"
    return None


def _bin_shell_values_np(values, shell_indices, n_shells):
    return np.bincount(
        np.asarray(shell_indices, dtype=np.int64),
        weights=np.asarray(values, dtype=np.float64),
        minlength=int(n_shells),
    )[: int(n_shells)]


_DENSE_TIMING_FIELDS = (
    "batch_fetch_s",
    "preprocess_s",
    "score_prep_s",
    "pass1_projection_s",
    "pass1_score_s",
    "pass1_postprocess_s",
    "pass1_logsumexp_s",
    "pass2_skipmask_s",
    "pass2_projection_s",
    "pass2_score_s",
    "pass2_postprocess_s",
    "mstep_s",
    "window_scatter_s",
    "adjoint_y_s",
    "adjoint_ctf_s",
    "noise_s",
    "assignment_s",
    "stats_finalize_s",
    "host_stats_s",
    "solve_s",
)


@dataclass
class _DenseTiming:
    """Mutable host-side timers for one dense EM call."""

    batch_fetch_s: float = 0.0
    preprocess_s: float = 0.0
    score_prep_s: float = 0.0
    pass1_projection_s: float = 0.0
    pass1_score_s: float = 0.0
    pass1_postprocess_s: float = 0.0
    pass1_logsumexp_s: float = 0.0
    pass2_skipmask_s: float = 0.0
    pass2_projection_s: float = 0.0
    pass2_score_s: float = 0.0
    pass2_postprocess_s: float = 0.0
    mstep_s: float = 0.0
    window_scatter_s: float = 0.0
    adjoint_y_s: float = 0.0
    adjoint_ctf_s: float = 0.0
    noise_s: float = 0.0
    assignment_s: float = 0.0
    stats_finalize_s: float = 0.0
    host_stats_s: float = 0.0
    solve_s: float = 0.0

    def accounted_s(self) -> float:
        return sum(float(getattr(self, field)) for field in _DENSE_TIMING_FIELDS)

    def profile_kwargs(self) -> dict[str, float]:
        return {field: float(getattr(self, field)) for field in _DENSE_TIMING_FIELDS}


@dataclass
class _SparsePass2Profile:
    """Sparse pass-2 counters for profiling and log summaries."""

    log_threshold: float = float(np.log(1e-6))
    total_blocks: int = 0
    skipped_blocks: int = 0
    omitted_mass_upper_sum: float = 0.0
    omitted_mass_upper_max: float = 0.0
    omitted_mass_upper_image_count: int = 0

    @property
    def omitted_mass_upper_mean(self) -> float:
        if self.omitted_mass_upper_image_count == 0:
            return 0.0
        return self.omitted_mass_upper_sum / self.omitted_mass_upper_image_count

    def profile_kwargs(self) -> dict[str, float | int]:
        return {
            "sparse_pass2_total_blocks": int(self.total_blocks),
            "sparse_pass2_skipped_blocks": int(self.skipped_blocks),
            "sparse_pass2_omitted_mass_upper_mean": float(self.omitted_mass_upper_mean),
            "sparse_pass2_omitted_mass_upper_max": float(self.omitted_mass_upper_max),
            "sparse_pass2_omitted_mass_upper_sum": float(self.omitted_mass_upper_sum),
        }


@dataclass(frozen=True)
class _DenseDebugOptions:
    """Environment-gated dense debug outputs for one EM call."""

    noise_component_dump_dir: object | None
    noise_component_dump_targets: frozenset[int]
    noise_component_dump_enabled: bool
    per_pose_score_dump: object
    return_noise_split: bool

    @classmethod
    def from_env(cls, current_size: int | None) -> "_DenseDebugOptions":
        dump_dir, dump_targets, dump_current_sizes = parse_dense_noise_component_dump_request()
        dump_enabled = dump_dir is not None and (
            dump_current_sizes is None or int(current_size or -1) in dump_current_sizes
        )
        return cls(
            noise_component_dump_dir=dump_dir,
            noise_component_dump_targets=frozenset(dump_targets),
            noise_component_dump_enabled=bool(dump_enabled),
            per_pose_score_dump=parse_dense_per_pose_score_dump_request(),
            return_noise_split=_noise_split_diagnostics_requested(),
        )


def _pad_dense_big_jit_image_axis(batch_data, ctf_params, target_batch_size: int):
    """Pad dense big-JIT raw batch inputs to a stable image shape class."""

    padded_batch_data, padded_ctf_params, valid_image_mask, actual_batch_size, _ = (
        pad_batch_data_ctf_and_valid_mask(batch_data, ctf_params, target_batch_size)
    )
    return (
        padded_batch_data,
        padded_ctf_params,
        valid_image_mask,
        actual_batch_size,
    )

def run_em(
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance,
    rotations,
    translations,
    disc_type: str,
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    current_size: int = None,
    rotation_log_prior: np.ndarray = None,
    translation_log_prior: np.ndarray = None,
    image_indices: np.ndarray = None,
    rotation_translation_mask: np.ndarray = None,
    *,
    score_with_masked_images: bool = False,
    return_stats: bool = False,
    accumulate_noise: bool = False,
    half_spectrum_scoring: bool = False,
    projection_padding_factor: int = 1,
    reconstruction_padding_factor: int = 1,
    image_corrections: np.ndarray = None,
    scale_corrections: np.ndarray = None,
    image_pre_shifts: np.ndarray = None,
    translation_prior_centers: np.ndarray = None,
    relion_firstiter_score_mode: str = "gaussian",
    relion_firstiter_winner_take_all: bool = False,
    use_float64_scoring: bool = False,
    use_float64_projections: bool = False,
    do_gridding_correction: bool = False,
    square_window: bool = False,
    return_profile: bool = False,
    sparse_pass2: bool = True,
    disable_adjoint_y: bool = False,
    disable_adjoint_ctf: bool = False,
):
    """One EM iteration with JIT-fused two-pass blockwise normalization and half-spectrum GEMMs.

    Key properties:
    - Never materializes full (n_images, n_rot, n_trans) tensor
    - E-step scores computed twice (for normalization stats, then for M-step)
    - All per-block operations are JIT-compiled
    - Projections computed directly in half-spectrum layout via slice_volume(half_image=True)
    - Half-spectrum GEMMs: N_half = H*(W//2+1) instead of N = H*W (~2x speedup)
    - Hermitian weights absorbed into projections (precomputed once per rotation block)
    - Optional Fourier windowing via current_size: restricts GEMMs to low-frequency
      subset of the half-spectrum for further speedup at early iterations.

    Parameters
    ----------
    current_size : int or None
        Diameter in pixels (like RELION's rlnCurrentImageSize).
        When None, use full resolution (same as Phase 1 behavior).
        When set, only frequencies with radius <= current_size // 2 are
        included in the E-step and M-step GEMMs.
    rotation_log_prior : np.ndarray or None
        Log-prior weights added to E-step scores before softmax. Supports
        either a shared vector of shape ``(n_rot,)`` or an image-specific
        matrix of shape ``(n_images, n_rot)`` for exact local-search unions.
        When None (default), a flat prior is used.
    translation_log_prior : np.ndarray or None
        Log-prior weights added to E-step scores before softmax over the
        translation axis. Supports either a shared vector of shape
        ``(n_trans,)`` or an image-specific matrix of shape
        ``(n_images, n_trans)``.
    image_indices : np.ndarray or None
        Optional subset of images to process. When provided, the returned
        hard assignments and per-image stats are ordered according to this
        subset rather than the full dataset.
    rotation_translation_mask : np.ndarray or None, shape (n_rot, n_trans)
        Optional boolean validity mask over the Cartesian rotation x
        translation grid. Entries set to False are excluded by forcing
        their scores to ``-inf`` in both E-step passes.
    score_with_masked_images : bool
        When True, compute E-step scores from masked images but keep the
        M-step reconstruction on unmasked images.
    return_stats : bool
        When True, also return a :class:`RelionStats` container with the
        per-image log normalizer, best score, maximum posterior
        probability, and additive posterior mass per rotation computed
        during the E-step.
    image_corrections : np.ndarray or None, shape (n_images,)
        Per-image multiplicative correction applied to Fourier images
        in cross terms. For RELION parity this is
        ``(avg_norm / normcorr[i]) * scale[group_id[i]]`` so the cross term
        matches RELION's norm-corrected image and scale-corrected reference.
        Image-only terms divide out ``scale_corrections`` and use only
        ``avg_norm / normcorr[i]`` because RELION applies group scale to the
        reference/CTF, not to image power.
    scale_corrections : np.ndarray or None, shape (n_images,)
        Per-image scale correction (``rlnGroupScaleCorrection``).
        RELION applies scale to the *reference* (``Frefctf *= myscale``
        at ml_optimiser.cpp:7298 and ``Mctf *= myscale`` at
        ml_optimiser.cpp:8516).  This means the E-step norm-term and
        the M-step CTF denominator must both carry a ``scale**2``
        factor.  When provided, ``ctf2_over_nv`` is multiplied by
        ``scale**2`` per image to match RELION's convention.
    image_pre_shifts : np.ndarray or None, shape (n_images, 2)
        Per-image old-offset pre-shift in pixels.  For integral shifts
        (RELION's rounded ``old_offset``), RECOVAR applies RELION's
        zero-filled real-space integer translation before FFT.  Non-integral
        shifts use half-spectrum Fourier phases.  The candidate translations
        from the grid are then relative to this centered position.
    return_profile : bool
        When True, append an :class:`EMProfileStats` timing summary to the
        return tuple.  This is diagnostic only.
    sparse_pass2 : bool
        When True, use pass-1 block maxima to skip pass-2 rotation blocks
        whose posterior mass is negligible for every image in the batch.
    disable_adjoint_y : bool
        Experimental ablation flag. When True, skip the weighted-image
        adjoint accumulation into ``Ft_y``.
    disable_adjoint_ctf : bool
        Experimental ablation flag. When True, skip the CTF adjoint
        accumulation into ``Ft_ctf``.
    """
    overall_t0 = time.time()
    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    image_indices = np.arange(experiment_dataset.n_units) if image_indices is None else np.asarray(image_indices)
    n_images = image_indices.size
    if relion_firstiter_score_mode not in {"gaussian", "normalized_cc"}:
        raise ValueError(
            f"relion_firstiter_score_mode must be 'gaussian' or 'normalized_cc', got {relion_firstiter_score_mode!r}",
        )
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    debug_options = _DenseDebugOptions.from_env(current_size)
    # Pad volume in real space for smoother trilinear projection.
    if projection_padding_factor > 1:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            mean,
            volume_shape,
            projection_padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=current_size,
        )
    else:
        mean_for_proj = mean
        proj_volume_shape = volume_shape

    precision_policy = DensePrecisionPolicy(
        use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
    )
    # NOTE: float64 projections tested (Slurm 7174969) — identical results to float32.
    # The 0.0074 Pmax gap is NOT float precision; it's boundary handling in trilinear interp.
    mean_for_proj = precision_policy.cast_projection_volume(mean_for_proj)

    # Backprojection padding: accumulate into a (pf*N)³ grid for finer
    # trilinear interpolation, matching RELION's --pad flag.
    if reconstruction_padding_factor > 1:
        recon_volume_shape = tuple(d * reconstruction_padding_factor for d in volume_shape)
        recon_volume_size = int(np.prod(recon_volume_shape))
    else:
        recon_volume_shape = volume_shape
        recon_volume_size = int(np.prod(volume_shape))

    H, W = image_shape
    n_half = H * (W // 2 + 1)
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )

    # TODO(local-engine-debt): If we keep any dense score path after the local
    # engine split, there is still an inner-product/GEMM-shaped optimization
    # opportunity around the translation dimension. RELION appears to fuse
    # project+translate+score in custom kernels instead of BLAS here, so this
    # is not a parity requirement. Still, we should remember to revisit that
    # opportunity once the local path stops forcing per-image neighborhoods
    # through the shared-grid dense engine.
    half_weights = make_scoring_half_image_weights(
        image_shape,
        relion_half_sum=half_spectrum_scoring,
    )

    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        square=square_window,
        include_recon_window=True,
    )
    use_window = window_spec.use_window
    window_indices = window_spec.score_indices
    recon_window_indices = window_spec.recon_indices
    n_windowed = window_spec.n_score
    n_recon_windowed = window_spec.n_recon
    projection_kwargs = window_spec.projection_kwargs()
    if use_window:
        window_desc = "square" if square_window else "circular"
        logger.info(
            "Fourier windowing (%s): current_size=%d, n_score_windowed=%d, n_recon_windowed=%d / n_half=%d (%.1f%% reduction)",
            window_desc,
            current_size,
            n_windowed,
            n_recon_windowed,
            n_half,
            100.0 * (1.0 - n_windowed / n_half),
        )

    half_weights = half_weights.astype(precision_policy.score_real_dtype)

    # Pad rotations to multiple of block size for fixed shapes
    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate([rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))], axis=0)
    else:
        rotations_padded = rotations

    timing = _DenseTiming()
    sync_timers = bool(return_profile)
    sparse_profile = _SparsePass2Profile()

    # Prepare per-rotation log-prior (pad to match rotations_padded)
    per_image_log_prior = False
    if rotation_log_prior is not None:
        rotation_log_prior = np.asarray(rotation_log_prior, dtype=np.float32)
        if rotation_log_prior.ndim == 1:
            log_prior_padded = np.full(n_rot_padded, -1e30, dtype=np.float32)
            log_prior_padded[:n_rot] = rotation_log_prior
        elif rotation_log_prior.ndim == 2:
            if rotation_log_prior.shape != (n_images, n_rot):
                raise ValueError(
                    "rotation_log_prior must have shape "
                    f"({n_images}, {n_rot}) when image-specific, got "
                    f"{rotation_log_prior.shape}",
                )
            log_prior_padded = np.full((n_images, n_rot_padded), -1e30, dtype=np.float32)
            log_prior_padded[:, :n_rot] = rotation_log_prior
            per_image_log_prior = True
        else:
            raise ValueError(
                f"rotation_log_prior must be 1D or 2D, got {rotation_log_prior.ndim} dimensions",
            )
        log_prior_padded_jnp = jnp.asarray(log_prior_padded)
        finite_prior = rotation_log_prior[np.isfinite(rotation_log_prior)]
        if finite_prior.size == 0:
            finite_prior = np.array([-1e30], dtype=np.float32)
        logger.info(
            "Using rotation log-prior: %d rotations%s, range [%.2f, %.2f]",
            n_rot,
            " (per-image)" if per_image_log_prior else "",
            float(finite_prior.min()),
            float(finite_prior.max()),
        )
    else:
        log_prior_padded_jnp = None

    per_image_translation_log_prior = False
    if translation_log_prior is not None:
        translation_log_prior = np.asarray(translation_log_prior, dtype=np.float32)
        if translation_log_prior.ndim == 1:
            if translation_log_prior.shape != (n_trans,):
                raise ValueError(
                    f"translation_log_prior must have shape ({n_trans},), got {translation_log_prior.shape}",
                )
            translation_log_prior_jnp = jnp.asarray(translation_log_prior)
        elif translation_log_prior.ndim == 2:
            if translation_log_prior.shape != (n_images, n_trans):
                raise ValueError(
                    "translation_log_prior must have shape "
                    f"({n_images}, {n_trans}) when image-specific, got "
                    f"{translation_log_prior.shape}",
                )
            translation_log_prior_jnp = jnp.asarray(translation_log_prior)
            per_image_translation_log_prior = True
        else:
            raise ValueError(
                f"translation_log_prior must be 1D or 2D, got {translation_log_prior.ndim} dimensions",
            )
        finite_translation_prior = translation_log_prior[np.isfinite(translation_log_prior)]
        if finite_translation_prior.size == 0:
            finite_translation_prior = np.array([-1e30], dtype=np.float32)
        logger.info(
            "Using translation log-prior: %d translations%s, range [%.2f, %.2f]",
            n_trans,
            " (per-image)" if per_image_translation_log_prior else "",
            float(finite_translation_prior.min()),
            float(finite_translation_prior.max()),
        )
    else:
        translation_log_prior_jnp = None

    translation_prior_centers_np = validate_translation_prior_centers(
        translation_prior_centers,
        n_images=n_images,
        n_dims=translations.shape[1],
    )

    candidate_mask_padded_jnp = None
    if rotation_translation_mask is not None:
        candidate_mask = np.asarray(rotation_translation_mask, dtype=bool)
        if candidate_mask.shape != (n_rot, n_trans):
            raise ValueError(
                f"rotation_translation_mask must have shape ({n_rot}, {n_trans}), got {candidate_mask.shape}",
            )
        candidate_mask_padded = np.zeros((n_rot_padded, n_trans), dtype=bool)
        candidate_mask_padded[:n_rot] = candidate_mask
        candidate_mask_padded_jnp = jnp.asarray(candidate_mask_padded)
        logger.info(
            "Using rotation-translation mask: %d / %d candidates valid",
            int(candidate_mask.sum()),
            int(candidate_mask.size),
        )

    dense_big_jit_requested = _dense_big_jit_enabled()
    dense_big_jit_unsupported_reason = _dense_big_jit_disabled_reason(
        relion_firstiter_winner_take_all=relion_firstiter_winner_take_all,
        accumulate_noise=accumulate_noise,
        dense_noise_component_dump_enabled=debug_options.noise_component_dump_enabled,
        per_pose_debug_dump_enabled=debug_options.per_pose_score_dump.enabled,
    )
    use_dense_big_jit = dense_big_jit_requested and dense_big_jit_unsupported_reason is None
    if dense_big_jit_requested and not use_dense_big_jit:
        logger.info("Dense big-JIT disabled for this run: unsupported %s", dense_big_jit_unsupported_reason)
    elif use_dense_big_jit:
        logger.info("Dense big-JIT enabled for dense/global rotation buckets")

    def _dense_big_jit_priors_and_masks(r0: int, r1: int, start: int, end: int, batch_count: int):
        actual_count = int(end - start)
        batch_count = int(batch_count)
        if log_prior_padded_jnp is None:
            rotation_prior_block = jnp.zeros((batch_count, rotation_block_size), dtype=jnp.float32)
        elif per_image_log_prior:
            rotation_prior_block = log_prior_padded_jnp[start:end, r0:r1]
            if batch_count != actual_count:
                rotation_prior_block = jnp.pad(
                    rotation_prior_block,
                    ((0, batch_count - actual_count), (0, 0)),
                    constant_values=0,
                )
        else:
            rotation_prior_block = jnp.broadcast_to(
                log_prior_padded_jnp[r0:r1][None, :],
                (batch_count, rotation_block_size),
            )

        if translation_log_prior_jnp is None:
            translation_prior_block = jnp.zeros((batch_count, n_trans), dtype=jnp.float32)
        elif per_image_translation_log_prior:
            translation_prior_block = translation_log_prior_jnp[start:end]
            if batch_count != actual_count:
                translation_prior_block = jnp.pad(
                    translation_prior_block,
                    ((0, batch_count - actual_count), (0, 0)),
                    constant_values=0,
                )
        else:
            translation_prior_block = jnp.broadcast_to(
                translation_log_prior_jnp[None, :],
                (batch_count, n_trans),
            )

        if candidate_mask_padded_jnp is None:
            candidate_mask_block = jnp.ones((rotation_block_size, n_trans), dtype=bool)
        else:
            candidate_mask_block = candidate_mask_padded_jnp[r0:r1]

        valid = max(0, min(rotation_block_size, n_rot - r0))
        valid_rotation_mask = jnp.arange(rotation_block_size) < valid
        return rotation_prior_block, translation_prior_block, candidate_mask_block, valid_rotation_mask

    # Initialize accumulators (at padded resolution for pf>1 backprojection)
    Ft_y = jnp.zeros(recon_volume_size, dtype=experiment_dataset.dtype)
    Ft_ctf = jnp.zeros(recon_volume_size, dtype=experiment_dataset.dtype)
    hard_assignment = np.empty(n_images, dtype=np.int32)
    log_evidence_per_image = None
    best_log_score_per_image = None
    max_posterior_per_image = None
    rotation_posterior_sums = None
    if return_stats:
        log_evidence_per_image = np.empty(n_images, dtype=np.float32)
        best_log_score_per_image = np.empty(n_images, dtype=np.float32)
        max_posterior_per_image = np.empty(n_images, dtype=np.float32)
        rotation_posterior_sums = np.zeros(n_rot, dtype=np.float64)

    # Noise accumulation precomputation (RELION parity)
    noise_wsum = None
    noise_img_power = None
    noise_sumw = 0.0
    noise_a2 = None  # diagnostic
    noise_xa = None  # diagnostic
    noise_sigma2_offset = 0.0
    if accumulate_noise:
        n_shells = image_shape[0] // 2 + 1
        shell_indices_half = make_relion_noise_shell_indices_half(image_shape)
        shell_indices_noise = window_spec.recon_values(shell_indices_half)
        noise_variance_windowed = window_spec.recon_values(noise_variance_half)
        noise_wsum = np.zeros(n_shells, dtype=np.float64)
        noise_img_power = np.zeros(n_shells, dtype=np.float64)
        if debug_options.return_noise_split:
            noise_a2 = np.zeros(n_shells, dtype=np.float64)
            noise_xa = np.zeros(n_shells, dtype=np.float64)
    dense_big_jit_shell_indices_half = (
        shell_indices_half if accumulate_noise else make_shell_indices_half(image_shape)
    )
    dense_big_jit_noise_variance_half = (
        noise_variance_half if accumulate_noise else jnp.ones(n_half, dtype=jnp.float32)
    )
    score_dc_index = half_spectrum_dc_index(image_shape)

    start_idx = 0

    batch_iter = experiment_dataset.iter_batches(
        image_batch_size,
        indices=image_indices,
        by_image=False,
    )
    while True:
        batch_fetch_t0 = time.time()
        try:
            batch_data, _, _, ctf_params, _, _, indices = next(batch_iter)
        except StopIteration:
            timing.batch_fetch_s += time.time() - batch_fetch_t0
            break
        timing.batch_fetch_s += time.time() - batch_fetch_t0
        actual_batch_size = len(indices)
        batch_indices_np = np.asarray(indices)
        end_idx = start_idx + actual_batch_size
        integer_pre_shifts = integer_pre_shifts_or_none(image_pre_shifts, indices, batch=batch_data)
        real_space_pre_shift_applied = integer_pre_shifts is not None
        if real_space_pre_shift_applied:
            batch_data = apply_relion_integer_pre_shifts(batch_data, integer_pre_shifts)
        if use_dense_big_jit:
            (
                batch_data,
                ctf_params,
                valid_image_mask_np,
                actual_batch_size,
            ) = _pad_dense_big_jit_image_axis(batch_data, ctf_params, image_batch_size)
        else:
            valid_image_mask_np = np.ones(actual_batch_size, dtype=bool)
        batch_size = int(np.asarray(batch_data).shape[0])
        valid_image_mask = jnp.asarray(valid_image_mask_np, dtype=bool)
        batch_data = jnp.asarray(batch_data)
        translation_sqdist_ang = None
        if translation_prior_centers_np is not None:
            centers = translation_prior_centers_for_images(
                translation_prior_centers_np,
                np.arange(start_idx, end_idx, dtype=np.int64),
                batch_size=actual_batch_size,
            )
            translation_sqdist_ang = translation_sqdist_angstrom(
                translations,
                centers,
                experiment_dataset.voxel_size,
            )
            if use_dense_big_jit and batch_size != actual_batch_size:
                translation_sqdist_ang = pad_axis(translation_sqdist_ang, 0, batch_size, value=0)
        projection_cache = None if use_dense_big_jit else []
        block_max_per_image = [] if sparse_pass2 else None
        block_pose_counts = [] if sparse_pass2 else None

        # -- PREPROCESS (once per image batch) -- returns half-spectrum --
        preprocess_t0 = time.time()
        if relion_firstiter_score_mode == "normalized_cc":
            shifted_half, batch_norm, ctf2_half_score, ctf2_over_nv_half = _preprocess_batch_firstiter_cc(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
                score_with_masked_images,
            )
        else:
            shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
                score_with_masked_images,
            )
            ctf2_half_score = None
        shifted_recon_half = (
            _prepare_reconstruction_batch(
                experiment_dataset,
                batch_data,
                ctf_params,
                noise_variance_half,
                translations,
                config,
            )
            if (score_with_masked_images or relion_firstiter_score_mode == "normalized_cc")
            else shifted_half
        )
        if sync_timers:
            _block_until_ready(shifted_half, shifted_recon_half)

        timing.preprocess_s += time.time() - preprocess_t0

        score_prep_t0 = time.time()
        if scale_corrections is not None:
            batch_scale_np = np.asarray(scale_corrections, dtype=np.float32)[batch_indices_np]
            if use_dense_big_jit and batch_size != actual_batch_size:
                batch_scale_np = pad_axis(batch_scale_np, 0, batch_size, value=1)
            batch_scale = jnp.asarray(batch_scale_np)
        else:
            batch_scale = jnp.ones(batch_size, dtype=batch_norm.dtype)

        # -- Per-image corrections (RELION parity: avg_norm/normcorr * scale) --
        # RELION: img *= avg_norm_correction / normcorr  (ml_optimiser.cpp:6240)
        # then   Frefctf *= scale                        (ml_optimiser.cpp:7298)
        # The cross-term multiplier is (avg_norm/normcorr)*scale. Image-only
        # terms use avg_norm/normcorr, so divide the scale back out below.
        # shifted_half has shape (batch_size * n_trans, N_half) — broadcast
        # the per-image correction across n_trans copies.
        if image_corrections is not None:
            batch_corr_np = np.asarray(image_corrections, dtype=np.float32)[batch_indices_np]
            if use_dense_big_jit and batch_size != actual_batch_size:
                batch_corr_np = pad_axis(batch_corr_np, 0, batch_size, value=1)
            batch_corr = jnp.asarray(batch_corr_np)
            image_only_corr = batch_corr / batch_scale
            if relion_firstiter_score_mode == "normalized_cc":
                score_batch_corr = batch_corr / (batch_scale**2)
                norm_batch_corr = image_only_corr
            else:
                score_batch_corr = batch_corr
                norm_batch_corr = image_only_corr
            score_corr_expanded = jnp.repeat(score_batch_corr, n_trans)
            recon_corr_expanded = jnp.repeat(batch_corr, n_trans)
            shifted_half = shifted_half * score_corr_expanded[:, None]
            shifted_recon_half = shifted_recon_half * recon_corr_expanded[:, None]
            batch_norm = batch_norm * (norm_batch_corr**2)[:, None]
        else:
            batch_corr = None
            image_only_corr = None

        # -- Per-image scale correction on CTF²/σ² (RELION parity) --
        # RELION applies scale_correction to the REFERENCE: Frefctf *= myscale
        # (ml_optimiser.cpp:7298) and Mctf *= myscale (ml_optimiser.cpp:8516).
        # This means both the E-step norm-term (ctf²/σ² @ |proj|²) and the
        # M-step denominator (Σ γ·ctf²/σ²) carry scale².  Apply it here so
        # all downstream uses of ctf2_over_nv_half see the correct factor.
        if scale_corrections is not None:
            ctf2_over_nv_half = ctf2_over_nv_half * (batch_scale**2)[:, None]
            if ctf2_half_score is not None:
                ctf2_half_score = ctf2_half_score * (batch_scale**2)[:, None]

        # -- Per-image pre-centering --
        # Integral RELION old-offsets were already applied to the real-space
        # image with zero fill before FFT. Keep the Fourier-phase path for
        # non-integral pre-shifts.
        if image_pre_shifts is not None and not real_space_pre_shift_applied:
            batch_shifts_np = np.asarray(image_pre_shifts, dtype=np.float32)[batch_indices_np]
            if use_dense_big_jit and batch_size != actual_batch_size:
                batch_shifts_np = pad_axis(batch_shifts_np, 0, batch_size, value=0)
            batch_shifts = jnp.asarray(batch_shifts_np)
            # Compute per-pixel phase factors in half-spectrum layout
            lattice_half = fourier_transform_utils.get_k_coordinate_of_each_pixel_half(
                image_shape, voxel_size=1, scaled=True
            )
            # phase_factors: (batch_size, N_half) complex
            phase_factors = jnp.exp(-2j * jnp.pi * (lattice_half @ batch_shifts.T)).T
            # Expand to (batch_size * n_trans, N_half)
            phase_expanded = jnp.repeat(phase_factors, n_trans, axis=0)
            shifted_half = shifted_half * phase_expanded
            shifted_recon_half = shifted_recon_half * phase_expanded

        if relion_firstiter_score_mode == "normalized_cc":
            score_weight_half = ctf2_half_score / jnp.maximum(batch_norm, jnp.asarray(1e-30, dtype=batch_norm.dtype))
            shifted_score_half = shifted_half * jnp.repeat(score_weight_half, n_trans, axis=0)
        else:
            score_weight_half = ctf2_over_nv_half
            shifted_score_half = shifted_half

        # -- Save pre-DC-exclusion arrays for M-step + noise accumulation --
        # RELION excludes DC from likelihood scores (Minvsigma2[0]=0) but
        # INCLUDES DC in reconstruction weights (backprojector CTF^2) and
        # noise estimation.  Save original arrays before DC zeroing.
        shifted_half_with_dc = shifted_score_half
        ctf2_over_nv_half_with_dc = ctf2_over_nv_half

        # -- DC exclusion (RELION parity: Minvsigma2[0] = 0) --
        # RELION excludes the DC pixel from likelihood scores.
        # In recovar's half-spectrum layout, DC is NOT generally at flat index 0.
        if half_spectrum_scoring and relion_firstiter_score_mode != "normalized_cc":
            shifted_score_half = shifted_score_half.at[:, score_dc_index].set(0.0)
            score_weight_half = score_weight_half.at[:, score_dc_index].set(0.0)

        # DC-zeroed arrays for scoring; with-DC arrays for M-step/noise.
        shifted_windowed = window_spec.score_values(shifted_score_half)
        shifted_recon_windowed = window_spec.recon_values(shifted_recon_half)
        ctf2_over_nv_windowed = window_spec.score_values(score_weight_half)
        ctf2_over_nv_windowed_mstep = window_spec.recon_values(ctf2_over_nv_half_with_dc)

        # -- Noise: precompute per-batch image power spectrum --
        if accumulate_noise:
            # P_img = sum_i |masked_img_i(k)|^2 per half-spectrum pixel
            # Use the masked processed images (score path).
            processed_masked_half = process_half_image(
                experiment_dataset,
                batch_data,
                score_with_masked_images,
            )
            if image_only_corr is not None:
                processed_masked_half = processed_masked_half * image_only_corr[:, None]
            # Sum |img|^2 over images in this batch, bin to shells (FULL spectrum, not windowed)
            batch_img_power = jnp.sum(jnp.abs(processed_masked_half) ** 2, axis=0)  # (N_half,)
            batch_img_power_shells = jnp.zeros(n_shells, dtype=jnp.float32)
            batch_img_power_shells = batch_img_power_shells.at[shell_indices_half].add(batch_img_power)
            noise_img_power += np.asarray(batch_img_power_shells, dtype=np.float64)
            noise_sumw += batch_size
            # Masked shifted images for the noise GEMM: use WITH-DC versions
            # (RELION includes DC in noise but excludes from scoring)
            shifted_masked_for_noise = window_spec.recon_values(shifted_half_with_dc)
            dense_noise_component_acc = {}
            if debug_options.noise_component_dump_enabled:
                indices_np = np.asarray(indices, dtype=np.int64)
                original_indices_np = np.asarray(
                    experiment_dataset.original_image_indices_from_local(indices_np),
                    dtype=np.int64,
                )
                target_rows = [
                    (row, int(local_idx), int(global_idx))
                    for row, (local_idx, global_idx) in enumerate(zip(indices_np.tolist(), original_indices_np.tolist()))
                    if int(global_idx) in debug_options.noise_component_dump_targets
                ]
                for row, local_idx, global_idx in target_rows:
                    p_img_pixel = np.asarray(jnp.abs(processed_masked_half[row]) ** 2, dtype=np.float64)
                    dense_noise_component_acc[global_idx] = {
                        "row": int(row),
                        "local_idx": int(local_idx),
                        "p_img_shells": _bin_shell_values_np(p_img_pixel, shell_indices_half, n_shells),
                        "a2_shells": np.zeros(n_shells, dtype=np.float64),
                        "xa_shells": np.zeros(n_shells, dtype=np.float64),
                    }
        else:
            dense_noise_component_acc = {}

        shifted_score_half, score_weight_half, shifted_recon_half = precision_policy.cast_scoring_inputs(
            shifted_score_half,
            score_weight_half,
            shifted_recon_half,
        )
        shifted_windowed = window_spec.score_values(shifted_score_half)
        ctf2_over_nv_windowed = window_spec.score_values(score_weight_half)
        shifted_recon_windowed = window_spec.recon_values(shifted_recon_half)

        if sync_timers:
            ready_values = [
                shifted_score_half,
                shifted_recon_half,
                batch_norm,
                score_weight_half,
                shifted_windowed,
                shifted_recon_windowed,
                ctf2_over_nv_windowed,
            ]
            if accumulate_noise:
                ready_values.append(shifted_masked_for_noise)
            _block_until_ready(*ready_values)
        timing.score_prep_s += time.time() - score_prep_t0

        dense_big_jit_window_indices = window_spec.score_or_full_indices(n_half)
        dense_big_jit_recon_window_indices = window_spec.recon_or_full_indices(n_half)
        dense_big_jit_max_r = window_spec.dense_big_jit_max_r()
        dense_big_jit_noise_wsum0 = jnp.zeros(1, dtype=jnp.float32)
        dense_big_jit_noise_a20 = jnp.zeros(1, dtype=jnp.float32)
        dense_big_jit_noise_xa0 = jnp.zeros(1, dtype=jnp.float32)
        dense_big_jit_offset0 = jnp.asarray(0.0, dtype=jnp.float32)
        dense_big_jit_translation_sqdist0 = jnp.zeros((batch_size, n_trans), dtype=jnp.float32)

        def _run_dense_big_jit_bucket(r0: int, r1: int, *, run_mstep: bool, log_z):
            (
                rotation_prior_block,
                translation_prior_block,
                candidate_mask_block,
                valid_rotation_mask,
            ) = _dense_big_jit_priors_and_masks(r0, r1, start_idx, end_idx, batch_size)
            return run_dense_bucket_big_jit(
                shifted_score_half,
                batch_norm,
                score_weight_half,
                shifted_recon_half,
                ctf2_over_nv_half_with_dc,
                mean_for_proj,
                Ft_y,
                Ft_ctf,
                jnp.asarray(rotations_padded[r0:r1]),
                half_weights,
                rotation_prior_block,
                translation_prior_block,
                candidate_mask_block,
                valid_rotation_mask,
                valid_image_mask,
                log_z,
                dense_big_jit_noise_wsum0,
                dense_big_jit_noise_a20,
                dense_big_jit_noise_xa0,
                dense_big_jit_offset0,
                shifted_half_with_dc,
                dense_big_jit_noise_variance_half,
                dense_big_jit_shell_indices_half,
                dense_big_jit_translation_sqdist0,
                dense_big_jit_window_indices,
                dense_big_jit_recon_window_indices,
                score_mode=relion_firstiter_score_mode,
                zero_dc_for_scoring=half_spectrum_scoring,
                use_window=use_window,
                use_float64_scoring=use_float64_scoring,
                use_float64_normalization=True,
                run_mstep=run_mstep,
                accumulate_noise=False,
                return_noise_split=False,
                has_translation_sqdist=False,
                image_shape=image_shape,
                proj_volume_shape=proj_volume_shape,
                recon_volume_shape=recon_volume_shape,
                disc_type=disc_type,
                projection_half_volume=False,
                projection_max_r=dense_big_jit_max_r,
                mstep_half_volume=False,
                backprojection_max_r=dense_big_jit_max_r,
                disable_adjoint_y=disable_adjoint_y,
                disable_adjoint_ctf=disable_adjoint_ctf,
                n_shells=1,
            )

        # -- PASS 1: streaming logsumexp over rotation blocks --
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size, dtype=precision_policy.normalization_real_dtype)
        best_score_pass1 = jnp.full(batch_size, -jnp.inf)
        best_argmax_pass1 = jnp.zeros(batch_size, dtype=jnp.int32)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            if use_dense_big_jit:
                score_t0 = time.time()
                dense_result = _run_dense_big_jit_bucket(
                    r0,
                    r1,
                    run_mstep=False,
                    log_z=jnp.zeros(batch_size, dtype=jnp.float32),
                )
                max_s, sum_exp = _merge_block_logsumexp(
                    max_s,
                    sum_exp,
                    dense_result.block_max,
                    dense_result.block_sum_exp,
                )
                if block_max_per_image is not None:
                    actual_rot = max(0, min(rotation_block_size, n_rot - r0))
                    block_max_per_image.append(dense_result.block_best)
                    block_pose_counts.append(actual_rot * n_trans)
                if sync_timers:
                    _block_until_ready(max_s, sum_exp)
                timing.pass1_score_s += time.time() - score_t0
                continue

            proj_t0 = time.time()
            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean_for_proj,
                rots_b,
                image_shape,
                proj_volume_shape,
                disc_type,
                **projection_kwargs,
            )
            if sync_timers:
                _block_until_ready(proj_half_b, proj_abs2_half_b)
            timing.pass1_projection_s += time.time() - proj_t0
            if projection_cache is not None:
                projection_cache.append((proj_half_b, proj_abs2_half_b))

            score_t0 = time.time()
            scores = _score_rotation_block(
                window_spec,
                shifted_score=shifted_windowed,
                batch_norm=batch_norm,
                score_weight=ctf2_over_nv_windowed,
                proj_half=proj_half_b,
                proj_abs2_half=proj_abs2_half_b,
                half_weights=half_weights,
                n_images=batch_size,
                n_trans=n_trans,
                image_shape=image_shape,
                volume_shape=volume_shape,
                score_mode=relion_firstiter_score_mode,
                precision_policy=precision_policy,
            )

            if sync_timers:
                _block_until_ready(scores)
            timing.pass1_score_s += time.time() - score_t0

            maybe_write_dense_per_pose_score_dump(
                request=debug_options.per_pose_score_dump,
                indices=indices,
                scores=scores,
                block_index=b,
                preprior=True,
            )

            pass1_postprocess_t0 = time.time()
            if relion_firstiter_score_mode == "gaussian":
                if log_prior_padded_jnp is not None:
                    if per_image_log_prior:
                        log_prior_block = log_prior_padded_jnp[start_idx:end_idx, r0:r1]
                        scores = scores + log_prior_block[:, :, None]
                    else:
                        log_prior_block = log_prior_padded_jnp[r0:r1]
                        scores = scores + log_prior_block[None, :, None]

                if translation_log_prior_jnp is not None:
                    if per_image_translation_log_prior:
                        translation_prior_block = translation_log_prior_jnp[start_idx:end_idx]
                        scores = scores + translation_prior_block[:, None, :]
                    else:
                        scores = scores + translation_log_prior_jnp[None, None, :]

            if candidate_mask_padded_jnp is not None:
                candidate_mask_block = candidate_mask_padded_jnp[r0:r1]
                scores = jnp.where(candidate_mask_block[None, :, :], scores, -jnp.inf)

            # Mask padding rotations (set their scores to -inf)
            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)

            if block_max_per_image is not None:
                actual_rot = max(0, min(rotation_block_size, n_rot - r0))
                block_max_per_image.append(jnp.max(scores, axis=(1, 2)))
                block_pose_counts.append(actual_rot * n_trans)

            maybe_write_dense_per_pose_score_dump(
                request=debug_options.per_pose_score_dump,
                indices=indices,
                scores=scores,
                block_index=b,
            )

            if sync_timers:
                _block_until_ready(scores)
            timing.pass1_postprocess_s += time.time() - pass1_postprocess_t0

            if relion_firstiter_winner_take_all:
                block_best = jnp.max(scores.reshape(batch_size, -1), axis=1)
                block_argmax = jnp.argmax(scores.reshape(batch_size, -1), axis=1)
                improved = block_best > best_score_pass1
                best_score_pass1 = jnp.where(improved, block_best, best_score_pass1)
                best_argmax_pass1 = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax_pass1)

            logsumexp_t0 = time.time()
            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)
            if sync_timers:
                _block_until_ready(max_s, sum_exp)
            timing.pass1_logsumexp_s += time.time() - logsumexp_t0

        log_Z = max_s + jnp.log(sum_exp)  # (batch_size,)
        skip_pass2_block = np.zeros(n_blocks, dtype=bool)
        pass2_skipmask_t0 = time.time()
        if block_max_per_image:
            block_max_matrix = jnp.stack(block_max_per_image, axis=0)
            block_log_pose_counts = jnp.log(
                jnp.asarray(block_pose_counts, dtype=precision_policy.normalization_real_dtype),
            )[:, None]
            finite_log_z = jnp.isfinite(log_Z) & valid_image_mask
            log_omitted_mass_upper = jnp.where(
                finite_log_z[None, :],
                block_log_pose_counts
                + block_max_matrix.astype(precision_policy.normalization_real_dtype)
                - log_Z[None, :].astype(precision_policy.normalization_real_dtype),
                jnp.inf,
            )
            skip_candidate = (log_omitted_mass_upper < sparse_profile.log_threshold) | (~valid_image_mask[None, :])
            skip_pass2_block = np.asarray(
                jnp.all(skip_candidate, axis=1),
                dtype=bool,
            )
            sparse_profile.total_blocks += int(n_blocks)
            sparse_profile.skipped_blocks += int(skip_pass2_block.sum())
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
                sparse_profile.omitted_mass_upper_sum += float(np.sum(skipped_mass_upper_np))
                sparse_profile.omitted_mass_upper_max = max(
                    sparse_profile.omitted_mass_upper_max,
                    float(np.max(skipped_mass_upper_np)),
                )
                sparse_profile.omitted_mass_upper_image_count += int(actual_batch_size)
            if sync_timers:
                _block_until_ready(block_max_matrix, log_omitted_mass_upper)
        timing.pass2_skipmask_s += time.time() - pass2_skipmask_t0

        # -- PASS 2: recompute scores, normalize, accumulate M-step --
        if relion_firstiter_winner_take_all:
            best_score = best_score_pass1
            best_argmax = best_argmax_pass1
        else:
            best_score = jnp.full(batch_size, -jnp.inf)
            best_argmax = jnp.zeros(batch_size, dtype=jnp.int32)

        for b in range(n_blocks):
            if skip_pass2_block[b]:
                continue

            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            if use_dense_big_jit:
                score_t0 = time.time()
                dense_result = _run_dense_big_jit_bucket(
                    r0,
                    r1,
                    run_mstep=True,
                    log_z=log_Z,
                )
                Ft_y = dense_result.Ft_y
                Ft_ctf = dense_result.Ft_ctf
                if sync_timers:
                    _block_until_ready(
                        Ft_y,
                        Ft_ctf,
                        dense_result.block_best,
                        dense_result.block_argmax,
                        dense_result.probs_sum_t,
                    )
                timing.pass2_score_s += time.time() - score_t0

                assignment_t0 = time.time()
                improved = dense_result.block_best > best_score
                best_score = jnp.where(improved, dense_result.block_best, best_score)
                best_argmax = jnp.where(
                    improved,
                    dense_result.block_argmax + r0 * n_trans,
                    best_argmax,
                )
                if sync_timers:
                    _block_until_ready(best_score, best_argmax)
                timing.assignment_s += time.time() - assignment_t0

                if return_stats:
                    host_stats_t0 = time.time()
                    actual_rot = max(0, min(rotation_block_size, n_rot - r0))
                    if actual_rot > 0:
                        block_rotation_sums = np.asarray(
                            jnp.sum(dense_result.probs_sum_t[:, :actual_rot], axis=0),
                            dtype=np.float64,
                        )
                        rotation_posterior_sums[r0 : r0 + actual_rot] += block_rotation_sums
                    timing.host_stats_s += time.time() - host_stats_t0
                continue

            if projection_cache is not None:
                proj_half_b, proj_abs2_half_b = projection_cache[b]
            else:
                proj_t0 = time.time()
                proj_half_b, proj_abs2_half_b = _compute_projections_block(
                    mean_for_proj,
                    rots_b,
                    image_shape,
                    proj_volume_shape,
                    disc_type,
                    **projection_kwargs,
                )
                timing.pass2_projection_s += time.time() - proj_t0

            score_t0 = time.time()
            scores = _score_rotation_block(
                window_spec,
                shifted_score=shifted_windowed,
                batch_norm=batch_norm,
                score_weight=ctf2_over_nv_windowed,
                proj_half=proj_half_b,
                proj_abs2_half=proj_abs2_half_b,
                half_weights=half_weights,
                n_images=batch_size,
                n_trans=n_trans,
                image_shape=image_shape,
                volume_shape=volume_shape,
                score_mode=relion_firstiter_score_mode,
                precision_policy=precision_policy,
            )

            if sync_timers:
                _block_until_ready(scores)
            timing.pass2_score_s += time.time() - score_t0

            pass2_postprocess_t0 = time.time()
            if relion_firstiter_score_mode == "gaussian":
                if log_prior_padded_jnp is not None:
                    if per_image_log_prior:
                        log_prior_block = log_prior_padded_jnp[start_idx:end_idx, r0:r1]
                        scores = scores + log_prior_block[:, :, None]
                    else:
                        log_prior_block = log_prior_padded_jnp[r0:r1]
                        scores = scores + log_prior_block[None, :, None]

                if translation_log_prior_jnp is not None:
                    if per_image_translation_log_prior:
                        translation_prior_block = translation_log_prior_jnp[start_idx:end_idx]
                        scores = scores + translation_prior_block[:, None, :]
                    else:
                        scores = scores + translation_log_prior_jnp[None, None, :]

            if candidate_mask_padded_jnp is not None:
                candidate_mask_block = candidate_mask_padded_jnp[r0:r1]
                scores = jnp.where(candidate_mask_block[None, :, :], scores, -jnp.inf)

            # Mask padding
            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)
            if sync_timers:
                _block_until_ready(scores)
            timing.pass2_postprocess_s += time.time() - pass2_postprocess_t0

            if use_window:
                # Windowed M-step: GEMM at reduced dimension, then scatter back
                # Use with-DC ctf2 for M-step accumulation (DC is excluded
                # from scoring but must be included in reconstruction weights).
                mstep_t0 = time.time()
                actual_rot = max(0, min(rotation_block_size, n_rot - r0))
                if relion_firstiter_winner_take_all:
                    probs = _winner_take_all_probs_for_block(
                        best_argmax,
                        r0,
                        actual_rot,
                        rotation_block_size,
                        n_trans,
                        scores.dtype,
                    )
                    P = probs.swapaxes(0, 1).reshape(rotation_block_size, batch_size * n_trans)
                    summed_windowed = P @ shifted_recon_windowed
                    probs_sum_t = jnp.sum(probs, axis=-1)
                    ctf_probs_windowed = probs_sum_t.T @ ctf2_over_nv_windowed_mstep
                    block_best = best_score
                    block_argmax = best_argmax - r0 * n_trans
                else:
                    (Ft_y, Ft_ctf, probs, block_best, block_argmax, summed_windowed, ctf_probs_windowed) = (
                        _m_step_block_windowed(
                            shifted_recon_windowed,
                            scores,
                            log_Z,
                            rots_b,
                            ctf2_over_nv_windowed_mstep,
                            Ft_y,
                            Ft_ctf,
                            batch_size,
                            n_trans,
                            n_recon_windowed,
                            image_shape,
                            volume_shape,
                        )
                    )
                if sync_timers:
                    _block_until_ready(
                        Ft_y,
                        Ft_ctf,
                        probs,
                        block_best,
                        block_argmax,
                        summed_windowed,
                        ctf_probs_windowed,
                    )
                timing.mstep_s += time.time() - mstep_t0

                if not disable_adjoint_y:
                    adjoint_y_t0 = time.time()
                    Ft_y = _adjoint_slice_volume_windowed(
                        summed_windowed,
                        recon_window_indices,
                        rots_b,
                        Ft_y,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        True,
                        False,
                        float(current_size // 2),
                    )
                    if sync_timers:
                        _block_until_ready(Ft_y)
                    timing.adjoint_y_s += time.time() - adjoint_y_t0

                if not disable_adjoint_ctf:
                    adjoint_ctf_t0 = time.time()
                    Ft_ctf = _adjoint_slice_volume_windowed(
                        ctf_probs_windowed,
                        recon_window_indices,
                        rots_b,
                        Ft_ctf,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        True,
                        False,
                        float(current_size // 2),
                    )
                    if sync_timers:
                        _block_until_ready(Ft_ctf)
                    timing.adjoint_ctf_s += time.time() - adjoint_ctf_t0
            else:
                # Non-windowed path: use with-DC ctf2 for M-step accumulation
                mstep_t0 = time.time()
                actual_rot = max(0, min(rotation_block_size, n_rot - r0))
                if relion_firstiter_winner_take_all:
                    probs = _winner_take_all_probs_for_block(
                        best_argmax,
                        r0,
                        actual_rot,
                        rotation_block_size,
                        n_trans,
                        scores.dtype,
                    )
                    P = probs.swapaxes(0, 1).reshape(rotation_block_size, batch_size * n_trans)
                    summed_half_block = P @ shifted_recon_half
                    probs_sum_t = jnp.sum(probs, axis=-1)
                    ctf_probs_half_block = probs_sum_t.T @ ctf2_over_nv_half_with_dc
                    block_best = best_score
                    block_argmax = best_argmax - r0 * n_trans
                else:
                    (probs, block_best, block_argmax, summed_half_block, ctf_probs_half_block) = (
                        _m_step_block_compute(
                            shifted_recon_half,
                            scores,
                            log_Z,
                            rots_b,
                            ctf2_over_nv_half_with_dc,
                            batch_size,
                            n_trans,
                        )
                    )
                if sync_timers:
                    _block_until_ready(
                        Ft_y,
                        Ft_ctf,
                        probs,
                        block_best,
                        block_argmax,
                        summed_half_block,
                        ctf_probs_half_block,
                    )
                timing.mstep_s += time.time() - mstep_t0

                if not disable_adjoint_y:
                    adjoint_y_t0 = time.time()
                    Ft_y = _adjoint_slice_volume_half(
                        summed_half_block,
                        rots_b,
                        Ft_y,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        True,
                    )
                    if sync_timers:
                        _block_until_ready(Ft_y)
                    timing.adjoint_y_s += time.time() - adjoint_y_t0

                if not disable_adjoint_ctf:
                    adjoint_ctf_t0 = time.time()
                    Ft_ctf = _adjoint_slice_volume_half(
                        ctf_probs_half_block,
                        rots_b,
                        Ft_ctf,
                        image_shape,
                        recon_volume_shape,
                        "linear_interp",
                        True,
                    )
                    if sync_timers:
                        _block_until_ready(Ft_ctf)
                    timing.adjoint_ctf_s += time.time() - adjoint_ctf_t0

            # -- Noise accumulation for this rotation block --
            if accumulate_noise:
                noise_t0 = time.time()
                if translation_sqdist_ang is not None:
                    translation_posterior = np.asarray(jnp.sum(probs, axis=1), dtype=np.float64)
                    noise_sigma2_offset += float(
                        np.sum(translation_posterior * translation_sqdist_ang, dtype=np.float64)
                    )
                rot_block_size_actual = rots_b.shape[0]
                # Compute masked GEMM: P @ shifted_masked (with DC intact)
                P_noise = probs.swapaxes(0, 1).reshape(rot_block_size_actual, batch_size * n_trans)
                summed_masked_noise = P_noise @ shifted_masked_for_noise  # (rot_block, N_noise)
                # ctf_probs for noise: recompute WITH DC (M-step used DC-zeroed version)
                probs_sum_t_noise = jnp.sum(probs, axis=-1)  # (n_images, rot_block)
                if use_window:
                    proj_recon_windowed_b = proj_half_b[:, recon_window_indices]
                    proj_abs2_recon_windowed_b = proj_abs2_half_b[:, recon_window_indices]
                    ctf2_nv_noise = ctf2_over_nv_half_with_dc[:, recon_window_indices]
                    ctf_probs_for_noise = probs_sum_t_noise.T @ ctf2_nv_noise
                    nv_for_noise = noise_variance_windowed
                    si_for_noise = shell_indices_noise
                    proj_for_noise = proj_recon_windowed_b
                    proj_abs2_for_noise = proj_abs2_recon_windowed_b
                else:
                    ctf_probs_for_noise = probs_sum_t_noise.T @ ctf2_over_nv_half_with_dc
                    nv_for_noise = noise_variance_half
                    si_for_noise = shell_indices_noise
                    proj_for_noise = proj_half_b
                    proj_abs2_for_noise = proj_abs2_half_b

                if dense_noise_component_acc:
                    for state in dense_noise_component_acc.values():
                        row = int(state["row"])
                        row_probs = probs[row]
                        row_shifted = shifted_masked_for_noise[row * n_trans : (row + 1) * n_trans]
                        row_summed_masked = row_probs @ row_shifted
                        row_ctf2_nv = ctf2_nv_noise[row] if use_window else ctf2_over_nv_half_with_dc[row]
                        row_ctf_probs = jnp.sum(row_probs, axis=-1)[:, None] * row_ctf2_nv[None, :]
                        row_ctf_probs_raw = row_ctf_probs * nv_for_noise[None, :]
                        row_a2_pixel = jnp.sum(proj_abs2_for_noise * row_ctf_probs_raw, axis=0)
                        row_xa_pixel = nv_for_noise * jnp.real(
                            jnp.sum(proj_for_noise * jnp.conj(row_summed_masked), axis=0)
                        )
                        state["a2_shells"] += _bin_shell_values_np(row_a2_pixel, si_for_noise, n_shells)
                        state["xa_shells"] += _bin_shell_values_np(row_xa_pixel, si_for_noise, n_shells)

                block_noise_shells, block_a2_shells, block_xa_shells = _compute_noise_block(
                    proj_for_noise,
                    proj_abs2_for_noise,
                    summed_masked_noise,
                    ctf_probs_for_noise,
                    nv_for_noise,
                    si_for_noise,
                    n_shells,
                    debug_options.return_noise_split,
                )
                if sync_timers:
                    if debug_options.return_noise_split:
                        _block_until_ready(block_noise_shells, block_a2_shells, block_xa_shells)
                    else:
                        _block_until_ready(block_noise_shells)
                noise_wsum += np.asarray(block_noise_shells, dtype=np.float64)
                if debug_options.return_noise_split:
                    noise_a2 += np.asarray(block_a2_shells, dtype=np.float64)
                    noise_xa += np.asarray(block_xa_shells, dtype=np.float64)
                timing.noise_s += time.time() - noise_t0

            if not relion_firstiter_winner_take_all:
                assignment_t0 = time.time()
                improved = block_best > best_score
                best_score = jnp.where(improved, block_best, best_score)
                best_argmax = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax)
                if sync_timers:
                    _block_until_ready(best_score, best_argmax)
                timing.assignment_s += time.time() - assignment_t0

            if return_stats:
                host_stats_t0 = time.time()
                actual_rot = max(0, min(rotation_block_size, n_rot - r0))
                if actual_rot > 0:
                    block_rotation_sums = np.asarray(
                        jnp.sum(probs[:, :actual_rot, :], axis=(0, 2)),
                        dtype=np.float64,
                    )
                    rotation_posterior_sums[r0 : r0 + actual_rot] += block_rotation_sums
                timing.host_stats_s += time.time() - host_stats_t0

        if dense_noise_component_acc:
            for global_idx, state in dense_noise_component_acc.items():
                p_img_shells = np.asarray(state["p_img_shells"], dtype=np.float64)
                a2_shells = np.asarray(state["a2_shells"], dtype=np.float64)
                xa_shells = np.asarray(state["xa_shells"], dtype=np.float64)
                total_shells = p_img_shells + a2_shells - 2.0 * xa_shells
                dump_path = (
                    debug_options.noise_component_dump_dir
                    / f"dense_noise_components_cs{int(current_size or -1):03d}_image_{int(global_idx)}.npz"
                )
                np.savez_compressed(
                    dump_path,
                    selected_global_image_indices=np.array([int(global_idx)], dtype=np.int64),
                    selected_local_image_indices=np.array([int(state["local_idx"])], dtype=np.int64),
                    current_size=np.array([int(current_size) if current_size is not None else -1], dtype=np.int32),
                    n_rot=np.array([int(n_rot)], dtype=np.int32),
                    n_trans=np.array([int(n_trans)], dtype=np.int32),
                    p_img_shells=p_img_shells,
                    a2_shells=a2_shells,
                    xa_shells=xa_shells,
                    total_shells=total_shells,
                    shell_indices_half=np.asarray(shell_indices_half, dtype=np.int32),
                    shell_indices_noise=np.asarray(shell_indices_noise, dtype=np.int32),
                )

        if return_stats:
            stats_finalize_t0 = time.time()
            log_score_offset = -0.5 * jnp.squeeze(batch_norm[:actual_batch_size], axis=1)
            log_Z_actual = log_Z[:actual_batch_size]
            best_score_actual = best_score[:actual_batch_size]
            pmax = jnp.exp(best_score - log_Z)
            log_evidence_per_image[start_idx:end_idx] = np.asarray(
                log_Z_actual + log_score_offset,
                dtype=np.float32,
            )
            best_log_score_per_image[start_idx:end_idx] = np.asarray(
                best_score_actual + log_score_offset,
                dtype=np.float32,
            )
            max_posterior_per_image[start_idx:end_idx] = np.asarray(
                pmax[:actual_batch_size],
                dtype=np.float32,
            )
            if sync_timers:
                _block_until_ready(log_Z, best_score, pmax)
            timing.stats_finalize_s += time.time() - stats_finalize_t0

        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax[:actual_batch_size])
        start_idx = end_idx

    # -- SOLVE --
    from recovar.reconstruction import relion_functions

    if reconstruction_padding_factor > 1:
        new_mean = None
    else:
        solve_t0 = time.time()
        new_mean = relion_functions.post_process_from_filter(
            experiment_dataset,
            Ft_ctf,
            Ft_y,
            tau=mean_variance,
            disc_type=disc_type,
        ).reshape(-1)
        if sync_timers:
            _block_until_ready(new_mean)
        timing.solve_s += time.time() - solve_t0

    noise_stats = None
    if accumulate_noise:
        # Diagnostic: log per-shell A2, XA, img_power, wsum for the first 6 shells
        # so we can compare across iterations of refine.
        try:
            n_log_shells = min(6, len(noise_wsum))
            logger.info(
                "[NOISE-DIAG] sumw=%.0f n_rot=%d use_window=%s",
                float(noise_sumw),
                int(n_rot),
                bool(use_window),
            )
            if debug_options.return_noise_split:
                logger.info(
                    "[NOISE-DIAG] A2 (first %d shells): %s",
                    n_log_shells,
                    ", ".join(f"{noise_a2[i]:.3e}" for i in range(n_log_shells)),
                )
                logger.info(
                    "[NOISE-DIAG] XA (first %d shells): %s",
                    n_log_shells,
                    ", ".join(f"{noise_xa[i]:.3e}" for i in range(n_log_shells)),
                )
            logger.info(
                "[NOISE-DIAG] img_power (first %d shells): %s",
                n_log_shells,
                ", ".join(f"{noise_img_power[i]:.3e}" for i in range(n_log_shells)),
            )
            logger.info(
                "[NOISE-DIAG] wsum=A2-2XA (first %d shells): %s",
                n_log_shells,
                ", ".join(f"{noise_wsum[i]:.3e}" for i in range(n_log_shells)),
            )
        except Exception as exc:
            logger.warning("noise diagnostic logging failed: %s", exc)
        noise_stats = NoiseStats(
            wsum_sigma2_noise=jnp.asarray(noise_wsum, dtype=jnp.float32),
            wsum_img_power=jnp.asarray(noise_img_power, dtype=jnp.float32),
            wsum_sigma2_offset=float(noise_sigma2_offset),
            sumw=float(noise_sumw),
            wsum_noise_a2=(jnp.asarray(noise_a2, dtype=jnp.float32) if debug_options.return_noise_split else None),
            wsum_noise_xa=(jnp.asarray(noise_xa, dtype=jnp.float32) if debug_options.return_noise_split else None),
        )

    if sparse_pass2 and sparse_profile.total_blocks:
        logger.info(
            "Sparse pass2 skipped %d / %d pass2 rotation blocks (%.1f%% of pass2 blocks); "
            "omitted posterior mass upper bound mean=%.3e max=%.3e sum=%.3e",
            sparse_profile.skipped_blocks,
            sparse_profile.total_blocks,
            100.0 * sparse_profile.skipped_blocks / sparse_profile.total_blocks,
            sparse_profile.omitted_mass_upper_mean,
            sparse_profile.omitted_mass_upper_max,
            sparse_profile.omitted_mass_upper_sum,
        )

    if return_stats:
        host_stats_t0 = time.time()
        relion_stats = RelionStats(
            log_evidence_per_image=jnp.asarray(log_evidence_per_image),
            best_log_score_per_image=jnp.asarray(best_log_score_per_image),
            max_posterior_per_image=jnp.asarray(max_posterior_per_image),
            rotation_posterior_sums=jnp.asarray(rotation_posterior_sums, dtype=jnp.float32),
        )
        timing.host_stats_s += time.time() - host_stats_t0
    if return_profile:
        ready_values = [new_mean, Ft_y, Ft_ctf]
        if noise_stats is not None:
            ready_values.extend([noise_stats.wsum_sigma2_noise, noise_stats.wsum_img_power])
        if return_stats:
            ready_values.extend(
                [
                    relion_stats.log_evidence_per_image,
                    relion_stats.best_log_score_per_image,
                    relion_stats.max_posterior_per_image,
                    relion_stats.rotation_posterior_sums,
                ]
            )
        _block_until_ready(*ready_values)
        total_wall_time = time.time() - overall_t0
        attributed_time = timing.accounted_s()
        em_profile = EMProfileStats(
            **timing.profile_kwargs(),
            accounted_s=float(attributed_time),
            total_wall_s=float(total_wall_time),
            unattributed_s=float(max(total_wall_time - attributed_time, 0.0)),
            n_images=int(n_images),
            n_trans=int(n_trans),
            n_rot=int(n_rot),
            n_rot_padded=int(n_rot_padded),
            n_blocks=int(n_blocks),
            n_windowed=int(n_windowed),
            use_window=bool(use_window),
            reused_pass1_projections=True,
            **sparse_profile.profile_kwargs(),
        )
    else:
        em_profile = None

    if return_stats:
        if accumulate_noise:
            if return_profile:
                return new_mean, hard_assignment, Ft_y, Ft_ctf, relion_stats, noise_stats, em_profile
            return new_mean, hard_assignment, Ft_y, Ft_ctf, relion_stats, noise_stats
        if return_profile:
            return new_mean, hard_assignment, Ft_y, Ft_ctf, relion_stats, em_profile
        return new_mean, hard_assignment, Ft_y, Ft_ctf, relion_stats

    if accumulate_noise:
        if return_profile:
            return new_mean, hard_assignment, Ft_y, Ft_ctf, noise_stats, em_profile
        return new_mean, hard_assignment, Ft_y, Ft_ctf, noise_stats

    if return_profile:
        return new_mean, hard_assignment, Ft_y, Ft_ctf, em_profile
    return new_mean, hard_assignment, Ft_y, Ft_ctf


def compute_e_step_weights(
    experiment_dataset,
    mean,
    noise_variance,
    rotations,
    translations,
    disc_type: str,
    image_batch_size: int = 500,
    rotation_block_size: int = 5000,
    current_size: int = None,
    score_with_masked_images: bool = False,
):
    """E-step only: compute posterior weights for all (rotation, translation) pairs.

    This runs pass 1 (logsumexp) and pass 2 (normalize) of the blockwise
    E-step but does NOT accumulate M-step statistics.  Used by the adaptive
    oversampling module to identify significant samples before pass 2.

    Returns the posterior weight matrix (n_images, n_rot * n_trans) which
    sums to ~1.0 per image.  For large grids this can be memory-intensive;
    the caller should use it for significance pruning then discard it.

    Parameters
    ----------
    experiment_dataset : dataset object
    mean : jnp.ndarray, shape (volume_size,)
    noise_variance : jnp.ndarray, shape (image_size,)
    rotations : np.ndarray, shape (n_rot, 3, 3)
    translations : jnp.ndarray, shape (n_trans, 2)
    disc_type : str
    image_batch_size : int
    rotation_block_size : int
    current_size : int or None

    Returns
    -------
    weights : np.ndarray, shape (n_images, n_rot * n_trans), dtype float32
        Posterior weights (probabilities).
    hard_assignments : np.ndarray, shape (n_images,), dtype int32
        Best (rotation_idx * n_trans + trans_idx) per image.
    """
    n_rot = rotations.shape[0]
    n_trans = translations.shape[0]
    n_images = experiment_dataset.n_units
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape

    H, W = image_shape
    n_half = H * (W // 2 + 1)

    config = ForwardModelConfig.from_dataset(
        experiment_dataset,
        disc_type=disc_type,
        process_fn=experiment_dataset.process_images,
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(noise_variance, image_shape).squeeze()

    half_weights = make_half_image_weights(image_shape)
    precision_policy = DensePrecisionPolicy()

    window_spec = make_fourier_window_spec(
        image_shape,
        current_size,
        n_half,
        include_recon_window=False,
    )
    use_window = window_spec.use_window
    window_indices = window_spec.score_indices
    n_windowed = window_spec.n_score
    projection_kwargs = window_spec.projection_kwargs()
    if use_window:
        half_weights_windowed = window_spec.score_values(half_weights)

    n_blocks = (n_rot + rotation_block_size - 1) // rotation_block_size
    n_rot_padded = n_blocks * rotation_block_size
    if n_rot_padded > n_rot:
        pad_size = n_rot_padded - n_rot
        rotations_padded = np.concatenate([rotations, np.tile(np.eye(3, dtype=np.float32), (pad_size, 1, 1))], axis=0)
    else:
        rotations_padded = rotations

    # Allocate output weights array on host
    all_weights = np.empty((n_images, n_rot * n_trans), dtype=np.float32)
    hard_assignment = np.empty(n_images, dtype=np.int32)

    image_indices = np.arange(n_images)
    start_idx = 0

    for batch_data, _, _, ctf_params, _, _, indices in experiment_dataset.iter_batches(
        image_batch_size,
        indices=image_indices,
        by_image=False,
    ):
        batch_size = len(indices)
        end_idx = start_idx + batch_size
        batch_data = jnp.asarray(batch_data)

        shifted_half, batch_norm, ctf2_over_nv_half = _preprocess_batch(
            experiment_dataset,
            batch_data,
            ctf_params,
            noise_variance_half,
            translations,
            config,
            score_with_masked_images,
        )

        if use_window:
            shifted_windowed = shifted_half[:, window_indices]
            ctf2_over_nv_windowed = ctf2_over_nv_half[:, window_indices]
        else:
            shifted_windowed = shifted_half
            ctf2_over_nv_windowed = ctf2_over_nv_half

        # Pass 1: streaming logsumexp
        max_s = jnp.full(batch_size, -jnp.inf)
        sum_exp = jnp.zeros(batch_size, dtype=precision_policy.normalization_real_dtype)

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean,
                rots_b,
                image_shape,
                volume_shape,
                disc_type,
                **projection_kwargs,
            )

            if use_window:
                proj_windowed_b = proj_half_b[:, window_indices]
                proj_abs2_windowed_b = proj_abs2_half_b[:, window_indices]
                proj_windowed_weighted_b = proj_windowed_b * half_weights_windowed
                proj_abs2_windowed_weighted_b = proj_abs2_windowed_b * half_weights_windowed
                scores = _e_step_block_scores_windowed(
                    shifted_windowed,
                    batch_norm,
                    ctf2_over_nv_windowed,
                    proj_windowed_weighted_b,
                    proj_abs2_windowed_weighted_b,
                    half_weights_windowed,
                    batch_size,
                    n_trans,
                    n_windowed,
                    image_shape,
                    volume_shape,
                )
            else:
                proj_half_weighted_b = proj_half_b * half_weights
                proj_abs2_weighted_b = proj_abs2_half_b * half_weights
                scores = _e_step_block_scores(
                    shifted_half,
                    batch_norm,
                    ctf2_over_nv_half,
                    proj_half_weighted_b,
                    proj_abs2_weighted_b,
                    half_weights,
                    batch_size,
                    n_trans,
                    image_shape,
                    volume_shape,
                )

            if r1 > n_rot:
                valid = n_rot - r0
                mask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(mask[None, :, None], scores, -jnp.inf)

            max_s, sum_exp = _update_logsumexp(max_s, sum_exp, scores)

        log_Z = max_s + jnp.log(sum_exp)

        # Pass 2: recompute scores and normalize to weights
        best_score = jnp.full(batch_size, -jnp.inf)
        best_argmax = jnp.zeros(batch_size, dtype=jnp.int32)
        batch_weights_blocks = []

        for b in range(n_blocks):
            r0 = b * rotation_block_size
            r1 = r0 + rotation_block_size
            rots_b = rotations_padded[r0:r1]

            proj_half_b, proj_abs2_half_b = _compute_projections_block(
                mean,
                rots_b,
                image_shape,
                volume_shape,
                disc_type,
                **projection_kwargs,
            )

            if use_window:
                proj_windowed_b = proj_half_b[:, window_indices]
                proj_abs2_windowed_b = proj_abs2_half_b[:, window_indices]
                proj_windowed_weighted_b = proj_windowed_b * half_weights_windowed
                proj_abs2_windowed_weighted_b = proj_abs2_windowed_b * half_weights_windowed
                scores = _e_step_block_scores_windowed(
                    shifted_windowed,
                    batch_norm,
                    ctf2_over_nv_windowed,
                    proj_windowed_weighted_b,
                    proj_abs2_windowed_weighted_b,
                    half_weights_windowed,
                    batch_size,
                    n_trans,
                    n_windowed,
                    image_shape,
                    volume_shape,
                )
            else:
                proj_half_weighted_b = proj_half_b * half_weights
                proj_abs2_weighted_b = proj_abs2_half_b * half_weights
                scores = _e_step_block_scores(
                    shifted_half,
                    batch_norm,
                    ctf2_over_nv_half,
                    proj_half_weighted_b,
                    proj_abs2_weighted_b,
                    half_weights,
                    batch_size,
                    n_trans,
                    image_shape,
                    volume_shape,
                )

            if r1 > n_rot:
                valid = n_rot - r0
                pmask = jnp.arange(rotation_block_size) < valid
                scores = jnp.where(pmask[None, :, None], scores, -jnp.inf)

            # Normalize to probabilities
            probs = jnp.exp(scores - log_Z[:, None, None])

            # Track hard assignment
            block_best = jnp.max(scores.reshape(batch_size, -1), axis=1)
            block_argmax = jnp.argmax(scores.reshape(batch_size, -1), axis=1)
            improved = block_best > best_score
            best_score = jnp.where(improved, block_best, best_score)
            best_argmax = jnp.where(improved, block_argmax + r0 * n_trans, best_argmax)

            # Trim padding rotations and store block weights
            actual_rot = min(rotation_block_size, n_rot - r0)
            block_probs = probs[:, :actual_rot, :]  # (batch, actual_rot, n_trans)
            batch_weights_blocks.append(np.asarray(block_probs.reshape(batch_size, -1)))

        # Concatenate blocks -> (batch_size, n_rot * n_trans)
        batch_weights = np.concatenate(batch_weights_blocks, axis=1)
        all_weights[start_idx:end_idx] = batch_weights
        hard_assignment[start_idx:end_idx] = np.asarray(best_argmax)
        start_idx = end_idx

    return all_weights, hard_assignment
