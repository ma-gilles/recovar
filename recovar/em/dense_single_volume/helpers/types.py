"""Typed containers for the dense single-volume EM path."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


def _stats_array(value, dtype, host_arrays):
    """Keep host-published fields on the host with JAX's dtype policy."""
    if type(host_arrays) is not bool:
        raise TypeError("host_arrays must be a bool")
    if not host_arrays:
        return jnp.asarray(value, dtype=dtype)
    value_dtype = value.dtype if hasattr(value, "dtype") else np.asarray(value).dtype
    dtype = jax.dtypes.canonicalize_dtype(value_dtype if dtype is None else dtype)
    if isinstance(value, jax.Array) and value.dtype != dtype:
        # Device casts can flush subnormals differently from NumPy. Keep the
        # established conversion semantics before publishing the result.
        value = jnp.asarray(value, dtype=dtype)
    return np.asarray(value, dtype=dtype)


class MeanStats(NamedTuple):
    """Accumulated M-step sufficient statistics.

    Both fields are additive over image batches and across devices,
    making this the natural unit for distributed all-reduce.

    Attributes:
        Ft_y: (volume_size,) complex -- weighted backprojected images.
        Ft_ctf: (volume_size,) real/complex -- weighted CTF^2 backprojection.
    """

    Ft_y: jax.Array
    Ft_ctf: jax.Array


class RelionStats(NamedTuple):
    """Per-image E-step statistics needed by the RELION-style refine loop.

    These fields are not additive like :class:`MeanStats`; they are emitted
    per iteration so convergence and current-size logic can reuse the exact
    normalization already computed inside ``run_em``. Fields may be NumPy
    arrays when explicitly published for a host consumer.

    Attributes:
        log_evidence_per_image: Log normalizer ``log_Z`` for each image.
        best_log_score_per_image: Maximum unnormalized log-score per image.
        max_posterior_per_image: Maximum posterior probability per image.
        rotation_posterior_sums: Posterior mass accumulated per rotation over
            the processed image subset. This is additive across batches and can
            be collapsed to RELION-style ``pdf_direction`` updates.
    """

    log_evidence_per_image: jax.Array | np.ndarray
    best_log_score_per_image: jax.Array | np.ndarray
    max_posterior_per_image: jax.Array | np.ndarray
    rotation_posterior_sums: jax.Array | np.ndarray


class NoiseStats(NamedTuple):
    """Posterior-weighted noise shell statistics (RELION parity).

    Accumulated during the M-step pass of ``run_em`` when
    ``accumulate_noise=True``.  These are additive over image batches
    and across half-sets, matching RELION's ``wsum_sigma2_noise``.

    RELION formula::

        sigma2_noise[s] = (wsum_sigma2_noise[s] + wsum_img_power[s])
                          / (2 * sumw * Npix_per_shell[s])

    Attributes:
        wsum_sigma2_noise: (n_shells,) float -- accumulated
            ``sum_{i,r,t} w * (A2 - 2*XA)`` per shell (reference-dependent
            part of the residual).
        wsum_img_power: (n_shells,) float -- accumulated
            ``sum_i mass_i * |img_masked_i|^2`` per shell, where ``mass_i`` is
            the same significant-support posterior mass used for A2/XA.
            For ungated full-grid updates ``mass_i == 1``.
        wsum_sigma2_offset: float -- accumulated
            ``sum_{i,r,t} w * ||offset_{i,t} - prior_i||^2`` in Angstrom^2.
            This is RELION's sufficient statistic for updating
            ``sigma2_offset`` via ``wsum_sigma2_offset / (2 * sumw)``.
        sumw: float -- total posterior/support weight processed (equals the
            number of images when posteriors are normalised to sum to 1 per
            image and no significant-support pruning is active).
        wsum_noise_a2: optional diagnostic split of ``wsum_sigma2_noise``.
        wsum_noise_xa: optional diagnostic split of ``wsum_sigma2_noise``.
        wsum_norm_correction: optional per-image residual sums for RELION's
            ``normcorr = old_normcorr / avg_norm * sqrt(2 * residual)`` update.
        wsum_scale_correction_xa: optional per-group signal-product sums for
            RELION's group scale update.
        wsum_scale_correction_aa: optional per-group reference-power sums for
            RELION's group scale update.
    """

    wsum_sigma2_noise: jax.Array | np.ndarray
    wsum_img_power: jax.Array | np.ndarray
    wsum_sigma2_offset: float
    sumw: float
    wsum_noise_a2: jax.Array | np.ndarray | None = None
    wsum_noise_xa: jax.Array | np.ndarray | None = None
    wsum_norm_correction: jax.Array | np.ndarray | None = None
    wsum_scale_correction_xa: jax.Array | np.ndarray | None = None
    wsum_scale_correction_aa: jax.Array | np.ndarray | None = None


def make_relion_stats(
    *,
    log_evidence_per_image,
    best_log_score_per_image,
    max_posterior_per_image,
    rotation_posterior_sums,
    image_dtype=None,
    rotation_dtype=None,
    host_arrays: bool = False,
) -> RelionStats:
    """Build a ``RelionStats`` object with consistent array conversion."""

    return RelionStats(
        log_evidence_per_image=_stats_array(log_evidence_per_image, image_dtype, host_arrays),
        best_log_score_per_image=_stats_array(best_log_score_per_image, image_dtype, host_arrays),
        max_posterior_per_image=_stats_array(max_posterior_per_image, image_dtype, host_arrays),
        rotation_posterior_sums=_stats_array(rotation_posterior_sums, rotation_dtype, host_arrays),
    )


def make_noise_stats(
    *,
    wsum_sigma2_noise,
    wsum_img_power,
    wsum_sigma2_offset,
    sumw,
    wsum_noise_a2=None,
    wsum_noise_xa=None,
    wsum_norm_correction=None,
    wsum_scale_correction_xa=None,
    wsum_scale_correction_aa=None,
    array_dtype=None,
    host_arrays: bool = False,
) -> NoiseStats:
    """Build a ``NoiseStats`` object without narrowing source precision.

    Pass ``array_dtype`` only for an explicit external precision boundary.
    Otherwise each accumulator retains the dtype chosen by its producer.
    """

    return NoiseStats(
        wsum_sigma2_noise=_stats_array(wsum_sigma2_noise, array_dtype, host_arrays),
        wsum_img_power=_stats_array(wsum_img_power, array_dtype, host_arrays),
        wsum_sigma2_offset=float(wsum_sigma2_offset),
        sumw=float(sumw),
        wsum_noise_a2=None if wsum_noise_a2 is None else _stats_array(wsum_noise_a2, array_dtype, host_arrays),
        wsum_noise_xa=None if wsum_noise_xa is None else _stats_array(wsum_noise_xa, array_dtype, host_arrays),
        wsum_norm_correction=None
        if wsum_norm_correction is None
        else _stats_array(wsum_norm_correction, array_dtype, host_arrays),
        wsum_scale_correction_xa=None
        if wsum_scale_correction_xa is None
        else _stats_array(wsum_scale_correction_xa, array_dtype, host_arrays),
        wsum_scale_correction_aa=None
        if wsum_scale_correction_aa is None
        else _stats_array(wsum_scale_correction_aa, array_dtype, host_arrays),
    )


class EMProfileStats(NamedTuple):
    """Host-side timing and work counters for one ``run_em`` call.

    These values are diagnostic only and must not change numerical behavior.
    """

    batch_fetch_s: float
    preprocess_s: float
    score_prep_s: float
    pass1_projection_s: float
    pass1_score_s: float
    pass1_postprocess_s: float
    pass1_logsumexp_s: float
    pass2_skipmask_s: float
    pass2_projection_s: float
    pass2_score_s: float
    pass2_postprocess_s: float
    mstep_s: float
    window_scatter_s: float
    adjoint_y_s: float
    adjoint_ctf_s: float
    noise_s: float
    assignment_s: float
    stats_finalize_s: float
    host_stats_s: float
    solve_s: float
    accounted_s: float
    total_wall_s: float
    unattributed_s: float
    n_images: int
    n_trans: int
    n_rot: int
    n_rot_padded: int
    n_blocks: int
    n_windowed: int
    use_window: bool
    reused_pass1_projections: bool
    sparse_pass2_total_blocks: int
    sparse_pass2_skipped_blocks: int
    sparse_pass2_omitted_mass_upper_mean: float
    sparse_pass2_omitted_mass_upper_max: float
    sparse_pass2_omitted_mass_upper_sum: float


@dataclass(frozen=True)
class LocalEMResult:
    """Exact-local engine outputs with a fixed field layout.

    Accumulators retain their dtype, device and requested Fourier layout.
    Image fields follow the selected dataset order. Optional pose, noise,
    profile and significant-count fields are ``None`` when not requested.
    Construction stores references without copying or synchronizing arrays.
    """

    Ft_y: jax.Array | np.ndarray
    Ft_ctf: jax.Array | np.ndarray
    hard_assignments: np.ndarray
    stats: RelionStats
    best_pose_rotations: jax.Array | np.ndarray | None = None
    best_pose_translations: jax.Array | np.ndarray | None = None
    best_pose_rotation_ids: np.ndarray | None = None
    noise_stats: NoiseStats | None = None
    profile: dict | None = None
    significant_counts: np.ndarray | None = None


@dataclass(frozen=True)
class DenseEMResult:
    """Result of ``em_engine.run_em`` with stable, named fields.

    ``hard_assignments`` follows the selected dataset image order and encodes
    ``rotation_index * n_translations + translation_index``. Fourier accumulators
    keep the engine's dtype, device and requested full/packed-half layout.
    ``mean`` is absent when the call does not reconstruct a volume.

    ``stats``, ``noise_stats`` and ``profile`` are ``None`` when their respective
    return/accumulation flags are disabled. This container only stores references;
    it does not copy arrays, synchronize devices or alter buffer lifetime itself.
    """

    mean: jax.Array | np.ndarray | None
    hard_assignments: np.ndarray
    Ft_y: jax.Array | np.ndarray
    Ft_ctf: jax.Array | np.ndarray
    stats: RelionStats | None = None
    noise_stats: NoiseStats | None = None
    profile: EMProfileStats | None = None
