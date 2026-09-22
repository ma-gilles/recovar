"""Device-resident noise, norm, scale and posterior statistics for K=1 pass 2.

Stage 7 of ``../em_device_resident_pass2_design_20260918.md``. This module is
the device twin of the host bucket tail in
:func:`recovar.em.sparse_pass2.sparse_pass2_bucketed.compute_pass2_stats_sparse_bucketed`
(its inner ``_bucket_tail``): instead of pulling ~35 per-bucket arrays to the
host and accumulating them with numpy, one traced program folds a chunk's
operands into float64 accumulators that stay on the device until
:func:`finalize_statistics` pulls them once per half.

Ported production branches (the fresh K=1 guard, confirmed from
``sparse_pass2_policy._fresh_k1_direct_noise_default`` and
``sparse_pass2_bucketed`` lines ~2190-2262): ``accumulate_noise=True``,
``use_relion_fine_mstep_prune=True`` (so the noise and statistics posteriors
are the same pruned ``reconstruction_probs``), RELION x-half M-step,
``relion_wavg_atomic_scale_aa=True``, ``relion_wavg_atomic_direct_noise=True``,
``relion_wavg_atomic_direct_norm=False`` ("per-particle norm
mode=production-algebraic"), ``translated_wavg_norm=False``. The two stopped
diagnostic arms (direct Wavg norm, translated Wavg norm) and the per-bucket
operand dumps are deliberately *not* ported; see the module's ticket report.

Layout and padding contract
---------------------------
Operands arrive in the flat, image-CSR row layout built by
:mod:`recovar.em.sparse_pass2.resident_candidates`: ``C_R`` padded candidate
rows (one per ``(image, fine rotation)`` pair) and ``C_B`` padded image slots
from the capacity ladders. Callers must guarantee that

* padded rows carry an all-zero posterior, an all-zero ``ctf_probs`` and an
  all-zero ``summed_masked`` (the kernels' own ``!= 0`` mass predicates then
  neutralise whatever is in ``proj``/``proj_abs2``), and a sentinel coarse
  rotation id;
* padded image slots carry ``image_ids = -1``, zero ``support``-bearing
  posterior rows and a zero Wavg triplet.

Scatter sentinels. ``.at[ids].add(..., mode="drop")`` does **not** drop
negative indices: JAX normalises ``-1`` to ``n - 1`` and the padded slots would
all accumulate into the last real image (measured on JAX 0.9.0.1, CPU and GPU
share the semantics). Every scatter here therefore maps padded ids to the
accumulator length, which ``mode="drop"`` does discard, and additionally zeroes
the padded values.

Numerics. Accumulators are float64. Element-wise arithmetic is copied from the
host helpers unchanged; only the association order of the reductions changes
(chunk partials instead of per-bucket host sums), which the user waived on
2026-09-18. Production precision policy is untouched: nothing here promotes the
float32 scoring, projection or Wavg streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.deterministic_reduce import (
    add_segment_sum,
    deterministic_reductions_enabled,
    fixed_order_segment_sum,
)
from recovar.em.helpers.env_flags import parse_env_flag
from recovar.em.helpers.projection import compute_noise_block
from recovar.em.sparse_pass2.sparse_pass2_policy import _RELION_POWERCLASS_SPECTRUM_NORM_ENV
from recovar.em.sparse_pass2.sparse_pass2_wavg import (
    _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp,
    _weighted_image_power_shells_and_per_image_core,
)

__all__ = [
    "ChunkStatisticsOperands",
    "FinalizedStatistics",
    "ResidentStatistics",
    "ResidentStatisticsConfig",
    "ResidentStatisticsTables",
    "accumulate_chunk_statistics",
    "finalize_statistics",
    "make_resident_statistics",
    "resolve_statistics_config",
    "segment_sum_by_image",
]


class ResidentStatistics(NamedTuple):
    """Device float64 accumulators for one half of one pass-2 iteration.

    A ``NamedTuple`` so the whole state is one pytree that can be threaded
    through ``jax.jit`` without host round trips. The first seven fields are
    accumulated (``+=``); the per-image score/pose fields are *written* once,
    because every image belongs to exactly one chunk.

    Units follow the host accumulators they replace: ``wsum_sigma2_noise`` and
    ``wsum_img_power`` are RELION ``wsum_model`` shell sums,
    ``sigma2_offset``/``sumw`` its scalar offset and support mass,
    ``norm_correction`` the per-particle normalisation correction,
    ``scale_xa``/``scale_aa`` the per-group scale sufficient statistics and
    ``rotation_posterior_sums`` posterior mass per *coarse* rotation.
    ``best_cell`` is ``global fine rotation id * n_fine_trans + t``;
    ``best_local_rot`` is the same winner's image-local rotation slot, which is
    what the host's ``hard_assignment`` counts in.
    """

    wsum_sigma2_noise: jax.Array  # float64 [n_shells]
    wsum_img_power: jax.Array  # float64 [n_shells]
    sigma2_offset: jax.Array  # float64 []
    sumw: jax.Array  # float64 []
    norm_correction: jax.Array  # float64 [n_images]
    scale_xa: jax.Array  # float64 [n_scale_groups]
    scale_aa: jax.Array  # float64 [n_scale_groups]
    rotation_posterior_sums: jax.Array  # float64 [n_coarse_rot]
    log_evidence: jax.Array  # float64 [n_images]
    best_log_score: jax.Array  # float64 [n_images]
    max_posterior: jax.Array  # score real dtype [n_images]
    best_cell: jax.Array  # int64 [n_images], -1 until written
    score_log_z: jax.Array  # float64 [n_images]
    best_local_rot: jax.Array  # int32 [n_images], -1 until written
    invalid_best_rows: jax.Array  # int64 [], padding sanity counter


class ResidentStatisticsTables(NamedTuple):
    """Iteration-global device tables shared by every chunk of one half.

    None of these depend on the chunk, so they are uploaded once. ``noise_*``
    live on the reconstruction/score window pixel axis ``P``; ``shell_indices_half``
    lives on the preprocessed half-image axis ``P_half``; ``wavg_*`` live on
    RELION's complete Wavg rectangle axis ``P_rect``.
    """

    translation_sqdist_ang: jax.Array | None  # float [n_fine_trans] or None
    noise_variance: jax.Array  # float [P]
    shell_indices_noise: jax.Array  # int32 [P]
    shell_indices_half: jax.Array  # int32 [P_half]
    wavg_shell_indices: jax.Array | None  # int32 [P_rect]
    wavg_scale_pixel_mask: jax.Array | None  # bool [P_rect]
    scale_pixel_mask: jax.Array | None  # bool [P], algebraic scale branch only


class ChunkStatisticsOperands(NamedTuple):
    """One capacity-padded chunk's operands, exactly as stages 3-6 leave them.

    Row-capacity arrays are indexed by the flat candidate row; image-capacity
    arrays by the chunk's image slot. ``row_image_local`` maps the former to
    the latter. Everything is already on the device; nothing here is pulled.
    """

    # --- row capacity C_R -------------------------------------------------
    row_image_local: jax.Array  # int32 [C_R]
    row_fine_rot: jax.Array  # int32 [C_R], global fine rotation id
    row_coarse_rot: jax.Array  # int32 [C_R], global coarse rotation id, padded -> >= n_coarse_rot
    row_posterior: jax.Array  # float32 [C_R, T], pruned reconstruction probs
    proj: jax.Array  # complex [C_R, P]
    proj_abs2: jax.Array  # float [C_R, P]
    summed_masked: jax.Array  # complex [C_R, P]
    ctf_probs: jax.Array  # float [C_R, P]
    # --- image capacity C_B ----------------------------------------------
    image_ids: jax.Array  # int32 [C_B], padded -> -1
    image_row_start: jax.Array  # int32 [C_B], chunk-local first row of the image
    image_row_count: jax.Array  # int32 [C_B], rows owned by the image
    group_ids: jax.Array  # int32 [C_B], global scale group, padded -> -1
    processed_image_half: jax.Array  # complex [C_B, P_half]
    relion_norm_high_shell: jax.Array  # float [C_B]
    scale: jax.Array  # float [C_B], the old group scale per image
    wavg_triplet_pixels: jax.Array | None  # float32 [C_B, P_rect, 3] = (XA, AA, diff2)
    best_row_local: jax.Array  # int32 [C_B], winner's chunk-local row
    best_translation: jax.Array  # int32 [C_B], winner's fine translation
    class_log_z: jax.Array  # float64 [C_B]
    min_diff2: jax.Array  # float [C_B], RELION's common-min centering
    best_log_score: jax.Array  # float64 [C_B]
    max_posterior: jax.Array  # float [C_B]
    batch_norm: jax.Array | None  # float [C_B, 1], non-exact-Gaussian branch only


@dataclass(frozen=True)
class ResidentStatisticsConfig:
    """Compile-time configuration of the statistics program.

    Hashable and passed as a ``jax.jit`` static argument, so every field here
    is part of the program's cache key. Shapes (``C_R``, ``C_B``, ``T``, ``P``,
    ``P_half``, ``P_rect``) are static through the operand avals, which is what
    keeps the number of traced programs at one per capacity class rather than
    one per bucket.
    """

    n_shells: int
    n_fine_trans: int
    n_images: int
    n_coarse_rot: int
    n_scale_groups: int
    # RELION powerClass split: shells strictly above the cutoff are unweighted.
    norm_unweighted_shell_cutoff: int | None
    include_unweighted_high_shell: bool
    # Resolved on the host so the environment is part of the jit cache key.
    disable_cuda_binning: bool
    deterministic_norm_reduction: bool
    use_exact_relion_gaussian: bool
    relion_wavg_atomic_direct_noise: bool
    relion_wavg_atomic_scale_aa: bool
    # ``current_size // 2 + 1`` in shell coordinates; only read when
    # ``relion_wavg_atomic_direct_noise`` is set.
    direct_noise_exclusive_shell_stop: int
    accumulate_scale: bool

    def __post_init__(self):
        for name in (
            "n_shells",
            "n_fine_trans",
            "n_images",
            "n_coarse_rot",
            "n_scale_groups",
            "direct_noise_exclusive_shell_stop",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.relion_wavg_atomic_direct_noise and not self.relion_wavg_atomic_scale_aa:
            raise ValueError(
                "direct Wavg noise replacement requires the atomic Wavg scale triplet "
                "(sparse_pass2_policy._relion_wavg_direct_modes enforces the same pairing)"
            )
        if self.accumulate_scale and int(self.n_scale_groups) <= 0:
            raise ValueError("accumulate_scale requires at least one scale group")


def resolve_statistics_config(
    *,
    n_shells: int,
    n_fine_trans: int,
    n_images: int,
    n_coarse_rot: int,
    n_scale_groups: int,
    current_size: int | None,
    include_unweighted_high_shell: bool = True,
    use_exact_relion_gaussian: bool = True,
    relion_wavg_atomic_direct_noise: bool = True,
    relion_wavg_atomic_scale_aa: bool = True,
    accumulate_scale: bool = True,
    source_faithful_spectrum_norm: bool | None = None,
) -> ResidentStatisticsConfig:
    """Build a config, reading the same environment the host tail reads.

    ``source_faithful_spectrum_norm`` mirrors
    ``sparse_pass2_wavg._weighted_image_power_shells_and_per_image``: it selects
    RELION's powerClass shell spectrum and, with it, the deterministic float64
    norm reduction. Resolving it here (rather than inside the traced function)
    keeps a later environment change from silently reusing a stale program.
    """

    if source_faithful_spectrum_norm is None:
        source_faithful_spectrum_norm = parse_env_flag(
            _RELION_POWERCLASS_SPECTRUM_NORM_ENV,
            default=False,
        )
    source_faithful_spectrum_norm = bool(source_faithful_spectrum_norm)
    deterministic_norm_reduction = source_faithful_spectrum_norm or parse_env_flag(
        "RECOVAR_K1_RELION_DETERMINISTIC_NORM_REDUCTION",
        default=False,
    )
    cutoff = None if current_size is None else int(current_size) // 2
    return ResidentStatisticsConfig(
        n_shells=int(n_shells),
        n_fine_trans=int(n_fine_trans),
        n_images=int(n_images),
        n_coarse_rot=int(n_coarse_rot),
        n_scale_groups=int(n_scale_groups),
        norm_unweighted_shell_cutoff=cutoff,
        include_unweighted_high_shell=bool(include_unweighted_high_shell),
        disable_cuda_binning=parse_env_flag("RECOVAR_DISABLE_CUDA", default=False),
        deterministic_norm_reduction=bool(deterministic_norm_reduction),
        use_exact_relion_gaussian=bool(use_exact_relion_gaussian),
        relion_wavg_atomic_direct_noise=bool(relion_wavg_atomic_direct_noise),
        relion_wavg_atomic_scale_aa=bool(relion_wavg_atomic_scale_aa),
        direct_noise_exclusive_shell_stop=int(n_shells) if cutoff is None else cutoff + 1,
        accumulate_scale=bool(accumulate_scale),
    )


def make_resident_statistics(
    config: ResidentStatisticsConfig,
    *,
    max_posterior_dtype=jnp.float32,
) -> ResidentStatistics:
    """Zeroed accumulators matching the host arrays this stage replaces.

    ``best_cell``/``best_local_rot`` start at ``-1`` so
    :func:`finalize_statistics` can tell an image that no chunk covered from
    one whose winner is genuinely row 0, translation 0.
    """

    n_images = int(config.n_images)
    zeros_images = jnp.zeros(n_images, dtype=jnp.float64)
    return ResidentStatistics(
        wsum_sigma2_noise=jnp.zeros(int(config.n_shells), dtype=jnp.float64),
        wsum_img_power=jnp.zeros(int(config.n_shells), dtype=jnp.float64),
        sigma2_offset=jnp.zeros((), dtype=jnp.float64),
        sumw=jnp.zeros((), dtype=jnp.float64),
        norm_correction=zeros_images,
        scale_xa=jnp.zeros(int(config.n_scale_groups), dtype=jnp.float64),
        scale_aa=jnp.zeros(int(config.n_scale_groups), dtype=jnp.float64),
        rotation_posterior_sums=jnp.zeros(int(config.n_coarse_rot), dtype=jnp.float64),
        log_evidence=jnp.full(n_images, -jnp.inf, dtype=jnp.float64),
        best_log_score=jnp.full(n_images, -jnp.inf, dtype=jnp.float64),
        max_posterior=jnp.zeros(n_images, dtype=max_posterior_dtype),
        best_cell=jnp.full(n_images, -1, dtype=jnp.int64),
        score_log_z=jnp.full(n_images, -jnp.inf, dtype=jnp.float64),
        best_local_rot=jnp.full(n_images, -1, dtype=jnp.int32),
        invalid_best_rows=jnp.zeros((), dtype=jnp.int64),
    )


def segment_sum_by_image(values, row_image_local, image_capacity: int):
    """Sum candidate rows into their image slots.

    ``values`` has the row axis first and any trailing axes; the result has the
    image axis first. Uses the same duplicate-index scatter as the host path,
    or the fixed-order masked reduction under
    ``RECOVAR_EM_DETERMINISTIC_REDUCTIONS=1``.
    """

    values = jnp.asarray(values)
    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    image_capacity = int(image_capacity)
    if deterministic_reductions_enabled():
        moved = jnp.moveaxis(values, 0, -1)
        return jnp.moveaxis(fixed_order_segment_sum(moved, row_image_local, image_capacity), -1, 0)
    accumulator = jnp.zeros((image_capacity,) + values.shape[1:], dtype=values.dtype)
    return accumulator.at[row_image_local].add(values)


def _drop_index(ids, length: int):
    """Map padded ids (any negative sentinel) to an index ``mode="drop"`` drops.

    ``mode="drop"`` keeps negative indices (it normalises ``-1`` to ``n - 1``),
    so a padded slot would otherwise contaminate the last real entry.
    """

    ids = jnp.asarray(ids, dtype=jnp.int32)
    return jnp.where(ids < 0, jnp.int32(length), ids)


def _flat_row_norm_and_scale_terms(
    proj,
    proj_abs2,
    summed_masked,
    ctf_probs,
    noise_variance,
    *,
    pixel_mask=None,
):
    """Per-row ``A2`` and ``XA`` partials in the flat candidate-row layout.

    Term for term this is
    :func:`recovar.em.helpers.projection.compute_norm_residual_per_image` (and,
    with ``pixel_mask``, its scale-correction sibling) with the ``(B, R, P)``
    rotation axis folded into the row axis: the same masks, the same
    ``noise_variance`` broadcast and the same natural dtype promotion, summed
    over pixels only. The caller sums rows into images and forms
    ``A2 - 2 * XA`` afterwards, exactly as the host does after its
    ``axis=(1, 2)`` reduction.
    """

    ctf_has_mass = ctf_probs != 0.0
    cross_has_mass = summed_masked != 0.0
    if pixel_mask is not None:
        pixel_mask = jnp.asarray(pixel_mask, dtype=bool).reshape(-1)
        ctf_has_mass = ctf_has_mass & pixel_mask[None, :]
        cross_has_mass = cross_has_mass & pixel_mask[None, :]
    ctf_probs_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance[None, :], 0.0)
    a2_terms = jnp.where(ctf_has_mass, proj_abs2 * ctf_probs_raw, 0.0)
    cross_terms = jnp.where(cross_has_mass, proj * jnp.conj(summed_masked), 0.0)
    xa_terms = noise_variance[None, :] * cross_terms.real
    return jnp.sum(a2_terms, axis=1), jnp.sum(xa_terms, axis=1)


@partial(jax.jit, static_argnames=("config",))
def _accumulate_chunk_statistics_jit(
    stats: ResidentStatistics,
    operands: ChunkStatisticsOperands,
    tables: ResidentStatisticsTables,
    *,
    config: ResidentStatisticsConfig,
) -> ResidentStatistics:
    """The single traced program; see :func:`accumulate_chunk_statistics`."""

    n_fine_trans = int(config.n_fine_trans)
    n_shells = int(config.n_shells)
    image_capacity = int(operands.image_ids.shape[0])

    probs = operands.row_posterior
    row_image = jnp.asarray(operands.row_image_local, dtype=jnp.int32)
    image_ids = jnp.asarray(operands.image_ids, dtype=jnp.int32)
    valid_image = image_ids >= 0
    image_slot = _drop_index(image_ids, int(config.n_images))

    # --- 1. sigma2 offset -------------------------------------------------
    # Host: translation_posterior = sum_R noise_probs (in the posterior dtype),
    # cast to float64, then dot with the per-image squared translation
    # distances. The per-image intermediate is kept so only the sum *across*
    # images changes order.
    sigma2_offset = stats.sigma2_offset
    translation_posterior = segment_sum_by_image(probs, row_image, image_capacity)
    if tables.translation_sqdist_ang is not None:
        sqdist = jnp.asarray(tables.translation_sqdist_ang, dtype=jnp.float64)
        sigma2_offset = sigma2_offset + jnp.sum(
            translation_posterior.astype(jnp.float64) * sqdist
        )

    # --- 2. support mass --------------------------------------------------
    support_mass = jnp.sum(translation_posterior, axis=1)
    support_mass = jnp.where(valid_image, support_mass, jnp.zeros((), support_mass.dtype))
    sumw = stats.sumw + jnp.sum(support_mass.astype(jnp.float64))

    # --- 3. weighted image power shells and per-image norm power ----------
    # ``valid_image_mask`` is the padding-aware form of the host's implicit
    # ones vector: identical arithmetic on real images, zero on padded slots
    # (which is what keeps the unweighted high shells padding-free).
    weighted_img_shells, weighted_img_per_image = _weighted_image_power_shells_and_per_image_core(
        operands.processed_image_half,
        tables.shell_indices_half,
        support_mass,
        operands.relion_norm_high_shell,
        valid_image,
        shell_count=n_shells,
        norm_unweighted_shell_cutoff=config.norm_unweighted_shell_cutoff,
        include_unweighted_high_shell=config.include_unweighted_high_shell,
        disable_cuda_binning=config.disable_cuda_binning,
        deterministic_norm_reduction=config.deterministic_norm_reduction,
    )

    # --- 4. per-particle norm correction (production algebraic mode) ------
    # The host adds the image-power term and the A2-2XA residual in two
    # separate ``+=`` statements; keep both scatters separate so the float64
    # rounding of the accumulator matches.
    norm_correction = stats.norm_correction.at[image_slot].add(
        jnp.where(valid_image, weighted_img_per_image, 0.0).astype(jnp.float64),
        mode="drop",
    )

    # --- 5. block noise shells -------------------------------------------
    block_noise_shells, _, _ = compute_noise_block(
        operands.proj,
        operands.proj_abs2,
        operands.summed_masked,
        operands.ctf_probs,
        tables.noise_variance,
        tables.shell_indices_noise,
        n_shells,
        return_split=False,
    )

    # --- 6. RELION direct-Wavg low-shell replacement ----------------------
    if config.relion_wavg_atomic_direct_noise:
        if operands.wavg_triplet_pixels is None:
            raise ValueError("direct Wavg noise replacement requires the Wavg triplet pixels")
        residual_shells, image_power_shells = (
            _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
                block_noise_shells.astype(jnp.float64),
                weighted_img_shells.astype(jnp.float64),
                operands.wavg_triplet_pixels[:, :, 2],
                tables.wavg_shell_indices,
                exclusive_shell_stop=int(config.direct_noise_exclusive_shell_stop),
                shell_count=n_shells,
            )
        )
    else:
        residual_shells = block_noise_shells.astype(jnp.float64)
        image_power_shells = weighted_img_shells.astype(jnp.float64)
    wsum_sigma2_noise = stats.wsum_sigma2_noise + residual_shells
    wsum_img_power = stats.wsum_img_power + image_power_shells

    # --- 7. per-image norm residual --------------------------------------
    a2_per_row, xa_per_row = _flat_row_norm_and_scale_terms(
        operands.proj,
        operands.proj_abs2,
        operands.summed_masked,
        operands.ctf_probs,
        tables.noise_variance,
    )
    a2_per_image = add_segment_sum(
        jnp.zeros(image_capacity, dtype=a2_per_row.dtype), row_image, a2_per_row
    )
    xa_per_image = add_segment_sum(
        jnp.zeros(image_capacity, dtype=xa_per_row.dtype), row_image, xa_per_row
    )
    block_norm_residual = a2_per_image - 2.0 * xa_per_image
    norm_correction = norm_correction.at[image_slot].add(
        jnp.where(valid_image, block_norm_residual, 0.0).astype(jnp.float64),
        mode="drop",
    )

    # --- 8. group scale sufficient statistics -----------------------------
    scale_xa = stats.scale_xa
    scale_aa = stats.scale_aa
    if config.accumulate_scale:
        if config.relion_wavg_atomic_scale_aa:
            # RELION's atomic Wavg stream already carries XA/AA in scale units;
            # the host masks the rectangle and sums in float64.
            if operands.wavg_triplet_pixels is None:
                raise ValueError("the atomic Wavg scale branch requires the Wavg triplet pixels")
            mask_rect = jnp.asarray(tables.wavg_scale_pixel_mask, dtype=bool).reshape(1, -1)
            zero_f32 = jnp.float32(0.0)
            scale_xa_per_image = jnp.sum(
                jnp.where(mask_rect, operands.wavg_triplet_pixels[:, :, 0], zero_f32).astype(
                    jnp.float64
                ),
                axis=1,
            )
            scale_aa_per_image = jnp.sum(
                jnp.where(mask_rect, operands.wavg_triplet_pixels[:, :, 1], zero_f32).astype(
                    jnp.float64
                ),
                axis=1,
            )
        else:
            scale_a2_row, scale_xa_row = _flat_row_norm_and_scale_terms(
                operands.proj,
                operands.proj_abs2,
                operands.summed_masked,
                operands.ctf_probs,
                tables.noise_variance,
                pixel_mask=tables.scale_pixel_mask,
            )
            scale_a2_image = add_segment_sum(
                jnp.zeros(image_capacity, dtype=scale_a2_row.dtype), row_image, scale_a2_row
            )
            scale_xa_image = add_segment_sum(
                jnp.zeros(image_capacity, dtype=scale_xa_row.dtype), row_image, scale_xa_row
            )
            safe_scale = jnp.maximum(
                jnp.asarray(operands.scale, dtype=scale_a2_image.dtype),
                jnp.asarray(1e-30, dtype=scale_a2_image.dtype),
            )
            scale_xa_per_image = scale_xa_image / safe_scale
            scale_aa_per_image = scale_a2_image / (safe_scale**2)
        group_slot = _drop_index(operands.group_ids, int(config.n_scale_groups))
        keep_group = valid_image & (jnp.asarray(operands.group_ids, dtype=jnp.int32) >= 0)
        scale_xa = scale_xa.at[group_slot].add(
            jnp.where(keep_group, scale_xa_per_image, 0.0).astype(jnp.float64), mode="drop"
        )
        scale_aa = scale_aa.at[group_slot].add(
            jnp.where(keep_group, scale_aa_per_image, 0.0).astype(jnp.float64), mode="drop"
        )

    # --- 9. rotation posterior sums ---------------------------------------
    probs_sum_t = jnp.sum(probs, axis=-1)
    coarse_slot = _drop_index(operands.row_coarse_rot, int(config.n_coarse_rot))
    rotation_posterior_sums = stats.rotation_posterior_sums.at[coarse_slot].add(
        probs_sum_t.astype(jnp.float64), mode="drop"
    )

    # --- 10. per-image score and pose fields ------------------------------
    if config.use_exact_relion_gaussian:
        # ``_relion_cuda_fine_log_evidence_offset`` is exactly ``-min_diff2``.
        log_score_offset = (-jnp.asarray(operands.min_diff2)).astype(jnp.float64)
    else:
        if operands.batch_norm is None:
            raise ValueError("the non-exact-Gaussian branch requires batch_norm")
        log_score_offset = -0.5 * jnp.squeeze(operands.batch_norm, axis=1).astype(jnp.float64)
    class_log_z = jnp.asarray(operands.class_log_z, dtype=jnp.float64)
    best_log_score_chunk = jnp.asarray(operands.best_log_score, dtype=jnp.float64)
    finite = jnp.isfinite(best_log_score_chunk)
    neg_inf = jnp.asarray(-jnp.inf, dtype=jnp.float64)

    log_evidence_values = jnp.where(finite, class_log_z + log_score_offset, neg_inf)
    score_log_z_values = jnp.where(
        finite,
        class_log_z + log_score_offset if config.use_exact_relion_gaussian else class_log_z,
        neg_inf,
    )
    log_evidence = stats.log_evidence.at[image_slot].set(log_evidence_values, mode="drop")
    score_log_z = stats.score_log_z.at[image_slot].set(score_log_z_values, mode="drop")
    best_log_score = stats.best_log_score.at[image_slot].set(
        best_log_score_chunk + log_score_offset, mode="drop"
    )
    max_posterior = stats.max_posterior.at[image_slot].set(
        jnp.asarray(operands.max_posterior, dtype=stats.max_posterior.dtype), mode="drop"
    )

    best_row_local = jnp.asarray(operands.best_row_local, dtype=jnp.int32)
    best_translation = jnp.asarray(operands.best_translation, dtype=jnp.int32)
    local_rot = best_row_local - jnp.asarray(operands.image_row_start, dtype=jnp.int32)
    # The host raises when the winner points into a bucket's rotation padding;
    # a traced program counts instead and finalize_statistics raises.
    row_count = jnp.asarray(operands.image_row_count, dtype=jnp.int32)
    out_of_range = valid_image & ((local_rot < 0) | (local_rot >= row_count))
    invalid_best_rows = stats.invalid_best_rows + jnp.sum(out_of_range.astype(jnp.int64))

    safe_row = jnp.clip(best_row_local, 0, max(int(operands.row_fine_rot.shape[0]) - 1, 0))
    best_fine_rot = jnp.asarray(operands.row_fine_rot, dtype=jnp.int64)[safe_row]
    best_cell_values = best_fine_rot * jnp.int64(n_fine_trans) + best_translation.astype(jnp.int64)
    best_cell = stats.best_cell.at[image_slot].set(best_cell_values, mode="drop")
    best_local_rot = stats.best_local_rot.at[image_slot].set(local_rot, mode="drop")

    return ResidentStatistics(
        wsum_sigma2_noise=wsum_sigma2_noise,
        wsum_img_power=wsum_img_power,
        sigma2_offset=sigma2_offset,
        sumw=sumw,
        norm_correction=norm_correction,
        scale_xa=scale_xa,
        scale_aa=scale_aa,
        rotation_posterior_sums=rotation_posterior_sums,
        log_evidence=log_evidence,
        best_log_score=best_log_score,
        max_posterior=max_posterior,
        best_cell=best_cell,
        score_log_z=score_log_z,
        best_local_rot=best_local_rot,
        invalid_best_rows=invalid_best_rows,
    )


def accumulate_chunk_statistics(
    stats: ResidentStatistics,
    operands: ChunkStatisticsOperands,
    tables: ResidentStatisticsTables,
    *,
    config: ResidentStatisticsConfig,
) -> ResidentStatistics:
    """Fold one capacity chunk into the device accumulators.

    One ``jax.jit`` program per capacity class: ``config`` and the operand
    shapes are static, ``n_valid_rows``/``n_valid_images`` never appear because
    padding is already neutral by construction (zero posterior, zero CTF mass,
    sentinel ids). Nothing is pulled to the host.
    """

    if not isinstance(config, ResidentStatisticsConfig):
        raise TypeError(f"config must be a ResidentStatisticsConfig, got {type(config)!r}")
    if operands.row_posterior.shape[1] != int(config.n_fine_trans):
        raise ValueError(
            "row posterior translation axis does not match the configured fine translation count: "
            f"{operands.row_posterior.shape[1]} vs {config.n_fine_trans}"
        )
    if operands.row_image_local.shape != operands.row_fine_rot.shape:
        raise ValueError("row_image_local and row_fine_rot must share the row capacity")
    return _accumulate_chunk_statistics_jit(stats, operands, tables, config=config)


class FinalizedStatistics(NamedTuple):
    """Host arrays for ``make_noise_stats``/``make_relion_stats`` and the poses.

    The first seven fields are the keyword arguments of
    ``recovar.em.helpers.stats.make_noise_stats``; the next four are
    ``make_relion_stats``'s. ``hard_assignment`` counts in image-local rotation
    slots exactly like the host tail's, while ``best_fine_rotation_indices``
    are the global fine-grid ids the host reads out of
    ``per_image_inputs["oversampled_rot_indices"]``.
    """

    wsum_sigma2_noise: np.ndarray
    wsum_img_power: np.ndarray
    wsum_sigma2_offset: float
    sumw: float
    wsum_norm_correction: np.ndarray
    wsum_scale_correction_xa: np.ndarray | None
    wsum_scale_correction_aa: np.ndarray | None
    log_evidence_per_image: np.ndarray
    best_log_score_per_image: np.ndarray
    max_posterior_per_image: np.ndarray
    rotation_posterior_sums: np.ndarray
    score_log_z_per_image: np.ndarray
    hard_assignment: np.ndarray
    best_fine_rotation_indices: np.ndarray
    best_translation_indices: np.ndarray


def finalize_statistics(
    stats: ResidentStatistics,
    *,
    config: ResidentStatisticsConfig,
    require_all_images: bool = True,
) -> FinalizedStatistics:
    """Pull the accumulators once and decode the pose fields.

    Raises the padding-sanity error the host tail raises inline (a winner that
    points outside its image's candidate rows), and, unless
    ``require_all_images`` is cleared, the missing-image error that a chunk
    plan which does not cover every image would otherwise hide.
    """

    invalid = int(np.asarray(jax.device_get(stats.invalid_best_rows)))
    if invalid:
        raise RuntimeError(
            f"Resident pass-2 statistics: {invalid} image(s) selected a candidate row outside "
            "their own row range; the chunk's best-row operand points into padding"
        )
    n_fine_trans = int(config.n_fine_trans)
    (
        wsum_sigma2_noise,
        wsum_img_power,
        sigma2_offset,
        sumw,
        norm_correction,
        scale_xa,
        scale_aa,
        rotation_posterior_sums,
        log_evidence,
        best_log_score,
        max_posterior,
        best_cell,
        score_log_z,
        best_local_rot,
        _,
    ) = jax.device_get(tuple(stats))

    best_cell = np.asarray(best_cell, dtype=np.int64)
    best_local_rot = np.asarray(best_local_rot, dtype=np.int64)
    if require_all_images and np.any(best_cell < 0):
        missing = int(np.count_nonzero(best_cell < 0))
        raise RuntimeError(
            f"Resident pass-2 statistics: {missing} image(s) were never written by a chunk"
        )
    best_translation_indices = np.where(best_cell < 0, -1, best_cell % n_fine_trans)
    best_fine_rotation_indices = np.where(best_cell < 0, -1, best_cell // n_fine_trans)
    hard_assignment = np.where(
        best_local_rot < 0,
        -1,
        best_local_rot * n_fine_trans + best_translation_indices,
    ).astype(np.int32)

    return FinalizedStatistics(
        wsum_sigma2_noise=np.asarray(wsum_sigma2_noise, dtype=np.float64),
        wsum_img_power=np.asarray(wsum_img_power, dtype=np.float64),
        wsum_sigma2_offset=float(sigma2_offset),
        sumw=float(sumw),
        wsum_norm_correction=np.asarray(norm_correction, dtype=np.float64),
        wsum_scale_correction_xa=(
            np.asarray(scale_xa, dtype=np.float64) if config.accumulate_scale else None
        ),
        wsum_scale_correction_aa=(
            np.asarray(scale_aa, dtype=np.float64) if config.accumulate_scale else None
        ),
        log_evidence_per_image=np.asarray(log_evidence, dtype=np.float64),
        best_log_score_per_image=np.asarray(best_log_score, dtype=np.float64),
        max_posterior_per_image=np.asarray(max_posterior),
        rotation_posterior_sums=np.asarray(rotation_posterior_sums, dtype=np.float64),
        score_log_z_per_image=np.asarray(score_log_z, dtype=np.float64),
        hard_assignment=hard_assignment,
        best_fine_rotation_indices=best_fine_rotation_indices.astype(np.int64),
        best_translation_indices=best_translation_indices.astype(np.int64),
    )
