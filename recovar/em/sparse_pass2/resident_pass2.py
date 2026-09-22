"""Device-resident K=1 sparse pass-2 driver (T9b integration).

This module wires the five stage modules of
``em_device_resident_pass2_design_20260918.md`` into one driver with the
signature and return type of
:func:`recovar.em.sparse_pass2.sparse_pass2_bucketed.compute_pass2_stats_sparse_bucketed`:

* T5 :mod:`recovar.em.sparse_pass2.resident_candidates` -- the flat image-CSR
  candidate table and its fixed-capacity chunks;
* T6 :mod:`recovar.em.sparse_pass2.resident_scoring` -- the resident per-image
  scoring operands and the one-program-per-capacity-class scoring stage;
* T7 ``recovar.cuda_backproject.sparse_pass2_segmented_*`` -- the segmented
  RELION float32 fine posterior over those CSR segments;
* T8 ``recovar.cuda_backproject.relion_wavg_*flat_rows*`` -- the flat-row RELION
  Wavg triplet and its atomic accumulation;
* T9a :mod:`recovar.em.sparse_pass2.resident_statistics` -- the device float64
  accumulators, their finalization and the pose decode.

Selection and scope
-------------------
The driver is selected only by ``RECOVAR_SPARSE_PASS2_RESIDENT=1`` and only in
the production K=1 configuration (RELION x-half M-step, exact RELION fine
Gaussian scoring, float32 fine posterior, fine M-step prune, float32 scoring,
no diagnostics or dumps). Every other configuration raises
:class:`NotImplementedError` naming the missing piece; the driver never falls
back to the compact engine silently, because a silent fallback would make a
measured comparison meaningless.

What is deliberately different from the compact engine
------------------------------------------------------
Three differences are layout or reduction-order changes, not arithmetic
changes, and each is measured rather than assumed:

1. **Translation application.** The compact K=1 route scores a pre-shifted
   ``(B, T, N)`` image tile; this driver calls the flat-row *fused translate*
   kernel, which applies the translation phase per pixel inside the scoring
   kernel. T6 measured the two to agree bitwise on every score-window pixel and
   to differ only on the ``ky = -N/2`` Nyquist row of a full (unwindowed) half
   image. The production window excludes that row, and the configuration gate
   below refuses the unwindowed case.
2. **log-Z reduction order.** The compact route reduces ``sum exp`` with an XLA
   float64 tree over ``(B, R*T)``; the segmented CUDA handler reduces per
   segment in block order. The posteriors themselves come from the float32
   ``sum_weight`` scan, which T7 showed is bitwise identical to the rectangular
   handler, so only ``log_evidence``/``score_log_z`` can move.
3. **Statistics reduction order.** Chunk partials replace per-bucket host sums.
   The user waived reduction-order parity for these accumulators on 2026-09-18;
   the same-source band is the gate.

Pixel-axis blocking
-------------------
Stages 5-7 carry a pixel axis (``N_recon``, 4324 at the hp3 production state),
so a whole chunk's ``proj``/``summed``/``ctf_probs`` would be tens of gigabytes
at the largest capacity class. The design anticipates this ("project in row
blocks inside the chunk program"). The driver therefore walks each chunk in
fixed-size row blocks, so the pixel-axis programs are keyed on one static block
shape regardless of the chunk's capacity class. Because of that split, the
image-level statistics are folded once per chunk by
:func:`_accumulate_chunk_image_terms` (T9a's arithmetic, with the row-pixel
reductions arriving as chunk partials) rather than by T9a's single fused
program, which assumes one call per chunk with the whole pixel axis resident.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.batch_fetch import fetch_indexed_batch
from recovar.em.helpers.deterministic_reduce import deterministic_reductions_enabled
from recovar.em.helpers.env_flags import parse_env_capacity_ladder, parse_env_flag
from recovar.em.helpers.half_spectrum import (
    make_relion_noise_shell_indices_half,
    mask_relion_noise_shell_indices_to_current_window,
)
from recovar.em.helpers.half_volume_mstep import (
    enforce_half_volume_x0,
    half_volume_accumulator_shape,
    relion_backprojector_volume_shape,
    relion_x_half_accumulators_to_public_layout,
    relion_x_half_mstep_accumulator_dtypes,
)
from recovar.em.helpers.preprocessing import half_translation_phase_table
from recovar.em.helpers.projection import compute_noise_block
from recovar.em.helpers.projection import (
    relion_scale_correction_pixel_mask as _relion_scale_correction_pixel_mask,
)
from recovar.em.helpers.scale_groups import prepare_scale_correction_groups
from recovar.em.helpers.translation_prior import (
    translation_prior_centers_for_images,
    translation_sqdist_angstrom,
    validate_translation_prior_centers,
)
from recovar.em.helpers.types import SparsePass2Output, make_noise_stats, make_relion_stats
from recovar.em.local.local_backprojection import (
    compute_local_ctf_sums_from_probs_sum_t,
    compute_local_weighted_sums,
)
from recovar.em.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs
from recovar.em.sparse_pass2.compile_ahead import (
    CompileAheadPool,
    resolve_compile_ahead_config,
)
from recovar.em.sparse_pass2.resident_candidates import (
    materialize_chunk,
    plan_capacity_chunks,
)
from recovar.em.sparse_pass2.resident_operands import (
    ResidentOperandsUnsupported,
    describe_resident_operand_mismatch,
    gather_resident_chunk_operands,
    prepare_resident_half_operands,
    resident_half_operand_avals,
    resident_half_operand_bytes,
    resident_half_operand_presence,
    resident_operands_max_bytes,
)
from recovar.em.sparse_pass2.resident_scoring import score_resident_chunk
from recovar.em.sparse_pass2.resident_significance import (
    resident_candidate_tables,
    resident_significance_csr,
)
from recovar.em.sparse_pass2.resident_statistics import (
    ResidentStatistics,
    _drop_index,
    _flat_row_norm_and_scale_terms,
    finalize_statistics,
    make_resident_statistics,
    resolve_statistics_config,
    segment_sum_by_image,
)
from recovar.em.sparse_pass2.sparse_pass2_adjoint import _accumulate_adjoint_block_chunked
from recovar.em.sparse_pass2.sparse_pass2_bucket_io import (
    _prepare_bucket_io,
    _relion_cuda_score_translation_angles_if_available,
)
from recovar.em.sparse_pass2.sparse_pass2_budget import (
    _max_adjoint_block_bytes_for_pass,
    _max_translation_tile_bytes_for_pass,
    _projection_cache_build_max_rotations_per_call,
    _projection_cache_fits_budget,
    _projection_cache_max_bytes_for_pass,
    _projection_cache_transient_bytes,
)
from recovar.em.sparse_pass2.sparse_pass2_policy import (
    _RELION_WAVG_ATOMIC_SCALE_AA_ENV,
    _fresh_k1_direct_noise_default,
    _projection_cache_enabled_for_pass,
    _relion_exact_bpref_operands_enabled,
    _relion_powerclass_spectrum_norm_enabled,
    _relion_wavg_direct_modes,
)
from recovar.em.sparse_pass2.sparse_pass2_posterior import (
    _relion_fine_parent_execution_order_enabled,
)
from recovar.em.sparse_pass2.sparse_pass2_projection_blocks import (
    _compute_sparse_pass2_windowed_projections_block,
    _projection_kwargs_for_relion_score_window,
)
from recovar.em.sparse_pass2.sparse_pass2_scoring import (
    _relion_cuda_fine_full_to_compact_lookup,
    _relion_powerclass_noise_terms,
    relion_powerclass_noise_dtypes,
)
from recovar.em.sparse_pass2.sparse_pass2_wavg import (
    _make_relion_wavg_rectangle,
    _relion_cuda_translate_wavg_norm_images,
    _relion_wavg_rectangle_image_power,
    _relion_wavg_rectangle_power_contraction,
    _relion_wavg_shifted_power,
    _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp,
    _weighted_image_power_shells_and_per_image_core,
)
from recovar.em.sparse_pass2.sparse_pass2_window import (
    _pass2_half_weights,
    _pass2_window_setup,
    _sparse_pass2_window_setup,
)
from recovar.reconstruction import noise as noise_utils

logger = logging.getLogger(__name__)

RESIDENT_PASS2_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT"
_ROW_CAPACITY_LADDER_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_ROW_CAPACITIES"
_IMAGE_CAPACITY_LADDER_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_IMAGE_CAPACITIES"
_MSTEP_BLOCK_ROWS_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_MSTEP_BLOCK_ROWS"
# Attribution only, default off. Logs one line per chunk with its occupancy,
# its M-step block count and a device-synchronised wall, and counts the T7
# offsets readbacks. The synchronisation perturbs the wall, so an arm with
# this set is a diagnostic arm and never a timing arm.
_CHUNK_TIMING_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_CHUNK_TIMING"
# T14: the chunk body as one jitted program per capacity class. Opt-in; the
# per-stage path is the default and the oracle both paths are compared against. ``..._CHUNK_STATIC_BLOCKS`` runs the M-step block loop over
# the whole row capacity instead of the chunk's live blocks; both forms trace
# one program per capacity class and are bitwise equal, because a padded block
# carries a zero posterior and contributes exact zeros.
_CHUNK_JIT_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_CHUNK_JIT"
_CHUNK_STATIC_BLOCKS_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_CHUNK_STATIC_BLOCKS"
# How many M-step blocks the chunk program emits per device-loop iteration.
# XLA:GPU reads a while predicate back to the host once per iteration, so an
# unroll of u divides those readbacks by u; it also multiplies the live
# pixel-axis transients by u, because the emitted copies no longer reuse one
# block's buffers. Emitting every block (no loop) is not an option: at the early
# state's second iteration that program asked the allocator for 54.5 GiB, and
# T16's branch point 6c2dad33e carried that fully unrolled form, which OOMs at
# hp3 (181 GiB); 7a29c2776's bounded unroll supersedes it and is kept here.
_CHUNK_BLOCK_UNROLL_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_CHUNK_BLOCK_UNROLL"
# T16: prepare the per-image operands once per half and keep them resident, and
# take the M-step's weighted sums with T15's flat-row translate-and-sum kernel
# instead of a gathered ``[images, translations, pixels]`` tile. Default on;
# ``RECOVAR_SPARSE_PASS2_RESIDENT_OPERANDS=0`` selects the per-chunk
# ``_prepare_bucket_io`` preparation and the XLA tile reduction, which stay as
# the oracle both forms are compared against.
_RESIDENT_OPERANDS_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_OPERANDS"
# Diagnostic, default off. For the first chunk of a half it also runs the
# per-chunk preparation and checks, on that chunk's real operands, that
# translating the resident per-image arrays reproduces the pre-shifted tiles
# bitwise and that the kernel's weighted sums equal the XLA reduction. It
# doubles that chunk's preparation cost, so an arm with it set is a diagnostic
# arm, never a timing arm.
_RESIDENT_OPERANDS_VERIFY_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_OPERANDS_VERIFY"
# Take ``ctf_probs`` from the translate-and-sum kernel's fourth output instead
# of the XLA statement. Measurement only: see
# ``_resident_block_weighted_sums_kernel`` for why it is not the default.
_KERNEL_CTF_PROBS_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_KERNEL_CTF_PROBS"
# P4-G phase 2: square the chunk's Wavg rectangle once per image and gather the
# float32 result, instead of gathering the complex rectangle to the block's rows
# and squaring once per row. Squaring is elementwise, so squaring then gathering
# and gathering then squaring are the same float32 values, and the contraction
# that follows keeps its shapes and its reduction order; the two settings are
# bitwise. Default off while the measurement arms are the ones in the ticket's
# report; `_resident_block_wavg_rectangle_terms` holds both paths.
_WAVG_POWER_PER_IMAGE_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_WAVG_POWER_PER_IMAGE"
# P3-A: dispatch the per-stage chunk loop's three stages as jitted programs
# keyed on the capacity class instead of as loose eager operations. Default on.
# ``RECOVAR_SPARSE_PASS2_RESIDENT_GLUE_JIT=0`` restores the loose dispatch,
# which stays the oracle every bitwise comparison of this change is made
# against. The stage bodies are the same functions in both settings, so the
# flag changes only where the JIT boundary sits.
_RESIDENT_GLUE_JIT_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_GLUE_JIT"
# Diagnostic, default off. Checks the statically computed M-step carry avals
# against a ``jax.eval_shape`` probe of the same block stages, once per
# capacity class. The probes are what this change removes from the chunk loop;
# the flag exists so a test, or a suspicious run, can prove the arithmetic
# still agrees with them.
_CARRY_AVAL_PROBE_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_CARRY_AVAL_PROBE"
# Relative band the racing shell-binning scatter is allowed in the verification
# arm: 12x the measured same-call spread of 5.8e-8, still far inside one
# float32 ulp of the accumulated shell power.
_RACING_SCATTER_RELATIVE_BAND = 7e-7
_SOFT_POSTERIOR_BLOCK_BPREF_PROTOTYPE_ENV = "RECOVAR_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF"

# Row capacities are multiples of the M-step block so every chunk decomposes
# into whole blocks; image capacities follow the design's ladder. The design's
# three classes, restored after the denser five-class ladder was measured and
# bought nothing: at hp3 the matched pairs put the two ladders inside the
# control's own drift (loop 22.6 versus 22.1 s per half) and at the early state
# they are indistinguishable (resident warm 67.6 / 60.4 versus 67.0 / 61.5 s,
# occupancy 0.95-0.97 either way, jobs 14143902 and 14143904), while the dense
# ladder costs 84 extra traced programs. Every extra class is one more program
# per capacity-class stage, and with the chunk program one more program again.
# The occupancy of each plan is still logged, so a future change has its number.
_DEFAULT_ROW_CAPACITY_LADDER = (8192, 32768, 131072)
_DEFAULT_IMAGE_CAPACITY_LADDER = (32, 128, 512)

__all__ = [
    "RESIDENT_PASS2_ENV",
    "ResidentPass2Plan",
    "compute_pass2_stats_resident",
    "resident_pass2_requested",
    "require_resident_production_configuration",
    "resident_pass2_out_of_scope_reason",
]


def resident_pass2_requested() -> bool:
    """Return whether ``RECOVAR_SPARSE_PASS2_RESIDENT`` selects this driver."""

    return parse_env_flag(RESIDENT_PASS2_ENV, default=False)


# ---------------------------------------------------------------------------
# Production-configuration gate
# ---------------------------------------------------------------------------

_DIAGNOSTIC_DIR_ENVS = (
    "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR",
    "RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR",
    "RECOVAR_BPREF_MEMBERSHIP_DUMP_DIR",
    "RECOVAR_PASS2_DUMP_DIR",
    "RECOVAR_BPREF_EXECUTION_ORDER_LOCAL_FILE",
    "RECOVAR_VDAM_KCLASS_STATS_DUMP_DIR",
)

_DIAGNOSTIC_FLAG_ENVS = (
    "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS",
    "RECOVAR_K1_RELION_TRANSLATED_WAVG_NORM",
    "RECOVAR_SPARSE_PASS2_LOG_CANDIDATE_DENSITY",
    "RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION",
    "RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH",
    "RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(
            "The device-resident K=1 sparse pass 2 "
            f"({RESIDENT_PASS2_ENV}=1) does not implement this configuration: {message}. "
            "Clear the flag to use the compact engine; this path never falls back silently."
        )


def resident_pass2_out_of_scope_reason(
    *, relion_firstiter_score_mode, relion_firstiter_winner_take_all, symmetry_label="C1"
) -> str | None:
    """Name the scoring modes the resident driver was never scoped to cover.

    These are not configuration drift inside the covered path, so they are not
    a reason to stop a run: RELION's ``--firstiter_cc`` iteration scores with
    normalized cross-correlation and takes the winner outright, which is a
    different pass-2 route with its own kernels. The caller sends those to the
    compact engine and says so. Everything else still raises through
    :func:`require_resident_production_configuration`, because a silent
    fallback there would hide a real mismatch.
    """

    if symmetry_label != "C1":
        return f"{symmetry_label} point-group reconstruction symmetry"
    if relion_firstiter_score_mode != "gaussian":
        return (
            "RELION normalized-CC scoring "
            f"(relion_firstiter_score_mode={relion_firstiter_score_mode!r})"
        )
    if relion_firstiter_winner_take_all:
        return "RELION --firstiter_cc winner-take-all posteriors"
    return None


def require_resident_production_configuration(**kwargs) -> None:
    """Raise :class:`NotImplementedError` unless this is the production path.

    Every check names the specific missing piece rather than reporting a
    generic refusal, so a caller that trips one knows which behaviour the
    resident driver would have to grow.
    """

    _require(bool(kwargs["relion_x_half_mstep"]), "the RELION x-half M-step is required")
    # The dispatcher opens a persistent texture only for the compact engine and
    # routes non-C1 symmetry there (resident_pass2_out_of_scope_reason); a
    # direct caller that supplies either gets a named refusal, not a drop.
    _require(
        kwargs["relion_projector_texture"] is None,
        "a persistent RELION projector texture belongs to the compact engine; "
        "the resident driver projects from relion_projector_half",
    )
    _require(
        kwargs["symmetry_label"] == "C1",
        f"{kwargs['symmetry_label']} point-group reconstruction symmetry is not implemented",
    )
    _require(
        bool(kwargs["relion_exact_fine_gaussian"])
        and kwargs["relion_firstiter_score_mode"] == "gaussian",
        "exact RELION fine Gaussian scoring is required "
        f"(got relion_exact_fine_gaussian={kwargs['relion_exact_fine_gaussian']!r}, "
        f"score_mode={kwargs['relion_firstiter_score_mode']!r})",
    )
    _require(not bool(kwargs["use_float64_scoring"]), "float64 scoring is a diagnostic mode")
    _require(
        not bool(kwargs["relion_firstiter_winner_take_all"]),
        "winner-take-all (--firstiter_cc) disables the float32 fine posterior",
    )
    _require(
        not (bool(kwargs["disable_adjoint_y"]) or bool(kwargs["disable_adjoint_ctf"])),
        "score-only passes have no M-step to make resident",
    )
    _require(not bool(kwargs["return_score_log_z_only"]), "score-logZ-only passes are score-only")
    _require(bool(kwargs["accumulate_noise"]), "the production pass accumulates noise statistics")
    _require(
        not bool(kwargs["mstep_subtract_ctf_projection"]),
        "subtracting the projected reference in the M-step is a diagnostic mode",
    )
    # The K=1 adaptive route always hands the M-step call an
    # ``normalization_other_score_log_z`` built from the *other* classes'
    # log-Z (k_class.py::_run_sparse_k_class_adaptive_pass2). At K=1 there are
    # no other classes, so that vector is all -inf and the compact engine's
    # own arithmetic collapses to its unnormalized branch: logaddexp(x, -inf)
    # is x, the reported log-evidence and score log-Z are taken from
    # ``local_score_log_z`` rather than from the combined value, and the
    # float32 reconstruction weights never read it at all. Accept exactly that
    # degenerate vector, which is the production K=1 case, and refuse any
    # finite entry, which would genuinely mix classes.
    other_log_z = kwargs["normalization_other_score_log_z"]
    other_log_z_is_degenerate = other_log_z is not None and bool(
        np.all(np.asarray(other_log_z) == -np.inf)
    )
    _require(
        kwargs["normalization_log_z"] is None,
        "an externally supplied log-Z belongs to the K-class engine",
    )
    _require(
        other_log_z is None or other_log_z_is_degenerate,
        "a finite cross-class score normalization belongs to the K-class engine",
    )
    _require(
        kwargs["relion_f32_normalization_sum_weight"] is None
        and kwargs["relion_coarse_hard_assignment"] is None,
        "the zero-oversampling coarse-normalization reuse is not wired yet; "
        "the segmented posterior supports it but its winner/Pmax substitution is untested here",
    )
    # ``preserve_bpref_particle_order`` is the production setting, and on its
    # own it forces one BPref launch per particle. The production run pairs it
    # with the soft-posterior block prototype, which turns those launches back
    # into one block launch per bucket; that is the semantics the resident
    # driver reproduces with one launch per row block. Without the prototype
    # the compact engine really would launch per particle, so refuse.
    _require(
        (not bool(kwargs["preserve_bpref_particle_order"]))
        or bool(kwargs["soft_posterior_block_bpref"]),
        "strict per-particle BPref launches are not implemented; set "
        "RECOVAR_EM_PROTOTYPE_SOFT_POSTERIOR_BLOCK_BPREF=1 (the production "
        "setting) so BPref accumulates per block, or clear "
        "preserve_bpref_particle_order",
    )
    _require(
        kwargs["fine_rotations_override"] is not None
        and kwargs["fine_rotation_parent_override"] is not None,
        "the resident driver gathers rotations from the caller's fine grid, "
        "so fine_rotations_override and fine_rotation_parent_override are required",
    )
    _require(
        int(kwargs["n_coarse_trans"]) <= 32,
        "the candidate bitset packs coarse translations into one uint32, so "
        f"n_coarse_trans must be <= 32 (got {int(kwargs['n_coarse_trans'])})",
    )
    _require(
        bool(kwargs["use_window"]),
        "the resident driver scores through the RELION current-size window; "
        "a full-half pass would include the ky=-N/2 Nyquist row, where the "
        "fused-translate and pre-shifted scorers are known to differ",
    )
    _require(
        bool(kwargs["projection_cache_available"]),
        "the resident scoring stage gathers cached fine-rotation projections; "
        "the per-iteration projection cache is disabled or did not fit its budget",
    )
    _require(
        bool(kwargs["relion_wavg_atomic_scale_aa"]),
        "the resident statistics stage consumes the RELION atomic Wavg triplet",
    )
    _require(
        bool(kwargs["relion_wavg_atomic_direct_noise"]),
        "the resident statistics stage uses RELION's direct low-shell residual",
    )
    _require(
        not bool(kwargs["relion_wavg_atomic_direct_norm"]),
        "the direct per-particle Wavg norm arm is a stopped diagnostic",
    )
    for name in _DIAGNOSTIC_DIR_ENVS:
        _require(
            not os.environ.get(name, "").strip(),
            f"the diagnostic dump {name} is set; the resident driver emits no dumps",
        )
    for name in _DIAGNOSTIC_FLAG_ENVS:
        _require(
            not parse_env_flag(name, default=False),
            f"the diagnostic flag {name} is set; the resident driver has no such arm",
        )


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidentPass2Plan:
    """Chunk plan plus the pixel-axis block size the M-step stages run at."""

    chunks: tuple
    row_capacity_ladder: tuple
    image_capacity_ladder: tuple
    mstep_block_rows: int


def _floor_power_of_two(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value.bit_length() - 1)


def _resolve_mstep_block_rows(
    *,
    n_recon_pixels: int,
    max_block_bytes: int,
    row_capacity_ladder: tuple,
) -> int:
    """Rows per pixel-axis block: a power of two dividing every row capacity.

    A block holds, per row and reconstruction pixel, one complex64 projection,
    one float32 ``|proj|^2``, two complex64 weighted sums, one float32 CTF sum
    and the float32 Wavg triplet: 44 bytes. Sizing from the same budget the
    compact engine uses for its adjoint blocks keeps the transient comparable
    to the bucket this replaces; quantizing to a power of two keeps the traced
    pixel-axis program count at one per pixel count.
    """

    override = os.environ.get(_MSTEP_BLOCK_ROWS_ENV, "").strip()
    if override:
        block = int(override)
        if block <= 0 or block & (block - 1):
            raise ValueError(f"{_MSTEP_BLOCK_ROWS_ENV} must be a positive power of two, got {block}")
    else:
        bytes_per_row = max(int(n_recon_pixels), 1) * 44
        block = _floor_power_of_two(max(int(max_block_bytes) // bytes_per_row, 1))
    smallest = int(row_capacity_ladder[0])
    block = min(block, smallest)
    while block > 1 and smallest % block:
        block //= 2
    return max(block, 1)


def _cap_image_capacity_ladder(
    ladder: tuple,
    *,
    n_fine_trans: int,
    n_recon_pixels: int,
    max_tile_bytes: int,
) -> tuple:
    """Drop image classes whose translation tiles exceed the tile budget.

    A chunk materializes three ``(images, T, P)`` complex64 tiles: the
    reconstruction operand, the noise operand and RELION's Wavg rectangle.
    """

    per_image = max(int(n_fine_trans), 1) * max(int(n_recon_pixels), 1) * 8 * 3
    cap = max(int(max_tile_bytes) // max(per_image, 1), 1)
    kept = tuple(value for value in ladder if int(value) <= cap)
    return kept if kept else (int(ladder[0]),)


# ---------------------------------------------------------------------------
# Device programs with a pixel axis (one per (block rows, pixel count) pair)
# ---------------------------------------------------------------------------


@jax.jit
def _resident_block_weighted_sums(
    row_posterior,  # float32 [block, T]
    row_image_local,  # int32 [block]
    shifted_recon,  # complex [C_B, T, P]
    shifted_noise,  # complex [C_B, T, P]
    ctf2_over_nv_recon,  # real [C_B, P]
):
    """Flat-row twin of the bucket's ``compute_local_mstep_sums`` pair.

    ``compute_local_weighted_sums`` contracts ``(B, R, T) x (B, T, N)`` with
    ``Precision.HIGHEST``; the flat-row form gathers each row's image tile and
    contracts ``(Q, T) x (Q, T, N)`` at the same precision, so the per-cell
    products and the translation reduction order are unchanged.
    ``compute_local_ctf_sums_from_probs_sum_t`` is reused verbatim, including
    its ``!= 0`` mass predicate. ``summed`` feeds the x-half BPref numerator;
    ``summed_masked`` is the noise operand the host tail builds from
    ``shifted_noise``.
    """

    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    weights = jnp.asarray(row_posterior)
    recon_tiles = jnp.asarray(shifted_recon)[row_image_local]
    noise_tiles = jnp.asarray(shifted_noise)[row_image_local]
    # ``compute_local_weighted_sums`` is called verbatim with a singleton
    # rotation axis, so the contraction keeps its pinned
    # ``Precision.HIGHEST``; only the gathered per-row tile replaces the
    # shared per-image tile.
    summed = compute_local_weighted_sums(weights[:, None, :], recon_tiles)[:, 0, :]
    summed_masked = compute_local_weighted_sums(weights[:, None, :], noise_tiles)[:, 0, :]
    probs_sum_t = jnp.sum(weights, axis=-1)
    # The rectangular helper takes ``(B, R)`` rotation sums against one
    # ``(B, P)`` CTF row per image. In the flat-row layout each row carries its
    # own gathered CTF row, so the call is made with a singleton rotation axis;
    # the ``!= 0`` mass predicate and the product order are unchanged.
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(
        probs_sum_t[:, None],
        jnp.asarray(ctf2_over_nv_recon)[row_image_local],
    )[:, 0, :]
    return summed, summed_masked, ctf_probs, probs_sum_t


def _resident_block_weighted_sums_kernel(
    row_posterior,  # float32 [block, T]
    row_image_ids,  # int32 [block], -1 on a padded row
    row_image_local,  # int32 [block], the chunk-local slot the XLA gather uses
    recon_image,  # complex64 [C_B, P], unshifted
    recon_weight,  # float32 [C_B, P] (BPref weighted CTF) or None
    noise_image,  # complex64 [C_B, P], unshifted
    ctf2_over_nv_recon,  # float32 [C_B, P]
    recon_pixel_indices,  # int32 [P], centered packed-half indices
    translation_angles,  # float32 [T, 2]
    *,
    image_shape,
    n_recon_pixels: int,
    kernel_ctf_probs: bool,
    cuda_backproject,
):
    """T15's translate-and-sum kernel in place of the gathered-tile reduction.

    Same four outputs as :func:`_resident_block_weighted_sums`, from the
    *unshifted* per-image operands: the kernel applies each translation inside
    the reduction with the phase arithmetic of the primitive that built the
    tile, so no ``[images, translations, pixels]`` tile exists. ``recon_weight``
    selects the convention pairing production uses -- BPref for the
    reconstruction operand, score for the noise operand -- and matches how
    ``_prepare_bucket_io`` builds the two shifted arrays.

    Rows are bounded by their image id rather than by a row count: a padded row
    carries ``-1`` and the kernel writes it as zeros, which is the value the XLA
    path reaches through a zero posterior.

    ``ctf_probs``. The kernel can also produce it, from its own sequential
    ``probs_sum_t``. That mass is bitwise against ``jnp.sum`` only where XLA
    happens to reduce the translations sequentially too: it does at
    ``[2048, 21]``, and it does not at ``[64, 21]``, where the fourth output
    moves 53% of the cells by up to 1 relative ulp. ``ctf_probs`` feeds the
    ``Ft_ctf`` accumulator and the Wavg and noise terms, so the default keeps
    the XLA statement the per-chunk path used, on the same gathered CTF row;
    the fused form stays selectable for measurement, and the kernel's own mass
    is never used for anything else.
    """

    row_posterior = jnp.asarray(row_posterior, dtype=jnp.float32)
    block_rows = int(row_posterior.shape[0])
    outputs = cuda_backproject.relion_translate_sum_flat_rows_f32(
        jnp.asarray(recon_image, dtype=jnp.complex64),
        jnp.asarray(noise_image, dtype=jnp.complex64),
        jnp.asarray(row_image_ids, dtype=jnp.int32),
        row_posterior,
        jnp.asarray(translation_angles, dtype=jnp.float32),
        jnp.asarray(recon_pixel_indices, dtype=jnp.int32),
        jnp.asarray(block_rows, dtype=jnp.int32),
        jnp.asarray(int(n_recon_pixels), dtype=jnp.int32),
        recon_weight=(
            None if recon_weight is None else jnp.asarray(recon_weight, dtype=jnp.float32)
        ),
        ctf2_over_nv=(
            jnp.asarray(ctf2_over_nv_recon, dtype=jnp.float32) if kernel_ctf_probs else None
        ),
        image_shape=tuple(int(size) for size in image_shape),
    )
    if kernel_ctf_probs:
        summed, summed_masked, probs_sum_t, ctf_probs = outputs
        return summed, summed_masked, ctf_probs, probs_sum_t
    summed, summed_masked, _kernel_mass = outputs
    ctf_probs, probs_sum_t = _resident_block_ctf_probs(
        row_posterior, row_image_local, ctf2_over_nv_recon
    )
    return summed, summed_masked, ctf_probs, probs_sum_t


@jax.jit
def _resident_block_ctf_probs(row_posterior, row_image_local, ctf2_over_nv_recon):
    """``ctf_probs`` and its mass, exactly as :func:`_resident_block_weighted_sums` forms them.

    One program, so the kernel path costs one dispatch here rather than three.
    The statements, the gather and the ``!= 0`` mass predicate are the tile
    path's own, which is what makes the two paths bitwise on this output.
    """

    probs_sum_t = jnp.sum(jnp.asarray(row_posterior), axis=-1)
    ctf_probs = compute_local_ctf_sums_from_probs_sum_t(
        probs_sum_t[:, None],
        jnp.asarray(ctf2_over_nv_recon)[jnp.asarray(row_image_local, dtype=jnp.int32)],
    )[:, 0, :]
    return ctf_probs, probs_sum_t


@partial(jax.jit, static_argnames=("n_shells", "image_capacity"))
def _resident_block_noise_and_norm(
    proj,  # complex [block, P]
    proj_abs2,  # real [block, P]
    summed_masked,  # complex [block, P]
    ctf_probs,  # real [block, P]
    noise_variance,  # real [P]
    shell_indices,  # int32 [P]
    row_image_local,  # int32 [block]
    *,
    n_shells: int,
    image_capacity: int,
):
    """One row block's noise shells plus its per-image ``A2``/``XA`` partials."""

    block_noise_shells, _, _ = compute_noise_block(
        proj,
        proj_abs2,
        summed_masked,
        ctf_probs,
        noise_variance,
        shell_indices,
        int(n_shells),
        return_split=False,
    )
    a2_per_row, xa_per_row = _flat_row_norm_and_scale_terms(
        proj, proj_abs2, summed_masked, ctf_probs, jnp.asarray(noise_variance)
    )
    a2_per_image = segment_sum_by_image(a2_per_row, row_image_local, int(image_capacity))
    xa_per_image = segment_sum_by_image(xa_per_row, row_image_local, int(image_capacity))
    return block_noise_shells.astype(jnp.float64), a2_per_image, xa_per_image


@jax.jit
def _resident_block_wavg_algebraic_terms(
    proj,  # complex [block, P]
    proj_abs2,  # real [block, P]
    summed_masked,  # complex [block, P]
    ctf_probs,  # real [block, P]
    noise_variance,  # real [P]
    scale,  # real [C_B]
    raw_shifted_images,  # complex64 [C_B, T, P]
    row_posterior,  # float32 [block, T]
    row_image_local,  # int32 [block]
):
    """Flat-row twin of ``_relion_wavg_atomic_triplet_terms``.

    Used when the pass does not carry RELION's RFLOAT CTF operand, which is
    the branch the host bucket tail takes when ``direct_ctf_rfloat_recon`` is
    ``None``. Term for term it is the rectangular helper with the
    ``(image, rotation)`` axes folded into the row axis: the two ``!= 0`` mass
    predicates, the per-image scale division, the float32 casts and the
    optimization barrier between the real and imaginary image-power halves all
    stay where they are.
    """

    proj = jnp.asarray(proj, dtype=jnp.complex64)
    proj_abs2 = jnp.asarray(proj_abs2, dtype=jnp.float32)
    summed_masked = jnp.asarray(summed_masked, dtype=jnp.complex64)
    ctf_probs = jnp.asarray(ctf_probs, dtype=jnp.float32)
    noise_variance = jnp.asarray(noise_variance, dtype=jnp.float32).reshape(-1)
    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    row_scale = jnp.asarray(scale, dtype=jnp.float32).reshape(-1)[row_image_local]
    posterior = jnp.asarray(row_posterior, dtype=jnp.float32)

    ctf_has_mass = ctf_probs != 0.0
    ctf_posterior_raw = jnp.where(ctf_has_mass, ctf_probs * noise_variance[None, :], 0.0)
    aa_raw = jnp.where(ctf_has_mass, proj_abs2 * ctf_posterior_raw, 0.0).astype(jnp.float32)
    cross_has_mass = summed_masked != 0.0
    cross = jnp.where(cross_has_mass, proj * jnp.conj(summed_masked), 0.0)
    xa_raw = (noise_variance[None, :] * cross.real).astype(jnp.float32)
    safe_scale = jnp.maximum(row_scale, jnp.asarray(1e-30, dtype=jnp.float32))
    xa = (xa_raw / safe_scale[:, None]).astype(jnp.float32)
    aa = (aa_raw / (safe_scale[:, None] ** 2)).astype(jnp.float32)

    tiles = jnp.asarray(raw_shifted_images, dtype=jnp.complex64)[row_image_local]
    image_power = _relion_wavg_rectangle_image_power(tiles, posterior[:, None, :])[:, 0, :]
    diff2 = (
        (image_power + aa_raw) - jnp.asarray(2.0, dtype=jnp.float32) * xa_raw
    ).astype(jnp.float32)
    return jnp.stack((xa, aa, diff2), axis=-1)


@partial(jax.jit, static_argnames=("power_per_image",))
def _resident_block_wavg_rectangle_terms(
    exact_terms,  # float32 [block, P_exact, 3]
    raw_shifted_rectangle,  # complex64 [C_B, T, P_rect]
    row_posterior,  # float32 [block, T]
    row_image_local,  # int32 [block]
    exact_positions,  # int32 [P_exact]
    *,
    power_per_image: bool = False,
):
    """Flat-row twin of ``_relion_wavg_rectangle_triplet_terms``.

    Same two statements as the rectangular helper: fill the whole rectangle's
    ``diff2`` slot with RELION's posterior-weighted image power, then overwrite
    the exact-radius positions with the projected triplet.

    ``power_per_image`` moves the squaring to the other side of the gather.
    ``|x|^2`` depends only on the image's rectangle, so with the flag on it is
    computed once for the chunk's ``C_B`` images and the float32 result is
    gathered to the block's rows; with it off the complex rectangle is gathered
    first and squared once per row. Squaring is elementwise, so both orders give
    the same float32 values, and the contraction that consumes them keeps the
    same operand shapes and the same reduction over the translation axis: the
    two settings are bitwise. The block is ``mstep_block_rows`` rows (2048 in
    production) over an image capacity of 32 or 128, so the flag removes 16x to
    64x of the squaring and halves the gathered bytes, float32 rather than
    complex64. P4-G measured that squaring at 2.011 s of a 4.5 s hp3 replay
    iteration.
    """

    row_image_local = jnp.asarray(row_image_local, dtype=jnp.int32)
    block_posterior = jnp.asarray(row_posterior, dtype=jnp.float32)[:, None, :]
    if power_per_image:
        image_power = _relion_wavg_rectangle_power_contraction(
            _relion_wavg_shifted_power(raw_shifted_rectangle)[row_image_local],
            block_posterior,
        )[:, 0, :]
    else:
        tiles = jnp.asarray(raw_shifted_rectangle, dtype=jnp.complex64)[row_image_local]
        image_power = _relion_wavg_rectangle_image_power(tiles, block_posterior)[:, 0, :]
    rectangle_terms = jnp.zeros(image_power.shape + (3,), dtype=jnp.float32)
    rectangle_terms = rectangle_terms.at[..., 2].set(image_power)
    return rectangle_terms.at[:, jnp.asarray(exact_positions, dtype=jnp.int32), :].set(
        jnp.asarray(exact_terms, dtype=jnp.float32)
    )


# ---------------------------------------------------------------------------
# Per-chunk image-level statistics (every term without a row-pixel axis)
# ---------------------------------------------------------------------------


class _ChunkImageOperands(NamedTuple):
    """One chunk's image-level statistics operands, all already on the device."""

    row_posterior: jax.Array  # float32 [C_R, T]
    row_image_local: jax.Array  # int32 [C_R]
    row_coarse_rot: jax.Array  # int32 [C_R], padded -> >= n_coarse_rot
    image_ids: jax.Array  # int32 [C_B], padded -> -1
    group_ids: jax.Array  # int32 [C_B], padded -> -1
    processed_image_half: jax.Array  # complex [C_B, P_half]
    relion_norm_high_shell: jax.Array  # real [C_B]
    wavg_triplet_pixels: jax.Array  # float32 [C_B, P_rect, 3]
    block_noise_shells: jax.Array  # float64 [n_shells]
    a2_per_image: jax.Array  # real [C_B]
    xa_per_image: jax.Array  # real [C_B]
    class_log_z: jax.Array  # float64 [C_B]
    min_diff2: jax.Array  # real [C_B]
    best_log_score: jax.Array  # float32 [C_B]
    max_posterior: jax.Array  # float32 [C_B]
    best_cell_index: jax.Array  # int64 [C_B], segment-relative (r_local * T + t)
    best_fine_rot: jax.Array  # int64 [C_B], global fine rotation id of the winner


class _ChunkImageTables(NamedTuple):
    """Iteration-global tables the image-level program reads every chunk."""

    shell_indices_half: jax.Array  # int32 [P_half]
    wavg_shell_indices: jax.Array  # int32 [P_rect]
    wavg_scale_pixel_mask: jax.Array  # bool [P_rect]
    translation_sqdist_ang: jax.Array | None  # real [T] or [C_B, T] or None


@partial(jax.jit, static_argnames=("config",))
def _accumulate_chunk_image_terms(
    stats: ResidentStatistics,
    operands: _ChunkImageOperands,
    tables: _ChunkImageTables,
    *,
    config,
) -> ResidentStatistics:
    """Fold one chunk's image-level terms into the device accumulators.

    Statement for statement this is the host bucket tail in its production
    configuration; the only change is that the row-pixel reductions arrive as
    already-summed chunk partials, because the pixel axis is walked in blocks.
    """

    n_shells = int(config.n_shells)
    n_fine_trans = int(config.n_fine_trans)
    image_capacity = int(operands.image_ids.shape[0])

    probs = operands.row_posterior
    row_image = jnp.asarray(operands.row_image_local, dtype=jnp.int32)
    image_ids = jnp.asarray(operands.image_ids, dtype=jnp.int32)
    valid_image = image_ids >= 0
    image_slot = _drop_index(image_ids, int(config.n_images))

    # --- 1/2. sigma2 offset and support mass -------------------------------
    translation_posterior = segment_sum_by_image(probs, row_image, image_capacity)
    sigma2_offset = stats.sigma2_offset
    if tables.translation_sqdist_ang is not None:
        sqdist = jnp.asarray(tables.translation_sqdist_ang, dtype=jnp.float64)
        sigma2_offset = sigma2_offset + jnp.sum(
            translation_posterior.astype(jnp.float64) * sqdist
        )
    support_mass = jnp.sum(translation_posterior, axis=1)
    support_mass = jnp.where(valid_image, support_mass, jnp.zeros((), support_mass.dtype))
    sumw = stats.sumw + jnp.sum(support_mass.astype(jnp.float64))

    # --- 3. weighted image power shells and per-image norm power -----------
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

    # --- 4. per-particle norm correction (production algebraic mode) -------
    # The host adds the image-power term and the A2-2XA residual in two
    # separate ``+=`` statements; keep both scatters separate.
    norm_correction = stats.norm_correction.at[image_slot].add(
        jnp.where(valid_image, weighted_img_per_image, 0.0).astype(jnp.float64),
        mode="drop",
    )

    # --- 5/6. noise shells with RELION's direct low-shell replacement ------
    residual_shells, image_power_shells = (
        _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
            jnp.asarray(operands.block_noise_shells, dtype=jnp.float64),
            weighted_img_shells.astype(jnp.float64),
            operands.wavg_triplet_pixels[:, :, 2],
            tables.wavg_shell_indices,
            exclusive_shell_stop=int(config.direct_noise_exclusive_shell_stop),
            shell_count=n_shells,
        )
    )
    wsum_sigma2_noise = stats.wsum_sigma2_noise + residual_shells
    wsum_img_power = stats.wsum_img_power + image_power_shells

    # --- 7. per-image norm residual ---------------------------------------
    block_norm_residual = operands.a2_per_image - 2.0 * operands.xa_per_image
    norm_correction = norm_correction.at[image_slot].add(
        jnp.where(valid_image, block_norm_residual, 0.0).astype(jnp.float64),
        mode="drop",
    )

    # --- 8. group scale sufficient statistics ------------------------------
    scale_xa = stats.scale_xa
    scale_aa = stats.scale_aa
    if config.accumulate_scale:
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
        group_slot = _drop_index(operands.group_ids, int(config.n_scale_groups))
        keep_group = valid_image & (jnp.asarray(operands.group_ids, dtype=jnp.int32) >= 0)
        scale_xa = scale_xa.at[group_slot].add(
            jnp.where(keep_group, scale_xa_per_image, 0.0).astype(jnp.float64), mode="drop"
        )
        scale_aa = scale_aa.at[group_slot].add(
            jnp.where(keep_group, scale_aa_per_image, 0.0).astype(jnp.float64), mode="drop"
        )

    # --- 9. rotation posterior sums ----------------------------------------
    probs_sum_t = jnp.sum(probs, axis=-1)
    coarse_slot = _drop_index(operands.row_coarse_rot, int(config.n_coarse_rot))
    rotation_posterior_sums = stats.rotation_posterior_sums.at[coarse_slot].add(
        probs_sum_t.astype(jnp.float64), mode="drop"
    )

    # --- 10. per-image score and pose fields -------------------------------
    # ``_relion_cuda_fine_log_evidence_offset`` is exactly ``-min_diff2``.
    log_score_offset = (-jnp.asarray(operands.min_diff2)).astype(jnp.float64)
    class_log_z = jnp.asarray(operands.class_log_z, dtype=jnp.float64)
    best_log_score_chunk = jnp.asarray(operands.best_log_score, dtype=jnp.float64)
    finite = jnp.isfinite(best_log_score_chunk)
    neg_inf = jnp.asarray(-jnp.inf, dtype=jnp.float64)
    absolute = class_log_z + log_score_offset

    log_evidence = stats.log_evidence.at[image_slot].set(
        jnp.where(finite, absolute, neg_inf), mode="drop"
    )
    score_log_z = stats.score_log_z.at[image_slot].set(
        jnp.where(finite, absolute, neg_inf), mode="drop"
    )
    best_log_score = stats.best_log_score.at[image_slot].set(
        best_log_score_chunk + log_score_offset, mode="drop"
    )
    max_posterior = stats.max_posterior.at[image_slot].set(
        jnp.asarray(operands.max_posterior, dtype=stats.max_posterior.dtype), mode="drop"
    )

    best_cell_index = jnp.asarray(operands.best_cell_index, dtype=jnp.int64)
    best_local_rot = (best_cell_index // jnp.int64(n_fine_trans)).astype(jnp.int32)
    best_translation = best_cell_index % jnp.int64(n_fine_trans)
    best_cell_values = (
        jnp.asarray(operands.best_fine_rot, dtype=jnp.int64) * jnp.int64(n_fine_trans)
        + best_translation
    )
    best_cell = stats.best_cell.at[image_slot].set(best_cell_values, mode="drop")
    best_local_rot_out = stats.best_local_rot.at[image_slot].set(best_local_rot, mode="drop")

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
        best_local_rot=best_local_rot_out,
        invalid_best_rows=stats.invalid_best_rows,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _pad_batch_to_capacity(values, capacity: int):
    """Repeat a host batch's first row up to ``capacity`` rows.

    Every per-chunk device program is keyed on its operand shapes, so a batch
    whose length is the chunk's *valid* image count traces a new program for
    every distinct occupancy. Padding the batch on the host, before anything
    is traced, gives one program per image-capacity class instead. The padded
    rows carry a duplicate image's real data and are zeroed after preparation
    by :func:`_zero_padded_images`, which is itself capacity-shaped.
    """

    values = np.asarray(values)
    n = int(values.shape[0])
    if n == capacity:
        return values
    if n > capacity or n == 0:
        raise ValueError(f"cannot pad a batch of {n} rows to capacity {capacity}")
    pad = np.repeat(values[:1], capacity - n, axis=0)
    return np.concatenate([values, pad], axis=0)


def _reorder_permutation(fetched_indices, requested_indices, capacity: int) -> np.ndarray:
    """Host permutation from the fetched batch order to the table order.

    The dataset may return a batch in its own order. The compact engine
    reorders its bucket arrays to follow the fetch; the resident driver keeps
    the table's (RELION particle) order and permutes the operands, which is a
    pure gather and changes no arithmetic. Padded slots point at fetched row
    0; :func:`_zero_padded_images` removes whatever they gathered.
    """

    fetched = np.asarray(fetched_indices).reshape(-1)
    requested = np.asarray(requested_indices).reshape(-1)
    n = int(requested.shape[0])
    position_of = {int(index): position for position, index in enumerate(requested.tolist())}
    if len(position_of) != n:
        raise ValueError("a chunk must not request the same image twice")
    order = np.zeros(capacity, dtype=np.int32)
    seen = np.zeros(n, dtype=bool)
    for slot, dataset_index in enumerate(fetched[:n].tolist()):
        position = position_of.get(int(dataset_index))
        if position is None:
            raise ValueError(f"the dataset returned image {dataset_index}, which was not requested")
        order[position] = slot
        seen[position] = True
    if not bool(seen.all()):
        raise ValueError("the dataset did not return every requested image")
    return order


class _ChunkOperandRowInputs(NamedTuple):
    """The per-chunk operands that are permuted into row order and zero-padded.

    Each optional field is ``None`` on the paths that do not produce it;
    ``None`` is a pytree structure, so those paths key their own program
    rather than carry a dead operand.
    """

    score_input: jax.Array
    corr_img_score: jax.Array
    highres_xi2_half: jax.Array | None
    shifted_recon: jax.Array
    shifted_noise: jax.Array
    ctf2_over_nv_recon: jax.Array
    direct_ctf_rfloat_recon: jax.Array | None
    processed_score_half_for_noise: jax.Array
    relion_norm_high_shell: jax.Array | None
    raw_translated_wavg_rectangle: jax.Array


@partial(jax.jit, static_argnames=("image_capacity", "n_fine_trans"))
def _chunk_operand_rows(
    arrays: _ChunkOperandRowInputs,
    permutation: jax.Array,
    valid_images: jax.Array,
    exact_positions: jax.Array,
    *,
    image_capacity: int,
    n_fine_trans: int,
) -> tuple:
    """Permute a chunk's operands into row order and zero the padded slots.

    Same statements, same order, same dtypes as the loose dispatch: two
    reshapes, ten permutation gathers, ten ``where`` masks and one rectangle
    gather. None of it is arithmetic, so no value can move; what leaves the
    host is the dispatch count. Eagerly, ``values[permutation]`` is five
    primitives rather than one, because JAX normalizes a fancy index
    (``add``, ``broadcast_in_dim``, ``select_n``) before every gather, and the
    resident local pass runs this once per chunk.
    """

    def take(values):
        return None if values is None else _zero_padded_images(
            values[permutation], valid_images
        )

    raw_translated_wavg_rectangle = take(arrays.raw_translated_wavg_rectangle)
    return (
        take(arrays.score_input),
        take(arrays.corr_img_score),
        take(arrays.highres_xi2_half),
        take(arrays.shifted_recon.reshape(image_capacity, n_fine_trans, -1)),
        take(arrays.shifted_noise.reshape(image_capacity, n_fine_trans, -1)),
        take(arrays.ctf2_over_nv_recon),
        take(arrays.direct_ctf_rfloat_recon),
        take(arrays.processed_score_half_for_noise),
        take(arrays.relion_norm_high_shell),
        raw_translated_wavg_rectangle,
        raw_translated_wavg_rectangle[:, :, exact_positions],
    )


def _zero_padded_images(values, valid_images):
    """Zero the padded image slots of a capacity-shaped operand.

    ``valid_images`` is a capacity-shaped bool, so this traces one program per
    (capacity, trailing shape) pair regardless of how many slots are valid.
    """

    values = jnp.asarray(values)
    mask = jnp.asarray(valid_images, dtype=bool).reshape((-1,) + (1,) * (values.ndim - 1))
    return jnp.where(mask, values, jnp.zeros((), dtype=values.dtype))


def compute_pass2_stats_resident(
    experiment_dataset,
    volume,
    noise_variance,
    translations,
    significant_sample_indices,
    nside_level,
    disc_type,
    *,
    oversampling_order,
    current_size,
    reconstruction_current_size=None,
    translation_step,
    rotation_log_prior,
    score_with_masked_images,
    return_stats,
    translation_log_prior,
    accumulate_noise,
    half_spectrum_scoring,
    projection_padding_factor,
    projection_mask_current_image_disk=False,
    reconstruction_padding_factor,
    image_corrections,
    scale_corrections,
    image_pre_shifts,
    use_float64_scoring,
    translation_prior_centers=None,
    do_gridding_correction=False,
    square_window=False,
    random_perturbation,
    group_ids=None,
    scale_correction_group_count=None,
    scale_correction_data_vs_prior=None,
    normalization_log_z=None,
    relion_f32_normalization_sum_weight=None,
    relion_coarse_hard_assignment=None,
    relion_coarse_max_posterior=None,
    normalization_other_score_log_z=None,
    normalization_score_mode=None,
    return_score_log_z=False,
    return_score_log_z_only=False,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    rotation_block_size_for_quantization=5000,
    fine_source_eulers_override=None,
    return_source_eulers=False,
    fine_rotations_override=None,
    fine_mstep_rotations_override=None,
    fine_rotation_parent_override=None,
    fine_translations_override=None,
    fine_translation_parent_override=None,
    relion_half_volume_mstep=False,
    relion_x_half_mstep=False,
    mstep_subtract_ctf_projection=False,
    relion_fine_mstep_prune=False,
    relion_firstiter_score_mode="gaussian",
    relion_firstiter_winner_take_all=False,
    relion_exact_fine_gaussian=True,
    relion_fine_diff2_fused_ffi=False,
    relion_f32_fine_posterior=False,
    relion_exact_fine_normalized_cc=False,
    relion_projector_half=None,
    relion_projector_texture=None,
    relion_projector_r_max=None,
    adaptive_fraction=0.999,
    bpref_device_signature_active: bool = False,
    bpref_class_index: int = 0,
    include_unweighted_norm_high_shell: bool = True,
    preserve_bpref_particle_order: bool = False,
    source_faithful_spectrum_norm: bool = False,
    symmetry_label: str = "C1",
):
    """Device-resident K=1 sparse pass 2; same signature and return as the compact engine.

    See the module docstring for what is layout-equal to the compact engine and
    what is a deliberate reduction-order change. The configuration gate runs
    before any device work, so an unsupported pass fails immediately instead of
    part way through a half.
    """

    from recovar import cuda_backproject
    from recovar.em.sampling import (
        get_oversampled_translation_grid,
        infer_translation_step,
        rotation_grid_size,
    )
    from recovar.em.sparse_pass2.sparse_pass2_bucketed import (
        _pass2_projector_complex64_enabled,
    )
    from recovar.em.sparse_pass2.sparse_pass2_window import (
        _fine_translation_prior_2d,
        _pass2_projection_budget,
        _pass2_relion_flags,
    )

    overall_t0 = time.time()
    (
        use_exact_relion_gaussian,
        _use_relion_fine_diff2_fused_ffi,
        use_relion_f32_fine_posterior,
    ) = _pass2_relion_flags(
        relion_exact_fine_gaussian=relion_exact_fine_gaussian,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        relion_fine_diff2_fused_ffi=relion_fine_diff2_fused_ffi,
        relion_f32_fine_posterior=relion_f32_fine_posterior,
    )

    n_images = experiment_dataset.n_units
    n_coarse_trans = int(np.asarray(translations).shape[0])
    n_coarse_rot = rotation_grid_size(nside_level)
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape

    (
        mstep_current_size,
        n_half,
        window_spec_kwargs,
        budget_window_spec,
        device_memory_bytes,
        precision_policy,
    ) = _pass2_window_setup(
        image_shape,
        current_size=current_size,
        reconstruction_current_size=reconstruction_current_size,
        half_spectrum_scoring=half_spectrum_scoring,
        square_window=square_window,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        use_float64_scoring=use_float64_scoring,
    )

    fresh_k1_guard = bool(source_faithful_spectrum_norm)
    resolved_spectrum_norm = _relion_powerclass_spectrum_norm_enabled(
        fresh_k1_guard=fresh_k1_guard,
    )
    relion_exact_bpref_operands = _relion_exact_bpref_operands_enabled(
        fresh_k1_guard=fresh_k1_guard,
        source_faithful_spectrum_norm=resolved_spectrum_norm,
    )
    direct_noise_default = _fresh_k1_direct_noise_default(
        preserve_bpref_particle_order=preserve_bpref_particle_order,
        relion_exact_bpref_operands=relion_exact_bpref_operands,
    )
    scale_groups_available = group_ids is not None
    relion_wavg_atomic_scale_aa = bool(
        accumulate_noise
        and scale_groups_available
        and parse_env_flag(_RELION_WAVG_ATOMIC_SCALE_AA_ENV, default=direct_noise_default)
    )
    relion_wavg_atomic_direct_noise, relion_wavg_atomic_direct_norm = _relion_wavg_direct_modes(
        accumulate_noise=bool(accumulate_noise),
        scale_groups_available=scale_groups_available,
        scale_aa_enabled=bool(relion_wavg_atomic_scale_aa),
        direct_noise_only_default=direct_noise_default,
    )

    soft_posterior_block_bpref = parse_env_flag(
        _SOFT_POSTERIOR_BLOCK_BPREF_PROTOTYPE_ENV, default=False
    )
    projection_cache_enabled = _projection_cache_enabled_for_pass(
        fine_rotations_override=fine_rotations_override,
        dump_pass2_operands=False,
    )
    require_resident_production_configuration(
        relion_x_half_mstep=relion_x_half_mstep,
        relion_exact_fine_gaussian=relion_exact_fine_gaussian,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        use_float64_scoring=use_float64_scoring,
        relion_firstiter_winner_take_all=relion_firstiter_winner_take_all,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        return_score_log_z_only=return_score_log_z_only,
        accumulate_noise=accumulate_noise,
        mstep_subtract_ctf_projection=mstep_subtract_ctf_projection,
        normalization_log_z=normalization_log_z,
        normalization_other_score_log_z=normalization_other_score_log_z,
        relion_f32_normalization_sum_weight=relion_f32_normalization_sum_weight,
        relion_coarse_hard_assignment=relion_coarse_hard_assignment,
        preserve_bpref_particle_order=preserve_bpref_particle_order,
        soft_posterior_block_bpref=soft_posterior_block_bpref,
        fine_rotations_override=fine_rotations_override,
        fine_rotation_parent_override=fine_rotation_parent_override,
        n_coarse_trans=n_coarse_trans,
        use_window=budget_window_spec.use_window,
        projection_cache_available=projection_cache_enabled,
        relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
        relion_wavg_atomic_direct_noise=relion_wavg_atomic_direct_noise,
        relion_wavg_atomic_direct_norm=relion_wavg_atomic_direct_norm,
        relion_projector_texture=relion_projector_texture,
        symmetry_label=symmetry_label,
    )
    _require(
        bool(use_relion_f32_fine_posterior),
        "the RELION float32 fine posterior is the segmented kernel's contract",
    )
    _require(
        bool(relion_fine_mstep_prune) or bool(relion_x_half_mstep),
        "the resident M-step reconstructs from RELION's pruned fine weights",
    )
    _require(
        jax.default_backend() == "gpu" and cuda_backproject.custom_cuda_requested(),
        "every resident stage is a CUDA FFI target",
    )

    # ---- accumulator layout (identical to the compact engine) -------------
    recon_volume_shape = relion_backprojector_volume_shape(
        volume_shape,
        reconstruction_padding_factor,
        current_size=mstep_current_size,
    )
    recon_accum_shape = half_volume_accumulator_shape(recon_volume_shape)
    recon_volume_size = int(np.prod(recon_accum_shape))
    recon_y_accum_dtype, recon_ctf_accum_dtype = relion_x_half_mstep_accumulator_dtypes(
        experiment_dataset.dtype,
        use_relion_x_half_mstep=True,
    )
    logger.info(
        "Resident pass-2 RELION x-half current-size BPref accumulator shape: "
        "volume_shape=%s score_current_size=%s model_current_size=%s padding_factor=%s "
        "recon_volume_shape=%s half_accum_shape=%s voxels=%d",
        tuple(volume_shape),
        current_size,
        mstep_current_size,
        reconstruction_padding_factor,
        tuple(recon_volume_shape),
        tuple(recon_accum_shape),
        recon_volume_size,
    )

    # ---- projection volume ------------------------------------------------
    use_relion_projector = relion_projector_half is not None
    if use_relion_projector:
        if relion_projector_r_max is None:
            raise ValueError("relion_projector_r_max is required when relion_projector_half is provided")
        relion_projector_half = jnp.asarray(relion_projector_half)
        # Narrowing the Projector::data slab engages the texture projector and
        # changes float32 projection arithmetic, so the compact engine keeps it
        # behind its own opt-in flag. Read the same gate: narrowing here on our
        # own would change every projection, every score and every posterior
        # relative to the engine this driver is measured against.
        if (
            _pass2_projector_complex64_enabled()
            and not use_float64_scoring
            and relion_projector_half.dtype == jnp.complex128
        ):
            relion_projector_half = relion_projector_half.astype(jnp.complex64)
    if projection_padding_factor > 1 and not use_relion_projector:
        from recovar.reconstruction.relion_functions import pad_volume_for_projection

        mean_for_proj, proj_volume_shape = pad_volume_for_projection(
            volume,
            volume_shape,
            projection_padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=mstep_current_size,
        )
    else:
        mean_for_proj = volume
        proj_volume_shape = volume_shape

    # ---- fine translations and priors -------------------------------------
    translations_source_np = np.asarray(translations)
    translations_np = np.asarray(translations_source_np, dtype=precision_policy.score_real_dtype)
    if translation_step is None:
        translation_step = infer_translation_step(translations_np)
    if fine_translations_override is None and fine_translation_parent_override is None:
        fine_translations_source, fine_translation_parent = get_oversampled_translation_grid(
            translations_source_np,
            translation_step,
            oversampling_order=oversampling_order,
        )
        fine_translations = np.asarray(
            fine_translations_source, dtype=precision_policy.score_real_dtype
        )
        fine_translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)
    elif fine_translations_override is not None and fine_translation_parent_override is not None:
        fine_translations_source = np.asarray(fine_translations_override)
        fine_translations = np.asarray(
            fine_translations_source, dtype=precision_policy.score_real_dtype
        )
        fine_translation_parent = np.asarray(fine_translation_parent_override, dtype=np.int32)
    else:
        raise ValueError(
            "fine_translations_override and fine_translation_parent_override must be provided together",
        )
    n_fine_trans = int(fine_translations.shape[0])

    translation_prior_centers_np = validate_translation_prior_centers(
        translation_prior_centers,
        n_images=n_images,
        n_dims=translations_np.shape[1],
    )
    fine_translation_prior_2d = _fine_translation_prior_2d(
        translation_log_prior,
        fine_translation_parent,
        n_images=n_images,
        n_fine_trans=n_fine_trans,
        dtype=precision_policy.score_real_dtype,
    )

    # ---- per-image hypotheses (unchanged) ---------------------------------
    prep_t0 = time.time()
    significance_csr = resident_significance_csr(
        significant_sample_indices,
        n_images=n_images,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
    )
    per_image_inputs = None if significance_csr is not None else _prepare_per_image_pass2_inputs(
        significant_sample_indices,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        nside_level=nside_level,
        oversampling_order=oversampling_order,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_translation_parent,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=random_perturbation,
        fine_source_eulers_override=fine_source_eulers_override,
        fine_rotations_override=fine_rotations_override,
        fine_mstep_rotations_override=fine_mstep_rotations_override,
        fine_rotation_parent_override=fine_rotation_parent_override,
        relion_parent_execution_order=_relion_fine_parent_execution_order_enabled(
            use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
        ),
        dtype=precision_policy.score_real_dtype,
    )
    prep_s = time.time() - prep_t0

    # ---- T5: candidate table and capacity chunks --------------------------
    table_t0 = time.time()
    tables = resident_candidate_tables(
        significance_csr,
        per_image_inputs,
        n_coarse_trans=n_coarse_trans,
        n_fine_trans=n_fine_trans,
        fine_translation_parent=fine_translation_parent,
        nside_level=nside_level,
        oversampling_order=oversampling_order,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=random_perturbation,
        fine_rotation_parent_override=fine_rotation_parent_override,
        relion_parent_execution_order=_relion_fine_parent_execution_order_enabled(
            use_relion_f32_fine_posterior=use_relion_f32_fine_posterior,
        ),
        dtype=precision_policy.score_real_dtype,
    )
    if int(tables.n_images) != int(n_images):
        raise ValueError(
            f"candidate table covers {tables.n_images} images but the dataset has {n_images}"
        )

    # ---- window / weights / lookups (unchanged) ---------------------------
    window_setup = _sparse_pass2_window_setup(
        experiment_dataset,
        disc_type=disc_type,
        image_shape=image_shape,
        current_size=current_size,
        n_half=n_half,
        mstep_current_size=mstep_current_size,
        square_window=square_window,
        window_spec_kwargs=window_spec_kwargs,
        use_relion_x_half_mstep=True,
        log_label="Resident pass-2",
    )
    config = window_setup.config
    window_spec = window_setup.window_spec
    window_indices_np = window_setup.window_indices_np
    window_indices = window_setup.window_indices
    recon_window_indices = window_setup.recon_window_indices
    relion_x_half_recon_indices = window_setup.relion_x_half_recon_indices
    windowed_prepare = window_setup.windowed_prepare
    n_windowed = window_setup.n_windowed
    n_recon_windowed = window_setup.n_recon_windowed

    half_weights, half_weights_windowed = _pass2_half_weights(
        image_shape,
        window_spec,
        half_spectrum_scoring=half_spectrum_scoring,
        relion_firstiter_score_mode=relion_firstiter_score_mode,
        use_float64_scoring=use_float64_scoring,
    )
    relion_score_full_to_compact = jnp.asarray(
        _relion_cuda_fine_full_to_compact_lookup(image_shape, current_size, window_indices_np),
        dtype=jnp.int32,
    )
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(
        noise_variance, image_shape
    ).squeeze()
    relion_score_translation_angles = _relion_cuda_score_translation_angles_if_available(
        fine_translations_source,
        image_shape,
        enabled=True,
        dtype=np.float64 if use_float64_scoring else np.float32,
    )
    if relion_score_translation_angles is None:
        raise ValueError("the resident scoring stage requires RELION translation angles")
    translation_phases_half = (
        None if windowed_prepare else half_translation_phase_table(fine_translations, image_shape)
    )

    n_shells = image_shape[0] // 2 + 1
    shell_indices_half = mask_relion_noise_shell_indices_to_current_window(
        make_relion_noise_shell_indices_half(image_shape),
        image_shape,
        current_size,
        window_indices,
    )
    shell_indices_noise = window_spec.recon_values(shell_indices_half)
    noise_variance_for_noise = window_spec.recon_values(noise_variance_half)
    scale_correction_pixel_mask = _relion_scale_correction_pixel_mask(
        scale_correction_data_vs_prior,
        shell_indices_noise,
        n_shells=n_shells,
    )
    relion_wavg_rectangle = _make_relion_wavg_rectangle(
        image_shape,
        current_size,
        recon_window_indices,
        reconstruction_current_size=mstep_current_size,
    )
    n_rect = int(relion_wavg_rectangle.centered_indices.size)
    scale_pixel_mask_rect_np = np.zeros(n_rect, dtype=bool)
    scale_pixel_mask_rect_np[relion_wavg_rectangle.exact_positions] = np.asarray(
        scale_correction_pixel_mask, dtype=bool
    )
    group_ids_np, n_scale_groups = prepare_scale_correction_groups(
        group_ids, scale_correction_group_count, n_images=n_images,
    )

    # ---- projection cache (same admission and build as the compact engine) -
    n_fine_rot = int(np.asarray(fine_rotations_override).shape[0])
    (
        _projection_complex_dtype,
        _projection_budget_pixels,
        max_projected_rotations_per_projection_call,
    ) = _pass2_projection_budget(
        jnp.asarray(mean_for_proj).dtype,
        precision_policy,
        n_half=n_half,
        use_relion_projector=use_relion_projector,
        budget_window_spec=budget_window_spec,
        device_memory_bytes=device_memory_bytes,
        include_abs2=False,
    )
    transient_projection_bytes = _projection_cache_transient_bytes(
        n_fine_rot,
        n_windowed,
        projection_complex_dtype=precision_policy.score_complex_dtype,
        include_abs2=False,
    ) + _projection_cache_transient_bytes(
        n_fine_rot,
        n_recon_windowed,
        projection_complex_dtype=precision_policy.score_complex_dtype,
        include_abs2=True,
    )
    max_projection_cache_bytes = _projection_cache_max_bytes_for_pass(device_memory_bytes)
    _require(
        _projection_cache_fits_budget(transient_projection_bytes, max_projection_cache_bytes),
        "the per-iteration projection cache did not fit its budget "
        f"({transient_projection_bytes / float(1024 ** 3):.2f} GiB > "
        f"{max_projection_cache_bytes / float(1024 ** 3):.2f} GiB)",
    )
    cache_t0 = time.time()
    projection_kwargs = _projection_kwargs_for_relion_score_window(
        window_spec.projection_kwargs(return_abs2=False),
        use_relion_projector=use_relion_projector,
        current_size=current_size,
    )
    projection_kwargs["mask_current_image_disk"] = bool(projection_mask_current_image_disk)
    score_cache, recon_cache, recon_abs2_cache = _compute_sparse_pass2_windowed_projections_block(
        mean_for_proj,
        jnp.asarray(fine_rotations_override, dtype=precision_policy.score_real_dtype),
        image_shape,
        proj_volume_shape,
        disc_type,
        score_indices=window_indices,
        recon_indices=recon_window_indices,
        max_projected_rotations=_projection_cache_build_max_rotations_per_call(
            max_projected_rotations_per_projection_call,
            n_fine_rot,
        ),
        output_complex_dtype=precision_policy.score_complex_dtype,
        output_abs2_dtype=precision_policy.score_real_dtype,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=relion_projector_r_max,
        projection_padding_factor=projection_padding_factor,
        **projection_kwargs,
    )
    recon_cache, recon_abs2_cache = precision_policy.cast_local_noise_projection_scores(
        recon_cache, recon_abs2_cache
    )
    logger.info(
        "Resident pass-2 projection cache: cached %d fine rotations in %.2fs "
        "(estimated transient %.2f GiB)",
        n_fine_rot,
        time.time() - cache_t0,
        transient_projection_bytes / float(1024**3),
    )

    # ---- capacity plan ----------------------------------------------------
    row_ladder = parse_env_capacity_ladder(_ROW_CAPACITY_LADDER_ENV, _DEFAULT_ROW_CAPACITY_LADDER)
    image_ladder = _cap_image_capacity_ladder(
        parse_env_capacity_ladder(_IMAGE_CAPACITY_LADDER_ENV, _DEFAULT_IMAGE_CAPACITY_LADDER),
        n_fine_trans=n_fine_trans,
        n_recon_pixels=n_recon_windowed,
        max_tile_bytes=_max_translation_tile_bytes_for_pass(
            device_memory_bytes, has_external_normalization=False
        ),
    )
    mstep_block_rows = _resolve_mstep_block_rows(
        n_recon_pixels=n_recon_windowed,
        max_block_bytes=_max_adjoint_block_bytes_for_pass(device_memory_bytes),
        row_capacity_ladder=row_ladder,
    )
    chunks = plan_capacity_chunks(
        tables,
        row_capacity_ladder=row_ladder,
        image_capacity_ladder=image_ladder,
    )
    plan = ResidentPass2Plan(
        chunks=tuple(chunks),
        row_capacity_ladder=tuple(row_ladder),
        image_capacity_ladder=tuple(image_ladder),
        mstep_block_rows=int(mstep_block_rows),
    )
    table_s = time.time() - table_t0
    row_slots = sum(int(chunk.row_capacity) for chunk in chunks)
    image_slots = sum(int(chunk.image_capacity) for chunk in chunks)
    logger.info(
        "Resident pass-2 plan: %d images, %d candidate rows -> %d chunks "
        "(row capacities %s, image capacities %s, M-step block rows %d, "
        "row occupancy %.3f of %d slots, image occupancy %.3f of %d slots); "
        "setup hypothesis_prep=%.2fs table+plan=%.2fs",
        tables.n_images,
        tables.n_rows,
        len(chunks),
        ",".join(str(v) for v in plan.row_capacity_ladder),
        ",".join(str(v) for v in plan.image_capacity_ladder),
        plan.mstep_block_rows,
        tables.n_rows / max(row_slots, 1),
        row_slots,
        tables.n_images / max(image_slots, 1),
        image_slots,
        prep_s,
        table_s,
    )

    # ---- preparation arguments --------------------------------------------
    # One keyword set, used by whichever preparation the pass selects: the
    # once-per-half resident preparation below, or the per-chunk call that
    # stays as its oracle. The preparation is per-image pure, so the two return
    # the same rows; the resident form runs it once for the half instead of
    # once per chunk, which is where the chunk loop's launches came from.
    bucket_io_kwargs = dict(
        noise_variance_half=noise_variance_half,
        fine_translations=fine_translations,
        config=config,
        n_trans=n_fine_trans,
        score_with_masked_images=score_with_masked_images,
        half_spectrum_scoring=half_spectrum_scoring,
        image_corrections=image_corrections,
        scale_corrections=scale_corrections,
        image_pre_shifts=image_pre_shifts,
        use_float64_scoring=use_float64_scoring,
        score_only=False,
        score_mode=relion_firstiter_score_mode,
        window_indices=window_indices,
        recon_window_indices=recon_window_indices,
        translation_phases_half=translation_phases_half,
        relion_score_translation_angles=relion_score_translation_angles,
        return_windowed_shifted=windowed_prepare,
        relion_exact_normalized_cc_operands=relion_exact_fine_normalized_cc,
        relion_exact_bpref_operands=relion_exact_bpref_operands,
    )

    # ---- resident row-aligned tables --------------------------------------
    fine_grid = jnp.asarray(fine_rotations_override, dtype=precision_policy.score_real_dtype)
    mstep_grid = (
        fine_grid
        if fine_mstep_rotations_override is None
        else jnp.asarray(fine_mstep_rotations_override, dtype=precision_policy.score_real_dtype)
    )
    coarse_parent_grid = jnp.asarray(
        np.asarray(fine_rotation_parent_override, dtype=np.int32), dtype=jnp.int32
    )
    projection_score_cache = jnp.asarray(score_cache)
    projection_recon_cache = jnp.asarray(recon_cache)
    projection_recon_abs2_cache = jnp.asarray(recon_abs2_cache)
    fine_translation_parent_device = jnp.asarray(fine_translation_parent, dtype=jnp.int32)

    scale_corrections_np = (
        None
        if scale_corrections is None
        else np.asarray(scale_corrections, dtype=precision_policy.score_real_dtype)
    )

    # ---- statistics accumulators ------------------------------------------
    stats_config = resolve_statistics_config(
        n_shells=n_shells,
        n_fine_trans=n_fine_trans,
        n_images=n_images,
        n_coarse_rot=n_coarse_rot,
        n_scale_groups=n_scale_groups,
        current_size=current_size,
        include_unweighted_high_shell=include_unweighted_norm_high_shell,
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        relion_wavg_atomic_direct_noise=relion_wavg_atomic_direct_noise,
        relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
        accumulate_scale=scale_groups_available,
        source_faithful_spectrum_norm=resolved_spectrum_norm,
    )
    stats = make_resident_statistics(
        stats_config, max_posterior_dtype=precision_policy.score_real_dtype
    )
    image_tables = _ChunkImageTables(
        shell_indices_half=jnp.asarray(shell_indices_half, dtype=jnp.int32),
        wavg_shell_indices=jnp.asarray(relion_wavg_rectangle.shell_indices, dtype=jnp.int32),
        wavg_scale_pixel_mask=jnp.asarray(scale_pixel_mask_rect_np, dtype=bool),
        translation_sqdist_ang=None,
    )

    Ft_y_total = jnp.zeros(recon_volume_size, dtype=recon_y_accum_dtype)
    Ft_ctf_total = jnp.zeros(recon_volume_size, dtype=recon_ctf_accum_dtype)
    max_adjoint_block_bytes = _max_adjoint_block_bytes_for_pass(device_memory_bytes)
    exact_positions_device = jnp.asarray(relion_wavg_rectangle.exact_positions, dtype=jnp.int32)
    rect_indices_device = jnp.asarray(relion_wavg_rectangle.centered_indices, dtype=jnp.int32)
    noise_variance_for_noise_device = jnp.asarray(noise_variance_for_noise)
    shell_indices_noise_device = jnp.asarray(shell_indices_noise, dtype=jnp.int32)
    recon_pixel_indices_device = jnp.asarray(recon_window_indices, dtype=jnp.int32)

    # ---- T16: per-image operands prepared once for the whole half ----------
    # The per-chunk preparation repeats this work for every chunk an image
    # appears in (image occupancy 0.38-0.66 at the early state) and was 71.5% of
    # the half's CUDA launches. The operands are per-image pure, so one pass
    # over the half produces the same rows; the chunk loop then gathers them.
    resident_operands = None
    warmup = None
    if _resident_operands_requested():
        _, _norm_high_shell_dtype = relion_powerclass_noise_dtypes(
            real_dtype=precision_policy.score_real_dtype,
            source_faithful_spectrum_norm=resolved_spectrum_norm,
        )
        operand_bytes = resident_half_operand_bytes(
            n_images=int(n_images),
            n_score_pixels=int(n_windowed),
            n_recon_pixels=int(n_recon_windowed),
            n_half_pixels=int(np.asarray(noise_variance_half).size),
            n_fine_trans=int(n_fine_trans),
            score_complex_bytes=np.dtype(precision_policy.score_complex_dtype).itemsize,
            real_bytes=np.dtype(precision_policy.score_real_dtype).itemsize,
            norm_high_shell_bytes=np.dtype(_norm_high_shell_dtype).itemsize,
        )
        budget_bytes = resident_operands_max_bytes(device_memory_bytes)
        if operand_bytes > budget_bytes:
            logger.info(
                "Resident pass-2 keeps the per-chunk operand preparation: one half's resident "
                "operands would take %.2f GiB against a %.2f GiB budget",
                operand_bytes / float(1024**3),
                budget_bytes / float(1024**3),
            )
        else:
            # ---- P4-J: compile the chunk programs while the operands prepare -
            # Everything the chunk programs are keyed on is decided by now, and
            # the preparation below is 2.4-5.8s of host-bound device dispatch
            # that leaves the compiler idle. Off unless asked for; a warm-up
            # that describes the wrong program costs its own compile time and
            # changes nothing else, so the two log lines after the block, not an
            # assertion, are what report it.
            warm_config = resolve_compile_ahead_config()
            warm_pool = CompileAheadPool(warm_config)
            warm_predicted = None
            with warm_pool:
                if warm_config.enabled:
                    # A warm-up must never fail a run. The pool swallows a
                    # failure on its helper thread; this covers the submission
                    # itself, which runs here on the main thread and reaches
                    # into the plan, the tables and the operand predictor.
                    try:
                        warm_t0 = time.time()
                        presence = resident_half_operand_presence(
                            relion_exact_bpref_operands=bool(
                                bucket_io_kwargs.get("relion_exact_bpref_operands")
                            ),
                            use_exact_relion_gaussian=use_exact_relion_gaussian,
                            accumulate_noise=accumulate_noise,
                            current_size=current_size,
                        )
                        warm_predicted = resident_half_operand_avals(
                            n_images=int(n_images),
                            n_score_pixels=int(n_windowed),
                            n_recon_pixels=int(n_recon_windowed),
                            n_half_pixels=int(np.asarray(noise_variance_half).size),
                            n_fine_trans=int(n_fine_trans),
                            score_complex_dtype=precision_policy.score_complex_dtype,
                            score_real_dtype=precision_policy.score_real_dtype,
                            acc_real_dtype=jnp.float64 if use_float64_scoring else jnp.float32,
                            norm_high_shell_dtype=_norm_high_shell_dtype,
                            has_recon_weight=presence.has_recon_weight,
                            has_direct_ctf_rfloat=presence.has_direct_ctf_rfloat,
                            has_highres_xi2=presence.has_highres_xi2,
                            has_relion_norm_high_shell=presence.has_relion_norm_high_shell,
                        )
                        warmup = _submit_resident_chunk_warmup(
                            warm_pool,
                            chunks=chunks,
                            tables=tables,
                            n_fine_trans=int(n_fine_trans),
                            half_operand_avals=warm_predicted,
                            stage_tables=_make_chunk_stage_tables(
                                projection_score_cache=projection_score_cache,
                                projection_recon_cache=projection_recon_cache,
                                projection_recon_abs2_cache=projection_recon_abs2_cache,
                                mstep_grid=mstep_grid,
                                coarse_parent_grid=coarse_parent_grid,
                                fine_translation_parent_device=fine_translation_parent_device,
                                half_weights=jnp.asarray(half_weights_windowed),
                                translation_angles=jnp.asarray(
                                    relion_score_translation_angles, dtype=jnp.float32
                                ),
                                full_to_compact=relion_score_full_to_compact,
                                noise_variance_for_noise=noise_variance_for_noise_device,
                                shell_indices_noise=shell_indices_noise_device,
                                exact_positions_device=exact_positions_device,
                                recon_pixel_indices=recon_pixel_indices_device,
                                relion_x_half_recon_indices=relion_x_half_recon_indices,
                                image_tables=image_tables,
                            ),
                            carry=(Ft_y_total, Ft_ctf_total, stats),
                            translation_angles=jnp.asarray(
                                relion_score_translation_angles, dtype=jnp.float32
                            ),
                            rect_indices=rect_indices_device,
                            exact_positions=exact_positions_device,
                            image_shape=image_shape,
                            spec_kwargs=dict(
                                n_fine_trans=n_fine_trans,
                                n_score_pixels=n_windowed,
                                n_recon_pixels=n_recon_windowed,
                                n_rect=n_rect,
                                mstep_block_rows=mstep_block_rows,
                                adaptive_fraction=adaptive_fraction,
                                current_size=current_size,
                                mstep_current_size=mstep_current_size,
                                image_shape=image_shape,
                                recon_volume_shape=recon_volume_shape,
                                max_adjoint_block_bytes=max_adjoint_block_bytes,
                                stats_config=stats_config,
                                use_rfloat_ctf_wavg=presence.has_direct_ctf_rfloat,
                                use_translate_sum_kernel=True,
                                bpref_recon_operand=presence.has_recon_weight,
                            ),
                            translation_prior_centers_np=translation_prior_centers_np,
                            fine_translations=fine_translations,
                            voxel_size=experiment_dataset.voxel_size,
                            default_translation_sqdist=image_tables.translation_sqdist_ang,
                        )
                        logger.info(
                            "Resident pass-2 compile-ahead: queued %d capacity classes %s "
                            "for the %s chunk path, host cost %.2fs",
                            len(warmup.classes),
                            ",".join(f"{r}x{b}" for r, b in warmup.classes) or "-",
                            warmup.path,
                            time.time() - warm_t0,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.info(
                            "Resident pass-2 compile-ahead could not be submitted (%s: %s); the chunk loop compiles its own programs",
                            type(exc).__name__,
                            exc,
                        )
                        warm_predicted = None
                        warmup = None
                operands_t0 = time.time()
                try:
                    resident_operands = prepare_resident_half_operands(
                        experiment_dataset,
                        np.arange(n_images, dtype=np.int64),
                        bucket_io_kwargs=bucket_io_kwargs,
                        window_indices=window_indices,
                        recon_window_indices=recon_window_indices,
                        image_shape=image_shape,
                        current_size=current_size,
                        n_fine_trans=int(n_fine_trans),
                        use_exact_relion_gaussian=use_exact_relion_gaussian,
                        accumulate_noise=accumulate_noise,
                        source_faithful_spectrum_norm=resolved_spectrum_norm,
                        fine_translation_prior_2d=fine_translation_prior_2d,
                        scale_corrections_np=scale_corrections_np,
                        group_ids_np=group_ids_np,
                        precision_policy=precision_policy,
                    )
                except ResidentOperandsUnsupported as reason:
                    logger.info(
                        "Resident pass-2 keeps the per-chunk operand preparation: %s", reason
                    )
                    resident_operands = None
                else:
                    if int(resident_operands.n_score_pixels) != int(n_windowed):
                        raise ValueError(
                            "resident score operand pixel count does not match the score window: "
                            f"{resident_operands.n_score_pixels} vs {int(n_windowed)}"
                        )
                    if int(resident_operands.n_recon_pixels) != int(n_recon_windowed):
                        raise ValueError(
                            "resident reconstruction operand pixel count does not match the "
                            f"reconstruction window: {resident_operands.n_recon_pixels} vs "
                            f"{int(n_recon_windowed)}"
                        )
                    logger.info(
                        "Resident pass-2 per-half operand preparation: %.2fs",
                        time.time() - operands_t0,
                    )
            # Leaving the block joined the helper: any compile still running
            # when the preparation finished was one the chunk loop was about to
            # wait for anyway.
            if warm_config.enabled:
                logger.info("Resident pass-2 %s", warm_pool.summary)
                for message in warm_pool.summary.errors:
                    logger.info("Resident pass-2 compile-ahead error: %s", message)
                # The prediction was made before the preparation and is compared
                # after it, so this cannot be satisfied by construction. A
                # mismatch means the warm-up described operands the loop will not
                # pass and its compiles were wasted; the run is unaffected.
                if warm_predicted is None:
                    pass
                elif resident_operands is None:
                    logger.info(
                        "Resident pass-2 compile-ahead predicted operands the half did not "
                        "prepare; its programs go unused"
                    )
                else:
                    difference = describe_resident_operand_mismatch(
                        warm_predicted, resident_operands
                    )
                    logger.info(
                        "Resident pass-2 compile-ahead operand prediction: %s",
                        difference if difference else "matches the prepared operands",
                    )
                    # The three spec booleans are the other thing the warm-up has
                    # to predict from configuration rather than read. They key the
                    # program, so getting one wrong warms a signature the loop
                    # never submits even when every operand aval is right.
                    spec_difference = describe_chunk_spec_prediction(
                        predicted_rfloat_ctf_wavg=presence.has_direct_ctf_rfloat,
                        predicted_bpref_recon_operand=presence.has_recon_weight,
                        predicted_translate_sum_kernel=True,
                        operands=resident_operands,
                    )
                    logger.info(
                        "Resident pass-2 compile-ahead spec prediction: %s",
                        spec_difference if spec_difference
                        else "matches the prepared operands",
                    )

    verify_operands = resident_operands is not None and _resident_operands_verify_enabled()

    # ---- chunk loop --------------------------------------------------------
    # With the warm-up on, collect the (program, spec) keys the loop actually
    # submits. Comparing them against what the helper compiled is the hit rate:
    # a key the loop used and the warm-up did not is a program the loop compiled
    # itself, which is what a mis-predicted spec looks like.
    submitted_keys = set() if warmup is not None else None
    loop_t0 = time.time()
    for chunk in chunks:
        Ft_y_total, Ft_ctf_total, stats = _run_resident_chunk(
            chunk,
            tables=tables,
            experiment_dataset=experiment_dataset,
            bucket_io_kwargs=bucket_io_kwargs,
            half_weights=jnp.asarray(half_weights_windowed),
            translation_angles=jnp.asarray(relion_score_translation_angles, dtype=jnp.float32),
            full_to_compact=relion_score_full_to_compact,
            n_score_pixels=int(n_windowed),
            fine_translation_prior_2d=fine_translation_prior_2d,
            score_real_dtype=precision_policy.score_real_dtype,
            projection_score_cache=projection_score_cache,
            projection_recon_cache=projection_recon_cache,
            projection_recon_abs2_cache=projection_recon_abs2_cache,
            fine_translation_parent_device=fine_translation_parent_device,
            mstep_grid=mstep_grid,
            coarse_parent_grid=coarse_parent_grid,
            n_fine_trans=n_fine_trans,
            n_recon_windowed=n_recon_windowed,
            n_rect=n_rect,
            mstep_block_rows=mstep_block_rows,
            adaptive_fraction=float(adaptive_fraction),
            windowed_prepare=windowed_prepare,
            window_indices=window_indices,
            recon_window_indices=recon_window_indices,
            relion_x_half_recon_indices=relion_x_half_recon_indices,
            exact_positions_device=exact_positions_device,
            rect_indices_device=rect_indices_device,
            recon_pixel_indices=recon_pixel_indices_device,
            resident_operands=resident_operands,
            verify_operands=verify_operands and chunk is chunks[0],
            image_shape=image_shape,
            current_size=current_size,
            mstep_current_size=mstep_current_size,
            recon_volume_shape=recon_volume_shape,
            max_adjoint_block_bytes=max_adjoint_block_bytes,
            noise_variance_for_noise=noise_variance_for_noise_device,
            shell_indices_noise=shell_indices_noise_device,
            group_ids_np=group_ids_np,
            scale_corrections_np=scale_corrections_np,
            translation_prior_centers_np=translation_prior_centers_np,
            fine_translations=fine_translations,
            voxel_size=experiment_dataset.voxel_size,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=resolved_spectrum_norm,
            stats=stats,
            stats_config=stats_config,
            image_tables=image_tables,
            Ft_y_total=Ft_y_total,
            Ft_ctf_total=Ft_ctf_total,
            cuda_backproject=cuda_backproject,
            submitted_keys=submitted_keys,
        )
    loop_s = time.time() - loop_t0
    if warmup is not None:
        used = submitted_keys or set()
        covered = used & warmup.keys
        missed = used - warmup.keys
        logger.info(
            "Resident pass-2 compile-ahead hit rate: %d of %d programs the chunk "
            "loop submitted were already compiled (%d warmed and unused)%s",
            len(covered),
            len(used),
            len(warmup.keys - used),
            "" if not missed
            else "; missed " + ", ".join(sorted(name for name, _ in missed)),
        )

    # ---- finalize (identical to the compact return block) ------------------
    Ft_y_total, Ft_ctf_total = enforce_half_volume_x0(
        Ft_y_total,
        Ft_ctf_total,
        recon_volume_shape,
        logger=logger,
        label="Resident pass-2",
    )
    Ft_y_total, Ft_ctf_total = relion_x_half_accumulators_to_public_layout(
        Ft_y_total,
        Ft_ctf_total,
        recon_volume_shape,
    )

    finalized = finalize_statistics(stats, config=stats_config)
    hard_assignment = np.asarray(finalized.hard_assignment, dtype=np.int32)
    best_fine_rotation_indices = np.asarray(finalized.best_fine_rotation_indices, dtype=np.int64)
    best_rotations = np.asarray(fine_rotations_override, dtype=precision_policy.score_real_dtype)[
        best_fine_rotation_indices
    ]
    best_translations = fine_translations[hard_assignment % n_fine_trans]
    best_eulers = None
    if return_source_eulers and fine_source_eulers_override is not None:
        best_eulers = np.asarray(fine_source_eulers_override, dtype=np.float64)[
            best_fine_rotation_indices
        ]

    merged_noise_stats = make_noise_stats(
        wsum_sigma2_noise=finalized.wsum_sigma2_noise,
        wsum_img_power=finalized.wsum_img_power,
        wsum_sigma2_offset=finalized.wsum_sigma2_offset,
        sumw=finalized.sumw,
        wsum_norm_correction=finalized.wsum_norm_correction,
        wsum_scale_correction_xa=finalized.wsum_scale_correction_xa,
        wsum_scale_correction_aa=finalized.wsum_scale_correction_aa,
    )
    relion_stats = None
    if return_stats:
        relion_stats = make_relion_stats(
            log_evidence_per_image=finalized.log_evidence_per_image,
            best_log_score_per_image=finalized.best_log_score_per_image,
            max_posterior_per_image=finalized.max_posterior_per_image,
            rotation_posterior_sums=finalized.rotation_posterior_sums,
        )
    logger.info(
        "Resident pass-2: %d images, %d chunks, %.2fs chunk loop, %.2fs total",
        n_images,
        len(chunks),
        loop_s,
        time.time() - overall_t0,
    )
    return SparsePass2Output(
        Ft_y_total,
        Ft_ctf_total,
        hard_assignment,
        best_rotations,
        best_translations,
        best_fine_rotation_indices,
        relion_stats=relion_stats,
        score_log_z=(
            finalized.score_log_z_per_image if (return_stats and return_score_log_z) else None
        ),
        noise_stats=merged_noise_stats,
        source_eulers=best_eulers if return_source_eulers else None,
    )


def _chunk_segment_offsets(tables, chunk, n_fine_trans: int) -> np.ndarray:
    """Cell offsets of each chunk image slot, in the segmented handler's units.

    Slot ``b`` of the chunk owns the cells of image ``image_start + b``; padded
    slots and the rows past ``n_valid_rows`` are covered by no segment, which
    the handler treats exactly as an all ``-inf`` rectangular row.
    """

    image_capacity = int(chunk.image_capacity)
    n_valid_images = int(chunk.n_valid_images)
    offsets = np.full(image_capacity + 1, chunk.n_valid_rows * int(n_fine_trans), dtype=np.int64)
    starts = (
        np.asarray(
            tables.row_offsets[chunk.image_start : chunk.image_start + n_valid_images + 1],
            dtype=np.int64,
        )
        - int(chunk.row_start)
    ) * int(n_fine_trans)
    offsets[: n_valid_images + 1] = starts
    if int(offsets[-1]) > int(chunk.row_capacity) * int(n_fine_trans):
        raise ValueError("chunk segment offsets exceed the chunk's cell capacity")
    return offsets.astype(np.int32)


class _Placement(NamedTuple):
    """How a chunk constructor turns host values into program inputs.

    The chunk loop places them on the device; the compile-ahead warm-up places
    their shape and dtype only. Having one constructor with two placements
    rather than two constructors means a warm-up cannot describe a program the
    loop does not run, which is the failure a hit rate would only report after
    the compile time had already been spent.
    """

    array: object
    scalar: object


_PLACE_ON_DEVICE = _Placement(
    array=lambda value, dtype: jnp.asarray(value, dtype=dtype),
    # ``np.asarray`` first: a NumPy *scalar* reaches the device through one
    # eager ``convert_element_type`` per chunk, a 0-d NumPy *array* of the same
    # dtype through a plain transfer. Same dtype, shape, weak type and value
    # either way.
    scalar=lambda value, dtype: _scalar_operand(value, dtype),
)

_PLACE_AS_AVAL = _Placement(
    array=lambda value, dtype: jax.ShapeDtypeStruct(np.shape(value), jnp.dtype(dtype)),
    scalar=lambda value, dtype: jax.ShapeDtypeStruct((), jnp.dtype(dtype)),
)


def _make_chunk_row_arrays(tables, chunk, n_fine_trans, *, place) -> _ChunkRowArrays:
    """One chunk's row-aligned inputs, on the device or as avals.

    Everything here is host NumPy over the plan, so the aval placement costs
    only the materialize and does no device work at all. The shapes are the
    chunk's capacity class and nothing else, which
    ``tests/unit/test_chunk_row_avals.py`` pins: that is why warming one chunk
    per class covers every chunk of that class.
    """

    image_capacity = int(chunk.image_capacity)
    host_chunk = materialize_chunk(tables, chunk)
    segment_offsets_np = _chunk_segment_offsets(tables, chunk, n_fine_trans)
    image_row_start_np = segment_offsets_np.astype(np.int64)[:image_capacity] // int(n_fine_trans)
    image_row_count_np = (
        segment_offsets_np.astype(np.int64)[1:] - segment_offsets_np.astype(np.int64)[:-1]
    ) // int(n_fine_trans)
    return _ChunkRowArrays(
        row_image_local=place.array(host_chunk["row_image_local"], jnp.int32),
        row_fine_rot=place.array(host_chunk["row_fine_rot"], jnp.int32),
        row_log_prior=place.array(host_chunk["row_log_prior"], jnp.float32),
        row_mask_bits=place.array(host_chunk["row_mask_bits"], jnp.uint32),
        row_mask_mode=place.array(host_chunk["row_mask_mode"], jnp.int8),
        image_ids=place.array(host_chunk["image_ids"], jnp.int32),
        n_valid_rows=place.scalar(host_chunk["n_valid_rows"], jnp.int32),
        n_valid_images=place.scalar(host_chunk["n_valid_images"], jnp.int32),
        segment_offsets=place.array(segment_offsets_np, jnp.int32),
        image_row_start=place.array(image_row_start_np, jnp.int64),
        image_row_count=place.array(image_row_count_np, jnp.int64),
    )


def _make_chunk_translation_sqdist(
    default,
    *,
    translation_prior_centers_np,
    image_indices,
    image_capacity,
    n_valid_images,
    fine_translations,
    voxel_size,
):
    """The chunk's per-image prior squared distances, at image capacity.

    A per-image host table built at capacity keeps the program keyed on the
    capacity class and takes the eager device ops out of the chunk loop. Padded
    slots multiply a zero posterior, so their value is never observable; they
    are zeroed anyway.
    """

    if translation_prior_centers_np is None:
        return default
    image_capacity = int(image_capacity)
    padded_image_indices = _pad_batch_to_capacity(
        np.asarray(image_indices).reshape(-1, 1), image_capacity
    ).reshape(-1)
    centers = translation_prior_centers_for_images(
        translation_prior_centers_np,
        padded_image_indices,
        batch_size=image_capacity,
    )
    sqdist_np = np.asarray(translation_sqdist_angstrom(fine_translations, centers, voxel_size))
    sqdist_np = np.where(
        (np.arange(image_capacity) < int(n_valid_images))[:, None], sqdist_np, 0.0
    )
    return jnp.asarray(sqdist_np)


def _make_chunk_stage_operands(recon, translation_sqdist_ang) -> _ChunkStageOperands:
    """Name one chunk's operands out of whatever produced them.

    ``recon`` is the per-chunk preparation's dict, the once-per-half gather's
    dict, or -- for the compile-ahead warm-up -- the same gather's output under
    ``jax.eval_shape``, which is a dict of the same keys holding avals.
    """

    return _ChunkStageOperands(
        score_input=recon["score_input"],
        corr_img_score=recon["corr_img_score"],
        highres_xi2_half=recon["highres_xi2_half"],
        translation_prior=recon["translation_prior"],
        shifted_recon=recon.get("shifted_recon"),
        shifted_noise=recon.get("shifted_noise"),
        recon_image=recon.get("recon_image"),
        recon_weight=recon.get("recon_weight"),
        noise_image=recon.get("noise_image"),
        ctf2_over_nv_recon=recon["ctf2_over_nv_recon"],
        direct_ctf_rfloat_recon=recon["direct_ctf_rfloat_recon"],
        processed_image_half=recon["processed_image_half"],
        relion_norm_high_shell=recon["relion_norm_high_shell"],
        raw_translated_wavg_rectangle=recon["raw_translated_wavg_rectangle"],
        raw_translated_wavg_for_atomic=recon["raw_translated_wavg_for_atomic"],
        scale=recon["scale"],
        group_ids=recon["group_ids"],
        translation_sqdist_ang=translation_sqdist_ang,
    )


def chunk_program_path() -> str:
    """Which programs a chunk of this half will actually submit.

    ``"fused"`` is the opt-in single chunk program, ``"per-stage"`` the default
    path's three glue programs, ``"eager"`` the loose dispatch with no program
    to warm. The compile-ahead warm-up reads this so it cannot warm a path the
    loop does not take: warming the fused program while the loop runs the
    per-stage one is not an error, it simply buys nothing, and it did exactly
    that until 2026-09-20.
    """

    if _chunk_jit_enabled():
        return "fused"
    if _resident_glue_jit_enabled():
        return "per-stage"
    return "eager"


class _ChunkWarmup(NamedTuple):
    """What the warm-up queued, in the terms the chunk loop can be compared in."""

    classes: tuple
    keys: frozenset
    path: str


def describe_chunk_spec_prediction(
    *,
    predicted_rfloat_ctf_wavg,
    predicted_bpref_recon_operand,
    predicted_translate_sum_kernel,
    operands,
) -> str:
    """Name every spec boolean the warm-up predicted differently from the loop.

    Returns an empty string when they agree. The loop reads these three from the
    prepared operands; the warm-up has to predict them from configuration before
    the preparation runs, so this is the same predict-early verify-late check the
    operand tree gets, applied to the part of the program key that is not an aval.
    """

    if operands is None:
        return "the half prepared no resident operands, so no spec was used"
    problems = []
    for name, predicted, actual in (
        ("use_rfloat_ctf_wavg", bool(predicted_rfloat_ctf_wavg),
         operands.direct_ctf_rfloat_recon is not None),
        ("bpref_recon_operand", bool(predicted_bpref_recon_operand),
         operands.recon_weight is not None),
        ("use_translate_sum_kernel", bool(predicted_translate_sum_kernel), True),
    ):
        if predicted != actual:
            problems.append(f"{name}: predicted {predicted}, really {actual}")
    return "; ".join(problems)


def chunk_program_keys(path: str, spec) -> frozenset:
    """The ``(program name, spec)`` keys a chunk of ``spec`` will submit.

    A compiled program is identified by its function and its static argument,
    so this is the unit in which "what the warm-up compiled" and "what the loop
    ran" are the same kind of thing. Comparing capacity classes alone would
    miss a spec that differs in one of the booleans the warm-up has to predict
    from configuration, which is a real way for a warmed program to go unused.
    """

    return frozenset(
        (program.__name__, spec) for program in chunk_programs_for_path(path)
    )


def chunk_programs_for_path(path: str) -> tuple:
    """The jitted programs a chunk will submit on ``path``.

    One list, read by the compile-ahead warm-up and asserted against the
    runners by `tests/unit/test_em_compile_ahead_consumer.py`. Warming a
    program the runner does not submit, or missing one it does, costs compile
    time and buys nothing; that happened once, on the fused-versus-per-stage
    split, and was found by reading a census rather than by a test.
    """

    if path == "fused":
        return (_run_resident_chunk_program,)
    if path == "per-stage":
        return (
            _resident_chunk_posterior_program,
            _resident_mstep_block_program,
            _resident_chunk_statistics_program,
        )
    if path == "eager":
        return ()
    raise ValueError(f"unknown chunk program path {path!r}")


def _make_mstep_block_inputs(rows, posterior) -> "_MstepBlockInputs":
    """The M-step block program's row inputs, from the chunk and its posterior.

    Shared with the compile-ahead warm-up so the warmed signature is the one
    the per-stage loop submits. ``projections`` is None here: the block program
    gathers them from the tables itself.
    """

    return _MstepBlockInputs(
        row_image_local=rows.row_image_local,
        kernel_row_image_ids=posterior.kernel_row_image_ids,
        row_posterior=posterior.row_posterior,
        row_fine_rot=rows.row_fine_rot,
        projections=None,
    )


def _make_chunk_program_spec(
    *,
    row_capacity,
    image_capacity,
    n_fine_trans,
    n_score_pixels,
    n_recon_pixels,
    n_rect,
    mstep_block_rows,
    adaptive_fraction,
    current_size,
    mstep_current_size,
    image_shape,
    recon_volume_shape,
    max_adjoint_block_bytes,
    stats_config,
    use_rfloat_ctf_wavg,
    use_translate_sum_kernel,
    bpref_recon_operand,
) -> _ChunkProgramSpec:
    """The static key of one chunk program.

    The last four environment-read fields are the reason this is a function
    and not a literal at each call site: a warm-up that read them at a
    different moment, or not at all, would key its program differently from the
    loop's and warm nothing.
    """

    return _ChunkProgramSpec(
        row_capacity=int(row_capacity),
        image_capacity=int(image_capacity),
        n_fine_trans=int(n_fine_trans),
        n_score_pixels=int(n_score_pixels),
        n_recon_pixels=int(n_recon_pixels),
        n_rect=int(n_rect),
        mstep_block_rows=int(mstep_block_rows),
        adaptive_fraction=float(adaptive_fraction),
        current_size=int(current_size),
        mstep_current_size=int(mstep_current_size),
        image_shape=tuple(int(v) for v in image_shape),
        recon_volume_shape=tuple(int(v) for v in recon_volume_shape),
        max_adjoint_block_bytes=int(max_adjoint_block_bytes),
        stats_config=stats_config,
        use_rfloat_ctf_wavg=bool(use_rfloat_ctf_wavg),
        use_translate_sum_kernel=bool(use_translate_sum_kernel),
        bpref_recon_operand=bool(bpref_recon_operand),
        kernel_ctf_probs=_kernel_ctf_probs_enabled(),
        wavg_power_per_image=_wavg_power_per_image_enabled(),
        block_unroll=_chunk_block_unroll(),
        static_block_trip=_chunk_static_block_trip_enabled(),
    )


def _submit_resident_chunk_warmup(
    pool,
    *,
    chunks,
    tables,
    n_fine_trans,
    half_operand_avals,
    stage_tables,
    carry,
    translation_angles,
    rect_indices,
    exact_positions,
    image_shape,
    spec_kwargs,
    translation_prior_centers_np,
    fine_translations,
    voxel_size,
    default_translation_sqdist,
):
    """Queue one chunk program per capacity class the plan will run.

    Called in the window between the admission check and the per-half operand
    preparation. The preparation is 2.4-5.8 s of host-bound device dispatch on
    the main thread, and the chunk programs the loop will need after it are
    fully determined by then: the capacity classes are in ``chunks``, the
    iteration-global tables exist, and the operands the programs consume are
    described by ``half_operand_avals`` without being prepared.

    Nothing here can change a result. The warm-up hands the helper thread
    shape/dtype stand-ins only, and if it describes a program the loop does not
    run, the loop compiles its own as before; the cost is the wasted warm-up and
    the log line below is how that is noticed.

    Returns the capacity classes submitted, for the hit-rate line.
    """

    import dataclasses

    from recovar.em.sparse_pass2.resident_operands import ResidentHalfOperands

    # Warm the programs the configured path will actually submit. The fused
    # chunk program is opt-in and off by default; the per-stage path with the
    # glue JIT on is what production runs, and it is three programs. Warming the
    # wrong one is not an error, it is simply useless, so the path is decided
    # here, before any work, and named in the log line.
    path = chunk_program_path()
    use_chunk_jit = path == "fused"
    if path == "eager":
        return ()

    def as_aval(value):
        if value is None:
            return None
        return jax.ShapeDtypeStruct(
            tuple(int(d) for d in np.shape(value)), jnp.dtype(value.dtype)
        )

    expected_programs = chunk_programs_for_path(path)

    def _checked(work):
        # The warm-up must submit exactly the programs the runner for this path
        # submits. Raising here lands in the pool's own error record, so a
        # mismatch is reported and the run is untouched.
        got = tuple(program for program, _, _ in work)
        if set(got) != set(expected_programs):
            raise ValueError(
                "compile-ahead would warm "
                f"{sorted(p.__name__ for p in got)} on the {path} path, but that path "
                f"runs {sorted(p.__name__ for p in expected_programs)}"
            )
        return work

    table_avals = _ChunkStageTables(*(as_aval(v) for v in stage_tables))
    carry_avals = jax.tree_util.tree_map(as_aval, carry)
    angle_aval = as_aval(translation_angles)
    rect_aval = as_aval(rect_indices)
    exact_aval = as_aval(exact_positions)

    names = [
        f.name for f in dataclasses.fields(ResidentHalfOperands) if not f.name.startswith("n_")
    ]
    present = [n for n in names if getattr(half_operand_avals, n) is not None]
    scalars = {
        f.name: getattr(half_operand_avals, f.name)
        for f in dataclasses.fields(ResidentHalfOperands)
        if f.name.startswith("n_")
    }
    present_avals = [getattr(half_operand_avals, n) for n in present]

    submitted = []
    submitted_keys = set()
    for chunk in chunks:
        capacity_class = (int(chunk.row_capacity), int(chunk.image_capacity))
        if capacity_class in submitted:
            continue
        row_capacity, image_capacity = capacity_class
        row_avals = _make_chunk_row_arrays(tables, chunk, n_fine_trans, place=_PLACE_AS_AVAL)
        sqdist = _make_chunk_translation_sqdist(
            default_translation_sqdist,
            translation_prior_centers_np=translation_prior_centers_np,
            image_indices=np.arange(chunk.image_start, chunk.image_stop, dtype=np.int64),
            image_capacity=image_capacity,
            n_valid_images=chunk.n_valid_images,
            fine_translations=fine_translations,
            voxel_size=voxel_size,
        )
        spec = _make_chunk_program_spec(
            row_capacity=row_capacity, image_capacity=image_capacity, **spec_kwargs
        )

        def thunk(_row=row_avals, _spec=spec, _images=image_capacity, _sq=as_aval(sqdist)):
            # The chunk operands are predicted by tracing the real gather, not
            # by a second constructor: the gather's own shape validation runs on
            # the way through, and there is no place for the two to disagree.
            def gather(slots, angles, rect, exact, *arrays):
                fields = dict(scalars)
                fields.update({name: None for name in names})
                fields.update(dict(zip(present, arrays)))
                return gather_resident_chunk_operands(
                    ResidentHalfOperands(**fields),
                    slots,
                    translation_angles=angles,
                    rect_indices=rect,
                    exact_positions=exact,
                    image_shape=image_shape,
                )

            recon = jax.eval_shape(
                gather,
                jax.ShapeDtypeStruct((_images,), jnp.int32),
                angle_aval,
                rect_aval,
                exact_aval,
                *present_avals,
            )
            operand_avals = _make_chunk_stage_operands(recon, _sq)
            if use_chunk_jit:
                return _checked(
                    [
                        (
                            _run_resident_chunk_program,
                            (_row, operand_avals, table_avals, carry_avals),
                            {"spec": _spec},
                        )
                    ]
                )
            # The per-stage path is the default, and it runs three programs.
            # Their later inputs are earlier stages' outputs, so they are taken
            # by tracing those stages rather than described a second time.
            posterior_avals = jax.eval_shape(
                partial(_resident_chunk_posterior_program, spec=_spec),
                _row,
                operand_avals,
                table_avals,
            )
            mstep_avals = jax.eval_shape(
                partial(_initial_mstep_carry, spec=_spec),
                carry_avals[0],
                carry_avals[1],
                operand_avals,
                table_avals,
            )
            block_avals = _make_mstep_block_inputs(_row, posterior_avals)
            return _checked([
                (
                    _resident_chunk_posterior_program,
                    (_row, operand_avals, table_avals),
                    {"spec": _spec},
                ),
                (
                    _resident_mstep_block_program,
                    (
                        jax.ShapeDtypeStruct((), jnp.int32),
                        block_avals,
                        operand_avals,
                        table_avals,
                        mstep_avals,
                    ),
                    {"spec": _spec},
                ),
                (
                    _resident_chunk_statistics_program,
                    (
                        carry_avals[2],
                        _row,
                        operand_avals,
                        table_avals,
                        posterior_avals,
                        mstep_avals,
                    ),
                    {"spec": _spec},
                ),
            ])

        if pool.submit_thunk(f"resident chunk rows={row_capacity} images={image_capacity}", thunk):
            submitted.append(capacity_class)
            submitted_keys.update(chunk_program_keys(path, spec))
    return _ChunkWarmup(classes=tuple(submitted), keys=frozenset(submitted_keys), path=path)


def _make_chunk_stage_tables(
    *,
    projection_score_cache,
    projection_recon_cache,
    projection_recon_abs2_cache,
    mstep_grid,
    coarse_parent_grid,
    fine_translation_parent_device,
    half_weights,
    translation_angles,
    full_to_compact,
    noise_variance_for_noise,
    shell_indices_noise,
    exact_positions_device,
    recon_pixel_indices,
    relion_x_half_recon_indices,
    image_tables,
) -> _ChunkStageTables:
    """Assemble the iteration-global tables every chunk of a half reads.

    One builder, called by the chunk loop with the real arrays and by the
    compile-ahead warm-up with their shape/dtype stand-ins. Assembling the
    tuple twice would be a place for the warm-up to drift from the loop: a
    warmed program with one field's dtype wrong is never used, which costs
    compile time and is invisible unless someone reads the hit rate.
    """

    return _ChunkStageTables(
        projection_score_cache=projection_score_cache,
        projection_recon_cache=projection_recon_cache,
        projection_recon_abs2_cache=projection_recon_abs2_cache,
        mstep_grid=mstep_grid,
        coarse_parent_grid=coarse_parent_grid,
        fine_translation_parent=fine_translation_parent_device,
        half_weights=half_weights,
        translation_angles=translation_angles,
        full_to_compact=full_to_compact,
        noise_variance_for_noise=noise_variance_for_noise,
        shell_indices_noise=shell_indices_noise,
        exact_positions=exact_positions_device,
        recon_pixel_indices=recon_pixel_indices,
        relion_x_half_recon_indices=relion_x_half_recon_indices,
        shell_indices_half=image_tables.shell_indices_half,
        wavg_shell_indices=image_tables.wavg_shell_indices,
        wavg_scale_pixel_mask=image_tables.wavg_scale_pixel_mask,
    )


def _prepare_chunk_reconstruction_operands(
    *,
    chunk,
    image_indices,
    experiment_dataset,
    bucket_io_kwargs,
    windowed_prepare,
    recon_window_indices,
    score_window_indices,
    fine_translation_prior_2d,
    score_real_dtype,
    n_fine_trans,
    n_recon_windowed,
    image_shape,
    current_size,
    use_exact_relion_gaussian,
    accumulate_noise,
    source_faithful_spectrum_norm,
    relion_score_translation_angles,
    rect_indices_device,
    exact_positions_device,
    scale_corrections_np,
    group_ids_np,
):
    """Build one chunk's translated reconstruction, noise and Wavg tiles.

    The oracle path, kept selectable by
    ``RECOVAR_SPARSE_PASS2_RESIDENT_OPERANDS=0``. These tiles carry the
    ``(images, translations, pixels)`` axis, so they are the one operand family
    that cannot be kept resident for a whole half (17 GiB at the hp3 state);
    they are rebuilt per chunk from the same :func:`_prepare_bucket_io` call,
    with the same keyword arguments, that the compact engine makes per bucket.
    The default path instead keeps the *unshifted* per-image operands resident
    (:mod:`recovar.em.sparse_pass2.resident_operands`) and lets T15's kernel
    apply the translations inside the M-step reduction, so no tile is built at
    all.

    Shape stability. The batch handed to ``_prepare_bucket_io`` is padded on
    the host to the chunk's image capacity before anything is traced, so every
    chunk of one capacity class runs the same program instead of one program
    per distinct occupancy. The padded slots carry a duplicate image's real
    data through preparation and are zeroed afterwards by a capacity-shaped
    mask. Nothing downstream reads them: their posterior rows are zero and
    their image ids are -1.
    """

    image_capacity = int(chunk.image_capacity)
    n_valid_images = int(chunk.n_valid_images)
    image_indices = np.asarray(image_indices)

    batch_data, ctf_params, fetched_indices = fetch_indexed_batch(
        experiment_dataset, image_indices
    )
    order = _reorder_permutation(fetched_indices, image_indices, image_capacity)
    prepared = _prepare_bucket_io(
        experiment_dataset,
        jnp.asarray(_pad_batch_to_capacity(batch_data, image_capacity)),
        _pad_batch_to_capacity(ctf_params, image_capacity),
        _pad_batch_to_capacity(np.asarray(fetched_indices), image_capacity),
        return_direct_scoring_io=True,
        **bucket_io_kwargs,
    )
    (
        _shifted_score_half,
        shifted_recon_half,
        _batch_norm,
        ctf2_over_nv_half,
        ctf2_over_nv_half_with_dc,
        shifted_score_half_with_dc,
        processed_score_half_for_noise,
        _shifted_corrected_score_half,
        direct_score_input,
        _direct_preprocessed_score_input,
        _direct_pixel_correction,
        _direct_preprocess_normalization_factors,
        _direct_integer_pre_shifts,
        _direct_batch_image_corrections,
        _direct_batch_scale_corrections,
        _direct_inverse_noise_half,
        direct_ctf_rfloat_half,
    ) = prepared

    gather_recon = jnp.asarray(recon_window_indices, dtype=jnp.int32)
    if windowed_prepare:
        shifted_recon = shifted_recon_half
        ctf2_over_nv_recon = ctf2_over_nv_half_with_dc
        shifted_noise = shifted_score_half_with_dc
    else:
        shifted_recon = shifted_recon_half[:, gather_recon]
        ctf2_over_nv_recon = ctf2_over_nv_half_with_dc[:, gather_recon]
        shifted_noise = shifted_score_half_with_dc[:, gather_recon]
    direct_ctf_rfloat_recon = (
        None if direct_ctf_rfloat_half is None else direct_ctf_rfloat_half[:, gather_recon]
    )

    # Score-side operands come from this same call. The driver used to make a
    # second pass over the whole half for them; the preparation is per-image
    # pure (bitwise at batch 256, 128, 32, 13 and 1), so taking them here is
    # the same arithmetic with one call per chunk instead of two per image.
    if windowed_prepare:
        score_input = direct_score_input
        corr_img_score = ctf2_over_nv_half
    else:
        gather_score = jnp.asarray(score_window_indices, dtype=jnp.int32)
        score_input = direct_score_input[:, gather_score]
        corr_img_score = ctf2_over_nv_half[:, gather_score]

    highres_xi2_half, relion_norm_high_shell = _relion_powerclass_noise_terms(
        processed_score_half_for_noise,
        image_shape=image_shape,
        current_size=current_size,
        use_exact_relion_gaussian=use_exact_relion_gaussian,
        accumulate_noise=accumulate_noise,
        source_faithful_spectrum_norm=source_faithful_spectrum_norm,
    )
    raw_translated_wavg_rectangle = _relion_cuda_translate_wavg_norm_images(
        processed_score_half_for_noise,
        relion_score_translation_angles,
        rect_indices_device,
        image_shape,
    )

    permutation = jnp.asarray(order, dtype=jnp.int32)
    valid_images = jnp.asarray(
        np.arange(image_capacity) < n_valid_images, dtype=bool
    )

    (
        score_input,
        corr_img_score,
        highres_xi2_half,
        shifted_recon,
        shifted_noise,
        ctf2_over_nv_recon,
        direct_ctf_rfloat_recon,
        processed_image_half,
        relion_norm_high_shell,
        raw_translated_wavg_rectangle,
        raw_translated_wavg_for_atomic,
    ) = _chunk_operand_rows(
        _ChunkOperandRowInputs(
            score_input=score_input,
            corr_img_score=corr_img_score,
            highres_xi2_half=highres_xi2_half,
            shifted_recon=shifted_recon,
            shifted_noise=shifted_noise,
            ctf2_over_nv_recon=ctf2_over_nv_recon,
            direct_ctf_rfloat_recon=direct_ctf_rfloat_recon,
            processed_score_half_for_noise=processed_score_half_for_noise,
            relion_norm_high_shell=relion_norm_high_shell,
            raw_translated_wavg_rectangle=raw_translated_wavg_rectangle,
        ),
        permutation,
        valid_images,
        exact_positions_device,
        image_capacity=int(image_capacity),
        n_fine_trans=int(n_fine_trans),
    )
    if int(shifted_recon.shape[-1]) != int(n_recon_windowed):
        raise ValueError(
            "reconstruction tile pixel count does not match the reconstruction window: "
            f"{int(shifted_recon.shape[-1])} vs {int(n_recon_windowed)}"
        )

    # Padded slots keep scale 1 so the Wavg kernel never divides by zero; their
    # posterior is zero, so the value is never observable.
    scale_chunk = np.ones(image_capacity, dtype=np.float32)
    if scale_corrections_np is not None:
        scale_chunk[:n_valid_images] = np.asarray(
            scale_corrections_np[image_indices], dtype=np.float32
        )
    group_ids_chunk = np.full(image_capacity, -1, dtype=np.int32)
    if group_ids_np is not None:
        group_ids_chunk[:n_valid_images] = np.asarray(
            group_ids_np[image_indices], dtype=np.int32
        )

    translation_prior = jnp.asarray(
        np.zeros((image_capacity, int(n_fine_trans)), dtype=np.float32)
        if fine_translation_prior_2d is None
        else _pad_batch_to_capacity(
            np.asarray(fine_translation_prior_2d)[image_indices], image_capacity
        ),
        dtype=score_real_dtype,
    )

    return {
        "score_input": score_input,
        "corr_img_score": corr_img_score,
        "highres_xi2_half": highres_xi2_half,
        "translation_prior": _zero_padded_images(translation_prior, valid_images),
        "shifted_recon": shifted_recon,
        "shifted_noise": shifted_noise,
        "ctf2_over_nv_recon": ctf2_over_nv_recon,
        "direct_ctf_rfloat_recon": direct_ctf_rfloat_recon,
        "processed_image_half": processed_image_half,
        "relion_norm_high_shell": relion_norm_high_shell,
        "raw_translated_wavg_rectangle": raw_translated_wavg_rectangle,
        "raw_translated_wavg_for_atomic": raw_translated_wavg_for_atomic,
        "scale": jnp.asarray(scale_chunk),
        "group_ids": jnp.asarray(group_ids_chunk),
    }


def _chunk_timing_enabled() -> bool:
    """Whether to log per-chunk occupancy, block count and synchronised wall."""

    return parse_env_flag(_CHUNK_TIMING_ENV, default=False)


def _chunk_jit_enabled() -> bool:
    """Whether the chunk body runs as one jitted program (T14).

    Opt-in: ``RECOVAR_SPARSE_PASS2_RESIDENT_CHUNK_JIT=1`` selects it, and the
    per-stage path is both the default and the oracle. Both paths call the same
    stage helpers on the same operands, so only the JIT boundary and the M-step
    loop's trip mechanism differ.

    Measured at 0bedf7672 on one H100 (jobs 14147789 hp3, 14147877 early,
    14147878 end-to-end). The program removes the chunk body's eager dispatch
    (267 -> 2 per chunk at hp3, 740 -> 2 at early) and 77% of its XLA glue
    launches, and the warm chunk loop is 0.9% (hp3) to 4.2% (early) faster. It
    is off by default because the end-to-end gate does not hold: with a cold
    persistent cache the fused program compiles once per (capacity class, pixel
    class) inside the chunk loop, and over the 16 iterations the 10k run took
    with an identical trajectory that cost 22.5 s (+1.9%), 84% of it in the two
    iterations that introduced a new pixel class. Turn it on with a warm
    persistent cache, or after that first-use compile is cheaper.
    """

    return parse_env_flag(_CHUNK_JIT_ENV, default=False)


def _chunk_block_unroll() -> int:
    """M-step blocks emitted per device-loop iteration; 1 keeps today's loop.

    XLA:GPU executes every ``while`` iteration by copying the loop predicate to
    the host and synchronizing, so the device loop drains the pipeline once per
    iteration: the hp3 nsys arm of job 14147789 charges 264 such
    ``cuStreamSynchronize`` calls, 5.94 s, to one warm half, and the early arm
    charges 1.88 ms to each of 1867 blocks. An unroll of ``u`` divides that by
    ``u`` and multiplies the live pixel-axis transients by ``u``, so it is a
    memory trade, not a free one; emitting every block cost 54.5 GiB.
    """

    raw = os.environ.get(_CHUNK_BLOCK_UNROLL_ENV, "").strip()
    if not raw:
        return 1
    value = int(raw)
    if value < 1:
        raise ValueError(f"{_CHUNK_BLOCK_UNROLL_ENV} must be >= 1, got {value}")
    return value


def _kernel_ctf_probs_enabled() -> bool:
    """Whether the M-step's ``ctf_probs`` comes from the kernel's fourth output."""

    return parse_env_flag(_KERNEL_CTF_PROBS_ENV, default=False)


def _wavg_power_per_image_enabled() -> bool:
    """Whether the Wavg rectangle is squared per image rather than per row.

    Default **on** since the P4-F/P4-G merge. P4-G measured the two forms
    bitwise on GPU at production shapes, 0 ULP over 6.3 million values, and the
    hoisted form 2.2 s per steady hp3 iteration faster;
    ``RECOVAR_SPARSE_PASS2_RESIDENT_WAVG_POWER_PER_IMAGE=0`` restores the
    per-row square as the oracle that equality is measured against.
    """

    return parse_env_flag(_WAVG_POWER_PER_IMAGE_ENV, default=True)


def _resident_operands_verify_enabled() -> bool:
    """Whether to check the first chunk of a half against the per-chunk path."""

    return parse_env_flag(_RESIDENT_OPERANDS_VERIFY_ENV, default=False)


def _verify_resident_chunk_operands(
    resident_recon,
    reference_recon,
    *,
    translation_angles,
    recon_pixel_indices,
    image_shape,
    n_recon_pixels: int,
    bpref_recon_operand: bool,
    label: str,
    cuda_backproject,
) -> None:
    """Prove one real chunk's resident operands equal the per-chunk preparation.

    The two paths differ in *when* the translation is applied, so the check
    translates the resident per-image operands with the primitives
    ``_prepare_bucket_io`` uses and compares against its own pre-shifted tiles.
    A mismatch raises: this runs only in a diagnostic arm, and a silent
    difference here would be a changed reconstruction operand.
    """

    image_capacity = int(np.asarray(reference_recon["shifted_recon"]).shape[0])
    angles = jnp.asarray(translation_angles, dtype=jnp.float32)
    indices = jnp.asarray(recon_pixel_indices, dtype=jnp.int32)
    if bpref_recon_operand:
        translated_recon = cuda_backproject.relion_translate_bpref_f32(
            jnp.asarray(resident_recon["recon_image"], dtype=jnp.complex64),
            jnp.asarray(resident_recon["recon_weight"], dtype=jnp.float32),
            angles,
            indices,
            image_shape,
        )
    else:
        translated_recon = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(resident_recon["recon_image"], dtype=jnp.complex64),
            angles,
            indices,
            image_shape,
        )
    translated_noise = cuda_backproject.relion_translate_score_f32(
        jnp.asarray(resident_recon["noise_image"], dtype=jnp.complex64),
        angles,
        indices,
        image_shape,
    )
    n_fine_trans = int(angles.shape[0])
    checks = {
        "shifted_recon": (
            np.asarray(translated_recon).reshape(image_capacity, n_fine_trans, n_recon_pixels),
            np.asarray(reference_recon["shifted_recon"]),
        ),
        "shifted_noise": (
            np.asarray(translated_noise).reshape(image_capacity, n_fine_trans, n_recon_pixels),
            np.asarray(reference_recon["shifted_noise"]),
        ),
    }
    for name in (
        "score_input",
        "corr_img_score",
        "highres_xi2_half",
        "translation_prior",
        "ctf2_over_nv_recon",
        "direct_ctf_rfloat_recon",
        "processed_image_half",
        "relion_norm_high_shell",
        "raw_translated_wavg_rectangle",
        "raw_translated_wavg_for_atomic",
        "scale",
        "group_ids",
    ):
        expected = reference_recon.get(name)
        actual = resident_recon.get(name)
        if expected is None and actual is None:
            continue
        if (expected is None) != (actual is None):
            raise AssertionError(f"resident operand {name} presence differs from the per-chunk path")
        checks[name] = (np.asarray(actual), np.asarray(expected))

    # ``relion_norm_high_shell`` bins the image power with a scatter-add over
    # duplicate indices. That races: two calls on the same array in one process
    # differ by about 6e-8 relative, so no two preparations of it are bitwise,
    # including two of the per-chunk path. It is checked against that spread
    # here, and exactly under ``RECOVAR_EM_DETERMINISTIC_REDUCTIONS=1``, where
    # the binning becomes a fixed-order masked reduction.
    racing_scatter = () if deterministic_reductions_enabled() else ("relion_norm_high_shell",)
    mismatched = []
    for name, (actual, expected) in checks.items():
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            mismatched.append(f"{name}: {actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}")
            continue
        if np.array_equal(actual, expected):
            continue
        differing = int(np.count_nonzero(actual != expected))
        left = actual.astype(np.complex128)
        right = expected.astype(np.complex128)
        worst = float(np.max(np.abs(left - right)))
        relative = float(
            np.max(np.abs(left - right) / np.maximum(np.abs(right), 1e-30))
        )
        if name in racing_scatter and relative <= _RACING_SCATTER_RELATIVE_BAND:
            logger.info(
                "Resident pass-2 operand verification on %s: %s is inside its racing "
                "scatter-add band (%d/%d cells, max |delta| %.3e, max relative %.3e); "
                "set RECOVAR_EM_DETERMINISTIC_REDUCTIONS=1 for an exact check",
                label, name, differing, actual.size, worst, relative,
            )
            continue
        mismatched.append(
            f"{name}: {differing}/{actual.size} cells differ, max |delta| {worst:.3e}, "
            f"max relative {relative:.3e}"
        )
    if mismatched:
        raise AssertionError(
            f"resident per-half operands differ from the per-chunk preparation on {label}: "
            + "; ".join(mismatched)
        )
    logger.info(
        "Resident pass-2 operand verification on %s: %d operands equal to the per-chunk "
        "preparation (BPref reconstruction operand=%d, deterministic reductions=%d)",
        label,
        len(checks),
        int(bpref_recon_operand),
        int(deterministic_reductions_enabled()),
    )


def _resident_operands_requested() -> bool:
    """Whether the per-image operands are prepared once per half (T16).

    Default on. ``RECOVAR_SPARSE_PASS2_RESIDENT_OPERANDS=0`` keeps the per-chunk
    ``_prepare_bucket_io`` preparation and the XLA tile reduction, which are the
    oracle for every bitwise comparison of the new path.
    """

    return parse_env_flag(_RESIDENT_OPERANDS_ENV, default=True)


def _resident_glue_jit_enabled() -> bool:
    """Whether the chunk loop's three stages are dispatched as jitted programs.

    Default on. With the flag off every stage runs as the loose sequence of
    eager operations the per-stage path used before P3-A: the same functions,
    the same order, only without the enclosing ``jax.jit``. That form is the
    oracle for the bitwise comparison, so it is kept rather than deleted.
    """

    return parse_env_flag(_RESIDENT_GLUE_JIT_ENV, default=True)


def _carry_aval_probe_enabled() -> bool:
    """Whether to check the static carry avals against a shape probe."""

    return parse_env_flag(_CARRY_AVAL_PROBE_ENV, default=False)


_DEVICE_INT32_CACHE: dict[int, jax.Array] = {}


def _scalar_operand(value, dtype) -> jax.Array:
    """Put a host scalar on the device without an eager conversion.

    ``jnp.asarray(np.int32(7), dtype=jnp.int32)`` dispatches a
    ``convert_element_type`` because a NumPy scalar is not an array; the same
    value wrapped in a 0-d NumPy array of the target dtype is transferred with
    no primitive at all. The chunk driver builds two of these per chunk, so on
    the early state that was 834 eager dispatches over two iterations for two
    integers whose value never leaves the host.
    """

    return jnp.asarray(np.asarray(value, dtype=jnp.dtype(dtype)))


def _device_int32(value: int) -> jax.Array:
    """A device int32 scalar, made once per distinct value for the process.

    The M-step block program takes its row offset as a device operand so that
    one program serves every block of a capacity class. Building that scalar
    with ``jnp.asarray`` inside the loop would put one eager dispatch back per
    block, which is the cost this program exists to remove; the offsets are a
    handful of multiples of the block size, so they are made once and reused.
    Single-device only, which is what the EM engines run on; a multi-device
    process falls through to a fresh array.
    """

    key = int(value)
    if len(jax.devices()) != 1:
        return jnp.asarray(key, dtype=jnp.int32)
    cached = _DEVICE_INT32_CACHE.get(key)
    if cached is None:
        cached = jnp.asarray(key, dtype=jnp.int32)
        _DEVICE_INT32_CACHE[key] = cached
    return cached


def _chunk_static_block_trip_enabled() -> bool:
    """Whether the chunk program's M-step loop runs the whole row capacity.

    Default off. The live-block bound skips blocks whose rows are all chunk
    padding; at the hp3 state that is about a quarter of the pixel-axis work
    (row occupancy 0.73 over the two measured iterations). Both forms are one
    program per capacity class, and a padded block contributes exact zeros, so
    the two are bitwise equal; the flag exists to measure that claim.
    """

    return parse_env_flag(_CHUNK_STATIC_BLOCKS_ENV, default=False)


# ---------------------------------------------------------------------------
# One program per chunk (T14)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ChunkProgramSpec:
    """Everything the chunk program is keyed on.

    Only capacity classes, pixel counts and resolved configuration appear here,
    so two chunks of one class share a program whatever their occupancy: the
    valid row and image counts travel as device scalars.
    """

    row_capacity: int
    image_capacity: int
    n_fine_trans: int
    n_score_pixels: int
    n_recon_pixels: int
    n_rect: int
    mstep_block_rows: int
    adaptive_fraction: float
    current_size: int
    mstep_current_size: int
    image_shape: tuple
    recon_volume_shape: tuple
    max_adjoint_block_bytes: int
    stats_config: object
    use_rfloat_ctf_wavg: bool
    # T16: whether the M-step's weighted sums come from the translate-and-sum
    # kernel on unshifted operands, and whether its reconstruction operand takes
    # the BPref convention (a weight) or the score convention (no weight).
    use_translate_sum_kernel: bool
    bpref_recon_operand: bool
    # Whether ``ctf_probs`` comes from the kernel's fourth output instead of the
    # XLA statement the per-chunk path used. Off by default: the kernel's own
    # translation mass is bitwise against ``jnp.sum`` only at some shapes.
    kernel_ctf_probs: bool
    # Whether the Wavg rectangle's image power is squared once per image and
    # gathered, instead of gathered and squared once per row. Bitwise either
    # way; see ``_WAVG_POWER_PER_IMAGE_ENV``.
    wavg_power_per_image: bool
    # M-step blocks emitted per device-loop iteration.
    block_unroll: int
    # Trip mechanism of the M-step block loop. False (default) bounds the loop
    # by the chunk's live block count, a device scalar, so the padded blocks the
    # per-stage loop breaks out of are skipped; True runs the full capacity as
    # the ticket's literal form does. Both trace one program per capacity class.
    static_block_trip: bool


class _MstepOnlyStatsConfig(NamedTuple):
    """The single statistics field the M-step block body reads.

    ``run_resident_mstep_blocks`` runs the M-step alone for a caller that owns
    its own scoring, posterior and statistics (local search, T12). Handing it
    the shell count in the shape ``_ChunkProgramSpec`` expects keeps one M-step
    body without making that caller build a full statistics configuration.
    """

    n_shells: int


class _ChunkRowArrays(NamedTuple):
    """One chunk's row-aligned tables and its runtime extents."""

    row_image_local: jax.Array  # int32 [C_R]
    row_fine_rot: jax.Array  # int32 [C_R]
    row_log_prior: jax.Array  # float32 [C_R]
    row_mask_bits: jax.Array  # uint32 [C_R], one packed word per row
    row_mask_mode: jax.Array  # int8 [C_R]
    image_ids: jax.Array  # int32 [C_B], global image id, -1 when padded
    n_valid_rows: jax.Array  # int32 []
    n_valid_images: jax.Array  # int32 []
    segment_offsets: jax.Array  # int32 [C_B + 1], cell offsets
    image_row_start: jax.Array  # int64 [C_B], chunk-local first row of a slot
    image_row_count: jax.Array  # int64 [C_B], rows owned by a slot


class _ChunkStageOperands(NamedTuple):
    """One chunk's per-image operands, already padded to the image capacity."""

    score_input: jax.Array
    corr_img_score: jax.Array
    highres_xi2_half: jax.Array | None
    translation_prior: jax.Array
    # Exactly one reconstruction operand pair is populated. The per-chunk
    # preparation fills the pre-shifted ``[C_B, T, P]`` tiles; the once-per-half
    # preparation fills the unshifted ``[C_B, P]`` images T15's kernel takes,
    # with ``recon_weight`` set only in the exact-BPref configuration.
    shifted_recon: jax.Array | None
    shifted_noise: jax.Array | None
    recon_image: jax.Array | None
    recon_weight: jax.Array | None
    noise_image: jax.Array | None
    ctf2_over_nv_recon: jax.Array
    direct_ctf_rfloat_recon: jax.Array | None
    processed_image_half: jax.Array
    relion_norm_high_shell: jax.Array
    raw_translated_wavg_rectangle: jax.Array
    raw_translated_wavg_for_atomic: jax.Array
    scale: jax.Array
    group_ids: jax.Array
    translation_sqdist_ang: jax.Array | None


class _ChunkStageTables(NamedTuple):
    """Iteration-global device tables every chunk of a half reads."""

    projection_score_cache: jax.Array
    projection_recon_cache: jax.Array
    projection_recon_abs2_cache: jax.Array
    mstep_grid: jax.Array
    coarse_parent_grid: jax.Array
    fine_translation_parent: jax.Array
    half_weights: jax.Array
    translation_angles: jax.Array
    full_to_compact: jax.Array
    noise_variance_for_noise: jax.Array
    shell_indices_noise: jax.Array
    exact_positions: jax.Array
    recon_pixel_indices: jax.Array
    relion_x_half_recon_indices: jax.Array
    shell_indices_half: jax.Array
    wavg_shell_indices: jax.Array
    wavg_scale_pixel_mask: jax.Array


class _ChunkPosterior(NamedTuple):
    """Stages 1-4 of one chunk."""

    row_posterior: jax.Array  # float32 [C_R, T]
    min_diff2: jax.Array  # real [C_B]
    class_log_z: jax.Array  # float64 [C_B]
    best_log_score: jax.Array  # float32 [C_B]
    best_cell_index: jax.Array  # int64 [C_B]
    max_posterior: jax.Array  # real [C_B]
    kernel_row_image_ids: jax.Array  # int32 [C_R], -1 on padded rows
    row_is_valid: jax.Array  # bool [C_R]


class _ChunkMstepCarry(NamedTuple):
    """Loop carry of the M-step block loop."""

    Ft_y: jax.Array
    Ft_ctf: jax.Array
    wavg_triplet_pixels: jax.Array  # float32 [C_B, P_rect, 3]
    noise_shells: jax.Array  # float64 [n_shells]
    a2_per_image: jax.Array  # real [C_B]
    xa_per_image: jax.Array  # real [C_B]


def _resident_chunk_posterior(
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
    cuda_backproject,
) -> _ChunkPosterior:
    """Gather, cached projection, score and the segmented RELION posterior.

    Identical calls to the per-stage path; factored out so the jitted program
    and its oracle cannot drift apart.
    """

    row_capacity = int(spec.row_capacity)
    image_capacity = int(spec.image_capacity)
    n_fine_trans = int(spec.n_fine_trans)

    row_index = jnp.arange(row_capacity, dtype=jnp.int32)
    row_is_valid = row_index < rows.n_valid_rows
    kernel_row_image_ids = jnp.where(row_is_valid, rows.row_image_local, jnp.int32(-1))
    image_index = jnp.arange(image_capacity, dtype=jnp.int32)
    chunk_image_ids = jnp.where(image_index < rows.n_valid_images, image_index, jnp.int32(-1))

    scored = score_resident_chunk(
        rows.row_image_local,
        rows.row_fine_rot,
        rows.row_log_prior,
        rows.row_mask_bits,
        rows.row_mask_mode,
        rows.n_valid_rows,
        chunk_image_ids,
        tables.projection_score_cache,
        operands.score_input,
        operands.corr_img_score,
        operands.highres_xi2_half,
        operands.translation_prior,
        half_weights=tables.half_weights,
        translation_angles=tables.translation_angles,
        full_to_compact=tables.full_to_compact,
        fine_translation_parent=tables.fine_translation_parent,
        logical_current_size=jnp.asarray(spec.current_size, dtype=jnp.int32),
        row_capacity=row_capacity,
        image_capacity=image_capacity,
        n_fine_trans=n_fine_trans,
        n_score_pixels=int(spec.n_score_pixels),
    )
    scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)

    log_z = cuda_backproject.sparse_pass2_segmented_log_z_f64(
        scores_flat, rows.segment_offsets, rows.n_valid_images
    )
    posterior = cuda_backproject.sparse_pass2_segmented_posterior_f32(
        scores_flat,
        rows.segment_offsets,
        rows.n_valid_images,
        log_z,
        jnp.ones((image_capacity,), dtype=jnp.float32),
        adaptive_fraction=float(spec.adaptive_fraction),
        keep_all=False,
        use_external_sum_weight=False,
    )
    (
        log_z_out,
        best_log_score,
        best_cell_index,
        max_posterior,
        _probs,
        _normalized_weights,
        reconstruction_probs,
        _mask,
        _n_significant,
        _sum_weight,
        _threshold,
    ) = posterior
    return _ChunkPosterior(
        row_posterior=jnp.asarray(reconstruction_probs, dtype=jnp.float32).reshape(
            row_capacity, n_fine_trans
        ),
        min_diff2=scored.min_diff2,
        class_log_z=jnp.asarray(log_z_out, dtype=jnp.float64),
        best_log_score=best_log_score,
        best_cell_index=jnp.asarray(best_cell_index, dtype=jnp.int64),
        max_posterior=max_posterior,
        kernel_row_image_ids=kernel_row_image_ids,
        row_is_valid=row_is_valid,
    )


def _cached_block_projections(tables: _ChunkStageTables, block_fine_rot):
    """The global pass's block projections: three gathers by fine-rotation id.

    Split out so the M-step block body takes its projections as operands. The
    global pass keeps gathering them out of the per-iteration caches exactly
    where it did before; local search (T12) has no cacheable fine grid and
    slices the projections it computed for the chunk's own rows instead.
    """

    return (
        tables.projection_recon_cache[block_fine_rot],
        tables.projection_recon_abs2_cache[block_fine_rot],
        tables.mstep_grid[block_fine_rot],
    )


def _resident_mstep_block(
    *,
    block_row_image,
    block_kernel_ids,
    block_posterior,
    block_projections,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    carry: _ChunkMstepCarry,
    spec: _ChunkProgramSpec,
    cuda_backproject,
) -> _ChunkMstepCarry:
    """One pixel-axis row block: weighted sums, Wavg, noise, both adjoints.

    Statement for statement the body the per-stage loop ran inline; every path
    calls this one copy, so the only differences between them are how the
    block's rows are sliced (static Python slice versus ``dynamic_slice``) and
    where its projections come from. ``block_projections`` is the block's
    ``(projection, |projection|^2, M-step rotations)``: the global pass passes
    ``_cached_block_projections(tables, block_fine_rot)``, local search passes
    a slice of the projections it computed for this chunk.
    """

    proj, proj_abs2, block_mstep_rotations = block_projections
    logical_recon_pixels = jnp.asarray(spec.n_recon_pixels, dtype=jnp.int32)
    logical_rect_pixels = jnp.asarray(spec.n_rect, dtype=jnp.int32)

    if spec.use_translate_sum_kernel:
        summed, summed_masked, ctf_probs, _probs_sum_t = _resident_block_weighted_sums_kernel(
            block_posterior,
            block_kernel_ids,
            block_row_image,
            operands.recon_image,
            operands.recon_weight,
            operands.noise_image,
            operands.ctf2_over_nv_recon,
            tables.recon_pixel_indices,
            tables.translation_angles,
            image_shape=spec.image_shape,
            n_recon_pixels=int(spec.n_recon_pixels),
            kernel_ctf_probs=bool(spec.kernel_ctf_probs),
            cuda_backproject=cuda_backproject,
        )
    else:
        summed, summed_masked, ctf_probs, _probs_sum_t = _resident_block_weighted_sums(
            block_posterior,
            block_row_image,
            operands.shifted_recon,
            operands.shifted_noise,
            operands.ctf2_over_nv_recon,
        )

    # RELION Wavg triplet in the flat-row layout, then its rotation atomics.
    # The host tail picks the sequential RELION reducer when the pass carries
    # the RFLOAT CTF operand and the algebraic form otherwise; follow the same
    # branch, resolved on the host into the program's static configuration.
    if not spec.use_rfloat_ctf_wavg:
        exact_terms = _resident_block_wavg_algebraic_terms(
            proj,
            proj_abs2,
            summed_masked,
            ctf_probs,
            tables.noise_variance_for_noise,
            operands.scale,
            operands.raw_translated_wavg_for_atomic,
            block_posterior,
            block_row_image,
        )
    else:
        exact_terms = cuda_backproject.relion_wavg_sequential_runtime_flat_rows_triplet_f32(
            jnp.asarray(proj, dtype=jnp.complex64),
            block_kernel_ids,
            jnp.asarray(operands.direct_ctf_rfloat_recon, dtype=jnp.float32),
            jnp.asarray(operands.scale, dtype=jnp.float32),
            jnp.asarray(operands.raw_translated_wavg_for_atomic, dtype=jnp.complex64),
            block_posterior,
            logical_recon_pixels,
        )
    rectangle_terms = _resident_block_wavg_rectangle_terms(
        exact_terms,
        operands.raw_translated_wavg_rectangle,
        block_posterior,
        block_row_image,
        tables.exact_positions,
        power_per_image=bool(spec.wavg_power_per_image),
    )
    wavg_triplet_pixels = (
        cuda_backproject.relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32(
            rectangle_terms,
            block_kernel_ids,
            carry.wavg_triplet_pixels,
            logical_rect_pixels,
        )
    )

    block_shells, block_a2, block_xa = _resident_block_noise_and_norm(
        proj,
        proj_abs2,
        summed_masked,
        ctf_probs,
        tables.noise_variance_for_noise,
        tables.shell_indices_noise,
        block_row_image,
        n_shells=int(spec.stats_config.n_shells),
        image_capacity=int(spec.image_capacity),
    )

    Ft_y = _accumulate_adjoint_block_chunked(
        summed,
        block_mstep_rotations,
        carry.Ft_y,
        window_indices=tables.relion_x_half_recon_indices,
        use_windowed_adjoint=True,
        image_shape=spec.image_shape,
        volume_shape=spec.recon_volume_shape,
        disc_type="linear_interp",
        half_image=True,
        half_volume=True,
        max_r=float(spec.mstep_current_size // 2),
        relion_x_half=True,
        max_block_bytes=int(spec.max_adjoint_block_bytes),
        log_label="resident-y-window",
    )
    Ft_ctf = _accumulate_adjoint_block_chunked(
        ctf_probs,
        block_mstep_rotations,
        carry.Ft_ctf,
        window_indices=tables.relion_x_half_recon_indices,
        use_windowed_adjoint=True,
        image_shape=spec.image_shape,
        volume_shape=spec.recon_volume_shape,
        disc_type="linear_interp",
        half_image=True,
        half_volume=True,
        max_r=float(spec.mstep_current_size // 2),
        relion_x_half=True,
        max_block_bytes=int(spec.max_adjoint_block_bytes),
        log_label="resident-ctf-window",
    )
    return _ChunkMstepCarry(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        wavg_triplet_pixels=wavg_triplet_pixels,
        noise_shells=carry.noise_shells + block_shells,
        a2_per_image=carry.a2_per_image + block_a2,
        xa_per_image=carry.xa_per_image + block_xa,
    )


def _mstep_block_operand_dtypes(
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
    projection_dtypes=None,
) -> dict:
    """Dtypes of the M-step block's intermediates, by promotion arithmetic.

    Every statement between the block's operands and its three accumulating
    outputs promotes; none of them casts, except the one explicit
    ``astype(float64)`` on the noise shells. So the output dtypes follow from
    the operand dtypes alone, on the host, without tracing anything:

    * ``summed_masked`` is the translate-and-sum kernel's declared complex64
      output, or ``compute_local_weighted_sums`` of the float32 posterior
      against the noise tile;
    * ``ctf_probs`` is the kernel's declared float32 fourth output when it is
      selected, and otherwise ``compute_local_ctf_sums_from_probs_sum_t`` of
      the float32 translation mass against the gathered CTF row;
    * ``A2`` promotes ``|proj|^2``, ``ctf_probs`` and the noise variance;
    * ``XA`` promotes the noise variance against the real part of
      ``proj * conj(summed_masked)``.

    ``_carry_aval_probe_enabled`` checks this against ``jax.eval_shape`` of the
    real block stages; the unit tests set it.
    """

    if projection_dtypes is None:
        # The global pass reads the per-iteration caches; local search has no
        # cache and hands the dtypes of the projections it just computed.
        projection_dtypes = (
            tables.projection_recon_cache.dtype,
            tables.projection_recon_abs2_cache.dtype,
        )
    proj_dtype, proj_abs2_dtype = (jnp.dtype(value) for value in projection_dtypes)
    noise_dtype = jnp.dtype(tables.noise_variance_for_noise.dtype)

    if spec.use_translate_sum_kernel:
        summed_masked_dtype = jnp.dtype(jnp.complex64)
    else:
        summed_masked_dtype = jnp.dtype(
            jnp.result_type(jnp.float32, operands.shifted_noise.dtype)
        )
    if spec.use_translate_sum_kernel and spec.kernel_ctf_probs:
        ctf_probs_dtype = jnp.dtype(jnp.float32)
    else:
        ctf_probs_dtype = jnp.dtype(
            jnp.result_type(jnp.float32, operands.ctf2_over_nv_recon.dtype)
        )

    a2_dtype = jnp.dtype(jnp.result_type(proj_abs2_dtype, ctf_probs_dtype, noise_dtype))
    cross_dtype = jnp.dtype(jnp.result_type(proj_dtype, summed_masked_dtype))
    # ``np.zeros`` rather than ``jnp.zeros``: this asks for the real part's
    # dtype, not for a value, and the device version dispatched one
    # ``convert_element_type`` per chunk to allocate a 0-d array that is read
    # for its dtype and thrown away. NumPy's promotion of a real part is the
    # same table JAX consults.
    xa_dtype = jnp.dtype(
        jnp.result_type(noise_dtype, np.zeros((), dtype=cross_dtype).real.dtype)
    )
    return {
        "proj": proj_dtype,
        "proj_abs2": proj_abs2_dtype,
        "summed_masked": summed_masked_dtype,
        "ctf_probs": ctf_probs_dtype,
        "noise": noise_dtype,
        "a2": a2_dtype,
        "xa": xa_dtype,
        # ``_resident_block_noise_and_norm`` casts the binned shells to float64
        # before they leave the block, so the carry is float64 whatever the
        # operands promote to.
        "noise_shells": jnp.dtype(jnp.float64),
    }


def _probe_mstep_block_output_avals(
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
    dtypes: dict,
):
    """``jax.eval_shape`` of the block's noise/norm stage, for the aval check.

    This is the probe the chunk loop used to run once per chunk. It is kept as
    a diagnostic only: :func:`_mstep_block_operand_dtypes` is what the driver
    uses, and this function exists so that arithmetic can be proved equal to
    the traced answer.
    """

    block_rows = int(spec.mstep_block_rows)
    n_pixels = int(spec.n_recon_pixels)
    image_capacity = int(spec.image_capacity)

    def probe(summed_masked, ctf_probs, proj, proj_abs2, noise, shells, row_image):
        return _resident_block_noise_and_norm(
            proj,
            proj_abs2,
            summed_masked,
            ctf_probs,
            noise,
            shells,
            row_image,
            n_shells=int(spec.stats_config.n_shells),
            image_capacity=image_capacity,
        )

    return jax.eval_shape(
        probe,
        jax.ShapeDtypeStruct((block_rows, n_pixels), dtypes["summed_masked"]),
        jax.ShapeDtypeStruct((block_rows, n_pixels), dtypes["ctf_probs"]),
        jax.ShapeDtypeStruct((block_rows, n_pixels), dtypes["proj"]),
        jax.ShapeDtypeStruct((block_rows, n_pixels), dtypes["proj_abs2"]),
        tables.noise_variance_for_noise,
        tables.shell_indices_noise,
        jax.ShapeDtypeStruct((block_rows,), jnp.int32),
    )


def _check_mstep_carry_avals(
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
    dtypes: dict,
) -> None:
    """Raise when the static dtypes disagree with the traced block stages."""

    shells_aval, a2_aval, xa_aval = _probe_mstep_block_output_avals(
        tables, spec=spec, dtypes=dtypes
    )
    expected = (
        ((int(spec.stats_config.n_shells),), dtypes["noise_shells"]),
        ((int(spec.image_capacity),), dtypes["a2"]),
        ((int(spec.image_capacity),), dtypes["xa"]),
    )
    probed = tuple(
        (tuple(int(size) for size in aval.shape), jnp.dtype(aval.dtype))
        for aval in (shells_aval, a2_aval, xa_aval)
    )
    if probed != expected:
        raise AssertionError(
            "resident M-step carry avals disagree with the traced block stages: "
            f"static={expected} probed={probed}"
        )


@lru_cache(maxsize=None)
def _zero_block_partials(shapes_and_dtypes: tuple) -> Callable[[], tuple]:
    """One program per capacity class that allocates the zero accumulators.

    ``jnp.zeros`` outside a jit is two eager dispatches, a
    ``convert_element_type`` of the scalar zero and a ``broadcast_in_dim`` to
    the shape; the M-step carry has four of them and the driver builds one
    carry per chunk, which on the early state was 3336 eager dispatches over
    two iterations. Inside a program with static shapes and dtypes the same
    four buffers cost none, and each call still returns fresh buffers, which
    the donated M-step block program requires.

    Keyed on the shapes and dtypes, so a class compiles once and every chunk
    of that class reuses it.
    """

    @jax.jit
    def build():
        return tuple(jnp.zeros(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes)

    return build


def _initial_mstep_carry(
    Ft_y,
    Ft_ctf,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
    projection_dtypes=None,
) -> _ChunkMstepCarry:
    """Zero-initialized block accumulators with the block stages' own dtypes.

    The per-image ``A2``/``XA`` partials take whatever dtype the noise and
    projection operands promote to. Those dtypes are computed from the operand
    dtypes and the capacity class by :func:`_mstep_block_operand_dtypes`, which
    is host arithmetic; the chunk loop used to learn them by tracing two
    ``jax.eval_shape`` probes per chunk instead, three traces of Python work
    for an answer that is the same for every chunk of a class.
    Zero-initializing (rather than seeding with the first block, as an earlier
    revision did) makes the loop a ``lax.fori_loop`` carry; adding a leading
    zero changes no float value except the unobservable ``-0.0`` case.
    """

    image_capacity = int(spec.image_capacity)
    dtypes = _mstep_block_operand_dtypes(
        operands, tables, spec=spec, projection_dtypes=projection_dtypes
    )
    if _carry_aval_probe_enabled():
        _check_mstep_carry_avals(tables, spec=spec, dtypes=dtypes)
    wavg_triplet_pixels, noise_shells, a2_per_image, xa_per_image = _zero_block_partials(
        (
            ((image_capacity, int(spec.n_rect), 3), jnp.dtype(jnp.float32)),
            ((int(spec.stats_config.n_shells),), jnp.dtype(dtypes["noise_shells"])),
            ((image_capacity,), jnp.dtype(dtypes["a2"])),
            ((image_capacity,), jnp.dtype(dtypes["xa"])),
        )
    )()
    return _ChunkMstepCarry(
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        wavg_triplet_pixels=wavg_triplet_pixels,
        noise_shells=noise_shells,
        a2_per_image=a2_per_image,
        xa_per_image=xa_per_image,
    )


class _MstepBlockInputs(NamedTuple):
    """Chunk-wide row arrays the M-step block program slices its block out of.

    ``row_fine_rot`` and ``projections`` are alternatives, and exactly one is
    populated: the global pass hands the fine-rotation ids and the program
    gathers the block's projections out of the per-iteration caches, while
    local search has no cacheable fine grid and hands the block's projections
    directly. ``None`` is a pytree structure, so the two callers key different
    programs without a flag.
    """

    row_image_local: jax.Array  # int32 [C_R]
    kernel_row_image_ids: jax.Array  # int32 [C_R]
    row_posterior: jax.Array  # float32 [C_R, T]
    row_fine_rot: jax.Array | None  # int32 [C_R]
    projections: tuple | None  # (proj, |proj|^2, M-step rotations) of one block


def _resident_mstep_block_at(
    block_start,
    blocks: _MstepBlockInputs,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    carry: _ChunkMstepCarry,
    *,
    spec: _ChunkProgramSpec,
    cuda_backproject,
) -> _ChunkMstepCarry:
    """Slice one block out of the chunk's row arrays and run the block body.

    The per-stage loop did these four slices and three gathers as loose eager
    operations, one dispatch each per block. They are the same slices: every
    row capacity is a whole number of blocks, so ``block_start + block_rows``
    never exceeds the capacity and ``dynamic_slice_in_dim`` never clamps, which
    makes the rows it returns the rows the Python slice returned.
    """

    block_rows = int(spec.mstep_block_rows)

    def take(values):
        return jax.lax.dynamic_slice_in_dim(values, block_start, block_rows, axis=0)

    if blocks.projections is None:
        block_projections = _cached_block_projections(tables, take(blocks.row_fine_rot))
    else:
        block_projections = blocks.projections
    return _resident_mstep_block(
        block_row_image=take(blocks.row_image_local),
        block_kernel_ids=take(blocks.kernel_row_image_ids),
        block_posterior=take(blocks.row_posterior),
        block_projections=block_projections,
        operands=operands,
        tables=tables,
        carry=carry,
        spec=spec,
        cuda_backproject=cuda_backproject,
    )


@partial(jax.jit, static_argnames=("spec",), donate_argnums=(4,))
def _resident_mstep_block_program(
    block_start,
    blocks: _MstepBlockInputs,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    carry: _ChunkMstepCarry,
    *,
    spec: _ChunkProgramSpec,
) -> _ChunkMstepCarry:
    """One M-step block as one program, keyed on the capacity and pixel class.

    Same statements, same order, same dtypes as the loose dispatch; the only
    change is where the JIT boundary sits. The carry is donated so the two
    half-volumes the windowed adjoint accumulates into keep being updated in
    place, as they are when the adjoint FFI is dispatched on its own.
    """

    from recovar import cuda_backproject

    return _resident_mstep_block_at(
        block_start,
        blocks,
        operands,
        tables,
        carry,
        spec=spec,
        cuda_backproject=cuda_backproject,
    )


@partial(jax.jit, static_argnames=("spec",))
def _resident_chunk_posterior_program(
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    *,
    spec: _ChunkProgramSpec,
) -> _ChunkPosterior:
    """:func:`_resident_chunk_posterior` as one program per capacity class.

    The stage's own arithmetic is unchanged; what leaves the chunk loop is the
    dozen eager operations around it -- the two ``arange`` row/image masks, the
    logical current size, the score reshape and the posterior's unit external
    weight -- each of which was a dispatch and a single-primitive program.
    """

    from recovar import cuda_backproject

    return _resident_chunk_posterior(
        rows, operands, tables, spec=spec, cuda_backproject=cuda_backproject
    )


@partial(jax.jit, static_argnames=("spec",), donate_argnums=(0,))
def _resident_chunk_statistics_program(
    stats,
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    posterior: _ChunkPosterior,
    mstep: _ChunkMstepCarry,
    *,
    spec: _ChunkProgramSpec,
):
    """:func:`_resident_chunk_statistics` as one program per capacity class.

    The statistics accumulator is donated: it is a running total the driver
    rebinds every chunk, so updating it in place is what the loose dispatch
    already did through ``_accumulate_chunk_image_terms``.
    """

    return _resident_chunk_statistics(
        stats, rows, operands, tables, posterior, mstep, spec=spec
    )


def _resident_chunk_statistics(
    stats,
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    posterior: _ChunkPosterior,
    mstep: _ChunkMstepCarry,
    *,
    spec: _ChunkProgramSpec,
):
    """Fold one chunk's image-level terms and its padding sanity counter."""

    image_tables = _ChunkImageTables(
        shell_indices_half=tables.shell_indices_half,
        wavg_shell_indices=tables.wavg_shell_indices,
        wavg_scale_pixel_mask=tables.wavg_scale_pixel_mask,
        translation_sqdist_ang=operands.translation_sqdist_ang,
    )
    best_row_local = posterior.best_cell_index // jnp.int64(int(spec.n_fine_trans))
    slot_is_valid = jnp.arange(int(spec.image_capacity), dtype=jnp.int32) < rows.n_valid_images
    invalid_best = slot_is_valid & (
        (best_row_local < 0) | (best_row_local >= rows.image_row_count)
    )
    best_chunk_row = jnp.clip(
        rows.image_row_start + best_row_local,
        0,
        jnp.int64(max(int(spec.row_capacity) - 1, 0)),
    ).astype(jnp.int32)
    best_fine_rot = jnp.asarray(rows.row_fine_rot, dtype=jnp.int64)[best_chunk_row]

    chunk_operands = _ChunkImageOperands(
        row_posterior=posterior.row_posterior,
        row_image_local=rows.row_image_local,
        row_coarse_rot=jnp.where(
            posterior.row_is_valid,
            tables.coarse_parent_grid[rows.row_fine_rot],
            jnp.int32(int(spec.stats_config.n_coarse_rot)),
        ),
        image_ids=rows.image_ids,
        group_ids=operands.group_ids,
        processed_image_half=operands.processed_image_half,
        relion_norm_high_shell=operands.relion_norm_high_shell,
        wavg_triplet_pixels=mstep.wavg_triplet_pixels,
        block_noise_shells=mstep.noise_shells,
        a2_per_image=mstep.a2_per_image,
        xa_per_image=mstep.xa_per_image,
        class_log_z=posterior.class_log_z,
        min_diff2=posterior.min_diff2,
        best_log_score=posterior.best_log_score,
        max_posterior=posterior.max_posterior,
        best_cell_index=posterior.best_cell_index,
        best_fine_rot=best_fine_rot,
    )
    stats = _accumulate_chunk_image_terms(
        stats, chunk_operands, image_tables, config=spec.stats_config
    )
    return stats._replace(
        invalid_best_rows=stats.invalid_best_rows + jnp.sum(invalid_best.astype(jnp.int64))
    )


@partial(jax.jit, static_argnames=("spec",), donate_argnums=(3,))
def _run_resident_chunk_program(
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    carry: tuple,
    *,
    spec: _ChunkProgramSpec,
):
    """Every device stage of one capacity chunk in one program.

    The Python chunk loop around this call contains only the host materialize,
    the host padding and the operand preparation: no ``block_until_ready``, no
    ``.item()``, no ``np.asarray`` of a device value. The M-step block loop is
    a ``lax.fori_loop`` whose trip count is the chunk's *live* block count, a
    device scalar, so the program is keyed on the capacity class alone while
    still skipping the padded blocks the per-stage loop breaks out of. A static
    trip count would instead run those blocks; at the hp3 state that is about
    half of the pixel-axis work, all of it multiplying a zero posterior.
    """

    from recovar import cuda_backproject

    Ft_y_total, Ft_ctf_total, stats = carry
    posterior = _resident_chunk_posterior(
        rows, operands, tables, spec=spec, cuda_backproject=cuda_backproject
    )

    block_rows = int(spec.mstep_block_rows)
    mstep = _initial_mstep_carry(Ft_y_total, Ft_ctf_total, operands, tables, spec=spec)

    def block(carry_in, take):
        return _resident_mstep_block(
            block_row_image=take(rows.row_image_local),
            block_kernel_ids=take(posterior.kernel_row_image_ids),
            block_posterior=take(posterior.row_posterior),
            block_projections=_cached_block_projections(tables, take(rows.row_fine_rot)),
            operands=operands,
            tables=tables,
            carry=carry_in,
            spec=spec,
            cuda_backproject=cuda_backproject,
        )

    unroll = max(int(spec.block_unroll), 1)
    if spec.static_block_trip:
        n_blocks = int(spec.row_capacity) // block_rows
        n_outer = (n_blocks + unroll - 1) // unroll
    else:
        n_blocks = jax.lax.div(
            rows.n_valid_rows + jnp.int32(block_rows - 1), jnp.int32(block_rows)
        )
        n_outer = jax.lax.div(n_blocks + jnp.int32(unroll - 1), jnp.int32(unroll))

    def outer(outer_index, carry_in):
        # ``unroll`` blocks per device-loop iteration: the predicate is read back
        # once per iteration, and a trailing block past the live count carries
        # only padded rows, whose posterior is zero.
        for offset in range(unroll):
            index = outer_index * unroll + offset
            carry_in = block(
                carry_in,
                lambda values, _i=index: jax.lax.dynamic_slice_in_dim(
                    values, _i * block_rows, block_rows, axis=0
                ),
            )
        return carry_in

    mstep = jax.lax.fori_loop(0, n_outer, outer, mstep)
    stats = _resident_chunk_statistics(
        stats, rows, operands, tables, posterior, mstep, spec=spec
    )
    return mstep.Ft_y, mstep.Ft_ctf, stats


def _run_resident_chunk_stages(
    rows: _ChunkRowArrays,
    operands: _ChunkStageOperands,
    tables: _ChunkStageTables,
    carry: tuple,
    *,
    spec: _ChunkProgramSpec,
    n_valid_rows: int,
    timing_hook=None,
):
    """Per-stage oracle: the same stages, dispatched one at a time.

    Kept selectable by ``RECOVAR_SPARSE_PASS2_RESIDENT_CHUNK_JIT=0`` so the
    fused program can be compared against the path it replaces inside one
    process. ``timing_hook(name)`` is called after each stage when the chunk
    timing diagnostic is on; it synchronizes, so an arm that passes it is a
    diagnostic arm.
    """

    from recovar import cuda_backproject

    glue_jit = _resident_glue_jit_enabled()
    Ft_y_total, Ft_ctf_total, stats = carry
    if glue_jit:
        posterior = _resident_chunk_posterior_program(rows, operands, tables, spec=spec)
    else:
        posterior = _resident_chunk_posterior(
            rows, operands, tables, spec=spec, cuda_backproject=cuda_backproject
        )
    if timing_hook is not None:
        timing_hook("posterior", posterior.row_posterior)

    block_rows = int(spec.mstep_block_rows)
    mstep = _initial_mstep_carry(Ft_y_total, Ft_ctf_total, operands, tables, spec=spec)
    blocks = _make_mstep_block_inputs(rows, posterior)
    for start in range(0, int(spec.row_capacity), block_rows):
        if start >= int(n_valid_rows):
            # Every row of this block is chunk padding: its posterior is zero,
            # so the weighted sums, the Wavg terms, the noise partials and both
            # adjoint scatters are exactly zero and adding them changes no
            # accumulator bit.
            break
        if glue_jit:
            mstep = _resident_mstep_block_program(
                _device_int32(start), blocks, operands, tables, mstep, spec=spec
            )
            continue
        block = slice(start, start + block_rows)
        mstep = _resident_mstep_block(
            block_row_image=rows.row_image_local[block],
            block_kernel_ids=posterior.kernel_row_image_ids[block],
            block_posterior=posterior.row_posterior[block],
            block_projections=_cached_block_projections(tables, rows.row_fine_rot[block]),
            operands=operands,
            tables=tables,
            carry=mstep,
            spec=spec,
            cuda_backproject=cuda_backproject,
        )
    if timing_hook is not None:
        timing_hook("mstep", (mstep.Ft_y, mstep.Ft_ctf))

    if glue_jit:
        stats = _resident_chunk_statistics_program(
            stats, rows, operands, tables, posterior, mstep, spec=spec
        )
    else:
        stats = _resident_chunk_statistics(
            stats, rows, operands, tables, posterior, mstep, spec=spec
        )
    return mstep.Ft_y, mstep.Ft_ctf, stats


def _run_resident_chunk(
    chunk,
    *,
    tables,
    experiment_dataset,
    bucket_io_kwargs,
    half_weights,
    translation_angles,
    full_to_compact,
    n_score_pixels,
    fine_translation_prior_2d,
    score_real_dtype,
    projection_score_cache,
    projection_recon_cache,
    projection_recon_abs2_cache,
    fine_translation_parent_device,
    mstep_grid,
    coarse_parent_grid,
    n_fine_trans,
    n_recon_windowed,
    n_rect,
    mstep_block_rows,
    adaptive_fraction,
    windowed_prepare,
    window_indices,
    recon_window_indices,
    relion_x_half_recon_indices,
    exact_positions_device,
    rect_indices_device,
    recon_pixel_indices,
    resident_operands,
    verify_operands,
    image_shape,
    current_size,
    mstep_current_size,
    recon_volume_shape,
    max_adjoint_block_bytes,
    noise_variance_for_noise,
    shell_indices_noise,
    group_ids_np,
    scale_corrections_np,
    translation_prior_centers_np,
    fine_translations,
    voxel_size,
    use_exact_relion_gaussian,
    accumulate_noise,
    source_faithful_spectrum_norm,
    stats,
    stats_config,
    image_tables,
    Ft_y_total,
    Ft_ctf_total,
    cuda_backproject,
    submitted_keys=None,
):
    """Run every resident stage for one capacity chunk.

    ``submitted_keys``, when given, collects the ``(program name, spec)`` keys
    this chunk submits, so the driver can say how many of them the compile-ahead
    warm-up had already compiled. Recording costs a set insert per chunk.

    Returns the updated ``(Ft_y_total, Ft_ctf_total, stats)``. Host work inside
    is the chunk's materialize/pad, its operand preparation and the T7 offsets
    readback the segmented posterior performs internally; no per-chunk result
    is pulled.
    """

    row_capacity = int(chunk.row_capacity)
    image_capacity = int(chunk.image_capacity)
    n_valid_rows = int(chunk.n_valid_rows)
    n_valid_images = int(chunk.n_valid_images)
    image_indices = np.arange(chunk.image_start, chunk.image_stop, dtype=np.int64)
    timing = _chunk_timing_enabled()
    chunk_t0 = time.time()
    stage_t = {}
    if timing:
        jax.block_until_ready(Ft_y_total)
        chunk_t0 = time.time()

    rows = _make_chunk_row_arrays(tables, chunk, n_fine_trans, place=_PLACE_ON_DEVICE)

    if resident_operands is None:
        recon = _prepare_chunk_reconstruction_operands(
            chunk=chunk,
            image_indices=image_indices,
            experiment_dataset=experiment_dataset,
            bucket_io_kwargs=bucket_io_kwargs,
            windowed_prepare=windowed_prepare,
            recon_window_indices=recon_window_indices,
            score_window_indices=window_indices,
            fine_translation_prior_2d=fine_translation_prior_2d,
            score_real_dtype=score_real_dtype,
            n_fine_trans=int(n_fine_trans),
            n_recon_windowed=int(n_recon_windowed),
            image_shape=image_shape,
            current_size=current_size,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            relion_score_translation_angles=translation_angles,
            rect_indices_device=rect_indices_device,
            exact_positions_device=exact_positions_device,
            scale_corrections_np=scale_corrections_np,
            group_ids_np=group_ids_np,
        )
    else:
        # The chunk's image slots are the half's images ``image_start`` to
        # ``image_stop``; the padded slots carry -1 and the gather zeroes them,
        # which is what the per-chunk preparation's capacity mask did.
        image_slots = np.full(image_capacity, -1, dtype=np.int32)
        image_slots[:n_valid_images] = image_indices[:n_valid_images]
        recon = gather_resident_chunk_operands(
            resident_operands,
            image_slots,
            translation_angles=translation_angles,
            rect_indices=rect_indices_device,
            exact_positions=exact_positions_device,
            image_shape=image_shape,
        )
        if verify_operands:
            _verify_resident_chunk_operands(
                recon,
                _prepare_chunk_reconstruction_operands(
                    chunk=chunk,
                    image_indices=image_indices,
                    experiment_dataset=experiment_dataset,
                    bucket_io_kwargs=bucket_io_kwargs,
                    windowed_prepare=windowed_prepare,
                    recon_window_indices=recon_window_indices,
                    score_window_indices=window_indices,
                    fine_translation_prior_2d=fine_translation_prior_2d,
                    score_real_dtype=score_real_dtype,
                    n_fine_trans=int(n_fine_trans),
                    n_recon_windowed=int(n_recon_windowed),
                    image_shape=image_shape,
                    current_size=current_size,
                    use_exact_relion_gaussian=use_exact_relion_gaussian,
                    accumulate_noise=accumulate_noise,
                    source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                    relion_score_translation_angles=translation_angles,
                    rect_indices_device=rect_indices_device,
                    exact_positions_device=exact_positions_device,
                    scale_corrections_np=scale_corrections_np,
                    group_ids_np=group_ids_np,
                ),
                translation_angles=translation_angles,
                recon_pixel_indices=recon_pixel_indices,
                image_shape=image_shape,
                n_recon_pixels=int(n_recon_windowed),
                bpref_recon_operand=resident_operands.recon_weight is not None,
                label=f"chunk images {chunk.image_start}-{chunk.image_stop}",
                cuda_backproject=cuda_backproject,
            )
    if timing:
        jax.block_until_ready(
            recon["shifted_recon"] if resident_operands is None else recon["recon_image"]
        )
        stage_t["operands"] = time.time() - chunk_t0

    # The prior squared distances are a per-image host table; building them at
    # capacity here keeps the program keyed on the capacity class and takes the
    # eager device ops out of the chunk loop. Padded slots multiply a zero
    # posterior, so their value is never observable; they are zeroed anyway.
    translation_sqdist_ang = _make_chunk_translation_sqdist(
        image_tables.translation_sqdist_ang,
        translation_prior_centers_np=translation_prior_centers_np,
        image_indices=image_indices,
        image_capacity=image_capacity,
        n_valid_images=n_valid_images,
        fine_translations=fine_translations,
        voxel_size=voxel_size,
    )

    operands = _make_chunk_stage_operands(recon, translation_sqdist_ang)
    stage_tables = _make_chunk_stage_tables(
        projection_score_cache=projection_score_cache,
        projection_recon_cache=projection_recon_cache,
        projection_recon_abs2_cache=projection_recon_abs2_cache,
        mstep_grid=mstep_grid,
        coarse_parent_grid=coarse_parent_grid,
        fine_translation_parent_device=fine_translation_parent_device,
        half_weights=half_weights,
        translation_angles=translation_angles,
        full_to_compact=full_to_compact,
        noise_variance_for_noise=noise_variance_for_noise,
        shell_indices_noise=shell_indices_noise,
        exact_positions_device=exact_positions_device,
        recon_pixel_indices=recon_pixel_indices,
        relion_x_half_recon_indices=relion_x_half_recon_indices,
        image_tables=image_tables,
    )
    spec = _make_chunk_program_spec(
        row_capacity=row_capacity,
        image_capacity=image_capacity,
        n_fine_trans=n_fine_trans,
        n_score_pixels=n_score_pixels,
        n_recon_pixels=n_recon_windowed,
        n_rect=n_rect,
        mstep_block_rows=mstep_block_rows,
        adaptive_fraction=adaptive_fraction,
        current_size=current_size,
        mstep_current_size=mstep_current_size,
        image_shape=image_shape,
        recon_volume_shape=recon_volume_shape,
        max_adjoint_block_bytes=max_adjoint_block_bytes,
        stats_config=stats_config,
        use_rfloat_ctf_wavg=recon["direct_ctf_rfloat_recon"] is not None,
        use_translate_sum_kernel=resident_operands is not None,
        bpref_recon_operand=(
            resident_operands is not None and resident_operands.recon_weight is not None
        ),
    )

    if submitted_keys is not None:
        submitted_keys.update(chunk_program_keys(chunk_program_path(), spec))

    use_jit = _chunk_jit_enabled()
    if use_jit:
        Ft_y_total, Ft_ctf_total, stats = _run_resident_chunk_program(
            rows, operands, stage_tables, (Ft_y_total, Ft_ctf_total, stats), spec=spec
        )
    else:
        def timing_hook(name, value):
            jax.block_until_ready(value)
            stage_t[name] = time.time() - chunk_t0

        Ft_y_total, Ft_ctf_total, stats = _run_resident_chunk_stages(
            rows,
            operands,
            stage_tables,
            (Ft_y_total, Ft_ctf_total, stats),
            spec=spec,
            n_valid_rows=n_valid_rows,
            timing_hook=timing_hook if timing else None,
        )

    if timing:
        jax.block_until_ready((Ft_y_total, Ft_ctf_total, stats.wsum_sigma2_noise))
        total = time.time() - chunk_t0
        operands_s = stage_t.get("operands", 0.0)
        if use_jit:
            split = "program=%.3fs unroll=%d" % (total - operands_s, spec.block_unroll)
        else:
            posterior_end = stage_t.get("posterior", operands_s)
            mstep_end = stage_t.get("mstep", total)
            split = "score+posterior=%.3fs mstep=%.3fs statistics=%.3fs" % (
                posterior_end - operands_s,
                mstep_end - posterior_end,
                total - mstep_end,
            )
        blocks = sum(1 for s in range(0, row_capacity, int(mstep_block_rows)) if s < n_valid_rows)
        logger.info(
            "Resident pass-2 chunk timing: jit=%d images=%d/%d rows=%d/%d occupancy=%.3f "
            "mstep_blocks=%d/%d operands=%.3fs %s total=%.3fs",
            int(use_jit),
            n_valid_images, image_capacity, n_valid_rows, row_capacity,
            n_valid_rows / max(row_capacity, 1),
            blocks, row_capacity // int(mstep_block_rows),
            operands_s,
            split,
            total,
        )
    return Ft_y_total, Ft_ctf_total, stats


def run_resident_mstep_blocks(
    block_projections=None,
    *,
    chunk_projections=None,
    row_capacity: int,
    n_valid_rows: int,
    mstep_block_rows: int,
    image_capacity: int,
    row_image_local,
    kernel_row_image_ids,
    row_posterior,
    recon,
    n_rect: int,
    n_shells: int,
    n_recon_windowed: int,
    noise_variance_for_noise,
    shell_indices_noise,
    exact_positions_device,
    Ft_y_total,
    Ft_ctf_total,
    image_shape,
    recon_volume_shape,
    mstep_current_size,
    relion_x_half_recon_indices,
    max_adjoint_block_bytes,
    cuda_backproject,
):
    """Walk one chunk's pixel axis in row blocks: Wavg, noise and both adjoints.

    Exactly one of ``block_projections`` and ``chunk_projections`` is given.

    ``block_projections(start, stop)`` returns this block's reconstruction-window
    projection, its ``|proj|^2`` and its M-step rotations. The global pass 2
    gathers all three out of the per-iteration fine-rotation caches; local
    search (T12) slices them out of the projections it computed for the chunk's
    own rows, because its fine grid is not cacheable.

    ``chunk_projections`` (P3-G) is the same three arrays for the **whole**
    chunk, in the layout's flat row order. The caller then does no slicing at
    all: the arrays ride in the stage tables and the block program takes a
    ``dynamic_slice`` of the chunk's row ids and reads its rows inside the jit,
    exactly as the global pass reads its per-iteration caches at the block's
    fine-rotation ids. The row ids are ``0 .. row_capacity-1``, so the read is
    the identity gather of the rows the callback sliced, value for value.

    The body is ``_resident_mstep_block``, the same copy the global pass runs in
    both its per-stage and its jitted form, so no accumulator has a second
    implementation. Only the M-step fields of the stage containers are filled
    here: this entry point runs the M-step alone, and the scoring, posterior and
    statistics fields are unused by that body.
    """

    if recon.get("shifted_recon") is None or recon.get("shifted_noise") is None:
        # Fail closed rather than hand ``None`` to the XLA weighted sums: a
        # ``recon`` without the pre-shifted tiles is T16's once-per-half
        # preparation, whose weighted sums are the translate-and-sum kernel's,
        # and this entry point has no kernel path (``spec`` below pins
        # ``use_translate_sum_kernel=False``).
        raise ValueError(
            "run_resident_mstep_blocks takes the per-chunk pre-shifted "
            "reconstruction operands ('shifted_recon'/'shifted_noise'); this "
            "chunk carries T16's once-per-half per-image operands instead "
            f"(keys present: {sorted(k for k, v in recon.items() if v is not None)}). "
            "Prepare the chunk with _prepare_chunk_reconstruction_operands, or "
            "give this entry point the kernel path before handing it resident "
            "operands."
        )

    if (block_projections is None) == (chunk_projections is None):
        raise ValueError(
            "run_resident_mstep_blocks takes exactly one of block_projections "
            "(a callback returning one block's arrays) and chunk_projections "
            "(the chunk's whole row arrays); got "
            f"block_projections={'set' if block_projections is not None else 'None'} "
            f"and chunk_projections={'set' if chunk_projections is not None else 'None'}."
        )
    if chunk_projections is not None:
        chunk_proj, chunk_proj_abs2, chunk_mstep_rotations = chunk_projections
        for name, value in (
            ("projection", chunk_proj),
            ("|projection|^2", chunk_proj_abs2),
            ("M-step rotations", chunk_mstep_rotations),
        ):
            if int(value.shape[0]) != int(row_capacity):
                raise ValueError(
                    "chunk_projections must carry the chunk's whole row axis: "
                    f"the {name} array has {int(value.shape[0])} rows, the chunk "
                    f"capacity is {int(row_capacity)}."
                )

    spec = _ChunkProgramSpec(
        row_capacity=int(row_capacity),
        image_capacity=int(image_capacity),
        n_fine_trans=int(row_posterior.shape[1]),
        n_score_pixels=0,
        n_recon_pixels=int(n_recon_windowed),
        n_rect=int(n_rect),
        mstep_block_rows=int(mstep_block_rows),
        adaptive_fraction=0.0,
        current_size=0,
        mstep_current_size=int(mstep_current_size),
        image_shape=tuple(int(v) for v in image_shape),
        recon_volume_shape=tuple(int(v) for v in recon_volume_shape),
        max_adjoint_block_bytes=int(max_adjoint_block_bytes),
        stats_config=_MstepOnlyStatsConfig(n_shells=int(n_shells)),
        use_rfloat_ctf_wavg=recon["direct_ctf_rfloat_recon"] is not None,
        # T12's local pass hands pre-shifted per-chunk operands, not T16's
        # once-per-half resident images, so the M-step body takes its weighted
        # sums from the XLA statement, as it did before T16.
        use_translate_sum_kernel=False,
        bpref_recon_operand=False,
        kernel_ctf_probs=False,
        wavg_power_per_image=_wavg_power_per_image_enabled(),
        block_unroll=1,
        static_block_trip=False,
    )
    operands = _ChunkStageOperands(
        score_input=None,
        corr_img_score=None,
        highres_xi2_half=None,
        translation_prior=None,
        shifted_recon=recon["shifted_recon"],
        shifted_noise=recon["shifted_noise"],
        # The unshifted per-image images are T16's once-per-half operands; the
        # local pass prepares pre-shifted tiles per chunk, so this pair stays
        # empty and the M-step body takes the XLA weighted sums.
        recon_image=None,
        recon_weight=None,
        noise_image=None,
        ctf2_over_nv_recon=recon["ctf2_over_nv_recon"],
        direct_ctf_rfloat_recon=recon["direct_ctf_rfloat_recon"],
        processed_image_half=None,
        relion_norm_high_shell=None,
        raw_translated_wavg_rectangle=recon["raw_translated_wavg_rectangle"],
        raw_translated_wavg_for_atomic=recon["raw_translated_wavg_for_atomic"],
        scale=recon["scale"],
        group_ids=None,
        translation_sqdist_ang=None,
    )
    tables = _ChunkStageTables(
        projection_score_cache=None,
        # P3-G: the chunk's own row arrays stand where the global pass keeps
        # its per-iteration fine-rotation caches, so the block program reads
        # its rows inside the jit instead of taking them from a host callback.
        projection_recon_cache=None if chunk_projections is None else chunk_proj,
        projection_recon_abs2_cache=(
            None if chunk_projections is None else chunk_proj_abs2
        ),
        mstep_grid=None if chunk_projections is None else chunk_mstep_rotations,
        coarse_parent_grid=None,
        fine_translation_parent=None,
        half_weights=None,
        translation_angles=None,
        full_to_compact=None,
        noise_variance_for_noise=noise_variance_for_noise,
        shell_indices_noise=shell_indices_noise,
        exact_positions=exact_positions_device,
        # T16's kernel path addresses the reconstruction window by pixel index;
        # the XLA path this caller takes does not read it.
        recon_pixel_indices=None,
        relion_x_half_recon_indices=relion_x_half_recon_indices,
        shell_indices_half=None,
        wavg_shell_indices=None,
        wavg_scale_pixel_mask=None,
    )

    block_rows = int(mstep_block_rows)
    carry = None
    glue_jit = _resident_glue_jit_enabled()
    if chunk_projections is None:
        chunk_row_ids = None
        projection_dtypes = None
    else:
        # A host-side index vector, so it costs one transfer for the chunk and
        # no eager primitive: the block program slices it and gathers the rows.
        chunk_row_ids = jnp.asarray(np.arange(int(row_capacity), dtype=np.int32))
        projection_dtypes = (chunk_proj.dtype, chunk_proj_abs2.dtype)

    def start_carry(projections):
        return _initial_mstep_carry(
            Ft_y_total,
            Ft_ctf_total,
            operands,
            tables,
            spec=spec,
            projection_dtypes=(
                projection_dtypes
                if projections is None
                else (projections[0].dtype, projections[1].dtype)
            ),
        )

    for start in range(0, int(row_capacity), block_rows):
        if start >= int(n_valid_rows):
            # Every row of this block is chunk padding: its posterior is zero,
            # so the weighted sums, the Wavg terms, the noise partials and both
            # adjoint scatters are all exactly zero and adding them changes no
            # accumulator bit.
            break
        stop = start + block_rows
        block = slice(start, stop)
        if chunk_projections is None:
            projections = block_projections(start, stop)
        else:
            projections = None
        if carry is None:
            carry = start_carry(projections)
        if glue_jit:
            carry = _resident_mstep_block_program(
                _device_int32(start),
                _MstepBlockInputs(
                    row_image_local=row_image_local,
                    kernel_row_image_ids=kernel_row_image_ids,
                    row_posterior=row_posterior,
                    row_fine_rot=chunk_row_ids,
                    projections=projections,
                ),
                operands,
                tables,
                carry,
                spec=spec,
            )
            continue
        carry = _resident_mstep_block(
            block_row_image=row_image_local[block],
            block_kernel_ids=kernel_row_image_ids[block],
            block_posterior=row_posterior[block],
            block_projections=(
                _cached_block_projections(tables, chunk_row_ids[block])
                if projections is None
                else projections
            ),
            operands=operands,
            tables=tables,
            carry=carry,
            spec=spec,
            cuda_backproject=cuda_backproject,
        )
    if carry is None:
        # A chunk with no live rows: the accumulators are the incoming ones and
        # the partials are the zeros the statistics program expects.
        carry = start_carry(
            block_projections(0, block_rows) if chunk_projections is None else None
        )

    return (
        carry.Ft_y,
        carry.Ft_ctf,
        carry.wavg_triplet_pixels,
        carry.noise_shells,
        carry.a2_per_image,
        carry.xa_per_image,
    )
