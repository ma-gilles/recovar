"""Device-resident local-search pass 2 (T12).

The exact local engine (:func:`recovar.em.local.local_em_engine.run_local_em_exact`)
runs the order-4 local iterations and the final all-data iteration of a K=1
auto-refine. The Phase 0 budget measured those at 313 s and 211 s of a 2113 s
run with the GPU 93 % idle, because the engine pays host time per bucket and
recompiles per bucket shape. This module runs the same pass on the
device-resident driver's stages: fixed-capacity flat-row chunks, one program
per capacity class, accumulators that stay on the device and one pull per half.

Scope, and how to select it
---------------------------
``RECOVAR_LOCAL_SEARCH_RESIDENT=1`` selects :func:`compute_local_search_resident`
for the **fine pass 2** of a K=1 local search, which is also the pass the final
all-data iteration runs (``iteration_loop.py`` reaches it through the same
``local_outputs = _run_local_search_iteration`` call site in
``recovar/em/refinement/half_scoring.py``). Any other configuration raises
:class:`NotImplementedError` naming the missing piece; this path never falls
back silently, because a silent fallback would make a measured comparison
meaningless.

The pass-1 parent probe deliberately stays on the exact local engine
--------------------------------------------------------------------
The probe selects pass 2's candidate set. It applies RELION's
``maximum_significants`` cap (500 by default, ``apply_max_significants_to_support=True``
in ``half_scoring``) on top of the 0.999 adaptive fraction, through
``oversampling._find_significant_mask_full_sort`` in float64. The segmented
float32 posterior kernel (T7) implements the adaptive fraction only and has no
cap, so a resident probe would select a *different* support. That would change
pass 2's candidate set and make every pass-2 number incomparable, so the probe
is left where it is and only the expensive pass is moved. Lifting this needs a
capped segmented significance kernel, which is not part of this ticket.

Semantics that differ from the exact local engine, deliberately and measurably
------------------------------------------------------------------------------
The resident stages implement the *compact* K=1 pass-2 arithmetic, which is the
RELION-parity path for the global iterations. The exact local engine implements
a mathematically equivalent but differently factored arithmetic. Three
differences are real and are reported rather than assumed away:

1. **Scoring.** The local engine forms ``-0.5 * (cross + norms)`` with a JAX
   einsum over a pre-shifted ``(B, T, N)`` image tile and adds the image power
   ``-0.5 * batch_norm`` on the host afterwards. This driver calls RELION's
   fused translate-and-score CUDA kernel, which forms the whole
   ``sum |X_t - A|^2 * Minvsigma2`` in one pass and carries the out-of-window
   tail as the ``powerClass`` operand, then converts with RELION's common
   minimum. The posteriors are shift invariant, so the two agree up to float32
   association; the reported ``log_evidence`` and ``best_log_score`` carry
   different offsets by construction (``-min_diff2`` here, ``-0.5*batch_norm``
   there) and are compared as differences, not values.
2. **Significance.** The local engine sorts the float64 posterior on the host
   (``_find_significant_mask_full_sort``); this driver uses T7's segmented CUDA
   posterior, which is bitwise against the compact engine's rectangular
   handler. Cutoff ties can therefore be resolved differently.
3. **Noise shells.** The local engine accumulates the algebraic
   ``A2 - 2*XA`` residual and a separate image-power vector; this driver uses
   RELION's Wavg triplet with the direct low-shell residual, which is the
   compact engine's production form. The sum of the two vectors is the same
   quantity; the split between them, and its float32 rounding, are not.

Everything else - the candidate set and its order, RELION particle order for
the x-half BPref, the projector, the translation operand, the M-step
contraction and the scale/norm statistics - is the same code the accepted
paths run.
"""

from __future__ import annotations

import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

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
from recovar.em.helpers.projection import (
    relion_scale_correction_pixel_mask as _relion_scale_correction_pixel_mask,
)
from recovar.em.helpers.scale_groups import prepare_scale_correction_groups
from recovar.em.helpers.types import LocalEMResult, make_noise_stats, make_relion_stats
from recovar.em.relion.relion_projector_setup import (
    cast_relion_projector_for_execution,
    prepare_local_projector_slab,
)
from recovar.em.sparse_pass2 import resident_pass2 as rp
from recovar.em.sparse_pass2.resident_local_layout import (
    materialize_local_chunk,
    plan_local_capacity_chunks,
    tables_from_local_layout,
)
from recovar.em.sparse_pass2.resident_scoring import (
    project_resident_rows,
    resident_row_projection_bytes,
    score_resident_projected_chunk,
)
from recovar.em.sparse_pass2.resident_statistics import (
    finalize_statistics,
    make_resident_statistics,
    resolve_statistics_config,
)
from recovar.em.sparse_pass2.sparse_pass2_bucket_io import (
    _relion_cuda_score_translation_angles_if_available,
)
from recovar.em.sparse_pass2.sparse_pass2_budget import (
    _max_adjoint_block_bytes_for_pass,
    _max_translation_tile_bytes_for_pass,
    _projection_cache_max_bytes_for_pass,
)
from recovar.em.sparse_pass2.sparse_pass2_policy import (
    _RELION_WAVG_ATOMIC_SCALE_AA_ENV,
    _relion_wavg_direct_modes,
)
from recovar.em.sparse_pass2.sparse_pass2_projection_blocks import (
    _projection_kwargs_for_relion_score_window,
)
from recovar.em.sparse_pass2.sparse_pass2_scoring import (
    _relion_cuda_fine_full_to_compact_lookup,
)
from recovar.em.sparse_pass2.sparse_pass2_wavg import (
    _make_relion_wavg_rectangle,
)
from recovar.em.sparse_pass2.sparse_pass2_window import (
    _pass2_half_weights,
    _pass2_window_setup,
    _sparse_pass2_window_setup,
)
from recovar.reconstruction import noise as noise_utils

logger = logging.getLogger(__name__)

RESIDENT_LOCAL_SEARCH_ENV = "RECOVAR_LOCAL_SEARCH_RESIDENT"
_ROW_CAPACITY_LADDER_ENV = "RECOVAR_LOCAL_SEARCH_RESIDENT_ROW_CAPACITIES"
_IMAGE_CAPACITY_LADDER_ENV = "RECOVAR_LOCAL_SEARCH_RESIDENT_IMAGE_CAPACITIES"
# Diagnostic only: log one line per chunk with its occupancy, padding and
# per-stage seconds. It inserts ``block_until_ready`` between stages, so it
# serialises work that normally overlaps and inflates the loop; never use a
# profiled arm for a wall-time comparison.
_CHUNK_PROFILE_ENV = "RECOVAR_LOCAL_SEARCH_RESIDENT_CHUNK_PROFILE"
# P3-G, opt-in and off by default. Hand the M-step entry point the chunk's
# whole row arrays instead of a Python callback that slices a block out of them
# per block: the block program then takes its own ``dynamic_slice`` of the row
# ids and reads the rows inside the jit. The callback is kept as the oracle
# every bitwise comparison of this change is made against.
_BLOCK_ROW_PROGRAM_ENV = "RECOVAR_LOCAL_SEARCH_RESIDENT_BLOCK_ROW_PROGRAM"
_PROJECTION_CALL_MAX_BYTES_ENV = "RECOVAR_LOCAL_SEARCH_RESIDENT_PROJECTION_CALL_MAX_BYTES"
# One projector call's transient. The exact local engine budgets its own fused
# projection matmul at 4 GiB by default
# (local_batch_planning.EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB); use the same figure
# so the two engines reserve comparable headroom.
_DEFAULT_PROJECTION_CALL_MAX_BYTES = 4 * 1024**3


def _projection_call_transient_max_bytes() -> int:
    """Bytes one projector call may hold for its full-half-spectrum rows."""

    raw = os.environ.get(_PROJECTION_CALL_MAX_BYTES_ENV, "").strip()
    if not raw:
        return _DEFAULT_PROJECTION_CALL_MAX_BYTES
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{_PROJECTION_CALL_MAX_BYTES_ENV} must be positive, got {raw!r}")
    return value

# Row capacities are powers of two so a chunk decomposes into whole M-step
# blocks; the ladder is truncated at run time by the projection byte budget,
# because the local route projects a chunk's rows rather than gathering a
# per-iteration cache.
_DEFAULT_ROW_CAPACITY_LADDER = (1024, 4096, 16384, 65536)
_DEFAULT_IMAGE_CAPACITY_LADDER = (32, 128, 512)

__all__ = [
    "RESIDENT_LOCAL_SEARCH_ENV",
    "compute_local_search_resident",
    "require_resident_local_configuration",
    "resident_local_search_requested",
]


def resident_local_search_requested() -> bool:
    """Return whether ``RECOVAR_LOCAL_SEARCH_RESIDENT`` selects this driver."""

    return parse_env_flag(RESIDENT_LOCAL_SEARCH_ENV, default=False)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(
            "The device-resident local-search pass 2 "
            f"({RESIDENT_LOCAL_SEARCH_ENV}=1) does not implement this configuration: "
            f"{message}. Clear the flag to use the exact local engine; this path "
            "never falls back silently."
        )


def require_resident_local_configuration(**kwargs) -> None:
    """Raise unless this is the K=1 local fine pass 2 in its production shape."""

    _require(kwargs["class_log_priors"] is None, "K-class local search keeps its own engine")
    _require(
        not bool(kwargs["score_only"]),
        "the pass-1 parent probe stays on the exact local engine; its RELION "
        "maximum_significants cap is not in the segmented posterior's contract",
    )
    _require(
        not (bool(kwargs["disable_adjoint_y"]) or bool(kwargs["disable_adjoint_ctf"])),
        "a score-only pass has no M-step to make resident",
    )
    _require(bool(kwargs["mstep_relion_x_half"]), "the RELION x-half M-step is required")
    _require(bool(kwargs["accumulate_noise"]), "the production pass accumulates noise statistics")
    _require(
        bool(kwargs["reconstruct_significant_only"]),
        "the resident M-step reconstructs from RELION's pruned fine weights, "
        "which the zero-oversampling local route does not request",
    )
    _require(
        kwargs["max_significants"] is None or int(kwargs["max_significants"]) <= 0,
        "a maximum_significants cap on the fine support is not in the segmented "
        "posterior's contract",
    )
    _require(
        bool(kwargs["stats_use_reconstruction_probs"]),
        "the resident statistics stage accumulates the pruned reconstruction posterior",
    )
    _require(
        not bool(kwargs["use_float64_scoring"]) and not bool(kwargs["use_float64_projections"]),
        "float64 local search is a diagnostic mode",
    )
    _require(
        bool(kwargs["relion_exact_score_translation"]),
        "the fused translate-and-score kernel needs RELION translation angles",
    )
    _require(bool(kwargs["half_spectrum_scoring"]), "RELION half-spectrum scoring is required")
    _require(
        kwargs["relion_projector_half"] is not None
        and kwargs["relion_projector_r_max"] is not None,
        "the resident row projection uses the RELION PPref projector",
    )
    _require(
        not bool(kwargs["mstep_subtract_ctf_projection"]),
        "subtracting the projected reference in the M-step is a diagnostic mode",
    )
    _require(
        kwargs["normalization_log_z"] is None
        and kwargs["normalization_log_evidence"] is None,
        "an externally supplied normalizer belongs to the broad-denominator probe",
    )
    _require(
        not bool(kwargs["return_reconstruction_sample_indices"]),
        "significant-sample capture belongs to the pass-1 parent probe",
    )
    _require(
        kwargs["group_ids"] is not None,
        "the resident statistics stage accumulates RELION's group scale terms",
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
    # RELION's final all-data iteration scores at current_size == ori_size.
    # There the exact local engine scores the whole centred half, corners of
    # the FFTW rectangle included, while RELION's radial support (which every
    # windowed size uses, and which the RELION Wavg rectangle requires) stops
    # at |k| <= current_size/2. Running this pass on the radial support was
    # measured on the 8x8 unit fixture to move the maps by 0.45 relative L2 and
    # to flip a winner, so it is a change of scoring support, not of layout.
    # Deciding which support the final iteration should use is a scientific
    # question outside a speed ticket, so refuse it here.
    _require(
        bool(kwargs["use_window"]),
        "RELION's final all-data current_size equals the image box, where the "
        "exact local engine scores the full rectangle and RELION's radial "
        "support does not; choosing between them is a scientific decision, "
        "not a layout change",
    )
    for name in rp._DIAGNOSTIC_DIR_ENVS:
        _require(
            not os.environ.get(name, "").strip(),
            f"the diagnostic dump {name} is set; this driver emits no dumps",
        )
    for name in rp._DIAGNOSTIC_FLAG_ENVS:
        _require(
            not parse_env_flag(name, default=False),
            f"the diagnostic flag {name} is set; this driver has no such arm",
        )


def _cap_row_capacity_ladder(
    ladder: tuple,
    *,
    n_score_pixels: int,
    n_recon_pixels: int,
    max_bytes: int,
) -> tuple:
    """Drop row classes whose resident projections exceed the projection budget.

    The global route amortizes one projection per fine rotation across every
    bucket that uses it; local search has no such grid, so a chunk's own
    projections are the memory that decides its row capacity. At the final
    all-data state (current size 256) a row costs about half a megabyte, which
    is why the largest classes disappear there and survive at current size 92.
    """

    per_row = resident_row_projection_bytes(
        n_score_pixels=n_score_pixels, n_recon_pixels=n_recon_pixels
    )
    cap = max(int(max_bytes) // max(per_row, 1), 1)
    kept = tuple(value for value in ladder if int(value) <= cap)
    return kept if kept else (int(ladder[0]),)


def compute_local_search_resident(
    experiment_dataset,
    mean,
    noise_variance,
    local_layout,
    disc_type,
    *,
    current_size,
    reconstruction_current_size=None,
    accumulate_noise=False,
    projection_padding_factor=1,
    reconstruction_padding_factor=1,
    score_with_masked_images=True,
    half_spectrum_scoring=False,
    relion_exact_score_translation=False,
    projection_relion_texture_interp=None,
    projection_relion_acc_double_floorf_quirk=False,
    projection_force_jax=False,
    projection_mask_current_image_disk=False,
    relion_projector_half=None,
    relion_projector_r_max=None,
    use_float64_scoring=False,
    use_float64_projections=False,
    square_window=False,
    image_corrections=None,
    scale_corrections=None,
    group_ids=None,
    scale_correction_group_count=None,
    scale_correction_data_vs_prior=None,
    image_pre_shifts=None,
    mstep_subtract_ctf_projection=False,
    mstep_relion_x_half=False,
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    reconstruct_significant_only=False,
    adaptive_fraction=0.999,
    max_significants=-1,
    return_best_pose_details=False,
    return_significant_counts=False,
    return_reconstruction_sample_indices=False,
    return_profile=False,
    stats_use_reconstruction_probs=True,
    translation_prior_centers=None,
    normalization_log_z=None,
    normalization_log_evidence=None,
    include_unweighted_norm_high_shell=True,
    source_faithful_spectrum_norm=False,
    class_log_priors=None,
    score_only=False,
) -> LocalEMResult:
    """Run one K=1 local-search fine pass 2 on the device-resident stages.

    Returns the same :class:`~recovar.em.helpers.types.LocalEMResult` the exact
    local engine returns for this configuration. See the module docstring for
    the three arithmetic differences that are deliberate and for why the
    pass-1 parent probe is not routed here.
    """

    from recovar import cuda_backproject

    overall_t0 = time.time()
    image_shape = experiment_dataset.image_shape
    volume_shape = experiment_dataset.volume_shape
    n_images = int(experiment_dataset.n_units)

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
        relion_firstiter_score_mode="gaussian",
        use_exact_relion_gaussian=True,
        use_float64_scoring=use_float64_scoring,
    )

    # ``run_local_em_exact`` uses this flag as passed rather than resolving it
    # against the environment, so do the same: it selects RELION's powerClass
    # shell spectrum for the image-power statistics and, with it, the
    # deterministic float64 norm reduction. It does not switch on RELION's
    # exact BPref operands here, because the exact local engine never enables
    # those on its production path.
    resolved_spectrum_norm = bool(source_faithful_spectrum_norm)
    scale_groups_available = group_ids is not None
    relion_wavg_atomic_scale_aa = bool(
        accumulate_noise
        and scale_groups_available
        and parse_env_flag(_RELION_WAVG_ATOMIC_SCALE_AA_ENV, default=True)
    )
    relion_wavg_atomic_direct_noise, relion_wavg_atomic_direct_norm = _relion_wavg_direct_modes(
        accumulate_noise=bool(accumulate_noise),
        scale_groups_available=scale_groups_available,
        scale_aa_enabled=bool(relion_wavg_atomic_scale_aa),
        direct_noise_only_default=True,
    )
    require_resident_local_configuration(
        class_log_priors=class_log_priors,
        score_only=score_only,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        mstep_relion_x_half=mstep_relion_x_half,
        accumulate_noise=accumulate_noise,
        reconstruct_significant_only=reconstruct_significant_only,
        max_significants=max_significants,
        stats_use_reconstruction_probs=stats_use_reconstruction_probs,
        use_float64_scoring=use_float64_scoring,
        use_float64_projections=use_float64_projections,
        relion_exact_score_translation=relion_exact_score_translation,
        half_spectrum_scoring=half_spectrum_scoring,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=relion_projector_r_max,
        mstep_subtract_ctf_projection=mstep_subtract_ctf_projection,
        normalization_log_z=normalization_log_z,
        normalization_log_evidence=normalization_log_evidence,
        return_reconstruction_sample_indices=return_reconstruction_sample_indices,
        group_ids=group_ids,
        use_window=budget_window_spec.use_window,
        relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
        relion_wavg_atomic_direct_noise=relion_wavg_atomic_direct_noise,
        relion_wavg_atomic_direct_norm=relion_wavg_atomic_direct_norm,
    )
    _require(
        jax.default_backend() == "gpu" and cuda_backproject.custom_cuda_requested(),
        "every resident stage is a CUDA FFI target",
    )

    # ---- candidate rows ---------------------------------------------------
    table_t0 = time.time()
    tables = tables_from_local_layout(
        local_layout, rotation_dtype=precision_policy.score_real_dtype
    )
    if tables.n_images != n_images:
        raise ValueError(
            f"the local layout covers {tables.n_images} images but the half has {n_images}"
        )
    n_fine_trans = tables.n_trans
    fine_translations = np.asarray(
        tables.translation_grid, dtype=precision_policy.score_real_dtype
    )

    # ---- window, weights and lookups --------------------------------------
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
        log_label="Resident local pass-2",
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
        relion_firstiter_score_mode="gaussian",
        use_float64_scoring=use_float64_scoring,
    )
    del half_weights
    relion_score_full_to_compact = jnp.asarray(
        _relion_cuda_fine_full_to_compact_lookup(image_shape, current_size, window_indices_np),
        dtype=jnp.int32,
    )
    # The exact local engine casts sigma2 to the score real dtype before it
    # divides by it (local_big_jit.py, the non-BPref-operand branch). Passing a
    # float64 sigma2 through instead makes every downstream operand double:
    # ctf^2/sigma2 becomes float64, the translated reconstruction tile becomes
    # complex128, and _prepare_bucket_io then picks relion_translate_score_f64
    # where the exact engine picks the f32 kernel. That is a different
    # computation and twice the memory on the largest per-chunk array.
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(
        noise_variance, image_shape
    ).squeeze().astype(precision_policy.score_real_dtype)
    relion_score_translation_angles = _relion_cuda_score_translation_angles_if_available(
        fine_translations,
        image_shape,
        enabled=True,
        dtype=np.float32,
    )
    if relion_score_translation_angles is None:
        raise ValueError("the resident local scorer requires RELION translation angles")
    # Supply the phase table even in the windowed case. Handed ``None``, the
    # compact prepare builds it from a float64 cached lattice, which is a
    # complex128 table; the exact local engine builds it at the score real
    # dtype and windows it. Windowing a supplied table is what that engine
    # does, so supply it.
    translation_phases_half = half_translation_phase_table(
        fine_translations, image_shape, dtype=precision_policy.score_real_dtype
    ).astype(precision_policy.score_complex_dtype)

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
        group_ids, scale_correction_group_count, n_images=n_images
    )

    # ---- accumulator layout (identical to the exact engine's x-half BPref) -
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

    # ---- projection setup -------------------------------------------------
    # The refinement loop hands local search a projector with a singleton class
    # axis; the exact local engine normalizes it with the same helper before
    # projecting, so do that here rather than letting the projector unpack a
    # 4-D shape.
    # The exact local engine selects the projection precision with
    # ``cast_relion_projector_for_execution`` (complex64 unless double
    # projection is requested) and then normalizes the slab. The precision
    # decides float32 projection arithmetic and whether the texture projector
    # or the vmapped fallback runs, so an arm with any other precision is both
    # a different computation and a differently timed one than its control.
    # The compact engine's RECOVAR_SPARSE_PASS2_PROJECTOR_COMPLEX64 gate is not
    # read here, because the exact local engine does not read it. Do exactly
    # what the exact local engine does.
    relion_projector_half = cast_relion_projector_for_execution(
        relion_projector_half, use_float64_projections=use_float64_projections
    )
    relion_projector_half = prepare_local_projector_slab(
        relion_projector_half, path_label="device-resident local projector path"
    )
    logger.info(
        "Resident local pass-2 projector: slab dtype=%s shape=%s r_max=%s "
        "(the exact local engine's execution precision)",
        relion_projector_half.dtype,
        tuple(relion_projector_half.shape),
        relion_projector_r_max,
    )
    projection_kwargs = _projection_kwargs_for_relion_score_window(
        window_spec.projection_kwargs(return_abs2=False),
        use_relion_projector=True,
        current_size=current_size,
    )
    # The exact local engine passes all four of these to the same projector;
    # ``relion_texture_interp=None`` means "resolve as strict parity does",
    # which is RELION's CUDA texture interpolator.
    projection_kwargs["relion_texture_interp"] = projection_relion_texture_interp
    projection_kwargs["relion_acc_double_floorf_quirk"] = bool(
        projection_relion_acc_double_floorf_quirk
    )
    projection_kwargs["force_jax"] = bool(projection_force_jax)
    projection_kwargs["mask_current_image_disk"] = bool(projection_mask_current_image_disk)

    # ---- capacity plan ----------------------------------------------------
    row_ladder = _cap_row_capacity_ladder(
        parse_env_capacity_ladder(_ROW_CAPACITY_LADDER_ENV, _DEFAULT_ROW_CAPACITY_LADDER),
        n_score_pixels=n_windowed,
        n_recon_pixels=n_recon_windowed,
        max_bytes=_projection_cache_max_bytes_for_pass(device_memory_bytes),
    )
    image_ladder = rp._cap_image_capacity_ladder(
        parse_env_capacity_ladder(_IMAGE_CAPACITY_LADDER_ENV, _DEFAULT_IMAGE_CAPACITY_LADDER),
        n_fine_trans=n_fine_trans,
        n_recon_pixels=n_recon_windowed,
        max_tile_bytes=_max_translation_tile_bytes_for_pass(
            device_memory_bytes, has_external_normalization=False
        ),
    )
    mstep_block_rows = rp._resolve_mstep_block_rows(
        n_recon_pixels=n_recon_windowed,
        max_block_bytes=_max_adjoint_block_bytes_for_pass(device_memory_bytes),
        row_capacity_ladder=row_ladder,
    )
    # Bound one projector call by the array it actually materializes. The
    # compact projection-block helper returns *full half-spectrum* rows and
    # windows them afterwards (projection.py, the dense_scale multiply runs on
    # proj_half before any gather), so the transient is
    # rows x n_half x itemsize(Projector::data), not rows x windowed pixels.
    # Budgeting on the window is what refused a 16.1 GiB allocation twice: at
    # current size 92 with 32768 rows, and again at 52 where the window is
    # 1104 px but the materialized row is still 33024.
    #
    # The exact local engine does not hit this because it hands the projector
    # its compact pixel indices; this driver goes through the shared compact
    # helper, so it pays the full row and must budget for it.
    n_projection_pixels = int(getattr(window_spec, "n_projection", n_recon_windowed))
    projector_slab_bytes = int(jnp.asarray(relion_projector_half).dtype.itemsize)
    projection_call_max_bytes = _projection_call_transient_max_bytes()
    projection_block_rows = max(
        1, projection_call_max_bytes // max(n_half * projector_slab_bytes, 1)
    )
    chunks = plan_local_capacity_chunks(
        tables,
        row_capacity_ladder=row_ladder,
        image_capacity_ladder=image_ladder,
    )
    table_s = time.time() - table_t0
    per_row_bytes = resident_row_projection_bytes(
        n_score_pixels=n_windowed, n_recon_pixels=n_recon_windowed
    )
    logger.info(
        "Resident local pass-2 plan: %d images, %d candidate rows, %d translations -> %d chunks "
        "(row capacities %s, image capacities %s, M-step block rows %d, projection block rows %d); "
        "row projections %.2f KiB/row, largest chunk %.2f GiB, projection window %d px, "
        "projector slab %d B/element; setup %.2fs",
        tables.n_images,
        tables.n_rows,
        n_fine_trans,
        len(chunks),
        ",".join(str(v) for v in row_ladder),
        ",".join(str(v) for v in image_ladder),
        mstep_block_rows,
        projection_block_rows,
        per_row_bytes / 1024.0,
        max((int(chunk.row_capacity) for chunk in chunks), default=0)
        * per_row_bytes
        / float(1024**3),
        n_projection_pixels,
        projector_slab_bytes,
        table_s,
    )

    # ---- per-image resident operands --------------------------------------
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
        score_mode="gaussian",
        window_indices=window_indices,
        recon_window_indices=recon_window_indices,
        translation_phases_half=translation_phases_half,
        relion_score_translation_angles=relion_score_translation_angles,
        return_windowed_shifted=windowed_prepare,
        relion_exact_normalized_cc_operands=False,
        # The exact local engine runs its production path with the plain
        # ``CTF^2 / sigma2`` operand order, not RELION's RFLOAT-square order,
        # so keep that here rather than silently switching operand families.
        relion_exact_bpref_operands=False,
    )
    fine_translation_prior_2d = np.asarray(
        tables.translation_log_prior, dtype=precision_policy.score_real_dtype
    )

    # ---- statistics accumulators ------------------------------------------
    stats_config = resolve_statistics_config(
        n_shells=n_shells,
        n_fine_trans=n_fine_trans,
        n_images=n_images,
        n_coarse_rot=tables.n_posterior_bins,
        n_scale_groups=n_scale_groups,
        current_size=current_size,
        include_unweighted_high_shell=include_unweighted_norm_high_shell,
        use_exact_relion_gaussian=True,
        relion_wavg_atomic_direct_noise=relion_wavg_atomic_direct_noise,
        relion_wavg_atomic_scale_aa=relion_wavg_atomic_scale_aa,
        accumulate_scale=scale_groups_available,
        source_faithful_spectrum_norm=resolved_spectrum_norm,
    )
    stats = make_resident_statistics(
        stats_config, max_posterior_dtype=precision_policy.score_real_dtype
    )
    image_tables = rp._ChunkImageTables(
        shell_indices_half=jnp.asarray(shell_indices_half, dtype=jnp.int32),
        wavg_shell_indices=jnp.asarray(relion_wavg_rectangle.shell_indices, dtype=jnp.int32),
        wavg_scale_pixel_mask=jnp.asarray(scale_pixel_mask_rect_np, dtype=bool),
        translation_sqdist_ang=None,
    )

    Ft_y_total = jnp.zeros(recon_volume_size, dtype=recon_y_accum_dtype)
    Ft_ctf_total = jnp.zeros(recon_volume_size, dtype=recon_ctf_accum_dtype)
    exact_positions_device = jnp.asarray(relion_wavg_rectangle.exact_positions, dtype=jnp.int32)
    rect_indices_device = jnp.asarray(relion_wavg_rectangle.centered_indices, dtype=jnp.int32)
    noise_variance_for_noise_device = jnp.asarray(noise_variance_for_noise)
    shell_indices_noise_device = jnp.asarray(shell_indices_noise, dtype=jnp.int32)
    max_adjoint_block_bytes = _max_adjoint_block_bytes_for_pass(device_memory_bytes)

    scale_corrections_np = (
        None
        if scale_corrections is None
        else np.asarray(scale_corrections, dtype=precision_policy.score_real_dtype)
    )
    translation_prior_centers_np = None
    if translation_prior_centers is not None:
        from recovar.em.helpers.translation_prior import validate_translation_prior_centers

        translation_prior_centers_np = validate_translation_prior_centers(
            translation_prior_centers,
            n_images=n_images,
            n_dims=int(fine_translations.shape[1]),
        )

    # ---- chunk loop --------------------------------------------------------
    significant_counts = (
        np.zeros(n_images, dtype=np.int32) if return_significant_counts else None
    )
    loop_t0 = time.time()
    for chunk in chunks:
        Ft_y_total, Ft_ctf_total, stats = _run_resident_local_chunk(
            chunk,
            tables=tables,
            experiment_dataset=experiment_dataset,
            bucket_io_kwargs=bucket_io_kwargs,
            fine_translation_prior_2d=fine_translation_prior_2d,
            half_weights=half_weights_windowed,
            full_to_compact=relion_score_full_to_compact,
            translation_angles=relion_score_translation_angles,
            n_score_pixels=int(n_windowed),
            mean=mean,
            volume_shape=volume_shape,
            disc_type=disc_type,
            projection_kwargs=projection_kwargs,
            projection_block_rows=projection_block_rows,
            projection_padding_factor=projection_padding_factor,
            relion_projector_half=relion_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            precision_policy=precision_policy,
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
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=resolved_spectrum_norm,
            stats=stats,
            stats_config=stats_config,
            image_tables=image_tables,
            Ft_y_total=Ft_y_total,
            Ft_ctf_total=Ft_ctf_total,
            cuda_backproject=cuda_backproject,
            significant_counts=significant_counts,
        )
    loop_s = time.time() - loop_t0

    # ---- finalize ----------------------------------------------------------
    Ft_y_total, Ft_ctf_total = enforce_half_volume_x0(
        Ft_y_total,
        Ft_ctf_total,
        recon_volume_shape,
        logger=logger,
        label="Resident local pass-2",
    )
    Ft_y_total, Ft_ctf_total = relion_x_half_accumulators_to_public_layout(
        Ft_y_total,
        Ft_ctf_total,
        recon_volume_shape,
    )

    finalized = finalize_statistics(stats, config=stats_config)
    # ``best_fine_rotation_indices`` carries the winner's position in the
    # layout's flat row order, because that is what the chunk wrote into
    # ``best_fine_rot``. Every pose field is a lookup at that row.
    best_row = np.asarray(finalized.best_fine_rotation_indices, dtype=np.int64)
    best_translation = np.asarray(finalized.best_translation_indices, dtype=np.int64)
    if np.any(best_row < 0):
        raise RuntimeError("Resident local pass 2: an image has no winning candidate row")
    # RELION's hard assignment counts in the layout's own fine rotation ids,
    # exactly like ``local_bucket_stages.encode_hard_assignment``; the resident
    # statistics stage's image-local encoding is not that convention.
    hard_assignments = (
        tables.row_rotation_id[best_row].astype(np.int64) * np.int64(n_fine_trans)
        + best_translation
    )

    relion_stats = make_relion_stats(
        log_evidence_per_image=finalized.log_evidence_per_image,
        best_log_score_per_image=finalized.best_log_score_per_image,
        max_posterior_per_image=finalized.max_posterior_per_image,
        rotation_posterior_sums=finalized.rotation_posterior_sums,
    )
    noise_stats = make_noise_stats(
        wsum_sigma2_noise=finalized.wsum_sigma2_noise,
        wsum_img_power=finalized.wsum_img_power,
        wsum_sigma2_offset=finalized.wsum_sigma2_offset,
        sumw=finalized.sumw,
        wsum_norm_correction=finalized.wsum_norm_correction,
        wsum_scale_correction_xa=finalized.wsum_scale_correction_xa,
        wsum_scale_correction_aa=finalized.wsum_scale_correction_aa,
    )

    best_pose_rotations = best_pose_translations = best_pose_rotation_ids = None
    if return_best_pose_details:
        best_pose_rotations = np.asarray(tables.rotations)[best_row]
        best_pose_translations = np.asarray(tables.translation_grid)[best_translation]
        best_pose_rotation_ids = tables.row_rotation_id[best_row].astype(np.int64)
    best_pose_eulers_deg = (
        None if tables.source_eulers is None else tables.source_eulers[best_row]
    )

    profile = None
    if return_profile:
        profile = {
            "resident_local_chunks": np.int32(len(chunks)),
            "resident_local_rows": np.int64(tables.n_rows),
            "resident_local_loop_time_s": np.float64(loop_s),
            "resident_local_total_time_s": np.float64(time.time() - overall_t0),
            "resident_local_row_projection_bytes": np.int64(per_row_bytes),
            "resident_local_mstep_block_rows": np.int32(mstep_block_rows),
            "resident_local_projection_block_rows": np.int32(projection_block_rows),
        }

    logger.info(
        "Resident local pass-2 done: %d images, %d chunks, %.2fs chunk loop, %.2fs total",
        n_images,
        len(chunks),
        loop_s,
        time.time() - overall_t0,
    )
    return LocalEMResult(
        Ft_y=Ft_y_total,
        Ft_ctf=Ft_ctf_total,
        hard_assignments=hard_assignments,
        stats=relion_stats,
        best_pose_rotations=best_pose_rotations,
        best_pose_translations=best_pose_translations,
        best_pose_rotation_ids=best_pose_rotation_ids,
        noise_stats=noise_stats,
        profile=profile,
        significant_counts=significant_counts,
        best_pose_eulers_deg=best_pose_eulers_deg,
    )


def _local_chunk_segment_offsets(tables, chunk, n_fine_trans: int) -> np.ndarray:
    """Cell offsets of each chunk image slot, in the segmented handler's units.

    Same contract as the global driver's: slot ``b`` owns the cells of image
    ``image_start + b``; padded slots and rows past ``n_valid_rows`` are covered
    by no segment, which the handler treats exactly as an all ``-inf``
    rectangular row.
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


def _run_resident_local_chunk(
    chunk,
    *,
    tables,
    experiment_dataset,
    bucket_io_kwargs,
    fine_translation_prior_2d,
    half_weights,
    full_to_compact,
    translation_angles,
    n_score_pixels,
    mean,
    volume_shape,
    disc_type,
    projection_kwargs,
    projection_block_rows,
    projection_padding_factor,
    relion_projector_half,
    relion_projector_r_max,
    precision_policy,
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
    accumulate_noise,
    source_faithful_spectrum_norm,
    stats,
    stats_config,
    image_tables,
    Ft_y_total,
    Ft_ctf_total,
    cuda_backproject,
    significant_counts,
):
    """Every resident stage for one local capacity chunk.

    The only host work inside is the chunk's operand upload, the T7 offsets
    readback the segmented posterior performs internally, and the optional
    significance-count pull; no per-chunk result is otherwise brought back.
    """

    row_capacity = int(chunk.row_capacity)
    image_capacity = int(chunk.image_capacity)
    n_valid_rows = int(chunk.n_valid_rows)
    n_valid_images = int(chunk.n_valid_images)
    image_indices = np.arange(chunk.image_start, chunk.image_stop, dtype=np.int64)

    profile = parse_env_flag(_CHUNK_PROFILE_ENV, default=False)
    marks: dict[str, float] = {}

    def mark(name, *values):
        if not profile:
            return
        if values:
            jax.block_until_ready(values)
        marks[name] = time.time()

    mark("t0")
    host_chunk = materialize_local_chunk(tables, chunk)
    row_image_local = jnp.asarray(host_chunk["row_image_local"], dtype=jnp.int32)
    row_log_prior = jnp.asarray(host_chunk["row_log_prior"], dtype=jnp.float32)
    row_mask_bits = (
        None
        if host_chunk["row_mask_bits"] is None
        else jnp.asarray(host_chunk["row_mask_bits"], dtype=jnp.uint8)
    )
    image_ids = jnp.asarray(host_chunk["image_ids"], dtype=jnp.int32)
    n_valid_rows_device = jnp.asarray(host_chunk["n_valid_rows"], dtype=jnp.int32)
    n_valid_images_device = jnp.asarray(host_chunk["n_valid_images"], dtype=jnp.int32)
    row_is_valid = jnp.arange(row_capacity, dtype=jnp.int32) < n_valid_rows_device
    kernel_row_image_ids = jnp.where(row_is_valid, row_image_local, jnp.int32(-1))

    # --- stage 0: this chunk's operands, both families, at image capacity ---
    # Upstream merged the per-half score preparation into this call, so
    # ``_prepare_bucket_io`` runs once per image instead of twice and every
    # per-chunk program is keyed on the image-capacity class rather than on
    # the chunk's occupancy.
    recon = rp._prepare_chunk_reconstruction_operands(
        chunk=chunk,
        image_indices=image_indices,
        experiment_dataset=experiment_dataset,
        bucket_io_kwargs=bucket_io_kwargs,
        windowed_prepare=windowed_prepare,
        recon_window_indices=recon_window_indices,
        n_fine_trans=int(n_fine_trans),
        n_recon_windowed=int(n_recon_windowed),
        image_shape=image_shape,
        current_size=current_size,
        use_exact_relion_gaussian=True,
        accumulate_noise=accumulate_noise,
        source_faithful_spectrum_norm=bool(source_faithful_spectrum_norm),
        score_window_indices=window_indices,
        fine_translation_prior_2d=fine_translation_prior_2d,
        score_real_dtype=precision_policy.score_real_dtype,
        relion_score_translation_angles=translation_angles,
        rect_indices_device=rect_indices_device,
        exact_positions_device=exact_positions_device,
        scale_corrections_np=scale_corrections_np,
        group_ids_np=group_ids_np,
    )

    mark("operands", recon["shifted_recon"], recon["score_input"])

    # --- stages 1-2: project this chunk's own rows -------------------------
    score_proj, recon_proj, recon_abs2 = project_resident_rows(
        mean,
        jnp.asarray(host_chunk["rotations"], dtype=precision_policy.score_real_dtype),
        image_shape,
        volume_shape,
        disc_type,
        score_indices=window_indices,
        recon_indices=recon_window_indices,
        max_projected_rotations=int(projection_block_rows),
        output_complex_dtype=precision_policy.score_complex_dtype,
        output_abs2_dtype=precision_policy.score_real_dtype,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=relion_projector_r_max,
        projection_padding_factor=projection_padding_factor,
        **projection_kwargs,
    )

    mark("project", score_proj, recon_proj, recon_abs2)

    # --- stage 3: score ----------------------------------------------------
    # Chunk-local image slots address the chunk's own operands.
    chunk_image_ids = jnp.where(
        jnp.arange(image_capacity, dtype=jnp.int32) < n_valid_images_device,
        jnp.arange(image_capacity, dtype=jnp.int32),
        jnp.int32(-1),
    )
    scored = score_resident_projected_chunk(
        score_proj,
        row_image_local,
        row_log_prior,
        row_mask_bits,
        n_valid_rows_device,
        chunk_image_ids,
        recon["score_input"],
        recon["corr_img_score"],
        recon["highres_xi2_half"],
        recon["translation_prior"],
        half_weights=half_weights,
        translation_angles=translation_angles,
        full_to_compact=full_to_compact,
        logical_current_size=jnp.asarray(current_size, dtype=jnp.int32),
        row_capacity=row_capacity,
        image_capacity=image_capacity,
        n_fine_trans=int(n_fine_trans),
        n_score_pixels=int(n_score_pixels),
    )
    del score_proj
    scores_flat = jnp.asarray(scored.scores, dtype=jnp.float32).reshape(-1)
    mark("score", scores_flat)

    # --- stage 4: segmented RELION float32 fine posterior -------------------
    segment_offsets_np = _local_chunk_segment_offsets(tables, chunk, n_fine_trans)
    segment_offsets = jnp.asarray(segment_offsets_np, dtype=jnp.int32)
    log_z = cuda_backproject.sparse_pass2_segmented_log_z_f64(
        scores_flat, segment_offsets, n_valid_images_device
    )
    posterior = cuda_backproject.sparse_pass2_segmented_posterior_f32(
        scores_flat,
        segment_offsets,
        n_valid_images_device,
        log_z,
        jnp.ones((image_capacity,), dtype=jnp.float32),
        adaptive_fraction=float(adaptive_fraction),
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
        n_significant,
        _sum_weight,
        _threshold,
    ) = posterior
    row_posterior = jnp.asarray(reconstruction_probs, dtype=jnp.float32).reshape(
        row_capacity, int(n_fine_trans)
    )
    mark("posterior", row_posterior, log_z_out, best_cell_index)
    if significant_counts is not None:
        significant_counts[chunk.image_start : chunk.image_stop] = np.asarray(
            jax.device_get(n_significant)[:n_valid_images], dtype=np.int32
        )

    mstep_rotations = jnp.asarray(
        host_chunk["mstep_rotations"], dtype=precision_policy.score_real_dtype
    )

    # P3-G: with the flag on the three chunk-wide arrays go to the M-step entry
    # point whole and the block program slices them inside the jit; with it off
    # the Python callback slices them per block, three eager dispatches each
    # time, which is the path this change is measured against.
    block_row_program = parse_env_flag(_BLOCK_ROW_PROGRAM_ENV, default=False)
    if block_row_program:
        block_projections = None
        chunk_projections = (recon_proj, recon_abs2, mstep_rotations)
    else:
        chunk_projections = None

        def block_projections(start, stop):
            # A slice, not a gather: the layout's flat order is the chunk's row
            # order, so a block's projections are contiguous in the arrays this
            # chunk just produced.
            return recon_proj[start:stop], recon_abs2[start:stop], mstep_rotations[start:stop]

    (
        Ft_y_total,
        Ft_ctf_total,
        wavg_triplet_pixels,
        block_noise_shells,
        a2_per_image,
        xa_per_image,
    ) = rp.run_resident_mstep_blocks(
        block_projections,
        chunk_projections=chunk_projections,
        row_capacity=row_capacity,
        n_valid_rows=n_valid_rows,
        mstep_block_rows=int(mstep_block_rows),
        image_capacity=image_capacity,
        row_image_local=row_image_local,
        kernel_row_image_ids=kernel_row_image_ids,
        row_posterior=row_posterior,
        recon=recon,
        n_rect=int(n_rect),
        n_shells=int(stats_config.n_shells),
        n_recon_windowed=int(n_recon_windowed),
        noise_variance_for_noise=noise_variance_for_noise,
        shell_indices_noise=shell_indices_noise,
        exact_positions_device=exact_positions_device,
        Ft_y_total=Ft_y_total,
        Ft_ctf_total=Ft_ctf_total,
        image_shape=image_shape,
        recon_volume_shape=recon_volume_shape,
        mstep_current_size=mstep_current_size,
        relion_x_half_recon_indices=relion_x_half_recon_indices,
        max_adjoint_block_bytes=max_adjoint_block_bytes,
        cuda_backproject=cuda_backproject,
    )

    mark("mstep", Ft_y_total, Ft_ctf_total, wavg_triplet_pixels, block_noise_shells)

    # --- stage 7: image-level statistics ------------------------------------
    translation_sqdist_ang = image_tables.translation_sqdist_ang
    if translation_prior_centers_np is not None:
        from recovar.em.helpers.translation_prior import (
            translation_prior_centers_for_images,
            translation_sqdist_angstrom,
        )

        # Build the centres at capacity on the host so the squared-distance
        # program is keyed on the capacity class, not the occupancy; padded
        # rows multiply a zero posterior, so their value is never observable.
        padded_image_indices = rp._pad_batch_to_capacity(
            np.asarray(image_indices).reshape(-1, 1), image_capacity
        ).reshape(-1)
        centers = translation_prior_centers_for_images(
            translation_prior_centers_np,
            padded_image_indices,
            batch_size=image_capacity,
        )
        translation_sqdist_ang = rp._zero_padded_images(
            jnp.asarray(translation_sqdist_angstrom(fine_translations, centers, voxel_size)),
            jnp.asarray(np.arange(image_capacity) < n_valid_images, dtype=bool),
        )
    chunk_tables = image_tables._replace(translation_sqdist_ang=translation_sqdist_ang)

    image_row_start_np = segment_offsets_np.astype(np.int64)[:image_capacity] // int(n_fine_trans)
    image_row_count_np = (
        segment_offsets_np.astype(np.int64)[1:] - segment_offsets_np.astype(np.int64)[:-1]
    ) // int(n_fine_trans)
    image_row_start = jnp.asarray(image_row_start_np, dtype=jnp.int64)
    image_row_count = jnp.asarray(image_row_count_np, dtype=jnp.int64)

    best_row_local = jnp.asarray(best_cell_index, dtype=jnp.int64) // jnp.int64(n_fine_trans)
    slot_is_valid = jnp.arange(image_capacity, dtype=jnp.int32) < n_valid_images_device
    invalid_best = slot_is_valid & ((best_row_local < 0) | (best_row_local >= image_row_count))
    best_chunk_row = jnp.clip(
        image_row_start + best_row_local, 0, jnp.int64(max(row_capacity - 1, 0))
    ).astype(jnp.int32)
    # The winner's global row in the layout's flat order; the pose decode reads
    # rotations, M-step rotations, fine ids and source Eulers at that row.
    best_global_row = jnp.int64(int(chunk.row_start)) + best_chunk_row.astype(jnp.int64)

    row_posterior_bin = jnp.asarray(host_chunk["row_posterior_id"], dtype=jnp.int32)
    chunk_operands = rp._ChunkImageOperands(
        row_posterior=row_posterior,
        row_image_local=row_image_local,
        row_coarse_rot=jnp.where(
            row_is_valid, row_posterior_bin, jnp.int32(int(stats_config.n_coarse_rot))
        ),
        image_ids=image_ids,
        group_ids=recon["group_ids"],
        processed_image_half=recon["processed_image_half"],
        relion_norm_high_shell=recon["relion_norm_high_shell"],
        wavg_triplet_pixels=wavg_triplet_pixels,
        block_noise_shells=block_noise_shells,
        a2_per_image=a2_per_image,
        xa_per_image=xa_per_image,
        class_log_z=jnp.asarray(log_z_out, dtype=jnp.float64),
        min_diff2=scored.min_diff2,
        best_log_score=best_log_score,
        max_posterior=max_posterior,
        best_cell_index=jnp.asarray(best_cell_index, dtype=jnp.int64),
        best_fine_rot=best_global_row,
    )
    stats = rp._accumulate_chunk_image_terms(
        stats, chunk_operands, chunk_tables, config=stats_config
    )
    stats = stats._replace(
        invalid_best_rows=stats.invalid_best_rows + jnp.sum(invalid_best.astype(jnp.int64))
    )
    mark("stats", stats)
    if profile:
        order = ("t0", "operands", "project", "score", "posterior", "mstep", "stats")
        spans = {
            name: marks[name] - marks[prev]
            for prev, name in zip(order, order[1:])
            if name in marks and prev in marks
        }
        n_blocks = len(range(0, min(row_capacity, max(n_valid_rows, 1)), int(mstep_block_rows))) or 1
        logger.info(
            "Resident local chunk profile: images=%d/%d rows=%d/%d row_pad=%.1f%% "
            "blocks=%d proj_rows=%d recon_tile=%s wavg_tile=%s | %s | chunk=%.3fs",
            n_valid_images, image_capacity, n_valid_rows, row_capacity,
            100.0 * (row_capacity - n_valid_rows) / max(row_capacity, 1),
            n_blocks, row_capacity,
            f"{recon['shifted_recon'].dtype}{tuple(recon['shifted_recon'].shape)}",
            f"{recon['raw_translated_wavg_rectangle'].dtype}"
            f"{tuple(recon['raw_translated_wavg_rectangle'].shape)}",
            " ".join(f"{k}={v:.3f}s" for k, v in spans.items()),
            marks["stats"] - marks["t0"],
        )
    return Ft_y_total, Ft_ctf_total, stats
