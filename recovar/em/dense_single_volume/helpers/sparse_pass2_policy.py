"""Execution-policy switches of the sparse bucketed pass 2.

The environment-gated opt-ins and per-pass policy values (BPref processing
order and execution buckets, native weighted sums, fused M-step noise, the
RELION wavg direct modes, projection cache, windowed prepare and translation
tiles, compact-pair execution and bucket coalescing). Each reader is the
single owner of its variable; the pass-2 kernels ask these once per pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import numpy as np

from recovar.em.dense_single_volume.helpers import pass2_diagnostics
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_flag, parse_env_nonnegative_int
from recovar.em.dense_single_volume.helpers.sparse_bucket_arrays import (
    _DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION,
    _DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE,
)
from recovar.em.dense_single_volume.helpers.sparse_pass2_budget import (
    _max_images_for_translation_tile,
    _optional_positive_float_env,
    _optional_positive_int_env,
)

_RELION_WAVG_ATOMIC_SCALE_AA_ENV = "RECOVAR_RELION_WAVG_ATOMIC_SCALE_AA"


_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV = (
    "RECOVAR_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL"
)


_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV = (
    "RECOVAR_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY"
)


_DEFAULT_SMALL_BUCKET_COALESCE_SIZE = 128


_DEFAULT_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES = 5_000


_DEFAULT_TAIL_BUCKET_COALESCE_MAX_IMAGES_FUSED_KCLASS = 0


_SMALL_BUCKET_COALESCE_SIZE_ENV = "RECOVAR_SPARSE_PASS2_SMALL_BUCKET_COALESCE_SIZE"


_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES_ENV = "RECOVAR_SPARSE_PASS2_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES"


_TAIL_BUCKET_COALESCE_MAX_IMAGES_ENV = "RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_IMAGES"


_TAIL_BUCKET_COALESCE_MAX_INFLATION_ENV = "RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MAX_INFLATION"


_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE_ENV = "RECOVAR_SPARSE_PASS2_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE"


_COMPACT_KCLASS_PAIRS_CHECK_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_CHECK"


_SPARSE_KCLASS_COMPACT_PAIRS_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS"


_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH"


_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE"


_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES_ENV = (
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES"
)


_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION_ENV = (
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION"
)


_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE_ENV = (
    "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE"
)


_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE_ENV = "RECOVAR_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE"


_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED_ENV = "RECOVAR_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED"


_SPARSE_KCLASS_NATIVE_DUAL_WEIGHTED_SUMS_ENV = (
    "RECOVAR_SPARSE_KCLASS_NATIVE_DUAL_WEIGHTED_SUMS"
)


_SPARSE_KCLASS_FUSED_MSTEP_NOISE_ENV = "RECOVAR_SPARSE_KCLASS_FUSED_MSTEP_NOISE"


_SPARSE_KCLASS_COMPACT_PAIR_MSTEP_ENV = "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MSTEP"


_RELION_POWERCLASS_SPECTRUM_NORM_ENV = "RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM"


_RELION_EXACT_BPREF_OPERANDS_ENV = "RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS"


_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV = "RECOVAR_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE"


_BPREF_REVERSE_PHYSICAL_ORDER_ENV = "RECOVAR_K1_BPREF_REVERSE_PHYSICAL_ORDER"


_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV = "RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE"


_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV = (
    "RECOVAR_K1_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT"
)


_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV = (
    "RECOVAR_K1_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE"
)


_DEFAULT_FRESH_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE = 220


_PASS2_DUMP_CONSERVATIVE_EXECUTION_ENV = "RECOVAR_PASS2_DUMP_CONSERVATIVE_EXECUTION"


_NORM_RESIDUAL_DUMP_ONLY_ENV = "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_ONLY"


_SPARSE_PASS2_PROJECTION_CACHE_ENV = "RECOVAR_SPARSE_PASS2_PROJECTION_CACHE"


_SPARSE_PASS2_CACHED_SCORE_ROT_CHUNK_ENV = "RECOVAR_SPARSE_PASS2_CACHED_SCORE_ROT_CHUNK"


_SPARSE_PASS2_WINDOWED_PREPARE_ENV = "RECOVAR_SPARSE_PASS2_WINDOWED_PREPARE"


_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP_ENV = (
    "RECOVAR_SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP"
)


_SPARSE_PASS2_WINDOWED_TRANSLATION_TILE_MAX_MULTIPLIER_ENV = (
    "RECOVAR_SPARSE_PASS2_WINDOWED_TRANSLATION_TILE_MAX_MULTIPLIER"
)


_DEFAULT_COMPACT_PAIR_MIN_BUCKET_SIZE = 512


_DEFAULT_COMPACT_PAIR_TAIL_BUCKET_COALESCE_MAX_IMAGES = 19


_DEFAULT_ACTIVE_ROW_PAD_MULTIPLE = 1024


_DEFAULT_CACHED_SCORE_ROT_CHUNK_SIZE = 8192


_DEFAULT_WINDOWED_TRANSLATION_TILE_MAX_MULTIPLIER = 4


def _load_bpref_execution_order_local_override(n_images: int) -> np.ndarray | None:
    """Load a fail-closed diagnostic K=1 particle execution permutation."""

    raw_path = os.environ.get(_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV)
    if raw_path is None or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError(
            f"{_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV} must name an existing absolute file",
        )
    order = np.asarray(np.loadtxt(path, dtype=np.int64, ndmin=1), dtype=np.int64).reshape(-1)
    if order.shape != (int(n_images),):
        raise ValueError(
            f"{_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV} must contain {int(n_images)} rows, "
            f"got {order.shape[0]}",
        )
    if not np.array_equal(np.sort(order), np.arange(int(n_images), dtype=np.int64)):
        raise ValueError(f"{_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV} must contain a permutation")
    return order


def _resolve_bpref_processing_order(
    n_images: int,
    *,
    preserve_bpref_particle_order: bool,
) -> np.ndarray | None:
    """Resolve production or diagnostic K=1 BPref execution ordering."""

    diagnostic_order = _load_bpref_execution_order_local_override(n_images)
    reverse_physical_order = parse_env_flag(
        _BPREF_REVERSE_PHYSICAL_ORDER_ENV,
        default=False,
    )
    if reverse_physical_order and not preserve_bpref_particle_order:
        raise ValueError(
            f"{_BPREF_REVERSE_PHYSICAL_ORDER_ENV}=1 requires the guarded fresh "
            "K=1 physical-order path"
        )
    if reverse_physical_order and diagnostic_order is not None:
        raise ValueError(
            f"{_BPREF_REVERSE_PHYSICAL_ORDER_ENV}=1 cannot be combined with "
            f"{_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV}"
        )
    if preserve_bpref_particle_order and diagnostic_order is not None:
        raise ValueError(
            "preserve_bpref_particle_order cannot be combined with "
            f"{_BPREF_EXECUTION_ORDER_LOCAL_FILE_ENV}"
        )
    if preserve_bpref_particle_order:
        order = np.arange(int(n_images), dtype=np.int64)
        return order[::-1].copy() if reverse_physical_order else order
    return diagnostic_order


def _resolve_bpref_execution_bucket_policy(
    *,
    preserve_bpref_particle_order: bool,
    processing_order_group_by_bucket_size: bool,
) -> tuple[int, bool]:
    """Resolve strict-order batching for the fresh-K=1 physical sequence.

    The production default pads consecutive mixed-support particles in bounded
    chunks.  An explicit chunk size overrides that bound.  The older adjacent
    equal-support batching remains available only as an explicit diagnostic.
    Every mode in this helper retains the exact particle sequence.
    """

    explicit_chunk_size = _optional_positive_int_env(
        _BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV,
    )
    batch_consecutive_requested = parse_env_flag(
        _BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV,
        default=False,
    )
    if batch_consecutive_requested and explicit_chunk_size is not None:
        raise ValueError(
            f"{_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV} cannot be "
            f"combined with {_BPREF_EXECUTION_ORDER_CHUNK_SIZE_ENV}"
        )
    if batch_consecutive_requested and not preserve_bpref_particle_order:
        raise ValueError(
            f"{_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV} requires "
            "the guarded fresh K=1 physical particle order"
        )
    if batch_consecutive_requested and processing_order_group_by_bucket_size:
        raise ValueError(
            f"{_BPREF_EXECUTION_BATCH_CONSECUTIVE_EQUAL_SUPPORT_ENV} cannot be "
            f"combined with {_BPREF_EXECUTION_GROUP_BY_BUCKET_SIZE_ENV}"
        )
    batch_consecutive_bucket_sizes = bool(
        batch_consecutive_requested
        and preserve_bpref_particle_order
        and not processing_order_group_by_bucket_size
    )
    processing_order_chunk_size = explicit_chunk_size or (
        _DEFAULT_FRESH_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE
        if preserve_bpref_particle_order
        and not processing_order_group_by_bucket_size
        and not batch_consecutive_bucket_sizes
        else 1
    )
    return processing_order_chunk_size, batch_consecutive_bucket_sizes


def _native_dual_weighted_sums_enabled_for_pass(
    *,
    use_exact_relion_gaussian: bool,
    use_relion_x_half_mstep: bool,
    accumulate_noise: bool,
) -> bool:
    """Select the qualified native reduction only on its exact GPU contract."""

    if os.environ.get(_SPARSE_KCLASS_NATIVE_DUAL_WEIGHTED_SUMS_ENV) is not None:
        return parse_env_flag(
            _SPARSE_KCLASS_NATIVE_DUAL_WEIGHTED_SUMS_ENV,
            default=False,
        )
    if not (use_exact_relion_gaussian and use_relion_x_half_mstep and accumulate_noise):
        return False
    from recovar.cuda_backproject import custom_cuda_requested

    return bool(jax.default_backend() == "gpu" and custom_cuda_requested())


def _fused_mstep_noise_enabled_for_pass(
    *,
    native_dual_weighted_sums: bool,
    use_exact_relion_gaussian: bool,
    use_relion_x_half_mstep: bool,
    accumulate_noise: bool,
    compact_noise_sums_match_mstep: bool,
) -> bool:
    """Select the fused reduction only on its qualified exact-GPU contract."""

    if not (
        native_dual_weighted_sums
        and use_exact_relion_gaussian
        and use_relion_x_half_mstep
        and accumulate_noise
        and not compact_noise_sums_match_mstep
        and parse_env_flag(_SPARSE_KCLASS_RESIDUAL_TERMS_FUSED_ENV, default=True)
    ):
        return False
    return parse_env_flag(_SPARSE_KCLASS_FUSED_MSTEP_NOISE_ENV, default=True)


def _fresh_k1_direct_noise_default(
    *,
    preserve_bpref_particle_order: bool,
    relion_exact_bpref_operands: bool,
) -> bool:
    """Enable the accepted noise path only inside the fresh K=1 guard."""

    return bool(preserve_bpref_particle_order and relion_exact_bpref_operands)


def _relion_powerclass_spectrum_norm_enabled(
    *,
    fresh_k1_guard: bool,
) -> bool:
    """Use RELION's shell spectrum by default only in the fresh K=1 guard."""

    return parse_env_flag(
        _RELION_POWERCLASS_SPECTRUM_NORM_ENV,
        default=bool(fresh_k1_guard),
    )


def _relion_exact_bpref_operands_enabled(
    *,
    fresh_k1_guard: bool,
    source_faithful_spectrum_norm: bool,
) -> bool:
    """Pair exact BPref with the qualified fresh-K=1 spectrum path."""

    return parse_env_flag(
        _RELION_EXACT_BPREF_OPERANDS_ENV,
        default=bool(fresh_k1_guard and source_faithful_spectrum_norm),
    )


def _relion_wavg_direct_modes(
    *,
    accumulate_noise: bool,
    scale_groups_available: bool,
    scale_aa_enabled: bool,
    direct_noise_only_default: bool = False,
) -> tuple[bool, bool]:
    """Resolve the stopped direct-Wavg noise/norm factorial arms.

    ``DIRECT_RESIDUAL`` preserves the existing coupled treatment: the native
    Wavg ``diff2`` stream supplies both shell noise and per-particle norm.
    ``DIRECT_NOISE_ONLY`` supplies only shell noise, leaving normalization on
    the production algebraic path.  The latter isolates the already-localized
    radial-noise boundary without silently changing a second state variable.
    """

    direct_residual_requested = bool(
        accumulate_noise
        and parse_env_flag(
            _RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV,
            default=False,
        )
    )
    direct_noise_only_requested = bool(
        accumulate_noise
        and parse_env_flag(
            _RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV,
            default=direct_noise_only_default,
        )
    )
    if direct_residual_requested and direct_noise_only_requested:
        raise ValueError(
            f"{_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV}=1 and "
            f"{_RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV}=1 are mutually exclusive"
        )
    direct_noise = direct_residual_requested or direct_noise_only_requested
    # Fresh iteration 1 intentionally has no scale-group accumulator.  The
    # established coupled diagnostic is dormant there and activates once the
    # scale state exists; preserve that lifecycle for the isolated arm.
    if direct_noise and not scale_groups_available:
        return False, False
    if direct_noise and not scale_aa_enabled:
        requested_name = (
            _RELION_WAVG_ATOMIC_DIRECT_RESIDUAL_ENV
            if direct_residual_requested
            else _RELION_WAVG_ATOMIC_DIRECT_NOISE_ONLY_ENV
        )
        raise ValueError(
            f"{requested_name}=1 requires "
            f"{_RELION_WAVG_ATOMIC_SCALE_AA_ENV}=1 and scale groups"
        )
    return direct_noise, direct_residual_requested


def _pass2_dump_enabled() -> bool:
    return bool(os.environ.get(pass2_diagnostics._PASS2_DUMP_DIR_ENV)) and not parse_env_flag(
        _NORM_RESIDUAL_DUMP_ONLY_ENV,
        default=False,
    )


def _pass2_conservative_dump_execution_enabled() -> bool:
    """Keep dump-only planner changes behind an explicit diagnostic opt-in."""

    return _pass2_dump_enabled() and parse_env_flag(
        _PASS2_DUMP_CONSERVATIVE_EXECUTION_ENV,
        default=False,
    )


def _projection_cache_enabled_for_pass(
    *,
    fine_rotations_override,
    dump_pass2_operands: bool,
) -> bool:
    """Resolve the diagnostic projection-cache override without changing defaults."""

    if fine_rotations_override is None:
        return False
    raw = os.environ.get(_SPARSE_PASS2_PROJECTION_CACHE_ENV)
    mode = "auto" if raw is None or raw.strip() == "" else raw.strip().lower()
    if mode == "auto":
        # Preserve the currently qualified paths while cache-on/cache-off is
        # adjudicated: production uses the cache and operand dumps do not.
        return not bool(dump_pass2_operands)
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{_SPARSE_PASS2_PROJECTION_CACHE_ENV} must be 'auto', 'on', or 'off', got {raw!r}",
    )


def _cached_score_rotation_chunk_size_for_pass(bucket_size: int) -> int:
    override = _optional_positive_int_env(_SPARSE_PASS2_CACHED_SCORE_ROT_CHUNK_ENV)
    chunk_size = _DEFAULT_CACHED_SCORE_ROT_CHUNK_SIZE if override is None else int(override)
    return max(1, min(int(bucket_size), int(chunk_size)))


def _compact_pair_mstep_mode_for_pass() -> str:
    """Return the compact-pair M-step reduction mode for this process."""

    raw = os.environ.get(_SPARSE_KCLASS_COMPACT_PAIR_MSTEP_ENV)
    if raw is None or raw.strip() == "":
        return "dense"
    mode = raw.strip().lower()
    if mode in {"dense", "default"}:
        return "dense"
    if mode == "pair_sparse":
        return mode
    raise ValueError(
        f"{_SPARSE_KCLASS_COMPACT_PAIR_MSTEP_ENV} must be 'dense' or 'pair_sparse', got {raw!r}",
    )


def _compact_pair_pair_sparse_mstep_enabled_for_pass(*, allow_pair_sparse: bool = True) -> bool:
    return bool(allow_pair_sparse) and _compact_pair_mstep_mode_for_pass() == "pair_sparse"


def _windowed_prepare_enabled_for_pass(use_window: bool) -> bool:
    """Return whether sparse pass-2 should materialize only active Fourier windows."""

    return bool(
        use_window
        and parse_env_flag(
            _SPARSE_PASS2_WINDOWED_PREPARE_ENV,
            default=True,
        )
    )


def _windowed_translation_tile_cap_enabled_for_pass() -> bool:
    """Return whether K-class sparse pass-2 should budget translation tiles on active windows."""

    return parse_env_flag(
        _SPARSE_KCLASS_WINDOWED_TRANSLATION_TILE_CAP_ENV,
        default=True,
    )


def _translation_tile_half_pixels_for_budget(
    *,
    use_window: bool,
    n_score_pixels: int,
    n_recon_pixels: int,
) -> int | None:
    """Return active half-pixel count for translation-tile budgeting."""

    if not _windowed_prepare_enabled_for_pass(bool(use_window)):
        return None
    if not _windowed_translation_tile_cap_enabled_for_pass():
        return None
    return max(int(n_score_pixels), int(n_recon_pixels))


def _windowed_translation_tile_max_multiplier_for_pass() -> int:
    explicit = _optional_positive_int_env(_SPARSE_PASS2_WINDOWED_TRANSLATION_TILE_MAX_MULTIPLIER_ENV)
    if explicit is not None:
        return int(explicit)
    return int(_DEFAULT_WINDOWED_TRANSLATION_TILE_MAX_MULTIPLIER)


def _max_images_for_sparse_pass2_translation_tile(
    image_shape,
    n_fine_trans,
    *,
    max_tile_bytes: int,
    complex_dtype,
    translation_tile_half_pixels: int | None,
) -> tuple[int, int, int | None, int | None]:
    full_cap = _max_images_for_translation_tile(
        image_shape,
        n_fine_trans,
        max_tile_bytes=max_tile_bytes,
        complex_dtype=complex_dtype,
    )
    if translation_tile_half_pixels is None:
        return full_cap, full_cap, None, None
    window_cap = _max_images_for_translation_tile(
        image_shape,
        n_fine_trans,
        max_tile_bytes=max_tile_bytes,
        complex_dtype=complex_dtype,
        n_half_pixels=translation_tile_half_pixels,
    )
    multiplier = _windowed_translation_tile_max_multiplier_for_pass()
    bounded_window_cap = max(full_cap, int(full_cap) * int(multiplier))
    return min(window_cap, bounded_window_cap), full_cap, window_cap, multiplier


def _compact_pair_execution_enabled_for_pass() -> bool:
    """Return whether fused K-class pass-2 should use compact-pair execution."""

    compact_pair_check = parse_env_flag(_COMPACT_KCLASS_PAIRS_CHECK_ENV, default=False)
    return parse_env_flag(
        _SPARSE_KCLASS_COMPACT_PAIRS_ENV,
        default=not compact_pair_check,
    )


def _compact_pair_min_bucket_size_for_pass(default_value: int | None = None) -> int:
    """Return the hybrid threshold for compact-pair execution buckets.

    An explicit environment setting wins over a caller-specific default so
    benchmark and diagnostic jobs retain their existing override behavior.
    """

    explicit = _optional_positive_int_env(_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE_ENV)
    if explicit is not None:
        return int(explicit)
    if default_value is not None:
        if int(default_value) <= 0:
            raise ValueError("compact-pair minimum bucket size must be positive")
        return int(default_value)
    return _DEFAULT_COMPACT_PAIR_MIN_BUCKET_SIZE


def _compact_pair_max_images_per_microbatch_for_pass(default_max_images_per_microbatch: int) -> int:
    """Return the compact-pair chunk cap, guarded by an explicit env override."""

    explicit = _optional_positive_int_env(_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_ENV)
    if explicit is not None:
        return explicit
    return max(1, int(default_max_images_per_microbatch))


def _compact_pair_prepare_max_images_per_microbatch(
    *,
    dense_max_images_per_microbatch: int,
    compact_pair_max_images_per_microbatch: int,
) -> int:
    """Return the compact-pair prepare cap used for execution bucket splitting."""

    return max(
        1,
        min(
            int(dense_max_images_per_microbatch),
            int(compact_pair_max_images_per_microbatch),
        ),
    )


def _active_row_pad_multiple_for_pass() -> int:
    """Return active-row gather padding multiple for stable JIT shapes."""

    explicit = _optional_positive_int_env(_SPARSE_KCLASS_ACTIVE_ROW_PAD_MULTIPLE_ENV)
    if explicit is not None:
        return int(explicit)
    return _DEFAULT_ACTIVE_ROW_PAD_MULTIPLE


def _small_bucket_coalesce_size_for_pass(n_images: int) -> int | None:
    explicit = _optional_positive_int_env(_SMALL_BUCKET_COALESCE_SIZE_ENV)
    if explicit is not None:
        return explicit
    max_images = parse_env_nonnegative_int(_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES_ENV)
    if max_images is None:
        max_images = _DEFAULT_AUTO_SMALL_BUCKET_COALESCE_MAX_IMAGES
    if int(n_images) > int(max_images):
        return None
    return _DEFAULT_SMALL_BUCKET_COALESCE_SIZE


def _tail_bucket_coalesce_params_for_pass(*, fused_k_class: bool) -> tuple[int | None, float | None, int | None]:
    """Return conservative tail-coalescing controls for sparse pass-2 buckets.

    Tail coalescing stays opt-in because the fused K-class 100k/256 probe
    showed that the old fused default could merge too many medium tail groups
    into 4096-row buckets and slow the sparse pass-2 path. Explicit env
    settings keep the diagnostic behavior available when a dataset has a true
    tiny high-rotation tail.
    """

    explicit_max_images = parse_env_nonnegative_int(_TAIL_BUCKET_COALESCE_MAX_IMAGES_ENV)
    if explicit_max_images is None:
        max_images = _DEFAULT_TAIL_BUCKET_COALESCE_MAX_IMAGES_FUSED_KCLASS if fused_k_class else 0
    else:
        max_images = explicit_max_images
    if max_images <= 1:
        return None, None, None

    max_inflation = _optional_positive_float_env(_TAIL_BUCKET_COALESCE_MAX_INFLATION_ENV)
    if max_inflation is None:
        max_inflation = _DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION

    min_bucket_size = _optional_positive_int_env(_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE_ENV)
    if min_bucket_size is None:
        min_bucket_size = _DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE
    return int(max_images), float(max_inflation), int(min_bucket_size)


def _compact_pair_tail_bucket_coalesce_params_for_pass(
    *,
    default_max_images: int | None = None,
    default_max_inflation: float | None = None,
    default_min_bucket_size: int | None = None,
) -> tuple[int | None, float | None, int | None]:
    """Return bounded tail-coalescing controls for compact-pair K-class buckets."""

    max_images = parse_env_nonnegative_int(_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_IMAGES_ENV)
    if max_images is None:
        max_images = parse_env_nonnegative_int(_TAIL_BUCKET_COALESCE_MAX_IMAGES_ENV)
    if max_images is None:
        max_images = (
            _DEFAULT_COMPACT_PAIR_TAIL_BUCKET_COALESCE_MAX_IMAGES
            if default_max_images is None
            else int(default_max_images)
        )
    if int(max_images) <= 1:
        return None, None, None

    max_inflation = _optional_positive_float_env(_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MAX_INFLATION_ENV)
    if max_inflation is None:
        max_inflation = _optional_positive_float_env(_TAIL_BUCKET_COALESCE_MAX_INFLATION_ENV)
    if max_inflation is None:
        max_inflation = (
            _DEFAULT_TAIL_BUCKET_COALESCE_MAX_INFLATION
            if default_max_inflation is None
            else float(default_max_inflation)
        )
    if float(max_inflation) <= 0.0:
        raise ValueError("compact-pair tail coalescing inflation must be positive")

    min_bucket_size = _optional_positive_int_env(_SPARSE_KCLASS_COMPACT_PAIR_TAIL_COALESCE_MIN_BUCKET_SIZE_ENV)
    if min_bucket_size is None:
        min_bucket_size = _optional_positive_int_env(_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE_ENV)
    if min_bucket_size is None:
        min_bucket_size = (
            _DEFAULT_TAIL_BUCKET_COALESCE_MIN_BUCKET_SIZE
            if default_min_bucket_size is None
            else int(default_min_bucket_size)
        )
    if int(min_bucket_size) <= 0:
        raise ValueError("compact-pair tail coalescing minimum bucket size must be positive")
    return int(max_images), float(max_inflation), int(min_bucket_size)
