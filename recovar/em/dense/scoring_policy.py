"""Execution defaults and existing diagnostic selectors for half-set scoring.

Static engine kwargs retain their import-time environment snapshot. Selector
functions read their environment at call time, including legacy override
precedence and invalid-value handling. The controller and scorers share the
same static kwargs object; this module performs no scoring or scheduling.
"""

import logging
import os

import jax
import numpy as np

from recovar.em.helpers.env_flags import parse_env_flag_or_false

logger = logging.getLogger(__name__)

# RELION parses ``--adaptive_fraction 0.999`` through ``textToFloat`` and
# stores that single-precision value in its optimiser state.  Python's literal
# 0.999 is a different binary64 boundary and can change a one-sample support
# cutoff when posterior weights are accumulated in float64.
RELION_ADAPTIVE_FRACTION = float(np.float32("0.999"))

_LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV = "RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT"
_LOCAL_ADAPTIVE_PASS2_DISABLE_FULL_PARENT_ENV = "RECOVAR_LOCAL_ADAPTIVE_PASS2_DISABLE_FULL_PARENT"
_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV = "RECOVAR_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY"
_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV = "RECOVAR_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT"
_K1_SKIP_SIGNIFICANCE_PRUNING_ENV = "RECOVAR_K1_SKIP_SIGNIFICANCE_PRUNING"
_K1_RELION_X_HALF_MSTEP_ENV = "RECOVAR_K1_RELION_X_HALF_MSTEP"
_K_CLASS_RELION_X_HALF_MSTEP_ENV = "RECOVAR_K_CLASS_RELION_X_HALF_MSTEP"
_K_CLASS_FULL_VOLUME_MSTEP_ENV = "RECOVAR_K_CLASS_FULL_VOLUME_MSTEP"
_K_CLASS_HALF_VOLUME_MSTEP_ENV = "RECOVAR_K_CLASS_HALF_VOLUME_MSTEP"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}

_FALSE_ENV_VALUES = {"0", "false", "no", "off"}

# RELION stores windowFourierTransform(in, out, current_size) as a rectangular
# FFTW half image, but the likelihood support is the nonzero Minvsigma2 mask:
# rounded radial shells, no DC, no redundant negative-row kx=0 entries.
RELION_FOURIER_WINDOW_SQUARE = False

# RELION uses pf=2 for both projection and reconstruction (--pad 2).
# Projection: real-space zero-pad N³→(2N)³, DFT, then trilinear slice.
# Reconstruction: backproject into (2N)³ Fourier grid, Wiener solve,
# iDFT at (2N)³, crop real-space to N³.
PADDING_FACTOR = 2

PROJECTION_PADDING_FACTOR = 2

# Dense ``run_em`` kwargs that are identical for every E-step in RELION mode.
# Per-iter and per-half values are layered on top at each call site via
# ``{**_DENSE_EM_STATIC_KWARGS, ...}``.

_DENSE_EM_STATIC_KWARGS: dict = {
    "score_with_masked_images": True,
    "half_spectrum_scoring": True,
    "projection_padding_factor": PROJECTION_PADDING_FACTOR,
    "reconstruction_padding_factor": PADDING_FACTOR,
    # Default float32. Set ``RECOVAR_USE_FLOAT64_SCORING=1`` /
    # ``RECOVAR_USE_FLOAT64_PROJECTIONS=1`` to upgrade to double precision.
    # Use fixed-state comparisons to evaluate precision changes; these flags
    # alone do not establish the cause of a trajectory mismatch.
    "use_float64_scoring": bool(
        os.environ.get("RECOVAR_USE_FLOAT64_SCORING", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    ),
    "use_float64_projections": bool(
        os.environ.get("RECOVAR_USE_FLOAT64_PROJECTIONS", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    ),
    # Default to RELION's float32 fine-search diff2/minimum ordering. This
    # diagnostic bypass retains the historical algebraic sparse scorer for
    # controlled full-trajectory A/B comparisons.
    "relion_exact_fine_gaussian": not bool(
        os.environ.get(
            "RECOVAR_DISABLE_RELION_EXACT_FINE_GAUSSIAN",
            "0",
        ).strip().lower()
        in {"1", "true", "yes", "on"}
    ),
    "do_gridding_correction": True,
    "square_window": RELION_FOURIER_WINDOW_SQUARE,
    "sparse_pass2": False,
}

# Off by default: reproduces RELION's GPU-accelerated projector/backprojector
# narrowing coordinates to float32 before flooring, unconditionally, even under
# ``ACC_DOUBLE_PRECISION`` (see ``recovar.core.relion_project`` module
# docstring). Set ``RECOVAR_RELION_ACC_DOUBLE_FLOORF_QUIRK=1`` to bit-match
# that GPU-double quirk in the local-search fine-pass projector fallback (it
# only has an effect when the texture path is unavailable, e.g. under
# ``use_float64_scoring``/``use_float64_projections``, since CUDA textures
# cannot hold complex128).
RELION_ACC_DOUBLE_FLOORF_QUIRK = bool(
    os.environ.get("RECOVAR_RELION_ACC_DOUBLE_FLOORF_QUIRK", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)


def _k_class_relion_half_volume_mstep_enabled() -> bool:
    """Return whether K-class should use the old native half-volume M-step."""

    half_value = os.environ.get(_K_CLASS_HALF_VOLUME_MSTEP_ENV)
    if half_value is not None and half_value.strip() != "":
        normalized = half_value.strip().lower()
        if normalized in _TRUE_ENV_VALUES:
            return True
        if normalized not in _FALSE_ENV_VALUES:
            logger.warning(
                "Ignoring invalid %s=%r; using K-class full-volume M-step default",
                _K_CLASS_HALF_VOLUME_MSTEP_ENV,
                half_value,
            )

    full_value = os.environ.get(_K_CLASS_FULL_VOLUME_MSTEP_ENV)
    if full_value is not None and full_value.strip() != "":
        normalized = full_value.strip().lower()
        if normalized in _FALSE_ENV_VALUES:
            return True
        if normalized in _TRUE_ENV_VALUES:
            return False
        logger.warning(
            "Ignoring invalid %s=%r; using K-class full-volume M-step default",
            _K_CLASS_FULL_VOLUME_MSTEP_ENV,
            full_value,
        )

    return False


def _k_class_relion_x_half_mstep_enabled() -> bool:
    """Return whether K-class should use RELION x-half BPref M-step accumulators."""

    value = os.environ.get(_K_CLASS_RELION_X_HALF_MSTEP_ENV)
    if value is not None and value.strip() != "":
        normalized = value.strip().lower()
        if normalized in _TRUE_ENV_VALUES:
            return True
        if normalized in _FALSE_ENV_VALUES:
            return False
        logger.warning(
            "Ignoring invalid %s=%r; using K-class RELION x-half M-step default",
            _K_CLASS_RELION_X_HALF_MSTEP_ENV,
            value,
        )

    # Preserve the legacy diagnostics as explicit overrides.  ``FULL=1`` means
    # reproduce the old full-volume path; ``FULL=0`` or ``HALF=1`` mean use the
    # native half-volume path.  With neither set, default to the RELION x-half
    # BPref layout used by the K=1 parity path.
    full_value = os.environ.get(_K_CLASS_FULL_VOLUME_MSTEP_ENV)
    if full_value is not None and full_value.strip() != "":
        normalized = full_value.strip().lower()
        if normalized in _TRUE_ENV_VALUES or normalized in _FALSE_ENV_VALUES:
            return False
        logger.warning(
            "Ignoring invalid %s=%r while resolving K-class RELION x-half M-step default",
            _K_CLASS_FULL_VOLUME_MSTEP_ENV,
            full_value,
        )

    half_value = os.environ.get(_K_CLASS_HALF_VOLUME_MSTEP_ENV)
    if half_value is not None and half_value.strip() != "":
        normalized = half_value.strip().lower()
        if normalized in _TRUE_ENV_VALUES:
            return False
        if normalized not in _FALSE_ENV_VALUES:
            logger.warning(
                "Ignoring invalid %s=%r while resolving K-class RELION x-half M-step default",
                _K_CLASS_HALF_VOLUME_MSTEP_ENV,
                half_value,
            )

    return True


def _jax_cpu_forced_from_env() -> bool:
    """Return whether JAX has been forced to CPU by environment."""

    platform_name = os.environ.get("JAX_PLATFORM_NAME", "").strip().lower()
    if platform_name == "cpu":
        return True
    platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if not platforms:
        return False
    requested = [token.strip() for token in platforms.split(",") if token.strip()]
    return bool(requested) and all(token == "cpu" for token in requested)


def _k1_relion_x_half_mstep_default_available() -> bool:
    """Return whether the default K=1 x-half M-step can use custom CUDA."""

    from recovar.utils.cuda_env import custom_cuda_disabled_from_env

    disabled, _ = custom_cuda_disabled_from_env()
    if disabled or _jax_cpu_forced_from_env():
        return False
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def _k1_relion_x_half_mstep_enabled() -> bool:
    """Default-on K=1 RELION x-half BPref accumulation."""

    value = os.environ.get(_K1_RELION_X_HALF_MSTEP_ENV)
    if value is None or value.strip() == "":
        return _k1_relion_x_half_mstep_default_available()
    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    logger.warning(
        "Ignoring invalid %s=%r; using K=1 RELION x-half M-step default",
        _K1_RELION_X_HALF_MSTEP_ENV,
        value,
    )
    return _k1_relion_x_half_mstep_default_available()


def _local_adaptive_pass2_full_parent_enabled() -> bool:
    """Return whether K=1 adaptive local pass-2 expands all parent samples."""

    if os.environ.get(_LOCAL_ADAPTIVE_PASS2_DISABLE_FULL_PARENT_ENV, "").strip().lower() in _TRUE_ENV_VALUES:
        return False
    value = os.environ.get(_LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV)
    if value is None or value.strip() == "":
        return False
    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    logger.warning(
        "Ignoring invalid %s=%r; using RELION pruned-parent local pass-2 default",
        _LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV,
        value,
    )
    return False


def _local_adaptive_pass2_rotation_only_enabled() -> bool:
    """Diagnostic: expand significant parent rotations to all parent translations."""

    return parse_env_flag_or_false(_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV, logger=logger)


def _local_adaptive_pass2_denominator_support_mode() -> str | None:
    """Diagnostic mode for broad-denominator local adaptive pass 2."""

    value = os.environ.get(_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV)
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower().replace("-", "_")
    if normalized in _FALSE_ENV_VALUES or normalized in {"none", "default", "pruned", "pruned_parent"}:
        return None
    if normalized in {"rotation", "rotations", "rotation_only", "significant_rotation_full_translation"}:
        return "rotation_only"
    if normalized in {"full", "full_parent", "all", "all_parent"}:
        return "full_parent"
    logger.warning(
        "Ignoring invalid %s=%r; expected rotation_only or full_parent",
        _LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV,
        value,
    )
    return None


def _k1_skip_significance_pruning_enabled() -> bool:
    """Diagnostic switch: evaluate the full K=1 adaptive fine grid."""

    return parse_env_flag_or_false(_K1_SKIP_SIGNIFICANCE_PRUNING_ENV, logger=logger)


def _dense_global_scoring_dtype() -> np.dtype:
    """Dtype for the dense/global (``use_local=False``) scoring path's
    float64-sensitive operands: the pass-1 rotation grid built by
    ``_relion_rotation_grid_float32``, and the offset/orientation log-prior
    arrays built by ``make_relion_translation_log_prior`` /
    ``make_relion_direction_log_prior`` and their prior-center helpers.

    None of these have a per-iteration diagnostic override (see
    ``_local_search_precision_flags`` for the local-search analog), so this
    collapses the global float64-scoring/-projections switches directly,
    matching RELION's ``ACC_DOUBLE_PRECISION`` build where the corresponding
    host ``RFLOAT`` values are never narrowed to float before the (no-op)
    ``XFLOAT`` cast.
    """

    if _DENSE_EM_STATIC_KWARGS["use_float64_scoring"] or _DENSE_EM_STATIC_KWARGS["use_float64_projections"]:
        return np.float64
    return np.float32
