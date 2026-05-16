"""Single source of truth for the RECOVAR_DISABLE_CUDA env-var contract.

The codebase historically read RECOVAR_DISABLE_CUDA in several places, but
the variable name is easy to mistype as RECOVAR_CUDA_DISABLE. This helper
centralizes the read, normalizes the truthiness rules, and surfaces a
warning when only the misspelled variant is set so users learn the
canonical name without losing the run.

Backend detection is co-located here so the planner / error hints can
report the active GPU path without scattering the same os.environ
inspection across the codebase.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

CANONICAL_DISABLE_CUDA_ENV = "RECOVAR_DISABLE_CUDA"
TYPO_DISABLE_CUDA_ENV = "RECOVAR_CUDA_DISABLE"

_FALSY = {"", "0", "false", "False", "FALSE", "no", "No", "off", "Off"}

_warned_typo = False


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value not in _FALSY


def custom_cuda_disabled_from_env() -> tuple[bool, list[str]]:
    """Return ``(disabled, warnings)`` based on env vars.

    - If only the canonical variable is set: respect it.
    - If only the typo variable is set: respect it AND emit a one-time
      warning telling the user the canonical spelling.
    - If both are set: prefer the canonical and warn that the typo is
      being ignored.
    """
    global _warned_typo

    canonical = os.environ.get(CANONICAL_DISABLE_CUDA_ENV)
    typo = os.environ.get(TYPO_DISABLE_CUDA_ENV)

    warnings: list[str] = []

    if typo is not None and canonical is None:
        msg = (
            f"Environment variable {TYPO_DISABLE_CUDA_ENV} is set, but RECOVAR uses "
            f"{CANONICAL_DISABLE_CUDA_ENV}. Treating {TYPO_DISABLE_CUDA_ENV} as an alias "
            "for this run; please rename the variable in your shell init."
        )
        warnings.append(msg)
        if not _warned_typo:
            logger.warning(msg)
            _warned_typo = True
        return _truthy(typo), warnings

    if typo is not None and canonical is not None:
        msg = (
            f"Both {CANONICAL_DISABLE_CUDA_ENV} and {TYPO_DISABLE_CUDA_ENV} are set. "
            f"Using {CANONICAL_DISABLE_CUDA_ENV}; ignoring {TYPO_DISABLE_CUDA_ENV}."
        )
        warnings.append(msg)
        if not _warned_typo:
            logger.warning(msg)
            _warned_typo = True

    return _truthy(canonical), warnings


def custom_cuda_requested() -> bool:
    """Mirror of ``cuda_backproject.custom_cuda_requested`` using the helper."""
    disabled, _ = custom_cuda_disabled_from_env()
    return not disabled


Backend = Literal["custom_cuda", "jax_fallback", "cpu"]


def detect_backend() -> Backend:
    """Best-effort backend probe.

    - "cpu" if jax has no GPU device.
    - "jax_fallback" if RECOVAR_DISABLE_CUDA forces it.
    - "custom_cuda" otherwise (the planner/calibration treats this as
      the fast path; if the .so is missing at runtime the import error
      surfaces via the error-hints classifier).
    """
    try:
        import jax  # local import; helper must work without GPU jax loaded
    except Exception:
        return "cpu"

    try:
        gpus = jax.devices("gpu")
    except Exception:
        gpus = []

    if not gpus:
        return "cpu"

    disabled, _ = custom_cuda_disabled_from_env()
    return "jax_fallback" if disabled else "custom_cuda"


def log_backend(target_logger: logging.Logger | None = None) -> None:
    log = target_logger if target_logger is not None else logger
    backend = detect_backend()
    if backend == "custom_cuda":
        log.info("RECOVAR GPU backend: custom CUDA extension")
    elif backend == "jax_fallback":
        log.info(
            "RECOVAR GPU backend: JAX fallback path because %s=1 "
            "(approximately 2-3x slower than custom CUDA; expect tighter memory budgets)",
            CANONICAL_DISABLE_CUDA_ENV,
        )
    else:
        log.info("RECOVAR GPU backend: CPU (no GPU detected)")
