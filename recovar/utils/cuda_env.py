"""Single source of truth for the RECOVAR_DISABLE_CUDA env-var contract.

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

_FALSY = {"", "0", "false", "False", "FALSE", "no", "No", "off", "Off"}


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value not in _FALSY


def custom_cuda_disabled_from_env() -> tuple[bool, list[str]]:
    """Return ``(disabled, warnings)`` for the canonical CUDA env var."""
    return _truthy(os.environ.get(CANONICAL_DISABLE_CUDA_ENV)), []


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
