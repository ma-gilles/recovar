"""Centralized helpers for spawning recovar subprocesses.

Used by scripts that shell out to ``recovar`` commands
(``run_test_dataset``, ``quickstart``, GUI executors) so the spawned
child inherits a consistent environment.

Policy — ``XLA_PYTHON_CLIENT_PREALLOCATE``:
  ``false`` is useful on shared GPUs because JAX allocates memory on
  demand instead of reserving most VRAM up front. RECOVAR should not
  force that policy globally, though: long pipeline phases can need
  large contiguous allocations, and allocator behavior is part of the
  user's deployment environment.

  This helper therefore passes the parent's XLA environment through
  unchanged. Test harnesses or users who want on-demand allocation can
  set ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` before launching RECOVAR.
"""

from __future__ import annotations

import os
from typing import Mapping


def recovar_subprocess_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an env dict for spawning a recovar subprocess.

    Pass-through of the parent's env, with any ``extra`` overrides
    applied on top. Callers should pass the result as
    ``subprocess.run(..., env=...)``.
    """
    env = dict(os.environ)
    if extra:
        env.update(extra)
    return env
