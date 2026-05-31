"""Centralized helpers for spawning recovar subprocesses.

Used by scripts that shell out to ``recovar`` commands
(``run_test_dataset``, ``quickstart``, GUI executors) so the spawned
child inherits a consistent environment.

History — ``XLA_PYTHON_CLIENT_PREALLOCATE``:
  Earlier versions of this helper forced
  ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` in every child to dodge a
  shared-GPU OOM at child startup. That default was reverted
  2026-05-23: forcing PREALLOCATE=false makes JAX allocate
  on-demand, which is more vulnerable to allocator fragmentation
  failures during later large single allocations (e.g. the
  ~37 GB noise-estimator buffer at grid=256 / batch=320 — slurm
  8333303). It also masks JAX's normal contiguous-pool behavior.

  Now: the helper passes the parent's env through unchanged. Users
  on shared GPUs who need PREALLOCATE=false can export it in their
  shell; recovar no longer overrides their choice.
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
