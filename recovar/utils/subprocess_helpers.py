"""Centralized helpers for spawning recovar subprocesses with the right
JAX/XLA environment.

Used by scripts that shell out to ``recovar`` commands
(``run_test_dataset``, ``quickstart``, GUI executors) so the spawned
child inherits the env variables JAX needs to avoid OOM/init failure
on shared/workstation GPUs — even if the user hasn't exported them.

Issue #143: ``XLA_PYTHON_CLIENT_PREALLOCATE`` was the main culprit;
without it JAX tries to preallocate ~90% of physical VRAM at child
startup, which races with other GPU consumers and the recovar parent
process. Setting it to ``"false"`` in the child env makes start-up
allocate-on-demand and prevents the cascade.
"""

from __future__ import annotations

import os
from typing import Mapping


def recovar_subprocess_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an env dict suitable for spawning a recovar subprocess.

    Starts from the current ``os.environ``, then forces the XLA flags
    that recovar children need to behave well on shared GPUs. Any
    entries in ``extra`` override the defaults.

    The current child-side requirements:

      ``XLA_PYTHON_CLIENT_PREALLOCATE=false``
          Stops JAX from grabbing ~90% of physical VRAM at startup.
          Without this the child often OOMs before user code runs.

    Callers should pass the result as ``subprocess.run(..., env=...)``.
    Do NOT mutate ``os.environ`` directly — these flags belong on the
    spawned process, not the parent.
    """
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if extra:
        env.update(extra)
    return env
