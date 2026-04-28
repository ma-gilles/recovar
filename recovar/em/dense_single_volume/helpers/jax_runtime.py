"""Small JAX runtime helpers shared by dense/local EM code."""

from __future__ import annotations

import jax


def block_until_ready(*values):
    """Synchronize one or more JAX values before host-side timing reads."""
    for value in values:
        if value is not None:
            jax.block_until_ready(value)
