"""Shared PPCA loading-matrix regularization helpers."""

from __future__ import annotations

import jax.numpy as jnp


def w_prior_precision(W_prior, *, reg_floor: float = 1e-16):
    """Return the PPCA precision implied by a variance-like ``W_prior``."""

    return 1.0 / (jnp.asarray(W_prior).real + reg_floor)


def w_prior_quadratic(W, W_prior, *, reg_floor: float = 1e-16):
    """Return ``sum |W|^2 / (W_prior + reg_floor)``.

    This is the exact loading-matrix L2 prior used by the PPCA M-step and
    negative-log-likelihood diagnostics. ``W_prior`` is variance-like: larger
    values weaken regularization.
    """

    W = jnp.asarray(W)
    precision = w_prior_precision(W_prior, reg_floor=reg_floor)
    return jnp.sum(precision * (jnp.abs(W) ** 2))
