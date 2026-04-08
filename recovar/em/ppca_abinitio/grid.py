"""Fixed pose / translation grid for the PPCA ab-initio v0 loop.

Per spec Q4, v0 supports HEALPix order 2 only. Order 3+ is Phase 4
and requires the streaming posterior API to scale. The constructor
enforces the order-2-only rule loudly so that nobody silently
crosses into the regime where the dense posterior tensors blow up.

Thin wrapper around `recovar.em.sampling`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import recovar.em.sampling as sampling

from .types import FixedGridSpec

_V0_MAX_HEALPIX_ORDER = 2


def build_fixed_grid(
    healpix_order: int,
    max_shift: int,
    shift_step: int = 1,
    *,
    enforce_v0_limits: bool = True,
) -> FixedGridSpec:
    """Build the FixedGridSpec for a v0 PPCA experiment.

    Parameters
    ----------
    healpix_order : int
        HEALPix order; v0 enforces `healpix_order <= 2` unless
        `enforce_v0_limits=False` is passed explicitly.
    max_shift : int
        Maximum integer translation in pixels along each axis.
    shift_step : int
        Translation grid step in pixels (default 1).
    enforce_v0_limits : bool
        If True (default), reject `healpix_order > _V0_MAX_HEALPIX_ORDER`
        with a `ValueError` referencing the spec. Override only when
        deliberately preparing a Phase 4 experiment.

    Returns
    -------
    FixedGridSpec
        Float64 rotations `(n_rot, 3, 3)` and translations
        `(n_trans, 2)`. Priors are flat (None).
    """
    if enforce_v0_limits and healpix_order > _V0_MAX_HEALPIX_ORDER:
        raise ValueError(
            f"healpix_order={healpix_order} exceeds v0 maximum of "
            f"{_V0_MAX_HEALPIX_ORDER} (spec Q4 / Section 5.1). Order 3+ "
            "requires the streaming posterior API and is deferred to "
            "Phase 4. Pass enforce_v0_limits=False if you are deliberately "
            "preparing a Phase 4 experiment."
        )
    if max_shift < 0:
        raise ValueError(f"max_shift must be >= 0, got {max_shift}")
    if shift_step < 1:
        raise ValueError(f"shift_step must be >= 1, got {shift_step}")

    rotations = sampling.get_rotation_grid_at_order(healpix_order, matrices=True)
    rotations = jnp.asarray(np.asarray(rotations, dtype=np.float64))

    translations = sampling.get_translation_grid(max_shift, shift_step)
    translations = jnp.asarray(np.asarray(translations, dtype=np.float64))

    return FixedGridSpec(rotations=rotations, translations=translations)
