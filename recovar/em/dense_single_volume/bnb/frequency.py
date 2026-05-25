"""Frequency-band schedule and low/high score-support split for cryoSPARC BnB.

The cryoSPARC bound (Suppl Eq 12, 22) splits the image alignment error into
an exact low-frequency part A_L(r,t) over Fourier coefficients with |k| <= L
and a bounded high-frequency part B_L(r,t) over L < |k| <= L_max.

In RECOVAR:
- The exact low part is computed via the existing scoring kernel
  (``helpers.scoring._score_rotation_block``) under a ``FourierWindowSpec``
  whose ``score_indices`` cover |k| <= L.
- L_max is bounded by the *current refinement resolution* (``current_size //
  2``), NOT the full Nyquist support.
- The high band is the set difference between the final score support and
  the low score support — see ``make_bnb_high_indices_np``.

This module is pure NumPy/JAX glue around
``helpers.fourier_window.make_fourier_window_spec`` and does not touch the
score kernels themselves.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.fourier_window import (
    FourierWindowSpec,
    make_fourier_window_spec,
)

from .options import BranchBoundOptions


def make_bnb_frequency_schedule(
    final_current_size: int | None,
    image_shape: tuple[int, int],
    options: BranchBoundOptions,
) -> list[int]:
    """Return monotonically increasing list of L radii [L_0, ..., L_max].

    L_0 = options.initial_fourier_radius (default 12).
    L_{j+1} = max(L_j + 1, round(L_j * options.fourier_radius_growth)).
    Capped at L_max = final_current_size // 2 (or image_shape[0] // 2 if None).
    Always ends at L_max. Maximum length = options.n_subdivisions + 1.

    Examples
    --------
    >>> from recovar.em.dense_single_volume.bnb.options import BranchBoundOptions
    >>> opts = BranchBoundOptions(n_subdivisions=7, initial_fourier_radius=12)
    >>> make_bnb_frequency_schedule(final_current_size=128, image_shape=(128,128),
    ...                              options=opts)
    [12, 24, 48, 64]
    """
    if final_current_size is None:
        L_max = int(image_shape[0]) // 2
    else:
        L_max = int(final_current_size) // 2
    if L_max < 1:
        raise ValueError(f"L_max must be >= 1, got {L_max}")

    L0 = max(1, int(options.initial_fourier_radius))
    growth = float(options.fourier_radius_growth)
    if growth <= 1.0:
        raise ValueError(f"fourier_radius_growth must be > 1, got {growth}")
    n_max_stages = int(options.n_subdivisions) + 1
    if n_max_stages < 1:
        raise ValueError(f"n_subdivisions must be >= 0, got {options.n_subdivisions}")

    schedule: list[int] = []
    L = L0
    for _ in range(n_max_stages):
        L_clamped = min(L, L_max)
        if not schedule or L_clamped > schedule[-1]:
            schedule.append(int(L_clamped))
        if L_clamped >= L_max:
            break
        next_L = max(L_clamped + 1, int(round(L * growth)))
        L = next_L

    if schedule[-1] != L_max:
        schedule.append(int(L_max))

    return schedule


def make_bnb_low_window_spec(
    image_shape: tuple[int, int],
    L: int,
    n_half: int,
) -> FourierWindowSpec:
    """Build a low-frequency FourierWindowSpec covering |k| <= L.

    cryoSPARC's exact low-frequency term A_L sums over |l| <= L. RECOVAR's
    ``make_fourier_window_spec`` uses radius = current_size // 2, so we pass
    current_size = 2L. We do not include DC in the score window (matching the
    existing score path) and skip the reconstruction window since BnB only
    uses this for support selection, not M-step.
    """
    L = int(L)
    if L < 0:
        raise ValueError(f"L must be >= 0, got {L}")
    return make_fourier_window_spec(
        image_shape,
        current_size=2 * L,
        n_half=int(n_half),
        score_include_dc=False,
        include_recon_window=False,
    )


def make_bnb_high_indices_np(
    final_score_indices_np: np.ndarray,
    low_score_indices_np: np.ndarray,
) -> np.ndarray:
    """Return packed-half indices in (final \\ low), sorted ascending.

    The cryoSPARC high-frequency band is bounded above by the current
    refinement resolution (NOT the full Nyquist support), so we subtract from
    the final score support, not from the entire half-spectrum. Both inputs
    must be packed-half index arrays from
    ``helpers.fourier_window.make_fourier_window_indices_np``.
    """
    final_set = np.asarray(final_score_indices_np, dtype=np.int64)
    low_set = np.asarray(low_score_indices_np, dtype=np.int64)
    diff = np.setdiff1d(final_set, low_set, assume_unique=False)
    return diff.astype(np.int32, copy=False)


def fourier_window_spec_from_indices(
    score_indices_np: np.ndarray,
    *,
    dtype=jnp.int32,
) -> FourierWindowSpec:
    """Construct a FourierWindowSpec from an arbitrary set of half-spectrum indices.

    Used by Phase-1 tests that need a "high-only" score window for verifying
    the algebraic identity score(full) == score(low) + score(high). The
    reconstruction window is intentionally None (BnB never uses it).
    """
    score_indices_np = np.sort(np.asarray(score_indices_np, dtype=np.int32))
    n_score = int(score_indices_np.size)
    projection_indices_np = score_indices_np
    score_projection_take_np = np.arange(n_score, dtype=np.int32)
    return FourierWindowSpec(
        use_window=True,
        score_indices_np=score_indices_np,
        recon_indices_np=None,
        projection_indices_np=projection_indices_np,
        score_projection_take_np=score_projection_take_np,
        recon_projection_take_np=None,
        score_indices=jnp.asarray(score_indices_np, dtype=dtype),
        recon_indices=None,
        projection_indices=jnp.asarray(projection_indices_np, dtype=dtype),
        score_projection_take=jnp.asarray(score_projection_take_np, dtype=dtype),
        recon_projection_take=None,
        n_score=n_score,
        n_recon=n_score,
        n_projection=n_score,
        max_r=None,
        projection_max_r=None,
    )
