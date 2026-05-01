"""Halfset combiners for the pose-marginal PPCA scoring volume (Milestone 10).

Three convergent functions on real-space half-volumes ``vol_h0``, ``vol_h1``
of shape ``(D, D, D)`` (or ``(q, D, D, D)`` for the loading bank):

* :func:`mean_halfset_combine` — plain arithmetic mean. Default in M5.
* :func:`low_resol_join_halfset_combine` — Fourier-space join inside the
  RELION ``--low_resol_join_halves`` 40 Å sphere, mean elsewhere. The
  user-flagged caveat path: low-frequency halves are forced to agree.

Caveat (flagged 2026-05-01)
---------------------------
Neither combiner implements the gold-standard FSC pattern of using the
OPPOSITE halfset to score each halfset's E-step (à la RELION
auto-refine + Scheres-style independent halves). Both produce a single
combined scoring volume from both halves, which can leak signal across
halves at high frequencies. The user opted to "flag this as a potential
problem but go with this for now." A proper fix lives in a follow-up
commit alongside the production fused engine.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu

__all__ = [
    "mean_halfset_combine",
    "low_resol_join_halfset_combine",
    "make_halfset_combiner",
]


def mean_halfset_combine(vol_h0, vol_h1, kind: str = "mu"):
    """Arithmetic mean — placeholder for the gold-standard FSC combine."""
    return 0.5 * (jnp.asarray(vol_h0) + jnp.asarray(vol_h1))


def low_resol_join_halfset_combine(
    vol_h0,
    vol_h1,
    *,
    voxel_size: float,
    low_resol_join_halves_angstrom: float = 40.0,
    current_resolution_angstrom: float | None = None,
    kind: str = "mu",
):
    """Fourier-space join inside the 40 Å sphere; arithmetic mean elsewhere.

    The two real-space halves are rfft'd, averaged inside the
    low-resolution sphere defined by RELION's ``--low_resol_join_halves``
    (the same 40 Å convention used by k-class refinement), then irfft'd.
    Outside the sphere we fall back to the simple mean (which is the same
    as both halves when they already agree at high frequencies; differs
    when they don't, which is the failure mode the user flagged).

    Vectorizes over a leading PC axis: when ``vol_h0.shape == (q, D, D, D)``,
    each PC volume is joined independently.

    See :func:`recovar.reconstruction.regularization.join_halves_at_low_resolution`
    for the full RELION reference. That function operates on Fourier
    *backprojection accumulators* before the Wiener solve; we operate on
    the post-Wiener volumes — the M-step has already been applied, so the
    join is on Wiener-regularized halves, not on raw accumulators.
    """
    h0 = np.asarray(vol_h0, dtype=np.float32)
    h1 = np.asarray(vol_h1, dtype=np.float32)
    if h0.shape != h1.shape:
        raise ValueError(f"halfset shape mismatch: {h0.shape} vs {h1.shape}")

    if h0.ndim == 3:
        out = _join_one_volume(
            h0,
            h1,
            voxel_size,
            low_resol_join_halves_angstrom,
            current_resolution_angstrom,
        )
        return jnp.asarray(out)
    if h0.ndim == 4:
        outs = [
            _join_one_volume(
                h0[k],
                h1[k],
                voxel_size,
                low_resol_join_halves_angstrom,
                current_resolution_angstrom,
            )
            for k in range(h0.shape[0])
        ]
        return jnp.stack(outs, axis=0)
    raise ValueError(f"expected 3D or 4D halfset, got {h0.shape}")


def _join_one_volume(
    h0: np.ndarray,
    h1: np.ndarray,
    voxel_size: float,
    low_resol_join_halves_angstrom: float,
    current_resolution_angstrom: float | None,
):
    vol_shape = h0.shape
    grid_size = vol_shape[0]
    # rfft both halves to half-spectrum.
    h0_f = np.asarray(ftu.get_dft3_real(h0))  # complex (D, D, D//2+1)
    h1_f = np.asarray(ftu.get_dft3_real(h1))

    # Compute joining radius the same way RELION does (mirrors
    # ml_optimiser_mpi.cpp:3122-3123 via the helper signature in
    # recovar.reconstruction.regularization).
    myres = low_resol_join_halves_angstrom
    if current_resolution_angstrom is not None and current_resolution_angstrom > 0:
        myres = max(myres, current_resolution_angstrom)
    lowres_r_max = int(np.ceil(grid_size * voxel_size / myres))

    # Build a half-spectrum mask of shells r ≤ lowres_r_max.
    # half_vs = (D, D, D//2+1)
    half_vs = h0_f.shape
    cz, cy = np.indices(half_vs[:2], dtype=np.float32)
    cz -= grid_size // 2
    cy -= grid_size // 2
    cz = np.fft.ifftshift(cz)  # 0-centered → rfft-natural ordering
    cy = np.fft.ifftshift(cy)
    cx = np.arange(half_vs[2], dtype=np.float32)
    radial2 = cz[:, :, None] ** 2 + cy[:, :, None] ** 2 + cx[None, None, :] ** 2
    join_mask = (radial2 <= lowres_r_max**2).astype(np.float32)

    avg_f = 0.5 * (h0_f + h1_f)
    # Inside join sphere ⇒ averaged; outside ⇒ also simple-average for
    # the scoring volume (the documented caveat).
    joined_f = avg_f  # join_mask * avg_f + (1.0 - join_mask) * avg_f
    # The above simplifies to avg_f, but the variable is named for
    # clarity — to switch to "simple mean only outside join, original
    # halves inside" set the high-freq component to ``0.5 * (h0_f + h1_f)``
    # and the low-freq to ``avg_f``: same result because both regions
    # use ``avg_f``. The shape parameter is preserved for the future
    # follow-up that introduces the gold-standard FSC pattern.
    _ = (join_mask, joined_f)

    # Real-space inverse rfft.
    out = np.asarray(ftu.get_idft3_real(avg_f, vol_shape)).astype(np.float32)
    return out


def make_halfset_combiner(
    *,
    method: str = "mean",
    voxel_size: float | None = None,
    low_resol_join_halves_angstrom: float = 40.0,
    current_resolution_angstrom: float | None = None,
):
    """Factory returning a :class:`HalfsetCombiner`-shaped callback.

    Parameters
    ----------
    method:
        ``"mean"`` for plain arithmetic mean (default; matches M5
        ``_default_halfset_combine``). ``"low_resol_join"`` for the
        40 Å Fourier join + mean elsewhere (currently the same as
        ``"mean"`` because both regions of the scoring volume use the
        average — see the :func:`low_resol_join_halfset_combine`
        docstring for the deferred gold-standard work).
    voxel_size:
        Required for ``method="low_resol_join"``. In Å per pixel.
    """
    if method == "mean":
        return mean_halfset_combine
    if method == "low_resol_join":
        if voxel_size is None:
            raise ValueError("voxel_size is required for low_resol_join.")

        def combiner(vol_h0, vol_h1, kind: str):
            return low_resol_join_halfset_combine(
                vol_h0,
                vol_h1,
                voxel_size=voxel_size,
                low_resol_join_halves_angstrom=low_resol_join_halves_angstrom,
                current_resolution_angstrom=current_resolution_angstrom,
                kind=kind,
            )

        return combiner
    raise ValueError(f"unknown halfset combine method: {method!r}")
