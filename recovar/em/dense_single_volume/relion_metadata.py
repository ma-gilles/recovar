"""RELION metadata helpers used by the dense single-volume iteration loop.

These helpers manipulate per-image RELION metadata (translations, rotation
grids, radial shell counts) and depend on the iteration_loop's namespace
(via ``_il``) for symbols that test fixtures monkeypatch at
``recovar.em.dense_single_volume.iteration_loop``.
"""

from __future__ import annotations

import numpy as np

from recovar.core import fourier_transform_utils
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    relion_translation_search_base,
)


def _relion_metadata_translations(previous_best_translations, selected_relative_translations):
    """Return RELION-style metadata offsets after selecting relative shifts.

    RELION applies the rounded previous offset to the image before scoring,
    evaluates the search grid as a relative sampled translation, then writes
    ``rounded_old_offset + sampled_translation`` back to metadata. Keeping
    that absolute value is required for the next iteration's pre-shift and
    sigma-offset sufficient statistic.
    """
    selected = np.asarray(selected_relative_translations, dtype=np.float32)
    base = relion_translation_search_base(previous_best_translations)
    if base is None:
        return selected
    return (np.asarray(base, dtype=np.float32).reshape(selected.shape) + selected).astype(np.float32)


def _relion_half_plane_shell_counts(image_shape):
    """Count RELION's non-redundant FFTW half-plane shell pixels."""
    height, width = int(image_shape[0]), int(image_shape[1])
    n_shells = height // 2 + 1
    counts = np.zeros(n_shells, dtype=np.float64)
    for iy in range(height):
        ky = iy if iy <= height // 2 else iy - height
        for ix in range(width // 2 + 1):
            # RELION excludes redundant jp==0, ip<0 FFTW half-plane entries.
            if ix == 0 and ky < 0:
                continue
            shell = int(np.rint(np.sqrt(float(ky * ky + ix * ix))))
            if shell < n_shells:
                counts[shell] += 1.0
    return counts


def _relion_rotation_grid_float32(healpix_order: int, *, dtype: np.dtype = np.float32):
    """Return scorer matrices/eulers using RELION's accelerated-path policy.

    ``dtype`` controls only the returned rotation *matrices*. Under
    ``ACC_DOUBLE_PRECISION`` RELION's host-side ``RFLOAT -> XFLOAT`` cast is a
    no-op, so a caller running float64 scoring should pass ``dtype=np.float64``
    here to keep the coarse scorer matrices at full precision instead of the
    single-precision default. The returned ``eulers`` stay float32 regardless
    -- they are RELION's public per-particle metadata angles, not scoring
    operands, and RELION always serializes them as float.
    """
    # Indirection through iteration_loop module so test monkeypatches on
    # ``iteration_loop.get_relion_rotation_grid`` / ``get_relion_rotation_grid_eulers``
    # win at the call site.
    from recovar.em.dense_single_volume import iteration_loop as _il

    order = int(healpix_order)
    source_eulers = _il._get_relion_rotation_grid_eulers_float64(order)
    eulers = source_eulers.astype(np.float32)
    # RELION's accelerated expectation path constructs inverse projector
    # matrices on the host in RFLOAT precision, casts to XFLOAT, then copies
    # them to the device.  Preserve source Euler precision until that cast.
    from recovar.em.sampling import _relion_mstep_rotations_from_eulers

    rotations = _relion_mstep_rotations_from_eulers(source_eulers, dtype=dtype)
    return rotations, eulers


def _rotation_eulers_for_canonical_or_custom_grid(rotations: np.ndarray, healpix_order: int) -> np.ndarray:
    """Avoid expensive matrix->Euler conversion for canonical RELION grids."""
    from recovar.em.dense_single_volume import iteration_loop as _il

    rotations = np.asarray(rotations, dtype=np.float32)
    order = int(healpix_order)
    if rotations.shape[0] == _il.rotation_grid_size(order):
        canonical_rotations, canonical_eulers = _relion_rotation_grid_float32(order)
        if np.allclose(rotations, canonical_rotations, rtol=0.0, atol=1e-6):
            return canonical_eulers
    return _il.utils.R_to_relion(np.asarray(rotations), degrees=True).astype(np.float32)


def _radial_profile_from_noise_variance(noise_variance, image_shape):
    """Average an image-shaped noise vector into integer radial shells."""
    n_shells = image_shape[0] // 2 + 1
    radial_dist = np.clip(
        fourier_transform_utils.get_grid_of_radial_distances(
            image_shape,
            scaled=False,
            frequency_shift=0,
        )
        .astype(int)
        .reshape(-1),
        0,
        n_shells - 1,
    )
    noise_np = np.asarray(noise_variance, dtype=np.float64).reshape(-1)
    radial = np.zeros(n_shells, dtype=np.float64)
    counts = np.zeros(n_shells, dtype=np.float64)
    np.add.at(radial, radial_dist[: noise_np.size], noise_np)
    np.add.at(counts, radial_dist[: noise_np.size], 1.0)
    return radial / np.maximum(counts, 1.0)
