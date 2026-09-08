"""RELION translation, rotation-grid and noise-shell metadata helpers."""

from __future__ import annotations

import numpy as np

from recovar.core import fourier_transform_utils
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    relion_translation_search_base,
)
from recovar.em.sampling import (
    _get_relion_rotation_grid_eulers_float64,
    _relion_mstep_rotations_from_eulers,
)


def _relion_metadata_translations(
    previous_best_translations, selected_relative_translations, *, dtype: np.dtype = np.float32
):
    """Return RELION-style metadata offsets after selecting relative shifts.

    RELION applies the rounded previous offset to the image before scoring,
    evaluates the search grid as a relative sampled translation, then writes
    ``rounded_old_offset + sampled_translation`` back to metadata. Keeping
    that absolute value is required for the next iteration's pre-shift and
    sigma-offset sufficient statistic.

    ``dtype`` defaults to float32 (RELION's accelerated-GPU precision);
    callers running a genuine double-precision comparison should pass
    ``np.float64``. RELION's own per-particle offset metadata
    (``exp_metadata``, ``EMDL_ORIENT_ORIGIN_X/Y_ANGSTROM``) is never narrowed
    to float -- ``exp_metadata`` is declared ``MultidimArray<RFLOAT>`` and
    ``EMDL_ORIENT_ORIGIN_X_ANGSTROM`` is registered ``EMDL_DOUBLE``, backed by
    ``std::vector<double>`` in ``MetaDataContainer`` (RELION
    ``src/ml_optimiser.h``, ``src/metadata_label.h``, ``src/metadata_container.h``).
    """
    selected = np.asarray(selected_relative_translations, dtype=dtype)
    base = relion_translation_search_base(previous_best_translations, dtype=dtype)
    if base is None:
        return selected
    return (np.asarray(base, dtype=dtype).reshape(selected.shape) + selected).astype(dtype)


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

    ``dtype`` controls the returned rotation matrices and working Euler grid. Under
    ``ACC_DOUBLE_PRECISION`` RELION's host-side ``RFLOAT -> XFLOAT`` cast is a
    no-op, so a caller running float64 scoring should pass ``dtype=np.float64``
    here to keep the coarse scorer operands at full precision instead of the
    single-precision default.  These Euler rows are subsequently perturbed and
    converted back to matrices, so they are working RFLOAT values rather than
    merely serialized metadata.
    """
    order = int(healpix_order)
    source_eulers = _get_relion_rotation_grid_eulers_float64(order)
    eulers = source_eulers.astype(dtype)
    # RELION's accelerated expectation path constructs inverse projector
    # matrices on the host in RFLOAT precision, casts to XFLOAT, then copies
    # them to the device.  Preserve source Euler precision until that cast.
    rotations = _relion_mstep_rotations_from_eulers(source_eulers, dtype=dtype)
    return rotations, eulers


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
