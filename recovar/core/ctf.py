"""CTF evaluation and parameter handling for cryo-EM images.

Provides :class:`CTFEvaluator`, a unified equinox module dispatched by
:class:`CTFMode` that evaluates the Contrast Transfer Function for all
supported cryo-EM/ET imaging modes.
"""

import functools
import logging
from enum import IntEnum
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CTF parameter indexing
# ---------------------------------------------------------------------------

class CTFParamIndex(IntEnum):
    """Enum for CTF parameter indices to avoid magic numbers."""

    DFU = 0
    DFV = 1
    DFANG = 2
    VOLT = 3
    CS = 4
    W = 5
    PHASE_SHIFT = 6
    BFACTOR = 7
    CONTRAST = 8
    DOSE = 9
    TILT_ANGLE = 10


# ---------------------------------------------------------------------------
# CTF mode and evaluator
# ---------------------------------------------------------------------------

class CTFMode(IntEnum):
    """CTF evaluation mode, determined by the imaging modality.

    Attributes:
        SPA: Standard single-particle analysis CTF.
        SPA_ANTIALIASED: SPA with 2x-upsampled antialiasing filter.
        TILT_SERIES: Parametric dose weighting from ``dose_per_tilt``
            and ``angle_per_tilt`` constants.
        CRYO_ET: Per-image dose and tilt-angle columns already present
            in the CTF parameter array.
    """
    SPA = 0
    SPA_ANTIALIASED = 1
    TILT_SERIES = 2
    CRYO_ET = 3


class CTFEvaluator(eqx.Module):
    """Unified CTF evaluator for all cryo-EM/ET imaging modes.

    An equinox module that is callable with the standard CTF signature::

        evaluator(ctf_params, image_shape, voxel_size, *, half_image=False)

    All fields are static (compile-time constants).  Changing ``mode`` or
    dose parameters triggers JAX recompilation, which is the correct
    behaviour since these are fixed per dataset.

    Parameters
    ----------
    mode : CTFMode
        Imaging modality.  Defaults to :attr:`CTFMode.SPA`.
    dose_per_tilt : float
        Dose per tilt in e-/A^2.  Only used when ``mode == TILT_SERIES``.
    angle_per_tilt : float
        Tilt-angle increment in degrees.  Only used when ``mode == TILT_SERIES``.
    """

    mode: CTFMode = eqx.field(static=True, default=CTFMode.SPA)
    dose_per_tilt: float = eqx.field(static=True, default=0.0)
    angle_per_tilt: float = eqx.field(static=True, default=0.0)

    def __call__(self, ctf_params, image_shape, voxel_size, *, half_image=False):
        """Evaluate the CTF for a batch of images.

        Parameters
        ----------
        ctf_params : jax.Array
            Per-image CTF parameters, shape ``(N, K)``.
        image_shape : tuple[int, int]
            Image dimensions in pixels.
        voxel_size : float
            Pixel size in Angstroms.
        half_image : bool
            If *True*, evaluate on the rfft-packed half-spectrum grid.
        """
        if self.mode == CTFMode.SPA:
            return _compute_spa_ctf(ctf_params, image_shape, voxel_size, half_image=half_image)
        elif self.mode == CTFMode.SPA_ANTIALIASED:
            return _compute_spa_ctf_antialiased(ctf_params, image_shape, voxel_size, half_image=half_image)
        elif self.mode == CTFMode.TILT_SERIES:
            return _compute_tilt_series_ctf(
                ctf_params, image_shape, voxel_size,
                self.dose_per_tilt, self.angle_per_tilt, half_image=half_image,
            )
        elif self.mode == CTFMode.CRYO_ET:
            return _compute_cryo_et_ctf(ctf_params, image_shape, voxel_size, half_image=half_image)
        raise ValueError(f"Unknown CTF mode: {self.mode}")


class _LegacyCTFAdapter(eqx.Module):
    """Wraps an arbitrary callable as a CTFEvaluator-compatible module.

    The callable must accept ``(ctf_params, image_shape, voxel_size, *,
    half_image=False)`` or use ``**kwargs`` to absorb the keyword argument.
    """

    _fn: Callable = eqx.field(static=True)

    def __call__(self, ctf_params, image_shape, voxel_size, *, half_image=False, **kwargs):
        return self._fn(ctf_params, image_shape, voxel_size, half_image=half_image, **kwargs)


def as_ctf_evaluator(fn_or_evaluator):
    """Coerce a callable into a CTFEvaluator-compatible eqx.Module.

    If *fn_or_evaluator* is already a :class:`CTFEvaluator` or
    :class:`_LegacyCTFAdapter`, return it unchanged.  Otherwise wrap it
    in a :class:`_LegacyCTFAdapter`.
    """
    if isinstance(fn_or_evaluator, (CTFEvaluator, _LegacyCTFAdapter)):
        return fn_or_evaluator
    return _LegacyCTFAdapter(fn_or_evaluator)


# ---------------------------------------------------------------------------
# Low-level CTF physics (unchanged)
# ---------------------------------------------------------------------------

@jax.jit
def evaluate_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift, bfactor):
    """Evaluate the Contrast Transfer Function at given frequencies.

    Args:
        freqs: 2-D frequency coordinates, shape ``(..., 2)`` in 1/Angstrom.
        dfu: Defocus U in Angstroms.
        dfv: Defocus V in Angstroms.
        dfang: Astigmatism angle in degrees.
        volt: Accelerating voltage in kV.
        cs: Spherical aberration in mm.
        w: Amplitude contrast fraction (0-1).
        phase_shift: Phase shift in degrees.
        bfactor: B-factor for envelope decay in Angstroms squared.

    Returns:
        CTF values with the same shape as ``freqs[..., 0]``.
    """
    if freqs.shape[-1] != 2:
        raise ValueError(f"freqs last dimension must be 2, got {freqs.shape[-1]}")
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * jnp.pi / 180
    phase_shift = phase_shift * jnp.pi / 180
    lam = 12.2642598 / jnp.sqrt(volt * (1.0 + volt * 9.78475598e-7))

    x = freqs[..., 0]
    y = freqs[..., 1]
    ang = jnp.arctan2(y, x)
    s2 = x**2 + y**2
    df = 0.5 * (dfu + dfv + (dfu - dfv) * jnp.cos(2 * (ang - dfang)))
    gamma = 2 * jnp.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam**3 * s2**2) - phase_shift
    ctf = (1 - w**2) ** 0.5 * jnp.sin(gamma) - w * jnp.cos(gamma)
    ctf = ctf * jnp.exp(-bfactor / 4 * s2)
    return ctf


@jax.jit
def evaluate_ctf_packed(freqs, ctf):
    return evaluate_ctf(
        freqs, ctf[0], ctf[1], ctf[2], ctf[3], ctf[4], ctf[5], ctf[6], ctf[7]
    ) * ctf[8]


batch_evaluate_ctf = jax.vmap(evaluate_ctf_packed, in_axes=(None, 0))


# ---------------------------------------------------------------------------
# Dose-filter helpers
# ---------------------------------------------------------------------------

def critical_exposure(freq, voltage):
    scale_factor = jnp.where(jnp.isclose(voltage, 200), 0.75, 1)
    critical_exp = freq ** (-1.665)
    critical_exp = critical_exp * scale_factor * 0.245
    return critical_exp + 2.81


def _dose_filter_from_freqs(freqs, cumulative_dose, tilt_angles, voltage):
    """Shared dose-filter logic for arbitrary frequency coordinates."""
    s2 = freqs[..., 0] ** 2 + freqs[..., 1] ** 2
    s = jnp.sqrt(s2)

    cd = cumulative_dose[:, None]                        # (n, 1)
    ce = critical_exposure(s, voltage)[None, :]          # (1, n_pixels)
    oe_mask = cd < ce * 2.51284                          # implicit broadcast -> (n, n_pixels)
    freq_correction = jnp.exp(-0.5 * cd / ce) * oe_mask

    angle_correction = jnp.cos(tilt_angles * jnp.pi / 180)
    return freq_correction * angle_correction[:, None]


def get_dose_filters(Apix, image_shape, cumulative_dose, tilt_angles, voltage, *, half_image=False):
    if half_image:
        freqs = fourier_transform_utils.get_k_coordinate_of_each_pixel_half(image_shape, Apix, scaled=True)
    else:
        freqs = fourier_transform_utils.get_k_coordinate_of_each_pixel(image_shape, Apix, scaled=True)
    return _dose_filter_from_freqs(freqs, cumulative_dose, tilt_angles, voltage)


def get_dose_filters_from_tilt_number(Apix, image_shape, dose_per_tilt, angle_per_tilt, tilt_numbers, voltage, *, half_image=False):
    cumulative_dose = tilt_numbers * dose_per_tilt
    tilt_angles = angle_per_tilt * jnp.ceil(tilt_numbers / 2)
    return get_dose_filters(Apix, image_shape, cumulative_dose, tilt_angles, voltage, half_image=half_image)


# ---------------------------------------------------------------------------
# Private CTF computation functions
# ---------------------------------------------------------------------------

def _compute_spa_ctf(CTF_params, image_shape, voxel_size, *, half_image=False):
    """Standard single-particle CTF evaluation on a frequency grid."""
    if half_image:
        psi = fourier_transform_utils.get_k_coordinate_of_each_pixel_half(image_shape, voxel_size, scaled=True)
    else:
        psi = fourier_transform_utils.get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled=True)
    return batch_evaluate_ctf(psi, CTF_params)


def _compute_spa_ctf_antialiased(CTF_params, image_shape, voxel_size, *, half_image=False):
    """SPA CTF with 2x-upsampled antialiasing filter."""
    if half_image:
        full = _compute_spa_ctf_antialiased(CTF_params, image_shape, voxel_size)
        return fourier_transform_utils.full_image_to_half_image(full, image_shape)

    upsample_factor = 2
    upsampled_shape = tuple(s * upsample_factor for s in image_shape)
    upsampled_ctf = _compute_spa_ctf(CTF_params, upsampled_shape, voxel_size)

    batch_size = upsampled_ctf.shape[0]
    ctf = upsampled_ctf.reshape(batch_size, *upsampled_shape)

    kernel_size = upsample_factor + upsample_factor // 2
    kernel = jnp.ones((kernel_size, kernel_size), dtype=upsampled_ctf.dtype) / (kernel_size * kernel_size)

    ctf = jnp.expand_dims(ctf, 1)
    kernel = kernel.reshape(1, 1, kernel_size, kernel_size)
    ctf = jax.lax.conv_general_dilated(
        ctf,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "IOHW", "NCHW"),
    )
    ctf = jnp.squeeze(ctf, axis=1)
    ctf = ctf[:, ::upsample_factor, ::upsample_factor]
    return ctf.reshape(ctf.shape[0], -1)


@functools.partial(jax.jit, static_argnames=['image_shape', 'half_image'])
def _compute_cryo_et_ctf(CTF_params, image_shape, voxel_size, *, half_image=False):
    """CTF with per-image dose and tilt-angle columns in CTF params."""
    dose_filter = get_dose_filters(
        voxel_size,
        image_shape,
        CTF_params[:, CTFParamIndex.DOSE],
        CTF_params[:, CTFParamIndex.TILT_ANGLE],
        CTF_params[0, CTFParamIndex.VOLT],
        half_image=half_image,
    )
    return dose_filter * _compute_spa_ctf(CTF_params[:, :9], image_shape, voxel_size, half_image=half_image)


@functools.partial(jax.jit, static_argnames=['image_shape', 'half_image'])
def _compute_tilt_series_ctf(CTF_params, image_shape, voxel_size, dose_per_tilt=None, angle_per_tilt=None, *, half_image=False):
    """CTF with parametric dose weighting from tilt numbers."""
    dose_filter = get_dose_filters_from_tilt_number(
        voxel_size,
        image_shape,
        dose_per_tilt,
        angle_per_tilt,
        CTF_params[:, CTFParamIndex.DOSE],
        CTF_params[0, CTFParamIndex.VOLT],
        half_image=half_image,
    )
    return dose_filter * _compute_spa_ctf(CTF_params[:, :9], image_shape, voxel_size, half_image=half_image)


__all__ = [
    # New API
    "CTFMode",
    "CTFEvaluator",
    "as_ctf_evaluator",
    # Parameter indexing
    "CTFParamIndex",
    # Low-level physics
    "evaluate_ctf",
    "evaluate_ctf_packed",
    "batch_evaluate_ctf",
    # Dose filters
    "critical_exposure",
    "get_dose_filters",
    "get_dose_filters_from_tilt_number",
]
