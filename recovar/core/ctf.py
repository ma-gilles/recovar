"""CTF evaluation and parameter handling for cryo-EM images."""

import functools
from enum import IntEnum

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils


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
        w: Amplitude contrast fraction (0–1).
        phase_shift: Phase shift in degrees.
        bfactor: B-factor for envelope decay in Angstroms squared.

    Returns:
        CTF values with the same shape as ``freqs[..., 0]``.
    """
    assert freqs.shape[-1] == 2
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


def critical_exposure(freq, voltage):
    scale_factor = jnp.where(jnp.isclose(voltage, 200), 0.75, 1)
    critical_exp = freq ** (-1.665)
    critical_exp = critical_exp * scale_factor * 0.245
    return critical_exp + 2.81


def get_dose_filters_from_tilt_number(Apix, image_shape, dose_per_tilt, angle_per_tilt, tilt_numbers, voltage):
    cumulative_dose = tilt_numbers * dose_per_tilt
    tilt_angles = angle_per_tilt * jnp.ceil(tilt_numbers / 2)
    return get_dose_filters(Apix, image_shape, cumulative_dose, tilt_angles, voltage)


def get_dose_filters(Apix, image_shape, cumulative_dose, tilt_angles, voltage):
    freqs = fourier_transform_utils.get_k_coordinate_of_each_pixel(image_shape, Apix, scaled=True)

    s2 = freqs[..., 0] ** 2 + freqs[..., 1] ** 2
    s = jnp.sqrt(s2)

    cd = cumulative_dose[:, None]                        # (n, 1)
    ce = critical_exposure(s, voltage)[None, :]          # (1, n_pixels)
    oe_mask = cd < ce * 2.51284                          # implicit broadcast → (n, n_pixels)
    freq_correction = jnp.exp(-0.5 * cd / ce) * oe_mask

    angle_correction = jnp.cos(tilt_angles * jnp.pi / 180)
    return freq_correction * angle_correction[:, None]


def cryodrgn_CTF(CTF_params, image_shape, voxel_size):
    psi = fourier_transform_utils.get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled=True)
    return batch_evaluate_ctf(psi, CTF_params)


@functools.partial(jax.jit, static_argnums=[1])
def cryodrgn_CTF_half(CTF_params, image_shape, voxel_size):
    """Half-spectrum CTF (packed real-FFT layout), equivalent to full CTF mapped to half."""
    full = cryodrgn_CTF(CTF_params, image_shape, voxel_size)
    return fourier_transform_utils.full_image_to_half_image(full, image_shape)


@functools.partial(jax.jit, static_argnums=[1])
def evaluate_ctf_wrapper_tilt_series_v2(CTF_params, image_shape, voxel_size):
    dose_filter = get_dose_filters(
        voxel_size,
        image_shape,
        CTF_params[:, CTFParamIndex.DOSE],
        CTF_params[:, CTFParamIndex.TILT_ANGLE],
        CTF_params[0, CTFParamIndex.VOLT],
    )
    return dose_filter * cryodrgn_CTF(CTF_params[:, :9], image_shape, voxel_size)


@functools.partial(jax.jit, static_argnums=[1])
def evaluate_ctf_wrapper_tilt_series_v2_half(CTF_params, image_shape, voxel_size):
    """Half-spectrum tilt-series CTF wrapper, equivalent to mapped full output."""
    full = evaluate_ctf_wrapper_tilt_series_v2(CTF_params, image_shape, voxel_size)
    return fourier_transform_utils.full_image_to_half_image(full, image_shape)


@functools.partial(jax.jit, static_argnums=[1])
def evaluate_ctf_wrapper_tilt_series(CTF_params, image_shape, voxel_size, dose_per_tilt=None, angle_per_tilt=None):
    dose_filter = get_dose_filters_from_tilt_number(
        voxel_size,
        image_shape,
        dose_per_tilt,
        angle_per_tilt,
        CTF_params[:, CTFParamIndex.DOSE],
        CTF_params[0, CTFParamIndex.VOLT],
    )
    return dose_filter * cryodrgn_CTF(CTF_params[:, :9], image_shape, voxel_size)


@functools.partial(jax.jit, static_argnums=[1])
def evaluate_ctf_wrapper_tilt_series_half(CTF_params, image_shape, voxel_size, dose_per_tilt=None, angle_per_tilt=None):
    """Half-spectrum tilt-series CTF wrapper, equivalent to mapped full output."""
    full = evaluate_ctf_wrapper_tilt_series(
        CTF_params,
        image_shape,
        voxel_size,
        dose_per_tilt=dose_per_tilt,
        angle_per_tilt=angle_per_tilt,
    )
    return fourier_transform_utils.full_image_to_half_image(full, image_shape)


def get_cryo_ET_CTF_fun(dose_per_tilt=2.9, angle_per_tilt=3):
    def CTF_ET_fun(*args):
        return evaluate_ctf_wrapper_tilt_series(*args, dose_per_tilt=dose_per_tilt, angle_per_tilt=angle_per_tilt)

    return CTF_ET_fun


def get_cryo_ET_CTF_fun_half(dose_per_tilt=2.9, angle_per_tilt=3):
    def CTF_ET_fun(*args):
        return evaluate_ctf_wrapper_tilt_series_half(
            *args, dose_per_tilt=dose_per_tilt, angle_per_tilt=angle_per_tilt
        )

    return CTF_ET_fun


def evaluate_ctf_wrapper(CTF_params, image_shape, voxel_size, antialiasing=False):
    if not antialiasing:
        return cryodrgn_CTF(CTF_params, image_shape, voxel_size)

    upsample_factor = 2
    upsampled_shape = tuple(s * upsample_factor for s in image_shape)
    upsampled_CTF_squared = cryodrgn_CTF(CTF_params, upsampled_shape, voxel_size)

    batch_size = upsampled_CTF_squared.shape[0]
    ctf = upsampled_CTF_squared.reshape(batch_size, *upsampled_shape)

    kernel_size = upsample_factor + upsample_factor // 2
    kernel = jnp.ones((kernel_size, kernel_size), dtype=upsampled_CTF_squared.dtype) / (kernel_size * kernel_size)

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


def evaluate_ctf_wrapper_half(CTF_params, image_shape, voxel_size, antialiasing=False):
    """Half-spectrum CTF wrapper, equivalent to mapped full output."""
    full = evaluate_ctf_wrapper(CTF_params, image_shape, voxel_size, antialiasing=antialiasing)
    return fourier_transform_utils.full_image_to_half_image(full, image_shape)


class CTFParams:
    """Class to handle CTF parameters in a more structured way."""

    def __init__(self, params_array):
        self.params = np.asarray(params_array)
        self.n_images = self.params.shape[0]
        self.n_params = self.params.shape[1]

    @classmethod
    def create_standard_params(
        cls, n_images, dfu=0, dfv=0, dfang=0, volt=300, cs=2.7, w=0.1, phase_shift=0, bfactor=0, contrast=1.0
    ):
        params = np.zeros((n_images, len(CTFParamIndex)))
        params[:, CTFParamIndex.DFU] = dfu
        params[:, CTFParamIndex.DFV] = dfv
        params[:, CTFParamIndex.DFANG] = dfang
        params[:, CTFParamIndex.VOLT] = volt
        params[:, CTFParamIndex.CS] = cs
        params[:, CTFParamIndex.W] = w
        params[:, CTFParamIndex.PHASE_SHIFT] = phase_shift
        params[:, CTFParamIndex.BFACTOR] = bfactor
        params[:, CTFParamIndex.CONTRAST] = contrast
        return cls(params)

    def get_param(self, param_index):
        return self.params[:, param_index]

    def set_param(self, param_index, values):
        self.params[:, param_index] = values

    def get_image_params(self, image_idx):
        return self.params[image_idx, :]

    def add_tilt_series_params(self, dose_values, tilt_angles):
        if self.n_params < len(CTFParamIndex):
            extended_params = np.zeros((self.n_images, len(CTFParamIndex)))
            extended_params[:, : self.n_params] = self.params
            self.params = extended_params
            self.n_params = len(CTFParamIndex)
        self.params[:, CTFParamIndex.DOSE] = dose_values
        self.params[:, CTFParamIndex.TILT_ANGLE] = tilt_angles

    def validate(self):
        if self.n_images == 0:
            raise ValueError("No images in CTF parameters")
        if np.any(self.params[:, CTFParamIndex.VOLT] <= 0):
            raise ValueError("Voltage must be positive")
        if np.any(self.params[:, CTFParamIndex.CONTRAST] <= 0):
            raise ValueError("Contrast must be positive")
        return True

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

    @property
    def shape(self):
        return self.params.shape

    def astype(self, dtype):
        return CTFParams(self.params.astype(dtype))


__all__ = [
    "CTFParamIndex",
    "evaluate_ctf",
    "evaluate_ctf_packed",
    "batch_evaluate_ctf",
    "evaluate_ctf_wrapper_tilt_series_v2",
    "evaluate_ctf_wrapper_tilt_series",
    "get_cryo_ET_CTF_fun",
    "get_cryo_ET_CTF_fun_half",
    "critical_exposure",
    "get_dose_filters_from_tilt_number",
    "get_dose_filters",
    "evaluate_ctf_wrapper",
    "cryodrgn_CTF",
    "evaluate_ctf_wrapper_half",
    "evaluate_ctf_wrapper_tilt_series_half",
    "evaluate_ctf_wrapper_tilt_series_v2_half",
    "cryodrgn_CTF_half",
    "CTFParams",
]
