"""Opt-in pure FP64 RELION VDAM M-step prototype; no production dispatch.

The device transaction uses full original-size Fourier capacity and a dynamic
logical radius. First-moment initialization certificates are explicit inputs:
RELION tests a sequential real-component sum, which a parallel reduction cannot
replace. The host wrapper obtains these certificates from existing host moments.
BPref accumulation and its CUDA atomic chronology are outside this module.

This is NOT a general replacement for the native transaction: the native
inverse FFT differs on some arbitrary non-Hermitian half-volume fixtures even
when their updated Fourier coefficients agree. Physical synthetic fixtures and
one captured N128 input pass the FP64 oracle, but neither establishes a general
input contract. Keep this prototype out of production pending that boundary
investigation and GPU oracle results.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.core import mask
from recovar.reconstruction.relion_functions import _relion_window_centered_half_fourier

from .relion_projector_setup import setup_relion_projector


def _complex(real, imag):
    return jax.lax.complex(real, imag)


def _norm(value):
    return jax.lax.optimization_barrier(value.real * value.real) + value.imag * value.imag


def _pad_moment(moment, capacity):
    before = capacity // 2 - moment.shape[0] // 2
    after = capacity - moment.shape[0] - before
    return jnp.pad(moment, ((before, after), (before, after), (0, capacity // 2 + 1 - moment.shape[2])))


def _unpad_moment(moment, shape):
    start = moment.shape[0] // 2 - shape[0] // 2
    return moment[start : start + shape[0], start : start + shape[1], : shape[2]]


def _first_moment(old, data, initialize, valid):
    # Separate product/add rounding like the native scalar EMA.
    real = jax.lax.optimization_barrier(0.9 * old.real) + (1.0 - 0.9) * data.real
    imag = jax.lax.optimization_barrier(0.9 * old.imag) + (1.0 - 0.9) * data.imag
    updated = jnp.where(initialize, data, _complex(real, imag))
    return jnp.where(valid, updated, old)


@partial(jax.jit, static_argnames=("ori_size", "padding_factor", "pseudo_halfsets", "return_intermediates"))
def relion_vdam_m_step_device(
    reference_relion,
    data_h0,
    weight_h0,
    data_h1,
    weight_h1,
    mom1_h0,
    mom1_h1,
    mom2,
    fsc_reconstruct,
    tau2,
    grad_stepsize,
    tau2_fudge,
    r_max,
    first_initializes_h0,
    first_initializes_h1,
    *,
    ori_size: int,
    padding_factor: int = 1,
    pseudo_halfsets: bool = True,
    return_intermediates: bool = False,
):
    """Return device arrays matching the native transaction's final outputs.

    Data/weights have shape (pf*N+3, pf*N+3, pf*N//2+2); moments retain their
    original (pf*N, pf*N, pf*N//2+1) layout. All numerical arrays are FP64 or
    complex128. Radius <=0 means full, matching the native M-step binding.
    The private invalid flags preserve native SSNR error conditions; the host
    wrapper raises after the one result transfer. No host operations occur here.
    """
    if not jax.config.x64_enabled:
        raise ValueError("RELION VDAM M-step requires JAX float64 support")
    if ori_size <= 0 or ori_size % 2 or padding_factor not in (1, 2):
        raise ValueError("device M-step supports positive even boxes and padding 1/2")
    fft_size = padding_factor * ori_size
    capacity = fft_size + 3
    half_shape = (capacity, capacity, capacity // 2 + 1)
    moment_shape = (fft_size, fft_size, fft_size // 2 + 1)
    if reference_relion.shape != (ori_size,) * 3:
        raise ValueError("reference must have shape (ori_size,)*3")
    pairs = [
        (data_h0, half_shape),
        (weight_h0, half_shape),
        (mom1_h0, moment_shape),
        (mom2, moment_shape),
        (fsc_reconstruct, (ori_size // 2 + 1,)),
        (tau2, (ori_size // 2 + 1,)),
    ]
    if pseudo_halfsets:
        pairs += [(data_h1, half_shape), (weight_h1, half_shape), (mom1_h1, moment_shape)]
    if any(value is None or value.shape != shape for value, shape in pairs):
        raise ValueError("M-step input has incompatible full-capacity shape")
    d0 = jnp.asarray(data_h0, jnp.complex128)
    w0 = jnp.asarray(weight_h0, jnp.float64)
    m10 = _pad_moment(jnp.asarray(mom1_h0, jnp.complex128), capacity)
    m2 = _pad_moment(jnp.asarray(mom2, jnp.complex128), capacity)
    tau2 = jnp.asarray(tau2, jnp.float64)
    fsc_reconstruct = jnp.asarray(fsc_reconstruct, jnp.float64)
    stepsize = jnp.asarray(grad_stepsize, jnp.float64)
    fudge = jnp.asarray(tau2_fudge, jnp.float64)
    radius = jnp.asarray(r_max, jnp.int32)
    radius = jnp.where(radius > 0, radius, ori_size // 2)
    coord = jnp.arange(capacity, dtype=jnp.int32) - capacity // 2
    x = jnp.arange(capacity // 2 + 1, dtype=jnp.int32)
    r2 = coord[:, None, None] ** 2 + coord[None, :, None] ** 2 + x[None, None, :] ** 2
    valid = r2 < (padding_factor * radius) ** 2
    n_shells = ori_size // 2 + 1
    shells = jnp.floor(jnp.sqrt(r2.astype(jnp.float64)) / padding_factor + 0.5).astype(jnp.int32)
    shell_indices = jnp.where(valid, shells, n_shells).reshape(-1)
    shell_lookup = jnp.minimum(shells, n_shells - 1)

    def shell_sum(values):
        # Invalid elements are dropped, rather than contending on a final bin.
        return (
            jnp.zeros(n_shells, jnp.float64)
            .at[shell_indices]
            .add(jnp.where(valid, values, 0.0).reshape(-1), mode="drop")
        )

    counts = shell_sum(jnp.ones(half_shape, jnp.float64))
    safe_counts = jnp.maximum(counts, 1.0)
    denominator = jnp.maximum(1.0, w0)
    d0 = jnp.where(valid, _complex(d0.real / denominator, d0.imag / denominator), d0)
    m10 = _first_moment(m10, d0, first_initializes_h0, valid)
    if pseudo_halfsets:
        d1 = jnp.asarray(data_h1, jnp.complex128)
        denominator = jnp.maximum(1.0, jnp.asarray(weight_h1, jnp.float64))
        d1 = jnp.where(valid, _complex(d1.real / denominator, d1.imag / denominator), d1)
        m11 = _pad_moment(jnp.asarray(mom1_h1, jnp.complex128), capacity)
        m11 = _first_moment(m11, d1, first_initializes_h1, valid)
        difference = d1 - d0
        average = _complex((d1.real + d0.real) / 2.0, (d1.imag + d0.imag) / 2.0)
        ratio = _norm(difference) / (_norm(average) + 1e-12)
        second_real = jax.lax.optimization_barrier(0.999 * m2.real) + (1.0 - 0.999) * ratio
        m2 = jnp.where(valid, _complex(second_real, jnp.zeros_like(second_real)), m2)
    else:
        # The native binding passes m10 twice: sameShape => do_half remains true.
        m11 = m10
    gradient = _complex((m10.real + m11.real) / 2.0, (m10.imag + m11.imag) / 2.0)
    denominator = jnp.sqrt(m2.real) + 1e-12
    # tComplex has only operator/=(tComplex): preserve its multiply/divide
    # sequence, including the zero imaginary denominator products.
    divisor = jax.lax.optimization_barrier(denominator * denominator)
    real_numerator = jax.lax.optimization_barrier(gradient.real * denominator) + gradient.imag * 0.0
    imag_numerator = jax.lax.optimization_barrier(gradient.imag * denominator) - gradient.real * 0.0
    gradient = _complex(real_numerator / divisor, imag_numerator / divisor)
    gradient = jnp.where(valid, gradient, d0)
    noise_power = shell_sum(jnp.sqrt(_norm(m10 - m11))) / safe_counts

    # updateSSNRarrays(false,false,false): half-0 weight and unchanged tau2.
    inverse_sigma = shell_sum(float(padding_factor**3) * w0)
    invalid_sigma = jnp.any((inverse_sigma != 0.0) & (inverse_sigma <= 1e-20))
    sigma2 = jnp.where(inverse_sigma > 1e-20, counts / inverse_sigma, 0.0)
    shell_tau = tau2[shell_lookup]
    inverse_tau = jnp.where(
        shell_tau > 0.0,
        1.0 / (float(padding_factor**3) * fudge * shell_tau),
        1.0 / (0.001 * w0),
    )
    evidence = w0 / inverse_tau
    data_vs_prior = shell_sum(evidence)
    data_vs_prior = jnp.where(
        jnp.arange(n_shells) > radius,
        0.0,
        jnp.where(counts < 0.001, 999.0, data_vs_prior / safe_counts),
    )
    coverage = shell_sum((evidence >= 1.0).astype(jnp.float64)) / safe_counts
    invalid_tau = jnp.any((tau2 < 0.0) & (counts > 0.0))

    # reconstructGrad: uncorrected projector, never the E-step's corrected FFT.
    projector, _unused_power = setup_relion_projector(
        reference_relion,
        radius,
        ori_size=ori_size,
        padding_factor=padding_factor,
        do_gridding=False,
    )
    previous_power = shell_sum(jnp.sqrt(_norm(projector))) / safe_counts
    snr = jnp.where(noise_power > 0.0, 2.0 * fudge * previous_power / noise_power, 0.0)
    fsc = jnp.where(counts > 0.0, jnp.minimum(jnp.maximum(snr / (1.0 + snr), fsc_reconstruct), 1.0), 0.0)
    fsc_voxel = fsc[shell_lookup]
    # use_fsc=false changes tau2_fudge to 1 before this update.
    update_real = jax.lax.optimization_barrier(fsc_voxel * gradient.real) - (1.0 - fsc_voxel) * projector.real
    update_imag = jax.lax.optimization_barrier(fsc_voxel * gradient.imag) - (1.0 - fsc_voxel) * projector.imag
    updated = _complex(
        projector.real + jax.lax.optimization_barrier(stepsize * update_real),
        projector.imag + jax.lax.optimization_barrier(stepsize * update_imag),
    )
    updated = jnp.where(valid, updated, projector)
    # Projector setup already supplies the inclusive decenter sphere. Its
    # +Nyquist-only native coordinates are retained by the shared FFTW window.
    fft_half = _relion_window_centered_half_fourier(updated, (capacity,) * 3, (fft_size,) * 3)
    real = ftu.get_idft3_real(fft_half, (fft_size,) * 3, norm="forward")
    start = (fft_size - ori_size) // 2
    real = real[start : start + ori_size, start : start + ori_size, start : start + ori_size]
    real = real / float(padding_factor**3 * ori_size)
    real, _mask = mask.soft_mask_outside_map(real)
    result = {
        "iref": real,
        "mom1_h0": _unpad_moment(m10, moment_shape),
        "mom1_h1": _unpad_moment(m11, moment_shape) if pseudo_halfsets else None,
        "mom2": _unpad_moment(m2, moment_shape),
        "tau2": tau2,
        "sigma2": sigma2,
        "data_vs_prior": data_vs_prior,
        "fourier_coverage": coverage,
        "mom1_noise_power": noise_power,
        "_invalid_sigma2": invalid_sigma,
        "_invalid_tau2": invalid_tau,
    }
    if return_intermediates:
        result.update(gradient=gradient, projector=projector, updated_projector=updated, fsc_estimate=fsc)
    return result


def relion_vdam_m_step_host(
    reference_relion,
    data_h0,
    weight_h0,
    data_h1,
    weight_h1,
    mom1_h0,
    mom1_h1,
    mom2,
    fsc_ssnr,
    fsc_reconstruct,
    tau2,
    grad_stepsize,
    tau2_fudge,
    ori_size,
    padding_factor=1,
    interpolator=1,
    r_max=-1,
    min_resol_shell=0.0,
):
    """Host-facing native-compatible oracle adapter; requires branch export.

    ``fsc_ssnr`` is unused with native update_tau2_with_fsc=false.
    ``min_resol_shell`` is unused in the pinned native reconstructGrad body.
    Native-equivalent validation errors are raised rather than concealed.
    """
    from recovar.relion_bind import _relion_bind_core as bind

    if interpolator != 1 or r_max > ori_size // 2:
        raise ValueError("unsupported interpolator or radius")
    pseudo = data_h1 is not None
    if pseudo != (weight_h1 is not None) or pseudo != (mom1_h1 is not None):
        raise ValueError("half-1 data, weight, and moment must be present together")
    if np.shape(fsc_ssnr) != (ori_size // 2 + 1,):
        raise ValueError("fsc_ssnr has incompatible shell shape")
    capacity = padding_factor * ori_size + 3

    def pack(value):
        value = np.asarray(value)
        if value.ndim != 3 or value.shape[0] != value.shape[1] or value.shape[2] != value.shape[0] // 2 + 1:
            raise ValueError("BPref must be a centered half volume")
        radius = r_max if r_max > 0 else ori_size // 2
        if value.shape[0] // 2 < padding_factor * radius or value.shape[0] > capacity:
            raise ValueError("BPref does not cover the logical radius or exceeds capacity")
        before = capacity // 2 - value.shape[0] // 2
        after = capacity - value.shape[0] - before
        return np.pad(value, ((before, after), (before, after), (0, capacity // 2 + 1 - value.shape[2])))

    if np.shape(data_h0) != np.shape(weight_h0) or (
        pseudo and (np.shape(data_h0) != np.shape(data_h1) or np.shape(data_h1) != np.shape(weight_h1))
    ):
        raise ValueError("halfset data/weight shapes differ")
    first0 = bind.vdam_first_moment_initializes(mom1_h0)
    first1 = bind.vdam_first_moment_initializes(mom1_h1) if pseudo else first0
    result = jax.device_get(
        relion_vdam_m_step_device(
            np.asarray(reference_relion, np.float64),
            pack(data_h0),
            pack(weight_h0),
            pack(data_h1) if pseudo else None,
            pack(weight_h1) if pseudo else None,
            np.asarray(mom1_h0, np.complex128),
            np.asarray(mom1_h1, np.complex128) if pseudo else None,
            np.asarray(mom2, np.complex128),
            np.asarray(fsc_reconstruct, np.float64),
            np.asarray(tau2, np.float64),
            np.float64(grad_stepsize),
            np.float64(tau2_fudge),
            np.int32(r_max),
            np.bool_(first0),
            np.bool_(first1),
            ori_size=ori_size,
            padding_factor=padding_factor,
            pseudo_halfsets=pseudo,
        )
    )
    if result.pop("_invalid_sigma2"):
        raise ValueError("native SSNR rejects unexpectedly small nonzero sigma2 sum")
    if result.pop("_invalid_tau2"):
        raise ValueError("native SSNR rejects negative tau2 in active shells")
    return result
