"""RELION projector construction shared by EM and VDAM.

Matches Projector::computeFourierTransformMap for a 3-D RELION-frame real
reference, data_dim=2 and trilinear gridding correction. No axis/contrast
conversion is performed by the device kernel. Host wrappers below convert
RECOVAR references and select the native or JAX implementation. The FFT and output capacity depend only on the
original size and padding; the logical radius remains a device scalar.

For M=padding_factor*ori_size, output capacity is
(M+3, M+3, M//2+2). Major payload sizes are 8*M**3 bytes for padded real data,
16*M*M*(M//2+1) for the FFT, and 16*(M+3)**2*(M//2+2) for output, plus FFT
workspace and compiler temporaries. One reference is processed per call.
"""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.core.relion_project import gridding_correct_volume_real
from recovar.em.helpers.deterministic_reduce import (
    deterministic_reductions_enabled,
    fixed_order_shell_sums,
    static_shell_voxel_lists,
)


@partial(jax.jit, static_argnames=("ori_size", "padding_factor"))
def setup_relion_projector(
    reference_relion,
    r_max,
    *,
    ori_size: int,
    padding_factor: int = 1,
    do_gridding=True,
):
    """Return fixed-capacity complex128 projector data and float64 power.

    ``r_max`` is the unpadded logical radius; negative means original Nyquist,
    zero means DC only, and larger radii are clamped like native initialiseData.
    For native comparison, crop the centered y/z region of width
    ``2*(padding_factor*r_max+1)+1`` and the corresponding positive-x prefix.
    Cast to complex64 only at a consumer boundary that already requires it.

    The global RECOVAR x64 policy is required. No Python callbacks, host
    materialization, persistent mutable cache, or per-class batching is used.
    """
    _validate_reference(reference_relion, ori_size, padding_factor)
    if not jax.config.x64_enabled:
        raise ValueError("RELION projector setup requires JAX float64 support")
    reference = jnp.asarray(reference_relion, dtype=jnp.float64)
    reference = jax.lax.cond(
        jnp.asarray(do_gridding, dtype=jnp.bool_),
        lambda volume: gridding_correct_volume_real(volume, ori_size, padding_factor),
        lambda volume: volume,
        reference,
    )
    return _project_reference(reference, r_max, ori_size, padding_factor)


@partial(jax.jit, static_argnames=("ori_size", "padding_factor", "compute_dtype"))
def setup_relion_projector_uncorrected(
    reference_relion,
    r_max,
    *,
    ori_size: int,
    padding_factor: int = 1,
    compute_dtype=jnp.float64,
):
    """Prepare an uncorrected M-step projector at the requested precision.

    The FFT, power reductions and shell geometry use ``compute_dtype``;
    outputs are complex64/float32 or complex128/float64. This static path
    does not trace the double-precision gridding-correction branch used by
    the existing E-step wrapper. Radius and Nyquist ownership are shared.
    """
    _validate_reference(reference_relion, ori_size, padding_factor)
    dtype = jnp.dtype(compute_dtype)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError("Projector computation dtype must be float32 or float64")
    if dtype == jnp.dtype(jnp.float64) and not jax.config.x64_enabled:
        raise ValueError("Float64 projector setup requires JAX float64 support")
    reference = jnp.asarray(reference_relion, dtype=dtype)
    return _project_reference(reference, r_max, ori_size, padding_factor)


def _validate_reference(reference_relion, ori_size, padding_factor):
    if ori_size <= 0 or ori_size % 2:
        raise ValueError("ori_size must be positive and even")
    if padding_factor not in (1, 2):
        raise ValueError("projector setup supports padding_factor 1 or 2")
    if reference_relion.shape != (ori_size,) * 3:
        raise ValueError("reference_relion must have shape (ori_size,)*3")


def _project_reference(reference, r_max, ori_size, padding_factor):
    """Shared FFT, native half-grid ownership and shell-power calculation."""
    fft_size = padding_factor * ori_size
    pad = (fft_size - ori_size) // 2
    padded = jnp.pad(reference, ((pad, pad),) * 3)
    # Native FourierTransformer divides by M^3, then projector.cpp multiplies
    # by pf^3*N for 3-D references projected into 2-D images. Keep that order.
    transformed = ftu.get_dft3_real(padded, norm="forward")
    transformed = transformed * float(padding_factor**3 * ori_size)

    capacity = fft_size + 3
    center = capacity // 2
    coord = jnp.arange(capacity, dtype=jnp.int32) - center
    x = jnp.arange(capacity // 2 + 1, dtype=jnp.int32)
    z = coord[:, None, None]
    y = coord[None, :, None]
    x_grid = x[None, None, :]
    r2 = z * z + y * y + x_grid * x_grid
    radius = jnp.asarray(r_max, dtype=jnp.int32)
    radius = jnp.where(radius < 0, ori_size // 2, jnp.minimum(radius, ori_size // 2))
    valid = (
        (z > -fft_size // 2)
        & (z <= fft_size // 2)
        & (y > -fft_size // 2)
        & (y <= fft_size // 2)
        & (x_grid <= fft_size // 2)
        & (r2 <= (padding_factor * radius) ** 2)
    )
    # ftu centers y/z with -Nyquist at index zero. RELION's FFTW traversal
    # instead gives that same coefficient +Nyquist; never duplicate it at -N/2.
    yz_indices = (coord + fft_size // 2) % fft_size
    x_indices = jnp.minimum(x, fft_size // 2)
    gathered = transformed[yz_indices[:, None, None], yz_indices[None, :, None], x_indices[None, None, :]]
    projector = jnp.where(valid, gathered, jnp.asarray(0, dtype=transformed.dtype))
    shells = jnp.floor(jnp.sqrt(r2.astype(reference.dtype)) / padding_factor + 0.5).astype(jnp.int32)
    shells = jnp.minimum(shells, ori_size // 2)
    # Native uses norm(complex)/2 rather than abs(complex)**2/2.
    power = (projector.real * projector.real + projector.imag * projector.imag) / 2.0
    if deterministic_reductions_enabled():
        # ``bincount`` lowers to a scatter-add with duplicate shells (float
        # atomics); use static per-shell gathers with fixed-order reductions.
        shell_lists = static_shell_voxel_lists(capacity, padding_factor, ori_size // 2 + 1, clamp_to_last=True)
        sums = fixed_order_shell_sums(power.reshape(-1), shell_lists, reference.dtype)
        counts = fixed_order_shell_sums(valid.reshape(-1).astype(reference.dtype), shell_lists, reference.dtype)
    else:
        sums = jnp.bincount(shells.reshape(-1), weights=power.reshape(-1), length=ori_size // 2 + 1)
        counts = jnp.bincount(
            shells.reshape(-1), weights=valid.reshape(-1).astype(reference.dtype), length=ori_size // 2 + 1
        )
    spectrum = jnp.where(counts >= 1, sums / jnp.maximum(counts, 1), jnp.zeros((), dtype=reference.dtype))
    return projector, spectrum


ProjectorSetupBackend = Literal["native", "jax"]


def reference_to_relion_projector_half_maps(
    references: np.ndarray,
    *,
    current_size: int,
    padding_factor: int = 1,
    interpolator: int = 1,
    projector_setup_backend: ProjectorSetupBackend = "native",
) -> tuple[np.ndarray, int]:
    """Convert references to RELION half maps without retaining their spectrum."""
    half_maps, _power, r_max = reference_to_relion_projector_half_maps_and_power(
        references,
        current_size=current_size,
        padding_factor=padding_factor,
        interpolator=interpolator,
        projector_setup_backend=projector_setup_backend,
    )
    return half_maps, r_max


def reference_to_relion_projector_half_maps_and_power(
    references: np.ndarray,
    *,
    current_size: int,
    padding_factor: int = 1,
    interpolator: int = 1,
    projector_setup_backend: ProjectorSetupBackend = "native",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert references to native-layout half maps and their corrected spectrum.

    The opt-in JAX backend keeps its FP64 FFT at full capacity as current_size
    changes. Only the logical crop and complex64 consumer conversion vary.
    Unsupported projector geometry retains the native implementation.
    """
    from recovar.utils.helpers import recovar_volume_to_relion

    if projector_setup_backend not in {"native", "jax"}:
        raise ValueError(f"Unknown projector_setup_backend: {projector_setup_backend!r}")
    refs = np.asarray(references)
    if refs.ndim != 4:
        raise ValueError(f"references must have shape (K, N, N, N), got {refs.shape}")
    n = int(refs.shape[-1])
    use_jax = (
        projector_setup_backend == "jax"
        and n > 0 and n % 2 == 0
        and refs.shape[1:] == (n, n, n)
        and int(padding_factor) in {1, 2}
        and int(interpolator) == 1
    )
    if use_jax:
        import jax
        import jax.numpy as jnp

    else:
        from recovar.relion_bind import _relion_bind_core as bind

    halves = []
    power_spectra = []
    r_max_values = []
    for ref in refs:
        ref_relion = np.asarray(recovar_volume_to_relion(ref), dtype=np.float64)
        if use_jax:
            # Projector::initialiseData uses a negative size for full resolution;
            # zero means radius zero here (state wrappers retain their defaults).
            r_max = n // 2 if int(current_size) < 0 else min(int(current_size) // 2, n // 2)
            projector_data, power = setup_relion_projector(
                ref_relion, np.int32(r_max), ori_size=n,
                padding_factor=int(padding_factor),
            )
            logical_size = 2 * (int(padding_factor) * r_max + 1) + 1
            start = projector_data.shape[0] // 2 - logical_size // 2
            projector_data = projector_data[
                start : start + logical_size, start : start + logical_size,
                : logical_size // 2 + 1,
            ].astype(jnp.complex64)
            projector_data, power = jax.device_get((projector_data, power))
        else:
            (
                projector_data, power, _ori_size, _padding_factor_out,
                r_max, _r_min_nn, _interpolator_out,
            ) = bind.compute_fourier_transform_map(
                ref_relion,
                n,
                int(padding_factor),
                int(interpolator),
                int(current_size),
                True,
                2,
            )
        halves.append(np.asarray(projector_data))
        power_spectra.append(np.asarray(power, dtype=np.float64))
        r_max_values.append(int(r_max))
    if len(set(r_max_values)) != 1:
        raise ValueError(f"RELION projector maps disagree on r_max: {r_max_values}")
    return (
        np.asarray(halves),
        np.asarray(power_spectra, dtype=np.float64),
        int(r_max_values[0]),
    )
