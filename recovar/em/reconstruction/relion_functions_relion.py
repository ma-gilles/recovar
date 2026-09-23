"""EM-only symbols relocated from ``recovar.reconstruction.relion_functions`` (relax split P1, pure move).

Function bodies are AST-identical to their originals at Q (tag q-reconcile-20260923); only the module
path changed.  See pr179_coordination/relax_split_plan_20260923/PLAN.md.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.core import mask, padding
from recovar.reconstruction.relion_functions import (  # noqa: F401  (staying helpers and shared loader state)
    _as_flat_single_volume,
    _relion_current_size_decenter_mask,
    _relion_idft3_real_from_fftw_half,
    _relion_reconstruction_padded_shape,
    adjust_regularization_relion_style,
    griddingCorrect,
    griddingCorrect_square,
)

_RELION_PROJECTION_PAD_HOST_FFT_MIN_VOXELS = 200_000_000


def _gridding_correct_trilinear(vol_real, ori_size, padding_factor):
    """Apply RELION-style gridding correction for trilinear interpolation.

    Divides each real-space voxel by sinc²(r / (N·pf)) where
    r = sqrt(x² + y² + z²) is the radial distance from the volume center.
    Matches RELION's ``Projector::griddingCorrect()`` (projector.cpp:27-60)
    for ``interpolator == TRILINEAR``.

    RELION uses a RADIAL sinc² correction (not separable per-axis).

    Parameters
    ----------
    vol_real : jnp.ndarray, shape (N, N, N)
        Real-space volume with origin at array center.
    ori_size : int
        Original box size (N).
    padding_factor : int
        Padding factor (1 or 2).

    Returns
    -------
    vol_corrected : jnp.ndarray, shape (N, N, N)
    """
    N = vol_real.shape[0]
    coords = jnp.arange(N, dtype=jnp.float64) - N / 2.0
    # 3D radial distance
    r = jnp.sqrt(coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2)
    arg = r / (ori_size * padding_factor)
    # sinc(x) = sin(πx)/(πx), sinc(0) = 1
    sinc_r = jnp.where(arg < 1e-15, 1.0, jnp.sin(jnp.pi * arg) / (jnp.pi * arg))
    sinc2_r = sinc_r**2
    return vol_real / sinc2_r


def _gridding_correct_trilinear_np(vol_real, ori_size, padding_factor):
    """NumPy equivalent of ``_gridding_correct_trilinear`` for host padding."""

    N = vol_real.shape[0]
    coords = np.arange(N, dtype=np.float64) - N / 2.0
    r = np.sqrt(coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2)
    arg = r / (ori_size * padding_factor)
    sinc_r = np.ones_like(arg)
    nz = arg >= 1e-15
    sinc_r[nz] = np.sin(np.pi * arg[nz]) / (np.pi * arg[nz])
    return vol_real / (sinc_r**2)


def _get_dft3_np(img, norm=fourier_transform_utils.DEFAULT_FFT_NORM, axes=(-3, -2, -1)):
    img = np.fft.fftshift(img, axes=axes)
    img = np.fft.fftn(img, axes=axes, norm=norm)
    img = np.fft.fftshift(img, axes=axes)
    return img


def _get_idft3_np(img, norm=fourier_transform_utils.DEFAULT_FFT_NORM, axes=(-3, -2, -1)):
    img = np.fft.ifftshift(img, axes=axes)
    img = np.fft.ifftn(img, axes=axes, norm=norm)
    img = np.fft.ifftshift(img, axes=axes)
    return img


def _pad_volume_for_projection_host(
    vol_ft_flat,
    volume_shape,
    padding_factor,
    *,
    do_gridding_correction=False,
    current_size=None,
):
    """Host-side projection padding for grids whose cuFFT workspace is too large.

    Preserves ``vol_ft_flat``'s own dtype throughout rather than forcing
    complex64: RELION's ``Projector::data``/``Iref`` are RFLOAT-precision
    (double, in our ACC_DOUBLE_PRECISION oracle) end to end, and this
    padding step sits directly in the projection path
    (``pad_volume_for_projection``, called every iteration whenever
    ``projection_padding_factor > 1``). Hardcoding complex64 here silently
    discarded whatever double precision the caller had already arranged
    for the input volume, regardless of RECOVAR_USE_FLOAT64_PROJECTIONS.
    """

    N = int(volume_shape[0])
    padded_shape = tuple(int(s) * int(padding_factor) for s in volume_shape)
    vol_ft_flat = np.asarray(vol_ft_flat)
    real_dtype = np.float64 if vol_ft_flat.dtype == np.complex128 else np.float32
    # _get_idft3_np's output is nominally real-valued but stays complex
    # (matching the original code) -- do not force a real cast here, that
    # would drop the (numerically negligible) imaginary residue with a
    # ComplexWarning for no benefit; np.pad/_get_dft3_np below operate
    # identically on the complex array either way.
    vol_real = _get_idft3_np(vol_ft_flat.reshape(volume_shape))
    if do_gridding_correction:
        vol_real = _gridding_correct_trilinear_np(vol_real, N, int(padding_factor))
    pad_amount = N * (int(padding_factor) - 1)
    pad_before = pad_amount // 2
    pad_after = pad_amount - pad_before
    vol_real_padded = np.pad(vol_real, [(pad_before, pad_after)] * 3, mode="constant")
    vol_ft_padded = _get_dft3_np(vol_real_padded).astype(vol_ft_flat.dtype, copy=False)

    if current_size is not None:
        r_max_ref = int(padding_factor) * (int(current_size) // 2)
        pN = int(padded_shape[0])
        coords = np.arange(pN, dtype=real_dtype) - pN / 2.0
        r2_3d = coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2
        vol_ft_padded = np.where(r2_3d <= r_max_ref**2, vol_ft_padded, 0.0)

    return jnp.asarray(vol_ft_padded.reshape(-1)), padded_shape


def pad_volume_for_projection(
    vol_ft_flat, volume_shape, padding_factor, do_gridding_correction=False, current_size=None
):
    """Pad a Fourier volume via real-space zero-padding for smoother projection.

    RELION pads volumes in REAL SPACE before FFT so that trilinear
    interpolation operates on a (pf*N)³ grid.  This is NOT the same as
    Fourier zero-padding (which leaves stride-pf gaps that degrade
    interpolation).

    When ``do_gridding_correction=True``, applies RELION's gridding correction
    (``Projector::griddingCorrect``) to the real-space volume before padding.
    This compensates for the smoothing inherent in trilinear Fourier-slice
    interpolation, matching RELION's projector behaviour.

    When ``current_size`` is provided, applies a spherical mask at
    ``r_max_ref = padding_factor * current_size // 2`` to the padded Fourier
    volume.  This matches RELION's ``Projector::computeFourierTransformMap``
    which calls ``decenter(data, Faux, max_r2)`` — only copying Fourier
    coefficients within ``r_max_ref`` and leaving everything beyond as zero.
    Without this mask, JAX's trilinear interpolation near the scoring-window
    boundary produces different (non-attenuated) values compared to RELION,
    because RELION's trilinear blends real data with zeros at the boundary.

    Parameters
    ----------
    vol_ft_flat : jnp.ndarray, shape (N³,)
        Flat centered Fourier volume at native resolution.
    volume_shape : tuple of int, (N, N, N)
        Native volume shape.
    padding_factor : int
        Padding factor (typically 2).
    do_gridding_correction : bool, optional
        If True, apply gridding correction before padding (default False).
    current_size : int, optional
        RELION's current resolution size.  When set, applies a spherical
        Fourier mask at radius ``padding_factor * current_size // 2`` to
        match RELION's projector data boundary.

    Returns
    -------
    padded_ft_flat : jnp.ndarray, shape ((pf*N)³,)
        Fourier volume on the (pf*N)³ grid, ready for slice_volume.
    padded_shape : tuple of int
        (pf*N, pf*N, pf*N).
    """
    if padding_factor == 1:
        return vol_ft_flat, volume_shape

    padded_shape = tuple(int(s) * int(padding_factor) for s in volume_shape)
    if int(np.prod(padded_shape)) > _RELION_PROJECTION_PAD_HOST_FFT_MIN_VOXELS:
        return _pad_volume_for_projection_host(
            vol_ft_flat,
            volume_shape,
            padding_factor,
            do_gridding_correction=do_gridding_correction,
            current_size=current_size,
        )

    N = volume_shape[0]
    vol_real = fourier_transform_utils.get_idft3(jnp.asarray(vol_ft_flat).reshape(volume_shape))
    if do_gridding_correction:
        vol_real = _gridding_correct_trilinear(vol_real, N, padding_factor)
    pad_amount = N * (padding_factor - 1)
    vol_real_padded = padding.pad_volume_spatial_domain(vol_real, pad_amount)
    vol_ft_padded = fourier_transform_utils.get_dft3(vol_real_padded)

    if current_size is not None:
        # Match RELION's Projector::computeFourierTransformMap decenter behaviour:
        # zero all Fourier coefficients beyond r_max_ref = pf * (cs // 2).
        r_max_ref = padding_factor * (current_size // 2)
        pN = padded_shape[0]
        coords = jnp.arange(pN, dtype=jnp.float32) - pN / 2.0
        r2_3d = coords[:, None, None] ** 2 + coords[None, :, None] ** 2 + coords[None, None, :] ** 2
        sphere_mask = (r2_3d <= r_max_ref**2).astype(vol_ft_padded.dtype)
        vol_ft_padded = vol_ft_padded.reshape(padded_shape) * sphere_mask
        vol_ft_padded = vol_ft_padded.reshape(-1)

    return vol_ft_padded.reshape(-1), padded_shape


def _regularize_large_relion_half_filter_impl(
    Ft_ctf,
    tau,
    og_volume_shape,
    volume_upsampling_factor,
    tau2_fudge,
    minres_map,
    current_size,
    accumulator_volume_shape,
    tau_is_1d,
    relion_filter_scale,
):
    """Build the large packed-half Wiener denominator without its numerator.

    Keeping this operation separate from the complex division is a memory
    boundary, not a different reconstruction formula.  In particular, the
    casts and arguments below mirror the large-grid branch in
    :func:`post_process_from_filter_v2` exactly.  Its donating executable can
    reuse the float32 CTF input while its regularization temporaries are live;
    the twice-as-large complex numerator may therefore remain on the host.
    """

    og_volume_shape = tuple(int(s) for s in og_volume_shape)
    volume_upsampling_factor = int(volume_upsampling_factor)
    upsampled_volume_shape = (
        tuple(3 * [og_volume_shape[0] * volume_upsampling_factor])
        if accumulator_volume_shape is None
        else tuple(int(s) for s in accumulator_volume_shape)
    )
    packed_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(upsampled_volume_shape)
    Ft_ctf_flat, _ = _as_flat_single_volume(Ft_ctf, packed_shape)
    Ft_ctf_flat = Ft_ctf_flat.real.astype(jnp.float32)
    tau_for_filter = None if tau is None else jnp.asarray(tau, dtype=jnp.float32)
    current_size_limited = current_size is not None and current_size > 0
    native_r_max = int(current_size) // 2 if current_size_limited else None
    regularized_filter = adjust_regularization_relion_style(
        Ft_ctf_flat,
        upsampled_volume_shape,
        tau=tau_for_filter,
        padding_factor=volume_upsampling_factor,
        half_volume=True,
        tau2_fudge=tau2_fudge,
        minres_map=minres_map,
        max_res_shell=native_r_max,
        relion_native_shell_floor=current_size_limited,
        native_volume_shape=og_volume_shape,
        tau_is_1d=tau_is_1d,
        relion_filter_scale=relion_filter_scale,
        large_grid_single_precision=True,
    )
    return regularized_filter.reshape(Ft_ctf.shape)


_regularize_large_relion_half_filter_donate_ctf = jax.jit(
    _regularize_large_relion_half_filter_impl,
    static_argnums=(2, 3, 5, 6, 7, 8, 9),
    donate_argnums=(0,),
)


def _divide_large_relion_half_numerator_impl(
    F_ty,
    regularized_filter,
    volume_upsampling_factor,
    current_size,
    accumulator_volume_shape,
):
    """Apply the exact packed-half support mask and Wiener division.

    XLA fuses this elementwise stage into the donated complex64 numerator, so
    it needs no box-scale temporary in addition to the numerator and the
    already-regularized float32 denominator.
    """

    upsampled_volume_shape = tuple(int(s) for s in accumulator_volume_shape)
    packed_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(upsampled_volume_shape)
    F_ty_flat, _ = _as_flat_single_volume(F_ty, packed_shape)
    F_ty_flat = F_ty_flat.astype(jnp.complex64)
    current_size_limited = current_size is not None and current_size > 0
    if current_size_limited:
        wiener_radius = int(volume_upsampling_factor) * (int(current_size) // 2)
        valid_mask = _relion_current_size_decenter_mask(
            upsampled_volume_shape,
            wiener_radius,
            half_volume=True,
        )
    else:
        wiener_radius = upsampled_volume_shape[0] // 2 - 1
        valid_mask = fourier_transform_utils.full_volume_to_half_volume(
            mask.get_radial_mask(upsampled_volume_shape, radius=wiener_radius),
            upsampled_volume_shape,
        )
    valid_indices = valid_mask.reshape(-1).astype(F_ty_flat.real.dtype)
    divided = (F_ty_flat * valid_indices) / jnp.asarray(regularized_filter).reshape(-1)
    return divided.astype(jnp.complex64).reshape(F_ty.shape)


_divide_large_relion_half_numerator_donate_numerator = jax.jit(
    _divide_large_relion_half_numerator_impl,
    static_argnums=(2, 3, 4),
    donate_argnums=(0,),
)


def _finish_large_relion_postprocess_from_unpadded_real_impl(
    vol,
    og_volume_shape,
    volume_upsampling_factor,
    kernel="triangular",
    use_spherical_mask=True,
    grid_correct=True,
    gridding_correct="square",
    kernel_width=1,
    volume_mask=None,
    return_real_space=False,
    return_half_volume=False,
    gridding_padding_factor=None,
    gridding_order=None,
):
    """Finish large-grid post-processing from an unpadded real volume."""

    vol = jnp.asarray(vol).reshape(og_volume_shape)
    if use_spherical_mask:
        vol, _ = mask.soft_mask_outside_map(vol, cosine_width=3)

    if volume_mask is not None:
        vol = vol * volume_mask

    vol = vol.astype(jnp.complex64 if np.issubdtype(vol.dtype, np.complexfloating) else jnp.float32)

    if grid_correct:
        order = gridding_order if gridding_order is not None else (1 if kernel == "triangular" else 0)
        grid_fn = griddingCorrect_square if gridding_correct == "square" else griddingCorrect
        gc_pf = gridding_padding_factor if gridding_padding_factor is not None else volume_upsampling_factor
        vol, _ = grid_fn(vol, og_volume_shape[0], gc_pf / kernel_width, order=order)
        vol = vol.astype(jnp.complex64 if np.issubdtype(vol.dtype, np.complexfloating) else jnp.float32)

    if return_real_space:
        return vol.real.astype(jnp.float32)

    vol = fourier_transform_utils.get_dft3_real(vol)
    if return_half_volume:
        return vol.reshape(-1).astype(jnp.complex64)
    vol = fourier_transform_utils.half_volume_to_full_volume(vol, og_volume_shape)
    return vol.astype(jnp.complex64)


@functools.partial(jax.jit, static_argnums=[1, 2, 3, 4, 5, 6, 7, 9, 10])
def _finish_large_relion_postprocess_from_unpadded_real(
    vol,
    og_volume_shape,
    volume_upsampling_factor,
    kernel="triangular",
    use_spherical_mask=True,
    grid_correct=True,
    gridding_correct="square",
    kernel_width=1,
    volume_mask=None,
    return_real_space=False,
    return_half_volume=False,
    gridding_padding_factor=None,
    gridding_order=None,
):
    """Finish a host-iFFT reconstruction after its real-space center crop."""

    return _finish_large_relion_postprocess_from_unpadded_real_impl(
        vol,
        og_volume_shape,
        volume_upsampling_factor,
        kernel=kernel,
        use_spherical_mask=use_spherical_mask,
        grid_correct=grid_correct,
        gridding_correct=gridding_correct,
        kernel_width=kernel_width,
        volume_mask=volume_mask,
        return_real_space=return_real_space,
        return_half_volume=return_half_volume,
        gridding_padding_factor=gridding_padding_factor,
        gridding_order=gridding_order,
    )


@functools.partial(jax.jit, static_argnums=[1, 2, 3, 4, 5, 6, 7, 9, 10])
def _finish_large_relion_postprocess_from_fftw_half(
    vol_half,
    og_volume_shape,
    volume_upsampling_factor,
    kernel="triangular",
    use_spherical_mask=True,
    grid_correct=True,
    gridding_correct="square",
    kernel_width=1,
    volume_mask=None,
    return_real_space=False,
    return_half_volume=False,
    gridding_padding_factor=None,
    gridding_order=None,
):
    """Finish a large c64/f32 reconstruction from a raw-FFTW half-volume.

    This is exactly the portion of :func:`post_process_from_filter_v2` after
    its padded inverse-FFT boundary.  Keeping it in a separate executable lets
    the eager caller release the accumulator executable and its device inputs
    before allocating the inverse-FFT workspace.
    """

    reconstruction_volume_shape = _relion_reconstruction_padded_shape(
        og_volume_shape,
        volume_upsampling_factor,
    )
    vol = _relion_idft3_real_from_fftw_half(vol_half, reconstruction_volume_shape)
    vol = padding.unpad_volume_spatial_domain(vol, reconstruction_volume_shape[0] - og_volume_shape[0])
    return _finish_large_relion_postprocess_from_unpadded_real_impl(
        vol,
        og_volume_shape,
        volume_upsampling_factor,
        kernel=kernel,
        use_spherical_mask=use_spherical_mask,
        grid_correct=grid_correct,
        gridding_correct=gridding_correct,
        kernel_width=kernel_width,
        volume_mask=volume_mask,
        return_real_space=return_real_space,
        return_half_volume=return_half_volume,
        gridding_padding_factor=gridding_padding_factor,
        gridding_order=gridding_order,
    )
