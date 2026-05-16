"""Bit-exact JAX port of RELION's `Projector::project` (relion/src/projector.cpp:630-790).

Replaces `recovar.core.slicing._jax_slice_half_image` for the RELION-parity
path so the projection of a Fourier volume matches RELION's `Projector::project`
output bit-for-bit.

Algorithm (per output pixel (i, x) of the half-image (Y, X//2+1)):
  1. y = i if i <= r_max_out else i - YSIZE     (FFTW-natural y indexing)
  2. (xp, yp, zp) = Ainv @ (x, y, 0)
  3. Skip if x*x + y*y > r_max_out² OR r² > r_max_ref²    → zero
  4. If xp < 0: negate (xp, yp, zp), record is_neg_x
  5. x0 = floor(xp); fx = xp - x0; y0 = floor(yp); fy = yp - y0; z0 = floor(zp); fz = zp - z0
  6. y0 -= STARTINGY; z0 -= STARTINGZ                     (Xmipp-origin shift)
  7. Skip if any of x0, x0+1, y0, y0+1, z0, z0+1 out of bounds  → zero
  8. Trilinear interp: f2d[i, x] = LIN_INTERP(fz, dxy0, dxy1)
  9. If is_neg_x: f2d[i, x] = conj(f2d[i, x])

Input volume is in RELION's Projector internal layout:
  shape (pad_size, pad_size, pad_size // 2 + 1), complex
  Xmipp origin: STARTINGX=0, STARTINGY=-(pad_size//2), STARTINGZ=-(pad_size//2)
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def relion_project_half(
    volume_relion_half: jnp.ndarray,
    R_relion: jnp.ndarray,
    image_size: int,
    r_max: int,
    padding_factor: int = 1,
) -> jnp.ndarray:
    """Project a RELION-frame Fourier volume to a 2D half-image.

    Parameters
    ----------
    volume_relion_half : complex array of shape ``(pad_size, pad_size, pad_size//2+1)``
        RELION's Projector internal data layout (FFTW-natural y/z, half-complex x).
        ``pad_size = ori_size * padding_factor``.
    R_relion : (3, 3) rotation matrix in RELION convention.
    image_size : 2D output image size (assumed square).
    r_max : RELION's `r_max` field on the projector (= ori_size//2 - 1 typically).
    padding_factor : projector padding (1 or 2 typically).

    Returns
    -------
    f2d : complex (image_size, image_size//2+1) half-image. Pixels outside the
          radial mask return 0.
    """
    pad_z, pad_y, half_x = volume_relion_half.shape
    starting_x = 0
    starting_y = -(pad_y // 2)
    starting_z = -(pad_z // 2)

    out_h = image_size
    out_w = image_size // 2 + 1
    r_max_out = out_w - 1
    r_max_out_2 = r_max_out * r_max_out
    r_max_ref = r_max * padding_factor
    r_max_ref_2 = r_max_ref * r_max_ref

    Ainv = jnp.linalg.inv(R_relion).astype(jnp.float64) * float(padding_factor)

    # FFTW-natural y indexing: y = i if i <= r_max_out else i - out_h
    i_arr = jnp.arange(out_h, dtype=jnp.int32)
    x_arr = jnp.arange(out_w, dtype=jnp.int32)
    y_arr = jnp.where(i_arr <= r_max_out, i_arr, i_arr - out_h)

    Y, X = jnp.meshgrid(y_arr, x_arr, indexing="ij")  # (out_h, out_w)
    Xf = X.astype(jnp.float64)
    Yf = Y.astype(jnp.float64)

    # (xp, yp, zp) = Ainv @ (x, y, 0)
    xp = Ainv[0, 0] * Xf + Ainv[0, 1] * Yf
    yp = Ainv[1, 0] * Xf + Ainv[1, 1] * Yf
    zp = Ainv[2, 0] * Xf + Ainv[2, 1] * Yf

    # Hermitian flip for negative xp
    is_neg_x = xp < 0
    xp = jnp.where(is_neg_x, -xp, xp)
    yp = jnp.where(is_neg_x, -yp, yp)
    zp = jnp.where(is_neg_x, -zp, zp)

    r2_ref = xp * xp + yp * yp + zp * zp
    r2_out = (X * X + Y * Y).astype(jnp.float64)

    # Floor + fractional
    x0 = jnp.floor(xp).astype(jnp.int32)
    y0 = jnp.floor(yp).astype(jnp.int32)
    z0 = jnp.floor(zp).astype(jnp.int32)
    fx = xp - x0.astype(jnp.float64)
    fy = yp - y0.astype(jnp.float64)
    fz = zp - z0.astype(jnp.float64)

    # Subtract Xmipp starting indices
    x0r = x0 - starting_x  # = x0
    y0r = y0 - starting_y
    z0r = z0 - starting_z

    x1r = x0r + 1
    y1r = y0r + 1
    z1r = z0r + 1

    # Bounds: matches RELION's "if (x0 < 0 || x0+1 >= data.xdim || ...) continue"
    in_bounds = (x0r >= 0) & (x1r < half_x) & (y0r >= 0) & (y1r < pad_y) & (z0r >= 0) & (z1r < pad_z)
    valid = in_bounds & (r2_ref <= r_max_ref_2) & (r2_out <= r_max_out_2)

    # Clip indices for safe gather (masked-out anyway)
    x0c = jnp.clip(x0r, 0, half_x - 1)
    x1c = jnp.clip(x1r, 0, half_x - 1)
    y0c = jnp.clip(y0r, 0, pad_y - 1)
    y1c = jnp.clip(y1r, 0, pad_y - 1)
    z0c = jnp.clip(z0r, 0, pad_z - 1)
    z1c = jnp.clip(z1r, 0, pad_z - 1)

    # 8-corner gather (RELION DIRECT_A3D_ELEM(data, z, y, x) = data[z][y][x])
    v = volume_relion_half
    d000 = v[z0c, y0c, x0c]
    d001 = v[z0c, y0c, x1c]
    d010 = v[z0c, y1c, x0c]
    d011 = v[z0c, y1c, x1c]
    d100 = v[z1c, y0c, x0c]
    d101 = v[z1c, y0c, x1c]
    d110 = v[z1c, y1c, x0c]
    d111 = v[z1c, y1c, x1c]

    # Nested LIN_INTERP, in the SAME ORDER as RELION (lines 733-740):
    #   dx00 = LIN_INTERP(fx, d000, d001)
    #   dx01 = LIN_INTERP(fx, d100, d101)
    #   dx10 = LIN_INTERP(fx, d010, d011)
    #   dx11 = LIN_INTERP(fx, d110, d111)
    #   dxy0 = LIN_INTERP(fy, dx00, dx10)
    #   dxy1 = LIN_INTERP(fy, dx01, dx11)
    #   f2d  = LIN_INTERP(fz, dxy0, dxy1)
    dx00 = d000 + fx * (d001 - d000)
    dx01 = d100 + fx * (d101 - d100)
    dx10 = d010 + fx * (d011 - d010)
    dx11 = d110 + fx * (d111 - d110)
    dxy0 = dx00 + fy * (dx10 - dx00)
    dxy1 = dx01 + fy * (dx11 - dx01)
    val = dxy0 + fz * (dxy1 - dxy0)

    val = jnp.where(is_neg_x, jnp.conj(val), val)
    val = jnp.where(valid, val, jnp.zeros_like(val))
    return val


def gridding_correct_volume_real(volume_real: jnp.ndarray, ori_size: int, padding_factor: int = 1) -> jnp.ndarray:
    """RELION's gridding correction (relion/src/projector.cpp:595-628).

    Divide each real-space voxel by ``sinc²(r / (ori * padding_factor))`` where
    r is the pixel-radius from the centered origin. This compensates for the
    sinc² smoothing introduced by trilinear interpolation in 3D Fourier space.

    Apply BEFORE the forward FFT and BEFORE conversion to RELION-half-complex
    layout.
    """
    N = volume_real.shape[0]
    c = N // 2
    coord = jnp.arange(N, dtype=jnp.float64) - c
    K, I, J = jnp.meshgrid(coord, coord, coord, indexing="ij")
    r = jnp.sqrt(K * K + I * I + J * J)
    # Avoid /0 at center: sinc(0) = 1 → divide by 1 there.
    rval = r / (ori_size * padding_factor)
    pirval = jnp.pi * rval
    sinc = jnp.where(r > 0.0, jnp.sin(pirval) / pirval, jnp.ones_like(r))
    return volume_real / (sinc * sinc)


def centered_full_to_relion_half(volume_centered_full: jnp.ndarray) -> jnp.ndarray:
    """Convert recovar's centered full Fourier volume (N, N, N) to RELION's
    Projector half-complex layout (N, N, N//2+1).

    Recovar stores the FFT centered (DC at [N/2, N/2, N/2]). RELION's Projector
    stores `vol[z, y, x]` with the y/z axes in centered storage (so raw index 0
    corresponds to k = -N/2, matching Xmipp setXmippOrigin), and the x axis in
    half-complex non-negative-frequency layout (raw index 0 = DC, raw index
    N//2 = Nyquist).

    The x conversion is `ifftshift on axis 2 then take [:N//2+1]` which produces
    the correct ordering [DC, k=1, ..., k=N//2-1, k=N//2 (Nyquist)] from the
    centered layout, with the Nyquist bin coming from the centered index 0
    (which equals k=-N/2 = +N/2 modulo N for real-input FFTs).
    """
    N = volume_centered_full.shape[0]
    return jnp.fft.ifftshift(volume_centered_full, axes=2)[:, :, : N // 2 + 1]
