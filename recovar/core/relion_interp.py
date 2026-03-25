"""RELION-style Fourier slice projection and backprojection.

This module implements trilinear and nearest-neighbor interpolation that is
functionally identical to RELION's CUDA kernels in ``BP.cuh`` and
``acc_projectorkernel_impl.h``, adapted to recovar's storage conventions.

Key RELION behaviors reproduced here:
  - Trilinear: cascaded-lerp form ``a + (b-a)*t`` (matches RELION's ``no_tex3D``)
  - Nearest: round-to-nearest (RELION only has trilinear; nearest added here)
  - Half-volume: Hermitian symmetry via per-pixel flip (if xp < 0, negate all
    coords and conjugate), matching RELION's convention
  - Radius clipping: ``|r|² > max_r²`` → zero (RELION's sphere mask)
  - Boundary: zero for out-of-bounds neighbors (equivalent to RELION's
    zero-padded volume allocation)

Storage conventions (kept from recovar, NOT changed to RELION's):
  - Complex: interleaved (jax complex64/128), not split real/imag
  - Volume: centered fftshift, last-axis Hermitian (kz ≥ 0 half)
  - Rotation: 3×3 matrix, applied as ``[k0, k1, 0] @ R`` (recovar convention,
    transpose of RELION's ``R @ [x, y, 0]^T``)

References:
  - RELION forward projection: ``AccProjectorKernel::project3Dmodel`` +
    ``no_tex3D`` in ``cuda_device_utils.cuh``
  - RELION backprojection: ``cuda_kernel_backproject3D`` in ``BP.cuh``
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu


# ---------------------------------------------------------------------------
# Coordinate generation
# ---------------------------------------------------------------------------


def _pixel_frequencies(image_shape, half_image=False):
    """Generate 2D frequency coordinates for each pixel.

    Uses the SAME ``meshgrid("xy")`` convention as recovar's existing
    ``get_k_coordinate_of_each_pixel``, ensuring identical pixel ordering.

    For full image: uses ``ftu.get_k_coordinate_of_each_pixel``.
    For half image: uses ``ftu.get_k_coordinate_of_each_pixel_half``.

    Returns shape ``(n_pixels, 2)`` where n_pixels is H*W (full) or
    H*(W//2+1) (half).
    """
    if half_image:
        return ftu.get_k_coordinate_of_each_pixel_half(image_shape, voxel_size=1, scaled=False)
    else:
        return ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size=1, scaled=False)


def _rotate_plane_coords(k2d, rotation_matrix):
    """Rotate 2D frequency coordinates to 3D.

    Uses recovar's convention: ``rotated = [freq0, freq1, 0] @ R``.
    Uses ``Precision.HIGHEST`` to match the existing ``geometry.py`` matmul
    precision, ensuring bit-exact agreement with ``rotations_to_grid_point_coords``.

    ``k2d``: shape ``(n_pixels, 2)``
    ``rotation_matrix``: shape ``(3, 3)``
    Returns ``(n_pixels, 3)``
    """
    k3d = jnp.pad(k2d, ((0, 0), (0, 1)))
    return jnp.matmul(k3d, rotation_matrix, precision=jax.lax.Precision.HIGHEST)


# ---------------------------------------------------------------------------
# Trilinear interpolation (RELION's no_tex3D style, cascaded lerp)
# ---------------------------------------------------------------------------


def _trilinear_gather_full(vol, g0, g1, g2, N0, N1, N2):
    """Trilinear interpolation from a full (centered) volume.

    Matches RELION's ``no_tex3D`` using cascaded lerp form.
    Out-of-bounds neighbors contribute zero (equivalent to RELION's padding).

    Args:
        vol: complex array shape ``(N0, N1, N2)``
        g0, g1, g2: float grid coordinates, shape ``(n,)``

    Returns:
        Interpolated values, shape ``(n,)``
    """
    # Floor and fractional parts
    b0 = jnp.floor(g0).astype(jnp.int32)
    b1 = jnp.floor(g1).astype(jnp.int32)
    b2 = jnp.floor(g2).astype(jnp.int32)
    fx = g0 - b0.astype(g0.dtype)
    fy = g1 - b1.astype(g1.dtype)
    fz = g2 - b2.astype(g2.dtype)

    # Read 8 neighbors with bounds checking (zero for OOB)
    def _safe_read(vol, i0, i1, i2):
        in_bounds = (i0 >= 0) & (i0 < N0) & (i1 >= 0) & (i1 < N1) & (i2 >= 0) & (i2 < N2)
        # Clamp to valid range for indexing, then mask
        ci0 = jnp.clip(i0, 0, N0 - 1)
        ci1 = jnp.clip(i1, 0, N1 - 1)
        ci2 = jnp.clip(i2, 0, N2 - 1)
        val = vol[ci0, ci1, ci2]
        return jnp.where(in_bounds, val, jnp.zeros_like(val))

    # 8 corners
    d000 = _safe_read(vol, b0, b1, b2)
    d001 = _safe_read(vol, b0, b1, b2 + 1)
    d010 = _safe_read(vol, b0, b1 + 1, b2)
    d011 = _safe_read(vol, b0, b1 + 1, b2 + 1)
    d100 = _safe_read(vol, b0 + 1, b1, b2)
    d101 = _safe_read(vol, b0 + 1, b1, b2 + 1)
    d110 = _safe_read(vol, b0 + 1, b1 + 1, b2)
    d111 = _safe_read(vol, b0 + 1, b1 + 1, b2 + 1)

    # Cascaded lerp (RELION's no_tex3D form):
    # lerp in z (last axis) first, then y, then x
    dx00 = d000 + (d001 - d000) * fz
    dx01 = d100 + (d101 - d100) * fz
    dx10 = d010 + (d011 - d010) * fz
    dx11 = d110 + (d111 - d110) * fz

    dxy0 = dx00 + (dx10 - dx00) * fy
    dxy1 = dx01 + (dx11 - dx01) * fy

    return dxy0 + (dxy1 - dxy0) * fx


def _trilinear_gather_half(half_vol, g0, g1, g2, N0, N1, N2):
    """Trilinear interpolation from a half (kz ≥ 0) volume.

    Matches RELION's Hermitian convention: for each of the 8 trilinear
    neighbors, if the centered kz index is negative, read the Hermitian
    partner at ``((N0-(N0&1)-j0)%N0, (N1-(N1&1)-j1)%N1, |kz|)`` and
    conjugate the value.

    Args:
        half_vol: complex array shape ``(N0, N1, N2//2+1)``
        g0, g1, g2: float grid coordinates in FULL centered convention
        N0, N1, N2: full volume dimensions (N2 is the FULL size, not half)
    """
    ic2 = N2 // 2
    N2_half = ic2 + 1

    b0 = jnp.floor(g0).astype(jnp.int32)
    b1 = jnp.floor(g1).astype(jnp.int32)
    b2 = jnp.floor(g2).astype(jnp.int32)
    fx = g0 - b0.astype(g0.dtype)
    fy = g1 - b1.astype(g1.dtype)
    fz = g2 - b2.astype(g2.dtype)

    def _safe_read_hermitian(i0, i1, i2_full):
        """Read from half-vol with Hermitian fold for negative kz."""
        kz = i2_full - ic2
        # For even N2, kz=-ic2 is Nyquist (self-conjugate) — map directly.
        # For odd N2, kz=-ic2 is a regular negative freq that needs fold.
        if N2 % 2 == 0:
            kz_mapped = jnp.where(kz == -ic2, ic2, kz)
        else:
            kz_mapped = kz

        # Positive kz (or Nyquist): read directly
        pos_i0 = i0
        pos_i1 = i1
        pos_kz = kz_mapped

        # Negative kz: Hermitian partner
        neg_i0 = (N0 - (N0 & 1) - i0) % N0
        neg_i1 = (N1 - (N1 & 1) - i1) % N1
        neg_kz = -kz_mapped

        # Choose based on mapped kz sign
        use_neg = kz_mapped < 0
        ri0 = jnp.where(use_neg, neg_i0, pos_i0)
        ri1 = jnp.where(use_neg, neg_i1, pos_i1)
        rkz = jnp.where(use_neg, neg_kz, pos_kz)

        # Bounds check
        in_bounds = (
            (i0 >= 0)
            & (i0 < N0)
            & (i1 >= 0)
            & (i1 < N1)
            & (i2_full >= 0)
            & (i2_full < N2)
            & (rkz >= 0)
            & (rkz < N2_half)
        )

        ci0 = jnp.clip(ri0, 0, N0 - 1)
        ci1 = jnp.clip(ri1, 0, N1 - 1)
        ckz = jnp.clip(rkz, 0, N2_half - 1)
        val = half_vol[ci0, ci1, ckz]

        # Conjugate for negative kz
        val = jnp.where(use_neg, jnp.conj(val), val)
        return jnp.where(in_bounds, val, jnp.zeros_like(val))

    # 8 corners
    d000 = _safe_read_hermitian(b0, b1, b2)
    d001 = _safe_read_hermitian(b0, b1, b2 + 1)
    d010 = _safe_read_hermitian(b0, b1 + 1, b2)
    d011 = _safe_read_hermitian(b0, b1 + 1, b2 + 1)
    d100 = _safe_read_hermitian(b0 + 1, b1, b2)
    d101 = _safe_read_hermitian(b0 + 1, b1, b2 + 1)
    d110 = _safe_read_hermitian(b0 + 1, b1 + 1, b2)
    d111 = _safe_read_hermitian(b0 + 1, b1 + 1, b2 + 1)

    # Cascaded lerp (RELION's no_tex3D form)
    dx00 = d000 + (d001 - d000) * fz
    dx01 = d100 + (d101 - d100) * fz
    dx10 = d010 + (d011 - d010) * fz
    dx11 = d110 + (d111 - d110) * fz

    dxy0 = dx00 + (dx10 - dx00) * fy
    dxy1 = dx01 + (dx11 - dx01) * fy

    return dxy0 + (dxy1 - dxy0) * fx


# ---------------------------------------------------------------------------
# Nearest-neighbor interpolation
# ---------------------------------------------------------------------------


def _nearest_gather_full(vol, g0, g1, g2, N0, N1, N2):
    """Nearest-neighbor interpolation from a full volume."""
    i0 = jnp.round(g0).astype(jnp.int32)
    i1 = jnp.round(g1).astype(jnp.int32)
    i2 = jnp.round(g2).astype(jnp.int32)
    in_bounds = (i0 >= 0) & (i0 < N0) & (i1 >= 0) & (i1 < N1) & (i2 >= 0) & (i2 < N2)
    ci0 = jnp.clip(i0, 0, N0 - 1)
    ci1 = jnp.clip(i1, 0, N1 - 1)
    ci2 = jnp.clip(i2, 0, N2 - 1)
    val = vol[ci0, ci1, ci2]
    return jnp.where(in_bounds, val, jnp.zeros_like(val))


def _nearest_gather_half(half_vol, g0, g1, g2, N0, N1, N2):
    """Nearest-neighbor interpolation from a half (kz ≥ 0) volume."""
    ic2 = N2 // 2
    N2_half = ic2 + 1

    i0 = jnp.round(g0).astype(jnp.int32)
    i1 = jnp.round(g1).astype(jnp.int32)
    i2 = jnp.round(g2).astype(jnp.int32)

    kz = i2 - ic2
    # For even N2, kz=-ic2 is Nyquist (self-conjugate) — map directly.
    # For odd N2, kz=-ic2 is a regular negative freq that needs fold.
    if N2 % 2 == 0:
        kz_mapped = jnp.where(kz == -ic2, ic2, kz)
    else:
        kz_mapped = kz
    use_neg = kz_mapped < 0
    ri0 = jnp.where(use_neg, (N0 - (N0 & 1) - i0) % N0, i0)
    ri1 = jnp.where(use_neg, (N1 - (N1 & 1) - i1) % N1, i1)
    rkz = jnp.where(use_neg, -kz_mapped, kz_mapped)

    in_bounds = (i0 >= 0) & (i0 < N0) & (i1 >= 0) & (i1 < N1) & (i2 >= 0) & (i2 < N2) & (rkz >= 0) & (rkz < N2_half)

    ci0 = jnp.clip(ri0, 0, N0 - 1)
    ci1 = jnp.clip(ri1, 0, N1 - 1)
    ckz = jnp.clip(rkz, 0, N2_half - 1)
    val = half_vol[ci0, ci1, ckz]
    val = jnp.where(use_neg, jnp.conj(val), val)
    return jnp.where(in_bounds, val, jnp.zeros_like(val))


# ---------------------------------------------------------------------------
# Forward projection (Fourier slice extraction)
# ---------------------------------------------------------------------------


def _project_one_image(vol, rotation_matrix, pixel_freqs, center, N0, N1, N2, order, half_volume, max_r2):
    """Project one image from a volume using one rotation matrix.

    Args:
        vol: ``(N0, N1, N2)`` for full, ``(N0, N1, N2//2+1)`` for half
        rotation_matrix: ``(3, 3)``
        pixel_freqs: ``(n_pixels, 2)`` with ``(k0, k1)``
        center: ``(3,)`` volume center = ``[N0//2, N1//2, N2//2]``
        max_r2: squared max radius for clipping (or -1 for no clip)
    """
    # Rotate 2D plane coords to 3D
    rk = _rotate_plane_coords(pixel_freqs, rotation_matrix)  # (n_pix, 3)
    rk0, rk1, rk2 = rk[:, 0], rk[:, 1], rk[:, 2]

    # Radius clipping — use pre-rotation 2D norm (exact for integer freqs).
    # Must match backproject's clipping to preserve the adjoint relationship.
    if max_r2 is not None:
        r2 = jnp.sum(pixel_freqs**2, axis=-1)
        in_radius = r2 <= max_r2
    else:
        in_radius = None

    # Grid coordinates (centered)
    g0 = rk0 + center[0]
    g1 = rk1 + center[1]
    g2 = rk2 + center[2]

    # Interpolation
    if half_volume:
        if order == 1:
            vals = _trilinear_gather_half(vol, g0, g1, g2, N0, N1, N2)
        else:
            vals = _nearest_gather_half(vol, g0, g1, g2, N0, N1, N2)
    else:
        if order == 1:
            vals = _trilinear_gather_full(vol, g0, g1, g2, N0, N1, N2)
        else:
            vals = _nearest_gather_full(vol, g0, g1, g2, N0, N1, N2)

    # Apply radius mask
    if in_radius is not None:
        vals = jnp.where(in_radius, vals, jnp.zeros_like(vals))

    return vals


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def project(vol, rotations, image_shape, volume_shape, order=1, half_volume=False, half_image=False, max_r=None):
    """RELION-style forward projection (Fourier slice extraction).

    Projects a 3D Fourier volume onto 2D Fourier images via central-slice
    interpolation, matching RELION's trilinear (order=1) or nearest-neighbor
    (order=0) algorithm.

    Args:
        vol: complex volume, flat ``(N0*N1*N2,)`` or ``(N0*N1*(N2//2+1),)``
        rotations: ``(n_images, 3, 3)`` rotation matrices
        image_shape: ``(H, W)`` — full real-space image dimensions
        volume_shape: ``(N0, N1, N2)`` — full volume dimensions
        order: 0 (nearest) or 1 (trilinear)
        half_volume: if True, vol is kz ≥ 0 half-volume
        half_image: if True, output is rfft-packed ``(n, H*(W//2+1))``
        max_r: optional spherical frequency cutoff (in grid units)

    Returns:
        Images: ``(n_images, n_pixels)`` complex
    """
    N0, N1, N2 = volume_shape
    H, W = image_shape

    if half_volume:
        vol_shape = (N0, N1, N2 // 2 + 1)
    else:
        vol_shape = volume_shape

    vol = jnp.asarray(vol).reshape(vol_shape)
    rotations = jnp.asarray(rotations)

    pixel_freqs = _pixel_frequencies(image_shape, half_image=half_image)
    center = jnp.array([N0 // 2, N1 // 2, N2 // 2], dtype=jnp.float32)
    max_r2 = float(max_r * max_r) if max_r is not None else None

    # vmap over images
    def _proj_one(rot):
        return _project_one_image(
            vol,
            rot,
            pixel_freqs,
            center,
            N0,
            N1,
            N2,
            order,
            half_volume,
            max_r2,
        )

    return jax.vmap(_proj_one)(rotations)


# ---------------------------------------------------------------------------
# Backprojection (adjoint of forward projection)
# ---------------------------------------------------------------------------


def _scatter_trilinear_full(n_voxels, g0, g1, g2, vals, N0, N1, N2):
    """Trilinear scatter-add to a full volume.

    Matches RELION's ``cuda_kernel_backproject3D``: for each pixel, compute
    8 trilinear weights and atomicAdd to the 8 neighbors.

    Args:
        n_voxels: total number of voxels = N0*N1*N2
        g0, g1, g2: grid coordinates, shape ``(n,)``
        vals: complex values to scatter, shape ``(n,)``

    Returns:
        Flat volume ``(n_voxels,)`` with accumulated values.
    """
    b0 = jnp.floor(g0).astype(jnp.int32)
    b1 = jnp.floor(g1).astype(jnp.int32)
    b2 = jnp.floor(g2).astype(jnp.int32)
    # Cast fractions to float32 to avoid float64 × complex64 → complex128 promotion
    # when grid coords are float64 (from float64 rotations).  Float32 fractions
    # are in [0, 1) and have ample precision for interpolation weights.
    wt = jnp.float32
    fx = (g0 - b0.astype(g0.dtype)).astype(wt)
    fy = (g1 - b1.astype(g1.dtype)).astype(wt)
    fz = (g2 - b2.astype(g2.dtype)).astype(wt)

    # Fractional weights per axis: d=0 → (1-f), d=1 → f
    mfx = 1 - fx
    mfy = 1 - fy
    mfz = 1 - fz

    vol = jnp.zeros(n_voxels, dtype=vals.dtype)

    stride1 = N2
    stride0 = N1 * N2

    # For each of the 8 neighbors, compute flat index and scatter
    for d0 in range(2):
        for d1 in range(2):
            for d2 in range(2):
                j0 = b0 + d0
                j1 = b1 + d1
                j2 = b2 + d2

                in_bounds = (j0 >= 0) & (j0 < N0) & (j1 >= 0) & (j1 < N1) & (j2 >= 0) & (j2 < N2)

                flat_idx = j0 * stride0 + j1 * stride1 + j2
                flat_idx = jnp.where(in_bounds, flat_idx, 0)

                # Weight = product of per-axis weights
                w = [mfx, fx][d0] * [mfy, fy][d1] * [mfz, fz][d2]
                weighted = jnp.where(in_bounds, w * vals, jnp.zeros_like(vals))
                vol = vol.at[flat_idx].add(weighted)

    return vol


def _scatter_trilinear_half(n_voxels_half, g0, g1, g2, vals, N0, N1, N2):
    """Trilinear scatter-add to a half (kz ≥ 0) volume.

    Per-neighbor Hermitian fold: for each of the 8 trilinear neighbors,
    if kz < 0, fold to Hermitian partner and conjugate the value.
    This matches RELION's backprojection into the asymmetric half.
    """
    ic2 = N2 // 2
    N2_half = ic2 + 1

    b0 = jnp.floor(g0).astype(jnp.int32)
    b1 = jnp.floor(g1).astype(jnp.int32)
    b2 = jnp.floor(g2).astype(jnp.int32)
    # Cast fractions to float32 to avoid float64 × complex64 → complex128 promotion
    wt = jnp.float32
    fx = (g0 - b0.astype(g0.dtype)).astype(wt)
    fy = (g1 - b1.astype(g1.dtype)).astype(wt)
    fz = (g2 - b2.astype(g2.dtype)).astype(wt)

    mfx, mfy, mfz = 1 - fx, 1 - fy, 1 - fz

    stride1 = N2_half
    stride0 = N1 * N2_half
    vol = jnp.zeros(n_voxels_half, dtype=vals.dtype)

    for d0 in range(2):
        for d1 in range(2):
            for d2 in range(2):
                j0 = b0 + d0
                j1 = b1 + d1
                j2 = b2 + d2

                in_bounds = (j0 >= 0) & (j0 < N0) & (j1 >= 0) & (j1 < N1) & (j2 >= 0) & (j2 < N2)

                kz = j2 - ic2
                # For even N2, kz=-ic2 is Nyquist (self-conjugate) — map directly.
                # For odd N2, kz=-ic2 is a regular negative freq that needs fold.
                if N2 % 2 == 0:
                    kz_mapped = jnp.where(kz == -ic2, ic2, kz)
                else:
                    kz_mapped = kz
                use_neg = kz_mapped < 0
                sj0 = jnp.where(use_neg, (N0 - (N0 & 1) - j0) % N0, j0)
                sj1 = jnp.where(use_neg, (N1 - (N1 & 1) - j1) % N1, j1)
                hkz = jnp.where(use_neg, -kz_mapped, kz_mapped)

                in_bounds = in_bounds & (hkz >= 0) & (hkz < N2_half)
                flat_idx = sj0 * stride0 + sj1 * stride1 + hkz
                flat_idx = jnp.where(in_bounds, flat_idx, 0)

                w = [mfx, fx][d0] * [mfy, fy][d1] * [mfz, fz][d2]
                wv = w * vals
                # Conjugate for negative kz
                wv = jnp.where(use_neg, jnp.conj(wv), wv)
                weighted = jnp.where(in_bounds, wv, jnp.zeros_like(wv))
                vol = vol.at[flat_idx].add(weighted)

    return vol


def _scatter_nearest_full(n_voxels, g0, g1, g2, vals, N0, N1, N2):
    """Nearest-neighbor scatter-add to a full volume."""
    i0 = jnp.round(g0).astype(jnp.int32)
    i1 = jnp.round(g1).astype(jnp.int32)
    i2 = jnp.round(g2).astype(jnp.int32)
    in_bounds = (i0 >= 0) & (i0 < N0) & (i1 >= 0) & (i1 < N1) & (i2 >= 0) & (i2 < N2)
    flat_idx = i0 * (N1 * N2) + i1 * N2 + i2
    flat_idx = jnp.where(in_bounds, flat_idx, 0)
    weighted = jnp.where(in_bounds, vals, jnp.zeros_like(vals))
    return jnp.zeros(n_voxels, dtype=vals.dtype).at[flat_idx].add(weighted)


def _scatter_nearest_half(n_voxels_half, g0, g1, g2, vals, N0, N1, N2):
    """Nearest-neighbor scatter-add to a half volume with Hermitian fold."""
    ic2 = N2 // 2
    N2_half = ic2 + 1

    i0 = jnp.round(g0).astype(jnp.int32)
    i1 = jnp.round(g1).astype(jnp.int32)
    i2 = jnp.round(g2).astype(jnp.int32)

    kz = i2 - ic2
    # For even N2, kz=-ic2 is Nyquist (self-conjugate) — map directly.
    # For odd N2, kz=-ic2 is a regular negative freq that needs fold.
    if N2 % 2 == 0:
        kz_mapped = jnp.where(kz == -ic2, ic2, kz)
    else:
        kz_mapped = kz
    use_neg = kz_mapped < 0
    sj0 = jnp.where(use_neg, (N0 - (N0 & 1) - i0) % N0, i0)
    sj1 = jnp.where(use_neg, (N1 - (N1 & 1) - i1) % N1, i1)
    hkz = jnp.where(use_neg, -kz_mapped, kz_mapped)

    in_bounds = (i0 >= 0) & (i0 < N0) & (i1 >= 0) & (i1 < N1) & (i2 >= 0) & (i2 < N2) & (hkz >= 0) & (hkz < N2_half)
    flat_idx = sj0 * (N1 * N2_half) + sj1 * N2_half + hkz
    flat_idx = jnp.where(in_bounds, flat_idx, 0)
    sv = jnp.where(use_neg, jnp.conj(vals), vals)
    weighted = jnp.where(in_bounds, sv, jnp.zeros_like(sv))
    return jnp.zeros(n_voxels_half, dtype=vals.dtype).at[flat_idx].add(weighted)


def _backproject_one_image(
    pixel_freqs, rotation_matrix, img_vals, center, N0, N1, N2, order, half_volume, max_r2, n_voxels_out
):
    """Backproject one image into a volume.

    Matches RELION's ``cuda_kernel_backproject3D``: for each pixel, compute
    rotated coordinates and scatter the pixel value to the volume.
    """
    rk = _rotate_plane_coords(pixel_freqs, rotation_matrix)
    rk0, rk1, rk2 = rk[:, 0], rk[:, 1], rk[:, 2]

    # Radius clipping — use pre-rotation 2D norm (exact for integer freqs).
    # Rotation preserves norms, so |rot @ k|² = |k|².  Using the 2D norm
    # avoids float32 rounding in the post-rotation sum, which can differ
    # by ~1 ULP and cause conjugate pairs to be clipped asymmetrically
    # in the half-image path.
    if max_r2 is not None:
        r2 = jnp.sum(pixel_freqs**2, axis=-1)
        in_radius = r2 <= max_r2
        img_vals = jnp.where(in_radius, img_vals, jnp.zeros_like(img_vals))

    g0 = rk0 + center[0]
    g1 = rk1 + center[1]
    g2 = rk2 + center[2]

    if half_volume:
        if order == 1:
            return _scatter_trilinear_half(n_voxels_out, g0, g1, g2, img_vals, N0, N1, N2)
        else:
            return _scatter_nearest_half(n_voxels_out, g0, g1, g2, img_vals, N0, N1, N2)
    else:
        if order == 1:
            return _scatter_trilinear_full(n_voxels_out, g0, g1, g2, img_vals, N0, N1, N2)
        else:
            return _scatter_nearest_full(n_voxels_out, g0, g1, g2, img_vals, N0, N1, N2)


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def backproject(imgs, rotations, image_shape, volume_shape, order=1, half_volume=False, half_image=False, max_r=None):
    """RELION-style backprojection (adjoint of forward projection).

    Scatters 2D Fourier image values back into a 3D Fourier volume using
    trilinear (order=1) or nearest-neighbor (order=0) weights, matching
    RELION's ``cuda_kernel_backproject3D``.

    For half_image inputs: each non-boundary rfft pixel also scatters its
    Hermitian conjugate at negated coordinates (matching the full-image
    behavior for the missing half).

    Args:
        imgs: ``(n_images, n_pixels)`` complex image values
        rotations: ``(n_images, 3, 3)`` rotation matrices
        image_shape: ``(H, W)`` full real-space image dimensions
        volume_shape: ``(N0, N1, N2)`` full volume dimensions
        order: 0 or 1
        half_volume: if True, output is kz ≥ 0 half-volume
        half_image: if True, input images are rfft-packed
        max_r: optional spherical frequency cutoff

    Returns:
        Volume: flat ``(n_voxels,)`` complex
    """
    N0, N1, N2 = volume_shape
    H, W = image_shape

    if half_volume:
        n_voxels = N0 * N1 * (N2 // 2 + 1)
    else:
        n_voxels = N0 * N1 * N2

    imgs = jnp.asarray(imgs)
    rotations = jnp.asarray(rotations)
    pixel_freqs = _pixel_frequencies(image_shape, half_image=half_image)
    center = jnp.array([N0 // 2, N1 // 2, N2 // 2], dtype=jnp.float32)
    max_r2 = float(max_r * max_r) if max_r is not None else None

    # Pre-compute image-independent data for half_image conjugate scatter.
    # These arrays depend only on image_shape, not on the rotation or pixel
    # values, so they are computed once and captured by the vmap closure.
    if half_image:
        W_half = W // 2 + 1
        k1_idx = jnp.arange(W_half)
        is_boundary = (k1_idx == 0) | (k1_idx * 2 == W)
        is_boundary_full = jnp.tile(is_boundary, H)

        # Conjugate frequencies: negate (k_col, k_row) → (-k_col, -k_row)
        # Exception: Nyquist row (k_row = -H/2, even H).  Negating k_row
        # gives +H/2 which is out of the representable range [-H/2, H/2-1],
        # so we keep k_row and only negate k_col: (-k_col, k_row).
        # This matches RELION's CUDA: `crk = rot @ (k0, -k1)` for k0_idx==0.
        conj_freqs = -pixel_freqs
        if H % 2 == 0:
            k0_idx = jnp.arange(H)[:, None].repeat(W_half, axis=1).ravel()
            is_nyquist_row = k0_idx == 0
            nyquist_conj = pixel_freqs * jnp.array([-1.0, 1.0])
            conj_freqs = jnp.where(
                is_nyquist_row[:, None],
                nyquist_conj,
                conj_freqs,
            )

    def _bp_one(rot_and_img):
        rot, img_vals = rot_and_img

        # Primary scatter
        vol = _backproject_one_image(
            pixel_freqs,
            rot,
            img_vals,
            center,
            N0,
            N1,
            N2,
            order,
            half_volume,
            max_r2,
            n_voxels,
        )

        if half_image:
            conj_vals = jnp.conj(img_vals)
            conj_vals = jnp.where(is_boundary_full, jnp.zeros_like(conj_vals), conj_vals)

            # Rotate conjugate frequencies (computed once, used for both radius clip and scatter)
            conj_rk = _rotate_plane_coords(conj_freqs, rot)

            if max_r2 is not None:
                # Use pre-rotation 2D norm to match the primary scatter's
                # radius check and avoid float32 boundary asymmetry.
                conj_r2 = jnp.sum(conj_freqs**2, axis=-1)
                conj_vals = jnp.where(conj_r2 <= max_r2, conj_vals, jnp.zeros_like(conj_vals))

            cg0 = conj_rk[:, 0] + center[0]
            cg1 = conj_rk[:, 1] + center[1]
            cg2 = conj_rk[:, 2] + center[2]

            if half_volume:
                if order == 1:
                    vol = vol + _scatter_trilinear_half(n_voxels, cg0, cg1, cg2, conj_vals, N0, N1, N2)
                else:
                    vol = vol + _scatter_nearest_half(n_voxels, cg0, cg1, cg2, conj_vals, N0, N1, N2)
            else:
                if order == 1:
                    vol = vol + _scatter_trilinear_full(n_voxels, cg0, cg1, cg2, conj_vals, N0, N1, N2)
                else:
                    vol = vol + _scatter_nearest_full(n_voxels, cg0, cg1, cg2, conj_vals, N0, N1, N2)

        return vol

    # Sum contributions from all images
    all_vols = jax.vmap(_bp_one)((rotations, imgs))
    return jnp.sum(all_vols, axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "project",
    "backproject",
]
