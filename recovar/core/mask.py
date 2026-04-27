"""Real-space and Fourier-space mask generation and manipulation.

Provides masks for the reconstruction pipeline:

- **Volume masks**: ``make_mask`` (standalone, RELION-style),
  ``make_mask_from_half_maps`` (averages half-maps then calls ``make_mask``),
  ``masking_options`` (pipeline entry point),
  ``make_mask_from_gt``, ``make_union_gt_mask``
- **Radial / spherical masks**: ``get_radial_mask``, ``raised_cosine_mask``,
  ``soft_mask_outside_map``
- **Image masks**: ``window_mask``, ``smooth_circular_mask``
- **Softening**: ``soften_volume_mask`` (distance-transform cosine taper)
"""

import logging

import jax.numpy as jnp
import numpy as np
import skimage
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt, label

import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.utils as utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def masking_options(
    volume_mask_option,
    means,
    volume_shape,
    dtype_real=np.float32,
    mask_dilation_iter=0,
    keep_input_mask=False,
    dilated_mask_dilations_iter=None,
):
    """Build volume mask and dilated mask from a pipeline mask specification.

    Args:
        volume_mask_option: One of: path to ``.mrc``, ``"from_halfmaps"``,
            ``"sphere"``, ``"none"``.
        means: Object with ``.corrected0reg`` / ``.corrected1reg`` attributes
            (used for ``from_halfmaps``).
        volume_shape: 3-tuple giving the grid dimensions.
        dtype_real: Output dtype.
        mask_dilation_iter: Extra dilation iterations for the input mask.
        keep_input_mask: If True, use the MRC mask as-is (no softening).
        dilated_mask_dilations_iter: Dilation iterations for the dilated mask.
            Defaults to ``ceil(6 * N / 128)`` where N is the grid size.

    Returns:
        ``(volume_mask, dilated_volume_mask)`` as float arrays in [0, 1].
    """
    if dilated_mask_dilations_iter is None:
        dilated_mask_dilations_iter = int(np.ceil(6 * volume_shape[0] / 128))

    kernel_size = 3

    if not isinstance(volume_mask_option, str):
        raise ValueError("mask option not recognized")

    if volume_mask_option.endswith(".mrc"):
        input_mask = utils.load_mrc(volume_mask_option).astype(np.float32)

        if keep_input_mask:
            volume_mask = input_mask
        else:
            logger.info("Using input mask")
            if mask_dilation_iter > 0:
                logger.info("Thresholding and dilating input mask")
                input_mask = input_mask > 0.99
                input_mask = binary_dilation(input_mask, iterations=mask_dilation_iter)

            if input_mask.shape[0] != volume_shape[0]:
                input_mask = skimage.transform.rescale(input_mask, volume_shape[0] / input_mask.shape[0])

            logger.info("Thresholding mask at 0.5 and softening with cosine kernel of radius %d pixels", kernel_size)
            input_mask = input_mask > 0.5
            volume_mask = soften_volume_mask(input_mask, kernel_size)

        dilated_volume_mask = binary_dilation(input_mask, iterations=dilated_mask_dilations_iter)
        dilated_volume_mask = soften_volume_mask(dilated_volume_mask, kernel_size)

    elif volume_mask_option == "from_halfmaps":
        volume_mask = _mask_from_halfmaps(means, smax=3)
        logger.info("Softening mask")

        dilated_volume_mask = binary_dilation(volume_mask, iterations=dilated_mask_dilations_iter)
        volume_mask = soften_volume_mask(volume_mask, kernel_size)
        dilated_volume_mask = soften_volume_mask(dilated_volume_mask, kernel_size)
        logger.info("using mask computed from mean")

    elif volume_mask_option == "sphere":
        volume_mask = get_radial_mask(volume_shape)
        dilated_volume_mask = get_radial_mask(volume_shape)
        logger.info("using spherical mask")

    elif volume_mask_option == "none":
        volume_mask = np.ones(volume_shape)
        dilated_volume_mask = volume_mask
        logger.info("using no mask")

    else:
        raise ValueError("mask option not recognized. Options: a .mrc path, from_halfmaps, sphere, none")

    return (np.array(volume_mask.astype(dtype_real)), np.array(dilated_volume_mask.astype(dtype_real)))


# ---------------------------------------------------------------------------
# Mask generation from half-maps / ground truth
# ---------------------------------------------------------------------------


def make_mask(volume, *, threshold="auto", lowpass_sigma=None, extend=None,
              soft_edge=3, cleanup=True):
    """Create a solvent mask from a 3-D real-space volume.

    Follows the same conceptual pipeline as RELION's ``relion_mask_create``:
    low-pass filter → threshold → (optional cleanup) → extend → soft cosine edge.

    Key differences from RELION:

    * Gaussian low-pass in real space (RELION uses a Fourier-space raised-cosine).
    * ``"auto"`` threshold via Otsu's method (RELION requires a user-specified
      density threshold, default 0.02 in postprocessing).
    * Optional morphological cleanup (fill holes, keep largest connected
      component) — RELION does not do this.
    * Extension uses ``distance_transform_edt`` for spherical (Euclidean)
      dilation, matching RELION's brute-force Euclidean approach.
    * Soft edge uses ``distance_transform_edt`` + cosine (RELION computes
      min-Euclidean-distance per voxel — mathematically equivalent).

    This function is the building block for :func:`make_mask_from_half_maps`
    and can also be called directly on any volume (e.g. a consensus
    reconstruction loaded from MRC).

    Args:
        volume: 3-D real-space array (e.g. averaged half-maps, or a single
            reconstruction).
        threshold: How to binarize.

            * ``"auto"`` (default): Otsu's method on voxels inside the radial
              mask.
            * A ``float``: fixed density threshold (like RELION's
              ``--ini_threshold``, default 0.02 in postprocessing).
        lowpass_sigma: Gaussian sigma in **voxels** for low-pass smoothing
            before thresholding.  ``None`` = auto (``max(2, N // 64)``).
            Set to ``0`` to disable.
        extend: Extension (dilation) of the binary mask in **voxels**.
            ``None`` = auto (``ceil(6 * N / 128)``).
            (RELION postprocessing default: 3 pixels.)
        soft_edge: Width of the cosine soft edge in **voxels**.
            (RELION postprocessing default: 6 pixels.)
        cleanup: If ``True``, fill holes and keep only the largest connected
            component after thresholding (RELION does not do this).

    Returns:
        Soft mask as a float32 array in [0, 1].

    Example::

        import mrcfile
        from recovar.core.mask import make_mask

        h1 = mrcfile.open("half1.mrc").data
        h2 = mrcfile.open("half2.mrc").data
        avg = (h1 + h2) / 2.0

        # Automatic (recommended)
        mask = make_mask(avg)

        # RELION-like with manual parameters
        mask = make_mask(avg, threshold=0.02, extend=3, soft_edge=6,
                         lowpass_sigma=3, cleanup=False)
    """
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu

    volume = np.asarray(volume, dtype=np.float64)
    N = volume.shape[0]

    # --- 1. Low-pass filter ---
    if lowpass_sigma is None:
        lowpass_sigma = max(2, N // 64)
    if lowpass_sigma > 0:
        filtered = gaussian_filter(volume, sigma=lowpass_sigma)
    else:
        filtered = volume

    # --- 2. Threshold ---
    radial = np.asarray(get_radial_mask(volume.shape)) > 0.5

    if threshold == "auto":
        vals_inside = filtered[radial]
        try:
            thresh_val = threshold_otsu(vals_inside)
        except ValueError:
            logger.warning("Otsu threshold failed, falling back to 99th percentile")
            thresh_val = np.percentile(vals_inside, 99)
    else:
        thresh_val = float(threshold)

    binary = (filtered > thresh_val) & radial
    logger.info("Mask threshold: %.4g  (%.1f%% of voxels inside radial mask)",
                thresh_val, 100.0 * binary.sum() / radial.sum())

    # --- 3. Morphological cleanup ---
    if cleanup:
        filled = binary_fill_holes(binary)
        labeled, n_components = label(filled)
        if n_components > 1:
            component_sizes = np.bincount(labeled.ravel())[1:]
            largest = np.argmax(component_sizes) + 1
            filled = labeled == largest
            logger.info("Mask cleanup: kept largest of %d components", n_components)
        elif n_components == 0:
            logger.warning("No mask voxels found, returning empty mask")
            return np.zeros(volume.shape, dtype=np.float32)
        binary = filled

    # --- 4. Extend (dilate) via Euclidean distance (spherical, like RELION) ---
    if extend is None:
        extend = int(np.ceil(6 * N / 128))
    if extend > 0:
        dist = distance_transform_edt(~np.asarray(binary, dtype=bool))
        binary = dist <= extend

    # --- 5. Soft cosine edge ---
    if soft_edge > 0:
        result = soften_volume_mask(binary, kern_rad=soft_edge)
    else:
        result = np.asarray(binary, dtype=np.float32)

    return np.asarray(result * (result >= 1e-3), dtype=np.float32)


def make_mask_from_half_maps(halfmap1, halfmap2, smax=3, method="auto", **kwargs):
    """Generate a solvent mask from two real-space half-map volumes.

    Averages the two half-maps and delegates to :func:`make_mask`.
    All keyword arguments are forwarded to ``make_mask``.

    For the legacy EMDA local-correlation method, pass
    ``method="local_correlation"``.

    Args:
        halfmap1, halfmap2: Real-space half-map volumes (same shape).
        smax: Kernel radius in pixels (only for ``method="local_correlation"``).
        method: ``"auto"`` or ``"local_correlation"``.
        **kwargs: Forwarded to :func:`make_mask` (``threshold``,
            ``lowpass_sigma``, ``extend``, ``soft_edge``, ``cleanup``).

    Returns:
        Soft mask as a float array in [0, 1].
    """
    if method == "local_correlation":
        soft_edge = kwargs.get("soft_edge", 2)
        return _make_mask_from_half_maps_local_corr(halfmap1, halfmap2, smax=smax,
                                                     soft_edge=soft_edge)

    avg = (np.asarray(halfmap1) + np.asarray(halfmap2)) / 2.0
    return make_mask(avg, **kwargs)


def _make_mask_from_half_maps_local_corr(halfmap1, halfmap2, smax=3, soft_edge=2):
    """Original EMDA-style mask from half-maps via local cross-correlation.

    Kept for backward compatibility and comparison.  See
    ``make_mask_from_half_maps`` with ``method="local_correlation"``.
    """
    dilation_iters = int(6 * halfmap1.shape[0] // 128)
    kern = make_soft_edged_kernel(smax, halfmap1.shape)

    h1 = threshold_map(halfmap1)
    h2 = threshold_map(halfmap2)
    halfcc3d = _local_correlation_3d(h1, h2, kern)
    halfcc3d *= get_radial_mask(halfmap1.shape)

    ccmap_binary = (halfcc3d >= 1e-3).astype(int)
    dilated = binary_dilation(ccmap_binary, iterations=dilation_iters)
    result = soften_volume_mask(dilated, kern_rad=soft_edge)
    return result * (result >= 1e-3)


def make_mask_from_gt(gt_map, smax=3, iter=10, from_ft=True):
    """Generate a mask from a ground-truth volume.

    Args:
        gt_map: Ground-truth volume (Fourier or real space).
        smax: Kernel radius for thresholding.
        iter: Dilation iterations.
        from_ft: If True, ``gt_map`` is in Fourier space.

    Returns:
        Soft mask as a float array in [0, 1].
    """
    vol_shape = utils.guess_vol_shape_from_vol_size(gt_map.size)
    if from_ft:
        vol_real = fourier_transform_utils.get_idft3(gt_map.reshape(vol_shape)).real
    else:
        vol_real = gt_map.reshape(vol_shape)

    thresholded = threshold_map(vol_real) > 0
    dilated = binary_dilation(thresholded, iterations=iter)
    return soften_volume_mask(dilated, kern_rad=2)


def make_union_gt_mask(gt_volumes_real, volume_shape, smax=3, iter=1, dilation_iters=None, kern_rad=3):
    """Create a union mask from multiple ground-truth real-space volumes.

    For each volume, generates a per-volume mask via ``make_mask_from_gt``,
    thresholds at 0.5, then takes the logical OR of all per-volume masks.
    The union is dilated and softened to produce the final mask.

    Args:
        gt_volumes_real: Either a list of 3-D arrays or a 2-D array of shape
            ``(n_vols, n_voxels)`` (reshaped internally to 3-D).
        volume_shape: Tuple giving the 3-D grid dimensions.
        smax: Gaussian kernel radius for ``make_mask_from_gt``.
        iter: Dilation iterations inside ``make_mask_from_gt``.
        dilation_iters: Additional dilation iterations applied to the union
            mask.  Defaults to ``ceil(6 * volume_shape[0] / 128)`` (pipeline
            convention).
        kern_rad: Kernel radius for ``soften_volume_mask``.

    Returns:
        Tuple ``(soft_mask, binary_mask)`` where *soft_mask* is a float array
        in [0, 1] and *binary_mask* is the pre-softening boolean array.
    """
    if dilation_iters is None:
        dilation_iters = int(np.ceil(6 * volume_shape[0] / 128))

    # Normalise input to a list of 3-D arrays
    if isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 2:
        gt_volumes_real = [gt_volumes_real[i].reshape(volume_shape) for i in range(gt_volumes_real.shape[0])]
    elif isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 4:
        gt_volumes_real = [gt_volumes_real[i] for i in range(gt_volumes_real.shape[0])]
    elif isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 3:
        gt_volumes_real = [gt_volumes_real]

    union_mask = np.zeros(volume_shape, dtype=bool)
    for vol in gt_volumes_real:
        vol_3d = np.asarray(vol).reshape(volume_shape)
        per_vol_mask = make_mask_from_gt(vol_3d, smax=smax, iter=iter, from_ft=False)
        union_mask |= per_vol_mask > 0.5

    dilated = binary_dilation(union_mask, iterations=dilation_iters)
    binary_mask = np.asarray(dilated, dtype=bool)
    soft_mask = soften_volume_mask(binary_mask, kern_rad=kern_rad)

    return np.asarray(soft_mask, dtype=np.float32), binary_mask


# ---------------------------------------------------------------------------
# Mask softening
# ---------------------------------------------------------------------------


def soften_volume_mask(binary_volume_mask, kern_rad=3):
    """Soften a binary mask with a cosine taper based on distance transform.

    Voxels inside the mask get value 1; voxels within ``kern_rad`` of the
    boundary get a raised-cosine transition; voxels further out get 0.

    Args:
        binary_volume_mask: Binary mask (values near 0 or 1).
        kern_rad: Width of the cosine transition in voxels.

    Returns:
        Soft mask as float32 array in [0, 1].
    """
    distance_to_mask = distance_transform_edt(binary_volume_mask < 0.9)
    mask = np.zeros_like(binary_volume_mask)
    mask = np.where(
        (distance_to_mask >= 0) & (distance_to_mask < kern_rad),
        0.5 + 0.5 * np.cos(np.pi * distance_to_mask / kern_rad),
        mask,
    )
    return np.asarray(mask.astype(np.float32))


# ---------------------------------------------------------------------------
# Radial / spherical masks
# ---------------------------------------------------------------------------


def get_radial_mask(shape, radius=None):
    """Binary spherical mask.

    Args:
        shape: Volume or image shape tuple.
        radius: Mask radius in voxels. Defaults to ``shape[0] // 2 - 1``.

    Returns:
        Boolean array with True inside the sphere.
    """
    radius = shape[0] // 2 - 1 if radius is None else radius
    volume_coords = fourier_transform_utils.get_k_coordinate_of_each_pixel(shape, voxel_size=1, scaled=False).reshape(
        list(shape) + [len(list(shape))]
    )
    return jnp.linalg.norm(volume_coords, axis=-1) < radius + 1e-7


def raised_cosine_mask(volume_shape, radius, radius_p, offset):
    """3D raised-cosine mask with adjustable center.

    Value is 1 for ``r < radius``, cosine-tapered for ``radius <= r < radius_p``,
    and 0 beyond.

    Args:
        volume_shape: 3-tuple of grid dimensions.
        radius: Inner radius (full value).
        radius_p: Outer radius (zero value).
        offset: 3-element center offset.

    Returns:
        Mask array of shape ``volume_shape``.
    """
    grid = fourier_transform_utils.get_k_coordinate_of_each_pixel_3d(volume_shape, voxel_size=1, scaled=False)
    grid -= offset
    distances = jnp.linalg.norm(grid, axis=-1)

    mask = jnp.where(distances < radius, 1, 0)
    mask = jnp.where(
        (distances >= radius) & (distances < radius_p),
        0.5 - 0.5 * jnp.cos(np.pi * (radius_p - distances) / (radius_p - radius)),
        mask,
    )
    return mask.reshape(volume_shape)


def soft_mask_outside_map(vol, radius=-1, cosine_width=3, Mnoise=None):
    """Soft mask outside map, adapted from RELION.

    Applies a raised-cosine mask to ``vol``, replacing voxels outside the
    mask with the mean background value (or ``Mnoise`` if provided).

    Args:
        vol: Input volume (JAX or numpy array).
        radius: Mask radius in voxels. Defaults to half the box size.
        cosine_width: Width of the cosine transition band.
        Mnoise: Optional replacement value for masked-out regions.

    Returns:
        ``(masked_vol, mask)`` tuple.
    """
    vol = jnp.asarray(vol)
    if radius < 0:
        radius = np.max(np.array(vol.shape) // 2)

    radius_p = radius + cosine_width
    shape = vol.shape

    volume_coords = fourier_transform_utils.get_k_coordinate_of_each_pixel(shape, voxel_size=1, scaled=False).reshape(
        list(shape) + [len(list(shape))]
    )
    r = jnp.linalg.norm(volume_coords, axis=-1)

    mask1 = r < radius
    mask2 = (r >= radius) & (r <= radius_p)
    mask3 = r > radius_p
    raised_cos = 0.5 + 0.5 * jnp.cos(jnp.pi * (radius_p - r) / cosine_width)

    mask = jnp.zeros_like(vol).real
    mask = jnp.where(mask1, 1, mask)
    mask = jnp.where(mask2, 1 - raised_cos, mask)
    background_weight = jnp.zeros_like(mask)
    background_weight = jnp.where(mask3, 1, background_weight)
    background_weight = jnp.where(mask2, raised_cos, background_weight)

    if Mnoise is None:
        sum_bg = jnp.sum(vol * background_weight)
        mask_sum = jnp.sum(background_weight)
        avg_bg = sum_bg / mask_sum
    else:
        Mnoise = jnp.asarray(Mnoise)
        avg_bg = None

    if Mnoise is None:
        add = avg_bg
    else:
        add = Mnoise

    vol = mask * vol + background_weight * add
    return vol, mask


# ---------------------------------------------------------------------------
# 2D image masks
# ---------------------------------------------------------------------------


def window_mask(D, in_rad, out_rad):
    """2D circular window mask with linear taper (normalised coordinates).

    Args:
        D: Image size in pixels (must be even).
        in_rad: Inner radius in normalised coordinates (1.0 = edge).
        out_rad: Outer radius in normalised coordinates.

    Returns:
        Float32 mask of shape ``(D, D)`` with values in [0, 1].
    """
    if D % 2 != 0:
        raise ValueError(f"D must be even, got {D}")
    x0, x1 = np.meshgrid(
        np.linspace(-1, 1, D, endpoint=False, dtype=np.float32),
        np.linspace(-1, 1, D, endpoint=False, dtype=np.float32),
    )
    r = (x0**2 + x1**2) ** 0.5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r - in_rad) / (out_rad - in_rad)))
    return mask.astype(np.float32)


def smooth_circular_mask(image_size, radius, thickness):
    """2D circular mask with a raised-cosine transition band.

    Values are 1 inside ``radius``, 0 outside ``radius + thickness``,
    and follow a cosine taper in between.

    Args:
        image_size: Image size in pixels.
        radius: Inner radius in pixels.
        thickness: Width of the cosine transition in pixels.

    Returns:
        Float mask of shape ``(image_size, image_size)``.
    """
    half = image_size // 2
    coords = np.arange(-half, image_size - half, dtype=float)
    gx, gy = np.meshgrid(coords, coords, indexing="xy")
    r = np.sqrt(gx**2 + gy**2)
    band = (r >= radius) & (r <= radius + thickness)
    mask = np.zeros((image_size, image_size))
    mask[r < radius] = 1.0
    mask[band] = 0.5 + 0.5 * np.cos(np.pi * (r[band] - radius) / thickness)
    return mask


def relion_soft_image_mask(image_size, pixel_size, particle_diameter_ang, width_mask_edge_px):
    """RELION-style 2D soft circular mask for experimental-image scoring.

    RELION applies ``softMaskOutsideMap`` in real space with:

    - radius = ``particle_diameter_ang / (2 * pixel_size)``
    - cosine width = ``width_mask_edge_px``

    This helper mirrors that convention and returns a mask in image-space
    pixels, ready to multiply against centered particle images before the DFT.
    """
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if pixel_size <= 0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}")
    if particle_diameter_ang <= 0:
        raise ValueError(
            f"particle_diameter_ang must be positive, got {particle_diameter_ang}"
        )
    if width_mask_edge_px < 0:
        raise ValueError(
            f"width_mask_edge_px must be non-negative, got {width_mask_edge_px}"
        )

    radius_px = float(particle_diameter_ang) / (2.0 * float(pixel_size))
    radius_px = min(radius_px, image_size / 2.0)
    thickness_px = float(width_mask_edge_px)

    if thickness_px == 0.0:
        thickness_px = 1e-6

    return smooth_circular_mask(
        image_size=image_size,
        radius=radius_px,
        thickness=thickness_px,
    ).astype(np.float32)


def apply_relion_soft_image_mask(images, image_mask):
    """Apply RELION's soft-mask background-fill image semantics."""

    image_mask_arr = jnp.asarray(image_mask)
    images_arr = jnp.asarray(images)

    if image_mask_arr.ndim != 2:
        raise ValueError(f"image_mask must be 2D, got shape {image_mask_arr.shape}")
    if images_arr.ndim not in (2, 3):
        raise ValueError(f"images must be 2D or 3D, got shape {images_arr.shape}")
    if images_arr.shape[-2:] != image_mask_arr.shape:
        raise ValueError(
            f"image_mask shape {image_mask_arr.shape} must match trailing image shape {images_arr.shape[-2:]}"
        )

    squeeze = images_arr.ndim == 2
    images_3d = images_arr[None, ...] if squeeze else images_arr

    mask64 = image_mask_arr.astype(jnp.float64)
    bg_weights = 1.0 - mask64
    bg_weight_sum = jnp.sum(bg_weights, dtype=jnp.float64)
    safe_bg_weight_sum = jnp.where(bg_weight_sum > 0.0, bg_weight_sum, 1.0)
    images64 = images_3d.astype(jnp.float64)
    avg_bg = jnp.tensordot(images64, bg_weights, axes=((-2, -1), (0, 1))) / safe_bg_weight_sum
    result = images64 * mask64[None, :, :] + avg_bg[:, None, None] * bg_weights[None, :, :]

    result = result.astype(images_arr.dtype)
    if squeeze:
        result = result[0]

    return np.asarray(result) if isinstance(images, np.ndarray) else result


# ---------------------------------------------------------------------------
# Internal helpers (adapted from EMDA - https://emda.readthedocs.io/)
# ---------------------------------------------------------------------------


def _mask_from_halfmaps(means, smax=3, method="auto"):
    """Generate mask from pipeline means object (Fourier-space half-maps)."""
    vol_shape = utils.guess_vol_shape_from_vol_size(means.corrected0.size)
    halfmap1 = fourier_transform_utils.get_idft3(means.corrected0reg.reshape(vol_shape)).real
    halfmap2 = fourier_transform_utils.get_idft3(means.corrected1reg.reshape(vol_shape)).real
    return make_mask_from_half_maps(halfmap1, halfmap2, smax=smax, method=method)


def make_soft_edged_kernel(r1, shape):
    """Create a soft-edged convolution kernel for local correlation.

    Adapted from EMDA (https://gitlab.com/ccpem/emda).

    Args:
        r1: Kernel radius in pixels.
        shape: Volume shape for coordinate generation.

    Returns:
        Normalised soft-edged kernel.
    """
    if r1 < 3:
        boxsize = 5
    else:
        boxsize = 2 * r1 + 1

    volume_coords = (
        fourier_transform_utils.get_k_coordinate_of_each_pixel(shape, voxel_size=1, scaled=False).reshape(
            list(shape) + [len(list(shape))]
        )
        + 1
    )
    distances = jnp.linalg.norm(volume_coords, axis=-1)

    half_boxsize = boxsize // 2
    r1 = half_boxsize
    r0 = r1 - 2

    kern = jnp.where(distances < r0, 1.0, 0.0)
    kern = jnp.where(
        (distances <= r1) & (distances >= r0),
        (1 + jnp.cos(jnp.pi * (distances - r0) / (r1 - r0))) / 2.0,
        kern,
    )
    return kern / jnp.sum(kern)


def threshold_map(arr, prob=0.99, dthresh=None):
    """Zero out voxels below the ``prob`` percentile threshold."""
    if dthresh is None:
        X2 = np.sort(arr.flatten())
        F2 = np.arange(len(X2)) / float(len(X2) - 1)
        loc = np.where(F2 >= prob)
        thresh = X2[loc[0][0]]
    else:
        thresh = dthresh
    return arr * (arr > thresh)


def _local_correlation_3d(half1, half2, kern):
    """3D real-space local correlation, adapted from EMDA realsp_local.py."""
    import scipy.signal

    loc3_A = scipy.signal.fftconvolve(half1, kern, "same")
    loc3_A2 = scipy.signal.fftconvolve(half1 * half1, kern, "same")
    loc3_B = scipy.signal.fftconvolve(half2, kern, "same")
    loc3_B2 = scipy.signal.fftconvolve(half2 * half2, kern, "same")
    loc3_AB = scipy.signal.fftconvolve(half1 * half2, kern, "same")

    cov3_AB = loc3_AB - loc3_A * loc3_B
    var3_A = loc3_A2 - loc3_A**2
    var3_B = loc3_B2 - loc3_B**2

    reg_a = np.max(var3_A) / 1000
    reg_b = np.max(var3_B) / 1000
    var3_A = np.where(var3_A < reg_a, reg_a, var3_A)
    var3_B = np.where(var3_B < reg_b, reg_b, var3_B)
    return cov3_AB / np.sqrt(var3_A * var3_B)


def make_moving_gt_mask(gt_volumes_real, volume_shape, smax=3, iter=1, dilation_iters=None, kern_rad=3):
    """Create a mask for the moving region across GT volumes."""
    if dilation_iters is None:
        dilation_iters = int(np.ceil(6 * volume_shape[0] / 128))

    if isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 2:
        gt_volumes_real = [gt_volumes_real[i].reshape(volume_shape) for i in range(gt_volumes_real.shape[0])]
    elif isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 4:
        gt_volumes_real = [gt_volumes_real[i] for i in range(gt_volumes_real.shape[0])]
    elif isinstance(gt_volumes_real, np.ndarray) and gt_volumes_real.ndim == 3:
        gt_volumes_real = [gt_volumes_real]

    if len(gt_volumes_real) == 0:
        raise ValueError("gt_volumes_real must contain at least one volume")

    volumes = np.asarray(
        [np.asarray(vol).reshape(volume_shape) for vol in gt_volumes_real],
        dtype=np.float32,
    )
    mean_volume = np.mean(volumes, axis=0)
    moving_signal = np.sqrt(np.mean((volumes - mean_volume[None]) ** 2, axis=0))

    moving_mask = (
        make_mask_from_gt(
            moving_signal,
            smax=smax,
            iter=iter,
            from_ft=False,
        )
        > 0.5
    )
    if dilation_iters > 0 and np.any(moving_mask):
        moving_mask = binary_dilation(moving_mask, iterations=dilation_iters)

    binary_mask = np.asarray(moving_mask, dtype=bool)
    if np.any(binary_mask):
        soft_mask = soften_volume_mask(binary_mask, kern_rad=kern_rad)
    else:
        soft_mask = np.zeros(volume_shape, dtype=np.float32)

    return np.asarray(soft_mask, dtype=np.float32), binary_mask
