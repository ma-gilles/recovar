"""Denovo-init Mavg + sigma2_noise computation (pure NumPy).

Mirrors RELION ``calculateSumOfPowerSpectraAndAverageImage``
(ml_optimiser.cpp:2891) + ``setSigmaNoiseEstimatesAndSetAverageImage``
(ml_optimiser.cpp:3243). For up to N≤1000 particles per optics group:
accumulate Mavg and per-shell radial power; then
``sigma2_noise[g] = sum_sigma2[g] / (2 * sumw[g]) - 0.5*|FFT(Mavg)|²``
with negative shells replaced by their nearest positive neighbour.

Public API: ``compute_avg_unaligned_and_sigma2``.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def _softmask_outside_map(
    image: np.ndarray,
    radius: float,
    cosine_width: float,
) -> np.ndarray:
    """RELION `softMaskOutsideMap` (mask.cpp:43) in NumPy.

    Three radial regions about the image center, with ``radius_p = radius + cosine_width``:
    ``r < radius`` keep image, ``r ∈ [radius, radius_p]`` cosine-blend toward
    background, ``r > radius_p`` set to background. ``bg`` is a weighted average
    over the edge+exterior, computed in a first pass and applied in a second.
    """
    H, W = image.shape[-2:]
    yy = np.arange(H) - H // 2
    xx = np.arange(W) - W // 2
    r = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2).astype(np.float64)

    radius_p = radius + cosine_width

    # Background weight map: 0 inside, raisedcos in edge, 1 outside
    w = np.zeros_like(r)
    outside = r > radius_p
    edge = (r >= radius) & (r <= radius_p)
    w[outside] = 1.0
    if cosine_width > 0 and edge.any():
        w[edge] = 0.5 + 0.5 * np.cos(np.pi * (radius_p - r[edge]) / cosine_width)

    w_sum = w.sum()
    if w_sum > 0:
        avg_bg = float((w * image).sum() / w_sum)
    else:
        avg_bg = 0.0

    out = image.astype(np.float64, copy=True)
    # Outside -> background
    out[outside] = avg_bg
    # Edge -> blend
    if cosine_width > 0 and edge.any():
        raisedcos = w[edge]  # same formula
        out[edge] = (1.0 - raisedcos) * image[edge] + raisedcos * avg_bg
    return out.astype(image.dtype)


def _radial_power_spectrum(
    image_real: np.ndarray,
    n_shells: int,
) -> np.ndarray:
    """Per-shell average of `|FFT(image)|^2`.

    Mirrors the FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM loop at
    ml_optimiser.cpp:3108-3117.

    Uses rfft2 (RELION stores the half-complex layout). Shell index is
    `round(sqrt(k^2 + i^2 + j^2))` with k==0 for 2D images; the j axis
    is already the half-complex direction from rfft2.
    """
    H, W = image_real.shape[-2:]
    # RELION's `transformer.FourierTransform(img(), Faux, false)` without
    # normalisation does a plain forward FFT scaled by H*W at read; the
    # equivalent numpy call is `np.fft.rfft2(image, norm=None) /
    # (H * W)` if we want the "normalized" version. But RELION's
    # `transformer.FourierTransform` with `normalize=false` returns the
    # unnormalised DFT (matches fftw forward convention). The critical
    # point is the *ratio* against Mavg's spectrum — both normalisations
    # must match. RELION's FourierTransform internal normalisation for
    # 2D images (see fftw.cpp:358) divides by N^d on forward, so output
    # is `F_norm = F_fftw / (H*W)`. We mirror that.
    F = np.fft.rfft2(image_real, norm=None) / (H * W)

    # Shell indices for each (i, j) in the rfft2 layout.
    # rfft2 output shape: (H, W//2 + 1). Row frequencies wrap around:
    #    ky = [0, 1, ..., H/2, -H/2+1, ..., -1] (fftfreq without 1/N factor)
    ky = np.fft.fftfreq(H, d=1.0) * H
    kx = np.arange(W // 2 + 1, dtype=np.float64)
    ires = np.round(np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)).astype(np.int64)

    power = F.real**2 + F.imag**2
    out = np.zeros(n_shells, dtype=np.float64)
    count = np.zeros(n_shells, dtype=np.int64)
    # Flatten for np.add.at
    np.add.at(out, ires[ires < n_shells], power[ires < n_shells])
    # Count pixels per shell
    flat_ires = ires.ravel()
    valid = flat_ires < n_shells
    np.add.at(count, flat_ires[valid], 1)
    count[count == 0] = 1
    return out / count


def _fix_negative_sigma2(sigma2: np.ndarray) -> np.ndarray:
    """Replace non-positive shell values with the nearest positive neighbour.

    RELION's pass (ml_optimiser.cpp:3293-3320): if sigma2[n] <= 0, try
    sigma2[n-1]; if that's also <= 0, walk forward until a positive value
    is found; fail if none exists.
    """
    out = sigma2.copy()
    n = out.size
    for i in range(n):
        if out[i] <= 0.0:
            if i - 1 >= 0 and out[i - 1] > 0.0:
                out[i] = out[i - 1]
            else:
                nn = i
                while nn < n - 1:
                    nn += 1
                    if out[nn] > 0.0:
                        out[i] = out[nn]
                        break
                else:
                    raise RuntimeError(f"sigma2_noise[{i}] is non-positive with no positive neighbour")
    return out


def compute_avg_unaligned_and_sigma2(
    image_iter: Iterator[Tuple[int, np.ndarray]],
    *,
    ori_size: int,
    pixel_size: float,
    particle_diameter_ang: float,
    width_mask_edge_px: int,
    do_zero_mask: bool,
    nr_optics_groups: int,
    minimum_nr_particles: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce calculateSumOfPowerSpectra + setSigmaNoiseEstimates.

    Parameters
    ----------
    image_iter
        Iterator yielding `(optics_group_id, real_space_image)` tuples. Each
        image must be `(ori_size, ori_size)`, centred origin (the real-space
        centre pixel at `(ori_size // 2, ori_size // 2)`).
    ori_size, pixel_size, particle_diameter_ang, width_mask_edge_px,
    do_zero_mask
        Particle-mask parameters matching RELION's `softMaskOutsideMap`.
    nr_optics_groups
        Number of optics groups present in the dataset.
    minimum_nr_particles
        Per-group cap on how many particles contribute to sigma2 estimation
        (RELION default 1000 for 2D).

    Returns
    -------
    Mavg : (ori_size, ori_size) real-space average image
    sigma2_per_group : (nr_optics_groups, ori_size // 2 + 1)
    """
    n_shells = ori_size // 2 + 1

    Mavg = np.zeros((ori_size, ori_size), dtype=np.float64)
    sum_sigma2 = np.zeros((nr_optics_groups, n_shells), dtype=np.float64)
    sumw = np.zeros(nr_optics_groups, dtype=np.float64)
    radius_px = particle_diameter_ang / (2.0 * pixel_size)

    # Count per-group to respect the 1000-particle cap
    per_group_done = np.zeros(nr_optics_groups, dtype=np.int64)
    total_target = minimum_nr_particles * nr_optics_groups

    total_done = 0
    for opt_grp, img in image_iter:
        if per_group_done[opt_grp] >= minimum_nr_particles:
            continue
        img = img.astype(np.float64, copy=False)
        if img.shape != (ori_size, ori_size):
            raise ValueError(f"image shape {img.shape} != expected {(ori_size, ori_size)}")

        if do_zero_mask:
            img = _softmask_outside_map(img, radius_px, float(width_mask_edge_px))

        Mavg += img
        ind_spect = _radial_power_spectrum(img, n_shells)
        sum_sigma2[opt_grp] += ind_spect
        sumw[opt_grp] += 1.0
        per_group_done[opt_grp] += 1
        total_done += 1

        if total_done >= total_target:
            break

    total_sum = sumw.sum()
    if total_sum <= 0:
        raise RuntimeError("no particles processed")
    Mavg /= total_sum

    # Power spectrum of the averaged image (divided by 2 for 2-dim complex plane)
    mavg_spect = _radial_power_spectrum(Mavg, n_shells) / 2.0

    sigma2_per_group = np.zeros_like(sum_sigma2)
    for g in range(nr_optics_groups):
        if sumw[g] <= 0:
            continue
        sigma2_per_group[g] = sum_sigma2[g] / (2.0 * sumw[g]) - mavg_spect
        sigma2_per_group[g] = _fix_negative_sigma2(sigma2_per_group[g])

    return Mavg, sigma2_per_group
