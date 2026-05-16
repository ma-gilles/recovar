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


def _softmask_outside_map(image: np.ndarray, radius: float, cosine_width: float) -> np.ndarray:
    """RELION ``softMaskOutsideMap`` (mask.cpp:43): cosine taper between ``r=radius`` and ``radius+cosine_width``."""
    H, W = image.shape[-2:]
    yy = np.arange(H) - H // 2
    xx = np.arange(W) - W // 2
    r = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2).astype(np.float64)
    radius_p = radius + cosine_width

    w = np.zeros_like(r)
    outside = r > radius_p
    edge = (r >= radius) & (r <= radius_p)
    w[outside] = 1.0
    if cosine_width > 0 and edge.any():
        w[edge] = 0.5 + 0.5 * np.cos(np.pi * (radius_p - r[edge]) / cosine_width)

    w_sum = w.sum()
    avg_bg = float((w * image).sum() / w_sum) if w_sum > 0 else 0.0

    out = image.astype(np.float64, copy=True)
    out[outside] = avg_bg
    if cosine_width > 0 and edge.any():
        out[edge] = (1.0 - w[edge]) * image[edge] + w[edge] * avg_bg
    return out.astype(image.dtype)


def _radial_power_spectrum(image_real: np.ndarray, n_shells: int) -> np.ndarray:
    """Per-shell mean ``|FFT(image)|²`` (ml_optimiser.cpp:3108-3117); RELION-normalised by ``H*W``."""
    H, W = image_real.shape[-2:]
    F = np.fft.rfft2(image_real, norm=None) / (H * W)
    ky = np.fft.fftfreq(H, d=1.0) * H
    kx = np.arange(W // 2 + 1, dtype=np.float64)
    ires = np.round(np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)).astype(np.int64)
    out = np.zeros(n_shells, dtype=np.float64)
    count = np.zeros(n_shells, dtype=np.int64)
    np.add.at(out, ires[ires < n_shells], (F.real**2 + F.imag**2)[ires < n_shells])
    flat = ires.ravel()
    np.add.at(count, flat[flat < n_shells], 1)
    count[count == 0] = 1
    return out / count


def _fix_negative_sigma2(sigma2: np.ndarray) -> np.ndarray:
    """Replace non-positive shells with the nearest positive neighbour (ml_optimiser.cpp:3293-3320)."""
    out = sigma2.copy()
    n = out.size
    for i in range(n):
        if out[i] > 0.0:
            continue
        if i - 1 >= 0 and out[i - 1] > 0.0:
            out[i] = out[i - 1]
            continue
        for nn in range(i + 1, n):
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
    """``calculateSumOfPowerSpectra`` + ``setSigmaNoiseEstimates`` (per-group cap defaults to RELION's 1000)."""
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
