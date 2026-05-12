"""Denovo Iref seeding (RELION ``--pad 1`` parity).

Production path is ``compute_bootstrap_iref_via_cpp`` (C++ binding mirrors
``calculateSumOfPowerSpectraAndAverageImage`` ml_optimiser.cpp:3127-3205 +
reconstruct :3265 + ``initialLowPassFilterReferences`` :3336-3372).
Parity target: ``run_it000_class001.mrc`` (|CC|>0.998).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WIDTH_FMASK_EDGE: float = 2.0  # ml_optimiser.h:91


@dataclass
class ParticleCTF:
    """Per-particle CTF + optics-group scalars (voltage kV, Cs mm, Q0, angpix Å)."""

    defU: float
    defV: float
    defAngle: float
    phase_shift: float = 0.0
    voltage: float = 300.0
    Cs: float = 2.7
    Q0: float = 0.07
    angpix: float = 8.5
    ori_size: int = 64


def reorder_particles_relion_style(
    main_star,
    images: np.ndarray,
    defU: np.ndarray,
    defV: np.ndarray,
    defAngle: np.ndarray,
    phase_shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stable-sort by ``_rlnMicrographName`` and re-index by stack frame (matches ``Experiment::read``)."""
    img_names = main_star["_rlnImageName"].tolist()
    mic_names = main_star["_rlnMicrographName"].tolist()
    order = sorted(range(len(mic_names)), key=lambda i: mic_names[i])
    frame_ids = [int(img_names[i].split("@")[0]) - 1 for i in order]
    return (
        np.ascontiguousarray(images[frame_ids]),
        np.ascontiguousarray(defU[order]),
        np.ascontiguousarray(defV[order]),
        np.ascontiguousarray(defAngle[order]),
        np.ascontiguousarray(phase_shift[order]),
    )


def compute_bootstrap_iref_via_cpp(
    *,
    images: np.ndarray,
    defU: np.ndarray,
    defV: np.ndarray,
    defAngle: np.ndarray,
    phase_shift: np.ndarray,
    voltage: float,
    Cs: float,
    Q0: float,
    pixel_size: float,
    ori_size: int,
    nr_classes: int,
    particle_diameter_ang: float,
    width_mask_edge_px: float,
    do_zero_mask: bool,
    do_ctf_correction: bool,
    random_seed: int,
    padding_factor: int = 1,
    current_size: int = -1,
    minimum_nr_particles: int = 1000,
) -> np.ndarray:
    """Run the full RELION InitialModel bootstrap in C++; returns Iref in recovar frame."""
    from recovar.relion_bind import _relion_bind_core as bind
    from recovar.utils.helpers import relion_volume_to_recovar

    if current_size <= 0:
        # RELION wsum_model.current_size = ROUND(0.07 * ori_size) (shell count, not Å).
        current_size = int(np.floor(0.07 * ori_size + 0.5))

    iref_relion = np.asarray(
        bind.vdam_bootstrap_iref(
            np.ascontiguousarray(images.astype(np.float64)),
            np.ascontiguousarray(defU.astype(np.float64)),
            np.ascontiguousarray(defV.astype(np.float64)),
            np.ascontiguousarray(defAngle.astype(np.float64)),
            np.ascontiguousarray(phase_shift.astype(np.float64)),
            voltage,
            Cs,
            Q0,
            pixel_size,
            ori_size,
            nr_classes,
            particle_diameter_ang,
            width_mask_edge_px,
            do_zero_mask,
            do_ctf_correction,
            random_seed,
            padding_factor,
            1,  # TRILINEAR
            current_size,
            minimum_nr_particles,
        )
    )
    return np.asarray([relion_volume_to_recovar(vol) for vol in iref_relion], dtype=np.float64)


def postprocess_bootstrap_iref_via_cpp(
    Iref: np.ndarray,
    *,
    pixel_size: float,
    ini_high_ang: float,
    particle_diameter_ang: float,
    width_mask_edge_px: float,
    do_init_blobs: bool = True,
    is_helical_segment: bool = False,
) -> np.ndarray:
    """Apply RELION's post-bootstrap blobs+LP+softMask pipeline (ml_optimiser.cpp:2940-2980).

    Call immediately after ``compute_bootstrap_iref_via_cpp`` to preserve RELION's
    global ``rand()`` state for the blob draws.
    """
    from recovar.relion_bind import _relion_bind_core as bind
    from recovar.utils.helpers import recovar_volume_to_relion, relion_volume_to_recovar

    arr = np.asarray(Iref, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[1] != arr.shape[2] or arr.shape[2] != arr.shape[3]:
        raise ValueError(f"Iref must have shape (K, N, N, N), got {arr.shape}")

    iref_relion = np.asarray([recovar_volume_to_relion(vol) for vol in arr], dtype=np.float64)
    post_relion = np.asarray(
        bind.vdam_postprocess_initial_iref(
            np.ascontiguousarray(iref_relion),
            float(pixel_size),
            float(ini_high_ang),
            float(particle_diameter_ang),
            float(width_mask_edge_px),
            bool(do_init_blobs),
            bool(is_helical_segment),
        ),
        dtype=np.float64,
    )
    return np.asarray([relion_volume_to_recovar(vol) for vol in post_relion], dtype=np.float64)


def initial_low_pass_filter_references(
    Iref: np.ndarray,
    *,
    ori_size: int,
    pixel_size: float,
    ini_high_ang: float,
) -> np.ndarray:
    """``initialLowPassFilterReferences`` (ml_optimiser.cpp:3336): cosine-taper from r=radius outward to r=radius_p."""
    radius = ori_size * pixel_size / ini_high_ang - WIDTH_FMASK_EDGE / 2.0
    radius_p = radius + WIDTH_FMASK_EDGE
    N = Iref.shape[1]
    kz = np.fft.fftfreq(N, d=1.0) * N
    kx = np.arange(N // 2 + 1, dtype=np.float64)
    r = np.sqrt(kz[:, None, None] ** 2 + kz[None, :, None] ** 2 + kx[None, None, :] ** 2)
    mask = np.zeros_like(r)
    mask[r < radius] = 1.0
    edge = (r >= radius) & (r <= radius_p)
    if WIDTH_FMASK_EDGE > 0:
        mask[edge] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - r[edge]) / WIDTH_FMASK_EDGE)

    out = np.zeros_like(Iref)
    for k in range(Iref.shape[0]):
        vol = Iref[k]
        F = np.fft.rfftn(vol, axes=(0, 1, 2), norm=None) / vol.size
        out[k] = np.fft.irfftn(F * mask * vol.size, s=vol.shape, axes=(0, 1, 2), norm=None)
    return out
