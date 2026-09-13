"""Denovo Iref seeding (RELION ``--pad 1`` parity).

Production path is ``compute_bootstrap_iref_via_cpp`` (C++ binding mirrors
``calculateSumOfPowerSpectraAndAverageImage`` ml_optimiser.cpp:3127-3205 +
reconstruct :3265 + ``initialLowPassFilterReferences`` :3336-3372).
Parity target: ``run_it000_class001.mrc`` (|CC|>0.998).
"""

from __future__ import annotations

import numpy as np


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
    particle_seed_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Run the full RELION InitialModel bootstrap in C++; returns Iref in recovar frame."""
    from recovar.relion_bind import _relion_bind_core as bind
    from recovar.utils.helpers import relion_volume_to_recovar

    if current_size <= 0:
        # RELION wsum_model.current_size = ROUND(0.07 * ori_size) (shell count, not Å).
        current_size = int(np.floor(0.07 * ori_size + 0.5))
    seed_ids = None if particle_seed_ids is None else np.ascontiguousarray(particle_seed_ids, dtype=np.int64)

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
            seed_ids,
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
