"""Denovo InitialModel state initialisation (RELION-parity).

Mirrors these RELION entry points:
  - MlModel::initialiseFromImages (fn_ref == "None" branch)  (ml_model.cpp:1082-1133)
  - MlOptimiser::calculateSumOfPowerSpectraAndAverageImage    (ml_optimiser.cpp:2891)
  - MlOptimiser::setSigmaNoiseEstimatesAndSetAverageImage     (ml_optimiser.cpp:3243)
  - MlOptimiser::initialLowPassFilterReferences              (ml_optimiser.cpp:3287+)
  - MlOptimiser `ini_high` fallback for do_average_unaligned (ml_optimiser.cpp:2513-2518)

Two callables:

  - `initialise_denovo_state(...)`
        Fills everything that does not depend on particle data: class
        references (zero), gradient moment slots, pdf_class / pdf_direction
        uniform priors, empty spectra, current_size / current_resolution
        from the 0.07·ori_size rule.

  - `seed_noise_from_avg_unaligned(state, Mavg_per_group, ...)`
        Given the per-optics-group average-unaligned power spectra (computed
        externally — this is a Phase 4 concern, not Phase 3), writes
        sigma2_noise. The handoff note confirms this step already matches
        RELION to ~3e-7 on the fixture, so we just wire the transfer.

Reference data-dependent steps (reading particle stacks, FFTing them) are
out of scope here; callers provide `Mavg_per_group` via the existing
recovar data-io machinery. That lets Phase 3 tests run without touching
the RELION fixture.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .state import (
    MOM2_INIT_CONSTANT,
    InitialModelState,
    half_slot_count,
)

# RELION's 0.07 digital-frequency low-pass rule for do_average_unaligned
# (ml_optimiser.cpp:2513-2518). The exact formula is:
#     ini_high_shell = ROUND(0.07 * ori_size)
#     ini_high_angstrom = ori_size * pixel_size / ini_high_shell
INI_HIGH_DIGITAL_FREQ: float = 0.07


def compute_ini_high_shell(ori_size: int) -> int:
    """Return `ROUND(0.07 * ori_size)` — the initial resolution shell.

    RELION source: ml_optimiser.cpp:2518 `mymodel.getResolution(ROUND(0.07 *
    mymodel.ori_size))`.  RELION's ROUND macro rounds ties away from zero;
    for positive inputs this is floor(x + 0.5).
    """
    x = INI_HIGH_DIGITAL_FREQ * ori_size
    return int(math.floor(x + 0.5))


def compute_ini_high_angstrom(ori_size: int, pixel_size: float) -> float:
    """Return the initial low-pass in Ångström for denovo init.

    `mymodel.getResolution(ires)` returns `ires / (ori_size * pixel_size)`
    (digital frequency in 1/Å). RELION stores `ini_high = 1 / resolution`
    in Ångström (ml_optimiser.cpp:2518).
    """
    ini_shell = compute_ini_high_shell(ori_size)
    return ori_size * pixel_size / ini_shell


def compute_current_size_for_denovo(ori_size: int) -> int:
    """Pre-iter-1 `current_size` for a denovo InitialModel run.

    At `do_average_unaligned=true`, RELION limits the initial reconstruction
    to the first `current_resolution_shell` of the image — see
    `wsum_model.current_size = 1./mymodel.getResolution(ROUND(0.07 *
    mymodel.ori_size))` at ml_optimiser.cpp:2939 then
    `updateImageSizeAndResolutionPointers` expands with incr_size.

    For a 64×8.5 Å fixture: 0.07·64 = 4.48 -> ROUND = 4 -> 2·(4 + 10) = 28.
    That matches the handoff's observed `current_size = 28`.
    """
    ini_shell = compute_ini_high_shell(ori_size)
    # updateImageSizeAndResolutionPointers (ml_optimiser.cpp:5826-5854):
    # maxres = ini_shell (+ optional 25% extension if ave_Pmax > 0.1);
    # at iter 0 / before E-step that gate is false, so just += incr_size.
    # RELION's default incr_size is 10 (set in parseContinue — hard-coded
    # elsewhere in the CLI parser at the same constant).
    incr_size = 10
    current_size = 2 * (ini_shell + incr_size)
    if current_size > ori_size:
        current_size = ori_size
    return current_size


def initialise_denovo_state(
    *,
    ori_size: int,
    pixel_size: float,
    K: int,
    nr_iter: int,
    n_directions: int,
    nr_optics_groups: int = 1,
    pseudo_halfsets: bool = True,
    padding_factor: int = 1,
) -> InitialModelState:
    """Produce a fresh `InitialModelState` equivalent to RELION's denovo
    init at iter 0 (before the first E-step).

    Mirrors ml_model.cpp:1082-1133 (denovo branch of initialiseFromImages).

    Padded Fourier shape of Igrad arrays:
        Nx_pad = ori_size * padding_factor
        Ny_pad = ori_size * padding_factor
        Nz_pad = ori_size * padding_factor  (3D)
        Igrad shape = (Nz_pad, Ny_pad, Nx_pad // 2 + 1)

    At the GUI InitialModel default `padding_factor = 1`.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if n_directions < 1:
        raise ValueError("n_directions must be >= 1")
    if ori_size < 2:
        raise ValueError("ori_size must be >= 2")

    # Fourier box (half-complex, padded to padding_factor)
    pf = padding_factor
    Nz_pad = ori_size * pf
    Ny_pad = ori_size * pf
    Nx_pad_half = (ori_size * pf) // 2 + 1

    # Real-space references: zero (denovo branch, ml_model.cpp:1099-1106)
    Iref = np.zeros((K, ori_size, ori_size, ori_size), dtype=np.float64)

    # Gradient first moments: zero (denovo branch, ml_model.cpp:1117-1125)
    n_slots = half_slot_count(K, pseudo_halfsets)
    Igrad1 = np.zeros((n_slots, Nz_pad, Ny_pad, Nx_pad_half), dtype=np.complex128)

    # Gradient second moments: Complex(MOM2_INIT_CONSTANT, MOM2_INIT_CONSTANT)
    Igrad2 = np.full(
        (K, Nz_pad, Ny_pad, Nx_pad_half),
        MOM2_INIT_CONSTANT + 1j * MOM2_INIT_CONSTANT,
        dtype=np.complex128,
    )

    # Spectra: empty. sigma2_noise is filled by
    # setSigmaNoiseEstimatesAndSetAverageImage (Phase 4).
    n_shells = ori_size // 2 + 1
    sigma2_noise = np.zeros((nr_optics_groups, n_shells), dtype=np.float64)
    tau2_class = np.zeros((K, n_shells), dtype=np.float64)
    fsc_halves_class = np.zeros((K, n_shells), dtype=np.float64)
    data_vs_prior_class = np.zeros((K, n_shells), dtype=np.float64)

    # Class mixing weights: uniform (ml_model.cpp::initialise path)
    pdf_class = np.full(K, 1.0 / K, dtype=np.float64)

    # pdf_direction: uniform 1/(K * n_directions) (ml_model.cpp:1154-1167)
    pdf_direction = np.full((K, n_directions), 1.0 / (K * n_directions), dtype=np.float64)

    # Resolution pointers
    ini_high_A = compute_ini_high_angstrom(ori_size, pixel_size)
    current_resolution = 1.0 / ini_high_A  # digital freq (1/Å)
    current_size = compute_current_size_for_denovo(ori_size)
    current_resolution_shell = compute_ini_high_shell(ori_size)

    return InitialModelState(
        iter=0,
        nr_iter=nr_iter,
        K=K,
        ori_size=ori_size,
        pixel_size=pixel_size,
        pseudo_halfsets=pseudo_halfsets,
        Iref=Iref,
        Igrad1=Igrad1,
        Igrad2=Igrad2,
        sigma2_noise=sigma2_noise,
        tau2_class=tau2_class,
        fsc_halves_class=fsc_halves_class,
        data_vs_prior_class=data_vs_prior_class,
        pdf_class=pdf_class,
        pdf_direction=pdf_direction,
        ini_high=ini_high_A,
        current_resolution=current_resolution,
        current_resolution_shell=current_resolution_shell,
        current_size=current_size,
        Mavg=None,
        subset_particle_ids=None,
        subset_halfset_ids=None,
    )


def seed_noise_from_mavg(
    state: InitialModelState,
    sigma2_per_group: np.ndarray,
) -> InitialModelState:
    """Write `sigma2_noise` from an externally-computed per-group spectrum.

    `sigma2_per_group` must have shape `(nr_optics_groups, ori_size/2 + 1)`.
    The caller is responsible for computing it via RELION's average-unaligned
    recipe (Phase 4 will use the existing recovar data-io to fetch a 500-
    particle batch and FFT it through the `calculateSumOfPowerSpectraAnd
    AverageImage` logic).
    """
    if sigma2_per_group.shape != state.sigma2_noise.shape:
        raise ValueError(f"sigma2_per_group shape {sigma2_per_group.shape} != expected {state.sigma2_noise.shape}")
    new_state = replace(state)
    new_state.sigma2_noise = np.asarray(sigma2_per_group, dtype=np.float64).copy()
    return new_state
