"""`InitialModelState`: the VDAM state carried through the iteration loop.

Shape conventions (RELION `pad=1`, single-optics-group GUI InitialModel):

    K            number of classes
    N            ori_size (= image box size in voxels)
    S            number of Fourier shells, `N // 2 + 1`
    H            2 if pseudo-halfsets active, else 1
    G            number of optics groups (always 1 for the GUI InitialModel
                 case on the current fixture)

Arrays:

    Iref:          (K, N, N, N)                  real-space references
    Igrad1:        (K*H, N, N, N//2+1) complex   first-moment slots
    Igrad2:        (K, N, N, N//2+1)  complex    second-moment slots
    sigma2_noise:  (G, S)                        per-shell noise power
    tau2_class:    (K, S)                        per-shell prior power
    sigma2_class:  (K, S)                        reconstruction noise estimate
    fsc_halves_class:     (K, S)                 per-shell fsc between mom1 slots
    fourier_coverage_class: (K, S)               per-shell Fourier coverage
    data_vs_prior_class:  (K, S)
    pdf_class:     (K,)                          class mixing weights
    pdf_direction: (K, n_directions)             class-conditional pdf
    sigma2_offset: scalar                         translation-prior variance in A^2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# RELION ml_model.cpp:23
MOM2_INIT_CONSTANT: float = 1.0


@dataclass
class InitialModelState:
    """Mutable NumPy-backed state for VDAM iterations.

    The state is deliberately a plain dataclass (not Equinox) so Phase 3
    tests can run CPU-only without triggering JAX tracing. Phase 4+ wraps
    this behind an Equinox container for GPU execution.
    """

    # Iteration bookkeeping
    iter: int = 0
    nr_iter: int = 200
    K: int = 1
    ori_size: int = 64
    pixel_size: float = 1.0
    pseudo_halfsets: bool = True

    # References / moments
    Iref: np.ndarray = field(default_factory=lambda: np.zeros((1, 64, 64, 64)))
    Igrad1: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 64, 64, 33), dtype=np.complex128)
    )  # K*H when pseudo_halfsets else K
    Igrad2: np.ndarray = field(
        default_factory=lambda: np.full(
            (1, 64, 64, 33), MOM2_INIT_CONSTANT + 1j * MOM2_INIT_CONSTANT, dtype=np.complex128
        )
    )

    # Spectra (per class / optics group)
    sigma2_noise: np.ndarray = field(default_factory=lambda: np.zeros((1, 33)))
    tau2_class: np.ndarray = field(default_factory=lambda: np.zeros((1, 33)))
    sigma2_class: np.ndarray = field(default_factory=lambda: np.zeros((1, 33)))
    fsc_halves_class: np.ndarray = field(default_factory=lambda: np.zeros((1, 33)))
    fourier_coverage_class: np.ndarray = field(default_factory=lambda: np.zeros((1, 33)))
    data_vs_prior_class: np.ndarray = field(default_factory=lambda: np.zeros((1, 33)))

    # Class/direction weights
    pdf_class: np.ndarray = field(default_factory=lambda: np.ones(1))
    pdf_direction: Optional[np.ndarray] = None  # (K, n_directions) — filled in by init
    sigma2_offset: float = 100.0

    # Resolution pointers
    ini_high: float = -1.0
    current_resolution: float = 0.0
    current_resolution_shell: int = 0
    current_size: int = 0
    incr_size: int = 10
    ave_Pmax: float = 0.0
    has_high_fsc_at_limit: bool = False

    # Noise averaged image (from calculateSumOfPowerSpectraAndAverageImage).
    # Kept for reproducibility / debug; not used after iter 0.
    Mavg: Optional[np.ndarray] = None

    # Per-iter subset plan (filled by `select_subset_for_iter`)
    subset_particle_ids: Optional[np.ndarray] = None
    subset_halfset_ids: Optional[np.ndarray] = None

    # Schedule snapshots recorded each iter
    grad_current_stepsize: float = 0.5
    tau2_fudge_factor: float = 1.0
    subset_size: int = -1

    # Convergence flags
    has_converged: bool = False
    grad_has_converged: bool = False


def half_slot_count(K: int, pseudo_halfsets: bool) -> int:
    """Number of first-moment slots per RELION convention
    (K when no halfsets, 2K when pseudo-halfsets)."""
    return 2 * K if pseudo_halfsets else K


def half_slot_index(k: int, h: int, K: int, pseudo_halfsets: bool) -> int:
    """Index into `Igrad1` for class `k`, halfset `h` (0 or 1).

    RELION packs halfset-0 of class k at Igrad1[k] and halfset-1 at
    Igrad1[K + k] (ml_model.cpp:935-937 then ml_optimiser.cpp:4907-5139).
    When pseudo_halfsets is off, h must be 0.
    """
    if not pseudo_halfsets:
        if h != 0:
            raise ValueError("halfset id must be 0 when pseudo_halfsets is off")
        return k
    if h not in (0, 1):
        raise ValueError(f"halfset id must be 0 or 1, got {h}")
    return h * K + k
