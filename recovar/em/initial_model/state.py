"""VDAM ``InitialModelState`` carried through the iteration loop.

Shape conventions (RELION ``pad=1``, single-optics GUI InitialModel):
``K`` classes, ``N=ori_size``, ``S=N//2+1`` Fourier shells, ``H=2`` if
pseudo-halfsets else 1, ``G`` optics groups (1 on the fixture).

Per-class spectra: tau2_class, sigma2_class:  (K, S);
fourier_coverage_class: (K, S); data_vs_prior_class: (K, S).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MOM2_INIT_CONSTANT: float = 1.0  # ml_model.cpp:23


@dataclass
class InitialModelState:
    """Plain dataclass (not Equinox) so tests run CPU-only without JAX tracing."""

    iter: int = 0
    nr_iter: int = 200
    K: int = 1
    ori_size: int = 64
    pixel_size: float = 1.0
    pseudo_halfsets: bool = True

    Iref: np.ndarray = field(default_factory=lambda: np.zeros((1, 64, 64, 64)))
    Igrad1: np.ndarray = field(default_factory=lambda: np.zeros((2, 64, 64, 33), dtype=np.complex128))
    Igrad2: np.ndarray = field(
        default_factory=lambda: np.full(
            (1, 64, 64, 33), MOM2_INIT_CONSTANT + 1j * MOM2_INIT_CONSTANT, dtype=np.complex128
        )
    )

    sigma2_noise: np.ndarray = field(default_factory=lambda: np.zeros((1, 33)))
    tau2_class: np.ndarray = field(default_factory=lambda: np.zeros((1, 33)))
    sigma2_class: np.ndarray = field(default_factory=lambda: np.zeros((1, 33)))
    fsc_halves_class: np.ndarray = field(default_factory=lambda: np.zeros((1, 33)))
    fourier_coverage_class: np.ndarray = field(default_factory=lambda: np.zeros((1, 33)))
    data_vs_prior_class: np.ndarray = field(default_factory=lambda: np.zeros((1, 33)))

    pdf_class: np.ndarray = field(default_factory=lambda: np.ones(1))
    pdf_direction: Optional[np.ndarray] = None
    sigma2_offset: float = 100.0

    ini_high: float = -1.0
    current_resolution: float = 0.0
    current_resolution_shell: int = 0
    current_size: int = 0
    incr_size: int = 10
    ave_Pmax: float = 0.0
    has_high_fsc_at_limit: bool = False

    Mavg: Optional[np.ndarray] = None

    subset_particle_ids: Optional[np.ndarray] = None
    subset_halfset_ids: Optional[np.ndarray] = None

    grad_current_stepsize: float = 0.5
    tau2_fudge_factor: float = 1.0
    subset_size: int = -1

    has_converged: bool = False
    grad_has_converged: bool = False


def half_slot_count(K: int, pseudo_halfsets: bool) -> int:
    """Per RELION: 2K slots when pseudo-halfsets active, else K."""
    return 2 * K if pseudo_halfsets else K


def half_slot_index(k: int, h: int, K: int, pseudo_halfsets: bool) -> int:
    """``Igrad1[h*K + k]`` (ml_model.cpp:935-937); h must be 0 when pseudo_halfsets off."""
    if not pseudo_halfsets:
        if h != 0:
            raise ValueError("halfset id must be 0 when pseudo_halfsets is off")
        return k
    if h not in (0, 1):
        raise ValueError(f"halfset id must be 0 or 1, got {h}")
    return h * K + k
