"""RELION's local-search orientational prior widths have one owner shared by both passes."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import recovar.em.refinement.iteration_loop as iteration_loop
from recovar.em.helpers.convergence import healpix_angular_step
from recovar.em.helpers.orientation_priors import relion_local_search_sigmas

pytestmark = pytest.mark.unit


def test_configured_widths_are_kept_and_psi_falls_back_to_rot():
    assert relion_local_search_sigmas(0.05, 0.0, use_local=True, healpix_order=3, adaptive_oversampling=1) == (0.05, 0.05)
    assert relion_local_search_sigmas(0.05, 0.03, use_local=True, healpix_order=3, adaptive_oversampling=1) == (0.05, 0.03)


@pytest.mark.parametrize("order,oversampling", [(3, 0), (4, 1), (5, 2)])
def test_unset_width_under_local_search_is_twice_the_oversampled_step(order, oversampling):
    rot, psi = relion_local_search_sigmas(0.0, 0.0, use_local=True, healpix_order=order, adaptive_oversampling=oversampling)
    expected = np.sqrt(2.0 * 2.0) * np.deg2rad(healpix_angular_step(order) / (2**oversampling))
    assert rot == expected and psi == expected


def test_global_search_keeps_unset_widths():
    assert relion_local_search_sigmas(0.0, 0.0, use_local=False, healpix_order=3, adaptive_oversampling=1) == (0.0, 0.0)


def test_controller_uses_the_owner_in_both_passes():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    assert source.count("relion_local_search_sigmas(") == 2
    assert "np.sqrt(2.0 * 2.0) * step_rad" not in source
