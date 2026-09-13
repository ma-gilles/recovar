"""RELION's per-iteration SamplingPerturbation advance has one owner shared by both passes."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import recovar.em.refinement.iteration_loop as iteration_loop
from recovar.em.sampling import advance_relion_perturbation, advance_relion_perturbation_from_seed

pytestmark = pytest.mark.unit


def test_explicit_seed_draws_from_seed_plus_iteration():
    rp, seed = iteration_loop._advance_relion_perturbation(0.1, perturb_factor=0.5, perturb_seed=np.int64(100), relion_iteration=7, rng=None)
    assert seed == 107 and type(seed) is int
    assert rp == advance_relion_perturbation_from_seed(0.1, 0.5, seed=107)


def test_generator_path_reports_no_seed():
    rng_a = np.random.default_rng(5)
    rng_b = np.random.default_rng(5)
    rp, seed = iteration_loop._advance_relion_perturbation(0.1, perturb_factor=0.5, perturb_seed=None, relion_iteration=7, rng=rng_a)
    assert seed is None and rp == advance_relion_perturbation(0.1, 0.5, rng_b)


def test_controller_advances_the_perturbation_through_the_owner_in_both_passes():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    assert source.count("_advance_relion_perturbation(") == 2
    assert "advance_relion_perturbation_from_seed(" not in source
    assert source.count("advance_relion_perturbation(") == 2
