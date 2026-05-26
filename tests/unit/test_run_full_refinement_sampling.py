"""CLI-level sampling contract tests for ``scripts/run_full_refinement.py``."""

from types import SimpleNamespace

import pytest

from recovar.em.sampling import relion_sampling_perturbation_for_iteration
from scripts.run_full_refinement import (
    _effective_perturb_seed,
    _refine_sampling_kwargs,
    _resolve_relion_sampling_orders,
)


def test_relion_healpix_order_is_coarse_pass1_order():
    coarse, fine = _resolve_relion_sampling_orders(healpix_order=2, adaptive_oversampling=1)

    assert coarse == 2
    assert fine == 3


def test_relion_sampling_order_rejects_negative_values():
    with pytest.raises(ValueError):
        _resolve_relion_sampling_orders(healpix_order=-1, adaptive_oversampling=1)

    with pytest.raises(ValueError):
        _resolve_relion_sampling_orders(healpix_order=1, adaptive_oversampling=-1)


def test_cli_translation_grid_parameters_seed_refinement_state():
    args = SimpleNamespace(
        adaptive_oversampling=1,
        offset_range=3.0,
        offset_step=1.0,
        auto_local_healpix_order=4,
    )

    kwargs = _refine_sampling_kwargs(args, init_healpix_order=2)

    assert kwargs["init_healpix_order"] == 2
    assert kwargs["init_translation_range"] == 3.0
    assert kwargs["init_translation_step"] == 1.0
    assert kwargs["translation_pixel_offset"] == 1.0


def test_cli_perturb_seed_defaults_to_relion_random_seed():
    assert _effective_perturb_seed(SimpleNamespace(seed=17, perturb_seed=None)) == 17
    assert _effective_perturb_seed(SimpleNamespace(seed=17, perturb_seed=23)) == 23
    assert _effective_perturb_seed(SimpleNamespace(seed=17, perturb_seed=-1)) is None


def test_relion_seeded_sampling_perturbation_sequence_matches_reference_star():
    values = [
        relion_sampling_perturbation_for_iteration(
            perturbation_factor=0.5,
            random_seed=1776701668,
            relion_iteration=iteration,
        )
        for iteration in range(3)
    ]

    assert values == pytest.approx([0.460047, -0.25278, 0.125066], abs=5e-6)
