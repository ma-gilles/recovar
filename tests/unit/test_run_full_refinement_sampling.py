"""CLI-level sampling contract tests for ``scripts/run_full_refinement.py``."""

import pytest

from scripts.run_full_refinement import _resolve_relion_sampling_orders


def test_relion_healpix_order_is_coarse_pass1_order():
    coarse, fine = _resolve_relion_sampling_orders(healpix_order=2, adaptive_oversampling=1)

    assert coarse == 2
    assert fine == 3


def test_relion_sampling_order_rejects_negative_values():
    with pytest.raises(ValueError):
        _resolve_relion_sampling_orders(healpix_order=-1, adaptive_oversampling=1)

    with pytest.raises(ValueError):
        _resolve_relion_sampling_orders(healpix_order=1, adaptive_oversampling=-1)
