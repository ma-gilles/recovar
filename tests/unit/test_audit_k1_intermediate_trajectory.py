import numpy as np
import pandas as pd
import pytest

from scripts.audit_k1_intermediate_trajectory import (
    AuditError,
    _values_in_original_image_order,
    array_metrics,
)


def test_array_metrics_reports_exact_and_direct_errors():
    exact = array_metrics(np.asarray([1, 2], dtype=np.float32), np.asarray([1, 2], dtype=np.float64))
    assert exact["exact_equal"]
    assert exact["max_abs"] == 0.0
    assert exact["relative_l2"] == 0.0

    changed = array_metrics([1.0, 2.0], [1.5, 1.5])
    assert not changed["exact_equal"]
    assert changed["max_abs"] == 0.5
    np.testing.assert_allclose(changed["rms"], 0.5)


def test_array_metrics_fails_closed_on_shape_or_nonfinite_values():
    mismatch = array_metrics([1.0], [1.0, 2.0])
    assert not mismatch["shape_equal"]

    rank_mismatch = array_metrics([1.0, 2.0], [[1.0, 2.0]])
    assert not rank_mismatch["shape_equal"]
    assert rank_mismatch["left_shape"] == [2]
    assert rank_mismatch["right_shape"] == [1, 2]

    nonfinite = array_metrics([1.0, np.nan], [1.0, 2.0])
    assert nonfinite["shape_equal"]
    assert not nonfinite["all_finite"]


def test_relion_particle_values_are_restored_to_original_image_order():
    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "10@particles.mrcs"]
            + [f"{i}@particles.mrcs" for i in range(2, 10)],
            "rlnMaxValueProbDistribution": [1.0, 10.0] + [float(i) for i in range(2, 10)],
        }
    )
    reordered, identity = _values_in_original_image_order(
        particles, "rlnMaxValueProbDistribution"
    )
    np.testing.assert_array_equal(reordered, np.arange(1.0, 11.0))
    assert identity["exact_permutation"]
    assert identity["identity_count"] == 10


def test_relion_particle_identity_mapping_fails_closed():
    duplicate = pd.DataFrame(
        {
            "rlnImageName": ["1@particles.mrcs", "1@particles.mrcs"],
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(AuditError, match="exact single-stack permutation"):
        _values_in_original_image_order(duplicate, "value")


def test_relion_particle_values_use_exact_reference_identities_across_stacks():
    particles = pd.DataFrame(
        {
            "rlnImageName": ["1@stack_b.mrcs", "1@stack_a.mrcs"],
            "value": [20.0, 10.0],
        }
    )
    reordered, identity = _values_in_original_image_order(
        particles,
        "value",
        ["1@stack_a.mrcs", "1@stack_b.mrcs"],
    )
    np.testing.assert_array_equal(reordered, [10.0, 20.0])
    assert identity["exact_identity_set"]
