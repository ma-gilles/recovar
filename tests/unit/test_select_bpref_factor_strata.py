import numpy as np
import pytest

from scripts import select_bpref_factor_strata as selector


def _arrays(n_particles=65, n_pixels=3):
    stacks = np.arange(1000, 1000 + n_particles, dtype=np.int64)
    recovar = np.ones((n_particles, n_pixels), dtype=np.complex64)
    errors = np.linspace(1e-5, 1e-3, n_particles, dtype=np.float32)
    relion = recovar * (1 + errors[:, None])
    return {
        "stack_indices_1based": stacks,
        "recovar_data": recovar,
        "relion_data_recovar_units": relion,
        "recovar_device_support_mask": np.ones_like(recovar, dtype=bool),
        "recovar_global_rotation_indices": np.arange(n_particles),
        "relion_orientation_class_keys": np.arange(n_particles, dtype=np.uint64) + 50,
        "relion_oversampled_rotations": np.arange(n_particles, dtype=np.uint64) % 8,
    }


def test_select_strata_is_deterministic_unique_and_excludes_tracked_stack():
    arrays = _arrays()
    excluded = int(arrays["stack_indices_1based"][17])
    first = selector.select_strata(arrays, excluded_stack=excluded)
    second = selector.select_strata(arrays, excluded_stack=excluded)

    assert first == second
    assert first["cohort"]["systematic_particle_count"] == 64
    assert first["cohort"]["selected_particle_count"] == 32
    selected = first["selected"]
    stacks = [row["stack_index_1based"] for row in selected]
    assert excluded not in stacks
    assert len(set(stacks)) == 32
    assert np.bincount([row["stratum_zero_based"] for row in selected]).tolist() == [4] * 8
    assert [row["within_stratum_zero_based"] for row in selected[:4]] == [0, 1, 2, 3]


def test_select_strata_rejects_missing_or_duplicated_excluded_stack():
    arrays = _arrays()
    with pytest.raises(ValueError, match="excluded stack"):
        selector.select_strata(arrays, excluded_stack=999)

    arrays["stack_indices_1based"][1] = arrays["stack_indices_1based"][0]
    with pytest.raises(ValueError, match="unique"):
        selector.select_strata(arrays, excluded_stack=1017)
