import numpy as np
import pytest

from recovar.em.dense_single_volume.debug_dumps import _save_iteration_particle_states


def _per_half(value1, value2):
    return [np.asarray(value1), np.asarray(value2)]


def test_save_iteration_particle_states_preserves_source_aligned_values(tmp_path):
    rotations = _per_half(np.eye(3, dtype=np.float32)[None], np.repeat(np.eye(3)[None], 2, axis=0))
    eulers = _per_half([[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    relative = _per_half([[0.25, -0.5]], [[1.0, 2.0], [3.0, 4.0]])
    absolute = _per_half([[10.25, 19.5]], [[31.0, 42.0], [53.0, 64.0]])
    pmax = _per_half([0.75], [0.5, 0.25])
    counts = _per_half([3], [4, 5])
    hard = _per_half([11], [12, 13])
    coarse = _per_half([21], [22, 23])
    original = _per_half([101], [205, 207])

    _save_iteration_particle_states(
        str(tmp_path),
        iteration=2,
        rotation_matrices_per_half=rotations,
        rotation_eulers_deg_per_half=eulers,
        relative_translations_pixels_per_half=relative,
        absolute_translations_pixels_per_half=absolute,
        max_posterior_per_half=pmax,
        significant_counts_per_half=counts,
        hard_assignments_per_half=hard,
        coarse_hard_assignments_per_half=coarse,
        original_image_indices_per_half=original,
    )

    half2 = np.load(tmp_path / "it002_particle_state_half2.npz")
    np.testing.assert_array_equal(half2["half_local_indices"], [0, 1])
    np.testing.assert_array_equal(half2["original_image_indices"], [205, 207])
    np.testing.assert_allclose(half2["rotation_eulers_deg"], eulers[1])
    np.testing.assert_allclose(half2["relative_translations_pixels"], relative[1])
    np.testing.assert_allclose(half2["absolute_translations_pixels"], absolute[1])
    np.testing.assert_allclose(half2["max_posterior"], pmax[1])
    np.testing.assert_array_equal(half2["significant_counts"], counts[1])
    np.testing.assert_array_equal(half2["one_based_iteration"], [3])
    np.testing.assert_array_equal(half2["half"], [2])


def test_save_iteration_particle_states_rejects_misaligned_fields(tmp_path):
    one = _per_half([1], [2])
    misaligned = _per_half([1, 2], [3])

    with pytest.raises(ValueError, match="not source-aligned for half 1"):
        _save_iteration_particle_states(
            str(tmp_path),
            iteration=0,
            rotation_matrices_per_half=one,
            rotation_eulers_deg_per_half=one,
            relative_translations_pixels_per_half=one,
            absolute_translations_pixels_per_half=one,
            max_posterior_per_half=misaligned,
            significant_counts_per_half=one,
            hard_assignments_per_half=one,
            coarse_hard_assignments_per_half=one,
            original_image_indices_per_half=one,
        )
