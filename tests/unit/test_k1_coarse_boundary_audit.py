import numpy as np
import pytest

from recovar.em.k1_coarse_boundary_audit import (
    align_relion_surface,
    rotation_bijection,
    translation_bijection,
)


def test_rotation_and_surface_alignment_are_exact_bijections():
    recovar = np.asarray([np.eye(3), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]], dtype=np.float32)
    relion = recovar[[1, 0]].transpose(0, 2, 1)
    rotation_map = rotation_bijection(relion, recovar)
    np.testing.assert_array_equal(rotation_map, [1, 0])

    relion_scores = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    aligned = align_relion_surface(relion_scores, rotation_map, np.asarray([1, 0]))
    np.testing.assert_array_equal(aligned, [[4, 3], [2, 1]])


def test_translation_bijection_decodes_relion_soa_phases():
    image_size = 256
    translations = np.asarray([[0.25, -0.75], [1.25, 0.25]], dtype=np.float32)
    phase_xy = -translations * np.float32(2.0 * np.pi / image_size)
    phases_soa = np.concatenate([phase_xy[:, 0], phase_xy[:, 1], np.zeros(2, np.float32)])
    mapping, decoded, max_error = translation_bijection(phases_soa, translations, image_size=image_size)
    np.testing.assert_array_equal(mapping, [0, 1])
    np.testing.assert_allclose(decoded, translations, atol=1.0e-6, rtol=0)
    assert max_error < 1.0e-6


def test_rotation_alignment_fails_closed_when_grid_is_incomplete():
    recovar = np.asarray([np.eye(3), 2 * np.eye(3)], dtype=np.float32)
    relion = np.asarray([np.eye(3), 3 * np.eye(3)], dtype=np.float32)
    with pytest.raises(ValueError, match="no bitwise RECOVAR transpose match"):
        rotation_bijection(relion, recovar)
