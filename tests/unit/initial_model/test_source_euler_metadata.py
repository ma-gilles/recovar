"""Exact source pose metadata survives partial InitialModel updates and publication."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from recovar.em.initial_model import dense_adapter, driver

pytestmark = pytest.mark.unit


def state(n=3):
    return driver.NativeParticleState(np.zeros((n, 2)), np.zeros(n, np.int32), np.zeros(n, np.float32))


def test_subset_source_validity_and_mixed_legacy_rows():
    value = state()
    value.best_pose_rotations = np.zeros((3, 3, 3), np.float32)
    eulers = np.array([[159.3271497477632, 126.91279408422895, 85.75518260708287], [-2.0, 30.0, 7.0]])
    driver._update_particle_state_from_estep_meta(
        value,
        dict(
            selected_particle_ids=np.array([2, 0]),
            best_pose_eulers_deg=eulers,
            best_pose_eulers_valid=np.array([True, False]),
        ),
        np.zeros((1, 2)),
    )
    np.testing.assert_array_equal(value.best_pose_eulers_valid, [False, False, True])
    value.best_pose_rotations[0] = driver.R_from_relion(eulers[1:], degrees=True).astype(np.float32)[0]
    got = driver._best_eulers_from_particle_state(value, np.array([2, 0]), rotation_grid_order=0)
    np.testing.assert_array_equal(got[0], eulers[0])
    np.testing.assert_array_equal(
        got[1], driver.R_to_relion(value.best_pose_rotations[[0]].astype(np.float64), degrees=True)[0]
    )
    # A matrix-only replacement invalidates that row, never a different particle.
    driver._update_particle_state_from_estep_meta(
        value,
        dict(selected_particle_ids=np.array([0]), best_pose_rotations=np.eye(3, dtype=np.float32)[None]),
        np.zeros((1, 2)),
    )
    np.testing.assert_array_equal(value.best_pose_eulers_valid, [False, False, True])
    np.testing.assert_array_equal(value.best_pose_eulers_deg[2], eulers[0])
    driver._update_particle_state_from_estep_meta(
        value,
        dict(selected_particle_ids=np.array([2]), best_pose_rotations=np.eye(3, dtype=np.float32)[None]),
        np.zeros((1, 2)),
    )
    assert not np.any(value.best_pose_eulers_valid)


def test_input_star_source_is_valid_before_first_visit():
    eulers = np.array([[159.3271497477632, 126.91279408422895, 85.75518260708287]])
    frame = pd.DataFrame(
        dict(
            _rlnImageName=["1@stack.mrcs"],
            _rlnAngleRot=eulers[:, 0],
            _rlnAngleTilt=eulers[:, 1],
            _rlnAnglePsi=eulers[:, 2],
        )
    )
    value = driver._particle_state_from_star(frame, SimpleNamespace(voxel_size=1.0, n_images=1))
    assert not value.visited[0] and value.best_pose_eulers_valid[0]
    np.testing.assert_array_equal(driver._best_eulers_from_particle_state(value, [0], rotation_grid_order=0), eulers)
    assert value.best_pose_rotations.dtype == np.float32


@pytest.mark.parametrize("fault", ["narrow", "shape", "nan", "valid_shape", "valid_dtype"])
def test_invalid_restored_source_metadata_rejected(fault):
    value = state(1)
    value.best_pose_eulers_deg = np.ones((1, 3), np.float64)
    value.best_pose_eulers_valid = np.ones(1, bool)
    if fault == "narrow":
        value.best_pose_eulers_deg = value.best_pose_eulers_deg.astype(np.float32)
    elif fault == "shape":
        value.best_pose_eulers_deg = np.ones((2, 3))
    elif fault == "nan":
        value.best_pose_eulers_deg[0, 0] = np.nan
    elif fault == "valid_shape":
        value.best_pose_eulers_valid = np.ones(2, bool)
    else:
        value.best_pose_eulers_valid = np.ones(1, np.int32)
    with pytest.raises(ValueError, match="Euler"):
        driver._best_eulers_from_particle_state(value, [0], rotation_grid_order=0)


def test_mixed_halfset_rows_keep_identity_and_validity():
    source = np.array([[2.0 + 2**-40, 30.0, 4.0]])
    results = {1: SimpleNamespace(best_pose_eulers_deg=source), 0: SimpleNamespace()}
    meta = dense_adapter._sparse_pass2_estep_meta(results, {1: np.array([2]), 0: np.array([1, 0])})
    np.testing.assert_array_equal(meta["selected_particle_ids"], [1, 0, 2])
    np.testing.assert_array_equal(meta["best_pose_eulers_valid"], [False, False, True])
    np.testing.assert_array_equal(meta["best_pose_eulers_deg"][2], source[0])
    value = state()
    driver._update_particle_state_from_estep_meta(value, meta, np.zeros((1, 2)))
    np.testing.assert_array_equal(value.best_pose_eulers_valid, [False, False, True])
    np.testing.assert_array_equal(value.best_pose_eulers_deg[2], source[0])


def test_coarse_winner_replaces_or_invalidates_fine_source_metadata():
    from recovar.em.dense_single_volume import k_class
    from recovar.em.dense_single_volume.helpers.types import make_relion_stats

    stats = make_relion_stats(
        log_evidence_per_image=np.zeros(1),
        best_log_score_per_image=np.zeros(1),
        max_posterior_per_image=np.ones(1),
        rotation_posterior_sums=np.ones(1),
    )
    result = k_class._assemble_result(
        class_log_evidence=np.zeros((1, 1)),
        new_means=None,
        Ft_y=[np.zeros(1, np.complex64)],
        Ft_ctf=[np.ones(1, np.float32)],
        per_class_hard_assignments=np.zeros((1, 1), np.int32),
        per_class_stats=(stats,),
        noise_stats=None,
        per_class_best_pose_eulers_deg=[np.array([[91.0, 33.0, 42.0]])],
    )
    kwargs = dict(
        full_stats=dict(
            log_evidence_per_image=np.zeros(1), best_log_score_per_image=np.zeros(1), max_posterior_per_image=np.ones(1)
        ),
        hard_assignment=np.array([1]),
        class_assignment=np.array([0]),
        coarse_rotations=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        coarse_translations=np.zeros((1, 2), np.float32),
    )
    legacy = dense_adapter._restore_zero_oversampling_coarse_metadata(result, **kwargs)
    assert legacy.best_pose_eulers_deg is None and legacy.per_class_best_pose_eulers_deg is None
    source = np.array([[0.0, 0.0, 0.0], [17.0 + 2**-40, 22.0, 31.0]])
    exact = dense_adapter._restore_zero_oversampling_coarse_metadata(result, coarse_source_eulers=source, **kwargs)
    np.testing.assert_array_equal(exact.best_pose_eulers_deg, source[1:2])
    np.testing.assert_array_equal(exact.best_pose_rotations, legacy.best_pose_rotations)
