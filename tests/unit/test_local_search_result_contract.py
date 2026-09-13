"""Optional local-search outputs retain their meanings for K-class callers."""

from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.helpers.types import LocalEMResult
from recovar.em.local import local_search_iteration

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("return_best_pose_details", [False, True])
@pytest.mark.parametrize("accumulate_noise", [False, True])
@pytest.mark.parametrize("return_class_details", [False, True])
def test_kclass_optional_outputs_preserve_statistics(
    monkeypatch, n_classes, return_best_pose_details, accumulate_noise, return_class_details
):
    stats, noise = object(), object()
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    translations = np.zeros((2, 2), dtype=np.float32)
    assignments = np.array([0, 1], dtype=np.int32)
    class_sums = np.arange(1, n_classes + 1, dtype=np.float64)
    engine_result = SimpleNamespace(
        Ft_y=np.zeros((n_classes, 8), dtype=np.complex64),
        Ft_ctf=np.ones((n_classes, 8), dtype=np.float32),
        pose_assignments=assignments,
        best_pose_rotations=rotations if return_best_pose_details else None,
        best_pose_translations=translations if return_best_pose_details else None,
        best_pose_rotation_ids=assignments if return_best_pose_details else None,
        stats=stats,
        aggregate_noise_stats=noise if accumulate_noise else None,
        class_assignments=assignments,
        class_posterior_sums=class_sums,
    )

    def run_kclass(*args, **kwargs):
        assert kwargs["return_best_pose_details"] == return_best_pose_details
        assert kwargs["accumulate_noise"] == accumulate_noise
        return engine_result

    monkeypatch.setattr(local_search_iteration, "run_local_k_class_em", run_kclass)
    monkeypatch.setattr(
        local_search_iteration,
        "_estimate_relion_em_batch_sizes",
        lambda **kwargs: SimpleNamespace(
            image_batch_size=kwargs["requested_image_batch_size"],
            rotation_block_size=kwargs["requested_rotation_block_size"],
        ),
    )
    result = local_search_iteration._run_local_search_iteration(
        SimpleNamespace(image_shape=(2, 2), volume_shape=(2, 2, 2)),
        engine_result.Ft_y, None, None, rotations, rotations,
        healpix_order=0, sigma_rot=1.0, sigma_psi=1.0,
        translations=translations[:1], prior_translations=translations,
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp", image_batch_size=2, rotation_block_size=1,
        current_size=2,
        pass2_layout=SimpleNamespace(rotation_counts=np.ones(2, dtype=np.int32), translation_grid=translations[:1]),
        class_log_priors=np.full(n_classes, -np.log(n_classes)),
        accumulate_noise=accumulate_noise,
        return_best_pose_details=return_best_pose_details,
        return_class_details=return_class_details,
    )

    assert result.relion_stats is stats
    assert result.noise_stats is (noise if accumulate_noise else None)
    assert result.Ft_y is engine_result.Ft_y
    assert result.Ft_ctf is engine_result.Ft_ctf
    np.testing.assert_array_equal(result.hard_assignment, assignments)
    assert result.best_pose_rotations is engine_result.best_pose_rotations
    assert result.best_pose_translations is engine_result.best_pose_translations
    assert result.best_pose_rotation_ids is engine_result.best_pose_rotation_ids
    assert result.profile_summary is None
    assert result.significant_counts is None
    if return_class_details:
        np.testing.assert_array_equal(result.class_assignments, assignments)
        np.testing.assert_array_equal(result.class_posterior_sums, class_sums)
        np.testing.assert_array_equal(result.class_full_posterior_sums, class_sums)
    else:
        assert result.class_assignments is None
        assert result.class_posterior_sums is None
        assert result.class_full_posterior_sums is None


@pytest.mark.parametrize("return_profile", [False, True])
@pytest.mark.parametrize("return_significant_counts", [False, True])
def test_local_sample_capture_does_not_shift_significant_counts(
    monkeypatch, return_profile, return_significant_counts
):
    """Sample capture enables an internal profile even if the caller hides it."""
    counts = np.array([3, 7], dtype=np.int32)
    profile = {"reconstruction_sample_indices_by_image": (np.array([1]), np.array([2]))}
    stats = object()

    def run_local(*args, **kwargs):
        assert kwargs["return_reconstruction_sample_indices"] is True
        assert kwargs["return_profile"] == return_profile
        assert kwargs["return_significant_counts"] == return_significant_counts
        return LocalEMResult(
            Ft_y=np.zeros(8, dtype=np.complex64),
            Ft_ctf=np.ones(8, dtype=np.float32),
            hard_assignments=np.array([0, 1], dtype=np.int32),
            stats=stats,
            profile=profile,
            significant_counts=counts if return_significant_counts else None,
        )

    monkeypatch.setattr(local_search_iteration, "run_local_em_exact", run_local)
    monkeypatch.setattr(
        local_search_iteration, "_estimate_relion_em_batch_sizes",
        lambda **kwargs: SimpleNamespace(
            image_batch_size=kwargs["requested_image_batch_size"],
            rotation_block_size=kwargs["requested_rotation_block_size"],
        ),
    )
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    translations = np.zeros((2, 2), dtype=np.float32)
    result = local_search_iteration._run_local_search_iteration(
        SimpleNamespace(image_shape=(2, 2), volume_shape=(2, 2, 2)),
        None, None, None, rotations, rotations,
        healpix_order=0, sigma_rot=1.0, sigma_psi=1.0,
        translations=translations[:1], prior_translations=translations,
        sigma_offset_angstrom=1.0,
        disc_type="linear_interp", image_batch_size=2, rotation_block_size=1,
        current_size=2,
        pass2_layout=SimpleNamespace(rotation_counts=np.ones(2, dtype=np.int32), translation_grid=translations[:1]),
        return_reconstruction_sample_indices=True,
        return_significant_counts=return_significant_counts,
        return_profile=return_profile,
    )
    assert result.relion_stats is stats
    assert result.significant_counts is (counts if return_significant_counts else None)
    if return_profile:
        assert result.profile_summary is not profile
        assert result.profile_summary["reconstruction_sample_indices_by_image"] is profile["reconstruction_sample_indices_by_image"]
    else:
        assert result.profile_summary is None
    assert set(profile) == {"reconstruction_sample_indices_by_image"}
