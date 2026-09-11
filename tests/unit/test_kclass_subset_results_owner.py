"""Both firstiter-CC global-winner subset passes assemble per-class results through one owner."""

import inspect

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume import k_class
from recovar.em.dense_single_volume.k_class_results import make_relion_stats


def _results(**overrides):
    kwargs = dict(n_images=3, accumulate_noise=False, return_best_pose_details=True, full_group_count=None,
                  pose_dtype=np.float32, score_dtype=np.float32)
    kwargs.update(overrides)
    return k_class._PerClassSubsetResults(**kwargs)


def test_empty_class_fills_zero_results_in_class_order():
    results = _results()
    mean = jnp.ones(4, dtype=jnp.complex64)
    results.append_empty_class(mean=mean, class_log_evidence=np.asarray([1.0, 2.0, 3.0]), noise_variance=None,
                               class_index=0, n_classes=1, n_rot=5, host_accumulators=True)
    assert isinstance(results.Ft_y[0], np.ndarray) and np.all(results.Ft_y[0] == 0)
    assert results.Ft_ctf[0].dtype == np.float32 and results.Ft_ctf[0].shape == (4,)
    assert np.array_equal(results.hard_assignments[0], np.zeros(3, dtype=np.int32))
    stats = results.per_class_stats[0]
    assert np.all(np.isneginf(stats.best_log_score_per_image)) and stats.rotation_posterior_sums.shape == (5,)
    assert np.array_equal(stats.log_evidence_per_image, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert results.best_pose_rotations[0].shape == (3, 3, 3) and results.best_pose_rotation_ids[0].dtype == np.int32
    assert results.per_class_noise is None

    device = _results()
    device.append_empty_class(mean=mean, class_log_evidence=np.zeros(3), noise_variance=None,
                              class_index=0, n_classes=1, n_rot=2, host_accumulators=False)
    assert isinstance(device.Ft_y[0], jnp.ndarray)


def test_class_results_expand_the_subset_to_the_full_image_axis():
    results = _results(return_best_pose_details=True)
    image_indices = np.asarray([0, 2], dtype=np.int64)
    stats_subset = make_relion_stats(
        log_evidence_per_image=np.asarray([0.5, 0.25], dtype=np.float32),
        best_log_score_per_image=np.asarray([-1.0, -2.0], dtype=np.float32),
        max_posterior_per_image=np.asarray([1.0, 1.0], dtype=np.float32),
        rotation_posterior_sums=np.asarray([2.0], dtype=np.float32),
    )
    best_pose = (
        np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([4, 5], dtype=np.int32),
    )
    results.append_class(
        image_indices=image_indices, Ft_y=jnp.ones(4, dtype=jnp.complex64), Ft_ctf=jnp.ones(4, dtype=jnp.float32),
        hard_full=np.asarray([7, 0, 9], dtype=np.int32), stats_subset=stats_subset,
        class_log_evidence=np.asarray([0.5, -9.0, 0.25]), noise=None, best_pose=best_pose, host_accumulators=True,
    )
    assert isinstance(results.Ft_y[0], np.ndarray) and isinstance(results.Ft_ctf[0], np.ndarray)
    assert np.array_equal(results.hard_assignments[0], [7, 0, 9])
    assert results.per_class_stats[0].best_log_score_per_image.shape == (3,)
    assert np.array_equal(results.best_pose_rotation_ids[0][image_indices], [4, 5])
    assert results.best_pose_rotations[0].shape == (3, 3, 3) and results.best_pose_translations[0].shape == (3, 2)


def test_subset_passes_use_the_owner():
    for fn, host_empty, host_class in (
        (k_class._run_firstiter_global_winner_subset_pass2, "True", "False"),
        (k_class._run_sparse_firstiter_global_winner_subset_pass2, "False", "True"),
    ):
        source = inspect.getsource(fn)
        assert source.count("results = _PerClassSubsetResults(") == 1
        assert source.count("results.append_empty_class(") == 1
        assert source.count("results.append_class(") == 1
        assert f"host_accumulators={host_empty},\n            )\n            continue" in source
        assert f"host_accumulators={host_class},\n        )" in source
        assert "per_class_best_pose_rotations.append(" not in source
        assert "Ft_y = []" not in source
        assert "results.assemble(" in source and "_assemble_result(" not in source
    assert "class_posterior_sums_override=np.asarray(self.subset_counts, dtype=np.float64)" in inspect.getsource(
        k_class._PerClassSubsetResults.assemble
    )
