"""K-class result collection, subset expansion, pose overrides and dtype/log contracts."""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import k_class
from recovar.em.dense_single_volume.k_class_results import make_relion_stats

pytestmark = pytest.mark.unit


def test_collector_appends_in_class_order_and_honours_options():
    r = k_class._PerClassResults(
        accumulate_noise=True, return_best_pose_details=True, return_profile=True, keep_means=True
    )
    r.append(
        mean="m",
        Ft_y="y",
        Ft_ctf="c",
        hard_assignment=[1, 2],
        stats="s",
        noise="n",
        best_pose=("R", "T", "I"),
        best_pose_eulers_deg="E",
        profile_summary={"em_time_s": 1.0},
    )
    assert r.new_means == ["m"] and r.Ft_y == ["y"] and r.Ft_ctf == ["c"] and r.per_class_stats == ["s"]
    assert r.hard_assignments[0].dtype == np.int32 and r.hard_assignments[0].tolist() == [1, 2]
    assert (
        r.noise_tuple() == ("n",)
        and r.best_pose_rotations == ["R"]
        and r.best_pose_eulers_deg == ["E"]
        and r.profile_summaries == [{"em_time_s": 1.0}]
    )
    off = k_class._PerClassResults(accumulate_noise=False, return_best_pose_details=False)
    off.append(Ft_y="y", Ft_ctf="c", hard_assignment=[0], stats="s", noise="ignored")
    assert (
        off.noise_tuple() is None
        and off.best_pose_rotations is None
        and off.profile_summaries is None
        and off.new_means is None
    )


def test_both_full_image_runners_collect_through_the_owner():
    for fn in (k_class.run_dense_k_class_em, k_class.run_local_k_class_em):
        src = inspect.getsource(fn)
        assert src.count("_PerClassResults(") == 1 and src.count("results.append(") == 1
        assert "per_class_best_pose_rotations = [] if" not in src and "hard_assignments.append(" not in src


def _results(**overrides):
    kwargs = dict(
        n_images=3,
        accumulate_noise=False,
        return_best_pose_details=True,
        full_group_count=None,
        pose_dtype=np.float32,
        score_dtype=np.float32,
    )
    kwargs.update(overrides)
    return k_class._PerClassSubsetResults(**kwargs)


def test_empty_class_fills_zero_results_in_class_order():
    results = _results()
    mean = jnp.ones(4, dtype=jnp.complex64)
    results.append_empty_class(
        mean=mean,
        class_log_evidence=np.asarray([1.0, 2.0, 3.0]),
        noise_variance=None,
        class_index=0,
        n_classes=1,
        n_rot=5,
        host_accumulators=True,
    )
    assert isinstance(results.Ft_y[0], np.ndarray) and np.all(results.Ft_y[0] == 0)
    assert results.Ft_ctf[0].dtype == np.float32 and results.Ft_ctf[0].shape == (4,)
    assert np.array_equal(results.hard_assignments[0], np.zeros(3, dtype=np.int32))
    stats = results.per_class_stats[0]
    assert np.all(np.isneginf(stats.best_log_score_per_image)) and stats.rotation_posterior_sums.shape == (5,)
    assert np.array_equal(stats.log_evidence_per_image, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert results.best_pose_rotations[0].shape == (3, 3, 3) and results.best_pose_rotation_ids[0].dtype == np.int32
    assert results.per_class_noise is None

    device = _results()
    device.append_empty_class(
        mean=mean,
        class_log_evidence=np.zeros(3),
        noise_variance=None,
        class_index=0,
        n_classes=1,
        n_rot=2,
        host_accumulators=False,
    )
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
        image_indices=image_indices,
        Ft_y=jnp.ones(4, dtype=jnp.complex64),
        Ft_ctf=jnp.ones(4, dtype=jnp.float32),
        hard_full=np.asarray([7, 0, 9], dtype=np.int32),
        stats_subset=stats_subset,
        class_log_evidence=np.asarray([0.5, -9.0, 0.25]),
        noise=None,
        best_pose=best_pose,
        host_accumulators=True,
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


def test_dtype_rules():
    assert k_class._pose_dtype_from_kwargs({}) is np.float32 and k_class._score_dtype_from_kwargs({}) is np.float32
    assert k_class._pose_dtype_from_kwargs({"use_float64_projections": True}) is np.float64
    assert k_class._score_dtype_from_kwargs({"use_float64_projections": True}) is np.float32
    assert k_class._score_dtype_from_kwargs({"use_float64_scoring": True}) is np.float64


def test_rules_and_assembly_are_not_repeated():
    src = inspect.getsource(k_class)
    assert src.count('kwargs.get("use_float64_projections", False)') == 1
    assert src.count("_pose_dtype_from_kwargs(") == 4 and src.count("_score_dtype_from_kwargs(") == 4
    for fn in (
        k_class._run_firstiter_global_winner_subset_pass2,
        k_class._run_sparse_firstiter_global_winner_subset_pass2,
    ):
        body = inspect.getsource(fn)
        assert "results.assemble(" in body and "_assemble_result(" not in body
    assert "firstiter_winner_take_all=True" in inspect.getsource(k_class._PerClassSubsetResults.assemble)


class _Result:
    def __init__(self, per_class_hard):
        self.per_class_hard_assignments = per_class_hard
        self.replaced = None

    def _replace(self, **kw):
        out = _Result(self.per_class_hard_assignments)
        out.replaced = kw
        return out


def test_override_takes_the_winning_class_pose_and_decodes_details(monkeypatch):
    per_class_hard = jnp.asarray([[5, 6, 7], [8, 9, 10]], dtype=jnp.int32)
    winners = np.asarray([1, 0, 1])
    monkeypatch.setattr(
        k_class,
        "_decode_dense_best_pose_details",
        lambda hard, rots, trans: (("R", hard.tolist()), ("T", trans.shape), np.asarray(hard)),
    )
    out = k_class._override_class_assignments_with_coarse_winner(
        _Result(per_class_hard),
        winners,
        return_best_pose_details=True,
        fine_rotations_np=np.zeros((11, 3, 3)),
        fine_translations_np=np.zeros((11, 2)),
    )
    kw = out.replaced
    assert kw["class_assignments"].dtype == jnp.int32 and kw["class_assignments"].tolist() == [1, 0, 1]
    assert kw["pose_assignments"].tolist() == [8, 6, 10]
    assert kw["best_pose_rotations"] == ("R", [8, 6, 10]) and kw["best_pose_translations"] == ("T", (11, 2))
    assert kw["best_pose_rotation_ids"].tolist() == [8, 6, 10]
    assert kw["best_pose_eulers_deg"] is None and kw["per_class_best_pose_eulers_deg"] is None


def test_override_without_details_replaces_only_assignments():
    per_class_hard = jnp.asarray([[5, 6], [8, 9]], dtype=jnp.int32)
    out = k_class._override_class_assignments_with_coarse_winner(
        _Result(per_class_hard),
        np.asarray([0, 1]),
        return_best_pose_details=False,
        fine_rotations_np=np.zeros((10, 3, 3)),
        fine_translations_np=np.zeros((10, 2)),
    )
    assert set(out.replaced) == {"class_assignments", "pose_assignments"} and out.replaced[
        "pose_assignments"
    ].tolist() == [5, 9]


def test_both_pass2_paths_use_the_owner():
    source = inspect.getsource(k_class.run_dense_k_class_em_adaptive)
    assert source.count("_override_class_assignments_with_coarse_winner(") == 2
    assert "per_class_hard[coarse_assn, image_indices]" not in source


def test_owner_orders_the_21_values_like_the_log_formats():
    stats = {
        k: k
        for k in (
            "rotation_median",
            "rotation_median_fraction",
            "rotation_mean",
            "rotation_mean_fraction",
            "rotation_max",
            "rotation_max_fraction",
            "pose_median",
            "pose_median_fraction",
            "pose_mean",
            "pose_mean_fraction",
            "pose_max",
            "pose_max_fraction",
        )
    }
    vals = k_class._pass2_support_log_args(
        stats,
        n_rot_fine=4,
        n_trans_fine=5,
        dense_support_threshold=0.25,
        mean_threshold_text="m",
        small_threshold_text="s",
    )
    assert vals == (
        "rotation_median",
        4,
        "rotation_median_fraction",
        0.25,
        "rotation_mean",
        4,
        "rotation_mean_fraction",
        "m",
        "s",
        "rotation_max",
        4,
        "rotation_max_fraction",
        "pose_median",
        20,
        "pose_median_fraction",
        "pose_mean",
        20,
        "pose_mean_fraction",
        "pose_max",
        20,
        "pose_max_fraction",
    )


def test_three_routing_logs_share_the_owner():
    src = inspect.getsource(k_class.run_dense_k_class_em_adaptive)
    assert src.count("*support_log_args,") == 3 and src.count("_pass2_support_log_args(") == 1
    assert 'support_stats["pose_max_fraction"]' not in src
