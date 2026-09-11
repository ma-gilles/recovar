"""``_PerClassResults`` is the one per-class collector of the full-image K-class runners."""

import inspect

import numpy as np

from recovar.em.dense_single_volume import k_class


def test_collector_appends_in_class_order_and_honours_options():
    r = k_class._PerClassResults(accumulate_noise=True, return_best_pose_details=True, return_profile=True, keep_means=True)
    r.append(mean="m", Ft_y="y", Ft_ctf="c", hard_assignment=[1, 2], stats="s", noise="n", best_pose=("R", "T", "I"), best_pose_eulers_deg="E", profile_summary={"em_time_s": 1.0})
    assert r.new_means == ["m"] and r.Ft_y == ["y"] and r.Ft_ctf == ["c"] and r.per_class_stats == ["s"]
    assert r.hard_assignments[0].dtype == np.int32 and r.hard_assignments[0].tolist() == [1, 2]
    assert r.noise_tuple() == ("n",) and r.best_pose_rotations == ["R"] and r.best_pose_eulers_deg == ["E"] and r.profile_summaries == [{"em_time_s": 1.0}]
    off = k_class._PerClassResults(accumulate_noise=False, return_best_pose_details=False)
    off.append(Ft_y="y", Ft_ctf="c", hard_assignment=[0], stats="s", noise="ignored")
    assert off.noise_tuple() is None and off.best_pose_rotations is None and off.profile_summaries is None and off.new_means is None


def test_both_full_image_runners_collect_through_the_owner():
    for fn in (k_class.run_dense_k_class_em, k_class.run_local_k_class_em):
        src = inspect.getsource(fn)
        assert src.count("_PerClassResults(") == 1 and src.count("results.append(") == 1
        assert "per_class_best_pose_rotations = [] if" not in src and "hard_assignments.append(" not in src
