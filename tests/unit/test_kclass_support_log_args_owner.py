"""``_pass2_support_log_args`` is the one owner of the 21 support values the adaptive pass-2 routing logs print."""

import inspect

from recovar.em.dense_single_volume import k_class


def test_owner_orders_the_21_values_like_the_log_formats():
    stats = {k: k for k in ("rotation_median", "rotation_median_fraction", "rotation_mean", "rotation_mean_fraction", "rotation_max", "rotation_max_fraction", "pose_median", "pose_median_fraction", "pose_mean", "pose_mean_fraction", "pose_max", "pose_max_fraction")}
    vals = k_class._pass2_support_log_args(stats, n_rot_fine=4, n_trans_fine=5, dense_support_threshold=0.25, mean_threshold_text="m", small_threshold_text="s")
    assert vals == ("rotation_median", 4, "rotation_median_fraction", 0.25, "rotation_mean", 4, "rotation_mean_fraction", "m", "s", "rotation_max", 4, "rotation_max_fraction", "pose_median", 20, "pose_median_fraction", "pose_mean", 20, "pose_mean_fraction", "pose_max", 20, "pose_max_fraction")


def test_three_routing_logs_share_the_owner():
    src = inspect.getsource(k_class.run_dense_k_class_em_adaptive)
    assert src.count("*support_log_args,") == 3 and src.count("_pass2_support_log_args(") == 1
    assert 'support_stats["pose_max_fraction"]' not in src
