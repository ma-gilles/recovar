"""The K=1 and K-class dense routes share one adaptive engine call owner."""

import inspect

import numpy as np

from recovar.em.classification import k_class
from recovar.em.dense import firstiter_cc, half_scoring


def test_sparse_pass2_switch_reads_only_positive_values(monkeypatch):
    for env_name in ("RECOVAR_K1_DENSE_PASS2", "RECOVAR_K_CLASS_DENSE_PASS2"):
        monkeypatch.delenv(env_name, raising=False)
        assert k_class._sparse_pass2_selected(env_name) is True
        for value in ("1", "true", " YES ", "on"):
            monkeypatch.setenv(env_name, value)
            assert k_class._sparse_pass2_selected(env_name) is False
        for value in ("", "0", "no", "off", "maybe"):
            monkeypatch.setenv(env_name, value)
            assert k_class._sparse_pass2_selected(env_name) is True


def _grids(fine_mstep):
    return half_scoring._AdaptivePass2Grids(
        coarse_rotations=np.zeros((2, 3, 3), dtype=np.float32),
        coarse_translations=np.zeros((1, 2), dtype=np.float32),
        fine_rotations=np.zeros((4, 3, 3), dtype=np.float32),
        fine_translations=np.zeros((3, 2), dtype=np.float32),
        rotation_parent_map=np.zeros(4, dtype=np.int64),
        translation_parent_map=np.zeros(3, dtype=np.int64),
        fine_mstep_rotations=fine_mstep,
        coarse_translation_phase_source=np.zeros((1, 2), dtype=np.float64),
        n_fine_translations=3,
    )


def test_shared_engine_keywords_follow_the_sparse_switch():
    fine_mstep = np.ones((4, 3, 3), dtype=np.float32)
    common = dict(
        class_log_priors="priors",
        significance_image_batch_size=8,
        significance_rotation_block_size=16,
        coarse_current_size=32,
        fine_current_size=48,
        coarse_healpix_order=np.int64(2),
        oversampling_order=np.int64(1),
        return_best_pose_details=True,
        bpref_device_signature_active=False,
        debug_iteration=3,
    )
    sparse = half_scoring._adaptive_engine_shared_kwargs(_grids(fine_mstep), max_significants=None, sparse_pass2=True, **common)
    dense = half_scoring._adaptive_engine_shared_kwargs(_grids(fine_mstep), max_significants=5, sparse_pass2=False, **common)
    expected_keys = {
        "class_log_priors", "accumulate_noise", "adaptive_fraction", "max_significants",
        "relion_fine_mstep_prune", "significance_image_batch_size", "significance_rotation_block_size",
        "coarse_current_size", "fine_current_size", "coarse_healpix_order", "oversampling_order",
        "fine_mstep_rotations_override", "return_best_pose_details", "bpref_device_signature_active",
        "debug_iteration",
    }
    assert set(sparse) == set(dense) == expected_keys
    assert sparse["accumulate_noise"] is True
    assert sparse["adaptive_fraction"] == half_scoring.RELION_ADAPTIVE_FRACTION
    assert sparse["max_significants"] == -1 and dense["max_significants"] == 5
    assert sparse["relion_fine_mstep_prune"] is True and dense["relion_fine_mstep_prune"] is False
    assert sparse["fine_mstep_rotations_override"] is fine_mstep and dense["fine_mstep_rotations_override"] is None
    assert type(sparse["coarse_healpix_order"]) is int and sparse["coarse_healpix_order"] == 2
    assert type(sparse["oversampling_order"]) is int and sparse["oversampling_order"] == 1
    assert sparse["class_log_priors"] == "priors" and sparse["debug_iteration"] == 3


def test_coarse_pose_assignments_need_a_fine_pass(monkeypatch):
    calls = []

    def fake_collapse(ha, **kwargs):
        calls.append(kwargs)
        return "collapsed"

    monkeypatch.setattr(half_scoring, "_collapse_fine_pose_assignments_to_coarse", fake_collapse)
    ha = np.zeros(2, dtype=np.int32)
    assert half_scoring._coarse_pose_assignments(ha, rot_parent_map=None, trans_parent_map=None, n_trans_coarse=1, n_trans_fine=None) is None
    assert half_scoring._coarse_pose_assignments(ha, rot_parent_map="rp", trans_parent_map="tp", n_trans_coarse=1, n_trans_fine=None) is None
    assert half_scoring._coarse_pose_assignments(ha, rot_parent_map="rp", trans_parent_map=None, n_trans_coarse=1, n_trans_fine=3) is None
    assert calls == []
    assert half_scoring._coarse_pose_assignments(ha, rot_parent_map="rp", trans_parent_map="tp", n_trans_coarse=1, n_trans_fine=3) == "collapsed"
    assert calls == [{"rot_parent_map": "rp", "trans_parent_map": "tp", "n_trans_coarse": 1, "n_trans_fine": 3}]


def test_dense_routes_use_the_shared_owners():
    source = inspect.getsource(half_scoring._score_half_dense)
    assert source.count("run_dense_k_class_em_adaptive(") == 2
    assert source.count("shared_kwargs = _adaptive_engine_shared_kwargs(") == 2
    assert source.count("**shared_kwargs,\n") == 2
    assert source.count("coarse_ha_k = _coarse_pose_assignments(") == 2
    assert "_collapse_fine_pose_assignments_to_coarse(" not in source
    assert 'kclass_sparse_pass2 = _sparse_pass2_selected("RECOVAR_K_CLASS_DENSE_PASS2")' in source
    assert 'k1_sparse_pass2 = _sparse_pass2_selected("RECOVAR_K1_DENSE_PASS2")' in source
    assert "os.environ.get(" not in source
    assert "coarse_rot = pass2_grids.coarse_rotations" not in source
    assert "coarse_translation_phase_source=pass2_grids.coarse_translation_phase_source" in source
    firstiter_source = inspect.getsource(firstiter_cc._score_kclass_firstiter_cc_pass2)
    assert 'firstiter_sparse_pass2 = _sparse_pass2_selected("RECOVAR_K_CLASS_DENSE_PASS2")' in firstiter_source
    assert "os.environ" not in firstiter_source
