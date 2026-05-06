"""Unit tests for parity_dump's wall-time / per-stage timing hooks."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture()
def parity_env(tmp_path, monkeypatch):
    """Activate parity dump for the duration of one test."""
    dump = tmp_path / "dump"
    dump.mkdir()
    monkeypatch.setenv("RECOVAR_PARITY_DUMP_DIR", str(dump))
    # Reload the module so its globals see the env. parity_dump reads the env on
    # every call via is_active(), so a reload is not strictly necessary, but
    # it ensures _ITER_TIMERS starts empty.
    from recovar.em.dense_single_volume import parity_dump as p

    p._ITER_TIMERS.clear()
    p._E_STEP.clear()
    return p, dump


def test_inactive_when_env_unset(monkeypatch):
    monkeypatch.delenv("RECOVAR_PARITY_DUMP_DIR", raising=False)
    from recovar.em.dense_single_volume import parity_dump as p

    p._ITER_TIMERS.clear()
    assert not p.is_active()
    p.start_iteration(0)
    p.mark_stage(0, "e_step")
    wall, stages = p.get_iteration_timing(0)
    assert wall is None
    assert stages == {}
    # No state should have been recorded.
    assert 0 not in p._ITER_TIMERS


def test_start_and_mark_stage_records_cumulative_seconds(parity_env):
    p, _dump = parity_env
    p.start_iteration(4)
    time.sleep(0.05)
    p.mark_stage(4, "e_step")
    time.sleep(0.03)
    p.mark_stage(4, "recon")
    wall, stages = p.get_iteration_timing(4)
    assert wall is not None and wall > 0.07
    assert "e_step" in stages and "recon" in stages
    # Cumulative semantics: recon stamp is later than e_step stamp.
    assert stages["recon"] > stages["e_step"]
    assert stages["e_step"] >= 0.05


def test_dump_iteration_writes_wall_and_stage_fields(parity_env):
    p, dump = parity_env
    p.start_iteration(0)
    time.sleep(0.02)
    p.mark_stage(0, "e_step")
    p.mark_stage(0, "noise_update")
    p.mark_stage(0, "convergence")

    p.dump_iteration(
        iteration=0,
        init_relion_iteration=3,
        current_size=80,
        sigma_offset=1.0,
        translation_step=2.0,
        translation_range=10.0,
        random_perturbation=0.0,
        random_perturbation_instance=0,
        tau2_fudge=4.0,
        voxel_size=4.25,
        grid_size=128,
        volume_shape=(8, 8, 8),
        ave_pmax=0.5,
        fsc=np.zeros(64),
        sigma2_noise=np.ones(64),
        means=[None, None],
        unreg_means=[None, None],
        new_iter_best_rotation_eulers=[None, None],
        new_iter_best_translations=[None, None],
    )
    # iter_004.npz (init_relion_iteration=3 + iteration=0 + 1 = 4)
    files = sorted(dump.glob("iter_*.npz"))
    assert len(files) == 1
    npz = np.load(files[0], allow_pickle=False)
    assert "wall_time_s" in npz.files
    assert "stage_seconds_e_step" in npz.files
    assert "stage_seconds_noise_update" in npz.files
    assert "stage_seconds_convergence" in npz.files
    assert float(npz["wall_time_s"]) > 0.02
    # Timer entry should have been cleared after dump.
    assert 0 not in p._ITER_TIMERS


def test_collect_e_step_writes_k_class_diagnostics(parity_env):
    p, dump = parity_env
    p.collect_e_step(
        half=0,
        em_stats=SimpleNamespace(
            log_evidence_per_image=np.array([1.0, 2.0], dtype=np.float64),
            best_log_score_per_image=np.array([1.5, 2.5], dtype=np.float64),
            max_posterior_per_image=np.array([0.7, 0.9], dtype=np.float32),
            rotation_posterior_sums=np.array([0.25, 0.75], dtype=np.float32),
        ),
        hard_assignment=np.array([3, 4], dtype=np.int32),
        coarse_hard_assignment=np.array([1, 2], dtype=np.int32),
        noise_stats=SimpleNamespace(
            wsum_sigma2_noise=np.array([5.0, 6.0], dtype=np.float64),
            wsum_img_power=np.array([7.0, 8.0], dtype=np.float64),
            wsum_sigma2_offset=9.0,
            sumw=2.0,
        ),
        Ft_y=np.array([1.0 + 2.0j, 3.0 + 4.0j]),
        Ft_ctf=np.array([1.0, 2.0]),
        pose_rotation_eulers=np.zeros((2, 3), dtype=np.float32),
        best_pose_rotation_eulers=np.ones((2, 3), dtype=np.float32),
        best_pose_translations=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        translation_search_base=np.array([[0.5, 0.5], [1.5, 1.5]], dtype=np.float32),
        original_image_indices=np.array([10, 12], dtype=np.int64),
        class_assignments=np.array([1, 0], dtype=np.int32),
        class_responsibilities=np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float32),
        class_posterior_sums=np.array([1.0, 1.0], dtype=np.float64),
    )
    p.dump_iteration(
        iteration=0,
        init_relion_iteration=0,
        current_size=80,
        sigma_offset=1.0,
        translation_step=2.0,
        translation_range=10.0,
        random_perturbation=0.0,
        random_perturbation_instance=0,
        tau2_fudge=4.0,
        voxel_size=4.25,
        grid_size=128,
        volume_shape=(8, 8, 8),
        ave_pmax=0.5,
        fsc=np.zeros(64),
        sigma2_noise=np.ones(64),
        means=[None, None],
        unreg_means=[None, None],
        new_iter_best_rotation_eulers=[None, None],
        new_iter_best_translations=[None, None],
    )

    npz = np.load(next(dump.glob("iter_001.npz")), allow_pickle=False)
    np.testing.assert_array_equal(npz["half1_class_assignments"], np.array([1, 0], dtype=np.int32))
    np.testing.assert_allclose(
        npz["half1_class_responsibilities"],
        np.array([[0.2, 0.8], [0.8, 0.2]], dtype=np.float32),
    )
    np.testing.assert_allclose(npz["half1_class_posterior_sums"], np.array([1.0, 1.0]))
    np.testing.assert_array_equal(npz["half1_original_image_indices"], np.array([10, 12], dtype=np.int64))


def test_dump_iteration_falls_back_to_iteration_start_arg(parity_env):
    p, dump = parity_env
    # Don't call start_iteration; pass iteration_start directly.
    t0 = time.time()
    time.sleep(0.02)
    p.dump_iteration(
        iteration=0,
        init_relion_iteration=3,
        current_size=80,
        sigma_offset=1.0,
        translation_step=2.0,
        translation_range=10.0,
        random_perturbation=0.0,
        random_perturbation_instance=0,
        tau2_fudge=4.0,
        voxel_size=4.25,
        grid_size=128,
        volume_shape=(8, 8, 8),
        ave_pmax=0.5,
        fsc=np.zeros(64),
        sigma2_noise=np.ones(64),
        means=[None, None],
        unreg_means=[None, None],
        new_iter_best_rotation_eulers=[None, None],
        new_iter_best_translations=[None, None],
        iteration_start=t0,
    )
    files = sorted(dump.glob("iter_*.npz"))
    assert len(files) == 1
    npz = np.load(files[0], allow_pickle=False)
    assert "wall_time_s" in npz.files
    assert float(npz["wall_time_s"]) >= 0.02
    # No stage_seconds_* expected.
    assert not any(k.startswith("stage_seconds_") for k in npz.files)


def test_downsample_volume_real_handles_k_class_leading_axis(monkeypatch):
    from recovar.em.dense_single_volume import parity_dump as p

    monkeypatch.setenv("RECOVAR_PARITY_DUMP_VOLUME_DOWNSAMPLE", "2")
    volume_shape = (8, 8, 8)
    volumes = np.zeros((2, np.prod(volume_shape)), dtype=np.complex64)

    downsampled = p._downsample_volume_real(volumes, volume_shape)

    assert downsampled.shape == (2, 64)
