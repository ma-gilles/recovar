"""Unit tests for parity_dump's wall-time / per-stage timing hooks."""

from __future__ import annotations

import time

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
