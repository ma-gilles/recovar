"""Tests for the centralized dense EM memory planner."""

import pytest

pytest.importorskip("jax")

from recovar import utils as rec_utils
from recovar.em.dense_single_volume.plan import (
    _DOT_MULT,
    _MSTEP_IMG_MULT,
    _MSTEP_ROT_DIV,
    _NORM_MULT,
    _PROB_DIV,
    _PROJ_MULT,
    plan_em_iteration,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_gpu(monkeypatch):
    """Deterministic GPU memory for testing."""
    monkeypatch.setattr(rec_utils, "get_gpu_memory_total", lambda device=0: 10)


def test_plan_matches_original_e_step_batch_sizes(mock_gpu):
    """Verify planner produces same batch sizes as original e_step.py code."""
    grid_size = 64
    n_rotations = 100
    n_translations = 5
    projection_size_gb = 0.5

    plan = plan_em_iteration(
        grid_size=grid_size,
        n_rotations=n_rotations,
        n_translations=n_translations,
        projection_size_gb=projection_size_gb,
    )

    # Replicate original e_step.py calculations
    gpu_memory = 10  # from mock
    base_batch = rec_utils.get_image_batch_size(grid_size, gpu_memory)

    expected_proj = rec_utils.safe_batch_size(base_batch * _PROJ_MULT)
    assert plan.projection_batch == expected_proj

    remaining = gpu_memory - projection_size_gb
    dot_base = rec_utils.get_image_batch_size(grid_size, remaining)
    expected_dot = rec_utils.safe_batch_size(dot_base / n_translations * _DOT_MULT)
    assert plan.dot_product_batch == expected_dot

    expected_norm = rec_utils.safe_batch_size(rec_utils.get_image_batch_size(grid_size, remaining) * _NORM_MULT)
    assert plan.norm_batch == expected_norm

    expected_prob = rec_utils.safe_batch_size(expected_proj // _PROB_DIV)
    assert plan.prob_batch == expected_prob


def test_plan_matches_original_m_step_batch_sizes(mock_gpu):
    """Verify planner produces same batch sizes as original m_step.py code."""
    grid_size = 64
    n_rotations = 100
    n_translations = 5

    plan = plan_em_iteration(
        grid_size=grid_size,
        n_rotations=n_rotations,
        n_translations=n_translations,
    )

    gpu_memory = 10
    expected_mstep = rec_utils.safe_batch_size(
        rec_utils.get_image_batch_size(grid_size, gpu_memory) // n_translations * _MSTEP_IMG_MULT
    )
    assert plan.mstep_image_batch == expected_mstep

    expected_rot = max(1, n_rotations // _MSTEP_ROT_DIV)
    assert plan.mstep_rotation_batch == expected_rot


def test_plan_matches_original_outer_batch(mock_gpu):
    """Verify planner produces same outer image batch as iterations.py."""
    n_rotations = 50
    n_translations = 10
    memory_to_use = 128.0

    plan = plan_em_iteration(
        grid_size=64,
        n_rotations=n_rotations,
        n_translations=n_translations,
        memory_to_use_gb=memory_to_use,
    )

    total_hidden = n_rotations * n_translations
    expected = max(1, int(memory_to_use * 1e9 / (total_hidden * 8)))
    assert plan.image_batch == expected


def test_plan_minimum_batch_sizes(mock_gpu):
    """All batch sizes must be >= 1."""
    plan = plan_em_iteration(
        grid_size=512,
        n_rotations=10000,
        n_translations=100,
        memory_to_use_gb=0.001,
        projection_size_gb=9.9,
    )

    assert plan.projection_batch >= 1
    assert plan.dot_product_batch >= 1
    assert plan.norm_batch >= 1
    assert plan.prob_batch >= 1
    assert plan.image_batch >= 1
    assert plan.mstep_image_batch >= 1
    assert plan.mstep_rotation_batch >= 1


def test_plan_single_rotation_translation(mock_gpu):
    """Edge case: 1 rotation, 1 translation."""
    plan = plan_em_iteration(
        grid_size=16,
        n_rotations=1,
        n_translations=1,
    )

    assert plan.mstep_rotation_batch == 1
    assert plan.image_batch >= 1
