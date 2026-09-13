"""Both passes estimate RELION's expected accuracy through one inputs owner and one class-label rule."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import recovar.em.refinement.iteration_loop as iteration_loop

pytestmark = pytest.mark.unit


def test_class_labels_use_k_class_assignments_when_present():
    ids = np.asarray([0, 1, 2], dtype=np.int32)
    assert iteration_loop._expected_accuracy_class_ids(ids, k_class_enabled=True, n_units=3) is ids


@pytest.mark.parametrize("k_class,assignments", [(True, None), (False, None), (False, np.asarray([1, 1, 1], dtype=np.int32))])
def test_class_labels_fall_back_to_class_zero(k_class, assignments):
    out = iteration_loop._expected_accuracy_class_ids(assignments, k_class_enabled=k_class, n_units=3)
    assert out.dtype == np.int32 and out.tolist() == [0, 0, 0]


def test_estimator_receives_run_constants_and_per_pass_operands(monkeypatch):
    recorded = {}
    monkeypatch.setattr(iteration_loop, "estimate_relion_expected_accuracy", lambda **kw: recorded.update(kw) or "acc")
    ea = SimpleNamespace(half1_particle_ids="pids", half1_ctf_params="ctf", do_ctf_correction=False)
    inputs = iteration_loop._ExpectedAccuracyInputs(
        trial_order_local="order", dataset="half1", volume_shape=[8, 8, 8], padding_factor=2, tau2_fudge=np.float64(4.0), optimizer_random_seed=np.int64(11), expected_accuracy=ea,
    )
    out = iteration_loop._estimate_half1_expected_accuracy(
        inputs, reference_fourier="ref", best_eulers_deg="eul", class_ids="ids", class_weights="w", sigma2_noise_native="noise", current_image_size=np.int64(56),
    )
    assert out == "acc"
    assert recorded == dict(
        reference_fourier="ref", volume_shape=(8, 8, 8), best_eulers_deg="eul", class_ids="ids", class_weights="w", sigma2_noise_native="noise", dataset="half1",
        trial_order_local="order", current_image_size=56, padding_factor=2, sigma2_fudge=4.0, random_seed=11, random_seed_particle_ids="pids", ctf_params_override="ctf", do_ctf_correction=False,
    )
    assert type(recorded["current_image_size"]) is int and type(recorded["sigma2_fudge"]) is float and type(recorded["random_seed"]) is int


def test_controller_estimates_accuracy_through_the_owner_in_both_passes():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    assert source.count("_estimate_half1_expected_accuracy(") == 2
    assert source.count("_expected_accuracy_class_ids(") == 2
    assert source.count("_ExpectedAccuracyInputs(") == 1
    assert "estimate_relion_expected_accuracy(" not in source
