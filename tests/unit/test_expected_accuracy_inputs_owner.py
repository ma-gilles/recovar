"""Both passes estimate RELION's expected accuracy through one inputs owner and one class-label rule."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.helpers import expected_accuracy as owner

pytestmark = pytest.mark.unit


def test_class_labels_use_k_class_assignments_when_present():
    ids = np.asarray([0, 1, 2], dtype=np.int32)
    assert owner._expected_accuracy_class_ids(ids, k_class_enabled=True, n_units=3) is ids


@pytest.mark.parametrize("k_class,assignments", [(True, None), (False, None), (False, np.asarray([1, 1, 1], dtype=np.int32))])
def test_class_labels_fall_back_to_class_zero(k_class, assignments):
    out = owner._expected_accuracy_class_ids(assignments, k_class_enabled=k_class, n_units=3)
    assert out.dtype == np.int32 and out.tolist() == [0, 0, 0]


def test_estimator_receives_run_constants_and_per_pass_operands(monkeypatch):
    recorded = {}
    monkeypatch.setattr(owner, "estimate_relion_expected_accuracy", lambda **kw: recorded.update(kw) or "acc")
    ea = SimpleNamespace(half1_particle_ids="pids", half1_ctf_params="ctf", do_ctf_correction=False)
    inputs = owner.Half1AccuracyInputs(
        trial_order_local="order", dataset="half1", volume_shape=[8, 8, 8], padding_factor=2, tau2_fudge=np.float64(4.0), optimizer_random_seed=np.int64(11), expected_accuracy=ea,
    )
    out = inputs.estimate( reference_fourier="ref", best_eulers_deg="eul", class_ids="ids", class_weights="w", sigma2_noise_native="noise", current_image_size=np.int64(56),
    )
    assert out == "acc"
    assert recorded == dict(
        reference_fourier="ref", volume_shape=(8, 8, 8), best_eulers_deg="eul", class_ids="ids", class_weights="w", sigma2_noise_native="noise", dataset="half1",
        trial_order_local="order", current_image_size=56, padding_factor=2, sigma2_fudge=4.0, random_seed=11, random_seed_particle_ids="pids", ctf_params_override="ctf", do_ctf_correction=False,
    )
    assert type(recorded["current_image_size"]) is int and type(recorded["sigma2_fudge"]) is float and type(recorded["random_seed"]) is int
