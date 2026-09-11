"""The state-swap probe returns one named value instead of positional 14-tuples."""

import inspect
from types import SimpleNamespace

import numpy as np

from recovar.em.dense_single_volume.helpers import state_swap_runtime


def _inputs():
    half_inputs = SimpleNamespace(
        image_corrections=[np.array([1.0]), np.array([2.0])],
        scale_corrections=[np.array([3.0]), np.array([4.0])],
        previous_best_translations=[np.zeros((1, 2)), np.zeros((1, 2))],
        previous_best_rotation_eulers=[np.zeros((1, 3)), np.zeros((1, 3))],
    )
    return dict(
        state=SimpleNamespace(a=1), cs=52, volume_shape=(1, 1, 1), means=[np.array([10.0]), np.array([20.0])],
        mean_variance=np.array([50.0]), noise_variance_per_half=[np.array([60.0]), np.array([70.0])], noise_variance=np.array([65.0]),
        previous_noise_radial_per_half=[np.array([80.0]), np.array([90.0])], previous_noise_radial=np.array([85.0]),
        relion_half_inputs=half_inputs, previous_best_rotations=[np.eye(3)[None], np.eye(3)[None]], current_sigma_offset_angstrom=3.1,
        current_sigma_offset_angstrom_per_half=[2.9, 3.3], class_direction_prior_per_half=[np.array([0.5]), np.array([0.5])],
        class_direction_prior_order_per_half=[4, 4], global_direction_prior_per_half=[np.array([0.5]), np.array([0.5])],
        global_direction_prior_order_per_half=[4, 4],
    )


def test_state_swap_values_keep_the_controller_order():
    assert state_swap_runtime._StateSwapValues._fields == (
        "cs", "means", "mean_variance", "noise_variance_per_half", "noise_variance", "previous_noise_radial_per_half",
        "previous_noise_radial", "previous_best_rotations", "current_sigma_offset_angstrom",
        "current_sigma_offset_angstrom_per_half", "class_direction_prior_per_half", "class_direction_prior_order_per_half",
        "global_direction_prior_per_half", "global_direction_prior_order_per_half",
    )


def test_unchanged_paths_return_the_input_objects():
    kw = _inputs()
    for probe, snapshot in ((None, {"x": 1}), ({"iteration": 6}, None), ({"iteration": 7, "variant": "recovar_sigma_offset"}, {"x": 1})):
        out = state_swap_runtime._apply_state_swap_probe(probe=probe, iteration=6, recovar_snapshot=snapshot, **kw)
        assert isinstance(out, state_swap_runtime._StateSwapValues)
        assert out.cs is kw["cs"] and out.means is kw["means"] and out.mean_variance is kw["mean_variance"]
        assert out.current_sigma_offset_angstrom_per_half is kw["current_sigma_offset_angstrom_per_half"]
        assert out[8] == 3.1 and len(out) == 14


def test_probe_no_longer_builds_positional_tuples():
    source = inspect.getsource(state_swap_runtime._apply_state_swap_probe)
    assert "_state_swap_return_tuple" not in source
    assert source.count("unchanged = _StateSwapValues(") == 1
    assert source.count("return unchanged") == 2
    assert source.count("return _StateSwapValues(") == 1
    assert not hasattr(state_swap_runtime, "_state_swap_return_tuple")
