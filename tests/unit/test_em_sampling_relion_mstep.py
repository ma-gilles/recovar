"""Exact host Euler-matrix parity guards for RELION M-step rotations."""

import inspect

import numpy as np

from recovar.em.dense_single_volume import iteration_loop as iteration_loop_module
from recovar.em.dense_single_volume import relion_metadata
from recovar.em.sampling import (
    _relion_adaptive_pass1_rotations,
    _relion_mstep_rotations_from_eulers,
    apply_relion_rotation_perturbation_to_eulers,
    get_oversampled_rotation_grid_from_samples,
    relion_sampling_perturbation_for_iteration,
)


def test_adaptive_pass1_routes_source_eulers_and_host_right_matrix_to_cuda_builder(monkeypatch):
    from recovar.em import sampling as sampling_module

    source_eulers = _UNPERTURBED_FINE_EULERS_F64[:2]
    sentinel = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    calls = []

    def fake_builder(eulers_deg, right_matrix=None):
        calls.append((np.asarray(eulers_deg), np.asarray(right_matrix)))
        return sentinel

    monkeypatch.setattr(sampling_module, "_relion_device_scoring_rotations_f32", fake_builder)
    result = _relion_adaptive_pass1_rotations(
        source_eulers,
        random_perturbation=-0.455874443054,
        angular_sampling_deg=7.5,
    )

    np.testing.assert_array_equal(result, sentinel)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], source_eulers)
    perturbation_deg = -0.455874443054 * 7.5
    expected_right = sampling_module._relion_euler_angles_to_matrix(
        np.asarray([[perturbation_deg] * 3], dtype=np.float64)
    )[0]
    np.testing.assert_array_equal(calls[0][1], expected_right)


def test_adaptive_pass1_omits_right_matrix_without_perturbation(monkeypatch):
    from recovar.em import sampling as sampling_module

    seen = []

    def fake_builder(eulers_deg, right_matrix=None):
        seen.append(right_matrix)
        return np.zeros((len(eulers_deg), 3, 3), dtype=np.float32)

    monkeypatch.setattr(sampling_module, "_relion_device_scoring_rotations_f32", fake_builder)
    _relion_adaptive_pass1_rotations(_UNPERTURBED_FINE_EULERS_F64[:1], 0.0, 7.5)
    assert seen == [None]


def test_adaptive_pass1_float64_routes_to_double_precision_builder(monkeypatch):
    """``use_float64=True`` must dispatch to the double-precision builder, not the CUDA f32 one."""
    from recovar.em import sampling as sampling_module

    source_eulers = _UNPERTURBED_FINE_EULERS_F64[:2]
    sentinel = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    f32_calls = []
    f64_calls = []

    monkeypatch.setattr(
        sampling_module,
        "_relion_device_scoring_rotations_f32",
        lambda *a, **k: f32_calls.append((a, k)) or None,
    )

    def fake_f64_builder(eulers_deg, right_matrix=None):
        f64_calls.append((np.asarray(eulers_deg), np.asarray(right_matrix) if right_matrix is not None else None))
        return sentinel

    monkeypatch.setattr(sampling_module, "_relion_device_scoring_rotations_f64", fake_f64_builder)
    result = _relion_adaptive_pass1_rotations(
        source_eulers,
        random_perturbation=-0.455874443054,
        angular_sampling_deg=7.5,
        use_float64=True,
    )

    np.testing.assert_array_equal(result, sentinel)
    assert len(f64_calls) == 1
    assert len(f32_calls) == 0
    np.testing.assert_array_equal(f64_calls[0][0], source_eulers)


def test_relion_device_scoring_rotations_f64_matches_euler_matrix_port():
    """No perturbation: f64 builder must reduce to ``_relion_euler_angles_to_matrix`` exactly."""
    from recovar.em.sampling import (
        _relion_device_scoring_rotations_f64,
        _relion_euler_angles_to_matrix,
    )

    source_eulers = _UNPERTURBED_FINE_EULERS_F64
    result = _relion_device_scoring_rotations_f64(source_eulers, right_matrix=None)
    expected = _relion_euler_angles_to_matrix(source_eulers)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)


def test_relion_device_scoring_rotations_f64_right_multiplies_perturbation_matrix():
    """With a perturbation: f64 builder must right-multiply, matching ``B = A @ right_matrix``
    from ``cuda_kernel_make_eulers_3D`` (``recovar/cuda/cuda_backproject.cu``)."""
    from recovar.em.sampling import (
        _relion_device_scoring_rotations_f64,
        _relion_euler_angles_to_matrix,
    )

    source_eulers = _UNPERTURBED_FINE_EULERS_F64[:2]
    right_matrix = _relion_euler_angles_to_matrix(np.asarray([[1.5, 1.5, 1.5]], dtype=np.float64))[0]

    result = _relion_device_scoring_rotations_f64(source_eulers, right_matrix=right_matrix)
    expected = _relion_euler_angles_to_matrix(source_eulers) @ right_matrix

    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.0)


# Five captured RELION iteration-1 winner rows that previously changed the
# outer-radius predicate. These are host RFLOAT Euler rows before the final
# XFLOAT cast in generateEulerMatrices().
_RELION_FINE_EULERS_F64 = np.asarray(
    [
        [-148.07116258510126, 159.73799185037856, 6.190743060629692],
        [-65.48923578673723, 80.24919217586016, 103.4673527691381],
        [86.6136702414223, 65.35556221495885, -13.533766806566508],
        [-121.68325289369227, 150.62937446903518, -1.2306848410672946],
        [-27.57463869823777, 134.66318059851523, 35.865537006317815],
    ],
    dtype=np.float64,
)

# RECOVAR-facing transpose(inv(A)) bit patterns captured from RELION's
# runWavgKernel Euler input for the same rows.
_RELION_MSTEP_ROTATION_BITS = np.asarray(
    [
        [1062812740, 1053666949, 3199223633, 1054948714, 3211113161, 1025046566, 3197533304, 3191573647, 3211798669],
        [1063147020, 1054929375, 1047202070, 3197074155, 1029314506, 1064656185, 1053906824, 3211104430, 1043164569],
        [1048829552, 1053303020, 3210885245, 3212245813, 1042189919, 3193556645, 1029433292, 1063798404, 1054179116],
        [1054928899, 1061204141, 3204124251, 1063014457, 3204602172, 3157038757, 3196314638, 3201675372, 3210680411],
        [3194968473, 1061713058, 3205729934, 1060994388, 1057429299, 1054169910, 1059153352, 3198718531, 3207852987],
    ],
    dtype=np.uint32,
).reshape(-1, 3, 3)

_UNPERTURBED_FINE_EULERS_F64 = np.asarray(
    [
        [212.14285714285714, 159.42254649458224, 5.625],
        [295.3125, 80.40593177313954, 103.125],
        [87.18749999999999, 65.37568164783592, 346.875],
        [238.50000000000003, 150.4344388449523, -1.875],
        [333.0, 134.99388015045713, 35.625],
    ],
    dtype=np.float64,
)

# Five iteration-10 K=4 weighted-sum matrices captured from RELION after a
# continuation from run_it009. The sampling STAR rounds the live perturbation
# to -0.12306, while the seed-exact value is -0.12305957078933716.
_K4_RESTART_PARENT_CHILD = np.asarray(
    [
        [293, 4],
        [342, 2],
        [342, 4],
        [342, 7],
        [90, 3],
    ],
    dtype=np.int64,
)
_K4_RESTART_MSTEP_ROTATION_BITS = np.asarray(
    [
        [1052386263, 3204951217, 1061429477, 1060489267, 1060152046, 1041217251, 3206176265, 1056729477, 1059098305],
        [1058153064, 3200386585, 1060796210, 1060846933, 1059276408, 3195481418, 3200189669, 1059825665, 1059334299],
        [1052875775, 3205887686, 1060602466, 1064029067, 1050783034, 3194572054, 3183361835, 1061098689, 1059631691],
        [1059608873, 3205391353, 1057100669, 1061027885, 1058542104, 3198083018, 3187955380, 1058326438, 1062055714],
        [1061673231, 3198169958, 3205136030, 3173386773, 3210977500, 1055480547, 3206522678, 3198873872, 3207918152],
    ],
    dtype=np.uint32,
).reshape(-1, 3, 3)


def test_relion_mstep_rotation_helper_matches_captured_float32_bits():
    rotations = _relion_mstep_rotations_from_eulers(_RELION_FINE_EULERS_F64)

    assert rotations.dtype == np.float32
    np.testing.assert_array_equal(rotations.view(np.uint32), _RELION_MSTEP_ROTATION_BITS)


def test_k4_restart_uses_seed_exact_perturbation_for_captured_mstep_bits():
    exact_perturbation = relion_sampling_perturbation_for_iteration(
        0.5,
        1778628798,
        10,
        restart_state_iteration=9,
    )
    assert exact_perturbation == -0.12305957078933716

    exact_rows = []
    rounded_rows = []
    for parent, child in _K4_RESTART_PARENT_CHILD:
        exact, _ = get_oversampled_rotation_grid_from_samples(
            np.asarray([parent]),
            parent_nside_level=1,
            oversampling_order=1,
            random_perturbation=exact_perturbation,
            rotation_index_order="recovar",
        )
        rounded, _ = get_oversampled_rotation_grid_from_samples(
            np.asarray([parent]),
            parent_nside_level=1,
            oversampling_order=1,
            random_perturbation=-0.12306,
            rotation_index_order="recovar",
        )
        exact_rows.append(exact[child])
        rounded_rows.append(rounded[child])

    exact_rows = np.asarray(exact_rows, dtype=np.float32)
    rounded_rows = np.asarray(rounded_rows, dtype=np.float32)
    np.testing.assert_array_equal(
        exact_rows.view(np.uint32),
        _K4_RESTART_MSTEP_ROTATION_BITS,
    )
    assert np.any(
        rounded_rows.view(np.uint32) != _K4_RESTART_MSTEP_ROTATION_BITS
    )


def test_relion_mstep_rotation_helper_preserves_matrix2d_inverse_source_order():
    source = inspect.getsource(_relion_mstep_rotations_from_eulers)
    cofactor_assignments = [
        "inverse[:, 0, 0] = matrix[:, 2, 2] * matrix[:, 1, 1] - matrix[:, 2, 1] * matrix[:, 1, 2]",
        "inverse[:, 0, 1] = -(matrix[:, 2, 2] * matrix[:, 0, 1] - matrix[:, 2, 1] * matrix[:, 0, 2])",
        "inverse[:, 0, 2] = matrix[:, 1, 2] * matrix[:, 0, 1] - matrix[:, 1, 1] * matrix[:, 0, 2]",
        "inverse[:, 1, 0] = -(matrix[:, 2, 2] * matrix[:, 1, 0] - matrix[:, 2, 0] * matrix[:, 1, 2])",
        "inverse[:, 1, 1] = matrix[:, 2, 2] * matrix[:, 0, 0] - matrix[:, 2, 0] * matrix[:, 0, 2]",
        "inverse[:, 1, 2] = -(matrix[:, 1, 2] * matrix[:, 0, 0] - matrix[:, 1, 0] * matrix[:, 0, 2])",
        "inverse[:, 2, 0] = matrix[:, 2, 1] * matrix[:, 1, 0] - matrix[:, 2, 0] * matrix[:, 1, 1]",
        "inverse[:, 2, 1] = -(matrix[:, 2, 1] * matrix[:, 0, 0] - matrix[:, 2, 0] * matrix[:, 0, 1])",
        "inverse[:, 2, 2] = matrix[:, 1, 1] * matrix[:, 0, 0] - matrix[:, 1, 0] * matrix[:, 0, 1]",
    ]
    assert all(assignment in source for assignment in cofactor_assignments)
    assert "np.linalg" not in source
    assert source.index("determinant = (") < source.index("inverse /= determinant")
    assert source.index("inverse /= determinant") < source.index("return np.swapaxes(inverse, 1, 2).astype(dtype)")


def test_relion_mstep_rotation_helper_defaults_to_float32_but_accepts_float64():
    """``dtype`` only changes the final cast -- default stays bit-identical to before."""
    default_result = _relion_mstep_rotations_from_eulers(_RELION_FINE_EULERS_F64)
    assert default_result.dtype == np.float32
    np.testing.assert_array_equal(default_result.view(np.uint32), _RELION_MSTEP_ROTATION_BITS)

    f64_result = _relion_mstep_rotations_from_eulers(_RELION_FINE_EULERS_F64, dtype=np.float64)
    assert f64_result.dtype == np.float64
    np.testing.assert_allclose(f64_result.astype(np.float32), default_result, rtol=0.0, atol=1e-6)


def test_perturbation_optional_mstep_return_keeps_legacy_tuple_and_float64_working_eulers():
    random_perturbation = -0.04961434006690979
    legacy_rotations, legacy_eulers = apply_relion_rotation_perturbation_to_eulers(
        _UNPERTURBED_FINE_EULERS_F64,
        random_perturbation,
        7.5,
    )
    rotations, eulers, mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
        _UNPERTURBED_FINE_EULERS_F64,
        random_perturbation,
        7.5,
        return_mstep_rotations=True,
    )

    np.testing.assert_array_equal(rotations.view(np.uint32), legacy_rotations.view(np.uint32))
    np.testing.assert_array_equal(eulers.view(np.uint32), legacy_eulers.view(np.uint32))
    np.testing.assert_array_equal(mstep_rotations.view(np.uint32), _RELION_MSTEP_ROTATION_BITS)

    # Reconstructing after the public Euler metadata is cast to float32 loses
    # the host RFLOAT bits required at RELION's outer-radius boundary.
    late_mstep_rotations = _relion_mstep_rotations_from_eulers(eulers)
    assert np.count_nonzero(late_mstep_rotations.view(np.uint32) != _RELION_MSTEP_ROTATION_BITS) == 26


def test_unperturbed_optional_mstep_return_is_backward_compatible():
    legacy = apply_relion_rotation_perturbation_to_eulers(
        _UNPERTURBED_FINE_EULERS_F64[:1],
        0.0,
        7.5,
    )
    extended = apply_relion_rotation_perturbation_to_eulers(
        _UNPERTURBED_FINE_EULERS_F64[:1],
        0.0,
        7.5,
        return_mstep_rotations=True,
    )

    assert len(legacy) == 2
    assert len(extended) == 3
    np.testing.assert_array_equal(extended[0], legacy[0])
    np.testing.assert_array_equal(extended[1], legacy[1])
    np.testing.assert_array_equal(
        extended[2],
        _relion_mstep_rotations_from_eulers(_UNPERTURBED_FINE_EULERS_F64[:1]),
    )


def test_perturbed_scorer_uses_captured_host_generated_matrix_bits():
    random_perturbation = np.float64(-0.04961434006690979)
    rotations, public_eulers, mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
        _UNPERTURBED_FINE_EULERS_F64,
        random_perturbation,
        7.5,
        return_mstep_rotations=True,
    )

    np.testing.assert_array_equal(rotations.view(np.uint32), _RELION_MSTEP_ROTATION_BITS)
    np.testing.assert_array_equal(mstep_rotations.view(np.uint32), _RELION_MSTEP_ROTATION_BITS)
    np.testing.assert_array_equal(rotations, mstep_rotations)
    assert public_eulers.dtype == np.float32
    np.testing.assert_array_equal(public_eulers, _RELION_FINE_EULERS_F64.astype(np.float32))


def test_unperturbed_scorer_and_mstep_share_host_generated_path():
    rotations, _, mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
        _UNPERTURBED_FINE_EULERS_F64[:1],
        0.0,
        7.5,
        return_mstep_rotations=True,
    )

    expected = _relion_mstep_rotations_from_eulers(_UNPERTURBED_FINE_EULERS_F64[:1])
    np.testing.assert_array_equal(rotations, expected)
    np.testing.assert_array_equal(mstep_rotations, expected)


def test_relion_global_grid_preserves_source_euler_precision_until_matrix_cast(monkeypatch):
    source_eulers = _UNPERTURBED_FINE_EULERS_F64[:2]
    monkeypatch.setattr(
        iteration_loop_module,
        "_get_relion_rotation_grid_eulers_float64",
        lambda _order: source_eulers,
    )

    rotations, returned_eulers = relion_metadata._relion_rotation_grid_float32(3)

    np.testing.assert_array_equal(rotations, _relion_mstep_rotations_from_eulers(source_eulers))
    np.testing.assert_array_equal(returned_eulers, source_eulers.astype(np.float32))
    late_cast_rotations = _relion_mstep_rotations_from_eulers(returned_eulers)
    assert np.any(rotations.view(np.uint32) != late_cast_rotations.view(np.uint32))


def test_oversampled_grid_optionally_returns_mstep_rotations_without_changing_score_grid():
    legacy_rotations, legacy_parent_map, legacy_child_indices = get_oversampled_rotation_grid_from_samples(
        np.array([0, 7], dtype=np.int64),
        parent_nside_level=2,
        oversampling_order=1,
        random_perturbation=-0.11648395657539368,
        return_rotation_indices=True,
        rotation_index_order="relion",
    )
    rotations, parent_map, child_indices, mstep_rotations = get_oversampled_rotation_grid_from_samples(
        np.array([0, 7], dtype=np.int64),
        parent_nside_level=2,
        oversampling_order=1,
        random_perturbation=-0.11648395657539368,
        return_rotation_indices=True,
        return_mstep_rotations=True,
        rotation_index_order="relion",
    )

    np.testing.assert_array_equal(rotations.view(np.uint32), legacy_rotations.view(np.uint32))
    np.testing.assert_array_equal(parent_map, legacy_parent_map)
    np.testing.assert_array_equal(child_indices, legacy_child_indices)
    assert rotations.dtype == np.float32
    assert mstep_rotations.dtype == np.float32
    assert mstep_rotations.shape == rotations.shape
    np.testing.assert_array_equal(mstep_rotations, rotations)


def test_oversampled_grid_empty_optional_return_order_is_stable():
    rotations, parent_map, child_indices, mstep_rotations = get_oversampled_rotation_grid_from_samples(
        np.empty((0,), dtype=np.int64),
        parent_nside_level=2,
        return_rotation_indices=True,
        return_mstep_rotations=True,
    )

    assert rotations.shape == (0, 3, 3)
    assert parent_map.shape == (0,)
    assert child_indices.shape == (0,)
    assert mstep_rotations.shape == (0, 3, 3)
