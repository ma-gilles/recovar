"""BPref capacity ABI must preserve legacy texture/scatter words and geometry."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from recovar import cuda_backproject as cb
from recovar.em.helpers.projection import prepare_relion_projector_capacity, relion_projector_half_to_texture_full

pytestmark = pytest.mark.unit
FUNCTION = "relion_vdam_mstep_fused_projector_x_half"
OLD_STATIC_POSITIONS = (10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28)


def _arguments(pf=1, grouped=False):
    """One active pixel/rotation per reconstruction group avoids atomic collisions."""
    b, r, t = (2, 2, 3) if grouped else (1, 1, 3)
    n, q = 32, 8
    volume = (q + 3,) * 3
    volume_count = volume[0] * volume[1] * (volume[2] // 2 + 1)
    carry_shape = (2, volume_count) if grouped else (volume_count,)
    carry = np.arange(np.prod(carry_shape), dtype=np.float32).reshape(carry_shape) / 1024 + 0.25
    posterior = np.zeros((b, r, t), np.float32)
    posterior[:, 0] = [0.125, 0.25, 0.0625]
    size = pf * q + 3
    arguments = [
        (carry + 1j * (carry + 0.125)).astype(np.complex64),
        carry.copy(),
        np.full((b, 1), 1.25 - 0.625j, np.complex64),
        np.full((b, 1), -0.75, np.float32),
        np.full((b, 1), 1.5, np.float32),
        posterior,
        np.asarray([[0, 0], [0.01, -0.02], [-0.03, 0.015]], np.float32),
        np.asarray([(n // 2) * (n // 2 + 1) + 1], np.int32),
        np.zeros((size, size, size // 2 + 1), np.complex64),
        np.broadcast_to(np.eye(3, dtype=np.float32), (b, r, 3, 3)).copy(),
        (n, n),
        volume,
        float(q // 2),
        0,
        pf,
    ]
    options = dict(
        stable_dense_positions=np.asarray([1], np.int32),
        logical_current_size=np.asarray(q, np.int32),
        runtime_projector_radius=np.asarray(3, np.int32),
    )
    if grouped:
        options.update(
            reconstruction_group_ids=np.arange(b, dtype=np.int32), worker_lane_ids=np.arange(b, dtype=np.int32)
        )
    return arguments, options


def _no_cuda(monkeypatch):
    monkeypatch.setattr(cb, "_ensure_ffi", lambda: pytest.fail("CUDA loaded before validation"))
    monkeypatch.setattr(
        cb, "_ensure_bpref_projector_capacity_ffi", lambda: pytest.fail("capacity CUDA loaded before validation")
    )


@pytest.mark.parametrize("value", [np.asarray(3, np.int64), np.asarray(3, np.float32), np.asarray([3], np.int32), 3])
def test_runtime_radius_requires_strong_s32_scalar(monkeypatch, value):
    _no_cuda(monkeypatch)
    args, options = _arguments()
    options["runtime_projector_radius"] = value
    with jax.enable_x64(False), pytest.raises((TypeError, ValueError)):
        getattr(cb, FUNCTION).__wrapped__(*args, **options)


@pytest.mark.parametrize("shape", [(11, 11, 11), (11, 9, 6), (10, 10, 6), (11, 11, 5)])
def test_capacity_storage_topology_rejected_before_cuda(monkeypatch, shape):
    _no_cuda(monkeypatch)
    args, options = _arguments()
    args[8] = np.zeros(shape, np.complex64)
    with pytest.raises((TypeError, ValueError)):
        getattr(cb, FUNCTION).__wrapped__(*args, **options)


@pytest.mark.parametrize("dtype", [np.float32, np.complex128])
def test_capacity_storage_dtype_rejected_before_cuda(monkeypatch, dtype):
    _no_cuda(monkeypatch)
    args, options = _arguments()
    args[8] = np.zeros(args[8].shape, dtype)
    with jax.enable_x64(False), pytest.raises((TypeError, ValueError)):
        getattr(cb, FUNCTION).__wrapped__(*args, **options)


@pytest.mark.parametrize("failure", ["static_radius", "padding", "positions", "logical_size"])
def test_capacity_static_and_runtime_image_contract(monkeypatch, failure):
    _no_cuda(monkeypatch)
    args, options = _arguments()
    if failure == "static_radius":
        args[13] = 3
    elif failure == "padding":
        args[14] = 3
    elif failure == "positions":
        options["stable_dense_positions"] = None
    else:
        options["logical_current_size"] = None
    with pytest.raises((TypeError, ValueError)):
        getattr(cb, FUNCTION).__wrapped__(*args, **options)


def test_runtime_radius_appended_without_changing_legacy_static_positions():
    source = Path(cb.__file__).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
    static = [
        ast.literal_eval(k.value)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        for k in decorator.keywords
        if k.arg == "static_argnums"
    ]
    parameters = inspect.signature(getattr(cb, FUNCTION)).parameters
    assert static == [OLD_STATIC_POSITIONS + (32, 33)]
    assert list(parameters).index("runtime_projector_radius") == 31
    assert list(parameters)[32] == "return_denominator"
    assert parameters["return_denominator"].default is True
    assert parameters["runtime_projector_radius"].default is None
    assert list(parameters).index("runtime_projector_radius") not in OLD_STATIC_POSITIONS


def _bits_equal(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def _logical_and_poisoned_capacity(radius, pf):
    size = 2 * pf * radius + 3
    rng = np.random.default_rng(29061 + pf * 100 + radius)
    logical = (rng.normal(size=(size, size, size // 2 + 1)) + 1j * rng.normal(size=(size, size, size // 2 + 1))).astype(
        np.complex64
    )
    # Ghost planes are deliberately nonzero. Capacity outside the logical extent
    # is poisoned, so reusing physical origin/support cannot silently pass.
    capacity, dynamic_radius = prepare_relion_projector_capacity(
        jnp.asarray(logical), r_max=radius, physical_size=8, padding_factor=pf
    )
    physical = np.asarray(capacity).copy()
    offset = (physical.shape[0] - size) // 2
    logical_region = (slice(offset, offset + size), slice(offset, offset + size), slice(0, size // 2 + 1))
    outside = np.ones(physical.shape, bool)
    outside[logical_region] = False
    physical[outside] = np.complex64(1e9 - 1e9j)
    _bits_equal(physical[logical_region], logical)
    return logical, jnp.asarray(physical), dynamic_radius


def _device_call(args, options):
    # All legacy statics are kept as Python scalars/tuples.
    return getattr(cb, FUNCTION)(
        *[jnp.asarray(v) if isinstance(v, np.ndarray) else v for v in args],
        **{k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in options.items()},
    )


@pytest.mark.gpu
@pytest.mark.parametrize("pf", [1, 2])
@pytest.mark.parametrize("grouped", [False, True])
def test_gpu_capacity_matches_legacy_all_three_outputs_and_one_executable(pf, grouped):
    assert jax.default_backend() == "gpu"
    cb._ensure_ffi()
    cases = []
    # FFTW +Nyquist, x-half origin, negative y, sphere boundary/outside border;
    # each launch contains one active pixel, preventing floating atomic races.
    coordinates = [(0, 1), (4, 0), (0, 4), (-3, 1), (4, 1)]
    rotations = [
        np.eye(3, dtype=np.float32),
        Rotation.from_euler("z", np.pi).as_matrix().astype(np.float32),
        Rotation.from_euler("zyx", [1 / 256, -0.13, 0.21]).as_matrix().astype(np.float32),
    ]
    for radius in (2, 3, 4):
        logical, capacity, dynamic_radius = _logical_and_poisoned_capacity(radius, pf)
        full = relion_projector_half_to_texture_full(jnp.asarray(logical))
        for index, (y, x) in enumerate(coordinates):
            args, options = _arguments(pf, grouped)
            args[7] = np.asarray([(y + 16) * 17 + x], np.int32)
            args[9][...] = rotations[index % len(rotations)]
            options["stable_dense_positions"] = np.asarray([(y % 8) * 5 + x], np.int32)
            legacy_args = list(args)
            legacy_args[8], legacy_args[13] = full, radius
            legacy_options = {k: v for k, v in options.items() if k != "runtime_projector_radius"}
            reference = tuple(np.asarray(v) for v in _device_call(legacy_args, legacy_options))
            repeat = tuple(np.asarray(v) for v in _device_call(legacy_args, legacy_options))
            for a, b in zip(reference, repeat, strict=True):
                _bits_equal(a, b)
            args[8] = capacity
            options["runtime_projector_radius"] = dynamic_radius
            cases.append((args, options, reference))
    function = getattr(cb, FUNCTION)
    function.clear_cache()
    for args, options, reference in cases:
        result = _device_call(args, options)
        for a, b in zip(result, reference, strict=True):
            _bits_equal(a, b)
        assert result[0].shape == args[0].shape and result[1].shape == args[1].shape
        assert result[2].shape == (*args[5].shape[:2], 1)
        assert function._cache_size() == 1
    # Verify this exercises real accumulation and nonzero denominator, not an
    # all-masked equality. The first coordinate lies inside every logical disk.
    assert np.any(cases[0][2][0] != cases[0][0][0])
    assert np.any(cases[0][2][1] != cases[0][0][1])
    assert np.any(cases[0][2][2] != 0)
    function.clear_cache()


def test_optional_registration_preserves_qualified_legacy_library(monkeypatch):
    symbol = "RelionVdamMstepFusedProjectorCapacityXHalf"
    legacy = {name: object() for _, name in cb._FFI_REGISTRATIONS}
    assert symbol not in legacy
    library = SimpleNamespace(**legacy)
    registrations = []
    monkeypatch.setattr(cb, "_ffi_registered", False)
    monkeypatch.setattr(cb, "_bpref_projector_capacity_ffi_registered", False)
    monkeypatch.setattr(cb, "_loaded_lib_path", None)
    monkeypatch.setattr(cb, "_get_lib", lambda: library)
    monkeypatch.setattr(jax.ffi, "pycapsule", lambda value: value)
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda target, *args, **kwargs: registrations.append(target))
    cb._ensure_ffi()
    assert registrations == [target for target, _ in cb._FFI_REGISTRATIONS]
    with pytest.raises(RuntimeError, match=symbol):
        cb._ensure_bpref_projector_capacity_ffi()
    assert cb._ffi_registered and not cb._bpref_projector_capacity_ffi_registered
    setattr(library, symbol, object())
    cb._ensure_bpref_projector_capacity_ffi()
    cb._ensure_bpref_projector_capacity_ffi()
    assert registrations.count(cb._TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_CAPACITY_X_HALF) == 1


def test_capacity_radius_operand_preserves_aliases_geometry_and_one_trace(monkeypatch):
    monkeypatch.delenv("RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY", raising=False)
    monkeypatch.setattr(cb, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cb, "_ensure_bpref_projector_capacity_ffi", lambda: None)
    records = []

    def fake_ffi(target, outputs, **options):
        def call(*operands, **attrs):
            assert len(operands) == 19
            radius = operands[18]
            records.append((target, radius.aval, attrs, options, tuple(x.shape for x in outputs)))
            return (*operands[14:17], jnp.full(outputs[3].shape, radius, jnp.float32))

        return call

    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi)
    function = getattr(cb, FUNCTION)
    function.clear_cache()
    args, options = _arguments()
    try:
        for radius in (2, 3, 4):
            options["runtime_projector_radius"] = np.asarray(radius, np.int32)
            result = _device_call(args, options)
            _bits_equal(result[0], args[0])
            _bits_equal(result[1], args[1])
            np.testing.assert_array_equal(np.asarray(result[2]), np.full((1, 1, 1), radius, np.float32))
        assert function._cache_size() == 1 and len(records) == 1
        target, radius, attrs, ffi_options, output_shapes = records[0]
        assert target == cb._TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_CAPACITY_X_HALF
        assert radius.shape == () and radius.dtype == np.dtype(np.int32) and not radius.weak_type
        assert attrs["projector_max_r"] == 0 and attrs["projection_padding_factor"] == 1
        assert ffi_options["input_output_aliases"] == {14: 0, 15: 1, 16: 2}
        assert output_shapes[-1] == (1, 1, 40)
    finally:
        function.clear_cache()


@pytest.mark.parametrize(
    "name,value",
    [
        ("rotation_replay_order", np.zeros((1, 1), np.int32)),
        ("rotation_replay_counts", np.ones(1, np.int32)),
        ("particle_start_offsets_ns", np.zeros(1, np.int32)),
        ("serial_rotation_replay", True),
        ("float64_accumulator_replay", True),
        ("reverse_rotation_replay", True),
        ("rotation_replay_stride", 1),
        ("native_trace_shape_replay", True),
        ("candidate_trace_active", True),
        ("persistent_serial_rotation_replay", True),
    ],
)
def test_capacity_unsupported_replay_modes_fail_before_cuda(monkeypatch, name, value):
    _no_cuda(monkeypatch)
    args, options = _arguments()
    options[name] = value
    with pytest.raises(ValueError, match="replay/trace"):
        getattr(cb, FUNCTION).__wrapped__(*args, **options)


@pytest.mark.gpu
@pytest.mark.parametrize("radius", [-1, 0, 5, np.iinfo(np.int32).max])
def test_gpu_runtime_radius_outside_capacity_fails_before_scatter(radius):
    assert jax.default_backend() == "gpu"
    args, options = _arguments()
    options["runtime_projector_radius"] = np.asarray(radius, np.int32)
    with pytest.raises(RuntimeError, match="(?i)invalid"):
        jax.block_until_ready(_device_call(args, options))
