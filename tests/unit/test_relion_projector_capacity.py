"""Runtime-radius texture ABI: validation, trace contract and GPU old/new oracle."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from recovar import cuda_backproject as cb
from recovar.em.helpers.projection import relion_projector_half_to_texture_full

pytestmark = pytest.mark.unit


def _inputs(q=32, pf=1):
    size = pf * q + 3
    return (
        jnp.zeros((size, size, size // 2 + 1), jnp.complex64),
        jnp.eye(3, dtype=jnp.float32)[None],
        jnp.asarray(15, jnp.int32),
    )


@pytest.mark.parametrize(
    "which,value,message",
    [
        (0, np.zeros((35, 35, 18), np.complex128), "projector_half"),
        (0, np.zeros((35, 33, 18), np.complex64), "projector_half"),
        (0, np.zeros((34, 34, 18), np.complex64), "projector_half"),
        (0, np.zeros((35, 35, 17), np.complex64), "projector_half"),
        (1, np.eye(3, dtype=np.float32), "rotation_matrices"),
        (1, np.eye(3, dtype=np.float64)[None], "rotation_matrices"),
        (2, np.asarray([15], np.int32), "S32 scalar"),
        (2, np.asarray(15, np.float32), "S32 scalar"),
    ],
)
def test_invalid_operands_rejected_before_loading_cuda(monkeypatch, which, value, message):
    monkeypatch.setattr(cb, "_ensure_projector_capacity_ffi", lambda: pytest.fail("loaded CUDA before validation"))
    args = list(_inputs())
    args[which] = value
    with pytest.raises(ValueError, match=message):
        cb.project_relion_half_capacity(*args, image_shape=(32, 32))


@pytest.mark.parametrize("shape,pf", [((31, 31), 1), ((32, 30), 1), ((32, 32), 3)])
def test_invalid_static_geometry(monkeypatch, shape, pf):
    monkeypatch.setattr(cb, "_ensure_projector_capacity_ffi", lambda: pytest.fail("loaded CUDA before validation"))
    with pytest.raises(ValueError):
        cb.project_relion_half_capacity(*_inputs(), image_shape=shape, padding_factor=pf)


def test_runtime_radius_is_operand_not_attribute_and_reuses_trace(monkeypatch):
    records = []
    monkeypatch.setattr(cb, "_ensure_projector_capacity_ffi", lambda: None)

    def fake_ffi(target, output, **options):
        def call(half, rotations, radius, **attrs):
            records.append((target, half.shape, rotations.shape, radius.aval, attrs))
            return jnp.full(output.shape, radius, dtype=output.dtype)

        return call

    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi)
    cb.project_relion_half_capacity.clear_cache()
    half, rotations, _ = _inputs()
    for radius in (15, 16, 0):
        result = cb.project_relion_half_capacity(half, rotations, jnp.asarray(radius, jnp.int32), image_shape=(32, 32))
        np.testing.assert_array_equal(result, np.full(result.shape, radius, np.complex64))
    assert len(records) == 1
    target, _, rotshape, radius_aval, attrs = records[0]
    assert target == cb._TARGET_PROJECT_RELION_HALF_RUNTIME
    assert rotshape == (1, 6)
    assert radius_aval.shape == () and radius_aval.dtype == np.dtype(np.int32)
    assert set(attrs) == {"image_h", "image_w", "padding_factor"}
    assert cb.project_relion_half_capacity._cache_size() == 1
    cb.project_relion_half_capacity.clear_cache()


def test_qualified_old_library_keeps_old_paths_and_capacity_fails_closed(monkeypatch):
    old_symbols = {symbol: object() for _, symbol in cb._FFI_REGISTRATIONS}
    assert "ProjectRelionHalfRuntime" not in old_symbols
    library = SimpleNamespace(**old_symbols)
    registrations = []
    monkeypatch.setattr(cb, "_ffi_registered", False)
    monkeypatch.setattr(cb, "_projector_capacity_ffi_registered", False)
    monkeypatch.setattr(cb, "_loaded_lib_path", None)
    monkeypatch.setattr(cb, "_get_lib", lambda: library)
    monkeypatch.setattr(jax.ffi, "pycapsule", lambda symbol: symbol)
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda target, *args, **kwargs: registrations.append(target))
    cb._ensure_ffi()
    assert registrations == [target for target, _ in cb._FFI_REGISTRATIONS]
    with pytest.raises(RuntimeError, match="lacks ProjectRelionHalfRuntime"):
        cb._ensure_projector_capacity_ffi()
    assert not cb._projector_capacity_ffi_registered
    assert cb._ffi_registered
    library.ProjectRelionHalfRuntime = object()
    cb._ensure_projector_capacity_ffi()
    cb._ensure_projector_capacity_ffi()
    assert registrations.count(cb._TARGET_PROJECT_RELION_HALF_RUNTIME) == 1


def _pad_with_poison(logical, q, pf):
    size = pf * q + 3
    physical = np.full((size, size, size // 2 + 1), complex(1e9, -1e9), np.complex64)
    offset = (size - logical.shape[0]) // 2
    physical[offset : offset + logical.shape[0], offset : offset + logical.shape[1], : logical.shape[2]] = logical
    return physical


def _rotations():
    random = Rotation.random(12, random_state=29).as_matrix().astype(np.float32)
    # Tiny and interpolation-fraction-scale perturbations about axis rotations.
    angles = [0, np.nextafter(np.float32(0), np.float32(1)), 1 / 256, -1 / 256, np.pi / 2, np.pi]
    boundary = Rotation.from_euler("z", angles).as_matrix().astype(np.float32)
    return jnp.asarray(np.concatenate((np.eye(3, dtype=np.float32)[None], random, boundary)))


@pytest.mark.gpu
@pytest.mark.parametrize("pf", [1, 2])
@pytest.mark.parametrize("q", [32, 64, 96])
def test_gpu_old_texture_bitwise_all_logical_radii_and_one_executable(q, pf):
    assert jax.default_backend() == "gpu"
    cb._ensure_ffi()
    rotations = _rotations()
    rng = np.random.default_rng(29)
    cb.project_relion_half_capacity.clear_cache()
    for radius in range(15, min(42, q // 2) + 1):
        size = 2 * pf * radius + 3
        # Nonzero random ghost planes and poisoned capacity catch support errors.
        logical = (
            rng.standard_normal((size, size, size // 2 + 1)) + 1j * rng.standard_normal((size, size, size // 2 + 1))
        ).astype(np.complex64)
        full = relion_projector_half_to_texture_full(jnp.asarray(logical))
        reference = cb.project(
            full.reshape(-1),
            rotations,
            image_shape=(q, q),
            volume_shape=full.shape,
            half_image=True,
            max_r=float(radius),
            relion_texture_interp=True,
        )
        result = cb.project_relion_half_capacity(
            jnp.asarray(_pad_with_poison(logical, q, pf)),
            rotations,
            jnp.asarray(radius, jnp.int32),
            image_shape=(q, q),
            padding_factor=pf,
        )
        np.testing.assert_array_equal(np.asarray(result).view(np.uint32), np.asarray(reference).view(np.uint32), err_msg=f"q={q}, pf={pf}, r={radius}")
    assert cb.project_relion_half_capacity._cache_size() == 1


@pytest.mark.gpu
@pytest.mark.parametrize("radius", [-1, 17, np.iinfo(np.int32).max])
def test_gpu_invalid_runtime_radius_is_nonfinite_without_oob(radius):
    half, rotations, _ = _inputs()
    result = cb.project_relion_half_capacity(half, rotations, jnp.asarray(radius, jnp.int32), image_shape=(32, 32))
    assert np.isnan(np.asarray(result)).all()
