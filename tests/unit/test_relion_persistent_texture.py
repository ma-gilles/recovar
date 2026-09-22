"""Persistent texture ownership contracts retained from Q."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
pytestmark = pytest.mark.unit

def _projector(*, r_max=1, padding_factor=1, seed=260830):
    padded = int(r_max) * int(padding_factor)
    shape = (2 * padded + 3, 2 * padded + 3, padded + 2)
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64))


def _rotations():
    base = []
    for angle_x, angle_y, angle_z in (
        (0.37, -0.52, 0.19),
        (-0.91, 0.43, 1.17),
        (1.20, -0.73, -0.44),
        (-0.28, -1.01, 0.66),
    ):
        cx, sx = np.cos(angle_x), np.sin(angle_x)
        cy, sy = np.cos(angle_y), np.sin(angle_y)
        cz, sz = np.cos(angle_z), np.sin(angle_z)
        rotation_x = np.asarray(
            [[1, 0, 0], [0, cx, -sx], [0, sx, cx]],
            dtype=np.float64,
        )
        rotation_y = np.asarray(
            [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]],
            dtype=np.float64,
        )
        rotation_z = np.asarray(
            [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        base.append((rotation_z @ rotation_y @ rotation_x).astype(np.float32))
    return np.asarray(base)[np.asarray([3, 1, 3, 0, 2, 1, 0])]


def test_persistent_texture_rejects_nonexact_host_inputs():
    import recovar.cuda_backproject as cuda_backproject

    projector = _projector()
    constructor = cuda_backproject.RelionPersistentHalfTextureF32
    with pytest.raises(TypeError, match="NumPy host array"):
        constructor(
            jnp.asarray(projector),
            padding_factor=1,
            projector_max_r=1,
        )
    with pytest.raises(TypeError, match="complex64"):
        constructor(
            projector.astype(np.complex128),
            padding_factor=1,
            projector_max_r=1,
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        constructor(
            projector[:, ::-1, :],
            padding_factor=1,
            projector_max_r=1,
        )
    with pytest.raises(ValueError, match="projector_scale=1.0"):
        constructor(
            projector,
            padding_factor=1,
            projector_max_r=1,
            projector_scale=2.0,
        )
    with pytest.raises(ValueError, match="geometry mismatch"):
        constructor(
            projector[:-1],
            padding_factor=1,
            projector_max_r=1,
        )


def _install_fake_texture_runtime(monkeypatch, *, fail_device_put=False):
    import recovar.cuda_backproject as cuda_backproject

    events = []

    class FakeDevice:
        platform = "gpu"
        id = 0
        local_hardware_id = 0

    device = FakeDevice()

    def create(*args):
        events.append(("create", args[0]))
        args[-1]._obj.value = 73
        return 0

    def destroy(handle):
        events.append(("destroy", int(handle.value)))
        return 0

    def device_put(value, selected_device):
        assert selected_device is device
        events.append(("device_put", int(np.asarray(value))))
        if fail_device_put:
            raise RuntimeError("synthetic handle transfer failure")
        return jnp.asarray(value, dtype=jnp.uint64)

    def block_until_ready(value):
        events.append(("ready", value))
        return value

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        cuda_backproject.jax,
        "devices",
        lambda backend=None: [device],
    )
    monkeypatch.setattr(cuda_backproject, "custom_cuda_requested", lambda: True)
    monkeypatch.setattr(cuda_backproject, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(
        cuda_backproject,
        "_persistent_relion_half_texture_c_api",
        lambda: (create, destroy),
    )
    monkeypatch.setattr(cuda_backproject.jax, "device_put", device_put)
    monkeypatch.setattr(cuda_backproject.jax, "block_until_ready", block_until_ready)
    return cuda_backproject, device, events


def test_persistent_texture_reuses_one_upload_and_closes_after_readiness(monkeypatch):
    cuda_backproject, device, events = _install_fake_texture_runtime(monkeypatch)
    calls = []

    def fake_project(owner_handle, rotations, **kwargs):
        calls.append((owner_handle, tuple(rotations.shape), kwargs))
        return np.full((rotations.shape[0], 6), 2 + 3j, dtype=np.complex64)

    monkeypatch.setattr(
        cuda_backproject,
        "_relion_projector_persistent_half_texture_f32",
        fake_project,
    )
    texture = cuda_backproject.RelionPersistentHalfTextureF32(
        _projector(),
        padding_factor=1,
        projector_max_r=1,
        device=device,
    )
    rotations = jnp.eye(3, dtype=jnp.float32)[None]
    first = cuda_backproject.relion_projector_persistent_half_texture_f32(
        texture,
        rotations,
        current_size=2,
        padding_factor=1,
        projector_max_r=1,
    )
    second = cuda_backproject.relion_projector_persistent_half_texture_f32(
        texture,
        rotations,
        current_size=2,
        padding_factor=1,
        projector_max_r=1,
    )
    assert len(calls) == 2
    assert sum(event[0] == "create" for event in events) == 1
    np.testing.assert_array_equal(first, second)

    pending = object()
    texture._last_output = pending
    texture.close()
    ready_index = next(index for index, event in enumerate(events) if event[0] == "ready" and event[1] is pending)
    destroy_index = next(index for index, event in enumerate(events) if event[0] == "destroy")
    assert ready_index < destroy_index
    texture.close()
    assert sum(event[0] == "destroy" for event in events) == 1

    with pytest.raises(RuntimeError, match="closed"):
        cuda_backproject.relion_projector_persistent_half_texture_f32(
            texture,
            rotations,
            current_size=2,
            padding_factor=1,
            projector_max_r=1,
        )


def test_persistent_texture_geometry_device_and_constructor_cleanup(monkeypatch):
    cuda_backproject, device, events = _install_fake_texture_runtime(monkeypatch)
    texture = cuda_backproject.RelionPersistentHalfTextureF32(
        _projector(),
        padding_factor=1,
        projector_max_r=1,
        device=device,
    )
    with pytest.raises(ValueError, match="geometry"):
        texture._require_live_geometry(
            padding_factor=2,
            projector_max_r=1,
        )
    texture.close()

    with pytest.raises(ValueError, match="not a local JAX GPU"):
        cuda_backproject.RelionPersistentHalfTextureF32(
            _projector(),
            padding_factor=1,
            projector_max_r=1,
            device=object(),
        )

    cuda_backproject, device, failed_events = _install_fake_texture_runtime(
        monkeypatch,
        fail_device_put=True,
    )
    with pytest.raises(RuntimeError, match="handle transfer failure"):
        cuda_backproject.RelionPersistentHalfTextureF32(
            _projector(),
            padding_factor=1,
            projector_max_r=1,
            device=device,
        )
    assert [event[0] for event in failed_events].count("create") == 1
    assert [event[0] for event in failed_events].count("destroy") == 1
    assert events



@pytest.mark.gpu
def test_persistent_host_texture_matches_transient_and_rejects_stale_token(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.helpers.projection import (
        compute_relion_projector_projections_block,
    )

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    projector = _projector(r_max=7, padding_factor=2)
    rotations = _rotations()

    with jax.default_device(gpu_device):
        rotations_jax = jnp.asarray(rotations)
        transient = cuda_backproject.relion_projector_half_texture_f32(
            jnp.asarray(projector),
            rotations_jax,
            current_size=16,
            padding_factor=2,
            projector_max_r=7,
        )
        texture = cuda_backproject.RelionPersistentHalfTextureF32(
            projector,
            padding_factor=2,
            projector_max_r=7,
            device=gpu_device,
        )
        handle = texture.owner_handle
        first = cuda_backproject.relion_projector_persistent_half_texture_f32(
            texture,
            rotations_jax[:4],
            current_size=16,
            padding_factor=2,
            projector_max_r=7,
        )
        second = cuda_backproject.relion_projector_persistent_half_texture_f32(
            texture,
            rotations_jax[4:],
            current_size=16,
            padding_factor=2,
            projector_max_r=7,
        )
        persistent = jnp.concatenate((first, second), axis=0)
        assert texture.owner_handle == handle
        np.testing.assert_array_equal(
            np.asarray(persistent).view(np.uint32),
            np.asarray(transient).view(np.uint32),
        )
        transient_production, transient_abs2 = (
            compute_relion_projector_projections_block(
                jnp.asarray(projector),
                rotations_jax,
                (16, 16),
                r_max=7,
                padding_factor=2,
                centered_rows=True,
                dense_scale=True,
                projector_output_size=16,
                relion_texture_interp=True,
            )
        )
        np.testing.assert_array_equal(
            np.asarray(persistent * np.float32(-256)).view(np.uint32),
            np.asarray(transient_production).view(np.uint32),
        )

        persistent_production, persistent_abs2 = compute_relion_projector_projections_block(
            None, rotations_jax, (16, 16), r_max=7, padding_factor=2,
            centered_rows=True, dense_scale=True, projector_output_size=16,
            persistent_texture=texture,
        )
        np.testing.assert_array_equal(np.asarray(persistent_production).view(np.uint32), np.asarray(transient_production).view(np.uint32))
        np.testing.assert_array_equal(np.asarray(persistent_abs2).view(np.uint32), np.asarray(transient_abs2).view(np.uint32))

        same_shape = rotations_jax[:4]
        cache_size_after_first_owner = cuda_backproject._relion_projector_persistent_half_texture_f32._cache_size()
        stale_token = texture._handle_array
        texture.close()
        texture.close()

        second_texture = cuda_backproject.RelionPersistentHalfTextureF32(
            projector,
            padding_factor=2,
            projector_max_r=7,
            device=gpu_device,
        )
        assert second_texture.owner_handle != handle
        cuda_backproject.relion_projector_persistent_half_texture_f32(
            second_texture,
            same_shape,
            current_size=16,
            padding_factor=2,
            projector_max_r=7,
        )
        assert (
            cuda_backproject._relion_projector_persistent_half_texture_f32._cache_size() == cache_size_after_first_owner
        )

        # A cached executable carrying the old dynamic token must fail while
        # a same-geometry replacement remains live; monotonic handles prevent
        # the stale token from aliasing that new texture.
        with pytest.raises(Exception, match="owner handle is not live"):
            jax.block_until_ready(
                cuda_backproject._relion_projector_persistent_half_texture_f32(
                    stale_token,
                    same_shape,
                    current_size=16,
                    padding_factor=2,
                    projector_max_r=7,
                )
            )
        second_texture.close()


@pytest.mark.gpu
@pytest.mark.parametrize('radius,current_size', [(8, 16), (7, 8)])
def test_persistent_texture_preserves_nyquist_and_current_radius(
    monkeypatch, custom_cuda_lib, gpu_device, radius, current_size
):
    import recovar.cuda_backproject as cb
    from recovar.em.helpers.projection import compute_relion_projector_projections_block
    monkeypatch.setenv('RECOVAR_CUDA_LIB', str(custom_cuda_lib))
    monkeypatch.delenv('RECOVAR_DISABLE_CUDA', raising=False)
    projector = _projector(r_max=radius, padding_factor=2)
    with jax.default_device(gpu_device):
        rotations = jnp.concatenate((jnp.eye(3, dtype=jnp.float32)[None], jnp.asarray(_rotations())))
        expected, _ = compute_relion_projector_projections_block(
            jnp.asarray(projector), rotations, (current_size, current_size),
            r_max=radius, padding_factor=2, centered_rows=True, dense_scale=False,
            projector_output_size=current_size, relion_texture_interp=True,
        )
        with cb.RelionPersistentHalfTextureF32(
            projector, padding_factor=2, projector_max_r=radius, device=gpu_device
        ) as texture:
            actual = cb.relion_projector_persistent_half_texture_f32(
                texture, rotations, current_size=current_size, padding_factor=2,
                projector_max_r=radius,
            )
            np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), np.asarray(expected).view(np.uint32))


def test_sparse_pass2_opens_eligible_texture_from_original_host_slab(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.sparse_pass2 import dispatch as oversampling
    from recovar.em.helpers import projection

    projector = _projector()
    created = []
    sentinel = object()

    def fake_constructor(value, **kwargs):
        created.append((value, kwargs))
        return sentinel

    monkeypatch.setattr(
        projection,
        "_relion_projector_texture_enabled",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        cuda_backproject,
        "RelionPersistentHalfTextureF32",
        fake_constructor,
    )
    actual = oversampling._open_persistent_relion_projector_texture(
        projector,
        relion_projector_r_max=1,
        projection_padding_factor=1,
    )
    assert actual is sentinel
    assert created == [
        (
            projector,
            {
                "padding_factor": 1,
                "projector_max_r": 1,
                "projector_scale": 1.0,
            },
        )
    ]


@pytest.mark.parametrize("raise_from_bucket", [False, True])
def test_sparse_pass2_routes_host_projector_and_always_closes(
    monkeypatch,
    raise_from_bucket,
):
    from recovar.em.sparse_pass2 import dispatch as oversampling
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed

    projector = _projector()

    class FakeTexture:
        close_count = 0

        def close(self):
            self.close_count += 1

    texture = FakeTexture()
    opened = []
    captured = []

    def fake_open(value, **kwargs):
        opened.append((value, kwargs))
        return texture

    def fake_bucket(*args, **kwargs):
        captured.append((args, kwargs))
        if raise_from_bucket:
            raise RuntimeError("synthetic bucket failure")
        return "bucket-result"

    monkeypatch.setattr(
        oversampling,
        "_open_persistent_relion_projector_texture",
        fake_open,
    )
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "compute_pass2_stats_sparse_bucketed",
        fake_bucket,
    )

    def call():
        return oversampling.compute_pass2_stats_sparse(
            object(),
            object(),
            None,
            None,
            np.zeros((1, 2), dtype=np.float32),
            [np.asarray([0], dtype=np.int64)],
            0,
            "linear_interp",
            relion_projector_half=projector,
            relion_projector_r_max=1,
        )

    if raise_from_bucket:
        with pytest.raises(RuntimeError, match="synthetic bucket failure"):
            call()
    else:
        assert call() == "bucket-result"
    assert opened[0][0] is projector
    assert captured[0][1]["relion_projector_half"] is None
    assert captured[0][1]["relion_projector_texture"] is texture
    assert texture.close_count == 1


@pytest.mark.parametrize("raise_from_finalize", [False, True])
def test_sparse_pass2_early_texture_release_precedes_finalize_and_outer_cleanup(
    caplog,
    raise_from_finalize,
):
    from recovar.em.sparse_pass2 import dispatch as oversampling
    from recovar.em.sparse_pass2 import sparse_pass2_projection_blocks as sparse_pass2_bucketed

    events = []

    class FakeTexture:
        def __init__(self):
            self.closed = False
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.closed:
                return
            self.closed = True
            events.append("destroy")

    texture = FakeTexture()

    def finalize_after_early_release():
        released = (
            sparse_pass2_bucketed._close_relion_projector_texture_after_sparse_scoring(
                texture,
            )
        )
        assert released is None
        events.append("finalize")
        if raise_from_finalize:
            raise RuntimeError("synthetic finalization failure")
        return "result"

    with caplog.at_level("INFO"):
        if raise_from_finalize:
            with pytest.raises(RuntimeError, match="synthetic finalization failure"):
                oversampling._call_with_persistent_texture_cleanup(
                    texture,
                    finalize_after_early_release,
                )
        else:
            result = oversampling._call_with_persistent_texture_cleanup(
                texture,
                finalize_after_early_release,
            )
            assert result == "result"

    assert events == ["destroy", "finalize"]
    assert texture.close_calls == 2
    assert "releasing persistent RELION projector texture before output finalization" in caplog.text



def test_large_static_projector_uses_half_storage_without_full_cube(monkeypatch):
    import recovar.cuda_backproject as cb
    from recovar.em.helpers import projection
    class HostShape:
        dtype = jnp.dtype(jnp.complex64)
        shape = (1603, 1603, 802)
    slab = HostShape()
    observed = []
    def project(value, rotations, **kwargs):
        assert value is slab
        observed.append(kwargs)
        return jnp.zeros((rotations.shape[0], 40), dtype=jnp.complex64)
    def forbid_full(*args, **kwargs):
        raise AssertionError('large static projector expanded to a full cube')
    monkeypatch.setattr(cb, 'relion_projector_half_texture_f32', project)
    monkeypatch.setattr(projection, 'relion_projector_half_to_texture_full', forbid_full)
    result = projection._project_relion_projector_texture(
        slab, jnp.eye(3, dtype=jnp.float32)[None], (8, 8),
        r_max=400, padding_factor=2, projector_output_size=8,
    )
    assert result.shape == (1, 40)
    assert observed == [dict(current_size=8, padding_factor=2, projector_max_r=400)]


@pytest.mark.parametrize("layout", ["valid", "strided", "double", "class_axis", "missing_radius"])
def test_host_texture_planning_and_dispatch_share_eligibility(monkeypatch, layout):
    from recovar.em.helpers import projection
    calls = []
    monkeypatch.setattr(projection, "_relion_projector_texture_enabled", lambda *a, **kw: calls.append(kw) or True)
    slab = np.zeros((7, 7, 4), dtype=np.complex64)
    radius = 1
    if layout == "strided": slab = slab[:, :, ::2]
    if layout == "double": slab = slab.astype(np.complex128)
    if layout == "class_axis": slab = slab[None]
    if layout == "missing_radius": radius = None
    assert projection._host_relion_projector_texture_enabled(slab, r_max=radius, padding_factor=2) is (layout == "valid")
    assert len(calls) == (1 if layout == "valid" else 0)


@pytest.mark.parametrize("classes", [1, 4])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_projector_class_selection_preserves_host_view(monkeypatch, classes, dtype):
    from recovar.em.classification import k_class_inputs
    source = np.arange(classes * 7 * 7 * 4).reshape(classes, 7, 7, 4).astype(dtype)
    def forbidden_upload(*args, **kwargs):
        raise AssertionError("class selection uploaded the host projector")
    monkeypatch.setattr(k_class_inputs.jnp, "asarray", forbidden_upload)
    result = k_class_inputs._select_projector_half_for_class(source, classes - 1, classes)
    assert isinstance(result, np.ndarray) and result.dtype == dtype
    assert result.shape == (7, 7, 4) and result.flags.c_contiguous
    assert np.shares_memory(source, result)
    np.testing.assert_array_equal(result, source[classes - 1])


def test_host_float32_upload_cast_preserves_double_source(monkeypatch):
    from recovar.em.sparse_pass2 import dispatch
    from recovar.em.helpers import projection
    from recovar import cuda_backproject
    source = _projector().astype(np.complex128)
    source += np.float64(2**-27)
    original = source.copy()
    captured = []
    monkeypatch.setattr(projection, "_relion_projector_texture_enabled", lambda value, **kw: value.dtype == np.complex64)
    monkeypatch.setattr(cuda_backproject, "RelionPersistentHalfTextureF32", lambda value, **kw: captured.append(value) or object())
    assert projection._host_relion_projector_texture_enabled(source, r_max=1, padding_factor=1, allow_float32_cast=True)
    dispatch._open_persistent_relion_projector_texture(source, relion_projector_r_max=1, projection_padding_factor=1)
    assert captured[0].dtype == np.complex64
    np.testing.assert_array_equal(captured[0], source.astype(np.complex64))
    np.testing.assert_array_equal(source, original)
    assert source.dtype == np.complex128 and not np.shares_memory(source, captured[0])


@pytest.mark.gpu
def test_host_cast_matches_device_cast_and_texture_projection(custom_cuda_lib, gpu_device):
    from recovar.em.sparse_pass2 import dispatch
    from recovar.em.helpers.projection import compute_relion_projector_projections_block
    source = _projector(r_max=7, padding_factor=2).astype(np.complex128)
    source += np.float64(2**-26)
    source.reshape(-1)[:4] = [complex(-0.0, 0.0), complex(0.0, -0.0), 1 + 2**-24, 1 + 3 * 2**-24]
    with jax.default_device(gpu_device):
        old_double = jnp.asarray(source)
        assert old_double.dtype == jnp.complex128
        old_float = old_double.astype(jnp.complex64)
        np.testing.assert_array_equal(np.asarray(old_float).view(np.uint32), source.astype(np.complex64).view(np.uint32))
        rotations = jnp.asarray(_rotations())
        kwargs = dict(r_max=7, padding_factor=2, projector_output_size=16, centered_rows=True, dense_scale=True)
        expected = compute_relion_projector_projections_block(old_float, rotations, (16, 16), **kwargs)
        texture = dispatch._open_persistent_relion_projector_texture(source, relion_projector_r_max=7, projection_padding_factor=2)
        assert texture is not None
        try:
            actual = compute_relion_projector_projections_block(None, rotations, (16, 16), persistent_texture=texture, **kwargs)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(np.asarray(a).view(np.uint32), np.asarray(b).view(np.uint32))
        finally:
            texture.close()
