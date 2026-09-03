"""Focused ownership guards for persistent host-uploaded RELION textures."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
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


def test_sparse_pass2_opens_eligible_texture_from_original_host_slab(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers import oversampling, projection

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
    from recovar.em.dense_single_volume.helpers import (
        oversampling,
        sparse_pass2_bucketed,
    )

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


@pytest.mark.parametrize("raise_from_engine", [False, True])
def test_exact_local_routes_host_projector_and_always_closes(
    monkeypatch,
    raise_from_engine,
):
    from recovar.em.dense_single_volume import local_search_iteration

    projector = _projector()
    events = []

    class FakeTexture:
        def close(self):
            events.append("close")

    texture = FakeTexture()

    def fake_open(value, **kwargs):
        assert value is projector
        assert kwargs == {
            "relion_projector_r_max": 1,
            "projection_padding_factor": 1,
            "relion_texture_interp": None,
            "log_label": "Exact local",
        }
        events.append("open")
        return texture

    def fake_engine(*args, **kwargs):
        events.append("engine")
        assert args == ("dataset",)
        assert kwargs["relion_projector_half"] is None
        assert kwargs["relion_projector_texture"] is texture
        if raise_from_engine:
            raise RuntimeError("synthetic exact-local failure")
        return "exact-local-result"

    monkeypatch.setattr(
        local_search_iteration._oversampling,
        "_open_persistent_relion_projector_texture",
        fake_open,
    )

    def call():
        return local_search_iteration._call_exact_local_engine_with_projector_texture(
            fake_engine,
            "dataset",
            relion_projector_half=projector,
            relion_projector_r_max=1,
            projection_padding_factor=1,
            projection_relion_texture_interp=None,
        )

    if raise_from_engine:
        with pytest.raises(RuntimeError, match="synthetic exact-local failure"):
            call()
    else:
        assert call() == "exact-local-result"
    assert events == ["open", "engine", "close"]


def test_exact_local_manual_projector_does_not_open_texture(monkeypatch):
    from recovar.em.dense_single_volume import local_search_iteration

    projector = _projector()

    def fail_open(*args, **kwargs):
        raise AssertionError("manual projection must not open a CUDA texture")

    def fake_engine(**kwargs):
        assert kwargs["relion_projector_half"] is projector
        assert "relion_projector_texture" not in kwargs
        return "manual-result"

    monkeypatch.setattr(
        local_search_iteration._oversampling,
        "_open_persistent_relion_projector_texture",
        fail_open,
    )
    assert (
        local_search_iteration._call_exact_local_engine_with_projector_texture(
            fake_engine,
            relion_projector_half=projector,
            relion_projector_r_max=1,
            projection_padding_factor=1,
            projection_relion_texture_interp=False,
        )
        == "manual-result"
    )


def test_compiled_local_projection_routes_dynamic_persistent_handle(monkeypatch):
    from recovar.em.dense_single_volume import local_big_jit

    handle = jnp.asarray(73, dtype=jnp.uint64)
    rotations = jnp.eye(3, dtype=jnp.float32)[None]
    calls = []

    def fake_projector(projector_half, rotations_arg, image_shape, **kwargs):
        calls.append((projector_half, rotations_arg, image_shape, kwargs))
        return jnp.ones((1, 6), dtype=jnp.complex64), None

    monkeypatch.setattr(
        local_big_jit,
        "compute_relion_projector_projections_block",
        fake_projector,
    )
    result = local_big_jit._project_local_half_spectrum(
        jnp.zeros(1, dtype=jnp.complex64),
        jnp.zeros((1, 1, 1), dtype=jnp.complex64),
        rotations,
        None,
        (2, 2),
        (1, 1, 1),
        "linear_interp",
        relion_projector_texture_handle=handle,
        projection_half_volume=False,
        projection_max_r=1,
        relion_projector_output_size=2,
        projection_relion_texture_interp=True,
        projection_force_jax=False,
        use_relion_projector=True,
        use_persistent_relion_texture=True,
        relion_projector_r_max=1,
        projection_padding_factor=1,
    )
    assert result.shape == (1, 6)
    assert calls[0][0].shape == (1, 1, 1)
    assert calls[0][3]["persistent_texture_handle"] is handle


def test_persistent_texture_source_pins_dynamic_token_and_scoring_lifetime():
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed

    python_source = Path(cuda_backproject.__file__).read_text()
    jit_start = python_source.index("def _relion_projector_persistent_half_texture_f32(")
    jit_block = python_source[
        jit_start : python_source.index(
            "def relion_projector_persistent_half_texture_f32(",
            jit_start,
        )
    ]
    assert "owner_handle: jax.Array" in jit_block
    assert ")(" in jit_block
    assert "owner_handle," in jit_block
    assert "owner_handle=np." not in jit_block

    cuda_source = (Path(cuda_backproject.__file__).resolve().parent / "cuda" / "cuda_backproject.cu").read_text()
    ffi_start = cuda_source.index("RelionProjectorPersistentHalfTextureF32Impl(")
    ffi_block = cuda_source[
        ffi_start : cuda_source.index(
            "RelionCoarseDiff2NativeTextureRectangularF32Impl(",
            ffi_start,
        )
    ]
    assert ".Arg<ffi::AnyBuffer>()" in ffi_block
    assert '.Attr<uint64_t>("owner_handle")' not in ffi_block
    assert "owner->device != current_device" in ffi_block
    assert "owner handle is not live" in ffi_block

    sparse_source = Path(sparse_pass2_bucketed.__file__).read_text()
    function_start = sparse_source.index("def compute_pass2_stats_sparse_bucketed(")
    function_end = sparse_source.index(
        "def compute_k_class_pass2_stats_sparse_fused(",
        function_start,
    )
    function_block = sparse_source[function_start:function_end]
    release_index = function_block.index(
        "_close_relion_projector_texture_after_sparse_scoring("
    )
    assert function_block.rindex(
        "relion_projector_texture=relion_projector_texture,",
        0,
        release_index,
    ) < release_index
    assert release_index < function_block.index(
        "_replay_deferred_firstiter_bpref_batches("
    )
    assert release_index < function_block.index("finalize_half_volume_bpref(")


@pytest.mark.parametrize("raise_from_finalize", [False, True])
def test_sparse_pass2_early_texture_release_precedes_finalize_and_outer_cleanup(
    caplog,
    raise_from_finalize,
):
    from recovar.em.dense_single_volume.helpers import (
        oversampling,
        sparse_pass2_bucketed,
    )

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


@pytest.mark.gpu
def test_persistent_host_texture_matches_transient_and_rejects_stale_token(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume.helpers.projection import (
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
        persistent_production, persistent_abs2 = (
            compute_relion_projector_projections_block(
                None,
                rotations_jax,
                (16, 16),
                r_max=7,
                padding_factor=2,
                centered_rows=True,
                dense_scale=True,
                projector_output_size=16,
                persistent_texture=texture,
            )
        )
        np.testing.assert_array_equal(
            np.asarray(persistent_production).view(np.uint32),
            np.asarray(transient_production).view(np.uint32),
        )
        np.testing.assert_array_equal(
            np.asarray(persistent_abs2).view(np.uint32),
            np.asarray(transient_abs2).view(np.uint32),
        )

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
        with pytest.raises(Exception, match="owner handle is not live"):
            jax.block_until_ready(
                cuda_backproject.relion_projector_persistent_half_texture_f32_from_handle(
                    jnp.asarray(0, dtype=jnp.uint64),
                    same_shape,
                    current_size=16,
                    padding_factor=2,
                    projector_max_r=7,
                )
            )
        second_texture.close()


@pytest.mark.gpu
def test_dynamic_texture_handle_compiles_through_local_big_jit_projection(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.dense_single_volume import local_big_jit

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    projector = _projector(r_max=7, padding_factor=2)
    rotations = _rotations()[:4]
    pixel_indices = jnp.asarray([0, 1, 9, 17, 26, 63], dtype=jnp.int32)

    def project(projector_half, rotations_arg, *, handle, persistent):
        return local_big_jit._project_local_half_spectrum(
            jnp.zeros(1, dtype=jnp.complex64),
            projector_half,
            rotations_arg,
            pixel_indices,
            (16, 16),
            (1, 1, 1),
            "linear_interp",
            relion_projector_texture_handle=handle,
            projection_half_volume=False,
            projection_max_r=7,
            relion_projector_output_size=16,
            projection_relion_texture_interp=True,
            projection_force_jax=False,
            use_relion_projector=True,
            use_persistent_relion_texture=persistent,
            relion_projector_r_max=7,
            projection_padding_factor=2,
        )

    @jax.jit
    def project_persistent(handle, rotations_arg):
        return project(
            jnp.zeros((1, 1, 1), dtype=jnp.complex64),
            rotations_arg,
            handle=handle,
            persistent=True,
        )

    with jax.default_device(gpu_device):
        rotations_jax = jnp.asarray(rotations)
        transient = project(
            jnp.asarray(projector),
            rotations_jax,
            handle=None,
            persistent=False,
        )
        texture = cuda_backproject.RelionPersistentHalfTextureF32(
            projector,
            padding_factor=2,
            projector_max_r=7,
            device=gpu_device,
        )
        first = project_persistent(texture.handle_array, rotations_jax)
        np.testing.assert_array_equal(
            np.asarray(first).view(np.uint32),
            np.asarray(transient).view(np.uint32),
        )
        assert project_persistent._cache_size() == 1
        texture.close()

        replacement = cuda_backproject.RelionPersistentHalfTextureF32(
            projector,
            padding_factor=2,
            projector_max_r=7,
            device=gpu_device,
        )
        second = project_persistent(replacement.handle_array, rotations_jax)
        np.testing.assert_array_equal(
            np.asarray(second).view(np.uint32),
            np.asarray(transient).view(np.uint32),
        )
        assert project_persistent._cache_size() == 1
        replacement.close()
