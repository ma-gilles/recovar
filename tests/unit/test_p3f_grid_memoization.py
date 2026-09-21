"""P3-F: the constant frequency / pixel-coordinate grid memoization.

Every check here is **bitwise**: the memoized helper must return the same bytes
as the unmemoized builder it wraps, for every shape, spacing, flag and dtype the
EM harness and the shared SPA / cryo-ET tests exercise.  Nothing in this file
uses a tolerance.

The other properties the memoization relies on are checked here too: identity on
a hit, a bounded cache, a fall-through for arguments that are not static
scalars, a fall-through while a JAX trace is active (so no compiled program's
jaxpr changes), and the absence of eager dispatches on a hit, which is the
saving the change exists for.
"""

import collections
import concurrent.futures
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.core import fourier_transform_utils as ftu

pytestmark = pytest.mark.unit


# Shapes the EM matched harness, the SPA/ET pipelines and the shared unit tests
# reach: the 10k/256 EM fixture and its coarse current sizes, the 128 and 64
# pipeline boxes, odd and non-square cases, and the small sizes the helper unit
# tests use.
_IMAGE_SIDES = (2, 3, 4, 5, 8, 16, 31, 32, 50, 52, 64, 92, 100, 128, 256)
_IMAGE_SHAPES_2D = (
    (2, 2),
    (4, 4),
    (5, 5),
    (8, 8),
    (16, 16),
    (31, 31),
    (32, 32),
    (50, 50),
    (52, 52),
    (64, 64),
    (92, 92),
    (128, 128),
    (256, 256),
    (4, 8),
    (8, 4),
    (5, 8),
    (8, 5),
    (31, 32),
    (32, 31),
)
_VOLUME_SHAPES_3D = (
    (2, 2, 2),
    (4, 4, 4),
    (5, 5, 5),
    (8, 8, 8),
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (4, 8, 16),
    (5, 4, 3),
)
_VOXEL_SIZES = (1, 1.0, 0.5, 1.5, 3.2)
_DTYPES = (jnp.float32, jnp.float64, np.float32, np.dtype("float32"), np.float64)


def _same_bytes(actual, expected):
    """Bitwise equality: dtype, shape and the raw buffer."""
    a = np.asarray(actual)
    b = np.asarray(expected)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.fixture(autouse=True)
def _clean_cache():
    ftu.clear_grid_cache()
    yield
    ftu.clear_grid_cache()


class _EagerDispatchCounter:
    """Count eager JAX primitive dispatches, the quantity P3-E's census ranks."""

    def __enter__(self):
        from jax._src import core

        self._core = core
        self._original = core.EvalTrace.process_primitive
        self.count = 0
        counter = self

        def process_primitive(trace_self, primitive, args, params):
            counter.count += 1
            return counter._original(trace_self, primitive, args, params)

        core.EvalTrace.process_primitive = process_primitive
        return self

    def __exit__(self, *exc):
        self._core.EvalTrace.process_primitive = self._original
        return False


# --------------------------------------------------------------- bitwise ---


@pytest.mark.parametrize("n", _IMAGE_SIDES)
@pytest.mark.parametrize("scaled", (False, True))
def test_1d_frequency_grid_is_bitwise_unchanged(n, scaled):
    for voxel_size in _VOXEL_SIZES:
        for dtype in _DTYPES:
            expected = ftu._build_1d_frequency_grid(n, voxel_size, scaled, dtype)
            first = ftu.get_1d_frequency_grid(n, voxel_size, scaled, dtype=dtype)
            second = ftu.get_1d_frequency_grid(n, voxel_size, scaled, dtype=dtype)
            assert _same_bytes(first, expected)
            assert _same_bytes(second, expected)
            assert first is second


@pytest.mark.parametrize("n", _IMAGE_SIDES)
@pytest.mark.parametrize("scaled", (False, True))
def test_1d_frequency_grid_rfft_is_bitwise_unchanged(n, scaled):
    for voxel_size in _VOXEL_SIZES:
        for dtype in _DTYPES:
            expected = ftu._build_1d_frequency_grid_rfft(n, voxel_size, scaled, dtype)
            got = ftu.get_1d_frequency_grid_rfft(n, voxel_size, scaled, dtype=dtype)
            assert _same_bytes(got, expected)
            assert got is ftu.get_1d_frequency_grid_rfft(n, voxel_size, scaled, dtype=dtype)


@pytest.mark.parametrize("image_shape", _IMAGE_SHAPES_2D)
@pytest.mark.parametrize("scaled", (False, True))
def test_k_coordinate_of_each_pixel_is_bitwise_unchanged(image_shape, scaled):
    for voxel_size in (1, 1.5):
        for dtype in (jnp.float32, jnp.float64):
            expected = ftu._build_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled, dtype)
            got = ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled, dtype=dtype)
            assert _same_bytes(got, expected)
            assert got is ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled, dtype=dtype)


@pytest.mark.parametrize("volume_shape", _VOLUME_SHAPES_3D)
@pytest.mark.parametrize("scaled", (False, True))
def test_k_coordinate_of_each_pixel_3d_is_bitwise_unchanged(volume_shape, scaled):
    for voxel_size in (1, 1.5):
        expected = ftu._build_k_coordinate_of_each_pixel_3d(volume_shape, voxel_size, scaled)
        got = ftu.get_k_coordinate_of_each_pixel_3d(volume_shape, voxel_size, scaled)
        assert _same_bytes(got, expected)
        assert got is ftu.get_k_coordinate_of_each_pixel_3d(volume_shape, voxel_size, scaled)


@pytest.mark.parametrize("image_shape", _IMAGE_SHAPES_2D)
@pytest.mark.parametrize("scaled", (False, True))
def test_k_coordinate_of_each_pixel_real_is_bitwise_unchanged(image_shape, scaled):
    for voxel_size in (1, 1.5):
        expected = ftu._build_k_coordinate_of_each_pixel_real(image_shape, voxel_size, scaled)
        got = ftu.get_k_coordinate_of_each_pixel_real(image_shape, voxel_size, scaled)
        assert _same_bytes(got, expected)
        assert got is ftu.get_k_coordinate_of_each_pixel_real(image_shape, voxel_size, scaled)


@pytest.mark.parametrize("volume_shape", _VOLUME_SHAPES_3D)
@pytest.mark.parametrize("scaled", (False, True))
def test_k_coordinate_of_each_pixel_3d_real_is_bitwise_unchanged(volume_shape, scaled):
    for voxel_size in (1, 1.5):
        expected = ftu._build_k_coordinate_of_each_pixel_3d_real(volume_shape, voxel_size, scaled)
        got = ftu.get_k_coordinate_of_each_pixel_3d_real(volume_shape, voxel_size, scaled)
        assert _same_bytes(got, expected)
        assert got is ftu.get_k_coordinate_of_each_pixel_3d_real(volume_shape, voxel_size, scaled)


@pytest.mark.parametrize("image_shape", _IMAGE_SHAPES_2D)
@pytest.mark.parametrize("scaled", (False, True))
def test_k_coordinate_of_each_pixel_half_is_bitwise_unchanged(image_shape, scaled):
    for voxel_size in (1, 1.5):
        for dtype in (jnp.float32, jnp.float64):
            expected = ftu._build_k_coordinate_of_each_pixel_half(image_shape, voxel_size, scaled, dtype)
            got = ftu.get_k_coordinate_of_each_pixel_half(image_shape, voxel_size, scaled, dtype=dtype)
            assert _same_bytes(got, expected)


@pytest.mark.parametrize("image_shape", _IMAGE_SHAPES_2D)
def test_half_image_pixel_indices_are_bitwise_unchanged(image_shape):
    expected = ftu._build_half_image_pixel_indices(image_shape)
    got = ftu._half_image_pixel_indices(image_shape)
    assert _same_bytes(got, expected)
    assert got is ftu._half_image_pixel_indices(image_shape)


@pytest.mark.parametrize("n", _IMAGE_SIDES)
def test_packed_last_axis_indices_are_bitwise_unchanged(n):
    expected = ftu._build_real_fft_packed_last_axis_indices(n)
    got = ftu.get_real_fft_packed_last_axis_indices(n)
    assert _same_bytes(got, expected)
    assert got is ftu.get_real_fft_packed_last_axis_indices(n)


@pytest.mark.parametrize("n", _IMAGE_SIDES)
def test_shifted_conjugate_partner_indices_are_bitwise_unchanged(n):
    expected = ftu._build_shifted_conjugate_partner_indices(n)
    got = ftu.get_shifted_conjugate_partner_indices(n)
    assert _same_bytes(got, expected)
    assert got is ftu.get_shifted_conjugate_partner_indices(n)


def test_half_coordinates_still_match_the_host_planned_numpy_variant():
    """The existing byte-equivalence contract with the host-planned helper holds."""
    for image_shape in _IMAGE_SHAPES_2D:
        device = ftu.get_k_coordinate_of_each_pixel_half(image_shape, 1, scaled=False)
        host = ftu.get_k_coordinate_of_each_pixel_half_np(image_shape, 1, scaled=False)
        assert _same_bytes(device, host)


# ------------------------------------------------------------ key fidelity ---


def test_distinct_static_arguments_do_not_collide():
    """Each argument of the tuple must select its own cache entry."""
    base = ftu.get_1d_frequency_grid(8, 1, False, dtype=jnp.float32)
    assert ftu.get_1d_frequency_grid(9, 1, False, dtype=jnp.float32) is not base
    assert ftu.get_1d_frequency_grid(8, 1, True, dtype=jnp.float32) is not base
    assert ftu.get_1d_frequency_grid(8, 2, True, dtype=jnp.float32) is not (
        ftu.get_1d_frequency_grid(8, 3, True, dtype=jnp.float32)
    )
    assert ftu.get_1d_frequency_grid(8, 1, False, dtype=jnp.float64) is not base
    shape_a = ftu.get_k_coordinate_of_each_pixel((4, 8), 1, scaled=False)
    shape_b = ftu.get_k_coordinate_of_each_pixel((8, 4), 1, scaled=False)
    assert shape_a is not shape_b
    assert not _same_bytes(shape_a, shape_b)


def test_equal_shape_spellings_share_one_entry_and_the_same_bytes():
    tuple_form = ftu.get_k_coordinate_of_each_pixel((16, 16), 1, scaled=False)
    list_form = ftu.get_k_coordinate_of_each_pixel([16, 16], 1, scaled=False)
    numpy_form = ftu.get_k_coordinate_of_each_pixel(np.array([16, 16]), 1, scaled=False)
    assert tuple_form is list_form  # the ints are the same Python ints
    assert _same_bytes(numpy_form, tuple_form)  # np.int64 entries key separately


def test_dtype_spellings_that_canonicalize_together_share_one_entry():
    a = ftu.get_1d_frequency_grid(16, 1, False, dtype=jnp.float32)
    b = ftu.get_1d_frequency_grid(16, 1, False, dtype=np.dtype("float32"))
    c = ftu.get_1d_frequency_grid(16, 1, False, dtype="float32")
    assert a is b is c


def test_cache_context_tracks_default_device_and_x64():
    context = ftu._grid_cache_context()
    assert context == (jax.config.jax_default_device, jax.config.jax_enable_x64)
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        assert ftu._grid_cache_context()[0] is not None
        assert ftu._grid_cache_context() != context or jax.config.jax_default_device == cpu


def test_a_different_default_device_does_not_reuse_another_devices_grid():
    cpu = jax.devices("cpu")[0]
    outside = ftu.get_1d_frequency_grid(16, 1, False, dtype=jnp.float32)
    with jax.default_device(cpu):
        inside = ftu.get_1d_frequency_grid(16, 1, False, dtype=jnp.float32)
    assert inside is not outside
    assert _same_bytes(inside, outside)


# -------------------------------------------------------------- fallbacks ---


def test_non_static_arguments_fall_through_without_caching():
    ftu.clear_grid_cache()
    traced_spacing = jnp.asarray(2.0, dtype=jnp.float32)
    got = ftu.get_1d_frequency_grid(8, traced_spacing, True, dtype=jnp.float32)
    expected = ftu._build_1d_frequency_grid(8, traced_spacing, True, jnp.float32)
    assert _same_bytes(got, expected)
    assert ftu.grid_cache_size() == 0


def test_a_numpy_array_shape_entry_is_still_keyed_by_value():
    ftu.clear_grid_cache()
    got = ftu.get_k_coordinate_of_each_pixel(np.array([8, 8]), 1, scaled=False)
    expected = ftu._build_k_coordinate_of_each_pixel(np.array([8, 8]), 1, False, jnp.float32)
    assert _same_bytes(got, expected)
    assert ftu.grid_cache_size() > 0


def test_uncacheable_token_is_detected_for_arrays_and_none():
    assert ftu._static_scalar_token(jnp.asarray(1.0)) is ftu._UNCACHEABLE
    assert ftu._static_scalar_token(np.asarray([1.0])) is ftu._UNCACHEABLE
    assert ftu._static_shape_token(jnp.asarray([4, 4])) is ftu._UNCACHEABLE
    assert ftu._static_dtype_token(None) == (type(None), None)
    assert ftu._static_scalar_token(np.float32(2.0)) == (np.float32, np.float32(2.0))


def test_the_cache_is_bypassed_while_a_trace_is_active():
    """A traced call must stage its own operations, not receive a constant."""
    ftu.clear_grid_cache()
    eager = ftu.get_k_coordinate_of_each_pixel((8, 8), 1, scaled=False)
    assert ftu.grid_cache_size() > 0
    seen = {}

    @jax.jit
    def traced(x):
        grid = ftu.get_k_coordinate_of_each_pixel((8, 8), 1, scaled=False)
        seen["is_tracer"] = isinstance(grid, jax.core.Tracer)
        seen["is_cached_object"] = grid is eager
        return x + grid.sum()

    traced(jnp.float32(0.0))
    assert seen["is_tracer"] is True
    assert seen["is_cached_object"] is False


def test_tracing_probe_agrees_with_the_trace_context():
    assert ftu._tracing() is False
    inside = {}

    @jax.jit
    def traced(x):
        inside["tracing"] = ftu._tracing()
        return x

    traced(jnp.float32(0.0))
    assert inside["tracing"] is True


def test_a_traced_grid_is_never_stored():
    ftu.clear_grid_cache()

    @jax.jit
    def traced(x):
        return x + ftu.get_1d_frequency_grid(7, 1, False, dtype=jnp.float32).sum()

    traced(jnp.float32(0.0))
    assert ftu.grid_cache_size() == 0


# ----------------------------------------------------------------- bounds ---


def test_the_cache_is_bounded_and_evicts_the_least_recently_used_entry():
    ftu.clear_grid_cache()
    sizes = range(2, 2 + ftu._GRID_CACHE_MAXSIZE + 20)
    for n in sizes:
        ftu.get_1d_frequency_grid(n, 1, False, dtype=jnp.float32)
    assert ftu.grid_cache_size() == ftu._GRID_CACHE_MAXSIZE
    assert isinstance(ftu._grid_cache, collections.OrderedDict)
    # The oldest key was evicted; the newest survived.
    newest = list(sizes)[-1]
    assert ftu.get_1d_frequency_grid(newest, 1, False, dtype=jnp.float32) is (
        ftu.get_1d_frequency_grid(newest, 1, False, dtype=jnp.float32)
    )


def test_clear_grid_cache_empties_both_caches():
    ftu.get_1d_frequency_grid(16, 1, False, dtype=jnp.float32)
    ftu.get_k_coordinate_of_each_pixel_half_np((16, 16))
    assert ftu.grid_cache_size() > 0
    ftu.clear_grid_cache()
    assert ftu.grid_cache_size() == 0
    assert ftu._get_k_coordinate_of_each_pixel_half_np_cached.cache_info().currsize == 0


# --------------------------------------------------- the saving being made ---


def test_a_cache_hit_dispatches_no_eager_primitives():
    """This is the whole point: the second call must not reach the device."""
    ftu.clear_grid_cache()
    with _EagerDispatchCounter() as cold:
        ftu.get_k_coordinate_of_each_pixel_half((64, 64), 1, scaled=False)
    assert cold.count > 0
    with _EagerDispatchCounter() as warm:
        for _ in range(5):
            ftu.get_k_coordinate_of_each_pixel_half((64, 64), 1, scaled=False)
    assert warm.count == 0


def test_the_helper_family_dispatches_nothing_once_warm():
    ftu.clear_grid_cache()
    shape = (92, 92)

    def call_all():
        ftu.get_1d_frequency_grid(shape[0], 1, False)
        ftu.get_1d_frequency_grid_rfft(shape[0], 1, False)
        ftu.get_k_coordinate_of_each_pixel(shape, 1, scaled=False)
        ftu.get_k_coordinate_of_each_pixel_real(shape, 1, scaled=False)
        ftu.get_k_coordinate_of_each_pixel_half(shape, 1, scaled=False)
        ftu.get_k_coordinate_of_each_pixel_3d((32, 32, 32), 1, scaled=False)
        ftu.get_k_coordinate_of_each_pixel_3d_real((32, 32, 32), 1, scaled=False)
        ftu._half_image_pixel_indices(shape)
        ftu.get_real_fft_packed_last_axis_indices(shape[1])
        ftu.get_shifted_conjugate_partner_indices(shape[1])

    with _EagerDispatchCounter() as cold:
        call_all()
    with _EagerDispatchCounter() as warm:
        call_all()
    assert cold.count >= 40  # the ten helpers cost 43 eager dispatches cold on CPU
    assert warm.count == 0


# -------------------------------------------------- callers through to core ---


def test_geometry_and_ctf_callers_are_bitwise_unchanged_on_a_hit():
    """Two shared non-EM callers, warm cache against a cleared cache."""
    from recovar.core import geometry

    ftu.clear_grid_cache()
    cold_plane = np.asarray(geometry.get_unrotated_plane_grid_points((64, 64)))
    cold_half = np.asarray(geometry.get_unrotated_half_plane_grid_points((64, 64)))
    warm_plane = np.asarray(geometry.get_unrotated_plane_grid_points((64, 64)))
    warm_half = np.asarray(geometry.get_unrotated_half_plane_grid_points((64, 64)))
    assert cold_plane.tobytes() == warm_plane.tobytes()
    assert cold_half.tobytes() == warm_half.tobytes()


def test_every_trace_kind_bypasses_the_cache():
    """jit, vmap, grad and jvp must all read as tracing, not only jit."""
    ftu.clear_grid_cache()
    seen = {}

    def probe(tag):
        seen[tag] = ftu._tracing()
        return 0.0

    jax.jit(lambda x: x + probe("jit"))(jnp.float32(1.0))
    jax.vmap(lambda x: x + probe("vmap"))(jnp.arange(3, dtype=jnp.float32))
    jax.grad(lambda x: x * (1.0 + probe("grad")))(jnp.float32(1.0))
    jax.jacfwd(lambda x: x * (1.0 + probe("jvp")))(jnp.float32(1.0))
    assert seen == {"jit": True, "vmap": True, "grad": True, "jvp": True}
    assert ftu._tracing() is False
    assert ftu.grid_cache_size() == 0


def test_the_default_device_key_is_thread_local():
    """`utils/multi_gpu.py` sets one `jax.default_device` per worker thread.

    The cache context must follow the thread, not the process, or the first
    thread's grid would be handed to every other thread; because these arrays
    are uncommitted, the downstream work would follow them to the wrong device.
    """
    ftu.clear_grid_cache()
    device = jax.devices()[0]
    barrier = threading.Barrier(2)

    def with_context():
        with jax.default_device(device):
            barrier.wait()
            return ftu.get_k_coordinate_of_each_pixel((16, 16), 1, scaled=False), ftu._grid_cache_context()[0]

    def without_context():
        barrier.wait()
        return ftu.get_k_coordinate_of_each_pixel((16, 16), 1, scaled=False), ftu._grid_cache_context()[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        inside = pool.submit(with_context)
        outside = pool.submit(without_context)
        grid_in, ctx_in = inside.result()
        grid_out, ctx_out = outside.result()
    assert ctx_in is not None and ctx_out is None
    assert grid_in is not grid_out
    assert _same_bytes(grid_in, grid_out)


def test_two_devices_do_not_share_one_cached_grid():
    """The same property with two real devices, when the host has two."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    ftu.clear_grid_cache()
    barrier = threading.Barrier(2)

    def work(index):
        with jax.default_device(devices[index]):
            barrier.wait()
            return ftu.get_k_coordinate_of_each_pixel((16, 16), 1, scaled=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        grid_a, grid_b = list(pool.map(work, (0, 1)))
    assert grid_a is not grid_b
    assert grid_a.devices() != grid_b.devices()
    assert _same_bytes(grid_a, grid_b)
