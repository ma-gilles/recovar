"""Exactness and memory guards for box-scale RELION solvent masks."""

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
mask = pytest.importorskip("recovar.core.mask")
mean_helpers = pytest.importorskip("recovar.em.dense_single_volume.mean_helpers")


@pytest.mark.parametrize(
    ("volume_shape", "offset"),
    [
        ((8, 8, 8), (0.0, 0.0, 0.0)),
        ((9, 9, 9), (0.125, -0.25, 0.375)),
        ((8, 9, 10), (-0.5, 0.25, 0.125)),
    ],
)
def test_compiled_relion_solvent_mask_is_bitwise_exact(volume_shape, offset):
    radius = np.float64(2.375)
    radius_p = np.float64(4.125)
    offset = jnp.asarray(offset, dtype=jnp.float64)

    expected = mask.raised_cosine_mask(
        volume_shape,
        radius=radius,
        radius_p=radius_p,
        offset=offset,
    )
    actual = mean_helpers._compiled_relion_solvent_mask(volume_shape)(
        radius,
        radius_p,
        offset,
    )

    assert actual.dtype == expected.dtype == jnp.float64
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    mean_helpers._compiled_relion_solvent_mask.cache_clear()


def test_compiled_relion_solvent_mask_has_no_coordinate_stack_temporary():
    volume_shape = (80, 80, 80)
    compiled = mean_helpers._compiled_relion_solvent_mask(volume_shape).lower(
        np.float64(20.0),
        np.float64(24.0),
        jnp.zeros(3, dtype=jnp.float64),
    ).compile()
    memory = compiled.memory_analysis()

    output_bytes = int(np.prod(volume_shape, dtype=np.int64)) * np.dtype(np.float64).itemsize
    assert memory.output_size_in_bytes == output_bytes
    assert memory.temp_size_in_bytes == 0
    mean_helpers._compiled_relion_solvent_mask.cache_clear()


def test_box800_relion_solvent_mask_routes_to_compiled_builder(monkeypatch, caplog):
    volume_shape = (800, 800, 800)
    expected_bytes = 12_288_000_000
    calls = []

    def fake_factory(static_shape):
        calls.append(("factory", static_shape))

        def fake_build(radius, radius_p, offset):
            calls.append(("build", radius, radius_p, tuple(np.asarray(offset))))
            return jnp.asarray([7.0], dtype=jnp.float64)

        return fake_build

    def reject_uncompiled_mask(*_args, **_kwargs):
        raise AssertionError("box-800 mask must not materialize the unfused coordinate stack")

    monkeypatch.setattr(mean_helpers, "_compiled_relion_solvent_mask", fake_factory)
    monkeypatch.setattr(mean_helpers.mask, "raised_cosine_mask", reject_uncompiled_mask)
    caplog.set_level("INFO", logger=mean_helpers.__name__)

    result = mean_helpers._make_relion_solvent_mask(
        volume_shape,
        radius=np.float64(100.0),
        radius_p=np.float64(105.0),
        offset=jnp.zeros(3, dtype=jnp.float64),
    )

    assert mean_helpers._relion_solvent_mask_unfused_coordinate_bytes(volume_shape) == expected_bytes
    assert mean_helpers._large_relion_solvent_mask_uses_compiled_builder(volume_shape)
    assert not mean_helpers._large_relion_solvent_mask_uses_compiled_builder((256, 256, 256))
    assert calls == [
        ("factory", volume_shape),
        ("build", np.float64(100.0), np.float64(105.0), (0.0, 0.0, 0.0)),
    ]
    np.testing.assert_array_equal(np.asarray(result), np.asarray([7.0]))
    assert (
        "RELION box-scale solvent mask fused construction: shape=(800, 800, 800) "
        f"estimated_unfused_coordinate_bytes={expected_bytes}"
    ) in caplog.text
    assert "RELION box-scale solvent mask ready: shape=(800, 800, 800) dtype=float64" in caplog.text


@pytest.mark.parametrize("volume_shape", [(8, 8, 8), (7, 8, 9)])
def test_box_scale_solvent_flatten_lifecycle_is_bitwise_exact(monkeypatch, volume_shape):
    rng = np.random.default_rng(20260901 + sum(volume_shape))
    volume_ft = (
        rng.standard_normal(volume_shape) + 1j * rng.standard_normal(volume_shape)
    ).astype(np.complex64)
    solvent_mask = rng.uniform(0.0, 1.0, volume_shape).astype(np.float64)
    expected = mean_helpers.fourier_transform_utils.get_dft3(
        mean_helpers.fourier_transform_utils.get_idft3(jnp.asarray(volume_ft))
        * jnp.asarray(solvent_mask),
    ).reshape(-1)
    expected.block_until_ready()
    monkeypatch.setattr(
        mean_helpers,
        "_large_relion_solvent_mask_uses_compiled_builder",
        lambda _shape: True,
    )

    volume_actual = jnp.asarray(volume_ft)
    mask_actual = jnp.asarray(solvent_mask)
    actual = mean_helpers._apply_relion_solvent_flatten_k1(
        volume_actual,
        mask_actual,
        volume_shape,
        half_index=0,
    )

    assert actual.shape == (int(np.prod(volume_shape)),)
    assert actual.dtype == expected.dtype == jnp.complex128
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert not volume_actual.is_deleted()
    assert mask_actual.is_deleted()


def test_box_scale_solvent_flatten_releases_dead_inputs_in_order(monkeypatch, caplog):
    events = []

    class FakeBuffer:
        def __init__(self, name, dtype):
            self.name = name
            self.dtype = np.dtype(dtype)
            self.deleted = False

        def reshape(self, shape):
            events.append(("reshape", self.name, shape))
            return self

        def __mul__(self, other):
            events.append(("multiply", self.name, other.name))
            return (self, other)

        def block_until_ready(self):
            events.append(("block", self.name))
            return self

        def delete(self):
            events.append(("delete", self.name))
            self.deleted = True

        def is_deleted(self):
            return self.deleted

    volume_ft = FakeBuffer("volume_ft", np.complex64)
    vol_real = FakeBuffer("vol_real", np.complex64)
    solvent_mask = FakeBuffer("solvent_mask", np.float64)
    flattened = FakeBuffer("flattened", np.complex128)

    def fake_idft(value):
        events.append(("idft", value.name))
        return vol_real

    def fake_dft(value):
        assert value == (vol_real, solvent_mask)
        events.append(("dft", "masked_real"))
        return flattened

    monkeypatch.setattr(mean_helpers.fourier_transform_utils, "get_idft3", fake_idft)
    monkeypatch.setattr(mean_helpers.fourier_transform_utils, "get_dft3", fake_dft)
    monkeypatch.setattr(
        mean_helpers,
        "_large_relion_solvent_mask_uses_compiled_builder",
        lambda _shape: True,
    )
    monkeypatch.setattr(mean_helpers.gc, "collect", lambda: events.append(("gc",)))
    caplog.set_level("INFO", logger=mean_helpers.__name__)

    result = mean_helpers._apply_relion_solvent_flatten_k1(
        volume_ft,
        solvent_mask,
        (800, 800, 800),
        half_index=1,
    )

    assert result is flattened
    assert events == [
        ("reshape", "volume_ft", (800, 800, 800)),
        ("idft", "volume_ft"),
        ("multiply", "vol_real", "solvent_mask"),
        ("dft", "masked_real"),
        ("reshape", "flattened", -1),
        ("block", "flattened"),
        ("delete", "vol_real"),
        ("delete", "solvent_mask"),
        ("gc",),
    ]
    assert (
        "RELION box-scale solvent flatten lifecycle: half=2 shape=(800, 800, 800) "
        "output_ready=True vol_real_deleted=True solvent_mask_deleted=True "
        "output_dtype=complex128"
    ) in caplog.text


def test_small_solvent_flatten_keeps_async_default_path(monkeypatch):
    events = []

    class FakeBuffer:
        dtype = np.dtype(np.complex64)

        def __init__(self, name):
            self.name = name

        def reshape(self, shape):
            events.append(("reshape", self.name, shape))
            return self

        def __mul__(self, other):
            events.append(("multiply", self.name, other.name))
            return (self, other)

        def block_until_ready(self):
            raise AssertionError("small/default solvent flatten must remain asynchronous")

        def delete(self):
            raise AssertionError("small/default solvent flatten must not delete caller buffers")

    volume_ft = FakeBuffer("volume_ft")
    vol_real = FakeBuffer("vol_real")
    solvent_mask = FakeBuffer("solvent_mask")
    flattened = FakeBuffer("flattened")

    monkeypatch.setattr(
        mean_helpers.fourier_transform_utils,
        "get_idft3",
        lambda value: events.append(("idft", value.name)) or vol_real,
    )
    monkeypatch.setattr(
        mean_helpers.fourier_transform_utils,
        "get_dft3",
        lambda value: events.append(("dft", value)) or flattened,
    )
    monkeypatch.setattr(
        mean_helpers,
        "_large_relion_solvent_mask_uses_compiled_builder",
        lambda _shape: False,
    )

    result = mean_helpers._apply_relion_solvent_flatten_k1(
        volume_ft,
        solvent_mask,
        (8, 8, 8),
        half_index=0,
    )

    assert result is flattened
    assert events == [
        ("reshape", "volume_ft", (8, 8, 8)),
        ("idft", "volume_ft"),
        ("multiply", "vol_real", "solvent_mask"),
        ("dft", (vol_real, solvent_mask)),
        ("reshape", "flattened", -1),
    ]
