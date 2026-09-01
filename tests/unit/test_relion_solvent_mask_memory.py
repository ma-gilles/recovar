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
