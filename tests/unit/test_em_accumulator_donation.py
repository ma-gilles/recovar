"""Donor accumulator ownership contracts migrated to current EM owners."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from recovar.em.local import local_bucket_stages as local_em_engine_module
from recovar.em.local.local_bucket_stages import _adjoint_slice_volume_maybe_windowed_row_chunks


def test_relion_x_half_per_particle_adjoint_donates_only_its_accumulator():
    from recovar.em.helpers import adjoint as adjoint_mod

    assert adjoint_mod.adjoint_slice_volume_windowed._jit_info.donate_argnums == ()
    assert (
        adjoint_mod.adjoint_slice_volume_windowed_donating._jit_info.donate_argnums
        == (3,)
    )


@pytest.mark.gpu
def test_relion_x_half_per_particle_donating_adjoint_aliases_and_matches(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    from recovar import cuda_backproject
    from recovar.em.helpers import adjoint as adjoint_mod

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY", raising=False)
    cuda_backproject._cuda_ok = None

    image_shape = (8, 8)
    volume_shape = (7, 7, 7)
    volume_size = 7 * 7 * 4
    pixel_indices_host = np.asarray([1], dtype=np.int32)
    rows_host = np.asarray([[1.0 + 2.0j]], dtype=np.complex64)
    rotations_host = np.eye(3, dtype=np.float32)[None]
    pixel_indices = jnp.asarray(pixel_indices_host)
    rows = jnp.asarray(rows_host)
    rotations = jnp.asarray(rotations_host)
    initial = (
        np.arange(volume_size, dtype=np.float32)
        * np.complex64(1.0 + 0.5j)
        / volume_size
    ).astype(np.complex64)
    static_args = (
        image_shape,
        volume_shape,
        "linear_interp",
        True,
        True,
        2.0,
        True,
    )

    with jax.default_device(gpu_device):
        rows = jax.device_put(rows, gpu_device)
        pixel_indices = jax.device_put(pixel_indices, gpu_device)
        rotations = jax.device_put(rotations, gpu_device)
        ordinary_compile_volume = jax.device_put(initial.copy(), gpu_device)
        donating_compile_volume = jax.device_put(initial.copy(), gpu_device)
        ordinary_lowered = adjoint_mod.adjoint_slice_volume_windowed.lower(
            rows,
            pixel_indices,
            rotations,
            ordinary_compile_volume,
            *static_args,
        )
        donating_lowered = adjoint_mod.adjoint_slice_volume_windowed_donating.lower(
            rows,
            pixel_indices,
            rotations,
            donating_compile_volume,
            *static_args,
        )
        ordinary_memory = ordinary_lowered.compile().memory_analysis()
        donating_memory = donating_lowered.compile().memory_analysis()
        ordinary_hlo_header = ordinary_lowered.compiler_ir(
            dialect="hlo"
        ).as_hlo_text().splitlines()[0]
        donating_hlo_header = donating_lowered.compiler_ir(
            dialect="hlo"
        ).as_hlo_text().splitlines()[0]

        ordinary_input = jax.device_put(initial.copy(), gpu_device)
        ordinary_pointer = ordinary_input.unsafe_buffer_pointer()
        ordinary_output = adjoint_mod.adjoint_slice_volume_windowed(
            rows,
            pixel_indices,
            rotations,
            ordinary_input,
            *static_args,
        )
        ordinary_output.block_until_ready()

        donating_input = jax.device_put(initial.copy(), gpu_device)
        donating_pointer = donating_input.unsafe_buffer_pointer()
        donating_output = adjoint_mod.adjoint_slice_volume_windowed_donating(
            rows,
            pixel_indices,
            rotations,
            donating_input,
            *static_args,
        )
        donating_output.block_until_ready()

    assert "input_output_alias" not in ordinary_hlo_header
    assert "input_output_alias={ {}: (3, {}, may-alias) }" in donating_hlo_header
    assert ordinary_memory.alias_size_in_bytes == 0
    assert donating_memory.alias_size_in_bytes == initial.nbytes
    assert not ordinary_input.is_deleted()
    assert ordinary_output.unsafe_buffer_pointer() != ordinary_pointer
    assert donating_input.is_deleted()
    assert donating_output.unsafe_buffer_pointer() == donating_pointer
    assert not rows.is_deleted()
    assert not pixel_indices.is_deleted()
    assert not rotations.is_deleted()
    np.testing.assert_array_equal(np.asarray(rows), rows_host)
    np.testing.assert_array_equal(np.asarray(pixel_indices), pixel_indices_host)
    np.testing.assert_array_equal(np.asarray(rotations), rotations_host)
    np.testing.assert_array_equal(
        np.asarray(donating_output),
        np.asarray(ordinary_output),
    )


def test_relion_x_half_sparse_adjoint_row_chunks_consume_accumulator(monkeypatch):
    rows = jnp.arange(10, dtype=jnp.float32).reshape(5, 2).astype(jnp.complex64)
    rotations = jnp.arange(45, dtype=jnp.float32).reshape(5, 3, 3)
    calls = []

    def fake_donating_adjoint(
        half_block,
        window_indices,
        rotations_block,
        volume,
        image_shape,
        volume_shape,
        disc_type,
        half_image,
        half_volume=False,
        max_r=None,
        relion_x_half=False,
    ):
        del image_shape, volume_shape, disc_type, max_r
        assert half_image is True
        assert half_volume is True
        assert relion_x_half is True
        calls.append(
            (
                np.asarray(half_block).copy(),
                np.asarray(window_indices).copy(),
                np.asarray(rotations_block).copy(),
                float(np.asarray(volume)),
            )
        )
        return volume + jnp.sum(jnp.real(half_block))

    def fail_non_donating_adjoint(*args, **kwargs):
        del args, kwargs
        pytest.fail("RELION x-half row updates must use the donating wrapper")

    monkeypatch.setattr(
        local_em_engine_module,
        "_adjoint_slice_volume_windowed_donating",
        fake_donating_adjoint,
    )
    monkeypatch.setattr(
        local_em_engine_module,
        "_adjoint_slice_volume_maybe_windowed",
        fail_non_donating_adjoint,
    )

    updated, n_chunks = _adjoint_slice_volume_maybe_windowed_row_chunks(
        rows,
        None,
        rotations,
        jnp.asarray(7.0, dtype=jnp.float32),
        (2, 2),
        (4, 4, 4),
        "linear_interp",
        use_window=False,
        max_r=1.0,
        relion_x_half=True,
        target_rows=2,
    )

    assert n_chunks == 3
    assert [call[0].shape[0] for call in calls] == [2, 2, 1]
    np.testing.assert_array_equal(calls[0][1], np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(calls[0][2], np.asarray(rotations[:2]))
    np.testing.assert_array_equal(calls[1][2], np.asarray(rotations[2:4]))
    np.testing.assert_array_equal(calls[2][2], np.asarray(rotations[4:]))
    np.testing.assert_allclose([call[3] for call in calls], [7.0, 13.0, 35.0])
    np.testing.assert_allclose(np.asarray(updated), 52.0)
