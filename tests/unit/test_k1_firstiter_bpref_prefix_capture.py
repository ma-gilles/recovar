from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as sparse


def test_firstiter_fused_bpref_prefix_capture_uses_immutable_identity_and_global_ordinal(
    monkeypatch,
):
    import recovar.cuda_backproject as cuda_backproject

    def prepare(values, _pixel_indices, _image_shape, _max_r):
        return jnp.asarray(values), jnp.arange(4, dtype=jnp.int32), 4, 3

    def accumulate(
        data_volume,
        weight_volume,
        image,
        ctf,
        _minvsigma2,
        _posterior,
        _translation_angles,
        _eulers,
        _threshold,
        _weight_norm,
        _image_shape,
        _volume_shape,
        _max_r,
    ):
        data_increment = jnp.asarray(jnp.real(image[0]), dtype=jnp.float32)
        weight_increment = jnp.asarray(ctf[0], dtype=jnp.float32)
        return (
            data_volume + data_increment.astype(jnp.complex64),
            weight_volume + weight_increment,
        )

    monkeypatch.setattr(
        cuda_backproject,
        "_prepare_relion_x_half_block_topology_operands",
        prepare,
    )
    monkeypatch.setattr(
        cuda_backproject,
        "relion_firstiter_bpref_fused_x_half",
        accumulate,
    )
    monkeypatch.setattr(
        sparse,
        "_bpref_accumulator_delta_config",
        lambda: {
            "directory": None,
            "original_indices": frozenset({20}),
            "iteration": 1,
            "half": 1,
            "max_particles": 1,
            "max_bytes": 1,
        },
    )
    monkeypatch.setitem(sparse._bpref_contribution_context, "iteration", 1)
    monkeypatch.setitem(sparse._bpref_contribution_context, "half", 1)
    captures = []
    monkeypatch.setattr(
        sparse,
        "_write_bpref_accumulator_delta_v1",
        lambda **kwargs: captures.append(kwargs),
    )

    data, weight = sparse._accumulate_relion_firstiter_bpref_fused(
        raw_images=jnp.asarray(
            [[1 + 0j, 0j, 0j, 0j], [3 + 0j, 0j, 0j, 0j]],
            dtype=jnp.complex64,
        ),
        raw_ctf=jnp.asarray(
            [[2, 0, 0, 0], [5, 0, 0, 0]],
            dtype=jnp.float32,
        ),
        raw_minvsigma2=jnp.ones(4, dtype=jnp.float32),
        posterior=jnp.ones((2, 1), dtype=jnp.float32),
        rotations=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (2, 1, 3, 3)),
        actual_counts=np.asarray([1, 1], dtype=np.int64),
        particle_half_local_indices=np.asarray([3, 7], dtype=np.int64),
        particle_original_indices=np.asarray([10, 20], dtype=np.int64),
        data_volume=jnp.zeros(2, dtype=jnp.complex64),
        weight_volume=jnp.zeros(2, dtype=jnp.float32),
        centered_pixel_indices=np.arange(4, dtype=np.int32),
        fftw_pixel_indices=np.arange(4, dtype=np.int32),
        translation_angles=jnp.zeros((1, 2), dtype=jnp.float32),
        physical_image_shape=(4, 4),
        volume_shape=(2,),
        max_r=2.0,
        adaptive_fraction=0.999,
    )

    np.testing.assert_array_equal(np.asarray(data), np.asarray([4, 4], dtype=np.complex64))
    np.testing.assert_array_equal(np.asarray(weight), np.asarray([7, 7], dtype=np.float32))
    assert len(captures) == 1
    capture = captures[0]
    assert capture["original_index"] == 20
    assert capture["particle_launch_ordinal"] == 7
    np.testing.assert_array_equal(capture["before_data"], np.asarray([1, 1], dtype=np.complex64))
    np.testing.assert_array_equal(capture["after_data"], np.asarray([4, 4], dtype=np.complex64))
    np.testing.assert_array_equal(capture["isolated_data"], np.asarray([3, 3], dtype=np.complex64))
    np.testing.assert_array_equal(capture["before_weight"], np.asarray([2, 2], dtype=np.float32))
    np.testing.assert_array_equal(capture["after_weight"], np.asarray([7, 7], dtype=np.float32))
    np.testing.assert_array_equal(capture["isolated_weight"], np.asarray([5, 5], dtype=np.float32))
    operands = capture["operand_bundle"]
    np.testing.assert_array_equal(
        operands["operand_source_image"],
        np.asarray([3, 0, 0, 0], dtype=np.complex64),
    )
    np.testing.assert_array_equal(
        operands["operand_ctf"], np.asarray([5, 0, 0, 0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        operands["operand_posterior"], np.ones(1, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        operands["operand_translation_angles"], np.zeros((1, 2), dtype=np.float32)
    )
